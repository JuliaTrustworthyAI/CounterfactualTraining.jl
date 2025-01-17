using Accessors
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CTExperiments
using CTExperiments.DataFrames
using CTExperiments: default_ce_evaluation_name
using DotEnv
using Logging
using MPI
using TaijaParallel

# Setup:
DotEnv.load!()

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
set_global_seed(rank)                       # rank-specific seed
nprocs = MPI.Comm_size(comm)
if rank != 0
    global_logger(NullLogger())             # avoid logging from other processes
    identifier = ExplicitOutputIdentifier("rank_$rank")
    global_output_identifier(identifier)    # set output identifier to avoid issues with serialization
    eval_list = nothing
else

    # Get config and set up grid:
    grid_file = get_config_from_args(;new_save_dir=ENV["OUTPUT_DIR"])
    eval_grid = EvaluationGrid(grid_file)
    @assert length(eval_grid.counterfactual_params["parallelizer"]) <= 1 "It does not make sense to specify multiple parallelizers. Aborting ..."

    # Set up evaluation configuration:
    eval_list = generate_list(eval_grid) |> li -> li[needs_results.(li)]
    length(eval_list) > 0 || error("No evaluations to run.")
    @info "Running $(length(eval_list)) evaluations ..."

    # Adjust parallelizer:
    for (i, _eval_cfg) in enumerate(eval_list)
        if _eval_cfg.counterfactual_params.parallelizer in ["threads", ""]
            @warn "It makes sense to use multi-processing ('mpi') for counterfactual search if grid is run sequentially. For multi-threading, use `run_evaluation_grid.jl` instead. Resetting to 'mpi' ..." maxlog =
                1
            @reset _eval_cfg.counterfactual_params.parallelizer = "mpi"
        end
        eval_list[i] = _eval_cfg
    end
end

# Broadcast eval_list from rank 0 to all ranks
eval_list = MPI.bcast(eval_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

for (i, eval_config) in enumerate(eval_list)

    # Setup:
    @reset eval_config.save_dir = mkpath(joinpath(eval_config.save_dir, "rank_$rank"))
    if rank != 0
        @reset eval_config.counterfactual_params.verbose = false
    end

    # Evaluate counterfactuals:
    if isfile(CTExperiments.default_bmk_name(eval_config))
        @info "Rank $(rank): Evaluation already exists. Skipping evaluation."
        continue
    else
        @info "Rank $(rank): Running evaluation $i of $(length(eval_list))."
        bmk = evaluate_counterfactuals(eval_config)
        bmk.evaluation.rank .= rank
    end

    @info "Rank $(rank): Done evaluating all counterfactuals. Waiting at barrier ..."
    MPI.Barrier(comm)

    if rank == 0
        if eval_config.counterfactual_params.concatenate_output
            # Save results:
            save_results(
                eval_config, bmk.evaluation, default_ce_evaluation_name(eval_config)
            )
            save_results(eval_config, bmk)
        else
            @info "Rank $(rank): Results for individual runs are stored in $(eval_config.save_dir)."
        end

        # Generate factual target pairs for plotting:
        generate_factual_target_pairs(eval_config)

        # Set up evaluation work dir:
        set_work_dir(eval_grid, eval_config, ENV["EVAL_WORK_DIR"], ENV["OUTPUT_DIR"])
    # else
    #     rm(eval_config.save_dir; recursive=true)
    end
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
if mpi_should_finalize()
    MPI.Finalize()          # finalize MPI
end
