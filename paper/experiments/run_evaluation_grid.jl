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
set_global_seed()

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if rank != 0
    global_logger(NullLogger())             # avoid logging from other processes
    identifier = ExplicitOutputIdentifier("rank_$rank")
    global_output_identifier(identifier)    # set output identifier to avoid issues with serialization
    eval_list = nothing
else

    # Get config and set up grid:
    grid_file = get_config_from_args(; new_save_dir=ENV["OUTPUT_DIR"])
    eval_grid = EvaluationGrid(grid_file)
    @assert length(eval_grid.counterfactual_params["parallelizer"]) <= 1 "It does not make sense to specify multiple parallelizers. Aborting ..."

    # Set up evaluation configuration:
    eval_list = generate_list(eval_grid) |> li -> li[needs_results.(li)]
    @info "Running $(length(eval_list)) evaluations ..."

    # Adjust parallelizer:
    for (i, _eval_cfg) in enumerate(eval_list)
        @info "Specified parallelizer: $(_eval_cfg.counterfactual_params.parallelizer)" maxlog =
            1
        if _eval_cfg.counterfactual_params.parallelizer == "mpi"
            @warn "Cannot distribute both evaluations and counterfactual search across processes. For multi-processing counterfactual search, use `run_evaluation_grid_sequentially.jl` instead. Resetting to 'threads' ..." maxlog =
                1
            if Threads.nthreads() > 1
                @reset _eval_cfg.counterfactual_params.parallelizer = "threads"
            else
                @reset _eval_cfg.counterfactual_params.parallelizer = ""
            end
        elseif _eval_cfg.counterfactual_params.parallelizer == "" && Threads.nthreads() > 1
            @warn "Found multiple available threads. Resetting to 'parallelizer' from '' to 'threads' ..." maxlog =
                1
            @reset _eval_cfg.counterfactual_params.parallelizer = "threads"
        end
        eval_list[i] = _eval_cfg
    end
end

# Broadcast eval_list from rank 0 to all ranks
eval_list = MPI.bcast(eval_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

if length(eval_list) < nprocs
    @warn "There are less evaluations than processes. Check CPU efficiency of job."
end
chunks = TaijaParallel.split_obs(eval_list, nprocs)     # split  evaluations into chunks for each process

# Set up dummies for processes without tasks to avoid deadlock:
max_chunk_size = maximum(length.(chunks))
chunks = Logging.with_logger(Logging.NullLogger()) do
    for (i, chunk) in enumerate(chunks)
        if length(chunk) < max_chunk_size
            n_missing = max_chunk_size - length(chunk)
            for j in 1:n_missing
                cfg = deepcopy(eval_list[1])
                cfg = make_dummy(cfg, i, j)
                push!(chunk, cfg)
            end
        end
    end
    return chunks
end

@assert allequal(length.(chunks)) "Need all processes to have the same number of tasks."

worker_chunk = MPI.scatter(chunks, comm)                # distribute across processes

for (i, eval_config) in enumerate(worker_chunk)
    if rank != 0
        @reset eval_config.counterfactual_params.verbose = false
    end

    # Evaluate counterfactuals:
    @info "Rank $(rank): Running evaluation $i of $(length(worker_chunk))."
    bmk = evaluate_counterfactuals(eval_config)

    @info "Rank $(rank): Done evaluating all counterfactuals. Waiting at barrier ..."
    MPI.Barrier(comm)

    if eval_config.counterfactual_params.concatenate_output
        # Save results:
        save_results(eval_config, bmk.evaluation, default_ce_evaluation_name(eval_config))
        save_results(eval_config, bmk)
    else
        @info "Rank $(rank): Results for individual runs are stored in $(eval_config.save_dir)."
    end

    # Generate factual target pairs for plotting:
    generate_factual_target_pairs(
        eval_config; nce=CTExperiments.get_global_param("nce_pairs", 10)
    )

    # Working directory:
    if !isdummy(eval_config)
        set_work_dir(eval_grid, eval_config, ENV["EVAL_WORK_DIR"], ENV["OUTPUT_DIR"])
    else
        remove_dummy!(eval_config)
    end
end

# Finalize MPI
MPI.Barrier(comm)       # Ensure all processes reach this point before finishing
if mpi_should_finalize()
    MPI.Finalize()          # finalize MPI
end
