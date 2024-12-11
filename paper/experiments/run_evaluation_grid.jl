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

# Get config and set up grid:
eval_grid = EvaluationGrid(get_config_from_args())
exper_grid = ExperimentGrid(eval_grid.grid_file)

# Meta data:
df_meta = CTExperiments.expand_grid_to_df(exper_grid)

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if rank != 0
    # global_logger(NullLogger())             # avoid logging from other processes
    identifier = ExplicitOutputIdentifier("rank_$rank")
    global_output_identifier(identifier)    # set output identifier to avoid issues with serialization
    eval_list = nothing
else
    # Set up evaluation configuration:
    eval_list = setup_evaluations(eval_grid)
    @info "Running $(length(eval_list)) evaluations ..."
end

# Broadcast eval_list from rank 0 to all ranks
eval_list = MPI.bcast(eval_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

if length(eval_list) < nprocs
    @warn "There are less evaluations than processes. Check CPU efficiency of job."
end
chunks = TaijaParallel.split_obs(eval_list, nprocs)    # split  evaluations into chunks for each process

# Set up dummy evaluation for processes without evaluations to avoid deadlock if no  evaluations are assigned to a process:
chunks = Logging.with_logger(Logging.NullLogger()) do
    for (i, chunk) in enumerate(chunks)
        if isempty(chunk)
            cfg = eval_list[1]
            @reset cfg.save_dir = mkpath(joinpath(tempdir(), "dummy_eval_$(i)"))
            chunks[i] = [cfg]
        end
    end
    return chunks
end

display(chunks)

worker_chunk = MPI.scatter(chunks, comm)                # distribute across processes

for (i, eval_config) in enumerate(worker_chunk)

    # Evaluate counterfactuals:
    @info "Running evaluation $i of $(length(worker_chunk))."
    bmk = evaluate_counterfactuals(eval_config, comm)

    if eval_config.counterfactual_params.concatenate_output
        # Save results:
        save_results(eval_config, bmk.evaluation, default_ce_evaluation_name(eval_config))
        save_results(eval_config, bmk)
    else
        @info "Results for individual runs are stored in $(eval_config.save_dir)."
    end

    # Generate factual target pairs for plotting:
    # generate_factual_target_pairs(eval_config)

    # Working directory:
    set_work_dir(eval_grid, eval_config, joinpath(ENV["EVAL_WORK_DIR"]))
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
