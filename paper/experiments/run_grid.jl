using Accessors
using BSON
using CTExperiments
using CounterfactualExplanations
using DotEnv
using Logging
using MPI
using Serialization
using TaijaParallel

# Setup:
DotEnv.load!()
set_global_seed()

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    global_logger(NullLogger())
    exper_list = nothing
else

    # Get config and set up grid:
    config_file = get_config_from_args()
    exper_grid = ExperimentGrid(config_file; new_save_dir=ENV["OUTPUT_DIR"])

    # Generate list of experiments and run them:
    exper_list = setup_experiments(exper_grid)
    @info "Running $(length(exper_list)) experiments ..."

    # Adjust parallelizer:
    for (i, cfg) in enumerate(exper_list)
        if cfg.training_params.parallelizer == "mpi"
            @warn "Cannot distribute both experiments and counterfactual search across processes. For multi-processing counterfactual search, use `run_grid_sequentially.jl` instead. Resetting ..." maxlog =
                1
            if Threads.nthreads() > 1
                @reset cfg.training_params.parallelizer = "threads"
            else
                @reset cfg.training_params.parallelizer = ""
            end
        elseif cfg.training_params.parallelizer == "" && Threads.nthreads() > 1
            @warn "Found multiple available threads. Resetting to 'parallelizer' from '' to 'threads' ..." maxlog =
                1
            @reset cfg.training_params.parallelizer = "threads"
        elseif cfg.training_params.parallelizer == "threads" && Threads.nthreads() <= 1
            @warn "Found only one available thread. Resetting to '' from 'threads' ..." maxlog =
                1
            @reset cfg.training_params.parallelizer = ""
        end
        exper_list[i] = cfg
    end
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

if length(exper_list) < nprocs
    @warn "There are less experiments ($(length(exper_list))) than processes ($(nprocs)). Check CPU efficiency of job."
end
chunks = TaijaParallel.split_obs(exper_list, nprocs)    # split experiments into chunks for each process

# Set up dummies for processes without tasks to avoid deadlock:
max_chunk_size = maximum(length.(chunks))
chunks = Logging.with_logger(Logging.NullLogger()) do
    for (i, chunk) in enumerate(chunks)
        if length(chunk) < max_chunk_size
            n_missing = max_chunk_size - length(chunk)
            for j in 1:n_missing
                exper = deepcopy(exper_list[1])
                make_dummy(exper, i, j)
                push!(chunk, exper)
            end
        end
    end
    return chunks
end

@assert allequal(length.(chunks)) "Need all processes to have the same number of tasks."

worker_chunk = MPI.scatter(chunks, comm)                # distribute across processes

for (i, experiment) in enumerate(worker_chunk)
    if rank != 0
        # Shut up logging for other ranks to avoid cluttering output
        @reset experiment.training_params.verbose = 0                       # shut off logging for non-root ranks
    end

    # Setup:
    _save_dir = experiment.meta_params.save_dir
    _name = experiment.meta_params.experiment_name

    # Skip if already finished
    if has_results(experiment)
        @info "Rank $(rank): Skipping $(_name), model already exists."
        continue
    end

    # Running the experiment
    @info "Rank $(rank): Running experiment: $(_name) ($i/$(length(worker_chunk)))"

    model, logs = run_training(experiment; checkpoint_dir=_save_dir)

    # Saving the results:
    if !isdummy(experiment)  # Avoid overwriting dummy results
        save_results(experiment, model, logs)
    else
        remove_dummy!(experiment)
    end
end

if rank == 0
    @info "All experiments for $(config_file) completed"
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
if mpi_should_finalize()
    MPI.Finalize()          # finalize MPI
end
