using Accessors
using BSON
using CTExperiments
using CounterfactualExplanations
using DotEnv
using Logging
using MPI
using Serialization
using TaijaParallel

DotEnv.load!()

# Get config and set up grid:
config_file = get_config_from_args()
root_name = CTExperiments.from_toml(config_file)["name"]
root_save_dir = joinpath(ENV["OUTPUT_DIR"], root_name)
exper_grid = ExperimentGrid(config_file; new_save_dir=root_save_dir)

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    # global_logger(NullLogger())
    exper_list = nothing
else
    # Generate list of experiments and run them:
    exper_list = setup_experiments(exper_grid)
    @info "Running $(length(exper_list)) experiments ..."
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

if length(exper_list) < nprocs
    @warn "There are less experiments ($(length(exper_list))) than processes ($(nprocs)). Check CPU efficiency of job."
end
chunks = TaijaParallel.split_obs(exper_list, nprocs)    # split experiments into chunks for each process

# Set up dummy experiment for processes without experiments to avoid deadlock if no experiments are assigned to a process:
chunks = Logging.with_logger(Logging.NullLogger()) do
    for (i, chunk) in enumerate(chunks)
        if isempty(chunk)
            exper = deepcopy(exper_list[1])
            exper.meta_params.experiment_name = "dummy"
            exper.meta_params.save_dir = tempdir()
            @reset exper.training_params.nepochs = 1
            chunks[i] = [exper]
        end
    end
    return chunks
end
worker_chunk = MPI.scatter(chunks, comm)                # distribute across processes

for (i, experiment) in enumerate(worker_chunk)
    if rank != 0
        # Shut up logging for other ranks to avoid cluttering output
        CTExperiments.shutup!(experiment.training_params)
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
    println("Saving checkpoints in: ", _save_dir)
    model, logs = run_training(experiment; checkpoint_dir=_save_dir)

    # Saving the results:
    save_results(experiment, model, logs)
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
