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

# Get config and set up grid:
config_file = get_config_from_args()
root_name = CTExperiments.from_toml(config_file)["name"]
root_save_dir = joinpath(ENV["OUTPUT_DIR"], root_name)
exper_grid = ExperimentGrid(config_file; new_save_dir=root_save_dir)
@assert "threads" ∉ exper_grid.training_params["parallelizer"] "Use multi-processing ('mpi') for counterfactual search if grid is run sequentially."

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    global_logger(NullLogger())
    exper_list = nothing
else
    # Generate list of experiments and run them:
    exper_list = setup_experiments(exper_grid)
    @info "Running $(length(exper_list)) experiments ..."
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

for (i, experiment) in enumerate(exper_list)

    # Setup:
    _name = experiment.meta_params.experiment_name
    if rank != 0
        CTExperiments.shutup!(experiment.training_params)                   # shut off logging for non-root ranks
        @reset experiment.training_params.generator_params.maxiter = 1      # decrease load on non-root ranks
        _save_dir = nothing                                                 # disable saving models for non-root ranks
    else
        _save_dir = experiment.meta_params.save_dir
    end

    # Skip if already finished
    if has_results(experiment)
        @info "Rank $(rank): Skipping $(_name), model already exists."
        continue
    end

    model, logs = run_training(experiment; checkpoint_dir=_save_dir)

    # Saving the results:
    if rank == 0
        save_results(experiment, model, logs)
    end
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
if mpi_should_finalize()
    MPI.Finalize()          # finalize MPI
end
