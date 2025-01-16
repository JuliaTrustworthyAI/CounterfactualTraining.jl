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
    config_file = get_config_from_args(; new_save_dir=ENV["OUTPUT_DIR"])
    exper_grid = ExperimentGrid(config_file; new_save_dir=ENV["OUTPUT_DIR"])

    # Generate list of experiments and run them:
    exper_list = generate_list(exper_grid) |> li -> li[needs_results.(li)]
    length(exper_list) > 0 || error("No experiments to run.")
    @info "Running $(length(exper_list)) experiments ..."

    # Adjust parallelizer to MPI if grid is run sequentially:
    for (i, cfg) in enumerate(exper_list)
        if cfg.training_params.parallelizer in ["threads", ""]
            @warn "It makes sense to use multi-processing ('mpi') for counterfactual search if grid is run sequentially. For multi-threading, use `run_grid.jl` instead. Resetting ..." maxlog =
                1
            @reset cfg.training_params.parallelizer = "mpi"
            exper_list[i] = cfg
        end
    end
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

for (i, experiment) in enumerate(exper_list)

    # Setup:
    _name = experiment.meta_params.experiment_name
    if rank != 0
        @reset experiment.training_params.verbose = 0                       # shut off logging for non-root ranks
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

if rank == 0
    @info "All experiments for $(config_file) completed"
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
if mpi_should_finalize()
    MPI.Finalize()          # finalize MPI
end
