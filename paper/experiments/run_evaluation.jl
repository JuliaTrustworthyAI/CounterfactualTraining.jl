using CTExperiments
using CounterfactualExplanations
using CTExperiments.DataFrames
using DotEnv
using Logging
using MPI

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(joinpath(ENV["EXPERIMENT_DIR"], "run_evaluation_config.toml"))
exper_grid = ExperimentGrid(eval_config.grid_file)

# Meta data:
df_meta = CTExperiments.expand_grid_to_df(exper_grid)

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if rank != 0
    global_logger(NullLogger())
else
    # Generate list of experiments and run them:
    @info "Running evaluation of $(nrow(df_meta)) experiments ..."
end

# Evaluate counterfactuals:
bmk = evaluate_counterfactuals(eval_config, comm)
bmk = innerjoin(df_meta, bmk, on=:id)

# Save results:
save_results(eval_config, bmk, "benchmark")

