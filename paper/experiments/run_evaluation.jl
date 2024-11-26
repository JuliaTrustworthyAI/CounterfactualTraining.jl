using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CTExperiments
using CTExperiments.DataFrames
using CTExperiments: default_ce_evaluation_name
using DotEnv
using Logging
using MPI

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(get_config_from_args())
exper_grid = ExperimentGrid(eval_config.grid_file)

# Meta data:
df_meta = CTExperiments.expand_grid_to_df(exper_grid)

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if rank != 0
    global_logger(NullLogger())             # avoid logging from other processes
    identifier = ExplicitOutputIdentifier("rank_$rank")
    global_output_identifier(identifier)    # set output identifier to avoid issues with serialization
else
    # Generate list of experiments and run them:
    @info "Running evaluation of $(nrow(df_meta)) experiments ..."
end

# Evaluate counterfactuals:
bmk = evaluate_counterfactuals(eval_config, comm)

if eval_config.counterfactual_params.concatenate_output
    # Save results:
    save_results(eval_config, bmk.evaluation, default_ce_evaluation_name(eval_config))
    save_results(eval_config, bmk)
else
    @info "Results for individual runs are stored in $(eval_config.save_dir)."
end

# Working directory:
set_work_dir(eval_config, ENV["EVAL_WORK_DIR"])

