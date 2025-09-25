@info "Generating table for ablation results ..."
using Pkg;
Pkg.status();

using CTExperiments
using CTExperiments.CairoMakie
using CTExperiments.CSV
using CTExperiments.DataFrames
using CTExperiments.Plots
using CTExperiments.PrettyTables
using CTExperiments.StatsBase
using Random
using Serialization

using DotEnv
DotEnv.load!()

res_dir = joinpath(ENV["FINAL_GRID_RESULTS"], "ablation")

# df = aggregate_ce_evaluation(
#     res_dir;
#     ratio=false,
#     verbose=true,
#     y="plausibility_distance_from_target", 
#     agg_further_vars=["run", "lambda_energy_eval"], 
#     total_uncertainty=false,    
#     drop_models=String[],
#     return_sig_level=true,
# )
#
#
# Serialization.serialize("paper/experiments/output/final_table_ablation.jls", df)
#
# df = aggregate_ce_evaluation(
#     res_dir;
#     ratio=false,
#     verbose=true,
#     y="mmd", 
#     agg_further_vars=["run", "lambda_energy_eval"], 
#     total_uncertainty=false,    
#     drop_models=String[],
#     return_sig_level=true,
# )
#
# Serialization.serialize("paper/experiments/output/final_table_ablation_mmd.jls", df)

df = aggregate_performance(
    res_dir;
    adversarial=true,
    bootstrap=100,
    measure=["acc"],
    drop_models=String[],
    eps=range(0.0, 0.1; length=10) |> collect,
)

Serialization.serialize("paper/experiments/output/final_table_ablation_ar.jls", df)
