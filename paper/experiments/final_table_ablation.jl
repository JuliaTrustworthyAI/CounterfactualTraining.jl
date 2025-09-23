
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

res_dir = joinpath(ENV["FINAL_GRID_RESULTS"],"ablation")

df = final_table(res_dir;)
Serialization.serialize("paper/experiments/output/final_table_ablation.jls", df)
