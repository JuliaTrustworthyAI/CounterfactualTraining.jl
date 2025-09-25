using CTExperiments.AlgebraOfGraphics
using CTExperiments.CairoMakie
using CTExperiments.DataFrames
using Serialization

df = Serialization.deserialize("paper/experiments/output/final_table_ablation.jls")
df_mmd = Serialization.deserialize("paper/experiments/output/final_table_ablation_mmd.jls")
df = vcat(df, df_mmd)
result =
    transform(
        groupby(df, [:dataset, :variable]),
        [:objective, :mean] =>
            ((obj, means) -> let vanilla_mean = means[findfirst(==("Vanilla"), obj)]
                @. (means - vanilla_mean) / vanilla_mean * -100
            end) => :pct_change,
    ) |> x -> filter(row -> row.objective != "Vanilla", x)
result = transform(result, :variable => (x -> [L"%$s" for s in x]) => :variable)
result = transform(
    result,
    :objective =>
        (x -> replace(x, "Adversarial" => "AR", "Energy" => "CD", "Full" => "CT")) =>
            :objective,
)
plt =
    data(result) *
    mapping(
        :objective, :pct_change; color=:dataset => "Data", dodge=:dataset, row=:variable
    ) *
    visual(BarPlot) |> draw(
        scales(; Color=(; palette=:tab10));
        figure=(; size=(900, 300)),
        axis=(; xlabel="Objective", ylabel="Percentage Reduction (%)"),
    )

save("paper/experiments/output/ablation.png", plt; px_per_unit=3)
