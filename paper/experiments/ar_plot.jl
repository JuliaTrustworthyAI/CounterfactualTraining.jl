using CTExperiments
using CTExperiments.AlgebraOfGraphics
using CTExperiments.CairoMakie

using DotEnv
DotEnv.load!()

res_dir = joinpath(ENV["FINAL_GRID_RESULTS"], "ablation")

plt = CTExperiments.plot_performance(
    res_dir;
    eps=range(0.0, 0.1; length=10) |> collect,
    adversarial=true,
    byvars=["objective", "eps"],
    drop_synthetic=true,
    attack_fun=[CTExperiments.pgd, CTExperiments.fgsm],
)
plt_out = draw(
    plt,
    scales(; Color=(; palette=:tab10));
    figure=(size=(900, 350),),
    axis=(
        yticks=[0.0, 0.5, 1.0],
        limits=(nothing, (0, 1)),
        xticklabelsvisible=false,
        xticksvisible=false,
    ),
)
save("paper/figures/acc_full_both.png", plt_out; px_per_unit=3)
