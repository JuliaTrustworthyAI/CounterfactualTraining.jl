using CTExperiments
using CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Run grid:
ENV["config"] = joinpath(ENV["EXPERIMENT_DIR"], "poc.toml")
include("run_grid.jl")

# Run evaluation:
ENV["config"] = joinpath(ENV["EXPERIMENT_DIR"], "poc_evaluation_config.toml")
include("run_evaluation.jl")



################### Results ###################
λ = [0.01, 25.0]
gen = ECCoGenerator(; opt=search_opt, λ=λ)
test_data = CounterfactualData(load_mnist_test()...)

conv = Convergence.MaxIterConvergence(; max_iter=100)

M = MLP(model_ecco; likelihood=:classification_multi)
serialize("paper/experiments/output/poc_model_ct_ecco.jls", M)
plt = plot_all_mnist(gen, M, test_data; convergence=conv)
savefig(plt, "paper/dump/poc_model_ct_ecco.png")

M = MLP(model_generic; likelihood=:classification_multi)
serialize("paper/experiments/output/poc_model_ct_generic.jls", M)
plt = plot_all_mnist(gen, M, test_data; convergence=conv)
savefig(plt, "paper/dump/poc_model_ct_generic.png")

M = MLP(model_revise; likelihood=:classification_multi)
serialize("paper/experiments/output/poc_model_ct_revise.jls", M)
plt = plot_all_mnist(gen, M, test_data; convergence=conv)
savefig(plt, "paper/dump/poc_model_ct_model_revise.png")

M = load_mnist_mlp()
plt = plot_all_mnist(gen, M, test_data; convergence=conv)
savefig(plt, "paper/dump/poc_model.png")