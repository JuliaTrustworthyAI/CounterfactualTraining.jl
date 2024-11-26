using CTExperiments
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CTExperiments.DataFrames
using DotEnv
using Logging

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(
    joinpath(ENV["EXPERIMENT_DIR"], "poc_evaluation_config.toml")
)
exper_grid = ExperimentGrid(eval_config.grid_file)

# Meta data:
df_meta = CTExperiments.expand_grid_to_df(exper_grid)

# Evaluate counterfactuals:
bmk = evaluate_counterfactuals(eval_config)

if eval_config.counterfactual_params.concatenate_output
    # Save results:
    save_results(
        eval_config, bmk.evaluation, CTExperiments.default_ce_evaluation_name(eval_config)
    )
    save_results(eval_config, bmk)
else
    @info "Results for individual runs are stored in $(eval_config.save_dir)."
end

# Working directory:
set_work_dir(eval_config, ENV["EVAL_WORK_DIR"])


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