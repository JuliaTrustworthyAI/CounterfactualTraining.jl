using CTExperiments
using CTExperiments.CounterfactualExplanations
using Images
using Random

## Integrated gradients:
ig_save_dir = "paper/figures/mnist_ig/"
ct_preffix = "mnist1"
bl_preffix = "mnist2"

all_digits = 0:9

imgs_ct = [
    Images.load(joinpath(ig_save_dir, "$(ct_preffix)_$(i).png")) |>
    x -> imresize(x, (28, 28)) for i in all_digits
]
imgs_bl = [
    Images.load(joinpath(ig_save_dir, "$(bl_preffix)_$(i).png")) |>
    x -> imresize(x, (28, 28)) for i in all_digits
]

img_ig_full = mosaicview(imgs_bl..., imgs_ct...; nrow=2, rowmajor=true)
Images.save("paper/figures/mnist_ig.png", img_ig_full)

chosen_digits = 5:9

img_ig = mosaicview(
    imgs_bl[chosen_digits .+ 1]..., imgs_ct[chosen_digits .+ 1]...; nrow=2, rowmajor=true
)

## Counterfactuals:
Random.seed!(42)    # change seed for different outcome
overwrite = false   # set to `true` if you want to overwrite the file
res_dir = "paper/experiments/output/final_run/mutability/"
fname = joinpath(res_dir, "mlp/mnist/grid_config.toml")
expers = ExperimentGrid(fname) |> load_list                 # load the experiments
models = (x -> load_results(x)[3]).(expers)                 # get models
data = get_ce_data(expers[1].data; test_set=true)           # get counterfactual data (test set)
gen =
    CTExperiments.GeneratorParams(; lambda_energy=1.0) |>  # get generator
    CTExperiments.get_generator
chosen_digits = 0:4
factual = chosen_digits[1]
both_predict_factual = false
while !both_predict_factual
    chosen = rand(findall(data.output_encoder.labels .== factual))
    global x = select_factual(data, chosen)
    both_predict_factual =
        predict_label(models[1], data, x)[1] ==
        predict_label(models[2], data, x)[1] ==
        factual
end

# Generate counterfactuals
imgs = []
conv = CounterfactualExplanations.Convergence.MaxIterConvergence(1000)
for M in models
    imgs_M = [CTExperiments.convert2mnist(x; blue=true)]
    for target in all_digits[2:end]
        img =
            generate_counterfactual(x, target, data, M, gen; convergence=conv) |>
            CounterfactualExplanations.counterfactual |>
            CTExperiments.convert2mnist
        push!(imgs_M, img)
    end
    push!(imgs, imgs_M)
end

imgs_ct = imgs[1]
imgs_bl = imgs[2]
img_ce = mosaicview(imgs_bl..., imgs_ct...; nrow=2, rowmajor=true)
Images.save("paper/figures/mnist_ce.png", img_ce)

# Combined image:
idx = [findall(all_digits .== x)[1] for x in chosen_digits]
img_ce = mosaicview(imgs_bl[idx]..., imgs_ct[idx]...; nrow=2, rowmajor=true)
img = mosaicview(img_ce, img_ig; nrow=1)
Images.save("paper/figures/mnist_body.png", img)
