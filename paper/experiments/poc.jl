using Pkg; Pkg.activate("paper/experiments")

using CounterfactualExplanations
using CounterfactualExplanations: Convergence, Generators, counterfactual
using CounterfactualExplanations.DataPreprocessing: fit_transformer
using CounterfactualTraining
using CounterfactualTraining: generate!, counterfactual_training
using Flux
using MultivariateStats: PCA
using Serialization
using TaijaData
using TaijaParallel

include("utils.jl")

################### Setup ###################

# Data and model:
Xtrain, y = load_mnist(10000)
data = CounterfactualData(Xtrain, y)
unique_labels = sort(unique(y))
ytrain = Flux.onehotbatch(y, unique_labels)
bs = 1000
train_set = Flux.DataLoader((Xtrain, ytrain), batchsize=bs)
nin = size(first(train_set)[1], 1)
nout = size(first(train_set)[2], 1)
nhidden = 64
activation = relu
model = Chain(
    Dense(nin, nhidden, activation),
    Dense(nhidden, nout)
)

# Input transformers:
vae = CounterfactualExplanations.Models.load_mnist_vae()
maxoutdim = vae.params.latent_dim
pca = fit_transformer(data, PCA; maxoutdim=maxoutdim);

################### Counterfactual Training ###################
burnin = 0.0
nepochs = 200
max_iter = 100
nce = 10
conv = Convergence.MaxIterConvergence(max_iter=max_iter)
pllr = ThreadsParallelizer()
search_opt = Descent(1.0)
verbose = true
domain = (-1.0f0, 1.0f0)    # restrict domain for images to [-1, 1]
λ = [0.001, 5.0]
λ₁ = λ[1]

# With ECCo:
generator = ECCoGenerator(; opt=search_opt, λ=λ)
model_ecco = deepcopy(model)
opt_state = Flux.setup(Adam(), model_ecco)
model_ecco, logs_ecco = counterfactual_training(
    loss,
    model_ecco,
    generator,
    train_set,
    opt_state;
    parallelizer=pllr,
    verbose=verbose,
    convergence=conv,
    nepochs=nepochs,
    burnin=burnin,
    nce=nce,
    domain=domain, 
)

# With Generic:
generator = GenericGenerator(; opt=search_opt, λ=λ₁)
model_generic = deepcopy(model)
opt_state = Flux.setup(Adam(), model_generic)
model_generic, logs_generic = counterfactual_training(
    loss,
    model_generic,
    generator,
    train_set,
    opt_state;
    parallelizer=pllr,
    verbose=verbose,
    convergence=conv,
    nepochs=nepochs,
    burnin=burnin,
    nce=nce,
    domain=domain,
)

# With REVISE
generator = REVISEGenerator(; opt=search_opt, λ=λ₁)
model_revise = deepcopy(model)
opt_state = Flux.setup(Adam(), model_revise)
model_revise, logs_revise = counterfactual_training(
    loss,
    model_revise,
    generator,
    train_set,
    opt_state;
    parallelizer=pllr,
    input_encoder=vae,
    verbose=verbose,
    convergence=conv,
    nepochs=nepochs,
    burnin=burnin,
    nce=nce,
    domain=domain,
)

################### Results ###################
λ = [0.001, 20.0]
gen = ECCoGenerator(; opt=search_opt, λ=λ)
test_data = CounterfactualData(load_mnist_test()...)
# test_data.input_encoder = pca

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