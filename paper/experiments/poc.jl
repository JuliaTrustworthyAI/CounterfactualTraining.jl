using Pkg; Pkg.activate("paper/experiments")

using CounterfactualExplanations
using CounterfactualExplanations: Convergence, Generators, counterfactual
using CounterfactualTraining
using CounterfactualTraining: generate!, counterfactual_training
using Flux
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
nhidden = 32
model = Chain(
    Dense(nin, nhidden, relu),
    Dense(nhidden, nout)
)

################### Counterfactual Training ###################
burnin = 0.9
nepochs = 100
max_iter = 100
nce = 10
conv = Convergence.MaxIterConvergence(max_iter)
pllr = ThreadsParallelizer()
search_opt = Descent(0.05)
verbose = true

# With ECCo:
generator = ECCoGenerator(; opt=search_opt, λ=[0.01, 0.1])
model_ecco = deepcopy(model)
opt_state = Flux.setup(Adam(), model_ecco)
model_ecco, logs = counterfactual_training(
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
)

# With Generic:
generator = GenericGenerator(; opt=search_opt, λ=0.01)
model_generic = deepcopy(model)
opt_state = Flux.setup(Adam(), model_generic)
model_generic, logs = counterfactual_training(
    loss,
    model_generic,
    generator,
    train_set,
    opt_state;
    parallelizer=pllr,
    verbose=verbose,
    convergence=Convergence.DecisionThresholdConvergence(),
    nepochs=nepochs,
    burnin=burnin,
    nce=nce,
)

# With REVISE
generator = REVISEGenerator(; opt=search_opt, λ=0.01)
model_revise = deepcopy(model)
opt_state = Flux.setup(Adam(), model_revise)
model_revise, logs = counterfactual_training(
    loss,
    model_revise,
    generator,
    train_set,
    opt_state;
    parallelizer=pllr,
    transformer=CounterfactualExplanations.Models.load_mnist_vae(),
    verbose=verbose,
    convergence=conv,
    nepochs=nepochs,
    burnin=burnin,
    nce=nce,
)

################### Results ###################
λ = [0.0, 5.0]
gen = ECCoGenerator(; opt=Descent(1.0), λ=λ)

M = MLP(model_ecco; likelihood=:classification_multi)
serialize("paper/experiments/output/poc_model_ct_ecco.jls", M)
plt = plot_all_mnist(gen, M; convergence=Convergence.MaxIterConvergence(100))
savefig(plt, "paper/dump/poc_model_ct_ecco.png")

M = MLP(model_generic; likelihood=:classification_multi)
serialize("paper/experiments/output/poc_model_ct_generic.jls", M)
plt = plot_all_mnist(gen, M; convergence=Convergence.MaxIterConvergence(100))
savefig(plt, "paper/dump/poc_model_ct_generic.png")

M = MLP(model_revise; likelihood=:classification_multi)
serialize("paper/experiments/output/poc_model_ct_revise.jls", M)
plt = plot_all_mnist(gen, M; convergence=Convergence.MaxIterConvergence(100))
savefig(plt, "paper/dump/poc_model_ct_model_revise.png")

M = load_mnist_mlp()
plt = plot_all_mnist(gen, M; convergence=Convergence.MaxIterConvergence(100))
savefig(plt, "paper/dump/poc_model.png")