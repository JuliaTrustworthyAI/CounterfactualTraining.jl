using Pkg;
Pkg.activate("paper/experiments");

using CounterfactualExplanations
using CounterfactualExplanations: counterfactual
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.Evaluation: plausibility, validity
using CounterfactualExplanations.Generators
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
train_set = Flux.DataLoader((Xtrain, ytrain); batchsize=bs)
nin = size(first(train_set)[1], 1)
nout = size(first(train_set)[2], 1)
nhidden = 32
model = Chain(Dense(nin, nhidden, relu), Dense(nhidden, nout))
