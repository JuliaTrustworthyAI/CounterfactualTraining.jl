using Pkg; Pkg.activate("paper/experiments")

using CounterfactualExplanations
using CounterfactualExplanations: counterfactual
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.Evaluation: plausibility
using CounterfactualExplanations.Generators
using Flux
using TaijaData
using TaijaParallel

# Countefactual generator:
generator = ECCoGenerator(opt=Descent(1.0))
max_iter = 1
conv =  MaxIterConvergence(max_iter)
pllr = ThreadsParallelizer()

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

# Loss function:
function loss(yhat, y, implausibility)
    class_loss = Flux.Losses.logitcrossentropy(yhat, y)
    return class_loss + sum(implausibility)/length(implausibility)
end
function accuracy(model, train_set)
    acc = 0
    for (x,y) in train_set
        yhat = [argmax(_x) for _x in eachcol(softmax(model(x)))] 
        y = Flux.onecold(y) 
        acc += sum(yhat .== y) 
    end
    return acc / (train_set.batchsize * length(train_set))
end

# Training
opt_state = Flux.setup(Adam(), model)
nepochs = 10
my_log = []
for epoch in 1:nepochs
    losses = Float32[]
    M = MLP(model; likelihood=:classification_multi)
    for (i, batch) in enumerate(train_set)
        input, label = batch

        # Generate counterfactuals:
        nbatch = size(input, 2)
        targets = rand(unique_labels, nbatch)               # randomly generate targets
        xs = [x[:, :] for x in eachcol(input)]              # factuals
        println("Generating counterfactuals for batch $i/$(length(train_set))")
        ces = TaijaParallel.parallelize(
            pllr, 
            generate_counterfactual, 
            xs,
            targets,
            data,
            M,
            generator;
            convergence=conv,
            verbose=false,
        )
        input = hcat(counterfactual.(ces)...)               # counterfactual inputs
        implaus = TaijaParallel.parallelize(
            pllr,
            CounterfactualExplanations.Evaluation.evaluate,
            ces;
            measure=plausibility,
            verbose=false,
        ) |> stack |> stack |> vec

        val, grads = Flux.withgradient(model) do m
            # Any code inside here is differentiated.
            # Evaluation of the model and loss must be inside!
            logits = m(input)
            loss(logits, label, implaus)
        end

        # Save the loss from the forward pass. (Done outside of gradient.)
        push!(losses, val)

        # Detect loss of Inf or NaN. Print a warning, and then skip update!
        if !isfinite(val)
            @warn "loss is $val on item $i" epoch
            continue
        end

        Flux.update!(opt_state, model, grads[1])
    end

    # Compute some accuracy, and save details as a NamedTuple
    acc = accuracy(model, train_set)
    @info "Accuracy in epoch $epoch/$nepochs: $acc"
    push!(my_log, (; acc, losses))

    # Stop training when some criterion is reached
    if acc > 0.95
        println("stopping after $epoch epochs")
        break
    end
end

# Check counterfactual:
M = MLP(model; likelihood=:classification_multi)
