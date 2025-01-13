"""
    NeuroTree

`NeuroTree` type. The following parameters can be specified:

- `nhidden`: the number of hidden units in each layer (default is 32).
- `nlayers`: the number of hidden layers in the model (default is 1).
- `activation`: the activation function to use (default is `relu`).
"""
Base.@kwdef struct NeuroTree <: ModelType
    nhidden::Int = 32
    nlayers::Int = 1
    activation::Function = relu
    function NeuroTree(nhidden, nlayers, activation)
        if isa(activation, String)
            activation = eval(Meta.parse(activation))
        end
        return new(nhidden, nlayers, activation)
    end
end

"""
    build_model(model::NeuroTree, nin::Int, nout::Int)

Builds a multi-layer perceptron model.
"""
function build_model(model::NeuroTree, nin::Int, nout::Int)
    model = Chain(
        Dense(nin, model.nhidden, model.activation),
        fill(Dense(model.nhidden, model.nhidden, model.activation), model.nlayers - 1)...,
        Dense(model.nhidden, nout),
    )
    return model
end
