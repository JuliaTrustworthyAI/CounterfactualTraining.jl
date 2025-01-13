"""
    LinearModel

`LinearModel` type has no parameters.
"""
struct LinearModel <: ModelType end

"""
    build_model(model::LinearModel, nin::Int, nout::Int)

Builds a linear model.
"""
function build_model(model::LinearModel, nin::Int, nout::Int)
    model = Chain(Dense(nin, nout))
    return model
end
