"""
    LeNetModel

`LeNetModel` type. The following parameters can be specified:

- `filter_size`: Size of the convolutional filter.
- `channels1`: Number of channels in the first convolutional layer.
- `channels2`: Number of channels in the second convolutional layer.
- `activation`: Activation function to use. Default is `relu`.
"""
Base.@kwdef struct LeNetModel <: ModelType
    filter_size::Int = 5
    channels1::Int = 6
    channels2::Int = 16
    activation::Function = relu
    function LeNetModel(filter_size, channels1, channels2, activation)
        if isa(activation, String)
            activation = eval(Meta.parse(activation))
        end
        return new(filter_size, channels1, channels2, activation)
    end
end

"""
    build_model(model::LeNetModel, nin::Int, nout::Int)

Builds a multi-layer perceptron model.
"""
function build_model(model::LeNetModel, nin::Int, nout::Int)

    # Setup:
    _n_in = Int(sqrt(nin))
    k, c1, c2 = model.filter_size, model.channels1, model.channels2
    mod(k, 2) == 1 || error("`filter_size` must be odd. ")
    p = div(k - 1, 2) # padding to preserve image size on convolution:

    # Model:
    front = Chain(
        Conv((k, k), 1 => c1, model.activation; pad=(p, p)),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, model.activation; pad=(p, p)),
        MaxPool((2, 2)),
        Flux.flatten,
    )
    d = first(Flux.outputsize(front, (_n_in, _n_in, 1, 1)))
    back = Chain(
        Dense(d, 120, model.activation), Dense(120, 84, model.activation), Dense(84, nout)
    )
    model = Chain(ToConv(_n_in), front, back)

    return model
end

"A simple functor to convert a vector to a convolutional layer."
struct ToConv
    n_in::Int
end

"""
    (f::ToConv)(x)

Method to convert a vector to a convolutional layer.
"""
function (f::ToConv)(x)
    return reshape(x, (f.n_in, f.n_in, 1, :))
end
