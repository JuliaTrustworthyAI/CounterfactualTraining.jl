using TaijaData

get_domain(d::Dataset) = nothing

"""
    MNIST

Keyword container for the `MNIST` data set. Can specify the number of samples `n`, the batch size `batchsize` and the feature `domain`.
"""
Base.@kwdef struct MNIST <: Dataset
    n::Int = 10000
    batchsize::Int = 1000
end

"""
    data_sets

Catalogue of available model types.
"""
const data_sets = Dict("mnist" => MNIST)

"""
    get_data(s::String)

Retrieves the data set from the catalogue if available.
"""
function get_data(s::String)
    s = lowercase(s)
    @assert s in keys(data_sets) "Unknown data set: $s. Available sets are $(keys(data_sets))"
    return data_sets[s]
end

"""
    MLPModel

`MLPModel` type. The following parameters can be specified:

- `nhidden`: the number of hidden units in each layer (default is 32).
- `nlayers`: the number of hidden layers in the model (default is 1).
- `activation`: the activation function to use (default is `relu`).
"""
Base.@kwdef struct MLPModel <: ModelType
    nhidden::Int = 32
    nlayers::Int = 1
    activation::Function = relu
    function MLPModel(nhidden, nlayers, activation)
        if isa(activation, String)
            activation = eval(Meta.parse(activation))
        end
        return new(nhidden, nlayers, activation)
    end
end

"""
    model_types

Catalogue of available model types.
"""
const model_types = Dict("mlp" => MLPModel)

"""
    get_model_type(s::String)

Retrieves the model type from the catalogue if available.
"""
function get_model_type(s::String)
    s = lowercase(s)
    @assert s in keys(model_types) "Unknown model type: $s. Available types are $(keys(model_types))"
    return model_types[s]
end

get_domain(d::MNIST) = (-1.0f0, 1.0f0)

"""
    setup(exp::AbstractExperiment, data::MNIST, model::ModelType)

Loads the MNIST data and builds a model corresponding to the specified `model` type. Returns the model and training dataset.
"""
function setup(exp::AbstractExperiment, data::MNIST, model::ModelType)

    # Data:
    Xtrain, y = load_mnist(data.n)
    counterfactual_data = CounterfactualData(Xtrain, y)
    unique_labels = sort(unique(y))
    ytrain = Flux.onehotbatch(y, unique_labels)
    train_set = Flux.DataLoader((Xtrain, ytrain); batchsize=data.batchsize)

    # Model:
    nin = size(first(train_set)[1], 1)
    nout = size(first(train_set)[2], 1)
    model = build_model(model, nin, nout)

    # Input encoding:
    input_encoder = get_input_encoder(exp)

    return model, train_set, input_encoder
end

"""
    get_input_encoder(
        exp::AbstractExperiment,
        data::MNIST,
        generator_type::AbstractGeneratorType
    )

For MNIST data, use PCA for dimensionality reduction if requested and set the `maxoutdim` to the size of the latent dimension of the VAE.
"""
function get_input_encoder(
    exp::AbstractExperiment, data::MNIST, generator_type::AbstractGeneratorType
)
    if exp.meta_params.dim_reduction
        # Input transformers:
        vae = CounterfactualExplanations.Models.load_mnist_vae()
        maxoutdim = vae.params.latent_dim
        input_encoder = fit_transformer(data, PCA; maxoutdim=maxoutdim)
    else
        input_encoder = nothing
    end
    return input_encoder
end

"""
    get_input_encoder(
        exp::AbstractExperiment, 
        data::MNIST, 
        generator_type::REVISE
    )

For MNIST data and the REVISE generator, use the VAE as the input encoder.
"""
function get_input_encoder(exp::AbstractExperiment, data::MNIST, generator_type::REVISE)
    vae = CounterfactualExplanations.Models.load_mnist_vae()
    return vae
end

"""
    build_model(model::MLPModel, nin::Int, nout::Int)

Builds a multi-layer perceptron model.
"""
function build_model(model::MLPModel, nin::Int, nout::Int)
    model = Chain(
        Dense(nin, model.nhidden, model.activation),
        fill(Dense(model.nhidden, model.nhidden, model.activation), model.nlayers - 1)...,
        Dense(model.nhidden, nout),
    )
    return model
end
