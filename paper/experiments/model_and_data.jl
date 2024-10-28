Base.@kwdef struct MNIST <: Dataset
    n::Int = 10000
    batchsize::Int = 1000
    domain::Union{Nothing,Tuple,Vector{<:Tuple}} = (-1.0f0, 1.0f0)
end

function setup(data::MNIST, model::ModelType)

    # Data:
    Xtrain, y = load_mnist(data.n)
    data = CounterfactualData(Xtrain, y)
    unique_labels = sort(unique(y))
    ytrain = Flux.onehotbatch(y, unique_labels)
    train_set = Flux.DataLoader((Xtrain, ytrain); batchsize=data.batchsize)

    # Input transformers:
    vae = CounterfactualExplanations.Models.load_mnist_vae()
    maxoutdim = vae.params.latent_dim
    input_encoder = fit_transformer(data, PCA; maxoutdim=maxoutdim)

    # Model:
    nin = size(first(train_set)[1], 1)
    nout = size(first(train_set)[2], 1)
    model = build_model(model, nin, nout)

    return model, train_set
end

function set_input_encoder!(
    exp::Experiment,
    data::MNIST,
    generator_type::AbstractGeneratorType
)
    if exp.meta_params.dim_reduction
        # Input transformers:
        vae = CounterfactualExplanations.Models.load_mnist_vae()
        maxoutdim = vae.params.latent_dim
        input_encoder = fit_transformer(data, PCA; maxoutdim=maxoutdim)
    else 
        input_encoder = nothing
    end
    exp.training_params.input_encoder = input_encoder
    return exp
end

Base.@kwdef struct MLPModel <: ModelType
    nhidden::Int = 32
    nlayers::Int = 1
    activation::Function = relu
end

function build_model(model::MLPModel, nin::Int, nout::Int)
    model = Chain(
        Dense(nin, model.nhidden, model.activation),
        fill(Dense(model.nhidden, model.nhidden, model.activation), model.nlayers - 1)...,
        Dense(model.nhidden, nout),
    )
    return model
end