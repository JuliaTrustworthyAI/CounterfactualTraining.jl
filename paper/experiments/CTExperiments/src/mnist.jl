using TaijaData
using MultivariateStats

"""
    MNIST

Keyword container for the `MNIST` data set. Can specify the number of samples `n`, the batch size `batchsize` and the feature `domain`.
"""
Base.@kwdef struct MNIST <: Dataset
    n::Int = 10000
    batchsize::Int = 1000
    train_val_test_split::Union{Nothing, Vector{<:AbstractFloat}} = nothing
end

get_domain(d::MNIST) = (-1.0f0, 1.0f0)

"""
    setup(exp::AbstractExperiment, data::MNIST, model::ModelType)

Loads the MNIST data and builds a model corresponding to the specified `model` type. Returns the model and training dataset.
"""
function setup(exp::AbstractExperiment, data::MNIST, model::ModelType)

    # Data:
    Xtrain, y = load_mnist(data.n)
    unique_labels = sort(unique(y))
    ytrain = Flux.onehotbatch(y, unique_labels)
    train_set = Flux.DataLoader((Xtrain, ytrain); batchsize=data.batchsize, parallel=true)

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
        Xtrain, y = load_mnist(data.n)
        counterfactual_data = CounterfactualData(Xtrain, y)
        input_encoder = fit_transformer(counterfactual_data, PCA; maxoutdim=maxoutdim)
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