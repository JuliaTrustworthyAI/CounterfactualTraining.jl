using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: train_test_split, unpack_data
using Flux
using Flux.MLUtils
using MultivariateStats
using StatsBase
using TaijaData

"""
    MNIST

Keyword container for the `MNIST` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef struct MNIST <: Dataset
    n_train::Int = 10000
    batchsize::Int = 1000
    n_validation::Int = 1000
end

get_domain(d::MNIST) = (-1.0f0, 1.0f0)

"""
    setup(exp::AbstractExperiment, data::MNIST, model::ModelType)

Loads the MNIST data and builds a model corresponding to the specified `model` type. Returns the model and training dataset.
"""
function setup(exp::AbstractExperiment, data::MNIST, model::ModelType)

    # Data:
    n_total = data.n_train + data.n_validation
    ce_data = CounterfactualData(load_mnist(n_total)...)
    test_size = data.n_validation / n_total
    Xtrain, ytrain, Xval, yval, unique_labels = (
        dt -> (dt[1].X, dt[1].y, dt[2].X, dt[2].y, dt[1].y_levels)
    )(
        train_test_split(ce_data; test_size=test_size, keep_class_ratio=false)
    )
    train_set = Flux.DataLoader((Xtrain, ytrain); batchsize=data.batchsize, parallel=true)
    val_set = if data.n_validation > 0
        Flux.DataLoader((Xval, yval); batchsize=data.batchsize, parallel=true)
    else
        nothing
    end

    # Model:
    nin = size(first(train_set)[1], 1)
    nout = size(first(train_set)[2], 1)
    model = build_model(model, nin, nout)

    # Input encoding:
    input_encoder = get_input_encoder(exp)

    return model, train_set, input_encoder, val_set
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
        Xtrain, y = load_mnist(data.n_train)
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

"""
    get_data(data::MNIST; n::Union{Nothing,Int}=data.n_train, test_set::Bool=false)

    
"""
function get_data(data::MNIST; n::Union{Nothing,Int}=data.n_train, test_set::Bool=false)
    if test_set
        X, y = load_mnist_test()
    else
        X, y = load_mnist()
    end
    n_total = size(X, 2)
    n = isnothing(n) ? n_total : n
    if n_total > n
        idx = sample(1:n_total, n; replace=false)

    elseif n_total < n
        idx = rand(1:n_total, n)
    else
        idx = 1:n_total
    end
    X = X[:, idx]
    y = y[idx]
    return X, y
end