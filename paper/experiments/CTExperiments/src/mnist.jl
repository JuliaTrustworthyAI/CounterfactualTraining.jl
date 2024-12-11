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

function get_ce_data(data::MNIST, n_total::Int)
    return CounterfactualData(load_mnist(n_total)...)
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
    get_data(data::MNIST, test_set::Bool=false)

Load the MNIST data set. If `test_set` is true, load the test set; otherwise, load the training set.
"""
function get_data(data::MNIST, test_set::Bool=false)
    if test_set
        X, y = load_mnist_test()
    else
        X, y = load_mnist()
    end
end
