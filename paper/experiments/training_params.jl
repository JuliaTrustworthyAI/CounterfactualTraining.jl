using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using Flux

const Opt = Flux.Optimise.AbstractOptimiser

abstract type AbstractGeneratorType end

struct ECCo end

function get_generator(params::GeneratorParams, type::ECCo=params.type)::AbstractGenerator
    return ECCoGenerator(; opt=params.search_opt, λ=params.λ[1:2])
end

Base.@kwdef struct GeneratorParams
    type::AbstractGeneratorType = ECCo()
    search_opt::Opt = Descent(1.0f0)
    max_iter::Int = 50
    λ::AbstractVector{<:AbstractFloat} = [0.001f0, 5.0f0]
end

Base.@kwdef struct TrainingParams
    burnin::AbstractFloat = 0.0f0
    nepochs::Int = 100
    generator_params::GeneratorParams = GeneratorParams()
    nce::Int = 100
    conv::AbstractConvergence = Convergence.MaxIterConvergence(; max_iter=max_iter)
    training_opt::Opt = Adam()
    input_encoder::Union{Nothing,DataPreprocessing.InputTransformer} = nothing
    verbose::Bool = true
end