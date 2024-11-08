using TaijaData
using MultivariateStats

get_domain(d::Dataset) = nothing

include("mnist.jl")

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

include("mlp.jl")

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


