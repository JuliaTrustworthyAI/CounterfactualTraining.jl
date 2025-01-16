using TaijaParallel
using TOML

"""
    to_dict(x)

When called on any object `x`, returns `x` as-is. 
"""
to_dict(x) = x

"""
    to_dict(fun::Function)

When called on any function `fun`, returns a string representation of its name. 
"""
to_dict(fun::Function) = String(nameof(fun))

"""
    to_dict(generator_type::AbstractGeneratorType)

When called on any generator type `generator_type`, returns a string representation of its name. 
"""
to_dict(generator_type::AbstractGeneratorType) = String(nameof(typeof(generator_type)))

"""
    to_dict(config::AbstractConfiguration)

When called on any configuration `config`, returns a dictionary representation of its fields and their values. 
"""
function to_dict(config::AbstractConfiguration)
    return Dict{String,Any}(
        String.(fieldnames(typeof(config))) .=>
            to_dict.(getfield.(Ref(config), fieldnames(typeof(config)))),
    )
end

"""
    to_toml(config::AbstractConfiguration, fname::Union{Nothing,String}=nothing)

Creates a [TOML](https://github.com/toml-lang/toml) file from the configuration `config`. If a file path (`fname`) is not provided then the file will be written to `STDOUT`. Otheriwse it will be written to `fname`.
"""
function to_toml(config::AbstractConfiguration, fname::Union{Nothing,String}=nothing)
    return to_toml(to_dict(config), fname)
end

function to_toml(dict::AbstractDict, fname::Union{Nothing,String}=nothing)
    if isnothing(fname)
        TOML.print(dict)
    else
        # Check if file exists and compare content
        if isfile(fname)
            dict_old = TOML.parsefile(fname)
            if dict == dict_old
                return 
            end
        end
        
        # Write only if file doesn't exist or content differs
        open(fname, "w") do io
            TOML.print(io, dict)
        end
        @info "Configuration written to $fname."
    end
end

"""
    to_toml(exp::AbstractExperiment, fname::Union{Nothing,String}=nothing)

Overloads the `to_toml` function for `Experiment`s. The file will be written to the path specified in the meta data (`exp.meta_params.config_file`). 
"""
to_toml(exp::AbstractExperiment) = to_toml(exp, config_file(exp.meta_params))

"""
    from_toml(fname::String)::Dict

Generates a dictionary from a TOML file at the path specified by `fname`.
"""
function from_toml(fname::String)::Dict
    dict = TOML.parsefile(fname)
    return dict
end

"""
    to_meta(dict::Dict{String,Any})::MetaParams

Converts a TOML dictionary to `MetaParams` object.
"""
function to_meta(dict::Dict{String,Any})::MetaParams
    dict = haskey(dict, "meta_params") ? dict["meta_params"] : dict
    meta_kwrgs = to_ntuple(dict)
    meta_params = MetaParams(; meta_kwrgs...)
    return meta_params
end

"""
    to_grid(dict::Dict{String,Any})::ExperimentGrid

Converts a TOML dictionary to `ExperimentGrid` object.
"""
function to_grid(dict::Dict{String,Any})::ExperimentGrid
    return (kwrgs -> ExperimentGrid(; kwrgs...))(CTExperiments.to_ntuple(dict))
end

"""
    to_ntuple(x)

When called on any object `x`, returns `x` as-is. 
"""
to_ntuple(x) = x

"""
    to_tuple(dict::Dict)

When called on any dictionary `dict`, returns a tuple of its key-value pairs.
"""
function to_ntuple(dict::Dict)
    _names = Symbol.([k for (k, _) in dict])
    _values = [to_ntuple(v) for (_, v) in dict]
    kwrgs = (; zip(_names, _values)...)
    return kwrgs
end

function has_results(cfg::AbstractConfiguration)
    return false
end

needs_results(cfg::AbstractConfiguration) = !has_results(cfg)
