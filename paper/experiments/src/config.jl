using TaijaParallel
using TOML

to_dict(x) = x

to_dict(fun::Function) = String(nameof(fun))

to_dict(generator_type::AbstractGeneratorType) = String(nameof(typeof(generator_type)))

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
    dict = to_dict(config)
    if isnothing(fname)
        TOML.print(dict)
    else
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
to_toml(exp::AbstractExperiment) = to_toml(exp, exp.meta_params.config_file)

"""
    from_toml(fname::String)::Dict

Generates a dictionary from a TOML file at the path specified by `fname`.
"""
function from_toml(fname::String)::Dict
    dict = TOML.parsefile(fname) 
    return dict
end

function to_meta(dict::Dict{String,Any})::MetaParams
    meta_kwrgs = to_ntuple(dict["meta_params"])
    meta_params = MetaParams(;meta_kwrgs...)
    return meta_params
end

to_ntuple(x) = x

function to_ntuple(dict::Dict)
    _names = Symbol.([k for (k, _) in dict])
    _values = [to_ntuple(v) for (_, v) in dict]
    kwrgs = (; zip(_names, _values)...)
    return kwrgs
end