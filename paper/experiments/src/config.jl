using TaijaParallel
using TOML

abstract type AbstractConfiguration end

abstract type AbstractExperiment <: AbstractConfiguration end

to_dict(x) = x

to_dict(fun::Function) = String(nameof(fun))

to_dict(generator_type::AbstractGeneratorType) = String(nameof(typeof(generator_type)))

function to_dict(config::AbstractConfiguration)
    return Dict(
        fieldnames(typeof(config)) .=>
            to_dict.(getfield.(Ref(config), fieldnames(typeof(config)))),
    )
end

function to_toml(exp::AbstractExperiment)
    TOML.print(to_dict(exp))
end