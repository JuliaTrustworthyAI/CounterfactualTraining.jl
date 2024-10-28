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

function to_toml(exp::AbstractExperiment)
    TOML.print(to_dict(exp))
end