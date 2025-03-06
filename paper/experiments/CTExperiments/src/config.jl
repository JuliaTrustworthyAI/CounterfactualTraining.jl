using Markdown
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
    to_dict(domain::Vector)

Handles the case where domain constraints are automatically inferred.
"""
function to_dict(domain::Vector)
    if eltype(domain) == Tuple
        dom = [[x...] for x in domain]
    else
        dom = domain
    end
    return dom
end

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
                return nothing
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

# Function to check if a value is effectively empty
function is_empty_value(v::Any)
    if v isa String
        return isempty(v)
    elseif v isa Vector
        return isempty(v)
    elseif v isa Dict
        # A dictionary is empty if it's empty itself or if all its filtered values would be empty
        filtered = filter_dict(v)
        return isempty(filtered)
    else
        return false
    end
end

# Function to filter dictionary
function filter_dict(dict::Dict; drop_fields=["name", "data", "data_params"])
    # Filter out empty values and specified fields
    return filter(dict) do (k, v)
        !is_empty_value(v) && !(k in drop_fields)
    end
end

global LatexReplacements = Dict(
    "lambda_energy" => "\$\\lambda_{\\text{energy}}\$",
    "lambda_cost" => "\$\\lambda_{\\text{cost}}\$",
    "lambda_adversarial" => "\$\\lambda_{\\text{adv}}\$",
    "lambda_energy_diff" => "\$\\lambda_{\\text{div}}\$",
    "lambda_energy_reg" => "\$\\lambda_{\\text{reg}}\$",
    "lambda_class_loss" => "\$\\lambda_{\\text{yloss}}\$",
)

function format_header(s::String; replacements::Dict=LatexReplacements)
    s =
        replace(s, "nce" => "ncounterfactuals") |>
        s ->
        replace(s, "_exper" => "") |>
        s ->
            replace(s, "_eval" => "") |>
            s ->
                replace(s, "_type" => "") |>
                s ->
                    replace(s, "_params" => "_parameters") |>
                    s ->
                        replace(s, "lr" => "learning_rate") |>
                        s ->
                            replace(s, "maxiter" => "maximum_iterations") |>
                            s ->
                                replace(s, "opt" => "optimizer") |>
                                s ->
                                    replace(s, "conv" => "convergence") |>
                                    s ->
                                        replace(s, r"\bopt\b" => "optimizer") |>
                                        s ->
                                            replace(s, r"^n" => "no._") |>
                                                s ->
                                                    replace(s, "__" => "_") |>
                                            s -> if s in keys(replacements)
                                                replacements[s]
                                            else
                                                s |>
                                                s ->
                                                    split(s, "_") |>
                                                    ss ->
                                                        [uppercasefirst(s) for s in ss] |> ss -> join(ss, " ")
                                            end
    return s
end

function to_mkd(dict::Dict, level::Int=0; header::Union{Nothing,String}=nothing)
    drop_fields = [
        "name",
        "concatenate_output",
        "parallelizer",
        "store_ce",
        "threaded",
        "verbose",
        "vertical_splits",
        "grid_file",
        "inherit",
        "save_dir",
        "test_time",
        "ndiv",
    ]
    dict = filter(((k, v),) -> length(v) > 0 && !(k in drop_fields), dict)

    # Create indent string based on level
    indent = repeat("    ", level)

    # Initialize array to store markdown lines
    if isnothing(header)
        lines = String[]
    else
        header = "\n*$header*\n"
        lines = [header]
    end

    # Sort dictionary keys for consistent output
    for key in sort(collect(keys(dict)))
        value = dict[key]
        key = format_header(key; replacements=LatexReplacements)

        if value isa Dict
            # Handle nested dictionary
            push!(lines, "$(indent)- $(key):")
            # Recursively process nested dictionary with increased indentation
            nested_lines = to_mkd(value, level + 1)
            push!(lines, nested_lines)
        elseif value isa Vector
            # Handle vector values by joining with commas
            value_str = join(value, ", ")
            push!(lines, "$(indent)- $(key): `$(value_str)`")
        else
            # Handle single values
            push!(lines, "$(indent)- $(key): `$(value)`")
        end
    end

    # Join all lines with newlines
    return join(lines, "\n")
end

# Function to create final Markdown string
function dict_to_markdown(dict::Dict; header::Union{Nothing,String}=nothing)
    filtered_dict = filter_dict(dict)
    return "md\"\"\"\n$(to_mkd(filtered_dict; header=header))\n\"\"\""
end

# New function specifically for Quarto output
function dict_to_quarto_markdown(dict::Dict; header::Union{Nothing,String}=nothing)
    filtered_dict = filter_dict(dict)
    return "$(to_mkd(filtered_dict; header=header))\n"
end
