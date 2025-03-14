using CounterfactualExplanations.Evaluation: validity_strict, feature_sensitivity, get_outid
using JSON

"""
    add_sensitivity!(bmk::Benchmark)

Adds a column indicating the index of feature `d` for which `feature_sensitivity` was computed.
"""
function add_sensitivity!(bmk::Benchmark)
    sens = get_global_param("sensitivity", nothing)
    if isnothing(sens)
        return bmk
    end
    sens = parse_sens(sens)
    feature_id = []
    for varname in bmk.evaluation.variable
        if !contains(varname, "sens")
            push!(feature_id, missing)
        else
            push!(feature_id, sens[get_outid(varname)])
        end
    end
    bmk.evaluation.feature_id .= feature_id
    return bmk
end

"""
    parse_sens(sensitivity)

Parses the `--sensitivity` command line argument.
"""
function parse_sens(sensitivity)
    return sensitivity = try
        parse.(Int, split(sensitivity, ","))
    catch
        error(
            "The `--sensitivity` argument should be either an integer (e.g. `--sensitivity=1`) or a comma-separated list of integers (e.g. `--sensitivity=1,2,5`)",
        )
    end
end

function get_ce_measures(;
    use_mmd=get_global_param("use_mmd", true),
    length_scale=get_global_param("length_scale", 5.0),
    compute_p=get_global_param("compute_p", nothing),
    sensitivity=get_global_param("sensitivity", nothing),
)
    measures = [
        validity,
        validity_strict,
        plausibility_distance_from_target,
        plausibility_energy_differential,
        distance,
        redundancy,
    ]

    if use_mmd
        push!(
            measures,
            MMD(;
                kernel=with_lengthscale(KernelFunctions.GaussianKernel(), length_scale),
                compute_p=compute_p,
            ),
        )
    end

    if !isnothing(sensitivity)
        sensitivity = parse_sens(sensitivity)
        sens(x; kwrgs...) = feature_sensitivity(x, sensitivity)
        push!(measures, sens)
    end

    return measures
end

function get_global_param(argname::String, defaultval::T) where {T<:Any}
    if any((x -> contains(x, "--$(argname)=")).(ARGS))
        arg = ARGS[(x -> contains(x, "--$(argname)=")).(ARGS)]
        @assert length(arg) == 1 "Please provide exactly one value for $(argname)."
        val = replace(arg[1], "--$(argname)=" => "")

        # Check if the value starts with [ to detect array format
        if startswith(val, "[") && endswith(val, "]")
            # Parse as JSON if it's an array format
            return JSON.parse(val)
        else
            # Original behavior for single values
            if !(T <: Nothing || T <: String)
                return parse(T, val)
            else
                return val
            end
        end
    else
        return defaultval
    end
end
