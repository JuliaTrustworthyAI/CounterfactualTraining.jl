using CounterfactualExplanations.Evaluation: validity_strict
using JSON

function get_ce_measures(;
    use_mmd=get_global_param("use_mmd", true),
    length_scale=get_global_param("length_scale", 5.0),
    compute_p=get_global_param("compute_p", nothing),
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
