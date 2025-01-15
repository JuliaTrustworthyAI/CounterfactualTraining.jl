"The default benchmarking measures."
const CE_MEASURES = [
    validity,
    plausibility_distance_from_target,
    plausibility_energy_differential,
    MMD(;
        kernel=with_lengthscale(KernelFunctions.GaussianKernel(), 5.0), compute_p=nothing
    ),
    distance,
    redundancy,
]

function get_global_param(argname::String, defaultval::T) where T <: Any
    if any((x -> contains(x, "--$(argname)=")).(ARGS))
        arg = ARGS[(x -> contains(x, "--$(argname)=")).(ARGS)]
        @assert length(arg) == 1 "Please provide exactly one value for $(argname)."
        val = replace(arg[1], "--$(argname)=" => "")
        return parse(T, val)
    else
        return defaultval
    end
end