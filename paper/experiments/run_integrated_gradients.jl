using Accessors
using BSON
using CTExperiments
using CTExperiments.CSV
using CTExperiments.DataFrames
using CounterfactualExplanations
using DotEnv
using Logging
using MPI
using Serialization
using Statistics
using TaijaParallel

res_dir = get_global_param("res_dir", "paper/experiments/output/satml/mutability")
keep_models = [get_global_param("drop_models", "mlp")]
nrounds = get_global_param("nrounds", 100)
nsamples = get_global_param("nsamples", 2500)
verbose = get_global_param("verbose", false)

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    global_logger(NullLogger())
    expers = nothing
else
    @info "Running on $nprocs processes" 
    @info "Looking for results in $res_dir"
    # Get experiments for all datasets:
    expers = final_results(res_dir; keep_models)[2]
    expers = load_list.(expers)
    output = DataFrame[]
end

# Broadcast exper_list from rank 0 to all ranks
expers = MPI.bcast(expers, comm; root=0)

# Compute IG:
for (i, exper_list) in enumerate(expers)
    @info "Data set $i/$(length(expers))"
    data = CTExperiments.dname(exper_list[1].data)      # get dataset name

    # Bootstrap sample indices
    X, y = get_data(exper_list[1].data; test_set=true)
    idx = [rand(1:size(X, 2), nsamples) for i in 1:nrounds]

    for exper in exper_list
        @assert exper.data.mutability isa Vector{Int} "No mutability constraints specified"
        exper_name = exper.meta_params.experiment_name
        obj = exper.training_params.objective

        igs = CTExperiments.integrated_gradients(
            exper;
            idx=idx,
            nrounds=nrounds,
            comm=comm,
            max_entropy=false,
            baseline_type="random",
            verbose,
        )

        if rank == 0

            # Collect:
            if size(X, 1) == 2
                igs = (ig -> (abs.(ig)) ./ (maximum(ig) .- minimum(ig))).(igs)    # compute normalized contributions
            else
                igs = (ig -> (ig .- minimum(ig)) ./ (maximum(ig) .- minimum(ig))).(igs)    # compute normalized contributions
            end

            igs = (ig -> ig[exper.data.mutability, 1]).(igs) |> igs -> reduce(hcat, igs)

            # aggregate:
            m = mean(igs; dims=1)           # across features
            lb = quantile(vec(m), 0.05/2)
            ub = quantile(vec(m), 1 - 0.05/2)
            m = median(vec(m))[1]        # across rounds
            df = DataFrame(Dict(:data => data, :objective => obj, :mean => m, :lb => lb, :ub => ub))
            display(df)

            push!(output, df)
        end
    end
end

if rank == 0
    output = reduce(vcat, output)
    Serialization.serialize(joinpath(res_dir, "ig.jls"), output)
    CSV.write(joinpath(res_dir, "ig.csv"), output)
end
