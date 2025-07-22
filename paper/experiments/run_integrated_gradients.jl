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

res_dir = get_global_param("res_dir", "paper/experiments/output/final_run/mutability")
keep_models = [get_global_param("drop_models", "mlp")]

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    global_logger(NullLogger())
    expers = nothing 
else
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
    data = CTExperiments.dname(exper_list[1].data)      # get dataset name
    X = get_data(exper_list[1].data)[1]
    dict = Dict(:data => data)
    for exper in exper_list
        @assert exper.data.mutability isa Vector{Int} "No mutability constraints specified"
        exper_name = exper.meta_params.experiment_name
        obj = exper.training_params.objective

        # distribute
        if data == "mnist"
            bl = -ones(size(X,1),1)
            igs = CTExperiments.integrated_gradients(exper; nrounds=10, verbose=true, comm=comm, max_entropy=false, baseline=bl)
        else
            igs = CTExperiments.integrated_gradients(exper; nrounds=10, verbose=true, comm=comm, max_entropy=false, baseline_type="random")
        end
        
        if rank == 0

            # Collect:
            igs = (ig -> ig ./ (maximum(ig) .- minimum(ig)) ).(igs)    # compute normalized contributions
            igs = (ig -> ig[exper.data.mutability,1]).(igs) |> igs -> reduce(hcat, igs)

            # Aggregate:
            m = mean(igs, dims=2)       # across rounds
            m = mean(m, dims=1)[1]      # across features
            se = std(igs, dims=2)       # across rounds 
            se = mean(se, dims=1)[1]    # across features
            df = DataFrame(Dict(
                :data => data,
                :objective => obj,
                :mean => m,
                :se => se,
            ))
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
