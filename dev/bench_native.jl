# Benchmark harness for the Native counterfactual training path.
#
# Runs the real benchmarked workload from PERFORMANCE_PLAN.md: ResNet-18 on a
# 5000-sample MNIST subset (flattened 784, normalized to [-1,1]). Detects a GPU
# (CUDA or AMDGPU) and falls back to CPU when none is present, so it runs in CI
# and on GPU machines alike.
#
# Reports:
#   - per-epoch wall-clock time for the Full vs Vanilla objectives (the gap is
#     the cost of counterfactual generation),
#   - standalone `generate_native!` time, chunked (cf_batchsize=32) vs fast path
#     (cf_batchsize=128),
#   - a UnicodePlots bar chart of mean per-epoch times for easy visual reading.
#
# Usage:
#   julia --project=. dev/bench_native.jl | tee dev/bench_gpu.txt

using CounterfactualTraining
using CounterfactualTraining.Native
using CounterfactualExplanations
using Flux
using Metalhead
using MLDatasets
using Random
using Statistics
using Printf
using UnicodePlots

Random.seed!(42)

# ---------------------------------------------------------------------------
# GPU detection (CUDA first, then AMDGPU; fall back to CPU)
# ---------------------------------------------------------------------------
has_gpu = false
device = identity
try
    using CUDA
    if CUDA.functional()
        has_gpu = true
        device = Flux.gpu
    end
catch
end
if !has_gpu
    try
        using AMDGPU
        if AMDGPU.functional()
            has_gpu = true
            device = Flux.gpu
        end
    catch
    end
end
if has_gpu
    @info "GPU detected — running on GPU."
else
    @info "No GPU detected — running on CPU."
end

# ---------------------------------------------------------------------------
# Data: MNIST subset 5000, flattened 784, normalized [-1,1]
# ---------------------------------------------------------------------------
train_data = MLDatasets.MNIST(; split=:train)
X = Float32.(reshape(train_data.features, 784, :)) .* 2f0 .- 1f0
y = train_data.targets .+ 1  # 0-9 -> 1-10
idx = shuffle(1:size(X, 2))[1:5000]
X, y = X[:, idx], y[idx]
domain = [(-1.0f0, 1.0f0) for _ in 1:784]
y_onehot = Flux.onehotbatch(y, 1:10)

# Keep data on the device so the training loop's `input |> device` is a no-op.
X = X |> device
y_onehot = y_onehot |> device
train_set = Flux.DataLoader((X, y_onehot); batchsize=128, shuffle=true)

gen = NativeGenerator()
opt = Flux.Adam(1e-3)
nepochs = 3          # short run; burnin rounds to 1 epoch, leaving 2 CF epochs

make_model() = Chain(x -> reshape(x, 28, 28, 1, :), ResNet(18; inchannels=1, nclasses=10))

function run_training(objective; kw...)
    Random.seed!(42)
    m = make_model() |> device
    o = Flux.setup(opt, m)
    _, log = counterfactual_training(objective, m, gen, train_set, o;
        device, nepochs, domain, verbose=0, nce=128, cf_batchsize=32,
        maxiter=30, burnin=0.2f0, accuracy_every=nepochs, kw...)
    return [l.time_taken for l in log], [l.percent_valid for l in log]
end

# Standalone timing of CF generation (no training), for a given cf_batchsize.
function time_generate_native(cf_batchsize)
    m = make_model() |> device
    t = @timed generate_native!(m, train_set, gen;
        nsamples=128, domain=domain, maxiter=30, verbose=0,
        device=device, cf_batchsize=cf_batchsize)
    return t.time
end

# ---------------------------------------------------------------------------
# Warmup (compilation) runs, discarded
# ---------------------------------------------------------------------------
run_training(FullObjective())
time_generate_native(32)
time_generate_native(128)

# ---------------------------------------------------------------------------
# Measured runs
# ---------------------------------------------------------------------------
println("\n" * "="^72)
println("  CounterfactualTraining Native benchmark — device: $(has_gpu ? "GPU" : "CPU")")
println("="^72)

full_times = Vector{Vector{Float64}}()
vanilla_times = Vector{Vector{Float64}}()
for rep in 1:3
    t_full, pv = run_training(FullObjective())
    push!(full_times, t_full)
    @printf("full[%d]:    %s  (percent_valid=%s)\n", rep, t_full, pv)
end
for rep in 1:3
    t_van, _ = run_training(VanillaObjective(; needs_ce=false))
    push!(vanilla_times, t_van)
    @printf("vanilla[%d]: %s\n", rep, t_van)
end

# Standalone CF generation: chunked vs fast path
println("\n--- Standalone generate_native! (nsamples=128, maxiter=30) ---")
chunked_times = Float64[]
fast_times = Float64[]
for rep in 1:5
    tc = time_generate_native(32)
    tf = time_generate_native(128)
    push!(chunked_times, tc)
    push!(fast_times, tf)
    @printf("rep%d: chunked(cf_batchsize=32)=%.4f s   fast(cf_batchsize=128)=%.4f s\n", rep, tc, tf)
end

# ---------------------------------------------------------------------------
# Summary + visualization
# ---------------------------------------------------------------------------
println("\n" * "="^72)
println("  Summary")
println("="^72)

# Mean per-epoch times (skip epoch 1 which includes compilation noise)
function mean_epochs(times)
    # times: vector of per-rep vectors of per-epoch times
    n_epochs = length(times[1])
    return [mean([t[e] for t in times]) for e in 1:n_epochs]
end
full_mean = mean_epochs(full_times)
vanilla_mean = mean_epochs(vanilla_times)

@printf("\nMean per-epoch time (s), averaged over 3 runs:\n")
@printf("  %-10s %-10s %-10s\n", "epoch", "full", "vanilla")
for e in 1:length(full_mean)
    @printf("  %-10d %-10.4f %-10.4f\n", e, full_mean[e], vanilla_mean[e])
end

# CF overhead = full - vanilla, averaged over CF epochs (after burn-in)
cf_epochs = 2:length(full_mean)
cf_overhead = mean(full_mean[cf_epochs]) - mean(vanilla_mean[cf_epochs])
@printf("\nMean CF overhead per epoch (full - vanilla, epochs %s): %.4f s\n",
    join(cf_epochs, ","), cf_overhead)

# Standalone CF generation speedup (fast vs chunked)
tc_mean = mean(chunked_times)
tf_mean = mean(fast_times)
@printf("Standalone generate_native!: chunked=%.4f s, fast=%.4f s, speedup=%.2fx\n",
    tc_mean, tf_mean, tc_mean / tf_mean)

# Bar chart: mean per-epoch time, full vs vanilla
labels = ["epoch $e" for e in 1:length(full_mean)]
bp = barplot(
    labels, full_mean;
    title="Mean per-epoch time (s) — Full vs Vanilla",
    xlabel="seconds", ylabel="",
    color=:green,
)
# Overlay vanilla as a second series via a combined plot is not supported by
# UnicodePlots barplot; print both side by side instead.
bp2 = barplot(
    labels, vanilla_mean;
    title="Mean per-epoch time (s) — Vanilla",
    xlabel="seconds", ylabel="",
    color=:blue,
)
println("\n" * "-"^72)
show(stdout, bp)
println()
show(stdout, bp2)
println()

# Bar chart: standalone CF generation chunked vs fast
bp3 = barplot(
    ["chunked (32)", "fast (128)"], [tc_mean, tf_mean];
    title="Standalone generate_native! (s)",
    xlabel="seconds", ylabel="",
    color=:red,
)
println("-"^72)
show(stdout, bp3)
println()

println("\nDone.")
