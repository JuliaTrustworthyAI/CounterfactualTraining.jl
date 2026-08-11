# Benchmark harness for the Native counterfactual training path.
#
# Adapted from the PERFORMANCE_PLAN.md Phase 0 sketch to run on CPU with only
# the deps available in this environment (no MLDatasets/Metalhead/CUDA/AMDGPU).
# Uses a Dense model on synthetic data, matching the test-suite models so that
# parity is verifiable and relative speedups are measurable.
#
# Device: CPU (identity). All before/after comparisons must use this same device.
#
# Usage:
#   julia --project=. dev/bench_native.jl | tee dev/bench_baseline.txt

using CounterfactualTraining
using CounterfactualTraining.Native
using CounterfactualExplanations
using Flux
using Random
using Statistics
using Printf

Random.seed!(42)

# --- Device (CPU in this environment) ---
device = identity

# --- Synthetic data: 2-class, D=128, N=8000, two clusters ---
D = 128
N = 8000
X = hcat(randn(Float32, D, N ÷ 2), randn(Float32, D, N ÷ 2) .+ 3.0f0)
y = vcat(fill(1, N ÷ 2), fill(2, N ÷ 2))
domain = [(-5.0f0, 5.0f0) for _ in 1:D]
y_onehot = Flux.onehotbatch(y, 1:2)
train_set = Flux.DataLoader((X, y_onehot); batchsize=128, shuffle=true)

gen = NativeGenerator()
opt = Flux.Adam(1e-3)
nepochs = 3          # short run; burnin rounds to 1 epoch, leaving 2 CF epochs

make_model() = Chain(Dense(D, 256, relu), Dense(256, 256, relu), Dense(256, 2))

function run_training(objective; kw...)
    Random.seed!(42)
    m = make_model()
    o = Flux.setup(opt, m)
    _, log = counterfactual_training(objective, m, gen, train_set, o;
        device, nepochs, domain, verbose=0, nce=128, cf_batchsize=32,
        maxiter=30, burnin=0.2f0, accuracy_every=nepochs, kw...)
    return [l.time_taken for l in log], [l.percent_valid for l in log]
end

# Standalone timing of CF generation (no training):
function time_generate_native()
    m = make_model() |> device
    t = @timed generate_native!(m, train_set, gen;
        nsamples=128, domain=domain, maxiter=30, verbose=0,
        device=device, cf_batchsize=32)
    return t.time
end

# Warmup (compilation) run, discard:
run_training(FullObjective())
time_generate_native()

# Measured runs:
for rep in 1:3
    t_full, pv = run_training(FullObjective())
    @printf("full[%d]:    %s  (percent_valid=%s)\n", rep, t_full, pv)
end
for rep in 1:3
    t_van, _ = run_training(VanillaObjective(; needs_ce=false))
    @printf("vanilla[%d]: %s\n", rep, t_van)
end
println("generate_native! standalone: ", time_generate_native(), " s")
