using Flux
using AMDGPU
using Statistics
using Printf

AMDGPU.allowscalar(false)

const N_EPOCHS = 10
const N_HIDDEN = 1024
const BATCHSIZE = 1000
const N_SAMPLES = 10000

function make_model()
    return Chain(
        Dense(div(N_HIDDEN, 2), N_HIDDEN, relu),
        Dense(N_HIDDEN, N_HIDDEN, relu),
        Dense(N_HIDDEN, N_HIDDEN, relu),
        Dense(N_HIDDEN, div(N_HIDDEN, 2), relu),
        Dense(div(N_HIDDEN, 2), 10),
    )
end

display(make_model())
println()

function make_data(n=N_SAMPLES)
    X = randn(Float32, div(N_HIDDEN, 2), n)
    y = Flux.onehotbatch(rand(1:10, n), 1:10)
    return X, y
end

function train!(model, X, y, opt_state; epochs=N_EPOCHS)
    loader = Flux.DataLoader((X, y); batchsize=BATCHSIZE, shuffle=true)
    for epoch in 1:epochs
        for (xb, yb) in loader
            grads = Flux.gradient(model) do m
                Flux.logitcrossentropy(m(xb), yb)
            end
            Flux.update!(opt_state, model, grads[1])
        end
    end
end

println("\nBuilding data + model...")
X, y = make_data()

println("\n── CPU ────────────────────────────────────")
model_cpu = make_model()
opt_cpu = Flux.setup(Flux.Adam(1e-3), model_cpu)
train!(model_cpu, X, y, opt_cpu; epochs=1)
cpu_r = @timed train!(model_cpu, X, y, opt_cpu; epochs=N_EPOCHS)
@printf(
    "CPU: %.3fs for %d epochs  (GC: %.1f%%)\n",
    cpu_r.time,
    N_EPOCHS,
    100 * cpu_r.gctime / cpu_r.time
)

println("\n── GPU ────────────────────────────────────")
model_gpu = make_model() |> gpu
opt_gpu = Flux.setup(Flux.Adam(1e-3), model_gpu)
X_gpu, y_gpu = X |> gpu, y |> gpu
train!(model_gpu, X_gpu, y_gpu, opt_gpu; epochs=1)
AMDGPU.synchronize()
gpu_r = @timed begin
    train!(model_gpu, X_gpu, y_gpu, opt_gpu; epochs=N_EPOCHS)
    AMDGPU.synchronize()
end
@printf(
    "GPU: %.3fs for %d epochs  (GC: %.1f%%)\n",
    gpu_r.time,
    N_EPOCHS,
    100 * gpu_r.gctime / gpu_r.time
)

println("\n── Result ─────────────────────────────────")
@printf("Speedup: %.2fx\n", cpu_r.time / gpu_r.time)
