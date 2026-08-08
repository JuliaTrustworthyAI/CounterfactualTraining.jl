using Flux
using AMDGPU
using Statistics

n_epochs = 50
n_hidden = 512

# A modestly-sized model — big enough that GPU parallelism matters,
# small enough to run in seconds either way.
function make_model()
    Chain(
        Dense(div(n_hidden, 2), n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, div(n_hidden, 2), relu),
        Dense(div(n_hidden, 2), 10),
    )
end

# Model architecture:
display(make_model())

function make_data(n=4096)
    X = randn(Float32, div(n_hidden, 2), n)
    y = Flux.onehotbatch(rand(1:10, n), 1:10)
    X, y
end

function train!(model, X, y, opt_state; epochs=n_epochs, batchsize=128)
    loader = Flux.DataLoader((X, y); batchsize=batchsize, shuffle=true)
    for epoch in 1:epochs
        for (xb, yb) in loader
            grads = Flux.gradient(model) do m
                Flux.logitcrossentropy(m(xb), yb)
            end
            Flux.update!(opt_state, model, grads[1])
        end
    end
end

println("Building data + model...")
X, y = make_data()

# ── CPU run ──────────────────────────────────────────────
model_cpu = make_model()
opt_cpu = Flux.setup(Flux.Adam(1e-3), model_cpu)
train!(model_cpu, X, y, opt_cpu; epochs=1)  # warmup/compile
t_cpu = @elapsed train!(model_cpu, X, y, opt_cpu; epochs=n_epochs)
println("CPU: $(round(t_cpu, digits=3))s for $n_epochs epochs")

# ── GPU run ──────────────────────────────────────────────
model_gpu = make_model() |> gpu
opt_gpu = Flux.setup(Flux.Adam(1e-3), model_gpu)
X_gpu, y_gpu = X |> gpu, y |> gpu
train!(model_gpu, X_gpu, y_gpu, opt_gpu; epochs=1)  # warmup/compile (JIT!)
AMDGPU.synchronize()
t_gpu = @elapsed begin
    train!(model_gpu, X_gpu, y_gpu, opt_gpu; epochs=n_epochs)
    AMDGPU.synchronize()
end
println("GPU: $(round(t_gpu, digits=3))s for $n_epochs epochs")

println("\nSpeedup: $(round(t_cpu / t_gpu, digits=2))x")
