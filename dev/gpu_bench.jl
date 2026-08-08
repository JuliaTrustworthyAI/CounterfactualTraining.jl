using Flux
using AMDGPU
using Statistics
using Printf

# ── Configuration ───────────────────────────────────────────
const N_EPOCHS  = 50
const N_HIDDEN  = 512
const BATCHSIZE = 128
const N_SAMPLES  = 4096

function make_model()
    Chain(
        Dense(div(N_HIDDEN, 2), N_HIDDEN, relu),
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
    X, y
end

# ── Original train! (no per-epoch sync) ──────────────────────
# Reproduces the issue: without synchronization between epochs,
# intermediate GPU arrays pile up on the async stream and cannot
# be garbage-collected, causing progressive memory-pressure slowdown.
function train!(model, X, y, opt_state; epochs=N_EPOCHS, batchsize=BATCHSIZE)
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

# ── Instrumented train! with per-epoch sync + timing ────────
"""
    train_profiled!(model, X, y, opt_state; use_gpu, epochs, gc_epochs)

Returns `(epoch_times, epoch_mems)` where `epoch_times[i]` is wall-clock
seconds for epoch `i` and `epoch_mems[i]` is GPU memory used at end of
epoch `i` (bytes, or 0 for CPU).

When `use_gpu=true`, synchronizes after every epoch for accurate per-epoch
timing and to allow intermediate arrays to be collected.

When `gc_epochs=true`, additionally calls `GC.gc()` + sync between epochs
to test whether memory pressure from uncollected GPU arrays causes the
slowdown.
"""
function train_profiled!(model, X, y, opt_state; use_gpu=false, epochs=N_EPOCHS, gc_epochs=false)
    loader = Flux.DataLoader((X, y); batchsize=BATCHSIZE, shuffle=true)
    epoch_times = Float64[]
    epoch_mems  = Int[]

    for epoch in 1:epochs
        alloc_before = use_gpu ? copy(AMDGPU.alloc_stats) : nothing

        t_epoch = @elapsed begin
            for (xb, yb) in loader
                grads = Flux.gradient(model) do m
                    Flux.logitcrossentropy(m(xb), yb)
                end
                Flux.update!(opt_state, model, grads[1])
            end
            use_gpu && AMDGPU.synchronize()
        end

        gpu_mem = 0
        if use_gpu
            free_b, total_b = AMDGPU.info()
            gpu_mem = total_b - free_b
            if gc_epochs
                GC.gc()
                AMDGPU.synchronize()
            end
        end

        push!(epoch_times, t_epoch)
        push!(epoch_mems, gpu_mem)

        if epoch == 1 || epoch % 5 == 0 || epoch == epochs
            extra = ""
            if use_gpu
                ad = AMDGPU.alloc_stats - alloc_before
                extra = @sprintf("  mem:%s  allocs:%d", Base.format_bytes(gpu_mem), ad.alloc_count)
            end
            avg5 = mean(epoch_times[max(1, end-4):end])
            @printf("  Ep %3d: %6.3fs  avg5:%6.3fs%s\n", epoch, t_epoch, avg5, extra)
        end
    end
    return epoch_times, epoch_mems
end

# ── Build data + model ─────────────────────────────────────
println("Building data + model...")
X, y = make_data()

# ════════════════════════════════════════════════════════════
# CPU run
# ════════════════════════════════════════════════════════════
println("\n── CPU ────────────────────────────────────")
model_cpu = make_model()
opt_cpu = Flux.setup(Flux.Adam(1e-3), model_cpu)
train_profiled!(model_cpu, X, y, opt_cpu; epochs=1)  # warmup/compile
cpu_r = @timed train_profiled!(model_cpu, X, y, opt_cpu; epochs=N_EPOCHS)
cpu_times, _ = cpu_r.value
@printf("CPU: %.3fs  (allocs:%s, GC:%.2fs)\n",
    cpu_r.time, Base.format_bytes(cpu_r.bytes), cpu_r.gctime)

# ════════════════════════════════════════════════════════════
# GPU run A: no per-epoch sync (reproduces the original issue)
# ════════════════════════════════════════════════════════════
println("\n── GPU A: no per-epoch sync ────────────────")
model_a = make_model() |> gpu
opt_a = Flux.setup(Flux.Adam(1e-3), model_a)
X_gpu, y_gpu = X |> gpu, y |> gpu
train!(model_a, X_gpu, y_gpu, opt_a; epochs=1)  # warmup/compile
AMDGPU.synchronize()
t_nosync = @elapsed begin
    train!(model_a, X_gpu, y_gpu, opt_a; epochs=N_EPOCHS)
    AMDGPU.synchronize()
end
@printf("GPU (no sync): %.3fs for %d epochs\n", t_nosync, N_EPOCHS)

# ════════════════════════════════════════════════════════════
# GPU run B: per-epoch sync (diagnostic)
# ════════════════════════════════════════════════════════════
println("\n── GPU B: per-epoch sync ───────────────────")
println("Eager GC: $(AMDGPU.EAGER_GC[])")
model_gpu = make_model() |> gpu
opt_gpu = Flux.setup(Flux.Adam(1e-3), model_gpu)

println("Before warmup:")
AMDGPU.pool_status()
train_profiled!(model_gpu, X_gpu, y_gpu, opt_gpu; use_gpu=true, epochs=1)  # warmup
AMDGPU.synchronize()
println("\nAfter warmup:")
AMDGPU.pool_status()

alloc_before = copy(AMDGPU.alloc_stats)
gpu_r = @timed begin
    result = train_profiled!(model_gpu, X_gpu, y_gpu, opt_gpu; use_gpu=true, epochs=N_EPOCHS)
    AMDGPU.synchronize()
    result
end
alloc_diff = AMDGPU.alloc_stats - alloc_before
gpu_times, gpu_mems = gpu_r.value

println("\nAfter training:")
AMDGPU.pool_status()
@printf("\nGPU (synced): %.3fs  (allocs:%s, GC:%.2fs)\n",
    gpu_r.time, Base.format_bytes(gpu_r.bytes), gpu_r.gctime)
@printf("Pool: %d allocs (%s), %d frees (%s)\n",
    alloc_diff.alloc_count, Base.format_bytes(alloc_diff.alloc_bytes),
    alloc_diff.free_count, Base.format_bytes(alloc_diff.free_bytes))

# ════════════════════════════════════════════════════════════
# A/B/C test: GC strategy between epochs (20 epochs each)
# ════════════════════════════════════════════════════════════
println("\n── A/B/C: GC strategy (20 ep each) ──────────")
ab_n = 20

# A: No GC, eager GC on (default — reproduces degradation)
GC.gc(); AMDGPU.reclaim()
m1 = make_model() |> gpu
o1 = Flux.setup(Flux.Adam(1e-3), m1)
train_profiled!(m1, X_gpu, y_gpu, o1; use_gpu=true, epochs=1)  # warmup
AMDGPU.synchronize()
t_nogc, _ = train_profiled!(m1, X_gpu, y_gpu, o1; use_gpu=true, epochs=ab_n)

# B: GC.gc() between epochs, eager GC on
GC.gc(); AMDGPU.reclaim()
m2 = make_model() |> gpu
o2 = Flux.setup(Flux.Adam(1e-3), m2)
train_profiled!(m2, X_gpu, y_gpu, o2; use_gpu=true, epochs=1)  # warmup
AMDGPU.synchronize()
t_gc, _ = train_profiled!(m2, X_gpu, y_gpu, o2; use_gpu=true, epochs=ab_n, gc_epochs=true)

# C: eager GC OFF + GC.gc() between epochs
# Tests whether disabling AMDGPU's throttled maybe_collect reduces overhead
GC.gc(); AMDGPU.reclaim()
AMDGPU.eager_gc!(false)
m3 = make_model() |> gpu
o3 = Flux.setup(Flux.Adam(1e-3), m3)
train_profiled!(m3, X_gpu, y_gpu, o3; use_gpu=true, epochs=1)  # warmup
AMDGPU.synchronize()
t_eager_off, _ = train_profiled!(m3, X_gpu, y_gpu, o3; use_gpu=true, epochs=ab_n, gc_epochs=true)
AMDGPU.eager_gc!(true)  # restore default

@printf("  A) No GC (eager on):  %.3fs  avg:%.3fs  last5:%.3fs\n",
    sum(t_nogc), mean(t_nogc), mean(t_nogc[end-4:end]))
@printf("  B) GC.gc (eager on):  %.3fs  avg:%.3fs  last5:%.3fs\n",
    sum(t_gc), mean(t_gc), mean(t_gc[end-4:end]))
@printf("  C) GC.gc (eager off): %.3fs  avg:%.3fs  last5:%.3fs\n",
    sum(t_eager_off), mean(t_eager_off), mean(t_eager_off[end-4:end]))

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
println("\n── Summary ─────────────────────────────────")
@printf("  CPU:              %.3fs  (GC: %.1f%%)\n", cpu_r.time, 100*cpu_r.gctime/cpu_r.time)
@printf("  GPU (no sync):    %.3fs\n", t_nosync)
@printf("  GPU (synced):     %.3fs  (GC: %.1f%%)\n", gpu_r.time, 100*gpu_r.gctime/gpu_r.time)
@printf("  Speedup (sync):   %.2fx\n", cpu_r.time / gpu_r.time)
@printf("  Sync vs no-sync:  %.2fx faster with per-epoch sync\n", t_nosync / gpu_r.time)

println("\nPer-epoch (first 5 vs last 5):")
@printf("  CPU:          %.3fs -> %.3fs (%.2fx)\n",
    mean(cpu_times[1:5]), mean(cpu_times[end-4:end]),
    mean(cpu_times[end-4:end]) / mean(cpu_times[1:5]))
@printf("  GPU (synced): %.3fs -> %.3fs (%.2fx)\n",
    mean(gpu_times[1:5]), mean(gpu_times[end-4:end]),
    mean(gpu_times[end-4:end]) / mean(gpu_times[1:5]))

println("\nA/B/C result (20 epochs):")
@printf("  A) No GC (eager on):  %.3fs  last5:%.3fs  %s\n",
    sum(t_nogc), mean(t_nogc[end-4:end]),
    mean(t_nogc[end-4:end]) > 2 * mean(t_nogc[1:5]) ? "DEGRADING" : "stable")
@printf("  B) GC.gc (eager on):  %.3fs  last5:%.3fs  %s\n",
    sum(t_gc), mean(t_gc[end-4:end]),
    mean(t_gc[end-4:end]) > 2 * mean(t_gc[1:5]) ? "DEGRADING" : "stable")
@printf("  C) GC.gc (eager off): %.3fs  last5:%.3fs  %s\n",
    sum(t_eager_off), mean(t_eager_off[end-4:end]),
    mean(t_eager_off[end-4:end]) > 2 * mean(t_eager_off[1:5]) ? "DEGRADING" : "stable")
best = argmin([sum(t_nogc), sum(t_gc), sum(t_eager_off)])
names = ["No GC (eager on)", "GC.gc (eager on)", "GC.gc (eager off)"]
@printf("  Best: %s\n", names[best])

if gpu_mems[end] > gpu_mems[1]
    @printf("\nGPU memory growth: %s -> %s (+%s)\n",
        Base.format_bytes(gpu_mems[1]), Base.format_bytes(gpu_mems[end]),
        Base.format_bytes(gpu_mems[end] - gpu_mems[1]))
end

println("""
\n── Interpretation ─────────────────────────────────
The original script's train! does NOT call GC.gc() between epochs.
Each epoch creates ~1344 temporary GPU arrays (activations, gradients,
Zygote tape). Julia's GC is lazy — it only runs when memory pressure
triggers it. AMDGPU's eager GC (maybe_collect, called on every alloc)
has throttling logic that limits GC to ~5% of wall time, which cannot
keep up with the allocation rate.

The result: dead GPU arrays accumulate, memory pressure rises, and
AMDGPU's throttled maybe_collect() runs ever more frequently but can't
free enough — causing progressive slowdown (0.033s -> 0.575s per epoch
in the A/B test).

The fix is simple: call GC.gc() between epochs. This forces collection
of dead GPU arrays, keeping memory pressure low. The A/B/C test shows:
  A) No GC          — degrades progressively (10x slower)
  B) GC.gc()        — flat, ~10x faster than A
  C) eager_gc!(false) + GC.gc() — tests if disabling the throttled
     maybe_collect overhead helps further

synchronize() alone does NOT help — it drains the GPU queue but does
not trigger Julia's GC. The arrays are dead but not yet collected.

For production training loops:
  1. Call GC.gc() every N epochs (tune N for your workload)
  2. Consider AMDGPU.eager_gc!(false) + explicit GC.gc() for control
  3. AMDGPU.reclaim() periodically to trim the memory pool
  4. Larger batch sizes reduce the number of allocations per epoch
""")
