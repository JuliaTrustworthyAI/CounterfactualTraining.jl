using CounterfactualExplanations
using CounterfactualExplanations: Convergence, ECCoGenerator
using CounterfactualTraining
using CounterfactualTraining.Native
using Flux
using Random
using Statistics

@testset "Training parity: native vs non-native" begin

    # Setup: simple 2D data
    Random.seed!(42)
    N = 200
    X = hcat(randn(Float32, 2, N ÷ 2), randn(Float32, 2, N ÷ 2) .+ 3.0f0)
    y = vcat(fill(1, N ÷ 2), fill(2, N ÷ 2))
    domain = [(-3.0f0, 6.0f0), (-3.0f0, 6.0f0)]
    mutability = [:both, :both]  # use :both only to isolate training parity

    y_onehot = Flux.onehotbatch(y, 1:2)
    train_set = Flux.DataLoader((X, y_onehot); batchsize=32, shuffle=true)

    # Shared parameters
    nepochs = 3
    maxiter = 10
    decision_threshold = 0.75f0
    obj = FullObjective(lambda=Float32[1.0, 0.5, 0.01, 0.1])

    # --- Non-native training (ECCoGenerator) ---
    Random.seed!(999)  # same model init
    model_ce = Chain(Dense(2, 32, relu), Dense(32, 2))
    opt_ce = Flux.setup(Flux.AMSGrad(), model_ce)
    gen_ce = ECCoGenerator(λ=[0.1f0, 1.0f0], opt=Flux.Descent(0.1f0))
    conv = Convergence.DecisionThresholdConvergence(;
        decision_threshold=decision_threshold, max_iter=maxiter
    )

    model_ce, log_ce = counterfactual_training(
        obj,
        model_ce,
        gen_ce,
        train_set,
        opt_ce;
        nepochs=nepochs,
        mutability=mutability,
        domain=domain,
        convergence=conv,
        verbose=0,
    )

    # --- Native training (NativeGenerator) ---
    Random.seed!(999)  # same model init
    model_native = Chain(Dense(2, 32, relu), Dense(32, 2))
    opt_native = Flux.setup(Flux.AMSGrad(), model_native)
    gen_native = NativeGenerator(λ=[0.1f0, 1.0f0], opt=Flux.Descent(0.1f0))

    model_native, log_native = counterfactual_training(
        obj,
        model_native,
        gen_native,
        train_set,
        opt_native;
        nepochs=nepochs,
        maxiter=maxiter,
        burnin=0.0f0,
        decision_threshold=decision_threshold,
        mutability=mutability,
        domain=domain,
        verbose=0,
    )

    # Compare model weights (tolerance for different gradient computation paths)
    # The weights won't be identical because:
    # 1. Non-native generates CFs per-sample with random target assignment
    # 2. Native generates CFs batched with random target assignment
    # 3. Different random number consumption patterns
    # So we check that both models achieve similar accuracy and loss
    @test log_ce[end].acc > 0.8
    @test log_native[end].acc > 0.8
    @test abs(log_ce[end].acc - log_native[end].acc) < 0.2
    @test isfinite(log_ce[end].train_loss)
    @test isfinite(log_native[end].train_loss)
end
