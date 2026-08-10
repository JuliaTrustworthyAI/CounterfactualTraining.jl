using CounterfactualExplanations
using CounterfactualExplanations: Convergence, Models, ECCoGenerator
using CounterfactualExplanations: generate_counterfactual, polynomial_decay
using CounterfactualTraining
using CounterfactualTraining.Native
using Flux
using Random
using Statistics

@testset "Generator parity: ECCoGenerator vs NativeGenerator" begin

    # Setup: simple 2D linearly separable data
    Random.seed!(42)
    N = 200
    X = hcat(randn(Float32, 2, N ÷ 2), randn(Float32, 2, N ÷ 2) .+ 3.0f0)
    y = vcat(fill(1, N ÷ 2), fill(2, N ÷ 2))
    domain = [(-3.0f0, 6.0f0), (-3.0f0, 6.0f0)]
    mutability = [:both, :both]  # use :both only to isolate generator loss parity

    # Fixed model (same weights for both generators)
    Random.seed!(123)
    model = Chain(Dense(2, 32, relu), Dense(32, 2))

    # CounterfactualData (shared)
    data = CounterfactualData(X, y; domain=domain, mutability=mutability)

    # CE.jl model wrapper
    M = Models.Model(model, Models.FluxNN(); likelihood=:classification_multi)

    # Generator parameters (matching ECCoGenerator defaults)
    λ = [0.1f0, 1.0f0]
    opt = Flux.Descent(0.1f0)
    maxiter = 30
    decision_threshold = 0.75f0
    decay = 0.9f0
    reg_strength = 1.0f-3

    # --- N=1 parity test ---
    @testset "N=1" begin
        # Pick a sample from class 1, target class 2
        x_test = X[:, 1:1]
        target = 2

        # CE.jl ECCoGenerator
        gen_ce = ECCoGenerator(λ=λ, opt=opt)
        conv = Convergence.DecisionThresholdConvergence(;
            decision_threshold=decision_threshold, max_iter=maxiter
        )
        ce = generate_counterfactual(
            x_test, target, data, M, gen_ce;
            initialization=:identity,  # match Native's X′ = copy(X)
            convergence=conv,
        )
        cf_ce = ce.counterfactual[:, 1]

        # Native NativeGenerator
        gen_native = NativeGenerator(λ=λ, opt=opt)
        cfs_native, _, _, _ = generate_counterfactuals!(
            model, x_test, [target], data, gen_native;
            maxiter=maxiter, decision_threshold=decision_threshold,
            decay=decay, reg_strength=reg_strength,
        )
        cf_native = cfs_native[:, 1]

        # Compare (tolerance for Float32 vs Float64 and Zygote vs Flux.withgradient)
        @test size(cf_ce) == size(cf_native)
        @test cf_ce ≈ cf_native atol = 1e-2 rtol = 1e-2
    end

    # --- N>1 parity test ---
    @testset "N=10" begin
        n_samples = 10
        # Pick samples from class 1, target class 2
        idx_class1 = findall(==(1), y)[1:n_samples]
        x_test = X[:, idx_class1]
        targets = fill(2, n_samples)

        # CE.jl: generate one at a time (N=1 per CE object)
        cfs_ce = similar(X, 2, n_samples)
        for j in 1:n_samples
            gen_ce = ECCoGenerator(λ=λ, opt=opt)
            conv = Convergence.DecisionThresholdConvergence(;
                decision_threshold=decision_threshold, max_iter=maxiter
            )
            ce = generate_counterfactual(
                x_test[:, j:j], targets[j], data, M, gen_ce;
                initialization=:identity,
                convergence=conv,
            )
            cfs_ce[:, j] = ce.counterfactual[:, 1]
        end

        # Native: generate batched (N=10)
        gen_native = NativeGenerator(λ=λ, opt=opt)
        cfs_native, _, _, _ = generate_counterfactuals!(
            model, x_test, targets, data, gen_native;
            maxiter=maxiter, decision_threshold=decision_threshold,
            decay=decay, reg_strength=reg_strength,
        )

        # Compare per-sample
        # N>1 tolerance is wider than N=1 to accommodate accumulated Float32 vs
        # Float64 differences over 30 iterations of batched vs per-sample computation.
        @test size(cfs_ce) == size(cfs_native)
        for j in 1:n_samples
            @test cfs_ce[:, j] ≈ cfs_native[:, j] atol = 5e-2 rtol = 5e-2
        end
    end
end
