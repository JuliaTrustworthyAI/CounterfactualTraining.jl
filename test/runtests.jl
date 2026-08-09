using CounterfactualTraining
using CounterfactualTraining.Native
using Test
using Aqua
using CounterfactualExplanations
using Flux
using LinearAlgebra

@testset "CounterfactualTraining.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(CounterfactualTraining)
    end

    @testset "Native data structures and helpers" begin
        # Test NativeGenerator construction
        gen = NativeGenerator()
        @test gen.λ == [0.1f0, 1.0f0]
        @test gen.opt isa Flux.Descent

        gen_custom = NativeGenerator(λ=[0.5f0, 2.0f0])
        @test gen_custom.λ == [0.5f0, 2.0f0]

        # Test batched_energy
        X = randn(Float32, 3, 10)
        model = Chain(Dense(3, 5, relu), Dense(5, 2))
        targets = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        energies = Native.batched_energy(model, X, targets)
        @test length(energies) == 10
        @test all(isfinite, energies)

        # Test batched_apply_mutability!
        mutability = [:increase, :decrease, :both, :none]
        ΔX = ones(4, 5)
        ΔX[1, 2] = -1.0  # negative in :increase row → should be zeroed
        ΔX[2, 1] = 1.0   # positive in :decrease row → should be zeroed
        Native.batched_apply_mutability!(ΔX, mutability)

        # :none → zeros entire row
        @test all(ΔX[4, :] .== 0.0f0)
        # :increase → negatives zeroed, positives kept
        @test ΔX[1, 1] == 1.0   # positive kept
        @test ΔX[1, 2] == 0.0   # negative zeroed
        # :decrease → positives zeroed, negatives kept
        @test ΔX[2, 1] == 0.0   # positive zeroed
        @test ΔX[2, 2] == 1.0   # positive (was 1.0, but :decrease zeros positives)
        @test ΔX[2, 3] == 1.0   # also zeroed
        # :both → no change
        @test all(ΔX[3, :] .== 1.0)

        # Test batched_apply_domain_constraints!
        X_data = randn(Float32, 3, 50)
        y_data = vcat(fill(1, 25), fill(2, 25))
        data = CounterfactualData(X_data, y_data; domain=[(-2.0f0, 2.0f0), (-2.0f0, 2.0f0), (-2.0f0, 2.0f0)])
        X_clamped = copy(X_data)
        X_clamped[1, 1] = 100.0f0  # out of bounds
        Native.batched_apply_domain_constraints!(X_clamped, data)
        @test X_clamped[1, 1] <= 2.0f0  # clamped to upper bound
        @test all(X_clamped[1, :] .<= 2.0f0)
        @test all(X_clamped[1, :] .>= -2.0f0)
    end

    @testset "Native generator" begin
        # Build a simple 2-class problem
        X = hcat(randn(Float32, 2, 50), randn(Float32, 2, 50) .+ 3.0f0)
        y = vcat(fill(1, 50), fill(2, 50))
        model = Chain(Dense(2, 4, relu), Dense(4, 2))
        gen = NativeGenerator()
        data = CounterfactualData(X, y)

        # Generate counterfactuals
        X_test = X[:, 1:20]
        targets = fill(2, 20)
        cfs, advexms, converged, maxiter_used = generate_counterfactuals!(
            model, X_test, targets, data, gen; maxiter=10
        )

        # Check outputs
        @test size(cfs) == (2, 20)
        @test size(advexms) == (2, 20)
        @test length(converged) == 20
        @test all(isfinite, cfs)
        @test maxiter_used == 10
    end

    @testset "generate_native!" begin
        # Build a simple dataset
        X = randn(Float32, 10, 100)
        y = vcat(fill(1, 50), fill(2, 50))

        # Create a DataLoader with batchsize=32
        train_set = Flux.DataLoader((Float32.(X), Flux.onehotbatch(y, 1:2)); batchsize=32)

        # Create a simple model
        model = Chain(Dense(10, 8, relu), Dense(8, 2))

        gen = NativeGenerator()

        # Generate native counterfactuals
        dl, pct_valid, _ = generate_native!(
            model, train_set, gen; nsamples=32, maxiter=10, verbose=0
        )

        # Check outputs
        @test length(dl) == length(train_set)
        @test 0 ≤ pct_valid ≤ 1

        # Check each batch is a 5-tuple
        for i in eachindex(dl)
            batch = dl[i]
            @test length(batch) == 5
            counterfactuals, advexms, targets_enc, neighbours, factual_enc = batch
            @test size(counterfactuals)[1] == 10  # D features
        end
    end
end
