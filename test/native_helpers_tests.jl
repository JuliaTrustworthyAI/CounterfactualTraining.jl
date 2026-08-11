@testset "Native helpers (coverage)" begin
    # check_batched_convergence - maxiter branch
    probs = rand(Float32, 2, 5)
    @test Native.check_batched_convergence(probs, [1, 2, 1, 2, 1], 10, 10, 0.75f0) ==
        trues(5)

    # check_batched_convergence - threshold branch
    probs2 = Float32[
        0.9 0.5 0.9 0.5 0.9
        0.1 0.5 0.1 0.5 0.1
    ]
    conv = Native.check_batched_convergence(probs2, fill(1, 5), 1, 10, 0.75f0)
    @test conv == [true, false, true, false, true]

    # track_adversarial_examples! - p == Inf branch
    X_ae = zeros(Float32, 2, 3)
    Xp_ae = zeros(Float32, 2, 3)
    Xp_ae[:, 2] .= 1.0f0
    last_valid = copy(X_ae)
    Native.track_adversarial_examples!(last_valid, X_ae, Xp_ae, 0.5f0, Inf)
    @test last_valid[:, 1] == Xp_ae[:, 1]
    @test last_valid[:, 2] == X_ae[:, 2]
    @test last_valid[:, 3] == Xp_ae[:, 3]

    # track_adversarial_examples! - finite p branch
    last_valid2 = copy(X_ae)
    Native.track_adversarial_examples!(last_valid2, X_ae, Xp_ae, 0.5f0, 2)
    # With p=2 and epsilon=0.5: perturbation norms are [0, sqrt(2), 0] (L2 norm per column).
    # Column 1: norm 0 <= 0.5 → valid → last_valid = X′
    # Column 2: norm sqrt(2) ≈ 1.414 > 0.5 → invalid → last_valid stays X (zeros)
    # Column 3: norm 0 <= 0.5 → valid → last_valid = X′
    @test last_valid2[:, 1] == Xp_ae[:, 1]
    @test last_valid2[:, 2] == X_ae[:, 2]
    @test last_valid2[:, 3] == Xp_ae[:, 3]

    # generator_loss
    gen = NativeGenerator()
    model = Chain(Dense(3, 5, relu), Dense(5, 2))
    X_gl = randn(Float32, 3, 8)
    Xp_gl = copy(X_gl)
    targets_oh = Flux.onehotbatch(fill(2, 8), 1:2)
    loss_val = Native.generator_loss(
        gen, model, Xp_gl, X_gl, targets_oh, fill(2, 8), 1, 1.0f-3, 0.9f0, 10
    )
    @test isfinite(loss_val)
    gen_big = NativeGenerator(; λ=[1.0f0, 1.0f0])
    loss_big = Native.generator_loss(
        gen_big, model, Xp_gl, X_gl, targets_oh, fill(2, 8), 1, 1.0f-3, 0.9f0, 10
    )
    @test isfinite(loss_big)

    # find_neighbours - normal case
    X_nb = randn(Float32, 3, 6)
    y_nb = [1, 1, 2, 2, 1, 2]
    rng = MersenneTwister(42)
    nb = Native.find_neighbours(X_nb, y_nb, [2, 1], [1, 2]; rng=rng)
    @test size(nb) == (3, 2)
    @test all(isfinite, nb)

    # find_neighbours - empty candidates branch
    rng2 = MersenneTwister(42)
    nb2 = Native.find_neighbours(X_nb, y_nb, [3], [1, 2]; rng=rng2)
    @test size(nb2) == (3, 1)
    @test all(isfinite, nb2)

    # protect_immutable! (Native) - all directions + nothing
    neigh = ones(Float32, 4, 3)
    cfs = fill(2.0f0, 4, 3)
    mut = [:both, :none, :increase, :decrease]
    masks = Native.prepare_mutability_masks(mut, 4)
    Native.protect_immutable!(neigh, cfs, masks)
    @test all(neigh[1, :] .== 1.0f0)  # :both -> keep neighbour
    @test all(neigh[2, :] .== 2.0f0)  # :none -> use counterfactual
    @test all(neigh[3, :] .== 2.0f0)  # :increase -> max(cf=2,neigh=1)=2
    @test all(neigh[4, :] .== 1.0f0)  # :decrease -> min(cf=2,neigh=1)=1
    # mutability=nothing early return
    neigh2 = ones(Float32, 4, 3)
    Native.protect_immutable!(neigh2, cfs, nothing)
    @test all(neigh2 .== 1.0f0)

    # split_obs
    parts = Native.split_obs(1:10, 3)
    @test length(parts) >= 3
    @test sort(reduce(vcat, parts)) == collect(1:10)
    parts2 = Native.split_obs(1:5, 10)
    @test length(parts2) >= 5
    @test sort(reduce(vcat, parts2)) == collect(1:5)
end
