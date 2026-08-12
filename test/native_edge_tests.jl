@testset "generate_native! edge cases" begin
    X = randn(Float32, 10, 100)
    y = vcat(fill(1, 50), fill(2, 50))
    train_set = Flux.DataLoader((Float32.(X), Flux.onehotbatch(y, 1:2)); batchsize=32)
    model = Chain(Dense(10, 8, relu), Dense(8, 2))
    gen = NativeGenerator()

    # nsamples < N subsampling branch
    dl, pct_valid, _ = generate_native!(
        model, train_set, gen; nsamples=20, maxiter=5, verbose=0
    )
    @test length(dl) == length(train_set)
    @test 0 ≤ pct_valid ≤ 1

    # nsamples < length(train_set) warning+clamp branch
    dl2, pct_valid2, _ = generate_native!(
        model, train_set, gen; nsamples=2, maxiter=5, verbose=0
    )
    @test length(dl2) == length(train_set)
    @test 0 ≤ pct_valid2 ≤ 1
end

@testset "Native training edge cases" begin
    X = hcat(randn(Float32, 2, 100), randn(Float32, 2, 100) .+ 3.0f0)
    y = vcat(fill(1, 100), fill(2, 100))
    train_set = Flux.DataLoader((X, Flux.onehotbatch(y, 1:2)); batchsize=32)
    model = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_state = Flux.setup(Flux.Adam(1e-2), model)
    gen = NativeGenerator()
    obj = VanillaObjective(; needs_ce=true)

    # burnin > 0
    model_b, log_b = counterfactual_training(
        obj,
        model,
        gen,
        train_set,
        opt_state;
        nepochs=4,
        verbose=0,
        maxiter=10,
        burnin=0.5f0,
    )
    @test length(log_b) == 4
    @test log_b[1].implaus === nothing
    @test isfinite(log_b[end].implaus)

    # val_set
    val_set = Flux.DataLoader((X[:, 1:20], Flux.onehotbatch(y[1:20], 1:2)); batchsize=10)
    model_v = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_v = Flux.setup(Flux.Adam(1e-2), model_v)
    model_v, log_v = counterfactual_training(
        obj,
        model_v,
        gen,
        train_set,
        opt_v;
        nepochs=2,
        verbose=0,
        maxiter=10,
        burnin=0.0f0,
        val_set=val_set,
        accuracy_every=1,
    )
    @test isfinite(log_v[end].acc_val)

    # checkpoint_dir - save then load
    ckpt = mktempdir()
    model_c = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_c = Flux.setup(Flux.Adam(1e-2), model_c)
    model_c, log_c = counterfactual_training(
        obj,
        model_c,
        gen,
        train_set,
        opt_c;
        nepochs=2,
        verbose=0,
        maxiter=10,
        burnin=0.0f0,
        checkpoint_dir=ckpt,
    )
    @test isfile(joinpath(ckpt, "checkpoint.jld2"))

    # re-run to exercise checkpoint load path (training already completed)
    model_c2 = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_c2 = Flux.setup(Flux.Adam(1e-2), model_c2)
    model_c2, log_c2 = counterfactual_training(
        obj,
        model_c2,
        gen,
        train_set,
        opt_c2;
        nepochs=2,
        verbose=0,
        maxiter=10,
        burnin=0.0f0,
        checkpoint_dir=ckpt,
    )
    @test length(log_c2) == 0

    # verbose=0
    model_v0 = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_v0 = Flux.setup(Flux.Adam(1e-2), model_v0)
    counterfactual_training(
        obj, model_v0, gen, train_set, opt_v0; nepochs=1, verbose=0, maxiter=5
    )

    # verbose=3
    model_v3 = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_v3 = Flux.setup(Flux.Adam(1e-2), model_v3)
    counterfactual_training(
        obj, model_v3, gen, train_set, opt_v3; nepochs=1, verbose=3, maxiter=5
    )
end
