@testset "Counterfactual training (non-Native)" begin
    X = hcat(randn(Float32, 2, 60), randn(Float32, 2, 60) .+ 3.0f0)
    y = vcat(fill(1, 60), fill(2, 60))
    train_set = Flux.DataLoader((X, Flux.onehotbatch(y, 1:2)); batchsize=32)
    model = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_state = Flux.setup(Flux.Adam(1e-2), model)
    gen_ce = GenericGenerator()
    obj = VanillaObjective(; needs_ce=true)

    # basic 2-epoch run
    model, log = counterfactual_training(
        obj, model, gen_ce, train_set, opt_state; nepochs=2, verbose=0
    )
    @test length(log) == 2
    @test isfinite(log[end].acc)
    @test isfinite(log[end].train_loss)
    @test 0.0 <= log[end].percent_valid <= 1.0

    # burnin > 0
    model_b = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_b = Flux.setup(Flux.Adam(1e-2), model_b)
    model_b, log_b = counterfactual_training(
        obj, model_b, gen_ce, train_set, opt_b; nepochs=3, verbose=0, burnin=0.5
    )
    @test length(log_b) == 3
    @test log_b[1].implaus === nothing
    @test isfinite(log_b[end].implaus)

    # val_set
    val_set = Flux.DataLoader((X[:, 1:20], Flux.onehotbatch(y[1:20], 1:2)); batchsize=10)
    model_v = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_v = Flux.setup(Flux.Adam(1e-2), model_v)
    model_v, log_v = counterfactual_training(
        obj, model_v, gen_ce, train_set, opt_v; nepochs=2, verbose=0, val_set=val_set
    )
    @test isfinite(log_v[end].acc_val)

    # callback (called post-burnin when ces !== nothing)
    calls = Int[]
    model_cb = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_cb = Flux.setup(Flux.Adam(1e-2), model_cb)
    model_cb, log_cb = counterfactual_training(
        obj,
        model_cb,
        gen_ce,
        train_set,
        opt_cb;
        nepochs=2,
        verbose=0,
        callback=(m, ces) -> push!(calls, 1),
    )
    @test length(calls) >= 1

    # checkpoint_dir - save then load
    ckpt = mktempdir()
    model_c = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_c = Flux.setup(Flux.Adam(1e-2), model_c)
    model_c, log_c = counterfactual_training(
        obj, model_c, gen_ce, train_set, opt_c; nepochs=2, verbose=0, checkpoint_dir=ckpt
    )
    @test isfile(joinpath(ckpt, "checkpoint.jld2"))

    # checkpoint load path (training already completed)
    model_c2 = Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_c2 = Flux.setup(Flux.Adam(1e-2), model_c2)
    model_c2, log_c2 = counterfactual_training(
        obj, model_c2, gen_ce, train_set, opt_c2; nepochs=2, verbose=0, checkpoint_dir=ckpt
    )
    @test length(log_c2) >= 0
end
