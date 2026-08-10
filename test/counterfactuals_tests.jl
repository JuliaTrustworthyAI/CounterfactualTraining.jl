@testset "Counterfactual generation (non-Native)" begin
    # protect_immutable! (non-Native) - operates on Vector{Matrix}
    # mutability is per-feature-row: [:both, :none, :increase, :decrease]
    samples = [rand(Float32, 4, 1) for _ in 1:3]
    cfs = [fill(2.0f0, 4, 1) for _ in 1:3]
    mut = [:both, :none, :increase, :decrease]
    CounterfactualTraining.protect_immutable!(samples, cfs, mut)
    for i in 1:3
        @test all(samples[i][1, :] .>= 0.0f0)   # :both -> keep sample (random)
        @test all(samples[i][2, :] .== 2.0f0)   # :none -> use cf
        @test all(samples[i][3, :] .>= 2.0f0)   # :increase -> max(cf=2, s) >= 2
        @test all(samples[i][4, :] .<= 2.0f0)   # :decrease -> min(cf=2, s) <= 2
    end

    # mutability=nothing early return
    samples2 = [rand(Float32, 4, 1) for _ in 1:3]
    original = copy(samples2[1])
    CounterfactualTraining.protect_immutable!(samples2, cfs, nothing)
    @test samples2[1] == original

    # isvalid
    X = hcat(randn(Float32, 2, 30), randn(Float32, 2, 30) .+ 3.0f0)
    y = vcat(fill(1, 30), fill(2, 30))
    train_set = Flux.DataLoader((X, Flux.onehotbatch(y, 1:2)); batchsize=32)
    model = Chain(Dense(2, 4, relu), Dense(4, 2))
    gen_ce = GenericGenerator()
    dl, pct, ces = CounterfactualTraining.generate!(model, train_set, gen_ce; nsamples=5)
    @test length(dl) == length(train_set)
    @test 0.0 <= pct <= 1.0
    data = CounterfactualData(X, y)
    v = CounterfactualTraining.isvalid(ces[1], model, data)
    @test v isa Bool
end
