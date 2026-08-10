@testset "Utils" begin
    # unwrap
    X = randn(Float32, 3, 20)
    y = vcat(fill(1, 10), fill(2, 10))
    train_set = Flux.DataLoader((X, Flux.onehotbatch(y, 1:2)); batchsize=10)
    X2, ycold = CounterfactualTraining.unwrap(train_set)
    @test size(X2) == (3, 20)
    @test sort(unique(ycold)) == [1, 2]

    # unwrap with labels (BUG: utils.jl:23 - ycold is Vector{Int}, can't hold Strings)
    @test_broken begin
        X3, ycold_labeled = CounterfactualTraining.unwrap(train_set; labels=["a", "b"])
        sort(unique(ycold_labeled)) == ["a", "b"]
    end

    # accuracy
    X_acc = hcat(randn(Float32, 2, 50), randn(Float32, 2, 50) .+ 3.0f0)
    y_acc = vcat(fill(1, 50), fill(2, 50))
    train_set_acc = Flux.DataLoader(
        (X_acc, Flux.onehotbatch(y_acc, 1:2)); batchsize=25
    )
    model_acc = Chain(Dense(2, 8, relu), Dense(8, 2))
    acc = CounterfactualTraining.accuracy(model_acc, train_set_acc)
    @test 0.0 <= acc <= 1.0

    # infer_domain_constraints
    X_dom = randn(Float32, 4, 100)
    bounds = CounterfactualTraining.infer_domain_constraints(X_dom)
    @test length(bounds) == 4
    @test all(b -> b isa Tuple, bounds)
    for i in 1:4
        lb, ub = bounds[i]
        @test lb <= ub
        @test lb <= minimum(X_dom[i, :])
        @test ub >= maximum(X_dom[i, :])
    end
end
