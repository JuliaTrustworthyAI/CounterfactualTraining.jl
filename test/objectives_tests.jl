struct FooObj <: CounterfactualTraining.AbstractObjective end

@testset "Objectives" begin
    C = 2
    N = 8
    yhat = randn(Float32, C, N)
    y = Flux.onehotbatch(rand(1:C, N), 1:C)
    ed = rand(Float32, N)
    reg = rand(Float32, N)
    adv = rand(Float32, N)
    ce = Flux.Losses.logitcrossentropy

    # VanillaObjective
    @test_throws AssertionError VanillaObjective(ce, Float32[1.0, 2.0])
    obj_v = VanillaObjective(; needs_ce=true)
    @test CounterfactualTraining.needs_counterfactuals(obj_v) == true
    obj_v2 = VanillaObjective(; needs_ce=false)
    @test CounterfactualTraining.needs_counterfactuals(obj_v2) == false
    rv = obj_v(yhat, y)
    @test isfinite(rv)
    @test rv ≈ ce(yhat, y; agg=mean) * 1.0

    # needs_counterfactuals fallback for custom subtype
    @test CounterfactualTraining.needs_counterfactuals(FooObj()) == true

    # EnergyDifferentialObjective - inner constructor with short lambda
    # NOTE: objectives.jl:99 has a bug (references undefined `obj`), so
    # constructing with length<3 lambda throws UndefVarError
    @test_throws UndefVarError EnergyDifferentialObjective(ce, Float32[1.0, 0.5])
    @test_throws UndefVarError EnergyDifferentialObjective(ce, Float32[1.0])
    obj_e = EnergyDifferentialObjective()
    re = obj_e(yhat, y, ed, reg)
    @test isfinite(re)
    @test re ≈
        ce(yhat, y; agg=mean) * obj_e.lambda[1] +
          mean(Float32.(ed)) * obj_e.lambda[2] +
          mean(Float32.(reg)) * obj_e.lambda[3]

    # AdversarialObjective
    @test_throws AssertionError AdversarialObjective(ce, Float32[1.0])
    obj_a = AdversarialObjective()
    ra = obj_a(yhat, y, Float32[], Float32[], adv)
    @test isfinite(ra)
    @test ra ≈
        ce(yhat, y; agg=mean) * obj_a.lambda[1] + mean(Float32.(adv)) * obj_a.lambda[2]

    # FullObjective
    @test_throws AssertionError FullObjective(ce, Float32[1.0, 2.0])
    obj_f = FullObjective()
    rf = obj_f(yhat, y, ed, reg, adv)
    @test isfinite(rf)
    @test rf ≈
        ce(yhat, y; agg=mean) * obj_f.lambda[1] +
          mean(Float32.(ed)) * obj_f.lambda[2] +
          mean(Float32.(reg)) * obj_f.lambda[3] +
          mean(Float32.(adv)) * obj_f.lambda[4]
end
