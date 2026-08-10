@testset "Loss functions" begin
    m = Chain(Dense(3, 2))
    N = 5
    C = 2
    targets = zeros(Float32, C, N)
    for i in 1:N
        targets[rand(1:C), i] = 1.0f0
    end
    samples = randn(Float32, 3, N)
    counterfactual = randn(Float32, 3, N)

    # implausibility
    impl = implausibility(m, counterfactual, samples, targets)
    @test length(impl) == N
    @test all(isfinite, impl)

    # reg_loss
    reg = reg_loss(m, counterfactual, samples, targets)
    @test length(reg) == N
    @test all(>=(0.0f0), reg)

    # adv_loss - branch with valid adversarial examples
    perturbations_small = zeros(Float32, 3, N)
    adv1 = CounterfactualTraining.adv_loss(m, counterfactual, perturbations_small, targets)
    @test isfinite(adv1)

    # adv_loss - branch with no valid adversarial examples
    perturbations_large = fill(10.0f0, 3, N)
    adv2 = CounterfactualTraining.adv_loss(
        m, counterfactual, perturbations_large, targets; epsilon=0.5f0
    )
    @test adv2 == 0.0f0

    # NormBound struct
    nb = CounterfactualTraining.NormBound(epsilon=0.3, p=Inf)
    @test nb.epsilon == 0.3
    @test nb.p == Inf
    nb2 = CounterfactualTraining.NormBound(; epsilon=1.0, p=2)
    @test nb2.epsilon == 1.0
    @test nb2.p == 2

    # NormBound callable
    @test nb(zeros(Float32, 3)) == true
    @test nb(fill(10.0f0, 3)) == false

    # isadvexm
    @test CounterfactualTraining.isadvexm(zeros(Float32, 3), 0.5, Inf) == true
    @test CounterfactualTraining.isadvexm(ones(Float32, 3), 0.5, Inf) == false
    @test CounterfactualTraining.isadvexm(zeros(Float32, 3), 0.5, 2) == true
    @test CounterfactualTraining.isadvexm(fill(1.0f0, 3), 0.5, 2) == false

    # get/set global AE criterium
    original = CounterfactualTraining.get_global_ae_criterium()
    custom = CounterfactualTraining.NormBound(epsilon=1.0, p=2)
    CounterfactualTraining.set_global_ae_criterium(custom)
    @test CounterfactualTraining.get_global_ae_criterium() === custom
    CounterfactualTraining.set_global_ae_criterium(original)
end
