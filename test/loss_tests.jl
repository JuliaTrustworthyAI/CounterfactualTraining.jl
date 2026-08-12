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

    # implausibility_and_reg_loss (combined — should match separate calls)
    impl_combined, reg_combined = CounterfactualTraining.implausibility_and_reg_loss(
        m, counterfactual, samples, targets
    )
    @test impl_combined ≈ impl rtol=1e-5
    @test reg_combined ≈ reg rtol=1e-5

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

@testset "Fused vs unfused CF forwards (BN-free)" begin
    # On a BN-free (Dense-only) model, fusing the three counterfactual forward
    # passes into one concatenated forward must produce identical loss values
    # and gradients to the unfused path.
    Random.seed!(42)
    m = Chain(Dense(8, 16, relu), Dense(16, 3))
    opt = Flux.Adam(1e-3)
    opt_state = Flux.setup(opt, m)

    n_cf = 5
    n_nb = 5
    n_ae = 5
    perturbed_input = randn(Float32, 8, n_cf)
    neighbours = randn(Float32, 8, n_nb)
    advexms = randn(Float32, 8, n_ae)
    targets_enc = Flux.onehotbatch([1, 2, 3, 1, 2], 1:3)
    factual_enc = Flux.onehotbatch([1, 1, 2, 2, 3], 1:3)

    # Unfused path
    implaus_u, regs_u = implausibility_and_reg_loss(
        m, perturbed_input, neighbours, targets_enc
    )
    adv_u = Flux.logitcrossentropy(m(advexms), factual_enc; agg=sum)

    # Fused path
    logits_all = m(cat(perturbed_input, neighbours, advexms; dims=2))
    logits_cf = @view(logits_all[:, 1:n_cf])
    logits_nb = @view(logits_all[:, n_cf+1:n_cf+n_nb])
    logits_ae = @view(logits_all[:, n_cf+n_nb+1:end])
    implaus_f, regs_f = implausibility_and_reg_loss_from_logits(
        logits_cf, logits_nb, targets_enc
    )
    adv_f = Flux.logitcrossentropy(logits_ae, factual_enc; agg=sum)

    @test implaus_u ≈ implaus_f rtol=1e-6
    @test regs_u ≈ regs_f rtol=1e-6
    @test adv_u ≈ adv_f rtol=1e-6

    # Gradients must match too
    loss_u = Flux.withgradient(m) do mm
        i, r = implausibility_and_reg_loss(mm, perturbed_input, neighbours, targets_enc)
        a = Flux.logitcrossentropy(mm(advexms), factual_enc; agg=sum)
        return sum(i) + sum(r) + a
    end
    loss_f = Flux.withgradient(m) do mm
        la = mm(cat(perturbed_input, neighbours, advexms; dims=2))
        lc = @view(la[:, 1:n_cf])
        ln = @view(la[:, n_cf+1:n_cf+n_nb])
        lae = @view(la[:, n_cf+n_nb+1:end])
        i, r = implausibility_and_reg_loss_from_logits(lc, ln, targets_enc)
        a = Flux.logitcrossentropy(lae, factual_enc; agg=sum)
        return sum(i) + sum(r) + a
    end
    @test loss_u[1] ≈ loss_f[1] rtol=1e-6
    # Compare gradient trees leaf-by-leaf (Flux 0.16 returns Tuple/NamedTuple trees).
    function grads_approx(a, b; rtol=1e-6)
        if a isa NamedTuple || a isa Tuple
            return all(grads_approx(ga, gb; rtol=rtol) for (ga, gb) in zip(a, b))
        elseif a === nothing && b === nothing
            return true
        else
            return isapprox(a, b; rtol=rtol)
        end
    end
    @test grads_approx(loss_u[2], loss_f[2])
end
