using ChainRulesCore: ChainRulesCore
using CounterfactualExplanations
using CounterfactualExplanations: Convergence
using CounterfactualExplanations: Models
using CounterfactualExplanations.Evaluation: plausibility
using TaijaParallel
using Flux

function counterfactual_training(
    loss::AbstractObjective,
    model,
    generator,
    train_set,
    opt_state;
    nepochs=100,
    burnin=0.5,
    nce=nothing,
    parallelizer::TaijaParallel.AbstractParallelizer=nothing,
    convergence=Convergence.MaxIterConvergence(),
    input_encoder=nothing,
    domain=nothing,
    verbose::Int=2,
    kwrgs...,
)

    # Set up:
    burnin = Int(round(burnin * nepochs))
    nce = isnothing(nce) ? train_set.batchsize : nce

    my_log = []
    for epoch in 1:nepochs

        # Logs:
        losses = Float32[]
        implausibilities = Float32[]
        reg_losses = Float32[]
        validity_losses = Float32[]

        # Generate counterfactuals outside of minibatching:
        if epoch > burnin
            perturbed_set = generate!(
                model,
                train_set,
                generator;
                nsamples=nce,
                convergence=convergence,
                parallelizer=parallelizer,
                input_encoder=input_encoder,
                domain=domain,
                verbose=verbose,
            )
        else
            dummy_data = fill(nothing, length(train_set))
            perturbed_set = Flux.DataLoader(
                (dummy_data, dummy_data, dummy_data, dummy_data); batchsize=1
            )
        end

        # Joint data loader:
        joint_loader = zip(train_set, perturbed_set)

        for (i, (batch, perturbed_batch)) in enumerate(joint_loader)

            # Unpack:
            input, label = batch
            perturbed_input, targets, targets_enc, neighbours = perturbed_batch
            neighbours = typeof(neighbours) <: AbstractVector ? neighbours : [neighbours]

            val, grads = Flux.withgradient(model) do m

                # Compute predictions:
                logits = m(input)

                # Compute implausibility and regulatization:
                if !isnothing(perturbed_input)
                    implaus = implausibility(m, perturbed_input, neighbours, targets)
                    regs = reg_loss(m, perturbed_input, neighbours, targets)
                    # Validity loss (counterfactual):
                    yhat_ce = m(perturbed_input)
                    adversarial_loss = Flux.Losses.logitcrossentropy(yhat_ce, targets_enc)
                else
                    implaus = [0.0f0]
                    regs = [0.0f0]
                    adversarial_loss = 0.0f0
                end

                # Save the implausibilities from the forward pass:
                ChainRulesCore.ignore_derivatives() do
                    push!(implausibilities, sum(implaus) / length(implaus))
                    push!(reg_losses, sum(regs) / length(regs))
                    push!(validity_losses, adversarial_loss)
                end

                return loss(
                    logits,
                    label;
                    energy_differential=implaus,
                    regularization=regs,
                    adversarial_loss=adversarial_loss,
                )
            end

            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(losses, val)

            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end

            Flux.update!(opt_state, model, grads[1])
        end

        # Compute some accuracy, and save details as a NamedTuple
        acc = accuracy(model, train_set)
        @info "Accuracy in epoch $epoch/$nepochs: $acc"
        push!(my_log, (; acc, losses, implausibilities, reg_losses, validity_losses))

        if epoch > burnin
            implaus = sum(implausibilities) / length(implausibilities)
            @info "Average energy differential in $epoch/$nepochs: $implaus"
            @info "Average energy regularization in $epoch/$nepochs: $(sum(reg_losses)/length(reg_losses))"
            @info "Average adversarial loss in $epoch/$nepochs: $(sum(validity_losses)/length(validity_losses))"
        end
    end
    return model, my_log
end
