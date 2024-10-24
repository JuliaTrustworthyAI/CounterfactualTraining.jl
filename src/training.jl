using ChainRulesCore: ChainRulesCore
using CounterfactualExplanations
using CounterfactualExplanations: Convergence
using CounterfactualExplanations: Models
using CounterfactualExplanations.Evaluation: plausibility
using TaijaParallel
using Flux

function generate!(
    input, model, data, generator;
    convergence=Convergence.MaxIterConvergence(),
    parallelizer=nothing,
    input_encoder=nothing,
    verbose=true,
    domain=nothing,
)

    # Set up:
    counterfactual_data = CounterfactualData(
        unwrap(data)...,
        domain=domain,
        input_encoder=input_encoder,
    )
    M = Models.Model(model, Models.FluxNN(); likelihood=:classification_multi)

    # Generate counterfactuals:
    nbatch = size(input, 2)
    targets = rand(counterfactual_data.y_levels, nbatch)                # randomly generate targets
    xs = [x[:, :] for x in eachcol(input)]                              # factuals

    !verbose || println("Generating counterfactuals ...")
    ces = TaijaParallel.parallelize(
        parallelizer,
        CounterfactualExplanations.generate_counterfactual,
        xs,
        targets,
        counterfactual_data,
        M,
        generator;
        convergence=convergence,
        verbose=false,
    )
    input = hcat(CounterfactualExplanations.counterfactual.(ces)...)                               # counterfactual inputs

    return input, ces, targets
    
end

function counterfactual_training(
    loss,
    model,
    generator,
    train_set,
    opt_state;
    nepochs=100,
    burnin=0.5,
    nce=nothing,
    parallelizer=nothing,
    convergence=Convergence.MaxIterConvergence(),
    input_encoder=nothing,
    domain=nothing, 
    verbose=false,
    kwrgs...
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

        for (i, batch) in enumerate(train_set)
            input, label = batch

            if epoch > burnin

                # Choose subset of inputs:
                nbatch = size(input, 2)
                if nce != nbatch
                    idx = rand(1:nbatch, nce)
                    chosen_input = input[:, idx]
                else
                    chosen_input = input
                end

                # Generate counterfactuals for chosen inputs:
                perturbed_input, ces, targets = generate!(
                    chosen_input,
                    model,
                    train_set,
                    generator;
                    convergence=convergence,
                    parallelizer=parallelizer,
                    input_encoder=input_encoder,
                    domain=domain,
                    verbose=verbose,
                )

                # Get neighbour in target class:
                samples = (X -> X[:,1])([
                    CounterfactualExplanations.find_potential_neighbours(ce, 10) for
                    ce in ces
                ])

                # Encoded targets:
                targets_enc = hcat((x -> x.target_encoded).(ces)...)
            else
                perturbed_input = nothing
                targets_enc = nothing
                ces = nothing
                implaus = [0.0f0]
                regs = [0.0f0]
                validity_loss = 0.0f0
            end

            val, grads = Flux.withgradient(model) do m

                # Compute predictions:
                logits = m(input)

                # Compute implausibility and regulatization:
                if !isnothing(perturbed_input)
                    implaus = implausibility(m, perturbed_input, samples, targets)
                    regs = reg_loss(m, perturbed_input, samples, targets)
                    # Validity loss (counterfactual):
                    yhat_ce = m(perturbed_input)
                    # validity_loss = Flux.Losses.logitcrossentropy(yhat_ce, targets_enc)
                    validity_loss = 0.0f0
                end

                # Save the implausibilities from the forward pass:
                ChainRulesCore.ignore_derivatives() do
                    push!(implausibilities, sum(implaus) / length(implaus))
                    push!(reg_losses, sum(regs) / length(regs))
                    push!(validity_losses, validity_loss)
                end

                return loss(logits, label, implaus, regs, validity_loss)
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
            @info "Average implausibility in $epoch/$nepochs: $implaus"
            @info "Average reg loss in $epoch/$nepochs: $(sum(reg_losses)/length(reg_losses))"
            @info "Average validity loss in $epoch/$nepochs: $(sum(validity_losses)/length(validity_losses))"
        end
    end
    return model, my_log
end