using ChainRulesCore: ChainRulesCore
using CounterfactualExplanations
using CounterfactualExplanations: Convergence
using CounterfactualExplanations: Models
using CounterfactualExplanations.Evaluation: plausibility
using Flux
using JLD2
using ProgressMeter
using TaijaParallel
using UnicodePlots

function counterfactual_training(
    loss::AbstractObjective,
    model,
    generator,
    train_set,
    opt_state;
    val_set=nothing,
    nepochs=100,
    burnin=0.0,
    nce::Union{Nothing,Int}=nothing,
    nneighbours::Int=100,
    parallelizer::TaijaParallel.AbstractParallelizer=nothing,
    convergence=Convergence.MaxIterConvergence(),
    input_encoder=nothing,
    domain=nothing,
    verbose::Int=1,
    checkpoint_dir::Union{Nothing,String}=nothing,
    kwrgs...,
)

    # Set up:
    burnin = Int(round(burnin * nepochs))
    nce = isnothing(nce) ? train_set.batchsize : nce

    # Initialize model:
    log = []
    start_epoch = 1
    if !isnothing(checkpoint_dir) && isfile(joinpath(checkpoint_dir, "checkpoint.jld2"))
        @info "Found checkpoint file in $checkpoint_dir. Loading..."
        try
            model, opt_state, epoch, log = JLD2.load(
                joinpath(checkpoint_dir, "checkpoint.jld2"),
                "model",
                "opt_state",
                "epoch",
                "log",
            )
            start_epoch = epoch + 1
            if start_epoch <= nepochs
                @info "Resuming training from epoch $start_epoch."
            else
                @info "Already completed 100% of training. Skipping..."
            end
        catch
            @warn "Could not load checkpoint. Starting training from scratch."
        end
    end

    p = Progress(nepochs; barglyphs=BarGlyphs("[=> ]"), color=:yellow)

    for epoch in start_epoch:nepochs

        # Logs:
        losses = Float32[]
        implausibilities = Float32[]
        reg_losses = Float32[]
        validity_losses = Float32[]

        # Generate counterfactuals:
        if epoch > burnin
            counterfactual_dl = generate!(
                model,
                train_set,
                generator;
                nsamples=nce,
                nneighbours=nneighbours,
                convergence=convergence,
                parallelizer=parallelizer,
                input_encoder=input_encoder,
                domain=domain,
                verbose=verbose,
            )
        else
            counterfactual_dl = fill(ntuple(_ -> nothing, 4), length(train_set))
        end

        # Backprop:
        for (i, batch) in enumerate(train_set)

            # Unpack:
            input, label = batch
            perturbed_input, target_indices, targets_enc, neighbours = counterfactual_dl[i]
            neighbours = typeof(neighbours) <: AbstractVector ? neighbours : [neighbours]

            val, grads = Flux.withgradient(model) do m

                # Compute predictions:
                logits = m(input)

                # Compute implausibility and regulatization:
                if !isnothing(perturbed_input)
                    implaus = implausibility(m, perturbed_input, neighbours, target_indices)
                    regs = reg_loss(m, perturbed_input, neighbours, target_indices)
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
                    label,
                    implaus,
                    regs,
                    adversarial_loss,
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

        # Logging:
        acc = accuracy(model, train_set)
        acc_val = isnothing(val_set) ? nothing : accuracy(model, val_set)
        train_loss = sum(losses) / length(losses)
        msg_acc = "Training accuracy in epoch $epoch/$nepochs: $acc"
        if !isnothing(val_set)
            msg_acc_train = msg_acc
            msg_acc = "Validation accuracy in epoch $epoch/$nepochs: $acc_val"
        end

        if epoch > burnin
            implaus = sum(implausibilities) / length(implausibilities)
            log_reg_loss = sum(reg_losses) / length(reg_losses)
            log_adv_loss = sum(validity_losses) / length(validity_losses)
            msg_imp = "Average energy differential in $epoch/$nepochs: $implaus"
            msg_reg = "Average energy regularization in $epoch/$nepochs: $log_reg_loss"
            msg_adv = "Average adversarial loss in $epoch/$nepochs: $log_adv_loss"
        else
            implaus = nothing
            log_reg_loss = nothing
            log_adv_loss = nothing
            msg_imp = "n/a"
            msg_reg = "n/a"
            msg_adv = "n/a"
        end

        push!(log, (; acc, acc_val, train_loss, implaus, log_reg_loss, log_adv_loss))

        # Checkpointing:
        if !isnothing(checkpoint_dir)
            jldsave(
                joinpath(checkpoint_dir, "checkpoint.jld2"); model, opt_state, epoch, log
            )
            previous_log = joinpath(checkpoint_dir, "checkpoint_$(epoch-1).md")
            isfile(previous_log) && rm(previous_log)
            fpath = joinpath(checkpoint_dir, "checkpoint_$(epoch).md")
            acc_plt = lineplot(
                [_log[1] for _log in log]; xlabel="Epochs", ylabel="Accuracy"
            )
            acc_val_plt = if isnothing(acc_val)
                ""
            else
                lineplot(
                    [_log[2] for _log in log];
                    xlabel="Epochs",
                    ylabel="Validation Accuracy",
                )
            end
            a = """
            Completed $epoch out of $nepochs epochs.

            ## Performance

            - *Accuracy*: $msg_acc
            - *Implausibility*: $msg_imp
            - *Regularization loss*: $msg_reg
            - *Adversarial loss*: $msg_adv

            ## History

            $acc_plt 

            $acc_val_plt
            """
            open(fpath, "w") do file
                write(file, a)
            end
        end

        if verbose == 1
            next!(p; showvalues = [("Validation accuracy", acc_val), ("Implausibility", implaus)])
        elseif verbose > 1
            @info msg_acc_train
            @info msg_acc
            @info msg_imp
            @info msg_reg
            @info msg_adv
        end
    end
    return model, log
end
