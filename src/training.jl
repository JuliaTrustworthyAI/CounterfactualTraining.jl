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

global _min_nce_ratio = 0.1

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
    parallelizer::Union{Nothing,TaijaParallel.AbstractParallelizer}=nothing,
    convergence=Convergence.MaxIterConvergence(),
    input_encoder=nothing,
    domain=nothing,
    mutability=nothing,
    verbose::Int=1,
    checkpoint_dir::Union{Nothing,String}=nothing,
    callback::Union{Nothing,Function}=nothing,
    kwrgs...,
)

    # Set up:
    burnin = Int(round(burnin * nepochs))
    nce = isnothing(nce) ? length(train_set) : nce
    nce_per_batch = Int(ceil(nce / length(train_set)))
    nce_batch_ratio = nce_per_batch / train_set.batchsize
    if nce_batch_ratio < 0.1
        @warn "The ratio of counterfactuals to training examples is less than $(_min_nce_ratio * 100)% ($(nce_batch_ratio * 100)%). Consider increasing  the `nce` parameter." maxlog =
            1
    end

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

    if verbose in [1, 2]
        p = Progress(nepochs - start_epoch; barglyphs=BarGlyphs("[=> ]"), color=:yellow)
    end

    for epoch in start_epoch:nepochs

        # Logs:
        losses = Float32[]
        implausibilities = Float32[]
        reg_losses = Float32[]
        validity_losses = Float32[]
        start = time()

        # Generate counterfactuals:
        if epoch > burnin && needs_counterfactuals(loss)
            counterfactual_dl, percent_valid = generate!(
                model,
                train_set,
                generator;
                nsamples=nce,
                nneighbours=nneighbours,
                convergence=convergence,
                parallelizer=parallelizer,
                input_encoder=input_encoder,
                domain=domain,
                mutability=mutability,
                verbose=verbose,
            )
        else
            counterfactual_dl = fill(ntuple(_ -> nothing, 4), length(train_set))
            percent_valid = nothing
        end

        # Backprop:
        for (i, batch) in enumerate(train_set)

            # Unpack:
            input, label = batch
            perturbed_input, targets_enc, neighbours, adversarial_targets = counterfactual_dl[i]

            val, grads = Flux.withgradient(model) do m

                # Compute predictions:
                logits = m(input)

                # Compute implausibility and regulatization:
                if !isnothing(perturbed_input)
                    implaus = implausibility(m, perturbed_input, neighbours, targets_enc)
                    regs = reg_loss(m, perturbed_input, neighbours, targets_enc)
                    # Validity loss (counterfactual):
                    yhat_ce = m(perturbed_input)
                    adversarial_loss = loss.class_loss(
                        yhat_ce, adversarial_targets
                    )
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

                return loss(logits, label, implaus, regs, adversarial_loss)
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

        if !isnothing(callback)
            counterfactuals = reduce(hcat, [x[1] for x in counterfactual_dl])
            callback(model, counterfactuals)
        end

        # Logging:
        time_taken = time() - start
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
            if !isnothing(percent_valid)
                msg_valid = "Valid counterfactuals: $(percent_valid * 100)%"
            else
                msg_valid = "n/a"
            end
        else
            implaus = nothing
            log_reg_loss = nothing
            log_adv_loss = nothing
            msg_imp = "n/a"
            msg_reg = "n/a"
            msg_adv = "n/a"
            msg_valid = "n/a"
        end

        push!(
            log,
            (;
                acc,
                acc_val,
                train_loss,
                implaus,
                log_reg_loss,
                log_adv_loss,
                time_taken,
                percent_valid,
            ),
        )

        # Checkpointing:
        if !isnothing(checkpoint_dir)
            jldsave(
                joinpath(checkpoint_dir, "checkpoint.jld2"); model, opt_state, epoch, log
            )
            previous_log = joinpath(checkpoint_dir, "checkpoint_$(epoch-1).md")
            isfile(previous_log) && rm(previous_log)
            fpath = joinpath(checkpoint_dir, "checkpoint_$(epoch).md")
            acc_plt = ""
            acc_val_plt = ""
            validity_plt = ""
            if verbose > 1 && epoch > burnin
                # Add plots:
                acc_plt = lineplot(
                    [_log[1] for _log in log]; xlabel="Epochs", ylabel="Accuracy"
                )
                if !isnothing(acc_val)
                    acc_val_plt = lineplot(
                        [_log[2] for _log in log];
                        xlabel="Epochs",
                        ylabel="Validation Accuracy",
                    )
                end
                if !isnothing(percent_valid)
                    validity_plt = lineplot(
                        [_log[8] for _log in log];
                        xlabel="Epochs",
                        ylabel="Valid Counterfatuals (%)",
                    )
                end
            end
            a = """
            Completed $epoch out of $nepochs epochs.

            ## Performance

            - *Accuracy*: $msg_acc
            - *Implausibility*: $msg_imp
            - *Regularization loss*: $msg_reg
            - *Adversarial loss*: $msg_adv
            - *Percent valid*: $msg_valid

            ## History

            $acc_plt 

            $acc_val_plt

            $validity_plt
            """
            open(fpath, "w") do file
                write(file, a)
            end
        end

        if verbose in [1, 2]
            next!(p)
        elseif verbose > 2
            @info "Iteration $epoch:"
            @info msg_acc_train
            @info msg_acc
            @info msg_imp
            @info msg_reg
            @info msg_adv
            @info msg_valid
        end
    end
    return model, log
end
