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
    transformer=nothing,
    verbose=true,
)

    # Set up:
    counterfactual_data = CounterfactualData(unwrap(data)...)
    if !isnothing(transformer)
        counterfactual_data.input_encoder = transformer
    end
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

    return input, ces
    
end

function counterfactual_training(
    loss,
    model,
    generator,
    train_set,
    opt_state;
    nepochs=100,
    burnin=0.5,
    parallelizer=nothing,
    convergence=Convergence.MaxIterConvergence(),
    transformer=nothing,
    verbose=false,
)
    burnin = Int(round(burnin * nepochs))

    my_log = []
    for epoch in 1:nepochs
        losses = Float32[]
        implausibilities = Float32[]
        for (i, batch) in enumerate(train_set)
            input, label = batch

            if epoch > burnin
                # Generate counterfactuals:
                perturbed_input, ces = generate!(
                    input,
                    model,
                    train_set,
                    generator;
                    convergence=convergence,
                    parallelizer=parallelizer,
                    transformer=transformer,
                    verbose=verbose,
                )

                # Implausibility and validity:
                println("Evaluating counterfactuals for batch $i/$(length(train_set))")
                implaus = (x -> .-x)(
                    vec(
                        stack(
                            stack(
                                TaijaParallel.parallelize(
                                    parallelizer,
                                    CounterfactualExplanations.Evaluation.evaluate,
                                    ces;
                                    measure=plausibility,
                                    verbose=false,
                                ),
                            ),
                        ),
                    ),
                )
                push!(implausibilities, sum(implaus) / length(implaus))
            else
                implaus = [0.0f0]
            end

            val, grads = Flux.withgradient(model) do m
                # Any code inside here is differentiated.
                # Evaluation of the model and loss must be inside!
                logits = m(input)
                loss(logits, label, implaus)
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
        push!(my_log, (; acc, losses))

        if epoch > burnin
            implaus = sum(implausibilities) / length(implausibilities)
            @info "Average implausibility in $epoch/$nepochs: $implaus"
        end
    end
    return model
end