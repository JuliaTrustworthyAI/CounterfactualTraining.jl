using EnergySamplers: EnergySamplers

function implausibility(model, perturbed_input, samples, targets)
    implausibilities =
        EnergySamplers.energy_differential.(
            (model,), eachcol(perturbed_input), samples, targets
        )
    return implausibilities
end

function reg_loss(model, perturbed_input, samples, targets)
    return EnergySamplers.energy_penalty.(
        (model,), eachcol(perturbed_input), samples, targets
    )
end