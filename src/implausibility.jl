using EnergySamplers: EnergySamplers

function implausibility(model, perturbed_input, samples, target)
    implausibilities = zeros(length(samples))
    for (i, input) in enumerate(eachcol(perturbed_input))
        implaus = EnergySamplers.energy_differential.(model, input, samples[i], target)
        implausibilities[i] = implaus
    end
    return implausibilities
end