module CounterfactualTraining

include("utils.jl")
include("loss.jl")
include("objectives.jl")
include("counterfactuals.jl")
export EnergyDifferentialObjective, AdversarialObjective, FullObjective, VanillaObjective
export implausibility, reg_loss, implausibility_and_reg_loss
export implausibility_and_reg_loss_from_logits
include("training.jl")
export counterfactual_training

module Native
    import ..CounterfactualTraining: counterfactual_training
    import ..CounterfactualTraining: AbstractObjective
    import ..CounterfactualTraining: implausibility, reg_loss, implausibility_and_reg_loss
    import ..CounterfactualTraining: implausibility_and_reg_loss_from_logits
    import ..CounterfactualTraining: infer_domain_constraints, unwrap
    import ..CounterfactualTraining: needs_counterfactuals
    import ..CounterfactualTraining: accuracy
    import ..CounterfactualTraining: VanillaObjective, EnergyDifferentialObjective
    import ..CounterfactualTraining: AdversarialObjective, FullObjective

    using CounterfactualExplanations
    import CounterfactualExplanations: polynomial_decay
    using Flux

    include("native/training.jl")

    export NativeGenerator, generate_counterfactuals!, generate_native!
end
export Native

end
