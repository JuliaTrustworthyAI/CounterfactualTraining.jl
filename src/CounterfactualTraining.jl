module CounterfactualTraining

include("utils.jl")
include("loss.jl")
include("objectives.jl")
include("counterfactuals.jl")
export EnergyDifferentialObjective, AdversarialObjective, FullObjective, VanillaObjective
export implausibility, reg_loss
include("training.jl")
export counterfactual_training

module Native
    import ..CounterfactualTraining: counterfactual_training
    import ..CounterfactualTraining: AbstractObjective
    import ..CounterfactualTraining: implausibility, reg_loss
    import ..CounterfactualTraining: infer_domain_constraints, unwrap
    import ..CounterfactualTraining: VanillaObjective, EnergyDifferentialObjective
    import ..CounterfactualTraining: AdversarialObjective, FullObjective

    using CounterfactualExplanations
    import CounterfactualExplanations: AbstractGenerator, ECCoGenerator
    using Flux

    include("native/training.jl")
end
export Native

end
