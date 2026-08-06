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

    include("native/data.jl")
    include("native/convergence.jl")
    include("native/generator.jl")
    include("native/counterfactuals.jl")
    include("native/training.jl")

    export AbstractNativeGenerator, ECCoGenerator
    export NativeCFData
    export MaxIterConvergence, DecisionThresholdConvergence
end
export Native

end
