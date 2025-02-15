module CounterfactualTraining

include("utils.jl")
include("loss.jl")
include("objectives.jl")
include("counterfactuals.jl")
export EnergyDifferentialObjective, AdversarialObjective, FullObjective, VanillaObjective
export implausibility, reg_loss
include("training.jl")
export counterfactual_training

end
