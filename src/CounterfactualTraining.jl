module CounterfactualTraining

include("utils.jl")
include("implausibility.jl")
include("objectives.jl")
include("counterfactuals.jl")
export EnergyDifferentialObjective, AdversarialObjective, FullObjective
export implausibility, _implausibility
include("training.jl")

end
