using CounterfactualTraining
using Test
using Aqua

@testset "CounterfactualTraining.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(CounterfactualTraining)
    end
    # Write your tests here.
end
