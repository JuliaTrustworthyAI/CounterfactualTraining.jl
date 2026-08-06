# Boost Performance

This package was created for research purposes. Specifically, it was built to run the experiments in this IEEE SaTML 2026 paper: https://arxiv.org/abs/2601.16205. The code in ../src/ depends on CounterfactualExplanations.jl. It generates counterfactuals in parallel through MPI, since the experiments were run on many CPUs of a HPC. 

## Goal

The goal for this PR is to add a second branch to ../src/ that does not depend on CounterfactualExplanations.jl to generate counterfactuals during training. Instead, it implements the counterfactual generation here from scratch with the following criteria: 1) enable training on GPU (including counterfactual generation); 2) make this as performant as possible. 



