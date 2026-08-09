# Boost Performance

This package was created for research purposes. Specifically, it was built to run the experiments in this IEEE SaTML 2026 paper: https://arxiv.org/abs/2601.16205. The code in ../src/ depends on CounterfactualExplanations.jl. It generates counterfactuals in parallel through MPI, since the experiments were run on many CPUs of a HPC. 

## Goal

The goal for this PR is to add a second branch to `src/` that uses CounterfactualExplanations.jl's components (e.g. `ECCoGenerator`, `generator_loss`, `search` utilities) directly in a GPU-compatible way, **without** allocating a full `CounterfactualExplanation` object per counterfactual. The key changes: 1) batched counterfactual generation on a single `D × N` matrix instead of iterating per sample; 2) GPU support (CUDA + AMDGPU); 3) removing the `CounterfactualExplanation` struct overhead. 



