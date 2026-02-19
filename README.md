

# CounterfactualTraining.jl

> [!NOTE]
> This is the version of the repository at the time of publication at SaTML. This version can be used to reproduce the results in the paper. It depends on the \#counterfactual-training branch of [CounterfactualExplanations.jl](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl/tree/counterfactual-training). To use the most up-to-date version of the code, switch to the main branch.

This repository contains all the code used to generate the experiments for the paper *Counterfactual Training: Teaching Models Plausible and Actionable Explanations*.

## Structure

The repository is structured like a standard Julia package with the main code base in `src/`, the documentation in `docs/` and the test suite in `test/`[1]. Everything relating specifically to the paper lives under `paper/`.

### Code Base in `src/`

The main code base for the CounterfactualTraining.jl package in `src/` is relatively simple and small. It implements the core functionality for counterfactual training. The code is commented with reference to relevant parts in the paper (look for `# ----- PAPER REF -----`).

### Code in `paper/`

The majority of code scripts live in `paper/experiments/`, which is a self-sufficient Julia project that depends on CounterfactualTraining.jl in the root folder. The code scripts in `paper/experiments` include Julia, bash and TOML scripts used to implement, run and configure the experiments specific to this paper. The project depends on another self-contained helper package that lives in `paper/experiments/CTExperiments/`. This helper package makes custom functions, structs and other definitions reusable in the project (`paper/experiments/`). All package dependencies are automatically handled in scripts using Julia’s package manager, Pkg.jl.

## Reproducibility

If you want to reproduce the results presented in our paper and the supplementary appendix, please see the [paper/experiments/README](./paper/experiments/README.md) for details.

[1] The latter two are both still incomplete
