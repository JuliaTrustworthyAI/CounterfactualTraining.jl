

# Experiments

This is the folder that contains all code used to generate the results in the paper. You can also use it to reproduce the results.

## Getting Started

To get started activate the environment in the Julia REPL:

``` julia
# eval: false

using Pkg; Pkg.activate("paper/experiments")
```

Next, we load the `CTExperiments` package—a helper package that ships some additional functionality relevant only to the experiments run for the paper.

``` julia
using CTExperiments
```

## Running Experiments

The Julia functions that live directly under `paper/experiments/` can be used to run the experiments. The main scripts of interest are:

- `run_grid.jl` and `run_grid_sequentially.jl` which can be used to train model architectures on data according to specifications in configuration files (TOML).
- `run_evaluation_grid.jl` and `run_evaluation_grid_sequentially.jl` which can be used to evaluate previously run experiments.

The `_sequentially` suffix indicates that experiments are run sequentially and tasks underlying experiments are distributed across processes (as opposed to distributed experiments across processes).

### Configurations

To configure and keep track of hyperparameters for differenct experiments, we use TOML files. The configurations used for our various experiments can be found in `paper/experiments/configs/`. The following files are of particular relevance:

- `single.toml`: configures the main experiments that produced the main results without mutability constraints.
- `mutability_*.toml`: configures the main experiments that produced the main results with mutability constraints.

The remaining configuration files include those for extensive grid searches used for hyperparameter tuning.

### Command Line Usage

The Julia scripts introduced above look for certain environment variables pertaining to the configurations of experiments and evaluations, as well as paths to directories that will used to store results. Before running batch jobs, you therefore want to specify these variables as illustrated in the following example:

    export CONFIG=paper/experiments/configs/single.toml   # path to configuration file
    export OUTPUT_SUBDIR="final_run"                      # path to store outputs (root directory is /paper/experiments/output by default)

You can then run scripts directly from the command line like so:

    julia --project=$EXPERIMENT_DIR $EXPERIMENT_DIR/run_grid.jl --config=$CONFIG --data="lin_sep" --model="mlp"

Note that it is possible to provide extra arguments (here `--data` or `--model`) to overwrite specifications in the configuration file. To understand what arguments you can provide, we recommend familiarizing yourself with the code base in `paper/experiments/CTExperiments/`.

## Other Things

Some anecdotal or illustrative results were produced in simple Julia scripts or interactive notebooks. In fact, many of the Quarto notebooks (.qmd) in `paper/sections/other/` contain code cells that produce tables and charts. Note that many of these depend on experiment output being available that is not shipped with this repository and therefore first needs to be created. We will release the experiment outputs upon publication in a designated data repository, fit to handle large amounts of data.

### Figure 1

The chart in Figure 1 of the paper is produced in [paper/sections/other/constraints.qmd](paper/sections/other/constraints.qmd).

### Figure 2

The chart in Figure 2 of the paper can be reproduced using the following script: [paper/experiments/mnist_chart.jls](./paper/experiments/mnist_chart.jls).

### Figure 3

The chart in Figure 3 of the paper is produced in [paper/sections/other/main_results.qmd](paper/sections/other/main_results.qmd).
