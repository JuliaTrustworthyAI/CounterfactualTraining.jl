

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

## Command Line Usage

    sbatch --ntasks=$(determine_resources) --cpus-per-task=$NTHREADS myfile.sh

    sbatch --ntasks=$(determine_resources --data=mydata --model=mymodel) --cpus-per-task=$NTHREADS myfile.sh --data=mydata --model=mymodel

> **Warning**
>
> The extra arguments `--data=mydata --model=mymodel` should be consistent, i.e. they should be passed consistently to `determine_resources` and the whole `sbatch` command.

## Interactive Usage

### Running a single experiment

The `CTExperiments` package ships with a helper function that can be used to set up the default experiment configuration:

``` julia
config_file = generate_template()
```

An experiment can be generated directly from that configuration file:

``` julia
experiment = Experiment(config_file)
```
