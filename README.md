

# CounterfactualTraining.jl

Teaching model plausible and actionable explanations.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaTrustworthyAI.github.io/CounterfactualTraining.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTrustworthyAI.github.io/CounterfactualTraining.jl/dev/) [![Build Status](https://github.com/JuliaTrustworthyAI/CounterfactualTraining.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaTrustworthyAI/CounterfactualTraining.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/JuliaTrustworthyAI/CounterfactualTraining.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaTrustworthyAI/CounterfactualTraining.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## TL;DR

CounterfactualTraining.jl can be used to train artificial neural networks that are inherently more explainable and robust than contenionally trained models.

## Paper

The package was initially developed as part of our [IEEE SaTML 2026](https://satml.org/2026/) paper *Counterfactual Training: Teaching Models Plausible and Actionable Explanations*: open the [preprint](https://arxiv.org/pdf/2601.16205).

**Abstract**: We propose a novel training regime termed counterfactual training that leverages counterfactual explanations to increase the explanatory capacity of models. Counterfactual explanations have emerged as a popular post-hoc explanation method for opaque machine learning models: they inform how factual inputs would need to change in order for a model to produce some desired output. To be useful in real-world decision-making systems, counterfactuals should be plausible with respect to the underlying data and actionable with respect to the feature mutability constraints. Much existing research has therefore focused on developing post-hoc methods to generate counterfactuals that meet these desiderata. In this work, we instead hold models directly accountable for the desired end goal: counterfactual training employs counterfactuals during the training phase to minimize the divergence between learned representations and plausible, actionable explanations. We demonstrate empirically and theoretically that our proposed method facilitates training models that deliver inherently desirable counterfactual explanations and additionally exhibit improved adversarial robustness.

## Package

The package provides GPU-friendly training routines for models trained in Flux.jl.

### Installation

The package can be installed as follows:

``` julia
using Pkg
Pkg.add("CounterfactualTraining")
```

### Introduction

A self-contained introductory example demonstrating counterfactual training with mutability protection is available on the [Introduction](https://www.taija.org/CounterfactualTraining.jl/dev/introduction/) page.

### Training on GPU

The package provides two training approaches: a branch using [CounterfactualExplanations.jl](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl) as the backend for generating counterfactuals and a performance-focused `Native` submodule with GPU support. The former approach was developed during the initial research stage of the project. See [Training on GPU](https://www.taija.org/CounterfactualTraining.jl/dev/gpu/) for a full walkthrough with MNIST.
