# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version [0.0.3] 

### Results

- Effect of mutability constraints is there but not sure about direction yet

### First Grid Search

#### Generator Parameters



### Engineering

- A sense of the scale of things:
   - **One grid search** for a single dataset: 
       1. *Training*: 600 experiments $\times$ 50 iterations $\times$ 300 counterfactuals -> 9_000_000 counterfactuals
       2. *Evaluation*: 600 experiments $\times$ 12 evaluations $\times$ 5 rounds  $\times$ 100 counterfactuals -> 3_600_000 counterfactuals ($\times$ 5 metrics)
       3. With around 50 search steps: **650 million backpropagations**
- Cannot use MMD during grid search, simply too much memory required even for synthetic data

## Version [0.0.2] - 2025-01-13

### Engineering

Tough few days trying things with MPI that simply do not seem to work:

- Nested distribution seems impossible. Specifically, I have been trying to distribute 1) models/experiments across processes and then 2) for each model/experiment distribute the counterfactual search across processes. This has entailed all sorts of issues:
    1. The number of models is typically smaller than the number of counterfactuals, making communication between processors very challenging.
    2. I have continuously run in out-of-memory issues.
    3. More recently, I have ended up with the model for each experiment implying data race issues. 
- Instead, just combine multi-processing with multi-threading

### Results

- Outcomes currently seem to be heavily tied to `GeneratorParams`:
   - Penalizing distance too much never yields useful/faithful counterfactuals during training
      - *Could tune/vary this during training?*
   - If `GeneratorParams` during evaluation differ a lot from the once chosen during training, the outcomes are also poor.
      - *Could introduce stochasticity during training?*
- It seems that high penalties on the contrastive divergence during training generally lead to good outcomes. 

## Version [0.0.1] 

Tried different things here.

**Training schemes** Two main ideas: 1) simply use the energy differential between counterfactuals and observed samples in the target domain; 2) same as 1), but also compute the classification loss for the counterfactual prediction and target. The latter so far seems to have no effect on performance (does not seem to improve robustness).

**Dimensionality reduction** Tried using PCA for generating counterfactuals during training, but have not observed effect so far. If anything, it makes things worse.

**Penalty strengths** The strengths of the penalties for the plausibility loss (generative loss) and energy regularization have a strong effect on the outcomes. 

**Different generators** Still not entirely clear, but for certain settings the hypothesized outcome appears: ECCo > Generic >>> REVISE.

**Burn-in period** This speeds things up but seems to make things worse.

