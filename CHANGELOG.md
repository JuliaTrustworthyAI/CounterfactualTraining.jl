# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version [0.0.4]

### Changed

- Treating nascent counterfactuals as adversarial exampels. [#19]
- Ensure that domain and mutability constraints are respected at evaluation time. [#15]

## Version [0.0.3] 

### Results

- Effect of mutability constraints is there but not sure about direction yet

### First Grid Search

#### Generator Parameters

##### Linearly Separable 

- **Energy Penalty**: *ECCo* generally does yield better results than *Vanilla* for higher choices of the energy penalty (10,15) during training. *Generic* performs poorly accross the board. *Omni* seems to have an anchoring effect, in that it never performs terribly but also never as good as the best *ECCo* results. *REVISE* performs poorly across the board.
- **Cost (distance penalty)**: Results for all generators (except *Omni*) are quite bad, which can likely be attributed to extremely bad results for some choices of the **Energy Penalty** (results here are averaged). For *ECCo* and *Generic*, higher cost values generally lead to worse results.
- **Maximum Iterations**: No clear patterns recognizable, so it seems that smaller choices are ok. 
- **Validity**: *ECCo* almost always valid except for very low values during training and high values at evaluation time. *Generic* often has poor validity.
- **Accuracy**: Seems largely unaffected.

##### Circles

- **Energy Penalty**: *ECCo* consistently yields better results than *Vanilla*, though primarily for low to medium choices of the energy penalty (<=5) during training. The same goes for *Generic*, which sometimes outperforms *ECCo* (for small energy penalty at evaluation time). *Omni* does alright for lower energy penalty at evaluation time, but loses out for higher choices. *REVISE* performs poorly across the board (except very low choices at evaluation time).
- **Cost (distance penalty)**: *ECCo* and *Generic* generally achieve the best results when no cost penalty is used during training. Both *Omni* and *REVISE* are largely unaffected.
- **Maximum Iterations**: *ECCo* consistently yields better results for higher numbers of iterations. *Generic* generally does best for a medium number (50). *Omni* is sometimes invalid (**???**).
- **Validity**: *ECCo* tends to outperform its *Vanilla* counterpart, though primarily for low to medium choices of the energy penalty (<=5) during training and evaluation. *Vanilla* typically worse across the board.
- **Accuracy**: Mostly unaffected, but *REVISE* again consistently some deterioration and *ECCo* deteriorates for high choices of energy penalty during training, reflecting other outcomes above.
  
##### Moons

- **Energy Penalty**: *ECCo* consistently yields better results than *Vanilla*, except for very low choices of the energy penalty during training for which it performs abismal. *Generic* performs quite badly across the board for high enough choices of the energy penalty at evaluation time. *Omni* has small positive effect. *REVISE* performs poorly across the board.
- **Cost (distance penalty)**: *Generic* generally does better for higher values, while *ECCo* does better for lower values.
- **Maximum Iterations**: No clear patterns recognizable, so it seems that smaller choices are ok. 
- **Validity**: *ECCo* generally achieves full validity except for very low choices the energy penalty during training and high choices at evaluation time. *Generic* performs poorly for high choices of the energy penalty during evaluation.
- **Accuracy**: Largely unaffected although *ECCo* suffers a bit for very low choices the energy penalty during training. *REVISE* suffers a lot in general (around 10 percentage points).

##### Overlapping

- **Energy Penalty**: All generators perform poor across the board here.
- **Validity**: Also quite poor across the board. 
- **Accuracy**: Unaffected.

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

