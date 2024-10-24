# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version [0.0.1] 

Tried different things here.

**Training schemes** Two main ideas: 1) simply use the energy differential between counterfactuals and observed samples in the target domain; 2) same as 1), but also compute the classification loss for the counterfactual prediction and target. The latter so far seems to have no effect on performance (does not seem to improve robustness).

**Dimensionality reduction** Tried using PCA for generating counterfactuals during training, but have not observed effect so far. If anything, it makes things worse.

**Penalty strengths** The strengths of the penalties for the plausibility loss (generative loss) and energy regularization have a strong effect on the outcomes. 

**Different generators** Still not entirely clear, but for certain settings the hypothesized outcome appears: ECCo > Generic >>> REVISE.

### Added

- 

