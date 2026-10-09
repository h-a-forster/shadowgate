# Related work

shadowgate sits at the intersection of three lines of work: LLM cascades and routers,
confidence/uncertainty signals for language models, and classical survey-sampling
estimators. This page lists the work it builds on, grouped by topic. Each entry gives
the authors, year, linked title, venue, and how it relates to shadowgate.

## Framing

- **Oh & Gobet, 2024.** [System 1.5: Designing Metacognition in Artificial Intelligence](https://openreview.net/forum?id=SEJg9yIhPz).
  NeurIPS 2024 Workshop on System-2 Reasoning at Scale ([workshop page](https://neurips.cc/virtual/2024/104306)).
  A theoretical framework for metacognitive regulation that monitors processing, generates
  responses and evaluates outcomes to decide when to rely on fast "intuitive" processing versus
  slower analysis. shadowgate borrows the "System 1.5" name for the gate between a cheap model and
  a strong model; it implements an engineering version of that idea, not the paper's framework.

## LLM cascades and routers

- **Chen, Zaharia & Zou, 2023.** [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176). arXiv.
  Calls LLMs in sequence and uses a learned scorer to decide whether to accept an answer or
  try the next model; evaluated as cost/accuracy trade-offs on labelled datasets.
- **Aggarwal, Madaan et al., 2024.** [AutoMix: Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963). NeurIPS 2024.
  The small model verifies its own answer with few-shot self-verification, and a POMDP-based
  router uses that noisy confidence to decide whether to escalate.
- **Gupta, Narasimhan, Jitkrittum, Rawat, Menon & Kumar, 2024.** [Language Model Cascades: Token-level Uncertainty and Beyond](https://arxiv.org/abs/2404.10136). ICLR 2024.
  Studies deferral rules for generative cascades; shows that whole-sequence uncertainty has a
  length bias and that learned post-hoc rules over token-level uncertainties defer better.
- **Yue, Zhao, Zhang, Du & Yao, 2024.** [Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning](https://arxiv.org/abs/2310.03094). ICLR 2024.
  Uses answer consistency of the weaker model across samples (including Chain-of-Thought vs.
  Program-of-Thought) as the escalation signal on reasoning benchmarks.
- **Ding, Mallick, Wang, Sim, Mukherjee, Ruhle, Lakshmanan & Awadallah, 2024.** [Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing](https://arxiv.org/abs/2404.14618). ICLR 2024.
  A router predicts query difficulty *before* generation and sends each query to a small or large
  model, with a quality level tunable at test time.
- **Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous & Stoica, 2025.** [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665). ICLR 2025.
  Trains strong-vs-weak routers from human preference data; routing happens before any model
  answers, and quality is measured on public benchmarks.
- **Zellinger & Thomson, 2025.** [Rational Tuning of LLM Cascades via Probabilistic Modeling](https://arxiv.org/abs/2501.09345). arXiv.
  Models the joint error distribution of the models in a cascade to tune confidence thresholds,
  as an alternative to black-box threshold search.
- **Dou, Lian & Li, 2026.** [Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference](https://arxiv.org/abs/2607.25018). arXiv.
  Replaces the hand-tuned confidence threshold with a conformal prediction-set-size deferral rule
  that carries a finite-sample accuracy guarantee (multiple-choice tasks).

### Routing benchmarks

- **Hu et al., 2024.** [RouterBench: A Benchmark for Multi-LLM Routing System](https://arxiv.org/abs/2403.12031). arXiv.
  A precomputed dataset of model outputs plus a framework for comparing routers on cost/quality.
- **Huang et al., 2025.** [RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs](https://arxiv.org/abs/2503.10657). arXiv.
  Large collection of per-model performance records across many LLMs for offline router evaluation.
- **Yang et al., 2026.** [TwinRouterBench: Fast Static and Live Dynamic Evaluation for Realistic Agentic LLM Routing](https://arxiv.org/abs/2605.18859). arXiv.
  Step-level routing for agents, with a static offline track and a live execution track.

## Confidence signals

- **Tian et al., 2023.** [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975). EMNLP 2023.
  For RLHF-tuned models, verbalized confidence is often better calibrated than token probabilities.
- **Xiong et al., 2024.** [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063). ICLR 2024.
  Benchmarks black-box confidence elicitation (verbalized, sampling, consistency aggregation);
  finds verbalized confidence tends to be overconfident.
- **Wang et al., 2023.** [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). ICLR 2023.
  Samples multiple reasoning paths and takes the majority answer; agreement rate is the basis of
  shadowgate's self-consistency signal.
- **Kadavath et al., 2022.** [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221). arXiv.
  Introduces P(True) self-evaluation of proposed answers and P(IK) prediction; the basis for
  using a model (or a separate monitor model) to score an answer's correctness.
- **Kuhn, Gal & Farquhar, 2023.** [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664). ICLR 2023.
  Semantic entropy: clusters sampled answers by meaning before computing entropy.
- **Farquhar, Kossen, Kuhn & Gal, 2024.** [Detecting hallucinations in large language models using semantic entropy](https://doi.org/10.1038/s41586-024-07421-0). Nature.
  Applies semantic entropy to detecting confabulations in free-form generation.

## Selective prediction and threshold selection

- **Geifman & El-Yaniv, 2017.** [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500). NeurIPS 2017.
  Classifier with a reject option and a risk-coverage trade-off; "keep vs. escalate" in a cascade
  is the same decision, with escalation in place of rejection.
- **Geifman, Uziel & El-Yaniv, 2019.** [Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers](https://arxiv.org/abs/1805.08206). ICLR 2019.
  Uses the area under the risk-coverage curve (AURC) as a summary metric for confidence scores.
- **Angelopoulos, Bates, Candès, Jordan & Lei, 2021.** [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052). arXiv.
  Chooses thresholds with finite-sample risk guarantees by framing selection as multiple testing.
- **Angelopoulos, Bates, Fisch, Lei & Schuster, 2024.** [Conformal Risk Control](https://arxiv.org/abs/2208.02814). ICLR 2024.
  Extends split conformal prediction to control the expected value of any monotone loss; a
  principled alternative to selecting a threshold on a held-out split.

## Label-efficient evaluation

- **Kossen, Farquhar, Gal & Rainforth, 2021.** [Active Testing: Sample-Efficient Model Evaluation](https://proceedings.mlr.press/v139/kossen21a.html). ICML 2021.
  Selects which test points to label with non-uniform probabilities and removes the resulting bias
  with importance weights, the same principle as shadowgate's confidence-stratified audit.
- **Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, 2023.** [Prediction-powered inference](https://doi.org/10.1126/science.adi6000). Science.
  Valid confidence intervals from a small labelled set plus many model predictions; relevant when
  a strong-model or judge label is used as a proxy for ground truth.

## Survey-sampling foundations

- **Horvitz & Thompson, 1952.** [A Generalization of Sampling Without Replacement from a Finite Universe](https://doi.org/10.1080/01621459.1952.10483446). JASA 47, 663-685.
  Unbiased totals under unequal inclusion probabilities by weighting each sampled unit by 1/π.
- **Hájek, 1971.** Comment on "An essay on the logical foundations of survey sampling, part one" by D. Basu. In Godambe & Sprott (eds.), *Foundations of Statistical Inference*, Holt, Rinehart and Winston.
  The ratio (self-normalised) form of the Horvitz-Thompson estimator used for weighted rates.
- **Kish, 1965.** *Survey Sampling*. Wiley.
  Design effects and the effective sample size of a weighted sample, used to size intervals.
- **Wilson, 1927.** [Probable Inference, the Law of Succession, and Statistical Inference](https://doi.org/10.1080/01621459.1927.10502953). JASA 22, 209-212.
  The Wilson score interval for a binomial proportion.
- **Clopper & Pearson, 1934.** [The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial](https://doi.org/10.1093/biomet/26.4.404). Biometrika 26, 404-413.
  The exact (conservative) binomial interval.

## Related tooling

- [safeswap](https://github.com/shubh-tiwari/safeswap) sends a small random share of cheap-model
  requests to the expensive model as a "shadow sample", compares answers with a judge and reports
  the quality loss with confidence intervals. It is the closest existing tool to shadowgate's audit.
- [vLLM Semantic Router, shadow dispatch](https://vllm-sr.ai/docs/tutorials/plugin/shadow-dispatch/)
  mirrors a sampled fraction of requests to a candidate model in the background and records
  outcomes; it leaves comparing outputs to downstream tooling.

## What shadowgate adds

Most cascade and router papers evaluate offline, on labelled benchmarks or precomputed outcome
tables, and report a cost/accuracy curve for a fixed dataset. In deployment, the cases the cheap
model keeps are never checked by the strong model, so their error rate is not observed. shadowgate
focuses on measuring that rate continuously: it samples kept cases with known, confidence-stratified
probabilities, re-answers them with the strong model, and reports the skipped-case disagreement
rate with inverse-probability-weighted confidence intervals. It combines this with offline threshold
sweeps (Pareto curves, held-out threshold selection) so the threshold chosen offline can be checked
against what happens in deployment. The estimators themselves are standard survey-sampling and
importance-weighting tools; the contribution is packaging them for confidence-gated cascades.
