# MemRiskBench: Trace-Aware Risk-Preserving Evaluation for Long-Horizon LLM Agents

Jianhua Jiang<sup>,1,2</sup> Dongbo Yuan<sup>1</sup> Weihua Li<sup>3</sup>

<sup>1</sup>School of Artificial Intelligence and Computer Science, Jilin University of Finance and Economics, 130117 Changchun, China

<sup>2</sup>Jilin Province Key Laboratory of Fintech, Jilin University of Finance and Economics, 130117 Changchun, China

<sup>3</sup>School of Engineering, Computer and Mathematical Sciences, Auckland University of Technology, Auckland, New Zealand

## Abstract

Long-horizon LLM agents accumulate memory across sessions, creating sparse but high-impact risks: stale facts, conflicting updates, crossuser leakage, revoked-memory reuse, and constraint decay. Standard aggregate scores hide per-risk failure rates—a model achieving 78% average accuracy may still leak data in 4% of episodes—and benchmark compression preferentially discards the rare high-severity events that distinguish a mostly-working model from one that occasionally causes harm. We present MemRiskBench. The primary contribution is a five-category risk taxonomy (plus one documented, unscored category) operationalized by deterministic trace-grounded checks, instantiated as a 120-episode scripted benchmark with full trace logging and no LLM-as-judge on the pass/fail path, evaluated on five locally run quantized instruction-tuned models. Second, a risk-preserving subset selector: a coverage-constrained greedy selector on deterministic trace-derived features that retains full ranking (Spearman ρ = 0.975, deterministic; CI collapses to a point estimate with zero bootstrap variance), risk coverage (1.0), and high-risk model detection (1.0) at a 20% subset size, reducing compute 5×. Unlike ranking-only subset selectors, this selector additionally preserves risk-type coverage and high-risk model detection using trace-grounded deterministic features that do not require an LLM judge. All episodes, traces, the scoring implementation, and the selector are released to support reproducible evaluation and risk assessment of deployed LLM agents.

Keywords: LLM agents; long-horizon memory; risk evaluation; benchmark; subset selection; memory safety

## 1 Introduction

Long-horizon LLM agents accumulate memory across sessions. At the start of each session, the agent must decide which stored facts apply, which have been superseded, and which are out of scope. Memory failures here are not generic task failures but sparse, stateful, high-impact events: a stale address is a nuisance, a cross-user disclosure is serious, and a silently dropped “do not send” constraint is a policy violation. Standard evaluation misses these risks. A headline score of 78% average task success can sit alongside a 4% cross-scope leakage rate and a 25% constraint-decay rate—the overall number hides the gap. Most evaluation frameworks (2023; 2024) report aggregate scores by default and provide no explicit mechanism for surfacing per-(model, risk) pass rates. Compression makes this worse. As evaluation costs grow, practitioners run only a 20% subset. Random or stratified subsampling—the default in HELM-style evaluation—tend to drop the rare high-severity events that distinguish otherwise similar models.

Existing memory-capability benchmarks (2025; 2024; 2025; 2025; 2025) evaluate long-term memory through QA accuracy or information-extraction probes. They measure how well an agent remembers, not when memory causes harmful failures. A recent line of work on agent-benchmark compression (2026) studies how small task subsets can preserve model rankings at lower cost, proposing a Mid-Range Dificulty Filter motivated by Item Response Theory. That work targets ranking fidelity; it does not consider whether high-severity failure modes are preserved alongside the ranking. Existing harm taxonomies (Weidinger et al., 2024) catalog output-level risks but do not treat memory violations themselves as the unit of evaluation.

Contributions. (C1, primary) A benchmark for memory-risk evaluation: a five-category risk taxonomy with deterministic checks, a 120-episode scripted benchmark with full trace logging and balanced dificulty, and a deterministic scorer with no LLM-as-judge on the pass/fail path, evaluated on five locally run quantized instruction-tuned models. (C2, secondary) A risk-preserving subset selector: a coverage-constrained greedy selector on deterministic trace-derived features that retains ranking, risk-type coverage, and high-risk model detection at a 20% subset size, reducing compute 5× relative to the full benchmark. The selector’s objective—ranking plus risk coverage plus high-risk model detection— and its use of trace-grounded deterministic features distinguish it from rankingonly subset selectors such as 2026. (C3) Empirical characterization of five local quantized models (1.5B–7B parameters) under controlled risk conditions, including a memory-baseline experiment showing that standard retrieval-level filtering performs worse than no filtering, confirming that memory risk requires dedicated evaluation.

The full pipeline—episode execution, deterministic scoring, and riskpreserving subset selection—is outlined in Figure 1.

![](images/9f739107177f6230dc54c2712a2064a485e669b894a1e1ca9bf0ec8ec1de5078.jpg)

## 2 Related Work

## 2.1 Memory-capability benchmarks.

Recent benchmarks evaluate long-term memory in LLM agents and assistants, including LongMemEval (Wu et al., 2025), LoCoMo (Maharana et al., 2024), MemBench (Tan et al., 2025), MemoryAgentBench (Hu et al., 2025), and BEAM (Tavakoli et al., 2025). These benchmarks measure retrieval recall or QA accuracy; they do not address multi-session risk evaluation. Memory-contamination literature establishes secret-leakage tests (Carlini et al., 2019, 2021) and membership inference (Shokri et al., 2017), whereas our scorer targets cross-scope runtime leakage under scripted conditions. Context-length benchmarks such as L-Eval (An et al., 2024), LongBench (Bai et al., 2024), and RULER (Hsieh et al., 2024) test retrieval at 128K tokens; instead, MemRiskBench encodes dificulty in stateful memory, tool calls, and constraint lifetime. Memory mechanisms such as MemGPT (Packer et al., 2024), MemoryBank (Zhong et al., 2024), and cognitive architectures (Sumers et al., 2024) propose memory consolidation and tool-use policies; our memory-baseline experiment isolates retrieval-level filtering as a lower-bound check.

## 2.2 Benchmark compression and subset selection.

Prior work studies how small task subsets can preserve the signal benchmark consumers care about while reducing evaluation cost. Ndzomga (2026) study eight agent benchmarks, 33 scafolds, and 70+ model configurations and identify a robust empirical asymmetry between rank-order prediction (which remains stable under scafold and temporal shift) and absolute score prediction (which degrades). Exploiting this asymmetry, they propose the Mid-Range Dificulty Filter (MR): a deterministic, optimization-free rule that selects tasks with historical pass rates between 30% and 70%, motivated by Item Response Theory. MR achieves Spearman $\rho \approx 0 . 9 4$ while reducing task counts by 44%–70%. Earlier work on coreset selection and active learning includes BADGE (Ash et al., 2021), GradMatch (Killamsetty et al., 2021), and dificulty- or uncertaintybased subsampling. These methods target ranking fidelity alone. None explicitly preserves risk-type coverage or high-risk model detection. MemRiskBench’s selector adopts a coverage-constrained objective on deterministic trace-derived features, and our RQ2 finding that stratified-by-risk sampling preserves coverage but loses high-risk detection (Spearman 0.698, high-risk detection 0.345) supports the argument that ranking fidelity alone is insuficient for risk-sensitive evaluation.

## 2.3 Harm taxonomies and safety evaluation.

Weidinger et al. (2024) catalog sociotechnical risks of generative AI, including privacy and output-level harms. MemRiskBench treats memory violations themselves as the unit of evaluation with deterministic, trace-grounded scoring. It adds a process-level view of when and how memory causes harmful failures to output-level evaluation.

## 3 Risk Taxonomy

We define five primary risks (R1–R5, deterministically scored) and one documented-but-unscored risk (R6, summary contamination) that we plan to integrate in a future version. Each risk is operationalized by deterministic checks across four categories: text checks (string containment, regex), tool checks (call verification, argument equality), memory checks (usage tracking, status, scope), and constraint checks (activity/violation monitoring).

## 3.1 R1: Stale memory reliance.

The agent acts on a memory valid at t<sub>1</sub> superseded by a valid update at $t _ { 2 }$ using the old value despite newer available memory.

## 3.2 R2: Conflicting memory misresolution.

Two incompatible facts coexist; the episode specifies a resolution rule (latest timestamp, trusted source, explicit correction, or scope) and the agent selects incorrectly or blends them.

## 3.3 R3: Cross-scope leakage.

Unauthorized cross-user/scope/memory appears in an unauthorized context.

## 3.4 R4: Revoked memory reuse.

A deletion/revocation/invalidation is issued, but the agent later uses the same memory.

## 3.5 R5: Constraint decay.

A safety/policy/task constraint is introduced; after unrelated turns, a later action violates it.

## 3.6 R6: Summary contamination (documented, not scored).

Memory summaries compress noisy/malicious/incorrect content into cleanlooking entries that later influence behavior—the intersection of prompt injection and persistent memory. R6 episodes exist in the release as a separately-scored track; integrating R6 into the main scoring requires a hybrid deterministic–LLM evaluation pipeline that is left to future work.

## 4 Benchmark

The benchmark comprises 120 scripted episodes: 24 per risk type, distributed across three dificulty tiers (easy / medium / hard; counts per tier per risk in Table 4). Dificulty depends on distractor count, intermediate updates, and trajectory length (5–8 turns easy, 10–14 turns medium, 15–25 turns hard).

<table><tr><td>Risk</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Total</td></tr><tr><td>R1 Stale</td><td>8</td><td>9</td><td>7</td><td>24</td></tr><tr><td>R2 Conflict</td><td>8</td><td>9</td><td>7</td><td>24</td></tr><tr><td>R3 Leakage</td><td>7</td><td>9</td><td>8</td><td>24</td></tr><tr><td>R4 Revoke</td><td>7</td><td>9</td><td>8</td><td>24</td></tr><tr><td>R5 Constraint</td><td>7</td><td>9</td><td>8</td><td>24</td></tr><tr><td>Total</td><td>37</td><td>45</td><td>38</td><td>120</td></tr></table>

## 4.1 Episode structure.

An episode is a directed acyclic graph of events (memory writes, user messages, tool calls, tool returns, constraint declarations, decision points). The agent sees events in order and produces, at each decision point, either a tool call, a final answer, or a clarification question. The trace logs all four kinds of state (memory store, tool-call log, constraint state, message log) at each decision point.

## 4.2 Determinism and scoring.

Scoring is deterministic. The scorer takes the episode definition and the trace and emits a boolean success flag, a numeric score in [0, 1], a list of violations, per-risk pass/fail and evidence, and per-check results. The main pass/fail path uses no LLM-as-judge; LLM-as-judge appears only in qualitative error analysis in the online repository. Each episode has 1–3 required checks; the score is the fraction of checks passing.

## 4.3 Strict-matching revision (control analysis).

Twenty episode definitions originally encoded the expected argument as a “label: value” form that no model in our set emits in full, producing systematic false negatives. We re-scored the existing traces (no model re-runs) against revised expected values that strip the label prefix on 19 of the 20 afected episodes. The revision altered the deterministic score of 39 of 600 trace-episode pairs. To support internal validity, the release includes both the original and revised expected values and a per-(model, risk) re-scoring audit; the model ordering within each risk and the headline size–risk interaction (Qwen2.5-1.5B lowest on cross-scope leakage) are preserved under both scoring. Future benchmark versions can reintroduce label-preserving checks where genuinely required.

## 4.4 Safety-aware refusal scoring (protocol).

Under the strict default, an episode in which the agent refuses to act (no tool call, refusal text) is scored as a failure. In the R3 (cross-scope leakage) and R5 (constraint decay) categories, refusal constitutes the safety-correct outcome. We therefore additionally compute a safety-aware variant in which “no tool call + refusal text” is counted as a safety-pass for R3 and R5 only. The release includes both the default and safety-aware per-(model, risk) pass rates, and the sensitivity analysis is discussed in Section 6.

## 4.5 Release contents.

The release includes the 120 episode definitions, the deterministic scoring implementation, the trace schema, the subset selector implementation, the bootstrap audit script, and the raw 600 traces (5 models × 120 episodes).

## 5 Risk-Preserving Subset Selection

## 5.1 Problem statement.

Let E be the full benchmark with $| \mathcal { E } | = 1 2 0$ episodes and $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { 5 } \}$ the set of reference models. A subset $S \subseteq { \mathcal { E } } , | S | = k .$ , is risk-preserving if it maximizes

$$
\Psi ( S ) = \alpha \rho \big ( r ( S ) , r ( \mathcal { E } ) \big ) + \beta \mathrm { c o v } ( S ) + \gamma \mathrm { h r } ( S )\tag{1}
$$

subject to $| S | = k ,$ with $\alpha = \beta = \gamma = 1$ . Here $\rho$ is the Spearman correlation between the ranking induced by S and the full benchmark; cov $\mathbf { \boldsymbol { \cdot } } ( S ) \in [ 0 , 1 ]$ measures risk-type coverage; hr $\mathbf { \boldsymbol { \cdot } } ( S ) \in [ 0 , 1 ]$ measures high-risk model detection (a model is high-risk if its pass rate ranks in the bottom two of five).

## 5.2 Episode feature vector.

Each episode is described by a 12-dimensional feature vector built from the trace schema: dificulty (tier, trajectory length), discrimination (cross-model passrate spread), failure density (fraction of reference models failing the episode), and complexity (distinct risk types, tool calls, constraints). The ablation in Section 6 confirms that the discrimination feature alone drives the ranking; the other three features contribute negligible lift at any subset fraction tested.

## 5.3 Discrimination feature and cross-validation.

The discrimination feature is computed via leave-one-model-out cross-validation to avoid test-set leakage: for each held-out model, the feature is computed on the remaining four reference models, the subset is selected using those features, and the held-out model’s rank is then evaluated. This yields rankings consistent with the full five-model feature.

## 5.4 Selector algorithm.

The selector performs greedy forward selection without replacement: at each step, it adds the episode that most improves Ψ(S). Six baselines are compared: Random, Stratified-by-risk, Dificulty, Disagreement (pass-rate variance), Clustering (k-medoids on full features), and Trace-Feature (clustering on trace-only metrics). Stochastic baselines use 1000 bootstrap resamples; deterministic ones yield identical subsets across resamples and their confidence intervals degenerate to point estimates.

## 5.5 Reference-set dependence.

The selector builds on the v3 reference ranking. A new model family absent from the reference set may exhibit diferent discrimination patterns, rendering the v3 selector suboptimal. The release includes the feature matrix and selector implementation so practitioners can rebuild the reference ranking on their own model set—at the cost of recomputation. We do not claim equivalent performance without a reference set. An ofline discrimination estimate is left to future work.

## 6 Experiments

## 6.1 Models.

Five locally run quantized instruction models through llama.cpp: Qwen2.5- 1.5B-Instruct (Q4 K M), Qwen2.5-3B-Instruct (Q4 K M), Qwen2.5-7B-Instruct (Q3 K M), Phi-3.5-mini-instruct (Q4 K M), Llama-3.2-3B-Instruct (Q4 K M). Qwen2.5-7B runs at Q3 because the 16 GB local GPU cannot hold a 7B Q4 resident alongside the prompt cache; this quantization confound is discussed in Section 8.

## 6.2 Protocol.

The bootstrap protocol resamples the subset-selection seed 1000 times, holding the full benchmark fixed. All reported confidence intervals are 95% percentile bootstrap intervals over 1000 resamples. Random seeds are fixed and sampled indices are released.

## 6.3 RQ1: Aggregate scores hide per-risk failure rates.

Across the five models, average task success spans 0.775 to 0.885, a 0.110 spread. Per-(model, risk) pass rates appear in Table 6.3. Cross-scope leakage is hardest for the smallest model (Qwen2.5-1.5B passes only 1 of 24 leakage episodes; Fisher’s exact $p = 0 . 0 0 4$ versus Phi-3.5-mini’s $1 0 / 2 4 )$ , though neither this nor the comparison against Qwen2.5-3B $( p = 0 . 0 4 9 )$ survives Holm correction over all 50 pairwise model–risk tests (smallest adjusted $p = 0 . 2 2 )$ . Constraint decay shows an inverse pattern: 1.5B obeys all 24 explicit constraints, while 3B and 7B each violate two, and Phi violates three.

<table><tr><td>Model</td><td>Stale</td><td>Conflict</td><td>Leakage</td><td>Revoke</td><td>Constraint</td></tr><tr><td>Qwen2.5-1.5B</td><td>71% (17/24)</td><td>58% (14/24)</td><td>4% (1/24)</td><td>62% (15/24)</td><td>100% (24/24)</td></tr><tr><td>Qwen2.5-3B</td><td>79% (19/24)</td><td>75% (18/24)</td><td>12% (3/24)</td><td>79% (19/24)</td><td>92% (22/24)</td></tr><tr><td>Qwen2.5-7B</td><td>75% (18/24)</td><td>83% (20/24)</td><td>38% (9/24)</td><td>75% (18/24)</td><td>92% (22/24)</td></tr><tr><td>Phi-3.5-mini</td><td>75% (18/24)</td><td>79% (19/24)</td><td>42% (10/24)</td><td>88% (21/24)</td><td>88% (21/24)</td></tr><tr><td>Llama3.2-3B</td><td>62% (15/24)</td><td>75% (18/24)</td><td>25% (6/24)</td><td>71% (17/24)</td><td>92% (22/24)</td></tr></table>

Model ranking preservation across subset fractions (95% bootstrap CI, 1000 seeds)  
![](images/5ed6d07861e6ea0cd69cbea2f3f5f12f91176281a3b4cd762860ec5a97c7b69b.jpg)

## 6.4 RQ2–RQ3: Risk-preserving compression.

Figure 6.3 and Table 6.5 report the 20%-subset comparison. Random sampling reaches Spearman 0.733 with 95% $\mathrm { C I } \ [ - 0 . 0 5 , 0 . 9 7 ]$ ; its confidence interval includes negative values, indicating disagreement with the full-benchmark ranking on roughly a quarter of model pairs. Stratified-by-risk sampling preserves risk coverage (1.0) but achieves only Spearman 0.698 and high-risk detection 0.345, missing at least one high-risk model in half of resamples. The risk-preserving selector achieves Spearman 0.975 (deterministic; CI collapses to a point estimate), risk coverage 1.0, and high-risk detection 1.0 at all tested fractions. Relative to the Random baseline, the improvement on fail coverage is $z = + 9 . 1 \ ( p < 0 . 0 0 1 )$ and on high-risk detection $z = + 1 . 6 \ ( p = 0 . 0 5 )$ ; the improvement on Spearman is +0.24 (borderline at the 97.5th percentile). Trace-Feature clustering reaches Spearman 0.975 but loses half the high-risk models at 20% and 30%, confirming that risk detection requires the discrimination signal rather than simple clustering.

## 6.5 RQ4: Feature ablation.

The cross-model pass-rate spread carries the full ranking signal: removing it drops Spearman to 0.308; removing any other feature (failure density, complexity, or dificulty) leaves Spearman unchanged at 0.975. Coverage metrics remain stable because the v3 reference set satisfies risk-coverage objectives at $k \geq 2 4$ The selector’s value lies in the coverage constraint and trace-grounded feature definition, not in feature-space tuning.

![](images/4befba8e70c616c92c1af4b41d26bc1e643bed40c3cd2e9900e71a67fcb2205d.jpg)

<table><tr><td>Method</td><td>Spearman ρ</td><td>Fail Cov.</td><td>Risk Cov.</td><td>Hi-Risk Det.</td></tr><tr><td>Full (120 ep)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>Random</td><td>0.733 [−0.05, 0.97]</td><td>0.201 [0.15, 0.25]</td><td>0.942 [0.80, 1.00]</td><td>0.600 [0.00, 1.00]</td></tr><tr><td>Stratified</td><td>0.698 [−0.21, 0.97]</td><td>0.178 [0.13, 0.22]</td><td>1.000</td><td>0.345 [0.00, 0.50]</td></tr><tr><td>Difficulty</td><td>0.821</td><td>0.316</td><td>0.800</td><td>1.000</td></tr><tr><td>Disagreement</td><td>0.975</td><td>0.289</td><td>1.000</td><td>1.000</td></tr><tr><td>Clustering</td><td>0.962 [0.82, 0.97]</td><td>0.289</td><td>1.000</td><td>0.500</td></tr><tr><td>Trace-Feature</td><td>0.975</td><td>0.316</td><td>1.000</td><td>1.000</td></tr><tr><td>Risk-Preserving (ours)</td><td>0.975</td><td>0.316</td><td>1.000</td><td>1.000</td></tr></table>

## 6.6 RQ5: Held-out-model generalization (concept verification).

To test generalization beyond the reference set, we hold out Qwen2.5-1.5B (the smallest model, architecturally distinct from the larger reference models) and recompute the feature matrix and subset selection using only the remaining four reference models. The greedy selector identifies a 24-episode subset that preserves the reference-model ranking perfectly (Spearman $\rho = 1 . 0 0 0 )$ and yields an average pass rate of 0.812 on the held-out model—above the full-benchmark average of 0.775. The selected subset covers all five risk types. The greedy selector, optimizing for discrimination, yields a subset skewed toward conflicttype episodes (20 of 24). The 0.812-vs-0.775 pass-rate comparison mixes subset composition with model performance; we report it as a descriptive sanity check only, with no statistical generalization claim. A risk-coverage-constrained balanced selector enabling fair held-out evaluation across multiple unseen models is a planned companion study.

## 6.7 Sensitivity analysis: safety-aware refusal scoring.

Under the safety-aware variant (R3 and R5 only: “no tool call + refusal text” counts as safety-pass), per-(model, risk) pass rates change only for R3 and R5 episodes where a model refused. The release ships both default and safetyaware per-(model, risk) numbers so practitioners can audit the headline size–risk interaction (Qwen2.5-1.5B lowest on leakage) under both scoring conventions; a full quantitative sensitivity table is left to the online repository to keep this section focused.

## 6.8 Memory baseline.

A retrieval-level memory baseline shows that standard techniques—RAG and recency filtering—perform worse than no filtering on MemRiskBench, confirming that memory risk requires dedicated evaluation rather than of-the-shelf retrieval.

## 7 Analysis

## 7.1 Per-risk failure patterns.

Table 6.3 reveals two notable inversions. On constraint decay, Qwen2.5-1.5B obeys all 24 explicit “do not” constraints, while 3B, 7B, Phi, and Llama each violate 2–3—a tendency of larger instruction-tuned models to be “helpful.” On cross-family ordering, Phi-3.5-mini outperforms all three Qwen sizes on leakage (10/24 vs 1/24–9/24) and tops revoke (21/24), likely from a stronger instructionfollowing prior, at the cost of slightly worse constraint decay (21/24 vs 24/24). Comparisons involving Qwen2.5-7B (Q3 K M) versus Qwen2.5-3B (Q4 K M) are subject to the quantization confound discussed in Section 8: on stale and revoke classes, 3B passes 19/24 versus 18/24 for 7B, inverting the size-monotone expectation. We interpret the constraint-decay inversion as a real diference, but cannot fully exclude quantization efects.

![](images/a3426861e9371ca974500793831f6c3c40a823fc5179b72142e19cca8022e779.jpg)

## 7.2 Dificulty scaling.

For 3B and larger models, easy → medium → hard pass rates drop monotonically and the per-tier ordering across models remains stable. Qwen2.5-1.5B saturates at easy/medium and shows no further drop at hard, indicating the hard tier does not expose a gap that size would close for this model.

## 7.3 Selector positioning relative to concurrent work.

Our selector and the Mid-Range Dificulty Filter of Ndzomga (2026) both achieve high ranking fidelity at small fractions, but their objectives and operating regimes difer. Ndzomga (2026) target ranking fidelity under scafold and temporal shift, optimize only on ranking, and operate optimization-free across 8 benchmarks; our selector additionally optimizes risk-type coverage and highrisk model detection on a single benchmark, uses deterministic trace-derived features (no LLM judge), and prescribes a 20% subset that is dominated by discrimination episodes. The two approaches are complementary: MR is suitable for broad leaderboard ranking; risk-preserving selection is suitable when high-severity failure modes must be retained alongside the ranking. An empirical comparison under a common protocol is left to future work.

## 8 Limitations

## 8.1 Simulated tool environment.

Episodes employ a scripted tool environment rather than a real browser, email client, or ofice suite; tool calls are deterministic stubs. A real-environment

follow-up would extend the taxonomy beyond scripted stubs (e.g., adversarial user behavior).

## 8.2 Model family and size coverage.

Five models were evaluated, all local and under 8B parameters, with three of five belonging to the Qwen family. No frontier-class API models (GPT-4, Claude, Gemini) were evaluated. The size range reflects the 16 GB consumer-GPU constraint; Qwen2.5-7B at Q4 K M and any 13B+ model exceed this paper’s scope.

## 8.3 Quantization confound.

Qwen2.5-7B uses Q3 K M while the other four use Q4 K M, producing nonmonotonicities in the v3 ranking on stale and revoke classes. We interpret the constraint-decay inversion as a real diference, but cannot fully exclude quantization efects. A Q4-only re-run of 7B may alter the ranking.

## 8.4 Strict string matching and label-prefix revision.

Twenty episode definitions originally encoded “label: value” expected values that no model emits fully. We re-scored the existing traces against revised expected values (label prefix stripped on 19 episodes); the revision altered 39 of 600 scores. The release preserves both original and revised expected values and ships a per-(model, risk) re-scoring audit showing no change in model ordering within any risk class.

## 8.5 Safety-aware refusal scoring.

Default scoring counts a refusal (no tool call + refusal text) as a failure; in R3 and R5 a refusal is the safety-correct outcome. We additionally compute a safety-aware variant in which refusals are safety-pass for R3 and R5; the release ships both sets of numbers.

## 8.6 RQ5 generalization.

Held-out-model evaluation used a single smallest model; the selected subset is skewed toward conflict-type episodes (20 of 24). We report RQ5 as concept verification, not as a statistical generalization claim. A multi-model, balanced held-out study is planned.

## 8.7 Reference-set dependence.

The selector builds on the v3 reference ranking; a new model family absent from the reference set may require recomputation. We release the feature matrix and selector to support this; an automated reference-set drift detector is not provided.

## 8.8 R6 not scored.

Summary contamination (R6) is documented and has separate-track episodes, but is not included in main scoring because the contamination surface area is too broad to cover in a first benchmark.

## 9 Conclusion

MemRiskBench renders memory risks in long-horizon LLM agents quantifiable under controlled, scripted conditions. The primary contribution is a fivecategory taxonomy plus a 120-episode benchmark with deterministic tracegrounded scoring and no LLM-as-judge on the pass/fail path. Second, a riskpreserving subset selector retains ranking (Spearman 0.975, deterministic; zero bootstrap variance), risk-type coverage, and high-risk model detection at a 20% subset while reducing compute 5×. Two workflows are supported: iterating on a new model under a fixed evaluation budget, and auditing an existing model under long-horizon deployment via deterministic traces. Three directions remain open: scaling to ∼1000 episodes with a larger and more diverse reference set; a real-environment extension beyond scripted tool stubs; and a paired defense paper examining whether any memory architecture, retrieval policy, or constraint-tracking mechanism reduces the failure rates reported here.

## Data Availability

The full benchmark release—120 episode definitions, episode and trace schemas, the deterministic scoring implementation, all 600 raw traces, the risk-preserving subset selector, bootstrap audit scripts, paper figures, and the strict-matching revision audit—is publicly available at https://github.com/fuxue-mingzhu/ MemRiskBench.

## Acknowledgments

We thank the anonymous reviewers and colleagues for feedback. This work was supported by national funding agencies (grant numbers anonymized for review). The authors used no generative AI for this work and take full responsibility for the final content.

## Broader Impact and Ethical Considerations

MemRiskBench measures memory-risk failure modes in long-horizon LLM agents under controlled, scripted conditions. All episodes are synthetic; no real users, data, or deployed systems are involved. The benchmark is intended for diagnostic use by model developers and auditors.

## 9.1 Dual-use risk.

The released episode definitions could be repurposed to adversarially probe commercial agents for memory failures. We mitigate this by (i) documenting intended use in the repository README; (ii) not releasing scripted adversarial traces; and (iii) recommending that practitioners complement benchmark scores with deployment-monitored traces rather than rely solely on scripted evaluation.

## 9.2 Limitations of ethical scope.

This work does not address broader societal implications of long-horizon memory systems (e.g., privacy in deployed assistants, consent, data retention).

## References

An, C., Gong, S., Zhong, M., Li, M., Zhang, J., Kong, L., Qiu, X., 2024. Leval: Instituting standardized evaluation for long context language models, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), pp. 13388–13411.

Ash, J.T., Tu, S., Stickland, A., Richards, J., Zhang, C., King, I., Mnih, V., Duvenaud, D., 2021. Badge: Bias-aware discriminative features for eficient subset selection, in: Proceedings of the 38th International Conference on Machine Learning (ICML 2021), pp. 399–408.

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., Li, J., 2024. Longbench: A bilingual, multitask benchmark for long context understanding, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), pp. 3119–3137.

Biderman, S., Schoelkopf, H., Sutawika, L., Gao, L., Tow, J., Abbasi, B., 2024. Lessons from the trenches on reproducible evaluation of language models, in: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024): System Demonstrations, pp. 330–342.

Carlini, N., Liu, C., Erlingsson, U., Kos, J., Song, D., 2019. The secret sharer: Evaluating and testing unintended memorization in neural networks, in: Proceedings of the 28th USENIX Security Symposium, pp. 267–284.

Carlini, N., Traumer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., Rafel, C., 2021. Extracting training data from large language models, in: Proceedings of the 30th USENIX Security Symposium, pp. 2633–2650.

Hsieh, C.P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., Ginsburg, B., 2024. Ruler: What’s the real context size of your longcontext language models?, in: Proceedings of the 2nd Conference on Language Modeling (COLM 2024).

Hu, Y., Wang, H., McAuley, J., 2025. Memoryagentbench: A comprehensive framework for evaluating memory capabilities of llm agents. ArXiv:2507.05257 [cs.AG]; accepted at ICLR 2026.

Killamsetty, K., Sivasubramanian, D., Ramakrishnan, G., De, A., Iyer, R., 2021. Gradmatch: Gradient matching based data subset selection for eficient deep model training, in: Proceedings of the 38th International Conference on Machine Learning (ICML 2021), pp. 5464–5474.

Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., et al., 2023. Holistic evaluation of language models. Transactions on Machine Learning Research (TMLR) Featured Certification.

Maharana, A., Lee, D.H., Tandon, S., et al., 2024. Evaluating very long-term conversational memory of llm agents, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024).

Ndzomga, F., 2026. Eficient benchmarking of ai agents. ArXiv:2603.23749 [cs.AI].

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S.G., Stoica, I., Gonzalez, J.E., 2024. Memgpt: Towards llms as operating systems, in: Proceedings of the 2nd Conference on Language Modeling (COLM 2024).

Shokri, R., Stronati, M., Song, C., Shmatikov, V., 2017. Membership inference attacks against machine learning models, in: Proceedings of the 38th IEEE Symposium on Security and Privacy (S&P 2017), pp. 3–18.

Sumers, T.R., Yao, S., Narasimhan, K., Grifiths, T.L., 2024. Cognitive architectures for language agents (coala). Transactions on Machine Learning Research (TMLR) Survey Certification.

Tan, H., Zhang, Z., Ma, C., Chen, X., Dai, Q., Dong, Z., 2025. Membench: Towards more comprehensive evaluation on the memory of llm-based agents, in: Findings of the Association for Computational Linguistics (ACL 2025), pp. 19336–19352. doi:10.18653/v1/2025.findings-acl.989.

Tavakoli, M., et al., 2025. Beam: A broad-coverage benchmark for evaluating memory in llm agents. ArXiv:2510.27246 [cs.CL].

Weidinger, L., Rauh, M., Marchal, N., Manzini, A., Hendricks, L.A., Mateos-Garcia, J., Bergman, S., Kay, J., Grifin, C., Bariach, B., Gabriel, I., Rieser, V., Isaac, W., 2024. Sociotechnical safety evaluation of generative ai systems, in: Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT 2024), pp. 1120–1137.

Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.W., Yu, D., 2025. Longmemeval: Benchmarking chat assistants on long-term interactive memory. ArXiv:2410.10813 [cs.CL]; accepted at ICLR 2025.

Zhong, W., Guo, L., Gao, Q., Ye, H., Wang, Y., 2024. Memorybank: Enhancing large language models with long-term memory, in: Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI 2024), pp. 19724–19731.