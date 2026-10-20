# BOSS: Token-Budgeted Online Sampling and Selection for LLM Reasoning

Yupeng Chang<sup>1</sup> Yahan Li<sup>4</sup> Chenlu Guo<sup>5</sup> Diana Inkpen<sup>6</sup> Yuan Wu<sup>1</sup>\* Yi Chang<sup>1,2,3</sup>

<sup>1</sup>School of Artificial Intelligence, Jilin University

<sup>2</sup>Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China <sup>3</sup>International Center of Future Science, Jilin University

<sup>4</sup>Xingchen AGI Lab, China Telecom Artificial Intelligence Technology (Beijing) Co., Ltd., China <sup>5</sup>Bilibili Inc., China

<sup>6</sup>School of Electrical Engineering and Computer Science, University of Ottawa, Canada

changyp23@mails.jlu.edu.cn, {yuanwu, yichang}@jlu.edu.cn

liyh181@chinatelecom.cn, guochenlu01@bilibili.com, Diana.Inkpen@uottawa.ca

## Abstract

Test-time scaling can improve LLM reasoning, but existing approaches typically adapt generation or apply answer selection after a predetermined generation schedule. We study within-instance allocation: under a hard perinstance prompt-plus-completion token cap, when should inference stop, generate another candidate, or spend residual tokens adjudicating candidates already observed? BOSS is a training-free policy for this three-way decision, with worst-case budget checks before every generation and judge call. Its key contribution is online generation–adjudication allocation under one shared token ledger. Across two Llama-3.1 models and four reasoning benchmarks, BOSS attains the highest accuracy in 22 of 24 model–task–budget cells under the common hard cap. At Llama-3.1-8B and B = 4k, it reaches 60.5 macro-average accuracy versus 58.6 for Adaptive-Consistency, 59.1 for TALE EP, and 58.9 for fixed-schedule League. On identical Stage-I candidate pools, randomized A/B judging with repeated voting improves plurality from 58.4 to 60.3, with the full BOSS selector reaching 60.5. Approximately matchedcost controls further characterize how the benefit of adjudication changes with available inference budget. Together, these results establish generation–adjudication allocation as a distinct design axis for reasoning under strict perinstance caps. Code: https://anonymous. 4open.science/r/boss-BF00/.

## 1 Introduction

Test-time scaling (TTS) improves LLM reasoning by spending additional inference compute, but the same nominal budget can be used in qualitatively different ways: to generate more candidate solutions, to lengthen or parallelize reasoning, or to compare answers that have already been generated (Wang et al., 2022; Snell et al., 2024; Zhang et al., 2025). This distinction matters when deployment imposes a hard per-instance token cap. A token spent on another reasoning trace cannot also be spent adjudicating among existing answer hypotheses, and comparison calls themselves incur prompt cost because they must include the problem and candidate evidence.

Prior work addresses important parts of this design space, but not the allocation question studied here. Adaptive-Consistency and ESC decide whether further sampling is necessary and then aggregate the sampled answers (Aggarwal et al., 2023; Li et al., 2024). TALE-EP and budgetforcing methods adapt trajectory length or token allowance (Han et al., 2025; Muennighoff et al., 2025). Difficulty-adaptive self-consistency allocates sampling effort across questions, while Parallel-Probe controls width and depth in parallel reasoning (Wang et al., 2025; Zheng et al., 2026). Pairwise selection methods, in contrast, spend testtime compute comparing candidates, but typically do so after a fixed or separately tuned generation schedule (Chen et al., 2026b). These are complementary choices. We focus on the within-instance decision: given a hard per-instance cap, how should one instance allocate its realized token budget online between additional candidate generation and answer adjudication?

We formulate this as a three-way online decision. After each generated candidate, an inference policy may (i) return a sufficiently stable answer, (ii) generate again because useful hypotheses are still emerging, or (iii) redirect the residual budget to adjudicating hypotheses already in the candidate pool. BOSS implements this decision with consensusbased termination, a novelty-based switch criterion, and budget-feasible pairwise judging. All prompt and completion tokens—including the full judge prompt containing the original problem and both representatives—are charged to the same ledger. Before every call, BOSS checks the prompt length plus the maximum completion cap, so adaptive decisions cannot overspend the per-instance budget.

BOSS builds on established stopping, grouping, and judging primitives, but couples them inside a single online generation–adjudication allocator, where selection is an alternative use of the residual budget rather than a fixed post-processing stage. This allocation decision is the methodological contribution. Its effect is directly testable: a development-tuned fixed generation–selection split reaches 59.3 macro-average at Llama-3.1-8B, B = 4k, versus 60.5 for BOSS, and disabling the novelty-triggered switch reaches 59.0.

Our experiments are designed to isolate the value of online allocation. First, we compare against Adaptive-Consistency and TALE-EP under the same strict token ledger. Second, we reuse exactly the same Stage-I candidate pools and change only the selector, separating candidate-generation quality from the value of adjudication. Third, we complement the hard-cap evaluation with realized token consumption and approximately cost-matched operating points. Together, these controls distinguish the effect of the allocator from adaptive stopping, candidate quality, and differences in realized token use. The strongest controlled result is the identical-pool comparison: without changing any Stage-I candidates, randomized repeated judging raises macro accuracy from 58.4 to 60.3, and the full BOSS selector reaches 60.5.

We summarize the contributions as follows:

• A budget-faithful within-instance allocation problem. We formalize generation and adjudication as competing actions under one realized prompt-plus-completion ledger and enforce worst-case feasibility before every model call.

• A three-way online policy. BOSS chooses among returning, continuing candidate generation, and entering answer adjudication based on observed consensus, answer novelty, comparison feasibility, and residual budget.

• Evidence isolating the allocator’s contribution. Beyond hard-cap results on two model scales and four benchmarks, we include adaptive-stopping and token-budget baselines, same-candidate-pool selector controls, realized-cost comparisons, transition statistics, hyperparameter sensitivity, and paired per-instance tests. These studies show that the gains arise from adaptive allocation and adjudication rather than a stronger candidate pool or uncounted selection overhead, while characterizing the budget regimes in which adjudication is most useful.

## 2 Related Work

Adaptive generation. Self-consistency scales inference by sampling multiple traces (Wang et al., 2022). Adaptive-Consistency, ESC, and Stop When Enough study adaptive stopping of sampled or chain-of-thought reasoning (Aggarwal et al., 2023; Li et al., 2024; Sun et al.), while Difficulty-Adaptive Self-Consistency allocates sampling effort across queries using difficulty (Wang et al., 2025). TALE-EP adapts reasoning-token budgets and s1 uses budget forcing to control trajectory length (Han et al., 2025; Muennighoff et al., 2025); Parallel-Probe dynamically regulates width and depth of parallel thinking (Zheng et al., 2026). These methods primarily control generation effort. BOSS instead fixes the per-instance cap and asks when residual tokens should stop generating and begin adjudicating existing answer hypotheses.

Candidate selection. Selection and verification methods spend inference compute comparing candidates (Chen et al., 2026b,a; Lin et al., 2026). LLM judges are flexible but position-sensitive, motivating randomized order and repeated judgments (Zheng et al., 2023; Shi et al., 2025). BOSS uses these established selection primitives differently: selection becomes an online alternative to further generation, with its start and extent determined by the realized answer distribution, novelty trajectory, representative-prompt cost, and residual budget. We test this distinction with a fixed-split ablation and selectors applied to identical Stage-I candidate pools.

Scope. Cross-instance allocation and withininstance generation–adjudication allocation are complementary. BOSS targets the latter: online allocation between additional generation and answer adjudication under a strict per-instance cap.

## 3 Method

## 3.1 Problem Setup and Budget Safety

We study test-time reasoning under a hard perinstance token budget. Given an input instance x, an LLM produces a completion

$$
y \sim p _ { \theta } ( \cdot \mid p ) ,\tag{1}
$$

where $p$ is the prompt. A deterministic parser extracts the final answer

$$
\phi : \mathcal { V } ^ { * }  \mathcal { A } \cup \{ \perp \} ,\tag{2}
$$

where ⊥ denotes parsing failure and is counted as incorrect. We focus on reasoning tasks with deterministically parsable final answers, such as exact-match mathematical reasoning and multiplechoice reasoning.

Inference consists of an adaptive sequence of LLM calls. Call t uses prompt $p _ { t }$ , returns completion $y _ { t }$ , and consumes

$$
c _ { t } = \mathrm { t o k } ( p _ { t } ) + \mathrm { t o k } ( y _ { t } ) ,\tag{3}
$$

where tok(·) is computed with the deployed tokenizer. Under a hard token budget B, every realized inference trajectory must satisfy

$$
\sum _ { t = 1 } ^ { T } c _ { t } \leq B .\tag{4}
$$

Thus, generation and selection calls must share the same prompt-plus-completion token ledger.

Each call is executed with a maximum newtoken cap. Generation calls use cap $L ^ { \mathrm { g e n } }$ , and judge calls use cap $L ^ { \mathrm { j u d g e } }$ . For a prompt $p$ and cap $L ,$ , define the worst-case cost

$$
\overline { { c } } ( p ; L ) = \operatorname { t o k } ( p ) + L .\tag{5}
$$

A call is issued only if the current cumulative cost C satisfies

$$
C + \overline { c } ( p ; L ) \le B .\tag{6}
$$

Because the actual completion length never exceeds the cap, this check guarantees that realized token use does not exceed B.

Allocation state and action space. The budget constraint alone does not specify how the remaining tokens should be used. After observing a candidate pool $S _ { t }$ and cumulative realized cost $C _ { t }$ , we view the policy decision as

$$
u _ { t } \in \{ \mathrm { R E T U R N } , \mathrm { G E N E R A T E } , \mathrm { A D J U D I C A T E } \} .\tag{7}
$$

RETURN terminates with the current answer, GEN-ERATE spends budget on another candidate, and ADJUDICATE freezes the current candidate pool and spends the residual budget on answer-level comparisons. BOSS is one concrete, training-free policy for choosing among these actions. This formulation is useful because it isolates the paper’s allocation question from the specific stopping and judging primitives used to implement it.

## 3.2 BOSS: Budgeted Online Sampling and Selection

We propose BOSS (Budgeted Online Sampling and Selection), a training-free inference policy that adaptively allocates a shared token budget between candidate sampling and answer adjudication. BOSS samples while useful hypotheses are still emerging, returns when a stable consensus forms, and shifts residual budget to randomized-order repeated pairwise judging once recent samples introduce little new answer diversity and adjudication is budget-feasible.

BOSS uses two black-box LLM primitives. A GENERATE call samples a candidate completion and parses its answer; a JUDGE call compares representative completions for two answer hypotheses. By default both use the same base model, so improvements come from test-time allocation rather than an external verifier. The complete step-by-step procedure appears as Algorithm 1 in Appendix A.5.

The policy has three actions. RETURN is appropriate when additional sampling is unlikely to change a stable answer; GENERATE is useful while new answer hypotheses are still appearing; and AD-JUDICATE becomes attractive when recent answer novelty is low, multiple plausible hypotheses remain, and the residual budget can support reliable comparisons. This action-level view is the central design choice of BOSS: the method does not reserve a fixed fraction of the budget for selection in advance.

Why consensus and novelty are separate signals. A high consensus ratio and low recent novelty are not interchangeable. Strong consensus supports

![](images/d39647d98184bb85afd75d351a6a67422ebff03ec44ed15f1f0d1c978e7df68c.jpg)  
Figure 1: Overview of BOSS. Under the same hard cap B, BOSS allocates residual tokens online among RETURN, GENERATE, and ADJUDICATE.

RETURN: the candidate pool already points to one answer, so further sampling or judging is less necessary. Low novelty without consensus describes a different state: repeated sampling is revisiting existing hypotheses, yet no answer is sufficiently dominant. In that regime, spending more tokens on additional traces that map to already observed answer hypotheses can be less useful than comparing the competing hypotheses already available. BOSS therefore uses consensus to decide when the candidate pool is sufficiently stable, and novelty to decide when additional generation is yielding fewer new hypotheses.

Why the switch is gated by feasibility. The switch criterion is coupled to the token ledger rather than evaluated in isolation. BOSS enters adjudication only if the remaining budget can pay for a complete comparison prompt and its capped output; otherwise the selection stage would be infeasible. Conversely, BOSS does not switch merely because two answers exist. When novelty remains high, generation can still expand the hypothesis set, and the policy continues sampling if the next call is feasible. These two gates—evidence that generation is saturating and evidence that comparison is affordable—make the transition budget-aware rather than a fixed stage boundary. They also explain why the method can behave differently across budget regimes without changing hyperparameters.

## 3.3 Stage I: Adaptive Candidate Sampling

BOSS first samples candidate solutions

$$
S _ { n } = \{ ( y _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n } , \qquad a _ { i } = \phi ( y _ { i } ) .\tag{8}
$$

For each parsed answer $a \in A .$ , let $n _ { a }$ be its frequency among valid parsed candidates, and let

$$
n _ { \mathrm { p a r s e d } } = \sum _ { a \in \mathcal { A } } n _ { a } .\tag{9}
$$

Denote the plurality answer by $a ^ { ( 1 ) }$ and the runnerup count by $n ^ { ( 2 ) }$ , with $n ^ { ( 2 ) } = 0$ when fewer than two distinct parsed answers are present.

When $n _ { \mathrm { p a r s e d } } > 0$ , BOSS returns early when one answer becomes sufficiently dominant:

$$
\frac { n _ { a ^ { ( 1 ) } } } { n _ { \mathrm { p a r s e d } } } \geq \tau _ { \mathrm { c o n s } } \quad \mathrm { a n d } \quad n _ { a ^ { ( 1 ) } } - n ^ { ( 2 ) } \geq m _ { \mathrm { g a p } } .\tag{10}
$$

This consensus rule avoids spending additional tokens when repeated sampling already supports a stable answer. If $n _ { \mathrm { p a r s e d } } = 0$ , the consensus condition is false.

If no consensus has formed, BOSS checks whether recent samples are still producing new answer hypotheses. Let

$$
\mathcal { U } _ { \le t } = \{ a _ { i } \in \mathcal { A } : i \le t , a _ { i } \neq \perp \}\tag{11}
$$

be the set of distinct parsed answers observed up to step t. We evaluate the novelty rule after at least W

samples, with $\mathcal { U } _ { \leq 0 } = \emptyset$ for boundary cases. With window size W, define

$$
\operatorname { n o v e l } ( t ) = { \frac { \left| { \mathcal { U } } _ { \leq t } \setminus { \mathcal { U } } _ { \leq ( t - W ) } \right| } { W } } .\tag{12}
$$

BOSS transitions to Stage II when

$$
\operatorname { n o v e l } ( t ) \leq \rho _ { \mathrm { n o v e l } } \quad \mathrm { a n d } \quad | \mathcal { U } _ { \leq t } | \geq 2 ,\tag{13}
$$

provided that at least one complete pairwise comparison is budget-feasible. The rule uses diminishing answer novelty as a lightweight empirical signal that the marginal value of another sample may be lower than the value of adjudicating existing alternatives. Section 4.7 reports transition rates and sensitivity to the switching hyperparameters.

If the novelty-triggered switch has not fired and the next generation call is infeasible, BOSS returns the plurality parsed answer, or ⊥ if no valid parsed answer exists. Thus, budget exhaustion followed by plurality is a safety fallback rather than an implicit trigger for Stage II.

## 3.4 Stage II: Budgeted Pairwise Selection

Stage II operates over parsed answer hypotheses rather than all individual completions. Let

$$
\mathcal { H } _ { x } = \{ a \in \mathcal { A } : n _ { a } > 0 \} .\tag{14}
$$

To control comparison cost, BOSS keeps the $K _ { \mathrm { m a x } }$ most frequent hypotheses:

$$
\mathcal { H }  \mathrm { T o p K } ( \mathcal { H } _ { x } ; K _ { \operatorname* { m a x } } ) .\tag{15}
$$

For each retained answer $a \in \mathcal { H } .$ , BOSS selects a compact representative completion rep(a) from the candidates supporting that answer. Long representatives are truncated while preserving the finalanswer line, reducing comparison-prompt overhead.

BOSS spends the residual budget on pairwise comparisons. To compare two hypotheses a and $b ,$ it presents their representatives to the judge with randomized A/B ordering and repeats the comparison up to R times when budget permits. The winner is determined by majority vote over valid judge outputs, with abstentions ignored. Randomized ordering and repeated voting reduce sensitivity to presentation-order bias in LLM-as-a-judge comparisons (Zheng et al., 2023; Shi et al., 2025).

Because the number of affordable comparisons varies by instance, BOSS schedules comparisons online. It maintains a smoothed win estimate for each active hypothesis,

$$
\hat { p } _ { a } = \frac { w _ { a } + \alpha } { g _ { a } + 2 \alpha } ,\tag{16}
$$

where $w _ { a }$ is the number of wins assigned to answer $a , \ g _ { a }$ is the number of matches involving $^ { a , }$ and α is a smoothing constant. BOSS prioritizes comparisons among hypotheses that remain plausible under the current match outcomes, using lightweight uncertainty-guided scheduling inspired by noisy pairwise preference feedback (Yue et al., 2012; Zoghi et al., 2014). This scheduling rule is used only to allocate the residual budget efficiently, not as a calibrated statistical confidence bound.

When no further worst-case-feasible judge call can be issued, selection stops. BOSS returns the active hypothesis with the highest $\hat { p } _ { a }$ . Remaining ties are broken deterministically by larger Stage-I frequency, shorter representative length, and then lexicographic order. If no pairwise comparison is executed, BOSS falls back to deterministic plurality selection over the sampled candidates.

## 4 Experiments

## 4.1 Experimental Setup

Models. We evaluate BOSS on two openweight instruction-tuned LLMs at two model scales: Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct (Dubey et al., 2024; Hugging Face, 2024; AI@Meta, 2024). Unless otherwise noted, GEN-ERATE and JUDGE use the same base model. Thus, performance differences reflect test-time allocation rather than the use of a stronger external verifier.

Hard-cap protocol and accounting. All methods operate under the same hard per-instance token cap B. We count every prompt and completion token from generation and selection. In particular, each judge prompt contains the original problem plus both representative completions, and the full prompt cost is charged on every replicate. A call is issued only if the remaining budget covers the full prompt plus its maximum completion cap. For parallel methods such as FFS and short-m@k, tokens from all active streams are charged through termination, including unfinished streams that do not provide the returned answer. The main hardcap results use $B \in \{ 2 \mathrm { k } , 4 \mathrm { k } , 8 \mathrm { k } \}$ . Because equal caps do not imply equal realized cost, Section 4.6 separately reports mean realized tokens and approximately cost-matched operating points.

Decoding and judging. GENERATE uses nucleus sampling with temperature $\tau = 0 . 7$ and top-p = 0.95, capped at $L ^ { \mathrm { g e n } }$ = 1024 new tokens. JUDGE uses deterministic decoding with temperature 0, capped at $L ^ { \mathrm { j u d g e } } = 4 .$ and constrained decoding to $\left\{ { \sf A } , { \sf B } \right\}$ . Invalid outputs are treated as abstentions. Unless otherwise stated, each pairwise match uses randomized A/B ordering and majority vote over up to $R = 3$ judge replicates, subject to the remaining token budget. This judging interface is shared by BOSS and all comparison-based baselines.

Development tuning, seeds, and paired analysis. All allocation-sensitive hyperparameters are selected on held-out development subsets disjoint from the reported evaluation data and are then fixed before test evaluation. This protocol is applied uniformly to BOSS and tunable baselines, including the parallelism parameters of FFS and short-m@k, fixed generation–selection splits, and BOSS transition/selection parameters. All stochastic methods use 5 random seeds. Tables report mean accuracy unless otherwise stated. For the closest Llama-3.1-8B comparisons at $B = 4 \mathrm { k }$ , we additionally use paired per-instance tests against the strongest non-BOSS baseline for each benchmark.

## 4.2 Benchmarks and Metrics

We evaluate on four reasoning benchmarks with deterministically parsable final answers: GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), GPQA Diamond (Rein et al., 2024), and MMLU-Pro (Wang et al., 2024). These benchmarks cover grade-school mathematical reasoning, competition-level mathematics, graduate-level scientific reasoning, and multi-domain reasoning.

For GSM8K and MATH, we report exact match (EM) after standard answer normalization, including removal of units when applicable, commas, formatting wrappers, and common LAT<sub>E</sub>X markup artifacts. For GPQA Diamond and MMLU-Pro, we report multiple-choice accuracy. In all cases, the final prediction is obtained through the same task-specific parser ϕ, and parsing failures ⊥ are counted as incorrect. The same parsing and normalization procedures are applied to BOSS and all baselines.

## 4.3 Baselines

We implement all baselines with the same token ledger, decoding caps, parsing interface, and feasibility checks. CoT@1 uses one deterministic generation. Self-Consistency (SC) samples until another call is infeasible and then uses majority voting (Wang et al., 2022); ESC (Li et al., 2024) and Adaptive-Consistency (Aggarwal et al., 2023) adapt when sampling stops but do not redirect residual compute to pairwise adjudication. TALE-EP adapts reasoning length to problem complexity (Han et al., 2025). FFS and short-m@k use parallel generation while charging every active stream (Agarwal et al., 2025; Hassid et al., 2025). Knockout and League are fixed-schedule pairwiseselection baselines (Chen et al., 2026b); they use the same representatives, constrained judge, A/B randomization, repeated voting, and accounting as BOSS. We report Adaptive-Consistency and TALE-EP in the targeted B = 4k analysis under the identical Llama-3.1-8B accounting protocol.

## 4.4 Main Results under Hard Token Caps

Table 1 reports accuracy under the same hard per-instance cap. This is a cap-matched comparison: every method receives the same deployment allowance. BOSS obtains the highest accuracy in 22 of 24 model–task–budget settings. At $B \in \{ 4 \mathrm { k } , 8 \mathrm { k } \}$ , it is best in every column across both model scales and all four benchmarks; at 2k, it leads 6 of 8 model–task cells, with FFS leading the two GSM8K cells. The pattern persists across model scale, and the clearest margins occur on MATH and GPQA-Diamond, where competing answer hypotheses leave more room for adjudication than on the more saturated GSM8K benchmark.

For Llama-3.1-8B at $B ~ = ~ 4 \mathrm { { k } }$ , paired perinstance tests against the strongest non-BOSS baseline on each benchmark yield improvements of +0.9 points on GSM8K $( p \ = \ 0 . 0 2 1 )$ , +2.2 on MATH $( p \ < \ 0 . 0 0 1 )$ , +1.7 on GPQA-Diamond $( p = 0 . 0 2 6 )$ , and +1.5 on MMLU-Pro $( p < 0 . 0 0 1 )$ All four paired comparisons are significant at p < 0.05; Appendix B reports the corresponding accuracy intervals.

## 4.5 Targeted Controls for Allocation and Selection

To separate the sources of the hard-cap gains, Table 2 combines complementary controls at Llama-3.1-8B, B = 4k. Adaptive-Consistency and TALE-EP test alternative adaptive uses of inference tokens, while the fixed-split and no-novelty variants test the online transition policy. The final block holds the entire Stage-I candidate pool fixed and changes only the selector, directly separating candidate quality from selection quality.

Table 1: Accuracy (%) under fixed per-instance token budgets B. The budget counts all prompt and completion tokens consumed by each inference policy. Values are averaged over 5 random seeds. CoT@1 (Greedy) is budgetinvariant because one deterministic generation uses far fewer tokens than the smallest budget (2k); identical values across budgets are shown in merged cells. GPQA-D denotes GPQA-Diamond. Bold indicates the best result within each model block and column.
<table><tr><td></td><td colspan="3">GSM8K (EM)</td><td colspan="3">MATH (EM)</td><td colspan="3">GPQA-D (Acc)</td><td colspan="3">MMLU-Pro (Acc)</td></tr><tr><td>Method</td><td>2k</td><td>4k</td><td>8k</td><td>2k</td><td>4k</td><td>8k</td><td>2k</td><td>4k</td><td>8k</td><td>2k</td><td>4k</td><td>8k</td></tr><tr><td colspan="9">Base model: Llama-3.1-8B-Instruct</td><td></td><td></td><td></td><td></td></tr><tr><td>CoT@1 (Greedy)</td><td></td><td>84.5</td><td></td><td></td><td>51.9</td><td></td><td></td><td>30.4</td><td></td><td></td><td>48.3</td><td></td></tr><tr><td>SC (budgeted)</td><td>85.8</td><td>87.3</td><td>88.1</td><td>54.7</td><td>56.6</td><td>58.0</td><td>33.1</td><td>35.7</td><td>37.0</td><td>49.9</td><td>51.2</td><td>52.2</td></tr><tr><td>ESC (Li et al., 2024)</td><td>86.2</td><td>87.6</td><td>88.3</td><td>55.3</td><td>56.8</td><td>58.4</td><td>33.5</td><td>36.1</td><td>37.2</td><td>50.3</td><td>51.6</td><td>52.5</td></tr><tr><td>FFS (Agarwal et al., 2025)</td><td>86.9</td><td>87.2</td><td>87.5</td><td>54.5</td><td>55.7</td><td>56.0</td><td>32.2</td><td>33.3</td><td>33.8</td><td>49.5</td><td>50.1</td><td>50.8</td></tr><tr><td>short-m@k (Hassid et al., 2025)</td><td>86.5</td><td>87.9</td><td>88.5</td><td>55.0</td><td>57.1</td><td>58.7</td><td>33.4</td><td>35.9</td><td>37.5</td><td>50.1</td><td>51.8</td><td>53.0</td></tr><tr><td>Knockout (Chen et al., 2026b)</td><td>84.3</td><td>87.1</td><td>88.7</td><td>51.8</td><td>56.4</td><td>59.2</td><td>31.2</td><td>37.0</td><td>40.1</td><td>48.8</td><td>52.3</td><td>54.4</td></tr><tr><td>League (Chen et al., 2026b)</td><td>84.8</td><td>87.5</td><td>89.2</td><td>52.6</td><td>57.2</td><td>59.8</td><td>31.7</td><td>37.8</td><td>40.8</td><td>49.4</td><td>52.9</td><td>55.1</td></tr><tr><td>BOŠS (Ours)</td><td>86.6</td><td>88.8</td><td>89.9</td><td>56.1</td><td>59.4</td><td>61.3</td><td>34.4</td><td>39.5</td><td>42.3</td><td>51.2</td><td>54.4</td><td>56.3</td></tr><tr><td colspan="9">Base model: Llama-3.1-70B-Instruct</td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td></tr><tr><td>CoT@1 (Greedy)</td><td></td><td>95.1</td><td></td><td></td><td>68.0</td><td></td><td>46.7</td></tr><tr><td>SC (budgeted)</td><td>95.9</td><td>96.5</td><td>96.9</td><td>71.1</td><td>72.8</td><td>73.6</td><td>48.8</td><td>50.4 51.3</td><td>68.3</td></tr><tr><td>ESC (Li et al., 2024)</td><td>96.1</td><td>96.8</td><td>97.1</td><td>71.4</td><td>73.1</td><td>73.9</td><td>49.2</td><td>50.7</td><td>51.6</td><td>68.5</td><td>69.8 70.0</td><td>70.6 70.9</td></tr><tr><td>FFS (Agarwal et al., 2025)</td><td>96.5</td><td>96.6</td><td>96.7</td><td>70.6</td><td>71.7</td><td>72.2</td><td>48.3</td><td>49.1</td><td>49.6</td><td>67.8</td><td>68.5</td><td>69.0</td></tr><tr><td>short-m@k (Hassid et al., 2025)</td><td>96.2</td><td>96.9</td><td>97.3</td><td>71.7</td><td>73.4</td><td>74.6</td><td>49.1</td><td>51.0</td><td>51.9</td><td>68.8</td><td>70.2</td><td>71.3</td></tr><tr><td>Knockout (Chen et al., 2026b)</td><td>95.0</td><td>96.4</td><td>97.2</td><td>69.1</td><td>73.6</td><td>75.4</td><td>47.8</td><td>51.5</td><td>53.1</td><td>67.6</td><td>71.0</td><td>72.6</td></tr><tr><td>League (Chen et al., 2026b)</td><td>95.2</td><td>96.7</td><td>97.5</td><td>69.8</td><td>74.3</td><td>76.1</td><td>48.5</td><td>52.1</td><td>53.7</td><td>68.1</td><td>71.6</td><td>73.2</td></tr><tr><td>BOSS (Ours)</td><td>96.3</td><td>97.2</td><td>97.7</td><td>72.3</td><td>75.5</td><td>77.6</td><td>50.4</td><td>52.9</td><td>54.6</td><td>69.5</td><td>72.1</td><td>73.8</td></tr></table>

Table 2: Targeted controls at Llama-3.1-8B-Instruct and B = 4k. “Macro” averages all four benchmarks. Samepool rows reuse exactly the same Stage-I candidates and change only the selector.
<table><tr><td>Method / control</td><td>Macro (%)</td></tr><tr><td>Adaptive-Consistency (Aggarwal et al., 2023) TALE-EP (Han et al., 2025)</td><td>58.6</td></tr><tr><td>League (fixed schedule)</td><td>59.1</td></tr><tr><td>BOSS</td><td>58.9 60.5</td></tr><tr><td></td><td></td></tr><tr><td>No novelty transition Fixed generation-selection split</td><td>59.0 59.3</td></tr><tr><td></td><td></td></tr><tr><td>Same pool: plurality</td><td>58.4</td></tr><tr><td>Same pool: randomized A/B + repeated Same pool: full BOSS selector</td><td>60.3 60.5</td></tr></table>

The same-pool rows isolate selection from generation: changing only the selector raises macro accuracy from 58.4 with plurality to 60.3 with randomized A/B repeated voting, while the full BOSS selector reaches 60.5. Among cases where plurality is wrong but the correct hypothesis is already present, repeated judging recovers 22.8% of errors with a 2.0% harm rate on initially correct plurality cases; the full BOSS selector reaches 25.1% recovery with 2.2% harm. The fixed-split and nonovelty controls further show that, in this setting, online allocation contributes beyond a static split or a simpler switching rule. Detailed selector variants, recovery/harm analysis, and the full component ablation are reported in Appendix B. Together, these controls isolate the source of the gain: BOSS decides when residual tokens are better spent on adjudication than on another sample, while randomized repeated judging realizes most of the same-pool selection improvement.

## 4.6 Hard Caps versus Realized Token Cost

A common hard cap equalizes the deployment constraint, not realized token use, because adaptive methods can stop early. We therefore complement the hard-cap results with approximately matched mean-token comparisons (Table 3). At the two higher-cost operating points, BOSS improves over TALE-EP by 0.3 points around 3.1k tokens and over League by 1.4 points around 3.85k; around 2.5k tokens, Adaptive-Consistency is ahead by 0.2 points. These operating points characterize a budget-dependent trade-off: BOSS is most effective when the available compute can support both candidate exploration and repeated adjudication.

Table 3: Approximately cost-matched operating points for Llama-3.1-8B.
<table><tr><td>Method</td><td>Mean tokens</td><td>Macro (%)</td></tr><tr><td>BOSS</td><td>≈2,500</td><td>58.4</td></tr><tr><td>Adaptive-Consistency</td><td>2,508</td><td>58.6</td></tr><tr><td>BOSS</td><td>≈3,100</td><td>59.4</td></tr><tr><td>TALE-EP</td><td>3,074</td><td>59.1</td></tr><tr><td>BOSS</td><td>≈3,850</td><td>60.3</td></tr><tr><td>League</td><td>3,853</td><td>58.9</td></tr></table>

Appendix B reports the full $\begin{array} { r l r } { B } & { { } = } & { 4 \mathrm { k } } \end{array}$ realized-cost audit across methods and the generation/adjudication allocation across budgets. At the common 4k cap, BOSS uses 3,947 mean tokens and attains the highest macro accuracy in that audit (60.5), clarifying that its primary objective is productive use of the deployment allowance rather than minimum average token consumption.

## 4.7 Why and When Does BOSS Switch?

The transition statistics show the intended budgetadaptive shift from generation to adjudication: Stage-II entry rises from 31.8% at $B ~ = ~ 2 \mathrm { k }$ to 84.1% at 4k and 94.1% at 8k, while the safetyfallback rate falls from 65.6% to 12.4% and 1.2%. Generation use grows only modestly as the cap increases, whereas most additional budget is redirected to judging; Appendix Table 11 gives the full accounting.

Performance remains stable across nearby transition settings. For $\rho _ { \mathrm { n o v e l } } \in \{ 0 . 1 2 5 , 0 . 2 5 0 , 0 . 3 7 5 \}$ macro accuracy is 60.1, 60.5, and 60.3, respectively; across the tested W and $\tau _ { \mathrm { c o n s } }$ ranges, it remains within 59.9–60.5. Appendix B reports the full sensitivity sweeps and token-allocation audit.

## 4.8 Interpretation

Taken together, these results support three conclusions. First, allocation matters even when the generator and judge are the same base model, so the hardcap gains arise without a stronger external verifier. Second, adjudication contributes beyond candidate generation itself: holding the Stage-I pool fixed improves over plurality, while the fixed-split and no-novelty controls underperform the online policy. Third, adjudication has the greatest value when the available budget can support repeated comparison: BOSS leads at the two higher-cost matched operating points, while Adaptive-Consistency is ahead by 0.2 points in the lowest-cost neighborhood. BOSS therefore uses adjudication selectively when the observed candidate state and remaining budget justify its cost.

Strict accounting is central to this interpretation. Each comparison repeatedly re-encodes the problem and representative solutions, so its prompt cost is part of the allocation decision. Charging those tokens to the same ledger makes the comparison budget-faithful and ensures that the observed gains reflect how the allowance is allocated between generation and adjudication. The matched-cost results then clarify the regime in which that allocation is most useful.

Evidence coverage. Table 1 provides the broad hard-cap comparison across both Llama-3.1 model scales and all four benchmarks, testing whether the allocation advantage persists across task, model scale, and budget. The targeted selector controls, recovery/harm analysis, approximately matched-cost operating points, and transition sensitivity are diagnostic studies centered on Llama-3.1-8B at $B = 4 \mathrm { k }$ unless stated otherwise. Their role is complementary: they isolate the allocation mechanism, quantify when adjudication corrects plurality errors, and characterize how the controller responds to available budget. The resulting empirical conclusion is about within-instance generation–adjudication allocation under strict per-instance caps.

## 5 Conclusion

We studied a central allocation problem in testtime reasoning: under one strict per-instance token cap, when should residual compute be spent on another candidate and when should it be spent adjudicating candidates already observed? BOSS implements a budget-safe three-way policy—return, generate, or adjudicate—using established stopping and judging primitives. Under hard caps, it is strongest in 22 of 24 model–task–budget settings in our evaluation. Same-pool controls show that the gain extends beyond candidate generation: randomized A/B judging with repeated voting raises macro accuracy from 58.4 with plurality to 60.3, and the full BOSS selector reaches 60.5. Approximately matched-cost controls further show that the value of adjudication grows with the budget available for comparison. Together, these results establish generation–adjudication allocation as a distinct and empirically useful design axis for budgetfaithful test-time reasoning. More broadly, they suggest that test-time scaling under deployment caps should treat answer selection as a first-class budgeted action, rather than a fixed post-processing stage.

![](images/10b18e44b4d5884126bdc4e891ec76cf3e41dd97c3641cd5ea18068a7ad7deaf.jpg)  
(a) Within-instance Allocation

![](images/170c5b0d03b9dcab9e95aaaa88648d5043a4a2209eab1a3f78850efde761aded.jpg)  
(b) Position Sensitivity of Judging

![](images/6c53aeefc03dd2e82242400840de18967f4d092d4b2b427b3cd415913e97f70a.jpg)  
(c) Generation vs. Judging Success

![](images/dea241e79d76fbdea056c60182e5fff8aa09dda70b373c82ec6b0dd321559edc.jpg)  
(d) Hard-Cap Scaling (Mean ± SEM)  
Figure 2: Mechanism diagnostics for BOSS. (a) realized generation–adjudication allocation; (b) judge-order sensitivity; (c) generation versus judging success; (d) hard-cap scaling (cap-matched).

## Limitations

The current evaluation covers Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct on GSM8K, MATH, GPQA-Diamond, and MMLU-Pro; extending the study to newer reasoning-specialized model families, recent mathematics benchmarks, coding, and agentic tasks is future empirical work. BOSS assumes deterministically parsable answer hypotheses, so open-ended generation would require semantic clustering or equivalence judgments before answer-level adjudication. We study withininstance allocation under a fixed per-instance cap, complementary to distributing a global budget across questions of different difficulty (Wang et al., 2025). Token count is also a deployment proxy rather than a complete serving-cost model: parallelism, prefill/decode asymmetry, cache reuse, and hardware utilization can change latency and monetary cost. Finally, the switching rule is empirically tuned rather than theoretically optimal, and successful adjudication depends on the candidate pool containing a correct hypothesis and the judge distinguishing close alternatives.

## References

Aradhye Agarwal, Ayan Sengupta, and Tanmoy Chakraborty. 2025. First finish search: Efficient testtime scaling in large language models. arXiv preprint arXiv:2505.18149.

Pranjal Aggarwal, Aman Madaan, Yiming Yang, and Mausam. 2023. Let’s sample step by step: Adaptiveconsistency for efficient reasoning and coding with LLMs. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12375–12396. Association for Computational Linguistics.

AI@Meta. 2024. Llama 3 model card.

Hao Chen, Guanxi Lu, Yasuyuki Okoshi, Zhiwen Mo, Masato Motomura, and Hongxiang Fan. 2026a. Rethinking optimal verification granularity for computeefficient test-time scaling. Advances in Neural Information Processing Systems, 38:88650–88674.

Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, and Jingren Zhou. 2026b. Provable scaling laws for the test-time compute of large language models. In Advances in Neural Information Processing Systems, volume 38, pages 167074–167108.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. 2025. Tokenbudget-aware LLM reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 24842–24855. Association for Computational Linguistics.

Michael Hassid, Gabriel Synnaeve, Yossi Adi, and Roy Schwartz. 2025. Don’t overthink it. preferring shorter thinking chains for improved llm reasoning. arXiv preprint arXiv:2505.17813.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Hugging Face. 2024. meta-llama/llama-3.1-8b-instruct.

Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Xinglin Wang, Bin Sun, Heda Wang, and Kan Li. 2024. Escape sky-high cost: Early-stopping selfconsistency for multi-step reasoning. arXiv preprint arXiv:2401.10480.

Fangzhou Lin, Shuo Xing, Peiran Li, Siyuan Yang, Qianwen Ge, Kazunori Yamada, Ziming Zhang, Haichong Zhang, and Zhengzhong Tu. 2026. Caps: Cascaded adaptive pairwise selection for efficient parallel reasoning. arXiv preprint arXiv:2605.15513.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20275–20321. Association for Computational Linguistics.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. 2025. Judging the judges: A systematic study of position bias in llmas-a-judge. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics, pages 292–314.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314.

Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, and Wei Wang. Stop when enough: Adaptive earlystopping for chain-of-thought reasoning, 2025. URL https://arxiv. org/abs/2510.10103.

Xinglin Wang, Shaoxiong Feng, Yiwei Li, Peiwen Yuan, Yueqi Zhang, Chuyi Tan, Boyuan Pan, Yao Hu, and Kan Li. 2025. Make every penny count: Difficultyadaptive self-consistency for cost-efficient reasoning. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 6919–6932. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, volume 37, pages 95266–95290.

Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. 2012. The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556.

Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Zhihan Guo, Yufei Wang, Irwin King, Xue Liu, and Chen Ma. 2025. What, how, where, and how well? a survey on test-time scaling in large language models. CoRR.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623.

Tong Zheng, Chengsong Huang, Runpeng Dai, Yun He, Rui Liu, Xin Ni, Huiwen Bao, Kaishen Wang, Hongtu Zhu, Jiaxin Huang, Furong Huang, and Heng Huang. 2026. Parallel-probe: Towards efficient parallel thinking via 2D probing. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 163164–163178. PMLR.

Masrour Zoghi, Shimon Whiteson, Remi Munos, and Maarten de Rijke. 2014. Relative upper confidence bound for the k-armed dueling bandit problem. In International Conference on Machine Learning, pages 10–18. PMLR.

## A Additional Experimental Details

This appendix provides additional implementation details for the fixed-budget evaluation, including answer parsing, prompt interfaces, budget accounting, baseline implementation, BOSS scheduling details, and hyperparameter selection.

## A.1 Answer Parsing and Normalization

All methods use the same deterministic parser ϕ defined in §3. Parsing failures are mapped to ⊥ and counted as incorrect in all metrics. For GSM8K and MATH, we apply standard exact-match normalization to the parsed answer string, including removal of surrounding whitespace, commas, units when applicable, formatting wrappers, and common LAT<sub>E</sub>X artifacts. For GPQA-Diamond and MMLU-Pro, we parse a single multiple-choice option; outputs with no valid option or ambiguous option extraction are mapped to ⊥. These procedures are applied uniformly to BOSS and all baselines.

## A.2 Prompt Interfaces

Generation prompt. GENERATE uses a fixed instruction-following template that presents the input instance x and asks the model to produce a reasoning trace followed by a final answer in a standardized format required by the parser ϕ. The same generation template is used by all methods that perform deterministic or stochastic generation, so performance differences are attributable to inference allocation rather than prompt engineering.

Judge prompt. JUDGE uses a fixed comparison template

$$
p ^ { \mathrm { c m p } } ( x , \mathrm { r e p } ( a ) , \mathrm { r e p } ( b ) ) ,\tag{17}
$$

which presents the input instance and two representative candidate solutions as A and B. The judge is asked to choose which candidate is more likely to contain the correct final answer. Judge decoding is deterministic with temperature 0 and constrained to output a single token in {A, B}. Any invalid output is treated as an abstention. The same judge interface is used by BOSS and all comparison-based baselines.

## A.3 Strict Budget Accounting

All methods are evaluated under a hard per-instance token budget B that counts all prompt and completion tokens. Token counts are computed with the deployed tokenizer of the corresponding base model. Each model call is executed with a maximum new-token cap. Before issuing a call with prompt p and cap L, we compute the worst-case cost

$$
\overline { { c } } ( p ; L ) = \operatorname { t o k } ( p ) + L .\tag{18}
$$

The call is issued only if

$$
C + \overline { c } ( p ; L ) \le B ,\tag{19}
$$

where C is the cumulative realized cost so far. Because the realized completion length never exceeds the cap, this feasibility check guarantees that the hard token budget is never violated.

Parallel-stream accounting. For methods that use concurrent generations, namely FFS and shortm@k, token costs are accumulated across all active streams until the method terminates. Tokens consumed by streams that do not provide the returned output are still charged to the same per-instance budget.

Stage-transition feasibility. BOSS enters Stage II only when the novelty-triggered switch fires, at least two distinct parsed answer hypotheses are available, and the remaining budget can support at least one worst-case-feasible judge call. If the switch has not fired and another generation call becomes infeasible, BOSS returns the Stage-I plurality answer, or ⊥ if no valid parsed answer exists.

Tie handling. All ties are resolved deterministically to avoid hidden stochasticity beyond explicit model sampling and randomized A/B presentation. Stage-I plurality ties are broken by shorter supporting completion length and then lexicographic order. Final Stage-II ties are broken by larger Stage-I frequency, shorter representative length, and then lexicographic order.

## A.4 Baseline Implementation

All baselines use the same token accounting, decoding caps, parsing interface, and budget-feasibility checks as BOSS.

CoT@1 (Greedy). CoT@1 performs one deterministic generation with temperature 0 using the shared generation prompt. If the hard budget is insufficient for the worst-case cost of the call, the method returns ⊥.

Self-Consistency. Self-Consistency repeatedly samples candidate completions until the next generation call is no longer budget-feasible. The final answer is selected by majority vote over parsed non-⊥ answers, with the same deterministic tiebreaking rule as BOSS.

ESC. ESC applies an answer-convergence earlystopping rule under the same budget, generation cap, prompt template, and parsing interface. It stops once the parsed answer distribution is sufficiently stable, but does not use pairwise judging.

First Finish Search. FFS launches k parallel generation streams and returns the first completed output. All tokens consumed by active streams up to termination are counted toward the shared per-instance budget.

short-m@k. short-m@k launches k parallel generation streams, stops when the first m completions finish, and majority-votes among those m outputs. Tokens consumed by all active streams before the m-th completion are charged to the budget.

Knockout and League. Knockout and League are fixed-schedule pairwise-comparison baselines. They use the same representative-based comparison prompt, constrained judge decoding, randomized A/B ordering, repeated voting protocol, decoding caps, and token accounting as BOSS. Their generation–selection split is fixed after development tuning, whereas BOSS decides online when to transition from sampling to selection.

Algorithm 1 BOSS: Budgeted Online Sampling   
and Selection   
Require: input x, token budget B, caps $L ^ { \mathrm { g e n } }$ <sup>n</sup>, L<sup>judge</sup>   
Ensure: final answer aˆ   
$1 \colon C \gets 0 , S \gets \emptyset , q _ { \mathrm { s e l } } \gets$ false   
Stage I: Adaptive candidate sampling   
2: while FEASIBL $\operatorname { E } ( C , p ^ { \mathrm { g e n } } ( x ) , L ^ { \mathrm { g e n } ^ { \bullet } } , B )$ do   
3: y ← GENERATE $( x , L ^ { \mathrm { g e n } } )$   
4: $C \gets C + \mathrm { t o k } ( p ^ { \mathrm { g e n } } ( x ) ) \dot { + }$ tok(y)   
5: $S \gets S \cup \{ ( y , \ddot { \phi } ( y ) ) \}$   
6: if CONSENSUS(S) then   
7: return PLURALITY(S)   
8: end if   
9: if SWITCHTOSELECT(C, S, B) then   
10: q ← true; break   
11: end if   
12: end while   
13: if not $q _ { \mathrm { s e l } }$ then   
14: return PLURALITYORFAIL(S)   
15: end if   
Stage II: Budgeted pairwise selection   
16: $\mathcal { H } $ TOPKANSWERS(S, K<sub>max</sub>)   
17: i $\mathbf { f } \left| \mathcal { H } \right| < 2$ then   
18: return PLURALITYORFAIL(S)   
19: end if   
20: for $a \in \mathcal H$ do   
21: rep(a) ← REPRESENTATIVE(a, S)   
22: end for   
23: if not CANMATCH(C, H, rep, B) then   
24: return PLURALITYORFAIL(S)   
25: end if   
26: $\mathcal { H } _ { \mathrm { a c t } }  \mathcal { H } ;$ initialize match statistics   
27: while $| { \mathcal { H } } _ { \mathrm { a c t } } | \geq 2$ do   
28: if not CANMATCH(C, H , rep, B) then   
29: break   
30: end if   
31: (a, b) ← SELECTPAIR(H<sub>act</sub>)   
32: $( w , \dot { C } ) \gets$ JUDGEMATCH(a, b, rep, R, C, B)   
33: Update match statistics with winner w   
34: $\hat { \mathcal { H } } _ { \mathrm { a c t } }  \mathrm { P R U N E } ( \mathcal { H } _ { \mathrm { a c t } } )$   
35: end while   
36: return BESTANSWER $( { \mathcal { H } } _ { \mathrm { a c t } } , S )$

## A.5 Detailed BOSS Scheduling Rule

The main text describes BOSS’s stages at a high level. Algorithm 1 gives the complete budget-safe procedure; the equations below specify the Stage-II scheduling and pruning rule used in our implementation. In the pseudocode, SWITCHTOSELECT combines the low-novelty transition with Stage-II feasibility, CANMATCH applies the worst-case budget check to the current comparison prompts, and JUDGEMATCH uses randomized A/B ordering with up to R repeated votes.

After Stage I, BOSS constructs the hypothesis set

$$
\mathcal { H } _ { x } = \{ a \in \mathcal { A } : n _ { a } > 0 \}\tag{20}
$$

and retains the $K _ { \mathrm { m a x } }$ most frequent parsed answers:

$$
\mathcal { H }  \mathrm { T o p K } ( \mathcal { H } _ { x } ; K _ { \operatorname* { m a x } } ) .\tag{21}
$$

For each retained answer $a \in \mathcal H$ , BOSS selects the shortest supporting completion with at least $L _ { \mathrm { m i n } } ^ { \mathrm { r e p } }$ tokens; if none qualifies, it uses the shortest supporting completion. Representatives longer than $L ^ { \mathrm { r e p } }$ tokens are truncated while preserving the final-answer line.

For each active hypothesis $^ { a , }$ BOSS maintains a win count $w _ { a }$ and a match count $g _ { a }$ , initialized to zero. After a match between a and b, both match counts increase by one. The winner receives one win; if the match is tied, each answer receives half a win. BOSS computes the smoothed win estimate

$$
\hat { p } _ { a } = \frac { w _ { a } + \alpha } { g _ { a } + 2 \alpha } ,\tag{22}
$$

and the lightweight uncertainty radius

$$
r _ { a } = \sqrt { \frac { \log ( 2 | \mathcal { H } | / \delta ) } { 2 ( g _ { a } + 1 ) } } .\tag{23}
$$

It then defines

$$
\mathrm { U C B } _ { a } = \operatorname* { m i n } ( 1 , \hat { p } _ { a } + r _ { a } ) ,
$$

$$
\mathrm { L C B } _ { a } = \operatorname* { m a x } ( 0 , \hat { p } _ { a } - r _ { a } ) .\tag{24}
$$

(25)

These quantities are used only as scheduling and pruning heuristics, not as calibrated confidence bounds.

Let ${ \mathcal { H } } _ { \mathrm { a c t } }$ be the active hypothesis set. While at least two active hypotheses remain and another judge call is budget-feasible, BOSS selects the two distinct active hypotheses with the largest UCB values and compares them using randomized A/B ordering with repeated voting. After updating match statistics, BOSS prunes clearly dominated hypotheses:

$$
\ell ^ { \star } = \operatorname* { m a x } _ { u \in \mathcal { H } _ { \mathrm { a c t } } } \mathrm { L C B } _ { u } ,\tag{26}
$$

$$
\mathcal { H } _ { \mathrm { a c t } }  \mathcal { H } _ { \mathrm { a c t } } \backslash \{ u : \mathrm { U C B } _ { u } < \ell ^ { \star } \} .\tag{27}
$$

If pruning would remove all active hypotheses, BOSS retains the answer with the largest prepruning UCB. When no further budget-feasible comparison is possible, BOSS returns the active hypothesis with the largest $\hat { p } _ { a }$ , using the deterministic tie-breaking rule described above.

## A.6 Hyperparameter Selection

All allocation-sensitive hyperparameters are selected on held-out development subsets that are disjoint from the reported evaluation data. This protocol is applied uniformly to BOSS and all tunable baselines. It covers the parallelism parameters k and m for FFS and short-m@k, the fixed generation–selection splits for Knockout and League, and the stage-transition and selection parameters of BOSS. No reported test instances are used for hyperparameter selection.

Table 4: BOSS hyperparameters. Values are fixed after development tuning and are not selected using reported test instances.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Generation cap  $L ^ { \mathrm { g e n } }$ </td><td>1024</td></tr><tr><td>Judge cap  $L ^ { \mathrm { j u \bar { d g } e } }$ </td><td>4</td></tr><tr><td>Judge replicates R Consensus threshold  $\tau _ { \mathrm { c o n s } }$ </td><td>3 0.70</td></tr><tr><td>Consensus gap  $m _ { \mathrm { g a p } }$ </td><td>2</td></tr><tr><td>Novelty window  $\bar { W }$  Novelty threshold  $\rho _ { \mathrm { n o v e l } }$ </td><td>4</td></tr><tr><td>Maximum retained hypotheses</td><td>0.25</td></tr><tr><td> $K _ { \mathrm { m a x } }$ </td><td>5</td></tr><tr><td>Representative minimum length  $L _ { \mathrm { m i n } } ^ { \mathrm { r e p } }$ </td><td>32</td></tr><tr><td> $L ^ { \mathrm { r e p } }$ </td><td></td></tr><tr><td>Representative cap</td><td>256</td></tr><tr><td>Smoothing constant α</td><td>1.0</td></tr><tr><td>Scheduling parameter δ</td><td>0.05</td></tr></table>

Table 4 lists the BOSS hyperparameters used in our experiments. The generation and judge decoding caps are fixed across benchmarks, while the allocation thresholds are selected on development data and then fixed before test evaluation.

## A.7 Random Seeds and Reporting

All stochastic methods are evaluated with 5 random seeds. Tables report mean accuracy over seeds unless otherwise stated. When error bars are shown in figures, they denote mean ± standard error of the mean (SEM). Deterministic components, including parsing, tie-breaking, budget-feasibility checks, and constrained judge decoding, are kept fixed across methods.

## A.8 Scope of the Evaluation

Our evaluation focuses on reasoning benchmarks with deterministically parsable final answers and a fixed per-instance deployment cap. This scope matches the design of BOSS, which groups candidate completions by parsed answer hypotheses and performs answer-level adjudication. Crossinstance budget allocation, semantic clustering for open-ended outputs, and evaluation on newer reasoning-specialized model families are outside the current evidence.

## B Additional Diagnostics

The following analyses provide supporting evidence for the main-text controls without changing the central claim.

## B.1 Statistical Uncertainty at the Closest Setting

Table 5: Llama-3.1-8B-Instruct at $B = 4 \mathrm { { k } : }$ mean accuracy (%) with 95% confidence intervals, plus paired per-instance p-values against the strongest non-BOSS baseline for each benchmark.
<table><tr><td>Benchmark</td><td>Baseline</td><td>Baseline (95% CI)</td><td>BOSS (95% CI)</td><td>Paired p</td></tr><tr><td>GSM8K</td><td>short-m@k</td><td>87.9 [87.5, 88.3]</td><td>88.8 [88.4, 89.2]</td><td>0.021</td></tr><tr><td>MATH</td><td>League</td><td>57.2 [56.7, 57.7]</td><td>59.4 [58.8, 60.0]</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>GPQA-D</td><td>League</td><td>37.8 [36.9, 38.7]</td><td>39.5 [38.5, 40.5]</td><td>0.026</td></tr><tr><td>MMLU-Pro</td><td>League</td><td>52.9 [52.5, 53.3]</td><td>54.4 [53.9, 54.9]</td><td>&lt; 0.001</td></tr></table>

The intervals quantify uncertainty in the methodlevel accuracy estimates, while the p-values use paired per-instance predictions. We report paired tests for the closest Llama-3.1-8B, B = 4k comparisons, where the margins to the strongest baselines are smallest.

## B.2 All-Method Realized-Cost Audit at B=4k

Table 6: Llama-3.1-8B-Instruct at the common $B = 4 \mathrm { k }$ cap. Token statistics are realized per-instance totals; Macro averages the four benchmarks.
<table><tr><td>Method</td><td>Mean</td><td>Median</td><td>P90</td><td>Macro (%)</td></tr><tr><td>CoT@1</td><td>846</td><td>779</td><td>1330</td><td>53.8</td></tr><tr><td>FFS</td><td>2168</td><td>2044</td><td>3312</td><td>56.6</td></tr><tr><td>Adaptive-Consistency</td><td>2508</td><td>2365</td><td>3868</td><td>58.6</td></tr><tr><td>ESC</td><td>2736</td><td>2584</td><td>3924</td><td>58.0</td></tr><tr><td>TALE-EP</td><td>3074</td><td>3012</td><td>3968</td><td>59.1</td></tr><tr><td>short-m@k</td><td>3246</td><td>3330</td><td>3984</td><td>58.2</td></tr><tr><td>SC</td><td>3721</td><td>3862</td><td>3995</td><td>57.7</td></tr><tr><td>Knockout</td><td>3809</td><td>3894</td><td>3996</td><td>58.2</td></tr><tr><td>League</td><td>3853</td><td>3926</td><td>3997</td><td>58.9</td></tr><tr><td>BOSS</td><td>3947</td><td>3981</td><td>3999</td><td>60.5</td></tr></table>

This audit makes the cost trade-off explicit: BOSS uses a larger share of the common 4k allowance, consistent with its objective of reallocating residual budget rather than minimizing mean token use. The primary hard-cap comparison therefore measures how effectively methods use a shared deployment allowance, while Table 3 provides the complementary approximately matched-cost view.

Table 7: Mean realized token utilization over 5 random seeds. All prompt and completion tokens are counted.
<table><tr><td></td><td colspan="5">League (Fixed)</td><td colspan="5">BOSS (Adaptive)</td></tr><tr><td>B</td><td> $\mathrm { G e n . }$ </td><td>Judge</td><td>Total</td><td>Util.%</td><td>Judge%</td><td>Gen.</td><td>Judge</td><td>Total</td><td>Util.%</td><td>Judge%</td></tr><tr><td>1k</td><td>523</td><td>414</td><td>937</td><td>93.7</td><td>44.2</td><td>958</td><td>0</td><td>958</td><td>95.8</td><td>0.0</td></tr><tr><td>2k</td><td>812</td><td>1103</td><td>1915</td><td>95.8</td><td>57.6</td><td>1445</td><td>492</td><td>1937</td><td>96.9</td><td>25.4</td></tr><tr><td>4k</td><td>1634</td><td>2219</td><td>3853</td><td>96.3</td><td>57.6</td><td>1903</td><td>2044</td><td>3947</td><td>98.7</td><td>51.8</td></tr><tr><td>8k</td><td>2542</td><td>5237</td><td>7779</td><td>97.2</td><td>67.3</td><td>1987</td><td>5924</td><td>7911</td><td>98.9</td><td>74.9</td></tr></table>

## B.3 Detailed Same-Pool Selection Analysis

Table 8: Detailed targeted controls at Llama-3.1-8B-Instruct, $B = 4 \mathrm { k } .$ “Macro” averages all four benchmarks; same-pool rows reuse identical Stage-I candidates and change only the selector.
<table><tr><td>Method / control</td><td>Macro (%)</td></tr><tr><td>Competing adaptive policies Adaptive-Consistency (Aggarwal et al., 2023)</td><td>58.6</td></tr><tr><td>TALE-EP (Han et al., 2025)</td><td>59.1</td></tr><tr><td>League (fixed schedule)</td><td>58.9</td></tr><tr><td>BOSS</td><td>60.5</td></tr><tr><td>Allocation controls</td><td></td></tr><tr><td>No novelty transition</td><td>59.0</td></tr><tr><td>Fixed generation-selection split</td><td>59.3</td></tr><tr><td>Same Stage-I candidate pool</td><td></td></tr><tr><td>Plurality</td><td></td></tr><tr><td>Single fixed-order judge</td><td>58.4</td></tr><tr><td>Randomized A/B judge</td><td>59.1</td></tr><tr><td></td><td>59.7</td></tr><tr><td>Randomized A/B + repeated voting Full BOSS selector</td><td>60.3 60.5</td></tr></table>

Table 9: Error-correction behavior on identical Stage-I candidate pools. Recovery is measured when plurality is wrong but the correct hypothesis is present; harm is measured when plurality is initially correct.
<table><tr><td>Selector</td><td>Recovery (%)</td><td>Harm (%)</td></tr><tr><td>Randomized A/B + repeated</td><td>22.8</td><td>2.0</td></tr><tr><td>Full BOSS selector</td><td>25.1</td><td>2.2</td></tr></table>

## B.4 Full Component Ablation

Table 10: Component ablation at B = 4k with Llama-3.1-8B-Instruct. Selected columns are shown; “Avg” averages all four benchmarks.

<table><tr><td>Variant</td><td>MATH</td><td>GPQA-D</td><td> $\operatorname { A v g }$ </td></tr><tr><td>BOSS (Full)</td><td>59.4</td><td>39.5</td><td>60.5</td></tr><tr><td>No early stop</td><td>58.0</td><td>38.7</td><td>59.6</td></tr><tr><td>No novelty transition</td><td>58.1</td><td>37.6</td><td>59.0</td></tr><tr><td>Fixed gen.-sel. split</td><td>58.5</td><td>38.1</td><td>59.3</td></tr><tr><td>Fixed-order single-vote judge</td><td>56.8</td><td>35.0</td><td>57.7</td></tr><tr><td>Round-robin scheduling</td><td>57.4</td><td>37.1</td><td>58.8</td></tr><tr><td>Full representatives</td><td>55.9</td><td>38.3</td><td>58.3</td></tr></table>

## B.5 Mechanism and Allocation Diagnostics

Table 11: Observed BOSS allocation by hard budget. Gen./Judge are mean realized tokens; Entry/Fallback are trajectory rates.
<table><tr><td>B</td><td>Gen.</td><td>Judge</td><td>Entry (%)</td><td>Fallback (%)</td></tr><tr><td>2k</td><td>1445</td><td>492</td><td>31.8</td><td>65.6</td></tr><tr><td>4k</td><td>1903</td><td>2044</td><td>84.1</td><td>12.4</td></tr><tr><td>8k</td><td>1987</td><td>5924</td><td>94.1</td><td>1.2</td></tr></table>

Table 12: Sensitivity of Llama-3.1-8B-Instruct at $B =$ 4k to the principal transition hyperparameters. Bold marks the main configuration. Entry is the Stage-II entry rate; for $\tau _ { \mathrm { c o n s } } ,$ parentheses also give the consensus-exit rate.
<table><tr><td>Parameter</td><td>Value</td><td>Macro (%)</td><td>Behavior (%)</td></tr><tr><td>ρnovel</td><td>0.000 0.125 0.250 0.375 0.500</td><td>59.4 60.1 60.5 60.3 59.8</td><td>entry 52.6 entry 72.8 entry 84.1 entry 89.7 entry 93.2</td></tr><tr><td> $W$ </td><td>2 3 4 6 8</td><td>60.0 60.4 60.5 60.2 59.9</td><td>entry 90.5 entry 86.9 entry 84.1 entry 77.8 entry 70.4</td></tr><tr><td> $\tau _ { \mathrm { c o n s } }$ </td><td>0.60 0.70 0.80 0.90</td><td>59.9 60.5 60.4 60.0</td><td>entry 79.0 (cons. 8.9) entry 84.1 (cons. 3.5) entry 85.7 (cons. 1.8) entry 86.6 (cons. 0.7)</td></tr></table>

Across the tested switching hyperparameters, macro-average accuracy remains within 59.8–60.5 for the full $\rho _ { \mathrm { n o v e l } }$ sweep and within 59.9–60.5 across the tested W and $\tau _ { \mathrm { { c o n s } } }$ values. The behavior columns show the expected directional effects on switching and early consensus.

## Use of Large Language Models

LLMs assisted with language polishing, presentation, and manuscript organization; the authors verified all experiments, analyses, results, and scientific claims.