# MASIF: Implicit Failures for Process Observability in Multi-Agent Systems

Yanwei Huang

HKUST

yanwei.huang@connect.ust.hk

Arpit Narechania   
HKUST   
arpit@ust.hk

Atanu Sinha

Adobe Research

atr@adobe.com

## Abstract

Multi-agent systems (MAS) can complete a task despite local execution errors that are corrected, bypassed, or carried forward without individually determining its outcome. We call these implicit failures (IFs) and study them as signals for process observability beyond final task outcome. Across 1,828 traces from nine MAS–benchmark configurations, IFs occur in 54.3% of successful and 75.2% of failed traces. We develop taxonomies of error types and subsequent absorption mechanisms, supported by human and cross-model annotation audits. Counterfactual replay further probes how repairing an IF changes the final outcome. To examine the practical value of IFs, we evaluate IFbased best-of-N trace selection against existing process reward models, and introduce a novel RL algorithm, IF-GRPO, that uses IF costs to refine preferences among successful trajectories in GRPO. Experiments show that IF-GRPO exhibits some gains in accuracy and process metrics. Importantly, it exposes trade-offs between task performance, IF count and cost, and token-normalized IF density. Overall, our results show that IFs provide complementary process observability into how MAS reach their outcomes, including error recovery, propagation, and IF-based process differences that final task success alone does not expose.

## 1 Introduction

Large language model (LLM) agents can reason, plan, and interact with environments (Xi et al., 2023; Yao et al., 2023), but a single agent struggles with complex problems that exceed its context window or require diverse expertise. Multi-agent systems (MAS), in which multiple LLM agents collaborate, address this limitation through task decomposition, role-playing, and parallel execution (Tran et al., 2025; Hong et al., 2024; Wu et al., 2023; Li et al., 2023).

However, MAS are prone to cascading errors and coordination breakdowns that compromise task success (Cemri et al., 2025; Xie et al., 2026), and training them requires step-level rather than trace-level reward signals (Xi et al., 2026). This has driven a growing line of research onfailure attribution, which identifies the responsible agent and step in a failed execution trace through causal inference, hierarchical causal graphs, automated diagnosis frameworks, and more realistic benchmarks (Ma et al., 2025a; Wang et al., 2026; Wang, 2026; Zhang et al., 2025c,b; In et al., 2026; Chen et al., 2026).

A key limitation shared by existing approaches is their focus on the decisive step that explains an eventual failure. This outcome-oriented view leaves part of the execution process unobserved: a trace may contain local errors that are corrected, bypassed, or carried forward without individually determining its final outcome. We call these implicitfailures (IFs). IFs can occur in both successful and failed trajectories, revealing how MAS encounter, absorb, and propagate errors during execution. We therefore study IFs as signals for process observability: complementary information about execution behavior that is not captured by the final success label alone.

We study IFs through four research questions:

RQ1: How common are IFs? (Section 2.2) Across 1,828 traces from nine MAS–benchmark configurations, IFs occur in 54.3% of successful and 75.2% of failed traces, showing that local process errors are prevalent on both sides of the outcome boundary.

![](images/833422e967f14bab4b6ad7f1601439963a50d1674241bdc13ed9a0f098676323.jpg)  
Figure 1: Example MAS traces for five absorption patterns. IFs are marked with crosses, with their resolution steps highlighted. Self-correction: WebWalker corrects its own navigation error to find the target. Self-workaround: The Assistant switches from flawed algebra to code to solve MathProxyAgent’s problem. Peer-repair: SimpleReviewer fixes a code error from SimpleTester. Orchestrator intervention: The Orchestrator redirects Assistant’s failed download to WebSurfer for scraping. Unresolved propagation: Browser’s wrong claim propagates downstream.

RQ2: Where are IFs from and how are they absorbed? (Section 2.3) We develop human-derived taxonomies of IF error types and absorption mechanisms, characterizing how local errors are corrected, bypassed, coordinated around, or propagated, complementing taxonomies that primarily study trace-level failures (Cemri et al., 2025).

RQ3: How do MAS outcomes respond to repairing IFs? (Section 2.4) Matched counterfactual replay tests whether IFs identify actionable process states. IF repair increases failure rescue by up to 46.2 percentage points over no-op replay.

RQ4: Can IFs provide useful training signals? (Section 3) To further demonstrate IFs’ practical utility, we apply them to trace selection and reinforcement learning (RL). Experiments show that our IF-based scoring improves best-of-N selection over known process reward models (PRMs). We further propose a novel IF-GRPO algorithm (Section 3.2) that incorporates IFs into Group Relative Policy Optimization, which improves accuracy in six of seven configurations and reduces IF cost per task in three, while exposing trade-offs between task performance and process quality.

Together, these studies position IFs as a complementary layer of process observability for MAS. Rather than asking only whether a system succeeds or which step caused its failure, IFs expose local errors encountered during execution, how the system responds to them, and process-level differences among trajectories with the same outcome.

## 2 Methodology

## 2.1 Data Collection

To study the prevalence and patterns of IFs, we constructed a benchmark including 9 MAS-benchmark configurations (Table 1), covering 1,828 MAS traces. We constructed it by selecting 1,048 traces from MAST (Cemri et al., 2025), a representative error-annotated benchmark of MAS traces, filtering out traces that lack verifiable ground-truth outcomes or cannot be reproduced. We further manually curated 780 traces leveraging two popular MAS–(WebWalker (Wu et al., 2025) and MrlX (Team, 2025)), and benchmarks–(WebWalkerQA (Wu et al., 2025) and BrowseComp-Plus (Chen et al., 2025)) for web navigation and deep research.

Table 1: Detailed profiles of the configurations in our study corpus.
<table><tr><td>MAS</td><td>Benchmark</td><td>Underlying LLM</td><td># Traces</td><td># Success</td><td># Failure</td><td>#IFs</td></tr><tr><td>Magentic-One</td><td>GAIA</td><td>GPT-40</td><td>165</td><td>52</td><td>113</td><td>383</td></tr><tr><td>WebAgent (WebWalker)</td><td>WebWalkerQA</td><td>GPT-4o-mini</td><td>680</td><td>71</td><td>609</td><td>1090</td></tr><tr><td>MrlX</td><td>BrowseComp-Plus</td><td>Qwen3-8b</td><td>100</td><td>4</td><td>96</td><td>119</td></tr><tr><td>AG2 (MathChat)</td><td>OlympiadBench</td><td>GPT-40</td><td>206</td><td>29</td><td>177</td><td>178</td></tr><tr><td>AG2 (MathChat)</td><td>GSMPlus</td><td>Claude-3.7-Sonnet &amp; GPT-4o</td><td>223</td><td>44</td><td>179</td><td>214</td></tr><tr><td>AG2 (MathChat)</td><td>MMLU</td><td>GPT-40</td><td>168</td><td>27</td><td>141</td><td>241</td></tr><tr><td>ChatDev</td><td>MMLU</td><td>GPT-40</td><td>42</td><td>23</td><td>19</td><td>1</td></tr><tr><td>MetaGPT</td><td>MMLU</td><td>GPT-40</td><td>21</td><td>8</td><td>13</td><td>33</td></tr><tr><td>HyperAgent</td><td>SWE-Bench</td><td>Claude-3.7-Sonnet</td><td>223</td><td>0</td><td>223</td><td>785</td></tr></table>

Each trace was annotated for IFs using an LLM-as-a-judge pipeline built on Gemini 3.1 Pro. Given the task context and execution trace, the judge inspected each step and flagged candidate IFs. We validated annotation quality through both human audit and cross-judge analysis. Two annotators independently audited a stratified random sample of 45 traces (5 per configuration). Among 73 existing IF annotations reviewed by both annotators, 64 (87.7%) were independently retained by both, and none were rejected by both. Separately, another random sample of 90 traces were re-annotated by GLM-5.2, Claude Opus 4.8, and DeepSeek V4 Pro in addition to the original judge (Gemini 3.1 Pro). Overall, 174 IFs were identified by at least three out of the four judges. Gemini covered 157 (90.2%) of this set, which were 69.2% of its identified IFs.

Annotation completeness is harder to establish. The human audit identified 21 additional IFs across 11 traces, with five independently recovered by both annotators. The missed and disputed cases primarily involve long-range evidence, repeated manifestations of an underlying error, and boundaries between local errors and intermediate task decomposition. Similarly, cross-judge differences concentrate on detection granularity and event boundaries rather than explicit trace-observable errors. A more detailed analysis is provided in Appendix C.

## 2.2 RQ1: How common are IFs?

RQ1 asks whether IFs are a marginal phenomenon or a pervasive property of MAS execution. To this end, we measure the fraction of traces containing at least one IF (prevalence) and the mean number of IFs per trace (intensity), stratified by outcome.

As shown in Table 1, across the full corpus, 140 of 258 successful traces (54.3%) and 1,181 of 1,570 failed traces (75.2%) contain at least one IF. Thus, IF presence does not collapse to the final outcome. Over half of successful traces contain IFs, while 24.8% of failed ones contain none. Therefore, IFs and outcome correctness expose distinct execution properties, motivating a closer examination of what errors occur and how they are absorbed.

Table 2: Taxonomy of IF Types and their share in the benchmark.
<table><tr><td>Type</td><td>Description</td><td>Example</td><td>Share</td></tr><tr><td>Failed Tool Use</td><td>execute, or returns unusable output because of dio file as text and receives a conversion tool or environment issues.</td><td>A tool call fails, is blocked, times out, cannot FileSurfer attempts to read an MP3 au- 24.5% exception.</td><td></td></tr><tr><td>Wrong Action</td><td>for the current task state or unlikely to make a valid link or play button, leaving the progress.</td><td>The agent takes an action that is inappropriate WebSurfer clicks page text rather than 23.0% browser on the same page.</td><td></td></tr><tr><td></td><td>Unnecessary Action The agent performs redundant, low-value, or WebSurfer handles a cookie prompt in- 16.4% repetitive actions that consume steps without stead of following the already identified advancing the task.</td><td>target page.</td><td></td></tr><tr><td>Hallucination</td><td>source content, or intermediate evidence not sup- is not present in the retrieved source. ported by the trace.</td><td>The agent fabricates information, tool behavior, The agent invents a ranked list item that 15.3%</td><td></td></tr><tr><td>Incomplete Action</td><td>The agent attempts an action but omits required FileSurfer repeatedly returns the first 9.1% details, formatting, inputs, or follow-through.</td><td>PDF page instead of searching or navi- gating to the relevant pages.</td><td></td></tr><tr><td>Reasoning Error</td><td>The agent makes an incorrect inference, intro- The assistant swaps document statuses 8.6% duces or propagates incorrect information, re- across several listed books, or the orches- peats a known-bad strategy, misses relevant evi- trator identifies the wrong single from dence, or terminates from insufficient reasoning. an album despite available evidence.</td><td></td><td></td></tr><tr><td>Misunderstanding</td><td>The agent misunderstands the task, role, con- The orchestrator fails to recognize that 2.8% straints, or operational setting.</td><td>another agent has already reached the correct page and restarts the search.</td><td></td></tr><tr><td>Malformed Output</td><td>The agent emits syntactically malformed text WebWalker emits a memory block with 0.2% or a corrupted structured block, but execution a stray closing quote and brace before continues.</td><td>recovering in the next step.</td><td></td></tr></table>

## 2.3 RQ2: Where are IFs from and how are they absorbed?

We investigate the types of local errors underlying IFs and how they are repaired, bypassed, or propagated.

## 2.3.1 Study design

To answer this, we had three annotators manually inspect 45 randomly sampled traces and apply open coding following the grounded theory (Corbin, 2017) approach, iterating on the label set for three rounds until it saturated. The coding process mainly focuses on a) the cause and type of an IF and b) the influence and the eventual outcome of an IF. Afterwards, we followed the same annotation pipeline as described in Section 2.1 to label each IF in our benchmark with error types and absorption patterns. The labels were audited by two additional human annotators mentioned in 2.1. Among their jointly retained IFs, agreement was 84.4% on error type and 89.1% on absorption pattern.

## 2.3.2 Results

We propose two taxonomies, one regarding the error type and the other regarding the absorption patterns of IFs. The taxonomy of step-level error types is shown in Table 2, which is complementary to the trace-level one by Cemri et al. (Cemri et al., 2025). Failed tool use (24.5%) and wrong action (23.0%) are the two most frequent types, together accounting for 47.5% of the identified IFs. The taxonomy shares a few similar classes with the one by Cemri et al., such as Wrong/Unnecessary/Incomplete Action (w.r.t. “Reasoning-Action Mismatch”) and Misunderstanding (w.r.t. “Disobey Task/Role Specification”).

Table 3: Distribution of IF absorption mechanisms across the corpus, separated by trace outcome.
<table><tr><td colspan="4">Absorption Mechanism Success Failure Total Share</td></tr><tr><td>Self-correction</td><td>34</td><td>585 619</td><td>20.3%</td></tr><tr><td>Trivial issue</td><td>77</td><td>1,037 1,114</td><td>36.6%</td></tr><tr><td>Orchestrator intervention</td><td>36</td><td>262 298</td><td>9.8%</td></tr><tr><td>Peer repair</td><td>4</td><td>38 42</td><td>1.4%</td></tr><tr><td>Self-workaround</td><td>23</td><td>412 435</td><td>14.3%</td></tr><tr><td>Unresolved propagation</td><td>51</td><td>485 536</td><td>17.6%</td></tr><tr><td>Total</td><td>225</td><td>2,8193,044100.0%</td><td></td></tr></table>

However, our taxonomy provides additional insights into local issues, such as hallucination, tool misuse, and malformed output.

The IF absorption pattern taxonomy consists of six patterns, and we provide a representative trace for each except trivial issue illustrated in Figure 1.

Self-correction. The same agent recognizes and repairs its own local error within the same or an immediately subsequent turn. For instance, in Figure 1A, a WebWalker agent visits the wrong education page and, upon receiving a negative result, immediately redirects itself to the correct page without external signal.

Self-workaround. The IF is bypassed by the originating agent by advancing the task through a different route, such as choosing a different query, tool, or implementation. In Figure 1B, an AG2 assistant (Wang et al., 2024a) recognizes that its algebraic derivation leads to an impossible equation and silently substitutes a programmatic search, reaching the correct answer without revisiting the flawed reasoning.

Peer repair. The error is compensated for by a different, non-managerial agent downstream. Figure 1C shows a MetaGPT trace in which SimpleTester produces tests that would raise a TypeError. SimpleReviewer detects the mismatch and issues a corrected version back to SimpleTester.

Orchestrator intervention. A manager or planner peer agent redirects the trajectory after the IF. Figure 1D, a MagenticOne orchestrator observes that local video extraction has failed and replans the entire subtask, switching to a WebSurfer-based search strategy that ultimately succeeds.

Unresolved propagation. The IF is left unrepaired with the erroneous state accepted, carried forward, or folded into later reasoning. Figure 1E illustrates an MrlX trace in which geographically inconsistent claims returned by the browser are silently accepted in the agent’s subsequent reasoning.

Trivial issue. A catch-all for absorption patterns with low analytic value, including mechanical retries, duplicate work, ignored or abandoned actions, no-effect errors, lucky coincidences, tool- or environmentlevel absorption, runtime blocking, timeout, and trace termination.

Table 3 reports the frequency of each absorption label across the corpus, broken down by the outcome of the trace. Several patterns stand out. First, self-correction and self-workaround are among the most common absorption mechanisms (20.3% and 14.3% of all IFs) other than trivial issues, confirming that local recovery within a single agent is the dominant route by which MAS tolerate intermediate errors. Besides, orchestrator intervention (9.8%) and peer repair (1.4%) indicate that inter-agent coordination contributes substantively to IF resolution. We further observe that 17.6% of IFs are labeled as unresolved propagation. Among the 1,930 IFs with informative absorption labels (excluding the combined trivial/unknown category), 72.2% are classified as absorbed or repaired and 27.8% as unresolved propagation. These absorption labels provide process information complementary to final outcome, but are descriptive and do not establish causal effects.

## 2.4 RQ3: How do MAS outcomes respond to repairing IFs?

IFs are locally non-decisive with respect to the observed outcome, but this does not imply that the execution state surrounding an IF is inconsequential. When IFs expose meaningful process state rather than incidental irregularities, locally repairing them may alter subsequent execution. RQ3 tests this possibility

Table 4: Matched no-op and IF-repair replay across seven MAS–benchmark configurations. Arrows indicate no-op → IF repair. SPR measures success preservation and RR measures failure rescue. ∆RR is the additional rescue rate under IF repair (percentage points), with 95% paired task-cluster bootstrap confidence intervals (5,000 resamples; unadjusted for multiple comparisons). Each configuration contains 50 originally successful and 50 failed traces. IF repair increases failure rescue by 23.0–46.2 pp beyond no-op replay in four configurations, demonstrating the potential of targeted IF interventions to recover failed executions.
<table><tr><td>MAS</td><td>Benchmark</td><td>Traces</td><td>Unique Tasks IFs</td><td></td><td>SPR (%)</td><td>RR (%)</td><td>∆RR (pp) [95% CI]</td></tr><tr><td>AG2 (MathChat)</td><td>MMLU</td><td>100</td><td>100</td><td>138</td><td> $1 0 0 . 0  9 9 . 0 $ </td><td> $1 . 0  4 2 . 0$ </td><td>+41.0 [28.6, 53.8]</td></tr><tr><td>ChatDev</td><td>MMLU</td><td>100</td><td>100</td><td>176</td><td> $9 6 . 3  9 6 . 7 $ </td><td> $8 . 0  3 5 . 8 $ </td><td>+27.8 [15.8, 39.9]</td></tr><tr><td>MetaGPT</td><td>MMLU</td><td>100</td><td>90</td><td>145</td><td> $9 8 . 0  9 1 . 7 $ </td><td> $1 . 0  4 7 . 2$ </td><td>+46.2 [33.0, 60.0]</td></tr><tr><td>AG2 (MathChat)</td><td>OlympiadBench</td><td>100</td><td>98</td><td>104</td><td> $8 2 . 0  7 8 . 0$ </td><td> $2 8 . 0  5 1 . 0 $ </td><td>+23.0 [7.9, 38.4]</td></tr><tr><td>AG2 (MathChat)</td><td>GSMPlus</td><td>100</td><td>82</td><td>164</td><td> $8 9 . 3  9 0 . 5 $ </td><td> $4 2 . 1  4 9 . 4 $ </td><td>+7.3 [−3.0, 17.7]</td></tr><tr><td>MagenticOne</td><td>GAIA</td><td>100</td><td>81</td><td>206</td><td> $3 6 . 7  4 0 . 4$ </td><td> $2 . 7  7 . 6$ </td><td>+4.9 [−2.9, 12.8]</td></tr><tr><td>MrlX</td><td>BrowseComp-Plus</td><td>100</td><td>100</td><td>209</td><td> $4 0 . 2  3 9 . 2 $ </td><td> $9 . 5  5 . 8 $ </td><td>-3.7 [−11.0, 2.5]</td></tr></table>

through matched counterfactual replay, asking whether IF repair preserves successful executions or rescues failed ones beyond replay-induced variation.

## 2.4.1 Preliminaries

Let $T = ( s _ { 1 } , \ldots , s _ { n } ) $ be a MAS trace and $\mathcal { T F } ( T )$ its annotated IF steps. For each $i \in \mathcal { I } \mathcal { F } ( T )$ , we construct two continuations from the same execution boundary: $T _ { i } ^ { \mathrm { n o o p } }$ , which preserves the original step, and $T _ { i } ^ { \mathrm { f i x } }$ , which replaces it with a local correction. Let $Y ( T ) \in \{ 0 , 1 \}$ denote task correctness.

For either intervention $b \in \{ \mathrm { n o o p } , \mathrm { f i x } \}$ , we measure success preservation and failure rescue:

$$
\begin{array} { r } { \mathrm { S P R } _ { b } ( C ) = P ( Y ( T _ { i } ^ { b } ) = 1 \mid Y ( T ) = 1 ) , } \\ { \mathrm { R R } _ { b } ( C ) = P ( Y ( T _ { i } ^ { b } ) = 1 \mid Y ( T ) = 0 ) . } \end{array}
$$

We average over IF positions within each trace and then over traces in the corresponding outcome stratum. The incremental rescue effect is $\Delta \mathrm { R R } = \mathrm { R R } _ { \mathrm { f i x } } - \mathrm { R R } _ { \mathrm { n o o p } }$

## 2.4.2 Counterfactual replay

Study Design. We conduct matched no-op and IF-repair replay across seven configurations in Table 4. Compared with the previously studied configurations (Table 1), we excluded HyperAgent because successful traces were difficult to acquire, as well as WebAgent due to environmental issues. For configurations without sufficient eligible traces on both outcome sides in the original corpus, we regenerated the traces using Qwen3.6-27B with reruns until at least 50 successful and 50 failed traces existed. Since replaying every trace is prohibitively costly, we sampled 50 successful and 50 failed traces randomly per configuration. The number of unique tasks in the sample is provided in the table. For each selected IF to be repaired, we restored the runtime state at the IF boundary, replaced the IF step with a local correction generated by Gemini 3.1 Pro under the original agent role and context, and resumed MAS execution from the resulting state. The detailed prompt templates are provided in Appendix D.2.

Results. Table 4 shows that IF repair provides substantial additional failure rescue beyond no-op replay in four configurations. The largest gains occur on MetaGPT/MMLU (+46.2 pp) and MathChat/MMLU $\left( + 4 1 . 0 \mathrm { p p } \right)$ , followed by ChatDev/MMLU (+27.8 pp) and MathChat/OlympiadBench (+23.0 pp) with their unadjusted 95% confidence intervals excluding zero. By contrast, no-op replay already rescues 42.1% of failed MathChat/GSMPlus executions, with no clear additional benefit from repair. These findings demonstrate that IF-guided corrections can rescue failed MAS executions beyond replay-induced variation. They highlight the potential of IFs as intervention targets for investigating error recovery.

## 3 IFs for MAS Training

Given the findings, we further ask whether IFs can provide useful training signals in practice (RQ4). We thus test on two downstream settings: inference-time trace selection (Section 3.1) and policy optimization (Section 3.2).

## 3.1 Study A: IF as a PRM

Baselines. We evaluate three settings that cover web navigation and math reasoning: MagenticOne on GAIA (Mialon et al., 2024) with PRInTS (Lee et al., 2025) (165 tasks), and AG2 (MathChat) on GSMPlus (Li et al., 2024) with either Qwen2.5-Math-PRM (Zhang et al., 2025d) or Math-Shepherd (Wang et al., 2024b) (80 tasks).

Protocol. Following the common practice of PRM evaluation, we adopt the best-of-N selection protocol. We divide the tasks into a tuning set and a test set by 20%-80%. For each task, we use the MAS to generate $N \in \{ 4 , 8 , 1 6 \}$ candidate traces from scratch. We then construct three PRM scorers to generate step-wise scores, which are then aggregated into a trace-level score. The trace with the highest score is selected, and the eventual task accuracy of the selected traces is used to evaluate the scorers. The scorers are introduced below.

PRM Scorer. We aggregate step-level PRM scores using four pooling functions: min, mean, product/geometric mean, and last-step. For each pooling function, we compute a trace-level PRM score for every candidate trace. The candidate with the highest trace-level PRM score is selected, and we compute the resulting task-level accuracy across all tasks in the training set. The pooling function that achieves the highest accuracy in training is selected, held fixed and applied unchanged in testing. Specially, for PRInTS, whose raw step scores range in [−4, 4], we normalize each score to [0, 1] before pooling, while for Qwen2.5-Math-PRM and Math-Shepherd the step scores are already normalized and are directly used as quality scores. The PRM-only selector chooses the highest-scoring candidate among the N traces.

IF Scorer. The IF scorer is a trace-only LLM judge. It reads only the task and the candidate trace without awareness of the expected answer or any oracle information. The judge examines the trace for IFs while noting for each IF its error step, agent, absorption type, and a heuristic severity designed to assign higher values to unresolved propagated IFs and non-trivial issues. The trace-level IF risk score is calculated by aggregating the severity of all IFs normalized by the trace length. The calculation is detailed in Appendix A, where we further show that candidate rankings are stable to independent ±10% perturbations of these heuristic weights (Spearman > 0.993 across 1,000 trials. The IF scorer selects the lowest-risk candidate among the $N$ traces.

PRM+IF Scorer. We further construct a PRM+IF scorer to study the opportunities of incorporating IFs into existing PRMs. To derive the score, we first calculate the scores $s _ { p r m } , s _ { i f }$ using the PRM scorer and IF scorer independently. We then perform a weighted average of both scores, i.e., ${ w \cdot s _ { p r m } } + ( 1 - w ) { * ( 1 - s _ { i f } ) }$ Note that the IF scorer’s original score $s _ { i f }$ is transformed into $1 - s _ { i f }$ because the former represents the risk by definition. Similar to the PRM scorer, both the pooling function and the weight w are trained on the training set and then fixed for the testing set.

Results. Table 5 shows best-of-N task accuracy across the three (MAS, benchmark, PRM) configurations and $N \in \{ 4 , 8 , 1 6 \}$ . We also present the task accuracy of a random trace selector for reference. Across the nine settings, PRM-only selection is generally close to random selection. Math-Shepherd and Qwen2.5-Math-PRM, both of which report strong single-agent results in the literature (Wang et al., 2024b; Zhang et al., 2025d), fail to provide a useful selection signal when applied to MAS traces. PRInTS, an agent-trace PRM, performs similarly. By contrast, the IF-only scorer matches or exceeds PRM-only selection in all nine settings, with a gain of 3.0–6.8 percentage points on MagenticOne/GAIA and up to 4.7 points on AG2/GSMPlus. The results indicate that IFs provide a useful process-level selection signal across the evaluated MAS and task settings, including settings where the existing PRMs provide little discrimination among candidate traces.

Furthermore, we observe that the PRM+IF scorer performs no better than IF alone. We interpret this as a consequence of the previous observation: when the PRM component is itself close to random, mixing it

Table 5: Best-of-N task accuracy (%) across 3 configurations (MAS, benchmark, PRM) and $N \in \{ 4 , 8 , 1 6 \}$ IF-based selection outperforms the evaluated PRMs in all nine comparisons, while PRM-only selection generally performs close to random, highlighting the additional process-quality information captured by IFs.
<table><tr><td>Configuration</td><td>N Random</td><td>PRM</td><td>IF</td><td>PRM+IF</td></tr><tr><td rowspan="3">MagenticOne / GAIA / PRInTS</td><td>4</td><td>19.3</td><td>18.2 25.0</td><td rowspan="3">23.5 23.5 25.0</td></tr><tr><td>8</td><td>19.8</td><td>21.2 24.2</td></tr><tr><td>16</td><td>17.7</td><td>19.7 26.5</td></tr><tr><td rowspan="3">AG2 / GSMPlus / Qwen2.5-Math- PRM</td><td>4</td><td>80.1</td><td>79.7 81.3</td><td rowspan="3">81.3 81.3 78.1</td></tr><tr><td>8</td><td>79.1</td><td>78.1 82.8</td></tr><tr><td>16</td><td>78.8</td><td>78.1 82.8</td></tr><tr><td rowspan="3">AG2 / GSMPlus / Math-Shepherd</td><td>4</td><td>80.1</td><td>79.7 81.3</td><td>79.7</td></tr><tr><td>8</td><td>79.1</td><td>81.3 82.8</td><td>81.3</td></tr><tr><td>16</td><td>78.8</td><td>79.7 82.8</td><td>79.7</td></tr></table>

Table 6: GRPO versus IF-GRPO across seven MAS–benchmark configurations. Each configuration reports GRPO → IF-GRPO (first row) and its change (second row). $\ " \# \ : \mathrm { I F } ^ { \prime } $ means the number of IFs. Green/red indicate improvement/deterioration. Bold indicates Holm-adjusted $p < 0 . 0 5$ (two-sided paired permutation tests). IF-GRPO improves accuracy in six configurations, but per-task IF burden and token-normalized IF density can change in opposing directions, revealing process-level trade-offs invisible to task accuracy alone.
<table><tr><td>MAS</td><td>Benchmark</td><td>Accuracy (%)</td><td>IF Cost / Task</td><td># IF / Task</td><td>IF Cost / 1K Tokens</td><td># IF / 1K Tokens</td></tr><tr><td rowspan="3">AG2 (MathChat)</td><td>MMLU-Pro</td><td>54.98 → 57.06 +2.08 pp</td><td> $^ { 1 . 4 2 \to 1 . 5 1 } _ { + 6 . 3 3 \% }$ </td><td> $\begin{array} { c } { 1 . 1 4  1 . 4 1 } \\ { + 2 3 . 8 4 \% } \end{array}$ </td><td>4.37 → 2.39 -45.38%</td><td>3.49 → 2.22 -36.49%</td></tr><tr><td>GSMPlus</td><td>71.75 → 72.21 +0.46 pp</td><td>0.49 → 0.34 -30.83%</td><td>0.49 → 0.30 -39.66%</td><td>2.33 → 1.54 -33.92%</td><td>2.31 → 1.33 -42.37%</td></tr><tr><td>OlympiadBench</td><td>37.69 → 40.00 +2.31 pp</td><td>2.35 → 2.32 -0.98%</td><td>3.37 → 2.63 -21.91%</td><td>2.40 → 1.64 -31.71%</td><td>3.45 → 1.86 -46.18%</td></tr><tr><td rowspan="2">Teamwork</td><td>MMLU-Pro</td><td>62.46 → 66.20 +3.74 pp</td><td>6.28 → 6.36 +1.34%</td><td>4.57 → 5.46 +19.40%</td><td>3.33 → 1.29 -61.23%</td><td>2.42 → 1.11</td></tr><tr><td>GSMPlus</td><td>77.38 → 72.50 -4.88 pp</td><td>1.35 → 1.38 +2.30%</td><td>1.56 → 1.70 +9.04%</td><td>1.10 → 0.80 -27.50%</td><td>-54.28% 1.28 → 0.99 -22.77%</td></tr><tr><td>Self-MoA</td><td>MMLU-Pro</td><td>64.70 → 65.70 +1.00 pp 75.63 → 76.38 +0.75 pp</td><td>4.39 → 4.55 +3.69% 0.97 → 0.97 -0.62%</td><td>5.06 → 5.42 +7.12% 1.15 → 1.14 -1.04%</td><td>0.92 → 0.92 -0.22% 0.45 → 0.43 -3.81%</td><td>1.06 → 1.10 +3.10% 0.53 → 0.51 -4.17%</td></tr></table>

with the IF signal adds noise.

## 3.2 Study B: IFs for RL training

To further investigate the application of IFs in RL, we incorporate them into Group Relative Policy Optimization (GRPO) (Shao et al., 2024).

GRPO Background. For a group of MAS traces, let $y _ { i } \in \{ 0 , 1 \}$ denote the final task correctness of trajectory $T _ { i }$ . Outcome-only GRPO uses the group-normalized advantage

$$
A _ { i } ^ { \mathrm { G R P O } } = \frac { y _ { i } - \bar { y } } { s _ { y } + \epsilon } ,\tag{1}
$$

where $\bar { y }$ and $s _ { y }$ denote the group mean and standard deviation.

IF-GRPO. We introduce IFs as a secondary signal to favor successful traces with fewer and less severe IFs while preserving task correctness as the primary objective. For each successful trajectory, an LLM judge identifies IFs and computes their aggregated cost:

$$
C _ { i } = \sum _ { e \in \mathrm { I F } ( T _ { i } ) } w _ { \mathrm { t y p e } } ( e ) w _ { \mathrm { a b s } } ( e ) ,\tag{2}
$$

where the weights $w _ { \mathrm { t y p e } } , w _ { \mathrm { a b s } }$ follow Appendix A.

We apply IF supervision only to mixed-outcome groups, which contain both successful and failed traces. Let $S = \{ i : y _ { i } = 1 \}$ and $c _ { \mathrm { m i n } } , c _ { \mathrm { m a x } }$ denote the minimum and maximum IF costs among successful traces. We define

$$
z _ { i } = \frac { c _ { \operatorname* { m a x } } - C _ { i } } { c _ { \operatorname* { m a x } } - c _ { \operatorname* { m i n } } } - \frac { 1 } { 2 } , \quad i \in S ,\tag{3}
$$

and construct the advantage as

$$
A _ { i } = \left\{ \begin{array} { l l } { A _ { i } ^ { \mathrm { G R P O } } ( 1 + z _ { i } ) , } & { y _ { i } = 1 , } \\ { A _ { i } ^ { \mathrm { G R P O } } , } & { y _ { i } = 0 . } \end{array} \right.\tag{4}
$$

If fewer than two traces succeed or their IF costs are identical, we set $z _ { i } = 0$ . All-success and all-failure groups receive zero advantage. Since $z _ { i } \in [ - 0 . 5 , 0 . 5 ]$ , IF supervision adjusts the relative reinforcement of successful traces without reversing the preference for success over failure. Thus, IFs refine rather than replace outcome supervision.

Setup. To examine the applicability of IF-guided optimization across different MAS architectures, we select three representative MAS with distinct coordination mechanisms: AG2 (MathChat), which uses assistant–proxy interaction; Teamwork (Pappu et al., 2026), which employs multi-agent discussion; and Self-MoA (Li et al., 2025), which aggregates independently generated proposals. These systems are evaluated across three benchmarks: MMLU-Pro (Wang et al., 2024c), GSMPlus (Li et al., 2024), and OlympiadBench (He et al., 2024). All agents use Qwen3-4B as the backbone. Each GRPO group contains eight MAS trajectories. We compare GRPO and IF-GRPO under matched training settings, with Qwen3.8-Flash-Next serving as the IF judge. Additional details are in Appendix D.

Results. As shown in Table 6, IF-GRPO yields higher accuracy in six configurations and reduces IF cost per task in three. Some configurations improve both: on AG2/GSMPlus, accuracy increases by 0.46 points while IF cost and count decrease by 30.83% and 39.66%. Others expose trade-offs across outcome and process measures. For example, Teamwork/GSMPlus reduces IF cost and count per 1K tokens significantly while losing 4.88 accuracy points. These divergences show that task accuracy, per-task IF burden, and token-normalized IF density capture different aspects of MAS execution. While IF-GRPO does not establish a uniformly superior training objective, it demonstrates how IF-based process measures expose changes that outcome accuracy alone would conceal.

## 4 Related Work

Despite their effectiveness across domains, MAS are prone to errors, motivating research into their failure patterns. Cemri et al. (Cemri et al., 2025) analyze 1,642 traces across seven MAS frameworks, identifying 14 failure modes spanning specification ambiguity, inter-agent misalignment, and inadequate verification. Wang et al. (Xie et al., 2026) model error cascades, showing how minor errors amplify through recursive context reuse and proposing a governance layer to suppress propagation. Similarly, Reid et al. (Reid et al., 2025) highlight 6 remarkable failure modes of governed MAS.

These risks motivate failure attribution: identifying the agent and step responsible for a failed trace. Drawing on traditional software debugging, researchers propose tools for prompt testing (Kahng et al., 2024), red-teaming (Reid et al., 2025), MAS trace visualization (Zhang et al., 2025a; Sheng et al., 2026), and trace intervention for what-if testing (Epperson et al., 2025). However, these tools demand substantial developer effort and expertise.

Recent work therefore explores automatic attribution. Zhang et al. (Zhang et al., 2025c) use LLMs to diagnose traces and introduce the Who&When benchmark with annotations of decisive failures. AgenTracer (Zhang et al., 2025b) trains a dedicated attribution model on curated success-failure trajectory pairs, while Ma et al. (Ma et al., 2025a) use causal inference to predict critical steps without a trained attributor. Wang et al. (Wang et al., 2026) transform flat execution logs into hierarchical causal graphs for systematic diagnosis, and AgentTrace (Wang, 2026) extends this approach to deployed systems through lightweight post-hoc tracing. DoVer (Ma et al., 2025b) tests automatically generated hypotheses by intervening in MAS traces and measuring outcome flips.

These approaches primarily seek steps that explain an eventual failure. In contrast, we study locally non-decisive errors in both successful and failed traces. IFs complement failure attribution by exposing how errors arise, are absorbed, or propagate during execution, providing process observability beyond the final outcome.

## 5 Conclusion

We present a systematic study of implicit failures (IFs) in multi-agent systems, analyzing 1,828 traces across nine MAS–benchmark configurations. We demonstrate their prevalence and characterize their error types and absorption patterns. Through counterfactual repair, we further show that targeted IF repair can rescue failed executions beyond the effects of replay alone, suggesting opportunities for IF-guided debugging and recovery. We also investigate the practical utility of IFs as process-level supervision. IF-based scoring compares favorably with established PRMs in best-of-N selection, while IF-guided GRPO shows both improvements and trade-offs between task accuracy and IF burden. Our findings establish IFs as a valuable dimension of MAS behavior and highlight process-level observability alongside outcome-based evaluation.

## Limitations

Our study has several limitations. First, IF identification relies on LLM-based annotation, which may introduce judgment bias and miss subtle errors. Although human audits and cross-judge comparisons support annotation quality, completeness remains challenging to establish.

Second, while our corpus spans nine MAS–benchmark configurations, its coverage of task domains, MAS architectures, and backbone models remains limited. Our counterfactual replay study examines one IF repair at a time under a fixed intervention protocol. Further controlled perturbation experiments could investigate how MAS respond to injected errors and multiple simultaneous failures.

Finally, our PRM study focuses on best-of-N selection, while IF-GRPO training uses a single trainable backbone with limited training seeds. IF supervision also relies on a stronger frozen LLM judge, introducing additional inference costs. Further experiments with diverse backbones, MAS architectures, and supervision baselines would help establish the generality and efficiency of IF-guided optimization.

## References

Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, et al. Why do multi-agent llm systems fail? Advances in Neural Information Processing Systems, 2025.

Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, and Qing Wang. Seeing the whole elephant: A benchmark for failure attribution in llm-based multi-agent systems. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19888–19905, 2026.

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, et al. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. arXiv preprint arXiv:2508.06600, 2025.

Juliet Corbin. Grounded theory. The Journal of Positive Psychology, 12(3):301–302, 2017.

Will Epperson, Gagan Bansal, Victor C Dibia, Adam Fourney, Jack Gerrits, Erkang Zhu, and Saleema

Amershi. Interactive debugging and steering of multi-agent ai systems. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems, pages 1–15, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. MetaGPT: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations (ICLR), 2024.

Yeonjun In, Mehrab Tanjim, Jayakumar Subramanian, Sungchul Kim, Uttaran Bhattacharya, Wonjoong Kim, Sangwu Park, Somdeb Sarkhel, and Chanyoung Park. Rethinking failure attribution in multi-agent systems: A multi-perspective benchmark and evaluation. arXiv preprint arXiv:2603.25001, 2026.

Minsuk Kahng, Ian Tenney, Mahima Pushkarna, Michael Xieyang Liu, James Wexler, Emily Reif, Krystal Kallarackal, Minsuk Chang, Michael Terry, and Lucas Dixon. Llm comparator: Visual analytics for side-by-side evaluation of large language models. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, pages 1–7, 2024.

Jaewoo Lee, Archiki Prasad, Justin Chih-Yao Chen, Zaid Khan, Elias Stengel-Eskin, and Mohit Bansal. Prints: Reward modeling for long-horizon information seeking. arXiv preprint arXiv:2511.19314, 2025.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for “mind” exploration of large language model society. Advances in neural information processing systems, 36:51991–52008, 2023.

Qintong Li, Leyang Cui, Xueliang Zhao, Lingpeng Kong, and Wei Bi. Gsm-plus: A comprehensive benchmark for evaluating the robustness of llms as mathematical problem solvers. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2961–2984, 2024.

Wenzhe Li, Yong Lin, Mengzhou Xia, and Chi Jin. Rethinking mixture-of-agents: Is mixing different large language models beneficial? arXiv preprint arXiv:2502.00674, 2025.

Guoqing Ma, Jia Zhu, Hanghui Guo, Weijie Shi, Jiawei Shen, Jingjiang Liu, and Yidan Liang. Automatic failure attribution and critical step prediction method for multi-agent systems based on causal inference. arXiv preprint arXiv:2509.08682, 2025a.

Ming Ma, Jue Zhang, Fangkai Yang, Yu Kang, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Dover: Intervention-driven auto debugging for llm multi-agent systems. arXiv preprint arXiv:2512.06749, 2025b.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: a benchmark for general ai assistants. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, International Conference on Learning Representations, volume 2024, pages 9025–9049, 2024.

Aneesh Pappu, Batu El, Hancheng Cao, Carmelo di Nolfo, Yanchao Sun, Meng Cao, and James Zou. Multi-agent teams hold experts back. arXiv preprint arXiv:2602.01011, 2026.

Alistair Reid, Simon O’Callaghan, Liam Carroll, and Tiberio Caetano. Risk analysis techniques for governed llm-based multi-agent systems. arXiv preprint arXiv:2508.05687, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open

language models. arXiv preprint arXiv:2402.03300, 2024.

Rui Sheng, Yukun Yang, Chuhan Shi, Yanna Lin, Zixin Chen, Huamin Qu, and Furui Cheng. Dills: Interactive diagnosis of llm-based multi-agent systems via layered summary of agent behaviors. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems, pages 1–17, 2026.

AQ Team. Mrlx: A multi-agent reinforcement learning framework. https://github.com/AQ-MedAI/ MrlX, 2025.

Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and Hoang D Nguyen. Multi-Agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322, 2025.

Chi Wang, Qingyun Wu, and AG2 Community. AG2: Open-source AgentOS for AI agents. https: //github.com/ag2ai/ag2, 2024a.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024b.

Yawen Wang, Wenjie Wu, Junjie Wang, and Qing Wang. From flat logs to causal graphs: Hierarchical failure attribution for llm-based multi-agent systems. arXiv preprint arXiv:2602.23701, 2026.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. Advances in Neural Information Processing Systems, 37: 95266–95290, 2024c.

Zhaohui Geoffrey Wang. AgentTrace: Causal graph tracing for root cause analysis in deployed multi-agent systems. arXiv preprint arXiv:2603.14688, 2026.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, et al. Webwalker: Benchmarking llms in web traversal. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10290–10305, 2025.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. AutoGen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023.

Zhiheng Xi, Chenyang Liao, Guanyu Li, Zhihao Zhang, Wenxiang Chen, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, et al. AgentPRM: Process reward models for llm agents via step-wise promise and progress. In Proceedings of the ACM Web Conference 2026, pages 4184–4195, 2026.

Yizhe Xie, Congcong Zhu, Xinyue Zhang, Tianqing Zhu, Dayong Ye, Minfeng Qi, Huajie Chen, and Wanlei Zhou. From spark to fire: Modeling and mitigating error cascades in llm-based multi-agent collaboration. arXiv preprint arXiv:2603.04474, 2026.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Ashley Ge Zhang, Victor Bursztyn, Gromit Chan, Shunan Guo, Eunyee Koh, Steve Oney, and Jane Hoffswell. Convomap: Interactive visualizations for exploring complex conversations in multi-agent

systems. In 2025 IEEE Symposium on Visual Languages and Human-Centric Computing (VL/HCC), pages 325–336. IEEE, 2025a.

Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. Agen-Tracer: Who is inducing failure in the llm agentic systems? arXiv preprint arXiv:2509.03312, 2025b.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, et al. Which agent causes task failures and when? on automated failure attribution of llm multi-agent systems. International Conference on Machine Learning, 2025c.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Findings ofthe Associationfor Computational Linguistics, pages 10495–10516, 2025d.

## A A Heuristic Algorithm for IF Severity Calculation

Let an MAS trace contain m IFs and L parsed trace steps. For IF i, let $w _ { t y p e } ( i )$ be its error-type weight and $w _ { a b s } ( i )$ be its absorption weight. The trace-level IF risk r is:

$$
r = \frac { \sum _ { i } w _ { t y p e } ( i ) * w _ { a b s } ( i ) } { m a x ( 1 , L ) }
$$

The score is therefore higher when a trace contains more IFs, more intrinsically dangerous IF types, less effective absorption mechanisms, or a higher density of IFs per trace step.

We assign heuristic values to $w _ { t y p e }$ and $w _ { a b s } ,$ as illustrated in Table 7 and Table 8. For $w _ { t y p e }$ , the highest weights are assigned to Hallucination and Reasoning Error since fabricated evidence and incorrect inferences silently contaminate downstream reasoning and directly degrade answer quality. Mid-range weights cover Wrong Action and Misunderstanding as bad navigation, tool, or planning actions can derail a trajectory but are often recoverable, while misreading task or role constraints is serious because subsequent steps may optimize for the wrong objective. Lower weights apply to Failed Tool Use, Malformed Output, Incomplete Action, and Unnecessary Action. Tool failures are frequently externally caused and retryable, missing follow-through is often repaired by the same or another agent, and redundant work mostly consumes budget without inflicting semantic damage.

For $w _ { a b s } ,$ , the lowest weight is reserved for Trivial Issue, capturing failures that are mechanical, redundant, ignored, or otherwise absorbed without substantive consequences. Self-correction sits slightly higher, since the originating agent itself detects and repairs the local failure, while Self-workaround reflects the same agent bypassing the issue via a different route rather than directly fixing it. Mid-range weights describe repairs that depend on other components: Peer Repair captures useful compensation by another agent at the cost of added coordination dependence, and Orchestrator Intervention reflects a manager or planner redirecting the trajectory while leaving the original worker state unrepaired. The highest weight, Unresolved Propagation, signals that the IF was carried forward or implicitly accepted, leaving the local failure active downstream and fully contributing to residual risk.

While our current weights are derived from qualitative analysis of error severity, we conducted a sensitivity study. On best-of-N selection, we perform 1,000 random perturbation trials, where each weight was independently varied by up to ±10%. We found that the resulting candidate rankings remained highly consistent with the original rankings, with the global Spearman rank correlation remaining above 0.993 across trials. On RL training, we vary the IF type and absorption weights on AG2/GSMPlus while keeping the remaining training protocol fixed. Table 9 reports the results. Our original values achieve the lowest per-task IF cost and count among the evaluated weight settings, while being suboptimal on IF count density. However, it also exhibits limitations on IF cost density. We leave a more systematic exploration of optimal parameters as future work.

Table 7: The heuristic values of $w _ { t y p e }$ for each error type.
<table><tr><td>Error Type</td><td> $w _ { t y p e }$ </td></tr><tr><td>Hallucination</td><td>1.30</td></tr><tr><td>Reasoning Error</td><td>1.20</td></tr><tr><td>Wrong Action</td><td>1.10</td></tr><tr><td>Misunderstanding</td><td>1.10</td></tr><tr><td>Failed Tool Use</td><td>1.00</td></tr><tr><td>Malformed Output</td><td>1.00</td></tr><tr><td>Incomplete Action</td><td>0.90</td></tr><tr><td>Unnecessary Action</td><td>0.60</td></tr></table>

Table 8: The heuristic values of $w _ { a b s }$ for each absorption pattern.
<table><tr><td>Absorption Pattern</td><td> $w _ { a b s }$ </td></tr><tr><td>Trivial Issue</td><td>0.35</td></tr><tr><td>Self-correction</td><td>0.55</td></tr><tr><td>Self-workaround</td><td>0.65</td></tr><tr><td>Peer Repair</td><td>0.75</td></tr><tr><td>Orchestrator Intervention</td><td>0.80</td></tr><tr><td>Unresolved Propagation</td><td>1.40</td></tr></table>

## B Data Consent and Ethical Considerations

Our study involved a small number of human annotators for two purposes: (a) auditing IF annotations produced by the LLM judge, and (b) performing open coding to derive the taxonomies described in Section 2.3. All annotators were members of the research team who participated voluntarily as part of their research role. Prior to commencing their tasks, each participant was thoroughly briefed on the research objectives, the specific use of their contributions (validating automated IF tags and building the taxonomies presented herein), the commitment to aggregate and report all data anonymously, and their unrestricted right to opt out at any stage. Formal informed consent was secured from every annotator before the evaluation phase began. In accordance with privacy standards, this study neither gathered nor published any demographic details, personal identifiers, or sensitive information regarding the participants. Additionally, the research protocol was formally reviewed and authorized by our institution’s Human Research Ethics Committee.

## B.1 Open Coding Instructions for Annotators

The goal of this round is to explore and map the space of implicit failure-related phenomena in MAS traces. All coders will read a small sample of MAS traces from a benchmark we curated. Use open coding to mark segments that are relevant to MAS implicit failures (broadly construed). Generate initial codes (short labels + notes) describing what is happening at those segments. Bring these codes to later meetings, where they will be clustered into higher-level concepts and tentative failure types. There is no expectation that you agree perfectly in this round. Diversity in codes is valuable.

For each assigned trace: Skim the whole trace first Get a sense of the task, involved agents, and how the trace ends (success/failure/mixed). Note roughly where things start to go wrong (if they do). Hint: The traces are lengthy with much repetitive info. You may use a LLM to help with your interpretation. Do line-by-line / segment-by-segment reading in the interface The interface will show the trace as a sequence of steps. Each “step” may be one agent message, one tool call + response, or a small contiguous block of log lines. Mark segments that seem relevant to implicit failure. Highlight the segment (a step, or a portion of a step). Attach one or more open codes and a brief note (1–3 sentences). Repeat until you reach the end of the trace. Not every step needs a code. After each trace, spend 5–10 minutes reviewing your own codes Merge obviously duplicated codes. Clarify unclear labels. Jot down any emerging patterns you notice (for discussion later).

Table 9: Sensitivity to IF severity weights on AG2/GSMPlus. Uniform sets the corresponding weights to one. Perturbed changes the original weights by up to 10%. All metrics are measured on the same evaluation benchmark.
<table><tr><td>Weights Accuracy</td><td colspan="5">IF Cost / Task # IF / Task IF Cost / 1K Tokens # IF / 1K Tokens</td></tr><tr><td>Original</td><td>72.21</td><td>0.34</td><td>0.30</td><td>1.54</td><td>1.33</td></tr><tr><td>Uniform absorption</td><td>72.42</td><td>0.36</td><td>0.45</td><td>0.84</td><td>1.04</td></tr><tr><td>Uniform type</td><td>71.96</td><td>0.57</td><td>0.95</td><td>1.07</td><td>1.79</td></tr><tr><td>Perturbed</td><td>73.29</td><td>0.46</td><td>0.60</td><td>1.04</td><td>1.37</td></tr></table>

## C Auditing IF Annotation

Previously unidentified IFs. The human audit identified 21 previously unlabeled IFs across 11 traces. The authors of the paper inspected these IFs and found that they concentrated in several recurring patterns: A prominent group involved unsupported execution-state claims in long software-engineering traces. For example, an agent reported that tests had passed without executing them, or proceeded as though a code modification had been applied when the underlying codebase had not changed. Other omissions involved incomplete actions that became visible only through a later orchestrator response, such as opening a date filter without setting the requested date, or navigating to a search service without issuing the requested query. We also observed malformed tool arguments and intermediate outputs that were subsequently consumed by later steps.

The five IFs independently recovered by both annotators provide the clearest examples. Two occurred in a Magentic-One trace in which a worker repeatedly failed to apply the requested date filter and was redirected by the orchestrator. In an AG2/GSMPlus trace, the solver introduced an unsupported numerical assumption that was used in later reasoning, although the system still reached the correct answer. In a WebWalker trace, the agent searched using a year that contradicted the task requirement; in another, an HTML fragment was accidentally included in a button argument, producing an invalid tool action.

Cases identified by only one annotator were concentrated in situations requiring more global interpretation. Long HyperAgent traces, for example, require the reviewer to connect a claimed code or test state with execution evidence appearing much earlier in the trace. Other cases lie near the boundary between an incomplete action and ordinary task decomposition: one annotator may regard a worker’s partial response as an error, while another may regard it as an intermediate result that the orchestrator is expected to refine. Persistent errors also introduce granularity differences, since a reviewer may annotate each repeated manifestation while another records only the earliest source event. These cases explain why independently discovered omissions exhibit lower agreement than review of an already localized IF.

Cross-judge agreement and disagreement. The four-LLM-judge audit produced 174 events supported by at least three judges, including 81 detected by all four. Agreement was strongest for events with explicit trace evidence, such as malformed tool arguments, unsupported intermediate claims, failed actions followed by an observable repair, and incomplete operations explicitly recognized by an orchestrator.

The main cross-judge difference was the detection boundary rather than failure to recognize a common core. Claude and GLM tended to annotate more fine-grained or repeated process irregularities, whereas Gemini and DeepSeek applied a more selective threshold. Judges also differed in whether repeated manifestations of the same underlying mistake should be represented as separate IFs. This pattern helps explain how Gemini can cover 90.2% of the majority-supported events while some events detected by only one or two judges remain outside its annotation set.

Directions for improving annotation. Our current selection of Gemini 3.1 Pro to identify IFs is based on a rationale where precision is prioritized to recall. Our audits suggest several practical improvements that we plan to implement in the future work. First, persistent manifestations of the same underlying error should be grouped under a common event identifier, with the earliest source step distinguished from later propagation steps. Second, long or nested traces would benefit from explicit links between claims about tool execution, code modification, or test results and the corresponding execution evidence. Finally, multi-judge agreement can be retained as an uncertainty signal rather than forcing every candidate into an equally confident binary label.

## D Additional Implementation Details

## D.1 Details of the GRPO and IF-GRPO studies

We provide the additional implementation details of the GRPO and IF-GRPO studies (Section 3.2) below. Data splits. AG2/GSMPlus uses 650 GSM8K training questions, 130 GSM8K validation questions, and 2,400 GSMPlus questions for testing. AG2/OlympiadBench uses fixed partitions of 650/130/130 questions for training/validation/testing. Validation questions are excluded from policy updates.

Training parameters. We use AdamW with constant learning rate $1 0 ^ { - 6 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 8 )$ weight decay 0.1, and PPO ratio-clipping bounds [0.8, 1.28]. Both the reference-policy KL coefficient and entropy regularization coefficient are zero. Training decoding uses temperature 0.7 and top-p = 0.95, with separate thinking mode disabled, while evaluation uses greedy decoding. AG2 allows up to 15 assistant calls, 2,048 generated tokens per call, and 8,192 per trace. Teamwork uses 21 calls (four initial answers, four discussion rounds, and one final answer), with 1,024 tokens per call.

IF scoring and evaluation. In IF-GRPO, IF cost only modulates positive outcome advantages for successful traces in mixed groups. Failed-trace advantages are unchanged, and all-success/all-failure groups have zero advantage. Entire zero-advantage batches skip optimizer updates. Outcome advantages use sample-standard-deviation normalization with ϵ = 10<sup>−6</sup>. Qwen3.8-Flash-Next uses temperature 0 for IF detection and absorption labeling.

## D.2 Prompt templates

We provide the prompt templates we use for IF detection (Listings 1 and 2), absorption pattern annotation (Listings 3 and 4), and counterfactual replay (Listings 5 and 6) below.

You are an expert analyzer of LLM-based multi-agent system execution traces.   
Identify implicit failures using only the task prompt and execution trace. An implicit failure is a   
concrete, trace-observable process error that does not individually determine the final outcome   
of the trace. Determine whether the event meets this definition, but do not classify its   
absorption pattern.   
Return only valid JSON matching the requested schema, with no markdown or extra text.

Listing 2: The user prompt template for IF detection.   
Analyze this multi-agent execution trace for concrete local process errors.   
## Task   
{{TASK\_PROMPT}}   
## Execution Trace   
{{EXECUTION\_TRACE}}   
## Instructions

Identify all clear local implicit errors visible in the trace. An implicit error is a specific   
mistake, failed action, unsupported claim, malformed tool use, invalid inference,   
misunderstanding, omission, hallucination, or materially unnecessary action at a particular   
trace step, which does NOT individually determine the final outcome of the trace.   
For each implicit error, return:   
- error\_step: the step index at which the error occurs   
- error\_agent: the agent responsible for the error   
- error\_type: one of failed\_tool\_use, wrong\_action, unnecessary\_action, hallucination,   
incomplete\_action, reasoning\_error, misunderstanding, malformed\_output   
error\_description: a concise description of what is wrong, grounded in trace evidence   
error\_evidence: a short excerpt or close paraphrase of the trace evidence   
- severity: low, medium, or high, based only on the local process error   
- confidence: high, medium, or low   
Detailed explanation of the error type taxonomy:   
| Type | Description | Example |   
| --- |   
| Failed Tool Use | A tool call fails, is blocked, times out, cannot execute, or returns unusable   
output because of tool or environment issues. | FileSurfer attempts to read an MP3 audio file as   
text and receives a conversion exception. |   
| Wrong Action | The agent takes an action that is inappropriate for the current task state or   
unlikely to make progress. | WebSurfer clicks page text rather than a valid link or play button,   
leaving the browser on the same page. |   
| Unnecessary Action | The agent performs redundant, low-value, or repetitive actions that consume   
steps without advancing the task. | WebSurfer handles a cookie prompt instead of following the   
already identified target page. |   
| Hallucination | The agent fabricates information, tool behavior, source content, or intermediate   
evidence not supported by the trace. | The agent invents a ranked list item that is not present   
in the retrieved source. |   
| Incomplete Action | The agent attempts an action but omits required details, formatting, inputs,   
or follow-through. | FileSurfer repeatedly returns the first PDF page instead of searching or   
navigating to the relevant pages. |   
| Reasoning Error | The agent makes an incorrect inference, introduces or propagates incorrect   
information, repeats a known-bad strategy, misses relevant evidence, or terminates from   
insufficient reasoning. | The assistant swaps document statuses across several listed books, or   
the orchestrator identifies the wrong single from an album despite available evidence. |   
| Misunderstanding | The agent misunderstands the task, role, constraints, or operational setting. |   
The orchestrator fails to recognize that another agent has already reached the correct page and   
restarts the search. |   
| Malformed Output | The agent emits syntactically malformed text or a corrupted structured block,   
but execution continues. | WebWalker emits a memory block with a stray closing quote and brace   
before recovering in the next step. |   
Return this JSON schema:   
"implicit\_failures": [   
{   
"error\_step": 5,   
"error\_agent": "AgentName",   
"error\_type": "reasoning\_error",   
"error\_description": "The agent reports an unsupported and internally inconsistent count.",   
"error\_evidence": "The relevant trace evidence.",   
"severity": "medium",   
"confidence": "high"   
}   
],   
"summary": {   
"total\_implicit\_failures": 1,   
"overall\_process\_assessment": "Brief trace-only assessment."   
}   
}

Listing 3: The system prompt template for IF absorption pattern annotation.
<table><tr><td>You are an expert analyzer of how multi-agent systems handle local process errors.</td></tr><tr><td>Classify the absorption pattern using only the task prompt, the given local-error annotation, and execution-trace evidence. Do not use or infer the expected answer, final-answer correctness, trace outcome, success/failure split, or decisive-error annotation. Do not assume that a correct final answer makes an error trivial.</td></tr></table>

Listing 4: The user prompt template for IF absorption pattern annotation.
<table><tr><td>Classify how the execution trace handles the annotated local error.</td></tr><tr><td>## Task {{TASK_PROMPT}}</td></tr><tr><td></td></tr><tr><td>## Local error annotation {{ERROR_ANNOTATION}}</td></tr><tr><td></td></tr><tr><td>## Trace context {{EXECUTION_TRACE}}</td></tr><tr><td>## Absorption labels</td></tr><tr><td>- self_corrected: The same agent recognizes and repairs its own local error. - self_workaround: The originating agent bypasses the local error through a genuinely different</td></tr><tr><td>route, query, tool, source, or implementation strategy. - peer_repair: A different non-managerial agent corrects the error or supplies a replacement result that is used downstream.</td></tr><tr><td>orchestrator_intervention: A manager, planner, or orchestrator recognizes the problem and redirects or corrects the trajectory. unresolved_propagation: The local error is not repaired before its erroneous artifact is accepted</td></tr><tr><td>or used by at least one downstream step as evidence, a premise, memory, goal, calculation input, action/tool input, patch input, or justification. trivial_issue: A catch-all for absorption patterns with low analytic value, including mechanical</td></tr><tr><td>retries that do not change strategy, duplicate work, ignored or abandoned actions, tool- or environment-level absorption, runtime blocking, timeout, and trace termination. Also use it when the available trace evidence is insufficient to distinguish a trivial issue from another absorption pattern. ## Clarification</td></tr></table>

```jsonl
evidence of top-down command or plan modification for orchestrator_intervention; if the fix
comes from a same-level collaborator without altering the overall task topology, use peer_repair
- To distinguish between unresolved_propagation and trivial_issue, the decisive factor is whether
the erroneous artifact--the faulty output, incorrect calculation, or flawed premise--is
demonstrably consumed by at least one downstream step. Apply unresolved_propagation only when
the trace provides clear evidence that the artifact is explicitly used as an input, evidence,
premise, memory update, or justification in a subsequent action, regardless of whether the
overall task ultimately succeeds. Conversely, apply trivial_issue if the issue is contained
locally by a mechanical retry, ignored, or occurs in a terminated or abandoned branch and does
not demonstrably feed into later processing.
Choose exactly one label. Return:
{
"absorption_pattern": "self_corrected|self_workaround|peer_repair|
orchestrator_intervention|unresolved_propagation|
trivial_issue",
"confidence": "high|medium|low",
"rationale": "One concise sentence grounded in the downstream trace evidence."
}
```

## Listing 5: The system prompt template for counterfactual local repair.

You are an expert in LLM-based Multi-Agent Systems. Given a partial execution trace where a specific   
agent step was wrong, you produce a corrected replacement of that single step in the agent's   
voice. You must:   
- Preserve the agent's persona, tool-call format, and writing style as evident from the trace.   
- Fix only the identified error; do not anticipate or pre-empt later steps.   
- Output ONLY the corrected message body. No commentary, no JSON wrapping, no leading "FixedMessage   
:" labels. Just the literal text the agent should have produced.

## Listing 6: The user prompt template for counterfactual local repair.

```markdown
## Task being solved by the multi-agent system
{question}
## Conversation history up to (but NOT including) the buggy step
{prefix}
## Buggy step (step {step_index}, agent: {agent}) -- this is what the agent actually said and what
we want to replace
<<<BUGGY MESSAGE BEGIN>>>
{buggy_message}
<<<BUGGY MESSAGE END>>>
## What was wrong with this step (annotation from an analyzer)
- error_type: {error_type}
description: {error_description}
severity: {severity}
## Your job
Produce a single corrected version of the buggy step that the {agent} agent should have output
instead.
Stay in the same persona and format. Do not include any text other than the message body itself.
```