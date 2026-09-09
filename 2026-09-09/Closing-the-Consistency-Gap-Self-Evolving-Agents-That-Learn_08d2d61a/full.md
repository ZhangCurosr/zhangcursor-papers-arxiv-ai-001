# Closing the Consistency Gap: Self-Evolving Agents That Learn to Stay on Course

Evelyn Duesterwald<sup>1</sup>, Benjamin Elder<sup>1</sup>, Lilian Ngweta<sup>1</sup>, Shashanka Ubaru<sup>2</sup>, Malgorzata Zimon<sup>2</sup>

<sup>1</sup>IBM Software Innovation Lab <sup>2</sup>IBM Research

## Abstract

Large language model (LLM)-powered agents can be accurate on average yet unreliable in production, a discrepancy that has been observed but remains largely unaddressed. When given the same task five times, a ReAct agent on the AppWorld benchmark using GPT-4.1 succeeds in all five runs only 53% of the time, even though its per-run pass rate averages 77%. We call this 24-point shortfall the consistency gap, and we argue that addressing it is a precondition for trustworthy AI agent deployment. We present a self-evolving agent framework that reduces this gap by identifying unstable, lowconsistency steps in agent trajectories and converting them into episodic memory the agent can draw on in future runs. At its core is a Consistency Analyzer that pinpoints where and why a trajectory is likely to flip across executions, and a Guideline Generator that converts the diagnosis into targeted guidelines, committed to memory and injected into future agent executions on similar tasks. On AppWorld with ReAct/GPT-4.1, our framework raises the fraction of tasks that succeed in all five runs by +16 points on same-task evaluation and +13 points on similar-task generalization.

## Code: https://github.com/AgentToolkit/altk-evolve

## 1 Introduction

Large language model (LLM)-powered agents have become the technology of choice for automating tasks that span from web navigation to API orchestration to interactive coding. Published benchmark accuracy numbers miss an underappreciated caveat: agents can be accurate on average but unreliable on repeat, since the same task asked of the same agent often produces different outcomes from one run to the next. We call this shortfall the consistency gap, and argue that reducing it is a precondition for the trust and reliability required for production deployment.

The gap is often large enough to matter in practice. On the test normal split of AppWorld (Trivedi et al. 2024), a ReAct (Yao et al. 2023) agent backed by GPT-4.1 achieves a mean pass rate of 77% across five runs (Mean@5), but consistently passes all five runs on only 53% of tasks (Passˆ5). This 24.4-point shortfall concentrates in a substantial population of tasks (∼32% of the benchmark) that mix passes and failures across runs, the scenarios where mean accuracy numbers are least informative about what a user actually experiences.

![](images/0706bc9aed9edcf091315d400b3b125835ecd496d7d7b8af7ed7ffbd093ac659.jpg)  
Figure 1: Consistency guideline pipeline: Detect (1) and Generate (2).

The design of our framework to address this gap rests on three observations. First, consistency is not the same as accuracy: a high mean pass rate can co-exist with a low consistent pass rate, so treating consistency as a first-class objective requires dedicated metrics (Section 2.1), an instrument that estimates it without re-running every task many times (Section 3.1), and a mechanism that targets it specifically (Section 3.2). Second, inconsistency has a measurable token-level fingerprint: at each inference point the model induces a distribution over next tokens, and a flat distribution with near-tied tokens lets the agent’s decision flip from run to run even under temperature-zero decoding; this can be estimated without model internals by resampling and measuring response variability, the basis of our Consistency Analyzer. Third, consistency can be improved by injecting consistency guidance at the prompt level: a short naturallanguage consistency guideline replaces an uncertain, previously flip-prone decision with a stable one the agent makes the same way on every subsequent run - without touching the inference loop, agnostic to agent architecture, and confined to an offline analysis stage.

Approach. Our framework is a two-stage pipeline (Figure 1) that is triggered when a new trajectory becomes available. Both stages run offline. Detect: a black-box Consistency Analyzer resamples each inference point in the trajectory, producing a step- and trajectory-level consistency scorecard that pinpoints decisions at risk of flipping, without access to model internals or agent code. Generate: an LLM-based Guideline Generator converts each flip-prone step into one or more candidate consistency guidelines, organized (following Fang et al. (2026)) into strategy, recovery, and optimization categories. Generated guidelines are committed to memory, clustered and deduplicated, and retrieved by cosine similarity into the prompt of subsequent similar tasks, stabilizing the inference points that are similar to those previously flagged. We build on the trajectoryinformed memory framework of Fang et al. (2026) for storage, clustering, consolidation, and retrieval; the principal cost of this cross-execution learning is a delay before useful guidelines accumulate, bounded by a feedback dynamic in which each guideline shrinks the inconsistent-task population that future guidelines are drawn from.

Contributions. (1) A formalization and empirical characterization of the consistency gap (Section 2). (2) A black-box Consistency Analyzer (Section 3.1) that estimates step- and trajectory-level consistency from response variability via configurable resampling; aggregate trajectory consistency also correlates with trajectory pass/fail (AUROC 0.69), letting it support the generator’s own judgment of trajectory outcome where no ground-truth rating is available. (3) Consistency-targeted guideline generation (Section 3.2) driven by step-level inconsistency. (4) Empirical results (Section 4) on AppWorld with ReAct/GPT-4.1 and GPT-OSS-120B showing +16 pp Passˆ5 same-task and +13 pp similar-task generalization (168 tasks), plus a sample-budget sensitivity study.

## 2 Consistency Gap

A common issue with current AI agents is: the same task, asked of the same agent backed by the same model, can produce different outcomes from one run to the next. In this section we make this discrepancy explicit, empirically characterize its magnitude, and sketch its underlying causes.

## 2.1 Pass@k, Mean@k, and Passˆk

Let T be a benchmark of agent tasks. For each task $t \in \tau$ run the agent k independent times and let $X _ { t } ~ \in ~ \{ 0 , 1 \} ^ { k }$ record outcome of a run of t. Three task-level aggregation metrics of $X _ { t }$ are commonly reported in the agentic literature:

$$
\begin{array} { r } { \mathrm { P a s s } @ k ( t ) \ \triangleq \ \mathbf { 1 } \Big [ \big ( \sum _ { j = 1 } ^ { k } X _ { t } ^ { ( j ) } \big ) \ge 1 \Big ] , } \end{array}\tag{1}
$$

$$
\begin{array} { r } { \mathrm { M e a n @ } k ( t ) \ \triangleq \ \frac { 1 } { k } \sum _ { j = 1 } ^ { k } X _ { t } ^ { ( j ) } , } \end{array}\tag{2}
$$

$$
\begin{array} { r } { \mathrm { P a s s } ^ { \cdot } \mathbf { k } ( t ) \triangleq \mathbf { 1 } \Big [ \big ( \sum _ { j = 1 } ^ { k } X _ { t } ^ { ( j ) } \big ) = k \Big ] . } \end{array}\tag{3}
$$

Pass@k asks whether the agent succeeds at least once in k tries, isolating capability from sampling noise. Mean@k is the per-task mean pass rate, the standard reported accuracy. Passˆk, by contrast, asks whether the agent succeeds on all k tries, isolating consistency from capability. For k independent Bernoulli trials with success probability $p ,$ the three satisfy E[Pass $\mathbf { \hat { k } } ] = p ^ { k } \leq \mathbb { E } [ \mathbf { M e a n } \mathbf { \tilde { @ } } k ] = p \leq \mathbb { E } [ \mathbf { P a s s } \mathbf { \ @ } k ] =$ $1 \bar { - } ( 1 \bar { - } p ) ^ { k }$ , with equality only at $\bar { p } \in \{ 0 , 1 \}$ . Aggregated over a benchmark, Pass@k, Mean@k and Passˆk refer to averages of the per-task quantities over $\tau$

Consistency gap. We define the consistency gap as $\mathrm { G a p @ } k ( { \mathcal T } ) \ \triangleq \ \mathrm { M e a n @ } k ( { \mathcal T } ) - \mathrm { P a s s } ^ { \cdot } \mathbf { k } ( { \mathcal T } ) ^ { 1 }$ , measured in percentage points (pp): it quantifies how much apparent accuracy is unreliable, i.e., may evaporate under repeated deployment. Figure 2 illustrates this for a ReAct agent on AppWorld $( k = 5 )$ , plotting Mean@5, Passˆ5, and the gap between them, per difficulty level and model.

Normalized consistency. The raw gap confounds capability with consistency, since it is mechanically bounded above by Mean@k: a low-accuracy model can exhibit at most a small absolute gap regardless of how variable its behavior is. We therefore also define normalized consistency Consistency ${ \mathrm { \Delta } } ^ { \mathrm { \prime } } \ @ k ( { \mathcal { T } } ) \ \triangleq \ { \mathrm { P a s s } } ^ { \cdot } { \mathrm { k } } ( { \mathcal { T } } ) / { \mathrm { M e a n @ } } k ( { \mathcal { T } } ) \ \in \ [ 0 , 1 ]$ which rescales Passˆk by the model’s own mean accuracy: 1 means the agent passes every run on every task it can ever solve, near 0 means successes are almost entirely nonreproducible. Substituting gives Gap@k = Mean@k · (1 − Consistency@k), so two agents with identical absolute gaps can have different Consistency@k whenever their mean accuracy differs.

## 2.2 Empirical Characterization of the Consistency Gap

We measure the consistency metrics for a ReAct (Yao et al. 2023) agent backed by two LLMs of different capabilities, GPT-4.1 and GPT-OSS-120B, evaluated on the 168-task test normal split of the AppWorld benchmark (Trivedi et al. 2024) with $\bar { k } = 5$ runs per task and the standard App-World grader (Figure 2).

Absolute gap. The absolute gap is large for both models (24.4 pp for GPT-4.1 and 23.8 pp for GPT-OSS-120B), despite the wide capability difference but distributes differently across difficulty. With GPT-4.1 it grows monotonically (17.5/25.0/30.2 pp for easy/medium/hard): harder tasks are both less likely to succeed and disproportionately likely to flip. With GPT-OSS-120B the pattern inverts, peaking on easy (38.6 pp) and shrinking on hard (9.5 pp) because the hard-task Mean@5 itself collapses to 9.5%, mechanically capping the gap. In both cases the consistent fraction of the benchmark falls far below mean accuracy, 53% vs. Mean@5 of 77% for GPT-4.1, only 10% vs. 34% for GPT-OSS-120B, so standard accuracy numbers are substantially uninformative about what a deployed-agent user actually experiences.

Normalized consistency. Normalizing by each model’s own mean accuracy removes the capability ceiling. GPT-4.1’s aggregate consistency is 0.69, declining to 0.51 on hard tasks: even where it can occasionally solve a task, it reliably delivers all-five-pass only about half the time. GPT-OSS-120B is more severe: aggregate consistency is only 0.30, collapsing to 0.00 on hard tasks - it never passes all five runs on a hard task it can occasionally solve. The $2 \times$ difference (0.69 vs. 0.30) shows GPT-OSS-120B’s successes are almost entirely non-reproducible, while GPT-4.1 retains meaningful consistency across difficulty levels.

![](images/26c80cb45e88ca289fd5155fb52de6e69cf44ad277f424f395687a1e45b3b938.jpg)  
(a) ReAct / GPT-4.1

![](images/009de7f25ec064cbeae3241bb387af496ea533aefdb5630e39107f3fbddf9e2a.jpg)  
(b) ReAct / GPT-OSS-120B  
Figure 2: The consistency gap on AppWorld for a ReAct agent backed by (a) GPT-4.1 and (b) GPT-OSS-120B.

The gap is unlikely to be idiosyncratic to a single agent or model: in informal testing we have also observed similar patterns with other agent architectures (e.g., CUGA (Marreed et al. 2025)) and other models, though we have not quantified the magnitude of the gap in these settings.

These sources of inconsistency trace back to the shape of the model’s token-probability distribution at each inference point (Section 3.1), motivating our approach: identify the inference points with flat response distributions and generate prompt-level guidance to sharpen them in future executions.

## 3 Consistency Guideline Generation

We build on the trajectory-informed agentic memory framework of Fang et al. (2026)<sup>2</sup>, which treats episodic memory as a primary mechanism for agent self-improvement: an LLM extracts structured tips (guidelines) from a raw trajectory in three categories - strategy, recovery, optimization - clusters them into a dual-indexed memory (embedding plus category/priority/context/provenance metadata), and retrieves them at runtime by cosine-similarity or LLM-guided query. We inherit this machinery unchanged; our contribution is a new upstream stage: the pipeline of Figure 1 that decides what guidelines to extract, targeted at consistency rather than outcome alone.

The approach operates as two coupled loops. The operational loop executes user tasks: the agent runs, optionally augmented with guidelines retrieved from memory, and an observability layer captures the full trajectory. The offline analysis loop transforms each captured trajectory into new guidelines via the Detect and Generate stages of Figure 1, which enter this dual-indexed episodic memory and are retrieved as described above. Unlike the prior framework, generation here is grounded in the consistency scorecard rather than solely left to the generating LLM’s judgment: a step that succeeded but exhibited high response variability is just as eligible for guideline generation as one that observably failed. The two loops form a self-evolving cycle: as trajectory experience accumulates, the analyzer surfaces increasingly fine-grained sources of inconsistency and subsequent runs become measurably more consistent.

## 3.1 Consistency Analyzer

The Consistency Analyzer takes a single recorded agent trajectory $T \ = \ \langle s _ { 1 } , s _ { 2 } , . . . , s _ { n } \rangle$ and produces a consistency scorecard: a per-step consistency score $C ( s _ { i } ) ~ \in ~ [ 0 , 1 ]$ together with an aggregate trajectory score $C ( T ) \in [ 0 , 1 ]$ , plus diagnostic metadata identifying which steps are most likely to flip on a future run. The analyzer is black-box in the sense that it only re-invokes the underlying LLM with the agent’s recorded prompts, with no access to model logits, internal activations, or agent source code, making it applicable to any agent built on a hosted LLM endpoint.

Why Greedy Decoding Is Not Enough Inconsistency can arise at LLM inference points and is governed by the shape of the token-probability distribution there. Sharp distributions place most mass on a single token and are resilient to noise: minor perturbations from GPU floating-point arithmetic, request batching, or tokenizer state are not enough to reorder the top token. Flat distributions, by contrast, place comparable mass on several near-tied tokens and are vulnerable to that same noise: a small perturbation is enough to change which token comes out on top. This is why the problem cannot be fixed by decoding strategy alone: switching to greedy decoding, or to sampling with a fixed seed, only fixes how a given probability distribution is turned into a token choice, it does nothing about the fact that, on a hosted endpoint, the probabilities themselves shift slightly from run to run due to platform noise. So the same prompt to the same model can still yield different token choices on different runs, even under temperature-zero (greedy) decoding. Critically, the confidence of a single response (e.g., the pertoken log-probability of the chosen path) reveals only how confident the model was in the path it took, not how close the runners-up were, that is, how resilient the decision actually is to this noise. Predicting whether a step will flip on a future run therefore requires estimating the shape of the distribution over agent-meaningful outcomes, not just the depth of the one path taken, an estimate we obtain via resampling.

Step Consistency via Resampling For each inference step $s _ { i } ,$ the analyzer reissues the recorded prompt to the same LLM endpoint N times (typically temperature ≤ 0.5), yielding a response set ${ D _ { i } = \bar { \{ r _ { i } ^ { ( 1 ) } , . . . , r _ { i } ^ { ( N ) } \} } }$ that constitutes the empirical estimate of the response distribution. The step consistency score $C ( s _ { i } ) \in [ 0 , 1 ]$ measures how concentrated $D _ { i }$ is around its mode, using one of three primitive measures or a structured combination, selected by the detected response type.

Free-text and code responses. For free natural-language text (e.g., a reasoning trace) or program code, we estimate step consistency as mean pairwise cosine similarity of SBERT (Reimers and Gurevych 2019) sentence (or code) embeddings:

$$
C ( s _ { i } ) = \frac { 2 } { N ( N { - } 1 ) } \sum _ { j < k } \cos \big ( { \bf e } ( r _ { i } ^ { ( j ) } ) , { \bf e } ( r _ { i } ^ { ( k ) } ) \big ) ,\tag{4}
$$

where ${ \bf e } ( r )$ is the (code) embedding of response r. Embedding-space similarity rewards semantic equivalence despite differing syntax, so paraphrases and functionallyequivalent code are correctly treated as consistent.

Categorical responses. For categorical values such as a tool name, enum argument, or structured identifier, step consistency is the mean pairwise Jaccard similarity over tokenized values,

$$
C ( s _ { i } ) = \frac { 2 } { N ( N - 1 ) } \sum _ { j < k } J ( r _ { i } ^ { ( j ) } , r _ { i } ^ { ( k ) } ) ,\tag{5}
$$

where $J ( a , b ) = | V ( a ) \cap V ( b ) | / | V ( a ) \cup V ( b ) |$ over token sets $V ( \cdot )$ . Unlike embedding-based measures, Jaccard penalizes any token-level mismatch, appropriate where exact match is semantically required: a different argument key or enum value can produce a different downstream outcome.

Structured combination responses. Semi-structured responses (e.g., a JSON object with a free-text thoughts field alongside categorical action/action input fields) are decomposed into fields ${ \mathcal { F } } _ { i }$ , each mapped to one of the primitive types above, and combined as a weighted sum:

$$
C ( s _ { i } ) = \sum _ { f \in \mathscr { F } _ { i } } \delta _ { f } \cdot \sin _ { f } ( D _ { i } ^ { f } ) , \sum _ { f } \delta _ { f } = 1 ,\tag{6}
$$

where $D _ { i } ^ { f }$ projects the $N$ samples onto field $f , \delta _ { f }$ is a configurable importance weight (zero to suppress irrelevant fields), and sim $^ { 1 } f$ is the primitive measure for that field’s type. This lets the analyzer tolerate paraphrastic and code-level noise while still flagging meaningful divergence in fields whose exact value determines the agent’s next action.

Trajectory Consistency Aggregation Step-level scores alone do not predict whether the trajectory as a whole is at risk, so we aggregate them into a trajectory-level score by mean aggregation: $\begin{array} { r } { C _ { \mathrm { m e a n } } ( T ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \dot { C ( s _ { i } ) } } \end{array}$ . Mean consistency is fast to compute and, as we show in Section 4, provides a useful predictor of trajectory outcome. Its limitation is that it treats steps as independent: a high-consistency step that nonetheless propagates an upstream low-consistency decision is not penalized for its dependence on that decision.

## 3.2 Guideline Generation

The Consistency Analyzer surfaces where an agent is at risk of flipping; the Guideline Generator turns that diagnosis into a short, agent-readable instruction targeted at reducing the uncertainty that caused it in the first place. To integrate with the memory framework of Fang et al. (2026), guidelines follow the same strategy/recovery/optimization taxonomy, adapted for consistency: a strategy guideline enforces a pattern that was correct despite high variability (the agent guessed right); a recovery guideline steers around a step whose variability correlates with failure; and an optimization guideline instructs the agent to skip or simplify a highly variable step that is not strictly necessary for task completion.

Given a trajectory $T$ and its consistency scorecard, a step is flagged as inconsistent if its step consistency $C ( s _ { i } )$ falls below a threshold $\theta _ { C }$ (we use $\theta _ { C } ~ = ~ 0 . 8 5 )$ and it lies on the agent’s decision path. The trajectory, with its flagged steps marked, is passed to an LLM-based generator, which is instructed to generate guidelines that specifically target the flagged inconsistent steps. Appendix A shows example guidelines generated using GPT-4.1 from the trajectory and scorecard of an AppWorld task.

Generated guidelines enter the dual-indexed episodic memory with metadata for category, priority, application context, task category and provenance. At runtime, guidelines are retrieved by either cosine similarity against the incoming task description or by an LLM-guided selector that constructs a metadata-filtered query from the task. We refer to Fang et al. (2026) for the full details of storage management and retrieval, including deduplication, semantic clustering, and LLM-based merging of overlapping guidelines.

## 4 Evaluation

We evaluate our framework along three dimensions relevant to production deployment: whether consistency guidelines improve repeat-run reliability on the same task; whether they generalize to similar tasks without per-task tuning, the common case once a system has accumulated experience across many related tasks; and whether the analyzer’s aggregate trajectory consistency is a reliable enough proxy for trajectory outcome to support the guideline generator’s own judgment when no ground-truth rating is available.

## 4.1 Experimental Setup

We evaluate on AppWorld (Trivedi et al. 2024), an interactive coding-agent benchmark in which an agent operates a controlled simulation of personal apps via Python tool calls, graded by the AppWorld harness, on the test normal split (168 tasks). AppWorld groups tasks into scenarios of three variants each, which we exploit below to test generalization to similar tasks. We evaluate a ReAct (Yao et al. 2023) agent backed by GPT-4.1 and GPT-OSS-120B, running each task five times and reporting Passˆ5, Mean@5, and (where relevant) Pass@5 as defined in Section 2.1. All experiments were run against cloud-hosted model endpoints: GPT-4.1 on Microsoft Azure and GPT-OSS-120B on Amazon Web Services (AWS).

The consistency analyzer parses the ReAct agent response at each step to extract actions as categorical pairs: (API name, API params). Each extracted pair is scored by average pairwise Jaccard similarity, with a default resampling budget of $N \ = \ 3 0$ at temperature 0.5. For the same-task and similar-task evaluations, guidelines are generated from a single baseline trajectory per task by prompting the same model that produced it (acting as judge on its own scorecard), and retrieved at runtime by cosine similarity.

## 4.2 Same-Task Consistency Improvement

We first evaluate whether guidelines generated from one baseline trajectory of a task improve consistency on repeat runs of that same task; results for these experiments are illustrated in Figure 3.

GPT-4.1. Consistency guidelines lift aggregate Passˆ5 by +16.0 pp (53.0% → 69.0%, a 30% relative gain), largest in absolute terms on Medium tasks (+22.9 pp) and in relative terms on Hard tasks (+14.3 pp on a 31.7% baseline, a 45% relative lift). Mean@5 is never degraded, rising by +3.6 pp aggregate (+1.6 to +6.2 pp per difficulty level), supporting the design hypothesis that consistency-targeted memory improves Passˆk without sacrificing Mean@k. Normalized consistency rises by +0.17 aggregate, most on Medium (+0.22) and Hard (+0.21), where the baseline was most volatile. Full per-difficulty numbers, including similar-task columns, are in Table 1 (Appendix B).

GPT-OSS-120B. Starting from a substantially lower baseline $( \mathrm { P a s s } \hat { \cdot } 5 =  1 0 . 1 \%$ , Mean $\textcircled { a } 5 = \ 3 3 . 9 \% )$ , consistency guidelines improve aggregate Passˆ5 by +6.0 pp and Mean@5 by +4.8 pp. The relative Passˆ5 gain (59%) is nearly twice that of GPT-4.1 (30%), indicating the framework extracts more of the available headroom when the baseline is lower. Gains concentrate where the model already has meaningful pass rates: Easy improves by +8.7 pp $\mathrm { P a s s } ^ { \sim } 5 $ , while Hard, starting from $0 \% ,$ gains only +1.6 pp. Normalized consistency rises from 0.30 to 0.42 aggregate, most on Medium (+0.28). Full per-difficulty numbers are in Table 2 (Appendix B).

Consistency Gap. Figure 3 visualizes gap-reduction under same-task guidelines versus baseline. For GPT-4.1 (a), the aggregate gap to perfect consistency narrows from 0.32 to 0.15 (53% reduction) across every difficulty level. For GPT-OSS-120B (b), baseline gaps are far larger (0.70 aggregate, 1.00 on Hard) and guidelines make meaningful but partial inroads (to 0.58 aggregate): the model’s low raw capability caps how much guidelines alone can recover.

## 4.3 Generalization to Similar Tasks

A more demanding test is whether guidelines extracted from one task improve consistency on a different but similar task. Each AppWorld scenario consists of three similar task variants v1, v2, v3 that differ in parameter values and other details; we derive guidelines from $v _ { 1 }$ and inject them into runs of $v _ { 2 }$ and $v _ { 3 }$ so each of the 168 tasks is evaluated twice, against guidelines from each of its scenario’s other two variants (336 evaluations total)

GPT-4.1. Sibling-task guidelines lift aggregate Passˆ5 by +13.0 pp (Table 1), only 3 pp below same-task; Hard shows a small Mean@5 regression (−0.8 pp), likely because some sibling guidelines target API patterns that differ subtly in the sibling variant. That transfer occurs at all suggests the analyzer surfaces recurring patterns (e.g., paginated retrieval, credential management) rather than task-idiosyncratic ones.

GPT-OSS-120B. Sibling-task guidelines lift aggregate Passˆ5 by +8.7 pp (Table 2), exceeding the +6.0 pp sametask gain, with all four difficulty levels improving and no regressions; Hard tasks move from 0.00 to 0.29 normalized consistency, the largest single-cell gain across both tables.

Consistency Gap. Figure 3’s dark teal bars show similartask gap reduction tracking same-task closely for GPT-4.1 (aggregate gap 0.17 vs. 0.15), and substantially outperforming same-task on GPT-OSS-120B’s Hard tasks (0.29 vs. 0.13 closure from zero). Overall, guidelines reliably push normalized consistency upward across difficulty levels, with similar-task transfer matching or exceeding same-task closure in most cases.

## 4.4 Trajectory Consistency as an Outcome Predictor

Section 3.1 noted that aggregate trajectory consistency correlates with grader outcome; we quantify this using AUROC of aggregate consistency as a binary classifier of trajectory pass/fail. On 50 AppWorld test challenge tasks with a single ReAct/GPT-4.1 trajectory each, mean aggregate consistency reaches AUROC 0.699, an acceptable predictor. This makes aggregate consistency a useful signal for the Guideline Generator: rather than relying solely on the LLM’s own unaided judgment of whether a trajectory succeeded or failed, the generator can draw on this empirical stability measure to support that judgment, which is otherwise unavailable in production settings that lack a ground-truth outcome rating.

## 5 Related Work

Self-consistency, inference scaling, and uncertainty quantification. Self-consistency (Wang et al. 2023) samples multiple chains of thought and selects the majority answer; inference-scaling work (Snell et al. 2024) and reasoning models such as OpenAI’s o1 family generalize the same intuition, and agent variants use tests or an LLM judge to select among rolled-out solutions. We differ on two axes: resampling happens offline rather than at runtime, and the objective is consistency, not accuracy - variability is a diagnostic signal for where the prompt should be hardened, not a voting mechanism. Predictive entropy (Malinin and Gales 2021) overstates uncertainty when responses vary in wording but agree in meaning, since it measures spread over surface forms rather than over meaning; semantic entropy (Kuhn, Gal, and Farquhar 2023; Farquhar et al. 2024) addresses this via clustering; our decomposed structuredresponse score (Section 3.1) targets the partially-structured nature of agent responses similarly.

Agentic memory. Recent surveys (Zhang et al. 2025b; Du et al. 2025) chart this growing space. MEM0 (Chhikara et al. 2025), A-Mem (Xu et al. 2025), and MemGPT (Packer et al. 2023) store and retrieve conversational facts; Agent Workflow Memory (Wang et al. 2024), AgentRR (Feng et al. 2025), MemP (Fang et al. 2025), ReasoningBank (Ouyang et al. 2025), ACE (Zhang et al. 2025a), and Memento (Zhou et al. 2025) extract reusable workflows from traces, though naive accumulation can propagate errors (Xiong et al. 2025). We build on Fang et al. (2026), adding a consistencytargeted extraction signal atop its storage and retrieval machinery.

![](images/c2ad787cdd252986475ad8b1b8987f92354ee70c04bbc0396ca912408944cb4f.jpg)  
(a) ReAct / GPT-4.1

![](images/64d989f7918734b1477b20db5d8d3a1c68d840f93a92e51d27e97408416ed650.jpg)  
(b) ReAct / GPT-OSS-120B  
Figure 3: Consistency gap: normalized consistency (Passˆ5/Mean@5) with and without consistency guidelines, by task difficulty.

Reliability, reproducibility, and observability. App-World (Trivedi et al. 2024) is one of several benchmarks reporting multi-run statistics, and Ye et al. (Ye et al. 2026) show that such run-to-run variance is itself an underspecified property of self-improving, memory-based agents, sensitive to task order and evaluation protocol; our guideline-injection loop targets exactly this source of fragility. Our Passˆk vs. Mean@k formulation generalizes pass@k from code generation (Chen et al. 2021) to per-difficulty granularity, and lands within the broader reliability taxonomy of Rabanser et al. (Rabanser et al. 2026), whose twelve metrics span consistency, robustness, predictability, and safety; our gap is a targeted probe of their consistency axis. CUGA (Marreed et al. 2025) adopts a related notion through policy memory and can consume our guidelines directly. Closest in spirit is concurrent work by Jha et al. (Jha et al. 2026), who identify a reliability gap between Pass@k and Majority@k addressed via a graph-guided architecture; we target the same variability at the prompt level. Our analyzer also functions as an observability tool, complementary to work on runtime logs (Moshkovich et al. 2025), process discovery (Fournier, Limonad, and David 2025), process supervision (Liu et al. 2026), and interactive debugging (Hutter and Pradel 2026), but with a focus on surfacing flip-prone points for targeted guideline generation rather than runtime intervention.

## 6 Discussion and Limitations

Guideline validation. The pipeline described above admits every generated guideline directly into memory. A natural extension, which we leave to future work, is a validation stage between generation and memory admission: before a guideline is stored, re-run the analyzer’s resampling procedure at its source step with the guideline injected into the prompt, and admit it only if step consistency measurably improves and does not regress on a held-out set of related trajectories. This would guard against guidelines that are vague, conflict with existing instructions, or overfit to incidental details of the trajectory they were derived from, a known risk in agentic memory more broadly (Xiong et al. 2025).

Generality and future directions. We evaluated a Re-Act agent with GPT-4.1 and GPT-OSS-120B; the framework is agent-agnostic by design. A cross-stack study, including models from other model families and agents with existing policy memory such as CUGA (Marreed et al. 2025), is our next step.

Limitations. Our Passˆ5 comparisons assume comparable platform-side non-determinism between baseline and guideline-injected runs, which we cannot directly measure or control. We do not claim the framework eliminates inconsistency (a substantial gap remains, especially on Hard tasks) or that the analyzer is a verifier: it scores stability, not correctness, and a confidently-wrong decision looks consistent to it. Our default resampling budget (N = 30 per inference step) multiplies token cost roughly 30× per trajectory; reducing this cost without degrading guideline quality, e.g., via adaptive resampling that concentrates samples on candidate flat steps, is an important direction for future work.

## 7 Conclusion

Current LLM agents pass tasks “on average” in a way that masks how often they fail to pass the same task on repeat. We presented a framework that reduces this consistency gap through consistency-targeted episodic memory guidelines: a black-box Consistency Analyzer resamples each inference point to surface decisions at risk of flipping, and a Guideline Generator converts them into consistency guidelines. On AppWorld with ReAct and GPT-4.1, the pipeline lifts Passˆ5 by 16 points same-task and 13 points similartask with no accuracy degradation; aggregate trajectory consistency alone correlates with trajectory pass/fail (AUROC ≈ 0.69), letting the analyzer support the generator’s own judgment of trajectory outcome where no ground-truth rating exists.

# Appendix

## A Example Consistency Guidelines

Section 3.2 references the example guidelines below, generated by GPT-4.1 from the consistency scorecard for AppWorld task 0a9d82a 2 (“What is my longest limited-screen-time-to-1-hr habit streak, in number of days, as per my Simple Note habit tracking logs?”).

[strategy] Always verify API parameter requirements and expected input/output formats by consulting the API documentation before making calls, especially for search and data retrieval APIs.

[strategy] When retrieving paginated data, implement a robust loop that continues fetching pages until all results are retrieved, checking for pagination end conditions as specified in the API documentation.

[recovery] Explicitly handle authentication and token management by retrieving fresh credentials as needed and verifying token validity before making authenticated API calls.

[optimization] After retrieving and parsing data, implement validation checks before performing calculations or returning results.

## B Full Per-Difficulty Results

Tables 1 and 2 report the complete per-difficulty breakdown (baseline, same-task guidelines, and similar-task guidelines) underlying the aggregate numbers discussed in Section 4.

Table 1: Consistency results on AppWorld test normal, ReAct / GPT-4.1.
<table><tr><td colspan="4">Baseline</td><td colspan="3">+ Guidelines: Same Task</td><td colspan="3">+ Guidelines: Similar Task</td></tr><tr><td>Difficulty</td><td>Pass^5</td><td>Mean@5</td><td>Consistency</td><td>Pass^5</td><td>Mean@5</td><td>Consistency</td><td>Pass^5</td><td>Mean@5</td><td>Consistency</td></tr><tr><td rowspan="2">Aggregate Δ</td><td>53.0%</td><td>77.4%</td><td>0.68</td><td>69.0%</td><td>81.0%</td><td>0.85</td><td>66.0%</td><td>79.5%</td><td>0.83</td></tr><tr><td></td><td></td><td></td><td>+16.0 pp</td><td>+3.6 pp</td><td>+0.17</td><td>+13.0 pp</td><td>+2.1 pp</td><td>+0.15</td></tr><tr><td rowspan="2">Easy Δ</td><td>77.2%</td><td>94.7%</td><td>0.82</td><td>89.4%</td><td>98.2%</td><td>0.91</td><td>90.3%</td><td>97.3%</td><td>0.93</td></tr><tr><td></td><td></td><td></td><td>+12.2 pp</td><td>+3.5 pp</td><td>+0.10</td><td>+13.1 pp</td><td>+2.6 pp</td><td>+0.11</td></tr><tr><td rowspan="2">Medium ∆</td><td>52.1%</td><td>77.1%</td><td>0.68</td><td>75.0%</td><td>83.3%</td><td>0.90</td><td>66.6%</td><td>83.3%</td><td>0.80</td></tr><tr><td></td><td></td><td></td><td>+22.9 pp</td><td>+6.2 pp</td><td>+0.22</td><td>+14.5 pp</td><td>+6.2 pp</td><td>+0.12</td></tr><tr><td rowspan="2">Hard ∆</td><td>31.7%</td><td>61.9%</td><td>0.51</td><td>46.0%</td><td>63.4%</td><td>0.73</td><td>43.6%</td><td>61.1%</td><td>0.71</td></tr><tr><td></td><td></td><td></td><td>+14.3 pp</td><td>+1.6 pp</td><td>+0.21</td><td>+11.9 pp</td><td>−0.8 pp</td><td>+0.20</td></tr></table>

Table 2: Consistency results on AppWorld test normal, ReAct / GPT-OSS-120B.
<table><tr><td rowspan="2"></td><td colspan="3">Baseline</td><td colspan="3">+ Guidelines: Same Task</td><td colspan="3">+ Guidelines: Similar Task</td></tr><tr><td>Difficulty Pass^5</td><td>Mean@5</td><td>Consistency</td><td>Pass^5</td><td>Mean@5</td><td>Consistency</td><td>Pass^5</td><td>Mean@5</td><td>Consistency</td></tr><tr><td rowspan="2">Aggregate Δ</td><td>10.1%</td><td>33.9%</td><td>0.30</td><td>16.1%</td><td>38.7%</td><td>0.42</td><td>18.8%</td><td>40.5%</td><td>0.46</td></tr><tr><td></td><td></td><td></td><td>+6.0 pp</td><td>+4.8 pp</td><td>+0.12</td><td>+8.7 pp</td><td>+6.6 pp</td><td>+0.17</td></tr><tr><td rowspan="2">Easy △</td><td>28.1%</td><td>66.7%</td><td>0.42</td><td>36.8%</td><td>75.4%</td><td>0.49</td><td>43.0%</td><td>74.6%</td><td>0.58</td></tr><tr><td></td><td></td><td></td><td>+8.7 pp</td><td>+8.7 pp</td><td>+0.07</td><td>+14.9 pp</td><td>+7.9 pp</td><td>+0.16</td></tr><tr><td rowspan="2">Medium ∆</td><td>2.1%</td><td>27.1%</td><td>0.08</td><td>10.4%</td><td>29.2%</td><td>0.36</td><td>8.3%</td><td>31.2%</td><td>0.27</td></tr><tr><td></td><td></td><td></td><td>+8.3 pp</td><td>+2.1 pp</td><td>+0.28</td><td>+6.2 pp</td><td>+4.1 pp</td><td>+0.19</td></tr><tr><td rowspan="2">Hard ∆</td><td>0.0%</td><td>9.5%</td><td>0.00</td><td>1.6%</td><td>12.7%</td><td>0.13</td><td>4.8%</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>+1.6 pp</td><td>+3.2 pp</td><td>+0.13</td><td>+4.8 pp</td><td>16.7% +7.2 pp</td><td>0.29 +0.29</td></tr></table>

## References

AgentToolkit. 2026. altk-evolve: On-the-job learning for AI agents. https://github.com/AgentToolkit/altk-evolve. GitHub repository.

Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; et al. 2021. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374.

Chhikara, P.; Khant, D.; Aryan, S.; Singh, T.; and Yadav, D. 2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv preprint arXiv:2504.19413.

Du, Y.; Huang, W.; Zheng, D.; Wang, Z.; Montella, S.; Lapata, M.; Wong, K.-F.; and Pan, J. Z. 2025. Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions. arXiv preprint arXiv:2505.00675.

Fang, G.; Isahagian, V.; Jayaram, K. R.; Kumar, R.; Muthusamy, V.; Oum, P.; and Thomas, G. 2026. Trajectory-Informed Memory Generation for Self-Improving Agent Systems. arXiv preprint arXiv:2603.10600.

Fang, R.; Liang, Y.; Wang, X.; Wu, J.; Qiao, S.; Xie, P.; Huang, F.; Chen, H.; and Zhang, N. 2025. Mem<sup>p</sup>: Exploring Agent Procedural Memory. arXiv preprint arXiv:2508.06433.

Farquhar, S.; Kossen, J.; Kuhn, L.; and Gal, Y. 2024. Detecting Hallucinations in Large Language Models Using Semantic Entropy. Nature, 630: 625–630.

Feng, E.; Zhou, W.; Liu, Z.; Chen, L.; Dong, Y.; Zhang, C.; Zhao, Y.; Du, D.; Hua, Z.; Xia, Y.; and Chen, H. 2025. Get Experience from Practice: LLM Agents with Record & Replay. arXiv preprint arXiv:2505.17716.

Fournier, F.; Limonad, L.; and David, Y. 2025. Agentic AI Process Observability: Discovering Behavioral Variability. arXiv preprint arXiv:2505.20127.

Hutter, R.; and Pradel, M. 2026. AgentStepper: Interactive Debugging of Software Development Agents. arXiv preprint arXiv:2602.06593.

Jha, S.; Arora, R.; Bhavya; Zheutlin, N.; Toro Isaza, P.; Shwartz, L.; Deng, Y.; Sow, D.; Mahindru, R.; and Puri, R. 2026. Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation. arXiv preprint arXiv:2601.17915.

Kuhn, L.; Gal, Y.; and Farquhar, S. 2023. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. In Proceedings of the 11th International Conference on Learning Representations (ICLR).

Liu, Y.; Zhang, C.; Han, Z.; Liu, H.; Wang, Y.; Yu, Y.; Wang, X.; and Yin, Y. 2026. TrajAD: Trajectory Anomaly Detection for Trustworthy LLM Agents. arXiv preprint arXiv:2602.06443.

Malinin, A.; and Gales, M. 2021. Uncertainty Estimation in Autoregressive Structured Prediction. Proceedings of the International Conference on Learning Representations (ICLR).

Marreed, S.; Oved, A.; Yaeli, A.; Shlomov, S.; Levy, I.; Akrabi, O.; Sela, A.; Adi, A.; and Mashkif, N. 2025. Towards Enterprise-Ready Computer Using Generalist Agent. arXiv preprint arXiv:2503.01861.

Moshkovich, D.; Mulian, H.; Zeltyn, S.; Eder, N.; Skarbovsky, I.; and Abitbol, R. 2025. Beyond Black-Box Benchmarking: Observability, Analytics, and Optimization of Agentic Systems. arXiv preprint arXiv:2503.06745.

Ouyang, S.; Yan, J.; Hsu, I.-H.; Chen, Y.; Jiang, K.; Wang, Z.; Han, R.; Le, L. T.; Daruki, S.; Tang, X.; Tirumalashetty, V.; Lee, G.; Rofouei, M.; Lin, H.; Han, J.; Lee, C.-Y.; and Pfister, T. 2025. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. arXiv preprint arXiv:2509.25140.

Packer, C.; Wooders, S.; Lin, K.; Fang, V.; Patil, S. G.; Stoica, I.; and Gonzalez, J. E. 2023. MemGPT: Towards LLMs as Operating Systems. arXiv preprint arXiv:2310.08560.

Rabanser, S.; Kapoor, S.; Kirgis, P.; Liu, K.; Utpala, S.; and Narayanan, A. 2026. Towards a Science of AI Agent Reliability. arXiv preprint arXiv:2602.16666.

Reimers, N.; and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 3982–3992. Association for Computational Linguistics.

Snell, C.; Lee, J.; Xu, K.; and Kumar, A. 2024. Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters. arXiv preprint arXiv:2408.03314.

Trivedi, H.; Khot, T.; Hartmann, M.; Manku, R.; Dong, V.; Li, E.; Gupta, S.; Sabharwal, A.; and Balasubramanian, N. 2024. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q.; Chi, E.; Narang, S.; Chowdhery, A.; and Zhou, D. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In Proceedings ofthe 11th International Conference on Learning Representations (ICLR).

Wang, Z. Z.; Mao, J.; Fried, D.; and Neubig, G. 2024. Agent Workflow Memory. arXiv preprint arXiv:2409.07429.

Xiong, Z.; Lin, Y.; Xie, W.; He, P.; Tang, J.; Lakkaraju, H.; and Xiang, Z. 2025. How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior. arXiv preprint arXiv:2505.16067.

Xu, W.; Liang, Z.; Mei, K.; Gao, H.; Tan, J.; and Zhang, Y. 2025. A-MEM: Agentic Memory for LLM Agents. arXiv preprint arXiv:2502.12110.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In Proceedings of the 11th International Conference on Learning Representations (ICLR).

Ye, Q.; Li, Y.; Pruksachatkun, Y.; Zhang, J.; and Wu, C.- S. 2026. On the Fragility of Self-Improving Agents: Variance, Task Order, and Underspecification. arXiv preprint arXiv:2608.18066.

Zhang, Q.; Hu, C.; Upasani, S.; Ma, B.; Hong, F.; Kamanuru, V.; Rainton, J.; Wu, C.; Ji, M.; Li, H.; Thakker, U.; Zou, J.; and Olukotun, K. 2025a. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. arXiv preprint arXiv:2510.04618.

Zhang, Z.; Bo, X.; Ma, C.; Li, R.; Chen, X.; Dai, Q.; Zhu, J.; Dong, Z.; and Wen, J.-R. 2025b. A Survey on the Memory Mechanism of Large Language Model based Agents. ACM Transactions on Information Systems (TOIS). ArXiv:2404.13501.

Zhou, H.; Chen, Y.; Guo, S.; Yan, X.; Lee, K. H.; Wang, Z.; Lee, K. Y.; Zhang, G.; Shao, K.; Yang, L.; and Wang, J. 2025. Memento: Fine-tuning LLM Agents without Finetuning LLMs. arXiv preprint arXiv:2508.16153.