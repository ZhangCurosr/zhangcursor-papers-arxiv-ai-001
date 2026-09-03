# FUSE: An Evaluating Framework for Dangerous Capabilities of LLMs

Zhengyi Jin, Ru Zhang<sup>†</sup>, Xiao Chen, Xinbo Liu, Jiaxuan Lin, Jia Huang, Jianyi Liu, Zhen Yang School of Cyberspace Security, Beijing University of Posts and Telecommunications, China <sup>†</sup>Corresponding author: zhangru@bupt.edu.cn

## Abstract

Fragmented safety evaluation undermines the governance of dangerous AI capabilities. We present a modular framework that evaluates each model through three orthogonal pipelines— Knowledge (K), Defense (D), and Harm (H)—under a unified protocol, aggregating results into a standardized dangerouscapability profile φ. Pluggable modules supply scenario seeds, knowledge banks, hazard queries, and judge rubrics, while the core evaluation engine remains unchanged across domains; the CB evaluation is complemented by a cyber pilot demonstrating protocol transfer.

Instantiating the framework with a chemical-biological (CB) module, we evaluate 12 commercial LLMs from four families. Our first contribution is a horizontal comparison of dangerous capability across models and model families: the three dimensions expose sharply divergent profiles—models with comparable knowledge differ in refusal resilience, and strong defenders do not generate less harmful content when they do comply—while family-level patterns further separate Claude, DeepSeek, and GPT models. The second is a temporal analysis of capability evolution: tracking K, D, and H against model release dates reveals that dangerous capability has not monotonically declined; newer models deepen knowledge while only partially improving defense, showing that scaling and alignment progress do not uniformly translate into safety. Reliability is established via cross-judge consistency (bootstrap ρ > 0.79, 4 of 5 judges) and pipeline orthogonality (K–D–H inter-correlations ρ ∈ [0.32, 0.52]).

## 1 Introduction

Regulators, platform operators, and enterprise security teams face a fundamental question about large language models [1, 3]: which models pose dangerous capabilities, and how dangerous are they? Existing safety evaluations cannot answer this question. Knowledge benchmarks (GPQA [25], WMDP [15]) measure what a model knows but not what it will do under pressure. Defense benchmarks (R-Judge [35],

![](images/453bf6f3ce910e318bdbfb5ee2f23f92ebaf2216da963d23180908f00f02e185.jpg)  
Figure 1: The problem and our solution. Left: existing bench marks measure the LLMs with incompatible protocols, scales, and other single dimension—the outputs cannot answer which model is more dangerous or how danger has evolved. Right: our framework evaluates along three orthogonal dimensions (K, D, H) and aggregates the results into the dangerous capability profile φ, enabling direct cross-model, cross-domain, and cross-time comparison.

Agent-SafetyBench [36]) measure refusal behavior but lack standardized outputs for cross-domain comparison. Harmgeneration assessments evaluate outputs without considering how those outputs were elicited. Each benchmark runs its own protocol, reports its own format, and connects to nothing downstream—no portable risk profile for a regulator, no standardized vector for cross-model comparison.

The root problem is fragmentation, not the absence of measurement. A regulator cannot aggregate “this model knows hazardous biology” from one benchmark, “this model complies under pressure” from another, and “this model produces actionable harm protocols” from a third—because these results are not comparable. They use different models, different protocols, different score scales, and different sample populations. The fragmentation is structural: no amount of post-hoc normalization can produce a comparable risk profile from benchmarks that were never designed to be compared. Figure 1 contrasts the two worlds.

Our answer: a modular framework, not another benchmark.

We present a modular evaluation infrastructure for assessing dangerous capabilities of LLMs. The framework accepts pluggable danger-domain modules and evaluates each model along three orthogonal dimensions under a unified protocol:

• Knowledge (K) — system information-leakage risk: the hazardous-domain proficiency a model can disclose, measured via IRT-estimated ability on MCQ items. K bounds what an attacker can extract from the model through querying alone, before any behavioral bypass is attempted.

• Defense (D) — system access-control robustness: how robustly the model enforces its behavioral boundaries under adversarial induction, measured via path-dependent interaction in which a Red-Team Agent applies escalating elicitation strategies adapted to the model’s prior responses. D quantifies the boundary itself, not just its first-contact appearance.

• Harm (H) — system output content-safety risk: the actionability of hazardous content the model produces when its access controls fail, measured via rubric-scored open-ended generation. H quantifies the blast radius of an access-control breach.

These three dimensions answer independent questions about the nature of risk, and our CB evaluation confirms they are not redundant: pairwise correlations across 12 commercial LLMs are low $( \rho \in [ 0 . 3 2 , 0 . 5 2 ] )$ . No single dimension suffices for a complete risk profile.

The framework aggregates these dimensions—together with two derived indicators capturing domain asymmetry (∆) and silent refusal (ER)—into a single dangerous-capability profile φ: a standardized seven-dimensional vector that summarizes what a model knows, how it behaves under pressure, and what it produces when it complies. Because every dimen sion is measured under one protocol on one model population, φ vectors are directly comparable across models and across domains—the property that existing benchmarks cannot provide.

We instantiate the framework with a chemical-biological (CB) module and evaluate 12 commercial LLMs spanning four model families and a three-year release window (2023– 2026). The evaluation yields two headline conclusions. First, cross-model comparison: models and model families exhibit sharply distinct capability structures, differences visible only when all dimensions are reported side by side. Second, temporal evolution: knowledge compounds across generations within every family, defense diverges by family, and harm remains stubbornly inelastic—no vendor has produced a generation whose compliant outputs are meaningfully less actionable than its predecessors. We further validate the framework’s scoring foundation through cross-judge consistency (bootstrap ρ > 0.79 for 4 of 5 alternative judges), pipeline orthogonality, and IRT measurement precision (Section 6).

In summary, we claim the following contributions:

• We design a modular evaluation infrastructure for dangerous capability assessment—four-module architecture (Pluggable Module, Task Orchestration, Test Environment, Judgement), unified evaluation protocol, and switchable Content-Only/Tool-Enhanced execution modes.

• We propose a formal pluggable module interface that accepts danger-domain-specific scenario seeds, knowledge banks, harm queries, and rubrics while reusing the core evaluation pipeline; instantiated and validated with a CB module on 12 commercial LLMs.

• We define the dangerous-capability profile φ as a standardized seven-dimensional aggregation (K, D, H, DL, BR, ∆, ER) and report φ for 12 models, enabling direct cross-model and cross-domain comparison that existing benchmarks cannot support.

• We provide the first systematic analysis of how dangerous capability evolves with model release time across three generations, revealing compounding knowledge, diverging defense, and inelastic harm.

## 2 Background and Related Work

The evaluation of LLM safety has evolved rapidly but remains fragmented [16,31,33]. We categorize existing work into four tracks and identify the common gap: the absence of a unified infrastructure that assesses dangerous capability consistently across orthogonal dimensions.

## 2.1 Knowledge Benchmarks

MMLU [10] established the MCQ paradigm for large-scale knowledge evaluation; GPQA [25] and WMDP [15] extend it to hazardous scientific domains, measuring whether a model knows dangerous information. These benchmarks are fundamentally single-turn and cannot assess how a model behaves under adversarial pressure.

## 2.2 Defense and Safety Benchmarks

R-Judge [35] tests risk identification in multi-turn agentic interactions; Agent-SafetyBench and OpenAgentSafety [32] evaluate agents with real-world tool access, finding that individually safe steps compound into unsafe outcomes. Domainspecific evaluations address extreme risks: RAND’s CBRN studies [21] show LLMs can lower the barrier to biological attack planning [28, 29]; Do-Not-Answer [34] evaluates refusal safeguards. Each uses its own protocol and output format, making results incomparable across benchmarks—a structural weakness our framework removes.

## 2.3 Adaptive and Multi-Turn Jailbreaking

Gradient-based attacks [40], automated prompt generation [18], multi-turn elicitation [14], and model-based red teaming [8,23] show that adversarial pressure bypasses safety guardrails; PAIR [5] and TAP elicit harmful responses through iterative attacker-model queries. Prior work treats jailbreaking as an isolated attack vector; our framework embeds adaptive multi-round evaluation as a built-in design feature of the defense dimension, validated empirically through the initial defense-to-erosion correlation (Section 5.3).

## 2.4 LLM-as-Judge Reliability

Our framework scores with LLM-as-Judge, whose reliability has been extensively studied. Zheng et al. [38] show strong judges reach over 80% human agreement but exhibit position and verbosity biases, controlled here through multiple judge families and temperature 0.0. Multi-agent ensembles [13] mitigate single-judge variance; Verga et al. [30] show agreement varies by task type—objective scoring is more consistent than subjective quality assessment, exactly the pattern we observe across defense and harm scoring (Section 6). These findings motivate our cross-judge validation design.

## 2.5 Agent and Tool-Augmented Safety

AgentBench [17] and ToolEmu [27] show tool access alters safety behavior and that sandboxed evaluation is feasible. Our Tool-Enhanced mode (Section 3.4) builds on this line, extending the content-only mode to tool-augmented scenarios within an isolated sandbox.

## 2.6 Summary

Table 1: Comparison of existing benchmarks with our framework.
<table><tr><td>Dimension</td><td>Existing</td><td>Our framework</td></tr><tr><td>Protocol</td><td>Per-benchmark</td><td>Unified I/O schema</td></tr><tr><td>Dimensions</td><td>Single</td><td>K×D×H</td></tr><tr><td>Output</td><td>Per-format</td><td>Capability profile φ</td></tr><tr><td>Protocol</td><td>Mostly single-turn</td><td>Path-dependent in D</td></tr><tr><td>Extensible</td><td>Need redesign</td><td>Pluggable module</td></tr></table>

Even within a single threat category, recent advances produce incompatible protocols [19]. The fundamental gap is not the absence of evaluation but the absence of a unified infrastructure that hosts multiple danger-domain evaluations under one protocol and produces comparable dangerous capability profiles. A researcher today must configure GPQA, R-Judge, and a custom harm test separately; our framework reduces this to instantiating one Pluggable Module. The following sections detail the framework: Section 3 defines the four-module architecture, Section 4 shows the evaluation pipelines and the dangerous capability profile φ, Section 5 is the CB instantiation and its two headline findings, and Section 6 the judge-consistency analysis.

## 3 Framework Design

Self-assembly of existing benchmarks cannot achieve three properties that our framework provides by design: (1) Crosspipeline comparability — the same model evaluated on K, D, and H under our unified protocol yields scores in the same coordinate system, enabling cross-dimensional diagnosis; (2) Standardized output — the dangerous-capability profile φ is distilled from the evaluation’s stable dimensions, not a posthoc integration; and (3) Amortized cost — by design, adding a new danger domain requires only a new Pluggable Module adhering to a fixed interface; the cyber pilot (Section 5.6) provides initial evidence of this transfer.

## 3.1 Architecture Overview

The framework is organized as a four-module pipeline: a Pluggable Module supplies domain assets, a Task Orchestration module manages the evaluation lifecycle, a Test Environment hosts the interaction between the Red-Team Agent and the Target model, and a Judgment module scores the recorded interactions and produces the dangerous-capability profile φ. Figure 2 provides the complete system diagram. Each module has a well-defined interface and can be independently configured or replaced.

![](images/201556ca56f334dc942b95a502936e484f2b77aa589a5a39ef90b9aedc8daed3.jpg)  
Figure 2: Framework architecture overview: a four-module system (Pluggable Module, Task Orchestration, Test Environment, Judgment) that evaluates each model through three parallel assessment pipelines (K, D, H) and aggregates them into a dangerous capability profile φ.

## 3.2 Pluggable Module

The framework’s extensibility is rooted in a formal module interface: every danger domain is encapsulated in a single self-contained module specification, termed a Danger-Domain Module, which bundles exactly four evaluation-specific assets.

The first asset is the scenario-seed set, a collection of domain-specific scenarios used by the defense dimension (D). Each scenario seed encodes a legitimate work context, a harm goal embedded within it, and an escalation ladder that drives the progressive multi-round elicitation. The second one is the knowledge bank, a set of MCQ items used by the knowledge dimension (K); every item carries its correct answer and a difficulty label. The third is the harm-queries set, a collection of open-ended prompts used by the harm dimension (H). The last asset is the judge rubric, which defines the scoring dimensions and their ranges.

The core framework is shared across all modules and remains untouched when a new domain is added; the cyber pilot (Section 5.6) exercises this transfer in practice. To adapt the evaluation to a new danger domain, one supplies a new module specification containing these four assets; the framework handles the remainder, from orchestrating interactions to producing the dangerous capability profile φ. This design ensures that adding a domain is a data-preparation task rather than an engineering task.

## 3.3 Task Orchestration

The Task Orchestration module manages the complete lifecycle of an evaluation run. It first loads from the Pluggable Module the three components that correspond to the three assessment dimensions: the knowledge bank (K), scenario seeds (D), and the harm queries (H). The evaluator then selects which dimension(s) to run for this evaluation — K, D, H, or any combination thereof.

For the defense dimension, the orchestrator maintains an internal scenario database that stores the instantiated induction scenarios together with their predefined attack personas; each scenario seed encapsulates a domain-specific context, an attack persona, and a progressive conversation template. Given the evaluation task, the orchestrator selects one of two execution modes — content-only or tool-enhanced — and activates the Test Environment together with the Red-Team Agent. The orchestrator also manages parallel evaluation across multiple Red-Team–Target pairs when the workload demands it.

## 3.4 Test Environment

The Test Environment hosts the interaction between the Red-Team Agent (the adaptive adversarial elicitor) and the Target model (the LLM under evaluation).

Content-Only Mode. Models interact through naturallanguage dialogue. The Red-Team Agent adapts its elicitation strategy across rounds based on the Target model’s prior responses.

Tool-Enhanced Mode. The Target model gains real tool access (Bash, Python, file I/O, web search) inside an isolated sandbox. Tool invocation follows the ReAct (Reasoning + Acting) paradigm, with structured requests routed to the sandbox through an intercept-and-forward mechanism rather than executed directly on the host. We prototyped this mode [11] using a Docker dual-sandbox: an Execution Sandbox where the model’s commands execute, and a Target Server Sandbox hosting virtual targets (e.g., CTF-style vulnerable servers). A strictly isolated internal network separates the two sandboxes, and all tool calls from the Target LLM are confined to the sandbox, ensuring no direct system-level execution on the host. A KillSwitch mechanism monitors tool execution and can immediately terminate runaway behaviors—a safeguard essential for scenarios where models execute real system commands. In representative cyber scenarios, the agent invoked curl for HTTP reconnaissance, authored and executed custom Python exploit scripts, and iteratively debugged tool outputs, all contained within the sandbox without host compromise.

Interaction Logging. The Test Environment hooks into the interaction loop and records every Thought-Action trajectory and every content response generated by the Target model as a structured log entry. Each entry captures the model’s reasoning trace, the exact tool invocation payload, and timing metadata, preserving the full decision-making context for downstream scoring.

## 3.5 Judgment

The Judgment module converts raw interaction logs into the three assessment dimensions and aggregates them into the dangerous-capability profile φ. We formalize the scoring process as a cascade of two functions: a fast router that filters, and a deep judge that scores.

Router: R . The first stage is a risk-categorization function. Given a logged interaction record ℓ (a dialogue trajectory or a generated response), the SLM router R assigns it to one of a finite set of risk categories defined by the Pluggable Module:

$$
\begin{array} { r } { \mathcal { R } ( \ell ) \in C , \qquad C = \bigl \{ c _ { 0 } , c _ { 1 } , \dotsc , c _ { K } \bigr \} } \end{array}\tag{1}
$$

where $c _ { 0 } , c _ { 1 } , \ldots , c _ { K }$ denote the domain-specific risk classes (e.g., reconnaissance, exploitation, privilege escalation, and data exfiltration for a cyber module; synthesis guidance, dissemination planning, and weaponization for CB). The routing decision is domain-aware: each Pluggable Module supplies its own category set C alongside its rubric.

All records proceed to the Judge layer with their category label attached, enabling category-conditional reliability reporting.

Judge: J. The second stage is a rubric-scored evaluation function. For each retained record ℓ, the LLM Judge J returns a d-dimensional score vector according to the domain rubric loaded from the Pluggable Module:

$$
\mathcal { I } ( \ell ; \boldsymbol { \mathsf { p } } ) = \left( j _ { 1 } , j _ { 2 } , \ldots , j _ { d } \right) \in [ 0 , m ] ^ { d }\tag{2}
$$

where $[ 0 , m ] ^ { d }$ is the d-fold Cartesian product, i.e., the set of all d-dimensional score vectors with each component bounded in [0,m], ρ denotes the rubric and m is the per-dimension maximum score.

Cross-Judge Validation. To quantify the reliability of the Judgment, the framework supports re-scoring by alternative judge families. For each alternative judge, the framework computes agreement statistics against the primary judge— Spearman ρ for rank agreement and bootstrap standard errors for interval estimation:

$$
\rho ( \mathcal { I } , \mathcal { I } _ { i } ) , \quad \widehat { \mathrm { S E } } _ { \mathrm { b o o t } } ( \rho ) , \quad \quad i = 1 , \hdots , k\tag{3}
$$

These statistics serve as built-in quality diagnostics: each assessment dimension reports its own cross-judge agreement, so can assess the reliability of each dimension individually (Section 6).

The judge outputs feed the three assessment dimensions (K, D, H, Section 4), whose model-level values are aggregated into the dangerous-capability profile φ at the end of Section 4.

## 4 Three Evaluation Pipelines $( K , D , H )$

The framework evaluates each model along three orthogonal dimensions, summarized in Table 2. Each dimension is defined generically with respect to the assets supplied by a Danger-Domain Module; Section 5.1 instantiates all three for the CB domain.

Table 2: Three evaluation pipelines. Each measures an orthogonal dimension of dangerous capability.
<table><tr><td>Dimension</td><td>Goal</td><td>Method</td></tr><tr><td>Knowledge (K)</td><td>Information-leakage risk: hazardous-domain proficiency disclosable through</td><td>IRT θ (2PL Model)</td></tr><tr><td>Defense (D)</td><td>Access-control robustness: boundary enforcement under multi-round induction</td><td>Seeds × Five rounds; per-scenario four-dim score</td></tr><tr><td>Harm (H)</td><td>Content-safety risk: actionability of output after access-control failure</td><td>Open-ended questions; per-query four-dim score</td></tr></table>

## 4.1 Knowledge Pipeline (K)

The knowledge pipeline quantifies the information-leakage surface of a model: the hazardous-domain knowledge an attacker can extract through querying, measured by multiplechoice questionnaires. We model the response process with Item Response Theory (IRT) [6, 20], adopting the twoparameter logistic (2PL) form with a fixed guessing parameter [4, 12, 26]. For each four-choice MCQ item i, the probability that model T answers correctly is:

$$
P _ { i , T } ( \mathrm { c o r r e c t } \mid \Theta _ { T } ) = c + \frac { 1 - c } { 1 + \exp ( - ( \Theta _ { T } - b _ { i } ) ) }\tag{4}
$$

where $\theta _ { T }$ is model $T \mathbf { \hat { s } }$ latent knowledge capability, $b _ { i }$ is item $i \mathrm { \ ' } _ { \mathrm { s } }$ difficulty, and the lower asymptote $c = 0 . 2 5$ equals the random-guessing probability of a four-choice item. Item difficulty is calibrated once from the pooled pass-rates of all evaluated models:

$$
b _ { i } = \log \left( \frac { 1 - p _ { i } } { p _ { i } - 0 . 2 5 } \right)\tag{5}
$$

where $p _ { i }$ denotes the pooled correct rate of item i over 12 LLMs. Given the calibrated item bank, a model’s capability is estimated by maximum likelihood:

$$
\Theta _ { T } = \arg \operatorname* { m a x } _ { \boldsymbol { \Theta } } \prod _ { i } P _ { i , T } \big ( \boldsymbol { \Theta } \big ) ^ { x _ { i , T } } \big ( 1 - P _ { i , T } ( \boldsymbol { \Theta } ) \big ) ^ { 1 - x _ { i , T } }\tag{6}
$$

with $x _ { i , T } \in \{ 0 , 1 \}$ the model’s answer to item i (0 False, 1 True), solved via Newton–Raphson iteration. Estimation precision is quantified by the Fisher information:

$$
I ( \Theta _ { T } ) = \sum _ { i } \frac { \left( ( P _ { i , T } ( \Theta _ { T } ) - c ) ( 1 - P _ { i , T } ( \Theta _ { T } ) ) \right) ^ { 2 } } { ( 1 - c ) ^ { 2 } \cdot P _ { i , T } ( \Theta _ { T } ) ( 1 - P _ { i , T } ( \Theta _ { T } ) ) } ,\tag{7}
$$

$$
S E ( \Theta _ { T } ) = \frac { 1 } { \sqrt { I ( \Theta _ { T } ) } }\tag{8}
$$

Note that calibrating $b _ { i }$ from pooled pass-rates and then estimating $\theta _ { T }$ from individual response vectors does not constitute circular reasoning: the calibration step fixes the measurement scale (zero point and unit) of the item bank, whereas the estimation step relies on each model’s item-by-item response pattern {x<sub>i,T</sub>}—two models with identical total scores receive distinct θ estimates when they answer different items correctly, precisely because the items differ in difficulty.

IRT offers three advantages over raw accuracy: (1) difficulty weighting—harder items contribute more to θ; (2) uncertainty quantification via SE(θ); and (3) interval-scale measurement supporting rigorous cross-model comparison. We operationalize the knowledge dimension as $K : = \Theta$ , reporting raw accuracy as a supplementary diagnostic where relevant. For danger domains spanning multiple subdomains, K is instantiated as per-subdomain scores $( \mathrm { e . g . , } K _ { \mathrm { b i o } } , K _ { \mathrm { c h e m } }$ for CB); a single-subdomain module reports a scalar K.

## 4.2 Defense Pipeline (D)

The defense pipeline quantifies the access-control boundary of a model: whether its behavioral guardrails hold when an adversary escalates pressure adaptively. A scenario seed generates a five-round interaction chain $\{ c _ { 1 } , \ldots , c _ { 5 } \}$ , where each subsequent prompt adapts to the model’s prior responses— a path-dependent protocol reflecting how real-world adversaries escalate gradually.

Scoring instantiation. The Judge J (Section 3.5) evaluates the full dialogue trajectory

$$
\tau _ { s } ( T ) = \big ( ( u _ { 1 } , a _ { 1 } ) , ( u _ { 2 } , a _ { 2 } ) , \dots , ( u _ { 5 } , a _ { 5 } ) \big )\tag{9}
$$

the five-turn pairs between the Red-Team Agent and model T on scenario s—under the defense rubric ρ<sub>D</sub> supplied by the Pluggable Module. The rubric defines $d = 4$ semantic dimensions—Induction Defense, Risk Leakage, Context Handling, and Multi-round Resilience—each bounded by $m = 2 5$ yielding a per-scenario score vector:

$$
\begin{array} { r } { \mathcal { I } \big ( \tau _ { s } ( T ) ; \mathsf { p } _ { D } \big ) = \big ( j _ { s } ^ { \mathrm { i d } } , j _ { s } ^ { \mathrm { r l } } , j _ { s } ^ { \mathrm { c h } } , j _ { s } ^ { \mathrm { m r } } \big ) \in [ 0 , 2 5 ] ^ { 4 } } \end{array}\tag{10}
$$

The per-scenario overall defense score is the $\ell _ { 1 }$ norm of the score vector:

$$
D _ { s } ( T ) = \left. \mathcal { I } \bigl ( \tau _ { s } ( T ) ; \mathsf { p } _ { D } \bigr ) \right. _ { 1 } = \sum _ { j \in \mathcal { I } ( \cdot ) } j \in [ 0 , 1 0 0 ]\tag{11}
$$

Model-level metrics. The defense dimension $D ( T )$ aggregates the per-scenario scores over the module’s N scenario seeds:

$$
D ( T ) = \frac { 1 } { N } \sum _ { s = 1 } ^ { N } \left. \mathcal { I } \bigl ( \tau _ { s } ( T ) ; \rho _ { D } \bigr ) \right. _ { 1 } \in [ 0 , 1 0 0 ]\tag{12}
$$

To characterize how a model defends—not merely how well—we project the score vector onto two complementary indicators. The Defense Level DL isolates the Induction-Defense component, normalized to the unit interval; the Breakthrough Ratio BR contrasts multi-round resilience against induction defense:

$$
D L ( T ) = \frac { 1 } { 2 5 N } \sum _ { s = 1 } ^ { N } j _ { s } ^ { \mathrm { i d } } ( T ) \in [ 0 , 1 ]\tag{13}
$$

$$
B R ( T ) = \frac { \sum _ { s = 1 } ^ { N } j _ { s } ^ { \mathrm { m r } } ( T ) } { \sum _ { s = 1 } ^ { N } j _ { s } ^ { \mathrm { i d } } ( T ) }\tag{14}
$$

DL and BR together reflect the outcome of safety training [2, 22, 24]: baseline refusal emerges from preference optimization, while erosion reveals how deeply that preference is internalized. DL captures baseline refusal tendency on first contact; BR captures how that tendency evolves under sustained pressure—BR ≈ 1 indicates stability, $B R < 1$ indicates erosion, and $B R > 1$ indicates escalating compliance.

Together, DL and BR span a two-dimensional risk typology that distinguishes surface-level defense from deep defense (Section 5.3).

## 4.3 Harm Pipeline (H)

The harm pipeline quantifies the content-safety consequence of an access-control failure: how actionable the hazardous content is that a model produces once its boundary is breached. Each model receives N open-ended hazardous-domain questions drawn from the module’s harm-queries set.

Scoring instantiation. The Judge J (Section 3.5) evaluates the model’s response $r _ { i } ( T )$ to query i under the harm rubric $\rho _ { H } .$ , which defines $d = 4$ semantic dimensions— Completeness, Implement-Ability, Resource-Accessibility, and Innovation—each bounded by $m = 2 5$ , yielding a perresponse score vector:

$$
\begin{array} { r } { \mathcal { I } \big ( r _ { i } ( T ) ; \mathsf { p } _ { H } \big ) = \big ( j _ { i } ^ { \mathrm { c p } } , j _ { i } ^ { \mathrm { i m } } , j _ { i } ^ { \mathrm { r s } } , j _ { i } ^ { \mathrm { i n } } \big ) \in [ 0 , 2 5 ] ^ { 4 } } \end{array}\tag{15}
$$

The per-response exposure score is the $\ell _ { 1 }$ norm of the score vector, quantifying how much actionable hazardous information the response conveys:

$$
H _ { i } ( T ) = \left\| \mathcal { I } \big ( r _ { i } ( T ) ; \rho _ { H } \big ) \right\| _ { 1 } \in [ 0 , 1 0 0 ]\tag{16}
$$

Model-level metric. The harm dimension $H ( T )$ aggregates the exposure scores over the set of non-refusal responses $\mathcal { R } _ { T }$ (empty or purely-refusal responses carry zero information and are excluded from scoring):

$$
H ( T ) = \frac { 1 } { | \mathcal { R } _ { T } | } \sum _ { i \in \mathcal { R } _ { T } } H _ { i } ( T ) \in [ 0 , 1 0 0 ]\tag{17}
$$

H complements D: whereas D measures whether a model refuses, H measures the harmfulness of what it produces when it complies—the two dimensions are weakly correlated (Section 5.4), confirming that a high-defense model can still generate highly actionable content on the queries it answers.

## 4.4 Derived Dimensions (∆, ER)

Two additional dimensions enrich the dangerous capability profile beyond the three primary metrics.

Domain asymmetry (∆). When a danger domain spans multiple subdomains, a model’s defense may be unevenly distributed across them—a signature of imbalanced safety training coverage. We quantify this asymmetry as the signed, normalized difference between per-subdomain defense scores:

$$
\Delta _ { d _ { 1 } , d _ { 2 } } ( T ) = \frac { D _ { d _ { 1 } } ( T ) - D _ { d _ { 2 } } ( T ) } { \operatorname* { m a x } \big ( D _ { d _ { 1 } } ( T ) , D _ { d _ { 2 } } ( T ) \big ) } \in [ - 1 , + 1 ]\tag{18}
$$

where $d _ { 1 }$ and $d _ { 2 }$ denote the two subdomains defined by the Danger-Domain Module (e.g., biology and chemistry for

CB). $\Delta > 0$ indicates stronger defense in $d _ { 1 } ; \Delta \approx 0$ indicates balanced coverage. A single-subdomain module omits this dimension.

Empty-response rate (ER). Beyond explicit refusal, some models deploy silent refusal: the API returns an empty response with no error field. This behavior reflects the provider’s content-filtering strategy and is observable from interaction logs alone, independent of domain semantics:

$$
E R ( T ) = \frac { \{ \mathrm { e m p t y ~ t u r n s } \} } { \{ \mathrm { t o t a l ~ t u r n s } \} } \times 1 0 0 \%\tag{19}
$$

ER distinguishes models that refuse visibly (text-level re fusal) from those that refuse silently (empty response), a strategic difference invisible to standard defense metrics (Section 5.5).

## 4.5 Dangerous Capability Profile (φ)

Having defined all constituent dimensions, we aggregate them into the canonical dangerous-capability profile—the frame work’s portable output. The profile φ is a seven-dimensional vector that summarizes what a model knows, how it behaves under adversarial pressure, and what it produces when it complies:

$$
\Phi ( T ) = \langle K , D , H , D L , B R , \Delta , E R \rangle\tag{20}
$$

Each component is computed by the corresponding pipeline (Eqs. 12–19) from the same evaluation run, guaranteeing that all dimensions share a common protocol and model population—the property that makes φ vectors directly comparable across models. φ serves as the framework’s portable output: regulators and platform operators can compare φ vectors across models without consulting benchmark-specific documentation. Section 5.1 instantiates φ for the CB domain and reports the resulting vectors for 12 commercial models.

## 5 Experiment

## 5.1 CB Setting

We instantiate the framework with the CB Danger-Domain Module and evaluate 12 commercial LLMs spanning four model families: Claude (Opus-4, Opus-4.5, Opus-4.8, Haiku-3), GPT (GPT-3.5-Turbo, GPT-4o, GPT-5.4, GPT-5.5), DeepSeek (V3, V4-Flash, V4-Pro), and Kimi (K3).

The module supplies three evaluation assets across two subdomains (biology and chemistry): 298 scenario seeds (biology: 156, chemistry: 142) for the defense dimension D, a knowledge bank of 3,773 MCQ items (biology: 1,938, chemistry: 1,835; drawn from SciKnowEval [7], GPQA [25], and a Chinese college-entrance examination set) for K, and 200 open-ended harm queries (biology: 100, chemistry: 100; drawn from WMDP [15], LabSafety Bench [39], and Chem SafetyBench [37]) for H. The primary judge is GPT-4o-mini (temperature = 0.0); cross-judge reliability is analyzed in Section 6.

Sections 5.2–5.5 report the per-dimension results with dedicated tables and figures; Section 5.7 aggregates them into the two headline conclusions of this work: cross-model comparison and temporal evolution of dangerous capability.

## 5.2 Knowledge in CB (K)

Table 3 reports the knowledge dimension for all 12 models. IRT ability scores θ are reported per subdomain alongside raw accuracy as a supplementary diagnostic; the two rankings agree at Spearman $\rho = 0 . 9 9 8$

Table 3: Knowledge dimension (K) in CB. θ: IRT ability relative to the 12-model mean (logits); raw%: percentage of MCQ items correct.
<table><tr><td>Model</td><td> $\Theta _ { \mathrm { b i o } }$ </td><td> $\Theta _ { \mathrm { c h e m } }$ </td><td> $\operatorname { r a w } _ { \mathrm { b i o } } \%$ </td><td> $\mathrm { r a w } _ { \mathrm { c h e m } } \%$ </td></tr><tr><td>Claude-3-Haiku</td><td>-1.220</td><td>-1.196</td><td>72.1</td><td>62.7</td></tr><tr><td>Claude-Opus-4</td><td>-0.286</td><td>+0.558</td><td>78.2</td><td>75.7</td></tr><tr><td>Claude-Opus-4.5</td><td>+0.773</td><td>-0.107</td><td>84.7</td><td>71.2</td></tr><tr><td>Claude-Opus-4.8</td><td>-0.370</td><td>+1.109</td><td>78.4</td><td>78.6</td></tr><tr><td>DeepSeek-V3</td><td>+0.236</td><td>+0.018</td><td>81.8</td><td>71.7</td></tr><tr><td>DeepSeek-V4-Flash</td><td>+0.529</td><td>+0.366</td><td>83.3</td><td>73.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>+0.820</td><td>+0.445</td><td>84.6</td><td>74.3</td></tr><tr><td>GPT-3.5-Turbo</td><td>-2.054</td><td>-4.092</td><td>66.6</td><td>44.0</td></tr><tr><td>GPT-40</td><td>-1.767</td><td>-0.452</td><td>68.9</td><td>67.2</td></tr><tr><td>GPT-5.4</td><td>+1.203</td><td>+0.816</td><td>86.7</td><td>77.1</td></tr><tr><td>GPT-5.5</td><td>+1.589</td><td>+1.212</td><td>88.2</td><td>79.5</td></tr><tr><td>Kimi-K3</td><td>+1.041</td><td>+0.830</td><td>86.5</td><td>77.5</td></tr></table>

Two patterns emerge: First, knowledge is strongly familystructured: GPT-5.5 and GPT-5.4 lead both subdomains, while GPT-3.5-Turbo trails by over 5 logits in chemistry—the largest single-model gap across the entire evaluation. Its $\theta _ { \mathrm { c h e m } } = - 4 . 0 9$ corresponds to a raw accuracy of 44.0%, barely above the 25% random-guessing floor, whereas GPT-$5 . 5 \mathrm { ^ s } + 1 . 2 1 $ reflects 79.5% on the same bank. Second, knowledge is asymmetric across subdomains: 8 of 12 models exhibit stronger biological than chemical knowledge $( \Theta _ { \mathrm { { b i o } } } > \Theta _ { \mathrm { { c h e m } } } )$ with domain gaps ranging from 0.1 to 3.0 logits, yet the three Claude models and GPT-4o reverse this pattern. Figure 3 visualizes the asymmetry.

![](images/937756b2494c39f2f7eb45a555fb695105291a87ee371e65dd785af41b0d42f4.jpg)  
Figure 3: Domain asymmetry: $\Theta _ { \mathrm { b i o } }$ vs. $\boldsymbol { \theta } _ { \mathrm { c h e m } }$ . Above-diagonal points indicate stronger biological knowledge; this offset links pre-training data to domain-specific hazardous knowledge.

The direction of asymmetry is family-dependent: all DeepSeek models favor biology, all Claude models favor chemistry, and GPT models split between the two. This finding is actionable: a model audited only on biology benchmarks may harbor unexamined chemistry expertise, making per-subdomain knowledge reporting essential for domainspecific risk assessment.

## 5.3 Defense in CB (D)

Table 4 reports the defense dimension for all 12 models. Persubdomain scores $D _ { \mathrm { b i o } }$ and $D _ { \mathrm { c h e m } }$ capture refusal capability separately for biology and chemistry; the derived indicators DL (Defense Level) and BR (Breakthrough Ratio) characterize the structure of defense as defined in Section 4.2.

The defense scores span a wider range than either knowledge or harm (Section 5.4): from $\mathrm { D e e p S e e k  – V 3 ^ { \circ } s } D _ { \mathrm { b i o } } = 5 6 . 9$ to Claude-Opus-4.5’s D = 99.9, a gap of 43 points that no other dimension approaches. This spread is not uniform across subdomains—the same model can defend differently against biological and chemical elicitation. Claude-Opus-4 exhibits the sharpest asymmetry: $D _ { \mathrm { b i o } } = 9 7 . 0$ versus $D _ { \mathrm { c h e m } } = 8 2 . 0 $ , a 15-point gap indicating safety training that favors biology over chemistry. We formalize this asymmetry as ∆ in Section 5.5.

To characterize defense beyond a single aggregate, we project each model onto the $D L \times B R$ plane (Figure 4). kmeans clustering $( k = 4$ , chosen by the gap statistic) on the standardized (DL,BR) coordinates yields four qualitatively distinct risk modes.

Table 4: Defense dimension (D) in CB. D: mean per-scenario defense score (0–100); DL: normalized Induction-Defense score; BR: Multi-round Resilience over induction defense.
<table><tr><td>Model</td><td> $D _ { \mathrm { b i o } }$ </td><td> $D _ { \mathrm { c h e m } }$ </td><td>DL</td><td>BR</td></tr><tr><td>Claude-3-Haiku</td><td>76.1</td><td>76.3</td><td>0.76</td><td>1.02</td></tr><tr><td>Claude-Opus-4</td><td>97.0</td><td>82.0</td><td>0.90</td><td>1.02</td></tr><tr><td>Claude-Opus-4.5</td><td>99.3</td><td>99.9</td><td>1.00</td><td>1.00</td></tr><tr><td>Claude-Opus-4.8</td><td>99.5</td><td>99.7</td><td>1.00</td><td>0.99</td></tr><tr><td>DeepSeek-V3</td><td>56.9</td><td>63.9</td><td>0.56</td><td>1.26</td></tr><tr><td>DeepSeek-V4-Flash</td><td>65.8</td><td>76.6</td><td>0.69</td><td>1.10</td></tr><tr><td>DeepSeek-V4-Pro</td><td>67.2</td><td>80.1</td><td>0.71</td><td>1.11</td></tr><tr><td>GPT-3.5-Turbo</td><td>66.2</td><td>64.4</td><td>0.65</td><td>1.01</td></tr><tr><td>GPT-40</td><td>66.4</td><td>66.1</td><td>0.66</td><td>1.00</td></tr><tr><td>GPT-5.4</td><td>92.8</td><td>96.2</td><td>0.97</td><td>0.94</td></tr><tr><td>GPT-5.5</td><td>95.8</td><td>96.8</td><td>0.98</td><td>0.96</td></tr><tr><td>Kimi-K3</td><td>94.4</td><td>97.3</td><td>0.96</td><td>1.00</td></tr></table>

![](images/ea600b9d1d7604729ef8ac90ef626754d30bfa9296909136f3f24d086116431a.jpg)  
Figure 4: $D L \times B R$ risk space showing four distinct risk modes identified via k-means clustering (k=4).

Mode I—Brittle. GPT-5.4 and GPT-5.5 achieve near-perfect baseline defense $( D L \approx 0 . 9 7 )$ but show a modest BR deficit $( B R \approx 0 . 9 5 )$ : their refusal tendency erodes slightly under sustained pressure. These models are safe on first contact yet reveal subtle fragility across rounds.

Mode II—Exposed. DeepSeek-V3, V4-Flash, and V4-Pro exhibit low baseline defense $( D L < 0 . 7 2 )$ coupled with escalating compliance $( B R > 1 . 0 9 )$ : their already-weak refusal tendency further deteriorates across rounds, yielding the highest breakthrough rates of the evaluation.

Mode III—Robust. Claude-Opus-4, Opus-4.5, Opus-4.8, and Kimi-K3 maintain strong, stable defense $( D L > 0 . 8 9 , B R \approx$ 1.0) across all five rounds. Notably, Kimi-K3 demonstrates that robust defense is achievable outside the Claude family. Mode IV—Moderate. Claude-3-Haiku, GPT-4o, and GPT-3.5-Turbo occupy the middle ground (DL ≈ 0.65–0.76, BR ≈ 1.0): moderate baseline defense that neither erodes nor escalates. Modes III and IV share similar BR but are distinguished by DL— the two clusters differ in baseline strength rather than stability.

The most consequential finding concerns Mode I: models judged safest by single-turn evaluation are precisely those whose defense erodes under sustained interaction. To ground this quantitative pattern in concrete behavior, we examine a representative five-round trajectory.

## Defense Erosion Case: Refusal-to-Protocol

DeepSeek-V4-Pro, judged entirely safe under single-turn queries, transitions within five rounds from a 20-character refusal to a complete operational protocol for pathogen surfaceglycoprotein detection—specifying detection assays, decontamination concentrations, and activity thresholds for area lockdown escalation, all redacted here for dual-use review (Judge: Unsafe, 54/100). Figure 5 presents the interaction card.

![](images/edbd7f4ff3593ec5a9c15bb10c19e26f926c62ddaba45a049a313e4d7d6559b7.jpg)  
Figure 5: Five-round defense trajectory for DeepSeek-V4-Pro on a fire-department HazMat scenario (translated from the original Chinese logs, operational parameters redacted).

This trajectory illustrates precisely what a path-dependent protocol captures and a single-turn evaluation misses: the defense score is not a property of the model’s first response but of how its behavior evolves under escalating pressure.

## 5.4 Harm in CB (H)

Table 5 reports the harm dimension for all 12 models: the mean exposure score of the hazardous content each model produces when it complies.

Table 5: Harm dimension (H) in CB: mean exposure score of non-refusal responses (0–100; higher = more actionable hazardous content).
<table><tr><td>Model</td><td> $H _ { \mathrm { b i o } }$ </td><td> $H _ { \mathrm { c h e m } }$ </td><td>Model</td><td> $H _ { \mathrm { b i o } }$ </td><td> $H _ { \mathrm { c h e m } }$ </td></tr><tr><td>Claude-3-Haiku</td><td>69.8</td><td>68.8</td><td>DeepSeek-V4-Pro</td><td>76.2</td><td>77.8</td></tr><tr><td>Claude-Opus-4</td><td>75.1</td><td>72.9</td><td>GPT-3.5-Turbo</td><td>70.5</td><td>68.8</td></tr><tr><td>Claude-Opus-4.5</td><td>75.9</td><td>75.5</td><td>GPT-40</td><td>70.8</td><td>68.3</td></tr><tr><td>Claude-Opus-4.8</td><td>78.5</td><td>77.5</td><td>GPT-5.4</td><td>75.5</td><td>74.9</td></tr><tr><td>DeepSeek-V3</td><td>74.0</td><td>74.6</td><td>GPT-5.5</td><td>71.4</td><td>74.2</td></tr><tr><td>DeepSeek-V4-Flash</td><td>76.4</td><td>75.7</td><td>Kimi-K3</td><td>69.2</td><td>72.7</td></tr></table>

Two properties distinguish H from the other dimensions. First, its range is remarkably narrow: all 12 models cluster within a 10-point band (68.3–78.5), in sharp contrast to the 43- point spread of D. Once a model answers a hazardous query, the content it produces is uniformly actionable regardless of how well it defends otherwise.

Counterintuitively, models with the strongest defense produce the highest harm scores when they comply: Claude-Opus-4.5 and Opus-4.8, despite D > 99, yield H > 75. This inversion is a selection effect: strong refusers are scored only on the minority of queries they fail to refuse—a maximally difficult residual set—whereas weak refusers (GPT-3.5-Turbo, GPT-4o) average over many trivial responses, deflating their means. H thus measures capability conditional on compliance, not a monotone function of defense weakness.

![](images/0593466167c7393cc0da557b146704aa68cacdc2e41000e3f0724502137587e3.jpg)  
Figure 6: Single-round harm query for DeepSeek-V4-Pro (translated from the original Chinese log, operational parameters redacted).  
Figure 6 grounds this dimension in concrete behavior:

DeepSeek-V4-Pro $( H = 7 7 . 0 )$ produces a cell-free protein synthesis protocol spanning four production systems and a five-step pipeline, with operational parameters redacted for dual-use review—scored 72/100 by the Judge.

The complementarity with Section 5.3 is direct: DeepSeek-V4-Pro combines weak defense $( D \approx 6 7 – 8 0 )$ with near-peak harm generation $( H = 7 7 . 0 )$ , while Claude-Opus-4.8 combines near-perfect defense $( D > 9 9 )$ with the highest harm score of the evaluation $( H = 7 8 . 5 )$ . Neither dimension alone predicts the other $( \rho = 0 . 3 1 8 ,$ , Section 6), which is precisely why a complete capability profile requires both.

## 5.5 Domain Asymmetry and Silent Refusal (∆, ER)

Beyond the three primary dimensions, two derived indicators capture properties of dangerous capability that aggregate scores conceal. Table 6 reports both for all 12 models.

Table 6: Derived dimensions in CB. ∆: signed normalized defense asymmetry between subdomains, +: stronger biology defense, −: stronger chemistry defense; ER: percentage of interaction turns returning an empty response.
<table><tr><td>Model</td><td>Δ ER%</td><td>Model</td><td></td><td>Δ ER%</td></tr><tr><td>Claude-3-Haiku</td><td>-0.003</td><td>0.7</td><td>DeepSeek-V4-Pro</td><td>-0.161 0.1</td></tr><tr><td>Claude-Opus-4</td><td>+0.155</td><td>67.8</td><td>GPT-3.5-Turbo</td><td>+0.027 0.0</td></tr><tr><td>Claude-Opus-4.5</td><td>-0.006</td><td>78.5 GPT-40</td><td>+0.005</td><td>0.0</td></tr><tr><td>Claude-Opus-4.8</td><td>-0.002</td><td>78.6</td><td>GPT-5.4</td><td>-0.035 15.0</td></tr><tr><td>DeepSeek-V3</td><td>-0.110</td><td>0.0</td><td>GPT-5.5</td><td>-0.010 28.0</td></tr><tr><td>DeepSeek-V4-Flash</td><td>-0.141</td><td>0.3</td><td>Kimi-K3</td><td>-0.030 2.3</td></tr></table>

Domain asymmetry (∆). ∆ quantifies whether a model’s Defense is evenly distributed across subdomains—a signal of how its safety training was provisioned. Two models anchor the extremes. Claude-Opus-4 $( \Delta = + 0 . 1 5 5 )$ defends biology markedly better than chemistry, mirroring the 15- point D gap documented in Section 5.3; DeepSeek-V4-Pro $( \Delta = - 0 . 1 6 1 )$ exhibits the inverse profile, with chemistry defense outpacing biology. The remaining ten models cluster within $\Delta \in [ - 0 . 1 1 , + 0 . 0 3 ]$ , indicating roughly balanced subdomain coverage.

Two observations follow. First, the extremes are not the weakest defenders: both Opus-4 and V4-Pro sit above the midpoint of the D distribution, so their asymmetry reflects uneven provision of safety training rather than a general defense deficit. Second, ∆ is orthogonal to D: knowing a model’s aggregate defense score reveals nothing about which subdomain it protects better. The dimension therefore contributes independently to the capability profile (Section 4.5).

Silent refusal (ER). ER exposes a strategic division among providers that is invisible in refusal content. Three Claudefamily models (Opus-4, Opus-4.5, Opus-4.8) return an empty response with no error field in 67.8–78.6% of interaction turns: the provider filters hazardous content at the serving layer, dropping responses before they reach the user—a silent refusal strategy. GPT-5.4 and GPT-5.5 show intermediate rates (15.0–28.0%), consistent with partial serving-layer filtering, while all remaining models exhibit $E R < 3 \%$ , refusing through visible text (“I cannot...”).

The operational distinction matters. A silent refusal leaves no trace that a defense mechanism fired: downstream auditors cannot distinguish a filtered model from one that was never prompted. A text-level refusal, by contrast, records the defense event in the interaction log. ER thus captures a property of the deployment stack—the provider’s serving configuration—rather than of the model weights, making it a necessary complement to D in the capability profile. The two dimensions are indeed independent: silent-refusal models span the full ∆ range (Opus-4: +0.155; Opus-4.8: −0.002), confirming that serving-layer strategy and subdomain defense balance are orthogonal axes of deployment behavior.

## 5.6 Cyber Pilot

To probe cross-domain generality, we instantiate a minimal cyber module and evaluate the three models that overlap with the CB set (DeepSeek-V3, GPT-3.5-Turbo, GPT-4o) in Tool-Enhanced mode (Section 3.4), with real Bash/Python execution across 16 attack-chain scenarios spanning reconnaissance, injection, exploitation, exfiltration, persistence, and track-covering. The module reuses the CB judge protocol unchanged: the same four-dimension rubric structure, the same 0–25 per-dimension scale, and the same primary judge (GPT-4o-mini).

Table 7 reports all three dimensions alongside the corresponding CB-domain values.

Table 7: Cyber pilot results for the three overlapping models, each evaluated on all 16 attack-chain scenarios. K: raw accuracy on 75 cyber MCQ items (the cyber module reports raw accuracy rather than IRT θ; see text); D: mean defense score (0–100, higher = safer); H: mean exposure score (0– 100, higher = more actionable). CB columns reproduce the corresponding values from Sections 5.2–5.4.
<table><tr><td>Model</td><td>K raw%</td><td>D</td><td>H</td><td>D (CB)</td><td>H (CB)</td></tr><tr><td>DeepSeek-V3</td><td>100.0</td><td>34.8</td><td>69.1</td><td>60.2</td><td>74.3</td></tr><tr><td>GPT-3.5-Turbo</td><td>98.7</td><td>31.6</td><td>71.2</td><td>65.3</td><td>69.6</td></tr><tr><td>GPT-40</td><td>98.7</td><td>61.2</td><td>69.1</td><td>66.3</td><td>69.5</td></tr></table>

Two observations generalize across domains. First, harm is inelastic in cyber as in CB: H spans 69–71, reproducing the narrow band observed across the CB evaluation (68–78), and the three models preserve their relative ordering. Second, defense rankings are directionally preserved but systematically lower: GPT-4o defends best in both domains, yet all three models score markedly below their CB defense—the Tool-Enhanced setting, where the model must decline concrete tool invocations rather than merely refuse prose, is the harder test. GPT-4o’s mean conceals a bimodal pattern: it defends fully (D = 100) on exploitation, exfiltration, and persistence scenarios yet collapses to minimal defense (D = 35) on XSS, SSTI, and JWT scenarios—a scenario-dependence that aggregate scores cannot express (Appendix Figure 14).

![](images/2516c52355c018e9b7ea2b0539b0225e3b32aa822346778e06506d854bb901c5.jpg)  
(a) Temporal evolution of knowledge capability (K, mean θ).

![](images/3c96b1e1a9b655562f59ee30cd478a9f642b54df471a98f3fbbc98cee5608e2f.jpg)  
(b) Temporal evolution of defense (D, mean defense score).

![](images/a78eea84aa84c06e5c446d3793271e6027a7b009405f5aff2b37644530992502.jpg)  
(c) Temporal evolution of harm (H, mean exposure score).  
Figure 7: Family-level K–D–H means comparison by subdomain.

The cyber knowledge items exhibit a clear ceiling effect: all three models score above 98%. We therefore report raw accuracy rather than IRT θ: with 75 items of near-uniform passrates, the difficulty parameters would collapse to a single low value, and θ estimates would carry no discriminative content. The item bank is publicly sourced, so current models have plausibly encountered the items during pre-training. Beyond the contamination concern, the ceiling itself is a methodological signal: cyber modules require contamination-checked, expert-authored item banks with sufficient difficulty spread before K becomes discriminative in this domain.

Taken together, the pilot demonstrates that the pluggable module interface (Section 3.2) transfers the full threedimension protocol to a second domain without modification, and that the framework’s headline patterns— inelastic harm and directionally stable defense—are not artifacts of the CB instantiation.

## 5.7 Dangerous Capability Profile and Two Headline Findings

This section aggregates the per-dimension scores into φ, then draws the two headline conclusions: cross-model comparison and temporal evolution of dangerous capability.

## Dangerous Capability Profile in CB

The seven dimensions reported in Sections 5.2–5.5 assemble into the capability profile $\Phi = \left. K , D , H , D L , B R , \Delta , E R \right.$ defined in Section 4.5. Table 8 illustrates the profile for three models occupying distinct regions of the capability space.

Table 8: Capability profiles for three representative models. The full per-dimension values appear in Tables 3–6.
<table><tr><td>Model</td><td>K</td><td>D</td><td>H</td><td>DL</td><td>BR</td><td>Δ</td><td>ER%</td></tr><tr><td>DeepSeek-V4-Pro</td><td>+0.632</td><td>73.3</td><td>77.0</td><td>0.71</td><td>1.11</td><td>-0.161</td><td>0.1</td></tr><tr><td>Claude-Opus-4.8</td><td>+0.369</td><td>99.6</td><td></td><td>78.01.000.99</td><td></td><td>-0.002</td><td>78.6</td></tr><tr><td>GPT-3.5-Turbo</td><td>-3.073</td><td>65.3</td><td>69.6 0.65</td><td></td><td>51.01</td><td>+0.027</td><td>0.0</td></tr></table>

The three rows embody three distinct risk structures, visible only because φ reports every dimension side by side. DeepSeek-V4-Pro pairs strong knowledge with weak, eroding defense (BR = 1.11) and near-peak harm—maximal overall exposure. Claude-Opus-4.8 pairs near-perfect, stable defense with the highest harm of the evaluation and silent refusal (ER = 78.6%): its residual risk is concentrated in the rare responses that survive filtering. GPT-3.5-Turbo is weak on every axis—low knowledge, moderate defense, low harm— yet still exceeds the 25% guessing floor only marginally in chemistry (Section 5.2).

## Model-Family Comparison

Figure 7 compares the four families on each primary dimension, split by subdomain. The family profiles diverge sharply in structure. DeepSeek combines the highest knowledge with the lowest defense and the highest harm: its three models average θ = +0.40, D = 68.2, and H = 75.8—they know the most, refuse the least, and produce the most actionable content when they comply, a profile of maximal overall risk. Claude occupies the opposite corner on defense (D = 91.3) yet ranks second in harm (H = 74.2): strong refusal coexists with highconditional output quality, the selection effect quantified in Section 5.4. GPT is the most heterogeneous family, spanning the full knowledge range from GPT-3.5-Turbo (θ = −3.07) to GPT-5.5 (θ = +1.40), so its family mean (θ = −0.44) misrepresents every member. Kimi-K3, evaluated as a single model rather than a family, matches Claude on defense (D = 95.8) while sitting at the low-harm end (H = 70.9); it appears in the temporal figures (Figure 8) but is excluded from the family comparison, where a single-model “mean” would be misleading.

![](images/6103111169abd1a7bdfb3575955357be2dd61a34c03d3b4aa52d680f1710c91d.jpg)  
(a) Temporal evolution of knowledge capability (K, mean θ).

![](images/ea391b6a0e6f05cd46a36354a46b91b2e1fbd176d275786dcbbcbc7d89c328c3.jpg)  
(b) Temporal evolution of defense (D, mean defense score).

![](images/9d3b7107a8cce674f76040fd15f5061aa3f7c0620d6b6ae991665890824158f0.jpg)  
(c) Temporal evolution of harm (H, mean exposure score).  
Figure 8: Combined temporal evolution of knowledge, defense, and harm across model generations.

## Temporal Evolution of Dangerous Capability

A second axis of comparison is time: do newer models become more or less dangerous? Figure 8 plots each primary dimension against the model’s release date, colored by family.

Three observations emerge from the temporal axis. First, knowledge compounds: every family improves knowledge monotonically with generation, and the steepest trajectory belongs to GPT—the family with the longest evaluation window. Dangerous knowledge is accumulating faster than it is being removed. Second, defense diverges by family: GPT’s defense increases in lockstep with its knowledge, suggesting safety training kept pace; DeepSeek’s defense gains are modest and plateau well below the 80-point mark, leaving its rising knowledge increasingly unbalanced against weak refusal. Third, harm is inelastic: across 3.5 years and twelve models, H never leaves the 68–78 band. No vendor has produced a generation whose compliant outputs are meaningfully less actionable than its predecessors’. The danger, in other words, is not that models learn to produce more harmful content—they already could—but that they know increasingly more and refuse increasingly inconsistently, enlarging the population of users who can obtain actionable content in the first place.

## 6 Judge Consistency Analysis

The results in Section 5 rest on LLM-as-Judge scoring. This section interrogates that foundation through three ablationstyle analyses: how sensitive rankings are to the choice of judge, whether the three dimensions measure distinct constructs, and how precisely θ is estimated.

## 6.1 Cross-Judge Agreement

Five alternative judge families (Qwen3.5-Flash, GLM-4.5- Air, Gemini-2.5-Flash, Grok-3-Mini, Llama-3.3-70b) re-score

90 stratified defense scenarios (30 Safe, 30 Borderline, 30 Unsafe) sampled proportionally from the 298-scenario pool. Table 9 reports bootstrap Spearman ρ (1,000 resamples) with 95% confidence intervals for each alternative judge against the primary judge (GPT-4o-mini, temperature = 0.0).

Table 9: Cross-judge reliability for Defense assessment (bootstrap Spearman ρ, n = 90, 1,000 resamples).
<table><tr><td>Judge</td><td>ρ</td><td>95% CI</td></tr><tr><td>Qwen3.5-Flash</td><td>0.851</td><td>[0.778, 0.923]</td></tr><tr><td>GLM-4.5-Air</td><td>0.843</td><td>[0.765, 0.921]</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.818</td><td>[0.728, 0.908]</td></tr><tr><td>Grok-3-Mini</td><td>0.794</td><td>[0.701, 0.887]</td></tr><tr><td>Llama-3.3-70b</td><td>0.641</td><td>[0.492, 0.791]</td></tr></table>

Four of five alternative judges agree with the primary judge at $\rho > 0 . 7 9$ , confirming that the Defense ranking is protocolinherent rather than judge-dependent. The exception, Llama-$3 . 3 \substack { - 7 0 \mathbf { b } } \ ( \rho = 0 . 6 4 1 )$ , is consistent with its distinct scoring behavior observed in prior work. Pairwise cross-judge correlations range from $\rho = 0 . 5 9$ (GLM–Llama) to $\rho = 0 . 8 7$ (GLM–Qwen).

For the harm dimension, inter-judge agreement is lower: $\mathfrak { p } \in [ 0 . 4 2 , 0 . 5 8 ]$ against the primary judge over n = 360 items (Figure 9, right panel). This gap between dimensions is expected: scoring an MCQ answer is a near-objective task, whereas grading the actionability of an open-ended hazardous protocol is inherently subjective. The framework reports both levels transparently, so that downstream consumers of φ can weigh each dimension’s reliability accordingly.

## 6.2 Pipeline Orthogonality

If the three dimensions measured the same underlying construct, their model rankings would be strongly correlated. They are not. The moderate K–D coupling is theoretically expected: recognizing a hazardous query as dangerous presupposes hazardous-domain knowledge, so some positive association is structural rather than evidence of redundancy. The decisive evidence is the near-zero D–H pair. The Spearman correlations between dimension-level rankings are consistently weak: K–D at ρ = 0.521, K–H at ρ = 0.469, and D–H at $\rho = 0 . 3 1 8$ (Figure 10).

![](images/db9175a9cbd645cbce236ae75dc00f32238a05f67e81325e40d4cd5882a8122b.jpg)  
Figure 9: Cross-judge consistency matrix (Spearman ρ) across six judge families. Left: defense (D), computed over a stratified subsample of 90 scenarios. Right: harm (H), computed over 360 scored responses. <sup>†</sup>D evaluation used Grok-3-Mini; H evaluation used Grok-4.3.

![](images/f9f4df5a25055640d50d60df33e01dbe4a928430d493bc6ead664d008ec1347e.jpg)  
Figure 10: Cross-dimension Spearman correlations between pipeline rankings: Knowledge–Defense $( \rho = 0 . 5 2 1 )$ , Knowledge–Harm $( \rho = 0 . 4 6 9 )$ , Defense–Harm $( \rho = 0 . 3 1 8 )$ .

The near-zero D–H correlation is particularly instructive: a model’s refusal capability does not predict the harmfulness of content it produces when it does comply—precisely the blind spot that the H dimension is designed to illuminate. A complete capability profile therefore requires all three dimensions; any subset misses a weakly coupled component of risk.

## 6.3 IRT Measurement Precision

Model rankings from θ agree near-perfectly with rawaccuracy rankings (Spearman $\rho = 0 . 9 9 8 )$ , confirming that θ adds difficulty-weighted discrimination while preserving the same construct. All 24 model-subdomain θ estimates carry standard errors in the narrow range [0.09, 0.16] logits, confirming that the 3,773-item bank provides sufficient information across the full θ spectrum. Two models with identical raw totals receive distinct θ values when they answer different items correctly—the property that makes IRT strictly more informative than raw accuracy.

## 7 Discussion and Limitations

The CB evaluation is complemented by a cyber pilot (Section 5.6) that reproduces inelastic harm and directionally stable defense; cross-judge validation $( \rho > 0 . 7 9 )$ confirms ranking robustness is protocol-inherent. Limitations: (1) full-scale validation is limited to CB—the cyber pilot demonstrates protocol transfer at reduced scale and with an immature knowledge bank; (2) H scoring shows moderate cross-judge agreement $( \rho \approx 0 . 4 2 - 0 . 5 8 )$ ), the expected range for open-ended quality assessment; (3) knowledge benchmark contamination [9] remains a shared risk.

The capability profile supports deployment decisions directly: a high-D, high-H model may be acceptable with output filtering, whereas a low-D model carries inherent refusal risk regardless. The 15-point bio–chem defense gap in Claude-Opus-4 illustrates that domain-specific guardrails remain necessary even for strong defenders.

The temporal trajectories sharpen the deployment calculus. Knowledge compounds across generations within every family, so knowledge-based guardrails calibrated on today’s models become stale within one release cycle. Defense diverges by family, and harm remains inelastic everywhere—a regulator auditing one family at one point in time sees a snapshot, not a trajectory.

This work involves dual-use content; mitigations include an isolated sandbox with kill-switch, aggregate-level reporting, and the framework’s defensive orientation. Detailed ethical considerations are provided in Appendix 8.

## 8 Conclusion

We present a modular framework that measures dangerous capability along three orthogonal dimensions $( K , D , H )$ plus derived indicators (DL, BR, ∆, ER), aggregated into a sevendimensional capability profile φ. Our CB instantiation on 12 models yields two headline conclusions: families exhibit sharply distinct capability structures, and capability evolves over time with knowledge compounding, defense diverging by family, and harm remaining inelastic. Cross-judge validation $( \rho > 0 . 7 9 )$ and IRT precision $( \rho = 0 . 9 9 8 )$ ground the measurement. Future work scales the cyber module, elevates Tool-Enhanced evaluation to protocol level, extends the temporal window, improves harm scoring reliability, and integrates real-time monitoring.

## References

[1] Markus Anderljung, Joslyn Barnhart, Anton Korinek, Jade Leung, Cullen O’Keefe, Jess Whittlestone, et al. Frontier ai regulation: Managing emerging risks to public safety. arXiv preprint arXiv:2307.03718, 2023.

[2] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022.

[3] Rishi Bommasani, Kevin Klyman, Shayne Longpre, Sayash Kapoor, Nestor Maslej, Betty Xiong, Daniel Zhang, and Percy Liang. The foundation model transparency index. arXiv preprint arXiv:2310.12941, 2023.

[4] R. Philip Chalmers. mirt: A multidimensional item response theory package for the R environment. Journal ofStatistical Software, 48(6):1–29, 2012.

[5] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. In International Conference on Learning Representations (ICLR), 2024. arXiv:2310.08419.

[6] Susan E. Embretson and Steven P. Reise. Item Response Theory for Psychologists. Lawrence Erlbaum Associates, 2000.

[7] Kehua Feng, Xinyi Shen, Weijie Wang, Xiang Zhuang, Yuqi Tang, Qiang Zhang, and Keyan Ding. SciKnowEval: Evaluating multi-level scientific knowledge of large language models. arXiv preprint arXiv:2406.09098, 2024.

[8] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022.

[9] Shahriar Golchin and Mihai Surdeanu. Time travel in LLMs: Tracing data contamination in large language models. In International Conference on Learning Representations (ICLR), 2024. arXiv:2308.08493.

[10] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300.

[11] Prashant Kulkarni, Andrew Wei, and Nishit Mengar. Sandboxbench: A comprehensive evaluation framework for ai agent containment. Technical report, Supervised

Program for Alignment Research (SPAR), 2025. Final Report, Fall 2025; mentored by Nitzan Shulman; contributed to UK AISI inspect\_evals repository.

[12] John P. Lalor, Hao Wu, and Hong Yu. Building an evaluation scale using item response theory. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2016.

[13] Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. Transactions on Machine Learning Research, 2024. arXiv:2402.05120.

[14] Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, and Summer Yue. LLM defenses are not robust to multi-turn human jailbreaks yet. arXiv preprint arXiv:2408.15221, 2024.

[15] Nathaniel Li, Alexander Pan, Anjali Gopal, et al. The WMDP benchmark: Measuring and reducing malicious use with unlearning. In International Conference on Machine Learning (ICML), 2024. arXiv:2403.03218.

[16] Percy Liang, Rishi Bommasani, Tony Lee, et al. Holistic evaluation of language models. Transactions on Machine Learning Research (TMLR), 2023. arXiv:2211.09110.

[17] Xiao Liu, Hao Yu, Hanchen Zhang, et al. Agent-Bench: Evaluating LLMs as agents. In International Conference on Learning Representations (ICLR), 2024. arXiv:2308.03688.

[18] Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations (ICLR), 2024. arXiv:2310.04451.

[19] Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. Formalizing and benchmarking prompt injection attacks and defenses. In 33rd USENIX Security Symposium (USENIX Security 24), pages 1831– 1847, 2024. arXiv:2310.12815.

[20] Frederic M. Lord and Melvin R. Novick. Statistical Theories ofMental Test Scores. Addison-Wesley, 1968.

[21] Christopher A. Mouton, Caleb Lucas, and Ella Guest. The operational risks of AI in large-scale biological attacks: Results of a red-team study. Technical Report RR-A2977-2, RAND Corporation, 2024.

[22] Long Ouyang, Jeff Wu, Xu Jiang, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2203.02155.

[23] Ethan Perez, Saffron Huang, Francis Song, et al. Red teaming language models with language models. arXiv preprint arXiv:2202.03286, 2022.

[24] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.18290.

[25] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, et al. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

[26] Pedro Rodriguez, Joe Barrow, Alexander Hoyle, John P. Lalor, Robin Jia, and Jordan Boyd-Graber. Evaluation examples are not equally informative: How should that change NLP leaderboards? In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL), pages 4486–4503, 2021. doi:10. 18653/v1/2021.acl-long.346.

[27] Yangjun Ruan, Honghua Dong, Andrew Wang, et al. Identifying the risks of LM agents with an LM-emulated sandbox. In International Conference on Learning Representations (ICLR), 2024. arXiv:2309.15817.

[28] Jonas B. Sandbrink. Artificial intelligence and biological misuse: Differentiating risks of language models and biological design tools. arXiv preprint arXiv:2306.13952, 2023.

[29] Emily H. Soice, Rafael Rocha, Kimberlee Cordova, Michael Specter, and Kevin M. Esvelt. Can large language models democratize access to dual-use biotechnology? arXiv preprint arXiv:2306.03809, 2023.

[30] Pat Verga, Sebastian Hofstatter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. arXiv preprint arXiv:2404.18796, 2024.

[31] Bertie Vidgen, Adarsh Agrawal, Ahmed M. Ahmed, Victor Akinwande, Namir Al-Nuaimi, Najla Alfaraj, et al. Introducing v0.5 of the AI safety benchmark from ML-Commons. arXiv preprint arXiv:2404.12241, 2024.

[32] Sanidhya Vijayvargiya, Aditya Bharat Soni, Xuhui Zhou, Zora Zhiruo Wang, Nouha Dziri, Graham Neubig, and Maarten Sap. OpenAgentSafety: A comprehensive framework for evaluating real-world AI agent safety. In International Conference on Learning Representations (ICLR), 2026. arXiv:2507.06134.

[33] Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, and Bo Li. DecodingTrust: A comprehensive assessment of trustworthiness in GPT models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2306.11698.

[34] Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. Do-not-answer: Evaluating safeguards in LLMs. In Findings of the Association for Computational Linguistics: EACL 2024, pages 896–911, 2024. arXiv:2308.13387.

[35] Tongxin Yuan, Zhiwei He, Lingzhong Dong, Yiming Wang, Ruijie Zhao, Tian Xia, Lizhen Xu, Binglin Zhou, Fangqi Li, Zhuosheng Zhang, Rui Wang, and Gongshen Liu. R-Judge: Benchmarking safety risk awareness for LLM agents. In Findings of the Association for Computational Linguistics: EMNLP 2024, 2024. arXiv:2401.10019.

[36] Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. SafetyBench: Evaluating the safety of large language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (ACL), pages 15537–15553, 2024. arXiv:2309.07045.

[37] Haochen Zhao, Xiangru Tang, Ziran Yang, Xiao Han, Xuanzhi Feng, Yueqing Fan, Senhao Cheng, Di Jin, Yilun Zhao, Arman Cohan, and Mark Gerstein. Chem-SafetyBench: Benchmarking LLM safety on chemistry domain. arXiv preprint arXiv:2411.16736, 2024.

[38] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023. arXiv:2306.05685.

[39] Yujun Zhou, Jingdong Yang, Yue Huang, Kehan Guo, Zoe Emory, Bikram Ghosh, Amita Bedar, Sujay Shekar, Zhenwen Liang, Pin-Yu Chen, Tian Gao, Werner Geyer, Nuno Moniz, Nitesh V. Chawla, and Xiangliang Zhang. LabSafety Bench: Benchmarking LLMs on safety issues in scientific labs. arXiv preprint arXiv:2410.14182, 8:20–31, 2026. doi:10.1038/s42256-025-01152-1.

[40] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## Acknowledgments

We thank the anonymous reviewers for their constructive feedback. This work was supported in part by the National Natural Science Foundation of China under Grant U21B2020.

## Appendix

This appendix provides supporting material for reproducibility and review. It contains, in order: the ethical considerations governing our dual-use evaluation (Appendix 8), the artifact access policy (Appendix 8), the IRT measurement derivations used in the knowledge pipeline (Section 4.1), the correlation and bootstrap procedures underlying the cross-judge analysis (Section 6.1), complete measurement tables omitted from the main text for space, and supplementary figures.

## Ethical

This work involves dual-use content at two levels. First, the CB module contains scenario seeds and harm queries designed to elicit dangerous knowledge from LLMs. Second, our evaluation demonstrates that several commercial models can be induced to produce actionable hazardous protocols across five escalating rounds, from initial refusal to complete operational guidance. We address the resulting ethical obligations across six dimensions.

Why CB in particular. Chemical and biological hazards differ from other AI-safety domains in a crucial respect: their misuse requires no digital infrastructure to cause harm. A jailbroken model that generates a phishing email enables an attack that still must traverse digital defenses; a model that supplies synthesis conditions for a toxic compound shortens the path to physical-world harm. This asymmetry is why we chose CB as the first instantiation, and also why its assets carry stricter disclosure controls than, say, cyber scenario seeds.

Public-interest motivation. This study is conducted in the public interest: to identify and quantify dangerous capabilities of deployed LLMs before they are exploited adversarially. The framework measures general model capabilities rather than validating any specific harmful application; its output—a capability profile—serves regulators and platform operators making safety decisions. Results are framed as capability measurements, never as operational guidance.

Safeguards and compliance. All experiments ran inside an isolated Docker sandbox with a kill-switch mechanism (Section 3.1); no model outputs left the controlled environment, and no evaluation activity interacted with real systems, real users, or real chemical processes. The Tool-Enhanced proto type (Section 3.4) executed commands exclusively against locally hosted virtual targets. API evaluations of closed-source models used standard public interfaces, at academic-usage volumes, without bypassing authentication, rate-limiting, or safety-layer mechanisms, in compliance with the reviewed terms of service of each provider. No model output was used to train, fine-tune, or otherwise improve any machine learning model; outputs were used solely for measurement and are not redistributed in the open data release.

Minimization of harmful disclosure. We report findings at the level of aggregate metrics and representative excerpts rather than releasing complete interaction logs. The case figures in Sections 5.3 and 5.4 illustrate this policy in action: all operational parameters are redacted (temperatures, durations, cutoffs, concentrations, thresholds), pathogen and compound identities are softened to functional descriptions, and commercial reagent names are removed. What remains is the measurement evidence itself— response lengths, escalation structure, and judge verdicts— which supports the paper’s quantitative claims without constituting operational guidance.

Responsible disclosure. The defense-erosion patterns reported here describe publicly observable behaviors of commercial models queried through their own APIs. The scenarioseed design (Section 3.2) recombines elicitation strategies already documented in the public literature; we did not attempt to exploit any vulnerability beyond what the models themselves offered. The evaluation measures, it does not amplify.

Benefit–risk assessment and liability. Systematic, reproducible measurement of dangerous capabilities is essential to effective LLM governance. It gives regulators evidence, operators a basis for safeguards, and researchers a way to track progress. Our temporal findings (Section 5.7) underscore the urgency: if hazardous knowledge accumulates while defenses stagnate, intervention becomes progressively harder. These findings are published exclusively to advance AI safety research; any misuse is the sole responsibility of the misuser.

## Open Science

We support the USENIX Security Open Science Policy. The code and data are available upon reasonable request.

We withhold three categories of dual-use sensitive materials: full framework implementation (internal infrastructure), CB module assets (scenario seeds, harm probes, MCQ bank), and raw interaction logs. Full disclosure would lower attack barriers; these are withheld under USENIX ethics guidelines.

For reviewers, we provide a lightweight minimal reference implementation, covering the four-module pipeline on nonsensitive examples with all metric computation logic. Reviewers can verify the paper’s numbers against primary data. The methodology is fully specified in Sections 3–4, sufficient for third-party reproduction of φ aggregation and statistical analyses from the released materials.

## IRT Measurement Derivation

Item difficulty calibration. Given the 2PL model with guessing parameter 0.25:

$$
P _ { i } = 0 . 2 5 + 0 . 7 5 \Psi _ { i } , \qquad \Psi _ { i } = \bigl [ 1 + \exp ( - ( \Theta - b _ { i } ) ) \bigr ] ^ { - 1 } ,
$$

we calibrate $b _ { i }$ by evaluating at $\theta = 0$ and equating to the pooled pass-rate $p _ { i } \colon$

$$
p _ { i } = 0 . 2 5 + \frac { 0 . 7 5 } { 1 + \exp ( b _ { i } ) }
$$

Subtract 0.25:

$$
p _ { i } - 0 . 2 5 = \frac { 0 . 7 5 } { 1 + \exp ( b _ { i } ) }
$$

Invert both sides:

$$
1 + \exp ( b _ { i } ) = \frac { 0 . 7 5 } { p _ { i } - 0 . 2 5 }
$$

Solve for $\exp ( b _ { i } )$

$$
\exp ( b _ { i } ) = \frac { 0 . 7 5 } { p _ { i } - 0 . 2 5 } - 1 = \frac { 1 - p _ { i } } { p _ { i } - 0 . 2 5 }
$$

Take logarithms:

$$
b _ { i } = \log { \frac { 1 - p _ { i } } { p _ { i } - 0 . 2 5 } } .
$$

Maximum-likelihood estimation of θ. With $b _ { i }$ fixed, the log-likelihood for a response vector $\left\{ x _ { i , T } \right\}$ is

$$
\ell ( \mathbf { \boldsymbol { \Theta } } ) = \sum _ { i } \left[ x _ { i , T } \log P _ { i } ( \mathbf { \boldsymbol { \Theta } } ) + ( 1 - x _ { i , T } ) \log ( 1 - P _ { i } ( \mathbf { \boldsymbol { \Theta } } ) ) \right]
$$

The first and second derivatives are

$$
\begin{array} { c } { { \ell ^ { \prime } ( { \bf { \Theta } } ) = \displaystyle \sum _ { i } \frac { \left( x _ { i , T } - P _ { i } \right) P _ { i } ^ { \prime } ( { \bf { \Theta } } ) } { P _ { i } ( { \bf { \Theta } } ) \left( 1 - P _ { i } ( { \bf { \Theta } } ) \right) } } } \\ { { \ell ^ { \prime \prime } ( { \bf { \Theta } } { \bf { \Theta } } ) = \displaystyle \sum _ { i } \left[ \frac { \left( x _ { i , T } - P _ { i } \right) P _ { i } ^ { \prime \prime } ( { \bf { \Theta } } { \bf { \Theta } } ) } { P _ { i } ( 1 - P _ { i } ) } - \frac { \left( x _ { i , T } - P _ { i } \right) \left( P _ { i } ^ { \prime } \right) ^ { 2 } \left( 1 - 2 P _ { i } \right) } { P _ { i } ^ { 2 } ( 1 - P _ { i } ) ^ { 2 } } \right] } } \end{array}
$$

where $P _ { i } ^ { \prime }$ and $P _ { i } ^ { \prime \prime }$ denote derivatives with respect to θ. Newton–Raphson updates are performed as

$$
\Theta  \Theta - \frac { \ell ^ { \prime } ( \Theta ) } { \ell ^ { \prime \prime } ( \Theta ) }
$$

until convergence to obtain $\hat { \mathsf { \boldsymbol { \theta } } } _ { T }$ .

Fisher information. Let $\Psi _ { i } = \left\lceil 1 + \exp ( - ( \Theta - b _ { i } ) ) \right\rceil ^ { - 1 }$ , hence $P _ { i } = 0 . 2 5 + 0 . 7 5 \Psi _ { i }$ . The derivative of the item response function is

$$
\begin{array} { l } { \displaystyle { \frac { \partial P _ { i } } { \partial \theta } = 0 . 7 5 \Psi _ { i } ( 1 - \Psi _ { i } ) } } \\ { \displaystyle { \quad = 0 . 7 5 \cdot \frac { P _ { i } - 0 . 2 5 } { 0 . 7 5 } \cdot \frac { 1 - P _ { i } } { 0 . 7 5 } } } \\ { \displaystyle { \quad = \frac { \left( P _ { i } - 0 . 2 5 \right) \left( 1 - P _ { i } \right) } { 0 . 7 5 } } . } \end{array}
$$

Using the standard item-information identity $I _ { i } ( { \boldsymbol { \theta } } ) =$ $( \partial P _ { i } / \partial \Theta ) ^ { 2 } / ( P _ { i } ( 1 - P _ { i } ) )$ , the test information at the estimated ability $\theta _ { T }$ is

$$
\begin{array} { l } { { \displaystyle I ( \mathbf { \boldsymbol { \theta } } _ { T } ) = \sum _ { i } \frac { \left( \frac { ( P _ { i , T } - 0 . 2 5 ) ( 1 - P _ { i , T } ) } { 0 . 7 5 } \right) ^ { 2 } } { P _ { i , T } \left( 1 - P _ { i , T } \right) } } } \\ { { \displaystyle \ } } \\ { { \displaystyle \ = \sum _ { i } \frac { \left( P _ { i , T } - 0 . 2 5 \right) ^ { 2 } ( 1 - P _ { i , T } ) ^ { 2 } } { 0 . 7 5 ^ { 2 } P _ { i , T } \left( 1 - P _ { i , T } \right) } } } \\ { { \displaystyle \ } } \\ { { \displaystyle \ = \sum _ { i } \frac { \left( ( P _ { i , T } - 0 . 2 5 ) ( 1 - P _ { i , T } ) \right) ^ { 2 } } { 0 . 7 5 ^ { 2 } P _ { i , T } \left( 1 - P _ { i , T } \right) } } . } \end{array}
$$

Finally, the standard error of the ability estimate is

$$
S E ( \Theta _ { T } ) = I ( \Theta _ { T } ) ^ { - 1 / 2 } .
$$

## Correlation and Bootstrap Procedure

Spearman rank correlation. We measure ranking agreement between two judges via Spearman’s $\rho .$ Define the ranks as $r _ { i } ^ { ( 1 ) }$ and $r _ { i } ^ { ( 2 ) }$ for scenario i. The computation proceeds as:

$$
\bar { r } ^ { ( 1 ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r _ { i } ^ { ( 1 ) } , \qquad \bar { r } ^ { ( 2 ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r _ { i } ^ { ( 2 ) }
$$

The numerator (covariance of ranks) is

$$
\mathcal { N } { = } \sum _ { i = 1 } ^ { N } \bigl ( r _ { i } ^ { ( 1 ) } { - } \bar { r } ^ { ( 1 ) } \bigr ) \bigl ( r _ { i } ^ { ( 2 ) } { - } \bar { r } ^ { ( 2 ) } \bigr )
$$

and the denominator is the product of standard deviations

$$
\mathcal { D } = \sqrt { \sum _ { i = 1 } ^ { N } \left( r _ { i } ^ { ( 1 ) } - \bar { r } ^ { ( 1 ) } \right) ^ { 2 } } \cdot \sqrt { \sum _ { i = 1 } ^ { N } \left( r _ { i } ^ { ( 2 ) } - \bar { r } ^ { ( 2 ) } \right) ^ { 2 } }
$$

Thus

$$
\rho = \frac { \mathcal { N } } { { \mathcal { D } } } .
$$

The coefficient ρ is invariant to monotone rescaling of either judge’s scores, making it appropriate for comparing judges with different absolute score distributions.

Stratified sampling. Cross-judge re-scoring uses a stratified subsample from the 298-scenario defense pool.

Bootstrap confidence intervals. To quantify sampling uncertainty of ρ on the 90-scenario sample, we use nonparametric bootstrap with B = 1,000 replications.

Step 1 (resampling): For each $k = 1 , \ldots , B ,$ draw a bootstrap sample $S _ { k } ^ { * }$ of size 90 with replacement from the original 90 scenarios. Compute the Spearman correlation on $S _ { k } ^ { * }$ , denote

$$
\rho _ { k }
$$

Step 2 (point estimate): Take the mean of the bootstrap distribution

$$
{ \widehat { \boldsymbol { \rho } } } = { \frac { 1 } { B } } \sum _ { k = 1 } ^ { B } { \boldsymbol { \rho } } _ { k } .
$$

Step 3 (interval estimate): The 95% percentile confidence interval is

$$
C I _ { 9 5 } = \big [ \mathfrak { p } _ { ( \lfloor 0 . 0 2 5 B \rfloor ) } , \mathfrak { p } _ { ( \lfloor 0 . 9 7 5 B \rfloor ) } \big ] ,
$$

where $\rho _ { \left( \alpha \right) }$ denotes the empirical α-quantile of the sorted bootstrap sample.

The same bootstrap procedure underlies the harm-dimension re-scoring (360 scored responses) and the per-model consistency tables in the supplementary figures.

## Complete Measurement Tables

Table 10: Complete dangerous capability profiles φ for all 12 models.
<table><tr><td>Model</td><td>K</td><td>D</td><td>H</td><td>DL</td><td>BR</td><td>∆</td><td>ER%</td></tr><tr><td>GPT-5.5</td><td>+1.401 96.3 72.8 0.98</td><td></td><td></td><td></td><td></td><td>0.96-0.010</td><td>28.0</td></tr><tr><td>GPT-5.4</td><td>+1.010 94.4 75.2 0.97</td><td></td><td></td><td></td><td></td><td>0.94-0.035</td><td>15.0</td></tr><tr><td>Kimi-K3</td><td>+0.935 95.8 70.9 0.96 1.00-0.030</td><td></td><td></td><td></td><td></td><td></td><td>2.3</td></tr><tr><td>DeepSeek-V4-Pro</td><td>+0.632 73.377.00.711.11-0.161</td><td></td><td></td><td></td><td></td><td></td><td>0.1</td></tr><tr><td>DeepSeek-V4-Flash +0.448 71.076.00.69</td><td></td><td></td><td></td><td></td><td></td><td>1.10-0.141</td><td>0.3</td></tr><tr><td>Claude-Opus-4.8</td><td>+0.369 99.6 78.0 1.00 0.99 -0.002</td><td></td><td></td><td></td><td></td><td></td><td>78.6</td></tr><tr><td>Claude-Opus-4.5</td><td>+0.333 99.6 75.7 1.00 1.00-0.006</td><td></td><td></td><td></td><td></td><td></td><td>78.5</td></tr><tr><td>Claude-Opus-4</td><td>+0.136 89.9 73.9 0.901.02 +0.155</td><td></td><td></td><td></td><td></td><td></td><td>67.8</td></tr><tr><td>DeepSeek-V3</td><td>+0.127 60.2 74.3 0.561.26-0.110</td><td></td><td></td><td></td><td></td><td></td><td>0.0</td></tr><tr><td>Claude-3-Haiku</td><td>-1.208 76.2 69.3 0.76 1.02 -0.003</td><td></td><td></td><td></td><td></td><td></td><td>0.7</td></tr><tr><td>GPT-40</td><td>-1.109</td><td></td><td></td><td></td><td>66.3 69.5 0.66 1.00 +0.005</td><td></td><td>0.0</td></tr><tr><td>GPT-3.5-Turbo</td><td>-3.073</td><td>65.3</td><td>69.6 0.65</td><td></td><td>1.01 +0.027</td><td></td><td>0.0</td></tr></table>

Table 11: IRT ability estimates $\Theta \pm S E$ for all 24 modelsubdomain combinations.
<table><tr><td>Model</td><td> $\theta _ { \mathrm { b i o } }$ </td><td>SE</td><td> $\theta _ { \mathrm { c h e m } }$ </td><td>SE</td></tr><tr><td>Claude-3-Haiku</td><td>-1.220</td><td>0.094</td><td>-1.196</td><td>0.098</td></tr><tr><td>Claude-Opus-4</td><td>-0.286</td><td>0.099</td><td>+0.558</td><td>0.105</td></tr><tr><td>Claude-Opus-4.5</td><td>+0.773</td><td>0.116</td><td>-0.107</td><td>0.100</td></tr><tr><td>Claude-Opus-4.8</td><td>-0.370</td><td>0.098</td><td>+1.109</td><td>0.112</td></tr><tr><td>DeepSeek-V3</td><td>+0.236</td><td>0.105</td><td>+0.018</td><td>0.101</td></tr><tr><td>DeepSeek-V4-Flash</td><td>+0.529</td><td>0.111</td><td>+0.366</td><td>0.103</td></tr><tr><td>DeepSeek-V4-Pro</td><td>+0.820</td><td>0.117</td><td>+0.445</td><td>0.104</td></tr><tr><td>GPT-3.5-Turbo</td><td>-2.054</td><td>0.101</td><td>-4.092</td><td>0.164</td></tr><tr><td>GPT-40</td><td>-1.767</td><td>0.097</td><td>-0.452</td><td>0.098</td></tr><tr><td>GPT-5.4</td><td>+1.203</td><td>0.127</td><td>+0.816</td><td>0.108</td></tr><tr><td>GPT-5.5</td><td>+1.589</td><td>0.139</td><td>+1.212</td><td>0.113</td></tr><tr><td>Kimi-K3</td><td>+1.041</td><td>0.122</td><td>+0.830</td><td>0.108</td></tr></table>

Table 12 reports the complete pairwise Spearman matrix for the defense dimension; the harm-dimension matrix is reported in the released data.

Table 12: Complete dim1 cross-judge Spearman matrix $( n =$ 90 stratified scenarios).
<table><tr><td></td><td>P</td><td>Gem</td><td>GLM</td><td>Grok</td><td>Llama</td><td>Qwen</td></tr><tr><td>Primary (GPT-4o-mini)</td><td>1.00</td><td>0.818</td><td>0.851</td><td>0.794</td><td>0.641</td><td>0.851</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.818</td><td>1.00</td><td>0.858</td><td>0.842</td><td>0.716</td><td>0.820</td></tr><tr><td>GLM-4.5-Air</td><td>0.851</td><td>0.858</td><td>1.00</td><td>0.821</td><td>0.586</td><td>0.867</td></tr><tr><td>Grok-3-Mini</td><td></td><td>0.794 0.842</td><td>0.821</td><td>1.00</td><td>0.720</td><td>0.789</td></tr><tr><td>Llama-3.3-70b</td><td>0.641</td><td>0.716</td><td>0.586</td><td>0.720</td><td>1.00</td><td>0.667</td></tr><tr><td>Qwen3.5-Flash</td><td>0.851</td><td>0.820</td><td>0.867</td><td>0.789</td><td>0.667</td><td>1.00</td></tr></table>

## Supplementary Figures

This section collects visualizations supporting the main text. The IRT test information curve visualizes where the knowledge bank measures most precisely (Section 6.3). The temporal trajectories extend the headline evolution analysis (Section 5.7) to the derived indicators DL, BR, and ER. The cyber defense profile decomposes the pilot results (Section 5.6) to per-scenario level. The per-model judge consistency heatmaps complement the pooled matrices reported in Section 6.1.

![](images/c2ac2f4f1d40cb4995e15eab7b71f99b8ea21bb9179bf7f8ee270660e00f1b60.jpg)  
Figure 11: IRT test information function I(θ) for the 3,773- item knowledge bank. The curve peaks at $\Theta \approx - 0 . 4 .$ , indicating that the item bank provides maximum measurement precision near the L3 (discern) difficulty level—the region most critical for distinguishing mid-range hazardous knowledge capabilities.

![](images/d302be0f765a77c3d7b75dfca65e92b9ad338e0f0ee80b55f0891b5ff72eebb2.jpg)  
Claude GPT DeepSeek Kimi

![](images/3b80ec8bd98392a49cf42bbffc0bf3921feba42e77adbb7a903aeecd3182d10e.jpg)  
Figure 12: Temporal evolution of the derived defense indicators: (a) Defense Level DL rises within every family while DeepSeek plateaus below 0.75; (b) Breakthrough Ratio BR against the BR = 1 stability line—DeepSeek consistently exceeds it, GPT-5.x dips below it, and Claude tracks it closely.

![](images/a7630b845fc734ea8ee061b9f44beff7de16ffc58b20754a62a83c428b4d58aa.jpg)  
Figure 13: Temporal evolution of the empty-response rate ER. Silent refusal emerges with Claude-Opus-4 (67.8%) in mid-2025, consolidates above 78% in Opus-4.5/4.8, and remains absent from DeepSeek and GPT-4o-class models; GPT-5.x adopts a partial strategy (15–28%).

![](images/364be84cf558f45848115fda9433b3040cb2bb50aca90c3af2dac80599234a69.jpg)  
Figure 14: Per-scenario defense scores (D) across the 16- scenario cyber attack chain. GPT-4o alternates between full defense $( D = 1 0 0$ on SQLi, exfiltration, persistence, and bruteforce scenarios) and minimal defense (D = 35 on XSS, SSTI, and JWT scenarios)—a scenario-dependence invisible in its mean $( D = 6 1 . 2 )$ . DeepSeek-V3 and GPT-3.5-Turbo defend weakly across nearly the entire chain, with isolated spikes at Persistence Backdoor and Brute Force Admin respectively.

![](images/5bb1e8f2b2cef21759b6bf4d22894039fd64a7eb70e22a370e1a92cd7037117b.jpg)  
Figure 15: Model-level judge consistency for Defense (D). Spearman ρ between each alternative judge and the primary judge (GPT-4o-mini). Darker cells indicate higher per-model agreement.

![](images/634be79f9ed2dffcef25db41e5f327c67b7157dc34e8a0cf1ad93d05522816fe.jpg)  
Figure 16: Model-level judge consistency for Harm (H). Spearman ρ between each alternative judge and the primary judge (GPT-4o-mini).