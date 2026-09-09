# MeClear: Cooperative Game-Theoretic Attribution and Risk-Aware Memory Clearance for Long-Horizon LLM Agents

Boyu Yang<sup>∗1</sup>, Jiazheng Sun<sup>∗1</sup>, Zilong Lu<sup>1</sup>, Zhi Qiu<sup>2</sup>, Xin Peng<sup>1</sup>, Jun Zheng<sup>2</sup>,

<sup>1</sup>College of Computer Science and Artificial Intelligence, Fudan University, Shanghai 200433, China <sup>2</sup>School of Cyberspace Science and Technology, Beijing Institute of Technology, Beijing 100081, China

## Abstract

Long horizon Large Language Model (LLM) agents rely on external memory systems to preserve user preferences and task knowledge across extended interactions. Conventional retrieval mechanisms optimize semantic compatibility rather than downstream utility, frequently introducing outdated, misleading, or conflicting evidence into the active context. We present MeClear, a task conditioned memory clearance framework that identifies memories featuring negative downstream utility through cooperative attribution and selectively suppresses them from agent execution. MeClear combines Leave One Out screening with sampled cooperative Shapley attribution to distribute utility across interacting evidence, efectively resolving redundant conflict masking where single removal evaluations fail. Utilizing attribution rankings, MeClear executes a query scoped minimal clearance strategy over a nested filtration, verifying task recovery on the cleared context without permanently altering the persistent memory bank. Comprehensive experimental evaluations across ten long dialogue memory pools demonstrate that MeClear achieves a target recall of 85.9% and an overall task recovery rate of 82.3%, representing a 25.5 percentage point improvement over Leave One Out (LOO) baselines.

Code — https://github.com/FudanSELab/MeClear

## Introduction

Long-horizon Large Language Model (LLM) agents increasingly depend on external memory to retain preferences, observations, and state across extended interactions (Wang et al. 2024a; Zhang et al. 2025). Early architectures demonstrated that persistent experience supports planning, reflection, and continual adaptation (Park et al. 2023). However, growing interaction histories accumulate stale, misleading, redundant, or incompatible records. Recent studies reveal that inaccurate experiences propagate errors across future tasks (Xiong et al. 2025), while semantically related memories often prove contextually inappropriate or functionally harmful (Zhang et al. 2026a; Ha et al. 2026). These findings expose a fundamental failure of relevance-centered retrieval: high semantic similarity does not ensure downstream task utility. Reliable memory deployment therefore demands task-conditioned, postretrieval evaluation of how retrieved records causally influence agent execution.

As illustrated in Figure 1, jointly processed memories exhibit complex dependencies such as redundancy and joint harm, causing isolated record evaluations to fail. While recent frameworks refine credit assignment via evidenceanchored rewards (Ma et al. 2026) or single-record causal interventions (Srivastava 2026), single-record deletion collapses when redundant memories independently sustain task failure, yielding zero observable counterfactual change. Coalition-based attribution resolves this local masking by measuring marginal contributions across subset permutations (Ghorbani and Zou 2019; Jia et al. 2019a; Nematov et al. 2025). Because exact coalition enumeration is computationally prohibitive for multi-record context windows, practical clearance must eficiently approximate cooperative interaction efects while preserving suficient contextual variation to uncover masked toxicity.

Directly modifying persistent memory introduces severe safety risks due to the inherently query-conditioned nature of memory utility across long-horizon deployment. A stored record that degrades performance on a query can provide indispensable contextual grounding for subsequent tasks, rendering permanent database overwrites prone to catastrophic cross-task performance regression (Wang et al. 2024b; Xu 2025). Retrieval-time defenses mitigate this danger by dynamically regulating active context visibility during inference rather than altering database records (Ha et al. 2026; Zhang et al. 2026a). Nevertheless, existing admission filters fail to isolate which interacting memories causally induce downstream task failures, nor do they verify whether suppressing records restores execution accuracy. Reliable memory maintenance consequently demands reversible, minimal context clearance accompanied by explicit, post-suppression behavioral recovery verification.

To address this problem, we present MeClear, a cooperative game-theoretic attribution and risk aware memory clearance framework for long horizon Large Language Model agents. MeClear formulates memory maintenance as a task conditioned decision process connecting counterfactual screening, cooperative Shapley attribution, and verified context clearance. The framework first isolates active contexts from retrieval stochasticity and applies Leave One Out (LOO) screening to filter strong positive evidence. It subsequently executes sampled cooperative attribution to quantify individual memory contributions across diverse coalitions, resolving redundant conflict masking where local deletion fails. Based on attribution priorities, MeClear executes a query scoped minimal clearance strategy over a nested candidate filtration, verifying task recovery prior to output generation without permanently altering the underlying memory bank. To the best of our knowledge, we are the first to leverage cooperative game-theoretic allocation to address harmful memory context clearance in LLM agents. Our contributions are summarized as follows:

![](images/d09e976dd5f6d92383b9df79bebcd56cd5a5e2322f5ef4244b3cbb973ed025b7.jpg)  
Figure 1: Motivation of MeRepair. Semantic relevance alone cannot determine whether retrieved memories are useful for future tasks. Historical utility may hide temporal degradation and memory interactions. MeRepair estimates future memory utility and performs uncertainty-aware repair to maintain reliable long-term memory for LLM agents.

• We propose MeClear, a task conditioned memory clearance framework for long horizon Large Language Model agents, unifying cooperative game attribution, structural interaction diagnosis, and verified risk aware context clearance.

• We develop a cooperative memory attribution methodology integrating Leave One Out screening with sampled Shapley estimation, efectively resolving redundant conflict masking and modeling multi memory interactions.

• We conduct comprehensive evaluations across ten long dialogue memory pools, demonstrating that MeClear achieves an injected target recall of 85.9% and an overall task recovery rate of 82.3%, representing a 25.5 percentage point improvement over Leave One Out baselines.

## Related Work

Memory Management and Operations for LLM Agents External memory enables LLM agents to retain information across extended interactions. Early paradigms rely on heuristic retrieval (Park et al. 2023), hierarchical context (Packer et al. 2023), or verbal reflection (Shinn et al. 2023), whereas recent frameworks optimize structured memory operations via graph representations, ofline consolidation, or reinforcement learning (Chhikara et al. 2025; Zhang et al. 2026b; Wu et al. 2026; Yu et al. 2026; Yan et al. 2026). However, superior retrieval does not guarantee downstream task utility. Semantic relevance frequently retrieves stale, redundant, or functionally incompatible records (Zhang et al. 2026a; Ha et al. 2026; Liu et al. 2026). Rather than relying solely on semantic similarity or admission filtering, MeClear evaluates post-retrieval functional efects and contextual interactions to dynamically regulate memory visibility.

Long-Term Memory Evaluation and Attribution Longterm memory benchmarks assess overall response quality across multi-session dialogues and reasoning (Maharana et al. 2024; Wu et al. 2024; Li et al. 2026), but ofer limited causal credit assignment for individual records. While operation-level rewards (Ma et al. 2026) and single-record counterfactual interventions (Saha Roy et al. 2025; Srivastava 2026) attempt fine-grained attribution, they miss interaction-dependent phenomena such as mutual redundancy or joint toxicity. Although Shapley-based valuations capture coalition-level marginal contributions (Ghorbani and Zou 2019; Jia et al. 2019a; Nematov et al. 2025), applying them to agentic context remains unexplored. MeClear adapts coalition-aware attribution to persistent memory, unifying interaction diagnostics with behaviorally verified context clearance under bounded computational budgets.

Memory Editing and Safe Context Clearance Model editing updates factual representations via parameter shifts or external patching (Meng et al. 2022a,b; Mitchell et al. 2021a,b; Wang et al. 2024b). In agent memory, record utility is dynamic and query-dependent; permanent modification risks propagating irreversible errors across future tasks (Yu et al. 2026; Yan et al. 2026; Xu 2025). Retrieval-time filtering avoids database overwrites (Ha et al. 2026; Zhang et al. 2026a) but fails to verify interaction-dependent behavioral consequences before suppressing context. MeClear bridges this gap by unifying reversible visibility control, counterfactual contribution analysis, and behavioral verification, suppressing harmful memory influences on a per-query basis while leaving the underlying database intact.

## Problem Formulation

Consider a long-horizon Large Language Model (LLM) agent that continuously accumulates information across extended interactions. At task step t, the agent maintains an external memory bank $B _ { t }$ containing historical observations, factual evidence, and evolving user-specific information. Given a query $q _ { t }$ , an existing retrieval mechanism $\rho$ returns a bounded execution context

$$
\mathcal { M } _ { t } = \rho ( q _ { t } , \mathcal { B } _ { t } ; K ) = \{ m _ { 1 } , m _ { 2 } , . . . , m _ { K } \} \subseteq \mathcal { B } _ { t } ,\tag{1}
$$

where K is the retrieval budget determined by the underlying memory system and the available context capacity. Each memory record is represented as $m _ { i } = ( x _ { i } , \xi _ { i } )$ , where $x _ { i }$ denotes its textual content and $\xi _ { i }$ contains auxiliary information such as timestamp, source, confidence, or provenance. MeClear does not replace the underlying retriever. Instead, it operates on the already retrieved context $\mathcal { M } _ { t }$ and evaluates whether each exposed memory actually benefits the current task. This separation is essential because semantic relevance does not guarantee downstream utility: a retrieved memory may be topically related to the query while being outdated, misleading, redundant, or incompatible with other active evidence. Let $A _ { \theta }$ denote the task agent parameterized by θ. For any active memory coalition $\breve { S } \subseteq \breve { M } _ { t } ,$ the agent generates an output y according to $p _ { \theta } ( y \mid q _ { t } , S )$ ). We define the task-conditioned value of coalition $S$ as

$$
v _ { t } ( S ) = \mathbb { E } _ { y \sim p _ { \theta } ( \cdot \vert q _ { t } , S ) } [ r _ { t } ( y ) ] , \qquad v _ { t } : 2 ^ { \mathcal { M } _ { t } }  [ 0 , 1 ] ,\tag{2}
$$

where $r _ { t } ( y ) \in [ 0 , 1 ]$ measures task correctness, and $v _ { t } ( S )$ denotes the query-conditioned expected utility under memory subset $S ,$ , empirically estimated as $\widehat { v } _ { t } ( S )$ via agent execution. Because retrieved memories interact non-independently through redundancy or complementarity, we formulate $( \bar { \mathcal { M } } _ { t } , \bar { \mathcal { v } } _ { t } )$ as a cooperative game with memories as players and $v _ { t }$ as the characteristic function. The cooperative contribution of memory $m _ { i }$ is defined as

$$
\sum _ { S \subseteq { \mathcal M } _ { t } \backslash \{ m _ { i } \} } { \frac { | S | ! ( K - | S | - 1 ) ! } { K ! } } \left[ v _ { t } ( S \cup \{ m _ { i } \} ) - v _ { t } ( S ) \right] .\tag{3}
$$

The coeficient in Equation (3) equals the probability that S forms the predecessor coalition of $m _ { i }$ under a uniformly random ordering of the K memories. Thus, $\psi _ { t , i }$ measures the marginal efect of $m _ { i }$ across diverse contextual coalitions rather than only around the complete retrieved context. The resulting allocation satisfies $\textstyle \sum _ { i = 1 } ^ { K } \psi _ { t , i } = v _ { t } ( { \mathcal { M } } _ { t } ) - v _ { t } ( \emptyset )$ A positive contribution indicates that the memory improves task utility on average, whereas a negative value indicates that its presence decreases task utility across coalition contexts. To separate meaningful harm from negligible negative variation, we introduce a single tolerance $\tau \geq 0$ and define the query-conditioned harmful memory set as

$$
\mathcal { H } _ { t } = \left\{ m _ { i } \in \mathcal { M } _ { t } \ | \ \psi _ { t , i } < - \tau \right\} .\tag{4}
$$

Membership in $\mathcal { H } _ { t }$ is not an intrinsic or permanent property of a memory record. It is determined jointly by the current query $q _ { t }$ , the retrieved context $\mathcal { M } _ { t }$ , the task agent $A _ { \theta }$ , and the task-value function $v _ { t }$ . MeClear therefore treats harmfulmemory attribution as a query-conditioned visibility decision rather than an irreversible modification of the persistent memory bank. For any candidate clearance set $\mathcal { C } \subseteq \bar { \mathcal { H } } _ { t }$ , clearing $\mathcal { C }$ produces the active context $\mathcal { M } _ { t } \backslash \mathcal { C }$ . The corresponding task gain is defined as $g _ { t } ( \mathcal { C } ) = v _ { t } ( \mathcal { M } _ { t } \backslash \mathcal { C } ) - v _ { t } ( \mathcal { M } _ { t } )$ . A desirable intervention should first maximize task recovery and then, among all interventions attaining the same recovery, remove the fewest memories. We formulate this parameter-free lexicographic objective as

$$
G _ { t } = \operatorname* { m a x } _ { \mathcal { C } \subseteq \mathcal { H } _ { t } } g _ { t } ( \mathcal { C } ) , \qquad \mathcal { C } _ { t } \in \underset { \mathcal { C } \subseteq \mathcal { H } _ { t } } { \arg \operatorname* { m i n } } \left\{ \vert \mathcal { C } \vert \vert g _ { t } ( \mathcal { C } ) = G _ { t } \right\} ,\tag{5}
$$

where $G _ { t }$ determines the maximum task gain attainable by clearing a subset of harmful memories, while $\mathcal { C } _ { t }$ selects a minimum-cardinality intervention among all clearance sets attaining this gain. The formulation implements the minimal-intervention principle without introducing an additional weighting coeficient. Because $\emptyset \subseteq \mathcal { H } _ { t }$ and $g _ { t } ( \emptyset ) = 0$ the optimal gain always satisfies $G _ { t } \geq 0 .$ . If no nonempty subset improves the task, the minimum-cardinality solution is $\mathcal { C } _ { t } = \emptyset$ . The exact contribution values and task gains are generally unavailable during execution. MeClear therefore approximates them through finite counterfactual evaluations and verifies the selected intervention on the current query.

## Method

As shown in Figure 2, MeClear selectively suppresses harmful memories from the active context without modifying persistent storage. It operates across four stages: adaptive retrieval, local screening, cooperative attribution, and verified clearance. The architecture introduces no stage specific tuning coeficients, governed transparently by context capacity, permutation budget, and a single tolerance threshold.

Adaptive Task-Conditioned Context Retrieval For query $q _ { t } ,$ the external memory system retrieves an active context $\mathcal { M } _ { t }$ . The term adaptive denotes the query-dependent nature of context selection rather than a re-trained retrieval model, as MeClear operates as a post-retrieval layer without modifying the parameters or indexing of $\rho .$ This separation prevents semantic similarity from being conflated with downstream utility, as the retriever estimates query-record compatibility whereas $v _ { t } ( S )$ evaluates agent behavior under exposed coalition S. Consequently, memories with comparable retrieval scores can produce distinct task outcomes due to temporal drift, factual conflicts, redundancy, or contextual interaction:

$$
\begin{array} { r } { \mathcal { M } _ { t } = \rho ( q _ { t } , B _ { t } ; K ) . } \end{array}\tag{6}
$$

To isolate the behavioral impact of memory suppression from retrieval volatility, MeClear freezes $\mathcal { M } _ { t }$ prior to attribution, executing all subsequent interventions on subsets of this fixed context without re-invoking the retriever. For each coalition $S \subseteq { \mathcal { M } } _ { t }$ , the empirical value $\widehat { v } _ { t } ( S )$ is obtained by executing the task agent on S under a fixed evaluator and cached by memory identity. Evaluating the complete context value $\widehat { v } _ { t } ( \ M _ { t } )$ first establishes a shared reference baseline for all subsequent counterfactual comparisons.

![](images/c7dcadb1a4663e11f90346e221a62c619c25fc27dbe384ef26e30b9515a372d3.jpg)  
Figure 2: Overall architecture of MeClear, combining local screening and cooperative Shapley attribution to output a queryscoped cleared context without modifying persistent storage.

Local Counterfactual Screening MeClear first establishes a local contribution profile using Leave-One-Out (LOO) counterfactual interventions. For each retrieved record $m _ { i } \in \mathcal { M } _ { t }$ , the empirical local efect is defined as:

$$
\widehat { d } _ { t , i } = \widehat { v } _ { t } ( { \mathcal { M } } _ { t } ) - \widehat { v } _ { t } \left( { \mathcal { M } } _ { t } \setminus \{ m _ { i } \} \right) .\tag{7}
$$

A negative value of $\widehat { d } _ { t , i }$ indicates that suppressing $m _ { i }$ improves execution performance on the complete context, whereas a positive value reflects a reduction in task utility upon removal. Values approaching zero remain inconclusive because auxiliary or substitutable evidence may mask the underlying efect. Consequently, MeClear utilizes LOO as an interpretable local screening signal rather than a definitive harmfulness criterion. The structural limitation of singlerecord deletion becomes particularly pronounced under redundant harmful evidence, where multiple records independently induce task degradation.

Theorem 1 (Redundant-Harm Blind Spot). Let $m _ { i }$ and $m _ { j }$ be substitutable harmful memories whose joint efect on the task valuefunction satisfies:

$$
\begin{array} { l } { { v _ { t } ( S ) = u _ { t } \left( S \setminus \{ m _ { i } , m _ { j } \} \right) } } \\ { { \phantom { = } - \Delta _ { t } \mathbf { 1 } \left[ S \cap \{ m _ { i } , m _ { j } \} \neq \emptyset \right] , \quad \Delta _ { t } > 0 , } } \end{array}\tag{8}
$$

where $u _ { t }$ is independent of $m _ { i }$ and $m _ { j } .$ . If both records are present in $\mathcal { M } _ { t } ,$ , then:

$$
d _ { t , i } = d _ { t , j } = 0 , \qquad \psi _ { t , i } = \psi _ { t , j } = - \frac { \Delta _ { t } } { 2 } .\tag{9}
$$

Deleting either record leaves the other active, maintaining the degradation penalty in Equation (8) and yielding zero LOO efect. Under random permutations, $m _ { i }$ precedes $m _ { j }$ with probability $1 / 2 .$ , yielding expected contribution $\dot { \psi _ { t , i } } = - \Delta _ { t } / 2$ as shown in Appendix $\mathbf { A } .$ Theorem 1 demonstrates that a zero LOO efect does not imply zero cooperative harm. While LOO measures utility only locally, cooperative attribution evaluates marginal contributions across coalitions where redundant substitutes are absent. MeClear therefore retains the local profile as a diagnostic reference while anchoring harmful memory selection on coalition-averaged contributions.

Counterfactual Cooperative Attribution MeClear next estimates memory contributions through the equivalent permutation representation. As depicted in Figure 3, individual memory utility is quantified via permutation sampled Shapley contribution estimation across contextual permutations, while pairwise interaction profiles diagnose structural dependencies among evidence. Let Π<sub>t</sub> denote the set of all permutations of $\mathcal { M } _ { t }$ . For $\pi \in \Pi _ { t }$ , let $P _ { i } ( \pi )$ contain the memories appearing before $m _ { i } .$ . The exact contribution is:

$$
\psi _ { t , i } = { \frac { 1 } { | \Pi _ { t } | } } \sum _ { \pi \in \Pi _ { t } } \left[ v _ { t } \left( P _ { i } ( \pi ) \cup \left\{ m _ { i } \right\} \right) - v _ { t } \left( P _ { i } ( \pi ) \right) \right] .\tag{10}
$$

Unlike single-record screening, Equation (10) evaluates the marginal contribution ofm across diverse predecessor coalitions, exposing negative utility masked by substitution or redundancy in the full context. Because enumerating all permutations is computationally infeasible for non-trivial context size K, MeClear draws L independent permutations $\pi ^ { ( 1 ) } , \dots , \pi ^ { ( L ) }$ and computes the sample-average estimate:

$$
\widehat { \psi } _ { t , i } ^ { ( L ) } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \left[ \widehat { v } _ { t } \left( P _ { i } ( \pi ^ { ( \ell ) } ) \cup \{ m _ { i } \} \right) - \widehat { v } _ { t } \left( P _ { i } ( \pi ^ { ( \ell ) } ) \right) \right] .\tag{11}
$$

Each permutation is sampled over the context $\mathcal { M } _ { t } .$ , ensuring beneficial, neutral, and harmful memories co-occur in prede cessor coalitions to preserve the underlying game structure.

Proposition 1 (Finite-Sample Cooperative Estimation). For independent uniformly sampled permutations, the estimator in Equation (11) is unbiased:

$$
\mathbb { E } \left[ \widehat { \psi } _ { t , i } ^ { ( L ) } \right] = \widetilde { \psi } _ { t , i } ,\tag{12}
$$

![](images/0c9b02f0715b11736c4949d68f506c3e5ad6803f6db31121a91edbbd4e96cebe.jpg)  
Figure 3: Counterfactual attribution and interaction profiling in MeClear. Top: Individual memory harm attribution. Bottom: Pairwise interaction analysis diagnosing direct, redundant, and joint memory harm dependencies.

andfor any $\varepsilon > 0 ,$ ε > , satisfies the concentration bound:

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { 1 \leq i \leq K } \left| \widehat { \psi } _ { t , i } ^ { ( L ) } - \widetilde { \psi } _ { t , i } \right| \geq \varepsilon \right) \leq 2 K \exp \left( - \frac { L \varepsilon ^ { 2 } } { 2 } \right) .\tag{13}
$$

The summands in Equation (11) represent independent bounded observations of marginal contribution. Applying Hoefding’s inequality with a union bound across all K records yields the concentration result in Appendix B. Proposition 1 characterizes the approximation error introduced by finite sampling for a fixed empirical game without assuming the underlying model is an unbiased population estimator. MeClear constructs the operational harmful set by thresholding estimated contributions with tolerance τ:

$$
\widehat { \mathcal { H } } _ { t } = \left\{ m _ { i } \in \mathcal { M } _ { t } \ \middle | \ \widehat { \psi } _ { t , i } ^ { ( L ) } < - \tau \right\} .\tag{14}
$$

Using the unified tolerance $\tau$ avoids introducing additional hyperparameter thresholds into attribution. To characterize higher-order structural dependencies, MeClear further estimates pairwise non-additivity. Let $\Delta _ { i j } { \widehat { v } } _ { t } ( S ) =$ $\widehat v _ { t } ( S \cup \{ m _ { i } , m _ { j } \} ) - \widehat v _ { t } ( S \cup \{ m _ { i } \} ) - \widehat v _ { t } ( S \cup \{ m _ { j } \} ) + \widehat v _ { t } ( S )$ For predecessor set $P _ { i j } ( \pi )$ preceding both m<sub>i</sub> and $m _ { j }$ , the sampled interaction efect is:

$$
\widehat { \omega } _ { t , i j } ^ { ( L ) } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \Delta _ { i j } \widehat { v } _ { t } \left( P _ { i j } ( \pi ^ { ( \ell ) } ) \right) , \qquad i \neq j .\tag{15}
$$

The resulting interaction profile diagnoses non-additive structures such as substitution, complementarity, and joint interference, serving as a structural behavioral diagnostic rather than a semantic classifier.

Verified Query-Scoped Clearance While the operational harmful set identifies candidate negative memories, suppressing all negatively attributed records may eliminate more context than necessary for task recovery. MeClear therefore approximates the ideal objective in Equation (5) through an ordered sequence of query-scoped clearance candidates. Let $h _ { t } = | \widehat { \mathcal { H } } _ { t } |$ . The records in $\widehat { \mathcal { H } } _ { t }$ are ordered from the most negative to the least negative estimated contribution:

$$
\widehat { \psi } _ { t , \sigma _ { t } ( 1 ) } ^ { ( L ) } \leq \widehat { \psi } _ { t , \sigma _ { t } ( 2 ) } ^ { ( L ) } \leq \cdots \leq \widehat { \psi } _ { t , \sigma _ { t } ( h _ { t } ) } ^ { ( L ) } ,\tag{16}
$$

where $\sigma _ { t }$ is a permutation of indices in $\widehat { \mathcal { H } } _ { t \cdot } \mathbf { A }$ smaller index indicates a stronger negative contribution and thus a higher priority for removal. MeClear constructs a nested family of clearance candidates along this priority chain:

$$
\mathcal { C } _ { t , j } = \left\{ m _ { \sigma _ { t } ( 1 ) } , m _ { \sigma _ { t } ( 2 ) } , \ldots , m _ { \sigma _ { t } ( j ) } \right\} , \qquad 0 \leq j \leq h _ { t } , \qquad\tag{17}
$$

with baseline $\mathcal { C } _ { t , 0 } ~ = ~ \emptyset .$ . The candidate sequence satisfies $\mathcal { C } _ { t , j - 1 } \subseteq \mathcal { C } _ { t , j }$ and $| \mathcal { C } _ { t , j } | = j$ , reducing the combinatorial search space from $2 ^ { h _ { t } }$ arbitrary subsets to $h _ { t } + 1$ candidate contexts. For each candidate, MeClear computes the empirical query-scoped gain:

$$
\widehat { g } _ { t } ( j ) = \widehat { v } _ { t } \left( \mathcal { M } _ { t } \setminus \mathcal { C } _ { t , j } \right) - \widehat { v } _ { t } ( \mathcal { M } _ { t } ) .\tag{18}
$$

Because all candidates are masked directly from the frozen context $\mathcal { M } _ { t }$ without re-invoking retrieval, any observed behavioral gain is strictly attributable to memory removal. Let $\nu _ { t } ( S ) ~ \in ~ \{ 0 , 1 \}$ denote a fixed task-recovery predicate for query $q _ { t }$ . The set of admissible candidate indices

Panel A. Case construction and causal signature
<table><tr><td>Case</td><td>Fault</td><td>QA → Gold</td><td>Injected pair M</td><td>Causal signature c</td><td>Mechanism</td></tr><tr><td>A</td><td>Temporal</td><td>John in Italy: which month? → Dec. 2023</td><td>Two paraphrases: Jan. 2023</td><td>(1.00, 0.50, 0.50, 0.50)</td><td>Flat plateau; each fault masks the other&#x27;s local effect</td></tr><tr><td>B</td><td>Conflict</td><td>Dogs&#x27; reaction to snow? → Confused</td><td>Two paraphrases: excited/thrilled</td><td>(1.00, 0.00, 0.00, 0.00)</td><td>Each fault independently reaches the failure floor</td></tr><tr><td>C</td><td>Factual</td><td>Effect on Calvin&#x27;s songs? → Fresh vibe</td><td>Two paraphrases: warm, vintage tone</td><td>(1.00, 0.00, 0.00, 0.00)</td><td>Floor masking; LOO instead selects background  $G _ { 3 }$ </td></tr></table>

Panel B. Attribution and deletion outcome
<table><tr><td>Case</td><td>MeClear contribution  ${ \widehat { \psi } } ( F _ { 1 } ) , { \widehat { \psi } } ( F _ { 2 } )$ </td><td>LOO</td><td>MeClear</td><td>ContextCite</td><td>ProxySPEX</td><td>LLM baseline</td></tr><tr><td>A</td><td>(−0.094, −0.219)</td><td>∅ 0.50 / No</td><td>M (Exact) 1.00 / Yes</td><td>∅ 0.50 / No</td><td>{F1} 0.50 /No</td><td>∅ 0.50 / No</td></tr><tr><td>B</td><td>(−0.281, −0.156)</td><td>∅ 0.00 / No</td><td>M (Exact) 1.00 / Yes</td><td>{F1} 0.10 /No</td><td>{F1} 0.10 /No</td><td>MU{G1} 0.00/No</td></tr><tr><td>C</td><td>(−0.206, −0.075)</td><td>{G3} 0.10 /No</td><td>M (Exact) 1.00 / Yes</td><td>{F2} 0.10 /No</td><td>∅ 0.00 / No</td><td>M (Exact) 1.00 / Yes</td></tr></table>

Notes. $\mathbf { c } = ( v ( G ) , v ( G + F _ { 1 } ) , v ( G + F _ { 2 } ) , v ( G + M ) )$ , with frozen top-5 context G and error set $\mathcal { M } = \{ F _ { 1 } , F _ { 2 } \}$ . Cells report deleted set H, post-deletion task value, and Recovery status (requiring 2/2 correct trials). Hyperparameters: MeClear $L = 1 6 , \kappa = \tau = 0 . 0 5 ;$ ; ContextCite and ProxySPEX B = 32; LLM baseline score threshold ≥ 0.5. Cases are qualitative examples from qualifying redundant scenarios.

Table 1: Representative redundant-conflict cases illustrating how MeClear resolves local masking.

$\mathcal { T } _ { t } \subseteq \{ 0 , 1 , \ldots , h _ { t } \}$ is defined as:

$$
\begin{array} { c } { \mathcal { I } _ { t } = \{ \boldsymbol { 0 } \} \cup \Big \{ j \in \{ 1 , \dots , h _ { t } \} \ \Big | } \\ { \widehat { g } _ { t } ( j ) > 0 , \ \nu _ { t } \left( \mathcal { M } _ { t } \setminus \mathcal { C } _ { t , j } \right) = 1 \Big \} . } \end{array}\tag{19}
$$

Including j = 0 ensures $\mathcal { T } _ { t }$ is non-empty. MeClear selects the admissible candidate maximizing verified empirical gain with minimal intervention cardinality:

$$
\widehat { j } _ { t } = \operatorname* { m i n } _ { \substack { j \in \mathcal { T } _ { t } } } \operatorname* { m a x } _ { \substack { j \in \mathcal { J } _ { t } } } ( j ) .\tag{20}
$$

The maximization isolates candidates with peak empirical recovery, while the outer minimum breaks ties by choosing the smallest index and minimal intervention cardinality. The resulting query-conditioned context is $\widetilde { \mathcal { M } } _ { t } = \mathcal { M } _ { t } \backslash \mathcal { C } _ { t , \widehat { j } _ { t } } .$ which serves as a query-scoped active context without permanently modifying the persistent memory bank $B _ { t }$

Theorem 2 (Query-Scoped Clearance Guarantee). For any fixed empirical evaluator $\widehat { v } _ { t }$ and nested clearance family in Equation (17), the context selected by Equation (20) guarantees empirical non-degradation:

$$
\widehat { v } _ { t } \left( \widetilde { \mathcal { M } } _ { t } \right) \geq \widehat { v } _ { t } \left( \mathcal { M } _ { t } \right) .\tag{21}
$$

$I f \widehat { j } _ { t } > 0 ;$ , the selected context satisfies verified recovery:

$$
\nu _ { t } \left( \widetilde { \mathcal { M } } _ { t } \right) = 1 .\tag{22}
$$

Furthermore, $\mathcal { C } _ { t , \widehat { j } _ { t } }$ achieves minimal cardinality among all admissible candidates attaining the maximum gain.

Because the unchanged baseline $\mathcal { C } _ { t , 0 } = \varnothing$ is always admissible with zero gain, the selected candidate cannot yield negative empirical gain, ensuring non-degradation as detailed in Appendix C.

## Experiments

We evaluate MeClear using Kimi-k2.6 as the task agent and Qwen3.6-Flash as the judge evaluator across 745 causally verified test cases containing 1,115 fault records, synthesized from 368 clean queries over ten LoCoMo conversations. Under a hybrid retrieval budget K = 5, memory faults span direct conflicts with $n = 3 7 5$ and |M| = 1, redundant conflicts with $n = 2 8 5$ and $| M | = 2$ , and joint interactions with n = 85 and $| M | = 2 .$ . We compare MeClear configured with $\tau = \kappa = 0 . 0 5$ and sampling budget $L = 1 6$ against Leave-One-Out, ContextCite, and ProxySPEX across four metrics: Target Recall at |M|, Complete Set Recall, Exact Set Match, and Binary Task Recovery. Detailed dataset construction, fault verification protocols, and metric definitions are provided in Appendix D.

## Overall Attribution Accuracy and Task Recovery

MeClear outperforms all baselines in attribution precision and downstream task recovery. As shown in Figure 4, MeClear achieves 85.9% memory micro-recall, 47.0% exact fault set identification, and 82.3% binary task recovery, whereas LOO sufers from local blind spots, obtaining only 38.3% recall and 56.8% recovery. While ContextCite and ProxySPEX yield high recall, their inability to isolate complete harmful coalitions limits exact identification to 43.5% and 54.2%. Paired comparisons across n = 745 samples in Figure 6 confirm MeClear’s dominance, yielding 212 wins against 22 losses over LOO and significant recovery gains of +25.5% over LOO, +5.0% over ProxySPEX, and +38.7% over Qwen3.5-Plus.

![](images/924b5c3cf19c8cdeaa465a4bf64b75001200c395fe78b76de84fd5827da2ed62.jpg)

Figure 4: Overall attribution accuracy and task recovery across evaluation cases compared to baselines.  
![](images/122609cfe29313c336148d99dd61cb861df44ce80e66f0c89f382881d534c98b.jpg)  
Figure 5: Performance across direct conflict, $n = 3 7 5 ,$ redundant masking, $n = 2 8 5$ , and joint interaction, $n = 8 5 .$

## Performance Across Structural Fault Mechanisms

Stratifying evaluation across direct conflicts with $n = 3 7 5 .$ redundant masking with $n \ = \ 2 8 5 .$ , and joint faults with $n = 8 5$ reveals that performance gaps stem from multimemory dependencies, as shown in Figure 5. In direct conflicts, LOO achieves 68.3% recall and 87.2% recovery, while MeClear reaches 83.2% recall and 89.6% recovery. Under redundant conflicts, local substitution causes LOO to collapse to 12.3% recall and 6.3% recovery. By evaluating marginal contributions across permutations, MeClear overcomes local masking, sustaining 83.2% recall, 49.1% exact identification, and 68.4% recovery. Under joint non-additive interactions where LOO fails with 0.0% exact match, MeClear achieves 87.1% recall and 91.8% recovery, demonstrating robust handling of higher-order evidence dependencies.

## Case Studies on Redundant Masking Resolution

Table 1 illustrates how MeClear resolves local masking in representative redundant fault scenarios. Under plateau $\begin{array} { r c l } { \bar { \mathbf { c } } } & { = } & { \left[ 1 . 0 0 , 0 . 5 0 , 0 . 5 0 , 0 . 5 0 \right] ) } \end{array}$ and floor $\begin{array} { r l } { ( \mathbf { c } } & { { } = } \end{array}$ $[ 1 . 0 0 , 0 . 0 0 , 0 . 0 0 , \dot { 0 } . 0 0 ] )$ ) masking signatures, LOO fails by returning ∅ or removing benign records because single-fault removals leave substitute harm active. In contrast, MeClear accurately attributes negative cooperative contributions (e.g., $\widehat { \psi } ( F _ { 1 } ) = - 0 . 0 9 4$ and $\widehat { \psi } ( F _ { 2 } ) = - 0 . 2 1 9$ in Case A), successfully isolating the exact harmful set M and restoring the full task utility.

![](images/db7b7947968c61986f99277ed79890e64128e4b2a7d865d02c0346eb8d507173.jpg)  
Figure 6: Paired comparative advantage and recovery gains. Panel A: head-to-head win/loss statistics across $n = 7 4 5$ cases; Panel B: recovery diferentials ∆Recovery with 95% confidence intervals.

![](images/e1fc68ce74ce56d530c52096e3126e1175fc1882f62230089c749c02f9e0b8a4.jpg)  
Figure 7: Query-scoped memory clearance and context composition. Panel A: clearance profiles by conflict type; Panel B: context composition across exact recovery, over-selection, and non-recovery.

## Query-Scoped Clearance and Context Composition

Figure 7 confirms that MeClear consistently maximizes task recovery while minimizing context distortion. MeClear achieves an 80.5% overall recovery rate (44.0% exact set and 36.5% harmless over-selection) with only 14.8% unrecovered failures. Conversely, LOO yields 40.5% failures due to missed redundant faults, and direct baseline LLM filtering via Qwen3.5-Plus results in 52.8% failures from overtruncation or hallucinated deletion. Thus, MeClear’s verified selection rule efectively isolates minimal harmful coalitions while preserving the beneficial task context.

## Conclusion

Unchecked memory accumulation threatens long-horizon LLM agents by turning external memory into an executiondegrading bottleneck of conflicting evidence. To break this impasse, we propose MeClear, a framework combining permutation-sampled Shapley attribution with a queryscoped verified clearance gateway. Extensive evaluations demonstrate that MeClear achieves an 85.9% target recall and an 82.3% task recovery rate, outperforming Leave-One-Out baselines by 25.5 percentage points. Ultimately, this work establishes coalition-aware context clearance as a foundational pillar for safe, non-destructive agent memory maintenance.

## References

Chhikara, P.; Khant, D.; Aryan, S.; Singh, T.; and Yadav, D. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Ghorbani, A.; and Zou, J. 2019. Data shapley: Equitable valuation of data for machine learning. In International conference on machine learning, 2242–2251. PMLR.

Ha, H.; Kim, J.; Qian, C.; Liu, J.; Campbell, W. M.; Wu, Y.; Zhang, Y.; McKeown, K.; Hakkani-Tur, D.; and Ji, H. 2026. MemGuard: Preventing Memory Contamination in Long-Term Memory-Augmented Large Language Models. arXiv preprint arXiv:2605.28009.

Jia, R.; Dao, D.; Wang, B.; Hubis, F. A.; Hynes, N.; Gürel, N. M.; Li, B.; Zhang, C.; Song, D.; and Spanos, C. J. 2019a. Towards eficient data valuation based on the shapley value. In The 22nd international conference on artificial intelligence and statistics, 1167–1176. PMLR.

Jia, R.; Dao, D.; Wang, B.; Hubis, F. A.; Hynes, N.; Gürel, N. M.; Li, B.; Zhang, C.; Song, D.; and Spanos, C. J. 2019b. Towards eficient data valuation based on the shapley value. In The 22nd international conference on artificial intelligence and statistics, 1167–1176. PMLR.

Li, Y.; Guo, W.; Zhang, L.; Xu, R.; Huang, M.; Liu, H.; Xu, L.; Xu, Y.; and Liu, J. 2026. Locomo-plus: Beyond-factual cognitive memory evaluation framework for llm agents. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), 25085–25100.

Liu, C.; Yang, Y.; Pu, S. X.; Liu, Y.; Long, L.; Guo, Y.; Chen, N.; Weng, Z.; Kochkina, E.; Kaur, S.; et al. 2026. World-MemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction. arXiv preprint arXiv:2605.29341.

Ma, W.; Feng, X.; Huang, L.; Feng, X.; Ma, Z.; Xu, J.; Gao, J.; Hao, J.; He, R.; and Qin, B. 2026. Fine-Mem: Fine-Grained Feedback Alignment for Long-Horizon Memory Management. arXiv preprint arXiv:2601.08435.

Maharana, A.; Lee, D.-H.; Tulyakov, S.; Bansal, M.; Barbieri, F.; and Fang, Y. 2024. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 13851–13870.

Meng, K.; Bau, D.; Andonian, A. J.; and Belinkov, Y. 2022a. Locating and editing factual associations in gpt. In Advances in neural information processing systems.

Meng, K.; Sharma, A. S.; Andonian, A. J.; Belinkov, Y.; and Bau, D. 2022b. Mass-editing memory in a transformer. In The eleventh international conference on learning representations.

Mitchell, E.; Lin, C.; Bosselut, A.; Finn, C.; and Manning, C. D. 2021a. Fast model editing at scale. arXiv preprint arXiv:2110.11309.

Mitchell, E.; Lin, C.; Bosselut, A.; Finn, C.; and Manning, C. D. 2021b. Fast model editing at scale. arXiv preprint arXiv:2110.11309.

Nematov, I.; Kalai, T.; Kuzmenko, E.; Fugagnoli, G.; Sacharidis, D.; Hose, K.; and Sagi, T. 2025. Source attribution in retrieval-augmented generation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 317–332. Springer.

Packer, C.; Fang, V.; Patil, S. G.; Lin, K.; Wooders, S.; and Gonzalez, J. E. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Park, J. S.; O’Brien, J.; Cai, C. J.; Morris, M. R.; Liang, P.; and Bernstein, M. S. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, 1–22.

Saha Roy, R.; Schlotthauer, J.; Hinze, C.; Foltyn, A.; Hahn, L.; and Kuech, F. 2025. Evidence contextualization and counterfactual attribution for conversational qa over heterogeneous data with rag systems. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining, 1040–1043.

Shinn, N.; Cassano, F.; Berman, E.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366.

Srivastava, S. S. 2026. Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents. arXiv preprint arXiv:2605.17641.

Wang, L.; Ma, C.; Feng, X.; Zhang, Z.; Yang, H.; Zhang, J.; Chen, Z.; Tang, J.; Chen, X.; Lin, Y.; et al. 2024a. A survey on large language model based autonomous agents. Frontiers ofComputer Science, 18(6): 186345.

Wang, P.; Li, Z.; Zhang, N.; Xu, Z.; Yao, Y.; Jiang, Y.; Xie, P.; Huang, F.; and Chen, H. 2024b. Wise: Rethinking the knowledge memory for lifelong model editing of large language models. Advances in Neural Information Processing Systems, 37: 53764–53797.

Wu, D.; Wang, H.; Yu, W.; Zhang, Y.; Chang, K.-W.; and Yu, D. 2024. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813.

Wu, Z.; Zhang, H.; Lin, F.; Xu, W.; Xu, X.; Chen, Y.; Zou, H. P.; Chen, S.; Zhang, W.; Liu, X.; et al. 2026. Gam: Hierarchical graph-based agentic memory for llm agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 34647–34664.

Xiong, Z.; Lin, Y.; Xie, W.; He, P.; Liu, Z.; Tang, J.; Lakkaraju, H.; and Xiang, Z. 2025. How memory management impacts llm agents: An empirical study of experiencefollowing behavior. arXiv preprint arXiv:2505.16067.

Xu, J. 2025. Memory Management and Contextual Consistency for Long-Running Low-Code Agents. arXiv preprint arXiv:2509.25250.

Yan, S.; Yang, X.; Huang, Z.; Nie, E.; Ding, Z.; Li, Z.; Ma, X.; Bi, J.; Kersting, K.; Pan, J. Z.; et al. 2026. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. In Proceedings ofthe

64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12805–12825.

Yu, Y.; Yao, L.; Xie, Y.; Tan, Q.; Feng, J.; Li, Y.; and Wu, L. 2026. Agentic memory: Learning unified long-term and short-term memory management for large language model agents. arXiv preprint arXiv:2601.01885.

Zhang, J.; Chen, K.; Ma, J.; Hu, Y.; He, L.; Zhang, Y.; Liu, J.; Yang, X.; Zhang, T.; and Jia, R. 2026a. Beyond Similarity: Trustworthy Memory Search for Personal AI Agents. arXiv preprint arXiv:2606.06054.

Zhang, J.; Zhang, C.; Chen, S.; Huang, Z.; Zheng, P.; Wang, Z.; Guo, P.; Mo, F.; Bae, S.-H.; Zou, J.; et al. 2026b. Lightweight llm agent memory with small language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12914–12929.

Zhang, Z.; Dai, Q.; Bo, X.; Ma, C.; Li, R.; Chen, X.; Zhu, J.; Dong, Z.; and Wen, J.-R. 2025. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6): 1–47.

## Appendix

Due to the strict page limit of the main paper, we provide supplementary theoretical analyses and experimental details in the appendix. Appendix A proves the redundant-harm blind spot theorem and analyzes why LOO fails under redundant harmful memories. Appendix B establishes the finite-sample guarantee of cooperative attribution estimation. Appendix C provides the theoretical guarantee for query-scoped verified clearance. Appendix D reports the complete experimental configurations, baseline settings, evaluation protocols, and additional analyses.

## Appendix A: Redundant-Harm Blind Spot

This appendix proves Theorem 1 and explains why a singlerecord Leave-One-Out (LOO) test can miss redundant harmful memories. Throughout Appendix A, the task step t and the retrieved context $\mathcal { M } _ { t }$ are fixed. For the theoretical analysis, let $d _ { t , i } = v _ { t } ( \mathcal { M } _ { t } ) - v _ { t } ( \mathcal { M } _ { t } \setminus \{ m _ { i } \} )$ denote the exact counterpart of the empirical LOO efect $d _ { t , i }$ in Equation (7).

## A.1 Proof of Theorem 1

Assume that $m _ { i }$ and $m _ { j }$ satisfy the redundant-harm model in Equation (8):

$$
\begin{array} { r l } & { v _ { t } ( S ) = } \\ & { u _ { t } ( S \setminus \{ m _ { i } , m _ { j } \} ) - \Delta _ { t } \mathbf { 1 } [ S \cap \{ m _ { i } , m _ { j } \} \neq \emptyset ] , \Delta _ { t } > 0 , } \end{array}\tag{23}
$$

where $u _ { t }$ does not depend on $m _ { i }$ or $m _ { j }$ , and both memories belong to $\mathcal { M } _ { t }$

Proof. Because both $m _ { i }$ and $m _ { j }$ are present in $\mathcal { M } _ { t } .$ the penalty in Equation (23) is active and

$$
v _ { t } ( \mathcal { M } _ { t } ) = u _ { t } ( \mathcal { M } _ { t } \setminus \{ m _ { i } , m _ { j } \} ) - \Delta _ { t } ,\tag{24}
$$

Removing $m _ { i }$ leaves $m _ { j }$ in the context, so the penalty remains:

$$
v _ { t } ( \mathcal { M } _ { t } \setminus \{ m _ { i } \} ) = u _ { t } ( \mathcal { M } _ { t } \setminus \{ m _ { i } , m _ { j } \} ) - \Delta _ { t } .\tag{25}
$$

Hence $d _ { t , i } = 0$ . By symmetry, $d _ { t , j } = 0$ . We next consider the cooperative contribution. Shapley-based valuation measures a player’s contribution by averaging its marginal efect over diferent coalitions (Ghorbani and Zou 2019; Jia et al. 2019b). Let π be a uniformly random permutation of $\mathcal { M } _ { t } ,$ and let $P _ { i } ( \pi )$ contain the memories appearing before $m _ { i } .$ . Define

$$
X _ { t , i } ( \pi ) = v _ { t } ( P _ { i } ( \pi ) \cup \{ m _ { i } \} ) - v _ { t } ( P _ { i } ( \pi ) ) .\tag{26}
$$

If m<sub>i</sub> appears before $m _ { j }$ , then $m _ { j } \notin P _ { i } ( \pi )$ . Adding m<sub>i</sub> activates the penalty for the first time, so ${ \dot { X } } _ { t , i } ( \pi ) = - \Delta _ { t } .$ If $m _ { j }$ appears before $m _ { i }$ , the penalty is already active and $X _ { t , i } ( \pi ) = 0$ . The two relative orders are equally likely, each with probability $1 / 2$ . Therefore, using Equation (10),

$$
\psi _ { t , i } = \mathbb { E } _ { \pi } [ X _ { t , i } ( \pi ) ] = - \frac { \Delta _ { t } } { 2 } .\tag{27}
$$

The argument gives $\psi _ { t , j } = - \Delta _ { t } / 2$ . Thus ${ d _ { t , i } } = { d _ { t , j } } = 0$ while $\psi _ { t , i } = \psi _ { t , j } = - \tilde { \Delta _ { t } } / 2$ , proving Theorem 1. □

## A.2 Why LOO Misses Redundant Harm

LOO and cooperative attribution evaluate a memory under diferent contexts. The LOO efect measures the performance change caused by removing one memory from the retrieved context. Specifically, for memory $m _ { i }$ , it evaluates

$$
d _ { t , i } = v _ { t } ( \mathcal { M } _ { t } ) - v _ { t } ( \mathcal { M } _ { t } \setminus \{ m _ { i } \} ) .\tag{28}
$$

Under Equation $( 8 )$ , removing $m _ { i }$ leaves the substitutable harmful memory $m _ { j }$ active. Therefore, the degradation term remains unchanged before and after the deletion:

$$
v _ { t } ( \mathcal { M } _ { t } ) = v _ { t } ( \mathcal { M } _ { t } \setminus \{ m _ { i } \} ) ,\tag{29}
$$

which directly leads to $d _ { t , i } = 0$ . Cooperative attribution instead evaluates the marginal contribution of $m _ { i }$ over diferent predecessor coalitions. Following the permutation formulation of Shapley-based valuation (Ghorbani and Zou 2019; Jia et al. 2019b), the contribution of $m _ { i }$ is obtained by averaging

$$
\Delta _ { t , i } ( \pi ) = v _ { t } ( P _ { i } ( \pi ) \cup \{ m _ { i } \} ) - v _ { t } ( P _ { i } ( \pi ) ) ,\tag{30}
$$

where $P _ { i } ( \pi )$ denotes the set of memories preceding $m _ { i }$ in permutation π. When $m _ { j }$ is not included in $P _ { i } ( \pi ) \bar { }$ , the marginal contribution of $m _ { i }$ reflects its harmful efect because no redundant memory provides the same evidence. When $m _ { j }$ is already included, the harmful evidence is already present and the additional contribution of $m _ { i }$ is masked by redundancy. Therefore, coalition-level marginal evaluation can reveal harmful contributions that remain invisible under the single-memory deletion test. The result should be interpreted narrowly. Theorem 1 only establishes that a zero LOO efect does not exclude harmful cooperative contribution under redundant harmful memories. It does not imply that every memory with a zero or small LOO efect is harmful. Accordingly, MeClear interprets the LOO profile using tolerance κ: $\widehat { d } _ { t , i } < - \kappa$ provides local evidence of harm, $| \widehat { d } _ { t , i } | \leq \kappa$ is treated as locally inconclusive, and $\widehat { d } _ { t , i } > \kappa$ provides local evidence of benefit. This local profile does not replace cooperative attribution; the operational harmful set remains determined by the sampled contribution in Equation (11) with harm tolerance τ.

## A.3 Extension to Multiple Redundant Memories

The same blind spot occurs for a redundant group with more than two memories.

Corollary A.1 (Multi-Memory Redundant Harm). Let $\mathcal { R } = \{ m _ { 1 } , . . . , m _ { r } \}$ with $r \geq 2 ,$ and suppose

$$
v _ { t } ( S ) = u _ { t } ( S \setminus \mathcal { R } ) - { \Delta _ { t } } \mathbf { 1 } [ S \cap \mathcal { R } \neq \emptyset ] , \qquad \Delta _ { t } > 0 ,\tag{31}
$$

where $u _ { t }$ does not depend on any memory in R. $\mathrm { I f } \mathcal { R } \subseteq \mathcal { M } _ { t }$ then every $m _ { i } \in \mathcal { R }$ satisfies $d _ { t , i } = 0$ and $\psi _ { t , i } = - \Delta _ { t } / r$

Proof. Since $r \geq 2$ , deleting one memory $m _ { i }$ leaves at least one member of R in the context. The penalty in Equation (31) therefore remains active, so $d _ { t , i } = 0$ . Along any permutation, the penalty $- \Delta _ { t }$ is introduced exactly once: when the first member of R appears. Under a uniformly random permutation, each member of R is first with probability $1 / r$ . Thus the marginal contribution of $m _ { i } \mathrm { i s } - \Delta _ { t }$ with probability $1 / r$ and zero otherwise, which gives $\psi _ { t , i } = - \Delta _ { t } / r$ □

Corollary A.1 shows that the blind spot is not limited to a pair of redundant memories. Under the stated substitution model, LOO assigns zero efect to every member of the redundant group, while the cooperative allocation distributes the total penalty $- \Delta _ { t }$ <sub>t</sub> across the group.

## Appendix B: Finite-Sample Cooperative Estimation

This appendix proves Proposition 1. The analysis conditions on the fixed empirical value function $\widehat { v } _ { t }$ and isolates the approximation error caused by sampling permutations.

B.1 Exact Shapley Value of the Empirical Game For the fixed empirical game, define

$$
\widetilde { \psi } _ { t , i } = \frac { 1 } { | \Pi _ { t } | } \sum _ { \pi \in \Pi _ { t } } \left[ \widehat { v } _ { t } ( P _ { i } ( \pi ) \cup \{ m _ { i } \} ) - \widehat { v } _ { t } ( P _ { i } ( \pi ) ) \right] .\tag{32}
$$

Conditional on $\widehat { v } _ { t } .$ , this quantity is fixed. The use of sampled permutations to approximate Shapley values follows the standard permutation view used in eficient Shapley estimation (Ghorbani and Zou 2019; Jia et al. 2019b). Equation (32) is equivalent to

$$
\sum _ { S \subseteq { \mathcal A } _ { t } \setminus \{ m _ { i } \} } ^ { { \mathcal D } _ { t , i } } \frac { | S | ! ( K - | S | - 1 ) ! } { K ! } \left[ \widehat { v } _ { t } ( S \cup \{ m _ { i } \} ) - \widehat { v } _ { t } ( S ) \right] .\tag{33}
$$

To see this, fix $S \subseteq { \mathcal { M } } _ { t } \setminus \{ m _ { i } \}$ with $| S | = s$ . There are $s ! ( K - s - 1 ) !$ permutations for which S is exactly the predecessor set of $m _ { i } \colon$ the s memories in S can appear before $m _ { i }$ in any order, and the remaining $K - s - 1$ memories can appear after $m _ { i }$ in any order. Dividing by the total number $K !$ of permutations gives the coeficient in Equation (33). The empirical Shapley values also satisfy the eficiency property

$$
\sum _ { i = 1 } ^ { K } \widetilde { \psi } _ { t , i } = \widehat { v } _ { t } ( \mathcal { M } _ { t } ) - \widehat { v } _ { t } ( \varnothing ) .\tag{34}
$$

Proof. Fix a permutation $\pi = ( m _ { \pi _ { 1 } } , \ldots , m _ { \pi _ { K } } )$ and define the prefix sets $S _ { 0 } ~ = ~ \emptyset$ and $S _ { r } ~ = ~ \{ m _ { \pi _ { 1 } } , . . . , m _ { \pi _ { r } } \}$ . The marginal contributions along this permutation telescope:

$$
\sum _ { r = 1 } ^ { K } [ \widehat { v } _ { t } ( S _ { r } ) - \widehat { v } _ { t } ( S _ { r - 1 } ) ] = \widehat { v } _ { t } ( M _ { t } ) - \widehat { v } _ { t } ( \emptyset ) .\tag{35}
$$

Averaging the left-hand side over all permutations gives $\textstyle \sum _ { i } { \tilde { \psi } } _ { t , i }$ , while the right-hand side is unchanged. This proves Equation (34). Replacing $\widehat { v } _ { t }$ with $v _ { t }$ gives the population identity stated after Equation (3). □

## B.2 Unbiasedness and Uniform Concentration

For a sampled permutation $\pi ^ { ( \ell ) }$ , define

$$
X _ { t , i } ^ { ( \ell ) } = \widehat { v } _ { t } \Big ( P _ { i } ( \pi ^ { ( \ell ) } ) \cup \{ m _ { i } \} \Big ) - \widehat { v } _ { t } \Big ( P _ { i } ( \pi ^ { ( \ell ) } ) \Big ) .\tag{36}
$$

Then Equation (11) can be written as $\begin{array} { r l } { \widehat { \psi } _ { t , i } ^ { ( L ) } } & { { } = } \end{array}$ $L ^ { - 1 } \sum _ { \ell = 1 } ^ { L } X _ { t , i } ^ { ( \ell ) }$ . For each fixed $i ,$ the variables

$X _ { t , i } ^ { ( 1 ) } , \ldots , X _ { t , i } ^ { ( L ) }$ are independent and identically distributed conditional on $\widehat { v } _ { t } .$ , because the permutations are sampled independently and uniformly. By Equation (32), $\mathbb { E } [ X _ { t , i } ^ { ( \ell ) } \mid \widehat { v } _ { t } ] =$ $\widetilde { \psi } _ { t , i } .$ Linearity of expectation therefore gives

$$
\mathbb { E } \left[ \widehat { \psi } _ { t , i } ^ { ( L ) } \mid \widehat { v } _ { t } \right] = \widetilde { \psi } _ { t , i } .\tag{37}
$$

Since the empirical game is fixed in Proposition 1, this is the unbiasedness statement in Equation (12). Because $\widehat { v } _ { t } ( S ) \in [ 0 , 1 ]$ for every coalition S, each sampled marginal satisfies $X _ { t , i } ^ { ( \ell ) } ~ \in ~ [ - 1 , 1 ]$ . Hoefding’s inequality therefore gives, for every fixed i and $\varepsilon > 0$

$$
\operatorname* { P r } ( | \widehat { \psi } _ { t , i } ^ { ( L ) } - \widetilde { \psi } _ { t , i } | \geq \varepsilon \ | \widehat { v } _ { t } ) \leq 2 \exp ( - \frac { L \varepsilon ^ { 2 } } { 2 } ) .\tag{38}
$$

Applying the union bound over the K memories yields

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { 1 \leq i \leq K } \left| \widehat { \psi } _ { t , i } ^ { ( L ) } - \widetilde { \psi } _ { t , i } \right| \geq \varepsilon \ \bigg | \ \widehat { v } _ { t } \right) \leq 2 K \exp \left( - \frac { L \varepsilon ^ { 2 } } { 2 } \right) ,\tag{39}
$$

which proves Equation (13). The estimates for diferent memories need not be independent because one sampled permutation contributes to several memories. This does not afect the proof: Hoefding’s inequality is applied separately to the L independent permutation samples for each fixed memory, and the final union bound does not require independence across memories.

## B.3 Consequences for Sampling, Thresholding, and Ranking

Equation (39) gives a suficient sample size for uniform accuracy. For any $\varepsilon > 0$ and $\delta \in ( 0 , 1 )$ , if

$$
L \geq { \frac { 2 } { \varepsilon ^ { 2 } } } \log \left( { \frac { 2 K } { \delta } } \right) ,\tag{40}
$$

then, with probability at least $1 - \delta ,$

$$
\operatorname* { m a x } _ { 1 \leq i \leq K } \left| \widehat { \psi } _ { t , i } ^ { ( L ) } - \widetilde { \psi } _ { t , i } \right| < \varepsilon .\tag{41}
$$

This is a suficient worst-case bound based only on $\widehat { v } _ { t } ( S ) \in$ $[ 0 , 1 ] ;$ ; it is not a claim that the resulting value of L is necessary or optimal in practice. We next consider the threshold in Equation (14). Define the exact harmful set of the empirical game as $\mathcal { \widetilde { H } } _ { t } = \{ m _ { i } \in \mathcal { M } _ { t } \ | \ \widetilde { \psi } _ { t , i } < - \tau \}$ , and assume that no exact empirical contribution lies on the threshold. Let $\begin{array} { r } { \gamma _ { t } = \operatorname* { m i n } _ { i } | \widetilde { \psi } _ { t , i } + \tau | > 0 } \end{array}$ . Setting $\varepsilon = \gamma _ { t }$ in Equation (40) gives

$$
L \geq \frac { 2 } { \gamma _ { t } ^ { 2 } } \log \left( \frac { 2 K } { \delta } \right) .\tag{42}
$$

Under this condition, $\widehat { \mathcal { H } } _ { t } = \widetilde { \mathcal { H } } _ { t }$ with probability at least $1 - \delta$ , because no estimate can cross the threshold −τ when its error is smaller than $\gamma _ { t }$ . The same argument controls the ordering in Equation (16). If $\tilde { \psi } _ { t , j } - \tilde { \psi } _ { t , i } > 2 \varepsilon$ and the uniform estimation error is smaller than $\varepsilon ,$ then $\widehat { \psi } _ { t , i } ^ { ( L ) } < \widehat { \psi } _ { t , j } ^ { ( L ) }$ Thus, pairs of memories separated by more than twice the estimation error keep the same order. These results apply to the fixed empirical game only. Proposition 1 controls the diference between $\widehat { \psi } _ { t , i } ^ { ( \overline { { L } } ) }$ and $\ddot { \psi } _ { t , i }$ . It does not by itself control the diference between $\widehat { \psi } _ { t , i } ^ { ( L ) }$ and the population contribution $\psi _ { t , i }$ . Such a statement would require an additional assumption relating $\widehat { v } _ { t }$ to $v _ { t }$ . The analysis above concerns the approximation error of cooperative attribution and therefore depends on the harm tolerance $\tau .$ . The LOO tolerance κ is used only to interpret the preceding local counterfactual profile and does not alter the finite-sample guarantee for the sampled Shapley estimator.

Algorithm 1: MeClear: Cooperative Attribution and Verified   
Clearance   
Input: Query $q _ { t } .$ , memory bank $B _ { t }$ , retriever $\rho ,$ evaluator $\widehat { v } _ { t } ,$   
recovery predicate $\nu _ { t }$   
Parameters: Retrieval budget $K ,$ permutation budget $L ,$   
screening tolerance $\kappa ,$ harm tolerance $\tau$   
Output: Cleared context $\widetilde { \mathcal { M } } _ { t }$   
1: Retrieve and freeze $\mathcal { M } _ { t }$ using Eq. (6)   
2: Evaluate and cache $\widehat { v } _ { t } ( \mathcal { M } _ { t } )$   
3: Compute LOO efects $\{ \widehat { d } _ { t , i } \}$ using $\mathrm { E q . } ( 7 )$   
4: Classify the local LOO efects using tolerance κ   
5: Estimate $\{ \widehat { \psi } _ { t , i } ^ { ( L ) } \}$ using Eq. (11)   
6: Construct $\widehat { \mathcal { H } } _ { t }$ using Eq. (14)   
7: Order $\widehat { \mathcal { H } } _ { t }$ using Eq. (16)   
8: Set $h _ { t } \gets | \widehat { \mathcal { H } } _ { t } |$ and $\mathcal { I } _ { t }  \{ 0 \}$   
9: for $j = 1$ to $h _ { t }$ do   
10: Construct $\mathcal { C } _ { t , j }$ using Eq. (17)   
11: Compute $\widehat { g } _ { t } ( \boldsymbol { j } )$ using Eq. (18)   
12: if $\widehat { g } _ { t } ( j ) > 0$ and $\nu _ { t } ( \mathbf { \bar { \mathcal { M } } } _ { t } \mathbf { \bar { \Psi } } \backslash \mathcal { C } _ { t , j } ) = 1$ then   
13: $\mathcal { T } _ { t }  \mathcal { T } _ { t } \cup \{ j \}$   
14: end if   
15: end for   
16: Select $j _ { t }$ using Eq. (20)   
17: $\widetilde { \mathcal { M } } _ { t } \gets \mathcal { M } _ { t } \setminus \mathcal { C } _ { t , \widehat { j } _ { t } }$   
18: return $\widetilde { \mathcal { M } } _ { t }$

## Appendix C: MeClear Procedure and Clearance Guarantee

This appendix gives the full MeClear procedure using the definitions in the main paper and proves Theorem 2. All counterfactual evaluations are performed on the same frozen context $\mathcal { M } _ { t } .$ , as specified in the main method.

## C.1 Proof of Theorem 2

Algorithm 1 follows the four stages described in the main paper: retrieval, local screening, cooperative attribution, and verified clearance. The LOO profile is interpreted using screening tolerance κ: strongly negative local efects provide direct evidence of harm, near-zero efects remain inconclusive, and strongly positive efects indicate local benefit. The final operational harmful set is determined by cooperative contributions with harm tolerance $\tau .$ The interaction score in Equation (15) is used only for structural analysis and does not afect harmful-set construction or clearance decisions. Repeated coalition values may be cached without changing the sampled estimator in Equation (11). Recall that Equation (19) always includes the baseline index 0, while every positive index must have both positive empirical gain and verified recovery. Equation (20) selects the maximum-gain admissible index and uses the smallest index to break ties.

Proof. Since $0 \ \in \ \mathcal { T } _ { t }$ and $\mathcal { C } _ { t , 0 } ~ = ~ \emptyset$ , Equation (18) gives $\widehat { g } _ { t } ( 0 ) = 0 .$ Therefore m $\arg _ { \boldsymbol { \xi } \in \mathcal { T } _ { \boldsymbol { t } } } \widehat { g } _ { \boldsymbol { t } } ( \boldsymbol { j } ) \ \geq \ 0$ . Because $\widehat { j } _ { t }$ is selected from this set of maximizers, $\widehat { g } _ { t } ( \widehat { j } _ { t } ) \geq 0$ . With $\widetilde { \mathcal { M } } _ { t } =$ $\mathcal { M } _ { t } \backslash \mathcal { C } _ { t , \widehat { j } _ { t } }$ , we obtain

$$
\widehat { v } _ { t } ( \widetilde { \mathcal { M } } _ { t } ) \geq \widehat { v } _ { t } ( \mathcal { M } _ { t } ) ,\tag{43}
$$

which proves Equation (21). Now suppose $\widehat { j } _ { t } \ > \ 0$ . Since $\widehat { j } _ { t } \in \mathcal I _ { t }$ , the definition of $\mathcal { T } _ { t }$ requires $\nu _ { t } ( \mathcal { M } _ { t } \setminus \mathcal { C } _ { t , \widehat { j } _ { t } } ) = 1$ Hence $\nu _ { t } ( \widetilde { \mathcal { M } } _ { t } ) = 1$ , which proves Equation (22). Finally, let

$$
\mathcal { T } _ { t } ^ { \operatorname* { m a x } } = \mathop { \arg \operatorname* { m a x } } _ { j \in \mathcal { I } _ { t } } \widehat { g } _ { t } ( j ) .\tag{44}
$$

By Equation $( 2 0 ) , \widehat { j } _ { t } = \operatorname* { m i n } \mathcal { I } _ { t } ^ { \operatorname* { m a x } }$ . The nested construction in Equation (17) satisfies $\left. \dot { \boldsymbol { C } } _ { t , j } \right. = j$ . Thus, for every $j \in \mathcal { I } _ { t } ^ { \operatorname* { m a x } } , | \mathcal { C } _ { t , \widehat { j } _ { t } } | = \widehat { j } _ { t } \leq j = | \mathcal { C } _ { t , j } |$ . Therefore $\mathcal { C } _ { t , \widehat { j } _ { t } }$ has minimum cardinality among the admissible candidates that attain the maximum empirical gain. □

The baseline also gives a simple fallback. If no positive index satisfies the gain and recovery conditions, then $\mathcal { T } _ { t } =$ {0} and MeClear returns the unchanged context. If at least one positive admissible candidate exists, its gain is strictly larger than the baseline gain, so the selected index is positive and the returned context satisfies the recovery predicate.

## C.2 Scope of the Guarantee and Evaluator Error

The ideal objective in Equation (5) considers all subsets of the harmful set, whereas Theorem 2 is established over the nested family in Equation (17) with $h _ { t } + 1$ candidates. It guarantees empirical non-degradation, verified recovery for positive admissible selections, and minimum cardinality among maximum-gain candidates within this family, but does not imply global optimality over all subsets of $\widehat { \mathcal { H } } _ { t }$ . The guarantee is conditional on the operational harmful set produced by the preceding attribution stage and does not assert that $\widehat { \mathcal { H } } _ { t }$ equals the unknown exact harmful set $\mathcal { H } _ { t }$ . Global optimality would additionally require that a minimum-cardinality unrestricted maximizer be represented in the nested family and, when nonempty, satisfy the recovery predicate. Suppose that, for the baseline and every candidate evaluated by MeClear,

$$
| \widehat { v } _ { t } ( S ) - v _ { t } ( S ) | \leq \eta _ { t }\tag{45}
$$

for some $\eta _ { t } \geq 0$ . Let $g _ { t } ( j ) = v _ { t } ( \mathcal { M } _ { t } \backslash \mathcal { C } _ { t , j } ) - v _ { t } ( \mathcal { M } _ { t } )$ . Then the triangle inequality gives

$$
| \widehat { g } _ { t } ( j ) - g _ { t } ( j ) | \leq 2 \eta _ { t } .\tag{46}
$$

Since Theorem 2 gives $\widehat { g } _ { t } ( \widehat { j } _ { t } ) \geq 0$ , it follows that $g _ { t } ( \widehat { j } _ { t } ) \geq$ $- 2 \eta _ { t }$ . Moreover, if $\widehat { g } _ { t } ( \widehat { j } _ { t } ) ~ > ~ 2 \eta _ { t }$ , then $g _ { t } ( \widehat { j } _ { t } ) > 0$ . This last statement is conditional on Equation (45). It does not turn the empirical guarantee into an unconditional population guarantee; it only shows how a known uniform evaluator error transfers to the selected gain.

<table><tr><td>Setting</td><td>Configuration</td></tr><tr><td>Dataset</td><td>LoCoMo, categories 1–4</td></tr><tr><td>Conversations</td><td>10</td></tr><tr><td>Clean queries</td><td>368</td></tr><tr><td>Verified cases</td><td>745</td></tr><tr><td>Fault records</td><td>1,115</td></tr><tr><td>Direct conflicts</td><td>375</td></tr><tr><td>Redundant conflicts</td><td>285</td></tr><tr><td>Joint interactions</td><td>85</td></tr><tr><td>Memory framework</td><td>Mem0 2.0.12 + Qdrant</td></tr><tr><td>Memory records</td><td>11,302</td></tr><tr><td>Retrieval budget K</td><td>5</td></tr></table>

Table 2: Dataset statistics and evaluation configuration.

Panel A: Cohort construction and paired-case funnel
<table><tr><td>Stage</td><td>Unit</td><td>N</td><td>Retention</td></tr><tr><td>LoCoMo QA</td><td>QA</td><td>1,540</td><td></td></tr><tr><td>Stable clean-correct QA</td><td>QA</td><td>624</td><td>40.5%</td></tr><tr><td>Injection proposals</td><td>Proposal</td><td>7,986</td><td></td></tr><tr><td>Causal-valid proposals</td><td>Proposal</td><td>1,178</td><td>14.8%</td></tr><tr><td>Selected cases</td><td>Case</td><td>747</td><td>63.4%</td></tr><tr><td>Final paired cases</td><td>Paired case</td><td>745</td><td>99.7%</td></tr></table>

Panel B: Final cohort by fault and interaction
<table><tr><td>Fault type</td><td>Direct</td><td>Redundant</td><td>Joint</td><td>All</td></tr><tr><td>Explicit conflict</td><td>173</td><td>135</td><td>36</td><td>344</td></tr><tr><td>Precise factual</td><td>161</td><td>122</td><td>44</td><td>327</td></tr><tr><td>Precise temporal</td><td>41</td><td>28</td><td>5</td><td>74</td></tr><tr><td>All faults</td><td>375</td><td>285</td><td>85</td><td>745</td></tr></table>

Table 3: Cohort construction and final paired-case coverage. Panel A summarizes the main data-selection milestones, and Panel B reports the final 745 cases by interaction structure and fault type.

## Appendix D: Experimental Details and Parameter Settings

Due to the strict page limit of the main paper, we provide additional experimental configuration, parameter settings, and supplementary analyses in this appendix.

## D.1 Experimental Setup

We evaluate MeClear on LoCoMo using Kimi-k2.6 as task agent and Qwen3.6-Flash as judge evaluator. The evaluation is constructed from 368 clean queries across ten conversations and contains 745 causally verified cases with 1,115 fault records. The cohort includes 375 direct conflicts with $| M | = 1$ , 285 redundant conflicts with $| M | = 2 ,$ , and 85 joint interactions with $| M | = 2 .$ . Two of the 747 initially selected cases were excluded before paired evaluation because complete attribution outputs could not be obtained. These exclusions resulted from evaluation failures rather than method scores, leaving 745 cases with valid outputs for compared methods under the same evaluation protocol. The interactionby-fault composition contains 344 explicit-conflict cases, 327 precise-factual cases, and 74 precise-temporal cases. Among them, the direct-conflict group contains 173 explicit, 161 factual, and 41 temporal cases; the redundant group contains 135 explicit, 122 factual, and 28 temporal cases; and the joint-interaction group contains 36 explicit, 44 factual, and 5 temporal cases.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>MeClear retrieval budget K</td><td>5</td></tr><tr><td>MeClear permutation budget L</td><td>16</td></tr><tr><td>MeClear checkpoints</td><td>4,8,16</td></tr><tr><td>MeClear screening tolerance κ</td><td>0.05</td></tr><tr><td>MeClear harm tolerance τ</td><td>0.05</td></tr><tr><td>Task-value range</td><td>[0, 1]</td></tr><tr><td>Task agent</td><td>Kimi-k2.6</td></tr><tr><td>Judge evaluator</td><td>Qwen3.6-Flash</td></tr><tr><td>Memory construction</td><td>Moonshot-v1-32k</td></tr><tr><td>Fault generation</td><td> $\mathrm { Q w e n } 3 . 7 – \mathrm { P l u s }$ </td></tr><tr><td>LOO threshold</td><td> $\widehat { d } _ { t , i } < - 0 . 0 5$ </td></tr><tr><td>ContextCite budget / α</td><td> $3 2 / 0 . 0 1$ </td></tr><tr><td>ProxySPEX budget / order</td><td>32 /2</td></tr><tr><td>Direct LLM</td><td>Qwen3.5-Plus</td></tr><tr><td>Direct LLM threshold</td><td>≥ 0.5</td></tr></table>

Table 4: Main method, model, and baseline settings.

Memories are constructed with Mem0 2.0.12 and stored in Qdrant, resulting in 11,302 memory records. For each query, the memory system uses the retrieval budget K = 5 specified in the main paper. The retrieved context is frozen before counterfactual attribution, and all coalition evaluations and clearance operations are performed without re-running retrieval, consistent with Equation (6). Controlled faults are retained only after behavioral verification. The clean context must remain correct in both trials, while a direct conflict must make the corrupted context incorrect in both trials. For redundant conflicts, the corrupted context must remain incorrect when either injected fault is retained alone, ensuring that each record can independently sustain the failure and mask the local efect of the other. For joint interactions, each injected record must remain harmless when evaluated alone, whereas their combination must induce task failure. These conditions provide behaviorally verified target sets for evaluating direct, redundant, and jointly expressed harmfulmemory efects.

Figure 8 reports performance across the interaction-byfault combinations. MeClear shows its advantage under redundant conflicts, where local deletion is most susceptible to masking. Across the three redundant-fault categories, MeClear achieves 81.9–84.8% Recall@|M| compared with only 10.7–16.1% for LOO. Exact Set Match increases from

![](images/d76c9b8436b69acba2ef773b404f713b798232025515a81f9a9794a6fd92ac4d.jpg)  
Figure 8: Interaction-by-fault performance heatmap for five memory-clearance methods. Panels show target-micro Recall@ $| M | ,$ case-macro Exact Set Match, and Binary Task Recovery across nine interaction-by-fault cells. Cell annotations report percentages and case counts; the MeClear column is outlined, and cells with $n < 1 0$ are hatched. All panels share a 0–100% scale with consistent visual encoding for comparison across fault structures.

![](images/79ac1d99d7dd01a3183a87914467230c49a20cfd2bbe37c37479cfa376217e9c.jpg)  
Figure 9: MeClear sensitivity to permutation budgets $L = 4 .$ 8, and 16 on 745 cases. Results report Recall@|M|, Complete Set Recall, Exact Set Match, and extra-background selection with 95% two-level cluster-bootstrap intervals.

1.6–3.6% for LOO to 45.2–53.6% for MeClear, while Binary Task Recovery increases from 5.9–7.1% to 63.7–78.6%. MeClear also attains the highest observed Recall@|M across the joint-interaction cells and achieves or ties the highest Recovery in most interaction-by-fault cells. The joint precise-temporal cell contains only five cases and is therefore interpreted descriptively rather than as a stable subgroup estimate. Overall, the structural breakdown supports the role of coalition-aware attribution when harmful evidence is redundant or jointly expressed.

![](images/e17a08b251ae5b21c24d627398d5d6b788b400f8a1c1923b4540b449e51c5ccb.jpg)  
Figure 10: Paired diferences between MeClear and each comparator on the common 745-case cohort. Points show MeClear-minus-comparator estimates for Recall@|M|, Complete Set Recall, Exact Set Match, and Binary Task Recovery; horizontal segments show 95% two-level clusterbootstrap intervals. The vertical line marks zero, and the right-side W/L counts report case-level positive and negative diferences.

## D.2 Parameter and Baseline Settings

MeClear follows the main-paper configuration with retrieval budget $K = 5 ,$ , permutation budget $L = 1 6 .$ , and tolerance settings $\kappa = \tau = 0 . 0 5$ . The LOO profile uses κ to identify locally informative or inconclusive memories, while cooperative contributions are estimated using Equation (11) and thresholded by τ in Equation (14). The task value is normalized to [0, 1] as assumed in Proposition 1. All primary results use $L = 1 6 ,$ , with $L \in \{ 4 , 8 , 1 6 \}$ evaluated for sensitivity analysis. ContextCite and ProxySPEX use sampling budget $B \stackrel { . } { = } 3 2$ , with $\alpha = 0 . 0 1$ for ContextCite and interaction order two for ProxySPEX. The direct LLM baseline uses Qwen3.5-Plus with threshold 0.5, while LOO uses $\widehat { d } _ { t , i } < - \kappa = - 0 . 0 5$ . Memory construction and fault generation use Moonshot-v1-32k and Qwen3.7-Plus, respectively. All paired comparisons use identical frozen retrieved contexts. Figure 9 shows that larger permutation budgets improve attribution quality. Recal $\dot { | \varrho | \cal M | }$ increases from 67.6% at $L = 4 \ : \mathrm { t o } \ : 8 5 . 9 \%$ at $L = 1 6 ,$ , Complete Set Recall increases from 68.1% to 86.7%, and Exact Set Match improves from 37.3% to 47.0%. Extra-background selection remains nonmonotonic (0.581, 0.685, and 0.623 for $L = 4 , 8 , 1 6 )$ , indicating that the improvement is not caused by excessive deletion. Therefore, $L = 1 6$ is adopted for the main experiments. All recovery evaluations use two independent Answer/Judge trials with fixed random seeds.

![](images/0fbc70e7d5d59f8d74470e79707e2c1443d085df08c3bbabd9861634843efb68.jpg)  
Figure 11: Risk frontier of five methods on the common cohort. The x-axis shows residual-error risk, defined as one minus Complete Set Recall, the y-axis shows the average number of extra background memories selected per case, and bubble size indicates Binary Task Recovery. Lower values are preferred on both axes.

## D.3 Metrics and Supplementary Evaluation

Following the main paper, we evaluate MeClear using four metrics: Target Recall at |M|, Complete Set Recall, Exact Set Match, and Binary Task Recovery. For a case with verified fault set F and $| F | = m$ , let $r _ { 1 } , \ldots , r _ { m }$ denote the top-m ranked memories. Recall@|M| is defined as

$$
\mathrm { T R @ } | M | = \frac { | \{ r _ { 1 } , \ldots , r _ { m } \} \cap F | } { | F | } .\tag{47}
$$

The overall Recall@|M| is target-micro averaged over the 1,115 verified fault records. Complete Set Recall measures whether all verified faults are selected, while Exact Set Match additionally requires no background selection. Binary Task Recovery counts only cases where both independent postclearance trials succeed. These metrics respectively evaluate ranking quality, harmful-set completeness, selection precision, and task recovery.

![](images/a3948d565a01eb9fb9fe385fbdb086d965571fd435064e90a4abb343b1c51e3a.jpg)  
Figure 12: Target-memory rank ECDF over 1,115 injected targets. Curves show cumulative target coverage within rank k, with summary rates reported at $k = 1$ , 2, and 3.

All comparisons use the same paired cohort of 745 cases to ensure that performance diferences are attributable to method behavior rather than changes in evaluation samples. We report percentile 95% confidence intervals using 2,000 two-level cluster-bootstrap replicates, with conversations and query clusters resampled hierarchically to preserve dependencies among cases from the same source. Paired diferences are computed before resampling. Figure 10 shows that MeClear improves over LOO by 47.6, 22.6, 12.8, and 25.5 percentage points in Recall@|M|, Complete Set Recall, Exact Set Match, and Recovery, respectively. Compared with ProxySPEX, MeClear improves Recall@|M|, Complete Set Recall, and Recovery by 5.0, 9.9, and 5.0 points, while ProxySPEX achieves higher Exact Set Match by 7.2 points. MeClear also improves Recovery over Qwen3.5-Plus by 38.7 points. ContextCite remains the closest comparator, with intervals overlapping zero across metrics. MeClear primarily improves harmful-memory coverage and recovery by leveraging cooperative attribution to identify interacting harmful evidence.

Figure 11 evaluates the trade-of between incomplete harmful-set removal and unnecessary context modification. MeClear achieves the lowest residual-error risk of 13.3% and highest Binary Task Recovery of 82.3%, while selecting 0.623 extra background memories per case. ProxySPEX selects fewer extra memories at 0.408 per case and obtains higher Exact Set Match of 54.2%, but sufers higher residual risk of 23.2% and lower Recovery of 77.3%. ContextCite achieves comparable Recovery of 81.5% with 0.576 extra selections, but retains higher residual risk of 18.8%. LOO and Qwen3.5-Plus show larger trade-ofs, with residual risk of 35.8% and extra selection of 0.934, respectively. Figure 12 further shows that MeClear provides stronger multirank concentration of harmful memories, reaching 88.1% and 93.6% target coverage at ranks 2 and 3. Although ContextCite slightly exceeds MeClear at rank 1 with 61.1% versus 60.5%, MeClear surpasses it at higher ranks. It is important for redundant and joint interactions, where efective clearance requires identifying multiple harmful memories rather than a single salient record.