# Behavioral Probes for Information Flow in LLM Swarms

Junhui Chen<sup>1∗</sup> Yu Zhang<sup>1∗</sup> Lantian Li<sup>1†</sup> Dong Wang<sup>2</sup> <sup>1</sup>School of Artificial Intelligence,

Beijing University of Posts and Telecommunications, China <sup>2</sup>Center for Speech and Language Technologies, BNRist, Tsinghua University, China {rinkovo,rayleighz,lilt}@bupt.edu.cn wangdong99@mails.tsinghua.edu.cn § github.com/HaRry-qaq/Swarmprobe

## Abstract

Most existing approaches to LLM swarm optimization are utility-driven, focusing on final task utility while leaving the processing, integration, and reliance of information within individual interactions largely implicit. To address this, we study LLM swarm interactions from an information-flow-based view and introduce behavioral probes to quantify information flow patterns. Specifically, we repurpose External Context Score (ECS) and Parametric Knowledge Score (PKS)—originally designed for hallucination analysis—to measure a node’s external-context dependence and parametric-knowledge dependence, respectively. Across controlled and realistic swarm settings, we observe consistent patterns: ECS increases with input relevance and distinguishes useful multi-source contributions, while PKS decreases as informative context accumulates. Building on this heuristic interpretation, we demonstrate how these probes can complement utility-driven structure optimization in two controlled applications. Probe-guided node pruning reduces redundant communication and token consumption, and probe-guided reasoning-path search accelerates convergence compared to utility-only baselines. Our empirical results suggest that behavioral probes can make the internal information flow of LLM swarms observable and serve as effective auxiliary signals for structure optimization.

## 1 Introduction

Large language model (LLM) swarms have demonstrated strong collaborative reasoning capabilities, with interaction structures evolving from static, hand-designed topologies to learnable directed acyclic graphs (DAGs) [1–10].

Existing methods for optimizing LLM swarm structures remain largely utility-driven: edge weight are learned from final task utility [11–13], global rewards are redistributed through Shapley values or heuristic decomposition [14–17], and recent variants incorporate external critiques or explicit topology optimization [18–24]. Some methods introduce LLM confidence as intermediate utility signals [25] or maintain explicit context states [26]. However, they still do not directly characterize the swarm’s internal information flow. In particular, it remains unclear how information is routed, integrated, and relied upon across interacting nodes. Thus, individual interactions remain entangled with final task utility, leaving the swarm’s internal information flow largely implicit [27].

In this work, we study LLM swarm interactions from an information-flow-based view. Instead of evaluating swarm structures only through final task utility, we examine the information-processing behavior behind individual interactions. To make the information flow formed by these interactions observable, we instantiate this view with behavioral probes. Concretely, we use Parametric Knowledge Score (PKS) and External Context Score (ECS), originally developed for single-model hallucination analysis [28], as one concrete but non-unique instantiation of such behavioral probes. This choice is motivated by their relevance to two behavioral aspects of information processing: external-context dependence and parametric-knowledge dependence. Rather than treating this distinction as a formal mechanism, we use it as a heuristic interpretation for examining information flow across interacting nodes.

Across controlled and realistic swarm settings, we observe systematic patterns in LLM swarm interactions: ECS rises with input relevance, segmented ECS distinguishes contributions from multiple upstream sources, while PKS decreases as informative context accumulates and reasoning converges. Together, these patterns suggest that behavioral probes can make the information flow formed by individual interactions observable.

These patterns support a heuristic interpretation derived from the observed behaviors: ECS approximates external-context dependence, while PKS approximates parametric-knowledge dependence. This interpretation further suggests two ways to combine behavioral probes with utility signals for structure optimization: ECS–PKS behavioral profiles can help identify and prune nodes with abnormal behavior, while PKS-based convergence signals can guide reasoning-path search toward convergent reasoning paths. It is worth noting that this interpretation is used to examine information flow across interacting nodes, rather than to provide a formal decomposition of generation.

Building on this heuristic interpretation, we benchmark behavioral probes in two controlled structureoptimization settings. In node pruning, ECS–PKS behavioral profiles reduce redundant communication and token consumption relative to utility-only baselines. In reasoning-path search, probe-based signals consistently identify convergent reasoning paths across three initialization conditions. In both settings, behavioral probes are combined with utility signals rather than used as standalone objectives. In the topology-optimization experiments, metric collection is only needed during diagnosis or training, while deployment remains standard LLM inference. Together, these results support the following contributions:

• We introduce an information-flow-based view for studying LLM swarm interactions, and repurpose ECS and PKS as behavioral probes under this view. Across controlled and realistic swarm settings, we show that they remain informative for making the information flow formed by individual interactions observable.

• We develop a heuristic interpretation of these probes for swarm structure optimization: ECS approximates external-context dependence, while PKS approximates parametric-knowledge dependence. Based on this interpretation, we derive two probe-guided ways to complement utility-driven optimization: using ECS–PKS behavioral profiles to identify nodes with abnormal behavior, and using PKS-based convergence signals to guide reasoning-path search.

• We benchmark the probe-guided utility strategies in two controlled structure-optimization settings. Together, these results suggest that behavioral probes can guide utility signals in structure optimization: ECS–PKS behavioral profiles improve redundant-node pruning with substantially lower token consumption than utility-only baselines, while PKS-based signals identify convergent reasoning paths and accelerate the growth of their path weights.

## 2 Related Work

LLM multi-agent systems. LLM-based multi-agent systems have evolved from static, hand-designed topologies [29–35] toward learnable DAGs [11, 17, 36–39]. Yet their optimization remains largely outcome-driven, leaving internal information-flow dynamics undercharacterized.

Interaction-level optimization in LLM swarms. A key step in optimizing LLM swarm structures is to assign learning signals to individual interactions. Existing methods remain largely utility-driven: edge, role, or structural weights are learned from final task utility [11, 17], global rewards are redistributed through Shapley values or heuristic rules [14–16, 40], and recent variants incorporate external critiques or explicit topology optimization [18–24]. Process-reward decomposition and context-state modeling introduce intermediate supervision signals [25, 26], but they still do not directly characterize the swarm’s internal information flow. In particular, they leave open how information is processed within individual interactions. In contrast, we use ECS and PKS as behavioral probes under an information-flow-based view, making the information flow formed by individual interactions observable.

Metrics for LLM behaviors. Recent work characterizes LLM behaviors related to parametric and contextual knowledge through discrete knowledge-state modeling [41, 42], attribution-based frameworks that distinguish external context from internal parametric knowledge [28, 43–45], and hallucination detection via internal signals without explicit attribution [46–50]. We adopt PKS and ECS from [28] and repurpose them from single-model hallucination analysis to LLM swarm interactions. This choice is not unique: other behavioral or attribution metrics could also instantiate an information-flow-based view. We use ECS and PKS because they are relevant to two behavioral aspects of information processing: external-context dependence and parametric-knowledge dependence.

## 3 Exploratory Analysis of Behavioral Probes in LLM Swarm Structures

This section investigates whether ECS and PKS, repurposed from single-model hallucination analysis [28], can remain informative for studying information flow in LLM swarms. Unlike the original Re-DeEP setting, which focuses on whether a single-model retrieval-augmented answer is hallucinationprone, we ask whether the same signals can reveal how information is processed within individual interactions and across interacting nodes in LLM swarms. We conduct experiments across controlled node-level, multi-source aggregation, and chain settings, as well as a realistic tool-augmented GAIA pipeline. Together, these settings provide the empirical foundation for interpreting ECS and PKS as behavioral probes. The illustrated results in the controlled experiments are obtained using Qwen2.5-32B-Instruct [51]; consistent findings are also observed with LLaMA3-70B-Instruct [52].

## 3.1 Node-Level Response Patterns: Sensitivity to Input Relevance

Before studying information flow formed by individual interactions in LLM swarms, we first examine whether ECS and PKS are sensitive to input relevance at a single node. This calibration establishes a single-node reference point for later analyses of interactions between nodes.

We design a controlled kinship-question task about public figures, using the open-source data from Berglund et al. [53] as the basis for constructing the questions. Their family relations are unlikely to be directly memorized by the model, which makes the task suitable for testing sensitivity to input relevance. For each question, we use four context conditions. These conditions range from irrelevant context to direct answer evidence: bubble is unrelated filler context, rel is person-related background that does not contain family-relation evidence, oracle gives inverse relational evidence sufficient for reverse inference, and answer directly states the target relation. This design separates superficial topical relatedness from information that is actually useful for answering. For each question–context pair, we generate one response and compute sequence-level ECS and PKS; detailed dataset construction is provided in Appendix C.1.

As shown in Figure 1, ECS and PKS show systematic response patterns at the single-node level. ECS increases with input relevance, separating answer-supporting conditions (oracle, answer) from non-answer-supporting ones (bubble, rel). It is lowest for unrelated filler context and highest when the answer is directly provided. Quantitatively, adjacent relevance levels show consistently higher ECS (mean adjacent Cliff’s δ = 0.624, 95% CI [0.610, 0.640]). ECS also distinguishes oracle from answer, suggesting sensitivity beyond a coarse relevant/irrelevant split.

PKS follows the opposite pattern. It is lowest when the provided context can support the answer, especially under answer and oracle, and becomes higher and more dispersed when the context is unrelated or only provides biographical background. The adjacent-pair trend is weaker but negative in the expected direction (mean adjacent Cliff’s δ = −0.157, 95% CI [-0.170, -0.143]).

Together, these results provide a single-node calibration: ECS is sensitive to useful external context, while PKS decreases when the answer can be supported by the provided context. Full effect sizes and statistical tests are provided in Appendix C.1.1. Additional decoding and probe-configuration analyses and comparisons with simpler signals are reported in Appendix H.

![](images/bebf346044a1356aeb7efc395e603ad203616a48425a9be9cb6abeb915e44479.jpg)  
(a) ECS

![](images/e98c4a3d59c75301f3debb0f5471cc598f9cef40998a8d224d9726c7de856ffd.jpg)  
(b) PKS  
Figure 1: Node-level response patterns under four context conditions: irrelevant (bubble), related but unhelpful (rel), oracle denotes inverse relational evidence that is sufficiently informative for reverse inference, and answer denotes context directly containing the answer. Curves and translucent bars show score densities, dashed vertical lines mark group means, and the annotation boxes report mean and standard deviation. Overall, ECS increases with context relevance, while PKS decreases when the answer can be inferred from the provided context.

## 3.2 Behavioral Probes in Structured Swarm Interactions

After the single-node calibration, we next examine whether ECS and PKS remain informative in structured interactions between nodes. We focus on two settings that reflect different forms of information flow in LLM swarms: multi-source aggregation, where a downstream node needs to identify and rely on useful information from multiple upstream sources, and chain reasoning, where informative context should accumulate across sequential steps.

## 3.2.1 Multi-Source Aggregation

This experiment tests whether ECS can distinguish how a downstream node relies on different upstream sources when useful evidence is mixed with less useful context.

In LLM swarms, a downstream node often receives concatenated inputs from multiple upstream sources. The key question is therefore not only whether the overall context is useful, but also which upstream segments contribute useful information. Recent work studies related attribution problems with post-hoc methods [54–56]. Here, we compute segment-level ECS, denoted as seg-ECS, as a behavioral probe for examining how the downstream node relies on each upstream segment in the concatenated context. It follows the same computation as the original ECS, but restricts the scope from the whole context to a specific context segment.

We instantiate this experiment on a sampled MMLU test set [57] using an Expert → Critic → FinalDecision pipeline with eight expert nodes: seven domain-specific experts and one general Direct Answer expert. The Critic receives a mixed context composed of their outputs. These experts are prompt-defined roles using the same model backbone, rather than agents with separate knowledge bases or tool-calling abilities. For each question, one expert domain matches the question category, while the other experts provide plausible but non-matched content. This design creates a controlled multi-source setting where useful and distracting upstream information are presented together, allowing us to test whether seg-ECS is higher for the relevant upstream source.

We compute seg-ECS for each upstream expert segment. For each question, we apply z-score normalization across experts, so that values are comparable across question categories. Figure 2a also includes the Direct Answer Expert as a visual reference for a generally useful answer-providing source. Unlike the domain experts, it attempts to answer each question regardless of the assigned domain role.

As shown in Figure 2a, when an expert domain matches the question category, the Critic assigns higher seg-ECS to that expert than to non-matched experts (mean paired difference 0.0735, 95% CI [0.0683, 0.0786]). This result extends the node-level ECS pattern from input relevance to sourcelevel contribution in multi-source aggregation. Detailed experimental setup, filtering criteria, metric computation, and statistical testing are provided in Appendix C.2.

![](images/564a367ec91b02c9bc5693da6001df29136a19178a3597ffd356bce45caa2a82.jpg)  
(a) Source-level seg-ECS

![](images/1de2b52c96b2048e5b1a8a8fbb573f2d00ea6743222c3ef6fb79ed074a0720c9.jpg)  
(b) Chain-level PKS trajectory  
Figure 2: Behavioral probes in multi-source aggregation and chain-reasoning settings. (a) Zstandardized segment-level ECS of the Critic with respect to upstream expert outputs; higher values indicate stronger reliance on the corresponding segment. (b) PKS trajectories along the reasoning chain, with labels indicating the last taxonomic level that introduces new information. Solid lines show means and shaded regions show the 15–85 percentile range. Overall, seg-ECS highlights domain-relevant upstream sources, while PKS decreases only when informative context continues to accumulate.

## 3.2.2 Chain Reasoning and Reasoning Convergence

Recent work has explored signals for monitoring reasoning dynamics [58–63]. For LLM swarms, this motivates a similar question: can a behavioral probe track information accumulation along a reasoning path? We therefore examine whether PKS can reflect reduced parametric-knowledge dependence as useful context is progressively supplied.

We construct a chain-reasoning setting where the model receives new information at each step. As the context accumulates, the model should rely less on parametric knowledge when making predictions. This experiment tests whether PKS can reflect this change along sequential interactions.

We study PKS dynamics with a biological-taxonomy classification task constructed from NCBI Taxonomy records [64]. We select 32 species instances and organize their taxonomic information into five rank-like levels: phylum, class, order, family, and genus. At each step, the model predicts the genus from the taxonomic information accumulated so far. To separate effective information gain from chain length, we use truncated conditions where only the first k steps introduce new taxonomic information (k = 1, 2, 3, 4), while later steps repeat the last informative level and continue passing previous outputs along the chain.

As shown in Figure 2b, PKS decreases monotonically in the full incremental condition, where each step adds new taxonomic information. In truncated conditions, PKS decreases only until the last informative step, and then stabilizes when later positions repeat the same effective context. This contrast suggests that PKS responds to information gain, rather than to chain length alone. PKS decline should not be interpreted as evidence of correctness: a lower PKS indicates reduced parametric-knowledge dependence, but does not guarantee that the provided context is correct (see Appendix D.2 for further discussion).

Together with the multi-source aggregation experiment, these results show that ECS reflects sourcelevel reliance on useful upstream information, while PKS reflects information accumulation and reasoning convergence along structured interactions between nodes.

## 3.3 Behavioral Probes in Realistic Tool-Augmented GAIA Reasoning

We next ask whether these behavioral probes remain informative in a more realistic tool-augmented reasoning pipeline, where the swarm involves tool calls, intermediate analysis, and final answer synthesis. This setting extends the analysis beyond controlled structures while allowing us to examine whether ECS aligns with useful or decision-critical tool evidence and whether PKS changes when key evidence is withheld from the final answer stage.

![](images/2bf23cd5fcdce18c41aa7a28ee06938c1d17979cc6e3a7e90707a24b10beb319.jpg)  
(a) Critical evidence

![](images/df396b402c38a9ccbf3f659d45f4060ff1996e4a53864b25ab031b1caaef9799.jpg)  
(b) WebSearch relevance

![](images/60077b94b8b692df9309d335e4d204ca8907a005d75803c5bf9e0bbbdee57494.jpg)  
(c) PKS after ablation  
Figure 3: Behavioral probes in a realistic tool-augmented GAIA pipeline. Panel (a) compares ECS for decision-critical and non-critical tool evidence across WebSearch and FileAnalyse; panel (b) compares ECS for function-calling search-result segments labeled as useful or irrelevant by the corresponding WebSearch analysis output; panel (c) compares CombineAnswer PKS before and after removing WebSearch evidence.

We conduct experiments on GAIA Level-1 validation [65], after manually excluding video-modality instances. Following the original GAIA setup, we construct a Tree-of-Thought (ToT) reasoning system with four stages. GenerateQuery generates search queries from the input question. WebSearch and FileAnalyse are two parallel tool-use branches that perform function calling and information extraction. CombineAnswer integrates the collected evidence into the final response.

We first ask whether ECS aligns with evidence that is decision-critical for final correctness. In this pipeline, the intermediate outputs from WebSearch and FileAnalyse serve as external evidence for CombineAnswer. We define an evidence branch as decision-critical if removing that branch alone changes a correct baseline answer into an incorrect one. The baseline ToT pipeline achieves 21.2% accuracy (11/52), and single-branch ablations identify six decision-critical evidence cases: three from WebSearch and three from FileAnalyse. For both branches, removing the entire branch output reduces accuracy from 21.2% to 15.4%. Pruning only high-ECS evidence above the approximately 40th-percentile threshold (ECS ≈ 0.35) produces the same drop, whereas pruning low-ECS evidence leaves accuracy unchanged at 21.2%. Figure 3a further shows higher ECS for critical evidence than for non-critical evidence (mean difference 0.150, 95% CI [0.052, 0.246]; permutation p = 0.066). Given the small number of critical cases, we treat this result as an exploratory trend rather than a standalone significance claim; detailed ablation definitions, critical-evidence counts, and statistical tests are provided in Appendix C.4.

Beyond decision-criticality, we test whether ECS aligns with local relevance within WebSearch. Here, segments refer to search results returned by the function-calling tool. We label each segment using the corresponding WebSearch analysis output: segments whose output contains “No useful information from WebSearch” are labeled irrelevant, while the others are treated as useful retrievals. We then compute segment-level ECS during WebSearch generation, measuring how much the generated analysis relies on each search-result segment. As shown in Figure 3b, useful retrieval segments have higher ECS than irrelevant segments, with a mean difference of 0.243 (95% CI [0.194, 0.285]).

Finally, we examine whether PKS reflects downstream changes when external evidence is withheld from the final answer stage. At the CombineAnswer stage, removing WebSearch outputs consistently increases PKS (Figure 3c), with a paired mean difference of 0.214 (95% CI [0.114, 0.309]). This suggests greater parametric-knowledge dependence when useful external evidence is missing.

Overall, GAIA extends the controlled analysis to a realistic tool-augmented pipeline: ECS aligns with useful or decision-critical evidence, while PKS increases when external evidence is withheld from answer synthesis.

## 3.4 Summary

Across controlled and realistic LLM swarm settings, ECS and PKS exhibit consistent behavioral patterns that make swarm information flow observable. ECS reflects how downstream nodes use external evidence, whereas PKS changes as informative context accumulates or is withheld. These findings motivate the probe-based interpretation developed in the next section. Detailed experimental configurations and dataset descriptions are provided in Appendix C.

## 4 A Heuristic Interpretation of Behavioral Probes for Structure Optimization

The experiments in Section 3 show that information flow in LLM swarms —how external context is routed, integrated, and balanced against internal parametric knowledge— can be characterized through behavioral probes. ECS rises with input relevance and distinguishes multi-source contributions, while PKS declines as informative context accumulates and rises when evidence is withheld. These patterns motivate the heuristic interpretation adopted throughout this work: ECS serves as a heuristic proxy for external-context dependence, while PKS serves as a heuristic proxy for parametricknowledge dependence. We use this interpretation as an empirically grounded lens for analyzing information flow, rather than as a formal decomposition of model computation.

Building on this empirical interpretation, we propose that these behavioral probes can complement utility-driven structure optimization in two specific directions.

Probe-Guided Node Pruning. From an information-flow-based view, effective communication requires that nodes both utilize incoming context and contribute non-redundant information to their successors. Each node can be represented by a behavioral profile $\phi ( v ) = ( E C S _ { v } , P K S _ { v } )$ , where ECS is normalized within local aggregation groups and PKS is normalized globally. Nodes whose profiles deviate from a task-appropriate region $\Omega _ { \mathrm { { t a s k } } } -$ for instance, those that largely ignore incoming messages or act as redundant relays—can be assigned lower structural reward. Intuitively, $\Omega _ { \mathrm { t a s k } }$ represents a “normal behavior region” in the ECS–PKS space. For example, in GAIA, useful WebSearch nodes tend to occupy a high-ECS, low-PKS region, whereas AdversarialAnswer nodes fall into a low-ECS, high-PKS region (visualized in Appendix D.3, Figure 5). This profile-based comparison provides a probe-guided view for identifying harmful or superfluous communication paths, complementing utility-driven pruning by characterizing information flow across interacting nodes rather than relying solely on final task utility.

Probe-Guided Reasoning-Path Search. In sequential reasoning, information flow converges when downstream nodes progressively shift from parametric-knowledge dependence toward externalcontext dependence. The systematic PKS decline observed in chain reasoning indicates such assimilation of incoming information, providing a behavioral signature of convergence along a communication path. By tracking these decreases across candidate reasoning edges, one can prioritize paths that move toward convergence within the swarm’s sequential information flow. Since PKS decline reflects perceived information sufficiency rather than factual correctness, we use it as a complementary signal that adds an information-flow dimension to reasoning-path search, not as a substitute for final task utility.

Appendix D provides additional discussion of the assumptions and limitations behind this heuristic interpretation.

## 5 Probe-Guided Applications for Structure Optimization

Building on the heuristic interpretation in Section 4, we instantiate two controlled application cases for structure optimization. Specifically, the node-pruning application uses the “abnormal ECS–PKS profiles” rule, while the reasoning-path search application uses the “PKS decline as a convergence signal” rule.

## 5.1 Case Study I: Probe-Guided Node Pruning

The first case uses ECS–PKS behavioral profiles for node pruning. Beyond final task utility, these profiles characterize two aspects of node behavior: whether a node’s output is used by downstream nodes, and how much the node itself relies on parametric knowledge. When ECS is substantially low, the node’s information is weakly used by downstream reasoning. When PKS is substantially low, the node may act as a redundant propagation step after reasoning has already converged. We therefore use these low-score regions of the ECS–PKS profile to identify nodes that contribute little useful information flow.

We instantiate this test in a random-DAG pruning task with three node types: DirectAnswer (DA) nodes answer the question directly, BubbleAnswer (BA) nodes produce irrelevant answers, and LazyAnswer (LA) nodes copy existing answers. Thus, BA outputs should induce low downstream ECS, while LA nodes themselves exhibit low PKS due to their reliance on copied context. Each swarm contains two DA source nodes, two LA and two BA intermediate nodes, and one final decision node (FD) as the only sink. Following the random-DAG construction used in GPTSwarm [36], the optimization space includes all DAG structures satisfying these source, sink, and intermediate-node constraints. An ideal pruned structure preserves useful DA-to-FD information flow while suppressing redundant or irrelevant LA/BA contributions. We evaluate the learned structure by cumulative inbound edge weight to FD, where higher DA weight and lower LA/BA weight indicate cleaner pruning.

Table 1: Node-pruning results across baselines. DA/LA/BA denote cumulative inbound weights to the final decision node; Rdn sums redundant LA/BA weights; Tok/Q is average tokens per question. The Signal column distinguishes utility-driven baselines from probe-guided utility strategies. <sup>∗</sup> marks variants using ECS–PKS behavioral profiles.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Signal</td><td rowspan="2">Method</td><td colspan="4">Weights</td><td colspan="2">Efficiency</td></tr><tr><td>DA</td><td>LA</td><td>BA</td><td>Rdn</td><td>Acc (%) ↑</td><td>Tok/Q ↓</td></tr><tr><td rowspan="6">Qwen-2.5-32B</td><td rowspan="3">Vanilla utility</td><td>GPT Swarm</td><td>1.515</td><td>1.505</td><td>1.438</td><td>2.943</td><td>80.6</td><td>2294</td></tr><tr><td>Agent Net</td><td>1.876</td><td>1.834</td><td>1.849</td><td>3.683</td><td>80.8</td><td>2744</td></tr><tr><td>Agent Prune</td><td>1.140</td><td>0.837</td><td>0.708</td><td>1.545</td><td>80.8</td><td>1879</td></tr><tr><td rowspan="3">Probe-guided utility</td><td>Probe-ECS*</td><td>1.859</td><td>1.650</td><td>0.551</td><td>2.201</td><td>79.4</td><td>2345</td></tr><tr><td>Probe-PKS*</td><td>1.052</td><td>0.432</td><td>1.599</td><td>2.031</td><td>79.4</td><td>1710</td></tr><tr><td>Probe-Joint*</td><td>1.474</td><td>0.491</td><td>0.506</td><td>0.997</td><td>80.0</td><td>1589</td></tr><tr><td rowspan="6">LLaMA-3-70B</td><td rowspan="3">Vanilla utility</td><td>GPT Swarm</td><td>1.287</td><td>1.147</td><td>1.081</td><td>2.228</td><td>72.0</td><td>2282</td></tr><tr><td>Agent Net</td><td>1.912</td><td>1.897</td><td>1.864</td><td>3.761</td><td>72.2</td><td>2984</td></tr><tr><td>Agent Prune</td><td>1.262</td><td>1.111</td><td>1.068</td><td>2.179</td><td>71.4</td><td>1980</td></tr><tr><td rowspan="3">Probe-guided utility</td><td>Probe-ECS*</td><td>1.684</td><td>1.394</td><td>0.123</td><td>1.517</td><td>71.4</td><td>2501</td></tr><tr><td>Probe-PKS*</td><td>0.353</td><td>0.308</td><td>1.866</td><td>2.174</td><td>70.2</td><td>1691</td></tr><tr><td>Probe-Joint*</td><td>1.528</td><td>0.297</td><td>0.201</td><td>0.498</td><td>71.2</td><td>1563</td></tr></table>

We compare against three utility-driven baselines. GPT Swarm [36] is the most basic utility-driven topology optimizer, which updates the structure using final task utility. Agent Prune [12] and Agent Net [11] inspired baselines also belong to this family, representing gradient-sensitivity pruning and trajectory-credit assignment, respectively. In our probe-guided variant, behavioral probes are combined with utility-based edge updates as a complementary structural signal. Detailed experimental settings, including graph construction, evaluation metrics, and baseline definitions, are provided in Appendices E and F.1.

Table 1 shows that Probe-Joint recovers the cleanest topology across both models. It achieves the lowest redundant LA/BA weight, while preserving comparable accuracy and reducing token consumption. In contrast, utility-driven baselines retain larger redundant LA/BA weights. This suggests that final task utility alone provides limited guidance for pruning unnecessary node contributions. The Agent-Prune-inspired baseline improves sparsity, but the recovered topology is still less clean than that obtained by Probe-Joint. The single-probe ablations reveal both complementarity and limitation. Probe-ECS mainly suppresses BA contributions, consistent with its role in identifying information that is not useful for downstream nodes. Probe-PKS mainly suppresses LA contributions, but it can also reduce DA weights. This occurs because BA is a divergent generation node: although its output is question-irrelevant, it can still receive high PKS due to stronger parametric-knowledge dependence.

Probe-Joint addresses this limitation by requiring both behavioral axes to be considered together. ECS filters question-irrelevant BA contributions, while PKS identifies redundant LA contributions after the reasoning has already converged. This highlights the value of behavioral profiles as complementary signals for probe-guided node pruning. A separate paired evaluation on 500 test questions quantifies the accuracy–communication trade-off; few-shot expert selection and conditional escalation experiments are also reported in Appendix H.

## 5.2 Case Study II: Probe-Guided Reasoning-Path Search

The second case uses a PKS-based convergence probe for reasoning-path search. For each candidate edge, a larger downstream PKS decrease suggests that the upstream information helps reduce parametric-knowledge dependence. We therefore use this signal to prioritize convergent paths, while retaining utility signals to preserve task correctness.

![](images/191c9b6d1cdf0fab0c62c56ed6fee76388d92a1de51c8095f5e7b7165d080ae4.jpg)  
(a) Good warm-starts

![](images/976c8890d9ecbbf309000a7a110a6345a76b19b2081e9d8946a7225e086deb91.jpg)  
(b) Moderate warm-starts

![](images/beae1627f288c448323b54fd2ee7f6b84035f5e2f5f3cc2969bbc631a2d5bccd.jpg)  
(c) Poor warm-starts  
Figure 4: Critical edge weight trends under three warm-starts. After 50 utility-only warm-up iterations, the PKS-based convergence probe is introduced. The panels show good, moderate, and poor warm-starts, defined by the critical-edge weight before probe intervention. Solid lines denote the mean weight over critical edges, with shaded regions showing the 15–85 percentile range. Across all cases, the probe accelerates critical-edge growth compared with the utility-only baseline.

We consider a taxonomic reasoning task similar to the chain analysis in Section 3.2.2, but adapt it for reasoning-path search. The task contains five ordered reasoning nodes corresponding to five taxonomic levels, and paths are sampled with a softmax-weighted Markov-chain procedure; see Appendix E.2 for details.

Taxonomic context is accumulated only when the sampled nodes follow the correct order; otherwise, downstream nodes receive only the original highest-level taxonomic information. Thus, among all sampled paths, only the fully ordered chain follows the complete taxonomic progression toward the final genus.

This design creates a targeted test for reasoning-path search: task utility is observed only after a sampled path produces a final genus prediction, while the ordered taxonomy provides intermediate structure for detecting reasoning convergence. We therefore examine whether the PKS-based convergence probe can guide path selection toward the correct ordered chain.

Unlike the chain analysis in Section 3.2.2, this experiment does not propagate full generated responses between nodes; downstream nodes receive only structured taxonomic clues, which keeps intermediate inputs controlled and reduces generation-induced variation.

For reasoning-path search, we introduce the PKS-based convergence probe after a 50-iteration utilityonly warm-up. During warm-up, PKS is collected on each edge and EMA-smoothed, allowing edgelevel PKS-drop patterns to stabilize before optimization uses them. We then compare Probe+Utility, which adds the PKS-based reward to utility, against Utility Only, which continues using only utility reward.

The probe reward is based on the smoothed PKS drop along each edge, $\Delta P \bar { K } S _ { e } = P \bar { K } S _ { u } - P \bar { K } S _ { \imath }$ A positive drop indicates that the downstream node becomes less dependent on parametric knowledge after receiving the upstream information. Edges with larger PKS drops are therefore treated as more likely to lie on convergent reasoning paths. The detailed reward formulation is provided in Appendix E.3.3.

Figure 4 compares the two settings under three different warm-starts. In the good warm-start, the first 50 iterations have already explored the critical convergent path. In the moderate warm-start, the warm-up phase receives little utility reward, so the weight of the critical edge remains close to its initial level. In the poor warm-start, the critical edge receives low weight after warm-up. Because edge selection is based on softmax sampling, a wrong path reinforced by utility signals can suppress the sampling probability of the critical edge, especially when that wrong path reaches the correct answer by chance.

Across all three cases, introducing the PKS-based convergence probe after iteration 50 helps identify the critical edge and accelerates the growth of its weight. This suggests that the PKS-based convergence probe can provide useful local guidance when utility signals are sparse, delayed, or misleading. We observe the same qualitative trends with LLaMA3-70B-Instruct [52].

## 6 Conclusion

This paper studies LLM swarms from an information-flow-based view. We show that ECS and PKS can serve as behavioral probes for making the information flow formed by individual interactions observable. Based on this heuristic interpretation, we further propose two ways to use these probes to complement utility rewards in structure optimization: ECS–PKS behavioral profiles for node pruning, and PKS-based convergence signals for reasoning-path search. The results demonstrate the potential of combining behavioral probes with utility rewards for LLM swarm structure optimization.

## Acknowledgment

This work was supported by the National Natural Science Foundation of China under Grant 62171250 and Grant 62301075.

## References

[1] Shuaihang Chen, Yuanxing Liu, Wei Han, Weinan Zhang, and Ting Liu. A survey on llmbased multi-agent system: Recent advances and new frontiers in application. arXiv preprint arXiv:2412.17481, 2024.

[2] Jiayi Chen, Junyi Ye, and Guiling Wang. From standalone llms to integrated intelligence: A survey of compound al systems. arXiv preprint arXiv:2506.04565, 2025.

[3] Mohamed Amine Ferrag, Norbert Tihanyi, and Merouane Debbah. From llm reasoning to autonomous ai agents: A comprehensive review. arXiv preprint arXiv:2504.19678, 2025.

[4] Zixuan Ke, Fangkai Jiao, Yifei Ming, Xuan-Phi Nguyen, Austin Xu, Do Xuan Long, Minzhi Li, Chengwei Qin, PeiFeng Wang, silvio savarese, Caiming Xiong, and Shafiq Joty. A survey of frontiers in LLM reasoning: Inference scaling, learning to reason, and agentic systems. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=SlsZZ25InC. Survey Certification.

[5] Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhong-Zhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Francisco Piedrahita Velez, Yue Liao, Hongru WANG, Mengyue Yang, Heng Ji, Jun Wang, Shuicheng YAN, Philip Torr, and LEI BAI. The landscape of agentic reinforcement learning for LLMs: A survey. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/forum?id=RY19y2RI1O. Survey Certification.

[6] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, pages 8048–8057, 2024. doi: 10.24963/ijcai.2024/890.

[7] Chaojia Yu, Zihan Cheng, Hanwen Cui, Yishuo Gao, Zexu Luo, Yijin Wang, Hangbin Zheng, and Yong Zhao. A survey on agent workflow–status and future. In 2025 8th International Conference on Artificial Intelligence and Big Data (ICAIBD), pages 770–781. IEEE, 2025.

[8] Mingjie Zhu, Mingcheng Li, Mingzhang Cui, Ziyun Qian, Yue Jiang, Minghao Han, Mingyang Sun, Zizhi Chen, Yizhou Liu, Jinghang Han, et al. A survey on large language model-based multi-agent systems: Paradigms, applications, and challenges. Authorea Preprints, 2026.

[9] Wenxiao Wang, Priyatham Kattakinda, and Soheil Feizi. Maestro: Joint graph & config optimization for reliable ai agents. arXiv preprint arXiv:2509.04642, 2025.

[10] Wenzhe Fan, Tommaso Tognoli, Henry Peng Zou, Chunyu Miao, Yibo Wang, and Xinhua Zhang. Todycomm: Task-oriented dynamic communication for multi-round llm-based multiagent system. arXiv preprint arXiv:2602.03688, 2026.

[11] Yingxuan Yang, Huacan Chai, Shuai Shao, Yuanyi Song, Siyuan Qi, Renting Rui, and Weinan Zhang. Agentnet: Decentralized evolutionary coordination for llm-based multi-agent systems. ArXiv, abs/2504.00587, 2025. URL https://api.semanticscholar.org/CorpusID: 277468263.

[12] Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, and Tianlong Chen. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. In The Thirteenth International Conference on Learning Representations, 2025.

[13] Ruijia Zhang, Xinyan Zhao, Ruixiang Wang, Sigen Chen, Guibin Zhang, An Zhang, Kun Wang, and Qingsong Wen. Safesieve: From heuristics to experience in progressive pruning for llm-based multi-agent communication, 2026. URL https://arxiv.org/abs/2508.11733.

[14] Yun Hua, Haosheng Chen, Shiqin Wang, Wenhao Li, Xiangfeng Wang, and Jun Luo. Shapleycoop: Credit assignment for emergent cooperation in self-interested LLM agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https: //openreview.net/forum?id=HnJ1UkuJXS.

[15] Yanming Li, Xuelin Zhang, WenJie Lu, Ziye Tang, Maodong Wu, Haotian Luo, Tongtong Wu, Zijie Peng, Hongze Mi, Yibo Feng, et al. Who deserves the reward? sharp: Shapley credit-based optimization for multi-agent system. arXiv preprint arXiv:2602.08335, 2026.

[16] Yugu Li, Zehong Cao, Jianglin Qiao, and Siyi Hu. Ssvpo: Effective step-level credit assignment for rl training of language models. In The Fourteenth International Conference on Learning Representations, 2026.

[17] Shangbin Feng, Zifeng Wang, Palash Goyal, Yike Wang, Weijia Shi, Huang Xia, Hamid Palangi, Luke Zettlemoyer, Yulia Tsvetkov, Chen-Yu Lee, et al. Heterogeneous swarms: Jointly optimizing model roles and weights for multi-llm systems. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[18] Chenchen Zhang. From reasoning to agentic: Credit assignment in reinforcement learning for large language models, 2026. URL https://arxiv.org/abs/2604.09459.

[19] Zhouyang Jiang, Bin Zhang, Yuanjun Li, and Zhiwei Xu. Qllm: Do we really need a mixing network for credit assignment in multi-agent reinforcement learning? arXiv preprint arXiv:2504.12961, 2025.

[20] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=QXEhBMNrCW.

[21] Kartik Nagpal, Dayi Dong, Jean-Baptiste Bouvier, and Negar Mehr. Leveraging large language models for effective and explainable multi-agent credit assignment. arXiv preprint arXiv:2502.16863, 2025.

[22] Yuxing Lu, Yucheng Hu, Xukai Zhao, and Jiuxin Cao. Dytopo: Dynamic topology routing for multi-agent reasoning via semantic matching. arXiv preprint arXiv:2602.06039, 2026.

[23] Tongtong Wu, Yanming Li, Ziye Tang, Chen Jiang, Linhao Luo, Guilin Qi, Shirui Pan, and Gholamreza Haffari. CARD: Towards conditional design of multi-agent topological structures. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=JgvJdICc6P.

[24] Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, and Dawei Cheng. G-designer: Architecting multi-agent communication topologies via graph neural networks. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 76678–76692, 2025.

[25] Zhitao He, Zijun Liu, Peng Li, Yi Fung, Ming Yan, Ji Zhang, Fei Huang, and Yang Liu. Enhancing multi-agent learning in real-world interactive environments through process reward decomposition, 2025. URL https://openreview.net/forum?id=E2CR6hmV1I.

[26] Edward Y Chang and Longling Geng. Sagallm: context management, validation, and transaction guarantees for multi-agent llm planning. arXiv preprint arXiv:2503.11951, 2025.

[27] Bingyu Yan, Zhibo Zhou, Litian Zhang, Lian Zhang, Ziyi Zhou, Dezhuang Miao, Zhoujun Li, Chaozhuo Li, and Xiaoming Zhang. Beyond self-talk: A communication-centric survey of llm-based multi-agent systems. arXiv preprint arXiv:2502.14321, 2025.

[28] ZhongXiang Sun, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao Zhang, Weijie Yu, Yang Song, and Han Li. RedeEP: Detecting hallucination in retrieval-augmented generation via mechanistic interpretability. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=ztzZDzgfrh.

[29] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In The twelfth international conference on learning representations, 2023.

[30] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First conference on language modeling, 2024.

[31] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large language model society. Advances in neural information processing systems, 36:51991–52008, 2023.

[32] Yiqun Zhang, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang, and Kaisong Song. Can llms beat humans in debating? a dynamic multi-agent framework for competitive debate. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 19377–19381. IEEE, 2026.

[33] Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. A dynamic llm-powered agent network for task-oriented agent collaboration. In First Conference on Language Modeling, 2024.

[34] Chunhao Tian, Yutong Wang, Xuebo Liu, Zhexuan Wang, Liang Ding, Miao Zhang, and Min Zhang. Agentinit: Initializing llm-based multi-agent systems via diversity and expertise orchestration for effective and efficient collaboration. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 11870–11902, 2025.

[35] Selim Furkan Tekin, Fatih Ilhan, Gaowen Liu, Ramana Rao Kompella, and Ling Liu. Dynamic optimizations of LLM ensembles with two-stage reinforcement learning agents, 2026. URL https://openreview.net/forum?id=mhalNLINnn.

[36] Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, 2024.

[37] Jiaxi Yang, Mengqi Zhang, Yiqiao Jin, Hao Chen, Qingsong Wen, Lu Lin, Yi He, Srijan Kumar, Weijie Xu, James Evans, et al. Topological structure learning should be a research priority for llm-based multi-agent systems. arXiv preprint arXiv:2505.22467, 2025.

[38] Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, et al. Scaling large language model-based multi-agent collaboration. In The Thirteenth International Conference on Learning Representations, 2025.

[39] Xu Shen, Yixin Liu, Yiwei Dai, Yili Wang, Rui Miao, Yue Tan, Shirui Pan, and Xin Wang. Understanding the information propagation effects of communication topologies in llm-based multi-agent systems. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12358–12372, 2025.

[40] Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, Rafael Rafailov, Philip Torr, Ivan Laptev, Fabio Pizzati, Ronald Clark, and Christian Schroeder de Witt. MALT: Improving reasoning with multi-agent LLM training. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=jXP9bgFack.

[41] Yilin Wang, Heng Wang, Yuyang Bai, and Minnan Luo. Continuously steering llms sensitivity to contextual knowledge with proxy models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 4682–4698, 2025.

[42] Sitao Cheng, Liangming Pan, Xunjian Yin, Xinyi Wang, and William Yang Wang. Understanding the interplay between parametric and contextual knowledge for large language models, 2025. URL https://openreview.net/forum?id=t21RmVmJrT.

[43] Patrick Kahardipraja, Reduan Achtibat, Thomas Wiegand, Wojciech Samek, and Sebastian Lapuschkin. The atlas of in-context learning: How attention heads shape in-context retrieval augmentation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=5h9mS87Pyt.

[44] Andrew Parry, Catherine Chen, Carsten Eickhoff, and Sean MacAvaney. Mechir: A mechanistic interpretability framework for information retrieval. In European Conference on Information Retrieval, pages 89–95. Springer, 2025.

[45] Han Wang, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. Adacad: Adaptively decoding to balance conflicts between contextual and parametric knowledge. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11636–11652, 2025.

[46] Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, and Emmanuel Malherbe. Learned hallucination detection in black-box llms using token-level entropy production rate. In European Conference on Information Retrieval, pages 115–130. Springer, 2026.

[47] Sharanya Dasgupta, Sujoy Nath, Arkaprabha Basu, Pourya Shamsolmoali, and Swagatam Das. Hallushift: Measuring distribution shifts towards hallucination detection in llms. In 2025 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2025.

[48] Ao Sun. Chair-classifier of hallucination as improver. In 2025 International Joint Conference on Neural Networks (IJCNN), pages 1–6. IEEE, 2025.

[49] Yu Zhao, Alessio Devoto, Giwon Hong, Xiaotang Du, Aryo Pradipta Gema, Hongru Wang, Xuanli He, Kam-Fai Wong, and Pasquale Minervini. Steering knowledge selection behaviours in llms via sae-based representation engineering. In Proceedings of the 2025 Conference of the Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5117–5136, 2025.

[50] Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, and Xueqi Cheng. Parameters vs. context: Fine-grained control of knowledge reliance in language models. In The Fourteenth International Conference on Learning Representations, 2026.

[51] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

[52] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[53] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". In The Twelfth International Conference on Learning Representations, 2024.

[54] Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, and Aleksander M ˛adry. Contextcite: Attributing model generation to context. Advances in Neural Information Processing Systems, 37:95764–95807, 2024.

[55] Yanting Wang, Wei Zou, Runpeng Geng, and Jinyuan Jia. {TracLLM}: A generic framework for attributing long context {LLMs}. In 34th USENIX Security Symposium (USENIX Security 25), pages 3845–3864, 2025.

[56] James Gao, Josh Zhou, Qi Sun, Ryan Huang, and Steven Yoo. Atomic information flow: A network flow model for tool attributions in rag systems. arXiv preprint arXiv:2602.04912, 2026.

[57] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

[58] Xinghao Zhao. Entropy trajectory shape predicts llm reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought. arXiv preprint arXiv:2603.18940, 2026.

[59] Aman Sharma and Paras Chopra. Think just enough: Sequence-level entropy as a confidence signal for LLM reasoning, 2026. URL https://openreview.net/forum?id=doR25jEjbj.

[60] Jingwei Ni, Ekaterina Fadeeva, Tianyi Wu, Mubashara Akhtar, Jiaheng Zhang, Elliott Ash, Markus Leippold, Timothy Baldwin, See-Kiong Ng, Artem Shelmanov, and Mrinmaya Sachan. Reasoning with confidence: Efficient verification of LLM reasoning steps via uncertainty heads, 2026. URL https://openreview.net/forum?id=svQuvBYaCA.

[61] Zihuiwen Ye, Luckeciano Carvalho Melo, Younesse Kaddar, Phil Blunsom, Sam Staton, and Yarin Gal. Uncertainty-aware step-wise verification with generative reward models. In ICLR Workshop: Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI, 2025. URL https://openreview.net/forum?id=6YICujbAtG.

[62] Chenghua Zhu, Siyan Wu, Xiangkang Zeng, Zishan Xu, Zhaolu Kang, Yifu Guo, Yuquan Lu, Junduan Huang, and Guojing Zhou. Edis: Diagnosing llm reasoning via entropy dynamics. arXiv preprint arXiv:2602.01288, 2026.

[63] Amirhosein Ghasemabadi, Keith G Mills, Baochun Li, and Di Niu. Guided by gut: Efficient test-time scaling with reinforced intrinsic confidence. arXiv preprint arXiv:2505.20325, 2025.

[64] Conrad L Schoch, Stacy Ciufo, Mikhail Domrachev, Carol L Hotton, Sivakumar Kannan, Rogneda Khovanskaya, Detlef Leipe, Richard Mcveigh, Kathleen O’Neill, Barbara Robbertse, et al. Ncbi taxonomy: a comprehensive update on curation, resources and tools. Database, 2020:baaa062, 2020.

[65] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations, 2023.

[66] Jun Zhao, Yongzhuo Yang, Xiang Hu, Jingqi Tong, Yi Lu, Wei Wu, Tao Gui, Qi Zhang, and Xuanjing Huang. Understanding parametric and contextual knowledge reconciliation within large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[67] Charles Westphal, Stephen Hailes, and Mirco Musolesi. A generalized information bottleneck theory of deep learning. arXiv preprint arXiv:2509.26327, 2025.

[68] Caiqi Zhang, Ruihan Yang, Xiaochen Zhu, Chengzu Li, Tiancheng Hu, Yijiang River Dong, Deqing Yang, and Nigel Collier. Confidence estimation for llms in multi-turn interactions. arXiv preprint arXiv:2601.02179, 2026.

[69] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

## A Limitations and Future Work

Several limitations warrant discussion. First, computing ECS and PKS requires access to internal model states, including hidden representations and attention weights. This incurs additional computational overhead during training and is not feasible for black-box API-based models. While this restricts immediate practical deployment, the information-flow-based view and heuristic interpretation advanced here remain useful for understanding information flow in LLM swarms and for motivating future research into more accessible behavioral probes. Second, our experiments are limited to controlled, small-scale benchmarks; scalability to complex, open-ended environments remains to be tested. Third, the reward designs derived from the heuristic interpretation are ad hoc, and more principled ways of combining behavioral probes with utility signals may improve performance. Fourth, ECS and PKS may vary in sensitivity across model architectures and tasks. Finally, PKS decline reflects perceived information sufficiency rather than correctness, motivating integration with complementary cues such as uncertainty estimates. Addressing these limitations is essential for turning the present empirical tests into practical tools.

These limitations also clarify the intended scope of our claims. We do not argue that ECS or PKS should replace utility signals, nor that the observed probe patterns provide causal guarantees about model behavior. Rather, the results support a narrower claim: under accessible-model settings where internal states can be measured, ECS and PKS can serve as behavioral probes that complement final task utility. Future work should test whether cheaper proxy signals, black-box approximations, or uncertainty-calibrated variants can preserve similar sensitivity to information flow at larger scale.

Importantly, this paper should be read as an empirical study rather than a ready-to-deploy method. Our primary goal is to show that behavioral probes can make meaningful information-flow patterns observable in LLM swarms. The two application experiments, node pruning and reasoning-path search, serve as controlled tests of downstream possibilities, not as fully optimized techniques. Claims of practical superiority over utility-driven optimization would require extensive evaluation on diverse real-world tasks, which lies beyond the scope of this initial study.

In addition, ECS and PKS should not be treated as definitive explanations or safety guarantees, especially when such systems are considered for high-stakes use. High ECS can reflect reliance on incorrect evidence; copying or false convergence can also improve probe rewards without improving correctness. Task/role calibration, utility checks, and evidence verification remain necessary. Better diagnosis and efficiency may benefit agent systems, while over-trusting these signals or strengthening harmful automation can have negative impacts.

## B LLM Swarm Topology and Behavioral Metrics

This section formally defines the Directed Acyclic Graph (DAG) representation of LLM swarms, the optimization objective for structure selection, and the computational procedures for the External Con text Score (ECS) and Parametric Knowledge Score (PKS). We adapt ECS and PKS from ReDeEP [28], but use a modified deterministic layer/head selection strategy informed by its empirical findings on layer and head selection; the DAG-based swarm optimization setup follows GPTSwarm [36].

## B.1 Swarm Topology as a Directed Acyclic Graph

In an LLM swarm, the collaborative topology is represented as a DAG. Specifically, an agent or a computational workflow can be formalized as a computational graph:

$$
G = ( N , E , F , o )\tag{1}
$$

where N denotes the set of nodes, each corresponding to a computational unit such as an LLM call, a tool invocation, or another processing module; $\vec { E ^ { \mathrm { ~ C ~ } } } \vec { N } \times \dot { \vec { N } }$ denotes the set of directed edges describing the information flow between nodes; $F = \{ f _ { n } \} _ { n \in N }$ denotes the computational functions associated with each node; and $o \in N$ denotes the designated output node.

During execution, the graph receives a task input $x ,$ representing the instance of the current task. Nodes are executed in topological order. For a node $n ,$ let pre(n) denote the set of its immediate predecessors. Node n receives the concatenated outputs from its predecessors:

$$
z _ { n } = \{ f _ { v } ( z _ { v } , x ) : v \in \mathrm { p r e } ( n ) \}\tag{2}
$$

The node then executes its computational function $f _ { n } ( z _ { n } , x )$ and passes the result to its successors. The final output of the graph is the result produced by the output node:

$$
\hat { y } = G ( x ) = f _ { o } ( z _ { o } , x )\tag{3}
$$

In a multi-agent scenario, each agent is itself a computational graph. Given a set of $K$ agents $\{ G _ { k } = ( N _ { k } , \mathbf { \bar { \phi } } _ { E _ { k } } , F _ { k } , o _ { k } ) \} _ { k = 1 } ^ { K }$ , a composite swarm graph is constructed by introducing cross-agent communication edges. Let:

$$
N ^ { \prime } = \bigcup _ { k } N _ { k } , \quad E ^ { \prime } = \bigcup _ { k } E _ { k } , \quad F ^ { \prime } = \bigcup _ { k } F _ { k }\tag{4}
$$

where $N ^ { \prime } , E ^ { \prime }$ , and $F ^ { \prime }$ represent the union of all nodes, internal edges, and computational functions, respectively. We introduce a set of cross-agent communication edges $\textstyle E \subset \bigcup _ { i \neq j } { \overline { { N _ { i } } } } \times N _ { j }$ , and define the complete edge set $E _ { E } = E ^ { \prime } \cup E$ . The composite graph for the swarm is then:

$$
G _ { E } = ( N ^ { \prime } , E _ { E } , F ^ { \prime } , o ^ { \prime } )\tag{5}
$$

where $o ^ { \prime }$ is the designated output node for the composite task.

## B.2 Topology Optimization Objective

Within this framework, the problem of optimizing the LLM swarm structure reduces to selecting an optimal set of communication edges E. Given a task τ and a corresponding utility function $u _ { \tau } ( G _ { E } )$ which typically depends on the correctness of the final task output, the objective of LLM swarm topology optimization is:

$$
\operatorname* { m a x } _ { E } u _ { \tau } ( G _ { E } )\tag{6}
$$

Let $d$ be the number of potential communication edges. The number of possible topologies is $2 ^ { d }$ making direct search in the discrete space computationally prohibitive. To address this, we introduce a parameterized probability distribution $D _ { \theta }$ to represent the sampling strategy for graph structures, where θ denotes the probability parameters for sampling each potential edge. The optimization is then formulated as maximizing the expected utility:

$$
\arg \operatorname* { m a x } _ { \theta \in \Theta } \mathbb { E } _ { G ^ { \prime } \sim D _ { \theta } } [ u _ { \tau } ( G ^ { \prime } ) ]\tag{7}
$$

Under the REINFORCE framework, the gradient estimate is given by:

$$
\nabla _ { \theta } \mathbb { E } _ { G ^ { \prime } \sim D _ { \theta } } [ u _ { \tau } ( G ^ { \prime } ) ] \approx \frac { 1 } { M } \sum _ { i = 1 } ^ { M } u _ { \tau } ( G _ { i } ) \nabla _ { \theta } \log p _ { \theta } ( G _ { i } ) , \quad G _ { i } \sim D _ { \theta }\tag{8}
$$

For each sampled graph $G _ { i } \sim D _ { \theta }$ , its probability is computed sequentially ensuring acyclicity:

$$
p _ { \theta } ( G _ { i } ) = \prod _ { j = 1 } ^ { d } \left\{ \begin{array} { l l } { \theta _ { j } } & { \mathrm { i f } \left( N , E \cup ( \{ e _ { k } \} _ { k = 1 } ^ { j - 1 } \cap \mathcal { E } ) \cup \{ e _ { j } \} \right) \mathrm { i s ~ a ~ D A G } , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{9}
$$

## B.3 External Context Score (ECS) Computation

We adopt the External Context Score (ECS) from the ReDeEP framework [28]. To reduce computational cost while focusing on the layers most directly related to the final generation behavior, we compute ECS on a deterministic subset of late-layer attention heads. Specifically, we select the final 5% of model layers and 10% of attention heads within each selected layer. Let the model have L layers, each containing H attention heads. We define

$$
M _ { E } = \left\lceil 0 . 0 5 L \right\rceil ,\tag{10}
$$

and the selected layer set as

$$
\mathcal { L } _ { \mathrm { E C S } } = \{ L - M _ { E } + 1 , . . . , L \} .\tag{11}
$$

For each selected layer $l \in \mathcal { L } _ { \mathrm { E C S } }$ , we deterministically select a subset of attention heads

$$
\mathcal { H } _ { \mathrm { E C S } } ^ { l } \subseteq \{ 1 , \dots , H \} ,\tag{12}
$$

with $| \mathcal { H } _ { \mathrm { E C S } } ^ { l } | = \lceil 0 . 1 H \rceil$ . The final ECS selection set is therefore

$$
S _ { \mathrm { E C S } } = \{ ( l , h ) \mid l \in \mathcal { L } _ { \mathrm { E C S } } , h \in \mathcal { H } _ { \mathrm { E C S } } ^ { l } \} .\tag{13}
$$

Segment-level ECS uses only the tokenizer indices belonging to the target content segment; role prompts and separators are excluded from the segment boundary.

For a generated token $t _ { n }$ and a selected head h in layer l, we define:

$$
E _ { n } ^ { l , h } = \frac { e \cdot x _ { n } ^ { L } } { \| e \| \| x _ { n } ^ { L } \| } , \quad e = \frac { 1 } { | I _ { n } ^ { l , h } | } \sum _ { j \in I _ { n } ^ { l , h } } x _ { j } ^ { L } ,\tag{14}
$$

where $x _ { n } ^ { L }$ denotes the hidden state of token $t _ { n }$ at the final layer $L ; I _ { n } ^ { l , h }$ denotes the set of context tokens that receive the highest attention weights from head h in layer l for token $t _ { n } ;$ and e is the average representation of these attended context tokens.

For a generated sequence r, the final ECS is averaged over all generated tokens and all selected layer-head pairs:

$$
E _ { r } = \frac { 1 } { | S _ { \mathrm { E C S } } | } \sum _ { ( l , h ) \in S _ { \mathrm { E C S } } } \frac { 1 } { | r | } \sum _ { t _ { n } \in r } E _ { n } ^ { l , h } .\tag{15}
$$

ECS characterizes the model’s reliance on external contextual information during generation by analyzing internal representations and attention flow, estimating the correlation between the generated output and the provided context.

## B.4 Parametric Knowledge Score (PKS) Computation

We adopt the Parametric Knowledge Score (PKS) from ReDeEP [28]. Since knowledge editing literature indicates that factual knowledge is predominantly localized in the later feed-forward network (FFN) layers, we restrict PKS computation to the final 10% of model layers. Let $M = \lceil 0 . 1 L \rceil$ . The deterministic layer selection set is defined as:

$$
S _ { \mathrm { P K S } } = \{ l | l \in \{ L - M + 1 , \ldots , L \} \}\tag{16}
$$

For a token $t _ { n }$ and a layer $l \in S _ { \mathrm { P K S } }$ , the token-level PKS is defined using the Jensen-Shannon Divergence (JSD) between the output distributions projected from the hidden states before and after the FFN block:

$$
P _ { n } ^ { l } = \mathrm { J S D } \left( q ( x _ { n } ^ { \mathrm { m i d } , l } ) \parallel q ( x _ { n } ^ { l } ) \right)\tag{17}
$$

where the projection function q maps a hidden state to a probability distribution over the vocabulary:

$$
q ( x ) = \mathrm { s o f t m a x } \bigl ( \mathrm { L o g i t L e n s } ( x ) \bigr )\tag{18}
$$

Here, $x _ { n } ^ { \mathrm { m i d } , l }$ denotes the residual stream representation immediately before the FFN block of layer l; $x _ { n } ^ { l }$ denotes the representation immediately after the FFN block; LogitLens(·) denotes the mapping of hidden states to vocabulary logits via the unembedding matrix; and $\mathrm { J S D } ( \cdot )$ is the Jensen-Shannon divergence.

For a generated sequence r, the sequence-level PKS for layer l is the average over all tokens:

$$
P _ { r } ^ { l } = \frac { 1 } { | r | } \sum _ { t \in r } P _ { t } ^ { l } , \quad l \in S _ { \mathrm { P K S } }\tag{19}
$$

PKS quantifies the extent to which the model’s parametric knowledge influences the output; a high PKS indicates a significant change in the output distribution induced by the FFN block, reflecting reliance on memorized information rather than purely contextual cues.

## B.5 Shared Inference and Metric Settings

Unless otherwise specified, all LLM calls use temperature 1.0, top-p 1.0, and a maximum generation length of 512 new tokens. We sample one generation per prompt and do not use greedy decoding. Models are evaluated without quantization. ECS and PKS are computed on the complete generated token sequence and averaged over all generated tokens.

## C Detailed Experimental Setup for Behavioral Probe Analysis

This appendix provides experimental details and reproducibility information for the analysis of ECS and PKS as behavioral probes across node-level, multi-source aggregation, chain-reasoning, and GAIA tool-augmented settings, as described in Section 3 of the main paper.

## C.1 Node-Level Response Experiment

Dataset. The node-level experiment employs all 1,513 cases from the celebrity\_relations dataset in the reversal\_curse repository [53]. This dataset contains kinship questions about public figures (e.g., “Who is [Person A]’s mother?”) where the correct answer is a named entity. For each question, four context conditions are constructed as described in Section 3.1 of the main paper: bubble, rel, oracle, and answer. All four conditions are evaluated for every question, and Figure 1 aggregates scores at the generation level across prompt types. The exact phrasing of each context fragment follows the templates provided in the original reversal\_curse benchmark.

Context Construction. The four context conditions are designed to form an ordered usefulness gradient. The bubble and rel contexts are generated by an LLM. The bubble context is a short, self-contained paragraph unrelated to the question, while the rel context provides general background about the person mentioned in the question but excludes family-relation information. In contrast, oracle and answer contexts are constructed directly from the ground-truth relation: oracle provides inverse relational evidence that supports reverse inference, and answer directly states the target relation.

The prompt used to generate bubble contexts is:

You will be given a question. Your task is NOT to answer it.

Instead, generate a short paragraph (20-30 words) that is completely unrelated to the question.

Requirements:   
- Do NOT answer the question   
- Do NOT mention or reference any part of the question   
- Do NOT use similar keywords or topics from the question   
- The paragraph should be self-contained and on a totally different topic

Question:

The prompt used to generate rel contexts is:

You will be given a question about a well-known person.

Your task is to generate a short paragraph (20-30 words) that provides general background information about the person mentioned in the question.

Requirements:

\- The paragraph should read like a neutral biographical description

\- Focus on aspects such as career, achievements, public image, or general life background

\- Do NOT include any information about family relationships (e.g., parents, children, siblings)

\- Do NOT answer or help answer the question

\- If you are not certain about factual details, you may generate plausible but realistic information

The goal is to produce text that appears informative and relevant to the person, but is not useful for answering the question.

Question:

Table 2 shows one representative example.

Table 2: Example context conditions for the node-level response experiment.
<table><tr><td>Field</td><td>Text</td></tr><tr><td>Question</td><td>Who is Kate Mara&#x27;s father?</td></tr><tr><td>bubble</td><td>The rural landscape bloomed with vibrant wildflowers, each petal soaking up the sun&#x27;s warmth. Farmers tended to their fields, hopeful for a bountiful harvest under the clear blue sky.</td></tr><tr><td>rel</td><td>Kate Mara is an American actress renowned for roles in film and television, including “House of Cards&quot; and “Captain America: The First Avenger.&quot;She is also recognized for her</td></tr><tr><td>oracle</td><td>fashion industry involvement and philanthropic work. Kate Mara is Timothy Christopher Mara&#x27;s Child.</td></tr><tr><td>answer</td><td>Timothy Christopher Mara is Kate Mara&#x27;s father.</td></tr></table>

Table 3: Statistical summary for the node-level response experiment in Figure 1. Mean adjacent Cliff’s δ averages the three ordered comparisons bubble→rel, rel→oracle, and oracle→answer. Confidence intervals are 95% bootstrap intervals, and p-values are from paired permutation Spearman tests.
<table><tr><td>Metric</td><td>Mean adj.δ</td><td>95% CI</td><td>Spearman ρ</td><td>p</td></tr><tr><td>ECS</td><td>0.624</td><td>[0.610, 0.640]</td><td>0.857</td><td> $< 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>PKS</td><td>-0.157</td><td>[-0.170, -0.143]</td><td>-0.308</td><td> $< 2 \times 1 0 ^ { - 4 }$ </td></tr></table>

Model Configuration. Experiments were conducted on two models: Qwen-2.5-32B-Instruct and Llama-3-70B-Instruct. The figures presented in the main paper correspond to results from Qwen-2.5- 32B-Instruct; identical qualitative conclusions were obtained with Llama-3-70B-Instruct. ECS and PKS are computed per token and averaged over the generated sequence, following the method of [28] and the shared settings above.

## C.1.1 Statistical Testing

For Figure 1, we evaluate monotonic association across the ordered relevance levels bubble, rel, oracle, and answer using a paired permutation Spearman test across the 1,513 questions. To preserve the paired structure, permutations are performed within each question by shuffling the four condition labels while keeping the four metric values associated with that question fixed. We also report adjacent-condition Cliff’s δ with bootstrap confidence intervals; positive values indicate that the later condition in the ordered pair has larger metric values. Table 3 reports the statistics for the plotted prompt set.

The pairwise adjacent comparisons further show that ECS increases reliably at every step of the relevance ordering. Cliff’s δ is 0.772 for bubble→rel (95% CI [0.750, 0.794]), 0.972 for rel→oracle ([0.965, 0.977]), and 0.129 for oracle→answer ([0.088, 0.171]). PKS exhibits a less strictly local pattern but a significant overall decline with relevance: the largest decreases occur from rel→oracle (δ = −0.211, 95% CI [-0.250, -0.171]) and oracle→answer (δ = −0.294, 95% CI [-0.333, -0.255]).

## C.2 Multi-Source Aggregation Experiment

Dataset and Task. The expert-specialization experiment is conducted on the MMLU test set [57] (Massive Multitask Language Understanding). We randomly sample 300 questions without balancing across topic groups and group subjects into seven broader topic categories to analyze how different role-defined experts are utilized across knowledge domains. The topic groups are summarized in Table 4.

For each sampled question, multiple experts with predefined roles first generate independent responses, which are concatenated and provided to a downstream Critic node for evaluation. A Direct Answer Expert is always included as a domain-agnostic baseline that directly answers the question. The Critic subsequently passes its assessment to a Final Decision node that produces the final answer. We compute segment-level ECS for each expert-output segment in the concatenated Critic input, measuring how strongly the Critic behaviorally relies on different experts under each topic group.

Table 4: Topic grouping of MMLU subjects used in the expert-specialization analysis. The first column gives the aggregated topic group used on figure axes, and the second column lists the original MMLU subjects assigned to that group. This grouping enables cross-domain comparison of expertconditioned ECS responses.
<table><tr><td>Topic Group</td><td>Included Subjects</td></tr><tr><td>Math</td><td>abstract algebra, college mathematics, elementary mathematics, high school mathematics, high school statistics, formal logic</td></tr><tr><td>Computer Science</td><td>college computer science, high school computer science, computer security, machine learning, electrical engineering</td></tr><tr><td>Physics</td><td>college physics, conceptual physics, high school physics, astronomy</td></tr><tr><td>Chemistry / Biology / Medicine</td><td>anatomy, clinical knowledge, college biology, college chemistry, college medicine, high school biology, high school chemistry, human aging, human</td></tr><tr><td>Law / Politics / History</td><td>sexuality, medical genetics, nutrition, professional medicine, virology high school US history, high school world history, high school European</td></tr><tr><td></td><td>history, high school government and politics, international law, jurisprudence, professional law, security studies, US foreign policy, prehistory</td></tr><tr><td>Business / Economics</td><td>business ethics, econometrics, global facts, high school macroeconomics, high school microeconomics, management, marketing, professional accounting, public relations</td></tr><tr><td>Psychology / Philosophy / Social Science</td><td>high school psychology, professional psychology, philosophy, moral disputes, moral scenarios, logical fallacies, sociology, world religions</td></tr></table>

Expert Role Prompts. Expert role prompts are taken from the MMLU prompt set in the open-source Agent Prune [12] implementation. Experts do not differ in external knowledge bases, retrieval access, or function-calling abilities; their domains are defined only through role prompts. Each expert receives a role description before the shared domain-assessment instruction. For example, the Doctor role is:

You are a doctor and come up with creative treatments for   
illnesses or diseases.   
You are able to recommend conventional medicines, herbal remedies and   
other natural alternatives.   
You also consider the patient’s age, lifestyle and medical history when   
providing your recommendations.   
red expert instruction is:   
Domain Assessment: First, determine if the question falls within your   
core expertise as:   
Condition A (Within Domain):   
If the question is directly related to your field:   
Conduct a brief Step-by-Step reasoning.   
You MUST provide the final choice in the following JSON format:   
{"answer": "X"} (where X is A, B, C, or D).   
Don’t forget to provide the reasoning step in your outputs   
Condition B (Outside Domain):   
If the question is NOT related to your field:   
DO NOT attempt to solve the problem or guess the answer.   
DO NOT use the JSON format {"answer": "X"}.   
Instead, provide a sophisticated, professional "fact or reflection" from   
your own domain that is entirely irrelevant to the question.   
Act as if you are preoccupied with your own research.   
For example, if you are a Biologist faced with a Math problem,   
discuss mitochondrial DNA or protein folding instead.   
DO NOT provide any JSON format answer in your outputs   
Constraint: Your output must either contain the JSON answer (if in-domain)   
or the irrelevant professional monologue (if out-of-domain).   
Never mix the two.

Segment-Level ECS (Seg-ECS) Computation. Within a single context window containing concatenated outputs from multiple upstream nodes, the input is partitioned into contiguous segments corresponding to each source. For each segment, ECS is computed independently following the standard token-level ECS method [28], restricted to the token indices belonging to that segment.

Normalization. To enable cross-question comparison, we apply z-score normalization within each question following the same procedure used in the node pruning experiments of Section 5. Specifically, for a given question with segments $1 , \ldots , M$ , let $\mu$ and σ be the mean and standard deviation of the M segment-level ECS values. The normalized score for segment i is:

$$
\operatorname { E C S } _ { i } = { \frac { \operatorname { E C S } _ { i } - \mu } { \sigma } } .\tag{20}
$$

This per-question standardization removes question-specific scale variations and highlights relative contribution differences among competing sources. The distributions shown in Figure 2a aggregate these normalized ECS values over all expert segments from the 300 sampled MMLU questions. If the within-question standard deviation is zero, all normalized segment scores for that question are set to zero.

## C.2.1 Statistical Testing for Figure 2a

To test whether domain-matched experts receive higher Critic reliance than non-matched experts, we use each question’s Critic node as the paired unit. Of the 300 sampled MMLU questions, 268 have a clearly matched expert after excluding miscellaneous or hard-to-map categories. The Direct Answer Expert is excluded from this inferential comparison because it serves only as a visual baseline in Figure 2a; it is not treated as either a matched or non-matched domain expert.

For each valid paired unit, we compute

$$
d = \overline { { \mathrm { E C S } } } _ { \mathrm { m a t c h e d e x p e r t } } - \overline { { \mathrm { E C S } } } _ { \mathrm { o t h e r n o n - D i r e c t - A n s w e r e x p e r t s } } ,\tag{21}
$$

where the matched expert is the expert whose domain corresponds to the question category, and the other non-Direct-Answer experts are all remaining domain experts excluding Direct Answer. We then test the one-sided alternative $\overline { { d } } > 0$ across paired units. Across n = 268 valid pairs, the matched expert has a higher mean ECS than other non-Direct-Answer experts (0.3555 vs. 0.2820), with a mean paired difference of 0.0735 and a 95% bootstrap confidence interval of [0.0683, 0.0786]. A one-sided Wilcoxon signed-rank test gives $p = 1 . 6 5 \times \mathrm { \dot { 1 } 0 ^ { - 6 2 } }$ , and a sign test shows that the matched expert is higher in 247 of 268 pairs $( \bar { p } = 1 . 9 7 \times 1 0 ^ { - 5 0 } )$

## C.3 Chain Reasoning Experiment

Task Structure. This appendix corresponds to the chain-reasoning analysis in Section 3.2.2. We manually select familiar high-level taxonomic groups so that the LLMs recognize the biologicalclassification context, then generate a controlled five-level hierarchy from NCBI Taxonomy records [64]. The resulting dataset contains 32 species instances with complete ranks for Phylum (level 1), Class (level 2), Order (level 3), Family (level 4), and Genus (level 5). The goal at each step is to predict the correct genus given the accumulated taxonomic information up to the current level. In addition to the taxonomic context, the output produced by the previous chain position is passed to the next agent; thus each downstream position receives both the current taxonomic evidence and the preceding agent’s generated response.

## Incremental and Truncated Conditions.

• Full incremental condition: At step $t \in \{ 1 , \ldots , 5 \}$ , the model receives the taxonomic description of the t-th level (e.g., Step 1: “Phylum: Chordata”; Step 2: adds “Class: Mammalia”, etc.). For $t > 1$ , the prompt also includes the previous position’s generated output. The model is prompted to predict the genus at every step.

• Truncated condition with parameter k: The model receives new taxonomic information only for steps $1 , \ldots , k .$ . For all subsequent steps $t > k ,$ , the taxonomic context remains unchanged: the prompt provides the same level-k taxonomic information without advancing to deeper ranks. The previous position’s generated output is still passed forward at every step, so the truncation removes only new taxonomic evidence, not inter-agent output propagation. The model is still required to make a genus prediction at each of these later steps, allowing observation of PKS behavior when no new taxonomic information is introduced.

PKS is computed as the sequence-averaged token-level PKS [28] for each generated prediction. Each species is evaluated under the full incremental condition and all truncated conditions (k = 1, 2, 3, 4). The trajectories shown in Figure 2b are averaged over all 32 species instances.

## C.4 GAIA Tool-Augmented Reasoning Pipeline

The GAIA analysis in Section 3.3 evaluates whether ECS and PKS remain informative in a realistic tool-augmented reasoning pipeline. Experiments use a manually filtered subset of GAIA Level-1 validation with video-modality instances removed. The resulting subset contains 52 instances. The system follows a Tree-of-Thought style route with four functional stages: GenerateQuery, WebSearch, FileAnalyse, and CombineAnswer. WebSearch uses search API, FileAnalyse extracts information from associated files when available, and CombineAnswer synthesizes the collected evidence into the final answer.

The prompting setup follows the original GAIA design. The only modification is a minor instructionfollowing clarification in the WebSearch stage: the original prompt states that when the “information from web search is useless,” the model should directly answer “No useful information from WebSearch.” We additionally specify that no extra analysis needs to be produced in this irrelevantinformation case, so that the fixed phrase can be used more reliably as a relevance label.

For ECS analysis, we distinguish two segment-level analyses in the GAIA pipeline. For Figure 3b, the analyzed segments are the search-result segments returned by the function-calling tool during WebSearch. We label each search-result segment using the corresponding WebSearch analysis output: segments whose analysis output contains the fixed phrase “No useful information from WebSearch” are labeled irrelevant, while the remaining segments are treated as useful retrievals. We then compute segment-level ECS during WebSearch generation, measuring the reliance of the generated WebSearch analysis on each search-result segment.

For Figure 3a, we analyze evidence consumed by the downstream CombineAnswer node. Decisioncriticality is defined branch-wise and estimated using two independent single-branch ablations. In the WebSearch ablation, we remove only WebSearch-derived evidence while keeping FileAnalyse evidence unchanged; an instance is marked WebSearch-critical only if the baseline answer is correct and the answer becomes incorrect after removing WebSearch evidence. Similarly, in the FileAnalyse ablation, we remove only FileAnalyse-derived evidence while keeping WebSearch evidence unchanged; an instance is marked FileAnalyse-critical only if the baseline answer is correct and the answer becomes incorrect after removing FileAnalyse evidence. The branch that remains unchanged in each ablation is treated only as background context for that ablation, and is not assigned criticality from the outcome of the other branch’s removal.

In the original run, the baseline ToT pipeline answers 11 of 52 instances correctly, corresponding to 21.2% accuracy. Removing WebSearch evidence identifies three WebSearch-critical cases and reduces accuracy to 8 of 52 instances; removing FileAnalyse evidence identifies three FileAnalyse-critical cases and also reduces accuracy to 8 of 52 instances. These two critical-evidence sets do not overlap, yielding six RAG-critical cases in total.

Under the threshold-based pruning analysis reported in the main text, removing evidence above the approximately 40th-percentile ECS threshold (ECS ≈ 0.35) reproduces the same 11/52 to 8/52 drop for each branch, whereas removing evidence below the threshold leaves the number of correct answers unchanged. For PKS analysis, we compare the CombineAnswer stage before and after WebSearch evidence ablation, measuring whether the absence of retrieved evidence increases reliance on parametric knowledge. The same prompt templates, tool-call formatting, and answer filtering rules are kept fixed between the original and ablated runs, so that observed metric shifts can be attributed to the presence or absence of the ablated evidence.

## C.4.1 Statistical Testing for Figure 3

For Figure 3, unpaired ECS comparisons use label-permutation tests and percentile bootstrap confidence intervals for mean differences. Panel (a) uses the merged RAG-critical labeling described above, which contains six non-overlapping critical cases across WebSearch and FileAnalyse. The CombineAnswer PKS comparison in panel (c) is paired across the 52 GAIA instances; we therefore report a paired bootstrap confidence interval, an exact two-sided Wilcoxon signed-rank test, and an exact two-sided sign test. Table 5 summarizes the results.

Table 5: Statistical summary for the GAIA tool-augmented reasoning experiment in Figure 3. Confidence intervals are 95% bootstrap intervals for mean differences. Positive differences indicate larger ECS for critical RAG evidence in panel (a), larger ECS for relevant WebSearch segments in panel (b), and larger PKS after removing WebSearch evidence in panel (c).  
Panel Comparison Means Mean diff. Additional tests   
Fig. 3a RAG crit. vs. non-crit. ECS 0.512 vs. 0.362 ∆µ = 0.150 [0.052, 0.246], p = 0.066 Cliff’s δ = 0.451, p = 0.074   
Fig. 3b Rel. vs. irrel. ECS 0.372 vs. 0.129 ∆µ = 0.243 [0.194, 0.285], p < 10<sup>−5</sup> Cliff’s δ = 0.974, p < 10<sup>−5</sup>   
Fig. 3c No-WS vs. WS PKS 1.330 vs. 1.116 ∆µ = 0.214 [0.114, 0.309] Wilcoxon W = 1118, p = 4.75 × 10<sup>−5</sup>; sign k = 41/52, p = 3.59 × 10<sup>−5</sup>

## D Conceptual Foundations and Interpretive Assumptions

This appendix clarifies the conceptual underpinnings of the heuristic interpretation presented in Section 4. We explicitly state the working assumptions that link ECS and PKS to information sources, and we provide a constructive counterexample illustrating why PKS decline alone does not guarantee correctness [66–68]. No formal mathematical proofs are claimed; the content below serves as an interpretive scaffold rather than a rigorous theory.

## D.1 Heuristic Mapping from Behavioral Probes to Information Sources

The core interpretive assumption of this work is a qualitative monotonic relationship between the behavioral probes and the information sources they reflect:

Assumption (Heuristic Monotonicity). For a given node v, the External Context Score (ECS) tends to increase as the node depends more on the provided context $Z _ { v }$ . Conversely, the Parametric Knowledge Score (PKS) tends to decrease as the node depends less on its internal parametric knowledge $\Theta _ { v }$

This assumption is not derived from first principles. It is an empirical generalization motivated by the controlled experiments in Section 3:

• ECS increases with input relevance (Figure 1a).

• PKS decreases as informative context accumulates (Figure 2b).

We do not assert a precise functional form, such as proportionality, between these probes and any information-theoretic quantity. The value of this assumption lies in providing a consistent heuristic interpretation for examining information flow across interacting nodes and for designing probe-guided variants of structure optimization. It should be viewed as an empirical interpretation, rather than a formal mechanism or theorem.

## D.2 Why PKS Decline Does Not Imply Correctness: A Constructive Counterexample

A monotonic decrease in PKS reflects perceived information sufficiency, but it does not guarantee that the accumulated information is factually correct. We illustrate this with a simple counterexample.

Scenario. Consider a two-step reasoning chain where the final answer is a binary choice, such as Yes/No. The true answer is “Yes.” At Step 1, the model receives uninformative context and produces a guess with moderate PKS. At Step 2, the model receives a context that confidently but incorrectly asserts “No.” Because the context is now specific and internally consistent, the model’s output becomes more determined by the misleading context. As a result:

• The model’s parametric-knowledge dependence drops sharply ⇒ PKS declines.

• The final output is $\mathrm { } ^ { 6 6 } \mathrm { N o } ^ { 3 3 } \Rightarrow$ the answer is wrong.

This scenario demonstrates that PKS decline reflects reduced parametric-knowledge dependence, not the correctness of the context. A declining PKS indicates that the model has become more settled under the provided context, not that the answer is correct. This is why, in our reasoning-path search experiment (Section 5.2), we combine the PKS-based convergence probe with utility signals.

![](images/95c656070ef4f834951bd2b0db98f8ea85f2ad59fd336e57cea27bee37a239f5.jpg)  
Figure 5: ECS–PKS behavioral profiles of functional categories in the GAIA dataset. The x-axis is PKS, the y-axis is ECS, colors denote functional log categories, and black crosses mark category centers. The profile clusters align with functional roles, supporting a task-dependent region $\Omega _ { \mathrm { t a s k } }$ for probe-guided pruning.

## D.3 GAIA ECS–PKS Behavioral Profiles as Empirical Support for Task-Dependent Regions

This subsection provides additional context for the empirical observations on the GAIA dataset referenced in Section 4. Using trajectories from the ToT reasoning pipeline, we collect nodelevel ECS and PKS values for five functional log categories: TotalQuery, WebQuery, FileAnalyse, WebAnalyse, and AdversarialAnswer. Here, WebQuery and WebAnalyse are aliases for two internal substeps of the same WebSearch node: WebQuery denotes query construction before the function call, while WebAnalyse denotes summarization and analysis of the function-call results returned by WebSearch. They are not separate nodes in the swarm topology; throughout the main paper and other appendices, this stage is referred to uniformly as WebSearch.

As shown in Figure 5, functional categories associated with distinct roles occupy separable regions in the ECS–PKS profile space:

• WebAnalyse alias entries, corresponding to the analysis substep of the WebSearch node, synthesize externally retrieved evidence and show high ECS with relatively low PKS.

• AdversarialAnswer nodes, which deliberately generate irrelevant responses, show low ECS and elevated PKS.

• FileAnalyse nodes, which process structured inputs and perform reasoning, show both high ECS and high PKS.

These clusters support the view that ECS–PKS behavioral profiles capture meaningful role-dependent patterns. Moreover, outliers, such as WebAnalyse alias entries with unusually low ECS, correspond to cases where the upstream retrieval is ineffective, forcing the WebSearch node to rely more on parametric knowledge. This empirical pattern supports the use of a task-dependent region $\Omega _ { \mathrm { t a s k } }$ and the corresponding probe-guided pruning criterion discussed in the main text.

## E Detailed Setup for Probe-Guided Structure-Optimization Applications

This appendix provides detailed task descriptions and task-specific reward definitions for the two probe-guided structure-optimization applications in Section 5: random-DAG node pruning and NCBI-derived taxonomic reasoning-path search.

## E.1 Random-DAG Node-Pruning Setup

The node-pruning experiment uses the MMLU development set filtered according to the GPTSwarm configuration, yielding 153 questions. All DA, LA, BA, and FD agents use the same backbone within a run, either Qwen-2.5-32B-Instruct or Llama-3-70B-Instruct. The graph contains two DA source nodes, two LA nodes, two BA nodes, and one FinalDecision (FD) sink node. DA nodes serve as fixed sources and FinalDecision serves as the fixed sink; beyond these input/output roles, we do not impose a chain, tree, or hand-designed intermediate topology. Instead, all admissible acyclic candidate connections among DA, LA, BA, and FD nodes are initialized with equal edge probability 0.5. This creates a controlled but nontrivial random-DAG search problem: the reward is the final 0/1 correctness signal, but many sampled paths pass through redundant LA/BA nodes or misleading intermediate combinations, while the target communication backbone is the direct DA→FD route. Training uses the 200-iteration, learning-rate 0.1 protocol specified in the node-pruning reward definition below. At test time, edges are sampled according to the learned weights rather than deterministically thresholded.

## E.1.1 Node Roles and Output Protocol

The four node labels used in the pruning graph correspond to DirectAnswer (DA), LazyAnswer (LA), BubbleAnswer (BA), and FinalDecision (FD). DA is the useful source type: it receives the question directly and is expected to solve it. LA is a redundant answer-preserving node: it receives an existing reference answer and is constrained to reproduce the same answer content with only minimal surface-level paraphrasing. BA is a distractor node: it produces confident but question-irrelevant content while still following the required answer format. FD is the sink node that aggregates the available upstream messages and produces the final multiple-choice decision.

The DA role prompt is intentionally minimal:

You are a knowledgeable expert in question answering.

The LA instruction below summarizes the constraint used in implementation:

You are a strict paraphrasing agent.

Your only task is to reproduce the reference answer exactly in meaning, reasoning process, structure, and format, including the JSON answer section. Do not add, remove, summarize, reorder, or reinterpret any content. You may only make minimal surface-level paraphrases; if a paraphrase would change the meaning or structure, copy the reference answer verbatim. Output only the resulting answer.

The BA instruction is:

Generate 1-2 confident sentences that are completely unrelated to the   
question with the proper answer format.

All answer-producing nodes are required to provide reasoning together with a final answer, and the final answer must appear in JSON form:

```json
{"answer": "<FINAL_ANSWER>"}
```

Here, <FINAL\_ANSWER> must be one of the options in the question, usually a single letter. This shared output constraint ensures that DA, LA, BA, and FD remain format-compatible even when their informational roles differ, allowing topology optimization to distinguish useful direct-answer paths from redundant or irrelevant paths by their behavioral metric profiles rather than by parser failures.

## E.1.2 Random-DAG Sampling Procedure

Candidate edges are defined by the input/output constraints above: the two DA nodes are source nodes, the FinalDecision node is the sink, and LA or BA nodes may be inserted through any directed connections that preserve acyclicity. All candidate edge logits are initialized so that the initial sampling probability is 0.5.

For each candidate edge $( u , v )$ , we first check whether adding $u  v$ preserves acyclicity. If so, the edge is sampled from a Bernoulli distribution with probability

$$
p _ { u v } = \sigma ( \theta _ { u v } / \tau ) ,
$$

where $\theta _ { u v }$ is the learnable edge logit and τ is the sampling temperature. The realization log-probability accumulates log $p _ { u v }$ for selected edges and log $\left( 1 - p _ { u v } \right)$ otherwise. After this first pass, we apply a predecessor-completion step to enforce the source-node constraint that the two DA nodes are the only sources. For every non-input node with zero in-degree, we randomly select one valid predecessor from its candidate incoming edges without violating acyclicity and force this edge to be included. The corresponding log-probability term is then replaced by the selected-edge term $\log p _ { u v }$ . This yields a realized DAG in which every non-input node has at least one predecessor, while only the two DA nodes can have zero in-degree.

## E.1.3 Computation of Edge Weight Metrics

To quantify the learned communication topology, we measure the cumulative inbound edge weight to the final decision node (FD). Given a trained swarm graph with edge set $\mathcal { E } ,$ each directed edge $( u , v ) \in \mathcal { E }$ is associated with a learned non-negative weight $w _ { u , v }$ that reflects the communication strength between two agents.

For each intermediate node type $s \in \{ \mathrm { D A } , \mathrm { L A } , \mathrm { B A } \}$ , we aggregate all edges from nodes of type s that terminate at FD:

$$
W _ { s } = \sum _ { ( u  \mathrm { F D } ) , u \in s } w _ { u , \mathrm { F D } } .
$$

Here, $W _ { s }$ represents the total contribution intensity of node type s to the final decision process. Specifically, higher $W _ { \mathrm { D A } }$ indicates stronger reliance on useful direct-answer agents, whereas lower $W _ { \mathrm { L A } }$ and $W _ { \mathrm { B A } }$ indicate that redundant or misleading communication paths have been successfully suppressed.

Following this definition, the total redundant communication weight is computed as:

$$
W _ { \mathrm { r e d u n d a n t } } = W _ { \mathrm { L A } } + W _ { \mathrm { B A } } .
$$

All values reported in Table 1 are averaged over evaluation samples after training convergence.

## E.2 NCBI-Derived Taxonomic Labyrinth for Path Selection

Task Structure. This appendix corresponds to the reasoning-path search experiment in Section 5.2. We use the same 32-instance NCBI-derived biological-classification set as the chain-reasoning analysis in Section 3.2.2. Each instance contains a complete five-level taxonomy: Phylum (level 1), Class (level 2), Order (level 3), Family (level 4), and Genus (level 5). Unlike the chain-reasoning analysis, where information is provided along a fixed sequence, we convert each instance into a path-selection problem over five ordered reasoning positions. The goal is to sample a reasoning path that follows the correct taxonomic progression and eventually predicts the correct genus.

Path-Sampling Construction. For each instance, we create five ordered reasoning positions, denoted as $\mathrm { p o s } _ { 1 } , \ldots , \mathrm { p o s } _ { 5 }$ , together with a terminal node T. Each reasoning position corresponds to one taxonomic level and simulates one step of hierarchical refinement. For example, pos provides the highest-level clue such as “Phylum: Chordata”; pos should refine this with the class-level clue such as “Class: Mammalia”; pos should add the order-level clue; $\mathrm { p o s } _ { 4 }$ should add the family-level clue; and $\mathrm { p o s } _ { 5 }$ corresponds to the genus-level clue.

At each iteration, a candidate reasoning path is sampled as a non-repeating Markov chain. Sampling always starts from $\mathrm { p o s } _ { 1 }$ . Given the current position, the next node is sampled from the remaining unvisited reasoning positions and the terminal node according to a softmax distribution over outgoing edge weights. The sampling process stops once T is reached. The terminal node does not participate in reasoning; it only serves as a stopping marker. The final utility is computed from the answer produced by the last sampled reasoning position before T. Under this construction, the complete correct reasoning chain is $\mathrm { p o s } _ { 1 }  \mathrm { p o s } _ { 2 }  \mathrm { p o s } _ { 3 }  \mathrm { p o s } _ { 4 }  \mathrm { p o s } _ { 5 }  \mathrm { T }$

Taxonomic context is accumulated and passed downstream only when the sampled reasoning positions follow the correct order. If the order is broken, downstream nodes receive only the initial highest-level taxonomic clue, rather than the intermediate clues needed for stepwise refinement. For example, a path that reaches $\mathrm { p o s } _ { 3 }$ without first visiting $\mathrm { p o s } _ { 2 }$ does not receive the accumulated class-level context; similarly, a path that jumps from pos to pos cannot access the missing class- and order-level clues. This construction makes the task harder when the sampled path deviates from the intended taxonomy, while keeping the input format controlled across candidate paths. Thus, only the fully ordered sampled chain reaches the complete taxonomic progression needed for reliable genus prediction.

Optimization Setup. Optimization starts from edge weight 0.5 and runs for 200 iterations with learning rate 0.1. Each reported group uses three random seeds. At each iteration, a non-repeating path is sampled using the softmax Markov-chain procedure described above. The utility reward is 1 if the final genus prediction from the last reasoning position before T is correct and 0 otherwise. For the probe-guided utility setting, optimization first uses utility-only rewards for 50 warm-up iterations, after which the PKS-based convergence probe is introduced. The detailed PKS-based reward design is provided in Appendix E.3.3.

## E.3 Reward Function Definitions for Probe-Guided Structure Optimization

This subsection provides the common reward-conversion procedure used by the two probe-guided structure-optimization studies, followed by the task-specific reward definitions for Section 5.1 and Section 5.2. EMA smoothing reduces single-run probe noise, robust normalization compares an edge- or node-level quantity against an appropriate comparison group, and the piecewise function maps relative deviation to a bounded reward. The semantic interpretation is determined by the input quantity $x _ { e } { \mathrm { : } }$ node pruning uses absolute ECS–PKS behavioral profiles, while reasoning-path search uses the PKS drop as a convergence signal.

## E.3.1 Shared Reward Conversion

Exponential Moving Average (EMA) Smoothing. For a generic metric $M \ \in \ \{ \mathrm { E C S } , \mathrm { P K S } \}$ associated with edge e, the smoothed value at training iteration t is computed as:

$$
\tilde { M } _ { e } ^ { ( t ) } = \beta \tilde { M } _ { e } ^ { ( t - 1 ) } + ( 1 - \beta ) M _ { e } ^ { ( t ) } ,\tag{22}
$$

where $\beta \in [ 0 , 1 )$ is the smoothing coefficient. $\mathrm { A t } ~ t = 0 , \tilde { M } _ { e } ^ { ( 0 ) }$ is initialized with the first observed value $M _ { e } ^ { ( 1 ) }$

We linearly increase $\beta$ from 0.4 to 0.9 over the first 50 warmup iterations and keep it fixed at 0.9 afterwards.

Robust Normalization via Median and MAD. Inspired by the GRPO framework[69], we standardize metrics within a comparison group using robust statistics. Let ${ \mathcal { G } } = \{ x _ { 1 } , \ldots , x _ { K } \}$ denote a set of smoothed metric values. The normalized score for $x _ { i }$ is:

$$
z _ { i } = \frac { x _ { i } - \mathrm { m e d i a n } ( \mathcal { G } ) } { c _ { s } \cdot \mathrm { M A D } ( \mathcal { G } ) } ,\tag{23}
$$

where $\mathbf { M A D } ( { \mathcal { G } } ) =$ median $\langle \{ | x _ { j } - \mathrm { m e d i a n } ( \mathcal { G } ) | \} _ { i = 1 } ^ { K } \rangle$ is the median absolute deviation, and $c _ { s } =$ 1.4826 ensures that $c _ { s } \mathrm { ~ } { \cdot } \mathrm { M A D }$ is a consistent estimator of the standard deviation for normally distributed data.

Piecewise Reward Function. Given a normalized score $z ,$ the edge-level reward is computed via a piecewise function that amplifies significant deviations from the group median while suppressing near-median variations:

$$
g ( z ) = \left\{ \begin{array} { l l } { - r _ { \mathrm { m i n } } - { \left( r _ { \mathrm { m a x } } - r _ { \mathrm { m i n } } \right) } \operatorname { t a n h } { \left( - { \left( z + \tau \right) } \right) } , } & { z < - \tau } \\ { 0 , } & { | z | \leq \tau } \\ { r _ { \mathrm { m i n } } + { \left( r _ { \mathrm { m a x } } - r _ { \mathrm { m i n } } \right) } \operatorname { t a n h } { \left( z - \tau \right) } , } & { z > \tau } \end{array} \right.\tag{24}
$$

where $\tau > 0$ defines a “dead zone” around the median, and $r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } > 0$ control asymptotic penalty and reward magnitudes. The hyperbolic tangent tanh(·) provides smooth saturation to extreme values. We set $\tau = 1 . 5$ and $r _ { \mathrm { m i n } } = r _ { \mathrm { m a x } } = 0 . 5$ in all structure-optimization experiments.

## E.3.2 Section 5.1 Node-Pruning Reward

For a downstream node n with predecessor set ${ \mathcal { P } } ( n )$ , we first compute smoothed ECS values $E \tilde { C } S _ { p _ { i }  n }$ for each incoming edge. Let $\mathcal { G } _ { n } = \{ E \tilde { C } S _ { p _ { i }  n } \}$ be the local comparison group. The normalized ECS score $z _ { e } ^ { E }$ is computed via robust normalization within ${ \mathcal { G } } _ { n }$

We then apply a two-step reward design.

Step 1: ECS gating. $\mathrm { I f } ~ z _ { e } ^ { E } < 0 .$ the edge is treated as context-irrelevant for the downstream node. It receives only the ECS penalty and is excluded from the subsequent PKS normalization and reward computation. Only edges with $\dot { z } _ { e } ^ { E } \ge 0$ enter the second step.

Step 2: PKS gating. For edges that pass the ECS gate, we pool their smoothed absolute PKS values to form the global comparison group $\bar { \boldsymbol { { g } } } _ { \mathrm { s w a r m } } .$ , and compute the normalized PKS score $z _ { e } ^ { P } . \mathrm { I f } z _ { e } ^ { P } < 0 .$ the edge is treated as a redundant aggregation path. It receives only the PKS penalty, and any positive ECS reward is ignored.

The resulting probe-derived pruning reward is:

$$
R ( e ) = \left\{ \begin{array} { l l } { \lambda _ { \mathrm { E C S } } g ( z _ { e } ^ { E } ) , } & { z _ { e } ^ { E } < 0 , } \\ { \lambda _ { \mathrm { P K S } } g ( z _ { e } ^ { P } ) , } & { z _ { e } ^ { E } \geq 0 \mathrm { a n d } z _ { e } ^ { P } < 0 , } \\ { \lambda _ { \mathrm { E C S } } g ( z _ { e } ^ { E } ) + \lambda _ { \mathrm { P K S } } g ( z _ { e } ^ { P } ) , } & { z _ { e } ^ { E } \geq 0 \mathrm { a n d } z _ { e } ^ { P } \geq 0 . } \end{array} \right.\tag{25}
$$

Here, $z _ { e } ^ { P }$ is computed only for edges that pass Step 1. This design forces an edge to satisfy both criteria: context-irrelevant edges are filtered before PKS profiling, and low-PKS redundant paths cannot be retained solely by a positive ECS score.

In the final implementation, this probe-derived edge reward is combined with the output-driven task utility by direct addition. Let $\bar { U _ { \tau } } \in \{ 0 , 1 \}$ denote the final-answer utility for task $\tau _ { \ast }$ , with $U _ { \tau } = 1$ when the sampled graph produces the correct output and 0 otherwise. The reward used for updating the probability of edge e is:

$$
R _ { \mathrm { e d g e } } ( e ) = U _ { \tau } + R ( e ) .\tag{26}
$$

Thus, the ECS/PKS reward provides edge-local pruning guidance, while the 0/1 utility keeps the update tied to final output correctness. When a pruning loss is used, it is optimized in parallel with this reward-based edge-probability update and serves as a complementary structural regularizer rather than a replacement for either the output utility or the probe-derived reward.

Edge weights $w _ { e }$ are updated via gradient ascent on cumulative reward. In our experiments, we use soft pruning only: no hard pruning threshold ϵ is applied, and evaluation samples edges stochastically according to the learned weights. Unless otherwise stated, $\lambda _ { \mathrm { E C S } } = \lambda _ { \mathrm { P K S } } = 0 . 5 .$ training runs for 200 iterations, and the learning rate is 0.1.

## E.3.3 Section 5.2 Path-Selection Reward

In the path optimization setting, the PKS reward is based on the PKS drop

$$
\Delta \tilde { P } _ { u  v } = P \tilde { K } S _ { u } - P \tilde { K } S _ { v } ,\tag{27}
$$

where u and v are consecutive reasoning nodes. The two PKS values are obtained before and after incorporating the information passed along $u  v ,$ , then smoothed using the shared EMA procedure described above. Larger positive drops indicate that the successor becomes less dependent on parametric memory after receiving the upstream information. We normalize these drops locally among successor edges of the same current node using the shared robust-normalization rule, and convert the normalized score into an edge-level process reward:

$$
R _ { \mathrm { P K S - d r o p } } ( u \to v ) = g \mathopen { } \mathclose \bgroup ( z _ { \Delta \tilde { P } _ { u  v } } \aftergroup \egroup ) .\tag{28}
$$

In the final Metric+Utility implementation, this process reward is combined with the output-driven utility by direct addition. Let $\dot { U } _ { \tau } \in \{ 0 , 1 \}$ denote the final-path utility for task $\tau ,$ with ${ U _ { \tau } } ^ { - } = 1$ if the sampled path produces the correct genus prediction and 0 otherwise. The edge reward used to update the edge probability is:

$$
R _ { \mathrm { e d g e } } ( u  v ) = U _ { \tau } + R _ { \mathrm { P K S - d r o p } } ( u  v ) .\tag{29}
$$

Thus, the PKS-drop term supplies edge-local process credit for convergence, while the 0/1 utility anchors optimization to final output correctness. In the Utility Only baseline, the PKS-drop term is omitted and edge updates rely only on $U _ { \tau }$ . This defines the reward used in the Section 5.2 path-selection experiment.

## F Details of Baseline-Inspired Loss Implementations

This section provides the formal definitions of the two baseline-inspired losses used in our experiments. The GPTSwarm baseline follows the original configuration from [36] as a task-utility-based topology optimizer. For Agent Prune and Agent Net, which represent gradient-sensitivity pruning and trajectory-credit allocation, we do not claim strict reproduction of the full original systems; instead, we implement pruning- and trace-inspired losses under the same edge-level log-probability parameterization log $p _ { e }$ used by our swarm graph. This keeps the optimization interface comparable while preserving the core structural-credit intuition of each method.

## F.1 Agent-Prune- and Agent-Net-Inspired Losses

Agent Prune. The Agent-Prune-inspired loss constructs edge importance using gradient-based sensitivity of the task loss. Let $L _ { \mathrm { t a s k } }$ denote the base policy loss. The gradient with respect to each edge log-probability is:

$$
g _ { e } = \frac { \partial L _ { \mathrm { t a s k } } } { \partial \log p _ { e } } .\tag{30}
$$

The edge importance is defined as:

$$
I _ { e } = \left| \log p _ { e } \cdot g _ { e } \right| .\tag{31}
$$

A sparsity-regularized reward is then constructed:

$$
r _ { e } ^ { \mathrm { p r u n e } } = \lambda _ { I } I _ { e } - \lambda _ { s } ,\tag{32}
$$

where $\lambda _ { I }$ controls the importance scaling and $\lambda _ { s }$ enforces uniform sparsity pressure across edges. We set $\lambda _ { I } = 1 . 0$ and $\lambda _ { s } = 0 . 0 5$

The policy is optimized via:

$$
\mathcal { L } _ { \mathrm { p r u n e } } = - \sum _ { e } r _ { e } ^ { \mathrm { p r u n e } } \log p _ { e } .\tag{33}
$$

Agent Net. The Agent-Net-inspired loss assigns edge-level credit via a reward trace mechanism. Let $\mathcal { E } _ { t }$ denote the set of activated edges at step $t ,$ and $R _ { t }$ be the trajectory reward.

A decayed trace is maintained as:

$$
T _ { e } ^ { ( t ) } = \gamma T _ { e } ^ { ( t - 1 ) } + \mathbb { I } ( e \in \mathcal { E } _ { t } ) \cdot R _ { t } ,\tag{34}
$$

where $\gamma \in ( 0 , 1 )$ is a decay factor.

The trace is normalized across edges:

$$
\tilde { T } _ { e } = \frac { T _ { e } - \operatorname* { m i n } ( T ) } { \operatorname* { m a x } ( T ) - \operatorname* { m i n } ( T ) + \epsilon } .\tag{35}
$$

The centered credit signal is defined as:

$$
g _ { e } = \tilde { T } _ { e } - \frac { 1 } { 2 } .\tag{36}
$$

This signal modulates the policy update:

$$
\mathcal { L } _ { \mathrm { { n e t } } } = - \sum _ { e } \left( \mathrm { s t o p g r a d } ( g _ { e } ) \cdot \log p _ { e } \right) ,\tag{37}
$$

which is combined with the task objective:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \alpha \mathcal { L } _ { \mathrm { n e t } } . } \end{array}\tag{38}
$$

We set $\alpha = 1 . 0$

## G Compute Resources

The original experiments were run on a server with 8 NVIDIA A800 GPUs. Reproducing the full set of reported experiments is estimated to require approximately four days on comparable hardware. The main computational cost comes from repeated model inference, probe computation, and the probe-guided optimization runs. This estimate covers the original experiments and does not include preliminary exploratory runs or failed pilot experiments.

## H Additional Evaluations and Robustness Analyses

These supplementary evaluations retain the shared settings in Appendix B, except for the explicitly varied decoding and probe selections.

## H.1 Simpler Signals and Configuration Robustness

Table 6 compares simpler signals with ECS and PKS on the same 6,052 single-node records (1,513 questions, four conditions), with signal directions fixed before comparison.

Table 6: Comparison of simpler signals with ECS and PKS on the same 6,052 single-node records.
<table><tr><td>Signal</td><td>Ordering consistency</td><td>Aligned Spearman</td></tr><tr><td>ECS comparison</td><td></td><td></td></tr><tr><td>Raw attention mass</td><td>0.4152</td><td>-0.2897</td></tr><tr><td>Length-normalized attention</td><td>0.9755</td><td>0.9170</td></tr><tr><td>ECS</td><td>0.9147</td><td>0.8579</td></tr><tr><td>PKS comparison</td><td></td><td></td></tr><tr><td>Entropy</td><td>0.7437</td><td>0.4868</td></tr><tr><td>Log-probability</td><td>0.7368</td><td>0.4710</td></tr><tr><td>PKS</td><td>0.6499</td><td>0.2885</td></tr></table>

Length-normalized attention is competitive with ECS, whereas raw attention mass performs substantially worse. Entropy and log-probability discriminate these conditions more strongly than PKS. These single-node results do not establish a universal ranking across swarm settings.

Table 7: Robustness to decoding settings. Stochastic results average seeds 42, 43, and 44 within each question–condition pair; questions remain the statistical units.
<table><tr><td>Decoding</td><td> $\operatorname { E C S } \rho$ </td><td>ECS mean δ</td><td> $\mathrm { P K S } \rho$ </td><td>PKS mean δ</td></tr><tr><td>Greedy</td><td>0.7995</td><td>0.5464</td><td>-0.2487</td><td>-0.1402</td></tr><tr><td> $T = 0 . 7 ,$  three seeds</td><td>0.8027</td><td>0.5537</td><td>-0.2638</td><td>-0.1457</td></tr><tr><td> $T = 1 . 0 ,$  three seeds</td><td>0.8603</td><td>0.6380</td><td>-0.3162</td><td>-0.1675</td></tr></table>

In Table 7, seeds 42/43/44 are averaged within each question–condition pair before testing; questions remain the statistical units $( p \leq 2 \times 1 0 ^ { - 4 } )$ . Greedy and $T = 0 . 7$ show a local oracle/answer ECS reversal, so only the overall association direction is robust. Six ECS subsets give $\rho \in [ 0 . 8 3 6 9 , 0 . 8 5 7 7 ]$ PKS over the last 4/7/13 FFN layers gives $\rho = - 0 . 1 6 1 4 , - 0 . 2 9 5 5 , - 0 . 3 8 5 2$ , respectively (all $p < 1 0 ^ { - 4 } )$ . Absolute thresholds require recalibration.

## H.2 Paired Accuracy–Communication Evaluation

Using the existing trained Qwen graphs without retraining or retuning, all methods are evaluated on the same 500 MMLU test questions. Probe-Joint obtains 80.2% (401/500).

Table 8: Paired accuracy and communication comparison on 500 MMLU questions using the existing trained Qwen graphs. Probe-Joint achieves 80.2% accuracy (401/500).
<table><tr><td>Baseline (accuracy)</td><td>Accuracy difference (pp)</td><td>95% CI (pp)</td><td>McNemar p</td><td>Token reduction (%)</td><td>95% CI (%)</td></tr><tr><td>GPTSwarm 81.2%</td><td>-1.0</td><td>[-3.2, 1.0]</td><td>0.458</td><td>29.7</td><td>[27.7, 31.6]</td></tr><tr><td>Agent Net 81.2%</td><td>-1.0</td><td>[-3.2, 1.0]</td><td>0.458</td><td>41.4</td><td>[39.9, 43.0]</td></tr><tr><td>Agent Prune 79.4%</td><td>+0.8</td><td>[−1.4, 3.2]</td><td>0.608</td><td>12.9</td><td>[9.6, 16.0]</td></tr></table>

Accuracy differences are Probe-Joint minus baseline; pp denotes percentage points. Confidence intervals are 95% intervals for paired differences and token reductions.

Table 8 reports the paired estimates. Intervals including zero do not prove equivalence; tokenreduction intervals are positive.

## H.3 Calibrated Workflow Evaluations

For sparse expert selection, three questions per domain from the original MMLU question pool calibrate a fixed seg-ECS ranking before testing on the remaining questions in a seven-expert workflow.

Table 9: Sparse expert selection using a fixed, few-shot-calibrated seg-ECS ranking. Accuracy, expert calls, and tokens are reported per test question.
<table><tr><td>Policy</td><td>Acc. (%)</td><td>Expert calls/question</td><td>Tokens/question</td></tr><tr><td>All-7</td><td>77.46</td><td>7</td><td>5601.7</td></tr><tr><td>ECS Top-5</td><td>76.94</td><td>5</td><td>4119.9</td></tr><tr><td>Random-5</td><td>67.43</td><td>5</td><td>3929.0</td></tr></table>

Table 9 reports sparse selection. For conditional escalation, an independent 115-question pool supplies 100 calibration resamples, each with 10 base-workflow errors and 10 correct cases. A fitted PKS threshold gates an external reasoning Adapter on the same 300 held-out questions; Random Gate matches each threshold’s call count.

Table 10: Conditional escalation to an external reasoning Adapter on 300 held-out questions. Incremental tokens are measured relative to the No-Adapter workflow.
<table><tr><td>Policy</td><td>Acc. (%)</td><td>Mean Adapter calls/300</td><td>Incremental tokens/question</td></tr><tr><td>Random Gate</td><td>79.95</td><td>102.09</td><td>613.74</td></tr><tr><td>PKS Gate</td><td>82.08</td><td>102.09</td><td>729.97</td></tr><tr><td>Always Adapter</td><td>85.67</td><td>300</td><td>1803.46</td></tr></table>

In Table 10, incremental tokens include additional Adapter and FinalDecision input/output relative to No-Adapter. PKS gains 2.13 pp over Random Gate (question-level paired bootstrap 95% CI [0.60,3.78], p = 0.0026), at matched calls, not matched tokens. These workflows provide preliminary transfer evidence, not general optimization guarantees.

## H.4 Probe Cost

For generation length T, context length C, hidden size $d ,$ vocabulary size $V ,$ selected ECS pairs $S _ { E }$ selected PKS layers $S _ { P }$ , and k retained context tokens/head, incremental costs are approximately $O ( T S _ { E } ( C + k \bar { d } ) )$ for ECS and $O ( T S _ { P } d V )$ plus $O ( T S _ { P } V )$ for PKS.

Table 11: Incremental probe-collection cost on six RTX 4090 GPUs. Overhead is relative to generation without probes; timing is in milliseconds per generated token.
<table><tr><td>Mode</td><td>ms/token</td><td>Overhead</td></tr><tr><td>Base</td><td>84.546</td><td></td></tr><tr><td>ECS</td><td>97.108</td><td>14.9%</td></tr><tr><td>PKS</td><td>110.247</td><td>30.4%</td></tr><tr><td>ECS+PKS</td><td>112.312</td><td>32.8%</td></tr></table>

Table 11 uses Qwen2.5-32B, FP16, eager attention, six RTX 4090 GPUs, 16 ECS layer–head pairs in the final four layers and seven final PKS layers; medians follow two warm-ups and five measured runs. Joint collection shares one generation. Online PKS gating retains this overhead; comparison with fused-attention production inference may differ.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction frame the paper as an empirical study of information flow in LLM swarms using process-level behavioral probes. The claims are limited to ECS/PKS as one concrete probe instantiation, their observed patterns in controlled swarm topologies and a realistic tool-augmented GAIA pipeline, and two controlled applications where these probes complement utility-driven structure optimization. We explicitly present the probe interpretation as heuristic, not as a formal decomposition of generation or a standalone optimization guarantee.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper includes a separate Limitations section. It states that the study is empirical and that ECS/PKS are used as heuristic behavioral probes rather than formal mechanisms. It also discusses the limited coverage of datasets and model families, the controlled nature of the optimization studies, and the additional computational cost of probe computation.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not present theoretical results such as theorems, lemmas, or formal proofs. The mathematical expressions in the paper define metrics, probe computations, and reward functions used in the empirical study.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper provides the information needed to reproduce the main empirical results at the level relevant to its claims. It describes the controlled topology experiments, the tool-augmented GAIA pipeline, the chain/path-selection task, model choices, probe computation procedures, ablation settings, reward definitions, and evaluation metrics. The appendix further specifies dataset construction, implementation details, and additional experimental definitions. Since the contribution is an empirical analysis and probe-guided application study rather than a new model or architecture, reproducibility is supported through detailed experimental and methodological descriptions.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code and processed artifacts will be released at https://github.com/ HaRry-qaq/Swarmprobe, together with instructions for reproducing the main experimental results. The paper and appendix describe the experimental settings, data construction, probe computations, rewards, and evaluation protocols. The release will include environment specifications, dependency versions, and commands for data preparation and experiment execution. It will also provide instructions for computing ECS and PKS and reproducing the main tables and figures.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

## Answer: [Yes]

Justification: The paper specifies the experimental details necessary to understand the results. Since the work does not train a new model, the relevant details are the task construction, dataset filtering, model choices, topology and pipeline designs, probe computation procedures, ablation settings, reward definitions, and evaluation metrics. Additional implementation details and extended experimental definitions are provided in the appendix.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper provides statistical evidence for the quantitative results that support the main empirical claims, including confidence intervals and effect-size measures where applicable. Some analyses are qualitative; for these, the paper reports them as descriptive or exploratory rather than as standalone significance claims. The statistical procedures and additional quantitative summaries are described in the appendix.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper reports the compute resources used for the experiments in the appendix. The original experiments were run on a server with 8 NVIDIA A800 GPUs, and reproducing the full set of experiments is estimated to require approximately four days on comparable hardware. The appendix additionally reports probe-latency measurements on six RTX 4090 GPUs.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

## Answer: [Yes]

Justification: The research conforms to the NeurIPS Code of Ethics. The experiments use public or constructed datasets and evaluate LLM swarm behavior through model outputs, probe computations, and controlled analyses. The work does not involve human-subject experiments, private personal data, or deployment in high-risk decision-making settings. The camera-ready version uses the final conference format.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses positive impacts through improved diagnosis, interpretability, and efficiency of LLM swarm and tool-augmented reasoning systems, and also notes potential negative impacts from misuse of more capable automated agent systems and over-interpretation of heuristic probes as definitive explanations or safety guarantees.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release a new pretrained model, image generator, or highrisk scraped dataset. The planned release is limited to code, configurations, controlled task data, and processed artifacts for reproducing the empirical analyses. These materials do not provide new generative capabilities or unsafe scraped content, so safeguards for high-risk model or dataset release are not applicable.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper cites the original sources of the datasets, models, and methods used in the experiments, including GAIA, MMLU, NCBI Taxonomy, the LLM backbones, and the ReDeEP probe definitions. Asset-specific licenses and terms of use will be documented in the appendix and release package. Any released materials are limited to code, configurations, controlled task data, and processed artifacts needed for reproducibility, with attribution and license information included in the release package.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The planned release contains code, configurations, and processed artifacts. The experiments use existing benchmarks, public model backbones, and constructed or processed experimental data described in the paper and appendix. These materials will be released at https://github.com/HaRry-qaq/Swarmprobe with accompanying documentation and license information.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments or research with human subjects. The experiments are conducted using existing benchmarks, constructed or processed task data, and LLM-generated outputs, without collecting data from human participants or crowd workers.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [N/A]

Justification: The paper does not involve crowdsourcing or research with human subjects. No data are collected from study participants, and the experiments are conducted using existing benchmarks, constructed or processed task data, and LLM-generated outputs. Therefore, IRB approval or equivalent human-subject review is not applicable.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: The paper describes the use of LLMs as core components of the studied swarm systems. LLMs are used as reasoning nodes in controlled topologies, tool-augmented GAIA pipelines, and probe-guided optimization studies. The paper specifies the model backbones used in the experiments and describes how LLM outputs, internal computations, and probe scores are analyzed. LLMs used only for writing or editing assistance, if any, are not part of the scientific methodology.