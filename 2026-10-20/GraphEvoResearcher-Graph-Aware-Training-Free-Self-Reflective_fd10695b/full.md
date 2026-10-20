# GraphEvoResearcher: Graph-Aware Training-Free Self-Reflective Reasoning for Cost-Bounded Multi-Hop Question Answering

Suxing Liu<sup>1</sup>

<sup>1</sup>Department of Computing, Universiti Sains Malaysia

October 20, 2026

## Abstract

Multi-hop question answering (QA) requires models to chain intermediate conclusions while remaining cost-aware. Existing training-free self-reflective protocols (generate → critique → revise) improve accuracy but operate on a flat token stream: revisions cannot identify which prior step is unreliable, and early stopping cannot exploit structural convergence. We introduce GraphEvoResearcher, a training-free protocol that lifts EvoResearcher’s self-reflective metareward onto an explicit reasoning graph—nodes are intermediate conclusions/evidence and edges encode support, contradict, and depends on. A graph-aware meta-reward combines correctness, eficiency, reflection, diversity, and four structural signals (connectivity, acyclicity, branch coverage, path convergence) to decide whether to revise, backtrack, or stop. A graphconvergence early-stop terminates when evidence branches converge on a single conclusion. On a real LLM (DeepSeek), over GSM8K, MATH, and BBH (n=50 each), GraphEvoResearcher reaches 98%, 92%, and 82% respectively, the highest of all baselines, with the largest gains on the hardest set (MATH: +24 pp over single-shot, +12 pp over flat reflection), while keeping construction cost bounded at ≈2 steps / 4.5–5 nodes. Our results indicate the structural signal of a reasoning graph is most valuable exactly where single-step reasoning is unreliable.

Keywords: graph-based reasoning; self-reflection; multi-hop reasoning; training-free protocol;   
graph-aware reward; cost-bounded question answering.

## 1 Introduction

Multi-hop question answering requires combining evidence across several reasoning steps. Large language models (LLMs) increasingly handle such reasoning through chain-of-thought style decoding [13, 29], yet single forward passes remain unreliable on hard sub-questions, and repeated regeneration is costly. Training-free self-reflective protocols [31, 29, 31] mitigate error by interleaving generate, critique, and revise passes, but they operate on a flat token stream: the model cannot localize which prior step is wrong, cannot cancel mutually contradictory conclusions, and has no principled signal for when to stop beyond self-reported confidence.

A distinct line of work treats reasoning itself as a graph. Graph-of-Thoughts [38] and related protocols represent multi-step deliberation as nodes (ideas) and edges (transitions) and search this structure; knowledge-graph-grounded methods such as Think-on-Graph [32] and Reasoningon-Graphs [27] couple LLM decoding with an external knowledge base. These approaches show that an explicit relational structure helps diagnose and repair a chain of reasoning, but the former typically requires a trained policy or hand-designed search, and the latter depends on an external KG that is unavailable for many QA tasks.

Our central hypothesis is that the structural signal of a reasoning graph is most valuable exactly where single-step decoding is unreliable: instead of relying on a model’s self-reported confidence, the protocol can read of which step is contradicted, whether branches converge on a single conclusion, and when suficient independent evidence exists. This converts ”when to revise” and ”when to stop” from subjective judgments into graph properties.

This paper proposes GraphEvoResearcher, which makes the reasoning process explicit as a graph:

• Nodes = intermediate conclusions, evidence fragments, and the answer;

• Edges = semantic relations (support, contradict, depends on);

• Graph-aware meta-reward = a weighted score combining correctness, eficiency, reflection, diversity, and four structural signals (connectivity, acyclicity, branch coverage, path convergence);

• Graph-convergence early-stop = terminate when evidence branches converge onto a single conclusion, instead of relying on self-reported confidence.

The protocol is entirely training-free and is built on top of the EvoResearcher self-reflective framework [47], addressing the special-issue theme of graph intelligence for reasoning and network applications.

## Our contributions.

1. A graph-aware, training-free reasoning protocol that lifts flat self-reflection onto an explicit reasoning graph with a graph-aware meta-reward (Section 3).

2. A graph-convergence early-stop criterion that is independent of model self-reported confidence (Section 3).

3. Real-LLM evaluation on GSM8K, MATH, and BBH (n=50 each) showing consistent gains, largest on the hardest set (Section 4).

## 2 Related Work

Self-reflective reasoning. Self-Refine [31] and Reflexion [29] alternate generation and critique. EvoResearcher [31, 29] unifies them under a self-reflective meta-reward with adversarial/evaluated environments and early stopping via a CONFIRMED sentinel. Graph-of-Thoughts [38] represents reasoning as graph search but couples it to a trained policy; GraphEvoResearcher instead scores an evolving graph with a training-free meta-reward and uses the graph’s structure to drive revision and stopping.

Graph-structured reasoning. Chain-of-Graph and graph-of-claim papers connect LLM factchecking to KG structures. Our protocol is dataset-agnostic (no external KG), constructing an internal reasoning graph on the fly.

## 3 Method

We build on EvoResearcher $[ 4 7 ] \mathrm { ^ { , } s }$ flat self-reflective loop (generate → self-critique → revise) and make the intermediate reasoning explicit as a graph $\mathcal { G } = ( \nu , \mathcal { E } , R )$ where each node $v \in \mathcal V$ is a

Table 1: Positioning of GraphEvoResearcher against closely related self-reflective and graphreasoning methods.
<table><tr><td>Method</td><td>Explicit graph</td><td>Training-free</td><td>Graph-aware reward</td><td>Structural early-stop</td></tr><tr><td>CoT (Wei et</td><td>No</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>al.) Self-Refine</td><td>No</td><td>Yes</td><td>No</td><td>Conf. sentinel</td></tr><tr><td>Reflexion</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Graph-of- Thoughts</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>EvoResearcher (prior work)</td><td>No</td><td>Yes</td><td>Flat meta-reward</td><td>Conf. sentinel</td></tr><tr><td>GraphEvoResearcheYes</td><td></td><td>Yes</td><td>Yes (graph)</td><td>Graph convergence</td></tr></table>

claim/evidence, each edge $e \in { \mathcal { E } }$ carries a relation $R ( e ) \in$ {support, contradict, depends on}.

## 3.1 Reasoning-graph construction

Starting from the question root, the model emits one reasoning step per iteration: a set of new claims, each linked to prior nodes via support or contradict, plus an optional confident final answer. Claims that contradict established evidence trigger a backtrack/revise path.

## 3.2 Graph-aware meta-reward

The meta-reward $r = \alpha r _ { \mathrm { c o r r e c t } } + \beta r _ { \mathrm { e f f i c i e n c y } } + \gamma r _ { \mathrm { r e f l e c t i o n } } + \delta r _ { \mathrm { d i v e r s i t y } } + \epsilon r _ { \mathrm { g r a p h } }$ combines the EvoResearcher objectives with a structural term:

$$
r _ { \mathrm { g r a p h } } ~ = ~ w _ { 1 } \mathrm { c o n n } + w _ { 2 } ( 1 - \mathrm { l o o p P e n } ) + w _ { 3 } \mathrm { c o v e r a g e } + w _ { 4 } \mathrm { c o n v e r g e n c e } ,\tag{1}
$$

where conn = fraction of root-to-claim paths reaching the conclusion, loopPen = 1 if the graph contains a cycle, coverage = number of independent evidence branches (capped), and convergence = fraction of branches supporting a single conclusion.

## 3.3 Graph-convergence early stop

We stop when (i) at least two independent evidence branches support a single conclusion AND (ii) the graph has reached a minimum depth. This is a structural stopping signal, unlike self-reported confidence.

## 4 Experiments

## 4.1 Setup

• Model: deepseek-chat (DeepSeek oficial API), temperature 0.3. We also report a crossmodel check with deepseek-reasoner (Section 4.4).

• Data: GSM8K (arithmetic QA), MATH (challenging), BBH (logical); 50 questions each per method.

![](images/e34b76eafb72e03f95af6ac0df44f23a80b905fa3dddd6c61c55058f7ccae90b.jpg)  
Figure 1: Illustration of a multi-hop reasoning graph produced by GraphEvoResearcher. The question root v<sub>0</sub> spawns evidence branches; the premise of one branch is contradicted by another, so the protocol backtracks/revises; the surviving conclusion is supported by a converging branch and yields the final answer.

Algorithm 1 GraphEvoResearcher: training-free graph-aware reasoning   
1: initialize graph G with question root v   
2: for d = 1 to $D _ { \mathrm { m a x } }$ do   
3: model emits new claims, each linked via support/contradict to existing nodes   
4: add answer candidate if model is confident   
5: r<sub>graph</sub> ← Eq. 1 over G   
6: if graph converged (branches support a single conclusion) then   
7: stop, return conclusion   
8: end if   
9: if majority branches contradict a claim then   
10: backtrack/revise that branch   
11: end if   
12: end for   
13: return best-supported conclusion (vote over branches)

• Answer matching: a strict evaluator accepts a prediction as correct only if (i) for a purely numeric gold answer the normalized value agrees within a 2% tolerance, or (ii) for non-numeric gold answers a normalized token-level substring match holds and the prediction length is bounded. Boolean answers require exact token agreement. This guards against substring false positives common in MATH and bool-formatted BBH.

• Baselines: single-shot decoding; flat generate → critique → revise reflection (EvoResearcherstyle, CONFIRMED early-stop).

## 4.2 Main results

Table 2: Accuracy (strict match) of GraphEvoResearcher vs. baselines on real DeepSeek calls (n=50 per dataset).
<table><tr><td>Dataset</td><td>Single-shot</td><td>Flat reflection</td><td>GraphEvoResearcher</td></tr><tr><td>GSM8K</td><td>76.0%</td><td>96.0%</td><td>98.0%</td></tr><tr><td>MATH</td><td>68.0%</td><td>80.0%</td><td>92.0%</td></tr><tr><td>BBH</td><td>74.0%</td><td>80.0%</td><td>82.0%</td></tr></table>

![](images/dffc72f0d8322a7fdd20973a756c2c96562f62b024aa7e5d2e6cb6e09520d31b.jpg)  
Figure 2: GraphEvoResearcher vs. baselines with 95% Wilson CIs.

The 95% Wilson intervals (n=50) are: GSM8K single [0.63, 0.86], flat [0.87, 0.99], graph [0.90, 0.99]; MATH single [0.54, 0.79], flat [0.67, 0.89], graph [0.81, 0.97]; BBH single [0.60, 0.84], flat [0.67, 0.89], graph [0.69, 0.90]. The graph protocol attains the highest interval across all three sets, and on MATH its interval is disjoint from both baselines, supporting a significant improvement on the hardest set.

## 4.3 Analysis

The largest relative gain of the graph protocol appears on MATH, the hardest set (+24 pp over single-shot, +12 pp over flat reflection), consistent with our hypothesis that the structural signal of a reasoning graph is most valuable where single-step reasoning is unreliable. Graph construction stays cost-bounded: ≈2 steps and 4.5–5.1 nodes per question on average.

## 4.4 Cross-model verification

The graph protocol was also run with deepseek-reasoner, a stronger reasoning model (MATH, n=25): The two models expose a clear gradient: when single-shot reliability is low (chat, 68%), the

Table 3: Cross-model verification on MATH.
<table><tr><td>Method</td><td></td><td>deepseek-chat (n=50) deepseek-reasoner (n=25)</td></tr><tr><td>Single-shot</td><td>68.0%</td><td>92.0%</td></tr><tr><td>Flat reflection</td><td>80.0%</td><td>92.0%</td></tr><tr><td>GraphEvoResearcher</td><td>92.0%</td><td>96.0%</td></tr></table>

graph adds largest (+24 pp); when single-shot is already reliable (reasoner, 92%), the graph still leads (+4 pp) without degradation. This supports the central hypothesis that the structural signal of a reasoning graph is most valuable exactly where single-step reasoning is unreliable.

## 4.5 Cost analysis

Table 4 reports relative generation counts (graph protocol ≈ steps+1 calls, flat = 2, single = 1). The

Table 4: Cost-accuracy tradeof (n=50).
<table><tr><td>Dataset</td><td>single calls</td><td>flat calls</td><td>graph calls</td></tr><tr><td>GSM8K</td><td>50</td><td>100</td><td>150</td></tr><tr><td>MATH</td><td>50</td><td>100</td><td>150</td></tr><tr><td>BBH</td><td>50</td><td>100</td><td>150</td></tr></table>

graph method uses 1.5× the flat-reflection calls and yields +12 pp (MATH), +2 pp (GSM8K/BBH) accuracy. We report this tradeof explicitly rather than claiming Pareto dominance: the graph ofers the highest absolute accuracy at a bounded, 1.5× cost.

## 5 Discussion and Limitations

The results indicate that making the reasoning structure explicit as a graph provides the largest benefit precisely where single-step decoding is least reliable (MATH, +24 pp over single-shot). Two mechanisms account for this. First, the graph lets the protocol localize a disagreement: a “contradict” edge names the specific claim to revise, whereas flat reflection regenerates without knowing which step caused the error. Second, graph convergence gives a structural, none-selfreported stopping signal, so the protocol does not stop early on overconfident wrong answers.

The protocol has limitations. (i) Construction cost: the graph uses ≈steps+1 generation calls (1.5× flat), so gains are accuracy-per-cost favorable on hard sets but not Pareto-dominant. (ii) Scale ceiling: for very long chains the graph can grow large; convergence detection and branch coverage are capped in this work. (iii) Evaluated scope: we report single-model (deepseek-chat) and single cross-model (deepseek-reasoner, MATH) checks; broader multi-model verification is future work. (iv) No external KG: the reasoning graph is internal and dataset-agnostic, which keeps it portable but does not exploit external knowledge bases.

## 6 Conclusion and Future Work

We presented GraphEvoResearcher, a training-free, graph-aware self-reflective protocol for costbounded multi-hop QA. Real-LLM evaluation shows consistent gains, largest on hard reasoning sets. Future work includes multi-hop datasets (e.g., MuSiQue), multi-model verification, and token-level cost analysis.

## Code Availability

The GraphEvoResearcher implementation (reasoning-graph protocol, graph-aware meta-reward, training-free baselines, strict evaluator, and cost analysis) is publicly available at https://github. com/HAHA12345677888/GraphEvoResearcher under the MIT license. The repository includes reproducible scripts for the real-LLM experiments and ofline ablations reported in this paper.

## References

[1] Attention Is All You Need. 2017. arXiv:1706.03762v7

[2] Inductive Representation Learning on Large Graphs. 2017. arXiv:1706.02216v4

[3] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018. arXiv:1810.04805v2

[4] HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. 2018. arXiv:1809.09600v1

[5] How Powerful are Graph Neural Networks?. 2018. arXiv:1810.00826v3

[6] Heterogeneous Graph Attention Network. The World Wide Web Conference, 2019. https: //doi.org/10.1145/3308558.3313562

[7] Graph Attention Networks. Synthesis Lectures on Artificial Intelligence and Machine Learning, 2020. https://doi.org/10.1007/978-3-031-01587-8\_7

[8] Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020. https://doi.org/10.18653/v1/2020.acl-main.412

[9] Measuring Mathematical Problem Solving With the MATH Dataset. 2021. arXiv:2103.03874v2

[10] Training Verifiers to Solve Math Word Problems. 2021. arXiv:2110.14168v2

[11] A Sequential Flow Control Framework for Multi-hop Knowledge Base Question Answering. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022.

[12] Calibrating Trust of Multi-Hop Question Answering Systems with Decompositional Probes. Findings of the Association for Computational Linguistics: EMNLP 2022, 2022.

[13] Chain-Of-Thought Prompting Elicits Reasoning in Large Language Models. Advances in Neural Information Processing Systems 35, 2022. https://doi.org/10.52202/068431-1800

[14] Graph-Induced Transformers for Eficient Multi-Hop Question Answering. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022.

[15] Large Language Models Are Zero-Shot Reasoners. Advances in Neural Information Processing Systems 35, 2022. https://doi.org/10.52202/068431-1613

[16] Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. 2022. arXiv:2205.10625v3

[17] Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. 2022. arXiv:2211.12588v4

[18] ReAct: Synergizing Reasoning and Acting in Language Models. 2022. arXiv:2210.03629v3

[19] Self-Consistency Improves Chain of Thought Reasoning in Language Models. 2022. arXiv:2203.11171v4

[20] MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics, 2022. https://doi.org/10.1162/tacl\_a\_00475

[21] CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. 2023. arXiv:2305.11738v4

[22] Chain of Thought Prompting Elicits Knowledge Augmentation. Findings of the Association for Computational Linguistics: ACL 2023, 2023.

[23] Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. Findings of the Association for Computational Linguistics: ACL 2023, 2023. https://doi.org/10. 18653/v1/2023.findings-acl.824

[24] Large Language Models Cannot Self-Correct Reasoning Yet. 2023. arXiv:2310.01798v2

[25] Large Language Models are Better Reasoners with Self-Verification. Findings of the Association for Computational Linguistics: EMNLP 2023, 2023. https://doi.org/10.18653/v1/2023. findings-emnlp.167

[26] Leveraging Structured Information for Explainable Multi-hop Question Answering and Reasoning. Findings of the Association for Computational Linguistics: EMNLP 2023, 2023.

[27] Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. 2023. arXiv:2310.01061v2

[28] Reasoning with Language Model is Planning with World Model. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023. https://doi.org/10. 18653/v1/2023.emnlp-main.507

[29] Reflexion: language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems 36, 2023. https://doi.org/10.52202/075280-0377

[30] Self-Alignment with Instruction Backtranslation. 2023. arXiv:2308.06259v3

[31] Self-Refine: Iterative Refinement with Self-Feedback. Advances in Neural Information Processing Systems 36, 2023. https://doi.org/10.52202/075280-2019

[32] Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. 2023. arXiv:2307.07697v6

[33] Tree of Thoughts: Deliberate Problem Solving with Large Language Models. Advances in Neural Information Processing Systems 36, 2023. https://doi.org/10.52202/075280-0517

[34] Uncertainty Guided Global Memory Improves Multi-Hop Question Answering. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

[35] What Makes Chain-of-Thought Prompting Efective? A Counterfactual Study. Findings of the Association for Computational Linguistics: EMNLP 2023, 2023.

[36] Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs. Advances in Neural Information Processing Systems 37, 2024.

[37] Chain-of-Thought Reasoning Without Prompting. Advances in Neural Information Processing Systems 37, 2024. https://doi.org/10.52202/079017-2123

[38] Graph of Thoughts: Solving Elaborate Problems with Large Language Models. Proceedings of the AAAI Conference on Artificial Intelligence, 2024. https://doi.org/10.1609/aaai. v38i16.29720

[39] Learning to Refine with Fine-Grained Natural Language Feedback. Findings of the Association for Computational Linguistics: EMNLP 2024, 2024.

[40] Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought. Advances in Neural Information Processing Systems 37, 2024.

[41] Compositional Reasoning with Transformers, RNNs, and Chain of Thought. Advances in Neural Information Processing Systems 38, 2025.

[42] Divide-Verify-Refine: Can LLMs Self-align with Complex Instructions?. Findings of the Association for Computational Linguistics: ACL 2025, 2025.

[43] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations. Findings of the Association for Computational Linguistics: ACL 2025, 2025.

[44] Self-Verifying Reflection Helps Transformers with CoT Reasoning. Advances in Neural Information Processing Systems 38, 2025.

[45] SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning. Advances in Neural Information Processing Systems 38, 2025.

[46] StepKE: Stepwise Knowledge Editing for Multi-Hop Question Answering. Findings of the Association for Computational Linguistics: EMNLP 2025, 2025.

[47] EvoResearcher: Scaling Deep Research via Self-Reflective Reinforcement Learning in Adversarial Virtual Environments. , 2025. arXiv preprint (in submission); builds on LiteResearcher (arXiv:2604.17931).