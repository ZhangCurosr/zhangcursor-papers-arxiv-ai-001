# Adaptive Semantic Branching: Disagreement-Localized Cross-Trajectory Regeneration for Reasoning Candidate Expansion

Jeong-Yoon Kim

Department of Electronic Engineering

Hanbat National University

Tae-Yoon Lee

Department of Electronic Engineering

Hanbat National University

## Abstract

Reasoning trajectories generated for the same problem often arrive at conflicting answers. These disagreements reveal alternative intermediate decisions that can be reused to explore new solutions. We introduce Adaptive Semantic Branching (ASB), a training-free inference procedure that turns cross-trajectory disagreement into targeted candidate generation. ASB first groups independently sampled trajectories by extracted final answer, selects a representative from each group, and aligns their reasoning steps with a monotonic dynamic program. A persistent-divergence objective identifies the point at which initially similar reasoning paths develop diferent continuations. For each selected directed pair, ASB retains the source trajectory’s textual prefix, extracts a short reasoning window from a competing donor trajectory, and generates a fresh solution conditioned on both. A diversity-aware policy allocates up to three additional generations according to the answer-cluster structure. In a 500-problem MATH evaluation with Qwen2.5-7B-Instruct, ASB increases oracle candidate coverage from 445/500 (89.0%) with seven initial samples to 455/500 (91.0%), recovering correct candidates for 10 of the 55 initially uncovered problems. The method uses 576 additional generation calls across the evaluation set, averaging 8.152 calls per problem. We provide a precise specification of the branching operator, alignment objective, generation policy, and experimental configuration.

## 1 Introduction

Sampling multiple chains of thought is a standard approach to improve the breadth of language-model reasoning Wang et al. (2023). Diferent samples can follow distinct solution strategies, encounter diferent intermediate obstacles, and occasionally reach diferent final answers. Tree- and graph-structured reasoning methods make these alternatives explicit by expanding and recombining intermediate states Yao et al. (2023); Besta et al. (2024). An additional opportunity arises when a model has already produced several complete solutions: the relationships among these trajectories provide a structured signal for allocating subsequent generation.

Consider two completed solutions to the same mathematics problem. Their opening steps may encode a shared interpretation or calculation, while later steps pursue diferent substitutions, assumptions, or subgoals. A new candidate can draw on the stable initial reasoning of one solution while exploring an alternative local direction extracted from the other. This requires three concrete decisions: locating the transition between common and competing reasoning, selecting the information passed between trajectories, and choosing which directed exchanges receive a limited generation budget.

We propose Adaptive Semantic Branching (ASB), an inference-time framework that makes these decisions from completed reasoning traces. The method begins with seven independently sampled trajectories. Their extracted answers determine clusters, and representatives of diferent clusters are compared through step-level monotonic alignment. A change-point objective rewards an earlier region of higher cross-trajectory similarity followed by a sustained decrease in similarity. The resulting boundary defines a textual prefix from a source trajectory. A donor trajectory provides a short, locally aligned reasoning window. Conditioning on the original problem, source prefix, and donor insight yields a new candidate trajectory (Figure 1).

The direction of exchange matters. A ← B retains the reasoning prefix from A and uses a local alternative from B, whereas B ← A constructs a diferent prompt from a diferent prefix and insight window. ASB evaluates these directed choices, uses a divergence-based score to rank candidate exchanges, and allocates zero, two, or three branch generations according to the number of answer clusters. The resulting procedure is lightweight: step features are computed locally, while the initial and branch generations use the same underlying language model.

Our study develops a concrete cross-trajectory regeneration operator and reports its first candidate-coverage evaluation. On the fixed 500-item MATH manifest, the seven initial samples contain a correct answer for 445 problems. Directed regeneration expands that coverage to 455 problems, recovering 18.18% of the cases missed by the initial pool. In addition to the empirical result, this paper details the complete sequence from raw reasoning segmentation to branch lineage, deterministic seed derivation, and generation prompts.

The contributions are threefold:

• Divergence-localized branching. A monotonic step alignment and persistent similarity-change objective produce a source-specific regeneration boundary from conflicting completed trajectories.

• Directed cross-trajectory regeneration. A source prefix and donor-local insight form a compact conditioning context for a fresh reasoning trajectory, with explicit directionality and traceable parent lineage.

• Adaptive candidate expansion. Answer-cluster disagreement determines a small branch budget, yielding a reproducible generation policy and a measured improvement in oracle candidate coverage on MATH-500.

## 2 Related Work

Test-time sampling and structured exploration. Self-consistency Wang et al. (2023) uses multiple independently sampled chains to aggregate predictions. Tree of Thoughts Yao et al. (2023) and Graph of Thoughts Besta et al. (2024) develop structured reasoning spaces with intermediate exploration and composition. ASB uses the structure already present across completed samples: the final-answer clusters determine which trajectories to compare, and aligned intermediate steps determine how new candidate generations are conditioned.

Information exchange across reasoning paths. Multi-Chain Reasoning Yoran et al. (2023) integrates information from multiple sampled chains through metareasoning. Contract-checked graph editing Li and Cao (2026) constructs combinations of partial solutions represented as structured reasoning graphs. ASB introduces a complementary natural-language operation: a directionally chosen textual prefix and a short, boundarylocalized donor window are assembled into a new generation prompt. The operation can be applied directly to free-form chain-of-thought outputs.

Divergence and selective intervention. Reasoning Path Divergence Ju et al. (2025) develops step-level trajectory comparison for measuring diversity. AgentAuditor Yang et al. (2026) analyzes agreements and disagreements within multi-agent reasoning trees for localized adjudication. Entropy-Gated Branching Li et al. (2026) uses uncertainty to allocate branch computation during generation, while Trap-Aware Adaptive Restart Chenkang et al. (2026) uses a learned diagnostic policy to restart reasoning from identified prefixes. ASB connects completed-path step alignment to a persistent-divergence boundary and a directed candidate-generation policy. Its branch location emerges from the relationship between two solutions, rather than from a single trajectory’s uncertainty signal.

## 3 Problem Formulation

For problem x, sample N independent model trajectories $T _ { 1 } , \dots , T _ { N }$ from $p _ { \theta }$ . Each trajectory contains the original reasoning text $r _ { i }$ and a parsed final answer $a _ { i }$ In the reference implementation $N = 7$ . Let $\kappa ( a _ { i } )$ be the extracted-answer clustering key, producing ordered answer clusters $\mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { K }$

The method constructs a collection of new branch trajectories $B ( x )$ and returns

$$
{ \mathcal { C } } ( x ) = { \mathcal { C } } _ { 0 } ( x ) \cup { \mathcal { B } } ( x ) , \qquad { \mathcal { C } } _ { 0 } ( x ) = \{ T _ { 1 } , \ldots , T _ { 7 } \} ,\tag{1}
$$

with up to three additional trajectories per problem. The primary evaluation quantity is oracle candidate coverage: the fraction of problems for which the returned collection contains a gold-equivalent answer. Let $y _ { q }$ be the reference answer to problem $x _ { q }$ and let $\operatorname { E q }$ be the MATH equivalence checker. Then

$$
\operatorname { C o v } ( { \mathcal { C } } ) = { \frac { 1 } { M } } \sum _ { q = 1 } ^ { M } \mathbf { 1 } [ \exists T \in { \mathcal { C } } ( x _ { q } ) : \operatorname { E q } ( a _ { T } , y _ { q } ) = 1 ] .\tag{2}
$$

This metric quantifies how many problems have at least one correct candidate available for a subsequent selection stage. The reference implementation evaluates Eq after generation and uses the parsed answer string as the inference-time cluster key.

## 4 Adaptive Semantic Branching

## 4.1 Initial generation and answer clusters

For every problem, ASB generates seven complete trajectories with temperature 0.7, top-p = 1.0, and maximum completion length 2,048 tokens. Each call receives an independently derived deterministic seed. The initial prompt requests a complete derivation and a boxed final answer. The seven calls use the same problem statement and are sampled independently.

The extracted answers partition the trajectories into clusters ordered by their first appearance. When $K = 1$ ASB finishes with the initial seven trajectories. When $K \geq 2$ , the method selects one representative trajectory from each cluster. With token log-probabilities available, the representative has the largest length-normalized mean token log-probability; otherwise the first observed trajectory is selected. All ordered cross-cluster representative pairs form candidates for directed regeneration.

![](images/b9e63ed66ebd9f4d5d9424cd0ff70362521d6e20cb2a4c84d7a6366d3bc12586.jpg)  
Figure 1: Adaptive Semantic Branching. (a) The inference pipeline groups seven independent trajectories by answer, aligns cross-cluster representatives, locates the persistent divergence boundary, and selects directed exchanges. (b) In $A  B ,$ source prefix $P _ { A }$ and donor insight $I _ { B }$ condition a newly generated trajectory $T _ { A  B } ^ { \prime }$ . The $k ^ { * } = 0$ case uses an empty source prefix and a fresh start.

## 4.2 Reasoning-step extraction

Let $r _ { A }$ be the raw reasoning text of a representative trajectory. ASB divides it into ordered steps $A \ =$ $( a _ { 1 } , \ldots , a _ { m } )$ , storing the start and end character ofsets of every step. The segmentation procedure gives priority to numbered lists and bullet formatting, using their line boundaries as steps. For other response formats, line breaks and sentence punctuation supply candidate boundaries. A final answer-only fragment, such as a standalone boxed answer, is excluded from the reasoning sequence. Segments shorter than five whitespace-delimited tokens are merged into the preceding step.

The retained ofsets serve a concrete role in regeneration: once a step boundary is selected, the source prefix is recovered directly from the original response text. The implementation also records the step index, text, and token count for inspection and analysis.

## 4.3 Deterministic step representation

The MATH-500 reference run employs a local word- and character-gram hashing encoder of dimension $D = 5 1 2$ For a step a, lowercase its text, split it on whitespace, and rejoin the words with single spaces. The feature multiset $F ( a )$ contains all whitespace-delimited words and all character substrings of length four from the rejoined string.

For each feature f, compute its SHA-256 digest $h ( f )$ The first four digest bytes, interpreted as an unsigned integer, select the coordinate $q ( f ) = \operatorname { i n t } ( h ( f ) _ { 0 : 4 } )$ mod 512. The least significant bit of the fifth byte determines the sign $\sigma ( f ) \in \{ - 1 , + 1 \}$ . The unnormalized vector is

$$
v _ { a } [ d ] = \sum _ { f \in F ( a ) } \sigma ( f ) \mathbf { 1 } [ q ( f ) = d ] , \quad d = 0 , \ldots , 5 1 1 .\tag{3}
$$

The embedding is $e ( a ) = v _ { a } / \lVert v _ { a } \rVert _ { 2 }$ for a nonzero vector and the zero vector otherwise. Cosine similarity involving a zero vector is defined as zero. This fixed-dimensional representation produces deterministic step similarities using local string operations and hashing.

## 4.4 Monotonic cross-trajectory alignment

For a directed representative pair $A  B ,$ , write source steps $A = ( a _ { 1 } , \ldots , a _ { m } ) $ and donor steps $B = ( b _ { 1 } , \ldots , b _ { n } )$ The pairwise cost is

$$
c ( i , j ) = 1 - \cos ( e ( a _ { i } ) , e ( b _ { j } ) ) .\tag{4}
$$

A dynamic time warping (DTW) program Sakoe and Chiba (1978) finds a minimum-cost, order-preserving alignment path π. For source step $a _ { i } .$ , aggregate the similarity of its aligned donor steps by

$$
s _ { i } = \operatorname* { m a x } _ { j : ( i , j ) \in \pi } \cos ( e ( a _ { i } ) , e ( b _ { j } ) ) .\tag{5}
$$

The implementation fills any degenerate unmatched source position with the donor step at the corresponding proportional position. The result $s _ { 1 } , \ldots , s _ { m }$ describes how agreement between the two reasoning sequences changes over the progress of the source path.

## 4.5 Persistent divergence localization

For each interior source boundary $k \in \{ 1 , \ldots , m - 1 \}$ , ASB computes

$$
J ( k ) = \frac { \sqrt { k ( m - k ) } } { m } \left( \frac { 1 } { k } \sum _ { i = 1 } ^ { k } { s _ { i } } - \frac { 1 } { m - k } \sum _ { i = k + 1 } ^ { m } { s _ { i } } \right)\tag{6}
$$

The first term rewards a reasonably balanced split, while the diference in average similarities favors a relatively aligned prefix followed by a persistently divergent suffix. ASB selects the earliest maximizing boundary when its score is positive. For a nonpositive maximum, the selected boundary is $k ^ { * } = 0$ , initiating a full-reset collaborative branch. This allows the donor direction to be explored even for pairs whose similarity profile favors restarting from the original problem.

For $k ^ { * } > 0 .$ , let $d _ { A } ( k ^ { * } )$ denote the original character ofset at the end of source step $k ^ { * }$ . The source prefix is recovered as

$$
P _ { A } = r _ { A } [ 0 : d _ { A } ( k ^ { * } ) ] .\tag{7}
$$

At $k ^ { * } = 0 , P _ { A }$ is the empty string. The metadata records the step-based prefix ratio $k ^ { * } / m$ together with the character ofset, allowing both the chosen reasoning boundary and the corresponding raw text to be reconstructed.

## 4.6 Donor-local insight and directed regeneration

The donor step aligned to the chosen source boundary determines the start of an insight window $I _ { B }$ . ASB extracts that donor step and up to two subsequent donor steps, for a maximum window length of three steps. The donor’s final-answer field is kept in lineage metadata; the branch prompt contains the partial insight text with any incidental final-answer strings masked.

A directed branch conditions the same model on the original problem, source prefix, and donor insight:

$$
T _ { A  B } ^ { \prime } \sim p _ { \theta } ( \cdot \mid { \mathrm { P r o m p t } } ( x , P _ { A } , I _ { B } ) ) .\tag{8}
$$

The generation instruction asks the model to check the competing directions, develop its own continuation, and return a complete derivation with a boxed answer. The prefix is supplied as text inside the user prompt. A branch with $k ^ { * } = 0$ presents the alternative donor insight alongside the original problem and requests a solution from the beginning. Each branch becomes an additional candidate with a full parent–donor lineage record.

The direction $A  B$ determines both the retained prefix and the borrowed insight, so reversing the pair creates a distinct search operation. Figure 1(b) depicts this asymmetric transfer.

## 4.7 Pair ranking and adaptive branch budget

For a directed pair, the divergence strength is $J _ { \mathrm { m a x } } =$ max<sub>k</sub> J(k). When token log-probabilities are stored, the mean log-probabilities for the source-prefix span and the donor-window span are exponentiated to obtain local confidence factors $c _ { A }$ and $c _ { B }$ . The branch score is

$$
q ( A  B ) = J _ { \operatorname* { m a x } } { \sqrt { c _ { A } c _ { B } } } .\tag{9}
$$

In the absence of token log-probabilities, the confidence factors take value one and pair ranking uses $J _ { \mathrm { m a x } }$ directly.

```latex
Algorithm 1 Adaptive Semantic Branching (per prob
lem)
Require: Problem $x ,$ model ${ \mathit { p } } _ { \theta } ,$ budget $B = 3$
1: C ← SampleIndependent $( x , 7 )$
2: $\mathcal { G } $ ClusterByAnswer $( \mathcal { C } _ { 0 } )$
3: if $| \mathcal { G } | = 1$ then
4: return $\mathcal { C } _ { 0 }$
5: end if
6: $R \gets \mathrm { R }$ epresentatives(G)
7: for each ordered cross-cluster pair (A, B) from R do
8: s, π ← AlignSteps(A, B)
9: k<sup>∗</sup> ← PersistentCut(s) using Eq. 6
10: $P _ { A } \gets \mathrm { R a w P r e f i x } ( A , k ^ { * } )$
11: $I _ { B }$ ← DonorWindow $( B , \pi , k ^ { * } )$
12: $q ( A  B ) $ BranchScor $\scriptstyle \left( A , B , k ^ { * } \right)$
13: end for
14: P ← ChoosePairs $( R , q , B )$
15: $B \gets \emptyset$
16: for each $( A , B )$ in $\mathcal { P }$ do
17: $T ^ { \prime } \gets \mathrm { G }$ enerate $\left( x , P _ { A } , I _ { B } \right)$
18: $B  B \cup \{ T ^ { \prime } \}$
19: end for
20: return $\mathcal { C } _ { 0 } \cup \mathcal { B }$
```

The branching policy follows the answer-cluster structure. For $K = 1$ , no additional generations are allocated. For $K = 2$ , both directed pairs are generated, yielding two branches. For $K \geq 3$ , candidates are ordered by branch score and up to three are chosen, prioritizing diferent unordered cluster pairs and diversity among sources and donors. Additional ranked pairs fill any remaining slots. The complete candidate collection therefore contains 7, 9, or up to 10 generated trajectories per problem.

## 5 Experimental Setup

Dataset and model. We evaluate the reference configuration on a fixed, ordered manifest of 500 MATH problems, following the established MATH benchmark Hendrycks et al. (2021); Lightman et al. (2023). The model is Qwen2.5-7B-Instruct Yang et al. (2024). All initial and branch calls use temperature 0.7, top-p = 1.0, and maximum completion length 2,048 tokens. The base seed is 100, with independent identifier-derived seeds for the seven initial trajectories and each new branch.

Evaluation and accounting. Candidate coverage is computed with the MATH answer-equivalence checker (Eq. 2). Inference-time answer clusters are formed from parsed answer strings. We report absolute problem counts, coverage percentages, newly recovered problems, and model generation calls. Each extra branch consumes one model generation call; the local embedding and alignment computations operate on stored trajectory text. The analysis uses the recorded seed-100 run described by the implementation specification.

Table 1: Oracle candidate coverage and generation counts for the stored MATH-500 run (Qwen2.5-7B-Instruct; base seed 100).
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Initial candidate coverage</td><td>445/500 (89.0%)</td></tr><tr><td>ASB candidate coverage</td><td>455/500 (91.0%)</td></tr><tr><td>Additional problems recovered Recovery among 55 initial misses</td><td>10/500 (+2.0 pp)</td></tr><tr><td>Initial model generation calls</td><td>10/55 (18.18%)</td></tr><tr><td>Additional branch calls</td><td>3,500</td></tr><tr><td>Total model generation calls</td><td>576</td></tr><tr><td></td><td>4,076</td></tr><tr><td>Average calls per problem</td><td>8.152</td></tr><tr><td>Problems with 0 / 2 / 3 branches</td><td>284 / 72 /  144</td></tr></table>

## 6 Results

Candidate expansion. The initial seven-sample collection contains a correct trajectory for 445 of the 500 problems, yielding 89.0% oracle coverage. After the directed branches are added, the expanded collection covers 455 problems, or 91.0%. The 10 newly recovered problems correspond to an absolute increase of 2.0 percentage points and recovery of 10/55 = 18.18% of the problems initially missing a correct candidate. These gains measure the enlarged solution space available to answer-selection methods.

Adaptive compute allocation. The branch policy responds to the observed answer-cluster count. For 284 problems (56.8%), the seven initial trajectories share one parsed-answer cluster, so the method retains the initial collection. For 72 problems (14.4%), two answer clusters trigger two directed exchanges. The remaining 144 problems (28.8%) receive three branches. The resulting total is 72 × 2 + 144 × 3 = 576 additional calls, corresponding to 1.152 extra calls per problem and an average of 8.152 calls per problem overall. Relative to the seven-call starting point, the generation-call budget increases by 16.46%.

Observed behavior. The key outcome is the appearance of gold-equivalent answers in trajectories generated after inspecting disagreement between the original candidates. The branch lineage and boundary metadata make these recoveries analyzable at the level of individual source–donor exchanges, prefix locations, and donor windows. This creates a direct experimental interface for studying how local diferences between reasoning paths translate into new candidate solutions.

## 7 Discussion and Evaluation Directions

The central design choice in ASB is to place collaboration at the transition between related and diverging reasoning paths. The method uses a balanced changepoint criterion to localize this transition, a short aligned donor window to introduce a competing direction, and an asymmetric prefix-conditioned operator to construct the next candidate. The result is a compact, auditable inference workflow with an explicit connection between disagreement structure and generation budget.

The reference implementation also makes several natural extensions straightforward. A matched-call independent sampling comparison will measure the value of crosstrajectory conditioning. Random boundary selection, prefix-only generation, and donor-only generation will isolate the roles of localization and information transfer. Alternative step encoders and answer-equivalence-based clustering can test sensitivity to the lexical representation and answer formatting. Repeated seeds, additional models, and a shared candidate selector will extend the evaluation from candidate discovery to downstream answer selection. Each extension reuses the stored lineage and step-alignment records.

## 8 Conclusion

We presented Adaptive Semantic Branching, a trainingfree method that converts disagreements between completed reasoning trajectories into directed, localized regeneration. Its core operation aligns source and donor reasoning steps, identifies a persistent divergence bound ary, preserves a source-specific textual prefix, and introduces a short competing reasoning window to generate a new solution. Answer-cluster structure determines how a small branch budget is allocated. On a stored MATH 500 run with Qwen2.5-7B-Instruct, the method expands correct-candidate coverage from 89.0% to 91.0%, recovering 10 initially missed problems with 576 additional generations across 500 questions. The full procedure sup plies a reproducible basis for studying divergence-guided candidate generation at test time.

## A Reproducibility Specification

Reference configuration. Table 2 collects the parameters used in the reported run. The evaluation manifest preserves its original 500-item order; each problem and each generated branch receives its own derived seed. Parsing, embedding, alignment, and scoring operate on the completed responses before branch prompts are formed.

Stable seed derivation. The seed function builds the UTF-8 string

Table 2: Reference implementation settings for the MATH-500 seed-100 run.
<table><tr><td>Parameter</td><td>Configuration</td></tr><tr><td>Model</td><td>Qwen2.5-7B-Instruct</td></tr><tr><td>Benchmark</td><td>fixed MATH-500 manifest</td></tr><tr><td>Initial trajectories</td><td>7 per problem</td></tr><tr><td>Maximum branches</td><td>3 per problem</td></tr><tr><td>Temperature / top-p</td><td>0.7 / 1.0</td></tr><tr><td>Max completion tokens</td><td>2,048</td></tr><tr><td>Base seed</td><td>100</td></tr><tr><td>Branch seed mode</td><td>independent per branch</td></tr><tr><td>Embedding dimension</td><td>512</td></tr><tr><td>Embedding features</td><td>words + character 4-grams</td></tr><tr><td>Alignment</td><td>monotonic DTW</td></tr><tr><td>Clustering key</td><td>parser-extracted answer</td></tr><tr><td>Coverage evaluator</td><td>MATH answer equivalence</td></tr></table>

```python
import hashlib
key = f"{BASE_SEED}|{sample_id}|{stage}|{agent_or_branch_id}"
seed = int.from_bytes(
hashlib.sha256(key.encode("utf-8")).digest()[:8], "big"
) % (2**31 - 1)
```

Here BASE\_SEED=100. Initial calls have distinct agent identifiers and stage names, while branches use stage="branch" and their branch identifier. The implementation also supports a source-seed experimental setting; the reported configuration uses independently derived branch seeds.

Step ofsets and embedding details. The step extractor records source-text character spans before computing embeddings. Its precedence is numbered/bulleted lines, then line breaks and punctuation, with final-answeronly fragments removed and fragments shorter than five whitespace tokens joined to the preceding step. The SHA-256 word/character feature encoder uses the first four digest bytes for the signed-hashing coordinate and the fifth byte for its sign. The final vectors are normalized in ℓ<sub>2</sub> and their cosine values enter the DTW cost matrix. The selected prefix is reconstructed using the original character ofset $d _ { A } ( k ^ { * } )$ .

Initial generation prompt. The reference run formats the following content as Qwen chat messages. The complete template is reproduced here with placeholders for the problem statement:

<|im\_start|>system   
Solve the mathematics problem independently and carefully.   
Show a complete derivation and put the final answer in \boxed{}.   
Do not use any external answer or majority information.<|im\_end   
|>   
<|im\_start|>user   
{problem}<|im\_end|>   
<|im\_start|>assistant

Branch generation prompt. For every selected di rected pair, the implementation constructs the following

prompt with the original problem, source prefix, and   
donor-local insight:   
<|im\_start|>system   
Solve the mathematics problem carefully. Show a complete   
derivation and put the final answer in \boxed{}.<|im\_end|>   
<|im\_start|>user   
Problem:   
{problem}   
{preserved\_source\_prefix\_or\_no\_reliable\_prefix\_message}   
Alternative partial reasoning direction (no final answer):   
{donor\_insight\_window}   
Continue solving the original problem. Use the alternative only   
as a competing hypothesis; do not copy it or assume it is   
correct. Explicitly check where the approaches diverge,   
derive the continuation yourself, and return a complete   
solution with the final answer in \boxed{}.   
<|im\_end|>   
<|im\_start|>assistant

For k<sup>∗</sup> > 0, the source block begins with Preserved source reasoning prefix (exact):, followed by the exact extracted source prefix and an instruction to retain it unless logically inconsistent. For k<sup>∗</sup> = 0, the source block requests a fresh solution that considers the competing partial direction. The donor window contains up to three aligned consecutive steps, and any occurrences of the donor final-answer string are masked during prompt construction.

Run artifacts and lineage. The experiment produces run\_config.json, results.jsonl, summary.json, and gold\_recovery\_cases.jsonl. Each stored initial trajectory includes prompt and response text, parsed answer, sampling seed, finish reason, and token metadata when supplied by the generation backend. Each branch additionally stores its source and donor trajectory identifiers, answer clusters, divergence step, raw ofset, J<sub>max</sub>, prefix ratio, extracted donor insight, branch score, seed, prompt, and generated response. These fields support reconstruction of the pair-selection decisions and inspection of newly recovered answers.

## References

Maciej Besta et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024. doi: 10.1609/aaai.v38i16.29720.

Chenkang, Fan Yu, Junjie Nian, Sihan Zhao, Zhuoka Feng, Zijun Yao, Wang Heng, Yu Minshen, and Yixin Cao. Thinking traps in long chain-of-thought: A measurable study and trap-aware adaptive restart. In Findings of the Association for Computational Linguistics: ACL 2026, 2026. URL https://aclanthology.org/ 2026.findings-acl.1930/.

Dan Hendrycks et al. Measuring mathematical problem solving with the math dataset. In Advances in Neural

Information Processing Systems, 2021. URL https: //arxiv.org/abs/2103.03874.

Feng Ju, Zeyu Qin, Rui Min, Zhitao He, Lingpeng Kong, and Yi R. Fung. Reasoning path divergence: A new metric and curation strategy to unlock llm diverse thinking, 2025. URL https://arxiv.org/abs/2510. 26122.

Rui Li and Shuang Cao. From trajectories to graphs: Contract-checked editing for verifier-guided llm reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, 2026. URL https://aclanthology.org/2026. acl-long.2004/.

Xianzhi Li, Ethan Callanan, Abdellah Ghassel, and Xiaodan Zhu. Entropy-gated branching for eficient test-time reasoning. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics, 2026. URL https: //aclanthology.org/2026.eacl-long.235/.

Hunter Lightman et al. Let’s verify step by step, 2023. URL https://arxiv.org/abs/2305.20050.

Hiroaki Sakoe and Seibi Chiba. Dynamic programming algorithm optimization for spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1):43–49, 1978. doi: 10.1109/TASSP. 1978.1163055.

Xuezhi Wang et al. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2203.11171.

An Yang et al. Qwen2.5 technical report, 2024. URL https://arxiv.org/abs/2412.15115.

Wei Yang, Shixuan Li, Gengshuo Liu, Heng Ping, Peiyu Zhang, Paul Bogdan, and Jesse Thomason. Auditing multi-agent llm reasoning trees outperforms majority vote and llm-as-judge. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, 2026. URL https://aclanthology.org/ 2026.emnlp-main.2642/.

Shunyu Yao et al. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2305.10601.

Ori Yoran, Tomer Wolfson, Ben Bogin, Uri Katz, Daniel Deutch, and Jonathan Berant. Answering questions by meta-reasoning over multiple chains of thought. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https: //aclanthology.org/2023.emnlp-main.364/.