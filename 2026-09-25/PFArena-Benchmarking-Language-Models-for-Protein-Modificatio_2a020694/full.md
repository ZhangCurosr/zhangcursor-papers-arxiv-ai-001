# PFArena: Benchmarking Language Models for Protein Modification

Shanghai Artificial Intelligence Laboratory Generative Symbolic Intelligence Lab (GenSI), Tsinghua University Institute for AI Industry Research (AIR), Tsinghua University School of Pharmaceutical Sciences, Tsinghua University

§ Github

![](images/41f0c4ca848a47ae3d5f129981aac1a05eab15325bad51f8de5f69f0490d0a38.jpg)

õ Dataset

![](images/5aaa5682b9fde29e29fc650351d78ece562b25f61e16a3119916a34ec3eac9fe.jpg)

 Leaderboard

![](images/0610c9a1daebaca5a0217b00cd9651b41efcd82a433f7a7d1a8c8bcde4cd348d.jpg)

<table><tr><td>T4</td><td>Single-mutant-informed multi-mutant ranking</td></tr><tr><td>W 0.646</td><td>AMixKit + Claude</td></tr><tr><td>Biomni + Claude</td><td>0.645</td></tr><tr><td>信</td><td>AMixKit + GPT 0.638</td></tr><tr><td>Biomni + GPT</td><td>0.637</td></tr><tr><td>GPT-6 Astra</td><td>0.599</td></tr><tr><td>米 Claude Opus 5</td><td>0.582</td></tr><tr><td>A AMixKit + AMix-2.1</td><td>0.553</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.533</td></tr><tr><td>Q DeepSeek V4 Pro</td><td>0.531</td></tr><tr><td>Z GLM-5.2</td><td>0.496</td></tr><tr><td>K Kimi K3</td><td>0.479</td></tr><tr><td>VenusREM</td><td>0.382</td></tr><tr><td>ProSST</td><td>0.378</td></tr><tr><td>Mila S3F-MSA</td><td>0.366</td></tr><tr><td>Mila S3F</td><td>0.349</td></tr><tr><td>0ESM2</td><td>0.315</td></tr><tr><td>Oenarr ProGen2</td><td>0.257</td></tr></table>

Figure 1 Benchmark performance across the four protein modification tasks in PFArena. Method classes are distinguished by cell background color: protein language models, large language models, and agents. For agents, Claude denotes Claude Opus 5 and GPT denotes GPT-6 Astra. Methods are ranked by Recall@40 for task T1 and by Spearman correlation for tasks T2–T4. Results for the remaining metrics are detailed in the experiments section.

## Contents

Introduction 3   
2 PFArena 4   
2.1 Task Suite   
2.1.1 T1: Single-mutant generation   
2.1.2 T2: Measurement-free multi-mutant ranking 6   
2.1.3 T3: Anchor-informed multi-mutant ranking 6   
2.1.4 T4: Single-mutant-informed multi-mutant ranking   
2.2 Evaluation Objectives   
2.3 Data Construction 8   
2.3.1 Assay Collection and Harmonization 8   
2.3.2 Quality Control 9   
3 Experimental Setup 10   
3.1 Baselines 10   
3.2 Metrics 10   
3.3 Implementation Details 11   
4 Main Results 13   
4.1 Overall Results 13   
4.2 Task-specific Results 14   
4.2.1 T1: Single-mutant generation 14   
4.2.2 T2: Measurement-free multi-mutant ranking 14   
4.2.3 T3: Anchor-informed multi-mutant ranking 16   
4.2.4 T4: Single-mutant-informed multi-mutant ranking 16   
5 Analysis 17   
5.1 Comparing PLMs and LLMs 17   
5.1.1 Divergent Single-mutant Proposal Strategies 17   
5.1.2 Utilization of Multi-mutant Component Additivity 18   
5.1.3 Common Limitations across Single-mutant and Multi-mutant Tasks 19   
5.2 Dive into LLMs 20   
5.2.1 Feedback-guided In-context Adaptation 20   
5.2.2 Uncertainty Estimation 21   
5.2.3 Test-time Scaling 23   
6 Related Work 23   
7 Conclusion 24   
A Evaluation Templates 30   
A.1 Single-mutant Generation 30   
A.2 Multi-mutant Ranking 31   
A.3 Confidence Elicitation for Uncertainty Estimation 34   
B Experimental Setup 36   
B.1 PLM Deployment . 36   
B.2 Test-time Scaling Methods 37   
C Extended Experimental Analysis . 39   
C.1 Evaluation Completeness and Missing-Query Handling . 39   
C.2 Complementary Strengths and Failure Modes of LLMs and PLMs . 40

## Abstract

Protein modification requires navigating an immense sequence space, yet wet-lab validation remains lowthroughput and costly. Although computational paradigms including protein language models (PLMs), large language models (LLMs), and LLM-based agents have shown promise in protein modification, their relative eficacy across realistic experimental decision-making settings remains unclear. To bridge this gap, we introduce PFArena, a benchmark comprising four controlled task interfaces that cover single-mutant generation and multi-mutant ranking. By providing varying levels of mutation fitness data, PFArena reflects four representative research scenarios characterized by difering degrees of prior experimental context. We assess six PLMs, six LLMs, and five LLM-based agents using complementary metrics to measure both peak and overall protein modification performance. Our evaluation reveals that model performance shifts systematicall with the availability of target-specific experimental evidence: PLMs demonstrate proficiency in open-ended single-mutant generation by leveraging protein-specific priors, whereas LLMs and agents perform strongly in multi-mutant ranking, particularly when target-specific fitness data are available. Nevertheless, all model families face fundamental challenges with the increase of search space and mutation depth. We release our code and benchmark suite to facilitate reproducible research in model-assisted protein modification.

## 1 Introduction

Protein modification is central to protein engineering, fundamentally relying on sequence mutations to alter protein properties. This process results in an enormous number of possible substitutions and combinations, making experimental characterization costly, time-consuming, and limited in throughput. Reliable fitness predictions allow researchers to prioritize mutants with the highest expected value and focus measurements on informative regions of the sequence–fitness landscape, thereby accelerating iterative design–build–test-learn cycles and supporting the discovery of mutations dificult to identify through empirical screening alone.

Computational models, including protein language models (PLMs) and large language models (LLMs), have shown substantial potential for protein modification. PLMs learn sequence, structural, and evolutionary constraints from large-scale protein data, enabling zero-shot prediction and prioritization of functional mutants [1–6]. General-purpose LLMs have also advanced rapidly on protein-specific prediction tasks. On ProteinGym Hard, the reported score increased from 37.7% for Claude Opus 4.7 to 39.6% for Claude Opus 4.8, followed by a further 7.7-percentage-point gain for Claude Opus 5 [7, 8]. This sustained improvement highlights the growing ability of LLMs to interpret protein sequences through natural-language interaction. Nevertheless, it remains unclear when specialized PLMs, general-purpose LLMs, or LLM-based agents are preferable across the diverse decision settings encountered in protein modification.

To answer this question, we introduce PFArena, an assay-grounded benchmark that systematically evaluates these model families under four protein modification settings: T1: Single-mutant generation, T2: Measurement-free multi-mutant ranking, T3: Anchor-informed multi-mutant ranking and T4: Singlemutant-informed multi-mutant ranking. These settings vary along two axes: the action space and the available experimental evidence. They encompass both open-ended generation and fixed-pool ranking tasks, and cover conditions with diferent mutant information supplied. This design enables a systematic comparison between PLMs and LLMs across single- and multi-mutant prioritization, while testing whether LLMs can translate target-specific experimental evidence into better mutation ranking.

We evaluate six PLMs and six LLMs across four tasks covering 202 unique assays and 293 task instances. Furthermore, to combine the reasoning capabilities of LLMs with protein-specific tools, we evaluate five LLM-based agents built on both the pioneering Biomni framework [9] and our lightweight AMixKit toolkit, detailed in Section 3.3. Our study yields two main findings:

• Divergent strengths. No single model family dominates across all four tasks. PLMs hold an overall advantage in the open-ended T1 setting, where no target-specific fitness measurements are available and performance depends largely on protein-specific sequence, structural, and evolutionary priors. Their predictions also span a broader range of residue-class substitutions. In contrast, LLM-based systems perform strongly in the multi-mutant ranking tasks (T2–T4). Their advantage is most evident when comprehensive target-specific experimental evidence is provided (T4). Together, these results highlight the complementary strengths of protein-specific representations and evidence-guided reasoning.

![](images/d0031dd9dcff1c78cc34a1cf9a2394744d47a90de0cf5ad11e3b9a597c32e723.jpg)  
Figure 2 PFArena evaluates protein modification from the perspective of experimental scientists. The benchmark is organized around four representative research scenarios that correspond to the questions scientists ask when deciding which mutants to study next. T1: Single-mutant generation generates high-performing single mutations without candidate input. T2: Measurement-free multi-mutant ranking ranks multi-mutant candidates without prior fitness measurements. T3: Anchor-informed multi-mutant ranking ranks multi-mutant successors conditioned on a tested anchor mutant. T4: Single-mutant-informed multi-mutant ranking ranks multi-mutant candidates using measured single-mutant fitness information. These scenarios span single- and multi-mutant discovery under progressively richer experimental evidence, enabling systematic comparison of PLMs, LLMs, and agents in realistic protein-engineering workflows. The protein structure depicted here corresponds to PDB entry 1RQC [10].

• Convergent bottlenecks. Despite these distinct strengths, all three model families encounter shared limitations. In T1, even the best-performing method recovers fewer than 5% of the ground-truth top-40 mutations, showing that the single-mutant search space remains dificult to cover under a limited prediction budget. Across T2–T4, all model families show a substantial drop in ranking accuracy beyond two substitutions, revealing a shared sensitivity to mutation depth.

## 2 PFArena

Conventional mutant-efect benchmarks typically assess whether a model can assign fitness scores to prespecified mutations [11, 12]. In contrast, PFArena evaluates whether PLMs, LLMs, and agents can prioritize mutants across four task interfaces. These interfaces vary in candidate space, spanning single substitutions and mutation combinations, and the amount and form of target-specific evidence available in each task.

## 2.1 Task Suite

As summarized in Figure 2, PFArena defines four task interfaces that correspond to common decisions in protein-engineering workflows. T1 considers the open-ended generation of single-mutant candidates from the full legal substitution space, whereas T2–T4 consider the ranking of a supplied pool of multi-mutant candidates under diferent levels of target-specific experimental evidence.

<table><tr><td>Task component</td><td>Specification</td></tr><tr><td>Definition</td><td>Budgeted generation of legal single substitutions without target-specific mutation mea- surements.</td></tr><tr><td>Input</td><td>Wild-type sequence x, assay context c and proposed budget K.</td></tr><tr><td>Output</td><td>Proposed mutation set of at most K single substitutions.</td></tr><tr><td>Instance</td><td>Input:</td></tr><tr><td></td><td>Wild-type sequence: MSSSLGKE.. .AMV</td></tr><tr><td></td><td>Assay context:</td></tr><tr><td></td><td>- Primary task class: activity function.</td></tr><tr><td></td><td>- Fitness type: organismal or cellular fitness.</td></tr><tr><td></td><td>- Readout subclass: growth or selection proxy.</td></tr><tr><td></td><td>Proposed budget K: 40</td></tr><tr><td></td><td>Output:</td></tr><tr><td></td><td>Proposed mutation set: P557L, K369Y, N434D, ....</td></tr><tr><td>Statistics</td><td>Assays: 123; measured single mutants: 594,695.</td></tr></table>

Table 1 Task specification for T1: Single-mutant generation.

Formal task definitions are provided in the following subsections, and the specification tables summarize the information available at the benchmark level. Because model families expose diferent native interfaces, this information is presented through model-appropriate interfaces. Natural-language task inputs are provided directly to LLMs and agents, whereas PLMs receive the subset supported by their architectures.

## 2.1.1 T1: Single-mutant generation

In a single-mutant campaign like deep mutational scanning (DMS), researchers assay a broad panel of legal single substitutions around a wild-type protein and use the measurements to identify mutants with favorable assay-specific phenotypes. Task T1 evaluates the analogous budgeted prioritization problem: given the wild-type sequence and assay context but no target-specific mutation measurements, which single substitutions should be selected first under a fixed experimental budget?

Table 1 summarizes the task specification. Formally, given a wild-type sequence x, assay context c, and a generation budget K, the model implements a generation function:

$$
f _ { \theta } : ( x , c , K ) \mapsto \widehat { \mathcal { O } } , \qquad \widehat { \mathcal { O } } \subseteq \mathcal { V } ( x ) , \qquad | \widehat { \mathcal { O } } | \leq K ,
$$

where V(x) denotes the legal single-substitution space of a canonical wild-type sequence x. For a sequence of length L, this space contains 19L possible substitutions under the standard amino-acid alphabet. The model must therefore return a limited set of legal candidates rather than a complete ordering of the entire space.

T1 assesses highly selective discovery in a large single-mutant space under a finite experimental budget. A successful system should identify at least one highly promising mutation while also exploring a suficiently broad region of the high-fitness landscape. For smaller PLMs, the task can be approached by evaluating all legal substitutions individually and ranking their scores. For LLMs and agents, exhaustive evaluation is generally impractical because of the associated computational and financial costs, so these systems must directly generate a small candidate set from the full substitution space.

Single-mutant discovery provides a natural starting point for protein engineering, but many desired phenotypes depend on combinations of substitutions that may reinforce, compensate for, or interfere with one another. The remaining tasks therefore move from open-ended single-mutant generation to fixed-pool ranking of multi-mutant candidates with progressively richer levels of target-specific experimental evidence made available.

<table><tr><td>Task component</td><td>Specification</td></tr><tr><td>Definition</td><td>Ranking multi-mutant candidates without target-specific mutation measurements.</td></tr><tr><td>Input</td><td>Wild-type sequence x, assay context  $c ,$  and a score-blind candidate pool.</td></tr><tr><td>Output</td><td>Complete ranking of the supplied candidate pool.</td></tr><tr><td>Instance</td><td>Input: Wild-type sequence: MLEGKVKW...KEA Assay context: - Primary task class: stability.</td></tr><tr><td></td><td>Assays: 74; candidate number: 6,288.</td></tr></table>

Table 2 Task specification for T2: Measurement-free multi-mutant ranking.

## 2.1.2 T2: Measurement-free multi-mutant ranking

Task T2 evaluates multi-mutant prioritization as fixed-pool ranking rather than open-ended generation. Given the wild-type sequence, assay context and a score-blind pool sampled from measured multi-mutants, the model returns a permutation of that pool for downstream experimental evaluation.

Table 2 summarizes the task specification. Formally, given a wild-type sequence x, assay context $c ,$ and a score-blind multi-mutant candidate pool C, the model implements a ranking function:

$$
f _ { \theta } : ( x , c , \mathcal { C } ) \mapsto \widehat { \mathcal { O } } , \widehat { \mathcal { O } } \in \Pi ( \mathcal { C } ) .
$$

Here, Π(C) denotes the set of all permutations of the candidate pool.

T2 assesses measurement-free combinatorial reasoning. Specifically, the model must prioritize variants using the available protein information and assay description while accounting for possible interactions among substitutions. It does not test whether a model can generate new combinations outside the supplied pool or assign calibrated fitness values to arbitrary mutants.

## 2.1.3 T3: Anchor-informed multi-mutant ranking

Task T3 evaluates successor ranking conditional on one measured sequence background. In the current release, 65 of 67 anchors are single substitutions; one anchor contains two substitutions and one contains three.

Table 3 summarizes the task specification. Formally, given a wild-type sequence x, assay context $c ,$ and a measured anchor mutant u with score $y _ { u } .$ , let C denote the supplied score-blind successor pool. Every candidate $v \in { \mathcal { C } }$ strictly contains the substitutions in u and introduces at least one additional substitution relative to this measured background. The model then implements a ranking function:

$$
f _ { \theta } : \left( x , c , u , y _ { u } , \mathcal { C } \right) \mapsto \widehat { \mathcal { O } } , \qquad \widehat { \mathcal { O } } \in \Pi ( \mathcal { C } ) .
$$

T3 assesses anchor-conditioned reasoning. The measured anchor provides a local experimental reference; thus, the model must determine whether additional substitutions are likely to improve or reduce fitness relative to the measured sequence background. This setting tests whether information from one experimentally characterized mutant can be transferred to the prioritization of related multi-mutant candidates.

<table><tr><td>Task component</td><td>Specification</td></tr><tr><td>Definition</td><td>Ranking strict successors of a measured anchor mutant.</td></tr><tr><td>Input</td><td>Wild-type sequence x, assay context c, a measured anchor mutant u with score yu, and a score-blind successor pool C.</td></tr><tr><td>Output</td><td>Complete ranking of the supplied candidate pool.</td></tr><tr><td>Instance</td><td>Input:</td></tr><tr><td></td><td>Wild-type sequence: QVQLVQSG...VSS</td></tr><tr><td></td><td>Assay context:</td></tr><tr><td></td><td>- Primary task class: binding.</td></tr><tr><td></td><td>- Fitness type: binding.</td></tr><tr><td></td><td>- Readout subclass: binding.</td></tr><tr><td></td><td>Anchor: T28P with score -1.586.</td></tr><tr><td></td><td>Successor pool:</td></tr><tr><td></td><td>{T28P+S30R+N59K+T76A, T28P+S30R+N59K+Q62P+S75F, T28P+S30R+L104V, ...}.</td></tr><tr><td></td><td>Output:</td></tr><tr><td></td><td>Ranked candidate set: T28P+S30R+N59K+Q62P+S75F, T28P+S30R+N59K+T76A, T28P+S30R+L104V, ...</td></tr><tr><td>Statistics</td><td>Assays: 67; candidate number: 3,648.</td></tr></table>

Table 3 Task specification for T3: Anchor-informed multi-mutant ranking.

## 2.1.4 T4: Single-mutant-informed multi-mutant ranking

Whereas T3 exposes one measured anchor, T4 exposes a broader set of measured single-mutant scores. This context is not required to cover the complete 19L single-substitution space. Instead, the construction requirement is candidate-specific completeness: every component substitution represented in a candidate has a corresponding measured single-mutant entry.

Table 4 summarizes the task specification. Formally, given a wild-type sequence x and assay context c, let S map each provided measured single substitution s to its assay score y(s), and let C denote the supplied score-blind multi-mutant candidate pool, whose component substitutions are all represented in S. The model then implements a ranking function:

$$
f _ { \theta } : ( x , c , \mathcal { S } , \mathcal { C } ) \mapsto \widehat { \mathcal { O } } , \qquad \widehat { \mathcal { O } } \in \Pi ( \mathcal { C } ) .
$$

T4 assesses component-informed combination ranking. Given the measured efects of all individual mutations, the model must prioritize multi-mutant combinations. Rather than simply adding individual scores, the system must account for non-additive interactions to translate distributed evidence into accurate rankings.

## 2.2 Evaluation Objectives

Table 5 summarizes the objective of each metric and its applicability to each task. Implementation details of the corresponding metrics are provided in Section 3.2.

• Global ranking with Spearman Correlation. Measures whether the predicted ordering agrees with the experimental ordering across the complete candidate pool, evaluating a model’s ability to distinguish relative fitness throughout the library rather than only among the candidates placed at the top.

• Top-weighted ranking with NDCG. Measures the quality of the upper part of the predicted list by assigning greater importance to candidates near the top, evaluating whether a model places the most promising variants in positions that are most useful for downstream experimental testing.

<table><tr><td>Task component</td><td>Specification</td></tr><tr><td>Definition</td><td>Multi-mutant ranking with measured context for every component substitution.</td></tr><tr><td>Input</td><td>Wild-type sequence x, assay context c, a measured single-mutant context map S, and a score-blind multi-mutant candidate pool C.</td></tr><tr><td>Output</td><td>Complete ranking of the supplied candidate pool.</td></tr><tr><td>Instance</td><td>Input: Wild-type sequence: EVKLDETG...EIK Assay context:</td></tr><tr><td></td><td>- Primary task class: stability. - Fitness type: abundance or expression.</td></tr><tr><td></td><td>- Readout subclass: cellular abundance stability proxy. Single-mutant context: {W108E: -0.354, G109P: -1.062, M34Q: 0.605, N35S: 1.154, ...}.</td></tr><tr><td></td><td>Candidate pool: {W108E+G109P, Y102P+M105K, M34Q+N35S, ...}.</td></tr><tr><td></td><td>Output:</td></tr><tr><td></td><td>Ranked candidate set: M34Q+N35S, Y102P+M105K, W108E+G109P, ...</td></tr></table>

Table 4 Task specification for T4: Single-mutant-informed multi-mutant ranking.

<table><tr><td>Objective</td><td>Metric</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td></tr><tr><td>Global ranking</td><td>Spearman Correlation</td><td>X</td><td>√</td><td>V</td><td>√</td></tr><tr><td>Top-weighted ranking</td><td>NDCG</td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Peak discovery</td><td>Normalized Maximum Score@K</td><td></td><td></td><td>VVVV</td><td></td></tr><tr><td>Peak coverage</td><td>Recall@K</td><td></td><td></td><td>VVVV</td><td></td></tr></table>

Table 5 Evaluation objectives, corresponding metrics, and metric applicability across the four tasks. Ranking metrics are not used for T1 because it is a generation task instead of a ranking task.

• Peak discovery with Normalized Maximum Score@K. Measures the highest assay-normalized fitness among the K submitted candidates, evaluating whether a model can identify at least one strong candidate within the available budget of K proposed mutations.

• Peak coverage with Recall@K. Measures how broadly a model recovers the ground-truth top-performing candidates within a prediction budget K, complementing peak discovery by assessing whether the model identifies multiple candidates from the high-fitness region rather than only a single strong candidate.

## 2.3 Data Construction

## 2.3.1 Assay Collection and Harmonization

We assembled assay-level protein fitness landscapes from ProteinGym [11], MaveDB [13], MegaScale [14], FLAb [15], Human Domainome [16], and CombinGym [17]. The collection also includes manually curated target DMS studies for CDKN2A [18], SLC13A5 [19], and TrpB [20]. These sources provide stability, binding, activity or organismal function measurements spanning single-mutant and combinatorial sequence–fitness landscapes. An assay is defined by its wild-type construct, experimental condition, readout, and score semantics, so measurements for one protein remain separate assays when they represent distinct experimental settings or engineering objectives. Each assay record contains biological and experimental metadata together with one or more assay-linked A3M alignments generated with MMseqs2 [21] against UniRef100 [22]. We curated an assay’s primary task class, fitness type and readout subclass based on DMS databases and papers, where primary task class is artificially set to 3 diferent types: stability, binding, or activity function.

We harmonized mutant representations and score semantics across sources. A single substitution is encoded as A12V, and a multi-mutant as a plus-separated combination such as A12V+G35L. We retained mutants composed of the 20 standard amino acids and removed synonymous substitutions, stop codons, deletions, non-finite scores, repeated mutation sites within a mutant, and mutations that could not be reconstructed from the assay-specific wild-type sequence. Repeated measurements of the same canonical mutant within an assay were averaged. Assay-specific score semantics are retained for the score-construction step below.

For each assay, we calculate DMS\_score from the assay-level efect values obtained from the source file or reconstructed from the source readout specified in the assay metadata. Published processed efect columns are imported directly; assays requiring reconstruction are converted according to their measurement semantics. For example, afinity values reported as $K _ { \mathrm { d } }$ are expressed $\mathrm { a s - l o g _ { 1 0 } } ( K _ { \mathrm { d } } )$ , and expression ratios are expressed as $\log _ { 1 0 } ( \mathrm { E R } )$ . The resulting efect values are standardized within each assay using a population z-score. For readouts in which lower values indicate better performance, the corresponding efect values are sign-reversed, so larger DMS\_score values consistently indicate better assay performance. The resulting assay-level scores populate the measured single-mutant tables and the candidate records used by T1–T4.

## 2.3.2 Quality Control

Every constructed assay–task instance underwent three groups of quality-control checks covering instance integrity, task consistency, and evaluation reliability:

• Instance integrity. We first verified the correctness and completeness of each individual instance. Checks included mutation–sequence consistency, finite fitness values, complete metadata, absence of duplicate mutants or rows, and valid links to the corresponding A3M alignments. These checks ensure that each instance faithfully represents the underlying assay data, contains information required for downstream evaluation, and avoids malformed or incomplete records that could introduce artificial errors during model inference or metric computation throughout the evaluation pipeline.

• Task consistency. We then validated whether each assay was correctly instantiated under the corresponding task definition. This included checking task-specific inputs, candidate-set construction, and mutation constraints, and task-specific formatting and consistency requirements. For assays shared across multiple task interfaces, we additionally cross-checked sequence, measurement, and visual data to ensure that the same underlying assay information remained consistent across tasks and that no discrepancies were introduced during task-specific preprocessing or instance conversion.

• Evaluationreliability. Finally, we examined whether individual instances supported stable and meaningful assessment across all reported evaluation metrics. We excluded severe tie-heavy T2–T4 instances and T1 instances in which exact top-region ties made top-K membership ambiguous. Local near-ties that did not alter the relevant ranking boundary were retained and documented. We also tested for accidental order leakage by measuring monotonicity and rank correlation between row position and DMS\_score, ensuring that benchmark ordering did not provide unintended information about the target labels or create spurious performance gains unrelated to genuine mutant prioritization ability.

After quality filtering, PFArena comprises 202 unique assays, 293 assay–task instances, and 607,269 target candidate rows. Per-task instance counts are listed in the task specification tables. Since tasks impose diferent evidence and candidate constraints, the same assay may contribute to multiple tasks. At the assay–task level, the current benchmark contains 124 stability, 101 activity or organismal function, and 68 binding instances. Its source composition comprises 76 MegaScale, 78 ProteinGym, 44 MaveDB, 34 FLAb, 36 Human Domainome, 19 CombinGym and 6 manually curated DMS instances.

## 3 Experimental Setup

## 3.1 Baselines

We compare four baseline families: random selection, zero-shot protein language models, standalone generalpurpose LLMs, and LLM-based scientific agents.

Random For T1, the random baseline uniformly samples K legal single substitutions without replacement.   
For T2–T4, it produces a uniformly random permutation of the supplied candidate pool.

Protein Language Models We evaluated six zero-shot protein language models spanning complementary biological input modalities. ESM-2 (650M) [1] and ProGen2-base (764M) [2] use protein sequences alone; ProSST-2048 [3] incorporates discrete structure tokens; S3F [4] integrates sequence, backbone, and surface representations; and VenusREM [5] additionally incorporates MSA-derived evolutionary information. S3F-MSA combines S3F with an ensemble of five independently trained EVE models [6].

LLMs We evaluated six general-purpose LLMs from diverse model families: GPT-6 Astra [23], Claude Opus 5 [8], Gemini 3.1 Pro [24], Kimi K3 [25], GLM-5.2 [26], and DeepSeek-V4-Pro [27]. All models were evaluated without any fine-tuning. For each task, they received the same prompt template and the same model-visible inputs defined in Section 2.1, and were required to return predictions in the prescribed structured format.

Agents We additionally evaluated scientific agent frameworks Biomni [9] and our AMixKit toolkit. They received the same task-visible inputs and followed the same structured-output requirements as the standalone LLMs, while retaining the ability to orchestrate the tools and scientific resources provided by their frameworks.

## 3.2 Metrics

We compute all metrics independently for each assay-level instance and then macro-average the resulting values, so that every instance contributes equally to the reported performance. After score harmonization during data construction, larger DMS\_score values consistently indicate better fitness. We use task-specific values of K to account for diferences in task formulation and candidate-space size. In T1, models generate candidates from a large single-mutant space, so we set K = 40 to provide a suficiently broad generation budget. In T2–T4, models rank a smaller supplied candidate pool, so we set K = 5 to maintain a selective evaluation of the top-ranked candidates within the reduced candidate space.

Spearman Correlation Measures agreement between the predicted and experimental rankings over the complete candidate pool. For an instance with n candidates, we assign each candidate an experimental rank and a predicted rank, and compute the Pearson correlation between the two rank vectors. Ties in the experimental scores are assigned their average rank.

NDCG Evaluates ranking quality by placing greater weight on high-fitness candidates near the top of the list. We first apply min-max normalization to the DMS\_score values to map them into relevance scores $r _ { i } = ( s _ { i } - s _ { \operatorname* { m i n } } ) / ( s _ { \operatorname* { m a x } } - s _ { \operatorname* { m i n } } ) \in [ 0 , 1 ]$ . For a predicted permutation π of n candidates, NDCG is defined as:

$$
\mathrm { N D C G } ( \pi ) = \frac { \mathrm { D C G } ( \pi ) } { \mathrm { I D C G } } , \qquad \mathrm { D C G } ( \pi ) = \sum _ { j = 1 } ^ { n } \frac { 2 ^ { r _ { \pi ( j ) } } - 1 } { \log _ { 2 } ( j + 1 ) } ,
$$

and IDCG represents the ideal maximum possible DCG, computed by evaluating the same discounted gain formula after sorting all candidates in decreasing order of true relevance $r _ { i } .$

Normalized Maximum Score Measures peak discovery among the submitted candidates. Let $\mathcal { H } _ { K }$ denote the legal candidates occurring within the first K submitted positions. We define

$$
\operatorname { N M S } @ K = \frac { \operatorname* { m a x } _ { i \in \mathcal { H } _ { K } } s _ { i } - s _ { \operatorname* { m i n } } } { s _ { \operatorname* { m a x } } - s _ { \operatorname* { m i n } } } .
$$

For T1, $\mathcal { H } _ { 4 0 }$ is the submitted single-mutant set under the fixed generation budget. In this task, $s _ { \mathrm { m i n } }$ and $s _ { \mathrm { m a x } }$ denote the minimum and maximum experimental scores over the full measured single-mutant space. For T2–T4, $\mathcal { H } _ { 5 }$ consists of the leading candidates in the predicted multi-mutant ranking, and $s _ { \mathrm { m i n } }$ and $s _ { \mathrm { m a x } }$ are computed over the corresponding supplied candidate pool.

Recall Measures coverage of the experimentally high-fitness region. Let $\mathcal { T } _ { K }$ denote the ground-truth top-K candidates, with all candidates tied at the ground-truth cutof included as relevant. For an instance with candidate-pool size n, we compute Recall@K as:

$$
{ \mathrm { R e c a l l @ } } K = { \frac { | { \mathcal { H } } _ { K } \cap { \mathcal { T } } _ { K } | } { \operatorname* { m i n } ( K , n ) } } .
$$

The denominator remains min $( K , n )$ when ties expand the ground-truth relevant set. Recall complements NMS, which depends only on the strongest recovered candidate.

LLM and agent outputs are not fully controllable and may include refusals, incomplete responses, or outputs that cannot be parsed successfully. The post-processing procedures for missing or incomplete assay responses are described in Section C.1 and applied uniformly across all models.

## 3.3 Implementation Details

We used a common evaluation pipeline across all baseline families. Task-visible inputs and required output formats followed Section 2.1, while the model-specific inference procedures are described below.

PLM Inference PLM inference used label-free, zero-shot mutation scores under fixed model-specific configurations, without natural-language assay descriptions or task-specific auxiliary information. Wild-type structures were generated with AlphaFold 3 [28], and evolutionary alignments were constructed against UniRef100 [22] using MMseqs2 [21]. For T1, the scores were used to rank all legal single substitutions; for T2–T4, they were used to rank the supplied candidate sets. Model-specific checkpoints, scoring definitions, and protocols for multi-substitution, multi-chain, and long-sequence inputs are documented in Section B.1.

LLM Inference Each prompt was instantiated from a task-specific template, provided in Section A, and submitted as a single non-streaming request through an asynchronous OpenAI-compatible client. The models gpt-6-astra, claude-opus-5, gemini-3.1-pro-preview, kimi-k3, glm-5.2, and deepseek-v4-pro were required to return a task-specific structured output: a proposed mutation set for T1 and a proposed ranking for T2–T4. Default maximum completion lengths were set to 65,536 tokens for gemini-3.1-pro-preview, kimi-k3 and glm-5.2, and 32,768 tokens for the remaining models. Sampling parameters and reasoning-efort settings were left at their API defaults for all evaluated model configurations. Responses that could be successfully parsed into the required structure were recorded as results, without repairing, replacing, or filtering predicted mutations before evaluation. Failed instances were retried for up to 10 complete inference runs, until all pending instances succeeded or a run produced no additional successful results.

Agent Inference We used Biomni [9] to instantiate each instance as an independent task with a task-specific agent prompt. We evaluated two backbone models: gpt-6-astra and claude-opus-5. Unlike standalone LLM inference, each agent operated in a multi-turn loop with iterative tool use and intermediate feedback. At each turn, the agent either emitted an <execute> block to invoke a Biomni tool, inspect public database or model outputs, or run focused Python/Bash code, or emitted a final <solution> block. After each tool-use turn, the observation was returned to the agent before the next turn. We required at least one execution–observation round before accepting a final answer and allowed at most 40 tool-use rounds. Each model call was limited to a maximum of 24,576 output tokens. Once the tool-use budget was exhausted, further tool calls were disabled and the agent was required to return the final answer in the specified structured-JSON format. GPU-intensive Biomni tools and eligible code-execution blocks were dispatched to a remote GPU worker, while the agent remained responsible for selecting actions and producing the final ranking. Only the structured JSON object within the <solution> block was parsed as the final response.

![](images/a01bedef9e7d901d49b24e441bf0a6c3a3b2905244f88eae1e35ccae3ffc5abc.jpg)  
Figure 3 Overview of AMixKit, integrating protein features, PLM inference, and mutant construction.

AMixKit Toolset We additionally evaluated agents equipped with AMixKit, a lightweight toolkit that provides standardized protein-specific operations for evidence retrieval, mutation construction, and PLM-based prediction. As illustrated in Figure 3, AMixKit contains seven tools:

• UniProt Features: retrieves functional sites, domains, structural annotations, natural variants, and curated mutagenesis records for a specified protein accession.

• Assay Features: summarizes assay-specific evolutionary or structural evidence, including MSA coverage and conservation or AlphaFold pLDDT confidence for an exact benchmark assay.

• Centroid Distance: ranks residue positions by their distance to the structural centroid, with chain-aware handling of multichain proteins, to identify structurally central or peripheral sites.

• Amino Acid Retrieval: retrieves the wild-type amino acid at specified sequence positions.

• Build Mutant: constructs and validates single-substitution annotations from selected positions and target amino acids while enforcing wild-type sequence consistency.

• PLM Mutant Suggestion: generates and ranks the 19 possible non-wild-type substitutions at selected positions using a specified PLM and reports the highest-scoring candidates per position.

• PLM Mutant Scoring: assigns PLM-based fitness scores to a set of single- or multi-mutant candidates.

The PLM tools support ESM-2, ProGen2, ProSST, S3F, S3F-MSA, and VenusREM, using cached scores when available and the scoring service otherwise. Collectively, these tools allow agents to gather biological evidence, construct valid mutations, and incorporate protein-model predictions in one workflow. AMixKit agents followed the same multi-turn execution–observation protocol as the Biomni agents.

We evaluate AMixKit with both frontier LLMs and our in-house model, AMix-2.1. Frontier LLMs demonstrate strong capabilities, but may still exhibit over-refusal or false refusals on biology-related tasks. Moreover, their API costs can become substantial when deployed at scale. Motivated by these limitations, we developed AMix-2.1 by scaling AMix-2 [29] to hundreds of billions of parameters and applying multi-task agentic RL. We will release further technical details and open-source both AMix-2.1 and AMixKit in future work.

![](images/ed212a14e39d9ed24178947392e9d0c2ba8cb3b6fad359e6f6890032d8823b04.jpg)  
Figure 4 Task-level and metric-level rank-based model capability. For each task and metric, all methods are ranked according to their performance, with rank = 1 indicating the best method.

## 4 Main Results

## 4.1 Overall Results

We use Figure 4 to summarize benchmark performance at two levels: at the metric level, each method is ranked based on each metric for fine-grained comparison; at the task level, metric-specific ranks are aggregated into a task-level mean rank for comprehensive performance. The random baseline, PLMs, LLMs, and tool-augmented agents are compared rank-based, demonstrating how method performance changes with task formulation, available experimental evidence, and evaluation objective. Two qualitative observations emerge:

• At the task level, relative strengths vary with task formulation and available evidence. In T1, PLMs lead overall. Agents lead in T2, whereas T3 shows a more interleaved ordering between PLMs and agents. In T4, agents achieve their clearest overall advantage under comprehensive target-specific evidence. This non-monotonic pattern highlights task formulation alongside experimental evidence availability.

• At the metric level, overall standing does not guarantee leadership on every metric. In T1, PLMs hold the strongest overall positions, yet GPT-6 Astra leads NMS@40. In T3, agents rank highest on global and top-weighted ordering, whereas ProSST-2048 ranks highest on peak discovery and top-candidate recovery. Even where agents lead at the task level, as in T2 and T4, the leading agent configuration varies across the four evaluation metrics. Task-level mean ranks therefore summarize overall strength but cannot identify the best method for every evaluation objective.

<table><tr><td>Category</td><td>Model</td><td>NMS@40</td><td>Recall@40</td></tr><tr><td>Statistic</td><td>Random</td><td>0.7749</td><td>0.0142</td></tr><tr><td rowspan="6">PLM</td><td>VenusREM</td><td>0.7932</td><td>0.0427</td></tr><tr><td>S3F-MSA</td><td>0.7816</td><td>0.0364</td></tr><tr><td>ProSST-2048</td><td>0.7847</td><td>0.0407</td></tr><tr><td>S3F</td><td>0.7804</td><td>0.0364</td></tr><tr><td>ESM-2</td><td>0.7854</td><td>0.0370</td></tr><tr><td>ProGen2-base</td><td>0.7709</td><td>0.0274</td></tr><tr><td rowspan="6">LLM</td><td>GPT-6 Astra Claude Opus 5</td><td>0.7966</td><td>0.0254 0.0205</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.7729 0.7620</td><td>0.0211</td></tr><tr><td>Kimi K3</td><td>0.7739</td><td></td></tr><tr><td></td><td></td><td>0.0199</td></tr><tr><td>GLM-5.2</td><td>0.7734</td><td>0.0185</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.7703</td><td>0.0185</td></tr><tr><td rowspan="5">Agent</td><td>Biomni + GPT-6 Astra</td><td>0.7677</td><td>0.0388</td></tr><tr><td>Biomni + Claude Opus 5</td><td>0.7651</td><td>0.0307</td></tr><tr><td>AMixKit + GPT-6 Astra</td><td>0.7866</td><td>0.0386</td></tr><tr><td>AMixKit + Claude Opus 5</td><td>0.7858</td><td>0.0354</td></tr><tr><td> $\mathrm { A M i x K i t } + \mathrm { A M i x } { - } 2 . 1$ </td><td>0.7872</td><td>0.0368</td></tr></table>

Table 6 Performance on T1: Single-mutant generation. The best results are in bold and the second-best results are underlined. The same convention is used in the tables below.

## 4.2 Task-specific Results

## 4.2.1 T1: Single-mutant generation

As shown in Table 6, single-mutant generation remains challenging under a fixed generation budget, with limited recovery of top-ranked mutants and only modest gains in peak discovery over random selection. Learned methods achieve higher Recall@40 than random selection, but the best value increases from only 0.0142 to 0.0427, still recovering fewer than 5% of the ground-truth top-40 mutations. The relative gains in NMS@40 are noticeably smaller, rising at most from 0.7749 to 0.7966, with numerous models showing no competitive edge over random guessing in this setting.

Across model categories, all six PLMs achieve higher Recall@40 than the evaluated standalone LLMs. PLMs also outperform standalone LLMs in NMS@40, with the exception of ProGen2-base and GPT-6 Astra, while AMixKit agents outperform all PLMs except VenusREM. Taken together, PLMs remain competitive in measurement-free mutant generation, while frontier LLMs and AMixKit tool-use supports identification of highly promising individual candidates.

## 4.2.2 T2: Measurement-free multi-mutant ranking

As shown in Table 7, all learned measurement-free ranking methods outperform the random baseline across all metrics, with the best Spearman correlation improving from −0.0215 to 0.3330. However, the best Recall@5 reaches only 0.2351, recovering fewer than one quarter of the ground-truth top-five candidates. While models can extract useful information regarding multi-mutant fitness, they cannot reliably reconstruct the ordering or consistently recover the most promising candidates without target-specific measurements.

Agents achieve the best score on all four T2 metrics. AMixKit + Claude Opus 5 achieves the highest Spearman correlation, NMS@5, and Recall@5, at 0.3330, 0.8120, and 0.2351, while Biomni + GPT-6 Astra achieves the highest NDCG at 0.8767. The strongest non-agent competitors are distributed across the LLM and PLM families: Claude Opus 5 is competitive in Spearman and NMS@5, whereas S3F-MSA leads in NDCG.

<table><tr><td>Category</td><td>Model</td><td>Spearman</td><td>NDCG</td><td>NMS@5</td><td>Recall@5</td></tr><tr><td>Statistic</td><td>Random</td><td>-0.0215</td><td>0.8196</td><td>0.7113</td><td>0.0811</td></tr><tr><td rowspan="6">PLM</td><td>VenusREM</td><td>0.2804</td><td>0.8684</td><td>0.8003</td><td>0.2162</td></tr><tr><td>S3F-MSA</td><td>0.2803</td><td>0.8700</td><td>0.7964</td><td>0.2027</td></tr><tr><td>ProSST-2048</td><td>0.2777</td><td>0.8657</td><td>0.7829</td><td>0.2108</td></tr><tr><td>S3F</td><td>0.2785</td><td>0.8626</td><td>0.7751</td><td>0.1757</td></tr><tr><td>ESM-2</td><td>0.2539</td><td>0.8634</td><td>0.7831</td><td>0.1892</td></tr><tr><td>ProGen2-base</td><td>0.1405</td><td>0.8563</td><td>0.7887</td><td>0.1784</td></tr><tr><td rowspan="6">LLM</td><td>GPT-6 Astra</td><td>0.2726</td><td>0.8635</td><td>0.7967</td><td>0.2216</td></tr><tr><td>Claude Opus 5</td><td>0.2841</td><td>0.8653</td><td>0.8031</td><td>0.1919</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.2239</td><td>0.8595</td><td>0.7723</td><td>0.1973</td></tr><tr><td>Kimi K3</td><td>0.1973</td><td>0.8507</td><td>0.7499</td><td>0.1919</td></tr><tr><td>GLM-5.2</td><td>0.1607</td><td>0.8519</td><td>0.7706</td><td>0.2000</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.2187</td><td>0.8587</td><td>0.7760</td><td>0.2135</td></tr><tr><td rowspan="5">Agent</td><td>Biomni + GPT-6 Astra</td><td>0.3194</td><td>0.8767</td><td>0.8038</td><td>0.2162</td></tr><tr><td>Biomni + Claude Opus 5</td><td>0.3064</td><td>0.8689</td><td>0.8016</td><td>0.2351</td></tr><tr><td>AMixKit + GPT-6 Astra</td><td>0.2851</td><td>0.8702</td><td>0.7781</td><td>0.2027</td></tr><tr><td>AMixKit + Claude Opus 5</td><td>0.3330</td><td>0.8764</td><td>0.8120</td><td>0.2351</td></tr><tr><td>AMixKit + AMix-2.1</td><td>0.2960</td><td>0.8706</td><td>0.7911</td><td>0.2000</td></tr></table>

Table 7 Performance on T2: Measurement-free multi-mutant ranking.

<table><tr><td>Category</td><td>Model</td><td>Spearman</td><td>NDCG</td><td>NMS@5</td><td>Recall@5</td></tr><tr><td>Statistic</td><td>Random</td><td>0.0411</td><td>0.8216</td><td>0.7659</td><td>0.1970</td></tr><tr><td rowspan="6">PLM</td><td>VenusREM</td><td>0.3698</td><td>0.8897</td><td>0.8578</td><td>0.3612</td></tr><tr><td>S3F-MSA</td><td>0.3159</td><td>0.8860</td><td>0.8382</td><td>0.3224</td></tr><tr><td>ProSST-2048</td><td>0.3836</td><td>0.8903</td><td>0.8620</td><td>0.3672</td></tr><tr><td>S3F</td><td>0.3357</td><td>0.8842</td><td>0.8523</td><td>0.2955</td></tr><tr><td>ESM-2</td><td>0.3128</td><td>0.8840</td><td>0.8099</td><td>0.3045</td></tr><tr><td>ProGen2-base</td><td>0.1870</td><td>0.8606</td><td>0.7835</td><td>0.2716</td></tr><tr><td rowspan="6">LLM</td><td>GPT-6 Astra</td><td>0.3516</td><td>0.8886</td><td>0.8258</td><td>0.3134</td></tr><tr><td>Claude Opus 5</td><td>0.3540</td><td>0.8858</td><td>0.8263</td><td>0.3015</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.2865</td><td>0.8814</td><td>0.8019</td><td>0.2806</td></tr><tr><td>Kimi K3</td><td>0.3131</td><td>0.8857</td><td>0.8124</td><td>0.2925</td></tr><tr><td>GLM-5.2</td><td>0.2676</td><td>0.8701</td><td>0.8149</td><td>0.2866</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.2627</td><td>0.8698</td><td>0.8106</td><td>0.2687</td></tr><tr><td rowspan="6">Agent</td><td>Biomni + GPT-6 Astra</td><td>0.3687</td><td>0.8828</td><td>0.7925</td><td>0.3045</td></tr><tr><td>Biomni + Claude Opus 5</td><td>0.4511</td><td>0.8985</td><td>0.8511</td><td>0.3463</td></tr><tr><td>AMixKit + GPT-6 Astra</td><td>0.3833</td><td>0.8907</td><td>0.8323</td><td>0.3433</td></tr><tr><td>AMixKit + Claude Opus 5</td><td>0.3665</td><td>0.8875</td><td>0.8515</td><td>0.3433</td></tr><tr><td>AMixKit + AMix-2.1</td><td>0.3616</td><td>0.8902</td><td>0.8515</td><td>0.3254</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Statistic</td><td>Random</td><td>-0.0204</td><td>0.8124</td><td>0.6371</td><td>0.0552</td></tr><tr><td rowspan="6">PLM</td><td>VenusREM</td><td>0.3821</td><td>0.8809</td><td>0.8212</td><td>0.2552</td></tr><tr><td>S3F-MSA</td><td>0.3662</td><td>0.9005</td><td>0.8686</td><td>0.2897</td></tr><tr><td>ProSST-2048</td><td>0.3779</td><td>0.8725</td><td>0.8078</td><td>0.2483</td></tr><tr><td>S3F</td><td>0.3487</td><td>0.8836</td><td>0.8410</td><td>0.2759</td></tr><tr><td>ESM-2</td><td>0.3153</td><td>0.8828</td><td>0.8129</td><td>0.2276</td></tr><tr><td>ProGen2-base</td><td>0.2568</td><td>0.8736</td><td>0.8357</td><td>0.1793</td></tr><tr><td rowspan="6">LLM</td><td>GPT-6 Astra</td><td>0.5988</td><td>0.9274</td><td>0.8935</td><td>0.4483</td></tr><tr><td>Claude Opus 5</td><td>0.5823</td><td>0.9244</td><td>0.8723</td><td>0.4276</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.5329</td><td>0.9059</td><td>0.8595</td><td>0.4000</td></tr><tr><td>Kimi K3</td><td>0.4794</td><td>0.8956</td><td>0.8337</td><td>0.3586</td></tr><tr><td>GLM-5.2</td><td>0.4959</td><td>0.8984</td><td>0.8338</td><td>0.3862</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.5308</td><td>0.9059</td><td>0.8595</td><td>0.4000</td></tr><tr><td rowspan="5">Agent</td><td>Biomni + GPT-6 Astra</td><td>0.6373</td><td>0.9347</td><td>0.9145</td><td>0.4828</td></tr><tr><td>Biomni + Claude Opus 5</td><td>0.6446</td><td>0.9293</td><td>0.9015</td><td>0.4345</td></tr><tr><td>AMixKit + GPT-6 Astra</td><td>0.6382</td><td>0.9336</td><td>0.9131</td><td>0.4897</td></tr><tr><td>AMixKit + Claude Opus 5</td><td>0.6462</td><td>0.9336</td><td>0.9090</td><td>0.4690</td></tr><tr><td>AMixKit + AMix-2.1</td><td>0.5531</td><td>0.9115</td><td>0.8642</td><td>0.3862</td></tr></table>

Table 8 Performance on T3: Anchor-informed multi-mutant ranking.

Table 9 Performance on T4: Single-mutant-informed multi-mutant ranking.

## 4.2.3 T3: Anchor-informed multi-mutant ranking

As shown in Table 8, models of all categories perform better under the anchor-informed setting, as compared to the measurement-free setting. All evaluated methods outperform random ranking across the four metrics, showing that anchor-conditioned reasoning can improve multi-mutant prioritization, while the efects of additional mutations remain dificult to infer reliably.

The leading method depends on the evaluation objective, revealing complementary strengths of agents and PLMs. Biomni + Claude Opus 5 achieves the highest Spearman correlation at 0.4511 and NDCG at 0.8985; whereas ProSST-2048 attains the highest NMS@5 and Recall@5, with scores of 0.8620 and 0.3672, respectively, and outperforms most other agents across metrics. When an anchor measurement is available, no model family consistently dominates; therefore, the choice of model should be guided by specific experiment objectives.

## 4.2.4 T4: Single-mutant-informed multi-mutant ranking

As shown in Table 9, providing candidate-complete single-mutant fitness context shifts the strongest results toward LLM-based systems, particularly tool-augmented agents, which generally outperform PLMs across all metrics. Unlike T3, which provides the measured score of one anchor variant, this task supplies evidence for every candidate component, directly testing whether models can integrate distributed assay information. Nevertheless, single-mutant scores do not uniquely determine multi-mutant fitness because non-additive interactions may alter substitution efects in combined sequences. Thus, this setting highlights both the value of component-level experimental evidence and the challenge of extrapolating it to combinatorial variants.

Tool augmentation through both Biomni and AMixKit consistently improves performance across metrics and backbones, while the best agent depends on the objective. AMixKit + Claude Opus 5 achieves the highest Spearman correlation at 0.6462; Biomni + GPT-6 Astra leads in NDCG at 0.9347 and NMS@5 at 0.9145; and AMixKit + GPT-6 Astra attains the highest Recall@5 at 0.4897. Tool-augmented agents demonstrate the broadest advantage on T4, while the difering leaders for global ordering, top-weighted ranking, peak discovery, and candidate coverage underscore that these metrics capture complementary ranking aspects.

Residue-Class Conservation Differs across Model Families  
![](images/4f65c3c0bb048f3962fa580ca1494ce740e63ecefaa142b769896cdd1f0376c3.jpg)

![](images/78f183640a97d078be91a8ce11ff3e77e8df1baa83af5c4a8006dbaac6aff547.jpg)

![](images/d91584a601e67f92cc0b98367a5f34547f846c085ab11100f8ded9d1d276f729.jpg)  
Figure 5 Directional six-class amino-acid transitions across model families. The three panels present $6 \times 6$ wild-type-to-mutant class transition matrices for PLMs, LLMs, and agents. The residue classes comprise nonpolar (NP), polar uncharged (P0), polar positively charged (P+), polar negatively charged (P-), aromatic (Ar), and Gly/Pro (G/P). Each cell denotes the percentage of successful T1 hits within the respective model-family matrix (normalized to 100%). Color intensity follows an exponential scale, with frequencies below 0.5% depicted in white. Charge-separated states are preserved, maintaining distinct of-diagonal transitions for acidic-to-basic and basic-to-acidic substitutions.

## 5 Analysis

## 5.1 Comparing PLMs and LLMs

As discussed in Section 4, PLMs and LLMs exhibit distinct performance trajectories across task environments. Mechanistically, this divergence stems from their contrasting modes of information processing: PLMs rely primarily on sequential and homologous signals, whereas LLMs demonstrate superior capacity in leveraging contextual and textual auxiliary conditions. From a biological perspective, the underlying driver of this disparity is task-dependent, displaying a pronounced demarcation between the single-mutant task of T1 and multi-mutant tasks of T2–T4. Nevertheless, both model families encounter shared operational bottlenecks. Specifically, Section 5.1.1 investigates family-specific preferences for amino-acid mutation types; Section 5.1.2 reveals that the utility of single-mutant fitness values in LLMs and agentic systems is largely constrained to additive landscapes; and Section 5.1.3 evaluates how search space complexity governs performance across architectures. We systematically analyze each aspect in the subsequent sections.

## 5.1.1 Divergent Single-mutant Proposal Strategies

As T1: Single-mutant generation lacks target-specific mutation measurements, models prioritize candidates using the supplied assay context and protein-specific evolutionary information when available; agents may also incorporate tool-derived evidence. We evaluated each model by the proportion of top-30% experimental hits within its top-40 predictions, aggregating results by model family. Mean success rates reached 48.0% for PLMs, 42.5% for LLMs, and 50.4% for agents. While overall predictive accuracy remains comparable across architectures, the underlying mutational strategies diverge substantially.

To profile candidate selections, we quantified directed transitions across six physicochemical residue classes among successful variants: nonpolar (NP), polar uncharged (P0), polar positively charged (P+), polar negatively charged (P-), aromatic (Ar), and Gly/Pro (G/P). As shown in Figure 5, the 6 × 6 transition matrices capture all 36 wild-type-to-mutant class shifts, normalized within each model family. Distinguishing P+ from P- preserves critical charge-reversal dynamics, whereas Gly and Pro are grouped together for compactness.

Class-preserving substitutions along the main diagonal accounted for 25.8% of successful PLM proposals, compared to 53.3% for LLMs and 36.8% for agents. Successful LLM proposals were thus predominantly conservative, whereas PLM successes spanned a broader spectrum of class-altering transitions, with LLM-based agents occupying an intermediate regime. This disparity is not an artifact of classification grouping, as intra- $\mathrm { \Delta G / P }$ transitions contribute minimally (0.21%, 0.06%, and 0.18%, respectively). Practically, these patterns indicate that PLMs explore a wider substitution landscape, whereas LLM-based architectures preferentially propose mutations that conserve fundamental physicochemical properties.

Conditional Use of Single-Mutant Evidence in T4  
![](images/de829b0b7d07e27e7215008dc1cc9c26d6b5d3383fe0e1aeae4a5fd955d64515.jpg)

(b)  
![](images/8d59b06905a1a348469a9d0a737d51585b26275017af74ba24586adfcae6f4b7.jpg)

(c)  
![](images/9fe8e7eec19e74919120961f99c6043e1f64da923e266794b168e26905d9ae63.jpg)  
Figure 6 Conditional utilization of single-mutant evidence in multi-mutant ranking. Panels (a)–(c) show PLMs, LLMs and agents, respectively, with each point corresponding to one T4 assay. The horizontal axis is the Spearman correlation between the sum of measured component single-mutant fitness and the measured multi-mutant fitness. This is an information-transfer diagnostic, not a direct estimate of a mechanistic epistasis coeficient. Circles mark assays with $\rho _ { \mathrm { s u m , c o m b i n a t i o n } } \geq 0$ and squares mark assays below this operational threshold. The line marks a linear fit, while $\rho$ is the ranking correlation between the diagnostic and model performance.

Both behaviors reflect plausible biological heuristics conditioned on available inputs. Sequence-based PLMs efectively identify substitutions compatible with evolutionary alignments and structural stability, which frequently transcend rigid residue classes. The conservative substitutions observed for standalone LLMs are consistent with general biochemical heuristics, although these results do not identify the source of their internal priors. AMixKit agents can additionally call PLMs through tools. However, neither prior explicitly captures assay-specific determinants (e.g., active-site geometry, interface kinetics, or conformational dynamics) that ultimately dictate measured phenotypes. Consequently, these trends reflect contrasting search strategies rather than an intrinsic hierarchy in model quality.

## 5.1.2 Utilization of Multi-mutant Component Additivity

For multi-mutant ranking tasks, the experimental context supplied to the model increases progressively. A key observation is that while PLMs, LLMs and agents exhibit comparable performance when single-mutant evidence is limited, LLMs and agents gain a distinct advantage as component-level measurements become available. Across T2: Measurement-free multi-mutant ranking and T3: Anchor-informed multi-mutant ranking, the three model families achieve mean assay-level Spearman correlations of 0.271 vs. 0.331 (PLMs), 0.223 vs. 0.311 (LLMs), and 0.317 vs. 0.383 (agents). However, this performance gap widens dramatically in T4: Single-mutant-informed multi-mutant ranking —where measured single-mutant efects are explicitly provided—with LLMs and agents achieving 0.535 and 0.635, substantially outperforming PLMs at 0.349.

This performance trajectory suggests that LLM-based architectures leverage individual component DMS\_score values through an additive heuristic. To test this hypothesis, we computed the sum of measured single-mutant DMS\_score values for each multi-mutant candidate in T4 and evaluated its correlation with actual combination fitness. As depicted in Figure 6, a high correlation indicates that combination rankings can be reliably approximated via additive components, whereas a low correlation denotes poor transferability.

![](images/09aa0f373201fdc961357fcb902a34b582f896cef80e8689e95280f8d075f72a.jpg)  
Figure 7 Common limitations across model families. (a) Sequence search space. Verified T1 top-40 hit counts, stratified by the number of legal variants per assay into three bins $( \leq 2 , 0 0 0 ; 2 , 0 0 1 - 8 , 0 0 0 ; > 8 , 0 0 0 )$ . Each violin shows the distribution of hit counts across assays for one family (PLMs, LLMs or agents), with a horizontal line marking the median; the three families are ofset within each bin and share a common color coding. (b) Mutation depth. Mean assay-level Spearman ranking accuracy for double mutants (solid) versus candidates with more than two substitutions (hatched), shown per family with the mean value labeled above each bar. PLMs, LLMs and agents use the same family colors in both panels.

The predictive accuracy of LLMs $( \rho = 0 . 7 3 3 , P = 6 . 1 2 \times 1 0 ^ { - 6 } )$ and agents $( \rho = 0 . 6 4 9 , P = 1 . 3 9 \times 1 0 ^ { - 4 } )$ strongly correlates with the applicability of this additive heuristic, whereas PLMs display a much weaker coupling $\left( \rho = 0 . 3 7 8 , P = 0 . 0 4 3 \right)$ . On the non-additive regime of $\rho _ { \mathrm { s u m , c o m b i n a t i o n } } < 0 _ { \cdot }$ , comprising 7 of 29 assays, LLM performance drops sharply to 0.015, compared to 0.242 for PLMs and 0.314 for agents. Conversely, on the high-transfer split, mean performance reaches 0.701 for LLMs, 0.737 for agents, but only 0.384 for PLMs.

These findings reveal a conditional dependency on component evidence: LLMs and agents excel when combination fitness is well-approximated by additive single-mutant values, but LLMs collapse when this assumption fails. Although agentic workflows provide partial robustness in non-additive regimes, neither LLMs nor agents show evidence of modeling complex non-additive residue interactions. PLMs are less coupled to this diagnostic and therefore show smaller performance degradation, but not a general advantage on the low-transfer assays. Consequently, the dominance of LLM-based systems in T4 reflects efective utilization of transferable component evidence rather than a comprehensive understanding of residue coupling.

## 5.1.3 Common Limitations across Single-mutant and Multi-mutant Tasks

Whether through distinct strategies for selecting amino-acid mutations, the application of single-mutation fitness values, or the response mechanisms to additive efects, these diferences ultimately reflect variations in the models’ capabilities across diverse tasks. Yet all models face the same two dilemmas: sensitivity to sequence length, and a decline in prediction accuracy for variants of higher-orders.

High-dimensional Search Spaces Navigating the search space of a protein sequence presents a severe challenge. In T1: Single-mutant generation, a single assay features a median of 2,964 valid single-point substitutions (ranging from 1,093 to 22,536). However, each model proposes only its top 40 candidates, thereby sampling a mere 1.35% of the median search space. As depicted in Figure 7(a), the average number of top-40 hits in the single-mutant scenario of T1 is only 1.21, with even the highest-performing case recovering only 13. Furthermore, this recovery rate degrades rapidly with increasing sequence length and search space size. This weak performance only marginally outperforms random sampling, and indicates that current models fall far short of truly mastering sequence space or providing reliable candidate fitness evaluations. Extending this task to multi-mutant proposal across full-length sequences will exacerbate this challenge, as the combinatorial search space expands exponentially.

![](images/19a3f199210831065b8b17e4a8ce2cefd13b97629f8fa4098d495dfd2bbe92c5.jpg)  
Figure 8 Cumulative performance comparison of round-wise single mutant generation. Single-round refers to the original T1: Single-mutant generation setting, where all 40 mutations are proposed in a single pass, and is indicated with dotted horizontal lines. Multi-round refers to the results after a complete four-round feedback-guided generation protocol, with 10 mutations proposed each round, where proposed mutants with an assay-recorded DMS\_- score are used as scored context for the next model pass, and is depicted in solid lines.

High-order Combinations The mutation depth within a variant (i.e., the number of substituted sites per combination) significantly impacts model ranking performance. Across the multi-mutant ranking tasks of T2–T4, ranking accuracy declines monotonically as mutation count increases $( \rho = - 0 . 4 3 0 , P = 4 . 7 5 \times 1 0 ^ { - 9 } )$ , as shown in Figure 7(b). Specifically, in T4: Single-mutant-informed multi-mutant ranking, the mean Spearman correlation for LLMs drops from 0.669 on 2 mutations to 0.514 for 3–4 mutations, and down to 0.222 for variants with over 5 mutated sites, demonstrating superior eficacy on lower-order combinations. PLMs exhibit a parallel reduction from 0.353 to 0.236 and 0.051, while LLM-based agents also show decay from 0.674 to 0.536 and 0.386. Notably, even when augmented with single-mutant fitness values—which substantially improve double-mutant ranking capabilities—models fail to generalize to higher-order variants involving complex epistatic interactions. This pervasive performance drop highlights the fundamental dificulty of modeling high-order combinations.

Together, these findings highlight how sequence dimension and mutation depth compound search pressure on computational models. Given how heavily performance degrades even within the controlled settings of T1–T4, unconstrained multi-site generation—a common scenario in practical protein engineering—presents an exponentially higher barrier. Addressing real-world bio-design problems requires robust predictive capability across high-dimensional search spaces and higher-order combinations. This underscores the core motivation behind our task design: models must maintain reliable analytical and judgment capabilities when subjected to simultaneous scaling in sequence length and mutation depth. PFArena reveals the shortcomings of existing models along these crucial dimensions.

## 5.2 Dive into LLMs

## 5.2.1 Feedback-guided In-context Adaptation

To emulate a sequential wet-laboratory campaign, in which mutation fitness is measured in batches and later candidates are selected using earlier outcomes, we extended the single-round T1: Single-mutant generation into a four-round adaptive search. This experiment tests whether assay-specific fitness measurements supplied in context help models refine subsequent generations and identify higher-fitness mutants within a fixed screening budget. In each round, the model proposed 10 single mutants unselected in earlier rounds. For the next round, the ground-truth DMS\_score fitness of all previously proposed mutants was added to the prompt as in-context feedback, if the fitness value of the mutant is available. Both settings therefore used the same total budget of 40 proposed mutations, but only the multi-round setting allowed later generations to condition on intermediate fitness measurements.

Overall-confidence distribution and alignment  
(a)  
![](images/0fd3a3d1eea87bf5cf591c18f9e0f6d91d379518268170d57f7f78f68ccf5985.jpg)

![](images/029a4e6d77c6b3131456e406b147c1cb7493045072dbf4339f440c31cc994213.jpg)  
Figure 9 Confidence distribution and quality alignment. (a) GPT-6 Astra’s self-reported confidence across T1–T4. (b) Within-setting Spearman correlations between confidence and ranking-quality metrics.

As shown in Figure 8, the multi-round protocol shows monotonic improvements in both NMS@40 and Recall@40 across rounds for every model. Relative to single-round generation, the final Round-4 endpoint NMS@40 increased from 0.7966 to 0.8367 for GPT-6 Astra, highest across all models, and increased by 0.0168 for Claude Opus 5, 0.0244 for Gemini 3.1 Pro, and 0.0076 for DeepSeek-V4-Pro. Recall@40 increased by 0.0211, 0.0175, 0.0128, and 0.0087 respectively, with GPT-6 Astra achieving the highest recovery of 0.0465. Averaged across the four models, multi-round feedback-guided generation improved NMS@40 by 0.0222 and Recall@40 by 0.0150. All Round-4 results exceeded the random baseline on both metrics, whereas three of the four single-round models remained below random sampling in NMS@40.

Across rounds, GPT-6 Astra exceeded its single-round NMS@40 baseline after two rounds, Claude Opus 5 and Gemini 3.1 Pro after three rounds, and DeepSeek-V4-Pro after four rounds. The first three models exceeded their single-round baselines in Recall@40 after three rounds, whereas DeepSeek-V4-Pro after the fourth. As the cumulative candidate budget increases across rounds, the earlier rounds alone do not establish the value of feedback; the decisive comparison is the Round-4 endpoint against single-round generation. The consistent performance gains demonstrate the efectiveness of feedback-guided in-context adaptation, particularly for GPT-6 Astra, which substantially surpassed its already strong single-round baseline, highlighting its capability in knowledge-based work tasks. Notably, LLMs frequently revisited previously high-scoring sites with alternative amino-acid mutations, indicating that further exploration of promising positions in subsequent rounds may help identify mutants of higher fitness.

## 5.2.2 Uncertainty Estimation

We evaluated self-reported confidence from GPT-6 Astra to examine whether LLMs can estimate their own predictive uncertainty. For queries in T1–T4, the model was prompted to assign an overall confidence score from 0 to 100 reflecting its confidence in identifying and prioritizing high-fitness mutations. Following five API policy refusals and two safety exclusions in T1, confidence was available for 116, 74, 67, and 29 queries in T1–T4, respectively. We treated this score as an inverse measure of relative query-level uncertainty and assessed its association with predictive performance using Spearman correlation. The confidence elicitation prompt used in our experiments is provided in Appendix A.3.

As shown in Figure 9(a), confidence varied substantially across settings, with mean scores of 22.5, 35.6,

![](images/c3137849d209a7a5b353756ea53bc95142a696b1ffeb9bdb5c31c9dfb999ada1.jpg)

Figure 10 Test-time scaling performance across sample counts. The panels show NMS@40 (left) and Recall@40 (right) as the number of sampled mutation lists increases, with the dotted line representing the sample average.
<table><tr><td>Setting</td><td>Method</td><td>NMS@40↑</td><td>Recall@40↑</td><td>Invalid per Assay </td></tr><tr><td rowspan="2">Vanilla</td><td>Sample Avg</td><td>0.7696</td><td>0.0186</td><td>2.2805</td></tr><tr><td>RRF</td><td>0.7847</td><td>0.0199</td><td>1.4472</td></tr><tr><td rowspan="2">Clean</td><td>Sample Avg</td><td>0.7771</td><td>0.0189</td><td>1.0531</td></tr><tr><td>RRF</td><td>0.7845</td><td>0.0197</td><td>1.1057</td></tr><tr><td rowspan="2">In-Assay</td><td>Sample Avg</td><td>0.7745</td><td>0.0172</td><td>0.0000</td></tr><tr><td>RRF</td><td>0.7842</td><td>0.0191</td><td>0.0000</td></tr></table>

Table 10 Validity-controlled RRF analysis at n = 16. Clean requires 40 unique and sequence-valid mutations. In-Assay further requires all mutations to appear in the assay’s DMS table.

38.3, and 63.0 for T1–T4, respectively, and no response reaching 90. T1: Single-mutant generation elicited relatively conservative judgments, whereas T4: Single-mutant-informed multi-mutant ranking produced a relatively high-confidence distribution. These absolute values should not be interpreted as success probabilities or directly compared across settings, which difer in data and evaluation objectives. The more meaningful tes is whether confidence distinguishes higher- from lower-quality predictions within each setting.

Self-reported confidence provides a partial, task- and context-dependent estimate of ranking reliability. Figure 9(b) shows positive evidence in T3: Anchor-informed multi-mutant ranking, where confidence aligned with all four quality dimensions $( \rho \approx 0 . 3 2 – 0 . 4 9 )$ after multiple-testing correction. In T2: Measurementfree multi-mutant ranking, confidence also aligned with all four dimensions $( \rho \approx 0 . 2 8 – 0 . 5 7 )$ , with stronger associations for global ordering than top-five selection. T1 showed weak-to-moderate associations with its top-40 metrics $( \rho = 0 . 3 5$ for NMS@40 and 0.23 for Recall@40), which remained significant after filtering dirty outputs. T4 showed the largest correlations for full-list Spearman and NDCG quality $( \rho = 0 . 6 9$ and 0.67), whereas neither top-five metric passed multiple-testing correction. Assay-level examples reinforce this limitation: one protein-stability query in T4 received high confidence (78) despite limited top-five recovery (Recall@5=0.2), whereas a fluorescence query received lower confidence (45) but substantially better recovery (Recall@5=0.8). These cases show that confidence does not uniformly track top-candidate recovery, even when it aligns with global ranking quality. It can therefore support coarse query triage within a fixed task, but should not be treated as a generally reliable or cross-assay uncertainty estimate.

## 5.2.3 Test-time Scaling

To test whether additional inference-time computation improves single-mutation generation without experimental feedback, we independently sampled 16 responses from DeepSeek-V4-Pro using exactly the same input for each T1: Single-mutant generation query. We extracted a ranked top-40 mutation list from each response and applied aggregation to nested prefixes with n ∈ {1, 2, 4, 8, 16}. Aggregation used no ground-truth DMS\_score, candidate labels, or evaluator metrics, and the mean performance of the corresponding raw samples at each prefix size served as the Sample Avg baseline.

We compared five TTS aggregation strategies. Best-of-N selected the complete sampled ranking most consistent with the other samples. Approval Voting prioritized mutations that appeared repeatedly across samples, whereas Borda Fusion and Reciprocal Rank Fusion (RRF) additionally incorporated within-sample rank using linear and reciprocal weighting, respectively. Hierarchical RRF further pooled evidence across diferent mutations at the same residue position. In the main scaling experiments, all methods operated directly on the extracted candidate strings without explicit sequence-validity or DMS-membership filtering. Complete method details and hyperparameters used in our experiments are provided in Section B.2.

As shown in Figure 10, increasing sample count generally improved performance. Approval scaled most consistently, reaching 0.7876 NMS@40 and 0.0222 Recall@40 at n = 16, versus 0.7696 and 0.0186 for Sample Avg. Borda and RRF also improved with more samples, while Best-of-N was less stable and Hierarchical RRF mainly benefited small n. All five methods outperformed Sample Avg at n = 16.

One source of improvement was invalid-candidate suppression. LLM-proposed mutations may be locally invalid due to sequence inconsistency or formatting errors, or assay invalid when they do not appear in the assay’s DMS table. In the main setting, TTS aggregation substantially suppressed invalid outputs. As shown in Table 10, RRF reduced invalid predictions from 2.2805 to 1.4472, without explicit validity filtering. To determine whether this fully explains the gain, we further evaluated Clean and In-Assay settings. Clean retained only samples with 40 unique and locally valid candidates, while In-Assay further required all candidates to appear in the assay’s DMS table. Under Clean, RRF improved NMS from 0.7771 to 0.7845 and recall from 0.0189 to 0.0197, despite slightly more assay-absent predictions (1.0531 to 1.1057). Under In-Assay, where both methods had zero invalid predictions, RRF still improved NMS from 0.7745 to 0.7842 and recall from 0.0172 to 0.0191. Thus, although invalid-candidate suppression significantly contributes to the TTS gain, it does not fully account for the improvement; even when validity is controlled, aggregation helps identify and prioritize higher-quality valid candidates across repeated samples from the same query.

Despite the gains from TTS at larger sample counts, a substantial oracle gap remains. At n = 16, best-sample oracles reached 0.8474 NMS@40 and 0.0461 Recall@40, compared with 0.7876 and 0.0222 for the best practical TTS method. This gap suggests that repeated sampling can already produce substantially better predictions, but reliably identifying them without experimental feedback remains challenging.

## 6 Related Work

Protein Fitness Benchmarks Protein fitness benchmarks have established standardized evaluation of computational models for predicting experimentally measured mutation efects. FLIP [12] evaluates fitnesslandscape inference under low-resource and extrapolative generalization settings, while FLIP2 [30] extends this framework to unseen mutation identities, positions, mutation counts, wild-type proteins, and high-fitness regions. ProteinGym [11] further enables large-scale evaluation of zero-shot and supervised mutation-efect predictors across diverse DMS assays and clinical labels. Together, these benchmarks characterize how accurately PLMs score and rank predefined mutants across heterogeneous fitness landscapes.

Recent benchmarks have extended protein fitness evaluation to general-purpose LLMs and LLM-based agents. ProteinGym-LLM [31] evaluates whether LLMs can rank a fixed set of protein mutants from the wild-type sequence and assay description, providing a direct assessment of LLM-based mutation prioritization. BioDesignBench [32] evaluates tool-using LLM agents across expert-curated protein-design workflows, with an emphasis on candidate generation, tool use, and iterative evaluation. These studies extend protein engineering benchmarks from specialized fitness predictors to general-purpose reasoning and agentic design systems.

Despite these advances, existing benchmarks typically focus on either fixed-list mutant ranking or end-toend protein-design workflows, leaving systematic comparison across diferent mutation-discovery settings underexplored. PFArena addresses this gap by evaluating PLMs, general-purpose LLMs, and LLM-based agents through one single-mutant generation task (T1) and three multi-mutant ranking tasks under measurement-free (T2), anchor-informed (T3), and single-mutant-informed (T4) conditions, applying consistent task interfaces and evaluation criteria across all model families.

Protein Fitness Methods Target-specific supervised methods provide the most direct approach to protein fitness prediction by learning sequence–fitness relationships from experimentally measured mutants. The Low-N framework [33] uses pretrained UniRep representations and as few as 24 assayed mutants to guide protein engineering, while Hsu et al. [34] combine site-specific amino-acid features with evolutionary density estimates to predict fitness from limited measurements. Because these methods are trained against the phenotype of interest, they can directly adapt to the assay-specific objective, but their performance depends on the number, diversity, and sequence-space coverage of available labels.

Zero-shot PLMs avoid target-specific training by deriving mutation scores from general biological priors learned from sequence, evolutionary, and structural data. Evolutionary models such as EVmutation [35], DeepSequence [36], and EVE [6] infer family-specific constraints from homologous sequences. Protein language models, including ESM-1v [37], ESM-2 [1], ProGen2 [2], Tranception [38], and DPLM-Evo [39] instead learn sequence compatibility across large protein corpora. Structure-aware models such as ProSST [3], S3F [4], and VenusREM [5] further incorporate geometric and evolutionary information. These models can score mutations without assay-specific labels, but their scores primarily reflect general protein plausibility rather than the phenotype and experimental evidence associated with a particular engineering objective.

General-purpose LLMs and scientific agents have also shown substantial potential for protein mutation discovery. Successive Claude releases have demonstrated continued progress in protein mutation-efect prediction, with Claude Opus 5 further improving upon previous generations on ProteinGym Hard [8]. LLMs have also been incorporated into budget-constrained protein optimization procedures [40], suggesting that their biological knowledge can support mutation selection beyond direct fitness scoring. Scientific agents further extend these capabilities through retrieval and tool use: ProtAgents [41] coordinates specialized agents for protein analysis and design, while Biomni [9] integrates biomedical databases, scientific software, and code execution. These developments motivate systematic evaluation of whether LLMs and agents can convert assay context, experimental evidence, and specialized protein-model outputs into efective mutation priorities.

Given the rapidly expanding set of PLMs, LLMs, and scientific agents, our evaluation focuses on representative methods spanning sequence-based, structure-based, evolution-based, and tool-based paradigms.

## 7 Conclusion

PFArena provides a unified evaluation of PLMs, LLMs, and LLM-based agents across realistic protein modification decisions. Taken together, our results show that the preferred model family depends on the task interface. Protein-specific representations are particularly efective for open-ended single-mutant search, whereas general-purpose reasoning and tool use become more useful for ranking mutation combinations, especially when measured single-mutant context is available. This complementarity does not eliminate shared bottlenecks: high-fitness candidate recovery remains sparse under limited prediction budgets, and ranking becomes substantially less reliable beyond double mutants. Our additional analyses show that feedback-guided adaptation and test-time scaling can improve LLM-based inference. We release our code and benchmark suite to support reproducible progress on these open problems.

Limitations Despite its broad coverage, PFArena has several limitations.

• Potential label leakage. All assays in PFArena are derived from publicly available datasets and publications. Their sequences, mutation labels, or experimental results may therefore have appeared in the pretraining corpora of the evaluated LLMs, making it impossible to fully exclude memorization or other forms of data contamination that could influence the evaluation results.

• Stochasticity and single-run evaluation. Because evaluating frontier LLMs and tool-augmented agents is computationally and financially expensive, each LLM or agent configuration was run only once for main evaluation. The reported results may therefore depend on sampling randomness.

• Missing outputs, refusals, and comparison fairness. Some LLMs, particularly proprietary systems such as GPT and Claude, refuse to answer a subset of assays because of their safety policies. To retain a fixed evaluation set and avoid selectively excluding dificult cases, we replace such outputs with a deterministic random baseline result, as detailed in Section C.1. This protocol measures the end-to-end usability of each deployed system, but it also conflates underlying protein-reasoning ability with provider-specific refusal policies and output reliability.

• Retrospective rather than prospective evaluation. PFArena is constructed from previously measured assays. Improvements in benchmark metrics consequently do not establish higher prospective wet-lab hit rates or guarantee that the selected mutations will succeed in a new experimental campaign. Prospective validation will be required to determine whether the observed performance gains translate into practical reductions in experimental cost and design cycles.

• Dataset coverage and heterogeneity. The assays vary in experimental protocol, measurement noise, sequence coverage, candidate-library construction, and phenotype definition. Harmonization and quality control reduce but cannot eliminate these diferences. Moreover, publicly available datasets may overrepresent well-studied proteins, and assay types that are convenient to measure, while underrepresenting negative results, rare protein families, and complex cellular or organism-level phenotypes. The conclusions may therefore not generalize to all protein-engineering settings.

## Contributions

Project Lead

Yawen Ouyang<sup>1,2</sup>

Co-first Authors

Yawen Ouyang<sup>1,2</sup>, Xinbo Zhang<sup>1,2</sup>, Ziyuan Ma<sup>1,4</sup>

Task Design and Data Collection

Ziyuan Ma<sup>1,4</sup>, Xinbo Zhang<sup>1,2</sup>, Yawen Ouyang<sup>1,2</sup>

Main Results

Yawen Ouyang<sup>1,2</sup>, Yixin Wu<sup>1,2</sup>, Wenbin Liao<sup>1,5</sup>, Xinbo Zhang<sup>1,2</sup>

Analysis

Ziyuan Ma<sup>1,4</sup>, Yixin Wu<sup>1,2</sup>, Wenjie Li<sup>1</sup>, Feiran Zhang<sup>1,6</sup>, Yawen Ouyang<sup>1,2</sup>

Other Contributors

Lihao Wang<sup>1,2</sup>, Hao Wang<sup>2,3</sup>, Xiaoqing Zheng<sup>6</sup>, Xuefeng Yan<sup>5</sup>, Lei Bai<sup>1</sup>, Ya-Qin Zhang<sup>3</sup>, Shuyi Zhang<sup>4</sup>, Wei-Ying Ma<sup>3,7</sup>, Dahua Lin<sup>1</sup>, Bowen Zhou<sup>1</sup>

Correspondence

Hao Zhou<sup>1,2,3</sup>

## Affiliation

<sup>1</sup>Shanghai Artificial Intelligence Laboratory

<sup>2</sup>Generative Symbolic Intelligence Lab (GenSI), Tsinghua University

<sup>3</sup>Institute for AI Industry Research (AIR), Tsinghua University

<sup>4</sup>School of Pharmaceutical Sciences, Tsinghua University

<sup>5</sup>School of Information Science and Engineering, East China University of Science and Technology

<sup>6</sup>College of Computer Science and Artificial Intelligence, Fudan University

<sup>7</sup>City University of Hong Kong

## Acknowledgments

This work is supported by Shanghai Artificial Intelligence Laboratory and NSFC (Grant No. 62406170).

## References

[1] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, and Alexander Rives. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, March 2023. doi: 10.1126/science.ade2574. URL https://doi.org/10.1126/ science.ade2574.

[2] Erik Nijkamp, Jefrey A. Rufolo, Eli N. Weinstein, Nikhil Naik, and Ali Madani. ProGen2: Exploring the boundaries of protein language models. Cell Systems, 14(11):968–978.e3, November 2023. doi: 10.1016/j.cels. 2023.10.002. URL https://doi.org/10.1016/j.cels.2023.10.002.

[3] Mingchen Li, Yang Tan, Xinzhu Ma, Bozitao Zhong, Huiqun Yu, Ziyi Zhou, Wanli Ouyang, Bingxin Zhou, Pan Tan, and Liang Hong. ProSST: Protein language modeling with quantized structure and disentangled attention. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 35700–35726. Curran Associates, Inc., 2024. doi: 10.52202/079017-1126. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf.

[4] Zuobai Zhang, Pascal Notin, Yining Huang, Aurélie Lozano, Vijil Chenthamarakshan, Debora Marks, Payel Das, and Jian Tang. Multi-scale representation learning for protein fitness prediction. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 101456–101473. Curran Associates, Inc., 2024. doi: 10.52202/079017-3217. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ b7d795e655c1463d7299688d489e8ef4-Paper-Conference.pdf.

[5] Yang Tan, Ruilin Wang, Banghao Wu, Liang Hong, and Bingxin Zhou. From high-throughput evaluation to wet-lab studies: advancing mutation efect prediction with a retrieval-enhanced model. Bioinformatics, 41 (Supplement\_1):i401–i409, July 2025. doi: 10.1093/bioinformatics/btaf189. URL https://doi.org/10.1093/ bioinformatics/btaf189.

[6] Jonathan Frazer, Pascal Notin, Mafalda Dias, Aidan Gomez, Joseph K. Min, Kelly Brock, Yarin Gal, and Debora S. Marks. Disease variant prediction with deep generative models of evolutionary data. Nature, 599(7883):91–95, October 2021. doi: 10.1038/s41586-021-04043-8. URL https://doi.org/10.1038/s41586-021-04043-8.

[7] Anthropic. Claude opus 4.8 system card. https://www-cdn.anthropic.com/ 0b4915911bb0d19eca5b5ee635c80fef830a37ea.pdf, May 2026.

[8] Anthropic. Introducing claude opus 5. https://www.anthropic.com/news/claude-opus-5, 2026.

[9] Kexin Huang, Serena Zhang, Hanchen Wang, Yuanhao Qu, Yingzhou Lu, Yusuf Roohani, Ryan Li, Lin Qiu, Junze Zhang, Yin Di, et al. Biomni: A general-purpose biomedical AI agent. bioRxiv, 2025. doi: 10.1101/2025.05.30. 656746. URL https://doi.org/10.1101/2025.05.30.656746. Preprint.

[10] Mark A. Robien, Kiet T. Nguyen, Abhinav Kumar, Irwin Hirsh, Stewart Turley, Dehua Pei, and Wim G. J. Hol. An improved crystal form of Plasmodium falciparum peptide deformylase. Protein Science, 13(4):1155–1163, 2004. doi: 10.1110/ps.03456404.

[11] Pascal Notin, Aaron Kollasch, Daniel Ritter, Lood van Niekerk, Stefanie Paul, Han Spinner, Nathan Rollins, Ada Shaw, Rose Orenbuch, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Yarin Gal, and Debora S. Marks. ProteinGym: Large-scale benchmarks for protein fitness prediction and design. In Advances in Neural Information Processing Systems, 2023. URL https://mlanthology.org/neurips/2023/ notin2023neurips-proteingym/.

[12] Christian Dallago, Jody Mou, Kadina E. Johnston, Bruce J. Wittmann, Nicholas Bhattacharya, Samuel Goldman, Ali Madani, and Kevin K. Yang. FLIP: Benchmark tasks in fitness landscape inference for proteins. In Neural Information Processing Systems Datasets and Benchmarks Track, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ 2b44928ae11fb9384c4cf38708677c48-Abstract-round2.html.

[13] Daniel Esposito, Jochen Weile, Jay Shendure, Lea M. Starita, Anthony T. Papenfuss, Frederick P. Roth, Douglas M. Fowler, and Alan F. Rubin. MaveDB: an open-source platform to distribute and interpret data from

multiplexed assays of variant efect. Genome Biology, 20(1):223, 2019. doi: 10.1186/s13059-019-1845-6. URL https://doi.org/10.1186/s13059-019-1845-6.

[14] Kotaro Tsuboyama, Justas Dauparas, Jonathan Chen, Elodie Laine, Yasser Mohseni Behbahani, Jonathan J. Weinstein, Niall M. Mangan, Sergey Ovchinnikov, and Gabriel J. Rocklin. Mega-scale experimental analysis of protein folding stability in biology and design. Nature, 620:434–444, 2023. doi: 10.1038/s41586-023-06328-6.

[15] Michael Chungyoun, Jef Rufolo, and Jefrey J. Gray. Flab: Benchmarking tasks in fitness landscape inference for antibodies. bioRxiv, 2024. doi: 10.1101/2024.01.13.575504.

[16] Antoni Beltran, Xiang’er Jiang, Yue Shen, and Ben Lehner. Site-saturation mutagenesis of 500 human protein domains. Nature, 637(8047):885–894, 2025. doi: 10.1038/s41586-024-08370-4.

[17] Yongcan Chen, Lihao Fu, Xuchao Lu, Wenzhuo Li, Yuan Gao, Yibo Wang, Zhicheng Ruan, and Tong Si. Combingym: a benchmark platform for machine learning-assisted design of combinatorial protein variants. bioRxiv, 2026. doi: 10.64898/2026.03.24.714074.

[18] Hirokazu Kimura, Kamel Lahouel, Cristian Tomasetti, and Nicholas Jason Roberts. Functional characterization of all cdkn2a missense variants and comparison to in silico models of pathogenicity. eLife, 13:RP95347, 2024. doi: 10.7554/eLife.95347.

[19] Wen-An Wang, Evandro Ferrada, Christoph Klimek, Tanja Osthushenrich, Aidan MacNamara, Tabea Wiedmer, and Giulio Superti-Furga. Large-scale experimental assessment of variant efects on the structure and function of the citrate transporter slc13a5. Science Advances, 11(26):eadx3011, 2025. doi: 10.1126/sciadv.adx3011.

[20] Kadina E. Johnston, Patrick J. Almhjell, Ella J. Watkins-Dulaney, Grace Liu, Nicholas J. Porter, Jason Yang, and Frances H. Arnold. A combinatorially complete epistatic fitness landscape in an enzyme active site. Proceedings of the National Academy of Sciences, 121(32):e2400439121, 2024. doi: 10.1073/pnas.2400439121.

[21] Martin Steinegger and Johannes Söding. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology, 35(11):1026–1028, October 2017. doi: 10.1038/nbt.3988. URL https://doi.org/10.1038/nbt.3988.

[22] Baris E. Suzek, Yuqi Wang, Hongzhan Huang, Peter B. McGarvey, Cathy H. Wu, and UniProt Consortium. UniRef clusters: a comprehensive and scalable alternative for improving sequence similarity searches. Bioinformatics, 31 (6):926–932, March 2015. doi: 10.1093/bioinformatics/btu739. URL https://doi.org/10.1093/bioinformatics/ btu739.

[23] Openai. Gpt-6 astra: A new generation of intelligence. https://openai.com/index/gpt-6-astra/, 2026.

[24] Google DeepMind. Gemini 3.1 pro. https://deepmind.google/models/gemini/pro, 2026.

[25] Kimi Team. Kimi k3: Open frontier intelligence, 2026. URL https://arxiv.org/abs/2607.24653.

[26] Z.ai. Glm-5.2: Built for long-horizon tasks. https://z.ai/blog/glm-5.2, 2026.

[27] DeepSeek AI. Deepseek-v4: Towards highly eficient million-token context intelligence. https://huggingface. co/deepseek-ai/DeepSeek-V4-Pro, 2026.

[28] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, Sebastian W. Bodenstein, David A. Evans, Chia-Chun Hung, Michael O’Neill, David Reiman, Kathryn Tunyasuvunakool, Zachary Wu, Akvil˙e Žemgulyt˙e, Eirini Arvaniti, Charles Beattie, Ottavia Bertolli, Alex Bridgland, Alexey Cherepanov, Miles Congreve, Alexander I. Cowen-Rivers, Andrew Cowie, Michael Figurnov, Fabian B. Fuchs, Hannah Gladman, Rishub Jain, Yousuf A. Khan, Caroline M. R. Low, Kuba Perlin, Anna Potapenko, Pascal Savy, Sukhdeep Singh, Adrian Stecula, Ashok Thillaisundaram, Catherine Tong, Sergei Yakneen, Ellen D. Zhong, Michal Zielinski, Augustin Žídek, Victor Bapst, Pushmeet Kohli, Max Jaderberg, Demis Hassabis, and John M. Jumper. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630(8016):493–500, May 2024. doi: 10.1038/s41586-024-07487-w. URL https://doi.org/10.1038/s41586-024-07487-w.

[29] Keyue Qiu, Yixin Wu, Lihao Wang, Yawen Ouyang, Jixiang Yu, Zihan Zhou, Changze Lv, Dongyu Xue, Yuxuan Song, Xinbo Zhang, et al. Amix-2: Establishing protein as a native modality in large language models. arXiv preprint arXiv:2605.30963, 2026.

[30] Kieran Didi, Sarah Alamdari, Alex X. Lu, Bruce Wittmann, Kadina E. Johnston, Ava A. Amini, Ali Madani, Maya Czeneszew, Christian Dallago, and Kevin K. Yang. FLIP2: Expanding protein fitness landscape benchmarks for real-world machine learning applications. In Forty-third International Conference on Machine Learning, 2026. URL https://flip.protein.properties/.

[31] Rohit Krishan Arora, Leo Tianlai Chen, Melissa Du, Debora Marks, and George Church. PG-LLM: Benchmarking general-purpose language models for protein variant ranking. bioRxiv, 2026. doi: 10.64898/2026.07.27.741045. URL https://www.biorxiv.org/content/10.64898/2026.07.27.741045v1.

[32] Jeonghyeon Kim and Philip Romero. Evaluating LLM-driven protein design: Agents lack iterative evaluation depth. GitHub repository, 2026. URL https://github.com/RomeroLab/BioDesignBench. BioDesignBench.

[33] Surojit Biswas, Grigory Khimulya, Ethan C. Alley, Kevin M. Esvelt, and George M. Church. Low-N protein engineering with data-eficient deep learning. Nature Methods, 18(4):389–396, 2021. doi: 10.1038/s41592-021-01100-y. URL https://doi.org/10.1038/s41592-021-01100-y.

[34] Chloe Hsu, Hunter Nisonof, Clara Fannjiang, and Jennifer Listgarten. Learning protein fitness models from evolutionary and assay-labeled data. Nature Biotechnology, 40(7):1114–1122, 2022. doi: 10.1038/s41587-021-01146-5. URL https://doi.org/10.1038/s41587-021-01146-5.

[35] Thomas A. Hopf, John B. Ingraham, Frank J. Poelwijk, Charlotta P. I. Schärfe, Michael Springer, Chris Sander, and Debora S. Marks. Mutation efects predicted from sequence co-variation. Nature Biotechnology, 35(2): 128–135, 2017. doi: 10.1038/nbt.3769. URL https://doi.org/10.1038/nbt.3769.

[36] Adam J. Riesselman, John B. Ingraham, and Debora S. Marks. Deep generative models of genetic variation capture the efects of mutations. Nature Methods, 15(10):816–822, 2018. doi: 10.1038/s41592-018-0138-4. URL https://doi.org/10.1038/s41592-018-0138-4.

[37] Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, and Alexander Rives. Language models enable zero-shot prediction of the efects of mutations on protein function. In Advances in Neural Information Processing Systems, volume 34, pages 29287–29303, 2021. URL https://proceedings.neurips.cc/paper/ 2021/hash/f51338d736f95dd42427296047067694-Abstract.html.

[38] Pascal Notin, Mafalda Dias, Jonathan Frazer, Javier Marchena-Hurtado, Aidan N. Gomez, Debora S. Marks, and Yarin Gal. Tranception: Protein fitness prediction with autoregressive transformers and inference-time retrieval. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 16990–17017. PMLR, 2022. URL https://proceedings.mlr.press/v162/ notin22a.html.

[39] Xinyou Wang, Liang Hong, Jiasheng Ye, Zaixiang Zheng, Shujian Huang, and Quanquan Gu. Towards a generative protein evolution machine with DPLM-evo. In ICLR 2026 Workshop on Generative and Experimental Perspectives for Biomolecular Design, 2026. URL https://openreview.net/forum?id=xqnTGzQmfB.

[40] Yinkai Wang, Jiaxing He, Yuanqi Du, Xiaohui Chen, Jianan Canal Li, Li-Ping Liu, Xiaolin Xu, and Soha Hassoun. Large language model is secretly a protein sequence optimizer. In ICLR 2025 Workshop on Learning Meaningful Representations of Life, 2025. URL https://arxiv.org/abs/2501.09274.

[41] Alireza Ghafarollahi and Markus J. Buehler. ProtAgents: Protein discovery via large language model multiagent collaborations combining physics and machine learning. Digital Discovery, 3(7):1389–1409, 2024. doi: 10.1039/D4DD00013G. URL https://doi.org/10.1039/D4DD00013G.

## Appendix

## A Evaluation Templates

To support standardized and reproducible evaluation, we provide the prompt templates used for the evaluation of general-purpose language models on PFArena. Agentic models use similar core templates, supplemented with additional instructions for tool calling.

## A.1 Single-mutant Generation

Task T1: Single-mutant generation provides the language model with the wildtype\_sequence of the evaluated protein, its sequence\_length, and its UniProt identifier (uniprot\_id, e.g., “P11413”). A concise description of its experimental assay context is also passed to the model, including the primary\_task\_class (e.g., “activity\_function”), fitness\_type (e.g., “enzymatic\_activity”), and assay\_readout\_subclass (e.g., “activity\_proxy”). The model is instructed to generate a list of plausible mutations and rank them according to their expected fitness based on its own internal knowledge.

## Single-mutant Generation Prompt Template

You are an expert protein engineer and computational biologist specializing in deep mutational scanning (DMS) and mutation   
-effect prediction.   
### TASK GOAL   
Given a wild-type protein sequence and its experimental assay context, predict the \*\*top 40 single point mutations\*\* (   
WTposMUT format) that optimize the target fitness metric, ordered from highest expected fitness to lowest expected fitness   
### REASONING & EVIDENCE BOUNDARIES   
1. \*\*Biochemical Deductions\*\*: Analyze residue chemistry, conservation, secondary structure propensities, steric packing,   
hydrophobic cores, electrostatic interactions, and sequence motifs within the supplied wild-type sequence.   
2. \*\*Assay Alignment\*\*: Align every ranked mutation strictly with the supplied assay readout. For example, if evaluating   
stability/abundance, prioritize mutations that improve hydrophobic packing or thermostability without disrupting necessary   
structural dynamics.   
3. \*\*No Hallucination\*\*: Do NOT invent or claim specific numerical model scores, experimental PDB coordinates, literature   
measurements, or alignments that are not logically derivable from sequence biochemistry.   
4. \*\*Single-Response Constraint\*\*: You have no external tools, browsing, or follow-up turns. Complete the analysis in this   
single response.   
### STRICT MUTATION & FORMAT CONSTRAINTS   
1. \*\*Ranking Size\*\*: The output list MUST contain \*\*exactly 40 mutations\*\*, ordered from best to worst.   
2. \*\*Format\*\*: Every mutant MUST be represented in standard 1-indexed ‘WTposMUT‘ format (e.g., ‘H24R‘, ‘A15V‘).   
3. \*\*Alphabet\*\*: Both ‘WT‘ and ‘MUT‘ must be standard 20 amino acid single-letter codes: ‘A, C, D, E, F, G, H, I, K, L, M,   
N, P, Q, R, S, T, V, W, Y‘.   
4. \*\*Uniqueness\*\*: Every mutation string MUST appear exactly once (no duplicates, no omissions within your top-40 list).   
5. \*\*Strict Validation\*\*:   
- Every mutation position follows ‘1 <= position <= Length‘.   
- ‘WT‘ MUST strictly match the character at ‘wildtype\_sequence[position - 1]‘.   
- ‘MUT‘ MUST be strictly different from ‘WT‘ (no synonymous/no-op mutations).   
6. \*\*Search Space\*\*: Consider all valid single amino acid substitutions across the full wild-type sequence, then return   
only the top 40.   
7. \*\*Prohibited Formats\*\*: Multi-site mutations, insertions (‘ins‘), deletions (‘del‘), stop codons (‘\*‘), HGVS notations,   
or numerical confidence scores.   
### OUTPUT FORMAT   
Return a \*\*valid JSON object ONLY\*\* with NO markdown code block wrappers, prefix, or conversational text. Use the   
following exact JSON schema:   
{   
"ranking": [   
"<best mutant>",   
"<second-best mutant>",   
"..."   
]   
}

```markdown
### INSTANCE DATA
1. ASSAY CONTEXT & TARGET METRICS
- **UniProt ID**: {uniprot_id}
- **Primary Task Class**: {primary_task_class}
- **Fitness Metric Type**: {fitness_type}
- **Assay Readout Subclass**: {readout_subclass}
2. INPUT WILD-TYPE SEQUENCE (Length: {sequence_length})
‘{wildtype_sequence}‘
```

## A.2 Multi-mutant Ranking

For the mutation-ranking tasks of T2–T4, the protein and assay metadata included in T1 is similarly provided. Additionally, the model receives the total amount of candidates to rank num\_candidates, and a shufled list of candidate\_mutants, each specified with the original amino-acid and position in the wild-type sequence followed by the mutant, in a “WTposMUT” format like “E19H”.

Measurement-free Ranking No additional experimental measurements are provided for T2: Measurementfree multi-mutant ranking. The model ranks the candidate mutants using only the shared information on the protein and assay context.

## Measurement-free Multi-mutant Ranking Prompt Template

You are an expert protein engineer and computational biologist specializing in deep mutational scanning (DMS) and mutation   
-effect prediction.   
### TASK GOAL   
Given a wild-type protein sequence, its experimental assay context, and a specific list of candidate multi-mutations, \*\*   
rank all candidate multi-mutations from best to worst\*\* according to their expected target fitness metric.   
### REASONING & EVIDENCE BOUNDARIES   
1. \*\*Biochemical Deductions\*\*: Compare the candidates based on residue chemistry, conservation, secondary structure   
propensities, steric packing, hydrophobic cores, electrostatic interactions, and sequence motifs within the supplied wild  
type sequence.   
2. \*\*Assay Alignment\*\*: Evaluate relative effects of these specific substitutions on the supplied assay readout. Rank   
mutations that better preserve or enhance structural/functional requirements above those that introduce severe clashes,   
charge mismatch, or instability.   
3. \*\*No Hallucination\*\*: Do NOT invent or claim specific numerical model scores, experimental PDB coordinates, literature   
measurements, or alignments that are not logically derivable from sequence biochemistry.   
4. \*\*Single-Response Constraint\*\*: You have no external tools, browsing, or follow-up turns. Complete the ranking in this   
single response.   
### STRICT MUTATION & FORMAT CONSTRAINTS   
1. \*\*Closed Set Principle\*\*: You MUST ONLY rank the mutations provided in the list above. Do NOT introduce new mutations,   
insertions, deletions, or wild-type strings.   
2. \*\*Multi-Mutation Candidates\*\*: A candidate contains multiple mutations joined by + (e.g., K23A+A40P+T52S). When ranking   
multi-mutation candidates, account for their combined effects.   
3. \*\*Exact Copy & Completeness\*\*:   
- The output list MUST contain \*\*exactly ‘total\_candidates‘ items\*\*.   
- Every candidate from the input list MUST appear \*\*exactly once\*\* (no duplicates, no omissions).   
- Each mutation string MUST be copied \*\*exactly as provided\*\*.   
4. \*\*Strict Validation\*\*:   
- Every mutation position follows ‘1 <= position <= Length‘.   
- ‘WT‘ MUST strictly match the character at ‘wildtype\_sequence[position - 1]‘.   
- ‘MUT‘ MUST be strictly different from ‘WT‘ (no synonymous/no-op mutations).   
### OUTPUT FORMAT   
Return a \*\*valid JSON object ONLY\*\* with NO markdown code block wrappers, prefix, or conversational text. Use the   
following exact JSON schema:   
{   
"ranking": [   
"<best mutant copied exactly from the provided list>",   
"<second-best mutant copied exactly from the provided list>",   
"..."   
]   
}

```markdown
### INSTANCE DATA
1. ASSAY CONTEXT & TARGET METRICS
- **UniProt ID**: {uniprot_id}
- **Primary Task Class**: {primary_task_class}
- **Fitness Metric Type**: {fitness_type}
- **Assay Readout Subclass**: {readout_subclass}
2. INPUT WILD-TYPE SEQUENCE (Length: {sequence_length})
‘{wildtype_sequence}‘
3. CANDIDATE MUTATIONS TO RANK (Total Candidates: {num_candidates})
{candidate_mutants}
```

Anchor-informed Ranking In task T3: Anchor-informed multi-mutant ranking, the model addition ally receives an anchor\_mutant shared by all candidates and its ground-truth anchor\_DMS\_score. This measurement provides a common experimental reference for ranking the candidate mutations.

## Anchor-informed Multi-mutant Ranking Prompt Template

You are an expert protein engineer and computational biologist specializing in deep mutational scanning (DMS) and mutation   
-effect prediction.   
### TASK GOAL   
Given a wild-type protein sequence, its experimental assay context, and a specific list of candidate multi-mutations, \*\*   
rank all candidate multi-mutations from best to worst\*\* according to their expected target fitness metric.   
The ground-truth DMS score of an anchor mutant appearing in every candidate, which may be single-site or multi-site, will   
also be provided.   
### REASONING & EVIDENCE BOUNDARIES   
1. \*\*Biochemical Deductions\*\*: Compare the candidates based on residue chemistry, conservation, secondary structure   
propensities, steric packing, hydrophobic cores, electrostatic interactions, and sequence motifs within the supplied wild  
type sequence.   
2. \*\*Assay Alignment\*\*: Evaluate relative effects of these specific substitutions on the supplied assay readout. Rank   
mutations that better preserve or enhance structural/functional requirements above those that introduce severe clashes,   
charge mismatch, or instability.   
3. \*\*No Hallucination\*\*: Do NOT invent or claim specific numerical model scores, experimental PDB coordinates, literature   
measurements, or alignments that are not logically derivable from sequence biochemistry.   
4. \*\*Single-Response Constraint\*\*: You have no external tools, browsing, or follow-up turns. Complete the ranking in this   
single response.   
### STRICT MUTATION & FORMAT CONSTRAINTS   
1. \*\*Closed Set Principle\*\*: You MUST ONLY rank the mutations provided in the list above. Do NOT introduce new mutations,   
insertions, deletions, or wild-type strings.   
2. \*\*Multi-Mutation Candidates\*\*: A candidate contains multiple mutations joined by + (e.g., K23A+A40P+T52S), with the   
anchor mutant included. When ranking multi-mutation candidates, account for their combined effects.   
3. \*\*Exact Copy & Completeness\*\*:   
- The output list MUST contain \*\*exactly ‘total\_candidates‘ items\*\*.   
- Every candidate from the input list MUST appear \*\*exactly once\*\* (no duplicates, no omissions).   
- Each mutation string MUST be copied \*\*exactly as provided\*\*.   
4. \*\*Strict Validation\*\*:   
- Every mutation position follows ‘1 <= position <= Length‘.   
- ‘WT‘ MUST strictly match the character at ‘wildtype\_sequence[position - 1]‘.   
- ‘MUT‘ MUST be strictly different from ‘WT‘ (no synonymous/no-op mutations).   
### OUTPUT FORMAT   
Return a \*\*valid JSON object ONLY\*\* with NO markdown code block wrappers, prefix, or conversational text. Use the   
following exact JSON schema:   
{   
"ranking": [   
"<best mutant copied exactly from the provided list>",   
"<second-best mutant copied exactly from the provided list>",   
"..."   
]   
}

```markdown
### INSTANCE DATA
1. ASSAY CONTEXT & TARGET METRICS
- **UniProt ID**: {uniprot_id}
- **Primary Task Class**: {primary_task_class}
- **Fitness Metric Type**: {fitness_type}
- **Assay Readout Subclass**: {readout_subclass}
2. INPUT WILD-TYPE SEQUENCE (Length: {sequence_length})
‘{wildtype_sequence}‘
3. ANCHOR MUTANT CONTEXT
- **Anchor Mutant**: {anchor_mutant}
- **Anchor DMS Score**: {anchor_DMS_score}
4. CANDIDATE MUTATIONS TO RANK (Total Candidates: {num_candidates})
{candidate_mutants}
```

Single-mutant-informed Ranking Task T4: Single-mutant-informed multi-mutant ranking additionally provides the ground-truth fitness scores of every component mutation appearing in the candidates as single\_mutant\_DMS\_score. Each mutation–score pair is formatted as “<mutant>: <dms\_score>”, and the pairs are joined by newline characters (“\n”). These measurements provide mutation-level evidence for ranking the multi-mutant combinations.

## Single-mutant-informed Multi-mutant Ranking Prompt Template

```markdown
You are an expert protein engineer and computational biologist specializing in deep mutational scanning (DMS) and mutation
-effect prediction.
### TASK GOAL
Given a wild-type protein sequence, its experimental assay context, and a specific list of candidate multi-mutations, **
rank all candidate multi-mutations from best to worst** according to their expected target fitness metric.
The ground-truth single-mutant DMS scores of every component appearing in any candidate will also be provided.
### REASONING & EVIDENCE BOUNDARIES
1. **Biochemical Deductions**: Compare the candidates based on residue chemistry, conservation, secondary structure
propensities, steric packing, hydrophobic cores, electrostatic interactions, and sequence motifs within the supplied wild
type sequence.
2. **Assay Alignment**: Evaluate relative effects of these specific substitutions on the supplied assay readout. Rank
mutations that better preserve or enhance structural/functional requirements above those that introduce severe clashes,
charge mismatch, or instability.
3. **No Hallucination**: Do NOT invent or claim specific numerical model scores, experimental PDB coordinates, literature
measurements, or alignments that are not logically derivable from sequence biochemistry.
4. **Single-Response Constraint**: You have no external tools, browsing, or follow-up turns. Complete the ranking in this
single response.
### STRICT MUTATION & FORMAT CONSTRAINTS
1. **Closed Set Principle**: You MUST ONLY rank the mutations provided in the list above. Do NOT introduce new mutations,
insertions, deletions, or wild-type strings.
2. **Multi-Mutation Candidates**: A candidate contains multiple mutations joined by + (e.g., K23A+A40P+T52S). When ranking
multi-mutation candidates, account for their combined effects.
3. **Exact Copy & Completeness**:
- The output list MUST contain **exactly ‘total_candidates‘ items**.
- Every candidate from the input list MUST appear **exactly once** (no duplicates, no omissions).
- Each mutation string MUST be copied **exactly as provided**.
4. **Strict Validation**:
- Every mutation position follows ‘1 <= position <= Length‘.
- ‘WT‘ MUST strictly match the character at ‘wildtype_sequence[position - 1]‘.
- ‘MUT‘ MUST be strictly different from ‘WT‘ (no synonymous/no-op mutations).
### OUTPUT FORMAT
Return a **valid JSON object ONLY** with NO markdown code block wrappers, prefix, or conversational text. Use the
following exact JSON schema:
{
"ranking": [
"<best mutant copied exactly from the provided list>",
"<second-best mutant copied exactly from the provided list>",
"..."
```

]   
}   
### INSTANCE DATA   
1. ASSAY CONTEXT & TARGET METRICS   
- \*\*UniProt ID\*\*: {uniprot\_id}   
- \*\*Primary Task Class\*\*: {primary\_task\_class}   
- \*\*Fitness Metric Type\*\*: {fitness\_type}   
- \*\*Assay Readout Subclass\*\*: {readout\_subclass}   
2. INPUT WILD-TYPE SEQUENCE (Length: {sequence\_length})   
‘{wildtype\_sequence}‘   
3. SINGLE MUTANT CONTEXT   
{single\_mutant\_dms\_scores}   
4. CANDIDATE MUTATIONS TO RANK (Total Candidates: {num\_candidates})   
{candidate\_mutants}

## A.3 Confidence Elicitation for Uncertainty Estimation

To elicit list-level confidence, we inserted the confidence elicitation instruction immediately before the original output contract. The original JSON ranking schema was retained, with one confidence field added after the ranking array; the non-numerical placeholder <your confidence> was used to avoid anchoring the model to an example value. The same confidence instruction was used in all settings. In T1, the original prohibition on “numerical confidence scores” was changed to “per-mutation confidence scores” to permit the required list-level value. All other task instructions, ranking constraints, assay information, sequences, candidate data, and contextual scores remained unchanged.

## Confidence Elicitation Prompt Template

After producing the ranking, report an overall confidence score from 0 to 100 indicating how confident you are in the   
predictive quality of the top-40 list for this specific protein and assay. The score should reflect confidence that the   
list contains and prioritizes genuinely high-fitness mutations, rather than confidence in formatting or instruction   
following.   
Report exactly one list-level ‘confidence‘ JSON number in [0,100]. Do not add per-mutation confidence scores or any other   
fields. This confidence instruction must not change the ranking or item-count rules.   
### OUTPUT FORMAT   
Return a \*\*valid JSON object ONLY\*\* with NO markdown code block wrappers, prefix, or conversational text. Use the   
following exact JSON schema:   
{   
"ranking": [   
"<best mutant>"   
"<second-best mutant>",   
"..."   
],   
"confidence": <your confidence>   
}   
### INSTANCE DATA   
1. ASSAY CONTEXT & TARGET METRICS   
- \*\*UniProt ID\*\*: <uniprot\_id>   
\*\*Primary Task Class\*\*: <primary\_task\_class>   
\*\*Fitness Metric Type\*\*: <fitness\_type>   
\*\*Assay Readout Subclass\*\*: <assay\_readout\_subclass>   
2. INPUT WILD-TYPE SEQUENCE (Length: <sequence\_length>)   
‘<wildtype\_sequence>‘

<table><tr><td>Model</td><td>Inputs and Checkpoint</td><td>Core Configuration</td><td>Candidate Score</td></tr><tr><td>ESM-2 [1]</td><td>Wild-type sequence; esm2_t33_650M_- UR50D</td><td>Masked inference with a 1,024-token context (at most 1,022 residues); de- terministic mutation-centered win- dows for longer chains</td><td>Sum of wild-type-context masked- marginal log odds, log  $p ( x _ { i } ^ { \mathrm { m u t } } ) \ -$  log p(xit), over substituted sites</td></tr><tr><td>ProGen2-base [2]</td><td>Complete wild-type and mutant se- quences; progen2-base</td><td>Forward and reversed causal se- quence scoring with a 2,048-token context; all evaluated chains were processed at full length</td><td>Difference between the bidirectional sequence scores of the complete mu- tant and wild-type chains</td></tr><tr><td>ProSST-2048 [3]</td><td>Wild-type sequence and AlphaFold 3 structure; AI4Protein/ProSST-2048</td><td>Official GVP quantizer with a 2,048- code structural vocabulary; all eval- uated chains were processed at full length within the 2,046-residue limit</td><td>Sum of structure-conditioned wild- type-context marginal log odds over substituted sites</td></tr><tr><td>S3F [4]</td><td>Wild-type sequence, AlphaFold 3 structure, and the corresponding molecular surface; released S3F checkpoint</td><td>Mutation-centered windows of at most 1,022 residues; sequence logits replace structure-conditioned logits where AF3 pLDDT is below 70</td><td>Sum of masked-marginal log-odds contributions over substituted sites</td></tr><tr><td>VenusREM [5]</td><td>Wild-type sequence, ProSST- 2048 structural tokens, and a UniRef100/MMseqs2 MSA</td><td>aa_seq_aln retrieval; α = 0.8, sam- pling ratio 1.0, and one sampling pass; full-chain inference for all eval- uated contexts</td><td>Sum of mutant-minus-wild-type val- ues from the fused sequence, struc- ture, and MSA representation</td></tr><tr><td>S3F-MSA [4, 6]</td><td>S3F score and an EVE ensemble trained on the corresponding wild- type-chain MSA</td><td>Five EVE seeds; 400,000 optimiza- tion steps, batch size 256, learning rate 10−4; 20,000 Monte Carlo sam- ples per seed at scoring time</td><td>Equal average of query-wise stan- dardized S3F and EVE scores, where the EVE score is the negative mean evolution index across seeds</td></tr></table>

Table 11 Deployment and scoring configurations of the protein-model baselines.

## B Experimental Setup

## B.1 PLM Deployment

Table 11 summarizes the model-specific inputs, checkpoints, inference configurations, and scoring definitions. Neural-network inference used FP32 precision, and all scores were oriented such that larger values indicate more favorable candidates. Each run was validated for input consistency, complete candidate coverage, finite outputs, and agreement between aggregate and component scores.

Multi-substitution scoring followed each baseline’s formulation. For ESM-2, ProSST-2048, S3F, and VenusREM, substitutions on the same chain were evaluated in a shared wild-type context and their sitewise contributions were summed. ProGen2-base instead evaluated the complete mutant sequence, whereas the EVE component of S3F-MSA assigned a joint score to the complete within-chain mutation set. For candidates spanning multiple chains, each naturally mutated chain was scored separately and the chain-level contributions were summed. Thus, the protocol preserves native chain boundaries but does not explicitly model inter-chain epistasis.

The MSAs used by VenusREM and EVE were generated against UniRef100 [22] with MMseqs2 [21]. One A3M alignment was associated with each wild-type chain; lowercase insertion symbols were removed where required by the parser, while alignment gaps were retained. For EVE, the focus-column and sequence-fragment gap thresholds were 1.0 and 0.5, respectively, and the sequence-reweighting threshold was 0.01 for viral proteins and 0.2 otherwise. Five EVE models were associated with each input context. Existing EVE ensembles were reused only when the wild-type sequence, processed MSA, and reweighting configuration were identical; all benchmark mutation sets were scored anew. Structure tokens, molecular surfaces, and MSA-derived models were prepared once per wild-type context rather than for every candidate mutation.

We generated a consistent set of wild-type structures with AlphaFold 3 (AF3) [28] for the structure-dependent baselines. Each natural chain was modeled independently as a monomer using random seed 1, 10 recycles, and five difusion samples. The corresponding UniRef100/MMseqs2 unpaired MSA was supplied directly; templates and additional database searches were disabled. The highest-ranked sample provided the PDB input used to derive ProSST structural tokens and S3F molecular surfaces. The canonical mmCIF files, confidence outputs, sample rankings, and all five samples were retained for reproducibility.

For the benchmark release, AF3 was run for all unique wild-type chains. Each result was required to reproduce the exact target sequence and residue mapping and to contain a complete protein backbone and valid confidence arrays. Of these, 184 structures passed the primary confidence criteria; the remaining 12 had lower pTM or mean pLDDT and were retained with a review flag because they remained sequence-consistent, structurally complete, and free of detected clashes. These flags were propagated with the structural resources so that confidence-dependent analyses can be performed without changing the benchmark coverage.

## B.2 Test-time Scaling Methods

For each T1: Single-mutant generation assay, let the i-th independently sampled top-40 ranking be

$$
R _ { i } = [ m _ { i 1 } , m _ { i 2 } , \ldots , m _ { i 4 0 } ] ,
$$

and let $r _ { i } ( m )$ denote the one-indexed rank of mutation m in $R _ { i }$ . We evaluated $n \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ using nested prefixes of the same 16 samples. Sample Avg was computed by evaluating each eligible raw ranking independently and averaging its assay-level metric.

All aggregation methods received the same extracted candidate strings. Candidates were deduplicated within each ranking while preserving their first occurrence. In the main scaling experiments, no explicit sequence-validity or DMS-membership filtering was applied. Incomplete rankings were padded to 40 entries using sample-specific invalid placeholders, preventing missing outputs from creating artificial agreement across samples. The aggregated top-40 output was mapped and scored only by the common evaluator.

Best-of-N We first computed the cross-sample reciprocal-rank consensus of each candidate,

$$
c ( m ) = \sum _ { i : m \in R _ { i } } \frac { 1 } { k + r _ { i } ( m ) } ,
$$

with $k = 1 0$ . Each complete ranking received the mean consensus score

$$
q _ { i } = { \frac { 1 } { | R _ { i } | } } \sum _ { m \in R _ { i } } c ( m ) ,
$$

and the ranking with the largest $q _ { i }$ was selected. Thus, Best-of-N always returned one intact sampled ranking and never recombined candidates across samples. Remaining ties were resolved in favor of the lower sample index.

Approval Voting Each mutation received one vote from every top-40 ranking in which it appeared,

$$
s _ { \mathrm { a p p r o v a l } } ( m ) = \sum _ { i } \mathbb { I } [ m \in R _ { i } ] .
$$

Candidates were sorted by vote count, followed by their mean observed rank, best observed rank, and mutation string. The 40 highest-ranked candidates from the union were returned.

Borda Fusion A mutation at rank r received 41 − r points, giving

$$
s _ { \mathrm { B o r d a } } ( m ) = \sum _ { i : m \in R _ { i } } \left( 4 1 - r _ { i } ( m ) \right) .
$$

This score jointly reflects occurrence frequency and a linear preference for mutations placed near the top of each sampled ranking.

Reciprocal Rank Fusion RRF assigned each mutation the score

$$
s _ { \mathrm { R R F } } ( m ) = \sum _ { i : m \in R _ { i } } \frac { 1 } { k + r _ { i } ( m ) } ,
$$

where $k = 1 0$ . Compared with Borda Fusion, reciprocal weighting places relatively greater emphasis on the highest-ranked candidates.

Hierarchical RRF To aggregate evidence shared by diferent substitutions at the same residue, we defined $p ( m )$ as the residue position of mutation m and $r _ { i } ^ { \mathrm { p o s } } ( p )$ as the first rank at which position $p$ occurred in $R _ { i }$ Position-level support was

$$
s _ { \mathrm { p o s } } ( p ) = \sum _ { i : \exists m \in R _ { i } , \ p ( m ) = p } \frac { 1 } { k + r _ { i } ^ { \mathrm { p o s } } ( p ) } ,
$$

and the final score was

$$
s _ { \mathrm { h i e r } } ( m ) = s _ { \mathrm { R R F } } ( m ) + \lambda s _ { \mathrm { p o s } } ( p ( m ) ) ,
$$

with $k = 1 0$ and $\lambda = 0 . 3 5$ . Candidates whose strings could not be parsed into residue positions retained their exact-mutation RRF support but received no position-level contribution.

For Borda, RRF, and Hierarchical RRF, ties were resolved by higher occurrence frequency, better mean observed rank, and then mutation string. We used no per-position diversity cap, so multiple substitutions at the same residue could appear in the final top-40 ranking.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Model</td><td colspan="4">Baseline</td><td rowspan="2">Adaptive</td></tr><tr><td>T1</td><td>T2</td><td>T3</td><td>T4</td></tr><tr><td>Statistic</td><td>Total</td><td>123</td><td>74</td><td>67</td><td>29</td><td>123</td></tr><tr><td rowspan="6">LLM</td><td>GPT-6 Astra</td><td>7</td><td>0</td><td>0</td><td>0</td><td>5</td></tr><tr><td>Claude Opus 5</td><td>12</td><td>6</td><td>6</td><td>1</td><td>9</td></tr><tr><td>Gemini 3.1 Pro</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Kimi K3</td><td>2</td><td>3</td><td>2</td><td>1</td><td></td></tr><tr><td>GLM-5.2</td><td>0</td><td>4</td><td>2</td><td>0</td><td></td></tr><tr><td>DeepSeek-V4-Pro</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr><tr><td rowspan="5">Agent</td><td>Biomni + GPT-6 Astra</td><td>9</td><td>1</td><td>1</td><td>1</td><td></td></tr><tr><td>Biomni + Claude Opus 5</td><td>0</td><td>4</td><td>1</td><td>0</td><td></td></tr><tr><td>AMixKit + GPT-6 Astra</td><td>2</td><td>0</td><td>0</td><td>0</td><td></td></tr><tr><td>AMixKit + Claude Opus 5</td><td>1</td><td>2</td><td>0</td><td>1</td><td></td></tr><tr><td>AMixKit + AMix-2.1</td><td>0</td><td>0</td><td>0</td><td>1</td><td></td></tr></table>

Table 12 Number of assay-level queries replaced by the deterministic SHA-256 random baseline during evaluation, for the baseline PFArena setting and the multi-round adaptive search setting. A dash indicates that the corresponding task was not run for that model.

## C Extended Experimental Analysis

## C.1 Evaluation Completeness and Missing-Query Handling

Despite repeated retries, some model–task pairs involving general-purpose LLMs produced no prediction that satisfied the required output format and evaluation constraints. These failures occurred particularly when the content-filtering policies of certain general-purpose LLMs restricted specific inputs, observed most frequently for Claude Opus 5, GPT-6 Astra, Kimi K3 and GLM-5.2, with frequency varying across models and tasks.

Fallback Substitutions To handle this issue, we distinguished successful predictions from missing assaylevel queries rather than assigning a score of zero to failed queries. Missing queries were replaced by a deterministic SHA-256 random baseline. For each missing query, the evaluator collected all measured mutants in the corresponding ground-truth table and computed its SHA-256 digest. The measured mutants were ordered lexicographically by these hexadecimal digests. For T1, the first 40 mutants in this order were used as the substitute prediction list. For T2–T4, the same procedure was applied independently to each candidate query, after which the evaluator applied the corresponding task-specific ranking budget.

The numbers of substituted queries are summarized in Table 12. Because this fallback ordering is independent of the DMS\_score ranking requirement, the reported aggregate metrics combine model predictions from successfully completed queries with the random-baseline lists for missing queries. Consequently, a larger number of missing queries will generally depress the aggregate score.

Matched-query Analysis To isolate intrinsic model performance from the efects of incomplete outputs, we curated the multi-round adaptive search evaluation of Section 5.2.1 on all 123 ground-truth assay queries to a common cohort of 113 assays, as shown in Figure 11. The common cohort contained only assays for which every model produced a valid cumulative prediction prefix across all four rounds; no random substitution was used for these assays. By contrast, the original multi-round evaluation retained all 123 assays and used the deterministic random fallback whenever a model lacked a complete prediction. At the fourth round, complete-prefix coverage ranged from 114 assays for Claude Opus 5 and 118 assays for GPT-6 Astra to all complete 123 assays for Gemini 3.1 Pro and DeepSeek-V4-Pro.

Across the common cohort, all models showed progressive improvements in both NMS@40 and Recall@40 over successive rounds. At round four, common-cohort NMS@40 exceeded the corresponding full-cohort value by 0.018 for GPT-6 Astra and by approximately 0.015 for the other three models. Recall@40 was boosted by approximately 0.0011 to 0.0023 across models. The higher common-cohort scores for Gemini 3.1 Pro and DeepSeek-V4-Pro, which had complete coverage of all 123 assays, suggest that the 10 assays excluded from the matched cohort due to incomplete predictions from other models were more dificult on average. These common-cohort results therefore provide a cleaner comparison of intrinsic model quality, whereas the full-cohort curves preserve performance under the original evaluation protocol.

![](images/63eef62106928c5e281af279d53b94068ec329174e0a1b2368f9db1664c89743.jpg)  
Figure 11 Matched-query analysis of round-wise single-mutant generation. Marker lines show cumulative performance on all 123 assay queries, with deterministic random-baseline substitutions used for missing queries. Thin curves show performance on the common cohort of 113 assays completed by every model in all four rounds. Shaded regions indicate the performance gap between the full and matched evaluations.

## C.2 Complementary Strengths and Failure Modes of LLMs and PLMs

We tested whether PLMs and LLMs identify the same useful mutants and make the same ranking errors. GPT-6 Astra represented the LLM family, and VenusREM represented the PLM family. The comparison had two parts: T1 measured overlap and experimental quality among generated single-mutant candidates, whereas T2–T4 measured whether one model could correct extreme ranking errors made by the other. This design separates complementary exploration in T1 from complementary error correction in multi-mutant ranking.

T1: Single-mutant Exploration GPT-6 Astra and VenusREM selected largely diferent single-mutant candidates, but their small consensus set was the most reliable. As shown in Figure 12, across 116 paired assays, VenusREM and GPT-6 Astra produced 4,640 and 4,501 evaluable top-40 selections, respectively, with 273 shared mutant–assay pairs. The shared candidates therefore represented only a small fraction of the selections made by either model. Nevertheless, the consensus set had the highest experimental quality: the median true rank score was 0.79 for shared candidates, compared with 0.66 for VenusREM-only candidates and 0.67 for GPT-6 Astra-only candidates, where 1 denotes the best experimental rank. Thus, agreement was uncommon but informative: consensus candidates can provide a high-confidence shortlist, whereas model-specific candidates expand the explored sequence space.

T2–T4: Ranking Errors We next examined whether LLMs and PLMs make complementary ranking errors in the multi-mutant tasks. For each task, experimental fitness and model predictions were converted into within-assay rank scores, with 1 denoting the best mutant. In Figure 13, the x-axis shows the true fitness rank score and the y-axis shows the predicted rank score. The diagonal corresponds to perfect rank agreement, while vertical arrows connect the two predictions for the same mutant. We defined an extreme error as placing a true bottom-30% mutant in the predicted top 10% or a true top-30% mutant in the predicted bottom

Fitness distribution of top-40 selections  
![](images/a1c8e6a9616e479e5676431a091f65c2b00ac62033ff4a73caf06c001d8b41aa.jpg)  
Figure 12 Complementary single-mutant selection in T1. Left, overlap between evaluable top-40 selections from VenusREM (PLM) and GPT-6 Astra (LLM) across 116 paired assays. Right, experimental fitness-rank distributions of PLM-only, shared and LLM-only selections. Rank scores were normalized within each assay from 1 (best) to 0 (worst). n denotes mutant–assay selections.

10%. A correction was counted when the other model assigned the same mutant a substantially less extreme rank, moving it closer to its measured rank. This paired, case-level analysis distinguishes shared failures from model-specific errors that can be rescued by the other model. The direction-specific counts further show that the corrected cases include both low-fitness over-rankings and high-fitness under-rankings, indicating that the complementarity is not restricted to a single type of ranking failure.

The error-correction patterns in T2 and T3 indicate that VenusREM remains at least as reliable as GPT-6 Astra when limited or localized experimental evidence is available, while they still provide complementary predictions. In T2, VenusREM corrected 45 extreme GPT-6 Astra errors, including 24 low-fitness over-rankings and 21 high-fitness under-rankings, whereas GPT-6 Astra corrected 30 VenusREM errors, including 24 and 6 of these two types, respectively. In T3, the two models corrected the same total number of extreme errors, with GPT-6 Astra correcting 16 low-fitness over-rankings and 5 high-fitness under-rankings, and VenusREM correcting 11 and 10, respectively.

The availability of complete single-mutant context reverses the direction of error correction in favor of GPT-6 Astra. In T4, GPT-6 Astra corrected 39 extreme VenusREM errors, including 29 low-fitness over-rankings and 10 high-fitness under-rankings, whereas VenusREM corrected 24 GPT-6 Astra errors, including 13 and 11 of these two types, respectively. This reversal mirrors the higher aggregate performance of LLM-based systems in T4, where measured fitness values for the component single mutations are available as additional evidence for ranking multi-mutant candidates. The results therefore support a context-dependent shift in model utility rather than a universal superiority of either model family.

Taken together, VenusREM and GPT-6 Astra exhibit complementary ranking errors across T2–T4, with each correcting both low-fitness over-rankings and high-fitness under-rankings made by the other. These patterns motivate combined or agent-mediated approaches, but this descriptive analysis does not establish that an ensemble would improve aggregate performance.

![](images/53764805ae41fdb716a44aa65980080b9914a6667b845a810223b3559fdc3150.jpg)  
Figure 13 Complementary correction of extreme ranking errors by GPT-6 Astra and VenusREM across T2 –T4. The left column shows mutants for which GPT-6 Astra corrected extreme VenusREM ranking errors, and the right column shows the converse. True fitness rank scores range from 0 (worst) to 1 (best), whereas predicted rank scores range from 0 (bottom) to 1 (top). Orange crosses and blue circles denote VenusREM and GPT-6 Astra predictions, respectively; grey arrows connect predictions for the same mutant. Shaded regions indicate true bottom-30% mutants predicted in the top 10% or true top-30% mutants predicted in the bottom 10%. Central labels report the total number of mutants in each direction.