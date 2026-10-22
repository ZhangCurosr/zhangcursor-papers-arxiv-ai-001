# From Scores to Safe Sets: Exact Safety-Constrained Decoding for Medication Recommendation

Animesh Agarwal Meysam Ghaffari Nina Fatehi Anushka Swarup Carlos Morato<sup>∗</sup> Optum

## Abstract

Recent generative medication recommendation systems ultimately produce sets of drugs, but drug–drug interaction (DDI) constraints depend on the set as a whole. A natural post-hoc approach is to first predict medications and then locally repair interacting pairs by removing one drug at a time. We ask whether this local repair step itself introduces avoidable error. Using fixed medication scores from FLAME’s frozen classifier on MIMIC-III, we compare increasingly strong local repair strategies with an exact global decoder while holding the scores, output size, and DDI constraint fixed. The global decoder jointly selects the highest-scoring feasible medication set using mixed-integer linear programming. Across validation and test splits, global decoding consistently achieves higher Jaccard similarity than greedy repair, degree-aware repair, and degree-aware repair strengthened with exhaustive 1-swap local search. Its advantage is small under loose DDI constraints and becomes larger as the constraint tightens; on the test split it reaches +0.0064 Jaccard over the strongest local baseline at the strict DDI operating point. The finding also holds across a fine accuracy–DDI sweep at matched realized DDI exposure and under three score representations. At the strict setting, global decoding retains more true-positive medications while removing more false positives. These results show that medication recommendation is not only a representation-learning problem: even with a frozen clinical model, how its scores are converted into a constrained medication set can materially affect the final prediction.

## 1 Introduction

Medication recommendation from electronic health records (EHRs) is a canonical set-valued prediction problem. Given diagnoses, procedures, and longitudinal context, a model must recommend a combination of medications rather than a single label. This distinction matters because interactions are relational, duplicate or incompatible therapies arise at the set level, and the predicted medication set may need to satisfy multiple constraints simultaneously. A large literature has therefore developed increasingly structured medication recommenders, including sequence models, memory networks, graph-augmented models, molecular encoders, set-to-set predictors, causal representations, multimodal models, and more recently large-language-model (LLM) systems [Zhang et al., 2017, Le et al., 2018, Shang et al., 2019b,a, Yang et al., 2021a,b, Wu et al., 2022, Tan et al., 2022, Wu et al., 2023, Yang et al., 2023, Chen et al., 2023, Zhao et al., 2024, Kim et al., 2024, Li et al., 2024, Tan et al., 2024, Liu et al., 2024, Li et al., 2025, Wu et al., 2025, Zhao et al., 2025, Fan et al., 2025, Lv et al., 2026, Huh et al., 2026]. A recurring objective across this literature is to preserve recommendation accuracy while reducing harmful drug–drug interactions (DDIs) [Shang et al., 2019b, Yang et al., 2021b, Tan et al., 2022, Wu et al., 2023, Yang et al., 2023, Chen et al., 2023].

Throughout this paper, references to “safety” refer only to the benchmark DDI constraint used in the evaluation. They should not be interpreted as establishing comprehensive clinical safety, clinical utility, or a prescribing recommendation for the resulting medication sets.

Most prior work attacks this tradeoff primarily through learning: DDI graphs are incorporated into representation learning, molecular structure is encoded to improve drug representations, DDI penalties are added to the training objective, or list-wise objectives are used to align recommendation behavior [Shang et al., 2019b, Yang et al., 2021b, 2023, Chen et al., 2023, Fan et al., 2025]. A systematic survey of medication recommendation also highlights a basic difficulty with DDI-based post-processing: removing one interacting drug can discard treatment value elsewhere in the regimen [Ali et al., 2023]. Prior work therefore mainly asks how to learn better scores or safer representations. We ask a complementary question: once a patient-specific scorer already exists, does error arise when those scores are converted into a DDI-constrained medication set? Our central finding is that it does: even strong local repair leaves measurable accuracy on the table relative to global constrained decoding. For modern generative medication recommenders, this final decoding step is the point where model-derived scores are converted into a coupled medication set.

A natural local strategy is to repair conflicts after prediction. Start from the model’s preferred set, identify a DDI conflict, remove one of the two drugs, and repeat until a target DDI level is reached. A stronger heuristic considers each drug’s conflict degree and removes high-degree nodes first, and stronger still, a local-search procedure repairs bad greedy commitments after the fact by searching every feasible swap. All three remain local: a removal is committed before the algorithm knows how that choice will constrain later removals and replacements.

The problem becomes combinatorial when conflicts are dense. In our evaluation regime, output sets contain approximately 27 medications on average, and a typical predicted set contains dozens of simultaneously active DDI edges. Removing one drug may resolve many edges at once but may also force the loss of a highly supported medication; retaining a high-scoring “hub” drug can make several other high-scoring medications mutually infeasible. The quality of a local decision therefore depends on the entire conflict graph, which motivates a global formulation that selects the full set jointly.

We formulate DDI-constrained decoding as a mixed-integer linear program (MILP). For each patient visit $q ,$ a frozen scorer provides one score $s _ { i } ( \boldsymbol q )$ per medication. We select exactly $K _ { q }$ medications to maximize the total score, subject to an explicit budget $B _ { q }$ on selected DDI edges. Every decoder returns the same number of medications and the same DDI budget, so the comparison isolates one factor: local versus global selection. Figure 1 summarizes this shared setup.

We instantiate the scorer with FLAME’s released drug-level classifier, $\pi _ { \mathrm { c l s } }$ [Fan et al., 2025], on MIMIC-III [Johnson et al., 2016], using the TWOSIDES-derived DDI graph released with FLAME’s preprocessing [Fan et al., 2025, Tatonetti et al., 2012]. All methodological choices are made on a training split; a disjoint validation split confirms the phenomenon before a held-out test split is scored once. The three splits follow a 4:1:1 train/validation/test ratio over approximately 14,000 total visits. We compare exact global decoding with (i) greedy repair, (ii) degree-aware repair, and (iii) degree-aware repair strengthened with exhaustive 1-swap local search. Across loose, moderate, and strict DDI budgets, the ordering $J _ { \mathrm { g l o b a l } } > J _ { \mathrm { 1 - s w a p } } > J _ { \mathrm { d e g r e e } } > J _ { \mathrm { g r e e d y } }$ holds on both validation and test. The advantage grows as the DDI constraint tightens, holds when methods are compared at matched realized (not merely target) DDI exposure across a fine budget sweep, and holds under three different score representations.

## Our contributions are:

1. A decoding-level view of DDI-constrained recommendation. We hold the patient–drug scorer, output cardinality, and DDI budget fixed and isolate the final set-selection problem.

2. A controlled comparison of local and global decoding. We compare exact MILP decoding with greedy repair, degree-aware repair, and degree-aware repair strengthened by exhaustive 1-swap local search.

3. Consistent and distinguishable evidence across splits and operating points. The advantage reproduces on validation and test, persists at matched realized DDI exposure over a fine DDI sweep, and holds under three score parameterizations.

![](images/46cc0fc92d73a7b200f07aa72557eb7e2f90280dd087ad8eadd990f095755fa4.jpg)  
Figure 1: Controlled comparison of local repair and global DDI-constrained decoding. A frozen FLAME scorer produces patient-specific medication scores $s ( q )$ . All decoding methods use the same scores, desired output cardinality $K _ { q } ,$ drug–drug interaction (DDI) graph G, and patient-specific DDI budget $B _ { q }$ . Local methods construct or repair the medication set through sequential decisions, whereas the exact MILP jointly selects the entire set to maximize the frozen model score while satisfying the same cardinality and DDI constraints. The resulting medication sets are compared using set accuracy and achieved DDI rate. Thus, the experiment isolates the effect of the decoding strategy while keeping the underlying model scores and DDI constraints fixed.

4. A mechanistic account. Under strict DDI-constraint pressure, global decoding retains more true positives while removing more false positives, consistent with the cost of locally committing to coupled medication choices.

## 2 Related work

Medication recommendation from EHRs. Medication recommendation has progressed from instance- and sequence-based prediction to increasingly structured set prediction. LEAP framed treatment-combination recommendation as sequential generation under multimorbidity [Zhang et al., 2017]; DMNC modeled asynchronous longitudinal views with external memory [Le et al., 2018]; GAMENet fused longitudinal patient representations with EHR and DDI graphs [Shang et al., 2019b]; and G-BERT introduced graph-augmented pretraining for medical-code representations [Shang et al., 2019a]. MICRON focused on medication changes across visits [Yang et al., 2021a], while COGNet used conditional generation with copy-or-predict behavior over historical medications [Wu et al., 2022]. DAPSNet added patient-similarity retrieval to longitudinal representation learning [Wu et al., 2023]. More recent work improves visit selection, rare-disease performance, causal patient-state modeling, or temporal treatment reasoning, including VITA, RAREMed, CausalMed, and time-aware dynamic-treatment approaches [Kim et al., 2024, Zhao et al., 2024, Li et al., 2024, 2025]. Surveys summarize the rapid expansion of architectures and evaluation practices in this area [Ali et al., 2023].

DDI-aware and set-structured recommendation. DDI-aware recommendation is often incorporated during training. GAMENet integrates a DDI graph into its memory architecture [Shang et al., 2019b]. SafeDrug uses molecular graph encoders together with a controllable DDI loss [Yang et al., 2021b]. 4SDrug explicitly targets small and safe drug sets [Tan et al., 2022]. MoleRec learns substructure-aware molecular representations and uses an annealed objective to balance recommenda tion accuracy and DDI safety [Yang et al., 2023]. Carmen designs context-aware molecular and DDI graph embeddings [Chen et al., 2023], and DAPSNet includes a DDI-guided loss in a longitudinal patient-similarity model [Wu et al., 2023]. These methods show that DDI information can improve the learned predictor. Our focus is orthogonal: we freeze the predictor and study the optimization problem induced at the final set-selection stage.

Language models and generative medication recommendation. Recent systems bring pretrained language models and LLMs into medication recommendation. LEADER distills knowledge from an LLM into a medication recommender [Liu et al., 2024]; NLA-MMR aligns natural-language patient and drug representations with molecular information [Tan et al., 2024]; LAMO fine-tunes LLMs for medication recommendation [Zhao et al., 2025]; FLAME adds list-wise alignment and step-wise GRPO to a point-wise LLM-based medication classifier [Fan et al., 2025]; GiantMed uses deep-model guidance to activate LLM reasoning near recommendation boundaries [Lv et al., 2026]; and PACE-RAG studies patient-aware retrieval for clinical drug recommendation [Huh et al., 2026]. Our experiments use FLAME only as a source of fixed patient–drug scores. The proposed decoder neither trains nor queries an additional language model, making it a modular inference layer that can in principle sit on top of any scorer.

Constrained inference. Mixed-integer optimization is a standard tool for exact discrete decision making under linear constraints [Nemhauser and Wolsey, 1988]. The medication-recommendation literature frequently describes the task itself as combinatorial [Zhang et al., 2017, Tan et al., 2022, Yang et al., 2023, Fan et al., 2025], but the dominant safety interventions in learned systems act through training losses, graph representations, sequential generation, or heuristic post-processing. We isolate the final inference problem and solve the fixed-score, fixed-cardinality, DDI-budgeted set-selection objective exactly, permitting a controlled comparison in which the score function is identical and only the decoder changes.

## 3 Problem formulation

Let $\mathcal { M } = \{ 1 , \dots , M \}$ denote the medication vocabulary, with $M = 1 5 1$ in our experiments. For each patient visit $q ,$ a frozen model provides scores

$$
\begin{array} { r } { s ( q ) = [ s _ { 1 } ( q ) , \dots , s _ { M } ( q ) ] \in \mathbb { R } ^ { M } , } \end{array}\tag{1}
$$

where larger $s _ { i } ( \boldsymbol q )$ indicates stronger model evidence for medication i. The base model also induces an initial medication set $\hat { Y } _ { q } ^ { ( 0 ) }$ under its fixed operating point. We define

$$
K _ { q } = | \hat { Y } _ { q } ^ { ( 0 ) } |\tag{2}
$$

and require every decoder to return exactly $K _ { q }$ medications. This is the cardinality-matching condition the comparison depends on: since no decoder can alter its cardinality, none can appear to satisfy a looser DDI constraint by returning fewer drugs.

Let $G _ { D } = ( \mathcal { M } , E _ { D } )$ be an undirected DDI graph, where an edge $( i , j ) \in E _ { D }$ indicates a flagged drug–drug interaction. For any predicted set $\bar { Y }$ , we use the pairwise DDI rate

$$
\operatorname { D D I } ( Y ) = { \frac { | \{ ( i , j ) \in E _ { D } : i \in Y , j \in Y \} | } { \binom { | Y | } { 2 } } } .\tag{3}
$$

We use three nominal target rates $\rho \in \{ 0 . 1 3 6 , 0 . 1 0 0 , 0 . 0 8 0 \}$ , treated as loose, moderate, and strict prespecified DDI regimes rather than a search for an optimal operating point. Each rate is mapped to the nearest feasible integer edge budget

$$
B _ { q } ( \rho ) = \mathrm { r o u n d } \left( \rho { \binom { K _ { q } } { 2 } } \right) ,\tag{4}
$$

using the same deterministic conversion for every decoder. Because of this discretization, we report achieved DDI rates in addition to the nominal targets. The task studied here is not to learn $s ( q ) ;$ ; it is to solve the decoding problem: among all medication sets of size $K _ { q }$ whose selected DDI edge count does not exceed $B _ { q } .$ , choose a set that best respects the frozen scorer.

## 4 Exact DDI-constrained decoding

## 4.1 MILP formulation

For each medication i, introduce a binary selection variable $x _ { i } ~ \in ~ \{ 0 , 1 \}$ . For each DDI edge $( i , j ) \in E _ { D }$ , introduce an auxiliary variable $y _ { i j } ~ \in ~ [ 0 , 1 ]$ indicating whether both endpoints are selected. We solve

$$
\operatorname* { m a x } _ { x , y } \quad \sum _ { i = 1 } ^ { M } s _ { i } ( q ) x _ { i }\tag{5}
$$

$$
\mathrm { s . t . } \sum _ { i = 1 } ^ { M } x _ { i } = K _ { q } ,\tag{6}
$$

$$
\sum _ { ( i , j ) \in E _ { D } } y _ { i j } \le B _ { q } ,\tag{7}
$$

$$
y _ { i j } \geq x _ { i } + x _ { j } - 1 , \qquad \forall ( i , j ) \in E _ { D } ,\tag{8}
$$

$$
y _ { i j } \leq x _ { i } , \quad y _ { i j } \leq x _ { j } , \qquad \forall ( i , j ) \in E _ { D } ,\tag{9}
$$

$$
x _ { i } \in \{ 0 , 1 \} , \quad 0 \leq y _ { i j } \leq 1 .\tag{10}
$$

With binary $x _ { i } .$ , constraints (8)–(9) give the exact product $y _ { i j } = x _ { i } x _ { j } , \mathrm { s o } \mathrm { E q . } ( 7 )$ counts the selected DDI edges exactly. The formulation has 151 binary variables and 1,425 auxiliary edge variables, for 1,576 variables total (Appendix A).

The MILP never sees the ground-truth medication set during inference; it only sees the frozen model scores, target cardinality, DDI graph, and DDI budget. At global optimality, the MILP has objective value at least as high as any heuristic solution obeying the same constraints, but a higher score sum is not guaranteed to correspond to better clinical-label agreement, nor is the same decoded set guaranteed under a rescoring of $s ( q )$ that preserves rank but not magnitude. We evaluate both questions directly in Section 7.

## 4.2 Why global optimization can differ from local repair

A local repair rule compares medications inside the conflict it is currently resolving. This can be suboptimal when a high-scoring drug participates in many conflicts: retaining that drug may force the subsequent removal of several slightly lower-scoring medications whose joint score is larger. The MILP evaluates those coupled consequences simultaneously. This effect should strengthen as $B _ { q }$ decreases, since tighter budgets reduce the number of feasible combinations and give each local commitment more downstream consequences. We evaluate whether the advantage in Jaccard grows as DDI-constraint pressure increases, and whether it depends on which baseline, operating point, or score parameterization is used.

## 4.3 Implementation

We solve the MILP with scipy.optimize.milp, using the HiGHS mixed-integer optimization backend [Virtanen et al., 2020, Huangfu and Hall, 2018]. No warm start, learning, or patient-specific tuning is required. Full solver settings are given in Appendix A.

## 5 Local repair baselines

Greedy + refill. We begin from the top- $K _ { q }$ medications under the frozen score. While the selected set exceeds the DDI budget, the decoder finds a conflicting pair and removes the lower-scoring endpoint. Once the budget is met, it scans excluded medications in descending score order and refills the set with the highest-scoring candidate whose addition preserves feasibility, continuing until cardinality $K _ { q }$ is restored.

Degree-aware greedy + refill. This baseline uses graph structure: at each repair step, it computes every selected medication’s current degree in the induced DDI subgraph and removes a medication with maximum conflict degree, breaking ties by lower model score. Refill is identical to the greedy baseline. Removing a high-degree medication can resolve several conflicts at once.

Degree-aware + 1-swap local search. Starting from the degree-aware solution, we repeatedly search every feasible 1-for-1 swap $( i \in S$ selected, j $\notin S$ excluded) and apply the swap that most increases total score while keeping the DDI budget satisfied, until no improving swap remains. Each candidate swap’s resulting conflict count is evaluated in $O ( 1 )$ via an incremental update, $C ^ { \prime } = C - \deg _ { S } ( i ) + \deg _ { S } ( j ) - \bigcup _ { i j }$ , where $\deg _ { S } ( \cdot )$ is a medication’s current conflict degree against the selected set S and C is the set’s current conflict count, making the full swap neighborhood cheap to search at every iteration (Appendix A). This baseline can undo a bad greedy commitment rather than only building on top of it.

Controlled comparison. All decoders receive exactly the same $s ( q ) , K _ { q }$ , DDI graph, and $B _ { q } .$ , and differ only in how they search the feasible set space.

## 6 Experimental setup

Data and scorer. Patient–drug scores are taken from the frozen drug-level classifier $\pi _ { \mathrm { c l s } }$ released with FLAME [Fan et al., 2025], which is not retrained or altered for this study. We use the MIMIC-III Clinical Database v1.4, a de-identified dataset distributed through PhysioNet [Johnson et al., 2016]. The medication vocabulary contains 151 drugs. The DDI adjacency is the 151-drug graph used by FLAME’s released MIMIC-III preprocessing, derived from TWOSIDES [Fan et al., 2025, Tatonetti et al., 2012], and contains 1,425 undirected edges.

Training, validation, and test protocol. We use three disjoint splits of MIMIC-III visits [Johnson et al., 2016], following a 4:1:1 train/validation/test ratio over approximately 14,000 visits in total, with no shared patient identifiers across splits. The margin-to-cardinality scorer that defines $K _ { q }$ (a single logistic calibration of the classifier’s confidence margin) is fit exclusively on the training split, with its decision threshold selected on a held-out 20% slice of training. This fitted scorer, together with the MILP formulation, the three local-repair baselines, and the three DDI regimes, is fixed before the validation split is examined. We report results on the validation split as a confirmatory check, then evaluate the identical, unmodified pipeline once on the test split. Test-split numbers are the primary results reported throughout; validation-split numbers are reported alongside them to show the phenomenon reproduces on an independent sample.

DDI regimes. We use target DDI rates 0.136, 0.100, and 0.080 as loose, moderate, and strict operating regimes. For the fine-grained frontier analysis (Section 7.3) we additionally sweep ten target rates from 0.14 to 0.02. We report both the nominal target and the achieved DDI rate throughout because the edge budget is integer-valued.

Accuracy metrics. For predicted set $\hat { Y } _ { q }$ and ground-truth set $Y _ { q } ,$ our primary accuracy metric is visit-level Jaccard,

$$
J _ { q } = \frac { | \hat { Y } _ { q } \cap Y _ { q } | } { | \hat { Y } _ { q } \cup Y _ { q } | } ,\tag{11}
$$

averaged across visits. Because all decoders output the same $K _ { q }$ for a visit, accuracy differences cannot be attributed to different set cardinalities.

Statistical analysis. We perform 2,000 patient-cluster bootstrap resamples over SUBJECT\_ID, rather than resampling visits independently, and report percentile 95% confidence intervals for the mean Jaccard difference between global decoding and each baseline.

Mechanism analysis. At the strict budget, we compare each decoded set with the patient’s original pre-repair top- $K _ { q }$ set and ground truth, recording true-positive retention and removal, false-positive removal, true-positive replacements added, and the final sum of frozen model scores.

Table 1: Main comparison on the test split, at matched per-patient cardinality and matched DDI budget. All methods output 26.78 medications on average.
<table><tr><td>Target</td><td>Decoder</td><td>Actual DDI</td><td>Jaccard</td><td>Avg. size</td></tr><tr><td rowspan="4">0.136</td><td>Greedy</td><td>0.1184</td><td>0.4107</td><td>26.78</td></tr><tr><td>Degree-aware</td><td>0.1157</td><td>0.4155</td><td>26.78</td></tr><tr><td>Degree-aware + 1-swap</td><td>0.1179</td><td>0.4186</td><td>26.78</td></tr><tr><td>Global MILP</td><td>0.1176</td><td>0.4207</td><td>26.78</td></tr><tr><td rowspan="4">0.100</td><td>Greedy</td><td>0.0933</td><td>0.3920</td><td>26.78</td></tr><tr><td>Degree-aware</td><td>0.0909</td><td>0.4035</td><td>26.78</td></tr><tr><td>Degree-aware + 1-swap</td><td>0.0929</td><td>0.4062</td><td>26.78</td></tr><tr><td>Global MILP</td><td>0.0928</td><td>0.4112</td><td>26.78</td></tr><tr><td rowspan="4">0.080</td><td>Greedy</td><td>0.0757</td><td>0.3790</td><td>26.78</td></tr><tr><td>Degree-aware</td><td>0.0738</td><td>0.3937</td><td>26.78</td></tr><tr><td>Degree-aware + 1-swap</td><td>0.0754</td><td>0.3963</td><td>26.78</td></tr><tr><td>Global MILP</td><td>0.0754</td><td>0.4027</td><td>26.78</td></tr></table>

Table 2: Patient-cluster bootstrap (2,000 resamples over SUBJECT\_ID) on the validation and test splits. All intervals exclude zero on both splits.
<table><tr><td>Target DDI</td><td>Global-1-swap</td><td>95% CI</td><td>Global-Degree</td><td>95% CI</td></tr><tr><td colspan="5">Validation split</td></tr><tr><td>0.136</td><td>+0.0020</td><td>[+0.0011,+0.0030]</td><td>+0.0047</td><td>[+0.0036,+0.0059]</td></tr><tr><td>0.100</td><td>+0.0050</td><td>[+0.0037,+0.0064]</td><td>+0.0083</td><td>[+0.0068,+0.0098]</td></tr><tr><td>0.080</td><td>+0.0061</td><td>[+0.0047,+0.0076]</td><td>+0.0089</td><td>[+0.0073,+0.0104]</td></tr><tr><td colspan="5">Test split</td></tr><tr><td>0.136</td><td>+0.0021</td><td>[+0.0010,+0.0031]</td><td>+0.0051</td><td>[+0.0040,+0.0062]</td></tr><tr><td>0.100</td><td>+0.0050</td><td>[+0.0036,+0.0063]</td><td>+0.0077</td><td>[+0.0062,+0.0091]</td></tr><tr><td>0.080</td><td>+0.0064</td><td>[+0.0049,+0.0078]</td><td>+0.0090</td><td>[+0.0074,+0.0106]</td></tr></table>

## 7 Results

## 7.1 Global decoding improves the accuracy–DDI tradeoff

Table 1 gives the main comparison on the test split. The ordering

$$
J _ { \mathrm { g l o b a l } } > J _ { \mathrm { 1 - s w a p } } > J _ { \mathrm { d e g r e e } } > J _ { \mathrm { g r e e d y } }\tag{12}
$$

holds at all three DDI regimes. At the strict target, global decoding reaches 0.4027 Jaccard, versus 0.3963 for degree-aware + 1-swap, 0.3937 for degree-aware alone, and 0.3790 for greedy. Average output size is 26.78 medications for every decoder at every budget, and the achieved DDI rate for global decoding is at or below the shared target throughout.

## 7.2 The advantage is statistically distinguishable on both validation and test

Table 2 reports patient-cluster bootstrap intervals against the two strongest baselines, on both splits. Every interval excludes zero. The advantage against degree-aware + 1-swap grows from +0.0021 (loose) to +0.0064 (strict) on test, closely matching +0.0020 to +0.0061 on the independently held-out validation split; against degree-aware alone it grows from +0.0051 to +0.0090 on test and +0.0047 to +0.0089 on validation.

## 7.3 The advantage holds across the accuracy–DDI frontier

Realized DDI rate is not always identical across decoders at a shared target, so we additionally evaluate a fine sweep of ten target rates from 0.14 to 0.02 and compare methods at matched realized DDI values via interpolation over the overlapping range of each method’s (achieved DDI, Jaccard) curve. Figure 2 shows the resulting frontier; Table 3 reports Jaccard at matched realized-DDI checkpoints against the strongest baseline. Global decoding is above degree-aware + 1-swap at every checkpoint.

Table 3: Jaccard at matched realized DDI rates within the overlapping range of the fine-sweep curves in Fig. 2.
<table><tr><td>Realized DDI</td><td>Degree-aware + 1-swap</td><td>Global MILP</td></tr><tr><td>0.12</td><td>0.4193</td><td>0.4210</td></tr><tr><td>0.10</td><td>0.4098</td><td>0.4139</td></tr><tr><td>0.08</td><td>0.3992</td><td>0.4050</td></tr><tr><td>0.06</td><td>0.3860</td><td>0.3911</td></tr></table>

![](images/a392cd67795441044daf76f8968e9219676940de55472d2061f776ff61a459f6.jpg)  
Figure 2: Accuracy–DDI frontier under identical frozen scores and fixed cardinality, from a fine sweep of ten target DDI rates on the test split. The global decoder’s curve remains above the three local baselines throughout, with separation widening toward stricter DDI constraints.

We additionally evaluate three score representations for the frozen scorer – raw log-margin, sigmoidtransformed probability $P ( \mathrm { Y e s } )$ , and rank-normalization – holding $K _ { q }$ fixed. Greedy and degreeaware repair depend only on score order and are therefore invariant to strictly monotone rescoring. In contrast, both the 1-swap local search and the MILP use score differences or sums and can depend on score spacing. Global decoding beats degree-aware + 1-swap in all nine score-by-budget combinations tested; detailed results are reported in Appendix B.

The frontier is not perfectly monotonic point-to-point (Fig. 2), so we do not claim a strict relationship between budget tightness and gap size at every adjacent pair of points. The pattern across the coarse and fine sweeps is that global decoding provides a small advantage when the DDI constraint is permissive and a larger advantage once the constraint tightens enough that conflicting choices become coupled.

## 7.4 Global decoding preserves more true positives and removes more false positives

At the strict target, global decoding retains more true-positive medications than the local baselines while also removing more false positives, rather than trading one error type for the other. The optimized score sum is ordered in the same direction as Jaccard; the full decomposition is reported in Appendix B.

## 8 Discussion

DDI-constrained medication recommendation is also a decoding problem. The dominant research question in medication recommendation has been how to learn better representations and score functions. The MILP formulation itself is standard; our contribution is the controlled empirical isolation of decoder choice. Our results show that, with the scorer completely frozen, the final mapping from scores to a constrained set materially changes the accuracy–DDI frontier. A model can assign useful drug-level scores while a local DDI-repair procedure discards that information through myopic decisions.

Comparison with local search. Degree-aware repair with 1-swap local search substantially improves over greedy repair and operates at nearly the same realized DDI exposure as the global decoder at the three primary targets. Exact decoding retains an advantage of +0.0021 to +0.0064 Jaccard over this baseline across the three operating regimes on the test split, with bootstrap intervals excluding zero and the same pattern present on validation, across the fine budget sweep, and under three score representations.

Why the gap widens under tighter budgets. When the budget is permissive, many high-score sets are feasible and local commitments have limited downstream cost. When the budget is strict, the feasible region contracts and the cost of keeping one conflict-heavy drug rises. The mechanism analysis is consistent with this: globally optimized sets remove more false positives while retaining more true positives.

Modularity. The decoder assumes only patient-specific medication scores, a target cardinality, and a pairwise conflict graph, independent of how the scores were learned. Foundation models are ultimately converted from candidate-level scores into coupled medication sets, and DDI constraints can live in this inference layer as well as in model training. FLAME supplies the clinical scores; our contribution studies DDI-constrained action decoding on top of a frozen model. We instantiate one frozen scorer to keep the empirical claim narrow: decoder choice matters under controlled scores and constraints, not that this particular scorer is universally best.

Clinical interpretation of the DDI budget. The DDI graph used here is a benchmark signal, not a complete clinical decision rule, and we do not claim the resulting sets are clinically safe in an absolute sense. Our empirical claim is narrower: global decoding improves medication-set agreement at matched DDI exposure under the same TWOSIDES-derived binary interaction constraint. Interaction severity, dosage, timing, indication, allergy history, organ function, and outcome data are not represented in this constraint, and clinician oversight remains necessary. The three budgets are controlled operating regimes for studying algorithmic behavior, not prescriptions for a clinically acceptable DDI rate.

## 9 Limitations

Our claims are intentionally scoped to one EHR benchmark, one frozen scorer, and a binary TWOSIDES-derived DDI graph [Tatonetti et al., 2012]. The fixed-cardinality design isolates decoder choice but does not optimize regimen size, and the MILP inherits any systematic bias in the underlying scores. The DDI constraint is a benchmark safety signal rather than a complete clinical rule because it omits interaction severity and patient-specific context. Accordingly, the present evaluation does not establish the clinical safety or clinical utility of the resulting recommended medication sets. Broader validation across datasets, scorers, patient populations, and richer clinically informed constraints would be needed before evaluating the approach for potential clinical deployment. Runtime and scaling under larger vocabularies or richer constraints, as well as comparisons with stronger multi-swap or beam-search baselines, remain important directions for future evaluation.

## 10 Conclusion

Medication recommendation is commonly treated as a representation-learning problem followed by a relatively simple set-construction step. We isolate that final step and show that it matters. With the scorer, output cardinality, and DDI budget held fixed, in this benchmark evaluation, exact global decoding achieves higher Jaccard than increasingly sophisticated local repair on validation and test, at matched realized DDI exposure and across multiple score parameterizations. These findings are limited to benchmark medication recommendations under the evaluated DDI constraint and do not establish clinical safety or utility; they do not prescribe how medication decisions should generally be made. Within this evaluation, when DDI constraints couple medication choices, the final set can be treated as a joint constrained decision rather than a sequence of local repairs.

## References

Zafar Ali, Yi Huang, Irfan Ullah, Junlan Feng, Chao Deng, Nimbeshaho Thierry, Asad Khan, Asim Ullah Jan, Xiaoli Shen, Rui Wu, and Guilin Qi. Deep learning for medication recommendation: A systematic survey. Data Intelligence, 5(2):303–354, 2023. doi: 10.1162/dint\_a\_00197.

Qianyu Chen, Xin Li, Kunnan Geng, and Mingzhong Wang. Context-aware safe medication recommendations with molecular graph and ddi graph embedding. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 7053–7060, 2023. doi: 10.1609/aaai.v37i6.25861.

Chenxiao Fan, Chongming Gao, Wentao Shi, Yaxin Gong, Zihao Zhao, and Fuli Feng. Fine-grained list-wise alignment for generative medication recommendation. In Advances in Neural Information Processing Systems, volume 38, pages 53569–53593, 2025. doi: 10.52202/085713-1602.

Qi Huangfu and J. A. Julian Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1):119–142, 2018. doi: 10.1007/s12532-017-0130-5.

Chaeyoung Huh, Hyunmin Hwang, Jung Hwan Shin, Jinse Park, and Jong Chul Ye. Pace-rag: Patient-aware contextual and evidence-based policy rag for clinical drug recommendation. arXiv preprint arXiv:2603.17356, 2026.

Alistair E. W. Johnson, Tom J. Pollard, Lu Shen, Li-wei H. Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G. Mark. Mimic-iii, a freely accessible critical care database. Scientific Data, 3:160035, 2016. doi: 10.1038/sdata.2016. 35.

Taeri Kim, Jiho Heo, Hongil Kim, Kijung Shin, and Sang-Wook Kim. Vita: “carefully chosen and weighted less” is better in medication recommendation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 8600–8607, 2024. doi: 10.1609/aaai.v38i8.28704.

Hung Le, Truyen Tran, and Svetha Venkatesh. Dual memory neural computer for asynchronous twoview sequential learning. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1637–1645, 2018. doi: 10.1145/3219819.3219981.

Xiang Li, Shunpan Liang, Yu Lei, Chen Li, Yulei Hou, Dashun Zheng, and Tengfei Ma. Causalmed: Causality-based personalized medication recommendation centered on patient health state. In Proceedings ofthe 33rd ACM International Conference on Information and Knowledge Management, pages 1276–1285, 2024. doi: 10.1145/3627673.3679542.

Yishuo Li, Qi Zhang, Wenpeng Lu, Xueping Peng, Weiyu Zhang, Jiasheng Si, Yongshun Gong, and Liang Hu. Time-aware medication recommendation via intervention of dynamic treatment regimes. In Proceedings of the ACM on Web Conference 2025, pages 5161–5172, 2025. doi: 10.1145/3696410.3714533.

Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Zijian Zhang, Feng Tian, and Yefeng Zheng. Large language model distilling medication recommendation model. arXiv preprint arXiv:2402.02803, 2024.

Hang Lv, Zixuan Guo, Yanchao Tan, Wanzi Shao, Hengyu Zhang, and Carl Yang. Awaken the giant: Activating llms via deep model guidance for boundary-aware medication recommendation. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1030–1041, 2026. doi: 10.1145/3770854.3780297.

George L. Nemhauser and Laurence A. Wolsey. Integer and Combinatorial Optimization. Wiley, 1988.

Junyuan Shang, Tengfei Ma, Cao Xiao, and Jimeng Sun. Pre-training of graph augmented transformers for medication recommendation. In Proceedings ofthe Twenty-Eighth International Joint Conference on Artificial Intelligence, pages 5953–5959, 2019a. doi: 10.24963/ijcai.2019/825.

Junyuan Shang, Cao Xiao, Tengfei Ma, Hongyan Li, and Jimeng Sun. Gamenet: Graph augmented memory networks for recommending medication combination. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 1126–1133, 2019b. doi: 10.1609/aaai. v33i01.33011126.

Jie Tan, Yu Rong, Kangfei Zhao, Tian Bian, Tingyang Xu, Junzhou Huang, Hong Cheng, and Helen Meng. Natural language-assisted multi-modal medication recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 2200–2209, 2024. doi: 10.1145/3627673.3679529.

Yanchao Tan, Chengjun Kong, Leisheng Yu, Pan Li, Chaochao Chen, Xiaolin Zheng, Vicki S. Hertzberg, and Carl Yang. 4sdrug: Symptom-based set-to-set small and safe drug recommendation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3970–3980, 2022. doi: 10.1145/3534678.3539089.

Nicholas P. Tatonetti, Patrick P. Ye, Roxana Daneshjou, and Russ B. Altman. Data-driven prediction of drug effects and interactions. Science Translational Medicine, 4(125):125ra31, 2012. doi: 10.1126/scitranslmed.3003377.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, et al. Scipy 1.0: Fundamental algorithms for scientific computing in python. Nature Methods, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2.

Feiyue Wu, Tianxing Wu, and Shenqi Jing. Armr: Adaptively responsive network for medication recommendation. In Proceedings ofthe Thirty-Fourth International Joint Conference on Artificial Intelligence, pages 7831–7839, 2025. doi: 10.24963/ijcai.2025/871.

Jialun Wu, Yuxin Dong, Zeyu Gao, Tieliang Gong, and Chen Li. Dual attention and patient similarity network for drug recommendation. Bioinformatics, 39(1):btad003, 2023. doi: 10.1093/ bioinformatics/btad003.

Rui Wu, Zhaopeng Qiu, Jiacheng Jiang, Guilin Qi, and Xian Wu. Conditional generation net for medication recommendation. In Proceedings of the ACM Web Conference 2022, pages 935–945, 2022. doi: 10.1145/3485447.3511936.

Chaoqi Yang, Cao Xiao, Lucas Glass, and Jimeng Sun. Change matters: Medication change prediction with recurrent residual networks. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence, pages 3728–3734, 2021a. doi: 10.24963/ijcai.2021/513.

Chaoqi Yang, Cao Xiao, Fenglong Ma, Lucas Glass, and Jimeng Sun. Safedrug: Dual molecular graph encoders for recommending effective and safe drug combinations. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, pages 3735–3741, 2021b. doi: 10.24963/ijcai.2021/514.

Nianzu Yang, Kaipeng Zeng, Qitian Wu, and Junchi Yan. Molerec: Combinatorial drug recommendation with substructure-aware molecular representation learning. In Proceedings of the ACM Web Conference 2023, pages 4075–4085, 2023. doi: 10.1145/3543507.3583872.

Yutao Zhang, Robert Chen, Jie Tang, Walter F. Stewart, and Jimeng Sun. Leap: Learning to prescribe effective and safe treatment combinations for multimorbidity. In Proceedings ofthe 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1315–1324, 2017. doi: 10.1145/3097983.3098109.

Zihao Zhao, Yi Jing, Fuli Feng, Jiancan Wu, Chongming Gao, and Xiangnan He. Leave no patient behind: Enhancing medication recommendation for rare disease patients. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 533–542, 2024. doi: 10.1145/3626772.3657785.

Zihao Zhao, Chenxiao Fan, Chongming Gao, Fuli Feng, and Xiangnan He. Addressing overprescribing challenges: Fine-tuning large language models for medication recommendation tasks. arXiv preprint arXiv:2503.03687, 2025.

## A Implementation details

Data splits. The training, validation, and test splits (4:1:1 ratio, approximately 14,000 visits in total) share no SUBJECT\_ID across splits. The medication vocabulary $( M = \mathrm { 1 5 1 } )$ and DDI graph $( | E _ { D } | = 1 , 4 2 5 )$ are shared across all splits and taken from FLAME’s released MIMIC-III preprocessing [Fan et al., 2025]. We do not use FLAME’s optional clinical-note augmentation step (data4LLM\_with\_note.csv, which requires GPT API access to extract note-derived features); our scores are computed from structured MIMIC-III fields only.

Scorer fitting. The cardinality-defining scorer is a one-dimensional logistic regression mapping the frozen classifier’s confidence margin to a calibrated probability, fit by maximum likelihood on an 80% slice of the training split (scikit-learn, LogisticRegression, L penalty, $C = 1$ max $\mathtt { . i t e r = l 0 0 0 } )$ . The decision threshold is selected on the remaining 20% slice of training by maximizing mean Jaccard over a grid $\{ 0 . 0 5 , 0 . 0 7 , \hdots , 0 . 9 3 \}$ . The selected threshold (0.310) and fitted coefficients are frozen and applied unchanged to the validation and test splits.

MILP solver settings. We use scipy.optimize.milp with the HiGHS backend [Virtanen et al., 2020, Huangfu and Hall, 2018] and default branching and cutting-plane settings. Every reported instance reached proven global optimality. Edge variables $y _ { i j }$ are declared continuous on [0, 1] rather than binary, since the linearization constraints $( 8 ) \AA { - } ( 9 )$ force them to $\{ 0 , 1 \}$ given binary x; this substantially reduces the number of integer variables relative to declaring $y _ { i j }$ binary, with no change to the optimal solution.

1-swap local search. Given a starting feasible set S (the degree-aware solution) and a DDI budget $B _ { q } ,$ , we maintain the per-medication conflict degree against the current set, $\begin{array} { r } { \deg _ { S } ( d ) = \sum _ { s \in S } \bar { D _ { d s } } , } \end{array}$ as a length-151 vector, updated incrementally after each swap rather than recomputed from scratch. At each iteration we evaluate, for every pair $( i \in S , j \notin \bar { S } )$ , the resulting conflict count $C ^ { \prime } =$ $C - \deg _ { S } ( i ) + \deg _ { S } ( j ) - D _ { i j }$ and score change $s _ { j } - s _ { i } ,$ select the feasible swap $( C ^ { \prime } \leq B _ { q } )$ with the largest score improvement, apply it, and update deg . The loop terminates when no feasible improving swap exists or after 200 iterations (never reached in practice).

Score representations (Section 7.3). Raw log-margin is the classifier’s native output $z _ { i } ( q )$ . Sigmoid probability is $P ( \mathrm { Y e s } \mid q , m _ { i } ) = 1 / ( 1 + e ^ { - z _ { i } ( q ) } )$ . Rank-normalization replaces $z _ { i } ( q )$ with $( 1 5 0 -$ rank $) / 1 5 0$ , where rank $\in \{ 0 , \ldots , 1 5 0 \}$ is the 0-indexed descending rank of drug i under $z ( q )$ for that visit; this preserves the descending medication order while changing the relative spacing between scores. Greedy and degree-aware repair depend only on this order, whereas both 1-swap local search and the MILP can depend on the score spacing.

## B Additional robustness and mechanism details

The main text summarizes two secondary analyses that support the central decoding result. Table 4 reports the score-representation sensitivity analysis, and Table 5 gives the strict-budget error decomposition.

Table 4: Global decoding vs. degree-aware + 1-swap under three score representations, at each DDI regime.
<table><tr><td>Score representation</td><td>Loose (0.136)</td><td>Moderate (0.100)</td><td>Strict (0.080)</td></tr><tr><td>Raw log-margin</td><td>Global√</td><td>Global√</td><td>Global√</td></tr><tr><td>Sigmoid  $P ( \mathrm { Y e s } )$ </td><td>Global √</td><td>Global√</td><td>Global√</td></tr><tr><td>Rank-normalized</td><td>Global√</td><td>Global√</td><td>Global√</td></tr></table>

Table 5: Mechanism analysis at target DDI 0.08, relative to the original pre-repair top- $K _ { q }$ set. Values are per-patient averages.
<table><tr><td>Decoder</td><td>TP retained</td><td>TP removed</td><td>FP removed</td><td>TP repl.</td><td>Score sum</td></tr><tr><td>Greedy</td><td>13.175</td><td>2.042</td><td>1.829</td><td>0.810</td><td>14.6425</td></tr><tr><td>Degree-aware</td><td>13.717</td><td>1.500</td><td>1.385</td><td>0.693</td><td>15.0374</td></tr><tr><td>Degree-aware + 1-swap</td><td>13.790</td><td>1.427</td><td>1.458</td><td>0.693</td><td>15.1224</td></tr><tr><td>Global MILP</td><td>13.858</td><td>1.359</td><td>2.049</td><td>0.799</td><td>15.3485</td></tr></table>

## C Additional bootstrap visualization

Figure 3 summarizes the patient-cluster bootstrap advantage over each local baseline on the test split.

![](images/441cfaac3b9b20c914b1c6680e507533c5ad6adef06479a9af8ccc5c68019a06.jpg)  
Figure 3: Global decoding’s advantage over each baseline as the DDI budget tightens, with patientcluster bootstrap 95% confidence intervals on the test split.