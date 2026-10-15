# Learning Individual Scientific Taste with Privileged Hindsight Distillation

Hanwei Fan<sup>1,2</sup> Yanqiao Wang<sup>1,3,4</sup> Yiqi Zhu<sup>1,2</sup> Xuanyu Lei<sup>1,2</sup> Peng Li<sup>1,∗</sup> Yang Liu<sup>1,2,∗</sup>

<sup>1</sup>Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China

<sup>2</sup>Department of Computer Science and Technology, Tsinghua University, Beijing, China

<sup>3</sup>Yau Mathematical Sciences Center, Tsinghua University, Beijing, China

<sup>4</sup>Qiuzhen College, Tsinghua University, Beijing, China

Abstract Scientific taste shapes which research directions researchers pursue, and it varies across individuals. Existing work evaluates research directions along multiple evaluation criteria or models scientific taste using community feedback signals such as citations and reviews, but neither directly captures how individuals translate assessments under the same criteria into diferent preferences. We model scientific taste as a researcher-specific mapping from criterion-level assessments in a shared, evidence-grounded evaluation space with proposal-visible evidence, to relative preferences. A population-level mapping learned from retrospective supervision provides a cold start when individual feedback is scarce. We introduce Privileged Hindsight Distillation (PHD), which uses later-emerging scientific evidence during training to improve proposal-time criterion assessment and the population mapping, while exposing only proposal-visible information at inference. We evaluate our approach primarily in quantum research, where structured public reviews are largely unavailable, and use pairwise proposal comparison as a direct interface for observing individual taste. With only 15 comparisons per researcher, our method captures sharply diferent tastes, including opposing preferences over the same research-direction pairs when they trade of across evaluation criteria. In an interactive study, personalized rankings also surface proposals associated with greater subsequent user engagement, connecting learned taste to downstream research exploration.

## 1 Introduction

Scientific taste broadly refers to the judgment that shapes what researchers choose to work on, how they approach scientific problems, and how they recognize the potential impact of research ideas and findings [1–3]. As large language models (LLMs) increasingly support scientific research from hypothesis and proposal generation to experimental design, execution, and paper drafting [4–7], generating research directions becomes more scalable, making it increasingly important to decide which are worth pursuing before their outcomes are known.

Scientific taste, however, varies across researchers. Researchers difer in which problems they consider important and which approaches they prefer [1], and substantial disagreement persists even when domain experts evaluate the same scientific work [8, 9]. Learning individual taste is particularly challenging when direct feedback is scarce. Recent work evaluates research directions along criteria such as novelty, significance, and feasibility [10, 11], but criterion-level assessments alone do not specify how a particular researcher translates them into preferences. Other work learns scientific judgment at scale from ecosystem signals such as citations, public reviews, and publication outcomes [3, 12–14]. When aggregated, these signals provide population-level supervision rather than the taste of any particular individual.

To model this variation, we separate shared evaluation from individual preference. At the proposal stage, each research direction is represented in a shared evaluation space over criteria such as novelty and generality, with assessments grounded in proposal claims and relevant literature [15]. This common representation supports comparison across directions. Population-level learning can be instantiated in this shared space to map criterion-leve assessments to aggregate preferences, while individual taste is modeled as a researcher-specific mapping from criterion-level assessments and proposal-visible evidence to relative preferences (Figure 1). The same shared evaluation can therefore support diferent choices: one researcher may favor conceptual novelty, while another prioritizes broad applicability. Although these criteria are neither fully objective nor exhaustive of scientific judgment, they provide a common basis for modeling variation across researchers.

![](images/23c47c8e35feb89e961c25ee38a9dabd3ac9d447bbd2a94912b0556ebdaec164.jpg)  
Figure 1: Modeling scientific taste. Proposal-stage evidence places research directions in a shared evaluation space, while researchers may map the same criterion profiles to diferent preferences. Later-emerging evidence can provide retrospective supervision for population-level learning but is unavailable at proposal-time inference.

A separate challenge is that most useful supervision signals emerge after research unfolds. Outcomes, community reviews, follow-up work and citations can reveal how a direction is received, but such later-emerging signals are unavailable when directions must be judged at the proposal stage. Their availability is also uneven across fields: computer science venues such as ICLR expose structured public reviews through OpenReview [16], while comparable records are more limited in quantum research, where platforms such as SciRate [17] provide weaker community signals (scites). When aggregated, these later signals can still support learning a population-level taste mapping, which provides a useful cold start for personalization under sparse direct feedback. We therefore use this mapping as a population-level prior for personalization, not assuming that it represents a unique community-wide taste.

To make such retrospective supervision usable at proposal time, we introduce Privileged Hindsight Distillation (PHD). In PHD, a privileged teacher uses later-emerging evidence to refine the rubrics, conditional rules, and memory of a proposal-only student, improving both shared evaluation and population-level taste learning while preserving proposal-only inference. The resulting population mapping is then adapted to individual researchers using sparse direct feedback.

We observe individual scientific taste through pairwise proposal comparison, a lightweight interface for eliciting relative preferences without requiring absolute scores [18–20]. We evaluate our approach primarily in quantum research, with additional cross-domain experiments using real and simulated researchers. Our results show that PHD improves proposal-time evaluation and population-level prior learning, while sparse feedback enables adaptation to individual researchers. We further observe substantial disagreement across researchers, particularly when competing directions trade of across evaluation criteria, and deploy the complete framework in an interactive proposal exploration system, where personalized rankings more efectively surface proposals that users subsequently choose to inspect, refine, or pursue.

Contributions. (1) We formulate scientific taste by separating criterion-level assessment from how those assessments are valued. A shared, evidence-grounded evaluation space supports population and researcher-specific mappings to relative preferences, with the former serving as a transferable prior for individual adaptation. (2) We introduce Privileged Hindsight Distillation (PHD), which uses later-emerging scientific evidence as privileged supervision to learn criterion-level assessment and the population prior while preserving proposal-only inference. (3) We show that individual scientific taste can diverge substantially from population-level judgment. Researchers exhibit sharply diferent preferences on the same research-direction pairs, and researcher-specific conditional mappings recover conflicting preferences that fixed criterion reweighting misses under sparse feedback.

## 2 Related Work

Modeling Scientific Taste. Prior work measures research taste through citation-based impact [12], or learns scientific judgment from citation comparisons [3], institutional publication decisions [13], and future-impact supervision for peer review [14]. More recent work has moved scientific judgment toward earlier research stages: TASTE [20] benchmarks expert pairwise preferences of AI-Safety research proposals, while other studies compare human and LLM-generated ideas using a research-taste taxonomy [21] or explicitly encode researcher-specific preferences [22]. These studies examine aggregate judgment, proposal-level preferences, or diferences in ideation patterns, whereas our focus is learning researcher-specific preferences from sparse feedback.

Research Artifact Evaluation. The rapid development of scientific ideation systems [4, 7, 23–25] has made the evaluation of ideas, proposals, and papers increasingly important [26, 27]. Existing benchmarks assess research artifacts along criteria such as novelty, feasibility, and potential impact [10, 28–30], yet holistic LLM judgments do not always predict the quality of executed research [31] and show particular limitations in novelty assessment [32]. Evidence-grounded methods such as OpenNovelty [15] instead support assessment with retrieved literature and explicit evidence. Our work builds on this direction by grounding proposal-time criterion assessment in proposal claims and relevant literature, and organizing these assessments within a shared evaluation space.

Privileged Information Learning. Learning using privileged information formalizes supervision that is available during training but absent at deployment [33, 34]. Recent language-model methods similarly distill privileged information into deployment-restricted policies [35, 36]; OPSD in particular lets a privileged teacher guide an unprivileged student on its own rollouts. Reflection and textual optimization methods improve agentic artifacts like prompts and LM programs [37–39], reusable experience [40], and episodic memory [41]. Our proposed PHD connects these two categories by exposing retrospective scientific evidence only to the teacher while distilling its benefit into a proposal-time agentic workflow.

Sparse Preference Learning. Personalized preference learning studies how user-specific judgments can be recovered from sparse supervision, including few-shot adaptation and population-informed preference learning [42–44]. Relevant work models heterogeneous preferences through structured preference representations [45, 46], while perspectivist learning preserves annotator-specific disagreement rather than collapsing judgments into a single consensus [47–50]. When feedback arrives sequentially, continual-learning methods commonly use replay to incorporate new observations while mitigating forgetting [51, 52]. Our setting brings these two problems together by learning individual scientific taste from sparse, sequential researcher feedback.

## 3 Method

Our framework learns individual scientific taste from sparse feedback through three components, as illustrated in Figure 2(a). Evaluation Space Learning maps each research direction to a researcher-independent representation consisting of criterion-level assessments and associated proposal-visible evidence. Population Taste Distillation learns a population-level mapping from this shared representation to relative preferences. Individual Taste Adaptation then adapts the population mapping into a researcher-specific taste mapping using sparse sequential feedback. The first two components are improved through Privileged Hindsight Distillation (PHD), a shared training procedure described in Section 3.2.

## 3.1 Shared Evaluation Space and Scientific Taste

Let C be a finite set of candidate research directions, with $P \in C$ representing one direction through its proposal. All proposal-stage information, including the proposal and literature available at that time, is denoted by ${ \cal { T } } _ { \mathrm { v i s } } ( P )$ with $\mathcal { E } ( P )$ representing the corresponding proposal-visible evidence. Both depend only on $P ,$ not on the researcher evaluating it. We define a shared �-dimensional evaluation space S, with one dimension per evaluation criterion. A criterion assessor maps proposal-visible information to

$$
{ \hat { s } } ( P ) = f _ { \mathrm { c r i t } } ( { \mathcal { T } } _ { \mathrm { v i s } } ( P ) ) = ( { \hat { s } } _ { 1 } ( P ) , \ldots , { \hat { s } } _ { K } ( P ) ) \in S ,
$$

where each ${ \hat { s } } _ { k } ( P )$ is researcher-independent. Researchers therefore share the same criterion representation but may translate it and its supporting evidence into diferent preferences.

For researcher $u ,$ scientific taste is a researcher-specific mapping from criterion assessments and proposal-visible evidence to relative preferences,

$$
T _ { u } : \{ ( \hat { s } ( P ) , \mathcal { E } ( P ) ) \} _ { P \in C } \longmapsto \succ _ { u } ,
$$

![](images/71acbd4099c5867d85c7edc91056c90fba15dd803ad4e6201d83c20548dc9cdc.jpg)  
Figure 2: Overview of our framework. (a) The framework learns a shared evaluation space, distills a population-level taste mapping, and sequentially adapts it to individual researchers; PHD refines the task-specific artifacts used by the first two components. (b) PHD iteratively identifies student failures, uses privileged hindsight evidence to diagnose them and revise the artifacts, and retains only revisions that pass leakage screening and held-out validation. At inference, the student uses only proposal-visible information and the final accepted artifacts.

where $A \succ _ { u }$ � means that � prefers � to �. We observe scientific taste through pairwise comparisons. For �, $B \in C$ let $x _ { A , B } = { \hat { s } } ( A ) - { \hat { s } } ( B )$ denote their criterion-diference vector and $\mathcal { E } _ { A , B } = ( \mathcal { E } ( A ) , \mathcal { E } ( B ) )$ their proposal-visible evidence. We instantiate $T _ { u }$ through a scalar preference function $U _ { u }$ , such that

$$
A \succ _ { u } B \Longleftrightarrow U _ { u } ( x _ { A , B } , { \mathcal { E } } _ { A , B } ) > 0 .
$$

Section 3.4 defines the shared functional form of �, and Section 3.5 personalizes it from researcher feedback.

## 3.2 Privileged Hindsight Distillation

Scientific judgments must be made at the proposal stage, before completed methods, experimental results, subsequent literature, citations, and community feedback become available. Once research unfolds, these later observations provide additional evidence for diagnosing errors in proposal-time judgments. We denote such later-emerging information by ${ \cal I } _ { \mathrm { p r i v } }$ and introduce Privileged Hindsight Distillation (PHD) to exploit it during training while preserving proposal-only inference.

Inspired by On-Policy Self-Distillation (OPSD) [36], we adapt its iterative self-improvement principle to an agentic workflow. With the base model frozen, PHD transfers hindsight-derived knowledge by refining the external artifacts that govern the student instead of updating model parameters. We denote the artifact state at refinement round � by $\Psi _ { t } ,$ comprising a task-specific subset of rubric, conditional rules, and memory. Rubrics specify criteria, scoring anchors, and evidence requirements; conditional rules encode context-dependent adjustments; and memory stores reusable proposal-visible guidance.

As illustrated in Figure 2(b), PHD provides a shared four-stage refinement procedure for Evaluation Space Learning and Population Taste Distillation. For either task, let $z$ denote a training input and $y ^ { * } ( z )$ its training-only reference target; their specific forms are defined in the following sections.

1. Student rollout. Given proposal-visible information and the current artifacts $\Psi _ { t } .$ , the student produces a rollout $\tau _ { t } ( z )$ containing the task prediction $\hat { y } _ { t } ( z )$ , confidence, and proposal-visible supporting evidence. Neither $y ^ { * } ( z )$ nor ${ \cal T } _ { \mathrm { p r i v } }$ is available to the student.

2. Failure identification. The analyzer compares $\hat { y } _ { t } ( z )$ with the reference target $y ^ { * } ( z )$ using the task-specific error criterion and collects failed cases into $\mathcal { F } _ { t }$ . Each failed case contains �, its rollout $\tau _ { t } ( z )$ , and $y ^ { * } ( z )$ . At this stage, the analyzer only identifies prediction failures; it does not access the privileged hindsight evidence used to construct $y ^ { * } ( z )$ or diagnose their causes.

3. Teacher revision. The training-only teacher receives $\Psi _ { t } , \mathcal { F } _ { t }$ , and ${ \cal T } _ { \mathrm { p r i v } }$ . At this stage, privileged hindsight is used to diagnose the identified failures and propose a candidate revision $\Psi _ { t } ^ { \prime } .$ . For example, hindsight may reveal that a high novelty score was driven mainly by unsupported novelty claims in the proposal. The teacher may then tighten the corresponding evidence requirement in the rubric, introduce a conditional rule for such cases, or retain

the resulting proposal-visible guidance in memory.

4. Screening and validation. The candidate revision first undergoes leakage screening, which rejects artifacts containing privileged outcomes, reference answers, or conditions unavailable at the proposal stage. Revisions that pass screening are evaluated on held-out data using the task-specific validation metric and are accepted as $\Psi _ { t + 1 }$ when they satisfy the corresponding acceptance criterion.

Accepted artifacts are reused in the next refinement round. At inference, only the final accepted artifacts and proposal-visible information remain; reference targets, the analyzer, teacher, and privileged hindsight are removed. Sections 3.3 and 3.4 specify the task input, reference target, editable artifacts, failure criterion, and validation metric for the two shared components.

## 3.3 Evaluation Space Learning

Evaluation Space Learning instantiates the criterion assessor $f _ { \mathrm { c r i t } }$ from Section 3.1. We use a shared $K = 6 \mathrm { - }$ dimensional space covering generality, novelty, importance, parsimony, depth, and algorithmic complexity. Following rubric-based evaluation [53] and evidence-grounded scientific assessment [15], we construct two rubrics over the same criteria and score scales but with diferent information access.

Rubric A is the fixed reference rubric. It uses both proposal-visible information ${ \cal { T } } _ { \mathrm { v i s } } ( P )$ and privileged hindsight ${ \cal T } _ { \mathrm { p r i v } } ( P )$ to produce the reference criterion vector $s ^ { * } ( P ) = ( s _ { 1 } ^ { * } ( P ) , \ldots , s _ { K } ^ { * } ( P ) )$ . Rubric B is the student rubric and a component of Ψ. It uses only ${ \cal { T } } _ { \mathrm { v i s } } ( P )$ and governs the criterion assessor producing ${ \hat { s } } ( P )$ . The details of the two rubrics are provided in Appendices B.1 and B.2.

For PHD, the task instance is $z = P$ , the reference target is $y ^ { * } ( P ) = s ^ { * } ( P )$ , and the rollout $\tau _ { t } ( P )$ contains $\hat { s } _ { t } ( P )$ confidence, and proposal-visible supporting evidence. The analyzer computes the per-proposal mean absolute error (MAE) between $\hat { s } _ { t } ( P )$ and $s ^ { * } ( P )$ across the � criteria and uses it to identify high-error proposals for the failure set $\mathcal { F } _ { t }$ . The editable artifacts are Rubric B and case memory. When the assessor repeatedly mis-scores a criterion by at least one score level, we revise the corresponding scoring anchor in Rubric B. Isolated failures are instead stored as proposal-visible case lessons in memory, each with a triggering condition and an absolute-level check. Held-out MAE determines whether a revision is retained. Complete prompts and an illustrative revision are provided in Appendices H.1 and B.4.

## 3.4 Population Taste Distillation

For this PHD instantiation, the task input is a proposal pair $z = ( A , B )$ . The student observes the criterion-diference vector $x _ { A , B }$ and proposal-visible evidence $\mathcal { E } _ { A , B }$ defined in Section 3.1, and predicts $\hat { y } _ { t } ( A , B ) \in \{ A , B \}$ . The corresponding population reference target is $y ^ { * } ( A , B )$ , whose construction is detailed in Appendix C.

We model population and individual preferences using a common preference function. Each conditional rule � is evaluated independently on a research direction using only its proposal-visible representation. Let $a _ { r } ( P )$ denote its activation on $P ;$ for a pair $( A , B )$ , its contribution is $\phi _ { r } ( x _ { A , B } , \mathcal { E } _ { A , B } ) = a _ { r } ( A ) - a _ { r } ( B )$ . The preference function is

$$
U ( x , { \mathcal { E } } ; w , R ) = w ^ { \top } x + \sum _ { r \in R } { \tilde { \alpha } } _ { r } \phi _ { r } ( x , { \mathcal { E } } ) ,
$$

where � captures a fixed weighting over criterion diferences, � contains conditional rules, and $\tilde { \alpha } _ { r }$ is the fitted rule coeficient adjusted by held-out reliability. The linear term captures preferences expressible by fixed criterion reweighting, while the rules capture context-dependent judgments beyond it. This functional form is shared by the population and individual mappings.

We first fit $\boldsymbol { w } _ { \mathrm { p o p } }$ on population training pairs using $y ^ { * } ( A , B )$ as supervision and no conditional rules, yielding $U _ { \mathrm { p o p } } ^ { ( 0 ) } ( x ) = w _ { \mathrm { p o p } } ^ { \top } x$ . We then freeze $\boldsymbol { w } _ { \mathrm { p o p } }$ and apply PHD to learn context-dependent residual judgments. The editable artifacts are population rules $R _ { \mathrm { p o p } }$ and memory: rules encode reusable context-dependent adjustments, while memory stores proposal-visible guidance for applying them. These artifacts jointly govern the context-dependent component represented by $R _ { \mathrm { p o p } }$ in the preference function. The resulting population mapping is

$$
U _ { \mathrm { p o p } } ( x , \mathcal { E } ) = U ( x , \mathcal { E } ; w _ { \mathrm { p o p } } , R _ { \mathrm { p o p } } ) .
$$

StrictPair, defined in Section 4.1, identifies failed pairs for $\mathcal { F } _ { t }$ and serves as the held-out validation metric. The resulting $U _ { \mathrm { p o p } }$ provides the population prior for individual adaptation. Complete workflow and implementation details are provided in Appendix H.2.

## 3.5 Individual Taste Adaptation

Individual Taste Adaptation learns directly from sparse pairwise feedback from researcher $u ,$ stored in a time-ordered replay bufer $\mathcal { B } _ { u }$ . Each observation contains a proposal pair $( A , B )$ , its representation $x _ { A , B }$ and $\mathcal { E } _ { A , B }$ defined

in Section 3.1, and the researcher preference $y _ { u } ( A , B ) \in \{ A , B \}$ . Using the common preference function from Section 3.4, we personalize the population mapping through an individual residual $\Delta w _ { u }$ and researcher-specific rules $R _ { u } \colon$

$$
U _ { u } ( x , { \mathcal { E } } ) = U { \big ( } x , { \mathcal { E } } ; w _ { u } , R _ { \mathrm { p o p } } \cup R _ { u } { \big ) } ,
$$

where $w _ { u } = w _ { \mathrm { p o p } } + \Delta w _ { u } .$ . Adaptation starts from $U _ { \mathrm { p o p } }$ with $\Delta w _ { u } = 0$ and $R _ { u } = \varnothing .$

As feedback is appended to $\mathcal { B } _ { u } , \Delta w _ { u }$ is refitted using the frozen population score as an ofset, allowing criterion weights to deviate from the population prior. Recurring leave-one-out errors that remain after this linear adaptation motivate researcher-specific conditional rules. Each candidate rule is evaluated through cross-fitted replay over $\mathcal { B } _ { u } \mathrm { : }$ in each leave-one-out fold, $\Delta w _ { u }$ and the rule coeficient are refitted on the remaining observations and evaluated on the held-out support comparison. A rule is retained only if replay yields a net predictive improvement without breaking previously correct preferences. Held-out query labels and privileged hindsight are never used during individual adaptation. Further fitting and rule-induction details are provided in Appendix H.3.

## 4 Experiments

## 4.1 Learning Individual Scientific Taste

Data. Quantum serves as our primary evaluation domain. We reconstruct research directions as proposal-stage inputs containing the research problem, methodology, evaluation plan, expected contributions, and risks, while excluding later-emerging information. All compared methods receive only proposal-visible inputs; as with other pretrained LLM evaluations, we do not assume control over parametric knowledge acquired during pretraining.

Individual-taste supervision must be collected directly from researchers, and is therefore naturally scarce. Each of seven quantum-algorithm researchers provides 15 sequential support comparisons for adaptation and 10 held-out query pairs for evaluation, with a binary preference and confidence annotation for each comparison. We complement this real-researcher cohort with 100 simulated researchers: 50 derived from publication histories of prominent quantum researchers (Sim.-Q) and 50 generated through controlled variation in criterion trade-ofs (Sim.-Ctrl.). Further construction and supervision details are provided in Appendix E.

Evaluation metric. For pairwise preference prediction, we use StrictPair (SP), which requires both presentation orientations of a proposal pair to be predicted correctly. For the �-th pair $( A _ { i } , B _ { i } )$ ,

$$
\mathrm { S P } _ { i } = \mathbb { I } [ \hat { y } ( A _ { i } , B _ { i } ) = y ^ { * } ( A _ { i } , B _ { i } ) \ \wedge \ \hat { y } ( B _ { i } , A _ { i } ) = y ^ { * } ( B _ { i } , A _ { i } ) ] ,
$$

where $\hat { y }$ and $y ^ { * }$ denote the predicted and target preferences, respectively. For individual taste, we report Regret-Weighted StrictPair (RW-SP) to account for researcher confidence. Following regret-based pairwise weighting [54], comparison � receives weight $d _ { i } = 2 | p _ { i } - 0 . 5 |$ , where $p _ { i }$ is the preference probability derived from the researcher’s confidence annotation, detailed in Appendix F.3. We compute ${ \mathrm { R W - S P } } = ( \sum _ { i } d _ { i } \mathrm { S P } _ { i } ) / ( \sum _ { i } d _ { i } )$

Baselines. We compare personalization approaches with increasingly explicit use of researcher feedback. Unless otherwise specified, all methods use gpt-5.5 as the backbone, and no privileged hindsight is available during individual adaptation or inference. Single Call (SC) directly places all support comparisons in the prompt when predicting each held-out preference, while Profile Prompting (PP) first summarizes them into a researcher profile. SynthesizeMe [55], FERMI [56], and CIPHER [57] represent agentic approaches that maintain or refine user-specific descriptions or preferences from interaction history. VPL [42], LoRe [46], and FSPO [43] instead learn parameterized user-specific preference or reward representations. Detailed configurations are in Appendix F.

Results. Table 1 shows that representing the same sparse feedback through in-context or agentic personalization remains less efective than explicitly learning preference structure. Our method performs best on both real and simulated researchers while keeping the backbone frozen, indicating that explicit preference structure can be introduced through a transferable population mapping and lightweight researcher-specific state without per-user model-parameter updates.

## 4.2 Robustness and Sensitivity Analysis

Shared-component data. We further evaluate whether PHD successfully distills privileged hindsight in the two shared components that support individual taste learning. Both Evaluation Space Learning and Population Taste Distillation use carefully curated, paper-disjoint splits. Constructing these evaluations requires controlled proposal-time reconstruction and privileged reference construction, while population comparisons additionally require temporally comparable and informative proposal pairs. A proposal may participate in multiple comparisons within the same split, but no underlying paper is shared across splits.

Table 1: Individual scientific taste prediction. RW-SP across real and simulated researchers.
<table><tr><td></td><td colspan="2">In-context</td><td colspan="4">Agentic</td><td colspan="3">Parametric</td></tr><tr><td></td><td>SC</td><td>PP</td><td>SynthesizeMe</td><td>FERMI</td><td>CIPHER</td><td>Ours</td><td>VPL</td><td>LoRe</td><td>FSPO</td></tr><tr><td>Real ↑</td><td>0.479</td><td>0.481</td><td>0.477</td><td>0.419</td><td>0.524</td><td>0.693</td><td>0.629</td><td>0.595</td><td>0.642</td></tr><tr><td>Sim. ↑</td><td>0.423</td><td>0.519</td><td>0.473</td><td>0.366</td><td>0.399</td><td>0.591</td><td>0.556</td><td>0.547</td><td>0.559</td></tr></table>

Table 2: Robustness across backbones. E-MAE/E-OA denote MAE/Ordering Accuracy for Evaluation Space Learning, and P-SP denotes StrictPair for Population Taste Distillation. Shared-component entries report Ours / w/o PHD. For individual adaptation, we report RW-SP for our complete method and alternative baselines under the same backbone.
<table><tr><td></td><td colspan="2">Evaluation Space</td><td>Population</td><td colspan="3">Individual RW-SP ↑</td></tr><tr><td>Backbone</td><td>E-MAE↓</td><td>E-OA↑</td><td>P-SP ↑</td><td>Ours</td><td>PP</td><td>CIPHER</td></tr><tr><td>gpt-5.5</td><td>0.183 / 0.216</td><td>0.721 / 0.713</td><td>0.833 / 0.667</td><td>0.693</td><td>0.481</td><td>0.524</td></tr><tr><td>gpt-5.4-mini</td><td>0.233 / 0.292</td><td>0.665 / 0.604</td><td>0.667 / 0.567</td><td>0.613</td><td>0.407</td><td>0.332</td></tr><tr><td>claude-opus-5</td><td>0.217 / 0.275</td><td>0.675 / 0.612</td><td>0.500 / 0.367</td><td>0.664</td><td>0.634</td><td>0.516</td></tr><tr><td>gemini-3.6-flash</td><td>0.216 / 0.258</td><td>0.699 / 0.662</td><td>0.767 / 0.367</td><td>0.647</td><td>0.422</td><td>0.405</td></tr></table>

For Evaluation Space Learning, reference criterion vectors $s ^ { * } ( P )$ are constructed with the fixed Rubric A using both proposal-visible information and privileged hindsight evidence. For Population Taste Distillation, reference preferences $y ^ { * } ( A , B )$ are aggregated from a multi-family LLM-as-a-Judge panel given the same information. Since these are LLM-generated references rather than independent human labels, we use them to evaluate whether PHD can distill hindsight supervision into proposal-time judgments and for corresponding ablations, rather than for external benchmarking. Construction details are provided in Appendices E, B.1, and C

Evaluation metrics. For Evaluation Space Learning, we report mean absolute error (MAE) and Ordering Accuracy (OA) against the reference criterion vectors. Given a test set P, MAE is computed as

$$
\mathbf { M A E } = \frac { 1 } { | \mathcal { P } | K } \sum _ { P \in \mathcal { P } } \sum _ { k = 1 } ^ { K } \left| \hat { s } _ { k } ( P ) - s _ { k } ^ { * } ( P ) \right| .
$$

Ordering Accuracy measures whether relative criterion-level orderings are preserved. For each criterion $k ,$ a proposal pair is correct when the predicted and reference diferences have the same sign:

$$
\mathrm { O A } = \frac { 1 } { | Q | } \sum _ { ( A , B , k ) \in Q } \mathbb { I } \big [ \mathrm { s g n } ( \hat { s } _ { k } ( A ) - \hat { s } _ { k } ( B ) ) = \mathrm { s g n } \big ( s _ { k } ^ { * } ( A ) - s _ { k } ^ { * } ( B ) \big ) \big ] ,
$$

where Q contains evaluated proposal-pair–criterion combinations. For Population Taste Distillation, since each reference preference is aggregated across multiple LLM judges, we report SP described in Section 4.1 rather than RW-SP, as judge-specific confidence no longer provides a meaningful pair-level measure of preference strength.

Robustness across backbones. We examine whether the gains from PHD persist across backbone models by substituting the backbone across all three learning components. For Evaluation Space Learning and Population Taste Distillation, we compare the complete method with its w/o PHD counterpart under each backbone. For Individual Taste Adaptation, we compare our complete method with PP and CIPHER, the strongest baselines within their respective categories, using the same backbone. Table 2 shows that PHD improves criterion estimation and population preference prediction under all four backbones. At the individual level, our complete method also outperforms PP and CIPHER for every backbone, although absolute RW-SP varies across models. These results indicate that the observed gains are not specific to the default gpt-5.5 configuration.

Table 3: Downstream evaluation and sensitivity to population prior. (a) Each column compares our personalized ranker against the specified comparator. For simulated researchers, each entry reports Quantum | Cross-domain. Real-user deployment uses Single Call (SC), while simulated cohorts additionally include Profile Prompting (PP) and SynthesizeMe (SM). (b) Individual RW-SP after replacing the population prior while keeping the backbone, support feedback, and adaptation procedure fixed.  
(a) Downstream evaluation
<table><tr><td></td><td colspan="3">Simulated researchers</td><td>Real researchers</td></tr><tr><td>Metric</td><td>SC</td><td>PP</td><td>SM</td><td>SC</td></tr><tr><td>∆cp ↑</td><td>0.169 | 0.120</td><td>0.130 | 0.067</td><td>0.146 | 0.151</td><td>0.125</td></tr><tr><td>Win Rate ↑</td><td>0.667 | 0.583</td><td>0.571 | 0.577</td><td>0.641 | 0.635</td><td>0.684</td></tr><tr><td>∆NDCG ↑</td><td>0.064 | 0.006</td><td>0.020 | 0.018</td><td>0.037 | 0.044</td><td>0.023</td></tr></table>

(b) Population prior
<table><tr><td>Population Prior RW-SP ↑</td><td></td></tr><tr><td>Auto-J</td><td>0.583</td></tr><tr><td>SciJudge</td><td>0.644</td></tr><tr><td>MLR-Judge</td><td>0.667</td></tr><tr><td>Ours</td><td>0.693</td></tr></table>

Sensitivity to population prior. We isolate the role of population knowledge by fixing the gpt-5.5 backbone and replacing only the population prior, while keeping the support feedback and individual adaptation procedure unchanged. We use the same researcher cohorts and RW-SP protocol as Section 4.1. We consider three existing judges as alternative priors: Auto-J [58], a general-purpose pairwise judge; MLR-Judge [59], which evaluates research proposals through rubric-based assessment; and SciJudge [3], a scientific judge for comparing scientific work. As shown in Table 3(b), diferences between population priors persist after identical individual adaptation. Sparse individual feedback therefore does not fully overwrite the population initialization, indicating that the choice of population prior continues to afect downstream personalization.

## 4.3 Downstream Evaluation

Data and protocol. We deploy the complete framework in an interactive proposal-exploration system to test whether learned individual taste improves downstream research-direction discovery. We evaluate 20 real researchers from quantum, computer science, statistics, and medicine, with simulated cohorts evaluated either within quantum or under cross-domain transfer, where taste learned from quantum is tested on proposals from other fields. Unlike the evaluation in Section 4.1, accumulated interactions dynamically shape the candidate pool in subsequent rounds.

We evaluate both real and simulated users with diferent comparator sets. For real users, we compare against Single Call (SC) in a matched setting with the same backbone, candidate pool, proposal-visible information, and interaction history; SC directly conditions on the history, whereas ours updates an explicit individual taste mapping. Given the cost of real-user interaction, we restrict live comparison to this matched setting. In simulation, where additional comparisons require no further researcher interaction, we broaden baseline coverage with Profile Prompting (PP) and SynthesizeMe (SM), the strongest in-context and agentic baselines, respectively, in our simulated-user evaluation (Table 1). For each comparison, we evaluate our method against one alternative method at a time using source-blind competitive-pair interleaving [60, 61], a widely used protocol for online evaluation in ranking and recommendation systems. Proposals selected by both methods are shown once, while ranking disagreements form adjacent competitive pairs with randomized order and hidden method identity. Each proposal receives an engagement score $e ( P ) \in \{ 0 , 1 , 2 , 3 \}$ , denoting no substantive engagement, inspection, refinement, and adoption, respectively. Further system and evaluation details are provided in Appendix G.

Evaluation metrics. For the � ranking disagreements, we define Competitive-Pair Engagement Gain as $\begin{array} { r } { \Delta _ { \mathrm { C P } } = \sum _ { c = 1 } ^ { N _ { \mathrm { C P } } } ( e _ { c } ^ { \mathrm { o u r s } } - e _ { c } ^ { \mathrm { b a s e } } ) / N _ { \mathrm { C P } } } \end{array}$ , measuring the average engagement advantage within directly competing proposal pairs. We also report Win Rate, the fraction of non-tied pairs with $e _ { c } ^ { \mathrm { o u r s } } > e _ { c } ^ { \mathrm { b a s e } }$ . To evaluate the full ranking beyond local disagreements, we treat engagement as graded relevance and report $\Delta _ { \mathrm { N D C G } } = \mathrm { N D C G } _ { \mathrm { o u r s } } - \mathrm { N D C G } _ { \mathrm { b a s e } }$ [62].

Results. Table 3(a) shows that learned individual taste improves proposal discovery over diferent baselines as the candidate pool evolves through interaction. The gains persist under cross-domain transfer, though with varying magnitude, highlighting diferences in scientific taste across both individuals and research domains.

## 4.4 Role of Privileged Hindsight

To isolate privileged hindsight, we use the shared-component evaluation from Section 4.2 under gpt-5.4-mini, comparing w/o PHD, w/o $\begin{array} { r } { \mathcal { I } _ { \mathrm { p r i v } } . } \end{array}$ , and Ours. The w/o PHD variant removes the distillation procedure entirely, whereas w/o ${ \cal T } _ { \mathrm { p r i v } }$ retains the same refinement workflow but restricts the teacher to proposal-visible information. Table 4(a) shows that proposal-visible refinement substantially improves criterion-level assessment over w/o PHD, but does not improve population preference prediction. Adding privileged hindsight yields a smaller further gain in E-MAE and E-OA but a much larger improvement in P-SP. This contrast suggests that hindsight is particularly important for learning how criterion-level diferences translate into population preferences, not for criterion assessment itself.

Table 4: Ablation studies. (a) Efect of PHD and privileged hindsight on the two shared components under proposal-only inference. (b) Efect of individual residuals and conditional rules on RW-SP. The w/o PHD variant starts from the seed Rubric B and a randomly initialized population prior.  
(a) Privileged hindsigh
<table><tr><td>Variant</td><td>E-MAE↓</td><td>E-OA↑</td><td>P-SP ↑</td></tr><tr><td>w/o PHD</td><td>0.292</td><td>0.604</td><td>0.567</td></tr><tr><td>w/o Ipriv</td><td>0.242</td><td>0.649</td><td>0.533</td></tr><tr><td>Ours</td><td>0.233</td><td>0.665</td><td>0.667</td></tr></table>

(b) Individual adaptation
<table><tr><td>Variant</td><td>Real ↑</td><td>Sim.-Q↑</td><td>Sim.-Ctrl. ↑</td></tr><tr><td>Population</td><td>0.680</td><td>0.537</td><td>0.592</td></tr><tr><td>+ Residual</td><td>0.679</td><td>0.573</td><td>0.606</td></tr><tr><td>+ Rules</td><td>0.693</td><td>0.588</td><td>0.593</td></tr><tr><td>w/o PHD</td><td>0.670</td><td>0.540</td><td>0.581</td></tr></table>

Table 5: Replay under sequential feedback. Efect of replay-based rule acceptance on held-out RW-SP as support feedback accumulates. The ablation bypasses the cross-fitted replay gate for accepting personal conditional rules.
<table><tr><td>Support count</td><td>0</td><td>5</td><td>10</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>Ours</td><td>0.538</td><td>0.535</td><td>0.580</td><td>0.603</td><td>0.648</td><td>0.599</td><td>0.613</td></tr><tr><td>w/o replay</td><td>0.538</td><td>0.535</td><td>0.580</td><td>0.575</td><td>0.615</td><td>0.578</td><td>0.602</td></tr></table>

## 4.5 Understanding Individual Taste Learning

Beyond Population Initialization. We first examine the two individual adaptation components under the same cohorts and RW-SP protocol as Section 4.1. As shown in Table 4(b), for real researchers, the population initialization is already strong, and residual reweighting alone does not improve the aggregate result. Conditional rules yield a modest gain on real researchers and Sim.-Q, whereas residual adaptation is suficient on Sim.-Ctrl., whose preferences are explicitly constructed from criterion-level trade-ofs. This suggests that some researcher-specific preferences require context-dependent adaptation beyond fixed criterion reweighting, motivating analysis of when individual taste departs from the population prior rather than assuming uniform gains. A concrete example in Appendix D.2 shows two real users (U04 and U05) making opposing choices over the same proposal pairs despite identical frozen evaluation features, with the complete mapping recovering more of these conflicting preferences than residual-only adaptation. Removing PHD and adapting from the seed Rubric B and a randomly initialized prior further reduces RW-SP, suggesting that the PHD-derived shared structure provides a better starting point for personalization.

Replay under sequential feedback. We ablate the cross-fitted replay gate from Section 3.5 under the same sequential support protocol as Section 4.1, while keeping the population initialization, feedback sequence, residual fitting, rule proposer, and coeficient estimation fixed. Since this analysis repeatedly evaluates adaptation over multiple support prefixes, we use gpt-5.4-mini to reduce inference cost. Table 5 shows that replay has little efect early but becomes beneficial as researcher-specific preference emerges. Held-out RW-SP remains non-monotonic, indicating that later feedback can interfere with earlier preferences and motivating replay to preserve prior evidence.

## 5 Conclusion

We propose a framework for learning individual scientific taste from sparse researcher feedback, representing research directions in a shared, evidence-grounded evaluation space and adapting a population-level preference mapping to individual researchers. Privileged Hindsight Distillation uses later-emerging evidence during training to improve proposal-time assessment and population-level learning without exposing hindsight at inference. Experiments reveal substantial researcher-specific variation, including opposing preferences over the same proposal pairs, and show that conditional rules capture preference patterns beyond fixed criterion reweighting in real researchers. In interactive deployment, personalized rankings also surface proposals associated with greater subsequent engagement than the matched Single Call baseline. While our study is limited by sparse real-expert feedback and a finite set of evaluation criteria, the results suggest that population-level judgment provides a useful starting point but does not fully capture individual scientific preferences.

## References

[1] Timothy J. Mitchison. A question of taste. Molecular Biology of the Cell, 24(21):3278–3280, 2013. doi: 10.1091/mbc.E13-07-0410.

[2] Robert Tjian. Terri and g&d: celebrating 50 years of enlightened scientific judgment. Genes & Development, 37(1-2):6–8, 2023.

[3] Jingqi Tong, Mingzhe Li, Hangcheng Li, Yongzhuo Yang, Yurong Mou, Weijie Ma, Zhiheng Xi, Hongji Chen, Xiaoran Liu, Qinyuan Cheng, et al. Ai can learn scientific taste. arXiv preprint arXiv:2603.14473, 2026.

[4] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

[5] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jef Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

[6] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents as research assistants. Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 5977–6043, 2025.

[7] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, et al. Accelerating scientific discovery with co-scientist. Nature, pages 1–3, 2026.

[8] Lutz Bornmann, Rüdiger Mutz, and Hans-Dieter Daniel. A reliability-generalization study of journal peer reviews: A multilevel meta-analysis of inter-rater reliability and its determinants. PloS one, 5(12):e14331, 2010.

[9] Rüdiger Mutz, Lutz Bornmann, and Hans-Dieter Daniel. Heterogeneity of inter-rater reliabilities of grant peer reviews and its determinants: A general estimating equations approach. PLoS One, 7(10):e48509, 2012.

[10] Shuofei Qiao, Yunxiang Wei, Xuehai Wang, Bin Wu, Boyang Xue, Ningyu Zhang, Hossein A Rahmani, Yanshan Wang, Qiang Zhang, Keyan Ding, et al. Innoeval: On research idea evaluation as a knowledge-grounded, multi-perspective reasoning problem. arXiv preprint arXiv:2602.14367, 2026.

[11] Rongcan Pei, Fang Guo, Qinglin Qi, Qi Zhu, Yun Luo, Jianhao Yan, Minjun Zhu, Qiujie Xie, Dehong Zheng, and Yue Zhang. Deepinstructor: An agentic ai instructor for experience-driven idea evaluation. arXiv preprint arXiv:2609.22104, 2026.

[12] Vladlen Koltun and David Hafner. A measure of research taste. arXiv preprint arXiv:2105.08089, 2021.

[13] Ziqin Gong, Ning Li, and Huaikang Zhou. Llms learn scientific taste from institutional traces across the socia sciences. arXiv preprint arXiv:2603.16659, 2026.

[14] Abdur Rasool, Xiaohui Huang, Yanqing Hu, and Linyi Yang. Reviewguard: Aligning llm-assisted peer review with long-term scientific impact. arXiv preprint arXiv:2606.24892, 2026.

[15] Ming Zhang, Kexin Tan, Yueyuan Huang, Yujiong Shen, Chunchun Ma, Li Ju, Xinran Zhang, Yuhui Wang, Wenqing Jing, Jingyi Deng, et al. Opennovelty: An llm-powered agentic system for verifiable scholarly novelty assessment. arXiv preprint arXiv:2601.01576, 2026.

[16] David Soergel, Adam Saunders, and Andrew McCallum. Open scholarship and peer review: a time for experimentation. 2013.

[17] SciRate Collaboration. SciRate. URL https://scirate.com/about. Accessed September 18, 2026.

[18] Mark Dras. Squibs: Evaluating human pairwise preference judgments. Computational Linguistics, 41(2): 309–317, 2015.

[19] Hasti Narimanzadeh, Arash Badie-Modiri, Iuliia G Smirnova, and Ted Hsuan Yun Chen. Crowdsourcing subjective annotations using pairwise comparisons reduces bias and error compared to the majority-vote method. Proceedings ofthe ACM on Human-Computer Interaction, 7(CSCW2):1–29, 2023.

[20] Hasan Baig, Hailey Joren, and Joe Benton. Taste: Can ai models judge ai safety research proposals? Anthropic Alignment Science Blog, 2026.

[21] Ziyu Chen, Yilun Zhao, and Arman Cohan. Measuring the gap between human and llm research ideas. arXiv preprint arXiv:2607.01233, 2026.

[22] Yiwen Zhang, Eloise Zeng, Jaeha Lee, and Tony Yue Yu. An ai scientist that doesn’t drift: Taste, structure, and falsifiable findings in a quadruped navigation research loop. arXiv preprint arXiv:2608.07542, 2026.

[23] Qingyun Wang, Doug Downey, Heng Ji, and Tom Hope. Scimon: Scientific inspiration machines optimized for novelty. In Proceedings ofthe 62nd annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 279–299, 2024.

[24] Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. Researchagent: Iterative research idea generation over scientific literature with large language models. In Proceedings of the 2025 conference of the nations of the Americas chapter of the association for computational linguistics: human language technologies (volume 1: long papers), pages 6709–6738, 2025.

[25] Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers. In International Conference on Learning Representations, volume 2025, pages 94003–94092, 2025.

[26] Tianyu Ding, Aditya Nannapaneni, Bingfan Liu, and Ling Zhang. Autonomous research agents: A survey of ai scientists and the verification gap. arXiv preprint arXiv:2608.05179, 2026.

[27] Runlong Ye, Matthew Varona, Oliver Huang, Patrick Yung Kang Lee, Michael Liut, and Carolina Nobre. The design space of recent ai-assisted research tools for ideation, sensemaking, and scientific creativity. arXiv preprint arXiv:2502.16291, 2025.

[28] Chenrun Wang, Mingxuan Zhu, Tiancheng Huang, Wenjie Li, Yujie Zhang, Zichen Zhu, Zhiying Zou, Kai Yu, and Lu Chen. Ligbench: A unified and human-aligned benchmark for llm-based research idea generation. arXiv preprint arXiv:2608.13136, 2026.

[29] Bo Jiang. Hindsight: Evaluating llm-generated research ideas via future impact. arXiv preprint arXiv:2603.15164, 2026.

[30] Bingchen Zhao, Jenny Zhang, Chenxi Whitehouse, Minqi Jiang, Michael Shvartsman, Abhishek Charnalia, Despoina Magka, Tatiana Shavrina, Derek Dunfield, Oisin Mac Aodha, et al. Apres: An agentic paper revision and evaluation system. arXiv preprint arXiv:2603.03142, 2026.

[31] Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. The ideation-execution gap: Execution outcomes of llm-generated versus human research ideas. In International conference on learning representations, volume 2026, pages 74195–74240, 2026.

[32] Soumitra Sinhahajari, Navonil Majumder, and Soujanya Poria. On the limits of llm-as-judge for scientific novelty assessment. arXiv preprint arXiv:2606.12071, 2026.

[33] Vladimir Vapnik and Akshay Vashist. A new learning paradigm: Learning using privileged information. Neural networks, 22(5-6):544–557, 2009.

[34] David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, and Vladimir Vapnik. Unifying distillation and privileged information. arXiv preprint arXiv:1511.03643, 2015.

[35] Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled rlvr. arXiv preprint arXiv:2604.03128, 2026.

[36] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

[37] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic "diferentiation" via text. arXiv preprint arXiv:2406.07496, 2024.

[38] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. In International Conference on Learning Representations, volume 2026, pages 8479–8565, 2026.

[39] Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab. Optimizing instructions and demonstrations for multi-stage language model programs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9340–9366, 2024.

[40] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

[41] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

[42] Sriyash Poddar, Yanming Wan, Hamish Ivison, Abhishek Gupta, and Natasha Jaques. Personalizing reinforcement learning from human feedback with variational preference learning. Advances in Neural Information Processing Systems, 37:52516–52544, 2024.

[43] Anikait Singh, Sheryl Hsu, Kyle Hsu, Eric Mitchell, Stefano Ermon, Tatsunori Hashimoto, Archit Sharma, and Chelsea Finn. Fspo: Few-shot optimization of synthetic preferences personalizes to real users. arXiv preprint arXiv:2502.19312, 2025.

[44] Avinandan Bose, Shuyue Stella Li, Faeze Brahman, Pang Wei Koh, Simon Shaolei Du, Yulia Tsvetkov, Maryam Fazel, Lin Xiao, and Asli Celikyilmaz. Cold-start personalization via training-free priors from structured world models. arXiv preprint arXiv:2602.15012, 2026.

[45] Idan Shenfeld, Felix Faltings, Pulkit Agrawal, and Aldo Pacchiano. Language model personalization via reward factorization. arXiv preprint arXiv:2503.06358, 2025.

[46] Avinandan Bose, Zhihan Xiong, Yuejie Chi, Simon Shaolei Du, Lin Xiao, and Maryam Fazel. Lore: Personalizing llms via low-rank reward modeling. arXiv preprint arXiv:2504.14439, 2025.

[47] Aida Mostafazadeh Davani, Mark Díaz, and Vinodkumar Prabhakaran. Dealing with disagreements: Looking beyond the majority vote in subjective annotations. Transactions of the Association for Computational Linguistics, 10:92–110, 2022.

[48] Yinuo Xu and David Jurgens. Beyond consensus: Perspectivist modeling and evaluation of annotator disagreement in nlp. arXiv preprint arXiv:2601.09065, 2026.

[49] Matthias Orlikowski, Jiaxin Pei, Paul Röttger, Philipp Cimiano, David Jurgens, and Dirk Hovy. Beyond demographics: Fine-tuning large language models to predict individuals’ subjective text perceptions. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2092–2111, 2025.

[50] Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, et al. A roadmap to pluralistic alignment. arXiv preprint arXiv:2402.05070, 2024.

[51] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc’Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019.

[52] Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Laurent Charlin, Massimo Caccia, Min Lin, and Lucas Page-Caccia. Online continual learning with maximal interfered retrieval. Advances in neural information processing systems, 32, 2019.

[53] Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo. Prometheus 2: An open source language model specialized in evaluating other language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4334–4353, 2024.

[54] Daniel Halpern, Evi Micha, Ariel D. Procaccia, and Itai Shapira. Pairwise calibrated rewards for pluralistic alignment. In Advances in Neural Information Processing Systems, volume 38, 2025.

[55] Michael J Ryan, Omar Shaikh, Aditri Bhagirath, Daniel Frees, William Held, and Diyi Yang. Synthesizeme! inducing persona-guided prompts for personalized reward models in llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8045–8078, 2025.

[56] Jaehyung Kim and Yiming Yang. Few-shot personalization of llms with mis-aligned responses. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11943–11974, 2025.

[57] Ge Gao, Alexey Taymanov, Eduardo Salinas, Paul Mineiro, and Dipendra Misra. Aligning llm agents by learning latent preference from user edits. Advances in neural information processing systems, 37:136873–136896, 2024.

[58] Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Pengfei Liu, et al. Generative judge for evaluating alignment. In International Conference on Learning Representations, volume 2024, pages 27547–27574, 2024.

[59] Hui Chen, Miao Xiong, Yujie Lu, Wei Han, Ailin Deng, Yufei He, Jiaying Wu, Yibo Li, Yue Liu, and Bryan Hooi. Mlr-bench: Evaluating ai agents on open-ended machine learning research. Advances in Neura Information Processing Systems, 38, 2026.

[60] Filip Radlinski, Madhu Kurup, and Thorsten Joachims. How does clickthrough data reflect retrieval quality? In Proceedings ofthe 17th ACM conference on Information and knowledge management, pages 43–52, 2008.

[61] Qing Zhang, Alex Deng, Michelle Du, Huiji Gao, Liwei He, and Sanjeev Katariya. Harnessing the power of interleaving and counterfactual evaluation for airbnb search ranking. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 5205–5214, 2025.

[62] Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS), 20(4):422–446, 2002.

## A Privileged Hindsight Distillation Details

We provide additional implementation details for the four-step PHD procedure described in Section 3.2 and illustrated in Figure 2(b). The procedure identifies informative student errors, uses privileged hindsight evidence to derive reusable corrections, and accepts artifact revisions only after leakage auditing and validation.

## A.1 Stage-Specific Failure Diagnosis

At each iteration, the analyzer compares the student rollout with the corresponding stage reference and organizes informative errors into a structured failure package. The analyzer identifies errors but does not use hindsight evidence to explain their underlying causes. Such explanations are produced by the privileged teacher in the subsequent step.

Rather than passing every disagreement to the teacher, the analyzer prioritizes failures that may yield reusable improvements. These include large prediction errors, inconsistencies between judgments and supporting evidence, explicit constraint violations, and errors recurring across multiple items. The resulting failure package is structured according to the learning task.

Evaluation-space assessment. Errors are identified at the proposal and criterion levels. For each selected case, the failure package records the student’s predicted score, confidence, supporting comment, evidence-gap flag, and constraint-violation flag, together with the signed error against the reference criterion score. Recurring errors across proposals are also recorded to support the identification of systematic problems.

Population taste prediction. Errors are identified at the pairwise comparison level. Each selected case records the student’s predicted preference, confidence, rationale, and reference preference, together with the relevant criterion coordinates and utility margin. When available, results from both comparison orientations are retained to expose orientation inconsistencies. Errors are grouped according to research type and decisive evaluation criteria to identify recurring preference failures.

The resulting package provides the teacher with structured evidence of which judgments failed and where the failures occurred. Determining why they occurred and how to correct them remains the responsibility of the privileged teacher.

## A.2 Privileged Teacher Editing

The privileged teacher receives the current artifacts, student rollout, structured failure package, and privileged hindsight information. It examines the selected failures in light of completed methods, experimental outcomes, subsequent literature, citations, and community feedback to identify potential causes of the student’s errors.

The teacher then proposes revisions to the agentic artifacts. These revisions may modify evaluation rubrics, introduce reusable memory lessons, or add and revise conditional rules. The teacher does not directly replace the student’s judgments with privileged answers.

The key constraint is that hindsight may inform the discovery of a correction, but the resulting artifact must remain executable using proposal-visible information. For example, retrospective evidence may reveal that a scoring rubric systematically undervalues proposals extending an established research direction. The teacher may consequently revise the rubric to distinguish incremental extensions from substantive methodological advances. It may not encode the later experimental success of a particular proposal as an inference-time instruction.

The teacher’s private diagnostic explanations are retained for analysis but are not incorporated into student-facing artifacts. Only the proposed artifact revisions are passed to the subsequent audit and validation steps.

## A.3 Artifact Leakage Audit

A leakage auditor examines each proposed revision before it can be accepted. The audit is applied to student-facing artifacts, including rubric instructions, memory lessons, conditional predicates, and their associated explanations. It does not constrain the privileged information used internally by the teacher to derive a revision.

The auditor rejects revisions that directly expose retrospective outcomes, citation counts, community responses, reference criterion scores, or preference labels associated with individual proposals. It also rejects references to held-out items and conditional rules that require features unavailable at the proposal stage.

For conditional rules, the auditor additionally checks that their predicates operate on proposal-visible properties and conform to the predefined predicate grammar. Instance-specific rules that efectively identify a single proposal rather than expressing a reusable judgment principle are rejected.

This distinction permits hindsight-derived knowledge to be distilled into general evaluation instructions while preventing privileged instance-level information from entering the student’s inference-time artifacts. The leakage constraints and artifact output schemas remain fixed throughout the refinement process.

## A.4 Gated Validation and Artifact Acceptance

Passing the leakage audit is necessary but insuficient for accepting a revision. A candidate artifact must also demonstrate improved performance on validation data that did not motivate the edit.

Let Ψ<sub>�</sub> denote the current artifacts and Ψ<sup>′</sup> the candidate revision. The student is evaluated under both configurations using the same held-out validation examples. The revision is accepted only when all three conditions are satisfied.

1. The leakage auditor approves all modified student-facing artifacts.

2. The candidate improves the task-specific validation objective.

3. The revision does not introduce material behavioral regressions.

For evaluation-space assessment, validation considers errors against reference criterion scores, with mean absolute error serving as the primary objective. Additional checks examine whether the revision introduces new errors in previously correct judgments.

For population taste prediction, validation uses StrictPair correctness on held-out comparisons. Regression checks examine correct-to-wrong preference flips, orientation inconsistencies, and failures concentrated in particular groups of comparisons.

The validation data exclude examples used to induce the candidate revision. A separate final evaluation set remains untouched by artifact selection and acceptance decisions.

If the candidate passes all acceptance conditions, its artifacts replace the incumbent configuration,

$$
\Psi _ { t + 1 } = { \left\{ \begin{array} { l l } { \Psi _ { t } ^ { \prime } , } & { { \mathrm { i f ~ t h e ~ r e v i s i o n ~ i s ~ a c c e p t e d } } , } \\ { \Psi _ { t } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Rejected revisions are rolled back, while accepted revisions guide subsequent student rollouts. This procedure aims to retain reusable improvements that generalize beyond the errors that originally motivated the edit.

Evaluation-space assessment. A revision is accepted only when the fixed leakage auditor returns a normalized risk below 0.15, its validation MAE improves by more than 0.01, its correct-to-wrong flip rate does not exceed 5%, and its net repair is positive.

Population taste prediction. Candidate revisions are induced on $D _ { \mathrm { i n d u c e } } .$ , fitted on $D _ { \mathrm { r u l e - s e l e c t } }$ , and evaluated for acceptance on $D _ { \mathrm { a g e n t - v a l } }$ . The final evaluation set $D _ { \mathrm { f i n a l } }$ remains untouched.

Acceptance requires an improvement in StrictPair, a McNemar test with $p < 0 . 0 5$ , and a paired bootstrap 95% confidence interval for the StrictPair diference that excludes zero.

Rule specificity. The leakage auditor rejects rules whose triggering behavior is concentrated on a single paper or cluster, according to the configured concentration threshold.

## B Evaluation Space Learning

This section provides the complete criterion-annotation and proposal-time assessment instructions used to learn the shared evaluation space, together with a concrete example of how Privileged Hindsight Distillation (PHD) transforms teacher-side supervision into deployable artifacts. We first define the reference annotation rubric, then give the seed proposal-time rubric, and finally illustrate how PHD revises the rubric and student-safe memory on a held-out proposal.

## B.1 Reference Criterion Annotation

Rubric A defines the reference annotation procedure used to construct

$$
\begin{array} { r } { s ^ { * } ( P ) = \left( s _ { \mathrm { g e n } } ^ { * } , s _ { \mathrm { n o v } } ^ { * } , s _ { \mathrm { i m p } } ^ { * } , s _ { \mathrm { p a r } } ^ { * } , s _ { \mathrm { d e p } } ^ { * } , s _ { \mathrm { c o m p } } ^ { * } \right) \in \{ 0 , 1 , 2 , 3 \} ^ { 6 } . } \end{array}
$$

The six coordinates correspond to generality, novelty, importance, parsimony, depth, and algorithmic complexity.   
Each coordinate is assigned independently on an absolute 0–3 scale.

Unlike the deployable proposal-time assessor, Rubric A may use both proposal-visible evidence and privileged hindsight evidence. Proposal-visible evidence includes the proposal text, its explicit claims and assumptions, and literature available at proposal time. Privileged evidence includes the completed methodology and experimental findings in the corresponding paper, post-publication technical discussion, and subsequent community signals such as citations. The annotator must connect each assigned score to criterion-specific evidence rather than infer one criterion from writing quality, venue, author identity, or overall paper reputation.

## Rubric A: Reference Criterion Annotation

Input. A research proposal, proposal-time prior literature, and teacher-side privileged hindsight evidence.

Evidence rule. Use all available evidence that is relevant to the target criterion. Every assigned score must be justified by criterion-specific evidence. If multiple sources disagree, record the disagreement and select the score best supported by the overall evidence. Reduce confidence when evidence is incomplete, assumptions are not directly comparable, or adjacent score levels remain similarly plausible.

## Generality

Score 0. The method applies only to one special setting, and its scope or applicable regime is missing or merely implied.

Score 1. The method covers multiple instances or variants within one problem family, with limited variation in access, noise, or hardware assumptions.

Score 2. The method covers multiple problem, input, or noise settings within one broader class, with explicit applicability assumptions.

Score 3. The method covers multiple problem classes or distinct access, error, or hardware models, with explicit applicability conditions.

## Novelty

Score 0. The contribution is substantially the same as identified prior work, with no meaningful structural diference.

Score 1. The contribution is a nearby transfer, implementation variation, or minor modification of a known method.

Score 2. The contribution forms a new workflow, formulation, or structured extension by combining known mechanisms and clearly distinguishes itself from prior work.

Score 3. The contribution introduces a new primitive, reduction, encoding, or formal object absent from the relevant prior-work lineage.

## Importance

Score 0. There is no clear evidence of field demand; the motivation is absent, generic, or primarily self-asserted.

Score 1. The work addresses a useful but local subproblem, with limited evidence of broader impact.

Score 2. The addressed limitation is repeatedly identified as important across multiple works, settings, or application areas.

Score 3. An authoritative survey, roadmap, benchmark, or broad body of evidence identifies the problem as a central open problem or foundational bottleneck.

## Parsimony

Score 0. The method requires many layers, assumptions, special cases, or ad hoc components for its stated functionality.

Score 1. The method is a moderately complex pipeline with several distinct construction or execution stages.

Score 2. A small number of coordinated core subroutines, oracles, or assumptions achieves the stated objective.

Score 3. A unified primitive, representation, or objective replaces multiple otherwise separate or ad hoc components.

## Depth

Score 0. The work addresses a single setting and provides little explanation of the underlying mechanism.

Score 1. The work remains within one task family and provides mainly intuition or partial technical detail, with central analysis deferred.

Score 2. The work unifies multiple tasks or primitives, or provides concrete analytical, experimental, or diagnostic grounding of the mechanism.

Score 3. The work develops a transferable formalism supported by theorem-level, lower-bound, causal, or comparably strong mechanistic evidence.

## Algorithmic Complexity

Score 0. The work makes no meaningful complexity or resource claim relative to an appropriate baseline.

Score 1. The work makes an informal eficiency claim without directly comparable quantitative evidence.

Score 2. The work presents an explicit resource improvement relative to a domain baseline under stated assumptions and resource metrics.

Score 3. The work establishes a theoretical bound or strong Pareto improvement while reasonably accounting for state preparation, oracle access, error dependence, measurement, and readout costs.

Criterion-specific evidence requirements. Generality requires evidence about covered problem classes, access models, hardware regimes, and the assumptions under which the claimed coverage holds. Novelty is assessed relative to proposal-time prior work across problem formulation, quantum primitive, reduction or encoding, complexity model, measurement or verification strategy, and assumption set. Importance requires external evidence that the target problem is recognized beyond the proposal’s own motivation. Parsimony is evaluated from the complete method structure, including necessary subroutines, oracles, ancillas, measurements, post-selection steps, and special-case handling. Depth jointly measures the breadth of unification and the strength of mechanistic grounding. Algorithmic complexity compares the claimed and realized resource behavior with appropriate baselines while accounting for hidden computational costs.

## B.2 Seed Proposal-Time Rubric

The deployable criterion assessor $f _ { \mathrm { c r i t } }$ uses the same six-dimensional evaluation space and the same absolute 0–3 scale as the reference annotation, producing

$$
\hat { s } ( P ) = \left( \hat { s } _ { \mathrm { g e n } } , \hat { s } _ { \mathrm { n o v } } , \hat { s } _ { \mathrm { i m p } } , \hat { s } _ { \mathrm { p a r } } , \hat { s } _ { \mathrm { d e p } } , \hat { s } _ { \mathrm { c o m p } } \right) .
$$

Its information boundary, however, is diferent. At deployment, the assessor can access only the proposal and literature available at evaluation time. Completed outcomes, post-publication discussion, subsequent citations, and other hindsight information are unavailable.

We therefore initialize the student with the proposal-visible instruction shown below, which we refer to as seed Rubric B. The seed rubric is an initial operationalization of the six evaluation criteria rather than a fixed definition of how the criteria must be inferred. PHD subsequently refines its score boundaries and evidence-use rules while preserving the same proposal-time information boundary.

## Seed Rubric B: Proposal-Time Criterion Assessment

Input. A research proposal and literature available at proposal-evaluation time.

Evidence constraint. Use only information available when the proposal is evaluated. Permitted evidence includes proposal claims, equations, assumptions, methodological descriptions, planned comparisons, named prior work, and proposal-time literature. Do not use completed experimental outcomes, future reception, post-publication discussion, subsequent citations, or other hindsight information.

Scoring rule. Score each criterion independently on an absolute 0–3 scale. Do not infer one criterion from another, and do not reward writing style or perceived author reputation. If the available evidence is insuficient to distinguish adjacent levels, mark the evidence gap rather than silently assuming the stronger claim.

## Generality

Score 0. The proposal addresses one special setting; its scope or applicable regimes are missing or merely suggested.

Score 1. The proposal explicitly covers multiple instances within one problem family, but provides little cross-setting variation.

Score 2. The proposal covers multiple problem, input, or noise settings within one class and states the relevant assumptions.

Score 3. The proposal covers multiple problem classes or distinct access, error, or hardware models and states their applicability conditions.

## Novelty

Score 0. The proposed method is substantially the same as named prior work, with no structural diference stated.

Score 1. The proposal describes a nearby transfer or minor modification of a known technique or formulation.

Score 2. The proposal defines a new workflow by combining known mechanisms and states a concrete diference from named prior work.

Score 3. The proposal introduces a new primitive, reduction, encoding, or formal object that is absent from the prior-work lineage described in the proposal.

## Importance

Score 0. The proposal provides no clear evidence of field demand; its motivation is weak, generic, or absent.

Score 1. The proposal addresses a useful local subproblem but gives limited support for broader impact.

Score 2. The proposal presents the limitation as repeatedly important across multiple settings, applications, or prior works.

Score 3. The proposal supports the claim that the problem is a central open problem or foundational bottleneck for a broad research area.

## Parsimony

Score 0. The proposed method is a multi-layered pipeline with many assumptions, special cases, or ad hoc components.

Score 1. The proposed method is moderately complex and contains several distinct construction or execution stages.

Score 2. A small number of core subroutines or oracles achieves the stated objective with limited additional machinery.

Score 3. A unified primitive, representation, or objective replaces multiple otherwise separate components.

## Depth

Score 0. The proposal addresses a single setting and gives little or no explanation of the underlying mechanism.

Score 1. The proposal stays within one task family and provides intuition or partial detail, while its main proofs or diagnostics remain deferred.

Score 2. The proposal covers multiple tasks or primitives, or gives concrete lemma, error-analysis, diagnostic, or ablation plans that ground the mechanism.

Score 3. The proposal develops a transferable formalism together with explicit plans for theorem-level, lower-bound, causal, or comparably strong evidence.

## Algorithmic Complexity

Score 0. The proposal makes no complexity or resource claim relative to any baseline.

Score 1. The proposal makes an informal eficiency claim without comparable quantitative evidence.

Score 2. The proposal states an explicit improvement relative to a domain baseline, with resource metrics and assumptions.

Score 3. The proposal states a theoretical bound or strong Pareto improvement and reasonably accounts for hidden state-preparation, oracle, error, measurement, and readout costs.

Required structured output. For every evaluation criterion �, return

�ˆ<sub>�</sub> (�) = (score, confidence, supporting comment, evidence-gap indicator, constraint-violation flag) .

The supporting comment must identify proposal-visible evidence for the score. The evidence-gap indicator is activated when the proposal does not provide enough information to distinguish adjacent score levels. The constraint-violation flag records a direct conflict between an assigned score and its anchor, such as assigning maximal algorithmic complexity when no resource claim is present.

## B.3 Learning Rubric B and Student-Safe Case Memory

During training, the privileged teacher observes the student’s assessment �ˆ(�), the Rubric-A reference �<sup>∗</sup>(�), and the supporting teacher-side evidence. Rather than transferring privileged facts into the deployable workflow, the teacher diagnoses recurring proposal-time reasoning errors and converts them into reusable student-safe supervision.

We update two complementary student-side artifacts. Systematic errors that expose ambiguity in score boundaries are translated into revisions of Rubric B. Errors that are more useful as concrete precedents are stored in a separate student-safe case memory. In both cases, hindsight-dependent facts are removed before the artifact becomes accessible to the student.

Rubric revisions. Accepted rubric updates refine evidence criteria and adjacent-score boundaries. Representative learned principles include the following:

• several benchmark models, parameter regimes, or subcases within one task family do not automatically establish broad generality;

• a new workflow, robustness regime, or theorem-level extension built from known primitives generally supports intermediate novelty rather than maximal novelty;

• multiple subroutines do not necessarily imply low parsimony when the subroutines directly support one coherent mechanism;

• explicit query, gate, depth, qubit, or error analysis can support an intermediate algorithmic-complexity score even when the strongest form of resource advantage has not been established.

Case memory. Each memory entry records a proposal-visible trigger, the afected evaluation criterion, a relevant score region, and a reusable scoring lesson. Teacher-only evidence, reference outcomes, and information unavailable at proposal time are removed. At inference time, retrieved entries therefore serve as calibration precedents rather than as additional facts about the target proposal.

Candidate rubric and memory states are evaluated on the validation split using criterion-level mean absolute error,

$$
\mathcal { L } _ { \mathrm { e v a l } } = \frac { 1 } { 6 | \mathcal { D } _ { \mathrm { v a l } } | } \sum _ { P \in \mathcal { D } _ { \mathrm { v a l } } } \sum _ { d = 1 } ^ { 6 } \left| \hat { s } _ { d } ( P ) - s _ { d } ^ { * } ( P ) \right| .
$$

A candidate state is retained only when it satisfies the validation gate. Otherwise, the student is rolled back to the incumbent rubric and memory. Student-facing artifacts are additionally screened by a pattern-based leakage check before being admitted into the deployable workflow.

Example Learned Rubric-B Revision

## Afected criterion Novelty

Observed failure pattern The student repeatedly treats a new arrangement of established mechanisms as evidence of a fundamentally new primitive.

Accepted calibration rule A new workflow, formal threshold, robustness regime, or structured extension of an existing method can justify novelty 2. Novelty 3 should be reserved for a new primitive, reduction, encoding, or formal object not already present in the relevant prior lineage.

## Example Student-Safe Memory Entry

## Dimension Parsimony

Trigger A proposal contains several proof or implementation components that all support the same central mechanism.

Calibration lesson Parsimony is not only about having few steps; it is about whether those steps perform separate jobs or jointly support one central mechanism. A proposal with several proof or implementation components can still receive an intermediate parsimony score when the components are coordinated around the same objective.

Excluded information Completed experimental outcomes, teacher-side hindsight evidence, the Rubric-A reference outcome, and any information unavailable at proposal time.

## B.4 Evaluation-Space Learning Case Study

We illustrate the learning process on the held-out proposal Quantum Hamiltonian Descent (arXiv 2303.01471). The held-out proposal is used only for analysis. Rubric revisions and memory entries are learned from analogous failure patterns in the training split rather than from this example itself. Scores are reported in the order generality, novelty, importance, parsimony, depth, and algorithmic complexity. The progress is shown in Table 6

Reference Annotation

Rubric A assigns the proposal the reference criterion vector

$$
s ^ { * } ( P ) = ( 3 , 2 , 2 , 2 , 2 , 2 ) .
$$

The high generality score reflects a common Hamiltonian-descent framework across smooth and non-convex optimization, digital fault-tolerant simulation, and analog Ising-machine implementations under distinct assumptions. Novelty is intermediate because the proposal introduces a distinctive workflow based on quantizing continuous-time optimization dynamics while relying on known components such as Hamiltonian simulation, product formulas, quantum Fourier transforms, and Ising embeddings. Parsimony is also intermediate because these components support one central mechanism despite a nontrivial implementation pipeline. Algorithmic complexity receives an intermediate score because the proposal states explicit resource quantities and baseline comparisons without establishing the strongest level of resource advantage with every hidden cost resolved.

## Assessment Before PHD Learning

Using the seed Rubric B without learned case memory, the proposal-time student predicts

$$
{ \hat { s } } _ { \mathrm { s e e d } } ( P ) = ( 3 , 3 , 2 , 0 , 2 , 1 ) .
$$

Three characteristic errors are visible. First, the student interprets the Hamiltonian formulation as a wholly new primitive and assigns novelty 3 instead of 2. Second, it counts implementation components without recognizing their shared conceptual role and assigns parsimony 0 instead of 2. Third, it treats the absence of a fully established quantum advantage as evidence of little algorithmic-complexity contribution and assigns 1 instead of 2.

How the teacher turns training failures into student artifacts. On training proposals, the privileged teacher identifies analogous recurring errors. For example, when the student treats multiple prime families, decoding regimes, or circuit variants within one task as evidence of broad generality, the teacher converts the diagnosis into a proposal-visible boundary rule: multiple instances or noise regimes within one task family do not by themselves justify a high generality score. Similarly, repeated confusion between a new combination of known mechanisms and a genuinely new primitive produces the novelty calibration shown above. For parsimony, the teacher learns to distinguish raw component count from functional coherence and stores the more concrete precedent in case memory.

## Assessment After PHD Learning

With the validation-selected Rubric B and case memory, the student predicts

$$
{ \hat { s } } _ { \mathrm { l e a r n e d } } ( P ) = ( 2 , 2 , 2 , 1 , 2 , 1 ) .
$$

The learned novelty boundary corrects the novelty score from 3 to 2. The parsimony prediction moves from 0 to 1, reducing but not eliminating the error. Algorithmic complexity remains underestimated. The generality score moves from the correct seed value of 3 to 2, illustrating that a transferable calibration rule can still be over-applied to an individual proposal.

The proposal-level MAE decreases from 0.667 with the seed Rubric B to 0.500 after learning. Across the held-out test split, absolute-score MAE decreases from 0.216 for the w/o PHD student to 0.183 for the validation-selected PHD student state. This aggregate comparison evaluates the revised rubric and memory jointly. It therefore measures improvement of the learned student state rather than isolating the causal contribution of the rubric revision or the case memory individually.

## C Population Taste Distillation

## C.1 Hindsight Reference Panel

We construct population-level reference supervision using an independent multi-family LLM panel. The panel is used only to construct ofline reference judgments for training and evaluation. Unlike the deployable student, panel judges may access privileged hindsight evidence in addition to proposal-time information. These privileged inputs are never exposed to the student during deployment.

Panel prompt. For every proposal pair $( P _ { A } , P _ { B } )$ , each judge independently receives the same normalized evidence bundle together with the following instruction.

## Hindsight Reference Panel Prompt

You are an ofline reference judge comparing two research proposals to construct a population-level preference reference. You receive (i) the complete text of proposals A and B, (ii) reference assessments along the six evaluation criteria, (iii) source-paper and proposal-time evidence supporting those assessments, and (iv) privileged evidence about subsequent community response, including citation function and polarity, methodological adoption, and reuse.

Using the complete evidence bundle, determine which proposal is preferable as a research direction at the population level. Consider the criterion-level assessments and community-response evidence jointly, and explain any disagreement between them. Your judgment must not depend on presentation order or proposal length. You must select exactly one winner, either A or B. Ties and uncertain outcomes are not permitted.

<table><tr><td>Dimension</td><td>Reference</td><td>Seed</td><td>Learned</td></tr><tr><td>Generality</td><td>3</td><td>3</td><td>2</td></tr><tr><td>Novelty</td><td>2</td><td>3</td><td>2</td></tr><tr><td>Importance</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Parsimony</td><td>2</td><td>0</td><td>1</td></tr><tr><td>Depth</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Algorithmic complexity</td><td>2</td><td>1</td><td>1</td></tr><tr><td>MAE</td><td>一</td><td>0.667</td><td>0.500</td></tr></table>

Table 6: Evaluation-space learning on a held-out proposal. Rubric A provides the reference coordinates. Seed and Learned show the proposal-time predictions before and after PHD learning.

<table><tr><td>Evidence source</td><td>Content</td><td>Student-visible</td></tr><tr><td>Proposal text</td><td>Complete proposal claims, assumptions, equations, and methodology</td><td>Yes</td></tr><tr><td>Proposal-time literature</td><td>Prior work and external evidence available when the proposal is evaluated</td><td>Yes</td></tr><tr><td>Criterion annotation</td><td>Six-dimensional Rubric-A assessment and its supporting evidence</td><td>No</td></tr><tr><td>Completed work</td><td>Source-paper methodology, experimental findings, and realized analyses</td><td>No</td></tr><tr><td>Community response</td><td>Follow-up citations, scites in SciRate, citation function and polarity, adoption distance, methodological integration, and primitive reuse</td><td>No</td></tr></table>

Table 7: Evidence available to the hindsight reference panel. Proposal-visible sources may also be accessed by the deployable workflow, whereas privileged sources remain teacher-side information.

Return a JSON object containing the selected winner, the estimated probability that A is preferred to B at the population level, confidence in the judgment, a short rationale, the decisive factors, an explanation of how the criterion-level and community-response evidence agree or conflict, and whether a proposal-only reader could plausibly reach the same conclusion.

Evidence bundle. All judges receive the same canonical A/B orientation and the same normalized evidence bundle. The bundle combines proposal-visible information with teacher-side hindsight evidence as summarized below.

Proposal texts are truncated only when they exceed the fixed context budget. No model-specific evidence preprocessing is applied, so all panel members judge the same logical pair from the same evidence state.

Independent annotation. Three model families independently annotate every proposal pair at temperature zero, namely GPT-5.5, Claude Opus 5, and Gemini 3.6 Flash. Judges cannot observe one another’s outputs. Each judge returns a hard winner

$$
y _ { j } \in \{ A , B \} ,
$$

together with an estimated probability

$$
q _ { j } = P _ { j } ( A \succ B ) ,
$$

confidence, rationale, decisive factors, and an explanation of how the criterion-level and hindsight evidence support the decision.

Single-judge normalization. Each response is parsed against the required output schema. The reported probability is clipped to [0, 1] and normalized to remain directionally consistent with the hard winner,

$$
y _ { j } = A \Longrightarrow q _ { j } \geq 0 . 5 1 , \qquad y _ { j } = B \Longrightarrow q _ { j } \leq 0 . 4 9 .
$$

A valid normalized annotation therefore cannot assign $q _ { j } = 0 . 5$ . Malformed responses are treated as annotation failures and regenerated or excluded rather than converted into tie labels.

Cross-family aggregation. The population reference label is determined only by the three hard winner labels,

$$
y _ { \mathrm { p a n e l } } ( A , B ) = { \mathrm { M a j o r i t y V o t e } } ( y _ { 1 } , y _ { 2 } , y _ { 3 } ) , \qquad y _ { j } \in \{ A , B \} .
$$

Because the panel contains three judges and every valid judge selects exactly one proposal, majority voting always produces a unique binary preference. Per-model probabilities, confidence values, rationales, and decisive factors are retained as privileged teacher-side metadata but do not alter the final reference label.

## Reference Construction Summary

For each proposal pair, all three model families independently inspect the same hindsight evidence bundle. Each model produces one binary preference. The majority vote forms $y _ { \mathrm { p a n e l } } ( A , B )$ , which provides the population-level reference supervision used during Population Taste Distillation. Rich judge outputs are retained only as teacher-side diagnostic information.

## C.2 Pairwise Instantiation of Population Taste

Section 3.4 defines population taste generally as a mapping from features in the shared evaluation space to research preference. In our experiments, scientific taste is observed through pairwise proposal comparison. We therefore instantiate the general mapping using the relative evaluation-space features of the two research directions.

For a proposal pair $( A , B )$ , let $\hat { s } ( A ) , \hat { s } ( B ) \in \mathbb { R } ^ { 6 }$ denote their estimated positions in the shared evaluation space and define the pairwise criterion diference as

$$
x _ { A , B } = \hat { s } ( A ) - \hat { s } ( B ) .
$$

The linear population prior uses $w _ { \mathrm { p o p } } ^ { \top } x _ { A , B } .$ , where $w _ { \mathrm { p o p } }$ captures population-level weighting of the six evaluation criteria. PHD augments this prior with proposal-visible conditional rules, giving the pairwise instantiation of the population mapping from Section 3.4,

$$
u _ { \mathrm { p o p } } ( x _ { A , B } , \mathcal { E } _ { A , B } ) = w _ { \mathrm { p o p } } ^ { \top } x _ { A , B } + \sum _ { r \in R _ { \mathrm { p o p } } } \tilde { \alpha } _ { r } \phi _ { r } ( x _ { A , B } , \mathcal { E } _ { A , B } ) ,
$$

where $\mathcal { E } _ { A , B }$ contains proposal-visible evidence for the pair, $\phi _ { r }$ denotes the activation of rule $r ,$ and $\tilde { \alpha } _ { r }$ is its reliability-adjusted contribution. The induced preference probability is $p _ { \mathrm { p o p } } ( A \succ B ) = \sigma ( u _ { \mathrm { p o p } } ( x _ { A , B } , \mathcal { E } _ { A , B } ) )$

From General Taste Mapping to Pairwise Supervision

The underlying object being learned is a population mapping over evaluation-space features. Pairwise comparison is the observation interface used in our task rather than the definition of population taste itself. Accordingly, the general feature representation �ˆ(�) becomes the relative representation $x _ { A , B }$ when the mapping is trained and evaluated against the panel preference $y _ { \mathrm { p a n e l } } ( A , B )$ .

## D Individual Taste Adaptation Details

Individual Taste Adaptation freezes the learned evaluation-space assessor and population state and updates only the current researcher’s personal state from revealed pairwise feedback. No privileged hindsight evidence, hidden researcher profile, or held-out query label is available during adaptation. The implementation first fits a populationcentered residual $\Delta w _ { u }$ by MAP logistic regression, then considers conditional personal rules only when recurring errors remain after continuous criterion reweighting. Accepted rule updates are checked by replay over the accumulated support bufer before the personal state is retained for evaluation.

Ablation variants. Cold Start uses the frozen population mapping without researcher-specific updates. Residualonly fits $\Delta w _ { u }$ while keeping the researcher-specific rule set empty. Complete uses both $\Delta w _ { u }$ and accepted researcher-specific conditional rules. The w/o PHD variant removes PHD from the shared initialization and runs the same downstream individual-adaptation procedure starting from the seed Rubric B and a randomly initialized population prior, while keeping the backbone, support budget, replay procedure, and adaptation mechanism unchanged.

## D.1 Replay-Based Rule Acceptance

Candidate personal rules are evaluated over the accumulated support bufer using leave-one-out cross-fitted replay. For each support comparison $i ,$ we refit the individual residual $\Delta w _ { u }$ and the candidate rule coeficient $\alpha _ { r }$ on the

Table 8: An illustrative case of divergent scientific taste. Two researchers evaluate the same ten proposal pairs using identical criterion features. Predictions are shown as U04/U05. Bold indicates a correct individual prediction. A star indicates that both opposing preferences are correctly recovered. Proposal titles are abbreviated for readability.
<table><tr><td>Pair</td><td>Research directions (A / B)</td><td>Gold</td><td>Cold</td><td>Residual</td><td>Full</td></tr><tr><td colspan="6">Opposing preferences between U04 and U05</td></tr><tr><td>005/008</td><td>Eigenvalue processing / Non-unitary dynamics</td><td>B/A</td><td>A/A</td><td>B/B</td><td>B/B</td></tr><tr><td>009/040</td><td>Optimal quantum simulation / Fourier-based LCHS</td><td>A/B</td><td>A/A</td><td>★A/B</td><td>★A/B</td></tr><tr><td>009/045</td><td>Optimal quantum simulation / Circuit-efficient randomized methods</td><td>A/B</td><td>B/B</td><td>B/B</td><td>★A/B</td></tr><tr><td>013/029</td><td>Linear matrix differential equations / Gibbs expectation</td><td>B/A</td><td>A/A</td><td>A/A</td><td>★B/A</td></tr><tr><td>027/034</td><td>Quantum Hamiltonian Descent / Provably efficient algorithm</td><td>B/A</td><td>A/A</td><td>A/A</td><td>A/A</td></tr><tr><td colspan="6">Aligned preferences between U04 and U05</td></tr><tr><td>013/023</td><td>Linear matrix differential equations / Sign-embedding algorithms</td><td>B/B</td><td>A/A</td><td>A/B</td><td>B/B</td></tr><tr><td>028/045</td><td>Quantum Hermite transform / Circuit-efficient randomized methods</td><td>A/A</td><td>A/A</td><td>B/B</td><td>A/B</td></tr><tr><td>028/049</td><td>Quantum Hermite transform / End-to-end quantum simulation</td><td>A/A</td><td>A/A</td><td>A/B</td><td>A/B</td></tr><tr><td>029/036</td><td>Gibbs expectation / Efficient quantum simulation</td><td>A/A</td><td>B/B</td><td>B/B</td><td>B/B</td></tr><tr><td>039/040</td><td>BELT block encoding / Fourier-based LCHS</td><td>A/A</td><td>A/A</td><td>A/A</td><td>A/A</td></tr><tr><td colspan="6">Correct predictions (U04 / U05)</td></tr><tr><td colspan="2">Opposing-choice recovery</td><td></td><td>4/10 / 7/10 0/5</td><td>4/10 / 6/10 1/5</td><td>8/10  / 6/10 3/5</td></tr></table>

remaining � − 1 comparisons and evaluate the held-out comparison. The resulting out-of-fold predictions are used to compute three rule efects: fixed, the number of previously incorrect StrictPair predictions corrected by the rule; broken, the number of previously correct predictions made incorrect; and

$$
{ \mathrm { n e t } } = { \mathrm { f i x e d } } - { \mathrm { b r o k e n } } .
$$

We additionally compute the change in negative log-likelihood, $\Delta _ { \mathrm { N L L } } = \mathrm { N L L } _ { \mathrm { b e f o r e } } - \mathrm { N L L } _ { \mathrm { a f t e r } } .$

Under the default replay gate, a candidate rule is accepted only if (i) its fitted coeficient is nonzero, (ii) it activates on at least max(4, round(�/3)) observed comparisons, (iii) net ≥ 1, (iv) broken = 0, and (v) $\Delta _ { \mathrm { N L L } } > 0$ The deployed rule coeficient is obtained by aggregating the out-of-fold fitted coeficients.

The replay ablation keeps the same rule proposer, coeficient fitting, support sequence, and downstream evaluation, but bypasses conditions (ii)–(v); only the nonzero-coeficient check is retained. Thus, the ablation isolates the efect of replay-based rule acceptance rather than removing individual residual fitting or altering RW-SP.

## D.2 A Case Study of Divergent Individual Taste

We examine two researchers, U04 and U05, who evaluate the same ten proposal pairs using identical frozen criterion features. Five pairs elicit opposing preferences, providing a direct test of whether personalized mappings can recover conflicting choices rather than reproducing a shared population prediction.

Table 8 shows that the cold-start mapping fails to recover both preferences on any of the five conflicting pairs. Residual-only adaptation recovers one, while the complete mapping recovers three. The improvement is particularly visible for U04, whose accuracy increases from 4/10 to 8/10. U05’s accuracy, however, decreases from 7/10 at cold start to 6/10 after adaptation. This example illustrates how conditional individual rules can recover opposing preferences that residual weighting misses, while also showing that improved diferentiation does not guarantee higher accuracy for every researcher.

## E Data Construction and Evaluation Protocols

Source pool. We construct the shared proposal corpus from SciRate under the search query quantum algorithm. After removing noise clusters and requiring clusters suficiently rich for controlled comparison, we stratify the retained papers by topical cluster, release time, and scites. From this filtered pool, we curate two complementary sources: a broad SciRate subset for topical coverage and a classic lookup set containing historically salient quantum-algorithm papers used as a diagnostic core.

Proposal reconstruction. Each retained paper is converted into a proposal-time document. We preserve the problem statement, proposed method, evaluation plan, expected contributions, and risks, while rewriting results and other post-hoc outcomes as objectives or hypotheses. Reconstruction uses full-text extraction followed by an LLM rewrite under a fixed paper-to-proposal prompt, so that downstream judges receive only information that could plausibly have been stated before the work unfolded. Controlled strengthen, weaken, and revision variants used elsewhere as comparator training signals are excluded from the Evaluation Space: absolute criterion assessment is defined only on canonical paper-derived proposals, rather than synthetic variants of the same underlying work.

Eligibility and paper-disjoint splits. Both Evaluation Space Learning and Population Taste Distillation use carefully curated, paper-disjoint splits. Constructing these evaluations requires controlled proposal-time reconstruction and privileged reference construction. A retained proposal must therefore be a canonical paper-derived proposal rather than a rewrite variant, admit a single six-dimensional absolute reference under Rubric A, support a frozen privileged-evidence package, and have its underlying paper assigned to exactly one of train, validation, or test. These constraints prevent alternative reconstructions of the same work from being treated as independent examples and ensure that privileged evidence cannot leak across splits. The validation split is used only for artifact-update gating, while the test split remains fixed throughout development.

Population Taste pairs. Population Taste Distillation constructs comparisons within the same frozen proposa universe, with additional constraints on temporal comparability and informativeness. We retain temporally matched comparisons with suficiently informative diferences, while controlling proposal degree and winner imbalance when forming the broader annotation queue. The splits remain paper-disjoint across train, validation, and test: an underlying paper may participate in multiple comparisons within the same split, but never appears in another split, and no proposal pair is duplicated across splits. Privileged population references are obtained from a three-family LLM-as-a-Judge panel that observes the same proposal-visible text together with hindsight evidence. The resulting benchmark is therefore determined jointly by proposal-time reconstructability, privileged-reference availability, cross-split paper isolation, temporal comparability, and pair quality rather than by unconstrained pair-level sampling.

Individual taste sessions. Individual Taste Adaptation operates under a diferent scarcity regime because its supervision must come directly from researchers. Ofline, each of seven researchers annotates 25 proposal pairs in one sitting: 15 sequential support comparisons used for adaptation and 10 held-out query pairs reserved for evaluation, each accompanied by a binary preference and confidence score. The support/query split is defined at the pair level: no query pair is used during adaptation, although the same proposal may appear in both support and query comparisons. The evaluation therefore measures generalization to unseen preference comparisons rather than to entirely unseen proposals.

The support budget is chosen to reflect the scale of supervision naturally available during interactive use. Across logged deployment sessions, a single sitting yields at most three pool rounds, corresponding to at most six reconstructed competitive preference signals, while even the most active user accumulates at most twelve such signals across sessions. The 15-support setting therefore evaluates adaptation at approximately the same supervision scale encountered in deployment rather than assuming dense preference annotation.

The seven-researcher study is intended to test whether sparse researcher-specific feedback can improve heldout pairwise preference prediction within individuals. It is not used to estimate the population distribution of scientific taste across researchers; population-level conclusions are evaluated separately through the Population Taste benchmark above.

Simulated researcher cohorts. We complement the seven real researchers with 100 simulated researchers under the same 15-support and 10-query protocol. Sim.-Q contains 50 quantum-researcher profiles derived from the publication histories of prominent researchers, while Sim.-Ctrl. contains 50 controlled profiles whose preferences are generated by varying criterion-level trade-ofs. Sim.-Ctrl. is designed to isolate preference variation that can be represented by fixed criterion reweighting, whereas Sim.-Q provides broader publication-history-derived preference patterns. Simulated cohorts are reported separately from the real-researcher results and are not treated as human evidence.

Researcher identifiers and privacy. Throughout the evaluation and appendix, real researchers are represented only by pseudonymous identifiers such as U04 and U05. Researcher-specific adaptation uses observed preference feedback rather than identity-derived attributes or hidden profile information.

Table 9: Baseline configurations. Backbones, input representations, and target-researcher feedback for the evaluated methods. External population judges are evaluated as alternative priors under the downstream individual-adaptation protocol. Detailed implementation settings are provided below.
<table><tr><td>Component</td><td>Method</td><td>Backbone</td><td>Input</td><td>Support</td></tr><tr><td>Alternative prior</td><td>Auto-J</td><td>autoj-13b</td><td>Proposal pair</td><td>0</td></tr><tr><td>Alternative prior</td><td>MLR-Judge</td><td>claude-opus-5, gemini-3.6-flash</td><td>Independent proposal scores</td><td>0</td></tr><tr><td>Alternative prior</td><td>SciJudge</td><td>SciJudge-4B</td><td>Title, abstract, date</td><td>0</td></tr><tr><td>Individual</td><td>SC</td><td>gpt-5.5</td><td>Proposal pair, support comparisons</td><td>15</td></tr><tr><td>Individual</td><td>PP</td><td>gpt-5.5</td><td>Proposal pair, researcher profile</td><td>15</td></tr><tr><td>Individual</td><td>SynthesizeMe</td><td>gpt-5.5</td><td>Proposal pair, persona, demonstrations</td><td>15</td></tr><tr><td>Individual</td><td>VPL</td><td>GPT-2 encoder, released VAE</td><td>Frozen embeddings</td><td>15</td></tr><tr><td>Individual</td><td>FERMI</td><td>gpt-5.5</td><td>Personalized judge prompts</td><td>15</td></tr><tr><td>Individual</td><td>CIPHER</td><td>gpt-5.5</td><td>Proposal pair, retrieved preference notes</td><td>15</td></tr><tr><td>Individual</td><td>LoRe</td><td>GPT-2 encoder, rank-8 basis</td><td>Preference-difference embeddings</td><td>15</td></tr><tr><td>Individual</td><td>FSPO</td><td>Qwen2.5-1.5B-Instruct</td><td>Proposal pair, in-context examples</td><td>15</td></tr></table>

## F Baseline Configurations and Metric Details

Table 9 summarizes the backbone, input representation, and feedback configuration of the evaluated baselines. Unless otherwise specified, API-based baselines use gpt-5.5 at temperature 0. Published methods retain their released backbones or use the adaptations specified below.

For individual taste evaluation, all methods use the same seven real-researcher sessions, with 15 support comparisons and 10 held-out query pairs per researcher. StrictPair requires correct predictions under both proposal orientations, and individual performance is reported using Regret-Weighted StrictPair (RW-SP).

## F.1 Alternative Population Priors

We evaluate external scientific judges as alternative population priors for downstream adaptation. They are not used to construct the LLM-panel reference for Population Taste Distillation. Their outputs are converted into preference priors and evaluated under the same adaptation protocol.

Auto-J [58] uses the released GAIR/autoj-13b model and its pairwise-tie protocol. We evaluate both proposal orientations using response-order swapping. The model receives full proposals, subject to a tokenizer maximum length of 3500 tokens, and runs in float16 precision.

MLR-Judge [59] scores each proposal independently using an adapted MLR-Bench proposal rubric. The oficial two-model ensemble is remapped to claude-opus-5 and gemini-3.6-flash. We average the models OverallAssessment scores for each proposal and convert the resulting pairwise diference into a soft preference prior,

$$
q _ { \mathrm { r e f } } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( 0 . 5 + \frac { \mathrm { s c o r e } _ { A } - \mathrm { s c o r e } _ { B } } { 1 8 } \right) ,
$$

where $\mathrm { s c o r e } _ { A }$ and $\mathrm { s c o r e } _ { B }$ denote the ensemble-averaged scores of proposals � and �, respectively.

SciJudge [3] uses the released OpenMOSS-Team/SciJudge-4B. Each proposal pair is represented using its title, abstract, and date. We evaluate both (�, �) and $( B , A )$ orientations. After reversing the labels from the latter orientation, we obtain two votes $\nu _ { a b }$ and $\nu _ { b a }$ in the original $( A , B )$ orientation, each taking a value in $\{ A , B , \emptyset \}$

We construct the soft preference prior for � as

$$
q _ { \mathrm { r e f } } = \left\{ \begin{array} { l l } { 0 . 7 5 , } & { \nu _ { a b } = \nu _ { b a } = A , } \\ { 0 . 2 5 , } & { \nu _ { a b } = \nu _ { b a } = B , } \\ { 0 . 6 5 , } & { \mathrm { o n l y ~ o n e ~ v a l i d ~ v o t e , ~ f a v o r i n g ~ } A , } \\ { 0 . 3 5 , } & { \mathrm { o n l y ~ o n e ~ v a l i d ~ v o t e , ~ f a v o r i n g ~ } B , } \\ { 0 . 5 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

The final case includes conflicting votes and cases where both votes are missing. These values are assigned by the voting protocol rather than obtained through probability calibration.

Generation uses sampling with do\_sample=True, temperature 0.7, top\_p=0.8, and top\_k=20.

## F.2 Individual-Taste Baselines

All personalization methods receive the same 15 support comparisons and are evaluated on the same 10 held-out query pairs. Support memory uses arXiv abstracts, while queries use full proposal text unless otherwise specified.

The SC baseline directly predicts pairwise preferences using the support comparisons in the input context. The PP (Profile Prompting) baseline first summarizes the researcher using the 15 support annotations and then uses this summary for preference prediction, rather than including all comparisons directly in the query prompt. Both variants use gpt-5.5 at temperature 0.

SynthesizeMe [55] uses gpt-5.5 to fit a researcher persona and select up to 4 informative demonstrations from the 15 text-only support preferences. We use a training-to-validation ratio of 0.7 and evaluate 10 search candidates.

VPL [42] trains the released variational autoencoder using frozen GPT-2 last-token embeddings. Training runs for 40 epochs with batch size 8, learning rate $3 \times 1 0 ^ { - 4 }$ , latent dimension 128, and KL weight 10<sup>−3</sup>.

FERMI [56] iteratively refines personalized judge prompts using gpt-5.5. We run $T = 4$ optimization rounds, generating $K = 3$ new prompts per round. The memory size is 4, and the retrieval procedure selects the top 3 entries. Optimizer sampling uses temperature 0.8, while other API-based inference follows the default temperature of 0.

CIPHER [57] induces a short preference note from each support judgment. At query time, it retrieves the top 5 relevant notes to condition pairwise preference prediction.

LoRe [46] learns a rank-8 shared preference basis from GPT-2 preference-diference embeddings. We use Adam with learning rate 0.05, 2000 joint-training steps, and 500 few-shot adaptation steps. Researcher-specific weights $w _ { u }$ are fitted using the 15 support comparisons.

FSPO [43] adapts Qwen2.5-1.5B-Instruct using LoRA and IPO on leave-one-support-out contexts. We use LoRA rank $r = 8 ,$ , 2 training epochs, learning rate $1 0 ^ { - 4 }$ , IPO coeficient $\beta = 0 . 1$ , and 4 in-context demonstrations. Candidate preferences are scored using length-normalized continuation log-probabilities.

## F.3 Confidence Mapping for RW-SP

RW-SP uses an orientation-specific preference probability rather than the raw choice-relative confidence annotation. Let $c _ { i } \in [ 0 , 1 ]$ denote the researcher’s confidence in the selected direction for comparison $( A _ { i } , B _ { i } )$ . We convert it to

$$
p _ { i } = P ( A _ { i } \succ B _ { i } ) = \left\{ { { c } _ { i } } , \atop { { 1 - c } _ { i } , \mathrm { ~ } } \mathrm { ~ i f ~ t h e ~ r e s e a r c h e r ~ s e l e c t s ~ } A _ { i } , \right.
$$

The corresponding regret weight is

$$
d _ { i } = 2 | p _ { i } - 0 . 5 | .
$$

Thus, $p _ { i } = 0 . 5$ receives zero weight, while a deterministic preference $p _ { i } \in \{ 0 , 1 \}$ receives unit weight.

When multiple persona judgments are aggregated, each choice-relative confidence is first converted to an orientation-specific probability using the same rule, and the probabilities are then averaged:

$$
{ q _ { \mathrm { u s e r } } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } p _ { i j } } .
$$

The aggregate confidence stored by the data pipeline, $| 2 q _ { \mathrm { u s e r } } - 1 |$ , is a preference-strength statistic and is not reused as $p _ { i }$ in RW-SP.

If an orientation-specific probability is unavailable, we fall back to the hard preference label, corresponding to $p _ { i } \in \{ 0 , 1 \}$ . If all comparisons have zero regret weight, RW-SP falls back to the unweighted StrictPair mean. The probabilities $p _ { i }$ are used only for evaluation weighting. Individual residual and rule fitting use the observed binary preference labels rather than $p _ { i }$ as soft training targets.

## G Interactive Proposal Exploration and Deployment Evaluation

System interface and interaction workflow. We integrate the learned individual taste mapping into an interactive web system for exploring research directions. The system supports proposal exploration and preference feedback, allowing researchers to interact with candidate directions and refine their interests through successive interactions. The learned taste mapping is used to personalize proposal rankings based on the shared evaluation-space representation and the researcher’s accumulated feedback

Figure 3 illustrates the system interface, its main functionalities, and an example of the user interaction process. The interface provides a practical setting in which preference learning afects which research directions are presented to researchers, rather than being evaluated solely through ofline pairwise predictions.

Researcher cohorts and evaluation protocol. We evaluate the system with 20 real researchers from multiple disciplines, including quantum research, computer science, statistics, and medicine. Their interaction trajectories provide behavioral evidence of which displayed proposals they choose to inspect, refine, or pursue.

![](images/19611daf433ee7c033a1a4856ac137a818b42544b09771cf5ef6b2d19949cae3.jpg)  
Figure 3: Interactive proposal exploration system. The illustration presents the web interface, its main functionalities, and an example of how researchers explore candidate directions and provide feedback. The learned individual taste mapping informs personalized proposal ranking throughout the interaction process.

To complement the real-user evaluation, we conduct ofline exploration simulations with the same 100 simulated researcher profiles described in Appendix E: 50 Sim.-Q profiles derived from quantum-researcher publication histories and 50 Sim.-Ctrl. profiles generated from controlled criterion trade-ofs. The simulated cohorts broaden the range of preference configurations, and we report them separately rather than treating simulated engagement as equivalent to observed human behavior.

At each interaction round, both ranking policies receive the same candidate proposal pool, proposal-visible information, and accumulated user feedback. Our personalized policy incorporates this feedback through the learned individual taste mapping, whereas the SC baseline uses the same gpt-5.5 backbone and directly conditions on the complete interaction history in its prompt and infers researcher-specific preferences in context. Both policies use the same base model. This controlled setup holds the candidate pool, backbone, and available interaction evidence fixed, allowing the comparison to test whether explicitly learning an individual taste mapping provides benefits beyond direct in-context personalization.

Interaction history and supervision scale. Across logged deployment sessions, a single sitting yields at most three pool rounds, corresponding to at most six reconstructed competitive preference signals, while even the most active user accumulates at most twelve such signals across sessions. The complete interaction history therefore remains compact enough to be included in every SC prompt without truncation. These trajectories also motivate the 15-support budget used in the ofline individual-taste evaluation in Appendix E, which operates at approximately the same scale of researcher supervision.

Source-blind competitive-pair interleaving. We adapt competitive-pair team-draft interleaving [60, 61] to compare the two ranking policies within a shared display.

Proposals selected by both policies are displayed only once and receive no policy-specific credit. When the policies disagree, their selections are placed together as adjacent competitive pairs. Presentation order within each pair is randomized to reduce systematic position efects. Researchers are not informed which ranking policy contributed each proposal.

This design focuses the evaluation on disagreements between the two ranking policies. Rather than comparing engagement across separately presented ranked lists, it observes user responses to competing proposals under a shared candidate pool and display.

Engagement signals and evaluation metrics. We assign each displayed proposal an engagement score �(�) ∈ {0, 1, 2, 3} based on the observed interaction: 0 indicates no substantive engagement, 1 inspection, 2 editing or refinement, and 3 adoption.

Our primary metric is Competitive-Pair Engagement Gain, which measures the average engagement diference between proposals contributed by the two policies within competitive pairs,

$$
\Delta _ { \mathrm { C P } } = \frac { 1 } { | C | } \sum _ { c \in C } \left( e _ { c } ^ { \mathrm { o u r s } } - e _ { c } ^ { \mathrm { b a s e } } \right) ,
$$

where C denotes the set of competitive pairs. A positive value indicates greater observed engagement with proposals contributed by the personalized policy.

We additionally report Win Rate over non-tied competitive pairs. A policy wins a pair when its contributed proposal receives a higher engagement score than the competing proposal.

To complement these pairwise comparisons, we evaluate the reconstructed rankings using NDCG, treating engagement scores as graded relevance. This metric examines whether proposals associated with greater engagement appear earlier in each policy’s reconstructed ranking.

Since engagement is observed under the shared interleaved display, NDCG is interpreted descriptively. Competitivepair outcomes remain the primary comparison between ranking policies.

Finally, engagement reflects researchers’ immediate interaction with candidate directions rather than the eventual scientific outcomes of pursuing them. The deployment evaluation therefore measures usefulness for proposal exploration, not long-term scientific impact.

## H Agentic Workflow and Prompt Specifications

This appendix specifies the language-model agents and deterministic operators used to learn the three components of our framework. We follow the terminology of the main text throughout. In particular, �ˆ(�) denotes the criterion vector of research direction �, equivalently its position in the shared evaluation space. The six coordinates correspond to generality, novelty, importance, parsimony, depth, and algorithmic complexity.

Rubric A provides hindsight-informed reference criterion assessments, whereas Rubric B is the proposal-visible instrument available to the deployable criterion assessor. The construction of the multi-family hindsight reference panel is described separately in Appendix C.1, and its pairwise instantiation is given in Appendix C.2.

All language-model calls use temperature zero unless otherwise stated. Dynamic inputs are shown in square brackets. Retrieval, regression, orientation swapping, replay, clustering, predicate parsing, reliability estimation, validation, and acceptance decisions are deterministic operations rather than language-model agents.

## H.1 Evaluation Space Learning

Evaluation Space Learning estimates

$$
\hat { s } ( P ) = f _ { \mathrm { c r i t } } ( \mathcal { I } _ { \mathrm { v i s } } ( P ) ) \in [ 0 , 3 ] ^ { 6 }
$$

from proposal-visible information. Its learned student-side artifacts are the deployable Rubric B and a student-safe case memory. The complete specifications of Rubric A and Rubric B are provided in Appendix B.1 and Appendix B.2.

## H.1.1 Proposal-Time Criterion Assessor

The deployable criterion assessor is the only language-model call in the forward perception path. Before each assessment, the system deterministically retrieves the � nearest student-safe memory entries and places them in the prompt together with the current Rubric B.

## Proposal-Time Criterion Assessor

## System

You evaluate one research proposal independently on an absolute 0–3 scale along six evaluation criteria: generality, novelty, importance, parsimony, depth, and algorithmic complexity.

There is no competing proposal and no pairwise preference task. Use only proposal-visible information, the supplied evaluation rubric, and the retrieved student-safe memory. Do not use publication outcomes, citations, adoption, community response, or any information that would only become available after the proposal stage.

Return exactly one JSON object.   
User   
Proposal identifier: [PROPOSAL ID]   
[RETRIEVED STUDENT-SAFE CASE MEMORY]   
[DEPLOYMENT RUBRIC B]   
[PROPOSAL TEXT]   
For each evaluation criterion, return: score in [0, 3]; confidence in [0, 1]; a proposal-grounded comment;   
evidence\_gap; and constraint\_violation.

## H.1.2 Privileged Criterion Teacher

A deterministic error selector compares the student’s evaluation-space prediction with the Rubric-A reference and prioritizes training proposals with large criterion-level error. The selected proposal, student assessment, reference annotation, and teacher-side evidence are then passed to the privileged teacher.

System   
You are the privileged teacher for Evaluation Space Learning. You receive one research proposal, the student’s six-dimensional criterion assessment, the corresponding Rubric-A reference assessment, and privileged teacher-side evidence.   
Diagnose why the proposal-time student under- or over-estimated one or more evaluation criteria. Convert each diagnosis into reusable guidance for future proposal-time assessments.   
Privileged evidence may motivate the diagnosis, but every emitted lesson must be executable using proposal-visible information alone. Do not copy reference scores, source-paper outcomes, evidence identifiers, citations, adoption, community-response facts, or other hindsight-dependent information into the student-facing lesson.   
Teach absolute proposal-level assessment rather than pairwise preference. Return exactly one JSON object.

User

[EVALUATION-CRITERION DEFINITIONS]

[PROPOSAL TEXT]

[STUDENT SCORES, CONFIDENCE, COMMENTS, AND FLAGS]

[RUBRIC-A REFERENCE ASSESSMENT — TEACHER ONLY]

[PRIVILEGED CRITERION EVIDENCE — TEACHER ONLY]

Return lessons[]. Each lesson contains: lesson\_type; evaluation\_criterion; proposal\_visible\_trigger;

Audit and validation. Teacher-generated lessons pass through schema validation, leakage screening, deduplication, and conflict resolution before being attached to a candidate Rubric B or case memory. The candidate student state is then rerun on held-out proposals. An update is retained only when it improves the validation criterion; otherwise the rubric and memory are rolled back to the incumbent checkpoint. Leakage screening and predictive validation are performed separately: the former enforces the proposal-visible information boundary, whereas the latter prevents student-side updates from degrading held-out criterion assessment.

## H.2 Population Taste Distillation

Population Taste Distillation freezes the evaluation-space assessor and learns how evaluation-space features are translated into population-level research preferences. Population supervision is provided by the hindsight reference panel described in Appendix C.1. The deployed student can access the evaluation-space features and proposalgrounded evidence but cannot access panel labels, panel rationales, source-paper outcomes, community response, or other privileged signals.

## H.2.1 Population Taste Comparator

The deployment comparator applies the current population mapping to a proposal pair. Conditional-rule activations are computed outside the language model and supplied as fixed inputs. Given the criterion-diference representation �<sub>� �</sub>, proposal-visible evidence $\mathcal { E } _ { A , B }$ , and the current population artifacts, the comparator instantiates the preference

function defined in Section 3.4,

$$
u _ { \mathrm { p o p } } ( x _ { A , B } , \mathcal { E } _ { A , B } ) = w _ { \mathrm { p o p } } ^ { \top } x _ { A , B } + \sum _ { r \in R _ { \mathrm { p o p } } } \tilde { \alpha } _ { r } \phi _ { r } ( x _ { A , B } , \mathcal { E } _ { A , B } ) .
$$

The binary preference is determined solely by the sign of this utility:

$$
\hat { y } ( A , B ) = \left\{ \begin{array} { l l } { A , } & { u _ { \mathrm { p o p } } ( x _ { A , B } , \mathcal { E } _ { A , B } ) > 0 , } \\ { B , } & { u _ { \mathrm { p o p } } ( x _ { A , B } , \mathcal { E } _ { A , B } ) < 0 . } \end{array} \right.
$$

Thus, auxiliary outputs such as confidence, decisive criteria, and rationale do not override the sign-based preference decision.

## Population Taste Comparator

## System

You are the deployment-constrained Population Taste Comparator. Your task is to apply the learned population-level research preference policy to two research directions.

Use only the supplied evaluation-space feature vectors, proposal-grounded evidence, externally computed conditional-rule activations, and learned student-safe guidance.

Never infer or request reference-panel labels, panel rationales, source-paper outcomes, citations, adoption, community response, or any other retrospective signal. Treat all externally computed numerical quantities as fixed inputs.

Apply the supplied population mapping to obtain the scalar preference utility $u _ { \mathrm { p o p } } .$ . The preferred research direction is determined solely by the sign of this utility: positive utility selects A and negative utility selects B. Confidence, decisive criteria, and rationale are auxiliary outputs and must not alter this sign-based decision.

When the two research directions are close under the evaluation-space features, use the proposal-grounded comments, confidence values, evidence-gap flags, and constraint-violation flags only as permitted by the learned student-safe policy and conditional rules.

Return exactly one JSON object.

## User

Apply the learned population mapping to research directions A and B.   
[CRITERION-LEVEL FEATURE VECTOR AND EVIDENCE FOR A]   
[CRITERION-LEVEL FEATURE VECTOR AND EVIDENCE FOR B]   
[EXTERNALLY COMPUTED CONDITIONAL-RULE ACTIVATIONS]   
[LEARNED STUDENT-SAFE POPULATION POLICY AND CASE MEMORY]   
Return: preference\_utility; preference\_probability; winner (A or B); confidence; decisive\_criteria;   
used\_rule\_ids; and a proposal-grounded rationale.   
The returned winner must be consistent with preference\_utility: return A if preference\_utility > 0 and B if   
preference\_utility < 0.

Pair orientation swapping is used only as a deterministic robustness and validation operation. It does not participate in construction of the reference-panel label. StrictPair and related orientation-consistency statistics are likewise computed outside the language-model workflow.

## H.2.2 Privileged Population Teacher

The privileged teacher analyzes systematic student disagreements with the population reference. It may inspect panel-derived teacher-side metadata but must convert those diagnoses into instructions that remain executable under the deployment information boundary.

## Privileged Population Teacher

## System

You are the privileged teacher for Population Taste Distillation. Your task is to improve the student-safe population preference policy used by a deployment-constrained comparator.

You may inspect the population reference, teacher-side panel reasoning, decisive factors, and the reported relationship between criterion-level evidence and privileged hindsight evidence. The deployment comparator cannot inspect these signals.

Translate recurring failures into reusable instructions that can be evaluated from evaluation-space features and proposalgrounded evidence alone. Never mention citations, adoption, popularity, community response, the reference panel, or the

reference label in a deployed instruction.   
If a reference decision depends only on retrospective information for which no proposal-visible proxy exists, abstain rather   
than inventing a proxy.   
Propose several small and distinct policy edits. Each candidate should target one identifiable failure mode so that it can be   
validated independently. Return exactly one JSON object.   
User   
Round: [ROUND INDEX]   
Current population policy: [POLICY INSTRUCTIONS AND CUED MEMORY]   
Student failures: [STUDENT INPUT, A/B AND B/A DECISIONS, AND PRIVILEGED REFERENCE RECORD]   
Student successes that should not regress: [ANTI-REGRESSION CASES]   
Return candidates[]. Each candidate contains: candidate\_id; diagnosis; the complete replacement   
instructions; optional memory\_additions; optional memory\_deletions; and abstained\_pair\_ids.

Candidate packages are atomized into individual instruction or memory changes, screened for privilegedinformation leakage, and evaluated independently. The acceptance decision is made by a deterministic validation gate on the agent-validation split.

## H.2.3 Conditional-Rule Proposer

A complementary teacher path proposes executable conditional rules. Privileged information may motivate a hypothesis, but every accepted rule must reduce to a predicate over one proposal’s deployment-visible representation.

Privileged Conditional-Rule Proposer   
System   
You are a privileged hypothesis proposer for Population Taste Distillation. You do not issue the final preference judgment.   
Instead, you propose falsifiable conditional-taste hypotheses that a deterministic compiler and validation procedure will   
accept or reject.   
Privileged information may inspire a hypothesis, but every deployed predicate must be evaluable independently on one   
research direction using its evaluation-space feature vector or a registered proposal-text extractor.   
Pair-specific predicates, source-paper outcomes, citations, adoption, community response, publication venue, and   
reference-panel opinions are forbidden in deployed rules.   
User   
[CURRENT FAILURE TAXONOMY]   
[WORKED EXAMPLES OF PREVIOUSLY ACCEPTED RULES]   
[REFINEMENT HEURISTICS]   
[ERROR CLUSTER]   
For each case, you receive:   
• evaluation-space features and proposal-grounded evidence;   
• the current model prediction and margin;   
• the population-reference preference and teacher-side rationale; and   
• privileged retrospective signals.   
Propose a small set of nonredundant hypotheses. Return hypotheses[], each containing: predicate; tier; direction   
in {−1, +1}; failure\_mode; a deployment-safe explanation; and a teacher-only privileged\_cue.

Tier-1 predicates are arithmetic or Boolean functions of the six evaluation-space features and their deploymentvisible evidence flags. Tier-2 predicates invoke previously registered semantic extractors over proposal-visible text. In both cases, a rule � defines a single-direction activation $a _ { r } ( P )$ . For a proposal pair (�, �), the rule enters the comparison only through the activation diference $a _ { r } ( A ) - a _ { r } ( B )$ . This preserves the general single-direction population mapping while instantiating it under the pairwise observation interface.

## H.2.4 Proposal-Text Property Evaluator

Tier-2 predicates use a frozen proposal-text evaluator.

Property: [REGISTERED PROPERTY DESCRIPTION]   
Proposal-grounded text: [PROPOSAL OR PROPOSAL-GROUNDED EVIDENCE RECORD]   
If the supplied text does not clearly establish the property, return holds=false.

Evaluation-Space Personal Rule Proposer   
System   
You diagnose why a personalized scientific-taste model still mispredicts a specific researcher’s choices after a six  
dimensional weight residual has already been fitted.   
Continuous criterion reweighting has therefore already been attempted. Do not propose that an evaluation criterion should   
simply receive more or less global weight. Instead, propose a conditional rule that activates only under a repeated pattern   
shared by the remaining errors.   
Each proposed trigger must be evaluable independently on one research direction from its evaluation-space feature vector.   
State only the preferred direction of the efect; rule strength will be estimated statistically outside the language model.

## Proposal-Text Property Evaluator

## System

Judge one research proposal against one precisely stated, present-tense property. Do not compare it with another proposal and do not speculate about future impact, adoption, reception, or other hindsight outcomes. Return exactly one JSON object containing: holds; a proposal-grounded evidence span; and confidence.

The property description, evaluator model, and prompt are frozen when the extractor is registered and become part of its cache key.

## H.2.5 Workflow Reflection

The executable-rule path additionally uses a reflection agent that receives only aggregate workflow statistics. It cannot observe evaluation examples or approve its own revisions.

Population-Rule Workflow Reflection   
System   
You improve the conditional-rule generation workflow using aggregate round-level statistics.   
You may revise the proposer preamble, worked examples, failure taxonomy, refinement heuristics, or case-selection strategy.   
You may not revise the predicate grammar, information-boundary contract, validation criterion, or output schema.   
You never observe the evaluation examples themselves.   
User   
[NUMBER OF PROPOSED AND TYPE-CHECKED HYPOTHESES]   
[VALIDATION SURVIVAL RATE AND HELD-OUT GAIN]   
[CLUSTER COVERAGE AND REDUNDANCY]   
[COMMON REJECTION REASONS]   
[CURRENT PROPOSER ARTIFACTS]   
Diagnose the largest current weakness and propose one bounded revision. Return: diagnosis; change\_kind; the   
corresponding replacement or addition; and change\_note.

The incumbent and revised proposer configurations are subsequently evaluated under identical budgets by deterministic operators. The reflection agent cannot approve its own revision.

## H.3 Individual Taste Adaptation

The numerical adaptation and replay gate are described in Appendix D. This section records the language-model prompts used only to propose conditional personal rules and proposal-text properties after residual fitting leaves recurring support errors.

## H.3.1 Evaluation-Space Personal Rule Proposer

The first agentic rule path searches for conditional preferences expressible within the learned evaluation space.

A rule that activates almost everywhere is a disguised global weight update. A rule that activates only on one example is   
an anecdote. Return only predicates under the frozen executable grammar.   
User   
Researcher: [USER ID]   
Fitted individual weighting: [POPULATION WEIGHTS PLUS PERSONAL RESIDUAL]   
Cross-fitted support errors that remain after residual fitting: [PER-SIDE EVALUATION-SPACE FEATURES, USER   
CHOICE, MODEL CHOICE, AND MARGIN]   
Shared trigger pattern: [PATTERN]   
Propose at most [BUDGET] conditional rules.

This proposer is invoked only after suficient individual support has accumulated and a repeated residual-error pattern can be identified.

## H.3.2 Proposal-Text Personal Property Proposer

If no evaluation-space conditional rule survives replay, the second agentic path may propose a proposal-visible property outside the representational span of the six predefined evaluation criteria.

Proposal-Text Personal Property Proposer   
System   
A personalized criterion-weight residual and evaluation-space conditional rules have already failed to explain the remaining   
choices. Propose a reusable property of one research proposal’s text that captures the unresolved taste pattern.   
The property must describe a dimension of research preference rather than a research topic, proposal identity, or pair   
identifier. It must describe what the proposal presently states, not its future impact, adoption, reception, or other hindsight   
outcome.   
User   
Researcher: [USER ID]   
Fitted individual weighting: [WEIGHTS]   
Remaining support errors: [EVALUATION-SPACE FEATURES, PROPOSAL-GROUNDED TEXT, USER CHOICES,   
AND OPTIONAL USER-PROVIDED RATIONALES]   
Propose at most [BUDGET] properties. Each property must:   
• be answerable for one research direction independently;   
• plausibly generalize across research topics;   
• separate a nontrivial subset of the support interactions rather than activating universally or on one proposal only; and   
• avoid citations, adoption, reception, hidden user information, and other hindsight-dependent signals.   
Return extractors[], each containing: name; description; direction; explanation; and why\_these\_cases.

## H.3.3 Property Rewrite Agent

If a proposed proposal-text property fails the statistical gate, a rewrite agent may inspect only the rejection diagnostics and the same observed support history. No additional labels are revealed.

Personal Property Rewrite Agent   
System   
A previously proposed personal-property extractor failed statistical validation. Propose a substantively diferent taste   
property rather than renaming the failed property.   
User   
[REMAINING CROSS-FITTED ERRORS]   
[FAILED PROPERTY DESCRIPTIONS]   
[ACTIVATION COUNTS, FIXED/BROKEN COUNTS, NLL CHANGE, AND REJECTION REASONS]   
Propose new present-tense, per-proposal properties that separate the unresolved cases and activate on enough of the   
complete support bufer to estimate their efect.   
Topic labels, pair memorization, hindsight outcomes, and hidden researcher information are forbidden. Return the same   
extractor schema as the initial proposal-text property proposer.

## H.4 Deterministic Components

For clarity, the following operations contain no language-model prompt: nearest-neighbor case retrieval, error ranking, logistic fitting of $\boldsymbol { w } _ { \mathrm { p o p } }$ and $\Delta w _ { u }$ , proposal-order swapping, StrictPair computation, predicate parsing, rule-activation computation, coeficient fitting, reliability updates, cross-fitted replay, policy atomization, leakage screening, validation-gate decisions, rollback, and held-out evaluation.

This separation ensures that language-model agents propose or diagnose candidate textual artifacts, whereas acceptance and numerical parameter estimation remain externally controlled.