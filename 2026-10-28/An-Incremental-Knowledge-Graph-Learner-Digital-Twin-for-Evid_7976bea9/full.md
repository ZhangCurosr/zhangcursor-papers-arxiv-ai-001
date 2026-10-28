# An Incremental Knowledge-Graph Learner Digital Twin for Evidence-Grounded Early Warning

Ngo<sub>.</sub> c Luyê<sub>.</sub>n Lê<sup>1,2[0000−0002−2043−0679]</sup>, Marie-Hélène Abel<sup>2[0000−0003−1812−6763]</sup>, and Bertrand Laforge<sup>3[0000−0002−4209−4194]</sup>

<sup>1</sup> Gamaizer, 93340 Le Raincy, France

2 Université de technologie de Compiègne, CNRS, Heudiasyc (Heuristics and Diagnosis of Complex Systems), CS 60319, 60203 Compiègne Cedex, France

{ngoc-luyen.le,marie-helene.abel}@hds.utc.fr

3 Sorbonne Université, CNRS UMR 7585, LPNHE (Laboratoire de Physique Nucléaire et des Hautes Énergies), 75252 Paris Cedex 05, France aforge@lpnhe.in2p3.fr

Abstract. Early-warning models in learning analytics can identify learners at risk of failure or withdrawal, but the resulting alerts are often only weakly connected to inspectable learning evidence. A risk score may indicate that intervention is needed without showing which traces, assessment events, or temporal patterns justify the alert. This paper introduces an incremental Learner Digital Twin for evidence-grounded early warning, implemented as a lightweight knowledge graph. The twin is updated at temporal checkpoints by replaying learning traces and assessment records, and it serves both as a feature-extraction substrate and as an explanation-grounding substrate. Predictive features are derived through graph-mediated queries, while features, attributions, alerts, evidence items, and raw records are represented as queryable graph entities. The framework is instantiated and evaluated on a public learninganalytics benchmark under a temporal protocol. The results show that graph-mediated feature extraction preserves the predictive behavior of equivalent raw-table features across model families and checkpoints. Generated alerts are materialized as evidence chains linking risk decisions to feature attributions, evidence items, and raw platform records. The study shows that early-warning prediction can be routed through a semantic and provenance-aware representation while preserving predictive performance and adding traceability, verifiability, and evidence-grounded explanation.

Keywords: Learner digital twin · Learning analytics · Early warning · Knowledge graph · Explainability · Provenance

## 1 Introduction

Web-based learning platforms continuously record learner activity, including resource accesses, submissions, scores, registration events, assessment results, and withdrawals. Learning analytics has built on these traces to develop earlywarning systems that estimate, before the end of a course, which learners may be at risk of failing or dropping out [1, 12]. Such systems can support timely pedagogical intervention, but their usefulness depends on more than predictive accuracy. For an alert to be actionable, instructors also need to understand which learning traces, assessment events, or temporal patterns justify the risk signal.

The predictive dimension of early-warning learning analytics is now relatively well served. Tree-based ensembles, gradient-boosting methods, and neural architectures can achieve strong discrimination when trained on interaction and assessment features [19, 31]. However, the explanatory status of the resulting alerts remains limited. In a typical pipeline, raw platform logs are transformed into tabular features, a predictive model assigns a risk score, and a post-hoc attribution method such as SHAP [24] may indicate that an engineered variable, for example total\_clicks, contributed to the prediction. Such feature-level explanations are useful for model inspection, but they often stop short of what instructors need for pedagogical interpretation and action. An instructor-facing alert should not only indicate that a learner is at risk; it should also make explicit which learning evidence supports this claim. A feature attribution names an engineered quantity, but it does not necessarily identify the concrete learning events behind it: which interactions decreased, which assessment was missed, which raw records substantiate the alert, and whether the explanation can be queried, traced, or contested. This creates a gap between predictive performance and evidence-grounded interpretability, a limitation emphasized in explainable learning analytics and AI-in-education research [17].

We address the following problem: given a stream of timestamped learning traces and assessment records, how can an early-warning system produce risk alerts that are not only predictive, but also evidence-grounded, queryable, and traceable? To this end, we introduce an incremental Learner Digital Twin (ldt) implemented as a lightweight knowledge graph (KG). The twin is updated at successive temporal checkpoints by replaying learning traces and assessment data in chronological order, thereby simulating the recurrent synchronization that a ldt would perform in a live learning platform. The central design principle is that the graph is not a post-hoc explanatory layer added after prediction; instead, it acts as the operational substrate for feature construction, prediction, attribution representation, and evidence verification. At each checkpoint, learner feature vectors are extracted from the twin using a graph-mediated query catalogue and passed to an early-warning classifier. The resulting prediction features, feature attributions, generated alerts, and evidence items are then materialized back into the graph, with each evidence item linked to raw learning records following a provenance-aware modeling perspective inspired by PROV-O [22]. Graph integrity is checked using SHACL validation [18], so that explanation chains can be verified as graph properties. The framework is later instantiated and evaluated on the Open University Learning Analytics Dataset (OULAD) [19].

The objective is not to improve predictive accuracy, but to show that earlywarning prediction can be routed through a semantic and provenance-aware representation while preserving predictive performance and making alerts evidencegrounded, queryable, and traceable. The paper makes three contributions: it introduces an incremental knowledge-graph ldt synchronized at temporal checkpoints from learning traces and assessment data; it represents predictive features, feature attributions, alerts, evidence items, and raw records as first-class graph entities; and it empirically verifies that graph-mediated feature extraction preserves the behavior of an equivalent raw-table pipeline while enabling SHACL-validated evidence grounding and traceability.

The remainder of this paper is organized as follows. Section 2 reviews related work on early warning, explainable learning analytics, learner digital twins, and semantic provenance. Section 3 presents the knowledge-graph ldt. Section 4 describes the experimental setup and results. Section 5 discusses implications and limitations. Section 6 concludes the paper.

## 2 Related Work

We review evidence-grounded early warning in learning analytics. We focus on three areas: early-warning and explainable learning analytics, learner modeling and digital twins, and knowledge graphs for provenance and traceability.

Early-warning systems are among the most established applications of learning analytics. They use traces of learner activity, assessment behavior, registration information, and other platform data to identify learners who may fail, withdraw, or require support before the end of a course. Operational systems such as Course Signals [2] and OU Analyse [13] illustrate how predictive learning analytics has been deployed as institutional decision support. In parallel, public benchmark datasets have enabled reproducible experimentation on early risk prediction using activity and assessment traces [19]. Related work has also examined leakage-excluded early-warning prediction under strict temporal constraints [21] and semantics-constrained counterfactual recourse for educational decision support [20]. Most work in this area has focused on predictive performance, timeliness, and deployment feasibility.

Explainability is particularly important in learning analytics because predictions may inform pedagogical decisions, learner support, or institutional interventions. Post-hoc explanation methods such as LIME and SHAP are commonly used because they attribute individual predictions to input features without requiring a fully transparent model architecture [27, 24]. However, a feature-level attribution remains incomplete if the instructor cannot inspect which resources were accessed, which assessment was missed, or which raw records substantiate the feature. The issue is therefore not only whether an attribution is locally faithful to the model, but whether the attributed feature is grounded in interpretable educational evidence.

Learner modeling has long addressed the representation of learner states. Overlay models, open learner models, and adaptive educational systems represent learners’ knowledge, progress, preferences, goals, and interaction histories [4]. Ontology-based learner models further contribute semantic structure and interoperability by representing learners, contexts, resources, and educational situations through explicit conceptual models [28]. However, many learner models are built from a given snapshot of learner data or are used primarily as profiles for adaptation and visualization. They are less often operationalized as time-indexed representations that are recurrently synchronized with platform traces and directly connected to predictive alert generation.

The notion of a digital twin originated in product lifecycle management, where it refers to a virtual counterpart of a physical asset connected to the real system through data flows [10]. Work in smart manufacturing emphasizes that a digital twin is not merely a static model, but a data-driven representation updated from observations and used to support analysis, simulation, and decision-making [29, 30]. When transposed to education, the twinned entity may be a learner, a course, or a learning environment [11]. In this paper, we use the term in a restricted operational sense: a checkpoint-indexed learner representation synchronized by replayed platform observations and used to support prediction and evidence-grounded explanation.

Ontologies and semantic web technologies have been used to structure learner information, educational resources, competencies, contexts, and learning activities. Early standards such as IMS LIP provided foundations for learner profile interoperability [15]. Educational knowledge graphs and linked learning resources have also shown how semantic representations can support search, recommendation, integration, and reasoning over educational data [5, 7]. These approaches demonstrate the value of semantic modeling, but they generally focus on representing learners, resources, activities, or competencies rather than the internal objects produced by a predictive early-warning pipeline.

Provenance modeling provides the conceptual and technical basis for connecting analytic outputs to their sources. The W3C PROV model represents entities, activities, agents, and derivation relations, allowing data products to be linked to the processes and sources from which they were generated [22, 25]. SHACL complements this perspective by providing a standard mechanism for validating structural constraints in RDF graphs [18]. In early-warning learning analytics, however, provenance is often associated with datasets or processing workflows, while the explanatory structure of individual alerts remains weakly represented. Existing approaches rarely integrate learner states, predictive features, attributions, alerts, evidence, and raw records into a single representation that can be jointly queried and traced.

## 3 Methodology

This section presents the incremental Learner Digital Twin for evidencegrounded early warning. We define the temporal prediction problem, introduce the checkpoint-indexed graph, and describe the ontology, synchronization, feature extraction, prediction, attribution mapping, and evidence grounding.

## 3.1 Problem Formulation

We consider an early-warning setting in which predictions are produced at several moments during a course presentation, before the final learner outcome is known. Let L denote the set of learners enrolled in a given presentation, and let D denote the set of timestamped raw records produced by the learning platform. These records may include learner–resource interaction records, assessment-result records, registration events, and other platform events relevant to learner activity. Each record $d \in D$ is associated with a timestamp t(d), expressed in the temporal unit used by the platform, such as days or weeks relative to the beginning of the presentation.

Given an ordered sequence of temporal checkpoints $\tau _ { 1 } < \tau _ { 2 } < \cdots < \tau _ { n }$ , the data available at checkpoint τ are defined as

$$
D _ { \leq \tau } = \{ d \in D \mid t ( d ) < \tau \} .\tag{1}
$$

The strict inequality enforces the temporal constraint of the early-warning task: at checkpoint $\tau ,$ , the system may only use records that would have been observed before that time. This prevents information from future interactions, future assessment results, or final outcomes from leaking into the prediction process.

For each learner $\ell \in \mathcal L$ , the target label is defined from the final course outcome:

$$
y _ { \ell } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ l e a r n e r ~ } } \ell { \mathrm { ~ h a s ~ a n ~ u n s u c c e s s f u l ~ f i n a l ~ o u t c o m e } } , } \\ { 0 , } & { { \mathrm { i f ~ l e a r n e r ~ } } \ell { \mathrm { ~ h a s ~ a ~ s u c c e s s f u l ~ f i n a l ~ o u t c o m e } } . } \end{array} \right. }\tag{2}
$$

The precise mapping between institutional outcome categories and the binary label is dataset-dependent and is specified in the experimental setup. Although $y _ { \ell }$ is used for training and evaluation, it is not available to the model at prediction time. At checkpoint $\tau _ { i }$ , the model must therefore estimate the risk of learner ℓ using only the temporally bounded data $D _ { \leq \tau _ { i } }$

Let $G _ { \tau _ { i } }$ denote the ldt graph constructed from the records available before checkpoint $\tau _ { i } ,$ as defined in Sect. 3.2. For each active learner ℓ at $\tau _ { i } ,$ a featureextraction function derives a checkpoint-specific feature vector from the graph:

$$
x _ { \ell , \tau _ { i } } = \psi _ { \mathcal { Q } } ( G _ { \tau _ { i } } , \ell ) ,\tag{3}
$$

where $\psi _ { \mathcal { Q } }$ denotes the feature-extraction process implemented by the SPARQL query catalogue $\mathcal { Q }$ . The early-warning model is a predictive function

$$
{ \hat { f } } _ { \theta } : \mathbb { R } ^ { p }  [ 0 , 1 ] ,\tag{4}
$$

which maps $x _ { \ell , \tau }$ to a risk probability:

$$
\hat { p } _ { \ell , \tau _ { i } } = \hat { f } _ { \boldsymbol { \theta } } ( x _ { \ell , \tau _ { i } } ) .\tag{5}
$$

A risk alert is emitted when the estimated probability exceeds a decision threshold $\gamma \colon$

$$
a _ { \ell , \tau _ { i } } = \mathbb { I } \big [ \hat { p } _ { \ell , \tau _ { i } } \ge \gamma \big ] ,\tag{6}
$$

where $a _ { \ell , \tau _ { i } } = 1$ denotes that an alert is generated for learner ℓ at checkpoint $\tau _ { i }$

The problem addressed in this paper is not only to estimate $\hat { p } _ { \ell , \tau _ { i } }$ , but also to preserve the evidential basis of the prediction. For each generated alert, the system should represent the predictive features used by $\hat { f } _ { \theta }$ , their feature-level attributions, the evidence supporting these features, and the raw records from which the evidence was derived. Formally, for an attributed feature $f _ { j }$ , the explanation is considered evidence-grounded only if there exists at least one evidence item e and one raw record r such that

$$
s u p p o r t e d B y ( f _ { j } , e ) \ \wedge \ d e r i v e d F r o m ( e , r ) \ \wedge \ R a w R e c o r d ( r ) .\tag{7}
$$

Thus, the methodological objective is twofold: first, to construct temporally valid predictive features from the learner state available at each checkpoint; second, to maintain explicit, queryable links from alerts to attributions, from attributions to features, from features to evidence, and from evidence to raw platform records.

![](images/f4bc6e57f300713e090246891b07dac607d60955028be4c98daa3643249f00f4.jpg)  
Fig. 1. Generic workflow of the incremental ldt for evidence-grounded early warning.

Consider, for example, the temporal data structure illustrated in the first stage of Fig. 1. At checkpoint $\tau _ { i }$ corresponding to day 25, only records observed before this day are available for analysis. These may include earlier interaction traces, registration information, and assessment results already recorded before the checkpoint. $\mathrm { B y }$ contrast, records occurring after day 25, future assessment results, and the final course outcome are excluded from prediction and explanation. The learner representation at $\tau _ { i }$ must therefore reflect only what the platform would have known at that moment. This makes the problem both predictive and evidential: the system must estimate learner risk while preserving traceable links to the data available at prediction time.

## 3.2 Overview of the Incremental Learner Digital Twin

Definition 1 (Learner Digital Twin). A Learner Digital Twin is a timeindexed digital representation of a learner, synchronized from observed learningplatform data to support analysis, prediction, explanation, and traceability. Following the general view of digital twins as data-driven representations connected to observed systems [10, 29, 30], and their adaptation to educational settings [11], this paper operationalizes the learner twin as a checkpoint-indexed knowledge graph $G _ { \tau _ { i } }$

Building on this definition, the ldt represents, at each temporal checkpoint $\tau _ { i } ,$ both the learner state derived from temporally available platform records and the analytic artifacts generated by the early-warning pipeline. The Learner Digital Twin at checkpoint $\tau _ { i }$ is defined as

$$
G _ { \tau _ { i } } = \underbrace { \mu ( D _ { \leq \tau _ { i } } ) } _ { \mathrm { o b s e r v a t i o n ~ l a y e r } } \cup \underbrace { \varPhi _ { \tau _ { 1 } } \cup \dots \cup \varPhi _ { \tau _ { i } } } _ { \mathrm { a n a l y t i c ~ l a y e r } } ,\tag{8}
$$

where $\mu$ is a declarative mapping from raw platform records to RDF triples over the ldt ontology, and $\varPhi _ { \tau _ { j } }$ denotes the set of analytic triples materialized at checkpoint $\tau _ { j }$ . The observation layer captures the learner state available before $\tau _ { i } .$ while the analytic layer stores the objects produced by the early-warning pipeline, including prediction features, feature attributions, risk alerts, evidence items, and prediction-run metadata.

The observation layer grows monotonically as the course progresses, so that $( \mu ( D _ { \leq \tau _ { i } } ) \subseteq \mu ( D _ { \leq \tau _ { i + 1 } } ) )$ . The analytic layer is checkpoint-versioned: features, attributions, and alerts generated at an earlier checkpoint are not overwritten by later computations, but remain linked to the corresponding Checkpoint entity. This design makes it possible to query the analytic history of the twin, for example to determine which learner state, feature values, and evidence items were available when a specific alert was generated.

Figure 1 summarizes the workflow implemented at each checkpoint. Temporally available records are first mapped into the ldt graph. The graph is then queried to construct checkpoint-specific feature vectors, which are passed to the predictive function. If the estimated risk exceeds the alert threshold, a risk alert is generated. The alert is then linked to feature attributions, predictive features, evidence items, and the raw records from which the evidence was derived. Thus, the same graph-based representation supports both prediction and traceability.

## 3.3 Ontology and Evidence Path

The ldt ontology is designed as a lightweight and task-oriented model for evidence-grounded early warning. It does not aim to cover all aspects of learner modeling or educational knowledge representation. Instead, it contains the classes and properties required to represent the entities involved in the early-warning workflow: platform observations, predictive features, attributions, alerts, evidence items, checkpoints, and raw records. Consistent with Eq. (8), the ontology is organized into an observation layer and an analytic layer.

The observation layer represents entities directly derived from platform data, including Learner, Course, LearningResource, LearningTrace, Assessment, AssessmentResult, and RegistrationEvent. Each observation-layer entity derived from a platform row is linked to a RawRecord individual through ldto:fromRecord. The RawRecord class stores the source table or stream name and record identifier, thereby anchoring the graph in the original data.

The analytic layer represents the objects produced by the early-warning pipeline. It includes PredictionFeature, FeatureAttribution, RiskAlert, EvidenceItem, PredictionRun, and Checkpoint. A PredictionFeature represents one feature value for one learner at one checkpoint and is linked to the evidence items that support its computation. A FeatureAttribution stores the contribution of a feature to a specific alert. A RiskAlert represents the generated risk decision and is linked to the prediction run, the corresponding checkpoint, the relevant attributions, and the evidence selected for presentation.

An EvidenceItem represents a typed and human-readable statement supporting a predictive feature. It may correspond to activity evidence, assessment evidence, temporal evidence, absence evidence, or another evidence type defined by the application. Each evidence item carries a textual description, a strength value, a temporal scope, and one or more links to RawRecord individuals through ldto:derivedFrom. The ontology aligns selected classes and properties with PROV-O: evidence items, raw records, prediction features, feature attributions, and alerts are modeled as provenance entities, while prediction runs are modeled as provenance activities.

![](images/3d7c4be53fc289efd58b3f63fe3649ec852c7e01b645d6819e709977038ffa58.jpg)  
Fig. 2. Core ldt ontology: observation layer, analytic layer, evidence links.

The explanation of a generated alert is represented as the following graph path:

$$
\begin{array} { r l } & { R i s k A l e r t ~ \frac { h a s A t t r i b u t i o n } { \gamma } ~ F e a t u r e A t t r i b u t i o n ~ \xrightarrow { r e f e r s T o } ~ P r e d i c t i o n F e a t u r e } \\ & { ~ \xrightarrow { s u p p o r t e d B y } ~ E v i d e n c e I t e m ~ \xrightarrow { d e r i v e d F r o m } ~ R a w R e c o r d . } \end{array}\tag{9}
$$

This path is the central evidence-grounding structure of the KG-based ldt. It makes the explanation chain queryable, verifiable, and traceable through standard graph queries.

## 3.4 Synchronization, Feature Extraction, and Alert Grounding

At each checkpoint $\tau _ { i } ,$ the replay scheduler exposes only records available before the checkpoint to the RDF mapping process. The set of newly available records between two consecutive checkpoints is denoted by

$$
\begin{array} { r } { \varDelta _ { i } = D _ { \leq \tau _ { i } } \backslash D _ { \leq \tau _ { i - 1 } } . } \end{array}\tag{10}
$$

Only records in $\varDelta _ { i }$ are added to the observation layer during the update from $G _ { \tau _ { i - 1 } }$ to $G _ { \tau _ { i } }$ . Assessment-related entities are included only when their timestamp or due date satisfies the temporal constraint. This prevents future assessment information from entering the learner state or the predictive features.

The graph is extended with newly available observation-layer triples, then validated against SHACL shapes before analytic write-back. The shapes encode the structural conditions required by the explanation mechanism: each LearningTrace must be linked to a learner, resource, timestamp, and RawRecord; each PredictionFeature must have a learner, checkpoint, name, and value; each EvidenceItem must be derived from at least one RawRecord; and each RiskAlert must be generated by one PredictionRun and justified by evidence.

Algorithm 1 summarizes the checkpoint update process. Feature values are extracted from the graph, materialized as PredictionFeature entities, and linked to supporting evidence. The predictive model scores each active learner, and

Algorithm 1 Checkpoint update of the Learner Digital Twin   
Require: twin $G _ { \tau _ { i - 1 } } ,$ new records ∆ , shapes S, trained classifier ${ \hat { f } } _ { \theta } ,$ feature queries Q, threshold   
1: $\underset { G } { \gamma }  G _ { \tau _ { i - 1 } } \cup \mu ( \Delta _ { i } )$   
2: assert S $\operatorname { I A C L } ( G , S ) = \varnothing$   
3: for all active learner ℓ at $\tau _ { i }$ do   
4: $x _ { \ell , \tau _ { i } } \gets { \psi _ { Q } ( G , \ell ) }$   
5: materialize PredictionFeature nodes and evidence links   
6: $\hat { p } _ { \ell , \tau _ { i } } \gets \hat { f } _ { \theta } ( x _ { \ell , \tau _ { i } } )$   
7: $\phi _ { \ell , \tau _ { i } } \gets$ Explain $( \hat { f } _ { \theta } , x _ { \ell , \tau _ { i } } )$   
8: if $\hat { p } _ { \ell , \tau _ { i } } \geq \gamma$ then   
9: materialize RiskAlert and FeatureAttributions   
10: select and link presented evidence through justified $B y$   
11: assert $\operatorname { S H A C L } ( G , S ) = \varnothing$   
12: return $G _ { \tau _ { i } }$

attributions, alerts, and evidence links are written back into the analytic layer.   
The output of one checkpoint becomes the input graph for the next checkpoint.

Overall, the methodology defines a checkpoint-indexed ldt in which temporally available records are mapped to an observation graph, predictive features are extracted from this graph, and generated alerts are linked back to feature attributions, evidence items, and raw records through the evidence path in Eq. (9). The next section describes how this framework is instantiated and evaluated on a public learning-analytics benchmark.

## 4 Experimentation

Our experimental study addresses three research questions: RQ1 whether the ldt can be incrementally constructed and validated at temporal checkpoints; RQ2 whether graph-mediated feature extraction preserves the predictive behavior of an equivalent raw-table feature pipeline; and RQ3 whether generated alerts are grounded in traceable evidence chains.

The experimental instantiation uses a compact feature catalogue covering activity indicators, resource-use counts, assessment progress, temporal behavior, and absence-based signals such as no activity or no submitted assessment before a checkpoint. The same feature definitions are used in both experimental variants.

## 4.1 Dataset and Temporal Protocol

The empirical evaluation uses OULAD [19] as a public benchmark for earlywarning learning analytics. The dataset covers 22 presentations of 7 modules, 32,593 students, and about 10.6 million VLE interaction records. Its day-based temporal structure and its inclusion of registrations, assessments, and final outcomes make it suitable for early-warning prediction and evidence-grounded alert generation.

Following common practice, learners who withdrew before the presentation start (date\_unregistration $\leq 0 )$ are excluded, since no in-course behavior exists to twin. Learners with no VLE activity and no assessment record before the first checkpoint are retained; their emptiness is itself predictive and generates absence evidence. After preprocessing, six modules have at least one 2013 presentation for training and one 2014 presentation for testing. Module CCC is excluded because it has no 2013 presentation in the filtered protocol.

Table 1. OULAD protocol and ontology-mapping summary.
<table><tr><td>Element</td><td>Setting</td></tr><tr><td>Dataset</td><td>OULAD; six eligible modules after preprocessing</td></tr><tr><td>Train/test</td><td>2013 presentations for training; 2014 presentations for testing</td></tr><tr><td>Checkpoints</td><td>Day 28 (Week 4), day 56 (Week 8), day 84 (Week 12)</td></tr><tr><td>Target</td><td>Fail/Withdrawn = at-risk; Pass/Distinction = non-risk</td></tr><tr><td>Observation tables</td><td>studentVle, studentAssessment, studentRegistration</td></tr><tr><td></td><td>Mapped observation entities LearningTrace, AssessmentResult, RegistrationEvent, RawRecord</td></tr><tr><td>Analytic entities</td><td>PredictionFeature, FeatureAttribution, RiskAlert, EvidenceItem</td></tr><tr><td>Evaluation</td><td>Paired raw-table/KG features; mean over six modules</td></tr></table>

We adopt a presentation-based temporal protocol: for each module, the classifier is trained on the 2013 presentations and tested on the 2014 presentations of the same module, simulating deployment on the next cohort. The checkpoints are day 28 (Week 4), day 56 (Week 8), and day 84 (Week 12). Final outcomes are binarized by mapping Fail and Withdrawn to the at-risk class, and Pass and Distinction to the non-risk class.

## 4.2 Baselines and Evaluation Metrics

The experiment evaluates seven paired model families at each checkpoint: XGBoost (XGB) [6], Random Forest (RF) [3], Extra Trees (ET) [9], HistGradientBoosting (HGB) [8], LightGBM (LGBM) [16], CatBoost (CAT) [26], and a numeric TabTransformer-style neural model (TABTX) [14]. Each family is run twice: once on conventional raw-table features and once on knowledgegraph-mediated features extracted from the twin. The ontology, SHACL shapes, SPARQL queries, scripts, and per-module results are publicly available<sup>4</sup>.

The conventional and KG-mediated catalogues compute the same feature definitions; what difers is the substrate and the provenance machinery. XGB-ldt is used as the alert-materialization configuration for the explanation-grounding analysis because TreeSHAP attributions are stable, eficient, and naturally aligned with the feature-attribution graph [23]. Unless otherwise stated, alerts are generated with a fixed threshold $\gamma = 0 . 5$ to evaluate evidence materialization under a simple reproducible alert policy; deployment-specific threshold calibration is left for future work.

For predictive performance, we compute precision, recall, F1 on the at-risk class, and AUC at each checkpoint. To keep the main benchmark table compact, Table 3 reports F1 and AUC, while precision and recall are retained in the public artifact. The success criterion is parity: for each paired model family, the KGmediated variant should match the corresponding raw-table variant when both use the same feature definitions and the same tuning budget.

For incremental construction, we report the number of updated learners, newly integrated traces and assessment results, newly added triples, cumulative graph size, and SHACL violations [18]. For explanation grounding, we define three task-specific metrics over the generated alerts ${ \mathcal { A } } .$ Let $T _ { k } ( a )$ be the top-(k) attributed features for alert $a , E ( f )$ the evidence items linked to feature $f ,$ and $R ( e )$ the raw records linked to evidence item e through derivedFrom [22]. Evidence Coverage@(k) measures whether top-(k) features are supported by

Table 2. Incremental twin construction per checkpoint; mean over 6 test modules.
<table><tr><td></td><td></td><td></td><td></td><td>Checkpoint Learners New traces New results New triples Cum. traces Cum. triples SHACL viol.</td><td></td><td></td><td></td></tr><tr><td>Week 4</td><td>2,210</td><td>148,560</td><td>1,386</td><td>1,349,804</td><td>148,560</td><td>1,367,486</td><td>0</td></tr><tr><td>Week 8</td><td>2,210</td><td>106,204</td><td>1,674</td><td>970,948</td><td>254,763</td><td>2,523,586</td><td>0</td></tr><tr><td>Week 12</td><td>2,210</td><td>66,638</td><td>950</td><td>608,326</td><td>321,401</td><td>3,278,846</td><td>0</td></tr></table>

evidence reaching raw records:

$$
\mathrm { C o v @ } k = \frac { 1 } { | \mathcal { A } | } \sum _ { a \in \mathcal { A } } \frac { 1 } { k } \sum _ { f \in T _ { k } ( a ) } \mathbb { I } \big [ \exists e \in E ( f ) , \exists r \in R ( e ) \big ] .\tag{11}
$$

Presented $F i d e l i t y @ ( k )$ measures whether presented evidence supports features among the top-(k) model drivers:

$$
\mathrm { F i d @ } k = \frac { 1 } { | \mathcal { A } | } \sum _ { a \in \mathcal { A } } \frac { | P ( a ) \cap T _ { k } ( a ) | } { | P ( a ) | } ,\tag{12}
$$

where $P ( a )$ is the set of features supported by the evidence presented for alert a. Traceability is the fraction of evidence items linked to at least one raw record:

$$
{ \mathrm { T r a c e a b i l i t y } } = { \frac { | \{ e \in { \mathcal { E } } \mid \exists r \in R ( e ) \} | } { | { \mathcal { E } } | } } .\tag{13}
$$

We report direct coverage for feature-specific activity or assessment records, and contextual coverage for absence or inactivity features grounded in records.

## 4.3 Results

All results reported in this section are obtained from the OULAD instantiation described above.

RQ1 - Incremental construction and validation: Table 2 reports the incremental construction results, averaged over the six test modules. The test graph contains 2,210 learners on average per module. The number of newly integrated traces decreases over time because a large share of the available activity occurs before the first checkpoint: the update adds on average 148,560 traces at Week 4, 106,204 at Week 8, and 66,638 at Week 12. The persistent graph nevertheless grows monotonically, reaching 321,401 cumulative learning traces and 3.28M triples on average by Week 12. Both observation graphs and analytic graphs conform to the SHACL shapes at all checkpoints, with zero violations.

These results show that the twin can be maintained incrementally while preserving the structural invariants required for traceable explanations. The graph grows as expected with newly available learning traces and assessment results, while SHACL validation confirms that generated evidence chains remain structurally valid.

RQ2 - Predictive parity: Table 3 reports the KG-mediated early-warning performance on OULAD. The corresponding raw-table results are omitted for compactness because they are identical to the KG-mediated results for every model, checkpoint, and metric. Performance improves from Week 4 to Week 12 as more activity and assessment evidence becomes available. The strongest KG-mediated model is TABTX at Week 4 (AUC 0.735), XGB at Week 8 (AUC 0.809), and ET at Week 12 (AUC 0.847). The central result is therefore feature-substrate parity: graph-mediated feature extraction preserves the predictive behavior of the equivalent raw-table pipeline while adding provenance, validation, and evidence materialization.

Table 3. KG-mediated early-warning performance over six modules; raw-table results are identical and omitted for compactness.
<table><tr><td></td><td colspan="2">Week 4</td><td colspan="2">Week 8</td><td colspan="2">Week 12</td></tr><tr><td>Method KG F1 KG</td><td colspan="6">AUC KG F1 KG AUC KG F1 KG AUC</td></tr><tr><td>XGB</td><td>0.610</td><td>0.735</td><td>0.680</td><td>0.809</td><td>0.718</td><td>0.842</td></tr><tr><td>RF</td><td>0.597</td><td>0.725</td><td>0.653</td><td>0.795</td><td>0.719</td><td>0.840</td></tr><tr><td>ET</td><td>0.605</td><td>0.733</td><td>0.674</td><td>0.805</td><td>0.723</td><td>0.847</td></tr><tr><td>HGB</td><td>0.581</td><td>0.703</td><td>0.644</td><td>0.778</td><td>0.710</td><td>0.822</td></tr><tr><td>LGBM</td><td>0.588</td><td>0.700</td><td>0.645</td><td>0.774</td><td>0.702</td><td>0.815</td></tr><tr><td>CAT</td><td>0.600</td><td>0.716</td><td>0.661</td><td>0.787</td><td>0.715</td><td>0.834</td></tr><tr><td>TABTX</td><td>0.619</td><td>0.735</td><td>0.658</td><td>0.785</td><td>0.701</td><td>0.821</td></tr></table>

RQ3 - Evidence grounding: Table 4 reports the explanation-grounding metrics for the fixed XGB-ldt alert generator on OULAD. Direct@3 and Direct@5 measure whether the top attributed features are supported by evidence chains reaching feature-specific activity or assessment records. Contextual coverage and traceability are equal to 1.0 at all checkpoints, showing that every generated evidence item is linked to a traceable graph basis. Direct coverage is lower because absence and inactivity features are grounded through contextual records, such as registrations or due-assessment metadata, rather than feature-specific activity rows. Fid@3 decreases at later checkpoints as the diversity-aware evidence policy selects a broader set of assessment and temporal evidence.

Table 4. Explanation-grounding metrics for the fixed XGB-LDT alert generator; alerts are aggregated over the six test modules.
<table><tr><td>Checkpoint Alerts</td><td></td><td>Direct@3</td><td>Direct@5</td><td>Fid@3</td></tr><tr><td>Week 4</td><td></td><td> $5 , 9 1 4 \ 0 . 7 6 4 \pm 0 . 1 1 9 \ 0 . 7 6 0 \pm 0 . 1 2 0 \ 0 . 8 9 8 \pm 0 . 0 6 9$ </td><td></td><td></td></tr><tr><td>Week 8</td><td></td><td> $^ { 5 , 6 0 9 } \ 0 . 7 0 6 \pm 0 . 1 2 6 \ 0 . 6 9 5 \pm 0 . 0 8 8 \ 0 . 8 2 8 \pm 0 . 1 3 1$ </td><td></td><td></td></tr><tr><td>Week 12</td><td></td><td> $6 , 3 8 5 ~ 0 . 6 4 8 \pm 0 . 1 5 0 ~ 0 . 6 6 5 \pm 0 . 0 7 9 ~ 0 . 7 4 4 \pm 0 . 0 6 3$ </td><td></td><td></td></tr></table>

Note. Direct@k is direct evidence coverage for the top-k attributed features; F id@k denotes presented fidelity. Contextual Cov@3=Contextual Cov@5=1.000, traceability=1.000, and evidence items per alert=5.0 at every checkpoint.

Case-level evidence: We close with two Week 8 alerts generated by the XGB-ldt run for the AAA/2014J test presentation. Learner identifiers are the anonymized OULAD identifiers. The first case is a true positive. The model assigns learner #111512 a risk probability of 0.928, and the learner’s final outcome in OULAD is Fail. The instructor-facing explanation reads:

Learner #111512 is predicted to be at risk at week 8 because: (1) no assessment due before day 56 had been submitted, so both n\_submitted and avg\_score are 0; (2) two TMAs were already due, on days 19 and 54; and (3) the learner’s last VLE activity before the checkpoint was on day 34, giving days\_since\_last=22 and an activity change from 25 clicks in days 28–41 to 0 clicks in days 42–55.

The second case is a borderline false positive. The model assigns learner #1112555 a risk probability of 0.517 at the same checkpoint, while the final outcome is Pass. The explanation reads:

Learner #1112555 is predicted to be at risk at week 8 because: (1) one assessment due before day 56 had not yet been submitted; (2) missing\_assessment=1, although one earlier TMA was submitted on day 19 with a score of $6 \%$ and (3) the learner remained recently active, with a VLE interaction on day 55 and days\_since\_last=1.

This second case is not presented as a predictive success. The missing assessment was submitted on day 57 with a score of 72, but this information is correctly excluded from the day-56 graph. Together, the two cases illustrate how the ldt distinguishes direct activity and submission records from contextual absence evidence, and how each alert can be traced from attributed features to checkpoint-bounded evidence items and their originating records.

## 5 Discussion

The results lead to three findings. First, the ldt can be populated incrementally across checkpoints while preserving the structural constraints required for evidence-grounded alerts. The graph grows from Week 4 to Week 12, and SHACL validation reports no violations, indicating that the observation and analytic layers remain coherent over time. Second, the KG-mediated pipeline preserves the predictive behavior of the raw-table pipeline. The identical F1 and AUC values are expected because both variants use the same feature definitions; the value of the KG is not predictive improvement, but the explicit connection between features, evidence items, and raw records. Third, alerts are traceable through evidence chains. Direct coverage is lower for absence and inactivity features because these signals are grounded through contextual records, such as registrations or due-assessment metadata, rather than direct activity rows.

A key methodological choice is to make the graph load-bearing rather than merely explanatory. If the graph were populated only after prediction, it would serve as an external explanation store and could diverge from the actual preprocessing pipeline. In contrast, the same representation is used here to derive predictive features and to preserve their evidential basis. This design reduces the risk of inconsistency between raw data processing, model input, and presented explanations. It also makes explanation completeness verifiable through SHACL constraints and graph queries.

The use of the term Learner Digital Twin should be understood in a restricted operational sense. The study simulates synchronization by chronologically replaying a historical dataset at predefined checkpoints. It does not evaluate live ingestion, out-of-order events, data correction, real-time system integration, or instructor feedback loops. The implementation therefore demonstrates a checkpoint-indexed ldt with recurrent synchronization and analytic history, rather than a fully deployed live educational digital twin.

## 6 Conclusion

This paper introduced an incremental knowledge-graph Learner Digital Twin for evidence-grounded early warning in learning analytics. The approach routes prediction through a semantic representation in which learner observations, predictive features, feature attributions, risk alerts, evidence items, and raw records are explicitly represented as graph entities. The ldt therefore serves both as a feature-extraction substrate and as an evidence-grounding substrate. The empirical instantiation on OULAD shows that graph-mediated feature extraction preserves the predictive behavior of equivalent raw-table features across model families and checkpoints. At the same time, generated alerts can be linked back to feature attributions, evidence items, and raw platform records through queryable evidence chains. These results suggest that semantic provenance can make early-warning pipelines more transparent, traceable, and verifiable without reducing predictive performance. Future work will extend the learner twin toward competency-aware learner states, evaluate the framework in live learning platforms, and study whether evidence-grounded alerts improve instructors’ understanding, trust, and intervention decisions.

## Acknowledgments

We thank the Ikigai consortium led by the association Games for Citizens, the company Gamaizer, and the FORTEIM project (winner of the AMI CMA France 2030 call for projects), for their support and collaboration. Their contributions have provided significant added value to the completion of this research.

## References

1. Akçapınar, G., Altun, A., Aşkar, P.: Using learning analytics to develop earlywarning system for at-risk students. International Journal of Educational Technology in Higher Education 16(1), 40 (2019)

2. Arnold, K.E., Pistilli, M.D.: Course signals at purdue: Using learning analytics to increase student success. In: Proceedings of the 2nd international conference on learning analytics and knowledge. pp. 267–270 (2012)

3. Breiman, L.: Random forests. Machine learning 45(1), 5–32 (2001)

4. Bull, S., Kay, J.: Student models that invite the learner in: The smili:() open learner modelling framework. International journal of artificial intelligence in education 17(2), 89–120 (2007)

5. Chen, P., Lu, Y., Zheng, V.W., Chen, X., Yang, B.: Knowedu: A system to construct knowledge graph for education. Ieee Access 6, 31553–31563 (2018)

6. Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. pp. 785–794 (2016)

7. Dietze, S., Taibi, D., Yu, H.Q., Dovrolis, N.: A linked dataset of medical educational resources. British Journal of Educational Technology 46(5) (2015)

8. Friedman, J.H.: Greedy function approximation: a gradient boosting machine. Annals of statistics pp. 1189–1232 (2001)

9. Geurts, P., Ernst, D., Wehenkel, L.: Extremely randomized trees. Machine learning 63(1), 3–42 (2006)

10. Grieves, M., Vickers, J.: Digital twin: Mitigating unpredictable, undesirable emergent behavior in complex systems. In: Transdisciplinary perspectives on complex systems: New findings and approaches, pp. 85–113. Springer (2016)

11. Habarurema, J.B., Di Fuccio, R., Limone, P.: Enhancing e-learning with a digital twin for innovative learning. The International Journal of Information and Learning Technology 42(3), 341–351 (2025)

12. Herodotou, C., Rienties, Boroowa, A., Zdrahal, Z., Hlosta, M.: A large-scale implementation of predictive learning analytics in higher education: the teacher’s role and perspective. Educational technology research and development (2019)

13. Hlosta, M., Zdrahal, Z., Zendulka, J.: Ouroboros: early identification of at-risk students without models based on legacy data. In: Proceedings of the seventh international learning analytics & knowledge conference. pp. 6–15 (2017)

14. Huang, X., Khetan, A., Cvitkovic, M., Karnin, Z.: Tabtransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678 (2020)

15. IMS Global Learning Consortium: IMS learner information package specification. Final Specification, Version 1.0 (Mar 2001), https://www.imsglobal.org/ content/learner-information-packaging-v1, archived by 1EdTech

16. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y.: Lightgbm: A highly eficient gradient boosting decision tree. Advances in neural information processing systems 30 (2017)

17. Khosravi, H., Shum, S.B., Chen, G., Conati, C., Tsai, Y.S., Kay, J., Knight, S., Martinez-Maldonado, R., Sadiq, S., Gašević, D.: Explainable artificial intelligence in education. Computers and education: artificial intelligence 3, 100074 (2022)

18. Knublauch, H., Kontokostas, D.: Shapes constraint language (SHACL). W3C Recommendation (2017), https://www.w3.org/TR/shacl/

19. Kuzilek, J., Hlosta, M., Zdrahal, Z.: Open university learning analytics dataset. Scientific data 4(1), 170171 (2017)

20. Le, N.L., Abel, M.H., Laforge, B.: From student risk prediction to sc2r: Semantics-constrained counterfactual recourse for educational decision support. arXiv preprint arXiv:2608.17618 (2026)

21. Le, N.L., Abel, M.H., Laforge, B.: When can we trust early warnings? leakageexcluded early outcome prediction from lms interaction logs. In: 22nd International Conference on Intelligent Tutoring Systems (2026)

22. Lebo, T., Sahoo, S., McGuinness, D., Belhajjame, K., Cheney, J., Corsar, D., Garijo, D., Soiland-Reyes, S., Zednik, S., Zhao, J.: Prov-o: The prov ontology (2013)

23. Lundberg, S.M., Erion, G.G., Lee, S.I.: Consistent individualized feature attribution for tree ensembles. arXiv preprint arXiv:1802.03888 (2018)

24. Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. Advances in neural information processing systems 30 (2017)

25. Moreau, L.: The foundations for provenance on the web. Foundations and Trends® in Web Science 2(2-3), 99–241 (2010)

26. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A.: Catboost: unbiased boosting with categorical features. Advances in neural information processing systems 31 (2018)

27. Ribeiro, M.T., Singh, S., Guestrin, C.: Why should i trust you? explaining the predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. pp. 1135–1144 (2016)

28. Saleh, M., Abel, M.H.: System of information systems to support learners (a case study at the university of technology of compiègne). Behaviour & Information Technology 37(10-11), 1097–1110 (2018)

29. Tao, F., Cheng, J., Qi, Q., Zhang, M., Zhang, H., Sui, F.: Digital twin-driven product design, manufacturing and service with big data. The International Journal of Advanced Manufacturing Technology 94(9), 3563–3576 (2018)

30. Tao, F., Zhang, H., Liu, A., Nee, A.Y.: Digital twin in industry: State-of-the-art. IEEE Transactions on industrial informatics 15(4), 2405–2415 (2018)

31. Waheed, H., Hassan, S.U., Aljohani, N.R., Hardman, J., Alelyani, S., Nawaz, R.: Predicting academic performance of students from vle big data using deep learning models. Computers in Human behavior 104, 106189 (2020)