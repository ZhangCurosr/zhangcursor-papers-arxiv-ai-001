# Learning Across Temporal Graph Representation Silos

Minwoo Yu and Young-guk Ha

Abstract—Temporal graph data are recorded in different representations, but whether these differences also determine the scope of learned knowledge reuse remains unclear. This study investigates whether predictive structure learned from relational histories can be reused between continuous-time dynamic graphs (CTDGs) and temporal knowledge graphs (TKGs). Bidirectional reuse emerges while retaining distinct input and prediction interfaces and independent vocabularies, and sources with the same representation are not always more useful. Controlled reorganizations of source histories show that the organization contributing to reuse varies across source–target conditions. We show that learned aspects of Relational Dynamics, the predictive structure present in relational histories, can be reused across conventional representation boundaries. Independently pretrained TKG and CTDG learners can also be fused at the score level to improve prediction on a third dataset without entity or relation-semantic alignment. Two proofs of concept in recommendation and security detection illustrate that external temporal sources without direct alignment can also become useful pretraining resources.

Index Terms—Temporal graph learning, relational dynamics, crossrepresentation transfer, graph foundation models, multi-source learning.

## 1 INTRODUCTION

Foundation models require reusable knowledge. The first question in temporal graph learning is therefore not which architecture to scale up, but what learned from different temporal graph datasets can be reused across representation boundaries. This study asks whether knowledge learned from recurring relations and patterns of change over time remains reusable when the representation in which the data are recorded changes.

Temporal graph data are recorded in different representations. Continuous-time dynamic graphs (CTDGs) describe timestamped interaction streams, whereas temporal knowledge graphs (TKGs) describe typed facts of the form (subject, relation, object, time) [1]. They differ in input structure, vocabulary, queries, and prediction interfaces. Accordingly, recent pretraining and transfer studies have largely expanded the scope of learning within a single representation. For CTDGs, TGPM [2] and DyGFM [3] address pretraining and adaptation across dynamic graph domains. For TKGs, POSTRA [4] and FITTER [5] study transfer to new entity and relation vocabularies and graphs. Existing approaches thus generally choose a representation first and then seek reusable knowledge within it.

Nevertheless, similar predictive regularities recur across different representations. In CTDGs, EdgeBank showed that the recurrence of past interactions provides a strong predictive signal [6]; in TKGs, History Repeats Itself also reported the predictive power of recurring facts [7]. A more fundamental question is therefore whether predictive structure exists that is actually reusable across heterogeneous temporal graph representations.

Section 5.2 examines this question through frozen crossrepresentation transfer between CTDGs and TKGs. We retain separate input interfaces and prediction heads for each representation and transfer only the history encoder learned on the source to the target. Despite sharing neither entity vocabularies nor relation semantics, source-pretrained backbones provide greater utility than random backbones in both directions, and such reuse is also observed across multiple backbones. Thus, learned predictive structure can survive changes in representation and prediction interface.

Does this result simply mean that a pretrained network is better than a random network? Section 5.3 shows that downstream utility changes when source event–time assignments and partner organization are perturbed before transfer. In particular, even partial partner corruption that preserves the source training budget and similar source validation performance yields lower target AP and AUC than the intact source in all three seeds for ICEWS14 → Wikipedia. Transfer utility therefore depends not only on whether pretraining occurred, but also on what was learned from the source relational history.

Relational Dynamics denotes predictive structure present in relational histories. A model captures part of this structure through learning, which we call Learned Relational Dynamics. The part that is also useful for prediction in other datasets or representations is termed Reusable Relational Dynamics. Reusability depends not only on the structure learned from the source, but also on the target and task to which it is applied.

Section 5.4 makes this distinction concrete through controlled reorganizations of source histories. We alter partners, event sequence, temporal placement, or relation direction while retaining source competence and then measure changes in target utility. Some conditions favor the original event sequence, whereas others transfer better after source histories are reorganized. Learned Relational Dynamics are therefore not reused in their entirety; the useful part varies with the target and task.

Are these relational dynamics specific to a particular dataset pair? Section 5.5 evaluates multiple CTDGs and TKGs, source–target pairs in both directions, and targetdata regimes of 5%, 20%, and 100%. Cross-representation reuse recurs in most conditions, and source initialization also provides gains under fine-tuning. Reusable predictive knowledge can thus be accumulated from heterogeneous temporal graph data.

This raises a further question for foundation models: is it sufficient to organize reusable knowledge within representation-specific silos such as CTDGs and TKGs? Section 5.6 compares all 30 directed transfers among six datasets, excluding self-transfer. Source utility is not consistently separated by representation membership, and crossrepresentation sources repeatedly prove more useful than same-representation sources for the same target. Representation identity does not fully explain which sources provide reusable knowledge.

Figure 1 summarizes the perspective on temporal graph foundation models suggested by these results. CTDGs and TKGs need not be converted into a single representation. Their observation and prediction interfaces can remain distinct while sharing a learner that processes temporalrelational histories beneath them.

We establish three findings. First, CTDG↔TKG transfer demonstrates that aspects of Relational Dynamics remain reusable after the representation and prediction interface change. Second, source interventions and functional characterization show that reuse depends on source relational organization and that the contributing organization varies across source–target conditions. Third, experiments across domains and data regimes, together with a complete transfer matrix, show that source utility orderings cross conventional representation taxonomies. Restricting foundationmodel pretraining sources to representation-specific silos can therefore exclude more useful knowledge. Section 5.7 further tests whether independently pretrained TKG and CTDG learners can be used jointly on a third target. Their score-level combination demonstrates that reused predictive knowledge can provide additional utility without directly aligning the sources. Finally, two proofs of concept (PoCs) in recommendation and security detection illustrate how this reuse can turn external temporal sources without direct alignment into practical learning resources.

## 2 RELATED WORK

Temporal graph representations and learning.: Kazemi et al. [1] distinguish the temporal representations of CTDGs and discrete-time dynamic graphs, cover dynamic KGs, and organize models from an encoder–decoder perspective. Temporal representation and relation typing are distinct axes. In this paper, CTDG denotes data represented as interaction streams, whereas TKG denotes data represented as typed temporal facts; we do not assume that they are mutually exclusive graph classes. Rather than using this distinction as a model taxonomy, we study whether learned history-processing capabilities are reusable across temporal graph representations.

UTG compares DTDG and CTDG models within a common framework through snapshot–event conversion and unified training and evaluation procedures [9]. Our study tests the reuse of source-learned predictive structure between independent CTDGs and TKGs, providing empirical evidence for relational dynamics.

Relational regularities in temporal prediction.: By remembering past edges, EdgeBank demonstrated the importance of recurrence and evaluation conditions in dynamic link prediction [6]. History Repeats Itself analyzed the predictive power of baselines that exploit recurring facts in TKG forecasting [7]. These findings show that similar predictive regularities arise independently in different representations. However, both studies conduct their analyses within their respective representations and do not directly test whether such regularities can be learned and reused across representations.

Heterogeneous graph integration and pre-training.: AdaVQGP connects graphs from different domains through a shared semantic codebook and virtual semantic nodes, combining feature-level alignment with model-level structural transfer [10]. GPH<sup>2</sup> constructs unified multi-view graphs from homogeneous and heterogeneous graphs, independently pretrains an expert for each graph, and freezes, aligns, and fuses these experts downstream [8]. In recommendation, IGSRec refines an LLM-based semantic graph using observed user–item interactions and combines embeddings learned from the interaction and semantic graphs [11].

Temporal patterns and cross-domain learning.: Re-MIT combines temporal-, neighborhood-, community-, and graph-level information through feature pretraining and knowledge distillation for temporal graph learning [12]. Our study asks whether predictive structure learned from relational histories remains reusable across distinct temporal graph representations and prediction interfaces.

TGPM represents timestamped interaction sequences as interaction patches derived from temporal random walks and learns generalized evolving patterns through masked token modeling and next-time prediction [2]. It evaluates cross-domain transfer of the learned model on temporal link prediction, including ICEWS1819 [2].

DyGFM proposes learning and adaptation that account for semantic and temporal differences across dynamic graph domains [3]. FLIP reduces dependence on graph-specific vocabularies through factorized edge-event intensity and addresses transfer between CTDGs by reusing parameters without target gradient updates while constructing states from target histories [13]. In this line of work, domains and node vocabularies change, but the interaction-based CTDG representation is retained.

For TKGs, POSTRA studies fully inductive temporal reasoning and transfer to graphs with new entities, relations, and timestamps [4]. FITTER transfers across TKG domains through vocabulary-agnostic structural learning [5]. Here too, domains and vocabularies change, but the typed temporal fact representation is retained.

STKGN combines a spatial-temporal knowledge graph with a continuous-time dynamic graph network for event prediction [14]. This approach combines graphs and modeling components within one application, whereas our setting reuses a learned history learner between independent CTDGs and TKGs while changing the input and prediction interfaces.

![](images/bf6de8a1546606ff5497f522295918fbaa790eb528b71258f202923be36c2919.jpg)  
Fig. 1. Relational dynamics across temporal graph representation silos. We ask what can be learned and reused beneath different observation and prediction formats. Representation-specific colors transitioning to a neutral color below illustrate the perspective of relational dynamics as a common object of study. Color boundaries and overlaps are conceptual and assume neither identical dynamics nor a common latent space. The dashed “Others” silo denotes future extensions. The bottom summarizes the existence of reuse, dependence on source organization, reuse across domains and training fractions, and source utility rankings. The right contrasts GPH<sup>2</sup> [8], which designs unified multi-view inputs and expert fusion, with our focus on the object of learning itself across representations.

Multi-source prediction fusion.: In multi-source domain adaptive object detection, Ma et al. [15] use feature alignment and cross-domain information exchange and combine predictions from multiple adapted detectors with weights. We investigate whether learners independently pretrained on distinct TKG and CTDG sources can be transferred to a third target and used jointly without cross-source entity or relation-semantic alignment.

Positioning of this study.: To our knowledge, this is the first empirical study of what is reusable beneath distinct temporal graph representations themselves. Rather than designing a common formulation for heterogeneous graphs, we preserve representation-specific interfaces and test whether learned predictive structure transfers across them. We examine the reusability of learned aspects of Relational Dynamics, their dependence on source organization, and their transfer scope.

## 3 PROBLEM FORMULATION

A temporal graph dataset D contains relational histories recorded in a representation R(D). A query q at time t is predicted using the preceding observation history $\mathcal { H } _ { D } ( t )$ We study whether predictive structure learned from one dataset’s history can be reused in another while retaining the inputs and prediction tasks of their respective representations. Although this formulation is not restricted to CTDGs and TKGs, our empirical study focuses on these two representations, for which substantial research on models and prediction tasks is available.

Conceptual distinction.: Relational Dynamics is the predictive structure present in relational histories. Learned Relational Dynamics is the part captured by a model during learning, and Reusable Relational Dynamics is the part that remains useful for target prediction in another dataset or representation. Reusable Relational Dynamics is therefore not a fixed property of a source encoder alone, but a property of the transfer relation jointly determined by source, target, and prediction query.

Terminology and transfer direction.: The source S is the dataset on which transferable parameters are first learned, and the target T is the dataset that receives these parameters for adaptation and predictive evaluation. S → T denotes this direction of learned information transfer. For example, in Wikipedia → ICEWS14, Wikipedia is the source and ICEWS14 is the target. Domain denotes the application environment in which data are observed, whereas representation denotes the format used to record events and relations. We use datasets as the unit of domain comparison; same- and cross-representation indicate whether the source and target share the same CTDG/TKG designation.

A dataset-specific input interface $A _ { \phi _ { D } }$ transforms observations into encoder inputs, a shared history encoder $F _ { \theta }$ (backbone) processes relational histories, and a prediction head $G _ { \psi _ { D } }$ transforms its outputs into prediction scores for queries. Shared parameters are transferred or shared across datasets, whereas private parameters are learned separately for each dataset. The prediction function is

$$
s _ { D } ( q ) = G _ { \psi _ { D } } ( q , F _ { \theta } ( A _ { \phi _ { D } } ( q , \mathcal { H } _ { D } ( t ) ) ) ) .\tag{1}
$$

Our implementation and experiments instantiate this formulation through CTDG interaction $( u , v , t )$ prediction and TKG head/tail entity forecasting. Here, q is a pairwise query that includes the candidate being evaluated, and $A _ { \phi _ { D } }$ extracts and processes the strict-past histories $\mathcal { H } _ { u } ^ { D } ( t ) , \mathcal { H } _ { v } ^ { D } ( t )$ incident to the two endpoints from $\mathcal { H } _ { D } ( t )$ . We therefore distinguish the general interface separation in Eq. (1) from the two-endpoint implementation used here. $F _ { \theta }$ learns part of the Relational Dynamics in relational histories; this study examines the part reused for target prediction. Section $5 . 2$ evaluates reuse through frozen transfer, Section 5.3 evaluates sensitivity to source organization, and Section 5.4 distinguishes which source-history organization contributes to reuse for each target.

Cross-representation reuse.: When the source $S$ and target $T$ satisfy $\mathsf { R } ( S ) \neq \mathsf { R } ( T )$ , only the shared parameters $\theta _ { S }$ learned on the source are transferred to the target. Let $M _ { T } ^ { \mathrm { f r z } } ( \theta )$ denote the evaluation score obtained after learning target-private parameters $\left( \phi _ { T } , \psi _ { T } \right)$ with θ fixed. The reuse gain is

$$
\begin{array} { r } { \Delta _ { S \to T } = M _ { T } ^ { \mathrm { f r z } } ( \theta _ { S } ) - M _ { T } ^ { \mathrm { f r z } } ( \theta _ { 0 } ) , } \end{array}\tag{2}
$$

where $\theta _ { 0 }$ is a randomly initialized encoder. Under identical target training conditions, $\Delta _ { S  T } > 0$ indicates a gain from reusing the source-learned encoder. Both conditions learn private parameters from the target training data.

Functional characterization.: A positive reuse gain shows that source-learned structure is useful for the target. To characterize that structure, we define a controlled operator $C _ { k }$ that reorganizes one aspect k of the source relational history. The variant $D _ { S } ^ { ( k ) }$ preserves the event count and specified statistics of the original source while reorganizing partners, event sequence, temporal placement, or relation direction:

$$
D _ { S } ^ { ( k ) } = C _ { k } ( D _ { S } ) , \qquad \theta _ { S } = \mathcal { A } ( D _ { S } ) , \qquad \theta _ { S } ^ { ( k ) } = \mathcal { A } ( D _ { S } ^ { ( k ) } ) .\tag{3}
$$

The two encoders use identical initialization, architecture, optimizer, and update count. Before comparing target utility, we measure their relative performance difference on the same original source history:

$$
g _ { S } ^ { ( k ) } = \frac { \left| M _ { S } ^ { \mathrm { o r i g } } ( \theta _ { S } ) - M _ { S } ^ { \mathrm { o r i g } } ( \theta _ { S } ^ { ( k ) } ) \right| } { M _ { S } ^ { \mathrm { o r i g } } ( \theta _ { S } ) } .\tag{4}
$$

Only conditions satisfying $g _ { S } ^ { ( k ) } \leq 0 . 0 3$ for all three seeds enter the target comparison. For source-competence-matched conditions, the target contribution of history aspect k is

$$
B _ { S \to T } ^ { ( k ) } = M _ { T } ^ { \mathrm { f r z } } ( \theta _ { S } ) - M _ { T } ^ { \mathrm { f r z } } ( \theta _ { S } ^ { ( k ) } ) .\tag{5}
$$

Here $B > 0$ indicates that the original organization contributes to target reuse, whereas $\bar { B } \ < \ 0$ indicates better transfer from the reorganized source. Because the same organization may function differently across targets and queries, we interpret each $S \times T \times k$ condition as a functional profile.

## 4 EXPERIMENTAL INSTANTIATION

To empirically examine cross-representation predictive reuse, we instantiate Eq. (1) using representation-specific input and prediction interfaces and a transferable history encoder. We adopt DyGFormer’s history-processing components as the primary experimental backbone [16].

## 4.1 Temporal-Relational Inputs

For each endpoint, we use up to 63 most recent strict-past events and a central-node token, fixing the input length to 64 including padding. Each position comprises four channels: node, event, relative time, and neighbor co-occurrence. The node channel uses raw features and is set to a zero vector for TKGs. We do not use trainable entity-ID embeddings; IDs are used for history lookup and co-occurrence computation. The time channel encodes the time difference $t - \tau _ { i }$ , and the co-occurrence channel encodes the number of times the corresponding neighbor appears in each endpoint’s input sequence.

The event channel uses edge features for CTDGs and the sum of embeddings for the historical relation $\rho _ { i }$ and its direction $d _ { i }$ relative to the central node for TKGs:

$$
e _ { i } ^ { D } = \left\{ \begin{array} { l l } { x _ { i } ^ { \mathrm { e d g e } } , } & { D \mathrm { i s } \mathrm { C T D G } , } \\ { E _ { \mathrm { r e l } } ^ { D } ( \rho _ { i } ) + E _ { \mathrm { d i r } } ^ { D } ( d _ { i } ) , } & { D \mathrm { i s } \mathrm { T K G } . } \end{array} \right.\tag{6}
$$

The direction $d _ { i }$ distinguishes outgoing, incoming, and selfloop events, preserving fact direction in both endpoints’ histories. Relation and direction embeddings and the eventchannel projection are dataset-specific parameters.

## 4.2 Shared History Encoder and Prediction

Each of the four channels is grouped into patches of length 2 and projected, after which the projected channels are concatenated. Let $Z _ { a }$ denote the patch sequence for endpoint a. The two sequences are processed jointly and pooled separately for each endpoint:

$$
\begin{array} { r l } & { [ Y _ { u } ; Y _ { v } ] = \operatorname { T r a n s f o r m e r } _ { \theta } ( [ Z _ { u } ; Z _ { v } ] ) , } \\ & { \quad \quad \quad h _ { a } = P _ { O } ( \operatorname { M e a n P o o l } ( Y _ { a } ) ) , \quad a \in \{ u , v \} . } \end{array}\tag{7}
$$

Joint self-attention makes $h _ { u } , h _ { v }$ pair-conditioned representations that reflect both histories. The implementation uses 50-dimensional projections per channel, a two-layer, twohead Transformer, and 172-dimensional output vectors. The CTDG head scores the two vectors, whereas the TKG head additionally receives the query relation embedding:

$$
\begin{array} { r l } & { ~ s _ { \mathrm { C T D G } } ( u , v , t ) = G _ { \psi _ { C } } ( [ h _ { u } \Vert h _ { v } ] ) , } \\ & { s _ { \mathrm { T K G } } ( u , \rho , v , t ) = G _ { \psi _ { K } } ( [ h _ { u } \Vert h _ { v } \Vert E _ { \mathrm { r e l } } ^ { D } ( \rho ) ] ) . } \end{array}\tag{8}
$$

Each head is an MLP with a hidden layer and ReLU that outputs a single logit. In TKGs, histories and queries use the same relation embedding table; evaluation replaces either the head or tail candidate while retaining the original fact direction.

TABLE 1  
Datasets and evaluation settings.
<table><tr><td>Setting</td><td>CTDG</td><td>TKG</td></tr><tr><td>Dataset</td><td>Wikipedia, Reddit, MOOC</td><td>ICEWS14, ICEWS05-15</td></tr><tr><td>Task / metrics</td><td>Link prediction; AP, AUC</td><td>Head/tail forecasting; fil- tered MRR, Hits@1/3/10</td></tr><tr><td>Checkpoint selection Full-validation AP</td><td></td><td>100-negative MRR; full val- idation (256 queries for the DyGFormer Wiki →</td></tr><tr><td>Test evaluation</td><td>positive</td><td>ICEWS14 diagnostic) 1 random destination per Bidirectional queries × all entities; timestamp filtering</td></tr></table>

## 4.3 Learning and Parameter Reuse

We construct one negative per positive and apply binary cross-entropy to their scores $s ^ { + } , s ^ { - } \colon$

$$
\begin{array} { r } { \mathcal { L } _ { D } = - \frac { 1 } { 2 } \left[ \log \sigma ( s ^ { + } ) + \log \sigma ( - s ^ { - } ) \right] . } \end{array}\tag{9}
$$

CTDG training replaces the destination; TKG training replaces the entity in the prediction direction and uses both head and tail queries for each fact. Negative sampling and evaluation candidate sets follow each target’s protocol.

The shared parameters θ comprise the time and cooccurrence encoders, node, time, and co-occurrence projections, the Transformer, and the output projection. The private parameters $\left( \phi _ { D } , \psi _ { D } \right)$ comprise the event projection, TKG relation and direction embeddings, and the prediction head. Only θ is transferred from the source; target-private parameters are freshly initialized, and feature tables and histories come from the target.

Frozen transfer fixes $\theta _ { S }$ and learns only target-private parameters. Random-frozen is a control that fixes random parameters $\theta _ { 0 } .$ . Source-finetune starts from $\theta _ { S }$ and updates the entire model, whereas target-only trains the entire model from random initialization. We match target-private initialization and training budgets between paired conditions. Frozen comparisons measure reuse of the learned encoder, while fine-tuning comparisons measure the effect of source initialization.

## 5 EXPERIMENTS

## 5.1 Experimental Setup

We use DyGFormer as the primary backbone and additionally examine frozen reuse with GraphMixer and TGAT. Frozen comparisons measure source-frozen − randomfrozen, and fine-tuning comparisons measure sourcefinetune − target-only, with matched target training budgets and private initialization. We report means, standard deviations, and paired gains across seeds 7, 17, and 37. Table 1 summarizes the datasets and evaluation settings.

The GraphMixer and TGAT experiments and Sections 5.5 and 5.6 use the first 32,768 events of each dataset, split chronologically into approximately 70/15/15, with 10 training epochs each for the source and target. The 5% and 20% target training conditions retain only the most recent training events, reducing supervision and training history together. Validation and test sets remain fixed, and evaluation histories contain only retained training events and validation/test observations preceding the query. Parameters are not updated during evaluation.

The bidirectional DyGFormer existence experiment evaluates the full test sets of the original data splits. GraphMixer, TGAT, and the complete source–target matrix use frozen transfer with 100% target training data. Appendix A provides experiment-specific details, including negative sampling, history updates, and corruption controls.

The recommendation PoC in Section 5.8 transfers an ICEWS14-pretrained encoder to LastFM and fine-tunes the entire model, comparing it with target-only training of the same model. The first 32,768 LastFM interactions are chronologically split into 22,937/4,915/4,916 training/validation/test interactions. We use the most recent 5% and 20% of training interactions (1,147 and 4,588). Target training uses Adam $( 1 0 ^ { - 4 } )$ , a batch size of 512, and 10 epochs. Checkpoints are selected by 100-negative NDCG@10 on 256 fixed validation queries, and Recall@10 and NDCG@10 are evaluated over all 1,000 items on a separate set of 256 fixed test queries. Previously listenedto items remain candidates, and ties receive average ranks. History updates follow the procedure above; no item–entity alignment is used.

## 5.2 Representation Boundaries Do Not Bound Predictive Reuse

Predictive structure learned in one representation remains useful after the representation and prediction interface change. Table 2 shows bidirectional frozen reuse between the Wikipedia CTDG and ICEWS14 TKG.

The source-pretrained backbone outperforms the random backbone in all seeds in both directions. Test MRR increases from 0.1165 to 0.2059 for Wiki → ICEWS14, and test AP increases from 0.8463 to 0.9822 for ICEWS14 → Wiki. These improvements occur even with the backbone fixed during target training, demonstrating that source-learned predictive structure is reusable across representations and prediction interfaces.

Across backbones.: Cross-representation reuse is not confined to DyGFormer. Figure 2 shows reuse gains by transfer direction for DyGFormer, the MLP-mixerbased GraphMixer, and the temporal-graph-attention-based TGAT. Table S5 in Appendix C provides the scores and paired gains for each backbone.

TGAT yields positive gains in all seeds in both directions, and GraphMixer does so in all seeds for ICEWS14 → Wiki. GraphMixer’s mean gain for Wiki → ICEWS14 is negative, indicating that the utility of reuse depends on the backbone and transfer direction. Improvements recurring across different encoders show that representation differences do not block reusable predictive structure and that this phenomenon is not specific to one architecture.

Same-dataset representation diagnostic.: To complement transfer across datasets, Table S10 in Appendix F compares pretraining with relation information retained or hidden in the same ICEWS14 source history. Pretraining on the relation-hidden CTDG-style view raises test MRR by 0.0746 over training from scratch; retaining relations in the TKG view yields a further 0.0213. Thus, history-based predictive structure learned without relation typing can provide useful initialization for subsequent TKG forecasting. Pretraining and adaptation use nonoverlapping training events, but additional optimization relative to scratch and the information difference introduced by hiding relations remain. This diagnostic therefore provides supporting evidence of reuse, rather than identifying an independent causal effect of representation change.

TABLE 2  
Bidirectional frozen transfer with DyGFormer.
<table><tr><td>Direction</td><td>Metric</td><td>Frozen backbone</td><td>Seed 7</td><td>Seed 17</td><td>Seed 37</td><td>Mean ± SD</td><td>Mean ∆</td></tr><tr><td>Wiki → ICEWS14</td><td>Test MRR</td><td>Random</td><td>0.1225</td><td>0.1177</td><td>0.1092</td><td> $0 . 1 1 6 5 \pm 0 . 0 0 6 7$ </td><td></td></tr><tr><td>Wiki → ICEWS14</td><td>Test MRR</td><td>Source-pretrained</td><td>0.2168</td><td>0.1894</td><td>0.2115</td><td> $\mathbf { 0 . 2 0 5 9 \pm 0 . 0 1 4 5 }$ </td><td> $\mathbf { + 0 . 0 8 9 4 }$ </td></tr><tr><td>ICEWS14 → Wiki</td><td>Test AP</td><td>Random</td><td>0.8414</td><td>0.8264</td><td>0.8713</td><td> $0 . 8 4 6 3 \pm 0 . 0 2 2 8$ </td><td></td></tr><tr><td>ICEWS14 → Wiki</td><td>Test AP</td><td>Source-pretrained</td><td>0.9793</td><td>0.9808</td><td>0.9865</td><td> $\mathbf { 0 . 9 8 2 2 \pm 0 . 0 0 3 8 }$ </td><td> $\mathbf { + 0 . 1 3 5 9 }$ </td></tr><tr><td>ICEWS14 → Wiki</td><td>Test AUC</td><td>Random</td><td>0.8320</td><td>0.8213</td><td>0.8601</td><td> $0 . 8 3 7 8 \pm 0 . 0 2 0 1$ </td><td></td></tr><tr><td>ICEWS14 → Wiki</td><td>Test AUC</td><td>Source-pretrained</td><td>0.9763</td><td>0.9789</td><td>0.9845</td><td> $\mathbf { 0 . 9 7 9 9 \pm 0 . 0 0 4 2 }$ </td><td> ${ \bf + 0 . 1 4 2 1 }$ </td></tr></table>

![](images/99a80e9bccb8c23206eefb10c2584562e7ea7479b13864f3a82ff9493b779867.jpg)  
(a) Wikipedia → ICEWS14

(b) ICEWS14 → Wikipedia  
![](images/bcbe0ddcdfda2ac6b34d6cba9564099dd93f2e5631c2b560a13fc9260300d07c.jpg)  
Fig. 2. Frozen reuse across representations and backbones. Each point represents a test gain of source-frozen − random-frozen; large diamonds denote three-seed means and small points individual seeds. Values to the right of zero indicate reuse gains. (a) MRR for Wikipedia → ICEWS14. (b) AP for ICEWS14 → Wikipedia. DyGFormer uses the original data splits, while GraphMixer and TGAT use 32,768-event splits; comparisons therefore concern paired gains within each backbone.

## 5.3 Cross-Representation Reuse Depends on Source Organization

The utility of predictive structure reused across representations depends on the organization of the source from which it was learned. Table 3 compares transfer utility after perturbing source event–time assignments (time shuffle) or interaction partners (partner shuffle).

Partner shuffle changes cross-representation utility in both directions and reduces performance in all three seeds for ICEWS14 → Wikipedia. On average, it reduces MRR by 0.0230 for Wiki → ICEWS14 and AP by 0.0160 for ICEWS14 → Wikipedia. For Wiki → ICEWS14, performance decreases in two of the three seeds but increases by 0.0060 in one, so the effect varies across seeds and directions.

The effect of temporal organization is less uniform. For Wiki → ICEWS14, time shuffle reduces performance in two of three seeds, decreasing mean MRR by 0.0070. For ICEWS14 → Wikipedia, however, mean AP increases by 0.0008. These experiments therefore do not establish temporal ordering itself as a consistent transfer signal in every direction.

The effect of partner organization remains under equal training budgets and similar source performance. To examine whether poorer source-task performance alone explains this effect, we compare partial partner corruption for ICEWS14 → Wikipedia. The 50% nominal corruption level, selected using source scores alone, changes approximately 35% of actual destinations, yet source MRR differs by less than 1% in all three seeds (Table 4).

The intact source yields higher target AP and AUC in all three seeds. The paired gains (Intact − Partial) are $+ 0 . 0 0 1 0 9 1 \pm 0 . 0 0 0 6 4 9$ for AP and $+ 0 . 0 0 1 8 9 4 \pm 0 . 0 0 0 5 6 1$ for AUC, where ± denotes the sample standard deviation across the three seeds. The effect is small but directionally consistent. In particular, for seed 7, the intact source yields higher target AP and AUC despite the corrupted model having higher source MRR. Figure 3 illustrates this contrast: source performance remains similar, whereas target utility favors the intact source in all three seeds.

The evidence for Relational Dynamics therefore extends beyond reuse gains alone. The next section separates aspects of source history to distinguish what contributes to reuse for each target.

## 5.4 Functional Profiles of Reusable Relational Dynamics

The preceding experiments show that source relational organization changes cross-representation utility. We now functionally characterize which organization is reused by a target. We construct controlled data variants that preserve major source statistics while reorganizing partners, event sequence, temporal placement, or relation direction, and train a new encoder in each condition. These variants do not assume a fixed inventory of Relational Dynamics; they probe whether different aspects of relational history contribute to reuse.

TABLE 3  
Transfer performance under source-organization corruption. ∆ denotes Intact − Corrupted. Wiki → ICEWS14 uses 256 validation queries; the reverse direction uses the full test set.
<table><tr><td>Direction / metric</td><td>Seed</td><td>Intact</td><td>Time shuffle</td><td>Partner shuffle</td><td>∆ Time</td><td>∆ Partner</td></tr><tr><td> $\operatorname { W i k i } \to \operatorname { I C E W S 1 4 } \cdot \operatorname { V a l . } \operatorname { M R R }$ </td><td>7</td><td>0.2018</td><td>0.1425</td><td>0.1989</td><td>+0.0593</td><td>+0.0028</td></tr><tr><td> $\operatorname { W i k i } \to \operatorname { I C E W S 1 4 } \cdot \operatorname { V a l . } \operatorname { M R R }$ </td><td>17</td><td>0.1899</td><td>0.2481</td><td>0.1177</td><td>-0.0582</td><td>+0.0722</td></tr><tr><td> $\operatorname { W i k i } \to \operatorname { I C E W S 1 4 } \cdot \operatorname { V a l . } \operatorname { M R R }$ </td><td>37</td><td>0.2490</td><td>0.2292</td><td>0.2550</td><td>+0.0198</td><td>-0.0060</td></tr><tr><td>Wiki → ICEWS14 · Val. MRR</td><td>Mean</td><td>0.2136</td><td>0.2066</td><td>0.1906</td><td>+0.0070</td><td>+0.0230</td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { W i k i } \cdot \mathrm { T e s t } \mathrm { A P }$ </td><td>7</td><td>0.9822</td><td>0.9834</td><td>0.9593</td><td>-0.0012</td><td>+0.0229</td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { W i k i } \cdot \mathrm { T e s t } \mathrm { A P }$ </td><td>17</td><td>0.9821</td><td>0.9846</td><td>0.9658</td><td>-0.0025</td><td>+0.0163</td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { W i k i } \cdot \mathrm { T e s t } \mathrm { A P }$ </td><td>37</td><td>0.9867</td><td>0.9855</td><td>0.9779</td><td>+0.0012</td><td>+0.0088</td></tr><tr><td> $\mathbf { I C E W S 1 4 } \to \mathbf { W i k i } \cdot \mathbf { T e s t \ A P }$ </td><td>Mean</td><td>0.9837</td><td>0.9845</td><td>0.9677</td><td>-0.0008</td><td>+0.0160</td></tr></table>

(a) Source performance  
![](images/6cc71b2059254ccd82642fcbfbbe5cfd872953aab6c7bf5f309e555130ca2c8f.jpg)

(b) Transfer utility  
![](images/fb042698aa25b96faa00b63adfe1cdd4b5fd9d7ae752a5d6219cf30eb58af717.jpg)  
Fig. 3. Different cross-representation utility despite similar source performance. Both source conditions for ICEWS14 → Wikipedia use 10 epochs and 4,980 updates. (a) Relative MRR difference from the intact source, 100×(Intact−Partial)/Intact; shading denotes the ±3% tolerance used for source-only strength selection. Actual differences are below 1% in all three seeds. (b) Within-seed target AP and AUC differences (Intact − Partial), in units of $1 0 ^ { - 3 }$ ; the intact source is better in all three seeds. The nominal 50% corruption level was selected using source scores alone and changed approximately 35% of actual destinations.

TABLE 4  
Partial partner-corruption comparison at similar source performance.
<table><tr><td>Seed</td><td>Source condition</td><td>Source MRR</td><td>Target AP</td><td>Target AUC</td></tr><tr><td>7</td><td>Intact</td><td>0.749487</td><td>0.982154</td><td>0.979528</td></tr><tr><td>7</td><td>50% partial partner</td><td>0.753258</td><td>0.981049</td><td>0.977285</td></tr><tr><td>17</td><td>Intact</td><td>0.755403</td><td>0.982111</td><td>0.979706</td></tr><tr><td>17</td><td>50% partial partner</td><td>0.753498</td><td>0.981675</td><td>0.978459</td></tr><tr><td>37</td><td>Intact</td><td>0.760991</td><td>0.986747</td><td>0.985207</td></tr><tr><td>37</td><td>50% partial partner</td><td>0.753794</td><td>0.985014</td><td>0.983014</td></tr></table>

To separate changes to the source task itself, we evaluate the original and variant encoders on the same original source history. Of 21 source–variant conditions, 15 satisfy the 3% source-competence criterion in all three seeds; only these conditions undergo frozen transfer. $B _ { S  T } ^ { ( k ) }$ is the target-score difference between the original and controlledvariant encoders. Appendix G reports the preserved statistics, source-performance gaps, and all target results.

Seven of the 15 conditions have the same sign across all three seeds. The original organization is consistently better in one condition, whereas the controlled variant is consistently better in six. Table 5 summarizes the six conditions with clear effect sizes. The seventh, temporal reorganization for Wikipedia → Reddit, is also negative in every seed but has a small mean difference of −0.0004 AP.

TABLE 5  
Major functional profiles with consistent directions across three seeds. B is the target-score difference between the original and controlled source variants.
<table><tr><td>Source → Target</td><td>Reorganized history aspect</td><td>Mean B± sample SD Seed signs</td></tr><tr><td>ICEWS14</td><td>→ Event sequence</td><td>+0.0330 ± 0.0389 MRR  $+ ~ + ~ +$ </td></tr><tr><td>ICEWS05-15 ICEWS05-15 Wikipedia</td><td>→ Event sequence</td><td> $- 0 . 0 0 4 1 \pm 0 . 0 0 1 7 \mathrm { A P }$ </td></tr><tr><td>ICEWS05-15</td><td>→ Temporal placement</td><td> $- 0 . 0 0 4 8 \pm 0 . 0 0 2 3 \mathrm { A P }$ </td></tr><tr><td>Wikipedia ICEWS05-15</td><td>→ Relation direction</td><td> $- 0 . 0 0 3 8 \pm 0 . 0 0 4 4 \mathrm { A P }$ </td></tr><tr><td>Wikipedia  $\mathrm { M O } \dot { \mathrm { O C } }  \mathrm { R e } { \mathrm { d } } { \mathrm { d i t } }$ </td><td>Temporal placement</td><td> $- 0 . 0 0 8 1 \pm 0 . 0 0 6 8 \mathrm { A P }$ </td></tr><tr><td>Reddit → Wikipedia Temporal placement</td><td></td><td> $- 0 . 0 0 8 6 \pm 0 . 0 1 0 8 \mathrm { A P }$ </td></tr></table>

For ICEWS14 → ICEWS05-15, the contribution of the original event sequence recurs across all three seeds. Reorganizing event sequence while preserving direction- and relation-specific pair frequencies and source competence reduces target MRR in every seed. The seed-level B values are +0.0061, +0.0154, and +0.0776: their magnitudes vary, but their direction is consistent.

TABLE 6  
Mean frozen gains across target training fractions.
<table><tr><td>Source → Target</td><td>Metric</td><td></td><td>5% train 20% train 100% train</td><td></td></tr><tr><td>Wikipedia → ICEWS14</td><td>MRR</td><td></td><td>+0.0494+0.0675</td><td>+0.0622</td></tr><tr><td> $\mathrm { R e d } \hat { \mathrm { d i t } } \to \mathrm { I C E } { \cal W } { \cal S } 1 4$ </td><td>MRR</td><td></td><td>+0.0559+0.0471</td><td>+0.0496</td></tr><tr><td> $\mathrm { M O O C } \to \mathrm { G D E L T }$ </td><td>MRR</td><td></td><td>-0.0409-0.0413</td><td>-0.0006</td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { W i k i p e d i a }$ </td><td>AP</td><td></td><td>+0.2046 +0.2138</td><td>+0.1544</td></tr><tr><td> $\mathrm { G D E L T } \to \mathrm { R e d d i t } ^ { \bullet }$ </td><td>AP</td><td></td><td>+0.0553 +0.0691</td><td>+0.0771</td></tr><tr><td>Wikipedia → ICEWS05-15 MRR</td><td></td><td></td><td>+0.0454+0.0510</td><td>+0.0649</td></tr><tr><td> $\mathrm { I C E } \hat { \mathsf { W S 0 5 } } { \cdot } 1 5  \mathrm { M O O C }$ </td><td>AP</td><td></td><td>+0.0453 +0.0394</td><td>+0.0680</td></tr></table>

Conversely, for ICEWS05-15 → Wikipedia, encoders trained after reorganizing event sequence, temporal placement, or relation direction outperform the original in AP for all three seeds. Learned Relational Dynamics that are useful on the source are therefore not necessarily the same as Reusable Relational Dynamics useful on a target. The same pattern recurs for temporal reorganization in MOOC → Reddit and Reddit → Wikipedia. Thus, the part of sourcelearned Relational Dynamics that is actually reused varies with the target.

## 5.5 Cross-Representation Reuse Generalizes Across Domains and Data Regimes

Relational dynamics reuse recurs across domains and in both directions beyond a single dataset pair. Across seven cross-representation pairs, three target training fractions (5%, 20%, and 100%), and three seeds, mean frozen gains are positive in 18 of the 21 pair–data conditions. Six of the seven pairs improve at all three training fractions, and the three TKG → CTDG pairs yield positive gains in all nine conditions. The reuse observed for Wikipedia ↔ ICEWS14 thus extends to other datasets and target-data regimes.

Substantial improvements occur even with the sourcelearned backbone fixed. AP increases from 0.7338 to 0.9476 for ICEWS14 → Wikipedia at 20% training data, and MRR increases from 0.0594 to 0.1216 for Wikipedia → ICEWS14 at 100%. Achieving these gains across different vocabularies and prediction interfaces without modifying the backbone during target training demonstrates the reusability of source-learned predictive structure.

Frozen transfer.: Table 6 compares mean paired gains (source-frozen − random-frozen) across training fractions. Appendix B provides all scores and standard deviations.

Reuse also provides practical benefits under fine-tuning. Mean gains over target-only training are positive in 16 of 21 conditions. In particular, for ICEWS14 → Wikipedia at 5%, AP improves from 0.7483 to 0.9533, with a paired gain of $+ 0 . 2 0 4 9 \pm 0 . 0 0 4 0$ . For Wikipedia → ICEWS05-15 at 100%, MRR increases from 0.1868 to 0.2383. Source training thus provides benefits not only as fixed features but also when the entire model is trained on the target.

Fine-tuning.: Table 7 compares mean paired gains (source-finetune − target-only) across training fractions. Appendix B provides all scores and standard deviations.

Cross-representation reuse persists across target training fractions. Six source–target pairs yield positive frozen gains at 5%, 20%, and 100% of target training data, showing that the utility of source-learned predictive structure is not restricted to a particular target-data regime.

TABLE 7  
Mean fine-tuning gains across target training fractions.
<table><tr><td>Source → Target</td><td>Metric</td><td>5% train 20% train 100% train</td><td></td></tr><tr><td>Wikipedia → ICEWS14</td><td>MRR</td><td>-0.0189+0.0280</td><td>+0.0165</td></tr><tr><td> $\mathrm { R e d } \hat { \mathrm { d i t } } \to \mathrm { I C E W S 1 4 }$ </td><td>MRR</td><td>-0.0024 -0.0002</td><td>+0.0371</td></tr><tr><td> $\mathrm { M O O C } \to \mathrm { G D E L T }$ </td><td>MRR</td><td>-0.0201+0.0230</td><td>-0.0036</td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { W i k i p e d i a }$ </td><td>AP</td><td>+0.2049 +0.0307</td><td>+0.0019</td></tr><tr><td> $\mathrm { G D E L T } \to \mathrm { R e d d i t } ^ { \bullet }$ </td><td>AP</td><td>+0.0500+0.0443</td><td>+0.0095</td></tr><tr><td> $W \mathrm { i k i p e d i a }  \mathrm { I C E W S 0 5 - } 1 5$ </td><td>MRR</td><td>+0.0244 +0.0491</td><td>+0.0515</td></tr><tr><td> $\mathrm { I C E W S 0 5 - 1 5 }  \mathrm { M O O C }$ </td><td>AP</td><td>+0.0563 +0.0786</td><td>+0.0776</td></tr></table>

Within this scope of reuse, source and adaptation choices still matter. Frozen gains for MOOC → GDELT are negative at all three training fractions, and Reddit → ICEWS14 at 5% yields a frozen gain of $+ 0 . 0 5 5 9$ but a fine-tuning gain of −0.0024. The utility of a fixed backbone and the gain from fine-tuning over target-only training are distinct outcomes.

Overall, relational dynamics are broadly reusable across representation boundaries, with utility varying by source– target combination and training conditions. The observation that sources crossing the same representation boundary provide different utility raises the question of whether representation membership alone sufficiently distinguishes useful sources. Section 5.6 tests this directly by connecting both same- and cross-representation sources to each target.

## 5.6 Representation Identity Does Not Fully Determine Source Utility

How well does representation identity distinguish sources whose learned knowledge is useful for a target? Across 30 directed transfers among six datasets, we compare the utility of two same-representation and three cross-representation sources for each target.

Cross-representation sources are useful for multiple targets. Mean paired gains are positive in 14 of the 18 crossrepresentation transfers, and five of the six targets obtain positive reuse from at least one cross-representation source. We go further by comparing the utility ordering of sameand cross-representation sources for each target.

Table 8 reports the mean frozen paired gain $\Delta _ { S  T }$ across three seeds. Italics indicate same-representation transfer, and bold denotes the highest gain for each target.

Figure 4 shows that cross-representation sources can provide higher utility than same-representation sources for the same target. For Wikipedia, all three TKG sources—ICEWS05-15 (+0.1577), ICEWS14 (+0.1544), and GDELT (+0.1454)—outperform both CTDG sources, Reddit (+0.1187) and MOOC (+0.0030), in mean gain. For Reddit, Wikipedia provides the highest gain, but all three TKG sources outperform the same-representation source MOOC. For MOOC, the three TKG sources rank after Wikipedia and before Reddit. Thus, for all three CTDG targets, at least one cross-representation source outperforms a same-representation source; for Wikipedia, every cross-representation source outperforms every samerepresentation source.

TABLE 8  
Mean frozen gains in the complete source–target matrix.
<table><tr><td>Source ↓ / Target →</td><td>Wikipedia (AP)</td><td>Reddit (AP)</td><td>MOOC (AP)</td><td>ICEWS14 (MRR)</td><td>GDELT (MRR)</td><td>ICEWS05-15 (MRR)</td></tr><tr><td>Wikipedia</td><td></td><td>+0.0858</td><td>+0.0722</td><td>+0.0622</td><td>-0.0429</td><td>+0.0649</td></tr><tr><td>Reddit</td><td>+0.1187</td><td></td><td>+0.0524</td><td>+0.0496</td><td>-0.0276</td><td>+0.0620</td></tr><tr><td>MOOC</td><td>+0.0030</td><td>+0.0096</td><td></td><td>-0.0016</td><td>-0.0006</td><td>+0.0014</td></tr><tr><td>ICEWS14</td><td>+0.1544</td><td>+0.0710</td><td>+0.0689</td><td></td><td>-0.0014</td><td>+0.1811</td></tr><tr><td>GDELT</td><td>+0.1454</td><td>+0.0771</td><td>+0.0614</td><td>+0.0819</td><td></td><td>+0.0974</td></tr><tr><td>ICEWS05-15</td><td>+0.1577</td><td>+0.0760</td><td>+0.0680</td><td>+0.0812</td><td>-0.0027</td><td></td></tr></table>

![](images/3924e70fbde0c1ef9a38f74be76768ef57dfb51b0872b72ead2733b31c877132.jpg)  
(d) ICEWS14

![](images/deb3b94a1c4b80109d6f675af08250a7320bef9e3054fc9d2bedb9ef7306df25.jpg)

![](images/c0e56c3ed7f4a6e301e7219978fde46da8bbeecfeefb1c0bfbd2c3917912c924.jpg)

![](images/b01cf2e11f9adddc816df229149fc416e4b4cb5f3ac1e34ce8a9d9c6548a261d.jpg)

![](images/20e2d3c0e45119cbf50a3ddefdc55951c903af26d6b7579dde57516c5dd95a7e.jpg)

![](images/991a0cda35769485b828304aa4ab4aaf3e8ec0efb2f5a194bd6298c70b893931.jpg)  
Fig. 4. Source utility rankings across representation membership. Sources are ordered by mean frozen gain (source-frozen − random-frozen) for each target. Large diamonds denote three-seed means and small points individual seeds; orange denotes same-representation sources and teal cross-representation sources. For Wikipedia, all three cross-representation sources outperform both same-representation sources in mean gain. The top row reports AP for CTDG targets and the bottom row MRR for TKG targets, with common axis limits within each row.

Restricting candidate pretraining sources by representation membership can exclude more useful learned knowledge. Restricting Wikipedia’s candidates to CTDG sources would exclude all three highest observed transfer gains. This result shows that representation identity alone is insufficient for organizing sources of reusable knowledge.

Source orderings vary by target. Same-representation sources dominate for ICEWS14, where GDELT and ICEWS05-15 provide the two highest gains, and for ICEWS05-15, where ICEWS14 and GDELT do so. For GDELT, all sources yield nonpositive mean gains regardless of representation. Representation groupings and source utility rankings can therefore either align or cut across one another.

In summary, temporal graph representation can be associated with source utility but does not fully determine its ordering. Restricting foundation-model pretraining sources to representation-specific silos in advance can miss useful knowledge outside those silos. These findings motivate studying which relational dynamics are reusable between sources and targets as a principle for organizing learning resources.

## 5.7 Unaligned Source-Learner Fusion

We have established that predictive structure learned from different temporal graph representations can be reused across representation boundaries. We further test whether predictive knowledge independently learned from different sources can be used jointly for a target query. We pretrain history encoders separately on ICEWS05-15, a TKG of political events, and LastFM, a CTDG of music listening, and train target-specific interfaces using 5% of Wikipedia training data. Source entities and relation semantics are not aligned; with each encoder frozen, we combine the two learners’ standardized prediction scores. The TKG source and fusion weights are selected using validation data.

TABLE 9  
Source-learner fusion from ICEWS05-15 and LastFM to Wikipedia (CTDG, 5% training data). Historical-negative AP and AUC on a subsequent interval, reported as means ± sample standard deviations over seeds 7, 17, and 37.
<table><tr><td>Source learner(s)</td><td>AP</td><td>AUC</td></tr><tr><td>ICEWS05-15 (TKG)</td><td> $0 . 6 7 7 8 \pm 0 . 0 0 4 5$ </td><td> $0 . 6 6 9 5 \pm 0 . 0 1 3 0$ </td></tr><tr><td>LastFM (CTDG)</td><td> $0 . 7 5 7 4 \pm 0 . 0 1 6 4$ </td><td> $0 . 7 8 5 9 \pm 0 . 0 2 5 0$ </td></tr><tr><td>ICEWS05-15 + LastFM</td><td> $\mathbf { 0 . 7 8 8 2 \pm 0 . 0 1 6 2 }$ </td><td> $\mathbf { 0 . 8 1 7 7 \pm 0 . 0 0 9 2 }$ </td></tr></table>

Table 9 reports evaluation on a subsequent Wikipedia interval with the models and fusion weights fixed. The fused model achieves higher AP and AUC than either individual source in all three seeds. Mean AP increases from 0.7574 for LastFM alone to 0.7882, and AUC from 0.7859 to 0.8177. In particular, ICEWS05-15 provides additional predictive utility when combined with the LastFM learner despite its standalone AP of 0.6778.

These results show that predictive learners trained on relational histories from different domains and representations can be used jointly for a target query without aligning source data or entity and relation semantics. Fusion combines the prediction scores produced by the transferred learners, rather than integrating the original graphs or their internal representation spaces.

## 5.8 Practical Reuse Without Entity Alignment

We connect the reusability established above to utility in recommendation and security detection. Both PoCs finetune a pretrained history encoder on a target task without aligning source and target entities.

## Unaligned TKG Pretraining for Temporal Recommendation

Can semantically distant data, such as political events and music listening, be connected as learning resources? We finetune an ICEWS14-pretrained history encoder for next-item recommendation on LastFM. Source and target share neither entity and relation vocabularies nor item–entity alignment. We compare against the same randomly initialized model at 5% and 20% of target training data, evaluating all 1,000 items for 256 held-out test queries.

In Table 10, Recall@10 improves in every seed at both training fractions. At 20%, mean Recall@10 increases by 19.53 percentage points, and NDCG@10 also increases. At 5%, NDCG@10 decreases, indicating different changes in hit rate and top-rank quality.

The improvements occur for repeat listening; the pretrained model’s Recall@10 for previously unheard items is zero at both training fractions. Appendix D provides results by recommendation situation and seed, along with the personal-recency heuristic, which achieves higher overall scores. This PoC shows that an external TKG without a semantic connection to the target can serve as a pretraining resource that improves the hit rate of the same recommendation model.

TABLE 10  
ICEWS14 → LastFM full fine-tuning. Each score cell reports Recall@10 / NDCG@10 as mean ± sample standard deviation. ∆ is the within-seed difference between pretrained fine-tuning and target-only training, using seeds 7, 17, and 37.
<table><tr><td>Target train</td><td colspan="3">Target-only ICEWS14 fine-tuning</td></tr><tr><td> $5 \% \cdot 1 , 1 4 7$ </td><td> $0 . 2 5 0 0 \pm 0 . 0 0 0 0 \mathrm { ~ / ~ }$   $0 . 1 9 8 1 \pm 0 . 0 0 2 1$ </td><td> $0 . 3 1 1 2 \pm 0 . 0 1 5 8 / + 0 . 0 6 1 2 \pm 0 . 0 1 5 8 /$   $0 . 1 6 9 3 \pm 0 . 0 0 4 6$ </td><td>Paired ∆  $- 0 . 0 2 8 7 \pm 0 . 0 0 3 3$ </td></tr><tr><td> $2 0 \% \cdot 4 { , } 5 8 8$ </td><td> $0 . 2 7 0 8 \pm 0 . 0 3 0 3 /$   $0 . 2 1 2 8 \pm 0 . 0 2 9 1$ </td><td> $0 . 4 6 6 1 \pm 0 . 0 3 1 6 / + 0 . 1 9 5 3 \pm 0 . 0 5 1 7 /$   $0 . 2 6 3 3 \pm 0 . 0 3 9 1$ </td><td> $+ 0 . 0 5 0 5 \pm 0 . 0 6 0 9$ </td></tr></table>

## External Threat Intelligence Before Integration

This PoC connects relational dynamics reuse to an industrial question: can an external temporal source without a direct connection to the target system serve as a learning resource for internal attack detection without semantic alignment?

Figure 5(a) contrasts external threat intelligence with internal security logs. Malware, attack techniques, and intrusion procedures accumulate externally, while authentication, network, and process activities are recorded internally; the entities and schemas of the two sources do not directly correspond. Figure 5(b) shows two routes for using them. For example, Microsoft Sentinel supports STIX/TAXII-based CTI ingestion, management of ATT&CKrelated attack patterns and indicator enrichment, and alert generation by matching internal events against IOCs [17]. The conventional path in Figure 5(b) summarizes these integration functions. In addition to connecting the sources through indicator matching, technique tagging, and log enrichment, this PoC examines a route that pretrains relational dynamics on external attack histories and adapts them to internal logs. The key is to use external CTI as a learning resource before integration is complete.

Figure 5(c) represents this utility as discovery coverage under the same alert budget. Repeatedly detecting similar attack events and uncovering attack activity involving different users and hosts provide different information for investigation. We therefore evaluate attack-event counts alongside the coverage of unique hosts, users, and activity groups discovered within a limited number of alerts.

The source comprises 371 temporal-relational events directly extracted from 33 public intrusion reports published by The DFIR Report [18], obtained through report links from ChronoCTI [19]. The target task is red-team event detection in LANL authentication logs. Only the encoder learned from report-specific strict-past histories is transferred; no entity mapping or common relation vocabulary is constructed. The LANL-only and CTI-initialized models undergo full finetuning for 10 epochs with identical target-private initialization and data. Source training duration is selected by reportlevel validation BCE, and the target checkpoint is selected by validation Recall@1% FPR.

In Table 11, host, user, and activity-group coverage improves in all three seeds with 20% training data under the same top-2500 alert budget. Mean host coverage increases from 6.67 to 8.67, with a paired increase of $+ 2 . 0 0 \pm 1 . 0 0 .$ More diverse attack-related entities and activities are uncovered even when fewer attack events are detected.

Appendix E provides complete three-seed results across training fractions and alert budgets, along with the evaluation settings. At 5%, mean host coverage increases from 2.00 to 2.33 for top-500 and top-1,000 alerts. At 100%, however, mean event and host coverage is lower than the control at every evaluated alert budget. Alert counts refer to retrospective ranking of a test set with sampled background events, not actual daily operational alert volumes or confirmed intrusion incident counts.

![](images/a975c666aebcdb530b203a10857457502b86c53f10826e7ff7a0ea41c45feace.jpg)  
Fig. 5. Conceptual routes for using external CTI before integration. (a) Distinct external CTI and internal logs. (b) Learning and reusing relationa dynamics without direct alignment. (c) Broader discovery of attack-related entities and activities under the same alert budget.

TABLE 11  
Comparison with 20% LANL training data (29,847 events) and top-2500 alerts. Cells show LANL-only → CTI, with means in the final row. Hosts are deduplicated across both endpoints of detected red-team events. Activity groups are separated when the gap between attack events for the same user exceeds one hour.
<table><tr><td>Seed</td><td>Hosts</td><td>Users</td><td>Activity groups</td><td>Attack events</td></tr><tr><td>7</td><td>6 → 8</td><td>6 → 7</td><td>6 → 7</td><td>10 → 7</td></tr><tr><td>17</td><td>6 → 9</td><td>5 → 8</td><td>6 → 8</td><td>7 → 8</td></tr><tr><td>37</td><td>8 → 9</td><td>4 → 8</td><td>6 → 8</td><td>15 → 10</td></tr><tr><td></td><td>Mean 6.67 → 8.67 5.00 → 7.67 6.00 → 7.67 10.67 → 8.33</td><td></td><td></td><td></td></tr></table>

These results show that reusing an encoder pretrained on an external temporal source without direct alignment can broaden discovery coverage in internal detection under some conditions with limited target data and alert budgets. This PoC alone cannot uniquely attribute the observed gains to relational dynamics. However, source and target use no entity, relation-vocabulary, or attack-technique alignment, and the preceding experiments established that crossrepresentation utility varies with source relational organization. The results are therefore consistent with the interpretation that relational dynamics learned from external temporal sources can contribute to this utility.

## 6 DISCUSSION AND LIMITATION

Temporal graph representations are interfaces through which Relational Dynamics are observed and queried; they do not necessarily define the boundaries of what can be learned. The bidirectional reuse in Section 5.2 and generalization across domains and training fractions in Section 5.5 show that predictive structure learned from CTDGs and TKGs is not confined to its representation. Source utility rankings in the complete transfer matrix in Section 5.6 also cross representation boundaries. These findings distinguish the divisions of a field that has treated CTDGs and TKGs separately from the boundaries of what its models learn.

The functional characterization further shows that reusable structure does not reduce to one inventory shared by every source and target. The original event sequence contributes consistently for ICEWS14 → ICEWS05-15, whereas reorganizing event sequence, temporal placement, or relation direction provides higher utility for ICEWS05-15 → Wikipedia. Learned and Reusable Relational Dynamics are therefore distinct: reuse depends not only on what is learned from a source, but also on the target and query to which it is applied.

Temporal graph foundation models should distinguish generalization across domains within one representation from learning Relational Dynamics that remain useful when the representation changes. Our results provide grounds for treating the latter as an independent learning objective. A future learner can retain representation-specific interfaces, accumulate Relational Dynamics from diverse sources, and select the part useful for a target condition.

Section 5.7 extends the reuse of relational dynamics to the combination of independently trained predictive learners from heterogeneous temporal sources. Predictive knowledge from different sources can thus be used jointly for a target query without first integrating data formats or entity and relation semantics. The experiment demonstrates utility at the score level; it does not identify a unique advantage of heterogeneous sources or a decomposition into specific dynamics mechanisms.

## 6.1 Limitation

Although this study poses relational dynamics reuse across temporal graph representations as a general research problem, its empirical analysis focuses on CTDGs and TKGs, which represent different observation and prediction interfaces in temporal graph learning. The results should therefore be interpreted as showing that representation differences are not absolute boundaries of learnable predictive structure between CTDGs and TKGs, rather than as claiming that all temporal graph representations share identical dynamics. Extending this perspective to other temporal graph representations remains future work.

## 7 CONCLUSION

This study demonstrates that aspects of Relational Dynamics can be learned and reused across temporal graph representations. Bidirectional frozen transfer between CTDGs and TKGs shows that learned predictive structure remains useful after changes in entity vocabulary and prediction interface. Controlled reorganizations of source histories show that the contributing organization varies across source– target conditions and that Learned Relational Dynamics are not reused in their entirety. Reuse across domains and training fractions and utility orderings across 30 source– target directions show that representation identity does not sufficiently explain the boundaries of learning and reuse.

These results extend the unit of study in temporal graph learning from representation-specific tasks to the relational dynamics reusable beneath them. Combining independently trained TKG and CTDG learners further shows that predictive knowledge can be used jointly on a third target without aligning the original sources’ entities or relation semantics. The recommendation and security detection PoCs show that such reuse can extend to using temporal sources without direct alignment as new learning resources. The scope of a temporal graph foundation may be determined not only by which representations it supports, but also by which relational dynamics it can learn and reuse across different observations and queries.

Acknowledgements. The authors have no acknowledgements to declare.

Funding. This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

Data availability. The core experiments use seven public datasets: Wikipedia, Reddit, MOOC, and LastFM from JODIE; ICEWS14 and ICEWS05-15 from TLogic; and GDELT from CyGNet. Table S16 lists their sources. The security PoC uses public LANL authentication and red-team data and timelines extracted from public DFIR reports [18] located through ChronoCTI [19]. Original reports remain with their publishers, and the derived annotations are not included in the code release. Retrieval, annotation, preprocessing, split, and evaluation instructions will be provided with the experiment code.

Code availability. The implementation and experiment configurations will be made available at https://github.com/ SnowyPainter/rdynamics-public.

## APPENDIX A

## EXPERIMENTAL DETAILS

Training fractions and history.: The 32,768-event splits preserve timestamp boundaries. The 5/20/100% target training conditions retain the most recent training events, including all events at the boundary timestamp, so the actual fractions can be slightly larger. Excluded training events are removed from both training and evaluation histories. Sources use 100% of their training data. Validation/test periods, query sets, and entity vocabularies remain identical across training fractions. During rolling evaluation, validation queries may use earlier validation observations, and test queries may use earlier validation and test observations; simultaneous and future events are excluded. Entity vocabularies are independent across datasets.

Candidates and filtering in the 32,768-event experiments.: CTDG training draws one negative per positive uniformly with replacement from the destinations in the retained training data. Validation and test also sample one destination but use the destination set of the entire dataset; true or observed edges are not explicitly excluded. TKG training samples one entity uniformly per query, excluding training answers at the same timestamp. Validation uses the true answer and 100 distinct negatives; test uses all canonical entities. Validation/test filtering excludes other true answers with the same timestamp, relation, and fixed endpoint. Answer lists are constructed from all splits within the 32,768-event subset, including excluded training facts, but are separate from model input histories. We report optimistic ranks and also store average-tie ranks. CTDG evaluation RNG seeds are seed +1000 for validation and seed +2000 for test; the TKG validation seed is 123456.

Training budget.: The 10 epochs are a fixed budget chosen for computational cost, not early stopping. The formal transfer experiments select the validation-best checkpoint. Equal event and epoch budgets do not imply equal numbers of optimizer updates across representations. The GraphMixer and TGAT controls are random-frozen instances of the corresponding backbone.

Bidirectional DyGFormer transfer settings.: Sections 5.2 and 5.3 train on the full target training set for 10 epochs. ICEWS14 has 63,685/13,823/13,222 training/validation/test facts, and Wikipedia has 70,000/15,000/15,000 events. Table S1 compares checkpoint selection and evaluation scope across experiments.

The existence experiment evaluates previously validation-selected checkpoints on the full test set without retraining or reselection. ICEWS14 test filtering excludes other true answers with the same timestamp, relation, and fixed endpoint across the original training, validation, and test splits; this answer list is separate from the strict-past input history. We report optimistic ranks and also store average-tie ranks. The source-organization experiment on ICEWS14 uses timestamp filtering based only on validation answers and average-tie ranks. Wikipedia uses one random destination per positive as a negative.

TABLE S1  
Checkpoint selection and evaluation scope for bidirectional DyGFormer experiments.
<table><tr><td>Direction</td><td>Target checkpoint selection</td><td>Existence evaluation</td><td>Source organization evaluation</td></tr><tr><td>Wiki → ICEWS14</td><td>100-negative MRR on 256 bidirec- Full test: 26,444 queries × 7,128 tional queries from 128 fixed vali- entities dation facts</td><td></td><td>Same 256 validation queries used for selection × 7,128 entities</td></tr><tr><td>ICEWS14 → Wiki</td><td>AP on all 15,000 validation events</td><td>AP and AUC on all 15,000 test events</td><td>AP and AUC on all 15,000 test events</td></tr></table>

The reverse-direction source is an ICEWS14 checkpoint without prior Wiki pretraining. The intact, time-shuffled, and partner-shuffled sources in the source-organization experiment are trained from identical initialization for 10 epochs, and their final checkpoints are transferred. Time shuffle preserves directed-pair and relation frequencies and the timestamp distribution; partner shuffle preserves node activity within time bins.

Partial partner-corruption control.: Within each timestamp–relation group in ICEWS14 training data, events are selected with the specified probability, and their destinations are permuted among the selected events. Source initialization, optimizer, and training budget remain identical, and the final epoch-10 checkpoint is used. All source models are evaluated with the original ICEWS14 history and identical queries and negative candidates. Strength selection uses 2,048 bidirectional queries from 1,024 validation facts that do not overlap the earlier 256-query diagnostic set (fact-sampling seed 20261005, 100-negative sampling seed 654321, average-tie ranks). Because this set is used to select strength, it is not treated as an independent validation set.

The relative difference is computed as |M<sup>partial</sup><sub>S</sub> M<sup>intact</sup><sub>S</sub> |/M<sup>intact</sup><sub>S</sub> . We select the strongest common level whose difference is within 3% for all three seeds. Table S2 yields a selected level of 50%.

This grid was designed as a follow-up after observing the target pilot at 25% corruption with seed 7. We then fixed the grid selection rule, selected 50% using source scores alone, and evaluated target results for that strength only after selection. We reuse the intact target results under the same conditions and apply identical target-private initialization, training budget, validation selection, and full-test evaluation to the corrupted condition. Corrupted event sets are not guaranteed to be nested across strengths, and results at one selected strength are not interpreted as a target dose– response.

## APPENDIX B

## FULL RESULTS ACROSS DOMAINS AND DATA REGIMES

This appendix provides the complete results for Section 5.5. Each value is the mean ± sample standard deviation across seeds 7, 17, and 37; paired gains are computed between matched conditions within each seed.

Frozen transfer.: Table S3 provides the full scores for each condition. R denotes random-frozen, S denotes sourcefrozen, and parentheses show paired gains (S − R).

Fine-tuning.: Table S4 provides the full scores for each condition. Source denotes source-finetune, and paired gains are source-finetune − target-only.

## APPENDIX C

## CROSS-BACKBONE TRANSFER RESULTS

Table S5 provides the scores and paired gains underlying the GraphMixer and TGAT results in Figure 2. Scores are means across seeds 7, 17, and 37, and paired gains are the mean ± sample standard deviation of within-seed source-frozen − random-frozen differences. DyGFormer scores appear in Table 2 in the main text.

## APPENDIX D

## RECOMMENDATION POC DETAILS

Table S6 provides per-seed results for Section 5.8. All conditions use identical validation/test queries and are evaluated with seeds 7, 17, and 37. Training fractions refer to supervision and initial training history, not reductions in the validation/test interactions observed during evaluation. An unheard item is one for which the corresponding user–item interaction is absent from the permitted history preceding the query (5%: 61 queries; 20%: 59 queries).

Table S7 reports heuristic scores on the same queries and candidate sets. Popularity ranks items by cumulative interaction count, personal frequency by each user’s listening count per item, and personal recency by each user’s most recent listening time per item. All use only retained training history and validation/test history preceding the query. Heuristics use the entire permitted past, whereas the model uses up to 63 recent events per endpoint. Tie handling is identical to that of the model, and the heuristics have no training seeds.

Table S8 reports reuse gains by recommendation situation. Repeat listening denotes a query whose user–item interaction occurs in the permitted history preceding that query.

## APPENDIX E

## SECURITY POC DETAILS

Table S9 provides the complete results across training fractions and alert budgets for Section 5.8. Target training subsets are nested samples drawn using a fixed, labelblind permutation; unselected training events are excluded from both supervision and history. Validation and test sets remain unchanged, and only observed strict-past events are added to the history. Source training duration is selected by

TABLE S2  
Selection of partial corruption strength based on source performance.
<table><tr><td>Nominal strength</td><td>Relative source MRR difference: seeds 7 / 17 / 37</td><td>All seeds within tolerance</td><td>Selected</td></tr><tr><td>25%</td><td>0.725% / 0.584% / 0.832%</td><td>Yes</td><td></td></tr><tr><td>50%</td><td>0.503% / 0.252% / 0.946%</td><td>Yes</td><td>Yes</td></tr><tr><td>75%</td><td>2.817% / 4.774% / 3.610%</td><td>No</td><td></td></tr></table>

TABLE S3

Full frozen-transfer scores and paired gains.
<table><tr><td>Source → Target</td><td>Metric</td><td>5% train</td><td>20% train</td><td>100% train</td></tr><tr><td>Wikipedia → ICEWS14</td><td>MRR</td><td> $\mathrm { ~ R ~ } 0 . 0 6 2 7 \pm 0 . 0 0 9 3$   $\mathsf { S } 0 . 1 1 2 1 \pm 0 . 0 0 7 9$   $\mathbf { \left( + 0 . 0 4 9 4 \pm 0 . 0 1 3 0 \right) }$ </td><td> $\mathrm { ~ R ~ } 0 . 0 7 0 9 \pm 0 . 0 0 5 4$   $\mathrm { \ S 0 . 1 3 8 4 \pm 0 . 0 0 5 9 }$   $\mathbf { ( + 0 . 0 6 7 5 \pm 0 . 0 0 9 6 ) }$ </td><td> $\mathrm { ~ R ~ } 0 . 0 5 9 4 \pm 0 . 0 0 0 9$   $\mathrm { \ S 0 . 1 2 1 6 \pm 0 . 0 0 9 3 }$   $\mathbf { ( + 0 . 0 6 2 2 \pm 0 . 0 0 9 5 ) }$ </td></tr><tr><td>Reddit → ICEWS14</td><td>MRR</td><td> $\mathrm { ~ R ~ } 0 . 0 6 2 7 \pm 0 . 0 0 9 3$   $\mathsf { S } 0 . 1 1 8 7 \pm 0 . 0 1 5 2$   $\mathbf { \left( + 0 . 0 5 5 9 \pm 0 . 0 0 6 4 \right) }$ </td><td> $\mathrm { ~ R ~ } 0 . 0 7 0 9 \pm 0 . 0 0 5 4$   $\mathrm { ~ S ~ } 0 . 1 1 8 0 \pm 0 . 0 2 4 7$   $\mathbf { ( + 0 . 0 4 7 1 \pm 0 . 0 2 5 9 ) }$ </td><td> $\mathrm { ~ R ~ } 0 . 0 5 9 4 \pm 0 . 0 0 0 9$   $\mathrm { ~ S ~ } 0 . 1 0 9 0 \pm 0 . 0 3 6 8$   $\mathbf { ( + 0 . 0 4 9 6 \pm 0 . 0 3 6 2 ) }$ </td></tr><tr><td>MOOC → GDELT</td><td>MRR</td><td> $\mathrm { ~ R ~ } 0 . 1 1 2 3 \pm 0 . 0 1 2 4$   $\mathrm { ~ S ~ } 0 . 0 7 1 4 \pm 0 . 0 1 2 9$   $\mathbf { ( - 0 . 0 4 0 9 \pm 0 . 0 1 2 8 ) }$ </td><td> $\mathrm { ~ R ~ } 0 . 0 8 3 6 \pm 0 . 0 3 0 4$   $\mathrm { ~ S ~ } 0 . 0 4 2 3 \pm 0 . 0 1 0 0$   $\mathbf { ( - 0 . 0 4 1 3 \pm 0 . 0 3 8 0 ) }$ </td><td> $\mathrm { ~ R ~ } 0 . 1 0 5 6 \pm 0 . 0 1 7 5$   $\mathrm { ~ S ~ } 0 . 1 0 5 0 \pm 0 . 0 1 4 2$   $\mathbf { \left( - 0 . 0 0 0 6 \pm 0 . 0 0 3 8 \right) }$ </td></tr><tr><td>ICEWS14 → Wikipedia</td><td>AP</td><td> $\mathrm { ~ R ~ } 0 . 7 3 1 4 \pm 0 . 0 1 5 4$   $\hphantom { - } S 0 . 9 3 5 9 \pm 0 . 0 0 2 5$   $\mathbf { \left( + 0 . 2 0 4 6 \pm 0 . 0 1 3 4 \right) }$ </td><td> $\mathrm { ~ R ~ } 0 . 7 3 3 8 \pm 0 . 0 0 7 6$   $\mathrm { ~ S ~ } 0 . 9 4 7 6 \pm 0 . 0 0 6 0$   $\mathbf { ( + 0 . 2 1 3 8 \pm 0 . 0 0 3 4 ) }$ </td><td> $\mathrm { ~ R ~ } 0 . 8 0 9 9 \pm 0 . 0 0 2 8$   $\mathrm { ~ S ~ } 0 . 9 6 4 3 \pm 0 . 0 0 5 1$   $\mathbf { ( + 0 . 1 5 4 4 \pm 0 . 0 0 2 8 ) }$ </td></tr><tr><td>GDELT → Reddit</td><td>AP</td><td> $\mathrm { ~ R ~ } 0 . 8 5 6 4 \pm 0 . 0 0 6 7$   $\hphantom { - } S 0 . 9 1 1 7 \pm 0 . 0 0 5 0$   $\mathbf { ( + 0 . 0 5 5 3 \pm 0 . 0 1 0 3 ) }$ </td><td> $\mathrm { ~ R ~ } 0 . 8 6 1 4 \pm 0 . 0 0 6 2$   $\mathrm { ~ S ~ } 0 . 9 3 0 6 \pm 0 . 0 0 2 7$   $\mathbf { \left( + 0 . 0 6 9 1 \pm 0 . 0 0 4 2 \right) }$ </td><td> $\mathrm { ~ R ~ } 0 . 8 7 8 4 \pm 0 . 0 0 1 3$  S 0.9555 ± 0.0033  $\mathbf { ( + 0 . 0 7 7 1 \pm 0 . 0 0 2 0 ) }$ </td></tr><tr><td>Wikipedia → ICEWS05-15 MRR</td><td></td><td> $\mathrm { ~ R ~ } 0 . 0 7 1 8 \pm 0 . 0 0 1 2$   $\mathbf { S } \mathbf { 0 . 1 1 7 2 } \pm 0 . 0 0 9 1$   $\mathbf { ( + 0 . 0 4 5 4 \pm 0 . 0 0 8 6 ) }$   $\mathrm { ~ R ~ } 0 . 6 0 9 4 \pm 0 . 0 0 2 9$ </td><td> $\mathrm { ~ R ~ } 0 . 0 9 2 9 \pm 0 . 0 0 3 9$   $\hphantom { - } 5 0 . 1 4 3 9 \pm 0 . 0 1 5 4$   $\mathbf { \left( + 0 . 0 5 1 0 \pm 0 . 0 1 6 2 \right) }$ </td><td> $\mathrm { ~ R ~ } 0 . 0 6 4 5 \pm 0 . 0 0 1 3$   $\mathrm { \ S 0 . 1 2 9 4 \pm 0 . 0 0 7 9 }$   $\mathbf { \left( + 0 . 0 6 4 9 \pm 0 . 0 0 8 7 \right) }$ </td></tr><tr><td>ICEWS05-15 → MOOC</td><td>AP</td><td> $\mathrm { ~ S ~ } 0 . 6 5 4 8 \pm 0 . 0 2 8 8$   $\mathbf { ( + 0 . 0 4 5 3 \pm 0 . 0 3 0 8 ) }$ </td><td> $\mathrm { ~ R ~ } 0 . 6 3 0 8 \pm 0 . 0 0 3 4$   $\mathsf { S } 0 . 6 7 0 1 \pm 0 . 0 1 7 7$   $\mathbf { \left( + 0 . 0 3 9 4 \pm 0 . 0 1 4 5 \right) }$ </td><td> $\mathrm { ~ R ~ } 0 . 6 1 7 5 \pm 0 . 0 0 8 2$   $\mathrm { ~ S ~ } 0 . 6 8 5 5 \pm 0 . 0 1 1 0$   $\mathbf { ( + 0 . 0 6 8 0 \pm 0 . 0 1 1 5 ) }$ </td></tr></table>

BCE over epochs 0–100 using 26 training reports and seven validation reports, followed by retraining on all 33 reports.

The first 30 days of LANL are split into training [0, 9), validation [9, 14), and test [14, 30) days. Events matching the public red-team records are retained, and the remainder are deterministically sampled at 1/1024. The test set contains 263,317 events, including 104 red-team events covering 71 hosts, 51 users, and 66 activity groups. Host counts denote the endpoints of attack events, not confirmed compromise of each host; activity groups are likewise not actual incident IDs. Top-K denotes retrospective ranking over the entire sampled test set.

## APPENDIX F

## SAME-DATASET REPRESENTATION DIAGNOSTIC

Table S10 reports the three-seed results for the diagnostic in Section 5.2.

The same-dataset diagnostic uses a shared MLP scorer over history frequency and recency statistics and adapts the entire scorer, separately from frozen transfer. Both source conditions use matched initialization, optimizer, negative sampling, and training budgets, with 10 pretraining epochs on the same 50,080 earlier events. The relationhidden condition disables relation-specific statistics while retaining endpoints, timestamps, and event multiplicity. All conditions then adapt for 10 epochs on the same 735 subsequent events (approximately 1% of training events), with the source period supplied as common input history only. Source, adaptation, validation, and test periods are chronologically separated; evaluation uses bidirectional timestamp-filtered MRR on the full test set.

## APPENDIX G

## FUNCTIONAL PROFILES OF REUSABLE RELA-TIONAL DYNAMICS

This appendix reports candidate regularities in the data, the construction and distributional effects of controlled variants, comparisons on the original source task, and the complete source–target results. Table 5 in the main text summarizes conditions with consistent directions across seeds and clear effect sizes.

Candidate regularities in the data.: The heuristics in Table S11 use only observations preceding each query and show that different predictive regularities are strong in different datasets. They describe candidate structure in source histories; the following controlled variants distinguish structure learned and reused by the model.

TABLE S4  
Full fine-tuning scores and paired gains.
<table><tr><td>Source → Target</td><td>Metric</td><td>Train</td><td>Target-only</td><td>Source</td><td>Paired ∆</td></tr><tr><td>Wikipedia → ICEWS14</td><td>MRR</td><td>5%</td><td> $0 . 1 5 3 7 \pm 0 . 0 3 3 0$ </td><td> $0 . 1 3 4 8 \pm 0 . 0 1 8 3$ </td><td> $- 0 . 0 1 8 9 \pm 0 . 0 2 2 2$ </td></tr><tr><td>Wikipedia → ICEWS14</td><td>MRR</td><td>20%</td><td> $0 . 1 4 9 9 \pm 0 . 0 2 9 0$ </td><td> $0 . 1 7 7 9 \pm 0 . 0 2 8 6$ </td><td> $+ 0 . 0 2 8 0 \pm 0 . 0 2 7 7$ </td></tr><tr><td>Wikipedia → ICEWS14</td><td>MRR</td><td>100%</td><td> $0 . 1 8 6 0 \pm 0 . 0 3 6 1$ </td><td> $0 . 2 0 2 5 \pm 0 . 0 2 1 8$ </td><td> $+ 0 . 0 1 6 5 \pm 0 . 0 5 3 1$ </td></tr><tr><td>Reddit → ICEWS14</td><td>MRR</td><td>5%</td><td> $0 . 1 5 3 7 \pm 0 . 0 3 3 0$ </td><td> $0 . 1 5 1 2 \pm 0 . 0 3 4 7$ </td><td> $- 0 . 0 0 2 4 \pm 0 . 0 0 5 7$ </td></tr><tr><td>Reddit → ICEWS14</td><td>MRR</td><td>20%</td><td> $0 . 1 4 9 9 \pm 0 . 0 2 9 0$ </td><td> $0 . 1 4 9 7 \pm 0 . 0 1 9 8$ </td><td> $- 0 . 0 0 0 2 \pm 0 . 0 1 5 9$ </td></tr><tr><td>Reddit → ICEWS14</td><td>MRR</td><td>100%</td><td> $0 . 1 8 6 0 \pm 0 . 0 3 6 1$ </td><td> $0 . 2 2 3 1 \pm 0 . 0 3 7 7$ </td><td> $+ 0 . 0 3 7 1 \pm 0 . 0 3 0 8$ </td></tr><tr><td>MOOC → GDELT</td><td>MRR</td><td>5%</td><td> $0 . 1 1 7 8 \pm 0 . 0 1 4 6$ </td><td> $0 . 0 9 7 7 \pm 0 . 0 2 6 7$ </td><td> $- 0 . 0 2 0 1 \pm 0 . 0 1 3 7$ </td></tr><tr><td>MOOC → GDELT</td><td>MRR</td><td>20%</td><td> $0 . 0 8 5 3 \pm 0 . 0 2 4 0$ </td><td> $0 . 1 0 8 3 \pm 0 . 0 0 8 0$ </td><td> $+ 0 . 0 2 3 0 \pm 0 . 0 2 9 4$ </td></tr><tr><td>MOOC → GDELT</td><td>MRR</td><td>100%</td><td> $0 . 1 2 3 8 \pm 0 . 0 0 9 2$ </td><td> $0 . 1 2 0 2 \pm 0 . 0 0 8 4$ </td><td> $- 0 . 0 0 3 6 \pm 0 . 0 0 9 8$ </td></tr><tr><td>ICEWS14 → Wikipedia</td><td>AP</td><td></td><td>5% 0.7483 ± 0.0068</td><td> $0 . 9 5 3 3 \pm 0 . 0 0 3 2$ </td><td> $+ 0 . 2 0 4 9 \pm 0 . 0 0 4 0$ </td></tr><tr><td>ICEWS14 → Wikipedia</td><td>AP</td><td>20%</td><td> $0 . 9 4 0 0 \pm 0 . 0 1 9 1$ </td><td> $0 . 9 7 0 7 \pm 0 . 0 0 1 8$ </td><td> $+ 0 . 0 3 0 7 \pm 0 . 0 2 0 9$ </td></tr><tr><td>ICEWS14 → Wikipedia</td><td>AP</td><td>100%</td><td> $0 . 9 8 3 6 \pm 0 . 0 0 1 5$ </td><td> $0 . 9 8 5 5 \pm 0 . 0 0 1 8$ </td><td> $+ 0 . 0 0 1 9 \pm 0 . 0 0 1 5$ </td></tr><tr><td>GDELT → Reddit</td><td>AP</td><td>5%</td><td> $0 . 8 7 6 0 \pm 0 . 0 0 4 9$ </td><td> $0 . 9 2 6 0 \pm 0 . 0 0 4 0$ </td><td> $+ 0 . 0 5 0 0 \pm 0 . 0 0 7 6$ </td></tr><tr><td>GDELT → Reddit</td><td>AP</td><td>20%</td><td> $0 . 8 9 8 2 \pm 0 . 0 0 1 8$ </td><td> $0 . 9 4 2 5 \pm 0 . 0 0 2 1$ </td><td> $+ 0 . 0 4 4 3 \pm 0 . 0 0 1 8$ </td></tr><tr><td>GDELT → Reddit</td><td>AP</td><td>100%</td><td> $0 . 9 5 7 8 \pm 0 . 0 0 2 0$ </td><td> $0 . 9 6 7 3 \pm 0 . 0 0 1 1$ </td><td> $+ 0 . 0 0 9 5 \pm 0 . 0 0 2 9$ </td></tr><tr><td>Wikipedia → ICEWS05-15</td><td>MRR</td><td>5%</td><td> $0 . 1 3 4 7 \pm 0 . 0 1 8 6$ </td><td> $0 . 1 5 9 1 \pm 0 . 0 1 9 2$ </td><td> $+ 0 . 0 2 4 4 \pm 0 . 0 3 3 7$ </td></tr><tr><td>Wikipedia → ICEWS05-15</td><td>MRR</td><td>20%</td><td> $0 . 1 5 2 7 \pm 0 . 0 2 3 6$ </td><td> $0 . 2 0 1 7 \pm 0 . 0 1 7 7$ </td><td> $+ 0 . 0 4 9 1 \pm 0 . 0 1 3 4$ </td></tr><tr><td>Wikipedia → ICEWS05-15</td><td>MRR</td><td>100%</td><td> $0 . 1 8 6 8 \pm 0 . 0 5 0 7$ </td><td> $0 . 2 3 8 3 \pm 0 . 0 3 9 5$ </td><td> $+ 0 . 0 5 1 5 \pm 0 . 0 4 8 2$ </td></tr><tr><td>ICEWS05-15 → MOOC</td><td>AP</td><td>5%</td><td> $0 . 6 3 6 3 \pm 0 . 0 0 5 8$ </td><td> $0 . 6 9 2 6 \pm 0 . 0 0 8 8$ </td><td> $+ 0 . 0 5 6 3 \pm 0 . 0 1 3 9$ </td></tr><tr><td>ICEWS05-15 → MOOC</td><td>AP</td><td>20%</td><td> $0 . 6 3 3 4 \pm 0 . 0 0 0 1$ </td><td> $0 . 7 1 2 0 \pm 0 . 0 0 6 9$ </td><td> $+ 0 . 0 7 8 6 \pm 0 . 0 0 7 0$ </td></tr><tr><td> $\mathrm { I C E W S 0 5 - 1 5 }  \mathrm { M O O C }$ </td><td>AP</td><td>100%</td><td> $0 . 6 3 5 1 \pm 0 . 0 0 5 2$ </td><td> $0 . 7 1 2 7 \pm 0 . 0 0 6 4$ </td><td> $+ 0 . 0 7 7 6 \pm 0 . 0 1 0 1$ </td></tr></table>

TABLE S5

Bidirectional frozen transfer with GraphMixer and TGAT.
<table><tr><td>Backbone</td><td>Direction</td><td></td><td>Test metric Random-frozen</td><td>Source-frozen</td><td> $\mathrm { P a i r e d } \ \Delta \pm \mathrm { S D }$ </td></tr><tr><td>GraphMixer</td><td> $W \mathrm { i k i } \to \mathrm { I C E W S 1 4 }$ </td><td>MRR</td><td>0.0529</td><td>0.0431</td><td> $- 0 . 0 0 9 9 \pm 0 . 0 1 1 9$ </td></tr><tr><td>GraphMixer</td><td> $\mathrm { I C E W S 1 4 } \to \mathrm { W i k i }$ </td><td>AP</td><td>0.7822</td><td>0.8042</td><td> $\mathbf { + 0 . 0 2 1 9 \pm 0 . 0 1 6 3 }$ </td></tr><tr><td>TGÁT</td><td> $W \mathrm { i k i } \to \mathrm { I C E W S 1 4 }$ </td><td>MRR</td><td>0.0155</td><td>0.0442</td><td> $\mathbf { + 0 . 0 2 8 7 \pm 0 . 0 0 6 5 }$ </td></tr><tr><td>TGAT</td><td> $\mathrm { I C E W S 1 4 } \to \mathrm { W i k i }$ </td><td>AP</td><td>0.7810</td><td>0.8956</td><td> $\mathbf { + 0 . 1 1 4 6 \pm 0 . 0 5 4 8 }$ </td></tr></table>

TABLE S6

Per-seed full fine-tuning results for the recommendation PoC. Each cell reports Recall@10 / NDCG@10.
<table><tr><td>Target train</td><td>Seed</td><td>Target-only</td><td>ICEWS14 fine-tuning</td></tr><tr><td>5%</td><td>7</td><td>0.2500 / 0.2004</td><td>0.2930  / 0.1741</td></tr><tr><td>5%</td><td>17</td><td>0.2500 / 0.1975</td><td>0.3203 / 0.1649</td></tr><tr><td>5%</td><td>37</td><td>0.2500 / 0.1964</td><td>0.3203 / 0.1690</td></tr><tr><td>20%</td><td>7</td><td>0.3047 / 0.2464</td><td>0.4609 / 0.2395</td></tr><tr><td>20%</td><td>17</td><td> $0 . 2 6 1 7 \mathrm { ~ / ~ } 0 . 1 9 8 2$ </td><td>0.4375 / 0.2420</td></tr><tr><td>20%</td><td>37</td><td>0.2461 / 0.1939</td><td>0.5000 / 0.3083</td></tr></table>

Controlled variants and source competence.: Table S12 defines the history organization preserved and reorganized by each variant, and Table S14 reports the resulting changes in pair and transition distributions. The original and variant encoders are evaluated on the same original source history, and only conditions whose relative difference in Eq. (4) is at most 3% for every seed are transferred. Thresholds of 1%, 2%, and 3% retain 8, 14, and 15 conditions, respectively. The 3% criterion was fixed before examining target results.

Complete target results.: For each source, the target was fixed in advance as the direction with the highest mean positive reuse gain in the original frozen-transfer matrix. Table S15 reports all 15 conditions that satisfy the sourcecompetence criterion.

## APPENDIX H PUBLIC DATASET SOURCES

TABLE S7  
Recommendation heuristics under identical evaluation conditions. Each cell reports Recall@10 / NDCG@10.
<table><tr><td>Method</td><td>5% train history</td><td>20% train history</td></tr><tr><td>Item popularity</td><td>0.2461  / 0.1825</td><td>0.2344 / 0.1718</td></tr><tr><td>Personal frequency</td><td>0.4297  / 0.2829</td><td>0.4102 / 0.2625</td></tr><tr><td>Personal recêncy</td><td>0.5078 / 0.3464</td><td>0.5078 / 0.3464</td></tr></table>

TABLE S8

Means ± sample standard deviations and paired gains by recommendation situation. Each score cell reports Recall@10 / NDCG@10; parentheses indicate query counts.
<table><tr><td>Target train Situation (queries)</td><td></td><td colspan="3">Target-only ICEWS14 fine-tuning</td></tr><tr><td>5%</td><td>Repeat listening (195)</td><td> $0 . 3 2 3 1 \pm 0 . 0 0 0 0 $   $0 . 2 5 7 4 \pm 0 . 0 0 2 3$ </td><td> $0 . 4 0 8 5 \pm 0 . 0 2 0 7 $   $0 . 2 2 2 3 \pm 0 . 0 0 6 0$ </td><td> $+ 0 . 0 8 5 5 \pm 0 . 0 2 0 7 \ /$   $- 0 . 0 3 5 1 \pm 0 . 0 0 5 0$ </td></tr><tr><td>5%</td><td>Unheard items (61)</td><td> $0 . 0 1 6 4 \pm 0 . 0 0 0 0 $   $0 . 0 0 8 3 \pm 0 . 0 0 2 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0 \mathrm { ~ / ~ }$   $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $- 0 . 0 1 6 4 \pm 0 . 0 0 0 0 \ /$   $- 0 . 0 0 8 3 \pm 0 . 0 0 2 0$ </td></tr><tr><td>20%</td><td>Repeat listening (197)</td><td> $0 . 3 4 6 9 \pm 0 . 0 3 9 4 \mid$   $0 . 2 7 4 5 \pm 0 . 0 3 8 4$ </td><td> $0 . 6 0 5 8 \pm 0 . 0 4 1 0 \mathrm { ~ / ~ }$   $0 . 3 4 2 1 \pm 0 . 0 5 0 8$ </td><td> $+ 0 . 2 5 8 9 \pm 0 . 0 6 7 2 \ /$   $+ 0 . 0 6 7 6 \pm 0 . 0 7 9 5$ </td></tr><tr><td>20%</td><td>Unheard items (59)</td><td> $0 . 0 1 6 9 \pm 0 . 0 0 0 0$   $0 . 0 0 6 9 \pm 0 . 0 0 1 8$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0 \mathrm { ~ / ~ }$   $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $- 0 . 0 1 6 9 \pm 0 . 0 0 0 0 \mathrm { \Omega } / \mathrm { \Omega }$   $- 0 . 0 0 6 9 \pm 0 . 0 0 1 8$ </td></tr></table>

TABLE S9

Means across seeds 7, 17, and 37 by LANL training fraction and alert budget. Each cell shows LANL-only → CTI. Training fractions of 5%, 20%, and 100% contain 7,461, 29,847, and 149,237 events, respectively.
<table><tr><td>Target train</td><td>Alerts K</td><td>Hosts</td><td>Users</td><td>Activity groups</td><td>Attack events</td></tr><tr><td>5%</td><td>100</td><td> $1 . 3 3  1 . 3 3$ </td><td>0.67 → 0.67</td><td>0.67 → 0.67</td><td>0.67 → 0.67</td></tr><tr><td>5%</td><td>500</td><td>2.00 → 2.33</td><td>1.00 → 1.33</td><td>1.00 → 1.33</td><td>1.00 → 1.33</td></tr><tr><td>5%</td><td>1,000</td><td>2.00 → 2.33</td><td>1.00 → 1.33</td><td>1.00 → 1.33</td><td>1.00 → 1.33</td></tr><tr><td>5%</td><td>2,500</td><td>6.33 → 5.67</td><td>6.33 → 4.67</td><td>6.33 → 4.67</td><td>6.33 → 5.00</td></tr><tr><td>20%</td><td>100</td><td>0.00 → 0.67</td><td>0.00 → 0.33</td><td>0.00 → 0.33</td><td>0.00 → 0.33</td></tr><tr><td>20%</td><td>500</td><td>3.00 → 2.33</td><td>2.00 → 1.67</td><td>2.33 → 1.67</td><td>3.67 → 1.67</td></tr><tr><td>20%</td><td>1,000</td><td>3.33 → 3.33</td><td>2.67 → 2.67</td><td> $3 . 0 0  2 . 6 7$ </td><td>5.67 → 2.67</td></tr><tr><td>20%</td><td>2,500</td><td>6.67 → 8.67</td><td>5.00 → 7.67</td><td> $6 . 0 0  7 . 6 7$ </td><td>10.67 → 8.33</td></tr><tr><td>100%</td><td>100</td><td>16.33 → 15.67</td><td>9.00 → 8.67</td><td> $1 2 . 0 0 \to 1 1 . 0 0$ </td><td>20.67 → 18.00</td></tr><tr><td>100%</td><td>500</td><td>28.00 → 27.00</td><td>18.00 → 15.00</td><td>22.33 → 21.00</td><td>35.33 → 34.67</td></tr><tr><td>100%</td><td>1,000</td><td>34.67 → 33.33</td><td>23.00 → 19.00</td><td>28.67 → 26.67</td><td>43.33 → 42.00</td></tr><tr><td>100%</td><td>2,500</td><td>42.67 → 39.00</td><td>29.33 → 23.33</td><td>36.67 → 33.00</td><td>55.67 → 52.67</td></tr></table>

TABLE S10

Same-dataset pretraining with relations retained or hidden in ICEWS14. Values are means ± standard deviations over three seeds; gains are relative to scratch.
<table><tr><td>Pretraining condition</td><td>Test MRR</td><td>Gain</td></tr><tr><td>No pretraining</td><td> $0 . 3 5 5 1 \pm 0 . 0 1 7 1$ </td><td></td></tr><tr><td>TKG: relations retained</td><td> $0 . 4 5 1 0 \pm 0 . 0 0 1 0$ </td><td>+0.0959</td></tr><tr><td>CTDG-style: relations hidden</td><td> $0 . 4 2 9 7 \pm 0 . 0 0 3 9$ </td><td>+0.0746</td></tr></table>

TABLE S11

Strict-past heuristic AUC by dataset. Parentheses for pair renewal show the fraction of applicable queries.
<table><tr><td>Dataset</td><td>Pair frequency</td><td>Pair recency</td><td>Activity</td><td>Recency</td><td>Latest partner</td><td>1-step transition</td><td>Pair renewal</td><td>Reciprocity</td><td>Neighbor overlap</td></tr><tr><td>Wikipedia</td><td>.631</td><td>.771</td><td>.493</td><td>.826</td><td>.805</td><td>.775</td><td>.781 (60.8%)</td><td>一</td><td>一</td></tr><tr><td>Reddit</td><td>.423</td><td>.370</td><td>.528</td><td>.504</td><td>.475</td><td>.519</td><td>.591 (25.6%)</td><td>一</td><td>一</td></tr><tr><td>MOOC</td><td>.313</td><td>.331</td><td>.414</td><td>.493</td><td>.483</td><td>.536</td><td>.611 (14.8%)</td><td></td><td>一</td></tr><tr><td>ICEWS14</td><td>.389</td><td>.390</td><td>.528</td><td>.541</td><td>.462</td><td>.518</td><td>.699 (14.6%)</td><td>.428</td><td>.570</td></tr><tr><td>GDELT</td><td>.195</td><td>.175</td><td>.464</td><td>.474</td><td>.366</td><td>.491</td><td>.539 (3.3%)</td><td>.342</td><td>.533</td></tr><tr><td>ICEWS05-15</td><td>.485</td><td>.457</td><td>.562</td><td>.571</td><td>.506</td><td>.553</td><td>.628 (21.1%)</td><td>.498</td><td>.573</td></tr></table>

TABLE S12  
Preserved statistics and reorganized history aspects for each controlled variant.
<table><tr><td>Variant</td><td>Preserved</td><td>Reorganized</td></tr><tr><td>Partner</td><td>Event count, relation frequencies, timestamp multiset, Directed partner pairs and successive transitions and binned destination frequencies</td><td></td></tr><tr><td>Sequence</td><td>Directed typed-pair frequencies and source-relation Partner placement and order across slots time slots</td><td></td></tr><tr><td>Time</td><td>Directed typed-pair frequencies and timestamp mul- Assignment of complete relational events to times tiset</td><td></td></tr><tr><td>Direction (TKG)</td><td>Undirected typed pairs, relation frequencies, and Subject-object direction timestamps</td><td></td></tr></table>

TABLE S13  
Performance differences between controlled-variant and original encoders on the original source history.
<table><tr><td>Source</td><td>Variant</td><td>Mean gap</td><td>Maximum gap</td><td>3% criterion</td></tr><tr><td>Wikipedia</td><td>Partner</td><td>31.70%</td><td>34.67%</td><td>Excluded</td></tr><tr><td>Wikipedia</td><td>Sequence</td><td>0.01%</td><td>0.02%</td><td>Included</td></tr><tr><td>Wikipedia</td><td>Time</td><td>0.17%</td><td>0.36%</td><td>Included</td></tr><tr><td>Reddit</td><td>Partner</td><td>7.02%</td><td>7.65%</td><td>Excluded</td></tr><tr><td>Reddit</td><td>Sequence</td><td>0.19%</td><td>0.35%</td><td>Included</td></tr><tr><td>Reddit</td><td>Time</td><td>0.66%</td><td>0.99%</td><td>Included</td></tr><tr><td>MOOC</td><td>Partner</td><td>0.28%</td><td>0.39%</td><td>Included</td></tr><tr><td>MOOC</td><td>Sequence</td><td>0.19%</td><td>0.33%</td><td>Included</td></tr><tr><td>MOOC</td><td>Time</td><td>0.70%</td><td>1.29%</td><td>Included</td></tr><tr><td>ICEWS14</td><td>Partner</td><td>14.62%</td><td>14.62%</td><td>Excluded</td></tr><tr><td>ICEWS14</td><td>Sequence</td><td>0.65%</td><td>0.81%</td><td>Included</td></tr><tr><td>ICEWS14</td><td>Time</td><td>0.79%</td><td>1.31%</td><td>Included</td></tr><tr><td>ICEWS14</td><td>Direction</td><td>0.85%</td><td>1.21%</td><td>Included</td></tr><tr><td>GDELT</td><td>Partner</td><td>3.16%</td><td>3.16%</td><td>Excluded</td></tr><tr><td>GDELT</td><td>Sequence</td><td>1.32%</td><td>2.20%</td><td>Included</td></tr><tr><td>GDELT</td><td>Time</td><td>2.05%</td><td>3.41%</td><td>Excluded</td></tr><tr><td>GDELT</td><td>Direction</td><td>0.61%</td><td>1.05%</td><td>Included</td></tr><tr><td>ICEWS05-15</td><td>Partner</td><td>11.15%</td><td>11.15%</td><td>Excluded</td></tr><tr><td>ICEWS05-15</td><td>Sequence</td><td>0.83%</td><td>1.31%</td><td>Included</td></tr><tr><td>ICEWS05-15</td><td>Time</td><td>0.50%</td><td>0.72%</td><td>Included</td></tr><tr><td>ICEWS05-15</td><td>Direction</td><td>0.83%</td><td>1.25%</td><td>Included</td></tr></table>

TABLE S14  
Total variation (TV) between pair and transition count distributions before and after each controlled reorganization.
<table><tr><td>Source</td><td>Variant</td><td>Pair TV</td><td>Transition TV</td></tr><tr><td>Wikipedia</td><td>Partner</td><td>.971</td><td>.998</td></tr><tr><td>Wikipedia</td><td>Sequence</td><td>.000</td><td>.161</td></tr><tr><td>Wikipedia</td><td>Time</td><td>.000</td><td>.165</td></tr><tr><td>Reddit</td><td>Partner</td><td>.955</td><td>.997</td></tr><tr><td>Reddit</td><td>Sequence</td><td>.000</td><td>.249</td></tr><tr><td>Reddit</td><td>Time</td><td>.000</td><td>.254</td></tr><tr><td>MOOC</td><td>Partner</td><td>.493</td><td>.923</td></tr><tr><td>MOOC</td><td>Sequence</td><td>.000</td><td>.759</td></tr><tr><td>MOOC</td><td>Time</td><td>.000</td><td>.757</td></tr><tr><td>ICEWS14</td><td>Partner</td><td>.820</td><td>.956</td></tr><tr><td>ICEWS14</td><td>Sequence</td><td>.000</td><td>.472</td></tr><tr><td>ICEWS14</td><td>Time</td><td>.000</td><td>.465</td></tr><tr><td>ICEWS14</td><td>Direction</td><td>.737</td><td>.788</td></tr><tr><td>GDELT</td><td>Partner</td><td>.796</td><td>.967</td></tr><tr><td>GDELT</td><td>Sequence</td><td>.000</td><td>.698</td></tr><tr><td>GDELT</td><td>Time</td><td>.000</td><td>.689</td></tr><tr><td>GDELT</td><td>Direction</td><td>.795</td><td>.933</td></tr><tr><td>ICEWS05-15</td><td>Partner</td><td>.782</td><td>.941</td></tr><tr><td>ICEWS05-15</td><td>Sequence</td><td>.000</td><td>.459</td></tr><tr><td>ICEWS05-15</td><td>Time</td><td>.000</td><td>.455</td></tr><tr><td>ICEWS05-15</td><td>Direction</td><td>.687</td><td>.739</td></tr></table>

TABLE S15  
Target results for the 15 conditions that satisfy the source-competence criterion. B is original minus controlled-variant target score; values are means ± sample standard deviations over three seeds.
<table><tr><td>Source → Target</td><td>Variant</td><td>Metric</td><td>Original</td><td>Reorganized</td><td></td><td>B</td><td colspan="3">Seed signs</td></tr><tr><td>Wikipedia → Reddit</td><td>Sequence</td><td>AP</td><td> $0 . 9 6 4 3 \pm 0 . 0 0 2 8$ </td><td> $0 . 9 6 4 5 \pm 0 . 0 0 2 6$ </td><td></td><td> $- 0 . 0 0 0 2 \pm 0 . 0 0 1 4$ </td><td></td><td> $+ ~ + ~ -$ </td><td></td></tr><tr><td>Wikipedia → Reddit</td><td>Time</td><td>AP</td><td> $0 . 9 6 4 3 \pm 0 . 0 0 2 8$ </td><td></td><td> $0 . 9 6 4 6 \pm 0 . 0 0 3 0$ </td><td> $- 0 . 0 0 0 4 \pm 0 . 0 0 0 2$ </td><td></td><td> $---$ </td><td></td></tr><tr><td>Reddit → Wikipedia</td><td>Sequence</td><td>AP</td><td> $0 . 9 2 8 6 \pm 0 . 0 0 7 2$ </td><td></td><td> $0 . 9 2 6 2 \pm 0 . 0 0 9 7$ </td><td> $+ 0 . 0 0 2 4 \pm 0 . 0 0 6 7$ </td><td></td><td> $\mathrm { ~ -- ~ } + \mathrm { ~ + ~ }$ </td><td></td></tr><tr><td>Reddit → Wikipedia</td><td>Time</td><td>AP</td><td> $0 . 9 2 8 6 \pm 0 . 0 0 7 2$ </td><td></td><td> $0 . 9 3 7 1 \pm 0 . 0 1 0 5$ </td><td> $- 0 . 0 0 8 6 \pm 0 . 0 1 0 8$ </td><td></td><td> $---$ </td><td></td></tr><tr><td> $\mathbf { M O O C }  \mathbf { R e d } \dot { \mathbf { d i t } }$ </td><td>Partner</td><td>AP</td><td> $0 . 8 8 8 0 \pm 0 . 0 0 4 8$ </td><td></td><td> $0 . 8 9 2 7 \pm 0 . 0 0 3 6$ </td><td> $- 0 . 0 0 4 7 \pm 0 . 0 0 6 6$ </td><td></td><td> $\mathit { \Pi } - \mathit { \Pi } + \mathit { \Pi } - \mathit { \Pi }$ </td><td></td></tr><tr><td> $\mathrm { M O O C }  \mathrm { R e d d i t }$ </td><td>Sequence</td><td>AP</td><td> $0 . 8 8 8 0 \pm 0 . 0 0 4 8$ </td><td></td><td> $0 . 8 9 1 4 \pm 0 . 0 0 5 8$ </td><td> $- 0 . 0 0 3 4 \pm 0 . 0 0 8 0$ </td><td></td><td> $\mathit { \Pi } - \mathit { \Pi } + \mathit { \Pi } - \mathit { \Pi }$ </td><td></td></tr><tr><td> $\mathrm { M O O C }  \mathrm { R e d d i t }$ </td><td>Time</td><td>AP</td><td> $0 . 8 8 8 0 \pm 0 . 0 0 4 8$ </td><td></td><td> $0 . 8 9 6 1 \pm 0 . 0 0 3 1$ </td><td> $- 0 . 0 0 8 1 \pm 0 . 0 0 6 8$ </td><td></td><td></td><td></td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { I C E W S 0 5 } { - } 1 5$ </td><td>Sequence</td><td>MRR</td><td> $0 . 2 4 5 5 \pm 0 . 0 3 8 4$ </td><td></td><td> $0 . 2 1 2 5 \pm 0 . 0 0 5 2$ </td><td> $+ 0 . 0 3 3 0 \pm 0 . 0 3 8 9$ </td><td></td><td> $+ ~ + ~ +$ </td><td></td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { I C E W S 0 5 } { - } 1 5$ </td><td>Time</td><td>MRR</td><td> $0 . 2 4 5 5 \pm 0 . 0 3 8 4$ </td><td></td><td> $0 . 2 2 3 8 \pm 0 . 0 2 6 7$ </td><td> $+ 0 . 0 2 1 8 \pm 0 . 0 3 2 1$ </td><td></td><td> $\mathrm { ~ -- ~ } + \mathrm { ~ + ~ }$ </td><td></td></tr><tr><td> $\mathrm { I C E W S 1 4 } \to \mathrm { I C E W S 0 5 } { - } 1 5$ </td><td>Direction</td><td>MRR</td><td> $0 . 2 4 5 5 \pm 0 . 0 3 8 4$ </td><td></td><td> $0 . 2 2 7 1 \pm 0 . 0 1 6 2$ </td><td> $+ 0 . 0 1 8 4 \pm 0 . 0 3 8 5$ </td><td></td><td> $+ ~ - ~ +$ </td><td></td></tr><tr><td> $\mathrm { { G D E L T } } \to \mathrm { { W i k i p e d i a } }$ </td><td>Sequence</td><td>AP</td><td> $0 . 9 5 5 3 \pm 0 . 0 0 8 0$ </td><td></td><td> $0 . 9 4 3 0 \pm 0 . 0 2 1 3$ </td><td> $+ 0 . 0 1 2 3 \pm 0 . 0 2 6 2$ </td><td></td><td> $\mathrm { ~ -- ~ } +$ </td><td></td></tr><tr><td> $\mathrm { { G D E L T } } \to { \mathrm { W i k i } } { \dot { \ p } } { \mathrm { e d i a } }$ </td><td>Direction</td><td>AP</td><td> $0 . 9 5 5 3 \pm 0 . 0 0 8 0$ </td><td></td><td> $0 . 9 5 3 6 \pm 0 . 0 0 7 1$ </td><td> $+ 0 . 0 0 1 7 \pm 0 . 0 0 2 1$ </td><td></td><td> $+ ~ - ~ +$ </td><td></td></tr><tr><td> $\mathrm { I C E W S 0 5 - 1 5 \to \hat { W i k i p e d i a } }$ </td><td>Sequence</td><td>AP</td><td> $0 . 9 6 7 5 \pm 0 . 0 0 3 1$ </td><td></td><td> $0 . 9 7 1 6 \pm 0 . 0 0 1 5$ </td><td> $- 0 . 0 0 4 1 \pm 0 . 0 0 1 7$ </td><td></td><td></td><td></td></tr><tr><td> $\mathrm { I C E W S 0 5 - 1 5 } \to \mathrm { W i k i } \mathrm { \bar { p } e d i a }$ </td><td>Time</td><td>AP</td><td> $0 . 9 6 7 5 \pm 0 . 0 0 3 1$ </td><td></td><td> $0 . 9 7 2 3 \pm 0 . 0 0 2 2$ </td><td> $- 0 . 0 0 4 8 \pm 0 . 0 0 2 3$ </td><td></td><td></td><td></td></tr><tr><td> $\mathrm { I C E W S 0 5 - 1 5 } \to \mathrm { W i k i \bar { p } e d i a }$ </td><td>Direction</td><td>AP</td><td> $0 . 9 6 7 5 \pm 0 . 0 0 3 1$ </td><td></td><td> $0 . 9 7 1 3 \pm 0 . 0 0 7 5$ </td><td> $- 0 . 0 0 3 8 \pm 0 . 0 0 4 4$ </td><td></td><td></td><td></td></tr></table>

Public dataset sources used in the experiments.  
TABLE S16
<table><tr><td>Format</td><td>Dataset</td><td>Data source</td></tr><tr><td>CTDG</td><td>Wikipedia</td><td>https://snap.stanford.edu/jodie/wikipedia.csv</td></tr><tr><td>CTDG</td><td>Reddit</td><td>https://snap.stanford.edu/jodie/reddit.csv</td></tr><tr><td>CTDG</td><td>MOOC</td><td>https://snap.stanford.edu/jodie/mooc.csv</td></tr><tr><td>CTDG</td><td>LastFM</td><td>https://snap.stanford.edu/jodie/lastfm.csv</td></tr><tr><td>TKG</td><td>ICEWS14</td><td>TLogic: https://github.com/liu-yushan/TLogic (data/icews14)</td></tr><tr><td>TKG</td><td>ICEWS05-15</td><td>TLogic: https://github.com/liu-yushan/TLogic (data/icews0515)</td></tr><tr><td>TKG</td><td>GDELT</td><td>CyGNet: https://github.com/CunchaoZ/CyGNet (data/GDELT)</td></tr></table>

REFERENCES

[1] S. M. Kazemi, R. Goel, K. Jain, I. Kobyzev, A. Sethi, P. Forsyth, and P. Poupart, “Representation learning for dynamic graphs: A survey,” Journal of Machine Learning Research, vol. 21, no. 70, pp. 1–73, 2020.

[2] Y. Ma, Z. Wang, W. Sun, and Y. Ye, “Temporal graph pattern machine,” 2026, version 1.

[3] H. Yuan, Q. Sun, J. Shi, X. Fu, J. Li, and P. S. Yu, “Decoupled and divergence-conditioned prompt for multi-domain dynamic graph foundation models,” 2026.

[4] J. Pan, M. Nayyeri, O. Mohammed, D. Hernandez, R. Zhang, C. Cheng, and S. Staab, “Towards foundation model on temporal knowledge graph reasoning,” 2025.

[5] ——, “FITTER: Vocabulary-agnostic cross-domain inference on temporal knowledge graphs,” 2026.

[6] F. Poursafaei, S. Huang, K. Pelrine, and R. Rabbany, “Towards better evaluation for dynamic link prediction,” in Advances in Neural Information Processing Systems, vol. 35. Curran Associates, Inc., 2022, pp. 32 928–32 941.

[7] J. Gastinger, C. Meilicke, F. Errica, T. Sztyler, A. Schulke, and ¨ H. Stuckenschmidt, “History repeats itself: A baseline for temporal knowledge graph forecasting,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 2024, pp. 4016– 4024.

[8] C. Liang, Y. Huang, D. He, P. Li, Y. Li, D. Jin, and W. Zhang, “Unified multi-domain graph pre-training for homogeneous and heterogeneous graphs via domain-specific expert encoding,” 2026, version 1.

[9] S. Huang, F. Poursafaei, R. Rabbany, G. Rabusseau, and E. Rossi, “UTG: Towards a unified view of snapshot and event based models for temporal graphs,” in Proceedings of the Third Learning on Graphs Conference, ser. Proceedings of Machine Learning Research, G. Wolf and S. Krishnaswamy, Eds., vol. 269. PMLR, 2025, pp. 28:1–28:16.

[10] Z. Xie, Z. Zhao, J. Yu, G. Zhou, C. Luo, and J. X. Huang, “One model connects all graphs: Towards training one unified model for multi-domain graph pre-training using adaptive vector quantization,” Information Fusion, vol. 129, p. 103998, 2026.

[11] W. Jeong, Y.-J. Kim, H.-Y. Koo, J. Seo, J. Choi, and B. Oh, “Interaction-grounded semantic graph refinement for llm-based recommendation,” IEEE Access, vol. 13, pp. 194 229–194 244, 2025.

[12] M. Liu, Y. Liu, Q. Ren, and M. Han, “Rethinking multi-level information fusion in temporal graphs: Pre-training then distilling for better embedding,” Information Fusion, vol. 121, p. 103127, 2025.

[13] M. Yu and Y.-g. Ha, “FLIP: A foundation model for CTDG link prediction using factorized edge-event intensity,” Information Sciences, vol. 757, p. 123955, 2026.

[14] Z. Huai, D. Zhang, G. Yang, and J. Tao, “Spatial-temporal knowledge graph network for event prediction,” Neurocomputing, vol. 553, p. 126557, 2023.

[15] Y. Ma, L. Chai, X. Zhang, and Y. Wang, “Independent-interactive knowledge learning for multi-source domain adaptive object detection,” Information Fusion, vol. 126, p. 103642, 2026.

[16] L. Yu, L. Sun, B. Du, and W. Lv, “Towards better dynamic graph learning: New architecture and unified library,” in Advances in Neural Information Processing Systems, vol. 36. Curran Associates, Inc., 2023, pp. 67 686–67 700.

[17] Microsoft, “Threat intelligence in Microsoft Sentinel,” Microsoft Learn, 2026, updated 14 May 2026; accessed 6 October 2026.

[18] The DFIR Report, “The DFIR Report: Public intrusion reports,” https://thedfirreport.com/, accessed 6 October 2026.

[19] M. R. Rahman, B. Wroblewski, Q. Matthews, B. Morgan, T. Menzies, and L. Williams, “Mining temporal attack patterns from cyberthreat intelligence reports,” Knowledge and Information Systems, vol. 67, no. 10, pp. 8941–8981, Oct. 2025.