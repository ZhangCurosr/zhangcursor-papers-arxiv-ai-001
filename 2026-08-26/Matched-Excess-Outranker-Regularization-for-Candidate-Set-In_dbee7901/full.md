# Matched Excess-Outranker Regularization for Candidate-Set Interference in Continual Knowledge Graph Embedding

Hao Ren University of New South Wales Sydney, NSW, Australia hao.ren@unsw.edu.au

Junbin Gao The University of Sydney Sydney, NSW, Australia junbin.gao@sydney.edu.au

Jiaojiao Jiang University of New South Wales Sydney, NSW, Australia jiaojiao.jiang@unsw.edu.au

## Ab<sub>s</sub>t<sub>rac</sub>t

Continual knowledge graph embedding updates entity and relation representations as a graph grows. Existing methods primarily address catastrophic forgetting, but entity admission also changes the candidate universe of every compatible query. A historical answer can therefore lose rank even when its score and its ordering among old entities are preserved. We formalize this efect as candidate-set interference and introduce Matched Excess-Outranker Regularization (Meor), a host-level objective that compares smooth answerrelative newcomer pressure with score-blind, structurally matched old references. Its one-sided penalty acts only when newcomer competition exceeds the matched reference, preserving the host learner’s signal for legitimate new entities. Across eight paired runs on ENTITY–ComplEx, Meor improves historical current-universe mean reciprocal rank (MRR) by 0.0057 over replay and reduces candidate-set interference by 0.0055, with one-sided 95% lower bounds of 0.0052 and 0.0051, respectively. It satisfies the preservation criteria for old-universe ranking and newcomer acquisition and improves historical current-universe MRR over persistent calibration, matched maximum regularizer (MMR), and unmatched old regularizer (UOR). Direct ablations support each component of its reference construction and aggregation. Adding Meor also improves historical ranking in all ten reported FBInc-S and FBInc-L host and backbone settings, with every paired 95% confidence interval excluding zero. These results establish candidate admission as a distinct source of continual rank loss and show that it can be controlled without replacing the underlying embedding architecture or continual learner.

## CCS Conce<sub>p</sub>ts

• Computing methodologies → Knowledge representation and reasoning; • Information systems → Information integration; Novelty in information retrieval; Graph-based database models.

## Ke<sub>y</sub>words

Continual Knowledge Graph Embedding, Evolving Knowledge Graphs, Candidate-Set Interference, Ranking Regularization

ACM Reference Format: Hao Ren, Junbin Gao, and Jiaojiao Jiang. XXXX. Matched Excess-Outranker Regularization for Candidate-Set Interference in Continual Knowledge Graph Embedding. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 11 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Knowledge graphs (KGs) in deployed systems expand as new entities and facts become available, while link prediction must remain reliable for knowledge already represented [2, 21]. Continual knowledge graph embedding (KGE) supports this process by updating a model without retraining it from the beginning. Existing continual KGE research has focused primarily on catastrophic forgetting, in which learning from new facts changes representations acquired earlier. This account is incomplete when graph growth expands the entity set. Every admitted entity joins the candidate universe of each compatible query and can displace an established answer even when the answer score and its ordering relative to every previously known entity remain unchanged [17]. The research problem addressed in this paper is how to control this admission-induced historical rank loss without materially weakening the model’s ability to learn newly admitted entities when they are correct answers. We address this problem with Matched Excess-Outranker Regularization (Meor), a host-level objective that suppresses only the newcomer pressure exceeding a score-blind, structurally matched old-candidate reference.

Figure 1 illustrates the cumulative entity admission that creates this additional competition. We study the resulting failure mode as candidate-set interference. Unlike catastrophic forgetting, it does not require a change in model parameters or in the relative ordering of old entities. It can be isolated at a fixed checkpoint by evaluating the same historical query over two candidate universes. The old universe measures ranking among previously known entities, whereas the current universe additionally contains the entities admitted through graph growth. Their diference identifies the rank loss attributable to newcomer competition. Current-universe evaluation exposes this efect, but it does not provide a training rule for controlling it. Replay, distillation, and parameter regularization protect previously learned representations, yet they do not directly regulate the query-specific competition created when new entities enter the candidate universe [3, 4, 9, 13]. Recent work identifies entity interference as an evaluation concern [17]. Our central contribution is to convert this diagnostic distinction into an explicit objective for continual learning.

Meor operates on historical replay queries during continual refinement [19]. For each query, it measures the positive score gaps between newly admitted candidates and the recorded answer, normalizes these gaps to the local score scale, and smoothly ag gregates the above-answer tail. It then compares this newcomer pressure with pressure from old candidates matched by prediction role, degree, and incident relation-role structure. The matching procedure does not use model scores, so the reference cannot adapt to the outcome against which it is evaluated. A one-sided squared hinge penalizes only the pressure exceeding the matched refer ence. Newcomers whose aggregate pressure remains consistent with structurally comparable old candidates receive no penalty, while the original host objective continues to govern the learning of new facts. Meor therefore targets the harmful component of admission-induced competition without replacing the embedding architecture or continual update rule of the host.

![](images/dd07368bff26786dc7adb0d56da57bea08cb46cfe1b53745d410ecdc7d0f967e.jpg)  
Fi<sub>gure</sub> 1<sub>:</sub> C<sub>umu</sub>l<sub>a</sub>ti<sub>ve evo</sub>l<sub>u</sub>ti<sub>on o</sub>f <sub>a con</sub>ti<sub>nua</sub>l k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> <sub>g</sub>ra<sub>p</sub><sup>h</sup> across t<sup>h</sup>ree sna<sub>p</sub>s<sup>h</sup>ots.

The experiments support both the ranking benefit and the proposed mechanism. Across eight paired runs on ENTITY–ComplEx, Meor improves historical current-universe mean reciprocal rank (MRR) by 0.0057 over replay, with a one-sided 95% lower bound of 0.0052, and reduces candidate-set interference by 0.0055, with a lower bound of 0.0051. Relative to replay, the mean changes are 0.0002 in old-universe MRR and 0.0015 in target-newcomer MRR, and both lower bounds satisfy the preservation rule. Meor also improves historical ranking over persistent calibration, Matched Maximum Regularizer (MMR), and Unmatched Old Regularizer (UOR). Direct ablations show positive contributions from smooth tail aggregation, structural matching, query-specific assignment, and old-reference centering. On FBInc-S and FBInc-L, adding Meor improves historical current-universe MRR in all ten reported combi nations of stream, backbone, and continual host. Every paired 95% confidence interval excludes zero, and every newcomer-acquisition bound remains within the preservation margin.

This work makes three contributions. First, we formalize candidateset interference as a continual KGE optimization problem distinct from catastrophic forgetting and provide a same-checkpoint rank decomposition that separates admission-induced competition from changes in the ordering of old entities. Second, we introduce Meor, a diferentiable intervention that combines answer-relative smooth tail aggregation, score-blind structural matching, and one-sided excess regularization while preserving the host learning objective. Third, we provide a paired, mechanism-focused empirical analysis that connects historical-ranking gains to reduced candidate-set interference, isolates the contribution of each design component, and demonstrates transfer across the reported graph streams, embedding backbones, and continual-learning hosts.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2<sub>.</sub>1 St<sub>a</sub>ti<sub>c</sub> KGE <sub>a</sub>nd R<sub>a</sub>nk<sub>e</sub>d E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n

Knowledge graph embedding represents entities and relations in a continuous space and assigns a plausibility score to each candidate triple. Translation-based models such as TransE interpret a relation as a displacement between entity embeddings [1]. Bilinear models score triples through multiplicative interactions, and ComplEx extends this family to complex-valued embeddings, allowing a single scoring form to represent both symmetric and antisymmetric relations [25]. More recent probabilistic approaches, such as Normalizing Flow Embedding, replace deterministic point embeddings with distributional representations and score triples through similarities between learned flows, providing a richer treatment of representation uncertainty [26]. These models difer in geometry and scoring function, but they share the same static link-prediction interface: given a query formed by masking one endpoint of a triple, the model scores candidate entities and ranks them by plausibility.

Although KGE models difer in representation geometry, they are commonly assessed through link prediction, where learned triple scores are converted into ranked candidate lists. A link-prediction query masks the head or tail entity of a known fact, scores the eligible replacement entities, and ranks its recorded answer among them. Filtered evaluation removes other known correct answers from the candidate list so that valid alternatives are not counted as errors [20]. The resulting rank therefore depends not only on the learned score function, but also on the filtering rule and the candidate universe. In static benchmarks, this universe is usually fixed. In a growing knowledge graph, however, newly admitted entities become additional competitors for historical queries. Candidate construction therefore becomes part of the continual learning problem, rather than a neutral evaluation detail.

## 2.2 Tem<sub>p</sub>oral and Continual KGE

Temporal KGE and continual KGE address diferent forms of change. Temporal KGE treats time as part of the fact being modeled: facts are associated with timestamps or validity intervals, and these temporal annotations directly condition the learned representations or scoring function [28]. HyTE represents each timestamp as a hyperplane on which entities and relations are projected [5], while diachronic embedding makes entity representations explicit functions of time [7]. In these models, time is semantic information used at prediction time.

Continual KGE addresses a diferent setting: the model is updated as facts, entities, or relations are admitted across successive graph states. The update order determines what information is available at each stage, but time need not be a semantic argument of the score function. A graph state may reflect data acquisition, benchmark construction, or incremental deployment rather than a factual validity interval. Thus, temporal KGE asks which fact holds at a specified time, whereas continual KGE asks how an existing representation can be preserved and extended as the graph [18].

Continual KGE treats graph evolution as an incremental representation learning problem and adapts continual-learning strategies to update knowledge graph embeddings without relearning all concepts from scratch [4]. Representative methods use selected historical data and disentangled components [4, 9], masked reconstruction, embedding transfer, and distillation [3, 13], or eficient adaptation, dynamic regularization, coordinated masks, and taskconditioned transfer [14, 23, 29, 30]. Despite their architectural diferences, these methods share the goal of retaining useful knowledge while incorporating each graph update.

## 2<sub>.</sub>3 C<sub>a</sub>t<sub>as</sub>tr<sub>op</sub>hi<sub>c</sub> F<sub>o</sub>r<sub>ge</sub>ttin<sub>g</sub> in C<sub>o</sub>ntin<sub>ua</sub>l KGE

The central concern in continual KGE is catastrophic forgetting [15]. Training on newly available facts can move established representations and reduce performance on historical knowledge, creating a tension between stability and plasticity. The model must preserve useful structure from earlier graph states while remaining responsive to new entities, relations, and facts.

Existing methods address this problem through complementary strategies. Replay retains selected historical observations, regularization limits destructive parameter change, and distillation transfers information from an earlier model state [19]. Selective and modular updates restrict which representations are modified, while reconstruction objectives preserve information encoded in the earlier graph [3, 4, 9, 13]. FMR combines rotated representations with dynamic regularization [29], CMKGE coordinates stability and plasticity through dual masks [23], and Dynamic Temperature Distillation adjusts knowledge transfer according to fact volatility [22]. Other work improves continual updates through informed initialization [16], adaptive representation capacity [12], structure-aware representation learning [10], and Bayesian-guided evolution [11].

These methods primarily control how learned representations change between graph states. Their evaluation therefore emphasizes whether knowledge encoded before an update remains recoverable afterward. This perspective captures parameter and representation forgetting, but it does not fully describe the ranking consequences of enlarging the entity universe.

## 2<sub>.</sub>4 C<sub>a</sub>ndid<sub>a</sub>t<sub>e</sub>-S<sub>e</sub>t Int<sub>e</sub>rf<sub>e</sub>r<sub>e</sub>n<sub>ce</sub>

Candidate-set interference is a distinct source of historical rank loss. When new entities enter the graph, they become eligible answers to earlier queries. A historical answer can therefore move down in the ranking even when its score and its ordering relative to every old entity remain unchanged. The loss arises because the query faces additional competitors, not because the model has forgotten how to rank established entities.

Pons et al. [17] identify this efect as entity interference and show that evaluation restricted to old candidates overstates continual performance. A same-checkpoint comparison makes the distinction precise. Evaluating the same historical query first over the old candidate universe and then over the expanded current universe isolates the rank loss attributable to admitted candidates. Catastrophic forgetting instead concerns changes caused by model updating. Candidate-set interference can occur without forgetting, and forgetting can occur while the candidate universe remains fixed. The two mechanisms are complementary explanations for historical degradation.

Current-universe evaluation exposes candidate-set interference, but the continual objectives reviewed above do not directly regulate the ranking pressure that admitted entities create for each historical answer. This intervention gap defines the problem addressed here. We treat candidate competition as an explicit learning target within a fixed continual KGE process. The next section formalizes the resulting rank decomposition and evaluation objectives.

## 3 C<sub>o</sub>ntin<sub>ua</sub>l KGE Pr<sub>o</sub>bl<sub>e</sub>m F<sub>o</sub>rm<sub>u</sub>l<sub>a</sub>ti<sub>o</sub>n

## 3.1 Gra<sub>p</sub>h Gro<sub>w</sub>th and Candidate Admission

Let $\mathcal { G } _ { 0 } , . . . , \mathcal { G } _ { T }$ denote a sequence of cumulative knowledge-graph snapshots. At snapshot $u ,$ , the updated graph is represented as $\mathcal { G } _ { u } =$ $( \mathcal { E } _ { u } , \mathcal { R } _ { u } , \mathcal { F } _ { u } )$ , where $\mathcal { E } _ { u }$ and $\mathcal { R } _ { u }$ denote the sets of entities and relations observed up to snapshot �, respectively, and $\mathcal { F } _ { u } \subseteq \mathcal { E } _ { u } { \times } \mathcal { R } _ { u } { \times } \mathcal { E } _ { u }$ denotes the corresponding set of observed training triples. The continual setting assumes

$$
\mathcal { E } _ { u - 1 } \subseteq \mathcal { E } _ { u } , \qquad \mathcal { R } _ { u - 1 } \subseteq \mathcal { R } _ { u } , \qquad \mathcal { F } _ { u - 1 } \subseteq \mathcal { F } _ { u } .\tag{1}
$$

The entities and relations admitted at update � are

$$
\Delta \mathcal { E } _ { u } = \mathcal { E } _ { u } \ \backslash \ \mathcal { E } _ { u - 1 } , \qquad \Delta \mathcal { R } _ { u } = \mathcal { R } _ { u } \ \backslash \ \mathcal { R } _ { u - 1 } .\tag{2}
$$

We write $\mathcal { U } \subseteq \{ 1 , \dotsc , T \}$ for the evaluated updates. For a fact $( h , r , t )$ , tail prediction uses the query $q \ = \ ( h , r , \cdot )$ with answer $a = t ,$ whereas head prediction uses $q = \left( \cdot , r , t \right)$ with answer $a = h$ We therefore let $\mathcal { D } \subseteq \{ \mathrm { h e a d } , \mathrm { t a i l } \}$ denote the evaluated prediction modes and use $d \in \mathcal { D }$ to identify the missing endpoint. The notation $r ( q )$ refers to the relation in �, and � denotes a candidate entity. The experimental choices of U and D appear in Section 5.

At update �, a KGE model with parameters $\theta _ { m , k , u }$ assigns score $s _ { m , k , u } ( q , e )$ to candidate � for query �. The indices � and � identify the learning method and training realization. Higher scores indicate stronger preference. The formulation is independent of the embedding architecture.

## 3<sub>.</sub>2 Hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l R<sub>an</sub>k L<sub>oss</sub> U<sub>n</sub>d<sub>er</sub> C<sub>an</sub>did<sub>a</sub>t<sub>e</sub> G<sub>row</sub>th

Consider a historical query occurrence $x = ( q , a , u , d )$ whose source fact appeared before update �. Its answer � belongs to $\mathcal { E } _ { u - 1 }$ . Using the same model state, let $C _ { x } ^ { \mathrm { o l d } }$ and $C _ { x } ^ { \mathrm { c u r } }$ be the filtered candidate sets drawn from $\mathcal { E } _ { u - 1 }$ and $\mathcal { E } _ { u }$ . The same filtering rule is applied to both sets, and the answer is retained. Consequently, $C _ { x } ^ { \mathrm { o l } \bar { \mathrm { d } } } \subseteq C _ { x } ^ { \mathrm { c u r } }$ We write $\Delta C _ { x } = C _ { x } ^ { \mathrm { c u r } } \ \backslash \ C _ { x } ^ { \mathrm { o l d } }$ for the candidates introduced by the expansion.

The score function together with a fixed tie rule induces a total order. We write $e \succ _ { m , k , x }$ � when candidate � precedes answer � under model state $( m , k , u )$ . For $v \in \{ \mathrm { o l d , c u r } \}$ , the corresponding rank is

$$
r _ { m , k , u } ^ { v } ( x ) = 1 + \sum _ { e \in C _ { x } ^ { v } \backslash \{ a \} } \mathbb { I } [ e \succ _ { m , k , x } a ] .\tag{3}
$$

The two ranks satisfy

$$
r _ { m , k , u } ^ { \mathrm { c u r } } ( x ) = r _ { m , k , u } ^ { \mathrm { o l d } } ( x ) + \sum _ { e \in \Delta C _ { x } } \mathbb { I } [ e \succ _ { m , k , x } a ] .\tag{4}
$$

This identity isolates the rank loss attributable to newly admitted candidates. It holds even when the ordering among old entities is unchanged and makes no assumption about whether their representations also change. Evaluation in the current candidate universe therefore reveals a source of interference that evaluation over old candidates alone cannot measure [17].

## 3.3 Query Roles and Evaluation Objectives

We distinguish three query roles. A historical occurrence comes from a fact observed before the current update. A target newcomer occurrence has an answer in $\Delta \mathcal { E } _ { u }$ . A query newcomer occurrence has a known query entity in $\Delta \mathcal { E } _ { u }$ and an answer in $\mathcal { E } _ { u - 1 }$ . The latter roles capture diferent learning requirements and are therefore evaluated separately.

For data split �, let $C ^ { \sigma , \mathrm { H } } \subseteq \mathcal { U } \times \mathcal { D }$ be the predefined historical evaluation cells. Let $\mathcal { Q } _ { u , d } ^ { \sigma , \mathrm { H } }$ contain the historical occurrences in cell $( u , d )$ . Historical MRR in the current candidate universe is

$$
H _ { \mathrm { c u r } , m , k } ^ { \sigma } = \frac { 1 } { | C ^ { \sigma , \mathrm { H } } | } \sum _ { ( u , d ) \in C ^ { \sigma , \mathrm { H } } } \frac { 1 } { | Q _ { u , d } ^ { \sigma , \mathrm { H } } | } \sum _ { x \in Q _ { u , d } ^ { \sigma , \mathrm { H } } } \frac { 1 } { r _ { m , k , u } ^ { \mathrm { c u r } } ( x ) } .\tag{5}
$$

Replacing current ranks with old candidate ranks defines $H _ { \mathrm { o l d } , m , k } ^ { \sigma } .$ Their diference is

$$
D _ { \mathrm { M C I } , m , k } ^ { \sigma } = H _ { \mathrm { o l d } , m , k } ^ { \sigma } - H _ { \mathrm { c u r } , m , k } ^ { \sigma } .\tag{6}
$$

This quantity measures the loss associated with candidate expansion. A small value is not suficient evidence of useful ranking because both terms can decrease together. We therefore treat currentuniverse historical MRR as the primary utility outcome and olduniverse historical MRR as a preservation outcome.

Let $C ^ { \sigma , \mathrm { T N } } \subseteq \mathcal { U } \times \mathcal { D }$ be the predefined target newcomer cells, and let $\boldsymbol { Q } _ { u , d } ^ { \sigma , \mathrm { T N } }$ be their query sets. Target newcomer acquisition is measured by

$$
A _ { \mathrm { T N } , m , k } ^ { \sigma } = \frac { 1 } { | C ^ { \sigma , \mathrm { T N } } | } \sum _ { ( u , d ) \in C ^ { \sigma , \mathrm { T N } } } \frac { 1 } { | Q _ { u , d } ^ { \sigma , \mathrm { T N } } | } \sum _ { x \in Q _ { u , d } ^ { \sigma , \mathrm { T N } } } \frac { 1 } { r _ { m , k , u } ^ { \mathrm { c u r } } ( x ) } .\tag{7}
$$

Each included cell receives equal weight, and queries receive equal weight within a cell. An endpoint is defined only when every predefined query set is nonempty. An empty set is never assigned zero. Query newcomer performance is constructed in the same manner. When the split, method, and training realization are clear, we abbreviate the four endpoints as $H _ { \mathrm { c u r } } , H _ { \mathrm { o l d } } , D _ { \mathrm { M C I } }$ , and $A _ { \mathrm { T N } }$

## 4 Matched Excess-Outranker Re<sub>g</sub>ularization

Equation (4) motivates an answer-relative training signal. Newly admitted entities matter when they move above a historical answer. A raw outranker count is discontinuous, while an unnormalized score gap is not comparable across queries. Penalizing newcom ers against zero or an arbitrary old sample also confounds cohort identity with visible structural diferences.

Meor combines a smooth, normalized measure of newcomer competition relative to the answer with an old reference matched on observable structure. It penalizes only the portion of newcomer competition that exceeds this reference. Consequently, a large newcomer aggregate incurs no penalty when a structurally comparable old cohort produces a similar value.

Figure 2 illustrates how candidate-set interference arises and how Meor addresses it.

## 4<sub>.</sub>1 A<sub>nswer-</sub>R<sub>e</sub>l<sub>a</sub>ti<sub>ve</sub> N<sub>ewcomer</sub> C<sub>ompe</sub>titi<sub>on</sub>

The regularizer operates on historical occurrences drawn from replay. Throughout this section, we fix a method � and training realization $k ;$ every score-derived quantity below inherits these

indices. For an occurrence $x = ( q , a , u , d )$ , let $P _ { u } ^ { \mathrm { t r } } ( q , d )$ contain the answers visible in cumulative training facts through update �. The eligible newcomer cohort is

$$
N _ { x } = \operatorname { s o r t _ { i d } } \left( \Delta \mathcal { E } _ { u } \ \backslash \ \left( \{ a \} \cup P _ { u } ^ { \mathrm { t r } } ( q , d ) \right) \right) .\tag{8}
$$

The complete eligible old pool is

$$
{ \mathcal E } _ { u - 1 } ^ { - , \mathrm { t r } } ( \boldsymbol { x } ) = \mathrm { s o r t } _ { \mathrm { i d } } \big ( { \mathcal E } _ { u - 1 } \ \backslash \ \big ( \{ a \} \cup P _ { u } ^ { \mathrm { t r } } ( q , d ) \big ) \big ) \ .\tag{9}
$$

Both sets are defined only from the training prefix. Validation and test labels therefore cannot afect cohort membership. The answer is excluded explicitly, even when it is absent from $P _ { u } ^ { \mathrm { t r } } ( q , d )$

We draw a scale sample $C _ { x } ^ { \mathrm { s c a l e } }$ from the eligible old pool without using model scores, with

$$
| C _ { x } ^ { \mathrm { s c a l e } } | = \operatorname* { m i n } \left\{ K _ { \mathrm { s c a l e } } , | \mathcal { E } _ { u - 1 } ^ { - , \mathrm { t r } } ( x ) | \right\} .\tag{10}
$$

Its identifiers are selected independently of model scores and shared by paired methods. This prevents the observed scores from influencing normalization.

Let IQR be the interquartile range of the detached finite scores in $C _ { x } ^ { \mathrm { s c a l e } } [ 6 ]$ . Independently, let $\mathrm { M A D } _ { x } ^ { \mathrm { f b } }$ be the median absolute deviation from the first adequate training-prefix pool in the following hierarchy: the same relation and prediction mode, the same prediction mode, and the complete update. We set this term to zero when no pool is adequate and define

$$
\tau _ { x } = \operatorname* { m a x } \left\{ { \mathrm { I Q R } } _ { x } , \eta _ { \mathrm { M A D } } \mathrm { M A D } _ { x } ^ { \mathrm { f b } } , \varepsilon _ { \tau } \right\} .\tag{11}
$$

The scale is detached from the gradient. It makes local score differences dimensionless and limits the influence of queries with unusually large score ranges. The parameters $\eta _ { \mathrm { M A D } } > 0$ and $\varepsilon _ { \tau } > 0$ are numerical safeguards rather than learned quantities. The pooled fallback is robust to isolated extreme scores but can reflect shifts in score location across occurrences. It is therefore not solely a measure of dispersion within individual queries.

For candidate $e ,$ the normalized gap relative to the answer is

$$
z _ { x } ( e ) = \frac { s _ { m , k , u } ( q , e ) - s _ { m , k , u } ( q , a ) } { \tau _ { x } } .\tag{12}
$$

For a nonempty ordered candidate multiset $C ~ = ~ ( e _ { 1 } , \ldots , e _ { n } )$ define

$$
U _ { \beta } ( x , C ) = \frac { 1 } { \beta } \log \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \exp \left\{ \beta \left[ z _ { x } ( e _ { i } ) \right] _ { + } \right\} \right) , \qquad \beta > 0 .\tag{13}
$$

The normalization by � prevents cohort size alone from increasing the aggregate and makes it invariant to exact replication of the complete multiset. A repeated entity remains a repeated score instance. The aggregate is zero exactly when no member scores above the answer, and it never exceeds the largest positive normalized gap in �. The parameter $\beta$ determines how strongly the largest active gaps influence the smooth aggregate. The quantity is neither an outranker count nor a probability.

## 4<sub>.</sub>2 S<sub>core-</sub>Bli<sub>n</sub>d M<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d R<sub>e</sub>f<sub>erences</sub>

Inspired by Jones and Love [8], we assume old reference should resemble the newcomer cohort in observable structural characteristics while remaining independent of the scores being compared. For entity � and prediction mode $d ,$ let $b _ { u } ( e , d )$ be its endpoint-degree bin and let $\rho _ { u } ( e , d )$ be its set of incident relation-role pairs, both computed from the cumulative training prefix. Matching uses three nested, score-blind keys. The finest key retains $( d , b _ { u } ( e , d ) , \rho _ { u } ( e , d ) )$ the next drops the relation-role signature, and the terminal key retains only �. For each newcomer, we select the first key whose corresponding old-entity cell is nonempty. Each coarser key only merges cells formed at the preceding level.

![](images/b9d5a4080f69e48b12e0d8139795ee4b82df81e574f7bf3b3927550d07e6bca0.jpg)  
Fi<sub>g</sub>ure 2: Candidate-set interference and the <sub>q</sub>uer<sub>y</sub>-s<sub>p</sub>ecific com<sub>p</sub>utation of Meor.

Within the selected cell, sampling proceeds without replacement until the cell is exhausted and uses replacement only for any remaining positions. Concatenating the selected candidates across newcomer instances yields an ordered old multiset $O _ { x } ^ { ( j ) }$ with $| O _ { x } ^ { ( j ) } | = | N _ { x } |$

We construct � matched draws. Their candidate identifiers are fixed independently of scores and shared by paired methods. Each model then supplies its own scores for those candidates. This design reduces observable cohort imbalance without treating the matched references as causal counterfactuals.

## 4.3 Matched Excess Objective

Let $\mathcal { B } _ { u }$ denote the current refinement batch. The ordered multiset �� contains its eligible replay query occurrences in batch order. An occurrence enters $B _ { M }$ only when its newcomer cohort, scale sample, and terminal old matching pool are nonempty and every required score and scale input is finite. Membership is frozen before computing any method’s aggregate and is shared by all paired controls. Repeated occurrences remain repeated instances. We write $\mathcal { L } _ { \mathrm { h o s t } } ( \theta _ { m , k , u } ; \mathcal { B } _ { u } )$ for the diferentiable refinement objective supplied by the underlying continual KGE learner before Meor is added. This objective retains the learner’s positive training signal for current facts and any replay facts selected by the host.

The mean old reference is

$$
\overline { { U } } _ { O } ( \boldsymbol { x } ) = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } U _ { \beta } \Big ( \boldsymbol { x } , O _ { \boldsymbol { x } } ^ { ( j ) } \Big ) .\tag{14}
$$

For nonempty $B _ { M } ,$ , the Meor loss is

$$
\mathcal { L } _ { \mathrm { M E O R } } = \frac { 1 } { | B _ { M } | } \sum _ { x \in B _ { M } } \left[ U _ { \beta } ( x , N _ { x } ) - \overline { { U } } _ { O } ( x ) \right] _ { + } ^ { 2 } .\tag{15}
$$

The mean in Equation (14) is computed before applying the hinge to positive excess. If the newcomer aggregate equals the mean reference, its contribution is zero even when it exceeds an individual draw. Applying a separate hinge to each draw before averaging would impose downward pressure in that case. The squared hinge has both value and gradient zero at the boundary, and its penalty increases with positive excess.

If $B _ { M }$ is empty, we define $\mathcal { L } _ { \mathrm { M E O R } } = 0$ . With regularization coeficient $\lambda \geq 0 ,$ , the complete refinement objective is

$$
\mathcal { L } ( \theta _ { m , k , u } ; \mathcal { B } _ { u } ) = \mathcal { L } _ { \mathrm { h o s t } } ( \theta _ { m , k , u } ; \mathcal { B } _ { u } ) + \lambda \mathcal { L } _ { \mathrm { M E O R } } .\tag{16}
$$

The host objective retains its positive learning signal for newly admitted entities. Thus Meor controls excessive competition on historical queries rather than treating every newcomer as an error.

## 4<sub>.</sub>4 C<sub>on</sub>t<sub>ro</sub>l<sub>s</sub> f<sub>or</sub> th<sub>e</sub> P<sub>ropose</sub>d M<sub>ec</sub>h<sub>an</sub>i<sub>sm</sub>

The central contribution of Meor is to combine smooth answerrelative aggregation with score-blind structural matching. Two controls isolate these components. Both control losses are defined as zero when $B _ { M }$ is empty; the displayed averages apply otherwise. The Matched Maximum Regularizer (MMR) replaces the smooth aggregate with

$$
V ( x , C ) = \operatorname* { m a x } _ { 1 \leq i \leq | C | } \left[ z _ { x } ( e _ { i } ) \right] _ { + }\tag{17}
$$

and uses the same matched reference cohorts. Its loss is

$$
\mathcal { L } _ { \mathrm { M M R } } = \frac { 1 } { \vert B _ { M } \vert } \sum _ { x \in B _ { M } } \left[ V ( x , N _ { x } ) - \frac { 1 } { J } \sum _ { j = 1 } ^ { J } V ( x , O _ { x } ^ { ( j ) } ) \right] _ { + } ^ { 2 } .\tag{18}
$$

This comparison isolates the efect of distributing pressure across the active score tail rather than controlling only the largest gap.

The Unmatched Old Regularizer (UOR) retains $U _ { \beta }$ but replaces each matched reference with an equal-cardinality multiset $R _ { x } ^ { ( j ) }$ drawn from the terminal role-only old pool. It uses

$$
\mathcal { L } _ { \mathrm { U O R } } = \frac { 1 } { | B _ { M } | } \sum _ { x \in B _ { M } } \left[ U _ { \beta } ( x , N _ { x } ) - \frac { 1 } { J } \sum _ { j = 1 } ^ { J } U _ { \beta } ( x , R _ { x } ^ { ( j ) } ) \right] _ { + } ^ { 2 } .\tag{19}
$$

A<sup>l</sup> orit<sup>h</sup>m 1: Meor objective for a refinement batch   
Input: training-prefix graph at update �, refinement batch   
${ \mathcal { B } } _ { u } ,$ model scores $s _ { m , k , u } ,$ host objective $\mathcal { L } _ { \mathrm { h o s t } } ,$ , and   
parameters $( J , \beta , \lambda )$   
Output: regularized refinement objective $\mathcal { L } ( \theta _ { m , k , u } ; \mathcal { B } _ { u } )$   
1 $B _ { M }  \varnothing ;$   
2 foreach historical replay occurrence $x = ( q , a , u , d ) \in \mathcal { B } _ { u }$ d<sub>o</sub>   
3 construct $N _ { x }$ using Equation (8);   
4 draw the score-blind scale sample $C _ { x } ^ { \mathrm { s c a l e } }$   
5 construct � score-blind matched cohorts $\{ O _ { x } ^ { ( j ) } \} _ { j = 1 } ^ { J }$ with   
$| O _ { x } ^ { ( j ) } | = | N _ { x } | ;$   
6 if � has all required nonempty cohorts and finite score   
and scale inputs then   
7 $\begin{array} { r } { \underline { { \mathbf { \Pi } } } [ \mathbf { \Pi } B _ { M }  B _ { M } \mathrel { \uplus } \{ x \} ] } \end{array}$   
8 $R \gets 0 ;$   
<sub>9</sub> f<sub>oreac</sub>h $x \in B _ { M }$ d<sub>o</sub>   
10 compute the detached scale $\tau _ { x }$ using Equation (11);   
11 compute $z _ { x } ( e )$ using Equation (12) for all   
$e \in N _ { x } \uplus \bigcup _ { j = 1 } ^ { J } O _ { x } ^ { ( j ) } ;$   
12 $\begin{array} { r } { \overline { { U } } _ { O } ( x ) \gets \frac { 1 } { J } \sum _ { j = 1 } ^ { J } U _ { \beta } \Big ( x , O _ { x } ^ { ( j ) } \Big ) ; } \end{array}$   
13 $R \gets R + \left[ U _ { \beta } ( x , N _ { x } ) - \overline { { U } } _ { O } ( x ) \right] _ { + } ^ { 2 } ;$   
$\mathcal { L } _ { \mathrm { M E O R } }  \Bigg \{ 0 ,$ $B _ { M } = \varnothing ,$   
14   
otherwise,   
15 $\mathcal { L } ( \theta _ { m , k , u } ; \mathcal { \vec { B } } _ { u } ) \gets \mathcal { L } _ { \mathrm { h o s t } } ( \theta _ { m , k , u } ; \mathcal { B } _ { u } ) + \lambda \mathcal { L } _ { \mathrm { M E O R } } ;$

This comparison isolates the contribution of structural matching. Replay uses the same host refinement with zero regularization and therefore isolates the contribution of the intervention.

A persistent admission-cohort calibrator provides a score-only control. Let �(�) denote the update at which entity � first entered the graph. For training realization $k , \alpha _ { k , c , r , d } > 0$ is a multiplicative coeficient and $g _ { k , c , r , d } \in \mathbb { R }$ is a dimensionless ofset coeficient. Both are fitted from replay validation scores for admission cohort �, relation �, and prediction mode � when the cohort is admitted, then retained at later updates. The detached statistic $\bar { \tau } _ { k , u , r , d } ^ { \mathrm { v a l } }$ is the median finite validation scale for the same relation and prediction mode at update �, with prediction-mode and update-level fallbacks. When this scale is available, the calibrator multiplies the raw replay score $s _ { \mathrm { R e p l a y } , k , u } ( q , e )$ by $\alpha _ { k , c ( e ) , r ( q ) , d }$ and adds the scale-adjusted ofset $g _ { k , c ( e ) , r ( q ) , d } \bar { \tau } _ { k , u , r ( q ) , d } ^ { \mathrm { v a l } } .$ If no finite scale summary is available, the raw replay score is retained. The initial cohort uses the identity transform, $\alpha = 1$ and $g = 0$ . This control adds no training trajectory and separates an intervention applied during training from a post hoc adjustment of cohort score scales.

All controls share the host model, update data, replay information, optimization budget, and trainable parameter scope. The matched maximum control and Meor also share their matched reference identifiers. Their concrete training and calibration settings appear in Section 5.

## 4.5 Inter<sub>p</sub>retation and Com<sub>p</sub>utational Cost

Equation (15) has zero gradient whenever newcomer competition is no greater than the mean matched reference. When active, the smooth aggregate distributes pressure across several above-answer candidates rather than only the largest one. The loss acts on scores, so the evaluation measures historical current-universe ranking and target-newcomer acquisition directly. These outcomes capture whether score changes cross the answer and whether legitimate newcomer learning is preserved.

If one candidate score requires $O ( p )$ operations, candidate scoring for one refinement batch without score reuse is

$$
O \left( p \sum _ { x \in B _ { M } } \big ( ( 1 + J ) | N _ { x } | + K _ { \mathrm { s c a l e } } \big ) \right) .\tag{20}
$$

The complete newcomer cohort and every reference cohort are scored, so computational cost increases with the size of the admission batch. Approximate cohort scoring lies outside the scope of this study.

## 5 Ex<sub>p</sub>eriments and Results

The evaluation centers on whether Meor improves historical ranking by reducing candidate-set interference. ENTITY–ComplEx provides the primary comparison with replay, persistent calibration, MMR, and UOR. Direct ablations identify the contribution of each component, while experiments on FBInc-S and FBInc-L examine transfer across diferent rates of entity admission, embedding models, and continual KGE hosts. All comparative claims are drawn from paired runs within the same experimental setting.

## 5.1 Ex<sub>p</sub>erimental Settin<sub>g</sub>

Our experiments use four five-snapshot entity-growth streams. EN-TITY is the primary FB15K-237 [24] stream introduced with LKGE [3]. FBInc-S and FBInc-L are the small-growth and large-growth streams distributed with the informed-initialization benchmark [16]. WN-CKGE is the WordNet stream used by FastKGE [14]. It represents the applicability boundary discussed in Section 6 and is not treated as a positive eficacy setting. Snapshot 0 initializes the model, and Snapshots 1 through 4 form the continual sequence. Table 1 reports the cumulative graph size and the facts introduced at each snapshot.

We evaluate filtered head and tail prediction after every update. The filter removes other positives visible in the current graph prefix while retaining the recorded answer, and exact score ties are resolved by increasing canonical entity identifier. Each endpoint assigns equal weight to the update and prediction-direction combinations defined in Section 3. Historical current-universe MRR $H _ { \mathrm { c u r } }$ measures the ranking of established answers against every entity available at the current update. Historical old-universe MRR $H _ { \mathrm { o l d } }$ restricts the same evaluation to previously known candidates. Their diference, $D _ { \mathrm { M C I } } = H _ { \mathrm { o l d } } - H _ { \mathrm { c u r } } ,$ , measures the rank loss caused by candidate admission. Target-newcomer MRR $A _ { \mathrm { T N } }$ evaluates queries whose correct answer is newly admitted.

Although Meor is independent of the embedding architecture, the primary study uses ComplEx with 200 complex coordinates and float64 arithmetic. Every method optimizes the same softplus link-prediction objective with Adam, a learning rate of $1 0 ^ { - 4 }$ $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 ) , \epsilon = 1 0 ^ { - 8 }$ , and no weight decay. Training uses batches of 2,048 positive facts with ten negative samples per fact. Base training runs for at most 200 epochs with an early-stopping patience of three, and replay retains at most 2,048 historical facts. Continual refinement makes one ordered pass over the facts introduced at the current snapshot. Previously learned entity and relation embeddings remain fixed during refinement, so the intervention updates only newly admitted parameters.

Table 1: Dataset statistics. Entit<sub>y</sub> and relation counts are cumulative<sub>;</sub> Facts is the number of tri<sub>p</sub>les introduced at each sna<sub>p</sub>shot
<table><tr><td>Dataset</td><td colspan="3">Snapshot 0</td><td colspan="3">Snapshot 1</td><td colspan="3">Snapshot 2</td><td colspan="3">Snapshot 3</td><td colspan="3">Snapshot 4</td></tr><tr><td></td><td>|8|</td><td>|R|</td><td>Facts</td><td>|ε|</td><td>|R|</td><td>Facts</td><td>|ε|</td><td>|R|</td><td>Facts</td><td>|ε|</td><td>|R|</td><td>Facts</td><td>|ε|</td><td>|R|</td><td>Facts</td></tr><tr><td>ENTITY</td><td>2,909</td><td>233</td><td>46,388</td><td>5,817</td><td>236</td><td>72,111</td><td>8,725</td><td>236</td><td>73,785</td><td>11,633</td><td>237</td><td>70,506</td><td>14,541</td><td>237</td><td>47,326</td></tr><tr><td>FBInc-S</td><td>2,909</td><td>233</td><td>46,388</td><td>2,919</td><td>233</td><td>235</td><td>2,930</td><td>233</td><td>152</td><td>2,940</td><td>233</td><td>180</td><td>2,950</td><td>233</td><td>239</td></tr><tr><td>FBInc-L</td><td>2,909</td><td>233</td><td>46,388</td><td>3,010</td><td>233</td><td>3,110</td><td>3,110</td><td>234</td><td>2,736</td><td>3,211</td><td>234</td><td>2,908</td><td>3,312</td><td>234</td><td>3,105</td></tr><tr><td>WN-CKGE</td><td>24,567</td><td>11</td><td>55,801</td><td>28,660</td><td>11</td><td>9,300</td><td>32,754</td><td>11</td><td>9,300</td><td>36,848</td><td>11</td><td>9,300</td><td>40,943</td><td>11</td><td>9,302</td></tr></table>

We instantiate Meor with $\beta = 5$ , four matched reference cohorts, and at most 256 old entities in each scale sample. Reference construction matches prediction role, degree bin, and incident relation-role signature before falling back to role and degree and then to role alone. The regularization coeficient for each ENTITY method is fixed before evaluation by the score-blind gradient-balance rule. Replay uses zero regularization. Persistent calibration applies the afine transformation �� + $g \bar { \tau } ^ { \mathrm { v a l } }$ to newcomer scores over the fixed grid

$$
\alpha \in \{ 0 . 5 , 0 . 7 5 , 1 , 1 . 2 5 , 1 . 5 \} , \qquad g \in \{ - 1 , - 0 . 5 , 0 , 0 . 5 , 1 \} ,
$$

where $\bar { \tau } ^ { \mathrm { v a l } }$ is the corresponding replay-validation score scale. Calibration retains transformations whose $A _ { \mathrm { T N } }$ is no more than 0.005 below replay and selects the retained transformation with the highest $H _ { \mathrm { c u r } } .$

The primary, ablation, and transfer studies each use eight paired runs. Within a pair, the compared methods share initialization, graph stream, fact order, replay sample, negative samples, stopping decisions, and evaluation queries. Comparisons involving matched references also share the relevant reference identities. The direct ablations use the same coeficient-selection rule and change only the component named in Table 4. The transfer study adds Meor to replay with ComplEx [25], DistMult [27], and TransE [1] and to the LKGE [3] and IncDE [13] update procedures with their TransE backbones. Comparisons are made only within the same dataset, host, and backbone.

For the primary and ablation studies, LB denotes the one-sided 95% lower confidence bound computed from the eight unrounded paired efects. A positive lower bound supports an improvement in $H _ { \mathrm { c u r } }$ or a reduction in $D _ { \mathrm { M C I } }$ . Preservation requires the lower bound for $\Delta H _ { \mathrm { o l d } }$ or $\Delta A _ { \mathrm { T N } }$ to remain above −0.005. W/T/L counts positive, zero, and negative full-precision $H _ { \mathrm { c u r } }$ efects. The transfer study reports two-sided paired 95% confidence intervals for $H _ { \mathrm { c u r } }$ and the same one-sided lower bound for $\Delta A _ { \mathrm { T N } }$ . These intervals are unadjusted summaries of prespecified, cell-specific comparisons rather than simultaneous family-wise intervals.

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> Ab<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t <sub>per</sub>f<sub>ormance</sub> <sub>o</sub>f M<sub>eor</sub> <sub>an</sub>d m<sub>ec</sub>h<sub>a</sub>ni<sub>s</sub>m-m<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d <sub>co</sub>ntr<sub>o</sub>l<sub>s</sub> in th<sub>e</sub> <sub>p</sub>rim<sub>a</sub>r<sub>y</sub> ENTITY– Com<sub>p</sub>lEx settin<sub>g</sub>.
<table><tr><td>Method</td><td> $H _ { \mathrm { c u r } }$  ↑</td><td>Hold ↑</td><td>DMCI↓</td><td>ATN ↑</td></tr><tr><td>Replay</td><td>0.0830</td><td>0.1199</td><td>0.0369</td><td>0.1700</td></tr><tr><td>Persistent calibration</td><td>0.0816</td><td>一</td><td>一</td><td>0.1753</td></tr><tr><td>MMR</td><td>0.0836</td><td>0.1200</td><td>0.0364</td><td>0.1695</td></tr><tr><td>UOR</td><td>0.0842</td><td>0.1200</td><td>0.0358</td><td>0.1698</td></tr><tr><td>MEOR</td><td>0.0887</td><td>0.1201</td><td>0.0314</td><td>0.1715</td></tr></table>

## 5.2 Primar<sub>y</sub> Results on ENTITY

Table 2 reports the endpoint means for ENTITY with ComplEx. At 0.0887, Meor attains the highest $H _ { \mathrm { c u r } }$ . It also records the highest $H _ { \mathrm { o l d } }$ and the lowest $D _ { \mathrm { M C I } } .$ , while increasing $A _ { \mathrm { T N } }$ above replay, MMR, and UOR. Persistent calibration attains the highest $A _ { \mathrm { T N } }$ but a lower $H _ { \mathrm { c u r } } .$ . The calibration analysis reports only $H _ { \mathrm { c u r } }$ and $A _ { \mathrm { T N } } ; H _ { \mathrm { o l d } }$ and $D _ { \mathrm { M C I } }$ were not recorded for this control. All endpoints are equalweight means over update and prediction-direction combinations.

The absolute endpoints show the aggregate ordering, while the paired analysis determines whether that ordering is consistent across runs. Table 3 reports the paired efects of Meor relative to each mechanism-matched control.

Meor improves $H _ { \mathrm { c u r } }$ against every control. The mean gain is 0.0057 over replay, 0.0071 over persistent calibration, 0.0051 over MMR, and 0.0045 over UOR. The corresponding lower bounds are 0.0052, 0.0065, 0.0047, and 0.0030, respectively. The efect is positive in all eight paired runs against replay, persistent calibration, and MMR and in seven of eight runs against UOR.

The decomposition against replay locates the gain in candidate admission rather than parameter retention. The 0.0057 increase in $H _ { \mathrm { c u r } }$ combines a much smaller 0.0002 change in $H _ { \mathrm { o l d } }$ with a 0.0055 reduction in $D _ { \mathrm { M C I } }$ . The same pattern appears against MMR, where a 0.0051 gain combines a 0.0001 old-universe change with a 0.0050 interference reduction, and against UOR, where the corresponding efects are 0.0045, 0.0001, and 0.0044. Every interference-reduction lower bound is positive. These paired decompositions indicate that the gains arise primarily from reduced competition by newly admitted entities, while the mean changes in old-universe ranking are small.

Our results meet the prespecified criterion for retained and newcomer performance. The $H _ { \mathrm { o l d } }$ lower bounds against replay, MMR, and UOR are −0.0001, −0.0003, and −0.0011, respectively, all above the −0.005 margin. Meor improves �<sub>TN</sub> against replay, MMR, and UOR by 0.0015, 0.0020, and 0.0017, with positive lower bounds of 0.0008, 0.0015, and 0.0012. Persistent calibration retains a mean $A _ { \mathrm { T N } }$ advantage of 0.0038. The one-sided lower bound for the corresponding Meor-minus-calibration efect is −0.0043, which also remains above the preservation margin.

Table 3: Paired efects of Meor relative to mechanism-matched controls on ENTITY–Com<sub>p</sub>lEx across ei<sub>g</sub>ht <sub>p</sub>aired trainin<sub>g</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>ons.</sub>
<table><tr><td></td><td colspan="2"> $\Delta H _ { \mathrm { c u r } } \ \cdot$  ←</td><td colspan="2"> $\Delta H _ { \mathrm { o l d } }$  ←</td><td colspan="2"> $D _ { \mathrm { M C I } }$  reduction↑</td><td colspan="2"> $\Delta A _ { \mathrm { T N } }$  ←</td><td></td></tr><tr><td>Comparator</td><td>Mean</td><td>LB</td><td>Mean</td><td>LB</td><td>Mean</td><td>LB</td><td>Mean</td><td>LB</td><td>W/T/L</td></tr><tr><td>Replay</td><td>0.0057</td><td>0.0052</td><td>0.0002</td><td>-0.0001</td><td>0.0055</td><td>0.0051</td><td>0.0015</td><td>0.0008</td><td> $8 / 0 / 0$ </td></tr><tr><td>Persistent calibration</td><td>0.0071</td><td>0.0065</td><td>一</td><td>一</td><td>一</td><td></td><td>-0.0038</td><td>-0.0043</td><td> $8 / 0 / 0$ </td></tr><tr><td>MMR</td><td>0.0051</td><td>0.0047</td><td>0.0001</td><td>-0.0003</td><td>0.0050</td><td>0.0049</td><td>0.0020</td><td>0.0015</td><td> $8 / 0 / 0$ </td></tr><tr><td>UOR</td><td>0.0045</td><td>0.0030</td><td>0.0001</td><td>-0.0011</td><td>0.0044</td><td>0.0041</td><td>0.0017</td><td>0.0012</td><td> $7 / 0 / 1$ </td></tr></table>

## 5<sub>.</sub>3 M<sub>ec</sub>h<sub>an</sub>i<sub>sm</sub> A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub>

The primary controls distinguish the two central components of Meor. MMR preserves query-specific matched references and oldreference centering but replaces smooth tail aggregation with a maximum. UOR preserves smooth aggregation and query-specific drawing but uses role-only old references. The shufled-reference comparison breaks query-specific assignment while retaining the matched pools, and the uncentered comparison removes subtraction of the old reference. All four ablations use the same score-blind coeficient rule and paired inputs. Table 4 reports the resulting efects.

Each component makes a distinct contribution. Smooth tail aggregation produces the largest gain over its matched alternative, with a mean $H _ { \mathrm { c u r } }$ efect of 0.0051 and a lower bound of 0.0047. Structural matching follows closely with a mean efect of 0.0045 and a lower bound of 0.0030. Old-reference centering contributes 0.0011, and query-specific assignment contributes 0.0008, with lower bounds of 0.0006 and 0.0005, respectively. Every ablation also yields a positive lower bound for reducing $D _ { \mathrm { M C I } } .$ . The $A _ { \mathrm { T N } }$ efects are positive on average in all four contrasts, and their lower bounds remain above the preservation margin. The complete method therefore benefits from each element of its reference construction and aggregation.

## 5<sub>.</sub>4 T<sub>rans</sub>f<sub>er</sub> A<sub>cross</sub> St<sub>reams,</sub> B<sub>ac</sub>kb<sub>ones, an</sub>d Hosts

The transfer study treats Meor as an intervention within a named host rather than as a replacement continual KGE architecture. Replay is evaluated with ComplEx, DistMult, TransE, LKGE–TransE, and IncDE–TransE. Each row of Table 5 is a separate paired comparison within the specified dataset, host, and backbone. The $H _ { \mathrm { c u r } }$ column reports the mean efect and its two-sided 95% interval; the final column reports the one-sided lower bound for $\Delta A _ { \mathrm { T N } }$

Adding Meor improves $H _ { \mathrm { c u r } }$ in all ten transfer settings, and every 95% interval excludes zero. The replay gains range from 0.0020 to 0.0026 across the two streams and three backbones. The gains within LKGE are 0.0023 on FBInc-S and 0.0026 on FBInc-L, while those within IncDE are 0.0029 and 0.0032. Four settings improve in all eight paired runs, and each remaining setting improves in at least seven.

The $A _ { \mathrm { T N } }$ efects are slightly negative, ranging from −0.0001 to −0.0015, but every lower bound remains above the −0.005 preser vation margin. The transfer study therefore shows that Meor improves historical current-universe ranking across the reported growth regimes, embedding models, and continual hosts while preserving newcomer acquisition under the stated criterion.

Together, the primary results, direct ablations, and transfer study support a coherent account of Meor. Its historical-ranking gains arise chiefly from reduced candidate-set interference, each component contributes to that reduction, and the intervention remains efective across the named host configurations without materially impairing newly admitted entities.

## 6 Limitations

Meor is designed for entity-growth streams in which newly admitted entities become overly competitive answers to historical queries. WN-CKGE defines the complementary boundary. In the registered WN-CKGE diagnostic, Meor and MMR each had zero active records among 64 calibration records, indicating that aggregate newcomer pressure did not exceed the matched old-candidate reference under the fixed construction. Their regularization terms therefore contributed neither loss nor gradient in the evaluated records. The suppression mechanism had no direct optimization efect because there was no excess candidate pressure to remove. This selective activation is central to the method. Meor intervenes when entity admission disturbs historical ranking and otherwise leaves refinement governed by the host objective. WN-CKGE thus documents an inactive applicability regime rather than a positive eficacy comparison.

The method does not address every source of degradation in a continual knowledge graph. Relation-only or fact-only updates can still alter established representations, but they do not create the entity-candidate expansion captured by $D _ { \mathrm { M C l } }$ . Meor is consequently a complement to replay, distillation, and parameter-stability methods rather than a replacement for them. Its role is specific and measurable. It controls admission-induced competition while the host learner remains responsible for retaining prior knowledge and acquiring new facts.

The quality of the reference depends on the structural information available for matching. Prediction role, degree, and incident relation-role signatures provide a score-blind basis for comparison, but they do not form a causal counterfactual. Sparse structural cells also require coarser fallbacks, which reduce the specificity ofthe reference cohort. The precision of the reference is therefore bounded by the structural resolution available in each update. Computational cost presents a separate boundary because exact cohort scoring grows with the number of admitted entities and matched draws. Very large admission batches therefore require eficient batching or approximate cohort evaluation. The reported experiments establish efectiveness on ENTITY, FBInc-S, and FBInc-L, while WN-CKGE identifies the inactive boundary of the intervention. The empirical claim is limited to the reported streams, backbones, hosts, and fixed aggregation and matching settings. Other prediction tasks require candidate and reference definitions appropriate to their ranking spaces.

Table 4: Direct ablations of Meor’s reference construction and a<sub>gg</sub>re<sub>g</sub>ation on ENTITY–Com<sub>p</sub>lEx.
<table><tr><td rowspan="2">Isolated property</td><td rowspan="2">Comparator</td><td colspan="2"> $\Delta H _ { \mathrm { c u r } }$  ←</td><td colspan="2">DMCI reduction↑</td><td colspan="2"> $\Delta A _ { \mathrm { T N } } \uparrow$ </td></tr><tr><td>Mean</td><td>LB</td><td>Mean</td><td>LB</td><td>Mean</td><td>LB</td></tr><tr><td>Smooth tail aggregation</td><td>MMR (maximum)</td><td>0.0051</td><td>0.0047</td><td>0.0050</td><td>0.0049</td><td>0.0020</td><td>0.0015</td></tr><tr><td>Structural matching</td><td>UOR (role-only references)</td><td>0.0045</td><td>0.0030</td><td>0.0044</td><td>0.0041</td><td>0.0017</td><td>0.0012</td></tr><tr><td>Query-specific assignment</td><td>Shuffled matched references</td><td>0.0008</td><td>0.0005</td><td>0.0007</td><td>0.0002</td><td>0.0002</td><td>-0.0002</td></tr><tr><td>Old-reference centering</td><td>Uncentered smooth penalty</td><td>0.0011</td><td>0.0006</td><td>0.0009</td><td>0.0003</td><td>0.0003</td><td>-0.0002</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 5<sub>:</sub> P<sub>a</sub>i<sub>re</sub>d <sub>e</sub>f<sub>ec</sub>t<sub>s o</sub>f <sub>a</sub>ddi<sub>ng</sub> M<sub>eor</sub> t<sub>o rep</sub>l<sub>ay</sub> h<sub>os</sub>t<sub>s an</sub>d <sub>source-</sub>b<sub>oun</sub>d <sub>con</sub>ti<sub>nua</sub>l<sub>-</sub>l<sub>earn</sub>i<sub>ng a</sub>d<sub>ap</sub>t<sub>ers on</sub> FBI<sub>nc-</sub>S <sub>an</sub>d FBI<sub>nc-</sub>L<sub>.</sub>
<table><tr><td>Dataset</td><td>Host family</td><td>Host-backbone</td><td> $\Delta H _ { \mathrm { c u r } } \ ^ { \prime }$  ←</td><td>Paired 95% CI</td><td>W/T/L</td><td> $\Delta A _ { \mathrm { T N } } \ ^ { \prime }$  ←</td><td> $\Delta A _ { \mathrm { T N } } \mathrm { L B }$ </td></tr><tr><td>FBInc-S</td><td>Replay</td><td>ComplEx</td><td>0.0026</td><td>[0.0017, 0.0043]</td><td>8/0/0</td><td>-0.0004</td><td>-0.0011</td></tr><tr><td>FBInc-S</td><td>Replay</td><td>DistMult</td><td>0.0022</td><td>[0.0010, 0.0034]</td><td>7/1/0</td><td>-0.0002</td><td>-0.0008</td></tr><tr><td>FBInc-S</td><td>Replay</td><td>TransE</td><td>0.0020</td><td>[0.0009, 0.0031]</td><td>7/1/0</td><td>-0.0001</td><td>-0.0006</td></tr><tr><td>FBInc-S</td><td>Source-Bound</td><td>LKGE-TransE</td><td>0.0023</td><td>[0.0011, 0.0035]</td><td>7/0/1</td><td>-0.0012</td><td>-0.0024</td></tr><tr><td>FBInc-S</td><td>Source-Bound</td><td>IncDE-TransE</td><td>0.0029</td><td>[0.0016, 0.0042]</td><td>8/0/0</td><td>-0.0010</td><td>-0.0021</td></tr><tr><td>FBInc-L</td><td>Replay</td><td>ComplEx</td><td>0.0026</td><td>[0.0019, 0.0045]</td><td>8/0/0</td><td>-0.0008</td><td>-0.0016</td></tr><tr><td>FBInc-L</td><td>Replay</td><td>DistMult</td><td>0.0023</td><td>[0.0011, 0.0035]</td><td>7/1/0</td><td>-0.0003</td><td>-0.0009</td></tr><tr><td>FBInc-L</td><td>Replay</td><td>TransE</td><td>0.0021</td><td>[0.0010, 0.0032]</td><td>7/1/0</td><td>-0.0002</td><td>-0.0007</td></tr><tr><td>FBInc-L</td><td>Source-Bound</td><td>LKGE-TransE</td><td>0.0026</td><td>[0.0012, 0.0040]</td><td>7/0/1</td><td>-0.0015</td><td>-0.0029</td></tr><tr><td>FBInc-L</td><td>Source-Bound</td><td>IncDE-TransE</td><td>0.0032</td><td>[0.0018, 0.0046]</td><td>8/0/0</td><td>-0.0014</td><td>-0.0027</td></tr></table>

## 7 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

Continual knowledge graph embedding must account for more than changes to previously learned representations. When the entity vocabulary grows, every admitted entity becomes a new candidate for historical queries and can lower the rank of an established answer even when the ordering among old entities is preserved. We formalized this efect as candidate-set interference and separated it from parameter forgetting through a same-checkpoint rank decomposition. We then introduced Meor, a host-level regularizer that compares answer-relative newcomer competition with scoreblind, structurally matched old references. Its one-sided objective suppresses only the excess competition attributable to admission, while the host loss continues to learn legitimate new knowledge.

The experiments show that this intervention improves the ranking quantity it is designed to control. On ENTITY–ComplEx, Meor increases historical current-universe MRR by 0.0057 over replay and reduces $D _ { \mathrm { M C I } }$ by 0.0055, while satisfying the preservation criteria for old-universe ranking and newcomer acquisition. Comparisons with persistent calibration, MMR, and UOR show that the gain persists relative to score rescaling, maximum-only control, and unmatched references. Direct ablations further support the contributions of smooth tail aggregation, structural matching, query-specific assignment, and old-reference centering. Across FBInc-S and FBInc-L, adding Meor improves historical current-universe MRR in all ten reported combinations of stream, host, and backbone, with every paired confidence interval excluding zero and every newcomeracquisition bound remaining within the preservation margin.

These findings establish candidate-set interference as a distinct and controllable source of historical rank loss in continual KGE. They also show that it can be addressed without redesigning the underlying embedding architecture or continual learner. By making admission-induced candidate pressure part of refinement, Meor extends continual learning from preserving representations to managing the ranking consequences of graph growth.

## Ethi<sub>ca</sub>l C<sub>o</sub>n<sub>s</sub>id<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>

This work studies continual knowledge graph embedding using established research benchmarks and does not involve human participants or the collection of new personal data. Its direct ethical risk is therefore limited. Nevertheless, knowledge graphs may contain inaccurate, outdated, sensitive, or socially biased information, and link prediction can reproduce these properties or infer unrecorded associations. Because Meor regulates the competitive pressure introduced by newly admitted entities, an improperly configured deployment could also favor established entities and delay the recognition of legitimate new facts or underrepresented entities. The one-sided intervention and the explicit preservation of newcomer acquisition reduce this risk, but they do not replace careful dataset and deployment auditing. Applications involving people or consequential decisions should examine subgroup rank ing behavior, verify data provenance, restrict access to sensitive predictions, provide correction and removal procedures, and retain appropriate human oversight. The method is intended for research on knowledge graph ranking and should not be used as the sole basis for high-stakes decisions.

## R<sub>e</sub>f<sub>erences</sub>

[1] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Ok sana Yakhnenko. 2013. Translating Embeddings for Modeling Multi-relational Data. In Advances in Neural Information Processing Systems, C.J. Burges, L. Bot tou, M. Welling, Z. Ghahramani, and K. Weinberger (Eds.), Vol. 26. Curran Associates, Inc. https://proceedings.neurips.cc/paper\_files/paper/2013/file/ 1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

[2] Jiahang Cao,Jinyuan Fang, Zaiqiao Meng, and Shangsong Liang. 2024. Knowledge Graph Embedding: A Survey from the Perspective ofRepresentation Spaces. ACM Comput. Surv. 56, 6, Article 159 (March 2024), 42 pages. doi:10.1145/3643806

[3] Yuanning Cui, Yuxin Wang, Zequn Sun, Wenqiang Liu, Yiqiao Jiang, Kexin Han, and Wei Hu. 2023. Lifelong embedding learning and transfer for growing knowl edge graphs. In Proceedings of the AAAI conference on artificial intelligence, Vol. 37. AAAI Press, 4217–4224. doi:10.1609/aaai.v37i4.25539

[4] Angel Daruna, Mehul Gupta, Mohan Sridharan, and Sonia Chernova. 2021. Con tinual Learning of Knowledge Graph Embeddings. IEEE Robotics and Automation Letters 6, 2 (2021), 1128–1135. doi:10.1109/LRA.2021.3056071

[5] Shib Sankar Dasgupta, Swayambhu Nath Ray, and Partha Talukdar. 2018. HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Ellen Rilof, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, Brussels, Belgium, 2001–2011. doi:10.18653/v1/D18-1225

[6] Frederik Michel Dekking, Cornelis Kraaikamp, Hendrik Paul Lopuhaä, and Ludolf Erwin Meester. 2005. A Modern Introduction to Probability and Statistics: Understanding Why and How. Springer London, London, United Kingdom, 234– 236. doi:10.1007/1-84628-168-7

[7] Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. 2020. Diachronic Embedding for Temporal Knowledge Graph Completion. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 34. AAAI Press 3988–3995. doi:10.1609/aaai.v34i04.5815

[8] Matt Jones and Bradley C. Love. 2007. Beyond common features: The role of roles in determining similarity. Cognitive Psychology 55, 3 (2007), 196–231. doi:10.1016/j.cogpsych.2006.09.004

[9] Xiaoyu Kou, Yankai Lin, Shaobo Liu, Peng Li, Jie Zhou, and Yan Zhang. 2020. Disentangle-based Continual Graph Representation Learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 2961–2972. doi:10.18653/v1/2020.emnlp main.237

[10] Kyung-Hwan Lee and Dong-Wan Choi. 2026. STARK: Structure-Aware and Adaptive Representation Learning for Continual Knowledge Graph Embedding. In Proceedings of the ACM Web Conference 2026 (United Arab Emirates) (WWW ’26). Association for Computing Machinery, New York, NY, USA, 3941–3951. doi:10.1145/3774904.3792391

[11] Linyu Li, Zhi Jin, Yuanpeng He, Dongming Jin, Yichi Zhang, Haoran Duan, Xuan Zhang, Zhengwei Tao, and Tashi Nyima. 2026. Learning to Evolve: Bayesian-Guided Continual Knowledge Graph Embedding. In Proceedings of the ACM Web Conference 2026 (United Arab Emirates) (WWW ’26). Association for Computing Machinery, New York, NY, USA, 4372–4383. doi:10.1145/3774904.3792692

[12] Yifei Li, Lingling Zhang, Hang Yan, Tianzhe Zhao, Zihan Ma, Muye Huang, and Jun Liu. 2025. SAGE: Scale-Aware Gradual Evolution for Continual Knowl edge Graph Embedding. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (Toronto ON, Canada) (KDD ’25). Association for Computing Machinery, New York, NY, USA, 1600–1611. doi:10.1145/3711896.3737115

[13] Jiajun Liu, Wenjun Ke, Peng Wang, Ziyu Shang, Jinhua Gao, Guozheng Li, Ke Ji, and Yanhe Liu. 2024. Towards Continual Knowledge Graph Embedding via Incremental Distillation. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 38. AAAI Press, 8759–8768. doi:10.1609/aaai.v38i8.28722

[14] Jiajun Liu, Wenjun Ke, Peng Wang, Jiahao Wang, Jinhua Gao, Ziyu Shang, Guozheng Li, Zijie Xu, Ke Ji, and Yining Li. 2024. Fast and Continual Knowledge Graph Embedding via Incremental LoRA. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, Kate Larson (Ed.). International Joint Conferences on Artificial Intelligence Organization, 2198–2206. doi:10.24963/ijcai.2024/243 Main Track.

[15] Mehrnoosh Mirtaheri, Mohammad Rostami, and Aram Galstyan. 2023. History repeats: Overcoming catastrophic forgetting for event-centric temporal knowledge graph completion. In Findings ofthe Association for Computational Linguistics: ACL 2023, Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 7740–7755. doi:10.18653/v1/2023.findings-acl.490

[16] Gerard Pons, Besim Bilalli, and Anna Queralt. 2026. Improving continual knowledge graph embedding via informed initialization. Neurocomputing 695 (2026), 134045. doi:10.1016/j.neucom.2026.134045

[17] Gerard Pons, Carlos Escolano, Besim Bilalli, and Anna Queralt. 2026. Revisiting Catastrophic Forgetting in Continual Knowledge Graph Embedding. arXiv:2604.19401 [cs.LG] doi:10.48550/arXiv.2604.19401

[18] Jing Qi, Yuxiang Wang, Zhiyuan Yu, Xiaoliang Xu, Yuanshi Zheng, and Tianxing Wu. 2026. Multi-Faceted Continual Knowledge Graph Embedding for Semantic Aware Link Prediction. In Proceedings ofthe 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (Australia) (SIGIR ’26). Association for Computing Machinery, New York, NY, USA, 1462–1472. doi:10.1145/3805712.3809683

[19] David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne. 2019. Experience Replay for Continual Learning. In Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Vol. 32. Curran Associates, Inc. https://proceedings.neurips.cc/paper\_files/paper/2019/file/ fa7cdfad1a5aaf8370ebeda47a1f1c3-Paper.pdf

[20] Andrea Rossi, Denilson Barbosa, Donatella Firmani, Antonio Matinata, and Paolo Merialdo. 2021. Knowledge Graph Embedding for Link Prediction: A Comparative Analysis. ACM Trans. Knowl. Discov. Data 15, 2, Article 14 (Jan. 2021), 49 pages. doi:10.1145/3424672

[21] Andrea Rossi, Donatella Firmani, Paolo Merialdo, and Tommaso Teofili. 2022. Explaining Link Prediction Systems based on Knowledge Graph Embeddings. In Proceedings of the 2022 International Conference on Management of Data (Philadel phia, PA, USA) (SIGMOD ’22). Association for Computing Machinery, New York, NY, USA, 2062–2075. doi:10.1145/3514221.3517887

[22] Xiangjun Shi, Chong Mu, Lizong Zhang, Jinchuan Zhang, and Qianghua Yuan. 2026. DTD: Dynamic temperature distillation for continual knowledge graph embedding. Neurocomputing 701 (2026), 134563. doi:10.1016/j.neucom.2026. 134563

[23] Ailin Song, Yuhong Chen, Yusong Wang, Shuai Zhong, and Mingkun Xu. 2025. Orchestrating Plasticity and Stability: A Continual Knowledge Graph Embedding Framework with Bio-Inspired Dual-Mask Mechanism. In Proceedings ofthe 16th Asian Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 260), Vu Nguyen and Hsuan-Tien Lin (Eds.). PMLR, 1208–1223. https: //proceedings.mlr.press/v260/song25a.html

[24] Kristina Toutanova and Danqi Chen. 2015. Observed versus latent features for knowledge base and text inference. In Proceedings ofthe 3rd Workshop on Continuous Vector Space Models and their Compositionality, Alexandre Allauzen, Edward Grefenstette, Karl Moritz Hermann, Hugo Larochelle, and Scott Wentau Yih (Eds.). Association for Computational Linguistics, Beijing, China, 57–66. doi:10.18653/v1/W15-4007

[25] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Eric Gaussier, and Guillaume Bouchard. 2016. Complex embeddings for simple link prediction. In Proceedings ofThe 33rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 48), Maria Florina Balcan and Kilian Q. Weinberger (Eds.). PMLR, New York, New York, USA, 2071–2080. https://proceedings.mlr.press/ v48/trouillon16.html

[26] Changyi Xiao, Xiangnan He, and Yixin Cao. 2023. Knowledge Graph Embedding by Normalizing Flows. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 37. AAAI Press, 4756–4764. doi:10.1609/aaai.v37i4.25600

[27] Bishan Yang, Scott Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. In Proceedings of the International Conference on Learning Representations (ICLR) 2015. https://www.microsoft.com/en-us/research/publication/embeddingentities-and-relations-for-learning-and-inference-in-knowledge-bases

[28] Yuchao Zhang, Xiangjie Kong, Zhehui Shen, Jianxin Li, Qiuhua Yi, Guojiang Shen, and Bo Dong. 2024. A survey on temporal knowledge graph embedding: Models and applications. Knowledge-Based Systems 304 (2024), 112454. doi:10. 1016/j.knosys.2024.112454

[29] Lijing Zhu, Dong Hyun Jeon, Wenbo Sun, Li Yang, Yixin Xie, and Shuteng Niu. 2024. Flexible Memory Rotation (FMR): Rotated Representation with Dynamic

Regularization to Overcome Catastrophic Forgetting in Continual Knowledge Graph Learning. In 2024 IEEE International Conference on Big Data (BigData). 6180–6189. doi:10.1109/BigData62323.2024.10825244

[30] Lijing Zhu, Qizhen Lan, Qing Tian, Wenbo Sun, Li Yang, Lu Xia, Yixin Xie, Xi Xiao, Tiehang Duan, Cui Tao, and Shuteng Niu. 2025. ETT-CKGE: Eficient Task Driven Tokens for Continual Knowledge Graph Embedding. In Machine Learning

and Knowledge Discovery in Databases. Research Track, Rita P. Ribeiro, Bernhard Pfahringer, Nathalie Japkowicz, Pedro Larrañaga, Alípio M. Jorge, Carlos Soares, Pedro H. Abreu, and João Gama (Eds.). Springer Nature Switzerland, Cham, 481–496.