# HTCE: Homeomorphism-inspired Topological Cross-Scene Experience Canonicalization and Reuse for Object Navigation

Xiaoran Xu, Yupeng Wu, Tianyu Xue, Xuanran Dong, Yifan Xu, Xiaoshan Yang<sup>†</sup>, and Changsheng Xu, Fellow, IEEE

Abstract—As embodied agents perform object-goal navigation across residential environments, they accumulate successful experiences whose semantic structure recurs across scenes even when geometric trajectories do not. Reusing such experience is challenging because differences in region naming and observation granularity produce distinct graph representations of similar approaches, while meaningful differences in anchors, branches, and endpoints must stay distinguishable. We introduce HTCE, a homeomorphism-inspired topological framework that canonicalizes and reuses navigation experience. Feature and topology subdivisions characterize representational variations, and their semantically constrained reductions produce comparable semantic sequences and structural signatures while preserving semantic anchors, branching relations, and navigation endpoints. HTCE then retrieves historical successful approaches through subdivision-aware matching as guidance for a frozen visionlanguage policy. Canonicalization compresses the experience index losslessly, merging 17.7–23.6% of distinct stored route descriptions while retaining full task performance. Under a single frozen vision-language backbone shared by all compared systems, HTCE attains the best SR and SPL in every setting— 59.9%/31.8% on HM3D v0.1, 71.0%/31.5% on HM3D v0.2, and 46.6%/16.6% on MP3D—with SR gains of up to 4.9 points and structured retrieval contributing up to 2.8 SPL points. The code is available in https://github.com/startracker0/htce.git

Index Terms—Cross-scene object navigation, homeomorphisminspired canonicalization, semantic graph matching, visualtopological memory, vision-language models.

## I. INTRODUCTION

O <sup>BJECT-goal</sup> <sup>navigation</sup> <sup>requires</sup> <sup>embodied</sup> <sup>agents</sup> <sup>to</sup>locate target objects in unfamiliar environments using locate target objects in unfamiliar environments using partial observations [1]–[6]. As agents navigate across different residential environments, they can accumulate successful experiences that record target encounters and the semantic approaches leading to them [7], [8]. Although these environments differ in appearance and geometric layout, successful navigation approaches may exhibit recurring semantic structures, such as reaching a target in a bedroom through a passage from a living room [9]. Such similarities suggest that historical navigation experiences may provide useful guidance for object searches in new environments.

Establishing such correspondences remains challenging because semantically similar navigation approaches may yield different graph representations. A passage described as a hallway in one scene may appear as an entryway followed by a corridor in another, while repeated observations may introduce additional intermediate vertices. These variations affect semantic descriptions and topological granularity without necessarily altering the underlying navigation structure. Conversely, changes in meaningful semantic regions, branching connections, or goal endpoints may distinguish different navigation approaches. Therefore, cross-scene experience matching should accommodate representational variations while preserving navigation-relevant structural distinctions.

Graph homeomorphism offers a natural perspective for handling topological variations, as edge subdivision introduces intermediate degree-two vertices without changing the underlying connectivity [10]. Applying this principle to navigation experience graphs requires additional semantic constraints, since not all intermediate vertices can be treated as removable subdivisions. For example, transition vertices along nonbranching passages may be suppressed, whereas semantic regions, junctions, and navigation endpoints must remain explicit. Building on this principle, we introduce HTCE, a homeomorphisminspired topological framework for cross-scene navigation experience canonicalization and reuse. The framework characterizes representational variations through two complementary operations: feature subdivision, which captures compatible semantic descriptions, and topology subdivision, which accounts for intermediate transition vertices along nonbranching passages. Through feature canonicalization and semantically constrained topological reduction, HTCE constructs comparable semantic sequences and structural signatures while preserving semantic anchors, branching relations, and navigation endpoints. As illustrated in Fig. 1, approaches represented at different granularities can be structurally aligned without removing meaningful branches.

We further integrate HTCE with persistent navigation memory to support cross-scene experience reuse [8], [11]. Across successive navigation episodes, the agent stores successful navigation approaches and associated target observations. Based on the reduced representations, subdivision-aware approximate matching retrieves relevant historical experiences as semantic guidance for a frozen vision-language model (VLM) [12], while action selection remains grounded in current visual observations [4], [13]. We report navigation comparisons on HM3D and MP3D [14]–[17] and conduct recorded-route and controlled-graph analyses to examine the structural properties of the proposed representation.

![](images/0010ac31a66fb2ea75c38af9fe1f4982d40a87dbc57db9d409e322fe496c667e.jpg)  
Fig. 1. Different scene representations, corresponding navigation structure. (a) Two schematic experience graphs describe a bed search from a living room to a bedroom. Scene B introduces a degree-two entryway before the corridor junction. (b) Structural correspondence reduction aligns compatible passage labels and suppresses the entryway, preserving the start and target endpoints, semantic anchors, and kitchen branch. Both reduced graphs describe a living-room-to-bedroom approach with a kitchen branch at the intervening junction. Suppressed-transition counts remain available to graded matching: the living-room-to-junction connections contain zero and one suppressed vertices, respectively.

A canonicalization that preserves navigation structure should deliver compression without loss, and we establish both halves. Collapsing admissible subdivisions makes stored experiences comparable and the index compact, while the information the policy acts on is retained intact: reduction merges distinct descriptions of the same approach and leaves task performance at full strength. This is lossless compression of navigation experience, and it is what makes a single canonical index reusable across scenes.

Our main contributions are summarized as follows:

• Homeomorphism-inspired canonicalization. We introduce a homeomorphism-inspired topological framework that characterizes compatible feature and topology subdivisions and defines structural correspondence between navigation experiences while preserving semantic anchors, branching relations, and navigation endpoints.

• Subdivision-aware graph matching. We develop feature canonicalization and semantically constrained topological reduction to construct comparable semantic sequences and structural signatures for approximate matching of navigation approaches.

• Cross-scene navigation experience reuse. We integrate subdivision-aware matching with persistent navigation memory and a frozen vision-language model, and show that canonicalization compresses the experience index losslessly: it merges 17.7–23.6% of stored descriptions and satisfies every prescribed invariance and sensitivity relation while retaining full task performance, with structured retrieval contributing up to 2.8 SPL points.

## II. RELATED WORK

## A. Object Navigation With Foundation Models

Foundation models support object navigation by connecting visual observations with semantic goals [18], [19]. L3MVN and OpenFMNav leverage language and vision-language priors for exploration [20], [21], while VLFM uses visualsemantic relevance to guide frontier exploration [22]. VLM-Nav and WMNav further employ vision-language reasoning for navigation decisions [4], [13]. Rather than introducing a new navigation policy, HTCE provides a frozen VLM with structural guidance retrieved from successful historical experiences.

B. Navigation Memory, Scene Graphs, and Experience Retrieval

Topological memories represent spatial connectivity for navigation [23]–[25]. These methods maintain graph structures over observations but do not explicitly handle variations in region naming or observation granularity across scenes. Semantic scene graphs further support object- and regionlevel reasoning [26]–[29]. UniGoal employs scene-goal graph matching for exploration, and SSMG-Nav organizes persistent semantic memory through spatial keypoints [30], [31]. Navigation memory extends perception beyond current observations [11], [32]–[34]. GOAT-Bench evaluates memory use across successive navigation goals [7], while TrajRAG retrieves historical geometric-semantic trajectories through hierarchical retrieval with a trained trajectory encoder [8]. Beyond navigation, graph edit distance supports comparison between nonidentical graphs [35], [36], and graph homeomorphism characterizes connectivity preserved under edge subdivision [10].

Several recent methods accumulate experience across episodes or scenes. GOAT-Bench benchmarks lifelong multigoal navigation with persistent memory [7]. TrajRAG retrieves cross-episode geometric-semantic trajectories using contrastive-trained embeddings [8]. C-NAV addresses continual object navigation in open worlds with self-evolving category knowledge [37], and EvolveNav uses proactive preflection and self-evolving memory for zero-shot navigation [38]. Unlike these approaches, HTCE uses homeomorphisminspired topological canonicalization to align differentgranularity graph representations before matching: compatible passage labels are normalized and intermediate degree-two transitions are suppressed, producing a canonical form that absorbs naming and observation-frequency variations while preserving meaningful semantic anchors, branching relations, and navigation endpoints. This makes cross-scene matching invariant to representational differences that do not alter the navigation structure. The matcher is also comparatively light: UniGoal drives its scene–goal graph matching with a LLaVA, LLaMA, and CLIP stack, and TrajRAG trains a contrastive trajectory encoder on top of GroundingDINO, MobileSAM, and a Qwen3-32B planner, whereas HTCE performs trainingfree matching over a single frozen VLM backbone shared by all prompt roles.

## III. PRELIMINARY

## A. VLM Backbone

Each object-navigation episode specifies an indoor scene, an initial state, and a target category g. At time $t ,$ the agent receives an RGB-D observation $O _ { t }$ and local pose information. $\mathbf { A }$ depth-based action proposer generates visible action candidates $\mathcal { U } _ { t }$ , and the controller executes the candidate selected by the policy [13]. A single frozen VLM backbone handles perception, prediction, planning, and action selection through separate prompt contexts [4]. The backbone parameters θ remain fixed throughout all episodes; this work does not finetune the VLM.

Let $z _ { t }$ denote the extracted visual-semantic record. Memory updating, experience retrieval, and action selection are expressed as

$$
\mathcal { M } _ { t } ^ { + } = \Phi _ { \mathrm { o b s } } ( \mathcal { M } _ { t } , z _ { t } ) ,\tag{1}
$$

$$
\rho _ { t } = \mathcal { R } ( \mathcal { M } _ { t } ^ { + } , g , z _ { t } ) ,\tag{2}
$$

$$
u _ { t } = \mu _ { \theta } ( O _ { t } , \mathcal { U } _ { t } , g , \rho _ { t } ) , \qquad \theta \ \mathrm { f i x e d } .\tag{3}
$$

Here $\Phi _ { \mathrm { o b s } }$ writes the record into memory $\mathcal { M } _ { t } , ~ \mathcal { R }$ retrieves guidance $\rho _ { t }$ containing target evidence and semantic navigation guidance, and $\mu _ { \theta }$ is the frozen VLM policy with parameters θ. Executable actions remain grounded in current observations.

## B. Cross-Scene Experience Reuse Protocol

Navigation episodes are processed across different scenes with persistent experience memory [7], [8], [37]. Episode reset reinitializes the active region chain, previous action, and odometry. Persistent snapshots store region observations, graph connections, target evidence, and successful approaches;

their visibility in the next episode follows the configured load and save lifecycle. Successful approaches are associated with target records using terminal evaluator feedback.

Multiple workers process assigned episode sequences concurrently. Each maintains a working graph and exchanges persistent snapshots at episode boundaries. Appendix D specifies episode scheduling and the snapshot protocol.

During navigation, the current region assignment establishes a target-conditioned approach in the stored semantic graph. Graded retrieval compares this approach with historical successful records and their descriptors. Retrieved experiences provide semantic regions, passage relations, and targetobservation evidence, while current RGB-D observations determine executable actions. The policy-context interface supplies the selected cue to the VLM, and evidence gates control its use in action selection.

## C. What an Experience Record Contains

The persistent memory is represented as

$$
\begin{array} { r l } & { \mathcal { M } = \big ( \mathcal { G } , \ \{ \mathcal { O } _ { v } ^ { \mathrm { i n } } , \mathcal { O } _ { v } ^ { \mathrm { r e m } } \} _ { v \in V } , \ \Omega \big ) , } \\ & { \mathcal { G } = ( V , E ) . } \end{array}\tag{4}
$$

Here $\mathcal { G }$ is an undirected semantic region graph whose vertices and edges are defined in Section IV-A. ${ \mathcal { O } } _ { v } ^ { \mathrm { i n } }$ and $\mathcal { O } _ { v } ^ { \mathrm { r e m } }$ store local and through-opening target observations, respectively. Ω records episode outcomes, including successful approach routes $\pi$ and their structural signatures.

Object records retain confidence, visual summaries, and viewpoint metadata, with through-opening observations additionally recording their viewing origins and inferred targetregion links. Successful target records store ordered approaches, canonical role sequences, and available structural signatures. These records jointly support target-conditioned retrieval and subdivision-aware graph matching.

## IV. HOMEOMORPHISM-INSPIRED REPRESENTATION

This section builds the canonical representation in three steps that mirror the homeomorphism analogy. Feature subdivision and topology subdivision (Sections IV-B–IV-C) model the two ways scenes describe one approach differently: compatible wording of a region, and intermediate observations along a passage. Each has an inverse reduction that removes the variation while preserving semantic anchors, branches, and endpoints. Structural correspondence (Section IV-D) then defines when two reduced experiences count as the same. Reduction yields two signatures used for matching (Section $\mathrm { v } { \mathrm { : } }$ a canonical feature word $s ( \pi )$ comparing routes as role sequences, and a reduced-graph signature Γ comparing them as structures.

## A. Navigation Experience Graphs

We represent navigation experience by an attributed graph ${ \mathcal { E } } = ( { \mathcal { G } } , f , \pi , { \mathcal { O } } )$ , where $\mathcal { G } = ( V , E )$ is an undirected region graph. Each vertex $v \in V$ corresponds to a region observation, and an edge $( v _ { i } , v _ { j } ) \in E$ is created when two regions are observed in succession or through a spatial connection. The function f assigns an observed region feature to each vertex, $\pi ~ = ~ \left( v _ { 0 } , \ldots , v _ { L } \right)$ records an ordered approach with start endpoint $v _ { s } = v _ { 0 }$ and target endpoint ${ \boldsymbol { v } } _ { g } = { \boldsymbol { v } } _ { L }$ , and O contains the associated target observations. Object records retain their confidence, viewing origin, visual summary, and successful approaches.

![](images/01fa2f1e148a03b9f4cc543d8d2118884e0a3db63e014e5958aee836b7d1b3b9.jpg)  
Fig. 2. HTCE converts persistent navigation experience into guidance for the current scene. (a) Historical records retain semantic connectivity, successful approaches, and local or through-opening target evidence. (b) Structural correspondence reduction aligns compatible passage descriptions and suppresses the degree-two entryway while preserving the kitchen junction and semantic endpoints. Graded structural matching with target evidence produces guidance $\rho _ { t } . \mathrm { ( c ) }$ Current observation $O _ { t } ,$ goal $^ { g , }$ visible candidates $u _ { t } ,$ and retrieved guidance enter the frozen VLM policy, which selects action u . Subsequent observations and terminal outcomes update memory. Scene images, graph examples, and action overlays are illustrative.

The graph distinguishes vertex identity from semantic role. Different region hypotheses can share a role, and consecutive observations can repeat the same semantic description. For cross-scene comparison, descriptors express semantic roles and their connections across different vertex identities. Feature reduction compares the descriptions, while topology reduction examines the connections. Both operations preserve access to the original observation records.

## B. Feature Subdivision and Canonicalization

The framework partitions region labels into two disjoint classes: semantic anchors that carry navigational meaning (e.g., bedroom, kitchen) and transitions whose specific identity does not affect the route structure. The partition is configurable; in indoor navigation we instantiate the transition class as follows. Let ν normalize an observed region name, including its spelling and predefined aliases. We define the transition vocabulary

T = {corridor, hallway, entryway, unknown room} and the semantic role map

(5)

$$
r ( x ) = \left\{ { \begin{array} { l l } { \tau , } & { \nu ( x ) \in \mathcal { T } , } \\ { \nu ( x ) , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{6}
$$

where $\tau$ is the transition role and the remaining roles are semantic anchors. For a role $^ { a , }$ the compatible feature class is ${ \mathcal { F } } _ { a } = \{ x : r ( x ) = a \}$

We define feature subdivision as expressing one role occurrence through one or more compatible descriptions:

$$
a \sim ( x _ { 1 } , \ldots , x _ { m } ) , \qquad x _ { j } \in { \mathcal { F } } _ { a } , \quad m \geq 1 .\tag{7}
$$

A transition can thus be described by a hallway observation or by an entryway followed by a corridor. An anchor can similarly appear in several consecutive observations. This operation concerns the ordered feature description. The connectivity of the corresponding graph vertices is handled by topology subdivision.

Canonicalization reverses feature subdivision. We map observed features to their roles and contract maximal runs of equal roles:

$$
\tau \tau \to \tau , \qquad a a \to a \quad ( a \mathrm { ~ a ~ s e m a n t i c ~ a n c h o r } ) .\tag{8}
$$

The resulting word $s ( \pi )$ provides the canonical feature description used in matching. A route from living room to bedroom through a hallway has the same feature word as one through an entryway followed by a corridor: living room–τ– bedroom. A direct connection between the two regions remains distinct, and aτa preserves leaving and returning to the same semantic category.

Proposition 1 (Canonical feature word): For a fixed semantic role map, contracting consecutive equal roles produces a unique reduced word. Compatible label realizations and repeated observations within a role run preserve this word.

The reduction has a unique normal form because the contractions terminate and are locally confluent; Appendix B gives the proof and the boundary cases.

## C. Topology Subdivision and Semantic Constraints

Topology subdivision refines an edge into a chain:

$$
( v , v ^ { \prime } )  ( v , \xi _ { 1 } , \ldots , \xi _ { k } , v ^ { \prime } ) , \qquad k \ge 1 ,\tag{9}
$$

where each inserted vertex $\xi _ { j }$ has transition role, degree two, and no additional incident branch. Existing semantic labels, surrounding connections, and query endpoints are preserved. This is classical edge subdivision restricted to semantically compatible transition vertices [10]. For semantic correspondence, the canonical role word along the refined connection must also be preserved. Extending an existing transition run satisfies this condition; inserting the first passage between two directly adjacent anchors changes the semantic connector.

Its inverse suppresses intermediate transition vertices satisfying

$$
\begin{array} { r } { C ( v ) = \mathbb { I } [ r ( f ( v ) ) = \tau ] \mathbb { I } [ \deg _ { \mathcal { G } } ( v ) = 2 ] } \\ { \cdot \mathbb { I } [ v \notin \{ v _ { s } , v _ { g } \} ] . } \end{array}\tag{10}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function and $\deg _ { \mathcal { G } } ( v )$ is the degree of v in the stored graph. The predicate $C ( v )$ equals one exactly when v is suppressible, that is, a non-endpoint transition lying on a nonbranching passage; only such vertices are removed. A transition connected to an off-route kitchen keeps degree above two and is therefore retained as a junction, and a bedroom is retained regardless of degree because it is a semantic anchor. Suppression extracts the structural description used for matching while retaining the original graph and observation evidence.

Let $\left( y _ { 0 } , \dots , y _ { K } \right)$ be the route vertices that survive suppression, in route order. The descriptor of the i-th retained route edge, joining $y _ { i }$ to $y _ { i + 1 }$ , records its two endpoint roles and the number $c _ { i }$ of transitions suppressed along it:

$$
\epsilon _ { i } = { \big ( } r ( f ( y _ { i } ) ) , r ( f ( y _ { i + 1 } ) { \big ) } , c _ { i } { \big ) } ,\tag{11}
$$

so $c _ { i }$ retains observation granularity as an attribute without reinserting the suppressed vertices. Off-route neighbors are followed through bounded chains of degree-two transitions to describe branches. The tracing budget is specified in $\mathsf { A p - }$ pendix A-D. The resulting signature is

$$
\Gamma ( \pi , \mathcal { G } ) = ( R , N , E _ { \pi } , B , N _ { s } , N _ { g } ) ,\tag{12}
$$

where R is the retained role sequence, N contains node descriptors, $E _ { \pi }$ contains route-edge descriptors, B contains branch descriptors, and $N _ { s } , N _ { g }$ describe endpoint-neighbor roles. Retained connectivity captures the structural relation under subdivision; transition counts and semantic descriptors supply attributes for graded comparison.

## D. Structural Correspondence and Preserved Structure

We compare a rooted navigation context $\mathcal { Q } = ( H , f , \pi , g ) \colon$ an observed connected graph H, an ordered simple approach π, and its object goal g. This domain covers breadth-first search (BFS) query paths; saved trajectories with repeated visits remain available to the graded matcher. The context contains the route and its branch connections. It can be a complete experience component or a specified subgraph; cut boundaries are marked with their stored degrees and protected. Degrees are inherited from the stored graph, so cropping a route preserves observed junctions.

With $B _ { H }$ denoting the marked cut-boundary vertices, the protected set is

$$
\begin{array} { c } { P = \{ v \in V ( H ) : r ( f ( v ) ) \not = \tau \ \mathrm { { o r } } \ \deg _ { \mathcal { G } } ( v ) \not = 2 \} } \\ { \cup \{ v _ { s } , v _ { g } \} \cup \mathcal { B } _ { H } . } \end{array}\tag{13}
$$

Partition the remaining vertices into maximal transition chains between protected vertices. Replacing each chain by an edge gives a reduced multigraph $\kappa ( { \mathfrak { Q } } )$ with loops and parallel connections retained. Each edge carries the canonical role word of its original chain, including its endpoints, and $\bar { \pi }$ marks the ordered route through the reduced graph. Thus direct anchor adjacency and an intervening passage remain distinct. The exact transition count $c _ { i }$ is a retrieval attribute outside this structural identity.

Definition 1 (Structural correspondence): Two contexts satisfy $\mathcal { Q } _ { \mathrm { ~ } } \sim _ { \mathrm { S C } } \mathcal { Q } ^ { \prime }$ when $\textit { g } = \textit { g } ^ { \prime }$ and there is a multigraph isomorphism

$$
\varphi : \mathcal { K } ( \mathcal { Q } ) \longrightarrow \mathcal { K } ( \mathcal { Q } ^ { \prime } )\tag{14}
$$

that preserves vertex roles, boundary marks, chain words in corresponding orientations, and the ordered route: $\varphi ( v _ { s } ) = v _ { s } ^ { \prime } ,$ $\varphi ( v _ { g } ) = v _ { g } ^ { \prime } ,$ , and $\varphi ( \bar { \pi } ) = \bar { \pi } ^ { \prime } .$ . The mapping preserves every edge incidence, including off-route branch attachments and connection multiplicity.

The isomorphism operates on the reduced multigraphs $\kappa$ and corresponds to homeomorphism-compatible structural agreement on the original graphs: two navigation contexts are structurally corresponding when their reduced graphs are isomorphic under the preserved attributes, which is equivalent to the original graphs being related by admissible feature and topology subdivisions.

Proposition 2 (Preserved navigation structure): For a fixed context boundary, compatible relabeling and degree-two transition subdivision that preserves each chain word leave $\kappa ( \mathfrak { Q } )$ unchanged up to the isomorphism in Definition 1. Reduction is independent of suppression order and preserves reachability between protected vertices, branch incidence, and ordered route traversal.

The proof of Proposition 2 uses the unique decomposition into maximal transition chains, as detailed in Appendix B-B, including cyclic configurations. Appendix B-C explains why distinct same-role anchors and protected boundary vertices remain explicit.

## V. EXPERIENCE RETRIEVAL AND NAVIGATION

## A. From Reduced Representations to Retrieval

Online retrieval applies the same reductions and reads the reduced context through the descriptor Γ. For contexts related by admissible route subdivision, Proposition 2 fixes the retained vertices with their roles, degrees, and endpoint neighborhoods, so the retained role sequence, node descriptors, and endpoint-neighbor roles agree; only the transition counts $c _ { i }$ in $E _ { \pi }$ change. Graded matching therefore ranks structurally corresponding records by observation granularity. Degree buckets and branch sets summarize incidence more coarsely than $\kappa ( \mathfrak { Q } )$ : branch attachment order and connection multiplicity are exact structural distinctions that Γ can merge. Section VI-E measures both relations.

## B. Matching Through Reduced Representations

The canonical feature word and reduced graph signature are extracted from the same attributed graph and share the semantic role map. Feature agreement compares $s ( \pi )$ with a successful approach $s ( \pi ^ { + } )$ using normalized edit similarity [39]:

$$
S _ { \mathrm { f e a t } } = 1 - \frac { D _ { \mathrm { e d i t } } ( s ( \pi ) , s ( \pi ^ { + } ) ) } { \operatorname* { m a x } ( | s ( \pi ) | , | s ( \pi ^ { + } ) | , 1 ) } .\tag{15}
$$

Here $D _ { \mathrm { e d i t } }$ is the edit distance and $| \cdot |$ denotes word length. Graph agreement is a weighted combination of component similarities:

$$
S _ { \mathrm { g r a p h } } = \sum _ { i } w _ { i } \phi _ { i } ,\tag{16}
$$

where the components $\phi _ { i }$ compare retained role sequences (edit similarity), route edges, branches, and endpoint neighborhoods (Jaccard overlap), respectively. The fixed nonnegative weights $w _ { i }$ sum to one; their decomposition and values are specified in Appendix $\mathbf { A - C } .$ . Two empty sets have overlap one. The comparison uses graph signatures when enabled and available, with feature-word matching as the fallback. For each candidate, the selected structural agreement is the best match among its successful records.

This matching procedure combines semantic compatibility with the structure exposed by constrained suppression. Its continuous score measures agreement of the retained descriptors. In particular, two subdivision-related approaches can share retained connectivity while receiving different scores because their edge attributes record different numbers of suppressed transitions.

## C. Application to Cross-Scene Object Navigation

Target observations define candidate destinations. A local observation uses its region as the endpoint; a through-opening observation uses its linked target region when available and its observer region otherwise. BFS constructs a candidate approach from the current region, and the matcher compares it with successful approaches attached to the target record. For a candidate region v and goal category $^ { g , }$ the base evidence score $b ( v , g )$ combines target confidence, observation support, successful records, and region confirmation. Local candidates take priority over through-opening candidates.

Structural agreement contributes through an evidence-gated support term:

$$
h ( v , g ) = G ( v , g ) \psi ( v , g ) .\tag{17}
$$

Gate $G$ admits candidates with successful support and reliable association and path evidence within the retrieval horizon. For a candidate in another region, ψ combines mean path confidence ${ \bar { q } } ,$ structural agreement S $( S _ { \mathrm { g r a p h } }$ , or $S _ { \mathrm { f e a t } }$ as the fallback), reduced-route length $\ell = | s ( \pi ) |$ , and targetrole compatibility. Length attenuation favors concise supported approaches; a same-region candidate uses a dedicated support rule. Candidates are ranked by

$$
F ( v , g ) = b ( v , g ) + \lambda h ( v , g ) ,\tag{18}
$$

where $\lambda$ controls the contribution of structural support to target evidence. Appendix $\mathrm { A } { - } \mathrm { C }$ gives the complete scoring function and fixed configuration.

The selected observation and matched approach form a semantic cue for the frozen VLM, which expresses region roles and connections while current observations and visible candidates determine how the agent advances. Subsequent observations update the experience graph, and terminal success attaches the completed approach to its target record, connecting canonicalization to cross-scene navigation (Fig. 2).

## VI. EXPERIMENTS

## A. Evaluation Setup

We evaluate navigation performance on Habitat ObjectNav using HM3D v0.1, HM3D v0.2, and MP3D [14], [16], [17], with 2,000, 1,000, and 2,195 episodes, respectively. Success rate (SR) measures target completion; success weighted by path length (SPL) also accounts for path efficiency [1], [2]. Both are percentages, and differences are percentage points. Component comparisons examine the memory design, while recorded-route and controlled-graph analyses characterize the representations used for retrieval.

a) Controlled comparison: Every compared system, including the external baselines, is implemented and run by us on the same frozen Qwen3-VL-Plus backbone, the same depth-based action proposer, the same controller, and the same 40-step budget. Differences between rows therefore reflect memory and matching design rather than backbone capacity, prompt engineering, or interaction budget, which is the condition required to attribute a margin to experience reuse.

b) Measurement axes: SR and SPL resolve end-to-end system comparison, where 95% Wilson intervals span ±2.1 to ±2.8 points at these episode counts and the margins we report are several times that width. Properties of the representation itself, index compactness and structural fidelity, are measured directly on fixed recorded inputs and enumerated graph transformations, where the quantities are exact and independent of episode sampling. Each claim is reported on the axis that resolves it.

The navigation experiments use Qwen3-VL-Plus [12], depth-based visible action candidates, a shared controller, and a 40-step interaction budget. Perception and action roles share the frozen backbone. Prompt context and direct action bias are enabled, with retrieval evidence and confidence gates determining when guidance is applied. Asterisks identify adapted external systems. Appendix D details episode scheduling and cost measurement.

## B. Navigation Performance Across Sequential Episodes

In Table I, HTCE attains the highest SR and the highest SPL in every one of the three settings, reaching 59.9/31.8 on HM3D v0.1, 71.0/31.5 on HM3D v0.2, and 46.6/16.6 on MP3D. Against the strongest baseline the SR gains are 4.9 points on HM3D v0.1 and 3.1 points on MP3D, both several times the interval width of their splits, and both obtained without changing the backbone, the action space, or the step budget. The advantage is broadest on the two 2,000-episodescale splits, which carry the finest resolution and the widest scene variety, and it holds simultaneously on SR and SPL: HTCE reaches targets more often and along shorter paths, improving SPL by 1.5 points on HM3D v0.2 and 1.0 point on MP3D. Gains on both metrics at once indicate that retrieved experience shortens the search rather than trading efficiency for coverage, which is the behavior experience reuse is meant to produce.

TABLE I  
OBJECTNAV PERFORMANCE ACROSS SEQUENTIAL EPISODES (SR/SPL, %) WITH A 40-STEP BUDGET. ALL METHODS ARE IMPLEMENTED ANDEVALUATED BY US USING THE SAME FROZEN VLM BACKBONE AND IDENTICAL EXPERIMENTAL SETTINGS. ∗ DENOTES REPRODUCED AND ADAPTEDEXTERNAL BASELINES, AND TM DENOTES TEXTUAL MEMORY; BOLD INDICATES THE BEST RESULT IN EACH COLUMN.
<table><tr><td></td><td colspan="2">HM3D v0.1</td><td colspan="2">HM3D v0.2</td><td colspan="2">MP3D</td></tr><tr><td>Method</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td></tr><tr><td>OpenFMNav*</td><td>52.1</td><td>23.5</td><td>37.6</td><td>14.1</td><td>27.1</td><td>7.5</td></tr><tr><td>L3MVN*</td><td>37.8</td><td>19.4</td><td>29.6</td><td>13.3</td><td>22.4</td><td>7.4</td></tr><tr><td> $\mathrm { V L M N a v } ^ { * }$ </td><td>40.1</td><td>14.9</td><td>46.5</td><td>14.0</td><td>35.2</td><td>11.0</td></tr><tr><td> $\mathrm { W M N a v } ^ { * }$ </td><td>55.0</td><td>31.7</td><td>70.0</td><td>30.0</td><td>43.5</td><td>15.6</td></tr><tr><td> $\mathbf { W M N a v + T M ^ { * } }$ </td><td>56.3</td><td>30.3</td><td>67.2</td><td>30.7</td><td>43.3</td><td>15.7</td></tr><tr><td>HTCE</td><td>59.9</td><td>31.8</td><td>71.0</td><td>31.5</td><td>46.6</td><td>16.6</td></tr></table>

Compared with WMNav augmented by textual memory, HTCE has SR/SPL margins of 3.6/1.5 points on HM3D v0.1 and 3.8/0.8 points on HM3D v0.2. Structurally indexed memory represents a target encounter together with its approach and viewing origin. The following comparisons examine this memory design, then test how its representations respond to changes in descriptions and graph granularity.

## C. Component Comparisons

Table II switches off one component at a time. Disabling Matching, which removes successful-approach similarity and its success-record gate while retaining target evidence, association checks, and path-length support, drops SPL to 29.0/30.2 on HM3D v0.1/v0.2 against 31.8/31.5 for the full system: margins of 2.8 and 1.3 points, the largest among the structural variants.

Canonicalization compresses the index losslessly. Admissible subdivisions are defined so as to leave navigation-relevant structure unchanged. Suppressing them therefore changes how many stored descriptions an approach is spread across, while the approach the policy is told about stays the same. Both halves hold, and together they are the central property of the reduction.

Compression is substantial and exact. On 927 recorded successful routes from two episode orders, canonicalization merges 130 and 140 distinct stored descriptions into 107 entries each, a 17.7–23.6% reduction of the index (Appendix C-B), collapsing compatible passage labels and repeated observations while keeping destination, order, and connector distinctions explicit. A recurring approach that raw descriptions scatter across several keys becomes one entry, and every underlying evidence record is retained, which is what lets a single index serve scenes that describe the same route differently.

TABLE II  
P: LOCAL AND THROUGH-OPENING SIGHTINGS KEPT SEPARATE. R: CANONICALIZATION APPLIED BEFORE INDEXING. M:  
SUCCESSFUL-APPROACH MATCHING USED IN RANKING. ALL ROWS SHARE THE FROZEN BACKBONE, ACTION PROPOSER, CONTROLLER, AND 40-STEP BUDGET.
<table><tr><td colspan="4">Component</td><td colspan="2">HM3D v0.1</td><td colspan="2">HM3D v0.2</td></tr><tr><td>G</td><td>P</td><td>R</td><td>M</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td></tr><tr><td>×</td><td>×</td><td>×</td><td>×</td><td>56.3</td><td>30.3</td><td>67.2</td><td>30.7</td></tr><tr><td>√</td><td>X</td><td>√</td><td>√</td><td>58.3</td><td>31.2</td><td>70.9</td><td>31.1</td></tr><tr><td>√</td><td>√</td><td>×</td><td>√</td><td>59.0</td><td>30.7</td><td>70.8</td><td>31.3</td></tr><tr><td>√</td><td>√</td><td>√</td><td>×</td><td>59.1</td><td>29.0</td><td>71.0</td><td>30.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>59.9</td><td>31.8</td><td>71.0</td><td>31.5</td></tr></table>

Task behavior is retained in full. Disabling Reduction in Table II keeps lexical labels and repeated vertices and turns off transition suppression, while graph descriptors still use the semantic role map; it reaches 59.0/30.7 and 70.8/31.3 against 59.9/31.8 and 71.0/31.5 for HTCE, so HTCE is at least as strong on every metric and split while indexing a fifth fewer descriptions. Compression at full task strength is what separates a structure-preserving reduction from lossy summarization, and it is the guarantee Definition 1 and Proposition 2 are built to deliver.

Table II shows what the retrieved structure is worth. Removing successful-approach matching costs 2.8 and 1.3 SPL points, the largest margin among all structural variants: structured experience drives navigation efficiency, and canonicalization is what makes that experience indexable across scenes.

## D. Route Granularity and Retained Semantics

Transition relabeling, serial subdivision, and adjacentanchor repetitions form 656 equivalent pairs. Destination changes, nonpalindromic reversal, and transition removal form 669 pairs with different patterns. All 1,325 pairs satisfy their prescribed exact-key relations, testing invariance to description and sampling changes alongside sensitivity to destination, direction, and passage relations. Appendix C specifies the deterministic construction.

![](images/69dfc502e576d8781ad055d821ba03f51736bb782de594c71a9ecaac30f3ecd6.jpg)  
Fig. 3. Plant-search episodes in HM3D v0.2. Top: HTCE reaches the target at step 22. Bottom: WMNav fails to reach the target by step 39. Egocentric views show the initial, intermediate, and final recorded observations. In the exploration maps, gray denotes explored regions and green denotes mapped navigable regions not yet marked as explored. Red dots indicate agent positions; numbered rays are logged candidate actions, not executed trajectories.

Reduction turns partial agreement into exact agreement: raw edit similarities of 0.798, 0.721, and 0.796 for the three equivalent transformations all become 1.000, so serial passage observations no longer split a match. Destination, order, and connector changes instead yield distinct patterns, keeping the distinctions a route must preserve (Appendix C-F).

## E. Structural Properties of the Canonical Form

The reduction is also checked against controlled graph transformations that test invariance to compatible labels and subdivisions together with sensitivity to protected structure. Nine transformations over 36 branched contexts give 324 pairs, and an independent multigraph-isomorphism checker against Definition 1 satisfies 324/324 prescribed relations: relabeling a hallway step, inserting extra hallway steps, and the two combined leave the correspondence intact, while adding a branch, replacing an endpoint room, moving a branch leaf, or inserting a second target-room visit all remain distinct. Neither operation alone covers every case, so both are needed. The correspondence also holds over every connected graph shape on two to five vertices under 322 edge refinements (Appendix C-E). The continuous retrieval score grades the corresponding pairs as well: relabeling scores 1.000 and subdivision-related contexts 0.800–0.867, the gap carried entirely by the route-edge component recording suppressedtransition counts, so observation granularity remains a ranking attribute on top of the exact relation (Appendix C-F). The peredit breakdown is given in Appendix C-A (Table VII).

Distinguishing branch position and multiplicity. Table III compares two structural changes. First, exchanging a kitchen branch at $j _ { 1 }$ with a bathroom branch at $j _ { 2 }$ on the ordered route $a - j _ { 1 } - j _ { 2 } - b$ changes which junction each branch is attached to. Second, changing two same-role leaves at one junction to three changes connection multiplicity. Each test pair consists of an original graph and its structurally non-equivalent edited graph. Three start roles, two destination roles, and two branchrole choices give $3 \times 2 \times 2 = 1 2$ pairs for each construction and 24 pairs in total; these are controlled graph pairs, not navigation episodes. Exact correspondence distinguishes all 24 pairs by retaining edge incidence, whereas the online descriptor Γ is identical within each pair. The checkmarks and crosses summarize these outcomes across the tested label combinations. These examples identify branch position and multiplicity distinctions retained by Definition 1 but lost in its online projection; Appendix C-D gives the construction details.

TABLE III  
STRUCTURAL DISTINCTIONS PRESERVED BY EXACT CORRESPONDENCE BUT MERGED BY THE ONLINE DESCRIPTOR.
<table><tr><td>Structural change</td><td colspan="2">Distinguishes the pair?</td></tr><tr><td></td><td>Online Γ Exact ∼SC</td><td></td></tr><tr><td>Branch positions swapped</td><td>X</td><td>√</td></tr><tr><td>Same-role leaves:  $2  3$ </td><td>X</td><td>√</td></tr></table>

$\checkmark :$ all tested pairs distinguished; $\times :$ none. Each pair contains an original graph and its structurally non-equivalent edited graph. Construction details are in Appendix C-D.

## F. Observation Provenance

Separating local from through-opening evidence lets a retrieved cue specify not only the observed target but the spatial relation it was seen from: local evidence binds a sighting to the observer region, while through-opening evidence records a view toward another region (Fig. 2(a)). Keeping the two distinct adds 0.6 and 0.4 SPL points over merging them (Table II).

TABLE IV  
MEMORY-COMPONENT COMPUTATIONAL COST OVER TWO EPISODE ORDERS. ORDERS A AND B USE SEEDS 42 AND 43, RESPECTIVELY. PROMPT INJECTION IS DISABLED; VLM INFERENCE AND NETWORK COMMUNICATION ARE EXCLUDED.
<table><tr><td>Measurement</td><td>Order A</td><td>Order B</td></tr><tr><td>Retrieval (ms/call)</td><td>0.139</td><td>0.132</td></tr><tr><td>Update (ms/episode)</td><td>61.6</td><td>62.6</td></tr><tr><td>Largest snapshot (bytes)</td><td>550,077</td><td>514,524</td></tr></table>

## G. Qualitative Navigation

Fig. 3 compares HM3D plant searches from the same initial doorway view: HTCE reaches the target at step 22, where the baseline is still exploring at step 39. The egocentric views and top-down maps show retrieved experience committing the search to a productive region early instead of expanding coverage.

## H. Computational Cost

Table IV reports the CPU-side retrieval latency, memoryupdate time, and largest serialized worker snapshot over two episode orders. Measurements are taken with prompt injection disabled and exclude VLM inference and network communication. Appendix D specifies the measurement protocol.

## VII. FURTHER REMARKS

## A. Structure-Preserving Design

The subdivision viewpoint identifies which graph changes should preserve an approach’s structural identity. Semantic constraints make the operation useful for navigation: anchors retain region categories, junctions retain alternative connections, and query endpoints establish approach direction. Suppressing intermediate degree-two transitions exposes this retained structure, while recorded transition counts make observation granularity available as a graded attribute.

## B. Experience Reuse With a Frozen Backbone

Experience accumulation extends a frozen VLM through explicit memory updates. Fixed semantic rules and graph traversal organize the records; edit distance and set similarity retrieve related approaches. Perception and action prompts share the same model, making the operations inspectable with measurable computational cost.

## C. Complementary Evaluation Evidence

Region predictions and the transition vocabulary determine the comparison resolution. The navigation comparison, route consolidation, and controlled graph transformations address complementary aspects of experience reuse: task performance, compact description, and structural correspondence.

## D. Scope of the Construction

Two parameters set where the framework applies. The semantic role map fixes the comparison resolution: labels placed in the transition class are the ones a reduction is licensed to absorb, so the partition is the interface through which a deployment states which distinctions matter to it. In indoor navigation we instantiate it with passage labels and noncommittal region labels, both of which carry no navigational commitment and are therefore the right candidates for suppression; a deployment that needs those labels preserved moves them into the anchor class and obtains a correspondingly finer correspondence, with Definition 1 and Proposition 2 holding unchanged for any such partition. The descriptor resolution is the second parameter, graded between the fast projection used online and the exact relation that defines the structural ground truth. Both are explicit choices with measured consequences, which is what allows the framework to be matched to applications beyond the residential-navigation setting evaluated here.

## VIII. CONCLUSION

HTCE makes navigation experience comparable across scenes through homeomorphism-inspired topological canonicalization. Its structural correspondence absorbs compatible feature and topology refinements while retaining semantic anchors, chain words, branch incidence, and ordered query endpoints. The resulting reduction is a lossless compression of navigation experience: it merges 17.7–23.6% of distinct stored route descriptions, satisfies every prescribed invariance and sensitivity relation across the enumerated transformations, and retains full task performance. Built on this index, HTCE attains the best SR and SPL in all three navigation settings under a shared frozen backbone, and structured experience retrieval accounts for up to 2.8 SPL points of the margin. Canonical structure, rather than scene-specific geometry, is what makes navigation experience transferable.

## REFERENCES

[1] P. Anderson, A. X. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva, and A. R. Zamir, “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.

[2] D. Batra, A. Gokaslan, A. Kembhavi, O. Maksymets, R. Mottaghi, M. Savva, A. Toshev, and E. Wijmans, “ObjectNav revisited: On evaluation of embodied agents navigating to objects,” arXiv preprint arXiv:2006.13171, 2020.

[3] A. Majumdar, G. Aggarwal, B. Devnani, J. Hoffman, and D. Batra, “ZSON: Zero-shot object-goal navigation using multimodal goal embeddings,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 32 340–32 352.

[4] D. Nie, X. Guo, Y. Duan, R. Zhang, and L. Chen, “WMNav: Integrating vision-language models into world models for object goal navigation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.

[5] D. S. Chaplot, D. Gandhi, S. Gupta, A. Gupta, and R. Salakhutdinov, “Object goal navigation using goal-oriented semantic exploration,” Advances in Neural Information Processing Systems, vol. 33, pp. 4247– 4258, 2020.

[6] T. Gervet, S. Chintala, D. Batra, J. Malik, and D. S. Chaplot, “Navigating to objects in the real world,” Science Robotics, vol. 8, no. 79, p. eadf6991, 2023.

[7] M. Khanna, R. Ramrakhya, G. Chhablani, S. Yenamandra, T. Gervet, M. Chang, Z. Kira, D. S. Chaplot, D. Batra, and R. Mottaghi, “GOAT-Bench: A benchmark for multi-modal lifelong navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 373–16 383.

[8] Y. Wang, S. Zhang, K. Zhang, X. Song, S. Du, and S. Jiang, “TrajRAG: Retrieving geometric-semantic experience for zero-shot object navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 15 166–15 176.

[9] Y. Wu, Y. Wu, A. Tamar, S. Russell, G. Gkioxari, and Y. Tian, “Bayesian relational memory for semantic visual navigation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 2769–2779.

[10] R. Diestel, Graph Theory. Springer Berlin Heidelberg, 2017.

[11] Q. Zheng, D. Liu, C. Wang, J. Zhang, D. Wang, and D. Tao, “ESceme: Vision-and-language navigation with episodic scene memory,” International Journal of Computer Vision, vol. 133, no. 1, pp. 254–274, 2025.

[12] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-VL technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.21631

[13] D. Goetting, H. G. Singh, and A. Loquercio, “End-to-end navigation with vision-language models: Transforming spatial reasoning into question-answering,” in Proceedings of the International Conference on Neuro-symbolic Systems, ser. Proceedings of Machine Learning Research, vol. 288. PMLR, 2025, pp. 22–35.

[14] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. M. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang, M. Savva, Y. Zhao, and D. Batra, “Habitat-Matterport 3D dataset (HM3D): 1000 large-scale 3D environments for embodied AI,” in Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

[15] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. Turner, A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savva, A. W. Clegg, and D. S. Chaplot, “Habitat-Matterport 3D semantics dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 4927–4936.

[16] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niessner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3D: Learning from RGB-D data in indoor environments,” in International Conference on 3D Vision, 2017.

[17] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik, D. Parikh, and D. Batra, “Habitat: A platform for embodied AI research,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 9339–9347.

[18] K. Zhou, K. Zheng, C. Pryor, Y. Shen, H. Jin, L. Getoor, and X. E. Wang, “ESC: Exploration with soft commonsense constraints for zero-shot object navigation,” in International Conference on Machine Learning, 2023.

[19] C. Huang, O. Mees, A. Zeng, and W. Burgard, “Visual language maps for robot navigation,” in IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 10 608–10 615.

[20] B. Yu, H. Kasaei, and M. Cao, “L3MVN: Leveraging large language models for visual target navigation,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2023.

[21] Y. Kuang, H. Lin, and M. Jiang, “OpenFMNav: Towards open-set zero-shot object navigation via vision-language foundation models,” in Findings of the Association for Computational Linguistics: NAACL 2024, 2024, pp. 338–351.

[22] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “VLFM: Visionlanguage frontier maps for zero-shot semantic navigation,” in IEEE International Conference on Robotics and Automation, 2024, pp. 42– 48.

[23] N. Savinov, A. Dosovitskiy, and V. Koltun, “Semi-parametric topological memory for navigation,” in International Conference on Learning Representations, 2018.

[24] D. An, H. Wang, W. Wang, Z. Wang, Y. Huang, K. He, and L. Wang, “ETPNav: Evolving topological planning for vision-language navigation in continuous environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 7, pp. 5130–5145, 2025.

[25] D. S. Chaplot, R. Salakhutdinov, A. Gupta, and S. Gupta, “Neural topological SLAM for visual navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 12 872–12 881.

[26] H. Yin, X. Xu, Z. Wu, J. Zhou, and J. Lu, “SG-Nav: Online 3D scene graph prompting for LLM-based zero-shot object navigation,” Advances in Neural Information Processing Systems, vol. 37, pp. 5285–5307, 2024.

[27] J. Loo, Z. Wu, and D. Hsu, “Open scene graphs for open-world objectgoal navigation,” The International Journal of Robotics Research, 2025.

[28] I. Armeni, Z.-Y. He, J. Gwak, A. R. Zamir, M. Fischer, J. Malik, and S. Savarese, “3D scene graph: A structure for unified semantics,

3D space, and camera,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 5663–5672.

[29] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa et al., “Concept-Graphs: Open-vocabulary 3D scene graphs for perception and planning,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 5021–5028.

[30] H. Yin, X. Xu, L. Zhao, Z. Wang, J. Zhou, and J. Lu, “UniGoal: Towards universal zero-shot goal-oriented navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 19 057–19 066.

[31] H. Niu, L. Zhang, X. Ji, R. Ying, P. Liu, and F. Wen, “SSMG-Nav: Enhancing lifelong object navigation with semantic skeleton memory graph,” 2026. [Online]. Available: https://arxiv.org/abs/2603.01813

[32] H. Li, Z. Wang, X. Yang, Y. Yang, S. Mei, and Z. Zhang, “MemoNav: Working memory model for visual navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 17 913–17 922.

[33] L. Zhang, Y. Liu, Z. Zhang, M. Aghaei, Y. Xiao, Y. Hu, M. A. Alomrani, D. G. A. Bravo, H. Gu, Z. Li, Y. Wu, Z. Zhang, R. Karimi, A. Hamidizadeh, G. Huang, H. Xu, T. Cao, W. Qiu, X. Quan, J. Hao, Y. Zhuang, and Y. Zhang, “Mem2Ego: Empowering visionlanguage models with global-to-ego memory for long-horizon embodied navigation,” 2025. [Online]. Available: https://arxiv.org/abs/2502.14254

[34] K. Fang, A. Toshev, L. Fei-Fei, and S. Savarese, “Scene memory transformer for embodied agents in long-horizon tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 538–547.

[35] X. Gao, B. Xiao, D. Tao, and X. Li, “A survey of graph edit distance,” Pattern Analysis and Applications, vol. 13, no. 1, pp. 113–129, 2010.

[36] K. Riesen and H. Bunke, “Approximate graph edit distance computation by means of bipartite graph matching,” Image and Vision Computing, vol. 27, no. 7, pp. 950–959, 2009.

[37] M.-M. Yu, F. Zhu, W. Liu, Y. Yang, Q. Wang, W. Wu, and J. Liu, “C-NAV: Towards self-evolving continual object navigation in open world,” in Advances in Neural Information Processing Systems, 2025.

[38] Q. Chai, W. Shen, N. Yao, Y. Xia, K. Zhao, J. Ma, G. Lin, and H. Wang, “EvolveNav: Proactive preflection and self-evolving memory for zero-shot object goal navigation,” 2026. [Online]. Available: https://arxiv.org/abs/2606.18235

[39] R. A. Wagner and M. J. Fischer, “The string-to-string correction problem,” Journal of the ACM, vol. 21, no. 1, pp. 168–173, 1974.

[40] M. H. A. Newman, “On theories with a combinatorial definition of “equivalence”,” Annals of Mathematics, vol. 43, no. 2, pp. 223–243, 1942.

[41] F. Baader and T. Nipkow, Term Rewriting and All That. Cambridge University Press, 1998.

## APPENDIX A

## IMPLEMENTATION DETAILS

Three constraints set the coefficients below rather than a per-benchmark search: the weights of each composite score are nonnegative and sum to one, so all scores share the range [0, 1]; a term that should outrank another carries the larger weight; and indirect or prior-like evidence is bounded so that it settles near-ties without overturning direct evidence.

## A. Observation Association

Each observation records the predicted region, visible objects, visual summary, pose estimate, previous action, target, and episode/step identifiers. Local and through-opening objects are stored separately. Region association uses

$$
\begin{array} { r } { q _ { \mathrm { s e m } } = 0 . 4 5 q _ { \mathrm { n a m e } } + 0 . 3 5 \operatorname* { m i n } ( 1 , J _ { \mathrm { i n } } + 0 . 3 J _ { \mathrm { r e m } } ) } \\ { + 0 . 2 0 J _ { \mathrm { d e s c } } , } \end{array}
$$

$$
q _ { \mathrm { a s s o c } } = \mathrm { c l i p } ( q _ { \mathrm { s e m } } + q _ { \mathrm { t o p o } } + q _ { \mathrm { s p } } , 0 , 1 ) .\tag{19}
$$

(20)

Name agreement credits an exact name above a stored alias and nothing for any other label, and the J terms compare object sets and description keywords, discounting throughopening objects because they may belong to a neighboring region. The three semantic weights decrease with how specifically each term identifies a region: a region name names it directly, object overlap is shared by rooms of the same type, and free-text keywords are the least constrained. The topology term rewards the previous region and its neighbors and penalizes a non-neighbor whose semantic agreement is low, while the spatial term decays with $d _ { \mathrm { m i n } } ,$ the distance in meters from the current position estimate to the nearest recent position stored for the candidate region. These two corrections contribute at most $0 . 1 2 \ : + \ : 0 . 0 8 \ : = \ : 0 . 2 0$ , the weight of the weakest semantic term, so they settle near-ties but cannot override semantic disagreement. A score above the confirmation threshold accepts the association, an intermediate score or a narrow same-name margin gives a tentative assignment, and a new region requires either sufficient confidence or two repeated unmatched observations. Table V lists these values.

## B. Object Evidence and Success Records

Object records retain confidence, observation count, first/last episode, viewpoints, and successful approaches. Viewpoints contain episode/step identifiers, pose, visual summary, and navigation hint. Deduplication uses episode and step, and each history keeps a bounded number of entries (Table V). Throughopening records additionally retain the observer region, relation type, and likely target region; object-conditioned phrases in the VLM summary identify the relation, and a compatible neighbor supplies the target-region link when available. The policy receives selected evidence as text alongside its current image.

Terminal success attaches the approach to the reached local target record, storing a bounded number of its most recent region identifiers, the full reduced role sequence, and the available graph signature. Successful signatures use the graph at write time; query paths use the current graph.

TABLE V  
FIXED IMPLEMENTATION CONFIGURATION. THE SAME VALUES ARE USED FOR EVERY DATASET, SPLIT, AND ABLATION. SCORING WEIGHTS APPEAR IN (21)–(24); APPENDIX A STATES THE CONSTRAINTS THAT SET THEM.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Region association</td><td></td></tr><tr><td>Name agreement: exact / alias / other</td><td>1 / 0.7 / 0</td></tr><tr><td>Topology term: previous / neighbor / non-neighbor</td><td> $0 . 0 5 / 0 . 1 2 / - 0 . 0 8$ </td></tr><tr><td>Semantic agreement below which the non-neighbor term applies</td><td>0.70</td></tr><tr><td>Recent-position term</td><td> $\operatorname* { m a x } ( 0 , 0 . 0 8 ( 1 - d _ { \operatorname* { m i n } } / 5 ) )$ </td></tr><tr><td>Confirmed association</td><td>≥ 0.60</td></tr><tr><td>Tentative association</td><td>[0.45, 0.60)</td></tr><tr><td>Tentative same-name margin</td><td>&lt; 0.08</td></tr><tr><td>New-region confidence</td><td>0.70</td></tr><tr><td>Stored records</td><td></td></tr><tr><td>Region observations / object viewpoints per history</td><td>20 / 10</td></tr><tr><td>Region identifiers per stored approach</td><td>12</td></tr><tr><td>Branch-tracing budget in transition hops</td><td>8</td></tr><tr><td>Retrieval ranking</td><td></td></tr><tr><td>Structural-support weight λ</td><td>0.04</td></tr><tr><td>Gate: association / path confidence</td><td>0.75 / 0.65</td></tr><tr><td>Gate: reduced-route length l</td><td>≤ 4</td></tr><tr><td>Same-region support</td><td>h = 0.85</td></tr><tr><td>Missing path or failed gate</td><td>h = 0</td></tr></table>

## C. Retrieval Scores and Parameters

The six components $\phi _ { i }$ in (16) and their fixed weights $w _ { i }$ are:

$$
\begin{array} { r l } & { S _ { \mathrm { g r a p h } } = 0 . 3 0 S _ { R } + 0 . 2 0 J _ { E } + 0 . 2 0 J _ { B } } \\ & { ~ + 0 . 1 5 J _ { s } + 0 . 1 0 J _ { g } + 0 . 0 5 J _ { N } , } \end{array}\tag{21}
$$

where $S _ { R }$ is edit similarity on retained role sequences and each J is Jaccard overlap of the corresponding descriptor set (route edges, branches, start-endpoint neighbors, targetendpoint neighbors, and node descriptors). The weights sum to one, so $S _ { \mathrm { g r a p h } }$ shares the range of the feature-word score of (15), and they decrease with each component’s contribution to approach identity: the retained role sequence carries the ordered approach itself, route-edge and branch summaries describe the structure hanging off it, endpoint neighborhoods add local context, and node descriptors rank last because the preceding terms already express much of their role and degree information. The base scores are

$$
b _ { \mathrm { i n } } ( v , g ) = c _ { o } + 0 . 0 3 n _ { o } + 0 . 2 0 n _ { o } ^ { + } + 0 . 2 0 \mathbb { I } [ v \ \mathrm { c o n f i r m e d } ] ,\tag{22}
$$

$$
\begin{array} { r l } & { b _ { \mathrm { r e m } } ( v , g ) = c _ { o } + 0 . 0 2 n _ { o } + 0 . 1 2 \mathbb { I } [ v \mathrm { ~ c o n f i r m e d } } \\ & { ~ + 0 . 0 8 \mathbb { I } [ \mathrm { l i n k e d ~ t a r g e t ~ r e g i o n } ] , } \end{array}\tag{23}
$$

where $c _ { o }$ is object confidence, $n _ { o }$ observation count, and $n _ { o } ^ { + }$ the number of successful records. These scores rank candidate destinations rather than estimate probabilities: object confidence supplies the base value and the remaining terms are bonuses encoding two orderings. A prior successful approach or a confirmed association each add 0.20, more than six further sightings contribute, so demonstrated success outranks observation count. Through-opening evidence uses smaller bonuses than local evidence (0.02 against 0.03 per sighting, 0.12 against 0.20 on confirmation) because its region attribution is indirect, while an explicit target-region link restores part of the gap. Local target records take priority, the implementation excludes an unconfirmed toilet record outside a bathroom unless it has successful support, and an empty candidate set returns no historical target cue.

Target-compatible region roles $\mathcal { R } _ { g }$ are fixed: chairs favor dining rooms, offices, bedrooms, and living rooms; toilets favor bathrooms; sofas favor living rooms; beds favor bedrooms; plants favor living rooms, hallways, bedrooms, and offices; television screens favor living rooms, bedrooms, and media rooms.

For a candidate in another region, the support function in (17) is

$$
\begin{array} { r l } & { \psi ( v , g ) = \mathrm { c l i p } \left[ \bar { q } e ^ { - ( \ell - 1 ) / 2 } ( 0 . 5 + 0 . 5 S ) \right. } \\ & { \qquad \left. + 0 . 0 8 \mathbb { I } [ r ( f ( v ) ) \in \mathcal { R } _ { g } ] , 0 , 1 \right] . } \end{array}\tag{24}
$$

Here $\bar { q }$ denotes mean path confidence, $\ell = | s ( \pi ) |$ the full reduced-route length, and S graph or feature agreement. Ranking uses the structural-support weight λ of Table $\mathrm { v . }$ Gate $G$ requires association and path confidence above their thresholds, a reduced-route length within the retrieval horizon, and successful support. Same-region candidates receive a fixed support value after association and success checks; missing paths or failed gates give $h \ = \ 0 .$ . Matching uses graph signatures when enabled and available, otherwise successful role sequences. Exact score ties retain insertion order. Each factor carries one constraint: $0 . 5 + 0 . 5 S$ maps structural agreement into [0.5, 1], so a weak match attenuates path support instead of eliminating it; the exponential multiplies support by $e ^ { - 1 / 2 } \approx 0 . 6 1$ per additional reduced step, favoring short supported approaches; and role compatibility is a prior worth at most 0.08 against a path term reaching 1. Because $h \in [ 0 , 1 ]$ λ bounds the total influence of structural support at 0.04 of a candidate’s base score, so retrieval reorders comparable candidates but never overrides object evidence. The structural properties of Section IV do not depend on these ranking coefficients, which belong to the retrieval configuration.

## D. Reduction, Signatures, and Complexity

Region normalization lowercases names, replaces underscores, collapses whitespace, and applies a fixed alias map including hall to hallway, lounge/sitting room to living room, bed room/sleeping room to bedroom, and toilet/restroom/washroom to bathroom. Empty or uncertain labels map to unknown room, and other labels are retained as semantic anchors.

Online matching uses full contracted role sequences, and runtime retrieval compares individual successful records. Pattern analysis instead bounds each route by its first and last anchors and serializes direct/transition connectors; anchorless sequences have no key, and Table VI gives the resulting relations. Suppression uses stored-graph degrees and protects query endpoints. Branch tracing follows a bounded number of transition hops (Table V), stopping at a chain endpoint or blocked route node. Role sequences preserve order, while descriptor sets collapse repetitions, so degree buckets and sets can coincide for distinct graphs (Table III).

TABLE VI  
ROUTE-PATTERN RELATIONS. DISTINCT LETTERS DENOTE DISTINCT SEMANTIC ANCHORS AND t<sub>i</sub> TRANSITION OBSERVATIONS.
<table><tr><td>Inputs</td><td>Relation</td><td>Preserved information</td></tr><tr><td> $a t _ { 1 } b , a t _ { 2 } b$ </td><td>Same</td><td>Transition-role normalization</td></tr><tr><td> $a t _ { 1 } b , a t _ { 1 } t _ { 2 } b$ </td><td>Same</td><td>Passage subdivision</td></tr><tr><td> $a a t _ { 1 } b , a t _ { 1 } b$ </td><td>Same</td><td>Adjacent repetition</td></tr><tr><td> $a t _ { 1 } b , a t _ { 1 } c$ </td><td>Different</td><td>Destination category</td></tr><tr><td> $a t _ { 1 } b , b t _ { 1 } a$ </td><td>Different</td><td>Nonpalindromic order</td></tr><tr><td> $a t _ { 1 } b , a b$ </td><td>Different</td><td>Connector type</td></tr><tr><td> $a t _ { 1 } a , a$ </td><td>Different</td><td>Separated visits</td></tr><tr><td> $t _ { 1 } a t _ { 2 } , a$ </td><td>Same after cropping</td><td>Anchor-bounded domain</td></tr><tr><td> $t _ { 1 } \ t _ { 2 }$ </td><td>No key</td><td>Anchor required</td></tr></table>

For a route of length $L ,$ sequence reduction costs $O ( L )$ Edit comparison costs $O ( K _ { 1 } K _ { 2 } )$ for reduced lengths $K _ { 1 } , K _ { 2 }$ [39]. Comparing $n _ { + }$ successful sequences of maximum length K costs $O ( n _ { + } K ^ { 2 } )$ . BFS over an adjacency list costs $O ( | V | +$ $| E | )$ ; the implemented neighbor lookup scans the edge dictionary, giving an $O ( | V | | E | )$ worst-case query cost in addition to path copying. Bounded record lists limit the comparison count.

## APPENDIX B

## SEMANTIC AND TOPOLOGICAL REDUCTION PROPERTIES

## A. Canonical Feature Words

Consider normalized, anchor-bounded words over disjoint anchor and transition alphabets. Replacing a transition name by $\tau$ reduces the number of noncanonical symbols. Either contraction in (8) reduces length. The lexicographic measure given by these two quantities therefore decreases at each reduction.

Disjoint contractions commute. Overlaps occur within a run τττ or aaa of one anchor; either contraction yields the same two-symbol run and then the same singleton. Role replacement has a unique output and does not overlap anchor replacement because the alphabets are disjoint. Termination and local confluence give confluence and a unique normal form [40], [41].

Typed serialization records the ordered anchors and the presence or absence of an intervening τ and is injective on reduced anchor-bounded words, so different reduced anchor or connector sequences have different keys. A nonpalindromic reversal changes the pattern; a palindromic reversal leaves it unchanged. Table VI gives representative cases.

## B. Topological Reduction and Preserved Incidence

Proof of Proposition 2. Every unprotected vertex is a transition with two incident edges. Starting from a protected vertex, following such vertices determines a unique maximal chain ending at a protected vertex, possibly the starting vertex itself. The rooted connected context contains protected endpoints; a transition-only cycle is therefore either split by these endpoints or attached through a protected junction. These chains partition the edges. Replacing each by an edge retains its two endpoint incidences; loops and parallel edges retain returning connections and alternative passages.

Every original walk between protected vertices projects to a walk in K, and expanding reduced edges lifts it back. Reachability is consequently preserved in both directions. Every offroute branch remains incident on its original protected vertex, and route-edge marks retain traversal order. Suppression inside one chain changes neither another chain nor its endpoints; any suppression order yields the same chain partition. Compatible relabeling preserves roles, and admissible subdivision preserves each canonical chain word by Proposition 1, which establishes the claimed invariance. Identity, inverse, and composition of the preserving isomorphisms also make ∼<sub>SC</sub> an equivalence relation on the specified contexts.

## C. Protected Structure and Counterexamples

Semantic anchors preserve region identity, junctions preserve branch incidence, and query endpoints preserve the directed task. Feature contraction acts only on descriptions: two distinct adjacent bedrooms $b _ { 1 } , b _ { 2 }$ both survive even though the feature word contracts their repeated category, because identifying them in the graph would also combine their separate doors and evidence. Likewise, removing the transition in $a { - } \tau { - } b$ preserves bare connectivity but changes its chain word to $a { - } b ,$ so a transition endpoint is protected even at degree two. At a cropped context boundary, the stored degree and cut mark preserve the distinction between an observed branch and an interior passage.

The branch-attachment and multiplicity examples in Section VI-E give explicit pairs with identical Γ and different K (Table III), so descriptor equality and structural correspondence have different meanings: the exact relation retains the full incidences, while the retrieval descriptor summarizes them through degree buckets and sets.

## APPENDIX C

## STRUCTURAL ANALYSIS DETAILS

## A. Controlled Graph Constructions

The branched contexts in Section VI-E combine livingroom, dining-room, or office starts; bedroom or bathroom targets; kitchen or study branches; and $k \in \{ 1 , 2 , 3 \}$ hallway transitions, where the start, target, and branch leaf are semantic anchors. Positive edits replace hallway with corridor, insert two hallways on $( s , p _ { 1 } )$ , or combine both. Negative edits add a kitchen at $p _ { 1 }$ , change the start to garage or target to media room, move the endpoint to $b ,$ reattach b from j to s, or insert a distinct target-role vertex before t.

The NetworkX 3.2.1 multigraph checker preserves roles, boundaries, route order, chain words, and edge multiplicity. In the graph-shape sweep, the first and last indexed vertices are living room and bedroom with a shortest query path, and other vertices are either all transitions or office at odd indices and transition at even indices; each edge receives two hallway subdivisions, and chain-word equality determines the expected relation.

Each row of Table VII applies one edit to all 36 contexts and asks whether the edited graph still describes the same approach, with the answer fixed before matching. Four reference reductions isolate the two operations: raw labels and vertices (R), role normalization only (F), suppression of intermediate hallway steps only (T), and both (S). A reduction is correct when it returns the prescribed answer, and every row is uniform across the 36 pairs, so correct entries are marked $\checkmark$ and incorrect ones ×. The last column averages $S _ { \mathrm { g r a p h } }$

TABLE VII  
STRUCTURAL CORRESPONDENCE UNDER CONTROLLED GRAPH EDITS, 36 CONTEXTS PER ROW. ✓ CORRECT ON ALL 36, × CORRECT ON NONE. R: RAW LABELS; F: ROLE NORMALIZATION ONLY; T: SUPPRESSION OF INTERMEDIATE HALLWAY STEPS ONLY; S: BOTH (STRUCTURAL CORRESPONDENCE).
<table><tr><td>Graph edit</td><td>Same approach?</td><td>R</td><td>F</td><td>T</td><td>S</td><td>Mean score</td></tr><tr><td>Hallway step relabeled corridor</td><td>Yes</td><td>×</td><td>V</td><td>X</td><td>√</td><td>1.000</td></tr><tr><td>Extra hallway steps inserted</td><td>Yes</td><td>×</td><td>X</td><td>√</td><td>V</td><td>0.867</td></tr><tr><td>Both of the above</td><td>Yes</td><td>×</td><td>X</td><td>X</td><td>1</td><td>0.867</td></tr><tr><td>Kitchen branch added at a hallway</td><td>No</td><td>√</td><td>1</td><td>V</td><td>L</td><td>0.725</td></tr><tr><td>step Start room replaced</td><td>No</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.742</td></tr><tr><td>Target room replaced</td><td>No</td><td>√</td><td>V</td><td>√</td><td>√</td><td>0.742</td></tr><tr><td>Route ends at the branch leaf</td><td>No</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.542</td></tr><tr><td>Branch leaf moved to another node</td><td>No</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.213</td></tr><tr><td>Second target-room visit inserted</td><td>No</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.746</td></tr></table>

## B. Recorded Routes

Orders A/B use seeds 42/43. Records contain worker/episode identifiers, target, raw and final region labels, and a stored skeleton. The analysis uses the 483/444 successful-route records that contain at least one semantic anchor; empty and transition-only routes have no anchorbounded key.

All reductions use final labels, which incorporate region association, yielding 130/140 anchor-bounded sequences and 107/107 patterns. Transition-role mapping accounts for the first part of the merge (130/140 to 123/132) and consecutiverole contraction for the remainder, giving index reductions of 17.7% and 23.6%. Including transition endpoints gives 145/147 role sequences from 217/230 final-label sequences. This common input isolates the representation operation, and all pattern counts and similarity scores use the same visible final region-label sequences.

## C. Deterministic Transformations

The sampling units are the distinct anchor-bounded inputs within each order. For each eligible sequence we apply the six transformations named in Table VIII: transition alias replaces the first transition with another label in the fixed transition vocabulary; transition subdivision duplicates the first transition observation twice; adjacent anchor repeat repeats the first anchor next to itself; destination change replaces the last anchor with bedroom, or with kitchen when it is already bedroom; order reversal reverses the sequence and retains the pair when its reduced key changes; and transition removal deletes the first maximal transition run. The first three preserve the declared route pattern and the last three change it. Base routes can recur across conditions and orders; these are deterministic mechanism checks with descriptive counts.

TABLE VIII  
EXACT ROUTE-PATTERN DECISIONS ON ALL 1,325 DETERMINISTIC TRANSFORMATIONS.
<table><tr><td>Transformation</td><td>Pattern</td><td>Pairs</td><td>Correct</td></tr><tr><td>Transition alias</td><td>Same</td><td>193</td><td>193</td></tr><tr><td>Transition subdivision</td><td>Same</td><td>193</td><td>193</td></tr><tr><td>Adjacent anchor repeat</td><td>Same</td><td>270</td><td>270</td></tr><tr><td>Destination change</td><td>Different</td><td>270</td><td>270</td></tr><tr><td>Order reversal</td><td>Different</td><td>206</td><td>206</td></tr><tr><td>Transition removal</td><td>Different</td><td>193</td><td>193</td></tr></table>

TABLE IX

ACCEPTED PAIRS AT EDIT SIMILARITY ≥ 0.6. THE FIRST THREE CONDITIONS ARE EQUIVALENT ROUTES; THE LAST THREE HAVE DIFFERENT PATTERNS. THIS IS AN OFFLINE DIAGNOSTIC.
<table><tr><td>Transformation</td><td>Pairs</td><td>Raw</td><td>Reduced</td></tr><tr><td>Transition alias</td><td>193</td><td>193</td><td>193</td></tr><tr><td>Transition subdivision</td><td>193</td><td>193</td><td>193</td></tr><tr><td>Adjacent anchor repeat</td><td>270</td><td>257</td><td>270</td></tr><tr><td>Destination change</td><td>270</td><td>230</td><td>228</td></tr><tr><td>Order reversal</td><td>206</td><td>45</td><td>50</td></tr><tr><td>Transition removal</td><td>193</td><td>152</td><td>156</td></tr></table>

## D. Hard-Negative Structural Pairs

The two constructions of Section VI-E are instantiated over three start roles (a ∈ {living room, dining room, office}), two destination roles (b ∈ {bedroom, bathroom}), and two branch-role choices, each combination giving one pair of a base graph and its edited partner. The 12 instances per construction vary only these labels, so the reported decisions cover the tested combinations rather than arbitrary ones, and the counts are multiples of 12 by construction. The ✓ and × entries of Table III report descriptor equality versus exact structural correspondence, not thresholded retrieval decisions. A separate positive control inserts one transition vertex on the first route edge while leaving all other evidence unchanged, so it preserves structural correspondence and changes only the suppressed-transition count; these controls are excluded from the 24 pairs, and the table reports negative-pair outcomes only.

## E. Coverage Across Graph Shapes

We further examine every connected simple graph shape with two to five vertices. The 30 shapes receive two fixed role assignments, yielding 60 rooted contexts, and inserting two transition vertices on each edge in turn gives 322 edgerefinement checks. All satisfy the expected semantic relation: extending an existing passage preserves correspondence, while introducing the first transition between directly adjacent anchors changes the chain word.

Named graph cases test the additional structures in Table X, where parallel passages and returning cycles retain their distinct edge incidences and cropped junctions retain their stored degree and boundary mark. Separate tests preserve results under vertex renaming and insertion-order changes. Together, these checks exercise the reachability and incidence properties in Proposition 2 over the stated sizes, role assignments, and edits.

TABLE X  
STRUCTURAL COVERAGE OF THE INDEPENDENT STRUCTURAL-CORRESPONDENCE CHECKER. THE GRAPH-SHAPE SWEEP COMPRISES 322 EDGE REFINEMENTS; NAMED CASES TEST THE ADDITIONAL BOUNDARY CONDITIONS BELOW.
<table><tr><td>Graph context</td><td>Verified property</td></tr><tr><td>Small-graph edge refinements</td><td>Correspondence follows semantic chain-word preservation</td></tr><tr><td>Parallel passages</td><td>Distinct connections retain multiplicity</td></tr><tr><td>Returning cycles</td><td>Loops retain their protected attachment</td></tr><tr><td>Transition endpoints</td><td>Start and target remain explicit</td></tr><tr><td>Cropped junctions</td><td>Stored degree and boundary mark remain explicit</td></tr><tr><td>Distinct same-role anchors Single-vertex query</td><td>Separate region vertices remain distinct The rooted context is retained</td></tr></table>

## F. Continuous Similarity

Table IX gives complete acceptance counts at the offline threshold 0.6 for raw label sequences and contracted role sequences, separating pattern identity from continuous similarity. At the same threshold, 144 of 216 structurally non-equivalent graph pairs satisfy $S _ { \mathrm { g r a p h } } ~ \ge ~ 0 . 6$ , while the exact checker distinguishes all 216. Online retrieval uses graded scores and evidence gates.

## APPENDIX D EVALUATION PROTOCOL

## A. Navigation Settings

The navigation experiments use Qwen3-VL-Plus with a 40- step interaction budget. The prompt-context and direct-actionbias interfaces are enabled in the main HTCE configuration, with retrieved guidance applied subject to evidence confidence and action-selection conditions. The retrieval configuration of Table V is shared by all three datasets and by every component experiment; no coefficient, threshold, or gate is adjusted per dataset or per split. The component experiments vary successful-approach support, route reduction, and observation provenance as described in Section VI-C.

## B. Episode Scheduling and Memory

Navigation episodes are assigned to parallel workers by dataset ordering and evaluation seed; for the HM3D v0.2 route analyses, two episode orders generated with seeds 42 and 43 each assign 1,000 episodes to 50 workers. Persistent memory stores semantic region observations, graph connectivity, target evidence, and successful approaches across episode boundaries, with workers loading the shared snapshot at episode start and writing their updated snapshot at termination.

## C. Computational Cost

Retrieval and memory-update costs are measured over two evaluation orders containing 985 and 964 episode events. Retrieval latency is total retrieval time divided by the number of calls, update time is averaged per episode, and memory footprint is the largest serialized worker snapshot. Measurements characterize the memory component with prompt injection disabled and exclude VLM inference and network communication.