# Network Effect Decomposition for Causal Effect Estimation under

Network Interference

Zhirong Huang, Wenzhen Zhang, Guangquan Lu, Boyan Chen, Debo Cheng, and Shichao Zhang Senior Member, IEEE

Abstract—Estimating causal effects from observational network data requires accounting for both individual treatment and neighborhood exposure. Existing outcome models commonly incorporate these intervention variables into a joint predictor and estimate causal effects through prediction contrasts, leaving the corresponding response components implicit. We propose the Network Effect Decomposition Network (NEDNet), which parameterizes the conditional outcome response as a queryinvariant pre-treatment baseline plus separate treatment- and exposure-response pathways. Path-specific neighborhood contextualization supplies local pre-treatment information to the two pathways. We characterize the fitted response through a unique reference-normalized decomposition, which is equivalent to a zero mixed treatment–exposure contrast. Experiments on semisynthetic benchmarks derived from BlogCatalog and Flickr show that NEDNet achieves the lowest PEHE among the compared methods for individual, peer, and total effects across homogeneous and covariate-heterogeneous response settings.

Index Terms—Causal effect estimation, network interference, exposure mapping, potential outcomes.

## I. INTRODUCTION

Estimating causal effects from observational data is fundamental to decision-making in public health [1], social science [2], online platforms [3], and other domains [4]. Con ventional causal inference commonly invokes the Stable Unit Treatment Value Assumption (SUTVA), whose no-interference condition requires that a unit’s potential outcome be unaffected by other units’ treatment assignments [5]. This condition may fail in interconnected systems. In social networks [6], spatial communities [7], and economic networks [8], an intervention can affect not only the treated unit but also its neighbors through network connections. An individual’s outcome may therefore depend on both its own treatment and the treatments assigned to its neighbors, complicating the characterization of causal effects.

To characterize network-induced interference, an exposure mapping summarizes neighboring treatment assignments into a neighborhood-level variable, allowing potential outcomes to be expressed as functions of individual treatment and neighborhood exposure [9], [10]. Under this formulation, the individual effect (IE), peer effect (PE), and total effect (TE)

![](images/1e4b0cb6c67f6d1aa4e36a954f631d02a5e21c7811678c11e1722592704eca57.jpg)  
Fig. 1. Illustrative causal graph for two connected units i and j under network interference. Here, x, t, and y denote pre-treatment covariates, treatment, and outcome, respectively. Black arrows denote within-unit causal relationships, whereas red arrows denote spillover effects from one unit’s treatment to the other unit’s outcome.

quantify outcome changes associated with individual treatment, neighborhood exposure, or their joint variation [10], [11]. Estimating these effects requires comparing potential outcomes across alternative treatment–exposure conditions, rather than merely predicting outcomes under observed assignments. This is challenging in observational networks because treatment and exposure assignments may depend on pre-treatment covariates and network structure, whereas outcomes are observed only under realized treatment–exposure combinations. Accurate factual outcome prediction alone therefore does not guarantee accurate estimation of the counterfactual contrasts that define these causal effects.

Learning-based methods for causal effect estimation under network interference can be grouped according to how they incorporate network information into causal modeling. One line of work uses graph-based representations to capture neighborhood dependencies and interference-related context. For example, SPNet combines graph-convolutional representations with an environment summary variable and masked attention to characterize network interference [12]. Another line incorporates individual treatment and neighborhood exposure into causal estimation, often alongside balancing or robust estimation techniques, as in NetEst, RRNet, and TNet [10], [11], [13]. Related studies further investigate learned exposure mappings and peer-effect decomposition [14], [15]. Collectively, these strategies enrich network representations and provide complementary tools for estimating causal effects under network interference.

An important modeling issue is how to explicitly represent the outcome responses associated with individual treatment and neighborhood exposure as distinct components of a conditional outcome model. A joint predictor can incorporate both variables and estimate IE and PE through intervention-based prediction contrasts, but it does not explicitly parameterize the corresponding response components. For example, NetEst incorporates individual treatment and neighborhood exposure into a joint outcome estimator and derives their effects through prediction contrasts [10]. gDIS addresses a different aspect of peer-effect analysis by decomposing peer effects into mediationbased direct and indirect pathways; its decomposition concerns causal pathways rather than separately parameterized outcome responses to the two intervention dimensions [15].

To address this issue, we propose the Network Effect Decomposition Network (NEDNet) for causal effect estimation under network interference. NEDNet represents the conditional outcome response through a query-invariant pre-treatment baseline and separate pathways for individual-treatment and neighborhood-exposure responses. Both pathways are conditioned on compact pre-treatment representations, while pathspecific neighborhood contextualization supplies local pretreatment context. This structured predictor supports potentialoutcome estimation under alternative treatment–exposure queries. The main contributions are summarized as follows:

• A novel response-decomposition framework, termed the Network Effect Decomposition Network (NEDNet), is proposed for conditional outcome modeling under network interference, explicitly separating the pre-treatment baseline from individual-treatment and neighborhood-exposure responses.

• The separability of NEDNet’s conditional outcome function is theoretically characterized under a fixed pretreatment context, establishing that a unique referencenormalized additive decomposition exists if and only if the mixed treatment–exposure contrast vanishes.

• The component-wise structure of NEDNet’s plug-in estimators for IE, PE, and TE is theoretically characterized, establishing the respective roles of treatment- and exposure-response representations in the estimated effects.

## II. RELATED WORK

## A. Causal Effect Estimation from Observational Data

Causal effect estimation from observational data addresses non-random treatment assignment by adjusting for observed pre-treatment covariates [4], [16]. Classical approaches include outcome regression, propensity score methods, inverse probability weighting (IPW), and doubly robust estimators [17]– [20]. Their identification typically relies on conditions such as consistency, conditional exchangeability, and positivity, together with the Stable Unit Treatment Value Assumption (SUTVA) in settings without interference.

Recent neural estimators employ representation learning and flexible response functions to model potential outcomes from high-dimensional covariates [21]–[24]. TARNet learns a shared representation with treatment-specific outcome heads, while CFR further balances treated and control representations through an integral probability metric (IPM) [25]. Despite their effectiveness for individual treatment-effect estimation, these methods operate under the no-interference setting and do not explicitly account for outcome variations induced by neighboring treatments.

## B. Causal Effect Estimation under Network Interference

Classical studies of causal inference under interference extend the potential-outcome framework to account for treatment spillovers between connected units. Hudgens and Halloran [26] formalize direct, indirect, total, and overall causal effects under interference, while Tchetgen Tchetgen and Vander-Weele [27] further develop the corresponding identification and estimation framework. Aronow and Samii [6] establish a general randomization-based framework for causal effect estimation under interference. For observational networks, Forastiere et al. [9], [28] develop covariate-adjustment and generalized propensity-score methods for estimating treatment and interference effects.

Some learning-based methods exploit network representations to account for latent confounding and structural imbalance in observational networks [29]–[32]. These approaches primarily focus on confounding adjustment, whereas subsequent studies further investigate the explicit modeling of network interference.

Learning-based studies have incorporated network information into interference-aware causal estimation through several modeling strategies. Graph-based approaches use network structure to represent dependencies among connected units. Ma and Tresp employ graph neural networks for effect estimation under network interference and further study interventionpolicy improvement [33]. SPNet combines graph-convolutional representations with an environment summary variable and masked attention to model interference, while CauGramer uses higher-order graph transformers and cross-attention to address unknown interference structures [12], [34]. EgoNetGNN takes an exposure-mapping perspective and learns mapping functions for heterogeneous peer effects [14].

Other methods incorporate treatment and neighborhood exposure into causal estimation together with balancing or robust estimation strategies. NetEst combines graph representation learning with adversarial balancing, RRNet integrates joint propensity-score reweighting with IPM-based representation learning, and TNet incorporates targeted learning to construct a doubly robust estimator under network interference [10], [11], [13]. Separately, gDIS decomposes peer effects into mediationbased direct and indirect pathways [15].

These studies advance network representation, interference modeling, exposure mapping, and estimation strategies. NEDNet focuses on the parameterization of the conditional outcome response, representing the pre-treatment baseline and the responses associated with individual treatment and neighborhood exposure as distinct components.

## III. PROBLEM FORMULATION AND IDENTIFICATION

## A. Networked Observational Data

Let $\mathcal { G } = ( \nu , \mathcal { E } )$ be an undirected network with $N = | \nu |$ units and adjacency matrix $A \in \{ 0 , 1 \} ^ { N \times N }$ , where $A _ { i j } =$ 1 indicates an edge between units i and $j , \ A _ { i i } \ = \ 0$ , and $A = A ^ { \top }$ . For each unit $i \in \mathcal V$ , we observe a pre-treatment covariate vector $X _ { i } ~ \in ~ \mathbb { R } ^ { d }$ , a binary treatment $T _ { i } ~ \in ~ \{ 0 , 1 \}$ and a continuous outcome $Y _ { i } \in \mathbb { R }$ . Collectively, the observed covariates, treatments, and outcomes are represented by $X =$ $[ X _ { 1 } , \dots , X _ { N } ] ^ { \top } , T = [ T _ { 1 } , \dots , T _ { N } ] ^ { \top }$ , and $\bar { Y } = [ Y _ { 1 } , \ldots , \bar { Y } _ { N } ] ^ { \top }$ Let $\mathcal { N } _ { i } ~ = ~ \{ j ~ \in ~ \mathcal { V } ~ : ~ A _ { i j } ~ = ~ 1 \}$ denote the first-order neighborhood of unit $i ,$ and let $d _ { i } = | { \mathcal { N } } _ { i } |$ denote its degree. We write $T _ { \mathcal { N } _ { i } } = ( T _ { j } ) _ { j \in \mathcal { N } _ { i } }$ and $X _ { \mathcal { N } _ { i } } = ( X _ { j } ) _ { j \in \mathcal { N } _ { i } }$ for the treatments and covariates of neighboring units, respectively. Throughout this work, the network topology and node covariates are assumed to be fixed prior to treatment assignment, and we consider networks without isolated units so that $d _ { i } > 0$ for every unit.

Under network interference, the outcome of a unit may depend on both its own treatment and the treatments assigned to neighboring units. Following the exposure-mapping framework [9], [10], we summarize neighboring treatments through the exposure mapping:

$$
Z _ { i } = g _ { i } ( T _ { \mathcal { N } _ { i } } ) = \frac { 1 } { d _ { i } } \sum _ { j \in \mathcal { N } _ { i } } T _ { j } ,\tag{1}
$$

where $g _ { i } ( \cdot )$ denotes the exposure mapping and $Z _ { i } ~ \in ~ [ 0 , 1 ]$ represents the proportion of treated neighbors of unit i. The corresponding exposure vector is denoted by $Z = [ Z _ { 1 } , \ldots , Z _ { N } ] ^ { \top }$

## B. Causal Estimands and Problem Definition

a) Causal Estimands.: Let $Y _ { i } ( t , t _ { \mathcal { N } _ { i } } )$ denote the potential outcome of unit i under individual treatment t and neighboring treatment assignment $t _ { \mathcal { N } _ { i } }$ . Under the exposure-mapping assumption, neighboring treatment assignments that induce the same exposure value are outcome-equivalent, allowing us to write $Y _ { i } ( t , z )$ for $t ~ \in ~ \{ 0 , 1 \}$ and $z \in \mathcal Z _ { i }$ , where $\mathcal { Z } _ { i } ~ = ~ \{ k / d _ { i } ~ : ~ k ~ = ~ 0 , \ldots , d _ { i } \}$ denotes the set of feasible exposure values for a unit with $d _ { i } > 0$ . Let $C _ { i } = ( X _ { i } , X _ { \mathcal { N } _ { i } } )$ ) denote the pre-treatment covariates of unit i and its firstorder neighbors. The corresponding conditional mean potential outcome is defined as

$$
\mu _ { i } ( t , z ; C _ { i } ) = \mathbb { E } [ Y _ { i } ( t , z ) \mid C _ { i } ] .\tag{2}
$$

For notational simplicity, we suppress $C _ { i }$ in the subsequent expressions, with the understanding that all potential-outcome comparisons for unit i are evaluated under the same fixed pre-treatment context.

Using this conditional response function, we define the IE, PE, and TE as

$$
\tau _ { i } ^ { \mathrm { I } } ( z ) = \mu _ { i } ( 1 , z ) - \mu _ { i } ( 0 , z ) ,\tag{3}
$$

$$
\tau _ { i } ^ { \mathrm { P } } ( t ; z _ { a } , z _ { b } ) = \mu _ { i } ( t , z _ { a } ) - \mu _ { i } ( t , z _ { b } ) ,\tag{4}
$$

$$
\tau _ { i } ^ { \mathrm { T } } ( t _ { a } , z _ { a } ; t _ { b } , z _ { b } ) = \mu _ { i } ( t _ { a } , z _ { a } ) - \mu _ { i } ( t _ { b } , z _ { b } ) ,\tag{5}
$$

where $t , t _ { a } , t _ { b } \in \{ 0 , 1 \}$ and $z , z _ { a } , z _ { b } \in \mathcal { Z } _ { i }$ . These are unitlevel, context-specific contrasts. The IE changes individual treatment while holding neighborhood exposure fixed, whereas the PE changes neighborhood exposure while holding individual treatment fixed. These unit-level contrasts are related to the direct and spillover (or indirect) effects studied in classical interference frameworks [26], [27]. The TE compares two joint treatment–exposure conditions. Population-level summaries can be obtained by averaging the corresponding unit-level contrasts over a specified target population.

b) Problem Definition.: Given the observed networked dataset $\mathcal { D } ~ = ~ ( A , X , T , Z , Y )$ , our goal is to estimate the conditional mean potential outcome $\mu _ { i } ( t , z ; C _ { i } )$ for each unit under alternative individual treatment and neighborhood exposure conditions, and subsequently derive the corresponding causal effects. Specifically, we seek to learn a response function that approximates

$$
{ \widehat { \mu } } _ { \theta , i } ( t , z ; C _ { i } ) \approx \mu _ { i } ( t , z ; C _ { i } ) , \quad t \in \{ 0 , 1 \} , \ z \in { \mathcal { Z } } _ { i } ,\tag{6}
$$

subject to the identification conditions specified below.

Throughout the estimation process, the pre-treatment network structure and covariates are held fixed for each target unit. Accordingly, a counterfactual query consists of an individual treatment and a neighborhood exposure pair $( t , z )$ , and neighboring treatments affect the potential outcome only through the specified exposure mapping. The resulting response function therefore characterizes how the outcome of a target unit changes under alternative intervention conditions within its fixed pretreatment network context.

## C. Identification Conditions

Following established frameworks for causal inference under network interference [10], [11], [13], [26], [27], we adopt four identification assumptions: Network Consistency, Neighborhood Interference, Network Overlap, and Network Unconfoundedness. Here, $C _ { i }$ retains the definition given above, with the fixed network topology treated as part of the pretreatment context. These assumptions specify the conditions under which the target causal contrasts are identifiable from observational data.

Assumption 1 (Network Consistency). For each unit $i ,$ the observed outcome equals the potential outcome under its observed individual treatment and neighborhood exposure, i.e., $Y _ { i } = Y _ { i } ( T _ { i } , Z _ { i } )$

Assumption 2 (Neighborhood Interference). The potential outcome of unit i depends on neighboring treatment assignments only through the exposure mapping $g _ { i }$ applied to its first-order neighborhood. Specifically, for any two neighboring treatment assignments $t _ { \mathcal { N } _ { i } }$ and $t _ { \mathcal { N } _ { i } } ^ { \prime }$ satisfying $g _ { i } ( t _ { \mathcal { N } _ { i } } ) = g _ { i } ( t _ { \mathcal { N } _ { i } } ^ { \prime } )$ , we have $Y _ { i } ( t , t _ { \mathcal { N } _ { i } } ) = Y _ { i } ( t , t _ { \mathcal { N } _ { i } } ^ { \prime } ) .$

Assumption 3 (Network Overlap). For each treatment– exposure pair (t, z) required by a target causal contrast, where $t \in \{ 0 , 1 \}$ and $z \in { \mathcal { Z } } _ { i } , \operatorname* { P r } ( T _ { i } = t , Z _ { i } = z \mid C _ { i } ) > 0$ almost surely in the target population.

Assumption 4 (Network Unconfoundedness). For each admissible treatment–exposure pair $( t , z ) , Y _ { i } ( t , z ) \perp \perp ( T _ { i } , Z _ { i } ) \mid C _ { i }$

Under these assumptions, the conditional mean potential outcome satisfies

$$
\begin{array} { r l } & { \mu _ { i } ( t , z ; C _ { i } ) =  { { \mathbb E } } [ Y _ { i } ( t , z ) \mid C _ { i } ] } \\ & { \phantom { \mu _ { i } ( t , z ; C _ { i } ) = } =  { { \mathbb E } } [ Y _ { i } ( t , z ) \mid T _ { i } = t , Z _ { i } = z , C _ { i } ] } \\ & { \phantom { \mu _ { i } ( t , z ; C _ { i } ) = } =  { { \mathbb E } } [ Y _ { i } \mid T _ { i } = t , Z _ { i } = z , C _ { i } ] . } \end{array}\tag{7}
$$

The second equality follows from Network Unconfoundedness, and the third follows from Network Consistency.

![](images/33355e8991b025b93f0d9c3d9ff03edf732e533c7214abc1e944de72df0473cc.jpg)  
Fig. 2. Overview of the proposed NEDNet framework. Given the network structure A, pre-treatment covariates X, individual treatment T, and neighborhood exposure $Z ,$ NEDNet first learns a graph-dependent context representation H and a compact covariate representation $X ^ { w }$ . Treatment and exposure are then processed by separate response encoders and contextualized through separate graph-aggregation operations to obtain $( R ^ { T } , \bar { R } ^ { T } )$ and $( R ^ { Z } , \bar { R } ^ { \bar { Z } } )$ . Finally, the pre-treatment baseline and two intervention-specific response components are integrated to predict factual or queried potential outcomes, from which the IE, PE, and TE are estimated.

Consequently, the IE, PE, and TE in Eqs. (3)–(5) are identified for contrasts whose treatment–exposure pairs satisfy Network Overlap. The exposure mapping specifies which neighboring assignments are regarded as equivalent, whereas the overlap condition concerns whether a particular pair is supported by the observational assignment distribution. Thus, an exposure level can be feasible under the network structure without having positive observational probability.

## IV. THE PROPOSED NEDNET

As illustrated in Figure 2, NEDNet models the conditional outcome response through a pre-treatment baseline component and two intervention-specific response pathways. The treatment and exposure variables are encoded separately to obtain their respective response representations. Each pathway is further contextualized using the target unit’s graph representation and neighboring pre-treatment covariates. The resulting components are combined to predict outcomes under alternative treatment– exposure conditions, enabling the estimation of IE, PE, and TE through their corresponding response contrasts.

## A. Pre-treatment Context Modeling

Outcome differences across networked units are not determined solely by treatment and neighborhood exposure. Units may exhibit different baseline outcomes because of their covariates, neighborhood compositions, and structural positions. To represent this query-invariant pre-treatment variation separately from intervention-dependent variation, NEDNet models the graph-dependent baseline and the covariate conditioning information through two representations:

$$
H _ { i } = [ h _ { \phi } ( A , X ) ] _ { i } ,\tag{8}
$$

$$
X _ { i } ^ { w } = \operatorname { t a n h } \left( W _ { x } X _ { i } + b _ { x } \right) ,\tag{9}
$$

where $h _ { \phi }$ is a graph convolutional network (GCN)-based context encoder and $X _ { i } ^ { w } \in \mathbb { R } ^ { q }$ is a learned compact covariate representation with $q < d$

The two representations serve different purposes. The graphcontext representation $H _ { i }$ captures topology- and covariatedependent baseline variation and remains unchanged across alternative treatment–exposure queries. In contrast, $X _ { i } ^ { w }$ provides a compact node-specific covariate representation for conditioning both intervention-response pathways. This separation allows graph-dependent context to be modeled in the baseline while retaining explicit covariate conditioning in the response pathways.

## B. Treatment and Exposure Response Encoding

Individual treatment and neighborhood exposure define distinct intervention dimensions in the causal contrasts of Eqs. (3)–(5). To retain this distinction during response learning, NEDNet uses two independent encoders conditioned on the compact pre-treatment covariate representation $X _ { i } ^ { w }$ . For a target query (t, z), the local response representations are defined as

$$
R _ { i } ^ { T } ( t ) = f _ { T } ( t \parallel X _ { i } ^ { w } \parallel t X _ { i } ^ { w } ) ,\tag{10}
$$

$$
R _ { i } ^ { Z } ( z ) = f _ { Z } ( z \parallel X _ { i } ^ { w } ) ,\tag{11}
$$

Here, $f _ { T }$ and $f _ { Z }$ are path-specific multilayer perceptrons, and ∥ denotes concatenation. The two encoders independently transform individual treatment and neighborhood exposure into their respective response representations while incorporating the same compact pre-treatment covariate information.

The explicit interaction term $t X _ { i } ^ { w }$ enables treatmentconditioned feature modulation, allowing the treatment pathway to capture variations in covariate contributions across treatment conditions. Meanwhile, the nonlinear exposure encoder captures exposure-dependent variations in the covariate representation. This path-specific encoding provides distinct representations for modeling individual-treatment and neighborhood-exposure responses.

## C. Path-specific Neighborhood Contextualization

The separate encoders produce intervention-dependent representations conditioned on the target unit’s pre-treatment covariates. NEDNet further incorporates neighboring pretreatment characteristics through path-specific neighborhood contextualization. Specifically, for each target unit i, the treatment and exposure pathways aggregate messages constructed from the corresponding response representation, the target unit’s graph context, and each neighbor’s compact pre-treatment covariate representation:

$$
\bar { R } _ { i } ^ { T } ( t ) = \frac { 1 } { \sqrt { d _ { i } } } \sum _ { j \in \mathcal { N } _ { i } } \psi _ { T } \big ( R _ { i } ^ { T } ( t ) , H _ { i } , X _ { j } ^ { w } \big ) ,\tag{12}
$$

$$
\bar { R } _ { i } ^ { Z } ( z ) = \frac { 1 } { \sqrt { d _ { i } } } \sum _ { j \in \mathcal { N } _ { i } } \psi _ { Z } \big ( R _ { i } ^ { Z } ( z ) , H _ { i } , X _ { j } ^ { w } \big ) .\tag{13}
$$

Here, $\psi _ { T }$ and $\psi _ { Z }$ are learnable path-specific message functions. Each message combines the target unit’s intervention-specific representation and graph context $H _ { i }$ with a neighbor’s pretreatment representation $X _ { j } ^ { w }$ . The graph context $H _ { i }$ is invariant to the intervention query, whereas $X _ { j } ^ { w }$ preserves neighborspecific covariate information. Separate message functions allow the two pathways to process this context according to their respective intervention representations.

The aggregated representations $\bar { R } _ { i } ^ { T } ( t )$ and $\bar { R } _ { i } ^ { Z } ( z )$ supply neighborhood context to their respective response pathways. Individual treatment and neighborhood exposure enter through their corresponding target-unit encoders, while neighboring pre-treatment covariates provide contextual information during aggregation. Neighboring treatment assignments are not supplied separately to the message functions; their influence is represented through the exposure mapping z. Each aggregated representation is combined with its local response representation to form the corresponding component in Eqs. (16) and (17).

## D. Structured Outcome Response Modeling

NEDNet combines the pre-treatment baseline with two intervention-specific response components to construct the conditional outcome predictor. The baseline represents queryinvariant outcome variation associated with pre-treatment characteristics and network context, whereas the two response components characterize outcome variations associated with individual treatment and neighborhood exposure, respectively. Specifically, the fitted response is parameterized as

$$
\begin{array} { r l } & { \widehat { \mu } _ { \theta , i } ( t , z ; C _ { i } ) = F _ { \theta , i } ( A , X ; t , z ) } \\ & { \qquad = B _ { \theta , i } ( A , X ) + \Phi _ { \theta , i } ^ { T } ( t ) + \Phi _ { \theta , i } ^ { Z } ( z ) , } \end{array}\tag{14}
$$

where the three components are defined as

$$
B _ { \theta , i } ( A , X ) = \mu _ { H } ( H _ { i } ) + \mu _ { X } ( X _ { i } ^ { w } ) ,\tag{15}
$$

$$
\Phi _ { \theta , i } ^ { T } ( t ) = \mu _ { T } ^ { R } \big ( R _ { i } ^ { T } ( t ) , \bar { R } _ { i } ^ { T } ( t ) \big ) + \mu _ { T } ( t ) ,\tag{16}
$$

$$
\boldsymbol { \Phi } _ { \theta , i } ^ { Z } ( \boldsymbol { z } ) = \mu _ { Z } ^ { R } \big ( R _ { i } ^ { Z } ( \boldsymbol { z } ) , \bar { R } _ { i } ^ { Z } ( \boldsymbol { z } ) \big ) + \mu _ { Z } ( \boldsymbol { z } ) .\tag{17}
$$

Here, $\mu _ { H } , \mu _ { X } , \mu _ { T } ^ { R } , \mu _ { T } , \mu _ { Z } ^ { R }$ , and $\mu _ { Z }$ are learnable scalarvalued functions. The baseline component combines graphdependent context with compact pre-treatment covariate information. Each intervention-specific component combines its local response representation and aggregated neighborhood context through a response head, together with a direct readout of the corresponding intervention variable. This organization aligns the model components with the intervention dimensions varied in the IE and PE contrasts. Thus, $\Phi _ { \theta , i } ^ { T } ( t )$ and $\Phi _ { \theta , i } ^ { Z } ( z )$ are distinct model components for representing outcome variations associated with individual treatment and neighborhood exposure, respectively.

The two response components are defined with respect to their corresponding intervention variables while conditioning on the fixed pre-treatment context. This parameterization determines how NEDNet organizes the fitted outcome response, whereas IE, PE, and TE remain defined by the potentialoutcome contrasts in Eqs. (3)–(5). The relationship between this response representation and the resulting intervention contrasts is formally characterized in Proposition 1.

a) Reference-normalized Response.: The baseline and response components can be reparameterized through constant shifts without changing the fitted outcome. To obtain a unique representation of these components, we introduce a reference query $( t _ { 0 } , z _ { 0 } )$ and normalize the treatment and exposure responses with respect to this query. Throughout this work, we set $( t _ { 0 } , z _ { 0 } ) = ( 0 , 0 )$ , which is feasible for every unit because $d _ { i } > 0$ and $0 \in { \mathcal { Z } } _ { i }$ . Specifically,

$$
\begin{array} { r l } & { \quad \beta _ { \theta , i } = B _ { \theta , i } ( A , \boldsymbol { X } ) + \Phi _ { \theta , i } ^ { T } ( t _ { 0 } ) + \Phi _ { \theta , i } ^ { Z } ( z _ { 0 } ) , } \\ & { \rho _ { \theta , i } ^ { T } ( t ) = \Phi _ { \theta , i } ^ { T } ( t ) - \Phi _ { \theta , i } ^ { T } ( t _ { 0 } ) , } \\ & { \rho _ { \theta , i } ^ { Z } ( z ) = \Phi _ { \theta , i } ^ { Z } ( z ) - \Phi _ { \theta , i } ^ { Z } ( z _ { 0 } ) . } \end{array}\tag{18}
$$

(19)

Then the fitted response admits the equivalent representation

$$
\begin{array} { c } { { \widehat { \mu } _ { \theta , i } ( t , z ; C _ { i } ) = \beta _ { \theta , i } + \rho _ { \theta , i } ^ { T } ( t ) + \rho _ { \theta , i } ^ { Z } ( z ) , } } \\ { { \rho _ { \theta , i } ^ { T } ( t _ { 0 } ) = \rho _ { \theta , i } ^ { Z } ( z _ { 0 } ) = 0 . } } \end{array}\tag{20}
$$

Here, $\beta _ { \theta , \mathrm { i } }$ is the fitted outcome at the reference query, while $\rho _ { \theta , i } ^ { T } ( t )$ and $\rho _ { \theta , i } ^ { Z } ( z )$ represent the corresponding changes relative to that reference. This normalization is performed on the trained response functions at query time and does not introduce an additional training constraint. Proposition 1 establishes the uniqueness of this reference-normalized representation.

## E. Factual Learning Objective

NEDNet is trained end-to-end using observed factual outcomes. For each training unit i, the model evaluates the conditional outcome predictor at its observed treatment–exposure pair $( T _ { i } , Z _ { i } )$ , yielding $\widehat { Y } _ { i } = \widehat { \mu } _ { \boldsymbol { \theta } , i } ( T _ { i } , Z _ { i } ; C _ { i } )$ . The training objective minimizes the mean squared factual prediction error over the training-node set $\mathcal { V } _ { \mathrm { t r } }$ :

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { \vert \mathcal { V } _ { \mathrm { t r } } \vert } \sum _ { i \in \mathcal { V } _ { \mathrm { t r } } } \left( Y _ { i } - \widehat { \mu } _ { \theta , i } ( T _ { i } , Z _ { i } ; C _ { i } ) \right) ^ { 2 } .\tag{21}
$$

The objective jointly optimizes the baseline and the two intervention-specific response pathways through factual outcome supervision. After training, the learned response function is evaluated under alternative treatment–exposure queries to obtain counterfactual outcome predictions and estimate the corresponding causal effects. The training procedure is summarized in Algorithm 1.

## F. Potential Outcome Prediction and Effect Estimation

After training, NEDNet estimates potential outcomes by evaluating the learned response function under alternative treatment–exposure queries. Given a target unit i and a query (t, z), the pre-treatment network context remains fixed, and the predicted potential outcome is expressed using the referencenormalized response representation:

Algorithm 1 Training Procedure of NEDNet   
Require: Adjacency matrix A, covariates X, treatments $\overline { { T } } ,$   
outcomes ${ \mathit { Y } } ,$ training nodes $\mathcal { V } _ { \mathrm { t r } }$   
Ensure: Trained parameters θ   
1: Z ← EXPOSUREMAPPING(A, T)   
2: Initialize model parameters θ   
3: for each training epoch do   
4: H ← CONTEXTENCODER $( A , X )$   
5: $X ^ { w } \gets \mathrm { C O V A R I A T E N C O D E R } ( X )$   
6: $R ^ { T }$ ← TREATMENTENCODER $( T , X ^ { w } )$   
7: $R ^ { Z } \gets \mathrm { E x P O S U R E E N C O D E R } ( \dot { Z } , X ^ { w } )$   
8: $\bar { R } ^ { T }$ ← TREATMENTAGGREGATION $( A , R ^ { T } , H , X ^ { w } )$   
9: $\bar { R } ^ { Z }$ ← EXPOSUREAGGREGATION(A, R<sup>Z</sup>, H, X<sup>w</sup>)   
10: $\mathbf { B } \gets \mathbf { B A S E L I N E H E A D } ( H , X ^ { w } )$   
11: $\Phi ^ { T } \longleftarrow \mathrm { T R E A T M E N T H E A D } ( R ^ { T } , \bar { R } ^ { T } , T )$   
12: Φ<sup>Z</sup> ← EXPOSUREHEAD $( \dot { R } ^ { Z } , \bar { R } ^ { Z } , Z )$   
13: $\widehat { \mathbf { Y } } \gets \mathbf { B } + \Phi ^ { T } + \Phi ^ { Z }$   
14: $\mathcal { L } _ { \mathrm { p r e d } }  \frac { 1 } { | \mathcal { V } _ { \mathrm { t r } } | } \sum _ { i \in \mathcal { V } _ { \mathrm { t r } } } ( Y _ { i } - \widehat { Y } _ { i } ) ^ { 2 }$   
15: Update θ by minimizing $\mathcal { L } _ { \mathrm { p r e d } }$   
16: end for   
17: return θ

$$
\begin{array} { r l } & { \widehat { \mu } _ { \theta , i } ( t , z ; C _ { i } ) = [ F _ { \theta } ( A , X ; t , z ) ] _ { i } } \\ & { \qquad = \beta _ { \theta , i } + \rho _ { \theta , i } ^ { T } ( t ) + \rho _ { \theta , i } ^ { Z } ( z ) , } \end{array}\tag{22}
$$

where $\beta _ { \theta , i }$ represents the reference outcome, and $\rho _ { \theta , i } ^ { T } ( t )$ and $\rho _ { \theta , i } ^ { Z } ( z )$ characterize the treatment- and exposure-specific response variations, respectively.

Following the causal estimands defined in Eqs. (3)–(5), NEDNet estimates the individual, peer, and total effects by substituting the predicted potential outcomes into the corresponding causal contrasts:

$$
\begin{array} { r } { \widehat { \tau } _ { i } ^ { \mathrm { I } } ( z ) = \widehat { \mu } _ { \boldsymbol { \theta } , i } ( 1 , z ; C _ { i } ) - \widehat { \mu } _ { \boldsymbol { \theta } , i } ( 0 , z ; C _ { i } ) , } \end{array}\tag{23}
$$

$$
\begin{array} { r } { \widehat { \tau } _ { i } ^ { \mathrm { P } } ( t ; z _ { a } , z _ { b } ) = \widehat { \mu } _ { \boldsymbol { \theta } , i } ( t , z _ { a } ; C _ { i } ) - \widehat { \mu } _ { \boldsymbol { \theta } , i } ( t , z _ { b } ; C _ { i } ) , } \end{array}\tag{24}
$$

$$
\begin{array} { r } { \widehat { \tau } _ { i } ^ { \mathrm { T } } ( t _ { a } , z _ { a } ; t _ { b } , z _ { b } ) = \widehat { \mu } _ { \theta , i } ( t _ { a } , z _ { a } ; C _ { i } ) - \widehat { \mu } _ { \theta , i } ( t _ { b } , z _ { b } ; C _ { i } ) . } \end{array}\tag{25}
$$

These plug-in estimators directly correspond to the three causal estimands. Their relationship to the intervention-specific response components is established by the following theoretical analysis.

## G. Theoretical Properties of the Structured Response

Because the treatment and exposure pathways depend on their respective intervention variables, NEDNet’s fitted response has a zero mixed treatment–exposure contrast under a fixed pre-treatment context. The following proposition characterizes this property and establishes the uniqueness of the referencenormalized components.

Proposition 1 (Characterization of Separable Responses). Fix a unit i and its pre-treatment context $C _ { i } .$ Let $f _ { i } ( t , z ) ~ =$ $\widehat { \mu } _ { \theta , i } ( t , z ; C _ { i } )$ denote the fitted response over $\{ 0 , 1 \} \times \mathcal { Z } _ { i } .$ . Let $( t _ { 0 } , z _ { 0 } ) \in \{ 0 , 1 \} \times \mathcal { Z } _ { i }$ be a fixed reference query. The following statements are equivalent:

(i) The fitted response admits a separable decomposition

$$
\begin{array} { r } { f _ { i } ( t , z ) = \beta _ { \theta , i } + \rho _ { \theta , i } ^ { T } ( t ) + \rho _ { \theta , i } ^ { Z } ( z ) , } \end{array}\tag{26}
$$

where $\rho _ { \theta , i } ^ { T } ( t _ { 0 } ) = \rho _ { \theta , i } ^ { Z } ( z _ { 0 } ) = 0$ for a fixed reference query $( t _ { 0 } , z _ { 0 } )$

(ii) Its mixed treatment–exposure contrast vanishes for all admissible $t _ { a } , t _ { b } , z _ { a } , z _ { b } \colon$

$$
\begin{array} { l } { { f _ { i } ( t _ { a } , z _ { a } ) - f _ { i } ( t _ { b } , z _ { a } ) } } \\ { { - f _ { i } ( t _ { a } , z _ { b } ) + f _ { i } ( t _ { b } , z _ { b } ) = 0 . } } \end{array}\tag{27}
$$

Moreover, when these conditions hold, the referencenormalized components are uniquely determined by

$$
\begin{array} { r l r } & { } & { \beta _ { \theta , i } = f _ { i } ( t _ { 0 } , z _ { 0 } ) , ~ } \\ & { } & { \rho _ { \theta , i } ^ { T } ( t ) = f _ { i } ( t , z _ { 0 } ) - f _ { i } ( t _ { 0 } , z _ { 0 } ) , ~ } \\ & { } & { \rho _ { \theta , i } ^ { Z } ( z ) = f _ { i } ( t _ { 0 } , z ) - f _ { i } ( t _ { 0 } , z _ { 0 } ) . ~ } \end{array}\tag{28}
$$

Proof. Suppose that $f _ { i } ( t , z )$ admits the separable decomposition in Eq. (26). Substituting this decomposition into the mixed treatment–exposure contrast in Eq. (27), we obtain

$$
\begin{array} { r l } & { f _ { i } ( t _ { a } , z _ { a } ) - f _ { i } ( t _ { b } , z _ { a } ) - f _ { i } ( t _ { a } , z _ { b } ) + f _ { i } ( t _ { b } , z _ { b } ) } \\ & { = \rho _ { \theta , i } ^ { T } ( t _ { a } ) - \rho _ { \theta , i } ^ { T } ( t _ { b } ) - \rho _ { \theta , i } ^ { T } ( t _ { a } ) + \rho _ { \theta , i } ^ { T } ( t _ { b } ) } \\ & { \quad + \rho _ { \theta , i } ^ { Z } ( z _ { a } ) - \rho _ { \theta , i } ^ { Z } ( z _ { a } ) - \rho _ { \theta , i } ^ { Z } ( z _ { b } ) + \rho _ { \theta , i } ^ { Z } ( z _ { b } ) } \\ & { = 0 . } \end{array}
$$

Thus, (i) implies (ii).

Conversely, suppose that Eq. (27) holds for all admissible intervention queries. Setting $t _ { b } = t _ { 0 }$ and $z _ { b } = z _ { 0 }$ yields

$$
\begin{array} { r } { f _ { i } ( t , z ) = f _ { i } ( t , z _ { 0 } ) + f _ { i } ( t _ { 0 } , z ) - f _ { i } ( t _ { 0 } , z _ { 0 } ) . } \end{array}\tag{29}
$$

Defining the three components according to Eq. (28) directly gives the separable decomposition in Eq. (26), with the required reference constraints. Hence, (ii) implies (i).

It remains to establish uniqueness. The reference constraints uniquely determine the baseline as $f _ { i } ( t _ { 0 } , z _ { 0 } )$ , the treatmentresponse component as $f _ { i } ( t , z _ { 0 } ) - f _ { i } ( t _ { 0 } , z _ { 0 } )$ , and the exposureresponse component as $f _ { i } ( t _ { 0 } , z ) - f _ { i } ( t _ { 0 } , z _ { 0 } )$ . Therefore, the reference-normalized decomposition is unique. □

Corollary 1 (Effect Contrast Separation). For the fitted response of NEDNet, the estimated IE, PE, and TE satisfy

$$
\begin{array} { r } { \widehat { \tau } _ { i } ^ { \mathrm { I } } ( z ) = \rho _ { \theta , i } ^ { T } ( 1 ) - \rho _ { \theta , i } ^ { T } ( 0 ) , } \end{array}
$$

$$
\begin{array} { r } { \widehat { \tau } _ { i } ^ { \mathrm { P } } ( t ; z _ { a } , z _ { b } ) = \rho _ { \theta , i } ^ { Z } ( z _ { a } ) - \rho _ { \theta , i } ^ { Z } ( z _ { b } ) , } \end{array}\tag{30}
$$

$$
\begin{array} { r l } & { \widehat { \tau } _ { i } ^ { \mathrm { T } } ( t _ { a } , z _ { a } ; t _ { b } , z _ { b } ) = \rho _ { \theta , i } ^ { T } ( t _ { a } ) - \rho _ { \theta , i } ^ { T } ( t _ { b } ) } \\ & { \qquad + \rho _ { \theta , i } ^ { Z } ( z _ { a } ) - \rho _ { \theta , i } ^ { Z } ( z _ { b } ) . } \end{array}\tag{31}
$$

(32)

Accordingly, the plug-in estimate of IE obtained from NEDNet does not vary with the exposure query, while the plug-in estimate of PE does not vary with the individual-treatment query. The plug-in estimate of TE is given by the sum of the corresponding treatment- and exposure-response contrasts.

Proof. By Proposition 1, the fitted response admits the reference-normalized decomposition

$$
f _ { i } ( t , z ) = \beta _ { \boldsymbol { \theta } , i } + \boldsymbol { \rho } _ { \boldsymbol { \theta } , i } ^ { T } ( t ) + \boldsymbol { \rho } _ { \boldsymbol { \theta } , i } ^ { Z } ( z ) .\tag{33}
$$

Substituting this expression into the plug-in IE gives

$$
\begin{array} { r } { \widehat { \tau } _ { i } ^ { \mathrm { I } } ( z ) = f _ { i } ( 1 , z ) - f _ { i } ( 0 , z ) } \\ { = \rho _ { \theta , i } ^ { T } ( 1 ) - \rho _ { \theta , i } ^ { T } ( 0 ) . } \end{array}\tag{34}
$$

Similarly, the plug-in PE is

$$
\begin{array} { r } { \widehat { \tau } _ { i } ^ { \mathrm { P } } ( t ; z _ { a } , z _ { b } ) = f _ { i } ( t , z _ { a } ) - f _ { i } ( t , z _ { b } ) } \\ { = \rho _ { \theta , i } ^ { Z } ( z _ { a } ) - \rho _ { \theta , i } ^ { Z } ( z _ { b } ) . } \end{array}\tag{35}
$$

Finally, substituting the same decomposition into the plug-in TE yields

$$
\begin{array} { r l r } & { } & { \widehat { \tau } _ { i } ^ { \mathrm { T } } ( t _ { a } , z _ { a } ; t _ { b } , z _ { b } ) = f _ { i } ( t _ { a } , z _ { a } ) - f _ { i } ( t _ { b } , z _ { b } ) } \\ & { } & { = \rho _ { \theta , i } ^ { T } ( t _ { a } ) - \rho _ { \theta , i } ^ { T } ( t _ { b } ) ~ } \\ & { } & { + \rho _ { \theta , i } ^ { Z } ( z _ { a } ) - \rho _ { \theta , i } ^ { Z } ( z _ { b } ) . } \end{array}\tag{36}
$$

The stated properties follow directly from these expressions.

## V. EXPERIMENTS

We evaluate NEDNet on counterfactual outcome and causal effect estimation, and examine the contributions of its architectural components through ablation studies.

## A. Datasets and Semi-synthetic Data Generation

Following the benchmark settings used in NetEst [10] and TNet [11], we use two real-world social networks, BlogCatalog (BC) and Flickr, to construct semi-synthetic benchmarks. BC contains blogger relationships and keyword-based node features, whereas Flickr contains user connections and interest-tag features. We retain the original network structures A and node covariates X, while simulating treatment assignments and potential outcomes to obtain ground-truth counterfactual outcomes and causal effects.

To prevent information exchange between training, validation, and test nodes through graph message passing, each network is divided into vertex-disjoint training, validation, and test subgraphs. All neighborhood-dependent operations, including degree calculations, neighborhood averages, treatment assignment, exposure computation, and graph aggregation, are restricted to the corresponding subgraph. Isolated nodes are removed, resulting in 5,196 nodes for BC and 7,301 nodes for Flickr.

For each independent run, we sample $W _ { 1 } \sim { \mathcal { N } } ( 0 , I _ { d } )$ and compute the individual and neighborhood treatment scores:

$$
p _ { i } ^ { T } = \sigma ( W _ { 1 } ^ { \top } X _ { i } ) , \qquad p _ { \mathcal { N } _ { i } } ^ { T } = \frac { 1 } { d _ { i } } \sum _ { j \in \mathcal { N } _ { i } } p _ { j } ^ { T } ,\tag{37}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function. The overall treatment score is $\overset { \cdot } { s _ { i } ^ { T } } = p _ { i } ^ { T } + p _ { \mathcal { N } _ { i } } ^ { T }$ . Let $\gamma _ { s }$ denote the graph partition containing node i. The binary treatment is assigned according to

$$
T _ { i } = \mathbb { I } \left( s _ { i } ^ { T } > \overline { { s } } _ { s } ^ { T } \right) , \qquad \overline { { s } } _ { s } ^ { T } = \frac { 1 } { | \mathcal { V } _ { s } | } \sum _ { k \in \mathcal { V } _ { s } } s _ { k } ^ { T } .\tag{38}
$$

Given the realized treatment assignments, the neighborhood exposure is calculated as

$$
Z _ { i } = \frac { 1 } { d _ { i } } \sum _ { j \in \mathcal { N } _ { i } } T _ { j } .\tag{39}
$$

For potential outcome generation, we independently sample $W _ { 2 } \sim { \mathcal { N } } ( 0 , I _ { d } )$ and compute

$$
p _ { i } ^ { Y } = \sigma ( W _ { 2 } ^ { \top } X _ { i } ) , \qquad p _ { { \mathcal N } _ { i } } ^ { Y } = \frac { 1 } { d _ { i } } \sum _ { j \in { \mathcal N } _ { i } } p _ { j } ^ { Y } .\tag{40}
$$

We consider three outcome-generation settings. The homogeneous setting follows the original semi-synthetic benchmark, while the two heterogeneous settings follow TNet [11]. Specifically, the potential outcome is defined as

$$
Y _ { i } ( t , z ) = \left\{ \begin{array} { l l } { t + z + p _ { i } ^ { Y } + 0 . 5 p _ { N _ { i } } ^ { Y } + \epsilon _ { i } , \quad \mathrm { h o m o } , } \\ { \quad } \\ { t + z + p _ { i } ^ { Y } + 0 . 5 p _ { N _ { i } } ^ { Y } } \\ { \quad + t ( p _ { i } ^ { Y } + 0 . 5 p _ { N _ { i } } ^ { Y } ) + \epsilon _ { i } , \quad \mathrm { h e t e } , } \\ { \quad } \\ { t + z + p _ { i } ^ { Y } + 0 . 5 p _ { N _ { i } } ^ { Y } } \\ { \quad } \\ { \quad + t ( p _ { i } ^ { Y } + 0 . 5 p _ { N _ { i } } ^ { Y } ) } \\ { \quad } \\ { \quad + z ( 0 . 5 p _ { i } ^ { Y } + p _ { N _ { i } } ^ { Y } ) + \epsilon _ { i } , } \end{array} \right.\tag{41}
$$

Here, $\epsilon _ { i }$ is a Gaussian disturbance, sampled once for each node and held fixed across all treatment–exposure queries within the same run. The observed factual outcome is therefore $Y _ { i } = Y _ { i } ( T _ { i } , Z _ { i } )$ . The hete setting introduces heterogeneous individual-treatment effects through node-dependent covariate modulation. The hete\_z setting further incorporates heterogeneous neighborhood-exposure effects through an additional covariate-dependent response term.

We conduct five independent runs, indexed by expID. For each run, $W _ { 1 } , \ W _ { 2 }$ , and the node-specific disturbances $\left\{ \epsilon _ { i } \right\}$ are independently resampled. Within each run, the sampled disturbances remain fixed across all treatment–exposure queries. All competing methods use identical graph partitions, treatment assignments, factual outcomes, and ground-truth potential outcomes within each run.

## B. Experimental Setup

a) Baselines.: We compare NEDNet with seven representative estimators, including exposure-augmented outcome models and methods developed for networked causal inference. TARNet+Z and CFR+Z [10] extend TARNet and CFR [25] by incorporating neighborhood exposure as an additional outcome-model input. NetDeconf+Z [10] similarly augments the Network Deconfounder [31] with an explicit exposure input. NetEst [10] combines graph representation learning with balancing objectives for treatment and exposure. RRNet [13] integrates network representation learning with propensityscore reweighting, whereas TNet [11] uses targeted learning to construct a doubly robust estimator. We also include gDIS [15], which decomposes peer effects through graph-based mediation analysis.

b) Counterfactual Outcome Evaluation.: Following NetEst [10], RRNet [13], and TNet [11], we evaluate counterfactual outcome prediction by randomly selecting a proportion of nodes and reversing their factual treatments. Let $\nu _ { \mathrm { f i p } }$ denote

![](images/1741312d3eed4453ce7c0ae9cea327b2dc06099536cce8b8d552b83db5a5b63d.jpg)  
Fig. 3. Counterfactual outcome estimation performance (CF-MSE) under different treatment Flip Rates on BC and Flickr with the homogeneous outcome generation setting. From left to right, the panels show within-sample and out-of-sample results on BC, followed by those on Flickr. Lower values indicate better performance.

TABLE I  
CAUSAL EFFECT ESTIMATION RESULTS ON THE HOMOGENEOUS (HOMO) DATASETS MEASURED BY $\sqrt { \epsilon _ { \mathrm { P E H E } } }$ . RESULTS ARE REPORTED AS MEAN ± STANDARD DEVIATION OVER FIVE INDEPENDENT RUNS. LOWER VALUES ARE BETTER; THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY. “IND.”, “PEER”, AND “TOTAL” DENOTE THE INDIVIDUAL, PEER, AND TOTAL EFFECTS.
<table><tr><td>Datasets</td><td>Effects</td><td>TARNet+Z</td><td>CFR+Z</td><td>NetDeconf+Z</td><td>NetEst</td><td>RRNet</td><td>TNet</td><td>gDIS</td><td>NEDNet</td></tr><tr><td rowspan="3">BC (Within-sample)</td><td>Ind.</td><td> $0 . 2 0 0 0 { \scriptstyle \pm 0 . 0 6 7 4 }$ </td><td> $0 . 1 5 0 6 { \scriptstyle \pm 0 . 0 6 7 8 }$ </td><td> $0 . 2 6 9 0 { \scriptstyle \pm 0 . 1 3 7 9 }$ </td><td> $0 . 0 7 8 2 _ { \pm 0 . 0 2 4 9 }$ </td><td> $0 . 0 9 7 2 _ { \pm 0 . 0 9 1 5 }$ </td><td> $0 . 0 8 0 9 { \scriptstyle \pm 0 . 0 3 6 0 }$ </td><td> $0 . 1 2 8 7 _ { \pm 0 . 0 8 8 2 }$ </td><td> $\mathbf { 0 . 0 4 7 2 _ { \pm 0 . 0 3 1 9 } }$ </td></tr><tr><td>Peer</td><td> $0 . 4 9 7 7 { \scriptstyle \pm 0 . 0 5 2 0 }$ </td><td> $0 . 3 8 5 4 { \scriptstyle \pm 0 . 0 8 5 7 }$ </td><td> $0 . 4 9 6 2 { \scriptstyle \pm 0 . 1 1 6 2 }$ </td><td> $\overline { { 0 . 0 7 8 2 _ { \pm 0 . 0 3 3 6 } } }$ </td><td> $0 . 0 5 0 1 { \scriptstyle \pm 0 . 0 2 4 8 }$ </td><td> $0 . 0 8 2 3 { \scriptstyle \pm 0 . 0 3 3 7 }$ </td><td> $0 . 0 9 2 2 { \scriptstyle \pm 0 . 0 4 7 4 }$ </td><td> $\mathbf { 0 . 0 2 5 1 { \scriptstyle \pm 0 . 0 0 9 8 } }$ </td></tr><tr><td>Total</td><td> $0 . 7 8 5 5 { \scriptstyle \pm 0 . 1 7 4 3 }$ </td><td> $0 . 5 1 3 2 { \scriptstyle \pm 0 . 1 9 8 6 }$ </td><td> $0 . 6 4 5 7 { \scriptstyle \pm 0 . 2 8 1 7 }$ </td><td> $0 . 1 8 7 1 { \scriptstyle \pm 0 . 0 8 9 8 }$ </td><td> $\overline { { 0 . 1 6 8 2 _ { \pm 0 . 0 8 4 3 } } }$ </td><td> $\underline { { 0 . 0 8 6 0 _ { \pm 0 . 0 4 6 4 } } }$ </td><td> $0 . 2 4 2 8 { \scriptstyle \pm 0 . 1 8 0 5 }$ </td><td> $\mathbf { 0 . 0 5 3 9 { \scriptstyle \pm 0 . 0 2 9 2 } }$ </td></tr><tr><td rowspan="3">BC (Out-of-sample)</td><td>Ind.</td><td> $0 . 2 0 5 8 { \scriptstyle \pm 0 . 0 6 8 6 }$ </td><td> $0 . 1 5 2 1 { \scriptstyle \pm 0 . 0 6 8 3 }$ </td><td> $0 . 2 6 3 5 { \scriptstyle \pm 0 . 1 3 6 5 }$ </td><td> $0 . 0 7 0 6 { \scriptstyle \pm 0 . 0 2 7 8 }$ </td><td> $0 . 0 9 5 7 { \scriptstyle \pm 0 . 0 9 5 1 }$ </td><td> $0 . 0 8 1 9 _ { \pm 0 . 0 3 5 6 }$ </td><td> $0 . 1 3 0 8 { \scriptstyle \pm 0 . 0 8 8 0 }$ </td><td> $\mathbf { 0 . 0 5 0 4 _ { \pm 0 . 0 3 4 2 } }$ </td></tr><tr><td>Peer</td><td> $0 . 4 9 5 8 { \scriptstyle \pm 0 . 0 5 5 0 }$ </td><td> $0 . 3 8 5 3 { \scriptstyle \pm 0 . 0 8 5 8 }$ </td><td> $0 . 4 9 6 2 { \scriptstyle \pm 0 . 1 1 6 4 }$ </td><td> $\overline { { 0 . 0 7 4 2 _ { \pm 0 . 0 3 5 2 } } }$ </td><td> $0 . 0 5 0 1 { \scriptstyle \pm 0 . 0 2 5 0 }$ </td><td> $0 . 0 8 2 1 { \scriptstyle \pm 0 . 0 3 3 6 }$ </td><td> $0 . 0 9 4 1 { \scriptstyle \pm 0 . 0 4 5 5 }$ </td><td> $\mathbf { 0 . 0 2 5 6 { \scriptstyle \pm 0 . 0 0 9 7 } }$ </td></tr><tr><td>Total</td><td> $0 . 7 8 0 2 { \scriptstyle \pm 0 . 1 7 4 5 }$ </td><td> $0 . 5 1 2 4 { \scriptstyle \pm 0 . 2 0 0 4 }$ </td><td> $0 . 6 3 4 2 _ { \pm 0 . 2 8 4 2 }$ </td><td> $0 . 1 7 8 6 { \scriptstyle \pm 0 . 0 8 7 4 }$ </td><td> $\overline { { 0 . 1 6 6 7 _ { \pm 0 . 0 8 0 0 } } }$ </td><td> $\underline { { 0 . 0 8 8 4 _ { \pm 0 . 0 4 8 2 } } }$ </td><td> $0 . 2 5 2 2 { \scriptstyle \pm 0 . 1 7 3 2 }$ </td><td> $\mathbf { 0 . 0 5 6 8 _ { \pm 0 . 0 3 2 7 } }$ </td></tr><tr><td rowspan="3">Flickr (Within-sample)</td><td>Ind.</td><td> $0 . 2 0 5 1 { \scriptstyle \pm 0 . 0 4 1 9 }$ </td><td> $0 . 1 6 4 3 { \scriptstyle \pm 0 . 0 6 0 9 }$ </td><td> $0 . 2 7 7 6 { \scriptstyle \pm 0 . 0 9 0 2 }$ </td><td> $0 . 1 3 6 3 _ { \pm 0 . 0 5 7 6 }$ </td><td> $0 . 1 0 4 9 _ { \pm 0 . 0 3 6 5 }$ </td><td> $0 . 0 3 9 5 { \scriptstyle \pm 0 . 0 2 2 7 }$ </td><td> $0 . 0 6 8 0 { \scriptstyle \pm 0 . 0 2 2 8 }$ </td><td> $\mathbf { 0 . 0 2 7 6 { \scriptstyle \pm 0 . 0 1 0 8 } }$ </td></tr><tr><td>Peer</td><td> $0 . 4 7 9 4 { \scriptstyle \pm 0 . 0 2 8 1 }$ </td><td> $0 . 3 7 1 6 { \scriptstyle \pm 0 . 0 5 1 0 }$ </td><td> $0 . 4 9 3 9 { \scriptstyle \pm 0 . 0 6 3 8 }$ </td><td> $0 . 0 7 8 9 { \scriptstyle \pm 0 . 0 3 1 8 }$ </td><td> $0 . 0 5 2 8 { \scriptstyle \pm 0 . 0 1 1 2 }$ </td><td> $\overline { { 0 . 0 8 5 7 _ { \pm 0 . 0 1 7 9 } } }$ </td><td> $0 . 0 3 8 4 { \scriptstyle \pm 0 . 0 0 6 2 }$ </td><td> $\mathbf { 0 . 0 1 9 6 { \scriptstyle \pm 0 . 0 0 4 8 } }$ </td></tr><tr><td>Total</td><td> $0 . 7 8 3 5 { \scriptstyle \pm 0 . 0 7 2 5 }$ </td><td> $0 . 5 4 7 9 { \scriptstyle \pm 0 . 0 7 4 6 }$ </td><td> $0 . 6 9 3 4 { \scriptstyle \pm 0 . 0 9 3 5 }$ </td><td> $0 . 2 6 5 4 { \scriptstyle \pm 0 . 0 6 9 5 }$ </td><td> $0 . 1 5 7 3 { \scriptstyle \pm 0 . 0 3 6 5 }$ </td><td> $0 . 0 5 5 1 { \scriptstyle \pm 0 . 0 2 0 9 }$ </td><td> $\overline { { 0 . 1 2 3 1 { \scriptstyle \pm 0 . 0 6 6 5 } } }$ </td><td> $\mathbf { 0 . 0 4 4 3 _ { \pm 0 . 0 2 6 8 } }$ </td></tr><tr><td rowspan="3">Flickr (Out-of-sample)</td><td>Ind.</td><td> $0 . 1 7 9 3 _ { \pm 0 . 0 3 5 1 }$ </td><td> $0 . 1 6 4 7 _ { \pm 0 . 0 6 1 1 }$ </td><td> $0 . 2 5 8 5 { \scriptstyle \pm 0 . 0 8 9 2 }$ </td><td> $0 . 1 2 3 5 { \scriptstyle \pm 0 . 0 5 0 8 }$ </td><td> $0 . 1 0 5 0 { \scriptstyle \pm 0 . 0 3 6 0 }$ </td><td> $0 . 0 3 8 6 { \scriptstyle \pm 0 . 0 2 2 9 }$ </td><td> $0 . 0 7 7 5 { \scriptstyle \pm 0 . 0 1 8 4 }$ </td><td> $\mathbf { 0 . 0 2 6 7 { \scriptstyle \pm 0 . 0 0 8 1 } }$ </td></tr><tr><td>Peer</td><td> $0 . 4 6 8 1 { \scriptstyle \pm 0 . 0 3 3 0 }$ </td><td> $0 . 3 7 1 6 { \scriptstyle \pm 0 . 0 5 1 0 }$ </td><td> $0 . 4 9 3 8 { \scriptstyle \pm 0 . 0 6 4 0 }$ </td><td> $0 . 0 6 6 1 { \scriptstyle \pm 0 . 0 3 0 7 }$ </td><td> $0 . 0 5 2 6 { \scriptstyle \pm 0 . 0 1 0 6 }$ </td><td> $\overline { { 0 . 0 8 5 7 _ { \pm 0 . 0 1 7 8 } } }$ </td><td> $0 . 0 4 8 7 { \scriptstyle \pm 0 . 0 1 0 3 }$ </td><td> $\mathbf { 0 . 0 2 0 4 { \scriptstyle \pm 0 . 0 0 4 4 } }$ </td></tr><tr><td>Total</td><td> $0 . 7 1 1 3 { \scriptstyle \pm 0 . 0 5 7 3 }$ </td><td> $0 . 5 4 7 5 { \scriptstyle \pm 0 . 0 7 5 7 }$ </td><td> $0 . 6 7 0 3 { \scriptstyle \pm 0 . 0 9 5 9 }$ </td><td> $0 . 2 3 5 0 { \scriptstyle \pm 0 . 0 5 7 4 }$ </td><td> $0 . 1 5 6 3 _ { \pm 0 . 0 3 5 5 }$ </td><td> $0 . 0 5 4 0 { \scriptstyle \pm 0 . 0 2 0 7 }$ </td><td> $\overline { { 0 . 1 3 9 0 _ { \pm 0 . 0 6 3 6 } } }$ </td><td> $\mathbf { 0 . 0 4 2 8 _ { \pm 0 . 0 2 2 4 } }$ </td></tr></table>

The corresponding ground-truth counterfactual outcome is obtained from the known DGP:

NEDNet predicts the counterfactual outcome by evaluating its learned response function under the counterfactual treatment– exposure pair:

$$
\begin{array} { r } { Y _ { i } ^ { \mathrm { c f } } = Y _ { i } ( T _ { i } ^ { \mathrm { c f } } , Z _ { i } ^ { \mathrm { c f } } ) . } \end{array}
$$

The ratio $| \gamma _ { \mathrm { f i i p } } | / | \nu |$ is referred to as the Flip Rate. Since treatment changes can also alter the exposures of neighboring nodes, the counterfactual neighborhood exposure is recomputed as

$$
\begin{array} { r } { \widehat { Y } _ { i } ^ { \mathrm { c f } } = \widehat { \mu } _ { \boldsymbol { \theta } , i } ( T _ { i } ^ { \mathrm { c f } } , Z _ { i } ^ { \mathrm { c f } } ; C _ { i } ) . } \end{array}
$$

The counterfactual prediction error is measured by

the selected node set. The counterfactual treatment assignment is defined as

$$
\mathrm { C F - M S E } = \frac { 1 } { \left| \mathcal { V } _ { \mathrm { e v a l } } \right| } \sum _ { i \in \mathcal { V } _ { \mathrm { e v a l } } } \left( \widehat { Y } _ { i } ^ { \mathrm { c f } } - Y _ { i } ^ { \mathrm { c f } } \right) ^ { 2 } .
$$

$$
Z _ { i } ^ { \mathrm { c f } } = \frac { 1 } { d _ { i } } \sum _ { j \in \mathcal { N } _ { i } } T _ { j } ^ { \mathrm { c f } } .
$$

$$
T _ { i } ^ { \mathrm { c f } } = \left\{ \begin{array} { l l } { 1 - T _ { i } , } & { i \in \mathcal { V } _ { \mathrm { f l i p } } , } \\ { T _ { i } , } & { i \notin \mathcal { V } _ { \mathrm { f l i p } } . } \end{array} \right.\tag{42}
$$

(43)

(44)

(46)

(45)

c) Causal Effect Evaluation.: For causal effect estimation, the ground-truth IE, PE, and TE are calculated from the potential outcomes generated by the DGP, while their estimates are obtained through the plug-in contrasts of the predicted potential outcomes. For each effect type $e \in \{ \mathrm { I } , \mathrm { P } , \mathrm { T } \}$ , we report the Precision in Estimation of Heterogeneous Effect (PEHE):

$$
\sqrt { \epsilon _ { \mathrm { P E H E } } ^ { e } } = \sqrt { \frac { 1 } { \left| \mathcal { V } _ { \mathrm { e v a l } } \right| } \sum _ { i \in \mathcal { V } _ { \mathrm { e v a l } } } \left( \widehat { \tau } _ { i } ^ { e } - \tau _ { i } ^ { e } \right) ^ { 2 } } .\tag{47}
$$

Lower CF-MSE and PEHE values indicate better counterfactual outcome and causal effect estimation, respectively. We report within-sample performance on the training subgraphs and outof-sample performance on the held-out test subgraphs.

d) Implementation Details.: NEDNet is implemented in PyTorch and evaluated on a server equipped with a 64-core Montage Jintide C6326 CPU and eight NVIDIA GeForce RTX 4090 GPUs with 24 GB of memory each. The same hyperparameter configuration is used for BlogCatalog and Flickr. The graph-context encoder contains one graph convolutional network (GCN) layer followed by a two-layer nodewise multilayer perceptron (MLP), while the dimension of the compact covariate representation is fixed at 2. The hidden dimensions of the GCN and path-specific MLP encoders are selected from prespecified candidate ranges according to the factual outcome prediction error on the validation set. NEDNet is trained for 500 epochs using the Adam optimizer with a learning rate of $1 0 ^ { - 3 }$ and no weight decay.

TABLE II  
ABLATION RESULTS ON BC AND FLICKR MEASURED BY $\scriptstyle { \sqrt { \epsilon _ { \mathrm { P E H E } } } } .$
<table><tr><td>Datasets</td><td>Effects</td><td>NEDNet</td><td>w/o T-Rep.</td><td> $\mathrm { w } / 0 ~ { \cal Z } { \mathrm { - R e p . } }$ </td><td>Shared encoder</td><td>Joint decoder</td></tr><tr><td rowspan="3">BC (Within-sample)</td><td>Ind.</td><td> $\mathbf { 0 . 0 4 7 2 { \scriptstyle \pm 0 . 0 3 1 9 } }$ </td><td> $1 . 2 0 0 8 { \scriptstyle \pm 0 . 0 2 5 4 }$ </td><td> $2 . 0 3 3 6 { \scriptstyle \pm 0 . 0 6 6 7 }$ </td><td> $0 . 0 6 7 8 { \scriptstyle \pm 0 . 0 4 0 6 }$ </td><td> $0 . 0 5 5 5 { \scriptstyle \pm 0 . 0 2 1 3 }$ </td></tr><tr><td>Peer</td><td> $\mathbf { 0 . 0 2 5 1 { \scriptstyle \pm 0 . 0 0 9 8 } }$ </td><td> $0 . 6 7 3 4 { \scriptstyle \pm 0 . 1 0 7 7 }$ </td><td> $0 . 7 9 2 9 { \scriptstyle \pm 0 . 0 2 8 6 }$ </td><td> $0 . 0 2 8 3 _ { \pm 0 . 0 0 7 6 }$ </td><td> $\overline { { 0 . 0 3 6 7 _ { \pm 0 . 0 1 0 4 } } }$ </td></tr><tr><td>Total</td><td> $\mathbf { 0 . 0 5 3 9 { \scriptstyle \pm 0 . 0 2 9 2 } }$ </td><td> $0 . 7 5 1 2 { \scriptstyle \pm 0 . 0 4 4 8 }$ </td><td> $0 . 8 6 6 5 { \scriptstyle \pm 0 . 0 3 4 1 }$ </td><td> $\overline { { 0 . 0 7 6 1 { \scriptstyle \pm 0 . 0 3 5 5 } } }$ </td><td> $0 . 0 8 7 0 { \scriptstyle \pm 0 . 0 4 0 0 }$ </td></tr><tr><td rowspan="3">BC (Out-of-sample)</td><td>Ind.</td><td> $\mathbf { 0 . 0 5 0 4 { \scriptstyle \pm 0 . 0 3 4 2 } }$ </td><td> $1 . 2 0 1 9 { \scriptstyle \pm 0 . 0 2 7 9 }$ </td><td> $1 . 8 0 5 1 { \scriptstyle \pm 0 . 0 6 2 9 }$ </td><td> $0 . 0 6 8 9 { \scriptstyle \pm 0 . 0 4 1 7 }$ </td><td> $0 . 0 5 8 3 { \scriptstyle \pm 0 . 0 2 1 8 }$ </td></tr><tr><td>Peer</td><td> $\mathbf { 0 . 0 2 5 6 { \scriptstyle \pm 0 . 0 0 9 7 } }$ </td><td> $0 . 6 2 0 3 _ { \pm 0 . 0 8 0 5 }$ </td><td> $0 . 7 9 3 2 { \scriptstyle \pm 0 . 0 2 7 6 }$ </td><td> $0 . 0 2 9 4 { \scriptstyle \pm 0 . 0 0 8 3 }$ </td><td> $\overline { { 0 . 0 3 7 2 _ { \pm 0 . 0 1 0 2 } } }$ </td></tr><tr><td>Total</td><td> $\mathbf { 0 . 0 5 6 8 _ { \pm 0 . 0 3 2 7 } }$ </td><td> $0 . 7 3 5 9 { \scriptstyle \pm 0 . 0 8 8 4 }$ </td><td> $0 . 5 9 3 3 { \scriptstyle \pm 0 . 0 1 7 3 }$ </td><td> $\overline { { 0 . 0 7 7 3 _ { \pm 0 . 0 3 5 1 } } }$ </td><td> $0 . 0 8 6 2 { \scriptstyle \pm 0 . 0 3 5 6 }$ </td></tr><tr><td rowspan="3">Flickr (Within-sample)</td><td>Ind.</td><td> $\mathbf { 0 . 0 2 7 6 { \scriptstyle \pm 0 . 0 1 0 8 } }$ </td><td> $1 . 1 8 9 6 { \scriptstyle \pm 0 . 0 1 7 9 }$ </td><td> $3 . 2 2 5 8 { \scriptstyle \pm 0 . 1 6 2 3 }$ </td><td> $0 . 0 3 8 7 { \scriptstyle \pm 0 . 0 1 0 7 }$ </td><td> $0 . 0 3 4 7 { \scriptstyle \pm 0 . 0 0 6 9 }$ </td></tr><tr><td>Peer</td><td> $\mathbf { 0 . 0 1 9 6 { \scriptstyle \pm 0 . 0 0 4 8 } }$ </td><td> $0 . 9 1 8 0 { \scriptstyle \pm 0 . 2 4 2 5 }$ </td><td> $0 . 7 4 7 9 { \scriptstyle \pm 0 . 0 1 4 1 }$ </td><td> $0 . 0 2 8 5 { \scriptstyle \pm 0 . 0 0 7 1 }$ </td><td> $\overline { { 0 . 0 3 0 3 _ { \pm 0 . 0 0 7 0 } } }$ </td></tr><tr><td>Total</td><td> $0 . 0 4 4 3 { \scriptstyle \pm 0 . 0 2 6 8 }$ </td><td> $1 . 1 3 4 1 { \scriptstyle \pm 0 . 4 2 8 2 }$ </td><td> $2 . 1 8 0 1 { \scriptstyle \pm 0 . 1 4 4 5 }$ </td><td> $\overline { { 0 . 0 4 4 0 { \scriptstyle \pm 0 . 0 0 9 1 } } }$ </td><td> $\mathbf { 0 . 0 4 2 9 { \scriptstyle \pm 0 . 0 0 5 0 } }$ </td></tr><tr><td rowspan="3">Flickr (Out-of-sample)</td><td>Ind.</td><td> $\mathbf { 0 . 0 2 6 7 { \scriptstyle \pm 0 . 0 0 8 1 } }$ </td><td> $1 . 1 9 0 6 { \scriptstyle \pm 0 . 0 1 7 9 }$ </td><td> $2 . 3 7 3 7 { \scriptstyle \pm 0 . 1 1 7 0 }$ </td><td> $0 . 0 3 8 6 { \scriptstyle \pm 0 . 0 1 1 8 }$ </td><td> $0 . 0 3 4 0 { \scriptstyle \pm 0 . 0 0 9 3 }$ </td></tr><tr><td>Peer</td><td> $\mathbf { 0 . 0 2 0 4 { \scriptstyle \pm 0 . 0 0 4 4 } }$ </td><td> $0 . 7 2 0 6 { \scriptstyle \pm 0 . 1 6 1 6 }$ </td><td> $0 . 7 4 8 7 { \scriptstyle \pm 0 . 0 1 4 9 }$ </td><td> $0 . 0 2 9 4 { \scriptstyle \pm 0 . 0 0 6 6 }$ </td><td> $\overline { { 0 . 0 3 0 6 { \scriptstyle \pm 0 . 0 0 6 1 } } }$ </td></tr><tr><td>Total</td><td> $\mathbf { 0 . 0 4 2 8 _ { \pm 0 . 0 2 2 4 } }$ </td><td> $0 . 8 9 3 9 { \scriptstyle \pm 0 . 2 0 9 7 }$ </td><td> $1 . 4 8 5 3 { \scriptstyle \pm 0 . 0 9 1 3 }$ </td><td> $\overline { { 0 . 0 4 2 9 _ { \pm 0 . 0 0 8 0 } } }$ </td><td> $0 . 0 4 3 8 { \scriptstyle \pm 0 . 0 0 7 3 }$ </td></tr></table>

For competing methods, we adopt the recommended hyper parameter settings reported in their original papers or released implementations. For TARNet+Z, CFR+Z, and NetDeconf+Z, we retain the original architectures and hyperparameters while incorporating neighborhood exposure as an additional input. All methods are evaluated using identical data realizations and graph partitions.

## C. Main Results

a) Counterfactual Outcome Estimation.: Figure 3 reports the counterfactual prediction errors under different treatment Flip Rates. Increasing the Flip Rate changes the treatment assignments of more nodes and consequently alters the neigh borhood exposures induced by the network. NEDNet maintains consistently low CF-MSE across the evaluated Flip Rates on both datasets. Moreover, its within-sample and out-of-sample performance remains comparable, indicating that the learned response function generalizes to held-out graph subgraphs under alternative treatment assignments.

b) Causal Effect Estimation.: Table I reports the estimation errors of IE, PE, and TE on BC and Flickr. NEDNet achieves the lowest PEHE in all twelve dataset–setting–effect combinations. The strongest competing method varies across effect types and datasets, suggesting that the relative performance of existing estimators depends on the particular causal contrast being evaluated. In comparison, NEDNet consistently provides the most accurate estimates across all three effects.

The gaps between within-sample and out-of-sample PEHE remain modest across the three effect types. This pattern suggests that the estimation gains are not confined to the nodes used for model fitting and are largely maintained on held-out graph subgraphs.

## D. Ablation Study

The advantage is particularly evident in the out-of-sample results. Compared with the best-performing baseline in each setting, NEDNet reduces the estimation error by 28.6%– 48.9% on BC and 20.7%–58.1% on Flickr. The reductions are especially pronounced for PE, with improvements of 48.9% on BC and 58.1% on Flickr. These results demonstrate the effectiveness of NEDNet in estimating neighborhood-exposure effects while maintaining strong performance in individual and joint intervention effect estimation.

We conduct ablation studies on BC and Flickr to examine the contributions of the response representations, path-specific encoding, and decoder design. Four variants are considered. w/o $T _ { - } R e p$ . and w/o $Z { \mathrm { - } } R e p .$ . remove the representation-based terms $\mu _ { T } ^ { R }$ and $\mu _ { Z } ^ { R } .$ , respectively, while retaining the corresponding direct readouts $\mu _ { T }$ and $\mu _ { Z }$ . Shared encoder jointly encodes treatment and exposure before graph contextualization, whereas Joint decoder replaces the path-specific response decoding with a unified MLP. All variants are evaluated under identical training and evaluation settings.

Table II reports the ablation results on BC and Flickr. Removing either representation-based response term leads to substantial increases in PEHE across the three effect types and both evaluation settings. These results indicate that the graph-contextualized treatment and exposure representations provide complementary information beyond the corresponding direct readouts, contributing to the estimation of individual, peer, and total effects.

The comparison with the shared encoder further highlights the importance of path-specific encoding. NEDNet consistently achieves lower errors than the shared-encoder variant across all BC settings and the individual and peer effects on Flickr. A similar pattern is observed for the joint decoder, which generally produces higher errors than NEDNet. These results suggest that separately encoding the two intervention dimensions and preserving their respective response pathways are beneficial to causal effect estimation.

On Flickr, the joint decoder and shared encoder obtain slightly lower within-sample errors for the total effect than NEDNet, with PEHE values of 0.0429 and 0.0440, respectively, compared with 0.0443 for NEDNet. However, neither variant achieves lower errors for the individual and peer effects in this setting, and both yield higher total-effect errors in the out-of-sample evaluation. This result indicates that the relative performance of decoder configurations can vary across effect types and evaluation settings, while the path-specific design of NEDNet provides consistently strong performance across both datasets.

TABLE III  
<sub>CAUSAL</sub> <sub>EFFECT</sub> <sub>ESTIMATION</sub> <sub>ON</sub> <sub>THE</sub> <sub>H</sub> <sub>E</sub> <sub>T</sub> <sub>E</sub> <sub>DATASETS</sub> <sub>MEASURED</sub> <sub>BY</sub> √<sub>ϵPEHE.</sub> <sub>TNET</sub>† <sub>DENOTES</sub> <sub>TNET</sub> <sub>WITH</sub> <sub>HYPERPARAMETERS</sub> <sub>SELECTED</sub> <sub>ON</sub> <sub>THE</sub> VALIDATION SET FOR THE CORRESPONDING HETEROGENEOUS SETTING.
<table><tr><td>Datasets</td><td>Effects</td><td>TARNet+Z</td><td>CFR+Z</td><td>NetDeconf+Z</td><td>NetEst</td><td>RRNet</td><td>TNet†</td><td> $\mathrm { \ g D I S }$ </td><td>NEDNet</td></tr><tr><td rowspan="3">BC (Within-sample)</td><td>Ind.</td><td> $0 . 3 4 3 8 { \scriptstyle \pm 0 . 0 8 7 6 }$ </td><td> $0 . 3 2 4 0 { \scriptstyle \pm 0 . 0 7 0 3 }$ </td><td> $0 . 4 8 5 6 { \scriptstyle \pm 0 . 1 3 7 8 }$ </td><td> $0 . 1 5 7 1 { \scriptstyle \pm 0 . 0 5 3 2 }$ </td><td> $0 . 1 1 0 8 { \scriptstyle \pm 0 . 0 4 4 6 }$ </td><td> $0 . 0 5 6 7 { \scriptstyle \pm 0 . 0 2 1 7 }$ </td><td> $0 . 2 6 9 1 { \scriptstyle \pm 0 . 0 6 3 5 }$ </td><td>0.0353±0.0064</td></tr><tr><td>Peer</td><td> $0 . 5 5 8 8 { \scriptstyle \pm 0 . 0 5 4 0 }$ </td><td> $0 . 5 0 5 2 { \scriptstyle \pm 0 . 0 5 9 6 }$ </td><td> $0 . 6 0 1 8 { \scriptstyle \pm 0 . 0 6 6 6 }$ </td><td> $0 . 0 7 7 8 { \scriptstyle \pm 0 . 0 2 6 9 }$ </td><td> $0 . 0 5 0 1 { \scriptstyle \pm 0 . 0 1 9 0 }$ </td><td> $\overline { { 0 . 1 0 3 7 _ { \pm 0 . 0 3 2 4 } } }$ </td><td> $0 . 1 3 9 8 { \scriptstyle \pm 0 . 0 6 7 2 }$ </td><td> $\mathbf { 0 . 0 2 2 9 { \scriptstyle \pm 0 . 0 0 9 9 } }$ </td></tr><tr><td>Total</td><td> $0 . 8 4 1 3 _ { \pm 0 . 1 4 6 7 }$ </td><td> $0 . 6 8 5 9 { \scriptstyle \pm 0 . 1 6 8 8 }$ </td><td> $0 . 7 5 8 6 { \scriptstyle \pm 0 . 1 8 7 5 }$ </td><td> $0 . 2 1 8 9 _ { \pm 0 . 0 9 1 8 }$ </td><td> $0 . 2 0 3 5 { \scriptstyle \pm 0 . 0 5 8 4 }$ </td><td> $0 . 1 2 6 5 { \scriptstyle \pm 0 . 0 8 2 7 }$ </td><td> $0 . 5 0 8 9 { \scriptstyle \pm 0 . 1 6 0 9 }$ </td><td> $\mathbf { 0 . 0 6 3 6 { \scriptstyle \pm 0 . 0 3 1 6 } }$ </td></tr><tr><td rowspan="3">BC (Out-of-sample)</td><td>Ind.</td><td> $0 . 3 4 4 4 { \scriptstyle \pm 0 . 0 9 4 2 }$ </td><td> $0 . 3 2 7 8 { \scriptstyle \pm 0 . 0 6 6 0 }$ </td><td> $0 . 4 7 8 1 { \scriptstyle \pm 0 . 1 3 9 8 }$ </td><td> $0 . 1 5 1 9 { \scriptstyle \pm 0 . 0 5 8 2 }$ </td><td>0.1091±0.0433</td><td> $0 . 0 5 6 2 { \scriptstyle \pm 0 . 0 2 1 7 }$ </td><td> $0 . 2 7 2 4 { \scriptstyle \pm 0 . 0 6 5 7 }$ </td><td> $\mathbf { 0 . 0 3 6 1 { \scriptstyle \pm 0 . 0 0 6 9 } }$ </td></tr><tr><td>Peer</td><td> $0 . 5 5 9 0 { \scriptstyle \pm 0 . 0 5 5 3 }$ </td><td> $0 . 5 0 5 2 { \scriptstyle \pm 0 . 0 5 9 6 }$ </td><td> $0 . 6 0 1 8 { \scriptstyle \pm 0 . 0 6 6 5 }$ </td><td> $0 . 0 7 3 4 { \scriptstyle \pm 0 . 0 3 0 5 }$ </td><td> $0 . 0 5 0 1 { \scriptstyle \pm 0 . 0 2 0 1 }$ </td><td> $\overline { { 0 . 1 0 2 7 _ { \pm 0 . 0 3 0 1 } } }$ </td><td> $0 . 1 4 0 9 _ { \pm 0 . 0 6 6 2 }$ </td><td> $\mathbf { 0 . 0 2 3 5 _ { \pm 0 . 0 0 9 9 } }$ </td></tr><tr><td>Total</td><td> $0 . 8 2 3 6 { \scriptstyle \pm 0 . 1 5 9 9 }$ </td><td> $0 . 6 8 8 5 { \scriptstyle \pm 0 . 1 6 8 9 }$ </td><td> $0 . 7 4 6 6 { \scriptstyle \pm 0 . 1 8 9 8 }$ </td><td> $0 . 2 1 1 0 { \scriptstyle \pm 0 . 0 9 1 2 }$ </td><td> $0 . 2 0 2 6 { \scriptstyle \pm 0 . 0 5 5 1 }$ </td><td> $\underline { { 0 . 1 2 8 1 _ { \pm 0 . 0 8 4 5 } } }$ </td><td> $0 . 5 1 2 0 { \scriptstyle \pm 0 . 1 5 9 2 }$ </td><td> $\mathbf { 0 . 0 6 6 3 _ { \pm 0 . 0 3 3 0 } }$ </td></tr><tr><td rowspan="3">Flickr (Within-sample)</td><td>Ind.</td><td> $0 . 3 7 0 9 { \scriptstyle \pm 0 . 0 4 9 0 }$ </td><td> $0 . 3 3 0 5 { \scriptstyle \pm 0 . 0 8 0 7 }$ </td><td> $0 . 4 7 3 8 { \scriptstyle \pm 0 . 0 4 6 7 }$ </td><td> $0 . 2 2 3 0 { \scriptstyle \pm 0 . 0 5 6 1 }$ </td><td> $0 . 1 2 0 8 _ { \pm 0 . 0 2 7 8 }$ </td><td> $0 . 0 7 5 4 { \scriptstyle \pm 0 . 0 2 8 2 }$ </td><td> $0 . 2 3 2 6 { \scriptstyle \pm 0 . 0 4 4 6 }$ </td><td> $\mathbf { 0 . 0 4 8 1 _ { \pm 0 . 0 3 5 2 } }$ </td></tr><tr><td>Peer</td><td> $0 . 5 2 8 7 { \scriptstyle \pm 0 . 0 3 6 1 }$ </td><td> $0 . 4 4 5 1 { \scriptstyle \pm 0 . 0 5 7 4 }$ </td><td> $0 . 5 6 5 7 { \scriptstyle \pm 0 . 0 2 6 2 }$ </td><td>0.0637±0.0165</td><td> $0 . 0 5 4 3 { \scriptstyle \pm 0 . 0 0 6 7 }$ </td><td> $\overline { { 0 . 0 9 7 7 _ { \pm 0 . 0 3 0 8 } } }$ </td><td> $0 . 0 9 8 5 { \scriptstyle \pm 0 . 0 2 6 6 }$ </td><td> $\mathbf { 0 . 0 2 2 2 { \scriptstyle \pm 0 . 0 0 8 8 } }$ </td></tr><tr><td>Total</td><td> $0 . 8 3 8 5 { \scriptstyle \pm 0 . 0 9 2 4 }$ </td><td> $0 . 6 2 3 6 { \scriptstyle \pm 0 . 0 9 6 0 }$ </td><td> $0 . 7 5 4 5 { \scriptstyle \pm 0 . 0 7 5 0 }$ </td><td> $0 . 3 1 0 5 { \scriptstyle \pm 0 . 0 7 4 0 }$ </td><td> $\overline { { 0 . 1 9 4 2 _ { \pm 0 . 0 3 1 6 } } }$ </td><td> $\underline { { 0 . 1 1 8 5 _ { \pm 0 . 0 3 6 5 } } }$ </td><td> $0 . 4 4 1 1 { \scriptstyle \pm 0 . 0 7 8 4 }$ </td><td> $\mathbf { 0 . 0 5 3 0 { \scriptstyle \pm 0 . 0 2 0 7 } }$ </td></tr><tr><td rowspan="3">Flickr (Out-of-sample)</td><td>Ind.</td><td> $0 . 3 3 4 3 { \scriptstyle \pm 0 . 0 4 0 6 }$ </td><td> $0 . 3 2 7 7 { \scriptstyle \pm 0 . 0 7 9 5 }$ </td><td> $0 . 4 3 2 1 { \scriptstyle \pm 0 . 0 4 5 3 }$ </td><td> $0 . 2 0 4 5 { \scriptstyle \pm 0 . 0 4 4 6 }$ </td><td> $0 . 1 2 1 1 { \scriptstyle \pm 0 . 0 2 6 2 }$ </td><td> $0 . 0 7 5 5 { \scriptstyle \pm 0 . 0 2 7 0 }$ </td><td> $0 . 2 3 1 6 { \scriptstyle \pm 0 . 0 3 9 9 }$ </td><td> $\mathbf { 0 . 0 5 0 4 _ { \pm 0 . 0 3 4 1 } }$ </td></tr><tr><td>Peer</td><td> $0 . 5 1 7 0 { \scriptstyle \pm 0 . 0 3 6 0 }$ </td><td> $0 . 4 4 5 1 { \scriptstyle \pm 0 . 0 5 7 4 }$ </td><td> $0 . 5 6 5 7 { \scriptstyle \pm 0 . 0 2 6 3 }$ </td><td> $0 . 0 5 4 3 { \scriptstyle \pm 0 . 0 1 6 9 }$ </td><td> $0 . 0 5 4 7 { \scriptstyle \pm 0 . 0 0 7 9 }$ </td><td> $\overline { { 0 . 0 9 8 0 { \scriptstyle \pm 0 . 0 3 1 0 } } }$ </td><td> $0 . 1 0 0 5 { \scriptstyle \pm 0 . 0 2 5 5 }$ </td><td> $\mathbf { 0 . 0 2 1 9 { \scriptstyle \pm 0 . 0 0 8 8 } }$ </td></tr><tr><td>Total</td><td> $0 . 7 6 2 5 { \scriptstyle \pm 0 . 0 7 6 7 }$ </td><td> $0 . 6 2 0 5 { \scriptstyle \pm 0 . 0 9 5 4 }$ </td><td> $0 . 7 0 7 0 { \scriptstyle \pm 0 . 0 6 9 1 }$ </td><td> $\overline { { 0 . 2 8 1 3 _ { \pm 0 . 0 6 4 3 } } }$ </td><td> $0 . 1 9 3 0 { \scriptstyle \pm 0 . 0 3 1 4 }$ </td><td> $0 . 1 2 0 3 _ { \pm 0 . 0 3 9 5 }$ </td><td> $0 . 4 4 0 0 { \scriptstyle \pm 0 . 0 7 6 9 }$ </td><td> $\mathbf { 0 . 0 5 3 1 { \scriptstyle \pm 0 . 0 2 0 1 } }$ </td></tr></table>

TABLE IV

<sub>CAUSAL</sub> <sub>EFFECT</sub> <sub>ESTIMATION</sub> <sub>ON</sub> <sub>THE</sub> <sub>H</sub> <sub>E</sub> <sub>T</sub> <sub>E\_Z</sub> <sub>DATASETS</sub> <sub>MEASURED</sub> <sub>BY</sub> √<sub>ϵPEHE.</sub>
<table><tr><td>Datasets</td><td>Effects</td><td>TARNet+Z</td><td> $\mathrm { C F R + Z }$ </td><td>NetDeconf+Z</td><td>NetEst</td><td>RRNet</td><td>TNet†</td><td>gDIS</td><td>NEDNet</td></tr><tr><td>BC</td><td>Ind.</td><td> $0 . 3 7 5 8 { \scriptstyle \pm 0 . 0 8 9 6 }$ </td><td> $0 . 2 4 8 3 _ { \pm 0 . 0 7 7 0 }$ </td><td> $0 . 4 5 7 1 { \scriptstyle \pm 0 . 0 6 7 2 }$ </td><td> $0 . 1 5 8 3 _ { \pm 0 . 0 4 4 6 }$ </td><td> $0 . 1 8 4 6 { \scriptstyle \pm 0 . 0 5 0 6 }$ </td><td> $0 . 0 8 2 6 { \scriptstyle \pm 0 . 0 3 5 3 }$ </td><td> $0 . 2 3 2 7 { \scriptstyle \pm 0 . 0 9 2 5 }$ </td><td> $\mathbf { 0 . 0 4 8 3 _ { \pm 0 . 0 3 0 8 } }$ </td></tr><tr><td>(Within-sample)</td><td>Peer</td><td> $0 . 8 7 3 8 { \scriptstyle \pm 0 . 0 9 0 7 }$ </td><td> $0 . 7 2 1 0 { \scriptstyle \pm 0 . 1 0 9 5 }$ </td><td> $0 . 8 7 6 3 _ { \pm 0 . 1 0 4 0 }$ </td><td> $0 . 1 1 8 0 { \scriptstyle \pm 0 . 0 2 7 6 }$ </td><td> $0 . 0 6 6 5 { \scriptstyle \pm 0 . 0 1 0 6 }$ </td><td> $\overline { { 0 . 1 4 0 8 _ { \pm 0 . 0 3 6 7 } } }$ </td><td> $0 . 1 8 8 1 { \scriptstyle \pm 0 . 0 5 2 3 }$ </td><td> $\mathbf { 0 . 0 3 1 6 _ { \pm 0 . 0 0 7 2 } }$ </td></tr><tr><td></td><td>Total</td><td> $1 . 3 5 9 1 { \scriptstyle \pm 0 . 2 0 5 0 }$ </td><td> $0 . 9 2 4 3 { \scriptstyle \pm 0 . 1 6 7 5 }$ </td><td> $1 . 1 5 9 2 { \scriptstyle \pm 0 . 2 4 1 0 }$ </td><td> $0 . 3 7 0 3 { \scriptstyle \pm 0 . 0 9 2 2 }$ </td><td> $\overline { { 0 . 3 4 8 6 { \scriptstyle \pm 0 . 0 9 9 5 } } }$ </td><td> $\underline { { 0 . 1 7 9 3 _ { \pm 0 . 0 6 6 9 } } }$ </td><td> $0 . 4 8 4 3 { \scriptstyle \pm 0 . 1 3 7 9 }$ </td><td> $\mathbf { 0 . 1 1 5 5 { \scriptstyle \pm 0 . 0 2 4 2 } }$ </td></tr><tr><td>BC</td><td>Ind.</td><td> $0 . 3 8 3 7 { \scriptstyle \pm 0 . 0 9 4 3 }$ </td><td> $0 . 2 4 4 8 { \scriptstyle \pm 0 . 0 7 5 3 }$ </td><td> $0 . 4 5 1 0 { \scriptstyle \pm 0 . 0 6 1 8 }$ </td><td> $0 . 1 6 0 2 _ { \pm 0 . 0 4 8 0 }$ </td><td> $0 . 1 8 4 1 { \scriptstyle \pm 0 . 0 5 0 4 }$ </td><td> $0 . 0 8 4 9 _ { \pm 0 . 0 3 6 0 }$ </td><td> $0 . 2 3 1 7 { \scriptstyle \pm 0 . 0 9 3 6 }$ </td><td> $\mathbf { 0 . 0 5 1 6 _ { \pm 0 . 0 3 2 9 } }$ </td></tr><tr><td>(Out-of-sample)</td><td>Peer</td><td> $0 . 8 7 2 2 { \scriptstyle \pm 0 . 0 9 1 3 }$ </td><td> $0 . 7 2 0 1 { \scriptstyle \pm 0 . 1 1 0 3 }$ </td><td> $0 . 8 7 5 3 { \scriptstyle \pm 0 . 1 0 4 4 }$ </td><td> $0 . 1 1 6 9 _ { \pm 0 . 0 3 2 3 }$ </td><td> $0 . 0 6 5 5 { \scriptstyle \pm 0 . 0 1 0 0 }$ </td><td> $\overline { { 0 . 1 4 3 7 _ { \pm 0 . 0 3 6 9 } } }$ </td><td> $0 . 1 8 8 5 { \scriptstyle \pm 0 . 0 5 1 3 }$ </td><td> $\mathbf { 0 . 0 3 2 1 { \scriptstyle \pm 0 . 0 0 6 5 } }$ </td></tr><tr><td></td><td>Total</td><td> $1 . 3 4 5 3 { \scriptstyle \pm 0 . 2 1 0 3 }$ </td><td> $0 . 9 1 7 9 { \scriptstyle \pm 0 . 1 6 5 8 }$ </td><td> $1 . 1 4 7 1 { \scriptstyle \pm 0 . 2 3 5 8 }$ </td><td> $0 . 3 6 8 2 { \scriptstyle \pm 0 . 0 9 4 9 }$ </td><td> $0 . 3 4 9 3 { \scriptstyle \pm 0 . 1 0 3 5 }$ </td><td> $\underline { { 0 . 1 7 9 4 } } \pm 0 . 0 7 3 9$ </td><td> $0 . 4 8 8 7 { \scriptstyle \pm 0 . 1 4 2 3 }$ </td><td> $\mathbf { 0 . 1 1 9 6 { \scriptstyle \pm 0 . 0 2 4 2 } }$ </td></tr><tr><td>Flickr</td><td>Ind.</td><td> $0 . 4 1 7 5 { \scriptstyle \pm 0 . 1 4 9 1 }$ </td><td>0.2877±0.0998</td><td> $0 . 5 2 5 4 { \scriptstyle \pm 0 . 1 5 0 4 }$ </td><td> $0 . 2 4 1 7 { \scriptstyle \pm 0 . 1 2 7 8 }$ </td><td>0.1586±0.1025</td><td> $0 . 1 2 7 1 { \scriptstyle \pm 0 . 0 3 7 6 }$ </td><td> $0 . 3 1 3 7 { \scriptstyle \pm 0 . 0 9 7 2 }$ </td><td>0.0800±0.0717</td></tr><tr><td>(Within-sample)</td><td>Peer</td><td> $0 . 8 6 9 3 { \scriptstyle \pm 0 . 0 2 7 0 }$ </td><td> $0 . 6 8 7 0 { \scriptstyle \pm 0 . 0 2 5 5 }$ </td><td> $0 . 8 7 4 7 { \scriptstyle \pm 0 . 0 3 2 0 }$ </td><td> $0 . 1 1 9 9 { \scriptstyle \pm 0 . 0 7 9 1 }$ </td><td> $0 . 0 7 9 3 _ { \pm 0 . 0 2 6 4 }$ </td><td> $\overline { { 0 . 1 8 8 6 _ { \pm 0 . 0 8 8 4 } } }$ </td><td> $0 . 0 8 1 3 { \scriptstyle \pm 0 . 0 2 1 8 }$ </td><td> $\mathbf { 0 . 0 2 8 5 _ { \pm 0 . 0 0 3 8 } }$ </td></tr><tr><td></td><td>Total</td><td> $1 . 4 2 6 6 { \scriptstyle \pm 0 . 2 5 2 5 }$ </td><td> $1 . 0 0 1 3 { \scriptstyle \pm 0 . 2 0 8 7 }$ </td><td> $1 . 2 4 6 6 { \scriptstyle \pm 0 . 2 3 3 0 }$ </td><td> $0 . 4 6 6 8 { \scriptstyle \pm 0 . 2 1 4 1 }$ </td><td> $0 . 2 8 1 2 { \scriptstyle \pm 0 . 1 3 2 7 }$ </td><td> $\underline { { 0 . 2 5 7 6 { \scriptstyle \pm 0 . 0 3 5 4 } } }$ </td><td> $0 . 4 6 9 0 { \scriptstyle \pm 0 . 0 8 1 5 }$ </td><td> $\mathbf { 0 . 1 3 0 7 { \scriptstyle \pm 0 . 0 5 1 6 } }$ </td></tr><tr><td>Flickr</td><td>Ind.</td><td> $0 . 4 0 8 8 { \scriptstyle \pm 0 . 1 5 4 3 }$ </td><td> $0 . 2 8 6 6 { \scriptstyle \pm 0 . 0 9 7 2 }$ </td><td> $0 . 4 9 7 4 { \scriptstyle \pm 0 . 1 4 1 8 }$ </td><td> $0 . 2 3 0 5 { \scriptstyle \pm 0 . 1 2 5 0 }$ </td><td>0.1601±0.1019</td><td> $0 . 1 2 5 9 { \scriptstyle \pm 0 . 0 3 7 3 }$ </td><td> $0 . 3 0 8 2 { \scriptstyle \pm 0 . 0 9 2 7 }$ </td><td> $\mathbf { 0 . 0 7 9 7 { \scriptstyle \pm 0 . 0 7 3 4 } }$ </td></tr><tr><td>(Out-of-sample)</td><td>Peer</td><td> $0 . 8 5 2 0 { \scriptstyle \pm 0 . 0 2 1 9 }$ </td><td> $0 . 6 8 5 9 { \scriptstyle \pm 0 . 0 2 2 2 }$ </td><td> $0 . 8 7 3 1 { \scriptstyle \pm 0 . 0 3 1 1 }$ </td><td> $0 . 1 0 2 3 { \scriptstyle \pm 0 . 0 6 7 7 }$ </td><td> $0 . 0 8 1 2 _ { \pm 0 . 0 2 2 6 }$ </td><td> $\overline { { 0 . 1 8 9 5 _ { \pm 0 . 0 8 8 0 } } }$ </td><td> $0 . 0 8 4 5 _ { \pm 0 . 0 2 2 7 }$ </td><td> $\mathbf { 0 . 0 3 0 9 _ { \pm 0 . 0 0 3 8 } }$ </td></tr><tr><td></td><td>Total</td><td> $1 . 3 4 2 4 { \scriptstyle \pm 0 . 2 5 3 4 }$ </td><td> $0 . 9 9 7 9 { \scriptstyle \pm 0 . 1 9 8 0 }$ </td><td> $1 . 2 1 1 9 { \scriptstyle \pm 0 . 2 1 3 2 }$ </td><td> $0 . 4 1 8 6 { \scriptstyle \pm 0 . 1 8 0 3 }$ </td><td> $\overline { { 0 . 2 8 1 3 \pm 0 . 1 2 2 5 } }$ </td><td> $0 . 2 5 9 5 { \scriptstyle \pm 0 . 0 4 0 4 }$ </td><td> $0 . 4 5 7 3 { \scriptstyle \pm 0 . 0 7 7 0 }$ </td><td> $\mathbf { 0 . 1 3 3 0 { \scriptstyle \pm 0 . 0 5 2 8 } }$ </td></tr></table>

NEDNet reduces the estimation error by 33.2%–59.7%. The improvements are consistent across both datasets and evaluation settings, covering individual, peer, and total effects. These results demonstrate the effectiveness of NEDNet in estimating heterogeneous individual-treatment effects while maintaining accurate estimation of neighborhood-exposure and total effects.

## E. Effect Estimation under Heterogeneous Responses

The results on the hete\_z setting further examine the estimation performance when both individual-treatment and neighborhood-exposure effects exhibit node-dependent heterogeneity. As reported in Table IV, NEDNet again achieves the lowest PEHE in all twelve evaluation combinations, reducing the error by 33.3%–64.1% relative to the best-performing baseline. The consistent improvements across IE, PE, and TE indicate that NEDNet can effectively estimate heterogeneous responses along both intervention dimensions. Moreover, the comparable within-sample and out-of-sample results suggest that its estimation performance is maintained on held-out graph subgraphs.

## VI. CONCLUSION

We further evaluate NEDNet under two heterogeneous outcome-generation settings adopted from TNet [11]. In the hete setting, individual-treatment effects vary across nodes through covariate-dependent response terms, while neighborhood-exposure effects remain homogeneous. The hete\_z setting further introduces node-dependent heterogene ity in neighborhood-exposure effects. The corresponding results are reported in Tables III and IV, respectively. Directly transferring the hyperparameter configuration from homogeneous to heterogeneous settings led to unstable optimization for TNet. We therefore selected its hyperparameters separately for each heterogeneous setting using validation data only. The resulting variants are denoted by TNet<sup>†</sup> in the tables.

Table III shows that NEDNet achieves the lowest PEHE across all twelve dataset–evaluation–effect combinations. Compared with the best-performing baseline in each setting,

This paper has presented NEDNet, a structured conditional outcome model for causal effect estimation under network interference. NEDNet has parameterized the conditional outcome response using a query-invariant pre-treatment baseline and separate pathways for individual-treatment and neighborhoodexposure responses. The path-specific neighborhood contextualization mechanism has further incorporated local pretreatment information into the two pathways. This design has supported potential-outcome prediction under alternative treatment–exposure conditions and the estimation of individual, peer, and total effects.

NEDNet has been evaluated on semi-synthetic benchmarks derived from BlogCatalog and Flickr under homogeneous and covariate-heterogeneous response settings. Across the evaluated comparisons, NEDNet has achieved the lowest mean PEHE among the compared methods for IE, PE, and TE, and has maintained competitive counterfactual outcome prediction performance. The ablation results have further supported the effectiveness of separate response representations and pathspecific encoding and decoding.

Future work will investigate more general response specifications, including joint treatment–exposure interactions, and more flexible exposure mappings to accommodate increasingly complex network interference patterns.

## REFERENCES

[1] T. A. Glass, S. N. Goodman, M. A. Hernan, and J. M. Samet, “Causal´ inference in public health,” Annual review of public health, vol. 34, no. 1, pp. 61–75, 2013.

[2] G. W. Imbens, “Causal inference in the social sciences,” Annual Review of Statistics and Its Application, vol. 11, 2024.

[3] Z. Huang, S. Zhang, D. Cheng, J. Li, L. Liu, G. Lu, and G. Zhang, “Learn ing instrumental variable representation for debiasing in recommender systems,” Neural Networks, p. 107977, 2025.

[4] L. Yao, Z. Chu, S. Li, Y. Li, J. Gao, and A. Zhang, “A survey on causal inference,” ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 15, no. 5, pp. 1–46, 2021.

[5] J. Pearl, “Causal inference in statistics: An overview,” Statistics Surveys, vol. 3, pp. 96–146, 2009.

[6] P. M. Aronow and C. Samii, “Estimating average causal effects under general interference, with application to a social network experiment,” The Annals of Applied Statistics, vol. 11, no. 4, pp. 1912 – 1947, 2017.

[7] M. E. Sobel, “What do randomized studies of housing mobility demonstrate? causal inference in the face of interference,” Journal of the American Statistical Association, vol. 101, no. 476, pp. 1398–1407, 2006.

[8] L. Forastiere, D. Del Prete, and V. L. Sciabolazza, “Causal inference on networks under continuous treatment interference,” Social Networks, vol. 76, pp. 88–111, 2024.

[9] L. Forastiere, E. M. Airoldi, and F. Mealli, “Identification and estimation of treatment and interference effects in observational studies on networks,” Journal of the American Statistical Association, vol. 116, no. 534, pp. 901–918, 2021.

[10] S. Jiang and Y. Sun, “Estimating causal effects on networked observational data via representation learning,” in Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM). ACM, 2022, pp. 4048–4057.

[11] W. Chen, R. Cai, Z. Yang, J. Qiao, Y. Yan, Z. Li, and Z. Hao, “Doubly robust causal effect estimation under networked interference via targeted learning,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 6457–6485.

[12] Q. Huang, J. Ma, J. Li, R. Guo, H. Sun, and Y. Chang, “Modeling interference for individual treatment effect estimation from networked observational data,” ACM Transactions on Knowledge Discovery from Data, vol. 18, no. 3, pp. 1–21, 2023.

[13] R. Cai, Z. Yang, W. Chen, Y. Yan, and Z. Hao, “Generalization bound for estimating causal effects from observational network data,” in Proceedings of the ACM Web Conference (WWW), 2023.

[14] S. Adhikari, S. Medya, and E. Zheleva, “Learning exposure mapping functions for inferring heterogeneous peer effects,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 49 057– 49 083.

[15] X. Du, J. Li, D. Cheng, L. Liu, W. Gao, X. Chen, and Z. Xu, “Telling peer direct effects from indirect effects in observational network data,” in Forty-second International Conference on Machine Learning, 2025.

[16] K. Hirano, G. W. Imbens, and G. Ridder, “Efficient estimation of average treatment effects using the estimated propensity score,” Econometrica, vol. 71, no. 4, pp. 1161–1189, 2003.

[17] D. B. Rubin, “Estimating causal effects of treatments in randomized and nonrandomized studies,” Journal of Educational Psychology, vol. 66, no. 5, pp. 688–701, 1974.

[18] P. R. Rosenbaum and D. B. Rubin, “The central role of the propensity score in observational studies for causal effects,” Biometrika, vol. 70, no. 1, pp. 41–55, 1983.

[19] J. M. Robins, M. A. Hernan, and B. Brumback, “Marginal structural models and causal inference in epidemiology,” pp. 550–560, 2000.

[20] H. Bang and J. M. Robins, “Doubly robust estimation in missing data and causal inference models,” Biometrics, vol. 61, no. 4, pp. 962–973, 2005.

[21] D. Cheng, J. Li, L. Liu, Z. Xu, W. Zhang, J. Liu, and T. D. Le, “Disentangled representation learning for causal inference with instruments,” IEEE Transactions on Neural Networks and Learning Systems, 2024.

[22] D. Cheng, Y. Xie, Z. Xu, J. Li, L. Liu, J. Liu, Y. Zhang, and Z. Feng, “Disentangled latent representation learning for tackling the confounding m-bias problem in causal inference,” in 2023 IEEE International Conference on Data Mining (ICDM). IEEE, 2023, pp. 51–60.

[23] R. Cai, Y. Wu, X. Huang, W. Chen, T. Z. Fu, and Z. Hao, “Granger causal representation learning for groups of time series,” Science China Information Sciences, vol. 67, no. 5, p. 152103, 2024.

[24] D. Cheng, J. Li, L. Liu, K. Yu, T. D. Le, and J. Liu, “Toward unique and unbiased causal effect estimation from data with hidden variables,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 9, pp. 6108–6120, 2022.

[25] U. Shalit, F. D. Johansson, and D. Sontag, “Estimating individual treatment effect: generalization bounds and algorithms,” in International conference on machine learning. PMLR, 2017, pp. 3076–3085.

[26] M. G. Hudgens and M. E. Halloran, “Toward causal inference with interference,” Journal of the american statistical association, vol. 103, no. 482, pp. 832–842, 2008.

[27] E. J. T. Tchetgen and T. J. VanderWeele, “On causal inference in the presence of interference,” Statistical methods in medical research, vol. 21, no. 1, pp. 55–75, 2012.

[28] L. Forastiere, F. Mealli, A. Wu, and E. M. Airoldi, “Estimating causal effects under network interference with bayesian generalized propensity scores,” Journal of Machine Learning Research, vol. 23, no. 289, pp. 1–61, 2022.

[29] S. Fan, X. Wang, C. Shi, K. Kuang, N. Liu, and B. Wang, “Debiased graph neural networks with agnostic label selection bias,” IEEE transactions on neural networks and learning systems, vol. 35, no. 4, pp. 4411–4422, 2022.

[30] V. Veitch, Y. Wang, and D. Blei, “Using embeddings to correct for unobserved confounding in networks,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[31] R. Guo, J. Li, and H. Liu, “Learning individual causal effects from networked observational data,” in Proceedings of the 13th International Conference on Web Search and Data Mining (WSDM). ACM, 2020, pp. 232–240.

[32] Z. Chu, S. L. Rathbun, and S. Li, “Graph infomax adversarial learning for treatment effect estimation with networked observational data,” in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, 2021, pp. 176–184.

[33] Y. Ma and V. Tresp, “Causal inference under networked interference and intervention policy enhancement,” in International Conference on Artificial Intelligence and Statistics. PMLR, 2021, pp. 3700–3708.

[34] A. Wu, H. Qiu, Z. Chen, Z. Li, R. Xiong, F. Wu, and K. Zhang, “Causal graph transformer for treatment effect estimation under unknown interference,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 99 579–99 608.