# Mechanistic Reaction Prediction via Discrete Flow Matching on Graph-Structured Electron Occupation

Xuan-Vu Nguyen<sup>1</sup>, Octavian Susanu<sup>1</sup>, Daniel Armstrong<sup>1</sup>, Philippe Schwaller<sup>1,2</sup> <sup>1</sup>École Polytechnique Fédérale de Lausanne (EPFL) <sup>2</sup>National Centre of Competence in Research (NCCR) Catalysis {nguyen.nguyen,philippe.schwaller}@epfl.ch

## Abstract

Chemical reactions are fundamentally transformations in electron space, yet most machine learning approaches model them either through de novo generation of product molecules or through heuristic graph edits that operate directly on molecular topology. We introduce MAELLE (MechAnistic Edit fLow-matching on eLectron rEarrangements), which instead models reactions as discrete flow matching over electron occupation vectors. Concretely, we formulate the reactant-to-product mapping as a Continuous-time Markov Chain (CTMC) over the graph-structured integer-valued electron occupation space defined on all bonding, non-bonding, and hydrogen sites. To construct the intermediate edit trajectories, we generalize the discrete flow matching mixture path to discrete electron rearrangements using Optimal Transport, yielding a sequence of mechanistically interpretable edit moves without requiring elementary step annotations. MAELLE achieves competitive performance on the USPTO-480K benchmark compared with leading reaction prediction models. Beyond in-distribution accuracy, we evaluate robustness across two out-of-distribution settings - structural complexity and reaction type - and find that MAELLE maintains strong performance where existing methods degrade. Finally, because the learned flow operates over the full electron redistribution, MAELLE naturally recovers mechanistic trajectories that align with known chemistry and can predict side products of a reaction.

## 1 Introduction

Predicting the outcome of a chemical reaction is one of the central problems at the interface of chemistry and machine learning, with consequences for drug discovery, materials design, and the broader question of how to realize a target molecule once it has been proposed [1, 2]. The space of possible reactions is larger than the already intractable space of drug-like molecules, and a single retrosynthetic plan may chain together a dozen or more steps, each of which must succeed for the route to be viable [3]. Reliable prediction, with calibrated uncertainty, is therefore a prerequisite for automating any meaningful portion of synthetic chemistry.

Reaction prediction has cycled through a succession of representations, each reflecting the dominant ML paradigm of its moment: fixed molecular fingerprints classified by similarity [4, 5, 6, 7]; SMILESto-SMILES translation with sequence and Transformer models [8, 9]; hybrid graph-encoder/SMILESdecoder architectures [10, 11]; and, most recently, generative graph-edit approaches that produce the product by editing the reactant graph directly [12, 13].

All of these approaches treat reactions as opaque input-to-output mappings, offering no account of the underlying transformation, while chemists reason via arrow-pushing diagrams that track electron flow between donor and acceptor sites [14]. As black-box predictors, they are exposed to spurious shortcuts in USPTO – patent-level stylistic signatures [15] and substrate-level cues (e.g., a single free hydroxyl among protected ones telegraphing an alcohol-selective transformation). Top-k accuracy on standard benchmarks therefore conflates first-principles reactivity reasoning with exploitation of such shortcuts, making out-of-distribution evaluation a more faithful test [16, 17, 18].

![](images/6ec6f99f0b571350e860e64fbbcdd21146b9cac324f9ae3ab4d1485bc75b28dc.jpg)  
Figure 1: Conceptual overview of MAELLE: a. MAELLE’s sampling process with mechanism-like trajectories that model the transition from reactants to products as CTMC. b. The three elementary actions on electrons, FLOW, DEL, and ADD, demonstrated on an oxidation and a reduction reactions. At training time: c. Electron actions are interpolated using OT; d. interpolant graphs are sampled by sampling a subset of electron moves and applying them to the reactant graphs. e. MAELLE model architecture and learning objective.

A few recent works have sought to incorporate mechanistic reasoning into reaction prediction. Early work by Kayala et al. [19] and Kayala and Baldi [20] pioneered the prediction of reactions at the mechanistic level by learning electron-flow patterns, decomposing the problem into identifying reactive sites and predicting electron sources and sinks. Bradshaw et al. [21] later extended this direction with a GNN-based model that predicts elementary steps for reactions with linear electron flow (LEF) topology, though this restricts coverage to a subset of reaction mechanisms. More recently, Joung et al. [22] model electron redistribution via continuous flow matching over bond-electron (BE) matrices, but require mechanistic trajectories imputed from expert-curated templates [23, 24, 25], limiting scalability to new reaction types.

In this work, we introduce MAELLE (MechAnistic Edit fLow-matching on eLectron rEarrangements), which models reactions as discrete flow matching (DFM) over integer-valued electron occupation vectors defined on the molecular graph (Figure 1). MAELLE formulates reaction prediction as a Continuous-time Markov Chain (CTMC) in the electron space: given a reactant, the model learns a rate function that drives the electron occupation from the reactant state to the product state through a sequence of sparse, interpretable edit moves, without requiring elementary step annotations. The intermediate trajectory is constructed via optimal transport over electron sites, and the mixture path factorizes over an auxiliary move-state space, enabling tractable training through the discrete flow matching framework.

Our contributions are: (i) we generalize DFM to graph-structured electron spaces with three chemistryaware edit operations (flow, deletion, addition), and lift the problem to an auxiliary move-state space derived from optimal transport over electron occupations, yielding a factorized mixture path with tractable training and no need for elementary-step labels; (ii) on USPTO-480K we are competitive with the best SMILES- and graph-edit baselines, and on out-of-distribution splits along molecular weight and reaction type we outperform all baselines; (iii) framing reaction prediction generatively gives MAELLE calibrated, ranked product predictions, enabling side-product prediction validated by LLM-as-judge plausibility.

Table 1: Comparison of reaction prediction approaches.
<table><tr><td>Method</td><td>Input</td><td>Output</td><td>Mech. step label-free</td><td>Directional elec. re-dist.</td><td>Broad coverage</td><td>Atom conservative</td></tr><tr><td>Molecular Transformer [9]</td><td>SMILES</td><td>SMILES</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Graph2SMILES [11]</td><td>SMILES</td><td>SMILES</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Chemformer [26]</td><td>SMILES</td><td>SMILES</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>MEGAN [12]</td><td>Graph</td><td>Graph edits</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>GTPN [13]</td><td>Graph</td><td>Graph edits</td><td>√</td><td>x</td><td>√</td><td>√</td></tr><tr><td>NERF [27]</td><td>Graph</td><td>Electron re-dist.</td><td>√</td><td>x</td><td>√</td><td>√</td></tr><tr><td>FlowER [22]</td><td>Graph</td><td>Electron re-dist.</td><td>x†</td><td>√</td><td>x†</td><td>√</td></tr><tr><td>ELECTRO [21]</td><td>Graph</td><td>Electron re-dist.</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>MAELLE (ours)</td><td>Graph</td><td>Electron re-dist.</td><td>√</td><td>√</td><td>√</td><td>√§</td></tr></table>

<sup>†</sup>FlowER covers 86 expert-curated reaction types and requires a mechanistic dataset with imputed elementary steps. <sup>‡</sup>ELECTRO is restricted to reactions with linear electron flow topology.  
<sup>§</sup>MAELLE is heavy-atom conservative because protonation and deprotonation are simplified.

## 2 Related Work

## 2.1 Translation-based reaction prediction

Early work by Schwaller et al. [9] cast reaction prediction as translation from the SMILES string of the reactants to that of the products. Because a molecule admits many valid SMILES strings, these models are typically trained with SMILES augmentation – multiple strings per molecule [9, 26] – which is data-inefficient. Graph2SMILES [11] partially addresses this by encoding the reactants as a 2D graph, removing the need to augment the input, while still decoding the product as a SMILES string with a Transformer decoder.

Sequence representations such as SMILES [28] and SELFIES [29] make products easy to generate with well-established autoregressive text models, but they come at a cost: the model generates the product de novo, with no structural link to the reactants. This has three consequences. First, generation is unconstrained, so there is no guarantee that the predicted reaction is atom-conservative – a critical chemical constraint. Second, because the reactant-product correspondence is not explicit, recovering what actually changed requires a separate atom-mapping tool [30]; and even when a prediction appears atom-conserving, nothing ensures a valid mechanistic pathway connects reactants to products, so a chemically sound transformation and a hallucinated one can look alike. Third, generating the entire product from scratch makes these models degrade on molecules whose complexity falls outside the training domain, even when the underlying chemistry is simple [16, 17, 18].

## 2.2 Reactions as graph edits

To build the chemical change directly into the model, another line of work predicts reactions as topological transformations of the reactant graph, and atom conservation can be enforced through the choice of edit operations. MEGAN [12] uses EditAtom, EditBond, AddAtom, and AddBenzene, explicitly modeling a reaction as a sequence of graph edits; operations such as EditAtom, AddAtom, and AddBenzene break atom conservation, though one can restrict the operation set to bond-only edits to preserve it, as in GTPN [13].

Graph edits make the transformation explicit and give atom mapping for free, and atom conservation is achievable – but the edits are purely topological, so no mechanistic pathway is guaranteed to connect the two endpoints. Bond formation is the clearest case: a graph-edit model simply inserts a bond between two atoms, whereas chemically the electrons forming that bond are drawn from elsewhere in the molecule.

## 2.3 Reactions as electron redistribution

At a level deeper than topological graph edits, a reaction is a redistribution of electrons: a sequence of elementary steps in which electrons flow from a source to a sink, described in chemistry by arrow-pushing notation. Modeling reactions this way is informative about what actually happens, but capturing the elementary steps explicitly has traditionally required expert-labeled mechanisms or heavy physical calculation. Kayala et al. [19] assemble a dataset of 1,630 reactions with 2,989 mechanistic steps and train a model to rank orbital reactivity for a single elementary step, wrapping it in a recursive loop to reach the product. In the same spirit, Joung et al. [22] scale the mechanistic dataset to 290K reactions and over 2 million elementary steps, and train a flow-matching model over the bond-electron matrix. Both are informative but depend on expert-labeled elementary steps, and are therefore less scalable than the end-to-end approaches above.

ELECTRO [21] removes the labeling requirement, interpolating electron moves directly from end-toend reactant-product pairs. However, it is restricted to reactions with linear electron flow topology – those whose mechanism has a well-defined start and end to the electron flow – and so does not cover cyclic mechanisms. NERF [27] also avoids mechanism labels but takes a non-autoregressive, one-shot view: it predicts, for each bond, the net number of electron pairs added or removed between reactants and products. Because it models only the change at each bond and not a transfer between sites, the direction of electron flow is left implicit – the prediction records that a bond gained or lost electrons, but not where those electrons came from or went. It also models bond electrons only, excluding lone pairs and other valence electrons from the representation.

MAELLE models reactions as directional electron redistribution while guaranteeing heavy-atom conservation (Table 1). Like ELECTRO, it needs no elementary-step labels, but it is not limited to linear flow: our FLOW operation moves an electron pair from a source site to a sink site, making the direction of flow explicit – answering where electrons come from and where they go – and composing freely into cyclic as well as linear mechanisms. In place of mechanism labels, we recover the net electron moves with Optimal Transport (Section 4.3), sample interpolant graphs between reactants and products, and train a discrete flow-matching model over these electron edits.

## 2.4 Discrete generative modeling on graphs

From the standpoint of generative modeling, discrete generative models on graphs typically cast generation as iterative substitution of node and edge types under a denoising diffusion or flowmatching process that maps from a source distribution that is easy to sample, such as a uniform distribution, to the data distribution. DiGress [31] and DeFoG [32] generate graphs by employing discrete denoising diffusion processes [33, 34] and discrete flow matching [35], respectively.

More related to our setting, RetroBridge [36] tackles the problem of retrosynthesis, i.e., predicting possible precursors given the products, in a reversed setting compared to our work. The model learns a bridge between two coupled distributions of products and reactants via Markov bridge [37].

Both lines model generation as resampling of categorical node/edge types — from a noise prior (DiGress, DeFoG) or a coupled endpoint distribution (RetroBridge). MAELLE instead operates over integer-valued electron occupation, where transitions are electron transfers (flow/add/delete) rather than type substitutions; this makes the process atom-conserving by construction and yields mechanistically interpretable trajectories.

## 3 Background

## 3.1 Discrete Flow Matching

Let T be a vocabulary and $\textstyle { \mathcal { X } } = { \mathcal { T } } ^ { * } = \bigcup _ { n = 0 } ^ { N } { \mathcal { T } } ^ { n }$ the space of token sequences of length at most $N$ . Discrete flow matching (DFM) [35, 38] learns a Continuous-time Markov Chain (CTMC) that transports a source distribution $p _ { 0 }$ at $t = 0$ to a target $p _ { 1 }$ at $t = 1$ . As in the continuous case, the central construction is a conditional mixture path $p _ { t } ( x \mid x _ { 0 } , x _ { 1 } )$ with boundary conditions $p _ { 0 } ( x \mid x _ { 0 } , x _ { 1 } ) = \delta _ { x _ { 0 } } ( x )$ and $p _ { 1 } ( x \mid x _ { 0 } , x _ { 1 } ) = \delta _ { x _ { 1 } } ( x )$ ; the endpoints $( x _ { 0 } , x _ { 1 } )$ are drawn from a coupling $\pi ( x _ { 0 } , x _ { 1 } )$ whose marginals are the source and target, and marginalizing the conditional path over π gives the marginal path $p _ { t }$ that interpolates $p _ { 0 }$ and $p _ { 1 }$

For sequences of a common length $n ,$ the standard choice is the factorized, token-wise mixture path

$$
p _ { t } ( x ^ { i } \mid x _ { 0 } , x _ { 1 } ) = ( 1 - \kappa ( t ) ) \delta _ { x _ { 0 } ^ { i } } ( x ^ { i } ) + \kappa ( t ) \delta _ { x _ { 1 } ^ { i } } ( x ^ { i } ) , \qquad p _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) = \prod _ { i = 1 } ^ { n } p _ { t } ( x ^ { i } \mid x _ { 0 } , x _ { 1 } ) ,\tag{1}
$$

where the schedule $\kappa : [ 0 , 1 ]  [ 0 , 1 ]$ is monotone with $\kappa ( 0 ) = 0 , \kappa ( 1 ) = 1$ . Each token independently switches from its source value $x _ { 0 } ^ { i }$ to its target value $x _ { 1 } ^ { \ i }$ , and does so by time t with probability $\kappa ( t )$ . The CTMC that generates (1) has the token-wise marginal rate

$$
u _ { t } ( x ^ { i } \mid x _ { t } ) = \frac { \kappa ^ { \prime } ( t ) } { 1 - \kappa ( t ) } \big [ p _ { 1 \mid t } ( x ^ { i } \mid x _ { t } ) - \delta _ { x _ { t } ^ { i } } ( x ^ { i } ) \big ] ,\tag{2}
$$

where $p _ { 1 | t } ( x ^ { i } \mid x _ { t } )$ is the model’s posterior over the target token at position i. Training therefore reduces to predicting $p _ { 1 | t }$ per position, and the coefficient $\kappa ^ { \prime } ( t ) / ( 1 - \kappa ( t ) )$ is the rate at which a position that has not yet reached its target switches to it.

## 3.2 Edit Flows

The token-wise path (1) assumes a fixed, common length and a position-to-position correspondence between $x _ { 0 }$ and $x _ { 1 }$ . Edit Flows [39] remove this assumption, allowing sequences of differing length related by edit operations – insertion, deletion, and substitution – rather than position-wise substitution alone. To recover a factorized path in this setting, the construction is lifted to an auxiliary space of aligned (padded) sequences $\mathcal { Z } = \dot { ( } T \cup \{ \epsilon \} ) ^ { N }$ , where ϵ is a padding symbol, together with the map $f _ { \mathrm { r m - b l a n k s } } : \mathcal { Z }  \mathcal { X }$ that deletes all padding. Given endpoints $x _ { 0 } , x _ { 1 }$ and a chosen alignment $z _ { 0 } , z _ { 1 } \in \mathcal { Z }$ with $f _ { \mathrm { r m - b l a n k s } } ( z _ { j } ) = x _ { j }$ , the mixture path is defined on the augmented space $\mathcal { X } \times \mathcal { Z }$ as

$$
p _ { t } ( x , z \mid z _ { 0 } , z _ { 1 } ) = p _ { t } ( z \mid z _ { 0 } , z _ { 1 } ) \delta _ { f _ { \mathrm { r m } - \epsilon } ( z ) } ( x ) ,\tag{3}
$$

where $p _ { t } { \big ( } z { \mathrm { ~ \big | ~ } } z _ { 0 } , z _ { 1 } { \big ) }$ is the token-wise path (1) applied on the aligned sequences. The data-space path is the pushforward of the auxiliary path through $f _ { \mathrm { r m - b l a n k s } } \mathrm { : }$ the factorization lives in ${ \mathcal { Z } } ,$ , while the observed sequence $x = f _ { \mathrm { r m - b l a n k s } } ( z )$ can have variable length. Because each aligned position independently takes its source or target value, this is equivalent to sampling edit operations to apply to $x _ { 0 }$ . We adopt this view in this paper, and show that it is equivalent to the token-wise sampling in Section 4.4 (Proposition 1).

## 4 Pseudo mechanistic reaction prediction with MAELLE

## 4.1 Electron occupation as the main source of change

We represent a molecule as an augmented graph $\mathcal { G } = ( \mathcal { V } , \mathcal { A } , \mathcal { E } , o )$ , where V is the set of heavy atoms, A is the heavy-atom adjacency, and E is a fixed set of electron sites comprising bonding sites $e _ { i j }$ $( i \neq j )$ , lone-pair sites $e _ { i i }$ , and per-atom hydrogen sites $e _ { i } ^ { H }$ (hydrogens on the same heavy atom are topologically equivalent and pooled into one site [40]). The occupation vector $o \in \mathbb { Z } ^ { | \mathcal { E } | }$ counts electron pairs at each site, including virtual ones with $o ^ { e } = 0$ . See Appendix B for the formal definitions and an illustration on ethanol.

A chemical reaction maps reactants ${ \mathcal { G } } _ { x }$ to products $\mathcal { G } _ { y }$ by redistributing electrons, i.e., changing o. The atom set $\nu$ is preserved. Moreover, the heavy-atom adjacency can be recovered deterministically from the occupation through

$$
a _ { i j } = \mathbb { 1 } [ o ^ { e _ { i j } } > 0 ] ,\tag{4}
$$

so $p ( \mathcal { G } _ { y } \mid \mathcal { G } _ { x } ) = p ( o _ { y } \mid \mathcal { G } _ { x } )$ . Thus, we could model a reaction by acting entirely on the vector o.

## 4.2 Edit operations on electron occupation

In this section, we describe the electron edit operations that enacts on $o .$ Let $\mathbf { 1 } _ { e } \in \{ 0 , 1 \} ^ { | \varepsilon | }$ denote the one-hot vector with a single nonzero entry at the position corresponding to site $e \in { \mathcal { E } }$ . We define the following actions on o (Figure 1b):

FLOW Transfer one electron pair from site $e _ { \mathrm { s o u r c e } }$ to site $e _ { \mathrm { s i n k } }$ , requiring $o ( e _ { \mathrm { s o u r c e } } ) > 0 \mathrm { : }$

$$
\mathrm { F L O W } ( e _ { \mathrm { s o u r c e } }  e _ { \mathrm { s i n k } } ) ( o ) = o - \mathbf { 1 } _ { e _ { \mathrm { s o u r c e } } } + \mathbf { 1 } _ { e _ { \mathrm { s i n k } } }\tag{5}
$$

DEL Remove one electron pair from site $e ,$ requiring $o ^ { e } > 0 ;$

$$
\mathrm { D E L } ( e ) ( o ) = o - \mathbf { 1 } _ { e }\tag{6}
$$

ADD Add one electron pair to site e:

$$
\mathbf { A D D } ( e ) ( o ) = o + \mathbf { 1 } _ { e }\tag{7}
$$

The sparsity of these three operations enables us to define a tractable distribution of the intermediate states between the occupation vectors of the reactants and products $o _ { x }$ and $o _ { y }$

## 4.3 Interpolation of pseudo mechanistic steps via Optimal Transport

To obtain a set of moves that transform $o _ { x }$ into $o _ { y } ,$ we solve a balanced integer Optimal Transport problem over the electron sites. To handle reactions where the total electron count is not conserved, we augment the site set with a virtual sink/source $e _ { \emptyset }$ , yielding $\bar { \mathcal { E } } = \mathcal { E } \cup \{ e _ { \emptyset } \}$ with augmented occupations $\bar { o } _ { x } , \bar { o } _ { y }$ . The transport cost between two sites is the minimum shortest-path distance between their constituent atoms on either the reactant or product graph, with a large penalty $c _ { \theta }$ for transport to or from $e _ { \emptyset }$ . The optimal transport plan could be solved with the simplex algorithm, where the design of the objective, marginal constraints, and cost matrix can be found in Appendix C. This results in the optimal transport matrix $T ^ { * } \in \mathbb { Z } ^ { | \bar { \mathcal { E } } | \times | \bar { \mathcal { E } } | }$

The move set as a deterministic alignment. Each non-zero off-diagonal entry $T _ { e , e ^ { \prime } } ^ { * } > 0$ is translated to an electron action m ∈ {FLOW, DEL, ADD} (Eqs. 5–7):

$$
m = \left\{ \begin{array} { l l } { \mathrm { F L O W } ( e \to e ^ { \prime } ) \quad \mathrm { i f } \quad e \neq e _ { \emptyset } , e ^ { \prime } \neq e _ { \emptyset } } \\ { \mathrm { D E L } ( e ) \quad \mathrm { i f } \quad e \neq e _ { \emptyset } , e ^ { \prime } = e _ { \emptyset } } \\ { \mathrm { A D D } ( e ^ { \prime } ) \quad \mathrm { i f } \quad e = e _ { \emptyset } , e ^ { \prime } \neq e _ { \emptyset } } \end{array} \right.\tag{8}
$$

Together, they form an un-ordered set of move $\mathcal { M } .$ , such that we would recover $o _ { y }$ if we fully apply $\mathcal { M }$ to $o _ { x }$ :

$$
\mathcal { M } = \{ m _ { 1 } , . . . , m _ { K } \} , K = \sum _ { e \neq e ^ { \prime } } T _ { e , e ^ { \prime } } ^ { * } ,\tag{9}
$$

$$
\mathrm { A p p l y E d i t } ( o _ { x } , { \mathcal { M } } ) = o _ { y }\tag{10}
$$

## 4.4 Discrete Flow Matching over Electron Edits

Edit-based mixture path Let $s \subseteq { \mathcal { M } }$ be a subset of the moves, interpreted as the moves that have been applied so far. We define the interpolant occupation corresponding to $s$ as $o _ { S } = \mathrm { A p p l y E d i t } ( o _ { x } , S )$ , which by (9) recovers $o _ { x }$ at $s = \emptyset$ and $o _ { y }$ at $s = \mathcal { M }$ . To interpolate between the two endpoints, we place a distribution over which subset has been applied at time t and read the occupation off through ApplyEdit. Following a monotone schedule $\kappa : [ 0 , 1 ]  [ 0 , 1 ]$ with $\kappa ( 0 ) = 0 , \tilde { \kappa ( 1 ) } = 1$ , each move is applied independently with probability $\kappa ( t )$ , giving the edit-based mixture path

$$
p _ { t } ( o \mid o _ { x } , o _ { y } ) = \sum _ { S \subseteq { \mathcal M } } \underbrace { \kappa ( t ) ^ { \mid S \mid } \left( 1 - \kappa ( t ) \right) ^ { \mid { \mathcal M } \mid - \mid S \mid } } _ { p _ { t } ( S \mid o _ { x } , o _ { y } ) } ~ { \mathbb 1 } [ \mathrm { A p p l y E d i t } ( o _ { x } , S ) = o ]\tag{11}
$$

Equivalently, the path is the law of ApplyEdit $( o _ { x } , S _ { t } )$ where $S _ { t }$ is obtained by including each move $m _ { k } \in \mathcal { M }$ independently with probability $\kappa ( t )$ : sampling an interpolant amounts to flipping one Bernoull $\dot { \left[ ( \kappa ( t ) \right) }$ mask per move and applying the selected subset to the reactant occupation (Algorithm 1). Since each move is an independent binary variable, the weight $p _ { t } ( \boldsymbol { S } \mid o _ { x } , o _ { y } )$ factorizes over M, and the endpoints follow directly: at $t = 0$ no moves are sampled so $p _ { 0 } = \delta _ { o _ { x } }$ and at $t = 1$ all moves are applied so $p _ { 1 } = \delta _ { o _ { y } }$

The path (11) is the electron-space instance of the augmented-space mixture path of Edit Flows [39]. There, the path is defined over an auxiliary space of padded sequences and pushed to token sequences by removing padding; here it is defined over subsets of electron moves and pushed to occupations by ApplyEdit. In both, an interpolant is obtained by sampling which edits to apply, and the two constructions yield the same conditional path. We state this equivalence in Proposition 1.

Conditional and marginal rates The mixture path (11) is generated by a CTMC whose conditional rate fires each unapplied move independently. Following the standard DFM derivation, the rate of switching a move $m _ { k }$ from unapplied to applied is the hazard rate of its Bernoulli(κ(t)) flip,

$$
u _ { t } \big ( m _ { k } \mid o _ { t } , o _ { x } , o _ { y } \big ) = \frac { \kappa ^ { \prime } ( t ) } { 1 - \kappa ( t ) } ,\tag{12}
$$

with the reverse rate zero (moves fire once and stay fired). This conditional rate assumes the product $o _ { y }$ is known. At inference it is not, so the training target is the marginal rate, obtained by taking the expectation over the posterior of the endpoints given the current state,

$$
u _ { t } ( m \mid o _ { t } , \mathcal { G } _ { x } ) = \mathbb { E } _ { p _ { t } ( o _ { y } \mid o _ { t } , \mathcal { G } _ { x } ) } u _ { t } \big ( m \mid o _ { t } , o _ { x } , o _ { y } \big ) ,\tag{13}
$$

which depends only on the current state and the reactant, and is therefore learnable by a neural network. By the auxiliary-variable result of Havasi et al. [39], a model trained to match the conditional rate in the move space yields the correct marginal rate after marginalization.

Training We model the reactant-to-product mapping as a coupling $\pi ( \mathcal { G } _ { x } , \mathcal { G } _ { y } )$ over reactant-product pairs. At training time, $\pi$ is the empirical distribution of reactions in the training set, each of which provides one coupled pair. The training target is the marginal rate $u _ { t } ( m \mid o _ { t } , \mathcal { G } _ { x } )$ of Eq. (13), which we approximate with a graph neural network $u _ { t } ^ { \theta } ( m \mid \mathcal { G } _ { x } , \mathcal { G } _ { t } , t )$ We adopt an encoder–decoder architecture: the encoder processes the fixed reactant $\mathcal { G } _ { x }$ into conditioning representations, and the decoder processes the interpolant $\mathcal { G } _ { t }$ and cross-attends to them at every layer, so the predicted rates are conditioned on the reactant throughout, analogous to conditioning on the source in conditional flow matching. Unlike standard molecular GNNs, which typically featurize only atoms and bonds, our edits act on the electron occupation o over the full site set E, including bonding sites, lone pairs, and per-atom hydrogen sites, so an ELECTRONFEATURIZER lifts atom representations into a vector for every electron site (Figure 1e). The per-move rates are read off these electron-site representations, matching the granularity at which the edits operate. Full architectural details can be found in Appendix E.

We fit $u _ { t } ^ { \theta }$ with the Bregman-divergence objective of Holderrieth et al. [41], Havasi et al. [39], which for a CTMC rate reduces to a total-rate penalty minus a log-rate reward on the correct transitions. Let ${ \mathcal { M } } _ { \mathrm { t a r g e t } } = { \mathcal { M } } \backslash { \mathcal { S } }$ be the moves not yet applied at the sampled interpolant. The loss is

$$
\mathcal { L } ( \theta ) \ = \ \underset { ( \mathcal { G } _ { x } , \mathcal { G } _ { y } ) \sim \pi } { \mathbb { E } } \left[ \underbrace { \sum _ { m } u _ { t } ^ { \theta } ( m \mid \mathcal { G } _ { x } , \mathcal { G } _ { t } , t ) } _ { \Lambda _ { \theta , t } } - \frac { \kappa ^ { \prime } ( t ) } { 1 - \kappa ( t ) } \sum _ { \substack { m \in \mathcal { M } _ { \mathrm { t a r g e t } } } } \log u _ { t } ^ { \theta } ( m \mid \mathcal { G } _ { x } , \mathcal { G } _ { t } , t ) \right]\tag{14}
$$

Because a FLOW move acts on pairs of atoms, and the ADD move could act on virtual electron sites, calculating the rates of all possible moves would make the term $\Lambda _ { \theta , t }$ scale as $O ( | \mathcal { E } | ^ { 2 } )$ ). We therefore factorize the move’s rate for these two operations into a per-site source rate and a sink distribution over destinations:

$$
u _ { t } ^ { \theta } ( m \mid \cdot ) = \left\{ \begin{array} { l l } { { \lambda _ { \theta , t } ^ { \mathrm { H o w } } ( e _ { \mathrm { s o u r c e } } \mid \cdot ) p _ { \theta } ^ { \mathrm { s i n k , f l o w } } ( e _ { \mathrm { s i n k } } \mid e _ { \mathrm { s o u r c e } } , \cdot ) } } & { { m = \mathrm { F L O w } ( e _ { \mathrm { s o u r c e } } \to e _ { \mathrm { s i n k } } ) , } } \\ { { \lambda _ { \theta , t } ^ { \mathrm { d e l } } ( e _ { \mathrm { s o u r c e } } \mid \cdot ) } } & { { m = \mathrm { D E L } ( e _ { \mathrm { s o u r c e } } ) , } } \\ { { \lambda _ { \theta , t } ^ { \mathrm { a d d } } ( v \mid \cdot ) p _ { \theta } ^ { \mathrm { s i n k , a d d } } ( e _ { \mathrm { s i n k } } \mid v , \cdot ) } } & { { m = \mathrm { A D D } ( e _ { \mathrm { s i n k } } ) , } } \end{array} \right.\tag{15}
$$

Here $\lambda _ { \theta , t } ^ { \mathrm { f l o w } }$ and $\lambda _ { \theta , t } ^ { \mathrm { d e l } }$ are per-site firing rates over non-virtual electron sites $e _ { \mathrm { s o u r c e } } \in \mathcal { E } , o ^ { e _ { \mathrm { s o u r c e } } } > 0 ;$ $\lambda _ { \theta , t } ^ { \mathrm { a d d } }$ is a per-atom rate over atoms $v \in \mathcal V$ , and $p _ { \theta } ^ { \mathrm { s i n k , f l o w } } , p _ { \theta } ^ { \mathrm { s i n k , a d d } }$ are separately parameterized distributions over destination sites. For DEL no sink is needed, since the operation only removes an electron pair.

We parameterize ADD through an atom v rather than a site because a new electron pair is not associated with any existing site a priori. Intuitively, the model first decides to donate a pair of electrons to atom v, and v then decides where to place them – selecting a destination site $e _ { \mathrm { s i n k } }$ among its associated sites through a dedicated sink head. In this sense ADD behaves as a pseudo-FLOW whose source is the atom itself rather than an occupied electron site, though the two use separate sink parameterizations. The total rate then factorizes as $\begin{array} { r } { \sum _ { m } u _ { t } ^ { \theta } ( m \mid \cdot ) = \breve { \sum _ { e } } ( \lambda ^ { \mathrm { { f o w } } } + \lambda ^ { \mathrm { { d e l } } } ) ^ { \cdot } + \sum _ { a } \lambda ^ { \mathrm { { a d d } } } } \end{array}$ , since each $p ^ { \mathrm { s i n k } }$ sums to one.

The training loop can be found in Algorithm 2.

Inference At inference, the products are generated by the marginal CTMC starting from the reactants. We initialize $\mathcal { G } _ { t } = \mathcal { G } _ { x }$ at $t = 0$ and take N Euler steps of size $\Delta t = 1 / N$ . Since the reactant is fixed, the encoder is run once on $\mathcal { G } _ { x }$ and its representations are reused at every step, while the decoder predicts the move rates on the current graph $\mathcal { G } _ { t } .$ . At each step, every admissible move fires independently as a thinned Poisson process: a move m with predicted rate $\mathbf { \bar { \mathbf { \chi } } } _ { u _ { t } ^ { \theta } ( m \mid \mathcal { G } _ { x } , \mathcal { G } _ { t } , t ) } ^ { }$ is applied with probability

$$
p _ { \mathrm { f i r e } } ( m ) \ = \ 1 - \exp \bigl ( - u _ { t } ^ { \theta } ( m \mid \mathcal { G } _ { x } , \mathcal { G } _ { t } , t ) \Delta t \bigr )\tag{16}
$$

The sampled edit moves are applied to $\mathcal { G } _ { t }$ via ApplyEdit. Iterating to $t = 1$ yields a predicted product $\hat { \mathcal { G } } _ { 1 }$ . The full procedure is given in Algorithms 4.

Confidence and ranking through multiple trajectories A chemical reaction may proceed along several competing pathways, yielding a mixture of products rather than a single deterministic outcome. Because firing in (16) is stochastic, running multiple independent CTMC trajectories from the same reactant $\mathcal { G } _ { x }$ produces a distribution over products. We canonicalize the resulting product graphs, group them by identity, and define the confidence of each unique product as its frequency among the samples; ranking by descending confidence yields a top-k list. This ranking plus the edits only acting on the electrons let MAELLE model a distribution of heavy atom-conservative products, capturing not only the main products as high-confidence predictions, but also plausible side products (Section 5.4).

## 5 Results

## 5.1 Experiments

Datasets and splits. All experiments use the USPTO-480K reaction corpus of ∼480K patent reactions [42, 43], evaluated under two partitionings. The first is the corpus’s standard train/val/test split, a common benchmark for reaction prediction [43, 9, 12, 27, 13] (Section 5.2). The standard split, however, measures only in-distribution learning and, as recent work notes, misses the failure modes that matter most in chemistry [16, 17, 18]. We therefore construct a second partitioning of the same corpus into three test sets probing different axes of generalization (Section 5.3): (i) an in-distribution control (IID); (ii) OOD Mass, holding out high-molecular-weight products to probe structural complexity; and (iii) OOD Ester, holding out non-methyl esterification reactions to probe generalization to unseen reaction types. Split construction, sizes, and preprocessing (atom-mapping with RXNMapper [30] and electron-move interpolation, Section 4.3) are detailed in Appendix F.

Baselines. We compare against representative methods spanning the three prediction modalities, which lets us probe the inductive biases of each modality rather than raw accuracy alone: SMILES generation – Molecular Transformer (MT) [9] and Graph2SMILES (G2S) [11]; graph edits – MEGAN [12] and GTPN [13]; and electron redistribution – NERF [27]. We exclude ELECTRO [21] and FlowER [22], which by design cannot be trained on USPTO-480K: ELECTRO applies only to reactions with linear-electron-flow topology, and FlowER requires elementary-step annotations, unavailable for most USPTO-480K reactions.

Metrics. We report top-k exact-match accuracy, with predicted and reference products canonicalized before comparison. Beyond exact match – which penalizes chemically plausible side products – we additionally report a plausibility rate judged by a large language model (Section 5.4; protocol in Appendix E.5). The directional-flow evaluation of Section 5.5 uses a separate reference dataset and its own coverage metrics, which are discussed there.

Table 2: Top-k accuracy comparison on USPTO-480K. Entries taken from the original papers are marked with "<sup>†</sup>"
<table><tr><td>Method</td><td>Prediction modality</td><td>Top-1</td><td>Top-3</td><td>Top-5</td><td>Top-10</td></tr><tr><td>Molecular Transformer† [9]</td><td>SMILES</td><td>0.886</td><td>0.935</td><td>0.942</td><td></td></tr><tr><td>Graph2SMILES† [11]</td><td>SMILES</td><td>0.903</td><td>0.940</td><td>0.948</td><td>0.953</td></tr><tr><td>MEGAN† [12]</td><td>Graph edits</td><td>0.863</td><td>0.924</td><td>0.940</td><td>0.954</td></tr><tr><td>GTPN† [13]</td><td>Graph edits</td><td>0.832</td><td>0.860</td><td>0.865</td><td></td></tr><tr><td>NERF† [27]</td><td>Electron edits</td><td>0.907</td><td>0.933</td><td>0.937</td><td></td></tr><tr><td>MAELLE (ours)</td><td>Electron flow</td><td>0.872</td><td>0.930</td><td>0.939</td><td>0.946</td></tr></table>

## 5.2 USPTO-480K benchmark

Table 2 compares MAELLE against representative baselines spanning the three prediction modalities: SMILES generation (Molecular Transformer [9], Graph2SMILES [11]), graph edits (MEGAN [12], GTPN [13]), and electron redistribution (NERF [40]). On the standard USPTO-480K split, MAELLE reaches a top-1 accuracy of 87.2%, rising to 93.0%, 93.9%, and 94.6% at top-3, top-5, and top-10 respectively. This places it on par with the graph-edit models – comparable to MEGAN (86.3% top-1) and above GTPN (83.2%) – and within 3.5 points of the strongest SMILES-based model, Graph2SMILES (90.3%), while trailing the electron-redistribution baseline NERF (90.7%) at top-1. At top-5, MAELLE performs on par with NERF and MEGAN, and lags only slightly behind the SMILES-based models.

The standard USPTO-480K split measures in-distribution learning, and by this measure all models capture the training distribution well: at top-5, every method except GTPN exceeds 93% accuracy. We therefore report USPTO-480K performance to establish that MAELLE is able to learn the coupled distribution between reactants and products, using the existing approaches as references. In the next section, we turn to out-of-distribution evaluation (Section 5.3), where differences between modalities are more pronounced.

## 5.3 Out-of-distribution benchmark

A major pitfall of reaction-prediction models is poor generalization outside their training domain [16, 17, 18]. We re-split USPTO-480K along two axes – molecular complexity (OOD Mass) and reaction novelty (OOD Ester, with all non-methyl esterifications held out from training) – alongside an in-distribution control (IID); see Appendix F.3.

Top-k accuracies on these three test sets are shown for MAELLE, Graph2SMILES, MolecularTransformers, and MEGAN in Figure 2a. As expected, all models perform on par with each other on the IID test, with top-10 accuracies of approximately 90%, agreeing with the trend in the test set of the original USPTO-480K benchmark. On the "OOD Mass", models that predict transformations - MAELLE and MEGAN - perform significantly better than the de-novo generative models like Molecular Transformers and Graph2SMILES. This is attributed to the former not having to reconstruct the products from scratch [15]. Indeed, plotting the top-1 accuracy against the molecular weight reveals Molecular Transformers performance decays rapidly as the target becomes more complex, while Graph2SMILES shows a less severe drop. The better accuracy of Graph2SMILES compared with Molecular Transformers could be attributed to its graph encoder, which supports the permutation invariance. On the other hand, MAELLE and MEGAN maintain a stable accuracy across the bins.

While MAELLE and MEGAN perform similarly on OOD Mass, MAELLE shows a significant advantage over MEGAN on the OOD Ester, being the most performant model in this domain. This highlights the benefit of CTMC framework: Even though the model does not see the exact reaction type in the training data, it most likely has seen the intermediate graph states of reactions similar to esterification, such as amide coupling.

Two examples are shown in Figure 2b and 2c for the "OOD Mass" and "OOD Ester", respectively. The first case features a complex macrocycle structure but a very simple amide coupling reaction. Both MAELLE and MEGAN successfully recover the ground truth, while Graph2SMILES and

![](images/a5748e26a1eed7bb5479b44ddc55e9d3d714bd5236dd8e7bb01549446cdaffa3.jpg)  
Figure 2: Comparative performance of MAELLE, Graph2SMILES, Molecular Transformers, and MEGAN on our OOD benchmark. a. From left to right, the first three plots show top-k accuracy on in-distribution test set (IID), esterification-split test set (OOD Ester), and molecular-weight-split test set (OOD Mass). The last plot describes the top-1 accuracy divided into weight bins. b. An illustrative example from the OOD Mass split. c. An illustrative example from the OOD Ester split.

Molecular Transformers struggle to reconstruct the molecular structure in SMILES format. The second case involves an esterification reaction with an anhydride. Out of the four models, only MAELLE correctly predicts the ester as the product

## 5.4 Side product prediction with confidence

By modeling reaction as a generative problem, simulating multiple trajectories, and taking the most popular outputs, we are somewhat mimicking how a real chemistry process happens, where multiple pathways compete with others. To characterize this behavior, we sampled 5000 cases from the test set of USPTO-480K where MAELLE has at least 2 high-confidence predictions with confidence no less than 7.8%, corresponding with 5 out of 64 samples. These examples are then matched with the predictions of Molecular Transformers and Graph2SMILES. As sequence-based methods, these model relies on beam-search for sampling different outcomes in the token space, in contrast with MAELLE which samples in a more chemistry-friendly electron space.

As recent LLMs have shown remarkable symbolic understanding and chemistry knowledge [44, 45], we opted to use LLM-as-a-judge to address the plausibility of the proposed outcomes. Due to the inference nature, the number of unique outcomes from MAELLE is much less than what comes out of Molecular Transformers or Graph2SMILES and thus we capped the number of predictions for test case to be the smallest number of predictions among the models. Given a precursor set and a list of products, the LLM has to give a verdict of whether each of the products is plausible or not, without knowing which prediction comes from which model. As a sanity check, we also include the ground truth product into this list and measure how much proportion of the ground truth is deemed as plausible by the LLM (see Appendix E.5 for more details). The plausibility rate is then calculated as the number of predictions classified as plausible by the LLMs divided by the total number of predictions according to top-k.

Figure 3a displays side-by-side top-k accuracy and top-k plausibility rate. Agreeing with the results in Table 2, top-k accuracy is ranked in the order of Graph2SMILES > Molecular Transformers > MAELLE. However, top-1 plausibility rate of MAELLE (80.7%) is significant higher than its top-1 accuracy (60.3%), suggesting that incorrect predictions are not always implausible. Furthermore, we observe a reversed order when it comes to top-2 and top-3 plausibility rates, with MAELLE being the highest, followed by Molecular Transformers and Graph2SMILES. This hints at the fact that SMILES-based models are likely trained to overfit to one single outcome, which conflicts with the stochastic nature of chemical reactions.

For demonstration, Figure 3b shows an example where an incorrect prediction by MAELLE is still considered plausible by the LLM-judge. In the test dataset, this reaction is a bromide substitution reaction involving an imidazole ring with two nucleophilic nitrogen N:10 and N:12. The ground truth product features an attack of N:12 to C:21 as the main product, likely attributed to the fact that N:12 is less sterically hindered by the benzoyl group compared with N:10. All three models manage to predict the correct ground truth as top-1. However, intuitively speaking, the side product involved N:10 is not entirely impossible, and MAELLE was able to recover this minor outcome as its top-2 prediction. On the other hand, due to the fact that Graph2SMILES and Molecular Transformers sample in the token space, top-2 predictions of these models are just hallucinated versions of their top-1 prediction. This example is also aligned with the observation that AI systems have limited capability in writing SMILES and other symbolic sequences, which is one of the major bottlenecks in AI applications for chemistry [46, 47].

Figure 3c showcases how the step-by-step CTMC of electron edits enables MAELLE to sample multiple outcomes that agree with chemical mechanisms. The confidence scores assigned to each outcome successfully differentiate the major and minor products, agreeing with the ground truth.

## 5.5 Recovery of net directional electron flows

Table 3: MAELLE’s coverage of bond-forming elementary steps categorized into persistent and transient bonds.
<table><tr><td rowspan="2">Bond formation type</td><td rowspan="2"># ground-truth elementary steps</td><td colspan="3">Recovered by MAELLE&#x27;s trajectories</td></tr><tr><td>Correct</td><td>Top 1</td><td>All</td></tr><tr><td>Persistent</td><td>17,925</td><td>13,137 (73.3%)</td><td>14,236 (79.4%)</td><td>17,847 (99.6%)</td></tr><tr><td>Transient</td><td>19,658</td><td>3,042 (15.5%)</td><td>747 (3.8%)</td><td>5,814 (29.6%)</td></tr><tr><td>Total</td><td>37,583</td><td>16,179 (43.0%)</td><td>14,983 (39.9%)</td><td>23,661 (63.0%)</td></tr></table>

Beyond predicting the product, a further advantage of electron-based trajectories is that they reveal how the electrons are redistributed. NERF also models reactions as electron redistributions, but because it predicts only the net change at each bond, it cannot say where the electrons come from or go. For MAELLE this information is explicit: the FLOW operation names a source and a sink, so each trajectory carries a directional electronflow that resembles an arrow-pushing mechanism.

We train MAELLE on the FlowER dataset [22], which contains ∼290K reactions extracted from USPTO-Full [42], each annotated with a sequence of expert-labeled elementary steps (∼2M steps in total). MAELLE is trained end-to-end and never consumes these labels. We draw S = 64 trajectories per reaction and, for each ground-truth bond-forming elementary step, check whether some MAELLE trajectory forms the same bond and, if so, whether the source and sink of its FLOW agree with the reference.

To support our claim that MAELLE can recover the directional electron flow, i.e. bonds created as a result of the FLOW operation, we collect all bond-forming elementary steps in the test set of FlowER, and check how many of them are recovered from MAELLE’s predictions, and check if the directions of the electron flow are correct, that is, when the source and sink are similar to the reference.

![](images/2672e723a05c58e21410f2e469855a622b9d611f6af7bd1b2c47cad6f5a81eeb.jpg)  
Figure 3: a. Top-k accuracy and plausibility rate, with the latter having mean and standard deviation reported for 3 runs of LLM evaluation. The LLM-judge has a ground truth coverage of 81.1 ± 0.1%.b. Test case 39591 from the test set of USPTO-480K serves as an illustrative example where MAELLE successfully predicts the main product as top-1 and the side product as top-2, while Graph2SMILES and MolecularTransformers propose hallucinated outcomes. c. Two example trajectories that MAELLE samples leading to the two different outcomes.

Table 4: MAELLE’s coverage of bond-forming elementary steps and their directional agreement with FlowER’s reference. For a MAELLE-predicted step and a ground-truth step that form the same bond, their directions agree if their sources and sinks are identical.
<table><tr><td>Sampling mode</td><td>Elementary steps coverage</td><td>Directional agreement of elementary steps</td></tr><tr><td>Correct</td><td>16,179/37,583 (43.0%)</td><td>13,694/16,179 (84.6%)</td></tr><tr><td>Top-1</td><td>14,983/37,583 (39.9%)</td><td>13,342/14,983 (89.0%)</td></tr><tr><td>Ali</td><td>23,661/37,583 (63.0%)</td><td>18,505/23,661 (78.2%)</td></tr></table>

FlowER’s test set contains 37,583 bond-forming steps. We consider three ways of selecting among the 64 trajectories: CORRECT keeps only trajectories that reach the ground-truth product; TOP-1 takes the trajectories leading to the highest-confidence product regardless of correctness; and ALL pools all 64. These recover 43.0%, 39.9%, and 63.0% of the annotated steps, respectively (Table 3). The gap from full coverage is expected: expert pathways contain many transient bonds – formed as short-lived intermediates and broken again before the product – whereas MAELLE is trained on moves interpolated by optimal transport with a topological cost (Sections 4.3, 4.4), which favors the persistent bonds that survive into the product. Indeed, sampling all trajectories recovers 99.6% of persistent bonds but only 29.6% of transient ones (Table 3). Additionally, the coverage when we sample all trajectories is higher compared with sampling those of just the correct products or top-1, meaning that stochastic sampling surfaces additional annotated bonds, which contributes to the multi-outcome behavior in Section 5.4.

![](images/e55b5fcb5cac6c84a2385feb90b588499c5bb3ba6f34cce2d8436461ce30e8af.jpg)  
Figure 4: a. An example from FlowER’s test set with expert-labeled elementary steps. b. Pseudo mechanistic steps in MAELLE’s trajectory. Note that during training, MAELLE does not consume the labeled elementary steps.

Among the annotated steps that MAELLE recovers, we also observe high agreement in terms of the direction of the electron moves (Table 4). For CORRECT, TOP-1, and ALL sampling schemes, the directional agreement rates are 84.6%, 89.0%, and 78.2%, respectively, showing that the electron moves produced by MAELLE, which is an artifact of the edit-based mixture path (Section 4.4), serve as a good approximation of the labeled steps.

Figure 4 shows a representative case from FlowER’s test set: the esterification of a carboxylic acid activated by thionyl chloride (SOCl<sub>2</sub>). The expert pathway proceeds through a chlorosulfite intermediate – the carboxylic oxygen attacks sulfur, chloride leaves then attacks the carboxylic carbon, displacing the chlorosulfite ester. The chloride itself ultimately becomes the leaving group for the substitution of methanol. Throughout the trajectory, several transient bonds are formed, including C-Cl and one electron pair of the C=O bond. On the other hand, MAELLE’s trajectory recovers the persistent bond formations that define the product – the new acyl C-O bond to the incoming alkoxy group, the change from C-O single bond to C=O double bond, and the formation of the S=O double bond. The predicted flows are directionally consistent with the reference on the persistent steps, illustrating both the coverage and the directional-agreement statistics above.

Together these results support our claim that MAELLE’s trajectories are pseudo-mechanistic: without any elementary-step supervision, they recover the persistent, product-defining electron flows and their direction, serving as an approximation of expert-annotated mechanisms.

## 6 Conclusion

We introduced MAELLE, a reaction prediction model that operates in the electron occupation space via discrete flow matching. By formulating reactions as CTMCs over an auxiliary move-state space derived from optimal transport, MAELLE bridges the gap between machine learning and the arrowpushing formalism that chemists use to reason about reactivity. Our experiments demonstrate three key findings: (i) competitive in-distribution performance on USPTO-480K despite operating in a more constrained representation than SMILES-based methods; (ii) strong out-of-distribution robustness on both molecular-weight and reaction-type generalization, where MAELLE outperforms all baselines; and (iii) the ability to predict chemically plausible side products through multiple CTMC trajectory sampling, a capability inaccessible to deterministic or beam-search-based approaches.

Limitations and future work. MAELLE currently models electron pairs rather than individual electrons, which precludes radical reactions involving unpaired electrons; extending the occupation space to single-electron resolution would address this. The framework predicts forward reactions only; adapting the edit operations to the atom level (e.g., atom insertion and deletion) would enable retrosynthetic prediction. While the OT-derived move trajectories are mechanistically interpretable and correlate with known chemistry, they represent pseudo-mechanisms rather than energy-optimal elementary steps – the model learns the most likely combinatorial path between reactant and product occupations, not the true potential energy surface. Incorporating energetic priors or non-optimaltransport pathways could improve mechanistic fidelity. We note that reagents and catalysts are already incorporated as context through the reactant graph G<sub>0</sub>; the model conditions on them even though they do not directly participate in the electron flow.

## References

[1] Connor W Coley, Pankaj Daga, Marco De Vivo, Willem Jespers, Ashutosh S Jogalekar, S Roy Kimura, Lucien Koenekoop, Anne-Grete Märtson, Timothy R Newhouse, Soumya Ray, et al. Grand challenges for predictive modeling in small molecule drug discovery. 2026.

[2] Elias James Corey and W Todd Wipke. Computer-assisted design of complex organic syntheses: Pathways for molecular synthesis can be devised with a computer and equipment for graphical communication. Science, 166(3902):178–192, 1969.

[3] Philippe Schwaller, Riccardo Petraglia, Valerio Zullo, Vishnu H Nair, Rico Andreas Haeuselmann, Riccardo Pisoni, Costas Bekas, Anna Iuliano, and Teodoro Laino. Predicting retrosynthetic pathways using transformer-based models and a hyper-graph exploration strategy. Chem. Sci., 11:3316–3325, 2020.

[4] Harry L Morgan. The generation of a unique machine description for chemical structures-a technique developed at chemical abstracts service. Journal ofchemical documentation, 5(2): 107–113, 1965.

[5] David Rogers and Mathew Hahn. Extended-connectivity fingerprints. J. Chem. Inf. Model., 50 (5):742–754, 2010.

[6] Marwin HS Segler and Mark P Waller. Neural-symbolic machine learning for retrosynthesis and reaction prediction. Chem. Eur. J., 23:5966–5971, 2017.

[7] Connor W Coley, Regina Barzilay, Tommi S Jaakkola, William H Green, and Klavs F Jensen. Prediction of organic reaction outcomes using machine learning. ACS Cent. Sci., 3(5):434–443, 2017.

[8] Philippe Schwaller, Theophile Gaudin, David Lanyi, Costas Bekas, and Teodoro Laino. “Found in Translation”: predicting outcomes of complex organic chemistry reactions using neural sequence-to-sequence models. Chem. Sci., 9:6091–6098, 2018.

[9] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A Hunter, Costas Bekas, and Alpha A Lee. Molecular transformer: a model for uncertainty-calibrated chemical reaction prediction. ACS central science, 5(9):1572–1583, 2019.

[10] Connor W Coley, Wengong Jin, Luke Rogers, Timothy F Jamison, Tommi S Jaakkola, William H Green, Regina Barzilay, and Klavs F Jensen. A graph-convolutional neural network model for the prediction of chemical reactivity. Chem. Sci., 10:370–377, 2019.

[11] Zhengkai Tu and Connor W Coley. Permutation invariant graph-to-sequence model for templatefree retrosynthesis and reaction prediction. Journal ofchemical information and modeling, 62 (15):3503–3513, 2022.

[12] Mikołaj Sacha, Mikołaj Błaz, Piotr Byrski, Paweł Dabrowski-Tumanski, Mikołaj Chrominski, Rafał Loska, Paweł Włodarczyk-Pruszynski, and Stanisław Jastrzebski. Molecule edit graph attention network: modeling chemical reactions as sequences of graph edits. Journal of Chemical Information and Modeling, 61(7):3273–3284, 2021.

[13] Kien Do, Truyen Tran, and Svetha Venkatesh. Graph transformation policy network for chemical reaction prediction. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 750–760, 2019.

[14] Robert B Grossman and Robert Grossman. The art of writing reasonable organic reaction mechanisms. Springer, 2003.

[15] Nguyen Xuan-Vu, Daniel Armstrong, Zlatko Joncev, and Philippe Schwaller. Tempre: Template generation for single and direct multi-step retrosynthesis. arXiv preprint arXiv:2507.21762, 2025.

[16] Victor Sabanza Gil, Andres M Bran, Malte Franke, Remi Schlama, Jeremy S Luterbacher, and Philippe Schwaller. Holistic chemical evaluation reveals pitfalls in reaction prediction models. arXiv preprint arXiv:2312.09004, 2023.

[17] John Bradshaw, Anji Zhang, Babak Mahjour, David E Graff, Marwin HS Segler, and Connor W Coley. Challenging reaction prediction models to generalize to novel chemistry. ACS Central Science, 11(4):539–549, 2025.

[18] Suong BA Tran, Jihye Roh, and Connor W Coley. Quantifying the failure modes of current one-step retrosynthesis models. Chemical Science, 2026.

[19] Matthew A Kayala, Chloé-Agathe Azencott, Jonathan H Chen, and Pierre Baldi. Learning to predict chemical reactions. J. Chem. Inf. Model., 51:2209–2222, September 2011.

[20] Matthew A Kayala and Pierre Baldi. Reactionpredictor: prediction of complex chemical reactions at the mechanistic level using machine learning. Journal ofchemical information and modeling, 52(10):2526–2540, 2012.

[21] John Bradshaw, Matt J Kusner, Brooks Paige, Marwin HS Segler, and José Miguel Hernández-Lobato. A generative model for electron paths. arXiv preprint arXiv:1805.10970, 2018.

[22] Joonyoung F Joung, Mun Hong Fong, Nicholas Casetti, Jordan P Liles, Ne S Dassanayake, and Connor W Coley. Electron flow matching for generative reaction mechanism prediction. Nature, 645(8079):115–123, 2025.

[23] Joonyoung F Joung, Mun Hong Fong, Jihye Roh, Zhengkai Tu, John Bradshaw, and Connor W Coley. Reproducing reaction mechanisms with machine-learning models trained on a large-scale mechanistic dataset. Angewandte Chemie International Edition, 63(43):e202411296, 2024.

[24] Mohammadamin Tavakoli, Ryan J Miller, Mirana Claire Angel, Michael A Pfeiffer, Eugene S Gutman, Aaron D Mood, David Van Vranken, and Pierre Baldi. Pmechdb: A public database of elementary polar reaction steps. Journal of Chemical Information and Modeling, 64(6): 1975–1983, 2024.

[25] Mohammadamin Tavakoli, Yin Ting T Chiu, Pierre Baldi, Ann Marie Carlton, and David Van Vranken. Rmechdb: A public database of elementary radical reaction steps. Journal of chemical information and modeling, 63(4):1114–1123, 2023.

[26] Ross Irwin, Spyridon Dimitriadis, Jiazhen He, and Esben Jannik Bjerrum. Chemformer: a pretrained transformer for computational chemistry. Machine Learning: Science and Technology, 3(1):015022, 2022.

[27] Hangrui Bi, Hengyi Wang, Chence Shi, Connor Coley, Jian Tang, and Hongyu Guo. Nonautoregressive electron redistribution modeling for reaction prediction. In International Conference on Machine Learning, pages 904–913. PMLR, 2021.

[28] Daylight Theory: SMILES. URL https://www.daylight.com/dayhtml/doc/theory/ theory.smiles.html. (Accessed Nov 15, 2021).

[29] Mario Krenn, Florian Häse, AkshatKumar Nigam, Pascal Friederich, and Alan Aspuru-Guzik. Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation. Mach. Learn.: Sci. Technol., 1:045024, 2020.

[30] Philippe Schwaller, Benjamin Hoover, Jean-Louis Reymond, Hendrik Strobelt, and Teodoro Laino. Extraction of organic chemistry grammar from unsupervised learning of chemical reactions. Science Advances, 7(15):eabe4166, 2021.

[31] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. arXiv preprint arXiv:2209.14734, 2022.

[32] Yiming Qin, Manuel Madeira, Dorina Thanou, and Pascal Frossard. Defog: Discrete flow matching for graph generation. arXiv preprint arXiv:2410.04263, 2024.

[33] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

[34] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[35] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. arXiv preprint arXiv:2402.04997, 2024.

[36] Ilia Igashov, Arne Schneuing, Marwin Segler, Michael Bronstein, and Bruno Correia. Retrobridge: Modeling retrosynthesis with markov bridges. In International Conference on Learning Representations, volume 2024, pages 39622–39640, 2024.

[37] Umut Çetin and Albina Danilova. Markov bridges: Sde representation. Stochastic Processes and their Applications, 126(3):651–679, 2016.

[38] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. Advances in Neural Information Processing Systems, 37:133345–133385, 2024.

[39] Marton Havasi, Brian Karrer, Itai Gat, and Ricky TQ Chen. Edit flows: Flow matching with edit operations. arXiv preprint arXiv:2506.09018, 2025.

[40] Shuan Chen, Kye Sung Park, Taewan Kim, Sunkyu Han, and Yousung Jung. Predicting chemical reaction outcomes based on electron movements using machine learning. arXiv preprint arXiv:2503.10197, 2025.

[41] Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola, Brian Karrer, Ricky TQ Chen, and Yaron Lipman. Generator matching: Generative modeling with arbitrary markov processes. arXiv preprint arXiv:2410.20587, 2024.

[42] Daniel Mark Lowe. Extraction ofchemical structures and reactionsfrom the literature. PhD thesis, University of Cambridge, 2012.

[43] Wengong Jin, Connor Coley, Regina Barzilay, and Tommi Jaakkola. Predicting organic reaction outcomes with weisfeiler-lehman network. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 2607–2616. Curran Associates, Inc., 2017.

[44] Andres M Bran, Theo A Neukomm, Daniel P Armstrong, Zlatko Joncev, and Philippe Schwaller.ˇ Chemical reasoning in llms unlocks steerable synthesis planning and reaction mechanism elucidation. arXiv preprint arXiv:2503.08537, 2025.

[45] Varvara Voinarovska, Rocío Mercado, Mikhail Kabeshov, and Samuel Genheden. Do humans and large language models agree on the quality of synthesis plans? 2026.

[46] Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho, and Heng Ji. Translation between molecules and natural language. Proc. Conf. Empirical Methods Nat. Lang. Process., pages 375–413, 2022.

[47] Hyosoon Jang, Yunhui Jang, Jaehyung Kim, and Sungsoo Ahn. Can llms generate diverse molecules? towards alignment with structural diversity. arXiv preprint arXiv:2410.03138, 2024.

[48] Alexandre Duval, Simon V Mathis, Chaitanya K Joshi, Victor Schmidt, Santiago Miret, Fragkiskos D Malliaros, Taco Cohen, Pietro Lio, Yoshua Bengio, and Michael Bronstein. A hitchhiker’s guide to geometric gnns for 3d atomic systems. arXiv preprint arXiv:2312.07511, 2023.

[49] Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua´ Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.

[50] Daniel Mark Lowe. Extraction of chemical structures and reactions from the literature. PhD thesis, University of Cambridge, 2012.

[51] Daniel Lowe. Chemical reactions from US patents (1976-Sep2016) http://doi.org/10. 6084/m9.figshare.5104873.v1, 6 2017.

[52] RDKit, online. RDKit: Open-source cheminformatics. http://www.rdkit.org, 2023.

## A Equivalence of the edit-based and augmented-space mixture paths

We show, in a setting-agnostic form, that the augmented-space mixture path of Edit Flows [39] is equivalent to directly sampling a subset of edit moves and applying them. The statement is independent of what the edits are: it applies verbatim to sequence edits (insertion, deletion, substitution) and to the electron-occupation edits (FLOW, DEL, ADD) used in the main text.

Setup. Let D be a state space (e.g. token sequences, or electron occupation vectors) and fix a source state $x _ { 0 } \in \mathcal { D }$ . Let $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { K } \}$ be a finite set of edit moves, each an operation on $\mathcal { D } _ { : }$ and let

$$
{ \mathrm { A p p l y : ~ } } { \mathcal { D } } \times 2 ^ { \mathcal { M } } \to { \mathcal { D } }\tag{17}
$$

apply a subset of moves to a state, assumed well defined (independent of the order in which the moves in the subset are applied). Write $x _ { 1 } = \mathrm { A p p l y } ( x _ { 0 } , { \mathcal { M } } )$ for the state obtained by applying all moves. Let κ : $[ 0 , 1 ]  [ 0 , 1 ]$ be a schedule with $\kappa _ { 0 } = 0$ and $\kappa _ { 1 } = 1$

We compare two ways of building a conditional path $p _ { t } ( \cdot \mid x _ { 0 } , x _ { 1 } )$ on $\mathcal { D }$ that interpolates from $x _ { 0 }$ at $t = 0$ to $x _ { 1 }$ at $t = 1$

(A) Augmented-space path. Introduce the auxiliary space $\mathcal { Z } = \{ 0 , 1 \} ^ { K }$ , one binary coordinate per move, with the readout

$$
g : { \mathcal { Z } } \to { \mathcal { D } } , \qquad g ( z ) = \operatorname { A p p l y } \left( x _ { 0 } , \{ m _ { k } : z _ { k } = 1 \} \right) .\tag{18}
$$

Treating each coordinate as an independent binary token that interpolates $z _ { k } : 0 \to 1$ [35, 38] gives the factorized path on $\mathcal { Z }$

$$
p _ { t } ( z \mid x _ { 0 } , x _ { 1 } ) = \prod _ { k = 1 } ^ { K } \Big [ ( 1 - \kappa _ { t } ) \delta _ { 0 } ( z _ { k } ) + \kappa _ { t } \delta _ { 1 } ( z _ { k } ) \Big ] ,\tag{19}
$$

and the augmented-space path on $\mathcal { D }$ is its pushforward through $^ { g , }$

$$
p _ { t } ^ { \mathrm { A } } ( x \mid x _ { 0 } , x _ { 1 } ) = \sum _ { z \in \mathcal { Z } } p _ { t } ( z \mid x _ { 0 } , x _ { 1 } ) \mathbb { 1 } [ g ( z ) = x ] .\tag{20}
$$

For sequence edits with $\mathcal { Z }$ realized as padded sequences and $g$ as padding removal, (20) is the mixture path of Havasi et al. [39].

(B) Edit-based path. Include each move independently with probability $\kappa _ { t }$ , collect the applied subset S, and apply it to $x _ { 0 }$ . The resulting law is

$$
p _ { t } ^ { \mathrm { B } } ( x \mid x _ { 0 } , x _ { 1 } ) = \sum _ { S \subseteq \mathcal { M } } \kappa _ { t } ^ { | S | } ( 1 - \kappa _ { t } ) ^ { K - | S | } \mathbb { 1 } [ \mathrm { A p p l y } ( x _ { 0 } , S ) = x ] .\tag{21}
$$

Proposition 1 (Edit-sampling form of the augmented-space path). For every $t \in [ 0 , 1 ]$ and every $x \in \mathcal { D }$

$$
p _ { t } ^ { \mathrm { A } } ( x \mid x _ { 0 } , x _ { 1 } ) = p _ { t } ^ { \mathrm { B } } ( x \mid x _ { 0 } , x _ { 1 } ) .\tag{22}
$$

Consequently sampling $z \sim p _ { t } ( \cdot \ | \ x _ { 0 } , x _ { 1 } )$ from (19) and returning $g ( z )$ has the same law as sampling a subset S of moves by independent Bernoull $\mathrm { i } ( \kappa _ { t } )$ inclusion and returning $\mathrm { A p p l y } ( x _ { 0 } , S )$ . In particular $p _ { 0 } ^ { \mathrm { A } } = p _ { 0 } ^ { \mathrm { B } } = \delta _ { x _ { 0 } }$ and $p _ { 1 } ^ { \mathrm { A } } \dot { = } p _ { 1 } ^ { \mathrm { B } } = \delta _ { x _ { 1 } }$ , so both are valid conditional interpolants.

Proof. The map $z \mapsto S ( z ) : = \{ m _ { k } : z _ { k } = 1 \}$ is a bijection between $\mathcal { Z } = \{ 0 , 1 \} ^ { K }$ and the power set $2 ^ { \ j M }$ , with inverse $S \mapsto z$ where $z _ { k } = \mathbb { 1 } [ m _ { k } \in S ]$ . We rewrite the augmented-space weight (19) under this bijection.

Each factor in (19) selects one of its two terms according to $z _ { k } \mathrm { : }$ it contributes $\kappa _ { t }$ when $z _ { k } = 1$ and $1 - \kappa _ { t }$ when $z _ { k } = 0$ . Hence for a fixed z,

$$
p _ { t } ( z \mid x _ { 0 } , x _ { 1 } ) = \prod _ { k = 1 } ^ { K } \kappa _ { t } ^ { z _ { k } } ( 1 - \kappa _ { t } ) ^ { 1 - z _ { k } } = \kappa _ { t } ^ { \sum _ { k } z _ { k } } \left( 1 - \kappa _ { t } \right) ^ { K - \sum _ { k } z _ { k } } = \kappa _ { t } ^ { | S ( z ) | } ( 1 - \kappa _ { t } ) ^ { K - | S ( z ) | } ,\tag{23}
$$

using $\begin{array} { r } { \sum _ { k } z _ { k } = | S ( z ) | } \end{array}$ |. By definition of the readout $( 1 8 ) , g ( z ) = \mathrm { A p p l y } ( x _ { 0 } , S ( z ) )$ . Substituting both identities into (20) and reindexing the sum from $z \mathrm { t o } S = { \cal S } ( z )$

$$
p _ { t } ^ { \mathrm { A } } ( x \mid x _ { 0 } , x _ { 1 } ) = \sum _ { z \in \mathcal { Z } } \kappa _ { t } ^ { | \mathcal { S } ( z ) | } ( 1 - \kappa _ { t } ) ^ { K - | \mathcal { S } ( z ) | } \mathbb { 1 } [ \mathrm { A p p l y } ( x _ { 0 } , \mathcal { S } ( z ) ) = x ] \qquad\tag{24}
$$

This proves the equality for all t and x.

For the boundary conditions, at $t = 0$ we have $\kappa _ { 0 } = 0 ;$ so the weight $\kappa _ { 0 } ^ { | S | } ( 1 - \kappa _ { 0 } ) ^ { K - | S | }$ is nonzero only for $s = \emptyset$ , where it equals 1; since $\mathrm { A p p l y } ( x _ { 0 } , \mathcal { O } ) = x _ { 0 }$ , we get $p _ { 0 } = \delta _ { x _ { 0 } } . \mathrm { A t } t = 1 , \kappa _ { 1 } = 1$ so the weight is nonzero only for ${ \mathcal { S } } = { \mathcal { M } }$ , where it equals 1; since $\mathrm { A p p l y } ( x _ { 0 } , { \mathcal { M } } ) = x _ { 1 }$ , we get $p _ { 1 } = \delta _ { x _ { 1 } }$ □

Corollary 1 (Sequence edits). Taking D to be token sequences and M a set of insertion, deletion, and substitution moves aligning x<sub>0</sub> to $x _ { 1 }$ , Proposition 1 recovers the augmented-space mixture path of Edit Flows [39], with $\mathcal { Z }$ realized as padded sequences and $g = f _ { \mathrm { r m - b l a n k s } }$ that strips all padding tokens.

Corollary 2 (Electron edits). Taking $\mathcal { D } = \mathbb { Z } _ { > 0 } ^ { | \mathcal { E } | } , x _ { 0 } = o _ { x } ,$ , M the move set ofEq. (9), and $\mathrm { A p p l y = }$ ApplyEdit, Proposition 1 shows that the edit-based mixture path (11) equals the augmented-space pushforward, justifying the interpolant-sampling scheme ofAlgorithm 1.

## B Problem-statement details

Atoms, adjacency, and electron sites. $\mathcal { V } = \{ v _ { i } \ | \ 1 \le i \le \ | \mathcal { V } | \}$ is the set of heavy atoms; hydrogens are not nodes but are pooled into per-atom electron sites. The adjacency $\mathcal { A } = \{ a _ { i j } \in$ $\{ \bar { 0 } , 1 \} \bar { | } 1 \leq i < j \leq | \mathcal { V } | \}$ follows the standard molecular graph convention [48]. The electron-site set

$$
\mathcal { E } = \{ e _ { i j } \mid 1 \leq i \leq j \leq | \mathcal { V } | \} \cup \{ e _ { i } ^ { H } \mid 1 \leq i \leq | \mathcal { V } | \}
$$

has $| \mathcal { E } | = \binom { | \mathcal { V } | } { 2 } + 2 | \mathcal { V } |$ entries: bonding sites $e _ { i j } ( i \neq j )$ , lone-pair sites $e _ { i i }$ , and per-atom hydrogen sites $e _ { i } ^ { H }$ . Hydrogens bonded to the same heavy atom are topologically equivalent so a single $e _ { i } ^ { H }$ suffices [40]. Each site is either realized (existing bond/lone pair/hydrogen attachment) or virtual $( o ( e ) = 0$ , available to become realized).

Occupation ⇒ topology. The map $f _ { o \to a } : ( \mathcal { V } , \mathcal { E } , o ) \mapsto A$ deterministically recovers the heavyatom adjacency from the occupation via Eq. (4). Combined with the fact that V (and hence $\mathcal { E } )$ is preserved by a reaction, this yields the simplification

$$
p ( \mathcal { G } _ { y } \mid \mathcal { G } _ { x } ) = p ( \mathcal { V } _ { y } , \mathcal { A } _ { y } , \mathcal { E } _ { y } , o _ { y } \mid \mathcal { V } _ { x } , \mathcal { A } _ { x } , \mathcal { E } _ { x } , o _ { x } ) = p ( o _ { y } \mid \mathcal { V } _ { x } , \mathcal { A } _ { x } , \mathcal { E } _ { x } , o _ { x } ) .
$$

Ethanol example. Consider ethanol $\mathrm { ( C H _ { 3 } C H _ { 2 } O H ) }$ with heavy atoms $\boldsymbol { \mathcal { V } } = \{ { \bf C } _ { 1 } , { \bf C } _ { 2 } , { \bf O } _ { 3 } \}$ . The electron sites and their occupations are:
<table><tr><td>Site</td><td>Electron type</td><td>Occupation o</td><td>Description</td></tr><tr><td> $e _ { 1 , 2 } \ ( { \bf C } _ { 1 } { - } { \bf C } _ { 2 } )$ </td><td>Bond</td><td>1</td><td>Single bond</td></tr><tr><td> $e _ { 2 , 3 } \left( { \bf C } _ { 2 } { - } 0 \right)$ </td><td>Bond</td><td>1</td><td>Single bond</td></tr><tr><td> $e _ { 1 , 3 } \left( \mathbf { C } _ { 1 } { - } \mathbf { O } \right)$ </td><td>Virtual</td><td>0</td><td>Virtual (no bond)</td></tr><tr><td>e3,3 (O lone pairs)</td><td>Valence</td><td>2</td><td>Two lone pairs</td></tr><tr><td> $e _ { 1 , 1 } , e _ { 2 , 2 }$ </td><td>Virtual</td><td>0</td><td>No lone pairs on carbons</td></tr><tr><td> $e _ { 1 } ^ { H } \left( \mathbf { C } _ { 1 } \right)$ </td><td>w/ Hydrogens</td><td>3</td><td>Three hydrogens</td></tr><tr><td> $e _ { 2 } ^ { H } \ ( \mathrm { C _ { 2 } } )$ </td><td>w/ Hydrogens</td><td>2</td><td>Two hydrogens</td></tr><tr><td> $e _ { 3 } ^ { H } \left( \mathrm { O } \right)$ </td><td>w/ Hydrogens</td><td>1</td><td>One hydrogen</td></tr></table>

This gives a total of 10 occupied electron pairs across 9 sites, consistent with ethanol’s 20 valence electrons.

## C Optimal Transport Details

To handle reactions where the total electron count is not conserved, we augment the electron site set with a virtual site $e _ { \emptyset }$ acting as a source and sink for electron pairs gained or lost. Let $\bar { \mathcal { E } } = \mathcal { E } \cup \{ e _ { \emptyset } \}$ and define augmented occupation vectors $\bar { o } _ { x } , \bar { o } _ { y } \in \mathbb { Z } _ { \ge 0 } ^ { | \bar { \mathcal { E } } | }$

so that $\begin{array} { r } { \sum _ { e \in \bar { \mathcal { E } } } \bar { o } _ { x } ( e ) = \sum _ { e \in \bar { \mathcal { E } } } \bar { o } _ { y } ( e ) } \end{array}$ , ensuring balanced transport.

Cost matrix. We define the cost matrix $C \in \mathbb { Z } _ { > 0 } ^ { | \bar { \mathcal { E } } | \times | \bar { \mathcal { E } } | }$ based on shortest-path distances. Let $d _ { x } ( v , v ^ { \prime } )$ and $d _ { y } ( v , v ^ { \prime } )$ denote shortest-path distances on the reactant and product graphs, respectively. Since bonds may form or break, we take the minimum over both:

$$
C _ { e , e ^ { \prime } } = \operatorname* { m i n } \left( \operatorname* { m i n } _ { \substack { v \in \mathrm { a t o m s } ( e ) } } d _ { x } ( v , v ^ { \prime } ) , \ \operatorname* { m i n } _ { \substack { v \in \mathrm { a t o m s } ( e ) } } d _ { y } ( v , v ^ { \prime } ) \right) , \qquad \forall e , e ^ { \prime } \in \mathcal { E } ,
$$

where atoms $( e ) \subseteq \nu$ denotes the atoms associated with site e:

$$
\operatorname { a t o m s } ( e ) = { \left\{ \begin{array} { l l } { \{ v _ { i } , v _ { j } \} } & { { \mathrm { i f } } e = e _ { i , j } { \mathrm { ~ w i t h ~ } } i \neq j , } \\ { \{ v _ { i } \} } & { { \mathrm { i f } } e = e _ { i , i } { \mathrm { ~ o r } } e = e _ { i } ^ { H } . } \end{array} \right. }
$$

The cost to or from the virtual site is set to a fixed large constant $C _ { e , e \infty } = C _ { e \infty , e } = c _ { \infty }$ , incentivizing the solver to use $e _ { \emptyset }$ only when necessary.

## D Algorithms

Algorithm 1 SAMPLEINTERMEDIATEGRAPH(G<sub>0</sub>, M, κ)   
Require: Reactant graph ${ \mathcal { G } } _ { 0 } ,$ set of electron moves M, interpolation level $\kappa \in [ 0 , 1 ]$   
Ensure: Intermediate graph $\mathcal { G } _ { t }$ , remaining target moves $\mathbf { M } _ { \mathrm { t a r g e t } }$   
1: for each move $( e ^ { s } , e ^ { d } ) \in \mathbf { M }$ do   
2: Sample $z _ { s , d } \sim$ Bernoulli(κ)   
3: end for   
4: $\mathbf { M } _ { \mathrm { f i r e d } }  \{ ( e ^ { s } , e ^ { d } ) : z _ { s , d } = 1 \}$ ▷ moves already applied   
5: $\mathbf { M } _ { \mathrm { t a r g e t } }  \{ ( e ^ { s } , e ^ { d } ) : z _ { s , d } = 0 \}$ ▷ moves still to predict   
6: $\mathcal { G } _ { t } \longleftarrow \mathrm { A P P L Y E L E C T R O N E D I T S } ( \mathcal { G } _ { 0 } , \ : \mathbf { M } _ { \mathrm { f i r e d } } )$   
7: return $\mathcal { G } _ { t } , \ \mathbf { M } _ { \mathrm { t a r g e t } }$

Algorithm 2 MAELLE’s Training Loop   
Require: Data distribution $p ( \mathbf x , \mathbf y )$ , model u , kappa schedule $\kappa ( \cdot )$   
1: Sample $( \mathcal G _ { 0 } , \mathcal G _ { 1 } ) \sim p ( \mathbf x , \mathbf y )$ ▷ reactant and product graphs   
2: M ← SOLVEOPTIMALTRANSPORT $( \mathcal { G } _ { 0 } , \mathcal { G } _ { 1 } )$ ▷ electron moves from OT plan   
3: Sample t ∼ Uniform(0, 1)   
4: G<sub>t</sub>, $\mathbf { \dot { M } } _ { \mathrm { t a r g e t } }  \mathbf { S } \mathbf { A } \mathbf { M } \mathbf { \dot { P } } \mathrm { L }$ EINTERMEDIATEGRAPH(G , M, κ(t)) ▷ Alg. 1   
5: // Forward pass   
6: $\lambda ^ { \mathrm { { f o w } } } , \lambda ^ { \mathrm { { d e l } } } , \bar { \lambda } ^ { \mathrm { { a d d } } } , \mathcal { E }  u _ { \theta } ( \mathcal { G } _ { t } , \mathcal { G } _ { 0 } , t )$   
▷ per-site flow/del rates over $\mathcal { E } ;$ per-atom add rates over V   
7: // Bregman divergence loss   
8: Compute L and update θ via back-propagation:   
$\mathcal { L } = \sum _ { e \in \mathcal { E } } \left( \lambda _ { e } ^ { \mathrm { f o w } } + \lambda _ { e } ^ { \mathrm { d e l } } \right) + \sum _ { a \in \mathcal { V } } \lambda _ { a } ^ { \mathrm { a d d } } \ - \ \frac { \kappa ^ { \prime } ( t ) } { 1 - \kappa ( t ) } \sum _  \substack { c ^ { * } = { c } ^ { d } \} \in \mathbf { M } _ { \mathrm { t a r s e } } } \left\{ \begin{array} { l l } { \log \lambda _ { a } ^ { \mathrm { a d d } } + \log p _ { \theta } ^ { \mathrm { s i n } } ( e ^ { d } \mid a , \mathcal { E } ) } & { \mathrm { i f ~ } e ^ { s } = \mathcal { Q } } \\ { \log \lambda _ { e } ^ { \mathrm { d e l } } } & { \mathrm { i f ~ } e ^ { d } = \mathcal { Q } } \end{array} \right.$

Algorithm 3 $\mathbf { S } \mathbf { A M P L E M o v E S } ( \mathcal { G } _ { t } , \ \lambda ^ { \mathrm { H o w } } , \ \lambda ^ { \mathrm { d e l } } , \ \lambda ^ { \mathrm { a d d } } , \ \mathcal { E } , \ \Delta t , \ \tau )$   
Require: Rates $\lambda ^ { \mathrm { f l o w } } , \lambda ^ { \mathrm { d e l } }$ (per site), $\lambda ^ { \mathrm { a d d } }$ (per atom), electron sites $\mathcal { E } ,$ step size $\Delta t ,$ temperature τ   
Ensure: Set of moves $\mathbf { M } _ { \mathrm { s t e p } }$   
1: // Fire sites via thinned Poisson process   
2: for each site $e \in { \mathcal { E } }$ do   
3: $p _ { e } ^ { \mathrm { s i t e } } \gets 1 - \exp \bigl ( - ( \lambda _ { e } ^ { \mathrm { f o w } } + \lambda _ { e } ^ { \mathrm { d e l } } ) \Delta t / \tau \bigr )$   
4: Sample $f _ { e } \sim$ Bernoull $\mathrm { i } ( p _ { e } ^ { \mathrm { s i t e } } )$   
5: end for   
6: $\mathcal { F }  \{ e : f _ { e } = 1 \}$ ▷ fired sites   
7: // Fire addition atoms   
8: for each atom $a \in \nu$ do   
9: $p _ { a } ^ { \mathrm { a d d } }  1 - \exp \bigl ( - \lambda _ { a } ^ { \mathrm { a d d } } \Delta t / \tau \bigr )$   
10: Sample $f _ { a } \sim$ Bernoull $\operatorname { i } ( p _ { a } ^ { \mathrm { a d d } } )$   
11: end for   
12: $\mathcal { A } _ { \mathrm { a d d } }  \{ a : f _ { a } = 1 \}$ ▷ fired addition atoms   
13: // Resolve actions for fired sites   
14: $\mathbf { M } _ { \mathrm { f l o w } } , \ \mathbf { M } _ { \mathrm { d e l } }  \emptyset , \ \emptyset$   
15: for each $e ^ { \mathrm { f i r e d } } \in \mathcal { F }$ do   
$\breve { \int } \bar { f } d o w \quad \mathrm { w . p . ~ } \lambda _ { e } ^ { \mathrm { f o w } } / ( \lambda _ { e } ^ { \mathrm { f o w } } + \lambda _ { e } ^ { \mathrm { d e l } } )$   
16: action ←   
del otherwise   
17: if action = flow then   
18: Sample sink $e ^ { d } \sim p _ { \theta } ^ { \mathrm { s i n k } } ( { \cdot } \mid e ^ { \mathrm { f i r e d } } , \mathcal { E } )$   
19: $\mathbf { M } _ { \mathrm { H o w } }  \mathbf { M } _ { \mathrm { H o w } } \cup \{ ( e ^ { \mathrm { f i r e d } } , e ^ { d } ) \}$   
20: else   
21: $\mathbf { M } _ { \mathrm { d e l } }  \mathbf { M } _ { \mathrm { d e l } } \cup \{ ( e ^ { \mathrm { f i r e d } } , \emptyset ) \}$   
22: end if   
23: end for   
24: Resolve conflicts in $\mathbf { M } _ { \mathrm { H o w } } \colon$ keep highest $\lambda _ { e } ^ { \mathrm { f l o w } }$ per destination   
// Resolve actions for fired additions   
26: $\mathbf { M } _ { \mathrm { a d d } }  \infty$   
27: for each $a \in \mathcal { A } _ { \mathrm { a d d } }$ do   
28: Sample sink $\hat { e } ^ { \bar { d } } \sim p _ { \theta } ^ { \mathrm { s i n k } } ( \cdot \mid a , \mathcal { E } )$   
29: $\mathbf { M } _ { \mathrm { a d d } }  \mathbf { M } _ { \mathrm { a d d } } \cup \{ ( \emptyset , e ^ { d } ) \}$   
30: end for   
31: return $\mathbf { M } _ { \mathrm { s t e p } }  \mathbf { M } _ { \mathrm { f l o w } } \cup \mathbf { M } _ { \mathrm { d e l } } \cup \mathbf { M } _ { \mathrm { a d d } }$

Algorithm 4 MAELLE’s Inference   
Require: Reactant graph $\mathcal { G } _ { 0 }$ , trained model $u _ { \theta }$ , number of steps N, temperature τ   
Ensure: Predicted product graph $\hat { \mathcal { G } } _ { 1 }$   
1: $\mathcal { G } _ { t }  \mathcal { G } _ { 0 } , \quad \Delta t  1 / N$   
2: Encode reactant: $\mathbf { z } _ { 0 } \gets \mathrm { E N C O D E } _ { u _ { \theta } } ( \mathcal { G } _ { 0 } )$ ▷ computed once, reused at every step   
3: for $n = 0 , \ldots , N - 1$ do   
4: $t  n / N$   
5: $\Delta t _ { \mathrm { a d a p t } } \gets \Delta \mathrm { D A P T I V E S T E P S I Z E } ( \Delta t , ~ t , ~ \kappa )$ ▷ adaptive h from schedule   
6: // Decode current state   
7: $\lambda ^ { \mathrm { { f o w } } } , \lambda ^ { \mathrm { { d e l } } } , \lambda ^ { \mathrm { { a d d } } } , \mathcal { E } \gets \mathrm { { D E C O D E } } _ { u _ { \theta } } ( \mathcal { G } _ { t } , \mathbf { z } _ { 0 } , t )$   
▷ per-site flow/del rates; per-atom add rates; E: electron sites in $\mathcal { G } _ { t }$   
8: // Sample and apply moves   
9: $\mathbf { M _ { \mathrm { s t e p } } } ^ { \bullet }  \mathbf { S A M P L E M O V E S } ( \mathcal { G } _ { t } , \lambda _ { \mathrm { - } } ^ { \mathrm { f o w } } , \lambda ^ { \mathrm { d e l } } , \lambda ^ { \mathrm { a d d } } , \mathcal { E } , \Delta t _ { \mathrm { a d a p t } } , \tau )$ ▷ Alg. 3   
10: $\mathcal { G } _ { t }  \mathbf { A }$ PPLYELECTRONEDITS $\left( \mathcal { G } _ { t } , \ \mathbf { M } _ { \mathrm { s t e p } } \right)$   
11: end for   
12: return $\hat { \mathcal { G } } _ { 1 }  \mathcal { G } _ { t }$

## E Neural Network Architecture

MAELLE uses a conditional encoder-decoder architecture. The encoder processes the fixed reactant graph $\mathcal { G } _ { 0 }$ to produce conditioning representations; the decoder processes the time-dependent intermediate graph $\mathcal { G } _ { t }$ and cross-attends to the encoder outputs. Both operate at two levels of granularity: atoms and electron sites (entries).

Notation. Let d denote the model dimension, N the number of heavy atoms in a graph, and $\begin{array} { r } { N _ { e } = \binom { N } { 2 } + 2 N } \end{array}$ the number of electron sites (valence, hydrogen, and bond entries). We write h $\in \mathbb { R } ^ { N \times d }$ for atom embeddings and e $\in \mathbb { R } ^ { N _ { e } \times d }$ for entry embeddings.

## E.1 Graph encoder (GNN backbone)

Both the encoder and decoder share the same GNN architecture but with separate weights. The backbone is a stack of L Graph Attention Network (GAT) layers [49]. Intermediate layers use multi-head attention with 4 heads and concatenation (output dimension $4 \times d _ { \mathrm { h i d d e n } } )$ , while the final layer uses a single head without concatenation (output dimension d). An ELU activation is applied between intermediate layers.

Input atom features are first projected via a linear layer $\mathbf { h } ^ { ( 0 ) } = W _ { \mathrm { n o d e } } \mathbf { x } + \mathbf { b } _ { \mathrm { n o d e } }$ , where $\mathbf { x } \in \mathbb { R } ^ { N \times d _ { \mathrm { i t } } }$ are the raw atom features.

Time embedding. The decoder’s GNN incorporates the flow time $t \in [ 0 , 1 ]$ via a sinusoidal positional embedding followed by a two-layer MLP with SiLU activation:

$$
\mathbf { t } _ { \mathrm { e m b } } = \mathrm { M L P } _ { \mathrm { t i m e } } ( \mathrm { S i n E m b } ( t ) ) \in \mathbb { R } ^ { d _ { \mathrm { h i d d e n } } } ,
$$

which is broadcast-added to the initial node embeddings $\mathbf { h } ^ { ( 0 ) }$ before the GAT layers. The encoder’s GNN omits the time embedding, as the reactant graph $\mathcal { G } _ { 0 }$ is fixed.

## E.2 Two-level attention

After the GNN backbone, both encoder and decoder refine representations through two stages of attention operating at different granularities.

Stage 1: Atom-level attention. The atom embeddings $\mathbf { h } \in \mathbb { R } ^ { N \times d }$ from the GNN are passed through a Transformer. In the encoder, this is a TransformerEncoder (self-attention only). In the decoder, this is a TransformerDecoder: each layer applies self-attention over the decoder’s atom embeddings, followed by cross-attention to the encoder’s atom embeddings. This allows the decoder to condition on the reactant atom representations at every layer.

Stage 2: Entry-level feature construction and attention. Atom embeddings are lifted to electronsite (entry) embeddings via three learned projections:

• Valence entries (lone pairs): $\mathbf { e } _ { i } ^ { \mathrm { v a l } } = \mathrm { R e L U } ( W _ { \mathrm { v a l } } [ \mathbf { h } _ { i } \ \lVert \phi ( n _ { i } ^ { \mathrm { n b } } ) ] )$ for each atom $i ,$ where $n _ { i } ^ { \mathrm { n b } }$ is the non-bonding electron count and $\phi ( \cdot )$ is a 4-bin one-hot encoding of the electron pair count (0, 1, 2, ≥3 pairs).

• Hydrogen entries: ${ \mathbf { e } } _ { i } ^ { H } = \mathrm { R e L U } ( W _ { H } [ { \bf h } _ { i } \parallel \phi ( n _ { i } ^ { H } ) ] )$ for each atom $i ,$ where $n _ { i } ^ { H }$ is the hydrogen electron count.

• Bond entries: $\mathbf { e } _ { i j } ^ { \mathrm { b o n d } } = \mathrm { R e L U } ( W _ { \mathrm { b o n d } } [ \mathrm { a g g } ( \mathbf { h } _ { i } , \mathbf { h } _ { j } ) \parallel \rho ( d _ { i j } ) \parallel \phi ( n _ { i j } ^ { \mathrm { b o n d } } ) ] )$ for each realized bond $( i , j )$ , where $\arg \mathrm { g } ( \cdot , \cdot )$ is a symmetric aggregation (sum, mean, Hadamard, or bilinear), $\rho ( d _ { i j } )$ is a one-hot encoding of the topological distance (up to 10 hops), and $n _ { i j } ^ { \mathrm { b o n d } }$ is the bond electron count.

These three sets of entry embeddings are interleaved into a single sequence of $N _ { e }$ entries per graph and passed through another Transformer block — self-attention in the encoder, self-attention plus cross-attention to the encoder’s entry embeddings in the decoder.

## E.3 Output heads

The decoder produces three types of predictions from the entry embeddings, corresponding to the source–sink factorization in Eq. (8):

Rate head. A 3-layer MLP (with ReLU activations and hidden dimension 2d) maps each entry embedding to three non-negative scalars via softplus:

$$
[ \lambda _ { \theta , t } ^ { \mathrm { f o w } } ( e ) , \lambda _ { \theta , t } ^ { \mathrm { d e l } } ( e ) , \lambda _ { \theta , t } ^ { \mathrm { a d d } } ( e ) ] = \mathrm { s o f t p l u s } ( \mathrm { M L P } _ { \mathrm { r a t e } } ( \mathbf { e } ) ) .
$$

The addition rate $\lambda _ { \theta , t } ^ { \mathrm { a d d } }$ is only used for atom-level entries (valence sites); for bond entries it is masked out.

Sink heads. Two separate MLPs parameterize the sink distributions $p _ { \theta } ^ { \mathrm { s i n k , f l o w } }$ and $p _ { \theta } ^ { \mathrm { s i n k , a d d } }$ . Given a fired source entry with embedding $\mathbf { e } ^ { s }$ and a candidate destination with embedding $\mathbf { e } ^ { d }$ , the unnormalized score is:

$$
s ( e ^ { s } , e ^ { d } ) = \mathrm { M L P } _ { \mathrm { s i n k } } ( [ \mathbf { e } ^ { s } \parallel \mathbf { e } ^ { d } ] ) ,
$$

where $[ \cdot \parallel \cdot ]$ denotes concatenation and the MLP maps $\mathbb { R } ^ { 2 d } $ R. The sink distribution is obtained by applying softmax over all valid candidates. The candidate sets are kept local: for FLOW, the destination must share at least one atom with the source, restricting the candidate set to $O ( | \nu | )$ per fired source; for ADD from atom $^ { a , }$ the candidates are $\{ e _ { a , a } \} \cup \{ e _ { a } ^ { H } \} \cup \{ e _ { a , j } \mid j \neq a \}$ , yielding at most $| \nu | + 1$ candidates. For virtual bond entries (destinations that do not yet exist in $\mathcal { G } _ { t } )$ , representations are constructed on-the-fly from the atomic embeddings of the endpoint atoms.

## E.4 Training details

The full MAELLE model has approximately 16M trainable parameters with hidden dimension $d = 2 5 6$ throughout. The GNN backbone uses 2 GAT layers; the encoder applies 4 fully-connected attention layers (2 atom-level + 2 entry-level), and the decoder applies 6 fully-connected attention layers (3 atom-level + 3 entry-level), with the latter two stages including cross-attention to the encoder. We train for ∼50 epochs (about 72 hours on a single NVIDIA H100 GPU) with an effective batch size of 64, using gradient accumulation to fit memory.

## E.5 LLM-as-a-judge for benchmarking side product prediction

To evaluate the chemical plausibility of predicted products beyond exact-match top-k accuracy (Section 5.4), we use a large language model as a chemistry judge. Concretely, we query gemini-3-flash with minimal thinking effort. For each reaction, the LLM is given the reactant SMILES and a anonymized list of candidate products pooled across MAELLE, Molecular Transformer, and Graph2SMILES (with the ground-truth product additionally injected as a sanitycheck item). The judge has no information about which model produced which candidate. Each of the three independent runs uses a fresh shuffle and a non-zero sampling temperature, providing the standard deviation reported in Figure 3a.

Prompt structure. The query consists of a system prompt followed by a per-reaction user message assembled from three templated blocks (PREFIX, INPUTS, SUFFIX):

System prompt. You are an expert organic chemist. Your task is to evaluate whether predicted products are chemically plausible outcomes of a given reaction. A product is plausible if it could reasonably arise from the listed reactants under standard organic chemistry conditions, considering valence, connectivity, common reactivity, and likely mechanisms. You do not need the product to be the major product — minor or side products are acceptable as long as they are mechanistically reasonable.

Prefix. Reaction <reaction\_id>. Reactants (SMILES): <reactant\_smiles>.

Inputs. Below is a list of candidate products. Evaluate each candidate independently.

[1] <product\_smiles\_1>

[2] <product\_smiles\_2>

Suffix. For each candidate, provide a briefmechanistic reasoning and a plausibility verdict (true/false). Return your answer as a JSON object of the form:

```jsonl
{
"verdicts": [
{"index": 1, "reasoning": "...", "plausible": true},
{"index": 2, "reasoning": "...", "plausible": false},
]
}
```

Aggregation. For each reaction and each model, the top-k plausibility rate is the fraction of the model’s first k predictions that the judge marks plausible. We report the mean and standard deviation over three independent LLM runs (Figure 3a). To control for systematic over- or underpermissiveness of the judge, we additionally inject the ground-truth product into the candidate list and report its plausibility coverage as a sanity check; across runs we obtain 81.1 ± 0.1% ground-truth coverage, indicating a moderately conservative but stable judge.

Anonymization and de-duplication. Because MAELLE typically yields fewer unique candidates than beam-search SMILES models, for every test case we cap the per-model candidate list at the smallest number of unique predictions across the three models, ensuring a balanced comparison. Within a single query the candidate list is shuffled, and SMILES are canonicalized with RDKit before being passed to the judge so that surface-form differences do not bias the verdict.

## F Dataset

All experiments use the USPTO-480K reaction corpus [50, 51]. This appendix describes the preprocessing applied to the corpus (Appendix F.1), the standard and re-split partitionings used in the main text, and the construction of the out-of-distribution test axes (Appendix F.3).

## F.1 Preprocessing

The following steps are applied once to every reaction in the corpus, prior to any split.

Re-mapping. USPTO-480K ships with atom-mapped SMILES of the form reactants>reagents>products, but the provided maps are noisy and occasionally inconsistent. Because MAELLE’s electron-site alignment relies on a correct atom correspondence between reactants and products (Section 4.1), we discard the original maps and re-map every reaction with RXNMapper [30], yielding a consistent atom mapping across the corpus.

Electron-move interpolation. For each re-mapped reaction we solve the optimal-transport problem of Section 4.3 once, ahead of training, to obtain the move set M that transforms the reactant occupation $o _ { x }$ into the product occupation $o _ { y }$ . These precomputed moves are cached and reused across epochs; the per-step Bernoulli sampling of the mixture path (Section 4.4) is then applied at training time.

Atom maps and model inputs. The re-mapped correspondence is used to construct MAELLE’s occupation targets and is therefore retained for our model. For the SMILES- and graph-based baselines, which do not consume atom maps, maps are stripped from the inputs so that all baselines receive the same map-free reactions they were designed for. This is a property of each model’s input format, not an asymmetry in the data: every model sees the same reactions, and only MAELLE makes use of the atom correspondence.

Deduplication. Exact duplicates are identified by a canonical reaction key canon(reactant) ≫ canon\_main(product), where canon\_main selects the largest product fragment by heavy-atom count after stripping atom maps and canonicalizing with RDKit [52]. Pairwise overlap between all splits is verified to be zero on this key.

## F.2 Standard and low-MW splits

For the standard benchmark (Section 5.2) we use the corpus’s original train/validation/test partition. For the out-of-distribution study (Section 5.3) we re-partition the corpus as follows. The base train/validation/IID-test partition is constructed by Gaussian sampling over the product molecular weight (MW), with mean 250 Da and variance 1643.782 Da<sup>2</sup> (std ≈ 40.5 Da); we refer to this as the low-MW split. The deliberately low mean ensures the training distribution is well separated from the high-mass OOD axis below. The pooled train+validation set is partitioned 80/20, giving 304 228 training reactions after deduplication, a validation set of 37 582, and an IID test set of 37 578 drawn from the remaining half of the validation fold.

## F.3 Out-of-distribution test axes

OOD Mass (high molecular weight). Similary to the training corpus, a test set was constructed using Gaussian sampling with mean 750 Da and variance of 6575.13 Da<sup>2</sup>. The parameters of the Gaussian were chosen in order to ensure a clear separation between this set and the training-validation one. The test set described will be called hereon test\_ood\_mass (38 909 reactions, median product MW 576 Da). Reactions that were present in test\_ood\_mass and also in the training-validation set were eliminated in the latter. Because the Molecular Transformer must generate the full product SMILES autoregressively, the length of the output sequence scales with molecular size; the mechanism model, which predicts only local bond edits, is invariant to this factor. This split therefore directly probes sensitivity to molecular size

OOD Ester (non-methyl ester chemistry). An additional axis was constructed to isolate esterforming reactions that produce linkages other than methyl esters – a transformation class underrepresented in the training set relative to its synthetic importance. Reactions belonging to the following named reaction classes were excluded from all training and IID evaluation splits and instead pooled into a dedicated test set, test\_ood\_ester (1779 reactions): Ethyl esterification, CO<sub>2</sub>H-tBu protection, O Acetylation, O-Piv protection, O-Formylation, Acetoxy thioether synthesis, Salol reaction, Yamaguchi esterification, Yamaguchi lactonization, Shiina macrolactonization, 2-Benzofuranone synthesis, and Baeyer–Villiger oxidation. In addition, eight reaction classes that produce ester linkages of ambiguous type (Esterification, Ester Schotten–Baumann, Fischer–Speier esterification, Steglich esterification, Mitsunobu ester synthesis, Transesterification, Carboxylic anhydride alcoholysis, and Diazoalkane esterification) were dropped from all splits entirely to prevent soft leakage of ester-chemistry signal into the training distribution.

## F.4 Summary statistics

Table 5: Split sizes, sampling MW (mean ± std of the Gaussian over product molecular weight), and OOD criteria for uspto\_unified\_v3.
<table><tr><td>Split</td><td>N</td><td>MW  $( \mathrm { m e a n } \pm \mathrm { s t d } )$  [Da]</td><td>OOD criterion</td></tr><tr><td>train</td><td>304228</td><td> $3 1 1 . 9 \pm 9 0 . 6$ </td><td>一</td></tr><tr><td>val</td><td>37582</td><td> $3 1 2 . 7 \pm 9 0 . 6$ </td><td></td></tr><tr><td>test_iid</td><td>37578</td><td> $3 1 1 . 3 \pm 9 0 . 5$ </td><td>none (IID)</td></tr><tr><td>test_ood_mass</td><td>38 909</td><td> $5 8 5 . 2 \pm 8 3 . 0$ </td><td>product MW &gt; 529 Da</td></tr><tr><td>test_ood_ester</td><td>1779</td><td> $3 2 1 . 1 \pm 1 1 2 . 1$ </td><td>non-methyl ester chemistry</td></tr></table>