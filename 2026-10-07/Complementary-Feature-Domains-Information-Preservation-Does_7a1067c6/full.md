# Complementary Feature Domains: Information Preservation Does Not Imply Predictive-Contribution Preservation

Timothy Oladunni and Farouk Ganiyu-Adewumi

Department of Computer Science, Morgan State University

Baltimore, Maryland, USA

Email: timothy.oladunni@morgan.edu

Abstract—Complementary Feature Domains (CFD) theory characterizes predictive value as a context-indexed contribution system induced jointly by representations and their realization family. We show that Shannon-information preservation does not imply preservation of this contribution system: an invertible repre sentation transformation can leave target information unchanged while altering predictive contribution under a restricted decision family. We formalize the resulting transition through a CFD contribution defect that measures how contextual contributions change under controlled recoding. For bounded Lipschitz utility, we show that each coalition utility shift is bounded by the behavioral distance between the attainable action sets before and after recoding; consequently, every contextual contribution defect is bounded by the sum of the corresponding coalition incompatibilities. Exact behavioral closure yields invariance, while increasingly accurate compensation yields restoration. A controlled ECG experiment illustrates the mechanism: a nonlinear bijective recoding preserves the information in a frozen time–frequency representation but changes accuracy under a fixed affine learner; applying the exact inverse restores all tested coalition accuracies. The result separates information preservation from realizationdependent contribution and provides a quantitative transition law for multi-representation prediction.

## I. INTRODUCTION

Modern prediction systems rarely operate on a single immutable description of an observation. The same underlying source may be represented in time, frequency, time–frequency, learned, compressed, or otherwise transformed coordinates, and these representations are then fused under a particular model class. A natural intuition is that an invertible recoding should be harmless: if $D ^ { \prime } = g ( D )$ for a bijection g, then no Shannon information about the target has been lost, so $I ( D ^ { \prime } ; Y ) = I ( D ; Y )$ . Yet this conclusion is representationlevel, whereas prediction is realized through a restricted family of decision rules. If that family cannot implement the compensating map $g ^ { - 1 }$ , two information-equivalent encodings can induce different attainable decisions and therefore different predictive utility.

This mismatch matters especially in fusion. The practical question is not only whether a representation contains target information, but what it contributes after a particular set of other representations is already available. A domain can be weak in isolation but decisive in one coalition, useful in another, and redundant in a richer coalition. Consequently, changing the realizability of one representation can alter a pattern of marginal contributions across many contexts even though the transformed representation, and indeed every corresponding representation subset, preserves the same Shannon information. A context-free relevance score cannot describe that transition.

Existing frameworks address related but distinct questions. Shannon established the classical information measure underlying statistical dependence [1]; standard information-theoretic treatment further develops these quantities and their operational roles [2]. The information bottleneck formulates relevancepreserving compression [3]. Williams and Beer introduced PID to separate redundant, unique, and synergistic information supplied by multiple sources [4]. Bertschinger et al. formalized unique information through decision-theoretic constraints [5], while Griffith and Koch developed a measure of synergistic information [6]. Ince proposed a pointwise construction for multivariate redundancy [7], and Finn and Lizier developed a pointwise PID based on specificity and ambiguity lattices [8]. Predictive V-information instead makes usable information predictor-family dependent [9]. Blackwell comparison orders statistical experiments by their value for decision making [10]. These frameworks characterize information content, decomposition, usability, or decision informativeness. To our knowledge, none directly characterizes how a complete coalition-indexed predictive contribution system changes under representation recoding with an explicit realization family. CFD studies precisely this representation-to-contribution transition.

Complementary Feature Domains (CFD) theory, as developed here, concerns the stability and transition of contextual predictive contributions under representation–realization interventions. It is motivated by this missing transition problem. Our prior ECG studies show that domain combinations and fusion stage materially alter realized predictive benefit [12], [13]. These observations motivate separating represented content from realization-dependent contribution. Here we turn that observation into a formal question: when does an informationpreserving transformation preserve the contextual predictive contribution of a representation, and when can those contributions change despite preservation of Shannon information? For a representation system Φ and realization family A, we therefore define a coalition utility envelope and the marginal contribution of each representation in every admissible context. The resulting object is not a single importance score but a complete contribution system. We then study its transition from $( \Phi , A )$ to a recoded system $( g \Phi , B )$

Our main contribution is a quantitative representation– realization stability result. We define a behavioral compatibility error $\varepsilon _ { g } ( S ; A , B )$ for each coalition and prove

$$
\Gamma _ { g } ( i \mid S ; { \cal A } , { \cal B } ) \le L [ \varepsilon _ { g } ( S \cup \{ i \} ; { \cal A } , { \cal B } ) + \varepsilon _ { g } ( S ; { \cal A } , { \cal B } ) ] ,\tag{1}
$$

where $\Gamma _ { g }$ is the change in contextual contribution caused by the intervention. Thus information-preserving recoding does not by itself guarantee contribution invariance; behavioral compatibility does. Exact closure is the zero-error case, and vanishing compatibility error restores the original contribution system.

## II. CONTRIBUTION SYSTEMS

Let $Z \sim P _ { Z }$ be an underlying source, $Y = \tau ( Z )$ a target, and $\Phi = \{ \phi _ { 1 } , \ldots , \phi _ { m } \}$ a representation system with $D _ { i } = \phi _ { i } ( Z )$ For $S \subseteq [ m ]$ , write $D _ { S } = ( D _ { i } ) _ { i \in S }$ . Let $\mathcal { A } _ { S }$ be a declared family of decision rules available from coalition S. For bounded utility $u ,$ define

$$
V _ { \Phi , A } ^ { * } ( S ) = \operatorname* { s u p } _ { a \in { \cal A } _ { S } } \mathbb { E } [ u ( Y , a ( D _ { S } ) ) ] .\tag{2}
$$

Thus $V _ { \Phi , A } ^ { * } ( S )$ is not the information contained in coalition $S ;$ it is the best population-level utility that the declared realization family can attain from that coalition. It therefore depends jointly on the representation and on what the learner is permitted to realize.

For $i \not \in S ,$ , the contextual marginal contribution is

$$
\Delta _ { \Phi , A } ^ { * } ( i \mid S ) = V _ { \Phi , A } ^ { * } ( S \cup \{ i \} ) - V _ { \Phi , A } ^ { * } ( S ) .\tag{3}
$$

A positive $\Delta ^ { * } ( i | S )$ means that adding representation i unlocks predictive utility beyond what was already attainable from S; zero means that i adds no realizable utility in that context. Because the baseline coalition S changes, the same representation can have different contributions in different contexts.

The complete contribution system is

$$
{ \mathfrak { C } } ^ { * } ( \Phi , { \mathcal { A } } ) = \left( \Delta _ { \Phi , { \mathcal { A } } } ^ { * } ( i \mid S ) \right) _ { S \subseteq [ m ] , i \notin S } .\tag{4}
$$

Equation (4) stacks these local increments over the entire coalition lattice. Two systems can therefore have the same best full-coalition accuracy yet differ in how predictive value is distributed across representations and contexts.

If the coalition value is Shannon target information, $V _ { I } ( S ) =$ $I ( D _ { S } ; Y )$ , then the chain rule gives

$$
\Delta _ { I } ( i \mid S ) = I ( D _ { i } ; Y \mid D _ { S } ) .\tag{5}
$$

This identity provides the information-theoretic reference point: if coalition value is defined purely by Shannon information, the incremental value of i is exactly the target information in $D _ { i }$ that is not already supplied by $D _ { S }$

Consider componentwise bijections $g _ { i }$ and the recoded system $g \Phi ~ = ~ \{ g _ { i } \circ \phi _ { i } \} _ { i = 1 } ^ { m }$ . Under unrestricted decision rules, or whenever the post-recoding family implements the corresponding inverse compositions, the attainable decisions coincide and therefore ${ \mathfrak { C } } ^ { * } ( g \Phi , B ) = { \mathfrak { C } } ^ { * } ( \Phi , A )$ . The question is what can be said when this closure is only approximate.

## III. SEPARATION UNDER RESTRICTED REALIZATION

The transition problem is nonvacuous even when the representation change is perfectly information preserving. The following finite witness separates Shannon information from optimized utility under a fixed restricted family.

Lemma 1 (Information preservation does not imply realized-utility preservation). There exist finite-valued representations D and $D ^ { \prime } = g ( D )$ related by a bijection and a fixed realization family A such that

$$
I ( D ; Y ) = I ( D ^ { \prime } ; Y ) , \qquad V _ { D , A } ^ { * } \neq V _ { D ^ { \prime } , A } ^ { * } .\tag{6}
$$

Proof. Let four source states be equiprobable and let D take the values $( 0 , 0 ) , ( 0 , 1 ) , ( 1 , 0 ) , ( 1 , 1 )$ with labels $0 , 1 , 1 , 0 ,$ respectively. Thus Y is a deterministic function of D and $I ( D ; Y ) = H ( Y ) = 1$ bit. Define the support bijection

$$
\begin{array} { l l l } { { ( 0 , 0 ) \mapsto ( - 2 , - 1 ) , } } & { { \qquad } } & { { ( 0 , 1 ) \mapsto ( 1 , - 1 ) , } } \\ { { ( 1 , 0 ) \mapsto ( 2 , 1 ) , } } & { { \qquad } } & { { ( 1 , 1 ) \mapsto ( - 1 , 1 ) . } } \end{array}\tag{7}
$$

Because $g$ is bijective, $I ( g ( D ) ; Y ) = 1$ bit. Let A be affine binary classifiers with 0–1 utility. The original XOR support is not linearly separable and its optimal affine accuracy is $3 / 4$ In the recoded support, the two classes are separated by the sign of the first coordinate, so the optimal affine accuracy is 1. Hence (6) holds. □

This witness is intentionally elementary. Predictive Vinformation already explains why a restricted family can access different amounts of predictive information from the two encodings [9]. The CFD question begins one level higher: when one component of a multi-representation system is recoded, how do those accessibility changes propagate through all representation contexts?

Proposition 1 (Subset-information equivalence does not imply contribution equivalence). There exist multi-representation systems Φ and Ψ such that

$$
I ( D _ { S } ; Y ) = I ( D _ { S } ^ { \prime } ; Y ) \quad f o r e \nu e r y S \subseteq [ m ] ,\tag{8}
$$

while, under one fixed restricted realization family A,

$$
{ \mathfrak { C } } ^ { * } ( \Phi , { \mathcal { A } } ) \neq { \mathfrak { C } } ^ { * } ( \Psi , { \mathcal { A } } ) .\tag{9}
$$

Proof. Let R be the XOR representation in the lemma and $R ^ { \prime } = g ( R )$ . Let $U _ { 1 } , U _ { 2 }$ be independent fair nuisance bits, independent of $( R , Y )$ , and define $\Phi ~ = ~ ( U _ { 1 } , U _ { 2 } , R )$ and $\Psi = ( U _ { 1 } , U _ { 2 } , R ^ { \prime } )$ . Every corresponding subset is related by a bijection, so (8) follows from invariance of mutual information. Let $\mathcal { A } _ { S }$ be affine classifiers on the concatenated coordinates supplied by S. For any fixed affine score on $( R , U _ { 1 } , U _ { 2 } )$ averaging its 0–1 correctness over the independent nuisance pair $( U _ { 1 } , U _ { 2 } )$ cannot exceed the best affine correctness attainable on the replicated finite support; direct enumeration of the resulting affine dichotomies gives the same optimum as on R alone. Thus appending these independent nuisance coordinates leaves the optimum at $3 / 4$ for R and 1 for $R ^ { \prime }$ . Every coalition containing the third representation therefore differs, whereas coalitions excluding it are unchanged. At least one contextual marginal must therefore differ, proving (9). □

Thus even complete Shannon subset-information equivalence is insufficient for contribution-system equivalence; the missing condition is compatibility between the transformation and the realization family.

## IV. COMPATIBILITY DEFECT AND STABILITY

Let $( \mathcal { Q } , d )$ be the action space and suppose u is L-Lipschitz in its action argument:

$$
| u ( y , q ) - u ( y , q ^ { \prime } ) | \leq L d ( q , q ^ { \prime } ) .\tag{10}
$$

For coalition S, define the random-action sets

$$
\mathcal { H } _ { S } = \{ a ( D _ { S } ) : a \in \mathcal { A } _ { S } \} ,\tag{11}
$$

$$
\begin{array} { r } { \mathcal { H } _ { S } ^ { g } = \{ b ( g _ { S } ( D _ { S } ) ) : b \in \mathcal { B } _ { S } \} . } \end{array}\tag{12}
$$

We measure realization incompatibility by the symmetric $L ^ { 1 } ( P )$ Hausdorff distance

$$
\begin{array} { r } { \varepsilon _ { g } ( S ; \mathcal { A } , \mathcal { B } ) = \operatorname* { m a x } \Big \{ \underset { Q \in \mathcal { H } _ { S } } { \operatorname* { s u p } } \underset { Q ^ { \prime } \in \mathcal { H } _ { S } ^ { g } } { \operatorname* { i n f } } \mathbb { E } d ( Q , Q ^ { \prime } ) , } \\ { \underset { Q ^ { \prime } \in \mathcal { H } _ { S } ^ { g } } { \operatorname* { i n f } } \underset { Q \in \mathcal { H } _ { S } } { \operatorname* { i n f } } \mathbb { E } d ( Q , Q ^ { \prime } ) \Big \} . } \end{array}\tag{13}
$$

The two directed terms ask whether every action realizable before recoding can be approximated after recoding and vice versa. Hence $\varepsilon _ { g } ( S ) = 0$ means that the two attainable-action sets have the same $\textstyle \left( L ^ { 1 } ( P ) \right.$ closure on coalition S; if those sets are closed, they coincide. Larger values quantify realization incompatibility in the action metric relevant to utility.

Define the coalition utility response

$$
E _ { g } ( S ) = V _ { g \Phi , B } ^ { * } ( S ) - V _ { \Phi , A } ^ { * } ( S )\tag{14}
$$

Thus $E _ { g } ( S )$ is the intervention-induced change in the best attainable utility of coalition S: its sign indicates whether recoding helps or hurts that coalition under the declared realization families. We then difference these coalition responses to determine whether the role of an individual representation changes. Define the CFD contribution defect

$$
\Gamma _ { g } ( i \mid S ; { \cal A } , { \cal B } ) = \left| \Delta _ { g \Phi , { \mathscr { B } } } ^ { * } ( i \mid S ) - \Delta _ { \Phi , { \mathscr { A } } } ^ { * } ( i \mid S ) \right| .\tag{15}
$$

Accordingly, $\Gamma _ { g } ( i \mid S )$ is zero precisely when the intervention leaves the contextual marginal role of i unchanged for that comparison. A nonzero value does not by itself imply information loss; it says that the contribution realized from i after S changed under the representation–realization intervention. Operationally, $\Gamma _ { g }$ follows directly from pre/post-intervention coalition utilities and does not require estimating $\varepsilon _ { g } ;$ the latter explains and bounds the transition and can be approximated over finite candidate realization families when needed.

Theorem 1 (Approximate representation–realization invariance). For every coalition S,

$$
| E _ { g } ( S ) | \leq L \varepsilon _ { g } ( S ; \mathcal { A } , B ) .\tag{16}
$$

The inequality converts behavioral proximity into utility stabil ity: if the two realization families can closely reproduce one another’s actions on S, then the optimal value assigned to S cannot move far.

Consequently, for i /∈ S, the contribution defect satisfies (1). The two terms in that bound have a direct interpretation: one measures compatibility before i is added (coalition S), and the other after it is added (coalition $S \cup \{ i \} ) .$ . Thus contextual contribution is stable whenever the learner is behaviorally compatible on both sides of the marginal comparison. Equivalently, with $\begin{array} { r } { \| \mathfrak { C } ^ { * } ( g \Phi , \mathcal { B } ) - \mathfrak { C } ^ { * } ( \Phi , \mathcal { A } ) \| _ { \infty } : = \operatorname* { m a x } _ { S , i \notin S } \Gamma _ { g } ( i \mid S ; \mathcal { A } , \mathcal { B } ) } \end{array}$ the theorem yields the system-level stability law

$$
\| \mathfrak { C } ^ { * } ( g \Phi , B ) - \mathfrak { C } ^ { * } ( \Phi , A ) \| _ { \infty } \leq 2 L \operatorname* { m a x } _ { \mathfrak { c } } \varepsilon _ { g } ( S ; A , B ) .\tag{17}
$$

Proof. Fix S. For $Q \in \mathcal { H } _ { S }$ and $Q ^ { \prime } \in \mathcal { H } _ { S } ^ { g }$ , Lipschitz continuity implies

$$
\lvert \mathbb { E } u ( Y , Q ) - \mathbb { E } u ( Y , Q ^ { \prime } ) \rvert \leq L \mathbb { E } d ( Q , Q ^ { \prime } ) .\tag{18}
$$

For each Q, approximate the relevant infimum over $\mathcal { H } _ { S } ^ { g }$ , then take the supremum over Q. This bounds $V _ { \Phi , A } ^ { * } ( S ) - V _ { g \Phi , B } ^ { * } ( S )$ by L times the first directed term of (13). Reversing the two behavioral sets bounds the opposite difference by the second directed term, proving (16). Finally,

$$
\Gamma _ { g } ( i \mid S ) = | E _ { g } ( S \cup \{ i \} ) - E _ { g } ( S ) |\tag{19}
$$

$$
\leq | E _ { g } ( S \cup \{ i \} ) | + | E _ { g } ( S ) | ,\tag{20}
$$

and substitution gives (1).

Corollary 1 (Exact closure and restoration). $I f \varepsilon _ { g } ( S ; { \mathcal { A } } , B ) = 0$ for every coalition, then ${ \mathfrak { C } } ^ { * } ( g \Phi , B ) = { \mathfrak { C } } ^ { * } ( \Phi , A )$ . More generally, if a sequence $( A _ { k } , B _ { k } )$ satisfies max<sub>S</sub> $\varepsilon _ { g } ( S ; \mathcal { A } _ { k } , \mathcal { B } _ { k } )  0 ,$ then the maximum contextual contribution defect converges to zero.

The metric comparison underlying (16) is standard; the CFD contribution is the coalition-indexed transition object and its induced bound on changes in contextual predictive role. In (15), differencing coalition responses across contexts converts behavioral incompatibility into a uniform control on changes in contextual predictive role. The bound is context local: the defect for adding i after S depends only on S and S ∪ {i}. It is also falsifiable: if compensating families drive both behavioral errors to zero while the population contribution defect stays bounded away from zero, the stated mechanism cannot explain that defect. Exact inverse closure instead forces restoration.

## V. RELATION TO EXISTING INFORMATION-THEORETIC OBJECTS

Table I states the novelty boundary. CFD does not redefine mutual information, PID atoms, usable information, or Blackwell ordering. Its primary object is the transition of a coalition-indexed predictive contribution system under a declared representation–realization intervention.

At the unrestricted information level, let actions be all conditional predictive distributions and let utility be log score. Then the optimal coalition value is

$$
V ^ { * } ( S ) = - H ( Y \mid D _ { S } ) ,\tag{21}
$$

TABLE I  
ANALYTICAL OBJECTS AND THE DISTINCT CFD TRANSITION QUESTION.
<table><tr><td>Framework</td><td>Primary object</td><td>Question answered</td></tr><tr><td>MI/CMI [2]</td><td>Target dependence</td><td>How much target information is present?</td></tr><tr><td>PID [4]</td><td>Information decom- position</td><td>How is target information distributed among sources?</td></tr><tr><td> $\nu _ { - }$ </td><td>Family-relative us-</td><td>How much information can a predictor family exploit?</td></tr><tr><td>information [9]</td><td>able information</td><td></td></tr><tr><td>Blackwell [10]</td><td>Statistical experiments</td><td>Which information structure is at least as useful for decisions?</td></tr><tr><td>CFD</td><td>Contribution transi-</td><td>Change of contextual roles under a</td></tr><tr><td></td><td>tion</td><td>representation-realization intervention</td></tr></table>

so the contextual marginal reduces to conditional mutual information,

$$
\Delta ^ { * } ( i \mid S ) = I ( Y ; D _ { i } \mid D _ { S } ) .\tag{22}
$$

Under restricted realization, however, Lemma 1 and Proposition 1 show that even equality of the entire Shannon subsetinformation profile need not preserve the optimized coalition utility function or its contextual differences. PID can therefore remain unchanged under an invertible source recoding while the decision-level contribution system changes. Predictive Vinformation captures the accessibility mechanism, whereas $\Gamma _ { g }$ measures the induced change in a representation’s marginal role across contexts. Blackwell comparison concerns ordering of experiments over decision problems; here the statistical information is held fixed by a bijection and the intervention instead probes compatibility with a declared restricted realization mechanism. Cooperative-game attribution is also adjacent but distinct. A Shapley value averages a representation’s marginal contribution over contexts [11], whereas CFD retains the unaveraged contextual marginal vector and studies how that vector changes under intervention. Averaging can therefore hide the location of a representation–realization mismatch that is visible in $\Gamma _ { g } ( i \mid S )$

These distinctions give the logical chain

information equivalence $\nRightarrow$ realization equivalence,

$$
\mathrm { i n f o r m a t i o n \ e q u i v a l e n c e \not = \ c o n t r i b u t i o n \ e q u i v a l e n c e , }
$$

behavioral realization equivalence ⇒ contribution equivalence.

(23)

The stability theorem identifies a sufficient quantitative bridge for the second implication: contribution differences vanish as the attainable behavioral sets before and after recoding approach one another.

## VI. REALIZATION-CAPACITY STRESS TEST

The finite witness in Lemma 1 also gives a direct capacity test of the proposed mechanism. We evaluated the same onebit XOR and bijectively recoded supports under increasingly expressive standard realization families. Table II reports the population optimum for affine rules and full-support fitted accuracy for the nonlinear families. The affine family exhibits the predicted .25 utility separation. Once the realization family can express both decision boundaries, the gap disappears.

TABLE II  
CAPACITY STRESS TEST. BOTH ENCODINGS CONTAIN EXACTLY ONE BIT OF TARGET INFORMATION.
<table><tr><td>Realization family</td><td>XOR</td><td>Recoded</td><td> $\mathrm { G a p }$ </td></tr><tr><td>Affine (exact optimum)</td><td>.75</td><td>1.00</td><td>.25</td></tr><tr><td>Quadratic logistic</td><td>1.00</td><td>1.00</td><td>0</td></tr><tr><td>RBF-kernel SVM</td><td>1.00</td><td>1.00</td><td>0</td></tr><tr><td>Decision tree, depth 2</td><td>1.00</td><td>1.00</td><td>0</td></tr></table>

TABLE III

EXACT FULL-CONTEXT CONTRIBUTION UNDER BAYES (B) AND AFFINE (A) REALIZATION.
<table><tr><td>Target</td><td> $V _ { B } ( 1 2 )$ </td><td> $V _ { B } ( 1 2 3 )$ </td><td> $\Delta _ { 3 , B } ( 1 2 )$ </td><td> $\Delta _ { 3 , A } ( 1 2 )$ </td></tr><tr><td>Parity</td><td>.500</td><td>1.000</td><td>.500</td><td>.250</td></tr><tr><td>Majority</td><td>.750</td><td>1.000</td><td>.250</td><td>.250</td></tr><tr><td>OR</td><td>.875</td><td>1.000</td><td>.125</td><td>.125</td></tr><tr><td>AND</td><td>.875</td><td>1.000</td><td>.125</td><td>.125</td></tr></table>

The information content is fixed while realization capacity changes. For classification accuracy, Theorem 1 applies to discrete prediction actions equipped with the discrete metric; the experiments evaluate the predicted invariance– failure mechanism rather than numerically estimating the theorem’s bound. The gap vanishes once the family realizes equivalent decision behavior on both supports. Predictive V-information diagnoses family-relative accessibility; CFD tracks how the resulting accessibility changes propagate into contextual coalition contributions and restoration.

## VII. CONTEXT AND CAPACITY ON A COALITION LATTICE

A second exact synthetic calculation isolates context dependence itself. Let $D _ { 1 } , D _ { 2 } , D _ { 3 }$ be independent fair bits and consider four deterministic targets under the uniform distribution on the eight states. For every coalition we computed Bayes-optimal accuracy and the exact best affine-threshold accuracy by enumerating projected dichotomies and testing separability. Table III reports the full-context marginal of $D _ { 3 }$ after $D _ { 1 } , D _ { 2 }$ are already available.

For parity, $D _ { 3 }$ contributes nothing in the empty or onecompanion contexts but contributes .5 after both other bits under Bayes realization; the affine family realizes only .25 of that full-context opportunity. In contrast, majority, OR, and AND show no Bayes–affine discrepancy in the reported fullcontext marginal. Thus context can create an informational opportunity while realization capacity determines how much of that opportunity becomes predictive utility. The parity contribution signatures are

$$
\mathbf { c } _ { 3 , B } = ( 0 , 0 , 0 , . 5 ) , \qquad \mathbf { c } _ { 3 , A } = ( 0 , 0 , 0 , . 2 5 ) ,\tag{24}
$$

with entries indexed by ∅, {1}, {2}, {1, 2}. A single contextfree “complementarity” score cannot represent this structure. Together with the recoding stress test, the battery separates

TABLE IV  
INFORMATION-PRESERVING RECODING AND INVERSE RESTORATION. INTERVALS ARE PAIRED 95% INTEGRITY-GROUP BOOTSTRAP INTERVALS FOR RECODED MINUS RAW ACCURACY.
<table><tr><td>Coalition</td><td>Raw</td><td>Recoded</td><td>Restored</td><td>Recoding gap</td></tr><tr><td> $T F$ </td><td>.7372</td><td>.6277</td><td>.7372</td><td>-.1095</td></tr><tr><td> $T + T F$ </td><td>.6569</td><td>.5839</td><td>.6569</td><td>-.0730</td></tr><tr><td> $F + T F$ </td><td>.7883</td><td>.5912</td><td>.7883</td><td>-.1971</td></tr><tr><td> $T + F + T F$ </td><td>.7372</td><td>.6058</td><td>.7372</td><td>-.1314</td></tr><tr><td colspan="3">95% intervals for the four gaps, respectively: [−.2073, .0357],  $[ - . 3 6 3 1 , - . 0 1 7 7 ]$ </td><td>[-.2391,-.0095], , and [−.2837, .0082].</td><td></td></tr></table>

two axes of CFD: where a representation contributes across the coalition lattice and how much of that contribution a realization family can attain.

## VIII. CONTROLLED RECODING EXPERIMENT

We tested the mechanism using frozen time (T), frequency $( F )$ , and time–frequency (TF) ECG embeddings from 928 observations used in prior CFD work [12]. Duplicate-linked observations were grouped before splitting into 491 integrity groups; the fixed split contained 648 training, 143 validation, and 137 sealed test observations. No integrity group crossed splits, and the post-split audit found no exact test-to-training duplicates in the constructed or frozen-embedding representations.

After train-fitted standardization, we applied the coordinatewise bijection

$$
g ( x ) = \mathrm { s g n } ( x ) | x | ^ { 3 } , \qquad g ^ { - 1 } ( z ) = \mathrm { s g n } ( z ) | z | ^ { 1 / 3 }\tag{25}
$$

to the $T F$ embedding. This transformation preserves all information in $T F$ but changes its geometry for an affine learner. We compared: (i) raw standardized embeddings, (ii) bijectively recoded embeddings under the same one-versusrest logistic learner, and (iii) recoded embeddings passed through the exact inverse before the same learner. All conditions used identical data, frozen embeddings, standardization, and learner settings; only the $T F$ intervention differed. The inverse condition applied analytical $g ^ { - 1 }$ before classification, not a learned decoder. Accuracy-difference uncertainty used 5,000 paired resamples of test integrity groups.

Table IV shows the predicted separation. The intervals exclude zero for $T F$ and $F + T F$ , despite the recoding being bijective. Applying the declared inverse restores every reported coalition accuracy exactly; the restoration error and its bootstrap interval are 0 and [0, 0] in every context. The cubic map is not proposed as an optimal ECG representation; it is a controlled, exactly invertible intervention that alters linear decision geometry. It therefore holds represented information fixed while perturbing compatibility with the declared realization family, directly testing the mechanism in Theorem 1.

Because the intervention acts only on $T F ,$ , every coalition not containing $T F$ is identical in the raw and recoded systems. Therefore, for each context $S \subseteq \{ T , F \}$ , the finite-sample analogue of the CFD contribution defect for adding $T F$ simplifies to the absolute recoding response of $S \cup \{ T F \}$ :

$$
\begin{array} { r l } & { \widehat { \Gamma } _ { g } ( T F \mid S ) = \Big \lvert \widehat { E } _ { g } ( S \cup \{ T F \} ) - \widehat { E } _ { g } ( S ) \Big \rvert } \\ & { \qquad = \Big \lvert \widehat { E } _ { g } ( S \cup \{ T F \} ) \Big \rvert . } \end{array}\tag{26}
$$

The second equality is specific to this intervention: because g touches only $T F$ , the response of every baseline coalition $S \subseteq \{ T , F \}$ is exactly zero. The empirical defect therefore becomes the absolute change in the corresponding coalition that contains $T F$ , making Table ?? a derived summary of Table ${ \mathrm { I V } } ,$ not a separate experiment.

Hence the observed finite-sample defects are .1095, .0730, .1971, and .1314 for contexts $\varnothing , \ \{ T \} , \ \{ F \}$ , and $\{ T , F \}$ respectively. The same information-preserving intervention therefore changes the realized role of $T F$ by different amounts depending on what other representations are available. This is the context-indexed transition that is invisible to a single global accuracy difference.

After applying $g ^ { - 1 }$ , the learner again receives the original standardized coordinates; the zero restored defect in all four contexts is therefore the finite-sample analogue of exact behavioral closure in Corollary 1.

## IX. DISCUSSION AND CONCLUSION

Information preservation and predictive-contribution preservation are distinct invariance requirements: a bijection can preserve Shannon information for every coalition while changing the best utility attainable by a restricted realization family. CFD makes this distinction explicit by indexing contribution by coalition and realization family. Its compatibility bound shows that $\Gamma _ { g } ( i \mid S )$ is controlled by realization incompatibility before and after adding i; exact closure forces invariance, whereas approximate closure bounds the change. A nonzero defect therefore need not indicate information loss, but may reveal representation–learner mismatch.

The experiments test the mechanism, not universal complementarity: the formulation is distribution independent, while ECG tests its predicted failure–restoration pattern in a nonsynthetic multi-representation system. The synthetic affine gap disappears for richer families; in ECG, recoding changes the time–frequency contribution across all four contexts, while exact inverse restoration returns every defect to zero. This closure–failure–restoration pattern is consistent with the predicted dependence on context and realization family. CFD complements PID, predictive V-information, Blackwell comparison, and Shapley attribution by retaining coalitionindexed contribution changes under representation intervention. The bound is sufficient; finite-sample error is outside the population theorem, and broader tasks, realization families, and uncertainty guarantees remain future work.

## REFERENCES

[1] C. E. Shannon, “A mathematical theory of communication,” Bell Syst. Tech. J., vol. 27, pp. 379–423, 623–656, 1948.

[2] T. M. Cover and J. A. Thomas, Elements of Information Theory, 2nd ed. Wiley, 2006.

[3] N. Tishby, F. C. Pereira, and W. Bialek, “The information bottleneck method,” in Proc. 37th Annu. Allerton Conf. Commun., Control, Comput., 1999, pp. 368–377.

[4] P. L. Williams and R. D. Beer, “Nonnegative decomposition of multivariate information,” arXiv:1004.2515, 2010.

[5] N. Bertschinger, J. Rauh, E. Olbrich, J. Jost, and N. Ay, “Quantifying unique information,” Entropy, vol. 16, no. 4, pp. 2161–2183, 2014.

[6] V. Griffith and C. Koch, “Quantifying synergistic mutual information,” in Guided Self-Organization: Inception. Springer, 2014, pp. 159–190.

[7] R. A. A. Ince, “Measuring multivariate redundant information with pointwise common change in surprisal,” Entropy, vol. 19, no. 7, Art. no. 318, 2017.

[8] C. Finn and J. T. Lizier, “Pointwise partial information decomposition using the specificity and ambiguity lattices,” Entropy, vol. 20, no. 4, Art. no. 297, 2018.

[9] Y. Xu, S. Zhao, J. Song, R. Stewart, and S. Ermon, “A theory of usable information under computational constraints,” in ICLR, 2020.

[10] D. Blackwell, “Equivalent comparisons of experiments,” Ann. Math. Statist., vol. 24, no. 2, pp. 265–272, 1953.

[11] L. S. Shapley, “A value for n-person games,” in Contributions to the Theory of Games II, H. W. Kuhn and A. W. Tucker, Eds. Princeton Univ. Press, 1953, pp. 307–317.

[12] T. Oladunni and A. Wong, “Rethinking multimodality: Optimizing mul timodal deep learning for biomedical signal classification,” IEEE Access, vol. 13, pp. 156436–156464, 2025, doi:10.1109/ACCESS.2025.3605315.

[13] T. Oladunni and E. Aneni, “Explainable deep neural network for multimodal ECG signals: Intermediate versus late fusion,” IEEE Access, vol. 13, pp. 202700–202736, 2025, doi:10.1109/ACCESS.2025.3631544.