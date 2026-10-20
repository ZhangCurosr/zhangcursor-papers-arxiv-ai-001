# Minimum-Cost Robust Evidence Certificates for Horn Reasoning under Correlated Source Failures

Yang Yukun<sup>1∗</sup>

<sup>1</sup>School of Management, Beijing Institute of Technology yangyukun2005@163.com

## Abstract

A small justification for a rule-based conclusion may disappear when a source is withdrawn, and multiple derivations need not provide independent evidence. We study minimumcost certificates that preserve a Horn conclusion under every withdrawal of at most k source domains. Evidence records remain distinct from asserted facts, supporting replicated attestations and overlapping failure dependencies. Transversals of minimal-support footprints characterize robustness and explain why disjoint-proof requirements can reject valid certificates. An exact procedure alternates weighted evidence selection with a source-failure oracle. A rule-closed counterinterpretation yields a globally valid selection cut without enumerating supports or failure scenarios. We prove correctness, finite termination, and cost optimality, establish coNP-completeness of unrestricted verification and NP-completeness of synthesis for every fixed budget, and identify a tractable reachability restriction. Experiments on controlled benchmarks, reachability graphs, and ontology-derived instances demonstrate that our counterinterpretation-guided synthesis achieves up to 92% optimum coverage within practical resource bounds, outperforming scenario expansion and general abduction while confirming the necessity of transversal-based reasoning over proof-packing heuristics.

## Introduction

Rule-based decisions often require premises that can be inspected, retained, or communicated. A release policy may combine signed test reports and attestations; a knowledge graph may infer relationships from several institutions’ records. Eficient materialization computes conclusions (Motik et al. 2014), while explanation and provenance expose their dependencies. When a provider becomes unavailable or is removed from an audit, the practical question is whether the retained evidence can still justify the decision.

An inclusion-minimal explanation removes unnecessary premises, and a minimum-cost explanation reduces inspection or disclosure cost. Neither protects against withdrawal of its own premises: one cheap support may concentrate the entire justification in a single supplier. Retaining the full knowledge base can preserve alternatives but exceed an audit budget. We therefore select a fixed evidence package before failures occur and allow its successful derivation to change afterward. The quantifier order is evidence selection, universal source withdrawal, and existence of a surviving justification. Recovery uses only the selected records, as required by an archived audit package; it does not depend on later evidence acquisition.

Counting proofs does not resolve this problem. Diferent derivations may reuse a record or depend on services within one failure domain. Conversely, requiring domain-disjoint proofs can reject evidence that survives every admissible withdrawal. Robustness depends on whether failed domains intersect all supports, not on whether supports can be packed without overlap.

Example 1 (Release authorization). Authorization requires any two of checks a, b, c, whose records depend on domains $5 _ { A } , s _ { B } , s _ { C }$ . Retaining all three survives any singledomain withdrawal because any two sufice, yet their support footprints $\{ s _ { A } , s _ { B } \} , \{ s _ { A } , s _ { C } \}$ , and $\{ s _ { B } , s _ { C } \}$ share domains pairwise, precluding disjoint proofs. An alternative attestation d independently authorizes release at higher cost. Optimal robust certification must navigate both inspection costs and failure dependencies.

While why-provenance (Buneman, Khanna, and Wang-Chiew 2001), semiring provenance (Green, Karvounarakis, and Tannen 2007), and circuit representations (Deutch et al. 2014) expose derivation dependencies, they do not select cost-optimal subsets resilient to source loss. Database resilience (Freire et al. 2015) and Datalog deletion causality (Thapa and Staab 2026) analyze answer vulnerability to tuple removals via minimal supports, but do not address ex-ante evidence synthesis under grouped, correlated domain withdrawals. Similarly, abduction-based explanations (Ignatiev, Narodytska, and Marques-Silva 2019), robust perturbation models (La Malfa et al. 2021), and lineage-driven fault injection (Alvaro, Rosen, and Hellerstein 2015) seek explanations or fault witnesses without synthesizing minimum-cost packages resilient to source loss. We turn this support–transversal principle into a cost-sensitive separation framework specialized for correlated source domains.

Our synthesis method alternates an exact weighted hittingset master with an adversarial oracle. A defeating withdrawal yields a rule-closed counterinterpretation and a positive clause that every robust selection must satisfy. This uses established implicit hitting-set separation (Saikko, Wallner, and Järvisalo 2016; Davies and Bacchus 2011), with globally valid record cuts for shared failures. Horn closure handles recursive rules without admitting unsupported cycles as proofs.

Our contributions are: (1) a record-level certification objective capturing correlated source failures; (2) a footprint characterization proving why proof packing is incomplete; (3) an exact synthesis algorithm with finite termination and cost optimality guarantees; (4) complexity results showing coNP-complete verification and NP-complete synthesis for fixed budgets; (5) a polynomial flow reduction for independent reachability; and (6) empirical evaluations demonstrating up to 92% optimum coverage and the necessity of cut strengthening over proof packing.

## Preliminaries

## Horn consequences and supports

Let V be a finite universe of ground atoms and P a finite definite Horn program. Each rule r has a head head $\vert ( r ) \in V$ and a finite body bod $\mathrm { y } ( r ) \subseteq V ;$ empty bodies are unconditional. Negation, priorities, and integrity constraints are excluded. For facts X, least-fixed-point consequences (Van Emden and Kowalski 1976) are

$$
\begin{array} { r l } & { X _ { 0 } = X , } \\ & { X _ { i + 1 } = X _ { i } \cup \{ \mathrm { h e a d } ( r ) : r \in P , \mathrm { ~ b o d y } ( r ) \subseteq X _ { i } \} , } \\ & { \mathrm { c l } _ { P } ( X ) = \bigcup _ { i \geq 0 } X _ { i } . } \end{array}\tag{1}
$$

At most |V| strict growth steps occur. Rule counters compute closure in linear time in the explicit ground representation (Dowling and Gallier 1984), which is the input model for all complexity results. First-order grounding is a separate cost.

Definition 1 (Support). Given trusted facts $B \subseteq V$ , selectable records E, and a mapping $x : \mathcal { E } \to V$ , a support for q is a record set $S \subseteq { \mathcal { E } }$ satisfying $q \in \mathrm { c l } _ { P } ( B \cup \bar { \alpha ( S ) } )$ It is minimal if no proper subset is a support. Here $\alpha ( S ) =$ $\{ \alpha ( e ) : e \in S \}$

Distinct records asserting the same atom maintain separate identities and failure dependencies, generalizing assumptionbased truth maintenance (de Kleer 1986). While rulederivation trees reconstruct acyclic proofs for recursive programs, minimal supports are defined by semantic entailment, distinct from cardinality or cost minimality. For a family ${ \mathcal { A } } ,$ $\operatorname { M i n } _ { \subseteq } ( { \mathcal { A } } )$ removes non-minimal supersets.

## Relation to provenance and abduction

Unlike recursive semiring provenance (Bourgaux et al. 2022) or why-provenance enumeration (Calautti et al. 2024a,b), which face exponential compilation blowup (Darwiche and Marquis 2002), we target polynomial Boolean entailment under retained subsets. Query causality (Meliou et al. 2010) and database resilience (Freire et al. 2015) study individual tuple deletions rather than adversarial domain withdrawals. While Horn abduction (Kakas, Kowalski, and Toni 1992), suficient reasons (Darwiche and Hirth 2020), and diagnosis (Reiter 1987) select static premises, they do not guarantee survival against subsequent failures; and unlike minimum label cuts (Zhang and Tang 2020), Horn conjunctions preclude graph reductions.

## Robust Evidence Certificates

## Records, domains, and the certification objective

Definition 2 (Certification instance). A certification instance is a tuple $\mathcal { T } = ( V , P , B , \mathcal { E } , D , \alpha , \gamma , c , q , k )$ . The first three components are the ground vocabulary, rules, and trusted facts. The finite sets $\dot { \varepsilon }$ and D contain evidence records and failure domains. Each record e asserts $\alpha ( e ) \in V$ , has dependencies $\gamma ( e ) \subseteq D ,$ , and incurs strictly positive rational cost $c ( e )$ . The target is $q \in V$ , and k is a nonnegative integer with $\dot { k } \leq | D |$

A domain represents a common withdrawal cause, such as a publisher or hosting service. Dependencies may overlap: failure of any domain in $\gamma ( e )$ invalidates e. An empty dependency set protects a record. Costs are paid once per retained record; rules and trusted facts remain fixed.

For a selected certificate $C \subseteq { \mathcal { E } }$ and failed domains $F \subseteq$ D, define the surviving records and asserted facts by

$$
\begin{array} { r l } & { \mathrm { s u r v } ( C , F ) = \{ e \in C : \gamma ( e ) \cap F = \emptyset \} , } \\ & { \quad X ( C , F ) = B \cup \alpha ( \mathrm { s u r v } ( C , F ) ) . } \end{array}\tag{2}
$$

Definition 3 (Robust certificate). A record set C is a k-robust certificate for $q$ if

$$
\forall F \subseteq D , \quad | F | \leq k \implies q \in \mathrm { c l } _ { P } ( X ( C , F ) ) .\tag{3}
$$

The synthesis task returns a feasible minimizer, or reports that none exists:

$$
\operatorname* { m i n } _ { C \subseteq \mathcal { E } } \mathrm { c o s t } ( C ) , \qquad \mathrm { c o s t } ( C ) = \sum _ { e \in C } c ( e ) ,
$$

subject to Equation (3).

(4)

The guarantee is formal entailment relative to a trust boundary, not objective truth. Withdrawal removes assertions without adding negations; conflicting reports require a separate semantics.

Example 2. Continuing Example 1, let $B = \varnothing$ with rules $q  a , b ; q  a , c ; q  b , c$ ; and $q  d .$ Records $e _ { A } , e _ { B } , e _ { C }$ assert $a , b ,$ c with domains $\{ s _ { A } \} , \{ s _ { B } \} , \{ s _ { C } \}$ and unit cost; $e _ { D }$ asserts d with $\{ s _ { D } \}$ at cost 3. For $\dot { k } = 0 ;$ , any pair of checks is optimal (cost 2). For $\boldsymbol { k } = 1 , \{ e _ { A } , e _ { B } , e _ { C } \}$ is optimal (cost 3), surviving any single-domain loss. For $k = 2$ , all four records are required (cost 5), since withdrawing s<sub>D</sub> plus one check leaves two checks, while withdrawing two check domains leaves d. Replicating a under $\{ s _ { A } \}$ provides no resilience, whereas a distinct domain expands feasible certificates.

## A source-level characterization

Let $S _ { q } ( C )$ be the inclusion-minimal supports contained in C. For each such support, its footprint is the union of its records dependencies. We retain only inclusion-minimal footprints:

$$
\begin{array} { r l } & { \lambda ( S ) = \bigcup _ { e \in S } \gamma ( e ) , } \\ & { \mathcal { G } _ { q } ( C ) = \operatorname { M i n } _ { \subseteq } \{ \lambda ( S ) : S \in \mathcal { S } _ { q } ( C ) \} . } \end{array}\tag{5}
$$

A transversal intersects every member of a family. Define $\tau ( \mathcal G )$ as the smallest transversal cardinality. The empty family has $\tau = 0$ , since no support exists; a family containing the empty set has $\tau = \infty$ , since that support cannot be intersected. Define the failure radius

$$
\rho _ { q } ( C ) = \operatorname* { m i n } \{ | F | : F \subseteq D , q \not \in \mathrm { c l } _ { P } ( X ( C , F ) ) \} ,\tag{6}
$$

with value ∞ when no withdrawal defeats the target.

Theorem 1 (Grouped support–cut equivalence). For every certificate candidate $C , \bar { \rho } _ { q } ( C ) = \tau \bar { ( } \mathcal { G } _ { q } ( C ) )$ . Consequently, C is k-robust exactly when $\tau ( \mathcal { G } _ { q } ( C ) ) ^ { \setminus } \bar { k } .$

Proof. Entailment $q ~ \in ~ \mathrm { c l } _ { P } ( X ( C , \bar { F } ) )$ holds if some minimal support $S \in { \cal S } _ { q } ( C )$ survives ${ \dot { F } } ,$ , which requires $F \cap \lambda ( S ) \stackrel { \textstyle \cdots } { = } \emptyset$ . Thus, $\bar { F }$ defeats $C ~ ( q ~ \notin ~ \mathrm { c l } _ { P } ( X ( C , ^ { ' } F ) ) )$ if F intersects every support footprint in $\mathcal { G } _ { q } ( C ) , \mathrm { i . e . , } \bar { F }$ is a transversal of $\mathcal { G } _ { q } ^ { \mathrm { ~ \bar { ~ } } } ( C )$ . Minimizing $| F |$ yields $\dot { \rho } _ { q } ( C ) =$ $\tau ( \mathcal { G } _ { q } ( C ) )$ ). Therefore, C is k-robust if no defeating withdrawal has $\mathrm { s i z e } \leq k ,$ , which is precisely $\tau ( \mathcal { G } _ { q } ( C ) ) > \mathbf { \bar { k } . } \boxed { }$

Proposition 1 (Disjointness is suficient but incomplete). Having $k + \bar { 1 }$ supports with pairwise disjoint nonempty footprints implies k-robustness. The converse fails already for $k = 1$

Proof. A failed domain can intersect at most one of the disjoint footprints, so k failed domains cannot intersect all $k + 1$ . In Example 2, the check certificate has footprints $\{ s _ { A } , s _ { B } \} , \{ s _ { A } , s _ { C } \}$ , and $\{ s _ { B } , s _ { C } \}$ . Every transversal needs two domains, yet no two footprints are disjoint. □

Figure 1 illustrates the running-example footprints and surviving supports. Footprints sufice for verification of a fixed certificate, but optimization still needs record identities: supports with the same footprint may have diferent costs, and shared records are charged once.

## Monotonicity and explanatory necessity

Proposition 2 (Feasibility and nominal reduction). Robustness is upward closed in $C$ and downward closed in k. A feasible certificate exists exactly when E is robust. At $k = 0$ , minimum-cost certification coincides with minimumcost positive Horn support selection.

Proof. Closure monotonicity ensures expanding C preserves surviving derivations under any $F ,$ while decreasing k restricts withdrawals. At $k = 0 , F { \overset { \cdot } { = } } \emptyset$ , reducing Equation (3) to Horn derivability where positive costs enforce minimal-support optimality. □

Proposition 3 (Necessity under a failure context). A robust certificate C is inclusion-minimal if and only if, for each $e \in C$ , some withdrawal $F _ { e }$ with $| F _ { e } | ~ \le ~ \dot { k }$ makes $C \backslash \{ e \}$ fail. Every such witness satisfies $\gamma ( e ) \cap F _ { e } = \emptyset$

Proof. Minimality requires that removing any $e \in C$ admits a defeating withdrawal $F _ { e } ( | F _ { e } | \leq k ) . { \bar { \mathrm { I f } } } \gamma ( { \bar { e } } ) \cap F _ { e }$ ̸= ∅, e is already invalidated by $F _ { e }$ , making ${ \cal { X } } ( C { \bf { \dot { \{ \bf { \Lambda } } } }  \{ e \} , F _ { e } ) =$ $X ( C , F _ { e } )$ , which contradicts that $F _ { e }$ refutes $C \backslash \{ e \}$ while C survives. □

These witnesses explain retained nominal redundancy: for the three-check certificate, $e _ { A }$ becomes necessary when $s _ { B }$ is withdrawn.

## Exact Certificate Synthesis

## A counterinterpretation oracle

Support and withdrawal enumeration can both be exponential. We instead seek a defeating withdrawal and a rule-closed interpretation $Z \subseteq V { : }$ every rule whose body lies in Z has its head there.

Lemma 1 (Closed-counterinterpretation test). A candidate C is not robust if and only if there exist $F \subseteq D$ and a rule-closed $Z \subseteq V$ such that

$$
| F | \leq k , \qquad X ( C , F ) \subseteq Z , \qquad q \not \in Z .\tag{7}
$$

Proof. If F defeats $C , Z = \mathrm { c l } _ { P } ( X ( C , F ) )$ is rule-closed, contains $X ( C , F )$ , and excludes $q .$ Conversely, if such Z exists, least-fixed-point induction yields c $_ { P } ( X ( { \dot { C } } , F ) ) \subseteq Z ;$ since $q \not \in Z , q \not \in \dot { \mathrm { c l } } _ { P } ( X ( C , F ) )$ , proving nonrobustness. □

Introduce Boolean variables $d _ { s }$ for failed domains and $z _ { a }$ for membership in Z. With C fixed, the oracle asks whether the following constraints are satisfiable:

$$
\begin{array} { r l } & { \displaystyle \sum _ { s \in D } d _ { s } \leq k , \qquad z _ { b } = 1 ( b \in B ) , \qquad z _ { q } = 0 , } \\ & { \displaystyle ( \bigwedge _ { a \in \mathrm { b o d y } ( r ) } z _ { a } )  z _ { \mathrm { h e a d } ( r ) } \quad ( r \in P ) , } \\ & { \displaystyle ( \bigwedge _ { s \in \gamma ( e ) } \lnot d _ { s } )  z _ { \alpha ( e ) } \quad ( e \in C ) . } \end{array}\tag{8}
$$

Empty conjunctions are true, so protected records force their atoms. Each selected record contributes $z _ { \alpha ( e ) } \vee$ $\vee _ { s \in \gamma ( e ) } d _ { s } .$ ; each rule contributes a Horn clause. A sequential counter can encode the domain bound (Sinz 2005). Satisfiability returns a counterexample; unsatisfiability establishes robustness.

Any closed superset sufices for nonderivability, avoiding inference-round layers. For unseeded rules u $ \ v$ and $v  u$ , the empty interpretation refutes u. One-way implications would allow unsupported cycles if used to certify positive derivability; they are exact here because a closed set excluding the target certifies absence.

## Globally valid evidence cuts

For an oracle witness $( F , Z )$ , consider all available records that survive $F$ and assert an atom outside $Z \colon$

$$
H ( F , Z ) = \{ e \in \mathcal { E } : \gamma ( e ) \cap F = \emptyset , \alpha ( e ) \notin Z \} .\tag{9}
$$

Lemma 2 (Selection cut). Every robust certificate intersects $H ( F , Z )$ , whereas the refuted candidate C does not.

Proof. Any certificate $C ^ { \prime }$ avoiding $H ( F , Z )$ satisfies $X ( C ^ { \prime } , \overset { \cdot } { F } ) \subseteq \overset { \cdot } { Z } , \ s \textnormal { o } q \not \in \mathrm { c l } _ { P } ( X ( C ^ { \prime } , F ) )$ by Lemma 1, meaning $C ^ { \prime }$ is not robust. Candidate C avoids ${ \dot { \boldsymbol { H } } } ( { \boldsymbol { F } } , { \boldsymbol { Z } } )$ by definition because all its surviving assertions were constrained to lie in Z by the oracle. □

![](images/051c36e99346ccde9ac4f338abe26c9b967f3dc433ff0e91e854840cd6cf4671.jpg)  
Figure 1: Running-example support footprints, their shared domains, and a surviving justification for each admissible singledomain withdrawal.

A cut rejects every selection whose surviving assertions fit inside Z. Failed records cannot repair the witness and are excluded, even when they assert the same atom as an available record.

Example 3 (A cut requiring alternative evidence). For $C = \{ e _ { A } ^ { - } , e _ { B } \}$ in Example $2 , { \overset { \smile } { F } } = \{ s _ { A } \}$ admits $Z = \{ b \}$ Then ${ \cal H } ( F , Z ) = \{ e _ { C } , e _ { D } \} :$ a robust certificate needs the remaining check or the independent attestation. A copy of a depending on $s _ { A }$ is excluded because it also fails.

Let H contain accumulated cuts, and introduce selection variables $x _ { e } .$ . The master solves

$$
\begin{array} { r l } { \operatorname* { m i n } \displaystyle \sum _ { e \in \mathcal { E } } c ( e ) x _ { e } , \quad } & { \displaystyle \sum _ { e \in H } x _ { e } \ge 1 \left( H \in \mathcal { H } \right) , } \\ { x _ { e } \in \{ 0 , 1 \} . } \end{array}\tag{10}
$$

The master is weighted hitting set, implemented by integer programming or weighted partial MaxSAT with hard positive clauses and soft clauses $\neg x _ { e } .$ . Exact optimization supplies cost optimality; the oracle checks the full guarantee.

An empty cut is retained and makes the master infeasible: even full evidence cannot repair that withdrawal. If trusted facts already derive q, the empty candidate is accepted immediately.

Theorem 2 (Exact synthesis). Algorithm 1 terminates after at most $2 ^ { | \varepsilon | }$ <sup>|</sup> refutations. It returns a minimum-cost robust certificate whenever one exists, and returns INFEASIBLE otherwise.

Proof. By Lemma 2, each cut is necessary for robustness and refutes the generating candidate C, preventing repeated candidates and guaranteeing termination in $\leq 2 ^ { | \varepsilon | }$ iterations. When the oracle is unsatisfiable, Lemma 1 certifies that C is robust; optimality follows because C minimizes cost over a relaxation of the feasible space. Master infeasibility implies no certificate satisfies all necessary cuts, proving I infeasible. □

Algorithm 1 Minimum-cost robust certification   
Input: Certification instance I   
Output: A minimum-cost k-robust certificate, or INFEASI-  
BLE   
1: H ← empty family   
2: loop   
3: C ← an exact minimum-cost hitting set of H   
4: if the master is infeasible then   
5: return INFEASIBLE   
6: end if   
7: witness ← solve the closed-counterinterpretation or  
acle for C   
8: if the oracle is unsatisfiable then   
9: return C   
10: end if   
11: (F, Z) ← witness   
12: Z ← optionally strengthen the closed counterinter  
pretation   
13: H ← all records surviving F whose asserted atoms   
are outside Z   
14: add H to H   
15: end loop

The master optimum provides a lower bound; any verified robust certificate (including E) supplies an upper bound, bounding any optimality gap upon solver interruption.

## Strengthening, preprocessing, and reuse

A larger closed counterinterpretation yields a stronger, more compact cut. Given witness $( F , Z )$ , we greedily test each surviving record e with $\alpha ( e ) \not \in Z ,$ , updating $Z \dot {  } \mathrm { c l } _ { P } ( Z \cup$ $\{ \alpha ( e ) \} )$ whenever $q \not \in \mathrm { c l } _ { P } ( Z \cup \{ \alpha ( e ) \} )$ . This terminates i $1 \leq | { \mathcal { E } } |$ closure passes and yields a subset-irreducible cut (Junker 2004; Lifiton and Sakallah 2007). Non-minimal cuts are removed via subsumption.

Proposition 4 (Record dominance). If distinct records $e , f$ satisfy $\alpha ( e ) = \alpha ( f ) , \gamma ( e ) \subseteq \gamma ( f )$ , and $c ( e ) \leq c ( f )$ removing f preserves the optimum (retaining one representative per tied equivalence class).

Proof. Replacing f with e in any certificate preserves surviving facts under all withdrawals without increasing cost. □

Backward rule slicing prunes atoms without dependency paths to q. Accumulated cuts remain valid under cost updates and increases in k.

## Relation to general propositional abduction

For an exact general-purpose baseline, compile certification into propositional abduction. Let $T _ { k }$ contain Equation $( 8 ) ^ { \circ } \mathbf { s }$ domain bound, trusted facts, and rules, omit $z _ { q } = 0$ , and guard all record clauses by selection:

$$
{ \Biggl ( } x _ { e } \wedge \bigwedge _ { s \in \gamma ( e ) } \neg d _ { s } { \Biggr ) } \to z _ { \alpha ( e ) } \qquad ( e \in \mathcal { E } ) .\tag{11}
$$

Selection variables are distinct from atom and domain variables. Positive literals $x _ { e }$ are the abductive hypotheses, with costs $c ( e )$ ; the manifestation is $z _ { q }$

Corollary 1 (Abduction compilation). For every $C \subseteq { \mathcal { E } }$

$$
C { \mathrm { ~ i s ~ } } k { \mathrm { - r o b u s t } } \quad \Longleftrightarrow \quad T _ { k } \land \bigwedge _ { e \in C } x _ { e } \vdash z _ { q } .\tag{12}
$$

Moreover, the selected hypotheses are always consistent with $T _ { k }$

Proof. Every model corresponds to an admissible withdrawal F and a closed interpretation containing $X ( C , F )$ robustness forces $z _ { q }$ in all such models. Conversely, a defeating withdrawal and its least closure yield a countermodel falsifying $z _ { q } .$ . Consistency holds by setting all atom variables true and domain variables false. □

A minimum-cost abduction solver can solve this compilation (Ignatiev, Morgado, and Marques-Silva 2016). Sourcespecific witnesses, dominance, and closure strengthening provide the algorithmic refinements tested in our evaluation.

## Computational complexity

Let $N = | { \mathcal { E } } | , m = | D |$ , and L be the explicit input size, including rules, facts, records, domains, dependency lists, and binary cost encodings. Verification tests a given C; budgeted synthesis asks whether some robust C costs at most an input bound b.

Theorem 3 (Verification complexity). Verification is coNP-complete when k is part of the input. For every fixed k, it is polynomial-time decidable. Hardness holds for acyclic programs with rule bodies of size at most two, distinct asserted atoms, and one distinct singleton failure domain per record.

Proof sketch. Membership in coNP follows because a defeating withdrawal $F \left( \left| F \right| \leq k \right)$ serves as a polynomial certificate verifiable via Horn closure. For hardness, we reduce from Hitting Set. Given universe U, family A, and bound k, create an asserted atom and a uniquely failing record for each element of $U _ { : }$ , and add rules deriving q from each $A \in { \mathcal { A } }$

Selecting all records forces any defeating withdrawal of size $\leq k$ to intersect every A, matching hitting sets of size $\leq k .$ Wide rule bodies are factored into binary conjunctions, preserving acyclicity and constant body width. For fixed k, testing all $\textstyle \sum _ { i = 0 } ^ { \bar { k } } { \binom { m } { i } }$ withdrawals takes polynomial time. Full induction and reduction details are deferred to the extended version. □

Theorem 4 (Synthesis complexity). Budgeted synthesis is NP-complete for every fixed $k \geq 0$ , even under the structural restrictions of Theorem 3. With unrestricted k, it belongs to $\Sigma _ { ? } ^ { P }$

Proof sketch. Membership in NP for fixed k follows by guessing $C$ and verifying robustness in polynomial time by Theorem 3; unrestricted k belongs to $\Sigma _ { ? } ^ { \dot { P } }$ via an existentialuniversal quantifier alternation. For NP-hardness, we reduce from Set Cover with universe $U$ , sets $A _ { 1 } , \dotsc , A _ { t }$ , and bound $b _ { 0 }$ . We create set records of cost $k + 1$ asserting atoms that derive elements ofU to derive $q ,$ alongside k unit-cost backup records asserting distinct atoms that independently derive q. Setting the cost budget to $( k + 1 ) b _ { 0 } + k$ forces selection of all k backups to survive backup-targeted withdrawals, leaving a budget that permits at most $b _ { 0 }$ set records to cover $\check { U }$ Binary conjunction trees maintain acyclicity and bounded body width. Full induction and reduction details are deferred to the extended version. □

Thus, verification is FPT in failure domains $( O ( 2 ^ { m } L )$ time) and polynomial for fixed k $: ( O ( ( m + 1 ) ^ { k } L ) )$ , while synthesis is NP-hard even at $k = 0 .$ . The oracle uses $O ( L + m k )$ clauses via sequential cardinality encoding, the master holds $N$ variables, and strengthening requires $\dot { \cal O } ( N L )$ time per refutation.

Proposition 5 (Independent-edge reachability). Suppose evidence records represent directed edges, each edge has a unique singleton failure domain, and the rules compute reachability from a fixed vertex s to a distinct vertex t. Use vertex atoms $r _ { v } ,$ edge atoms $g _ { u v } ,$ the sole trusted fact $r _ { s } .$ , and precisely the rules $r _ { v } \gets r _ { u } , g _ { u v }$ . Each edge has one record asserting $g _ { u v } ;$ the target is $r _ { t }$ . Minimum-cost k-robust certification is solvable by a minimum-cost flow of value $k + 1$ with unit edge capacities. Infeasible flow implies infeasible certification.

Proof. By the max-flow/min-cut theorem, a subgraph survives all $\leq \dot { k }$ edge deletions if it contains $k + 1$ edge-disjoint $^ { s - t }$ paths. A minimum-cost integral unit flow of value $k + 1$ identifies an optimal cycle-free edge subset, which directly defines an optimal k-robust certificate. □

When a source labels multiple edges or rules require joint premises, path formulations miss correlated cuts, but Algorithm 1 remains exact.

## Experiments

We evaluate the empirical performance of minimum-cost robust certification, focusing on four primary objectives: (1) quantifying the scalability of counterinterpretation-guided separation relative to explicit scenario expansion, provenance compilation, and general weighted abduction, (2) assessing the empirical incompleteness and false-negative rate of domain-disjoint proof packing (Proposition 1), (3) calibrat-

<table><tr><td>(a) Runtime vs. Records |ε|</td><td>(k = 2)</td></tr><tr><td>|ε|Alg. 1</td><td>Scenario</td></tr><tr><td>64 1.2 s</td><td>8.4 s</td></tr><tr><td>128 3.8 s</td><td>42.1 s</td></tr><tr><td>256 11.5 s</td><td>218.0 s</td></tr><tr><td></td><td>512 28.4s &gt; 600 s (TO)</td></tr></table>

$$
( O ( ( m + 1 ) ^ { k } L ) )
$$

(b) Optimum Coverage vs. Budget k
<table><tr><td>k</td><td>Alg. 1</td><td>Proof Pack.</td></tr><tr><td>0</td><td>100.0%</td><td>100.0%</td></tr><tr><td>1</td><td>95.2%</td><td>42.5%</td></tr><tr><td>2</td><td>91.8%</td><td>28.0%</td></tr><tr><td>3</td><td>89.4%</td><td>24.0%</td></tr></table>

Proof packing coverage drops precipitously due to overlapping dependencies (Proposition 1); Alg. 1 maintains ≈ 90% coverage.

(c) Master Calls & Strengthening
<table><tr><td>Variant</td><td>Master Calls</td><td>Cut Len.</td></tr><tr><td>Full procedure</td><td> ${ \bf 1 4 . 2 \pm 3 . 1 }$ </td><td> ${ \bf 1 2 . 3 \pm 2 . 1 }$ </td></tr><tr><td>Raw cuts</td><td> $2 2 . 8 \pm 5 . 2$ </td><td> $2 8 . 6 \pm 4 . 5$ </td></tr><tr><td>Candidate excl.</td><td> $8 6 . 5 \pm 1 8 . 4$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr></table>

Cut strengthening cuts master solver calls by 37.7% and yields compact, general cuts with minimal Horn closure overhead.  
Figure 2: Empirical scaling profiles across ground record sizes |E| and failure budgets $k \in \{ 0 , 1 , 2 , 3 \}$ : (a) end-to-end runtime scaling demonstrating polynomial growth for Algorithm 1 against exponential explosion in explicit scenario expansion; (b) optimum coverage highlighting the high false-negative rate of domain-disjoint proof packing on correlated dependencies; and (c) search reduction enabled by closed counterinterpretation strengthening.

ing certified optima against polynomial-time flow references (Proposition 5), and (4) isolating the computational contributions of closed counterinterpretation strengthening, dominance pruning, and relevance slicing. Across all benchmarks, exact methods are held to strict optimality verification; unverified candidate evidence is rejected as an invalid solution to Equation (4).

## Benchmarks and configuration

Synthetic benchmarks comprise controlled Horn systems and reachability networks. Horn systems vary ground records |E| ∈ {64, 128, 256, 512}, failure domains $\left| D \right| \in$ {8, 16, 32, 64}, body widths $w \in \{ 1 , 2 , 4 \}$ , domain overlap sizes $d \in \{ 1 , 2 , \bar { 4 } \}$ , and budgets $\mathrm { ~ \bar { \boldsymbol { k } } ~ } \in \mathrm { ~ \{ \bar { 0 } , 1 , 2 , 3 \} ~ }$ across acyclic, recursive, and planted overlapping-support hypergraphs (20 instances/setting) under uniform, proportional, and log-normal costs. Reachability graphs evaluate route bottlenecks; independent-edge topologies provide min-cost flow ground truth (Proposition 5), while paired variants inject correlated infrastructure failures.

Knowledge-base benchmarks use positive Horn translations of Lehigh University Benchmark (Guo, Pan, and Heflin 2005) data (LUBM-1 and LUBM-5), mapping domains to institutions and data pipelines. Targets are sampled across shallow and recursive derivations with backward rule slicing; replicated assertions receive distinct record identities with shared or separate dependencies.

## Baselines and output verification

We compare Algorithm 1 against six baseline approaches representing diverse reasoning paradigms:

Nominal minimum support solves the zero-failure problem (k = 0) via Horn abduction to check whether nominal proofs withstand withdrawals. Domain-disjoint proof packing selects a minimum-cost union of $k + 1$ supports with pairwise disjoint footprints. Explicit scenario synthesis unrolls |V| closure rounds across all $\textstyle \sum _ { i = 0 } ^ { k } { \binom { m } { i } }$ failure scenarios over shared selection variables. Provenance compilation builds a Boolean provenance circuit (Deutch et al.

2014) optimized via binary decision diagrams (Bryant 1986). Compiled weighted abduction compiles Equation (12) into propositional abduction via AbHS (Ignatiev, Morgado, and Marques-Silva 2016). Greedy repair iteratively adds surviving supports queried from the oracle until robustness is achieved. On independent reachability networks, minimumcost flow supplies polynomial ground truth.

All certificates were independently validated via external forward chaining and SAT verification oracles. Incumbent bounds, certified optima, and proven infeasibilities are tracked separately; unverified outputs are never credited as solutions.

## Implementation, metrics, and ablations

Our framework is implemented in Python 3.11 using PySAT (Ignatiev, Morgado, and Marques-Silva 2018), combining RC2 for the master with Glucose 4.1 and Sinz’s cardinality encoding (Sinz 2005) for the oracle. All experiments were conducted single-threaded on AMD EPYC 7742 processors (2.25 GHz) running Ubuntu 22.04, with a 600 s timeout and 8 GiB memory limit. Primary metrics include feasible cost, certified optimum coverage, proven infeasibility, runtime, and memory. The price of robustness is measured by

$$
\mathrm { p r e m i u m } ( k ) = \frac { \cos ( C _ { k } ^ { * } ) - \cos ( C _ { 0 } ^ { * } ) } { \cos ( C _ { 0 } ^ { * } ) } ,\tag{13}
$$

where $C _ { k } ^ { * }$ and $C _ { 0 } ^ { * }$ denote certified robust and nominal optima, respectively. Targets derivable solely from trusted facts require zero nominal cost and are reported separately.

Scalability and runtime dynamics. As shown in Table 1 and Figure 2(a), Algorithm 1 scales gracefully across both synthetic Horn systems and knowledge bases, achieving up to 92% optimum coverage. In contrast, explicit scenario synthesis sufers severe combinatorial explosion $( O ( ( m + 1 ) ^ { k } L ) )$ , rapidly timing out and exhausting memory for $k \geq 2 . \mathbf { C o m } \mathrm { { - } }$ piled abduction and provenance compilation avoid full unrolling but stall on large or recursive instances due to circuit blowup and lack of source-aware cuts. On all mutually solved instances, all exact solvers yield identical optimal costs, confirming empirical correctness.

<table><tr><td>Method</td><td>Feasible Cost</td><td>Opt. Coverage</td><td>Proven Infeas.</td><td>Runtime (s)</td><td>Peak Mem. (MB)</td></tr><tr><td colspan="6">Controlled Horn Systems  $( | \mathcal { E } | \leq 5 1 2 , k \in \{ 1 , 2 , 3 \} )$ </td></tr><tr><td>Nominal (k = 0), checked at k</td><td> $8 . 4 \pm 1 . 6 ^ { \dagger }$ </td><td>4.2%</td><td></td><td> $0 . 4 \pm 0 . 1$ </td><td> $2 4 \pm 5$ </td></tr><tr><td>Domain-disjoint packing</td><td> $2 6 . 8 \pm 4 . 3$ </td><td>31.5%</td><td></td><td> $1 8 . 2 \pm 5 . 4$ </td><td> $6 8 \pm 1 4$ </td></tr><tr><td>Explicit scenario synthesis</td><td> $2 3 . 2 \pm 3 . 7$ </td><td>46.8%</td><td> $1 0 0 . 0 \%$ </td><td> $1 8 4 . 6 \pm 4 2 . 1$ </td><td> $2 , 8 4 0 \pm 6 2 0$ </td></tr><tr><td>Provenance compilation</td><td> $2 2 . 6 \pm 3 . 5$ </td><td>39.2%</td><td>85.0%</td><td> $2 1 0 . 4 \pm 5 1 . 3$ </td><td> $1 , 9 2 0 \pm 4 5 0$ </td></tr><tr><td>Compiled weighted abduction</td><td> $2 1 . 9 \pm 3 . 4$ </td><td>61.4%</td><td>100.0%</td><td> $9 5 . 7 \pm 2 2 . 8$ </td><td> $4 1 0 \pm 8 5$ </td></tr><tr><td>Greedy repair</td><td> $2 7 . 9 \pm 4 . 8$ </td><td></td><td></td><td> $2 . 8 \pm 0 . 7$ </td><td> $4 2 \pm 9$ </td></tr><tr><td>Algorithm 1 (Ours)</td><td> ${ \bf 2 1 . 5 \pm 3 . 2 }$ </td><td>92.1%</td><td>100.0%</td><td> ${ \bf 1 9 . 4 \pm 4 . 2 }$ </td><td> ${ \bf 1 8 5 \pm 3 6 }$ </td></tr><tr><td colspan="6">Knowledge-Base &amp; Network Benchmarks (LUBM-1/5, Reachability)</td></tr><tr><td>Domain-disjoint packing</td><td> $4 8 . 2 \pm 7 . 1$ </td><td>28.0%</td><td></td><td> $3 4 . 5 \pm 8 . 2$ </td><td> $1 1 2 \pm 2 5$ </td></tr><tr><td>Explicit scenario synthesis</td><td> $3 8 . 8 \pm 6 . 2$ </td><td>38.5%</td><td>100.0%</td><td> $2 4 5 . 2 \pm 5 8 . 7$ </td><td> $4 , 1 5 0 \pm 8 9 0$ </td></tr><tr><td>Compiled weighted abduction</td><td> $3 7 . 2 \pm 5 . 7$ </td><td>54.0%</td><td>100.0%</td><td> $1 4 2 . 8 \pm 3 1 . 5$ </td><td> $6 8 0 \pm 1 3 0$ </td></tr><tr><td>Greedy repair</td><td> $4 5 . 6 \pm 6 . 9$ </td><td></td><td></td><td> $4 . 1 \pm 1 . 1$ </td><td> $7 8 \pm 1 6$ </td></tr><tr><td>Algorithm 1 (Ours)</td><td> ${ \bf 3 6 . 1 \pm 5 . 4 }$ </td><td>88.5%</td><td>100.0%</td><td> ${ \bf 3 1 . 2 \pm 6 . 8 }$ </td><td> ${ \bf 2 6 5 \pm 4 8 }$ </td></tr></table>

Table 1: Main performance comparison across benchmark families. Feasible cost reports mean ± std over instances where methods found verified certificates (<sup>†</sup>nominal support only satisfies $k = 0$ and fails robustness check on 95.8% of instances; unresolved instances for exact baselines report best verified incumbents). On the common subset of instances solved to certified optimality by all exact methods, their optimal costs are strictly identical (16.4±2.5 for Horn systems, 27.9±4.1 for KB/networks). Optimum coverage and proven infeasibility percentages indicate exact resolution within the 600 s limit.
<table><tr><td>Variant</td><td>Master calls</td><td>Oracle time (s)</td><td>Cut length</td><td>Strengthening time (s)</td><td>Total time (s)</td></tr><tr><td>Full procedure</td><td> ${ \bf 1 4 . 2 \pm 3 . 1 }$ </td><td> $6 . 4 \pm 1 . 2$ </td><td> ${ \bf 1 2 . 3 \pm 2 . 1 }$ </td><td> $1 . 8 \pm 0 . 4$ </td><td> ${ \bf 1 9 . 4 \pm 4 . 2 }$ </td></tr><tr><td>Raw counterinterpretations</td><td> $2 2 . 8 \pm 5 . 2$ </td><td> $6 . 1 \pm 1 . 1$ </td><td> $2 8 . 6 \pm 4 . 5$ </td><td></td><td> $2 7 . 4 \pm 6 . 8$ </td></tr><tr><td>Candidate exclusion only</td><td> $8 6 . 5 \pm 1 8 . 4$ </td><td> $2 4 . 8 \pm 5 . 6$ </td><td> $1 . 0 \pm 0 . 0$ </td><td></td><td> $9 4 . 2 \pm 2 1 . 0$ </td></tr><tr><td>Without dominance</td><td> $1 9 . 1 \pm 4 . 0$ </td><td> $7 . 2 \pm 1 . 5$ </td><td> $1 5 . 8 \pm 2 . 6$ </td><td> $2 . 1 \pm 0 . 5$ </td><td> $2 4 . 7 \pm 5 . 3$ </td></tr><tr><td>Without relevance slicing</td><td> $1 4 . 6 \pm 3 . 2$ </td><td> $1 4 . 8 \pm 3 . 1$ </td><td> $1 2 . 5 \pm 2 . 2$ </td><td> $3 . 9 \pm 0 . 8$ </td><td> $2 9 . 1 \pm 6 . 2$ </td></tr></table>

Table 2: Algorithmic ablation on controlled Horn benchmarks $( | \mathcal { E } | \leq 5 1 2 , k = 2 )$ , showing the efect of cut strengthening, dominance pruning, and relevance slicing on master solver invocations and cumulative runtimes.

Incompleteness of proof packing. Domain-disjoint proof packing exhibits a severe false-negative rate, resolving under a third of benchmark instances due to overlapping support footprints (generalizing Example 2). Even when disjoint supports exist, enforcing strict disjointness over-restricts search and inflates costs, confirming Proposition 1: disjoint packing cannot substitute for transversal reasoning.

Exact ground-truth calibration. On reachability instances with unique edge domains, Proposition 5 equates robust certification to minimum-cost unit-capacity flow of value k + 1. Across all 80 topologies, Algorithm 1 achieves 100% agreement with the polynomial flow optimum. Under correlated failures, min-cost flow becomes unsound, whereas Algorithm 1 continues to synthesize verified optima.

Robustness premium and greedy repair. Evaluating Equation (13) demonstrates that securing conclusions against correlated withdrawals requires substantial cost premiums as k increases; nominal solutions (k = 0) fail checks in > 95% of cases. Greedy repair accumulates redundant records, producing uncompetitive certificates far costlier than certified optima.

Ablation analysis (Table 2). Table 2 isolates each component’s contribution on controlled Horn instances $( | \mathcal { E } | \leq$ 512, k = 2). Cut strengthening cuts master calls by 37% and yields cuts under half the length of raw witnesses, reducing total runtime by 29% with negligible closure overhead. Candidate exclusion thrashes severely. Dominance pruning eliminates redundant duplicate records upfront, while relevance slicing halves oracle verification time.

## Sensitivity and error analysis

Denser source overlap requires more master-oracle iterations but maintains stable memory. Wider rule bodies scale linearly in oracle check time, while log-normal costs accelerate master search through aggressive pruning.

## Conclusion

We presented an exact framework for synthesizing minimumcost evidence certificates for Horn reasoning under correlated source failures. By separating record selection from asserted facts, our footprint characterization explains why proof packing is incomplete, while closed counterinterpretations yield globally valid cuts that bypass exponential scenario unrolling. We established finite termination, cost optimality, tight complexity bounds, and a tractable flow restriction. Experiments confirm that our method resolves up to 92% of benchmarks, outperforming scenario expansion and general abduction. Future work includes multiquery cut reuse, structured withdrawal constraints, and non-additive costs.

## References

Alvaro, P.; Rosen, J.; and Hellerstein, J. M. 2015. Lineagedriven Fault Injection. In Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data, 331–346. ACM.

Bourgaux, C.; Bourhis, P.; Peterfreund, L.; and Thomazo, M. 2022. Revisiting Semiring Provenance for Datalog. In Proceedings of the Nineteenth International Conference on Principles ofKnowledge Representation and Reasoning, 91– 101. California: International Joint Conferences on Artificial Intelligence Organization.

Bryant. 1986. Graph-Based Algorithms for Boolean Function Manipulation. IEEE Transactions on Computers, C-35(8): 677–691.

Buneman, P.; Khanna, S.; and Wang-Chiew, T. 2001. Why and Where: A Characterization of Data Provenance. In Lecture Notes in Computer Science, Lecture Notes in Computer Science, 316–330. Berlin, Heidelberg: Springer Berlin Heidelberg. ISBN 978-3-540-41456-8.

Calautti, M.; Livshits, E.; Pieris, A.; and Schneider, M. 2024a. The Complexity of Why-Provenance for Datalog Queries. Proceedings of the ACM on Management of Data, 2(2): 1–16.

Calautti, M.; Livshits, E.; Pieris, A.; and Schneider, M. 2024b. Computing the Why-Provenance for Datalog Queries via SAT Solvers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 10459–10466. Association for the Advancement of Artificial Intelligence (AAAI).

Darwiche, A.; and Hirth, A. 2020. On The Reasons Behind Decisions. ArXiv:2002.09284 [cs.AI].

Darwiche, A.; and Marquis, P. 2002. A Knowledge Compilation Map. Journal ofArtificial Intelligence Research, 17: 229–264.

Davies, J.; and Bacchus, F. 2011. Solving MAXSAT by Solving a Sequence of Simpler SAT Instances. In Lecture Notes in Computer Science, Lecture Notes in Computer Science, 225–239. Berlin, Heidelberg: Springer Berlin Heidelberg. ISBN 978-3-642-23785-0.

de Kleer, J. 1986. An assumption-based TMS. Artificial Intelligence, 28(2): 127–162.

Deutch, D.; Milo, T.; Roy, S.; and Tannen, V. 2014. Circuits for Datalog Provenance. OpenProceedings.org.

Dowling, W. F.; and Gallier, J. H. 1984. Linear-time algorithms for testing the satisfiability of propositional horn formulae. The Journal of Logic Programming, 1(3): 267– 284.

Freire, C.; Gatterbauer, W.; Immerman, N.; and Meliou, A. 2015. The complexity of resilience and responsibility for self-join-free conjunctive queries. Proceedings of the VLDB Endowment, 9(3): 180–191.

Green, T. J.; Karvounarakis, G.; and Tannen, V. 2007. Provenance semirings. In Proceedings of the twenty-sixth ACM SIGMOD-SIGACT-SIGART symposium on Principles ofdatabase systems, 31–40. ACM.

Guo, Y.; Pan, Z.; and Heflin, J. 2005. LUBM: A benchmark for OWL knowledge base systems. Journal of Web Semantics, 3(2-3): 158–182.

Ignatiev, A.; Morgado, A.; and Marques-Silva, J. 2016. Propositional Abduction with Implicit Hitting Sets. ArXiv:1604.08229 [cs.AI].

Ignatiev, A.; Morgado, A.; and Marques-Silva, J. 2018. PySAT: A Python Toolkit for Prototyping with SAT Oracles. In Beyersdorf, O.; and Wintersteiger, C. M., eds., Theory and Applications of Satisfiability Testing – SAT 2018, 428–437. Cham: Springer International Publishing. ISBN 978-3-319-94144-8.

Ignatiev, A.; Narodytska, N.; and Marques-Silva, J. 2019. Abduction-Based Explanations for Machine Learning Models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, 1511–1519. Association for the Advancement of Artificial Intelligence (AAAI).

Junker, U. 2004. QUICKXPLAIN: preferred explanations and relaxations for over-constrained problems. In Proceedings of the 19th national conference on Artifical intelligence, AAAI’04, 167–172. San Jose, California: AAAI Press. ISBN 978-0-262-51183-4.

Kakas, A. C.; Kowalski, R. A.; and Toni, F. 1992. Abductive Logic Programming. Journal of Logic and Computation, 2(6): 719–770.

La Malfa, E.; Michelmore, R.; Zbrzezny, A. M.; Paoletti, N.; and Kwiatkowska, M. 2021. On Guaranteed Optimal Robust Explanations for NLP Models. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence, 2658–2665. California: International Joint Conferences on Artificial Intelligence Organization.

Lifiton, M. H.; and Sakallah, K. A. 2007. Algorithms for Computing Minimal Unsatisfiable Subsets of Constraints. Journal ofAutomated Reasoning, 40(1): 1–33.

Meliou, A.; Gatterbauer, W.; Moore, K. F.; and Suciu, D. 2010. The complexity of causality and responsibility for query answers and non-answers. Proceedings of the VLDB Endowment, 4(1): 34–45.

Motik, B.; Nenov, Y.; Piro, R.; Horrocks, I.; and Olteanu, D. 2014. Parallel Materialisation of Datalog Programs in Centralised, Main-Memory RDF Systems. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 28. Association for the Advancement of Artificial Intelligence (AAAI).

Reiter, R. 1987. A theory of diagnosis from first principles. Artificial Intelligence, 32(1): 57–95.

Saikko, P.; Wallner, J.; and Järvisalo, M. 2016. Implicit Hitting Set Algorithms for Reasoning Beyond NP.

Sinz, C. 2005. Towards an Optimal CNF Encoding of Boolean Cardinality Constraints. In Lecture Notes in Computer Science, Lecture Notes in Computer Science, 827–831. Berlin, Heidelberg: Springer Berlin Heidelberg. ISBN 978- 3-540-29238-8.

Thapa, R. B.; and Staab, S. 2026. Causality and Minimal Supports in Recursive Datalog. Version 2, arXiv:2607.16443.

Van Emden, M. H.; and Kowalski, R. A. 1976. The Semantics of Predicate Logic as a Programming Language. Journal of the ACM, 23(4): 733–742.

Zhang, P.; and Tang, L. 2020. Minimum Label s-t Cut has large integrality gaps. Information and Computation, 275: 104543.