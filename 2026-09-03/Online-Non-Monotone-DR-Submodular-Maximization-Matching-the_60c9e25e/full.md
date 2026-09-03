# Online Non-Monotone DR-Submodular Maximization Matching the Ofline 0.401 Factor

Vaneet Aggarwal Purdue University

Yiyang Lu Purdue University

## Abstract

We study online maximization of nonnegative, non-monotone DR-submodular functions over compact convex down-closed subsets of the d-dimensional unit cube. The best known constructive ofline approximation factor is 0.401 under the corresponding meta-solvability assumptions, whereas comparable adversarial online guarantees had remained at $1 / e \mathrm { . }$ . We show that this factor is also achievable online. In the post-decision full-information value-oracle model, our algorithm attains factor 0.401 with sublinear approximate regret when oracle feedback is conditionally unbiased and bounded.

The online algorithm does not run the ofline construction on a changing objective. Instead, it replaces the ofline objective-dependent box step by a weighted online learner that controls the required residual terms cumulatively. An exact asymmetric balance theorem preserves the ofline coeficients despite adversarial variation. The direct implementation has $\overset { \mathcal { \mathrm { ~ \textstyle ~ - ~ } } } { O (} T ^ { 3 / 4 } )$ regret and uses $O ( d T ^ { 1 / 4 } )$ oracle calls per round. More generally, for every $\delta \in [ 0 , 1 / 4 ]$ , batching gives $O ( T ^ { \delta } )$ calls per round and $O ( \tilde { T } ^ { 4 / 5 - \delta / 5 } )$ regret, including a one-call $O ( \bar { T } ^ { 4 / 5 } )$ endpoint. Under a positive-anchor condition, randomized blocking retains factor 0.401 with $\dot { O } ( T ^ { 5 / 6 } )$ one-point bandit regret.

## 1 Introduction

DR-submodular functions are continuous objectives with diminishing marginal returns: increasing one coordinate can only decrease the marginal value of increasing another. They provide a natural model when the coordinates of a decision encode divisible allocations, activation probabilities, or continuous intensities. Non-monotonicity is essential in many such models, because allocating more of a resource can introduce congestion, redundancy, or interference and can therefore reduce the objective. Down-closed constraints capture the complementary feasibility principle that reducing an allocation preserves feasibility.

In the online problem, the learner repeatedly chooses a feasible point while the reward function changes over time. The action at round t must be committed before any feedback from $f _ { t }$ is available, and performance is compared with the best fixed feasible point in hindsight. This setting combines two genuinely diferent sources of loss. First, the objective is nonconcave, so even an ofline algorithm that knows a single function in advance generally ofers only an approximation guarantee. Second, the learner must estimate useful directions and structural choices from past observations, which produces an additive regret term. A satisfactory online result should keep these two efects separate: temporal variation should contribute sublinear additive loss without unnecessarily degrading the best available approximation factor.

The online/ofline gap. The central question of this paper is an approximation-factor question. For non-monotone DR-submodular maximization under down-closed convex constraints, the best known constructive ofline factor is 0.401: Buchbinder and Feldman obtain this guarantee using a delayed continuous-greedy construction together with an asymmetric Box-Maximization step [10]. Here and throughout, 0.401 refers to the best known constructive ofline guarantee under the corresponding meta-solvability assumptions of [10]; our comparison concerns the approximation coeficient, not identical algorithms or oracle assumptions. Comparable adversarial online results, however, had remained at $1 / e .$ . Starting with Thang and Srivastav, later post-decision value-oracle, one-shot, projection-free, and bandit methods improved regret and oracle complexity without improving the leading $1 / e$ coeficient [19, 24, 29, 30]. Lu et al. explicitly identified breaking this barrier as an open problem [30].

We ask whether the best known constructive ofline factor can survive adversarial temporal variation when the action must be selected before the current objective is available. Our answer is yes:

$$
\Bigl [ \mathrm { o n l i n e ~ a p p r o x i m a t i o n ~ f a c t o r ~ } = 0 . 4 0 1 \Bigr ] .\tag{1}
$$

The guarantee has sublinear approximate regret and therefore matches the best known constructive ofline approximation factor of 0.401. We do not claim that 0.401 is optimal. A related value-oracle hardness benchmark is 0.478: multilinear-extension maximization over down-closed polytopes is inapproximable beyond this value already for a partition matroid polytope, and the same threshold holds under a cardinality constraint [12, 13]. Thus 0.401 is the current constructive ofline benchmark, while the known hardness threshold is higher; our result shows that this constructive benchmark can also be attained online in the present value-oracle model.

Motivating settings. The ingredients of our model occur across several established application classes. Continuous submodular objectives have been used for influence and revenue maximization with continuous assignments, sensor-energy management, and facility location [2]; robust continuous budget allocation gives a closely related influence-maximization model [3]. Non-monotone DRsubmodular objectives arise in MAP inference for determinantal point processes and in provable mean-field inference for probabilistic submodular models [1, 4]. Online continuous submodular optimization was initiated by Chen, Hassani, and Karbasi [20]; time-varying DR-submodular welfare also appears in online resource allocation and shared-mobility rebalancing [21], while bandit DRsubmodular methods have been used to derive adversarial submodular-bandit guarantees [25]. These papers motivate the individual modeling ingredients; they do not all impose our exact combination of non-monotonicity, down-closed feasibility, and adversarial value feedback.

Table 1 isolates the approximation-factor comparison that motivates the paper. Table 2 then records only online value-feedback guarantees; it does not mix the ofline progression into the online regret comparison. We show the F-oracle rows reported by Lu et al. [30], rather than first-order gradient-feedback results.

Table 1: The ofline benchmark and the headline online result. The ofline row uses the metasolvability assumptions of [10]; the online row has sublinear approximate regret. The comparison is between approximation coeficients, not identical oracle models.
<table><tr><td>Role</td><td>Reference</td><td>Factor</td></tr><tr><td>Offline achievable</td><td>Buchbinder-Feldman [10]</td><td>0.401</td></tr><tr><td>Adversarial online achievable</td><td>This work</td><td>0.401</td></tr><tr><td>Related multilinear hardness</td><td>Oveis Gharan–Vondrák; Qi [12, 13]</td><td>0.478</td></tr></table>

Table 2: Online value-feedback guarantees for non-monotone DR-submodular maximization over down-closed convex sets. Problem-dependent constants and lower-order terms are suppressed.
<table><tr><td>Feedback</td><td>Reference</td><td>Approx. α</td><td>Calls/round</td><td>Static α-regret</td></tr><tr><td>PD value-oracle</td><td>Pedramfar et al. [29]</td><td> $1 / e$ </td><td>1</td><td> $O ( T ^ { 4 / 5 } )$ </td></tr><tr><td>PD value-oracle</td><td>Lu et al. [30]</td><td> $1 / e$ </td><td>1</td><td> $O ( T ^ { 3 / 4 } )$ </td></tr><tr><td>PD value-oracle</td><td>Ours (tradeoff)</td><td>0.401</td><td> $O ( T ^ { \delta } )$ </td><td> $O ( T ^ { 4 / 5 - \delta / 5 } )$ </td></tr><tr><td>PD value-oracle</td><td>Ours  $( \delta = 0 )$ </td><td>0.401</td><td>1</td><td> $O ( T ^ { 4 / 5 } )$ </td></tr><tr><td>PD value-oracle</td><td>Ours  $( \delta = 1 / 4 )$ </td><td>0.401</td><td> $O ( T ^ { 1 / 4 } )$ </td><td> $O ( T ^ { 3 / 4 } )$ </td></tr><tr><td>One-point bandit</td><td>Zhang et al. [24]</td><td> $1 / e$ </td><td>1</td><td> ${ \cal O } ( T ^ { 8 / 9 } )$ </td></tr><tr><td>One-point bandit</td><td>Pedramfar et al. [29]</td><td> $1 / e$ </td><td>1</td><td> $O ( T ^ { 5 / 6 } )$ </td></tr><tr><td>One-point bandit</td><td>Lu et al. [30]</td><td> $1 / e$ </td><td>1</td><td> $O ( T ^ { 4 / 5 } )$ </td></tr><tr><td>One-point bandit</td><td>Ours</td><td>0.401</td><td>1</td><td> $O ( T ^ { 5 / 6 } )$ </td></tr></table>

“PD value-oracle” denotes post-decision full-information value-oracle feedback: after committing to the played action, the learner may call the oracle at points diferent from the played action; in the bandit model the single observation is the noisy value of the played action, whose true value is the reward earned. More generally, for every $\delta \in [ 0 , 1 / 4 ]$ , our batched value-oracle result uses $O ( T ^ { \delta } )$ calls per physical round and has $O ( T ^ { 4 / 5 - \delta / 5 } )$ regret. The endpoint $\delta = 0$ uses one call, whereas $\delta = 1 / 4$ gives the best regret rate. Dimension and regularity factors are absorbed into the constants in this table, and the “Approx. $\alpha ^ { \mathfrak { N } }$ column is the coeficient of the benchmark in (9), an asymptotic quantity in the sense of Remark 3.2, not a finite-horizon approximation ratio. Our bandit row additionally assumes the positive-anchor condition stated in Section 5. Our rows allow a conditionally unbiased oracle with bounded error $\sigma \geq 0$ and assume an oblivious adversary.

The approximation factor is the leading-order quantity in approximate regret. If

$$
\mathrm { O P T } _ { T } : = \operatorname* { m a x } _ { o \in K } \sum _ { t = 1 } ^ { T } f _ { t } ( o ) = \Theta ( T ) ,
$$

then replacing $1 / e$ by 0.401 improves the guaranteed reward by approximately 0.0331 $\mathrm { O P T } _ { T } ,$ , a linear-in-T term. By contrast, every regret term in Table 2 is $o ( T )$ for fixed problem parameters. Thus the factor and regret rate are not interchangeable: the approximation factor determines the asymptotic fraction of the benchmark, while regret determines how quickly the algorithm approaches that fraction. In this precise sense, crossing the approximation barrier is the more fundamental advance, even though finite-horizon performance still depends on the regret exponent, dimension, oracle budget, and hidden constants.

Why online is harder than ofline. The standard $1 / e$ analysis is unusually compatible with online learning. Measured continuous greedy produces a local linear certificate, and an online linear-optimization algorithm can control its comparator-dependent linear term after summation over rounds. The stronger ofline construction uses information in a qualitatively diferent way. It works with one fixed objective that is available throughout the computation, and its local-search and Box-Maximization queries may adapt repeatedly to that same objective. Given $f$ and an outer point x, it solves an asymmetric box problem whose answer depends on the complete objective. That answer supplies two nonlinear values that cancel the negative terms in the delayed-trajectory certificate.

This corrective step is unavailable online. At round t, the outer state $x _ { t } ,$ the box decision, and the played action must all be selected before any feedback from $f _ { t }$ is available. Choosing the ofline box solution after observing $f _ { t }$ would violate the protocol, while using the solution from a previous round is useless against an adversarially changing sequence. Moreover, a pointwise linearization of the current played point cannot retain the two asymmetric value terms that create the improvement beyond $1 / e .$ . This is why the ofline proof does not admit a black-box online conversion, even under post-decision full-information value-oracle feedback: the extra queries can update future states, but they cannot retroactively change the current action.

In particular, the ofline construction underlying the 0.401 guarantee cannot simply be inserted into an online wrapper. Its Box-Maximization step is anticipatory from the online viewpoint: it must inspect the current function before selecting the corrective point. Running that step after round t produces an action too late to earn reward from $f _ { t }$ , while running it on past or averaged objectives does not control the current reward under adversarial variation. Consequently, the existing ofline guarantee does not by itself yield an online guarantee through follow-the-leader, delayed execution, or a standard ofline-to-online black-box reduction.

The online construction therefore does not run the ofline algorithm on a changing objective. It transfers the delayed-trajectory inequalities and the asymmetric coeficient requirements, then replaces the unavailable per-round box optimum by an independent online unconstrained-maximization state. Its certificate is cumulative: it pays the delayed-trajectory debts after summation over the sequence, rather than eliminating them pointwise. This amortization is the conceptual reason the ofline coeficient can be preserved under changing objectives.

The online composition. The central technical contribution is the chain

$$
| f _ { t } \longrightarrow G _ { t } ( a ) = f _ { t } ( x _ { t } \odot a ) \longrightarrow \mathrm { w e i g h t e d ~ o n l i n e ~ U S M ~ \longrightarrow ~ o u t e r ~ d e l a y e d ~ t r a j e c t o r y } .\tag{2}
$$

Here USM denotes unconstrained submodular maximization. The structural inequality exposes the debts that the transformed objectives $G _ { t }$ must control; the weighted online USM learner realizes exactly the asymmetric coeficients needed to pay those debts cumulatively; and the outer online learner absorbs the remaining linear term. A randomized choice between the inner and outer candidates then produces the reward guarantee without using concavity along an invalid direction.

Online timing. The adversary fixes $f _ { 1 } , \ldots , f _ { T }$ before play begins. At the start of round t, the history determines the outer state $x _ { t }$ and the states of all weighted online USM learners. Using only those states and fresh internal randomness, the inner learner selects $a _ { t } ;$ the algorithm then forms $x _ { t } \odot a _ { t }$ and $Y _ { 1 } ( x _ { t } )$ , draws its mixture coin, and commits to the played action $p _ { t }$ . Only after this commitment does $f _ { t }$ become available, and then only through calls to a noisy oracle. Those observations update the inner learners and construct the field used to form $x _ { t + 1 }$ . Thus $x _ { t } , a _ { t } , p _ { t }$ are all selected without current-round feedback, while $f _ { t }$ is used only for future decisions. Placing every oracle call strictly after every decision of the round is also what makes the noise analysis go through: each estimate is conditionally unbiased given the entire round’s decisions, which is the hypothesis of Lemma 4.13.

The outer ingredient is a comparator-uniform specialization of the delayed continuous-greedy comparison developed by Buchbinder and Feldman [10]; Appendix D supplies a self-contained proof of the precise form used here. For an outer state $x \in K$ , a delay parameter $s ,$ and the feasible endpoint $Y _ { 1 } ( x )$ , the inequality has the schematic form

$$
f ( Y _ { 1 } ( x ) ) \geq \theta _ { s } f ( o ) - \chi _ { s } f ( x \odot o ) - 2 \zeta _ { s } f ( x ) - \langle \widetilde { q } _ { s } ( f , x ) , o - x \rangle .\tag{3}
$$

Here $o \in K$ is an arbitrary fixed comparator, $\theta _ { s } , \chi _ { s } , \zeta _ { s }$ are explicit scalar functions of the delay, ⊙ denotes coordinatewise multiplication, and $\widetilde { q } _ { s } ( f , x )$ is a path-integrated gradient field. The first term contains the desired comparator value. The next two terms are the structural debts that must be repaid, and the final term is linear in $o - x$ . The outer learner performs projected online gradient ascent with $\widetilde { q } _ { s }$ . Consequently, the last term in (3) contributes only ordinary online linear-optimization regret.

The nonlinear debts require a diferent mechanism. Before receiving any current-function feedback, an inner learner chooses $a _ { t } \in [ 0 , 1 ] ^ { d }$ and, after the action is committed, obtains the value-oracle evaluations needed for the transformed objective

$$
G _ { t } ( a ) = f _ { t } ( x _ { t } \odot a ) .\tag{4}
$$

This transformation is geometrically exact: $x _ { t } \odot a \le x _ { t }$ , so down-closedness guarantees that every inner action maps to a feasible point. It also exposes precisely the values appearing in the structural inequality:

$$
G _ { t } ( o ) = f _ { t } ( x _ { t } \odot o ) , \qquad G _ { t } ( { \bf 1 } ) = f _ { t } ( x _ { t } ) , \qquad G _ { t } ( { \bf 0 } ) = f _ { t } ( { \bf 0 } ) ,\tag{5}
$$

where 1 and 0 are the all-ones and all-zeros vectors, respectively, and nonnegativity makes $G _ { t } ( \mathbf { 0 } )$ harmless. We prove a weighted online USM guarantee that simultaneously lower-bounds the cumulative inner reward by weighted sums of these three values. Combining this guarantee with (3), and randomly choosing between $Y _ { 1 } ( x _ { t } )$ and $x _ { t } \odot a _ { t } .$ , cancels both structural debts. Optimizing the delay, asymmetry, and mixture weights yields the certified factor 0.401.

The central new technical result behind the inner guarantee is an exact weighted online balance theorem. We discretize each coordinate into m levels, lift the box objective to a submodular set function on dm elements, and run an online Double-Greedy decision for each lifted element. The two Double-Greedy marginals enter with unequal coeficients, so the usual symmetric balance theorem is insuficient. For arbitrary positive weights $c _ { X } , c _ { Y }$ , we characterize the largest feasible balance constant as

$$
a _ { \mathrm { { m a x } } } ( c _ { X } , c _ { Y } ) = 2 { \sqrt { c _ { X } c _ { Y } } } .\tag{6}
$$

The proof uses the support function of the Blackwell target set and supplies an explicit polynomialtime approachability strategy. This distinction is important: a response that balances one adversarial marginal pair need not approach the target set over a sequence. The support-function calculation establishes the required uniform halfspace condition and also proves optimality of the constant. Under the Buchbinder–Feldman normalization $\left( c _ { X } , c _ { Y } \right) = \left( 1 / r , r \right)$ , the geometric mean is one, so (6) retains the full constant 2 for every asymmetry parameter $r > 0$ . Thus the ofline asymmetric coeficients survive online; only additive balance and discretization errors remain.

This is also why the online and ofline factor-revealing programs agree at the certified delay. In Section 4.3.3 we show that both programs equalize at the same $r _ { s } = 4 \zeta _ { s } / \chi _ { s }$ and, whenever $r _ { s } \ge 2$ are optimized over the asymmetry at that common switch. The two constructions repay the $f ( x \oplus o )$ debt by diferent mechanisms: an approximate local maximum ofline and a lattice inequality whose linear residue is absorbed by the outer online learner here. At the certified delay and balanced asymmetry, the two mechanisms nevertheless have the same value.

The primary feedback theorem gives a complete post-decision full-information value-oracle realization of this composition. All Double-Greedy marginals are estimated from $O ( d m )$ calls to the current function. The outer field is an integral of gradients along the delayed trajectory; adapting standard black-box finite-diference ideas for continuous submodular optimization [23], midpoint quadrature and inward one-sided diferences reconstruct it, at a field accuracy $\varepsilon ,$ using $O ( d / \varepsilon )$ additional calls with $O ( ( B + L ) \varepsilon { \sqrt { d } } )$ bias, where B bounds the gradients and L their Lipschitz constant. Crucially, the complete query list is nonadaptive once the pre-round outer and inner decisions have been made. This preserves the online timing requirement. Directly, the method uses

$O ( d ( m + \varepsilon ^ { - 1 } ) )$ calls per round; with $m = \Theta ( T ^ { 1 / 4 } )$ and $\varepsilon = \Theta ( T ^ { - 1 / 4 } )$ , it has $O ( T ^ { 3 / 4 } )$ regret and $O ( d T ^ { 1 / 4 } )$ calls per round.

Noisy feedback throughout. We do not assume that the oracle answers exactly. Each call returns a value that is conditionally unbiased given everything decided before it and difers from the truth by at most a known $\sigma \geq 0$ , with independent errors on distinct calls; the exact model is the case $\sigma = 0$ . We call this bounded conditionally unbiased noise. The analysis requires three adjustments. Values are recomputed rather than cached across occurrences; the outer field estimate is left unclipped and its step size is calibrated to a second moment; and the outer regret is split so that only estimator bias is charged at rate $T$ , while random fluctuation is charged at rate $\sqrt { T }$ Consequently, the noise level enters the guarantees only through the observation scale $M _ { \sigma } = M + \sigma$ where M bounds the reward values, and one lower-order term. For every fixed $\sigma _ { : }$ , the approximation factor, regret exponents, and per-round call budgets are unchanged.

As secondary consequences, the nonadaptive query list can be spread over a block, giving $O ( T ^ { \delta } )$ calls per round and $O ( T ^ { 4 / 5 - \delta / 5 } )$ regret for every $\delta \in [ 0 , 1 / 4 ]$ . The same nonadaptivity also enables a one-point bandit conversion by randomized block injection, which under the positive-anchor condition gives $O ( T ^ { 5 / 6 } )$ regret. These extensions alter the additive terms and oracle budget, not the 0.401 approximation factor.

What is inherited and what is new. The outer comparison inequalities are specializations of Lemmas 5.7 and 5.9 of Buchbinder and Feldman [10]; Appendix D proves the specialized, comparatoruniform statements from first principles. The asymmetric coeficient triple $\left( u _ { r } , v _ { r } , w _ { r } \right)$ is likewise inherited from the ofline unbalanced Double-Greedy analyses of Mualem and Feldman and of Buchbinder and Feldman [14, 10]. On the online side, our sequential binary reduction follows the online Double-Greedy template of Roughgarden and Wang, whose symmetric balance subproblem can be implemented through Blackwell approachability [15, 17]. The outer update is projected online gradient ascent in the sense of Zinkevich [16], and the value reconstruction builds on the black-box finite-diference literature [23].

The new statements proved here are the exact weighted balance constant $2 \sqrt { c _ { X } c _ { Y } }$ and its explicit approachability strategy, including the estimated-marginal guarantee needed for noisy feedback; the cumulative asymmetric online box certificate that retains both endpoint terms; and the amortized composition coupling this certificate to the delayed outer trajectory. We also prove that the resulting factor-revealing program has the same asymmetry-optimized value as the ofline program at the certified delay. Finally, we use randomized rather than convex mixing of the two candidate actions and exploit query-list nonadaptivity through a uniform injection for limited-query and bandit feedback. The contribution is the weighted online realization and its composition, not a re-derivation of the ofline coeficient frontier or the standard online-learning primitives.

Contributions. Our results are as follows.

1. We prove an adversarial online approximation factor of 0.401 with sublinear regret, matching the best known constructive ofline factor of Buchbinder and Feldman [10]. We make no optimality claim; a related multilinear-extension value-oracle hardness benchmark is 0.478 [12, 13].

2. We introduce the online composition $f _ { t } \mapsto G _ { t } ( a ) = f _ { t } ( x _ { t } \odot a )$ 7→ weighted online USM 7→ outer delayed trajectory. Its cumulative box certificate amortizes the comparator-dependent debts that cannot be removed pointwise. Proposition 4.26 gives the explicit algebra showing that, at the certified delay, the online and ofline coeficient programs agree after optimizing the asymmetry parameter. We stress that this is an equality of two numbers, namely the optimized coeficients, and not a correspondence between the two algorithms: ofline, the $f ( x \oplus o )$ debt is discharged by an approximate local maximum computed from the current objective; here it is discharged by a lattice inequality whose linear residue the outer learner absorbs over time.

3. We formulate the weighted balance game, in which the two Double-Greedy decisions earn at unequal rates, and solve it exactly: the largest approachable constant is $a _ { \mathrm { m a x } } ( c _ { X } , c _ { Y } ) = 2 \sqrt { c _ { X } c _ { Y } } ,$ with a matching impossibility direction and an explicit polynomial-time Blackwell strategy of $O ( \sqrt { T } )$ balance regret. New here are the weighted game, the exact constant in both directions, the uniform support-function argument, and the strategy; the coeficient triple $\left( u _ { r } , v _ { r } , w _ { r } \right)$ that the constant then yields at $\left( c _ { X } , c _ { Y } \right) = \left( 1 / r , r \right)$ is the ofline frontier of [14, 10], reached here through a count lifting and a cumulative online guarantee rather than reproved (Remark 4.11).

4. We implement the composition in the post-decision full-information value-oracle model with $O ( T ^ { 3 / 4 } )$ regret and $O ( d T ^ { 1 / 4 } )$ oracle calls per round. Secondary extensions give the limited-query frontier and an $O ( T ^ { 5 / 6 } )$ one-point bandit rate under the positive-anchor condition. These guarantees hold under conditionally unbiased bounded noisy value feedback.

Together, these results isolate the role of feedback. Neither adversarial variation nor observation noise nor the considered one-point bandit model under the additional positive-anchor condition forces the approximation factor back to $1 / e ;$ weaker or noisier feedback instead appears through the additive learning rate and the number of rounds needed to simulate the structural certificate.

## 2 Related work

Ofline DR-submodular maximization. Continuous DR-submodular maximization extends multilinear-extension methods for discrete submodular maximization. Measured continuous greedy and its continuous variants established $1 / e$ as a basic guarantee for nonnegative non-monotone objectives over down-closed convex bodies [5, 1]. Subsequent nonsymmetric and aided constructions improved the guarantee through approximately 0.372 and 0.385 [8, 9]. Buchbinder and Feldman introduced a sharper DR-submodular inequality and an asymmetric Box-Maximization component, obtaining the 0.401 benchmark used in this paper [10]. Jadav, Singh, and Aggarwal subsequently extended this framework to non-monotone γ-weakly DR-submodular objectives, recovering the same 0.401 factor when $\gamma = 1 ~ [ 1 1 ]$ . For the related multilinear-extension value-oracle problem, Oveis Gharan and Vondr´ak proved that no polynomial-query algorithm can guarantee better than 0.478, even for a partition matroid polytope, and Qi showed that the same bound applies under a cardinality constraint [12, 13]. Continuous Double-Greedy gives the optimal $1 / 2$ guarantee on the unconstrained box, matching the hardness of [6], and motivates the inner primitive in our construction [7, 4, 18]. The unbalanced form of the Double-Greedy guarantee that we reproduce online, namely the $\left( u _ { r } , v _ { r } , w _ { r } \right)$ frontier, was isolated by Mualem and Feldman [14] and, for DRsubmodular objectives, is Theorem 2.5 and Corollary 2.6 of [10].

Online submodular maximization. Chen, Hassani, and Karbasi initiated online continuous submodular maximization for monotone objectives [20], and Sessa et al. studied time-varying DRsubmodular resource allocation motivated by shared mobility [21]. For set functions, Roughgarden and Wang gave an optimal no-regret reduction for online unconstrained submodular maximization based on Blackwell approachability [15, 17]. We require a weighted, asymmetric form of their balance game and determine its exact constant. For continuous non-monotone DR-submodular objectives,

Thang and Srivastav gave early approximate-regret guarantees over down-closed and general convex sets [19]. Mualem and Feldman studied the distinct geometry of general, not necessarily downclosed, convex bodies [22]. More recent linearization frameworks organize first-order, adaptive, and nonstationary guarantees for broad classes of DR-submodular problems [26, 27, 30, 28]. Those works reduce a structural inequality to online linear optimization; our focus is instead the asymmetric box layer needed to reach the 0.401 down-closed benchmark.

Value-query and bandit feedback. Black-box continuous submodular optimization uses smoothing or finite diferences to replace gradients by value queries [23]. For monotone multilinear DR-submodular rewards, Wan et al. developed bandit algorithms and reductions to adversarial submodular bandits [25]. In the online non-monotone setting, Zhang et al. obtained a $1 / e$ guarantee with $O ( \sqrt { T } )$ regret using many stochastic gradients, a one-shot $O ( T ^ { 4 / 5 } )$ first-order method, and a one-point bandit method with ${ \cal O } ( T ^ { 8 / 9 } )$ regret [24]. Pedramfar et al. developed unified projection-free algorithms and obtained, for value feedback, $\bar { O } ( \bar { T } ^ { 4 / 5 } )$ post-decision value-oracle regret and $O ( T ^ { 5 / 6 } )$ bandit regret [29]. Pedramfar and $\mathrm { A }$ ggarwal subsequently gave general feedback conversions through upper-linearizability [26]. Lu et al. report value-feedback rates of $O ( T ^ { 3 / 4 } )$ under post-decision value-oracle queries and $O ( T ^ { 4 / 5 } )$ under bandit feedback, while retaining factor $1 / e \ [ 3 0 ]$ . They explicitly left breaking the $1 / e$ online approximation barrier as an open problem. Our result resolves that question under the feedback models studied here. Its bandit $O ( T ^ { 5 / 6 } )$ additive term is larger than the preceding $1 / e$ rate, but it preserves the strictly stronger 0.401 approximation factor. The argument is tailored to the present value-oracle construction and uses the nonadaptivity of the within-round query list to estimate an entire block-average objective from one observation per physical round.

## 3 Problem formulation

Let $K \subseteq [ 0 , 1 ] ^ { d }$ be nonempty, compact, convex, and down-closed: if $x \in K$ and $0 \leq y \leq x$ , then $y \in K$ . Since $K$ is nonempty and down-closed, $\mathbf { 0 } \in K$ . Write $D = \dim ( K )$ , so that $D \leq \sqrt { d }$ . For a positive integer $n ,$ write $[ n ] = \{ 1 , \dots , n \}$ , and let $e _ { i }$ denote the ith standard basis vector of $\mathbb { R } ^ { d }$ . For vectors $x , o \in [ 0 , 1 ] ^ { d }$ , define

$$
( x \odot o ) _ { i } = x _ { i } o _ { i } , \qquad ( x \oplus o ) _ { i } = x _ { i } + o _ { i } - x _ { i } o _ { i } .
$$

Inequalities between vectors are coordinatewise, and a vector in an exponent is understood coordinatewise.

A diferentiable function $f : [ 0 , 1 ] ^ { d } \to \mathbb { R }$ is DR-submodular if

$$
x \leq y \quad \implies \quad \partial _ { i } f ( x ) \geq \partial _ { i } f ( y )
$$

for every coordinate $i ;$ equivalently, $\nabla f$ is an antitone map. It is L-smooth if its gradient is L-Lipschitz in Euclidean norm. Throughout, $T , d \geq 1 , M > 0$ , and $B , L \geq 0$ . An oblivious adversary fixes diferentiable, nonnegative, L-smooth DR-submodular rewards $f _ { 1 } , \dots , f _ { T } : [ 0 , 1 ] ^ { d } \to [ 0 , M ]$ satisfying

$$
\operatorname* { s u p } _ { t , x } \| \nabla f _ { t } ( x ) \| _ { 2 } \leq B .
$$

The bounds $D , M , B , L$ , and the noise level $\sigma$ introduced below, are known to the learner for parameter tuning. For a nonempty closed convex set $C ,$ let $\Pi _ { C }$ denote Euclidean projection onto $C$ The learner has a polynomial-time oracle for $\Pi _ { K }$

At round t, the learner chooses a played point $p _ { t } \in K$ measurable with respect to the history $\mathcal { H } _ { t - 1 }$ and its fresh randomization, where $\mathcal { H } _ { t }$ is the sigma-field generated by the learner’s actions, randomization, and observations through round t. The choice is committed before any information about $f _ { t }$ is received. The reward credited on round t is the true value $f _ { t } ( p _ { t } )$ ; what the learner observes about that function is produced by the stochastic oracle described next.

Noisy value oracle. All feedback in this paper comes from a single noisy value oracle. We introduce it here, rather than as a later robustness extension, because every result below is proved directly in this model. A call to the oracle specifies a point $u \in [ 0 , 1 ] ^ { d }$ and returns a scalar $\widehat { f } _ { t } ( u )$ Let G be the sigma-field generated by everything determined before the call: the history $\mathcal { H } _ { t - 1 }$ , the obliviously fixed current function $f _ { t } ,$ all randomization used to select the round’s actions and its query list, and all previous calls. For a known noise level $\sigma \geq 0$ we assume

$$
\mathbb { E } \left[ \widehat { f } _ { t } ( u ) \mid \mathcal { G } \right] = f _ { t } ( u ) , \qquad \left| \widehat { f } _ { t } ( u ) - f _ { t } ( u ) \right| \leq \sigma \quad \mathrm { a l m o s t ~ s u r e l y } ,\tag{7}
$$

and that distinct calls carry conditionally independent errors, even when they specify the same point. We refer to $( 7 )$ throughout as bounded conditionally unbiased noise, and to the oracle as a noisy value oracle; $\sigma$ is called the noise level and bounds the error in magnitude, so that the conditional variance of a single call is at most $\sigma ^ { 2 }$ . Both quantities occur below: $\sigma$ wherever a range is needed and $\sigma ^ { 2 }$ wherever a second moment is. No other notion of noise appears in the paper. Conditional independence is what makes repeated sampling informative; in exchange, no algorithm below may cache a value across two of its occurrences, so every occurrence of a value inside a marginal or a finite diference is charged as a separate call. Observed values lie in $[ - \sigma , M + \sigma ]$ , and we write

$$
M _ { \sigma } : = M + \sigma\tag{8}
$$

for this observation scale. The exact-feedback model is the special case $\sigma = 0$ , in which case $M _ { \sigma } = M$ and every guarantee below reduces to its noiseless form; no separate exact-oracle statement is needed.

Within this oracle we consider two feedback models, which difer in where calls may be placed, not in how they are corrupted.

1. Post-decision full-information value-oracle feedback. After playing $p _ { t } .$ , the learner may call the oracle at polynomially many points of $[ 0 , 1 ] ^ { d }$ . The number of returned scalar values is part of the guarantee; the algorithm does not require a separate observation at $p _ { t }$

2. One-point bandit feedback. The learner receives exactly one scalar per round, the noisy value $\widehat { f _ { t } } ( \boldsymbol { p } _ { t } )$ of the point it actually played. Every value used in an update must therefore be obtained as the observed reward of a feasible played point, and the learner never sees the noiseless reward it earns.

Remark 3.1 (Which parts of (7) are needed). Condition (7) states two requirements: conditional unbiasedness, $\mathbb { E } [ \widehat { f } _ { t } ( u ) \mid \mathcal { G } ] = f _ { t } ( u )$ , and the almost-sure error bound $| \widehat { f } _ { t } ( u ) - f _ { t } ( u ) | \leq \sigma$ . The two are used diferently. In the outer layer the error enters only through a conditional second moment, so there the almost-sure bound may be replaced by a conditional variance bound $\sigma ^ { 2 }$ or a sub-Gaussian parameter $\sigma$ without changing any exponent. The inner layer is more demanding: Lemma 4.13 needs the estimated payof vectors to lie in a bounded set, both for the Blackwell approachability rate and for the martingale bound that transfers its conclusion to the true payofs. Almost-sure boundedness is therefore assumed globally. The frequently used assumption that the observed value itself lies in $[ 0 , M ]$ is the special case $\sigma \leq M$ , for which $M _ { \sigma } \leq 2 M$ and all our bounds coincide with their exact-feedback form up to absolute constants. Conditional unbiasedness, by contrast, cannot be weakened for free: a systematic bias $\beta$ propagates undamped through the Double-Greedy marginals and is divided by the finite-diference step in the outer field, contributing terms of order $d m \beta T$ and $D \beta { \sqrt { d } } T / h$ that no amount of repetition removes. At the balanced parameters of Corollary 4.2 the second of these is of order $\beta T ^ { 5 / 4 }$ , so a bias must already vanish as fast as $T ^ { - 1 / 2 }$ merely to stay within our $T ^ { 3 / 4 }$ rate, and any slower decay dominates it.

For $\alpha \in ( 0 , 1 ]$ , define approximate static regret

$$
\mathrm { R e g } _ { \alpha } ( T ) = \alpha \operatorname* { m a x } _ { o \in K } \sum _ { t = 1 } ^ { T } f _ { t } ( o ) - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } f _ { t } ( p _ { t } ) \right] .\tag{9}
$$

The expectation is over all learner randomization. We seek a constant α and regret $o ( T )$ . The horizon $T$ is assumed known when setting the algorithmic parameters; a standard doubling schedule removes this assumption at the cost of a constant factor (Remark 4.27).

Remark 3.2 (The factor is asymptotic). Throughout, “approximation factor $\alpha ^ { \mathfrak { N } }$ means the coeficient of the benchmark in (9), and nothing more. A guarantee $\mathrm { R e g } _ { \alpha } ( T ) \le R ( T )$ says that the ratio actually delivered at horizon $T$ is at least

$$
\alpha - \frac { R ( T ) } { \mathrm { O P T } _ { T } } , \qquad \mathrm { O P T } _ { T } = \operatorname* { m a x } _ { o \in K } \sum _ { t = 1 } ^ { T } f _ { t } ( o ) ,
$$

which is a weaker statement than $^ { 6 6 } \mathrm { a n }$ α-approximation” and can be vacuous at small $T .$ . With $\mathrm { O P T } _ { T } = \Theta ( T )$ , the deficit is $O ( T ^ { - 1 / 4 } )$ for the direct algorithm and ${ \cal O } ( T ^ { - 1 / 6 } )$ for the bandit algorithm, each multiplied by the problem-dependent constants of Corollaries 4.2 and $5 . 5 ;$ and if $\mathrm { O P T } _ { T }$ is itself sublinear, no constant factor is guaranteed at all. Our claim to match the ofline factor is a claim about the coeficient $\alpha ^ { \star }$ in the limit, not about finite-horizon performance, and the reader should read every “factor $0 . 4 0 1 ^ { \ ' }$ in this paper, including the rows of Table 2, in that sense.

Notation. Because the construction has an outer and an inner layer, several families of symbols appear together. We fix them once here.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $d , K , D , M , B , L$ </td><td>dimension, feasible set, diameter, value bound, gradient bound, smoothness</td></tr><tr><td>S</td><td>delay parameter of the trajectory; fixed to  $s _ { 0 }$  in the theorems</td></tr><tr><td> $Y _ { \tau } ( x ) , \omega _ { \tau } ( x )$ </td><td>delayed trajectory from x and its multiplier; endpoint  $Y _ { 1 } ( x )$ </td></tr><tr><td> $\theta _ { s } , \mu _ { s } , \nu _ { s } , \kappa _ { s }$ </td><td>endpoint coefficients of  $f ( o ) , f ( x \odot o ) , f ( x \oplus o )$ </td></tr><tr><td> $\zeta _ { s } = \nu _ { s } - \kappa _ { s } , \ \chi _ { s } = \mu _ { s } - \zeta _ { s }$ </td><td>two-coefficient form of the same inequality</td></tr><tr><td> $q _ { s } , \widetilde { q } _ { s } , \Gamma _ { s }$ </td><td>path-integrated field, its modification, and the bound  $\| \widetilde { q } _ { s } \| _ { 2 } \le \Gamma _ { s }$ </td></tr><tr><td> $\sigma , M _ { \sigma } = M + \sigma$ </td><td>oracle noise level (7) and observation scale (8)</td></tr><tr><td> $\widehat { q } _ { t } , \Delta _ { h , Q }$ </td><td>stochastic field estimate and its bias bound</td></tr><tr><td> $V _ { h , Q } , V _ { h , Q } ^ { \mathrm { b } }$  M</td><td>second-moment proxies for the direct and the block-simulated field estimate  $\bar { M } = 2 M _ { \sigma }$  in every application</td></tr><tr><td> $r , \left( c _ { X } , c _ { Y } \right) = \left( 1 / r , r \right)$ </td><td>generic payoff scale in the balance lemmas;</td></tr><tr><td> $u _ { r } , v _ { r } , w _ { r }$ </td><td>asymmetry parameter and balance weights box coefficients of  $G ( o ) , G ( { \bf 1 } ) , G ( { \bf 0 } )$ </td></tr><tr><td> $a , \lambda$ </td><td>balance constant; probability of playing the trajectory endpoint</td></tr><tr><td> $\alpha ( s , r ) , \alpha ^ { \star }$ </td><td>factor-revealing expression (52) and the headline constant 0.401</td></tr><tr><td> $g _ { t } ^ { + } , g _ { t } ^ { - }$ </td><td>the two adversarial marginals of the balance game</td></tr><tr><td> $m , \phi _ { m } , V _ { m }$ </td><td>grid resolution, count lifting, lifted ground set</td></tr><tr><td> $\varepsilon , h , Q$ </td><td>field accuracy, finite-difference step, number of quadrature nodes</td></tr><tr><td> $H , N , I _ { b } , \overline { { f } } _ { b }$ </td><td>block length, number of blocks, bth block, block-average objective</td></tr><tr><td> $\gamma , \bar { x } , \rho$ </td><td>contraction parameter, positive anchor, anchor margin</td></tr><tr><td>δ</td><td>query-budget exponent in the limited-query frontier</td></tr></table>

## 4 Post-decision full-information value-oracle result

All statements in this section are for the noisy oracle (7) at an arbitrary level $\sigma \geq 0 ;$ exact feedback is the case $\sigma = 0$

Main guarantees. Throughout the remainder of the paper, set

$$
\alpha ^ { \star } : = 0 . 4 0 1 .\tag{10}
$$

We state all theorem-level guarantees at this certified rounded factor; the explicit parameter certificate in Proposition 4.24 yields a slightly larger numerical coeficient. The construction combines the delayed endpoint certificate of [10], the asymmetric ofline box coeficients of [14, 10], an online Double-Greedy reduction modeled on [17], and projected stochastic online gradient ascent. The theorem below is the guarantee of this composition under the noisy value oracle (7). Since the structural parameters are fixed numerical constants, all constants hidden by $O ( \cdot )$ are absolute.

Two places in the algorithm are shaped by the noise and are worth flagging in advance. First, each lifted element’s pair of Double-Greedy marginals is formed from four fresh oracle calls, two per marginal, so the pair is an unbiased estimate of the true pair on a scale $O ( M _ { \sigma } )$ ; the balance learners are then driven by these estimates through Lemma 4.13. Second, the outer field estimate is not clipped to the ball of radius $\Gamma _ { s }$ , because clipping a noisy estimate would destroy its conditional unbiasedness; instead the outer step size is set from the second-moment proxy $V _ { h , Q }$ of Lemma 4.22, which reduces to the deterministic choice when $\sigma = 0$ . Since $C _ { 2 }$ and $C _ { 3 }$ in that lemma are absolute constants, $V _ { h , Q }$ is computable from $B , L , \sigma , d , Q , h ;$ any computable upper bound may be used instead, at the cost of a constant factor in the regret, and we write $\overline { V }$ for whichever bound the algorithm uses.

Theorem 4.1 (Online factor 0.401 with noisy post-decision value-oracle feedback). Under the assumptions of Section ${ \mathcal { B } } ,$ including the noisy value oracle (7) with level $\sigma \geq 0$ , let the oblivious adversary fix $f _ { 1 } , \ldots , f _ { T }$ before play. Run Algorithm 1 with the certified parameters $\left( s _ { 0 } , r _ { 0 } , \lambda _ { 0 } \right)$ of Proposition 4.24. For every integer grid size m $\geq 1$ and accuracy $\varepsilon \in ( 0 , 1 )$ , the algorithm selects $x _ { t } , a _ { t } , p _ { t }$ before receiving any feedback from $f _ { t }$ and satisfies, for every $o \in K$

$$
\begin{array} { c } { \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } f _ { t } ( p _ { t } ) \geq \alpha ^ { \star } \displaystyle \sum _ { t = 1 } ^ { T } f _ { t } ( o ) - O ( D B \sqrt { T } ) - O \big ( d m M _ { \sigma } \sqrt { T } \big ) - O \big ( B \sqrt { d } T / m \big ) } \\ { - O \Big ( D \big ( B + L \big ) \varepsilon \sqrt { d } T \Big ) - O \Big ( D \sigma \sqrt { d T / \varepsilon } \Big ) . } \end{array}\tag{11}
$$

The number of oracle calls to the current function in each round is

$$
O \big ( d ( m + \varepsilon ^ { - 1 } ) \big ) .\tag{12}
$$

For $\sigma = 0$ the last term vanishes and (11) is the exact-oracle guarantee. For $\sigma > 0$ that term is a genuine constraint $o n \varepsilon$ rather than a lower-order nuisance: it grows without bound as $\varepsilon \downarrow 0$ , and a suficiently small ε makes (11) vacuous. The assertion that noise does not afect the exponent is therefore a statement about the balanced parameters of Corollary $4 . 2 ,$ not a property of (11) at arbitrary $( m , \varepsilon )$

Corollary 4.2 (Balanced value-oracle rate). For $T \ \geq \ 2$ , with $m \ = \ \lceil T ^ { 1 / 4 } \rceil$ and $\varepsilon = T ^ { - 1 / 4 }$ 2 Algorithm 1 has

$$
\mathrm { R e g } _ { \alpha ^ { \star } } ( T ) = O \Big ( \big [ d M _ { \sigma } + B \sqrt { d } + D ( B + L ) \sqrt { d } \big ] T ^ { 3 / 4 } + D B \sqrt { T } \Big )
$$

and uses $O ( d T ^ { 1 / 4 } )$ oracle calls per round.

Remark 4.3 (The price of noise). Corollary 4.2 is, term by term, the exact-oracle bound with M replaced by $M _ { \sigma } = M + \sigma$ . Noise therefore costs a constant factor at the balanced parameters and does not afect the approximation factor, the $T ^ { 3 / 4 }$ exponent, or the $O ( d T ^ { 1 / 4 } )$ per-round call budget. The reason is a mismatch of scales: repetition suppresses the noise in the outer field at rate $\sigma { \sqrt { 1 / ( Q h ^ { 2 } ) } }$ , which at $Q = \Theta ( T ^ { 1 / 4 } ) , h = \Theta ( T ^ { - 1 / 4 } )$ contributes only at order $T ^ { 5 / 8 }$ , while the inner layer is dominated by the dm balance learners, whose $O ( M _ { \sigma } \sqrt { T } )$ regret is insensitive to the split of $M _ { \sigma }$ into signal and noise. Only if the noise level were allowed to grow with T, say $\sigma = \Theta ( T ^ { c } )$ for some $c > 0$ , would the exponent move, and then only through the product $d M _ { \sigma } T ^ { 3 / 4 }$

Two further points about ε. First, $\varepsilon = T ^ { - 1 / 4 }$ is not the choice that balances the two field terms of (11) against each other: equating $D ( B + L ) \varepsilon \sqrt { d } T$ with $D \sigma { \sqrt { d T / \varepsilon } } { \mathrm { ~ g i v e s ~ } } \varepsilon = \Theta { \bigl ( } ( \sigma / ( B + L ) ) ^ { 2 / 3 } T ^ { - 1 / 3 } { \bigr ) }$ and a field cost of order $T ^ { 2 / 3 }$ . We keep $\varepsilon = T ^ { - 1 / 4 }$ because the overall rate is set by the inner and grid terms at $T ^ { 3 / 4 }$ regardless, and because the same choice serves the exact case; nothing is lost by it. Second, the dominance used above is specific to this choice, as Theorem 4.1 notes.

One composition, three implementations. Three parameterizations appear in this paper, and it is worth saying plainly that they are not three algorithms. There is one composition and, for a given field accuracy, one query list; what changes is where that list is issued.

1. Direct (Section 4.3). The whole list is issued in the round whose action it serves, at field accuracy ε, with $h = \varepsilon / 2$ and $Q = \lceil \varepsilon ^ { - 1 } \rceil$

2. Batched (Theorem 4.4). The state is frozen over a block of H rounds; the same list is built, for the block average ${ \overline { { f } } } _ { b }$ instead of a single $f _ { t } .$ , and injected uniformly at random into the block at $\lceil T ^ { \delta } \rceil$ labels per round.

3. Bandit (Section 5). As batched, except that an injected label must be played rather than issued alongside the action, which costs exploration reward and forces every probe to be feasible.

So the frontier is the direct algorithm at a diferent accuracy plus a batching conversion, and the bandit algorithm is the batched one with the added feasibility requirement. In particular, the direct algorithm uses all of its queries in the round in which the action is played; nonadaptivity of the list is what permits the other two.

Theorem 4.4 (Limited-query post-decision value-oracle frontier). For every $\delta \in [ 0 , 1 / 4 ]$ and all suficiently large $T ,$ , run the composition of Algorithm 1 at the field accuracy

$$
\varepsilon _ { \delta } : = T ^ { - ( 1 + \delta ) / 5 } , \qquad h = \frac { \varepsilon _ { \delta } } { 2 } , \qquad m = Q = \lceil \varepsilon _ { \delta } ^ { - 1 } \rceil ,\tag{13}
$$

with its state frozen over blocks of length $H = \lceil C _ { \mathrm { q } } d T ^ { ( 1 - 4 \delta ) / 5 } \rceil$ , for a suficiently large numerical constant $C _ { \mathrm { q } }$ , and its query list injected into each block as in Appendix A. This algorithm uses at most $\lceil T ^ { \delta } \rceil$ calls to the noisy oracle (7) per physical round and satisfies

$$
\mathrm { R e g } _ { \alpha ^ { \star } } ( T ) = O \Big ( T ^ { 4 / 5 - \delta / 5 } \Big ) .\tag{14}
$$

The hidden constant is independent of T and depends polynomially on the problem parameters, including the noise level σ through $M _ { \sigma }$ . Thus $\delta = 0$ gives one call per round and $O ( T ^ { 4 / 5 } )$ regret, whereas $\delta = 1 / 4$ gives $O ( T ^ { 1 / 4 } )$ calls per round and $O ( T ^ { 3 / 4 } )$ regret.

The accuracy $\varepsilon _ { \delta }$ is the single parameter that ties the frontier to the direct algorithm, and we state it explicitly because the proof in Appendix B is written in the exponent $\xi = ( 1 + \delta ) / 5 .$ , so that $\varepsilon _ { \delta } = T ^ { - \xi }$ . Reading (13) at the two endpoints makes the relation to Corollary 4.2 concrete:
<table><tr><td></td><td> $\varepsilon _ { \delta }$ </td><td> $m = Q$ </td><td>block length H calls/round</td><td></td><td>regret</td></tr><tr><td> $\delta = 0$ </td><td> $T ^ { - 1 / 5 }$ </td><td> $T ^ { 1 / 5 }$ </td><td> $\Theta ( d T ^ { 1 / 5 } )$ </td><td>1</td><td> $O ( T ^ { 4 / 5 } )$ </td></tr><tr><td> $\delta = 1 / 4$ </td><td> $T ^ { - 1 / 4 }$ </td><td> $T ^ { 1 / 4 }$ </td><td> $\Theta ( d )$ </td><td> $O ( T ^ { 1 / 4 } )$ </td><td> $O ( T ^ { 3 / 4 } )$ </td></tr></table>

At $\delta = 1 / 4$ the accuracy is exactly the balanced $\varepsilon = T ^ { - 1 / 4 }$ of Corollary 4.2 and the block length $H = \Theta ( d )$ is constant in $T ,$ so the frontier endpoint reproduces the direct algorithm’s rate and budget in $T$ , the batching conversion contributing nothing asymptotically. It is not free in the dimension: the block of length $\Theta ( d )$ inflates the inner term from ${ \bar { d M } } _ { \sigma } T ^ { 3 / { \bar { 4 } } }$ to $d ^ { 3 / 2 } M _ { \sigma } T ^ { 3 / 4 }$ , while cutting the per-round budget from $\scriptstyle { \dot { O } } ( d T ^ { 1 / 4 } )$ to $\lceil T ^ { 1 / 4 } \rceil$ . Batching trades a factor $\sqrt { d }$ of regret for a factor d of query budget. Decreasing δ coarsens the accuracy and lengthens the block, buying a smaller per-round budget at a worse rate. The proof appears in Appendix B, using the block simulation machinery of Appendix A, which it shares with the bandit conversion.

## 4.1 Structural certificate

Delayed trajectory. For $\tau \in [ 0 , 1 ]$ , define

$$
\psi _ { \tau } ( x ) = { \bf 1 } - e ^ { - \tau x }
$$

coordinatewise. For a delay parameter $s \in ( 0 , 1 )$ , define

$$
Y _ { \tau } ( x ) = \left\{ \begin{array} { l l } { \left( 1 - x \right) \odot \psi _ { \tau } ( x ) , } & { 0 \le \tau < s , } \\ { \left( 1 - x \right) \odot \psi _ { \tau } ( x ) + x \odot \psi _ { \tau - s } ( x ) , } & { s \le \tau \le 1 . } \end{array} \right.\tag{15}
$$

The delay s is fixed throughout and suppressed from the notation. The endpoint of the trajectory is $Y _ { 1 } ( x )$ ; it plays the role of the output of a delayed measured continuous greedy run started at 0 with the constant direction x and the freeze $z = x$

The associated multiplier is

$$
\omega _ { \tau } ( x ) = \left\{ \begin{array} { l l } { ( 1 - x ) \odot e ^ { - \tau x } , } & { 0 \leq \tau < s , } \\ { ( 1 - x ) \odot e ^ { - \tau x } + x \odot e ^ { - ( \tau - s ) x } , } & { s \leq \tau \leq 1 . } \end{array} \right.\tag{16}
$$

Lemma 4.5 (Trajectory feasibility). For every $x \in K$ and $\tau \in [ 0 , 1 ]$ 2

$$
0 \leq Y _ { \tau } ( x ) \leq x , \qquad Y _ { \tau } ( x ) \in K , \qquad 0 \leq \omega _ { \tau } ( x ) \leq 1 .
$$

Moreover,

$$
\frac { d } { d \tau } Y _ { \tau } ( x ) = \omega _ { \tau } ( x ) \odot x
$$

almost everywhere, and

$$
\omega _ { \tau } ( x ) = \left\{ \begin{array} { l l } { \mathbf { 1 } - Y _ { \tau } ( x ) - x , } & { 0 \leq \tau < s , } \\ { \mathbf { 1 } - Y _ { \tau } ( x ) , } & { s \leq \tau \leq 1 . } \end{array} \right.\tag{17}
$$

In both phases $Y _ { \tau } ( x ) + \omega _ { \tau } ( x ) \odot o \le \mathbf { 1 }$ for every $o \in [ 0 , 1 ] ^ { d }$

Proof. For $\tau < s .$

$$
Y _ { \tau } ( x ) = ( \mathbf { 1 } - x ) \odot ( \mathbf { 1 } - e ^ { - \tau x } ) \le ( \mathbf { 1 } - x ) \odot \tau x \le x .
$$

For $\tau \geq s .$

$$
\begin{array} { c } { { Y _ { \tau } ( x ) \leq ( { \bf 1 } - x ) \odot \tau x + x \odot ( \tau - s ) x } } \\ { { \mathrm { } } } \\ { { = \tau x - s x \odot x \leq x . } } \end{array}
$$

Nonnegativity is immediate. Down-closedness gives $Y _ { \tau } ( x ) \in K$ . Diferentiation gives the stated dynamics. For (17), direct substitution gives, in the early phase,

$$
\mathbf { 1 } - Y _ { \tau } ( x ) - x = \mathbf { 1 } - ( \mathbf { 1 } - x ) \odot ( \mathbf { 1 } - e ^ { - \tau x } ) - x = ( \mathbf { 1 } - x ) \odot e ^ { - \tau x } = \omega _ { \tau } ( x ) ,
$$

and in the late phase

$$
\mathbf { 1 } - Y _ { \tau } ( x ) = ( \mathbf { 1 } - x ) \odot e ^ { - \tau x } + x \odot e ^ { - ( \tau - s ) x } = \omega _ { \tau } ( x ) .
$$

In the late phase $Y _ { \tau } + \omega _ { \tau } \odot o = Y _ { \tau } \oplus o \le { \bf 1 }$ ; in the early phase $Y _ { \tau } + \omega _ { \tau } \odot o \le Y _ { \tau } + \omega _ { \tau } \le { \bf 1 }$ . This last identity is what makes the two comparison inequalities of Lemma 4.6 applicable: the early phase compares against $Y _ { \tau } \oplus o - x \odot o$ and the late phase against $Y _ { \tau } \oplus o$ □

Define the path-integrated gradient

$$
q _ { s } ( f , x ) = \int _ { 0 } ^ { 1 } e ^ { \tau - 1 } \omega _ { \tau } ( x ) \odot \nabla f ( Y _ { \tau } ( x ) ) d \tau .\tag{18}
$$

Define the four scalar coeficients

$$
\theta _ { s } = \frac { ( 2 - s ) e ^ { s } - 1 } { e } , \qquad \mu _ { s } = \frac { e ^ { s } - 1 } { e } ,\tag{19}
$$

$$
\nu _ { s } = \frac { ( 2 - s ) e ^ { s } - 2 } { e } , \qquad \kappa _ { s } = \frac { ( 1 - s ) ^ { 2 } } { 2 e } .\tag{20}
$$

We now record the endpoint inequality that drives the outer layer. Its two ingredients (22)–(23) are the imported structural ingredients of this certificate: they are Lemmas 5.7 and 5.9 of [10], specialized to $z = x$ and $x ( \tau ) \equiv x$ . Since we use them with an arbitrary comparator rather than an optimizer, and since the exact coeficients (19)–(20) are used quantitatively later, we do not import them as a black box: Appendix D proves both from first principles.

Lemma 4.6 (Comparator-uniform endpoint inequality). For every $s \in ( 0 , 1 )$ , every nonnegative diferentiable DR-submodular $f _ { i }$ , and every x, $o \in K$ 2

$$
\begin{array} { c } { { f ( Y _ { 1 } ( x ) ) \geq \theta _ { s } f ( o ) - \mu _ { s } f ( x \odot o ) - \nu _ { s } f ( x \oplus o ) + \kappa _ { s } f ( x \oplus o ) } } \\ { { - \left. q _ { s } ( f , x ) , o - x \right. . } } \end{array}\tag{21}
$$

Proof. Appendix D establishes the two delayed exponential-mixture comparisons. For $\tau \ < \ s$ (Lemma D.6),

$$
\begin{array} { r } { f ( Y _ { \tau } + \omega _ { \tau } \odot o ) \geq f ( o ) - f ( x \odot o ) - ( 1 - e ^ { - \tau } ) f ( x \oplus o ) , } \end{array}\tag{22}
$$

and for $\tau \geq s$ (Lemma D.7),

$$
f ( Y _ { \tau } + \omega _ { \tau } \odot o ) \geq e ^ { - \tau } \left[ e ^ { s } f ( o ) - ( e ^ { s } - 1 ) f ( x \oplus o ) + ( \tau - s ) f ( x \oplus o ) \right] .\tag{23}
$$

These inequalities use only nonnegativity and DR-submodularity and therefore hold for every fixed comparator $o ;$ in particular they do not require o to maximize $f .$ This comparator-uniformity is what allows the same inequality to be summed against a single hindsight benchmark across a changing sequence.

Write $Y _ { \tau } = Y _ { \tau } ( x )$ and $\omega _ { \tau } = \omega _ { \tau } ( x )$ . Since $\dot { Y } _ { \tau } = \omega _ { \tau } \odot$ x almost everywhere and $Y _ { \tau } + \omega _ { \tau } \odot o \le { \bf 1 }$ by Lemma 4.5, property (P3) of Lemma D.1 gives

$$
\begin{array} { r } { \langle \omega _ { \tau } \odot \nabla f ( Y _ { \tau } ) , o \rangle \ge f ( Y _ { \tau } + \omega _ { \tau } \odot o ) - f ( Y _ { \tau } ) \ge L _ { \tau } - f ( Y _ { \tau } ) . } \end{array}
$$

Here $L _ { \tau }$ is defined piecewise: it is the right-hand side of (22) when $\tau < s .$ , and the right-hand side of (23) when $\tau \geq s .$ . Subtracting $\langle \omega _ { \tau } \odot \nabla f ( Y _ { \tau } ) , o - x \rangle$ yields

$$
\frac { d } { d \tau } f ( Y _ { \tau } ) + f ( Y _ { \tau } ) = \left. \omega _ { \tau } \odot \nabla f ( Y _ { \tau } ) , x \right. + f ( Y _ { \tau } ) \ge L _ { \tau } - \left. \omega _ { \tau } \odot \nabla f ( Y _ { \tau } ) , o - x \right. .
$$

Multiplying by $e ^ { \tau - 1 }$ and integrating from 0 to 1 telescopes the left-hand side to

$$
f ( Y _ { 1 } ( x ) ) - e ^ { - 1 } f ( \mathbf { 0 } ) .
$$

The early contribution is

$$
\frac { e ^ { s } - 1 } { e } \big ( f ( o ) - f ( x \odot o ) \big ) - \frac { e ^ { s } - 1 - s } { e } f ( x \circledast o ) ,
$$

and the late contribution is

$$
{ \frac { ( 1 - s ) e ^ { s } } { e } } f ( o ) - { \frac { ( 1 - s ) ( e ^ { s } - 1 ) } { e } } f ( x \oplus o ) + { \frac { ( 1 - s ) ^ { 2 } } { 2 e } } f ( x \oplus o ) .
$$

Adding them gives (21) with $( 1 9 ) \div ( 2 0 )$ : the coeficient of $f ( o )$ is $( e ^ { s } - 1 ) / e + ( 1 - s ) e ^ { s } / e = \theta _ { s }$ , that of $f ( x \odot o ) { \mathrm { ~ i s ~ } } - \mu _ { s } .$ and that of $f ( x \oplus o )$ is

$$
- \frac { ( e ^ { s } - 1 - s ) + ( 1 - s ) ( e ^ { s } - 1 ) } { e } + \frac { ( 1 - s ) ^ { 2 } } { 2 e } = - \frac { ( 2 - s ) e ^ { s } - 2 } { e } + \kappa _ { s } = - \nu _ { s } + \kappa _ { s } .
$$

The linear terms integrate to $- \left. q _ { s } ( f , x ) , o - x \right.$ . Finally, $e ^ { - 1 } f ( { \bf 0 } ) \geq 0$ may be discarded. □

The important point is that the comparator o is arbitrary; it need not be optimal for $f$

Two-coeficient decomposition. Define

$$
\zeta _ { s } : = \nu _ { s } - \kappa _ { s } , \qquad \chi _ { s } : = \mu _ { s } - \nu _ { s } + \kappa _ { s } = \mu _ { s } - \zeta _ { s } .\tag{24}
$$

Define the modified path field

$$
\widetilde { q } _ { s } ( \boldsymbol { f } , \boldsymbol { x } ) = q _ { s } ( \boldsymbol { f } , \boldsymbol { x } ) + \zeta _ { s } \nabla \boldsymbol { f } ( \boldsymbol { x } ) .\tag{25}
$$

The standard DR-submodular lattice comparison is

$$
f ( x \odot o ) + f ( x \oplus o ) \leq 2 f ( x ) + \langle \nabla f ( x ) , o - x \rangle .\tag{26}
$$

Indeed, $x \odot o \leq x \leq x \oplus o$ . Integrating the gradient on the segment from $x \odot o$ to x and using diminishing returns gives

$$
f ( x \odot o ) - f ( x ) \leq \langle \nabla f ( x ) , x \odot o - x \rangle ,
$$

and directional concavity on the nonnegative segment from x to $x \oplus o$ gives

$$
f ( x \oplus o ) - f ( x ) \leq \langle \nabla f ( x ) , x \oplus o - x \rangle .
$$

Adding these inequalities and using $( x \odot o - x ) + ( x \oplus o - x ) = o - x$ proves (26).

Lemma 4.7 (Two-coeficient endpoint form). For every s for which $\zeta _ { s } \geq 0$ and every $x , o \in K$

$$
f ( Y _ { 1 } ( x ) ) \geq \theta _ { s } f ( o ) - \chi _ { s } f ( x \odot o ) - 2 \zeta _ { s } f ( x ) - \langle \widetilde { q } _ { s } ( f , x ) , o - x \rangle .\tag{27}
$$

Proof. From Lemma 4.6,

$$
f ( Y _ { 1 } ( x ) ) \geq \theta _ { s } f ( o ) - \mu _ { s } f ( x \odot o ) - \zeta _ { s } f ( x \oplus o ) - \langle q _ { s } ( f , x ) , o - x \rangle .
$$

Since $\zeta _ { s } \geq 0$ , (26) gives

$$
- \zeta _ { s } f ( x \oplus o ) \geq - \zeta _ { s } \left( 2 f ( x ) + \langle \nabla f ( x ) , o - x \rangle - f ( x \odot o ) \right) .
$$

Collecting terms proves (27).

Two features of (27) deserve emphasis. First, the only comparator-dependent nonlinear quantities left are $f ( x \odot o )$ and $f ( x )$ , which are exactly the two values that the transformed objective $G ( a ) = f ( x \odot a )$ exposes at $a = o$ and $a = \mathbf { 1 }$ . Second, the residue of the lattice step is the linear term $\zeta _ { s } \left. \nabla f ( x ) , o - x \right.$ , which is absorbed into the outer online linear optimization rather than paid for pointwise. That is the mechanism that replaces the approximate local maximum of the ofline construction, and Section 4.3.3 shows it gives the same asymmetry-optimized value as the ofline coeficient program at the certified delay.

## 4.2 Weighted online unconstrained submodular maximization (USM)

Weighted balance game. We next isolate the binary online problem generated by Double-Greedy. Fix $c _ { X } , c _ { Y } > 0$ and $M > 0$ . At round t, the learner computes a mixed action $\pi _ { t } \in [ 0 , 1 ]$ using only its past observations and announces it. An admissible adversarial pair of marginals

$$
( g _ { t } ^ { + } , g _ { t } ^ { - } )
$$

is then selected; it may depend on the preceding history and on $\pi _ { t }$ , but not on the learner’s fresh binary draw. The learner next draws $\mathrm { ^ { 6 6 } y e s ^ { 7 } }$ with probability $\pi _ { t }$ and $\mathrm { ^ { 6 6 } n o ^ { 5 9 } }$ otherwise, and the payof vector is realized. We require

$$
g _ { t } ^ { + } , g _ { t } ^ { - } \in [ - M , M ] , \qquad g _ { t } ^ { + } + g _ { t } ^ { - } \geq 0 .
$$

In the noiseless idealization the realized pair is then revealed exactly. Under the oracle (7) the learner instead observes a conditionally unbiased estimate of the pair, formed from fresh calls after the draw; Lemma 4.13 below shows that this costs only a constant factor, with M replaced by the observation scale. We first solve the exact game, since its geometry is what determines the balance constant.

The two weighted decisions are defined as follows. A yes decision accepts the current element into the lower Double-Greedy set and earns the weighted marginal $c _ { X } g _ { t } ^ { + }$ ; a no decision rejects it from the upper set and earns $c _ { Y } g _ { t } ^ { - }$ . The opposite marginal is not earned, and is recorded in the appropriate missed-opportunity pile. Thus the reward and the two piles are part of one vector payof, rather than three independent quantities.

A realized “yes” action produces the payof vector

$$
U _ { t } ^ { Y } = ( c _ { X } g _ { t } ^ { + } , 0 , g _ { t } ^ { - } ) ,
$$

whereas a realized $^ { 6 6 } \mathrm { n o } ^ { 9 9 }$ action produces

$$
U _ { t } ^ { N } = ( c _ { Y } g _ { t } ^ { - } , g _ { t } ^ { + } , 0 ) .
$$

Thus the realized algorithmic reward and missed-opportunity piles are

$$
R _ { t } = \left\{ \begin{array} { l l l } { c _ { X } g _ { t } ^ { + } , } & { \mathrm { i f ~ y e s , } } \\ { c _ { Y } g _ { t } ^ { - } , } & { \mathrm { i f ~ n o , } } \end{array} \right. \quad \quad ( C _ { Y , t } , C _ { N , t } ) = \left\{ \begin{array} { l l l } { ( 0 , g _ { t } ^ { - } ) , } & { \mathrm { i f ~ y e s , } } \\ { ( g _ { t } ^ { + } , 0 ) , } & { \mathrm { i f ~ n o . } } \end{array} \right.
$$

Thus $C _ { Y }$ accumulates the missed yes-marginal on no decisions, while $C _ { N }$ accumulates the missed no-marginal on yes decisions. For a mixed action with probability π of $\mathrm { ^ { 6 6 } y e s ^ { 9 } }$ , the conditional expected payof vector is

$$
\bar { U } ( \pi ; g ^ { + } , g ^ { - } ) = \left( \pi c _ { X } g ^ { + } + ( 1 - \pi ) c _ { Y } g ^ { - } , ( 1 - \pi ) g ^ { + } , \pi g ^ { - } \right) .
$$

For $a > 0$ , define the target cone

$$
S _ { a } = \left\{ ( R , C _ { Y } , C _ { N } ) : R \geq a C _ { Y } , \quad R \geq a C _ { N } \right\} .\tag{28}
$$

Let $a _ { \mathrm { m a x } } ( c _ { X } , c _ { Y } )$ denote the supremum of the values of a for which $ { \boldsymbol { S } } _ { a }$ is approachable. For a horizon $T .$ , define the corresponding weighted balance regret by

$$
\mathrm { B a l R e g } _ { a } ( T ) = \mathbb { E } \left[ \operatorname* { m a x } \left\{ a \sum _ { t = 1 } ^ { T } C _ { Y , t } - \sum _ { t = 1 } ^ { T } R _ { t } , \ : a \sum _ { t = 1 } ^ { T } C _ { N , t } - \sum _ { t = 1 } ^ { T } R _ { t } , \ : 0 \right\} \right] .\tag{29}
$$

Expected distance ${ \cal O } ( T ^ { - 1 / 2 } )$ of the average payof from $ { \boldsymbol { S } } _ { a }$ implies $\mathrm { B a l R e g } _ { a } ( T ) = O ( { \sqrt { T } } )$ , up to the fixed geometric constants of the cone.

Writing $\mathcal { A } _ { M } = [ - M , M ] ^ { 2 } \cap \{ g ^ { + } + g ^ { - } \geq 0 \}$ and denoting the polar cone of $ { \boldsymbol { S } } _ { a }$ by $ { \boldsymbol { S } } _ { a } ^ { \circ }$ , Blackwell’s halfspace condition is

$$
\forall q \in S _ { a } ^ { \circ } \quad \exists \pi ( q ) \in [ 0 , 1 ] \quad \forall ( g ^ { + } , g ^ { - } ) \in A _ { M } : \quad q \cdot \bar { U } ( \pi ( q ) ; g ^ { + } , g ^ { - } ) \leq 0 .\tag{30}
$$

The final quantifier is uniform over all admissible pairs, including pairs chosen after seeing $\pi ( q )$ but before the fresh binary draw. This is the nonanticipating information pattern used below. In the sequential Double-Greedy reduction, the pair presented to learner i may depend on the current decisions of learners $1 , \ldots , i - 1$ , but not on learner i’s fresh randomization. The next theorem solves this condition exactly.

Theorem 4.8 (Exact asymmetric balance constant). For the weighted online balance game, the cone $ { \boldsymbol { S } } _ { a }$ is approachable if and only $i f a \le 2 \sqrt { c _ { X } c _ { Y } }$ . Consequently,

$$
\sqrt { a _ { \mathrm { m a x } } ( c _ { X } , c _ { Y } ) = 2 \sqrt { c _ { X } c _ { Y } } . }
$$

Proof. The cone $ { \boldsymbol { S } } _ { a }$ can be written as

$$
S _ { a } = \left\{ u \in \mathbb { R } ^ { 3 } : - ( 1 , - a , 0 ) \cdot u \leq 0 , \ - ( 1 , 0 , - a ) \cdot u \leq 0 \right\} .
$$

Hence every supporting normal in the polar cone may be written as

$$
q ( \lambda _ { 1 } , \lambda _ { 2 } ) = \bigl ( - ( \lambda _ { 1 } + \lambda _ { 2 } ) , a \lambda _ { 1 } , a \lambda _ { 2 } \bigr ) , \qquad \lambda _ { 1 } , \lambda _ { 2 } \geq 0 .
$$

$\mathrm { B y }$ Blackwell’s approachability theorem [15], it sufices to verify (30) for every such normal. Let $\Lambda = \lambda _ { 1 } + \lambda _ { 2 }$ . If $\Lambda = 0$ , there is nothing to prove. Otherwise put

$$
u = \frac { \lambda _ { 1 } } { \Lambda } \in [ 0 , 1 ] .
$$

For a mixed action with probability π of $\mathrm { ^ { 6 6 } y e s ^ { 7 9 } }$ , its inner product with $q$ is

$$
A _ { + } g ^ { + } + A _ { - } g ^ { - } ,
$$

where

$$
A _ { + } = - \pi c _ { X } \Lambda + a \lambda _ { 1 } ( 1 - \pi ) ,\tag{31}
$$

$$
A _ { - } = - ( 1 - \pi ) c _ { Y } \Lambda + a \lambda _ { 2 } \pi .\tag{32}
$$

The admissible adversary region

$$
[ - M , M ] ^ { 2 } \cap \{ g ^ { + } + g ^ { - } \geq 0 \}
$$

is the triangle with vertices $( M , M ) , ( M , - M ) , ( - M , M )$ . Therefore

$$
\operatorname* { s u p } _ { g ^ { + } + g ^ { - } \geq 0 } ( A _ { + } g ^ { + } + A _ { - } g ^ { - } ) = M \operatorname* { m a x } \{ A _ { + } + A _ { - } , ~ A _ { + } - A _ { - } , ~ - A _ { + } + A _ { - } \} .
$$

This maximum is nonpositive whenever $A _ { + } = A _ { - } \leq 0$

Choose

$$
\pi = { \frac { c _ { Y } + a u } { c _ { X } + c _ { Y } + a } } .\tag{33}
$$

This is a valid probability because its numerator is positive and the diference between its denominator and numerator is $c _ { X } + a ( 1 - u ) > 0$ . A direct substitution gives

$$
A _ { + } = A _ { - } = - \Lambda \frac { a ^ { 2 } u ^ { 2 } - a ^ { 2 } u + c _ { X } c _ { Y } } { a + c _ { X } + c _ { Y } } = \Lambda \frac { a ^ { 2 } u ( 1 - u ) - c _ { X } c _ { Y } } { a + c _ { X } + c _ { Y } } .\tag{34}
$$

Hence $A _ { + } = A _ { - } \leq 0$ for every $u \in [ 0 , 1 ]$ if and only if $a ^ { 2 } u ( 1 - u ) \leq c _ { X } c _ { Y }$ for every such $u .$ Since $\mathrm { n a x } _ { u \in [ 0 , 1 ] } u ( 1 - u ) = 1 / 4$ , this is equivalent to $a \leq 2 \sqrt { c _ { X } c _ { Y } }$ . Thus $\scriptstyle { \mathcal { S } } _ { a }$ is approachable for $a \leq 2 { \sqrt { c _ { X } c _ { Y } } }$

For the converse, take the fixed adversary point

$$
( g ^ { + } , g ^ { - } ) = c _ { 0 } \left( \sqrt { c _ { Y } } , \sqrt { c _ { X } } \right) , \qquad c _ { 0 } = \frac { M } { \mathrm { m a x } \{ \sqrt { c _ { X } } , \sqrt { c _ { Y } } \} } ,
$$

which is admissible and has both coordinates positive. Repeat this same move for $T$ rounds, and let $p _ { T } \in [ 0 , 1 ]$ be the realized empirical fraction of $\mathrm { ^ { 6 6 } y e s ^ { 9 } }$ decisions. The normalized cumulative payof vector has coordinates

$$
R = p _ { T } c _ { X } g ^ { + } + ( 1 - p _ { T } ) c _ { Y } g ^ { - } , \qquad C _ { Y } = ( 1 - p _ { T } ) g ^ { + } , \qquad C _ { N } = p _ { T } g ^ { - } .
$$

The unnormalized cumulative coordinates are exactly $T ( R , C _ { Y } , C _ { N } )$ . Writing $D _ { p _ { T } } = \operatorname* { m a x } \{ C _ { Y } , C _ { N } \}$ gives

$$
p _ { T } \leq \frac { D _ { p _ { T } } } { g ^ { - } } , \qquad 1 - p _ { T } \leq \frac { D _ { p _ { T } } } { g ^ { + } } ,
$$

and hence

$$
\frac { R } { D _ { p _ { T } } } \leq c _ { X } \frac { g ^ { + } } { g ^ { - } } + c _ { Y } \frac { g ^ { - } } { g ^ { + } } = c _ { X } \sqrt { \frac { c _ { Y } } { c _ { X } } } + c _ { Y } \sqrt { \frac { c _ { X } } { c _ { Y } } } = 2 \sqrt { c _ { X } c _ { Y } } .
$$

Moreover, uniformly over every $p _ { T } \in [ 0 , 1 ]$ 2

$$
{ \cal D } _ { p _ { T } } \geq \frac { g ^ { + } g ^ { - } } { g ^ { + } + g ^ { - } } = : d _ { 0 } > 0 .
$$

Consequently, for every $a > 2 { \sqrt { c _ { X } c _ { Y } } }$ , the normalized payof violates at least one defining inequality of $\scriptstyle { \mathcal { S } } _ { a }$ by

$$
a D _ { p _ { T } } - R \geq \left( a - 2 \sqrt { c _ { X } c _ { Y } } \right) d _ { 0 } > 0 .
$$

Its distance from the cone is therefore bounded away from zero uniformly in $T .$ , and the corresponding violation for the cumulative payof grows linearly with T. Thus $ { \boldsymbol { S } } _ { a }$ cannot be approached. □

Corollary 4.9 (Buchbinder–Feldman asymmetry). For every $r > 0$ , setting $\displaystyle ( c _ { X } , c _ { Y } ) = ( 1 / r , r )$ gives $a _ { \mathrm { m a x } } ( 1 / r , r ) = 2$

This specialization is the coeficient-transfer step. With $\left( c _ { X } , c _ { Y } \right) = \left( 1 / r , r \right)$ and $a = 2$ , the weighted Double-Greedy reduction below divides by

$$
a + c _ { X } + c _ { Y } = 2 + r + 1 / r = { \frac { ( 1 + r ) ^ { 2 } } { r } }
$$

and therefore assigns the three coeficients

$$
u _ { r } = { \frac { 2 r } { ( 1 + r ) ^ { 2 } } } , \qquad v _ { r } = { \frac { r ^ { 2 } } { ( 1 + r ) ^ { 2 } } } , \qquad w _ { r } = { \frac { 1 } { ( 1 + r ) ^ { 2 } } }\tag{35}
$$

to the comparator, all-ones, and all-zeros values, respectively. These are the asymmetric ofline box coeficients needed by the 0.401 construction. Theorem 4.18 proves that they hold cumulatively online, and Proposition 4.26 shows algebraically that their composition with the outer certificate has the same value as the ofline coeficient program.

Remark 4.10. The distinction between the preceding proof and the invalid one-shot argument is important. The learner chooses $\pi _ { t }$ before observing $( g _ { t } ^ { + } , g _ { t } ^ { - } )$ . Blackwell’s theorem is what converts the support-function condition into an online strategy; the learner does not choose π as a function of the current adversary point.

Remark 4.11 (What is new here). Because the coeficient frontier that this theorem eventually delivers is inherited from ofline work [14, 10], it is worth being exact about what Theorem 4.8 adds and what it does not.

New: the weighted game itself, in which the two decisions earn marginals at unequal rates $c _ { X } \neq c _ { Y }$ (the symmetric case $c _ { X } = c _ { Y } = 1 / 2$ , where $a _ { \mathrm { m a x } } = 1$ , is the balance subproblem of Roughgarden and Wang [17]); the exact constant $2 \sqrt { c _ { X } c _ { Y } }$ , proved in both directions, so that the theorem also certifies that no larger constant is approachable and hence that the box coeficients of Theorem 4.18 cannot be improved by reweighting (Remark 4.28); the support-function computation that establishes Blackwell’s halfspace condition uniformly over admissible adversary moves, which is what makes the constant survive an adversary who sees the announced mixed action; and the explicit strategy (36), whose per-round cost is a constant-dimensional quadratic program independent of d and m.

Not new: the triple $( u _ { r } , v _ { r } , w _ { r } )$ that the constant $a = 2$ produces at $\left( c _ { X } , c _ { Y } \right) = \left( 1 / r , r \right)$ , which is the ofline unbalanced Double-Greedy frontier of [14, 10]; the Double-Greedy template into which the balance game is inserted, which is that of [17]; and Blackwell approachability itself [15].

The content of the theorem, in one sentence, is that asymmetric weighting costs the balance constant exactly its geometric mean, so that at the normalization $( 1 / r , r )$ used ofline it costs nothing at all.

Explicit asymmetric balancer. For completeness, the Blackwell argument yields an eficient implementation rather than only an existence result. Let

$$
U _ { t } = \left\{ \begin{array} { l l } { ( c _ { X } g _ { t } ^ { + } , 0 , g _ { t } ^ { - } ) , } & { \mathrm { i f ~ \bar { \Psi } ^ \circ \mathrm { y e s } ^ { ; \tau } ~ i s ~ s e l e c t e d } , } \\ { ( c _ { Y } g _ { t } ^ { - } , g _ { t } ^ { + } , 0 ) , } & { \mathrm { i f ~ \bar { \Psi } ^ \circ \mathrm { n o } ^ { ; \tau } ~ i s ~ s e l e c t e d } . } \end{array} \right.
$$

Set $\bar { U } _ { 0 } = \mathbf { 0 } \in \mathbb { R } ^ { 3 }$ and choose any fixed $\pi _ { 1 } \in [ 0 , 1 ]$ . For $t \geq 2 ,$ , let

$$
\bar { U } _ { t - 1 } = \frac { 1 } { t - 1 } \sum _ { j < t } U _ { j }
$$

and let $\Pi _ { S _ { a } } ( \bar { U } _ { t - 1 } )$ be the Euclidean projection onto $ { \boldsymbol { S } } _ { a }$ . If

$$
q _ { t - 1 } = \bar { U } _ { t - 1 } - \Pi _ { S _ { a } } ( \bar { U } _ { t - 1 } ) \ne 0 ,
$$

write

$$
q _ { t - 1 } = \bigl ( - ( \lambda _ { 1 } + \lambda _ { 2 } ) , a \lambda _ { 1 } , a \lambda _ { 2 } \bigr ) , \qquad \lambda _ { 1 } , \lambda _ { 2 } \geq 0 ,
$$

and set $u _ { t - 1 } = \lambda _ { 1 } / ( \lambda _ { 1 } + \lambda _ { 2 } )$ . The probability of selecting $\mathrm { ^ { 6 6 } y e s ^ { 7 9 } }$ is

$$
\pi _ { t } = { \frac { c _ { Y } + a u _ { t - 1 } } { c _ { X } + c _ { Y } + a } } .\tag{36}
$$

When $q _ { t - 1 } = 0$ , choose any fixed $\pi _ { t } \in [ 0 , 1 ]$

The projection onto $ { \boldsymbol { S } } _ { a }$ , and hence the decomposition of its normal into $( \lambda _ { 1 } , \lambda _ { 2 } )$ , is a constantdimensional convex quadratic program. Therefore the balancer runs in time polynomial in the input size, with a per-round cost independent of d and m.

Proposition 4.12 (Eficient weighted balancer). If $a \leq 2 { \sqrt { c _ { X } c _ { Y } } }$ , the strategy (36) approaches $ { \boldsymbol { S } } _ { a }$ with

$$
\operatorname { d i s t } \left( { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } U _ { t } , { \mathcal { S } } _ { a } \right) = O \left( { \frac { M } { \sqrt { T } } } \right)
$$

in expectation and therefore has weighted balance regret $O ( M { \sqrt { T } } )$ . The hidden constant depends only on $a , c _ { X } , c _ { Y }$

Proof. By Moreau’s decomposition for closed convex cones, the residual $q _ { t - 1 }$ belongs to the polar cone of $\scriptstyle { \mathcal { S } } _ { a }$ and is a separating normal for the target at the current average payof. The calculation in the proof of Theorem 4.8 shows that (36) makes the conditional expected increment have nonpositive inner product with this normal for every admissible adversary move. Blackwell’s approachability theorem then gives the standard ${ \cal O } ( T ^ { - 1 / 2 } )$ approachability rate because the payof set is bounded. Since membership in $\scriptstyle { S _ { a } }$ is equivalent to $R \geq a C _ { Y }$ and $R \geq a C _ { N }$ , distance ${ \cal O } ( T ^ { - 1 / 2 } )$ from the cone is equivalent to weighted balance regret $O ( M { \sqrt { T } } )$ after multiplying by T. □

Balancing from noisy marginals. The balancer above is stated with exact feedback, but under (7) a learner never sees its marginal pair; it sees an estimate assembled from oracle calls issued after the whole round’s decisions have been committed. The next lemma is the form in which every inner guarantee in this paper is actually used. It is deliberately stated with a generic scale M<sup>¯</sup> and a generic post-decision filtration, so that the same statement covers the direct algorithm of Section 4.3 (where $\bar { M } = { \cal O } ( M _ { \sigma } )$ and each of the two estimates is a diference of two oracle calls), the batched algorithm of Appendix B, and the bandit algorithm of Section 5 (where $\bar { M } = { \cal O } ( M _ { \sigma } )$ and the estimate additionally averages over a random injection).

Lemma 4.13 (Weighted balance with unbiased estimated marginals). Fix $c _ { X } , c _ { Y } > 0 , a \le 2 \sqrt { c _ { X } c _ { Y } }$ and $\bar { M } > 0$ . Suppose the true pair $( g _ { t } ^ { + } , g _ { t } ^ { - } )$ belongs to $[ - \bar { M } , \bar { M } ] ^ { 2 }$ , satisfies $g _ { t } ^ { + } + g _ { t } ^ { - } \ge 0 ,$ and is measurable with respect to a sigma-field $\mathcal { F } _ { t } ^ { - }$ <sup>−</sup> containing the past and the announced mixed action but not the fresh binary draw $A _ { t }$ . Draw $A _ { t }$ conditionally independently from that mixed action, and set $\mathcal { F } _ { t } ^ { + } = \mathcal { F } _ { t } ^ { - } \vee \sigma ( A _ { t } )$ . Suppose the learner then observes estimates $( \widehat { g } _ { t } ^ { + } , \widehat { g } _ { t } ^ { - } ) \in [ - \bar { M } , \bar { M } ] ^ { 2 }$ satisfying

$$
\mathbb { E } [ ( \widehat { g } _ { t } ^ { + } , \widehat { g } _ { t } ^ { - } ) \mid \mathcal { F } _ { t } ^ { + } ] = ( g _ { t } ^ { + } , g _ { t } ^ { - } ) .
$$

If the Blackwell strategy (36) uses the estimated payof vectors in its running average, then its expected weighted balance regret (29), evaluated at the true realized payofs, is $O ( \bar { M } \sqrt { T } )$

Proof. Let $U _ { t }$ and $\widehat { U } _ { t }$ denote the true and estimated realized payof vectors. At the start of a round, the projection residual and the mixed action are measurable with respect to $\mathcal { F } _ { t } ^ { - }$ . Conditioning first on $\mathcal { F } _ { t } ^ { + }$ and then on $\mathcal { F } _ { t } ^ { - }$ gives

$$
\mathbb { E } [ \widehat { U } _ { t } \mid \mathcal { F } _ { t } ^ { - } ] = \overline { { U } } ( \pi _ { t } ; g _ { t } ^ { + } , g _ { t } ^ { - } ) ,
$$

where the right side is the mixed payof from Theorem 4.8. Its support-function calculation is uniform over all admissible pairs and makes the inner product of this conditional mean with the projection residual nonpositive. Hence the usual squared-distance recursion gives

$$
\mathbb { E } \operatorname { d i s t } \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \widehat { U } _ { t } , \mathcal { S } _ { a } \right) = O \left( \frac { \bar { M } } { \sqrt { T } } \right) .
$$

Moreover $\mathbb { E } [ \widehat { U } _ { t } - U _ { t } \ | \ \mathcal { F } _ { t } ^ { + } ] = 0$ , so the estimation errors form a bounded martingale-diference sequence under the post-observation filtration, and

$$
\mathbb { E } \left. \sum _ { t = 1 } ^ { T } ( \widehat { U } _ { t } - U _ { t } ) \right. _ { 2 } = O ( \bar { M } \sqrt { T } ) .
$$

Distance to a closed set is one-Lipschitz, so transferring the bound from the estimated cumulative payof to the true cumulative payof and multiplying by T proves the claim. □

Two features of this statement matter later. The learner’s own decision may enter the conditioning, so the estimate is allowed to be observed only after the draw, and indeed after all of the round’s draws. The estimates need not themselves be admissible: the requirement $\widehat { g _ { t } } ^ { + } + \widehat { g _ { t } } ^ { - } \ge 0$ is never imposed, because admissibility is a property of the true marginals of a submodular function, and it is only the true marginals that the reduction of Proposition 4.17 manipulates.

Continuous box reduction. Fix $x \in K$ and define $G ( a ) = f ( x \odot a )$

Lemma 4.14 (Box transformation). If f is nonnegative and DR-submodular, then $G$ is nonnegative and DR-submodular on $[ 0 , 1 ] ^ { d }$ . Moreover,

$$
G ( o ) = f ( x \odot o ) , \qquad G ( { \bf 1 } ) = f ( x ) , \qquad G ( { \bf 0 } ) = f ( { \bf 0 } ) ,
$$

and $\nabla G ( \boldsymbol { a } ) = \boldsymbol { x } \odot \nabla f ( \boldsymbol { x } \odot \boldsymbol { a } ) , \ s o \ \| \nabla G ( \boldsymbol { a } ) \| _ { 2 } \le \| \nabla f ( \boldsymbol { x } \odot \boldsymbol { a } ) \| _ { 2 } .$

Proof. Nonnegativity is immediate; the statement is the special case $y = x$ of Lemma D.2, which we verify directly here. For $a \leq b$ in $[ 0 , 1 ] ^ { d }$ , an increment $\delta e _ { i }$ changes the argument of f by $\delta x _ { i } e _ { i } \geq 0$ ， and $x \odot a \le x \odot b ,$ so DR-submodularity of f gives

$$
G ( a + \delta c _ { i } ) - G ( a ) = f ( x \odot a + \delta x _ { i } c _ { i } ) - f ( x \odot a ) \ge f ( x \odot b + \delta x _ { i } c _ { i } ) - f ( x \odot b ) = G ( b + \delta c _ { i } ) - G ( b ) ,
$$

which is the diminishing-returns inequality for G. The three displayed identities and the gradient formula follow by substitution and the chain rule, and $x \leq \mathbf { 1 }$ gives the norm bound. □

Count lifting. Fix $m \geq 1$ and let $V _ { m } = [ d ] \times [ m ]$ . For $S \subseteq V _ { m } ,$ , define

$$
\phi _ { m } ( S ) _ { i } = \frac { 1 } { m } \left| \{ j \in [ m ] : ( i , j ) \in S \} \right| ,\tag{37}
$$

and define the lifted set function ${ \widetilde { G } } _ { m } ( S ) = G ( \phi _ { m } ( S ) )$ . Define the grid

$$
\mathcal { G } _ { m } = \left\{ 0 , \frac { 1 } { m } , \dots , 1 \right\} ^ { d } .
$$

Lemma 4.15 (Count lifting is submodular). If G is nonnegative and DR-submodular, then $\widetilde { G } _ { m }$ is a nonnegative submodular set function on $V _ { m }$ . Furthermore, $\phi _ { m }$ maps $2 ^ { V _ { m } }$ onto $\mathcal { G } _ { m }$ , and

$$
\operatorname* { m a x } _ { S \subseteq V _ { m } } \widetilde { G } _ { m } ( S ) = \operatorname* { m a x } _ { a \in \mathcal { G } _ { m } } G ( a ) .
$$

Proof. Let $S \subseteq S ^ { \prime }$ and let $e = ( i , j ) \not \in S ^ { \prime }$ . Adding e increases coordinate i of both $\phi _ { m } ( S )$ and $\phi _ { m } ( S ^ { \prime } )$ by exactly $1 / m$ and leaves all other coordinates unchanged. Since $\phi _ { m } ( S ) \leq \phi _ { m } ( S ^ { \prime } )$ , and since both segments remain in the unit cube, antitonicity of the gradient gives, for every $\vartheta \in [ 0 , 1 / m ]$ 2

$$
\partial _ { i } G \bigl ( \phi _ { m } ( S ) + \vartheta e _ { i } \bigr ) \geq \partial _ { i } G \bigl ( \phi _ { m } ( S ^ { \prime } ) + \vartheta e _ { i } \bigr ) .
$$

Integrating over $\vartheta _ { \textrm { J } }$ ields

$$
\widetilde { G } _ { m } ( S \cup \{ e \} ) - \widetilde { G } _ { m } ( S ) \geq \widetilde { G } _ { m } ( S ^ { \prime } \cup \{ e \} ) - \widetilde { G } _ { m } ( S ^ { \prime } ) .
$$

This is the diminishing-marginal characterization of submodularity, so $\widetilde { G } _ { m }$ is submodular. Every subset has an integer count between 0 and m in each coordinate block and therefore maps to $\mathcal { G } _ { m }$ Conversely, every grid point $a \in \mathcal { G } _ { m }$ is realized by selecting exactly $m a _ { i }$ elements from the ith block. Thus $\phi _ { m } ( 2 ^ { V _ { m } } ) = \mathcal { G } _ { m }$ , and the two maxima coincide. □

The count lifting is not interchangeable with the more obvious max-level lifting $\phi _ { m } ^ { \operatorname* { m a x } } ( S ) _ { i } =$ $m ^ { - 1 } \operatorname* { m a x } \{ j : ( i , j ) \in S \}$ , with the maximum defined as zero when the set is empty, which fails to be submodular for non-monotone G. A minimal counterexample has $d = 1 , m = 2$ and $G ( t ) = 1 - t .$ which is nonnegative and concave, hence DR-submodular: with $S = \emptyset$ and $S ^ { \prime } = \{ ( 1 , 2 ) \}$ , adding $( 1 , 1 )$ has marginal $G ( 1 / 2 ) - G ( 0 ) = - 1 / 2$ at S but $G ( 1 ) - G ( 1 ) = 0$ at $S ^ { \prime } .$ , violating diminishing returns. The mechanism of Lemma 4.15 is precisely that the count lifting makes the two marginals increments of the same size $1 / m$ in the same coordinate at comparable points, which is exactly the DR hypothesis; under the max-level lifting the increments have diferent sizes and the argument collapses.

Lemma 4.16 (Grid approximation). Suppose $\| \nabla G ( \boldsymbol { a } ) \| _ { 2 } \le B _ { G }$ throughout the box. For every $a \in [ 0 , 1 ] ^ { d }$ the downward rounding $\bar { a } _ { i } = \lfloor m a _ { i } \rfloor / m$ satisfies $\bar { a } \in \mathcal { G } _ { m } , \bar { a } \le a$ , and

$$
G ( \bar { a } ) \geq G ( a ) - \frac { B _ { G } \sqrt { d } } { m } .\tag{38}
$$

Proof. Rounding down keeps each coordinate in [0, 1] and gives $\Vert a - \bar { a } \Vert _ { 2 } \leq \sqrt { d } / m$ , so the mean value theorem and the gradient bound give (38). Downward rounding is not needed for the Lipschitz estimate itself; it is a canonical choice that also preserves $\bar { a } \leq a .$ , while the grid representation follows from Lemma 4.15. □

Together, Lemmas 4.15 and 4.16 give the complete count-lifting reduction used below. For a fixed continuous comparator $o \in [ 0 , 1 ] ^ { d }$ , set $\bar { o } _ { i } = \lfloor m o _ { i } \rfloor / m$ and choose, in the ith block of $V _ { m }$ , any $m { \bar { o } } _ { i }$ copies. The resulting fixed set $S _ { o }$ satisfies $\phi _ { m } ( S _ { o } ) = \bar { o }$ and hence

$$
\widetilde { G } _ { m } ( S _ { o } ) = G ( \bar { o } ) \geq G ( o ) - \frac { B _ { G } \sqrt { d } } { m } .
$$

Thus the lifting represents every grid point exactly, preserves submodularity, and loses at most $B _ { G } \sqrt { d } / m$ per round when the continuous comparator is rounded to the grid.

Online Double-Greedy on the lifted box. We now give a weighted extension of the online Double-Greedy reduction of Roughgarden and Wang [17]. The sequential sets $X _ { i } ^ { t } , Y _ { i } ^ { t }$ follow their template; the change is the asymmetric balance game and the retention of both endpoint terms.

Let $V$ be a finite ground set with $| V | = n$ , and let

$$
\widetilde { G } _ { 1 } , \dots , \widetilde { G } _ { T } : 2 ^ { V } \to [ 0 , M ]
$$

be a nonanticipating sequence of nonnegative submodular functions. Namely, conditional on the pre-round history (and any current outer state), $\widetilde { G } _ { t }$ is fixed and independent of all fresh round-t binary draws. The value feedback used for the update is supplied only after the final set is committed, and it is noisy: each value is a separate call to the oracle (7). This condition does not make $\widetilde { G } _ { t }$ known to the learners before their decisions. Relabel the elements so that $V = [ n ]$ , and fix a comparator $S _ { o } \subseteq V$

For the within-round information pattern, let $\mathcal { F } _ { t , 0 }$ be the analysis sigma-field generated by the past history together with the current function $\widetilde { G } _ { t }$ , which is fixed before any current binary draw. After the first i binary decisions, let

$$
\mathcal { F } _ { t , i } = \mathcal { F } _ { t , 0 } \vee \sigma ( B _ { 1 } ^ { t } , \ldots , B _ { i } ^ { t } ) ,
$$

where $B _ { i } ^ { t } \in \{ \mathrm { y e s } , \mathrm { n o } \}$ is learner i’s current draw. The sigma-field $\mathcal { F } _ { t , 0 }$ is used only for analysis: the learners do not observe $\widetilde { G } _ { t }$ when choosing their current probabilities. Learner i computes its mixed action from its own past observations, so its probability is $\mathcal { F } _ { t , i - 1 }$ -measurable, and its fresh draw $B _ { i } ^ { t }$ is conditionally independent of $\mathcal { F } _ { t , i - 1 }$ given that probability.

Run one weighted balance learner $\mathcal { L } _ { i }$ for every $i \in V$ . In round $t ,$ the balance learners are queried sequentially:

$$
X _ { 0 } ^ { t } = \varnothing , \qquad Y _ { 0 } ^ { t } = V ,
$$

and

$$
\begin{array} { c c c c } { { \mathrm { y e s } } } & { {  } } & { { X _ { i } ^ { t } } } & { { Y _ { i } ^ { t } } } \\ { { \mathrm { n o } } } & { {  } } & { { X _ { i - 1 } ^ { t } \cup \{ i \} } } & { { Y _ { i - 1 } ^ { t } } } \\ { { \mathrm { n o } } } & { {  } } & { { X _ { i - 1 } ^ { t } } } & { { Y _ { i - 1 } ^ { t } \setminus \{ i \} . } } \end{array}
$$

At the end, $X _ { n } ^ { t } = Y _ { n } ^ { t } = : \mathrm { A L G } ^ { t }$ . The set is committed before any value feedback from $\widetilde { G } _ { t }$ is supplied. Afterward, define the two marginals

$$
g _ { i } ^ { + , t } = \widetilde { G } _ { t } ( X _ { i - 1 } ^ { t } \cup \{ i \} ) - \widetilde { G } _ { t } ( X _ { i - 1 } ^ { t } ) , \qquad g _ { i } ^ { - , t } = \widetilde { G } _ { t } ( Y _ { i - 1 } ^ { t } \setminus \{ i \} ) - \widetilde { G } _ { t } ( Y _ { i - 1 } ^ { t } ) .\tag{39}
$$

Because $X _ { i - 1 } ^ { t }$ and $Y _ { i - 1 } ^ { t }$ depend only on $B _ { 1 } ^ { t } , \ldots , B _ { i - 1 } ^ { t }$ , the pair $( g _ { i } ^ { + , t } , g _ { i } ^ { - , t } )$ is $\mathcal { F } _ { t , i - 1 }$ -measurable and is fixed before the fresh draw $B _ { i } ^ { t }$ . Thus each balance learner faces an admissible nonanticipating adversarial move even though its pair may depend on the earlier learners’ current decisions.

These marginals are not observed. After the round’s set is committed, the learner issues four fresh oracle calls per element and forms

$$
\widehat { g } _ { i } ^ { + , t } = \widehat { \widetilde { G } } _ { t } ( X _ { i - 1 } ^ { t } \cup \{ i \} ) - \widehat { \widetilde { G } } _ { t } ( X _ { i - 1 } ^ { t } ) , \qquad \widehat { g } _ { i } ^ { - , t } = \widehat { \widetilde { G } } _ { t } ( Y _ { i - 1 } ^ { t } \setminus \{ i \} ) - \widehat { \widetilde { G } } _ { t } ( Y _ { i - 1 } ^ { t } ) ,\tag{40}
$$

where each hatted symbol denotes the return value of its own call. By (7) these estimates are conditionally unbiased for $( g _ { i } ^ { + , t } , g _ { i } ^ { - , t } )$ given everything decided in the round, and they lie in $[ - 2 M _ { \sigma } , 2 M _ { \sigma } ]$ . Learner $\mathcal { L } _ { i }$ updates on (40), and Lemma 4.13 applies to it with $\bar { M } = 2 M _ { \sigma } ;$ ; the sets $X _ { i } ^ { t } , Y _ { i } ^ { t }$ themselves are maintained combinatorially and are never inferred from noisy values. Since every occurrence is a separate call, the chain values cannot be cached across successive elements, and the round costs 4n calls rather than $2 n + 2$ . For learner $\mathcal { L } _ { i } .$ let $C _ { Y , i } ^ { t }$ and $C _ { N , i } ^ { t }$ denote the two missed-opportunity components generated from $( g _ { i } ^ { + , t } , g _ { i } ^ { - , t } )$ as in the weighted balance game.

Proposition 4.17 (Weighted online Double-Greedy reduction). Suppose each balance learner has weighted balance regret at most $g ( T )$ at parameter a in the sense of (29), where the regret is evaluated at the true realized payofs even if the learner updates on the estimates (40). Then

$$
\begin{array} { r l r } {  { ( a + c _ { X } + c _ { Y } ) \mathbb { E } [ \sum _ { t = 1 } ^ { T } \widetilde { G } _ { t } ( \mathrm { A L G } ^ { t } ) ] } } \\ & { } & { \geq a \mathbb { E } [ \sum _ { t = 1 } ^ { T } \widetilde { G } _ { t } ( S _ { o } ) ] + c _ { X } \mathbb { E } [ \sum _ { t = 1 } ^ { T } \widetilde { G } _ { t } ( \varpi ) ] + c _ { Y } \mathbb { E } [ \sum _ { t = 1 } ^ { T } \widetilde { G } _ { t } ( V ) ] - n g ( T ) . } \end{array}\tag{41}
$$

Proof. Since $X _ { i - 1 } ^ { t } \subseteq Y _ { i - 1 } ^ { t } \setminus \{ i \}$ , submodularity gives $g _ { i } ^ { + , t } + g _ { i } ^ { - , t } \ge 0$ , so the balance input is admissible; boundedness of $\widetilde { G } _ { t }$ by M places both marginals in $[ - M , M ]$ . By the filtration defined above, this admissible pair is fixed conditional on $\mathcal { F } _ { t , i - 1 }$ before learner i draws $B _ { i } ^ { t }$ . The weighted balance guarantee therefore applies to every learner separately despite the acyclic within-round dependence on earlier learners. Everything below is written in terms of the true marginals and the true realized payofs $R _ { i } ^ { t } , C _ { Y , i } ^ { t } , C _ { N , i } ^ { t } ;$ the estimates (40) enter only through the hypothesis on $g ( T )$ which Lemma 4.13 supplies.

Define the weighted potential

$$
\Psi _ { i } ^ { t } = c _ { X } \widetilde { G } _ { t } ( X _ { i } ^ { t } ) + c _ { Y } \widetilde { G } _ { t } ( Y _ { i } ^ { t } ) .
$$

If $\mathcal { L } _ { i }$ answers yes, $\Psi _ { i } ^ { t } - \Psi _ { i - 1 } ^ { t } = c _ { X } g _ { i } ^ { + , t }$ ; if it answers no, $\Psi _ { i } ^ { t } - \Psi _ { i - 1 } ^ { t } = c _ { Y } g _ { i } ^ { - , t }$ . Thus the potential increase is exactly the weighted balance reward $R _ { i } ^ { t }$

Next define $\mathrm { O P T } _ { 0 } ^ { t } = S _ { o }$ and let $\mathrm { O P T } _ { i } ^ { t }$ agree with the algorithm’s first i decisions and with $S _ { o }$ on the remaining elements. There are two wrong-decision cases. If $i \in S _ { o }$ and the algorithm says no, set $A = \mathrm { O P T } _ { i } ^ { t } = \mathrm { O P T } _ { i - 1 } ^ { t } \setminus \{ i \}$ . Since $X _ { i - 1 } ^ { t } \subseteq A$ , submodularity gives the explicit comparison

$$
\begin{array} { r l } & { \widetilde { G } _ { t } ( \mathrm { O P T } _ { i - 1 } ^ { t } ) - \widetilde { G } _ { t } ( \mathrm { O P T } _ { i } ^ { t } ) = \widetilde { G } _ { t } ( A \cup \{ i \} ) - \widetilde { G } _ { t } ( A ) } \\ & { \qquad \leq \widetilde { G } _ { t } ( X _ { i - 1 } ^ { t } \cup \{ i \} ) - \widetilde { G } _ { t } ( X _ { i - 1 } ^ { t } ) = g _ { i } ^ { + , t } . } \end{array}
$$

This is exactly the increment of $C _ { Y , i } ^ { t } . \mathrm { ~ H ~ } i \notin S _ { o }$ and the algorithm says yes, set $A = \mathrm { O P T } _ { i - 1 } ^ { t }$ , so that $\mathrm { O P T } _ { i } ^ { t } = A \cup \{ i \}$ and $A \subseteq Y _ { i - 1 } ^ { t } \setminus \{ i \}$ . Submodularity now gives

$$
\begin{array} { r l } & { \widetilde { G } _ { t } ( \mathrm { O P T } _ { i - 1 } ^ { t } ) - \widetilde { G } _ { t } ( \mathrm { O P T } _ { i } ^ { t } ) = - \left[ \widetilde { G } _ { t } ( A \cup \{ i \} ) - \widetilde { G } _ { t } ( A ) \right] } \\ & { \qquad \leq - \left[ \widetilde { G } _ { t } ( Y _ { i - 1 } ^ { t } ) - \widetilde { G } _ { t } ( Y _ { i - 1 } ^ { t } \setminus \{ i \} ) \right] = g _ { i } ^ { - , t } . } \end{array}
$$

This inequality remains valid when both sides are negative, and its right-hand side is exactly the increment of $C _ { N , i } ^ { t }$ . In all other cases the hybrid value does not change.

The two cases cannot both occur for the same element, because the comparator $S _ { o }$ is the same in every round. This matters: the marginals of a non-monotone $\widetilde { G } _ { t }$ may be negative, so discarding rounds is not free. If $i \in S _ { o }$ , the hybrid value changes only on rounds where $\mathcal { L } _ { i }$ answers no, and on exactly those rounds $C _ { Y , i } ^ { t } = g _ { i } ^ { + , \dot { t } }$ while $C _ { Y , i } ^ { t } = 0$ on the remaining rounds; hence the round-wise bounds sum to $\textstyle \sum _ { t } C _ { Y , i } ^ { t }$ with no discarded terms. Symmetrically, if $i \notin S _ { o }$ they sum to $\textstyle \sum _ { t } C _ { N , i } ^ { t }$ Therefore

$$
\sum _ { t } \left[ \widetilde { G } _ { t } ( \mathrm { O P T } _ { i - 1 } ^ { t } ) - \widetilde { G } _ { t } ( \mathrm { O P T } _ { i } ^ { t } ) \right] \leq \operatorname* { m a x } \left\{ \sum _ { t } C _ { Y , i } ^ { t } , \sum _ { t } C _ { N , i } ^ { t } \right\} .
$$

Summing this inequality over the learners makes the hybrid sequence explicitly telescope:

$$
a \sum _ { t } \left[ \widetilde { G } _ { t } ( S _ { o } ) - \widetilde { G } _ { t } ( \mathrm { A L G } ^ { t } ) \right] \leq a \sum _ { i = 1 } ^ { n } \operatorname* { m a x } \left\{ \sum _ { t } C _ { Y , i } ^ { t } , \sum _ { t } C _ { N , i } ^ { t } \right\} .\tag{42}
$$

Since $\begin{array} { r } { \sum _ { t } R _ { i } ^ { t } = \sum _ { t } ( \Psi _ { i } ^ { t } - \Psi _ { i - 1 } ^ { t } ) } \end{array}$ , subtracting these rewards from both sides of (42) gives

$$
\begin{array} { r l } & { a \displaystyle \sum _ { t } \Big [ \widetilde { G } _ { t } ( S _ { o } ) - \widetilde { G } _ { t } ( \mathrm { A L G } ^ { t } ) \Big ] - \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { t } R _ { i } ^ { t } } \\ & { \qquad \le \displaystyle \sum _ { i = 1 } ^ { n } \left[ a \operatorname* { m a x } \left\{ \sum _ { t } C _ { Y , i } ^ { t } , \displaystyle \sum _ { t } C _ { N , i } ^ { t } \right\} - \sum _ { t } R _ { i } ^ { t } \right] . } \end{array}
$$

Taking expectations and applying (29) to every $\mathcal { L } _ { i }$ bounds the right-hand side by $n g ( T )$ . Telescoping the potentials therefore yields

$$
\mathbb { E } \left[ a \sum _ { t } \bigl ( \widetilde { G } _ { t } ( S _ { o } ) - \widetilde { G } _ { t } ( \mathrm { A L G } ^ { t } ) \bigr ) - \sum _ { t } \bigl ( \Psi _ { n } ^ { t } - \Psi _ { 0 } ^ { t } \bigr ) \right] \leq n g ( T ) .
$$

Finally, $\Psi _ { 0 } ^ { t } = c _ { X } \widetilde { G } _ { t } ( \varpi ) + c _ { Y } \widetilde { G } _ { t } ( V )$ and $\Psi _ { n } ^ { t } = ( c _ { X } + c _ { Y } ) \widetilde { G } _ { t } ( \mathrm { A L G } ^ { t } )$ . Rearranging gives (41). □

The coeficient triple in the next theorem is the asymmetric ofline Box-Maximization frontier used in [14, 10]. The theorem’s content is that the same triple is achievable cumulatively online, with only additive balance and grid-discretization losses.

Theorem 4.18 (Weighted online USM over the box). At round t, let $\mathcal { F } _ { t } ^ { - }$ <sup>−</sup> be the analysis pre-round sigma-field; in the main composition it contains the obliviously fixed reward sequence and the current outer state $x _ { t }$ , but not the inner learner’s fresh round-t randomization. Let $G _ { t } : [ 0 , 1 ] ^ { d } \to [ 0 , M ]$ be a diferentiable DR-submodular function that is $\mathcal { F } _ { t } ^ { - } \cdot m e a s u r a b l e .$ , and assume the fresh inner randomization is conditionally independent of $\mathcal { F } _ { t } ^ { - }$ . This measurability is for analysis and does not make $G _ { t }$ known to the learner: the value feedback used for its update is supplied only after $a _ { t }$ is committed, and it consists of calls to the noisy oracle (7) at level $\sigma .$ Suppose $\Vert \nabla G _ { t } ( a ) \Vert _ { 2 } \leq B _ { G }$ throughout the box. For every integer m $\geq 1$ and every fixed $r > 0$ , there is a polynomial-time online algorithm producing $a _ { t } \in [ 0 , 1 ] ^ { d }$ such that, for every fixed comparator $o \in [ 0 , 1 ] ^ { d }$ 2

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } G _ { t } ( a _ { t } ) \right] \geq \frac { 2 r } { ( 1 + r ) ^ { 2 } } \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } G _ { t } ( o ) \right] } \\ & { \quad \quad \quad + \frac { r ^ { 2 } } { ( 1 + r ) ^ { 2 } } \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } G _ { t } ( \mathbf { 1 } ) \right] + \frac { 1 } { ( 1 + r ) ^ { 2 } } \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } G _ { t } ( \mathbf { 0 } ) \right] } \\ & { \quad \quad \quad - O ( d m M _ { \sigma } \sqrt { T } ) - O \left( \frac { B _ { G } \sqrt { d } T } { m } \right) . } \end{array}\tag{43}
$$

The lifted Double-Greedy update uses 4dm calls to the noisy oracle for the current function $G _ { t }$ per round; its remaining operations are combinatorial or constant-dimensional convex computations. At $\sigma = 0$ the bound is the exact-feedback guarantee, with $M _ { \sigma } = M$

Proof. Lift each $G _ { t }$ to ${ \widetilde { G } } _ { t , m } ( S ) = G _ { t } ( \phi _ { m } ( S ) )$ . By Lemma 4.15, this is a nonnegative submodular set function on dm elements taking values in [0, M]. Let $\bar { o } _ { i } = \lfloor m o _ { i } \rfloor / m$ and choose a fixed set $S _ { o } \subseteq V _ { m }$ with $\phi _ { m } ( S _ { o } ) = \bar { o } ,$ which exists by Lemma 4.15.

Choose $\displaystyle \left( c _ { X } , c _ { Y } \right) = \left( 1 / r , r \right)$ . By Corollary 4.9, $a = 2$ is the largest feasible balance constant. Every learner runs the strategy (36) on the estimated marginals (40), which are conditionally unbiased and lie in $[ - 2 M _ { \sigma } , 2 M _ { \sigma } ]$ , so Lemma 4.13 with $\bar { M } = 2 M _ { \sigma }$ gives $g ( T ) = O ( M _ { \sigma } \sqrt { T } )$ for every balance learner; when $\sigma = 0$ this is Proposition 4.12 itself. Dividing (41) by $a + c _ { X } + c _ { Y } = 2 + r + 1 / r = ( 1 + r ) ^ { 2 } / r$ gives the coeficients

$$
{ \frac { a } { a + c _ { X } + c _ { Y } } } = { \frac { 2 r } { ( 1 + r ) ^ { 2 } } } , \qquad { \frac { c _ { Y } } { a + c _ { X } + c _ { Y } } } = { \frac { r ^ { 2 } } { ( 1 + r ) ^ { 2 } } } , \qquad { \frac { c _ { X } } { a + c _ { X } + c _ { Y } } } = { \frac { 1 } { ( 1 + r ) ^ { 2 } } } ,
$$

on $\widetilde { G } _ { t , m } ( S _ { o } ) = G _ { t } ( \bar { o } ) , \widetilde { G } _ { t , m } ( V _ { m } ) = G _ { t } ( { \bf { 1 } } )$ and $\widetilde { G } _ { t , m } ( \varnothing ) = G _ { t } ( \mathbf { 0 } )$ respectively. The total number of balance learners is dm, so the accumulated balance regret is $O ( d m M _ { \sigma } \sqrt { T } )$ . Finally, Lemma 4.16 replaces the grid comparator o¯ by the continuous comparator o at a cost of at most $B _ { G } \sqrt { d } / m$ per round, that is $O ( B _ { G } \sqrt { d } T / m )$ in total. □

To see directly how Theorem 4.18 produces the headline factor, apply it to $G _ { t } ( a ) = f _ { t } ( x _ { t } \odot a )$ The coeficients $u _ { r }$ and $v _ { r }$ in (35) multiply, respectively, the cumulative quantities $f _ { t } ( x _ { t } \odot o )$ and $f _ { t } ( x _ { t } )$ . The outer certificate (27) contains these same quantities with debts $\chi _ { s }$ and $2 \zeta _ { s }$ . Mixing the inner and outer actions with probability λ pays both debts whenever

$$
( 1 - \lambda ) u _ { r } \geq \lambda \chi _ { s } , \qquad ( 1 - \lambda ) v _ { r } \geq 2 \lambda \zeta _ { s } .
$$

The largest admissible mixture coeficient is therefore

$$
\lambda ( s , r ) = \operatorname* { m i n } \left\{ \frac { u _ { r } } { u _ { r } + \chi _ { s } } , \frac { v _ { r } } { v _ { r } + 2 \zeta _ { s } } \right\} ,
$$

and the comparator coeficient is $\theta _ { s } \lambda ( s , r )$ . At the certified parameters of Proposition 4.24, this coeficient exceeds 0.401. Thus the route from the weighted decisions to the approximation factor is explicit: $a _ { \mathrm { m a x } } = 2 \sqrt { c _ { X } c _ { Y } }$ , then $a = 2$ , then $\left( u _ { r } , v _ { r } , w _ { r } \right)$ , and finally $\theta _ { s } \lambda ( s , r ) > 0 . 4 0 1$

Corollary 4.19 (Optimized inner rate). Taking $m = \lceil T ^ { 1 / 4 } \rceil$ gives

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } G _ { t } ( a _ { t } ) \geq \frac { 2 r } { ( 1 + r ) ^ { 2 } } \mathbb { E } \displaystyle \sum _ { t } G _ { t } ( o ) + \frac { r ^ { 2 } } { ( 1 + r ) ^ { 2 } } \mathbb { E } \displaystyle \sum _ { t } G _ { t } ( \mathbf { 1 } ) + \frac { 1 } { ( 1 + r ) ^ { 2 } } \mathbb { E } \displaystyle \sum _ { t } G _ { t } ( \mathbf { 0 } ) } \\ & { \qquad - O \Big ( ( d M _ { \sigma } + B _ { G } \sqrt { d } ) T ^ { 3 / 4 } \Big ) . } \end{array}\tag{44}
$$

## 4.3 Algorithm and factor-revealing composition

Algorithm 1 Online 0.401 composition with noisy post-decision value-oracle feedback   
Require: delay s with $\zeta _ { s } , \chi _ { s } \geq 0 ,$ , asymmetry $r ,$ mixture probability $\lambda ,$ grid size $m _ { : }$ , finite-diference   
accuracy ε, noise level $\sigma \geq 0 .$   
1: Choose any $x _ { 1 } \in K$ and initialize dm balance learners with weights $\left( c _ { X } , c _ { Y } \right) = \left( 1 / r , r \right)$ and   
constant $a = 2$   
2: Set $h = \varepsilon / 2 , Q = \lceil \varepsilon ^ { - 1 } \rceil$ and $\Gamma _ { s } = ( 1 + \zeta _ { s } ) B .$   
3: Set $\eta = D / \sqrt { T \overline { { V } } }$ if $\overline { { V } } > 0 .$ , and $\eta = 0$ otherwise, where $\overline { V }$ is any computable upper bound on   
the second-moment proxy $V _ { h , Q }$ of (60).   
4: for $t = 1 , \dots , T$ do   
5: Each balance learner selects its binary decision using only its own history; collect the resulting   
lifted set $S _ { t }$ and set $a _ { t } = \phi _ { m } ( S _ { t } )$   
6: Set $y _ { t } = x _ { t } \odot a _ { t }$ and $z _ { t } = Y _ { 1 } ( x _ { t } )$   
7: Draw $Z _ { t } \sim \mathrm { B e r n o u l l i } ( \lambda )$ independently.   
8: Play   
$p _ { t } = { \left\{ \begin{array} { l l } { z _ { t } , } & { Z _ { t } = 1 , } \\ { y _ { t } , } & { Z _ { t } = 0 . } \end{array} \right. }$   
9: Obtain post-decision access to the noisy oracle for $f _ { t } .$   
10: With $G _ { t } ( a ) : = f _ { t } ( x _ { t } \odot a )$ , issue four fresh oracle calls per lifted element and form the   
estimated marginals (40).   
11: Estimate $\widetilde { q } _ { s } ( f _ { t } , x _ { t } )$ by (58): one fresh diference pair per coordinate at each of the Q midpoint   
nodes, and $Q$ fresh diference pairs per coordinate at $x _ { t } ,$ all with step $h .$ Do not clip the result;   
call it $\widehat { q _ { t } }$   
12: Update all balance learners using the estimated marginals.   
13: Update the outer linear-optimization state: $x _ { t + 1 } = \Pi _ { K } \big ( x _ { t } + \eta \widehat { q _ { t } } \big )$   
14: end for

Remark 4.20 (Why the field estimate is not clipped). With exact values one would project $\widehat { q _ { t } }$ onto the ball of radius $\Gamma _ { s }$ , which cannot increase the distance to the true field. Under (7) the same projection is a nonlinear map applied to a random vector and would introduce a bias that is not controlled by σ alone, so we omit it and pay instead through the second moment $V _ { h , Q }$ in the step size. The iterate is of course still projected onto K after every update. When $\sigma = 0$ the estimate is deterministic and bounded by $\Gamma _ { s } + \Delta _ { h , Q }$ , so $\sqrt { V _ { h , Q } } = O ( \Gamma _ { s } + \Delta _ { h , Q } )$ and the step size agrees with the deterministic $\eta = D / ( \Gamma _ { s } \sqrt { T } )$ up to the factor $1 + \Delta _ { h , Q } / \Gamma _ { s } ;$ the extra $D \Delta _ { h , Q } \sqrt { T }$ this can cost is dominated by the field-bias term $D \Delta _ { h , Q } T$ that is present in any case. Throughout the analysis we write $V _ { h , Q }$ for the step-size denominator; replacing it by any computable upper bound $\overline { { V } } \geq V _ { h , Q }$ , as Algorithm 1 does, multiplies the outer regret by $\sqrt { \overline { { V } } / V _ { h , Q } }$ and changes nothing else.

At the beginning of round t, the outer learner has a state $x _ { t } \in K$ that is $\mathcal { H } _ { t - 1 }$ -measurable. Independently, the inner online unconstrained-submodular-maximization learner has a state from previous rounds and chooses $a _ { t } \in [ 0 , 1 ] ^ { d }$ before feedback from $f _ { t }$ is available. Define

$$
y _ { t } = x _ { t } \odot a _ { t } , \qquad z _ { t } = Y _ { 1 } ( x _ { t } ) .
$$

Both belong to $K \colon$ the first by down-closedness since $y _ { t } \le x _ { t }$ , the second by Lemma 4.5.

We do not use a deterministic convex combination: the two points need not be comparable, so directional concavity does not apply to the joining line. Instead, draw $Z _ { t } \sim$ Bernoulli(λ) and play

$$
p _ { t } = { \left\{ \begin{array} { l l } { z _ { t } , } & { Z _ { t } = 1 , } \\ { y _ { t } , } & { Z _ { t } = 0 . } \end{array} \right. }\tag{45}
$$

Then

$$
\mathbb { E } \left[ f _ { t } ( p _ { t } ) \mid f _ { t } , \mathcal { H } _ { t - 1 } , a _ { t } \right] = \lambda f _ { t } ( z _ { t } ) + ( 1 - \lambda ) f _ { t } ( y _ { t } ) .\tag{46}
$$

Crucially, the play coin $Z _ { t }$ is not used by the outer or inner state updates. Hence it does not enter the construction of $G _ { t }$ or the outer surrogate direction.

Factor-revealing master inequality. Fix s with $\zeta _ { s } , \chi _ { s } \geq 0$ and write

$$
\widetilde { q } _ { t } : = \widetilde { q } _ { s } ( f _ { t } , x _ { t } ) .\tag{47}
$$

By Lemma 4.7,

$$
f _ { t } ( z _ { t } ) \geq \theta _ { s } f _ { t } ( o ) - \chi _ { s } f _ { t } ( x _ { t } \odot o ) - 2 \zeta _ { s } f _ { t } ( x _ { t } ) - \langle \widetilde { q } _ { t } , o - x _ { t } \rangle .\tag{48}
$$

Recall the coeficients $u _ { r } , v _ { r } , w _ { \ i }$ <sub>r</sub> from (35). By Lemma 4.14 the transformed objective $G _ { t }$ has gradient norm at most $B ,$ so with

$$
E _ { \mathrm { b o x } } = O ( d m M _ { \sigma } \sqrt { T } ) + O ( B \sqrt { d } T / m )
$$

denoting the inner learning and grid-discretization error, Theorem 4.18 gives

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } f _ { t } ( y _ { t } ) \geq u _ { r } \mathbb { E } \sum _ { t = 1 } ^ { T } f _ { t } ( x _ { t } \odot o ) + v _ { r } \mathbb { E } \sum _ { t = 1 } ^ { T } f _ { t } ( x _ { t } ) + w _ { r } \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { 0 } ) - E _ { \mathrm { b o x } } .\tag{49}
$$

Combining (46), (48), and (49), and discarding the nonnegative term $w _ { r } \sum _ { t } f _ { t } ( \mathbf { 0 } )$

$$
\begin{array} { l } { \displaystyle \mathbb { E } \sum _ { t } f _ { t } ( p _ { t } ) \geq \lambda \theta _ { s } \sum _ { t } f _ { t } ( o ) + [ ( 1 - \lambda ) u _ { r } - \lambda \chi _ { s } ] \mathbb { E } \sum _ { t } f _ { t } ( x _ { t } \odot o ) } \\ { \displaystyle + [ ( 1 - \lambda ) v _ { r } - 2 \lambda \zeta _ { s } ] \mathbb { E } \sum _ { t } f _ { t } ( x _ { t } ) - \lambda \mathbb { E } \sum _ { t } \langle \widetilde { q } _ { t } , o - x _ { t } \rangle - E _ { \mathrm { b o x } } . } \end{array}\tag{50}
$$

Choose

$$
\lambda ( s , r ) = \operatorname* { m i n } \left\{ \frac { u _ { r } } { u _ { r } + \chi _ { s } } , \frac { v _ { r } } { v _ { r } + 2 \zeta _ { s } } \right\}\tag{51}
$$

and define the corresponding approximation factor

$$
\boxed { \alpha ( s , r ) = \theta _ { s } \operatorname* { m i n } \left\{ \frac { u _ { r } } { u _ { r } + \chi _ { s } } , \ \frac { v _ { r } } { v _ { r } + 2 \zeta _ { s } } \right\} . }\tag{52}
$$

Because $u _ { r } , v _ { r } > 0$ and $\chi _ { s } , \zeta _ { s } \geq 0$ , this choice lies in $( 0 , 1 ]$ , and the two coeficients multiplying the nonnegative quantities $f _ { t } ( x _ { t } \odot o )$ and $f _ { t } ( x _ { t } )$ in (50) are nonnegative and may be discarded. Consequently

$$
\mathbb { E } \sum _ { t } \xi _ { t } ( p _ { t } ) \geq \alpha ( s , r ) \sum _ { t } f _ { t } ( o ) - \lambda ( s , r ) \mathbb { E } \sum _ { t } \left. \widetilde { q } _ { t } , o - x _ { t } \right. - E _ { \mathrm { b o x } } .\tag{53}
$$

This is the central factor-revealing expression.

## 4.3.1 Noisy post-decision value-oracle implementation

Outer stochastic online linear optimization. By Lemma 4.5 we have $0 \leq \omega _ { \tau } ( x _ { t } ) \leq 1$ , so under $\begin{array} { r } { \operatorname* { s u p } _ { t , x } \| \nabla f _ { t } ( x ) \| _ { 2 } \leq B } \end{array}$

$$
\begin{array} { r } { \| \widetilde { q } _ { t } \| _ { 2 } \leq ( 1 + \zeta _ { s } ) B = : \Gamma _ { s } . } \end{array}\tag{54}
$$

The estimate $\widehat { q _ { t } }$ built below is not bounded by $\Gamma _ { s } \dot { : }$ a one-sided diference of two noisy values difers from the corresponding exact diference by up to $2 \sigma / h$ in each coordinate and is itself bounded only by $( M + 2 \sigma ) / h$ , while by Remark 4.20 we do not clip it. The outer layer therefore has to be run as stochastic online gradient ascent, with the step size governed by a second moment rather than by a deterministic radius. Let $\mathcal { G } _ { t }$ denote the round-t pre-call sigma-field of (7): it contains the history, the obliviously fixed $f _ { t } ,$ the state $x _ { t } ,$ and all randomization used to choose the round’s actions and query list, but none of the round’s oracle errors. Note that $x _ { t } .$ and hence $o - x _ { t }$ , is $\mathcal { G } _ { t } .$ -measurable. Given a bound V with E $\| \widehat { q } _ { t } \| _ { 2 } ^ { 2 } \leq V$ for all $t ,$ the learner updates

$$
x _ { t + 1 } = \Pi _ { K } \bigl ( x _ { t } + \eta \widehat { q } _ { t } \bigr ) , \qquad \eta = \frac { D } { \sqrt { T V } } ,
$$

and keeps the outer state fixed if $V = 0$

Lemma 4.21 (Projected stochastic online linear optimization). Let $\widehat { q } _ { 1 } , \ldots , \widehat { q } _ { T }$ be random vectors with E $\| \widehat { q } _ { t } \| _ { 2 } ^ { 2 } \leq \dot { V }$ for every t, and let $x _ { t + 1 } = \Pi _ { K } ( x _ { t } + \eta \widehat { q } _ { t } )$ with $\eta = D / \sqrt { T V }$ . Then for every $o \in K$

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } \left. \widehat { q } _ { t } , o - x _ { t } \right. \leq D \sqrt { T V } .\tag{55}
$$

Proof. This is the standard projected online gradient-ascent inequality [16] applied to the realized vectors $\widehat { q _ { t } } .$ telescoping $\lVert \boldsymbol { x } _ { t } - \boldsymbol { o } \rVert _ { 2 } ^ { \dot { 2 } }$ and using nonexpansiveness of $\Pi _ { K }$ gives

$$
\sum _ { t = 1 } ^ { T } \langle \widehat { q } _ { t } , o - x _ { t } \rangle \leq \frac { D ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } \| \widehat { q } _ { t } \| _ { 2 } ^ { 2 } .
$$

Taking expectations, bounding each second moment by $V ,$ , and substituting $\eta = D / \sqrt { T V }$ gives (55). No boundedness of the individual vectors is used. □

The master inequality contains the true field, so we split it into an optimization term, a martingale term, and a bias term. Writing ${ \bar { q } } _ { t } : = \mathbb { E } [ { \widehat { q } } _ { t } \mid { \mathcal { G } } _ { t } ]$

$$
\sum _ { t } \left. \widetilde { q } _ { t } , o - x _ { t } \right. = \sum _ { t } \left. \widehat { q } _ { t } , o - x _ { t } \right. + \sum _ { t } \left. \bar { q } _ { t } - \widehat { q } _ { t } , o - x _ { t } \right. + \sum _ { t } \left. \widetilde { q } _ { t } - \bar { q } _ { t } , o - x _ { t } \right. .\tag{56}
$$

The middle sum has zero expectation, because $o \mathrm { ~ - ~ } x _ { t }$ is G -measurable and $\mathbb { E } [ \widehat { q } _ { t } - \bar { q } _ { t } \ | \ \mathcal { G } _ { t } ] = 0 ;$ this is exactly where conditional unbiasedness of the oracle is spent. The last sum is bounded deterministically by

$$
\sum _ { t } { \left. { \widetilde { q } } _ { t } - { \bar { q } } _ { t } , o - x _ { t } \right. } \leq D \sum _ { t } \left\| { \widetilde { q } } _ { t } - { \bar { q } } _ { t } \right\| _ { 2 } ,\tag{57}
$$

so only the bias of the estimator, not its fluctuation, is charged at the linear-in-T rate. With exact values, $\widehat { q } _ { t } = \widehat { q } _ { t }$ and (56) collapses to the deterministic splitting used in the noiseless analysis.

Noisy value-oracle reconstruction. Following the finite-diference approach to black-box continuous submodular optimization [23], after committing to the action the learner may call the oracle for the current $f _ { t }$ at selected points of the cube.

Let $Q \geq 1 , h \in ( 0 , 1 / 2 )$ , and use midpoint nodes

$$
\tau _ { j } = \frac { j - \frac 1 2 } { Q } , \qquad j = 1 , \dots , Q .
$$

At $z _ { j } = Y _ { \tau _ { j } } ( x _ { t } )$ , estimate each coordinate of $\nabla f _ { t } ( z _ { j } )$ by an inward one-sided finite diference of step h. If $( z _ { j } ) _ { i } \geq h$ , use

$$
\widehat { \partial } _ { i } f _ { t } ( z _ { j } ) = \frac { \widehat { f _ { t } } ( z _ { j } ) - \widehat { f _ { t } } ( z _ { j } - h e _ { i } ) } { h } ;
$$

otherwise use

$$
\widehat { \partial } _ { i } f _ { t } ( z _ { j } ) = \frac { \widehat { f } _ { t } ( z _ { j } + h e _ { i } ) - \widehat { f } _ { t } ( z _ { j } ) } { h } .
$$

Each of the two values is a separate oracle call, so the estimate is conditionally unbiased for the corresponding exact finite diference and has conditional variance at most $2 \sigma ^ { 2 } / h ^ { 2 }$ . Because $h < 1 / 2$ the selected forward or backward point always remains in $[ 0 , 1 ] ^ { d }$ . We use the same rule at $x _ { t }$ for the term $\zeta _ { s } \nabla f _ { t } ( x _ { t } )$ , but repeat it $Q$ times with independent calls and average, which is what makes the noise in that term decay at the same rate as in the path term. The estimator is

$$
\widehat { q } _ { t } = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } e ^ { \tau _ { j } - 1 } \omega _ { \tau _ { j } } ( x _ { t } ) \odot \widehat { \nabla } f _ { t } ( z _ { j } ) \ + \ \frac { \zeta _ { s } } { Q } \sum _ { k = 1 } ^ { Q } \widehat { \nabla } ^ { ( k ) } f _ { t } ( x _ { t } ) ,\tag{58}
$$

where $\widehat { \nabla } ^ { ( k ) } f _ { t } ( x _ { t } )$ are the $Q$ independent repetitions at $x _ { t }$ . Note that all $2 d Q + 2 d Q$ locations are determined by $x _ { t }$ alone, so the list is nonadaptive; this is used again in Appendices $\mathrm { A - B }$

Lemma 4.22 (Noisy value-oracle reconstruction). Suppose every $f _ { t }$ is L-smooth on $[ 0 , 1 ] ^ { d }$ and the oracle satisfies (7). For $Q \geq 1$ and $h \in ( 0 , 1 / 2 )$ , the estimator (58) uses 4dQ oracle calls and satisfies

$$
\| \mathbb { E } [ \widehat { q } _ { t } \mid \mathcal { G } _ { t } ] - \widetilde { q } _ { s } ( f _ { t } , x _ { t } ) \| _ { 2 } \leq C _ { 2 } \sqrt { d } \left( L h + \frac { B + L } { Q } \right) = : \Delta _ { h , Q } ,\tag{59}
$$

$$
\mathbb { E } \big [ \| \widehat { q _ { t } } \| _ { 2 } ^ { 2 } \bigm | \mathcal { G } _ { t } \big ] \leq C _ { 3 } \left( \Gamma _ { s } ^ { 2 } + \Delta _ { h , Q } ^ { 2 } + \frac { d \sigma ^ { 2 } } { Q h ^ { 2 } } \right) = : V _ { h , Q } ,\tag{60}
$$

for universal constants $C _ { 2 } , C _ { 3 }$ . In particular, with $h = \varepsilon / 2$ and $Q = \lceil \varepsilon ^ { - 1 } \rceil$ the estimator uses $O ( d / \varepsilon )$ calls and has

$$
\Delta _ { h , Q } \leq C _ { 0 } ( B + L ) \varepsilon \sqrt { d } , \qquad V _ { h , Q } = O \biggl ( B ^ { 2 } + d ( B + L ) ^ { 2 } \varepsilon ^ { 2 } + \frac { d \sigma ^ { 2 } } { \varepsilon } \biggr ) .\tag{61}
$$

When $\sigma = 0$ the estimator is deterministic, $V _ { h , Q } = O ( \Gamma _ { s } ^ { 2 } + \Delta _ { h , Q } ^ { 2 } )$ , and (59) is the exact-oracle error bound.

Proof. Bias. The oracle errors are conditionally centred and every value in (58) is a fresh call, so $\mathbb { E } [ \widehat { q } _ { t } \ | \ \mathcal { G } _ { t } ]$ is exactly the estimator built from exact values. It therefore sufices to bound the deterministic error of the latter. L-smoothness bounds the error of every one-sided finite-diference coordinate by $L h / 2$ , hence the Euclidean error of a reconstructed gradient by $L h { \sqrt { d } } / 2$

It remains to make the quadrature bound explicit. For fixed $f = f _ { t }$ and $x = x _ { t }$ , define

$$
\Phi ( \tau ) = e ^ { \tau - 1 } \omega _ { \tau } ( x ) \odot \nabla f ( Y _ { \tau } ( x ) ) .
$$

On each smooth phase, $\left\| \dot { \boldsymbol Y } _ { \tau } \right\| _ { 2 } = \left\| \omega _ { \tau } \odot { \boldsymbol x } \right\| _ { 2 } \leq \sqrt { d } , \left\| \omega _ { \tau } \right\| _ { \infty } \leq 1$ , and direct diferentiation of (16) gives $\begin{array} { r } { \| \dot { \boldsymbol { \omega } } _ { \tau } \| _ { \infty } \leq 1 } \end{array}$ . Since an L-Lipschitz gradient is diferentiable almost everywhere with Hessian operator norm at most $L , \Phi$ is absolutely continuous on each phase and, almost everywhere there,

$$
\dot { \Phi } ( \tau ) = e ^ { \tau - 1 } \left[ \left( \omega _ { \tau } + \dot { \omega } _ { \tau } \right) \odot \nabla f ( Y _ { \tau } ) + \omega _ { \tau } \odot \nabla ^ { 2 } f ( Y _ { \tau } ) \dot { Y } _ { \tau } \right] .
$$

Consequently,

$$
\left\| \dot { \Phi } ( \tau ) \right\| _ { 2 } \leq 2 B + L \sqrt { d } \leq 2 ( B + L ) \sqrt { d }\tag{62}
$$

on both smooth pieces. This displays the two contributions separately: the exponential and multiplier derivatives cost $O ( B )$ , while transporting the gradient along $Y _ { \tau }$ costs $O ( L \sqrt { d } )$

For every midpoint cell not containing s, integrating (62) around the midpoint bounds its quadrature error by $O ( ( B + L ) \sqrt { d } / Q ^ { 2 } )$ . Summing these cells gives $O ( ( B + L ) { \sqrt { d } } / Q )$ . At $s ,$ the trajectory is continuous and $\omega _ { s + } - \omega _ { s - } = x$ , so

$$
\| \Phi ( s + ) - \Phi ( s - ) \| _ { 2 } \leq B .
$$

There is at most one cell of width $1 / Q$ containing this jump. Since $\| \Phi ( \tau ) \| _ { 2 } \le B$ on both sides, the integral over that cell and its midpoint approximation difer by at most $2 B / Q$ . Thus the jump contributes the same claimed order rather than a constant error.

Combining quadrature and finite-diference errors, including the identical finite-diference estimate of $\zeta _ { s } \nabla f _ { t } ( x _ { t } )$ and the elementary bound $| \zeta _ { s } | \le 1$ from (20)–(24), gives (59) with a universal $C _ { 2 }$ Averaging the $Q$ repetitions at $x _ { t }$ does not change this bound, since all of them have the same conditional mean.

Second moment. Fix a coordinate i. In the path term the multipliers satisfy $| e ^ { \tau _ { j } - 1 } ( \omega _ { \tau _ { i } } ( x _ { t } ) ) _ { i } | \leq 1$ and the $Q$ one-sided diferences at the distinct nodes use disjoint sets of oracle calls, hence are conditionally independent with variance at most $2 \sigma ^ { 2 } / h ^ { 2 }$ each; the ith coordinate of the path term therefore has conditional variance at most $2 \sigma ^ { 2 } / ( Q h ^ { 2 } )$ . The same computation applies to the second term of (58), using $| \zeta _ { s } | \le 1$ and the independence of the $Q$ repetitions. Summing over the d coordinates gives

$$
\mathbb { E } \left[ \Vert \widehat { q } _ { t } - \mathbb { E } [ \widehat { q } _ { t } \mid \mathcal { G } _ { t } ] \Vert _ { 2 } ^ { 2 } \mid \mathcal { G } _ { t } \right] \leq \frac { C d \sigma ^ { 2 } } { Q h ^ { 2 } } ,
$$

and $\lVert \mathbb { E } [ \widehat { q } _ { t } \mid \mathcal { G } _ { t } ] \rVert _ { 2 } \leq \Gamma _ { s } + \Delta _ { h , Q }$ by (54) and (59). Adding the squared mean to the variance proves (60).

Calls. Each of the $Q$ nodes uses $2 d$ calls and the $Q$ repetitions at $x _ { t }$ use $2 d Q$ calls, for a total of $4 d Q ;$ ; at $Q = \lceil \varepsilon ^ { - 1 } \rceil$ this is $O ( d / \varepsilon )$ . Substituting $h = \varepsilon / 2$ and $Q = \lceil \varepsilon ^ { - 1 } \rceil$ into $( 5 9 ) ‐ ( 6 0 )$ and using $\Gamma _ { s } \leq 2 B$ gives (61). □

Combining Lemma 4.21 with the splitting (56) and the bias bound (57), the outer layer contributes

$$
\mathbb { E } \sum _ { t } \left. \widetilde { q } _ { t } , o - x _ { t } \right. \le D \sqrt { T V _ { h , Q } } + D \Delta _ { h , Q } T .\tag{63}
$$

At $h = \varepsilon / 2$ and $Q = \lceil \varepsilon ^ { - 1 } \rceil$ , (61) turns this into

$$
O \Big ( D B \sqrt { T } + D ( B + L ) \varepsilon \sqrt { d T } + D \sigma \sqrt { d T / \varepsilon } \Big ) + O \Big ( D ( B + L ) \varepsilon \sqrt { d } T \Big ) ,
$$

whose second entry dominates the middle entry of the first; the outer cost is therefore

$$
O \Big ( D B \sqrt { T } + D ( B + L ) \varepsilon \sqrt { d } T + D \sigma \sqrt { d T / \varepsilon } \Big ) ,\tag{64}
$$

at an outer cost of $O ( d / \varepsilon )$ oracle calls per round. The last entry is the only place where the noise level appears, and it is the price of dividing an $O ( \sigma )$ error by the finite-diference step $h = \varepsilon / 2$ before averaging $Q = \Theta ( \varepsilon ^ { - 1 } )$ repetitions.

## 4.3.2 The approximation factor

Recall

$$
\alpha ( s , r ) = \theta _ { s } \operatorname* { m i n } \left\{ \frac { u _ { r } } { u _ { r } + \chi _ { s } } , \frac { v _ { r } } { v _ { r } + 2 \zeta _ { s } } \right\} .
$$

Lemma 4.23 (Equalizing asymmetry). Fix s with $\zeta _ { s } > 0$ and $\chi _ { s } > 0$ . The two terms in the minimum are equal precisely at

$$
\boxed { r _ { s } = \frac { 4 \zeta _ { s } } { \chi _ { s } } . }\tag{65}
$$

Moreover, for $r _ { s } \ge 1$ the map $r \mapsto \alpha ( s , r )$ is maximized at $r = r _ { s }$

Proof. Writing the two candidate values as $\lambda ,$ the conditions $\lambda / ( 1 - \lambda ) = u _ { r } / \chi _ { s }$ and $\lambda / ( 1 - \lambda ) =$ $v _ { r } / ( 2 \zeta _ { s } )$ give

$$
\frac { u _ { r } / \chi _ { s } } { v _ { r } / ( 2 \zeta _ { s } ) } = \frac { 2 u _ { r } \zeta _ { s } } { v _ { r } \chi _ { s } } = \frac { 4 \zeta _ { s } } { r \chi _ { s } } ,
$$

using $u _ { r } / v _ { r } = 2 / r$ . This ratio is strictly decreasing in r and equals one exactly at $r = r _ { s }$ . Hence the $u _ { r }$ branch exceeds the $v _ { r }$ branch for $r < r _ { s }$ and is smaller for $r > r _ { s }$ , so the minimum equals $\theta _ { s } v _ { r } / ( v _ { r } + 2 \zeta _ { s } )$ for $r \leq r _ { s }$ and $\theta _ { s } u _ { r } / ( u _ { r } + \chi _ { s } )$ for $r \geq r _ { s }$ . On $r \leq r _ { s }$ the selected $\left( v _ { r } \right)$ branch is increasing in r because $v _ { r } = r ^ { 2 } / ( 1 + r ) ^ { 2 }$ is increasing; on $r \geq r _ { s }$ the selected $\left( u _ { r } \right)$ branch is decreasing whenever $r \geq 1$ , because $u _ { r } ^ { \prime } = 2 ( 1 - r ) / ( 1 + r ) ^ { 3 }$ . For $r _ { s } \ge 1$ the maximum is therefore attained at $r _ { s }$ □

Thus a numerical search may be reduced to one dimension. The theorem itself does not rely on a floating-point optimization claim: the next proposition fixes rational parameters and certifies the displayed factor by direct evaluation.

Proposition 4.24 (Certified factor). Set

$$
s _ { 0 } = { \frac { 1 8 3 } { 5 0 0 } } , \qquad r _ { 0 } = { \frac { 5 4 1 } { 2 5 0 } } , \qquad \lambda _ { 0 } = \lambda ( s _ { 0 } , r _ { 0 } ) .\tag{66}
$$

Then $\zeta _ { s _ { 0 } } > 0 , \chi _ { s _ { 0 } } > 0$ , and

$$
\alpha ( s _ { 0 } , r _ { 0 } ) > 0 . 4 0 1 .\tag{67}
$$

Numerically, $\lambda _ { 0 } \approx 0 . 8 0 3 8$ and $\alpha ( s _ { 0 } , r _ { 0 } ) \approx 0 . 4 0 1 0 2$ . The rational parameters above are a certified rounding of the stationary point located by the one-dimensional search of Lemma $4 . 2 3 ;$ the formal theorem uses the headline factor 0.401.

Proof. Substituting the rational numbers (66) into (19), (20), (24) and (35) expresses every quantity as a rational function of $e ^ { s _ { 0 } }$ and $e ^ { - 1 }$ . We use outward-rounded rational interval arithmetic. Concretely, the enclosures

$$
e ^ { s _ { 0 } } \in [ 1 . 4 4 1 9 5 5 2 4 2 6 5 4 5 , 1 . 4 4 1 9 5 5 2 4 2 6 5 4 6 ] , \qquad e ^ { - 1 } \in [ 0 . 3 6 7 8 7 9 4 4 1 1 7 1 4 , 0 . 3 6 7 8 7 9 4 4 1 1 7 1 5 ]
$$

already sufice. The first follows by summing the positive Taylor series through degree 11 and bounding its tail by the geometric majorant whose first term is $s _ { 0 } ^ { 1 2 } / 1 2 !$ and ratio is at most $s _ { 0 } / 1 3$ The second follows from the alternating Taylor partial sums of degrees 16 and 17. All endpoints are rational, so the remaining interval operations are exact. Substitution gives

$$
\theta _ { s _ { 0 } } > 0 . 4 9 8 9 0 1 , \qquad \zeta _ { s _ { 0 } } \in ( 0 . 0 5 7 0 8 , 0 . 0 5 7 0 9 ) , \qquad \chi _ { s _ { 0 } } \in ( 0 . 1 0 5 4 9 , 0 . 1 0 5 5 1 ) .
$$

In particular $\zeta _ { s _ { 0 } } > 0$ and $\chi _ { s _ { 0 } } > 0$ , so Lemma 4.7 applies and $\lambda ( s _ { 0 } , r _ { 0 } )$ is well defined. The two box coeficients are the exact rationals

$$
u _ { r _ { 0 } } = { \frac { 2 7 0 5 0 0 } { 6 2 5 6 8 1 } } , \qquad v _ { r _ { 0 } } = { \frac { 2 9 2 6 8 1 } { 6 2 5 6 8 1 } } .
$$

Both $\varsigma \mapsto u _ { r _ { 0 } } / ( u _ { r _ { 0 } } + \varsigma )$ and $\varsigma \mapsto v _ { r _ { 0 } } / ( v _ { r _ { 0 } } + 2 \varsigma )$ are decreasing, so the upper ends of the two intervals, followed only by exact rational arithmetic, give

$$
\frac { u _ { r _ { 0 } } } { u _ { r _ { 0 } } + \chi _ { s _ { 0 } } } > 0 . 8 0 3 8 2 6 , \qquad \frac { v _ { r _ { 0 } } } { v _ { r _ { 0 } } + 2 \zeta _ { s _ { 0 } } } > 0 . 8 0 3 8 0 0 .
$$

Thus $\lambda _ { 0 } > 0 . 8 0 3 8 0 0$ and

$$
\alpha ( s _ { 0 } , r _ { 0 } ) > 0 . 4 9 8 9 0 1 \cdot 0 . 8 0 3 8 0 0 > 0 . 4 0 1 ,
$$

which proves (67). The one-dimensional search of Lemma 4.23 locates the stationary point near these parameters, but that search is not needed for the certified guarantee. □

Remark 4.25. The theorem-level constant is $\alpha ^ { \star } = 0 . 4 0 1$ ; the more precise number above is used only to certify that rounded factor. The online theorem is not a restatement of an ofline one: the objective changes adversarially, and the action is committed before current-function value feedback is available.

## 4.3.3 Equality with the ofline coeficient program

Our composition and the ofline construction of [10] repay the $f ( x \oplus o )$ debt by diferent mechanisms. Ofline, the input point to Box-Maximization is an approximate local maximum, which converts $f ( x )$ into $\scriptstyle { \frac { 1 } { 2 } } [ f ( x \oplus o ) + f ( x \odot o ) ]$ and yields the coeficients $( 4 r + r ^ { 2 } ) / ( 2 ( 1 + r ) ^ { 2 } )$ on $f ( \boldsymbol { x } \odot \boldsymbol { o } )$ and $r ^ { 2 } / ( 2 ( 1 + r ) ^ { 2 } )$ on $f ( x \oplus o )$ . Online, no local-maximum step is available; instead the lattice inequality (26) trades $f ( x \oplus o )$ for $2 f ( x )$ and a linear residue that the outer learner absorbs. The next proposition shows that the two mechanisms have the same optimized value at the balanced asymmetry associated with our certified delay.

For completeness, the ofline expression used below follows directly from the last convex combination in the proof of [10]. If λ is the weight on the delayed-trajectory candidate (so $1 - \lambda$ is the weight on the ofline box candidate), its limiting lower bound is

$$
\lambda \theta _ { s } f ( o ) + \left[ ( 1 - \lambda ) \widehat { u } _ { r } - \lambda \mu _ { s } \right] f ( x \odot o ) + \left[ ( 1 - \lambda ) \widehat { v } _ { r } - \lambda \zeta _ { s } \right] f ( x \oplus o ) .
$$

Making both residual coeficients nonnegative permits precisely

$$
\lambda \leq \operatorname* { m i n } \left\{ \frac { \widehat { u } _ { r } } { \widehat { u } _ { r } + \mu _ { s } } , \frac { \widehat { v } _ { r } } { \widehat { v } _ { r } + \zeta _ { s } } \right\} ,
$$

which yields the definition of $\alpha ^ { \mathrm { o f f } }$ in the proposition.

Proposition 4.26 (Online/ofline coeficient equality). For $s \in ( 0 , 1 )$ define the limiting ofline factor-revealing expression of $[ 1 0 ] ,$ with its vanishing accuracy terms suppressed,

$$
\alpha ^ { \mathrm { o f f } } ( s , r ) = \theta _ { s } \operatorname* { m i n } \left\{ \frac { \widehat { u } _ { r } } { \widehat { u } _ { r } + \mu _ { s } } , \frac { \widehat { v } _ { r } } { \widehat { v } _ { r } + \zeta _ { s } } \right\} , \qquad \widehat { u } _ { r } = \frac { 4 r + r ^ { 2 } } { 2 ( 1 + r ) ^ { 2 } } , \quad \widehat { v } _ { r } = \frac { r ^ { 2 } } { 2 ( 1 + r ) ^ { 2 } } .
$$

The common branch underlying the comparison is the identity

$$
\left| \frac { \widehat { v } _ { r } } { \widehat { v } _ { r } + \zeta _ { s } } = \frac { v _ { r } } { v _ { r } + 2 \zeta _ { s } } \right| \qquad ( \widehat { v } _ { r } = v _ { r } / 2 ) .\tag{68}
$$

Then for every s with $\zeta _ { s } > 0$ and $\chi _ { s } > 0$

(i) both minima are attained by their second argument for small r and by their first for large $r ,$ and both switch at the same value $r _ { s } = 4 \zeta _ { s } / \chi _ { s }$ of (65);

(ii) $\alpha ( s , r ) = \alpha ^ { \mathrm { o f f } } ( s , r )$ for every $0 < r \le r _ { s }$ , and in particular at $\boldsymbol { r } = \boldsymbol { r } _ { s } ;$

(iii) $i f r _ { s } \ge 2$ , then both expressions are maximized over $r > 0$ at $r = r _ { s }$ , and hence max<sub>r</sub> $\alpha ( s , r ) =$ max<sub>r</sub> $\alpha ^ { \mathrm { o f f } } ( s , r )$

For the certified delay s<sub>0</sub> of Proposition $4 . 2 4 .$ , one has $r _ { s _ { 0 } } > 2$ and

$$
\boxed { \operatorname* { m a x } _ { r > 0 } \alpha ( s _ { 0 } , r ) = \operatorname* { m a x } _ { r > 0 } \alpha ^ { \mathrm { { o f f } } } ( s _ { 0 } , r ) > 0 . 4 0 1 } .
$$

Proof. Write $\Xi ( s , r )$ for the common value in (68).

For the online expression, the two branches are equal precisely when

$$
\frac { u _ { r } } { \chi _ { s } } = \frac { v _ { r } } { 2 \zeta _ { s } } .
$$

Their ratio is

$$
\frac { u _ { r } / \chi _ { s } } { v _ { r } / ( 2 \zeta _ { s } ) } = \frac { 4 \zeta _ { s } } { r \chi _ { s } } ,
$$

which is strictly decreasing, exceeds one for $r < r _ { s } ,$ , and is below one for $r > r _ { s } ,$ , where $r _ { s } = 4 \zeta _ { s } / \chi _ { s }$ Hence the online minimum selects its second branch before $r _ { s }$ and its first branch after $r _ { s }$

For the ofline expression, the corresponding ratio is

$$
\frac { \widehat { u } _ { r } / \mu _ { s } } { \widehat { v } _ { r } / \zeta _ { s } } = \frac { \widehat { u } _ { r } \zeta _ { s } } { \widehat { v } _ { r } \mu _ { s } } = \frac { ( 4 r + r ^ { 2 } ) \zeta _ { s } } { r ^ { 2 } \mu _ { s } } = \left( 1 + \frac { 4 } { r } \right) \frac { \zeta _ { s } } { \mu _ { s } } ,
$$

which is also strictly decreasing in r and equals one exactly when $4 / r = \mu _ { s } / \zeta _ { s } - 1 = ( \mu _ { s } - \zeta _ { s } ) / \zeta _ { s } =$ $\chi _ { s } / \zeta _ { s }$ , that is at $r = 4 \zeta _ { s } / \chi _ { s } = r _ { s }$ . This proves the branch order and common switch in part (i).

For (ii), when $0 < r \le r _ { s }$ , both minima equal their common second argument $\Xi ( s , r )$ . Thus $\alpha ( s , r ) = \theta _ { s } \Xi ( s , r ) = \alpha ^ { \mathrm { o f f } } ( s , r )$ throughout this interval.

For (iii), the monotonicity established in Lemma 4.23 shows that $r \mapsto \alpha ( s , r )$ increases up to $r _ { s }$ and decreases after it when $r _ { s } \ge 1$ . For the ofline expression, the common second branch is likewise increasing, while

$$
\widehat { u } _ { r } = \frac { r ( r + 4 ) } { 2 ( 1 + r ) ^ { 2 } }
$$

has derivative proportional to $2 - r$ and is therefore decreasing for $r \geq 2$ . If $r _ { s } \geq 2$ , both minima are consequently maximized at their common switch, with value $\theta _ { s } \Xi ( s , r _ { s } )$

Finally, the interval bounds in Proposition 4.24 imply

$$
r _ { s _ { 0 } } = \frac { 4 \zeta _ { s _ { 0 } } } { \chi _ { s _ { 0 } } } > \frac { 4 ( 0 . 0 5 7 0 8 ) } { 0 . 1 0 5 5 1 } > 2 .
$$

Part (iii) therefore applies at $s _ { 0 }$ , and the strict lower bound follows from Proposition 4.24. □

Proposition 4.26 is the precise comparison needed here: at the certified delay, both programs are optimized over the asymmetry at the same switch and attain the same value above 0.401. It does not assert that the programs have identical global behavior for every delay. Only this certified-delay comparison is needed to identify the online coeficient with the ofline benchmark; the online regret theorem itself uses the certified inequality of Proposition 4.24. The broader equality in Proposition 4.26 is included to explain the matching coeficient frontier. The ofline guarantee and our theorem both use the rounded factor 0.401; no ofline improvement is claimed or implied here. The equality concerns only the optimized approximation coeficient; the online algorithm, oracle model, and mechanism controlling the residual terms are diferent from those of the ofline algorithm.

## 4.3.4 Proof of the post-decision value-oracle theorem

Proof of Theorem $4 . 1 .$ . Fix $s = s _ { 0 } , r = r _ { 0 }$ and $\lambda = \lambda _ { 0 }$ as in Proposition 4.24, and run Algorithm 1 with the stated m and ε.

The action is feasible because both candidates in (45) belong to K. It is chosen before feedback from $f _ { t }$ is available because $x _ { t }$ is determined by previous feedback and $a _ { t }$ uses only the prior history and fresh pre-round randomization; the current function is accessed only afterward, through the oracle calls used to estimate the Double-Greedy marginals, reconstruct the outer field, and update the states.

The order of operations inside a round is what makes the noise harmless. All dm lifted binary decisions, the mixture coin, and the played action are committed before any call to the round’s oracle. Only then are the estimates (40) and (58) formed, from fresh calls. Consequently, for every lifted element i the true pair $( g _ { i } ^ { + , t } , \dot { g _ { i } } ^ { - , \dot { t } } )$ is $\mathcal { F } _ { t , i - 1 }  – \mathrm { m e a s u r a b l e }$ and the estimate $( \hat { g } _ { i } ^ { + , t } , \hat { g } _ { i } ^ { - , \bar { t } } )$ is conditionally unbiased given $\mathcal { F } _ { t , i } ,$ , and indeed given the whole round’s decisions, even though it is observed only after the remaining binary decisions have been made. Lemma 4.13 therefore applies to each of the dm learners separately, despite the acyclic within-round dependence, and the current objective influences only round $t + 1$ and later.

Fix a comparator $o \in K$ Theorem 4.18 applied to $G _ { t } ( a ) \ : = \ : f _ { t } ( x _ { t } \odot a )$ , with $B _ { G } \leq B$ by Lemma 4.14, gives the master inequality (53) with $E _ { \mathrm { b o x } } = { \cal O } ( d m M _ { \sigma } \sqrt { T } ) + { \cal O } ( B \sqrt { d } T / m )$ . Bounding its linear term by (64) yields

$$
\begin{array} { c } { { \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { T } f _ { t } ( p _ { t } ) \geq \alpha ( s _ { 0 } , r _ { 0 } ) \sum _ { t = 1 } ^ { T } f _ { t } ( o ) - O ( D B \sqrt { T } ) - O ( d m M _ { \sigma } \sqrt { T } ) - O ( B \sqrt { d } T / m ) } } \\ { { - O \Big ( D ( B + L ) \varepsilon \sqrt { d } T \Big ) - O \Big ( D \sigma \sqrt { d T / \varepsilon } \Big ) . } } \end{array}
$$

Proposition 4.24 gives $\alpha ( s _ { 0 } , r _ { 0 } ) > \alpha ^ { \star }$ , which proves (11); since the sequence is fixed obliviously, taking $o = o ^ { * } \in \arg \operatorname* { m a x } _ { o \in K } \sum _ { t } f _ { t } ( o )$ is legitimate and turns (11) into a bound on $\mathrm { R e g } _ { \alpha ^ { \star } } ( T )$ . Finally, the inner chain uses 4dm oracle calls and the outer reconstruction uses $4 d Q = O ( d / \varepsilon )$ , proving (12). □

Proof of Corollary $4 . 2 .$ . Substitute m $\begin{array} { r l r } { \mathbf { \Phi } } & { { } = } & { \left\lceil T ^ { 1 / 4 } \right\rceil } \end{array}$ and $\varepsilon ~ = ~ T ^ { - 1 / 4 }$ into (11): the inner term is $O ( d M _ { \sigma } T ^ { 3 / 4 } )$ , the grid term is $O ( B \sqrt { d } T ^ { 3 / 4 } )$ , and the field-bias term is $O ( D ( B + L ) \sqrt { d } T ^ { 3 / 4 } )$ . The stochastic-field term is $O ( D \sigma { \sqrt { d } } T ^ { 5 / 8 } )$ , and $D \leq \sqrt { d }$ gives $D \sigma \sqrt { d } \leq d \sigma \leq d M _ { \sigma }$ , so it is dominated by the inner term. The call count is $O ( d ( m + \varepsilon ^ { - 1 } ) ) = O ( d T ^ { 1 / 4 } )$ □

Remark 4.27 (Unknown horizon). Algorithm 1 uses T only through η, m and ε. Running it on consecutive epochs of doubling length, with the parameters recomputed and all learner states reset at the start of each epoch, multiplies every bound above by a constant: each regret term is a positive power of the epoch length, so its geometric sum is dominated by the last completed epoch. The same argument applies to the batched and bandit algorithms; initial epochs too short to satisfy their block-size feasibility conditions can use any fixed feasible action and contribute only a constant-order term.

## 4.4 Consequences, complexity, and robustness

## 4.4.1 Why the ofline factor survives online

The key diference from a round-by-round application of an ofline algorithm is that the quantities $f _ { t } ( x _ { t } \odot o )$ and $f _ { t } ( x _ { t } )$ need not be eliminated pointwise. Instead, the transformed objective $G _ { t } ( a ) =$ $f _ { t } ( x _ { t } \odot a )$ is given to the inner online USM learner, whose cumulative guarantee controls these two quantities after summation over time:

$$
\mathbb { E } \sum _ { t } f _ { t } ( x _ { t } \odot a _ { t } ) \ge u _ { r } \mathbb { E } \sum _ { t } f _ { t } ( x _ { t } \odot o ) + v _ { r } \mathbb { E } \sum _ { t } f _ { t } ( x _ { t } ) - o ( T ) .
$$

This is an amortized certificate: the residual terms produced by the outer endpoint inequality are paid for by cumulative online learning rather than by a pointwise local-optimality condition. At the certified delay, the resulting coeficient optimized over the asymmetry equals the corresponding ofline coeficient, as established in Proposition 4.26.

## 4.4.2 Oracle complexity and timing

Query accounting. The count lifting has $n _ { m } = d m \ \mathrm { e l e m e n t s }$ . The sequential Double-Greedy construction requires $O ( d m )$ calls to the current-function oracle. Under exact feedback one would cache repeated evaluations of the same point. The values of the successive $X _ { i } ^ { t }$ and $Y _ { i } ^ { t }$ states can be reused, so storing the two running chain values costs at most $2 d m + 2$ calls per round. Under (7) caching is not permitted, because two occurrences of the same geometric point must carry independent errors for the estimates (40) to be conditionally unbiased and for Lemma 4.13 to apply. The inner layer therefore costs 4dm calls per round instead of 2dm + 2: a factor of two, not a change of order. All updates of the lifted sets and balance learners between these calls are purely combinatorial and use no oracle values. Lemma 4.22 uses $4 d Q = O ( d / \varepsilon )$ additional calls per round. Consequently the total oracle cost is

$$
O \big ( d ( m + \varepsilon ^ { - 1 } ) \big )\tag{69}
$$

calls per round, and the balanced choice $m = \Theta ( T ^ { 1 / 4 } ) , \varepsilon = \Theta ( T ^ { - 1 / 4 } )$ makes all discretization, learning and noise errors $O ( T ^ { 3 / 4 } )$ with $O ( d T ^ { 1 / 4 } )$ calls per round under direct same-round implementation. Appendix B reduces the per-round budget to $O ( T ^ { \delta } )$ by spreading this list over a block.

Online timing. The dependency graph is

$$
x _ { t } , \ a _ { t } \longrightarrow p _ { t } \longrightarrow f _ { t } \longrightarrow ( G _ { t } , { \widehat { q } } _ { t } ) \longrightarrow ( x _ { t + 1 } , a _ { t + 1 } ) ,
$$

so $x _ { t }$ is $\mathcal { H } _ { t - 1 ^ { - } } \mathrm { m e a s u r a b l e }$ , while $a _ { t }$ and $p _ { t }$ are measurable with respect to $\mathcal { H } _ { t - 1 }$ augmented by fresh pre-round randomization. The transformed objective $G _ { t } ( a ) = f _ { t } ( x _ { t } \odot a )$ is therefore an ordinary nonanticipating online objective sequence for the inner learner: the feedback defining $G _ { t }$ is supplied only after the inner decision $a _ { t }$ has been made. That $G _ { t }$ depends on the evolving outer state creates no drift problem: under an obliviously fixed reward sequence the outer projected update determines $x _ { t }$ before $f _ { t }$ is used at round $t ,$ and the inner learner simply receives the resulting current objective after its decision, exactly as required by online unconstrained submodular maximization. Within a round, the balance learner $\mathcal { L } _ { i }$ receives inputs that depend only on the decisions of $\mathcal { L } _ { 1 } , \ldots , \mathcal { L } _ { i - 1 }$ , so the dependency graph among learners is acyclic, as in [17].

Relation to the ofline construction. The ofline construction uses the same transformation $G ( a ) = f ( x \odot a )$ and performs unconstrained Box-Maximization on $[ 0 , 1 ] ^ { d }$ , mapping the returned point back by multiplication with $x ;$ down-closedness makes it feasible. Our construction preserves this geometry but replaces ofline Box-Maximization by online unconstrained maximization on the sequence $G _ { t }$ . This is not an ofline-to-online black-box reduction: the outer state changes over time, and the inner learner must track a changing sequence of transformed objectives using decisions made before those objectives are observed. That the asymmetric coeficients survive is the content of Theorem 4.8; hence the online replacement incurs only an additive learning loss. Proposition 4.26 verifies that, at the certified delay, this replacement preserves the ofline factor after optimizing the asymmetry.

## 4.4.3 Relation to the ofline frontier

The usual constant-sequence argument gives the qualitative ceiling needed here. If an online algorithm satisfies

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } f ( p _ { t } ) \geq \alpha T \operatorname* { m a x } _ { o \in K } f ( o ) - R ( T )
$$

on the repeated objective $f _ { t } \equiv f$ , then a uniformly random played point has expected value at least

$$
\alpha \operatorname* { m a x } _ { o \in K } f ( o ) - \frac { R ( T ) } { T } .
$$

Thus a polynomial-time online factor with a suitably normalized polynomial-rate regret bound would also yield the same ofline factor up to arbitrarily small error. Appendix C states the precise computational assumptions and proof.

This observation does not make 0.401 optimal. It says only that an online factor beyond the current ofline record would simultaneously improve that record. A related multilinear-extension problem over down-closed polytopes is value-oracle inapproximable beyond 0.478, already for a partition matroid polytope; the same threshold holds for a cardinality constraint [12, 13].

Remark 4.28 (Modularity). The composition is modular in both of its layers. The outer layer consumes only the endpoint inequality (21) through the three coeficients $\theta _ { s } , \chi _ { s } , \zeta _ { s } ,$ , and the inner layer consumes only the triple $\left( u _ { r } , v _ { r } , w _ { r } \right)$ of box coeficients. Any improvement of either layer, whether a sharper delayed-trajectory inequality or a box guarantee with a better $\left( u _ { r } , v _ { r } , w _ { r } \right)$ frontier, propagates through (52) verbatim, provided the box guarantee is available in the cumulative online form of Proposition 4.17. Within the present architecture the box layer is already extremal: Theorem 4.8 caps the balance constant at $2 \sqrt { c _ { X } c _ { Y } }$ , so the coeficients of Theorem 4.18 cannot be improved by reweighting.

## 4.4.4 Where the noise is paid for

It is worth collecting in one place how the oracle noise (7) enters, since it is distributed across the construction rather than confined to a single step.

1. Inner layer. Each of the dm balance learners is driven by an estimated marginal pair on the scale $M _ { \sigma }$ instead of M. By Lemma 4.13 its regret is $O ( M _ { \sigma } \sqrt { T } )$ , so the inner term becomes $O ( d m M _ { \sigma } \sqrt { T } )$ . This is a change of constant, not of rate, because a balance learner’s regret was already proportional to the range of its payofs.

2. Outer layer. The field estimate acquires a variance $O ( d \sigma ^ { 2 } / ( Q h ^ { 2 } ) )$ , which enters through the step size and contributes $O ( D \sigma { \sqrt { d T / \varepsilon } } )$ at field accuracy $\varepsilon _ { i }$ hence $O ( D \sigma { \sqrt { d } } T ^ { 5 / 8 } )$ at the balanced choice $\varepsilon = \Theta ( T ^ { - 1 / 4 } )$ The bias is unchanged, which is the essential point: only the bias is charged at rate T, while the fluctuation is charged at rate $\sqrt { T }$ through (56).

3. Budget. Caching is forbidden, which doubles the inner call count and leaves the order $O ( d ( m +$ $\varepsilon ^ { - 1 } ) )$ unchanged.

4. Nothing else. The structural certificate of Section 4.1, the balance geometry of Theorem 4.8, the coeficients $\left( u _ { r } , v _ { r } , w _ { r } \right)$ , and the factor-revealing program (52) are statements about the true functions and are untouched by feedback noise. This is why the approximation factor is exactly $\alpha ^ { \star }$ at every noise level, with no restriction on how $\sigma$ may depend on $T ;$ only the additive terms see $\sigma$ at all.

The same three mechanisms recur, with the same arithmetic, in the batched algorithm of Appendix B and the bandit algorithm of Section 5. There the injection sampling of Appendix A already contributes a variance term of the same shape with M in place of $\sigma ,$ so the oracle noise is absorbed simply by replacing M with $M _ { \sigma }$ throughout, and no exponent changes at all.

## 5 Secondary extension to one-point bandit feedback

We now assume that round t returns a single scalar, the noisy value $\widehat { f _ { t } } ( p _ { t } )$ of the played point, while the learner still earns the true value $f _ { t } ( p _ { t } )$ . The post-decision value-oracle algorithm cannot be used directly because its update needs many values of the same current function. The remedy is the block simulation of Appendix A, with one additional requirement: every query location must itself be a feasible action. Noise requires no separate treatment here. In the bandit model each label is already realized on a randomly chosen round, so the observation is a random variable in any case; the oracle error is a second, conditionally independent centred perturbation of the same observation, and it enters the analysis exactly where the injection noise does, through the observation scale $M _ { \sigma }$ in the second moment (89) and in the balance regret. In particular it leaves the bias (88) untouched.

## 5.1 Interior geometry and feasible finite diferences

Assumption 5.1 (Known positive anchor). There are a known point ${ \bar { x } } \in K$ and a known $\rho > 0$ such that $\bar { x } _ { i } \geq \rho$ for every $i \in [ d ]$

If a coordinate is identically zero on K, it may be deleted before applying the assumption. For $\gamma \in ( 0 , 1 )$ , define the contracted set

$$
K _ { \gamma } = ( 1 - \gamma ) K + \gamma \bar { x } .\tag{70}
$$

It is compact and convex, is contained in K, and every $x \in K _ { \gamma }$ satisfies $x _ { i } \geq \gamma \rho$ . A projection oracle for $K _ { \gamma }$ follows from the oracle for K: project $( z - \gamma \bar { x } ) / ( 1 - \gamma )$ onto K and apply the afine map in (70).

Lemma 5.2 (Feasible probe points). Let $x \in K _ { \gamma }$ and $0 < h \leq \gamma \rho / 2$ . Every inner query $x \odot a$ $a \in [ 0 , 1 ] ^ { d }$ , lies in K. Moreover, for every $z = Y _ { \tau } ( x )$ and coordinate i, at least one $o f z - h e _ { i }$ and $z + h e _ { i }$ lies in the box $[ \mathbf { 0 } , x ]$ ; at x, the point $x - h e _ { i }$ lies in [0, x]. Consequently all inner and outer finite-diference queries can be played in $K$

Proof. All inner points and all trajectory points are coordinatewise between 0 and $x ,$ so downclosedness makes them feasible. Since $x _ { i } \geq \gamma \rho \geq 2 h$ , either $z _ { i } \geq h$ , in which case $z - h e _ { i } \in [ 0 , x ]$ , or $z _ { i } < h$ , in which case $z _ { i } + h < 2 h \leq x _ { i }$ and $z + h e _ { i } \in [ 0 , x ]$ . Finally $x _ { i } \geq 2 h$ implies $x - h e _ { i } \in [ 0 , x ]$ .

For a comparator $o \in K$ , let

$$
o ^ { \gamma } = ( 1 - \gamma ) o + \gamma \bar { x } \in K _ { \gamma } .\tag{71}
$$

The gradient bound gives, on every round,

$$
f _ { t } ( o ^ { \gamma } ) \geq f _ { t } ( o ) - B \gamma \left. \bar { x } - o \right. _ { 2 } \geq f _ { t } ( o ) - B \gamma D .\tag{72}
$$

Remark 5.3 (On the anchor margin). Assumption 5.1 deserves a careful reading, because it is easy to mistake for a restriction on the geometry when the real cost is quantitative.

It excludes almost no bodies. Delete every coordinate that is identically zero on K. Since K is down-closed and nonempty, any surviving coordinate i has some $y \in K$ with $y _ { i } > 0$ , and averaging d such points gives a point with all coordinates positive; down-closedness then puts the whole box below it in $K$ . So after the deletion an anchor exists for every K we consider, and what Assumption 5.1 really adds is that the learner knows one, together with a margin $\rho .$

The cost is the size of $\rho .$ . The quantity ρ is a genuine geometric parameter, not a constant. For a box $K = [ 0 , c ] ^ { d }$ one may take $\bar { x } = c { \bf 1 }$ and $\rho = c ,$ , so $B D / \rho = B \sqrt { d } .$ which is dominated by every other term of Corollary 5.5: the anchor costs nothing there. For the scaled simplex $K = \{ x \geq 0 : \textstyle \sum _ { i } x _ { i } \leq 1 \}$ , however, every feasible point has some coordinate at most $1 / d .$ , so $\rho \leq 1 / d$ and, since diam $( K ) = \Theta ( 1 )$ there, the anchor term is of order Bd, comparable to the Bd term already present in (83), so on the simplex specifically nothing is lost. The concern is not this example but the general one: every other coeficient in (83) is bounded by a $d ^ { 3 / 2 } \mathrm { - t y p e }$ quantity, whereas $B D / \rho$ is governed by a geometric parameter that d does not control. Fixing the dimension and letting K become thin in one direction sends $\rho \to 0$ and the bound to infinity while every other coeficient stays put. The bandit result is therefore not confined to boxes, but it degrades with the aspect ratio of the body in a way the post-decision result does not.

Origin of the $1 / \rho$ dependence. The assumption enters only in Lemma 5.2. A bandit probe must itself be played, so an axis-aligned finite diference of step $h$ at a point z requires room of size h along the relevant coordinate. The contraction $K _ { \gamma }$ provides that room uniformly at the price $B D \gamma T$ with $\gamma = \Theta ( h / \rho )$ . Thus the guarantee can degrade for a body that is thin in one direction. The post-decision model has no corresponding cost because its probes need only lie in the ambient cube.

## 5.2 Bandit algorithm

Algorithm 2 Batched one-point bandit extension of the online composition   
Require: $s = s _ { 0 } , r = r _ { 0 } , \lambda = \lambda _ { 0 } ;$ grid size $m ;$ block length $\overline { { H ; } }$ midpoint count $\overline { { Q ; } }$ diference step h;   
contraction $\gamma$ with $h \leq \gamma \rho / 2$   
1: Set $N = \lfloor T / H \rfloor$ , form $K _ { \gamma } ,$ and initialize the outer linear-optimization state in $K _ { \gamma }$ and dm   
balance learners with weights $\left( c _ { X } , c _ { Y } \right) = \left( 1 / r _ { 0 } , r _ { 0 } \right)$   
2: for $b = 1 , \dots , N$ do   
3: Using only past-block feedback and fresh pre-block randomness, obtain $x _ { b } \in K _ { \gamma }$ and the   
lifted inner set $S _ { b } ;$ set $a _ { b } = \phi _ { m } ( S _ { b } )$   
4: Set $y _ { b } = x _ { b } \odot a _ { b } , z _ { b } = Y _ { 1 } ( x _ { b } )$ , draw $Z _ { b } \sim \mathrm { B e r n o u l l i } ( \lambda )$ , and set $p _ { b } = z _ { b }$ if $Z _ { b } = 1$ and $p _ { b } = y _ { b }$   
otherwise.   
5: Construct at most J labeled feasible query points as in Appendix A and draw a uniform   
injection of the labels into the block.   
6: for each physical round in block b do   
7: Play its assigned query point if it has one; otherwise play $p _ { b }$ . Observe only the noisy   
value of that played point.   
8: end for   
9: Form unbiased estimates of all Double-Greedy marginals.   
10: Form the finite-diference/quadrature estimate ${ \widehat { q } } _ { b } ,$ unclipped.   
11: Update all weighted balance learners with the estimated marginals.   
12: Update $x _ { b + 1 } = \Pi _ { K _ { \gamma } } ( x _ { b } + \eta \widehat { q } _ { b } )$ , where $\eta = D / \sqrt { N V _ { h , Q } ^ { \mathrm { b } } }$ if $V _ { h , Q } ^ { \mathrm { b } } > 0 ;$ and $\eta = 0$ otherwise.   
13: end for   
14: On the fewer than H remaining rounds, play any fixed point in $K .$

In the bandit analysis, $p _ { t }$ denotes the actual action played on physical round $t ,$ whereas $p _ { b }$ denotes only the exploitation candidate held fixed during block b. The constant $C _ { 3 } ^ { \prime }$ appearing in $V _ { h , Q } ^ { \mathrm { b } }$ is the absolute constant of Lemma A.2, so the step size is computable from $B , L , M , \sigma , d , Q , h ;$ any computable upper bound on $V _ { h , Q } ^ { \mathrm { b } }$ may be substituted, at the cost of a constant factor in the regret. As in the direct algorithm, $\widehat { q _ { b } }$ is left unclipped (Remark 4.20).

Lemma 5.2 shows that Algorithm 2 uses only feasible actions. It also makes clear why the positive anchor is not needed in the post-decision oracle model: there, finite-diference locations need only remain in the ambient cube, whereas bandit queries must be played.

## 5.3 Bandit guarantee and parameter tradeof

Theorem 5.4 (One-point bandit guarantee). Assume Section 3 and Assumption 5.1. Let $m , Q \ge 1$ be integers, let J be defined by (84), and let the integer $H \in [ 1 , T ]$ and $h , \gamma \in ( 0 , 1 )$ satisfy

$$
h \leq \frac { \gamma \rho } { 2 } , \qquad J \leq \frac { H } { 2 } .\tag{73}
$$

Then Algorithm $\mathcal { Q }$ uses one-point bandit feedback from the noisy oracle (7) and, for every $o \in K$ satisfies

$$
\begin{array} { l } { \displaystyle \mathbb E \sum _ { t = 1 } ^ { T } f _ { t } ( p _ { t } ) \geq \alpha ^ { \star } \sum _ { t = 1 } ^ { T } f _ { t } ( o ) - O \Big ( d m M _ { \sigma } \sqrt { T H } \Big ) - O \Bigg ( \frac { B \sqrt { d } T } { m } \Bigg ) } \\ { \displaystyle \qquad - O \Big ( D \sqrt { T H V _ { h , Q } ^ { \mathrm { b } } } \Big ) - O ( D \Delta _ { h , Q } T ) - O ( B D \gamma T ) } \\ { \displaystyle \qquad - O \Bigg ( \frac { M d ( m + Q ) T } { H } \Bigg ) - O ( M H ) , } \end{array}\tag{74}
$$

where $\alpha ^ { \star }$ is defined in (10) and $\Delta _ { h , Q } , V _ { h , Q } ^ { \mathrm { b } }$ in (88)–(89). The two terms with an explicit M measure lost reward on exploration rounds and on the incomplete block, so they involve the true value bound rather than the observation scale.

Proof. Fix $s = s _ { 0 } , r = r _ { 0 } , \lambda = \lambda _ { 0 }$ and apply the post-decision value-oracle analysis at the meta-round level to the nonanticipating block-average sequence $\overline { { f } } _ { 1 } , \ldots , \overline { { f } } _ { N }$ with comparator $o ^ { \gamma } \in K _ { \gamma }$ . Because the adversary is oblivious, every function in block b is fixed before the pre-block randomization that selects $( x _ { b } , a _ { b } , p _ { b } )$ and the labeled query schedule.

The query list is frozen before the block. This is the step that licenses everything below, and it deserves to be spelled out, because the dm inner learners are sequentially coupled: the pair presented to learner i depends on the current decisions of learners $1 , \ldots , i - 1$ through $X _ { i - 1 } ^ { b }$ and $Y _ { i - 1 } ^ { b }$ . The coupling is nevertheless harmless here. Learner i’s mixed action is a function of its own past-block observations only, so at the start of block b the algorithm may draw $B _ { 1 } ^ { b }$ , then $B _ { 2 } ^ { b }$ , and so on through $B _ { d m } ^ { b }$ , without consulting any round of block b. These draws determine the chains $X _ { 0 } ^ { b } \subseteq \cdots \subseteq X _ { d m } ^ { b }$ and $Y _ { 0 } ^ { b } \supseteq \cdots \supseteq Y _ { d m } ^ { b }$ , hence the 4dm inner locations of (40); the outer state $x _ { b }$ determines the 4dQ field locations of (58). Sequential coupling among the learners is a dependence on decisions, not on current feedback, and only the latter would violate nonadaptivity.

Accordingly, let

$$
\mathcal { F } _ { b } = \sigma ( \mathrm { t h e ~ p a s t ~ t h r o u g h ~ b l o c k ~ } b - 1 ; ~ f _ { 1 } , \dotsc , f _ { T } ;
$$

the pre-block draws fixing $( x _ { b } , a _ { b } , p _ { b } )$ and the labeled list $u _ { 1 } , \ldots , u _ { J _ { b } } )$

(75)

so that the list, the exploitation point $p _ { b } .$ , and the state $x _ { b }$ are all ${ \mathcal { F } } _ { b } .$ -measurable. The injection $\iota _ { b }$ is drawn before the block begins as well, but independently of $\mathcal { F } _ { b }$ and deliberately not included in it: nonadaptivity requires only that the schedule use no feedback from block $b ,$ and the analysis below needs $\iota _ { b }$ to remain uniform conditionally on $\mathcal { F } _ { b }$ . The block rewards and all oracle errors are likewise realized after $\mathcal { F } _ { b }$

Each label is an unbiased sample of the block average. Fix a label $j$ at location $u _ { j }$ and condition on $\mathcal { F } _ { b }$ . Writing $\mathcal { G } _ { b , j }$ for the sigma-field generated by $\mathcal { F } _ { b }$ together with the realized round $\iota _ { b } ( j )$ and all earlier calls,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \widehat { f } _ { \iota _ { b } ( j ) } ( u _ { j } ) \bigm | \mathcal { F } _ { b } \right] = \mathbb { E } \Big [ \mathbb { E } \left[ \widehat { f } _ { \iota _ { b } ( j ) } ( u _ { j } ) \bigm | \mathcal { G } _ { b , j } \right] \Big | \mathcal { F } _ { b } \Big ] = \mathbb { E } \big [ f _ { \iota _ { b } ( j ) } ( u _ { j } ) \bigm | \mathcal { F } _ { b } \big ] } \\ & { \qquad = \displaystyle \sum _ { t \in I _ { b } } \mathbb { P } \big [ \iota _ { b } ( j ) = t \bigm | \mathcal { F } _ { b } \bigm ] f _ { t } ( u _ { j } ) = \frac { 1 } { H } \displaystyle \sum _ { t \in I _ { b } } f _ { t } ( u _ { j } ) = \overline { { f } } _ { b } ( u _ { j } ) . } \end{array}\tag{76}
$$

Three separate facts are used, and it is worth naming them. The second equality is conditional unbiasedness of the oracle (7), applied after the round has been revealed, so the noise is centred whatever round the label landed on. The fourth is that a uniformly random injection has uniform marginals, $\mathbb { P } [ \iota _ { b } ( j ) = t ] = 1 / H$ for every $t \in I _ { b } ,$ , which is where the block average comes from even though $f _ { t }$ genuinely varies across the block. The third is where obliviousness is used: the functions $f _ { t } , t \in I _ { b } ,$ are fixed before play and in particular are not chosen in response to $\iota _ { b } ,$ so they may be treated as constants inside the conditional expectation. Against an adaptive adversary this last step fails and (76) is false, which is why Section 3 assumes an oblivious adversary. Equation (76) is (86), and everything the block analysis needs, namely unbiased marginals for Lemma 4.13, an unbiased-up-to-quadrature field for Lemma A.2, follows from it by linearity.

With this in hand, Lemma 4.13, Proposition 4.17 and Lemma 4.16 give

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle \sum _ { b = 1 } ^ { N } \overline { { f } } _ { b } ( x _ { b } \odot a _ { b } ) \ge u _ { r } \mathbb { E } \sum _ { b } \overline { { f } } _ { b } ( x _ { b } \odot \boldsymbol { \sigma } ^ { \gamma } ) + v _ { r } \mathbb { E } \sum _ { b } \overline { { f } } _ { b } ( x _ { b } ) } \\ & { \qquad + w _ { r } \displaystyle \sum _ { b } \overline { { f } } _ { b } ( \mathbf { 0 } ) - E _ { \mathrm { b o x } } ^ { ( N ) } , \qquad E _ { \mathrm { b o x } } ^ { ( N ) } = O ( d m M _ { \sigma } \sqrt { N } ) + O \bigg ( \frac { B \sqrt { d } N } { m } \bigg ) . } \end{array}\tag{77}
$$

The endpoint inequality of Lemma 4.7 applies to each ${ \overline { { f } } } _ { b } ,$ , so combining it with (77) and the choice of λ in (51) gives

$$
\mathbb { E } \sum _ { b = 1 } ^ { N } \overline { { f } } _ { b } ( p _ { b } ) \geq \alpha ^ { \star } \sum _ { b } \overline { { f } } _ { b } ( o ^ { \gamma } ) - \lambda \mathbb { E } \sum _ { b } \left. \widetilde { q } _ { s } ( \overline { { f } } _ { b } , x _ { b } ) , o ^ { \gamma } - x _ { b } \right. - E _ { \mathrm { b o x } } ^ { ( N ) } .\tag{78}
$$

Recall $\mathcal { F } _ { b }$ from (75): the pre-block random choices $( x _ { b } , a _ { b } , p _ { b } )$ and the labeled query list have been fixed, but the random injection, the block rewards, and the oracle errors have not been realized. Set $q _ { b } ^ { h , Q } = \mathbb { E } [ \widehat { q } _ { b } \mid \mathcal { F } _ { b } ]$ . Lemma 4.21 applied to the meta-rounds with the stochastic vectors $\widehat { q } _ { b }$ gives

$$
\mathbb { E } \sum _ { b = 1 } ^ { N } \left. \widehat { q } _ { b } , o ^ { \gamma } - x _ { b } \right. \le \frac { D ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \sum _ { b = 1 } ^ { N } \mathbb { E } \left. \widehat { q } _ { b } \right. _ { 2 } ^ { 2 } \le D \sqrt { N V _ { h , Q } ^ { \mathrm { b } } } .\tag{79}
$$

The diference $q _ { b } ^ { h , Q } - \widehat { q } _ { b }$ is a martingale diference and has zero expected inner product with the $\mathcal { F } _ { b }$ -measurable vector $o ^ { \gamma } - x _ { b }$ , so with (88),

$$
\mathbb { E } \sum _ { b } \left. \widetilde { q } _ { s } ( \overline { { f } } _ { b } , x _ { b } ) , o ^ { \gamma } - x _ { b } \right. \le D \sqrt { N V _ { h , Q } ^ { \mathrm { b } } } + D N \Delta _ { h , Q } .\tag{80}
$$

It remains to translate meta-round reward into physical reward. The reward credited to the learner is the true value of the point it plays, so the oracle noise plays no role in this step. By (87), the expected reward on the complete blocks is at least $\begin{array} { r } { ( H - J ) \mathbb { E } \sum _ { b } \overline { { f } } _ { b } ( p _ { b } ) } \end{array}$ . Multiplying (78) by $H - J ,$ upper-bounding every error multiplier by H and using $N H \leq T$ gives the first four error terms in (74). Replacing the comparator coeficient $( H - J ) \alpha ^ { \star }$ by $H \alpha ^ { \star }$ costs at most $M J N = O ( M d ( m + Q ) T / H )$ ; (72) costs at most $B D \gamma T ;$ ; and the incomplete block contains fewer than H rounds and costs at most MH relative to the comparator. This proves the theorem.

Corollary 5.5 (Explicit bandit rate). Choose a numerical constant $C _ { 4 }$ large enough that the second condition in (73) holds for the parameters below. Assume

$$
T > \left( { \frac { 2 } { \rho } } \right) ^ { 6 } , \qquad \left\lceil C _ { 4 } d T ^ { 1 / 3 } \right\rceil \leq T ,\tag{81}
$$

and take

$$
m = Q = \lceil T ^ { 1 / 6 } \rceil , \qquad h = T ^ { - 1 / 6 } , \qquad \gamma = \frac { 2 T ^ { - 1 / 6 } } { \rho } , \qquad H = \left\lceil C _ { 4 } d T ^ { 1 / 3 } \right\rceil .\tag{82}
$$

The first condition in (81) ensures $\gamma = 2 T ^ { - 1 / 6 } / \rho \in ( 0 , 1 )$ , and the second ensures $H \leq T$ . Then Algorithm 2 satisfies

$$
\mathrm { R e g } _ { \alpha ^ { \star } } ( T ) = O \left( \left[ M _ { \sigma } d ^ { 3 / 2 } + B d + D d ( B + L ) + \frac { B D } { \rho } \right] T ^ { 5 / 6 } \right) .\tag{83}
$$

For smaller horizons the trivial MT bound can be used. The coeficient $B D / \rho$ is discussed in Remark 5.3; it is the one place where the bandit guarantee is materially weaker than the post-decision one, since $D \leq \sqrt { d }$ bounds the geometric factor in every other term while $D / \rho$ is bounded by no function of d. In particular the additive term is $o ( T )$ for fixed problem parameters, and the approximation factor is the same 0.401 as in the post-decision full-information value-oracle model. As in Corollary 4.2, the noise level enters only through $M _ { \sigma } = M + \sigma$ , so the exponent $5 / 6$ is the same for every noise level held fixed as T grows.

Proof. The choices in (82) give $\Delta _ { h , Q } = O ( ( B + L ) \sqrt { d } T ^ { - 1 / 6 } )$ and $V _ { h , Q } ^ { \mathrm { b } } = O ( B ^ { 2 } + d ( B + L ) ^ { 2 } T ^ { - 1 / 3 } +$ $d M _ { \sigma } ^ { 2 } T ^ { 1 / 6 } )$ . Substituting into (74): the inner learning term is $O ( M _ { \sigma } d ^ { 3 / 2 } T ^ { 5 / 6 } )$ , the grid term is $O ( B \sqrt { d } T ^ { 5 / 6 } )$ , the quadrature bias term is $O ( D ( B + L ) \sqrt { d } T ^ { 5 / 6 } )$ , the contraction term is $O ( ( B D / \rho ) T ^ { 5 / 6 } )$ and the exploration term is $O ( M T ^ { 5 / 6 } )$ . The stochastic-field term is $O ( D B \sqrt { d } T ^ { 2 / 3 } +$ $D d ( B + L ) T ^ { 1 / 2 } + D M _ { \sigma } d T ^ { 3 / 4 } )$ and the incomplete-block term is $O ( M d T ^ { 1 / 3 } )$ ; both are dominated by the displayed $T ^ { 5 / 6 }$ bound because $K \subseteq [ 0 , 1 ] ^ { d }$ implies $D \leq { \sqrt { d } }$ □

## 6 Conclusion

We show that the best known constructive ofline approximation factor 0.401 can also be achieved in an adversarial online setting with sublinear regret. An exact weighted balance theorem and an amortized online composition preserve the asymmetric ofline coeficients. The direct post-decision value-oracle implementation has $O ( T ^ { 3 / 4 } )$ regret with $O ( d T ^ { 1 / 4 } )$ calls per round; batching gives the limited-query tradeof, and randomized blocking gives ${ \dot { O } } ( T ^ { 5 / 6 } )$ one-point bandit regret under the positive-anchor condition. These guarantees also allow conditionally unbiased bounded oracle noise, which changes the additive terms but not the approximation factor.

## References

[1] A. A. Bian, K. Y. Levy, A. Krause, and J. Buhmann. Continuous DR-submodular maximization: Structure and algorithms. In Advances in Neural Information Processing Systems, 2017.

[2] A. A. Bian, B. Mirzasoleiman, J. M. Buhmann, and A. Krause. Guaranteed non-convex optimization: Submodular maximization over continuous domains. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR 54:111–120, 2017.

[3] M. Staib and S. Jegelka. Robust budget allocation via continuous submodular functions. In Proceedings of the 34th International Conference on Machine Learning, PMLR 70:3230–3240, 2017.

[4] Y. Bian, J. M. Buhmann, and A. Krause. Optimal continuous DR-submodular maximization and applications to provable mean field inference. In Proceedings of the 36th International Conference on Machine Learning, PMLR 97:644–653, 2019.

[5] M. Feldman, J. Naor, and R. Schwartz. A unified continuous greedy algorithm for submodular maximization. In Proceedings of the 52nd IEEE Symposium on Foundations of Computer Science, 2011.

[6] U. Feige, V. S. Mirrokni, and J. Vondr´ak. Maximizing non-monotone submodular functions. SIAM Journal on Computing, 40(4):1133–1153, 2011.

[7] N. Buchbinder, M. Feldman, J. Naor, and R. Schwartz. A tight linear time (1/2)-approximation for unconstrained submodular maximization. SIAM Journal on Computing, 44(5):1384–1402, 2015.

[8] A. Ene and H. L. Nguyen. Constrained submodular maximization: Beyond 1/e. In Proceedings of the 57th IEEE Symposium on Foundations of Computer Science, 2016.

[9] N. Buchbinder and M. Feldman. Constrained submodular maximization via a nonsymmetric technique. Mathematics of Operations Research, 44(3):988–1005, 2019.

[10] N. Buchbinder and M. Feldman. Constrained submodular maximization via new bounds for DR-submodular functions. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, 2024. Full version: arXiv:2311.01129.

[11] H. Jadav, R. Singh, and V. Aggarwal. Stronger approximation guarantees for non-monotone γ-weakly DR-submodular maximization. Transactions on Machine Learning Research, 2026.

[12] S. Oveis Gharan and J. Vondr´ak. Submodular maximization by simulated annealing. In Proceedings of the 22nd Annual ACM-SIAM Symposium on Discrete Algorithms, pages 1098– 1116, 2011.

[13] B. Qi. On maximizing sums of non-monotone submodular and linear functions. Algorithmica, 86(4):1080–1134, 2024.

[14] L. Mualem and M. Feldman. Using partial monotonicity in submodular maximization. In Advances in Neural Information Processing Systems, 2022.

[15] D. Blackwell. An analog of the minimax theorem for vector payofs. Pacific Journal of Mathematics, 6(1):1–8, 1956.

[16] M. Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th International Conference on Machine Learning, pages 928–935, 2003.

[17] T. Roughgarden and J. R. Wang. An optimal algorithm for online unconstrained submodular maximization. In Proceedings of the 31st Conference on Learning Theory, PMLR 75:1307–1325, 2018.

[18] R. Niazadeh, T. Roughgarden, and J. R. Wang. Optimal algorithms for continuous nonmonotone submodular and DR-submodular maximization. In Advances in Neural Information Processing Systems, 2018.

[19] N. K. Thang and A. Srivastav. Online non-monotone DR-submodular maximization. In Proceedings of the AAAI Conference on Artificial Intelligence, 35(11):9868–9876, 2021.

[20] L. Chen, H. Hassani, and A. Karbasi. Online continuous submodular maximization. In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics, PMLR 84:1896–1905, 2018.

[21] P. G. Sessa, I. Bogunovic, A. Krause, and M. Kamgarpour. Online submodular resource allocation with applications to rebalancing shared mobility systems. In Proceedings of the 38th International Conference on Machine Learning, PMLR 139:9455–9464, 2021.

[22] L. Mualem and M. Feldman. Resolving the approximability of ofline and online non-monotone DR-submodular maximization over general convex sets. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, PMLR 206:2542–2564, 2023.

[23] L. Chen, M. Zhang, H. Hassani, and A. Karbasi. Black box submodular maximization: Discrete and continuous settings. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics, PMLR 108:1058–1070, 2020.

[24] Q. Zhang, Z. Deng, Z. Chen, K. Zhou, H. Hu, and Y. Yang. Online learning for non-monotone DR-submodular maximization: From full information to bandit feedback. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, PMLR 206:3515–3537, 2023.

[25] Z. Wan, J. Zhang, W. Chen, X. Sun, and Z. Zhang. Bandit multi-linear DR-submodular maximization and its applications on adversarial submodular bandits. In Proceedings of the 40th International Conference on Machine Learning, PMLR 202:35491–35524, 2023.

[26] M. Pedramfar and V. Aggarwal. From linear to linearizable optimization: A novel framework with applications to stationary and non-stationary DR-submodular optimization. In Advances in Neural Information Processing Systems, 2024.

[27] M. Pedramfar, C. Quinn, and V. Aggarwal. Uniform wrappers: Bridging concave to quadratizable functions in online optimization. Advances in Neural Information Processing Systems, 38:60109–60145, 2025.

[28] M. Pedramfar and V. Aggarwal. γ-weakly θ-up-concavity: Linearizable non-convex optimization with applications to DR-submodular and OSS functions. arXiv preprint arXiv:2602.13506, 2026.

[29] M. Pedramfar, Y. Y. Nadew, C. J. Quinn, and V. Aggarwal. Unified projection-free algorithms for adversarial DR-submodular optimization. In Proceedings of the Twelfth International Conference on Learning Representations, 2024.

[30] Y. Lu, H. Jadav, M. Pedramfar, R. Singh, and V. Aggarwal. Upper-linearizability of online non-monotone DR-submodular maximization over down-closed convex sets. In Proceedings of the 43rd International Conference on Machine Learning, 2026. Full version: arXiv:2602.20578v2.

## A Block simulation of the query list

Both the limited-query frontier of Theorem 4.4 and the bandit conversion of Section 5 rest on a single observation: once the pre-round outer state and the inner binary decisions have been chosen, the complete list of oracle calls needed for one update is nonadaptive, so its labels can be assigned to distinct physical rounds in advance. Noise changes nothing about this: the list is determined before any value is observed, and each label simply returns a noisy value instead of an exact one. Blocking and feedback-conversion ideas are standard in online DR-submodular optimization [26, 29, 30]; the feature used here is a uniform random injection of an entire nonadaptive query list. This appendix develops that machinery once. The two applications then difer in exactly one respect, recorded in Appendix B: in the post-decision model the queries are issued in addition to the played action, so there is no exploration loss and no feasibility requirement, whereas in the bandit model every query must itself be played.

Blocks and block averages. Fix an integer block length H and let $N = \lfloor T / H \rfloor$ . For $b \in [ N ]$ 2 define the bth complete block and its average reward by

$$
{ \cal I } _ { b } = \{ ( b - 1 ) H + 1 , \ldots , b H \} , { \overline { { { f } } } } _ { b } ( x ) = \frac { 1 } { H } \sum _ { t \in { \cal I } _ { b } } f _ { t } ( x ) .
$$

The average $f _ { b }$ is again nonnegative, L-smooth, DR-submodular, bounded by $M$ , and has gradient norm at most $B ,$ since all of these properties are preserved by convex combinations.

At the start of block $b ,$ the algorithm chooses the state $x _ { b } ,$ all inner binary decisions, and the exploitation point $p _ { b }$ using only previous-block feedback and fresh pre-block randomization, with no feedback from the current block. It then constructs a labeled list of oracle locations:

1. the two Double-Greedy chains require $O ( d m )$ values of $\overline { { f } } _ { b } ( x _ { b } \odot a )$ ;

2. the outer field uses $Q$ midpoint nodes, with two labels per node and coordinate for a one-sided diference of step $h ,$ , together with $Q$ independent labeled diference pairs per coordinate at $x _ { b }$ for the term $\zeta _ { s } \nabla \overline { { f } } _ { b } ( x _ { b } )$

Repeated labels at the same geometric point are retained as distinct labels; this keeps the second moment of the vector estimate proportional to $1 / Q$ . For a universal constant $C _ { 1 }$ , set

$$
J = \left\lceil C _ { 1 } d ( m + Q ) \right\rceil ,\tag{84}
$$

so the list has at most J labels. For notational simplicity the analysis uses $J$ as a uniform upper bound on the number of exploration rounds.

Choose a uniformly random injection $\iota _ { b } : [ J _ { b } ] \to I _ { b }$ , where $J _ { b } \leq J$ is the actual list length (we write ι rather than $\sigma ,$ which now denotes the oracle noise level). On round $\iota _ { b } ( j )$ play or query the point attached to label $j$ and record the value the oracle returns; on every unassigned round play $p _ { b }$ . The complete schedule is sampled before the block begins, so every physical action is nonanticipating.

Lemma A.1 (Variance under a random injection). Let $1 \leq J \leq H$ and let $\iota : [ J ]  [ H ]$ be a uniformly random injection. For deterministic numbers $v _ { j , t } \in [ 0 , M ]$ and weights $w _ { 1 } , \dots , w _ { J } \in \mathbb { R }$ ，

$$
\mathrm { V a r } \left( \sum _ { j = 1 } ^ { J } w _ { j } v _ { j , \iota ( j ) } \right) \leq C M ^ { 2 } \sum _ { j = 1 } ^ { J } w _ { j } ^ { 2 }\tag{85}
$$

for a universal constant $C .$

Proof. Extend ι to a uniform permutation of $[ H ]$ by adding $H - J$ dummy labels with weight zero. The Poincar´e inequality for the random-transposition chain bounds the variance by a constant multiple of $H ^ { - 1 }$ times the sum, over label pairs, of the squared change caused by swapping their assigned times. Swapping labels $j$ and $k$ changes the weighted sum by at most $M ( | w _ { j } | + | w _ { k } | )$ Using $( | w _ { j } | + | w _ { k } | ) ^ { 2 } \leq 2 ( w _ { j } ^ { 2 } + w _ { k } ^ { 2 } )$ and summing over the $H - 1$ partners of each label gives (85).

Lemma A.2 (Block-oracle simulation). Condition on the history before block b and on the labeled query list. For a label j with location $u _ { j }$ , the observation $\widehat { f } _ { \iota _ { b } ( j ) } ( \boldsymbol { u } _ { j } )$ returned on the round to which the label is assigned satisfies

$$
\begin{array} { r } { \mathbb { E } \big [ \widehat { f } _ { \iota _ { b } ( j ) } ( u _ { j } ) \big ] = \overline { { f } } _ { b } ( u _ { j } ) , \qquad \widehat { f } _ { \iota _ { b } ( j ) } ( u _ { j } ) \in [ - \sigma , M _ { \sigma } ] . } \end{array}\tag{86}
$$

If $J \leq H$ and the query points are played rather than issued in addition to the action, the expected reward collected in the block is at least

$$
\left( 1 - { \frac { J } { H } } \right) \sum _ { t \in I _ { b } } f _ { t } ( p _ { b } ) .\tag{87}
$$

Let $\widehat { q _ { b } }$ be the field estimate obtained from the labeled finite-diference samples, using $Q$ midpoint nodes and $Q$ repeated diferences at $x _ { b }$ . Then, for constants depending only on $s ,$

$$
\big \vert \big \vert \mathbb { E } [ \widehat { q } _ { b } ] - \widetilde { q } _ { s } ( \overline { { f } } _ { b } , x _ { b } ) \big \vert \big \vert _ { 2 } \leq C _ { 2 } \sqrt { d } \left( L h + \frac { B + L } { Q } \right) = \Delta _ { h , Q } ,\tag{88}
$$

$$
\mathbb { E } \left\| \widehat { q } _ { b } \right\| _ { 2 } ^ { 2 } \leq C _ { 3 } ^ { \prime } \left( B ^ { 2 } + \Delta _ { h , Q } ^ { 2 } + \frac { d M _ { \sigma } ^ { 2 } } { Q h ^ { 2 } } \right) = : V _ { h , Q } ^ { \mathrm { b } } .\tag{89}
$$

Here $\Delta _ { h , Q }$ is the same bias functional as in (59), while $C _ { 3 } ^ { \prime }$ is a universal constant distinct from the $C _ { 3 }$ of Lemma 4.22. The essential diference between the block second moment $V _ { h , Q } ^ { \mathrm { b } }$ and the direct one (60) is that the former carries $M _ { \sigma } ^ { 2 }$ where the latter carries $\sigma ^ { 2 }$ , because a block estimate fluctuates both through the injection and through the oracle; the two displays also write $B ^ { 2 }$ and $\Gamma _ { s } ^ { 2 }$ for the mean scale, which difer by at most the absolute factor $( 1 + \zeta _ { s } ) ^ { 2 } \leq 4$

Proof. For a fixed label, $\iota _ { b } ( j )$ is uniform on $I _ { b } ,$ so $\mathbb { E } [ f _ { \iota _ { b } ( j ) } ( u _ { j } ) ] = \overline { { f } } _ { b } ( u _ { j } )$ ; the oracle error is centred conditionally on the round to which the label is assigned, so the tower rule gives (86), and the stated range is (7). The image of a uniform injection is a uniform J-subset of the block, so every round is left for exploitation with probability at least $1 - J / H$ ; since the true rewards actually earned on exploration rounds are nonnegative, the noise corrupts what is observed but not what is earned. Linearity of expectation gives (87).

Taking expectations in each sampled diference produces the corresponding one-sided finite diference of ${ \overline { { f } } } _ { b } .$ the injection contributes the block average and the oracle contributes nothing, by (86). Smoothness gives bias $O ( L h )$ , and the midpoint argument from Lemma 4.22 gives $O ( ( B + L ) / Q )$ quadrature bias; summing over coordinates proves (88). Note that the noise leaves the bias untouched, which is why (88) has no σ in it.

The second moment is the step on which the claim “noise does not change the exponent” rests, so we derive it in full. Fix a coordinate i and consider the path term of $\widehat { q _ { b } }$ ; the repeated estimate at $x _ { b }$ is identical in form. That coordinate is a weighted average

$$
W _ { i } = \frac { 1 } { Q } \sum _ { k = 1 } ^ { Q } c _ { k } \frac { \widehat { v } _ { k } - { \widehat { v } _ { k } ^ { \prime } } } { h } , \qquad | c _ { k } | \le 1 ,
$$

of $Q$ observed one-sided diferences, where each of the 2Q observations is a distinct label. Write $\widehat { v } = v + \epsilon$ , splitting an observation into the true value $v = f _ { \iota _ { b } ( \cdot ) } ( \cdot )$ at the round the label landed on and the oracle error ϵ. This splits $W _ { i }$ into an injection part and an oracle part,

$$
W _ { i } = \underbrace { \frac { 1 } { Q } \sum _ { k } c _ { k } \frac { v _ { k } - v _ { k } ^ { \prime } } { h } } _ { W _ { i } ^ { \mathrm { i n j } } } + \underbrace { \frac { 1 } { Q } \sum _ { k } c _ { k } \frac { \epsilon _ { k } - \epsilon _ { k } ^ { \prime } } { h } } _ { W _ { i } ^ { \mathrm { o r c } } } ,
$$

and Va $\cdot ( W _ { i } ) \leq 2 \operatorname { V a r } ( W _ { i } ^ { \mathrm { i n j } } ) + 2 \operatorname { V a r } ( W _ { i } ^ { \mathrm { o r c } } )$

For $W _ { i } ^ { \mathrm { i n j } }$ , the values $v _ { k } , v _ { k } ^ { \prime }$ lie in $[ 0 , M ]$ and are read of a uniformly random injection, so Lemma $\mathrm { A . 1 }$ applies with 2Q weights of magnitude $O ( 1 / ( Q h ) )$ and gives

$$
\mathrm { V a r } ( W _ { i } ^ { \mathrm { i n j } } ) \leq C M ^ { 2 } \cdot 2 Q \cdot O \left( { \frac { 1 } { Q ^ { 2 } h ^ { 2 } } } \right) = O \left( { \frac { M ^ { 2 } } { Q h ^ { 2 } } } \right) .
$$

For $W _ { i } ^ { \mathrm { o r c } }$ , the $2 Q$ errors are conditionally independent and centred with $| \epsilon | \le \sigma$ , so their variances simply add:

$$
\mathrm { V a r } ( W _ { i } ^ { \mathrm { o r c } } ) \leq \frac { 1 } { Q ^ { 2 } h ^ { 2 } } \sum _ { k = 1 } ^ { Q } \bigl ( \mathrm { V a r } ( \epsilon _ { k } ) + \mathrm { V a r } ( \epsilon _ { k } ^ { \prime } ) \bigr ) \leq \frac { 2 Q \sigma ^ { 2 } } { Q ^ { 2 } h ^ { 2 } } = \frac { 2 \sigma ^ { 2 } } { Q h ^ { 2 } } .
$$

Adding the two and using $M ^ { 2 } + \sigma ^ { 2 } \le ( M + \sigma ) ^ { 2 } = M _ { \sigma } ^ { 2 }$ gives $\operatorname { V a r } ( W _ { i } ) = O ( M _ { \sigma } ^ { 2 } / ( Q h ^ { 2 } ) )$ per coordinate, hence $O ( d M _ { \sigma } ^ { 2 } / ( Q h ^ { 2 } ) )$ after summing over the d coordinates. The point of the display is that the two sources of fluctuation are of exactly the same shape and are both damped by the same factor $1 / Q ;$ the oracle noise therefore cannot change the exponent unless it changes the order of $M _ { \sigma } ;$ and in the frequently assumed case $\sigma \leq M$ it does not even change the constant by more than a factor of four, since then $M _ { \sigma } ^ { 2 } \le 4 M ^ { 2 }$

Finally, the conditional mean difers from a vector of norm at most $( 1 + \zeta _ { s } ) B$ by at most $\Delta _ { h , Q }$ Summing the coordinate variances and adding the squared norm of this mean proves (89). □

The inner layer also remains valid with sampled values, and no new lemma is needed: the marginal estimate for a lifted element is now a diference of four observations, each drawn from a uniformly injected round and each corrupted by oracle noise, so it lies in $[ - 2 M _ { \sigma } , 2 M _ { \sigma } ]$ and, by (86) together with $( 7 )$ , is conditionally unbiased for the corresponding true marginal of ${ \overline { { f } } } _ { b } .$ . Lemma 4.13 applies with $\bar { M } = 2 M _ { \sigma }$ . Applying it independently to the dm lifted elements gives an $O ( d m M _ { \sigma } \sqrt { N } )$ inner learning term, which for $\sigma = 0$ is the exact-oracle bound. Note that the observed marginal estimates need not satisfy $\widehat { g _ { b } } ^ { + } + \widehat { g _ { b } } ^ { - } \ge 0$ : admissibility is required only of the true marginals of the submodular function ${ \overline { { f } } } _ { b } .$ , and those are what Proposition 4.17 manipulates.

## B Proof of the limited-query frontier

Proof of Theorem $4 { \cdot } 4 .$ All oracle locations needed for one update are fixed before any value feedback from the current objective is received. We may therefore hold the algorithmic state fixed over a block, play its reward-bearing action on every physical round, and distribute the post-decision calls across the block. Unlike in the bandit conversion, these calls do not replace the played action, so the block reward is exactly $H \overline { { f } } _ { b } ( p _ { b } )$ and no feasibility restriction on the probe points is needed. Throughout, every observation is a call to the noisy oracle (7); the block machinery of Appendix A is stated in that model, so noise enters only through $M _ { \sigma }$

Set

$$
\ell = \frac { 1 - 4 \delta } { 5 } , \qquad \xi = \frac { 1 + \delta } { 5 } , \qquad k = \lceil T ^ { \delta } \rceil ,\tag{90}
$$

and take

$$
H = \left\lceil C _ { \mathrm { q } } d T ^ { \ell } \right\rceil , \qquad m = Q = \lceil T ^ { \xi } \rceil , \qquad h = \frac { T ^ { - \xi } } { 2 } ,
$$

where $C _ { \mathrm { q } }$ is a suficiently large numerical constant. These are exactly the parameters (13) of the theorem: the field accuracy is $\varepsilon _ { \delta } = T ^ { - \xi } = T ^ { - ( 1 + \delta ) / 5 }$ , and $h = \varepsilon _ { \delta } / 2 , Q = \lceil \varepsilon _ { \delta } ^ { - 1 } \rceil$ is the direct algorithm’s own convention, so the only thing this proof changes about the query list is the value of ε. Note $\ell \geq 0$ exactly because $\delta \leq 1 / 4$ , and that $\delta = 1 / 4$ gives $\ell = 0 , \varepsilon _ { 1 / 4 } = T ^ { - 1 / 4 }$ , recovering the parameters of Corollary 4.2 with a block length constant in T. Divide the horizon into $N = \lfloor T / H \rfloor$ complete blocks and adopt the notation of Appendix A. At the start of block $b ,$ choose the outer state, all inner binary decisions, and the exploitation action $p _ { b }$ from previous-block feedback and fresh pre-block randomization only, and play $p _ { b }$ throughout $I _ { b }$

One update on $f _ { b }$ needs a nonadaptive list of $J = { \cal O } ( d ( m + Q ) ) = { \cal O } ( d T ^ { \xi } )$ value labels. Because $\delta + \ell = \xi$ , the choice of $C _ { \mathrm { q } }$ ensures $J \le k H$ . Partition the labels into k groups of size at most H and independently inject each group uniformly without replacement into the H rounds. After the played action on a round is committed, issue the labels assigned to that round. This uses at most k oracle calls per round, and every observed label at location u has expectation $\overline { { f } } _ { b } ( u )$ by (86), the two sources of randomness being which round realizes the label and the oracle error on that call. Both are centred.

Use the sampled values to form all Double-Greedy marginals and the outer finite-diference estimator $\widehat { q _ { b } }$ . Applying Lemma $\mathrm { A . 1 }$ within each group and using independence across groups, Lemma A.2 gives

$$
\big \vert \big \vert \mathbb { E } [ \widehat { q } _ { b } ] - \widetilde { q } _ { s } ( \overline { { f } } _ { b } , x _ { b } ) \big \vert \big \vert _ { 2 } \leq O \bigg ( \sqrt { d } \left[ L h + \frac { B + L } { Q } \right] \bigg ) = \Delta _ { h , Q } ,\tag{91}
$$

$$
\mathbb { E } \left. \widehat { q } _ { b } \right. _ { 2 } ^ { 2 } \leq O \left( B ^ { 2 } + \Delta _ { h , Q } ^ { 2 } + \frac { d M _ { \sigma } ^ { 2 } } { Q h ^ { 2 } } \right) = V _ { h , Q } ^ { \mathrm { b } } .\tag{92}
$$

The sampled Double-Greedy marginals are bounded by $2 M _ { \sigma }$ and unbiased, so Lemma 4.13 gives $O ( M _ { \sigma } \sqrt { N } )$ expected balance regret per lifted element, and Lemma 4.21 applied to the meta-rounds, with step size $\eta = D / \sqrt { N V _ { h , Q } ^ { \mathrm { b } } }$ and the splitting (56), incurs $O ( D _ { \sqrt { N V _ { h , Q } ^ { \mathrm { b } } } } )$ expected linear regret plus the bias term already accounted for.

Applying the factor-revealing composition of Section 4.3 to the block averages $\overline { { f } } _ { 1 } , \ldots , \overline { { f } } _ { N }$ and multiplying by H therefore gives

$$
\begin{array} { r l } & { { \mathrm { R e g } } _ { \alpha ^ { \star } } ( T ) \leq O \Big ( d m M _ { \sigma } \sqrt { T H } \Big ) + O \bigg ( \frac { B \sqrt { d } T } { m } \bigg ) + O \Big ( D \sqrt { T H V _ { h , Q } ^ { \mathrm { b } } } \Big ) } \\ & { \qquad + O ( D \Delta _ { h , Q } T ) + O ( M H ) , } \end{array}\tag{93}
$$

where the final term covers the incomplete block. With the parameter choices above, $\Delta _ { h , Q } = O ( T ^ { - \xi } )$ and $V _ { h , Q } ^ { \mathrm { b } } = O ( 1 + T ^ { \xi } )$ up to problem-dependent factors, the latter now carrying $M _ { \sigma } ^ { 2 }$ in place of $M ^ { 2 }$ The first, second, and bias terms have exponent

$$
\xi + \frac { 1 + \ell } { 2 } = 1 - \xi = \frac { 4 - \delta } { 5 } ,
$$

the stochastic-field term has exponent $( 1 + \ell + \xi ) / 2 = ( 7 - 3 \delta ) / 1 0$ , and the incomplete-block term has exponent $\ell ;$ both of the latter are smaller than $( 8 - 2 \delta ) / 1 0 = ( 4 - \delta ) / 5$ . Substituting in (93) proves (14). The noise level appears only inside $M _ { \sigma }$ , which sits in the constant, so the frontier holds at every noise level and its exponents are unafected by any $\sigma$ held fixed as $T$ grows. □

## C Ofline implications of constant sequences

This appendix records the standard constant-sequence argument behind the brief discussion in Section 4.4.3. Throughout it, ε denotes an ofline approximation slack and is unrelated to the field accuracy of Section $^ { 4 , }$ which does not appear here. A computational conclusion requires both an explicit regret rate and a normalization of the ofline optimum.

Proposition C.1 (Constant-sequence implication). Suppose that, on the constant sequence $f _ { t } \equiv f .$ an online algorithm satisfies

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } f ( p _ { t } ) \ge \alpha T \operatorname* { m a x } _ { o \in K } f ( o ) - R ( T ) .
$$

$I f \varsigma$ is uniform on $[ T ]$ and independent of the algorithm, then

$$
\mathbb { E } f ( p _ { \varsigma } ) \geq \alpha \operatorname* { m a x } _ { o \in K } f ( o ) - \frac { R ( T ) } { T } .\tag{94}
$$

In particular, if $R ( T ) = o ( T )$ , the expected approximation converges to α for every fixed instance. $I f$ moreover $R ( T ) \leq C T ^ { \varkappa }$ for some $\varkappa < 1$ , and a known $\mathsf { v } > 0$ satisfies $\operatorname* { m a x } _ { o \in K } f ( o ) \geq \mathsf { v } .$ , then

$$
T \geq \left( { \frac { C } { \varepsilon v } } \right) ^ { 1 / ( 1 - \varkappa ) }
$$

gives an $( \alpha - \varepsilon )$ -approximation in expectation.

Proof. Divide the assumed guarantee by T and use $\begin{array} { r } { \mathbb { E } f ( p _ { \varsigma } ) = T ^ { - 1 } \mathbb { E } \sum _ { t < T } f ( p _ { t } ) } \end{array}$ . The two consequences follow by letting $R ( T ) / T  0$ and by substituting the displayed lower bound on T.

Proposition C.2 (Online factors imply ofline factors). Let A attain $\mathrm { R e g } _ { \alpha } ( T ) \leq C T ^ { \varkappa } .$ , with fixed $\varkappa < 1$ , on every admissible sequence in either feedback model of Section 3. Assume its per-round computation and query budget are polynomial in T and the problem parameters. Suppose also that a known $\mathsf { v } > 0$ satisfies $\operatorname* { m a x } _ { o \in K } f ( o ) \geq \mathsf { v } ,$ and that a computable upper bound on $C _ { i }$ , together with $\mathsf { v } ^ { - 1 }$ is polynomial in the ofline input parameters. Then, for every $\varepsilon > 0 ,$ simulating A on a constant sequence gives a polynomial-time ofline $( \alpha - \varepsilon )$ -approximation in expectation.

Proof. Given an ofline instance $f ,$ run A on $f _ { t } \equiv f$ , record the played points $p _ { 1 } , \ldots , p _ { T }$ , and return $p _ { \varsigma }$ for a uniformly random $\varsigma \in [ T ]$ independent of the run. No value of $f$ is needed to make this selection, which matters because under (7) the simulator observes only noisy values and, in the bandit model, never observes the reward it earns; returning “the best point played” would not be implementable. With

$$
T = \left\lceil \left( \frac { C } { \varepsilon \mathsf { v } } \right) ^ { 1 / ( 1 - \varkappa ) } \right\rceil ,
$$

Proposition C.1 yields

$$
\mathbb { E } f ( p _ { \varsigma } ) \geq \alpha \operatorname* { m a x } _ { o \in K } f ( o ) - \varepsilon \mathsf { v } \geq ( \alpha - \varepsilon ) \operatorname* { m a x } _ { o \in K } f ( o ) .
$$

If exact values happen to be available one may instead return the best played point, which can only increase the left-hand side. The assumptions make the horizon and the total simulation cost polynomial. □

The lower bound v is essential for this quantitative polynomial-time implication: without a normalization of the optimum, an additive $o ( T )$ term need not become relatively negligible in polynomially many rounds. This observation does not prove that 0.401 is optimal; it only says that an online improvement with the stated complexity properties would also yield an ofline improvement.

## D Proof of the endpoint comparison inequalities

This appendix proves the two inequalities (22)–(23) used in Lemma 4.6. They are the specializations to $z = x$ and to the constant direction $x ( \tau ) \equiv x$ of Lemmas 5.7 and 5.9 of [10], restated with an arbitrary comparator o in place of an optimizer. Since these are the nonstandard ofline inequalities imported by the outer certificate, and since comparator-uniformity is used quantitatively, we give complete proofs. Throughout, $f : [ 0 , 1 ] ^ { d } \to { \mathbb { R } } _ { \geq 0 }$ is diferentiable, nonnegative and DR-submodular, and $o \in [ 0 , 1 ] ^ { d }$ is arbitrary. The two general-purpose lemmas of Appendices D.1 and D.2 are stated in an unspecified dimension $n ,$ since Lemma D.4 applies the second of them on a doubled ground set with $n = 2 d .$

## D.1 DR-submodular toolbox

Lemma D.1 (Basic properties). Let $f : [ 0 , 1 ] ^ { n } \to { \mathbb { R } } _ { \geq 0 }$ be diferentiable and DR-submodular. Then:

(P1) for all $u \leq w \ i n \ [ 0 , 1 ] ^ { n }$ and $\varDelta \geq 0$ with $w + \varDelta \leq \mathbf { 1 }$ 2

$$
f ( u + \varDelta ) - f ( u ) \geq f ( w + \varDelta ) - f ( w ) ;
$$

(P2) for all $u \leq w$ in $[ 0 , 1 ] ^ { n }$ and $\lambda \in [ 0 , 1 ]$

$$
f ( \lambda u + ( 1 - \lambda ) w ) \geq \lambda f ( u ) + ( 1 - \lambda ) f ( w ) ;
$$

(P3) $\langle \nabla f ( u ) , \varDelta \rangle \geq f ( u + \varDelta ) - f ( u ) \ f o r \ \varDelta \geq 0 \ a n d \ u + \varDelta \leq 1 ,$

(P4) $\langle \nabla f ( u ) , \Delta \rangle \leq f ( u ) - f ( u - \varDelta ) \ f o r \ \varDelta \geq 0 \ a n d \ u - \varDelta \geq 0 .$

Proof. DR-submodularity says that $\nabla f$ is antitone: $u \leq w$ implies $\nabla f ( u ) \geq \nabla f ( w )$ coordinatewise. Item (P1): since $u + \vartheta \varDelta \leq w + \vartheta \varDelta$ for every $\vartheta \in [ 0 , 1 ]$ and $\varDelta \geq 0$

$$
f ( u + \varDelta ) - f ( u ) = \int _ { 0 } ^ { 1 } \left. \nabla f ( u + \vartheta \varDelta ) , \varDelta \right. d \vartheta \geq \int _ { 0 } ^ { 1 } \left. \nabla f ( w + \vartheta \varDelta ) , \varDelta \right. d \vartheta = f ( w + \varDelta ) - f ( w ) .
$$

Item (P2): the map $\vartheta \mapsto f ( u + \vartheta ( w - u ) )$ has derivative $\langle \nabla f ( u + \vartheta ( w - u ) ) , w - u \rangle$ , which is nonincreasing in ϑ because $w - u \geq 0$ and $\nabla f$ is antitone. Hence the map is concave on $[ 0 , 1 ]$ , and item (P2) is concavity evaluated at the endpoints.

Item $\begin{array} { r } { ( \mathrm { P 3 } ) \colon f ( u + \varDelta ) - f ( u ) = \int _ { 0 } ^ { 1 } \langle \nabla f ( u + \vartheta \varDelta ) , \varDelta \rangle d \vartheta \leq \langle \nabla f ( u ) , \varDelta \rangle } \end{array}$ , again by antitonicity and $\varDelta \geq 0$ . Item (P4) is the same computation on the segment from $u - \varDelta$ to u. □

Lemma D.2 (Closure under $\odot$ and ⊕). For every $y \in [ 0 , 1 ] ^ { n }$ , the functions $a \mapsto f ( a \odot y )$ and $a \mapsto f ( a \oplus y )$ are nonnegative and DR-submodular on $[ 0 , 1 ] ^ { n }$

Proof. Both maps are of the form $a \mapsto f ( c + A a )$ with A a diagonal matrix with entries $y _ { i } \geq 0$ (for $\odot ,$ with $c = \mathbf { 0 } )$ or $1 - y _ { i } \ge 0$ (for ⊕, with $c = y )$ , and both are order preserving. Hence the gradient of the composition is $A \nabla f ( c + A a )$ , a nonnegative diagonal scaling of an antitone map composed with an order-preserving map, which is antitone. Nonnegativity is inherited from $f .$ □

## D.2 An exponential lower bound

The next lemma is the engine of both comparisons. It is Lemma 4.1 of [10]; we reproduce its proof.

Lemma D.3 (Basic exponential bound). Let $F : [ 0 , 1 ] ^ { n } \to \mathbb { R } _ { > 0 }$ be diferentiable, nonnegative and DR-submodular, let $t \geq 0$ , let $\xi : [ 0 , t ] \to [ 0 , 1 ] ^ { n }$ be integrable and let $a \in [ 0 , 1 ] ^ { n }$ . Write $\begin{array} { r } { y ( t ) = \int _ { 0 } ^ { t } \xi ( \tau ) d \tau } \end{array}$ . Then

$$
F \big ( \mathbf { 1 } - a \odot e ^ { - y ( t ) } \big ) \geq e ^ { - t } \left[ F ( \mathbf { 1 } - a ) + \sum _ { i = 1 } ^ { \infty } \frac { 1 } { i ! } \int _ { \tau \in [ 0 , t ] ^ { i } } F \Big ( ( \mathbf { 1 } - a ) \oplus \bigoplus _ { j = 1 } ^ { i } \xi ( \tau _ { j } ) \Big ) d \tau \right] .\tag{95}
$$

Proof. We prove by induction on $k \geq 0$ that (95) holds with the infinite sum replaced by $\scriptstyle \sum _ { i = 1 } ^ { k }$ since every term of the sum is nonnegative, monotone convergence then gives the lemma.

Base case $k = 0$ . Because $0 \leq y ( t ) \leq t \mathbf { 1 }$ , we have $e ^ { - t } \leq e ^ { - y ( t ) } \leq { \bf 1 }$ coordinatewise, so

$$
\varpi : = { \frac { e ^ { - y ( t ) } - e ^ { - t } \mathbf { 1 } } { 1 - e ^ { - t } } } \in [ 0 , 1 ] ^ { n }
$$

(for t = 0 the claim is trivial). A direct computation gives

$$
\mathbf { 1 } - a \odot e ^ { - y ( t ) } = e ^ { - t } ( \mathbf { 1 } - a ) + ( 1 - e ^ { - t } ) \big ( \mathbf { 1 } - a \odot \varpi \big ) ,
$$

and $\mathbf { 1 } - a \leq \mathbf { 1 } - a \odot \varpi$ since $\varpi \le { \bf 1 }$ . Item (P2) of Lemma D.1 and nonnegativity of $F$ give

$$
F \big ( \mathbf { 1 } - a \odot e ^ { - y ( t ) } \big ) \geq e ^ { - t } F ( \mathbf { 1 } - a ) + ( 1 - e ^ { - t } ) F \big ( \mathbf { 1 } - a \odot \varpi \big ) \geq e ^ { - t } F ( \mathbf { 1 } - a ) .
$$

Induction step. Assume the claim for $k - 1$ (for every a and every t). Let $g ( t ) = F ( { \bf 1 } - a \odot e ^ { - y ( t ) } )$ $g$ is absolutely continuous, and for almost every t the chain rule gives

$$
\begin{array} { r } { g ^ { \prime } ( t ) = \left. \nabla F \big ( \mathbf { 1 } - a \odot e ^ { - y ( t ) } \big ) , a \odot \xi ( t ) \odot e ^ { - y ( t ) } \right. . } \end{array}
$$

The direction $a \odot \xi ( t ) \odot e ^ { - y ( t ) }$ is nonnegative and

$$
\mathbf { 1 } - a \odot e ^ { - y ( t ) } + a \odot \xi ( t ) \odot e ^ { - y ( t ) } = \mathbf { 1 } - a \odot ( \mathbf { 1 } - \xi ( t ) ) \odot e ^ { - y ( t ) } \le \mathbf { 1 } ,
$$

so item (P3) gives

$$
g ^ { \prime } ( t ) \geq F \big ( \mathbf { 1 } - a ^ { \prime } \odot e ^ { - y ( t ) } \big ) - g ( t ) , \qquad a ^ { \prime } : = a \odot ( \mathbf { 1 } - \xi ( t ) ) ,
$$

and ${ \bf 1 } - a ^ { \prime } = ( { \bf 1 } - a ) \oplus \xi ( t )$ . Applying the induction hypothesis at level $k - 1$ to $a ^ { \prime } ,$

$$
g ^ { \prime } ( t ) + g ( t ) \geq \Theta _ { k } ( t ) : = e ^ { - t } \left[ F \big ( ( \mathbf { 1 } - a ) \ : \oplus \ : \xi ( t ) \big ) \ : + \sum _ { i = 1 } ^ { k - 1 } \frac { 1 } { i ! } \int _ { \tau \in [ 0 , t ] ^ { i } } F \Big ( ( \mathbf { 1 } - a ) \ : \oplus \ : \xi ( t ) \ : \oplus \ : \bigoplus _ { j = 1 } ^ { i } \xi ( \tau _ { j } ) \Big ) \ : d \tau \right] .
$$

Let $h ( t )$ denote the right-hand side of (95) truncated at k. Since $\bigoplus$ is symmetric and associative, each integrand is a symmetric function of $( \tau _ { 1 } , \dots , \tau _ { i } )$ , so the Leibniz rule gives

$$
\frac { d } { d t } \int _ { [ 0 , t ] ^ { i } } F \Big ( ( { \bf 1 } - a ) \oplus \bigoplus _ { j \leq i } \xi ( \tau _ { j } ) \Big ) d \tau = i \int _ { [ 0 , t ] ^ { i - 1 } } F \Big ( ( { \bf 1 } - a ) \oplus \xi ( t ) \oplus \bigoplus _ { j \leq i - 1 } \xi ( \tau _ { j } ) \Big ) d \tau ,
$$

whence, after reindexing,

$$
\begin{array} { r l } & { h ^ { \prime } ( t ) = - h ( t ) + e ^ { - t } \left[ F \big ( ( { \bf 1 } - a ) \oplus \xi ( t ) \big ) + \displaystyle \sum _ { i = 1 } ^ { k - 1 } \frac { 1 } { i ! } \int _ { [ 0 , t ] ^ { i } } F \Big ( ( { \bf 1 } - a ) \oplus \xi ( t ) \oplus \bigoplus _ { j \leq i } \xi ( \tau _ { j } ) \Big ) d \tau \right] } \\ & { \quad \quad = - h ( t ) + \Theta _ { k } ( t ) . } \end{array}
$$

Therefore $\begin{array} { r } { \frac { d } { d t } \big [ e ^ { t } ( g ( t ) - h ( t ) ) \big ] = e ^ { t } \big [ g ^ { \prime } ( t ) + g ( t ) - h ^ { \prime } ( t ) - h ( t ) \big ] \geq 0 } \end{array}$ almost everywhere, and $g ( 0 ) =$ $h ( 0 ) = F ( \mathbf { 1 } - a )$ , so $g ( t ) \geq h ( t )$ , which is the claim at level k. □

Lemma D.4 (Two-block exponential bound). Let $f : [ 0 , 1 ] ^ { d } \to { \mathbb { R } } _ { > 0 }$ be diferentiable, nonnegative and DR-submodular, let $b \in [ 0 , 1 ] ^ { d }$ , let $a ^ { ( 1 ) } , a ^ { ( 2 ) } \in [ 0 , 1 ] ^ { \dot { d } }$ , let $\xi ^ { ( 1 ) } , \bar { \xi ^ { ( 2 ) } } : [ 0 , t ] \to [ 0 , 1 ] ^ { d }$ be integrable, and write $\begin{array} { r } { y ^ { ( i ) } ( t ) = \int _ { 0 } ^ { t } \xi ^ { ( i ) } } \end{array}$ . Then

$$
\begin{array} { r l } {  { f \Big ( \mathbf 1 - b \odot a ^ { ( 1 ) } \odot e ^ { - y ^ { ( 1 ) } ( t ) } - ( \mathbf 1 - b ) \odot a ^ { ( 2 ) } \odot e ^ { - y ^ { ( 2 ) } ( t ) } \Big ) } \quad } & { } \\ & { \ge e ^ { - t } \Bigg [ f \big ( \mathbf 1 - b \odot a ^ { ( 1 ) } - ( \mathbf 1 - b ) \odot a ^ { ( 2 ) } \big ) } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { \infty } \frac { 1 } { i ! } \int _ { \tau \in [ 0 , t ] ^ { i } } f \Big ( b \odot \big ( ( \mathbf 1 - a ^ { ( 1 ) } ) \oplus \bigoplus _ { j \le i } \xi ^ { ( 1 ) } ( \tau _ { j } ) \big ) } \\ & { \qquad + ( \mathbf 1 - b ) \odot \big ( ( \mathbf 1 - a ^ { ( 2 ) } ) \oplus \bigoplus _ { j \le i } \xi ^ { ( 2 ) } ( \tau _ { j } ) \big ) \Big ) d \tau \Bigg ] . } \end{array}\tag{96}
$$

Proof. Work on the ground set $[ d ] \sqcup [ d ]$ of size $n = 2 d$ and define $F : [ 0 , 1 ] ^ { 2 d } \to \mathbb { R } _ { \ge 0 }$ by

$$
F \bigl ( c ^ { ( 1 ) } , c ^ { ( 2 ) } \bigr ) = f \bigl ( b \odot c ^ { ( 1 ) } + ( { \mathbf 1 } - b ) \odot c ^ { ( 2 ) } \bigr ) .
$$

$F$ is nonnegative, and for $u \in [ d ]$

$$
\frac { \partial F } { \partial c _ { u } ^ { ( 1 ) } } = b _ { u } \partial _ { u } f \big ( b \odot c ^ { ( 1 ) } + ( { \bf 1 } - b ) \odot c ^ { ( 2 ) } \big ) , \qquad \frac { \partial F } { \partial c _ { u } ^ { ( 2 ) } } = \left( 1 - b _ { u } \right) \partial _ { u } f \big ( \cdot \big ) ,
$$

with nonnegative prefactors. The argument $b \odot c ^ { ( 1 ) } + ( { \bf 1 } - b ) \odot c ^ { ( 2 ) }$ is coordinatewise nondecreasing in $( c ^ { ( 1 ) } , c ^ { ( 2 ) } )$ , and $\nabla f$ is antitone, so $\nabla F$ is antitone and F is DR-submodular.

Apply Lemma D.3 to $F$ with $a = ( a ^ { ( 1 ) } , a ^ { ( 2 ) } )$ and $\xi = ( \xi ^ { ( 1 ) } , \xi ^ { ( 2 ) } )$ . All operations act blockwise, and

$$
F ( \mathbf { 1 } - a ) = f { \big ( } b \odot ( \mathbf { 1 } - a ^ { ( 1 ) } ) + ( \mathbf { 1 } - b ) \odot ( \mathbf { 1 } - a ^ { ( 2 ) } ) { \big ) } = f { \big ( } \mathbf { 1 } - b \odot a ^ { ( 1 ) } - ( \mathbf { 1 } - b ) \odot a ^ { ( 2 ) } { \big ) } ,
$$

which gives exactly (96).

## D.3 A comparison lemma

Lemma D.5 (Comparison lemma). Let $x \in [ 0 , 1 ] ^ { d }$ . For every $\varDelta \in [ 0 , 1 ] ^ { d }$ with $\varDelta \leq \mathbf { 1 } - o$ and every $o ^ { \prime } \in [ 0 , 1 ] ^ { d }$ with $\smash { o ^ { \prime } \leq o }$

$$
f \big ( \boldsymbol { o } ^ { \prime } + ( \mathbf { 1 } - \boldsymbol { x } ) \odot \varDelta \big ) \geq f ( \boldsymbol { o } ) + f ( \mathbf { 0 } ) - f ( \boldsymbol { x } \oplus \boldsymbol { o } ) - f ( \boldsymbol { o } - \boldsymbol { o } ^ { \prime } ) \geq f ( \boldsymbol { o } ) - f ( \boldsymbol { x } \oplus \boldsymbol { o } ) - f ( \boldsymbol { o } - \boldsymbol { o } ^ { \prime } ) .\tag{97}
$$

Proof. Apply item (P1) with base points ${ \mathbf 0 \le o - o ^ { \prime } }$ and increment $\varDelta ^ { \prime } = o ^ { \prime } + \left( \mathbf { 1 } - x \right) \odot \varDelta \geq 0$ . This is legitimate because

$$
\left( o - o ^ { \prime } \right) + \varDelta ^ { \prime } = o + \left( \mathbf { 1 } - x \right) \odot \varDelta \leq o + \left( \mathbf { 1 } - o \right) = \mathbf { 1 } .
$$

It gives

$$
f \big ( o ^ { \prime } + ( \mathbf { 1 } - x ) \odot \Delta \big ) - f ( \mathbf { 0 } ) \geq f \big ( o + ( \mathbf { 1 } - x ) \odot \Delta \big ) - f ( o - o ^ { \prime } ) .
$$

Next apply item (P1) again, now with base points $o \leq x \oplus o$ and increment $( { \bf 1 } - x ) \odot \varDelta ;$ this is legitimate because $\mathbf { 1 } - \left( x \oplus o \right) = \left( \mathbf { 1 } - x \right) \odot \left( \mathbf { 1 } - o \right) \geq \left( \mathbf { 1 } - x \right) \odot \varDelta$ . It gives

$$
f \big ( o + ( \mathbf { 1 } - x ) \odot \varDelta \big ) - f ( o ) \geq f \big ( x \oplus o + ( \mathbf { 1 } - x ) \odot \varDelta \big ) - f ( x \oplus o ) \geq - f ( x \oplus o ) ,
$$

using nonnegativity of $f$ in the last step. Adding the two displays yields the first inequality of (97), and $f ( { \bf 0 } ) \geq 0$ yields the second. □

## D.4 The two comparisons

Recall from (15) and (16) that, with the constant direction x and delay $s ,$

$$
\mathbf { 1 } - Y _ { \tau } ( x ) = { \left\{ \begin{array} { l l } { x \oplus e ^ { - \tau x } , } & { \tau < s , } \\ { \left( x \oplus e ^ { - s x } \right) \odot e ^ { - ( \tau - s ) x } , } & { \tau \geq s , } \end{array} \right. }
$$

as is checked by expanding x ⊕ $e ^ { - \tau x } = x + e ^ { - \tau x } - x \odot e ^ { - \tau x }$ and comparing with (15).

Lemma D.6 (Early comparison). For every $x \in [ 0 , 1 ] ^ { d }$ , every $o \in [ 0 , 1 ] ^ { d }$ and every $\tau \in [ 0 , s )$

$$
f \big ( Y _ { \tau } ( x ) + \omega _ { \tau } ( x ) \odot o \big ) \geq f ( o ) - f ( x \odot o ) - ( 1 - e ^ { - \tau } ) f ( x \oplus o ) .
$$

Proof. Write $Y _ { \tau } = Y _ { \tau } ( x )$ and $\omega _ { \tau } = \omega _ { \tau } ( x )$ . By $( 1 7 ) , \omega _ { \tau } = \mathbf { 1 } - Y _ { \tau } - x$ in the early phase, so

$$
P _ { \tau } : = Y _ { \tau } + \omega _ { \tau } \odot o = Y _ { \tau } \oplus o - x \odot o = ( \mathbf { 1 } - x ) - ( \mathbf { 1 } - x ) \odot ( \mathbf { 1 } - o ) \odot e ^ { - \tau x } ,
$$

where the last equality follows by substituting $Y _ { \tau } = ( { \bf 1 } - x ) \odot ( { \bf 1 } - e ^ { - \tau x } )$ and simplifying. Equivalently,

$$
P _ { \tau } = \mathbf { 1 } - x \odot \mathbf { 1 } \odot e ^ { - \int _ { 0 } ^ { \tau } \mathbf { 0 } } - ( \mathbf { 1 } - x ) \odot ( \mathbf { 1 } - o ) \odot e ^ { - \int _ { 0 } ^ { \tau } x } .
$$

Apply Lemma D.4 with $t = \tau , b = x , a ^ { ( 1 ) } = { \bf 1 } , \xi ^ { ( 1 ) } \equiv { \bf 0 } , a ^ { ( 2 ) } = { \bf 1 } - o \mathrm { ~ a n d ~ } \xi ^ { ( 2 ) } \equiv x .$

The base term is $f ( \mathbf { 1 } - x - ( \mathbf { 1 } - x ) \odot ( \mathbf { 1 } - o ) ) = f ( ( \mathbf { 1 } - x ) \odot o )$ . Applying item (P1) with base points $\mathbf { 0 } \leq ( \mathbf { 1 } - x ) \odot o$ and increment $x \odot o ,$ whose sum with the larger base point is $o \le { \bf 1 }$ , gives

$$
f ( x \odot o ) - f ( \mathbf { 0 } ) \geq f ( o ) - f { \big ( } ( \mathbf { 1 } - x ) \odot o { \big ) } , \qquad { \mathrm { i . e . } } \qquad f { \big ( } ( \mathbf { 1 } - x ) \odot o { \big ) } \geq f ( o ) - f { \big ( } x \odot o { \big ) } .
$$

For the term of order $i \geq 1$ , note $\mathbf { 1 } - a ^ { ( 1 ) } = \mathbf { 0 }$ and $\bigoplus _ { j \leq i } \mathbf { 0 } = \mathbf { 0 } .$ , so the first block contributes $x \odot \mathbf { 0 } = \mathbf { 0 }$ , while $\mathbf { 1 } - a ^ { ( 2 ) } = o ;$ writing $W _ { i } = \bigoplus _ { j = 1 } ^ { i } x \in [ 0 , 1 ] ^ { d }$ the integrand equals

$$
f \bigl ( ( \mathbf { 1 } - x ) \odot ( o \oplus W _ { i } ) \bigr ) = f \bigl ( o ^ { \prime } + ( \mathbf { 1 } - x ) \odot \varDelta \bigr ) , \qquad o ^ { \prime } = ( \mathbf { 1 } - x ) \odot o , \quad \varDelta = ( \mathbf { 1 } - o ) \odot W _ { i } ,
$$

which is of the form treated by Lemma D.5 because $\smash { o ^ { \prime } \leq o }$ and $\begin{array} { r } { \varDelta \leq \mathbf { 1 } - o . } \end{array}$ . Since $o - o ^ { \prime } = x \odot o .$ ， that lemma gives

$$
f { \big ( } ( 1 - x ) \odot ( o \oplus W _ { i } ) { \big ) } \geq f ( o ) - f ( x \oplus o ) - f ( x \odot o ) .
$$

The integrand is independent of $\tau _ { \mathrm { { ; } } }$ , and $\begin{array} { r } { \sum _ { i \geq 1 } \tau ^ { i } / i ! = e ^ { \tau } - 1 } \end{array}$ . Substituting the two bounds into (96),

$$
\begin{array} { r l } & { f ( P _ { \tau } ) \geq e ^ { - \tau } \Big [ \big ( f ( o ) - f ( x \odot o ) \big ) + ( e ^ { \tau } - 1 ) \big ( f ( o ) - f ( x \oplus o ) - f ( x \odot o ) \big ) \Big ] } \\ & { \qquad = f ( o ) - f ( x \odot o ) - ( 1 - e ^ { - \tau } ) f ( x \oplus o ) , } \end{array}
$$

as claimed.

Lemma D.7 (Late comparison). For every $x \in [ 0 , 1 ] ^ { d }$ , every $o \in [ 0 , 1 ] ^ { d }$ and every $\tau \in [ s , 1 ]$

$$
\begin{array} { r } { f \big ( Y _ { \tau } ( x ) + \omega _ { \tau } ( x ) \odot o \big ) \ge e ^ { - \tau } \Big [ e ^ { s } f ( o ) - ( e ^ { s } - 1 ) f ( x \oplus o ) + ( \tau - s ) f ( x \oplus o ) \Big ] . } \end{array}
$$

Proof. By (17), $\boldsymbol { \omega } _ { \tau } = \mathbf { 1 } - Y _ { \tau }$ in the late phase, so $Y _ { \tau } + \omega _ { \tau } \odot o = Y _ { \tau } \oplus o$ and

$$
\mathbf { 1 } - \left( Y _ { \tau } \oplus o \right) = \left( \mathbf { 1 } - Y _ { \tau } \right) \odot \left( \mathbf { 1 } - o \right) = x \odot \left( \mathbf { 1 } - o \right) \odot e ^ { - ( \tau - s ) x } + \left( \mathbf { 1 } - x \right) \odot \left( \mathbf { 1 } - o \right) \odot e ^ { - \tau x } ,
$$

using the displayed formula for ${ \bf 1 } - Y _ { \tau }$ and expanding $x \oplus e ^ { - s x }$ . Define the piecewise-constant direction

$$
\xi ^ { \prime } ( \varsigma ) = \left\{ \begin{array} { l l } { \mathbf { 0 } , } & { \varsigma < s , } \\ { x , } & { \varsigma \geq s , } \end{array} \right.
$$

so that $\begin{array} { r } { \int _ { 0 } ^ { \tau } \xi ^ { \prime } = ( \tau - s ) x } \end{array}$ . Apply Lemma D.4 with $t = \tau , b = x , a ^ { ( 1 ) } = a ^ { ( 2 ) } = \mathbf { 1 } - o , \xi ^ { ( 1 ) } = \xi ^ { \prime }$ and $\xi ^ { ( 2 ) } \equiv x .$

The base term is $f \big ( \mathbf { 1 } - x \odot ( \mathbf { 1 } - o ) - ( \mathbf { 1 } - x ) \odot ( \mathbf { 1 } - o ) \big ) = f ( o )$

For the term of order $i \geq 1$ at a point $\varsigma \in [ 0 , \tau ] ^ { i }$ , the integrand is

$$
\Xi _ { i } ( \varsigma ) = f { \Big ( } x \odot { \big ( } o \oplus \bigoplus _ { j \leq i } \xi ^ { \prime } ( \varsigma _ { j } ) { \big ) } + ( \mathbf { 1 } - x ) \odot { \big ( } o \oplus \bigoplus _ { j \leq i } x { \big ) } { \Big ) } .
$$

We distinguish three cases.

(a) All coordinates $o f \varsigma$ lie in $[ 0 , s )$ . Then $\xi ^ { \prime } ( \varsigma _ { j } ) = \mathbf { 0 }$ for all $j ,$ so the first block contributes $x \odot o$ and

$$
\Xi _ { i } ( \varsigma ) = f { \big ( } x \odot o + ( \mathbf { 1 } - x ) \odot ( o \oplus W _ { i } ) { \big ) } = f { \big ( } o + ( \mathbf { 1 } - x ) \odot ( \mathbf { 1 } - o ) \odot W _ { i } { \big ) } , \qquad W _ { i } = \bigoplus _ { j \leq i } x .
$$

Lemma D.5 with $o ^ { \prime } = o$ and $\varDelta = \left( \mathbf { 1 } - o \right) \odot W _ { i }$ gives $\Xi _ { i } ( \varsigma ) \geq f ( o ) - f ( x \oplus o )$

$( b ) \ i = 1$ and $\varsigma _ { 1 } \geq s$ . Then $\xi ^ { \prime } ( \varsigma _ { 1 } ) = x$ and both blocks carry the same argument $o \oplus x$ , so $\Xi _ { 1 } ( \varsigma ) = f ( x \oplus o )$ exactly.

$( c ) \ i \geq 2$ and at least one coordinate $o f \varsigma$ is at least s. We use only $\Xi _ { i } ( \varsigma ) \geq 0$

The set in case (a) has measure $s ^ { i }$ within $[ 0 , \tau ] ^ { i }$ , and case (b) contributes the interval $[ s , \tau ]$ of length $\tau - s$ . Hence

$$
\sum _ { i \ge 1 } \frac { 1 } { i ! } \int _ { [ 0 , \tau ] ^ { \sharp } } \Xi _ { i } \ge \sum _ { i \ge 1 } \frac { s ^ { i } } { i ! } \big ( f ( o ) - f ( x \oplus o ) \big ) + ( \tau - s ) f ( x \oplus o ) = ( e ^ { s } - 1 ) \big ( f ( o ) - f ( x \oplus o ) \big ) + ( \tau - s ) f ( x \oplus o ) .
$$

Substituting into (96),

$$
f \big ( Y _ { \tau } \oplus o \big ) \geq e ^ { - \tau } \Big [ f ( o ) + ( e ^ { s } - 1 ) \big ( f ( o ) - f ( x \oplus o ) \big ) + ( \tau - s ) f ( x \oplus o ) \Big ] ,
$$

which is the claim after collecting the $f ( o )$ terms into $e ^ { s } f ( o )$

Lemmas D.6 and D.7 are exactly (22) and (23), completing the proof of Lemma 4.6. We note that only nonnegativity and DR-submodularity of $f$ were used, and that o was an arbitrary point of $[ 0 , 1 ] ^ { d } ;$ no optimality of $^ { O , }$ and no property of K beyond $x , o \in [ 0 , 1 ] ^ { d }$ , enters these two lemmas. This is the comparator-uniformity on which the online argument depends.