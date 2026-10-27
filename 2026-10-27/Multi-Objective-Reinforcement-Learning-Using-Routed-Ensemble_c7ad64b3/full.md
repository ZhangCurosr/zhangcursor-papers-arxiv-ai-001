# Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution

Zhenshuo Zhang Northeastern University Boston, MA

Ruoxuan Xiong Emory University Atlanta, GA

Hongyang R. Zhang Northeastern University Boston, MA

## Abstract

Multi-objective reinforcement learning is a fundamental problem for modern AI, with applications spanning post-training, reasoning models, and robotic control. A central challenge is to learn policies that balance competing objectives, such as helpfulness versus honesty in safety and alignment, or speed versus energy in robotics. The optimal policy can vary sharply with objective-weighting combinations, and the extent of this variation depends crucially on the degree of conflict among objectives, making per-preference policy learning sample-inefficient. In this paper, we introduce objective-conflict coefficients and derive near-optimal regret bounds for online settings using this measure, thereby quantifying the impact of misalignment on learning efficiency. Then, we design routed ensemble policy learning, an algorithm that maintains a small set of basis policies and dynamically routes trajectories to the policy with the highest potential reward. Finally, we consider trajectory selection as another approach to mitigate objective conflict, and design a kernel surrogate trajectory attribution algorithm that estimates how feedback signals from a set of trajectories influence policy learning on other trajectories. The kernel can capture nonlinear interactions such as synergy and antagonism between trajectories. We evaluate our approach across diverse environments and benchmarks: routed ensembles improve average performance by 4.3% over baselines for online environments from Meta-World and MO-Gymnasium, and kernel surrogate attribution improves correlation with leave-one-out analysis over existing baselines by 50%. When applied to trajectory selection, kernel-based attribution yields a 27% improvement in reward over baseline attribution methods.

## 1 Introduction

Given m potentially competing objectives in a reinforcement learning (RL) environment, how can we learn a small set of policies that maximizes expected reward under any specified preference weighting? This problem is fundamental in multi-agent systems [29], and is ubiquitous in modern AI systems. In robotics, agents often need to balance speed and energy efficiency. In reinforcement learning with human feedback, reasoning models, and post-training, models are often evaluated under competing criteria such as helpfulness, safety, and honesty [20, 37].

A core challenge in multi-objective RL (MORL) is that the optimal policy can vary drastically with objective weight combinations. For example, even small changes in preference can lead to qualitatively different optimal policies [34, 55]. Prior work has focused on approximating the Pareto frontier of the optimal reward trade-offs [42, 52, 55], or learning one policy for each objective [44, 53], and it remains unclear whether or when transferring trajectories across objectives is effective.

A key yet underexplored question is that the difficulty of learning across preferences is fundamentally tied to the degree of misalignment among objectives. When objective weightings are well aligned (e.g., rewarding both final and intermediate progress), additional objectives can facilitate learning, and data collected under one preference can benefit policy learning for others. When objectives are misaligned or adversarially-injected (e.g., RLHF with corrupted reward components), naive transfer can become inefficient or even harmful [20, 27]. This problem is especially acute in high-dimensional settings, making it a critical open challenge to identify objective alignment [21].

To address this challenge, we introduce a new theoretical framework based on measuring objective conflict coefficients that quantify the degree of misalignment among objectives through the $\chi ^ { 2 } .$ divergence. Our analysis shows that this coefficient interacts with the coverability coefficient [46]— which captures how well a single exploration policy can cover the preference space—to determine the sample complexity of online MORL. Our regret bound generalizes single-objective results [1, 19, 46, 7, 9] and is the first to formalize how objective misalignment affects learning efficiency across preferences. Our theoretical framework builds on recent advances in transfer learning under distribution shifts [50]. However, the technical challenge in MORL lies in quantifying the discrepancy in policy learning. Our main result, Theorem 3.6, relies on a low-rank basis assumption, along with a coverability condition that ensures sufficient exploration of the basis.

Motivated by our theoretical framework, we propose routed ensemble policy learning, an online algorithm for learning a small set of basis policies, which can be applied to a real-world RL environment. This algorithm alternates between two steps: (i) routing each trajectory to the basis policy with the highest expected return under the corresponding preference weight, and (ii) partitioning trajectories into disjoint clusters and updating each basis policy using its assigned trajectories with a standard policy learning algorithm such as proximal policy optimization.

Next, we consider an alternative perspective for tackling objective conflicts, based on selecting trajectories. This motivates the problem of quantifying the influence of individual trajectories upon the learned optimal policy, a problem for which we term trajectory attribution. We design a scalable kernel-based trajectory attribution procedure that estimates how trajectories help or hinder policy learning. Our procedure fits a kernel ridge regression surrogate to approximate policy performance across trajectory subsets. This can capture both individual contributions and higher-order interactions between trajectories. These interactions are common in high-dimensional settings and often underlie unexpected transfer behavior [21]. However, they are not captured by first-order influence approximations such as influence functions [36], linear surrogate models [26, 56], or gradient-similarity-based analysis [45, 17]. Our approach goes beyond prior work by leveraging efficient approximations grounded in policy gradients and random projections [55]. The results from more accurate trajectory attribution can guide trajectory selection in downstream policy learning.

Lastly, we empirically validate our approach on the Meta-World robotics benchmark and highdimensional MO-Gymnasium tasks. Our algorithm consistently outperforms multi-objective baselines, achieving a significant reduction in cumulative regret in environments with structured objective conflicts. Specifically, routed ensemble learning outperforms the strongest baseline by 4.3% on the Meta-World benchmark and achieves an 8% average improvement in hypervolume on MO-Gymnasium tasks. Similar results are also observed in reinforcement learning with human feedback (RLHF) settings. Additionally, kernel-based attribution provides a 50% improvement in terms of the Spearman correlations with ground-truth leave-one-out influence scores. When applied to downstream task selection, kernel-based selection yields a 27% higher reward than existing attribution baselines. We present detailed ablation studies to validate the theoretical insights from our proposed framework. For instance, after the routed ensemble procedure, the coverability metric drops from 1.3 for a single policy to 0.2 for the ensemble policies, indicating that routing mitigates the exploration bottleneck caused by objective conflicts. We also conduct a detailed analysis on the number of basis policies in the ensemble, as well as validate a low-rank basis assumption in our theoretical setup.

In summary, the contributions of this paper are threefold.

• First, we introduce a new theoretical framework for MORL based on a $\chi ^ { 2 }$ measure that we call the objective conflict coefficient. The theoretical result can be relaxed to a setting with an approximate low-rank basis, and we empirically evaluate this assumption in various real-world environments.

• Second, we design two algorithms for tackling objective conflicts: (1) routed ensemble policy learning, and (2) kernel-based trajectory attribution to estimate inter-trajectory influence.

• Third, we demonstrate significant gains in sample efficiency and robustness across diverse MORL benchmarks covering language modeling, robotic control, and RLHF settings.

For the rest of this paper, we begin by stating the formal problem setup in Section 2. Afterward, we present our theoretical framework and proposed approach to tackle objective conflicts in Section 3. We will present our experiments to validate our proposed algorithms in Section 4. Lastly, we discuss the related work in Section 5 and conclude in Section 6. We provide the omitted algorithmic description and proofs in Appendix A and B. Omitted experimental setup and empirical evaluations are provided in Appendix C.

## 2 Preliminaries

We consider an RL environment modeled as a multi-objective Markov decision process (MOMDP), defined by the tuple $\mathcal { M } = ( S , \mathcal { A } , \mathrm { P r } ( \cdot \mid \cdot , \cdot ) , \mathbf { r } , \Omega , f _ { \Omega } , \dot { H } )$ , where and denote the state and action spaces, $\textstyle \operatorname { P r } ( { \cdot } \mid s , a )$ is the transition kernel, $\mathbf { r } : \mathcal { S } \times \mathcal { A }  \mathbb { R } ^ { m }$ is a vector-valued reward function with $m \geq 2$ objectives, and H is the horizon. The preference space is denoted as $\Omega \subseteq \mathbb { R } _ { + } ^ { m }$ The scalarization operator $f _ { w }$ maps the return vector to a scalar utility under preference $w \in \dot { \Omega }$ . We evaluate performance under a target preference distribution $w \sim \mathcal { P }$

A stochastic policy $\pi \in$ Π maps states to action distributions, and induces a trajectory $\tau =$ $( s _ { 0 } , a _ { 0 } , { \bf r } _ { 0 } , \ldots , s _ { H } , a _ { H } , { \bf r } _ { H } )$ , where $a _ { h } \sim \pi ( \cdot \mid s _ { h } )$ and $s _ { h + 1 } \sim \operatorname* { P r } [ \cdot \mid s _ { h } , a _ { t } ]$ . More generally, the policy can be time dependent $\pi _ { 1 } , \cdots , \pi _ { H }$ with $a _ { h } \sim \pi _ { h } ( \cdot \mid s _ { h } )$ and induces a trajectory similarly.

The trajectory return is the cumulative reward vector $\begin{array} { r } { \mathbf { G } ( \tau ) = \sum _ { h = 0 } ^ { H } \mathbf { r } ( s _ { h } , a _ { h } ) \in \mathbb { R } ^ { m } } \end{array}$ . The expected return of policy π from an initial distribution $s _ { 0 } \sim \rho$ is given by

$$
\begin{array} { r } { \mathbf { J } ( \pi ) = \underset { s _ { 0 } \sim \rho , \tau \sim \pi } { \mathbb { E } } \left[ \mathbf { G } ( \tau ) \right] \in \mathbb { R } ^ { m } . } \end{array}
$$

We primarily focus on linear operators, and thus the policy’s expected return can be written as $\mathcal { J } ( \dot { \pi } ; w ) = \dot { \left. w , \mathbf { J } ( \pi ) \right. }$ . We consider an episodic online MORL setting with episodes $t = 1 , \ldots , T .$ . At the beginning of episode t, the system designer has access to the history $\bar { \mathcal { D } ^ { ( t - 1 ) } } = \{ ( \tau ^ { ( u ) } , w _ { u } ) \} _ { u = 1 } ^ { t - 1 }$ and observes the current preference vector $w _ { t } \in \Omega$ . Based on $( \mathcal { D } ^ { ( t - 1 ) } , w _ { t } )$ , a learner applies a possibly randomized policy learning algorithm to select a policy $\pi ^ { ( t ) }$ <sup>)</sup>, interacts with the MOMDP for H horizons, and observes a trajectory $\bar { \mathbf { \Gamma } } ^ { ( t ) } = ( s _ { 0 } ^ { ( t ) } , a _ { 0 } ^ { ( t ) } , \bar { \mathbf { r } _ { 0 } ^ { ( t ) } } , \bar { \mathbf { \Gamma } } _ { 0 } ^ { ( t ) } , a _ { H } ^ { ( t ) } , a _ { H } ^ { ( t ) } , \mathbf { r } _ { H } ^ { ( t ) } )$

After T episodes, the learner collects a set of trajectories $\boldsymbol { \mathcal { D } } = \{ ( \tau ^ { ( t ) } , w _ { t } ) \} _ { t = 1 } ^ { T }$ and aims to learn a preference-conditioned policy $\pi ( \cdot ; w )$ that attains high utility for target preferences w in the support of . In this paper, we focus on learning an ensemble policy corresponding to a small set of $\pi ( \cdot ; w )$

Our theoretical framework builds on the notion of coverability, which is a complexity measure defined on a fixed policy class Π [1, 46]:

$$
C _ { \mathrm { c o v } } : = \operatorname* { i n f } _ { \mu \in \Delta ( S \times A ) } \operatorname* { s u p } _ { \pi \in \Pi } \operatorname* { m a x } _ { h \in [ H ] } \operatorname* { s u p } _ { ( s , a ) } \frac { d _ { h } ^ { \pi } ( s , a ) } { \mu _ { h } ( s , a ) } ,
$$

where $d _ { h } ^ { \pi } ( s , a ) = \operatorname* { P r } ^ { \pi } ( s _ { h } = s , a _ { h } = a )$ denotes the state-action probability under policy $\pi ,$ , and the infimum is over sequences of $\mu _ { h } \in { \Delta ( S \times \mathcal { A } ) }$ . Intuitively, $C _ { \mathrm { c o v } }$ captures how hard it is for a single exploration distribution to simultaneously cover the occupancy measures of all policies in Π, serving as a hardness measure for online exploration.

## 3 Our Approach

We first provide several examples to illustrate how multi-objective misalignment affects policy learning. We then formalize this observation through the notion of objective conflict coefficient and show how it interacts with coverability to derive online regret bounds. The regret bound is shown to be tight with a matching lower bound. These theoretical insights motivate our algorithm, routed ensemble policy learning, which learns a small set of basis policies and routes trajectories to the policies that give the highest reward. Lastly, to interpret objective conflict and routing behavior, we develop a kernel-surrogate trajectory attribution algorithm that quantifies how individual trajectories influence policy learning across preference weights.

## 3.1 Examples and motivation

We start with two common MOMDP use cases to motivate our theoretical framework.

![](images/e182fe361f199a2d91bcb27dee733a3fa79a2c1a366f55e846e9c0758c125474.jpg)  
r<sub>1</sub>

![](images/a85268af65e5d67fa663ea166db5a16bcc82dd4fb301efcd4d7d491fa2db73f4.jpg)  
r<sub>1</sub>

![](images/4f9d6c449a1ebb1d5f083a418ba3db700de74b2c1bacf1ed18b6732fab723082.jpg)  
r<sub>1</sub>

![](images/3639e357525ea931a98318f62ec9a1a42acefe3ffe6a8368f7f3e9b05ea3cc6a.jpg)  
r<sub>1</sub>  
Figure 1: We visualize the objective space returns $( r _ { 1 } , r _ { 2 } )$ for policies optimized under varying preference weights $w \in [ 0 , 1 ]$ in three canonical regimes — clustered, linear and concave Pareto frontier. Light yellow indicates that w is closer to 1, while dark blue indicates w is closer to 0.

Example 3.1 (Learning the Pareto frontier of a MOMDP). The Paretofrontier consists ofpolicies for which no objective can be improved without worsening at least one other objective. Formally, a policy π dominates $\pi ^ { \prime }$ (denoted $\pi \succ \pi ^ { \prime } )$ $i f J _ { i } ( \pi ) \ge J _ { i } ( \check { \pi ^ { \prime } } )$ for all objectives $i \in [ m ]$ ], with strict inequalityfor at least one i, where $J _ { i } ( \pi )$ is the i-th entry of $\mathbf { \Delta } \cdot \mathbf { J } ( \pi )$

Example 3.2 (Online exploration under corrupted objectives). Consider an MOMDP with $m = 2$ where $r _ { 1 }$ is the clean task reward and $r _ { 2 }$ may be corrupted (possibly adversarial). The target preference is $w = [ 1 , 0 ] ^ { \top }$ , so the goal is to optimize $r _ { 1 } .$ . However, if corruption is present, trajectories collected online may be influenced by the spurious signal in $r _ { 2 } .$ . This motivates the need to identify corrupted trajectories and to attribute their alignment with w.

We illustrate that learning in an MOMDP depends on how objectives relate to one another. In the two-objective case $( m \ : = \ : 2 )$ , there are three canonical regimes, as shown in Figure 1: (i) aligned objectives, (ii) conflicting objectives with an (approximately) linear Paretofrontier, and (iii) conflicting objectives with a strictly concave frontier. Aligned objectives admit a single policy that is optimal for all preferences. Linear conflict is captured by two corner policies whose mixtures span the frontier. Concave conflict requires additional intermediate policies beyond the corner solutions. When m grows, different objectives can simultaneously fall into different regimes—some aligned, linearly conflicting, and others exhibiting concave trade-offs. These induce discontinuous shifts and make policy optimization harder.

## 3.2 Theoretical characterization through conflict coefficients

With the above examples in mind, next, we quantify how objective conflict impacts sample complexity in MORL. We assume that there exists a set of K latent basis policies that (approximately) represent optimal policies across preferences for w $\mathbf \xi \in \Omega$ . The number of basis policies $\dot { K }$ captures the intrinsic complexity of the objective trade-offs. We capture the extent of conflict via the objective conflict coefficient Γ, which measures the diversity of basis occupancies relative to their mixture.

Definition 3.3. Let the basis-mixture occupancy be the uniform average of the K basis policies, $\begin{array} { r } { \overline { d } _ { h } ( s , a ) : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { * } } ( s , a ) } \end{array}$ . The objective conflict coefficient is

$$
\Gamma : = \operatorname* { m a x } _ { h \in \left[ H \right] } \operatorname* { m a x } _ { k \in \left[ K \right] } \chi ^ { 2 } \left( d _ { h } ^ { \pi _ { b k } ^ { * } } \| \overline { { d } } _ { h } \right) .\tag{1}
$$

Smaller Γ indicates substantial overlap among basis policies and easier transfer across preferences, while large values correspond to near-disjoint occupancies and greater learning difficulty. Moreover, Γ is always bounded by $K - 1$ (Lemma B.5), so its magnitude scales with the number of basis policies. For instance, $\Gamma = 0$ in Case 1, and Γ is larger in Case 3 than in Case 2.

We now analyze online MORL through cumulative regret:

$$
\mathsf { R e g } ( T ) = \sum _ { t = 1 } ^ { T } \left( \mathcal { I } ( \pi _ { w _ { t } } ^ { \star } ; w _ { t } ) - \mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \right) ~ .
$$

We impose two structural assumptions. The first states that optimal policies lie in the convex hull of the basis occupancies.

Assumption 3.4. There exists a finite set of latent basis policies $\{ \pi _ { 1 } ^ { \star } , \ldots , \pi _ { K } ^ { \star } \}$ such that for any $w \in \Omega$ , there exists a mixing vector $\nu _ { w } \in \Delta ^ { K - 1 }$ (the probability simplex over K elements) satisfying $\begin{array} { r } { d _ { h } ^ { \pi _ { w } ^ { \star } } ( s , a ) = \sum _ { k = 1 } ^ { K } \nu _ { w } ( k ) d _ { h } ^ { \pi _ { b k } ^ { \star } } ( s , a ) } \end{array}$ for all $( s , a ) \in S \times A$

This assumption is imposed to yield a clean characterization of the role of K in regret. We relax this condition in Appendix B.4 by allowing optimal occupancies to lie within an ϵ-neighborhood of the convex hull, under which the same qualitative conclusions continue to hold. The second assumption ensures sufficient exploration of all basis components.

Assumption 3.5. For each $t \_ { 1 }$ , there exists $\nu ^ { ( t ) } ~ \in ~ [ 0 , 1 ] ^ { K }$ such that $d _ { h } ^ { \pi ^ { ( t ) } } ( s , a ) =$ $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \nu ^ { ( t ) } ( k ) d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { \star } } ( s , a ) } \end{array}$ for all $( s , a )$ and H. Letting $\begin{array} { r } { \overline { { \nu } } ^ { ( t ) } ( k ) = \frac { 1 } { t - 1 } \sum _ { i = 1 } ^ { t - 1 } \nu ^ { ( i ) } ( k ) } \end{array}$ , we assume there exist $c _ { 0 } \in ( 0 , 1 ]$ and a burn-in time $t _ { 0 }$ such that mi $1 _ { k \in [ K ] } \overline { { \nu } } ^ { ( t ) } ( k ) \geq c _ { 0 } / ( C _ { \mathrm { c o v } } K ) f o r a l l t \geq t _ { 0 }$

This condition holds, for example, when preferences are sampled with full support over Ω, ensuring persistent coverage of all basis policies.

We now derive a regret bound that explicitly quantifies the impact of K and Γ. We study a Kpolicy variant of the optimistic confidence-set algorithm [19]. The learner maintains K confidence sets $\mathcal { F } _ { k } ^ { ( t - 1 ) } ( \beta )$ , one for each basis index $k \in [ K ]$ , and at episode t executes the optimistic policy $\pi ^ { ( t ) } = \pi _ { f ^ { ( t ) } }$ associated with some $f ^ { ( t ) } \in \mathscr { F } _ { k _ { t } } ^ { ( t - 1 ) } ( \beta )$ for the routed basis $k _ { t }$ . A detailed description is provided in Appendix B.1. Each $\mathcal { F } _ { k }$ is given by $\mathcal { F } _ { k , 1 } \times \cdot \cdot \cdot \times \mathcal { F } _ { k , H }$ , with $\mathcal { F } _ { k , h } \subseteq ( S \times A \times \Omega \to [ 0 , H ] )$ . We assume that $\mathcal { F } _ { k }$ is finite and write $| \mathcal { F } | = \operatorname* { m a x } _ { k \in [ K ] } | \mathcal { F } _ { k } |$ . For infinite classes, one can replace log  via covering numbers [19]. Our main result is stated as follows.

Theorem 3.6. Suppose scalarized rewards are bounded in [0, 1] per step, Bellman realizability and completeness hold for each per-basis class $\mathcal { F } _ { k } ,$ , and Assumptions 3.4–3.5 hold. Let $\beta \ =$ c<sub>1</sub> log $\langle K \vert \mathcal { F } \vert T / \delta )$ be the confidence-set radius after a union bound across the K basis classes. After $t _ { 0 } = \mathcal { \bar { O } } ( \dot { C } _ { \mathrm { c o v } } \dot { K } \acute { \beta } )$ , then with probability at least $1 - \delta ,$ , there exists an algorithm with K policies such that the cumulative regret satisfies

$$
\mathsf { R e g } ( T ) \leq \mathcal { O } _ { T } \left( H \sqrt { C _ { \mathrm { c o v } } ( 1 + \Gamma ) \log ( K | \mathcal { F } | T \delta ^ { - 1 } ) T } \right) .\tag{2}
$$

Theorem 3.6 decomposes learning difficulty into (i) exploration hardness, governed by $C _ { \mathrm { c o v } } .$ , and (ii) transfer hardness induced by objective conflict, captured by Γ. When objectives are well aligned, both Γ and K are small, and the regret approaches the single-objective rate; when basis occupancies are nearly disjoint, Γ increases, reflecting the cost of transferring information across preferences. The full proof of Theorem 3.6 is given in Appendix B.1.

Nearly-tight regret lower bound. We complement this upper bound with a matching lower bound in Theorem B.6 in Appendix B.2, in the order of $\Omega _ { T } \big ( H \sqrt { ( 1 + \Gamma ) T } \big )$ , establishing the tightness of the dependence on Γ and T up to $C _ { \mathrm { c o v } }$ and logarithmic factors.

Remark 3.7. Although Theorem 3.6 is developed for a stylized online algorithm, similar proof techniques can be extended to more general settings. Proposition B.10 in Appendix B.4 shows that, when the low-rank basis Assumption 3.4 is relaxed to approximate low-rankness, the resulting regret bound preserves the same qualitative dependence on K and Γ.

Furthermore, empirical analysis in Appendix C.2 demonstrates that these insights extend beyond the theoretical setting. To assess the approximation empirically, we measure the convex-hull residual over a preference grid. In a tabular concave setting, the mean residual decreases from 0.186 at $K = 3 \ t o \ 1 . 4 5 \times 1 0 ^ { - 4 } \ a t \ K = 2 1$ ; on MO-Hopper, it decreases from 0.443 at $K = 3$ to 0.231 at $K = 1 5$ using candidate-selected bases. The analysis below uses the exact assumption, while Appendix B.4 treats nonzero residuals. We alsofind that increasing the size ofN gradually reduces the approximation error imposed by the low-rank condition (See Table 8for the evaluation).

## 3.3 Learning routed ensemble policies

In Sections 3.1 and 3.2, the basis policies are assumed latent. Next, we design a scalable routed ensemble policy learning algorithm that:

(i) explicitly learn a set of N basis policies, with each trajectory routed to the highest-reward basis, and

(ii) learn a preference-conditioned transfer operator $f _ { w } : \Pi ^ { N } \to \Pi$ that maps these basis policies to a composite policy for any given preference w.

We also clarify that the ensemble size $N$ may or may not be the same as $K$ as stated in Assumption 3.4. Thus, the algorithms studied will apply to a real-world MORL environment.

Our proposed Algorithm 1 alternates between scoring, routing, and updating: (i) Scoring: For each trajectory and each basis policy, compute a score using its estimated scalarized return (cf. Equation (4) below). (ii) Routing: Assign each trajectory to the basis policy that achieves the highest score for the corresponding preference, inducing a partition of the data into N clusters. (iii) Updating: Update each basis policy using only the trajectories assigned to its cluster via a standard RL subroutine (e.g., upper-confidence bound value iteration for MOMDPs [44]). (iv) Iteration: Repeat the above steps across online batches until both the cluster assignments and the learned policies stabilize. We formalize the learning objective as:

$$
\operatorname* { m a x } _ { \pi _ { 1 } , \pi _ { 2 } , \ldots , \pi _ { N } } \underset { s _ { 0 } \sim \rho , w \sim \mathcal { P } } { \mathbb { E } } \left[ \left. w , \sum _ { h = 0 } ^ { H } \mathbf { r } ( s _ { h } , f _ { w } ) \right. \right] ,\tag{3}
$$

which maximizes the expected scalarized return $\mathcal { I } ( \pi ; w )$ of the composed policy $f _ { w } ( \pi _ { 1 } , \ldots , \pi _ { N } )$ under the target preference distribution $\mathcal { P }$

Importantly, the routing step is optimal under this formulation: when $f _ { w }$ lies in the convex hull of a fixed set of basis policies, the optimal solution reduces to deterministically selecting the basis policy that maximizes the return under w. This follows from the linearity of $\mathcal { I } ( \pi ; w )$ in $w ,$ , and is formalized in the following lemma.

Lemma 3.8. Fix a set of N policies $\{ \pi _ { 1 } , \ldots , \pi _ { N } \}$ . Suppose $f _ { w } ~ \in ~ \mathrm { c o n v } \{ \pi _ { 1 } , \ldots , \pi _ { N } \}$ and $\begin{array} { r } { d _ { h } ^ { f _ { w } } ( s , a ) = \sum _ { i = 1 } ^ { N } \alpha _ { i } ( w ) d _ { h } ^ { \pi _ { i } } ( s , a ) } \end{array}$ for ${ \alpha ( w ) \in \Delta ^ { N - 1 } }$ . There exists an optimal routing function fbfor Equation (3) such that, for every w $\in \Omega , \widehat { f } _ { w } = \pi _ { j }$ , where $j \in \arg \operatorname* { m a x } _ { i \in \{ 1 , 2 , . . . , N \} } \langle w , J ( \pi _ { i } ) \rangle$ .

Proof. For any w, since $f _ { w }$ lies in the convex hull of $\pi _ { 1 } , \ldots , \pi _ { N }$ , the state-action occupancy probabilities satisfy $\begin{array} { r } { d _ { h } ^ { f _ { w } } ( s , a ) = \sum _ { i } \alpha _ { i } ( w ) d _ { h } ^ { \pi _ { i } } ( s , a ) } \end{array}$ . Thus, the scalarized return satisfies

$$
\begin{array} { l } { { \displaystyle J _ { \alpha } ( { \boldsymbol f } _ { w } ) = \sum _ { h } \sum _ { a _ { h } , s _ { h } } d _ { h } ^ { f _ { w } } \big ( { \boldsymbol s } _ { h } , { \boldsymbol a } _ { h } \big ) \left. w , r _ { h } ( { \boldsymbol s } , { \boldsymbol a } ) \right. } \ ~ } \\ { { \displaystyle ~ = \sum _ { h } \sum _ { a _ { h } , s _ { h } } \sum _ { i = 1 } ^ { N } \alpha _ { i } ( w ) d _ { h } ^ { \pi _ { i } } \big ( { \boldsymbol s } _ { h } , { \boldsymbol a } _ { h } \big ) \left. w , r _ { h } ( { \boldsymbol s } , { \boldsymbol a } ) \right. = \sum _ { i = 1 } ^ { N } \alpha _ { i } ( w ) \left. J ( \pi _ { i } ) , w \right. } . } \end{array}\tag{4}
$$

The maximum over α $\in \Delta ^ { N - 1 }$ is therefore attained at an extreme point, corresponding to selecting a single basis policy. Thus, the optimal strategy is to choose the policy $\pi _ { i }$ that maximizes $\langle J ( \pi _ { i } ) , w \rangle$

Lemma 3.8 motivates the scoring and routing steps: we compute scores using equation (4) and route each trajectory to the basis policy that maximizes the score. Note that in the updating step, each basis policy is trained only on its assigned trajectories. This is crucial because a policy that performs well for one preference w may perform poorly for a different $w ^ { \prime }$ . Consequently, mixing trajectories across preferences can degrade learning [44]. Moreover, Proposition B.8 in $\mathbf { A }$ ppendix B.3 connects this algorithm to Γ, showing that the intuition underlying Theorem 3.6 carries over to its performance.

Finally, while our algorithm assumes N is given, in practice it can be selected via cross-validation (Section 4.3). One may adaptively grow N by identifying new clusters when existing policies do not explain incoming trajectories, drawing on the clustering literature [8, 28]. Compared to previous work [44] that learns one optimal policy for each objective, the routed ensemble algorithm can leverage potential overlap that is latent in the underlying RL environment.

## 3.4 Trajectory attribution

Another approach to address the problem of objective conflict is through trajectory selection, for instance, removing trajectories that are far from the learned policy [55]. This motivates the question of quantifying how individual training trajectories contribute to policy performance, a problem that we term trajectory attribution. This complements our theoretical analysis above by providing a data-driven measure of which trajectories help or hinder learning and by improving the interpretability of routed ensemble policies.

Algorithm 1 Learning Routed Ensemble Policies for Multi-Objective Markov Decision Processes   
Input: A MOMDP ; Initial state distribution $\rho ;$ Preference distribution ${ \overline { { \mathcal { P } } } } ;$ Number of policies $\overline { { N } }$   
Require: A policy learning algorithm $F ; \mathbf { A }$ policy evaluation function J; Number of iterations $T ;$   
Horizon H; Number of observed trajectories B per iteration   
Output: A set of $N$ policies   
$\iota \colon \pi _ { 1 } ^ { ( 0 ) } , \pi _ { 2 } ^ { ( 0 ) } , \ldots , \pi _ { N } ^ { ( 0 ) } : S \to \Delta ( A )$ , start from $N$ random initializations   
2: for $t { \overset { - } { = } } 1 , \dots , T$ do   
3: w<sub>t</sub> $\sim \mathcal { P } , \mathcal { D } _ { t } \gets \emptyset , S \gets N$ by B score matrix   
4: for $i = 1 , \ldots , B$ do   
5: $j _ { i } \gets$ sample uniformly from $\{ 1 , 2 , \ldots , N \}$   
6: $s _ { 0 } \gets \rho \mathrm { : }$ sample from the initial state distribution   
7: $\tau _ { i } ^ { ( t ) } \gets ( s _ { 0 } , a _ { 0 } , \ldots , s _ { H } , a _ { H } ) \colon$ : simulate from $s _ { 0 }$ for H steps using $\pi _ { j _ { i } } ^ { ( t - 1 ) }$   
8: for $j = 1 , \ldots , N$ do   
9: $S _ { j , i } \gets \sum _ { h = 0 } ^ { H } \left. w _ { t } , \sum _ { a \in \mathcal { A } } \mathbf { r } ( s _ { h } , a ) \pi _ { j } ^ { ( t - 1 ) } ( s _ { h } , a ) \right.$ ▷ Score policy $j$ on trajectory i   
10: end for   
11: $\mathcal { D } _ { t }  \mathcal { D } _ { t } \cup \{ ( \tau _ { i } ^ { ( t ) } , w _ { t } ) \}$   
12: end for   
13: for $j = 1 , \ldots , N$ do   
14: $\mathcal { D } _ { t , j } \gets \left\{ ( \tau _ { i } ^ { ( t ) } , w _ { t } ) \in \mathcal { D } _ { t } \ | \ j \in \arg \operatorname* { m a x } _ { 1 < i ^ { \prime } < N } S _ { j ^ { \prime } , i } \right\}$ ▷ Route trajectories to policy $j$   
$1 { \stackrel { \sim } { \leq } } j ^ { \prime } \leq N$   
15: $\pi _ { j } ^ { ( t ) } \gets F \big ( \pi _ { j } ^ { ( t - 1 ) } , { \mathcal { D } } _ { t , j } \big )$ $\triangleright$ Update policy $j$ using algorithm F   
16: end for   
17: end for   
18: return $\pi _ { 1 } ^ { ( T ) } , \pi _ { 2 } ^ { ( T ) } , \dots , \pi _ { N } ^ { ( T ) }$

A natural starting point for tackling trajectory attribution is via leave-one-out (LOO) retraining, which measures the change in performance when trajectory t is excluded:

$$
\mathcal { T } _ { t } ^ { \mathrm { L O O } } = \mathcal { V } \big ( [ \mathbf { 1 } ] _ { T } \big ) - \mathcal { V } \big ( [ \mathbf { 1 } ] _ { T } - e _ { t } \big ) ,
$$

where, for any subset indicator $S \in \{ 0 , 1 \} ^ { T }$ , we define $\mathcal { V } ( S ) : = \mathbb { E } _ { w \sim \mathcal { P } } \left[ \mathcal { I } ( \pi _ { S } ; w ) \right] , \pi _ { S } : = F ( \mathcal { D } _ { S } )$ with $\mathcal { D } _ { S } : = \left\{ ( \tau ^ { ( j ) } , w _ { j } ) \in \mathcal { D } : S _ { j } = 1 \right\}$ . Here $\pi _ { S }$ is the policy trained on the subset selected by S using algorithm $F , [ \mathbf { 1 } ] _ { T }$ is the all-ones vector, and $e _ { t }$ is the t-th standard basis vector.

Thus, $\mathcal { T } _ { t } ^ { \mathrm { L O O } }$ reflects the marginal importance of trajectory t. However, exact LOO retraining becomes computationally infeasible as the number of trajectories grows [26]. Moreover, it only captures one-at-a-time effects and misses nonlinear interactions among trajectories.

To address these limitations, Algorithm 2 introduces a scalable kernel-surrogate approach [26, 56] that approximates the subset-performance function $\mathcal { V } ( \cdot )$ . This procedure proceeds in three steps. First, we randomly sample n trajectory subsets $S _ { 1 } , \ldots , S _ { n } ,$ , where each $\bar { S } _ { i } \in \{ 0 , 1 \} ^ { T }$ indicates which trajectories are included. Second, for each subset $S _ { i } ,$ we estimate its performance using a surrogate loss $v ( S _ { i } ) \approx \mathcal { V } ( S _ { i } )$ . Third, we fit a kernel ridge regression surrogate: we form the radial basis function kernel $\Psi _ { i j } = \exp ( - \gamma \| S _ { i } - S _ { j } \| ^ { 2 } )$ , set $\bar { V = [ \bar { v } ( S _ { 1 } ) , \dots , \widehat { v } ( \bar { S _ { n } } ) ] ^ { \top } }$ , and compute $\alpha = ( \Psi + \lambda \operatorname { I d } _ { n } ) ^ { - 1 } V$ . This yields the predictor

$$
g _ { \alpha } ( S ) = \sum _ { i = 1 } ^ { n } \alpha _ { i } \exp ( - \gamma \| S _ { i } - S \| ^ { 2 } ) , \mathrm { f o r } \mathcal V ( S ) .
$$

Because this kernel model $g _ { \alpha } ( S )$ is nonlinear in the subset indicator, it can capture higher-order interactions such as synergy and antagonism among trajectories. Provided with $g _ { \alpha } ( S )$ , we could then use this to score each trajectory; for instance, by averaging over all the subsets that include a particular trajectory. Then, trajectories with the highest scores are selected for downstream applications.

The procedure above is formulated for trajectory attribution. However, the same procedure can be applied to objective attribution by sampling subsets of objectives. Each subset retains only the training contributions associated with its selected objectives. Removing an objective therefore excludes its contribution across all trajectories and allows us to estimate its effect on policy performance.

Algorithm 2 Kernel Surrogates for Trajectory Attribution Using Policy Gradient Estimation   
Input: A multi-objective RL environment with m reward objectives   
Requires: Subset sampling distribution ; Number of subsets n; ℓ -regularization parameter λ; A   
policy learning algorithm $\breve { F }$   
Output: A kernel surrogate model $g _ { \theta } : \{ 0 , 1 \} ^ { m } \to \mathbb { R }$   
1: ${ \theta ^ { \star } } \gets 1$ Learn a meta policy with all the m reward objectives simultaneously using $F$   
2: $S _ { 1 } , \ldots , S _ { n }  S$ ample n subsets from independently   
3: $\widehat { v } ( S _ { 1 } ) , \widehat { v } ( S _ { 2 } ) , \ldots , \widehat { v } ( S _ { n } ) \gets \mathtt { A }$ pply POLICYGRADEX (Algorithm 3) with $( S _ { 1 } , S _ { 2 } , \ldots , S _ { n } ; \theta ^ { \star } )$   
4: $\Psi _ { i j }  \mathrm { e x p } ( - \gamma \| S _ { i } - S _ { j } \| ^ { 2 } )$ , for all $1 \leq i , j \leq n$ ▷ Construct kernel matrix   
5: $V \dot { \gets } [ \widehat { v } ( S _ { 1 } ) , \dots , \widehat { v } ( S _ { n } ) ] ^ { \top }$ ▷ Collect estimated outcomes   
6: $\alpha  ( { \dot { \Psi } } + { \dot { \lambda } } \mathrm { I d } _ { n } ) ^ { - 1 } V$ ▷ Fit kernel ridge regression   
7: return $\begin{array} { r } { g _ { \theta } ( S ) = \sum _ { i = 1 } ^ { n } \alpha _ { i } \exp ( - \gamma \| S _ { i } - S \| ^ { 2 } ) } \end{array}$

Implementation. In online RL, exact retraining for each sampled subset is often impractical because the training data evolve. We therefore estimate the subset performance $v ( S _ { i } )$ locally around a reference policy trained on all trajectories. Specifically, we use policy gradients from local rollouts to approximate the parameter update induced by each subset, avoiding the cost of full retraining. The algorithm and the corresponding derivation are presented in Appendix A.

The experimental results support this approximation. In Figure 3, Appendix C.2, we show that the fitted kernel model $g _ { \alpha } ( [ \mathbf { 1 } ] _ { T } - e _ { t } )$ is strongly positively correlated with the LOO scores $\mathcal { I } _ { t } ^ { \mathrm { L O O } }$ and strongly negatively correlated with the objective conflict coefficient Γ. Moreover, Algorithm 2 achieves a Spearman rank correlation of 0.75 with the true policy-gradient influence scores, confirming that it identifies influential trajectories. More details of the approximation procedure are presented in Appendix A.

## 4 Experiments

Our experiments are designed to address the following two aspects: (1) to verify the theoretical insights from the regret analysis, and (2) to evaluate the empirical performance of our algorithms across multi-objective RL settings, including synthetic environments, continuous robotic control, and RLHF settings.

We highlight three findings. First, routing mitigates negative transfer and reduces cumulative regret in synthetic environments. Second, routed ensembles outperform baselines by a 4.3% relative margin in multi-task RL, up to 48% in multi-objective RL, and 17% in large language model benchmarks. Third, our kernel surrogates improve correlation with ground-truth marginal gains by 50% for trajectory attribution and improve downstream task-selection reward by 27%. Our code repository can be found at https://github.com/VirtuosoResearch/Multi-objective-reinforcement-learning-code/.

## 4.1 Evaluation in synthetic environments

We validate our theoretical insights in the following two environments (details can be found in Appendix C.1).

1. Learning Paretofrontier (Example 3.1). The environment has $m = 4$ conflicting objectives. The reward functions are designed such that maximizing one objective necessitates sacrificing the others, creating a complex, high-dimensional Pareto frontier. We sample preference vectors w uniformly from the simplex to evaluate the approximation of the frontier.

2. Learning under corrupted examples (Example 3.2). The environment has m = 2 objectives, where r<sub>1</sub> represents a clean task signal and $r _ { 2 }$ represents a corrupted signal. The target preference is fixed at [1, 0], meaning the agent solves a single-objective problem but is exposed to conflicting gradients from r<sub>2</sub> during exploration. The training preference is sampled uniformly in the simplex.

3. Routing under varying goal separation. We keep the environment dynamics fixed and vary the separation between goals. For each setting, we compute Γ from the resulting occupancies and measure the reduction in cumulative regret achieved by routing relative to the comparison method.

![](images/64850bd32ecb75b6588492ed3e0a30da87add8c0777f88cfc2890b8ccc4dd565.jpg)  
(a) Learning Pareto frontier

![](images/4693ab54b68a2aaa26422ee5fbf2ba876fc0a05e5c364129e1af05327f76106d.jpg)  
(b) Learning under corrupted examples  
Figure 2: We compare the cumulative regret of routed ensembles (red) against MO-UCBVI (green dashed), uniform ensembles (yellow), and the best stationary policy (blue dotted).

For Pareto frontier learning, Figure 2a shows that routed ensemble learning substantially reduces cumulative regret relative to MO-UCBVI [44], using one policy per objective. This suggests that routed ensembles maintain diverse basis policies that cover different regions of the frontier, while the single-policy baseline fails to capture the complex four-way trade-off.

For learning under corrupted examples, Figure 2b shows that our approach mitigates negative transfer from corrupted inputs and achieves near-optimal performance. This suggests that routing can filter out harmful interference and assign trajectories to support optimization of the clean objective r .

## 4.2 Evaluation on robotic control and language models

Next, we present our evaluations applied to several robotic control and language modeling settings.   
We first describe the setups.

Environments and models. We evaluate on four regimes, each with a different source of conflict: (i) shared dynamics on the 10-task MT10 benchmark from Meta-World [54]; (ii) explicit Pareto trade-offs on the continuous MO-HalfCheetah/Hopper/Ant and discrete Deep-Sea-Treasure/Fruit-Tree environments of MO-Gymnasium [12]; (iii) preference-induced conflict on HelpSteer2 [43], with Qwen3-1.7B fine-tuned with PPO via per-basis LoRA adapters; and (iv) trajectory attribution on Multitask Minigrid, MT10, and a synthetic synergy MTRL environment (See Appendix C.1). In all regimes, routing is applied as a drop-in module on top of the published base learner.

Baselines. We compare to the following baselines: (i) For multi-task RL, Soft Modularization [51], PaCo [39], CARE [38]; (ii) For continuous multi-objective RL, PGMORL [49], Q-Pensieve [18], CAPQL [31], GPI-LS [3], and learned-gate MoE; for discrete multi-objective RL, PQL [42], MP-MOQL [42], and Envelop [52]; (iii) For RLHF, a single-policy model; (iv) For trajectory attribution, Influence Functions [23], TracIn [35] adapted to online RL setting [17], and linear surrogate models [26]. The MoE and GPI-LS comparisons hold the base policy-learning algorithm and interaction budget fixed to examine different routing and policy-selection mechanisms. For the first three regimes, we also include uniform assignment, which randomly assigns trajectories to basis policies.

Evaluation. We use the following evaluation metrics: (i) success rate on MT10; (ii) hypervolume (HV) and utility (UT) on continuous MO-Gymnasium, and HV and cardinality (Card.) on discrete MO-Gymnasium; (iii) preference-induced conflict on HelpSteer2 [43] and UltraFeedback [10], using Qwen3-1.7B and Llama-3.2-1B fine-tuned with PPO via per-basis LoRA adapters; (iv) for trajectory attribution, linear data modeling score (LDS) and selection performance. LDS is the Spearman rank correlation between predicted and ground-truth marginal gains. Selection performance is the target objective return on a target objective when training within the top-3 most influential source tasks selected by each procedure.

Results for policy optimization. Our proposed routed ensemble Algorithm 1 consistently outperforms baselines in multi-task and multi-objective settings. The comparison is presented in Table 1. On MT10, it achieves an average success rate of 85.6%, outperforming soft modularization (82.0%) by a relative margin of 4.3%.

Table 1: Comparison of our approach with several baseline methods. We report the average success rate on Meta-World, HV and UT on MuJoCo, as well as HV and Cardinality on Grid-World.
<table><tr><td></td><td>Metrics</td><td>Soft Modularization</td><td>PaCo</td><td>CARE</td><td>Uniform Assignment</td><td>Algorithm 1</td></tr><tr><td>Meta-World</td><td>Success rate</td><td> $8 2 . 0 { \scriptstyle \pm 1 . 1 \% }$ </td><td> $7 3 . 1 { \pm } 1 . 1 \%$ </td><td> $8 0 . 2 { \scriptstyle \pm 1 . 8 } \%$ </td><td> $8 1 . 3 { \scriptstyle \pm 1 . 4 } \%$ </td><td> ${ \bf 8 5 . 6 _ { \pm 1 . 8 } \% }$ </td></tr><tr><td></td><td>Metrics</td><td>PGMORL</td><td>Q-Pensieve</td><td>CAPQL</td><td>Uniform Assignment Algorithm 1</td><td></td></tr><tr><td rowspan="2">MO-HalfCheetah2d</td><td> $\mathrm { H V } ( \times 1 0 ^ { 6 } )$ </td><td> $0 . 1 1 _ { \pm 0 . 0 8 }$ </td><td> $4 . 8 5 _ { \pm 0 . 0 6 }$ </td><td> $3 . 2 9 _ { \pm 0 . 1 8 }$ </td><td> $3 . 1 7 _ { \pm 0 . 1 1 }$ </td><td> ${ \bf 5 . 1 7 _ { \pm 0 . 1 6 } }$ </td></tr><tr><td> $\mathrm { U T } ( \times 1 0 ^ { 2 } )$ </td><td> $0 . 1 2 _ { \pm 0 . 1 3 }$ </td><td> $2 . 1 4 _ { \pm 0 . 1 3 }$ </td><td> $0 . 6 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 . 0 1 _ { \pm 0 . 1 2 }$ </td><td> $\mathbf { 2 . 4 7 _ { \pm 0 . 1 4 } }$ </td></tr><tr><td rowspan="2">MO-Hopper</td><td> $\mathrm { H V } ( \times 1 0 ^ { 7 } )$ </td><td> $1 . 2 3 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 . 6 1 _ { \pm 0 . 3 2 }$ </td><td> $1 . 3 9 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 . 5 6 _ { \pm 0 . 1 6 }$ </td><td> $\mathbf { 1 . 8 3 { \scriptstyle \pm 0 . 1 4 } }$ </td></tr><tr><td> $\mathrm { U T } ( \times 1 0 ^ { 2 } )$ </td><td> $0 . 0 4 _ { \pm 0 . 0 2 }$ </td><td> $1 . 4 7 _ { \pm 0 . 1 4 }$ </td><td> $1 . 2 9 _ { \pm 0 . 1 2 }$ </td><td> $1 . 1 7 _ { \pm 0 . 0 9 }$ </td><td> ${ \bf 1 . 5 6 _ { \pm 0 . 0 6 } }$ </td></tr><tr><td rowspan="2">MO-Ant</td><td> $\mathrm { H V } ( \times 1 0 ^ { 7 } )$ </td><td> $0 . 1 7 _ { \pm 0 . 1 1 }$ </td><td> $1 . 8 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $0 . 8 4 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 . 8 1 _ { \pm 0 . 1 5 }$ </td><td> $\mathbf { 2 . 0 3 _ { \pm 0 . 1 6 } }$ </td></tr><tr><td> $\mathrm { U T } ( \times 1 0 ^ { 2 } )$ </td><td> $0 . 2 1 _ { \pm 0 . 0 6 }$ </td><td> $1 . 7 5 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 . 7 6 _ { \pm 0 . 1 6 }$ </td><td> $1 . 6 8 _ { \pm 0 . 0 8 }$ </td><td> ${ \bf 1 . 8 2 _ { \pm 0 . 1 2 } }$ </td></tr><tr><td></td><td>Metrics</td><td> $\mathrm { P Q L }$ </td><td> $\mathbf { M P M O Q L }$ </td><td> $\mathrm { E n v e l o p }$ </td><td>Uniform Assignment</td><td>Algorithm 1</td></tr><tr><td rowspan="2">Deep-Sea-Treasure</td><td> $\mathrm { H V } ( \times 1 0 ^ { 3 } )$ </td><td> $0 . 7 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $0 . 2 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $0 . 7 9 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 6 2 _ { \pm 0 . 2 0 }$ </td><td> ${ \bf 1 . 1 0 { \scriptstyle \pm 0 . 0 0 } }$ </td></tr><tr><td> $\mathbf { C a r d . } \left( \times 1 0 \right)$ </td><td> $0 . 8 3 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $0 . 1 7 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 1 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 8 0 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $\mathbf { 1 . 0 0 } _ { \pm 0 . 0 0 }$ </td></tr><tr><td rowspan="2">Fruit-Tree</td><td> $\mathrm { H V } ( \times 1 0 ^ { 3 } )$ </td><td> $6 . 6 9 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $1 . 0 1 _ { \pm 0 . 2 4 }$ </td><td> $3 . 9 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 . 2 9 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $\mathbf { 9 . 3 0 } _ { \pm 0 . 0 0 }$ </td></tr><tr><td> $\mathbf { C a r d . } \left( \times 1 0 \right)$ </td><td> $3 . 4 0 _ { \pm 0 . 3 6 }$ </td><td> $0 . 1 7 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $3 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 . 4 7 _ { \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 4 0 { \scriptstyle \pm 0 . 0 0 } }$ </td></tr></table>

Table 2: Comparison results for RLHF settings between our approach, using a single policy, and uniform assignment to the same number of basis policies.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Synthetic</td><td colspan="2">Qwen3-1.7B</td><td colspan="2">Llama-3.2-1B</td></tr><tr><td>HelpSteer2</td><td>UltraFeedback</td><td>HelpSteer2</td><td>UltraFeedback</td></tr><tr><td>Single Policy</td><td>0.48</td><td>0.51</td><td>0.46</td><td>0.34</td><td>0.52</td></tr><tr><td>Uniform Assignment</td><td>0.49</td><td>0.47</td><td>0.47</td><td>0.35</td><td>0.54</td></tr><tr><td>Algorithm 1</td><td>0.56</td><td>0.54</td><td>0.48</td><td>0.35</td><td>0.55</td></tr></table>

Our routing algorithm also improves RLHF performance on both the synthetic environment and real-data benchmarks. In Table 2, we compare Algorithm 1, uniform assignment, and learning a single policy for each objective. Our algorithm consistently increases hypervolume on the induced frontier. In the synthetic environment, our method improves hypervolume by 17% over the single-policy baseline and 14% over the uniform ensemble. Averaged across the two models, the corresponding improvements are 5.0% and 8.2% on HelpSteer2, and 6.2% and 3.0% on UltraFeedback.

Next, we present comparisons in continuous MO-Gymnasium environments in Table 3. Our routing algorithm yields an 8% average improvement over Q-Pensieve and achieves higher HV than learnedgate MoE and GPI-LS on all three benchmarks. The capacity-matched MoE comparison supports the benefit of trajectory routing beyond additional expert capacity. These results suggest that routing trajectories across basis policies helps distinguish aligned from conflicting data and approximates the Pareto frontier more efficiently than uniform assignment.

Results for trajectory attribution. By applying Algorithm 2 at the objective level, we can achieve higher LDS scores and downstream task-selection performance. The results are presented in Table 5. The kernel surrogate improves LDS by 50% over attribution baselines across Multitask-Minigrid and MT10. Moreover, when used to select source tasks for co-training, kernel-surrogate attribution improves the target reward by 27%, demonstrating that the learned attributions are useful for downstream policy learning.

## 4.3 Ablation studies

Ensemble policies reduce coverability. We measure coverability using the union of state-action occupancies in the synthetic environment. A single policy results in a high $\chi ^ { 2 }$ -divergence of 1.3. The routed ensembles distribute exploration, reducing the metric to 0.2. This suggests that routing mitigates the exploration bottleneck.

Varying the number of basis policies in the ensemble. We vary N in 2, 3, 4, 5 to evaluate the sensitivity of Algorithm 1 on three MuJoCo benchmarks. For MO-HalfCheetah-v4, the Hypervolumes are highest at $5 . 2 1 \times 1 0 ^ { 6 }$ when $N = 3 .$ For MO-Hopper, it is $1 . 8 7 \times 1 0 ^ { 7 }$ when $N = 3$ . For MO-Ant, it is $2 . 1 2 \times 1 0 ^ { 7 }$ when $N = 5$ . See Appendix C.2 for the illustration of these ablation results.

Table 3: Hypervolume comparison with learned-gate MoE and GPI-LS.  
Table 4: Measuring conflict and regret reduction under varying goal separation levels.
<table><tr><td>Method</td><td>HalfCheetah2d</td><td>Hopper</td><td>Ant</td></tr><tr><td>MoE</td><td> $4 . 4 0 _ { \pm 0 . 0 9 }$ </td><td> $1 . 5 6 _ { \pm 0 . 1 3 }$ </td><td> $1 . 6 0 { \scriptstyle \pm 0 . 1 8 }$ </td></tr><tr><td>GPI-LS</td><td> $5 . 0 0 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $1 . 5 0 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $1 . 8 5 { \scriptstyle \pm 0 . 2 4 }$ </td></tr><tr><td>Alg.1</td><td> ${ \bf 5 . 1 7 _ { \pm 0 . 1 6 } }$ </td><td> $\mathbf { 1 . 8 3 _ { \pm 0 . 1 4 } }$ </td><td> $\mathbf { 2 . 0 3 _ { \pm 0 . 1 6 } }$ </td></tr></table>

<table><tr><td>Separation</td><td>Γ</td><td>Regret reduction</td></tr><tr><td>0</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>2</td><td> $0 . 7 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 . 7 7 { \scriptstyle \pm 0 . 1 1 }$ </td></tr><tr><td>8</td><td>1.00±0.00</td><td> $3 6 . 0 3 { \scriptstyle \pm 0 . 5 7 }$ </td></tr></table>

Table 5: We report the average target reward when the target policy is trained with source tasks selected by each attribution algorithm.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Multitask Minigrid</td><td colspan="2">Meta-World MT10</td><td colspan="2">Synergy MTRL</td></tr><tr><td>LDS</td><td>Reward</td><td>LDS</td><td>Reward</td><td>LDS</td><td>Reward</td></tr><tr><td>Influence functions</td><td> $- 0 . 2 3 _ { \pm 0 . 0 8 }$ </td><td> $0 . 0 7 _ { \pm 0 . 1 3 }$ </td><td> $0 . 7 1 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 6 . 3 { \scriptstyle \pm 1 . 4 }$ </td><td> $0 . 0 4 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 . 7 7 _ { \pm 0 . 6 7 }$ </td></tr><tr><td>TracIn</td><td> $0 . 3 1 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 2 5 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $0 . 4 5 _ { \pm 0 . 1 5 }$ </td><td> $1 6 . 0 { \scriptstyle \pm 2 . 3 }$ </td><td> $0 . 1 6 _ { \pm 0 . 0 4 }$ </td><td> $2 . 6 5 _ { \pm 0 . 0 3 }$ </td></tr><tr><td>Linear surrogates</td><td> $0 . 3 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $0 . 3 3 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $0 . 4 2 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 8 { \scriptstyle \pm 4 . 5 }$ </td><td> $0 . 1 6 _ { \pm 0 . 0 4 }$ </td><td> $2 . 6 7 _ { \pm 0 . 0 1 }$ </td></tr><tr><td>Algorithm 2</td><td> $\mathbf { 0 . 4 1 { \scriptstyle \pm 0 . 0 4 } }$ </td><td> $\mathbf { 0 . 4 5 { \scriptstyle \pm 0 . 1 7 } }$ </td><td> $\mathbf { 0 . 8 0 _ { \pm 0 . 0 4 } }$ </td><td> $\mathbf { 1 8 . 8 \pm 5 . 6 }$ </td><td> $\mathbf { 0 . 9 1 { \scriptstyle \pm 0 . 0 1 } }$ </td><td> $\mathbf { 5 . 8 5 { \scriptstyle \pm 0 . 0 1 } }$ </td></tr></table>

Nonlinear kernels improve trajectory attribution. We compare linear, quadratic, cubic, and RBF kernels by measuring the fitting loss of the kernel surrogate model in equation (5) on the synthetic dataset. The linear kernel converges to a loss of 0.2, while the quadratic, cubic, and RBF kernels all achieve a lower loss of 0.1. This gap demonstrates that the relationship between objective subsets and policy returns involves high-order interactions that a linear model cannot capture.

From occupancy conflict to regret reduction. Lastly, we evaluate how occupancy conflict relates to regret reduction via routing in Table 4. We vary the separation between two reward goals. As goal separation increases, measured Γ rises from 0 to 1, while regret reduction from routing rises from 0 to 36.03. Across these settings, routing is more beneficial when preference-induced occupancies diverge. This trend is consistent with Theorem 3.6, where larger occupancy conflict increases the conflict-dependent term in the regret bound.

## 5 Related Work

A reinforcement learning environment is typically formalized by means of a Markov decision process [40, 2]. Existing approaches in the literature on multi-objective reinforcement learning (MORL) generally fall into two categories. The first reduces MORL to single-objective RL through scalarization, optimizing a fixed or sampled preference vector to learn a single policy [14, 33, 32, 38]. The second, which includes our work, examines the multi-objective nature of the problem by learning multiple policies (e.g., to approximate the Pareto frontier) to address sharp variation in optimal policies across preferences [42, 44, 30]. Recent work [55] proposes a convex relaxation algorithm for partitioning multiple objectives into disjoint, related groups. We build on this line of work by providing a theoretical characterization ofobjective conflicts in MORL, introducing a routed ensemble algorithm, and developing trajectory-level attribution to interpret model behavior. One interesting future direction is to examine the policy learning dynamics in multi-objective Markov decision processes [22, 24]. Lastly, sequential decision-making problems also occur in experimental design, where one often needs to roll out an experiment through staggered adoption [48, 47].

Recent work studies both offline and online RL with linear function approximation [15, 16]. It would be interesting to understand the techniques therein and examine whether it would be possible to extend to learning optimal policies under multiple objectives under linear Bellman completeness. More broadly, we refer the reader to a recent monograph by Foster and Rakhlin [13] for references.

Influence estimation techniques [23, 35] have recently been extended from supervised learning to reinforcement learning to trace model behaviors. In the context of offline RL, [11] proposes a framework to attribute specific policy decisions to past trajectories. A fundamental challenge in the online RL setting is the circular dependency where current policy updates influence future data distribution. Recent work [17] addresses this by employing local data attribution via gradient similarity to identify and filter harmful transitions.

## 6 Conclusion

In this paper, we study how to learn policies that balance competing objectives in multi-objective reinforcement learning. We first analyze the theoretical limits of transfer learning across preferences, highlighting that structural misalignment between objectives is the primary cause of negative transfer. To overcome this challenge, we propose a routed ensemble algorithm guided by a new measure, the objective conflict coefficient. Furthermore, we introduce a kernel-based trajectory-attribution method that captures widespread nonlinear interactions among trajectories and can improve subsequent policy learning. The experimental results demonstrate that our approach achieves superior sample efficiency and attribution accuracy compared to state-of-the-art baselines across diverse benchmarks.

One limitation of our theoretical framework is that the regret bounds do not directly apply to the routed ensemble algorithm. Further understanding the sample complexity of learning ensemble policies in multi-objective Markov decision processes is an interesting question for future work.

## Acknowledgment

We are grateful to the anonymous reviewers for their constructive comments that greatly improved this paper. Thanks to Dongruo Zhou for pointing out references to prior work. This work received financial assistance from NSF award IIS-2412008 and a startup fund from Northeastern University. Zhenshuo Zhang received additional financial support from Northeastern University’s Khoury PhD fellowships. R. Xiong is supported in part by the URC award at Emory University.

## References

[1] A. Agarwal, N. Jiang, S. M. Kakade, and W. Sun. “Reinforcement learning: Theory and algorithms”. In: CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep 32 (2019), p. 96.

[2] A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan. “On the theory of policy gradient methods: Optimality, approximation, and distribution shift”. In: Journal of Machine Learning Research 22.98 (2021), pp. 1–76.

[3] L. N. Alegre, A. L. Bazzan, D. M. Roijers, A. Nowé, and B. C. da Silva. “Sample-efficient multi-objective learning via generalized policy improvement prioritization”. In: arXiv preprint arXiv:2301.07784 (2023).

[4] M. Bateni, V. Cohen-Addad, Y. Gu, S. Lattanzi, S. Meierhans, and C. Mohri. “Algorithmic Thinking Theory”. In: arXiv preprint arXiv:2512.04923 (2025).

[5] A. Beygelzimer, S. Kale, and H. Luo. “Optimal and adaptive algorithms for online boosting”. In: International Conference on Machine Learning. PMLR. 2015, pp. 2323–2331.

[6] E. Chen, N. Dullerud, T. Niedermayr, E. Kidd, R. Senanayake, P. W. Koh, S. Koyejo, and C. Guestrin. “MoSH: Modeling Multi-Objective Tradeoffs with Soft and Hard Bounds”. In: arXiv preprint arXiv:2412.06154 (2024).

[7] F. Chen, Z. Jia, A. Rakhlin, and T. Xie. “Outcome-Based Online Reinforcement Learning: Algorithms and Fundamental Limits”. In: arXiv preprint arXiv:2505.20268 (2025).

[8] J. Chen, E. Sadeqi Azer, and Q. Zhang. “A practical algorithm for distributed clustering and outlier detection”. In: Advances in Neural Information Processing Systems 31 (2018).

[9] Y. Chi, Y. Chen, and Y. Wei. “Statistical and algorithmic foundations of reinforcement learning”. In: Tutorials in Operations Research: Advances in Analytics and Operations Research: Improving Decisions to Secure the Future. INFORMS, 2025, pp. 104–144.

[10] G. Cui, L. Yuan, N. Ding, G. Yao, B. He, W. Zhu, Y. Ni, G. Xie, R. Xie, Y. Lin, Z. Liu, and M. Sun. “Ultrafeedback: Boosting language models with scaled AI feedback”. In: arXiv preprint arXiv:2310.01377 (2023).

[11] S. V. Deshmukh, A. Dasgupta, B. Krishnamurthy, N. Jiang, C. Agarwal, G. Theocharous, and J. Subramanian. “Explaining RL Decisions with Trajectories”. In: The Eleventh International Conference on Learning Representations. 2023.

[12] F. Felten, L. N. Alegre, A. Nowé, A. L. C. Bazzan, E. G. Talbi, G. Danoy, and B. C. d. Silva. “A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning”. In: Proceedings ofthe 37th Conference on Neural Information Processing Systems (NeurIPS 2023). 2023.

[13] D. J. Foster and A. Rakhlin. “Foundations of reinforcement learning and interactive decision making”. In: arXiv preprint arXiv:2312.16730 (2023).

[14] Z. Gábor, Z. Kalmár, and C. Szepesvári. “Multi-criteria reinforcement learning.” In: ICML. Vol. 98. 1998, pp. 197–205.

[15] N. Golowich and A. Moitra. “Linear bellman completeness suffices for efficient online reinforcement learning with few actions”. In: The Thirty Seventh Annual Conference on Learning Theory. PMLR. 2024, pp. 1939–1981.

[16] N. Golowich and A. Moitra. “The role of inherent bellman error in offline reinforcement learning with linear function approximation”. In: Reinforcement Learning Conference. 2024.

[17] Y. Hu, F. Wu, H. Ye, D. Forsyth, J. Zou, N. Jiang, J. W. Ma, and H. Zhao. “A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems. 2025.

[18] W. Hung, B.-K. Huang, P.-C. Hsieh, and X. Liu. “Q-Pensieve: Boosting Sample Efficiency of Multi-Objective RL Through Memory Sharing of Q-Snapshots”. In: arXiv preprint arXiv:2212.03117 (2022).

[19] C. Jin, Q. Liu, and S. Miryoosefi. “Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms”. In: Advances in neural information processing systems 34 (2021), pp. 13406–13418.

[20] M. Joglekar, J. Chen, G. Wu, J. Yosinski, J. Wang, B. Barak, and A. Glaese. “Training LLMs for Honesty via Confessions”. In: arXiv preprint arXiv:2512.08093 (2025).

[21] V. Joshi, Z. Xu, B. Liu, P. Stone, and A. Zhang. “Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks”. In: Reinforcement Learning Conference. 2025.

[22] S. M. Kakade. “A natural policy gradient”. In: Advances in neural information processing systems 14 (2001).

[23] P. W. Koh and P. Liang. “Understanding black-box predictions via influence functions”. In: International conference on machine learning. PMLR. 2017, pp. 1885–1894.

[24] G. Lan. “Policy mirror descent for reinforcement learning: linear convergence, new sampling complexity, and generalized problem classes: G. Lan”. In: Mathematical programming 198.1 (2023), pp. 1059–1106.

[25] T. Lattimore and C. Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.

[26] D. Li, H. Nguyen, and H. R. Zhang. “Identification of Negative Transfers in Multitask Learning Using Surrogate Models”. In: Transactions on Machine Learning Research (2023).

[27] D. Li, Z. Zhang, L. Wang, and H. R. Zhang. “MoPLEx: Estimating Plackett-Luce Mixture Models for Multi-Objective Alignment”. In: arXiv preprint arXiv:2608.25200 (2026).

[28] Y. Li, Y. Song, and Q. Zhang. “Learning to cluster via same-cluster queries”. In: Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021, pp. 978–987.

[29] M. L. Littman. “Markov games as a framework for multi-agent reinforcement learning”. In: Machine learning proceedings 1994. Elsevier, 1994, pp. 157–163.

[30] E. Liu, Y.-C. Wu, X. Huang, C. Gao, R.-J. Wang, K. Xue, and C. Qian. “Pareto set learning for multi-objective reinforcement learning”. In: Proceedings ofthe AAAI Conference on Artificial Intelligence. Vol. 39. 18. 2025, pp. 18789–18797.

[31] H. Lu, D. Herman, and Y. Yu. “Multi-objective reinforcement learning: Convexity, stationarity and pareto optimality”. In: The Eleventh International Conference on Learning Representations. 2023.

[32] S. Mannor and N. Shimkin. “A geometric approach to multi-criterion reinforcement learning”. In: Journal ofmachine learning research 5.Apr (2004), pp. 325–360.

[33] K. Miettinen and M. M. Mäkelä. “On scalarizing functions in multiobjective optimization”. In: OR spectrum 24.2 (2002), pp. 193–213.

[34] C. H. Papadimitriou and M. Yannakakis. “On the approximability of trade-offs and optimal access of web sources”. In: Proceedings 41st annual symposium on foundations of computer science. IEEE. 2000, pp. 86–92.

[35] G. Pruthi, F. Liu, S. Kale, and M. Sundararajan. “Estimating training data influence by tracing gradient descent”. In: Advances in Neural Information Processing Systems 33 (2020), pp. 19920–19930.

[36] I. Rubinstein and S. B. Hopkins. “On the Accuracy of Newton Step and Influence Function Data Attributions”. In: arXiv preprint arXiv:2512.12572 (2025).

[37] S. Sengupta, G. Liu, O. Gottesman, J. W. Durham, M. Kearns, A. Roth, and M. Caldara. “Multi-Objective Reinforcement Learning for Large-Scale Tote Allocation in Human-Robot Collaborative Fulfillment Centers”. In: arXiv preprint arXiv:2602.24182 (2026).

[38] S. Sodhani, A. Zhang, and J. Pineau. “Multi-task reinforcement learning with context-based representations”. In: International Conference on Machine Learning. 2021, pp. 9767–9779.

[39] L. Sun, H. Zhang, W. Xu, and M. Tomizuka. “Paco: Parameter-compositional multi-task reinforcement learning”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 21495–21507.

[40] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. Second. Adaptive Computation and Machine Learning series. MIT Press, 2018.

[41] M. Towers, A. Kwiatkowski, J. Terry, J. U. Balis, G. De Cola, T. Deleu, M. Goulão, A. Kallinteris, M. Krimmel, A. KG, R. Perez-Vicente, A. Pierré, S. Schulhoff, J. J. Tai, H. Tan, and O. G. Younis. “Gymnasium: A standard interface for reinforcement learning environments”. In: NeurIPS Datasets and Benchmarks (2025).

[42] K. Van Moffaert and A. Nowé. “Multi-objective reinforcement learning using sets of pareto dominating policies”. In: The Journal of Machine Learning Research (2014), pp. 3483–3512.

[43] Z. Wang, Y. Dong, O. Delalleau, J. Zeng, G. Shen, D. Egert, J. J. Zhang, M. N. Sreedhar, and O. Kuchaiev. “Helpsteer 2: Open-source dataset for training top-performing reward models”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 1474–1501.

[44] J. Wu, V. Braverman, and L. Yang. “Accommodating picky customers: Regret bound and exploration complexity for multi-objective reinforcement learning”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 13112–13124.

[45] S. Wu, H. R. Zhang, and C. Ré. “Understanding and improving information transfer in multitask learning”. In: ICLR (2020).

[46] T. Xie, D. J. Foster, Y. Bai, N. Jiang, and S. M. Kakade. “The Role of Coverage in Online Reinforcement Learning”. In: The Eleventh International Conference on Learning Representations. 2023.

[47] R. Xiong. “Automated experimental design with optimization from historical data simulations”. In: Available at SSRN 5126080 (2025).

[48] R. Xiong, A. Chin, and S. J. Taylor. “Data-driven switchback experiments: Theoretical tradeoffs and empirical bayes designs”. In: arXiv preprint arXiv:2406.06768 (2024).

[49] J. Xu, Y. Tian, P. Ma, D. Rus, S. Sueda, and W. Matusik. “Prediction-guided multi-objective reinforcement learning for continuous robot control”. In: International conference on machine learning. PMLR. 2020, pp. 10607–10616.

[50] F. Yang, H. R. Zhang, S. Wu, C. Ré, and W. J. Su. “Precise high-dimensional asymptotics for quantifying heterogeneous transfers”. In: Journal ofMachine Learning Research 26.113 (2025), pp. 1–88.

[51] R. Yang, H. Xu, Y. Wu, and X. Wang. “Multi-task reinforcement learning with soft modularization”. In: Advances in Neural Information Processing Systems. Vol. 33. 2020, pp. 4767– 4777.

[52] R. Yang, X. Sun, and K. Narasimhan. “A generalized algorithm for multi-objective reinforcement learning and policy adaptation”. In: NeurIPS 32 (2019).

[53] T. Yang, S. Cen, Y. Wei, Y. Chen, and Y. Chi. “Federated natural policy gradient and actor critic methods for multi-task reinforcement learning”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 121304–121375.

[54] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine. “Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning”. In: Conference on Robot Learning. 2020, pp. 1094–1100.

[55] Z. Zhang, M. Duan, Y. Ye, and H. R. Zhang. “Scalable multi-objective and meta reinforcement learning via gradient estimation”. In: arXiv preprint arXiv:2511.12779 (2025).

[56] Z. Zhang, M. Duan, and H. Zhang. “Efficient estimation of kernel surrogate models for task attribution”. In: International Conference on Learning Representations. 2026, pp. 61822– 61848.

## A Policy Gradient Estimation

In Algorithm 2, we compute the kernel by minimizing the following objective over the training set:

$$
\operatorname* { m i n } _ { g _ { \alpha } \in \mathcal { H } } \sum _ { i = 1 } ^ { n } \left( v ( S _ { i } ) - g _ { \alpha } ( S _ { i } ) \right) ^ { 2 } + \lambda \| g _ { \alpha } \| _ { \Psi } ^ { 2 } ,\tag{5}
$$

where $\lambda > 0$ is a regularization parameter and $\Psi$ is an $n \times n$ kernel matrix with entries $\Psi _ { i , j } =$ $\psi ( S _ { i } , S _ { j } )$ . Since S is binary, consider the radial basis function (RBF) kernel: $\psi ( S _ { i } , S _ { j } ) = \exp  ( -$ γ $| S _ { i } - S _ { j } | | ^ { 2 } )$ . By the representer theorem of kernel methods, the minimizer is:

$$
g _ { \alpha } ( S ) = \sum _ { i = 1 } ^ { n } \alpha _ { i } \psi \big ( S _ { i } , S \big ) ,
$$

where the coefficients $\alpha$ are obtained via $( \Psi + \lambda \operatorname { I d } ) ^ { - 1 } V$ with $V = [ v ( S _ { 1 } ) , \ldots , v ( S _ { n } ) ] ^ { \top }$ , and Id is the identity matrix. Below we describe how we obtain $v ( S _ { i } )$

First-order approximation. We approximate the PPO surrogate objective (See Appendix C.3 for a detailed review of PPO) using a first-order Taylor expansion. Let $\varphi _ { t } = \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } | s _ { t } ) | _ { \theta = \theta ^ { * } }$ be the gradient feature vector. The policy objective is approximated as:

$$
J ( \theta ) \approx \mathbb { E } _ { t } \left[ \widehat { A } _ { t } \cdot ( 1 + \varphi _ { t } ^ { \top } \Delta \theta ) \right] ,
$$

where $\Delta \theta = \theta - \theta ^ { * }$ and $\widehat { A } _ { t }$ is the advantage function.

Directly maximizing this linear approximation is unbounded. To obtain a robust estimator, we reformulate this optimization as a weighted binary classification problem. Intuitively, we seek an update $\Delta \theta$ that aligns the policy change $\varphi _ { t } ^ { \top } \Delta \theta$ with the sign of the advantage $\widehat { A } _ { t }$ . By defining a binary label $y _ { t } = \mathrm { s i g n } ( \widehat { A } _ { t } )$ and a sample weight $\eta _ { t } = | \widehat { A } _ { t } |$ , we maximize the margin $y _ { t } \varphi _ { t } ^ { \top } \Delta \theta$ . This is effectively approximated by minimizing the weighted logistic loss, which provides a bounded surrogate for the sign-agreement objective:

$$
v ( S _ { i } ) : = \operatorname* { m i n } _ { \Delta \theta } L _ { S _ { j } } ( \theta ^ { * } + \Delta \theta ) \qquad \mathrm { w h e r e } ~ L _ { S _ { j } } ( \theta ^ { * } + \Delta \theta ) = \frac { 1 } { | \mathscr { D } _ { S _ { i } } | } \sum _ { t \in \mathscr { D } _ { S _ { i } } } \eta _ { t } \cdot \log \left( 1 + \exp \left( - y _ { t } \tilde { \varphi } _ { t } ^ { \top } \Delta \theta \right) \right)
$$

This formulation enables rapid estimation of the policy performance for any subset $S _ { i }$ without full retraining. As shown in Table 6, the linearization error remains low $( < 2 \% )$ within the trust region. Another challenge is the high dimensionality of gradient vectors. We address this by applying random Gaussian projections from $\varphi _ { t }$ to a low-dimensional space. By the Johnson–Lindenstrauss Lemma, this projection provably preserves the pairwise similarity between pairs of gradients. Solving the regression problem in the reduced space then takes only a few seconds.

Table 6: We empirically find that the approximation error of ϵ is negligible for several interactive RL environments. We report the mean and standard deviation of the approximation error across 10 random subsets. We defer a detailed description of the setup to the experiments section.
<table><tr><td>Distance</td><td>MT10</td><td>CartPole</td><td>Highway</td><td>Lunarlander</td></tr><tr><td>0.1%</td><td> $0 . 0 1 { \scriptstyle \pm 0 . 0 1 } \%$ </td><td> $0 . 1 2 _ { \pm 0 . 1 4 } \%$ </td><td> $0 . 0 2 _ { \pm 0 . 0 2 } \%$ </td><td> $0 . 0 6 _ { \pm 0 . 0 1 } \%$ </td></tr><tr><td>0.5%</td><td> $0 . 4 3 { \scriptstyle \pm 0 . 7 3 } \%$ </td><td> $0 . 7 3 { \scriptstyle \pm 0 . 1 0 } \%$ </td><td> $0 . 1 1 { \scriptstyle \pm 0 . 0 9 } \%$ </td><td> $0 . 0 3 { \scriptstyle \pm 0 . 0 2 } \%$ </td></tr><tr><td>1.0%</td><td> $0 . 3 2 { _ { \pm 0 . 5 6 } } \%$ </td><td> $0 . 9 8 _ { \pm 0 . 6 5 } \%$ </td><td> $2 . 0 4 _ { \pm 0 . 5 8 } \%$ </td><td> $0 . 4 8 { \scriptstyle \pm 0 . 0 1 } \%$ </td></tr></table>

Specifically, we compute the relative residual sum of squares (RSS) error across several RL environments: MT10 from MetaWorld [54], CartPole, LunarLander, and Highway from Gymnasium [41]. In each case, a four-layer MLP policy is trained, and gradients are estimated from 2,048 steps per task. The results are averaged over 10 randomly sampled subsets of size 5. As shown in Table 6, the RSS error remains below $2 \%$ when $\lVert \theta - \theta ^ { * } \rVert$ is small with a proper initialization $\theta ^ { \star }$ , indicating high accuracy in the linear approximation. The error gradually increases to around 10% for perturbations up to $5 \% .$ . These results validate the surrogate model’s ability to approximate outcomes near the meta-policy.

Algorithm 3 Scalable Kernel Surrogate Estimation using Policy Gradient Estimation (POLICY-  
GRADEX)   
Input: n subsets $S _ { 1 } , \ldots , S _ { n }$ of $\{ 1 , \ldots , m \}$ ; Initial meta-policy $\pi _ { \theta ^ { \star } }$   
Require: Number of episodes $\dot { T ; }$ Projection dimension d   
1: $\pi _ { \boldsymbol { \theta } ^ { \star } } $ Set up the meta-policy for all objectives in $\{ 1 , \ldots , m \}$   
2: $P  \mathbf { A }$ p by d isotropic Gaussian random projection matrix with mean zero and variance $\textstyle { \frac { 1 } { d } }$   
3: for $t = \bar { 1 } , \ldots , T$ do   
4: for $i \in \{ 1 , \ldots , m \}$ do   
5: $\widetilde { \varphi } _ { i , t } \gets P ^ { \top }$ ∇θ <sup>log</sup> $\left. \pi _ { \boldsymbol { \theta } } ( a _ { i , t } \mid s _ { i , t } ) \right| _ { \boldsymbol { \theta } = \boldsymbol { \theta } ^ { \star } }$   
6: $\eta _ { i , t } \gets | \widehat { A } _ { i , t } |$   
7: $y _ { i , t } \gets \mathrm { s i g n } ( \widehat { A } _ { i , t } )$   
8: end for   
9: end for   
10: for $j = 1 , \dotsc , n$ do   
11: $\begin{array} { r } { \Delta \widehat { \theta } _ { d } ^ { ( j ) } \gets \arg \operatorname* { m i n } _ { \Delta \theta _ { d } \in \mathbb { R } ^ { d } } \widehat { L } _ { S _ { j } } ( \theta ^ { \star } + P \Delta \theta _ { d } ) } \end{array}$   
12: $\widehat { \theta } ^ { ( j ) } \gets \theta ^ { \star } + P \Delta \widehat { \theta } _ { d } ^ { ( j ) }$   
13: $\widehat { v } ( S _ { j } ) \gets - \widehat { L } _ { S _ { j } } ( \widehat { \theta } ^ { ( j ) } )$   
14: end for   
15: return ${ \widehat { v } } ( S _ { 1 } ) , \ldots , { \widehat { v } } ( S _ { n } )$

Random projection. We now describe the implementation of the random projection step in our attribution procedure. We sample a random projection matrix $P \in \mathbb { R } ^ { p \times d }$ , where d is smaller than p and every entry of $P$ is independently drawn from a Gaussian distribution $\mathcal { N } ( 0 , d ^ { - 1 } )$ . We project the high-dimensional gradients to a d-dimensional space as

$$
\boldsymbol { \tilde { \varphi } } _ { i , t } = \boldsymbol { P } ^ { \top } \boldsymbol { \varphi } _ { i , t } .
$$

Then, we solve a logistic regression problem in the d-dimensional space with $\theta _ { d } \in \mathbb { R } ^ { d }$

$$
\widehat { \theta } _ { d } \gets \arg \operatorname* { m i n } _ { \theta _ { d } \in \mathbb { R } ^ { d } } \frac { 1 } { | \mathscr { D } _ { S _ { j } } | } \sum _ { ( \varphi , y , w ) \in \mathscr { D } _ { S _ { j } } } \ell ( \tilde { \varphi } , y , w ; \theta _ { d } ) ,\tag{6}
$$

for any $j = 1 , 2 , \dots , m$ . Finally, we map the solution from dimension d back to the original dimension $p$ as

$$
\widehat { \theta } ^ { ( j ) } = \theta ^ { \star } + P \widehat { \theta } _ { d } .
$$

Based on the Johnson-Lindenstrauss lemma, when $\begin{array} { r } { d = O \left( { \frac { \log p } { \epsilon ^ { 2 } } } \right) } \end{array}$ , for any $\varphi _ { i }$ with $\| \varphi _ { i } \| \leq G$ and any $\theta , \theta _ { 0 }$ with ℓ -norm at most $D .$ , we have

$$
| \langle g _ { i } , \theta - \theta _ { 0 } \rangle - \langle P g _ { i } , P ( \theta - \theta _ { 0 } ) \rangle | \leq \epsilon | \langle \varphi _ { i } , \theta - \theta _ { 0 } \rangle | \leq 2 G D \epsilon .
$$

This ensures that the random projection step will preserve the accuracy of the gradient estimation procedure. See Algorithm 3 for the complete description.

Correlation between attribution scores and LOO analysis. The goal of this experiment is to evaluate whether an attribution method can predict which trajectory subsets are more useful for policy improvement. For each task, we first collect a fixed pool of trajectories and sample multiple trajectory subsets from that pool. Each subset is used under the same update protocol to estimate its true downstream effect on the policy. The resulting subset performance can be measured by hypervolume.

Given these true subset effects, each attribution method assigns a predicted score to every subset. We then compare the ranking induced by the predicted scores with that induced by the true subset effects. The LDS score is the Spearman rank correlation between these two rankings. A higher LDS means the attribution method better identifies which trajectory subsets are actually more beneficial.

In Figure 3, we illustrate the correlation between the kernel surrogate attribution scores and leaveone-out analysis results. The results show that these two are strongly correlated across different tests.

![](images/56977e6245fad828c2778b165dcb19b02e831bb67744e1e64c446f892bd101bb.jpg)  
(a) Trajectory-level attribution.

![](images/295cc80ec3f6fdec9504290ec3d1e9742cffc051acbbdcc2f6f98980e904dbab.jpg)  
(b) Objective-level attribution.  
Figure 3: Correlation analysis of attribution scores against LOO scores (left) and objective conflict coefficient Γ (right).

Remark A.1. We describe several natural extensions ofour approach. First, our algorithms naturally extend to online multitask RL [55], where the state-action space varies across tasks. We leavefurther exploration of these benchmarks to future work.

Second, kernel surrogates can also be used to measure the influence ofan individual trajectory on the overall model behavior. This can be accomplished by sampling subsets oftrajectories instead of objectives in the above algorithm.

Lastly, while we primarilyfocus on probabilistic policies, similar ideas can be extended to deterministic policies.

Remark A.2. Our problemformulation relates to multi-objective optimization in prior literature [34, 6], and we extend this line ofwork to the online learning setting. Further exploring the connection between ensemble learning and online boosting [5] is another directionforfuture work.

Our work connects to the literature on influence estimation and mechanistic interpretability, which aims to trace model behavior back to influential training data. Prior work include influencefunctions [23], TracIn [35] and linear surrogate models [26]. In reinforcement learning, recent efforts have adapted these techniques to explain policy decisions, including trajectory-level attribution in offline RL [11] and snapshot influencefunctionsforfiltering harmful data in online RL [17]. One of the open questions in this area is to develop better theoretical characterizations of attribution methods. Our kernel-based objective attribution thus offers a principled approach and accurately estimates second-order interactions across preferences. This connection seems particularly relevant in light of recent advances in RL reasoning [4].

## B Complete Proofs

In this section, we present the proof of Theorem 3.6 in Appendix B.1. We introduce a lower bound in Appendix B.2 that matches the upper bound. Next, we analyze the magnitude of Γ in Algorithm B.3. Finally, in Appendix B.4, we present a relaxation when only an approximate low-rank basis assumption holds.

## B.1 Proof of Theorem 3.6

This section gives a complete proof of Theorem 3.6. The proof has four key steps. Lemma B.1 reduces cumulative regret to a sum of on-policy Bellman residuals. $\bar { d } _ { h }$ splits each residual into a conflict factor (the $\chi ^ { 2 }$ deviation of the deployed occupancy from ${ \bar { d } } _ { h } )$ and an in-sample factor (the squared Bellman residual measured under $d _ { h } )$ . Lemma B.2 uses $\Gamma ,$ and Lemma B.3 uses the schedule regularity in Assumption 3.5. Lemma B.4 converts the witness-side error into the confidence radius $\beta .$ Steps 3 and 4 give a $\sqrt { T }$ stable-phase rate, while the burn-in phase contributes $\mathcal { O } ( H t _ { 0 } )$ with $t _ { 0 } = \mathcal { O } ( C _ { \mathrm { c o v } } K \beta )$

Notations. We analyze the pooled-confidence version of the routed optimistic algorithm. At the beginning of episode $t ,$ each basis class $\mathcal { F } _ { k }$ has a confidence set $\mathcal { F } _ { k } ^ { ( t - 1 ) } ( \beta )$ constructed from the pooled history $\mathcal { D } ^ { ( t - 1 ) } = \{ ( \tau ^ { ( i ) } , w _ { i } ) : i < t \}$ , meaning that the regression residual is computed against every past trajectory, not only the trajectories routed to basis k. Routing determines which optimistic basis is executed; the confidence guarantee used below controls the Bellman residual of the selected function relative to the cumulative occupancy across all previous episodes. After observing $w _ { t }$ , the learner chooses

$$
( k _ { t } , f ^ { ( t ) } ) \in \arg \operatorname* { m a x } _ { k \in [ K ] , f \in \mathcal { F } _ { k } ^ { ( t - 1 ) } ( \beta ) } V _ { 1 } ^ { f } ( s _ { 0 } ; w _ { t } ) , V _ { h } ^ { f } ( s ; w ) : = \operatorname* { m a x } _ { a } f _ { h } ( s , a ; w ) .\tag{7}
$$

The deployed policy $\pi ^ { ( t ) }$ is greedy with respect to $f ^ { ( t ) }$ under $w _ { t }$ , with state-action occupancy $d _ { h } ^ { ( t ) } : = \overline { { d _ { h } ^ { \pi } } } ^ { \bar { ( t ) } }$ . Write the scalarized Bellman residual of the selected optimistic function as

$$
\delta _ { h } ^ { ( t ) } ( s , a ) = f _ { h } ^ { ( t ) } ( s , a ; w _ { t } ) - ( \mathcal { T } _ { h } ^ { w _ { t } } f _ { h + 1 } ^ { ( t ) } ) ( s , a ) ,
$$

with the convention $f _ { H + 1 } ^ { ( t ) } \equiv 0$ . Finally, define the basis-mixture witness $\begin{array} { r } { \bar { d } _ { h } : = K ^ { - 1 } \sum _ { k = 1 } ^ { K } d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { \star } } } \end{array}$ and the un-normalized cumulative empirical occupancy $\begin{array} { r } { \bar { d } _ { h } ^ { ( t ) } : = \sum _ { i = 1 } ^ { t - 1 } d _ { h } ^ { ( i ) } } \end{array}$

Lastly, recall that the $\chi ^ { 2 } .$ -divergence between two distributions $P ( x )$ and $Q ( x )$ is defined as

$$
\chi ^ { 2 } ( P \| Q ) = \sum _ { x : Q ( x ) > 0 } \frac { ( P ( x ) - Q ( x ) ) ^ { 2 } } { Q ( x ) } .
$$

Proof structure. The proof relies on two events holding simultaneously, each with probability at least $1 - \delta / 2 \ : ( \mathrm { s o }$ that their intersection holds with probability at least $1 - \delta )$ . For $\bar { E } _ { \mathrm { o p t } }$ , every realizable $Q ^ { \pi _ { \mathfrak { b } k } ^ { \star } } ( \cdot ; w )$ remains in its confidence set $\mathcal { F } _ { k } ^ { ( t - 1 ) } ( \beta )$ for all relevant $t , k , w .$ This is the standard optimism event for confidence-set value iteration, union-bounded over the K basis classes. As for $E _ { \mathrm { r e g } } ,$ , for every selected $f ^ { ( t ) }$ , the cumulative squared Bellman residual on past trajectories is bounded by β. This is the standard in-sample regression event, again, union-bounded over the K basis classes. Next, we work on the intersection of $E _ { \mathrm { o p t } }$ and $E _ { \mathrm { r e g } }$

Lemma B.1. On the event $E _ { \mathrm { o p t } } , f o r$ every episode t,

$$
{ V _ { 1 } ^ { f } } ^ { ( t ) } ( s _ { 0 } ; w _ { t } ) \geq \mathcal { I } ( \pi _ { w _ { t } } ^ { \star } ; w _ { t } ) .
$$

Proof. Assumption 3.4 expresses the occupancy of $\pi _ { w } ^ { \star }$ as a convex combination of the basis occupancies. Since the scalarized value is linear in the occupancy measure,

$$
\mathcal { I } ( \pi _ { w } ^ { \star } ; w ) = \sum _ { k } \nu _ { w } ( k ) \mathcal { I } ( \pi _ { \mathfrak { b } k } ^ { \star } ; w ) .
$$

By Lemma 3.8, there exists $k ^ { \star } ( w _ { t } ) \in [ K ]$ such that the optimal MORL policy and the optimal basis policy at index $k ^ { \star } ( w _ { t } )$ have the same scalarized return: $\bar { \mathcal { I } } ( \pi _ { w _ { t } } ^ { \star } ; w _ { t } ) = \bar { \mathcal { I } } ( \pi _ { \mathsf { b } k ^ { \star } ( w _ { t } ) } ^ { \star } ; w _ { t } )$

By event $E _ { \mathrm { o p t } }$ , the corresponding true Q-function $Q ^ { \pi _ { \mathsf { b } k } ^ { \star } \star } ( w _ { t } ) \left( \cdot ; w _ { t } \right)$ lies in $\mathcal { F } _ { k ^ { \star } ( w _ { t } ) } ^ { ( t - 1 ) } ( \beta )$ . Restricting the joint argmax in (7) to $k = k ^ { \star } ( w _ { t } )$ then yields

$$
\begin{array} { r l } { V _ { 1 } ^ { f ^ { ( t ) } } \big ( s _ { 0 } ; w _ { t } \big ) \ : \geq \ : \displaystyle \operatorname* { m a x } _ { f \in \mathcal { F } _ { k ^ { \star } ( w _ { t } ) } ^ { ( t - 1 ) } ( \beta ) } V _ { 1 } ^ { f } \big ( s _ { 0 } ; w _ { t } \big ) \ : \geq \ : V _ { 1 } ^ { Q ^ { \pi _ { \mathtt { b } k ^ { \star } ( w _ { t } ) } ^ { \star } } \big ( s _ { 0 } ; w _ { t } \big ) } } & { } \\ { \quad = \ : \mathcal { I } \big ( \pi _ { \mathtt { b } k ^ { \star } ( w _ { t } ) } ^ { \star } ; w _ { t } \big ) \ : = \ : \mathcal { I } \big ( \pi _ { w _ { t } } ^ { \star } ; w _ { t } \big ) . } \end{array}
$$

Lemma B.1 allows us to upper bound regret by the gap between the optimistic value and the value actually obtained by the greedy policy. The usual telescoping identity for finite-horizon Bellman residuals gives

$$
\begin{array} { r l } & { { \mathsf { R e g } } ( T ) \leq \displaystyle \sum _ { t = 1 } ^ { T } \left[ V _ { 1 } ^ { f ^ { ( t ) } } ( s _ { 0 } ; w _ { t } ) - \mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \right] } \\ & { \quad \quad \quad = \displaystyle \sum _ { t = 1 } ^ { T } \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { ( t ) } } \left[ \delta _ { h } ^ { ( t ) } ( s , a ) \right] . } \end{array}\tag{8}
$$

The last equality follows from the standard layerwise telescoping along the trajectory of $\pi ^ { ( t ) }$ . Because $\pi ^ { ( t ) }$ is greedy with respect to $f ^ { ( t ) }$ , at each step $h , V _ { h } ^ { f ^ { ( t ) } } ( s _ { h } ; w _ { t } ) = f _ { h } ^ { ( t ) } ( s _ { h } , a _ { h } ; w _ { t } )$ . Subtracting the scalarized reward plus the next optimistic value at each layer telescopes from $V _ { 1 } ^ { f ^ { ( t ) } } ( s _ { 0 } ; w _ { t } )$ down to the realized scalarized return, leaving exactly the expected sum of $\delta _ { h } ^ { ( t ) }$

Let us analyze the term $\mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { ( t ) } } \left[ \delta _ { h } ^ { ( t ) } ( s , a ) \right]$ in the regret upper bound in equation (8). For each t and $h ,$ multiply and divide by $\sqrt { \bar { d } _ { h } ( s , a ) }$ inside the inner product:

$$
\mathbb { E } _ { d _ { h } ^ { ( t ) } } [ \delta _ { h } ^ { ( t ) } ] = \sum _ { s , a } \frac { d _ { h } ^ { ( t ) } ( s , a ) } { \sqrt { \bar { d } _ { h } ( s , a ) } } \cdot \sqrt { \bar { d } _ { h } ( s , a ) } \delta _ { h } ^ { ( t ) } ( s , a ) \le \sqrt { \phi _ { h } ( d _ { h } ^ { ( t ) } ) } \cdot \sqrt { \mathbb { E } _ { \bar { d } _ { h } } \left[ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } \right] } .\tag{9}
$$

Notably the conflicting factor $\phi _ { h } ( d )$ is connected with the $\chi ^ { 2 } \cdot$ -divergence:

$$
\phi _ { h } ( d ) : = \sum _ { s , a } \frac { d ( s , a ) ^ { 2 } } { \bar { d } _ { h } ( s , a ) } = 1 + \chi ^ { 2 } ( d \| \bar { d } _ { h } ) .
$$

The two factors in equation (9) are controlled separately: the conflict factor by Assumption 3.5 and the in-sample factor by combining the witness-comparison bound with the pooled-regression guarantee.

Lemma B.2. Under Assumption 3.5, for every t and $h ,$

$$
\phi _ { h } ( d _ { h } ^ { ( t ) } ) : = \sum _ { s , a } \frac { d _ { h } ^ { ( t ) } ( s , a ) ^ { 2 } } { \bar { d } _ { h } ( s , a ) } = 1 + \chi ^ { 2 } ( d _ { h } ^ { ( t ) } \| \bar { d } _ { h } ) \leq 1 + \Gamma .
$$

Proof. By Assumption 3.5, $\begin{array} { r } { d _ { h } ^ { ( t ) } = \sum _ { k } \nu ^ { ( t ) } ( k ) d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { \star } } } \end{array}$ . The map

$$
d \mapsto \phi _ { h } ( d ) = \sum _ { s , a } \frac { d ( s , a ) ^ { 2 } } { \bar { d } _ { h } ( s , a ) }
$$

is a squared norm in the measure weighted by $\bar { d } _ { h } ^ { - 1 }$ , hence convex. Jensen’s inequality gives

$$
\phi _ { h } ( d _ { h } ^ { ( t ) } ) \leq \sum _ { k } \nu ^ { ( t ) } ( k ) \phi _ { h } ( d _ { h } ^ { \pi _ { \mathsf { b } k } ^ { \star } } ) \leq 1 + \Gamma ,
$$

where the last step is Definition 3.3.

Lemma B.3. Under Assumption 3.5 (with constants $c _ { 0 } \in ( 0 , 1 ]$ and burn-in time $t _ { 0 } \in \mathbb { N } )$ , for every $t \geq t _ { 0 } , h \in [ H ]$ , and $( s , a )$

$$
\bar { d } _ { h } ^ { ( t ) } ( s , a ) \geq \frac { c _ { 0 } ( t - 1 ) } { C _ { \mathrm { c o v } } } \bar { d } _ { h } ( s , a ) .
$$

Proof. The cumulative empirical occupancy can be expanded in terms of the basis occupancies:

$$
\frac { \bar { d } _ { h } ^ { ( t ) } ( s , a ) } { t - 1 } = \sum _ { k = 1 } ^ { K } \bar { \nu } ^ { ( t ) } ( k ) d _ { h } ^ { \pi _ { b k } ^ { \star } } ( s , a ) \geq \frac { c _ { 0 } } { C _ { \mathrm { c o v } } K } \sum _ { k = 1 } ^ { K } d _ { h } ^ { \pi _ { b k } ^ { \star } } ( s , a ) = \frac { c _ { 0 } } { C _ { \mathrm { c o v } } } \bar { d } _ { h } ( s , a ) .
$$

The inequality is exactly the stable-phase lower bound in Assumption 3.5. Thus after burn-in, the data distribution accumulated by the algorithm pointwise dominates the uniform basis mixture up to the factor $C _ { \mathrm { c o v } } / c _ { 0 }$

Lemma B.4. On event $E _ { \mathrm { r e g } } , f o { }$ r every $t \geq t _ { 0 }$ and $h \in [ H ]$

$$
\mathbb { E } _ { \bar { d } _ { h } } \bigg [ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } \bigg ] \leq \frac { C _ { \mathrm { c o v } } \beta } { c _ { 0 } ( t - 1 ) } .
$$

Proof. The standard regression guarantee for optimistic confidence sets, union bounded over $k \in [ K ]$ gives for the selected $\bar { f } ^ { ( t ) } \left( \mathrm { i . e . } \right.$ , on the event $E _ { \mathrm { r e g } } )$

$$
\sum _ { i = 1 } ^ { t - 1 } \mathbb { E } _ { d _ { h } ^ { ( i ) } } \left[ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } \right] = \mathbb { E } _ { \bar { d } _ { h } ^ { ( t ) } } \left[ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } \right] \leq \beta .\tag{10}
$$

This is exactly where the proof uses pooled confidence updates: the measure on the left is the cumulative occupancy across all previous episodes, not just those routed to $k _ { t }$ . Lemma B.3 gives

$$
\bar { d } _ { h } ( s , a ) \leq \frac { C _ { \mathrm { c o v } } } { c _ { 0 } ( t - 1 ) } \bar { d } _ { h } ^ { ( t ) } ( s , a ) , \mathrm { f o r } t \geq t _ { 0 } .
$$

Multiplying this pointwise inequality by $( \delta _ { h } ^ { ( t ) } ( s , a ) ) ^ { 2 }$ and summing over $( s , a )$ , then applying (10), yields

$$
\mathbb { E } _ { \bar { d } _ { h } } \left[ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } \right] \leq \frac { C _ { \mathrm { c o v } } } { c _ { 0 } ( t - 1 ) } \mathbb { E } _ { \bar { d } _ { h } ^ { ( t ) } } \left[ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } \right] \leq \frac { C _ { \mathrm { c o v } } \beta } { c _ { 0 } ( t - 1 ) } .
$$

Proof of Theorem 3.6. Combining equation (9) with Lemmas B.2 and B.4 gives, for every $t \geq t _ { 0 }$ and $\dot { h } \in [ H ]$

$$
\mathbb { E } _ { d _ { h } ^ { ( t ) } } [ \delta _ { h } ^ { ( t ) } ] \leq \sqrt { 1 + \Gamma } \cdot \sqrt { \frac { C _ { \mathrm { c o v } } \beta } { c _ { 0 } ( t - 1 ) } } = \sqrt { \frac { C _ { \mathrm { c o v } } \left( 1 + \Gamma \right) \beta } { c _ { 0 } ( t - 1 ) } } .\tag{11}
$$

We split the regret (8) at the burn-in time $t _ { 0 } \mathrm { i }$

$$
\mathsf { R e g } ( T ) \ \leq \ \underbrace { \sum _ { t = 1 } ^ { t _ { 0 } - 1 } \sum _ { h = 1 } ^ { H } \mathbb { E } _ { d _ { h } ^ { ( t ) } } [ \delta _ { h } ^ { ( t ) } ] } _ { \mathrm { ( I ) } } + \underbrace { \sum _ { t = t _ { 0 } } ^ { T } \sum _ { h = 1 } ^ { H } \mathbb { E } _ { d _ { h } ^ { ( t ) } } [ \delta _ { h } ^ { ( t ) } ] } _ { \mathrm { ( I I ) } } .
$$

During burn-in, for each $\begin{array} { r } { t < t _ { 0 } , \sum _ { h } \mathbb { E } _ { d _ { \star } ^ { ( t ) } } [ \delta _ { h } ^ { ( t ) } ] = V _ { 1 } ^ { f ^ { ( t ) } } ( s _ { 0 } ; w _ { t } ) - \mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \le H } \end{array}$ since both $V _ { 1 } ^ { f ^ { ( t ) } } \in [ 0 , H ]$ and $\mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \in [ 0 , H ]$ . Hence $( \mathrm { I } ) \le H t _ { 0 }$

During the stable phase, substituting (11) and summing over h and t, II is at most

$$
H \sum _ { t = t _ { 0 } } ^ { T } \sqrt { \frac { C _ { \mathrm { c o v } } \left( 1 + \Gamma \right) \beta } { c _ { 0 } \left( t - 1 \right) } } = H \sqrt { \frac { C _ { \mathrm { c o v } } \left( 1 + \Gamma \right) \beta } { c _ { 0 } } } \sum _ { t = t _ { 0 } } ^ { T } \frac { 1 } { \sqrt { t - 1 } } \leq 2 H \sqrt { \frac { C _ { \mathrm { c o v } } \left( 1 + \Gamma \right) \beta T } { c _ { 0 } } } ,
$$

using $\begin{array} { r } { \sum _ { t = t _ { 0 } } ^ { T } ( t - 1 ) ^ { - 1 / 2 } \leq \sum _ { u = 1 } ^ { T - 1 } u ^ { - 1 / 2 } \leq 2 \sqrt { T } . } \end{array}$

Adding (I) and (II) and absorbing the universal constant $1 / c _ { 0 }$ into $\mathcal { O } ( \cdot )$

$$
\mathsf { R e g } ( T ) \leq H t _ { 0 } + 2 H \sqrt { \frac { C _ { \mathrm { c o v } } ( 1 + \Gamma ) \beta T } { c _ { 0 } } } = \mathcal { O } \Big ( H \sqrt { C _ { \mathrm { c o v } } ( 1 + \Gamma ) \beta T } \Big ) + \mathcal { O } ( H t _ { 0 } ) .
$$

The preference process satisfies the schedule regularity of Assumption 3.5 after $t _ { 0 } = \mathcal { O } ( C _ { \mathrm { c o v } } K \beta )$ episodes (a standard Hoeffding’s bound on the visitation counts of the K basis components, with the log factor absorbed into β). Substituting yields

$$
\mathsf { R e g } ( T ) \leq \mathcal { O } \Bigl ( H \sqrt { C _ { \mathrm { c o v } } \left( 1 + \Gamma \right) \beta T } + H C _ { \mathrm { c o v } } K \beta \Bigr ) ,
$$

which is equation (2). The corresponding worst-case scaling follows by applying $1 + \Gamma \le K$ from Lemma B.5.

For completeness, we record the structural lemma used above; it shows that the conflict coefficient is at most $K - 1$ for any basis, with equality (up to lower-order terms) in the disjoint regime that powers the lower bound.

Lemma B.5. For any collection ofK basis occupancies $\{ d _ { h } ^ { \pi _ { \flat k } ^ { \star } } \} _ { k \in [ K ] }$ and every $h \in [ H ]$

$$
\Gamma = \underset { h \in [ H ] } { \operatorname* { m a x } } \operatorname* { m a x } _ { k \in [ K ] } \chi ^ { 2 } \Bigl ( d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { \star } } \bigl \| \bar { d } _ { h } \Bigr ) \le K - 1 .
$$

Equality holds (up to lower-order terms) when the basis occupancies have pairwise disjoint supports.

Proof. Fix h and k. Pointwise, $\begin{array} { r } { \bar { d } _ { h } ( s , a ) \geq \frac { 1 } { K } d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { \star } } ( s , a ) } \end{array}$ on the support of $d _ { h } ^ { \pi _ { \flat k } ^ { \star } }$ . Hence

$$
\chi ^ { 2 } \Big ( d _ { h } ^ { \pi _ { b k } ^ { \star } } \big \| \bar { d } _ { h } \Big ) = \sum _ { s , a } \frac { ( d _ { h } ^ { \pi _ { b k } ^ { \star } } ( s , a ) ) ^ { 2 } } { \bar { d } _ { h } ( s , a ) } - 1 \leq \sum _ { s , a } \frac { ( d _ { h } ^ { \pi _ { b k } ^ { \star } } ( s , a ) ) ^ { 2 } } { ( 1 / K ) d _ { h } ^ { \pi _ { b k } ^ { \star } } ( s , a ) } - 1 = K - 1 .
$$

The disjoint-support case attains equality because then $\begin{array} { r } { \bar { d } _ { h } ( s , a ) = \frac { 1 } { K } d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { \star } } ( s , a ) } \end{array}$ on the support.

## B.2 Proof of the regret lower bound

Theorem B.6. There exists afamily ofMOMDPs with K basis policies, $C _ { \mathrm { c o v } } \leq K$ , and $\Gamma = K - 1$ such thatfor any (possibly randomized) learning algorithm A and any $T \geq K$ , a preference sequence that cycles uniformly through $\{ e _ { 1 } , \ldots , e _ { K } \}$ satisfies

$$
\begin{array} { r } { \mathbb { E } \big [ \mathsf { R e g } ( T ) \big ] \geq \Omega \Big ( H \sqrt { K T } \Big ) = \Omega \Big ( H \sqrt { ( 1 + \Gamma ) T } \Big ) . } \end{array}
$$

Together, Theorems 3.6 and B.6 pin down the dependence on K and Γ up to factors of $\sqrt { C _ { \mathrm { c o v } } \log | \mathcal { F } | }$

In the following proof, we prove the lower bound by embedding a standard Bernoulli two-armed bandit problem into a simple subclass of MOMDPs. This is sufficient because a lower bound for one valid MOMDP family is automatically a lower bound for the larger MOMDP class. The constructed family has K pairwise-disjoint sub-chains, one per objective, each containing one Bernoulli twoarmed bandit. Here, a sub-chain means a branch of the environment state space, not a policy: the k-th optimal basis policy is the policy that routes into sub-chain $\boldsymbol { S _ { k } }$ and then plays the better binary action.

Under a cyclic preference sequence, regret has two sources: 1) a routing term for episodes whose first action does not match the active preference; 2) K independent bandit terms for wrong binary choices inside correctly routed sub-chains.

We then show that for each preference $e _ { k }$ , the learner faces a two-armed bandit on the episodes with active preference $e _ { k } .$ , with mis-routing acting as a no-information mistake that also costs order $\Delta$ The horizon does not create a harder statistical problem in this construction: after the binary decision, the remaining states reveal no additional choices or independent samples. Instead, the same Bernoulli outcome is paid over the remaining $H - 1$ stages, so the bandit regret is simply multiplied by $H - 1$ Tuning the bandit gap to $\Delta = { \textstyle { \frac { 1 } { 8 } } } \sqrt { K / T }$ makes the total unavoidable cost scale as $\Delta T = \Theta ( \sqrt { K T } )$ , and the repeated payoff yields the claimed $\Omega ( H \sqrt { K T } )$ lower bound.

Hard MOMDP instances. Fix $K \ge 2 , H \ge 2$ , and $\Delta \in ( 0 , 1 / 4 ]$ . The MOMDP has $m = K$ objectives and a single start state $s _ { 0 }$ . From $s _ { 0 } ,$ the K deterministic actions $a \in [ K ]$ route the agent to disjoint state-space branches $\mathcal { S } _ { 1 } , \ldots , \mathcal { S } _ { K }$ , which we call sub-chains. Inside sub-chain $S _ { k }$ , the agent makes one binary decision $u \in \{ 0 , 1 \}$ at the second step; thereafter, the chain is uninformative. There is an unknown bit $b _ { k } \in \{ 0 , 1 \}$ , and the k-th reward coordinate at every step inside $\boldsymbol { S } _ { k }$ equals a single Bernoulli outcome drawn at the moment of the binary decision, with mean

$$
\begin{array} { r } { \frac { 1 } { 2 } + \Delta \quad \mathrm { i f } u = b _ { k } , \frac { 1 } { 2 } \quad \mathrm { i f } u \neq b _ { k } . } \end{array}
$$

All other reward coordinates are zero. Thus, each episode yields only one statistically informative bandit observation, even though it has a horizon of H. A correctly routed episode where the wrong binary action is chosen contributes regret $( H - 1 ) \Delta$ , and an incorrectly routed episode contributes regret at least $( H - 1 ) \Delta$ as well.

Preference sequences. The preference sequence cycles through $e _ { 1 } , e _ { 2 } , . . . , e _ { K } , e _ { 1 } , e _ { 2 } , . . . ,$ , so each preference $e _ { k }$ appears between $\lfloor T / K \rfloor$ and $\lceil T / K \rceil$ times. The optimal basis policy $\pi _ { \mathsf { b } k } ^ { \star }$ routes to sub-chain $S _ { k }$ and plays $b _ { k } ;$ the K basis occupancies are supported on disjoint sub-chains, so by Lemma $8 . 5 , \Gamma = K - 1$ , and the basis-mixture witness $\bar { d } _ { h }$ gives $C _ { \mathrm { c o v } } \leq K$ . We use a lower bound result from the work of Lattimore and Szepesvári [25].

Lemma B.7 (Bandit lower bound [25]). For any randomized algorithm interacting with a Bernoulli two-armed bandit of gap $\Delta \in ( 0 , 1 / 4 ]$ , the expected cumulative regret after n pulls is at least ${ \mathit { c } } _ { \mathrm { L B } }$ min(∆n, 1/∆)for a universal constant $c _ { \mathrm { L B } } > 0$ , in the worst case over the unknown optimal arm.

ProofofTheorem B.6. We proceed in three steps. First, for a fixed interaction history, partition the $T$ episodes into two disjoint sets: M episodes where the learner mis-routes, meaning that its first action does not equal the index $k _ { t }$ of the active preference $e _ { k _ { t } }$ . Each such episode loses at least $( H - 1 ) \Delta$ relative to the optimal policy for $e _ { k _ { t } }$ . For each $k \in [ K ] , n _ { k }$ episodes where the learner correctly routes to $\scriptstyle { S _ { k } }$ under preference $e _ { k }$ . Conditional on these episodes, the learner observes exactly $n _ { k }$ pulls from the two-armed Bernoulli bandit associated with unknown bit $b _ { k }$

Let $M _ { k }$ be the number of mis-routed episodes whose active preference is $e _ { k } .$ , and let $m _ { k } : = M _ { k } + n _ { k }$ be the total number of appearances of preference $e _ { k }$ . The cyclic sequence fixes $m _ { k }$ deterministically, with $\lfloor T / K \rfloor \leq m _ { k } \leq \lceil T / K \rceil$ , and $\begin{array} { r } { M + \sum _ { k = 1 } ^ { K } n _ { k } = \sum _ { k = 1 } ^ { K } m _ { k } = T } \end{array}$ pathwise. Taking expectation over the learner’s randomness and the bandit rewards, the cumulative regret satisfies

$$
\mathbb { E } [ \mathsf { R e g } ( T ) ] \geq ( H - 1 ) \Delta \mathbb { E } [ M ] + ( H - 1 ) \sum _ { k = 1 } ^ { K } \mathbb { E } [ \mathsf { R e g } _ { k } ^ { \mathrm { b a n d i t } } ( n _ { k } ) ] ,\tag{12}
$$

where $\mathsf { R e g } _ { k } ^ { \mathrm { b a n d i t } } ( n _ { k } )$ denotes the cumulative regret of the induced bandit on sub-chain $\scriptstyle { S _ { k } }$ (the leading $H - 1$ factor accounts for the fact that each Bernoulli outcome is paid out over $H - 1$ steps).

Next, set $\Delta = \textstyle { \frac { 1 } { \mathrm { s } } } { \sqrt { K / T } }$ , which satisfies $\Delta \le 1 / 8$ since $T \geq K$ . The cyclic preference sequence ensures $m _ { k } \le \left\lceil \stackrel { \smile } { T } / K \right\rceil \le 2 T / K$ , so

$$
\begin{array} { r } { \Delta m _ { k } \leq \frac { 1 } { 8 } \sqrt { K / T } \cdot \frac { 2 T } { K } = \frac { 1 } { 4 } \sqrt { T / K } \leq \frac { 1 } { \Delta } . } \end{array}
$$

For a fixed $k ,$ restrict attention to the $m _ { k }$ episodes whose active preference is $e _ { k }$ . In these episodes, the learner must either route to $\boldsymbol { S } _ { k }$ and choose one of the two bandit arms, or mis-route and incur one-stage scalarized regret at least $\Delta$ while receiving no observation from the k-th bandit. Thus, the sum $\Delta M _ { k } + { \sf R e g } _ { k } ^ { \mathrm { b a n d i t } } ( n _ { k } )$ is lower bounded by the regret of a Bernoulli two-armed bandit problem with $m _ { k }$ opportunities. Lemma ${ \bf B . 7 }$ gives

$$
\begin{array} { r } { \mathbb { E } \Big [ \Delta M _ { k } + \mathsf { R e g } _ { k } ^ { \mathrm { b a n d i t } } ( n _ { k } ) \Big ] \geq c _ { \mathrm { L B } } \operatorname* { m i n } ( \Delta m _ { k } , 1 / \Delta ) = c _ { \mathrm { L B } } \Delta m _ { k } . } \end{array}
$$

Summing over k yields

$$
\Delta \mathbb { E } [ M ] + \sum _ { k = 1 } ^ { K } \mathbb { E } [ \mathsf { R e g } _ { k } ^ { \mathrm { b a n d i t } } ( n _ { k } ) ] \ge c _ { \mathrm { L B } } \Delta \sum _ { k = 1 } ^ { K } m _ { k } = c _ { \mathrm { L B } } \Delta T .
$$

Lastly, substituting the preceding bound into equation (12),

$$
\begin{array} { r } { \mathbb { E } [ \mathsf { R e g } ( T ) ] \geq c _ { \mathrm { L B } } ( H - 1 ) \Delta T , } \end{array}
$$

Substituting $\begin{array} { r } { \Delta = \frac { 1 } { 8 } \sqrt { K / T } } \end{array}$ , we have

$$
\mathbb { E } [ \mathsf { R e g } ( T ) ] \geq \frac { c _ { \mathrm { L B } } } { 8 } ( H - 1 ) \sqrt { K T } = \Omega \Big ( H \sqrt { K T } \Big ) .
$$

Since the construction has $\Gamma = K - 1$ , this is equivalent to $\Omega ( H \sqrt { ( 1 + \Gamma ) T } )$ . Thus, we have completed the proof of the lower bound.

## B.3 Bounds on objective conflict coefficients

Although Algorithm 1 does not require Γ as an input, its off-policy transfer step is governed by the same overlap quantity. At iteration $t ,$ sampling a basis policy uniformly yields the routed-ensemble mixture and its associated conflict measure

$$
\bar { d } _ { h , t } ^ { \mathrm { R E } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } d _ { h } ^ { \pi _ { j } ^ { ( t - 1 ) } } , \Gamma _ { \mathrm { R E } } ^ { ( t ) } = \operatorname* { m a x } _ { h \in [ H ] } \operatorname* { m a x } _ { j \in [ K ] } \chi ^ { 2 } \left( d _ { h } ^ { \pi _ { j } ^ { ( t - 1 ) } } \Vert \bar { d } _ { h , t } ^ { \mathrm { R E } } \right) .\tag{13}
$$

When the learned ensemble recovers the latent basis policies, $\Gamma _ { \mathrm { R E } } ^ { ( t ) }$ coincides with Γ. The following proposition formalizes that the off-policy estimators used to route and update the ensemble have variance inflated by exactly the factor $1 + \Gamma _ { \mathrm { R E } } ^ { ( t ) }$

Proposition B.8. Fix an iteration t ofAlgorithm $^ { l , }$ a stage $h ,$ and a target basis index $j \in [ K ]$ Suppose B replay samples $( s _ { i } , a _ { i } )$ are drawn i.i.d. from $\bar { d } _ { h , t } ^ { \mathrm { R E } }$ , and let $g _ { h } : \check { S \times A }  [ - G , \check { G } ]$ be any bounded measurable signal with target value

$$
\theta _ { j , h } = \mathbb { E } _ { d _ { h } ^ { \pi _ { j } ^ { ( t - 1 ) } } } [ g _ { h } ( s , a ) ] .
$$

The importance-weighted off-policy estimator

$$
\widehat { \theta } _ { j , h } : = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { d _ { h } ^ { \pi _ { j } ^ { ( t - 1 ) } } \left( s _ { i } , a _ { i } \right) } { \bar { d } _ { h , t } ^ { \mathrm { R E } } \left( s _ { i } , a _ { i } \right) } g _ { h } ( s _ { i } , a _ { i } )
$$

is unbiased and satisfies

$$
\mathbb { E } \Big [ ( \widehat { \theta } _ { j , h } - \theta _ { j , h } ) ^ { 2 } \Big ] \leq \frac { G ^ { 2 } \big ( 1 + \Gamma _ { \mathrm { R E } } ^ { ( t ) } \big ) } { B } .\tag{14}
$$

In the idealized case $\pi _ { j } ^ { ( t - 1 ) } = \pi _ { \mathsf { b } j } ^ { \star }$ for all $j , \Gamma _ { \mathrm { R E } } ^ { ( t ) } = \Gamma$

Proof. Fix iteration $t ,$ stage $h ,$ and target basis index $j .$ . For brevity, write $d _ { j } : = d _ { h } ^ { \pi _ { j } ^ { ( t - 1 ) } } , \mu : = \bar { d } _ { h , t } ^ { \mathrm { R E } }$ and $\rho _ { j } ( s , a ) : = d _ { j } ( s , a ) / \mu ( s , a )$ . Finite $\Gamma _ { \mathrm { R E } } ^ { ( t ) }$ implies $d _ { j } \ll \mu ,$ so $\rho _ { j }$ is well defined on supp(µ). Since $d _ { j }$ and $\mu$ are probability distributions over $\boldsymbol { s } \times \boldsymbol { A }$ at stage h, $\begin{array} { r } { \sum _ { s , a } d _ { j } = \sum _ { s , a } \mu = 1 } \end{array}$

Each replay sample $( s _ { i } , a _ { i } )$ is drawn from $\mu ,$ so

$$
\mathbb { E } [ \rho _ { j } ( s _ { i } , a _ { i } ) g _ { h } ( s _ { i } , a _ { i } ) ] = \sum _ { s , a } \mu ( s , a ) \frac { d _ { j } ( s , a ) } { \mu ( s , a ) } g _ { h } ( s , a ) = \sum _ { s , a } d _ { j } ( s , a ) g _ { h } ( s , a ) = \theta _ { j , h } .
$$

Averaging over B samples preserves the mean: $\mathbb { E } [ \widehat { \theta } _ { j , h } ] = \theta _ { j , h }$

For a single sample, the second moment of the importance-weighted signal is bounded by

$$
\mathbb { E } _ { \mu } \left[ \rho _ { j } ( s , a ) ^ { 2 } g _ { h } ( s , a ) ^ { 2 } \right] \leq G ^ { 2 } \sum _ { s , a } \frac { d _ { j } ( s , a ) ^ { 2 } } { \mu ( s , a ) } = G ^ { 2 } \big ( 1 + \chi ^ { 2 } ( d _ { j } \| \mu ) \big ) \leq G ^ { 2 } \big ( 1 + \Gamma _ { \mathrm { R E } } ^ { ( t ) } \big ) ,\tag{15}
$$

where the equality uses the standard χ<sup>2</sup>-identity $\textstyle \sum _ { s , a } d _ { j } ^ { 2 } / \mu = 1 + \chi ^ { 2 } ( d _ { j } \| \mu )$ (which holds because $d _ { j }$ and $\mu$ are probability distributions), and the last inequality is the definition of $\Gamma _ { \mathrm { R E } } ^ { ( t ) }$ . The B samples are independent, so

$$
\mathbb { E } \Big [ ( \widehat { \theta } _ { j , h } - \theta _ { j , h } ) ^ { 2 } \Big ] = \mathrm { V a r } ( \widehat { \theta } _ { j , h } ) \leq \frac { \mathbb { E } _ { \mu } \big [ \rho _ { j } ^ { 2 } g _ { h } ^ { 2 } \big ] } { B } \leq \frac { G ^ { 2 } \big ( 1 + \Gamma _ { \mathrm { R E } } ^ { ( t ) } \big ) } { B } ,
$$

which leads to equation (14). $\operatorname { I f } \pi _ { j } ^ { ( t - 1 ) } = \pi _ { \mathsf { b } j } ^ { \star }$ for every $j \in [ K ]$ , then $\mu = \bar { d } _ { h }$ and $\Gamma _ { \mathrm { R E } } ^ { ( t ) } = \Gamma$ by Definition 3.3.

## B.4 Approximate low-rank conditions

Next, we show how the regret bound changes when Assumption 3.4 only holds approximately. We separate two effects. Relative to the best policy in the chosen K-policy basis, the proof of Theorem 3.6 still gives a sublinear regret bound, with the conflict factor inflated by ε. If the chosen basis does not contain a truly optimal policy for a preference, then true regret also includes the gap between the true optimum and the best basis policy.

Let $\begin{array} { r } { \bar { d } _ { h } = K ^ { - 1 } \sum _ { k = 1 } ^ { K } d _ { h } ^ { \pi _ { \mathrm { b } k } ^ { \star } } } \end{array}$ be the same basis-mixture witness used in the proof of Theorem 3.6. We replace Assumption 3.4 by the following approximate version.

Assumption B.9. There exist latent basis policies $\{ \pi _ { \mathsf { b } 1 } ^ { \star } , \hdots , \pi _ { \mathsf { b } K } ^ { \star } \}$ and a tolerance $\varepsilon \geq 0$ such that every deployed policy $\pi \in \{ \pi ^ { ( t ) } : t \leq T \}$ admits a single mixing vector $\nu _ { \pi } \in \Delta ^ { K - 1 }$ satisfying, for every stage $h \in [ H ]$

$$
d _ { h } ^ { \pi } = q _ { \pi , h } + e _ { \pi , h } , q _ { \pi , h } : = \sum _ { k = 1 } ^ { K } \nu _ { \pi } ( k ) d _ { h } ^ { \pi _ { b k } ^ { \star } } , \| e _ { \pi , h } \| _ { \bar { d } _ { h } ^ { - 1 } } ^ { 2 } : = \sum _ { s , a } \frac { e _ { \pi , h } ( s , a ) ^ { 2 } } { \bar { d } _ { h } ( s , a ) } \leq \varepsilon ^ { 2 } .
$$

The norm is interpreted with the usual support convention: finite norm requires $e _ { \pi , h } ( s , a ) = 0$ whenever $\bar { d } _ { h } ( s , a ) = 0$

The $\bar { d } _ { h } ^ { - 1 }$ -weighted norm is chosen to match the Cauchy–Schwarz step in Theorem 3.6: the proof controls $\begin{array} { r } { \phi _ { h } ( d ) = \sum _ { s , a } d ( s , a ) ^ { 2 } / \bar { d } _ { h } ( s , a ) = \| d \| _ { \bar { d } _ { \star } ^ { - 1 } } ^ { 2 } } \end{array}$ . Thus, an error bound in this same norm can be carried through the original argument by the triangle inequality.

For this approximate statement, we keep the stable-coverage consequence of Assumption 3.5: for all $t \geq t _ { 0 } , h \in [ H ]$ , and $( s , a )$

$$
\bar { d } _ { h } ^ { ( t ) } ( s , a ) : = \sum _ { i = 1 } ^ { t - 1 } d _ { h } ^ { ( i ) } ( s , a ) \geq \frac { c _ { 0 } ( t - 1 ) } { C _ { \mathrm { c o v } } } \bar { d } _ { h } ( s , a ) .\tag{16}
$$

This is the only property of coverage needed for the pooled-regression step. It can be assumed directly or derived from an approximate exploration condition that guarantees the accumulated actual occupancies dominate the basis mixture.

For a preference w, define the best value achievable by the chosen basis as

$$
\mathcal { I } _ { K } ^ { \star } ( w ) : = \operatorname* { m a x } _ { k \in [ K ] } \mathcal { I } ( \pi _ { \mathfrak { b } k } ^ { \star } ; w ) = \operatorname* { m a x } _ { \nu \in \Delta ^ { K - 1 } } \sum _ { k = 1 } ^ { K } \nu ( k ) \mathcal { I } ( \pi _ { \mathfrak { b } k } ^ { \star } ; w ) .
$$

The equality holds because the objective is linear in the mixing weights. We measure basis-relative regret by

$$
\mathsf { R e g } _ { K } ( T ) : = \sum _ { t = 1 } ^ { T } \left[ \mathcal { T } _ { K } ^ { \star } ( w _ { t } ) - \mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \right] .
$$

Proposition B.10. Suppose scalarized rewards are bounded in [0, 1] per step, Bellman realizability and completeness hold for each per-basis class $\mathcal { F } _ { k } ,$ , Assumption B.9 holds with tolerance $\varepsilon ,$ and the stable-coverage condition (16) holds after a burn-in time $t _ { 0 } ~ = ~ \mathcal { O } ( C _ { \mathrm { c o v } } K \beta )$ , where $\beta : =$ $c _ { 1 } \log ( K | \mathcal { F } | T / \delta )$ . Then, with probability at least $1 - \delta _ { \mathrm { { ; } } }$ , the pooled-confidence routed-ensemble algorithm satisfies

$$
\mathsf { R e g } _ { K } ( T ) \le \mathcal { O } \Bigl ( H \sqrt { C _ { \mathrm { c o v } } \beta T } \left( \sqrt { 1 + \Gamma } + \varepsilon \right) + H C _ { \mathrm { c o v } } K \beta \Bigr ) .\tag{17}
$$

When $\varepsilon = 0$ and the basis is exact, $\mathsf { R e g } _ { K } ( T ) = \mathsf { R e g } ( T )$ and this reduces to Theorem 3.6.

Proof. The proof follows the same sketch as Theorem 3.6 (Appendix B.1), with two simplifications and one change. First, the comparator is changed from the true optimal value $\mathcal { I } ( \pi _ { w } ^ { \star } ; w )$ to the basisoracle value $\bar { \mathcal { I } } _ { K } ^ { \star } ( w )$ , so routed optimism becomes exact against this weaker comparator (Lemma B.11 below). Second, the conflict factor $\sqrt { \phi _ { h } ( d _ { h } ^ { ( t ) } ) }$ inflates from $\sqrt { 1 + \Gamma }$ (Lemma B.2) to $\sqrt { 1 + \Gamma } + \varepsilon$ (Lemma B.12 below), because deployed occupancies are only approximately in the basis hull. Third, the witness-comparison and pooled-regression steps are unchanged: they use only the stable coverage condition (16) and Bellman completeness, neither of which is affected by relaxing Assumption 3.4. We carry out the modifications below.

First, we show the following claim on the optimism of the routed basis.

Claim B.11. On the optimism event $E _ { \mathrm { o p t } }$ (defined in Appendix $B . I ) ,$ for every episode t,

$$
{ V _ { 1 } ^ { f } } ^ { ( t ) } ( s _ { 0 } ; w _ { t } ) ~ \geq ~ \mathcal { T } _ { K } ^ { \star } ( w _ { t } ) .
$$

Proof. Let $k _ { K } ^ { \star } ( w _ { t } ) \in \arg \operatorname* { m a x } _ { k \in [ K ] } \mathcal { I } ( \pi _ { \mathsf { b } k } ^ { \star } ; w _ { t } )$ . By the optimism event $E _ { \mathrm { o p t } }$ , the true Q-function $Q ^ { \pi _ { \mathsf { b } k } ^ { \star } } \kappa ^ { ( w _ { t } ) } \left( \cdot ; w _ { t } \right)$ belongs to the confidence set $\mathcal { F } _ { k _ { K } ^ { \star } \left( w _ { t } \right) } ^ { \left( t - 1 \right) } \left( \beta \right)$ . The routed optimistic choice in equation (7) maximizes over all basis indices and all functions in their confidence sets, so we may restrict the maximization to the single feasible candidate $( k _ { K } ^ { \star } ( w _ { t } ) , Q ^ { \pi _ { \mathsf { b } k _ { K } ^ { \star } } ^ { \star } ( w _ { t } ) } )$ . This gives

$$
{ V _ { 1 } ^ { f ^ { ( t ) } } ( s _ { 0 } ; w _ { t } ) \geq V _ { 1 } ^ { Q ^ { \pi _ { \mathsf { b } ^ { k } } ^ { \star } \star ( w _ { t } ) } } ( s _ { 0 } ; w _ { t } ) = \mathcal { I } ( \pi _ { \mathsf { b } ^ { k } _ { K } ( w _ { t } ) } ^ { \star } ; w _ { t } ) = \mathcal { I } _ { K } ^ { \star } ( w _ { t } ) . }
$$

Second, we demonstrate regret reduction via a certain identity. This step is identical to the regret reduction in equation (8). Working on the intersection $E _ { \mathrm { o p t } } \cap E _ { \mathrm { r e g } }$ (probability at least $1 - \delta )$ Lemma B.11 together with the standard layerwise telescoping identity gives

$$
\mathsf { R e g } _ { K } ( T ) \le \sum _ { t = 1 } ^ { T } \Bigl [ V _ { 1 } ^ { f ^ { ( t ) } } \bigl ( s _ { 0 } ; w _ { t } \bigr ) - \mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \Bigr ] = \sum _ { t = 1 } ^ { T } \sum _ { h = 1 } ^ { H } \mathbb { E } _ { d _ { h } ^ { ( t ) } } \Bigl [ \delta _ { h } ^ { ( t ) } \Bigr ] .\tag{18}
$$

Below, we control the Bellman residual term on the right-hand side.

Identical to equation (9), for every $t \in [ T ]$ and $h \in [ H ]$ , we have

$$
\mathbb { E } _ { d _ { h } ^ { ( t ) } } \left[ \delta _ { h } ^ { ( t ) } \right] \leq \sqrt { \phi _ { h } ( d _ { h } ^ { ( t ) } ) } \cdot \sqrt { \mathbb { E } _ { \bar { d } _ { h } } \left[ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } \right] } .\tag{19}
$$

Next, we show the following approximate conflict bound, which is the only step affected by ε.

Claim B.12. Every policy π satisfying the decomposition in Assumption B.9 also satisfies, for every $h \in [ H ]$

$$
\sqrt { \phi _ { h } ( d _ { h } ^ { \pi } ) } \ \leq \ \sqrt { 1 + \Gamma } + \varepsilon .\tag{20}
$$

Proof. Since $\phi _ { h } ( d ) = \| d \| _ { \bar { d } _ { h } ^ { - 1 } } ^ { 2 }$ , Assumption B.9 and the triangle inequality in $\ell _ { 2 }$ norm give

$$
\sqrt { \phi _ { h } ( d _ { h } ^ { \pi } ) } \ = \ \lVert q _ { \pi , h } + e _ { \pi , h } \rVert _ { \vec { d } _ { h } ^ { - 1 } } \ \le \ \lVert q _ { \pi , h } \rVert _ { \vec { d } _ { h } ^ { - 1 } } + \lVert e _ { \pi , h } \rVert _ { \vec { d } _ { h } ^ { - 1 } } \ \le \ \sqrt { \phi _ { h } ( q _ { \pi , h } ) } + \varepsilon .
$$

The first summand is controlled exactly as in Lemma B.2: by convexity of $\phi _ { h }$ and Definition 3.3,

$$
\begin{array} { r } { \phi _ { h } ( q _ { \pi , h } ) = \phi _ { h } \Bigl ( \sum _ { k } \nu _ { \pi } ( k ) d _ { h } ^ { \pi _ { b k } ^ { \star } } \Bigr ) \leq \sum _ { k } \nu _ { \pi } ( k ) \phi _ { h } ( d _ { h } ^ { \pi _ { b k } ^ { \star } } ) \leq 1 + \Gamma . } \end{array}
$$

Combining this display with the triangle-inequality bound proves equation (20). Applying Lemma B.12 to the deployed policy $\pi ^ { ( t ) }$ yields

$$
\sqrt { \phi _ { h } ( d _ { h } ^ { ( t ) } ) } \leq \sqrt { 1 + \Gamma } + \varepsilon .\tag{21}
$$

The cumulative-occupancy lower bound in equation (16), combined with the pooled-regression event $E _ { \mathrm { r e g } } ,$ gives the same in-sample-to-witness bound as Lemma B.4: for every $t \geq t _ { 0 }$ and ${ \bf \bar { \boldsymbol { h } } } \in [ H ]$

$$
\begin{array} { r l r } {  { \sqrt { { \mathbb { E } } _ { \bar { d } _ { h } } [ ( \delta _ { h } ^ { ( t ) } ) ^ { 2 } ] } \ \leq \ \sqrt { \frac { C _ { \mathrm { c o v } } \beta } { c _ { 0 } ( t - 1 ) } } . } } \end{array}\tag{22}
$$

This step depends only on equation (16), the pooled-regression event $E _ { \mathrm { r e g } }$ , and Bellman completeness.   
It does not use exact basis decomposition.

Finally, combining equations (19), (21), and (22), for every $t \geq t _ { 0 }$ and $h \in [ H ]$

$$
\mathbb { E } _ { d _ { h } ^ { ( t ) } } \Big [ \delta _ { h } ^ { ( t ) } \Big ] \ \leq \ \big ( \sqrt { 1 + \Gamma } + \varepsilon \big ) \sqrt { \frac { C _ { \mathrm { c o v } } \beta } { c _ { 0 } ( t - 1 ) } } .\tag{23}
$$

We split the regret in equation (18) at the burn-in time $t _ { 0 }$ and bound each part.

For each $\begin{array} { r } { t < t _ { 0 } , \sum _ { h } \mathbb { E } _ { d _ { k } ^ { ( t ) } } [ \delta _ { h } ^ { ( t ) } ] = V _ { 1 } ^ { f ^ { ( t ) } } ( s _ { 0 } ; w _ { t } ) - \mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \le H } \end{array}$ , since both quantities lie in $[ 0 , H ]$ . Hence, the burn-in contribution is at most $H t _ { 0 }$

Substituting equation (23) and using $\begin{array} { r } { \sum _ { t = t _ { 0 } } ^ { T } ( t - 1 ) ^ { - 1 / 2 } \leq \sum _ { u = 1 } ^ { T - 1 } u ^ { - 1 / 2 } \leq 2 \sqrt { T } , } \end{array}$

$$
\begin{array} { r l } { \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { d _ { h } ^ { ( t ) } } \left[ \delta _ { h } ^ { ( t ) } \right] \leq H \big ( \sqrt { 1 + \Gamma } + \varepsilon \big ) \sqrt { \frac { C _ { \mathrm { c o v } } \beta } { c _ { 0 } } } \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \frac { 1 } { \sqrt { t - 1 } } } & { \leq 2 H \big ( \sqrt { 1 + \Gamma } + \varepsilon \big ) \sqrt { \frac { C _ { \mathrm { c o v } } \beta T } { c _ { 0 } } } . } \end{array}
$$

Combined together, substituting $t _ { 0 } = \mathcal { O } ( C _ { \mathrm { c o v } } K \beta )$ , and treating $t _ { 0 }$ as a fixed constant yields

$$
\mathsf { R e g } _ { K } ( T ) \leq \mathcal { O } \Bigl ( H \sqrt { C _ { \mathrm { c o v } } \beta T } \cdot \bigl ( \sqrt { 1 + \Gamma } + \varepsilon \bigr ) + H C _ { \mathrm { c o v } } K \beta \Bigr ) ,
$$

which leads to equation (17).

Lastly, we connect basis-relative regret $\mathsf { R e g } _ { K } ( T )$ to the cumulative regret $\mathsf { R e g } ( T )$ against the true optimal policy via a clean two-term decomposition. For a fixed basis $\{ \pi _ { \mathsf { b } 1 } ^ { \star } , \hdots , \overline { { \pi _ { \mathsf { b } K } ^ { \star } } } \}$ , define the cumulative basis-approximation error along the realized preference sequence as

$$
A _ { K } ( T ) = \sum _ { t = 1 } ^ { T } [ \mathcal { I } ( \pi _ { w _ { t } } ^ { \star } ; w _ { t } ) - \mathcal { I } _ { K } ^ { \star } ( w _ { t } ) ] .
$$

Since every basis policy is feasible, $\mathcal { T } _ { K } ^ { \star } ( w _ { t } ) \leq \mathcal { I } ( \pi _ { w _ { t } } ^ { \star } ; w _ { t } )$ for every t, so $A _ { K } ( T ) \geq 0$ . The quantity $A _ { K } ( T )$ depends on the basis $\{ \pi _ { \mathsf { b } k } ^ { \star } \}$ and the preference distribution $\mathcal { P } _ { \cdot }$ , but not on the algorithm.

First, we present the exact decomposition of regret relative to the true optimal policy. By telescoping the comparator $\mathcal { I } _ { K } ^ { \star } ( w _ { t } )$ in and out of the regret sum,

$$
\mathsf { R e g } ( T ) = \underbrace { \sum _ { t = 1 } ^ { T } \bigl [ \mathcal { I } ( \pi _ { w _ { t } } ^ { \star } ; w _ { t } ) - \mathcal { I } _ { K } ^ { \star } ( w _ { t } ) \bigr ] } _ { A _ { K } ( T ) } + \underbrace { \sum _ { t = 1 } ^ { T } \Bigl [ \mathcal { I } _ { K } ^ { \star } ( w _ { t } ) - \mathcal { I } ( \pi ^ { ( t ) } ; w _ { t } ) \Bigr ] } _ { \mathsf { R e g } _ { K } ( T ) } .\tag{24}
$$

Proposition B.10 controls the second term; the first depends only on the chosen basis. Second, we bound $A _ { K } ( T )$ when optimal policies are also approximately in the hull. Following Assumption B.9, we have $\boldsymbol { d } _ { h } ^ { \pi _ { w } ^ { \star } } = \boldsymbol { q } _ { w , h } + \boldsymbol { e } _ { w , h }$ . Since scalarized rewards $r _ { h } ^ { w } ( s , a )$ are bounded in $[ 0 , 1 ]$ , its inner product with approximation error $e _ { w , h }$ has

$$
\begin{array} { r } { \langle e _ { w , h } , r _ { h } ^ { w } \rangle \leq \big | \langle e _ { w , h } , r _ { h } ^ { w } \rangle \big | \leq \| e _ { w , h } \| _ { \bar { d } _ { h } ^ { - 1 } } \sqrt { \mathbb { E } _ { \bar { d } _ { h } } [ ( r _ { h } ^ { w } ) ^ { 2 } ] } \leq \varepsilon . } \end{array}
$$

Using the same mixing vector $\nu _ { w }$ at every stage,

$$
\begin{array} { r l } & { \mathcal { J } ( \pi _ { w } ^ { \star } ; w ) = \displaystyle \sum _ { h = 1 } ^ { H } \langle d _ { h } ^ { \pi _ { w } ^ { \star } } , r _ { h } ^ { w } \rangle \leq \displaystyle \sum _ { h = 1 } ^ { H } \langle q _ { w , h } , r _ { h } ^ { w } \rangle + H \varepsilon } \\ & { \quad \quad \quad = \displaystyle \sum _ { k = 1 } ^ { K } \nu _ { w } ( k ) \mathcal { I } ( \pi _ { \mathsf { b } k } ^ { \star } ; w ) + H \varepsilon \leq \mathcal { I } _ { K } ^ { \star } ( w ) + H \varepsilon . } \end{array}
$$

Summing over $t \in [ T ]$ yields $A _ { K } ( T ) \leq H \varepsilon T$ . Lastly, combining equation (24) with the above, and Proposition B.10, we have

$$
\mathsf { R e g } ( T ) \leq \mathcal { O } \Bigl ( H \varepsilon T + H \sqrt { C _ { \mathrm { c o v } } \beta T } \cdot \bigl ( \sqrt { 1 + \Gamma } + \varepsilon \bigr ) + H C _ { \mathrm { c o v } } K \beta \Bigr ) .
$$

## C Omitted Experiments and Implementation Details

In this section, we describe several omitted experimental results as well as the implementation details that are left out from Section 4.

We conduct the MO-Gymnasium experiments on an AMD Ryzen 7 9700X CPU, with a total training time of approximately 72 hours. Meta-World and RLHF experiments are performed on a server with an Intel Xeon E5-2623 v3 CPU and NVIDIA RTX A6000 GPUs. The Meta-World training takes 12 hours, and the RLHF experiment takes 1 hour.

Next, we define the performance metrics used in our experiments. We compute Γ based on the $\chi ^ { 2 } \cdot$ divergence between the state occupancy distributions of single-objective basis policies. To account for the temporal diffusion effect described above, we use the time-averaged state occupancy

$$
\bar { d } ^ { \pi } ( s ) = \frac { 1 } { H } \sum _ { t = 0 } ^ { H } P ( s _ { t } = s ) , \mathrm { t h e n } \Gamma = \operatorname* { m a x } _ { k } { \sum _ { s \in S } { \frac { { ( \bar { d } ^ { \pi _ { k } } ( s ) - \bar { d } ^ { \mathrm { a v g } } ( s ) ) } ^ { 2 } } { { \bar { d } } ^ { \mathrm { a v g } } ( s ) } } } ,
$$

Table 7: Hyper-parameters for the routed ensemble procedure used in Algorithm 1.
<table><tr><td>Parameter</td><td>Meta-World</td><td>MO-Gymnasium</td></tr><tr><td>Number of basis policies</td><td>3</td><td>3</td></tr><tr><td>Base optimizer</td><td> $\mathbf { A d a m }$ </td><td> $\mathbf { A d a m }$ </td></tr><tr><td>SAC buffer size</td><td> ${ 1 0 } ^ { 6 }$ </td><td> ${ 1 0 } ^ { 6 }$ </td></tr><tr><td>Learning rate (Actor)</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate (Critic)</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Discount factor</td><td> $0 . 9 9$ </td><td> $0 . 9 9$ </td></tr><tr><td>Batch size</td><td>1280</td><td>256</td></tr><tr><td>Network architecture</td><td> $\mathrm { M L P 2 5 6 \times 2 5 6 }$ </td><td> $\mathrm { M L P 2 5 6 \times 2 5 6 }$ </td></tr></table>

where $\pi _ { k }$ is the policy trained on objective k alone, and $\bar { d } ^ { \mathrm { a v g } }$ is the uniform mixture of all basis occupancies.

We train a kernel ridge regression model <sup>ˆ</sup>f on the dataset of $\left\{ \left( \mathbf { m } _ { i } , y _ { i } \right) \right\}$ . The attribution of a conflicting objective j (where $j \in \{ 5 , \ldots , 9 \} )$ is calculated as the predicted marginal gain $\hat { f } ( \mathbf { m } _ { \mathrm { t a r g e t } } \cup \{ j \} ) -$ $\hat { f } ( \mathbf { m } _ { \mathrm { t a r g e t } } )$ , where $\mathbf { m } _ { \mathrm { t a r g e t } }$ is the mask containing only the target objective.

Lastly, we provide the detailed hyperparameter settings used in our experiments to ensure reproducibility. Table 7 lists the parameters for the routed ensemble algorithm across different benchmarks. For routed ensemble, we adapt the base policy-learning algorithm for each environment. We use Soft Modularization [51] for Meta-World, Q-Pensieve [18] for continuous ${ \bf M O - G y }$ mnasium tasks, and PQL [42] for discrete MO-Gymnasium tasks. For all baselines, we use the hyperparameters from their official implementations/repositories to ensure a fair comparison. Each baseline is trained for $5 \times 1 0 ^ { 5 }$ time steps on continuous benchmarks and $1 0 ^ { 4 }$ time steps on discrete benchmarks.

We implement kernel-based objective attribution in Algorithm 2 using a RBF kernel defined as $\psi ( S _ { i } , \hat { S } _ { j } ) = \exp ( - \gamma \| S _ { i } - S _ { j } \| ^ { 2 } )$ . We determine the kernel bandwidth $\gamma = 0 . 1$ and the regularization coefficient $\lambda = 0 . 1$ via cross-validation on the validation set.

## C.1 Synthetic environments

Multi-objective environments. We utilize a 2D parametric optimization environment. The policy is parameterized by a coordinate vector $\theta \in \mathbb { R } ^ { 2 }$ . The agent aims to maximize a scalarized return $J ( \theta , w ) = w \cdot r _ { 1 } ( \theta ) + ( 1 - w ) \cdot r _ { 2 } ( \theta )$ , where $w \in [ 0 , 1 ]$ represents the preference weight. We optimize θ using the Adam optimizer across a sweep of discrete weights w. All environments include a small regularization penalty $- 0 . 1 \| \theta \| _ { 2 } ^ { 2 }$ . We define three distinct regimes:

Case 1 (Aligned objectives). The reward functions are quadratic penalties centered at proximal locations $\mu _ { 1 } = ( 1 , 1 )$ and $\mu _ { 2 } = ( 1 . 2 , 1 . 2 )$ . This configuration simulates tasks where improving one objective implicitly benefits the other.

Case 2 (Conflicting objectives with a linearfrontier). We construct a non-convex reward landscape to induce a disjoint set of optimal policies. The targets are separated at $\mu _ { 1 } = ( - 2 , 0 )$ and $\mu _ { 2 } = ( 2 , 0 )$ Crucially, we introduce a Gaussian barrier penalty $P ( \theta ) \stackrel { \cdot } { = } - 1 0 \exp \bigl ( - \theta _ { x } ^ { 2 } \bigr )$ centered at the origin. This barrier prevents the optimization from settling in the intermediate region. Consequently, the optimal deterministic policies collapse to the extreme corners, and the Pareto frontier is formed by the linear convex hull of these solutions.

Case 3 (Conflicting objectives with a concavefrontier). We remove the central barrier while maintaining the separated targets $\mu _ { 1 } = ( - 2 , 0 )$ and $\mu _ { 2 } = ( 2 , 0 )$ . The reward functions are standard negative Euclidean distances. This convex landscape allows for smooth interpolation between objectives. It results in a strictly concave Pareto frontier, in which intermediate policies outperform the linear combination of the corner solutions.

Nonlinear synergy. We introduce an environment for conducting synthetic experiments to demonstrate synergy effects. This environment creates a synergy trap in which two objectives are individually detrimental or neutral but highly beneficial when combined.

We define a 2D parametric optimization task with $M = 1 2$ distinct objectives, indexed by $j \in$ $\{ 0 , 1 , \ldots , M - 1 \bar  \}$ . The objectives are arranged equidistantly on a unit circle (a clock face). The goal vector for the j-th objective, $v _ { j } \in \mathbb { R } ^ { 2 }$ , is defined by the angle $\begin{array} { r } { \phi _ { j } = \frac { \pi } { 2 } - j \cdot \frac { 2 \pi } { M } } \end{array}$

$$
\begin{array} { r } { v _ { j } = [ \cos ( \phi _ { j } ) , \sin ( \phi _ { j } ) ] ^ { \top } . } \end{array}
$$

Here, objective 0 (north) is the designated target with $v _ { \mathrm { t a r g e t } } = [ 0 , 1 ] ^ { \top }$ . The environment employs a specialized reward function $R ( a , S )$ for an action $a \in \mathbb { R } ^ { 2 }$ given a subset of active objectives $S \subseteq \{ 0 , \dots , M - 1 \}$ , represented by a binary mask $m \in \{ 0 , 1 \} ^ { \tilde { M } }$ . The reward consists of a standard linear component and a nonlinear synergy component:

$$
R ( a , S ) = a ^ { \top } \left( \sum _ { j \in S } v _ { j } \right) + \mathbb { I } _ { 3 \in S \wedge 9 \in S } \cdot \kappa _ { \mathrm { s y n } } \cdot ( a ^ { \top } v _ { \mathrm { t a r g e t } } ) ,
$$

where $\kappa _ { s y n } = 5 . 0$ is the synergy bonus coefficient.

The agent is modeled as a stochastic policy $\pi _ { \boldsymbol { \theta } } ( a ) = \mathcal { N } ( \mu _ { \boldsymbol { \theta } } , \Sigma )$ , where $\mu _ { \theta } = \operatorname { t a n h } ( \theta )$ and $\Sigma = \sigma ^ { 2 } I$ (with $\sigma = 0 . 2 )$ . The parameter $\theta \ \bar { \in } \ \mathbb { R } ^ { 2 }$ is optimized via policy gradient, which optimizes the parameters of a stochastic policy by maximizing expected reward through Monte Carlo sampling, using the Adam optimizer with a learning rate of 0.1 for a horizon of $T _ { e p } = 6 0$ steps.

A tabular environment on a grid. To validate the theoretical interactions between the objective conflict coefficient Γ and the coverability coefficient $C _ { c o v }$ , we implement a controlled tabular environment. This setup allows for the exact calculation of occupancy measures and $\chi ^ { 2 } .$ -divergences, avoiding the approximation errors inherent in deep RL settings.

We consider a MOMDP modeled as an $N \times N$ grid (default $N = 5 )$ , defined by the tuple $\mathcal { M } =$ $( S , \mathcal { A } , P r , r , \gamma )$ . Consistent with the conflicting-objectives regime, we construct m objectives based on spatial navigation to distinct goal coordinates. Let $\mathcal { G } = \mathbf { \bar { \{ } }  g _ { 1 } , \dotsc , g _ { m } \} \subset \mathcal { S }$ be a set of diverse goal states. The vector-valued reward function $r ( s , a ) \in \mathbb { R } ^ { m }$ is defined using the negative Manhattan distance $( \ell _ { 1 }$ norm): $r _ { k } ( s , a ) = - \| { \mathrm { c o o r d } } ( s ^ { \prime } ) - { \mathrm { c o o r d } } ( g _ { k } ) \| _ { 1 }$ <sub>1</sub>, where $s ^ { \prime }$ is the next state and coord( ) maps a state index to its $( x , y )$ grid position.

We compute optimal policies exactly via soft value iteration. For a given preference weight $w \in$ $\Delta ^ { m - 1 }$ , the scalarized reward is $r _ { w } = \langle w , r \rangle$ . The policy $\pi _ { w } ^ { * }$ is derived from the soft Bellman operator:

$$
V ( s ) = \tau \log \sum _ { a } \exp \left( \frac { Q ( s , a ) } { \tau } \right) , \quad \pi ( a | s ) \propto \exp \left( \frac { Q ( s , a ) } { \tau } \right) ,
$$

with temperature $\tau = 0 . 5$ . The exact state-action occupancy distribution $d ^ { \pi } \in \mathbb { R } ^ { | S | | A | }$ is computed by solving the linear system:

$$
d ^ { \pi } = ( I - \gamma P _ { \pi } ) ^ { - 1 } d _ { 0 } ,
$$

where $P _ { \pi }$ is the policy-induced transition matrix and $d _ { 0 }$ is the uniform initial distribution.

We explicitly calculate the conflict coefficient. We define the target mixture distribution $\bar { d }$ as the uniform average of the occupancies of the single-objective experts:

$$
\bar { d } ( s , a ) = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } { d ^ { \pi _ { k } ^ { \star } } ( s , a ) } .
$$

The conflict coefficient is then measured as the $\chi ^ { 2 } .$ -divergence between a candidate policy’s occupancy and this target mixture:

$$
\Gamma ( \pi ) = \chi ^ { 2 } ( d ^ { \pi } \| \bar { d } ) = \sum _ { s , a } \frac { ( d ^ { \pi } ( s , a ) - \bar { d } ( s , a ) ) ^ { 2 } } { \bar { d } ( s , a ) + \epsilon } .
$$

We utilize this metric to compare coverability in Section 4.3. We find that a single scalarized policy $\pi _ { s i n g l e }$ exhibits a high $\chi ^ { 2 }$ -divergence, indicating poor coverage of the Pareto frontier. In contrast, a routed ensemble with $K$ basis policies significantly reduces this divergence.

## C.2 Additional empirical results

Empirical evaluation of the basis assumption. We measure the residual ϵ obtained by approximating preference-policy occupancies with convex combinations of basis-policy occupancies. Tabular occupancies are computed exactly, while benchmark occupancies are estimated from visitation distributions. Table 8 reports the mean and maximum residuals over the evaluated preferences.

Table 8: Convex-hull occupancy residuals for exact tabular occupancies and empirical benchmark visitation distributions. Mean and maximum residuals are computed over the evaluated preferences.
<table><tr><td>Setting</td><td>Basis construction</td><td>Number of policies</td><td>Mean €</td><td>Max €</td></tr><tr><td>Tabular, aligned</td><td>Exact</td><td>1</td><td> $\sim 1 0 ^ { - 1 8 }$ </td><td> $\sim 1 0 ^ { - 1 7 }$ </td></tr><tr><td>Tabular, linear</td><td>Exact</td><td>2</td><td> $\sim 1 0 ^ { - 9 }$ </td><td> $\sim 1 0 ^ { - 8 }$ </td></tr><tr><td>Tabular, concave</td><td>Exact</td><td>3</td><td> $0 . 1 8 6$ </td><td> $0 . 5 5$ </td></tr><tr><td>Tabular, concave</td><td>Exact</td><td>21</td><td> $1 . 4 5 \times 1 0 ^ { - 4 }$ </td><td> $6 . 1 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>MO-Hopper</td><td>Candidate-selected</td><td>3</td><td>0.443</td><td>0.80</td></tr><tr><td>MO-Hopper</td><td>Candidate-selected</td><td>15</td><td>0.231</td><td>0.51</td></tr><tr><td>MO-Hopper</td><td>Learned by Alg. 1</td><td>3</td><td>1.33</td><td>1.42</td></tr><tr><td>Deep-Sea-Treasure</td><td>Learned by Alg. 1</td><td>6</td><td>0.15</td><td>0.75</td></tr><tr><td>Fruit-Tree</td><td>Oracle</td><td>6</td><td>0.553</td><td>0.85</td></tr><tr><td>GridWorld, 21 distinct optima</td><td>Restricted learner</td><td>2</td><td>0.475</td><td>3.474</td></tr></table>

In the concave tabular setting, the mean residual decreases from 0.186 at three bases to 0.088, 0.024, and $1 . 4 5 \times 1 0 ^ { - 4 }$ at five, nine, and 21 bases, respectively. On MO-Hopper, increasing the candidateselected basis from three to 15 policies decreases the mean residual from 0.443 to 0.231, whereas the three policies learned by Algorithm 1 have mean residual 1.33. These measurements show that a larger candidate basis can approximate the evaluated occupancies more closely, but the learned ensemble need not recover the exact structure assumed in the analysis.

Severe basis under-specification. In a controlled GridWorld with 21 distinct preference-optimal policies, we restrict the learner to two policies for 3000 episodes. The resulting mean and maximum occupancy residuals are 0.475 and 3.474, and the cumulative value excess is 2780.4. This test measures performance when the available ensemble is substantially smaller than the set of preference optimal policies.

Comparison on RLHF tasks. We evaluate the extension of our approach to RLHF settings on synthetic data, HelpSteer2 [43], and UltraFeedback [10]. For the synthetic experiment, we construct a synthetic preference dataset that captures the helpfulness-safety trade-off in RLHF. Each prompt is paired with candidate textual responses corresponding to different response styles, while rewards are generated from a controlled two-objective preference function. This allows us to study expert routing under interpretable language scenarios while retaining access to an oracle policy. For the real-data experiments, we evaluate Qwen3-1.7B and Llama-3.2-1B on both HelpSteer2 and UltraFeedback, yielding four model–dataset configurations. All four configurations follow the same training protocol, with per-basis LoRA adapters trained using PPO.

Table 9: Summary of trajectory attribution results on two MO-Gymnasium benchmarks, Deep Sea Treasure and Fruit Tree.
<table><tr><td></td><td>Deep Sea Treasure</td><td>Fruit Tree</td></tr><tr><td>Influence functions</td><td> $\overline { { 0 . 2 8 _ { \pm 0 . 5 1 } } }$ </td><td> $\overline { { 0 . 3 0 { \scriptstyle \pm 0 . 5 7 } } }$ </td></tr><tr><td>Linear surrogate models</td><td> $0 . 6 2 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $0 . 7 1 { \scriptstyle \pm 0 . 5 3 }$ </td></tr><tr><td>TracIn</td><td> $0 . 7 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $0 . 7 5 { \scriptstyle \pm 0 . 4 5 }$ </td></tr><tr><td>Algorithm 2</td><td> $\mathbf { 0 . 7 5 { \scriptstyle \pm 0 . 1 4 } }$ </td><td> $\mathbf { 0 . 8 1 } _ { \pm 0 . 1 8 }$ </td></tr></table>

Choice of N. Finally, we present comparison results by varying the size of the ensemble policy. In Figure 4, we illustrate the choice of N on the three datasets, including MO-HalfCheetah2d, MO-Hopper, and MO-Ant environments.

![](images/cde8a0d7e703cabd1b12db0f38d07ca81be4ae7b127c25ad6b2ae23125e58b9c.jpg)  
N

![](images/07b46a3d3a8b69167e3b22115cde23ac567a2ab7547f9bc161093881e9415900.jpg)  
N

![](images/1be3dc7ce80021c8167b44edde285a485d73c4a0391bc393247b5180f28fbfd5.jpg)  
Figure 4: Ablation analysis of the choice of the ensemble size $N ,$ measured by hypervolume (HV).

## C.3 Proximal policy optimization

In proximal policy optimization (PPO), access to the exact $A ^ { \pi }$ is not available. Instead, one must use an advantage estimator $\widehat { A } _ { t }$ computed from rollout trajectories and a learned value function $V _ { \phi } . \mathrm { ~ A ~ }$ common choice is generalized advantage estimation (GAE), where the temporal-difference residual at time t is

$$
\delta _ { t } = r _ { t } + \gamma V _ { \phi } ( s _ { t + 1 } ) - V _ { \phi } ( s _ { t } ) ,
$$

and the advantage estimate is

$$
\widehat { A } _ { t } = \sum _ { l = 0 } ^ { L - 1 } ( \gamma \lambda ) ^ { l } \delta _ { t + l } ,
$$

for some truncation horizon $L ,$ the discount factor $\gamma \in [ 0 , 1 )$ , and GAE parameter $\lambda \in [ 0 , 1 ]$ . More generally, we use $\widehat { A } _ { t }$ to denote any scalar advantage estimate derived from the rollout and a baseline; our theory treats these $\widehat { A } _ { t }$ as given and is independent of the particular estimator.

Training proceeds in rounds $k = 0 , 1 , 2 , \ldots$ . In round k, the current policy $\pi _ { \theta ^ { ( k ) } }$ interacts with the environment and collects a rollout buffer $B ^ { ( k ) } = \{ z _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { n _ { k } }$ , where each record

$$
z _ { i } ^ { ( k ) } = ( s _ { i } , a _ { i } , r _ { i } , \log \pi _ { \theta ^ { ( k ) } } ( a _ { i } \mid s _ { i } ) , v _ { i } , \widehat { A } _ { i } )
$$

contains the transition $\left( { { s _ { i } } , { a _ { i } } , { r _ { i } } } \right)$ together with auxiliary quantities such as the log-probability under the current policy, a discounted value estimate $v _ { i }$ , and an advantage estimate $\widehat { A } _ { i }$ . These records are the atomic units used in PPO updates and form the basic entities for local attribution.

Given a buffer $B ^ { ( k ) }$ , PPO performs a number of gradient steps to update θ from $\theta ^ { ( k ) } \mathrm { t o } \theta ^ { ( k + 1 ) }$ by maximizing a clipped surrogate objective. For a single record $z = ( s , a , r , \ldots )$ , let $\ell _ { \mathrm { P P O } } ( \boldsymbol { \theta } ; z )$ denote the per-record loss corresponding to the standard clipped PPO objective (possibly combined with value and entropy terms). The empirical PPO loss over the buffer is then

$$
\hat { L } ^ { ( k ) } ( \theta ) = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } \ell _ { \mathrm { P P O } } ( \theta ; z _ { i } ^ { ( k ) } ) ,
$$

and PPO updates $\theta$ to approximately minimize $\hat { L } ^ { ( k ) } ( \theta )$ , starting from $\theta ^ { ( k ) }$