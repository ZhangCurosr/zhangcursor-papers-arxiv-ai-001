# Beyond Reward Suppression: Near-Optimal Offline Attacks on Warm-Start Bandits with Bounded Rewards

Qirun Zeng<sup>1</sup> Manhin Poon<sup>1</sup> Xiangxiang Dai<sup>2</sup> Qixin Zhang<sup>3</sup> Jinhang Zuo<sup>1</sup>

<sup>1</sup>City University of Hong Kong, Hong Kong SAR, China <sup>2</sup>The Chinese University of Hong Kong, Hong Kong SAR, China <sup>3</sup>Nanyang Technological University, Singapore

{qirun.zeng,manhpoon4-c}@my.cityu.edu.hk xxdai23@cse.cuhk.edu.hk qixin.zhang@ntu.edu.sg jinhang.zuo@cityu.edu.hk

## Abstract

Adversarial attacks on bandits aim to mislead a learner toward a target arm while keeping the attack cost small. Existing attacks typically achieve this by suppressing non-target arms. In practice, however, manipulation such as fake reviews often directly promotes the target item. We study this gap through bounded offline attacks on warm-start bandits, where an attacker can inject only valid action–reward pairs into the warm-start history before deployment. We show that target promotion is not merely a heuristic: when the target arm lies near the lower reward boundary, any order-optimal-cost attack against UCB that makes it selected in nearly all online rounds must allocate a nonvanishing fraction of its cost to the target arm. We then design an attack that achieves the optimal sublinear cost and characterize its allocation between target promotion and non-target suppression. We further extend the attack to Thompson Sampling, ϵ-greedy, and a broader class of bandit algorithms. Experiments on real-world and synthetic data validate the effectiveness of our attacks.

## 1 Introduction

Multi-armed bandit (MAB) algorithms provide a principled framework for sequential decisionmaking under uncertainty and have been widely adopted in applications such as recommender systems and online advertising. In many practical applications, however, bandit algorithms are not deployed from scratch. Instead, they are initialized with historical interaction data, giving rise to the warm-start bandit setting [Shivaswamy and Joachims, 2012, Li et al., 2021, Banerjee et al., 2022, Oetomo et al., 2023]. While such historical data can improve early decisions and reduce exploration costs, they also create a natural attack surface: fake feedback injected before deployment can bias the learner’s initial estimates, and this bias may persist even when all subsequent feedback is genuine.

Adversarial attacks on bandits have been studied extensively, but most existing work [Jun et al., 2018, Liu and Shroff, 2019, Zuo, 2024] focuses on manipulating rewards during online interaction, typically allowing arbitrary modifications to observed rewards and measuring attack cost by the total magnitude of these modifications. Zeng et al. [2025] takes a step further by injecting fake action– reward records to model fake users. Building on these works, we consider a more natural setting in which the learner is initialized with offline logs before online deployment, and the attacker can inject fake action–reward records only during this offline stage. We further measure attack cost by the number of injected samples, which more closely reflects the practical cost of generating fake interactions.

Table 1: Comparison of bandit attack models and guarantees.
<table><tr><td>Attack Strategy</td><td>Timing</td><td>Bounded Rewards</td><td>Target Promotion</td><td>Allocation Analysis</td><td>Opt.</td></tr><tr><td>Jun et al. [2018]</td><td>online</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Liu and Shroff [2019]</td><td>offline</td><td>x</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Xu et al. [2021]</td><td>online</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>Zuo [2024]</td><td>online</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Zeng et al. [2025]</td><td>online</td><td>√</td><td>x</td><td>x</td><td>X</td></tr><tr><td>Hosseini et al. [2026]</td><td>offline</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Ours (Sec. 4)</td><td>offline</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Prior work separately studies offline poisoning, bounded attacks, or attack-cost optimality, but does not characterize how attack effort should be allocated across arms. Our work combines bounded offline injection with direct target promotion and proves that cost-optimal attacks require substantial effort on the target arm itself. The online variant of Liu and Shroff [2019] is omitted since it does not satisfy any of the properties in the columns.

This bounded fake-data perspective not only better reflects settings such as review manipulation, but also makes how the attacker allocates its limited effort across arms a central question.

More importantly, existing attacks on stochastic bandits typically steer the learner toward a target arm by suppressing the observed rewards of non-target arms [Jun et al., 2018, Zuo et al., 2023, Zuo, 2024, Zeng et al., 2025]. Although such attacks admit strong theoretical guarantees with controlled cost, their mechanism differs from common real-world manipulation practices, where attackers often directly promote a chosen item through favorable feedback. For example, researchers have observed Amazon sellers recruiting reviewers through private Facebook groups and reimbursing purchases in exchange for five-star reviews, sometimes with additional commissions [He et al., 2022b]. Direct promotion is also intuitively attractive: a favorable review benefits the target itself, whereas suppressing a single competitor affects only one of many alternatives. This gap raises a natural question: when is directly promoting the target not only effective, but necessary for a cost-efficient attack?

Motivated by this mismatch, we study bounded offline attacks on warm-start stochastic bandits under two practical constraints. First, every injected reward must lie within the valid feedback range, such as binary clicks or 1–5 star ratings. Second, the attacker can intervene only before deployment, since maintaining access throughout the online learning process may be difficult or impossible. Under these constraints, target promotion and non-target suppression play fundamentally different roles. Injecting favorable feedback on the target improves its comparison against all competitors, but this initial advantage is gradually diluted as genuine online feedback accumulates. In contrast, suppressing a competitor requires injecting unfavorable feedback specifically for that arm, and bounded rewards limit how much each injected sample can change its empirical estimate. The attacker must therefore decide not only how many samples to inject, but also how to allocate these injections across arms. This leads to our central question:

## How should an attacker allocate its power between the target arm and its competitors?

We answer this question by jointly optimizing target promotion and non-target suppression. We propose a bounded offline attack against warm-start UCB and establish both an upper bound on its cost and a lower bound for any successful attack. Our lower bound analysis further reveals that the optimal allocation is necessarily target-centric: any attack achieving the leading-order optimal cost that makes the target arm selected in nearly all online rounds must spend a nonvanishing fraction of its cost on promoting the target. We further extend our strategy to the class of suppression-bounded algorithms studied by Zeng et al. [2025], with algorithm-specific analyses for Thompson Sampling and ϵ-greedy in Appendices B and C. These results demonstrate the broader applicability of our strategy beyond UCB.

## 1.1 Our Contributions

We summarize our contributions as follows:

• We formulate bounded offline attacks on warm-start stochastic bandits, measuring attack cost by the number of valid action–reward samples injected before deployment. This model captures settings such as fake-review manipulation, where the cost is associated with generating fake interactions rather than the magnitude of reward modification.

• We establish upper and lower bounds on the cost of bounded offline attacks against UCB. In the near-boundary regime, these bounds match to leading order, yielding the optimal sublinear cost scale $\Theta \big ( \bar { T } ^ { 2 / 3 } ( K \log T ) ^ { 1 / 3 } \big )$ for $K$ arms and $T$ rounds. Notably, we bridge this theory–practice gap by showing that cost-optimal attacks must allocate a nonvanishing fraction of their cost to the target arm.

• We extend our attack strategy beyond UCB to Thompson Sampling, ϵ-greedy, and the broader class of suppression-bounded algorithms introduced by Zeng et al. [2025]. This extension shows that the same combination of target promotion and non-target suppression applies to a broad class of bandit algorithms.

• We evaluate our attacks on MovieLens [Harper and Konstan, 2015] against existing baselines. The experiments demonstrate lower attack costs while maintaining effective targetarm manipulation, and empirically validate the predicted target-centric cost allocation.

## 1.2 Related Work

Research on adversarial manipulation of bandits follows two complementary directions: designing attack algorithms, and developing robust algorithms against adversarial attack.

Adversarial Attacks. Jun et al. [2018] showed that sublinear-cost attacks can steer UCB and ϵ- greedy toward a target arm, while Liu and Shroff [2019] developed an adaptive attack without knowledge of the learner’s algorithm. Zuo [2024] established near-optimal attacks against UCB and TS with matching lower bounds. Other work studies restricted intervention: Xu et al. [2021] proposed an observation-free attack during prescribed online phases, and Zeng et al. [2025] introduced bounded fake data injection with constraints on injection timing and frequency. A common strategy in this line of work is to favor the target by suppressing the observed rewards of non-target arms. Attacks have also been studied in contextual, linear, adversarial, and learning-to-rank bandits [Garcelon et al., 2020, Wang et al., 2022, Ma and Zhou, 2023, Zuo et al., 2023]. Beyond these works, Liu and Shroff [2019] formulated offline reward-poisoning attacks as convex optimization problems for several bandit algorithms. Hosseini et al. [2026] studied reward-model perturbations before training an offline bandit, without subsequent online interaction.

Robust Bandits. Lykouris et al. [2018] established corruption-dependent regret bounds for stochastic bandits without prior knowledge of the corruption level, and Gupta et al. [2019] improved these guarantees. Kapoor et al. [2019] used robust statistical estimation to tolerate corrupted rewards in multi-armed and linear contextual bandits. For linear bandits, robust phased elimination [Bogunovic et al., 2021] and confidence-weighted regression [He et al., 2022a] control the influence of corrupted observations, with guarantees that account for whether the corruption level is known. Robustness has also been studied in Gaussian process [Bogunovic et al., 2022], Lipschitz [Kang et al., 2023], and combinatorial bandits [Xu and Li, 2021]. The threat models considered in these works are typically weaker than those studied in the attack literature, since the adversary is not allowed to observe the learner’s actions before corrupting the feedback. Our work follows the attack-design direction and adopts the fake data injection model of Zeng et al. [2025] in an offline-to-online setting.

## 2 Warm-Start Bandit Setup

We consider a stochastic multi-armed bandit with arm set ${ \cal K } \triangleq \{ 1 , \ldots , K \}$ . Each arm $i \in \mathcal { K }$ is associated with an unknown reward distribution with mean $\mu _ { i }$ . Rewards are supported on $[ r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ]$ and are $\sigma ^ { 2 } { \mathrm { - s u b { \mathrm { - G a u s s i a n } } } }$ , where $\sigma ^ { 2 }$ is known to the learner.

The learner is initialized with an offline log $\mathcal { D } ^ { \mathrm { o f f } } \ \triangleq \ \big ( ( a _ { s } , r _ { s } ) \big ) _ { s = 1 } ^ { T _ { 0 } }$ containing $T _ { 0 }$ action–reward observations, where $a _ { s } \in \mathcal { K }$ denotes the arm selected in observation round $s ,$ , and $r _ { s } \in [ r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ]$ denotes the corresponding observed reward. For each arm $i ,$ let $\begin{array} { r } { N _ { i } ^ { \mathrm { o f f } } \triangleq \sum _ { s = 1 } ^ { T _ { 0 } } \mathbf { 1 } \{ a _ { s } = i \} } \end{array}$ denote its offline count. After initialization, the learner interacts with the bandit environment from round $T _ { 0 } +$ 1 through round $T .$ . Thus, the online rounds are $\mathcal { T } ^ { \mathrm { o n } } \triangleq \{ T _ { 0 } + 1 , \dots , T \}$ , with online horizon $H \triangleq | { \mathcal { T } } ^ { \mathrm { o n } } | = T - T _ { 0 }$

At the beginning of each online round $t \in \mathcal { T } ^ { \mathrm { o n } }$ , the learner’s history consists of the offline log and the preceding online observations:

$$
\mathcal { H } _ { t } \triangleq \Big ( \boldsymbol { \mathcal { D } } ^ { \mathrm { o f f } } , \big ( \big ( \boldsymbol { a } _ { s } , \boldsymbol { r } _ { s } \big ) \big ) _ { s = T _ { 0 } + 1 } ^ { t - 1 } \Big ) .
$$

A general warm-start bandit algorithm is specified by a policy sequence $\Pi \triangleq \{ \pi _ { t } \} _ { t \in \mathcal { T } ^ { \mathrm { o n } } }$ , where $\pi _ { t } ( \cdot \ | \ \mathcal { H } _ { t } ) \in \Delta ( \mathcal { K } )$ . In round t, the learner samples $a _ { t } \sim \pi _ { t } ( \cdot \mid \mathcal { H } _ { t } )$ , observes the corresponding reward $r _ { t } ,$ , and appends $\left( { { a _ { t } } , { r _ { t } } } \right)$ to its history. We write $\begin{array} { r } { N _ { i } ^ { \mathrm { o n } } ( t ) \ \triangleq \ \sum _ { s = T _ { 0 } + 1 } ^ { t - 1 } \mathbf { 1 } \{ a _ { s } = \ i \} } \end{array}$ for the number of online pulls of arm i before round t.

UCB Specialization. We next instantiate this general framework using the UCB algorithm considered by Jun et al. [2018], following the prototype in Bubeck and Cesa-Bianchi [2012, Section 2.2]. UCB summarizes the available history $\bar { \mathcal { H } } _ { t }$ through the number of observations and the empirical mean associated with each arm. For every arm $i \in \ K$ , define $\begin{array} { r } { N _ { i } ( t ) \ \triangleq \ \sum _ { s = 1 } ^ { t - 1 } { \bf 1 } \{ a _ { s } \ = \ i \} \ = \ } \end{array}$ $N _ { i } ^ { \mathrm { o f f } } + N _ { i } ^ { \mathrm { o n } } ( t )$ and $\begin{array} { r } { \hat { \mu } _ { i } ( t ) \triangleq \sum _ { s = 1 } ^ { t - 1 } \mathbf { 1 } \{ a _ { s } = i \} r _ { s } / N _ { i } ( t ) } \end{array}$ whenever $N _ { i } ( t ) > 0$ . The UCB index of arm i is

$$
\begin{array} { r } { u _ { i } ( t ) \triangleq \left\{ \begin{array} { l l } { \hat { \mu } _ { i } ( t ) + 3 \sigma \sqrt { \frac { \log { t } } { N _ { i } ( t ) } } , } & { N _ { i } ( t ) > 0 , } \\ { + \infty , } & { N _ { i } ( t ) = 0 . } \end{array} \right. } \end{array}
$$

The learner selects $a _ { t } \in \arg \operatorname* { m a x } _ { i \in \mathcal { K } } u _ { i } ( t )$ . In the absence of attacks, it achieves sublinear regret.

## 3 Offline Attack Model

## 3.1 Attack Setting and Attack Cost

Let $\begin{array} { r } { \mathcal { D } ^ { \mathrm { c l } } \triangleq \big ( ( a _ { s } , r _ { s } ) \big ) _ { s = 1 } ^ { T _ { 0 } ^ { \mathrm { c l } } } } \end{array}$ denote the clean offline log, where $T _ { 0 } ^ { \mathrm { c l } } = | \mathcal { D } ^ { \mathrm { c l } } |$ is the number of genuine observations. For each arm, its clean-log and subsequent online rewards form a prefix of the same i.i.d. reward stream. Arm choices may depend on past observations; each chosen arm’s next reward is then observed and retained, with no filtering based on its value. Before deployment, the attacker observes $\mathcal { D } ^ { \mathrm { c l } }$ and knows the learner’s policy Π, the terminal time $T ,$ , and the valid reward range $[ r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ]$ . The attacker acts only at this stage and cannot modify the learner’s actions, observed rewards, or history after online interaction begins.

For each arm $i ,$ define its clean offline count and empirical mean as

$$
N _ { i } ^ { \mathrm { c l } } \triangleq \sum _ { s = 1 } ^ { T _ { 0 } ^ { \mathrm { c l } } } \mathbf { 1 } \{ a _ { s } = i \} , \qquad \hat { \mu } _ { i } ^ { \mathrm { c l } } \triangleq \frac { 1 } { N _ { i } ^ { \mathrm { c l } } } \sum _ { s = 1 } ^ { T _ { 0 } ^ { \mathrm { c l } } } \mathbf { 1 } \{ a _ { s } = i \} r _ { s } ,
$$

where the empirical mean is defined whenever $N _ { i } ^ { \mathrm { c l } } > 0$ . During online interaction, $N _ { i } ^ { \mathrm { c l } } ( t )$ and $\hat { \mu } _ { i } ^ { \mathrm { c l } } ( t )$ denote the count and empirical mean of all genuine observations of arm i available before round t, including the clean offline observations but excluding injected samples. Thus, at the start of online interaction, $N _ { i } ^ { \mathrm { c l } } ( T _ { 0 } + 1 ) = N _ { i } ^ { \mathrm { c l } }$ and $\hat { \mu } _ { i } ^ { \mathrm { c l } } ( T _ { 0 } + 1 ) = \check { \mu } _ { i } ^ { \mathrm { c l } }$

A candidate injection plan is represented by $( \mathbf { n } , \mathbf { y } )$ , where n $\triangleq ( n _ { 1 } , . . . , n _ { K } ) \in \mathbb { Z } _ { \geq 0 } ^ { K }$ specifies the number of fake observations injected for each arm and $\mathbf { y } \triangleq ( y _ { i , j } : i \in \mathcal { K } , j = 1 , \dots , n _ { i } )$ specifies the corresponding fake rewards. Every injected reward must lie in $[ r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ]$ and, for discrete feedback, in the legal feedback alphabet.

Let $\mathcal { D } ^ { \mathrm { F } } ( \mathbf { n } , \mathbf { y } )$ denote the collection of fake action–reward pairs $\{ ( i , y _ { i , j } ) : i \in K , j = 1 , \ldots , n _ { i } \}$ in the order chosen by the strategy. The learner is initialized using the attacked warm-start log

$$
\begin{array} { r } { \mathcal { D } ^ { \mathrm { o f f } } ( { \bf n } , { \bf y } ) \triangleq \mathcal { D } ^ { \mathrm { c l } } \parallel \mathcal { D } ^ { \mathrm { F } } ( { \bf n } , { \bf y } ) , } \end{array}
$$

where ∥ denotes sequence concatenation.

Definition 3.1 (Attack Cost). The attack cost is the number of injected samples:

$$
{ \mathcal { C } } ( { \bf n } ) \triangleq \sum _ { i = 1 } ^ { K } n _ { i } .\tag{1}
$$

$$
N _ { i } ^ { \mathrm { o f f } } = N _ { i } ^ { \mathrm { c l } } + n _ { i }
$$

$$
T _ { 0 } = \left| \mathcal { D } ^ { \mathrm { o f f } } \right| = T _ { 0 } ^ { \mathrm { c l } } + \mathcal { C } ( \mathbf { n } ) , \qquad H = T - T _ { 0 } .
$$

Without loss of generality, we label the target arm as $K$

Definition 3.2 $( ( \xi , \delta )$ -Success Criterion). Fix $\delta \in ( 0 , 1 )$ and $\xi \colon  { \mathbb { N } } \to  { \mathbb { R } } _ { \geq 0 }$ . An offline attack is (ξ, δ)-successful if

$$
\operatorname* { P r } \left( \sum _ { i = 1 } ^ { K - 1 } N _ { i } ^ { \mathrm { o n } } ( T + 1 ) \leq \xi \right) \geq 1 - \delta \quad \mathrm { a n d } \quad \mathcal { C } ( \mathbf { n } ) = o ( T ) .\tag{2}
$$

We summarize the offline injection workflow in Algorithm 1; concrete allocation strategies are developed in the next section.

Algorithm 1: Offline Injection Attack   
Input: $\mathcal { D } ^ { \mathrm { c l } } , K , \Pi , T , [ r _ { \mathrm { m i n } } , r _ { \mathrm { m a x } } ] , \xi$ and δ, Target Arm (K in our analysis)   
Output: Attacked warm-start log $\mathcal { D } ^ { \mathrm { o f f } }$   
1 Determine an injection plan $( \mathbf { n } , \mathbf { y } )$ satisfying Eq. (2);   
2 Construct a fake log $\mathcal { D } ^ { \mathrm { F } } ( \mathbf { n } , \mathbf { y } )$ containing $( i , y _ { i , j } )$ for every $i \in \mathcal { K }$ and $j = 1 , \ldots , n _ { i } ;$   
3 Set ${ \mathcal { D } } ^ { \mathrm { o f f } }  { \mathcal { D } } ^ { \mathrm { c l } } \parallel { \mathcal { D } } ^ { \mathrm { F } } ( \mathbf { n } , \mathbf { y } )$ ;   
4 return ${ \mathcal { D } } ^ { \mathrm { o f f } } ;$

## 4 Attack Strategies

We first instantiate Algorithm 1 for UCB, then extend the strategy to suppression-bounded algorithms from Zeng et al. [2025].

## 4.1 Attack on UCB

We consider the zero-tolerance case $\xi = 0$ , where the attacker aims to make UCB select the target arm in every online round. The high-level idea is to (1) enlarge the gap between the target arm and the non-target arms, and (2) make the exploration bonus sufficiently small for each non-target arm, so that $u _ { K } ( t ) > u _ { i } ( t )$ throughout the online phase. To ensure this, we seek to solve

$$
\operatorname* { m i n } _ { \mathbf { n } , \mathbf { y } } \quad \mathcal { C } ( \mathbf { n } ) \quad \mathrm { s u b j e c t ~ t o } \quad u _ { K } ( t ) > u _ { i } ( t ) , \qquad \forall i < K , t \in \mathcal { T } ^ { \mathrm { o n } } .
$$

Let $R \triangleq r _ { \operatorname* { m a x } } - r _ { \operatorname* { m i n } }$ and assume $N _ { i } ^ { \mathrm { c l } } \geq 1$ for every arm. Write $b _ { i } ( t ) = 3 \sigma \sqrt { \log t / N _ { i } ( t ) }$ . Our construction keeps the target arm’s empirical mean above a threshold z and bounds each non-target arm’s UCB index by $z + \bar { b } _ { K } ( t )$ . Specifically, we enforce

$$
u _ { i } ( t ) = \hat { \mu } _ { i } ( t ) + b _ { i } ( t ) \leq z + b _ { K } ( t ) < \hat { \mu } _ { K } ( t ) + b _ { K } ( t ) = u _ { K } ( t ) , \qquad \forall i < K , t \in \mathcal { T } ^ { \mathrm { o n } } .
$$

For a confidence parameter $\delta \in ( 0 , 1 / 2 )$ , define

$$
\beta _ { \delta } ( N ) \triangleq \sqrt { \frac { 2 \sigma ^ { 2 } } { N } \log \frac { \pi ^ { 2 } K N ^ { 2 } } { 3 \delta } } , \qquad \underline { { \mu } } _ { K } \triangleq \operatorname* { m a x } \{ r _ { \operatorname* { m i n } } , \hat { \mu } _ { K } ^ { \mathrm { c l } } - 2 \beta _ { \delta } ( N _ { K } ^ { \mathrm { c l } } ) \} .
$$

Set $\varepsilon \triangleq \underline { { \mu } } _ { K } - r _ { \operatorname* { m i n } }$ . We choose

$$
z _ { T } ^ { \star } = r _ { \mathrm { m i n } } + \bigg ( \frac { 1 8 R \sigma ^ { 2 } ( K - 1 ) \log T } { T } \bigg ) ^ { 1 / 3 } , \qquad z = r _ { \mathrm { m i n } } + \operatorname* { m a x } \left\{ z _ { T } ^ { \star } - r _ { \mathrm { m i n } } , \operatorname* { m i n } \{ \varepsilon , R / 2 \} \right\} .
$$

Write $B _ { i } = N _ { i } ^ { \mathrm { c l } } ( \hat { \mu } _ { i } ^ { \mathrm { c l } } - r _ { \mathrm { m i n } } )$ and $d ( z ) = z - r _ { \operatorname* { m i n } } +$ 3σ $\sqrt { \log T / T }$ . We set

$$
n _ { i } = \left\{ \begin{array} { l l } { \displaystyle \left[ \frac { T ( z - \underline { { \mu } } _ { K } ) + } { r _ { \operatorname* { m a x } } - z } \right] + 1 , \quad \mathrm { f o r } i = K ; } \\ { \displaystyle \left[ \left[ \left( \frac { 3 \sigma \sqrt { \log T } + \sqrt { 9 \sigma ^ { 2 } \log T + 4 d ( z ) B _ { i } } } { 2 d ( z ) } \right) ^ { 2 } - N _ { i } ^ { \mathrm { c l } } \right] _ { + } \right] , \quad \mathrm { f o r } i < K . } \end{array} \right.\tag{3}
$$

Using these counts, construct the fake log in Algorithm 1 as

$$
\mathcal { D } ^ { \mathrm { F } } = \left( \Big \Vert _ { i = 1 } ^ { K - 1 } \big ( ( i , r _ { \operatorname* { m i n } } ) \big ) _ { j = 1 } ^ { n _ { i } } \right) \Big \Vert \big ( ( K , r _ { \operatorname* { m a x } } ) \big ) _ { j = 1 } ^ { n _ { K } } .
$$

We analyze this construction under the following scaling assumptions.

Assumption 4.1. $R = \Theta ( 1 ) , \sigma = \Theta ( 1 )$ , and $N _ { i } ^ { \mathrm { c l } } = \Theta ( 1 )$ uniformly over $i \in \mathcal { K }$

In particular, $T _ { 0 } ^ { \mathrm { c l } } = \Theta ( K ) = o ( T )$ , so every attack satisfying ${ \mathcal { C } } ( { \bf n } ) = o ( T )$ has $H = T ( 1 - o ( 1 ) )$ Define $S _ { T } \triangleq { T ^ { 2 / 3 } ( K \log T ) ^ { 1 / 3 } }$ . We now give the theoretical guarantee of our attack strategy.

Theorem 4.2. Fix $\delta \in ( 0 , 1 / 2 )$ . Under Assumption 4.1,for all sufficiently large T, the above attack satisfies

$$
\operatorname* { P r } \left( \sum _ { i = 1 } ^ { K - 1 } N _ { i } ^ { \mathrm { o n } } ( T + 1 ) = 0 \right) \geq 1 - \delta \quad a n d \quad \mathcal { C } ( \mathbf { n } ) = \mathcal { O } \left( \frac { K \log T } { ( \varepsilon \vee ( S _ { T } / T ) ) ^ { 2 } } \right) .
$$

ProofSketch. If $\varepsilon \leq z _ { T } ^ { \star } - r _ { \operatorname* { m i n } } ,$ then $z = z _ { T } ^ { \star } ;$ ; substituting into Eq. (3) gives $n _ { K } = O ( S _ { T } )$ and $\begin{array} { r } { \sum _ { i < K } n _ { i } = O ( S _ { T } ) } \end{array}$ . If $\varepsilon > z _ { T } ^ { \star } - r _ { \operatorname* { m i n } }$ , then $z \le { \underline { { \mu } } } _ { K }$ and $z - r _ { \operatorname* { m i n } } = \Theta ( \varepsilon )$ , giving $n _ { K } = 1$ and $\textstyle \sum _ { i < K } n _ { i } = O ( K \log T / \varepsilon ^ { 2 } )$ . Combining the two cases yields the stated cost bound. □

Theorem 4.3. Under Assumption 4.1, fix $\delta \in ( 0 , 1 / 2 )$ . Let $\Delta _ { K } \triangleq \mu _ { K } - r _ { \operatorname* { m i n } }$ . For all sufficiently large T, every (0, δ)-successful bounded offline attack on UCB satisfies

$$
\mathcal { C } ( { \bf n } ) = \Omega \left( \frac { K \log T } { \left( \Delta _ { K } \vee \left( S _ { T } / T \right) \right) ^ { 2 } } \right)
$$

with probability at least $1 - \delta - 1 / T _ { ☉ }$

ProofSketch. Consider the final round $T ,$ when UCB selects the target on the zero-tolerance success event. By then, target injections have been diluted by Θ(T) genuine observations. Writing $u _ { K } ( T ) =$ $r _ { \operatorname* { m i n } } + \rho ,$ we have $\rho \le \Delta _ { K } + O ( n _ { K } / T + \sqrt { \log T / T } )$ with high probability. Since the target is selected at $T ,$ , every competitor satisfies $u _ { i } ( T ) \leq u _ { K } ( T )$ . Its empirical mean is at least $r _ { \mathrm { m i n } } ,$ , so its exploration bonus cannot exceed $\rho ,$ forcing $N _ { i } ^ { \mathrm { c l } } + n _ { i } \ge 9 \sigma ^ { 2 } \log T / \rho ^ { 2 }$ . Combining these constraints on target and non-target injection costs and minimizing over $\rho$ yields the cost lower bound. □

The upper and lower bounds depend on $\varepsilon$ and $\Delta _ { K }$ , respectively. Their gap-dependent rates match when $\varepsilon = \Omega ( \Delta _ { K } )$ . When $\varepsilon = \Omega ( 1 )$ , our upper and lower bounds match at $\Theta \bar { ( K \log T ) }$ . This cost order also agrees with the attack costs of Zeng et al. [2025].

Our bounds identify $S _ { T } / T$ as the boundary scale for the target arm’s gap from the lower reward bound $r _ { \mathrm { m i n } }$ . We refer to the regime $\Delta _ { K } = o \dot { ( } S _ { T } / T )$ as the near-boundary regime. In this regime, we further sharpen the bounds to match in their leading constants, thereby establishing the asymptotic optimality of our construction. The optimal leading cost is obtained by minimizing the sum of the leading costs of target promotion and non-target suppression. This minimization yields the closedform value

$$
\Lambda _ { T } \triangleq 3 \bigg ( \frac { 9 \sigma ^ { 2 } ( K - 1 ) T ^ { 2 } \log T } { 4 R ^ { 2 } } \bigg ) ^ { 1 / 3 } = \Theta ( S _ { T } ) .
$$

Corollary 4.4. Under Assumption 4.1, fix $\delta \in ( 0 , 1 / 2 )$ and assume $\Delta _ { K } = o ( \Lambda _ { T } / T )$ . For all sufficiently large T, our construction has $\begin{array} { r } { \mathcal { C } ( { \bf n } ) \leq ( 1 + o ( 1 ) ) \Lambda _ { T } . } \end{array}$ . And every $( 0 , \delta )$ -successful bounded offline attack on UCB satisfies

$$
\mathrm { P r } \left[ \mathcal { C } ( { \bf n } ) \geq ( 1 - o ( 1 ) ) \Lambda _ { T } \right] \geq 1 - \delta - \frac { 1 } { T } .
$$

We next characterize the cost allocated to the target arm and show that this allocation is necessary for asymptotic optimality.

Lemma 4.5. Under Assumption 4.1, fix $\delta \in ( 0 , 1 / 2 )$ . For $\varepsilon = o ( \Lambda _ { T } / T )$ , our construction satisfies $n _ { K } = \left( { \textstyle { \frac { 2 } { 3 } } } + o ( 1 ) \right) \Lambda _ { T }$ . ${ \cal I } f \Delta _ { \cal K } = o ( \Lambda _ { \cal T } / T )$ , then for all sufficiently large T, every $( o ( \Lambda _ { T } ) , \delta )$ successful bounded offline attack on UCB with $\mathcal { C } ( { \bf n } ) \leq ( 1 + o ( 1 ) ) \Lambda _ { T }$ satisfies

$$
\mathrm { P r } \bigg [ n _ { K } = \left( \frac { 2 } { 3 } + o ( 1 ) \right) \Lambda _ { T } \bigg ] \geq 1 - \delta - \frac { 1 } { T } .
$$

Remark 4.6. The $2 : 1$ cost split balances target promotion against non-target suppression. In the near-boundary regime, raising the target’s empirical mean has a leading cost linear in its elevation above the lower reward bound. UCB’s $N _ { i } ^ { - 1 / 2 }$ exploration bonuses make each competitor’s required injection count, and hence their sum, decrease as the inverse square of that elevation. Minimizing the sum of these leading cost terms yields the 2 : 1 split between target and aggregate non-target costs. Appendix E compares this mechanism across algorithms.

For larger tolerances $\xi = O ( \Lambda _ { T } )$ , Lemma A.2 establishes that near-boundary successful attacks with a deterministic $O ( \Lambda _ { T } )$ cost bound must still incur a nonvanishing fraction of their total cost on the target arm. Proofs of Theorems 4.2 and 4.3 appear in Appendices A.1 and A.2, respectively; those of Corollary 4.4 and Lemma 4.5 appear in Appendix A.3.

In the near-boundary regime, our construction is target-centric: for any fixed number of competitors, injections on the target account for most of the total cost with high probability, as required for leading-order optimality. This emphasis on favorable feedback for the target qualitatively reflects real-world manipulation practices, such as sellers purchasing positive reviews for their own products [He et al., 2022b].

Comparison to Xu et al. [2021]. In the near-boundary regime, the prescribed cost of Xu et al. [2021]’s UCB-based attack scales as $\Theta ( K$ log $T / \Delta _ { K } ^ { 2 } )$ . Consequently, the cost is $\Omega ( S _ { T } )$ when $\Delta _ { K } ~ = ~ \mathcal { O } ( S _ { T } / T )$ , ceases to be sublinear when $\Delta _ { K } ~ = ~ \mathcal { O } ( \sqrt { K \log T / T } )$ , and is undefined at $\Delta _ { K } = 0$ . In contrast, our attack remains applicable and covers this boundary case.

## 4.2 Extension to Suppression-Bounded Algorithms

We extend the strategy to algorithms whose additional pulls of persistently dominated arms admit a high-probability bound.

Definition 4.7 (Suppression-Bounded Algorithms from Zeng et al. [2025]). Fix a horizon $T$ and a confidence level $\delta \in ( 0 , 1 )$ . A bandit algorithm A is said to be suppression-bounded if, for any arm i, there exists a deterministic bound $B _ { i } ^ { \mathbf { \bar { \mathcal { A } } } } ( t , T , \delta )$ such that the following implication holds with probability at least $1 - \delta \colon \mathrm { i f }$

$$
{ \hat { \mu } } _ { i } ( s ) < \operatorname* { m a x } _ { j \in { \cal K } } { \hat { \mu } } _ { j } ( s ) , \qquad \mathrm { f o r ~ a l l } ~ s \in \{ t + 1 , \ldots , T \} ,
$$

then

$$
N _ { i } ( T ) - N _ { i } ( t ) \leq B _ { i } ^ { \mathcal { A } } ( t , T , \delta ) .
$$

The quantity $B _ { i } ^ { \mathcal { A } } ( t , T , \delta )$ is referred to as the post-suppression pull bound for arm i. This is a conservative upper bound and $B _ { i } ^ { \mathcal { A } } ( t , T , \delta ) \geq B _ { i } ^ { \mathcal { A } } ( t , T ^ { \prime } , \delta )$ for all $T ^ { \prime } < T$

Theorem 4.8 (Generic Suppression-Bounded Attack). Suppose A is suppression-bounded and write $B _ { i } ^ { \mathcal { A } } \triangleq B _ { i } ^ { \mathcal { A } } ( T _ { 0 } ^ { \mathrm { c l } } , T , \delta / K )$ . Assume $N _ { i } ^ { \mathrm { c l } } \geq 1$ for every arm. There exists a bounded offline attack that satisfies:

$$
\operatorname* { P r } \left( \sum _ { i = 1 } ^ { K - 1 } N _ { i } ^ { \mathrm { o n } } ( T + 1 ) \le \sum _ { i = 1 } ^ { K - 1 } B _ { i } ^ { A } \right) \ge 1 - 2 \delta
$$

and

$$
\mathcal { C } ( \mathbf { n } ) \leq \mathcal { O } \left( \operatorname* { i n f } _ { \rho \in ( 0 , R ) } \left\{ K + T \frac { \left( \rho - \varepsilon \right) _ { + } } { R - \rho } + \frac { R - \rho } { \rho } \sum _ { i = 1 } ^ { K - 1 } \left( N _ { i } ^ { \mathrm { c l } } + B _ { i } ^ { A } \right) \right\} \right) .
$$

The construction raises the target’s empirical mean above $r _ { \operatorname* { m i n } } + \rho$ and keeps each non-target mean at or below this threshold. The target injections counteract dilution by genuine feedback. The nontarget injections offset the clean observations and up to $B _ { i } ^ { \mathcal { A } }$ online observations, even if each has

Clipped Suppression $N _ { K } / T$  
![](images/d9ddec296cbb52340d82a844ba9085d76a34b5916d2f421fb6cc9b6b9dee3ae1.jpg)  
(a) Attack Cost vs. Horizon T  
Clipped Suppression

![](images/b1f0e12507b0aa5f629310ffaedbcef57fb768198e268eba0eefc022f60c4639.jpg)  
(b) Attack Cost vs. Number of Arms K  
Total Attack Cost

Figure 1: Attack Cost for UCB.  
![](images/43518ea538af225bae37661cd24a9d0d7eb0fd3e0ed6d8c3a8e334098c40f9d6.jpg)  
(a) Target-arm Selection Ratio

![](images/14294826724e0e6562c709771a5cebaea8ad34e145ff24a60d935980682756fa.jpg)  
(b) Target-arm Selection Ratio  
vs. Number of Arms K  
Figure 2: Target-arm Selection Ratio for UCB.

reward $r _ { \mathrm { m a x } }$ . Each such observation requires approximately $( R - \rho ) / \rho$ compensating injections at $r _ { \mathrm { m i n } }$

Under Assumption 4.1, if $\begin{array} { r } { \sum _ { i < K } B _ { i } ^ { \mathcal { A } } = o ( T ) } \end{array}$ , let $\begin{array} { r } { A _ { T } \triangleq \sum _ { i < { K } } ( N _ { i } ^ { \mathrm { c l } } + B _ { i } ^ {  { \mathcal { A } } } ) = o ( T ) } \end{array}$ . Choosing $\rho = R \sqrt { A _ { T } / T } / ( 1 + \sqrt { A _ { T } / T } )$ and using $\varepsilon \geq 0$ gives

$$
\begin{array} { r } { \mathcal { C } ( { \bf n } ) = O \left( K + \sqrt { T A _ { T } } \right) = o ( T ) . } \end{array}
$$

Consequently, $H = T ( 1 - o ( 1 ) )$ and $N _ { K } ^ { \mathrm { o n } } ( T + 1 ) / H = 1 - o ( 1 )$ with probability at least $1 - 2 \delta$ yielding a successful attack. UCB, TS, and ϵ-greedy all fall into the scope of suppression-bounded algorithms [Zeng et al., 2025].

## 5 Numerical Illustration

We evaluate our attacks on MovieLens-25M [Harper and Konstan, 2015], mapping ratings of at least four to one and the rest to zero. Each movie defines a Bernoulli arm with its empirical positive-rating rate as the mean; clean and online rewards are sampled independently with replacement. The target, Glitter (ID 4775), has the smallest positive mean, $\mu _ { K } = 1 1 / 6 6 9$ ; competitors are the most-rated remaining movies. We use $N _ { i } ^ { \mathrm { c l } } = 5 , \sigma = 0 . 5$ , and $\delta = 0 . 0 5$

Figure 1 compares our attack strategy with Clipped Suppression by varying T at $K = 1 0$ and K at $\bar { T } = 1 0 ^ { 6 }$ . Our method consistently incurs a substantially lower attack cost. As T increases, the observed cost growth is consistent with the sublinear bound in Theorem 4.2. In contrast, Clipped Suppression intervenes whenever a non-target arm is pulled, so its attack cost is $T - N _ { K }$ and grows faster with T. The cost gap also increases as K grows.

Figure 2 further shows that our method achieves lower attack cost while maintaining success. For our method, $N _ { K } ^ { \mathrm { o n } } / H = 1$ in every repeat across all tested values of T and $K ,$ , meaning that UCB selects the target arm in every online round after deployment. The overall ratio $N _ { K } / T$ , which also includes the warm-start samples, remains close to 1. In contrast, Clipped Suppression yields more non-target pulls. Even at $T = 1 0 ^ { 6 }$ , the learner selects non-target arms in nearly half of the rounds, and its target-arm selection ratio decreases further as K increases.

![](images/9f6bb6ade11ef9dcf974a5333418676b600c9a026cad4a0072f986a90017a76f.jpg)  
K = 5 K = 10 K = 15 K = 20 K = 25

![](images/417afbbf4c50f5e095da8bf3e94bf9a862284c5ef87798f290bb7999957072d3.jpg)  
Target n<sub>K</sub> /C Non-target $\scriptstyle \sum _ { i < K } n _ { i } / { \mathcal { C } }$  
(b) Cost Share vs. Horizon T

(a) Normalized Attack Cost vs. Horizon T  
Figure 3: Near-Boundary Cost and Allocation for UCB.  
![](images/9dbd6cfd656df70b0c1e21c7e96ccbf939ac06c3af96a3d5d804a4d432ed3fde.jpg)  
(a) Attack Cost vs. $\Delta _ { K } / ( S _ { T } / T )$ Ours Xu (2021)

![](images/1c43b0308ade28de8abca81119e12defec184677a9ac5d9a1fc4b1df15b4bbe8.jpg)  
(b) Target-arm Selection Ratio vs. $\Delta _ { K } / ( S _ { T } / T )$ Ours N<sub>K</sub>/T Ours N<sup>on</sup>/H Xu (2021) N<sub>K</sub>/T  
Figure 4: Comparison with Xu et al. [2021] for UCB.

We next illustrate the leading-order optimality and target-centric allocation predicted by our theory. As shown in Figure 3(a), for each fixed K, the normalized attack cost ${ \mathcal { C } } ( { \bf n } ) / \Lambda _ { T }$ approaches 1 as T increases, consistent with Corollary 4.4. In Figure 3(b), at $K = 1 0 .$ , the target-arm cost share $n _ { K } / { \mathcal { C } } ( { \bf n } )$ approaches $2 / 3 ,$ consistent with Lemma 4.5.

We then compare our method with the attack of Xu et al. [2021] under the same experimental setting. When $\Delta _ { K } / ( \dot { S } _ { T } / T ) < 1$ , the target lies within the boundary scale $S _ { T } / T$ , where attacks become challenging. Our attack maintains sublinear cost in this regime and makes UCB select the target arm in every online round after deployment with high probability. In contrast, the attack cost of Xu et al. [2021] scales as $O ( K \log T / \bar { \Delta } _ { K } ^ { 2 ^ { * } } )$ and grows rapidly as $\Delta _ { K }$ approaches zero. Hence, their points are omitted when this cost is undefined at $\Delta _ { K } = 0$ or is at least $\mathbf { \bar { \rho } } _ { T } ,$ while their target-arm selection ratio remains below 1 in the most challenging near-boundary settings.

These experiments demonstrate the effectiveness of our method, with cost shares consistent with the theoretical analysis. We also provide experiments for TS and ϵ-greedy in Appendix B.2 and Appendix C.2, respectively.

## 6 Concluding Remarks

In the near-boundary regime, target-centric allocation is necessary for successful UCB attacks with tolerance $\xi = O ( S _ { T } )$ and a deterministic $O ( S _ { T } )$ cost cap: the target must receive a nonvanishing share of the injections on the high-probability event of our lower bound. Our UCB construction attains this cost scale in the near-boundary regime. We also give constructions for TS and ϵ-greedy in the Appendix, while the suppression-bounded analysis covers a broader class of learners.

However, our analysis assumes a learner that uncritically trusts all offline data. A possible direction for future work is to investigate whether discounting, filtering, or auditing offline observations can mitigate the effects of favorable target injections, and how such defenses trade off the benefits of a clean warm start against robustness to corrupted initialization. Furthermore, our cost and allocation results assume a small clean offline log. More clean observations make empirical means harder to shift but also reduce exploration uncertainty, potentially changing both the optimal attack cost and its allocation. Characterizing how these competing effects depend on the size and distribution of clean observations across arms is an interesting direction for future work.

## References

Shipra Agrawal and Navin Goyal. Near-optimal regret bounds for thompson sampling. Journal of the ACM (JACM), 64(5):1–24, 2017.

Siddhartha Banerjee, Sean R Sinclair, Milind Tambe, Lily Xu, and Christina Lee Yu. Artificial replay: a meta-algorithm for harnessing historical data in bandits. arXiv preprint arXiv:2210.00025, 2022.

Ilija Bogunovic, Arpan Losalka, Andreas Krause, and Jonathan Scarlett. Stochastic linear bandits robust to adversarial attacks. In International Conference on Artificial Intelligence and Statistics, pages 991–999. PMLR, 2021.

Ilija Bogunovic, Zihan Li, Andreas Krause, and Jonathan Scarlett. A robust phased elimination algorithm for corruption-tolerant gaussian process bandits. Advances in Neural Information Processing Systems, 35: 23951–23964, 2022.

Sébastien Bubeck and Nicolo Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends® in Machine Learning, 5(1):1–122, 2012.

Evrard Garcelon, Baptiste Roziere, Laurent Meunier, Jean Tarbouriech, Olivier Teytaud, Alessandro Lazaric, and Matteo Pirotta. Adversarial attacks on linear contextual bandits. Advances in Neural Information Processing Systems, 33:14362–14373, 2020.

Anupam Gupta, Tomer Koren, and Kunal Talwar. Better algorithms for stochastic bandits with adversarial corruptions. In Conference on Learning Theory, pages 1562–1578. PMLR, 2019.

F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4):1–19, 2015.

Jiafan He, Dongruo Zhou, Tong Zhang, and Quanquan Gu. Nearly optimal algorithms for linear contextual bandits with adversarial corruptions. Advances in neural information processing systems, 35:34614–34625, 2022a.

Sherry He, Brett Hollenbeck, Gijs Overgoor, Davide Proserpio, and Ali Tosyali. Detecting fake-review buyers using network structure: Direct evidence from amazon. Proceedings of the National Academy of Sciences, 119(47):e2211932119, 2022b. doi: 10.1073/pnas.2211932119.

Seyed Mohammad Hadi Hosseini, Amir Najafi, and Mahdieh Baghshah. Efficient adversarial attacks on highdimensional offline bandits. In International Conference on Learning Representations, volume 2026, pages 37587–37612, 2026.

Kwang-Sung Jun, Lihong Li, Yuzhe Ma, and Jerry Zhu. Adversarial attacks on stochastic bandits. Advances in neural information processing systems, 31, 2018.

Yue Kang, Cho-Jui Hsieh, and Thomas Chun Man Lee. Robust lipschitz bandits to adversarial corruptions. Advances in Neural Information Processing Systems, 36:10897–10908, 2023.

Sayash Kapoor, Kumar Kshitij Patel, and Purushottam Kar. Corruption-tolerant bandit learning. Machine Learning, 108(4):687–715, 2019.

Ye Li, Hong Xie, Yishi Lin, and John CS Lui. Unifying offline causal inference and online bandit learning for data driven decision. In Proceedings of the Web Conference 2021, pages 2291–2303, 2021.

Fang Liu and Ness Shroff. Data poisoning attacks on stochastic bandits. In International Conference on Machine Learning, pages 4042–4050. PMLR, 2019.

Thodoris Lykouris, Vahab Mirrokni, and Renato Paes Leme. Stochastic bandits robust to adversarial corruptions. In Proceedings ofthe 50th annual ACM SIGACT symposium on theory ofcomputing, pages 114–122, 2018.

Yuzhe Ma and Zhijin Zhou. Adversarial attacks on adversarial bandits. arXiv preprint arXiv:2301.12595, 2023.

Bastian Oetomo, R Malinga Perera, Renata Borovica-Gajic, and Benjamin IP Rubinstein. Cutting to the chase with warm-start contextual bandits. Knowledge and Information Systems, 65(9):3533–3565, 2023.

Pannagadatta Shivaswamy and Thorsten Joachims. Multi-armed bandit problems with history. In Artificial intelligence and statistics, pages 1046–1054. PMLR, 2012.

Huazheng Wang, Haifeng Xu, and Hongning Wang. When are linear stochastic bandits attackable? In International Conference on Machine Learning, pages 23254–23273. PMLR, 2022.

Haike Xu and Jian Li. Simple combinatorial algorithms for combinatorial bandits: Corruptions and approximations. In Uncertainty in Artificial Intelligence, pages 1444–1454. PMLR, 2021.

Yinglun Xu, Bhuvesh Kumar, and Jacob D Abernethy. Observation-free attacks on stochastic bandits. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 22550–22561. Curran Associates, Inc., 2021.

Qirun Zeng, Eric He, Richard Hoffmann, Xuchuang Wang, and Jinhang Zuo. Practical adversarial attacks on stochastic bandits via fake data injection. arXiv preprint arXiv:2505.21938, 2025.

Jinhang Zuo, Zhiyao Zhang, Zhiyong Wang, Shuai Li, Mohammad Hajiesmaili, and Adam Wierman. Adversarial attacks on online learning to rank with click feedback. Advances in Neural Information Processing Systems, 36:41675–41692, 2023.

Shiliang Zuo. Near optimal adversarial attacks on stochastic bandits and defenses with smoothed responses. In International Conference on Artificial Intelligence and Statistics, pages 2098–2106. PMLR, 2024.

## Contents

1 Introduction 1   
1.1 Our Contributions . 2   
1.2 Related Work 3   
Warm-Start Bandit Setup 3   
3 Offline Attack Model 4   
3.1 Attack Setting and Attack Cost . 4   
4 Attack Strategies 5   
4.1 Attack on UCB 5   
4.2 Extension to Suppression-Bounded Algorithms . 7   
5 Numerical Illustration 8   
6 Concluding Remarks 9   
Appendix 12   
A Proofs for UCB 12   
A.1 Upper Bound Proof 13   
A.2 Lower Bound Proof 13   
A.3 Optimality Proofs . 14   
B Analysis for Thompson Sampling 16   
B.1 Attack Strategy 16   
B.2 Numerical Illustration . 17   
C Analysis for ϵ-Greedy 18   
C.1 Attack Strategy 18   
C.2 Numerical Illustration . 19   
D Analysis of Suppression-Bounded Algorithms 20   
E Exploration and Allocation 20

## A Proofs for UCB

For $\delta \in ( 0 , 1 )$ , recall that

$$
\beta _ { \delta } ( N ) = \sqrt { \frac { 2 \sigma ^ { 2 } } N \log \frac { \pi ^ { 2 } K N ^ { 2 } } { 3 \delta } } .
$$

Define the event

$$
\mathcal { E } _ { \delta } \triangleq \left\{ \forall i \in \mathcal { K } , \forall t \in \mathcal { T } ^ { \mathrm { o n } } , \quad \left. \hat { \mu } _ { i } ^ { \mathrm { c l } } ( t ) - \mu _ { i } \right. < \beta _ { \delta } ( N _ { i } ^ { \mathrm { c l } } ( t ) ) \right\} .
$$

By Jun et al. [2018, Lemma 1], $\mathrm { P r } ( \mathcal { E } _ { \delta } ) \geq 1 - \delta .$ . The radius is decreasing on the positive integers when $K \geq 2$ and $\delta \in ( 0 , 1 )$ . On ${ \mathcal { E } } _ { \delta }$ , we have $0 \le \varepsilon \le \Delta _ { K }$ , and

$$
\begin{array} { r l } & { \hat { \mu } _ { K } ^ { \mathrm { c l } } ( t ) \geq \operatorname* { m a x } \{ r _ { \operatorname* { m i n } } , \mu _ { K } - \beta _ { \delta } ( N _ { K } ^ { \mathrm { c l } } ( t ) ) \} } \\ & { \qquad \geq \operatorname* { m a x } \{ r _ { \operatorname* { m i n } } , \hat { \mu } _ { K } ^ { \mathrm { c l } } - 2 \beta _ { \delta } ( N _ { K } ^ { \mathrm { c l } } ) \} = \underline { { \mu } } _ { K } \qquad ( \forall t \in \mathcal { T } ^ { \mathrm { o n } } ) . } \end{array}
$$

## A.1 Upper Bound Proof

Under the event ${ \mathcal { E } } _ { \delta }$ and by Eq. (3), at round $t = T _ { 0 } + 1$ , we have

$$
\hat { \mu } _ { K } ( t ) \geq \frac { n _ { K } r _ { \operatorname* { m a x } } + N _ { K } ^ { \mathrm { c l } } ( t ) \underline { { \mu } } _ { K } } { n _ { K } + N _ { K } ^ { \mathrm { c l } } ( t ) } \geq \frac { n _ { K } r _ { \operatorname* { m a x } } + T \underline { { \mu } } _ { K } } { n _ { K } + T } > z .
$$

Since $N _ { K } ( t ) \leq t - 1$ and $\log { t } / ( t - 1 )$ decreases for $t > 1$ , we have $\begin{array} { r } { u _ { K } ( t ) = \hat { \mu } _ { K } ( t ) + 3 \sigma \sqrt { \frac { \log t } { N _ { K } ( t ) } } > } \end{array}$ $z + 3 \sigma { \sqrt { \frac { \log T } { T } } }$ . Similarly, for every non-target arm $i < K$

$$
u _ { i } ( t ) \leq z + 3 \sigma { \sqrt { \frac { \log T } { T } } } .
$$

Together, these inequalities imply that

$$
u _ { K } ( t ) > u _ { i } ( t ) , \qquad \forall i < K ,
$$

and hence UCB selects the target arm at round t. Repeating the same argument inductively from round $T _ { 0 } + 1$ through round T shows that UCB selects the target arm in every online round.

Proof of Theorem 4.2. Write $\rho = z - r _ { \mathrm { m i n } } , r _ { T } = S _ { T } / T _ { \mathrm { m } }$ , and $\rho _ { T } ^ { \star } = z _ { T } ^ { \star } - r _ { \operatorname* { m i n } } = \Theta ( r _ { T } )$ . We have

$$
n _ { K } = \frac { T ( \rho - \varepsilon ) _ { + } } { R - \rho } + O ( 1 ) \leq \frac { T \rho } { R - \rho } + 2 .
$$

Since $d ( r _ { \operatorname* { m i n } } + \rho ) = \rho + 3 \sigma \sqrt { \log T / T } \sim \rho$ and $B _ { i } , N _ { i } ^ { \mathrm { c l } } = O ( 1 )$ uniformly, Eq. (3) gives

$$
n _ { i } = ( 1 + o ( 1 ) ) { \frac { 9 \sigma ^ { 2 } \log T } { \rho ^ { 2 } } } \qquad ( i < K ) .
$$

(1) $\mathrm { I f } \varepsilon \le \rho _ { T } ^ { \star }$ , then $\rho = \rho _ { T } ^ { \star }$ , and the preceding estimates give $n _ { K } = O ( T r _ { T } )$ and $n _ { i } = O ( \log T / r _ { T } ^ { 2 } )$ uniformly over $i < K$ . Their sum is $O ( S _ { T } )$ , the asserted scale since $\varepsilon = O ( r _ { T } )$ .

(2) $\mathrm { I f } \varepsilon > \rho _ { T } ^ { \star }$ , then $\rho = \operatorname* { m i n } \{ \varepsilon , R / 2 \} \leq \varepsilon$ and $\rho = \Theta ( \varepsilon )$ . Thus $n _ { K } = 1$ , while $\operatorname { E q . } \left( 3 \right)$ gives

$$
n _ { i } = O \left( 1 + B _ { i } / \rho + \sigma ^ { 2 } \log T / \rho ^ { 2 } \right) = O ( \log T / \varepsilon ^ { 2 } ) \qquad ( i < K ) .
$$

Combining these two cases yields the result.

## A.2 Lower Bound Proof

Lemma A.1. Under Assumption $4 . l ,$ there exist constants $c _ { 0 } , c _ { 1 } , c _ { 2 } , c _ { 3 } > 0$ such that the following holds for all sufficiently large T. On $\mathcal { E } _ { 1 / T } ,$ , every bounded offline UCB attack realization with at most $\xi \ge 0$ non-target online pulls and $\mathcal { C } ( { \bf n } ) + \xi \le T / 2$ admits $a \rho > 0$ satisfying

$$
\begin{array} { c } { { n _ { K } \geq c _ { 1 } T ( \rho - \Delta _ { K } - c _ { 0 } \sqrt { \log T / T } ) _ { + } , } } \\ { { \displaystyle \sum _ { i < K } n _ { i } \geq \left( c _ { 2 } \frac { K \log T } { \rho ^ { 2 } } - \xi - c _ { 3 } K \right) _ { + } . } } \end{array}
$$

Proof. Since $H - \xi \ge T / 2 - T _ { 0 } ^ { \mathrm { c l } } > 0$ for large $T ,$ the target is selected in at least one online round. Let $t ^ { \star }$ be its last such round and set $\rho = u _ { K } ( t ^ { \star } ) - r _ { \operatorname* { m i n } } > 0$ . We have

$$
\begin{array} { r l } & { N _ { K } ( t ^ { \star } ) = N _ { K } ^ { \mathrm { o f f } } + N _ { K } ^ { \mathrm { o n } } ( T + 1 ) - 1 , } \\ & { N _ { K } ^ { \mathrm { c l } } ( t ^ { \star } ) = N _ { K } ^ { \mathrm { c l } } + N _ { K } ^ { \mathrm { o n } } ( T + 1 ) - 1 . } \end{array}
$$

The bound $N _ { K } ^ { \mathrm { o n } } ( T + 1 ) \ge H - \xi$ gives $N _ { K } ^ { \mathrm { c l } } ( t ^ { \star } ) \ge T / 3$ for large $T .$ Since all rounds after $t ^ { \star }$ select non-target arms, we also have $t ^ { \star } \geq T - \xi \geq T / 2$

On $\mathcal { E } _ { 1 / T }$ , the genuine target observations have empirical mean at most $\mu _ { K } + \beta _ { 1 / T } ( N _ { K } ^ { \mathrm { c l } } ( t ^ { \star } ) )$ , and each injected target reward is at most $r _ { \mathrm { m a x } }$ . Meanwhile, every non-target empirical mean is at least $r _ { \mathrm { m i n } }$ , and its index is at most the selected target’s index $r _ { \operatorname* { m i n } } + \rho .$ . These comparisons give

$$
\begin{array} { l } { \displaystyle { \rho \le \Delta _ { K } + \frac { ( R - \Delta _ { K } ) n _ { K } } { N _ { K } ( t ^ { \star } ) } + \beta _ { 1 / T } ( N _ { K } ^ { \mathrm { c l } } ( t ^ { \star } ) ) + 3 \sigma \sqrt { \frac { \log t ^ { \star } } { N _ { K } ( t ^ { \star } ) } } , } } \\ { \displaystyle { N _ { i } ( t ^ { \star } ) \ge \frac { 9 \sigma ^ { 2 } \log t ^ { \star } } { \rho ^ { 2 } } \qquad ( i < K ) . } } \end{array}\tag{4}
$$

The concentration radius and target exploration bonus are both $O ( { \sqrt { \log T / T } } )$ for large T. Thus the first inequality gives

$$
\rho \le \Delta _ { K } + { \frac { 3 R n _ { K } } { T } } + c _ { 0 } { \sqrt { \frac { \log T } { T } } } .
$$

Rearranging and using $R ~ = ~ \Theta ( 1 )$ and $n _ { K } \ \ge \ 0$ yields the target cost constraint. Write $\begin{array} { r } { M = \sum _ { i < K } n _ { i } } \end{array}$ . Summing the second inequality and using $N _ { i } ( t ^ { \star } ) \stackrel { - } { = } N _ { i } ^ { \mathrm { c l } } + n _ { i } + N _ { i } ^ { \mathrm { o n } } ( t ^ { \star } )$ and $\begin{array} { r } { \sum _ { i < K } N _ { i } ^ { \mathrm { { o n } } } ( t ^ { \star } ) \le \xi } \end{array}$ gives

$$
M + T _ { 0 } ^ { \mathrm { c l } } + \xi \geq \sum _ { i < K } N _ { i } ( t ^ { \star } ) \geq \frac { 9 \sigma ^ { 2 } ( K - 1 ) \log t ^ { \star } } { \rho ^ { 2 } } .
$$

Using $K - 1 \ge K / 2 , t ^ { \star } \ge T / 2 , T _ { 0 } ^ { \mathrm { c l } } = O ( K )$ , and $M \geq 0$ yields the non-target cost constraint with uniform constants $c _ { 2 } , c _ { 3 } > 0$ □

ProofofTheorem 4.3. Work on the intersection of the attack’s success event with $\mathcal { E } _ { 1 / T }$ , which has probability at least $1 - \delta - 1 / T$ by a union bound.

Recall $r _ { T } = S _ { T } / T$ and set $a _ { T } = ( \Delta _ { K } + c _ { 0 } \sqrt { \log T / T } ) \vee r _ { T }$ , with $c _ { 0 }$ from Lemma A.1. On paths with $\mathcal { C } ( { \bf n } ) \le T / 2$ , add the two constraints in Lemma A.1 at $\xi = 0$ . Enlarging the offset to a<sub>T</sub> only decreases the linear term, giving

$$
{ \mathcal { C } } ( { \bf n } ) + c _ { 3 } K \geq \operatorname* { i n f } _ { \rho > 0 } \left\{ c _ { 1 } T ( \rho - a _ { T } ) _ { + } + { \frac { c _ { 2 } K \log T } { \rho ^ { 2 } } } \right\} .
$$

(1) If $\rho \leq 2 a _ { T }$ , the inverse-square term is at least $c _ { 2 } K$ log $T / ( 4 a _ { T } ^ { 2 } )$ .

(2) Otherwise, the linear term is at least $c _ { 1 } T a _ { T } \geq c _ { 1 } K \log T / a _ { T } ^ { 2 }$ , since $a _ { T } ~ \ge ~ r _ { T }$ and $T r _ { T } ^ { 3 } =$ K log T.

Both cases give Ω(K log $T / ( \Delta _ { K } \vee r _ { T } ) ^ { 2 } )$ , because $a _ { T } \sim \Delta _ { K } \lor r _ { T } \ b \mathbf { y } \ { \sqrt { \log T / T } } = o ( r _ { T } )$

## A.3 Optimality Proofs

Proof of Corollary 4.4. Upper bound. Recall $\rho _ { T } ^ { \star } = z _ { T } ^ { \star } { - } r _ { \operatorname* { m i n } }$ . The cost expansions in Appendix A.1 $\mathrm { g i v e }$ , uniformly over clean logs and including rounding,

$$
\mathcal { C } ( \mathbf { n } ( z _ { T } ^ { \star } ) ) \leq ( 1 + o ( 1 ) ) \left\{ \frac { T \rho _ { T } ^ { \star } } { R } + \frac { 9 \sigma ^ { 2 } ( K - 1 ) \log T } { ( \rho _ { T } ^ { \star } ) ^ { 2 } } \right\} = ( 1 + o ( 1 ) ) \Lambda _ { T } .
$$

(1) $\operatorname { I f } \varepsilon \leq \rho _ { T } ^ { \star }$ , our construction uses $z = z _ { T } ^ { \star }$

(2) Otherwise, the target count is one at both thresholds, and raising z only reduces non-target counts.

Lower bound. Consider a $( 0 , \delta )$ -successful attack on the intersection of its success event with $\mathcal { E } _ { 1 / T }$ which has probability at least $1 - \delta { - } 1 / T$ . It suffices to consider paths with ${ \mathcal { C } } ( { \bf n } ) \leq 2 \Lambda _ { T } .$ , since larger costs already satisfy the conclusion. Write $\textstyle M = \sum _ { i < K } n _ { i }$ . Zero tolerance gives $N _ { K } ^ { \mathrm { o n } } ( T + 1 ) = H$ and $t ^ { \star } = T .$ . Since $\mathcal { C } ( \mathbf { n } ) + T _ { 0 } ^ { \mathrm { c l } } = o ( T )$ , the count identities in the proof of Lemma A.1 give $N _ { K } ( T ) , N _ { K } ^ { \mathrm { c l } } ( T ) = T ( 1 - o ( 1 ) )$ . Set $\rho = u _ { K } ( T ) - r _ { \operatorname* { m i n } }$ . Since $\Delta _ { K } = o ( \Lambda _ { T } / T )$ and ${ \sqrt { \log T / T } } =$ $o ( \Lambda _ { T } / T )$ , the index constraints in Eq. (4) imply, uniformly over these paths,

$$
\rho \leq { \frac { R } { T } } { \big ( } n _ { K } + o ( \Lambda _ { T } ) { \big ) } , \qquad M + T _ { 0 } ^ { \mathrm { c l } } \geq { \frac { 9 \sigma ^ { 2 } ( K - 1 ) \log T } { \rho ^ { 2 } } } .
$$

Eliminating ρ gives

$$
\big ( n _ { K } + o ( \Lambda _ { T } ) \big ) ^ { 2 } ( M + T _ { 0 } ^ { \mathrm { c l } } ) \geq \frac { 9 \sigma ^ { 2 } ( K - 1 ) T ^ { 2 } \log T } { R ^ { 2 } } = \frac { 4 \Lambda _ { T } ^ { 3 } } { 2 7 } .
$$

Since $n _ { K } , M = O ( \Lambda _ { T } )$ and $T _ { 0 } ^ { \mathrm { c l } } = o ( \Lambda _ { T } )$ , this yields

$$
n _ { K } ^ { 2 } M \geq ( 1 - o ( 1 ) ) \frac { 4 \Lambda _ { T } ^ { 3 } } { 2 7 } .\tag{5}
$$

AM–GM applied to $n _ { K } / 2 , n _ { K } / 2 , M$ gives

$$
( 1 - o ( 1 ) ) \frac { 4 \Lambda _ { T } ^ { 3 } } { 2 7 } \leq n _ { K } ^ { 2 } M \leq \frac { 4 ( n _ { K } + M ) ^ { 3 } } { 2 7 } = \frac { 4 \mathcal { C } ( \mathbf { n } ) ^ { 3 } } { 2 7 } ,
$$

so $\mathcal { C } ( { \bf n } ) \geq ( 1 - o ( 1 ) ) \Lambda _ { T }$ on the stated event.

Proof of Lemma 4.5. $\varepsilon = o ( \Lambda _ { T } / T )$ gives $\varepsilon = o ( \rho _ { T } ^ { \star } )$ and hence $z = z _ { T } ^ { \star }$ . Eq. (3) then gives

$$
n _ { K } = \frac { T ( \rho _ { T } ^ { \star } - \varepsilon ) } { R - \rho _ { T } ^ { \star } } + O ( 1 ) \sim \frac { T \rho _ { T } ^ { \star } } { R } = \frac { 2 } { 3 } \Lambda _ { T } .
$$

This is deterministic under the stated clean-log condition, which follows from $\Delta _ { K } = o ( \Lambda _ { T } / T )$ on ${ \mathcal { E } } _ { \delta }$ because $0 \le \varepsilon \le \Delta _ { K }$

Necessity. Work on the intersection of the attack’s success event with $\mathcal { E } _ { 1 / T }$ , which has probability at least $1 - \delta - 1 / T$ . Write $\textstyle M = \sum _ { i < K } n _ { i }$ , let $t ^ { \star }$ be the last online round selecting the target, and set $\rho = u _ { K } ( t ^ { \star } ) - r _ { \operatorname* { m i n } }$ . Since $\xi = o ( \Lambda _ { T } )$ and $\mathcal { C } ( { \bf n } ) \leq ( 1 + o ( 1 ) ) \Lambda _ { T }$ , we still have $t ^ { \star } \geq T - \xi ,$ log $t ^ { \star } \sim \log T$ , and $N _ { K } ( t ^ { \star } ) , N _ { K } ^ { \mathrm { c l } } ( t ^ { \star } ) = T ( 1 - o ( 1 ) )$ . The preceding product argument therefore applies with $M + T _ { 0 } ^ { \mathrm { c l } } + \xi$ in the non-target constraint. Because $T _ { 0 } ^ { \mathrm { c l } } + \xi = o ( \Lambda _ { T } )$ , eliminating $\rho$ again yields Eq. (5). Together with $n _ { K } ^ { 2 } M \le 4 \mathcal { C } ( { \bf n } ) ^ { 3 } / 2 7$ and the cost cap, this implies ${ \mathcal { C } } ( { \bf n } ) \sim \Lambda _ { T }$ Set $x = n _ { K } / \mathcal { C } ( { \bf n } ) \in [ 0 , 1 ]$ . Dividing the product bound by $\mathcal { C } ( { \bf n } ) ^ { 3 }$ gives

$$
\frac { 4 } { 2 7 } - o ( 1 ) \leq x ^ { 2 } ( 1 - x ) \leq \frac { 4 } { 2 7 } .
$$

The identity

$$
{ \frac { 4 } { 2 7 } } - x ^ { 2 } ( 1 - x ) = \left( x - { \frac { 2 } { 3 } } \right) ^ { 2 } \left( x + { \frac { 1 } { 3 } } \right) \geq { \frac { 1 } { 3 } } \left( x - { \frac { 2 } { 3 } } \right) ^ { 2 }
$$

forces $x = 2 / 3 + o ( 1 )$ , and hence

$$
n _ { K } = x { \mathcal { C } } ( { \bf n } ) = \left( { \frac { 2 } { 3 } } + o ( 1 ) \right) \Lambda _ { T } .
$$

Lemma A.2 (Necessary Target Cost). Under Assumption 4.1, fix $\delta \in ( 0 , 1 / 2 )$ and assume $\Delta _ { K } =$ $o ( \Lambda _ { T } / T )$ and $\xi = O ( \bar { \Lambda _ { T } } )$ ). For all sufficiently large ${ \bar { T _ { * } } }$ , every $( \xi , \delta )$ -successful bounded offline attack on UCB with a deterministic $O ( \Lambda _ { T } )$ cost bound satisfies

$$
n _ { K } = \Omega ( \Lambda _ { T } ) , \qquad \frac { n _ { K } } { \mathcal { C } ( { \bf n } ) } = \Omega ( 1 )
$$

with probability at least $1 - \delta - 1 / T .$

Proof. Work on the intersection of success with $\mathcal { E } _ { 1 / T }$ , which has probability at least $1 - \delta - 1 / T$ Since $\mathcal { C } ( { \bf n } ) + \xi = O ( \Lambda _ { T } ) = o ( T )$ , apply Lemma A.1 and use its notation. Since $\textstyle \sum _ { i < K } n _ { i } +$ $\xi + c _ { 3 } K = O ( \Lambda _ { T } )$ ), the non-target cost constraint gives $\rho = \Omega ( \Lambda _ { T } / T )$ . With $\Delta _ { K } , \sqrt { \log T / T } =$ $o ( \Lambda _ { T } / T )$ , the target cost constraint gives $n _ { K } = \Omega ( \Lambda _ { T } )$ . The deterministic cost bound then yields $\dot { n _ { K } } / \dot { \mathcal { C } } ( { \bf n } ) = \Omega ( 1 )$ □

## B Analysis for Thompson Sampling

## B.1 Attack Strategy

We consider the Thompson Sampling (TS) algorithm of Zeng et al. [2025], based on the $( \alpha , \psi )$ framework of Agrawal and Goyal [2017]. In each online round, it independently samples $\nu _ { i } ( t ) \sim$ $\mathcal { N } ( \hat { \mu } _ { i } ( t ) , 1 / N _ { i } ( \bar { t } ) )$ for every arm and selects

$$
a _ { t } = \arg \operatorname* { m a x } _ { i \in \mathcal { K } } \nu _ { i } ( t ) .
$$

For $\delta \in ( 0 , 1 )$ , define

$$
\gamma ( t ) \triangleq \sqrt { 2 \log \frac { \pi ^ { 2 } K t ^ { 2 } } { 3 \delta } } , \qquad { \mathcal { F } } \triangleq \left\{ \forall i \in K , \forall t \in { \mathcal { T } } ^ { \mathrm { o n } } , | \nu _ { i } ( t ) - \hat { \mu } _ { i } ( t ) | < { \frac { \gamma ( t ) } { \sqrt { N _ { i } ( t ) } } } \right\} .
$$

By Zeng et al. [2025, Lemma B.1], $\operatorname* { P r } ( { \mathcal { F } } ) \geq 1 - \delta .$

Fix $z \in ( r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } )$ , write $\gamma = \gamma ( T )$ , and recall $B _ { i } = N _ { i } ^ { \mathrm { c l } } ( \hat { \mu } _ { i } ^ { \mathrm { c l } } - r _ { \mathrm { m i n } } )$ . Inject $r _ { \mathrm { m a x } }$ on the target and $r _ { \mathrm { m i n } }$ on each non-target arm, with counts

$$
n _ { i } = \left\lceil \left[ \left( \frac { \gamma + \sqrt { \gamma ^ { 2 } + 4 ( z - r _ { \operatorname* { m i n } } ) B _ { i } } } { 2 ( z - r _ { \operatorname* { m i n } } ) } \right) ^ { 2 } - N _ { i } ^ { \mathrm { c l } } \right] _ { + } \right\rceil , \qquad i < K ,\tag{6}
$$

and

$$
\begin{array} { r } { n _ { K } = 1 + \operatorname* { m i n } \left\{ n \in \mathbb { Z } _ { \ge 0 } : ( r _ { \operatorname* { m a x } } - z ) n \ge ( z - \underline { { \mu } } _ { K } ) m + \gamma \sqrt { n + m } , \forall m \in [ N _ { K } ^ { \mathrm { c l } } , T ] \right\} . } \end{array}\tag{7}
$$

Lemma B.1 (TS Threshold Certificate). For $a n y \ z \in \left( r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } \right)$ , the above construction makes TS select the target in every online round on ${ \mathcal { E } } _ { \delta } \cap { \mathcal { F } } .$

Proof. On ${ \mathcal { E } } _ { \delta } \cap { \mathcal { F } }$ , suppose every preceding online pull selected the target. Applying the count constraints with $m = \dot { N } _ { K } ^ { \mathrm { c l } } ( t )$ gives

$$
\nu _ { K } ( t ) > \frac { n _ { K } r _ { \operatorname* { m a x } } + N _ { K } ^ { \mathrm { c l } } ( t ) \underline { { \mu } } _ { K } } { n _ { K } + N _ { K } ^ { \mathrm { c l } } ( t ) } - \frac { \gamma } { \sqrt { n _ { K } + N _ { K } ^ { \mathrm { c l } } ( t ) } } \geq z ,
$$

$$
\nu _ { i } ( t ) < r _ { \operatorname* { m i n } } + \frac { B _ { i } } { N _ { i } ^ { \mathrm { c l } } + n _ { i } } + \frac { \gamma } { \sqrt { N _ { i } ^ { \mathrm { c l } } + n _ { i } } } \leq z \qquad ( i < K ) .
$$

And hence $\nu _ { K } ( t ) > \nu _ { i } ( t )$ . Induction from the first online round proves the claim.

Theorem B.2 (Attack Cost Upper Bound and Allocation for TS). Fix $\delta \in ( 0 , 1 / 2 )$ . Under Assumption 4.1, define

$$
\Lambda _ { T } ^ { \mathrm { t s } } \triangleq 3 \left( \frac { ( K - 1 ) \gamma ( T ) ^ { 2 } T ^ { 2 } } { 4 R ^ { 2 } } \right) ^ { 1 / 3 } = \Theta ( S _ { T } ) .
$$

The following hold for all sufficiently large T.

$I f \Delta _ { K } = o ( S _ { T } / T )$ , there exists a threshold for the above allocation satisfying

$$
\operatorname* { P r } \left( \sum _ { i = 1 } ^ { K - 1 } N _ { i } ^ { \mathrm { o n } } ( T + 1 ) = 0 \right) \geq 1 - 2 \delta \quad a n d \quad \mathcal { C } ( \mathbf { n } ) \leq ( 1 + o ( 1 ) ) \Lambda _ { T } ^ { \mathrm { t s } } .
$$

Moreover, under ${ \mathcal { E } } _ { \delta }$

$$
n _ { K } = \left( \frac { 2 } { 3 } + o ( 1 ) \right) \Lambda _ { T } ^ { \mathrm { t s } } \quad a n d \quad n _ { i } = ( 1 + o ( 1 ) ) \frac { \Lambda _ { T } ^ { \mathrm { t s } } } { 3 ( K - 1 ) } f o r i < K .
$$

N<sub>K</sub>/T N<sup>on</sup>/H Clipped Suppression $N _ { K } / T$  
![](images/8d8b840e6757eb43b5b5b294a865a34c500990a74ee739c3d76c7bea3208748b.jpg)  
(a) Attack Cost vs. Horizon T

![](images/d801e6bd725d521c373174eb56f6c25185766f4c38098d217c41e9933d188037.jpg)  
Clipped Suppression  
(b) Attack Cost vs. Number of Arms K  
Total Attack Cost

Figure 5: Attack Cost for TS.  
![](images/8a9b15154b7e88ddde90508c236b9636c73db4111273fb51db3669513331acdb.jpg)  
(a) Target-arm Selection Ratio

![](images/2f7528f7bc19bb35c15a587290bfa6180a8e2d7ff1e766f9779de3fa54f730b3.jpg)  
(b) Target-arm Selection Ratio vs. Number of Arms K  
Figure 6: Target-arm Selection Ratio for TS.

ProofofTheorem B.2. For $\rho = z - r _ { \operatorname* { m i n } } = \Theta ( S _ { T } / T )$ , expanding the count formulas gives, uniformly over clean logs and including rounding,

$$
n _ { K } = \frac { T ( \rho - \varepsilon ) _ { + } } { R } + O ( T \rho ^ { 2 } + \gamma \sqrt { T } + 1 ) , \qquad n _ { i } = ( 1 + o ( 1 ) ) \frac { \gamma ^ { 2 } } { \rho ^ { 2 } } \quad ( i < K ) .
$$

The target lower bound follows by setting $m = T$ in its constraint; the upper bound uses $( \rho - \varepsilon ) m \leq$ $T ( \rho - \varepsilon ) _ { + }$ and $\sqrt { n + m } \leq \sqrt { T } + n / ( 2 \sqrt { T } )$ . The error is $o ( S _ { T } )$ . Choose

$$
\rho = \left( \frac { 2 R ( K - 1 ) \gamma ^ { 2 } } { T } \right) ^ { 1 / 3 } , \qquad \frac { T \rho } { R } = \frac 2 3 \Lambda _ { T } ^ { \mathrm { t s } } , \qquad \frac { ( K - 1 ) \gamma ^ { 2 } } { \rho ^ { 2 } } = \frac 1 3 \Lambda _ { T } ^ { \mathrm { t s } } .
$$

Since $( \rho - \varepsilon ) _ { + } \leq \rho ,$ , the cost bound holds deterministically and $T _ { 0 } ^ { \mathrm { c l } } + { \mathcal C } ( { \bf n } ) = o ( T )$ . On $\mathcal { E } _ { \delta } ,$ $0 \leq \varepsilon \leq \Delta _ { K } = o ( \rho )$ , giving the stated allocation. Finally, Lemma B.1 and $\operatorname* { P r } ( \mathcal { E } _ { \delta } \cap \dot { \mathcal { F } } ) \geq 1 - 2 \delta$ give the success guarantee. □

## B.2 Numerical Illustration

We repeat the Clipped Suppression comparisons and near-boundary experiments of Section 5 with TS, and numerically minimize the continuous total count defined by Eq. (6) and Eq. (7) before rounding. All other settings and reporting conventions are unchanged.

Figures 5 and 6 show exclusively target online pulls for our attack in every repeat. Its cost is lower than Clipped Suppression except at the smallest arm count, where the baseline costs less but still pulls non-target arms.

Figure 7 shows normalized cost approaching one and cost shares approaching $2 / 3$ for the target and $1 / 3$ for the non-target arms collectively, consistent with the TS construction. Every repeat has zero non-target online pulls.

![](images/f9a0f007123ea35e959c8f813f6ceaa3dcd72875d7056928b3e0acb21be95056.jpg)  
K = 5 K = 10 K = 15 K = 20 K = 25  
(a) Normalized Attack Cost vs. Horizon T

![](images/da0de080470b2c5b97b011e703ba8ae0b7ced8dc419343acc514341c402caa46.jpg)  
(b) Cost Share vs. Horizon T  
Figure 7: Near-Boundary Cost and Allocation for TS.

## C Analysis for ϵ-Greedy

## C.1 Attack Strategy

At round s, ϵ-greedy explores with probability $\epsilon _ { s }$ by choosing uniformly from all K arms; otherwise it selects an empirical-mean maximizer. The schedule $\{ \check { \epsilon } _ { s } \} _ { s = 1 } ^ { T } \subset \mathsf { \bar { [ 0 , 1 ] } }$ is deterministic, and exploration randomization is independent across rounds.

For simplicity, assume that the entire offline log is constructed by the attacker, so $N _ { i } ^ { \mathrm { c l } } = 0$ for every arm and $T _ { 0 } \doteq \mathcal { C } ( \mathbf { n } )$

For $0 \leq t < T$ and $\delta \in ( 0 , 1 )$ , define the exploration bound

$$
B _ { i } ^ { \mathrm { e g } } ( t , T , \delta ) = \sum _ { s = t + 1 } ^ { T } \frac { \epsilon _ { s } } { K } + \sqrt { 2 \left( \sum _ { s = t + 1 } ^ { T } \frac { \epsilon _ { s } } { K } \right) \log \frac { 1 } { \delta } } + \frac { 1 } { 3 } \log \frac { 1 } { \delta } .
$$

Lemma C.1 (Corollary C.1 of Zeng et al. [2025]). Fix $T _ { 0 } \leq t < T$ and $\delta \in ( 0 , 1 )$ . With probability at least $1 - \delta ,$ , the following implication holds: $i f \hat { \mu } _ { i } ( s ) < \operatorname* { m a x } _ { j } \hat { \mu } _ { j } ( s )$ for all $s \in \{ t + 1 , \ldots , T \}$ then

$$
N _ { i } ^ { \mathrm { o n } } ( T + 1 ) - N _ { i } ^ { \mathrm { o n } } ( t + 1 ) \leq B _ { i } ^ { \mathrm { e g } } ( t , T , \delta ) .
$$

Fix $\delta \in ( 0 , 1 / 2 )$ , write $B = \lceil B _ { i } ^ { \mathrm { e g } } ( 0 , T , \delta / ( K - 1 ) ) \rceil$ for short, and define

$$
r _ { T } ^ { \mathrm { e g } } = R \sqrt { \frac { ( K - 1 ) B } { T } } .
$$

Inject reward $r _ { \mathrm { m a x } }$ on the target and $r _ { \mathrm { m i n } }$ on each non-target arm, with

$$
n _ { K } = \left\lceil \sqrt { T ( K - 1 ) B } \right\rceil + 1 , \qquad n _ { i } = \left\lceil \sqrt { \frac { T B } { K - 1 } } \right\rceil \quad ( i < K ) .
$$

Theorem C.2 (ϵ-Greedy Upper Bound). Fix $\delta ~ \in ~ ( 0 , 1 / 2 )$ . Suppose $\begin{array} { r } { R \ = \ \Theta ( 1 ) , \ \sum _ { s = 1 } ^ { T } \epsilon _ { s } \ + } \end{array}$ $K \log ( K / \delta ) = o ( T )$ . For all sufficiently large T, the above attack satisfies

$$
\operatorname* { P r } \left( \sum _ { i < K } N _ { i } ^ { \mathrm { o n } } ( T + 1 ) \leq ( K - 1 ) B \right) \geq 1 - \delta , \qquad { \mathcal { C } } ( { \mathbf { n } } ) = ( 2 + o ( 1 ) ) \sqrt { T ( K - 1 ) B } .
$$

All greedy online decisions select the target on the same event.

Proof. The definition of B gives $\begin{array} { r } { ( K - 1 ) B = O ( \sum _ { s } \epsilon _ { s } + K \log ( K / \delta ) ) = o ( T ) } \end{array}$ , while rounding yields

$$
2 { \sqrt { T ( K - 1 ) B } } < { \mathcal { C } } ( \mathbf { n } ) \leq 2 { \sqrt { T ( K - 1 ) B } } + K + 1 .
$$

Since $B \geq 1$ and $K = o ( T )$ , the rounding term is lower order, proving the cost bound and $T _ { 0 } =$ ${ \mathcal { C } } ( { \bf n } ) < T$ eventually.

![](images/184acd161a644e8c397a7ee4e5c1d1b8af089f92f9f89c8d7bb646bf92f8dd3b.jpg)  
(a) Attack Cost vs. Horizon T

![](images/53794904339edda35d83c6f56b1ef2e5594e90c1a456e58a5ae68ab8622e02b4.jpg)  
Clipped Suppression  
(b) Attack Cost vs. Number of Arms K  
Total Attack Cost

Figure 8: Attack Cost for ϵ-greedy.  
![](images/8aa48d644bd5cdce2d654e643db56c2afe0eb5b4ee555ba88aa11324ef2316b2.jpg)  
(a) Target-arm Selection Ratio  
vs. Horizon T

![](images/090dd78359a0b7b7936c5fdf28ae661ff2f3babe9830ac365331df4a25d6555e.jpg)  
(b) Target-arm Selection Ratio vs. Number of Arms K  
Clipped Suppression $N _ { K } / T$  
Figure 9: Target-arm Selection Ratio for ϵ-greedy.

The Bernoulli bound underlying Lemma C.1 and a union bound ensure at most B potential exploratory selections per non-target arm over rounds $1 , \ldots , T$ , with probability at least $1 - \delta .$ . On this event, if preceding greedy decisions selected the target, bounded rewards and $n _ { K } n _ { i } > T B$ give

$$
\hat { \mu } _ { i } ( s ) \leq r _ { \operatorname* { m i n } } + \frac { R B } { n _ { i } + B } < r _ { \operatorname* { m i n } } + \frac { R n _ { K } } { n _ { K } + T } \leq \hat { \mu } _ { K } ( s ) .
$$

Induction from the first online round makes every greedy decision select the target, so all non-target pulls are exploratory and total at most $( K - 1 ) B$ □

For $\epsilon _ { s } = \operatorname* { m i n } \{ 1 , c K / s \}$ with fixed $c > 0$ and K log $T = o ( T )$ , this gives $r _ { T } ^ { \mathrm { e g } } = \Theta ( \sqrt { K \log T / T } )$ cost $O ( { \sqrt { T K \log T } } )$ , and (ξ, δ)-success with $\xi ( T ) = O ( K \log T )$

## C.2 Numerical Illustration

We use Bernoulli rewards with non-target means equally spaced in [0.2, 0.8] and target mean $r _ { T } ^ { \mathrm { e g } } / \log T ~ > ~ 0 .$ , with $\epsilon _ { s } = \mathrm { m i n } \{ 1 , K / \bar { s } \}$ and $\delta \ = \ 0 . 0 5$ Online statistics use 50 independent runs per configuration; shaded bands show one sample standard deviation. Greedy ties are broken by arm index.

Clipped Suppression sets every non-target online reward to zero and counts each such round as attack cost. It has no offline log and initializes with one online pull per arm, starting with the target.

Figures 8 and 9 vary T at $K = 1 0$ and K at $T = 1 0 ^ { 6 }$ . Our attack has lower mean costs and higher mean target-arm selection ratios. All greedy decisions selected the target; the remaining non-target pulls were exploratory and stayed below the theoretical bound.

## D Analysis of Suppression-Bounded Algorithms

ProofofTheorem 4.8. Fix $\rho \in ( 0 , R )$ and write $B _ { i } ^ { \mathcal { A } } = B _ { i } ^ { \mathcal { A } } ( T _ { 0 } ^ { \mathrm { c l } } , T , \delta / K )$ . Choose

$$
n _ { K } = \left\lceil T \frac { ( \rho - \varepsilon ) _ { + } } { R - \rho } \right\rceil + 1 , \quad n _ { i } = \left\lceil \frac { R - \rho } { \rho } \big ( N _ { i } ^ { \mathrm { c l } } + B _ { i } ^ { A } \big ) \right\rceil , \mathrm { ~ f o r ~ a l l ~ } i < K .
$$

On ${ \mathcal { E } } _ { \delta }$ , every empirical average of the genuine target-arm rewards is at least $\underline { { \mu } } _ { K } = r _ { \operatorname* { m i n } } + \varepsilon$ . Thus, for any $t \in \mathcal { T } ^ { \mathrm { o n } }$

$$
\hat { \mu } _ { K } ( t ) \geq \frac { n _ { K } r _ { \operatorname* { m a x } } + N _ { K } ^ { \mathrm { c l } } ( t ) \underline { { \mu } } _ { K } } { n _ { K } + N _ { K } ^ { \mathrm { c l } } ( t ) } > r _ { \operatorname* { m i n } } + \rho .
$$

For any bandit algorithm A, the pull count $N _ { i } ( t )$ , excluding injected samples, is nondecreasing in t. Thus, for each arm $i < K$ , we use the conservative upper bound $B _ { i } ^ { \mathcal { A } }$ uniformly over $t \in [ T _ { 0 } ^ { \mathrm { c l } } + 1 , T ]$ We now use induction over online rounds. Initially, $N _ { i } ^ { \mathrm { o n } } ( T _ { 0 } + 1 ) = 0 . \ \mathrm { A t }$ any later round $t ,$ if arm i has been suppressed at every preceding online decision, we have $N _ { i } ^ { \mathrm { o n } } ( t ) \leq B _ { i } ^ { A }$ . In either case, $N _ { i } ^ { \mathrm { c l } } ( t ) = N _ { i } ^ { \mathrm { c l } ^ { - } } + N _ { i } ^ { \mathrm { o n } } ( t ) \leq \bar { N _ { i } ^ { \mathrm { c l } } } + B _ { i } ^ { A }$ , so, since all genuine rewards are at most $r _ { \mathrm { m a x } } ,$

$$
\hat { \mu } _ { i } ( t ) \leq \frac { r _ { \operatorname* { m a x } } N _ { i } ^ { \mathrm { c l } } ( t ) + r _ { \operatorname* { m i n } } n _ { i } } { N _ { i } ^ { \mathrm { c l } } ( t ) + n _ { i } } \leq r _ { \operatorname* { m i n } } + \rho .
$$

Hence

$$
\hat { \mu } _ { i } ( t ) \le r _ { \mathrm { m i n } } + \rho < \hat { \mu } _ { K } ( t ) .
$$

Intersecting ${ \mathcal { E } } _ { \delta }$ with the $K - 1$ pull-bound events gives probability at least $1 - 2 \delta$ by a union bound. On this event,

$$
\sum _ { i = 1 } ^ { K - 1 } N _ { i } ^ { \mathrm { o n } } ( T + 1 ) \leq \sum _ { i = 1 } ^ { K - 1 } B _ { i } ^ { \mathcal { A } } .
$$

Summing the displayed injection counts and taking the infimum yields the claimed cost bound, including an $O ( \bar { K } )$ rounding overhead. □

## E Exploration and Allocation

The allocation depends on how injections control exploration. Let $\rho = z - r _ { \operatorname* { m i n } }$ and $\textstyle M = \sum _ { i < K } n _ { i }$ In the near-boundary regime, the genuine target rewards contribute little to maintaining the threshold, so the injections must sustain a mean increase of approximately $\rho$ over $T$ observations. This gives the leading target cost n $\kappa \sim T \rho / R$ . The same expression follows from the worst-case reward bound used in our ϵ-greedy construction.

UCB and TS. For UCB, suppressing a competitor requires its exploration bonus to fit below the target’s threshold. Since the bonus decays as $N _ { i } ^ { - 1 / 2 }$ , a threshold height $\rho$ requires an observation count proportional to $\rho ^ { - 2 }$ . Injected samples supply this count before online learning begins. TS has the same dependence because its sampling uncertainty also decays as $N _ { i } ^ { - 1 / 2 }$ . Because all competitors share the threshold $\rho ,$ summing their leading injection costs changes only the coefficient of $\rho ^ { - 2 }$ so $\textstyle M = \sum _ { i < K } n _ { i }$ has the same dependence. Thus, for both constructions, raising $\rho$ increases target cost linearly but reduces aggregate non-target cost as $\rho ^ { - 2 }$ . A small relative increase in $\rho$ adds that fraction of $n _ { K }$ and saves twice that fraction of M. At the minimizing threshold these amounts agree, giving $n _ { K } \sim 2 M ;$ the target accounts for $2 / 3$ of the total cost.

ϵ-greedy. Here exploration is prescribed by the schedule and cannot be removed by increasing observation counts. Each competitor can receive up to B exploratory rewards. Even if all equal $r _ { \mathrm { m a x } }$ , injecting $n _ { i }$ rewards equal to $r _ { \mathrm { m i n } }$ keeps its mean below the threshold whenever

$$
{ \frac { R B } { n _ { i } + B } } \leq \rho .
$$

For small $\rho ,$ this requires approximately $R B / \rho$ injections per competitor. Hence M decreases as $\rho ^ { - 1 }$ : a small relative increase in $\rho$ saves only that same fraction of $M$ . Balancing this saving against the increase in target cost gives $n _ { K } \sim M$ , so the target accounts for $1 / 2$ of the total cost.