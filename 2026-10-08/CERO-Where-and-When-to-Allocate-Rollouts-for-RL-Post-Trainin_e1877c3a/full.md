# CERO: Where and When to Allocate Rollouts for RL Post-Training

Yiming Zong<sup>a</sup>, Yige Wang<sup>a</sup>, Xing Hu<sup>b</sup>, Jiashuo Jiang<sup>a</sup>, Zuo-Jun Max Shen<sup>c</sup>

<sup>a</sup> Department of Industrial Engineering & Decision Analytics, Hong Kong University of Science and Technology <sup>b</sup> Innovation and Information Management, The University of Hong Kong

<sup>c</sup> Faculty of Engineering & Faculty of Business and Economics, The University of Hong Kong

Adaptive rollout methods for group-relative reinforcement learning typically allocate a fixed per-update budget across prompts. We instead study how to coordinate a finite rollout budget over the entire training horizon. We formulate this problem using a concave surrogate utility of cumulative prompt exposure and in troduce CERO, an online primal–dual scheduler for prompt admission and budget pacing. In our experiments, each admitted prompt receives a fixed-size response group. CERO instead adapts which prompts are selected, how often they are revisited across rounds, and how many groups are generated in each round. A compact Fenchel representation linearizes the dependence on cumulative exposure, while projected online gradient descent updates prompt-specific supporting slopes and a shared budget price using reward-variation feed back and budget deviations. We establish pathwise guarantees for the surrogate allocation objective against fixed-rate and same-path time-varying benchmarks, with explicit terms for proxy discrepancy and rate variation. Under matched training-response budgets, CERO attains the highest avg@16 macro-average on each of three backbones across five mathematical reasoning benchmarks. Mechanistic analyses link CERO’s prompt choices to within-group reward contrast, while multi-seed ablations show gains from adaptive pacing over both uniform and preset spending schedules.

Key words : Reinforcement Learning, LLM Post-training, Online Resource Allocation, Rollout Optimization

## 1. Introduction

Reinforcement learning with verifiable rewards has become a standard approach for post-training reasoning models. Within this paradigm, recent work has increasingly focused on adaptive rollout allocation to improve training eficiency. Standard GRPO-style training typically assigns the same number of responses to each selected prompt (Shao et al. 2024, Guo et al. 2025), even though the resulting groups can difer substantially in informativeness. In particular, all-correct or all-incorrect groups provide no within-group reward contrast. Recent methods therefore adapt how rollout budgets are distributed across prompts: KnapsackRL (Li et al. 2025) allocates a fixed exploration budget according to estimated task value, VIP (Nguyen et al. 2026) allocates minibatch rollouts to reduce expected gradient variance, and HORA (Wang et al. 2026) redistributes a batch budget according to marginal hit utility. Collectively, these methods primarily adapt where rollouts are allocated within a given round-level budget. We focus on a complementary temporal dimension that arises when rollout decisions are coupled by a shared finite training budget:

Given a finite rollout budget, when should we spend more, and when less, across training?

This horizon-level problem remains nontrivial even when each admitted prompt receives a fixedsize response group. The scheduler must still decide which prompts to admit, how often to revisit them across rounds, and how much of the remaining budget to spend at diferent training stages. Rollouts spent now cannot be reassigned later, and repeated exposure to the same prompt may yield diminishing marginal utility. We capture this efect with a concave utility over cumulative prompt exposure. These couplings motivate a resource allocation formulation over prompts and rounds.

However, this finite-horizon formulation is not directly separable across rounds: each prompt’s cumulative allocation appears inside a nonlinear concave utility, and the global budget further couples spending decisions over time. Our key methodological contribution is to adapt Fenchel dualization and projected online gradient descent (OGD) to this rollout optimization problem. The Fenchel representation linearizes dependence on cumulative exposure, reducing the coupled horizon objective to tractable round-level allocation problems. Projected OGD then adapts promptspecific supporting slopes from reward feedback and a shared budget price from aggregate allocation deviations.

These dual signals play complementary roles in CERO: prompt-specific slopes guide prompt prioritization, while the shared price regulates budget pacing across rounds. A prompt-level Beta state provides a predictive signal of future within-group reward variation, while the dual variables and the remaining budget are carried forward across rounds, enabling coordination throughout training. This online structure admits pathwise guarantees for the surrogate allocation objective under fixed rates and a same-path time-varying benchmark. Empirically, CERO achieves the highest avg@16 macro-average on all three evaluated backbones under matched training-response budgets. Matched controls and mechanistic analyses further connect these gains to prompt-aware assignment and horizon-level budget adaptation, with CERO outperforming each ablated variant in multi-seed average performance.

Our contributions are threefold:

• We formulate finite-horizon rollout optimization as a resource-allocation problem with a concave utility over cumulative prompt exposure, and derive an online primal–dual scheduler using a compact Fenchel representation and projected OGD. The resulting scheduler coordinates prompt admission, repeated exposure across rounds, and per-round spending under a shared finite budget.

![](images/804cefd1ef942a95b309c5f32135c1d79da7b79fe83e4b65639fc099284df3f4.jpg)

![](images/d403e1fc4e0bd886f266d9d210143a348cfd99b29a09092f057b232eff530b75.jpg)  
Figure 1 Schematic comparison of rollout optimization. (a) Typical adaptive methods redistribute a fixed per-round budget across prompts. (b) CERO coordinates prompt admission and per-round budget pacing under a shared finite training budget, with admitted prompts receiving fixed-size response groups.

• We establish pathwise guarantees for the surrogate allocation objective under fixed rates and a same-path time-varying benchmark. The bounds explicitly separate online optimization error, reward-proxy discrepancy, and temporal rate variation.

• We evaluate CERO on three backbones and five mathematical reasoning benchmarks under the same rollout budgets. CERO obtains the highest avg@16 macro-average on each backbone. Matched controls and multi-seed ablations further connect these gains to prompt-aware assignment and horizon-level coordination, with CERO outperforming fixed-pacing, preset-pacing, and direct marginal-utility variants.

Relation to prior work. CERO lies at the intersection of three related directions: dificulty-aware sampling and filtering, adaptive rollout allocation, and online resource allocation. The first focuses on identifying and prioritizing informative prompts, the second studies how limited rollout budgets should be allocated across prompts, and the third provides a broader framework for coordinating sequential decisions under shared resource constraints. Appendix A provides a detailed review of these three lines of research.

## 2. Finite-Horizon Rollout Optimization

## 2.1. Problem Formulation

We consider a finite training process of K rounds over a fixed pool $\mathcal { X } = \{ x _ { 1 } , \dots , x _ { M } \}$ of M prompts. Let $N _ { i , k }$ denote the number of units assigned to prompt i in round k, where one allocation unit corresponds to a fixed quantum of $g \in \mathbb { Z } _ { > 0 }$ generated responses. Define its cumulative exposure by $\begin{array} { r } { z _ { i , k } : = \sum _ { t = 1 } ^ { k } N _ { i , t } , ~ z _ { i , 0 } = 0 } \end{array}$ . Let $N _ { \mathrm { m a x } } , C , B \in \mathbb { Z } _ { \ge 0 }$ be the per-prompt per-round cap, the aggregate per-round capacity, and the total rollout budget. Our goal is to maximize the total utility of cumulative prompt exposure by allocating a finite rollout budget across prompts and rounds:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \mathbf { N } \in \{ 0 , \ldots , N _ { \operatorname* { m a x } } \} ^ { M \times K } } } & { \displaystyle \sum _ { i = 1 } ^ { M } U _ { i } ( z _ { i , K } ) } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { i = 1 } ^ { M } N _ { i , k } \leq C , \qquad k \in [ K ] , } \\ & { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { M } N _ { i , k } \leq B . } \end{array}\tag{1}
$$

Here $U _ { i }$ is a nondecreasing concave utility of cumulative exposure, specified in the next subsection. We assume

$$
0 < B \leq K C , \quad C \leq M N _ { \operatorname * { m a x } } .\tag{2}
$$

Denote the feasible set in eq. (1) by $\mathcal { N } _ { B , C }$ . We distinguish allocation units from generated-response counts using the superscript resp: $N _ { i , k } ^ { \mathrm { r e s p } } = g N _ { i , k } , N _ { \mathrm { m a x } } ^ { \mathrm { r e s p } } = g N _ { \mathrm { m a x } } , C ^ { \mathrm { r e s p } } = g C$ , and $\begin{array} { r } { B ^ { \mathrm { r e s p } } = g B } \end{array}$ . Finally, define

$$
Z : = K N _ { \mathrm { m a x } } , \qquad \bar { B } : = { \frac { B } { K } } , \qquad R : = { \frac { B } { M } } ,
$$

where $Z$ is the maximum per-prompt exposure over training, $\bar { B }$ is the average per-round budget, and R is the equal-share exposure scale.

## 2.2. Value-capture utility

We use a concave surrogate utility to model diminishing returns to cumulative prompt exposure. Each prompt has a latent value-capture rate $q _ { i } \in [ 0 , 1 / 4 ]$ , for which Section 3 provides a rewardvariation proxy. We instantiate the utility with the normalized exponential form

$$
U ( q , z ) : = 1 - \exp \left( - 4 \lambda q \frac { z } { R } \right) , \quad q \in [ 0 , 1 / 4 ] , \quad z \in [ 0 , Z ] ,
$$

where $\lambda > 0$ controls saturation and $z / R$ measures exposure relative to the equal-share horizon scale. By construction, the utility is nondecreasing and concave in $z ,$ encoding diminishing marginal utility with exposure. The marginal utility of an additional allocation unit satisfies

$$
0 \leq \partial _ { z } U ( q , z ) = \frac { 4 \lambda q } { R } \exp \left( - 4 \lambda q \frac { z } { R } \right) \leq \frac { \lambda } { R } = : \bar { c } .\tag{3}
$$

This bound yields the compact slope domain [0, c¯] used in the Fenchel representation below. For comparison, consider a full-information oracle that knows $\pmb q = ( q _ { 1 } , \dots , q _ { M } )$ in advance and chooses the feasible allocation maximizing total value capture:

$$
\mathsf { O P T } _ { K } ^ { \mathrm { s t a t } } ( \pmb q ) : = \operatorname* { m a x } _ { \mathbf { N } \in { \cal N } _ { B , C } } \sum _ { i = 1 } ^ { M } U ( q _ { i } , z _ { i , K } ) .\tag{4}
$$

Since the objective depends on the allocation history only through $z _ { i , K } .$ , the oracle problem is determined by the terminal exposure vector. Appendix C.2 gives its optimal allocation.

## 2.3. Compact Fenchel representation

To simplify notation, let $U _ { i } ( z ) : = U ( q _ { i } , z )$ . Using the uniform slope bound in $\mathrm { e q . ~ ( 3 ) }$ , define the restricted concave conjugate

$$
U _ { i } ^ { \ast } ( \theta ) : = \operatorname* { i n f } _ { 0 \leq s \leq Z } \{ \theta s - U _ { i } ( s ) \} , \quad \theta \in [ 0 , \bar { c } ] .
$$

The interval [0, c¯] contains a supporting slope at every exposure level, so the representation remains exact on this compact domain. By concavity,

$$
U _ { i } ( z ) = \operatorname* { m i n } _ { 0 \leq \theta \leq \bar { c } } \{ \theta z - U _ { i } ^ { * } ( \theta ) \} , \quad \quad 0 \leq z \leq Z .
$$

Thus, the nonlinear dependence on cumulative exposure can be represented exactly through afine functions of $z .$ To capture the scarcity of the shared budget across rounds, let $\mu \in [ 0 , \bar { c } ]$ denote the corresponding dual price. For a single round, let $\pmb { n } = ( n _ { 1 } , \dots , n _ { M } )$ denote the allocation vector and define the continuous feasible region

$$
\mathcal { P } _ { C } : = \left\{ \pmb { n } \in \mathbb { R } ^ { M } : 0 \leq n _ { i } \leq N _ { \operatorname* { m a x } } , \ \sum _ { i } n _ { i } \leq C \right\} .
$$

Since $C$ and $N _ { \mathrm { m a x } }$ are integers, $\mathcal { P } _ { C }$ is integral, so the continuous relaxation is exact for the roundlevel linear problem. Let $\pmb { \theta } = ( \theta _ { 1 } , \dots , \theta _ { M } ) \in [ 0 , \bar { c } ] ^ { M }$ collect the prompt-specific supporting slopes. Combining the Fenchel representation with the shared budget price $\mu$ makes the horizon-level upper bound separable across rounds, yielding the per-round dual objective

$$
L ( \pmb { \theta } , \mu ) = \bar { B } \mu + \operatorname* { m a x } _ { \pmb { n } \in \mathcal { P } _ { C } } \sum _ { i = 1 } ^ { M } ( \theta _ { i } - \mu ) \boldsymbol { n } _ { i } - \frac { 1 } { K } \sum _ { i = 1 } ^ { M } U _ { i } ^ { * } ( \theta _ { i } ) ,
$$

By weak duality, OP $\mathsf { T } _ { K } ^ { \mathrm { s t a t } } ( \pmb q ) \leq K L ( \pmb \theta , \mu )$ for all $( \theta , \mu )$ in the stated domains. Let $d _ { i } : = \theta _ { i } - \mu$ , which compares prompt $x _ { i } \mathrm { { ^ { * } s } }$ supporting slope against the shared budget price.

## 3. Reward-Variation Proxy

We use predicted within-group reward variance as a proxy for the latent value-capture rate in Section 2.2. This quantity characterizes expected reward contrast in a fresh response group, rather than directly estimating the downstream improvement from a policy update. Let $r ( x , y ) \in \{ 0 , 1 \}$ be the verifiable reward. At the beginning of round k, we maintain an auxiliary Beta state for prompt $x _ { i } { : }$

$$
P _ { i , k } \mid \mathcal { H } _ { i } ^ { k } \sim \mathrm { B e t a } ( a _ { i , k } , b _ { i , k } ) , \quad a _ { i , k } , b _ { i , k } > 0 ,
$$

where $\mathcal { H } _ { i } ^ { k }$ denotes the reward history of prompt i. This state provides a compact predictive summary of the prompt’s observed reward history. Let $G _ { i , k } : = N _ { i , k } ^ { \mathrm { r e s p } } = g N _ { i , k }$ denote the number of responses generated for prompt $x _ { i }$ in round k. Let $S _ { i , k }$ and $F _ { i , k }$ denote the numbers of successful and failed

responses, and set $S _ { i , k } = F _ { i , k } = 0$ if there is no allocation to this prompt. After round k, we update the Beta state with forgetting factor $\rho \in ( 0 , 1 ]$ :

$$
a _ { i , k + 1 } = a _ { 0 } + \rho ( a _ { i , k } - a _ { 0 } ) + S _ { i , k } , \ b _ { i , k + 1 } = b _ { 0 } + \rho ( b _ { i , k } - b _ { 0 } ) + F _ { i , k } ,\tag{5}
$$

where $a _ { 0 } , b _ { 0 } > 0$ are the prior parameters. When $\rho = 1$ , the update reduces to standard cumulative Beta–Bernoulli updating. When $\rho < 1$ , older evidence is gradually discounted toward the prior, allowing the state to track changes in the policy. We use predicted reward variation as a proxy for the latent value-capture rate:

$$
\widehat { q } _ { i , k } : = \mathbb { E } [ P _ { i , k } ( 1 - P _ { i , k } ) \mid \mathcal { H } _ { i } ^ { k } ] = \frac { a _ { i , k } b _ { i , k } } { ( a _ { i , k } + b _ { i , k } ) ( a _ { i , k } + b _ { i , k } + 1 ) } \in [ 0 , 1 / 4 ] .\tag{6}
$$

Proposition 1 (Predicted within-group reward variance). For any group size $G \geq 2$ suppose that conditional on $( P _ { i , k } , \mathcal { H } _ { i } ^ { k } )$ the next G binary rewards are i.i.d. Bernoulli( $P _ { i , k } )$ . Let $\begin{array} { r } { \bar { r } _ { G } : = G ^ { - 1 } \sum _ { j = 1 } ^ { G } r ^ { ( j ) } } \end{array}$ and $\begin{array} { r } { \widehat { \mathrm { V a r } } _ { G } ( r ) : = \frac { 1 } { G - 1 } \sum _ { j = 1 } ^ { G } \left( r ^ { ( j ) } - \bar { r } _ { G } \right) ^ { 2 } } \end{array}$ . Then it holds that

$$
\mathbb { E } [ \widehat { \mathrm { V a r } } _ { G } ( r ) | \mathcal { H } _ { i } ^ { k } ] = \widehat { q _ { i , k } } .
$$

Thus, $\widehat { q } _ { i , k }$ is the predictive expected within-group reward variance. Let $m _ { i , k } : = \mathbb { E } [ P _ { i , k } \mid \mathcal { H } _ { i } ^ { k } ]$ denote the predictive mean and we can obtain

$$
\widehat { q } _ { i , k } = m _ { i , k } ( 1 - m _ { i , k } ) - \operatorname { V a r } ( P _ { i , k } \mid \mathcal { H } _ { i } ^ { k } ) .
$$

Therefore, $\widehat { q } _ { i , k }$ is largest when the predicted success rate is near $1 / 2$ , but is reduced when the Beta state is more uncertain.

Proposition 2 (Fixed-sample stationary calibration). $L e t \qquad Y _ { 1 } , \ldots , Y _ { n } \qquad b e \qquad i . i . d .$ Bernoulli(p) observations with fixed sample size n. Define $\begin{array} { r } { a : = a _ { 0 } + \sum _ { j = 1 } ^ { n } Y _ { j } , b : = b _ { 0 } + n - } \end{array}$ $\begin{array} { r } { \sum _ { j = 1 } ^ { n } Y _ { j } , \ s _ { 0 } : = a _ { 0 } + b _ { 0 } } \end{array}$ , and let $\widehat { q } _ { n } : = a b / [ ( a + b ) ( a + b + 1 ) ]$ and $q : = p ( 1 - p )$ . Then it holds that

$$
\mathbb { E } | \widehat { q } _ { n } - q | \leq \frac { \sqrt { n } } { 2 ( n + s _ { 0 } ) } + \frac { s _ { 0 } } { n + s _ { 0 } } + \frac { 1 } { 4 ( n + s _ { 0 } + 1 ) } .\tag{7}
$$

Consequently, $\widehat { q _ { n } }$ is consistent at rate $O ( n ^ { - 1 / 2 } )$ under a stationary Bernoulli model with a fixed, nonadaptive sample size

Proposition 2 concerns fixed-sample stationary calibration. Our allocation guarantees instead hold pathwise for arbitrary realized proxy sequences, with the bounds explicitly accounting for proxy error relative to the target rates.

## 4. CERO Algorithm

To solve the finite-horizon rollout-allocation problem, we develop CERO, an online primal–dual scheduler summarized in Algorithm 1. At each round, CERO maintains normalized prompt-specific supporting slopes and a normalized shared budget price:

$$
\theta _ { i } ^ { k } = \bar { c } \vartheta _ { i } ^ { k } , \quad \mu ^ { k } = \bar { c } \nu ^ { k } , \quad \vartheta _ { i } ^ { k } , \nu ^ { k } \in [ 0 , 1 ] .
$$

After generating the selected responses and observing their rewards, CERO updates the predictive state and dual variables for the next round.

Round-level allocation Since $\theta _ { i } ^ { k } - \mu ^ { k } = \bar { c } ( \vartheta _ { i } ^ { k } - \nu ^ { k } )$ , define the normalized margin $d _ { i , k } : = \vartheta _ { i } ^ { k } - \nu ^ { k }$ A positive margin indicates that prompt $x _ { i } \mathrm { { ' s } }$ supporting slope exceeds the shared budget price. CERO first computes the capacity-constrained virtual allocation

$$
\widetilde { N } _ { k } \in \arg \operatorname* { m a x } _ { n \in \mathcal { P } _ { C } \cap \mathbb { Z } ^ { M } } \sum _ { i = 1 } ^ { M } d _ { i , k } n _ { i } .\tag{8}
$$

This problem has a simple exact solution: prompts with positive margins are ranked in decreasing order, with up to $N _ { \mathrm { m a x } }$ units assigned to each while capacity remains. CERO then allocates rollout units in this order up to the remaining global budget, which is updated as $\begin{array} { r } { B _ { k + 1 } = B _ { k } - \sum _ { i } N _ { i , k } } \end{array}$

Since the round-level objective is linear in the allocation units, the ranked solution assigns selected prompts the per-prompt cap, except possibly for a residual allocation. In Section 6, we use $N _ { \mathrm { m a x } } = 2$ and choose the round capacity and total budget as multiples of $N _ { \mathrm { m a x } } .$ , so every selected prompt receives one eight-response group. CERO therefore adapts which prompts receive groups, how often they are revisited across rounds, and how many groups are generated per round.

Reward feedback and supporting exposure After observing the rewards from round k, CERO updates the Beta state using eq. (5) and recomputes the reward-variation proxy $\widehat { q } _ { i , k + 1 }$ using eq. (6). Let $\begin{array} { r } { \widehat { U } _ { i , k + 1 } ( z ) : = U ( \widehat { q } _ { i , k + 1 } , z ) , c _ { i , k + 1 } : = \frac { 4 \lambda \widehat { q } _ { i , k + 1 } } { R } } \end{array}$ . For a supporting slope $\theta ,$ the corresponding exposure is

$$
\widehat { z } _ { i , k + 1 } ^ { \ast } ( \theta ) \in \arg \operatorname* { m i n } _ { 0 \leq z \leq z } \left\{ \theta z - \widehat { U } _ { i , k + 1 } ( z ) \right\} .
$$

For our exponential utility, this minimizer has the closed form $\hat { z } _ { i , k + 1 } ^ { * } ( \theta ) = \hat { z } ^ { * } ( \theta ; c _ { i , k + 1 } )$ , where

$$
\hat { z } ^ { * } ( \theta ; c ) = \left\{ \begin{array} { l l } { 0 , } & { c = 0 \mathrm { ~ o r ~ } \theta \geq c , } \\ { Z , } & { c > 0 \mathrm { ~ a n d ~ } 0 \leq \theta \leq c e ^ { - c Z } , } \\ { \frac { 1 } { c } \log \frac { c } { \theta } , } & { c > 0 \mathrm { ~ a n d ~ } c e ^ { - c Z } < \theta < c . } \end{array} \right.\tag{9}
$$

Algorithm 1 CERO: Cross-Epoch Adaptive Rollout Optimization   
Require: Prompt pool $\{ x _ { i } \} _ { i = 1 } ^ { M } ;$ horizon K; global budget $B ;$ round capacity $C ;$ per-prompt cap $N _ { \mathrm { m a x } } .$   
1: Set $R = B / M , Z = K N _ { \operatorname* { m a x } } , \bar { c } = \lambda / R , \bar { B } = B / K , \mathrm { a n d } G _ { B } = \operatorname* { m a x } \{ \bar { B } , C - \bar { B } \}$   
2: Initialize $B _ { 1 } = B , ( a _ { i , 1 } , b _ { i , 1 } ) = ( a _ { 0 } , b _ { 0 } ) , { \widehat { q } } _ { i , 1 }$ by (6), $\vartheta _ { i } ^ { 1 } = 4 \widehat { q } _ { i , 1 }$ , and $\nu ^ { 1 } \in [ 0 , 1 ]$   
3: for $k = 1 , \ldots , K$ do   
4: Compute margins $d _ { i , k } = \vartheta _ { i } ^ { k } - \nu ^ { k }$ and virtual proposal $\widetilde { N } _ { k }$ by the top-score solution of (8)   
5: Execute proposals in decreasing-margin order subject to remaining budget $B _ { k } ,$ producing $N _ { k }$   
6: for $i = 1 , \ldots , M$ do   
7: If $N _ { i , k } > 0 ,$ generate $N _ { i , k } ^ { \mathrm { r e s p } } = g N _ { i , k }$ responses from $\pi _ { \phi ^ { k } } ( \cdot \mid x _ { i } )$ and observe $S _ { i , k }$ successes and $F _ { i , k }$   
failures; otherwise, set $S _ { i , k } = F _ { i , k } = 0 .$   
8: Update $\left( a _ { i , k + 1 } , b _ { i , k + 1 } \right)$ by (5) and compute $\widehat { q } _ { i , k + 1 }$   
9: Compute $\widehat { z } _ { i , k + 1 } ^ { * } \big ( \bar { c } \vartheta _ { i } ^ { k } \big )$ by (9)   
10: Update $\vartheta _ { i } ^ { k + 1 }$ by (10)   
11: end for   
12: Update $\nu ^ { k + 1 }$ by (10) and set $\begin{array} { r } { B _ { k + 1 } = B _ { k } - \sum _ { i } N _ { i , k } } \end{array}$   
13: If $\textstyle \sum _ { i } N _ { i , k } > 0$ , update the policy using all executed round-k groups; otherwise, set $\phi ^ { k + 1 } = \phi ^ { k }$   
14: if $B _ { k + 1 } = 0$ then   
15: Terminate.   
16: end if   
17: end for

Dual updates Let $G _ { B } : = \operatorname* { m a x } \{ \bar { B } , C - \bar { B } \}$ denote the normalization scale for the budget-price update. CERO applies the normalized projected updates

$$
\vartheta _ { i } ^ { k + 1 } = \Pi _ { [ 0 , 1 ] } \left[ \vartheta _ { i } ^ { k } - \gamma _ { \vartheta } \frac { \widetilde { N } _ { i , k } - \widehat z _ { i , k + 1 } ^ { * } ( \bar { c } \vartheta _ { i } ^ { k } ) / K } { N _ { \mathrm { m a x } } } \right] , \ : \ : \nu ^ { k + 1 } = \Pi _ { [ 0 , 1 ] } \left[ \nu ^ { k } - \gamma _ { \nu } \frac { \bar { B } - \sum _ { i } \widetilde { N } _ { i , k } } { G _ { B } } \right]\tag{10}
$$

The prompt-specific update compares the virtual allocation $\widetilde { N } _ { i , k }$ with the per-round supporting exposure $\hat { z } _ { i , k + 1 } ^ { * } \big ( \bar { c } \vartheta _ { i } ^ { k } \big ) / K$ : allocating above this level decreases the supporting slope, while allocating below it increases the slope. The shared price increases when total virtual allocation exceeds B<sup>¯</sup> and decreases when it falls below B<sup>¯</sup>. Both updates use the virtual allocations $\widetilde { N _ { k } }$ rather than the budget-clipped allocations. Overall, the prompt-specific dual variables adapt relative prompt priority, while the shared price regulates budget pacing across rounds.

## 5. Performance Guarantees

We now analyze CERO with respect to the surrogate allocation objective in Section 2, while the downstream policy performance is evaluated empirically in Section 6.

Fixed-rate benchmark. Let $\mathbf { N } ^ { C E R O }$ be the executed allocation and define

$$
\mathsf { A L G } _ { K } ^ { \mathrm { s t a t } } ( \pmb q ) : = \sum _ { i = 1 } ^ { M } U ( \pmb q _ { i } , \boldsymbol z _ { i , K } ^ { C E R O } ) , \qquad \mathsf { R e g } _ { K } ^ { \mathrm { s t a t } } : = \mathsf { O P T } _ { K } ^ { \mathrm { s t a t } } ( \pmb q ) - \mathsf { A L G } _ { K } ^ { \mathrm { s t a t } } ( \pmb q ) .
$$

Let $\begin{array} { r } { \bar { N } = \frac { B } { K M } = \frac { R } { K } } \end{array}$ denote the average per-prompt per-round budget. Let $\widehat { q } _ { i , k + 1 }$ be the post-feedback proxy used in round-k prompt-specific dual update and define the average absolute proxy discrepancy

$$
\displaystyle \bar { e } _ { K } ^ { \mathrm { s t a t } } : = \frac { 1 } { M K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { M } | \widehat { q } _ { i , k + 1 } - q _ { i } | .
$$

Theorem 1 (Fixed-rate pathwise guarantee). Assume eq. (2). For any fixed target vector $\pmb { q } \in [ 0 , 1 / 4 ] ^ { M }$ and any realized proxy sequence $\{ \hat { q } _ { i , k + 1 } \}$ , Algorithm 1 with $\begin{array} { r } { \gamma _ { \vartheta } = \gamma _ { \nu } = \frac { 1 } { \sqrt { K } } } \end{array}$ satisfies

$$
\frac { \mathrm { R e g } _ { K } ^ { \mathrm { s t a t } } } { M } \leq \operatorname* { m i n } \left\{ 1 , ~ \lambda \frac { N _ { \operatorname* { m a x } } + G _ { B } / M } { \bar { N } } \frac { 1 } { \sqrt { K } } + 8 \lambda \frac { N _ { \operatorname* { m a x } } } { \bar { N } } \bar { e } _ { K } ^ { \mathrm { s t a t } } \right\} .\tag{11}
$$

This pathwise bound separates online optimization error from reward-proxy discrepancy. In particular, if $N _ { \mathrm { m a x } } / \bar { N } , G _ { B } / ( M \bar { N } )$ , and λ remain bounded and $\bar { e } _ { K } ^ { \mathrm { s t a t } } = O ( K ^ { - 1 / 2 } )$ , then the regret per prompt vanishes at rate $O ( K ^ { - 1 / 2 } )$ . The proof is given in Appendix D.

Time-varying rates and a same-path benchmark. For a realized rate path $\pmb { q } _ { 1 : K }$ , define the value of round-k exposure using the current rate:

$$
\Delta U _ { i , k } ( { \bf N } ; q _ { i , k } ) : = U ( q _ { i , k } , z _ { i , k } ) - U ( q _ { i , k } , z _ { i , k - 1 } ) .
$$

Using the same $q _ { i , k }$ in both terms ensures that $\Delta U _ { i , k }$ reflects only the utility gained from the new exposure $N _ { i , k } .$ , rather than changes due to variation in the rate itself. Writing $\pmb { q } _ { 1 : K } = ( \pmb { q } _ { 1 } , \dots , \pmb { q } _ { K } )$ ， the same-path ofline benchmark evaluates every feasible allocation plan against the same realized rate path:

$$
\mathsf { O P T } _ { K } ^ { \mathrm { d y n } } ( \pmb { q } _ { 1 : K } ) : = \operatorname* { m a x } _ { \mathbf { N } \in \mathscr { N } _ { B , C } } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { M } \Delta U _ { i , k } ( \mathbf { N } ; \pmb { q } _ { i , k } ) .
$$

Define $\mathsf { A L G } _ { K } ^ { \mathrm { d y n } }$ by evaluating the executed CERO allocation in the same way, and let $\mathrm { R e g } _ { K } ^ { \mathrm { d y n } } =$ ${ \mathsf { O P T } } _ { K } ^ { \mathrm { d y n } } - { \mathsf { A L G } } _ { K } ^ { \mathrm { d y n } }$ . By holding the realized rate path fixed across allocation plans, the benchmark focuses specifically on the quality of allocation decisions along a common realized rate path. Set

$$
\bar { e } _ { K } ^ { \mathrm { d y n } } : = \frac { 1 } { M K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { M } \big | \widehat { q } _ { i , k + 1 } - q _ { i , k } \big | , ~ V _ { K } : = \sum _ { k = 1 } ^ { K - 1 } \lVert \pmb { q } _ { k + 1 } - \pmb { q } _ { k } \rVert _ { \infty } .
$$

Theorem 2 (Same-path drifting-rate guarantee). Under the same algorithmic setup and step sizes as in Theorem 1, for every realized rate path $\pmb { q } _ { 1 : K }$ and proxy sequence $\{ \widehat { q } _ { i , k + 1 } \}$ , Algorithm 1 satisfies

$$
\frac { \mathrm { R e g } _ { K } ^ { \mathrm { d y n } } } { M } \leq \operatorname* { m i n } \left\{ \lambda , \ \lambda \frac { N _ { \operatorname* { m a x } } + G _ { B } / M } { \bar { N } } \frac { 1 } { \sqrt { K } } + 8 \lambda \frac { N _ { \operatorname* { m a x } } } { \bar { N } } \bar { e } _ { K } ^ { \mathrm { d y n } } + 4 \lambda \left( 1 + \frac { N _ { \operatorname* { m a x } } } { \bar { N } } \right) V _ { K } \right\} .\tag{12}
$$

Comparing with Theorem 1, the additional $V _ { K }$ term explicitly isolates the efect of temporal drift from reward-proxy discrepancy. The proof is given in Appendix E.

## 6. Experiments

## 6.1. Main Results

Experimental setup. We evaluate DeepSeek-R1-Distill-Qwen-1.5B (Guo et al. 2025), Qwen3-4B-Base (Yang et al. 2025), and Qwen2.5-7B-Instruct (Qwen et al. 2025), training all methods on the same uniformly sampled 50% subset of DAPO-Math-17K. Each run spans K = 500 scheduler rounds with a training-response budget of $B ^ { \mathrm { r e s p } } = 2 5 6 , 0 0 0$ , and CERO uses eight responses per admitted prompt, with adaptation occurring through prompt admission and per-round group counts. We evaluate on AIME, AMC, MATH, MINERVA, and OlympiadBench using 16 responses per problem. We report avg@16 as the mean binary reward across responses and problems within each benchmark, and macro-average the five benchmark scores. Implementation details and compute measurements are provided in Appendices G and B.4, respectively.

Baselines. We evaluate four methods: (i) GRPO (Guo et al. 2025), a widely used reinforcement learning method for training reasoning models; (ii) KnapsackRL (Li et al. 2025), a representative prior method for adaptive rollout allocation, which dynamically allocates diferent numbers of rollouts across prompts; (iii) CERO, our proposed method; and (iv) Random-matched CERO, an ablation that preserves the rollout allocation produced by CERO while assigning it to a random permutation of prompts. The latter controls for the allocation pattern and isolates the efect of our prompt selection mechanism.

Table 1 shows that CERO attains avg@16 macro-averages of 32.81, 45.36, and 41.81, improving over GRPO by 2.62, 2.10, and 1.59 points, respectively. It also achieves a higher macro-average than the adaptive KnapsackRL baseline on all three backbones, with the clearest improvement on DSR1-Q1.5B and smaller margins on the larger models. The Random-matched control further isolates the contribution of prompt-aware assignment. Its macro-average falls below CERO by 3.63, 1.50, and 1.63 points across the three backbones. Since Random-matched CERO preserves allocation sizes and timing while randomizing prompt assignment, this comparison supports the contribution of prompt-aware assignment under the same rollout schedule.

## 6.2. Ablation and Mechanism Analysis

Is the reward-variation proxy informative? We examine whether the proxy anticipates the reward contrast available in future rollout groups. On the DSR1-Q1.5B backbone, we pair each pre-decision prediction $\hat { q } _ { i , k }$ with the unbiased sample variance of the eight binary rewards generated afterward, covering 32, 000 selected prompt–round groups. Across ten equal-budget intervals, higher predicted means tend to coincide with greater observed reward variation (Pearson $r = 0 . 7 6 9$ ; Figure 2), showing that the proxy tracks changes in reward contrast during training. Across seven prediction bins, higher-prediction bins generally exhibit larger observed mean variance, with closely preserved relative ordering (Pearson $r = 0 . 9 7 1$ , Spearman $\rho = 0 . 9 6 4$ ; Figure 6). Together, these analyses provide evidence that the proxy ofers an informative pre-decision signal for subsequent reward contrast among selected prompt–round groups. Appendix B.2 further examines absolute calibration.

Table 1 Evaluation performance (avg@16) comparison across diferent models and benchmarks.
<table><tr><td>Base Model</td><td>Method</td><td>AIME</td><td>AMC</td><td>MATH</td><td>MINERVA</td><td>Olympiad</td><td>Avg.</td></tr><tr><td rowspan="4">DSR1-Q1.5B</td><td>GRPO</td><td>5.62</td><td>32.23</td><td>66.12</td><td>21.90</td><td>25.07</td><td>30.19</td></tr><tr><td>KnapsackRL</td><td>4.58</td><td>32.38</td><td>65.14</td><td>20.93</td><td>24.59</td><td>29.53</td></tr><tr><td>CERO (ours)</td><td>7.92</td><td>38.03</td><td>68.05</td><td>22.66</td><td>27.41</td><td>32.81</td></tr><tr><td>Random-matched CERO</td><td>4.17</td><td>31.02</td><td>64.54</td><td>21.48</td><td>24.69</td><td>29.18</td></tr><tr><td rowspan="4">Qwen3-4B</td><td>GRPO</td><td>13.54</td><td>50.98</td><td>80.10</td><td>30.72</td><td>40.94</td><td>43.26</td></tr><tr><td>KnapsackRL</td><td>17.50</td><td>53.24</td><td>80.08</td><td>31.36</td><td>41.26</td><td>44.69</td></tr><tr><td>CERO (ours)</td><td>17.29</td><td>53.69</td><td>80.83</td><td>31.94</td><td>43.06</td><td>45.36</td></tr><tr><td>Random-matched CERO</td><td>17.50</td><td>50.00</td><td>79.13</td><td>32.84</td><td>39.82</td><td>43.86</td></tr><tr><td rowspan="4">Q2.5-7B-Inst</td><td>GRPO</td><td>12.92</td><td>48.42</td><td>74.36</td><td>28.06</td><td>37.34</td><td>40.22</td></tr><tr><td>KnapsackRL</td><td>13.33</td><td>50.90</td><td>77.15</td><td>28.72</td><td>38.64</td><td>41.75</td></tr><tr><td>CERO (ours)</td><td>15.83</td><td>49.77</td><td>77.44</td><td>27.55</td><td>38.45</td><td>41.81</td></tr><tr><td>Random-matched CERO</td><td>11.88</td><td>46.23</td><td>76.35</td><td>29.30</td><td>37.17</td><td>40.18</td></tr></table>

![](images/9d5c1a57a057e8223c45add6c692aea1b0d3f0f2c1527be0ebc11ced863c64b0.jpg)  
Figure 2 Temporal tracking of CERO’s reward-variation proxy on DSR1-Q1.5B.

![](images/4a305a950a3ec8d6b8b76ab6f33b1e131c0c44ec37944c4efcbc536e551f9381.jpg)  
Figure 3 Prompt-selection lift by pre-decision predicted success rate.

Does CERO prioritize informative prompts? We characterize CERO’s selection pattern by partitioning the 500 scheduler rounds into five 100-round stages and binning candidate prompt–round pairs by their pre-decision predicted success rate. For bin b in stage t, we define the selection lift as

$$
L _ { b , t } = \frac { n _ { b , t } ^ { \mathrm { s e l } } / n _ { b , t } ^ { \mathrm { c a n d } } } { n _ { t } ^ { \mathrm { s e l } } / n _ { t } ^ { \mathrm { c a n d } } } ,
$$

where $n ^ { \mathrm { s e l } }$ and $n ^ { \mathrm { c a n d } }$ denote the numbers of selected and candidate prompt–round pairs. Consequently, $L _ { b , t } = 1$ corresponds to uniform random selection, while $L _ { b , t } > 1$ and $L _ { b , t } < 1$ indicate overselection and under-selection respectively. Figure 3 shows that CERO preferentially selects prompts with intermediate predicted success rates. Over the full run, the [0.4, 0.6) bin has a 1.72× selection lift, while both extreme bins have approximately 0.65×. With the full prompt pool available in every round, the lift directly measures CERO’s selection preference across predicted-success ranges relative to uniform selection. This pattern aligns with CERO’s reward-variation-based design: prompts for which both success and failure remain plausible ofer opportunities for the within-group reward contrast needed by group-relative learning. The strength and breadth of the preference vary across training stages, with all three interior bins over-selected in the final stage.

![](images/ecfa8dd19317b5fd2da5d539cbedeb4d9b5b1e7cb7f02e828a44fd26f48713e6.jpg)

![](images/01745f824cf19e03a147950a420d3c77d2b62e4c259efaad843849d8c32df77b.jpg)  
Figure 4 Efective-group rate over consumed rollout budget.  
Figure 5 Cross-epoch budget pacing in CERO. Epoch boundaries are marked above trajectory.

Does prompt-aware admission increase the fraction of efective groups? Let $S _ { i , k }$ of $G _ { i , k }$ responses in group (i, k) be correct. A group is efective if $0 < S _ { i , k } < G _ { i , k }$ . Define the following efective-group rate

$$
\mathrm { E G R } : = \frac { \sum _ { i , k } { \bf 1 } \{ 0 < S _ { i , k } < G _ { i , k } \} } { \sum _ { i , k } { \bf 1 } \{ G _ { i , k } > 0 \} } ,
$$

the fraction of executed groups supplying nonzero reward contrast. Figure 4 shows that under the matched budget of 256, 000 training responses, CERO produces 18, 347 efective groups out of 32, 000 executed groups, yielding an EGR of 57.3%, compared with 52.7% for Random-matched CERO and 48.7% for KnapsackRL. Because Random-matched CERO preserves CERO’s group sizes and spending schedule while randomizing prompt assignment, the 4.6-point gap provides controlled evidence that prompt-aware admission increases the fraction of mixed-outcome groups under the same rollout schedule.

What do dual-based selection and adaptive pacing contribute? We first characterize CERO’s budget pacing through the cumulative response gap from a uniform spending schedule:

$$
G _ { k } ^ { \mathrm { e x e c } } : = \sum _ { \tau = 1 } ^ { k } \sum _ { i } N _ { i , \tau } ^ { \mathrm { r e s p } } - k \bar { B } ^ { \mathrm { r e s p } } , \qquad \bar { B } ^ { \mathrm { r e s p } } : = \frac { B ^ { \mathrm { r e s p } } } { K } = 5 1 2 .
$$

Positive and negative gaps indicate spending ahead of and behind the uniform schedule, respectively. Figure 5 shows that spending deviations persist across data-epoch boundaries. The cumulative surplus peaks at 1,920 responses and returns to zero at the end of training, exhausting the 256,000-response budget exactly. This trajectory characterizes CERO’s nonuniform budget consumption.

Table 2 evaluates the contributions of selection and pacing. CERO-fixed retains CERO’s rewardproxy and prompt-dual updates but fixes $\nu = 0$ and allocates 512 responses per round. CEROpreset follows a predetermined front-loaded schedule: 528 responses per round for the first 250 rounds and 496 for the remaining 250 rounds. CERO-greedy retains CERO-fixed’s proxy estimator, utility function, and per-round quota, but selects prompts by the marginal utility of an additional complete group rather than by prompt-dual scores.

Across three training seeds, CERO achieves the highest macro-average of $3 2 . 2 6 \pm 0 . 9 4$ , outperforming CEROfixed, CERO-preset, and CERO-greedy by 1.13, 1.55, and 1.29 points on average, respectively. The gains over CEROfixed and CERO-preset support the value of adapting budget consumption over the training horizon rather than using a fixed or predetermined spending schedule. Under the fixed 512-response quota, dual-guided and direct marginalutility ranking yield similar mean scores of 31.13 and 30.97, whereas full CERO improves on both by more than one point. Together, these results support CERO’s central de-

Table 2: Prompt-selection and pacing ablations on DSR1-Q1.5B. Avg. denotes the five-benchmark avg@16 macro-average (mean ± std over 3 seeds).
<table><tr><td>Method</td><td>Avg.</td></tr><tr><td>CERO</td><td> ${ \bf 3 2 . 2 6 \pm 0 . 9 4 }$ </td></tr><tr><td>CERO-fixed</td><td> $3 1 . 1 3 \pm 0 . 7 2$ </td></tr><tr><td>CERO-greedy</td><td> $3 0 . 9 7 \pm 0 . 3 8$ </td></tr><tr><td>CERO-preset</td><td> $3 0 . 7 1 \pm 0 . 5 4$ </td></tr></table>

sign: jointly coordinating prompt prioritization and budget pacing over the training horizon.

## 7. Conclusion

We introduced CERO, a finite-horizon scheduler that coordinates prompt admission, repeated selection, and budget pacing under a shared rollout budget. The Fenchel-based primal–dual formulation yields tractable round-level decisions and pathwise guarantees for the surrogate allocation objective under fixed-rate and same-path time-varying benchmarks. Under matched training-response budgets, CERO achieves the highest avg@16 macro-average on all three backbones. Mechanistic analyses link its prompt choices to stronger within-group supervision, while multi-seed ablations show gains from adaptive pacing over fixed and preset schedules. Together, these results demonstrate the practical value of horizon-level rollout coordination for eficient RL post-training.

## References

Shipra Agrawal and Nikhil R Devanur. Fast algorithms for online stochastic convex programming. In Proceedings of the twenty-sixth annual ACM-SIAM symposium on Discrete algorithms, pages 1405– 1424. SIAM, 2014.

Shipra Agrawal, Zizhuo Wang, and Yinyu Ye. A dynamic near-optimal algorithm for online linear programming. Operations Research, 62(4):876–890, 2014.

Santiago Balseiro, Haihao Lu, and Vahab Mirrokni. Dual mirror descent for online allocation problems. In International Conference on Machine Learning, pages 613–628. PMLR, 2020.

Niv Buchbinder and Joseph Naor. Online primal-dual algorithms for covering and packing problems. In European Symposium on Algorithms, pages 689–701. Springer, 2005.

Niv Buchbinder and Joseph Naor. Improved bounds for online routing and packing via a primal-dual approach. In 2006 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS’06), pages 293–304. IEEE, 2006.

Xiao Alison Chen and Zizhuo Wang. A dynamic learning algorithm for online matching problems with concave returns. European Journal of Operational Research, 247(2):379–388, 2015.

Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Pich´e, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025.

Nikhil R Devanur and Thomas P Hayes. The adwords problem: online keyword matching with budgeted bidders under random permutations. In Proceedings of the 10th ACM conference on Electronic commerce, pages 71–78, 2009.

Nikhil R Devanur and Kamal Jain. Online matching with concave returns. In Proceedings of the forty-fourth annual ACM symposium on Theory of computing, pages 137–144, 2012.

Jon Feldman, Monika Henzinger, Nitish Korula, Vahab S Mirrokni, and Clif Stein. Online stochastic packing applied to display ad allocation. In European Symposium on Algorithms, pages 182–194. Springer, 2010.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Zengjie Hu, Jiantao Qiu, Tianyi Bai, Haojin Yang, Binhang Yuan, Qi Jing, Conghui He, and Wentao Zhang. Vade: Variance-aware dynamic sampling via online sample-level dificulty estimation for multimodal rl. arXiv preprint arXiv:2511.18902, 2025.

Stefanus Jasin and Sunil Kumar. A re-solving heuristic with bounded revenue loss for network revenue management with customer choice. Mathematics of Operations Research, 37(2):313–345, 2012.

Heyang Jiang, Henry Liu, and Baharan Mirzasoleiman. Learning as reasoning unfolds: Progressive rollout allocation for eficient reinforcement learning. arXiv preprint arXiv:2607.22002, 2026.

Jiashuo Jiang, Will Ma, and Jiawei Zhang. Degeneracy is ok: Logarithmic regret for network revenue management with indiscrete distributions. Operations Research, 73(6):3405–3420, 2025.

Thomas Kesselheim, Andreas T¨onnis, Klaus Radke, and Berthold V¨ocking. Primal beats dual on online packing lps in the random-order model. In Proceedings of the forty-sixth annual ACM symposium on Theory of computing, pages 303–312, 2014.

Woojeong Kim, Ziyi Yang, Jing Nathan Yan, and Jialu Liu. Spend your rollouts where it counts: Rollout allocation for group-based rl post-training. arXiv preprint arXiv:2605.26606, 2026.

Xiaocheng Li and Yinyu Ye. Online linear programming: Dual convergence, new algorithms, and regret bounds. Operations Research, 70(5):2948–2966, 2022.

Ziniu Li, Congliang Chen, Tianyun Yang, Tian Ding, Ruoyu Sun, Ge Zhang, Wenhao Huang, and Zhi-Quan Luo. Knapsack rl: Unlocking exploration of llms via optimizing budget allocation. arXiv preprint arXiv:2509.25849, 2025.

Mengqi Liao, Xiangyu Xi, Chen Ruinian, Jia Leng, Yangen Hu, Ke Zeng, Shuai Liu, and Huaiyu Wan. Enhancing eficiency and exploration in reinforcement learning for LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1451–1463, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/ v1/2025.emnlp-main.75. URL https://aclanthology.org/2025.emnlp-main.75/.

Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. Cppo: Accelerating the training of group relative policy optimization-based reasoning models. arXiv preprint arXiv:2503.22342, 2025.

Ilia Mahrooghi, Aryo Lotfi, and Emmanuel Abbe. Goldilocks rl: Tuning task dificulty to escape sparse rewards for reasoning. arXiv preprint arXiv:2602.14868, 2026.

Aranyak Mehta, Amin Saberi, Umesh Vazirani, and Vijay Vazirani. Adwords and generalized online matching. Journal of the ACM (JACM), 54(5):22–es, 2007.

Marco Molinaro and Ramamoorthi Ravi. The geometry of online packing linear programs. Mathematics of Operations Research, 39(1):46–59, 2014.

Trung Hieu Nguyen, Bao Nguyen, Wenao Ma, Yuzhi Zhao, Ruifeng She, and Viet Anh Nguyen. Adaptive rollout allocation for online reinforcement learning with verifiable rewards. In International Conference on Learning Representations, volume 2026, pages 136100–136132, 2026.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/ 2412.15115.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

Vaishnavi Shrivastava, Ahmed Awadallah, Vidhisha Balachandran, Shivam Garg, Harkirat Behl, and Dimitris Papailiopoulos. Sample more to think less: Group filtered policy optimization for concise reasoning. arXiv preprint arXiv:2508.09726, 2025.

Kalyan T Talluri and Garrett J Van Ryzin. The theory and practice of revenue management, volume 68. Springer Science & Business Media, 2006.

Sijun Tan, Michael Luo, Justin Wong, Colin Cai, Xiaoxiang Shi, William Yuan Tang, Manan Roongta, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, et al. Deepscaler: Efective rl scaling of reasoning models via iterative context lengthening. 2025.

Tao Wang, Shuo Li, Yan Sun, Dongsheng Ding, and Edgar Dobriban. Where to spend rollouts: Hit-utility optimal rollout allocation for group-based rlvr. arXiv preprint arXiv:2605.07114, 2026.

Yixuan Even Xu, Yash Savani, Fei Fang, and J Zico Kolter. Not all rollouts are useful: Down-sampling rollouts in llm reinforcement learning. arXiv preprint arXiv:2504.13818, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zhiyuan Yao, Yi-Kai Zhang, Yuxin Chen, Yueqing Sun, Zishan Xu, Yu Yang, Tianhao Hu, Qi Gu, Hui Su, and Xunliang Cai. Coba-rl: Capability-oriented budget allocation for reinforcement learning in llms. arXiv preprint arXiv:2602.03048, 2026.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Ruiqi Zhang, Daman Arora, Song Mei, and Andrea Zanette. Speed-rl: Faster training of reasoning models via online curriculum learning. arXiv preprint arXiv:2506.09016, 2025a.

Yuheng Zhang, Wenlin Yao, Changlong Yu, Yao Liu, Qingyu Yin, Bing Yin, Hyokun Yun, and Lihong Li. Improving sampling eficiency in rlvr through adaptive rollout and response reuse. arXiv preprint arXiv:2509.25808, 2025b.

Zhi Zhang, Zhen Han, Costas Mavromatis, Qi Zhu, Yunyi Zhang, Sheng Guan, Dingmin Wang, Xiong Zhou, Shuai Wang, Soji Adeshina, et al. Train less, learn more: Adaptive eficient rollout optimization for group-based reinforcement learning. arXiv preprint arXiv:2602.14338, 2026.

## Appendix A: Related Work

Dificulty-aware sampling and filtering. Recent work has explored improving RL eficiency by prioritizing training examples and rollouts according to their dificulty or informativeness. At the query level, online curriculum and sample-selection methods dynamically adjust the training distribution to favor examples with greater learning value. Goldilocks RL (Mahrooghi et al. 2026) emphasizes problems of intermediate difficulty, while SPEED-RL (Zhang et al. 2025a), SEC (Chen et al. 2025), and VADE (Hu et al. 2025) similarly use online estimates of dificulty or training utility to prioritize informative samples. Related approaches operate after rollout generation by filtering or pruning trajectories with limited contribution to policy updates. CPPO (Lin et al. 2025) removes completions with low learning signals, PODS (Xu et al. 2025) selects informative subsets of generated responses, and GFPO (Shrivastava et al. 2025) filters rollouts based on their utility relative to generation cost. These methods collectively reduce computation spent on overly easy, overly dificult, redundant, or otherwise uninformative samples.

Adaptive rollout allocation. Recent studies have increasingly treated rollout generation as a resource allocation problem, replacing the fixed group size in standard GRPO (Shao et al. 2024) with query-dependent rollout budgets. Some methods adapt allocation according to estimated dificulty, assigning more rollouts to challenging or uncertain queries and fewer to those with saturated outcomes (Liao et al. 2025, Zhang et al. 2025b). Others formulate allocation through explicit estimates of training value, uncertainty, or marginal utility, including Knapsack RL (Li et al. 2025), CoBA-RL (Yao et al. 2026) and VIP (Nguyen et al. 2026). A related direction makes allocation decisions progressively from intermediate rollout outcomes: Pilot-Commi (Kim et al. 2026), AERO (Zhang et al. 2026), and VIGOR (Jiang et al. 2026) use early samples to determine whether additional generation is worthwhile. Collectively, these methods improve rollout eficiency by direct ing computation toward queries where additional exploration is expected to provide the greatest learning value.

Online resource allocation. Online resource allocation is a central topic in the online optimization literature and has been applied in many operational settings, including ad assignment (Mehta et al. 2007), revenue management (Talluri and Van Ryzin 2006, Jasin and Kumar 2012, Jiang et al. 2025), and online packing/covering problems (Buchbinder and Naor 2005, 2006, Feldman et al. 2010). The existing algorithmic literature can be broadly organized according to how it exploits dual information. A dominant class of methods uses dual prices as signals for making online allocation decisions. Within this class, some papers adopt a learn-once strategy: they reserve an initial set of arrivals for estimation, compute a price vector from that sample, and then use the same prices for all future decisions (Devanur and Hayes 2009, Feldman et al. 2010, Molinaro and Ravi 2014, Devanur and Jain 2012). Other papers advocate adaptive updating, where the pricing rule is revised at multiple points over time through repeated re-solving, leading to stronger robustness and improved horizon dependence (Agrawal et al. 2014, Li and Ye 2022, Chen and Wang 2015). In parallel, other works depart from this paradigm by developing primal-oriented algorithms (Kesselheim et al. 2014) or first-order schemes that do not require repeated computation of dual prices (Agrawal and Devanur 2014, Balseiro et al. 2020).

Table 3 Evaluation performance (pass@16) comparison across diferent models and benchmarks.
<table><tr><td>Base Model</td><td>Method</td><td>AIME</td><td>AMC</td><td>MATH</td><td>MINERVA</td><td>Olympiad</td><td>Avg.</td></tr><tr><td rowspan="4">DSR1-Q1.5B</td><td>GRPO</td><td>20.00</td><td>55.42</td><td>83.40</td><td>46.69</td><td>40.89</td><td>49.28</td></tr><tr><td>KnapsackRL</td><td>20.00</td><td>56.63</td><td>82.80</td><td>42.65</td><td>40.59</td><td>48.53</td></tr><tr><td>CERO (ours)</td><td>20.00</td><td>62.65</td><td>84.20</td><td>44.85</td><td>43.26</td><td>50.99</td></tr><tr><td>Random-matched CERO</td><td>13.33</td><td>51.81</td><td>81.40</td><td>45.96</td><td>39.70</td><td>46.44</td></tr><tr><td rowspan="4">Qwen3-4B</td><td>GRPO</td><td>40.00</td><td>79.52</td><td>92.00</td><td>51.10</td><td>62.67</td><td>65.06</td></tr><tr><td>KnapsackRL</td><td>46.67</td><td>80.72</td><td>92.20</td><td>51.84</td><td>64.00</td><td>67.09</td></tr><tr><td>CERO (ours)</td><td>40.00</td><td>83.13</td><td>92.20</td><td>52.21</td><td>59.70</td><td>65.45</td></tr><tr><td>Random-matched CERO</td><td>43.33</td><td>73.49</td><td>92.00</td><td>51.10</td><td>61.04</td><td>64.19</td></tr><tr><td rowspan="4">Q2.5-7B-Inst</td><td>GRPO</td><td>33.33</td><td>73.49</td><td>91.40</td><td>47.79</td><td>57.78</td><td>60.76</td></tr><tr><td>KnapsackRL</td><td>30.00</td><td>74.70</td><td>92.00</td><td>48.53</td><td>60.30</td><td>61.10</td></tr><tr><td>CERO (ours)</td><td>36.67</td><td>78.31</td><td>89.40</td><td>48.53</td><td>58.81</td><td>62.34</td></tr><tr><td>Random-matched CERO</td><td>30.00</td><td>71.08</td><td>88.80</td><td>46.69</td><td>56.74</td><td>58.66</td></tr></table>

## Appendix B: Additional Results and Diagnostics

## B.1. Pass@16 Evaluation

In addition to the avg@16 results reported in the main text, we evaluate pass@16 performance to measure success coverage under repeated sampling. Pass@16 is the percentage of problems with at least one correct response among these 16 samples:

$$
\mathrm { p a s s @ 1 6 } = \frac { 1 0 0 } { | \mathcal { D } | } \sum _ { x \in \mathcal { D } } \mathbf { 1 } \left\{ \sum _ { j = 1 } ^ { 1 6 } r _ { x , j } > 0 \right\} .
$$

For binary rewards, this is equivalent to averaging the maximum reward among the 16 responses for each problem. Table 3 shows that CERO improves the pass@16 macro-average over GRPO on all three backbones, with gains of 1.71, 0.39, and 1.58 points. It achieves the highest macro-average on DeepSeek-R1-Distill-Qwen-1.5B and Qwen2.5-7B-Instruct, reaching 50.99 and 62.34, respectively. On Qwen3-4B, KnapsackRL achieves the highest score of 67.09, compared with 65.45 for CERO. Together with the avg@16 results, these findings show improvements over GRPO in both average response accuracy and repeated-sampling success coverage.

## B.2. Reliability of the Reward-Variation Proxy

We examine whether CERO’s pre-decision reward-variation proxy captures diferences in subsequently observed group variance, both across prediction levels and over the training trajectory.

Diagnostic setup. We analyze 32,000 executed prompt–round groups from the DSR1-Q1.5B run. For each group, we pair the pre-decision prediction $\hat { q } _ { i , k }$ with the unbiased sample variance ${ v _ { i , k } } = \widehat { \mathrm { V a r } } _ { 8 } ( { r _ { i , k } } )$ of the eight binary rewards generated afterward. Figure 6 aggregates these pairs into seven nonempty prediction bins and plots the mean prediction $\bar { q } _ { \ell }$ against the mean observed variance $\bar { v } _ { \ell } .$ Vertical error bars show nominal 95% normal-approximation intervals,

$$
\bar { v } _ { \ell } \pm 1 . 9 6 \frac { s _ { \ell } } { \sqrt { n _ { \ell } } } ,
$$

where $n _ { \ell }$ is the number of groups in bin ℓ and $s _ { \ell }$ is the sample standard deviation of their observed group variances. These intervals use group-level standard errors within a single run, without adjustment for repeated-prompt or temporal dependence.

Association across prediction levels. Figure 6 shows a strong positive association between predicted and observed bin means, with Pearson correlation $r = 0 . 9 7 1$ and Spearman correlation $\rho = 0 . 9 6 4$ . Across the plotted prediction range, the observed mean group variance rises overall from approximately 0.03 to 0.20. Higher-prediction bins generally correspond to greater realized reward variation, and the ordering of the bin means is closely preserved despite a local fluctuation in the intermediate range. The proxy therefore distinguishes prediction ranges associated with substantially diferent levels of subsequent reward contrast. Because the predictions are recorded before the current-round responses are generated, this association reflects information available when CERO makes its allocation decisions.

Absolute calibration. The predicted values also exhibit an upward ofset in numerical scale. Across the selected groups, the mean predicted variance is 0.154, compared with a mean observed variance of 0.119. Consistent with this diference, the bin means in Figure 6 lie below the diagonal marking equality between predictions and observations. The strong correlations and the displacement from the diagonal characterize two distinct properties: the proxy captures diferences across prediction levels while tending to overestimate absolute reward variance. This distinction matters for CERO because $\hat { q } _ { i , k }$ enters the surrogate utility and prompt-dual updates numerically, rather than serving only as a ranking statistic. The diagnostic thus identifies both the information carried by the proxy and the calibration ofset present in the analyzed run.

Temporal tracking and allocation relevance. Figure 2 provides a complementary view by aggregating the same trajectory into ten consecutive equal-budget intervals of 3,200 groups each. Predicted and observed interval means co-vary over training, with Pearson correlation $r = 0 . 7 6 9$ . Their trajectories capture common decreases and rebounds, showing that the proxy reflects changes in reward variation as training progresses, in addition to the diferences across prediction levels shown in Figure 6. This behavior is relevant to group relative supervision: mixed-outcome groups provide positive reward variance, whereas all-correct and allincorrect groups provide no within-group reward contrast. Together, the prediction-bin and temporal analyses support $\hat { q } _ { i , k }$ as an informative prospective signal for CERO’s allocation mechanism. They complement the selection-lift and efective-group analyses in Section 6.2, which examine how the resulting allocation translates that signal into prompt choices and usable supervision.

## B.3. Cross-Round Prompt Exposure

To examine how CERO redistributes repeated exposure over the training horizon, we compare terminal cumulative prompt exposure with GRPO. Following Section 2, we normalize each prompt’s terminal exposure by the equal-share scale $R = B / M .$ . In Figure 7, the empirical CDF at x gives the fraction of prompts with $z _ { i , K } / R \leq x$ . Because both methods use fixed-size response groups and the same total rollout budget, diferences in terminal exposure reflect how many times each prompt is selected over the training horizon rather than diferences in group size or aggregate exposure.

Figure 7 shows that CERO induces a broader terminal-exposure distribution than GRPO. CERO spreads terminal exposure across three levels, with 57.5% of prompts at 0.84R, 27.8% at 1.12R, and a 14.7% highexposure tail at 1.40R. GRPO is more concentrated, with 41.0% and 58.1% of prompts at 0.84R and 1.12R, respectively. Thus, CERO shifts exposure away from the more concentrated GRPO pattern: a larger fraction of prompts receive below-equal-share exposure, while a smaller subset is revisited suficiently often to reach substantially above the equal-share level. Since both methods use the same total rollout budget, this broader distribution reflects a redistribution of repeated selections across prompts rather than additional aggregate exposure. In particular, CERO trades fewer revisits to some prompts for more persistent revisiting of others over the training horizon.

![](images/24ca25ef1d4652f0fc16ec9a311f2ae5178532168615404f4da109ce3a12dd83.jpg)  
Figure 6 Calibration of CERO’s reward-variation proxy on DSR1-Q1.5B. Points show predicted and observed bin means across seven prediction bins; vertical bars indicate nominal 95% within-run intervals for the observed means. The dashed diagonal marks perfect calibration.

## B.4. Computational Cost

We compare the training cost of CERO, Random-matched CERO, KnapsackRL, and vanilla GRPO. For every backbone, all methods use the same number of GPUs, 500 scheduler rounds, and a matched budget of 256,000 training responses. End-to-end GPU-hours are calculated as the Slurm elapsed time multiplied by the number of allocated GPUs. This measurement includes initialization, in-training validation, and checkpointing.

Figure 8 provides a detailed comparison for the 1.5B backbone. In addition to end-to-end GPU-hours, we report core training time normalized by the number of generated response tokens. The normalized measurement excludes validation and checkpointing but includes rollout generation, reward computation, log-probability computation, policy optimization, and allocation-related operations.

![](images/255058b5cbbb61a6d3c8e538c5d15d239ee5afe06ff7b57a4af23d145c2b44cf.jpg)  
Figure 7 Empirical CDF of normalized terminal prompt exposure $z _ { i , K } / R$ for CERO and GRPO on DSR1-Q1.5B. The dashed line marks the equal-share exposure $R = B / M .$

On the 1.5B backbone, CERO, Random-matched CERO, KnapsackRL, and vanilla GRPO consume 16.94, 17.15, 17.53, and 17.57 GPU-hours, respectively. Their token-normalized core costs are 84.2, 84.5, 83.6, and 84.2 seconds per million generated response tokens. The normalized measurements difer by only approximately 1.1% of their mean value, with CERO being nearly identical to vanilla GRPO. CERO therefore combines adaptive allocation with token-normalized core training cost comparable to the baselines in the 1.5B experiment.

We also measure end-to-end GPU-hours for the larger backbones. On Qwen3-4B-Base, CERO uses 36.31 GPU-hours, Random-matched CERO uses 36.78 GPU-hours, KnapsackRL uses 35.85 GPU-hours, and vanilla GRPO uses 36.89 GPU-hours. These values correspond to wall-clock times of 9.08, 9.19, 8.96, and 9.22 hours, respectively, using four H800 GPUs. The diferences among the four methods are within approximately 3% of their mean compute cost.

On Qwen2.5-7B-Instruct, CERO, Random-matched CERO, KnapsackRL, and vanilla GRPO use 31.46, 32.78, 34.99, and 31.72 GPU-hours, respectively. Their corresponding wall-clock times are 7.86, 8.19, 8.75, and 7.93 hours on four H800 GPUs. CERO therefore remains comparable to the baselines in end-to-end computational cost as the backbone size increases.

![](images/913d6e15472d0e69ce4cfdedbc931189d19d7e11030e314835111eb96a28d286.jpg)

![](images/9588fe29110c409c1728a041bca93f7a9a3d3760b083e5ef12733daf125565e6.jpg)  
Figure 8 Training-compute comparison on DeepSeek-R1-Distill-Qwen-1.5B under a matched response budget. The left panel reports end-to-end GPU-hours, while the right panel reports core training time normalized by generated response tokens.

The end-to-end measurements capture the practical cost of each complete training protocol under the matched response budget, including diferences in response lengths and update counts. CERO and Randommatched CERO perform 488, 486, and 487 policy updates on the 1.5B, 4B, and 7B backbones, respectively, while GRPO and KnapsackRL perform 500. Zero-allocation rounds account for the lower update counts.

Each cost measurement is obtained from one completed run per method and backbone. Across the reported runs, CERO combines adaptive prompt selection and cross-epoch budget pacing with comparable end-to-end training cost. On the 1.5B backbone, its token-normalized core cost matches GRPO at 84.2 seconds per million generated response tokens.

## Appendix C: Utility Properties and Full-Information Oracle

## C.1. Exact Conjugate, Fenchel Representation, and Stability

Fix $q \in [ 0 , 1 / 4 ]$ , set $c = 4 \lambda q / R ,$ and write $U _ { c } ( z ) = 1 - e ^ { - c z } ~ \mathrm { o n } ~ [ 0 , Z ]$ . Define

$$
U _ { c , Z } ^ { * } ( \theta ) : = \operatorname* { i n f } _ { 0 \leq z \leq z } \{ \theta z - U _ { c } ( z ) \} .\tag{13}
$$

Lemma 1 (Closed-form compact conjugate). $H c = 0$ , then $U _ { c , Z } ^ { * } ( \theta ) = 0$ for every $\theta \ge 0$ $I f c > 0$ , then

$$
\begin{array} { r } { U _ { c , z } ^ { * } ( \theta ) = \left\{ \begin{array} { l l } { \theta Z - 1 + e ^ { - c Z } , } & { 0 \leq \theta \leq c e ^ { - c Z } , } \\ { \theta } & { \frac { \displaystyle e } { c } \log \frac { c } { \theta } + \frac { \theta } { c } - 1 , } & { c e ^ { - c Z } < \theta < c , } \\ { 0 , } & { \theta \geq c . } \end{array} \right. } \end{array}\tag{14}
$$

The minimizer is given by (9).

F or $c > 0$ , diferentiate $\theta z - 1 + e ^ { - c z }$ . Its derivative is $\theta - c e ^ { - c z }$ and is strictly increasing in z. $\operatorname { I f } \theta \leq c e ^ { - c Z }$ 2 the objective is nonincreasing on $[ 0 , Z ]$ and the minimum is at Z. If $c e ^ { - c Z } < \theta < c ,$ the unique root is $z = c ^ { - 1 } \log ( c / \theta )$ . If $\theta \geq c _ { i }$ , the minimum is at 0. Substitution yields (14). The case $c = 0$ is immediate.

We also write $U _ { q } ^ { * } ( \theta ) : = U _ { 4 \lambda q / R , Z } ^ { * } ( \theta )$ for the compact conjugate parameterized by $q ;$ in particular, $U _ { i } ^ { * } ( \theta ) =$ $U _ { q _ { i } } ^ { * } ( \theta )$ in Section 2.3.

Lemma 2 (Exact compact biconjugate and stability). For every $q \in [ 0 , 1 / 4 ]$ and $z \in [ 0 , Z ]$

$$
U ( q , z ) = \operatorname* { m i n } _ { 0 \leq \theta \leq \bar { c } } \{ \theta z - U _ { q } ^ { * } ( \theta ) \} .\tag{15}
$$

Moreover, for $q , q ^ { \prime } \in [ 0 , 1 / 4 ]$ ，

$$
\operatorname* { s u p } _ { 0 \leq z \leq Z } | U ( q , z ) - U ( q ^ { \prime } , z ) | \leq \frac { 4 \lambda Z } { R } | q - q ^ { \prime } | ,\tag{16}
$$

$$
\operatorname* { s u p } _ { 0 \leq \theta \leq \overline { { c } } } | U _ { q } ^ { * } ( \theta ) - U _ { q ^ { \prime } } ^ { * } ( \theta ) | \leq \frac { 4 \lambda Z } { R } | q - q ^ { \prime } | .\tag{17}
$$

$T$ he slope of $U ( q , \cdot )$ lies in $[ 0 , \bar { c } ]$ . For a fixed z, choose the supporting slope $\theta _ { z } = \partial _ { z } U ( q , z )$ , with one-sided derivatives at the endpoints. Concavity gives

$$
\begin{array} { r } { U ( q , s ) \leq U ( q , z ) + \theta _ { z } ( s - z ) , } \end{array}\tag{18}
$$

so $U _ { q } ^ { * } ( \theta _ { z } ) = \theta _ { z } z - U ( q , z )$ . The reverse inequality follows directly from the definition of the conjugate.

For stability,

$$
| \partial _ { q } U ( q , z ) | = \frac { 4 \lambda z } { R } e ^ { - 4 \lambda q z / R } \leq \frac { 4 \lambda Z } { R } .\tag{19}
$$

The first bound follows from the mean-value theorem. Applying a common uniform perturbation before taking the infimum over $z \in [ 0 , Z ]$ gives the conjugate bound.

## C.2. Full-Information Oracle

We characterize the fixed-rate full-information benchmark when the target rates $q = ( q _ { 1 } , \dots , q _ { M } )$ are known. Since the objective depends only on terminal prompt exposures, the oracle can first determine the optimal exposure vector and then distribute these exposures across the K rounds.

Terminal-exposure formulation. Writing $\begin{array} { r } { z _ { i } = \sum _ { k = 1 } ^ { K } N _ { i , k } } \end{array}$ and $c _ { i } = 4 \lambda q _ { i } / R$ , the fixed-rate benchmark can be expressed as

$$
\mathrm { O P T } _ { K } ^ { \mathrm { s t a t } } ( q ) = \operatorname* { m a x } _ { z \in \mathbb { Z } _ { \geq 0 } ^ { M } } \sum _ { i = 1 } ^ { M } \left( 1 - e ^ { - c _ { i } z _ { i } } \right) .\tag{20}
$$

Every feasible rollout plan induces an exposure vector satisfying these constraints. Conversely, every such integer vector admits a feasible round-level realization, as shown below. For the continuous relaxation, the uniform assignment $N _ { i , k } = z _ { i } / K$ is feasible because $z _ { i } \le K N _ { \operatorname* { m a x } }$ and $\begin{array} { r } { \sum _ { i } z _ { i } \le B \le K C } \end{array}$

Continuous water-filling solution. Relaxing the integrality constraint gives a separable concave optimization problem. Let $\xi \ge 0$ be the multiplier for the total budget. For $c _ { i } > 0 .$ , an interior optimum satisfies $c _ { i } e ^ { - c _ { i } z _ { i } } = \xi$ . Consequently, for $\xi > 0$ , the optimal exposure has the clipped water-filling form

$$
z _ { i } ^ { \mathrm { c o n t } } ( \xi ) = \left[ \frac { 1 } { c _ { i } } \log \frac { c _ { i } } { \xi } \right] _ { [ 0 , Z ] } , \qquad c _ { i } > 0 ,\tag{21}
$$

where $[ x ] _ { [ 0 , Z ] } = \operatorname* { m i n } \{ Z , \operatorname* { m a x } \{ 0 , x \} \}$

Let ${ \mathcal { T } } _ { + } = \{ i : c _ { i } > 0 \}$ . When $0 < B < Z | \mathcal { I } _ { + } |$ , the budget is binding and ξ can be chosen so that $\begin{array} { r } { \sum _ { i \in \mathcal { T } _ { + } } z _ { i } ^ { \mathrm { c o n t } } ( \xi ) = B } \end{array}$ . The left-hand side is continuous and nonincreasing in $\xi ,$ allowing the price to be found by one-dimensional bisection. Prompts with $c _ { i } = 0$ contribute zero utility and may be assigned zero exposure. When $B \geq Z | \mathcal { I } _ { + } |$ , assigning $Z$ units to every positive-rate prompt is optimal; any remaining budget may be left unused or assigned to zero-rate prompts without changing the objective.

Exact integer allocation. For integer exposures, the marginal utility of the ℓ-th allocation unit assigned to prompt i is

$$
\Delta _ { i } ( \ell ) : = U ( q _ { i } , \ell ) - U ( q _ { i } , \ell - 1 ) = ( 1 - e ^ { - c _ { i } } ) e ^ { - c _ { i } ( \ell - 1 ) } , \qquad \ell = 1 , \dots , Z .\tag{22}
$$

These marginal gains are nonnegative and nonincreasing in ℓ. Starting from $z _ { i } = 0$ for every prompt, the oracle repeatedly assigns one unit to an unsaturated prompt with the largest currently available gain $\Delta _ { i } ( z _ { i } +$ 1). Repeating this operation for B units produces an optimal integer exposure vector, allowing zero-gain assignments when necessary.

The optimality follows from the decreasing marginal-gain sequences. A set of the B largest gains can be chosen to contain a prefix of each prompt’s sequence, with ties resolved in favor of earlier units. The greedy procedure constructs such a set while respecting $z _ { i } \leq Z .$ . A max-heap containing the next available marginal gain for each unsaturated prompt computes the exposure vector in $O ( M + B \log ( M + 1 ) )$ time and $O ( M )$ auxiliary memory.

Recovery of a feasible rollout plan. To realize any feasible integer exposure vector z, let $T = \textstyle \sum _ { i } z _ { i }$ and list its T allocation units, placing the units for each prompt in one consecutive block. Assign the r-th unit in this list to round $1 + ( ( r - 1 )$ mod K). Each prompt’s units are thereby spread as evenly as possible across rounds, and the same is true of the total allocation. The resulting counts satisfy

$$
\sum _ { k = 1 } ^ { K } N _ { i , k } = z _ { i } , \qquad N _ { i , k } \leq \left\lceil \frac { z _ { i } } { K } \right\rceil \leq N _ { \operatorname* { m a x } } , \qquad \sum _ { i } N _ { i , k } \leq \left\lceil \frac { T } { K } \right\rceil \leq C .\tag{23}
$$

The final inequalities use $z _ { i } \le K N _ { \operatorname* { m a x } } , T \le B \le K C$ , and the integrality of $N _ { \mathrm { m a x } }$ and C. Thus, the exposurelevel optimum is attainable under the original global-budget and per-round capacity constraints.

Full-capacity boundary. When $B = K M N _ { \operatorname* { m a x } }$ and $C = M N _ { \operatorname* { m a x } }$ , assigning $N _ { i , k } = N _ { \operatorname* { m a x } }$ to every prompt in every round is feasible and yields $z _ { i , K } = Z$ . Since each utility is nondecreasing in exposure, this allocation attains

$$
\mathrm { O P T } _ { K } ^ { \mathrm { s t a t } } ( q ) = \sum _ { i = 1 } ^ { M } U ( q _ { i } , Z ) .
$$

The full-allocation rule therefore achieves zero fixed-rate regret without estimating the target rates.

The oracle makes the diminishing-return structure explicit: additional units are assigned according to their remaining marginal utility rather than a prompt’s initial rate alone. It characterizes the fixed-rate benchmark, while CERO uses reward-derived proxy states and online dual updates to allocate rollouts as feedback becomes available.

## Appendix D: Proof of the Fixed-Rate Guarantee

For the analysis, if the execution budget is exhausted before round K, we continue the virtual iterates through the remaining rounds under the same update rules, with all subsequent executed allocations set to zero.

For each round k, let $\widehat { U } _ { i , k + 1 }$ be the post-feedback proxy utility and $\widehat { U } _ { i , k + 1 } ^ { * }$ its compact conjugate. Let $\widetilde { N } _ { k }$ be the virtual proposal and $\begin{array} { r } { \widetilde { A } _ { k } = \sum _ { i } \widetilde { N } _ { i , k } } \end{array}$ . Work with the unnormalized duals $\theta _ { i } ^ { k } = \bar { c } \vartheta _ { i } ^ { k }$ and $\mu ^ { k } = \bar { c } \nu ^ { k }$ . The normalized updates in (10) are equivalent to projected subgradient descent with

$$
\alpha _ { \theta } = \frac { \bar { c } } { N _ { \mathrm { { m a x } } } \sqrt { K } } , \qquad \alpha _ { \mu } = \frac { \bar { c } } { G _ { B } \sqrt { K } } .\tag{24}
$$

Define convex losses, after the round-k proposal and proxy utilities have been realized, by

$$
h _ { k } ( \pmb { \theta } ) : = \sum _ { i = 1 } ^ { M } \left[ \theta _ { i } \widetilde { N } _ { i , k } - \frac { 1 } { K } \widehat { U } _ { i , k + 1 } ^ { * } ( \theta _ { i } ) \right] ,\tag{25}
$$

$$
m _ { k } ( \mu ) : = \mu ( \bar { B } - \widetilde { A } _ { k } ) .\tag{26}
$$

Any ${ \widehat { z } } _ { i , k + 1 } ^ { * } ( \theta _ { i } ^ { k } )$ is a supergradient of the concave conjugate at $\theta _ { i } ^ { k }$ , so

$$
g _ { \theta _ { i } } ^ { k } = \widetilde { N } _ { i , k } - \frac { 1 } { K } \widehat { z } _ { i , k + 1 } ^ { * } ( \theta _ { i } ^ { k } )\tag{27}
$$

is a subgradient of $h _ { k }$ . Also $g _ { \mu } ^ { k } = \bar { B } - \widetilde { A } _ { k }$ is the gradient of $m _ { k }$ . The gradient bounds are

$$
\| \pmb { g } _ { \theta } ^ { k } \| _ { 2 } \leq \sqrt { M } N _ { \operatorname* { m a x } } , \qquad | \pmb { g } _ { \mu } ^ { k } | \leq G _ { B } .\tag{28}
$$

The domain diameters are $\bar { c } \sqrt { M }$ and ¯c. Standard projected-OGD analysis therefore gives

$$
\sum _ { k = 1 } ^ { K } h _ { k } \mathopen { } \mathclose \bgroup \left( \theta ^ { k } \aftergroup \egroup \right) - \operatorname* { m i n } _ { \theta \in \left[ 0 , \bar { c } \right] ^ { M } } \sum _ { k = 1 } ^ { K } h _ { k } \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right) \leq \bar { c } M N _ { \operatorname* { m a x } } \sqrt { K } ,\tag{29}
$$

$$
\sum _ { k = 1 } ^ { K } m _ { k } ( \mu ^ { k } ) - \operatorname* { m i n } _ { \mu \in [ 0 , \bar { c } ] } \sum _ { k = 1 } ^ { K } m _ { k } ( \mu ) \leq \bar { c } G _ { B } \sqrt { K } .\tag{30}
$$

These inequalities are pathwise; the losses may depend on the realized feedback and current proposal.

Define

$$
\widetilde { z } _ { i } : = \sum _ { k = 1 } ^ { K } \widetilde { N } _ { i , k } , \qquad \widetilde { T } : = \sum _ { i } \widetilde { z } _ { i } , \qquad \widetilde { V } : = [ \widetilde { T } - B ] _ { + } .\tag{31}
$$

Let

$$
\overline { { U } } _ { i } ^ { * } ( \theta ) : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \widehat { U } _ { i , k + 1 } ^ { * } ( \theta ) ,\tag{32}
$$

$$
\overline { { U } } _ { i } ( z ) : = \operatorname* { m i n } _ { 0 \leq \theta \leq \overline { { c } } } \{ \theta z - \overline { { U } } _ { i } ^ { * } ( \theta ) \} .\tag{33}
$$

Separability gives

$$
\operatorname* { m i n } _ { \pmb { \theta } } \sum _ { k = 1 } ^ { K } h _ { k } ( \pmb { \theta } ) = \sum _ { i = 1 } ^ { M } \overline { { U } } _ { i } ( \widetilde { z } _ { i } ) ,\tag{34}
$$

$$
\operatorname* { m i n } _ { 0 \leq \mu \leq \bar { c } } \sum _ { k = 1 } ^ { K } m _ { k } ( \mu ) = - \bar { c } \widetilde { V } .\tag{35}
$$

Hence

$$
\sum _ { k = 1 } ^ { K } [ h _ { k } ( \pmb { \theta } ^ { k } ) + m _ { k } ( \mu ^ { k } ) ] \leq \sum _ { i } \overline { { U } } _ { i } ( \widetilde { z } _ { i } ) + \bar { c } ( M N _ { \operatorname* { m a x } } + G _ { B } ) \sqrt { K } - \bar { c } \widetilde { V } .\tag{36}
$$

Let $\mathbf { N } ^ { * }$ attain (4), define $\begin{array} { r } { z _ { i } ^ { * } = \sum _ { k } N _ { i , k } ^ { * } } \end{array}$ , and set $n _ { i } ^ { * } = z _ { i } ^ { * } / K$ . Then $0 \leq n _ { i } ^ { * } \leq N _ { \operatorname* { m a x } }$ and

$$
\sum _ { i } n _ { i } ^ { * } = \frac { 1 } { K } \sum _ { i } z _ { i } ^ { * } \leq \frac { B } { K } = \bar { B } \leq C .\tag{37}
$$

Thus ${ \pmb n } ^ { * } \in \mathcal { P } _ { C }$ . Since $\mathcal { P } _ { C }$ is integral, the integer proposal maximizes the same linear function over the continuous polytope. Therefore

$$
\sum _ { i } ( \theta _ { i } ^ { k } - \mu ^ { k } ) \widetilde { N } _ { i , k } \geq \sum _ { i } ( \theta _ { i } ^ { k } - \mu ^ { k } ) n _ { i } ^ { * } .\tag{38}
$$

Using $\begin{array} { r } { \mu ^ { k } \geq 0 , \sum _ { i } n _ { i } ^ { * } \leq \bar { B } } \end{array}$ , and the proxy Fenchel representation yields

$$
h _ { k } ( \pmb { \theta } ^ { k } ) + m _ { k } ( \mu ^ { k } ) \geq \frac { 1 } { K } \sum _ { i } \left[ \theta _ { i } ^ { k } z _ { i } ^ { * } - \widehat { U } _ { i , k + 1 } ^ { * } ( \theta _ { i } ^ { k } ) \right]\tag{39}
$$

$$
\geq \frac { 1 } { K } \sum _ { i } \widehat { U } _ { i , k + 1 } \big ( z _ { i } ^ { * } \big ) .\tag{40}
$$

Summing and combining with (36) gives

$$
\sum _ { i } \frac { 1 } { K } \sum _ { k } \widehat { U } _ { i , k + 1 } ( z _ { i } ^ { * } ) \leq \sum _ { i } \overline { { U } } _ { i } ( \widetilde { z } _ { i } ) + \bar { c } ( M N _ { \operatorname* { m a x } } + G _ { B } ) \sqrt { K } - \bar { c } \widetilde { V } .\tag{41}
$$

Let $e _ { i , k } = | \widehat { q } _ { i , k + 1 } - q _ { i } |$ and

$$
\delta _ { i , k } : = \frac { 4 \lambda Z } { R } e _ { i , k } , \qquad \Delta : = \frac { 1 } { K } \sum _ { i , k } \delta _ { i , k } .\tag{42}
$$

By Lemma 2, both the proxy utility and its conjugate difer from their target versions by at most $\delta _ { i , k }$ Consequently,

$$
\mathsf { O P T } _ { K } ^ { \mathrm { s t a t } } ( \pmb q ) - \sum _ { i } U \big ( q _ { i } , \widetilde { z } _ { i } \big ) + \bar { c } \widetilde { V } \le \bar { c } \big ( M N _ { \operatorname* { m a x } } + G _ { B } \big ) \sqrt { K } + 2 \Delta .\tag{43}
$$

Since $Z = K { { N } _ { \operatorname* { m a x } } }$

$$
\Delta = \frac { 4 \lambda N _ { \mathrm { m a x } } } { R } \sum _ { i , k } e _ { i , k } = \frac { 4 \lambda M N _ { \mathrm { m a x } } } { \bar { N } } \bar { e } _ { K } ^ { \mathrm { s t a t } } .\tag{44}
$$

It remains to pass from virtual proposals to executed allocations. The prefix clipping rule implies

$$
\sum _ { i } ( \widetilde { z } _ { i } - z _ { i , K } ^ { C E R O } ) = \widetilde { V } .\tag{45}
$$

Every target utility has derivative at most ${ \bar { c } } ,$ so

$$
0 \leq \sum _ { i } [ U ( q _ { i } , \widetilde { z } _ { i } ) - U ( q _ { i } , z _ { i , K } ^ { C E R O } ) ] \leq \bar { c } \widetilde { V } .\tag{46}
$$

This clipping loss is exactly absorbed by the violation term on the left of (43). Therefore

$$
\mathrm { R e g } _ { K } ^ { \mathrm { s t a t } } \leq \bar { c } ( M N _ { \operatorname* { m a x } } + G _ { B } ) \sqrt { K } + 2 \Delta .\tag{47}
$$

Substitute $\bar { c } = \lambda / R , R = K \bar { N }$ , use (44), and divide by M. Since every prompt utility lies in [0, 1], the average regret is also at most one. Combining the two bounds gives (11).

## Appendix E: Proof of the Same-Path Guarantee

We first record a stability inequality for one incremental utility.

Lemma 3 (Increment stability). For any $q , q ^ { \prime } \in [ 0 , 1 / 4 ]$ and $z , n \geq 0$ satisfying $z + n \le Z$

$$
| [ U ( q , z + n ) - U ( q , z ) ] - [ U ( q ^ { \prime } , z + n ) - U ( q ^ { \prime } , z ) ] | \leq \frac { 4 \lambda } { R } n | q - q ^ { \prime } | .\tag{48}
$$

L et $\kappa = 4 \lambda / R$ and

$$
G ( q ; z , n ) = \int _ { z } ^ { z + n } \kappa q e ^ { - \kappa q s } d s .\tag{49}
$$

Then

$$
\partial _ { q } G ( q ; z , n ) = \int _ { z } ^ { z + n } \kappa ( 1 - \kappa q s ) e ^ { - \kappa q s } d s .\tag{50}
$$

Because $| ( 1 - x ) e ^ { - x } | \leq 1$ for $x \ge 0$ , the derivative magnitude is at most κn. Apply the mean-value theorem.

For each prompt choose the midpoint of its realized rate range,

$$
q _ { i } ^ { \circ } : = \frac { 1 } { 2 } \left( \operatorname* { m i n } _ { k } q _ { i , k } + \operatorname* { m a x } _ { k } q _ { i , k } \right) .\tag{51}
$$

Then $| q _ { i , k } - q _ { i } ^ { \circ } | \leq V _ { K } / 2$ for every i, k. For any feasible allocation N, let

$$
J ^ { \mathrm { d y n } } ( { \bf N } ) : = \sum _ { i , k } \Delta U _ { i , k } ( { \bf N } ; q _ { i , k } ) ,\tag{52}
$$

$$
J ^ { \circ } ( { \bf N } ) : = \sum _ { i } U ( q _ { i } ^ { \circ } , z _ { i , K } ) .\tag{53}
$$

Using (48) and total budget feasibility,

$$
| J ^ { \mathrm { d y n } } ( \mathbf { N } ) - J ^ { \circ } ( \mathbf { N } ) | \leq \frac { 4 \lambda } { R } \sum _ { i , k } N _ { i , k } | q _ { i , k } - q _ { i } ^ { \circ } |\tag{54}
$$

$$
\leq \frac { 2 \lambda V _ { K } } { R } B = 2 \lambda M V _ { K } .\tag{55}
$$

Applying this inequality to the dynamic optimum and to the executed CERO allocation gives

$$
\mathrm { R e g } _ { K } ^ { \mathrm { d y n } } \leq \mathsf { O P T } _ { K } ^ { \mathrm { s t a t } } ( { \pmb q } ^ { \circ } ) - \sum _ { i } U ( q _ { i } ^ { \circ } , z _ { i , K } ^ { C E R O } ) + 4 \lambda M V _ { K } .\tag{56}
$$

The average proxy error relative to $\pmb q ^ { \circ }$ satisfies

$$
\frac { 1 } { M K } \sum _ { i , k } \left| \widehat { q } _ { i , k + 1 } - q _ { i } ^ { \circ } \right| \leq \bar { e } _ { K } ^ { \mathrm { d y n } } + \frac { 1 } { M K } \sum _ { i , k } \left| q _ { i , k } - q _ { i } ^ { \circ } \right|\tag{57}
$$

$$
\leq \bar { e } _ { K } ^ { \mathrm { d y n } } + \frac { { \cal V } _ { K } } { 2 } .\tag{58}
$$

Apply Theorem 1 with reference vector $\pmb q ^ { \circ }$ and substitute the last inequality into (56). In addition, (3) and total budget feasibility imply that every same-path objective is at most λM, so the normalized regret is at most λ. Dividing by M and combining the two bounds yields (12).

## Appendix F: Proofs of the Beta-state Propositions

[ Proof of Proposition 1] Conditional on $P _ { i , k }$ , each pair of future Bernoulli rewards disagrees with probability $2 P _ { i , k } ( 1 - P _ { i , k } )$ . The unbiased sample variance for binary data satisfies

$$
2 \widehat { \mathrm { V a r } } _ { G } ( r ) = \binom { G } { 2 } ^ { - 1 } \displaystyle \sum _ { 1 \leq j < \ell \leq G } { \bf 1 } \{ r ^ { ( j ) } \neq r ^ { ( \ell ) } \} .
$$

Taking conditional expectations first given $P _ { i , k }$ and then given $\mathcal { H } _ { i } ^ { k }$ proves the claim.

[ Proof of Proposition 2] Let $t = n + s _ { 0 }$ and $\widehat { m } = a / t$ . Then

$$
{ \widehat { q } } _ { n } = { \frac { t } { t + 1 } } { \widehat { m } } ( 1 - { \widehat { m } } ) .\tag{59}
$$

The function $x ( 1 - x )$ is one-Lipschitz on [0, 1], so

$$
\left| \widehat { q } _ { n } - p ( 1 - p ) \right| \leq \left| \widehat { m } - p \right| + \frac { 1 } { 4 ( t + 1 ) } .\tag{60}
$$

Moreover,

$$
| \mathbb { E } \widehat { m } - p | = \frac { \left| a _ { 0 } - p s _ { 0 } \right| } { t } \leq \frac { s _ { 0 } } { t } ,\tag{61}
$$

and

$$
\mathbb { E } | \widehat { m } - \mathbb { E } \widehat { m } | \leq \sqrt { \mathrm { V a r } ( \widehat { m } ) } = \frac { \sqrt { n p ( 1 - p ) } } { t } { \leq \frac { \sqrt { n } } { 2 t } } .\tag{62}
$$

Combining the bounds proves (7).

## Appendix G: Implementation Details

Training. We use two locally modified veRL-based (Sheng et al. 2025) stacks with matched learner and decoding hyperparameters, vLLM 0.8.3 for generation, and FSDP for actor training. Bfloat16 is used for generation and actor computation. Experiments use two NVIDIA H800 GPUs for the 1.5B backbone and four for the 4B and 7B backbones. All methods use GRPO with within-group standard-deviation normalization, advantage clipping to [−5, 5], and loss averaged over valid response tokens. AdamW uses a constant learning rate of $1 0 ^ { - 6 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ , zero weight decay, and no warmup. We use one PPO epoch per collected batch, no reshufling, and a global PPO mini-batch size of 32. The lower/upper policy clipping parameters are 0.20/0.28, the dual-clip constant is 10, the importance-ratio cap is 2, and the gradient-norm threshold is 1. KL and entropy regularization are disabled. Training prompts are left-truncated to 2,048 tokens and responses are capped at 2,048 tokens; sampling uses temperature 1.0, top-p 1.0, and top-k −1. Training rewards are computed using the rule-based mathematical verifier from DAPO (Yu et al. 2025), which treats +1 as success and 0 as failure. The CERO run in Table 1 is included in the multi-seed results in Table 2.

A scheduler round is one allocation opportunity, whereas a training step is one policy-update call on a collected rollout batch. Zero-allocation rounds advance the scheduler without updating the policy. A data epoch is one traversal of the training data loader, distinct from a PPO epoch over a collected rollout batch.

Allocation settings. We use Algorithm 1 with $g = 4$ responses per allocation unit and $N _ { \mathrm { m a x } } = 2 , C = 1 3 2$ and $B = 6 4 { , } 0 0 0$ in allocation units. CERO uses a Beta(1, 1) prior, $\lambda = \rho = 1 , \nu ^ { 1 } = 0 . 5 .$ , and $\gamma _ { \vartheta } = \gamma _ { \nu } = 1 / \sqrt { 5 0 0 }$ Ties use a fixed seed permutation. Reserve-aware catch-up was enabled but never activated in the reported CERO runs. KnapsackRL uses positive and negative base allocations of two, a reward-history bufer of length 16, a minimum initialized history length of four, and robust epoch estimation.

CERO-greedy uses the same reward-proxy estimator, utility function, and fixed 512-response quota as CERO-fixed, but replaces prompt-dual scores with the marginal utility of one additional complete group. In our experiments, each group contains eight responses, corresponding to $N _ { \mathrm { m a x } } = 2$ allocation units. At the beginning of round k, the selection score for prompt i is

$$
\Delta _ { i , k } ^ { \mathrm { g r p } } : = U \big ( \widehat { q } _ { i , k } , z _ { i , k - 1 } + N _ { \operatorname* { m a x } } \big ) - U \big ( \widehat { q } _ { i , k } , z _ { i , k - 1 } \big ) ,
$$

where $\hat { q } _ { i , k }$ is the pre-decision proxy and $z _ { i , k - 1 }$ is the cumulative executed exposure in the control run. We use CERO’s exponential utility with the same saturation parameter and exposure normalization. Each round scores the full training prompt pool and selects the 64 prompts with the largest scores, generating one eight-response group per prompt. This control preserves the reward-based utility and its diminishing-return structure while replacing dual-based selection with direct marginal-utility ranking.

Evaluation. We follow Tan et al. (2025) and evaluate the checkpoint obtained after 500 scheduler rounds using 16 responses per problem, temperature 0.6, top-p 1.0, and top-k −1. All methods use the same rulebased mathematical equivalence checker without an LLM judge. Metric definitions are given in Section 6 and Appendix B.1; computational costs and schedule audits are reported in Appendix B.4.