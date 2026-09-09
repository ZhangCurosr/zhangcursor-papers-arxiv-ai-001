# Inference-Time Nash Alignment

Hadi Hosseini Penn State University, USA hadi@psu.edu

Debmalya Mandal University of Warwick, UK Debmalya.Mandal@warwick.ac.uk

Duohan Zhang Penn State University, USA dqz5235@psu.edu

## Abstract

Preference-based fine-tuning methods such as RLHF and DPO require substantial compute and large preference datasets. They also need direct access to the model parameters which are not provided by many state-of-the art models. Inference-time alignment offers a cost-effective alternative without updating model parameters. However, existing inference-time methods rely on a scalar reward model derived under a Bradley-Terry assumption, which cannot represent general preferences. Following recent work on fine-tuning with generalized preferences, in this work, we initiate the study of inference-time alignment under general preferences. We formulate the problem as obtaining a Nash equilibrium of a two-player zero-sum game between policies. We propose two algorithms: Best-of-Nash (BoN) and Nash Mirror Descent (NMD). We prove that both algorithms achieve a duality gap that matches the problem lower bound. Empirically, we implement the two methods on three datasets, which shows that our methods substantially outperform the base policy, converging to the performance of the fine-tuned models. Moreover, our results show that NMD remains robust across the regularization parameter.

## 1 Introduction

Preference-based fine-tuning methods such as Reinforcement Learning with Human Feedback (RLHF) [Christiano et al., 2017] and Direct Preference Optimization (DPO) [Rafailov et al., 2023] have become quintessential for aligning Large Language Models (LLMs) with human preferences. These methods have proven highly effective across several domains including mathematical reasoning and finance. However, they require substantial effort in acquiring high-quality human data, along with considerable computational costs for training LLMs. For example, fine-tuning LLMs with either Proximal Policy Optimization (PPO) [Schulman et al., 2017] or Group Relative Policy Optimization (GRPO) [Shao et al., 2024] requires a large number of samples and often leads to instability in training. Equally importantly, these approaches rely on white-box access to model parameters, whereas many state-of-the-art models (e.g., GPT-5.4 Thinking [OpenAI, 2026], Gemini 3.1 Pro [Google, 2026], and Claude Opus 4.7 [Anthropic, 2026]) are only accessible as black-box APIs.

To bridge this gap, inference-time alignment has recently drawn much attention. Unlike preferencebased fine-tuning, inference-time alignment modifies the generation process at run time without updating model parameters. A popular line of work adopts a sample-and-rank paradigm, of which Best-of-N sampling [Stiennon et al., 2020, Nakano et al., 2021, Huang et al., 2025] is the most widely used – N candidate responses are drawn for a given prompt and the one with the highest score under a reward model is returned. While simple in its nature, Best-of-N alignment is vulnerable to reward hacking: as $N$ grows, the estimated reward of the selected response increases monotonically while its true task performance can degrade [Gao et al., 2023, Stroebl et al., 2024, Chow et al., 2024]. This phenomenon arises because reward model is at best an imperfect proxy for the true human preference distribution, a manifestation of Goodhart’s law.

Furthermore, most inference-time methods exclusively assume that preferences can be modeled by a scalar reward function under the Bradley-Terry model [Bradley and Terry, 1952]. This assumption is restrictive: even when individual preferences are transitive, aggregated group-level preferences need not be [Munos et al., 2024]. Generalized preference models, which directly specify a probability that one response is preferred to another, have recently been studied for fine-tuning in, e.g., Nash Learning from Human Feedback (NLHF) [Munos et al., 2024]. To the best of our knowledge, tackling generalized preferences remains an open challenge in inference-time methods.

In this work, we initiate the study of black-box inference-time alignment under generalized preferences. The preference model $\dot { \mathbb { P } } ^ { * }$ takes two responses $y$ and $y ^ { \prime }$ conditioned on a prompt $x ,$ and evaluates the score $\mathbb { P } ^ { * } ( y \succ y ^ { \prime } | x )$ as the probability that a randomly chosen human prefers response $y$ over $y ^ { \prime }$ given prompt x. Then we formulate inference-time alignment with generalized preferences as a two-player zero-sum game $\mathbb { P } ^ { * } ( \pi \succ \pi ^ { \prime } | x ) : = \mathbb { E } _ { y \sim \pi , y ^ { \prime } \sim \pi ^ { \prime } } \mathbb { P } ^ { * } \bar { ( } y \succ y ^ { \prime } | x )$ between row player’s policy π and column player’s policy $\pi ^ { \prime }$ under the true preference $\mathbb { P } ^ { * }$ . However, we do not have access to $\mathbb { P } ^ { * }$ , and instead, use an imperfect preference model $\widehat { \mathbb { P } }$ (e.g. one learned from preference data [Jiang et al., 2023, Dong et al., 2024, Munos et al., 2024]) to query at inference time. Specifically, we ask the following question:

Given a base policy $\pi _ { r e f }$ from which we can query responses and an imperfect preference model ${ \widehat { \mathbb { P } } } ,$ can we design an efficient algorithm to approximate the Nash equilibrium at inference time?

## 1.1 Our Contributions

Prior work has explored Nash-based objectives in the fine-tuning setting [Munos et al., 2024] and has separately identified reward hacking as a key vulnerability of inference-time methods like Bestof-N [Stroebl et al., 2024, Chow et al., 2024]. We connect these two threads by proposing new inference-time methods with general preference models that achieve optimal regret. On the technical front, we link the alignment problem to the policy coverage and preference model error, which we will define formally in Section 2.

In particular, our main contributions are the following.

1. Best-of-Nash (BoN): We propose Best-of-Nash Alignment method which computes a minimax solution from N samples according to an estimate of the preference matrix P<sup>ˆ</sup>. We show that the duality gap of BoN alignment is at most $O ( \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) )$ on a given prompt x, where $\varepsilon ( x )$ is the error in the preference oracle, and $\mathcal { C } _ { \mathrm { u n i } } ( x )$ is a measure of data coverage.

2. Nash Mirror Descent (NMD): As a more computationally efficient alternative, we propose Nash Mirror Descent, a self-play algorithm that takes a KL-regularized mirror descent step iteratively. We show that NMD is faster to implement, and the upper bound on the duality gap of NMD is at most $O ( \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) ) ,$ ), matching the exact Best-of-Nash bound.

3. Matching Lower Bound: We then show that the upper bound of BoN and NMD is essentially optimal by constructing problem instances with lower bound at least $\Omega ( \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) )$ .

4. Experimental Evaluation: We evaluate our proposed mechanisms BoN and NMD on three preference datasets: TLDR, HelpSteer2, and UltraFeedback. We show that both methods substantially improve over the SFT base model, and match the win-rate of the fine-tuned alternative on TLDR. We also show that the win-rates of NMD are robust across the regularization parameter, removing the need for hyperparameter tuning.

## 1.2 Related Work

Fine-tuning with General Preferences. Azar et al. [2024] initiated the study of general preferences in LLM fine-tuning, and proposed an algorithm that maximizes the KL-regularized objective against a fixed policy. A subsequent line of works [Munos et al., 2024, Ye et al., 2024, Calandriello et al.,

2024, Rosset et al., 2024, Wu et al., 2024, Zhang et al., 2024, 2025, Swamy et al., 2024, Zhou et al., 2025] formulated the alignment problem as a two-player zero-sum game, and proposed fine-tuning algorithms to learn the Nash policy. Maura-Rivero et al. [2025] connected this line of works to social choice theory, showing that the Nash policy approximates maximal lottery outcomes.

Inference-Time Alignment and Reward Hacking. Best-of-N sampling is a popular inference-time alignment approach, but it is vulnerable to reward hacking problem [Skalse et al., 2022], when an LLM exploits the learned reward model rather than the ground-truth reward. Huang et al. [2025] showed that Best-of-N incurs suboptimal regret relative to the problem’s lower bound, and proposed $\mathbf { a } \ \chi ^ { 2 }$ -regularized algorithm that implements pessimism in the face of uncertainty to close this gap. Yu et al. [2026] took a different route, using lower confidence bounds on value estimates to mitigate reward hacking. Gui et al. [2024] combined RLHF with Best-of-N sampling to improve fine-tuning, however, they are concerned with reward-based setting. Finally, there is inference-time method based on search [Khanov et al., 2024, Yao et al., 2023] and rejection sampling [Chen et al., 2024, Shi et al., 2024], but these approaches don’t handle general preferences.

Learning in Zero-sum Games. The central ingredient of our approach is no-regret learning algorithm for solving zero-sum games. Freund and Schapire [1999] showed that multiplicative weights update has no-regret guarantee. We use Mirror descent algorithm [Nemirovski, 2004, Nesterov, 2009, Rakhlin and Sridharan, 2013] for inference-time alignment. Our second algorithm is inspired by the optimistic variants of Mirror descent algorithm [Rakhlin and Sridharan, 2013]. Mirror Prox algorithm in Nemirovski [2004] is an extragradient type of method and closely related to optimistic MD (with similar convergence rate). It queries the gradient twice whereas we use the same gradient twice (predicted and current).

## 2 Preliminary

Denote $\mathcal { X }$ as the prompt space and $\mathcal { V }$ as the response space. We begin with a base policy $\pi _ { \mathrm { r e f } } : \mathcal { X } $ $\Delta ( \mathcal { V } )$ , where $\pi _ { \mathrm { r e f } } ( y | x )$ is the probability that the base policy $\pi _ { \mathrm { r e f } }$ generates a response y given the prompt x. We also assume that there exists an unknown true preference oracle $\mathbb { P } ^ { * } : \bar { \mathcal { X } } \times \mathcal { Y } \times \bar { \mathcal { Y } }  [ 0 , 1 ]$ where $\mathbb { P } ^ { * } ( y \succ y ^ { \prime } | x )$ denotes the probability that the population prefers response $y$ to $y ^ { \prime }$ given a prompt x. We assume that the preference model is skew-symmetric:

$$
\begin{array} { r } { \mathbb { P } ^ { * } ( y \succ y ^ { \prime } | x ) + \mathbb { P } ^ { * } ( y ^ { \prime } \succ y | x ) = 1 , \forall x , y , y ^ { \prime } , } \end{array}
$$

which implies $\mathbb { P } ^ { * } ( y \succ y | x ) = 1 / 2$ . As a proxy, we have access to an imperfect preference oracle ${ \widehat { \mathbb { P } } } : { \mathcal { X } } \times { \mathcal { Y } } \times { \mathcal { Y } } \to [ 0 , 1 ]$ , also assumed skew-symmetric. For a given prompt x, we measure the quality of the oracle $\widehat { \mathbb { P } }$ via the square error with respect to $\mathbb { P } ^ { * }$ , where responses are drawn independently from the base policy $\pi _ { \mathrm { r e f } } .$

$$
\varepsilon ^ { 2 } ( x ) : = \mathbb { E } _ { y \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) , y ^ { \prime } \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } [ ( \widehat { \mathbb { P } } ( y \succ y ^ { \prime } | x ) - \mathbb { P } ^ { * } ( y \succ y ^ { \prime } | x ) ) ^ { 2 } ] .
$$

Nash Equilibrium and Duality Gap. We formulate the problem as a two-player zero-sum game. Given row player’s policy π and column player’s policy $\pi ^ { \prime } .$ , we denote the expected win-rate as

$$
\begin{array} { r } { \mathbb { P } ^ { * } ( \pi \succ \pi ^ { \prime } | x ) : = \mathbb { E } _ { y \sim \pi ( \cdot | x ) , y ^ { \prime } \sim \pi ^ { \prime } ( \cdot | x ) } [ \mathbb { P } ^ { * } ( y \succ y ^ { \prime } | x ) ] . } \end{array}
$$

Here the row player aims to maximize the win-rate, and the column player aims to minimize the win-rate. It is well-known that there exists a Nash Equilibrium (NE) of the game:

$$
\begin{array} { r } { \pi _ { 1 } ^ { * } , \pi _ { 2 } ^ { * } : = \operatorname * { a r g m a x } _ { \pi _ { 1 } } \operatorname { a r g m i n } _ { \pi _ { 2 } } \mathbb { P } ^ { * } ( \pi _ { 1 } \succ \pi _ { 2 } | x ) . } \end{array}
$$

We denote the Nash Equilibrium as $\pi ^ { * } = \pi _ { 1 } ^ { * } = \pi _ { 2 } ^ { * }$ due to the skew-symmetric nature of $\mathbb { P } ^ { * }$ . To measure how close a given policy $\pi$ is to $\pi ^ { * }$ , we use the duality gap. It captures the regret of policy $\pi$ as the gap between its strongest adversary and its weakest dominance under the true preference model:

$$
\operatorname { D u a l G a p } ( \pi ) : = \operatorname* { m a x } _ { \pi _ { 1 } } \mathbb { P } ^ { * } ( \pi _ { 1 } \succ \pi | x ) - \operatorname* { m i n } _ { \pi _ { 2 } } \mathbb { P } ^ { * } ( \pi \succ \pi _ { 2 } | x )
$$

The duality gap is nonnegative and ${ \mathrm { D u a l G a p } } ( \pi ) = 0 { \mathrm { i f } } \pi = \pi ^ { * }$ . Now given a reference policy $\pi _ { \mathrm { r e f } } .$ an imperfect preference oracle ${ \widehat { \mathbb { P } } } ,$ , and a prompt $x \in \mathcal { X }$ , our goal is to generate a high-quality policy πˆ with small duality gap:

$$
\mathrm { D u a l G a p } ( \hat { \pi } ) \leq \epsilon .
$$

We say πˆ is an ϵ-approximate Nash policy.

Universal Coverage. Coverage plays a significant role in the analysis of inference-time alignment [Huang et al., 2025]. It measures how much a policy, π concentrates probability mass relative to a reference policy, $\pi _ { \mathrm { r e f } } .$ upweighting outcomes that are more likely under $\pi .$ We define the coverage of policy π as the ratio of $\pi ( y | x )$ over $\pi _ { \mathrm { r e f } } ( y | x )$ ), where $y$ is sampled from $\pi ( \cdot | x )$

$$
\mathcal { C } ^ { \pi } ( \boldsymbol { x } ) : = \underset { \boldsymbol { y } \sim \pi ( \cdot | \boldsymbol { x } ) } { \mathbb { E } } \left[ \frac { \pi ( \boldsymbol { y } | \boldsymbol { x } ) } { \pi _ { \mathrm { r e f } } ( \boldsymbol { y } | \boldsymbol { x } ) } \right] .
$$

Coverage is closely related to the chi-square divergence: a direct calculation gives

$$
\chi ^ { 2 } ( \pi ( \cdot | x ) , \pi _ { \mathrm { r e f } } ( \cdot | x ) ) = \mathcal { C } ^ { \pi } ( x ) - 1 ,
$$

where the chi-square divergence is defined as $\begin{array} { r } { \chi ^ { 2 } ( \pi ( \cdot | x ) , \pi ^ { \prime } ( \cdot | x ) ) = \mathbb { E } _ { y \sim \pi ^ { \prime } ( \cdot | x ) } [ ( \frac { \pi ( y | x ) } { \pi ^ { \prime } ( y | x ) } - 1 ) ^ { 2 } ] } \end{array}$ . Thus, we have $\mathcal { C } ^ { \pi } ( x ) \geq 1$ and the equality holds if and only if $\pi ( \boldsymbol { y } | \boldsymbol { x } ) = \pi _ { \mathrm { r e f } } ( \boldsymbol { y } | \boldsymbol { x } )$ for all $y .$ . Inspired by literature in offline learning in zero-sum games [Cui and Du, 2022, Zhong et al., 2022, Zhang et al., 2023], we also define the universal coverage as the maximum coverage over any policy:

$$
\mathcal { C } _ { \mathrm { u n i } } ( x ) : = \operatorname* { m a x } _ { \pi } \operatorname* { \mathbb { E } } _ { y \sim \pi ( \cdot | x ) } \left[ \frac { \pi ( y | x ) } { \pi _ { \mathrm { r e f } } ( y | x ) } \right] .
$$

Intuitively, $\mathcal { C } _ { \mathrm { u n i } } ( x )$ captures the difficulty of recovering a Nash policy from samples drawn under $\pi _ { \mathrm { r e f } } .$ The two quantities $\varepsilon ( x )$ and $\mathcal { C } _ { \mathrm { u n i } } ( x )$ are the fundamental difficulty of inference-time alignment: no algorithm can output a good Nash approximation when the preference oracle $\widehat { \mathbb { P } }$ has high error or when $\pi _ { \mathrm { r e f } }$ poorly covers the response space.

## 3 The Best-of-Nash Algorithm

The inference-time Best-of-N alignment method relies on a scalar reward function by drawing N samples from policy $\pi _ { \mathrm { r e f } }$ and selecting a single response by taking the argmax under a reward model. However, it condenses the preference information into one number and could result in reward hacking.

We propose an alternative approach, namely Best-of-Nash (Algorithm 1), that retains the sample-andrank structure, but replaces the argmax with an equilibrium computation solely based on preference data: instead of picking one response, we output a distribution over the N samples that solves the Nash equilibrium of the empirical preference game.

Formally, given an input x, we draw N candidate responses $\widehat { \mathcal { V } } _ { N } = ( y _ { 1 } , \ldots , y _ { N } ) \sim \pi _ { \mathrm { r e f } } ( \cdot | x )$ i.i.d. We then construct a probability matrix by querying $\widehat { \mathbb { P } }$ on each ordered pair, i.e. $\widehat { \mathbb { P } } ( y _ { i } \succ y _ { j } )$ for $1 \leq i \leq j \leq N$ . The Nash equilibrium of the resulting two-player zero-sum game can be computed by Linear Programming (LP) [Adler, 2013].

Algorithm 1 Best-of-Nash (BoN) Alignment   
1: Input: Prompt x, reference policy $\pi _ { \mathrm { r e f } } .$ , preference oracle ${ \widehat { \mathbb { P } } } ,$ sample size $N$   
2: Draw $\widehat { y } _ { N } = ( y _ { 1 } , \ldots , y _ { N } ) \sim \pi _ { \mathrm { r e f } } ( \cdot | x )$ i.i.d.   
3: Query $\widehat { P } _ { i j } \gets \widehat { \mathbb { P } } ( y _ { i } \succ y _ { j } \mid x )$ for all $1 \leq i < j \leq N ,$ and set $\begin{array} { r } { \widehat { P } _ { j i } \gets 1 - \widehat { P } _ { i j } , \widehat { P } _ { i i } \gets \frac { 1 } { 2 } } \end{array}$   
4: Compute a Nash Equilibrium πˆ by solving the linear program   
$\operatorname* { m a x } _ { \pi \in \Delta ( \widehat { \mathcal { V } } _ { N } ) , v \in \mathbb { R } } v \quad \mathrm { s . t . } \quad \sum _ { j = 1 } ^ { N } \pi ( y _ { j } ) \widehat { P } _ { j i } \geq v \quad \forall i \in [ N ] .$   
5: Return $\hat { \pi } .$

We provide the duality gap guarantee for BoN. The bound depends on the two fundamental parameters: the preference-oracle error $\varepsilon ( x )$ and the universal coverage $\dot { C } _ { \mathrm { u n i } } ( x )$

Theorem 1. For any prompt x, the policy πˆ returned by Algorithm 1 satisfies

$$
\mathrm { D u a l G a p } ( \hat { \pi } ) \leq 3 \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x )
$$

when $\begin{array} { r } { N \geq 4 \log ( \frac { 2 } { \varepsilon ( x ) } ) \cdot \mathcal { C } _ { \mathrm { u n i } } ( x ) } \end{array}$

We provide the full proof below, and defer omitted lemmas to Section B.

Proof. We omit the dependence on x for cleanliness. From now on we write $S : = \mathcal { V } _ { N }$ as the candidate set, and we write $\hat { \pi } _ { S }$ to denote the dependence on S. By skew-symmetry of the zero-sum game, it suffices to upper-bound $\mathbb { P } ^ { * } \big ( \tilde { \pi } _ { S } \succ \hat { \pi } _ { S } \big ) ^ { * } - 1 / 2$ , where $\tilde { \pi } _ { S } : = \delta _ { y ^ { * } }$ ∗ with $\boldsymbol { y } ^ { * } \in \arg \operatorname* { m a x } _ { \boldsymbol { y } } \mathbb { P } ^ { * } \big ( \boldsymbol { y } \succ \hat { \pi } _ { S } \mid \boldsymbol { x } \big )$ The central difficulty is a support mismatch: $\hat { \pi } _ { S }$ is supported on the $N$ sampled responses, while $\tilde { \pi } _ { S }$ may place mass anywhere in Y. To bridge this, we introduce $\pi _ { R , S } .$ , the distribution induced by approximate rejection sampling [Block and Polyanskiy, 2023, Huang et al., 2025] of $\tilde { \pi } _ { S }$ from $\pi _ { \mathrm { r e f } } .$ Specifically, we denote $\pi _ { R , S }$ as the distribution induced by RejectionSampling $\begin{array} { r } { N - 1 , M } \end{array} \big ( \frac { \tilde { \pi } _ { S } } { \pi _ { \mathrm { r e f } } } ; \pi _ { \mathrm { r e f } } , x \big )$ (Algorithm 3) as an approximation to π˜. Then we decompose the probability that

$$
\mathbb { P } ^ { * } ( \tilde { \pi } _ { S } \succ \hat { \pi } _ { S } ) \le \mathbb { P } ^ { * } ( \pi _ { R , S } \succ \hat { \pi } _ { S } ) + | \mathbb { P } ^ { * } ( \tilde { \pi } _ { S } \succ \hat { \pi } _ { S } ) - \mathbb { P } ^ { * } ( \pi _ { R , S } \succ \hat { \pi } _ { S } ) | .
$$

We bound the first term $\begin{array} { r } { \mathbb { P } ^ { * } ( \pi _ { R , S } \succ \hat { \pi } _ { S } ) \le \frac { 1 } { 2 } + \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) } \end{array}$ by Lemma 1. Then we bound the second term

$$
\begin{array} { r l } & { \displaystyle | \mathbb { P } ^ { * } ( \tilde { \pi } _ { S } \succ \hat { \pi } _ { S } ) - \mathbb { P } ^ { * } ( \pi _ { R , S } \succ \hat { \pi } _ { S } ) | \leq \sum _ { y } | \tilde { \pi } _ { S } ( y ) - \pi _ { R , S } ( y ) | \sum _ { y ^ { \prime } } \hat { \pi } _ { S } ( y ^ { \prime } ) \mathbb { P } ^ { * } ( y \succ y ^ { \prime } ) } \\ & { \qquad \leq \sum _ { y } | \tilde { \pi } _ { S } ( y ) - \pi _ { R , S } ( y ) | } \\ & { \qquad \leq 2 D _ { \mathrm { T V } } ( \tilde { \pi } _ { S } , \pi _ { R , S } ) , } \end{array}
$$

where the total-variation distance is defined as $\begin{array} { r } { D _ { \mathrm { T V } } ( \pi , \pi ^ { \prime } ) : = \frac { 1 } { 2 } \sum _ { u } | \pi ( y ) - \pi ^ { \prime } ( y ) } \end{array}$ |. Thus, the second term is reduced to the TV distance between $\tilde { \pi } _ { S }$ and $\pi _ { R , S }$ , and then can be bounded due to the fact that $\pi _ { R , S }$ is an approximation to $\tilde { \pi } _ { S }$ . By Lemma 2, We have

$$
D _ { \mathrm { T V } } ( \tilde { \pi } _ { S } , \pi _ { R , S } ) \leq \frac { 1 } { 4 } \varepsilon ^ { 2 } ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x )
$$

by setting $\begin{array} { r } { M = \frac { N - 1 } { \log ( 4 / \varepsilon ^ { 2 } ( x ) ) } } \end{array}$ and $\begin{array} { r } { N \ge 4 \log ( \frac { 2 } { \varepsilon ( x ) } ) \cdot \mathcal { C } _ { \mathrm { u n i } } ( x ) } \end{array}$

By aggregating the bounds for both terms, we derive the upper bound

$$
\begin{array} { r l } & { \mathrm { D u a l G a p } ( \hat { \pi } _ { S } ) \leq 2 \mathbb { P } ^ { * } ( \tilde { \pi } _ { S } \succ \hat { \pi } _ { S } ) - 1 } \\ & { \phantom { \mathrm { ~ \theta ~ } } \leq 2 ( \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) + \frac { 1 } { 2 } \varepsilon ^ { 2 } ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) ) } \\ & { \phantom { \mathrm { ~ \theta ~ } } \leq 3 \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) . } \end{array}
$$

For any fixed y, linearity gives

$$
\mathbb { P } ^ { * } ( y \succ \hat { \pi } ) = \mathbb { P } ^ { * } ( y \succ \mathbb { E } _ { S } [ \hat { \pi } _ { S } ] ) = \mathbb { E } _ { S } [ \mathbb { P } ^ { * } ( y \succ \hat { \pi } _ { S } ) ] .
$$

Finally, we take the expectation over the randomness of S:

$$
\begin{array} { r l } { \mathrm { D u a l G a p } = 2 \operatorname* { m a x } \mathbb { E } _ { S } [ \mathbb { P } ^ { * } ( y \succ \hat { \pi } _ { S } ) ] - 1 } & { } \\ { \quad } & { \leq 2 \mathbb { E } _ { S } [ \operatorname* { m a x } \mathbb { P } ^ { * } ( y \succ \hat { \pi } _ { S } ) ] - 1 } \\ { \quad } & { = \mathbb { E } _ { S } [ \mathrm { D u a l G a p } ( \hat { \pi } _ { S } ) ] } \\ { \quad } & { \leq 3 \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) } \end{array}
$$

and the proof is complete.

Remark 1. In Theorem 1 we assume that the LP solution is exact. When Algorithm 1 returns an ϵ -approximation equilibrium, then we have DualGap $\leq 3 \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) + 2 \epsilon _ { L P }$

## 4 The Nash Mirror Descent Algorithm

Best-of-Nash requires solving a linear programming with post-query time ${ \cal O } ( N ^ { 3 . 5 } \log ( 1 / \epsilon ) )$ via interior-point methods. In this section, we propose Nash Mirror Descent (Algorithm 2), a self-play algorithm that achieves the same duality gap bound while replacing the LP with a sequence of closed-form updates.

Algorithm 2 Nash Mirror Descent Alignment   
1: Input: Prompt x, reference policy $\pi _ { \mathrm { r e f } } ,$ preference oracle ${ \hat { \mathbb { P } } } ,$ sample size N, regularization   
parameter $\beta .$   
2: Sample N data $\mathcal { V } _ { N } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { N } \}$ i.i.d. from $\pi _ { \mathrm { r e f } } ( \cdot | x )$   
3: Initialize $\pi _ { 1 } ^ { \prime }$ and $\pi _ { 1 }$ as the uniform distribution on $\mathcal { D } _ { N }$   
4: for $t = 1 , \cdots , T - 1$ do   
5: Calculate $\begin{array} { r } { \hat { r } _ { t } ( y ) = \widehat { \mathbb P } ( y \succ \pi _ { t } | x ) = \mathbb E _ { y ^ { \prime } \sim \pi _ { t } } [ \widehat { \mathbb P } ( y \succ y ^ { \prime } | x ) ] , \forall y \in \mathcal y _ { N } . } \end{array}$   
6: Calculate $\begin{array} { r } { \pi _ { t + 1 } ^ { \prime } = \mathrm { a r g m a x } _ { \pi \in \Delta ( \mathcal { Y } _ { N } ) } \langle \bar { \pi } , \hat { r } _ { t } \rangle - \beta \cdot \mathrm { K L } ( \pi | | \pi _ { t } ^ { \prime } ) . } \end{array}$   
7: Calculate $\pi _ { t + 1 } = \mathrm { a r g m a x } _ { \pi \in \Delta ( y _ { N } ) } \langle \pi , \hat { r } _ { t } \rangle - \beta \cdot \mathrm { K L } ( \pi | | \pi _ { t + 1 } ^ { \prime } )$   
8: end for   
9: Calculate $\begin{array} { r } { \hat { \pi } ( y | x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \pi _ { t } ( y | x ) , \forall y \in \mathcal { V } _ { N } . } \end{array}$   
10: Return $\hat { \pi } .$

NMD is inspired by Rakhlin and Sridharan [2013] for solving zero-sum games. The algorithm maintains two coupled policies: $\pi _ { t }$ and $\pi _ { t } ^ { \prime } ,$ both supported on the N sampled responses. At each iteration, $\pi _ { t + 1 } ^ { \prime }$ takes a mirror-descent step from $\pi _ { t }$ while staying close to $\pi _ { t } ^ { \bar { \prime } } \colon$

$$
\begin{array} { r } { \pi _ { t + 1 } ^ { \prime } = \operatorname * { a r g m a x } _ { \pi \in \Delta ( \mathcal { V } _ { N } ) } \widehat { \mathbb { P } } ( \pi \succ \pi _ { t } | x ) - \beta \cdot \mathbf { K L } ( \pi , \pi _ { t } ^ { \prime } ) , } \end{array}
$$

and $\pi _ { t + 1 }$ takes the same mirror descent step but stays close to $\pi _ { t + 1 } ^ { \prime } \colon$

$$
\begin{array} { r } { \pi _ { t + 1 } = \operatorname * { a r g m a x } _ { \pi \in \Delta ( \mathcal { V } _ { N } ) } \widehat { \mathbb { P } } ( \pi \succ \pi _ { t } | x ) - \beta \cdot \mathbf { K L } ( \pi , \pi _ { t + 1 } ^ { \prime } ) . } \end{array}
$$

Here $\pi _ { t + 1 } ^ { \prime }$ aims to maximize the (estimated) probability that it wins against policy $\pi _ { t } ,$ with an KL regularization term ensuring staying close to $\pi _ { t } ^ { \prime }$ (KL divergence is defined as ${ \mathrm { K L } } ( \pi , \pi ^ { \prime } ) : = $ $\begin{array} { r } { \sum _ { y } \pi ( y ) \log ( \frac { \pi ( y ) } { \pi ^ { \prime } ( y ) } ) ) } \end{array}$ . Both updates admit closed-form solutions. Denote $\widehat { r } _ { t } ( y ) : = \widehat { \mathbb { P } } ( y \succ \pi _ { t } | x ) =$ $\mathbb { E } _ { y ^ { \prime } \sim \pi _ { t } } [ \widehat { \mathbb { P } } ( y \succ y ^ { \prime } | x ) ] , \forall y \in \mathcal { V } _ { N }$ , we have

$$
\pi _ { t + 1 } ^ { \prime } = \operatorname * { a r g m a x } _ { \pi \in \Delta ( \mathcal { Y } _ { N } ) } \sum _ { i } \hat { r } _ { t } ( y _ { i } ) \pi ( y _ { i } ) - \beta \cdot \sum _ { i } \pi ( y _ { i } ) \log ( \frac { \pi ( y _ { i } ) } { \pi _ { t } ^ { \prime } ( y _ { i } ) } ) ,
$$

and the solution is

$$
\pi _ { t + 1 } ^ { \prime } ( y _ { i } ) = \frac { \pi _ { t } ^ { \prime } ( y _ { i } ) \exp ( \hat { r } _ { t } ( y _ { i } ) / \beta ) } { \sum _ { j } \pi _ { t } ^ { \prime } ( y _ { j } ) \exp ( \hat { r } _ { t } ( y _ { j } ) / \beta ) } .
$$

Theorem 2. For any prompt x, by setting $\begin{array} { r } { \beta = 2 , T = \lceil \frac { 2 \log N + 1 / 2 } { \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) } \rceil } \end{array}$ , the policy πˆ returned by Algorithm 2 has the duality gap

$$
\mathrm { D u a l G a p } ( \hat { \pi } ) \leq 5 \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) .
$$

when $\begin{array} { r } { N \geq 4 \log ( \frac { 2 } { \varepsilon ( x ) } ) \cdot \mathcal { C } _ { \mathrm { u n i } } ( x ) . } \end{array}$

We provide a proof sketch below, and defer the whole proof to Section C.

ProofSketch. The structure parallels the proof of Theorem 1. By the skew-symmetry of the zero-sum game, it suffices to upper-bound $\mathbb { P } ^ { * } ( \tilde { \pi } \succ \overline { { \hat { \pi } } } ) - 1 / 2$ , where $\tilde { \pi } : = \delta _ { y ^ { * } }$ ∗ with $\boldsymbol y ^ { * } \in \arg \operatorname* { m a x } _ { \boldsymbol y } \mathbb { P } ^ { * } ( \boldsymbol y \succ \hat { \pi } \mid$ x). Introducing the rejection-sampling approximation $\pi _ { R } .$ , we decompose

$$
\mathbb { P } ^ { * } ( \tilde { \pi } \succ \hat { \pi } ) \le \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } ) + \big \vert \mathbb { P } ^ { * } ( \tilde { \pi } \succ \hat { \pi } ) - \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } ) \big \vert .
$$

The second term is controlled by the property of approximate rejection sampling, similarly to that in Theorem 1. For the first term, we further decompose

$$
\begin{array} { r } { \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } ) \le \widehat { \mathbb { P } } ( \pi _ { R } \succ \hat { \pi } ) + \big | \widehat { \mathbb { P } } ( \pi _ { R } \succ \hat { \pi } ) - \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } ) \big | , } \end{array}
$$

bounding the empirical term via the cumulative-regret guarantee of Nash Mirror Descent on the game ${ \widehat { \mathbb { P } } } ,$ , and the transfer term by $\mathcal { C } _ { \mathrm { u n i } } ( x ) \varepsilon ( x )$ □

Remark 2. Our theory prescribes a specific $\beta = 2 .$ In practice, our experiments (Section 6) show that the performance of NMD is empirically robust to the choice of β.

Remark 3 (Comparison of BoN and NMD). Theorem 1 and Theorem 2 give the same duality gap bound $O ( \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) )$ , and both algorithms share the same query complexity: $O ( N )$ samples from $\pi _ { r e f } a n d O ( N ^ { 2 } )$ preference queries to ${ \widehat { \mathbb { P } } } .$ . The two algorithms differ in post-query complexity: BoN solves an LP in ${ \cal O } ( N ^ { 3 . 5 } \log ( 1 / \epsilon ) )$ time, while NMD runs $\begin{array} { r } { T = O \big ( \frac { \log ( N ) } { \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) } \big ) } \end{array}$ updates of $\cdot O ( N ^ { 2 } )$ each, for a total of $\begin{array} { r } { { \overleftrightarrow { O ( \frac { N ^ { 2 } \log N } { \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) } ) } } } \end{array}$

## 5 Lower Bound

The upper bounds in Theorems 1 and 2 show that both BoN and NMD achieve duality gap $O ( \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) )$ with an appropriate choice of parameters. A natural question is whether this rate is optimal. In this section, we answer this affirmatively by constructing problem instances on which any inference-time algorithm must incur duality gap $\dot { \Omega ( \varepsilon ( x ) C _ { \mathrm { u n i } } ( x ) ) }$ .

Theorem 3. Given a prompt x and K responses $\left\{ y _ { 1 } , \dots , y _ { K } \right\}$ , let $\pi _ { r e f } ( y _ { i } | x ) = 1 / K$ for all $i \in [ K ]$ For any alignment algorithm A and any $\varepsilon _ { 0 } \in ( 0 , \sqrt { 2 } / ( 3 K ) ]$ , there exist preference oracles $\mathbb { P } ^ { * }$ and Pb with $\varepsilon ( x ) = \varepsilon _ { 0 }$ such that:

$$
\mathrm { D u a l G a p } ( \mathcal { A } ( \widehat { \mathbb { P } } ) ) \geq \frac { \varepsilon _ { 0 } \cdot \mathcal { C } _ { \mathrm { u n i } } ( x ) } { 2 \sqrt { 2 } } .
$$

ProofSketch. We set the construction: $\begin{array} { r } { \widehat { \mathbb { P } } ( y _ { 1 } \succ y _ { j } ) = \frac { 1 } { 2 } + \delta _ { 0 } , \forall j \ge 2 . } \end{array}$ , and $\begin{array} { r } { \widehat { \mathbb { P } } ( y _ { j } \succ y _ { k } ) = \frac { 1 } { 2 } , \forall j , k \ge 2 } \end{array}$ for a small value $\delta _ { 0 } = \varepsilon _ { 0 } \mathcal { C } _ { \mathrm { u n i } } ( x ) / ( 2 \sqrt { 2 } )$ . This makes $y _ { 1 }$ the dominant strategy under $\widehat { \mathbb { P } }$ . We assume that $p : = \hat { \pi } ( y _ { 1 } )$ . Then we construct two real worlds A and $B$ . In world A, P<sup>∗</sup> amplifies $y _ { 1 } \mathrm { \cdot s }$ margin over $y _ { 2 } .$ , keeping $y _ { 1 }$ dominant. In world $B , \mathbb { P } _ { B } ^ { * }$ flips the $( y _ { 1 } , y _ { 2 } )$ entry so that y<sub>2</sub> narrowly beats $y _ { 1 }$ making $y _ { 2 }$ the new dominant strategy. The duality gaps in the two worlds scale as $2 ( 1 - p ) \delta _ { 0 }$ and $2 p \delta _ { 0 }$ respectively, and the adversary picks the larger. The algorithm minimizes the maximum at $p = 1 / 2$ , leaving a duality gap of at least $\delta _ { 0 } = \varepsilon _ { 0 } \mathcal { C } _ { \mathrm { u n i } } ( x ) / ( 2 \sqrt { 2 } )$ □

The full proof is relegated to Section D.

Remark 4 (Optimality of BoN and NMD). Combining Theorem 3 with Theorem 1 and Theorem 2 yields a tight characterization: the optimal duality gap for inference-time alignment with general preferences is $\Theta ( \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) )$ ), achieved by both of our algorithms.

Remark 5 (Contrast with reward-based alignment.). The duality gap $\Omega ( \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) )$ stands in sharp contrast to the optimal ratefor the reward-based inference time alignment, which [Huang et al., 2025] show is $\Theta ( \varepsilon _ { R M } ( x ) \sqrt { \mathcal { C } ^ { \pi ^ { * } } ( x ) } )$ . Here $\varepsilon _ { R M } ( x )$ is the reward model error, and $\pi ^ { * }$ is the comparator policy. Two structural differences are notable. First, the coverage dependence is linear in $\mathcal { C } _ { u n i } ( x )$ in our setting versus $\sqrt { { \mathscr C } ^ { \pi ^ { * } } ( x ) }$ in the reward setting. This implies that bad coverage of the base policy hurts more when preferences are general than when they are scalar. Second, our relevant coverage quantity is the universal coverage $\mathcal { C } _ { u n i } ( x )$ , not single-policy coverage $\boldsymbol { \mathcal { C } ^ { \pi ^ { * } } }$ , reflecting that inference-time alignment with general preferences is fundamentally harder than its reward-based counterpart.

## 6 Experiments

We evaluate our proposed alignment algorithms, Best-of-Nash and Nash Mirror Descent, on three datasets and study how their performance varies with sample size $N ,$ regularization parameter $\beta ,$ , and the choice of base models and preference models.

## 6.1 Setup

Datasets and Models. We conduct experiments on three preference datasets that are commonly used for training and evaluating LLM alignment: TLDR (text summarization), HelpSteer2 (general-purpose helpfulness) [Wang et al., 2024b], and UltraFeedback [Cui et al., 2023] (instruction following). We sample 100 prompts for evaluation in each dataset. In our experiments, we consider three supervised fine-tuned (SFT) models as base models: LLaMA3-SFT (8B) [Dong et al., 2024], Mistral-Instruct (7B), and Gemma-SFT (2B). Two aligned models are selected as our estimate preference oracle, P<sup>ˆ</sup>: LLaMA3-PM (8B) [Dong et al., 2024] and PairRM (0.4B) [Jiang et al., 2023].

<table><tr><td>Dataset</td><td>Base SFT</td><td>BoN</td><td>NMD</td></tr><tr><td>TLDR</td><td>62.9%</td><td>73.5%</td><td>73.1%</td></tr><tr><td>HelpSteer2</td><td>44.1%</td><td>68.8%</td><td>67.8%</td></tr><tr><td>UltraFeedback</td><td>23.9%</td><td>48.8%</td><td>46.7%</td></tr></table>

Table 1: Comparison of expected win-rate (EWR) across three datasets. BoN and NMD sample N = 64 responses from LLaMA3-SFT and we use LLaMA3-PM as preference model; $\beta = 1$ for NMD.

Evaluation. Our headline metric is expected win rate (EWR := win + draw/2). Results in the main paper mainly use LLaMA3-SFT as the base model until otherwise stated. We also compare results to LLaMA3-DPO (8B), a fine-tuned version of LLaMA3-SFT, to illustrate the performance of our algorithm against a model fine-tuned on preference data. We use an LLM as a judge [Zheng et al., 2023] since alignment is inherently preference-based, and LLMs provide a scalable and consistent proxy for human evaluations of model outputs. Concretely, we compare each generated response against a reference answer from the dataset, with judgments produced by DeepSeek-V4-Flash. To control positional bias [Zheng et al., 2023, Wang et al., 2024a], we query the judge twice per pair with orderings swapped, counting a win/loss only when both orderings agree and a draw otherwise.

## 6.2 Main Results.

Table 1 presents our headline comparison: BoN and NMD against the SFT base policy across all three datasets. Both BoN and NMD outperform the base SFT across all three datasets: both algorithms have an improvement of roughly 10% in TLDR, and ≈ 24% in HelpSteer2 and UltraFeedback. BoN and NMD have similar expected win-rates on all three datasets, consistent with the theoretical guarantees in Theorem 1 and Theorem 2.

Comparison to DPO Figure 1 shows that BoN and NMD match the expected win-rate of LLaMA3- DPO with a reasonable number of samples, without any parameter updates. This suggests that careful inference-time alignment can substitute for fine-tuning when the preference oracle is sufficiently strong.

Sample size N and regularization β. Figure 1a shows BoN’s expected win-rate on TLDR as N increases. EWR rises monotonically from 62.9% at N = 1 to 73.5% at N = 64, approaching the DPO baseline. This confirms the predicted scaling: with more samples, BoN better approximates the Nash equilibrium. Figure 1b shows that NMD’s EWR is flat across β, thus removing the need for hyper-parameter tuning.

Preference oracle and base model. We show the effect of the preference model Pb and base model π in Figure 2. The left panel compares LLaMA3-PM (8B) and PairRM (0.4B) as the preference oracle with LLaMA3-SFT fixed as the base model: despite the 20 times difference in size, the two yield rather similar EWRs on TLDR for both BoN and NMD. The right panel compares the three base models with LLaMA3-PM fixed as the preference oracle: Mistral-Instruct attains the highest EWR, followed by LLaMA3-SFT, with Gemma-SFT trailing by a clear margin.

Post-query alignment run-time. As we show in Remark 3, BoN and NMD share the same query complexity, but differ in post-query computation. Figure 3 compares the post-query time of the two algorithms as N grows: BoN scales rapidly with N, while NMD remains nearly flat. This is consistent with our time complexity analysis.

Omitted experiment details and additional experiments are deferred to Section E, including comparing our methods against three baseline methods.

![](images/baeab72a4bbfb761b115ec62a047bcf9de81658251c5b257184a4751c015b6e0.jpg)  
(a) BoN with varying sample size N.

![](images/1401821ef90527fa6e94ace1dd184ea9cda90bffe520f06f2e0bda84f1e676ca.jpg)  
(b) NMD with varying β and fixed $N = 6 4$

Figure 1: Win/Draw/Lose distribution and expected win-rate (Win + Draw/2) on the TLDR dataset. The dashed line indicates the baseline score for LLaMA3-DPO. We use LLaMA3-SFT as the reference policy and LLaMA3-PM as the preference oracle Pb.  
![](images/8101d1e16f9e83b08b8accdd72f7aa04520bd681d8c17a2b58755139cc55506b.jpg)

![](images/f341ae603f8afad149faca19ddc1d0ece855b23884f15ad9549774bb1f706a9e.jpg)  
Figure 2: Win/Draw/Lose distribution on the TLDR dataset. Left: we compare the Llama3-PM and PairRM for preference model, with the base model = LLama3-SFT. Right: we compare LLama3-SFT, Mistral-instruct, and Gemma-SFT for base model, with the preference model = LLama3-PM.

## 7 Conclusion and Limitations

We initiate the study of inference-time alignment under general preferences, formulating the problem as computing a Nash equilibrium of a two-player zero-sum game between policies under an imperfect preference oracle. We propose two algorithms, Best-of-Nash (BoN) and Nash Mirror Descent (NMD), and prove that both algorithms achieve a duality gap of $O ( \varepsilon ( x ) C _ { \mathrm { u n i } } ( x ) )$ , which we show to be tight via a matching lower bound. The two algorithms differ in post-query computation: BoN solves a linear programming in ${ \cal O } ( N ^ { 3 . 5 } \log ( 1 / \epsilon ) )$ time, while NMD requires $\begin{array} { r } { \dot { O } ( \frac { \bar { N } ^ { 2 } } { \varepsilon ( x ) C _ { \mathrm { u n i } } ( x ) } \log ( N ) ) } \end{array}$ time.

Our work has several limitations. First, our theoretical guarantees depend on the quality of preference models via $\varepsilon ( x )$ . When $\widehat { \mathbb { P } }$ is a poor proxy for the true preference $\mathbb { P } ^ { * }$ , for example, on prompts that fall outside the distribution on which $\widehat { \mathbb { P } }$ is trained, the methods may inherit the biases of Pb. Second, inference-time alignment can only re-weight responses that are reachable under $\pi _ { \mathrm { r e f } }$ . When the base policy assigns negligible probability to high-quality responses, neither BoN nor NMD can recover them, and fine-tuning remains necessary. Lastly, our analysis treats each prompt independently and provides per-prompt guarantees, thereby neglecting shared structure across prompts such as similar tasks or styles.

![](images/41981e0d0f7e1d55d54d193a3dd3bd0679d1d6673b7ef22ac8940f08ad2191d4.jpg)  
Figure 3: Post-query time comparison of BoN and NMD for varying size N.

We end this section with a few future directions. Both BoN and NMD require $O ( N ^ { 2 } )$ pairwise queries to Pb, which can be a huge query cost; reducing this cost via active selection of pairs is an interesting direction. Another direction is to combine our inference-time theory with fine-tuning with general preferences. Finally, exploiting structure across prompts could improve sample-efficiency beyond the per-prompt rates we establish here.

## Acknowledgments

We thank the anonymous reviewers for their comments and constructive feedback. HH acknowledges support from the National Science Foundation, NSF Awards IIS-2144413 and IIS-2107173.

## References

Ilan Adler. The equivalence of linear programs and zero-sum games. International Journal ofGame Theory, 42(1):165–177, 2013.

Anthropic. Introducing Claude Opus 4.7. https://www.anthropic.com/news/claude-opus-4 -7, 2026.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR, 2024.

Adam Block and Yury Polyanskiy. The sample complexity of approximate rejection sampling with applications to smoothed online learning. In The Thirty Sixth Annual Conference on Learning Theory, pages 228–273. PMLR, 2023.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Daniele Calandriello, Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires, Pierre Harvey Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, et al. Human alignment of large language models through online preference optimisation. arXiv preprint arXiv:2403.08635, 2024.

Ruizhe Chen, Xiaotian Zhang, Meng Luo, Wenhao Chai, and Zuozhu Liu. Pad: Personalized alignment of llms at decoding-time. arXiv preprint arXiv:2410.04070, 2024.

Yinlam Chow, Guy Tennenholtz, Izzeddin Gur, Vincent Zhuang, Bo Dai, Sridhar Thiagarajan, Craig Boutilier, Rishabh Agarwal, Aviral Kumar, and Aleksandra Faust. Inference-aware fine-tuning for best-of-n sampling in large language models. arXiv preprint arXiv:2412.15287, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback, 2023.

Qiwen Cui and Simon S Du. When are offline two-player zero-sum markov games solvable? Advances in Neural Information Processing Systems, 35:25779–25791, 2022.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

Yoav Freund and Robert E Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 29(1-2):79–103, 1999.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

Google. Gemini 3.1 Pro Model Card. https://storage.googleapis.com/deepmind-media/M odel-Cards/Gemini-3-1-Pro-Model-Card.pdf, 2026.

Lin Gui, Cristina Gârbacea, and Victor Veitch. Bonbon alignment for large language models and the sweetness of best-of-n sampling. Advances in Neural Information Processing Systems, 37: 2851–2885, 2024.

Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Akshay Krishnamurthy, and Dylan J Foster. Is best-of-n the best of them? coverage, scaling, and optimality in inference-time alignment. arXiv preprint arXiv:2503.21878, 2025.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. Llm-blender: Ensembling large language models with pairwise ranking and generative fusion. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14165–14178, 2023.

Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. Args: Alignment as reward-guided search. arXiv preprint arXiv:2402.01694, 2024.

Roberto-Rafael Maura-Rivero, Marc Lanctot, Francesco Visin, and Kate Larson. Jackpot! alignment as a maximal lottery. arXiv preprint arXiv:2501.19266, 2025.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Côme Fiegel, et al. Nash learning from human feedback. In Forty-first International Conference on Machine Learning, 2024.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Arkadi Nemirovski. Prox-method with rate of convergence o (1/t) for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 15(1):229–251, 2004.

Yurii Nesterov. Primal-dual subgradient methods for convex problems. Mathematical programming, 120(1):221–259, 2009.

OpenAI. GPT-5.4 Thinking System Card. https://deploymentsafety.openai.com/gpt-5-4 -thinking/gpt-5-4-thinking.pdf, 2026.

QwenTeam. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Sasha Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. Advances in Neural Information Processing Systems, 26, 2013.

Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Ruizhe Shi, Yifang Chen, Yushi Hu, Alisa Liu, Hannaneh Hajishirzi, Noah A Smith, and Simon S Du. Decoding-time language model alignment with multiple objectives. Advances in Neural Information Processing Systems, 37:48875–48920, 2024.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471, 2022.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.

Benedikt Stroebl, Sayash Kapoor, and Arvind Narayanan. Inference scaling flaws: The limits of llm resampling with imperfect verifiers. arXiv preprint arXiv:2411.17501, 3(8):14, 2024.

Gokul Swamy, Christoph Dann, Rahul Kidambi, Zhiwei Steven Wu, and Alekh Agarwal. A minimaximalist approach to reinforcement learning from human feedback. arXiv preprint arXiv:2401.04056, 2024.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, et al. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, 2024a.

Zhilin Wang, Alexander Bukharin, Olivier Delalleau, Daniel Egert, Gerald Shen, Jiaqi Zeng, Oleksii Kuchaiev, and Yi Dong. Helpsteer2-preference: Complementing ratings with preferences. arXiv preprint arXiv:2410.01257, 2024b.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. arXiv preprint arXiv:2405.00675, 2024.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Chenlu Ye, Wei Xiong, Yuheng Zhang, Nan Jiang, and Tong Zhang. A theoretical analysis of nash learning from human feedback under general kl-regularized preference. arXiv preprint arXiv:2402.07314, 4(5):10, 2024.

Zhuohao Yu, Zhiwei Steven Wu, and Adam Block. From curiosity to caution: Mitigating reward hacking for best-of-n with pessimism. arXiv preprint arXiv:2604.04648, 2026.

Yuheng Zhang, Yu Bai, and Nan Jiang. Offline learning in markov games with general function approximation. In International Conference on Machine Learning, pages 40804–40829. PMLR, 2023.

Yuheng Zhang, Dian Yu, Baolin Peng, Linfeng Song, Ye Tian, Mingyue Huo, Nan Jiang, Haitao Mi, and Dong Yu. Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning. arXiv preprint arXiv:2407.00617, 2024.

Yuheng Zhang, Dian Yu, Tao Ge, Linfeng Song, Zhichen Zeng, Haitao Mi, Nan Jiang, and Dong Yu. Improving llm general preference alignment via optimistic online mirror descent. arXiv preprint arXiv:2502.16852, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Han Zhong, Wei Xiong, Jiyuan Tan, Liwei Wang, Tong Zhang, Zhaoran Wang, and Zhuoran Yang. Pessimistic minimax value iteration: Provably efficient equilibrium learning from offline datasets. In International Conference on Machine Learning, pages 27117–27142. PMLR, 2022.

Runlong Zhou, Maryam Fazel, and Simon S Du. Extragradient preference optimization (egpo): Beyond last-iterate convergence for nash learning from human feedback. arXiv preprint arXiv:2503.08942, 2025.

## A Rejection Sampling

In this section, we introduce the rejection sampling algorithm (Algorithm 3) [Block and Polyanskiy, 2023, Huang et al., 2025]. The algorithm draws N samples i.i.d. from $\pi _ { \mathrm { r e f } } ( \cdot | x )$ . For each $y _ { i } .$ , it samples a Bernoulli random variable $\xi _ { i }$ where $\begin{array} { r } { \mathbb { P } ( \xi _ { i } = 1 \mid y _ { i } ) = \operatorname* { m i n } \left\{ \frac { w ( y _ { i } | x ) } { M } , 1 \right\} } \end{array}$ . The algorithm returns $y _ { i }$ if the Bernoulli random variable $\xi _ { i } = 1$ . If $\xi _ { i } = 0$ for $\forall i \in [ N ]$ , then the algorithm randomly samples a response from the reference policy $\pi _ { \mathrm { r e f } } .$

The main purpose of the rejection sampling algorithm is to approximate a target policy $\pi ( \cdot | x )$ from $\pi _ { \mathrm { r e f } }$ by setting $\begin{array} { r } { w ( \cdot | x ) = \frac { \pi ( \cdot | x ) } { \pi _ { \mathrm { r e f } } ( \cdot | x ) } } \end{array}$ . In the analysis of this paper, we mainly use the distribution induced by rejection sampling as an intermediate step.

Algorithm 3 Rejection Sampling (RejectionSampling $_ { N , M } ( w ; \pi _ { \mathrm { r e f } } , x ) )$   
1: Input: Prompt x, base policy $\pi _ { \mathrm { r e f } } ,$ importance weight $w ,$ , truncation level $M .$   
2: $\mathrm { D r a w } = ( y _ { 1 } , \underbrace { \dots } _ { \cdot \cdot } , y _ { N } , y _ { N + 1 } ) \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ i.i.d.   
3: for $i = 1 \ldots N$ do   
4: Sample Bernoulli random variable $\xi _ { i }$ such that $\begin{array} { r } { \mathbb { P } ( \xi _ { i } = 1 \mid y _ { i } ) = \operatorname* { m i n } \left\{ \frac { w ( y _ { i } \mid x ) } { M } , 1 \right\} } \end{array}$   
5: if $\xi _ { i } = 1$ then   
6: Return response $y = y _ { i } .$   
7: end if   
8: end for   
9: Return response $y = y _ { N + 1 } $

## B Omitted Proofs from Section 3

In this section, let ${ \mathcal { V } } _ { N } = \{ y _ { 1 } , \ldots , y _ { N } \}$ denote the candidate set drawn i.i.d. from $\pi _ { \mathrm { r e f } } ( \cdot \mid x )$ , let πˆ be the policy returned by Algorithm 1, and let $\tilde { \pi } = \delta _ { y ^ { * } }$ with $y ^ { \ast } \in a r g m a x _ { y \in { \mathcal { y } } } \mathbb { P } ^ { \ast } ( y \succ \hat { \pi } | x )$ denote a pure best response to πˆ under the true preference. Since the maximum of the linear functional $\pi \mapsto \mathbb { P } ^ { * } ( \pi \succ \hat { \pi } | x )$ is attained at a point mass, we have that $\mathrm { D u a l G a p } ( \hat { \pi } ) = 2 \mathbb { P } ^ { * } ( \tilde { \pi } \succ \hat { \pi } | x ) - 1$ . We define $\pi _ { R }$ as the distribution induced by RejectionSampling $\cdot N - 1 , M  \left( \frac { \tilde { \pi } } { \pi _ { \mathrm { r e f } } } ; \pi _ { \mathrm { r e f } } , x \right)$ run on the same candidate set $\mathcal { D } _ { N }$ . Under this coupling, every branch of Algorithm 3 returns an element of ${ \mathcal { V } } _ { N } .$ , so conditionally on $\mathcal { D } _ { N }$ we have $\pi _ { R } \mid \mathcal { V } _ { N } \in \Delta ( \mathcal { V } _ { N } )$ . We write $D _ { \mathrm { T V } } ( \tilde { \pi } , \pi _ { R } | \mathcal { V } _ { N } )$ for the TV distance between π˜ and $\pi _ { R }$ given $\mathcal { D } _ { N }$

Lemma 1. With ${ \hat { \pi } } , { \tilde { \pi } } ,$ , and π<sub>R</sub> as defined above, we have

$$
\mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } | x ) \le \frac { 1 } { 2 } + \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x )
$$

Proof. All displays below hold conditionally on $\mathcal { D } _ { N }$ , and taking expectation over the draw of $\mathcal { D } _ { N }$ at the end yields the stated bound. By the construction, $\{ { \widehat { \mathbb { P } } } ( y _ { i } \succ y _ { j } | x ) \} _ { i \in [ N ] , j \in [ N ] }$ is a two-player constant-sum matrix game with domain on the dataset ${ \mathcal { V } } _ { N } = \{ y _ { i } \} _ { i \in [ N ] } \subseteq { \dot { \mathcal { V } } }$ . Since $( \hat { \pi } , \hat { \pi } )$ is a Nash Equilibrium computed by Algorithm 1, we have

$$
\begin{array} { r } { \left( \widehat { \pi } , \widehat { \pi } \right) = \operatorname * { a r g m a x } _ { \pi \in \Delta ( \mathcal { V } _ { N } ) } \operatorname * { a r g m i n } _ { \pi ^ { \prime } \in \Delta ( \mathcal { V } _ { N } ) } \widehat { \mathbb { P } } ( \pi \succ \pi ^ { \prime } ) . } \end{array}
$$

Since $\pi _ { R } \mid \mathcal { V } _ { N } \in \Delta ( \mathcal { V } _ { N } )$ , the definition of the Nash equilibrium gives

$$
\begin{array} { r l } & { \widehat { \mathbb { P } } ( \pi _ { R } \succ \widehat \pi | x ) - \widehat { \mathbb { P } } ( \widehat \pi \succ \widehat \pi | x ) } \\ & { \ = \mathbb { E } _ { \mathcal { V } _ { N } } \mathbb { E } _ { y \sim \pi _ { R } | \mathcal { V } _ { N } , y ^ { \prime } \sim \widehat \pi | \mathcal { V } _ { N } , y ^ { \prime \prime } \sim \widehat \pi | \mathcal { V } _ { N } } [ \widehat { \mathbb { P } } ( y \succ y ^ { \prime \prime } | x ) - \widehat { \mathbb { P } } ( y ^ { \prime } \succ y ^ { \prime \prime } | x ) ] } \\ & { \ \leq 0 . } \end{array}
$$

More concretely, given the column player’s policy πˆ, the best policy that the row player takes should be πˆ. On the other hand, conditioned on $\mathcal { D } _ { N }$ , we calculate that

$$
\begin{array} { r l } & { \mathbb { P } ^ { * } ( \pi _ { R } \succ \widehat { \pi } | \mathcal { Y } _ { N } ) - \widehat { \mathbb { P } } ( \pi _ { R } \succ \widehat { \pi } | \mathcal { Y } _ { N } ) | } \\ & { \leq \mathbb { E } _ { y \sim \pi _ { R } | \mathcal { Y } _ { N } } \mathbb { E } _ { y ^ { \prime } \sim \pi _ { R } ^ { * } } | \mathcal { Y } _ { N } | \mathcal { Y } ^ { * } ( y \vee \mathcal { Y } ^ { \prime } | x ) - \widehat { \mathbb { P } } ( y \succ y ^ { \prime } | x ) | } \\ & { \leq \mathbb { E } _ { y \sim \pi _ { R } } \mathbb { E } _ { y ^ { \prime } \sim \pi _ { R } ^ { * } } [ \frac { \pi _ { R } ( y ) } { \pi _ { \mathrm { e f f } } ( y | x ) } \frac { \widehat { \pi } ( y ^ { \prime } | \mathcal { Y } _ { N } ) } { \pi _ { \mathrm { e f f } } ( y ^ { \prime } | x ) } ] \mathbb { P } ^ { * } ( y \succ y ^ { \prime } | x ) - \widehat { \mathbb { P } } ( y \succ y ^ { \prime } | x ) | ] } \\ &  \leq \sqrt { \mathbb { E } _ { y \sim \pi _ { R } ^ { * } } [ ( \frac { \pi _ { R } ( y | \mathcal { Y } _ { N } ) } { \pi _ { \mathrm { e f f } } ( y | x ) } ) ^ { 2 } ] \mathbb { E } _ { y ^ { \prime } \sim \pi _ { R } \in [ ( \frac { \widehat { \pi } ( y ^ { \prime } | \mathcal { Y } _ { N } ) } { \pi _ { \mathrm { e f f } } ( y ^ { \prime } | x ) } ) ^ { 2 } ] } \sqrt { \mathbb { E } _ { y \sim \pi _ { R } } \mathbb { E } _ { y ^ { \prime } \sim \pi _ { R } \in [ | \mathbb { P } ^ { * } ( y \succ y ^ { \prime } | x ) } - \widehat { \mathbb { P } } ( y \succ y ^ { \prime } | x ) | ^ { 2 } ] } } \\ &  \leq \sqrt  \mathbb { E } _ { y \sim \pi _ { R } } [ \pi _  \end{array}
$$

Combining the above computation, we have

$$
\begin{array} { r l } & { \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } | x ) \leq \mathbb { E } _ { \mathcal { Y } _ { N } } [ \widehat { \mathbb { P } } ( \pi _ { R } \succ \hat { \pi } \mid \mathcal { Y } _ { N } ) + \left| \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } \mid \mathcal { Y } _ { N } ) - \widehat { \mathbb { P } } ( \pi _ { R } \succ \hat { \pi } \mid \mathcal { Y } _ { N } ) \right| ] } \\ & { \qquad \leq \mathbb { E } _ { \mathcal { Y } _ { N } } [ \widehat { \mathbb { P } } ( \hat { \pi } \succ \hat { \pi } \mid \mathcal { Y } _ { N } ) + \left| \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } \mid \mathcal { Y } _ { N } ) - \widehat { \mathbb { P } } ( \pi _ { R } \succ \hat { \pi } \mid \mathcal { Y } _ { N } ) \right| ] } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } + \mathbb { E } _ { \mathcal { Y } _ { N } } [ \left| \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } | x ) - \widehat { \mathbb { P } } ( \pi _ { R } \succ \hat { \pi } | x ) \right| ] } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } + \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) , } \end{array}
$$

and the proof is complete.

Lemma 2. With π˜ and $\pi _ { R }$ as defined above with $\begin{array} { r } { M = \frac { N - 1 } { \log ( 4 / \varepsilon ^ { 2 } ( x ) ) } } \end{array}$ , and any $\sigma ( \mathcal { V } _ { N } )$ )-measurable $p o l i c y \hat { \pi } \in \Delta ( \mathcal { V } _ { N } )$ , we have that

$$
D _ { T V } ( \tilde { \pi } , \pi _ { R } ) \leq \frac { 1 } { 4 } \mathcal { C } _ { \mathrm { u n i } } ( x ) \varepsilon ^ { 2 } ( x )
$$

when $\begin{array} { r } { N \geq 4 \log ( \frac { 2 } { \varepsilon ( x ) } ) \cdot \mathcal { C } _ { u n i } ( x ) } \end{array}$ . The statement holdsfor the output ofAlgorithm 1 and Algorithm 2.

Proof. We omit the dependence on x. First, the condition on N implies $N - 1 \geq 2 \mathcal { C } _ { \mathrm { u n i } } \log ( 2 / \varepsilon ) =$ $\mathcal { C } _ { \mathrm { u n i } } \breve { \log ( 4 / \varepsilon ^ { 2 } ) }$ , i.e. $M \stackrel { \cdot } { \geq } \mathcal { C } _ { \mathrm { u n i } } \geq 1 / \pi _ { \mathrm { r e f } } ( y )$ for any $y \in \mathcal { V }$ ; hence $w ( y ) / M \leq 1$ for all $y$ and the acceptance probability in Algorithm 3 is exactly $w ( y ) / M$ . Since $w = \tilde { \pi } / \pi _ { \mathrm { r e f } }$ is supported on the single point $y ^ { * }$ , only candidates equal to $y ^ { * }$ can be accepted. Therefore, conditioned on $\mathcal { D } _ { N }$ , we have

$$
\pi _ { R } | \mathcal { V } _ { N } = q \delta _ { y _ { N } } + ( 1 - q ) \delta _ { y ^ { * } } , q = ( 1 - \frac { 1 } { M \pi _ { \mathrm { r e f } } ( y ^ { * } ) } ) ^ { m ( y ^ { * } ) } ,
$$

where $m ( y ) = \# \{ i \leq N - 1 : y _ { i } = y \}$ , and consequently $D _ { \mathrm { T V } } ( \tilde { \pi } , \pi _ { R } | \mathcal { V } _ { N } ) \le q$

Then we bound $q \colon$

$$
q \leq \sum _ { y \in \mathcal { Y } } ( 1 - \frac { 1 } { M \pi _ { \mathrm { r e f } } ( y ) } ) ^ { m ( y ) } .
$$

For any fixed $y , m ( y ) \sim \mathrm { B i n } ( N - 1 , \pi _ { \mathrm { r e f } } ( y ) )$ , and the probability generating function of the binomial gives

$$
\mathbb { E } [ ( 1 - \frac { 1 } { M \pi _ { \mathrm { r e f } } ( y ) } ) ^ { m ( y ) } ] = ( 1 - \pi _ { \mathrm { r e f } } ( y ) \cdot \frac { 1 } { M \pi _ { \mathrm { r e f } } ( y ) } ) ^ { N - 1 } = ( 1 - \frac { 1 } { M } ) ^ { N - 1 } \le e ^ { - ( N - 1 ) / M } .
$$

By the definition of $\mathcal { C } _ { \mathrm { u n i } } ( x )$ we have $\begin{array} { r } { \mathcal { C } _ { \mathrm { u n i } } ( x ) \geq \frac { 1 } { \pi _ { \mathrm { r e f } } ( y ) } } \end{array}$ for any y, which along with $\begin{array} { r } { \sum _ { y } \pi _ { \mathrm { r e f } } ( y ) = 1 } \end{array}$ implies $| \mathcal { V } | \le \mathcal { C } _ { \mathrm { u n i } }$ . By substituting $\begin{array} { r } { M = \frac { N - 1 } { \log ( 4 / \varepsilon ^ { 2 } ) } } \end{array}$ , we bound

$$
\begin{array} { r l } & { D _ { \mathrm { T V } } ( \tilde { \pi } , \pi _ { R } ) \leq \mathbb { E } [ q ] } \\ & { \qquad \leq { \mathcal C } _ { \mathrm { u n i } } e ^ { - ( N - 1 ) / M } } \\ & { \qquad \leq { \mathcal C } _ { \mathrm { u n i } } \cdot \frac { \varepsilon ^ { 2 } } { 4 } . } \end{array}
$$

Then we take the expectation on $\mathcal { D } _ { N }$ and derive the desired result.

## C Omitted Proofs from Section 4

The following technical lemma takes the result from optimistic mirror descent literature [Rakhlin and Sridharan, 2013] that studies the general Bregman divergence. The KL divergence can be written by the Bregman divergence property:

$$
\mathrm { K L } ( \pi \| \pi ^ { \prime } ) = D _ { \psi } ( \pi , \pi ^ { \prime } ) = \psi ( \pi ) - \psi ( \pi ^ { \prime } ) - \langle \nabla \psi ( \pi ^ { \prime } ) , \pi - \pi ^ { \prime } \rangle ,
$$

where $\begin{array} { r } { \psi ( \pi ) = \sum _ { y } \pi ( y ) \log \pi ( y ) } \end{array}$

Lemma 3 (Corollary of Lemma 1 in Rakhlin and Sridharan [2013]). Denote $\pi _ { t }$ and $\hat { r } _ { t }$ as is computed in Algorithm 2. For any $\pi ^ { \prime } \in \Delta ( \mathcal { V } _ { N } )$ , we have that

$$
\sum _ { t = 1 } ^ { T } \langle \pi ^ { \prime } - \pi _ { t } , \hat { r } _ { t } \rangle \leq \beta \cdot K L ( \pi ^ { \prime } \| \pi _ { 1 } ^ { \prime } ) + \frac { 1 } { \beta } \sum _ { t = 1 } ^ { T } \| \hat { r } _ { t } - \hat { r } _ { t - 1 } \| _ { \infty } ^ { 2 } - \frac { \beta } { 4 } \sum _ { t = 2 } ^ { T } \| \pi _ { t } - \pi _ { t - 1 } \| _ { 1 } ^ { 2 } .
$$

In the following, we provide a technical lemma that quantifies the samples required to compute an approximate Nash policy under the zero-sum matrix game ${ \widehat { \mathbb { P } } } .$

Lemma 4. Denote $\begin{array} { r } { \hat { \pi } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } } \end{array}$ π<sub>t</sub> as the policy returned by Algorithm 2. Given data size N and a sufficiently small value $\epsilon > 0 ,$ we set $\beta = 2$ and $T = \lceil ( 2 \log N + 1 / 2 ) / \epsilon \rceil$ . Then for any $\pi ^ { \prime } \in \Delta ( \tilde { \mathcal { V } } _ { N } )$ , we have

$$
\widehat { \mathbb { P } } ( \pi ^ { \prime } \succ \hat { \pi } ) \leq \frac { 1 } { 2 } + \epsilon .
$$

Proof. Define $\hat { r } _ { 0 } : = 0$ . Recall that $\begin{array} { r } { \hat { \pi } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \pi _ { t } } \end{array}$ . For any policy $\pi ^ { \prime } \in \Delta ( \mathcal { V } _ { N } )$ , we have that

$$
\begin{array} { l } { \displaystyle \widehat { \mathbb { P } } ( \pi ^ { \prime } \succ \hat { \pi } ) = \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \widehat { \mathbb { P } } ( \pi ^ { \prime } \succ \pi _ { t } ) } \\ { \displaystyle \quad \leq \frac { 1 } { 2 } + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } [ \widehat { \mathbb { P } } ( \pi ^ { \prime } \succ \pi _ { t } ) - \widehat { \mathbb { P } } ( \pi _ { t } \succ \pi _ { t } ) ] } \\ { \displaystyle \quad \leq \frac { 1 } { 2 } + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \langle \pi ^ { \prime } - \pi _ { t } , \hat { r } _ { t } \rangle . } \end{array}
$$

Note that for any $\pi ^ { \prime } \in \Delta ( \mathcal { V } _ { N } )$ and $\pi _ { 1 } ^ { \prime }$ is a uniform distribution on $\mathcal { D } _ { N }$ , we have

$$
\begin{array} { r } { \mathrm { K L } ( \pi ^ { \prime } \| \pi _ { 1 } ^ { \prime } ) \leq \log ( N ) . } \end{array}
$$

For $t \geq 2 .$ , we have $\begin{array} { r } { | \hat { r } _ { t } ( y ) - \hat { r } _ { t - 1 } ( y ) | = | \sum _ { y ^ { \prime } } \hat { P } ( y \succ y ^ { \prime } ) ( \pi _ { t } ( y ^ { \prime } ) - \pi _ { t - 1 } ( y ^ { \prime } ) ) | \leq \| \pi _ { t } - \pi _ { t - 1 } \| _ { 1 } . } \end{array}$ Substituting into Lemma 3, for any $\beta \geq 2 \colon$

$$
\begin{array} { r } { \displaystyle \sum _ { t = 1 } ^ { T } \langle \pi ^ { \prime } - \pi _ { t } , \hat { r } _ { t } \rangle \leq \beta \log N + \frac { 1 } { \beta } \| \hat { r } _ { 1 } \| _ { \infty } ^ { 2 } + \bigg ( \frac { 1 } { \beta } - \frac { \beta } { 4 } \bigg ) \displaystyle \sum _ { t = 2 } ^ { T } \| \pi _ { t } - \pi _ { t - 1 } \| _ { 1 } ^ { 2 } \leq \beta \log N + \frac { 1 } { \beta } } \end{array}
$$

where the last inequality holds because $\| \hat { r } _ { 1 } \| _ { \infty } \leq 1$ and, for any $\begin{array} { r } { \beta \ge 2 , \frac { 1 } { \beta } - \frac { \beta } { 4 } \le 0 } \end{array}$ . Setting $\beta = 2$ and $T = \lceil ( 2 \log N + 1 / 2 ) / \epsilon \rceil$ yields

$$
\hat { P } ( \pi ^ { \prime } \succ \hat { \pi } ) \leq \frac { 1 } { 2 } + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \langle \pi ^ { \prime } - \pi _ { t } , \hat { r } _ { t } \rangle \leq \frac { 1 } { 2 } + \epsilon
$$

for all $\pi ^ { \prime } \in \Delta ( \mathcal { V } _ { N } )$

With the above technical lemma, we can finally show the duality gap bound for NMD.

Theorem 2. For any prompt x, by setting $\begin{array} { r } { \beta = 2 , T = \lceil \frac { 2 \log N + 1 / 2 } { \varepsilon ( x ) \mathcal { C } _ { u n i } ( x ) } \rceil } \end{array}$ , the policy πˆ returned by Algorithm 2 has the duality gap

$$
\mathrm { D u a l G a p } ( \hat { \pi } ) \leq 5 \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x ) .
$$

when $\begin{array} { r } { N \geq 4 \log ( \frac { 2 } { \varepsilon ( x ) } ) \cdot \mathcal { C } _ { \mathrm { u n i } } ( x ) } \end{array}$

Proof. First fix data size N. For any $\pi \in \Delta ( \mathcal { V } _ { N } )$ , we have that

$$
\begin{array} { r l } & { \mathbb { P } ^ { * } ( \pi \succ \hat { \pi } ) \leq \mathbb { \widehat { P } } ( \pi \succ \hat { \pi } ) + | \mathbb { \widehat { P } } ( \pi \succ \hat { \pi } ) - \mathbb { P } ^ { * } ( \pi \succ \hat { \pi } ) | } \\ & { \qquad \leq \mathbb { \widehat { P } } ( \pi \succ \hat { \pi } ) + { \mathcal { C } } _ { \mathrm { u n i } } ( x ) \varepsilon ( x ) } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } + 2 { \mathcal { C } } _ { \mathrm { u n i } } ( x ) \varepsilon ( x ) . } \end{array}
$$

Here the third line follows from Lemma 4, and the second line is the same computation as in Lemma 1 so we omit the details.

Let $\tilde { \pi } = \delta _ { y ^ { * } }$ be a pure best response to πˆ and π<sub>R</sub> the coupled rejection-sampling la $\mathbf { W } ,$ both as defined at the head of Appendix B. Conditioned on $\mathcal { D } _ { N }$

$$
\begin{array} { r l } & { \mathbb { P } ^ { * } ( \tilde { \pi } \succ \hat { \pi } | \mathcal { V } _ { N } ) \leq \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } | \mathcal { V } _ { N } ) + | \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } | \mathcal { V } _ { N } ) - \mathbb { P } ^ { * } ( \tilde { \pi } \succ \hat { \pi } | \mathcal { V } _ { N } ) | } \\ & { \qquad \leq \mathbb { P } ^ { * } ( \pi _ { R } \succ \hat { \pi } | \mathcal { V } _ { N } ) + 2 D _ { \mathrm { T V } } ( \tilde { \pi } , \pi _ { R } | \mathcal { V } _ { N } ) } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } + 2 \mathcal { C } _ { \mathrm { u n i } } ( x ) \varepsilon ( x ) + 2 D _ { \mathrm { T V } } ( \tilde { \pi } , \pi _ { R } | \mathcal { V } _ { N } ) , } \end{array}
$$

where the last inequality applies the first display of this proof to the policy $\pi _ { R } \mid \mathcal { V } _ { N } \in \Delta ( \mathcal { V } _ { N } )$ Taking expectation over $\mathcal { D } _ { N }$ and applying Lemma 2 with $\begin{array} { r } { \bar { M } = \frac { N - 1 ^ { - } } { \log ( 4 / \varepsilon ^ { 2 } ( x ) ) } } \end{array}$

$$
\begin{array} { r } { \mathbb { P } ^ { * } ( \tilde { \pi } \succ \hat { \pi } \mid \mathcal { Y } _ { N } ) \le \frac { 1 } { 2 } + 2 \mathcal { C } _ { \mathrm { u n i } } ( x ) \varepsilon ( x ) + \frac { 1 } { 2 } \mathcal { C } _ { \mathrm { u n i } } ( x ) \varepsilon ^ { 2 } ( x ) \le \frac { 1 } { 2 } + \frac { 5 } { 2 } \mathcal { C } _ { \mathrm { u n i } } ( x ) \varepsilon ( x ) , } \end{array}
$$

and hence $\mathrm { D u a l G a p } ( \hat { \pi } ) = 2 \mathbb { E } [ \mathbb { P } ^ { * } ( \tilde { \pi } \succ \hat { \pi } | \mathcal { V } _ { N } ) ] - 1 \leq 5 \varepsilon ( x ) \mathcal { C } _ { \mathrm { u n i } } ( x )$

## D Omitted Proofs from Section 5

Theorem 3. Given a prompt x and K responses $\left\{ y _ { 1 } , \dots , y _ { K } \right\}$ , let $\pi _ { r e f } ( y _ { i } | x ) = 1 / K$ for all $i \in [ K ]$ For any alignment algorithm A and any $\varepsilon _ { 0 } \in ( 0 , \sqrt { 2 } / ( 3 K ) ]$ ], there exist preference oracles P<sup>∗</sup> and Pb $\mathbb { P } ^ { * }$ with $\varepsilon ( x ) = \varepsilon _ { 0 }$ such that:

$$
\mathrm { D u a l G a p } ( \mathcal { A } ( \widehat { \mathbb { P } } ) ) \geq \frac { \varepsilon _ { 0 } \cdot \mathcal { C } _ { \mathrm { u n i } } ( x ) } { 2 \sqrt { 2 } } .
$$

Proof. Set $\delta _ { 0 } : = \varepsilon _ { 0 } K / ( 2 \sqrt { 2 } ) \le 1 / 6$ by the assumption on $\varepsilon _ { \mathrm { 0 } }$ . Define the imperfect oracle:

$$
\begin{array} { r } { \widehat { { \mathbb { P } } } ( y _ { 1 } \succ y _ { j } ) = \frac { 1 } { 2 } + \delta _ { 0 } \quad \mathrm { f o r ~ a l l ~ } K \geq j \geq 2 , \qquad \widehat { { \mathbb { P } } } ( y _ { j } \succ y _ { k } ) = \frac { 1 } { 2 } \quad \mathrm { f o r ~ a l l ~ } K \geq j , k \geq 2 . } \end{array}
$$

Response $y _ { 1 }$ is the unique dominant strategy by ${ \widehat { \mathbb { P } } } .$ Now write

$$
\hat { \pi } ( y _ { 1 } ) = p , \hat { \pi } ( y _ { 2 } ) = q ,
$$

for some $p , q \in [ 0 , 1 ]$

In the following, we construct two true preference oracles in two worlds A and B. In the world A, $\mathbb { P } _ { A } ^ { * }$ agrees with $\widehat { \mathbb { P } }$ except

$$
\begin{array} { r } { \mathbb { P } _ { A } ^ { * } ( y _ { 1 } \succ y _ { 2 } ) = \frac { 1 } { 2 } + 3 \delta _ { 0 } , \qquad \mathbb { P } _ { A } ^ { * } ( y _ { 2 } \succ y _ { 1 } ) = \frac { 1 } { 2 } - 3 \delta _ { 0 } . } \end{array}
$$

In the world $B , \mathbb { P } _ { B } ^ { * }$ agrees with $\widehat { \mathbb { P } }$ except

$$
\begin{array} { r } { \mathbb { P } _ { B } ^ { * } ( y _ { 1 } \succ y _ { 2 } ) = \frac { 1 } { 2 } - \delta _ { 0 } , \qquad \mathbb { P } _ { B } ^ { * } ( y _ { 2 } \succ y _ { 1 } ) = \frac { 1 } { 2 } + \delta _ { 0 } . } \end{array}
$$

Since $\delta _ { 0 } \leq 1 / 6$ , both $P _ { A } ^ { * }$ and $P _ { B } ^ { * }$ take values in [0, 1].

Now we compute the oracle quality $\varepsilon _ { A } ^ { 2 } ( x )$ and $\varepsilon _ { B } ^ { 2 } ( x )$

$$
\varepsilon _ { A } ^ { 2 } ( x ) = \varepsilon _ { B } ^ { 2 } ( x ) = \frac { 8 \delta _ { 0 } ^ { 2 } } { K ^ { 2 } } = \varepsilon _ { 0 } ^ { 2 } .
$$

In World $\mathbf { A } , y _ { 1 }$ remains strictly dominant, so we compute:

$$
\mathbb { P } _ { A } ^ { * } ( y _ { 1 } \succ \hat { \pi } ) = \frac { p } { 2 } + q ( \frac { 1 } { 2 } + 3 \delta _ { 0 } ) + ( 1 - p - q ) ( \frac { 1 } { 2 } + \delta _ { 0 } ) \geq \frac { 1 } { 2 } + ( 1 - p ) \delta _ { 0 } .
$$

Therefore, we lower bound the duality gap in world A:

$$
\begin{array} { r } { \mathrm { D u a l G a p } _ { A } ( \hat { \pi } ) \geq 2 \mathbb { P } _ { A } ^ { * } ( y _ { 1 } \succ \hat { \pi } ) - 1 \geq 2 ( 1 - p ) \delta _ { 0 } . } \end{array}
$$

In World B, $y _ { 2 }$ beats $y _ { 1 }$ by margin $\delta _ { 0 } .$ , while $y _ { 2 }$ ties with $y _ { j }$ for $j \geq 3 ,$ , thus $y _ { 2 }$ is dominant. So we compute

$$
\mathbb { P } _ { B } ^ { * } ( y _ { 2 } \succ \hat { \pi } ) = p ( \frac { 1 } { 2 } + \delta _ { 0 } ) + \frac { q } { 2 } + ( 1 - p - q ) \cdot \frac { 1 } { 2 } = \frac { 1 } { 2 } + p \delta _ { 0 } .
$$

Therefore, we lower bound the duality gap in world B:

$$
\begin{array} { r } { \mathrm { D u a l G a p } _ { B } ( \hat { \pi } ) \geq 2 \mathbb { P } _ { B } ^ { * } ( y _ { 2 } \succ \hat { \pi } ) - 1 \geq 2 p \delta _ { 0 } . } \end{array}
$$

The adversary selects the world that is worse for the algorithm:

$$
\begin{array} { r } { \mathrm { D u a l G a p } ( \hat { \pi } ) \geq \operatorname* { m a x } \mathopen { } \mathclose \bgroup \left( 2 ( 1 - p ) \delta _ { 0 } , \ 2 p \delta _ { 0 } \aftergroup \egroup \right) . } \end{array}
$$

The algorithm minimizes this by setting $p = 1 / 2 ,$ , yielding:

$$
\mathrm { D u a l G a p } ( \hat { \pi } ) \geq \delta _ { 0 } = \frac { \varepsilon _ { 0 } K } { 2 \sqrt { 2 } } = \frac { \varepsilon _ { 0 } \mathcal C _ { \mathrm { u n i } } } { 2 \sqrt { 2 } }
$$

since ${ \mathcal { C } } _ { \mathrm { u n i } } ( x ) = K$

## E Additional Empirical Results

## E.1 Further Experimental Details

All win/draw/lose judgments are produced by DeepSeek-V4-Flash (temperature = 0, at most 4 output tokens). The verbatim prompt given to the judge:

System: You are an impartial expert judge evaluating the quality   
of two AI assistant responses to the same user prompt. Judge which   
response better follows the user’s instructions and is more helpful,   
correct, coherent and appropriately detailed for the request. Do   
not let the length of a response, the order in which the responses   
are presented, or stylistic flourishes bias your decision. Output   
exactly one character: ’A’ if Response A is better, or ’B’ if   
Response B is better. Do not output anything else.   
User:   
[User Prompt]   
{prompt}   
[Response A]   
{a}   
[Response B]   
{b}   
Which response is better? Answer with a single letter: A or B.

## E.2 Additional Experiments

In this section, We compare our methods against three baselines: Borda Best-of-N, standard rewardbased Best-of-N, and a fine-tuned Nash-MD-PG model [Munos et al., 2024].

<table><tr><td>Dataset</td><td>Base SFT</td><td>Best-of-Nash</td><td>Borda Best-of-N</td></tr><tr><td>TLDR</td><td>62.9%</td><td>73.5%</td><td>72.5%</td></tr><tr><td>HelpSteer2</td><td>44.1%</td><td>68.8%</td><td>67.9%</td></tr><tr><td>UltraFeedback</td><td>23.9%</td><td>48.8%</td><td>45.7 %</td></tr></table>

Table 2: Comparison of expected win-rate (EWR) across three datasets. Best-of-Nash and Borda Best-of-N sample N = 64 responses from LLaMA3-SFT and we use LLaMA3-PM as preference model.

Algorithm 4 Borda Best-of-N   
1: Input: Prompt $x ,$ reference policy $\pi _ { \mathrm { r e f } } ,$ , preference oracle ${ \widehat { \mathbb { P } } } ,$ sample size N.   
2: Draw $\widehat { y } _ { N } = ( y _ { 1 } , \ldots , y _ { N } ) \sim \pi _ { \mathrm { r e f } } ( \cdot | x )$ i.i.d.   
3: Query ${ \widehat { P } } _ { i j } \gets { \widehat { \mathbb { P } } } ( y _ { i } \succ y _ { j } \mid x )$ for all $1 \leq i < j \leq N ,$ and set $\begin{array} { r } { \widehat { P } _ { j i } \gets 1 - \widehat { P } _ { i j } , \widehat { P } _ { i i } \gets \frac { 1 } { 2 } . } \end{array}$   
4: Compute the Borda score for each response:   
$\hat { r } ( y _ { i } ) \gets \frac { 1 } { N - 1 } \sum _ { j \neq i } \widehat { P } _ { i j } , \quad \forall i \in [ N ] .$   
5: Return $\widehat { y } \gets \arg \operatorname* { m a x } _ { y \in \widehat { y } _ { N } } \widehat { r } ( y )$

A New Borda Best-of-N Baseline. To control the oracle strength, we propose a Borda Best-of-N method under the same preference oracle, shown in Algorithm 4. We report the result in Table 2. We find that Borda Best-of-N attains win-rates close to Best-of-Nash and NMD across all three datasets. To understand this, we ran a diagnostic on the dataset: in 73.8% of N = 64 sub-samples, the Borda winner is a Condorcet winner of the empirical preference matrix—it beats every other candidate pairwise—in which case the Nash equilibrium of the sub-game is exactly the pure strategy on that response, so all three methods return the same response. The agreement is thus a structural property of the data rather than evidence that the equilibrium computation is redundant. On the remaining non-Condorcet sub-samples, argmax-style rules carry no guarantee.

Standard Best-of-N. We implement the standard Best-of-N method under a fine-tuned Bradley– Terry reward model. We train the reward model and a pairwise preference model under an identical protocol, differing only in architecture: both fine-tune the same backbone (Qwen3-4B-Instruct-2507 [QwenTeam, 2025]) with a six-criterion linear head on the same HelpSteer2 training pairs. we draw N = 128 candidates per prompt from the base policy (LLaMA3-SFT, temperature 1.0) and compare three selectors: base SFT, Best-of-N under the Bradley-Terry reward, and Best-of-Nash under the pairwise preference matrix. We report the expected win rate (EWR) against the prompt’s human-preferred response, following the protocol of Section 6. Table 3 reports the results: both methods improve over the base policy by 17–21 percent, confirming that a well-trained oracle of either form provides a strong selection signal. Best-of-Nash attains a 4.0% higher win rate than reward-based Best-of-N.

<table><tr><td>Selector</td><td>Expected win rate</td></tr><tr><td>Base policy (random candidate)</td><td>47.0%</td></tr><tr><td>Best-of-N (Bradley-Terry reward)</td><td>64.0%</td></tr><tr><td>Best-of-Nash (pairwise preference)</td><td>68.0%</td></tr></table>

Table 3: Comparison of Best-of-N and Best-of-Nash on held-out HelpSteer 2 prompts.

A Fine-tuned Nash-MD Baseline. Finally, we compare our methods against fine-tuning with general preferences: we fine-tune the base policy with Nash-MD-PG method [Munos et al., 2024] on HelpSteer2 and evaluate it against Best-of-Nash and NMD applied to the same base policy at inference time. We use Qwen3-0.6B as the base policy and LLaMA3-PM as the preference model. For the Nash-MD-PG method, we train the total of 1 epoch with learning rate of 2e-6, KL regularization coefficient of 0.01, a mixture coefficient of 0.5, and temperature 0.7. We report the result in Table 4. Fine-tuned Nash-MD-PG improves over the base policy (40.5% vs. 34.2% EWR), while Best-of-Nash and NMD—using the same preference model and no parameter updates—improve substantially further (50.5% and 52.0%). Notably, Nash-MD’s gain comes largely from converting losses into draws (a 59.0% draw rate vs. 42.3% for the base) while its outright win rate does not increase, whereas Best-of-Nash and NMD raise the win rate itself. At this model scale and training budget, inference-time equilibrium computation thus extracts more from the same preference model than fine-tuning on it. Thus, we read this as evidence that our methods are a strong training-free alternative.

<table><tr><td>Method</td><td>Win</td><td>Draw</td><td>Lose</td><td>EWR</td></tr><tr><td>Base policy</td><td>13.0%</td><td>42.3%</td><td>44.7%</td><td>34.2%</td></tr><tr><td>Nash-MD (fine-tuned)</td><td>14.0%</td><td>55.0%</td><td>31.0%</td><td>41.5%</td></tr><tr><td>Best-of-Nash (N = 64)</td><td>29.0%</td><td>43.0%</td><td>28.0%</td><td>50.5%</td></tr><tr><td>NMD  $( N = 6 4 , \beta = 1 )$ </td><td>33.0%</td><td>38.0%</td><td>29.0%</td><td>52.0%</td></tr></table>

Table 4: Comparison of Best-of-Nash, NMD, and a fine-tuned Nash-MD on HelpSteer2. Best-of-Nash and NMD sample N = 64 responses from Qwen3-0.6B and we use LLaMA3-PM as preference model.