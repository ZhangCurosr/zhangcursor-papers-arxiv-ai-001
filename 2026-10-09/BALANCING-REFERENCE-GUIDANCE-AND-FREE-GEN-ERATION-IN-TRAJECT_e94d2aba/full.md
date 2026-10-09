# BALANCING REFERENCE GUIDANCE AND FREE GEN-ERATION IN TRAJECTORY ROLLOUTS FOR REASONING RL

Hanyu Wang<sup>1,2∗</sup>, Nakul Agarwal<sup>2</sup>, Hossein Nourkhiz Mahjoub<sup>2</sup>, Ehsan Moradi Pari<sup>2</sup>, Makoto Fukushima<sup>3</sup>, Jinghui Chen<sup>1</sup>, Vaishnav Tadiparthi<sup>2</sup>

<sup>1</sup>The Pennsylvania State University <sup>2</sup>Honda Research Institute USA

<sup>3</sup>Honda Research Institute Japan

## ABSTRACT

A verified reference solution provides a correct trajectory for training a reasoning model. Alternatively, a prefix of the reference can guide the model in generating a trajectory of its own. How much reference guidance should we provide? We study this question through prefix continuation, where the model continues from a reference prefix and keeps the resulting trajectory if it passes verification, falling back to the reference otherwise. Since both procedures produce correct trajectories, we compare their distributions with the ideal distribution, the model’s own distribution conditioned on successful verification. For one continuation, we derive the KL divergence in closed form, which, up to a bounded term, decreases with the product of the probability of generating a different correct trajectory and the reference surprisal, the negative log probability of the reference suffix given the prefix. Since a longer prefix tends to raise the former but lowers the latter, continuation success alone does not determine the preferred amount of guidance. From this analysis, we learn a prefix selector shared across training questions from continuation outcomes, without estimating success probabilities or additional generation. The resulting Adaptive Reference Guidance (ARG) constructs correct trajectories within a fixed generation budget, and we apply it to all-failure groups in Group Relative Policy Optimization (GRPO). Experiments on Qwen3-4B and Qwen3- 8B across five mathematical reasoning benchmarks show that ARG achieves the highest aggregate pass@12 among the evaluated methods with competitive average sampled accuracy.

## 1 INTRODUCTION

Reinforcement learning with verifiable rewards (RLVR) (Lambert et al., 2024) trains Large Language Models (LLMs) using feedback on their own generated responses (Guo et al., 2025, Team et al., 2025, Jaech et al., 2024). On difficult questions, however, the model may rarely generate a response that reaches the correct answer, leaving little feedback to learn from. When such a question comes with a reference solution written by humans or a stronger model, a straightforward approach is to train on it directly (Yan et al., 2026, Fu et al., 2026, Ma et al., 2026). However, the refer ence may differ substantially from the trajectories the model would produce itself, which can make it a poor learning target (Setlur et al., 2026, Qu et al., 2026). Since the model learns more easily from familiar responses (Ren et al., 2024), another line of work uses the reference as guidance so that the model generates correct trajectories by itself. Some methods transform the reference into hints (Zhang et al., 2026c, Liao et al., 2026), which requires additional LLM generation and may introduce bias or errors. Other methods, in contrast, let the model generate a continuation from a reference prefix, which is an initial segment of the reference. This approach uses the reference without modification and controls how much guidance the model receives through how much of the reference is revealed in the prefix.

However, when generating a trajectory following a prefix, deciding how much to reveal involves a trade-off. Specifically, using a longer reference prefix, which provides more guidance, makes the continuation more likely to be correct. Yet the resulting trajectory contains more content the model would not generate itself, making learning harder. Using a shorter prefix keeps the trajectory closer to the model’s own generation, but increases the chance of failure. At the two extremes, revealing the entire reference reduces to training on the reference directly, while revealing nothing reduces to standard RLVR, which provides no learning signal on questions the model cannot yet solve. Balancing the two is therefore essential. The prefix should reveal just enough for the model to succeed, and leave the rest for the model to generate on its own.

To analyze this trade-off, we first ask which trajectories are ideal to learn from. Ideally, these trajectories should be correct and generated entirely by the model. We therefore define the distribution of the model’s own correct samples as the ideal distribution. We show that the Kullback-Leibler (KL) divergence from the distribution of generated trajectories used for training to this ideal distribution decreases as $d _ { t } S _ { t }$ increases, up to a small bounded term. Here $d _ { t }$ is the probability that the model follows the reference’s prefix to generate a successful trajectory, and $S _ { t }$ is the negative log probability of the remaining reference given the prefix. Intuitively, $d _ { t }$ reflects how much the prefix helps the model succeed, while $S _ { t }$ indicates how unfamiliar the reference suffix is. Moreover, $d _ { t } S _ { t }$ captures both sides of the trade-off. A longer prefix tends to increase $d _ { t } ,$ , since more guidance makes success more likely, but it decreases $S _ { t }$ , since less of the reference is left.

As a result, a good prefix should maximize $d _ { t } S _ { t }$ . However, $d _ { t }$ is unknown, and estimating it for every prefix of every question is impractical. We avoid this estimation by learning a prefix selector, a distribution over a few guidance levels shared across questions. Since the selector only needs to work well on average, the outcome of each continuation already provides the feedback it needs, without any additional generation.

These analyses lead to Adaptive Reference Guidance (ARG). Given a question and its reference, ARG samples guidance levels from the prefix selector, generates continuations from the resulting prefixes within a fixed budget, and falls back to the reference when no continuation succeeds. The same continuations also update the prefix selector, which therefore requires no additional generation. To examine how the accepted trajectories affect policy learning, we apply ARG to Group Relative Policy Optimization (GRPO) (Shao et al., 2024) for groups in which every sampled response fails, resulting in no learning signals.

Empirically, on Qwen3-4B and Qwen3-8B (Yang et al., 2025) across five mathematical reasoning benchmarks, ARG achieves the highest aggregate pass@12 of 67.33% and 70.00%, respectively. Moreover, its average sampled accuracy remains competitive, with a small gap to the strongest baseline. Our contributions are threefold.

• We analyze reference-prefix continuation through the KL divergence to the model’s own correct trajectories, revealing a trade-off between reference guidance and surprisal.

• We derive an unbiased update of the prefix selector from the same continuations, adapting the guidance level across questions and during training without additional generation.

• We develop ARG from this analysis and apply it to all-failure groups in GRPO, achieving the highest aggregate pass@12 across five math benchmarks on two model scales.

## 2 PRELIMINARIES

Reasoning with verifiable rewards. Let $\pi _ { \theta } ( y \mid x )$ be a language model that generates a trajectory y for a question x drawn from the training distribution $\mathcal { D }$ . The trajectory includes the reasoning, the final answer, and the termination token. A verifier assigns a binary reward $R _ { \mathrm { a c c } } ( x , y ) \in \{ 0 , 1 \}$ and we call $y$ correct when $R _ { \operatorname { a c c } } ( x , y ) = 1$ . Let $C _ { x }$ be the set of correct trajectories and $p _ { \theta } ( x ) =$ $\pi _ { \theta } ( C _ { x } \mid x )$ be the success probability. During training, $\theta _ { \mathrm { o l d } }$ denotes the parameters of the current model, which generates the samples at each training step. We use natural logarithms throughout.

Reference-prefix continuation with fallback. Suppose that the question comes with a correct reference $r \in C _ { x }$ . For a prefix length $0 \leq t < | r |$ , we write $r = h _ { t } \oplus a _ { t }$ , where the reference prefix $h _ { t }$ contains the first t tokens of $r ,$ the reference suffix $a _ { t }$ contains the remaining tokens, and ⊕ denotes token concatenation. The model samples a continuation $Z \sim \pi _ { \boldsymbol { \theta } } ( \cdot \cdot \mid x , h _ { t } )$ , which forms the candidate trajectory $\widetilde { Y } _ { t } = h _ { t } \oplus Z . \mathrm { A t } t = 0$ , the prefix is empty and the model generates under the original question alone. We keep the candidate trajectory if it is correct and fall back to the reference otherwise, which gives the accepted trajectory

$$
Y = \left\{ \begin{array} { l l } { \widetilde { Y } _ { t } , } & { \widetilde { Y } _ { t } \in C _ { x } , } \\ { r , } & { \widetilde { Y } _ { t } \notin C _ { x } , } \end{array} \right. \quad Y \sim Q _ { t } .\tag{1}
$$

Thus, $Q _ { t }$ assigns all probability to correct trajectories for every prefix length. In contrast, training on the reference corresponds to the distribution $\delta _ { r }$ , which assigns all probability to r.

Ideal distribution. As discussed in Section 1, we take the model’s own correct trajectories as the ideal distribution. Formally, for $p _ { \theta } ( x ) > 0$ , sampling from $\pi _ { \theta } ( \cdot \mid x )$ and keeping only correct trajectories gives

$$
\nu _ { \theta } ( y \mid x ) = { \frac { \pi _ { \theta } ( y \mid x ) \mathbf { 1 } \{ y \in C _ { x } \} } { p _ { \theta } ( x ) } } .\tag{2}
$$

This distribution also has a direct connection to likelihood learning (Phan et al., 2023, Singh et al., 2023). Specifically, for a finite trajectory space and a differentiable policy with positive probabilities,

$$
\operatorname { \mathbb { E } } _ { Y \sim \nu _ { \theta } ( \cdot \vert x ) } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( Y \mid x ) \right] = \nabla _ { \theta } \log p _ { \theta } ( x ) .\tag{3}
$$

In other words, learning the likelihood of trajectories drawn from $\nu _ { \theta }$ increases the success probability in expectation at the current model. Finally, we measure how far a distribution $Q$ of correct trajectories is from the ideal distribution by the Kullback-Leibler (KL) divergence

$$
D _ { \mathrm { K L } } ( Q \| \nu _ { \theta } ) = \mathbb { E } _ { Y \sim Q } \left[ \log { \frac { Q ( Y ) } { \nu _ { \theta } ( Y \mid x ) } } \right] .
$$

Appendix A.2 derives Equation (3) and explains this direction of KL.

## 3 ANALYZING AND SELECTING REFERENCE PREFIXES

In this section, we analyze how the prefix length affects the accepted trajectory and turn the analysis into a practical way of choosing prefixes. Section 3.1 derives the distance from the accepted trajectory to the ideal distribution, which identifies the preferred prefix of a question. Section 3.2 explains why this prefix is difficult to find during training, and Section 3.3 addresses both challenges by learning a prefix selector that performs well on average across questions.

## 3.1 DISTANCE TO THE IDEAL DISTRIBUTION

Fix a question $x$ with reference r and the current model $\pi _ { \theta _ { \mathrm { o l d } } } .$ , and suppress the conditioning on x and the parameters by writing $\pi , C , p = \pi ( C )$ , and $\nu .$ We assume a finite trajectory space with $0 < \pi ( r ) < 1$ . For a prefix length t, we define the successful replacement probability $d _ { t }$ and the reference surprisal $S _ { t }$ as

$$
d _ { t } = \operatorname* { P r } \bigl ( \widetilde { Y } _ { t } \in C , \ \widetilde { Y } _ { t } \neq r \bigr ) , \qquad S _ { t } = - \log \pi ( a _ { t } \mid h _ { t } ) .\tag{4}
$$

Here $d _ { t }$ excludes an exact copy of the reference, since the accepted trajectory is then still $r .$ In other words, $d _ { t }$ is the probability that the model replaces the reference suffix with a different correct continuation, and $S _ { t }$ measures how unfamiliar the reference suffix is to the model after the prefix.

Theorem 1 (Distance to the ideal distribution). Under the setup above, for every prefix length t,

$$
D _ { \mathrm { K L } } ( Q _ { t } \Vert \nu ) = \underbrace { \log \frac { p } { \pi ( r ) } } _ { D _ { \mathrm { K L } } ( \delta _ { r } \Vert \nu ) } - d _ { t } S _ { t } + ( 1 - d _ { t } ) \log ( 1 - d _ { t } ) .\tag{5}
$$

Moreover, every $t ^ { * } \in$ arg max<sub>t</sub> $d _ { t } S _ { t }$ satisfies $D _ { \mathrm { K L } } ( Q _ { t ^ { * } } \| \nu ) \leq \operatorname* { m i n } _ { t } D _ { \mathrm { K L } } ( Q _ { t } \| \nu ) + 1 / e .$

Theorem 1 expresses the distance between $Q _ { t }$ and the ideal distribution ν in closed form with three terms. The first term equals that of training on the reference, which does not depend on the prefix. The third term is bounded within $[ - 1 / e , 0 ]$ . Therefore, the prefix affects the distance mainly through the second term, $d _ { t } S _ { t }$ . Intuitively, this product formalizes the trade-off in choosing the prefix length. A longer prefix never increases $S _ { t }$ , since it leaves less of the reference for the continuation to replace. In contrast, a longer prefix can increase $d _ { t }$ by making the continuation easier to succeed. Consequently, both a nearly complete prefix and an empty prefix on a hard question make $d _ { t } S _ { t }$ close to zero and leave the distance close to $D _ { \mathrm { K L } } ( \delta _ { r } \Vert \nu )$ , indicating learning from the reference alone. Moreover, a prefix with a higher success probability of the continuation need not give a smaller distance, as illustrated in Appendix A.5. Instead, maximizing $d _ { t } S _ { t }$ attains the smallest distance up to $1 / e$ by the second statement in the theorem, and we call $h _ { t ^ { * } }$ the preferred prefix of the question under the current model. Appendix A gives the proof.

## 3.2 CHALLENGES IN FINDING THE PREFERRED PREFIX

Although Theorem 1 identifies the preferred prefix, finding it during training is difficult. The reference surprisal $S _ { t }$ is available for every prefix length from a single forward pass of the model on the reference. However, the successful replacement probability $d _ { t }$ is unknown, which leads to the following two challenges.

Challenge 1: Variation across questions and training steps. Theorem 1 concerns a fixed question and a fixed model, and $d _ { t }$ depends on both. Therefore, the estimates of $d _ { t }$ for one question do not apply to another question. Furthermore, since $d _ { t }$ changes as the model is updated, the estimates obtained at one training step may become outdated at the next. As a result, the search for $t ^ { * }$ would have to be repeated for every question and every model update.

Challenge 2: Limited sampling budget. Moreover, each such search is expensive. Identifying $t ^ { * }$ compares $d _ { t } S _ { t }$ across prefix lengths and therefore requires an accurate estimate of $d _ { t }$ at every prefix length. Since $d _ { t }$ is often small on hard questions, such an estimate requires many continuations even for a single prefix length, and this cost multiplies across prefix lengths.

## 3.3 LEARNING A PREFIX SELECTOR ACROSS QUESTIONS

Both challenges arise from seeking the preferred prefix of each question at each training step. Instead, we learn a way of choosing prefixes that performs well on average over the questions that receive guidance, which addresses both challenges.

Sharing across questions and training steps. Averaging over questions requires describing prefixes in the same way for every question, which the prefix length does not achieve for references of different lengths. Since Theorem 1 measures the replaced suffix by its surprisal rather than its length, we describe a prefix through the reference surprisal. Specifically, for a question x with reference $r _ { x }$ let $S _ { 0 , x } = - \log \pi _ { \theta _ { \mathrm { o l d } } } ( r _ { x } \bar { \mid x } )$ be the surprisal of the entire reference. A guidance level α in a set ${ \mathcal { A } } \subset { \mathrm { ( 0 , 1 ] } }$ selects a prefix length $t _ { x } ( \alpha )$ whose reference surprisal is approximately $\alpha S _ { 0 , x }$ . Thus, α is the fraction of the reference surprisal left to the continuation, where $\alpha = 1$ gives the empty prefix. As with prefix lengths, the levels cannot all be tried within the sampling budget. Therefore, A contains only a few levels, and Section 4.1 specifies the mapping from levels to prefix lengths.

We then learn a prefix selector $\rho _ { \phi }$ , which is a distribution over A with parameters $\phi$ shared across questions. Let $\mathcal { D } _ { \mathrm { g u i d e } }$ denote the distribution of questions that receive guidance at the current training step, which Section 4.3 specifies for our application. For $X \sim \mathcal { D } _ { \mathrm { g u i d e } }$ and $\alpha \sim \rho _ { \phi }$ , we write $d _ { X , \alpha } ,$ $S _ { X , \alpha } ,$ , and $Q _ { X , \alpha }$ for $d _ { t } , S _ { t }$ , and $Q _ { t }$ at the prefix length $t _ { X } ( \alpha )$ , and $\nu _ { X }$ for the ideal distribution of X. The selector maximizes

$$
J ( \phi ) = \mathbb { E } _ { X \sim \mathcal { D } _ { \mathrm { g u i d e } } , \alpha \sim \rho _ { \phi } } \left[ \frac { d _ { X , \alpha } S _ { X , \alpha } } { S _ { 0 , X } } \right] ,\tag{6}
$$

which averages $d _ { t } S _ { t }$ over the questions and the sampled levels in units of the surprisal of the entire reference. This normalization keeps questions with long or unfamiliar references from dominating the average, and it does not change which prefix maximizes $d _ { t } S _ { t }$ within any question. Since the selector is shared across questions and updated at every training step, continuations on all questions and at all steps inform the same choice, while the selected prefix still adapts to each question through its reference surprisal. This addresses Challenge 1, and Appendix B.3 bounds the sharing cost.

Learning without estimating $d _ { t } .$ . The average objective also avoids the accurate estimates of $d _ { t }$ required in Challenge 2. For a continuation from the prefix $h _ { t }$ , we define the selector feedback

$$
W _ { t } = \mathbf { 1 } \big \{ \widetilde { Y } _ { t } \in C , \widetilde { Y } _ { t } \neq r \big \} \frac { S _ { t } } { S _ { 0 } } ,\tag{7}
$$

and write $W _ { X , \alpha }$ for the feedback at the prefix length $t _ { X } ( \alpha )$ . In other words, a successful replacement receives the fraction of the reference surprisal left to the continuation, while a failure or an exact copy of the reference receives zero. Since the indicator has expectation $d _ { t }$ , the feedback satisfies

$$
\mathbb { E } [ W _ { t } ] = \frac { d _ { t } S _ { t } } { S _ { 0 } } .\tag{8}
$$

Thus, the feedback of a single continuation is an unbiased estimate of its term in J. A single estimate is noisy, and noisy estimates cannot reliably rank the prefix lengths of one question. However, since J is an average that is linear in $d _ { X , \alpha } ,$ , the errors of single continuations cancel when pooled over many questions. Moreover, since $\mathcal { A }$ contains only a few levels and the selector keeps a positive probability on every level, every level receives feedback from many questions, which addresses Challenge 2. The following theorem shows that J still measures the distance in Theorem 1 and that the selector can be learned from this feedback.

Theorem 2 (Learning the prefix selector from feedback). Fix the current model, the references, and $\begin{array} { r } { \mathcal { D } _ { \mathrm { g u i d e } } , } \end{array}$ , where every question satisfies the setup ofSection 3.1. Let ${ \mathcal { L } } ( \phi )$ and ${ \mathcal { L } } _ { \mathrm { r e f } }$ be the expectations of $\mathsf { \bar { D } } _ { \mathrm { K L } } ( Q _ { X , \alpha } \| \nu _ { X } ) / \bar { \langle S _ { 0 , X } }$ and $D _ { \mathrm { { K L } } } ( \delta _ { r _ { X } } \| \nu _ { X } ) \big / \bar { S } _ { 0 } .$ <sub>,X</sub> over $X \sim \mathcal { D } _ { \mathrm { g u i d e } }$ and $\alpha \sim \rho _ { \phi }$ . Then

$$
J ( \phi ) \leq \mathcal { L } _ { \mathrm { r e f } } - \mathcal { L } ( \phi ) \leq J ( \phi ) + \mathbb { E } _ { X } \left[ \frac { 1 } { e S _ { 0 , X } } \right] .\tag{9}
$$

Moreover, for a differentiable prefix selector with positive probabilities and a baseline b fixed before sampling,

$$
\begin{array} { r } { \mathbb { E } \left[ ( W _ { X , \alpha } - b ) \nabla _ { \phi } \log \rho _ { \phi } ( \alpha ) \right] = \nabla _ { \phi } J ( \phi ) . } \end{array}\tag{10}
$$

The first statement shows that, compared with training on the reference, the selector reduces the average normalized distance by at least J, where each distance is divided by the surprisal of the entire reference. The gap is at most $\mathbb { E } _ { X } [ 1 / ( e S _ { 0 , X } ) ]$ , which is small when no reference is highly likely under the model. Therefore, maximizing J pursues the goal of Theorem 1 across questions. The second statement shows that raising the probability of the levels whose feedback exceeds the baseline b moves the selector toward a larger J on average, where b introduces no bias and serves to reduce the variance. Since these continuations also produce the accepted trajectories, the update requires no additional generation. Appendix B proves Theorem 2 and extends it to batches, changing models, and randomly sampled levels.

## 4 ARG: ADAPTIVE REFERENCE GUIDANCE

We now build ARG on Theorems 1 and 2. Section 4.1 describes how ARG generates accepted trajectories for a question, and Section 4.2 describes how the same continuations update the prefix selector. Neither step depends on a particular reinforcement learning algorithm. The algorithm only determines the questions that receive guidance and how the model learns from the accepted trajectories, and Section 4.3 specifies both for GRPO.

## 4.1 GENERATING ACCEPTED TRAJECTORIES

Consider a question x that receives guidance, with reference r. ARG first scores r once under the current model $\pi _ { \theta _ { \mathrm { o l d } } }$ , which gives the reference surprisal $S _ { t }$ for every prefix length. Each guidance level $\alpha \in { \mathcal { A } }$ then maps to a prefix length $t _ { x } ( \alpha )$ . Specifically, $\alpha = 1$ gives the empty prefix. For $\alpha < 1$ , we take the latest prefix length whose reference surprisal is at least $\alpha S _ { 0 }$ and move it to a nearby sentence or formula boundary, so that the prefix ends at a natural break. Appendix E.2 gives the boundary rules.

The prefix selector assigns a logit $\phi _ { \alpha }$ to each level and samples from

$$
\rho _ { \phi } ( \alpha ) = ( 1 - \varepsilon _ { \mathrm { s e l } } ) \frac { \exp ( \phi _ { \alpha } ) } { \sum _ { \alpha ^ { \prime } \in \mathcal { A } } \exp ( \phi _ { \alpha ^ { \prime } } ) } + \frac { \varepsilon _ { \mathrm { s e l } } } { | \mathcal { A } | } ,\tag{11}
$$

where $0 < \varepsilon _ { \mathrm { s e l } } < 1$ keeps every level observed as the model changes. Within a fixed budget of K continuations per question, we independently sample $\alpha _ { j } \sim \rho _ { \phi } \mathrm { f o r } j = 1 , \ldots , K$ . For the resulting prefix length $t _ { j } = t _ { x } ( \alpha _ { j } )$ , we sample a continuation $Z _ { j } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x , h _ { t _ { j } } )$ and verify the candidate trajectory $\widetilde { Y } _ { j } = h _ { t _ { j } } \oplus Z _ { j }$

Every correct candidate trajectory different from r can serve as an accepted trajectory. If there is none, ARG falls back to the reference. Therefore, the question always receives a correct trajectory. When the number of accepted trajectories is limited, as in our application, we prefer candidate trajectories that keep less of the reference. Specifically, a correct candidate trajectory from the empty prefix keeps none of the reference. Indeed, it is generated entirely by the model and follows the ideal distribution restricted to trajectories other than r. Therefore, we accept these candidate trajectories first. Among candidate trajectories from nonempty prefixes, we prefer the shortest prefix, since the surprisal of the prefix, − log π<sub>θ</sub> (h | x) = S − S , never decreases with t. Finally, Theorem 1 concerns a single continuation. For accepted trajectories from several continuations, Appendix C.2 gives a sufficient condition under which the distance remains no larger than that of training on the reference.

## 4.2 UPDATING THE PREFIX SELECTOR

Every continuation provides the selector feedback $W _ { j } = W _ { t _ { j } }$ in Equation $( 7 ) .$ , whether or not its candidate trajectory is accepted. Since the feedback does not depend on which trajectories are accepted, the second statement of Theorem 2 holds under any acceptance rule, while the first statement, like Theorem 1, concerns a single continuation.

We collect continuations for all questions in a batch while keeping the current model and the prefix selector fixed. Let $\mathcal { U }$ index these continuations, with sampled levels $\alpha _ { j }$ and selector feedback $W _ { j }$ For a nonempty $u ,$ we update the prefix selector by

$$
\phi  \phi + \eta _ { \mathrm { s e l } } \sum _ { j \in \mathcal { U } } ( W _ { j } - b ) \nabla _ { \phi } \log \rho _ { \phi } ( \alpha _ { j } ) ,\tag{12}
$$

where $\eta _ { \mathrm { s e l } } > 0$ is the step size and b is a running average of past feedback fixed before collection. By the second statement of Theorem $^ { 2 , }$ the update follows the gradient of J in expectation. Appendix B gives the baseline update and derives the expected update as the model and the questions change during training.

## 4.3 APPLYING ARG TO GRPO

GRPO (Shao et al., 2024) samples a group of $G = 8$ responses for each question and compares their rewards within the group. When every response in a group is incorrect, the group provides no signal for improving correctness, since all correctness rewards are identical. In such an all-failure group, adding a correct trajectory restores the learning signal. Therefore, we apply ARG to all-failure groups, using $K = 8$ continuations. These questions define $\mathcal { D } _ { \mathrm { g u i d e } }$

We insert at most four accepted trajectories into each group, which keeps at least four original failures and permits a balanced group, motivated by prior work on intermediate pass rates (Bae et al., 2026, Liu et al., 2026b). Following the preference in Section 4.1, we first accept correct candidate trajectories from the empty prefix. If fewer than four are accepted, we additionally accept at most one from a nonempty prefix, which limits repeated reference content in the same group. If no candidate trajectory is accepted, we insert the reference once as fallback. Each inserted trajectory replaces an original failure at a uniformly chosen position. We then recompute the group advantages $A _ { i }$ from the final correctness and format rewards, as specified in Appendix E.3.

The model learns each inserted trajectory under the original question through its complete likelihood, including the prefix. This matches Theorem 1, which measures the complete accepted trajectory under the original question. Let $\mathcal { T }$ index the inserted trajectories $( x _ { i } , y _ { i } )$ and let $T$ be the total number of tokens in the loss. Their contribution to the gradient is

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { i n s e r t e d } } = - \frac { 1 } { T } \sum _ { i \in \mathcal { I } } A _ { i } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i } \mid x _ { i } ) ,\tag{13}
$$

where the trajectories, advantages, and token count are fixed during differentiation. The remaining original responses use the clipped GRPO loss with the same denominator. Moreover, for binary rewards, Appendix D shows that inserting $Y \sim Q$ into an all-failure group gives an expected update proportional to the gradient of log $\begin{array} { r } { \frac { p _ { \theta } } { 1 - p _ { \theta } } { \bf \bar { \theta } } - D _ { \mathrm { K L } } \bf \bar { ( } Q \| \nu _ { \theta } ) } \end{array}$ at the current model. The gradient of the second term vanishes when Q is the ideal distribution, in which case the update only increases the odds of success. This connects the distance in Theorem 1 to the GRPO update.

After collecting continuations, we update the prefix selector by Equation (12) and take one policy update on the final groups. Algorithm 1 summarizes the training procedure.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETTING

Data and models. We use DeepMath-103k with offline reference solutions (He et al., 2026) as our training set. We use Qwen3-4B and Qwen3-8B with thinking mode disabled (Yang et al., 2025) as our base LLMs. For all methods, we evaluate on five mathematical reasoning benchmarks, AIME 2024, AIME 2025, AIME 2026, HMMT February 2025, and HMMT November 2025 (Balunovic et al., 2026, Dekoninck et al., 2026). Appendix E gives further details.

Baselines and evaluation. We compare against the initial Qwen model, vanilla supervised finetuning (SFT), GRPO (Shao et al., 2024), OPSD (Zhao et al., 2026), SAGE (Liao et al., 2026), BREAD (Zhang et al., 2026d), and LUFFY (Yan et al., 2026). During evaluation, we draw 12 responses per question. pass@12 measures whether a question has at least one correct response, while avg@12 measures the fraction of correct responses. We use pass@12 as the primary metric.

<table><tr><td>Model</td><td>Method</td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>HMMT_Feb</td><td>HMMT_Nov</td><td>Avg.</td></tr><tr><td rowspan="8">Qwen3-4B</td><td>Base</td><td>50.00</td><td>43.33</td><td>43.33</td><td>23.33</td><td>33.33</td><td>38.67</td></tr><tr><td>SFT</td><td>63.33</td><td>63.33</td><td>66.67</td><td>36.67</td><td>36.67</td><td>53.33</td></tr><tr><td>GRPO</td><td>76.67</td><td>70.00</td><td>73.33</td><td>53.33</td><td>46.67</td><td>64.00</td></tr><tr><td>OPSD</td><td>56.67</td><td>36.67</td><td>40.00</td><td>40.00</td><td>40.00</td><td>42.67</td></tr><tr><td>SAGE</td><td>73.33</td><td>70.00</td><td>70.00</td><td>43.33</td><td>56.67</td><td>62.67</td></tr><tr><td>BREAD</td><td>80.00</td><td>73.33</td><td>73.33</td><td>50.00</td><td>56.67</td><td>66.67</td></tr><tr><td>LUFFY</td><td>80.00</td><td>73.33</td><td>70.00</td><td>43.33</td><td>56.67</td><td>64.67</td></tr><tr><td>ARG (ours)</td><td>80.00</td><td>73.33</td><td>73.33</td><td>43.33</td><td>66.67</td><td>67.33</td></tr><tr><td rowspan="8">Qwen3-8B</td><td>Base</td><td>46.67</td><td>43.33</td><td>53.33</td><td>16.67</td><td>46.67</td><td>41.33</td></tr><tr><td>SFT</td><td>80.00</td><td>70.00</td><td>73.33</td><td>46.67</td><td>56.67</td><td>65.33</td></tr><tr><td>GRPO</td><td>80.00</td><td>63.33</td><td>70.00</td><td>46.67</td><td>53.33</td><td>62.67</td></tr><tr><td>OPSD</td><td>66.67</td><td>60.00</td><td>50.00</td><td>36.67</td><td>43.33</td><td>51.33</td></tr><tr><td>SAGE</td><td>80.00</td><td>70.00</td><td>70.00</td><td>50.00</td><td>60.00</td><td>66.00</td></tr><tr><td>BREAD</td><td>80.00</td><td>66.67</td><td>66.67</td><td>53.33</td><td>70.00</td><td>67.33</td></tr><tr><td>LUFFY</td><td>76.67</td><td>76.67</td><td>73.33</td><td>46.67</td><td>63.33</td><td>67.33</td></tr><tr><td>ARG (ours)</td><td>83.33</td><td>73.33</td><td>73.33</td><td>56.67</td><td>63.33</td><td>70.00</td></tr></table>

Table 1. Main experimental results in pass@12 percentages. Avg. is the mean over the five bench marks. Within each model, bold and underline mark the best and second-best values in each column.

## 5.2 RESULTS

Table 1 evaluates ARG through its application to GRPO. ARG achieves the highest aggregate pass@12 at both model scales, reaching 67.33% on Qwen3-4B and 70.00% on Qwen3-8B. Compared with GRPO, these results improve pass@12 by 3.33 and 7.33 percentage points, respectively. ARG also exceeds the reference-informed baselines in aggregate pass@12, which use reference solutions as off-policy responses in LUFFY, as prefixes in BREAD, and as hints in SAGE (Yan et al., 2026, Zhang et al., 2026d, Liao et al., 2026). Thus, ARG solves the largest fraction of evaluation questions under the 12-response budget. For avg@12, ARG improves over GRPO on both models and remains competitive with the strongest baseline, as discussed in Appendix F. Sections 5.3 and 5.4 examine the contributions of the continuations and the prefix selector.

<table><tr><td colspan="2">Method pass@12 avg@12</td></tr><tr><td>Reference insertion</td><td>56.67 28.44</td></tr><tr><td>Resampling</td><td>60.00 34.00</td></tr><tr><td>ARG 67.33</td><td>41.11</td></tr></table>

(a) Sources of accepted trajectories on Qwen3-4B

<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen3-4B</td><td colspan="2">Qwen3-8B</td></tr><tr><td>pass@12 avg@12 pass@12 avg@12</td><td></td><td></td><td></td></tr><tr><td>w/o weighting</td><td>63.33</td><td>34.67</td><td>64.00</td><td>31.89</td></tr><tr><td>ARG</td><td>67.33</td><td>41.11</td><td>70.00</td><td>42.28</td></tr></table>

(b) Surprisal weighting on both models  
Table 2. Ablation results averaged over the five benchmarks. Resampling uses eight continuations from the empty prefix without reference fallback. The weighting ablation removes $S _ { t } / S _ { 0 }$ from the selector feedback.

## 5.3 ABLATIONS

We examine the contribution of the continuations, compare with unguided resampling, and evaluate the surprisal weighting in the selector feedback.

Reference insertion. For each all-failure group, we replace a random failure in the original group with the reference without any extra generation. This ablation corresponds to training on the reference in Theorem 1. Table 2(a) shows that reference insertion achieves 56.67% in pass@12 and 28.44% in avg@12 on Qwen3-4B, compared with 67.33% and 41.11% for ARG. These results support using the reference to guide continuations and reserving the reference itself for fallback.

Resampling. We allocate the same eight continuations to the empty prefix for each all-failure group. Correct candidate trajectories replace original failures, and groups with no correct candidate trajectory remain unchanged. Thus, this comparison evaluates ARG against additional sampling without reference guidance or fallback. Table 2(a) shows that ARG improves pass@12 from 60.00% to 67.33% and avg@12 from 34.00% to 41.11% on Qwen3-4B. During their respective training runs, candidate trajectories are accepted in 66.0% of all-failure groups with ARG and only 32.5% with resampling. More details are shown in Appendix G.2.

Without surprisal weighting. We replace the selector feedback in Equation (7) with the successful replacement indicator. The guidance levels, surprisal-based prefix construction, selector update rule, and acceptance rule remain unchanged. Table 2(b) shows that removing the weighting lowers both evaluation metrics on both models, with pass@12 decreasing by 4.00 and 6.00 percentage points on Qwen3-4B and Qwen3-8B, respectively. These results support the surprisal weighting in the selector feedback. Section 5.4 examines the selector feedback and prefix selection under ARG, while Appendix G.3 compares the prefix selection with that of the ablation.

## 5.4 TRAINING DYNAMICS

Figure 1 examines Qwen3-8B training dynamics through successful replacement rates, selector feedback, and prefix selection.

Successful replacement and selector feedback. Figure 1(a) reports the successful replacement rate for each guidance level. Each point averages all continuations at that level within a window of 50 training steps. The level $\alpha = 1 / 4$ has a higher successful replacement rate than $\alpha = 1 / 2$ in every window. However, Figure 1(b) shows that their ordering can change when the successful replacement indicator is weighted by the fraction $S _ { t } / S _ { 0 }$ of the reference surprisal left to the continuation. In the final window, for example, $\alpha = 1 / 4$ has lower mean feedback than $\alpha = 1 / 2 \ : ( 0 . 0 8 0 2$ versus 0.0991), although its successful replacement rate is higher (43.84% versus 23.65%). Thus, the feedback favors less reference guidance in this window, consistent with Section 3.1, where a higher success probability of the continuation need not give a smaller distance.

Prefix selection during training. The selector uses the feedback $W _ { t }$ from each continuation to update its selection probabilities. Figure 1(c) shows how these probabilities change during training. The selector favors $\alpha = 1 / 4$ through much of training and increases its probability of selecting $\alpha = 1 / 2$ later, assigning it the highest final probability of 49.0%. Since a larger α leaves more of the reference surprisal to the continuation, this shift supplies less reference guidance. Appendix H provides additional training-dynamics results.

![](images/95c1083298bcb86b063bbfcb663a9c8638e51dbf3575aceee0b26f419be0ee29.jpg)  
(a) Successful replacement

![](images/b033fbd87c152a4dbdfac44006621b4fdd8556a43429872a56cc140507cdb5d8.jpg)  
(b) Selector feedback

![](images/9eed3a261f89eaa1734b824fc5bf31f5a1c9710c35ff25f5d19254c5576f7247.jpg)  
(c) Prefix selection  
Figure 1. Training dynamics of a Qwen3-8B ARG run. Panel (a) reports successful replacement rates, panel (b) reports mean selector feedback $W _ { t } ,$ , and panel (c) shows the selection probabilities. In panels (a) and (b), a point at step s summarizes all continuations at that level across the past 50 steps, including failures and candidate trajectories not accepted for training. Panel (c) shows the state after each training step without smoothing.

## 6 RELATED WORK

Learning signals beyond on-policy rollouts. LUFFY, SRFT, ReLIFT, and related methods combine reference or teacher trajectories with the model’s own rollouts, through mixed-policy groups or combined supervised and reinforcement updates (Yan et al., 2026, Fu et al., 2026, Ma et al., 2026, Zhang et al., 2026b, Liu et al., 2025a, Dong et al., 2026, Lv et al., 2025). OPSD and SDPO instead use the model conditioned on solution information or feedback as a teacher for token-level supervision (Zhao et al., 2026, Hubotter et al., 2026). Other methods recover learning signals from failed¨ rollouts, either by repairing them into correct ones (Li et al., 2026b, Yang et al., 2026, Zhang et al., 2025, Ren et al., 2026) or by reshaping the rewards or advantages of groups with identical rewards (Chen et al., 2025, Nie et al., 2026, Nan et al., 2025, Le et al., 2026).

Reference-guided generation. Closer to our work, many methods use the reference only to guide the model’s own generation, either by revealing a prefix of the reference for the model to continue (Liu et al., 2026a, Setlur et al., 2026, Qu et al., 2026, Zhang et al., 2026a, Su et al., 2026, Liu et al., 2025b) or by providing hints derived from the reference or its answer (Liao et al., 2026, Zhang et al., 2026c, Nath et al., 2025, Chen et al., 2026, Zhou et al., 2026a). The amount of guidance is fixed or scheduled (Xi et al., 2024, Huang et al., 2025, Li et al., 2026a), adapted to the observed success of the model (Zhang et al., 2026d, amani et al., 2026, Li et al., 2025b, Guo et al., 2026, Beliaev, 2026), or chosen by the gain of a prefix or the transfer of a hint (Zhou et al., 2026c, Xia et al., 2026). In contrast, we derive the amount of guidance from the distance between the accepted trajectory and the ideal distribution, which shows that success alone does not determine the preferred prefix, and we learn a prefix selector shared across questions from the continuations themselves.

Posterior views of learning from correct trajectories. Training on the model’s own correct samples, as in STaR and ${ \tt R e S T } ^ { \mathrm { E M } }$ , approximates learning from our ideal distribution (Zelikman et al., 2022, Singh et al., 2023), which is also the correct distribution closest to the model in KL and relates to the Bayesian view of KL-regularized RL (Dymetman, 2026, Korbak et al., 2022). Latent-variable methods approximate the posterior over rationales given the correct answer (Phan et al., 2023, Zhou et al., 2026b), and TRICE keeps a stored rationale until a new proposal is correct, similar to our fallback. Other work analyzes or modifies the function of the success probability that RL maximizes (Davis & Recht, 2025, Tajwar et al., 2026, Ramasubramanian et al., 2026), or shows that targets unlikely under the model are harder to learn from (Ren et al., 2024, Li et al., 2025a, Setlur et al., 2024).

## 7 CONCLUSION AND LIMITATIONS

We study reference-prefix continuation with fallback through the distance from the accepted trajectory to the ideal distribution. Up to a small bounded term, this distance decreases with $d _ { t } S _ { t }$ , which captures the trade-off between successful replacement and the reference surprisal. This analysis leads to ARG, which learns a prefix selector across questions from the continuations themselves.

Applied to all-failure groups in GRPO, ARG achieves the highest aggregate pass@12 on Qwen3-4B and Qwen3-8B across five mathematical benchmarks, with competitive average sampled accuracy.

Limitations. ARG is only applied to GRPO, although the method does not depend on a particular reinforcement learning algorithm. Moreover, the selector chooses among a predefined set of guidance levels, and our experiments cover mathematical reasoning with one model family. Future work may extend the method to other domains with verifiable rewards.

## REFERENCES

mohammad hossein amani, Aryo Lotfi, Nicolas Baldwin, Samy Bengio, Mehrdad Farajtabar, Emmanuel Abbe, and Robert West. Rl for reasoning by adaptively revealing rationales. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 118599–118620, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ c073a9d002f3dab99a7af18642d74bdb-Paper-Conference.pdf.

Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak. Online difficulty filtering for reasoning oriented reinforcement learning. In Proceedings of the 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 700–719, 2026.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovic, and Martin Vechev. Matharena:´ Evaluating llms on uncontaminated math competitions. Advances in Neural Information Processing Systems, 38, 2026.

Vladislav Beliaev. Max out grpo signal: Adaptive trace prefix control for hard reasoning problems. arXiv preprint arXiv:2607.07674, 2026.

Justin Chen, Xiangyu Peng, Prafulla Kumar Choubey, Kung-Hsiang Huang, Jiaxin Zhang, Mohit Bansal, and Chien-Sheng Wu. Nudging the boundaries of llm reasoning. In International Conference on Learning Representations, volume 2026, pp. 30315–30335, 2026.

Peter Chen, Xiaopeng Li, Ziniu Li, Xi Chen, and Tianyi Lin. Stepwise guided policy optimization: Coloring your incorrect reasoning in grpo. arXiv preprint arXiv:2505.11595, 2025.

Damek Davis and Benjamin Recht. What is the objective of reasoning with reinforcement learning? arXiv preprint arXiv:2510.13651, 2025.

Jasper Dekoninck, Nikola Jovanovic, Tim Gehrunger, K´ ari R´ ognvaldsson, Ivo Petrov, Chenhao Sun,¨ and Martin Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. arXiv preprint arXiv:2605.00674, 2026.

Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, et al. Rl-plus: Countering capability boundary collapse of llms in reinforcement learning with hybrid-policy optimization. In Proceedings ofthe 64th Annual Meet ing ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 43029–43053, 2026.

Marc Dymetman. Binary rewards and reinforcement learning: Fundamental challenges. arXiv preprint arXiv:2605.02375, 2026.

Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. Srft: A single-stage method with supervised and reinforcement fine-tuning for reasoning. In International Conference on Learning Representations, volume 2026, pp. 102430–102459, 2026.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Yongxin Guo, Wenbo Deng, Zhenglin Cheng, and Xiaoying Tang. G2rpo-a: Guided group relative policy optimization with adaptive guidance. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 4525–4539, 2026.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. In International Conference on Learning Representations, volume 2026, pp. 138306–138322, 2026.

Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M Ponti, and Ivan Titov. Blending supervised and reinforcement fine-tuning with prefix sampling. arXiv preprint arXiv:2507.01679, 2025.

Jonas Hubotter, Frederike L¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta,¨ Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Tomasz Korbak, Ethan Perez, and Christopher Buckley. Rl with kl penalties is better viewed as bayesian inference. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 1083–1091, 2022.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Thanh-Long V Le, Myeongho Jeon, Kim Vu, Viet Lai, and Eunho Yang. No prompt left behind: Exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping. In International Conference on Learning Representations, volume 2026, pp. 121956– 121982, 2026.

Jiazheng Li, Hongzhou Lin, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Yi Wu, and Jingzhao Zhang. Questa: Expanding reasoning capacity in llms via question augmentation. In International Conference on Learning Representations, volume 2026, pp. 144191–144215, 2026a.

Shuozhe Li, Vaishnav Tadiparthi, Kwonjoon Lee, Nakul Agarwal, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Lizhang Chen, Amy Zhang, and Liu Leqi. Learning robust reasoning through guided adversarial self-play. arXiv preprint arXiv:2602.00173, 2026b.

Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, and Radha Poovendran. Small models struggle to learn from strong reasoners. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 25366–25394, 2025a.

Ziheng Li, Zexu Sun, Jinman Zhao, Erxue Min, Yongcheng Zeng, Hui Wu, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Xu Chen, et al. Staying in the sweet spot: Responsive reasoning evolution via capability-adaptive hint scaffolding. arXiv preprint arXiv:2509.06923, 2025b.

Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, and Jiang Bian. Self-hinting language models enhance reinforcement learning. arXiv preprint arXiv:2602.03143, 2026.

Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. Uft: Unifying supervised and reinforcement fine-tuning. Advances in Neural Information Processing Systems, 38:101347–101383, 2026a.

Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, and Dongmei Zhang. Superrl: Reinforcement learning with supervision to boost language model reasoning. arXiv preprint arXiv:2506.01096, 2025a.

Zehao Liu, Yuanpu Cao, Jinghui Chen, and Vasant G Honavar. Restoring the sweet spot: Pass-rate weighted self-distillation for llm reasoning. arXiv preprint arXiv:2605.27765, 2026b.

Ziru Liu, Cheng Gong, Xinyu Fu, Yaofang Liu, Ran Chen, Shoubo Hu, Suiyun Zhang, Rui Liu, Qingfu Zhang, and Dandan Tu. Ghpo: Adaptive guidance for stable and efficient llm reinforcement learning. arXiv preprint arXiv:2507.10628, 2025b.

Xingtai Lv, Yuxin Zuo, Youbang Sun, Hongyi Liu, Yuntian Wei, Zhekai Chen, Xuekai Zhu, Kaiyan Zhang, Bingning Wang, Ning Ding, et al. Towards a unified view of large language model posttraining. arXiv preprint arXiv:2509.04419, 2025.

Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Wong, Junbo Niu, Chengyu Shen, Runming He, Yanhao Li, et al. Learning what reinforcement learning can’t: Interleaved online fine-tuning for hardest questions. In International Conference on Learning Representations, volume 2026, pp. 80802–80821, 2026.

Gongrui Nan, Siye Chen, Jing Huang, Mengyu Lu, Dexun Wang, Chunmei Xie, Weiqi Xiong, Xianzhou Zeng, Qixuan Zhou, Yadong Li, et al. Ngrpo: Negative-enhanced group relative policy optimization. arXiv preprint arXiv:2509.18851, 2025.

Vaskar Nath, Elaine Lau, Anisha Gunjal, Manasi Sharma, Nikhil Baharte, and Sean Hendryx. Adaptive guidance accelerates reinforcement learning of reasoning models. arXiv preprint arXiv:2506.13923, 2025.

Wenhua Nie, Jianan Wu, Junlin Liu, Ziwei Li, Zheng Lin, Zhang Zijian, Yilong Fan, Haoran Zheng, and Jyh-Shing Roger Jang. Gradient starvation in binary-reward grpo: Why group-mean centering fails and why the simplest fix works. arXiv preprint arXiv:2605.07689, 2026.

Du Phan, Matthew Douglas Hoffman, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, and Rif A Saurous. Training chain-of-thought via latent-variable inference. Advances in Neural Information Processing Systems, 36:72819–72841, 2023.

Yuxiao Qu, Amrith Setlur, Virginia Smith, Ruslan Salakhutdinov, and Aviral Kumar. Pope: Learning to reason on hard problems via privileged on-policy exploration. arXiv preprint arXiv:2601.18779, 2026.

Shrinivas Ramasubramanian, Daman Arora, Fahim Tajwar, Guanning Zeng, Qingyang Wu, Zhongzhu Zhou, Chenfeng Xu, Haiwen Feng, Yuda Song, Aarti Singh, et al. Tail-likelihood reinforcement learning. arXiv preprint arXiv:2609.02987, 2026.

Xuan Ren, Biao Wu, and Lingqiao Liu. I learn better if you speak my language: Understanding the superior performance of fine-tuning large language models with llm-generated responses. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 10225–10245, 2024.

Yanwei Ren, Haotian Zhang, Likang Xiao, Xikai Zhang, Jiaxing Huang, Jiayan Qiu, Baosheng Yu, Quan Chen, and Liu Liu. Recycling failures: Salvaging exploration in rlvr via fine-grained off-policy guidance. arXiv preprint arXiv:2602.24110, 2026.

Amrith Setlur, Saurabh Garg, Xinyang Geng, Naman Garg, Virginia Smith, and Aviral Kumar. Rl on incorrect synthetic data scales the efficiency of llm math reasoning by eight-fold. Advances in Neural Information Processing Systems, 37:43000–43031, 2024.

Amrith Setlur, Zijian Wang, Andrew Cohen, Paria Rashidinejad, and Sang Michael Xie. Reuse your flops: Scaling rl on hard problems by conditioning on very off-policy prefixes. arXiv preprint arXiv:2601.18795, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, et al. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

Mingyu Su, Jian Guan, Yuxian Gu, Minlie Huang, and Hongning Wang. Trust-region adaptive policy optimization. In International Conference on Learning Representations, volume 2026, pp. 51806–51828, 2026.

Fahim Tajwar, Guanning Zeng, Yueer Zhou, Yuda Song, Daman Arora, Yiding Jiang, Jeff Schneider, Ruslan Salakhutdinov, Haiwen Feng, and Andrea Zanette. Maximum likelihood reinforcement learning. arXiv preprint arXiv:2602.02710, 2026.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, et al. Training large language models for reasoning through reverse curriculum reinforcement learning. arXiv preprint arXiv:2402.05808, 2024.

Yu Xia, Canwen Xu, Zhewei Yao, Julian McAuley, and Yuxiong He. Learning to hint for reinforcement learning. arXiv preprint arXiv:2604.00698, 2026.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. Advances in Neural Information Processing Systems, 38:117157–117186, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Matthew Yang, Hao Bai, Ian Wu, Gene Yang, Amrith Setlur, and Aviral Kumar. Int: Self-proposed interventions enable credit assignment in llm reasoning. In International Conference on Learning Representations, volume 2026, pp. 85054–85091, 2026.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, and Rui Yan. Stephint: Multi-level stepwise hints enhance reinforcement learning to reason. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 37846–37864, 2026a.

Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. On-policy rl meets off-policy experts: Harmonizing supervised fine-tuning and reinforcement learning via dynamic weighting. In International Conference on Learning Representations, volume 2026, pp. 120693–120726, 2026b.

Xiaoying Zhang, Yipeng Zhang, Hao Sun, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen Meng. Critique-grpo: Advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106, 2025.

Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, and Jiaya Jia. Scafgrpo: Scaffolded group relative policy optimization for enhancing llm reasoning. In International Conference on Learning Representations, volume 2026, pp. 131946–131974, 2026c.

Xuechen Zhang, Zijian Huang, Yingcong Li, Chenshun Ni, Jiasi Chen, and Samet Oymak. Bread: Branched rollouts from expert anchors bridge sft & rl for reasoning. Advances in Neural Information Processing Systems, 38:96726–96752, 2026d.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

Ruiyang Zhou, Shuozhe Li, Amy Zhang, and Liu Leqi. Expo: Unlocking hard reasoning with selfexplanation-guided reinforcement learning. Advances in Neural Information Processing Systems, 38:90989–91016, 2026a.

Xiangxin Zhou, Zichen Liu, Haonan Wang, Chao Du, Min Lin, Chongxuan Li, Liang Wang, and Tianyu Pang. Variational reasoning for language models. In International Conference on Learning Representations, volume 2026, pp. 154370–154407, 2026b.

Yuhang Zhou, Yixin Cao, and Guangnan Ye. From correctness to utility: Gain-based prefix evaluation for llm reasoning. arXiv preprint arXiv:2606.07190, 2026c.

## A PROOFS FOR THE DISTANCE TO THE IDEAL DISTRIBUTION

This appendix proves Theorem 1 and collects its consequences. Appendix A.1 fixes the setup. Appendix A.2 derives the gradient identity in Equation (3) and the distance of training on the reference. Appendix A.3 describes the accepted trajectory and the reference surprisal, and then proves Theorem 1. Appendix A.4 gives an equivalent reading of the closed form. Finally, Appendix A.5 illustrates the theorem with a finite example.

## A.1 SETUP

We use the setup of Section 3.1. Fix a question $x ,$ its correct reference r, and the current model, and suppress the conditioning on x. The trajectory space is finite, and π denotes the current policy $\pi _ { \theta _ { \mathrm { o l d } } } .$ Moreover, $C$ is the set of correct trajectories, $p = \pi ( C )$ is the success probability, and $\nu = \pi ( \cdot | \stackrel { \triangledown } { C } )$ is the ideal distribution in Equation (2). We assume $0 < \pi ( r ) < 1$ . For a token sequence $h ,$ we write $\pi ( h )$ for the probability that a trajectory sampled from $\pi$ begins with $h ,$ and $\pi ( z \mid h )$ for the probability that the model continues h with z. Since the continuation follows the same policy, including its termination rule, autoregressive factorization gives $\pi ( h \oplus z ) = \pi ( h ) \pi ( z \mid h )$ for every complete trajectory $h \oplus z .$

For a prefix length $0 \leq t < | r | .$ , the reference splits as $r = h _ { t } \oplus a _ { t }$ , and $Q _ { t }$ is the distribution of the accepted trajectory in Equation (1). Besides $d _ { t }$ and $S _ { t }$ in Equation (4), the proofs use the probabilities of an exact copy of the reference and of a correct candidate trajectory,

$$
u _ { t } = \pi ( a _ { t } \mid h _ { t } ) , \qquad v _ { t } = \operatorname* { P r } _ { Z \sim \pi ( \cdot \mid h _ { t } ) } \big ( h _ { t } \oplus Z \in C \big ) .
$$

Here an exact copy means equality of the complete token sequences, and different token sequences can still follow the same mathematical strategy. By definition, $S _ { t } = - \log u _ { t } .$ , and in particular $S _ { 0 } = - \log \pi ( r )$ is the reference surprisal of the empty prefix. Since autoregressive factorization gives $u _ { t } = \pi ( r ) / \pi ( h _ { t } ) \geq \pi ( r ) > 0 ,$ , we have $d _ { t } \leq 1 - u _ { t } < 1$ , and every $S _ { t }$ is finite.

## A.2 THE IDEAL DISTRIBUTION

We first justify the choice of the ideal distribution as the learning target in Section 2. Lemma A.1 shows that learning the likelihood of its trajectories increases the success probability in expectation. Lemma A.2 then gives the distance of training on the reference, which is the first term in Theorem 1.

Lemma A.1 (Gradient identity). Suppose that π<sub>θ</sub> is differentiable in θ and assigns positive probability to every trajectory in a neighborhood of $\mathrm { \Delta \cdot \theta _ { o l d } }$ . Write $p _ { \theta } = \pi _ { \theta } ( C )$ . Then

$$
\mathbb { E } _ { Y \sim \nu } \Big [ \nabla _ { \theta } \log \pi _ { \theta } ( Y ) \big | _ { \theta = \theta _ { \mathrm { o l d } } } \Big ] = \nabla _ { \theta } \log p _ { \theta } \big | _ { \theta = \theta _ { \mathrm { o l d } } } .
$$

Proof. Since the trajectory space is finite, differentiating $\begin{array} { r } { p _ { \theta } = \sum _ { y \in C } \pi _ { \theta } ( y ) } \end{array}$ term by term gives

$$
\nabla _ { \theta } \log { p _ { \theta } } = \frac { 1 } { p _ { \theta } } \sum _ { y \in C } \nabla _ { \theta } \pi _ { \theta } ( y ) = \sum _ { y \in C } \frac { \pi _ { \theta } ( y ) } { p _ { \theta } } \nabla _ { \theta } \log { \pi _ { \theta } ( y ) } .
$$

At $\theta = \theta _ { \mathrm { o l d } }$ , the weight $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ) / p _ { \boldsymbol { \theta } }$ of each $y \in C$ equals $\nu ( y )$ by Equation (2), which proves the claim. □

Here ν is fixed at the current model, and differentiation applies only to the trajectory log likelihood. Lemma A.2 (Distance of training on the reference). Training on the reference has distance

$$
D _ { \mathrm { K L } } ( \delta _ { r } \| \nu ) = \log \frac { p } { \pi ( r ) } = S _ { 0 } + \log p \in [ 0 , S _ { 0 } ] .
$$

In contrast, if ν assigns positive probability to a trajectory other than r, then $D _ { \mathrm { K L } } ( \nu \Vert \delta _ { r } ) = \infty$

Proof. Since $\delta _ { r }$ assigns all probability to r and $r \in C .$ , Equation (2) gives $D _ { \mathrm { K L } } ( \delta _ { r } \Vert \nu ) ~ =$ $- \log \nu ( r ) = \log ( p / \pi ( r ) )$ . Substituting $S _ { 0 } = - \log \pi ( r )$ gives $S _ { 0 } + \log p$ . This quantity is nonnegative since $\nu ( r ) \leq 1$ , and it is at most $S _ { 0 }$ since $p \leq 1$ . For the reverse direction, a trajectory $y \neq r$ with $\nu ( y ) > 0$ has $\delta _ { r } ( y ) = 0$ , which makes the term $\nu ( y ) \log ( \nu ( y ) / \delta _ { r } ( y ) )$ infinite. □

Thus, the direction of KL in Section 2 permits a finite comparison with training on the reference.

## A.3 PROOF OF THEOREM 1

The closed form in Theorem 1 combines two ingredients. Lemma A.3 describes the distribution of the accepted trajectory, and Lemma A.4 relates the reference surprisal to the probability of the prefix.

Lemma A.3 (Distribution of the accepted trajectory). For every prefix length t, we have $d _ { t } = v _ { t } - u _ { t }$ and $Q _ { t } ( r ) = 1 - d _ { t }$ . Moreover, every trajectory $y \neq r$ with $Q _ { t } ( y ) > 0$ is correct and has the form $y = h _ { t } \oplus z ,$ , and it satisfies

$$
Q _ { t } ( y ) = \pi ( z \mid h _ { t } ) , \qquad \frac { Q _ { t } ( y ) } { \nu ( y ) } = \frac { p } { \pi ( h _ { t } ) } .
$$

These trajectories have total probability $d _ { t }$ under $Q _ { t }$

Proof. The candidate trajectory $\widetilde { Y } _ { t } = h _ { t } \oplus Z$ equals r exactly when $Z = a _ { t }$ , which has probability $u _ { t }$ . Since $r \in C$ , the event $\{ \widetilde { Y } _ { t } \in C \}$ is the disjoint union of $\{ \widetilde { Y } _ { t } = r \}$ and $\{ \widetilde { Y } _ { t } \in C , \ \widetilde { Y } _ { t } \neq r \}$ Therefore, $v _ { t } = u _ { t } + d _ { t }$ . By Equation (1), the accepted trajectory equals r when the candidate trajectory is an exact copy or is incorrect. These two events are disjoint and have total probability $u _ { t } + ( 1 - v _ { t } ) = 1 - d _ { t }$

For $y \ne r ,$ , the accepted trajectory equals y only when $\widetilde { Y } _ { t } ~ = ~ y$ and $y \in C .$ . Hence, $Q _ { t } ( y ) > 0$ requires $y \in C$ and $y = h _ { t } \oplus z$ for some z, in which case $Q _ { t } ( y ) ~ { = } ~ \pi ( z ~ | ~ h _ { t } )$ Autoregressive factorization gives $\pi ( z \mid h _ { t } ) = \pi ( y ) / \pi ( h _ { t } )$ , where $\pi ( h _ { t } ) \geq \pi ( r ) > 0$ . Since $\nu ( y ) = \pi ( y ) / p$ for $y \in C$ , the ratio $Q _ { t } ( y ) / \nu ( y )$ equals $p / \pi ( h _ { t } )$ . Finally, these trajectories carry the remaining probability $1 - Q _ { t } ( r ) = d _ { t }$ □

Lemma A.4 (Monotonicity of the reference surprisal). For every prefix length $t ,$

$$
S _ { 0 } - S _ { t } = - \log \pi ( h _ { t } ) , \qquad 0 \leq S _ { t } \leq S _ { 0 } .
$$

Moreover, $S _ { t }$ is nonincreasing in t.

Proof. Autoregressive factorization gives $\pi ( r ) = \pi ( h _ { t } ) \pi ( a _ { t } \mid h _ { t } )$ . Taking negative logarithms gives $S _ { 0 } \stackrel { \cdot } { = } - \log \pi ( \bar { h } _ { t } ) + S _ { t }$ . Since $\tau ( a _ { t } \mid h _ { t } ) \leq 1$ and π $\cdot ( h _ { t } ) \leq 1$ , both $S _ { t }$ and $S _ { 0 } - S _ { t }$ are nonnegative. For $s \leq t ,$ , every trajectory that begins with $h _ { t }$ also begins with $h _ { s }$ . Therefore, $\pi ( h _ { t } ) \leq \bar { \pi } ( h _ { s } ) .$ which gives $S _ { t } = S _ { 0 } + \log \pi ( h _ { t } ) \leq S _ { s }$ . □

With these two lemmas, the proof of Theorem 1 reduces to a direct computation.

Theorem 1 (Distance to the ideal distribution, restated). Under the setup in Appendix A.1, for every prefix length t,

$$
D _ { \mathrm { K L } } ( Q _ { t } \Vert \nu ) = \underbrace { \log \frac { p } { \pi ( r ) } } _ { D _ { \mathrm { K L } } ( \delta _ { r } \Vert \nu ) } - d _ { t } S _ { t } + ( 1 - d _ { t } ) \log ( 1 - d _ { t } ) .
$$

Moreover, every $t ^ { * } \in$ arg max<sub>t</sub> $d _ { t } S _ { t }$ satisfies $D _ { \mathrm { K L } } ( Q _ { t ^ { * } } \| \nu ) \leq \operatorname* { m i n } _ { t } D _ { \mathrm { K L } } ( Q _ { t } \| \nu ) + 1 / e .$

Proof. Write $d = d _ { t } . \ \mathbf { B } \mathbf { y }$ Lemma ${ \bf A } . 3 , Q _ { t }$ assigns probability $1 - d$ to r, whose likelihood ratio is $Q _ { t } ( r ) / \nu ( r ) = ( 1 - d ) p / \pi ( r )$ . It assigns the remaining probability d to trajectories whose likelihood ratio is $p / \pi ( h _ { t } )$ . Therefore,

$$
D _ { \mathrm { K L } } ( Q _ { t } \Vert \nu ) = ( 1 - d ) \log \frac { ( 1 - d ) p } { \pi ( r ) } + d \log \frac { p } { \pi ( h _ { t } ) } .
$$

Lemma A.4 gives log $\pi ( h _ { t } ) = \log \pi ( r ) + S _ { t }$ , and hence $d \log ( p / \pi ( h _ { t } ) ) = d \log ( p / \pi ( r ) ) - d S _ { t }$ Collecting the two terms in $\log ( p / \pi ( r ) )$ proves the closed form. Its first term equals $\hat { D } _ { \mathrm { K L } } \big ( \delta _ { r } \| \nu \big )$ by Lemma A.2 and does not depend on t.

For the second statement, write $D _ { r } = D _ { \mathrm { K L } } ( \delta _ { r } \| \nu )$ and $c _ { t } = ( 1 - d _ { t } ) \log ( 1 - d _ { t } )$ . Since −a log $a \in$ $[ 0 , 1 / e ]$ for $a \in [ 0 , 1 ]$ , we have $c _ { t } \in [ - 1 / e , 0 ]$ . For every prefix length t, the maximality of $d _ { t ^ { * } } S _ { t ^ { * } }$

gives

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( Q _ { t ^ { * } } \Vert \nu ) = D _ { r } - d _ { t ^ { * } } S _ { t ^ { * } } + c _ { t ^ { * } } \leq D _ { r } - d _ { t ^ { * } } S _ { t ^ { * } } \leq D _ { r } - d _ { t } S _ { t } } \\ & { \qquad = D _ { \mathrm { K L } } ( Q _ { t } \Vert \nu ) - c _ { t } \leq D _ { \mathrm { K L } } ( Q _ { t } \Vert \nu ) + \displaystyle \frac { 1 } { e } . } \end{array}
$$

Taking the minimum over the finitely many prefix lengths completes the proof.

Since $d _ { t } S _ { t } \geq 0$ and $( 1 - d _ { t } ) \log ( 1 - d _ { t } ) \leq 0$ , the closed form also gives $D _ { \mathrm { K L } } ( Q _ { t } \| \nu ) \leq D _ { \mathrm { K L } } ( \delta _ { r } \| \nu )$ for every prefix length t. Therefore, a continuation with fallback never moves the distribution of the accepted trajectory farther from the ideal distribution than training on the reference.

## A.4 CONSEQUENCES OF THEOREM 1

We now give an equivalent reading of the closed form. Corollary A.5 rewrites the distance in terms of the reference content kept in the accepted trajectory.

Corollary A.5 (Reference content kept in the accepted trajectory). Let $\bar { h }$ be the reference content kept in the accepted trajectory $Y \sim \bar { Q } _ { t }$ , which is $r$ when $\bar { Y } = r$ and the prefix $h _ { t }$ otherwise. Then, for every prefix length t,

$$
D _ { \mathrm { K L } } ( Q _ { t } \| \nu ) = \log p + \underbrace { ( 1 - d _ { t } ) S _ { 0 } + d _ { t } ( S _ { 0 } - S _ { t } ) } _ { \mathbb { E } [ - \log \pi ( \bar { h } ) ] } + ( 1 - d _ { t } ) \log ( 1 - d _ { t } ) .
$$

Proof. Lemma A.2 gives $\log ( p / \pi ( r ) ) = \log p + S _ { 0 }$ . Substituting this expression and $S _ { 0 } = ( 1 -$ $d _ { t } ) \dot { S _ { 0 } } + d _ { t } S _ { 0 }$ into the closed form of Theorem 1 gives the display. By Lemma $\mathbf { A } . 3 , \bar { h } = r$ with probability $1 - d _ { t }$ and $\bar { h } = h _ { t }$ with probability $d _ { t }$ . Since ${ } _ { - } \log \pi ( r ) = S _ { 0 }$ and $- \log \pi ( h _ { t } ) = S _ { 0 } - S _ { t }$ by Lemma ${ \bf A . 4 } .$ , the middle two terms equal $\dot { \mathbb { E } } [ - \log \pi ( \bar { h } ) ]$ □

In other words, apart from log p, which does not depend on the prefix, and a term in $[ - 1 / e , 0 ]$ , the distance to the ideal distribution is the expected surprisal of the reference content that the model has not replaced. Both extremes of the prefix length keep nearly all of the reference surprisal $S _ { 0 }$ . A prefix covering almost the entire reference keeps it through the prefix, while an empty prefix on a hard question keeps it through fallback.

## A.5 A FINITE EXAMPLE

This example shows that neither the success probability of the continuation nor the successful replacement probability alone determines which prefix gives the smaller distance. Consider five complete trajectories whose displayed strings omit a shared deterministic termination token, H⊕r, H⊕s, $\mathbf { H } \oplus \mathbf { f } , 0 \oplus \mathbf { t }$ , and ${ \mathsf { D } } \oplus { \mathsf { g } } .$ , with respective probabilities 0.08, 0.01, 0.01, 0.20, and 0.70. The correct trajectories are H $\oplus \mathbf { r } , \mathtt { H } \oplus \mathbf { s }$ , and ${ 0 } \oplus \mathtt { t }$ , and the reference is $\mathtt { H } \oplus \mathtt { r }$ . The first tokens are H and O, with probabilities 0.10 and 0.90, and the second-token probabilities are the corresponding conditional probabilities. These probabilities define a finite autoregressive distribution.

<table><tr><td>Prefix</td><td> $v _ { t }$ </td><td> $u _ { t }$ </td><td> $d _ { t }$ </td><td> $S _ { t }$ </td><td> $d _ { t } S _ { t }$ </td><td> $D _ { \mathrm { K L } } ( Q _ { t } \Vert \nu )$ </td></tr><tr><td>Empty</td><td>0.29</td><td>0.08</td><td>0.21</td><td>2.525729</td><td>0.530403</td><td>0.571231</td></tr><tr><td>H</td><td>0.90</td><td>0.80</td><td>0.10</td><td>0.223144</td><td>0.022314</td><td>1.170715</td></tr></table>

Training on the reference gives $D _ { \mathrm { { K L } } } ( \delta _ { r } \Vert \nu ) = 1 . 2 8 7 8 5 4$ . Supplying H increases $v _ { t }$ while making an exact copy more likely. In this setting, both the successful replacement probability $d _ { t }$ and the reference surprisal $S _ { t }$ decrease. Therefore, the empty prefix gives the larger $d _ { t } S _ { t }$ and the smaller distance despite the lower success probability of its continuation.

To isolate the role of the reference surprisal, keep the three probabilities under H unchanged and assign probabilities 0.04 and 0.86 to ${ \sf 0 } \oplus { \sf t }$ and ${ \mathsf { 0 } } \oplus { \mathsf { g } } ,$ respectively. The resulting comparison is

<table><tr><td>Prefix</td><td> $v _ { t }$ </td><td> $u _ { t }$ </td><td> $d _ { t }$ </td><td> $S _ { t }$ </td><td> $d _ { t } S _ { t }$ </td><td> $D _ { \mathrm { K L } } ( Q _ { t } \Vert \nu )$ </td></tr><tr><td>Empty</td><td>0.13</td><td>0.08</td><td>0.05</td><td>2.525729</td><td>0.126286</td><td>0.310493</td></tr><tr><td>H</td><td>0.90</td><td>0.80</td><td>0.10</td><td>0.223144</td><td>0.022314</td><td>0.368369</td></tr></table>

Training on the reference now gives $D _ { \mathrm { K L } } ( \delta _ { r } \Vert \nu ) = 0$ .485508. Here supplying H increases both $v _ { t }$ and $d _ { t }$ . However, the decrease in the reference surprisal outweighs the increase in $d _ { t }$ . Therefore, the empty prefix still gives the larger $d _ { t } S _ { t }$ and the smaller distance. Thus, even the successful replacement probability alone can favor a different prefix from the one preferred by Theorem 1. Both examples concern a fixed model and illustrate why the choice of prefix should account for $d _ { t }$ and $S _ { t }$ together.

## B PROOFS FOR LEARNING THE PREFIX SELECTOR

This appendix proves Theorem 2 and analyzes the prefix selector during training. Appendix B.1 fixes the setup and derives the properties of the selector feedback, and Appendix B.2 proves Theorem $2 .$ Appendix B.3 examines the effects of sharing the selector across questions and of randomizing the guidance level. Finally, Appendix B.4 derives the expected update of the selector in the GRPO application as the model and the questions change, and Appendix B.5 bounds the accumulated sampling noise.

## B.1 SETUP AND SELECTOR FEEDBACK

Fix the current model, the references, and $\mathcal { D } _ { \mathrm { g u i d e } }$ as in Section 3.3. We assume that every question satisfies the setup of Appendix A.1, which includes $0 < \pi _ { \theta _ { \mathrm { o l d } } } ( r _ { x } \mid x ) < 1$ and hence $0 < S _ { 0 , x } < \infty$ For a question $x ,$ we write $\nu _ { x }$ for its ideal distribution. For a guidance level $\alpha ,$ we write $Q _ { x , \alpha } , d _ { x , \alpha } ,$ and $S _ { x , \alpha }$ for the distribution of the accepted trajectory, the successful replacement probability, and the reference surprisal at the prefix length $t _ { x } ( \alpha )$ . Here $t _ { x } ( \alpha )$ is the actual prefix length, including the boundary adjustment in Section 4.1. Therefore, the nominal level α can differ from the realized ratio $S _ { x , \alpha } / \bar { S } _ { 0 , x }$ . We first collect the properties of the selector feedback for a single question, for which we suppress x as in Appendix $\mathbf { A }$

Lemma B.1 (Selector feedback). For every prefix length t, the selectorfeedback satisfies $W _ { t } \in [ 0 , 1 ]$ and $\mathbb { E } [ W _ { t } ] = d _ { t } S _ { t } / S _ { 0 }$ . Moreover, with $c _ { t } = ( 1 - d _ { t } ) \log ( 1 - d _ { t } )$ ),

$$
\frac { D _ { \mathrm { K L } } ( Q _ { t } \| \nu ) } { S _ { 0 } } = \frac { D _ { \mathrm { K L } } ( \delta _ { r } \| \nu ) } { S _ { 0 } } - \mathbb { E } [ W _ { t } ] + \frac { c _ { t } } { S _ { 0 } } , \qquad \frac { c _ { t } } { S _ { 0 } } \in \left[ - \frac { 1 } { e S _ { 0 } } , 0 \right] ,\tag{14}
$$

and the normalized distances satisfy $0 \le D _ { \mathrm { K L } } ( Q _ { t } \| \nu ) / S _ { 0 } \le D _ { \mathrm { K L } } ( \delta _ { r } \| \nu ) / S _ { 0 } \le 1$

Proof. The indicator in Equation (7) has expectation $d _ { t }$ by Equation (4). Lemma A.4 gives $0 \leq S _ { t } \leq$ $S _ { 0 }$ , and hence $W _ { t } \in [ 0 , 1 ]$ and $\mathbb { E } [ W _ { t } ] = \bar { d _ { t } } S _ { t } / S _ { 0 }$ , which proves Equation (8). Dividing the closed form of Theorem 1 by $S _ { 0 }$ gives Equation (14), and the range of $c _ { t }$ follows from −a log $\bar { a } \in [ 0 , 1 / e ]$ for $a \in [ 0 , 1 ]$ . Finally, since $\mathbb { E } { \lvert W _ { t } \rvert } ^ { - } \ge 0$ and $c _ { t } \leq 0$ , Equation (14) gives $D _ { \mathrm { K L } } ( Q _ { t } \| \nu ) \leq D _ { \mathrm { K L } } \dot { ( } \delta _ { r } \| \dot { \nu } ) ,$ and Lemma A.2 gives $0 \leq \dot { D _ { \mathrm { K L } } } ( \dot { \delta } _ { r } \Vert \nu ) \leq S _ { 0 }$ □

Thus, a larger expected feedback corresponds to a smaller normalized distance, up to the bounded last term. Since normalization divides $d _ { t } S _ { t }$ at every prefix of the same reference by the same $S _ { 0 }$ , it preserves the ranking of $d _ { t } S _ { t }$ across these prefixes. Because of the bounded last term, this ranking need not equal the ranking of the exact distances. Across questions, normalization expresses each distance relative to the reference surprisal of the empty prefix.

The ablation without surprisal weighting in Section 5.3 replaces $W _ { t }$ with the indicator of successful replacement, whose expectation is $d _ { t } .$ . Since this indicator still excludes exact copies of the reference, comparing it with $W _ { t }$ isolates the effect of weighting a successful replacement by the reference surprisal.

For the rest of this appendix, we write

$$
g _ { x } ( \alpha ) = \mathbb { E } [ W _ { x , \alpha } ] = \frac { d _ { x , \alpha } S _ { x , \alpha } } { S _ { 0 , x } } \in [ 0 , 1 ]
$$

for the expected selector feedback of question x at level α, where the equality follows from Lemma B.1.

## B.2 PROOF OF THEOREM 2

Theorem 2 measures a selector by its average normalized distance and compares it with training on the reference. Written out, these distances are

$$
\mathcal { L } ( \phi ) = \mathbb { E } _ { X \sim \mathcal { D } _ { \mathrm { g u i d e } } , \alpha \sim \rho \phi } \left[ \frac { D _ { \mathrm { K L } } ( Q _ { X , \alpha } \| \nu _ { X } ) } { S _ { 0 , X } } \right] , \qquad \mathcal { L } _ { \mathrm { r e f } } = \mathbb { E } _ { X \sim \mathcal { D } _ { \mathrm { g u i d e } } } \left[ \frac { D _ { \mathrm { K L } } ( \delta _ { r _ { X } } \| \nu _ { X } ) } { S _ { 0 , X } } \right] .
$$

By Lemma B.1 and $g _ { x } ( \alpha ) \in [ 0 , 1 ]$ , the quantities $\mathcal { L } ( \phi ) , \mathcal { L } _ { \mathrm { r e f } }$ , and $J ( \phi )$ lie in $[ 0 , 1 ]$

Theorem 2 (Learning the prefix selector from feedback, restated). Fix the current model, the references, and $\begin{array} { r } { \mathcal { D } _ { \mathrm { g u i d e . } } } \end{array}$ , where every question satisfies the setup of Section 3.1. Let ${ \mathcal { L } } ( \phi )$ and $\mathcal { L } _ { \mathrm { r e f } }$ be the expectations of $D _ { \mathrm { K L } } ( Q _ { X , \alpha } | | \dot { \nu _ { X } } ) / S _ { 0 , X }$ and $\dot { D } _ { \mathrm { K L } } ( \delta _ { r x } \| \dot { \nu _ { X } } ) \big / S _ { 0 , X }$ over $X \sim \mathcal { D } _ { \mathrm { g u i d e } }$ and $\alpha \sim \rho _ { \phi }$ Then

$$
J ( \phi ) \leq \mathcal { L } _ { \mathrm { r e f } } - \mathcal { L } ( \phi ) \leq J ( \phi ) + \mathbb { E } _ { X } \left[ \frac { 1 } { e S _ { 0 , X } } \right] .
$$

Moreover, for a differentiable prefix selector with positive probabilities and a baseline b fixed before sampling,

$$
\begin{array} { r } { \mathbb { E } \left[ ( W _ { X , \alpha } - b ) \nabla _ { \phi } \log \rho _ { \phi } ( \alpha ) \right] = \nabla _ { \phi } J ( \phi ) . } \end{array}
$$

Proof. For the first statement, averaging Equation (14) over $X \sim \mathcal { D } _ { \mathrm { g u i d e } }$ and α $\sim \rho _ { \phi }$ gives

$$
\mathcal { L } ( \phi ) = \mathcal { L } _ { \mathrm { r e f } } - J ( \phi ) + \epsilon ( \phi ) , \qquad - \mathbb { E } _ { X } \frac { 1 } { e S _ { 0 , X } } \le \epsilon ( \phi ) \le 0 ,\tag{15}
$$

where $\epsilon ( \phi ) = \mathbb { E } [ c _ { X , \alpha } / S _ { 0 , X } ]$ and $c _ { x , \alpha }$ is the last term of Theorem 1 at the prefix length $t _ { x } ( \alpha )$ Lemma B.1 bounds each $c _ { x , \alpha } / S _ { 0 , x } .$ , and rearranging Equation (15) gives the first statement.

For the second statement, the level is sampled independently of the question. Condition on $X = x$ Since $\alpha \sim \rho _ { \phi }$ and $\mathbb { E } [ W _ { x , \alpha } \mid \alpha ] = g _ { x } ( \alpha )$ ,

$$
\mathbb { E } \left[ ( W _ { x , \alpha } - b ) \nabla _ { \phi } \log \rho _ { \phi } ( \alpha ) \mid X = x \right] = \sum _ { \alpha \in A } ( g _ { x } ( \alpha ) - b ) \nabla _ { \phi } \rho _ { \phi } ( \alpha ) = \sum _ { \alpha \in A } g _ { x } ( \alpha ) \nabla _ { \phi } \rho _ { \phi } ( \alpha ) .
$$

The baseline term vanishes since $\begin{array} { r } { \sum _ { \alpha } \nabla _ { \phi } \rho _ { \phi } ( \alpha ) = \nabla _ { \phi } 1 = 0 } \end{array}$ . Define $\mu _ { \alpha } = \mathbb { E } _ { X \sim \mathcal { D } _ { \mathrm { g u i d e } } } [ g _ { X } ( \alpha ) ] \ \in$ [0, 1], which does not depend on ϕ. Then

$$
J ( \phi ) = \sum _ { \alpha \in A } \rho _ { \phi } ( \alpha ) \mu _ { \alpha } , \qquad \nabla _ { \phi } J ( \phi ) = \sum _ { \alpha \in A } \mu _ { \alpha } \nabla _ { \phi } \rho _ { \phi } ( \alpha ) ,
$$

where differentiation acts on a finite sum with fixed bounded coefficients. Averaging the conditional identity over $X \sim \mathcal { D } _ { \mathrm { g u i d e } }$ gives the same expression, which proves the second statement. □

The first statement leads to the following guarantee for a selector that maximizes J.

Corollary B.2 (Maximizing the objective). $I f J ( \phi ^ { * } ) \geq J ( \phi )$ , then $\mathcal { L } ( \phi ^ { * } ) \leq \mathcal { L } ( \phi ) + \mathbb { E } _ { X } [ 1 / ( e S _ { 0 , X } ) ]$

Proof. By Equation (15), $J ( \phi ^ { * } ) \ge J ( \phi )$ gives

$$
\mathcal { L } ( \phi ^ { * } ) \leq \mathcal { L } _ { \mathrm { r e f } } - J ( \phi ^ { * } ) \leq \mathcal { L } _ { \mathrm { r e f } } - J ( \phi ) \leq \mathcal { L } ( \phi ) + \mathbb { E } _ { X } \frac { 1 } { e S _ { 0 , X } } .
$$

Thus, a larger objective corresponds to a smaller average normalized distance, up to a bounded correction. Since the correction $\epsilon ( \phi )$ can change with the selector, J and L can rank selectors differently within this bound. The bound is informative only when $\mathbb { E } _ { X } [ 1 / S _ { 0 , X } ]$ is small, and it can be large when some reference surprisals are close to zero.

In Section 4.2, the baseline is a running average of past feedback. For such a baseline, the same argument applies after conditioning on the history and the current question. Given this information, the baseline is fixed, the level still follows $\rho _ { \phi } ,$ and its feedback still has conditional mean $g _ { x } ( \alpha )$ Moreover, the baseline update in Appendix B.4 keeps $b \in [ 0 , 1 ]$ , which ensures integrability.

## B.3 SHARING AND RANDOMIZING GUIDANCE LEVELS

The prefix selector shares one distribution over guidance levels across questions and samples a level for each continuation. We now examine both choices. Proposition B.3 bounds the cost of sharing within the objective J. Proposition B.4 then compares the distance after randomizing the level with the average over levels in J.

For any distribution $\rho$ over A, write $\begin{array} { r } { J ( \rho ) = \sum _ { \alpha } \rho ( \alpha ) \mu _ { \alpha } } \end{array}$ with $\mu _ { \alpha }$ from the proof of Theorem $^ { 2 , }$ which gives $J ( \rho _ { \phi } ) = J ( \phi )$ ). The best shared distribution has value $J _ { \mathrm { s h a r e d } } ^ { * } = \operatorname* { m a x } _ { \alpha } \mu _ { \alpha }$ . In contrast, an oracle that knows $g _ { x }$ for every question and chooses a level for each question separately has value $J _ { \mathrm { i n d i v i d u a l } } ^ { * } = \mathbb { E } _ { X } \operatorname* { m a x } _ { \alpha } g _ { X } ( \alpha )$

Proposition B.3 (Cost of sharing). For every distribution ρ over ${ \mathcal A } ,$

$$
J _ { \mathrm { i n d i v i d u a l } } ^ { * } - J ( \rho ) = \big ( J _ { \mathrm { i n d i v i d u a l } } ^ { * } - J _ { \mathrm { s h a r e d } } ^ { * } \big ) + \big ( J _ { \mathrm { s h a r e d } } ^ { * } - J ( \rho ) \big ) ,\tag{16}
$$

where both terms are nonnegative. Moreover, the prefix selector in Equation (11) satisfies

$$
\operatorname* { s u p } _ { \phi } J ( \phi ) = ( 1 - \varepsilon _ { \mathrm { s e l } } ) \operatorname* { m a x } _ { \alpha } \mu _ { \alpha } + \frac { \varepsilon _ { \mathrm { s e l } } } { | A | } \sum _ { \alpha } \mu _ { \alpha } .
$$

Proof. The decomposition is an identity. The first term is nonnegative since $\begin{array} { r } { \operatorname* { m a x } _ { \alpha } \mathbb { E } _ { X } [ g _ { X } ( \alpha ) ] \le } \end{array}$ $\mathbb { E } _ { X } \operatorname* { m a x } _ { \alpha } g _ { X } ( \alpha )$ . The second term is nonnegative since $J ( \rho )$ is a convex combination of the values $\mu _ { \alpha }$ . For the selector in Equation (11), write $q _ { \phi }$ for its softmax component. Then $J ( \phi ) = ( 1 -$ $\dot { \varepsilon } _ { \mathrm { s e l } } ) \sum _ { \alpha } q _ { \phi } ( \alpha ) \mu _ { \alpha } + ( \varepsilon _ { \mathrm { s e l } } / | A | ) \sum _ { \alpha } \mu _ { \alpha }$ . The first sum is at most $\operatorname* { m a x } _ { \alpha } \mu _ { \alpha }$ , and it approaches this value as $q _ { \phi }$ concentrates on a maximizing level. □

The first term in Equation (16) measures the cost of using the same distribution over levels for every question. It is zero if one level maximizes $g _ { x }$ for almost every question. The second term measures the gap between the current selector and the best shared distribution. If all $\mu _ { \alpha }$ are equal, every shared distribution has the same objective even when individual questions prefer different levels. Moreover, the uniform component of the selector has an explicit cost in the current objective, while it keeps every level observed as the model changes.

Through Theorem 2, the objective J corresponds to the average of the normalized distance over the selected levels. However, sampling a level and then making one continuation produces a single distribution of the accepted trajectory, which mixes the distributions for different levels. For a fixed question x and a distribution $\rho$ over ${ \mathcal { A } } ,$ this mixture is $\begin{array} { r } { Q _ { x , \rho } = \sum _ { \alpha } \rho ( \alpha ) Q _ { x , \alpha } } \end{array}$ . Let $I _ { x }$ be the mutual information between the sampled level and the accepted trajectory, and let $\begin{array} { r } { \mathcal { H } ( \rho ) = - \sum _ { \alpha } \rho ( \alpha ) \log \rho ( \alpha ) } \end{array}$ be the entropy of $\rho .$

Proposition B.4 (Randomizing the guidance level). For every question x and distribution ρ over ${ \mathcal A } ,$

$$
D _ { \mathrm { K L } } ( Q _ { x , \rho } | | \nu _ { x } ) = \sum _ { \alpha } \rho ( \alpha ) D _ { \mathrm { K L } } ( Q _ { x , \alpha } | | \nu _ { x } ) - I _ { x } ,
$$

where $0 \leq I _ { x } \leq \mathcal { H } ( \rho )$ . Consequently, with $\begin{array} { r } { j _ { x } ( \rho ) = \sum _ { \alpha } \rho ( \alpha ) g _ { x } ( \alpha ) } \end{array}$

$$
\frac { D _ { \mathrm { K L } } ( \delta _ { r _ { x } } \| \nu _ { x } ) } { S _ { 0 , x } } - j _ { x } ( \rho ) - \frac { 1 / e + \mathcal { H } ( \rho ) } { S _ { 0 , x } } \leq \frac { D _ { \mathrm { K L } } ( Q _ { x , \rho } \| \nu _ { x } ) } { S _ { 0 , x } } \leq \frac { D _ { \mathrm { K L } } ( \delta _ { r _ { x } } \| \nu _ { x } ) } { S _ { 0 , x } } - j _ { x } ( \rho ) .
$$

Proof. Every $Q _ { x , \alpha }$ has finite distance by Lemma B.1. Splitting the log probability ratio through $Q _ { x , \rho }$ and summing over trajectories with positive probability gives

$$
\begin{array} { r l r } {  { \sum _ { \alpha } \rho ( \alpha ) D _ { \mathrm { K L } } ( Q _ { x , \alpha } \| \nu _ { x } ) = \sum _ { \alpha , y } \rho ( \alpha ) Q _ { x , \alpha } ( y ) [ \log \frac { Q _ { x , \alpha } ( y ) } { Q _ { x , \rho } ( y ) } + \log \frac { Q _ { x , \rho } ( y ) } { \nu _ { x } ( y ) } ] } } \\ & { } & { = I _ { x } + D _ { \mathrm { K L } } ( Q _ { x , \rho } \| \nu _ { x } ) , } \end{array}
$$

since $\begin{array} { r } { \sum _ { \alpha } \rho ( \alpha ) D _ { \mathrm { K L } } ( Q _ { x , \alpha } | | Q _ { x , \rho } ) } \end{array}$ equals the mutual information $I _ { x }$ . The mutual information is nonnegative and at most the entropy of the level, which gives $0 \leq I _ { x } \leq \mathcal { H } ( \rho )$ . For the bounds, divide the identity by $S _ { 0 , x }$ and apply Equation (14) to each level. The upper bound uses $I _ { x } \geq 0$ and $c _ { x , \alpha } \leq 0$ and the lower bound uses $I _ { x } \leq \mathcal { H } ( \rho )$ and $c _ { x , \alpha } \geq - 1 / e$ □

The mutual information is zero when all distributions with positive weight coincide. Otherwise, randomizing the level gives a smaller distance than the average over levels. This construction returns one accepted trajectory from one continuation, and Appendix C.2 analyzes acceptance after multiple continuations.

## B.4 SELECTOR UPDATES IN THE GRPO APPLICATION

Theorem 2 fixes the model, the references, and $\mathcal { D } _ { \mathrm { g u i d e } }$ . We now specify $\mathcal { D } _ { \mathrm { g u i d e } }$ and the baseline in the GRPO application of Section 4.3, and then derive the expected update of the selector for a batch and across training iterations.

Questions selected for reference guidance. Fix the current model and a reference $r _ { x }$ for each question. Since the G responses of a group are sampled independently, the group is an all-failure group with probability $f _ { x } \dot { = } ( 1 - p \dot { \theta _ { \mathrm { o l d } } } \dot { ( x ) ) ^ { G } }$ . For $\bar { X } \sim \mathcal { D }$ , define $\zeta = \mathbb { E } _ { \mathcal { D } } [ f _ { X } ]$ . When $\zeta > 0$ , the questions that receive reference guidance follow

$$
{ \mathcal { D } } _ { \mathrm { g u i d e } } ( x ) = { \frac { { \mathcal { D } } ( x ) f _ { x } } { \zeta } } .\tag{17}
$$

Given the question, the guidance levels and continuations are sampled independently of the group. Therefore, restricting guidance to questions with an all-failure group changes their relative weights while preserving $\mathbb { E } [ { \bar { W } } _ { x , \alpha } ] = g _ { x } ( { \bar { \alpha } } )$ for each question and level. If $\zeta = 0$ , no question receives guidance and the selector update is zero almost surely.

Baseline and update order. The selector starts with $\phi = 0$ and $b = 0$ . Every term in Equation (12) uses the same selector parameters and baseline, which are fixed before the batch is collected. After a nonempty batch of continuations, the baseline is updated by

$$
\boldsymbol { b } \gets ( 1 - \beta ) \boldsymbol { b } + \frac { \beta } { | \mathcal { U } | } \sum _ { j \in \mathcal { U } } \boldsymbol { W } _ { j } ,\tag{18}
$$

where $0 < \beta \le 1$ . Since this update is a convex combination of values in [0, 1], induction gives $b \in \ [ 0 , 1 ]$ at every iteration. Centering the updated logits by subtracting their mean leaves the selection probabilities unchanged. An empty batch of continuations leaves $\phi$ and b unchanged. Table 3 specifies the numerical hyperparameters. Reference surprisals, verification outcomes, and model parameters are fixed during the selector update. This update precedes the policy update, and both use the continuations generated by the same current model. All continuations supply feedback, including K zero feedback values when no continuation gives a successful replacement and ARG falls back to the reference.

Expected batch update. Let Ψ denote the summed direction in Equation (12) before multiplication by the step size,

$$
\Psi = \sum _ { j \in \mathcal { U } } ( W _ { j } - b ) \nabla _ { \phi } \log \rho _ { \phi } ( \alpha _ { j } ) ,
$$

with $\Psi = 0$ when U is empty.

Proposition B.5 (Expected batch update). Consider a batch of n question positions, each with marginal distribution D. Suppose that the model, the selector, and the baseline are fixed during collection, and that every all-failure group receives K continuations whose levels and feedback follow the laws above. Then

$$
\mathbb { E } [ \Psi ] = n K \zeta \nabla _ { \phi } J ( \phi ) ,\tag{19}
$$

where the right-hand side is zero when $\zeta = 0 .$

Proof. Write Ψ as a sum over question positions i and continuations $\ell = 1 , \ldots , K$ of the terms 1{group i is all-failure} $\ r \large \{ ( W _ { i \ell } - \bar { b } ) \nabla _ { \phi } \log ^ { } \rho _ { \phi } ( \alpha _ { i \ell } )$ . Conditional on the question x at position i and on its all-failure event, each term has expectation $\begin{array} { r } { \sum _ { \alpha } g _ { x } ( \alpha ) \nabla _ { \phi } \rho _ { \phi } ( \alpha ) } \end{array}$ by the proof of Theorem 2. Therefore, position i contributes

$$
\sum _ { x } \mathcal { D } ( x ) f _ { x } K \sum _ { \alpha } g _ { x } ( \alpha ) \nabla _ { \phi } \rho _ { \phi } ( \alpha ) = K \zeta \sum _ { x } \mathcal { D } _ { \mathrm { g u i d e } } ( x ) \sum _ { \alpha } g _ { x } ( \alpha ) \nabla _ { \phi } \rho _ { \phi } ( \alpha ) = K \zeta \nabla _ { \phi } J ( \phi ) ,
$$

where the first equality uses Equation (17). Summing over the n positions by linearity proves the claim. When $\zeta \ = \ 0 ,$ , no group is an all-failure group almost surely, and hence $\Psi \ = \ 0$ almost surely. □

Since the proof uses only the marginal laws, it requires no independence between question positions or between continuations. Holding J fixed, the factors n, K, and ζ only scale the expected update at a fixed step size. However, changes in the all-failure probabilities also change the question weights in J and hence the update direction.

Changing models and questions. Across training iterations, the model, the question distribution, and the selector all change. Let $( \mathcal { F } _ { k } ) _ { k }$ be the increasing filtration of the training history, where $\mathcal { F } _ { k }$ contains the history before iteration k, including the current model, the selector parameters $\phi _ { k }$ and the baseline. For the question positions $X _ { k , 1 } , \ldots , X _ { k , n }$ at iteration $k ,$ define their average conditional marginal distribution by

$$
{ \mathcal { D } } _ { k } ( x ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \operatorname* { P r } ( X _ { k , i } = x \mid { \mathcal { F } } _ { k } ) .
$$

With the current all-failure probabilities $f _ { k , x }$ and expected feedback $g _ { k , x } ( \alpha )$ , define

$$
\zeta _ { k } = \mathbb { E } _ { X \sim \mathcal { D } _ { k } } [ f _ { k , X } ] , \qquad \mathcal { D } _ { \mathrm { g u i d e } , k } ( x ) = \frac { \mathcal { D } _ { k } ( x ) f _ { k , x } } { \zeta _ { k } } ,
$$

$$
J _ { k } ( \phi ) = \mathbb { E } _ { X \sim \mathcal { D } _ { \mathrm { g u i d e } , k } } \sum _ { \alpha } \rho _ { \phi } ( \alpha ) g _ { k , X } ( \alpha ) ,
$$

where the last two are defined when $\zeta _ { k } > 0$ . Let $\Psi _ { k }$ be the summed direction at iteration k.

Proposition B.6 (Expected update under changing models and questions). Suppose that, conditional on $\mathcal { F } _ { k } ,$ , the groups and continuations at iteration k follow the laws of Proposition $B . 5$ under the current model, except that position i has the marginal distribution $\operatorname* { P r } ( X _ { k , i } = { \boldsymbol { \cdot } } \mid { \mathcal { F } } _ { k } )$ . Then

$$
\begin{array} { r } { \mathbb { E } [ \Psi _ { k } \ | \ \mathcal { F } _ { k } ] = n K \zeta _ { k } \nabla _ { \phi } J _ { k } ( \phi _ { k } ) , } \end{array}\tag{20}
$$

where the right-hand side is zero when $\zeta _ { k } = 0 .$

Proof. Apply the proof of Proposition B.5 conditionally on $\mathcal { F } _ { k }$ , with the marginal distribution of position i given by $\operatorname* { P r } ( X _ { k , i } = { \mathsf { \bar { \cdot } } } \mid { \mathcal { F } } _ { k } )$ . Since the proof uses only marginal laws, the positions may have different marginal distributions. Summing their contributions replaces $n D$ with $n D _ { k }$ , which gives Equation (20). □

The gradient in Equation (20) holds the history-dependent model, question distribution, and prefix mapping fixed. Thus, the update follows the current objective in expectation even as training changes these quantities. The setting includes sampling with replacement from a fixed distribution, for which $\mathcal { D } _ { k } = \mathcal { D }$ , and sampling without replacement from a remaining pool of questions.

## B.5 ACCUMULATED SAMPLING NOISE

Proposition B.6 describes the expected update at each iteration. We now bound how far the accumulated updates deviate from these expectations, starting with the size of each update.

Lemma B.7 (Bounded update direction). For the selector in Equation (11) and a baseline in $[ 0 , 1 ]$ every term $o f \Psi _ { k }$ has Euclidean norm at most ${ \sqrt { 2 } } .$ Consequently, $\| \Psi _ { k } / ( n K ) \| _ { 2 } \leq \sqrt { 2 } .$

Proof. Let $q _ { \phi }$ be the softmax component of Equation (11) and $\mathbf { e } _ { \alpha }$ the coordinate vector for level $\alpha .$ Then

$$
\nabla _ { \phi } \log \rho _ { \phi } ( \alpha ) = \frac { ( 1 - \varepsilon _ { \mathrm { s e l } } ) q _ { \phi } ( \alpha ) } { \rho _ { \phi } ( \alpha ) } \big ( \mathbf { e } _ { \alpha } - q _ { \phi } \big ) .
$$

The prefactor is at most one since $\rho _ { \phi } ( \alpha ) \geq ( 1 - \varepsilon _ { \mathrm { s e l } } ) q _ { \phi } ( \alpha )$ . Moreover, $\| { \mathbf { e } } _ { \alpha } - q _ { \phi } \| _ { 2 } ^ { 2 } = ( 1 -$ $\begin{array} { r } { q _ { \phi } ( \alpha ) ) ^ { 2 } + \sum _ { \alpha ^ { \prime } \ne \alpha } q _ { \phi } ( \alpha ^ { \prime } ) ^ { 2 } \le 2 ( 1 - q _ { \phi } ( \alpha ) ) ^ { 2 } \le 2 } \end{array}$ . Since $W _ { j }$ and b both lie in $[ 0 , 1 ]$ , we have $| W _ { j } - b | \leq 1$ , which bounds each term by $\sqrt { 2 }$ . Finally, $\Psi _ { k }$ has at most $n K$ terms, including the case of no continuations. □

Define the centered errors

$$
\xi _ { k } = \frac { \Psi _ { k } } { n K } - \zeta _ { k } \nabla _ { \phi } J _ { k } ( \phi _ { k } ) .
$$

Proposition B.8 (Accumulated sampling noise). Under the conditions of Proposition B.6 and a baseline in [0, 1], for a deterministic number $N _ { \mathrm { i t e r } }$ of iterations,

$$
\mathbb { E } \left. \frac { 1 } { N _ { \mathrm { i t e r } } } \sum _ { k = 1 } ^ { N _ { \mathrm { i t e r } } } \xi _ { k } \right. _ { 2 } ^ { 2 } \le \frac { 2 } { N _ { \mathrm { i t e r } } } .\tag{21}
$$

Proof. Proposition B.6 gives $\mathbb { E } [ \xi _ { k } \ | \ \mathcal { F } _ { k } ] = 0$ . The conditional variance identity and Lemma B.7 then give

$$
\mathbb { E } \left[ \left. \xi _ { k } \right. _ { 2 } ^ { 2 } \bigm \rvert \mathcal { F } _ { k } \right] = \mathbb { E } \left[ \left. \frac { \Psi _ { k } } { n K } \right. _ { 2 } ^ { 2 } \middle \rvert \mathcal { F } _ { k } \right] - \left. \mathbb { E } \left[ \frac { \Psi _ { k } } { n K } \middle \rvert \mathcal { F } _ { k } \right] \right. _ { 2 } ^ { 2 } \leq 2 .
$$

For $k < k ^ { \prime }$ , the error $\xi _ { k }$ is measurable with respect to $\mathcal { F } _ { k ^ { \prime } }$ . Therefore, $\mathbb { E } \langle \xi _ { k } , \xi _ { k ^ { \prime } } \rangle = \mathbb { E } \langle \xi _ { k } , \mathbb { E } [ \xi _ { k ^ { \prime } } \ |$ $\mathcal { F } _ { k ^ { \prime } } ] \rangle = 0$ . Expanding the squared norm of the sum, the cross terms vanish and each of the $N _ { \mathrm { i t e r } }$ diagonal terms is at most $^ { 2 , }$ which proves the bound. □

The baseline update in Equation (18) satisfies the condition on the baseline. Proposition B.8 controls the noise around the changing expected directions.

## C ANALYSIS OF MULTIPLE CONTINUATIONS

Theorem 1 concerns a single continuation, whereas ARG samples K continuations for each question and accepts candidate trajectories by the rule in Section 4.1. This appendix analyzes the distribution of the accepted trajectory after multiple continuations. Appendix C.1 fixes one prefix and studies how the distance to the ideal distribution changes with the number of continuations. Appendix C.2 gives a sufficient condition under which an arbitrary acceptance rule keeps the distance no larger than that of training on the reference.

## C.1 REPEATED CONTINUATIONS FROM A FIXED PREFIX

Fix a prefix length t and let $H _ { t }$ be the event that a trajectory begins with $h _ { t } .$ . Since the reference is correct and has positive probability, $0 < u _ { t } \le v _ { t } \le 1$ . For a positive integer k, draw k independent continuations from $h _ { t }$ . The first-success rule returns the first correct candidate trajectory, including an exact copy of $r ,$ and returns r if every candidate trajectory is incorrect. This rule treats an exact copy as a success. Let ${ Q } _ { t } ^ { ( k ) }$ be the resulting distribution of the accepted trajectory, and let $\nu _ { h _ { t } } = \pi ( \cdot \mid C \cap H _ { t } )$ be the ideal distribution restricted to trajectories that begin with $h _ { t }$

Lemma C.1 (Distribution after repeated continuations). Let $w _ { k } = 1 - ( 1 - v _ { t } ) ^ { k }$ be the probability that at least one candidate trajectory is correct, and define $\lambda _ { k } = w _ { k } / v _ { t }$ and $d _ { t } ^ { ( k ) } = d _ { t } \lambda _ { k }$ . Then

$$
Q _ { t } ^ { ( k ) } = ( 1 - w _ { k } ) \delta _ { r } + w _ { k } \nu _ { h _ { t } } ,\tag{22}
$$

and

$$
D _ { \mathrm { K L } } ( Q _ { t } ^ { ( k ) } | | \nu ) = D _ { \mathrm { K L } } ( \delta _ { r } | | \nu ) - d _ { t } ^ { ( k ) } S _ { t } + \left( 1 - d _ { t } ^ { ( k ) } \right) \log \left( 1 - d _ { t } ^ { ( k ) } \right) + d _ { t } ^ { ( k ) } \log \lambda _ { k } .
$$

Proof. A correct trajectory $\boldsymbol { y } = \boldsymbol { h } _ { t } \oplus \boldsymbol { z }$ is returned at continuation j with probability $( 1 - v _ { t } ) ^ { j - 1 } \pi ( z \mid$ $h _ { t } )$ . Summing over $j = 1 , \dots , k$ gives $Q _ { t } ^ { ( k ) } ( y ) \ : = \ : \lambda _ { k } \pi ( z \mid h _ { t } )$ for every correct $y \ne r$ . Since $\nu _ { h _ { t } } ( y ) = \pi ( z \mid h _ { t } ) / v _ { t }$ for these trajectories, this probability equals $w _ { k } \nu _ { h _ { t } } ( y )$ . The same calculation gives $\lambda _ { k } u _ { t }$ for an exact copy, and fallback adds $( 1 - v _ { t } ) ^ { k }$ . Therefore, $Q _ { t } ^ { ( k ) } ( r ) ~ = ~ ( 1 - w _ { k } ) +$ $w _ { k } u _ { t } / v _ { t } = 1 - d _ { t } ^ { ( k ) }$ , which proves Equation (22). For the distance, the trajectories other than r have total probability $d _ { t } ^ { ( k ) }$ and likelihood ratio $\lambda _ { k } p / \pi ( h _ { t } )$ . Following the proof of Theorem 1 with this ratio gives the closed form. □

Compared with the closed form for one continuation, the additional term $d _ { t } ^ { ( k ) }$ log $\lambda _ { k }$ accounts for the larger probability of each trajectory other than the reference after repeated continuations. For $k = 1$ , we have $\lambda _ { 1 } = 1$ , and the lemma recovers Theorem 1. The mixture form in Equation (22) shows that additional continuations move probability from the reference to $\nu _ { h _ { t } }$ , which leads to the following limit.

Proposition C.2 (The limit of a fixed prefix). The distance $D _ { \mathrm { K L } } ( Q _ { t } ^ { ( k ) } \Vert \nu )$ is nonincreasing in k and satisfies

$$
- \log \nu ( H _ { t } ) \leq D _ { \mathrm { K L } } ( Q _ { t } ^ { ( k ) } \| \nu ) \leq - \log \nu ( H _ { t } ) + ( 1 - v _ { t } ) ^ { k } \log \frac { v _ { t } } { u _ { t } } .
$$

Consequently, it converges $t o \mathrm { ~ - ~ } \log \nu ( H _ { t } )$ as $k \to \infty$

Proof. For $y \in C \cap H _ { t }$ , we have $\nu ( y ) = \nu ( H _ { t } ) \nu _ { h _ { t } } ( y )$ . Hence, every distribution $Q$ supported on C∩H satisfies $D _ { \mathrm { K L } } ( Q \| \nu ) = D _ { \mathrm { K L } } ( Q \| \nu _ { h _ { t } } ) - \log \nu ( H _ { t } )$ . By Lemma $\mathbf { C } . 1 , Q _ { t } ^ { ( k ) }$ is such a distribution. Therefore, it suffices to study $D _ { \mathrm { K L } } ( Q _ { t } ^ { ( k ) } \Vert \nu _ { h _ { t } } )$ , and the lower bound follows from its nonnegativity. For the upper bound, convexity of KL applied to Equation (22) gives $D _ { \mathrm { K L } } ( Q _ { t } ^ { ( k ) } | | \nu _ { h _ { t } } ) \leq ( 1 -$ $w _ { k } ) D _ { \mathrm { K L } } ( { \bar { \delta } } _ { r } \| \nu _ { h _ { t } } )$ , where $D _ { \mathrm { K L } } ( \dot { \delta } _ { r } | | \nu _ { h _ { t } } ) = \dot { - } \log \nu _ { h _ { t } } ( \dot { r ) } = \log ( v _ { t } / \dot { u _ { t } } )$ and $1 - \widehat { w _ { k } } = ( 1 - v _ { t } ) ^ { k }$ . For monotonicity, take $\ell > k . \mathrm { I f } w _ { k } = 1$ , then $Q _ { t } ^ { ( k ) } = Q _ { t } ^ { ( \ell ) } = \nu _ { h _ { t } }$ . Otherwise, $a = ( 1 - w _ { \ell } ) / ( 1 - w _ { k } ) \in$ [0, 1] satisfies $Q _ { t } ^ { ( \ell ) } = a Q _ { t } ^ { ( k ) } + ( 1 - a ) \nu _ { h _ { i } }$ , and convexity gives

$$
D _ { \mathrm { K L } } ( Q _ { t } ^ { ( \ell ) } | | \nu _ { h _ { t } } ) \leq a D _ { \mathrm { K L } } ( Q _ { t } ^ { ( k ) } | | \nu _ { h _ { t } } ) \leq D _ { \mathrm { K L } } ( Q _ { t } ^ { ( k ) } | | \nu _ { h _ { t } } ) .
$$

Finally, the upper bound converges $\mathrm { t o } - \log \nu ( H _ { t } )$ since $v _ { t } \geq u _ { t } > 0$

For the empty prefix, $H _ { t }$ contains every trajectory. Therefore, the limiting distribution is ν and the limiting distance is zero. A nonempty prefix has the same limit exactly when $H _ { t }$ contains all correct trajectories with positive probability. Proposition C.2 orders the numbers of continuations for each fixed prefix. However, comparing different prefixes also requires their respective $u _ { t }$ and $v _ { t } .$ , which determine both the rate and the limit of the decrease.

## C.2 A SUFFICIENT CONDITION FOR GENERAL ACCEPTANCE RULES

To cover the acceptance rule of ARG, which excludes exact copies, mixes several prefixes, prefers some candidate trajectories over others, and may accept several of them, we consider a general procedure. Choose $K \geq 1$ reference prefixes $h _ { t _ { 1 } } , \ldots , \bar { h } _ { t _ { K } }$ before generation. Conditional on this list, draw a continuation $Z _ { j } \sim \pi ( \cdot \mid h _ { t _ { i } } )$ and form the candidate trajectory $\widetilde { Y } _ { j } = h _ { t _ { i } } \oplus Z _ { j }$ for each j. The continuations may depend on each other, provided that each candidate trajectory has this conditional marginal law given the entire list. A procedure returns r or one of the correct candidate trajectories. If several trajectories are accepted, we consider the distribution obtained by selecting one of them with nonnegative weights that sum to one. Let $Q$ denote the resulting distribution of the accepted trajectory, including fallback, and use the convention 0 log $0 = 0$

Proposition C.3 (Multiple continuations and acceptance). Under the setup above, define $d _ { Q } =$ $1 - Q ( r )$ and $\begin{array} { r } { \Lambda = \sum _ { j = 1 } ^ { K } \exp ( - S _ { t _ { j } } ) } \end{array}$ . Then

$$
D _ { \mathrm { K L } } ( Q \| \nu ) \leq D _ { \mathrm { K L } } ( \delta _ { r } \| \nu ) + d _ { Q } \log \Lambda + ( 1 - d _ { Q } ) \log ( 1 - d _ { Q } ) .\tag{23}
$$

In particular, $\Lambda \leq 1$ guarantees $D _ { \mathrm { K L } } ( Q \| \nu ) \leq D _ { \mathrm { K L } } ( \delta _ { r } \| \nu )$ . A sufficient condition is min<sub>j</sub> $S _ { t _ { j } } ~ \geq$ log K.

Proof. A trajectory $y \neq r$ can be selected only if it appears among the candidate trajectories. Thus,

$$
Q ( y ) \leq \sum _ { j = 1 } ^ { K } \operatorname* { P r } ( \widetilde { Y } _ { j } = y ) = \sum _ { j = 1 } ^ { K } \mathbf { 1 } \{ y \mathrm { ~ b e g i n s ~ w i t h ~ } h _ { t _ { j } } \} \frac { \pi ( y ) } { \pi ( h _ { t _ { j } } ) } \leq \frac { \pi ( y ) \Lambda } { \pi ( r ) } ,
$$

where the last inequality uses $\pi ( r ) / \pi ( h _ { t _ { i } } ) = \exp ( - S _ { t _ { i } } )$ from Lemma $\mathsf { A } . 4$ . Every $y \ne r$ with $Q ( y ) > 0$ is correct and has $\pi ( y ) > 0$ . Hence, $Q ( y ) / \nu ( y ) ^ { ' } \leq p \Lambda / \pi ( r )$ . Since these trajectories have total probability $d _ { Q }$ and the reference has probability $1 - d _ { Q }$ with $\dot { Q ( r ) } / \nu ( r ) = ( 1 - \dot { d } _ { Q } ) p / \pi ( r )$

$$
D _ { \mathrm { K L } } ( Q \| \nu ) \leq \log \frac { p } { \pi ( r ) } + ( 1 - d _ { Q } ) \log ( 1 - d _ { Q } ) + d _ { Q } \log \Lambda .
$$

Lemma A.2 gives $\log ( p / \pi ( r ) ) = D _ { \mathrm { K L } } ( \delta _ { r } \| \nu )$ , which proves Equation (23). Since $( 1 - d _ { Q } ) \log ( 1 -$ $d _ { Q } ) \leq 0 ,$ and $d _ { Q }$ log $\Lambda \leq 0$ when $\Lambda \leq 1$ , the condition $\Lambda \leq \mathrm { i }$ gives the comparison with training on the reference. Finally, min<sub>j</sub> $S _ { t _ { j } } \geq \log K$ gives $\Lambda \le K \exp ( - \log K ) = 1$ □

The argument permits arbitrary selection among correct candidate trajectories, including the exclusion of exact copies and the priority rule of ARG. For one prefix used K times, $\Lambda = K \exp ( - S _ { t } )$ and the bound becomes $D _ { \mathrm { K L } } ( \delta _ { r } | | \dot { \nu } ) - d _ { Q } ( S _ { t } - \log K ) + \mathrm { \hat { ( 1 - } } d _ { Q } ) \log ( 1 - d _ { Q } )$ . The term log K accounts for the possible concentration from selecting among several candidate trajectories. For $K = 1$ and the acceptance rule in Equation (1), we have $d _ { Q } = d _ { t }$ , and the bound becomes the equality in Theorem 1.

In ARG, the prefixes come from guidance levels sampled before generation. The following corollary extends the comparison to this case.

Corollary C.4 (Guidance levels sampled before generation). Under the setup of Proposition $C . 3 ,$ suppose that the K guidance levels are sampled before generation from any distribution over $\mathcal { A } ^ { K }$ and that min $_ { ( \alpha \in \mathcal { A } } S _ { t _ { x } ( \alpha ) } \geq$ log K. Then the distribution $Q$ ofthe accepted trajectory, averaged over the sampled levels, satisfies $D _ { \mathrm { K L } } ( Q \| \nu ) \leq D _ { \mathrm { K L } } ( \delta _ { r } \| \nu )$

Proof. Conditional on each list of levels, every prefix satisfies $S _ { t _ { j } } ~ \geq ~ \log K$ . Therefore, Proposition C.3 bounds the conditional distance by $D _ { \mathrm { K L } } ( \delta _ { r } \Vert \nu )$ . Since $Q$ is the average of these conditional distributions, convexity of KL in its first argument gives the same bound for $Q .$ □

The condition in Corollary C.4 uses the actual prefix lengths after the boundary adjustment and is available from the same reference surprisals used for prefix selection.

## D ANALYSIS OF LEARNING FROM INSERTED TRAJECTORIES

The preceding appendices compare distributions of accepted trajectories at a fixed model. This appendix examines how the inserted trajectories contribute to the policy update in Section 4.3. Appendix D.1 derives the gradient of inserted trajectories in Equation (13). Appendix D.2 derives the expected group update under binary rewards for one inserted trajectory, and Appendix D.3 extends it to multiple inserted trajectories. Finally, Appendix D.4 relates learning the complete trajectory and learning only the continuation to their respective success events.

## D.1 THE GRADIENT OF INSERTED TRAJECTORIES

Condition on the realized batch, its masks, and the recomputed group advantages. Let I index the inserted trajectories, and let $T > 0$ be the total number of active response tokens in the batch loss. For token j of an inserted trajectory $y _ { i } .$ , let $\ell _ { i , j } ( \theta )$ be its log probability given the original question and all preceding tokens of the trajectory. Inserted trajectories use the clipped objective of Equation (29) in Appendix E.3 with the stopped current probability in the ratio denominator,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { i n s e r t e d } } ( \theta ) = \displaystyle - \frac { 1 } { T } \sum _ { i \in \mathcal { I } } \sum _ { j = 1 } ^ { | y _ { i } | } \operatorname* { m i n } \bigl \{ \widetilde { \omega } _ { i , j } ( \theta ) A _ { i } , \mathrm { c l i p } ( \widetilde { \omega } _ { i , j } ( \theta ) , 0 . 8 , 1 . 3 ) A _ { i } \bigr \} , } \\ & { \qquad \widetilde { \omega } _ { i , j } ( \theta ) = \displaystyle \exp \bigl ( \ell _ { i , j } ( \theta ) - \mathrm { s t o p g r a d } ( \ell _ { i , j } ( \theta ) ) \bigr ) , } \end{array}
$$

where stopgrad preserves a value but assigns zero derivative to it.

Lemma D.1 (Gradient of inserted trajectories). Suppose that the advantages $A _ { i }$ and the token count $T$ are heldfixed during differentiation and that every token of an inserted trajectory is active. Then the gradient of $\mathcal { L } _ { \mathrm { i n s e r t e d } }$ is given by Equation (13).

Proof. By construction, $\widetilde { \omega } _ { i , j } = 1$ and $\nabla _ { \theta } \widetilde { \omega } _ { i , j } = \nabla _ { \theta } \ell _ { i , j }$ . Since the clipping interval contains one in its interior, both branches of the clipped loss have the same derivative at this ratio. Therefore, for any fixed real advantage $A _ { i }$ , the loss of token $j$ has derivative $- A _ { i } \nabla _ { \theta } \ell _ { i , j } / T$ . Summing over all positions, including any supplied prefix and the termination token, gives $- \tilde { A _ { i } } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i } \mid x _ { i } ) / T$ for each inserted trajectory. Summing over $i \in \mathcal { T }$ proves Equation (13). □

Lemma D.1 identifies the gradient under the prescribed rule of automatic differentiation. For $A _ { i } > 0$ minimizing $\mathcal { L } _ { \mathrm { i n s e r t e d } }$ gives a positively weighted supervised fine-tuning (SFT) contribution. The advantage specifies its learning weight, while candidate generation and acceptance determine the distribution of the trajectory.

## D.2 THE GROUP UPDATE WITH ONE INSERTED TRAJECTORY

We now use binary correctness rewards to separate the contributions of correct and incorrect responses. This simplification ignores the format reward, which leaves the advantages unchanged when all responses in a group receive the same format reward. The calculation is at the current model and precedes division by the batch token count. Fix a question x with set of correct trajectories $C ,$ and suppress the conditioning on x unless shown explicitly. Assume a finite trajectory space, a differentiable policy with positive trajectory probabilities in a neighborhood of $\theta _ { \mathrm { o l d } }$ , and $0 < p = \pi _ { \theta _ { \mathrm { o l d } } } ( C ) < 1$ . Draw $G \geq 2$ independent responses from $\pi _ { \theta _ { \mathrm { o l d } } }$ and condition on all of them failing verification. Then replace one uniformly chosen response by $\ddot { Y } \sim Q$ , where $Q$ is supported on $C ,$ and where $Y$ and the replaced position are independent of the original failures. Both $Q$ and the collected responses are fixed during differentiation. Set

$$
p _ { \theta } = \pi _ { \theta } ( C ) , \qquad \nu _ { \theta } = \pi _ { \theta } ( \cdot \mid C ) , \qquad \mu _ { \theta } = \pi _ { \theta } ( \cdot \mid C ^ { c } ) ,
$$

and write $\nu = \nu _ { \theta _ { \mathrm { o l d } } }$ for the ideal distribution at the current model.

At the current model, the ratios of the original responses equal one and clipping is inactive. The inserted trajectory follows Lemma D.1, and all response tokens are included. We consider the group ascent direction, which is the sum of the reward-dependent gradients before division by the batch token count. With one inserted trajectory, the reward mean is $\bar { 1 } / G$ and the sample standard deviation is $s _ { R } = 1 / \sqrt { G }$ . Therefore, the inserted trajectory has advantage $( G - 1 ) / [ G ( s _ { R } + \delta ) ]$ ], and each failure has advantage $- 1 / [ G ( s _ { R } + \delta ) ]$ , where $\delta > 0$ is the stabilizer in Equation (28). Define the positive coefficient

$$
c _ { G } = \frac { G - 1 } { G ( 1 / \sqrt { G } + \delta ) } .
$$

Let $F _ { 1 } , \ldots , F _ { G - 1 }$ denote the original failures that remain after replacement. The realized group direction is

$$
g = c _ { G } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( Y \mid x ) - \frac { 1 } { G - 1 } \sum _ { j = 1 } ^ { G - 1 } \nabla _ { \theta } \log \pi _ { \theta } ( F _ { j } \mid x ) \right] _ { \theta = \theta _ { \mathrm { o l d } } } .\tag{24}
$$

Proposition D.2 (Inserted trajectories and the group update). Under the setup above, let $\bar { g } _ { Q } = \mathbb { E } [ g ]$ be the expectation over the original failures, the replaced position, and the inserted trajectory. Then

$$
\bar { g } _ { Q } = c _ { G } \left. \nabla _ { \theta } \left[ \log \frac { p _ { \theta } } { 1 - p _ { \theta } } - { \cal D } _ { \mathrm { K L } } ( Q \| \nu _ { \theta } ) \right] \right. _ { \theta = \theta _ { \mathrm { o l d } } } .\tag{25}
$$

In particular, for $Q = \nu ,$

$$
\bar { g } _ { \nu } = \frac { c _ { G } } { p ( 1 - p ) } \left. \nabla _ { \theta } p _ { \theta } \right| _ { \theta = \theta _ { \mathrm { o l d } } } .\tag{26}
$$

Proof. Conditioning independent responses on all of them belonging to $C ^ { c }$ gives independent draws from $\mu _ { \theta _ { \mathrm { o l d } } }$ , and uniform replacement preserves this distribution for the remaining failures. Therefore, with $F \sim \mu _ { \theta _ { \mathrm { o l d } } }$

$$
\frac { \bar { g } _ { Q } } { c _ { G } } = \Bigl [ \mathbb { E } _ { Q } \nabla _ { \theta } \log \pi _ { \theta } ( Y ) - \mathbb { E } _ { \mu _ { \theta _ { \mathrm { o l d } } } } \nabla _ { \theta } \log \pi _ { \theta } ( F ) \Bigr ] \Big | _ { \theta = \theta _ { \mathrm { o l d } } } .
$$

Applying the argument of Lemma $\mathrm { A . 1 }$ to the event $C ^ { c }$ gives $\mathbb { E } _ { \mu _ { \theta _ { \mathrm { o l d } } } } \nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( F ) = \nabla _ { \boldsymbol { \theta } } \log ( 1 - p _ { \boldsymbol { \theta } } )$ at $\theta _ { \mathrm { o l d } }$ . For fixed $Q ,$ , expanding log $\nu _ { \theta } ( y ) = \log \pi _ { \theta } ( y ) - \log p _ { \theta }$ gives

$$
\begin{array} { r } { \nabla _ { \theta } D _ { \mathrm { K L } } ( Q \| \nu _ { \theta } ) = - \mathbb { E } _ { Q } \nabla _ { \theta } \log \pi _ { \theta } ( Y ) + \nabla _ { \theta } \log p _ { \theta } . } \end{array}
$$

Substituting both expressions proves Equation (25). When $Q = \nu ,$ the gradient of the KL term vanishes at $\theta _ { \mathrm { o l d } }$ by Lemma ${ \mathrm { A . 1 } }$ . Since $\nabla _ { \theta } \log [ p _ { \theta } / ( 1 - p _ { \theta } ) ] = \nabla _ { \theta } p _ { \theta } / [ p _ { \theta } ( 1 - p _ { \theta } ) ]$ ], this gives Equation (26). □

The ratio $p _ { \theta } / ( 1 - p _ { \theta } )$ is the odds of passing verification. The first term in Equation (25) favors success over failure. The second term records which correct trajectories the distribution Q favors.

For training on the reference, $D _ { \mathrm { K L } } ( \delta _ { r } \| \nu _ { \theta } ) = - \log \nu _ { \theta } ( r )$ , and this term favors the particular reference within the set of correct trajectories. In contrast, Equation (26) shows that inserting trajectories from the ideal distribution gives an expected group update proportional to the gradient of the success probability under the original question.

For reference fallback, $Y = r$ makes the positive term in Equation (24) a weighted SFT gradient. For an accepted candidate trajectory, the positive term learns its complete sequence, including the supplied prefix. In both cases, the original failures supply the negative terms in the same group update. This combination of learning from correct trajectories with feedback on the model’s own failures connects our update to prior work on integrating supervised and reinforcement fine-tuning (Yan et al., 2026, Fu et al., 2026).

## D.3 MULTIPLE INSERTED TRAJECTORIES

We keep the preceding setup but insert a random number $N _ { + }$ of correct trajectories with $1 \leq N _ { + } <$ G, where a single reference is inserted when no candidate trajectory is accepted. The inserted trajectories may depend on each other. However, they, their number $N _ { + }$ , and the replaced positions, which are uniformly chosen and distinct, are independent of the original failures. Conditional on $N _ { + } = m$ , let $\widehat { Q }$ and $\widehat { M }$ be the empirical distributions of the inserted trajectories and of the remaining failures. Substituting the advantages $( G - m ) / [ G ( s _ { m } + \delta ) ]$ and $- \hat { m _ { \ L } } / [ G ( s _ { m } + \delta ) ]$ of the inserted trajectories and the remaining failures gives the group direction

$$
\begin{array} { r } { g = \kappa _ { m } \left[ \mathbb { E } _ { \widehat { Q } } \nabla _ { \theta } \log \pi _ { \theta } ( Y ) - \mathbb { E } _ { \widehat { M } } \nabla _ { \theta } \log \pi _ { \theta } ( F ) \right] _ { \theta = \theta _ { \mathrm { o l d } } } , } \end{array}
$$

where

$$
\kappa _ { m } = \frac { m ( G - m ) } { G ( s _ { m } + \delta ) } , \qquad s _ { m } = \sqrt { \frac { m ( G - m ) } { G ( G - 1 ) } } .
$$

Define the average coefficient and the advantage-weighted distribution of inserted trajectories by

$$
\bar { \kappa } = \mathbb { E } [ \kappa _ { N _ { + } } ] , \qquad Q _ { + } ( y ) = \frac { \mathbb { E } [ \kappa _ { N _ { + } } \widehat { Q } ( y ) ] } { \bar { \kappa } } .
$$

Since $1 \leq N _ { + } < G$ , we have $\bar { \kappa } > 0 ,$ and $Q _ { + }$ is a probability distribution on $C .$

Proposition D.3 (Multiple inserted trajectories). Under the setup above,

$$
\mathbb { E } [ g ] = \bar { \kappa } \left. \nabla _ { \theta } \left[ \log \frac { p _ { \theta } } { 1 - p _ { \theta } } - { D } _ { \mathrm { K L } } ( Q _ { + } \| \nu _ { \theta } ) \right] \right. _ { \theta = \theta _ { \mathrm { o l d } } } .
$$

For one inserted trajectory, $\kappa _ { 1 } = c _ { G }$ and $Q _ { + } = Q$ , which recovers Proposition D.2.

Proof. Conditional on the inserted trajectories and their number, uniform replacement preserves the failure distribution $\mu _ { \theta _ { \mathrm { o l d } } }$ for the remaining failures. Therefore, $\mathbb { E } [ \kappa _ { N _ { + } } \widehat { M } ( y ) ] = \bar { \kappa } \mu _ { \theta _ { \mathrm { o l d } } } ( y )$ , and taking expectations in the group direction gives

$$
\begin{array} { r } { \mathbb { E } [ g ] = \bar { \kappa } \left[ \mathbb { E } _ { Q _ { + } } \nabla _ { \theta } \log \pi _ { \theta } ( Y ) - \mathbb { E } _ { \mu _ { \theta _ { \mathrm { o l d } } } } \nabla _ { \theta } \log \pi _ { \theta } ( F ) \right] _ { \theta = \theta _ { \mathrm { o l d } } } . } \end{array}
$$

The proof of Proposition D.2 then applies with $c _ { G }$ replaced by κ¯ and $Q$ replaced by $Q _ { + }$ . Finally, $s _ { 1 } = 1 / \sqrt { G }$ gives $\kappa _ { 1 } = c _ { G }$ □

The distribution $Q _ { + }$ weights each realized set of inserted trajectories by its total positive advantage $\kappa _ { { N + } }$ . Therefore, it accounts for both acceptance and the resulting composition of the group.

## D.4 LEARNING THE PREFIX AND THE CONTINUATION

The likelihood in Equation (13) includes every token of the complete accepted trajectory under the original question. For a candidate trajectory from a nonempty prefix, the chain rule gives

$$
\log \pi _ { \theta } ( h _ { t } \oplus z \mid x ) = \log \pi _ { \theta } ( h _ { t } \mid x ) + \log \pi _ { \theta } ( z \mid x , h _ { t } ) .\tag{27}
$$

Hence, learning covers both producing the supplied prefix and completing the trajectory, and reference fallback trains the complete reference through Equation (13). We now connect the two terms in Equation $( 2 7 )$ to different success events. Fix a prefix length t, let $H _ { t }$ be the event that a trajectory begins with $h _ { t }$ , and let $\nu _ { h _ { t } } = \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid C \cap H _ { t } )$ as in Appendix C.1. This distribution is well defined since the correct reference has positive probability. Let $Q _ { Z , t } ( z ) = \nu _ { h _ { t } } ( h _ { t } \oplus z )$ be the distribution of its continuations. Here $\pi _ { \theta } ( C \mid H _ { t } )$ is the probability that a continuation from $h _ { t }$ gives a correct candidate trajectory, which equals $v _ { t }$ at $\theta _ { \mathrm { o l d } }$

Proposition D.4 (Learning the prefix and the continuation). Under the assumptions ofLemma A.1, at $\theta = \theta _ { \mathrm { o l d } }$

$$
\begin{array} { r } { \mathbb { E } _ { Y \sim \nu _ { h _ { t } } } \nabla _ { \theta } \log \pi _ { \theta } ( Y ) = \nabla _ { \theta } \log \pi _ { \theta } ( C \cap H _ { t } ) , \qquad \mathbb { E } _ { Z \sim Q _ { \varepsilon , t } } \nabla _ { \theta } \log \pi _ { \theta } ( Z \mid h _ { t } ) = \nabla _ { \theta } \log \pi _ { \theta } ( C \mid H _ { t } ) . } \end{array}
$$

Moreover, the distribution $Q _ { t } ^ { ( k ) }$ of the accepted trajectory after k continuations in Lemma C.1 satisfies

$$
\begin{array} { r } { \mathbb { E } _ { Y \sim Q _ { t } ^ { ( k ) } } \nabla _ { \theta } \log \pi _ { \theta } ( Y ) = ( 1 - w _ { k } ) \nabla _ { \theta } \log \pi _ { \theta } ( r ) + w _ { k } \nabla _ { \theta } \log \pi _ { \theta } ( C \cap H _ { t } ) . } \end{array}
$$

Proof. The first identity follows from Lemma A.1 with $C$ replaced by $C \cap H _ { t }$ . For the second, Equation (27) gives $\begin{array} { r } { \nabla _ { \theta } \log \pi _ { \theta } ( h _ { t } \oplus Z ) = \nabla _ { \theta } \log \pi _ { \theta } ( h _ { t } ) + \nabla _ { \theta } ^ { \circ } \log \pi _ { \theta } ( \dot { Z } \mid h _ { t } ) } \end{array}$ , and $\pi _ { \theta } ( C \cap H _ { t } ) =$ $\pi _ { \theta } \overline { { ( h _ { t } ) } } \pi _ { \theta } ( C \mid \bar { H } _ { t } )$ . Subtracting $\nabla _ { \theta } \log \pi _ { \theta } ( h _ { t } )$ from both sides of the first identity gives the second. Finally, the mixture form in Equation (22), whose weights are fixed at $\theta _ { \mathrm { o l d } }$ , and the first identity give the last statement. □

Under $\nu _ { h _ { t } }$ , the expected gradient of the complete trajectory addresses the joint event of producing the prefix and passing verification. In contrast, learning only the continuation addresses verification given an externally supplied prefix. The two coincide for the empty prefix. The joint event covers the correct trajectories that begin with this prefix, while the overall success event C also includes correct trajectories that begin differently. Finally, the mixture identity in Proposition D.4 shows how fallback and successful continuations contribute to the expected gradient of the accepted trajectory before any weighting.

## E IMPLEMENTATION AND EVALUATION DETAILS

We first specify the training procedure, reference boundaries, and policy loss. We then give the training configurations, question prompt, and evaluation protocol. Together, these details connect the single-continuation analysis in Theorem 1 to the complete training procedure.

## E.1 TRAINING PROCEDURE

Training uses 11,068 screened DeepMath examples with matched offline DeepSeek-R1 reference solutions (Guo et al., 2025). We start from DeepMath-103K, keep problems with difficulty at least 5.0, and drop problems whose gold answer is not a single short expression, is a multiple-choice letter, or cannot be verified by our checker when written inside \boxed{}. From the remaining problems, we randomly sample 16,000 in proportion to difficulty and let the base Qwen3-4B attempt each one eight times under the training prompt and sampling settings. Problems solved in all eight attempts are removed. We also drop the few problems whose reference solution does not end in the gold answer. Table 3 gives the ARG configuration. Algorithm 1 separates collection under a fixed policy from updates of the selector and policy. All continuations, including failures and candidate trajectories not selected for the policy update, provide selector feedback.

In the algorithm, $\mathcal { U } _ { x }$ contains K distinct batch-wide continuation indices for question x. Candidate quantities are computed for every $j \in \mathcal { U } _ { x }$ . The operation Accept returns the candidate indices selected by the rule in Section 4.3.

## E.2 PREFIX BOUNDARIES

The level $\alpha = 1$ always uses the empty prefix. For $\alpha < 1$ , we take the latest prefix length whose reference surprisal is at least $\alpha S _ { 0 }$ and, if this length is positive, move it to the first valid sentence or formula boundary at or after it. If no such boundary exists, we use the latest earlier valid boundary, or the empty prefix if there is none. A valid boundary falls on a token boundary, closes all mathematical delimiters, and leaves a nontrivial suffix containing the final boxed answer. The selector feedback uses the reference surprisal $S _ { t }$ at this adjusted prefix length, and the prefix counts toward the maximum trajectory length.

```latex
Algorithm 1 One ARG training update with GRPO
Input Batch B with references, policy θ, selector $\phi ,$ baseline $b .$
1. $\theta _ { \mathrm { o l d } }  \theta , \mathcal { U }  \emptyset$
2. $\mathbf { y } _ { x } = ( y _ { x , i } ) _ { i = 1 } ^ { G } , y _ { x , i } \overset { \mathrm { i i d } } { \sim } \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x )$ for $x \in B$
3. for $\begin{array} { r } { x \in \mathcal { B } \mathrm { ~ w i t h ~ } \sum _ { i = 1 } ^ { G } R _ { \operatorname { a c c } } ( x , y _ { x , i } ) = 0 } \end{array}$ do
4. $S _ { t } \gets - \log \overrightarrow { \pi } _ { \theta _ { \mathrm { o l d } } } ^ {  - } ( a _ { t } \mid \boldsymbol { x } , h _ { t } ) \mathrm { f o r } 0 \leq t < | r | , \mathcal { T } _ { x } \gets \emptyset$
5. $\mathbf { i f } \mathrm { 0 } < S _ { 0 } < \infty$ then
6. $\alpha _ { j } \stackrel { \mathrm { \tiny ~ 1 1 d } } { \sim } \rho _ { \phi } , t _ { j }  t _ { x } ( \alpha _ { j } )$
7. $Z _ { j } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x , h _ { t _ { j } } ) , \widetilde { Y } _ { j }  h _ { t _ { j } } \oplus Z _ { j }$
8. $W _ { j }  \mathbf { 1 } \{ \widetilde { Y } _ { j } \in C _ { x } , \widetilde { Y } _ { j } \neq r \} S _ { t _ { j } } / S _ { 0 }$
9. $\mathcal { U }  \mathcal { U } \cup \mathcal { U } _ { x } , \mathcal { I } _ { x }  \operatorname { A c c e p t } ( ( \widetilde { Y } _ { j } , t _ { j } ) _ { j \in \mathcal { U } _ { x } } , r )$
10. end if
11. Replace rows in $\mathbf { y } _ { x }$ by $( \widetilde { Y } _ { j } ) _ { j \in \mathcal { I } _ { x } }$ , or by r if ${ \mathcal { I } } _ { x } = { \mathcal { O } } .$
12. end for
13. Recompute final rewards $R _ { i } ,$ then $A _ { i }  ( R _ { i } - \bar { R } ) / ( s _ { R } + \delta )$
14. if U ̸= ∅ then
15. $\begin{array} { r } { \dot { \phi }  \phi + \eta _ { \mathrm { s e l } } \sum _ { j \in \mathcal { U } } ( W _ { j } - b ) \nabla _ { \phi } \log \rho _ { \phi } ( \alpha _ { j } ) } \end{array}$
16. $\begin{array} { r } { b  ( 1 - \beta ) b + \frac { \beta } { | \mathcal { U } | } \sum _ { j \in \mathcal { U } } W _ { j } } \end{array}$
17. end if
18. Update θ using Equation (13) and the remaining-response loss in Appendix E.3.
```

<table><tr><td>Setting</td><td>ARG configuration</td></tr><tr><td>Questions per update</td><td>32</td></tr><tr><td>Ordinary responses per question G</td><td>8</td></tr><tr><td>Continuations per all-failure group K</td><td>8</td></tr><tr><td>Guidance levels A</td><td> $\{ 1 , 1 / 2 , 1 / 4 \}$ </td></tr><tr><td>Total acceptance cap in Algorithm 1</td><td>4</td></tr><tr><td>Maximum accepted trajectories with empty prefixes</td><td>4</td></tr><tr><td>Maximum accepted trajectories with nonempty prefixes</td><td>1</td></tr><tr><td>Selector exploration  $\varepsilon _ { \mathrm { s e l } }$ </td><td>0.2</td></tr><tr><td>Selector step size  $\eta _ { \mathrm { s e l } }$ </td><td>0.3</td></tr><tr><td>Baseline update rate  $\beta$ </td><td>0.1</td></tr><tr><td>Training budget</td><td>200 updates</td></tr><tr><td>Learning rate</td><td>10⁻⁶</td></tr><tr><td>Warmup updates</td><td>10</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient norm clipping</td><td>1</td></tr><tr><td>Policy optimization epochs per batch</td><td>1</td></tr><tr><td>Clipping ratio interval</td><td>[0.8, 1.3]</td></tr><tr><td>Policy loss averaging</td><td>Batch token mean</td></tr><tr><td>Total reward</td><td></td></tr><tr><td></td><td>Accuracy plus 0.2 format</td></tr><tr><td>Advantage stabilizer δ</td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Training temperature</td><td>1</td></tr><tr><td>Training top-p and top-k</td><td>1 and no truncation</td></tr><tr><td>Question length limit</td><td>3072 tokens</td></tr><tr><td>Trajectory length budget including the prefix</td><td>8192 tokens</td></tr><tr><td>Additional KL loss</td><td>None</td></tr><tr><td>Entropy loss coefficient</td><td>0</td></tr></table>

Table 3. ARG configuration for both model sizes.

## E.3 POLICY LOSS

Final rewards $R _ { i }$ combine binary verifier accuracy with 0.2 times a binary format reward indicating an extracted boxed answer. Thus, a correct response receives 1.2, while an incorrect response receives 0.2 or zero according to its format. We recompute advantages within each final group as

$$
A _ { i } = \frac { R _ { i } - \bar { R } } { s _ { R } + \delta } ,\tag{28}
$$

where $\bar { R }$ and $s _ { R }$ are the mean and sample standard deviation of the group’s final rewards, and $\delta = 1 0 ^ { - 6 }$ ensures numerical stability. All-equal rewards give exactly zero advantages.

For an ordinary response $y _ { i } .$ , let $y _ { i , j }$ be token $j$ and define

$$
\omega _ { i , j } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , j } \mid x _ { i } , y _ { i , < j } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , j } \mid x _ { i } , y _ { i , < j } ) } .
$$

Let O index the original responses that remain in their groups, and let $T$ be the total number of active response tokens in the final batch. Their loss contribution uses the asymmetric clipping interval [0.8, 1.3],

$$
\mathcal { L } _ { \mathrm { o r i g i n a l } } ( \theta ) = - \frac { 1 } { T } \sum _ { i \in \mathcal { O } } \sum _ { j = 1 } ^ { | y _ { i } | } \operatorname* { m i n } \big \{ \omega _ { i , j } ( \theta ) A _ { i } \mathrm { , c l i p } ( \omega _ { i , j } ( \theta ) , 0 . 8 , 1 . 3 ) A _ { i } \big \} .\tag{29}
$$

Inserted trajectories use the stopped current probability in the ratio denominator, as derived in Appendix D. With unit token weights over each complete inserted trajectory, this gives Equation (13). Loss-active tokens include supplied prefixes and termination tokens when present, and both contributions share the same batch token denominator.

## E.4 TRAINING CONFIGURATIONS AND QUESTION PROMPT

The GRPO, SAGE, BREAD, LUFFY, and ARG runs share the optimizer and student rollout settings in Table 4. The resampling and weighting ablations use the same settings. SFT learns from the offline references. For OPSD, we follow Zhao et al. (2026) in using one student response per question, a training generation limit of 1024 tokens, and temperature 1.1. Table 4 specifies the remaining settings used in our experiments. All listed runs use 32 questions per update and one optimization epoch per collected batch. Their data and parameter-initialization seeds are zero, and the learning rate is constant after warmup.

<table><tr><td>Setting</td><td>GRPO-based runs</td><td>SFT</td><td>OPSD</td></tr><tr><td>Learning rate</td><td>10⁻⁶</td><td>10⁻⁶</td><td> $5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Warmup updates</td><td>10</td><td>10</td><td>0</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0</td></tr><tr><td>Gradient norm clipping</td><td>1</td><td>1</td><td>0.1</td></tr><tr><td>Responses per question</td><td>8</td><td>Reference</td><td>1</td></tr><tr><td>Response length limit</td><td>8192</td><td>8192</td><td>1024</td></tr><tr><td>Generation temperature</td><td>1</td><td>N/A</td><td>1.1</td></tr><tr><td>Generation top-p</td><td>1</td><td>N/A</td><td>0.95</td></tr><tr><td>Generation top-k</td><td>Unrestricted</td><td>N/A</td><td>20</td></tr></table>

Table 4. Recorded configurations for both model sizes. GRPO-based runs here comprise GRPO, SAGE, BREAD, LUFFY, ARG, resampling, and the weighting ablation. The reference-insertion ablation is specified by its replacement rule in Section 5.3.

The ordinary student question prompt and the evaluation prompt contain a single user message with the question, a blank line, and the instruction below. They use no system message or fewshot example. Auxiliary teacher or reference conditioning is specific to each training method. The

Qwen3 chat template is applied with thinking mode disabled, which appends an empty thinking block before response generation. The rendered question prompt is shown schematically below.

<|im\_start|>user   
<question>   
Please reason step by step, and put your final answer within \boxed{}.   
<|im\_end|>   
<|im\_start|>assistant   
<think>   
</think>

Here <question> denotes the question text, and the line break before <|im end|> is added for display. For ARG generation, the selected reference prefix supplies the beginning of the candidate trajectory. Evaluation supplies no reference prefix.

## E.5 CHECKPOINT AND EVALUATION RECORDS

Table 5 lists the reported checkpoints. GRPO, SAGE, BREAD, LUFFY, ARG, resampling, and the weighting ablations use the fixed budget of 200 updates. Reference insertion also reports its step-200 checkpoint. For SFT and OPSD, we select the reported checkpoints by comparing mean pass@12 and avg@12 on the five evaluation benchmarks.

<table><tr><td>Method</td><td>4B checkpoint</td><td>8B checkpoint</td></tr><tr><td>Base</td><td>0</td><td>0</td></tr><tr><td>SFT</td><td>25</td><td>25</td></tr><tr><td>OPSD</td><td>60</td><td>100</td></tr><tr><td>GRPO, SAGE, BREAD, LUFFY, ARG</td><td>200</td><td>200</td></tr><tr><td>Resampling</td><td>200</td><td>N/A</td></tr><tr><td>w/o weighting</td><td>200</td><td>200</td></tr><tr><td>Reference insertion</td><td>200</td><td>N/A</td></tr></table>

Table 5. Reported checkpoint steps for both model sizes. Base uses the released parameters. SFT and OPSD use benchmark-selected checkpoints, and the remaining trained entries report step 200.

All 20 reported model and method entries use the same 150 evaluation questions, comprising 30 questions from each of the five benchmarks. The benchmark files, answer extractor, and verifier implementation match across entries. Table 6 gives the common decoding settings. Each question contributes 12 scored responses.

<table><tr><td>Setting</td><td>Evaluation value</td></tr><tr><td>Temperature</td><td>1</td></tr><tr><td>Top-p</td><td>0.95</td></tr><tr><td>Top-k</td><td>20</td></tr><tr><td>Maximum new tokens</td><td>16,384</td></tr><tr><td>Maximum context length</td><td>32,768</td></tr><tr><td>Scored responses per question</td><td>12</td></tr><tr><td>Random seed</td><td>1234</td></tr><tr><td>Parameter precision</td><td>bfloat16</td></tr><tr><td>Generation engine</td><td>vLLM 0.11.0</td></tr><tr><td>Thinking mode</td><td>Disabled</td></tr></table>

Table 6. Evaluation configuration shared by all reported entries.

The scorer extracts the final balanced boxed answer. A response without a boxed answer is incorrect. Extracted answers are checked with Math-Verify 0.9.0 and a normalized string comparison. For each question, pass@12 records whether at least one of its 12 responses is correct, while avg@12 is the fraction of those responses that are correct. Both metrics are averaged over questions and reported as percentages. Since the benchmarks have equal sizes, the mean over benchmarks also equals the mean over all 150 questions.

## F ADDITIONAL MAIN RESULTS

Table 7 reports avg@12 for the same checkpoints and the same 12 scored responses per question as Table 1, following the scoring protocol in Appendix E.5.
<table><tr><td>Model</td><td>Method</td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>HMMT_Feb</td><td>HMMT_Nov</td><td>Avg.</td></tr><tr><td rowspan="7">Qwen3-4B</td><td>Base</td><td>22.78</td><td>22.22</td><td>13.33</td><td>11.94</td><td>9.17</td><td>15.89</td></tr><tr><td>SFT</td><td>19.44</td><td>20.56</td><td>16.94</td><td>9.72</td><td>9.17</td><td>15.17</td></tr><tr><td>GRPO</td><td>47.22</td><td>42.22</td><td>46.67</td><td>26.94</td><td>29.17</td><td>38.44</td></tr><tr><td>OPSD</td><td>22.22</td><td>18.89</td><td>20.83</td><td>14.44</td><td>13.06</td><td>17.89</td></tr><tr><td>SAGE</td><td>44.17</td><td>39.17</td><td>40.83</td><td>25.00</td><td>28.06</td><td>35.44</td></tr><tr><td>BREAD</td><td>56.39</td><td>47.78</td><td>47.78</td><td>29.17</td><td>35.56</td><td>43.33</td></tr><tr><td>LUFFY</td><td>50.28</td><td>43.61</td><td>41.67</td><td>25.00</td><td>33.06</td><td>38.72</td></tr><tr><td rowspan="8">Qwen3-8B</td><td>ARG (ours)</td><td>52.50</td><td>46.67</td><td>46.39</td><td>26.94</td><td>33.06</td><td>41.11</td></tr><tr><td>Base</td><td>25.28</td><td>21.67</td><td>18.61</td><td>9.72</td><td>9.44</td><td>16.94</td></tr><tr><td>SFT</td><td>33.06</td><td>27.78</td><td>28.06</td><td>12.22</td><td>17.50</td><td>23.72</td></tr><tr><td>GRPO</td><td>48.89</td><td>35.83</td><td>36.67</td><td>22.22</td><td>30.56</td><td>34.83</td></tr><tr><td>OPSD</td><td>33.89</td><td>29.44</td><td>26.67</td><td>14.17</td><td>19.44</td><td>24.72</td></tr><tr><td>SAGE</td><td>54.72</td><td>40.00</td><td>44.17</td><td>27.22</td><td>35.28</td><td>40.28</td></tr><tr><td>BREAD</td><td>60.56</td><td>42.50</td><td>43.06</td><td>27.78</td><td>39.44</td><td>42.67</td></tr><tr><td>LUFFY ARG (ours)</td><td>50.83 55.28</td><td>42.22 46.94</td><td>43.89 46.39</td><td>25.28 27.22</td><td>31.94 35.56</td><td>38.83 42.28</td></tr></table>

Table 7. Main experimental results in avg@12 percentages. Avg. is the mean over the five benchmarks. Within each model, bold and underline mark the best and second-best values in each column.

ARG improves aggregate avg@12 over GRPO by 2.67 and 7.45 percentage points on Qwen3-4B and Qwen3-8B, respectively, and achieves the second-highest aggregate avg@12 on both models. The difference between the two metrics is consistent with where ARG intervenes. Specifically, ARG modifies the GRPO update only for all-failure groups, where all G responses to a training question are incorrect, and leaves the update for all other questions unchanged. Therefore, the additional learning signal concentrates on questions that the model currently fails to solve. When a question changes from unsolved to solved by a few of the 12 responses, pass@12 counts the entire question, whereas avg@12 counts only the fraction of correct responses. As a result, the effect of ARG appears most clearly in pass@12, which measures the fraction of questions that the model solves within the sampling budget.

## G ADDITIONAL ABLATION RESULTS

We provide benchmark-level results, then examine how the ablations change the accepted trajectories and the prefix selector during training. Reference insertion and resampling use Qwen3-4B, while the surprisal-weighting comparison covers both model sizes.

## G.1 EVALUATION RESULTS

Tables 8 and 9 report pass@12 and avg@12 separately for all five benchmarks. All entries use the checkpoint after 200 training steps and the same evaluation protocol as the main results. Each question contributes 12 scored responses, as specified in Appendix E.5.

For the ablations in Table 9, ARG has higher average sampled accuracy on every benchmark.

## G.2 ACCEPTED CANDIDATE TRAJECTORIES DURING TRAINING

To measure how often continuations supply the trajectories used for training, we count all-failure groups that accept at least one candidate trajectory. Reference fallback is excluded from this count.

<table><tr><td>Model</td><td>Method</td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>HMMT_Feb</td><td>HMMT_Nov</td><td>Avg.</td></tr><tr><td rowspan="4">Qwen3-4B</td><td>Reference insertion</td><td>70.00</td><td>60.00</td><td>63.33</td><td>43.33</td><td>46.67</td><td>56.67</td></tr><tr><td>Resampling</td><td>76.67</td><td>63.33</td><td>63.33</td><td>43.33</td><td>53.33</td><td>60.00</td></tr><tr><td>w/o weighting</td><td>76.67</td><td>70.00</td><td>66.67</td><td>50.00</td><td>53.33</td><td>63.33</td></tr><tr><td>ARG</td><td>80.00</td><td>73.33</td><td>73.33</td><td>43.33</td><td>66.67</td><td>67.33</td></tr><tr><td rowspan="2">Qwen3-8B</td><td>w/o weighting</td><td>83.33</td><td>70.00</td><td>66.67</td><td>50.00</td><td>50.00</td><td>64.00</td></tr><tr><td>ARG</td><td>83.33</td><td>73.33</td><td>73.33</td><td>56.67</td><td>63.33</td><td>70.00</td></tr></table>

Table 8. Benchmark-level ablation results in pass@12 percentages. Avg. is the mean over benchmarks. Here w/o weighting denotes ARG without surprisal weighting.

<table><tr><td>Model</td><td>Method</td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>HMMT_Feb</td><td>HMMT_Nov</td><td>Avg.</td></tr><tr><td rowspan="5">Qwen3-4B</td><td>Reference insertion</td><td>36.67</td><td>34.17</td><td>34.17</td><td>17.78</td><td>19.44</td><td>28.44</td></tr><tr><td>Resampling</td><td>44.44</td><td>35.56</td><td>40.00</td><td>21.67</td><td>28.33</td><td>34.00</td></tr><tr><td>w/o weighting</td><td>43.61</td><td>40.83</td><td>39.17</td><td>21.39</td><td>28.33</td><td>34.67</td></tr><tr><td>ARG</td><td>52.50</td><td>46.67</td><td>46.39</td><td>26.94</td><td>33.06</td><td>41.11</td></tr><tr><td>w/o weighting</td><td>45.28</td><td>33.89</td><td>38.33</td><td>18.33</td><td>23.61</td><td>31.89</td></tr><tr><td>Qwen3-8B</td><td>ARG</td><td>55.28</td><td>46.94</td><td>46.39</td><td>27.22</td><td>35.56</td><td>42.28</td></tr></table>

Table 9. Benchmark-level ablation results in avg@12 percentages, computed from the same scored responses as Table 8. Avg. is the mean over benchmarks.

Figure 2 reports the proportion in consecutive windows of 50 training steps, using the total group counts within each window. Each curve describes the all-failure questions encountered in its own training run.

![](images/d171361a1043bcbcebf8e85942a510ccd08c11610e3779678317b8c611e70fce.jpg)  
(a) Qwen3-4B resampling

![](images/f69812ddc55d6f8d02754065da7228b476f08a03b44bc5670d324f2b2eec08f5.jpg)  
(b) Qwen3-8B surprisal weighting  
Figure 2. Proportion of all-failure groups that accept candidate trajectories. A point at step s aggregates steps s − 49 through s. Reference fallback is excluded. The curves come from their respective training runs.

On Qwen3-4B, the proportion increases from 59.4% to 71.8% for ARG across the first and last windows, while resampling decreases from 37.6% to 26.4%. Over the full runs, ARG accepts candidate trajectories in 318 of 482 all-failure groups and uses reference fallback in the remaining 164. Resampling accepts candidate trajectories in 145 of 446 all-failure groups, leaving the remaining 301 unchanged. Resampling retains all correct candidate trajectories. Both methods use eight continuations per all-failure group, and ARG also uses reference guidance and fallback.

The weighting ablation gives a different comparison. On Qwen3-8B, it accepts candidate trajectories in 64.8% of all-failure groups, compared with 60.2% for ARG, despite its lower evaluation scores. On Qwen3-4B, these proportions are similar at 65.9% and 66.0%. Thus, the proportion of groups that accept candidate trajectories alone does not explain the evaluation results.

## G.3 SURPRISAL WEIGHTING AND PREFIX SELECTION

The weighting ablation uses the successful replacement indicator as feedback, while retaining the same guidance levels, prefix construction, selector hyperparameters, and acceptance rule. Figure 3 compares the resulting selection probabilities. On Qwen3-4B, both selectors favor $\alpha = 1 / 4$ for most of training. On Qwen3-8B, the unweighted selector favors $\alpha = 1 / 4$ throughout training, whereas the probabilities under ARG change more over time. The final probabilities under ARG are 49.0% for $\alpha = 1 / 2$ and 34.6% for $\alpha = 1 \bar { / } 4$ , compared with 6.8% and 86.4% without weighting.

![](images/4f05eaa87c8c38d372e5b8af6c5034089f99f78771adc39acfbe5efd7ff9611c.jpg)  
(a) Qwen3-4B ARG

![](images/6aa333b609817538068de3512dd852c0b948f9c37aea52060bb789d7d85e4d82.jpg)

![](images/dff96a1d7fb6ea59433be82c705937fb31d94845a5d9a090b70e53096915869b.jpg)  
(c) Qwen3-8B ARG

(b) Qwen3-4B w/o weighting  
![](images/20d0869bce011e2a19345e46d4c02724b2670c8b0950f74155ea74809e3e483f.jpg)  
(d) Qwen3-8B w/o weighting  
Figure 3. Selection probabilities of the prefix selector with and without surprisal weighting. Curves show the state after each training step without smoothing. Steps with no selector update retain the previous probabilities. The level $\alpha = 1$ uses an empty prefix, while smaller levels leave less of the reference surprisal to the continuation.

## H TRAINING DYNAMICS

We supplement Section 5.4 with the corresponding analysis on Qwen3-4B and the sources of accepted trajectories for both models. Comparisons with the ablated methods appear in Appendix G.

## H.1 SUCCESSFUL REPLACEMENT, FEEDBACK, AND PREFIX SELECTION

Figure 4 reports successful replacement rates, mean selector feedback, and prefix selection on Qwen3-4B. As in the $\mathrm { Q w e n } 3 { - } 8 \mathrm { B }$ plots, each point in the first two panels uses all continuations sampled at that level within a window of 50 training steps. Successful replacement requires a correct candidate trajectory different from the reference. We compute $W _ { t }$ with the actual $S _ { t } / S _ { 0 }$ after prefix-boundary adjustment, including zero feedback from failures and exact reference copies. The third panel shows the selection probabilities after each training step without smoothing.

The level $\alpha = 1 / 4$ has the highest successful replacement rate in each window. Weighting reduces its advantage over $\alpha = 1 / 2$ , and their mean-feedback ordering varies across windows. Over the full run, the mean feedback is $0 . 0 7 5 4$ for $\alpha = 1 / 4$ and 0.0588 for $\alpha = 1 / 2$ . These observations accompany the persistent preference for $\alpha = 1 / 4$ shown in Figure 4(c).

![](images/b393aa07c193f24a58282ce6e3ff5ddc08bb82ad4a83fa288eff38225d609e83.jpg)  
(a) Successful replacement

![](images/3a6406708073a05d68f71c0a1dbde4b3a4b4a0dcfa59e5978e015adc8df39c6f.jpg)  
(b) Selector feedback

![](images/0b67ae60148cca452374f2552d7eacd8397bfcf89121f1979c971d5b89818015.jpg)  
(c) Prefix selection  
Figure 4. Training dynamics of a Qwen3-4B ARG run. In panels (a) and (b), a point at step s averages the continuations at each level from steps s − 49 through s. Panel (c) shows the selection probabilities after each training step without smoothing. The colors and corresponding axis ranges match Figure 1.

The windowed statistics summarize continuations collected as the policy, the all-failure questions, and the selection probabilities change. Since the selector updates sequentially from these continuations, its state reflects their order and the preceding updates. Steps with no selector update retain the previous probabilities. This gives a view of the feedback observed during training alongside the resulting selection history.

## H.2 SOURCES OF ACCEPTED TRAJECTORIES

Figure 5 records which trajectories enter policy learning. We call a candidate trajectory guided if it comes from a nonempty prefix and unguided if it comes from the empty prefix. Each all-failure group belongs to one of four categories, according to whether it accepts only guided candidate trajectories, only unguided ones, both types, or the reference as fallback. Thus, the figure describes groups, with each group counted once even when it accepts multiple trajectories.

![](images/2b3cb22f519e56781b90405c7ca88c4ccfe983f2ba265ad858f4fa8ebbb040a5.jpg)  
(a) Qwen3-4B

![](images/2f3523e9a4ec8046d0b7f06085ce7fd5e455bd407f5d1ee8ffde5018908c6d8c.jpg)  
(b) Qwen3-8B  
Figure 5. Sources of accepted trajectories among all-failure groups in consecutive windows of 50 training steps. The mixed category contains groups that accept both guided and unguided candidate trajectories. Fallback supplies the complete reference when no candidate trajectory is accepted. Counts below the windows give the number of all-failure groups.

Candidate trajectories are accepted in 66.0% and 60.2% of all-failure groups for Qwen3-4B and Qwen3-8B, respectively. The remaining groups use reference fallback. Most groups that accept candidate trajectories use guided ones, while a smaller fraction accept unguided ones. Fallback remains present throughout training, and its proportion varies across windows as the all-failure questions change. These records complement the selection curves by showing the trajectories used for the policy updates.