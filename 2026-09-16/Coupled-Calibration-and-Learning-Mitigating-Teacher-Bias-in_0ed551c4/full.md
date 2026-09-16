# Coupled Calibration and Learning: Mitigating Teacher Bias in LLM Distillation without Target-Domain Reward Feedback

Haichen Hu<sup>♮</sup>

Yuheng Zhang<sup>†</sup>

David Simchi-Levi<sup>∗</sup>

MIT

UIUC

MIT

## Abstract

Large language model (LLM) distillation aims to transfer the capabilities of a powerful teacher to a smaller student. Direct imitation, however, can also transfer the teacher’s systematic bias and errors. This challenge is particularly pronounced under covariate shift, when the teacher’s reliability on target questions is uncertain and targetdomain reward feedback is unavailable. We propose Coupled Calibration and Learning (CCL), an LLM distillation algorithm that couples teacher calibration with student updates through token-level branching, using reward feedback only on source questions. Each iteration calibrates the teacher using source feedback and then uses the calibrated teacher to train the student on target questions. The updated student, in turn, informs subsequent calibration. In an autoregressive policy framework, we prove that the output student’s expected average Kullback-Leibler divergence to the oracle student converges to zero at a polynomial rate in the number of iterations. The oracle maximizes the true reference-regularized target reward within the student class, which need not represent the unrestricted optimal policy. Our analysis quantifies the progress of projected student gradient updates while controlling the error in teacher calibration. We further establish a separation from regularized direct matching: its error relative to the oracle student can remain bounded away from zero even when the teacher achieves higher regularized target reward than every student policy. These results demonstrate that LLM distillation can overcome persistent teacher bias and recover the optimal student through coupled calibration and learning, without target-domain reward feedback.

## 1 Introduction

Large language models (LLMs) have become a central component of modern artificial intelligence, with applications ranging from language understanding and content generation to mathematical reasoning and program synthesis (OpenAI, 2023; Gemini Team, 2023; Grattafiori et al., 2024; Qwen Team, 2024; DeepSeek-AI, 2025). As these models increasingly supply predictions, decisions, and training data for other systems, understanding their statistical behavior has become an important research problem.

Statistical learning theory provides a principled framework for understanding the capabilities and limitations of modern AI. It studies when overparameterized predictors generalize, how model predictions can support valid statistical inference, and how to evaluate black-box prediction procedures. For language models, theoretical analyzes also examine how pretraining benefits downstream tasks, how transformers learn from examples in context, and how preference feedback guides policy learning. These questions have motivated progress in generalization theory, statistical inference, and the analysis of language-model training (Bartlett et al., 2020; Angelopoulos et al., 2023; Saunshi et al., 2021; Bai et al., 2023; Kim et al., 2024a; Ye et al., 2024; Wainwright, 2025; Hu and Simchi-Levi, 2025b,a, 2026). Within this broader program, a central challenge is to explain when information from an existing model can support the reliable training of another model.

Knowledge distillation is a prominent approach to this challenge. A larger or more capable teacher supplies output probabilities or generated responses to train a student, often with substantially lower deployment costs. This idea underlies both classical model compression and recent methods for transferring language and reasoning capabilities to smaller models (Hinton et al., 2015; Gu et al., 2024; DeepSeek-AI, 2025). The teacher provides a rich source of synthetic supervision, making distillation particularly attractive when collecting task-specific demonstrations is dificult.

A central dificulty, however, is the propagation of teacher-bias. Matching the teacher’s predictions can reproduce its systematic errors as well as its useful knowledge. For example, image-classification experiments show that distillation can amplify teacher errors on dificult classes even when average student accuracy improves (Lukasik et al., 2022). This concern is especially relevant under covariate shift, where the target questions difer from the source questions on which reliable supervision is available. Strong source performance does not by itself certify the teacher’s accuracy on the target questions. Related experiments with LLMs show substantial accuracy losses when in-context demonstrations and evaluation examples come from diferent topic domains (Roussinov et al., 2025). Nevertheless, the teacher may retain useful knowledge acquired from source data. The statistical challenge is therefore to exploit that information while correcting the errors that direct imitation can propagate (Yamamoto and Wainwright, 2026).

The absence of reliable target feedback makes this problem more dificult. A reward model or verifier developed for one collection of questions need not remain reliable on another: multilingual evaluations, for example, find lower reward-model accuracy and inconsistent preferences across languages (Gureja et al., 2025). Constructing new feedback can also require substantial resources. Training mathematical process verifiers has involved extensive human step annotations (Lightman et al., 2024), while executable code evaluation relies on task-specific tests and execution infrastructure (Chen et al., 2021). These considerations motivate a setting in which trusted reward feedback is available on source questions, but only the questions themselves are available in the target dataset. The learner must then use source feedback to address target teacher bias, without evaluating target answers against their true rewards. This leads to the following question:

Can we distill a near optimal student from a biased teacher under covariate shift, using reward feedback only on source questions and none on the target questions?

Our contribution. We answer this question afirmatively by proposing Coupled Calibration and Learning (CCL), an LLM distillation algorithm that learns from a biased teacher using reward feedback only on source questions. In an autoregressive policy framework, our benchmark is the oracle student: the policy within the student class that maximizes the true target reward with KL regularization to a pre-trained reference policy.

Our algorithm couples teacher calibration with student updates in an iterative procedure. Each iteration calibrates the teacher using source reward feedback and then uses the calibrated teacher to train the student on target questions. The updated student, in turn, informs the next calibration step so that teacher calibration and student learning proceed together throughout training.

We prove that the expected average KL divergence from the student returned by CCL to the oracle student converges to zero at a polynomial rate in the number of iterations. The analysis uses student gradient progress near the oracle and exploration outside a fixed near-optimal region, while controlling calibration and finite-rollout errors. We further establish a separation: direct distillation can retain a strictly positive KL divergence between the oracle student and the trained student policy even when the teacher achieves a higher true regularized reward than every policy in the student class. Thus, our method can eliminate a persistent error of direct imitation and recover the optimal student without target-domain reward feedback.

Paper structure. Our paper is organized with the following structure. Section 3 introduces the distillation problem, the autoregressive policy model, and the oracle student benchmark. Section 4 presents our algorithm and explains how it couples teacher calibration with student updates. Then, Section 5 establishes convergence to the optimal oracle student and outlines the main steps of the analysis. Section 6 establishes a separation from regularized direct teacher matching to show that our bound is strictly better than direct distillation.

Notation. We write $\mathbb { E } _ { X } [ \cdot ]$ and $\mathbb { P } _ { X } ( \cdot )$ for expectation and probability with respect to the random variable X, respectively. Conditional expectation and probability are denoted by $\mathbb { E } [ \cdot \mid { \mathcal { F } } ]$ and $\mathbb { P } ( \cdot \mid \mathcal { F } )$ for a sigma-algebra ${ \mathcal F } .$ We use $\sigma ( X _ { 1 } , \ldots , X _ { k } )$ for the sigma-algebra generated by $( X _ { i } ) _ { i = 1 } ^ { k }$ , and ${ \mathcal { F } } \vee { \mathcal { G } }$ for the smallest sigma-algebra containing both $\mathcal { F }$ and ${ \mathcal { G } } .$ For a vector v, $\lVert \boldsymbol { v } \rVert _ { 2 }$ denotes its Euclidean norm; for a matrix $A , \ \| A \| _ { \mathrm { o p } }$ denotes its induced Euclidean operator norm. For probability distributions $P , Q$ on a common finite set ${ \mathcal { Z } } ,$ their Kullback-Leibler divergence is denoted by $\begin{array} { r } { \mathrm { K L } ( P \| Q ) = \sum _ { z \in \mathcal { Z } } P ( z ) \log \left( P ( z ) / Q ( z ) \right) } \end{array}$ , where terms with $P ( z ) = 0$ are zero, and the divergence is $+ \infty$ if $P ( z ) > 0 = Q ( z )$ for some $z .$ . For a nonempty closed convex set $\begin{array} { r } { \mathcal { C } , \mathrm { P r o j } _ { \mathcal { C } } \left( \boldsymbol { v } \right) : = \arg \operatorname* { m i n } _ { \boldsymbol { u } \in \mathcal { C } } \| \boldsymbol { u } - \boldsymbol { v } \| _ { 2 } ^ { 2 } } \end{array}$ denotes Euclidean projection. The notation $\operatorname { U n i f } ( { \mathcal { C } } )$ denotes the uniform distribution over $\mathcal { C } .$ . For two sequences $( a _ { k } ) _ { k \geq 1 }$ and $( b _ { k } ) _ { k \geq 1 }$ with $b _ { k } > 0$ , we write $a _ { k } = O ( b _ { k } )$ if there exist constants $C > 0$ and $k _ { 0 }$ such that $| a _ { k } | \le C b _ { k }$ for all $k \geq k _ { 0 }$ , and $a _ { k } = o ( b _ { k } ) { \mathrm { ~ i f ~ } } | a _ { k } | / b _ { k } \to 0 { \mathrm { ~ a s ~ } } k \to \infty$

## 2 Related Work

Statistical theory of distillation and imitation learning. Statistical analyzes of distillation study the benefits of teacher-generated supervision and the propagation of teacher error. Menon et al. (2021) explain the bias–variance tradeof of soft labels, while Ildiz et al. (2025) characterize high-dimensional distillation risk under model and covariate shift. Xie et al. (2026) identify settings where a risk-minimizing teacher preserves the restricted student’s optimum and improves the statistical eficiency of averaged SGD. To correct imperfect supervision, Dao et al. (2021) develop cross-fitting and loss corrections, and Iliopoulos et al. (2022) analyze student-dependent reweighting of noisy pseudo-labels. Relatedly, Xia and Wainwright (2024) construct pseudo-responses using training-only helper covariates and obtain prediction bounds combining an oracle rate with surrogate error. For sequential prediction, Ross et al. (2011) establish a no-regret foundation for learning from expert feedback at learner-visited states, while Czarnecki et al. (2019) analyze policy-distillation updates and convergence in tabular settings. Foster et al. (2024) show that, under realizability, online expert access need not improve worst-case statistical complexity over ofline behavior cloning with logarithmic loss. With noisy expert feedback, Sriraman et al. (2026) establish an ofline on-policy separation for learning a realizable clean expert. Beyond realizability, Zhang et al. (2026) study how student misspecification and alignment between expert scores and rewards afect the benefits of online imitation, and give finite-sample guarantees using base-policy sampling. Closest to our motivation, Yamamoto and Wainwright (2026) study bias propagation under source-target covariate shift. Their method refits teachers to student residuals and achieves a provable separation from direct soft matching. Our work instead couples source-reward-based teacher calibration with target-side LLM distillation. We establish convergence to the regularized oracle within the student class and separation from regularized direct matching, without target reward feedback or requiring the student to represent the unrestricted optimal policy.

Reinforcement learning and LLM post-training. Reinforcement learning plays an important role in modern deep learning, with a substantial theoretical literature on exploration, policy optimization, and learning with function approximation (Jiang et al., 2017; Jin et al., 2020; Agarwal et al., 2021; Xie et al., 2021, 2023; Zhang et al., 2023; Qian et al., 2024; Hu et al., 2026). For preference-based learning, Zhu et al. (2023) connect reward estimation to policy performance and establish guarantees for pessimistic learning. Xiong et al. (2024) develop ofline, online, and hybrid algorithms for KL-regularized preference learning with finite-sample guarantees. Xie et al. (2025) in troduce exploration bonuses for provably sample-eficient preference optimization, while Zhao et al. (2025) characterize how KL regularization and reference-policy coverage afect sample complexity. For LLM post-training, Chen et al. (2026) study how pre-training provides response coverage for downstream improvement, and Foster et al. (2025) distinguish the statistical and computational roles of base-model coverage. Huang et al. (2025) analyze self-improvement through sharpening, where training amortizes the selection of high-likelihood responses. Related inference-time analyses characterize Best-of-N through win-rate guarantees (Sriraman and Block, 2026) and use pessimistic scoring to mitigate reward hacking (Yu et al., 2026). For outcome-supervised learning, Jia et al. (2025) connect outcome feedback to process-level learning, Yuan et al. (2025) develop trajectory Bellman residual minimization for KL-regularized policy learning, and Chen et al. (2025) establish sample-complexity guarantees for outcome-based online RL. Kim et al. (2026) analyze coverage improvement and convergence in on-policy preference learning and reward distillation, explicitly accounting for reward-model error in the latter. These works study how feedback and coverage support policy improvement and response selection. Our work addresses biased-teacher LLM distillation under source-target covariate shift. We couple source-reward-based teacher calibration with target-side student learning and prove convergence in average target KL to the regularized oracle within the student class, together with separation from regularized direct matching, without target reward feedback.

Transfer learning under covariate shift. Transfer-learning theory studies how source supervision supports prediction on a diferent target distribution. Under covariate shift, linear and kernel regression analyzes quantify this transfer through source-target covariance geometry and distributional overlap (Lei et al., 2021; Ma et al., 2023). In well-specified parametric models, Ge et al. (2024) establish minimax guarantees for source-only maximum likelihood estimation, with transfer dificulty governed by source and target Fisher information rather than a bounded density ratio. Pseudo-labeling methods further use unlabeled target covariates to select source-trained estimators in kernel ridge regression and kernel generalized linear models (Wang, 2026; Weill and Wang, 2026). Beyond covariate shift, Xia and Klusowski (2026) study oversampling under label shift, separating balanced-data risk from the cost of estimating the minority-class distribution. These results characterize prediction under distribution shift, but do not study LLM distillation without target feedback. Under shared policy realizability and joint source identification, our analysis controls the student’s error on fixed target questions using reward feedback confined to source questions.

## 3 Model setup

In this section, we formulate LLM distillation for post-training under a fixed source–target design. We model autoregressive generation as a finite-horizon, token-level Markov decision process

$$
\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , R , H ) ,
$$

where $\boldsymbol { \mathcal { S } }$ is the prefix-state space, A is a finite token vocabulary, $P$ is the deterministic transition kernel, R is the terminal answer reward, and $H \geq 1$ is the generation horizon. Denote $\mathcal { X }$ as a context space representing the set of potential questions. Each $x \in \mathcal { X }$ represents a question presented to the LLM, including its instructions and any accompanying context; we refer to this complete input as a prompt. Generation starts at $s _ { 1 } = ( x , \alpha )$ . At step $h ,$ the state $s _ { h } = \left( x , a _ { 1 : h - 1 } \right)$ records the question and all previously generated answer tokens. The LLM acts as a policy π: it selects a legal token $a _ { h } \sim \pi ( \cdot \mid s _ { h } )$ and appends it to the prefix, so

$$
P ( s ^ { \prime } \mid s _ { h } , a _ { h } ) = \mathbf 1 \{ s ^ { \prime } = ( x , a _ { 1 : h } ) \} , \ h = 1 , \ldots , H .
$$

The reward is evaluated on the completed answer and is specified below. We first make precise which tokens and answers are feasible.

Definition 3.1 (Legal tokens and feasible trajectories). Let A contain EOS and null. For $1 \leq h \leq$ $H ,$ define the legal-token set at $s _ { h } = \left( x , a _ { 1 : h - 1 } \right)$ by

$$
\mathcal { B } ( s _ { h } ) : = \left\{ \begin{array} { l l } { \mathcal { A } \backslash \{ \mathrm { n u 1 1 } \} , } & { \mathrm { E 0 S } \notin \{ a _ { 1 } , \dots , a _ { h - 1 } \} , } \\ { \{ \mathrm { n u 1 1 } \} , } & { \mathrm { E 0 S } \in \{ a _ { 1 } , \dots , a _ { h - 1 } \} . } \end{array} \right.
$$

The state space $\boldsymbol { \mathcal { S } }$ consists of the prefixes generated from $( x , \alpha ) , x \in \mathcal { X }$ , by repeatedly appending legal tokens, up to length H. States $s _ { H + 1 }$ are terminal, with $B ( s _ { H + 1 } ) : = \emptyset$ . For a reachable state $s _ { h }$ with $h \leq H$ , define

$$
\begin{array} { r } { \mathcal { A } ( s _ { h } ) : = \left\{ b _ { h : H } \in \mathcal { A } ^ { H - h + 1 } : b _ { k } \in \mathcal { B } ( ( x , a _ { 1 : h - 1 } , b _ { h : k - 1 } ) ) \mathrm { ~ f o r ~ } k = h , \dots , H \right\} . } \end{array}
$$

Here $\left( x , a _ { 1 : h - 1 } , b _ { h : k - 1 } \right)$ appends $b _ { h : k - 1 }$ to the existing prefix, with empty blocks omitted. Set

$$
\ A ( s _ { H + 1 } ) : = \{ \emptyset \} , \ A ( x ) : = A ( ( x , \emptyset ) ) .
$$

Thus $B ( s _ { h } )$ contains individual legal tokens, $\boldsymbol { A } ( s _ { h } )$ contains feasible remaining token sequences, and $\mathcal { A } ( x )$ contains full feasible answers.

Generation ends at EOS or horizon $H ,$ with null padding after EOS. This convention gives every answer length $H ,$ and feasibility is independent of reward. A policy assigns probability zero to

illegal tokens; on each nonterminal state, its probabilities over $B ( s _ { h } )$ sum to one. The deterministic transitions and successive token choices induce the full-answer law

$$
\pi ( a _ { 1 : H } \mid x ) = \prod _ { h = 1 } ^ { H } \pi ( a _ { h } \mid x , a _ { 1 : h - 1 } ) , \ a _ { 1 : H } \in A ( x ) .\tag{3.1}
$$

The same notation $\pi$ therefore describes both the token policy and its induced distribution over complete answers.

Post-training seeks to improve answer quality while retaining the behavior of a pretrained model. We study this task under covariate shift, with a fixed source dataset of training questions and a fixed target dataset of questions that the student is intended to answer:

$$
\mathcal { D } _ { \mathrm { s r c } } = \{ x _ { i } \} _ { i = 1 } ^ { n } , ~ \mathcal { D } _ { \mathrm { t a r } } = \{ \widetilde { x } _ { j } \} _ { j = 1 } ^ { m } , ~ n , m \geq 1 .
$$

Here $x _ { i }$ is the ith source question and $\widetilde { x } _ { j }$ is the jth target question, each including its associated context. Reward supervision is available for candidate answers to source questions, whereas the learning objective concerns the student’s answers to target questions. The two datasets may contain diferent mixtures of question types. We condition throughout on these fixed datasets; randomness comes from policy sampling and the training algorithm.

The terminal reward is a deterministic function

$$
R : \left\{ ( x , a _ { 1 : H } ) : x \in \mathcal { X } , \ a _ { 1 : H } \in \mathcal { A } ( x ) \} \longrightarrow [ 0 , 1 ] . \right.
$$

We only have access to an exact verifier on the source prompts: given $x _ { i }$ and any $a _ { 1 : H } \in \mathcal { A } ( x _ { i } )$ , it returns $R \big ( x _ { i } , a _ { 1 : H } \big )$ . Source supervision is therefore supplied by evaluations of candidate answers, and the dataset itself contains only prompts. Reward feedback is unavailable on the target prompts; $R ( \widetilde { x } _ { j } , a _ { 1 : H } )$ denotes their latent true answer quality and is never queried during training. This restriction reflects the dificulty of extending reliable reward evaluation to new questions. Developing a target-domain reward model can require expert-designed rubrics, labeled answers, and careful validation of the grading criteria. These requirements make target reward very costly to obtain.

Let $\pi _ { \mathrm { p r e } }$ be the frozen pretrained reference policy, with positive probability on every legal token. For $\lambda > 0$ , our target post-training objective is

$$
J _ { \lambda , m } ( \pi ) : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi ( \cdot | \widetilde { x } _ { j } ) } \left[ R ( \widetilde { x } _ { j } , a _ { 1 : H } ) - \lambda \sum _ { h = 1 } ^ { H } \log \frac { \pi ( a _ { h } \mid \widetilde { x } _ { j } , a _ { 1 : h - 1 } ) } { \pi _ { \mathrm { p r e } } ( a _ { h } \mid \widetilde { x } _ { j } , a _ { 1 : h - 1 } ) } \right] .\tag{3.2}
$$

The first term measures answer quality; the second penalizes deviation from the pretrained policy. By (3.1), the expected log-ratio sum equals the full-answer KL:

$$
\mathrm { K L } ( \pi ( \cdot \mid x ) \parallel \pi _ { \mathrm { p r e } } ( \cdot \mid x ) ) = \mathbb { E } _ { a _ { 1 : H } \sim \pi ( \cdot \mid x ) } \left[ \sum _ { h = 1 } ^ { H } \log \frac { \pi ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } { \pi _ { \mathrm { p r e } } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } \right] .
$$

Thus λ controls the tradeof between reward and proximity to the reference. The objective specifies the desired target behavior, although its reward term is unavailable during training. We address this information constraint through teacher distillation: source reward feedback calibrates an available teacher model, whose likelihoods then supervise the student on the target prompts.

To formalize the teacher, its calibration, and the student, we use linear-softmax policy classes. Fix

known feature maps $\phi : \mathcal { S } \times \mathcal { A }  \mathbb { R } ^ { D }$ and $\phi _ { \mathrm { s t u } } : \mathcal { S } \times \mathcal { A }  \mathbb { R } ^ { d }$ , where $D , d \geq 1$ , with

$$
\| \phi ( s , a ) \| _ { 2 } \leq 1 , \ \| \phi _ { \mathrm { s t u } } ( s , a ) \| _ { 2 } \leq 1 .
$$

The features may depend on the entire prefix, preserving the autoregressive dependence of the policy. For every nonterminal state s and legal token $a \in B ( s )$ , define

$$
\pi _ { w } ( a \mid s ) = \frac { \exp ( w ^ { \top } \phi ( s , a ) ) } { \sum _ { b \in B ( s ) } \exp ( w ^ { \top } \phi ( s , b ) ) } , \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) = \frac { \exp ( \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , a ) ) } { \sum _ { b \in B ( s ) } \exp ( \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) ) } .\tag{3.3}
$$

Both policies assign zero probability to illegal tokens. For $B > 0$ , the calibration parameter belongs to a nonempty compact convex set W, and the student parameter belongs to the closed ball Θ:

$$
W \subseteq \{ w \in \mathbb { R } ^ { D } : \| w \| _ { 2 } \leq B \} , \Theta : = \{ \theta \in \mathbb { R } ^ { d } : \| \theta \| _ { 2 } \leq B \} .
$$

The given teacher is $\pi _ { \mathrm { t e a } } ~ = ~ \pi _ { w _ { \mathrm { t e a } } }$ with $w _ { \mathrm { t e a } } ~ \in ~ W$ . Its proposal policy remains frozen, while calibration adjusts a separate parameter within W, initialized at $w _ { \mathrm { t e a } }$ . The teacher may be biased relative to the optimal target behavior. The reference $\pi _ { \mathrm { p r e } }$ need not belong to either softmax class, and we impose neither sparsity nor an upper bound relating D to H.

For the performance metric, our benchmark is the best post-trained policy within the student class. Specifically, we choose

$$
\theta _ { \lambda , m } ^ { \dag } \in \underset { \theta \in \Theta } { \operatorname { a r g m a x } } J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta } ) .
$$

$\theta _ { \lambda , m } ^ { \dagger }$ exists because Θ is compact, and the objective is continuous in θ. We call $\pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } }$ an oracle student: it optimizes the true target objective using the same class available to the learned student. For comparison, at each source or target prompt x, we define the unrestricted optimal policy by

$$
\pi _ { \lambda } ^ { \star } ( \cdot \mid x ) \in \operatorname * { a r g m a x } _ { \pi ( \cdot \mid x ) \in \Delta ( A ( x ) ) } \mathbb { E } _ { a _ { 1 : H } \sim \pi ( \cdot \mid x ) } \left[ R ( x , a _ { 1 : H } ) - \lambda \sum _ { h = 1 } ^ { H } \log \frac { \pi ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } { \pi _ { \mathrm { p r e } } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } \right] .\tag{3.4}
$$

Here $\Delta ( \mathcal { A } ( x ) )$ is the simplex of distributions over full feasible answers. Token conditionals are obtained from prefix marginals. Lemma B.1 establishes that this optimum is unique and assigns positive probability to every feasible answer. The student class may be unable to represent this unrestricted optimum.

We connect source reward information to target behavior through the following realizability assumption.

Assumption 3.2 (Realizability). There exists a single parameter $w _ { \lambda } ^ { \star } \in { \cal W }$ such that, at every legal nonterminal prefix state s of every source or target prompt,

$$
\pi _ { \lambda } ^ { \star } ( a \mid s ) = \pi _ { w _ { \lambda } ^ { \star } } ( a \mid s ) { \mathrm { ~ f o r ~ e v e r y ~ } } a \in { \mathcal { B } } ( s ) .
$$

This assumption provides the shared structure needed for transfer: source and target prompts use the same optimal calibration coeficients, evaluated through their respective prefix features. Realizability is imposed on the calibration class and allows the student class to remain misspecified. Our fixed-design analysis uses this shared structure rather than a density-ratio assumption.

For a calibrated parameter $w \in W$ , define the target distillation cost

$$
C _ { w } ( \boldsymbol \theta ) : = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } \big ( \pi _ { \mathrm { s t u } , \boldsymbol \theta } ( \cdot \mid \widetilde { x } _ { j } ) \mid \mid \pi _ { w } ( \cdot \mid \widetilde { x } _ { j } ) \big ) .\tag{3.5}
$$

Its integrand is computable from the student and calibrated-policy likelihoods, so the cost can be estimated using student rollouts on the fixed target prompts. Under realizability, Lemma B.1 gives

$$
\operatorname * { a r g m a x } _ { \theta \in \Theta } J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta } ) = \operatorname * { a r g m i n } _ { \theta \in \Theta } C _ { w _ { \lambda } ^ { \star } } ( \theta ) .
$$

This identity makes the role of calibration explicit: at the true calibration parameter, distillation targets exactly the regularized oracle student.

We impose the following uniqueness assumption on the oracle student.

Assumption 3.3. The function $C _ { w _ { \lambda } ^ { \star } }$ has a unique minimizer over Θ, argmin $\mathsf { \Omega } _ { \theta \in \Theta } C _ { w _ { \lambda } ^ { \star } } ( \theta ) = \{ \theta _ { \lambda , m } ^ { \dagger } \}$

This assumption uniquely determines the oracle parameter $\theta _ { \lambda , m } ^ { \dagger }$ and hence its induced answer law on every target question. Neither interiority nor a positive student Hessian is required. For a learned calibration $w _ { t }$ and student $\theta _ { t }$ , write $C _ { t } ( \theta ) : = C _ { w _ { t } } ( \theta )$ and define

$$
\begin{array} { r l } & { \qquad \varepsilon _ { t } : = C _ { t } ( \theta _ { t } ) - \underset { \theta \in \Theta } { \operatorname* { m i n } } C _ { t } ( \theta ) , } \\ & { \Delta _ { \lambda , m } ( \theta ) : = C _ { w _ { \lambda } ^ { \star } } ( \theta ) - C _ { w _ { \lambda } ^ { \star } } ( \theta _ { \lambda , m } ^ { \dagger } ) , } \\ & { \mathcal { K } _ { \lambda , m } ( \theta ) : = \cfrac { 1 } { m } \underset { j = 1 } { \overset { m } { \sum } } \mathrm { K L } \Big ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid \widetilde { x } _ { j } ) \Big \Vert \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \mid \widetilde { x } _ { j } ) \Big ) . } \end{array}\tag{3.6}
$$

The first quantity measures optimization error for the current calibrated objective; the second measures excess cost under the true calibration; and the third measures the average KL from the learned student to the oracle student. Our goal is to drive $\mathbb { E } [ \mathcal { K } _ { \lambda , m } ( \theta _ { T } ) ]$ to zero. This compares policies within the student class, allowing the minimum distillation cost itself to remain positive.

Finally, source comparisons must contain enough information to identify the shared calibration parameter. For a teacher-generated prefix $s _ { i , h } = \left( x _ { i } , a _ { i , 1 : h - 1 } \right)$ , compare its next token $a _ { i , h }$ with a legal alternative b. The diference $\phi ( s _ { i , h } , a _ { i , h } ) - \phi ( s _ { i , h } , b )$ determines the calibration direction observed in that comparison. For each $h = 1 , \ldots , H$ , define the source information matrix $G _ { h }$ by

$$
G _ { h } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { a _ { i , 1 : h } \sim \pi _ { \mathrm { t e a } } ( \cdot | x _ { i } ) } \left[ \frac { 1 } { | B ( s _ { i , h } ) | } \sum _ { b \in B ( s _ { i , h } ) } \left( \phi ( s _ { i , h } , a _ { i , h } ) - \phi ( s _ { i , h } , b ) \right) \times \left( \phi ( s _ { i , h } , a _ { i , h } ) - \phi ( s _ { i , h } , b ) \right) ^ { \top } \right] ,\tag{3.7}
$$

The expectation is over the teacher’s output $a _ { i , 1 : h }$ , and the inner average is uniform over legal tokens. We impose the following identification condition on their average across token positions.

Assumption 3.4. Define the joint source information matrix as $\begin{array} { r } { G _ { \mathrm { j o i n t } } : = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } G _ { h } } \end{array}$ . We assume that it is positive definite, i.e.,

$$
\mu _ { \mathrm { j o i n t } } : = \lambda _ { \mathrm { m i n } } ( G _ { \mathrm { j o i n t } } ) > 0 .
$$

This assumption requires comparisons across all token positions to identify every calibration direction; individual matrices $G _ { h }$ may be singular. At a padded state, the only legal token is null, so its feature diference and information contribution are zero.

## 4 Algorithm

In this section, we present the CCL algorithm and provide a detailed explanation of its steps. The algorithm couples two components: teacher calibration using source reward feedback, and student distillation using the calibrated teacher on target prompts. Calibration adjusts the teacher’s predictions toward the regularized optimal policy, while distillation uses these adjusted predictions to train the student. The student also participates in calibration by supplying alternatives to the teacher’s proposed tokens. These components therefore interact throughout training: calibration changes the student’s training objective, and the updated student changes the comparisons used for subsequent calibration.

The central mechanism for estimating the calibration update is token-level branching. We select a token position in a teacher-generated source answer and retain the preceding prefix. At that prefix, we pair the teacher’s next token with an alternative sampled from the current student. This creates a comparison between two token choices in the same context. We then select one of these choices using the reference policy, complete the selected branch with that policy, and evaluate the resulting answer using the source reward oracle. A reward-dependent acceptance rule turns this observation into a statistical signal for estimating the calibration update. Repeating this construction at diferent token positions gathers information about the calibration coeficients across the generation process.

The calibrated teacher subsequently provides supervision for student distillation on the target prompts. Under our model, the calibration learned from source comparisons also applies to target questions. The student update uses a batch of sampled answers and policy likelihoods to estimate its gradient and take one projected step. We compare this proposal with a randomly sampled student. After evaluating these two candidates, the selected student supplies token alternatives for the next calibration round. This coupled procedure transfers source reward information into target-side training through the calibrated teacher. The full pseudocode is given in Algorithm 1.

CCL (Algorithm 1) maintains two trainable parameters: $w _ { t }$ for the calibrated teacher $\pi _ { \mathrm { c a l } , t } = \pi _ { w _ { t } }$ and $\theta _ { t }$ for the student $\pi _ { \mathrm { s t u } , \theta _ { t } }$ . Line 1 initializes the calibration from the teacher. Each round first uses source rewards and a current-student alternative to update $w _ { t }$ , then uses the updated calibrated teacher to train and select the next student. The frozen $\pi _ { \mathrm { t e a } }$ supplies the source proposals, and π<sub>pre</sub> supplies the reference-weighted branch and completion.

To obtain information for the calibration update, Lines 3-7 construct a token-level comparison. At a randomly selected position in a teacher answer, we retain the teacher prefix and compare its next token with a current-student alternative:

$$
s _ { t } = ( x _ { i _ { t } } , a _ { t , 1 : h _ { t } - 1 } ) , \ c _ { t , 1 } = a _ { t , h _ { t } } , \ c _ { t , 0 } \sim \pi _ { \mathrm { s t u } , \theta _ { t } } ( \cdot \mid s _ { t } ) .
$$

This comparison provides information about the calibration coeficients because the softmax parameterization gives

$$
z _ { t } : = \phi ( s _ { t } , c _ { t , 1 } ) - \phi ( s _ { t } , c _ { t , 0 } ) , ~ \mathrm { l o g } \frac { \pi _ { w } ( c _ { t , 1 } \mid s _ { t } ) } { \pi _ { w } ( c _ { t , 0 } \mid s _ { t } ) } = z _ { t } ^ { \top } w .
$$

Thus, learning the optimal relative probabilities of these two tokens constrains the coeficients along $z _ { t } .$ . Sampling the branching position across all H steps collects the joint source information in (3.7).

The next step uses source reward feedback to obtain a label for this comparison. Lines 8-13 select

Algorithm 1 Coupled Calibration and Learning (CCL) via Token-Level Branching   
Require: Source and target datasets $\{ x _ { i } \} _ { i = 1 } ^ { n } , \{ \widetilde { x } _ { j } \} _ { j = 1 } ^ { m } ;$ source reward oracle R; teacher $\pi _ { \mathrm { t e a } } = \pi _ { w _ { \mathrm { t e a } } }$ and   
$\pi _ { \mathrm { p r e } } ;$ features $\phi , \phi _ { \mathrm { s t u } }$ and sets $W , \Theta ; w _ { \mathrm { t e a } } \in W , \bar { \theta } _ { 0 } \in \Theta ; \lambda , \gamma > 0 ,$ integer $T \geq 1 ;$ ; schedules (4.3).   
All draws use fresh randomness conditional on the preceding variables.   
1: w<sub>0</sub> $ w _ { \mathrm { t e a } } , \pi _ { \mathrm { c a l , 0 } }  \pi _ { w _ { 0 } } .$   
2: for $t = 0 , \ldots , \dot { T } - 1$ do   
Token-Level branching on source   
3: Draw independently $i _ { t } \sim \operatorname { U n i f } \{ 1 , \dots , n \}$ and $h _ { t } \sim \operatorname { U n i f } \{ 1 , \dots , H \}$   
4: Draw $a _ { t , 1 : H } \sim \pi _ { \mathrm { t e a } } ( \cdot \mid x _ { i _ { t } } )$ autoregressively.   
5: $s _ { t } \gets ( x _ { i _ { t } } , a _ { t , 1 : h _ { t } - 1 } ) , c _ { t , 1 } \gets a _ { t , h _ { t } } .$ ▷ Teacher prefix and token.   
6: Draw $c _ { t , 0 } \sim \pi _ { \mathrm { s t u } , \theta _ { t } } ( \cdot \mid s _ { t } ) .$ ▷ Student alternative.   
7: $z _ { t } \gets \phi \big ( s _ { t } , c _ { t , 1 } \big ) - \phi \big ( s _ { t } , c _ { t , 0 } \big )$   
Teacher calibration.   
8: $Y _ { t } \sim$ Bernoulli $\left( { \frac { \pi _ { \mathrm { p r e } } ( c _ { t , 1 } \mid s _ { t } ) } { \pi _ { \mathrm { p r e } } ( c _ { t , 1 } \mid s _ { t } ) + \pi _ { \mathrm { p r e } } ( c _ { t , 0 } \mid s _ { t } ) } } \right)$   
9: $a _ { t , 1 : h _ { t } - 1 } ^ { \mathrm { p r e } }  a _ { t , 1 : h _ { t } - 1 } , a _ { t , h _ { t } } ^ { \mathrm { p r e } }  c _ { t , Y _ { t } } .$ ▷ Construct the selected branch.   
10: Draw $a _ { t , h _ { t } + 1 : H } ^ { \mathrm { p r e } } \sim \pi _ { \mathrm { p r e } } ( \cdot \mid x _ { i _ { t } } , a _ { t , 1 : h _ { t } } ^ { \mathrm { p r e } } )$ autoregressively.   
11: $R _ { t } \gets R ( x _ { i _ { t } } , a _ { t , 1 : H } ^ { \mathrm { p r e } } )$ ▷ Exactly one source reward query.   
12: Draw $U _ { t } \sim \mathrm { U n i f } [ 0 , 1 ] .$   
13: $I _ { t } \gets \mathbf { 1 } \{ U _ { t } \leq \exp \bigl ( \bigl ( R _ { t } - 1 \bigr ) / \lambda \bigr ) \}$   
14: $g _ { t }  I _ { t } \dot { z } _ { t } [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } ] .$ ▷ One trial; rejection gives $g _ { t } = 0 .$   
15: $w _ { t + 1 }  \operatorname { P r o j } _ { W } ( w _ { t } - \eta _ { t } g _ { t } ) .$   
16: $\pi _ { \mathrm { c a l } , t + 1 }  \pi _ { w _ { t + 1 } } .$   
Student gradient update.   
17: Define $\begin{array} { r } { Z _ { t + 1 } ( \theta , x , a _ { 1 : H } ) : = \lambda \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } | x ) } { \pi _ { \mathrm { c a l } , t + 1 } ( a _ { 1 : H } | x ) } , S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) : = \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } | x ) , \forall \theta \in \Theta . } \end{array}$   
18: Draw independent prompt-answer pairs for $\ell = 1 , \dots , b _ { t + 1 } \colon$   
$j _ { t + 1 , \ell } ^ { \mathrm { g } } \sim \mathrm { U n i f } \{ 1 , \dots , m \} , \quad a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } \mid j _ { t + 1 , \ell } ^ { \mathrm { g } } \sim \pi _ { \mathrm { s t u } , \theta _ { t } } ( \cdot \mid \widetilde { x } _ { j _ { t + 1 , \ell } ^ { \mathrm { g } } } ) .$   
b   
19: $\widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \gets \frac { 1 } { b _ { t + 1 } } \sum _ { \ell = - 1 } ^ { \boldsymbol { \operatorname { \texttt { o } } } _ { t + 1 } } S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j _ { t + 1 , \ell } ^ { \mathrm { g } } } , a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } ) Z _ { t + 1 } ( \theta _ { t } ; \widetilde { x } _ { j _ { t + 1 , \ell } ^ { \mathrm { g } } } , a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } ) .$   
20: $\vartheta _ { t + 1 , 1 } \gets \mathrm { P r o j } _ { \Theta } \left( \theta _ { t } - \alpha _ { t + 1 } \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \right)$ ; independently draw $\vartheta _ { t + 1 , 2 } \sim \mathrm { U n i f } ( \Theta )$   
Candidate student evaluation and selection.   
21: for $k = 1 , 2$ do   
22: Draw fresh independent prompt-answer pairs for $\ell = 1 , \ldots , q _ { t + 1 } \colon$   
$j _ { t + 1 , k , \ell } \sim \mathrm { U n i f } \{ 1 , \dots , m \} , \quad a _ { t + 1 , k , \ell , 1 : H } ^ { \mathrm { v a l } } \mid j _ { t + 1 , k , \ell } \sim \pi _ { \mathrm { s t u } , \vartheta _ { t + 1 , k } } ( \cdot \mid \widetilde { x } _ { j _ { t + 1 , k , \ell } } ) .$   
23: $\widehat { C } _ { t + 1 , k } \gets \frac { 1 } { q _ { t + 1 } } \sum _ { \ell = 1 } ^ { q _ { t + 1 } } Z _ { t + 1 } \big ( \vartheta _ { t + 1 , k } ; \widetilde { x } _ { j _ { t + 1 , k , \ell } } , a _ { t + 1 , k , \ell , 1 : H } ^ { \mathrm { v a l } } \big ) .$   
24: end for   
25: $\widehat { k } _ { t + 1 } \gets$ min argmin $\iota _ { k \in \{ 1 , 2 \} } \widehat { C } _ { t + 1 , k } , \theta _ { t + 1 } \gets \vartheta _ { t + 1 , \widehat { k } _ { t + 1 } } .$   
26: end for   
27: return $\pi _ { \mathrm { s t u } , \theta _ { T } }$

one reference-weighted branch, complete it with $\pi _ { \mathrm { p r e } } .$ and query its reward $R _ { t }$ . The acceptance indicator satisfies

$$
\operatorname* { P r } \mathopen { } \mathclose \bgroup \left( I _ { t } = 1 \aftergroup \egroup | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } , R _ { t } \aftergroup \egroup \right) = \exp \mathopen { } \mathclose \bgroup \left( \left( R _ { t } - 1 \aftergroup \egroup \right) / \lambda \aftergroup \egroup \right) .
$$

Here $\mathcal { F } _ { t }$ denotes the history before round t. The algorithm samples the branch and completion using the reference policy, and determines acceptance using the observed source reward. Under

Assumption 3.2, Lemma B.2 shows that the resulting accepted branch index satisfies

$$
\operatorname* { P r } ( Y _ { t } = 1 \mid I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) = \frac { \pi _ { \lambda } ^ { \star } ( c _ { t , 1 } \mid s _ { t } ) } { \pi _ { \lambda } ^ { \star } ( c _ { t , 1 } \mid s _ { t } ) + \pi _ { \lambda } ^ { \star } ( c _ { t , 0 } \mid s _ { t } ) } = \sigma ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } ) .
$$

This identity characterizes the unknown parameter underlying the accepted labels. Algorithm 1 learns this parameter from the sampled labels by evaluating the logistic gradient at the current estimate $g _ { t } = I _ { t } z _ { t } \Big [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } \Big ]$ . Thus, source reward feedback provides logistic supervision for estimating $w _ { \lambda } ^ { \star }$ . This observation motivates the calibration update in Lines 14-15:

$$
g _ { t } = I _ { t } z _ { t } [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } ] , \ w _ { t + 1 } = \mathrm { P r o j } _ { W } ( w _ { t } - \eta _ { t } g _ { t } ) , \ \pi _ { \mathrm { c a l } , t + 1 } = \pi _ { w _ { t + 1 } } .
$$

Here $g _ { t }$ is the stochastic gradient of the acceptance-weighted logistic loss. The projection keeps the updated calibration parameter in W.

The calibrated teacher then defines the student’s training objective on the target prompts:

$$
C _ { t + 1 } ( \theta ) : = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } \bigl ( \pi _ { \mathrm { s t u } , \theta } ( \cdot  { | } \widetilde { x } _ { j } )  { | } | \pi _ { w _ { t + 1 } } ( \cdot  { | } \widetilde { x } _ { j } ) \bigr ) .\tag{4.1}
$$

Assumption 3.2 links source calibration to this target objective: the same $w _ { \lambda } ^ { \star }$ represents the regularized optimal policy on both datasets. The source-identification condition in (3.7) makes this parameter identifiable from source comparisons, and the known feature map evaluates the learned policy at target prefixes. Moreover, Lemma B.1 gives

$$
\arg \operatorname* { m i n } _ { \theta \in \Theta } C _ { w _ { \lambda } ^ { \star } } ( \theta ) = \arg \operatorname* { m a x } _ { \theta \in \Theta } J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta } ) .
$$

Thus, the calibration aligns the distillation objective with the regularized oracle-student objective. To update the student, Lines 18-19 estimate the gradient of $C _ { t + 1 }$ using $b _ { t + 1 }$ independent currentstudent rollouts. The required quantities are the full-answer score and the sampled log-ratio cost:

$$
S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) : = \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) = \sum _ { h = 1 } ^ { H } \left[ \phi _ { \mathrm { s t u } } ( s _ { h } , a _ { h } ) - \sum _ { b \in B ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) \right] ,
$$

$$
Z _ { t + 1 } ( \vartheta ; x , a _ { 1 : H } ) : = \lambda \log \frac { \pi _ { \mathrm { s t u } , \vartheta } ( a _ { 1 : H } \mid x ) } { \pi _ { w _ { t + 1 } } ( a _ { 1 : H } \mid x ) } = \lambda \sum _ { h = 1 } ^ { H } \log \frac { \pi _ { \mathrm { s t u } , \vartheta } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } { \pi _ { w _ { t + 1 } } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } ,
$$

where $s _ { h } = \left( x , a _ { 1 : h - 1 } \right)$ and $B ( s _ { h } )$ is the legal-token set defined in the model setup. The score measures how the answer’s log probability changes with the student parameters. The cost $Z _ { t + 1 }$ measures its log-likelihood discrepancy from the calibrated teacher.

For each $\ell \in \{ 1 , \dots , b _ { t + 1 } \}$ , we draw a fresh uniform target index $j _ { t + 1 , \ell } ^ { \mathrm { g } }$ and then sample an answer $a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } }$ from the current student at that question. The $b _ { t + 1 }$ prompt-answer pairs are independent conditional on the history and the calibration update. Averaging their score-cost products gives

$$
\widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } = \frac { 1 } { b _ { t + 1 } } \sum _ { \ell = 1 } ^ { b _ { t + 1 } } S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j _ { t + 1 , \ell } } , a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } ) Z _ { t + 1 } ( \theta _ { t } ; \widetilde { x } _ { j _ { t + 1 , \ell } ^ { \mathrm { g } } } , a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } ) .
$$

Lemma B.6 shows that

$$
\mathbb E \left[ \left. \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \right| w _ { t + 1 } , \theta _ { t } \right] = \nabla _ { \theta } C _ { t + 1 } \big ( \theta \big ) \vert _ { \theta = \theta _ { t } } .
$$

The calibrated teacher is held fixed throughout this batch. Averaging preserves unbiasedness and reduces the conditional gradient variance by a factor of $b _ { t + 1 }$

Line 20 uses this estimate to form two student candidates:

$$
\vartheta _ { t + 1 , 1 } = \mathrm { P r o j } _ { \Theta } \left( \theta _ { t } - \alpha _ { t + 1 } \hat { g } _ { t + 1 } ^ { \mathrm { s t u } } \right) , \ \vartheta _ { t + 1 , 2 } \sim \mathrm { U n i f } ( \Theta ) .
$$

The projected-gradient candidate takes one step using the averaged gradient, with a constant step size justified by the smoothness bound in Lemma B.7. The random candidate explores the full parameter ball uniformly. In Lemma B.9, exploration provides progress outside a fixed neighborhood of the oracle student, while the gradient candidate provides quantitative progress near the oracle student. Each round thus constructs one projected-gradient proposal and the batch improves the accuracy of this update.

To compare these candidates, the validation steps ending at Line 23 average $Z _ { t + 1 }$ over $q _ { t + 1 }$ fresh rollouts from each candidate, $k \in \{ 1 , 2 \}$ , by Monte Carlo simulation:

$$
\widehat { C } _ { t + 1 , k } = \frac { 1 } { q _ { t + 1 } } \sum _ { \ell = 1 } ^ { q _ { t + 1 } } Z _ { t + 1 } \Bigl ( \vartheta _ { t + 1 , k } ; \widetilde { x } _ { j _ { t + 1 , k , \ell } } , a _ { t + 1 , k , \ell , 1 : H } ^ { \mathrm { v a l } } \Bigr ) .
$$

Since the target indices are uniform and each answer is sampled from the candidate being evaluated,

$$
\mathbb { E } \left[ \widehat { C } _ { t + 1 , k } \Big | w _ { t + 1 } , \vartheta _ { t + 1 , k } \right] = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { s _ { \mathrm { t u } , \vartheta _ { t + 1 , k } } } ( \cdot | \widetilde { x } _ { j } ) } \big [ Z _ { t + 1 } \big ( \vartheta _ { t + 1 , k } ; \widetilde { x } _ { j } , a _ { 1 : H } \big ) \big ] = C _ { t + 1 } \big ( \vartheta _ { t + 1 , k } \big ) .
$$

The sampled token log ratios therefore estimate the KL in the required student-to-calibrated-teacher direction. The algorithm selects the smallest estimated cost:

$$
\widehat { k } _ { t + 1 } = \operatorname* { m i n } _ { k \in \{ 1 , 2 \} } \widehat { C } _ { t + 1 , k } , \ \theta _ { t + 1 } = \vartheta _ { t + 1 , \widehat { k } _ { t + 1 } } .
$$

The same quantity $Z _ { t + 1 }$ consequently serves as both the weight in the student gradient estimate and the validation cost. Both quantities are computed from policy likelihoods and target rollouts, so target reward feedback is not required. The selected student supplies the alternative-token distribution in the next source comparison, completing the coupling between teacher calibration and student distillation.

For the calibration step size and its analysis, let $\sigma ( u ) = ( 1 + e ^ { - u } ) ^ { - 1 }$ and define

$$
\gamma : = e ^ { - 1 / \lambda } e ^ { - 2 B } \sigma ^ { \prime } ( 2 B ) \mu _ { \mathrm { j o i n t } } > 0 .\tag{4.2}
$$

This constant quantifies the source-calibration information used in the convergence analysis. The algorithm uses γ to set its calibration step size. Define

$$
L _ { \mathrm { s t } } : = \lambda H ( 1 + 8 B H ) , ~ \alpha _ { \mathrm { s t u } } : = \frac { 1 } { 2 L _ { \mathrm { s t } } } .
$$

Lemma B.7 proves that $L _ { \mathrm { s t } }$ is a uniform smoothness bound for the student objectives. For $t =$

$0 , \ldots , T - 1$ , we use

$$
\eta _ { t } = \frac { 1 } { \gamma ( t + 2 ) } , \ \alpha _ { t + 1 } = \alpha _ { \mathrm { s t u } } , \ b _ { t + 1 } = t + 2 , \ q _ { t + 1 } = ( t + 2 ) ^ { 2 } ,\tag{4.3}
$$

where $\gamma = e ^ { - 1 / \lambda } e ^ { - 2 B } \sigma ^ { \prime } ( 2 B ) \mu _ { \mathrm { j o i n t } } > 0$ is defined in (4.2). These schedules specify the calibration step size, student step size, gradient batch size, and validation budget per candidate, respectively. The increasing gradient batch makes its sampling error vanish while the student step size remains fixed. All draws use fresh randomness conditional on their stated sampling laws, and the two validation batches are independent conditional on the history and candidate parameters. Thus target update $t + 1$ uses $b _ { t + 1 } + 2 q _ { t + 1 }$ full-answer rollouts: $b _ { t + 1 }$ for its gradient estimate and $q _ { t + 1 }$ for each candidate evaluation. Over T rounds, the algorithm makes exactly T source reward queries and uses $O ( T ^ { 3 } )$ target rollouts, of which $O ( T ^ { 2 } )$ estimate student gradients. Each round forms one projected student-gradient candidate.

## 5 Theoretical guarantee

In this section, we establish a finite-iteration convergence guarantee for CCL (Algorithm 1). We bound the average KL divergence between the returned student and the oracle student, accounting for teacher calibration, stochastic student gradients, and finite-rollout evaluation of the two student candidates. Throughout this section, the source and target datasets are fixed, and expectations include the complete adaptive randomness of the algorithm.

Theorem 5.1. Under the model and assumptions in Section 3, there exist fixed-model constants $C _ { \lambda , m } > 0$ and $p _ { \lambda , m } \ge 1$ , independent of $T ,$ such that the output of Algorithm 1 satisfies, for every integer $T \geq 1$

$$
\mathbb { E } \left[ \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } \Big ( \pi _ { \mathrm { s t u } , \theta _ { T } } ( \cdot \mid \widetilde { x } _ { j } ) \Big \| \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \mid \widetilde { x } _ { j } ) \Big ) \right] \leq C _ { \lambda , m } ( T + 1 ) ^ { - 1 / ( 4 p _ { \lambda , m } ) } .\tag{5.1}
$$

Proof sketch. We first identify the objective that the student should minimize. The true regularized return equals a constant minus $C _ { w _ { \lambda } ^ { \star } } ( \theta )$ . Hence, its minimizer over Θ is exactly the oracle student. We study the excess true cost $\Delta _ { \lambda , m } ( \theta ) = C _ { w _ { \lambda } ^ { \star } } ( \theta ) - C _ { w _ { \lambda } ^ { \star } } ( \theta _ { \lambda , m } ^ { \dagger } )$

We next control the source calibration update. The accepted branch index is a logistic observation with parameter $w _ { \lambda } ^ { \star }$ . Source identification then gives

$$
\mathbb { E } \Big [ \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } \Big ] \leq \frac { 4 } { \gamma ^ { 2 } ( t + 2 ) } , \ t \geq 1 .
$$

This bound holds for the adaptive teacher sequence generated by CCL.

For the student update, we first analyze a population projected gradient step for $C _ { w _ { \lambda } ^ { \star } }$ . The softmax model gives a uniform smoothness bound. Analyticity and the uniqueness of the oracle yield a local Lojasiewicz inequality, which implies that this gradient step decreases the excess cost by at least a positive constant times its square in a fixed near-optimal region. Outside that region, the uniform candidate has a fixed positive probability of proposing a student with lower true cost. The population gradient step never increases the true cost, so selecting the better of the gradient and uniform proposals combines these two sources of progress. Thus gradient descent supplies the improvement near the oracle, while exploration supplies progress outside that region.

The algorithm uses a sampled gradient of $C _ { w _ { t } }$ instead of the population gradient of $C _ { w _ { \lambda } ^ { \star } }$ . We control this diference using the minibatch variance and the calibration bound above. Fresh validation controls the error when comparing candidates. With $\boldsymbol { e } _ { t } : = \mathbb { E } [ \Delta _ { \lambda , m } ( \theta _ { t } ) ]$ , the resulting recursion is

$$
e _ { t } \leq e _ { t - 1 } - \kappa _ { \mathrm { o p t } } e _ { t - 1 } ^ { 2 } + \underbrace { \frac { 1 6 \alpha _ { \mathrm { s t u } } \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } } { \sqrt { t + 1 } } } _ { \mathrm { g r a d i e n t ~ e s t i m a t i o n } } + \underbrace { \frac { 1 6 \alpha _ { \mathrm { s t u } } \lambda ^ { 2 } B H ^ { 3 } + 8 \lambda H } { \gamma \sqrt { t + 2 } } } _ { \mathrm { c a l i b r a t i o n } } + \underbrace { \frac { 1 6 \lambda B H } { t + 1 } } _ { \mathrm { v a l i d a t i o n } } ,
$$

where $\kappa _ { \mathrm { o p t } } > 0$ is independent of t. An induction then gives $e _ { T } \leq K _ { \mathrm { o p t } } ( T + 1 ) ^ { - 1 / 4 }$ for a fixed constant $K _ { \mathrm { o p t } } > 0$

Finally, recall the average oracle-policy KL $\kappa _ { \lambda , m }$ from (3.6). Its zero set contains the zero set of $\Delta _ { \lambda , m }$ , and both functions are analytic on the compact parameter ball. A second Lojasiewicz inequality therefore gives

$$
\begin{array} { r } { \Delta _ { \lambda , m } ( \theta ) \geq a _ { \lambda , m } [ K _ { \lambda , m } ( \theta ) ] ^ { p _ { \lambda , m } } , \ a _ { \lambda , m } > 0 , \ p _ { \lambda , m } \geq 1 . } \end{array}
$$

Applying Jensen’s inequality, we obtain

$$
\mathbb { E } [ K _ { \lambda , m } ( \theta _ { T } ) ] \le \left( \frac { \mathbb { E } [ \Delta _ { \lambda , m } ( \theta _ { T } ) ] } { a _ { \lambda , m } } \right) ^ { 1 / p _ { \lambda , m } } \le \left( \frac { K _ { \mathrm { o p t } } } { a _ { \lambda , m } } \right) ^ { 1 / p _ { \lambda , m } } ( T + 1 ) ^ { - 1 / ( 4 p _ { \lambda , m } ) } .
$$

This proves the stated rate with $C _ { \lambda , m } = ( K _ { \mathrm { o p t } } / a _ { \lambda , m } ) ^ { 1 / p _ { \lambda , m } }$ . Appendix B gives the constants and the complete proof.

Theorem 5.1 establishes that LLM distillation can recover the optimal student from a biased teacher, even when true reward feedback is entirely unavailable on the target dataset. Under some regularity assumptions, the expected average KL distance to the oracle student vanishes at a polynomial rate. Crucially, this oracle is defined by the true reference-regularized target objective, so the guarantee concerns the student’s actual target performance rather than its agreement with the teacher. The benchmark also respects the student’s limited capacity: the student class need not represent the unrestricted optimal policy. The result, therefore, connects source reward feedback to optimal target-side learning within a fixed student class. Through coupled calibration and distillation, source supervision corrects the policy that guides target training, allowing the student to overcome errors in the original teacher. Consequently, teacher bias need not impose a persistent loss relative to the best achievable student, and recovering this benchmark does not require collecting reward feedback on the target questions.

## 6 Separation from Direct Teacher Matching

In this section, to illustrate the power of our algorithm, we show that regularized direct matching can retain a positive error relative to the oracle student. Specifically, we construct a target instance in which the teacher is better than any model in the student model class; yet, direct matching learns a student policy that is strictly separated from the oracle student.

The distillation problem that we consider is a LLM-as-judge setting. Fix $H = 2 , 1 \leq d < D , \lambda > 0$ and $\alpha \in [ 1 / 2 , 1 )$ . Each prompt contains a question and a candidate answer to be assessed. At $h = 1$ , the model outputs a verdict: 1 declares the candidate correct, 0 declares it incorrect, and null expresses abstention due to uncertainty. Then when $h = 2$ , it outputs EOS to terminate the answer.

This setting models the practical task of distilling a compact LLM judge for automatic response evaluation. Training such judges on feedback from stronger models has been demonstrated using GPT-4-generated judgments and feedback (Zhu et al., 2025; Kim et al., 2024b). Our formulation captures the verdict-generation component of this task, including an explicit abstention option.

The fixed target dataset consists of $m = 2 d$ distinct prompts ${ \mathcal { D } } _ { \operatorname { t a r } } = \{ \widetilde { x } _ { i , + } , \widetilde { x } _ { i , - } : i \in [ d ] \}$ . For an equivalent single-index enumeration, set $\widetilde x _ { 2 i - 1 } : = \widetilde x _ { i , + }$ and $\widetilde { x } _ { 2 i } : = \widetilde { x } _ { i , - }$ . Let ${ \mathcal X } = { \mathcal D } _ { \mathrm { t a r } }$ and $\mathcal { A } = \{ 0 , 1 , \mathtt { n u l l } , \mathtt { E 0 S } \}$ . The legal-token sets are

$$
\mathcal { B } ( ( x , \mathcal { O } ) ) = \{ 0 , 1 , \mathrm { n u l 1 } \} , \ \mathcal { B } ( ( x , a _ { 1 } ) ) = \{ \mathrm { E 0 S } \} , \ a _ { 1 } \in \{ 0 , 1 , \mathrm { n u l 1 } \} .
$$

The state transition appends the selected token, and the state $( x , a _ { 1 } , \mathtt { E 0 S } )$ is terminal. Thus

$$
\begin{array} { r } { \boldsymbol { A } ( \boldsymbol { x } ) = \{ ( 0 , \boldsymbol { \mathrm { E 0 S } } ) , ( 1 , \boldsymbol { \mathrm { E 0 S } } ) , ( \mathrm { n u l l } , \boldsymbol { \mathrm { E 0 S } } ) \} . } \end{array}\tag{6.1}
$$

Here null is a first-step abstention verdict; the legal-token sets above specify this task’s output format. Every policy therefore satisfies

$$
\pi ( \operatorname { E O S } \mid x , a _ { 1 } ) = 1 , \pi ( ( a _ { 1 } , \operatorname { E O S } ) \mid x ) = \pi ( a _ { 1 } \mid x , \emptyset ) , a _ { 1 } \in \{ 0 , 1 , \mathrm { n u l 1 } \} .
$$

Now, we model the ground truth verifier. We set that the candidate answer is correct at $\tilde { x } _ { i , + }$ and incorrect at $\widetilde { x } _ { i , - }$ . Hence, we define its ground-truth verdict and the exact evaluation reward by

$$
y ( \widetilde x _ { i , + } ) = 1 , \ y ( \widetilde x _ { i , - } ) = 0 , \ R ( x , ( a _ { 1 } , \mathtt { E 0 S } ) ) : = { \bf 1 } \{ a _ { 1 } = y ( x ) \} .\tag{6.2}
$$

That is, a correct judgment receives reward 1. An incorrect judgment or abstention receives reward 0. In particular, the verdict 0 receives reward 1 when the candidate answer is incorrect.

These target labels and rewards define the true benchmark; direct matching has access only to the target prompts and policy likelihoods.

We assume that our pre-trained reference policy $\pi _ { \mathrm { p r e } }$ is uniform over {0, 1, null},

$$
\pi _ { \mathrm { p r e } } ( a _ { 1 } \mid x , \emptyset ) = \frac { 1 } { 3 } , \ \pi _ { \mathrm { p r e } } ( \mathtt { E 0 S } \mid x , a _ { 1 } ) = 1 , \ a _ { 1 } \in \{ 0 , 1 , \mathtt { n u l l } \} .
$$

Let $u _ { 1 } , \ldots , u _ { D }$ be an orthonormal basis of $\mathbb { R } ^ { D }$ , and let $e _ { 1 } , \ldots , e _ { d }$ be the standard basis of $\mathbb { R } ^ { d }$ . For the linear softmax teacher class, the feature vector is specified

$$
\begin{array} { r l } { \frac { \ d } { \ d t } \frac { a = 1 } { \ d t } } & { { } a = 0 \quad a = \mathrm { n u 1 1 } } \\ { \phi ( ( \widetilde { x } _ { i , + } , \emptyset ) , a ) \ d u \frac { \ d u } { \ d t } \big | \frac { \ d u = 1 } { \ d u } } & { { } u _ { 2 } / \sqrt { 2 } \qquad 0 } \\ { \phi ( ( \widetilde { x } _ { i , - } , \emptyset ) , a ) \ d u \big | \ d u _ { 2 } / \sqrt { 2 } } & { { } u _ { 1 } / \sqrt { 2 } \qquad 0 } \end{array}\tag{6.3}
$$

For $\epsilon \in \{ + , - \}$ , we define the student features by

$$
\phi _ { \mathrm { s t u } } ( ( \widetilde { x } _ { i , \epsilon } , \mathcal { O } ) , a ) = e _ { i } { \bf 1 } \{ a = 1 \} , ~ a \in \{ 0 , 1 , \mathtt { n u l 1 } \} .\tag{6.4}
$$

For student and teacher classes, we set both feature maps to zero at every second-step state, and

set all unspecified feature values to zero. Every feature has norm at most one. Set

$$
B : = \frac { \sqrt { d } + 2 } { \lambda } , W : = \{ w \in \mathbb { R } ^ { D } : \| w \| _ { 2 } \leq B \} , \Theta : = \{ \theta \in \mathbb { R } ^ { d } : \| \theta \| _ { 2 } \leq B \} .
$$

We use the linear-softmax policies

$$
\pi _ { w } ( a \mid s ) = \frac { \exp ( w ^ { \top } \phi ( s , a ) ) } { \sum _ { c \in \mathcal { B } ( s ) } \exp ( w ^ { \top } \phi ( s , c ) ) } , \pi _ { \operatorname { s t u } , \theta } ( a \mid s ) = \frac { \exp ( \theta ^ { \top } \phi _ { \operatorname { s t u } } ( s , a ) ) } { \sum _ { c \in \mathcal { B } ( s ) } \exp ( \theta ^ { \top } \phi _ { \operatorname { s t u } } ( s , c ) ) } .\tag{6.5}
$$

At the second step, the single legal token EOS has probability $e ^ { 0 } / e ^ { 0 } = 1$ . We set that our frozen teacher has parameter

$$
w _ { \mathrm { t e a } } : = \frac { \alpha \sqrt 2 } { \lambda } u _ { 1 } \in W , \ \pi _ { \mathrm { t e a } } : = \pi _ { w _ { \mathrm { t e a } } } .\tag{6.6}
$$

The objective function in this post-training process is

$$
J _ { \lambda , m } ( \pi ) : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : 2 } \sim \pi ( \cdot | \widetilde { x } _ { j } ) } \left[ R ( \widetilde { x } _ { j } , a _ { 1 : 2 } ) - \lambda \sum _ { h = 1 } ^ { 2 } \log \frac { \pi ( a _ { h } \mid \widetilde { x } _ { j } , a _ { 1 : h - 1 } ) } { \pi _ { \mathrm { p r e } } ( a _ { h } \mid \widetilde { x } _ { j } , a _ { 1 : h - 1 } ) } \right] .\tag{6.7}
$$

The oracle student policy has the parameter $\theta _ { \lambda , m } ^ { \dagger } \in \operatorname { a r g m a x } _ { \theta \in \Theta } J _ { \lambda , m } \left( \pi _ { \mathrm { s t u } , \theta } \right)$

At each target prompt, we use $\pi _ { \lambda } ^ { \star }$ to denote the unrestricted maximizer of the corresponding reward-minus-reference-KL objective over all laws in $\Delta ( \mathcal { A } ( x ) )$ .

In the next proposition, we compute the optimal parameter $\theta _ { \lambda , m } ^ { \dagger }$ for the oracle student policy explicitly. It also verifies that the teacher is stronger than this oracle student, although the teacher itself is biased.

Proposition 6.1. For the setting above, the unique oracle parameter is $\begin{array} { r } { \theta _ { \lambda , m } ^ { \dagger } = \frac { 1 } { 4 \lambda } \mathbf { 1 } _ { d } \in \mathrm { i n t } ( \Theta ) } \end{array}$ For every $i \in [ d ] , \epsilon \in \{ + , - \}$ , and feasible answer,

$$
\pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( ( a _ { 1 } , \mathrm { E } 0 \mathbf { S } ) \mid \widetilde { x } _ { i , \epsilon } ) = \frac { \exp ( \mathbf { 1 } \{ a _ { 1 } = 1 \} / ( 4 \lambda ) ) } { e ^ { 1 / ( 4 \lambda ) } + 2 } .\tag{6.8}
$$

The unrestricted optimal policy is $\pi _ { \lambda } ^ { \star } = \pi _ { w _ { \lambda } ^ { \star } }$ with $w _ { \lambda } ^ { \star } = ( \sqrt { 2 } / \lambda ) u _ { 1 } \in W$ . The student class cannot represent it, and

$$
J _ { \lambda , m } ( \pi _ { \mathrm { t e a } } ) > J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ) .\tag{6.9}
$$

However, in direct matching distillation, we do not have the golden answers to the prompts, and thus we do not have access to the reward verifier function R. Therefore, we train our student based on the synthetic outputs from the fixed teacher model $\pi _ { \mathrm { t e a } }$

The student maximizes the teacher-student log-likelihood ratio minus the reference penalty:

$$
J _ { \mathrm { S M } } ( \theta ) : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } \left[ \log \frac { \pi _ { \mathrm { t e a } } ( a _ { 1 : 2 } \mid \widetilde { x } _ { j } ) } { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid \widetilde { x } _ { j } ) } - \lambda \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid \widetilde { x } _ { j } ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : 2 } \mid \widetilde { x } _ { j } ) } \right] .\tag{6.10}
$$

Equivalently, it minimizes the cost

$$
C _ { \mathrm { S M } } ( \theta ) : = - J _ { \mathrm { S M } } ( \theta ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } [ \mathrm { K L } ( \pi _ { \mathrm { s t n } , \theta } ( \cdot \ | \ \widetilde { x } _ { j } ) | | \pi _ { \mathrm { t e a } } ( \cdot \ | \ \widetilde { x } _ { j } )  ) + \lambda \mathrm { K L } ( \pi _ { \mathrm { s t n } , \theta } ( \cdot \ | \ \widetilde { x } _ { j } ) | | \pi _ { \mathrm { p r e } } ( \cdot \ | \ \widetilde { x } _ { j } )  ) ) ] .\tag{6.11}
$$

For a feasible answer $a _ { 1 : 2 }$ , we define the evaluable cost and student score:

$$
\begin{array} { r } { Z _ { \mathrm { S M } } ( \theta ; x , a _ { 1 : 2 } ) : = \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid x ) } { \pi _ { \mathrm { t e a } } ( a _ { 1 : 2 } \mid x ) } + \lambda \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid x ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : 2 } \mid x ) } , \ S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : 2 } ) : = \nabla \theta \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid x ) . } \end{array}\tag{6.12}
$$

For the step size in the direct matching distillation algorithm, we set $\kappa _ { \mathrm { S M } } = \sigma ^ { \prime } ( B + \log 2 ) >$ $\begin{array} { r } { 0 , \ G _ { \mathrm { S M } } = ( 2 + \lambda ) B , \ \mu _ { \mathrm { S M } } = \frac { ( 1 + \lambda ) \kappa _ { \mathrm { S M } } } { d } , \ \eta _ { t } ^ { \mathrm { S M } } = \frac { 1 } { \mu _ { \mathrm { S M } } ( t + 2 ) } } \end{array}$ . The pseudocode for the regularized direct-matching baseline is given in Algorithm 2.

Algorithm 2 Direct Teacher Matching   
Require: Fixed target prompts $\{ \widetilde { x } _ { j } \} _ { j = 1 } ^ { m } ;$ frozen $\pi _ { \mathrm { t e a } } , \pi _ { \mathrm { p r e } } ;$ student class $\Theta ; \lambda > 0 ;$ stepsizes $\eta _ { t } ^ { \mathrm { S M } } ; T \ge 1$   
1: $\theta _ { 0 } ^ { \mathrm { S M } }  0 .$   
2: for $t = 0 , \ldots , T - 1$ do   
3: Draw $j _ { t \_ \cdot } ^ { \mathrm { S M } } \sim \operatorname { U n i f } \{ 1 , \dots , m \} .$   
4: Draw $a _ { t , \mathrm { 1 : 2 } } ^ { \mathrm { \scriptsize { S M } } } \sim \pi _ { \mathrm { s t u } , \boldsymbol { \theta } _ { t } ^ { \mathrm { { S M } } } } ( \cdot \mid \widetilde { x } _ { j _ { t } ^ { \mathrm { { S M } } } } )$ autoregressively.   
5: $\begin{array} { r } { \widehat { g } _ { t } ^ { \mathrm { S M } } \gets S _ { \mathrm { s t u } , \theta _ { t } ^ { \mathrm { S M } } } ( \widetilde { x } _ { j _ { t } ^ { \mathrm { S M } } } ^ { \mathrm { ~ \alpha ~ } } , a _ { t , 1 : 2 } ^ { \mathrm { S M } ^ { - } } ) Z _ { \mathrm { S M } } ( \theta _ { t } ^ { \mathrm { S M } } ; \widetilde { x } _ { j _ { t } ^ { \mathrm { S M } } } , a _ { t , 1 : 2 } ^ { \mathrm { S M } } ) . } \end{array}$ ▷ Estimate the cost gradient.   
6: $\theta _ { t + 1 } ^ { \mathrm { S M } }  \mathrm { P r o j } _ { \Theta } ^ { \cdot } ( \theta _ { t } ^ { \mathrm { S M } ^ { \cdot } } - \eta _ { t } ^ { \mathrm { S M } } \widehat { g } _ { t } ^ { \mathrm { S M } } )$   
7: end for   
8: return $\pi _ { \mathrm { s t u } , \theta _ { T } ^ { \mathrm { S M } } } .$

Now, we compare the output of direct matching with the explicit oracle student in Proposition 6.1.   
We will show that there is a separation between the trained student and the oracle student policy.   
Specifically, our next theorem shows that their KL divergence is lower bounded by a constant.

Theorem 6.2. For the target setting in Section 6, for every $T \geq 1$ , Algorithm 2 satisfies

$$
\mathbb { E } [ \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta _ { T } ^ { \mathrm { S M } } } ( \cdot \ | \ \widetilde { \boldsymbol { x } } _ { j } )  \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \ | \ \widetilde { \boldsymbol { x } } _ { j } ) ) ] \geq \frac { \kappa _ { \mathrm { S M } } } { 2 d } [ \frac { \sqrt { d } ( 1 - \alpha ) } { 4 \lambda } - \frac { G _ { \mathrm { S M } } } { \mu _ { \mathrm { S M } } \sqrt { T + 1 } } ] _ { + } ^ { 2 } .\tag{6.13}
$$

Here $[ u ] _ { + } : = \operatorname* { m a x } \{ u , 0 \}$ . In particular, for every integer $\begin{array} { r } { T \geq \left\lceil \frac { 6 4 \lambda ^ { 2 } d G _ { \mathrm { S M } } ^ { 2 } } { ( 1 + \lambda ) ^ { 2 } \kappa _ { \mathrm { S M } } ^ { 2 } ( 1 - \alpha ) ^ { 2 } } \right\rceil } \end{array}$ , the expected average KL divergence is at least $\kappa _ { \mathrm { S M } } ( 1 - \alpha ) ^ { 2 } / \big ( 1 2 8 \lambda ^ { 2 } \big )$

Therefore, we conclude that direct matching retains a nonvanishing distillation error relative to the oracle student $\pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } }$

This theorem highlights the importance of teacher calibration in CCL. Regularized direct matching can retain teacher bias in the learned student, whereas CCL iteratively calibrates the policy that defines the student’s training objective. It complements Theorem 5.1: CCL converges under the source-calibration assumptions, whereas teacher quality alone does not guarantee that direct imitation recovers the oracle student.

## 7 Discussion

We developed Coupled Calibration and Learning (CCL) and a statistical framework for mitigating teacher bias in LLM distillation when reward feedback is available only for source questions. CCL couples teacher calibration with student updates through token-level branching, using source feedback to guide distillation on target questions. Each student update selects between a minibatch projected-gradient proposal and a uniformly sampled candidate using fresh target rollouts. We established its polynomial convergence in expected average KL divergence to the oracle student for the reference-regularized target objective. The proof quantifies progress from the student gradient update near the oracle and uses exploration to obtain progress outside a fixed near-optimal region. It also controls the errors from teacher calibration and finite-rollout estimation. This benchmark accounts for the limitations of the student class and does not require it to represent the unrestricted optimal policy. We also established a separation from regularized direct matching, which can retain a nonvanishing error even when the teacher outperforms every student policy. Together, these results identify coupled calibration and learning as a mechanism for recovering the optimal student without target-domain reward feedback.

Several directions remain for future investigation. First, computational experiments with pretrained LLMs on coding and mathematical reasoning tasks would help assess the method under realistic rollout budgets, reward-evaluation costs, and source-target shifts. Such experiments could also examine the contributions of calibration, student updates, and candidate selection to practical performance. Second, our algorithm updates the calibration parameter and the student in every iteration. It would be useful to study less frequent calibration, with multiple student updates between successive calibration steps. The central question is whether such schedules preserve convergence to the oracle student while reducing calibration costs, and how their relative update frequencies afect the rate. Finally, our analysis gives an explicit one-quarter exponent for the excess true cost, while conversion to oracle-policy KL still uses a model-dependent Lojasiewicz exponent. The constants also depend on the local geometry and the probability of exploring a fixed near-optimal region. Deriving explicit bounds on these quantities is an important theoretical direction. A sharper analysis may exploit the stronger local gradient inequality before its quadratic relaxation, reduce the gradient and validation budgets, and clarify the dependence on the horizon and model dimensions.

## References

Alekh Agarwal, Sham M. Kakade, Jason D. Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22(98):1–76, 2021.

Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. Prediction-powered inference. Science, 382(6671):669–674, 2023.

Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. In Advances in Neural Information Processing Systems, volume 36, pages 57125–57211, 2023.

Peter L Bartlett, Philip M Long, G´abor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. Proceedings of the National Academy of Sciences, 117(48):30063–30070, 2020.

Edward Bierstone and Pierre D. Milman. Semianalytic and subanalytic sets. Publications Math´ematiques de l’IHES<sup>´</sup> , 67:5–42, 1988.

J´erˆome Bolte, Aris Daniilidis, and Adrian Lewis. The lojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems. SIAM Journal on Optimization, 17(4):1205–1223, 2007.

Fan Chen, Zeyu Jia, Alexander Rakhlin, and Tengyang Xie. Outcome-based online reinforcement learning: Algorithms and fundamental limits. In Advances in Neural Information Processing Systems, volume 38, 2025.

Fan Chen, Audrey Huang, Noah Golowich, Sadhika Malladi, Adam Block, Jordan T. Ash, Akshay Krishnamurthy, and Dylan J. Foster. The coverage principle: How pre-training enables posttraining. In The Fourteenth International Conference on Learning Representations, 2026.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, et al. Evaluating large language models trained on code, 2021. arXiv preprint arXiv:2107.03374.

Wojciech M. Czarnecki, Razvan Pascanu, Simon Osindero, Siddhant Jayakumar, Grzegorz Swirszcz, and Max Jaderberg. Distilling policy distillation. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 1331–1340, 2019.

Tri Dao, Govinda M. Kamath, Vasilis Syrgkanis, and Lester Mackey. Knowledge distillation as semiparametric inference. In International Conference on Learning Representations, 2021.

DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, 2025. arXiv preprint arXiv:2501.12948.

Dylan J. Foster, Adam Block, and Dipendra Misra. Is behavior cloning all you need? understanding horizon in imitation learning. In Advances in Neural Information Processing Systems, volume 37, pages 120602–120666, 2024.

Dylan J. Foster, Zakaria Mhammedi, and Dhruv Rohatgi. Is a good foundation necessary for eficient reinforcement learning? the computational role of the base model in exploration. In Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 2026–2142, 2025.

Jiawei Ge, Shange Tang, Jianqing Fan, Cong Ma, and Chi Jin. Maximum likelihood estimation is all you need for well-specified covariate shift. In The Twelfth International Conference on Learning Representations, 2024.

Gemini Team. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024.

Srishti Gureja, Lester James V. Miranda, Shayekh Bin Islam, Rishabh Maheshwary, Drishti Sharma, Gusti Winata, Nathan Lambert, Sebastian Ruder, Sara Hooker, and Marzieh Fadaee. M-RewardBench: Evaluating reward models in multilingual settings. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 43–58, 2025.

Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network, 2015. arXiv preprint arXiv:1503.02531.

Haichen Hu and David Simchi-Levi. Perturbing the derivative: Doubly wild refitting for model-free evaluation of opaque machine learning predictors, 2025a.

Haichen Hu and David Simchi-Levi. Perturbing the derivative: Wild refitting for model-free evaluation of machine learning models under bregman losses, 2025b.

Haichen Hu and David Simchi-Levi. Interleaved resampling and refitting: Data and computeeficient evaluation of black-box predictors. arXiv preprint arXiv:2603.14218, 2026.

Haichen Hu, Jian Qian, and David Simchi-Levi. Model-based reinforcement learning with double oracle eficiency in policy optimization and ofline estimation, 2026.

Audrey Huang, Adam Block, Dylan J. Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T. Ash, and Akshay Krishnamurthy. Self-improvement in language models: The sharpening mechanism. In The Thirteenth International Conference on Learning Representations, 2025.

Muhammed Emrullah Ildiz, Halil Alperen Gozeten, Ege Onur Taga, Marco Mondelli, and Samet Oymak. High-dimensional analysis of knowledge distillation: Weak-to-strong generalization and scaling laws. In The Thirteenth International Conference on Learning Representations, 2025.

Fotis Iliopoulos, Vasilis Kontonis, Cenk Baykal, Gaurav Menghani, Khoa Trinh, and Erik Vee. Weighted distillation with unlabeled examples. In Advances in Neural Information Processing Systems, volume 35, 2022.

Zeyu Jia, Alexander Rakhlin, and Tengyang Xie. Do we need to verify step by step? rethinking process supervision from a theoretical perspective. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 27373–27398, 2025.

Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E. Schapire. Contextual decision processes with low Bellman rank are PAC-learnable. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1704–1713, 2017.

Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably eficient reinforcement learning with linear function approximation. In Conference on learning theory, pages 2137–2143. PMLR, 2020.

Juno Kim, Tai Nakamaki, and Taiji Suzuki. Transformers are minimax optimal nonparametric in-context learners. In Advances in Neural Information Processing Systems, volume 37, pages 106667–106713, 2024a.

Juno Kim, Jihun Yun, Jason D. Lee, and Kwang-Sung Jun. Coverage improvement and fast convergence of on-policy preference learning. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing finegrained evaluation capability in language models. In International Conference on Learning Representations, 2024b.

Qi Lei, Wei Hu, and Jason D. Lee. Near-optimal linear regression under distribution shift. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 6164–6174, 2021.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.

Michal Lukasik, Srinadh Bhojanapalli, Aditya Krishna Menon, and Sanjiv Kumar. Teacher’s pet: understanding and mitigating biases in distillation. Transactions on Machine Learning Research, 2022.

Cong Ma, Reese Pathak, and Martin J. Wainwright. Optimally tackling covariate shift in RKHSbased nonparametric regression. The Annals of Statistics, 51(2):738–761, 2023.

Aditya Krishna Menon, Ankit Singh Rawat, Sashank J. Reddi, Seungyeon Kim, and Sanjiv Kumar. A statistical perspective on distillation. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 7632–7642, 2021.

OpenAI. GPT-4 technical report, 2023. arXiv preprint arXiv:2303.08774.

Jian Qian, Haichen Hu, and David Simchi-Levi. Ofline oracle-eficient learning for contextual mdps via layerwise exploration-exploitation tradeof. arXiv preprint arXiv:2405.17796, 2024.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Stephane Ross, Geofrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 627–635, 2011.

Dmitri Roussinov, Serge Sharof, and Nadezhda Puchnina. Controlling out-of-domain gaps in LLMs for genre classification and generated text detection. In Proceedings of the 31st International Conference on Computational Linguistics, pages 3329–3344, 2025.

Nikunj Saunshi, Sadhika Malladi, and Sanjeev Arora. A mathematical exploration of why language models help solve downstream tasks. In International Conference on Learning Representations, 2021.

Ved Sriraman and Adam Block. Revisiting the (sub)optimality of best-of-N for inference-time alignment. In Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 5980–6028, 2026.

Ved Sriraman, Peihan Liu, Daniel Hsu, and Adam Block. Behavior cloning is not all you need: The optimality of on-policy distillation for noisy expert feedback, 2026.

Martin J Wainwright. Wild refitting for black box prediction. arXiv preprint arXiv:2506.21460, 2025.

Kaizheng Wang. Pseudo-labeling for kernel ridge regression under covariate shift. The Annals of Statistics, 54(1):252–276, 2026.

Nathan Weill and Kaizheng Wang. Pseudo-labeling for unsupervised domain adaptation with kernel GLMs, 2026.

Eric Xia and Jason M. Klusowski. Classification imbalance as transfer learning, 2026.

Eric Xia and Martin J. Wainwright. Prediction aided by surrogate training, 2024.

Audrey Xie, Ludwig Schmidt, and John Duchi. Two mathematical models of knowledge distillation. In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 4726–4734, 2026.

Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent pessimism for ofline reinforcement learning. In Advances in Neural Information Processing Systems, volume 34, pages 6683–6694, 2021.

Tengyang Xie, Dylan J. Foster, Yu Bai, Nan Jiang, and Sham M. Kakade. The role of coverage in online reinforcement learning. In The Eleventh International Conference on Learning Representations, 2023.

Tengyang Xie, Dylan J. Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed H. Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit Q\*-approximation for sample-eficient RLHF. In The Thirteenth International Conference on Learning Representations, 2025.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 54715–54754, 2024.

Kakei Yamamoto and Martin J. Wainwright. Residual-as-teacher: Mitigating bias propagation in student–teacher estimation, 2026.

Chenlu Ye, Wei Xiong, Yuheng Zhang, Hanze Dong, Nan Jiang, and Tong Zhang. Online iterative reinforcement learning from human feedback with general preference model. In Advances in Neural Information Processing Systems, volume 37, 2024.

Zhuohao Yu, Zhiwei Steven Wu, and Adam Block. From curiosity to caution: Mitigating reward hacking for best-of-N with pessimism. In The Fourteenth International Conference on Learning Representations, 2026.

Yurun Yuan, Fan Chen, Zeyu Jia, Alexander Rakhlin, and Tengyang Xie. Trajectory Bellman residual minimization: A simple value-based method for LLM reasoning. In Advances in Neural Information Processing Systems, volume 38, 2025.

Huaqing Zhang, Jingchu Gai, Juno Kim, Bingbin Liu, and Andrej Risteski. When does online imitation learning help in LLM post-training? the role of (non-)realizability beyond horizon, 2026.

Yuheng Zhang, Yu Bai, and Nan Jiang. Ofline learning in Markov games with general function approximation. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 40804–40829, 2023.

Heyang Zhao, Chenlu Ye, Quanquan Gu, and Tong Zhang. Sharp analysis for KL-regularized contextual bandits and RLHF. In Advances in Neural Information Processing Systems, volume 38, 2025.

Banghua Zhu, Michael I. Jordan, and Jiantao Jiao. Principled reinforcement learning with human feedback from pairwise or K-wise comparisons. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 43037–43067, 2023.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. JudgeLM: Fine-tuned large language models are scalable judges. In International Conference on Learning Representations, 2025.

## A Mathematical Tools

Lemma A.1 ( Lojasiewicz inequality Bierstone and Milman (1988)). Let $K \subseteq \mathbb { R } ^ { q }$ be nonempty, and let $f , g : K \to \mathbb { R }$ . Suppose that their graphs are compact subanalytic subsets of $\mathbb { R } ^ { q + 1 }$ and that

$$
\{ x \in K : f ( x ) = 0 \} \subseteq \{ x \in K : g ( x ) = 0 \} .
$$

Then there are constants $c > 0$ and $\rho > 0$ for which

$$
| f ( x ) | \geq c | g ( x ) | ^ { \rho } \ ( x \in K ) .
$$

In particular, if $f , g \ge 0$ on $K$ , we choose any $M \geq \operatorname* { m a x } \{ 1 , \operatorname* { s u p } _ { x \in K } g ( x ) \}$ . Then, the same constants $c , \rho$ defined above yield

$$
f ( x ) \geq a [ g ( x ) ] ^ { p } , p : = \operatorname* { m a x } \{ 1 , \rho \} \geq 1 , a : = c M ^ { \rho - p } > 0 .
$$

To quantify the progress of a projected student step near the oracle, we use the following subgradient form of the Lojasiewicz inequality. The domain in this result can include the boundary of the student parameter ball.

Lemma A.2 (Bolte et al., 2007, Theorem 3.1). Let $f : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ have a subanalytic graph and closed domain, and suppose that $f$ is continuous on its domain. If θ<sup>¯</sup> is a critical point, meaning $0 \in \partial f ( { \bar { \theta } } )$ , then there are a neighborhood U of $\bar { \theta } , c > 0$ , and $\rho \in [ 0 , 1 )$ such that

$$
\mathrm { d i s t } ( 0 , \partial { f ( \theta ) } ) \geq c | f ( \theta ) - f ( { \bar { \theta } } ) | ^ { \rho } \mathrm { f o r } \theta \in U \cap \mathrm { d o m } f \mathrm { w i t h } f ( \theta ) \neq f ( { \bar { \theta } } ) .
$$

Here $\partial f$ is the limiting subdiferential and dist $( 0 , \partial f ( \theta ) ) : = \operatorname* { i n f } _ { v \in \partial f ( \theta ) } \| v \| _ { 2 }$

## B Proofs in Section 5

In this appendix, we provide the proof of Theorem 5.1. Specifically, we will first provide a sequence of lemmas that are useful and finally, we will show how to combine them together to prove the main convergence rate theorem.

We begin by identifying the correct distillation objective: the following lemma shows that maximizing the regularized student return is equivalent to minimizing the KL cost against $\pi _ { \boldsymbol { w } _ { \lambda } ^ { \star } }$

Lemma B.1. For each source or target prompt x, define

$$
Z _ { \lambda } ( x ) : = \sum _ { b _ { 1 : H } \in { \cal A } ( x ) } \pi _ { \mathrm { p r e } } ( b _ { 1 : H } \mid x ) e ^ { R ( x , b _ { 1 : H } ) / \lambda } .
$$

The unique unrestricted optimal answer policy is

$$
\pi _ { \lambda } ^ { \star } ( a _ { 1 : H } \mid x ) = \frac { \pi _ { \mathrm { p r e } } ( a _ { 1 : H } \mid x ) e ^ { R ( x , a _ { 1 : H } ) / \lambda } } { Z _ { \lambda } ( x ) } .\tag{B.1}
$$

Moreover, for every student parameter,

$$
J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta } ) = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \log Z _ { \lambda } ( \widetilde { x } _ { j } ) - C _ { w _ { \lambda } ^ { \star } } ( \theta ) .\tag{B.2}
$$

The Gibbs representation in Lemma B.1 lets us identify the distribution of accepted branch indices. The following lemma shows that these indices form logistic observations with parameter $w _ { \lambda } ^ { \star } ,$ providing the statistical basis for the source update.

We now define a filtration $\{ \mathcal { F } _ { t } \} _ { t = 1 } ^ { T }$ . We treat the source and target datasets, the reward oracle, the policy features, the frozen policies, and the algorithmic schedules as fixed. Let $\mathcal { F } _ { t }$ denote the information available immediately before the source sampling in round t. Formally, set $\mathcal { F } _ { 0 } =$ $\sigma ( w _ { 0 } , \theta _ { 0 } )$ . For every $t > 0$ , we define $\mathcal { F } _ { t }$ recursively

$$
\begin{array} { r } { \mathcal { F } _ { t + 1 } = \mathcal { F } _ { t } \vee \sigma \left( i _ { t } , h _ { t } , a _ { t , 1 : H } , c _ { t , 0 } , Y _ { t } , a _ { t , 1 : H } ^ { \mathrm { p r e } } , U _ { t } , \left( j _ { t + 1 , \ell } ^ { \sharp } , a _ { t + 1 , \ell , 1 : H } ^ { \sharp } \right) _ { \ell = 1 } ^ { h _ { t + 1 } } , \vartheta _ { t + 1 , 2 , \cdot } \left( j _ { t + 1 , k , \ell } , a _ { t + 1 , k , \ell , 1 : H } ^ { \mathrm { z d l } } \right) _ { \ell = 1 , \ldots , q _ { t + 1 } } \right) . } \end{array}
$$

The three lines collect the source samples, the student-gradient batch, and the exploration candidate and candidate-validation samples, respectively.

All other quantities computed in round t, including $s _ { t } , c _ { t , 1 } , z _ { t } , R _ { t } , I _ { t } , g _ { t }$ , the gradient candidate $\vartheta _ { t + 1 , 1 }$ , and the selected student, are measurable functions of $\mathcal { F } _ { t }$ and these samples. In particular, $w _ { t }$ and $\theta _ { t }$ are $\mathcal { F } _ { t } .$ -measurable, whereas $w _ { t + 1 }$ and $\theta _ { t + 1 }$ are $\mathcal { F } _ { t + 1 }$ -measurable. For deterministic initialization, $\mathcal { F } _ { 0 }$ is the trivial sigma-algebra. Throughout the following lemmas, conditioning on $\mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 }$ means conditioning on $\mathcal { F } _ { t } \vee \sigma ( s _ { t } , c _ { t , 1 } , c _ { t , 0 } )$ . Then, we have the following lemma.

Lemma B.2. Under Assumption 3.2, for every round $t \geq 0$ , Algorithm 1 satisfies, almost surely,

$$
\operatorname* { P r } ( I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) \geq e ^ { - 1 / \lambda } .\tag{B.3}
$$

Moreover, conditional on acceptance, the branch index satisfies

$$
\operatorname* { P r } ( Y _ { t } = 1 | I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) = \frac { \pi _ { \lambda } ^ { \star } ( c _ { t , 1 } \mid s _ { t } ) } { \pi _ { \lambda } ^ { \star } ( c _ { t , 1 } \mid s _ { t } ) + \pi _ { \lambda } ^ { \star } ( c _ { t , 0 } \mid s _ { t } ) } = \sigma ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } ) ,\tag{B.4}
$$

where

$$
z _ { t } = \phi ( s _ { t } , c _ { t , 1 } ) - \phi ( s _ { t } , c _ { t , 0 } ) , \ \sigma ( u ) = \frac { 1 } { 1 + e ^ { - u } } .
$$

These statements also hold when the two proposal tokens coincide, including prefixes after EOS.   
The random variable $Y _ { t }$ denotes the branch index, not the token value.

Combining the accepted-label identity in Lemma B.2 with joint source identification, we obtain convergence of the calibration updates despite the adaptive student proposals. The bounds below will control both the calibration error and the changes in the target objective across rounds.

Lemma B.3. With $G _ { \mathrm { j o i n t } }$ and $\gamma$ defined in (3.7)-(4.2), every round $t \geq 0$ satisfies

$$
\mathbb { E } \left[ z _ { t } z _ { t } ^ { \top } \mid \mathcal { F } _ { t } \right] \succeq e ^ { - 2 B } G _ { \mathrm { j o i n t } } , \| z _ { t } \| _ { 2 } \leq 2 ,\tag{B.5}
$$

$$
\begin{array} { r } { \mathbb { E } \Big [ \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) ^ { \top } g _ { t } \Big | \mathcal F _ { t } \Big ] \geq \gamma \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } , \ \| g _ { t } \| _ { 2 } \leq 2 . } \end{array}\tag{B.6}
$$

Consequently, for every integer $t \geq 1$ , the projected updates in Algorithm 1 satisfies

$$
\mathbb { E } \Big [ \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } \Big ] \leq \frac { 4 } { \gamma ^ { 2 } ( t + 2 ) } ,\tag{B.7}
$$

$$
\| w _ { t } - w _ { t - 1 } \| _ { 2 } \leq \frac { 2 } { \gamma ( t + 1 ) } .\tag{B.8}
$$

Having controlled the source updates, we turn to the quantities estimated from target rollouts. We first bound the log-ratio cost of a single answer, which will be used to control the population cost and the error in its Monte Carlo evaluation.

Lemma B.4. For $t = 0 , \ldots , T , w _ { t } \in W , \vartheta \in \Theta$ , and a feasible answer $a _ { 1 : H } \in \mathcal { A } ( x )$ at a target prompt x, define

$$
Z _ { t } ( \vartheta ; x , a _ { 1 : H } ) : = \lambda \log \frac { \pi _ { \mathrm { s t u } , \vartheta } \left( a _ { 1 : H } \mid x \right) } { \pi _ { w _ { t } } \left( a _ { 1 : H } \mid x \right) } = \lambda \sum _ { h = 1 } ^ { H } \log \frac { \pi _ { \mathrm { s t u } , \vartheta } \left( a _ { h } \mid x , a _ { 1 : h - 1 } \right) } { \pi _ { w _ { t } } \left( a _ { h } \mid x , a _ { 1 : h - 1 } \right) } .
$$

Then, uniformly over these choices,

$$
| Z _ { t } ( \vartheta ; x , a _ { 1 : H } ) | \le 4 \lambda B H .\tag{B.9}
$$

The preceding trajectory bound gives a uniform bound on its expected cost. We also quantify sensitivity to the student and calibration parameters, so that parameter changes can be translated into cost changes in the optimization analysis.

Lemma B.5. For $w \in W$ and $\theta \in \Theta ,$ define $\begin{array} { r } { C _ { w } ( \theta ) : = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } \big ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid \widetilde { x } _ { j } ) \mid \mid \pi _ { w } ( \cdot \mid \widetilde { x } _ { j } ) \big ) . } \end{array}$

The cost used in round t is $\begin{array} { r } { C _ { t } ( \theta ) : = C _ { w _ { t } } ( \theta ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } [ Z _ { t } ( \theta ; \widetilde { x } _ { j } , a _ { 1 : H } ) ] } \end{array}$ , where $Z _ { t }$ is

defined in Lemma B.4. For every $w , v \in W$ and $\theta , \vartheta \in \Theta$ ,

$$
0 \leq C _ { w } ( \theta ) \leq 4 \lambda B H ,\tag{B.10}
$$

$$
| C _ { w } ( \theta ) - C _ { w } ( \vartheta ) | \leq 4 \lambda B H ^ { 3 / 2 } \| \theta - \vartheta \| _ { 2 } ,\tag{B.11}
$$

$$
\operatorname* { s u p } _ { \theta \in \Theta } | C _ { w } ( \theta ) - C _ { v } ( \theta ) | \leq 2 \lambda H \| w - v \| _ { 2 } .\tag{B.12}
$$

In particular, these bounds apply to $C _ { t }$ by setting $w = w _ { t }$

For the newly calibrated objective $C _ { t + 1 }$ , we next verify that the algorithm’s fresh target batch provides an unbiased gradient estimate. We also bound its variance, which controls the error in the projected-gradient proposal.

Lemma B.6. Let $\mathcal { F } _ { t }$ be the history before the source draws in round $t ,$ so that $\theta _ { t }$ is $\mathcal { F } _ { t } .$ -measurable, and set $\mathcal G _ { t + 1 } : = \mathcal F _ { t } \vee \sigma ( w _ { t + 1 } )$ . Define the score

$$
\begin{array} { r } { S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) : = \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) . } \end{array}
$$

Conditional on $\mathcal { G } _ { t + 1 }$ , independently for $\ell = 1 , \dots , b _ { t + 1 }$ , draw

$$
j _ { t + 1 , \ell } ^ { \mathrm { g } } \sim \mathrm { U n i f } \{ 1 , \dots , m \} , \ a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } \sim \pi _ { \mathrm { s t u } , \theta _ { t } } ( \cdot \mid \widetilde { x } _ { j _ { t + 1 , \ell } ^ { \mathrm { g } } } ) ,
$$

and define

$$
\widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } : = S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j _ { t + 1 } ^ { \mathrm { g } } , \ell } , a _ { t + 1 , \ell , 1 ; H } ^ { \mathrm { g } } ) Z _ { t + 1 } ( \theta _ { t } ; \widetilde { x } _ { j _ { t + 1 , \ell } ^ { \mathrm { g } } } , a _ { t + 1 , \ell , 1 ; H } ^ { \mathrm { g } } ) , \ \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } : = \frac { 1 } { b _ { t + 1 } } \sum _ { \ell = 1 } ^ { b _ { t + 1 } } \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } .
$$

Here $Z _ { t + 1 }$ and $C _ { t + 1 } = C _ { w _ { t + 1 } }$ are defined in Lemmas B.4 and B.5, respectively. For every $t =$ $0 , \ldots , T - 1$ , almost surely, we have

$$
\begin{array} { r } { \mathbb { E } \Big [ \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \Big | \mathcal { G } _ { t + 1 } \Big ] = \mathbb { E } \Big [ \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \Big | w _ { t + 1 } , \theta _ { t } \Big ] = \nabla _ { \theta } C _ { t + 1 } \big ( \theta _ { t } \big ) , } \end{array}\tag{B.13}
$$

$$
\mathbb { E } \bigg [ \Big \| \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } - \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) \Big \| _ { 2 } ^ { 2 } \bigg | \mathcal { G } _ { t + 1 } \bigg ] \leq \frac { 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } } { b _ { t + 1 } } .\tag{B.14}
$$

The preceding bounds control the size of the student gradient. We next bound its variation, both as the student parameter changes and as the calibrated teacher changes. These bounds determine a fixed student step size and control the error caused by using the current calibration.

Lemma B.7. Recall the cost $C _ { w }$ in (3.5) and the oracle-policy average KL divergence $\kappa _ { \lambda , m }$ in (3.6). For any answer laws $Q _ { j }$ that are positive on $\mathcal { A } ( \widetilde { x } _ { j } )$ , the function $\begin{array} { r } { \theta \mapsto m ^ { - 1 } \sum _ { j = 1 } ^ { m } \operatorname { K L } ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \ | } \end{array}$ $\widetilde { x } _ { j } ) \lvert \lvert Q _ { j } )$ is real analytic on $\mathbb { R } ^ { d }$ . In particular, this holds for $C _ { w }$ and $\kappa _ { \lambda , m }$ . Define

$$
L _ { \mathrm { s t } } : = \lambda H ( 1 + 8 B H ) , ~ \alpha _ { \mathrm { s t u } } : = \frac { 1 } { 2 L _ { \mathrm { s t } } } .
$$

For all $w , v \in W$ and $\theta , \vartheta \in \Theta$ , we have

$$
\| \nabla _ { \theta } ^ { 2 } C _ { w } ( \theta ) \| _ { \mathrm { o p } } \leq L _ { \mathrm { s t } } ,\tag{B.15}
$$

$$
\| \nabla _ { \theta } C _ { w } ( \theta ) - \nabla _ { \theta } C _ { w } ( \vartheta ) \| _ { 2 } \leq L _ { \mathrm { s t } } \| \theta - \vartheta \| _ { 2 } ,\tag{B.16}
$$

$$
\begin{array} { r } { \| \nabla _ { \theta } C _ { w } ( \theta ) - \nabla _ { \theta } C _ { v } ( \theta ) \| _ { 2 } \le 2 \lambda H ^ { 3 / 2 } \| w - v \| _ { 2 } . } \end{array}\tag{B.17}
$$

After constructing the candidates, the algorithm chooses among them using sampled costs. The uniform log-ratio bound in Lemma B.4 allows the following lemma to control how far the selected candidate’s true cost can exceed the better of the two proposals.

Lemma B.8. For each target update $t \geq 1$ , define $\mathcal { H } _ { t } : = \mathcal { F } _ { t - 1 } \vee \sigma ( w _ { t } , \vartheta _ { t , 1 } , \vartheta _ { t , 2 } )$ . Define the nonnegative, proof-only comparison error $\begin{array} { r } { \xi _ { t } : = 2 \operatorname* { m a x } _ { k \in \{ 1 , 2 \} } | \widehat { C } _ { t , k } - C _ { t } ( \vartheta _ { t , k } ) | } \end{array}$ . Then, we have

$$
C _ { t } ( \theta _ { t } ) \leq \operatorname* { m i n } _ { k \in \{ 1 , 2 \} } C _ { t } ( \vartheta _ { t , k } ) + \xi _ { t } , \mathbb { E } [ \xi _ { t } \mid \mathcal { H } _ { t } ] \leq \frac { 1 6 \lambda B H } { t + 1 } .\tag{B.18}
$$

The next lemma combines two sources of improvement for the fixed true cost. A projected gradient step provides progress near the oracle; uniform exploration provides progress when the current cost is bounded away from the minimum. The exact true gradient is used only to define the comparison step in this lemma.

Lemma B.9. Let $F ( \theta ) : = C _ { w _ { \lambda } ^ { \star } } ( \theta )$ and $\Delta _ { \lambda , m } ( \theta ) : = F ( \theta ) - F ( \theta _ { \lambda , m } ^ { \dagger } )$ . Define

$$
y ( \theta ) : = \mathrm { P r o j } _ { \Theta } \left( \theta - \alpha _ { \mathrm { { s t u } } } \nabla _ { \theta } F ( \theta ) \right) , \vartheta ^ { \mathrm { u n i f } } \sim \mathrm { U n i f } ( \Theta ) , M _ { \mathrm { o p t } } : = 8 \lambda B ^ { 2 } H ^ { 3 / 2 } .
$$

Under Assumption 3.3, there is a fixed-model constant $0 < \kappa _ { \mathrm { o p t } } \leq 1 / ( 4 M _ { \mathrm { o p t } } )$ such that, for every $\theta \in \Theta$

$$
\begin{array} { r } { \mathbb { E } _ { \vartheta ^ { \mathrm { u n i f } } } \Big [ F ( \theta ) - \operatorname* { m i n } \Big \{ F ( y ( \theta ) ) , F ( \vartheta ^ { \mathrm { u n i f } } ) \Big \} \Big ] \geq \kappa _ { \mathrm { o p t } } [ \Delta _ { \lambda , m } ( \theta ) ] ^ { 2 } . } \end{array}\tag{B.19}
$$

The constant does not depend on the iteration number.

We now compare the actual student update with the population update in Lemma B.9. The preceding gradient and selection bounds control the diference between these updates. This gives a recursion for the excess true cost, with separate contributions from gradient estimation, calibration, and validation.

Lemma B.10. Recall $\Delta _ { \lambda , m } ( \theta )$ and $\varepsilon _ { t }$ from (3.6). Let $\kappa _ { \mathrm { o p t } } > 0$ be the constant in Lemma B.9, chosen so that $\kappa _ { \mathrm { o p t } } \leq 1 / ( 4 M _ { \mathrm { o p t } } )$ , where $M _ { \mathrm { o p t } } : = 8 \lambda B ^ { 2 } H ^ { 3 / 2 }$ . Define

$$
E _ { \mathrm { o p t } } : = 1 6 \alpha _ { \mathrm { s t u } } \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } + \frac { 1 6 \alpha _ { \mathrm { s t u } } \lambda ^ { 2 } B H ^ { 3 } + 8 \lambda H } { \gamma } + 1 6 \lambda B H ,
$$

$$
K _ { \mathrm { o p t } } : = \operatorname* { m a x } \left\{ M _ { \mathrm { o p t } } , \sqrt { \frac { 2 E _ { \mathrm { o p t } } } { \kappa _ { \mathrm { o p t } } } } , \frac { 1 } { 2 \kappa _ { \mathrm { o p t } } } \right\} .
$$

Then Algorithm 1 satisfies, for every integer $T \geq 1$

$$
\mathbb { E } [ \Delta _ { \lambda , m } ( \theta _ { T } ) ] \le K _ { \mathrm { o p t } } ( T + 1 ) ^ { - 1 / 4 } , \ \mathbb { E } [ \varepsilon _ { T } ] \le K _ { \mathrm { o p t } } ( T + 1 ) ^ { - 1 / 4 } + \frac { 8 \lambda H } { \gamma \sqrt { T + 2 } } .\tag{B.20}
$$

Lemma B.10 controls the expected excess true cost, whereas Theorem 5.1 concerns KL divergence

to the oracle student. Under uniqueness of the optimal student policy, the following lemma uses analyticity and compactness to connect these two errors.

Lemma B.11. Under the model setup and Assumption 3.3, fix $\theta _ { \lambda , m } ^ { \dagger } \in \mathrm { a r g m i n } _ { \theta \in \Theta } C _ { w _ { \lambda } ^ { \star } } ( \theta )$ . Recall the excess true cost and the average oracle-policy KL from (3.6):

$$
\begin{array} { r l } & { \Delta _ { \lambda , m } ( \theta ) : = C _ { w _ { \lambda } ^ { \star } } ( \theta ) - C _ { w _ { \lambda } ^ { \star } } ( \theta _ { \lambda , m } ^ { \dagger } ) , } \\ & { \mathcal { K } _ { \lambda , m } ( \theta ) : = \cfrac { 1 } { m } \displaystyle \sum _ { j = 1 } ^ { m } \mathrm { K L } \Bigl ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid \widetilde { x } _ { j } ) \left\| \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \mid \widetilde { x } _ { j } ) \right) . } \end{array}
$$

There exist constants $a _ { \lambda , m } > 0$ and $p _ { \lambda , m } \ge 1$ such that

$$
\Delta _ { \lambda , m } ( \theta ) \geq a _ { \lambda , m } [ \mathcal { K } _ { \lambda , m } ( \theta ) ] ^ { p _ { \lambda , m } } \mathrm { ~ f o r ~ e v e r y ~ } \theta \in \Theta .\tag{B.21}
$$

These constants depend on the fixed model and target design, but not on the iteration number.

Combining Lemmas B.10 and B.11 through Jensen’s inequality now yields the convergence rate in Theorem 5.1, as shown in the proof below.

Proof of Theorem 5.1. Starting from the left-hand side of (5.1), the definition in (3.6) gives

$$
\mathbb { E } [ \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta _ { T } } ( \cdot \mid \widetilde { x } _ { j } ) \| \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \mid \widetilde { x } _ { j } ) ) ] = \mathbb { E } [ K _ { \lambda , m } ( \theta _ { T } ) ] .
$$

We first apply Lemma B.11 to compare this policy error with the excess true cost. Its inequality holds for every $\theta \in \Theta$ , and hence on every sample path at $\theta _ { T }$ :

$$
a _ { \lambda , m } [ \mathcal { K } _ { \lambda , m } ( \theta _ { T } ) ] ^ { p _ { \lambda , m } } \leq \Delta _ { \lambda , m } ( \theta _ { T } ) .
$$

Dividing by $a _ { \lambda , m } > 0$ and taking the increasing power $1 / p _ { \lambda , m }$ yields

$$
\begin{array} { r } { \mathcal { K } _ { \lambda , m } ( \theta _ { T } ) \leq a _ { \lambda , m } ^ { - 1 / p _ { \lambda , m } } [ \Delta _ { \lambda , m } ( \theta _ { T } ) ] ^ { 1 / p _ { \lambda , m } } . } \end{array}
$$

The excess cost is bounded by Lemma B.5, so the expectations below are finite. Since $p _ { \lambda , m } \ge 1$ ， the function $u \mapsto u ^ { 1 / p _ { \lambda , m } }$ is concave: for $u > 0$

$$
\frac { d ^ { 2 } } { d u ^ { 2 } } u ^ { 1 / p _ { \lambda , m } } = \frac { 1 } { p _ { \lambda , m } } \left( \frac { 1 } { p _ { \lambda , m } } - 1 \right) u ^ { 1 / p _ { \lambda , m } - 2 } \leq 0 ,
$$

and continuity extends concavity to zero. Taking expectations and applying Jensen’s inequality therefore gives

$$
\mathbb { E } [ { \bar { K } } _ { \lambda , m } ( \theta _ { T } ) ] \leq a _ { \lambda , m } ^ { - 1 / p _ { \lambda , m } } \mathbb { E } \left[ [ \Delta _ { \lambda , m } ( \theta _ { T } ) ] ^ { 1 / p _ { \lambda , m } } \right] \leq a _ { \lambda , m } ^ { - 1 / p _ { \lambda , m } } \left( \mathbb { E } [ \Delta _ { \lambda , m } ( \theta _ { T } ) ] \right) ^ { 1 / p _ { \lambda , m } } = \left( \frac { \mathbb { E } [ \Delta _ { \lambda , m } ( \theta _ { T } ) ] } { a _ { \lambda , m } } \right) ^ { 1 / p _ { \lambda , m } } .
$$

Next, we use Lemma B.10 to bound the numerator by $K _ { \mathrm { o p t } } ( T + 1 ) ^ { - 1 / 4 }$ . This bound already combines the student gradient progress, calibration error, and finite-rollout errors. Substituting it

into the increasing power gives

$$
\mathbb { E } [ K _ { \lambda , m } ( \theta _ { T } ) ] \leq \left( \frac { K _ { \mathrm { o p t } } ( T + 1 ) ^ { - 1 / 4 } } { a _ { \lambda , m } } \right) ^ { 1 / p _ { \lambda , m } } = \left( \frac { K _ { \mathrm { o p t } } } { a _ { \lambda , m } } \right) ^ { 1 / p _ { \lambda , m } } ( T + 1 ) ^ { - 1 / ( 4 p _ { \lambda , m } ) } .
$$

Taking $C _ { \lambda , m } : = ( K _ { \mathrm { o p t } } / a _ { \lambda , m } ) ^ { 1 / p _ { \lambda , m } }$ proves (5.1). All constants are independent of $T .$ , and $p _ { \lambda , m } \ge 1$ is finite, so the bound converges to zero. □

## C Proofs in Appendix B

In this appendix, we provide the proofs of the lemmas in Appendix B

Proof of Lemma B.1. We fix x first and call the right-hand side of (B.1) as $\bar { \pi } _ { \lambda } ( \boldsymbol { a } _ { 1 : H } \mid \boldsymbol { x } )$

Since $R \in [ 0 , 1 ]$ and the pre-trained policy $\pi _ { p r e }$ is a probability law, we have

$$
1 \leq e ^ { R ( x , a _ { 1 : H } ) / \lambda } \leq e ^ { 1 / \lambda } \implies 1 \leq Z _ { \lambda } ( x ) \leq e ^ { 1 / \lambda } .
$$

Thus, $\bar { \pi } _ { \lambda }$ is positive and sums to one. By the definition of an autoregressive policy, we have that

$$
\pi _ { \lambda } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) = { \frac { \bar { \pi } _ { \lambda } ( a _ { 1 : h } \mid x ) } { \bar { \pi } _ { \lambda } ( a _ { 1 : h - 1 } \mid x ) } } .
$$

Thus, $\bar { \pi } _ { \lambda }$ is an admissible autoregressive policy.

For any competing policy π, by the autoregressive product in (3.1) gives, we have that

$$
\sum _ { h = 1 } ^ { H } \log { \frac { \pi ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } { \pi _ { \mathrm { p r e } } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) } } = \log { \frac { \pi ( a _ { 1 : H } \mid x ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : H } \mid x ) } } .\tag{C.1}
$$

Recall that $\begin{array} { r } { \bar { \pi } _ { \lambda } \bigl ( a _ { 1 : H } | x \bigr ) = \frac { \pi _ { \mathrm { p r e } } ( a _ { 1 : H } | x ) e ^ { R ( x , a _ { 1 : H } ) / \lambda } } { Z _ { \lambda } ( x ) } } \end{array}$ . Taking the logarithm on both sides and rearrange, we obtain

$$
\begin{array} { r } { R ( x , a _ { 1 : H } ) = \lambda \log \frac { \bar { \pi } _ { \lambda } \left( a _ { 1 : H } \mid x \right) } { \pi _ { \mathrm { p r e } } \left( a _ { 1 : H } \mid x \right) } + \lambda \log Z _ { \lambda } ( x ) . } \end{array}\tag{C.2}
$$

Theerfore, we Subtract (C.1) from both sides of (C.2) and then take expectation to get

$$
\begin{array} { r } { \mathbb { E } _ { a _ { 1 : H } \sim \pi \left( \cdot \vert x \rangle \right)} \left[ R ( x , a _ { 1 : H } ) - \lambda \displaystyle \sum _ { h = 1 } ^ { H } \log \frac { \pi \left( a _ { h } \mid x , a _ { 1 : h - 1 } \right) } { \pi _ { \mathrm { p r e } } \left( a _ { h } \mid x , a _ { 1 : h - 1 } \right) } \right] = \lambda \log Z _ { \lambda } ( x ) - \lambda \mathbb { E } _ { a _ { 1 : H } \sim \pi ( \cdot \vert x \rangle } \left[ \log \frac { \pi \left( a _ { 1 : H } \mid x \right) } { \overline { \pi } _ { \lambda } \left( a _ { 1 : H } \mid x \right) } \right] } \\ { = \lambda \log Z _ { \lambda } ( x ) - \lambda \mathrm { K L } ( \pi ( \cdot \vert x \rangle \| \overline { \pi } _ { \lambda } ( \cdot \vert x )  . } \end{array}
$$

The second is by the definition of KL.

The KL divergence is nonnegative and $K L ( p | | p ) = 0$ , Thus, the preceding objective is uniquely maximized by $\pi = \bar { \pi } _ { \lambda }$ , which proves (B.1). Average the identity over the target prompts and substitute $\pi _ { \lambda } ^ { \star } = \pi _ { w _ { \lambda } ^ { \star } }$ from Assumption 3.2. This proves (B.2) and the equivalence of the original regularized benchmark and the minimum of $C _ { w _ { \lambda } ^ { \star } }$ . The unknown target rewards appear in the analysis, not in an algorithmic query. □

Let $\mathcal { F } _ { t }$ be the history immediately before the source step of round $t ,$ containing all source and target random draws from rounds $0 , \ldots , t - 1$ and the initial parameters. $\mathrm { A t } ~ t = ~ 0$ , there are no preceding rounds. Thus $w _ { t }$ and $\theta _ { t }$ are $\mathcal { F } _ { t } .$ -measurable. The source draws in round t are fresh conditional on this history. In particular, the student may depend on every preceding calibration update, exploration draw, and target evaluation.

Proof of Lemma B.2. We show that the acceptance rule reweights the two branch indices by exactly the factors appearing in the regularized optimal policy.

Fix the history $\mathcal { F } _ { t }$ , the selected prefix $s _ { t } = ( x _ { i _ { t } } , a _ { t , 1 : h _ { t } - 1 } )$ , and the two legal proposal tokens $c _ { t , 1 }$ and $c _ { t , 0 }$ . Here $\mathcal { F } _ { t }$ is the history before round t. After additionally conditioning on $Y _ { t } = k$ , the first $h _ { t }$ tokens are fixed:

$$
a _ { t , 1 : h _ { t } } ^ { \mathrm { p r e } } = ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } ) .
$$

The algorithm then generates only the remaining sufix at random:

$$
a _ { t , h _ { t } + 1 : H } ^ { \mathrm { p r e } } \sim \pi _ { \mathrm { p r e } } ( \cdot \mid s _ { t } , c _ { t , k } ) .
$$

Thus, although the reward function R is deterministic,

$$
R _ { t } = R { \Big ( } x _ { i _ { t } } , ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , a _ { t , h _ { t } + 1 : H } ^ { \mathrm { p r e } } ) { \Big ) }
$$

is random because the reference-generated sufix is random.

For $k \in \{ 0 , 1 \}$ , define the expected exponential reward

$$
M _ { t , k } : = \mathbb { E } _ { b _ { h _ { t } + 1 : H } \sim \pi _ { \mathrm { p r e } } \left( \cdot | s _ { t } , c _ { t , k } \right) } \left[ \exp \left( \frac { R \left( x _ { i _ { t } } , \left( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } \right) \right) } { \lambda } \right) \right] .
$$

The expectation is only over the reference-generated sufix; the prompt, prefix, and candidate token are held fixed. Equivalently,

$$
M _ { t , k } = \mathbb { E } \left[ e ^ { R _ { t } / \lambda } \Big | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k \right] .
$$

To make this expectation explicit, let $b _ { h _ { t } + 1 : H } = ( b _ { h _ { t } + 1 } , \ldots , b _ { H } )$ denote a possible sufix, whose reference probability is

$$
\pi _ { \mathrm { p r e } } ( b _ { h _ { t } + 1 : H } \mid s _ { t } , c _ { t , k } ) = \prod _ { h = h _ { t } + 1 } ^ { H } \pi _ { \mathrm { p r e } } ( b _ { h } \mid x _ { i t } , ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : h - 1 } ) ) .
$$

By the definition of expectation for a finite distribution, we can compute $M _ { t , k }$ as

$$
M _ { t , k } = \sum _ { \substack { ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { t _ { t + 1 : H } } ) \in A ( x _ { i _ { t } } ) } } \pi _ { \mathrm { p r e } } ( b _ { h _ { t } + 1 : H } \mid s _ { t } , c _ { t , k } ) \times \exp \left( \frac { R \left( x _ { i _ { t } } , \left( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } \right) \right) } { \lambda } \right) .
$$

The feasibility condition ensures that the concatenated sequence is a legal full answer. If $h _ { t } = H$ there is one empty sufix, with probability one.

Since R takes values in [0, 1], and the sufix probabilities sum to one, we have that

$$
\begin{array} { r } { 1 \leq M _ { t , k } \leq e ^ { 1 / \lambda } , \ k \in \{ 0 , 1 \} . } \end{array}\tag{C.3}
$$

We next calculate how acceptance changes the branch distribution. Recall that

$$
I _ { t } = \mathbf { 1 } \Big \{ U _ { t } \leq e ^ { ( R _ { t } - 1 ) / \lambda } \Big \} , U _ { t } \sim \mathrm { U n i f } \big [ 0 , 1 \big ] , 
$$

where $U _ { t }$ is independent uniform distribution random variables. Conditioned on $\mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } .$ $Y _ { t } , R _ { t }$ , the acceptance threshold is fixed. Moreover, notice that $0 \leq R _ { t } \leq 1$ implies $e ^ { ( R _ { t } - 1 ) / \lambda } \in$ $[ e ^ { - 1 / \lambda } , 1 ]$

By the independence and the uniform density of $U _ { t } .$ , we obtain

$$
\begin{array} { r l } & { \operatorname* { P r } \bigl ( I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } , R _ { t } \bigr ) = \operatorname* { P r } \Bigl ( U _ { t } \leq e ^ { ( R _ { t } - 1 ) / \lambda } \Big | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } , R _ { t } \Bigr ) } \\ & { \qquad = \displaystyle \int _ { 0 } ^ { 1 } \mathbf { 1 } \Big \{ u \leq e ^ { ( R _ { t } - 1 ) / \lambda } \Big \} d u = e ^ { ( R _ { t } - 1 ) / \lambda } . } \end{array}
$$

This calculation averages over $U _ { t }$ only; it does not yet average over the reference sufix.

Now conditioned only on $\mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k$ , the selected branch is fixed, but $R _ { t }$ still depends on the random reference sufix. Recall that $I _ { t }$ is an indicator and we apply the law of iterated expectation to get

$$
\begin{array} { r l } & { \mathrm { P r } ( I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k ) = \mathbb { E } [ I _ { t } | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k ] } \\ & { \phantom { m m m m m m m m } = \mathbb { E } [ \mathbb { E } [ I _ { t } | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } , R _ { t } ] | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k ] } \\ & { \phantom { m m m m m m m } = \mathbb { E } \Big [ e ^ { ( R _ { t } - 1 ) / \lambda } \Big | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k \Big ] . } \end{array}
$$

The third uses the acceptance probability computed above.

Finally, $e ^ { ( R _ { t } - 1 ) / \lambda } = e ^ { - 1 / \lambda } e ^ { R _ { t } / \lambda }$ , and the factor $e ^ { - 1 / \lambda }$ is deterministic. Hence, we have

$$
\mathbb { E } \left[ e ^ { ( R _ { t } - 1 ) / \lambda } \Big | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k \right] = e ^ { - 1 / \lambda } \mathbb { E } \Big [ e ^ { R _ { t } / \lambda } \Big | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k \Big ] = e ^ { - 1 / \lambda } M _ { t , k } .
$$

The last equality is the definition of $M _ { t , k }$ . Thus the acceptance probability for branch $k$ is the average of its sufix-specific acceptance probabilities, weighted by the reference sufix law.

The branch-selection rule in the algorithm is

$$
\operatorname* { P r } ( Y _ { t } = k \mid { \mathcal { F } } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) = { \frac { \pi _ { \operatorname { p r e } } ( c _ { t , k } \mid s _ { t } ) } { \pi _ { \operatorname { p r e } } ( c _ { t , 1 } \mid s _ { t } ) + \pi _ { \operatorname { p r e } } ( c _ { t , 0 } \mid s _ { t } ) } } .
$$

By the property of conditional probability, we have

$$
\begin{array} { r l r } {  { \operatorname* { P r } ( Y _ { t } = k , I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) = \operatorname* { P r } ( I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } , Y _ { t } = k ) \cdot \operatorname* { P r } ( Y _ { t } = k | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) } } \\ & { } & { = e ^ { - 1 / \lambda } \frac { \pi _ { \operatorname* { p r e } } ( c _ { t , k } \mid s _ { t } ) M _ { t , k } } { \pi _ { \operatorname* { p r e } } ( c _ { t , 1 } \mid s _ { t } ) + \pi _ { \operatorname* { p r e } } ( c _ { t , 0 } \mid s _ { t } ) } . ~ ( \mathrm { C } . 4 ) } \end{array}
$$

All denominators are positive because the reference policy is positive.

Summing (C.4) over the two branch indices $Y _ { t } = 0 , 1$ and using $M _ { t , k } \ge 1$ , we obtain

$$
\begin{array} { r } { \mathrm { P r } \big ( I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \big ) = e ^ { - 1 / \lambda } \frac { \pi _ { \mathrm { p r e } } \big ( c _ { t , 1 } \mid s _ { t } \big ) M _ { t , 1 } + \pi _ { \mathrm { p r e } } \big ( c _ { t , 0 } \mid s _ { t } \big ) M _ { t , 0 } } { \pi _ { \mathrm { p r e } } \big ( c _ { t , 1 } \mid s _ { t } \big ) + \pi _ { \mathrm { p r e } } \big ( c _ { t , 0 } \mid s _ { t } \big ) } } \\ { \geq e ^ { - 1 / \lambda } \frac { \pi _ { \mathrm { p r e } } \big ( c _ { t , 1 } \mid s _ { t } \big ) + \pi _ { \mathrm { p r e } } \big ( c _ { t , 0 } \mid s _ { t } \big ) } { \pi _ { \mathrm { p r e } } \big ( c _ { t , 1 } \mid s _ { t } \big ) + \pi _ { \mathrm { p r e } } \big ( c _ { t , 0 } \mid s _ { t } \big ) } = e ^ { - 1 / \lambda } . } \end{array}\tag{C.5}
$$

This proves (B.3). By Bayes’ rule, we have that

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } ( Y _ { t } = 1 | I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) = \frac { \operatorname* { P r } ( Y _ { t } = 1 , I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) } { \operatorname* { P r } ( I _ { t } = 1 | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) } } \\ { = \frac { \pi _ { \mathrm { p r e } } ( c _ { t , 1 } | s _ { t } ) M _ { t , 1 } } { \pi _ { \mathrm { p r e } } ( c _ { t , 1 } | s _ { t } ) M _ { t , 1 } + \pi _ { \mathrm { p r e } } ( c _ { t , 0 } | s _ { t } ) M _ { t , 0 } } . } \end{array}\tag{C.6}
$$

It remains to present this distribution in terms of the regularized truth. By Lemma B.1, the optimal policy is

$$
\pi _ { \lambda } ^ { \star } ( a _ { 1 : H } \mid x ) = \frac { \pi _ { \mathrm { p r e } } ( a _ { 1 : H } \mid x ) e ^ { R ( x , a _ { 1 : H } ) / \lambda } } { Z _ { \lambda } ( x ) } .
$$

Its probability of the prefix $\left( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } \right)$ is obtained by summing over all feasible sufixes. The corresponding full-answer events are disjoint, and their union is precisely the prefix event. Hence, by marginalization and Lemma B.1, we have

$$
\begin{array} { r l } & { \pi _ { \lambda } ^ { \star } \big ( a _ { t , 1 ; h _ { t } - 1 } , c _ { t , k } \big \vert \ x _ { i _ { t } } \big ) } \\ { = } & { \qquad \displaystyle \sum _ { b _ { h _ { t } + 1 ; H ^ { \star } } } \pi _ { \lambda } ^ { \star } \big ( ( a _ { t , 1 ; h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 ; H } ) \bigm \vert x _ { i _ { t } } \big ) } \\ & { \quad \displaystyle ( a _ { t , 1 ; h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 ; H } ) \in A ( x _ { i _ { t } } ) } \\ { = } & { \frac { \pi _ { \mathrm { p r e } } \big ( \big ( a _ { t , 1 ; h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 ; H } \bigm ) \bigm \vert x _ { i _ { t } } \big ) } { b _ { h _ { t } + 1 ; H ^ { \star } } } \times \exp \biggl ( \frac { R \big ( x _ { i _ { t } } , \big ( a _ { t , 1 ; h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 ; H } \bigm ) \bigm ) } { \lambda } \biggr ) . } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{C.7}
$$

We now expand this sum. For every feasible sufix $b _ { h _ { t } + 1 : H }$ , the autoregressive chain rule separates the full-answer probability into the prefix, the selected token, and the sufix:

$$
\begin{array} { r l } & { \pi _ { \mathrm { p r e } } \big ( ( a _ { t , \lfloor \cdot \lambda _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } ) \mid x _ { i } \big ) } \\ & { = \left[ \displaystyle \prod _ { h = 1 } ^ { h _ { t } - 1 } \pi _ { \mathrm { p r e } } ( a _ { t , h } \mid x _ { i } , a _ { t , 1 : h - 1 } ) \right] \pi _ { \mathrm { p r e } } ( c _ { t , k } \mid x _ { i } , a _ { t , 1 : h _ { t } - 1 } ) \times \left[ \displaystyle \prod _ { h = h _ { t } + 1 } ^ { H } \pi _ { \mathrm { p r e } } ( b _ { h } \mid x _ { i } , ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : h - 1 } ) ) \right] . } \end{array}
$$

The first product is the probability of generating the fixed prefix. The middle factor is the conditional probability of the selected token. The last product is the conditional probability of generating the sufix after that token. More explicitly, using $s _ { t } = ( x _ { i _ { t } } , a _ { t , 1 : h _ { t } - 1 } )$ , we have

$$
\prod _ { h = 1 } ^ { h _ { t } - 1 } \pi _ { \mathrm { p r e } } ( a _ { t , h } \mid x _ { i _ { t } } , a _ { t , 1 : h - 1 } ) = \pi _ { \mathrm { p r e } } ( a _ { t , 1 : h _ { t } - 1 } \mid x _ { i _ { t } } ) , \pi _ { \mathrm { p r e } } ( c _ { t , k } \mid x _ { i _ { t } } , a _ { t , 1 : h _ { t } - 1 } ) = \pi _ { \mathrm { p r e } } ( c _ { t , k } \mid s _ { t } ) ,
$$

$$
\prod _ { h = h _ { t } + 1 } ^ { H } \pi _ { \mathrm { p r e } } \bigl ( b _ { h } \mid x _ { i t } , ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : h - 1 } ) \bigr ) = \pi _ { \mathrm { p r e } } \bigl ( b _ { h _ { t } + 1 : H } \mid s _ { t } , c _ { t , k } \bigr ) .
$$

Thus, we have the factorization

$$
\begin{array} { r } { \pi _ { \mathrm { p r e } } \big ( \big ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } \big ) \mid x _ { i _ { t } } \big ) = \pi _ { \mathrm { p r e } } \big ( a _ { t , 1 : h _ { t } - 1 } \mid x _ { i _ { t } } \big ) \pi _ { \mathrm { p r e } } \big ( c _ { t , k } \mid s _ { t } \big ) \pi _ { \mathrm { p r e } } \big ( b _ { h _ { t } + 1 : H } \mid s _ { t } , c _ { t , k } \big ) . } \end{array}
$$

This is a factorization into conditional probabilities, not an independence assumption on the three parts of the answer. At $h _ { t } = 1 \ \mathrm { o r } \ h _ { t } = H$ , the corresponding empty product is one.

Substituting this factorization into (C.7) gives

$$
\begin{array} { r l } & { \pi _ { \lambda } ^ { \star } ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } \mid x _ { i _ { t } } ) } \\ { } & { = \underbrace { \sum _ { b _ { h _ { t } + 1 : H } : \quad } \frac { \pi _ { \mathrm { p r e } } \left( a _ { t , 1 : h _ { t } - 1 } \mid x _ { i _ { t } } \right) \pi _ { \mathrm { p r e } } \left( c _ { t , k } \mid s _ { t } \right) } { Z _ { \lambda } \left( x _ { i _ { t } } \right) } \times \pi _ { \mathrm { p r e } } \left( b _ { h _ { t } + 1 : H } \mid s _ { t } , c _ { t , k } \right) } _ { \left( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } \right) \in A \left( x _ { i _ { t } } \right) } } \\ { } & { \times \exp \left( \frac { R \left( x _ { i _ { t } } , \left( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } \right) \right) } { \lambda } \right) . } \end{array}
$$

Here the summation variable is only $b _ { h _ { t } + 1 : H }$ . The quantities $\pi _ { \mathrm { p r e } } ( a _ { t , 1 : h _ { t } - 1 } ~ \vert ~ x _ { i _ { t } } ) , \pi _ { \mathrm { p r e } } ( c _ { t , k } ~ \vert ~$ $s _ { t } ) , Z _ { \lambda } ( x _ { i _ { t } } )$ do not depend on this sufix. Thus, we factor them out of the sum, and retain exactly the same feasible sufixes to obtain

$$
\begin{array} { r l } & { \pi _ { \lambda } ^ { * } \big ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } \big \vert \ x _ { i _ { t } } \big ) } \\ & { = \frac { \pi _ { \mathrm { p r e } } \big ( a _ { t , 1 : h _ { t } - 1 } \big \vert \ x _ { i _ { t } } \big ) \pi _ { \mathrm { p r e } } \big ( c _ { t , k } \big \vert \ s _ { t } \big ) } { Z _ { \lambda } \big ( x _ { i _ { t } } \big ) } } \\ & { \quad \times \left[ \begin{array} { c } { \sum _ { \qquad \mathrm { b } _ { h _ { t } + 1 : H } : } \qquad \pi _ { \mathrm { p r e } } \big ( b _ { h _ { t } + 1 : H } \big \vert \ s _ { t } , c _ { t , k } \big ) \times \exp \left( \frac { R \big ( x _ { i _ { t } } , \big ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } \big ) \big ) } { \lambda } \right) } \\ { \big ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } , b _ { h _ { t } + 1 : H } \big ) \in A ( x _ { i _ { t } } ) } \end{array} \right] . } \end{array}
$$

The expression in square brackets is precisely the defining sufix sum for $M _ { t , k }$ . Therefore, we get

$$
\pi _ { \lambda } ^ { \star } ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } \mid x _ { i _ { t } } ) = \frac { \pi _ { \mathrm { p r e } } ( a _ { t , 1 : h _ { t } - 1 } \mid x _ { i _ { t } } ) } { Z _ { \lambda } ( x _ { i _ { t } } ) } \pi _ { \mathrm { p r e } } ( c _ { t , k } \mid s _ { t } ) M _ { t , k } .
$$

Recall the conditional token probability $\begin{array} { r } { \pi _ { \lambda } ^ { \star } ( c _ { t , k } \ | \ s _ { t } ) = \frac { \pi _ { \lambda } ^ { \star } ( a _ { t , 1 : h _ { t } - 1 } , c _ { t , k } | \boldsymbol { x } _ { i _ { t } } ) } { \pi _ { \lambda } ^ { \star } ( a _ { t , 1 : h _ { t } - 1 } | \boldsymbol { x } _ { i _ { t } } ) } } \end{array}$ . The denominator and the preceding prefix factor are common to both branch indices and are positive. They therefore cancel and we get

$$
\frac { \pi _ { \lambda } ^ { \star } ( c _ { t , \mathrm { ~ 1 ~ } } | \ s _ { t } ) } { \pi _ { \lambda } ^ { \star } ( c _ { t , \mathrm { ~ 1 ~ } } | \ s _ { t } ) + \pi _ { \lambda } ^ { \star } ( c _ { t , 0 } \mid \ s _ { t } ) } = \frac { \pi _ { \mathrm { p r e } } ( c _ { t , \mathrm { ~ 1 ~ } } \vert \ s _ { t } ) M _ { t , 1 } } { \pi _ { \mathrm { p r e } } ( c _ { t , \mathrm { ~ 1 ~ } } \vert \ s _ { t } ) M _ { t , 1 } + \pi _ { \mathrm { p r e } } ( c _ { t , 0 } \mid \ s _ { t } ) M _ { t , 0 } } .
$$

Together with (C.6), this proves the first equality in (B.4).

Finally, Assumption 3.2 gives $\pi _ { \lambda } ^ { \star } = \pi _ { w _ { \lambda } ^ { \star } }$ . The common softmax normalizer cancels, so we have

$$
\frac { \pi _ { \lambda } ^ { \star } ( c _ { t , 1 } \mid s _ { t } ) } { \pi _ { \lambda } ^ { \star } ( c _ { t , 1 } \mid s _ { t } ) + \pi _ { \lambda } ^ { \star } ( c _ { t , 0 } \mid s _ { t } ) } = \frac { e ^ { ( w _ { \lambda } ^ { \star } ) ^ { \top } \phi ( s _ { t } , c _ { t , 1 } ) } } { e ^ { ( w _ { \lambda } ^ { \star } ) ^ { \top } \phi ( s _ { t } , c _ { t , 1 } ) } + e ^ { ( w _ { \lambda } ^ { \star } ) ^ { \top } \phi ( s _ { t } , c _ { t , 0 } ) } } = \frac { 1 } { 1 + e ^ { ( - ( w _ { \lambda } ^ { \star } ) ^ { \top } [ \phi ( s _ { t } , c _ { t , 1 } ) - \phi ( s _ { t } , c _ { t , 0 } ) ] ) } } = \sigma ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } ) .
$$

This establishes teh second equality in (B.4).

If $c _ { t , 1 } = c _ { t , 0 } ,$ the two branch indices have identical reference-completion laws, hence $M _ { t , 1 } = M _ { t , 0 }$ Their prior and accepted probabilities are both $1 / 2 ,$ consistent with $z _ { t } = 0$ and $\sigma ( 0 ) = 1 / 2$ . After EOS, both candidates are null, so the same argument applies. □

Proof of Lemma B.3. We first recall the random definition vector $z _ { t }$ from Algorithm 1. The history $\mathcal { F } _ { t }$ contains all randomness before round t, so $\theta _ { t }$ is fixed conditional on $\mathcal { F } _ { t }$ . After choosing the source index $i _ { t }$ and branching time $h _ { t }$ , the algorithm draws $a _ { t , 1 : H } \sim \pi _ { \mathrm { t e a } } ( \cdot \mid x _ { i _ { t } } )$ and sets $s _ { t } =$ $( x _ { i _ { t } } , a _ { t , 1 : h _ { t } - 1 } ) , c _ { t , 1 } = a _ { t , h _ { t } }$ is the teacher token, and $ { c _ { t , 0 } } \sim \pi _ { \mathrm { s t u } , \theta _ { t } } ( \cdot \ | \ s _ { t } )$ is a fresh student token at the same state. The vector in the lemma is

$$
z _ { t } : = \phi ( s _ { t } , c _ { t , 1 } ) - \phi ( s _ { t } , c _ { t , 0 } ) \in \mathbb { R } ^ { D } .
$$

First, we fix the history $\mathcal { F } _ { t }$ and a legal nonterminal state $s ,$ so that $B ( s )$ is nonempty. Recall that $B > 0$ is the scalar bound on the parameter norm, whereas $B ( s )$ is the set of legal next tokens at s. Both b and c below denote individual tokens, not trajectories.

The current student policy is then explicitly

$$
\pi _ { \operatorname { s t u } , \theta _ { t } } ( b \mid s ) = \left\{ \begin{array} { l l } { \displaystyle \frac { \exp \bigl ( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) \bigr ) } { \sum _ { c \in \mathcal { B } ( s ) } \exp \bigl ( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } ( s , c ) \bigr ) } , } & { b \in \mathcal { B } ( s ) , } \\ { 0 , } & { b \in \mathcal { A } \backslash \mathcal { B } ( s ) . } \end{array} \right.
$$

Here $\theta _ { t } \in \mathbb { R } ^ { d }$ is the current student parameter, and $\phi _ { \mathrm { s t u } } ( s , c ) \in \mathbb { R } ^ { d }$ is the student feature vector of the state-token pair $( s , c )$ . Since $\theta _ { t }$ is $\mathcal { F } _ { t ^ { - } }$ -measurable, it is fixed after we condition on the history. The model bounds are

$$
\begin{array} { r } { \| \theta _ { t } \| _ { 2 } \leq B , \ \| \phi _ { \mathrm { s t u } } ( s , c ) \| _ { 2 } \leq 1 , \ \forall c \in \mathcal { B } ( s ) . } \end{array}
$$

Consequently, by the Cauchy-Schwarz inequality, for each legal token $^ { c , }$ we have

$$
\begin{array} { r } { \mathopen { } \mathclose \bgroup \left| \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } ( s , c ) \aftergroup \egroup \right| \leq \| \theta _ { t } \| _ { 2 } \| \phi _ { \mathrm { s t u } } ( s , c ) \| _ { 2 } \leq B . } \end{array}
$$

Equivalently, $- B \le \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } ( s , c ) \le B$ . Because $u \mapsto e ^ { u }$ is increasing, we have

$$
\begin{array} { r } { e ^ { - B } \le \exp \Bigl ( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } \bigl ( s , c \bigr ) \Bigr ) \le e ^ { B } , ~ c \in \mathcal { B } ( s ) . } \end{array}
$$

Now fix any $b \in B ( s )$ . Applying these bounds to the full softmax expression yields

$$
\begin{array} { r l } & { \pi _ { \mathrm { s t u } , \theta _ { t } } ( b \mid s ) = \frac { \displaystyle \exp \left( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) \right) } { \displaystyle \sum _ { c \in \mathcal { B } ( s ) } \exp \left( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } ( s , c ) \right) } \geq \frac { e ^ { - B } } { \displaystyle \sum _ { c \in \mathcal { B } ( s ) } \exp \left( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } ( s , c ) \right) } } \\ & { \qquad \quad \geq \frac { e ^ { - B } } { \displaystyle \sum _ { c \in \mathcal { B } ( s ) } e ^ { B } } = \frac { e ^ { - B } } { | \mathcal { B } ( s ) | e ^ { B } } = \frac { e ^ { - 2 B } } { | \mathcal { B } ( s ) | } . } \end{array}\tag{C.8}
$$

The first equality is the definition of the student policy. The first inequality replaces its numerator

by the lower bound $e ^ { - B }$ , leaving the positive denominator unchanged. The second inequality increases each denominator term to its upper bound $e ^ { B }$ ; increasing a positive denominator decreases the fraction.

The lower bound applies only to legal tokens; illegal tokens have probability zero. At an absorbing prefix before the terminal horizon, $B ( s ) = \{ { \tt n u l 1 } \}$ , and the softmax formula gives

$$
\pi _ { \mathrm { s t u } , \theta _ { t } } ( \mathtt { n u l l } \mid s ) = \frac { \exp \Bigl ( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } \bigl ( s , \mathtt { n u l l } \bigr ) \Bigr ) } { \exp \bigl ( \theta _ { t } ^ { \top } \phi _ { \mathrm { s t u } } \bigl ( s , \mathtt { n u l l } \bigr ) \bigr ) } = 1 \ge e ^ { - 2 B } .
$$

Thus (C.8) also holds at these prefixes.

To prove the matrix inequality, fix a deterministic vector $\boldsymbol { v } \in \mathbb { R } ^ { D }$ and notice $v ^ { \top } z _ { t } z _ { t } ^ { \top } v = ( v ^ { \top } z _ { t } ) ^ { 2 }$ Conditional expectation is linear, and v is fixed, so we have

$$
v ^ { \top } \mathbb { E } \left[ z _ { t } z _ { t } ^ { \top } \mid \mathcal { F } _ { t } \right] v = \mathbb { E } \left[ v ^ { \top } z _ { t } z _ { t } ^ { \top } v \mid \mathcal { F } _ { t } \right] = \mathbb { E } \left[ ( v ^ { \top } z _ { t } ) ^ { 2 } \mid \mathcal { F } _ { t } \right] .
$$

We now expand this expectation in the order of sampling. The source index $i _ { t }$ and branching time $h _ { t }$ are fresh, independent, uniform draws. Hence, we have $\begin{array} { r } { \operatorname* { P r } ( i _ { t } = i , h _ { t } = h \mid \mathcal { F } _ { t } ) = \frac { 1 } { n H } } \end{array}$

The conditional law of total expectation gives us

$$
\begin{array} { r l } & { \mathbb { E } \left[ ( v ^ { \top } z _ { t } ) ^ { 2 } \mid \mathcal { F } _ { t } \right] = \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { h = 1 } ^ { H } \operatorname* { P r } ( i _ { t } = i , h _ { t } = h \mid \mathcal { F } _ { t } ) \mathbb { E } \left[ ( v ^ { \top } z _ { t } ) ^ { 2 } \mid \mathcal { F } _ { t } , i _ { t } = i , h _ { t } = h \right] } \\ & { \qquad = \displaystyle \frac { 1 } { n H } \sum _ { i = 1 } ^ { n } \sum _ { h = 1 } ^ { H } \mathbb { E } \left[ ( v ^ { \top } z _ { t } ) ^ { 2 } \mid \mathcal { F } _ { t } , i _ { t } = i , h _ { t } = h \right] . } \end{array}
$$

For fixed $i , h .$ , let $a _ { i , 1 : h }$ denote a possible realization of the first h teacher tokens, and define $s _ { i , h } : =$ $\left( x _ { i } , a _ { i , 1 : h - 1 } \right)$ . Conditioning on $i _ { t } = i , \ h _ { t } = h , \ a _ { t , 1 : h } = a _ { i , 1 : h }$ , the definitions of $s _ { t }$ and $c _ { t , 1 }$ yield $\boldsymbol { s } _ { t } = \boldsymbol { s } _ { i , h }$ and $c _ { t , 1 } = a _ { i , h }$ . The only remaining randomness in $z _ { t }$ is the fresh student token. For each $b \in B ( s _ { i , h } )$ , its conditional probability is

$$
\operatorname* { P r } ( c _ { t , 0 } = b \mid \mathcal { F } _ { t } , i _ { t } = i , h _ { t } = h , a _ { t , 1 : h } = a _ { i , 1 : h } ) = \pi _ { \mathrm { s t u } , \theta _ { t } } ( b \mid s _ { i , h } ) .
$$

If $c _ { t , 0 } = b$ , substitution into the recalled definition gives

$$
z _ { t } = \phi ( s _ { i , h } , a _ { i , h } ) - \phi ( s _ { i , h } , b ) .
$$

Thus, we have

$$
\begin{array} { r l } & { \mathbb { E } \Big [ ( v ^ { \top } z _ { t } ) ^ { 2 } \mid \mathcal { F } _ { t } , i _ { t } = i , h _ { t } = h , a _ { t , 1 : h } = a _ { i , 1 : h } \Big ] } \\ & { = \displaystyle \sum _ { b \in \mathcal { B } ( s _ { i , h } ) } \pi _ { \mathrm { s t u } , \theta _ { t } } \big ( b \mid s _ { i , h } \big ) \Big ( v ^ { \top } \big [ \phi \big ( s _ { i , h } , a _ { i , h } \big ) - \phi \big ( s _ { i , h } , b \big ) \big ] \Big ) ^ { 2 } . } \end{array}
$$

The frozen teacher’s prefix law is $\textstyle \pi _ { \mathrm { t e a } } ( a _ { i , 1 : h } \mid x _ { i } ) = \prod _ { k = 1 } ^ { h } \pi _ { \mathrm { t e a } } ( a _ { i , k } \mid x _ { i } , a _ { i , 1 : k - 1 } )$ . The remaining teacher sufix $a _ { t , h + 1 : H }$ is absent from $z _ { t } ;$ summing its conditional probabilities gives one.

For fixed $i _ { t } = i$ and $h _ { t } = h$ , the preceding calculation conditions on the teacher tokens $a _ { t , 1 : h } = a _ { i , 1 : h }$ and averages over the student token $c _ { t , 0 }$ . We now average over all possible teacher token sequences

$a _ { i , 1 : h }$ , each weighted by its probability $\pi _ { \mathrm { t e a } } ( a _ { i , 1 : h } \mid x _ { i } )$ . By the law of total expectation, this removes the conditioning on $a _ { t , 1 : h }$ and gives $\mathbb { E } [ ( v ^ { \top } z _ { t } ) ^ { 2 } \mid \mathcal { F } _ { t } , i _ { t } = i , h _ { t } = h ]$

$$
\begin{array} { r l } & { v ^ { \top } \mathbb { E } \left[ z _ { t } z _ { t } ^ { \top } \mid \mathcal { F } _ { t } \right] v } \\ & { = \displaystyle \frac { 1 } { n H } \sum _ { i = 1 } ^ { n } \sum _ { h = 1 } ^ { H } \mathbb { E } _ { a _ { i , \mathtt { l } ; h } \sim \pi _ { \mathtt { t e a } } ( \cdot \vert x _ { i } ) } \Biggl [ \sum _ { b \in \mathcal { B } ( s _ { i , h } ) } \pi _ { \mathtt { s t u } , \theta _ { t } } ( b \mid s _ { i , h } ) \left( v ^ { \top } \left[ \phi ( s _ { i , h } , a _ { i , h } ) - \phi ( s _ { i , h } , b ) \right] \right) ^ { 2 } \bigg | \mathcal { F } _ { t } \Biggr ] } \\ & { \ge \displaystyle \frac { e ^ { - 2 B } } { n H } \sum _ { i = 1 } ^ { n } \sum _ { h = 1 } ^ { H } \mathbb { E } _ { a _ { i , \mathtt { l } ; h } \sim \pi _ { \mathtt { t e a } } ( \cdot \vert x _ { i } ) } \Biggl [ \frac { 1 } { \left. \mathcal { B } ( s _ { i , h } ) \right. } \sum _ { b \in \mathcal { B } ( s _ { i , h } ) } \left( v ^ { \top } \left[ \phi ( s _ { i , h } , a _ { i , h } ) - \phi ( s _ { i , h } , b ) \right] \right) ^ { 2 } \bigg | \mathcal { F } _ { t } \Biggr ] } \\ & { = e ^ { - 2 B } v ^ { \top } \left( \frac { 1 } { H } \sum _ { h = 1 } ^ { H } G _ { h } \right) v = e ^ { - 2 B } v ^ { \top } G _ { \mathtt { j o i n t } } v . } \end{array}
$$

The inequality holds by $\left( \mathrm { C } . 8 \right)$ , and the second equality is by definition (3.7). The final equality uses $\begin{array} { r } { G _ { \mathrm { j o i n t } } = H ^ { - 1 } \sum _ { h = 1 } ^ { H } G _ { h } } \end{array}$

Here $a _ { i , 1 : h }$ is an integration variable, not an additional rollout: each round still draws only one source index and one teacher answer. Since the calculation holds for every $v ,$ it proves the matrix inequality in (B.5).

The feature norm bound also gives

$$
\| z _ { t } \| _ { 2 } = \| \phi ( s _ { t } , c _ { t , 1 } ) - \phi ( s _ { t } , c _ { t , 0 } ) \| _ { 2 } \leq \| \phi ( s _ { t } , c _ { t , 1 } ) \| _ { 2 } + \| \phi ( s _ { t } , c _ { t , 0 } ) \| _ { 2 } \leq 2 .
$$

Thus, we have proved (B.5), and we will prove the drift bound (B.6) next.

Starting from the definition of the actual stochastic update in Algorithm 1, we have

$$
g _ { t } : = I _ { t } z _ { t } \Big [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } \Big ] .
$$

Recall that $I _ { t } \in \{ 0 , 1 \}$ is the acceptance indicator, $Y _ { t } \in \{ 0 , 1 \}$ is the selected branch index, and $\sigma ( u ) = ( 1 + e ^ { - u } ) ^ { - 1 }$

Conditioned on $\mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 }$ , the parameter $w _ { t }$ is $\mathcal { F } _ { t }$ -measurable, $w _ { \lambda } ^ { \star }$ is fixed, and $z _ { t } = \phi ( s _ { t } , c _ { t , 1 } ) -$ $\phi ( s _ { t } , c _ { t , 0 } )$ is also determined. Thus, $w _ { t } , \ z _ { t }$ , and $\sigma ( z _ { t } ^ { \top } w _ { t } )$ are measurable under this conditioning and $I _ { t }$ and $Y _ { t }$ are still randomized.

First, by definition of $g _ { t }$ , we have

$$
\begin{array} { r l } & { ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } g _ { t } = ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } \left\{ I _ { t } z _ { t } [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } ] \right\} } \\ & { \phantom { = \ } = I _ { t } \Big [ ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } z _ { t } \Big ] [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } ] = z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } ) I _ { t } \big [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } \big ] . } \end{array}
$$

Taking conditional expectations and pulling out the measurable scalars, we have

$$
\begin{array} { r } { \mathbb { E } \left[ ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } g _ { t } \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \right] = z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } ) \mathbb { E } \left[ I _ { t } \big [ \sigma \big ( z _ { t } ^ { \top } w _ { t } \big ) - Y _ { t } \big ] \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \right] . } \end{array}
$$

We evaluate the remaining expectation by splitting the two possible values of $I _ { t }$ . When $I _ { t } = 0$ the

product is zero; when $I _ { t } = 1$ it is $\sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t }$ . By the conditional law of total expectation,

$$
\begin{array} { r l } & { \mathbb { E } \left[ I _ { t } [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } ] \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \right] } \\ { } & { = \operatorname* { P r } ( I _ { t } = 0 \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) \cdot 0 + \operatorname* { P r } ( I _ { t } = 1 \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) \times \mathbb { E } \left[ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } \mid I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \right] } \\ { } & { = \operatorname* { P r } ( I _ { t } = 1 \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) \times \left[ \sigma ( z _ { t } ^ { \top } w _ { t } ) - \mathbb { E } [ Y _ { t } \mid I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ] \right] . } \end{array}
$$

Because $Y _ { t }$ is binary, its conditional expectation is the conditional probability that it equals one. By Lemma B.2, we have that

$$
\begin{array} { r l } & { \quad \mathbb { E } [ Y _ { t } \mid I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ] } \\ & { = 0 \cdot \operatorname* { P r } ( Y _ { t } = 0 \mid I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) + 1 \cdot \operatorname* { P r } ( Y _ { t } = 1 \mid I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) } \\ & { = \operatorname* { P r } ( Y _ { t } = 1 \mid I _ { t } = 1 , \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) = \sigma ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } ) . } \end{array}
$$

Plugging this back, we obtain

$$
\begin{array} { r } { \mathbb { E } \left[ \left( w _ { t } - w _ { \lambda } ^ { \star } \right) ^ { \top } g _ { t } \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \right] = z _ { t } ^ { \top } \left( w _ { t } - w _ { \lambda } ^ { \star } \right) \operatorname* { P r } ( I _ { t } = 1 \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) \times \left[ \sigma ( z _ { t } ^ { \top } w _ { t } ) - \sigma ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } ) \right] . } \end{array}\tag{C.9}
$$

Finally, we pull out the fixed vector $z _ { t }$ instead of the scalar inner product, which gives the corresponding vector conditional mean:

$$
\begin{array} { r l } & { \mathbb { E } \big [ g _ { t } \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \big ] = z _ { t } \mathbb { E } \Big [ I _ { t } \big [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } \big ] \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \Big ] } \\ & { \qquad = \operatorname* { P r } ( I _ { t } = 1 \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) z _ { t } \times \Big [ \sigma ( z _ { t } ^ { \top } w _ { t } ) - \sigma ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } ) \Big ] . } \end{array}\tag{C.10}
$$

By direct diferentiation, we have $\begin{array} { r } { \sigma ^ { \prime } ( u ) = \frac { e ^ { - u } } { ( 1 + e ^ { - u } ) ^ { 2 } } = \frac { 1 } { 2 + e ^ { u } + e ^ { - u } } > 0 } \end{array}$ . The denominator is even and nondecreasing in |u|, because its derivative on $[ 0 , \infty )$ is $e ^ { u } - e ^ { - u } \geq 0$ . Hence $\sigma ^ { \prime } ( u ) \geq \sigma ^ { \prime } ( 2 B )$ whenever $| u | \leq 2 B$ . For $r \in [ 0 , 1 ]$ , convexity of W places $\boldsymbol { w } _ { \lambda } ^ { \star } + \boldsymbol { r } \big ( \boldsymbol { w } _ { t } - \boldsymbol { w } _ { \lambda } ^ { \star } \big )$ in W, and therefore we have

$$
\left| z _ { t } ^ { \top } [ w _ { \lambda } ^ { \star } + r ( w _ { t } - w _ { \lambda } ^ { \star } ) ] \right| \leq \| z _ { t } \| _ { 2 } \| w _ { \lambda } ^ { \star } + r ( w _ { t } - w _ { \lambda } ^ { \star } ) \| _ { 2 } \leq 2 B .
$$

Now, we apply the fundamental theorem of calculus along this segment to get

$$
\begin{array} { r l } & { \sigma \big ( z _ { t } ^ { \top } w _ { t } \big ) - \sigma \big ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } \big ) = \displaystyle \int _ { 0 } ^ { 1 } \frac { d } { d r } \sigma \Big ( z _ { t } ^ { \top } \big [ w _ { \lambda } ^ { \star } + r \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \big ] \Big ) \ d r } \\ & { \quad \quad \quad = \displaystyle \int _ { 0 } ^ { 1 } \sigma ^ { \prime } \Big ( z _ { t } ^ { \top } \big [ w _ { \lambda } ^ { \star } + r \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \big ] \Big ) \ z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \ d r } \\ & { \quad \quad = z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \displaystyle \int _ { 0 } ^ { 1 } \sigma ^ { \prime } \Big ( z _ { t } ^ { \top } \big [ w _ { \lambda } ^ { \star } + r \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \big ] \Big ) \ d r . } \end{array}
$$

Multiplying the equality by $z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } )$ gives

$$
\begin{array} { r l } & { z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \left[ \sigma \big ( z _ { t } ^ { \top } w _ { t } \big ) - \sigma \big ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } \big ) \right] = \left[ z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \right] ^ { 2 } \displaystyle \int _ { 0 } ^ { 1 } \sigma ^ { \prime } \Big ( z _ { t } ^ { \top } \big [ w _ { \lambda } ^ { \star } + r \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \big ] \Big ) \ d r } \\ & { \qquad \geq \left[ z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \right] ^ { 2 } \displaystyle \int _ { 0 } ^ { 1 } \sigma ^ { \prime } \big ( 2 B \big ) d r } \\ & { \qquad = \sigma ^ { \prime } \big ( 2 B \big ) \left[ z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \big ) \right] ^ { 2 } \geq 0 . } \end{array}\tag{C.11}
$$

Recall that $I _ { t } = \mathbf { 1 } \{ U _ { t } \leq e ^ { ( R _ { t } - 1 ) / \lambda } \}$ and that Lemma B.2, (B.3) already establishes

$$
\operatorname* { P r } ( I _ { t } = 1 \mid { \mathcal { F } } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) \geq e ^ { - 1 / \lambda } .
$$

Combining all these together, we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left( w _ { t } - w _ { \lambda } ^ { \star } \right) ^ { \top } g _ { t } \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \right] = \operatorname* { P r } ( I _ { t } = 1 \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) z _ { t } ^ { \top } \left( w _ { t } - w _ { \lambda } ^ { \star } \right) \left[ \sigma ( z _ { t } ^ { \top } w _ { t } ) - \sigma ( z _ { t } ^ { \top } w _ { \lambda } ^ { \star } ) \right] } \\ & { \qquad \geq \operatorname* { P r } ( I _ { t } = 1 \mid \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } ) \sigma ^ { \prime } ( 2 B ) \left[ z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } ) \right] ^ { 2 } } \\ & { \qquad \geq e ^ { - 1 / \lambda } \sigma ^ { \prime } ( 2 B ) \left[ z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } ) \right] ^ { 2 } . } \end{array}
$$

The equality was derived by (C.9). The first inequality uses (C.11) and The second one is by (C.5). Now, we apply the tower property once more to obtain

$$
\begin{array} { r l } & { \mathbb { E } \Big [ ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } g _ { t } \bigm | \mathcal { F } _ { t } \Big ] = \mathbb { E } \Big [ \mathbb { E } \Big [ \big ( w _ { t } - w _ { \lambda } ^ { \star } \bigm ) ^ { \top } g _ { t } \bigm | \mathcal { F } _ { t } , s _ { t } , c _ { t , 1 } , c _ { t , 0 } \Big ] \Big | \mathcal { F } _ { t } \Big ] } \\ & { \qquad \ge \mathbb { E } \Big [ e ^ { - 1 / \lambda } \sigma ^ { \prime } \big ( 2 B \bigm ) \Big [ z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \bigm ) \Big ] ^ { 2 } \Big | \mathcal { F } _ { t } \Big ] } \\ & { \qquad = e ^ { - 1 / \lambda } \sigma ^ { \prime } ( 2 B ) \mathbb { E } \Big [ \Big [ z _ { t } ^ { \top } \big ( w _ { t } - w _ { \lambda } ^ { \star } \bigm ) \Big ] ^ { 2 } \Big | \mathcal { F } _ { t } \Big ] . } \end{array}
$$

Notice that $\left[ z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } ) \right] ^ { 2 } = ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } z _ { t } z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } )$ . Because $w _ { t } - w _ { \lambda } ^ { \star }$ is $\mathcal { F } _ { t }$ -measurable, by the linearity of conditional expectation, we have

$$
\mathbb { E } \left[ \left[ z _ { t } ^ { \top } ( w _ { t } - w _ { \lambda } ^ { \star } ) \right] ^ { 2 } \Big | \mathcal { F } _ { t } \right] = ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } \mathbb { E } \left[ z _ { t } z _ { t } ^ { \top } \mid \mathcal { F } _ { t } \right] ( w _ { t } - w _ { \lambda } ^ { \star } ) .
$$

Using the matrix lower bound (B.5), followed by $G _ { \mathrm { j o i n t } } \succeq \mu _ { \mathrm { j o i n t } } I _ { D }$ , we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } g _ { t } \Big | \mathcal { F } _ { t } \right] \geq e ^ { - 1 / \lambda } \sigma ^ { \prime } ( 2 B ) ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } \mathbb { E } \left[ z _ { t } z _ { t } ^ { \top } \mid \mathcal { F } _ { t } \right] ( w _ { t } - w _ { \lambda } ^ { \star } ) } \\ & { \qquad \geq e ^ { - 1 / \lambda } e ^ { - 2 B } \sigma ^ { \prime } ( 2 B ) ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } G _ { \mathrm { j o i n t } } ( w _ { t } - w _ { \lambda } ^ { \star } ) } \\ & { \qquad \geq e ^ { - 1 / \lambda } e ^ { - 2 B } \sigma ^ { \prime } ( 2 B ) \mu _ { \mathrm { j o i n t } } \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } } \\ & { \qquad = \gamma \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Here $I _ { D }$ is the D-dimensional identity matrix. The final equality uses the definition of $\gamma$ in (4.2). By the Cauchy-Schwarz inequality, we have $\| g _ { t } \| _ { 2 } = I _ { t } \| z _ { t } \| _ { 2 } | \sigma ( z _ { t } ^ { \top } w _ { t } ) - Y _ { t } | \le 2$ . We prove (B.6). For any $\boldsymbol { u } \in \mathbb { R } ^ { D }$ , let $p = { \mathrm { P r o j } } _ { W } ( u )$ . Since W is compact and convex, p exists and is unique. For any $r \in W$ , the segment $p + v ( r - p ) , 0 \leq v \leq 1$ , lies in W. The right derivative at zero of its

squared distance to u is nonnegative:

$$
0 \leq \frac { d } { d v } \| p + v ( r - p ) - u \| _ { 2 } ^ { 2 } \bigg | _ { v = 0 ^ { + } } = 2 \langle p - u , r - p \rangle .
$$

Thus $\langle u - p , r - p \rangle \leq 0$ . Taking $r = w _ { \lambda } ^ { \star }$ and expanding a square gives

$$
\| u - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } = \| u - p \| _ { 2 } ^ { 2 } + \| p - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } + 2 \langle u - p , p - w _ { \lambda } ^ { \star } \rangle \geq \| p - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } ,
$$

because the cross term and the first squared norm are nonnegative. We use this inequality with $u = w _ { t } - g _ { t } / [ \gamma ( t + 2 ) ]$ and $p = w _ { t + 1 }$ to get

$$
\| w _ { t + 1 } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } \leq \left\| w _ { t } - w _ { \lambda } ^ { \star } - \frac { g _ { t } } { \gamma ( t + 2 ) } \right\| _ { 2 } ^ { 2 } = \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } - \frac { 2 ( w _ { t } - w _ { \lambda } ^ { \star } ) ^ { \top } g _ { t } } { \gamma ( t + 2 ) } + \frac { \| g _ { t } \| _ { 2 } ^ { 2 } } { \gamma ^ { 2 } ( t + 2 ) ^ { 2 } } .
$$

Conditioning on $\mathcal { F } _ { t }$ , we apply (B.5) and (B.6) to get

$$
\mathbb { E } \Big [ \| w _ { t + 1 } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } \Big | \mathscr { F } _ { t } \Big ] \leq \left( 1 - \frac { 2 } { t + 2 } \right) \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } + \frac { 4 } { \gamma ^ { 2 } ( t + 2 ) ^ { 2 } } = \frac { t } { t + 2 } \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } + \frac { 4 } { \gamma ^ { 2 } ( t + 2 ) ^ { 2 } } .\tag{C.12}
$$

We now prove (B.7) by induction.

At $t = 0$ the first coeficient is zero. Taking expectations gives

$$
\mathbb { E } \Big [ \| w _ { 1 } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } \Big ] \leq \frac { 1 } { \gamma ^ { 2 } } \leq \frac { 4 } { 3 \gamma ^ { 2 } } .
$$

This is the base case of (B.7). Suppose its assertion holds at some $t \geq 1$ . By (C.12), we have

$$
\mathbb { E } \left[ \| w _ { t + 1 } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } \right] \le \frac { t } { t + 2 } \frac { 4 } { \gamma ^ { 2 } ( t + 2 ) } + \frac { 4 } { \gamma ^ { 2 } ( t + 2 ) ^ { 2 } } = \frac { 4 ( t + 1 ) } { \gamma ^ { 2 } ( t + 2 ) ^ { 2 } } \le \frac { 4 } { \gamma ^ { 2 } ( t + 3 ) } .
$$

The final inequality follows from $( t + 1 ) ( t + 3 ) = ( t + 2 ) ^ { 2 } - 1 \leq ( t + 2 ) ^ { 2 }$ . This proves (B.7).

For the increment bound, apply the same projection variational inequality with $r = w _ { t } \in W$ and the same $u , p$ as above. It implies

$$
\| w _ { t + 1 } - w _ { t } \| _ { 2 } ^ { 2 } \leq \langle u - w _ { t } , w _ { t + 1 } - w _ { t } \rangle \leq \| u - w _ { t } \| _ { 2 } \| w _ { t + 1 } - w _ { t } \| _ { 2 } = \frac { \| g _ { t } \| _ { 2 } } { \gamma ( t + 2 ) } \| w _ { t + 1 } - w _ { t } \| _ { 2 } .
$$

Thus, we obtain that $\begin{array} { r } { \| w _ { t + 1 } - w _ { t } \| _ { 2 } \leq \frac { 2 } { \gamma ( t + 2 ) } } \end{array}$ for every $t \geq 0$

Proof of Lemma $B . 4 .$ Recall the common feasible sets and the two softmax policies. The set $B ( s )$ contains the legal next tokens at a nonterminal prefix s, and $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } )$ contains the feasible complete answers of length H for prompt x. These sets are finite and do not depend on either parameter. Every nonterminal reachable state has at least one legal token. For $a \in B ( s )$

$$
\pi _ { w } ( a \mid s ) = \frac { \exp ( w ^ { \top } \phi ( s , a ) ) } { \sum _ { b \in B ( s ) } \exp ( w ^ { \top } \phi ( s , b ) ) } , \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) = \frac { \exp ( \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , a ) ) } { \sum _ { b \in B ( s ) } \exp ( \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) ) } .
$$

The feature maps are fixed and satisfy

$$
\| \phi ( s , a ) \| _ { 2 } \leq 1 , \ \| \phi _ { \mathrm { s t u } } ( s , a ) \| _ { 2 } \leq 1 , \ \| w \| _ { 2 } \leq B , \ \| \theta \| _ { 2 } \leq B .
$$

For a fixed feasible answer, $s _ { h } = \left( x , a _ { 1 : h - 1 } \right)$ , the autoregressive definition gives

$$
\pi _ { w } ( a _ { 1 : H } \mid x ) = \prod _ { h = 1 } ^ { H } \pi _ { w } ( a _ { h } \mid s _ { h } ) , \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) = \prod _ { h = 1 } ^ { H } \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { h } ) .
$$

Both probabilities are strictly positive on $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } )$

Starting from the definition of $Z _ { t }$ and the autoregressive product,

$$
\begin{array} { r l } & { Z _ { t } ( \vartheta ; x , a _ { 1 : H } ) = \lambda \log \frac { \pi _ { \mathrm { s t n } , \hat { \psi } } \left( a _ { 1 : H } \mid x \right) } { \pi _ { \mathrm { w } _ { t } } \left( a _ { 1 : H } \mid x \right) } = \lambda \log \frac { \prod _ { h = 1 } ^ { H } \pi _ { \mathrm { s t n } , \hat { \psi } } \left( a _ { h } \mid s _ { h } \right) } { \prod _ { h = 1 } ^ { H } \pi _ { \mathrm { w } _ { t } } \left( a _ { h } \mid s _ { h } \right) } } \\ & { = \lambda \displaystyle \sum _ { h = 1 } ^ { H } \left. \log \pi _ { \mathrm { s t n } , \hat { \vartheta } } \left( a _ { h } \mid s _ { h } \right) - \log \pi _ { \mathrm { w } _ { t } } \left( a _ { h } \mid s _ { h } \right) \right. } \\ & { = \lambda \displaystyle \sum _ { h = 1 } ^ { H } \left[ \vartheta ^ { \top } \phi _ { \mathrm { s t n } } ( s _ { h } , a _ { h } ) - w _ { t } ^ { \top } \phi ( s _ { h } , a _ { h } ) - \log \left( \displaystyle \sum _ { b \in B ( s _ { h } ) } c ^ { \hat { \vartheta } ^ { \top } \phi _ { \mathrm { s t n } } ( s _ { h } , b ) } \right) + \log \left( \displaystyle \sum _ { b \in B ( s _ { h } ) } c ^ { w _ { t } ^ { \top } \phi ( s _ { h } , b ) } \right) \right] . } \end{array}
$$

We now bound the token log ratios in this exact expression. The calculation is uniform in the parameters, so we carry it out for arbitrary $w \in W$ and $\theta \in \Theta$ before substituting $w = w _ { t }$ and $\theta = \vartheta$ . By the Cauchy-Schwarz inequality, for every legal token b, we have that

$$
\begin{array} { r } { | w ^ { \top } \phi ( s , b ) | \leq \| w \| _ { 2 } \| \phi ( s , b ) \| _ { 2 } \leq B , ~ | \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) | \leq \| \theta \| _ { 2 } \| \phi _ { \mathrm { s t u } } ( s , b ) \| _ { 2 } \leq B . } \end{array}
$$

Exponentiation preserves these inequalities, so we get

$$
e ^ { - B } \leq \exp ( w ^ { \top } \phi ( s , b ) ) \leq e ^ { B } , ~ e ^ { - B } \leq \exp ( \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) ) \leq e ^ { B } .
$$

Summing the lower and upper bounds over all legal tokens gives

$$
| \mathcal { B } ( s ) | e ^ { - B } \leq \sum _ { b \in \mathcal { B } ( s ) } \exp ( w ^ { \top } \phi ( s , b ) ) \leq | \mathcal { B } ( s ) | e ^ { B } , \ | \mathcal { B } ( s ) | e ^ { - B } \leq \sum _ { b \in \mathcal { B } ( s ) } \exp ( \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) ) \leq | \mathcal { B } ( s ) | e ^ { B } .
$$

The numerators and denominators are positive. Therefore, substituting the preceding bounds yields

$$
\pi _ { w } ( a \mid s ) \geq { \frac { e ^ { - B } } { | { \mathcal { B } } ( s ) | e ^ { B } } } = { \frac { e ^ { - 2 B } } { | { \mathcal { B } } ( s ) | } } , \pi _ { w } ( a \mid s ) \leq { \frac { e ^ { B } } { | { \mathcal { B } } ( s ) | e ^ { - B } } } = { \frac { e ^ { 2 B } } { | { \mathcal { B } } ( s ) | } } ,\tag{C.13}
$$

$$
\pi _ { \mathrm { s t u } , \theta } ( a \mid s ) \geq { \frac { e ^ { - B } } { | { \mathcal { B } } ( s ) | e ^ { B } } } = { \frac { e ^ { - 2 B } } { | { \mathcal { B } } ( s ) | } } , \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) \leq { \frac { e ^ { B } } { | { \mathcal { B } } ( s ) | e ^ { - B } } } = { \frac { e ^ { 2 B } } { | { \mathcal { B } } ( s ) | } } .\tag{C.14}
$$

Hence, for $\pi _ { s t u , \theta }$ and $\pi _ { w } .$ , we have

$$
\frac { \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) } { \pi _ { w } ( a \mid s ) } \geq \frac { e ^ { - 2 B } / | \mathcal { B } ( s ) | } { e ^ { 2 B } / | \mathcal { B } ( s ) | } = e ^ { - 4 B } , \frac { \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) } { \pi _ { w } ( a \mid s ) } \leq \frac { e ^ { 2 B } / | \mathcal { B } ( s ) | } { e ^ { - 2 B } / | \mathcal { B } ( s ) | } = e ^ { 4 B } .
$$

We thus obtain $\begin{array} { r } { - 4 B \le \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a | s ) } { \pi _ { w } ( a | s ) } \le 4 B } \end{array}$ . Finally, we have that

$$
\left| \log { \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } { \pi _ { w } ( a _ { 1 : H } \mid x ) } } \right| = \left| \log { \prod _ { h = 1 } ^ { H } \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { h } ) } { \pi _ { w } ( a _ { h } \mid s _ { h } ) } } \right| = \left| \sum _ { h = 1 } ^ { H } \log { \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { h } ) } { \pi _ { w } ( a _ { h } \mid s _ { h } ) } } \right| \leq \sum _ { h = 1 } ^ { H } \left| \log { \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { h } ) } { \pi _ { w } ( a _ { h } \mid s _ { h } ) } } \right| \leq 4 B H .
$$

In particular, setting $w = w _ { t }$ and $\theta = \vartheta$ gives

$$
| Z _ { t } ( \vartheta ; x , a _ { 1 : H } ) | = \lambda \left| \log \frac { \pi _ { \mathrm { s t u } , \vartheta } ( a _ { 1 : H } \mid x ) } { \pi _ { w _ { t } } ( a _ { 1 : H } \mid x ) } \right| \leq 4 \lambda B H .
$$

This proves (B.9).

Proof of Lemma B.5. Starting from the definition of $C _ { w } ( \theta )$ , applying Lemma B.4, we have

$$
\begin{array} { l } { \displaystyle C _ { w } ( \theta ) = \frac { \lambda } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } { \pi _ { w } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } } \\ { \displaystyle \quad \le \frac { \lambda } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) \left. \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } { \pi _ { w } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } \right. } \\ { \displaystyle \quad \le \frac { 4 \lambda B H } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) = 4 \lambda B H . } \end{array}
$$

For the lower bound, apply the KL non-negativity and we immediately observe that

$$
C _ { w } ( \theta ) = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta } ( \cdot  { | } \ \widetilde { x } _ { j } )  { | | } \ \pi _ { w } ( \cdot  { | } \ \widetilde { x } _ { j } ) ) \geq 0 .
$$

Here $\lambda / m > 0$ , so summing the nonnegative divergences preserves the inequality. Together with the upper bound, this proves (B.10).

Now, we bound the derivative of $C _ { w } .$ , define the trajectory score

$$
S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) : = \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) .
$$

We first bound $ { S _ { \mathrm { s t u } , \theta } }$ . Fix a prompt x and parameter $\theta ,$ by direct diferentiation, we have

$$
S _ { \mathrm { s t u } , \theta } ( \boldsymbol { x } , \boldsymbol { a } _ { 1 : H } ) = \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( \boldsymbol { a } _ { 1 : H } \mid \boldsymbol { x } ) = \nabla _ { \theta } \log \prod _ { h = 1 } ^ { H } \pi _ { \mathrm { s t u } , \theta } ( \boldsymbol { a } _ { h } \mid \boldsymbol { s } _ { h } ) = \sum _ { h = 1 } ^ { H } \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( \boldsymbol { a } _ { h } \mid \boldsymbol { s } _ { h } ) .
$$

We evaluate each derivative in this expression, by the linear softmax parametrization, we have

$$
\log \pi _ { \operatorname { s t u } , \theta } ( a _ { h } \mid s _ { h } ) = \theta ^ { \top } \phi _ { \operatorname { s t u } } ( s _ { h } , a _ { h } ) - \log \left[ \sum _ { b \in \mathcal { B } ( s _ { h } ) } e ^ { \theta ^ { \top } \phi _ { \operatorname { s t u } } ( s _ { h } , b ) } \right] ,
$$

$$
\begin{array} { l } { \displaystyle \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { h } ) = \phi _ { \mathrm { s t u } } ( s _ { h } , a _ { h } ) - \frac { \sum _ { b \in \mathcal { B } ( s _ { h } ) } e ^ { \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s _ { h } , b ) } \phi _ { \mathrm { s t u } } ( s _ { h } , b ) } { \sum _ { c \in \mathcal { B } ( s _ { h } ) } e ^ { \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s _ { h } , c ) } } } \\ { = \phi _ { \mathrm { s t u } } ( s _ { h } , a _ { h } ) - \displaystyle \sum _ { b \in \mathcal { B } ( s _ { h } ) } \frac { e ^ { \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s _ { h } , b ) } } { \sum _ { c \in \mathcal { B } ( s _ { h } ) } e ^ { \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s _ { h } , c ) } } \phi _ { \mathrm { s t u } } ( s _ { h } , b ) } \\ { = \phi _ { \mathrm { s t u } } ( s _ { h } , a _ { h } ) - \displaystyle \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) } \\ { = : D _ { \theta , h } ( x , a _ { 1 : h } ) . } \end{array}
$$

Substituting the computed token derivative into the score expansion at the start of this step yields

$$
S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) = \sum _ { h = 1 } ^ { H } \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { h } ) = \sum _ { h = 1 } ^ { H } D _ { \theta , h } ( x , a _ { 1 : h } ) .
$$

The pointwise norm of each increment is bounded by

$$
\begin{array} { r l } { \| D _ { \theta , h } ( x , a _ { 1 : h } ) \| _ { 2 } = \displaystyle \left\| \phi _ { \mathrm { s t n } } ( s _ { h } , a _ { h } ) - \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t n } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t n } } ( s _ { h } , b ) \right\| _ { 2 } } & { } \\ { \leq \| \phi _ { \mathrm { s t n } } ( s _ { h } , a _ { h } ) \| _ { 2 } + \displaystyle \left\| \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t n } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t n } } ( s _ { h } , b ) \right\| _ { 2 } } & { } \\ { \leq 1 + \displaystyle \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t n } , \theta } ( b \mid s _ { h } ) \| \phi _ { \mathrm { s t n } } ( s _ { h } , b ) \| _ { 2 } } & { } \\ { \leq 1 + \displaystyle \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t n } , \theta } ( b \mid s _ { h } ) = 2 . } & { } \end{array}
$$

Consequently, we have

$$
\| S _ { \mathrm { s t u } , \theta } ( \boldsymbol { x } , a _ { 1 : H } ) \| _ { 2 } = \left\| \sum _ { h = 1 } ^ { H } D _ { \theta , h } ( \boldsymbol { x } , a _ { 1 : h } ) \right\| _ { 2 } \leq \sum _ { h = 1 } ^ { H } \| D _ { \theta , h } ( \boldsymbol { x } , a _ { 1 : h } ) \| _ { 2 } \leq 2 H .\tag{C.15}
$$

To obtain the sharper second-moment bound, we use the conditional centering of the increments.

Given $x , a _ { 1 : h - 1 }$ , the next token has law $a _ { h } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid s _ { h } )$ . Hence, we have

$$
\begin{array} { l } { { \mathbb { E } _ { a _ { h } \sim \pi _ { \mathrm { s t u } , \theta } \cdot \{ \vert s _ { h } ) } } [ D _ { \theta , h } ( x , a _ { 1 : h } ) \vert x , a _ { 1 : h - 1 } ] }  \\ { { = \displaystyle \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) [ \phi _ { \mathrm { s t u } } ( s _ { h } , b ) - \sum _ { c \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( c \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , c ) ] } } \\ { { = \displaystyle \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) - [ \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) ] [ \sum _ { c \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( c \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , c ) ] } } \\ { { = \displaystyle \sum _ { b \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) - \sum _ { c \in \mathcal { B } ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( c \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , c ) = 0 . } } \end{array}
$$

The second equality uses that the probabilities sum to one. For the conditional second moment, expanding the square using $\| u - v \| _ { 2 } ^ { 2 } = \| u \| _ { 2 } ^ { 2 } - 2 u ^ { \top } v + \| v \| _ { 2 } ^ { 2 }$ , we get

$$
\begin{array} { r l } & { \mathbb { E } _ { \varphi \sim \infty } \lambda \leq \lambda \leq \sqrt { \pi } ( \operatorname* { P o r } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } ) \Big \lVert \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \Big \lVert \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \Big \rVert ^ { 2 } } \\ & { = \displaystyle \sum _ { \ell = 0 } ^ { \infty } \sum _ { \ell \in \mathcal { N } / \mathcal { N } / \mathcal { N } }  \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta }  ^ { 2 } } \\ &  = \displaystyle \sum _ { \ell = 0 } ^ { \infty } \sum _ { \ell \in \mathcal { N } / \mathcal { N } / \mathcal { N } }  \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta } \hat { \sigma } _ { \theta }  ^ { 2 } \displaystyle \sum _ { \ell = 0 } ^ { \infty } \sum _ { \ell \in \mathcal { N } / \mathcal { N } / \mathcal { N } }  \sum _ { \ell = 0 } ^ { \infty } \varphi _ { \theta } \sum _ { \ell = 0 } ^ { \infty } \varphi _ \end{array}
$$

If $h < k$ , then $D _ { \theta , h } ( x , a _ { 1 : h } )$ is already determined by $x , a _ { 1 : k - 1 }$ . Marginalizing the sufix after time

k and then applying iterated expectation, we get

$$
\begin{array} { r l } & { \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert x ) } \left[ D _ { \theta , h } ( x , a _ { 1 : h } ) ^ { \top } D _ { \theta , k } ( x , a _ { 1 : k } ) \right] } \\ & { = \mathbb { E } _ { a _ { 1 : k - 1 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert x ) } \left[ \mathbb { E } _ { a _ { k } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert s _ { k } ) } \left[ D _ { \theta , h } ( x , a _ { 1 : h } ) ^ { \top } D _ { \theta , k } ( x , a _ { 1 : k } ) \Big \vert x , a _ { 1 : k - 1 } \right] \right] } \\ & { = \mathbb { E } _ { a _ { 1 : k - 1 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert x ) } \left[ D _ { \theta , h } ( x , a _ { 1 : h } ) ^ { \top } \mathbb { E } _ { a _ { k } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert s _ { k } ) } \big [ D _ { \theta , k } ( x , a _ { 1 : k } ) \vert x , a _ { 1 : k - 1 } \big ] \right] } \\ & { = \mathbb { E } _ { a _ { 1 : k - 1 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert x ) } \left[ D _ { \theta , h } ( x , a _ { 1 : h } ) ^ { \top } 0 \right] = 0 . } \end{array}
$$

Pulling the earlier increment outside the inner expectation is valid precisely because the earlier increment is measurable given that prefix.

By the linearity of conditional means, we further have

$$
\begin{array} { l } { \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | x ) } [ S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) ] = \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { a _ { 1 : h - 1 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | x ) } \left[ \mathbb { E } _ { a _ { h } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | s _ { h } ) } [ D _ { \theta , h } ( x , a _ { 1 : h } ) | x , a _ { 1 : h - 1 } ] \right] } \\ { \displaystyle \qquad = \sum _ { h = 1 } ^ { H } \mathbb { E } _ { a _ { 1 : h - 1 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | x ) } [ 0 ] = 0 . } \end{array}
$$

When $h = 1$ , the prefix is empty and its expectation has just one possible value. Expanding the square of the sum of the increments, including all cross terms, gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathrm { q } _ { 1 / N ^ { - \eta } \times \pi _ { 0 } , \boldsymbol { \hat { q } } ( \cdot ) } ; \boldsymbol { \mathcal { E } } } \Big [ \big \| S _ { \mathrm { s t } , \boldsymbol { \hat { q } } ( \cdot ) , \boldsymbol { \hat { q } } ( \cdot ) } \big \| _ { 2 } ^ { 2 } \Big ] } \\ & { = \mathbb { E } _ { \mathrm { q } _ { 1 / N ^ { - \eta } \times \pi _ { 0 } , \boldsymbol { \hat { q } } ( \cdot ) } ; \boldsymbol { \mathcal { E } } } \Big [ \bigg \| \displaystyle \sum _ { h = 1 } ^ { H } D _ { \phi , h } ( x , a _ { 1 : h } ) \bigg \| _ { 2 } ^ { 2 } \Big ] } \\ & { = \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { \alpha _ { 1 / N ^ { - \eta } \pi _ { 0 } , \boldsymbol { \hat { q } } ( \cdot ) } ; \boldsymbol { \mathcal { E } } } \Big [ \big \| D _ { \phi , h } \big ( x , a _ { 1 : h } \big ) \big \| _ { 2 } ^ { 2 } \Big ] + 2 \sum _ { 1 \leq h < \chi \leq \mu } \mathbb { E } _ { \boldsymbol { \pi } _ { 1 / N ^ { - \eta } \times \pi _ { 0 } , \boldsymbol { \hat { q } } ( \cdot ) } ; \boldsymbol { \mathcal { E } } } \big [ D _ { \phi , h } \big ( x , a _ { 1 : h } \big ) ^ { \top } D _ { \phi , h } \big ( x , a _ { 1 : h } \big ) \Big ] } \\ &  = \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { \alpha _ { 1 / N ^ { - \eta } \times \pi _ { 0 } , \boldsymbol { \hat { q } } ( \cdot ) } ; \boldsymbol { \mathcal { E } } } \bigg [ \mathbb { E } _ { \boldsymbol { \pi } _ { h } \sim \pi _ { 0 } , \boldsymbol { \hat { q } } ( \cdot ) x } \bigg [ \big \| D _ { \phi , h } \big ( x , a _ { 1 : h } \big ) \big \| _ { 2 } ^ { 2 } \Big ] x , a _ { 1 : h - 1 } \bigg ] \ \end{array}
$$

We have therefore proved the two score identities

$$
\begin{array} { r l } { \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | x ) } [ S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) ] = 0 , \ \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | x ) } \left[ \| S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) \| _ { 2 } ^ { 2 } \right] } & { \le H . } \end{array}\tag{C.16}
$$

Now, we can bound $\nabla _ { \boldsymbol { \theta } } C _ { w } ( \boldsymbol { \theta } )$ . Fix w while diferentiating with respect to $\theta .$

$$
\begin{array} { r l r } & { } & { \nabla _ { \theta } C _ { w } ( \theta ) = \nabla _ { \theta } \left[ \frac { \displaystyle \lambda } { \displaystyle m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in \mathcal { A } ( \tilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } { \pi _ { w } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } \right] } \\ & { } & { = \frac { \displaystyle \lambda } { \displaystyle m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in \mathcal { A } ( \tilde { x } _ { j } ) } \nabla _ { \theta } \left[ \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } { \pi _ { w } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } \right] . } \end{array}
$$

We now expand the derivative of each summand. First, by direct algebra, for any $x ,$ we have

$$
\begin{array} { r l } & { \nabla _ { \theta } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) = \nabla _ { \theta } \exp [ \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) ] = \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } \\ & { \phantom { { = } } = \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) . } \end{array}
$$

Moreover, $\pi _ { w }$ does not depend on $\theta ,$ and hence we have

$$
\nabla _ { \theta } \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } { \pi _ { w } ( a _ { 1 : H } \mid x ) } = \nabla _ { \theta } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) - \nabla _ { \theta } \log \pi _ { w } ( a _ { 1 : H } \mid x ) = S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) - 0 .
$$

Therefore, by the product rule of diferentiation, we have

$$
\begin{array} { r l } & { \quad \nabla _ { \theta } \left[ \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } { \pi _ { w } ( a _ { 1 : H } \mid x ) } \right] } \\ & { = \left[ \nabla _ { \theta } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) \right] \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } { \pi _ { w } ( a _ { 1 : H } \mid x ) } + \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) \nabla _ { \theta } \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } { \pi _ { w } ( a _ { 1 : H } \mid x ) } } \\ & { = \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } { \pi _ { w } ( a _ { 1 : H } \mid x ) } + \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) . } \end{array}
$$

For the second term, its sum is the score expectation already evaluated in (C.16):

$$
\sum _ { a _ { 1 : H } \in A ( x ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) = \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert x ) } [ S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) ] = 0 .
$$

The first equality is the definition of expectation and the second is the first identity in (C.16), applied to this $x$ and $\theta .$ Substituting both product-rule terms into the definition of $C _ { w }$ , and then using this zero sum, we obtain

$$
\begin{array} { r l } & { \nabla _ { \theta } C _ { w } ( \theta ) = \frac { \lambda } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \tilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } \big ( a _ { 1 : H } \mid \tilde { x } _ { j } \big ) S _ { \mathrm { s t u } , \theta } \big ( \tilde { x } _ { j } , a _ { 1 : H } \big ) \times \log \frac { \pi _ { \mathrm { s t u } , \theta } \big ( a _ { 1 : H } \mid \tilde { x } _ { j } \big ) } { \pi _ { w } \big ( a _ { 1 : H } \mid \tilde { x } _ { j } \big ) } } \\ & { \qquad + \frac { \lambda } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \tilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } \big ( a _ { 1 : H } \mid \tilde { x } _ { j } \big ) S _ { \mathrm { s t u } , \theta } \big ( \tilde { x } _ { j } , a _ { 1 : H } \big ) } \\ & { \qquad = 0 } \\ & { \quad \quad = \frac { \lambda } { m } \displaystyle \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } \cdot \mid \tilde { x } _ { j } } \left[ S _ { \mathrm { s t u } , \theta } \big ( \tilde { x } _ { j } , a _ { 1 : H } \big ) \times \log \frac { \pi _ { \mathrm { s t u } , \theta } \big ( a _ { 1 : H } \mid \tilde { x } _ { j } \big ) } { \pi _ { w } \big ( a _ { 1 : H } \mid \tilde { x } _ { j } \big ) } \right] . } \end{array}\tag{C.17}
$$

In the last line, we rewrite the finite weighted sum as an expectation.

The quantity we want to control has the exact representation

$$
| C _ { w } ( \theta ) - C _ { w } ( \vartheta ) | = \left| \int _ { 0 } ^ { 1 } \frac { d } { d u } C _ { w } \Big ( \vartheta + u ( \theta - \vartheta ) \Big ) d u \right| = \left| \int _ { 0 } ^ { 1 } \nabla _ { \theta } C _ { w } \Big ( \vartheta + u \big ( \theta - \vartheta \big ) \Big ) ^ { \top } \big ( \theta - \vartheta \big ) d u \right| .\tag{C.18}
$$

These identities follow from the fundamental theorem of calculus. We bound the gradient appearing here first and then return to this expression. By the triangle inequality, the pointwise log-ratio bound in Lemma B.4, we have

$$
\begin{array} { r l } { \| \nabla _ { \theta } C _ { w } ( \theta ) \| _ { 2 } = \displaystyle \frac { \lambda } { m } | \sum _ { j = 1 , a \in A \setminus \theta } ^ { m } \sum _ { \pi \leq u , \theta } ( a _ { 1 : H } | \ \tilde { x } _ { j }  S _ { \cup \cup \theta } ( \overline { { x } } _ { j } , a _ { 1 : H } ) \times \log \frac { \pi _ { \mathrm { s t a n } , \theta } ( a _ { 1 : H } | \ \tilde { x } _ { j }  | } { \pi _ { \mathrm { s t a n } , H } ^ { \theta } ( a _ { 1 : H } | \tilde { x } _ { j }  ) } | | } & \\ { \leq \displaystyle \frac { \lambda } { m } \sum _ { j = 1 } ^ { N } \mathbb { E } _ { a _ { 1 : H } < v _ { a _ { 1 : H } , a _ { 1 : H } } ( \overline { { x } } _ { j } ) } \Bigg [ \| S _ { \mathrm { s t a n } , \theta } ( \tilde { x } _ { j } , a _ { 1 : H } ) \| _ { 2 } \times \log \frac { \pi _ { \mathrm { s t a n } , \theta } ( a _ { 1 : H } | \tilde { x } _ { j }  | } { \pi _ { \mathrm { s t a n } , H } ^ { \theta } ( a _ { 1 : H } | \tilde { x } _ { j }  ) } \Bigg | \Bigg ] } & \\ { \leq \displaystyle \frac { 4 \lambda B H } { m } \displaystyle \sum _ { j = 1 } ^ { N } \mathbb { E } _ { a _ { 1 : H } \sim v _ { a _ { 1 : H } , a _ { 1 : H } } ( \overline { { x } } _ { j } ) } \Big [ \| S _ { \mathrm { e t a n } , \theta } ( \tilde { x } _ { j } , a _ { 1 : H } ) \| _ { 2 } \Big ] } & \\  \leq \displaystyle \frac { 4 \lambda B H } { m } \displaystyle \sum _ { j = 1 } ^ { N } \Big ( \mathbb { E } _ { a _ { 1 : H } \sim v _ { a _ { 1 : H } , a _ { 1 : H } } ( \overline { { x } } _ { j } ) } \Big [ \| S _  \ \end{array}
$$

In the second inequality, we apply Lemma B.4 and in the last inequality, we apply (C.16). For $u \in [ 0 , 1 ]$ , the segment between ϑ and θ remains inside the student ball because

$$
\begin{array} { r } { \| \vartheta + u ( \theta - \vartheta ) \| _ { 2 } = \| ( 1 - u ) \vartheta + u \theta \| _ { 2 } \leq ( 1 - u ) \| \vartheta \| _ { 2 } + u \| \theta \| _ { 2 } \leq ( 1 - u ) B + u B = B . } \end{array}
$$

Thus, the gradient bound applies at every point of the segment in (C.18). Return to that representation and we get

$$
\begin{array} { r l } { \displaystyle \lvert C _ { w } ( \theta ) - C _ { w } ( \vartheta ) \rvert \le \int _ { 0 } ^ { 1 } \Big \lvert \nabla _ { \theta } C _ { w } \Big ( \vartheta + u ( \theta - \vartheta ) \Big ) ^ { \top } ( \theta - \vartheta ) \Big \rvert d u } & { } \\ { \displaystyle \le \lVert \theta - \vartheta \rVert _ { 2 } \int _ { 0 } ^ { 1 } \Big \lVert \nabla _ { \theta } C _ { w } \Big ( \vartheta + u ( \theta - \vartheta ) \Big ) \Big \rVert _ { 2 } d u } & { } \\ { \displaystyle \le \lVert \theta - \vartheta \rVert _ { 2 } \int _ { 0 } ^ { 1 } 4 \lambda B H ^ { 3 / 2 } d u = 4 \lambda B H ^ { 3 / 2 } \lVert \theta - \vartheta \rVert _ { 2 } . } \end{array}
$$

This proves (B.11). To prove (B.12), we start with the diference that must be bounded and expand

two costs, we have

$$
\begin{array} { l } { \displaystyle C _ { w } ( \theta ) - C _ { v } ( \theta ) = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \tilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) \times [ \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } { \pi _ { w } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } - \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } { \pi _ { v } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) } ] } \\ { \displaystyle \qquad = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \tilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) \times \Big [ - \log \pi _ { w } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) + \log \pi _ { v } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) \Big ] } \\ { \displaystyle \qquad = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \tilde { x } _ { i } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) \times [ \log \pi _ { v } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) - \log \pi _ { w } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) ] . } \end{array}\tag{C.19}
$$

Thus, it sufices to control the diference of calibration log probabilities. We derive that control from the calibration softmax itself:

$$
\log \pi _ { w } ( a \mid s ) = w ^ { \top } \phi ( s , a ) - \log \left[ \sum _ { b \in \mathcal { B } ( s ) } e ^ { w ^ { \top } \phi ( s , b ) } \right] ,
$$

$$
\begin{array} { c l } { \displaystyle \nabla _ { w } \log \pi _ { w } ( a \mid s ) = \phi ( s , a ) - \frac { \sum _ { b \in \mathbb { B } ( s ) } e ^ { w ^ { \top } \phi ( s , b ) } \phi ( s , b ) } { \sum _ { c \in \mathbb { B } ( s ) } e ^ { w ^ { \top } \phi ( s , c ) } } = \phi ( s , a ) - \displaystyle \sum _ { b \in \mathbb { B } ( s ) } \frac { e ^ { w ^ { \top } \phi ( s , b ) } } { \sum _ { c \in \mathbb { B } ( s ) } e ^ { w ^ { \top } \phi ( s , c ) } } \phi ( s , b ) } \\ { \displaystyle = \phi ( s , a ) - \displaystyle \sum _ { b \in \mathbb { B } ( s ) } \pi _ { w } ( b \mid s ) \phi ( s , b ) . } \end{array}
$$

We bound this derivative by the following:

$$
\begin{array} { l } { \displaystyle \| \nabla _ { w } \log \pi _ { w } ( a \mid s ) \| _ { 2 } = \left\| \phi ( s , a ) - \displaystyle \sum _ { b \in B ( s ) } \pi _ { w } ( b \mid s ) \phi ( s , b ) \right\| _ { 2 } \leq \| \phi ( s , a ) \| _ { 2 } + \left\| \displaystyle \sum _ { b \in B ( s ) } \pi _ { w } ( b \mid s ) \phi ( s , b ) \right\| _ { 2 } } \\ { \displaystyle \leq 1 + \displaystyle \sum _ { b \in B ( s ) } \pi _ { w } ( b \mid s ) \| \phi ( s , b ) \| _ { 2 } \leq 1 + \displaystyle \sum _ { b \in B ( s ) } \pi _ { w } ( b \mid s ) = 2 . } \end{array}
$$

Since W is convex, $v + u ( w - v ) \in W$ for $u \in [ 0 , 1 ]$ . The same one-dimensional integration argument, now displayed for the token log probability, gives

$$
\begin{array} { l } { \displaystyle \log \pi _ { w } ( a \mid s ) - \log \pi _ { v } ( a \mid s ) = \int _ { 0 } ^ { 1 } \frac { d } { d u } \log \pi _ { v + u ( w - v ) } ( a \mid s ) d u } \\ { \displaystyle \phantom { \frac { 1 } { 1 } \log \pi _ { w } ( a \mid s ) \ln \left( \pi _ { v } ( a \mid s ) \right) } = \int _ { 0 } ^ { 1 } \nabla _ { w } \log \pi _ { w } ( a \mid s ) \big \rvert _ { w = v + u ( w - v ) } ^ { \top } \left( w - v \right) d u . } \end{array}
$$

Consequently, we have

$$
\begin{array} { r l } { | \log \pi _ { w } ( a \mid s ) - \log \pi _ { v } ( a \mid s ) | \le \displaystyle \int _ { 0 } ^ { 1 } \left\| \nabla _ { w } \log \pi _ { w } ( a \mid s ) | _ { w = v + u ( w - v ) } \right\| _ { 2 } \| w - v \| _ { 2 } d u } & { } \\ & { \le \displaystyle \int _ { 0 } ^ { 1 } 2 \| w - v \| _ { 2 } d u = 2 \| w - v \| _ { 2 } . } \end{array}
$$

Thus, we have

$$
\begin{array} { l } { \displaystyle \log \pi _ { w } ( a _ { 1 : H } \mid x ) - \log \pi _ { v } ( a _ { 1 : H } \mid x ) \big | = \displaystyle \left| \sum _ { h = 1 } ^ { H } \Bigl [ \log \pi _ { w } ( a _ { h } \mid s _ { h } ) - \log \pi _ { v } ( a _ { h } \mid s _ { h } ) \Bigr ] \right| } \\ { \displaystyle \qquad \leq \sum _ { h = 1 } ^ { H } \vert \log \pi _ { w } ( a _ { h } \mid s _ { h } ) - \log \pi _ { v } ( a _ { h } \mid s _ { h } ) \vert } \\ { \displaystyle \qquad \leq \sum _ { h = 1 } ^ { H } 2 \Vert w - v \Vert _ { 2 } = 2 H \Vert w - v \Vert _ { 2 } . } \end{array}
$$

Now, we return to the cost-diference representation (C.19). Plugging the bound derived above back, we obtain,

$$
\begin{array} { l } { \displaystyle | C _ { w } ( \theta ) - C _ { v } ( \theta ) | \le \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x _ { j } } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } | \widetilde { x } _ { j } ) \times | \log \pi _ { v } ( a _ { 1 : H } | \widetilde { x } _ { j } ) - \log \pi _ { w } ( a _ { 1 : H } | \widetilde { x } _ { j } ) | } \\ { \displaystyle \quad \le \frac { 2 \lambda H \| w - v \| _ { 2 } } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x _ { j } } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } | \widetilde { x } _ { j } ) } \\ { \displaystyle \quad = \frac { 2 \lambda H \| w - v \| _ { 2 } } { m } \sum _ { j = 1 } ^ { m } 1 = 2 \lambda H \| w - v \| _ { 2 } . } \end{array}
$$

The right-hand side is independent of θ. Taking the supremum over $\theta \in \Theta$ proves (B.12). □

Proof of Lemma B.7. We first verify analyticity, which will also be used in the two Lojasiewicz arguments below. For each fixed legal state s and token $a \in B ( s )$ , the softmax expressions are

$$
\begin{array} { r l } & { \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) = \frac { e ^ { \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , a ) } } { \sum _ { b \in \mathcal { B } ( s ) } e ^ { \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) } } , } \\ & { \log \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) = \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , a ) - \log \left[ \sum _ { b \in \mathcal { B } ( s ) } e ^ { \theta ^ { \top } \phi _ { \mathrm { s t u } } ( s , b ) } \right] . } \end{array}
$$

The exponentials of these linear functions are analytic, and their finite sum is strictly positive on $\mathbb { R } ^ { d }$ . Since reciprocals and logarithms are analytic on $( 0 , \infty )$ , both displayed expressions are analytic. For each fixed feasible answer, the chain rule for trajectory probabilities gives

$$
\begin{array} { r l } & { ~ \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) = \displaystyle \prod _ { h = 1 } ^ { H } \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) , } \\ & { \log \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) = \displaystyle \sum _ { h = 1 } ^ { H } \log \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid x , a _ { 1 : h - 1 } ) . } \end{array}
$$

Finite products and sums preserve analyticity. For any fixed positive answer laws $Q _ { j }$ , expanding the average KL gives

$$
\frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } \big ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \ | \ \widetilde { x } _ { j } ) \ | \ Q _ { j } \big ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : \mathcal { H } } \in A ( \widetilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } \big ( a _ { 1 : \mathcal { H } } \ | \ \widetilde { x } _ { j } \big ) \times \big [ \log \pi _ { \mathrm { s t u } , \theta } \big ( a _ { 1 : \mathcal { H } } \ | \ \widetilde { x } _ { j } \big ) - \log Q _ { j } \big ( a _ { 1 : \mathcal { H } } \big ) \big ] .
$$

Each log $Q _ { j } ( a _ { 1 : H } )$ is a finite constant, and the sum has finitely many analytic summands. Thus this function is analytic. Choosing $Q _ { j } = \pi _ { w } ( { \cdot } \mid { \widetilde x } _ { j } )$ and multiplying by λ proves the assertion for $C _ { w } ;$ choosing $Q _ { j } = \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \mid \widetilde { x } _ { j } )$ proves it for $\kappa _ { \lambda , m }$ . In particular, $F = C _ { w _ { \lambda } ^ { \star } }$ and $\Delta _ { \lambda , m } = F - F ( \theta _ { \lambda , m } ^ { \dagger } )$ are analytic.

We next bound the Hessian of $C _ { w }$ . For this calculation, define the unscaled trajectory log ratio

$$
\ell _ { w } ( \theta ; x , a _ { 1 : H } ) : = \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid x ) } { \pi _ { w } ( a _ { 1 : H } \mid x ) } .
$$

By Lemma B.4, $| \ell _ { w } | \le 4 B H$ on Θ. Recall from (C.17) that

$$
\nabla _ { \theta } C _ { w } ( \theta ) = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) \ell _ { w } ( \theta ; \widetilde { x } _ { j } , a _ { 1 : H } ) S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : H } ) .
$$

We take the derivative again and use the product rule to compute the Hessian matrix

$$
\begin{array} { l } { \displaystyle \nabla _ { \theta } ^ { 2 } C _ { w } ( \theta ) = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } \Big [ \Big ( \ell _ { w } ( \theta ; \widetilde { x } _ { j } , a _ { 1 : H } ) + 1 \Big ) S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : H } ) S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : H } ) ^ { \top } } \\ { \displaystyle \qquad + \ell _ { w } ( \theta ; \widetilde { x } _ { j } , a _ { 1 : H } ) \nabla _ { \theta } S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \Big ] . } \end{array}\tag{C.20}
$$

We already know from (C.16) that the expected squared score norm is at most H. To bound its derivative, we diferentiate the token-score formula in the proof of Lemma B.5:

$$
\begin{array} { r l } { \nabla _ { \theta } S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) = \displaystyle - \sum _ { h = 1 } ^ { H } \left[ \sum _ { b \in B ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) ^ { \top } \right. } & { } \\ { \displaystyle - \left. \left( \sum _ { b \in B ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) \right) \left( \sum _ { b \in B ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \phi _ { \mathrm { s t u } } ( s _ { h } , b ) \right) ^ { \top } \right] . } & { } \end{array}
$$

This is a covariance matrix and is positive semidefinite. For any unit vector $\boldsymbol { v } \in \mathbb { R } ^ { d }$ , we have

$$
\begin{array} { r l } & { 0 \leq - v ^ { \top } \nabla _ { \theta } S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) v } \\ & { \quad = \displaystyle \sum _ { h = 1 } ^ { H } \left[ \sum _ { b \in B ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \Big ( v ^ { \top } \phi _ { \mathrm { s t u } } \big ( s _ { h } , b \big ) \Big ) ^ { 2 } - \left( \sum _ { b \in B ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) v ^ { \top } \phi _ { \mathrm { s t u } } \big ( s _ { h } , b \big ) \right) ^ { 2 } \right] } \\ & { \quad \leq \displaystyle \sum _ { h = 1 } ^ { H } \sum _ { b \in B ( s _ { h } ) } \pi _ { \mathrm { s t u } , \theta } ( b \mid s _ { h } ) \| v \| _ { 2 } ^ { 2 } \| \phi _ { \mathrm { s t u } } \big ( s _ { h } , b \big ) \| _ { 2 } ^ { 2 } \leq \displaystyle \sum _ { h = 1 } ^ { H } 1 = H . } \end{array}
$$

The first upper bound discards a nonnegative square and applies Cauchy-Schwarz; the last uses the feature norm bound and that the token probabilities sum to one. Thus $\| \nabla _ { \theta } S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : H } ) \| _ { \mathrm { o p } } \leq H$

Taking operator norms in (C.20) and using $\| S S ^ { \top } \| _ { \mathrm { o p } } = \| S \| _ { 2 } ^ { 2 }$ now yields

$$
\begin{array} { l } { \displaystyle | | \nabla _ { \theta } ^ { 2 } C _ { w } ( \theta ) | | _ { \mathrm { o p } } \leq \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } \Big [ \big ( 4 B H + 1 \big ) \| S _ { \mathrm { s t u } , \theta } \big ( \widetilde { x } _ { j } , a _ { 1 : H } \big ) \| _ { 2 } ^ { 2 } + 4 B H ^ { 2 } \Big ] } \\ { \displaystyle \leq \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \Big [ \big ( 4 B H + 1 \big ) H + 4 B H ^ { 2 } \Big ] = \lambda H \big ( 1 + 8 B H \big ) = L _ { \mathrm { s t } } . } \end{array}
$$

This proves (B.15). Since Θ is convex, the segment joining ϑ and θ lies in Θ. The fundamental theorem of calculus then gives

$$
\| \nabla _ { \theta } C _ { w } ( \theta ) - \nabla _ { \theta } C _ { w } ( \vartheta ) \| _ { 2 } = \left\| \int _ { 0 } ^ { 1 } \nabla _ { \theta } ^ { 2 } C _ { w } ( \vartheta + u ( \theta - \vartheta ) ) ( \theta - \vartheta ) d u \right\| _ { 2 } \leq \int _ { 0 } ^ { 1 } L _ { \mathrm { s t } } \| \theta - \vartheta \| _ { 2 } d u = L _ { \mathrm { s t } } \| \theta - \vartheta \| _ { 2 } ,
$$

which proves (B.16).

Finally, subtract the two representations in (C.17). The student law and its score agree in both terms, so cancellation of their log probabilities gives

$$
\nabla _ { \theta } C _ { w } ( \theta ) - \nabla _ { \theta } C _ { v } ( \theta ) = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } \left[ S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \log \frac { \pi _ { v } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } { \pi _ { w } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } \right] .
$$

The proof of (B.12) already established the pointwise bound $| \log ( \pi _ { v } ( a _ { 1 : H } \mid x ) / \pi _ { w } ( a _ { 1 : H } \mid x ) ) | \leq$ $2 H \parallel w - v \parallel _ { 2 }$ . Applying that bound and Cauchy-Schwarz to the score expectation, and then using (C.16), we obtain

$$
\begin{array} { r l } & { \| \nabla _ { \theta } C _ { w } ( \theta ) - \nabla _ { \theta } C _ { v } ( \theta ) \| _ { 2 } \leq \frac { 2 \lambda H \| w - v \| _ { 2 } } { m } \displaystyle \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } [ \| S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \| _ { 2 } ] } \\ & { \qquad \leq \frac { 2 \lambda H \| w - v \| _ { 2 } } { m } \displaystyle \sum _ { j = 1 } ^ { m } \left( \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } \left[ \| S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \| _ { 2 } ^ { 2 } \right] \right) ^ { 1 / 2 } } \\ & { \qquad \leq 2 \lambda H ^ { 3 / 2 } \| w - v \| _ { 2 } . } \end{array}
$$

This proves (B.17).

Proof of Lemma B.6. We first compute the conditional mean of each summand, then bound the variance of their average. Conditional on $\mathcal { G } _ { t + 1 } , ~ \theta _ { t }$ and $w _ { t + 1 }$ are fixed. For every ℓ, Algorithm 1 samples $j _ { t + 1 , \ell } ^ { \mathrm { g } }$ uniformly and then samples an answer from the current student at that question. Thus, for $a _ { 1 : H } \in \mathcal { A } ( \widetilde { x } _ { j } )$ , we have

$$
\begin{array} { r l r } { \mathrm { P r } \left( j _ { t + 1 , \ell } ^ { \mathrm { g } } = j , a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } = a _ { 1 : H } \middle | \mathcal { G } _ { t + 1 } \right) = \mathrm { P r } \left( j _ { t + 1 , \ell } ^ { \mathrm { g } } = j \middle | \mathcal { G } _ { t + 1 } \right) \times \mathrm { P r } \left( a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } = a _ { 1 : H } \middle | \mathcal { G } _ { t + 1 } , j _ { t + 1 , \ell } ^ { \mathrm { g } } = j \right) } & \\ { = \frac { 1 } { m } \pi _ { \mathrm { s t u } , \theta _ { t } } ( a _ { 1 : H } \mid \tilde { x } _ { j } ) . } & { \quad { \scriptstyle ( \mathrm { C . 2 1 } ) } } & \end{array}
$$

Starting from the definition of $\widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } .$ , we obtain

$$
\begin{array} { r l } { \mathbb { E } \left[ \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } \Big \vert \mathcal { G } _ { t + 1 } \right] = \mathbb { E } \left[ S _ { \mathrm { s t u } , \theta _ { t } } \big ( \widetilde { x } _ { j _ { t } ^ { \mathrm { g } } + 1 , \ell } , a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } \big ) Z _ { t + 1 } ( \theta _ { t } ; \widetilde { x } _ { j _ { t } ^ { \mathrm { g } } + 1 , \ell } , a _ { t + 1 , \ell , 1 : H } ^ { \mathrm { g } } \big ) \Big \vert \mathcal { G } _ { t + 1 } \right] } & { } \\ { = \frac { 1 } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta _ { t } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j } , a _ { 1 : H } ) Z _ { t + 1 } ( \theta _ { t } ; \widetilde { x } _ { j } , a _ { 1 : H } ) \quad } & { } \\ { = \frac { \lambda } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x } _ { j } ) } \pi _ { \mathrm { s t u } , \theta _ { t } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \log \frac { \pi _ { \mathrm { s t u } , \theta _ { t } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } { \pi _ { w _ { t + 1 } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } } & { } \\ { = \nabla _ { \theta } C _ { w t + 1 } \left( \theta _ { t } \right) = \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) . } & { } \end{array}
$$

The second equality uses (C.21), the third substitutes the definition of $Z _ { t + 1 }$ , and the fourth applies the cost-gradient identity (C.17). By linearity of conditional expectation, we have,

$$
\mathbb { E } \Big [ \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \Big | \mathcal { G } _ { t + 1 } \Big ] = \frac { 1 } { b _ { t + 1 } } \sum _ { \ell = 1 } ^ { b _ { t + 1 } } \mathbb { E } \Big [ \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } \Big | \mathcal { G } _ { t + 1 } \Big ] = \frac { b _ { t + 1 } } { b _ { t + 1 } } \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) = \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) .
$$

To control the variance, we first bound the second moment. Its definition and (C.21) give

$$
\begin{array} { r l } {  { \mathbb { E } [ \| \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t } } \| _ { 2 } ^ { 2 } \Big | \mathcal { G } _ { t + 1 } ] = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : t } \in A ( \widehat { x } _ { j } ) } \pi _ { s _ { t } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } \| S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \| _ { 2 } ^ { 2 } \big [ Z _ { t + 1 } ( \theta _ { t } ; \widetilde { x } _ { j } , a _ { 1 : H } ) \big ] ^ { 2 } } } \\ & { \leq \frac { 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 2 } } { m } \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : t } \in A ( \widetilde { x } _ { j } ) } \pi _ { s _ { t + 1 : H } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) \| S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \| _ { 2 } ^ { 2 } } \\ & { = \frac { 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 2 } } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : H } \sim \pi _ { s _ { t + 1 : a _ { t } } } ( \cdot \vert \widetilde { x } _ { j } ) } \Big [ \big \| S _ { \mathrm { s t u } , \theta _ { t } } ( \widetilde { x } _ { j } , a _ { 1 : H } ) \big \| _ { 2 } ^ { 2 } \Big ] } \\ & { \leq \frac { 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 2 } } { m } \sum _ { j = 1 } ^ { m } H = 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } . } \end{array}
$$

The first inequality uses $| Z _ { t + 1 } | \le 4 \lambda B H$ from Lemma B.4. The last inequality applies the score second-moment bound (C.16) at each target question.

For this proof, we set $e _ { t + 1 , \ell } : = \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } - \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } )$ . The identity above yields $\mathbb { E } [ e _ { t + 1 , \ell } \ | \ \mathcal { G } _ { t + 1 } ] = 0$ Expanding the square yields

$$
\begin{array} { r l } & { \mathbb { E } [ \| e _ { t + 1 , \ell } \| _ { 2 } ^ { 2 } \mid \mathcal { G } _ { t + 1 } ] = \mathbb { E } [ \| \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } \| _ { 2 } ^ { 2 } \mid \mathcal { G } _ { t + 1 } ] - 2 \left. \mathbb { E } [ \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } \mid \mathcal { G } _ { t + 1 } ] , \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) \right. + \| \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad = \mathbb { E } [ \| \widehat { g } _ { t + 1 , \ell } ^ { \mathrm { s t u } } \| _ { 2 } ^ { 2 } \mid \mathcal { G } _ { t + 1 } ] - \| \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) \| _ { 2 } ^ { 2 } \leq 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } . } \end{array}
$$

For $\ell \neq \ell ^ { \prime }$ , the two prompt-answer pairs are independent conditional on $\mathcal { G } _ { t + 1 }$ , and therefore

$$
\begin{array} { r } { \mathbb { E } \left[ \langle e _ { t + 1 , \ell } , e _ { t + 1 , \ell ^ { \prime } } \rangle \mid \mathcal { G } _ { t + 1 } \right] = \langle \mathbb { E } \left[ e _ { t + 1 , \ell } \mid \mathcal { G } _ { t + 1 } \right] , \mathbb { E } \left[ e _ { t + 1 , \ell ^ { \prime } } \mid \mathcal { G } _ { t + 1 } \right] \rangle = 0 . } \end{array}
$$

Using the definition of the averaged estimator, we conclude that

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } - \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) \right. _ { 2 } ^ { 2 } \middle | \mathcal { G } _ { t + 1 } \right] = \mathbb { E } \left[ \left. \frac { 1 } { b _ { t + 1 } } \displaystyle \sum _ { \ell = 1 } ^ { b _ { t + 1 } } e _ { t + 1 , \ell } \right. _ { 2 } ^ { 2 } \middle | \mathcal { G } _ { t + 1 } \right] } \\ & { \qquad = \frac { 1 } { b _ { t + 1 } ^ { 2 } } \displaystyle \sum _ { \ell = 1 } ^ { b _ { t + 1 } } \mathbb { E } \left[ \left. e _ { t + 1 , \ell } \right. _ { 2 } ^ { 2 } \middle | \mathcal { G } _ { t + 1 } \right] + \frac { 2 } { b _ { t + 1 } ^ { 2 } } \displaystyle \sum _ { \ell < \ell ^ { \prime } } \mathbb { E } \left[ \left. e _ { t + 1 , \ell } , e _ { t + 1 , \ell ^ { \prime } } \right. \middle | \mathcal { G } _ { t + 1 } \right] } \\ & { \qquad \leq \frac { b _ { t + 1 } \left( 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } \right) } { b _ { t + 1 } ^ { 2 } } = \frac { 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } } { b _ { t + 1 } } . } \end{array}
$$

This proves (B.14).

Finally, $\theta _ { t }$ is $\mathcal { F } _ { t }$ -measurable, so $\sigma ( w _ { t + 1 } , \theta _ { t } ) \subseteq \mathcal G _ { t + 1 }$ . The tower property gives the remaining conditional identity:

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \middle | w _ { t + 1 } , \theta _ { t } \right] = \mathbb { E } \left[ \mathbb { E } [ \widehat { g } _ { t + 1 } ^ { \mathrm { s t u } } \mid { \mathcal G } _ { t + 1 } ] \middle | w _ { t + 1 } , \theta _ { t } \right] = \mathbb { E } [ \nabla _ { \theta } C _ { w _ { t + 1 } } ( \theta _ { t } ) \vert w _ { t + 1 } , \theta _ { t } ] = \nabla _ { \theta } C _ { t + 1 } ( \theta _ { t } ) . } \end{array}
$$

The last equality holds because the displayed gradient is a function of $w _ { t + 1 }$ and $\theta _ { t }$ . This completes the proof. □

Proof of Lemma B.8. Conditional on $\mathcal { H } _ { t }$ , the candidate policies and $w _ { t }$ are fixed. Since we sample $j _ { t , k , \ell }$ uniformly from $\{ 1 , \ldots , m \}$ and then generate $a _ { t , k , \ell , 1 : H } ^ { \mathrm { v a l } }$ from $\pi _ { \mathrm { s t u } , \boldsymbol { \vartheta } _ { t , k } } ( \cdot \mid \widetilde { x } _ { j _ { t , k , \ell } } )$ , we have

$$
\begin{array} { r l } & { \quad \mathbb E \left[ Z _ { t } ( \vartheta _ { t , k } ; \widetilde { x } _ { j _ { t , k , \ell } } , a _ { t , k , \ell , 1 : H } ^ { \mathrm { v a l } } ) \mid \mathcal H _ { t } \right] } \\ & { = \displaystyle \frac 1 m \sum _ { j = 1 } ^ { m } \sum _ { a _ { 1 : H } \in A ( \widetilde { x } _ { j } ) } ^ { \infty } \pi _ { \mathrm { s t u } , \vartheta _ { t , k } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) \lambda \log \frac { \pi _ { \mathrm { s t u } , \vartheta _ { t , k } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } { \pi _ { w _ { t } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } = C _ { t } ( \vartheta _ { t , k } ) . } \end{array}
$$

By Lemma B.4, the squared summand of $Z _ { t } ( \vartheta _ { t , k } ; \widetilde { x } _ { j _ { t , k , \ell } } , a _ { t , k , \ell , 1 : H } ^ { \mathrm { v a l } } ) , \ell = 1 , \cdot \cdot \cdot , q _ { t }$ is at most $1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 2 }$ Since the samples within a batch are independent and centered after subtracting their mean, the cross terms vanish when expanding the squared sample-mean error and we have

$$
\mathbb { E } [ ( \widehat { C } _ { t , k } - C _ { t } ( \vartheta _ { t , k } ) ) ^ { 2 } \mid \mathcal { H } _ { t } ] = \frac { 1 } { q _ { t } ^ { 2 } } \sum _ { \ell = 1 } ^ { q _ { t } } \operatorname { V a r } \Bigl ( Z _ { t } \bigl ( \vartheta _ { t , k } ; \widetilde { x } _ { j _ { t , k , \ell } } , a _ { t , k , \ell , 1 : H } ^ { \mathrm { v a l } } \bigr ) \mid \mathcal { H } _ { t } \Bigr ) \le \frac { 1 6 \lambda ^ { 2 } B ^ { 2 } H ^ { 2 } } { q _ { t } } .
$$

Then, we apply the Cauchy-Schwarz inequality and get

$$
\mathbb { E } [ \vert \widehat { C } _ { t , k } - C _ { t } ( \vartheta _ { t , k } ) \vert \vert \mathcal { H } _ { t } ] \le \Big ( \mathbb { E } [ ( \widehat { C } _ { t , k } - C _ { t } ( \vartheta _ { t , k } ) ) ^ { 2 } \vert \mathcal { H } _ { t } ] \Big ) ^ { 1 / 2 } \le \frac { 4 \lambda B H } { \sqrt { q _ { t } } } .
$$

The maximum of two nonnegative numbers is no larger than their sum. Consequently, we have

$$
\mathbb { E } \big [ \xi _ { t } ~ | ~ \mathcal { H } _ { t } \big ] \le 2 \sum _ { k = 1 } ^ { 2 } \mathbb { E } \big [ | \widehat C _ { t , k } - C _ { t } ( \vartheta _ { t , k } ) | ~ | ~ \mathcal { H } _ { t } \big ] \le \frac { 1 6 \lambda B H } { \sqrt { q _ { t } } } = \frac { 1 6 \lambda B H } { t + 1 } .
$$

Choose an index $k _ { t } ^ { \star }$ that minimizes the two true costs, only for this proof. The definition of $\xi _ { t }$ and

empirical minimization implies that

$$
C _ { t } ( \theta _ { t } ) = C _ { t } ( \vartheta _ { t , \widehat { k } _ { t } } ) \leq \widehat { C } _ { t , \widehat { k } _ { t } } + \xi _ { t } / 2 \leq \widehat { C } _ { t , k _ { t } ^ { \star } } + \xi _ { t } / 2 \leq C _ { t } ( \vartheta _ { t , k _ { t } ^ { \star } } ) + \xi _ { t } .\tag{C.22}
$$

The first and last inequalities bound the evaluation errors; the middle inequality is the actual two-way selection rule. This proves (B.18) and we finish the proof. □

Proof of Lemma B.9. Recall that $\boldsymbol { F } = C _ { w _ { \lambda } ^ { \star } }$ and $\Delta _ { \lambda , m } ( \theta ) = F ( \theta ) - F ( \theta _ { \lambda , m } ^ { \dagger } )$ . By optimality, the student Lipschitz bound in (B.11), and the diameter 2B of Θ,

$$
0 \leq \Delta _ { \lambda , m } ( \theta ) \leq 4 \lambda B H ^ { 3 / 2 } \| \theta - \theta _ { \lambda , m } ^ { \dagger } \| _ { 2 } \leq 8 \lambda B ^ { 2 } H ^ { 3 / 2 } = M _ { \mathrm { o p t } } .\tag{C.23}
$$

We first show that the exact projected step gives a quadratic improvement when this gap is small, and then use uniform exploration for the remaining values of the gap.

To apply Lemma A.2, we define the extended-real function and the normal cone

$$
\Psi ( \theta ) : = \left\{ \begin{array} { l l } { F ( \theta ) , } & { \theta \in \Theta , } \\ { + \infty , } & { \theta \notin \Theta , } \end{array} \right.
$$

$$
N _ { \Theta } ( \theta ) : = \{ v \in \mathbb { R } ^ { d } : v ^ { \top } ( \vartheta - \theta ) \leq 0 \mathrm { ~ f o r ~ e v e r y ~ } \vartheta \in \Theta \} , \ \theta \in \Theta .
$$

For this smooth function on a closed convex set, We claim that

$$
\partial \Psi ( \theta ) = \nabla _ { \theta } F ( \theta ) + N _ { \Theta } ( \theta ) , \ \theta \in \Theta .\tag{C.24}
$$

For $\theta , \vartheta \in \Theta$ , we first have $F ( \vartheta ) - F ( \theta ) = \nabla _ { \theta } F ( \theta ) ^ { \top } ( \vartheta - \theta ) + o ( \| \vartheta - \theta \| _ { 2 } )$

Hence, by the definition of the Fr´echet subdiferential, we have

$$
v \in \widehat { \partial } \Psi ( \theta ) \ \Longleftrightarrow \ \operatorname* { l i m i n f } _ { \vartheta  \theta , \neq \theta } \frac { F ( \vartheta ) - F ( \theta ) - v ^ { \top } ( \vartheta - \theta ) } { \| \vartheta - \theta \| _ { 2 } } \geq 0 \ \Longleftrightarrow \ \operatorname* { l i m i n f } _ { \vartheta  \theta , \neq \theta } \frac { ( \nabla _ { \theta } F ( \theta ) - v ) ^ { \top } ( \vartheta - \theta ) } { \| \vartheta - \theta \| _ { 2 } } \geq 0 .
$$

On one hand, by the definition of $N _ { \Theta } ( \theta )$ , we know that

$$
\boldsymbol { v } - \nabla _ { \boldsymbol { \theta } } F ( { \boldsymbol { \theta } } ) \in N _ { \Theta } ( { \boldsymbol { \theta } } ) \implies ( \nabla _ { \boldsymbol { \theta } } F ( { \boldsymbol { \theta } } ) - { \boldsymbol { v } } ) ^ { \top } ( { \boldsymbol { \vartheta } } - { \boldsymbol { \theta } } ) \geq 0 \forall { \boldsymbol { \vartheta } } \in \Theta \implies { \boldsymbol { v } } \in \widehat { \partial } \Psi ( { \boldsymbol { \theta } } ) .
$$

Conversely, let $v \in \widehat { \partial } \Psi ( \theta ) . \ \forall \zeta \in \Theta \setminus \{ \theta \}$ , by the convexity of Θ, we have that

$$
\begin{array} { r } { \vartheta _ { u } = \theta + u ( \zeta - \theta ) \in \Theta , \ u \in ( 0 , 1 ] . } \end{array}
$$

Therefore,

$$
0 \le \operatorname* { l i m i n f } _ { u \downarrow 0 } \frac { \left( \nabla _ { \theta } F ( \theta ) - v \right) ^ { \top } ( \vartheta _ { u } - \theta ) } { \| \vartheta _ { u } - \theta \| _ { 2 } } = \operatorname* { l i m i n f } _ { u \downarrow 0 } \frac { u ( \nabla _ { \theta } F ( \theta ) - v ) ^ { \top } ( \zeta - \theta ) } { u \| \zeta - \theta \| _ { 2 } } = \frac { ( \nabla _ { \theta } F ( \theta ) - v ) ^ { \top } ( \zeta - \theta ) } { \| \zeta - \theta \| _ { 2 } } .
$$

Consequently, we obtain $( v - \nabla _ { \boldsymbol { \theta } } F ( { \boldsymbol { \theta } } ) ) ^ { \top } ( { \boldsymbol { \zeta } } - { \boldsymbol { \theta } } ) \leq 0 \forall { \boldsymbol { \zeta } } \in \Theta$ . Thus,

$$
v - \nabla _ { \theta } F ( \theta ) \in N _ { \Theta } ( \theta ) .
$$

Combining both inclusions yields $\widehat { \partial } \Psi ( \theta ) = \nabla _ { \theta } F ( \theta ) + N _ { \Theta } ( \theta )$ . By the definition of the limiting

subdiferential, we know that

$$
v \in \partial \Psi ( \theta ) \iff \exists ( \theta _ { n } , v _ { n } ) _ { n \geq 1 } : \left\{ { \begin{array} { l } { \theta _ { n } \in \Theta , \ \theta _ { n } \to \theta , } \\ { \Psi ( \theta _ { n } ) \to \Psi ( \theta ) , \ v _ { n } \to v , } \\ { v _ { n } \in { \widehat { \partial } } \Psi ( \theta _ { n } ) . } \end{array} } \right.
$$

For such a sequence and every $\zeta \in \Theta$ , we have $( v _ { n } - \nabla _ { \theta } F ( \theta _ { n } ) ) ^ { \top } ( \zeta - \theta _ { n } ) \leq 0$ . Continuity of $\nabla _ { \boldsymbol { \theta } } F$ yields that

$$
( v - \nabla _ { \theta } F ( \theta ) ) ^ { \top } ( \zeta - \theta ) = \operatorname* { l i m } _ { n \to \infty } ( v _ { n } - \nabla _ { \theta } F ( \theta _ { n } ) ) ^ { \top } ( \zeta - \theta _ { n } ) \leq 0 .
$$

Thus, we obtain

$$
\partial \Psi ( \theta ) \subseteq \nabla _ { \theta } F ( \theta ) + N _ { \Theta } ( \theta ) .
$$

For the reverse inclusion, the constant sequences $\theta _ { n } = \theta$ and $v _ { n } = v$ give

$$
v \in \nabla _ { \theta } F ( \theta ) + N _ { \Theta } ( \theta ) = { \widehat { \partial } } \Psi ( \theta ) \implies v \in \partial \Psi ( \theta ) .
$$

Therefore, we have proved the claim.

Lemma B.7 shows that F is analytic. Consequently, the finite graph of Ψ is

$$
\{ ( \theta , u ) \in \mathbb { R } ^ { d } \times \mathbb { R } : \| \theta \| _ { 2 } ^ { 2 } \leq B ^ { 2 } , \ u - F ( \theta ) = 0 \} ,
$$

which is semianalytic and hence subanalytic. Its domain is the closed ball Θ, and Ψ is continuous on this domain. Moreover, optimality of $\theta _ { \lambda , m } ^ { \dagger }$ on every segment in Θ implies that

$$
\nabla _ { \boldsymbol { \theta } } F ( \boldsymbol { \theta } _ { \lambda , m } ^ { \dagger } ) ^ { \top } ( \boldsymbol { \vartheta } - \boldsymbol { \theta } _ { \lambda , m } ^ { \dagger } ) \geq 0 , \ \boldsymbol { \vartheta } \in \Theta .
$$

Thus $- \nabla _ { \theta } F ( \theta _ { \lambda , m } ^ { \dagger } ) \in N _ { \Theta } ( \theta _ { \lambda , m } ^ { \dagger } )$ , and (C.24) gives $0 \in \partial \Psi ( \theta _ { \lambda , m } ^ { \dagger } )$ . All hypotheses of Lemma A.2 therefore hold. There are an open neighborhood U of $\theta _ { \lambda , m } ^ { \dagger } , c _ { \mathrm { l o c } } > 0$ , and $\rho _ { \mathrm { l o c } } \in [ 0 , 1 )$ such that

$$
\begin{array} { r } { \mathrm { d i s t } \big ( 0 , \partial \Psi ( \theta ) \big ) \geq c _ { \mathrm { l o c } } \big [ \Delta _ { \lambda , m } ( \theta ) \big ] ^ { \rho _ { \mathrm { l o c } } } , \ \theta \in U \cap \Theta , \ \Delta _ { \lambda , m } ( \theta ) > 0 . } \end{array}\tag{C.25}
$$

Uniqueness of the minimizer now lets us choose $0 < \delta _ { \mathrm { l o c } } \leq \operatorname* { m i n } \{ 1 , M _ { \mathrm { o p t } } \}$ such that

$$
\{ \theta \in \Theta : \Delta _ { \lambda , m } ( \theta ) \leq \delta _ { \mathrm { l o c } } \} \subset U .
$$

To justify this choice, if $\Theta \backslash U$ is nonempty, its compactness and continuity of $\Delta _ { \lambda , m }$ give an attained minimum there. This minimum is strictly positive because the only zero is $\theta _ { \lambda , m } ^ { \dagger } \in U$ . Choose $\delta _ { \mathrm { l o c } }$ smaller than that minimum. If the complement is empty, any positive value satisfying the displayed upper bound sufices. For $0 < \Delta _ { \lambda , m } ( \theta ) \leq \delta _ { \mathrm { l o c } } \leq 1$ , the fact that $\rho _ { \mathrm { l o c } } < 1$ yields

$$
\begin{array} { r } { [ \Delta _ { \lambda , m } ( \theta ) ] ^ { \rho _ { \mathrm { l o c } } } = \Delta _ { \lambda , m } ( \theta ) [ \Delta _ { \lambda , m } ( \theta ) ] ^ { \rho _ { \mathrm { l o c } } - 1 } \geq \Delta _ { \lambda , m } ( \theta ) . } \end{array}
$$

Hence (C.25) implies the weaker but suficient linear bound

$$
\mathrm { d i s t } ( 0 , \partial \Psi ( \theta ) ) \geq c _ { \mathrm { l o c } } \Delta _ { \lambda , m } ( \theta ) , \ 0 < \Delta _ { \lambda , m } ( \theta ) \leq \delta _ { \mathrm { l o c } } .\tag{C.26}
$$

We apply this bound to the exact projected step y(θ) defined in the lemma. The first-order

condition for Euclidean projection is

$$
\Big ( \theta - \alpha _ { \mathrm { s t u } } \nabla _ { \theta } F ( \theta ) - y ( \theta ) \Big ) ^ { \top } \big ( \vartheta - y ( \theta ) \big ) \leq 0 \forall \vartheta \in \Theta .
$$

Setting $\vartheta = \theta$ and rearranging gives

$$
\nabla _ { \theta } F ( \theta ) ^ { \top } ( y ( \theta ) - \theta ) \leq - \frac { \| y ( \theta ) - \theta \| _ { 2 } ^ { 2 } } { \alpha _ { \mathrm { s t u } } } .
$$

The gradient Lipschitz bound in (B.16), integrated along the segment from θ to $y ( \theta )$ , gives

$$
\begin{array} { r l } & { \displaystyle F ( \boldsymbol { y } ( \boldsymbol { \theta } ) ) - F ( \boldsymbol { \theta } ) = \nabla _ { \boldsymbol { \theta } } F ( \boldsymbol { \theta } ) ^ { \top } ( \boldsymbol { y } ( \boldsymbol { \theta } ) - \boldsymbol { \theta } ) + \int _ { 0 } ^ { 1 } \left[ \nabla _ { \boldsymbol { \theta } } F ( \boldsymbol { \theta } + \boldsymbol { u } ( \boldsymbol { y } ( \boldsymbol { \theta } ) - \boldsymbol { \theta } ) ) - \nabla _ { \boldsymbol { \theta } } F ( \boldsymbol { \theta } ) \right] ^ { \top } ( \boldsymbol { y } ( \boldsymbol { \theta } ) - \boldsymbol { \theta } ) d \boldsymbol { u } } \\ & { \qquad \le - \frac { \left\| \boldsymbol { y } ( \boldsymbol { \theta } ) - \boldsymbol { \theta } \right\| _ { 2 } ^ { 2 } } { \alpha _ { \mathrm { s t u } } } + \int _ { 0 } ^ { 1 } u L _ { \mathrm { s t } } \left\| \boldsymbol { y } ( \boldsymbol { \theta } ) - \boldsymbol { \theta } \right\| _ { 2 } ^ { 2 } d \boldsymbol { u } } \\ & { \qquad = - \left( \frac { 1 } { \alpha _ { \mathrm { s t u } } } - \frac { L _ { \mathrm { s t } } } { 2 } \right) \left\| \boldsymbol { y } ( \boldsymbol { \theta } ) - \boldsymbol { \theta } \right\| _ { 2 } ^ { 2 } \le - \frac { \left\| \boldsymbol { y } \left( \boldsymbol { \theta } \right) - \boldsymbol { \theta } \right\| _ { 2 } ^ { 2 } } { 2 \alpha _ { \mathrm { s t u } } } . } \end{array}
$$

The last inequality uses $\alpha _ { \mathrm { s t u } } = 1 / ( 2 L _ { \mathrm { s t } } )$ . In particular, $F ( y ( \theta ) ) \leq F ( \theta )$ . The projection condition also implies

$$
\frac { \theta - y ( \theta ) } { \alpha _ { \mathrm { s t u } } } - \nabla _ { \theta } F ( \theta ) \in N _ { \Theta } ( y ( \theta ) ) .
$$

Using (C.24) at $y ( \theta )$ , and then (B.16), we obtain

$$
\mathrm { d i s t } ( 0 , \partial \Psi ( y ( \theta ) ) ) \leq \left\| \nabla _ { \theta } F ( y ( \theta ) ) - \nabla _ { \theta } F ( \theta ) + \frac { \theta - y ( \theta ) } { \alpha _ { \mathrm { s t u } } } \right\| _ { 2 } \leq \left( L _ { \mathrm { s t } } + \frac { 1 } { \alpha _ { \mathrm { s t u } } } \right) \| y ( \theta ) - \theta \| _ { 2 } .
$$

Suppose now that $0 < \Delta _ { \lambda , m } ( \theta ) \leq \delta _ { \mathrm { l o c } } . \mathrm { ~ I f ~ } \Delta _ { \lambda , m } ( y ( \theta ) ) \leq \Delta _ { \lambda , m } ( \theta ) / 2$ , then (C.23) gives

$$
F ( \theta ) - F ( y ( \theta ) ) = \Delta _ { \lambda , m } ( \theta ) - \Delta _ { \lambda , m } ( y ( \theta ) ) \geq \frac { \Delta _ { \lambda , m } ( \theta ) } { 2 } \geq \frac { [ \Delta _ { \lambda , m } ( \theta ) ] ^ { 2 } } { 2 M _ { \mathrm { o p t } } } .
$$

Otherwise, $0 < \Delta _ { \lambda , m } ( \theta ) / 2 < \Delta _ { \lambda , m } ( y ( \theta ) ) \leq \delta _ { \mathrm { l o c } }$ , so (C.26) applies at $y ( \theta )$ . Combining the preceding two estimates with that inequality gives

$$
\begin{array} { r l } & { F ( \theta ) - F ( y ( \theta ) ) \geq \frac { | | y ( \theta ) - \theta | | _ { 2 } ^ { 2 } } { 2 \alpha _ { \mathrm { s t u } } } \geq \frac { \mathrm { d i s t } ( 0 , \partial \Psi ( y ( \theta ) ) ) ^ { 2 } } { 2 \alpha _ { \mathrm { s t u } } ( L _ { \mathrm { s t } } + 1 / \alpha _ { \mathrm { s t u } } ) ^ { 2 } } } \\ & { \qquad \geq \frac { c _ { \mathrm { l o c } } ^ { 2 } [ \Delta _ { \lambda , m } ( y ( \theta ) ) ] ^ { 2 } } { 2 \alpha _ { \mathrm { s t u } } ( L _ { \mathrm { s t } } + 1 / \alpha _ { \mathrm { s t u } } ) ^ { 2 } } \geq \frac { c _ { \mathrm { l o c } } ^ { 2 } } { 8 \alpha _ { \mathrm { s t u } } ( L _ { \mathrm { s t } } + 1 / \alpha _ { \mathrm { s t u } } ) ^ { 2 } } [ \Delta _ { \lambda , m } ( \theta ) ] ^ { 2 } . } \end{array}
$$

The same quadratic lower bound, with the smaller of the two coeficients, therefore holds throughout the small sublevel set. When the gap is zero, the required bound is immediate.

It remains to control $\Delta _ { \lambda , m } ( \theta ) > \delta _ { \mathrm { l o c } }$ . We establish the needed exploration probability for a calibrated objective. Fix $w \in W$ and $0 < u \leq M _ { \mathrm { o p t } }$ . By continuity and compactness, we can choose

$$
\vartheta _ { w } ^ { \star } \in \mathop { \mathrm { a r g m i n } } _ { \vartheta \in \Theta } C _ { w } \mathopen { } \mathclose \bgroup \left( \vartheta \aftergroup \egroup \right) .
$$

Set $\tau = u / M _ { \mathrm { o p t } } \in ( 0 , 1 ]$ and consider $\{ ( 1 - \tau ) \vartheta _ { w } ^ { \star } + \tau \zeta : \zeta \in \Theta \}$ . By convexity, we know that this

set lies in Θ. Each of its points ϑ has distance at most $2 B \tau$ from $\vartheta _ { w } ^ { \star } ,$ so (B.11) gives

$$
C _ { w } ( \vartheta ) - \underset { \zeta \in \Theta } { \operatorname* { m i n } } C _ { w } ( \zeta ) \leq ( 4 \lambda B H ^ { 3 / 2 } ) ( 2 B \tau ) = M _ { \mathrm { o p t } } \tau = u .
$$

The afine map defining the set scales d-dimensional volume by $\tau ^ { d }$ . Since $\vartheta ^ { \mathrm { u n i f } }$ is uniform on $\Theta ,$ we conclude that

$$
\operatorname* { P r } \biggl ( C _ { w } \bigl ( \vartheta ^ { \mathrm { u n i f } } \bigr ) - \operatorname* { m i n } _ { \vartheta \in \Theta } C _ { w } \bigl ( \vartheta \bigr ) \le u \biggr ) \ge \biggl ( \frac { u } { M _ { \mathrm { o p t } } } \biggr ) ^ { d } , \ 0 < u \le M _ { \mathrm { o p t } } .\tag{C.27}
$$

Apply this bound with $\boldsymbol { w } = \boldsymbol { w } _ { \lambda } ^ { \star }$ and $u = \delta _ { \mathrm { l o c } } / 2$ to obtain

$$
\mathrm { P r } \bigg ( \Delta _ { \lambda , m } \big ( \vartheta ^ { \mathrm { u n i f } } \big ) \leq \frac { \delta _ { \mathrm { l o c } } } { 2 } \bigg ) \geq \bigg ( \frac { \delta _ { \mathrm { l o c } } } { 2 M _ { \mathrm { o p t } } } \bigg ) ^ { d } .
$$

On this event and when $\Delta _ { \lambda , m } ( \theta ) > \delta _ { \mathrm { l o c } } .$ , we have $F ( \theta ) - F ( \vartheta ^ { \mathrm { u n i f } } ) \geq \Delta _ { \lambda , m } ( \theta ) - \delta _ { \mathrm { l o c } } / 2 \geq \Delta _ { \lambda , m } ( \theta ) / 2$ Since $F ( y ( \theta ) ) \leq F ( \theta )$ , the two-candidate gain is nonnegative and at least $F ( \theta ) - F ( \vartheta ^ { \mathrm { u n i f } } )$ for every draw. It therefore dominates the positive part below:

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol { \vartheta } ^ { \operatorname* { m i f } } } \Big [ F ( \boldsymbol { \theta } ) - \operatorname* { m i n } \{ F ( \boldsymbol { y } ( \boldsymbol { \theta } ) ) , F ( \boldsymbol { \vartheta } ^ { \operatorname* { m i f } } ) \} \Big ] \geq \mathbb { E } _ { \boldsymbol { \vartheta } ^ { \operatorname* { m i f } } } \Big [ [ F ( \boldsymbol { \theta } ) - F ( \boldsymbol { \vartheta } ^ { \operatorname* { m i f } } ) ] _ { + } \Big ] } \\ & { \qquad \geq \frac { 1 } { 2 } \left( \frac { \delta _ { \mathrm { l o c } } } { 2 M _ { \mathrm { o p t } } } \right) ^ { d } \Delta _ { \lambda , m } ( \boldsymbol { \theta } ) \geq \frac { 1 } { 2 M _ { \mathrm { o p t } } } \left( \frac { \delta _ { \mathrm { l o c } } } { 2 M _ { \mathrm { o p t } } } \right) ^ { d } [ \Delta _ { \lambda , m } ( \boldsymbol { \theta } ) ] ^ { 2 } . } \end{array}
$$

The last inequality uses (C.23). For the small sublevel set, the minimum of the two costs is at most $F ( y ( \theta ) )$ , so the projected-step bounds already proved apply directly to the same left-hand side. Taking

$$
\kappa _ { \mathrm { o p t } } : = \operatorname* { m i n } \left\{ \frac { 1 } { 4 M _ { \mathrm { o p t } } } , \frac { c _ { \mathrm { l o c } } ^ { 2 } } { 8 \alpha _ { \mathrm { s t u } } ( L _ { \mathrm { s t } } + 1 / \alpha _ { \mathrm { s t u } } ) ^ { 2 } } , \frac { 1 } { 2 M _ { \mathrm { o p t } } } \left( \frac { \delta _ { \mathrm { l o c } } } { 2 M _ { \mathrm { o p t } } } \right) ^ { d } \right\} > 0
$$

therefore proves (B.19) for every $\theta \in \Theta$ . All constants were chosen using the fixed objective and parameter set, and hence are independent of t. □

Proof of Lemma B.10. We study the fixed true cost $F ( \theta ) : = C _ { w _ { \lambda } ^ { \star } } ( \theta )$ , so that $\Delta _ { \lambda , m } ( \theta ) = F ( \theta ) -$ $F ( \theta _ { \lambda , m } ^ { \dagger } )$ . The algorithm evaluates $C _ { t }$ , whereas F is used only in this proof. Recall from (C.23) that $0 \leq \Delta _ { \lambda , m } ( \theta ) \leq M _ { \mathrm { o p t } }$ . For target update $t \geq 1$ , let $\mathcal { G } _ { t } : = \mathcal { F } _ { t - 1 } \vee \sigma ( w _ { t } )$ . The current student $\theta _ { t - 1 }$ is fixed conditional on $\mathcal { G } _ { t }$ . Compare the actual gradient candidate with the population candidate

$$
\begin{array} { c } { { \vartheta _ { t , 1 } = \mathrm { P r o j } _ { \Theta } \left( \theta _ { t - 1 } - \alpha _ { \mathrm { s t u } } \widehat { g } _ { t } ^ { \mathrm { s t u } } \right) , } } \\ { { y ( \theta _ { t - 1 } ) = \mathrm { P r o j } _ { \Theta } \left( \theta _ { t - 1 } - \alpha _ { \mathrm { s t u } } \nabla _ { \theta } F ( \theta _ { t - 1 } ) \right) . } } \end{array}
$$

The map y is the proof-only update in Lemma B.9.

We first write the one-step error comparison, before bounding its perturbation terms. Lemma B.8 and (B.12) imply, on every sample path,

$$
\begin{array} { r l } & { F ( \theta _ { t } ) \le C _ { t } ( \theta _ { t } ) + \displaystyle \operatorname* { s u p } _ { \theta \in \Theta } | F ( \theta ) - C _ { t } ( \theta ) | \le \displaystyle \operatorname* { m i n } _ { k \in \{ 1 , 2 \} } C _ { t } ( \vartheta _ { t , k } ) + \xi _ { t } + \displaystyle \operatorname* { s u p } _ { \theta \in \Theta } | F ( \theta ) - C _ { t } ( \theta ) | } \\ & { \qquad \le \displaystyle \operatorname* { m i n } _ { k \in \{ 1 , 2 \} } F ( \vartheta _ { t , k } ) + \xi _ { t } + 2 \displaystyle \operatorname* { s u p } _ { \theta \in \Theta } | F ( \theta ) - C _ { t } ( \theta ) | \le \operatorname* { m i n } \{ F ( \vartheta _ { t , 1 } ) , F ( \vartheta _ { t , 2 } ) \} + \xi _ { t } + 4 \lambda H \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } . } \end{array}
$$

Replacing one entry of a minimum changes that minimum by at most the absolute change in the entry. Since $F$ is $4 \lambda B H ^ { 3 / 2 }$ -Lipschitz according to Lemma B.5, we thus have

$$
\begin{array} { r } { \operatorname* { m i n } \{ F ( \vartheta _ { t , 1 } ) , F ( \vartheta _ { t , 2 } ) \} \leq \operatorname* { m i n } \{ F ( y ( \theta _ { t - 1 } ) ) , F ( \vartheta _ { t , 2 } ) \} + 4 \lambda B H ^ { 3 / 2 } \| \vartheta _ { t , 1 } - y ( \theta _ { t - 1 } ) \| _ { 2 } . } \end{array}
$$

Conditional on $\mathcal { G } _ { t }$ , the exploration candidate $\vartheta _ { t , 2 }$ is uniform on Θ. By Lemma B.9, we have

$$
\begin{array} { r } { \mathbb { E } \big [ F \big ( \theta _ { t - 1 } \big ) - \operatorname* { m i n } \{ F \big ( y \big ( \theta _ { t - 1 } \big ) \big ) , F \big ( \vartheta _ { t , 2 } \big ) \} | \mathcal { G } _ { t } \big ] \ge \kappa _ { \mathrm { o p t } } \big [ \Delta _ { \lambda , m } \big ( \theta _ { t - 1 } \big ) \big ] ^ { 2 } . } \end{array}
$$

Since $F ( \theta _ { t - 1 } )$ is measurable with respect to $\mathcal { G } _ { t }$ , rearrange and we have

$$
\mathbb { E } [ \operatorname* { m i n } \{ F ( y ( \theta _ { t - 1 } ) ) , F ( \vartheta _ { t , 2 } ) \} | \mathcal { G } _ { t } ] \leq F ( \theta _ { t - 1 } ) - \kappa _ { \mathrm { o p t } } [ \Delta _ { \lambda , m } ( \theta _ { t - 1 } ) ] ^ { 2 }
$$

Thus, utilizing the inequalities we have proved above, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \Delta _ { \lambda , m } ( \theta _ { t } ) \mid \mathcal { G } _ { t } ] } \\ & { = \mathbb { E } [ F ( \theta _ { t } ) \mid \mathcal { G } _ { t } ] - F ( \theta _ { \lambda , m } ^ { \dagger } ) } \\ & { \le \mathbb { E } [ \operatorname* { m i n } \{ F ( y ( \theta _ { t - 1 } ) ) , F ( \vartheta _ { t , 2 } ) \} \mid \mathcal { G } _ { t } ] - F ( \theta _ { \lambda , m } ^ { \dagger } ) + 4 \lambda B H ^ { 3 / 2 } \mathbb { E } [ \left\| \vartheta _ { t , 1 } - y ( \theta _ { t - 1 } ) \right\| _ { 2 } \vert \mathcal { G } _ { t } ] + 4 \lambda H \| w _ { t } - w _ { \lambda } ^ { * } \| _ { 2 } + \mathbb { E } [ \xi _ { t } \vert \mathcal { G } _ { t } ] } \\ & { \le F ( \theta _ { t - 1 } ) - F ( \theta _ { \lambda , m } ^ { \dagger } ) - \kappa _ { \mathrm { o p t } } [ \Delta _ { \lambda , m } ( \theta _ { t - 1 } ) ] ^ { 2 } + 4 \lambda B H ^ { 3 / 2 } \mathbb { E } [ \left\| \vartheta _ { t , 1 } - y ( \theta _ { t - 1 } ) \right\| _ { 2 } \vert \mathcal { G } _ { t } ] + 4 \lambda H \| w _ { t } - w _ { \lambda } ^ { * } \| _ { 2 } + \mathbb { E } [ \xi _ { t } \vert \mathcal { G } _ { t } ] . } \end{array}\tag{C.28}
$$

We next bound the candidate discrepancy in this recursion. Nonexpansiveness of Euclidean projection gives

$$
\begin{array} { r } { \| \vartheta _ { t , 1 } - y ( \theta _ { t - 1 } ) \| _ { 2 } \leq \alpha _ { \mathrm { s t u } } \| \widehat { g } _ { t } ^ { \mathrm { s t u } } - \nabla _ { \theta } F ( \theta _ { t - 1 } ) \| _ { 2 } \leq \alpha _ { \mathrm { s t u } } \left[ \| \widehat { g } _ { t } ^ { \mathrm { s t u } } - \nabla _ { \theta } C _ { t } ( \theta _ { t - 1 } ) \| _ { 2 } + \| \nabla _ { \theta } C _ { t } ( \theta _ { t - 1 } ) - \nabla _ { \theta } F ( \theta _ { t - 1 } ) \| _ { 2 } \right] . } \end{array}
$$

Conditional Cauchy-Schwarz and Lemma B.6 bound the first term by

$$
\mathbb { E } \left[ \| \widehat { g } _ { t } ^ { \mathrm { s t u } } - \nabla _ { \theta } C _ { t } ( \theta _ { t - 1 } ) \| _ { 2 } \Big | \mathcal { G } _ { t } \right] \leq \left( \mathbb { E } \left[ \| \widehat { g } _ { t } ^ { \mathrm { s t u } } - \nabla _ { \theta } C _ { t } ( \theta _ { t - 1 } ) \| _ { 2 } ^ { 2 } \Big | \mathcal { G } _ { t } \right] \right) ^ { 1 / 2 } \leq \frac { 4 \lambda B H ^ { 3 / 2 } } { \sqrt { b _ { t } } } .
$$

Lemma B.7 bounds the second term by $2 \lambda H ^ { 3 / 2 } \lVert w _ { t } - w _ { \lambda } ^ { \star } \rVert _ { 2 }$ . Multiplying by the cost Lipschitz constant therefore gives

$$
4 \lambda B H ^ { 3 / 2 } \mathbb { E } [ \| \vartheta _ { t , 1 } - y ( \theta _ { t - 1 } ) \| _ { 2 } \mid \mathcal { G } _ { t } ] \leq \frac { 1 6 \alpha _ { \mathrm { s t u } } \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } } { \sqrt { b _ { t } } } + 8 \alpha _ { \mathrm { s t u } } \lambda ^ { 2 } B H ^ { 3 } \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } .
$$

For validation, $\mathcal { G } _ { t } \subseteq \mathcal { H } _ { t }$ in Lemma B.8. The tower property yields

$$
\mathbb { E } [ \xi _ { t } \mid \mathcal { G } _ { t } ] = \mathbb { E } [ \mathbb { E } [ \xi _ { t } \mid \mathcal { H } _ { t } ] \mid \mathcal { G } _ { t } ] \leq \frac { 1 6 \lambda B H } { t + 1 } .
$$

To combine these bounds, write $e _ { t } : = \mathbb { E } [ \Delta _ { \lambda , m } ( \theta _ { t } ) ]$ . Taking total expectations in (C.28), using $b _ { t } = t + 1$ , and substituting the two perturbation bounds gives

$$
e _ { t } \leq \epsilon _ { t - 1 } - \kappa _ { \mathrm { o p t } } \mathbb { E } \left[ [ \Delta _ { \lambda , m } ( \theta _ { t - 1 } ) ] ^ { 2 } \right] + \frac { 1 6 \alpha _ { \mathrm { s t n } } \lambda ^ { 2 } B ^ { 2 } H ^ { 3 } } { \sqrt { t + 1 } } + ( 8 \alpha _ { \mathrm { s t n } } \lambda ^ { 2 } B H ^ { 3 } + 4 \lambda H ) \mathbb { E } [ \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ] + \frac { 1 6 \lambda B H } { t + 1 } .
$$

Applying the calibration bound (B.7) and the Cauchy-Schwarz inequality, we have

$$
\mathbb { E } [ \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ] \le \left( \mathbb { E } [ \| w _ { t } - w _ { \lambda } ^ { \star } \| _ { 2 } ^ { 2 } ] \right) ^ { 1 / 2 } \le \frac { 2 } { \gamma \sqrt { t + 2 } } \le \frac { 2 } { \gamma \sqrt { t + 1 } } .
$$

Also, $\mathbb { E } [ [ \Delta _ { \lambda , m } ( \theta _ { t - 1 } ) ] ^ { 2 } ] \geq e _ { t - 1 } ^ { 2 }$ by Jensen’s inequality, and $( t + 1 ) ^ { - 1 } \leq ( t + 1 ) ^ { - 1 / 2 }$ . Consequently,

$$
e _ { t } \le e _ { t - 1 } - \kappa _ { \mathrm { o p t } } e _ { t - 1 } ^ { 2 } + \frac { E _ { \mathrm { o p t } } } { \sqrt { t + 1 } } , \ t \ge 1 .\tag{C.29}
$$

We solve (C.29) by induction with the deterministic bound $v _ { t } : = K _ { \mathrm { o p t } } ( t + 1 ) ^ { - 1 / 4 }$ . Initially, we have $e _ { 0 } \le M _ { \mathrm { o p t } } \le K _ { \mathrm { o p t } } = v _ { 0 }$

Suppose $e _ { t } \leq v _ { t } . \ \mathrm { I f } \ v _ { t + 1 } \geq M _ { \mathrm { o p t } }$ , the bound $e _ { t + 1 } \leq M _ { \mathrm { o p t } }$ proves the next step. Otherwise, we have

$$
v _ { t } = v _ { t + 1 } \left( { \frac { t + 2 } { t + 1 } } \right) ^ { 1 / 4 } < 2 ^ { 1 / 4 } M _ { \mathrm { o p t } } < 2 M _ { \mathrm { o p t } } .
$$

Since $\kappa _ { \mathrm { o p t } } \leq 1 / ( 4 M _ { \mathrm { o p t } } )$ , the derivative of $u - \kappa _ { \mathrm { o p t } } u ^ { 2 }$ on $[ 0 , 2 M _ { \mathrm { o p t } } ]$ is $1 - 2 \kappa _ { \mathrm { o p t } } u \geq 0$ . Apply this monotonicity to the recursion at time $t + 1$ and we have

$$
e _ { t + 1 } \leq v _ { t } - \kappa _ { \mathrm { o p t } } v _ { t } ^ { 2 } + \frac { E _ { \mathrm { o p t } } } { \sqrt { t + 2 } } \leq v _ { t } - \frac { \kappa _ { \mathrm { o p t } } K _ { \mathrm { o p t } } ^ { 2 } - E _ { \mathrm { o p t } } } { \sqrt { t + 1 } } .
$$

By the definition of $K _ { \mathrm { o p t } }$ , we have $\begin{array} { r } { \kappa _ { \mathrm { o p t } } K _ { \mathrm { o p t } } ^ { 2 } - E _ { \mathrm { o p t } } \geq \frac { \kappa _ { \mathrm { o p t } } K _ { \mathrm { o p t } } ^ { 2 } } { 2 } \geq \frac { K _ { \mathrm { o p t } } } { 4 } } \end{array}$ . The first inequality uses $K _ { \mathrm { o p t } } ^ { 2 } \geq 2 E _ { \mathrm { o p t } } / \kappa _ { \mathrm { o p t } } $ the second uses $K _ { \mathrm { o p t } } ~ \ge ~ 1 / ( 2 \kappa _ { \mathrm { o p t } } )$ . On the other hand, integration of the derivative of $u ^ { - 1 / 4 }$ gives

$$
v _ { t } - v _ { t + 1 } = { \frac { K _ { \mathrm { o p t } } } { 4 } } \int _ { t + 1 } ^ { t + 2 } u ^ { - 5 / 4 } d u \leq { \frac { K _ { \mathrm { o p t } } } { 4 ( t + 1 ) ^ { 5 / 4 } } } \leq { \frac { K _ { \mathrm { o p t } } } { 4 { \sqrt { t + 1 } } } } .
$$

Combining the last three displays yields $e _ { t + 1 } \leq v _ { t + 1 }$ . By induction, we prove the first bound in (B.20) for every $T \geq 1$

Finally, we translate the true-cost bound to the calibrated optimization error. By the elementary inequality | min $f - \operatorname* { m i n } g | \leq \operatorname* { s u p } | f - g |$ , applied on Θ, we have

$$
\begin{array} { r l } & { | \varepsilon _ { T } - \Delta _ { \lambda , m } ( \theta _ { T } ) | = \Big | C _ { T } ( \theta _ { T } ) - F ( \theta _ { T } ) + \underset { \theta \in \Theta } { \operatorname* { m i n } } F ( \theta ) - \underset { \theta \in \Theta } { \operatorname* { m i n } } C _ { T } ( \theta ) \Big | } \\ & { \quad \quad \leq 2 \underset { \theta \in \Theta } { \operatorname* { s u p } } | C _ { T } ( \theta ) - F ( \theta ) | \leq 4 \lambda H \| w _ { T } - w _ { \lambda } ^ { \star } \| _ { 2 } . } \end{array}\tag{C.30}
$$

Taking expectations and applying (B.7) proves the second bound in (B.20). We finish the proof.

Proof of Lemma B.11. We will apply the nonnegative form of Lemma A.1 with

$$
K = \Theta , f ( \theta ) = \Delta _ { \lambda , m } ( \theta ) , g ( \theta ) = \mathcal { K } _ { \lambda , m } ( \theta ) .
$$

Accordingly, we verify that the two functions are nonnegative, that their graphs are subanalytic and compact, and that $\Delta _ { \lambda , m } ( \theta ) = 0$ implies $\mathcal { K } _ { \lambda , m } ( \theta ) = 0$ . We also establish an upper bound on $\kappa _ { \lambda , m }$ to use in the theorem’s exponent normalization. Their definitions are recalled in the lemma statement.

Nonnegativity. By the optimality of $\theta _ { \lambda , m } ^ { \dagger }$ and the nonnegativity of KL divergence, respectively, we have

$$
\Delta _ { \lambda , m } ( \theta ) \geq 0 , \ : \mathcal { K } _ { \lambda , m } ( \theta ) \geq 0 , \ : \theta \in \Theta .
$$

Analyticity. Lemma B.7 proves analyticity for the average KL against any fixed positive answer laws. Apply that result with $Q _ { j } = \pi _ { w _ { \lambda } ^ { \star } } ( \cdot \mid \widetilde { x } _ { j } )$ and with $Q _ { j } = \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \mid \widetilde { x } _ { j } )$ , respectively. Thus $C _ { w _ { \lambda } ^ { \star } }$ and $\kappa _ { \lambda , m }$ are analytic. Subtracting the constant $C _ { w _ { \lambda } ^ { \star } } ( \theta _ { \lambda , m } ^ { \dagger } )$ shows that $\Delta _ { \lambda , m }$ is analytic. Compact subanalytic graphs. For $F \in \{ \Delta _ { \lambda , m } , K _ { \lambda , m } \}$ , the restricted graph is

$$
\mathrm { g r a p h } ( F | _ { \Theta } ) = \{ ( \theta , u ) \in \mathbb { R } ^ { d + 1 } : B ^ { 2 } - \| \theta \| _ { 2 } ^ { 2 } \geq 0 , \ u - F ( \theta ) = 0 \} .
$$

The defining inequality is polynomial and the equality is analytic by Lemma B.7. Hence the graph is semianalytic, and therefore subanalytic. It is also compact because it is the image of the compact ball Θ under the continuous map $\theta \mapsto ( \theta , F ( \theta ) )$ .

Zero-set inclusion. If $\Delta _ { \lambda , m } ( \theta ) = 0$ , its definition gives

$$
C _ { w _ { \lambda } ^ { \star } } ( \theta ) = C _ { w _ { \lambda } ^ { \star } } ( \theta _ { \lambda , m } ^ { \dagger } ) = \operatorname* { m i n } _ { \vartheta \in \Theta } C _ { w _ { \lambda } ^ { \star } } ( \vartheta ) .
$$

Thus θ is also a global minimizer. By Assumption 3.3, we have

$$
\pi _ { \mathrm { s t u } , \boldsymbol { \theta } } ( \cdot  { | }  { \widetilde { x } } _ { j } ) = \pi _ { \mathrm { s t u } , \boldsymbol { \theta } _ { \lambda , m } ^ { \dagger } } ( \cdot  { | }  { \widetilde { x } } _ { j } ) , \ j = 1 , \dots , m .
$$

Each KL summand defining $\kappa _ { \lambda , m } ( \theta )$ is consequently the divergence of a law from itself, hence zero. We obtain the required inclusion

$$
\{ \theta \in \Theta : \Delta _ { \lambda , m } ( \theta ) = 0 \} \subseteq \{ \theta \in \Theta : \mathcal { K } _ { \lambda , m } ( \theta ) = 0 \} .
$$

A uniform bound for the KL divergence Apply the student probability envelope (C.14) from Lemma B.4 to θ and $\theta _ { \lambda , m } ^ { \dagger }$ . Dividing the lower bound for the numerator by the upper bound for the denominator, and conversely, gives

$$
e ^ { - 4 B } \leq \frac { \pi _ { \mathrm { s t u } , \theta } ( a \mid s ) } { \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( a \mid s ) } \leq e ^ { 4 B } ( a \in B ( s ) ) .
$$

Taking logarithms gives an absolute token log-ratio bound of 4B. For $a _ { 1 : H } \in \mathcal { A } ( \widetilde { x } _ { j } )$ , put $s _ { j , h } =$ $\left( \widetilde { x } _ { j } , a _ { 1 : h - 1 } \right)$ . By (3.1) and the triangle inequality, we have

$$
\left| \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } { \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( a _ { 1 : H } \mid \widetilde { x } _ { j } ) } \right| = \left| \sum _ { h = 1 } ^ { H } \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { j , h } ) } { \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( a _ { h } \mid s _ { j , h } ) } \right| \le \sum _ { h = 1 } ^ { H } \left| \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { h } \mid s _ { j , h } ) } { \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( a _ { h } \mid s _ { j , h } ) } \right| \le 4 B H .
$$

Using the expectation form of KL in the definition of $\kappa _ { \lambda , m }$ , this pointwise bound gives

$$
0 \leq { \mathcal K } _ { \lambda , m } ( \theta ) \leq \frac { 1 } { m } \sum _ { j = 1 } ^ { m } { \mathbb E } _ { a _ { 1 : H } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde x _ { j } ) } [ 4 B H ] = 4 B H ,
$$

Via the steps above, we verify the nonnegativity, compact subanalytic graphs, and zero-set inclusion required by Lemma A.1. Moreover, the last step provides the bound $K _ { \lambda , m } ( \theta ) \leq 4 B H \leq M _ { \lambda , m }$ with

M<sub>λ,m</sub> := max{1, 4BH}.

Since the functions and their domain are fixed independently of the iteration number, applying Lemma A.1 and we get that there exist $a _ { \lambda , m } > 0$ and $p _ { \lambda , m } \ge 1$ such that

$$
\Delta _ { \lambda , m } ( \theta ) \geq a _ { \lambda , m } [ \mathcal { K } _ { \lambda , m } ( \theta ) ] ^ { p _ { \lambda , m } } \ \theta \in \Theta ,
$$

We finish the proof.

## D Proofs in Section 6

Proof of Proposition 6.1. We first compute the true student objective. For either prompt in pair $( \widetilde { x } _ { i , + } , \widetilde { x } _ { i , - } )$ , we have

$$
\pi _ { \operatorname { s t u } , \theta } ( a \mid \widetilde { x } _ { i , \epsilon } , \mathcal { O } ) = \frac { \exp \bigl ( \theta _ { i } \mathbf { 1 } \{ a = 1 \} \bigr ) } { e ^ { \theta _ { i } } + 2 } , \ a \in \{ 0 , 1 , \operatorname { n u 1 1 } \} .
$$

We define $\begin{array} { r } { p ( \theta _ { i } ) : = \frac { e ^ { \theta _ { i } } } { e ^ { \theta _ { i } } + 2 } = \sigma ( \theta _ { i } - \log 2 ) , \sigma ( u ) = ( 1 + e ^ { - u } ) ^ { - 1 } } \end{array}$ . Then, the probabilities of the student policy to output 0 and null are each $\left[ 1 - p ( \theta _ { i } ) \right] / 2$ . Therefore, we have

$$
\begin{array} { l } { \displaystyle \frac { 1 } { 2 } \sum _ { \epsilon \in \{ + , - \} } \mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { i , \epsilon } ) } [ R ( \widetilde { x } _ { i , \epsilon } , a _ { 1 : 2 } ) ] = \displaystyle \frac { 1 } { 2 } \left[ \pi _ { \mathrm { s t u } , \theta } ( 1 \mid \widetilde { x } _ { i , + } , \mathcal { Q } ) + \pi _ { \mathrm { s t u } , \theta } ( 0 \mid \widetilde { x } _ { i , - } , \mathcal { Q } ) \right] } \\ { \displaystyle \qquad = \displaystyle \frac { 1 } { 2 } \left[ p ( \theta _ { i } ) + \frac { 1 - p ( \theta _ { i } ) } { 2 } \right] = \frac { 1 + p ( \theta _ { i } ) } { 4 } . } \end{array}
$$

Both the student and reference emit EOS with probability one at $h = 2$ , therefore, for $\epsilon \in \{ + , - \}$ ， at any $\widetilde { x } _ { i , \epsilon } .$ we can explicitly compute the KL divergence as

$$
\begin{array} { l } { \displaystyle \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { i , \epsilon } ) | | \pi _ { \mathrm { p r e } } ( \cdot | \widetilde { x } _ { i , \epsilon } ) ) = p ( \theta _ { i } ) \log \frac { p ( \theta _ { i } ) } { 1 / 3 } + 2 \frac { 1 - p ( \theta _ { i } ) } { 2 } \log \frac { [ 1 - p ( \theta _ { i } ) ] / 2 } { 1 / 3 } } \\ { = p ( \theta _ { i } ) \log \frac { p ( \theta _ { i } ) } { 1 / 3 } + [ 1 - p ( \theta _ { i } ) ] \log \frac { 1 - p ( \theta _ { i } ) } { 2 / 3 } } \\ { = \theta _ { i } p ( \theta _ { i } ) - \log \frac { e ^ { \theta _ { i } } + 2 } { 3 } . } \end{array}\tag{D.1}
$$

Recall that $p ^ { \prime } ( \theta _ { i } ) = p ( \theta _ { i } ) [ 1 - p ( \theta _ { i } ) ]$ . Hence, for each $i \in [ d ]$ and $\epsilon \in \{ + , - \}$ , we have

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathrm { K L } \big ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid \widetilde { x } _ { i , \epsilon } ) \mid \mid \pi _ { \mathrm { p r e } } ( \cdot \mid \widetilde { x } _ { i , \epsilon } ) \big ) = \nabla _ { \theta } \Bigg [ \theta _ { i } p ( \theta _ { i } ) - \log \frac { e ^ { \theta _ { i } } + 2 } { 3 } \Bigg ] } \\ & { \qquad = \Bigg [ p ( \theta _ { i } ) + \theta _ { i } p ^ { \prime } ( \theta _ { i } ) - \frac { e ^ { \theta _ { i } } } { e ^ { \theta _ { i } } + 2 } \Bigg ] e _ { i } } \\ & { \qquad = \theta _ { i } p ^ { \prime } ( \theta _ { i } ) e _ { i } . } \end{array}
$$

Here $e _ { i }$ appears because the expression depends on θ only through its ith coordinate.

Thus, using the fact that the KL regularization terms are equal at $\widetilde { x } _ { i , + }$ and $\widetilde { x } _ { i , - }$ , we obtain

$$
J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta } ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left[ \frac { 1 + p ( \theta _ { i } ) } { 4 } - \lambda \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta } ( \cdot  { | } \widetilde { x } _ { i , + } )  { | | } \pi _ { \mathrm { p r e } } ( \cdot  { | } \widetilde { x } _ { i , + } ) ) \right] .
$$

Diferentiating this finite sum and substituting the preceding KL-gradient identity gives

$$
\begin{array} { l } { \displaystyle \nabla _ { \theta } J _ { \lambda , m } \big ( \pi _ { \mathrm { s t u } , \theta } \big ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left[ \frac { p ^ { \prime } ( \theta _ { i } ) } { 4 } e _ { i } - \lambda \nabla _ { \theta } \mathrm { K L } \big ( \pi _ { \mathrm { s t u } , \theta } \big ( \cdot \big | \ \widetilde { x } _ { i , + } \big ) \big \| \pi _ { \mathrm { p r e } } \big ( \cdot \big | \ \widetilde { x } _ { i , + } \big ) \big ) \right] } \\ { \displaystyle \qquad = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } p ^ { \prime } ( \theta _ { i } ) \left( \frac { 1 } { 4 } - \lambda \theta _ { i } \right) e _ { i } } \\ { \displaystyle \qquad = \frac { 1 } { d } \left( \begin{array} { c } { p _ { 1 } ( \theta _ { 1 } ) \big [ 1 - p _ { 1 } ( \theta _ { 1 } ) \big ] \left( \frac { 1 } { 4 } - \lambda \theta _ { 1 } \right) } \\ { \vdots } \\ { p _ { d } ( \theta _ { d } ) \big [ 1 - p _ { d } ( \theta _ { d } ) \big ] \left( \frac { 1 } { 4 } - \lambda \theta _ { d } \right) } \end{array} \right) . } \end{array}
$$

Each summand strictly increases up to $1 / ( 4 \lambda )$ and strictly decreases afterward. The vector of these maximizers is feasible and interior, since $\sqrt { d } / ( 4 \lambda ) < B$ . Then, we apply the first order optimality condition to prove that $\begin{array} { r } { \theta _ { \lambda , m } ^ { \dagger } = \frac { 1 } { 4 \lambda } \mathbf { 1 } _ { d } \in \mathrm { i n t } ( \Theta ) } \end{array}$

We next identify the unrestricted optimal policy and verify that it belongs to the teacher policy class but not to the student policy class. Specifically, by Lemma B.1, we have that

$$
\pi _ { \lambda } ^ { \star } ( ( a _ { 1 } , { \tt E O S } ) \mid x ) = \frac { ( 1 / 3 ) e ^ { { \bf 1 } \{ a _ { 1 } = y ( x ) \} / { \lambda } } } { ( e ^ { { \bf 1 } / \lambda } + 2 ) / 3 } = \frac { e ^ { { \bf 1 } \{ a _ { 1 } = y ( x ) \} / { \lambda } } } { e ^ { { \bf 1 } / \lambda } + 2 } = \pi _ { ( \sqrt { 2 } / \lambda ) u _ { 1 } } ( ( a _ { 1 } , { \tt E O S } ) \mid x ) .
$$

For $w _ { \lambda } ^ { \star } = ( \sqrt { 2 } / \lambda ) u _ { 1 }$ , by the feature definition in teacher class, we have

$$
( w _ { \lambda } ^ { \star } ) ^ { \top } \phi ( ( x , \mathcal { D } ) , a _ { 1 } ) = \left\{ 1 / \lambda , \begin{array} { l l } { a _ { 1 } = y ( x ) , } \\ { 0 , } \end{array} \right.
$$

The resulting softmax probabilities are exactly those in the preceding expression. Since $\| w _ { \lambda } ^ { \star } \| _ { 2 } =$ ${ \sqrt { 2 } } / \lambda < B$ , this proves that $w _ { \lambda } ^ { \star } \in W$ and $\pi _ { \lambda } ^ { \star } = \pi _ { w _ { \lambda } ^ { \star } }$

To prove that no student represents this policy, we compare the probability of the answer (1, EOS) at the two prompts $\widetilde { x } _ { i , + }$ and ${ \widetilde { x } } _ { i , \astrosun }$ <sub>−</sub>. The unrestricted optimum satisfies

$$
\pi _ { \lambda } ^ { \star } ( ( 1 , { \tt E O S } ) \mid \widetilde { x } _ { i , + } ) = \frac { e ^ { 1 / \lambda } } { e ^ { 1 / \lambda } + 2 } > \frac { 1 } { e ^ { 1 / \lambda } + 2 } = \pi _ { \lambda } ^ { \star } ( ( 1 , { \tt E O S } ) \mid \widetilde { x } _ { i , - } ) .
$$

In contrast, every student satisfies

$$
\pi _ { \mathrm { s t u } , \theta } ( ( 1 , \mathtt { E O S } ) \mid \pi _ { i , + } ) = p ( \theta _ { i } ) = \pi _ { \mathrm { s t u } , \theta } ( ( 1 , \mathtt { E O S } ) \mid \tilde { x } _ { i , - } ) .
$$

Thus no $\theta \in \Theta$ can match the unrestricted optimum at both prompts.

It remains to compare the teacher’s regularized return with that of every student. We first obtain a student upper bound using the oracle parameter already proved optimal.

Substituting $( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } = 1 / ( 4 \lambda )$ into the objective and the KL expression in (D.1) gives

$$
\begin{array} { l }  { \displaystyle { J _ { \lambda , m } \big ( \pi _ { \mathrm { s t u } , \theta } \big ) \leq J _ { \lambda , m } \big ( \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } \big ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left[ \frac { 1 + p \big ( 1 / ( 4 \lambda ) \big ) } { 4 } - \lambda \left( \frac { p \big ( 1 / \left( 4 \lambda \right) \big ) } { 4 \lambda } - \log \frac { e ^ { 1 / ( 4 \lambda ) } + 2 } { 3 } \right) \right] } } \\ { { \displaystyle ~ = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left[ \frac { 1 } { 4 } + \lambda \log \frac { e ^ { 1 / ( 4 \lambda ) } + 2 } { 3 } \right] = \frac { 1 } { 4 } + \lambda \log \frac { e ^ { 1 / ( 4 \lambda ) } + 2 } { 3 } . } } \end{array}
$$

We now compute the teacher’s regularized return. Since $\begin{array} { r } { w _ { \mathrm { t e a } } = \alpha w _ { \lambda } ^ { \star } } \end{array}$ , its answer probabilities are

$$
\pi _ { \mathrm { { t e a } } } { \big ( } { \big ( } a _ { 1 } , \mathtt { E O S } { \big ) } \mid x { \big ) } = { \frac { \exp ( \alpha \mathbf { 1 } \{ a _ { 1 } = y ( x ) \} / \lambda ) } { e ^ { \alpha / \lambda } + 2 } } .
$$

In particular, its expected reward at every target prompt is

$$
\mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { t e a } } ( \cdot | x ) } [ R ( x , a _ { 1 : 2 } ) ] = \pi _ { \mathrm { t e a } } ( ( y ( x ) , \mathrm { E 0 S } ) \mid x ) = \frac { e ^ { \alpha / \lambda } } { e ^ { \alpha / \lambda } + 2 } .
$$

Because $\alpha < 1$ , this correct-verdict probability is strictly smaller than that of $\pi _ { \lambda } ^ { \star }$ . Hence the teacher difers from the unrestricted optimal policy.

The reference assigns probability $1 / 3$ to each feasible answer. Consequently, we know that

$$
\log { \frac { \pi _ { \mathrm { t e a } } ( a _ { 1 : 2 } \mid x ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : 2 } \mid x ) } } = \log \left[ { \frac { \exp ( \alpha R ( x , a _ { 1 : 2 } ) / \lambda ) } { e ^ { \alpha / \lambda } + 2 } } \cdot 3 \right] = { \frac { \alpha } { \lambda } } R ( x , a _ { 1 : 2 } ) - \log { \frac { e ^ { \alpha / \lambda } + 2 } { 3 } } .\tag{D.2}
$$

Substituting this identity into the regularized objective yields

$$
\begin{array} { l } { { J _ { \lambda , m } ( \pi _ { \mathrm { t e a } } ) = \displaystyle \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { t e a } } ( \cdot | \widetilde { x } _ { j } ) } \left[ ( 1 - \alpha ) R ( \widetilde { x } _ { j } , a _ { 1 : 2 } ) + \lambda \log \frac { e ^ { \alpha / \lambda } + 2 } { 3 } \right] } } \\ { { \displaystyle \qquad = ( 1 - \alpha ) \frac { e ^ { \alpha / \lambda } } { e ^ { \alpha / \lambda } + 2 } + \lambda \log \frac { e ^ { \alpha / \lambda } + 2 } { 3 } } . } \end{array}
$$

Viewing this expression as a function of α, diferentiation implies that

$$
\frac { d } { d \alpha } J _ { \lambda , m } ( \pi _ { \mathrm { t e a } } ) = - \frac { e ^ { \alpha / \lambda } } { e ^ { \alpha / \lambda } + 2 } + \frac { 2 ( 1 - \alpha ) e ^ { \alpha / \lambda } } { \lambda ( e ^ { \alpha / \lambda } + 2 ) ^ { 2 } } + \frac { e ^ { \alpha / \lambda } } { e ^ { \alpha / \lambda } + 2 } = \frac { 2 ( 1 - \alpha ) e ^ { \alpha / \lambda } } { \lambda ( e ^ { \alpha / \lambda } + 2 ) ^ { 2 } } > 0 .
$$

Thus, for every $\alpha \in [ 1 / 2 , 1 )$ , we have that $\begin{array} { r } { J _ { \lambda , m } ( \pi _ { \mathrm { t e a } } ) \geq \frac { 1 } { 2 } \frac { e ^ { 1 / ( 2 \lambda ) } } { e ^ { 1 / ( 2 \lambda ) } + 2 } + \lambda \log \frac { e ^ { 1 / ( 2 \lambda ) } + 2 } { 3 } } \end{array}$

We finally compare this teacher lower bound with the student upper bound. Since $e ^ { s } > 1$ for $s > 0$ we have $e ^ { s } / ( e ^ { s } + 2 ) > 1 / 3$ . Hence by direct algebra, we have

$$
\frac { e ^ { 1 / ( 2 \lambda ) } } { e ^ { 1 / ( 2 \lambda ) } + 2 } > \frac { 1 } { 3 } , \log \frac { e ^ { 1 / ( 2 \lambda ) } + 2 } { e ^ { 1 / ( 4 \lambda ) } + 2 } = \int _ { 1 / ( 4 \lambda ) } ^ { 1 / ( 2 \lambda ) } \frac { e ^ { s } } { e ^ { s } + 2 } d s > \frac { 1 } { 3 } \left( \frac { 1 } { 2 \lambda } - \frac { 1 } { 4 \lambda } \right) = \frac { 1 } { 1 2 \lambda } .
$$

Combining these two inequalities with the preceding bounds, we obtain

$$
J _ { \lambda , m } ( \pi _ { \mathrm { t e a } } ) - J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ) \geq \frac { 1 } { 2 } \frac { e ^ { 1 / ( 2 \lambda ) } } { e ^ { 1 / ( 2 \lambda ) } + 2 } - \frac { 1 } { 4 } + \lambda \log \frac { e ^ { 1 / ( 2 \lambda ) } + 2 } { e ^ { 1 / ( 4 \lambda ) } + 2 } > \frac { 1 } { 6 } - \frac { 1 } { 4 } + \lambda \frac { 1 } { 1 2 \lambda } = 0 .
$$

Therefore, we have $J _ { \lambda , m } ( \pi _ { \mathrm { t e a } } ) > J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ) \geq J _ { \lambda , m } ( \pi _ { \mathrm { s t u } , \theta } )$ for every $\theta \in \Theta$ . We finish the proof. □

Proof of Theorem 6.2. We first identify the minimizer of the direct-matching objective and its distance from the oracle student. Then, we bound the SGD error around this minimizer. Finally, a quadratic lower bound for the policy KL divergence converts the remaining parameter distance into the claimed separation.

Recall from the proof of Proposition 6.1 that $p ( \theta _ { i } ) = e ^ { \theta _ { i } } / ( e ^ { \theta _ { i } } + 2 )$ . For a target prompt x and a feasible answer $a _ { 1 : 2 } .$ , all three policy probabilities are positive. Factoring the likelihood ratio through the reference gives

$$
\begin{array} { r l } & { Z _ { \mathrm { S M } } ( \theta ; x , a _ { 1 : 2 } ) = \log \left[ \frac { \pi _ { \mathrm { s t u } , \theta } \left( a _ { 1 : 2 } \mid x \right) } { \pi _ { \mathrm { p r e } } \left( a _ { 1 : 2 } \mid x \right) } \frac { \pi _ { \mathrm { p r e } } \left( a _ { 1 : 2 } \mid x \right) } { \pi _ { \mathrm { t e a } } \left( a _ { 1 : 2 } \mid x \right) } \right] + \lambda \log \frac { \pi _ { \mathrm { s t u } , \theta } \left( a _ { 1 : 2 } \mid x \right) } { \pi _ { \mathrm { p r e } } \left( a _ { 1 : 2 } \mid x \right) } } \\ & { \qquad = ( 1 + \lambda ) \log \frac { \pi _ { \mathrm { s t u } , \theta } \left( a _ { 1 : 2 } \mid x \right) } { \pi _ { \mathrm { p r e } } \left( a _ { 1 : 2 } \mid x \right) } - \log \frac { \pi _ { \mathrm { t e a } } \left( a _ { 1 : 2 } \mid x \right) } { \pi _ { \mathrm { p r e } } \left( a _ { 1 : 2 } \mid x \right) } } \\ & { \qquad = ( 1 + \lambda ) \log \frac { \pi _ { \mathrm { s t u } , \theta } \left( a _ { 1 : 2 } \mid x \right) } { \pi _ { \mathrm { p r e } } \left( a _ { 1 : 2 } \mid x \right) } - \frac { \alpha } { \lambda } R ( x , a _ { 1 : 2 } ) + \log \frac { e ^ { \alpha / \lambda } + 2 } { 3 } . } \end{array}
$$

The second equality uses $\log ( u v ) = \log u + \log v$ and $\log ( 1 / u ) = - \log u ;$ the last uses (D.2).

We now derive the population cost from its definition. Using (6.11), the definition of $Z _ { \mathrm { S M } }$ , and its pointwise expansion above, we obtain

$$
\begin{array} { l } { \displaystyle C _ { \mathrm { S M } } ( \theta ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } [ { \mathrm { K L } \big ( } \pi _ { \mathrm { s t n } , \theta } ( \cdot \ \vert \ \widetilde { x } _ { j } ) \| \pi _ { \mathrm { t e a } } ( \cdot \ \vert \ \widetilde { x } _ { j } ) { \big ) } + \lambda { \mathrm { K L } \big ( } \pi _ { \mathrm { s t n } , \theta } ( \cdot \ \vert \ \widetilde { x } _ { j } ) \| \pi _ { \mathrm { p r e } } ( \cdot \ \vert \ \widetilde { x } _ { j } ) { \big ) } ] } \\ { \displaystyle = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } { \mathrm { E } } _ { a _ { 1 2 } \sim \pi _ { \mathrm { a t n } , \theta } ( \cdot \vert \widetilde { x } _ { j } ) } [ Z _ { \mathrm { S M } } ( \theta ; \widetilde { x } _ { j } , a _ { 1 : 2 } ) ] } \\ { \displaystyle = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } { \mathrm { E } } _ { a _ { 1 2 } \sim \pi _ { \mathrm { a t n } , \theta } ( \cdot \vert \widetilde { x } _ { j } ) } [ ( 1 + \lambda ) \log \frac { \pi _ { \mathrm { s t n } , \theta } ( a _ { 1 : 2 } \ \vert \widetilde { x } _ { j } ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : 2 } \ \vert \widetilde { x } _ { j } ) } - \frac { \alpha } { \lambda } R ( \widetilde { x } _ { j } , a _ { 1 : 2 } ) + \log \frac { e ^ { \alpha / \lambda } + 2 } { 3 } ] } \\  \displaystyle = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } [ ( 1 + \lambda ) { \mathrm { K L } \big ( } \pi _ { \mathrm { s t n } , \theta } ( \cdot \ \vert \widetilde { x } _ { j } ) \| \pi _ { \mathrm { p r e } } ( \cdot \ \vert \widetilde { x } _ { j } ) { \big ) } - \frac { \alpha } { \lambda } \mathbb { E } _  a _ { 1 2 } \sim \pi _  \mathrm \end{array}
$$

The last equality uses linearity of expectation and the definition of KL. The final logarithm is independent of both the answer and the prompt, so averaging leaves it unchanged.

Since $m = 2 d .$ , we can replace the sum over $j$ by the sum over $i \in [ d ]$ and $\epsilon \in \{ + , - \}$ . The calculations in Proposition 6.1 give

$$
\sum _ { \epsilon \in \{ + , - \} } \mathrm { K L } \bigl ( \pi _ { \mathrm { s t u } , \theta } ( \cdot  { | } \widetilde { x } _ { i , \epsilon } )  { \| } \pi _ { \mathrm { p r e } } ( \cdot  { | } \widetilde { x } _ { i , \epsilon } ) \bigr ) = 2 \mathrm { K L } \bigl ( \pi _ { \mathrm { s t u } , \theta } ( \cdot  { | } \widetilde { x } _ { i , + } )  { \| } \pi _ { \mathrm { p r e } } ( \cdot  { | } \widetilde { x } _ { i , + } ) \bigr ) ,
$$

$$
\sum _ { \epsilon \in \{ + , - \} } \mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { i , \epsilon } ) } [ R ( \widetilde { x } _ { i , \epsilon } , a _ { 1 : 2 } ) ] = 2 \frac { 1 + p ( \theta _ { i } ) } { 4 } = \frac { 1 + p ( \theta _ { i } ) } { 2 } .
$$

Substituting these two identities into the preceding cost expression gives

$$
\begin{array} { r l r } & { } & { C _ { \mathrm { S M } } ( \theta ) = \displaystyle \frac { 1 } { 2 d } \sum _ { i = 1 } ^ { d } \left[ 2 ( 1 + \lambda ) \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid \widetilde { x } _ { i , + } ) \parallel \pi _ { \mathrm { p r e } } ( \cdot \mid \widetilde { x } _ { i , + } ) ) - \frac { \alpha } { \lambda } \frac { 1 + p ( \theta _ { i } ) } { 2 } \right] + \log \frac { e ^ { \alpha / \lambda } + 2 } { 3 } } \\ & { } & { = \displaystyle \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left[ ( 1 + \lambda ) \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid \widetilde { x } _ { i , + } ) \parallel \pi _ { \mathrm { p r e } } ( \cdot \mid \widetilde { x } _ { i , + } ) ) - \frac { \alpha } { 4 \lambda } [ 1 + p ( \theta _ { i } ) ] \right] + \log \frac { e ^ { \alpha / \lambda } + 2 } { 3 } . } \end{array}
$$

We next diferentiate this expression with respect to the full parameter vector θ. Notice that the final logarithm is constant in $\theta ,$ and $\nabla _ { \theta } p ( \theta _ { i } ) = p ^ { \prime } ( \theta _ { i } ) e _ { i }$ . Using the KL-gradient identity already proved in Proposition 6.1, we obtain

$$
\begin{array} { c } { { \nabla _ { \theta } C _ { \mathrm { S M } } ( \theta ) = \displaystyle \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left[ ( 1 + \lambda ) \nabla _ { \theta } \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta } ( \cdot \mid \widetilde { x } _ { i , + } ) \parallel \pi _ { \mathrm { p r e } } ( \cdot \mid \widetilde { x } _ { i , + } ) ) - \frac { \alpha } { 4 \lambda } \nabla _ { \theta } [ 1 + p ( \theta _ { i } ) ] \right] } } \\ { { = \displaystyle \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left[ ( 1 + \lambda ) \theta _ { i } p ^ { \prime } ( \theta _ { i } ) e _ { i } - \frac { \alpha } { 4 \lambda } p ^ { \prime } ( \theta _ { i } ) e _ { i } \right] = \displaystyle \frac { 1 } { d } \sum _ { i = 1 } ^ { d } p ^ { \prime } ( \theta _ { i } ) \left[ ( 1 + \lambda ) \theta _ { i } - \frac { \alpha } { 4 \lambda } \right] e _ { i } . } } \end{array}
$$

By the first order optimality condition, the minimizer is $\begin{array} { r } { \theta _ { \mathrm { S M } } ^ { \star } : = \operatorname * { a r g m i n } _ { \theta \in \Theta } C _ { \mathrm { S M } } ( \theta ) = \frac { \alpha } { 4 \lambda ( 1 + \lambda ) } \mathbf { 1 } _ { d } . } \end{array}$

It is feasible and interior because $\| \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } < \sqrt { d } / ( 4 \lambda ) < B$ . Using the oracle parameter from Proposition 6.1, we obtain the fixed parameter gap

$$
\| \theta _ { \mathrm { S M } } ^ { \star } - \theta _ { \lambda , m } ^ { \dagger } \| _ { 2 } = \frac { \sqrt { d } } { 4 \lambda } \left( 1 - \frac { \alpha } { 1 + \lambda } \right) \geq \frac { \sqrt { d } ( 1 - \alpha ) } { 4 \lambda } .\tag{D.3}
$$

The inequality follows from $\alpha / ( 1 + \lambda ) \leq \alpha$

We now bound the error of the SGD iterate relative to $\theta _ { \mathrm { S M } } ^ { \star }$ , starting from the update itself. Define $\mathcal { F } _ { t } ^ { \mathrm { S M } } : = \sigma \Big ( j _ { s } ^ { \mathrm { S M } } , a _ { s , 1 : 2 } ^ { \mathrm { S M } } : 0 \leq s < t \Big )$ . The parameter $\theta _ { t } ^ { \mathrm { S M } }$ is $\mathcal { F } _ { t } ^ { \mathrm { S M } }$ -measurable, and $\theta _ { \mathrm { S M } } ^ { \star }$ is a fixed point of the projection onto Θ. Therefore, the update and nonexpansiveness of projection yields that

$$
\begin{array} { r l } & { \| \theta _ { t + 1 } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } = \left\| \mathrm { P r o j } _ { \Theta } ( \theta _ { t } ^ { \mathrm { S M } } - \eta _ { t } ^ { \mathrm { S M } } \hat { g } _ { t } ^ { \mathrm { S M } } ) - \mathrm { P r o j } _ { \Theta } ( \theta _ { \mathrm { S M } } ^ { \star } ) \right\| _ { 2 } ^ { 2 } } \\ & { \qquad \leq \| \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } - \eta _ { t } ^ { \mathrm { S M } } \hat { g } _ { t } ^ { \mathrm { S M } } \| _ { 2 } ^ { 2 } } \\ & { \qquad = \| \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } - 2 \eta _ { t } ^ { \mathrm { S M } } ( \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } ) ^ { \top } \hat { g } _ { t } ^ { \mathrm { S M } } + ( \eta _ { t } ^ { \mathrm { S M } } ) ^ { 2 } \| \hat { g } _ { t } ^ { \mathrm { S M } } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Taking conditional expectations on both sides, we have that

$$
\begin{array} { r } { \mathbb { E } [  \theta _ { t + 1 } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star }  _ { 2 } ^ { 2 } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } \Big ] \leq \lVert \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \rVert _ { 2 } ^ { 2 } - 2 \eta _ { t } ^ { \mathrm { S M } } \mathbb { E } [ ( \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } ) ^ { \top } \widehat { g } _ { t } ^ { \mathrm { S M } } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } ] + ( \eta _ { t } ^ { \mathrm { S M } } ) ^ { 2 } \mathbb { E } [  \widehat { g } _ { t } ^ { \mathrm { S M } }  _ { 2 } ^ { 2 } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } ] . } \end{array}\tag{D.4}
$$

Since $\eta _ { t } ^ { \mathrm { S M } } > 0$ , an upper bound for the next error requires a lower bound for the cross term and an upper bound for the gradient second moment. We establish these two bounds in turn.

For the cross term, we first identify the conditional mean of $\widehat { g } _ { t } ^ { \mathrm { S M } }$ . At $\boldsymbol { x } = \widetilde { \boldsymbol { x } } _ { i , \epsilon }$ , the student score is

$$
S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : 2 } ) = \nabla _ { \theta } \left[ \theta _ { i } { \bf 1 } \{ a _ { 1 } = 1 \} - \log ( e ^ { \theta _ { i } } + 2 ) \right] = e _ { i } [ { \bf 1 } \{ a _ { 1 } = 1 \} - p ( \theta _ { i } ) ] .
$$

Thus, we obtain $\begin{array} { r } { \| S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : 2 } ) \| _ { 2 } \leq 1 , \ \mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot \vert x ) } [ S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : 2 } ) ] = e _ { i } [ p ( \theta _ { i } ) - p ( \theta _ { i } ) ] = 0 . } \end{array}$

From the definitions of the score and sampled cost, we have

$$
\begin{array} { r l } & { \nabla _ { \theta } \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid x ) = \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid x ) S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : 2 } ) , } \\ & { \nabla _ { \theta } Z _ { \mathrm { S M } } ( \theta ; x , a _ { 1 : 2 } ) = ( 1 + \lambda ) S _ { \mathrm { s t u } , \theta } ( x , a _ { 1 : 2 } ) . } \end{array}
$$

The product rule for the finite sum defining C<sub>SM</sub>, followed by the zero-mean score identity, gives

$$
\begin{array} { r l } & { \nabla _ { \theta } C _ { \mathrm { S M } } ( \theta ) = \displaystyle \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } \left[ S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : 2 } ) \Big ( Z _ { \mathrm { S M } } ( \theta ; \widetilde { x } _ { j } , a _ { 1 : 2 } ) + 1 + \lambda \Big ) \right] } \\ & { \quad \quad \quad \quad = \displaystyle \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { a _ { 1 : 2 } \sim \pi _ { \mathrm { s t u } , \theta } ( \cdot | \widetilde { x } _ { j } ) } \left[ S _ { \mathrm { s t u } , \theta } ( \widetilde { x } _ { j } , a _ { 1 : 2 } ) Z _ { \mathrm { S M } } ( \theta ; \widetilde { x } _ { j } , a _ { 1 : 2 } ) \right] . } \end{array}
$$

Since the algorithm samples $j _ { t } ^ { \mathrm { S M } }$ uniformly and then samples the answer from the current student,

$$
\operatorname* { P r } \left( j _ { t } ^ { \mathrm { { S M } } } = j , a _ { t , 1 : 2 } ^ { \mathrm { S M } } = a _ { 1 : 2 } \Big | \mathscr { F } _ { t } ^ { \mathrm { S M } } \right) = \frac { 1 } { m } \pi _ { \mathrm { s t u } , \theta _ { t } ^ { \mathrm { S M } } } ( a _ { 1 : 2 } \mid \widetilde { x } _ { j } ) , a _ { 1 : 2 } \in A ( \widetilde { x } _ { j } ) .
$$

Thus the preceding gradient identity implies

$$
\mathbb { E } [ \widehat { g } _ { t } ^ { \mathrm { S M } } \mid \mathcal { F } _ { t } ^ { \mathrm { S M } } ] = \nabla _ { \theta } C _ { \mathrm { S M } } ( \theta ) \vert _ { \theta = \theta _ { t } ^ { \mathrm { S M } } } .\tag{D.5}
$$

We now use this unbiasedness identity to bound the cross term. For every $u \in [ - B , B ]$ , by algebra, we have

$$
p ^ { \prime } ( u ) = \sigma ^ { \prime } ( u - \log 2 ) \geq \sigma ^ { \prime } ( B + \log 2 ) = \kappa _ { \mathrm { S M } } ,
$$

Using the population gradient already computed above, together with $( 1 + \lambda ) ( \theta _ { \mathrm { S M } } ^ { \star } ) _ { i } = \alpha / ( 4 \lambda )$ , we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ ( \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } ) ^ { \top } \hat { \mathcal { G } } _ { t } ^ { \mathrm { S M } } \Big \vert \mathcal { F } _ { t } ^ { \mathrm { S M } } \right] = ( \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } ) ^ { \top } \mathbb { E } [ \hat { g } _ { t } ^ { \mathrm { S M } } \mid \mathcal { F } _ { t } ^ { \mathrm { S M } } ] } \\ & { \quad \quad \quad = ( \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } ) ^ { \top } \ \nabla _ { \theta } C _ { \mathrm { S M } } ( \theta ) | _ { \theta = \theta _ { t } ^ { \mathrm { S M } } } } \\ & { \quad \quad \quad = \displaystyle \frac { 1 + \lambda } { d } \sum _ { i = 1 } ^ { d } p ^ { \prime } \Big ( ( \theta _ { t } ^ { \mathrm { S M } } ) _ { i } \Big ) \left[ ( \theta _ { t } ^ { \mathrm { S M } } ) _ { i } - ( \theta _ { \mathrm { S M } } ^ { \star } ) _ { i } \right] ^ { 2 } } \\ & { \quad \quad \quad \geq \displaystyle \frac { ( 1 + \lambda ) \kappa _ { \mathrm { S M } } } { d } \lVert \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \rVert _ { 2 } ^ { 2 } = \mu _ { \mathrm { S M } } \lVert \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \rVert _ { 2 } ^ { 2 } . } \end{array}\tag{D.6}
$$

The first equality uses the measurability of $\theta _ { t } ^ { \mathrm { S M } }$ ; the second uses (D.5).

For the second moment in (D.4), the definition of the gradient estimate and the score bound already proved give

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left\| \widehat { g } _ { t } ^ { \mathrm { S M } } \right\| _ { 2 } ^ { 2 } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } \right] = \mathbb { E } \left[ \left\| S _ { \mathrm { s t u } , \theta _ { t } ^ { \mathrm { S M } } } ( \widetilde { x } _ { j _ { t } ^ { \mathrm { S M } } } , a _ { t , 1 : 2 } ^ { \mathrm { S M } } ) \right\| _ { 2 } ^ { 2 } \Big | Z _ { \mathrm { S M } } ( \theta _ { t } ^ { \mathrm { S M } } ; \widetilde { x } _ { j _ { t } ^ { \mathrm { S M } } } , a _ { t , 1 : 2 } ^ { \mathrm { S M } } ) \Big | ^ { 2 } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } \right] } \\ & { \qquad \leq \mathbb { E } \left[ \left| Z _ { \mathrm { S M } } ( \theta _ { t } ^ { \mathrm { S M } } ; \widetilde { x } _ { j _ { t } ^ { \mathrm { S M } } } , a _ { t , 1 : 2 } ^ { \mathrm { S M } } ) \right| ^ { 2 } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } \right] . } \end{array}
$$

It therefore sufices to bound the sampled cost uniformly. The explicit student/reference ratio is

$$
\log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid \widetilde { x } _ { i , \epsilon } ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : 2 } \mid \widetilde { x } _ { i , \epsilon } ) } = \theta _ { i } { \bf 1 } \{ a _ { 1 } = 1 \} - \log \frac { e ^ { \theta _ { i } } + 2 } { 3 } .
$$

The average $( e ^ { \theta _ { i } } + 1 + 1 ) / 3$ lies between $e ^ { \operatorname* { m i n } \{ 0 , \theta _ { i } \} } { \mathrm { ~ a n d ~ } } e ^ { \operatorname* { m a x } \{ 0 , \theta _ { i } \} }$ . Hence both terms on the right lie between min $\{ 0 , \theta _ { i } \}$ and max $\{ 0 , \theta _ { i } \}$ , and the absolute log ratio is at most $| \theta _ { i } |$ . Similarly, (D.2) and $0 \le \log [ ( e ^ { \alpha / \lambda } + 2 ) / 3 ] \le \alpha / \lambda$ yields $\begin{array} { r } { \left| \log \frac { \pi _ { \mathrm { t e a } } \left( a _ { 1 : 2 } | x \right) } { \pi _ { \mathrm { p r e } } \left( a _ { 1 : 2 } | x \right) } \right| \le \frac { \alpha } { \lambda } } \end{array}$

Returning to the reference decomposition of $Z _ { \mathrm { S M } }$ , we obtain

$$
\begin{array} { r l r } {  {  Z _ { \mathrm { S M } } ( \theta ; x , a _ { 1 : 2 } )  \le ( 1 + \lambda )  \log \frac { \pi _ { \mathrm { s t u } , \theta } ( a _ { 1 : 2 } \mid x ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : 2 } \mid x ) }  +  \log \frac { \pi _ { \mathrm { t e a } } ( a _ { 1 : 2 } \mid x ) } { \pi _ { \mathrm { p r e } } ( a _ { 1 : 2 } \mid x ) }  } } \\ & { } & { \le ( 1 + \lambda ) \vert \theta _ { i } \vert + \frac { \alpha } { \lambda } \le ( 2 + \lambda ) B = G _ { \mathrm { S M } } . } \end{array}
$$

Here $x = \widetilde { x } _ { i , \epsilon }$ and $\alpha / \lambda \leq B$ . This uniform bound proves the required second-moment inequality:

$$
\mathbb { E } \left[ \lVert \widehat { g } _ { t } ^ { \mathrm { S M } } \rVert _ { 2 } ^ { 2 } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } \right] \leq G _ { \mathrm { S M } } ^ { 2 } .\tag{D.7}
$$

Substituting (D.6) and (D.7) into the original error recurrence (D.4), we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ \| \theta _ { t + 1 } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } \Big | \mathcal { F } _ { t } ^ { \mathrm { S M } } \right] \leq \big ( 1 - 2 \mu _ { \mathrm { S M } } \eta _ { t } ^ { \mathrm { S M } } \big ) \| \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } + \big ( \eta _ { t } ^ { \mathrm { S M } } \big ) ^ { 2 } G _ { \mathrm { S M } } ^ { 2 } } \\ & { \qquad = \displaystyle \frac { t } { t + 2 } \| \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } + \frac { G _ { \mathrm { S M } } ^ { 2 } } { \mu _ { \mathrm { S M } } ^ { 2 } ( t + 2 ) ^ { 2 } } , } \end{array}
$$

Taking expectations and applying the law of total expectation, we have that

$$
\mathbb { E } \Big [ \| \theta _ { t + 1 } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } \Big ] \leq \frac { t } { t + 2 } \mathbb { E } \Big [ \| \theta _ { t } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } \Big ] + \frac { G _ { \mathrm { S M } } ^ { 2 } } { \mu _ { \mathrm { S M } } ^ { 2 } ( t + 2 ) ^ { 2 } } .\tag{D.8}
$$

We solve this recurrence by induction to obtain

$$
\mathbb { E } \Big [ \| \theta _ { T } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } ^ { 2 } \Big ] \leq \frac { G _ { \mathrm { S M } } ^ { 2 } } { \mu _ { \mathrm { S M } } ^ { 2 } ( T + 1 ) } .\tag{D.9}
$$

Indeed, the initial bound follows from $\theta _ { 0 } ^ { \mathrm { S M } } = 0 , \| \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } \leq B$ , and

$$
\frac { G _ { \mathrm { S M } } } { \mu _ { \mathrm { S M } } } = \frac { d ( 2 + \lambda ) B } { ( 1 + \lambda ) \kappa _ { \mathrm { S M } } } \geq 4 d B \geq B ,
$$

where $\kappa _ { \mathrm { S M } } \leq 1 / 4$ . For the induction step, substitution of the bound at time t into the expected recurrence gives

$$
\mathbb { E } \left[ \big \lVert \theta _ { t + 1 } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \big \rVert _ { 2 } ^ { 2 } \right] \leq \frac { G _ { \mathrm { S M } } ^ { 2 } } { \mu _ { \mathrm { S M } } ^ { 2 } } \left[ \frac { t } { ( \iota + 2 ) ( \iota + 1 ) } + \frac { 1 } { ( \iota + 2 ) ^ { 2 } } \right] = \frac { G _ { \mathrm { S M } } ^ { 2 } } { \mu _ { \mathrm { S M } } ^ { 2 } } \left[ \frac { 1 } { \iota + 2 } - \frac { 1 } { ( \iota + 1 ) ( \iota + 2 ) ^ { 2 } } \right] \leq \frac { G _ { \mathrm { S M } } ^ { 2 } } { \mu _ { \mathrm { S M } } ^ { 2 } ( \iota + 2 ) } .
$$

It remains to translate the fixed parameter gap and the SGD error into the policy KL in the

theorem. For any $\theta \in \Theta$ , the explicit student policy $\pi _ { \mathrm { s t u } , \theta }$ gives

$$
\begin{array} { r l } & { \quad \mathrm { K L } \Bigl ( \pi _ { \mathrm { s t u } , \boldsymbol { \theta } } ( \cdot \vert \ \widetilde { x } _ { i , c } ) \Bigr \Vert \pi _ { \mathrm { s t u } , \boldsymbol { \theta } _ { \lambda , m } ^ { \dagger } } ( \cdot \vert \widetilde { x } _ { i , c } ) \Bigr ) } \\ & { = \displaystyle \sum _ { a _ { 1 } \in \{ 0 , 1 , \mathrm { m u l l } \} } \pi _ { \mathrm { s t u } , \boldsymbol { \theta } } ( ( a _ { 1 } , \mathrm { E } \boldsymbol { \Theta } ) \vert \widetilde { x } _ { i , c } ) \Biggl [ \Bigl ( \theta _ { i } - ( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } \Bigr ) \mathbf { 1 } \{ a _ { 1 } = 1 \} + \log \frac { e ^ { ( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } } + 2 } { e ^ { \theta _ { i } } + 2 } \Biggr ] } \\ & { = \Bigl ( \theta _ { i } - ( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } \Bigr ) p ( \theta _ { i } ) + \log \frac { e ^ { ( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } } + 2 } { e ^ { \theta _ { i } } + 2 } } \\ & { = \Bigl ( ( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } - \theta _ { i } \Bigr ) ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - s ) p ^ { \prime } \Bigl ( \theta _ { i } + s [ ( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } - \theta _ { i } ] \Bigr ) \ d s \geq \frac { \kappa _ { \mathrm { S M } } } { 2 } \Bigl ( \theta _ { i } - ( \theta _ { \lambda , m } ^ { \dagger } ) _ { i } \Bigr ) ^ { 2 } . } \end{array}
$$

The last equality is by Taylor’s integral formula for $u \mapsto \log ( e ^ { u } + 2 )$ , whose first and second derivatives are $p ( u )$ and $p ^ { \prime } ( u )$ . The segment between the two coordinates lies in $[ - B , B ]$ , so the already proved lower bound $p ^ { \prime } ( u ) \geq \kappa _ { \mathrm { S M } }$ applies throughout the integral. Averaging over the 2d prompts gives

$$
\frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathrm { K L } \Bigl ( \pi _ { \mathrm { s t u } , \theta } ( \cdot  { | } \ : \widetilde { x } _ { j } ) \left\| \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot  { | } \ : \widetilde { x } _ { j } ) \right) \geq \frac { \kappa _ { \mathrm { S M } } } { 2 d } \| \theta - \theta _ { \lambda , m } ^ { \dagger } \| _ { 2 } ^ { 2 } .\tag{D.10}
$$

By the triangle inequality, Cauchy-Schwarz, and (D.9), we have

$$
\| \theta _ { \mathrm { S M } } ^ { \star } - \theta _ { \lambda , m } ^ { \star } \| _ { 2 } \leq \mathbb { E } \Big [ \| \theta _ { T } ^ { \mathrm { S M } } - \theta _ { \lambda , m } ^ { \star } \| _ { 2 } \Big ] + \mathbb { E } \Big [ \| \theta _ { T } ^ { \mathrm { S M } } - \theta _ { \mathrm { S M } } ^ { \star } \| _ { 2 } \Big ] \leq \Big ( \mathbb { E } \Big [ \| \theta _ { T } ^ { \mathrm { S M } } - \theta _ { \lambda , m } ^ { \dagger } \| _ { 2 } ^ { 2 } \Big ] \Big ) ^ { 1 / 2 } + \frac { G _ { \mathrm { S M } } } { \mu _ { \mathrm { S M } } \sqrt { T + 1 } } .
$$

Combining this with (D.3), we obtain $\begin{array} { r } { \left( \mathbb { E } \left[ \| \theta _ { T } ^ { \mathrm { S M } } - \theta _ { \lambda , m } ^ { \dagger } \| _ { 2 } ^ { 2 } \right] \right) ^ { 1 / 2 } \geq \left[ \frac { \sqrt { d } ( 1 - \alpha ) } { 4 \lambda } - \frac { G _ { \mathrm { S M } } } { \mu _ { \mathrm { S M } } \sqrt { T + 1 } } \right] _ { + } . } \end{array}$

We can therefore start from the required policy error and conclude

$$
\begin{array} { r l } & { \mathbb E [ \frac { 1 } { m } \displaystyle \sum _ { j = 1 } ^ { m } \mathrm { K L } ( \pi _ { \mathrm { s t u } , \theta _ { T } ^ { \mathrm { S M } } } ( \cdot \ \vert \ \widetilde { x } _ { j } )  \pi _ { \mathrm { s t u } , \theta _ { \lambda , m } ^ { \dagger } } ( \cdot \vert \widetilde { x } _ { j } ) ) ] \geq \frac { \kappa _ { \mathrm { S M } } } { 2 d } \mathbb E [ \Vert \theta _ { T } ^ { \mathrm { S M } } - \theta _ { \lambda , m } ^ { \dagger } \Vert _ { 2 } ^ { 2 } ] } \\ & { \qquad \geq \displaystyle \frac { \kappa _ { \mathrm { S M } } } { 2 d } [ \frac { \sqrt { d } ( 1 - \alpha ) } { 4 \lambda } - \frac { G _ { \mathrm { S M } } } { \mu _ { \mathrm { S M } } \sqrt { T + 1 } } ] _ { + } ^ { 2 } . } \end{array}
$$

The first inequality is (D.10); the second is the square of the preceding bound.

For the stated iteration threshold, substituting the definition of µ<sub>SM</sub> gives

$$
T + 1 \geq \frac { 6 4 \lambda ^ { 2 } d G _ { \mathrm { S M } } ^ { 2 } } { ( 1 + \lambda ) ^ { 2 } \kappa _ { \mathrm { S M } } ^ { 2 } ( 1 - \alpha ) ^ { 2 } } = \left( \frac { 8 \lambda G _ { \mathrm { S M } } } { \mu _ { \mathrm { S M } } \sqrt { d } ( 1 - \alpha ) } \right) ^ { 2 } .
$$

Hence $G _ { \mathrm { S M } } / [ \mu _ { \mathrm { S M } } \sqrt { T + 1 } ] \le \sqrt { d } ( 1 - \alpha ) / \big ( 8 \lambda )$ , and the KL lower bound is at least

$$
\frac { \kappa _ { \mathrm { S M } } } { 2 d } \left[ \frac { \sqrt { d } ( 1 - \alpha ) } { 4 \lambda } - \frac { \sqrt { d } ( 1 - \alpha ) } { 8 \lambda } \right] ^ { 2 } = \frac { \kappa _ { \mathrm { S M } } ( 1 - \alpha ) ^ { 2 } } { 1 2 8 \lambda ^ { 2 } } > 0 .
$$