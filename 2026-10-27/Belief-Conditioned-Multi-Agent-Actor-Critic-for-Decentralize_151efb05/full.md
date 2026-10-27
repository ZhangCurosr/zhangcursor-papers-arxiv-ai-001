# Belief-Conditioned Multi-Agent Actor Critic for Decentralized LLM Collaboration

Tianle Chen Shuo Liu Christopher Amato<sup>∗</sup> Khoury College of Computer Sciences, Northeastern University, Boston, MA, 02215, USA {chen.tianle, liu.shuo2, c.amato}@northeastern.edu

## Abstract

Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL), but many approaches still rely on predefined workflows, limiting efficiency and scalability. Decentralized LLM collaboration is challenging because agents execute independently with limited communication, share a joint objective, and must infer useful information from communication traces. In this paper, we propose BCMAAC, a Belief-Conditioned Multi-Agent Actor-Critic method that maintains beliefs over contextual information and uses them to facilitate training. Experiments on Minecraft game-playing, web searching and tool-using tasks show that BCMAAC improves task performance and collaborative behavior compared to naive actor-critic baselines. The improvements are particularly pronounced in multi-turn tasks with richer contextual information, where belief modeling provides a clear advantage.

## 1 Introduction

Recent advances in LLMs have motivated researchers to use them as intelligent agents and interact with environment in multiple turns Yao et al. [2022]. Many works use reinforcement learning with verifiable rewards (RLVR), where rewards are derived from verifiable signals such as rules and unit tests [Lambert et al., 2024, Guo et al., 2025, Kimi Team et al., 2025, Liu et al., 2025, Guo et al., 2025]. Compare with reinforcement learning from human feedback (RLHF) that trains a reward model using human preference data, RLVR typically has lower cost, greater scalability, and more objective feedback in scenarios where task success can be automatically verified [Lambert et al., 2024, Guo et al., 2025].

Existing works have employed multi-LLM collaboration to improve performance. Prompt-based approaches for multi-agent systems have demonstrated improved problem-solving through role assignment and agent-to-agent communications [Du et al., 2024, Qian et al., 2024, Wu et al., 2023]. However, they often follow a manually designed interaction protocols. Recent work has combined reinforcement learning approaches with LLM collaboration. For instance, CoLLM-CC [Liu et al., 2026] is a multi-agent actor-critic (MAAC) methods that utilize centralized value estimation while keeping decentralized execution.

Despite the advantages of MAAC in LLM collaboration, it faces many challenges. Given the partial observability of the task, the policy is a pre-trained LLM, and actions are generated as sequences of natural language tokens. Action selection relies not only on task reasoning but also on inferences about the environment and the other agents’ internal states, thereby increasing the difficulty of decision-making and optimization. Furthermore, the problem of long-horizon sparse rewards is another challenge, making it difficult to capture effective learning signals in a timely manner. This manifests as rewards that are typically delayed, sparse, and concentrated in the terminal state, making it hard for the model to determine which intermediate communications, inferences, or the environmental interactions indeed contributed to the final success.

We study decentralized LLM collaboration optimized with RLVR, focusing on environmental state estimation and the other agents’ intention inference. We propose BCMAAC, a belief-conditioned multi-agent actor-critic method. BCMAAC uses a recurrent belief encoder to maintain latent belief context from structured side information and conditions this context onto each LLM actor; during the training phase, a centralized critic aggregates the multi-agent belief context to estimate the value, while the execution phase remains decentralized, with each agent relying solely on its own history and local belief context for decision-making.

Our contributions can be summarized as follows. (i) We propose BCMAAC. BCMAAC learns execution-time belief context from structured side information and uses it for both actor conditioning and centralized value estimation. (ii) We show that explicit belief context is particularly useful for long-horizon, partially observable collaboration, where useful actions depend on persistent hidden state, the specific information of other agents, and delayed or sparse feedback. This gives agents an intermediate representation for deciding what to do and communicate across turns. (iii) Experiments on collaboration tasks show that BCMAAC outperforms MAAC. And the ablation show that both task-belief and partner-action auxiliary signals help improve performance.

## 2 Related Works

LLM-based multi-agent collaboration. Recent work uses multiple LLM agents to solve tasks that are difficult for a single model. Multi-agent debate improves the answer by independent reasoning and iterative refinement [Du et al., 2024]. ChatDev organizes role-specialized agents along a structured software-development pipeline [Qian et al., 2024]. AutoGen provides a general framework in which conversable agents cooperate with tools and optional human input [Wu et al., 2023]. CAMEL relies on role-playing between agents, which cooperate through prompting [Li et al., 2023]. These systems demonstrate the benefits of communication and task decomposition. However, their collaboration strategies are largely specified by prompts, leaving the collaboration pattern fixed and unable to improve with task experience.

RL for LLM collaboration. Besides prompt engineering, recent studies also train multi-LLM systems. MAGRPO jointly trains cooperating LLM agents on joint rewards without a critic [Liu et al., 2025]. Dr. MAS investigates stability issues arising in GRPO-style critic-free training of multi-agent LLM systems. They improve performance by normalizing each agent’s advantage with its own reward statistics [Feng et al., 2026]. MAPoRL trains multiple LLMs with verifier-based rewards to encourage collaborative behavior [Park et al., 2025]. MARFT gives a general formulation for reinforcement fine-tuning of LLM multi-agent systems [Liao et al., 2025]. Most closely related is MAAC, which formulates LLM collaboration as cooperative MARL. It adopts centralized training with decentralized execution, with a centralized critic and shared rewards, for long-horizon collaboration [Kraemer and Banerjee, 2016, Liu et al., 2026]. Unlike prompt-defined systems, these methods allow collaboration policies to adapt to task feedback. Nevertheless, they usually use raw interaction histories as context. The actor and critic must reconstruct task state and the knowledge of other agents from long textual histories, making representation learning difficult.

Partner modeling and belief learning in MARL. Early methods condition an agent’s policy on observed opponent behavior or infer another agent’s hidden goal [He et al., 2016, Raileanu et al., 2018]. Later work learns more general representations of agents’ latent characteristics from their behavioral histories [Rabinowitz et al., 2018, Papoudakis et al., 2021]. Also some research improves agent collaboration by learned communication [Foerster et al., 2016, Sukhbaatar et al., 2016, Das et al., 2019]. These methods usually assume low-dimensional observations, compact policies, or continuous communication. In contrast, LLM agents act over long textual histories and communication, making belief and partner modeling more challenging.

## 3 Background

## 3.1 Decentralized LLM Collaboration

In decentralized LLM collaboration, given a task, each agent is provided with a natural language prompt. The process can be described as a multi-turn interaction. Each agent maintains its own context locally and generates a response based on its own policy. The responses of all agents are aggregated within the environment into a joint action, after which the environment updates the task state based on this joint action and returns new observations, feedback, and rewards. When generating actions, each agent rely on its own local observations, historical context, received messages, and tools to which it has access. Agents can exchange information through the communication channels. The communication itself is part of the action. And decentralized execution supports parallel reasoning and reduces context pressure on individual agents.

## 3.2 LLM Dec-POMDP Formulation

We formulate decentralized LLM collaboration as a cooperative Dec-POMDP [Oliehoek and Amato, 2016, Liu et al., 2025], denoted by $\mathcal { M } = \langle \mathcal { T } , \mathcal { V } , C , \dot { M } , \mathcal { S } , \{ \mathcal { O } _ { i } \} _ { i \in \mathcal { I } } , \{ A _ { i } \} _ { i \in \mathcal { I } } , R , T , H , \gamma \rangle$ . Here, ${ \mathcal { T } } = \{ 1 , \ldots , n \}$ is the set of n LLM agents, V is the token vocabulary, C is the context-window length, M is the maximum response length, H is the episode horizon, and $\gamma \in [ 0 , 1 ]$ is the discount factor. The global state space is $\mathcal { S } = \tilde { \cal S } ^ { \mathrm { s y s } } \times \tilde { \cal S } ^ { \mathrm { u s r } }$ , where $S ^ { \mathrm { s y s } }$ denotes the system-accessible state and $S ^ { \mathrm { u s r } }$ the user-hidden state. At turn t, the global state is $\mathbf { \Psi } _ { s _ { t } } = ( s _ { t } ^ { \mathrm { s y s } } , s _ { t } ^ { \mathrm { u s r } } ) \dot { \in } \mathcal { S }$

Each agent i has a local observation space O<sub>i</sub>, with joint observation space $\mathcal { O } = \times _ { i \in \mathcal { I } } \mathcal { O } _ { i }$ . At turn t, agent i receives a local observation $o _ { i , t } \in \mathcal { O } _ { i }$ , typically represented as a natural-language prompt, and the joint observation is $\mathbf { o } _ { t } = ( o _ { i , t } ) _ { i \in \mathcal { T } } \in \mathcal { O }$ . Each agent also has an action space $A _ { i } ,$ , with joint action space $\mathcal { A } = \times _ { i \in \mathcal { T } } \mathcal { A } _ { i }$ . An action is a generated response sequence $a _ { i , t } = ( a _ { i , t } ^ { 1 } , a _ { i , t } ^ { 2 } , \ldots , a _ { i , t } ^ { m _ { i , t } } )$ where $m _ { i , t } \leq M$ and thus $\mathcal { A } _ { i } \subseteq \mathcal { V } ^ { \leq M }$ . The joint action is $\mathbf { a } _ { t } = ( a _ { i , t } ) _ { i \in \mathcal { T } } \in \mathcal { A }$ . The environment evolves according to $T : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ , with $s _ { t + 1 } \sim T ( \cdot \mid s _ { t } , \mathbf { a } _ { t } )$ . The joint reward is given by $R : S ^ { \mathrm { s y s } } \times \mathcal { A }  \mathcal { \bar { \mathbb { R } } }$ , with $r _ { t } = R ( s _ { t } ^ { \mathrm { s y s } } , \mathbf { \dot { a } } _ { t } )$

Agent i acts from its local history $h _ { i , t } = \left( o _ { i , 0 } , a _ { i , 0 } , o _ { i , 1 } , a _ { i , 1 } , \ldots , a _ { i , t - 1 } , o _ { i , t } \right)$ , represented within a context window of at most C tokens. The joint history is $\mathbf { h } _ { t } = ( h _ { i , t } ) _ { i \in \mathcal { I } }$ . Each policy maps a local history to a distribution over response sequences, $\pi _ { \theta _ { i } } \left( a _ { i , t } \ \mid \ h _ { i , t } \right)$ , and the joint policy is $\pi _ { \theta } = \{ \pi _ { \theta _ { i } } \} _ { i \in \mathbb { Z } }$ . Under decentralized execution, $\begin{array} { r } { \pi _ { \theta } ( \mathbf { a } _ { t } \mid \mathbf { h } _ { t } ) = \prod _ { i \in \mathcal { T } } \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } ) } \end{array}$ . At the token level, $\begin{array} { r } { \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } ) = \prod _ { \mu = 1 } ^ { m _ { i , t } } \pi _ { \theta _ { i } } ( a _ { i , t } ^ { \mu } \mid h _ { i , t } , a _ { i , t } ^ { < \mu } ) } \end{array}$

The objective is to maximize the discounted expected return over the episode, $\begin{array} { r l } { J ( \pmb { \theta } ) } & { { } = } \end{array}$ $\begin{array} { r } { \mathbb { E } _ { \pi _ { \theta } } [ \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } r _ { t } ] } \end{array}$ , yielding the optimal joint policy $\begin{array} { r } { \pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } [ \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } r _ { t } ] } \end{array}$ . An episode trajectory is $\tau ~ = ~ ( { \bf o } _ { 0 } , { \bf a } _ { 0 } , r _ { 0 } , { \bf o } _ { 1 } , { \bf a } _ { 1 } , r _ { 1 } , \dots , { \bf o } _ { H - 1 } , { \bf a } _ { H - 1 } , r _ { H - 1 } , { \bf o } _ { H } )$ The discounted return from turn t is $\begin{array} { r } { G _ { t } \ = \ \sum _ { \ell = t } ^ { H - 1 } \gamma ^ { \ell - t } r _ { \ell } } \end{array}$ , and the corresponding joint value on the joint history is $V ^ { \pi } ( \mathbf h _ { t } ) = \mathbb { E } _ { \pi } [ \sum _ { \ell = t } ^ { H - 1 } \gamma ^ { \ell - t } r _ { \ell } \mid \mathbf h _ { t } ]$

## 3.3 MAAC

In centralized-training decentralized-execution MAAC [Liu et al., 2026], each actor depends only on its local history at execution time, $\pi _ { \theta _ { i } } \left( \boldsymbol { a } _ { i , t } \ | \ h _ { i , t } \right)$ . During training, a centralized critic estimates the team value. We write $V _ { \phi } ( \mathbf { h } _ { t } ) = \mathbb { E } _ { \pi } [ \dot { G } _ { t } \mid \mathbf { h } _ { t } ]$ to indicate that the critic is conditioned on the joint history. The corresponding advantage estimate is $\hat { A } _ { t } = r _ { t } + \gamma V _ { \phi } ( \mathbf { h } _ { t + 1 } ) - V _ { \phi } ( \mathbf { h } _ { t } )$ , and the actor objective uses the log-probability of the complete response sequence,

$$
\mathcal { L } _ { \pi } ^ { \mathrm { M A A C } } = - \sum _ { t } \sum _ { i \in \mathcal { I } } \hat { A } _ { t } \log \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } ) .\tag{1}
$$

MAAC reduces the variance of Monte Carlo returns by bootstrapping from a learned value function and propagates terminal rewards back to intermediate turns. However, MAAC remains limited by the difficulty of extracting task-relevant state from long histories. It also lacks explicit modeling of the knowledge of other agents and future actions.

## 4 Belief-Conditioned MAAC

## 4.1 BCMAAC Formulation

Each episode samples a task $\xi \sim \mathcal { D }$ , which determines the environment Env. Besides the language history $h _ { i , t }$ , each agent receives structured side information $x _ { i , t } \in \mathcal { X } _ { i }$ that can be constructed at execution time; the joint side information is $\mathbf { x } _ { t } = ( x _ { i , t } ) _ { i \in \mathcal { I } }$ . The key constraint is that $x _ { i , i }$ <sub>t</sub> only depends on information available to agent i at execution time, including received communication. BCMAAC maintains a latent belief through a recurrent belief encoder $f _ { \psi }$ , yielding

$$
\begin{array} { r } { ( c _ { i , t } , z _ { i , t + 1 } , \hat { { \bf b } } _ { i , t } , \hat { { \bf p } } _ { i , t } ) = f _ { \psi } ( x _ { i , t } , z _ { i , t } ) . } \end{array}\tag{2}
$$

Here, $z _ { i , t }$ is the recurrent state carried across turns, $c _ { i , t }$ is the belief context passed to the actor and critic. And $\hat { \mathbf { b } } _ { i , t }$ is a task belief prediction, and $\hat { \mathbf { p } } _ { i , t }$ is a partner action prediction.

The BCMAAC actor is conditioned on local history and local belief context, $\pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } , c _ { i , t } )$ . For an LLM actor, the action remains a response sequence. The predictions $\hat { \mathbf { b } } _ { i , t }$ and $\hat { \mathbf { p } } _ { i , t }$ are train-time auxiliary heads, supervised by per-turn targets $\mathbf { b } _ { i , i } ^ { * }$ and $\mathbf { p } _ { i , t } ^ { * } .$ These targets are produced by a labeling rule $\Gamma _ { \xi }$ . For $\Gamma _ { \xi }$ , it reads the completed trajectory τ together with the environment’s internal state, and writes down the ground truth that the agents could not observe. Such privileged information is used only during training.

During training, the centralized critic aggregates all agent contexts. Let $\mathbf { c } _ { t } = \mathrm { J o i n t } ( \{ c _ { i , t } \} _ { i \in \mathcal { T } } )$ Joint(·) may be implemented by concatenation with fixed order. The critic estimates the joint value $V _ { \phi } ( \mathbf { c } _ { t } ) = \mathbb { E } _ { \pmb { \pi } } [ G _ { t } \ | \ \mathbf { c } _ { t } ]$

The BCMAAC advantage is $\hat { A } _ { t } = r _ { t } + \gamma V _ { \phi } ( \mathbf { c } _ { t + 1 } ) - V _ { \phi } ( \mathbf { c } _ { t } )$ , and the actor and critic losses are

$$
\mathcal { L } _ { \pi } = - \sum _ { t } \sum _ { i \in \mathcal { I } } \hat { A } _ { t } \log \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } , c _ { i , t } ) ,\tag{3}
$$

$$
\mathcal { L } _ { V } = \sum _ { t } \left( V _ { \phi } ( \mathbf { c } _ { t } ) - r _ { t } - \gamma V _ { \phi } ( \mathbf { c } _ { t + 1 } ) \right) ^ { 2 } .\tag{4}
$$

## 4.2 Theoretical Properties

Let $\mathcal { F } _ { t }$ denote the information available before sampling the joint action at turn $t .$ The joint context $\mathbf { c } _ { t }$ is computed as a compressed representation of the execution information in $\mathcal { F } _ { t }$ . Then $\mathbf { c } _ { t }$ is $\mathcal { F } _ { t }$ -measurable.

Remark 4.1 shows that replacing $G _ { t }$ with $G _ { t } - \beta ( \mathbf { c } _ { t } )$ does not change the expected value of the policy gradient estimate; the form is entirely analogous to the score-function baseline identity [Williams, 1992].

Remark 4.1. Let $\mathbf { c } _ { t }$ be fixed before sampling the current joint action $\mathbf { a } _ { t } ,$ , and let $\beta ( \mathbf { c } _ { t } )$ be independent of the current action. Then replacing $G _ { t }$ with $G _ { t } - \beta ( \mathbf { c } _ { t } )$ does not change the expectation of the policy-gradient estimator:

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \boldsymbol \pi } \left[ \sum _ { i \in \mathcal { T } } \nabla _ { \theta _ { i } } \log \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } , c _ { i , t } ) \left( G _ { t } - \beta ( \mathbf { c } _ { t } ) \right) \right] } \\ { \displaystyle = \mathbb { E } _ { \boldsymbol \pi } \left[ \sum _ { i \in \mathcal { T } } \nabla _ { \theta _ { i } } \log \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } , c _ { i , t } ) G _ { t } \right] . } \end{array}
$$

Proof. Conditioning on $\mathcal { F } _ { t } , \ \beta ( \mathbf { c } _ { t } )$ is pre-action and does not depend on $a _ { i , t }$ . Since the joint policy factorizes, marginalizing over $\mathbf { a } _ { - i , t }$ reduces each term to a single-agent identity. Using $\pi _ { \boldsymbol { \theta } _ { i } } \nabla _ { \boldsymbol { \theta } _ { i } } \log \pi _ { \boldsymbol { \theta } _ { i } } = \nabla _ { \boldsymbol { \theta } _ { i } } \pi _ { \boldsymbol { \theta } _ { i } }$

$$
\begin{array} { r l } & { \mathbb { E } _ { a _ { i , t } \sim \pi _ { \theta _ { i } } } \left[ \nabla _ { \theta _ { i } } \log \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } , c _ { i , t } ) \beta ( \mathbf { c } _ { t } ) \mid \mathcal { F } _ { t } \right] } \\ & { \quad = \beta ( \mathbf { c } _ { t } ) \nabla _ { \theta _ { i } } \displaystyle \sum _ { a _ { i , t } } \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } , c _ { i , t } ) = 0 . } \end{array}
$$

Therefore the baseline term contributes zero in expectation. By linearity and then taking expectation over $\mathcal { F } _ { t }$ , the two policy-gradient estimators have the same expectation. □

Proposition 4.2 shows the ideal case: when the learned context is sufficient for future return, it lose no value-relevant information compared to the full-history context (proof in Appendix B).

Proposition 4.2. Ifthe joint beliefcontext $\mathbf { c } _ { t }$ is sufficientforfuture return in the sense that

$$
\mathbb { E } _ { \pi } [ G _ { t } \mid { \mathcal F } _ { t } , { \mathbf c } _ { t } ] = \mathbb { E } _ { \pi } [ G _ { t } \mid { \mathbf c } _ { t } ] ,
$$

then the value conditioned on all available information equals the value conditioned on context: $V ^ { \pi } ( \mathcal { F } _ { t } ) = V ^ { \pi } ( \mathbf { c } _ { t } )$

Remark 4.3 notes that the auxiliary losses supply the belief encoder with a per-turn representation signal, which is especially useful when task rewards are sparse.

Remark 4.3. In sparse-reward tasks where task reward is concentrated near the end of the episode, return-driven representation learning depends on discounted returns. In BCMAAC, the belief encoder also receives gradients from

$$
\mathcal { L } _ { \mathrm { a u x } } = \sum _ { t } \sum _ { i \in \mathcal { T } } \left[ \lambda _ { b } \ell _ { b } ( \hat { \mathbf { b } } _ { i , t } , \mathbf { b } _ { i , t } ^ { * } ) + \lambda _ { p } \ell _ { p } ( \hat { \mathbf { p } } _ { i , t } , \mathbf { p } _ { i , t } ^ { * } ) \right] ,
$$

where the labels $\mathbf { b } _ { i , t } ^ { * }$ and $\mathbf { p } _ { i , t } ^ { * }$ are generated from the trajectory by the labeling rule $\Gamma _ { \xi } ( \tau )$ . These auxiliary gradients compare predictions directly with per-turn labels. Thus, whenever such labels are available, the belief encoder receives an additional per-turn representation signal, indicating that BCMAAC supplies an additional source of representation learning for state estimation and the modeling of agents.

## 4.3 BCMAAC Algorithm

Figure 1 and Algorithm 1 summarize the full training procedure, which has three components: trajectory collection, auxiliary label generation, and actor-critic updates. After each rollout, one-step TD targets are computed, and auxiliary labels are generated by $\{ \mathbf { y } _ { i , t } ^ { * } \} _ { i , t } \gets \Gamma _ { \xi } \dot { ( } \tau )$ , where $\mathbf { y } _ { i , t } ^ { * } = ( \mathbf { b } _ { i , t } ^ { * } , \mathbf { p } _ { i , t } ^ { * } )$ collects the task-belief and partner-action targets that supervise the auxiliary predictions $\hat { \mathbf { y } } _ { i , t } = ( \hat { \mathbf { b } } _ { i , t } , \hat { \mathbf { p } } _ { i , t } )$

For auxiliary supervision, $\ell _ { b }$ is a masked BCE loss for task-belief prediction and $\ell _ { p } \ \mathrm { ~ a ~ }$ cross-entropy loss for partneraction prediction, with the mask $\omega _ { i , t }$ excluding unlabeled task-belief entries. Summing over agents and turns gives $\begin{array} { r } { \mathcal { L } _ { b } = \sum _ { t } \sum _ { i \in \mathcal { T } } \ell _ { b } ( \hat { \mathbf { b } } _ { i , t } , \mathbf { b } _ { i , t } ^ { * } ) } \end{array}$ and $\mathcal { L } _ { p } \ =$ $\begin{array} { r } { \sum _ { t } \sum _ { i \in \mathcal { T } } \ell _ { p } \big ( \hat { \mathbf { p } _ { i , t } } , \mathbf { p } _ { i , t } ^ { * } \big ) . } \end{array}$ , and the auxiliary objective is $\begin{array} { r } { \mathcal { L } _ { \mathrm { a u x } } = \dot { \lambda } _ { b } \mathcal { L } _ { b } + \lambda _ { p } \mathcal { L } _ { p } , } \end{array}$

During training, contexts are recomputed from the stored structured side information, $\begin{array} { r c l } { ( c _ { i , t } ^ { b } , z _ { i , t + 1 } ^ { b } , \hat { \mathbf { b } } _ { i , t } ^ { b } , \hat { \mathbf { p } } _ { i , t } ^ { b } ) } & {  } & { f _ { \psi } ( x _ { i , t } ^ { b } , z _ { i , t } ^ { b } ) } \end{array}$

![](images/fc0c6e02aa390dbd74881fb225220fe19aca28f3e9defb8b2687f600976d3864.jpg)  
Figure 1: Illustration of the BCMAAC framework.

and $\mathbf { c } _ { t } ^ { b } = \mathrm { J o i n t } ( \{ c _ { i , t } ^ { b } \} _ { i \in \mathcal { T } } )$ . Given a rollout buffer B, BCMAAC computes one-step TD advantages. The critic is optimized by TD regression, the actors by advantage-weighted response log-probabilities, and the belief encoder by the combined value, policy, and auxiliary losses. Thus, with a value-loss weight $\lambda _ { V }$ , ϕ is updated only by the value loss, θ<sub>i</sub> by the policy loss, and ψ by $\lambda _ { V } \mathcal { L } _ { V } + \mathcal { L } _ { \mathrm { a u x } } + \mathcal { L } _ { \pi }$

During execution, the centralized critic and auxiliary heads are not required. Each agent recursively computes $( c _ { i , t } , z _ { i , t + 1 } ) \gets f _ { \psi } ( x _ { i , t } , z _ { i , t } )$ and samples $a _ { i , t } \sim \pi _ { \theta _ { i } } ( \cdot \mid h _ { i , t } , c _ { i , t } )$ . Since $x _ { i , t }$ is constructed only from agent $i \ ' _ { \mathbf { S } }$ execution-time information, $c _ { i , t }$ is local as well. Thus BCMAAC preserves decentralized execution: $\begin{array} { r } { \pi _ { \theta } ( \mathbf { a } _ { t } \mid \mathbf { h } _ { t } , \mathbf { c } _ { t } ) = \prod _ { i \in \mathcal { T } } \pi _ { \theta _ { i } } ( a _ { i , t } \mid h _ { i , t } , c _ { i , t } ) } \end{array}$

Algorithm 1 BCMAAC   
1: Input: Task distribution $\mathcal { D } ;$ initial parameters $( \theta , \psi , \phi ) ;$   
2: for epoch $e = 1 , \ldots , E$ do   
3: Initialize rollout buffer $B  \emptyset$   
4: for rollout $k = 1 , \ldots , K _ { \mathrm { r o l l } }$ do   
5: Sample $\xi \sim \mathcal { D } ;$ reset $\xi ;$ initialize $h _ { i , 0 } , x _ { i , 0 } , z _ { i , 0 }$ and trajectory $\tau  \emptyset$   
6: for turn $\dot { t } = 0 , \dots , \dot { H _ { \mathrm { ~ - ~ } } } 1$ do   
7: Receive current prompt $o _ { i , t }$ and structured observation $x _ { i , t } ;$ update local history $h _ { i , t }$   
8: $( c _ { i , t } , z _ { i , t + 1 } , \hat { \mathbf { b } } _ { i , t } , \hat { \mathbf { p } } _ { i , t } ) \gets f _ { \psi } ( x _ { i , t } , z _ { i , t } )$ , ∀i   
9: Sample $a _ { i , t } \sim \pi _ { \theta _ { i } } ( \cdot \mid h _ { i , t } , c _ { i , t } )$ , ∀i; execute $\mathbf { a } _ { t }$ in Env   
10: Observe $( \mathbf { o } _ { t + 1 } , \mathbf { x } _ { t + 1 } , r _ { t } ) ;$ append transition to τ   
11: end for   
12: Compute one-step TD targets; generate labels $\{ ( \mathbf { b } _ { i , t } ^ { * } , \mathbf { p } _ { i , t } ^ { * } ) \} _ { i , t }  \Gamma _ { \xi } ( \tau )$ ; add τ to B   
13: end for   
14: Initialize losses $\bar { \mathcal { L } } _ { V } , \bar { \mathcal { L } } _ { \pi } , \bar { \mathcal { L } } _ { \mathrm { a u x } }  0$   
15: for trajectory $\tau ^ { b } \in \dot { \boldsymbol { B } }$ do   
16: Reset recurrent states $z _ { i , 0 } ^ { b } \gets 0 ,$ ∀i   
17: for transition t in $\tau ^ { b }$ do   
18: Recompute local contexts and auxiliary predictions via $f _ { \psi } , \forall i$   
19: Form joint context $\mathbf { c } _ { t } ^ { b }$ and advantage $\hat { A } _ { t } ^ { b }$   
20: Accumulate value, policy, and auxiliary losses   
21: end for   
22: end for   
23: Update critic ϕ $ \phi - \alpha _ { V } \nabla _ { \phi } ( \lambda _ { V } \bar { \mathcal { L } } _ { V } )$   
24: Update actors $\theta _ { i }  \theta _ { i } - \alpha _ { \pi } \nabla _ { \theta _ { i } } \bar { \mathcal { L } } _ { \pi } , \forall i \in \mathcal { T }$   
25: Update encoder $\psi  \psi - \overset { \cdot } { \alpha } _ { \psi } \overset { \cdot } { \nabla } _ { \psi } ( \lambda _ { V } \bar { \mathcal { L } } _ { V } + \bar { \mathcal { L } } _ { \pi } + \bar { \mathcal { L } } _ { \mathrm { a u x } } )$   
26: end for   
27: Output: Belief encoder $f _ { \psi }$ and joint policy π<sub>θ</sub>

## 5 Experiments

## 5.1 Setup

Minecraft Collaboration. Minecraft games (Figure 2) are long-horizon testbeds with explicit physical state and partial observability. Each agent observes only its local view and received messages, where the agents maintain environment beliefs, track the other agents’ progress, avoid redundant exploration, and coordinate actions through communication. In Bridge Building, two agents start on opposite sides of a damaged gap and should construct a safe connected path within limited turns. The gap contains visually ambiguous pillars that may either provide useful support or indicate hazardous regions, where nearby construction incurs large penalties. Because pillar types are initially hidden, agents must explore informative viewpoints, communicate discoveries, and coordinate bridge segments from both sides. In Resource Gathering, resource cells contain different amounts of wood, stone, and iron, and agents must satisfy all collection targets within a fixed horizon. Agents have asymmetric extraction capabilities, so they may observe resources that only the other agents’ can collect. Effective collaboration therefore requires separating actionable observations from information that should be communicated. At each turn, agents specify broadcast resource facts, movement paths, and extraction targets, and extraction is only allowed near the final location.

Web Collaboration. In Web Collaboration, agents operate in an open-ended environment. The relevant state is not a location or block layout, but the progress of the search process: what evidence each agent has collected, which sources still look promising, whether the available evidence is enough to support an answer, and what the other agent are likely to do next. As the dominant reward is given after the final answer is checked, agents receive little direct feedback while searching. They should decide whether to continue gathering evidence, share partial findings, wait for more information from the other agents, or commit to an answer. In Cross-Lingual Gathering, agents gather cross-lingual information sources [Longpre et al., 2021] and produce a final answer. Agents should bridge language and phrasing differences, convert locally retrieved information into evidence useful to the other agents, and decide whether evidence from different languages supports the same answer. Simple parallel search may improve coverage, but without modeling what the other agent have found, agents can still duplicate search effort or submit incomplete answers too early, leading to inefficiencies in the collaboration process. In Multi-Source Gathering, agents aggregate evidence from multiple sources [Chen et al., 2020]. A single source may provide only partial evidence, multiple sources can be redundant or conflicting, and the final answer depends on combining evidence across sources. Effective collaboration requires agents to know not only what they have found, but also what the other agents have covered, what remains missing, and whether the agents are ready to answer.

Table 1: Evaluation results across Minecraft, Web, and Tool collaboration. AC denotes singleagent actor-critic method; Parallel and Pipeline represents 2 test-time prompt-based multi-agent collaboration methods; MAAC is the centralized-critic multi-agent actor-critic baseline; BCMAAC-P and BCMAAC-B are variants that only partner-action and task-belief auxiliary beliefs, respectively. Success is the success rate and match score is the normalized answer-matching score, both are reported as percentages and averaged over evaluation runs. Standard deviations are reported in subscripts.
<table><tr><td rowspan="2">Method</td><td colspan="2">Minecraft</td><td colspan="2">Web</td><td colspan="2">Tool</td></tr><tr><td>Resource Gathering Success (%)</td><td>Bridge Building Success (%)</td><td>Cross-Lingual Gathering Match Score (%)</td><td>Multi-Source Gathering Match Score (%)</td><td>Stateful Tool Success (%)</td><td>Filesystem Wrangling Success (%)</td></tr><tr><td>Raw Model</td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $4 5 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $1 1 . 3 _ { \pm 0 . 0 }$ </td><td> $3 3 . 3 { \scriptstyle \pm 0 . 0 }$ </td><td> $5 3 . 8 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td>AC</td><td> $7 0 . 1 { \scriptstyle \pm 5 . 7 }$ </td><td> $3 0 . 1 { \pm } 7 . 2 $ </td><td> $5 1 . 9 { \scriptstyle \pm 3 . 4 }$ </td><td> $1 9 . 7 { \scriptstyle \pm 3 . 8 }$ </td><td> $5 5 . 8 { \scriptstyle \pm 7 . 5 }$ </td><td> $6 6 . 9 { \scriptstyle \pm 8 . 0 }$ </td></tr><tr><td>Parallel Pipeline</td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $3 7 . 3 { \scriptstyle \pm 1 . 0 }$ </td><td> $1 6 . 9 { \scriptstyle \pm 2 . 7 }$ </td><td> $3 3 . 0 { \scriptstyle \pm 5 . 6 }$ </td><td> $2 5 . 3 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td></td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $3 5 . 8 { \scriptstyle \pm 1 . 6 }$ </td><td> $1 8 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $2 6 . 7 { \scriptstyle \pm 0 . 0 }$ </td><td> $4 . 6 { \pm } 4 . 2$ </td></tr><tr><td>MAAC</td><td> $7 8 . 4 { \scriptstyle \pm 8 . 9 }$ </td><td> $8 0 . 6 { \scriptstyle \pm 1 0 . 3 }$ </td><td> $4 5 . 1 { \scriptstyle \pm 3 . 9 }$ </td><td> $4 7 . 6 { \scriptstyle \pm 2 . 8 }$ </td><td> $5 9 . 2 { \scriptstyle \pm 6 . 1 }$ </td><td> $5 8 . 9 { \scriptstyle \pm 8 . 0 }$ </td></tr><tr><td> $\mathbf { B C M A A C { - } P }$ </td><td> $8 3 . 7 { \scriptstyle \pm 5 . 3 }$ </td><td> $8 9 . 3 { \scriptstyle \pm 9 . 8 }$ </td><td> $5 0 . 9 { \scriptstyle \pm 0 . 9 }$ </td><td> $4 9 . 4 { \scriptstyle \pm 3 . 5 }$ </td><td> $6 4 . 5 { \scriptstyle \pm 1 1 . 2 }$ </td><td> $6 4 . 6 { \scriptstyle \pm 2 . 6 }$ </td></tr><tr><td> $\mathbf { B C M A A C { - B } }$ </td><td> $8 8 . 9 { \scriptstyle \pm 8 . 5 }$ </td><td> $8 4 . 8 { \scriptstyle \pm 1 3 . 6 }$ </td><td> $5 1 . 6 { \scriptstyle \pm 6 . 5 }$ </td><td> $5 3 . 3 { \scriptstyle \pm 6 . 4 }$ </td><td> $7 0 . 4 { \scriptstyle \pm 6 . 3 }$ </td><td> $7 1 . 2 { \scriptstyle \pm 5 . 0 }$ </td></tr><tr><td> $\mathbf { B C M A A C }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> ${ \bf 9 2 . 8 _ { \pm 3 . 7 } }$ </td><td> $\mathbf { 9 4 . 4 } _ { \pm 9 . 6 }$ </td><td> ${ \bf 5 3 . 2 _ { \pm 5 . 1 } }$ </td><td> ${ \bf 5 8 . 5 { \scriptstyle \pm 2 . 3 } }$ </td><td> $7 7 . 5 _ { \pm 6 . 4 }$ </td><td> ${ \bf 7 3 . 0 _ { \pm 1 . 9 } }$ </td></tr></table>

Tool Collaboration. The tool-using task, one agent inspect the environment, while the other can modify it. Information acquisition and execution must occur in separate turns. The inspecting agent must retrieve and communicate the necessary facts, after which the acting agent can execute. In Stateful Tool, we adapt Apple ToolSandbox [Lu et al., 2025] into a two-agent setting, where a phone environment contains contacts, messages, reminders, and device settings. Two agents are conducting tasks such as retrieving the latest message, disabling cellular data, or rescheduling a reminder, are provided in natural language. The state inspector has readonly access, and the tool executor can modify the environment, send messages, and submit the final answer. Arguments such as phone numbers, contact IDs, and timestamps can be retrieved by the inspector and communicated in a prior turn

![](images/247f7b6cbbb435b892c299d771471f42b000474f04102479bcb4dd76ee016fe8.jpg)  
(a) Bridge Building

![](images/570c28867606a1034ca199c65f9b10de66650033bdd9fe4829510eed7a8f0c1f.jpg)  
(b) Resource Gathering  
Figure 2: Minecraft tasks. (a) In Bridge Building, agents start at the blue beacon endpoints and must construct a white-block bridge under limited vision. Brown wooden pillars are safe, whereas gray stone pillars with wolves are dangerous decoys. (b) In Resource Gathering, agents with complementary tools observe only local regions and collect from trees, stone, and minerals of iron.

before the executor use them. Each agent produces a structured JSON action specifying broadcast facts, information requests, the intended next step, a readiness indicator, at most one tool call, and an optional final answer. Filesystem Wrangling is adapted from the multi-turn GorillaFileSystem task in BFCL V3 [Patil et al., 2025]. It uses a deterministic in-memory filesystem that can be regarded as a miniature MCP server accessed by two clients with restricted permissions. The explorer has read-only tools, including ls, find, cat, and grep, while the editor has write tools such as mkdir, touch, echo, mv, cp, and rm. Since the editor cannot inspect the filesystem, the explorer must communicate all relevant paths. Hard operations such as mv, cp, rm, and rmdir are permitted only when the source path has been found in a previous turn; also, paths discovered in the current turn cannot be consumed immediately by the editor.

![](images/2476ffa42ae20f1158b30de6951544ad4177e710eabeb265abba0e605ade513e.jpg)  
(a) Minecraft | Resource Gathering

![](images/3793482f9808099ee0df9cf2355aa74723d5610ceb9a57cfa0657743618b4208.jpg)  
(b) Web | Cross-Lingual Gathering

![](images/821b11f2b9c3552aa6b5509f5f43bf12e6a4fbf47630749c6843905e0cec2403.jpg)  
(c) Tool | Stateful Tool

![](images/5ed7df6f272549dc14654c1d385475553f93f646b038378be3c7d4c37a702a0b.jpg)  
(d) Minecraft | Bridge Building

![](images/04fe036ba0218afa535081caffe4457a3e56f7fa440df2fd8919fbe85d4d0631.jpg)  
(e) Web | Multi-Source Gathering

![](images/62861adf39818fd1c2188f3c84641040b8e07b3e4b06b67338ca48db4841d38d.jpg)  
(f) Tool | Filesystem Wrangling  
Figure 3: A comparison between BCMAAC (red) and MAAC (blue) across the six tasks. The y-axis denotes the expected return, and the x-axis denotes the environment steps. Results are all averaged over five runs, and shaded regions indicate the standard deviation. Curves are smoothed using a time-weighted exponential moving average.

## 5.2 Results

Table 1 reports the main evaluation results. BCMAAC achieves the best performance on all six tasks. Trained multi-agent methods substantially outperform single-agent AC on tasks that can be ef fectively decomposed across agents, particularly Resource Gathering, Bridge Building, Multi-Source Gathering, and Stateful Tool, indicating that these tasks are not merely tests of individual model capabilities, but require task decomposition, complementary information, and cross-agent collaboration. In contrast, collaboration based solely on prompts (Parallel, Pipeline) fails to consistently improve performance, indicating that simply adding agents without learning effective collaboration methods is insufficient. For standard deviations, BCMAAC exhibits comparable or lower variability than MAAC overall, indicating that it can improve stability; BCMAAC-P and BCMAAC-B show higher standard deviations than full BCMAAC on many tasks, indicating that BCMAAC with two auxiliary objectives not only improve average performance but provide a complementary regularization effect.

Tasks with low parallelism are an exception, the most typical being Filesystem Wrangling: the Editor can only act on paths that the Explorer has already found and communicated in the previous turn, making the two roles more akin to a sequential pipeline than a parallel division of labor. Consequently, a single-agent AC (66.9%) that handles both reading and writing and eliminates the need for crossagent handoffs is inherently strong, even outperforming the decentralized MAAC (58.9%). Even so, BCMAAC can still track the Editor’s readiness status, outperforming the single-agent approach with 73.0%, indicating that what truly matters is effective collaboration.

Compared to MAAC, BCMAAC’s largest gain occurs in Stateful Tool (59.2%→77.5%, +18.3), followed by Resource Gathering (78.4%→92.8%) and Filesystem Wrangling (58.9%→73.0%). Both Stateful Tool and Resource Gathering require maintenance of hidden state, the former must track which facts the executor still lacks and when to submit them, while the latter must remember resource locations and coordinate with agents of different capabilities. This is exactly the setting where belief context helps. While MAAC already performs strongly on Bridge Building, yet BCMAAC still improves from 80.6% to 94.4%, indicating that structured belief remains beneficial for collaboration quality. On Web tasks, BCMAAC also consistently outperforms MAAC. Cross-Lingual

Gathering shows moderate gains (45.1%→53.2%), while Multi-Source Gathering is even greater (47.6%→58.5%), consistent with the task structure, as multi-source question-answering relies more on complementary evidence.

Our ablation studies demonstrate that the two auxiliary signals are complementary. BCMAAC-P and BCMAAC-B perform between MAAC and the full BCMAAC on all tasks, indicating that both partner-action prediction and task-belief supervision can improve upon the original MAAC critic. BCMAAC-B performs better on Resource Gathering, Cross-Lingual Gathering, Multi-Source Gathering, and the two Tool tasks, all of which center on maintaining the state of resources/evidence/tools and the other agents’ readiness; BCMAAC-P performs better on Bridge Building, where coordinating the other agents’ actions and dividing construction tasks is particularly important. The full BCMAAC achieves the best overall performance, demonstrating that combining the modeling of task beliefs with partner actions yields the best collaboration.

## 5.3 Analysis

Figure 3 shows the performance on six tasks. Overall, BCMAAC converges faster, achieves higher rewards, and is more stable than MAAC on most tasks, as the structured belief context improves advantage estimation.

In Filesystem Wrangling, BCMAAC overtakes MAAC and maintains a markedly higher, less volatile reward. This is consistent with better tracking of Editor readiness and more accurate timing of hard actions. In Resource Gathering, BCMAAC likewise improves more smoothly, suggesting that the belief encoder captures resource states, remaining objectives, and the other agents’ capabilities useful for intermediate decisions. In Bridge Building and Stateful Tool, the methods are initially similar, but BCMAAC converges to a higher final reward. Gains are smaller on the Web tasks: BCMAAC maintains a modest advantage in Multi-Source Gathering, while Cross-Lingual Gathering remains noisy and shows improvement at end. The task-belief and partner-action objectives help track retrieved evidence and answer readiness, reducing redundant searches and premature responses.

Results suggest that BCMAAC benefits from latent collaboration state explicitly rather than reconstructing it solely from raw histories. Its centralized critic estimates returns from structured joint beliefs, while the auxiliary objectives provide dense turn-level supervision when communication and exploration produces little immediate reward.

In nearly fully observable tasks, or when short histories are sufficient, the advantage of BCMAAC may be limited. Its effectiveness depends on whether the reduction in estimation error outweighs the information lost through compression. In the partially observable, long-horizon tasks studied in this paper, this trade-off falls clearly on BCMAAC’s side.

## 6 Conclusion

We study decentralized LLM collaboration and introduce BCMAAC. BCMAAC maintains structured belief context during execution, performs centralized value estimation from joint belief context during training, and preserves decentralized decision-making at inference time. BCMAAC consistently outperforms MAAC and improves training stability across Minecraft, Web, and tool-based collaboration tasks. The gains are strongest when agents must track latent state, integrate the information of other agents, coordinate delayed actions, and combine evidence from multiple sources. These results establish structured belief conditioning as an effective mechanism for decentralized LLM collaboration.

Our study has a few limitations. Although our framework is applicable to multiple agents, all six tasks are two-agent tasks, as the characteristics of the tasks we designed are best suited for pairing the two agents into complementary roles. For example, in Filesystem Wrangling, the toolset splits cleanly into read-only commands and write commands, so an explorer with read-only permissions and an editor with write permissions can be paired accordingly.

## References

Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang Wang. HybridQA: A dataset of multi-hop question answering over tabular and textual data. In Findings

of the Association for Computational Linguistics: EMNLP 2020, pages 1026–1036, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.91. URL https://aclanthology.org/2020.findings-emnlp.91/.

Abhishek Das, Théophile Gervet, Joshua Romoff, Dhruv Batra, Devi Parikh, Mike Rabbat, and Joelle Pineau. TarMAC: Targeted multi-agent communication. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 1538–1546. PMLR, 2019.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 11733–11763. PMLR, 2024. URL https://proceedings.mlr.press/v235/ du24e.html.

Lang Feng, Longtao Zheng, Shuo He, Fuxiang Zhang, and Bo An. Dr. MAS: Stable reinforcement learning for multi-agent LLM systems. arXiv preprint arXiv:2602.08847, 2026. doi: 10.48550/ arXiv.2602.08847. URL https://arxiv.org/abs/2602.08847.

Jakob N. Foerster, Ioannis Alexandros Assael, Nando de Freitas, and Shimon Whiteson. Learning to communicate with deep multi-agent reinforcement learning. In Advances in Neural Information Processing Systems, volume 29, 2016.

Daya Guo, Dejian Yang, Haowei Zhang, et al. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645:633–638, 2025. doi: 10.1038/s41586-025-09422-z.

He He, Jordan Boyd-Graber, Kevin Kwok, and Hal Daumé, III. Opponent modeling in deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1804–1813. PMLR, 2016.

Kimi Team, Angang Du, Bofei Gao, et al. Kimi k1.5: Scaling reinforcement learning with LLMs. arXiv preprint arXiv:2501.12599, 2025. doi: 10.48550/arXiv.2501.12599. URL https://arxiv. org/abs/2501.12599.

Landon Kraemer and Bikramjit Banerjee. Multi-agent reinforcement learning as a rehearsal for decentralized planning. Neurocomputing, 190:82–94, 2016. doi: 10.1016/j.neucom.2016.01.031.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024. doi: 10.48550/arXiv.2411.15124. URL https://arxiv.org/abs/2411.15124.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for “mind” exploration of large language model society. In Advances in Neural Information Processing Systems, volume 36, pages 51991–52008, 2023. Main Conference Track.

Junwei Liao, Muning Wen, Jun Wang, and Weinan Zhang. MARFT: Multi-agent reinforcement fine-tuning. arXiv preprint arXiv:2504.16129, 2025. doi: 10.48550/arXiv.2504.16129. URL https://arxiv.org/abs/2504.16129.

Shuo Liu, Tianle Chen, Zeyu Liang, Xueguang Lyu, and Christopher Amato. LLM collaboration with multi-agent reinforcement learning. arXiv preprint arXiv:2508.04652, 2025. doi: 10.48550/ arXiv.2508.04652. URL https://arxiv.org/abs/2508.04652.

Shuo Liu, Tianle Chen, Ryan Amiri, and Christopher Amato. Learning decentralized LLM collaboration with multi-agent actor critic. arXiv preprint arXiv:2601.21972, 2026. doi: 10.48550/arXiv.2601.21972. URL https://arxiv.org/abs/2601.21972.

Shayne Longpre, Yi Lu, and Joachim Daiber. MKQA: A linguistically diverse benchmark for multilingual open domain question answering. Transactions of the Association for Computational Linguistics, 9:1389–1406, 2021. doi: 10.1162/tacl\_a\_00433. URL https://aclanthology. org/2021.tacl-1.82/.

Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. ToolSandbox: A stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1160–1183. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-naacl.65. URL https: //aclanthology.org/2025.findings-naacl.65/.

Frans A. Oliehoek and Christopher Amato. A Concise Introduction to Decentralized POMDPs. SpringerBriefs in Intelligent Systems. Springer, 2016. doi: 10.1007/978-3-319-28929-8.

Georgios Papoudakis, Filippos Christianos, and Stefano V. Albrecht. Agent modelling under partial observability for deep reinforcement learning. In Advances in Neural Information Processing Systems, volume 34, pages 19210–19222, 2021.

Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman Ozdaglar, Kaiqing Zhang, and Joo-Kyung Kim. MAPoRL: Multi-agent post-co-training for collaborative large language models with reinforcement learning. arXiv preprint arXiv:2502.18439, 2025. doi: 10.48550/arXiv.2502.18439. URL https://arxiv.org/abs/2502.18439.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 48371–48392. PMLR, 2025. URL https://proceedings.mlr.press/v267/patil25a.html.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. ChatDev: Communicative agents for software development. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 15174–15186, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.810. URL https://aclanthology.org/2024.acl-long.810/.

Qwen Team. Qwen3-4B-Instruct-2507. Hugging Face model card, 2025. URL https:// huggingface.co/Qwen/Qwen3-4B-Instruct-2507.

Neil Rabinowitz, Frank Perbet, H. Francis Song, Chiyuan Zhang, S. M. Ali Eslami, and Matthew Botvinick. Machine theory of mind. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 4218–4227. PMLR, 2018.

Roberta Raileanu, Emily Denton, Arthur Szlam, and Rob Fergus. Modeling others using oneself in multi-agent reinforcement learning. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 4257–4266. PMLR, 2018.

Sainbayar Sukhbaatar, Arthur Szlam, and Rob Fergus. Learning multiagent communication with backpropagation. In Advances in Neural Information Processing Systems, volume 29, 2016.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 1992. doi: 10.1007/BF00992696.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang. AutoGen: Enabling next-gen LLM applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023. doi: 10.48550/arXiv.2308.08155. URL https://arxiv.org/ abs/2308.08155.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. doi: 10.48550/arXiv.2505.09388. URL https://arxiv.org/abs/2505.09388.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

## A Experimental Settings

## A.1 Tasks and Datasets

We evaluate on six decentralized, two-agent collaboration tasks. Minecraft tasks are based on the game Minecraft, where agents interact through structured actions on the map of the game; Web tasks use datasets. Agents observe different information sources; Tool tasks use deterministic stateful environments (a simulated phone assistant and an in-memory filesystem). Agents operate through structured tool calls.

Table 2: Dataset splits and primary metrics.
<table><tr><td>Task</td><td>Split</td><td>Horizon</td><td>Metric</td></tr><tr><td>Resource Gathering</td><td>7/7</td><td>4</td><td>success</td></tr><tr><td>Bridge Building</td><td>6/3</td><td>4</td><td>success</td></tr><tr><td>Cross-Lingual</td><td>120/30</td><td>2</td><td>match score</td></tr><tr><td>Multi-Source</td><td>30/15</td><td>3</td><td>match score</td></tr><tr><td>Stateful Tool</td><td>25/15</td><td>4</td><td>success</td></tr><tr><td>Filesystem Wrangling</td><td>22/13</td><td>4</td><td>success</td></tr></table>

Resource Gathering. The map of the dataset sizes include $8 \times 8 , 9 \times 9$ , and 10 × 10. Each cell may contain a tuple of (wood, stone, iron) resources. The task goal is gathering enough required instances. Each turn, an agent’s decision is limited to a movement path and at most 2 extraction targets. The view radius is 2 and the maximum path length is 4. Extraction succeeds only when the target cell lies within Manhattan distance 2 of the agent’s final position. The two agents have asymmetric tools: one can harvest wood while the other can harvest stone and iron. Thus, communication is necessary, as an agent may observe resources that are useful to its teammate. Success is defined as achieving the task goal.

Bridge Building. All maps of dataset are 13 × 11 grids. The agent view radius is 3. The task goal is to connects the two anchor regions. The maps differ in things like pillar layout and positions of anchor. Each map contains candidate pillars, including true supports and fake hazardous supports. Agents should discover useful supports through local observation and probing, or communicate probe results and coordinate construction. Each response may probe up to 3 candidate pillars and issue bridge-construction commands within a command budget. Success is defined as achieving the task goal.

Cross-Lingual Gathering. This task is built from a Chinese MKQA subset [Longpre et al., 2021]. Each instance contains paired English and Chinese retrieval corpora and a two-piece support chain. One agent searches English Wikipedia passages, then identify an English page title or bridge clue. The other searches Chinese passages. It is responsible for producing the final answer from target-language evidence. The retriever returns the top 4 passages, each prompt shows up to 6 visible pieces, and communication can include up to 3 evidence items. Success is defined as finding the final answer.

Multi-Source Gathering. This task uses HybridQA examples [Chen et al., 2020], with training instances drawn from the train split and evaluation instances from the development split. The task is balanced across passage-to-table, table-to-passage, and passage-to-table-to-passage reasoning chains. One agent can inspect passage evidence while the other can inspect table-row evidence. Each instance contains a table, a set of passage pieces, and cross-source support annotations. The final answer often depends on combining one table row with one or more passage pieces. The retriever returns the top 4 pieces, communication is limited to 3 evidence item. Success is defined as finding the final answer.

Stateful Tool. This task is adapted from Apple ToolSandbox [Lu et al., 2025] scenarios. The environment simulates a phone holding hidden state (contacts, messages, reminders, device settings, timestamps). The goal is a natural-language user goal (e.g., checking the latest message, turning off cellular data, or rescheduling a reminder to a specific time). The State Inspector can only retrieve information; the Tool Executor can only modify settings/contacts/reminders, send messages, and submit the final answer. Execution tools require the relevant fact to have been known by the Executor before the current turn. Since only the Inspector can retrieve such facts, the team should hand them off across turns in sequence. Task types include setting toggles, message/reminder lookups, sending messages, contact phone/relationship updates, contact removal, and reminder add/modify/remove. Success is defined as the target action being achieved in the world state.

Table 3: Main training hyperparameters. The context learning rate applies to BCMAAC variants.  
Task Training hyperparameters   
Resource Gathering 180 epochs; rollout/batch $8 / 4 ; \gamma = 0 . 9 0 ;$ actor/critic LR $2 . 5 { \times } 1 0 ^ { - 6 } ;$ context LR   
$3 . 0 \times \bar { 1 } 0 ^ { - 5 }$ ; max tokens 160.   
Bridge Building 180 epochs; rollout/batch $8 / 4 ; \gamma = 0 . 9 0 ;$ actor/critic $\mathrm { L R ~ 2 . 5 } { \times } 1 0 ^ { - 6 } ;$ context LR   
$3 . 0 \times \mathrm { { i 0 } ^ { - 5 } } ;$ ; max tokens 160.   
Cross-Lingual 30 epochs; rollout/batch $1 6 / 4 ; \gamma = 0 . 9 5 ;$ actor/critic LR $2 . 0 { \times } 1 0 ^ { - 6 } ;$ context LR   
$1 . 0 \times 1 0 ^ { - 5 }$ ; max tokens 192.   
Multi-Source 30 epochs; rollout/batch 12/2 or $1 6 / 4 ; \gamma = 0 . 9 5 ;$ ; actor/critic LR $1 . 0 { \times } 1 0 ^ { - 6 } ;$ context   
$\mathrm { L R \ i . 0 { \times } 1 0 ^ { - 4 } } ;$ ; max tokens 192.   
Stateful Tool 120 epochs; rollout/batch $1 0 / 2 ;$ actor/critic $\mathrm { L R ~ 1 . 5 } { \times } 1 0 ^ { - 6 } ;$ context $\mathrm { L R ~ 5 . 0 } \times 1 0 ^ { - 5 } ,$   
max tokens 160.   
Filesystem Wrangling 120 epochs; rollout/batch $1 0 / 2 ;$ actor/critic ${ \mathrm { L R ~ } } 1 . 5 { \times } 1 0 ^ { - 6 } ;$ ; context $\mathrm { L R ~ 3 . 0 { \times } 1 0 ^ { - 5 } } ;$   
max tokens 160.

Filesystem Wrangling. This task is a faithful port of the multi-turn GorillaFileSystem tasks in BFCL V3 [Patil et al., 2025]. The filesystem is a deterministic in-memory state machine. It can be viewed as a mini MCP: a shared in-memory file server accessed by two scope-restricted clients. The Explorer has read-only tools (ls, find, cat, grep, etc.), while the Editor has write tools (mkdir, touch, echo, mv, cp, rm, etc.). Since the editor cannot list or search the directory tree, target paths can only be obtained from the Explorer’s communication. Irreversible operations (mv, cp, rm, rmdir) succeed only if the source path was learned by the Editor in a previous turn. Success is defined by a final-state file tree check, or answer matching for the report family.

## A.2 Structured Context and Auxiliary Targets

For BCMAAC, each environment exposes compact structured side information $x _ { i , t }$ constructed only from information visible to agent i at execution time. It never exposes the global answer nor hidden training labels.

The belief targets vary across tasks. In Resource Gathering, they mark the resource cells that match the agent’s tool ability. In Bridge Building, they mark the true supports among pillars. In the Web tasks, they mark the evidence pieces. In the Tool tasks, they mark the required progress slots, such as whether the target action is done.

The partner-action signal is teammate action-type prediction, implemented with a four-class taxonomy in every task. Bridge Building uses place/scout/communicate/no-op. Resource Gathering uses gather/move/communicate/no-op.

Auxiliary labels are used only during training. Notice that at execution time agents keep only their actor and recurrent belief encoder.

## A.3 Training Details

All base model of agents are initialized from Qwen/Qwen3-4B-Instruct-2507 [Yang et al., 2025, Qwen Team, 2025]. We train in bfloat16 and sample actions with temperature 0.5 and $\mathrm { t o p } { - } p = 0 . 6$ Within each task, MAAC and BCMAAC use the same environments, reward definitions, rollout budgets, and evaluation protocols; the actor and critic learning rates are matched across methods, and BCMAAC additionally trains the belief encoder with a separate context learning rate. The belief encoder uses a hidden dimension of 128 and the structured value head a hidden dimension of 256. Table 3 lists the main training hyperparameters.

## A.4 Reward Designs

All rewards are computed by deterministic environment rules after parsing the agents’ JSON actions. We use dense shaping only to expose intermediate progress; the main evaluation metrics remain the task-native success or answer-match scores.

Resource Gathering. Let $p _ { t }$ be normalized collection progress, $c _ { t }$ indicate whether the resource goal is completed, $u _ { t }$ count useful resource communications, $z _ { t }$ and $e _ { t }$ denote move-to-zone and first-zone-entry indicators, and $w _ { t }$ be the number of wasted extraction attempts. The shaped reward is

$$
r _ { t } ^ { \mathrm { R G } } = 8 . 0 \Delta p _ { t } + 7 . 0 c _ { t } + 0 . 1 u _ { t } + 0 . 0 5 z _ { t } + 0 . 1 5 e _ { t } - 0 . 1 w _ { t } .
$$

Bridge Building. Let $g _ { t }$ be the normalized remaining connection gap, $m _ { t }$ the normalized number of newly merged connected components, $y _ { t }$ the normalized newly connected true-support signal, $n _ { t }$ the normalized number of newly unsafe adjacencies to fake supports, and $b _ { t }$ the normalized number of newly placed blocks; $m _ { t } , y _ { t }$ , and $n _ { t }$ are normalized by the corresponding per-map totals. The bridge reward is

$$
\begin{array} { r l } & { r _ { t } ^ { \mathrm { B B } } = 5 . 0 ( g _ { t - 1 } - g _ { t } ) + 1 . 5 m _ { t } + 5 . 0 y _ { t } + 2 . 0 c _ { t } } \\ & { ~ + 2 . 5 \Delta d _ { t } - 1 2 . 0 n _ { t } - 5 . 0 b _ { t } + q _ { t } ^ { \mathrm { l a t e } } , } \end{array}
$$

where $c _ { t }$ indicates terminal connectivity, $\Delta d _ { t }$ is movement progress toward the opposite side, and $q _ { t } ^ { \mathrm { l a t e } }$ is a small late-turn quality term favoring cleaner, more compact, and closer-to-finish bridges.

Web Gathering. Both Web tasks share the same reward template. Let $a _ { t }$ be the answer-match score, $s _ { t }$ support coverage, $u _ { t }$ useful communication, $x _ { t }$ cross-source communication, $h _ { t }$ informed search, $k _ { t }$ support-chain completion, $b _ { t }$ shared-board completion, $e _ { t }$ answer agreement, and $f _ { t }$ a fully supported answer indicator that fires when a correct answer is submitted with a complete support board. The reward is

$$
\begin{array} { r l } & { r _ { t } ^ { \mathrm { W e b } } = 0 . 2 \Delta s _ { t } + 2 . 0 a _ { t } + 0 . 2 u _ { t } + 0 . 0 3 x _ { t } + 0 . 3 h _ { t } } \\ & { ~ + 0 . 6 k _ { t } + 0 . 2 5 b _ { t } + 0 . 4 e _ { t } + 0 . 6 f _ { t } . } \end{array}
$$

Substring answer matches receive partial credit. Explicit penalties for wrong, premature, conflicting, or unsupported answers are set to zero in the reported runs, relying instead on support and agreement rewards to shape answer readiness.

Stateful Tool. Let $\Delta p _ { t }$ be the progress increment (fraction of required progress slots satisfied), $c _ { t }$ indicate terminal success, $u _ { t } , g _ { t }$ useful/grounded communication, $v _ { t }$ the number of valid tool calls, $\operatorname { i n v } _ { t } , \operatorname { e r r } _ { t }$ the numbers of invalid actions / tool errors, $\mathrm { p r e } _ { t }$ premature answering, $\rho _ { t }$ the “ready-butnot-executed/answered” penalty, fail<sub>t</sub> terminal failure, and no $\mathrm { \Delta } ) \mathrm { p } _ { t }$ the number of no-ops. The reward is

$$
\begin{array} { r l } & { r _ { t } ^ { \mathrm { S T } } = 0 . 8 \Delta p _ { t } + 4 . 0 c _ { t } + 0 . 0 8 u _ { t } + 0 . 0 1 g _ { t } + 0 . 0 3 v _ { t } } \\ & { ~ - 0 . 0 8 \mathrm { i n v } _ { t } - 0 . 1 2 \mathrm { e r r } _ { t } - 0 . 1 5 \mathrm { p r e } _ { t } } \\ & { ~ - 0 . 7 5 \rho _ { t } - 0 . 2 5 \mathrm { f a i l } _ { t } - 0 . 0 2 \mathrm { n o o p } _ { t } . } \end{array}
$$

Filesystem Wrangling. Using the same notation, with der $\mathrm { { r } } _ { t }$ the number of irreversible (destructive) operation errors, the reward is

$$
\begin{array} { r l } & { r _ { t } ^ { \mathrm { F W } } = 1 . 3 \Delta p _ { t } + 3 . 3 c _ { t } + 0 . 1 8 u _ { t } + 0 . 0 5 g _ { t } + 0 . 0 6 v _ { t } } \\ & { ~ - 0 . 0 5 \mathrm { i n v } _ { t } - 0 . 0 6 \mathrm { e r r } _ { t } - 0 . 1 2 \mathrm { d e r r } _ { t } - 0 . 1 0 \mathrm { p r e } _ { t } } \\ & { ~ - 0 . 4 5 \rho _ { t } - 0 . 1 5 \mathrm { f a i l } _ { t } - 0 . 0 1 \mathrm { n o o p } _ { t } . } \end{array}
$$

The largest penalty $\rho _ { t } \ ( 0 . 4 5 )$ fires only on terminal failure when all non-terminal prerequisite slots are satisfied but the irreversible action was never executed.

## A.5 Prompt Details

All environments require agents to return exactly one JSON object, which makes the parser deterministic. And it helps convert natural-language decisions into structured actions. We show abbreviated prompt skeletons below.

1 Task setup :   
2 - Current turn : { turn\_idx }/{ max\_turns }   
3 - Goal / progress : { goal\_or\_connection\_state }   
4 - Local state : { position , visible\_objects }   
5 - Received teammate messages : { received\_messages }   
6   
7 Return exactly one JSON object :   
8 {   
9 " comm ": { ... structured facts ... },   
10 " probe ": [[x,z], ...] ,   
11 " cmds ": [... extraction targets or bridge commands ...] ,   
12 " path ": [[ x ,z], ...]   
13 }   
1 Task setup :   
2 - Question : { question }   
3 - Your source : { English / Chinese / passage / table }   
4 - Search history : { search\_history }   
5 - Visible evidence pieces : { visible\_pieces }   
6 - Shared evidence board : { shared\_board }   
7   
8 Return exactly one JSON object :   
9 {   
10 " comm ": {" evidence\_ids ": [" piece\_id "], " notes ": " short grounded clue "},   
11 " search ": " query string ",   
12 " answer ": ""   
13 }   
1 Task setup :   
2 - Role : { State Inspector / Tool Executor | Explorer / Editor }   
3 - Current turn : { turn\_idx }/{ max\_turns }   
4 - User goal : { user\_goal }   
5 - Allowed tools : { allowed\_tools }   
6 - Known facts ( self ): { known\_facts }   
7 - Recent tool results : { tool\_results }   
8 - Received teammate messages : { received\_messages }   
9   
10 Return exactly one JSON object :   
11 {   
12 " comm ": {" facts ": { ... }, " request ": [" needed fact "],   
13 " intent ": " short next - step intent ", " ready ": false },   
14 " tool ": {" name ": " tool\_name ", " args ": { ... }} ,   
15 " answer ": ""   
16 }

## A.6 Evaluation Protocol and Metrics

In the main results we report task-native metrics: success for Minecraft and Tool, and match score for Web. We report the mean and standard deviation across 5 independent training runs, where each run’s score is averaged over several late-training evaluation checkpoints.

Resource Gathering Success. Let R be the set of resource types and $C _ { r } , G _ { r }$ the final collected and required counts. An episode succeeds iff all resource targets are met:

$$
\mathbf { 1 } _ { \mathrm { R G } } = \mathbf { 1 } [ \forall r \in \mathcal { R } , \ C _ { r } \geq G _ { r } ] , \qquad \mathrm { s c o r e } = 1 0 0 \cdot \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \mathbf { 1 } _ { \mathrm { R G } } ^ { ( j ) } .
$$

Bridge Building Success. Let $A _ { S } , A _ { T }$ be the two anchor regions and $\mathcal { F }$ the final traversable set (anchors, pillars, placed bridge blocks). If there is a finite path $( p _ { 0 } , \ldots , p _ { L } )$ with

$$
\begin{array} { c } { p _ { 0 } \in  { \mathcal { A } } _ { S } , \quad p _ { L } \in  { \mathcal { A } } _ { T } , } \\ { p _ { \ell } \in  { \mathcal { F } } ( \ell = 0 , \dots , L ) , \quad \| p _ { \ell + 1 } - p _ { \ell } \| _ { 1 } = 1 ( \ell < L ) , } \end{array}
$$

then $\mathrm { C o n n } _ { \mathcal { F } } ( \mathcal { A } _ { S } , \mathcal { A } _ { T } )$ holds (four-neighbor connectivity). The success indicator is $\begin{array} { r l } { \mathbf { 1 } _ { \mathrm { B B } } } & { { } = } \end{array}$ $\mathbf { 1 } [ \mathrm { C o n n } _ { \mathcal { F } } ( \mathcal { A } _ { S } , \mathcal { A } _ { T } ) ]$ and the score is 100 · $\begin{array} { r } { { \frac { 1 } { N } } \sum _ { j } \mathbf { 1 } _ { \mathrm { B B } } ^ { ( j ) } } \end{array}$

Cross-Lingual Gathering Match Score. Let $n ( \cdot )$ denote answer normalization (lowercasing and removal of superficial formatting differences) and G the gold answer and its annotated aliases. For a submitted answer yˆ and a gold answer or alias $y ,$

$$
s _ { \mathrm { C L } } ( \hat { y } , y ) = \left\{ \begin{array} { l l } { 1 , } & { n ( \hat { y } ) = n ( y ) , } \\ { \alpha , } & { \mathrm { o n e ~ n o r m a l i z e d ~ a n s w e r ~ c o n t a i n s ~ t h e ~ o t h e r , } } \\ { 0 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

where $\alpha = 0 . 7$ is the substring-match credit. Let $\widehat { \mathcal { V } } _ { t }$ be the set of non-empty answers submitted at turn $t ;$ if none is submitted then $m _ { t } ^ { \mathrm { C L } } = 0$ , otherwise

$$
m _ { t } ^ { \mathrm { C L } } = \kappa _ { t } \operatorname* { m a x } _ { \hat { y } \in \widehat { \mathcal { Y } _ { t } } } q _ { \mathrm { C L } } ( \hat { y } ) , \quad q _ { \mathrm { C L } } ( \hat { y } ) = \operatorname* { m a x } _ { y \in \mathcal { G } } s _ { \mathrm { C L } } ( \hat { y } , y ) ,
$$

with $\kappa _ { t } = 1$ unless multiple submitted answers conflict, in which case $\kappa _ { t }$ is a configured downweighting factor. The episode score is $\begin{array} { r } { M _ { \mathrm { C L } } = \sum _ { t = 0 } ^ { H - 1 } m _ { t } ^ { \mathrm { C L } } } \end{array}$ ; since submitting an answer terminates the episode, at most one term is nonzero during evaluation. The reported score is $\begin{array} { r } { 1 0 0 \cdot \frac { 1 } { N } \sum _ { j } M _ { \mathrm { C L } } ^ { ( j ) } } \end{array}$

Multi-Source Gathering Match Score. This uses the same per-turn accumulation protocol but matches against the task’s final annotated answer $y ^ { \star }$ :

$$
s _ { \mathrm { M S } } ( \hat { y } , y ^ { \star } ) = \left\{ \begin{array} { l l } { 1 , } & { n ( \hat { y } ) = n ( y ^ { \star } ) , } \\ { \alpha , } & { n ( \hat { y } ) \mathrm { c o n t a i n s } n ( y ^ { \star } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

with $\alpha = 0 . 7$ and the same down-weighting rule, $\begin{array} { r } { M _ { \mathrm { M S } } = \sum _ { t } m _ { t } ^ { \mathrm { M S } } } \end{array}$ , and reported score 100 · $\begin{array} { r } { \frac { 1 } { N } \sum _ { j } M _ { \mathrm { M S } } ^ { ( j ) } } \end{array}$

Stateful Tool / Filesystem Wrangling Success. Tool tasks use a state-verified binary success. Let don $\mathrm { l e } _ { t }$ indicate whether the target action is achieved in the world state after turn t. An episode succeeds if any turn achieves the target:

$$
{ \bf 1 } _ { \mathrm { T o o l } } = { \bf 1 } [ \exists t \leq H - 1 : ~ \mathrm { d o n e } _ { t } ] , \qquad \mathrm { s c o r e } = 1 0 0 \cdot \frac { 1 } { N } \sum _ { j = 1 } ^ { N } { \bf 1 } _ { \mathrm { T o o l } } ^ { ( j ) } .
$$

## A.7 Baseline Details

All methods are evaluated under the same task instances, environment transitions, reward definitions, parsing rules, and task-native metrics. The differences are the training signal, whether multiple agents are used, and whether the actor/critic receives belief context. Table 4 summarizes the main comparisons. The prompt-only baselines use the same action format as the trained agents; MAAC and BCMAAC use the same rollout budgets, base LLM initialization, actor/critic optimization settings, and environment rewards; the ablations change only the auxiliary supervision attached to the belief encoder.

## A.8 Structured Side Information

Recall that the side information $x _ { i , t }$ is built only from information available to agent i at execution time. The following examples are schematic; the implementation converts these fields into grid channels and scalar vectors.

1 Resource Gathering :   
2 { " grid ": { visibility , visible\_wood / stone / iron , msg\_wood / stone / iron ,   
3 self\_pos , teammate\_pos , extraction\_reach },   
4 " scalars ": { turn\_ratio , normalized\_position , goal [w ,s , i],   
5 collected\_ratio [w,s,i], inbox\_load , completed } }

Table 4: Detailed definitions of baselines and ablation variants.
<table><tr><td>Method</td><td>Execution</td><td>Training signal</td><td>Notes</td></tr><tr><td>Raw Model</td><td>Decentralized LLM agents None. without RL updates.</td><td></td><td>Uses the same prompts and parsers as the trained methods; measures the capability of the initial instruction model.</td></tr><tr><td>AC</td><td>Single-agent actor-critic.</td><td></td><td>Return regression and Tests how far task-specific RL improves a policy-gradient updates single actor without learned multi-agent coordination.</td></tr><tr><td>Parallel</td><td>Prompt-only multi-agent None. parallel execution.</td><td>from the task reward.</td><td>Agents act in parallel from their own prompts and received messages; no policy update or centralized critic.</td></tr><tr><td>Pipeline</td><td>Prompt-only serial execu- None. tion.</td><td></td><td>Agents act in a fixed order, so later agents can condition on earlier outputs; the col- laboration pattern is manually specified.</td></tr><tr><td>MAAC</td><td>Decentralized actors with Actor-critic updates with a The closest RL baseline; actors execute centralized training.</td><td>centralized critic over joint from local histories while the critic esti- histories.</td><td>mates team return during training.</td></tr><tr><td>BCMAAC-P</td><td>belief supervision.</td><td>partner-action / action- behavior / action preferences. preference auxiliary</td><td>BCMAAC without task- Actor-critic updates plus Isolates the effect of modeling teammate</td></tr><tr><td>BCMAAC-B</td><td>action supervision.</td><td>task-belief supervision.</td><td>BCMAAC without partner- Actor-critic updates plus Isolates the effect of grounding the con- text in task-state variables; in Web/Tool tasks answer-readiness is grouped with</td></tr><tr><td>BCMAAC</td><td>tralized actors with a struc- task-belief and partner- parison. tured centralized critic.</td><td>action auxiliary signals.</td><td>task belief. Belief-conditioned decen- Actor-critic updates plus The full method used for the main com-</td></tr></table>

Table 5: Structured side information and task-belief targets used by BCMAAC.
<table><tr><td>Task</td><td>Structured grid</td><td>Scalars</td><td>Task-belief target</td></tr><tr><td>Resource Gathering</td><td>10 spatial channels over the map.</td><td>11 normalized scalar features.</td><td>Non-empty resource cells relevant to the agent&#x27;s tool role.</td></tr><tr><td>Bridge Building</td><td>12 spatial channels over the map.</td><td>features.</td><td>8 normalized scalar True support pillars among candidate pillars.</td></tr><tr><td>ing</td><td>Cross-Lingual Gather- 8 piece channels over the evidence list.</td><td>20 normalized scalar features.</td><td>Evidence pieces in the annotated sup- port chain.</td></tr><tr><td>ing</td><td>Multi-Source Gather- 8 piece channels over table and passage pieces.</td><td>20 normalized scalar features.</td><td>Evidence pieces in the annotated cross- source support chain.</td></tr><tr><td>Stateful Tool</td><td>8 channels over 6 belief slots.</td><td>16 normalized scalars.</td><td>Required progress slots (whether the executor holds the key fact, whether the setting is ready, whether the target</td></tr><tr><td></td><td>Filesystem Wrangling 8 channels over 6 belief slots.</td><td>16 normalized scalars.</td><td>action is done, etc.). Required progress slots (whether the target is located, whether the executor knows the path, whether the destination is ready, whether the target action is done, etc.).</td></tr></table>

```python
1 Stateful Tool / Filesystem Wrangling ( slot grid ) :
2 { " grid ": { self_knows_slot , inbox_mentions_slot , last_tool_relates_to_slot
,
3 slot_in_read_set , slot_in_write_set , slot_required ,
4 success_flag , turn_fraction }, # 8 channels over belief slots
5 " scalars ": { turn_ratio , role_onehot , per_fact_known_flags ,
6 inbox_size , tool_history_len , success , ...} } # 16- dim
```

Table 5 summarizes the tensor dimensions and task-belief targets.

## A.9 Auxiliary Label Generation

The labeling rule $ { \Gamma } _ { \xi } ( \tau )$ produces all auxiliary labels after each rollout, reading the completed trajectory together with the environment’s internal state. The task-belief loss is a masked binary cross-entropy,

$$
\mathcal { L } _ { b } = \mathrm { B C E W i t h L o g i t s } \Big ( \hat { \mathbf { b } } _ { i , t } [ \omega _ { i , t } ] , ~ \mathbf { b } _ { i , t } ^ { * } [ \omega _ { i , t } ] \Big ) ,
$$

where $\mathbf { b } _ { i , t } ^ { * }$ marks the ground-truth entries listed in Table $5 ,$ and the mask $\omega _ { i , t }$ restricts supervision to unresolved or required entries. For the Web and Tool tasks, an answer-readiness label is grouped with task belief: the target is 1 at the first successful turn and 0 otherwise. The partner-action loss is a cross-entropy over a task-specific four-class action taxonomy,

$$
\mathcal { L } _ { p } = \mathrm { C E } ( \hat { \mathbf { p } } _ { i , t } , a _ { - i , t } ^ { * } ) ,
$$

where $a _ { - i , t } ^ { * }$ is the parsed teammate action type; Minecraft tasks add action-preference supervision over the comm, probe, cmds, and path fields. The full auxiliary objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a u x } } = \lambda _ { b } \mathcal { L } _ { b } + \lambda _ { p } \mathcal { L } _ { p } + \lambda _ { r } \mathcal { L } _ { r } + \lambda _ { a } \mathcal { L } _ { a } , } \end{array}
$$

where $\mathcal { L } _ { r }$ is the answer-readiness loss and ${ \mathcal { L } } _ { a }$ the action-preference loss; in the main text, $\mathcal { L } _ { r }$ is grouped into the task-belief term and $\mathcal { L } _ { a }$ into the partner-action term. BCMAAC-B sets $\lambda _ { p }$ and $\lambda _ { a }$ to zero; BCMAAC-P sets $\lambda _ { b }$ and $\lambda _ { r }$ to zero.

## B Proofs

$$
\begin{array} { r l } & { V ^ { \pi } ( \mathcal { F } _ { t } ) = \mathbb { E } _ { \pi } [ G _ { t } \mid \mathcal { F } _ { t } ] } \\ & { \quad \quad \quad = \mathbb { E } _ { \pi } [ G _ { t } \mid \mathbf { c } _ { t } ] = V ^ { \pi } ( \mathbf { c } _ { t } ) . } \end{array}
$$

Proof of Proposition 4.2. Because $\mathbf { c } _ { t }$ is determined by information in $\mathcal { F } _ { t } .$ , conditioning on $\mathcal { F } _ { t }$ also determines $\mathbf { c } _ { t } .$ . Hence