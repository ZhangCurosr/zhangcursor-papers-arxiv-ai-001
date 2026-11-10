# A Modular Perspective for Modeling MARL Systems

Tiziano Maisonhaute

tiziano.maisonhaute@lirmm.fr

LIRMM, Universit´e de Montpellier, CNRS, Montpellier, France CIRAD, UPR Recyclage et Risque, F-97743 Saint-Denis, La R´eunion, France Recyclage et Risque, CIRAD, Universit´e de Montpellier, Montpellier, France

Jean-Christophe Souli´e jean-christophe.soulie@cirad.fr CIRAD, UPR Recyclage et Risque, F-97743 Saint-Denis, La R´eunion, France Recyclage et Risque, CIRAD, Universit´e de Montpellier, Montpellier, France

Fabien Michel

fmichel@lirmm.fr

LIRMM, Universit´e de Montpellier, CNRS, Montpellier, France

## Abstract

The growing complexity of Multi-Agent Reinforcement Learning (MARL) has fragmented the literature, obscuring underlying architectural choices behind diverse implementations. This fragmentation makes it dificult to evaluate critical modeling dimensions in isolation, identify how components afect the resulting system dynamics, and compare, reuse, or combine contributions across the literature.

This paper proposes a modular perspective for modeling, analyzing, and implementing MARL systems. We decompose them into eight functional modules: agents, environments, learning, reward modeling, training and execution modes, communication, modeling of other agents, and evaluation criteria. For each module, we clarify its role and review representative approaches.

We instantiate this decomposition in MARLKIT, a framework whose modules are explicit and interchangeable. MARLKIT demonstrates how the proposed perspective sup ports transparent experimental assumptions and the controlled variation and combination of MARL mechanisms. Experiments illustrate its ability to prototype, compare, reuse and combine modules while making diferences between configurations explicit.

Overall, the paper provides a structured perspective on MARL modeling and implementation, ofering a common language for making modeling choices explicit, supporting clearer descriptions of MARL approaches, and supporting more controlled experimentation through the systematic isolation and combination of MARL mechanisms.

Keywords: Multi-Agent Reinforcement Learning, Reinforcement Learning, Multi-Agent Systems, Machine Learning, Modular Design

## 1 Introduction

Reinforcement Learning (RL) has proven to be a flexible Machine Learning paradigm for producing autonomous agent behaviors across a wide range of, possibly very complex, tasks ranging from game theory problems (Moody et al., 2004) to autonomous driving (Elallid et al., 2022). The associated learning process relies on making the agent interact with its environment so that a reward is obtained with respect to the taken action and the new environment state, thus enabling the agent to evolve its behavior accordingly. More fundamentally, using RL presupposes the reward hypothesis, stating that ”Goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)” (Sutton and Barto, 2018).

Today there exists a number of mature libraries allowing to develop RL-based experiments such as Gymnasium<sup>1</sup> (Towers et al., 2025), RLlib (Liang et al., 2018), Stable-Baselines3 (Rafin et al., 2021) or TorchRL (Bou et al., 2024). These frameworks are relevant for the RL community because they are backed by decades of RL research which has produced a common background and vocabulary for RL practitioners. Moreover, they are usually designed having in mind the MDP framework, which provides a useful representation of key RL modeling elements, which are themselves part of the RL community common background.

Multi-Agent Reinforcement Learning (MARL) extends RL to settings where multiple learning agents interact within a shared environment. While many RL core concepts still hold in a multi-agent setting, it turns out that crucial diferences exist, new challenges arise and existing ones could be exacerbated. Most especially, the reward hypothesis requires extension in MARL because making the whole Multi-Agent System (MAS) achieve the desired task cannot be directly reduced to the maximization of a single agent’s expected cumulative reward. Furthermore, having all the agents maximize individual rewards may not lead to an eficient collective outcome. In fact, one main challenge in MARL is that the outcome of an agent’s action depends on the actions of others, which makes simultaneous learning quite challenging, since computing the reward of a specific action is not trivial in many cases.

So, designing a MARL experiment is thus more dificult and contains many modeling challenges and pitfalls. Therefore, a significant part of the MARL research still seeks for useful generalizable modeling principles, which could shape the field in a way that would both foster comparison among proposed approaches and ease the capitalization of relevant contributions.

From this perspective, this paper makes the hypothesis that, while it is natural that the MARL field has been built upon RL-based foundations, certain conceptual gaps in MARL research stem from this strong inheritance. This observation is also consistent with recent analyses of MARL experimental practices. We agree with the observations made by Gorsane et al. (2022), In MARL, high variance across implementations and inconsistent comparative baselines make it dificult to test and compare the results of diferent algorithms, thereby afecting the quality of reported results. In response to this problem, Bettini et al. (2024) introduce BenchMARL. The objective of BenchMARL is to improve reproducibility and comparability in MARL. BenchMARL allows users to run experiments on the same environment with diferent algorithms and compare the results. This is an important step towards best practices in MARL experiments.

However, we believe that this issue goes beyond the comparison of experimental results. In our view, the problem also lies in how MARL approaches are defined and described. Rather than treating an approach as a monolithic entity, we argue that it should be decomposed into a set of distinct modules. Specifically, reviewing the MARL literature reveals that it is often dificult to identify how a particular contribution addresses specific MARL aspects (for example communication among agents, models of other agents, reward modeling, etc.). This is precisely because such aspects have not yet been made suficiently explicit in the reference literature. In particular, papers frequently present contributions at the level of complete algorithms, rather than being explicitly associated with the specific MARL aspects they modify, extend, or address. Consequently, authors frequently fail to clearly distinguish which MARL aspects their solution afects and how it may consequently modify others. As a result, presentation and terminology vary significantly across MARL articles.

Several popular MARL libraries exist, including PettingZoo (Terry et al., 2021), Open-Spiel (Lanctot et al., 2019), RLlib (Liang et al., 2018), MARLlib (Hu et al., 2023) and EPy-MARL (Papoudakis et al., 2021b), each with its own scope. However, their architectures are generally not organized around MAS-specific responsibilities. Agents are rarely represented as independent components coordinating their own responsibilities, while mechanisms such as communication and reward modeling are generally embedded in other components (environments, policies, etc.). This paper argues that the MARL community would benefit from adopting an explicit Agent-Oriented Software Engineering (AOSE) perspective (Winikof and Padgham, 2013), together with a clearer decomposition of important MAS-specific responsibilities into first-class components.

Following this view, the main contribution of this paper is proposing an explicit decomposition of the major concerns involved in modeling and implementing a MARL experiment. The underlying goal is threefold: (1) to promote a comprehensive overview of MARL that improves the understanding of each key element in the field; (2) to use this decomposition to classify the main approaches proposed with respect to each identified concern; and (3) to demonstrate that such an approach can be used to easily implement MARL frameworks, enabling the prototyping of diferent methods through the eficient modularization of core MARL aspects.

To this end, this research proposes to decompose MARL systems into eight core modules: (1) the agent, (2) the environment, (3) the learning module, (4) the reward model, (5) the training and execution paradigm (is the training and/or execution centralized or decentralized), (6) the communication mechanisms, (7) the modeling of other agents, and (8) the evaluation criteria. Explicitly defining the role of each dimension may facilitate a more precise classification and comparison of MARL approaches.

To support this perspective, we implemented a MARL framework based on this decomposition as a proof of concept, namely MARLKIT<sup>2</sup>.

This article is organized as follows. Section 1 presents the main concepts of RL that serve as a basis for MARL concepts. Section 3 provides the fundamentals of MARL. It is structured in the same way as Section 1 and shows the progression from RL to MARL. Section 4 constitutes the main contribution of this article. It partitions MARL implementation into eight core modules presented above. Finally, Section 6 presents MARLKIT, and more specifically how the eight core modules are defined in MARLKIT.

## 2 Reinforcement Learning Backgrounds

This section gives the fundamentals of Reinforcement Learning (RL) and more specifically highlights the key aspects that are necessary to understand the specifities and new challenges of MARL.

## 2.1 Reinforcement Learning Principle

RL enables an autonomous agent (for example robot, conversational agent, video game character, etc.), to learn behaviors adapted to a dedicated task through trial-and-error interaction with an environment, with the objective of maximizing cumulative rewards (Sutton and Barto, 2018).

The learning process thus consists of a loop, as illustrated in Figure 1. First, the agent observes the state of the environment, then performs an action. Once this action has been performed, the environment computes (1) its next state, according to its dynamics, and (2) a reward for the agent. Finally, the agent updates its policy based on the received reward, using a specific learning algorithm.

![](images/a098a8831c450a25289fbdc34e9fc1fc2ea282841de02e042affbf3a63d2ee73.jpg)  
Figure 1: Reinforcement Learning Scheme

## 2.2 Markov Decision Process

As a mathematical model for sequential decision-making in stochastic environments $( \mathrm { P u t } -$ erman, 2014), the Markov Decision Process (MDP) is the formalism of choice in $\mathrm { R L , }$ as it provides a complete framework for modeling agent-environment interactions and defines many of the concepts and terms used in the field.

An MDP is defined by a tuple (S, A, T, R) where S is the state space, $\mathcal { A }$ is the action space, $T ( s ^ { \prime } | s , a )$ is the transition function, which gives the probability of transitioning from a state s to a state $s ^ { \prime }$ after an action $^ { a , }$ and $R ( s , a , s ^ { \prime } )$ is the reward function, which computes the reward received by an agent when it performs an action a leading to the transition from s to $s ^ { \prime }$

The interaction within an MDP is described as a sequence of steps. At each step, the agent observes the current state $s \in S$ , takes an action $a \in A$ , and the environment transitions to a new state $s ^ { \prime } \in \mathcal { S }$ based on the state-transition function T. The agent then receives a reward $r = R ( s , a , s ^ { \prime } ) $ determined by the action and the resulting state. Furthermore, the Markov property is generally assumed, meaning that the current state s contains all the information required to determine the next state; past agent-environment interactions are not needed.

## 2.3 Environment Dynamics

The environment refers to the dynamical system in which the agent operates. The state $s \in S$ can be finite $( \cal S = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ where n is the number of states in S) or infinite (for example when states are continuous variables such as a position).

The modeling of the environment requires defining the two core functions defined in the previous subsection: the Transition function $T ( s ^ { \prime } | s , a )$ and the Reward function $R ( s , a , s ^ { \prime } )$

Dynamics are deterministic if $T \in \{ 0 , 1 \}$ , meaning the next state is fully determined by the current state and action. Conversely, dynamics are stochastic if $T$ takes values in [0, 1], implying the next state is drawn from a probability distribution over possible outcomes.

## 2.4 Reward Modeling

A reward is the immediate feedback the environment gives after an action, indicating how good that outcome is with respect to the task’s objectives. To guide the agent efectively, the reward function must accurately reflect these goals.

An RL process is typically structured into episodes, though continuous tasks may consist of a single, possibly not ending, episode. An episode is a sequence of states, actions, and rewards, starting from an initial state and terminating upon reaching a terminal state or exceeding a step limit. Through experience, the agent updates its policy to improve performance in subsequent episodes and possibly in the current episode.

Following the reward hypothesis, the primary goal of an RL agent is to maximize cumulative rewards over time within an episode, rather than just the immediate reward. Thus, a key concept is $G _ { t } ,$ the discounted return which is defined as the sum of discounted rewards received from time t to the end of the episode. A parameter called the discount rate γ is used to balance the importance of immediate versus future rewards.

Defining an efective reward function is challenging, as it fundamentally shapes the agent’s behavior. Poorly defined reward functions can lead to unintended behaviors or significantly slow down the learning process.

## 2.5 Agent

In the scope of RL, the term agent refers to the learning entity that actually operates in the environment by taking actions. Assuming that an action space A is already defined, specifying an RL agent requires defining at least the following:

• its policy: a function that maps a state $s \in S$ to an action $a \in { \mathcal { A } } .$

• its learning algorithm: a set of rules used to update its policy. The updates are based on experiences (obtained rewards).

At the start of the learning process, the agent’s policy is initialized, often by randomly mapping states to actions. The learning algorithm then iteratively updates this policy to improve performance on the task. Specifically, the algorithm uses the obtained rewards as feedback signals to guide these updates.

## 2.6 Learning

## 2.6.1 Policy

A policy<sup>3</sup> is a mapping from states to probabilities of taking actions:

$$
\begin{array} { r } { \pi : { \mathcal { S } } \times { \mathcal { A } } \to [ 0 , 1 ] } \\ { \pi ( s , a ) \mapsto P ( a | s ) } \end{array}\tag{1}
$$

Where S is the state space and A is the action space. $\pi ( s , a )$ is the probability of taking action a in state s.

A policy is either deterministic or stochastic. A policy is deterministic if the probability of selecting action a in state s is strictly 0 or 1. In this case, the policy is thus a direct mapping:

$$
\begin{array} { r } { \pi : { \cal S }  { \cal A } } \\ { \pi ( s ) \mapsto a } \end{array}\tag{2}
$$

In RL, policies are implemented either as tabular methods or via function approximation. Tabular methods store values in a table indexed by states, actions, or both, typically maintaining a value function. The table is updated by the learning algorithm during the process.

However, tabular approaches become infeasible when the state space is large or continuous. Furthermore, tabular methods treat similar states as distinct, preventing generalization. Function approximation addresses these limitations by learning a parameterized function to approximate the policy or value function. This approximation uses a large number of parameters θ, which are updated during the learning process. Most of the time, a neural network is used to approximate the function and the parameters θ are the network’s weights. Both methods are detailed by Sutton and Barto (2018).

With respect to their underlying mechanisms, there are three main types of policies an agent can apply to select actions.

Value-based Value-based algorithms use a value function to estimate the expected return. If this value function accurately estimates expected returns, the agent can identify states or actions associated with higher returns. There are two primary types of value functions: the state-value function, which evaluates states, and the action-value function, which evaluates state-action pairs.

Policy-based Policy-based algorithms directly optimize policies without explicitly using a value function. A policy is typically represented by the function π<sub>θ</sub>(s, a), where θ is the vector of policy parameters. During training, θ is updated to maximize the expected return.

A common approach is to model the policy as a neural network, where the network’s weights correspond to the parameters θ. The network takes a state representation as input and outputs either the probability distribution over actions (for a stochastic policy) or the specific action to take (for a deterministic policy).

Actor-Critic Actor-Critic algorithms combine value-based and policy-based approaches by executing two processes in parallel: the actor and the critic (Konda and Tsitsiklis, 1999).

The actor represents the policy-based component; it selects actions based on the current policy and the environment’s state.

The critic represents the value-based component; it estimates a value function to evaluate the states and actions selected by the actor. By providing feedback on the quality of these decisions, the critic guides the actor’s policy updates. The value function estimated by the critic may be the state-value function V, the action-value function Q, or the advantage function A.

## 2.6.2 Learning Algorithm

To support a modular description of reinforcement learning systems, we consider that a learning algorithm specifically denotes the process responsible for updating the agent’s policy from available data. And therefore, it is not responsible for data generation or data collection, which are handled by other components of the system.

In reinforcement learning, the data used by the learning algorithm typically comes from the agent’s interaction with the environment. It is commonly represented as experiences, where each experience is a tuple containing the state (or observation), the action taken, the reward received, and possibly the next state (or observation). Using these experiences, the learning algorithm updates the agent’s policy, value function, or other learned elements involved in decision-making.

Many learning algorithms have been proposed in the literature. For a broader presentation of learning algorithms, we refer the reader to the review by Shakya et al. (2023).

Each algorithm has its own learning parameters. These learning parameters depends on the learning algorithm, however, most use a learning rate α to update the policy. The learning rate determines the magnitude of the policy update based on new experiences. The greater α is, the larger the update.

A widespread criterion<sup>4</sup> for the learning rate is to have it decrease over time, satisfying the following two conditions (Sutton and Barto, 2018):

$$
\sum _ { t = 1 } ^ { \infty } \alpha _ { t } = \infty \mathrm { : } \sum _ { t = 1 } ^ { \infty } \alpha _ { t } ^ { 2 } < \infty\tag{3}
$$

The first condition ensures that the learning rate does not decrease too rapidly, guarantee ing that the cumulative updates are suficient to overcome a poor initial policy or initial conditions. The second condition ensures that the learning rate decreases fast enough to allow convergence.

## 2.7 Challenges

Implementing an RL solution presents significant challenges. The dificulty depends on the use case, but there are main challenges that occur in many RL implementations. This section identifies these main challenges, analyzing their causes and the problems they pose in the learning process. These challenges persist in MARL, and are often exacerbated by the multi-agent paradigm.

The primary dificulties identified in this work are: (1) non-stationarity, (2) partial observability, (3) temporal credit assignment, and (4) the curse of dimensionality.

## 2.7.1 Non-stationarity

RL algorithms typically assume a stationary environment, meaning the state-transition and reward functions are time-invariant and constant over time.

Stationarity simplifies learning. Agents learn through trial and error by associating actions in specific states with received rewards. If the environment is stationary, the reward function satisfies:

$$
s _ { t 1 } , a _ { t 1 } = s _ { t 2 } , a _ { t 2 } \Rightarrow R ( s _ { t 1 } , a _ { t 1 } ) = R ( s _ { t 2 } , a _ { t 2 } )\tag{4}
$$

Where t1 and t2 are two diferent time step. The same holds for the state-transition function.

Consequently, the expected reward for a specific action in a specific state remains constant, facilitating stable learning. Conversely, in a completely chaotic or highly nonstationary environment, past experiences become irrelevant for evaluating current actions, making learning impossible. However, an RL environment can be non-stationary while still allowing an agent to learn (Padakandla et al., 2020), though this presents a significant challenge that must be explicitly addressed.

## 2.7.2 Partial Observability

Agents often lack access to the full state of the environment, receiving only a partial observation instead. In such cases, the agent must make decisions based on incomplete information, which may not fully capture the environment’s dynamics. Partially observable environments are common in real-world applications, such as robotics or autonomous driving, where sensors may not capture all relevant information.

Learning is significantly more challenging in these settings. Consider an agent operating in an environment where two distinct states are indistinguishable from its perspective. Since the agent selects actions based on observations, it may behave identically in both states, even if the optimal actions difer. Consequently, even with perfect knowledge of the optimal policy, the agent cannot act optimally because it cannot determine its current state.

To address this issue, the Partially Observable Markov Decision Process (POMDP) framework is widely used. In this framework, the agent maintains a belief state—a prob ability distribution over possible states—or leverages memory mechanisms to track the history of observations and actions (Igl et al., 2018).

## 2.7.3 Temporal Credit Assignment

The temporal credit assignment problem refers to the dificulty of determining which specific actions are responsible for a received reward (Sutton, 1984; Ausin et al., 2021). Efec tively, an agent’s actions afect not only the immediate reward but also future rewards.

As discussed in Section 2.4, an action should not be evaluated solely based on the immediate reward it yields, as it can have long-term consequences by altering the environment’s state and influencing future outcomes. Consequently, when an agent receives a reward, it is often unclear whether that reward is attributable to the most recent action or a sequence of previous actions.

Consider a game of chess as a simple example. The agent receives a reward only at the end of the game: +1 for a win, −1 for a loss, and 0 for a draw. If the agent wins, identifying the specific cause of the reward is challenging; it is certainly not attributable solely to the final move, but rather to the cumulative efect of the entire sequence of moves.

## 2.7.4 Curse of Dimensionality

The size of the state-action space is given by |S|×|A|. The dificulty of exploring this space grows exponentially with the number of states and actions, a phenomenon known as the curse of dimensionality (Wang et al., 2021).

This poses a significant challenge for tabular RL solutions 2.6.1, which rely on maintaining a value table for every state-action pair. Furthermore, high dimensionality can severely slow down the learning process due to the vast exploration requirements. In some high-dimensional spaces, it may be impossible to explore all state-action pairs (Korkmaz, 2024).

## 3 Multi-Agent Reinforcement Learning

The transition from single-agent to multi-agent learning in a shared environment fundamentally alters the concepts discussed previously. This section outlines the new setting, organized as follows: (1) an introduction to Multi-Agent Systems (MAS) and the revised learning loop; (2) the definition of Stochastic Games as a generalization of MDPs for MAS; (3) an analysis of dynamic shifts; (4) a discussion on why simple reward maximization is no longer a suficient objective; (5) an examination of agent and policy adaptations in MARL; and (6) the identification of key challenges, which represent exacerbated versions of those found in single-agent RL.

## 3.1 From RL to MARL

## 3.1.1 Multi-Agent Systems

Multi-Agent Systems (MAS) comprise several autonomous agents that interact with their environment and with one another. These interactions can vary in intensity (from loose to tight coupling) and nature (cooperative, competitive, or mixed), and may involve explicit communication (Ferber, 1999).

MAS ofer several practical advantages for solving complex problems (Balaji and Srinivasan, 2010): (1) Parallelism: Multiple agents allow for the parallelization of the problemsolving process; (2) Robustness: The system is more resilient to failures, as the malfunction of a single agent does not necessarily compromise the entire system; (3) Flexibility: Agents can be dynamically added, removed, or modified to adapt to changing requirements; (4) Decentralization: MAS do not require a central entity to coordinate all tasks, enabling distributed control.

Beyond problem-solving, MAS are also widely used to model and simulate complex systems (Michel et al., 2009), particularly those involving multiple interacting entities. Common applications include economics (Guessoum et al., 2004), crowd simulation (Ulicny and Thalmann, 2002), or stakeholder behavior analysis (Arnejo et al., 2025; Zhang et al., 2026).

## 3.1.2 MARL Learning Loop

In contrast to RL, Multi-Agent Reinforcement Learning (MARL) enables multiple agents to learn and interact within a shared environment. Consequently, the MARL learning loop is illustrated in Figure 2. In this setting, state transitions and rewards depend on the joint action of all agents. In most MARL formulations, agents select their actions simultaneously; therefore, they cannot condition their decisions on the concurrent actions of others.

![](images/3e6749b7bdb55034ade4a4e76d38b002b9982a4700c32c928ddc5f11854f8b87.jpg)  
Figure 2: Multi-Agent Reinforcement Learning Scheme

Since MARL operates within a Multi-Agent System (MAS), agents can interact directly with one another. This interaction enables the emergence of new mechanisms, such as coordination and communication. However, these interactions also increase system complexity, often making the learning process significantly more dificult than in single-agent settings.

## 3.2 Stochastic Games

Stochastic games (SG), introduced by Shapley (1953), generalize multi-agent MDPs. They are also known as Markov games.

Just as some RL systems do not strictly adhere to the MDP formalism, MARL problems do not always strictly adhere to the SG formalism. However, SG terminology and concepts are widely used in MARL, particularly in MARL systems where agents act simultaneously.

An SG is defined by five elements $( N , S , \{ A _ { i } \} , T , \{ R _ { i } \} )$ where N is the number of agents, S is the state space, $\mathbf { \mathcal { A } } _ { i }$ is the action space for an agent $i , T ( s ^ { \prime } | s , a _ { 1 } , \ldots , a _ { N } )$ is the transition function, and $R _ { i } ( s , a _ { 1 } , \ldots , a _ { N } , s ^ { \prime } )$ is the reward function for an agent i. Transitions and rewards depend on the joint action of all agents. The interaction within an SG is similar to that of an MDP.

## 3.3 Environment and Dynamics

Except in extensive-form games<sup>5</sup> modelisation (Kuhn, 1953), where agents take actions in turns, most MARL settings follow the stochastic game framework, where agents select actions simultaneously. Therefore, the main diference between the environment in RL and MARL is that in MARL, the environment is influenced by the joint actions of all agents, not each action individually. That will change the definition domain of the dynamics functions.

For describing the environment dynamics in MARL, the state transition function and the reward function are still used. However, since agents select actions simultaneously, these functions do not have the same form, now they take as input the joint actions of all agents.

The new definition domains of the two functions are as follows:

$$
T ( s ^ { \prime } | s , \mathbf { a } ^ { * } ) : \mathcal { S } \times \mathcal { A } ^ { * } \times \mathcal { S }  [ 0 , 1 ]\tag{5}
$$

$$
R ( s , \mathbf { a } ^ { * } , s ^ { \prime } ) : \mathcal { S } \times \mathcal { A } ^ { * } \times \mathcal { S } \to \mathbb { R }\tag{6}
$$

Where $\mathcal { A } ^ { * } = \mathcal { A } _ { 1 } \times \mathcal { A } _ { 2 } \times \ldots \times \mathcal { A } _ { n }$ is the joint action space of all agents.

When moving from single-agent RL to MARL, the dynamics function must be adapted. In MARL, transitions depend on joint actions. Because agents act concurrently, the outcomes cannot be computed based on individual actions without considering others’.

## 3.4 From Rewards to System Goal

As mentioned in Section 1, the objective in a single-agent system is to maximize its cumulative reward. Designing the reward function is crucial in RL, as it fully determines the agent’s objective and how it will learn from its interactions with the environment. The evaluation criterion in reinforcement learning is well defined and is based on reward maximization. In a multi-agent system, the question is more nuanced because the goal has to be defined at the system level (Zhang et al., 2021). Increasing one agent’s reward may decrease another’s. The measurement of learning quality must be done at the system level, and cannot be related to the performance of only one entity. What has to be maximize to achieve the system’s goal?

The modeling of the reward functions will determine the dynamics that emerge in the system. To evaluate if the resulting dynamics is good, one must choose evaluation criteria. Bu¸soniu et al. (2010) identify two dimensions for the system objectives: the search for stability and the search for adaptability. Stability is a convergence criterion that aims to achieve a stationary policy. Adaptability is a performance criterion that aims at having agents adapt their strategy to that of others. The choice of an appropriate solution concept depends on the use case and the desired outcome, some of them are presented in Section 4.9.

## 3.5 Agent

In MARL, agents may be able to interact with each other, depending on the setting. This enable new mechanisms such as communication, model predictive of others, centralisation etc.

This new mechanisms can be used to help the agents to be more eficient. Communication allows agents to share information and increase the potential of coordination (Zhu et al., 2024). Predictive models of others can be used to predict the actions and therefore adapt their own action accordingly (Albrecht and Stone, 2018). Centralisation can be used to share information, build a global model to better evaluate actions, etc (Sharma et al., 2021).

However, these interactions can also increase system complexity and introduce or worsen new challenges, they are presented in Section 3.6. Agents needs to face these new challenges using the new tools at their disposal.

## 3.5.1 Self-Play vs Mixed-Play

In MARL there are two distinct approaches: self-play and mixed-play (Albrecht et al., 2024). In self-play, agents learn by playing against others that use the same learning algorithm (Bowling and Veloso, 2002; Prasad et al., 2015). This is the most common approach in MARL papers. Self-play simplifies the learning process, there is no need to design multiple algorithms. Self-play is also sometimes used in a more restricted definition, where agents learn by playing with copies of themselves (Tesauro, 1995; Silver et al., 2016). In this case agents share the same policy. Albrecht et al. (2024) call them algorithm self-play and policy self-play respectively.

In mixed-play agents learn by playing against others that use diferent learning algorithms. This approach is less common in MARL papers, but it may be a necessity when agents are heterogeneous. In zero-sum games, mixed-play allows one to compare the per formance of diferent algorithms (Lee et al., 2022).

Mixed-Play tends to exacerbate more the non-stationarity problem (Albrecht et al., 2024).

## 3.6 Challenges

The major dificulties identified here are based on two recent articles by Canese et al.   
(2021); Zhang et al. (2021) which provide a synthesis of MARL.

## 3.6.1 Non-stationarity

As mentioned in Section 2.7.1, learning in a stationary environment makes learning easier since the dynamics do not change over time. Learning in a non-stationary environment, where the dynamics can change, is more challenging (Gronauer and Diepold, 2022; Hernandez-Leal et al., 2017; Albrecht et al., 2024).

In MARL, the environment may be stationary in principle. However, because other agents are also learning and updating their policies, the dynamics perceived by any individual agent become non-stationary (Albrecht et al., 2024; Zhang et al., 2021; Gronauer and Diepold, 2022). Since agents do not have access to the policies of others, two states may appear identical to one agent while in reality the underlying dynamics can difer significantly due to changing strategies of others. In other words, for the same observed state of the environment, taking the same action may yield diferent distributions of rewards and state transitions. Therefore the optimal action to take may vary. This is why the nonstationarity is also called the moving target problem (Albrecht et al., 2024; Hernandez-Leal et al., 2017; Tuyls and Weiss, 2012; Gronauer and Diepold, 2022). The less agents know about each other’s policies, the more pronounced this non-stationarity efect becomes. This problem causes the loss of the Markov property (Gronauer and Diepold, 2022).

![](images/3488254c1255f29e28102832bdca777841505a54cd862405d8b2cab5ec319db0.jpg)  
Figure 3: Partial knowledge representation. Red arrows represent missing information for Agent 1.

For example, consider a two-agent Rock-Paper-Scissors game. If agent A plays Rock with probability 1, then the best response for agent B is to play Paper with probability 1. However, as agent A updates its policy during learning, the best response for agent B will change. Agent B must also adapt its strategy accordingly. From agent B’s perspective, the environment is therefore non-stationary because for the same observed state, the optimal action will vary over time.

The non-stationarity problem in MARL is not exactly the same as the non-stationarity problem in single-agent RL, where it is caused directly by the environment itself (Albrecht et al., 2024).

## 3.6.2 Partial Knowledge of the Environment

As explained in Section 2.7.2, in RL, partial observability can already pose dificulties 2.7.2. An agent may not have access to the full state of the environment when making observations, which deprives it of information that can sometimes be crucial for decision-making. Diferent strategies can be employed to mitigate this limitation, such as maintaining a history of past observations and actions (Igl et al., 2018).

Thus, partial observability is not exclusive to MARL. However, in the multi-agent setting this issue becomes more pronounced. Agents do not have access to each other’s observations, rewards or policies and must choose their actions without access to the joint action at decision time 3.

## 3.6.3 Credit Assignment

As mentioned in Section 2.7.3, in RL, the credit assignment issue refers to the fact that an action may result in a high reward while pushing the environment into an undesirable state, or conversely, an action with little immediate reward may alter the environment in a way that leads to higher future rewards. Identifying which actions truly contributed to a received reward is therefore a non-trivial task.

In MARL, this problem becomes even more complex (Albrecht et al., 2024; Gronauer and Diepold, 2022). The question is no longer only which action led to a given reward, but also which agent was responsible for it. Because the reward function depends on the joint actions of all agents, an individual agent cannot easily determine whether an improvement in its reward comes from its own policy update or from the behavior changes of others.

For instance, consider a fully cooperative MARL scenario where all agents share the same global reward. Two agents work together to collect resources in a partially observable 2D grid. Collecting a resource gives +1 to the global reward. Suppose that at time t, agent A observes a resource and decides to move away from it, this action should be penalized. However, if at the same time agent B collects one resource, the global reward will be positive. Therefore, agent A may reinforce its action of moving away from the resource, even though it was detrimental to the team’s objective. If agents could estimate their individual contributions to the global reward, it would prevent this issue.

## 3.6.4 Scalability

As explained in Section 2.7.4, scalability can be a great challenge in RL. The size of the state-action space given by $| S | \times | A |$ grows exponentially with the number of states and actions.

With the multiplication of agents, scalability becomes all the more a major challenge in MARL. As the number of agents increases, the joint action space grows exponentially. For example, in a system where N agents can choose among 10 possible actions, the joint action space consists of $1 0 ^ { N }$ possible combinations of actions. Agents whose policies depend on the joint action space will therefore face significant dificulties as the number of agents increases. Consequently, some algorithms become impractical for large MARL systems.

This scalability challenge also amplifies other challenges. As the number of agents increases, the dynamics become more non-stationary, each agent’s knowledge becomes more partial, and credit assignment becomes more complex.

## 4 Identifying MARL Core Modules

## 4.1 MARL Modular Decomposition

## 4.1.1 Motivating a MARL Modular Decomposition

When designing a MARL system, many choices have to be made depending on the use case. This article argues that these design choices should be presented in a more standardized way, by organizing MARL implementations into explicit modules.

This perspective is motivated by observations from the MARL literature. The term algorithm is often used to encompass a wide range of components and processes in MARL. When presenting their approach, articles do not always clearly specify all components of their approach.

When it comes to explaining how it works, the organization and terms vary from one article to another. Lowe et al. (2017) use a CTDE approach where agents are said to infer policies of others, which is referred to as agent modeling by Albrecht et al. (2024), modeling other agents (MOA) by Jaques et al. (2019) or opponent modeling by Chernyavskiy et al. (2025).

Catacora Ocana et al. (2019) claim comparing IL and JAL agents. However, in this context the term JAL is the equivalent of central learning by Albrecht et al. (2024), while in the latter, JAL refers to agents considering joint action to select their action but while keeping their autonomy and not merging into a super agent. Zhang et al. (2022) use JAL with the same definition as Albrecht et al. (2024). Central learning is also named centralized RL by Bu¸soniu et al. (2006).

Not only do the terms vary from one article to another, but the way the approaches are presented also difers. For instance, the training and execution paradigm is clearly specified in some works (Shou and Di, 2020; Lowe et al., 2017), but not in others (Catacora Ocana et al., 2019; Yang et al., 2023).

Shou and Di (2020) clearly specify the reward model, whereas Catacora Ocana et al. (2019) refer to a cooperative reward without clarifying whether it is fully cooperative or follows another structure, and Lowe et al. (2017) present multiple experiments but do not always specify the reward model.

Approaches that rely on communication clearly specify the communication mechanism (Kar et al., 2013; Zhang et al., 2018; Jaques et al., 2019). Conversely, when communication is not included, this absence is usually not explicitly stated or justified (Catacora Ocana et al., 2019; Shou and Di, 2020), although there are some exceptions (Chernyavskiy et al., 2025).

Taken together, these examples highlight the lack of a standardized way to describe MARL design choices, both in terms of terminology and in terms of the explicitness of the components involved. This makes it harder to identify which components are involved, compare approaches, and systematically explore possible solutions.

Identifying core modules and explicitly stating the choice made for each of these modules would make it easier to understand and compare approaches, even when the chosen option is the absence of an additional mechanism, such as no communication or fully decentralized execution. It could also help identify which choices contribute to improvements and facilitate their transfer across MARL approaches and experimental settings by reusing the corresponding core module implementations.

## 4.1.2 Identified Modules

The core modules identified in this paper correspond to the main concerns involved in modeling, describing, and implementing MARL experiments. In the following, we present and justify the criteria used to identify them.

A MARL experiment can be described through three main dimensions:

• the definition of the problem faced by the agents;

• the behavior of the agents, including how they learn, make decisions, and interact;

• the evaluation of the resulting behaviors and outcomes.

A MARL experiment must specify the problem faced by the agents and how the agents behave within it. It must also specify how the resulting behaviors and outcomes are evaluated, since evaluation tasks, protocols, and metrics are critical to the validity and comparability of MARL results (Papoudakis et al., 2021b; Gorsane et al., 2022).

Accordingly, the first criterion is that, to be considered a core module, a module must be central to the description of one of these three dimensions.

One objective of the decomposition into core modules is to provide a clearer view of the modeling choices made in MARL experiments and facilitate their comparison. Therefore, a core module should correspond to a genuine modeling choice for which multiple implementations can be defined, substituted, and studied as independent experimental factors.

Finally, to avoid defining multiple overlapping modules, core modules should be nonredundant with one another. Each module should capture a distinct aspect of the MARL experiment and have a clearly identified responsibility. This does not imply that modules are independent: interactions between them may make some combinations of module implementations incompatible.

<table><tr><td rowspan=1 colspan=1>Core module</td><td rowspan=1 colspan=1>Necessary dimension</td><td rowspan=1 colspan=1>Modeling choices</td><td rowspan=1 colspan=1>Specific responsibility</td></tr><tr><td rowspan=1 colspan=1>Environment Dynamic4.2</td><td rowspan=1 colspan=1>Definition of the problemfaced by the agents: in-teraction dynamics andstate of the environment</td><td rowspan=1 colspan=1>State    representationsand environment dynam-ics</td><td rowspan=1 colspan=1>Scheduling the agents&#x27;action and handling theReaction phase by com-puting state transitionsfrom agents&#x27; actions</td></tr><tr><td rowspan=1 colspan=1>Reward modeling 4.3</td><td rowspan=1 colspan=1>Definition of the problemfaced by the agents: ob-jectives and feedback sig-nals provided through re-wards</td><td rowspan=1 colspan=1>Different reward struc-tures</td><td rowspan=1 colspan=1>Computing the rewardsreturned to agents</td></tr><tr><td rowspan=1 colspan=1>Learning 4.4</td><td rowspan=1 colspan=1>Behavior of the agents:actionselectionandlearning</td><td rowspan=1 colspan=1>Many policies and learn-ing algorithms are avail-able</td><td rowspan=1 colspan=1>Selecting actions and up-dating policies</td></tr><tr><td rowspan=1 colspan=1>Training and executionmodes 4.5</td><td rowspan=1 colspan=1>Behavior of the agents:degree of centralizationof information and pro-cesses during trainingand execution</td><td rowspan=1 colspan=1>Different degrees of cen-tralization during train-ing and execution can beselected and compared</td><td rowspan=1 colspan=1>Specifying the informa-tion available in eachphase and defines cen-tralized process when re-quired</td></tr><tr><td rowspan=1 colspan=1>Communication 4.6</td><td rowspan=1 colspan=1>Behavior of the agents:whether and how agentscommunicate</td><td rowspan=1 colspan=1>Message content, com-municationstructure,andmessage-handlingmechanisms can eachvary</td><td rowspan=1 colspan=1>Managing the exchangeof information betweenagents when communica-tion is enabled</td></tr><tr><td rowspan=1 colspan=1>Model of Other Agents4.7</td><td rowspan=1 colspan=1>Behavior of the agents:whether other agents aremodeled, and how suchmodels are represented,learned, and used</td><td rowspan=1 colspan=1>MOA can vary in its rep-resentation and learningmethod</td><td rowspan=1 colspan=1>Determining    whetherand how agents modelothers, and for whatpurposes, such as actionprediction</td></tr><tr><td rowspan=1 colspan=1>Agent 4.8</td><td rowspan=1 colspan=1>Behavior of the agents:agentdecision-makingand interaction with theenvironment</td><td rowspan=1 colspan=1>Alternative ways of or-chestrating internal mod-ules and carrying agent-level responsibilities</td><td rowspan=1 colspan=1>Orchestrating the differ-ent modules used by anagent</td></tr><tr><td rowspan=1 colspan=1>Evaluation criteria 4.9</td><td rowspan=1 colspan=1>Evaluation of the result-ing behaviors and out-comes</td><td rowspan=1 colspan=1>Alternative measures ofperformance,  fairness,convergence, and otherproperties</td><td rowspan=1 colspan=1>Measuring specific as-pects of system behaviorand outcomes accordingto the objectives</td></tr></table>

Table 1: Core modules and their correspondence with the identification criteria.

Table 1 summarizes how each identified core module satisfies these criteria.

The implementation chosen for each core module can significantly afect the experimental outcomes and the dynamics of the underlying MARL system. Making these modules explicit therefore helps isolate and study the impact of specific design choices.

This Section identifies eight core modules that should be specified when implementing a MARL system: (1) agent, (2) environment dynamics, (3) learning, (4) reward modeling, (5) training and execution modes, (6) communication, (7) model of other agents (MOA), and (8) evaluation criteria.

For each module implementation presented, we provide a definition, discuss the motivation behind its use, identify the challenges it addresses or is most afected by, and, when relevant, give illustrative examples.

## 4.1.3 Potential Modules Not Retained

Several potential modules were considered but ultimately rejected because they did not satisfy all the identification criteria. The following paragraphs discuss some examples to illustrate how these criteria were applied.

Several potential modules were considered but ultimately rejected because they did not satisfy all the identification criteria. The following paragraphs discuss two examples to illustrate how these criteria were applied. For each potential module, we provide its definition, the rationale for considering it, and the reasons for its exclusion.

Not being retained as a core module in the conceptual decomposition of MARL does not imply that a concept is unimportant or that it should not be represented as a dedicated component in a software implementation. The core module criteria are intended to provide a conceptual decomposition of MARL solutions and experiments, rather than prescribe a software architecture.

Scheduler The scheduler is the component that orchestrates all the other components. It has a specific responsibility. In the literature, we mainly identified two types of scheduling: agents either select their actions simultaneously or make decisions sequentially. However, the choice of scheduling mechanism is completely determined by the environment dynamics. More specifically, the scheduler must call agents simultaneously when state transitions depend on their joint actions, and sequentially when each transition depends on the action of a single agent. Therefore, we did not include the scheduler as a separate component in our core module decomposition, since its behavior is entirely determined by the environment dynamics<sup>6</sup>.

Exploration Strategy The exploration vs. exploitation dilemma is an important question in both RL and MARL, and diferent exploration strategies exist. However, exploration strategies are too tightly coupled with the learning module to constitute an independent core module and are most commonly integrated into the policy, which maps inputs to actions.

Learning Parameter Adaptation Policies and learning algorithms have parameters. It is possible to use approach that adapt learning parameter based on the dynamics. Several approaches, such as WoLF and hysteretic learning, exist and can improve learning. However, these techniques apply only to specific learning mechanisms. They are therefore too specialized and tightly coupled with the learning module to constitute an independent core module.

Neural Network Architecture Neural networks architectures can take many different forms. Some components rely on neural networks, whose architecture can vary and afect the results. However, providing mechanisms for selecting or switching between neural network architectures is primarily a software design concern. The neural network model is therefore considered an internal aspect of the learning module rather than an independent core module.

## 4.2 Environment Dynamic

The environment is a central module of MARL systems. Agents interact and evolve within the environment, whose dynamics determine how the system changes over time. Among the components of the environment, the dynamics are the most implementation-dependent, since they define how the environment reacts to agents’ actions.

The environment also defines how agents are scheduled, namely whether they select their actions simultaneously or in a sequential order.

As explained in Section 3.3, the transition function must be computed based on joint actions of all agents (not each action individually). While such a perspective is widely admitted, its conceptual consequences are rarely translated into explicit modeling guidelines, and are often handled in an ad-hoc manner.

The need to manage the transition depending on multiple agents is a problem that already exist in the MAS literature. To capture this, the model of the environment we use is based on IRM4S (Influence Reaction Model for Simulation) (Ferber and M¨uller, 1996). Since the dynamics depend on joint actions, the environment reaction should be computed separately from the phase in which agents choose and perform their actions. In this paper, we make this separation explicit by relying on the Influence/Reaction principle, a two-phase modeling paradigm introduced in the MAS literature. This separation helps preserve the independence of the environment module by keeping agents’ intended actions distinct from the computation of their efects.

The Influence/Reaction principle ofers a paradigm for modeling MAS in which agents never alter the environment directly. Instead, they generate influences that express their intended actions, while the environment subsequently reacts to the complete set of influences produced simultaneously. This approach clearly separates the process into two complementary phases: an Influence phase, where agents and the environment emit their influences, and a Reaction phase, during which the system combines these influences and computes the resulting state (Michel, 2007). The dynamics of the environment then change accordingly to the joint actions, namely the influences, of all agents. Allowing the agents to directly modify the environment may lead to conflicts and will be in opposition to the stochastic game framework where the dynamics depend on the joint action.

Figure 4 illustrates the Influence/Reaction principle and its separation between the Influence and Reaction phases. In the MARL setting considered here, this process is followed by an additional phase dedicated to computing agents’ rewards. This third phase, called the Reward phase, is executed after the Reaction phase.

Unlike other core modules, whose implementations may introduce mechanisms to support or modify agents’ behavior, the environment primarily defines the situation faced by the agents, including the constraints and dificulties they must handle.

## 4.3 Reward Modeling

The reward modeling is more dificult to design in MARL than in RL because one has to compute a reward value for each agent, in a way that can drive the whole system toward its goal. The modeling of the reward function in a MAS is a critical aspect that determines the behavior of the agents in the environment.

In the same environment, variations in reward functions can cause agents to engage in either intense competition or fully cooperative behavior. For instance, Tampuu et al. (2017) experiments with diferent reward models for its system and shows that with the same environment, they afect the dynamics of the system and how agents learn to act.

![](images/6d320faeb1de7f91ec153779222379975b68741270ae9af0728e66fee77b9e67.jpg)  
Figure 4: Influence/Reaction principle: agents produce influences that are combined into a joint action, which is used to compute the environment reaction (state transition). Then the rewards are compute in the reward phase.

Although there are infinite numbers of ways to define the reward function, they can be grouped into three main categories: (1) fully cooperative, (2) fully competitive, and (3) mixed (Liang et al., 2025; Canese et al., 2021).

## 4.3.1 Fully Cooperative

Definition A MARL system is said to be fully cooperative<sup>7</sup> when all agents share a unique global reward: $\forall ( i , j ) \in [ 1 , N ] , R _ { i } \ : = \ : R _ { j }$ . In this case, the agents should work together to maximize the global reward (Liang et al., 2025; Canese et al., 2021).

Motivation Whenever agents optimize non-identical reward functions, conflicts of interest may emerge, even in settings intended to be cooperative. A fully cooperative formulation, where all agents share the same global reward, removes this misalignment of incentives. In that case, an agent cannot improve its own return without also improving the others’, which prevents individually beneficial yet globally suboptimal behaviors.

Example Consider a foraging task in which two agents are tasked with collecting resources. The episode ends after all resources are collected or a maximum number of timesteps is reached. Let’s consider this experience with two diferent reward function designs:

• Fully Cooperative Reward: All agents receive the same global reward: a positive reward is given whenever any agent collects a resource, and a negative reward, representing time penalty, is given at each timestep.

• Individual Reward: Each agent receives positive rewards for collecting a resource and negative rewards every timestep.

In the first case, the optimal policy for the agents is to divide the resources among themselves to minimize overall travel time and reduce the time to collect all the resources. However, in the second case, there is a balance to strike between collecting resources and minimizing the global time to collect all the resources. In specific situations, it may be optimal for an agent to collect all the resources, prioritizing the one close to the other agent, even if it increases the global time to collect all the resources.

In MARLKIT, there is a foraging experiment that illustrates this situation (see Section 6.3)

Challenges In this setting, the objective is straightforward: optimize the global reward.   
This objective is close to single-agent RL, which is one-dimensional.

The main drawback of this approach is related to the credit assignment problem (Albrecht et al., 2024; Ning and Xie, 2024; Gronauer and Diepold, 2022). It is harder for an agent to improve its policy because it cannot identify which actions contributed to the global reward. As the number of agents increases, the credit assigned to an agent’s individual actions becomes diluted.

## 4.3.2 Fully Competitive

Definition A MARL system is said to be fully competitive<sup>8</sup> when the sum of all agents’ rewards is constant: $\begin{array} { r } { \sum _ { i = 1 } ^ { N } R _ { i } = C , } \end{array}$ where C is a constant and N is the number of agents (Liang et al., 2025; Canese et al., 2021). When the constant is zero (C = 0), the system is said to be a zero-sum game.

Motivation Consequently, if one agent increases their reward, the others’ rewards decrease by the same amount. In this setting, agents have completely opposed interests. This is a common setting in 1v1 competitive games, such as chess or Rock-Paper-Scissors.

Example In this setting, agents are incentivized to compete against each other. The dynamics that emerge from this setting are diferent from the ones that emerge in other settings. Tampuu et al. (2017) show that in a simple 1v1 Pong game, the reward modeling afects the dynamics of the game and how agents learn to play. In a fully competitive setting, agents learn to make angled shots that bounce of the walls. The number of paddle-bounces per point is lower than in other settings.

Challenges In this setting, agents’ rewards are strictly opposed and thus completely dependent on other agents’ actions. This exacerbates the non-stationarity problem (Lowe et al., 2017), as each agent’s optimal action is highly sensitive to the actions of other agents.

## 4.3.3 Mixed

Definition The reward modeling is considered mixed when it is neither fully cooperative nor fully competitive<sup>9</sup> (Liang et al., 2025; Canese et al., 2021).

Motivation Mixed systems can exhibit varying degrees of cooperative or competitive behavior (Liang et al., 2025). In fully cooperative and fully competitive settings, agents’ rewards exhibit a strong correlation, reflecting the aligned or strictly opposed nature of their objectives. In contrast, in mixed settings, the correlation between agents’ rewards can be weaker or reflect more complex interdependencies among their objectives.

Example A simple example of a mixed game oriented toward competition is a resource collection game, in which agents should collect resources to maximize their own reward. Collecting a resource increases an agent’s reward without directly decreasing the others’ rewards. However, it decreases the amount of resources available to them. An example of a cooperatively oriented game is the robovac scenario presented above, with reward functions that are not shared.

Challenges The credit assignment problem is due to agent reward being afected by other agents’ actions (Gronauer and Diepold, 2022). Therefore, the less an agent’s reward is afected by the actions of other agents, the easier it is for the agent to identify which actions contributed to its reward. This is the reason why the credit assignment problem is important in fully cooperative settings (Albrecht et al., 2024). In fully competitive settings, agents share the same reward function, in fully competitive settings, one agent’s reward function is fully determined by the others’ reward functions. In mixed settings, the correlation between agents’ rewards function can be weaker. Less correlation between agents’ rewards and others’ actions can mitigate the credit assignment problem.

The non-stationarity problem is due to the fact that an agent’s optimal action is afected by other agents’ actions (see Section 2.7.1). Because the policy of other agents changes over time, the optimal action for an agent can change over time, then the agents face a moving target problem (Gronauer and Diepold, 2022; Hernandez-Leal et al., 2017). Therefore, the fewer an agent’s reward is afected by the actions of other agents, the less non-stationarity issues the agent will face. Depending on the reward model chosen, the non-stationarity problem can be less problematic than in fully cooperative or fully competitive settings. However, even if the reward function of an agent is only afected by its actions and the current state, the transition function of the environment is still afected by the actions of other agents, then the non-stationarity problem is still present even in a completely independent reward setting.

## 4.4 Learning

The learning module is responsible for managing agents’ decision-making and learning processes. It consists of two sub-modules that are too closely tied to be considered as independent modules: learning algorithm and policy.

The policy is still a function that takes as input the observation of the agent and outputs an action. However, now the policy may also take as input other information. Diferent approach in MARL consist in considering other information before selecting an action, such as communication from other agents, or a prediction of the action of other agents. Then the policy function will have to be adapted to take this new information as input. Figure 5 illustrates this idea, where the policy can now take other input.

![](images/775a09b661e136b8edf14bbfef23609e001a06e09737c25ed049d111c8494369.jpg)  
Figure 5: policy can now take other input such as prediction of other agent action, or information that came from communication

Even if the input may difer, the diferent types of policies presented in Section 2.6.1 remains relevant in MARL.

MARL implementations can use single-agent RL learning algorithms. However, MARL learning algorithms are often more complex: although many of them build upon single-agent RL algorithms, they include additional mechanisms to account for interactions between agents.

The core RL algorithmic categories introduced in Section 2.6 remain relevant in MARL. However, some algorithmic aspects require particular attention in multi-agent settings.

This is especially the case for learning parameters, whose adaptation can help address the non-stationarity induced by simultaneously learning agents. The following subsections present several MARL techniques based on adaptive learning parameters.

In stationary single-agent RL settings, the transition and reward functions are fixed. Consequently, the optimal value function and the set of optimal policies are also fixed, providing the agent with a stable target toward which it can converge. In this context, learning parameters, such as the learning rate or exploration rate, are often decreased over time to progressively stabilize learning and support convergence toward an optimal behavior.

In a multi-agent system, agents learn simultaneously. Moreover, the optimal action for a given agent depends on the actions of the other agents, and therefore on their policies. As a result, agents must update their policies with respect to those of others, which are themselves evolving. This gives rise to the non-stationarity problem discussed in Section 3.6.1. To address this issue and support policy convergence, one algorithmic approach is to use adaptive learning parameters.

Adaptive learning parameters approaches are more commonly used in MARL than in single-agent RL. Learning parameters can also vary in single-agent RL, but these variations are usually predefined rather than driven by interactions with the environment.

The following subsections present diferent adaptive learning-parameter techniques used in MARL.

## 4.4.1 Win or Learn Fast (WoLF)

Definition In the Win or Learn Fast (WoLF) approach (Bowling and Veloso, 2002), the learning rate is adapted according to whether the agent is winning or losing: when winning, the agent learns slowly; when losing, it learns faster in order to improve its performance.

Motivation The main motivation behind WoLF is to promote rational learning behavior while improving the stability and convergence properties of the overall system.

From the perspective of a fast-learning agent, slower-learning agents have more stable and predictable policies, which makes learning easier for fast-learning agents.

Examples Several implementations of WoLF have been proposed. For example, WoLF-PHC (WoLF-Policy Hill Climbing) (Bowling and Veloso, 2002), WoLF-IGA (WoLF-Infinitesimal Gradient Ascent) (Bowling and Veloso, 2002), and GIGA-WoLF (Generalized Infinitesimal Gradient Ascent-WoLF) (Bowling, 2004).

WoLF-PHC has been applied in Rock-Paper-Scissors game, where two agents learn to play against each other (Bowling and Veloso, 2002). WoLF-PHC is compared to PHC. The results show that WoLF-PHC converges to the Nash equilibrium, while PHC does not.

Challenge Because slower-learning agents have more stable and predictable policies, faster-learning agents face a less severe form of non-stationarity. Therefore, WoLF helps reduce the non-stationarity problem (Bowling and Veloso, 2002; Bowling, 2004).

## 4.4.2 Two-Timescale (2TS)

Definition The 2TS approach, originally introduced by Borkar (1997), consists of defining two diferent learning rates for two processes<sup>10</sup>, so that one process evolves much faster than the other<sup>11</sup>.

Motivation From the perspective of one agent, non-stationarity is caused by other agents updating their policies. Consequently, the optimal policy for this agent keeps changing as its learning progresses.

The key idea of 2TS is that when one process evolves much faster than the other, the faster process sees the slower one as nearly stationary, which helps it converge toward an appropriate response. Conversely, from the slower timescale, the faster process can be viewed as being close to its response to the current state of the slower process, since it adapts much more rapidly.

In this sense, the motivation behind 2TS is close to that of WoLF: both approaches rely on asymmetric learning speeds to improve stability in multi-agent learning.

Examples Several implementations of 2TS have been proposed. Leslie and Collins (2003) generalize 2TS to n agents by assigning a diferent timescale to each agent. In this work, agent 1 learns slower than agent 2 which learns slower than agent 3 and so on. As a result, faster-learning agents can adapt their policies to slower-learning ones.

The 2TS approach can also be applied to diferent learning components within an agent. Perkins et al. (2015) use 2TS in an Actor Critic algorithm to separate Critic (fast) and Actor (slow). In this second example, all agents learn at the same overall pace but the critic learns faster than the actor, leading to a more stable evaluation of the current policy.

The 2TS approach was applied in a Stick Together Game (STG), where two agents have to learn to group and stay together (Prasad et al., 2015). 2TS is compared to two other algorithms: Nash Q-learning and Friend-or-Foe Q-learning, two MARL variants of Q-learning. The results show that 2TS outperforms these two algorithms in terms of convergence and performance.

Challenge By separating learning dynamics across diferent timescales, 2TS makes some processes appear nearly stationary from the perspective of others. In this way, it helps mitigate the non-stationarity problem (Borkar, 2002; Leslie and Collins, 2003; Perkins et al., 2015).

## 4.4.3 Hysteretic Learning Rates

Definition Hysteretic learning rates consist of using two diferent learning rates: α for Q-value increases and β for Q-value decreases, with $\alpha > \beta > 0$

Motivation The same action can lead to a high or low reward, depending on the joint actions of all agents. Therefore, an agent can receive a low reward even when it takes the optimal action, because others are exploring and taking suboptimal actions.

Hysteretic learning rates are designed to address this issue by using two diferent learning rates, so agents remain optimistic about beneficial experiences and are less misled by outcomes induced by others’ exploration.

Examples This approach was first introduced by Matignon et al. (2007), with the Hysteretic Q-learning (HQL) algorithm. It has since been combined with other algorithms, such as DQN (HDQN) (Zhang et al., 2022).

HQL was applied in a prey-predator game with partial observability (Matignon et al., 2007). In this setting, four predators have to coordinate to catch a prey. HQL is compared to Distributed Q-learning (DQL, defined by Lauer and Riedmiller (2000)). The results show that HQL outperforms DQL.

Challenge By reducing the impact of negative updates caused by other agents’ exploration, hysteretic learning rates prevent agents from prematurely discarding actions that may be beneficial under diferent joint actions. Therefore, hysteretic learning rates make learning more robust to the variability induced by simultaneously learning agents and help mitigate one aspect of the non-stationarity problem.

## 4.5 Training and Execution Modes

In MARL, a distinction can be made between the training phase, during which agents update their policies, and the execution phase, during which agents use these policies to act in the environment. In some settings, these two phases are merged: agents are evaluated while they are still learning. In other settings, they are separated: agents are first trained, and their performance is then evaluated during a distinct execution phase.

This distinction is important because agents may not have access to the same information during training and execution. For example, centralized information may be available during training to facilitate learning, while agents may have to act only from local information during execution. Training and execution modes therefore define how centralized or decentralized the agents are in each phase.

In many use cases, the training and execution phases are distinct, allowing for diferent approaches in each phase. There are three main modes for training and execution in MARL: Centralized Training and Execution (CTE), Decentralized Training and Execution (DTE), and Centralized Training Decentralized Execution (CTDE).

## 4.5.1 CTE

Definition CTE implies that agents are trained and executed using centralized information (Zhang et al., 2021; Albrecht et al., 2024). This means that during both training and execution, agents should have access to a central process that can aggregate information from all agents and communicate it back to them. These information can include observations, environment dynamics, other agent models... They can keep their autonomy, but they interact with a central entity.

But CTE is sometimes used to refer to a more restrictive definition. In this case, all agents are interacting with a central process that pilot the joint policy of all agents (Gronauer and Diepold, 2022; Wong et al., 2023). This is sometimes referred as Central Learning (Albrecht et al., 2024). Using this is transforming a multi-agent problem into a single-agent one. In this case, agents are not autonomous.

Motivation CTE allows agents to access more information during training and execution, which can facilitate learning and coordination (Albrecht et al., 2024; Sharma et al., 2021).

Challenges Increasing knowledge of other agents tends to reduce non-stationarity issues. CTE then helps to mitigate this dificulty (Sharma et al., 2021; Amato, 2024). For the same reason, it also helps to reduce the partial knowledge dificulty and could reduce credit assignment issues depending on the shared information (Albrecht et al., 2024). However, CTE can be impractical because it worsens the scalability issue (Gronauer and Diepold, 2022; Albrecht et al., 2024).

## 4.5.2 DTE

Definition In DTE, agents are trained and executed using only their own local information. There is no central entity to aggregate information or coordinate actions.

Motivation This approach requires fewer agents-coordination mechanisms. This is a natural choice when the agent cannot be trained or executed in a centralized setting (Albrecht et al., 2024). It can be more realistic for many applications where agents operate independently because a central entity cannot coordinate them (Zhang et al., 2021; Albrecht et al., 2024).

Challenges In this setting, the non-stationarity problem is at its peak (Amato, 2024), and becomes more severe if agents’ rewards are highly correlated with others’ actions. However, DTE can be more scalable because it does not rely on joint observation, or joint action spaces which grow exponentially with the number of agents (Albrecht et al., 2024).

## 4.5.3 CTDE

Definition The execution phase can involve numerous constraints and hidden information (about other agents’ policies, about the environment, etc). This hinders learning, especially in cooperative settings where agents would benefit from freely sharing all this information. The CTDE idea is to provide agents with additional information during training so that they can learn better policies, and then execute these policies in a decentralized manner (Amato, 2024). The first instance of CTDE was proposed by Kraemer and Banerjee (2016).

Motivation CTDE allows agents to exchange and access substantially more information than they will have during execution. This additional information facilitates learning by making it easier for agents to assess the consequences of their actions. CTDE is designed to combine the benefits of both CTE and DTE (Albrecht et al., 2024).

Examples There are diferent versions of CTDE. This section presents two. The most common being the combination of CTDE with Actor Critic algorithms, in which a centralized critic is used during training while the actors are decentralized (Lowe et al., 2017; Amato, 2024). The motivation is to obtain a better-trained critic that can mitigate the non-stationarity problem.

Parameter sharing is another technique used by Gupta et al. (2017) that can be used for centralized training. It consists of training a shared policy using the experiences collected by all agents. Gupta et al. (2017) argue that this technique increases training eficiency.

Challenges CTDE requires a training phase, and if training and execution are not separated, it cannot be applied. This condition makes it sometimes less realistic than DTE where agents are trained independently (Jaques et al., 2019).

Some CTDE techniques, such as parameter sharing, requieres agents to be homogeneous (Gupta et al., 2017).

For the same reasons as CTE, by increasing knowledge of other agents, CTDE tends to reduce non-stationarity issues (Gronauer and Diepold, 2022; Amato, 2024; Ning and Xie, 2024), helps to reduce the partial knowledge dificulty and could reduce credit assignment issues depending on the shared information.However, centralized training itself raises scalability issues: as the number of agents increases, the training process becomes progressively more expensive and complex (Amato, 2024).

## 4.6 Communication

As with training and execution modes, selecting a communication method is a design choice that shapes the ways in which agents interact.

The messages exchanged between agents can take various forms. Agents can share their observations, communicate their intentions, share value functions, etc.(Albrecht et al., 2024; Wong et al., 2023). The content of the messages exchanged can also be learned (Zhu et al., 2024).

![](images/9b976693a83be7de4a7e8d794e63f9ca8251e52f04d45d806e655b271e7412f7.jpg)  
(a) Broadcasting communication.

![](images/a8b6378e756b30c140ec4d4c343ef1bdf6d5eecd4aca896e48e1b7e37d7d3f64.jpg)  
(b) Networked communication.

![](images/77fc769d35bfe60600bd7855b37f97acd81fba1da7c461359dbaf4b1f22d4ee6.jpg)  
(c) Targeted communication.  
Figure 6: Comparison of communication modes: broadcasting, networked, neighborhood and targeted. Broadcasting communication (a) involves all agents sending messages to all others. Networked communication (b) involves agents communicating only with their direct neighbors in a predefined network structure. Targeted communication (c) allows agents to send messages to specific other agents. Figures adapted from Gronauer and Diepold (2022).

By sharing their observations, agents can have a better understanding of the environment state, which helps mitigate the partial observability challenge. By sharing their intentions or estimation of the value of actions, agents can better predict the behavior of others, helping to mitigate the challenge of non-stationarity (Wong et al., 2023; Gronauer and Diepold, 2022).

The main drawback of communication is the potential increase in computational complexity and scalability concerns. The impact depends heavily on the chosen communication mode.

Several communication modes exist. This section presents the main categories. For a more detailed survey of communication methods in multi-agent systems, readers are referred to Zhu et al. (2024).

## 4.6.1 No Communication

Definition The first trivial communication mode is: no communication. In this case, agents do not share any information with each other proactively. It limits the information available to each agent to its local observation.

Motivation This prevents the need for communication protocols, which can be complex to design.

Challenges The absence of communication does not ofer a solution to partial observability and non-stationarity, since agents lack explicit access to information about other agents’ states, actions, or policies. However, it does not introduce additional computational complexity nor scalability concerns.

## 4.6.2 Broadcast Communication

Definition Broadcast communication, illustrated in Figure 6a, involves each agent sending messages to all others in the system.

Motivation In this setup, agents receive information from all others. Then, agents can take actions based on more information: their local observation and more global information from other agents (Chen, 2020).

Challenges This method can be very computationally expensive, since each agent has to process information from all others.

Broadcast communication faces strong scalability issues. The more agents there are in the system, the more messages each agent has to process. This is not just computationally expensive, but also can lead to information overload, where agents have to process large amounts of information, including irrelevant messages (Jiang and Lu, 2018).

## 4.6.3 Networked Communication

Definition Networked communication, illustrated in Figure 6b, involves agents communicating only with predefined agents. Such a communication graph must be defined.

How to set up the communication graph is a design choice. For instance, it can be based on a distance metric, such as spatial proximity.

The network can be static or dynamic. When the network is dynamic, the connections between agents can change over time. If the neighbors are defined based on spatial proximity, then the network is dynamic because it can change when agents move.

Motivation In this setup, agents can send information to their network’s neighborhood but do not have to be able to communicate with all others, nor do they have to process information from all agents in the system. When the dynamics are mostly local, it is often suficient to condition decisions on information only from nearby agents.

Example Kar et al. (2013) present a networked communication method where agents communicate with neighboring agents over a sparse, pre-specified, and possibly timevarying communication graph. The communication is motivated by a setting where the global evaluation is not directly observable by individual agents, while agents have to minimize a global cost. A centralized architecture would require each agent to continuously send its local cost to a remote central unit, which is impractical, especially due to limited energy resources. Therefore, the authors adopt a networked communication approach.

Challenges This technique is less computationally expensive than broadcast communication, since each agent only has to process information from a subset of agents.

## 4.6.4 Targeted Communication

Definition Targeted communication, illustrated in Figure 6c, involves agents able to choose which other agents to communicate with.

Motivation While networked communication tackles the scalability issue of broadcast communication by restricting communication to predefined neighbors, targeted communi cation methods enable agents to dynamically select what, with whom, or when to communicate, depending on the specific architecture.

Examples Agents in papers by Jiang and Lu (2018); Das et al. (2019) can choose whether to share information with other agents either by selecting the content and recipient of a message (Das et al., 2019) or by deciding whether to initiate communication (Jiang and Lu, 2018).

Jiang and Lu (2018) present an application of this communication mode. In this work, agents can initiate a group of communication with other agents named collaborators. For a given number of steps, agents in the group can exchange messages. Agents use an attention mechanism to select when to initiate communication. However, the agents cannot select with whom to communicate, because agents initiate communication with other agents in their field.

Das et al. (2019) present TarMAC, a targeted communication method where agents can select which other agents to communicate with and also what message to send.

Challenges Unlike fixed network structures, targeted communication allows agents to potentially address any other agent in the system, while avoiding unnecessary message exchanges. This design aims to improve scalability and flexibility compared to broadcast communication, while preserving relevant information (Jiang and Lu, 2018; Das et al., 2019).

## 4.7 Model of Other Agents

In MARL, agents interact with the environment by receiving observations and rewards, and by taking actions. The non-stationarity problem arises because from the point of view of an agent, the dynamics of the environment are not only determined by the state and the actions of the agent, but also time-dependent. The time dependency is due to the policies of the other agents that are updating during the learning process. However, learning in a non-stationary environment is more challenging than learning in a stationary environment (see Section 3.6.1).

Being able to consider the behavior of other agents is crucial in many MARL systems (Huh and Mohapatra, 2023). By modeling the other agents, separately from the environment, an agent can better understand the dynamics of the environment and predict other agents’ behavior, which can help to take better actions and to reduce the impact of non-stationarity (Albrecht et al., 2024). Jaques et al. (2019) argue that MOA is a good alternative to CTDE. The information or predictions produced by MOA can then be used by the learning module for action selection or policy updates.

This section focuses on how agent can predicts others’ actions. However, MOA also encompasses other tasks, including classification<sup>12</sup> and plan recognition<sup>13</sup>. These two topics will not be discussed further here, a more detailed treatment is provided by Albrecht and Stone (2018).

This section identifies two main components of MOA: (1) the representation of MOA, and (2) the learning of MOA (how such models are learned).

By representation of MOA, we mean how other agents are modeled. Are they modeled individually as learning agents, are they modeled as a group, do they assume a default strategy for other agents? After one has decided on the representation of MOA, the next step is to learn the MOA. Some modeling do not require learning, but for those that do, the learning of MOA is about how an agent learns its models of others. There are many ways to learn the MOA, Section 4.7.2 identifies some of them.

## 4.7.1 Representation of MOA

Independent Learner (IL) In Independent Learning, each agent ignores the others and treats them as part of the environment. Agents do not attempt to predict each other’s behavior; they base their actions solely on their own observations, as in standard RL.

This allows the use of single-agent RL algorithms, which are often simpler than multiagent methods.

IL was used by Tan (1993) and subsequently identified as Independent Learning by Claus and Boutilier (1998). Bu¸soniu et al. (2008) argue that Independent Learning is a valid approach with advantages (scalability, robustness, parallel computation). More recently, Independent Learning has been implemented in the StarCraft Multi-Agent Chal lenge (SMAC) (Samvelyan et al., 2019) and has demonstrated good performance (De Witt et al., 2020).

Matignon et al. (2012) identify five major dificulties that arise with IL in a cooperative framework. These dificulties are: Pareto-selection (the existence of at least two incompatible Pareto equilibria), non-stationarity, stochasticity (in state transitions or reward calculations), alter-exploration (asynchronous exploration-exploitation by agents), and shadowed equilibria (reward mechanics can lead to suboptimal Nash equilibria and mask action combinations that yield optimal outcomes).

IL does not address the non-stationarity or the multi-agent credit assignment problems.

Joint Action Learning Game Theory (JAL-GT) In some settings, agents can observe the actions of other agents. In this situation, they can evaluate their actions based on the joint action, rather than considering only their own action.

If an agent uses a Q-value function, it can extend it to depend on the joint action a∗, leading to a value function of the form Q(s, a∗). However, using such a function requires access to the actions of the other agents. In practice, this means that the agent must predict the actions of other agents.

Instead of using a prediction model, agents can use solutions based on game theory (Albrecht et al., 2024). Such game-theoretic solution concepts can be for instance minimax (Littman, 1994) or Nash equilibrium (Hu and Wellman, 2003).

The advantage of using game-theoretic solutions is that agents do not need to learn a model of other agents. However, they need to observe others’ actions in order to update their policies.

Taking into account the actions of others and using more accurate action evaluations helps mitigate both the non-stationarity and the credit assignment problems.

Joint Action Learning Agent Modeling (JAL-AM) Relying on game theory solution without predicting the actions of other agents prevents agents from exploiting weaknesses of other agents (Albrecht et al., 2024).

An alternative approach, still based on agents observing joint actions, consists of learning models of other agents in order to predict their future actions, rather than relying on game-theoretic solution concepts. This approach is named JAL-AM by Albrecht et al. (2024).

An agent using JAL-AM can learn a best response to the predicted actions of other agents. This approach allows agents to exploit weaknesses of other agents, or to coordinate better with other agents. However, learning a model of other agents can be complex. The problem of learning a MOA is discussed in Section 4.7.2.

This approach presents advantages similar to those of JAL-GT, as it also helps mitigate the non-stationarity and credit assignment problems.

Group Modeling Modeling each agent independently assumes that their behavior can be predicted separately. However, sometimes there are strong correlations between agents’ actions. The main limitation of modeling all agents independently with separate models, it may fail to capture correlations or coordination patterns among agents.

Group Modeling addresses this limitation by modeling multiple agents jointly rather than independently. More precisely, the group is represented through a single model whose action space is the joint action space of the agents. It enables the prediction of joint actions while capturing dependencies between agents’ actions (Albrecht and Stone, 2018).

A major drawback of this approach is its sensitivity to the scalability issue, since the model of the group depends on the joint action space, whose grows exponentially with the number of agents. One possible mitigation strategy of this issue is to divide the full group into subgroups (Albrecht and Stone, 2018)

Mean Field In the Mean Field (MF) modeling approach, each agent considers the actions of the other agents collectively rather than individually. From an agent’s perspective, the system reduces to two entities: the agent itself and the others, treated as a single entity: a mean agent. MF MARL was first introduced by Yang et al. (2018).

This approach may appear similar to Group Modeling, since in both cases the other agents are represented collectively. However, there is an important diference: unlike group modeling, MF does not aim to preserve the individual contribution of each other agent. Instead, it aggregates them and considers mean action. As a result, the identities and individual actions of the other agents are not explicitly represented in the model.

Accounting for the behavior of other agents can be computationally too expensive (see Section 3.6.4). This is why many algorithms work well with two agents but become impractical when the number of agents grows well beyond two. The Mean Field approach is motivated by the goal of recasting the problem into an efective two-entity setting (Yang et al., 2018; Subramanian et al., 2021).

We describe the method from the viewpoint of agent i. Agent i models the others as a mean agent. This mean agent does mean action ${ \bar { a } } ^ { i }$ defined by the Equation 7. In the latter, ${ \mathcal { N } } ^ { i }$ denotes the set of agents in its neighborhood, excluding agent i itself, $a ^ { k }$ is the one-hot encoding<sup>14</sup> of the action of agent k and ${ \bar { a } } ^ { i }$ is the average action of the agents in the neighborhood of agent i.

$$
\bar { a } ^ { i } = \frac { 1 } { | \mathcal { N } ^ { i } | } \sum _ { k \in \mathcal { N } ^ { i } } a ^ { k }\tag{7}
$$

To select its action, agent i uses the mean action and act accordingly. Originally, this mean action was not predicted but observed (Yang et al., 2018), but then, it was proposed to predict the mean action instead (Subramanian et al., 2021).

In some situations, it is not desirable to summarize the actions of all other agents with a single mean field, especially when the agents are heterogeneous. For example, in a team-versus-team system, it may be important to maintain diferent treatments for allies and enemies. Thus, it is possible to consider multiple types of agents. Subramanian et al. (2020) present the Multi-Type MF MARL approach.

## 4.7.2 Learning MOA

To construct the MOA, agents usually use both their local observations and the actions performed by other agents (Jaques et al., 2019; Papoudakis et al., 2021a; Albrecht et al., 2024).

The central questions in policy reconstruction are: (1) what elements in the history of actions should be used, (2) how to use these elements to make predictions (Albrecht and Stone, 2018). This section presents diferent approaches for learning the MOA. The following partition of the approaches draws inspiration from Albrecht and Stone (2018).

Action Frequencies A simple approach to learn a MOA is to base predictions on past action frequencies. In multi-state settings, these frequencies are typically conditioned on the current observation or context. Fictitious Play (FP) is the best-known example of this approach. It was introduced by Brown (1951). In FP, agents assume that others follow a stationary mixed strategy (Berger, 2005), that is, they assume that the strategies of the other agents do not change over time. To predict the actions of others, each agent records the past actions of the others and uses this information to estimate their mixed strategies. For example, in a Rock-Paper-Scissors game, if Agent A has played Rock three times and Scissors once, then Agent B will assign a 75% probability to Rock on the next move and 25% to Scissors. Since it’s introduction by Brown (1951), FP has been used in MARL. The first adaptation of FP to MARL was proposed by Claus and Boutilier (1998), where it was used in conjunction with JAL-AM. More recently, Heinrich and Silver (2016) incorporated FP into a deep learning framework and was used in a poker setting to converge to an approximate Nash equilibrium. However, assuming stationary strategies is restrictive: the faster other agents’ policies are updated, the less valid this assumption becomes. Some approaches still rely on action frequencies, but compute these frequencies using selected portions of the history elements. The main dificulty with this type of MOA training lies in selecting appropriate elements from the history. Using inappropriate or not enough elements may lead to unreliable predictions, whereas using too many elements may be too slow and not eficient (Albrecht and Stone, 2018).

Case-Based Reasoning Policy reconstruction methods based on action frequencies have one major drawback: in order to predict the next action in a given state, they require records of past actions taken by others in that same state. Action frequencies based predic tions lack the ability to extrapolate from past observations to unseen situations (Albrecht and Stone, 2018).

Case-based reasoning (CBR) addresses this limitation by relying on similarity functions. A similarity function defines a distance between observations. Using these functions, the approach retrieves the most similar past observations in order to predict the future actions of others.

CBR was used by Borck et al. (2015) to predict the behavior of hostile and allied agents in the context of air combat. In that study, CBR outperformed the baseline strategy used for comparison.

Compact Model Representations Having to consider a multitude of other agents actions and observations for every action prediction may be too computationally expensive. An alternative to action-frequency-based and CBR predictions is to use a compact machine model from machine learning method for each agent. For instance, one can use neural networks (NN) to model others. The neural networks are trained to make good predictions based on observations of the state and actions of others. Davidson et al. (2000) present agents that used NNs to predict the next action of their opponent in a game of poker, achieving high prediction accuracy.

Machine learning methods ofer greater flexibility and do not require fully observable situations (Albrecht and Stone, 2018).

Type-Based Reasoning Learning a model of an opponent from scratch is a long and dificult process. During one simulation, an agent may not have enough information to correctly model others (and adapt to it). An alternative of online MOA is to train the agent against a variety of diferent agents and to record models of these agents. Then, during the simulation, an agent avoids reconstructing a full model for each agent and instead determines which learned model most closely approximates that agent’s behavior. Type-based reasoning methods consist in identifying and learning models for a set of agent types, and using them to model previously unseen agents from limited observations.

Baker and Cowling (2007) use this method for their agents. This article defines four diferent play styles and the agent uses its observations to determine to which style their opponent resembles most and bases its strategy accordingly. In this example, the diferent types of agents are explicitly defined by the authors. However, it is also possible to learn them. Hou et al. (2022) generate 1000 types of agent, initialized with diverse architectures and trained on historical data from online interactions. Selecting the appropriate type for an opponent among so many possibilities is a dificult task, Hou et al. (2022) propose a solution based on two techniques: (1) the extraction of a small and representative subset of candidates, and (2) the identification of the most appropriate model using multi-armed bandits.

## 4.8 Agent

The agent component plays a central role in a MARL system. Its importance comes from its role as an interface between several core modules, such as the learning algorithm, the policy, and, when present, the communication and MOA modules. As illustrated in Figure 2, agents call their policy to map received observations to actions and provide the resulting experience to the learning algorithm, which updates the policy.

Most MARL approaches do not require agents to perform operations beyond orchestrating their appropriate modules in the appropriate order. However, some approaches require more specialized agent implementations, in which the agent itself carries additional responsibilities. The following subsections first present the default agent implementation, then illustrate how agents can be extended with additional responsibilities through one representative alternative.

## 4.8.1 Default Agent

By default, agents receive observations and reward and redirect this information to other specialized modules, namely the policy and the learning algorithm. Agents can sometimes communicate or model other agents, but this responsibility rely on the dedicated core modules. In this perspective, the agent work more like an orchestrator, aggregating diferent modules and call them appropriately.

This agent perspective is consistent with the definition give in Section 2.5.

## 4.8.2 Multi-Policy Agent

Definition The multi-policy approach consist of agents having diferent policies they use depending on the context. The said policies must be learn separately during diferent training phase and can be use during the execution phase.

Then the multi-policy agents need to define how it select which policy to use depending on the context. Diferent possibilities are presented in the examples below.

Motivation During training, agents learn alongside other agents. Since the best action for an agent depends on the joint action, the optimal policy may depend on the policies of the other agents. This dependency gives rise to the non-stationarity challenge discussed in Section 3.6.1. Consequently, a policy trained with a specific set of agents may not generalize well when paired with diferent agents (Barrett and Stone, 2015; Zheng et al., 2018).

Using a library of policies allow for training with diferent type of agents and have an appropriate policy against one type of agents. After the training phases against diferent type of agents, when the agent face a new agent, it will select the more appropriate policy (Barrett and Stone, 2015). Therefore, the agent do not need to learn a new policy from scratch in order to adapt to the other agents, which is usually a very slow process.

Examples Barrett and Stone (2015) use a multi-policy approach in a cooperative setting, namely RoboCup, where agents within the same team cooperate while competing against an opposing team. In their situation, they use a multi-policy agent equipped with several learned policies because one agent is trained to cooperate with teams exhibiting diferent play styles and must adapt to previously unseen teammates. To select which policy the agent should use, they consider two mechanisms for policy selection. The first one consists of modeling policy selection as a multi-armed bandit problem (Sutton and Barto, 2018). However, this approach is considered too slow in their use case. Instead, they propose PLASTIC-Policy, which selects among the learned policies by updating beliefs over teammate types. PLASTIC-Policy maintains probabilities over the new team being similar to previously observed teams, using the polynomial weights algorithm introduced by Barrett et al. (2013).

Zheng et al. (2018) also use a multi-policy approach but in settings that are not necessarily cooperative. In their situation, other agents are non-stationary and may switch between diferent policies over time. The motivation for maintaining a library of policies comes from the need for fast adaptation to the others’ current policy. To select the appropriate policy, the agent uses an extension of Bayesian Policy Reuse (BPR) (Hernandez-Leal et al., 2016), namely BPR+, designed to handle non-stationary agents; and combines it with deep neural networks, resulting in deep BPR+. Deep BPR+ bases its policy selection on both received cumulative reward and the observed behavior of other agents. In this version of multi-policy approach, the agent does not only rely on a set of policies learned in a training phase and fixed afterward. Here, the agent can detect previously unseen behaviors of other agents, and start a new policy learning process, and add the resulting policy to its library.

Challenges Zheng et al. (2018) motivate their approach as a solution against a nonstationary problem occurring because agents do not have a fixed policy. Using library of diferent policies allow for faster adaptation to other agents behavior, reducing the nonstationary issue.

## 4.9 Evaluation Criteria

Evaluating a MARL system is more complex than evaluating a single-agent RL system (see Section 3.4). In a single-agent RL setting, a solution is better than another if it achieves a higher cumulative reward. Therefore, evaluating a solution is straightforward. In MARL, except in fully cooperative settings (see Section 4.3.1), the objective is not to maximize a single value.

The evaluation of a MARL system can rely on a wide range of experiment-specific measures, depending on the objectives of the experiment and the dynamics of the environment. Such measures may include, for example, the number of prey captured (for a prey/hunter experiment), the time required to complete a collective task, or the level of competition between agents for limited resources. Beyond these experiment-specific measures, general solution concepts provide criteria that can be applied across diferent MARL settings. This section focuses on such concepts, which formalize how the outcomes of multi-agent inter actions can be evaluated. The presentation draws inspiration from Albrecht et al. (2024).

Bu¸soniu et al. (2008) identify two main types of learning objectives in MARL: stabil ity and adaptability. Stability refers to a system’s ability to converge to a stable state. Adaptability refers to an agent’s ability to adapt its strategy as the other agents learn. The concepts presented below can be linked to these two objectives.

## 4.9.1 Stability

Convergence Convergence is a criterion related to stability. A MARL algorithm is said to converge if the agents’ policies converge to a stationary policy (Bowling and Veloso, 2002).

Analyzing convergent system is easier by looking at the final policies of the agents. This criterion is often sought after in MARL applications (Bowling and Veloso, 2002; Conitzer and Sandholm, 2007).

Opponent Independent Solution An another stability criteria for the agent is to have a learning process that is independent of the others’ actions. The motivation of such criteria is to have a learning process which stability does not depend on the others algorithms (Bu¸soniu et al., 2008; Bowling and Veloso, 2000).

For instance, Littman (1994) uses a Minimax-Q algorithm to learn for its agents. The agents learn a policy that is optimal in the worst case scenario. No matter what the others do, agents using Minimax-Q will learn a policy that converges.

## 4.9.2 Adaptability

Best Response A best response is a policy that maximizes an agent’s expected reward, given the policies of the other agents. An agent’s best response can be defined mathematically using the equation 8.

$$
B R _ { i } ( \pi _ { - i } ) = \arg \operatorname* { m a x } _ { \pi _ { i } } \mathbb { E } [ G _ { i } | \pi _ { i } , \pi _ { - i } ]\tag{8}
$$

Where $G _ { i }$ is the return (cumulative reward) of agent $i , \pi _ { i }$ is the policy of agent i and $\pi _ { - i }$ is the joint policy of all agents except agent i.

A best response does not have to be unique. An agent can have multiple best responses to the same joint policy of the other agents.

Here’s an example that illustrates the concept of best response. Consider a simple two-player game: Rock-Paper-Scissors. Assume that player A always plays Rock. Given player A’s policy, the best response for player B would be to always play Paper. However, if Player A’s policy changes to playing randomly, then playing Rock or playing randomly would both be best responses.

No-regret Regret is a measure and is defined as the diference between the rewards an agent could have received by playing the optimal policy and the rewards it actually receives during an episode. It can be mathematically defined with the equation 9.

$$
R e g r e t _ { i } ( T ) = \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { i } } \sum _ { t = 1 } ^ { T } \left( \mathbb { E } [ r _ { i } ^ { t } \mid \pi _ { i } , \pi _ { - i } ^ { t } ] - \mathbb { E } [ r _ { i } ^ { t } \mid \pi _ { i } ^ { t } , \pi _ { - i } ^ { t } ] \right)\tag{9}
$$

Where $T$ is the number of learning steps, t is a step of learning, $r _ { i } ^ { t }$ is the reward of agent i at time $t , \pi _ { i } ^ { t }$ is the policy of agent i at time $t , \pi _ { - i } ^ { t }$ is the joint policy of all agents except agent i at time $t ,$ and $\Pi _ { i }$ is the set of all policies of agent i.

An algorithm is said to be no-regret if for all agents i:

$$
\operatorname* { l i m } _ { T  \infty } \frac { 1 } { T } R e g r e t _ { i } ( T ) = 0\tag{10}
$$

The no-regret approach has a significant limitation because it evaluates an agent’s performance based on the observed policies of other agents. However, since the other agents are also learning, their policies depend on the agent’s own policy. Therefore, regret cannot predict what could have happened under diferent circumstances. Additionally,

Albrecht et al. also argue that no-regret does not necessarily equate to maximized rewards. The prisoner’s dilemma is a simple example of this: if both agents defect, the outcome is a no-regret situation, yet both agents could have achieved a higher reward by cooperating (Albrecht et al., 2024).

## 4.9.3 E<sub>q</sub>uilibrium

Equilibrium is a concept that combines both stability and adaptability. The objective is to converge to a stable and rational solution. The motivation is the combined the benefits of both stability and adaptability.

There are diferent types of equilibrium, the most common ones are Nash equilibrium and Pareto optimality, which are presented below.

Nash Equilibrium In MARL, the dynamics of learning may converge to an equilibrium. Convergence is important because it provides a basis for evaluating the outcomes that emerge from their interactions. However, not all equilibria are equally desirable. A widely used concept to evaluate the outcomes of multi-agent interactions is the Nash equilibrium (NE) (Nash, 1951).

A Nash equilibrium occurs when each agent simultaneously adopts the best possible policy, given the choices made by the others. This term can be defined mathematically: $\pi ^ { * }$ is a Nash equilibrium if and only if:

$$
V _ { \pi _ { i } ^ { * } , \pi _ { - i } ^ { * } } ^ { i } ( s ) \geq V _ { \pi _ { i } , \pi _ { - i } ^ { * } } ^ { i } ( s ) , \forall \pi _ { i } \in \Pi _ { i } , \forall s \in S , \forall i\tag{11}
$$

where $\pi ^ { i }$ is the policy of agent $i , \pi ^ { - i }$ the policy of agents other than i, $\pi ^ { * } = \{ \pi ^ { * , i } , \pi ^ { * , - i } \}$ is the joint policies of all agents and s the state of the environment.

There are two types of Nash equilibrium, depending on whether we consider only deterministic policies (no probabilistic choice, $\pi ( s , a ) \in \{ 0 , 1 \} )$ or also stochastic policies $( \pi ( s , a ) \in [ 0 , 1 ] )$ . In the case of a prisoner’s dilemma, there is a deterministic policy that constitutes a Nash equilibrium: both players defect. In other cases, such as Rock-Paper-Scissors, no deterministic policy satisfies the Nash equilibrium condition. However, in this case, there is a stochastic policy that forms a Nash equilibrium: both players randomly select each option with equal probability.

More generally, the Nash existence theorem (Nash, 1951) ensures the existence of at least one Nash equilibrium under the following conditions:

• Stochastic policies are allowed;

• The number of players is finite;

• The number of actions of each player is finite.

In practice, it can be dificult to reach a Nash equilibrium. In these dificult cases, an approximation of the Nash equilibrium, the ϵ-Nash equilibrium $( \epsilon > 0 )$ , can be used. It is defined almost the same way as the Nash equilibrium, except that each agent’s policy can be slightly suboptimal. The greater ϵ is, the more suboptimal the policies can be. $\pi ^ { * }$ is an ϵ-Nash equilibrium if and only if:

$$
V _ { \pi _ { i } ^ { * } , \pi _ { - i } ^ { * } } ^ { i } ( s ) \geq V _ { \pi _ { i } , \pi _ { - i } ^ { * } } ^ { i } ( s ) - \epsilon , \forall \pi _ { i } \in \Pi _ { i } , \forall s \in { \cal S } , \forall i\tag{12}
$$

Pareto Optimality As mentioned previously in the Nash equilibrium Section 4.9.3, equilibria are not equal and can have diferent properties. Another one interesting property is Pareto optimality.

A joint policy $\pi ^ { \prime }$ is Pareto dominant over $\pi$ if and only if:

$$
\forall i : U _ { i } ( \pi ^ { \prime } ) \geq U _ { i } ( \pi ) , \ \mathrm { a n d } \ \exists i : U _ { i } ( \pi ^ { \prime } ) > U _ { i } ( \pi )\tag{13}
$$

where $U _ { i } ( \pi )$ is the expected cumulative reward of agent i when all agents follow the joint policy π.

A joint policy $\pi ^ { * }$ is Pareto optimal if and only if no other joint policy Pareto dominates it (Albrecht et al., 2024). In other words, an equilibrium is Pareto optimal if and only if it is impossible to increase one agent’s reward without decreasing at least one other agent’s reward.

In fully competitive settings, the concept of Pareto optimality is meaningless because all joint policies are Pareto optimal. Efectively, because the sum of all agents’ rewards is constant, increasing one agent’s reward implies decreasing another’s. Therefore all equilibria are Pareto optimal.

Limitations of Equilibrium Solutions The equilibrium solution concepts presented above have two important limitations. First, convergence to these equilibria does not guarantee maximized rewards. They can be suboptimal for all agents. For instance, in the prisoner’s dilemma, both players defecting is the only Nash equilibrium; however, both players would have received a greater reward if they had cooperated. This is a limitation shared with the no-regret concept presented above. In the Prisoner’s Dilemma, there are three Pareto equilibria: both players cooperate, or one cooperates while the other defects. The first one is optimal for both players, while the other two are suboptimal for one of them.

This introduces the second limitation: a system can have multiple equilibria of the same type. If an algorithm is designed to guide agents toward strategies that lead to an equilibrium, then the agents should be able to select which equilibrium to converge to. This problem is known as equilibrium selection (Albrecht et al., 2024).

## 4.9.4 Fairness

One important aspect that is not captured by the previous solution concepts is fairness. Fairness is a measure of how equitably the rewards are distributed among agents. Albrecht et al. (2024) define the fairness of a joint policy π as follows:

$$
F a i r n e s s ( { \pi } ) = \prod _ { i = 1 } ^ { N } U _ { i } ( { \pi } )\tag{14}
$$

Where $U _ { i } ( \pmb { \pi } )$ is the expected cumulative reward of agent $i , N$ is the number of agents and $\pi$ is the joint policy of all agents.

If one agent receives a low reward, it will significantly reduce the overall fairness. This measure encourages fairer distribution of rewards among agents. A joint policy π that satisfies $\pmb { \pi } = \arg \operatorname* { m a x } _ { \pmb { \pi } ^ { \prime } } F a i r n e s s ( \pmb { \pi } ^ { \prime } )$ is said to be fairness optimal (Albrecht et al., 2024).

This fairness function is related to performance. For example, consider a two-player game and two joint policies π and $\pi ^ { \prime }$ where π leads to $U _ { 1 } ( \pmb { \pi } ) = 1 0 , U _ { 2 } ( \pmb { \pi } ) = 1$ and $\pi ^ { \prime }$ leads to $U _ { 1 } ( { \pmb \pi } ^ { \prime } ) = 3 , U _ { 2 } ( { \pmb \pi } ^ { \prime } ) = 3$ . In this example, $F a i r n e s s ( \pi ) = 1 0 > F a i r n e s s ( \pi ^ { \prime } ) = 9$ This fairness function encourages more equitable distribution of rewards while considering the system’s overall performance. This definition of fairness is one possibility, but other definitions could be used that give more importance to equity.

The concept of fairness is irrelevant in fully cooperative settings, since all agents share the same reward function. Under this definition of the fairness function 14, maximizing fairness is equivalent to maximizing the global reward.

## 4.10 Modular Perspective on Other Papers’ Approaches

To illustrate the usefulness of this categorization into core modules, we analyze three recent MARL articles and describe their proposed approaches from this modular perspective.

## 4.10.1 MADDPG

Lowe et al. (2017) introduce MADDPG, a MARL actor-critic approach designed for mixed cooperative-competitive environments. The main motivation is to address two dificulties of applying standard RL algorithms to MARL: the non-stationarity induced by simultaneously learning agents, and the high variance of policy-gradient methods when rewards depend on the actions of several agents.

Several experiments are conducted in diferent environments. Since the system configuration difers across experiments, this analysis focuses on the cooperative navigation experiment using MADDPG.

MADDPG follows an actor-critic approach based on deep deterministic policy gradient (DDPG) and combines it with a CTDE paradigm, where each agent uses a centralized critic during training and a decentralized actor during execution.

Agents have to cover all landmarks and are collectively rewarded based on the proximity of any agent to each landmark. They are also penalized when they collide with each other. Thus, the reward model encourages agents to distribute themselves across landmarks while avoiding physical conflicts. The reward model is cooperative. However, the article leaves a slight ambiguity as to whether the reward is fully shared. The landmark-coverage component is explicitly described as collective, but it is unclear whether the collision penalty applies to all agents or only to those involved in a collision. The released code shows that the penalty is first assigned to the colliding agents and then collectivized, making the reward model fully cooperative.

This experiment does not rely on an explicit communication mechanism. Agents coordinate through their physical actions and through the information available in their observations, which include the relative positions of other agents and landmarks.

MADDPG requires access to the other agents’ policies, or learned approximations of them, to obtain their actions during training. In the default setting, these models are not learned: each agent has direct access to the policies of the other agents. The paper presents this requirement as closely tied to the use of a centralized critic, although it is more accurately understood as a consequence of MADDPG’s specific critic formulation. Other applications of centralized critics, such as COMA, do not impose the same requirement (Foerster et al., 2018). While the paper considers policy inference as an alternative when the true policies are unavailable, this mechanism is presented as an additional extension rather than as the main configuration of the cooperative navigation experiment. We believe that the distinction between centralized critics and explicit modeling of other agents’ policies should therefore have been made more prominent.

The agent module follows a standard orchestration role: it calls the appropriate internal modules when needed, such as the policy for action selection and the learning algorithm for policy updates.

The system is evaluated by considering how well agents cover the landmarks and avoid collisions. Lowe et al. (2017) report in particular the average distance to each landmark and the average number of collisions per episode when comparing MADDPG with DDPG.

## 4.10.2 WoLF-PHC

Yang et al. (2023) study how Virtual Power Plants (VPPs) can define bidding strategies in electricity markets. The main motivation is to obtain optimal bidding strategies and market equilibrium results when VPPs participate alongside traditional power plants.

The environment corresponds to a power market game. Agents are power producers, either traditional power plants or VPPs.

The reward model is mixed. Each agent has its own objective and reward, corresponding to its own profit. Rewards are not shared among agents.

The learning module is based on WoLF-PHC, which combines Policy Hill Climbing with the Win or Learn Fast mechanism. In order to avoid local optima and improve convergence speed, Yang et al. (2023) use WoLF, which adjusts the learning rate based on performance. Each agent maintains its own Q-table, updated based on its own observations, actions, and rewards.

The training and execution mode corresponds to DTE: agents learn independently and no centralized training or execution mechanism is introduced. The paper does not describe any explicit communication mechanism between agents, nor any MOA strategy. Other power producers are therefore treated as part of the environment. The agent module does not introduce any specific agent-level mechanism: each agent simply selects a bidding strategy according to its current policy and updates this policy through WoLF-PHC.

The system evaluation is mainly based on two market outcomes: power output and revenue. Yang et al. (2023) compare these quantities in two scenarios: one with only traditional power plants, and one where one traditional power plant is replaced by a VPP.

## 4.10.3 NC-HDQN

During epidemic episodes, trafic flow patterns can change significantly, with heavier trafic near hospitals and lighter trafic in controlled epidemic communities. To optimize trafic signal control in this context, Zhang et al. (2022) propose NC-HDQN, a neighborhood cooperative MARL approach for adaptive trafic signal control. The term NC-HDQN encompasses various pieces of information regarding specific, distinct modules. We will go into detail to explain it.

The environment corresponds to a trafic network modeled as a networked Markov game. Each intersection is modeled as an agent, and actions correspond to trafic signal phases. Agents use local observations and information from physically connected neighboring intersections.

The reward model is cooperative, but not fully cooperative. The objective is to optimize regional trafic conditions, but each agent receives a reward based on its local traffic situation and the rewards of its neighbors, weighted by correlation coeficients. The reward model encourages cooperation while preserving neighborhood-level diferences between agents.

Communication is present through neighborhood information sharing. Agents receive observation information from neighboring agents. The network structure is fixed by the road network and the received information is weighted using a correlation mechanism.

The learning module is based on Hysteretic DQN (HDQN). DQN provides the basic value-based architecture, while HDQN introduces diferent update rates for positive and negative TD errors. Zhang et al. (2022) justify this choice by the limitations of independent learners in cooperative MARL: because other agents are simultaneously learning and exploring, an agent may observe poor outcomes even when its own action is beneficial. Hysteretic updates reduce the impact of such negative experiences and give more importance to positive ones, which helps agents avoid suboptimal cooperative strategies. To use the information provided by the communication module, Zhang et al. (2022) propose

Neighborhood Cooperative HDQN (NC-HDQN), which weights neighboring observations and rewards according to correlation coeficients.

The training and execution mode corresponds to DTE. Although CTDE is common in MARL, Zhang et al. (2022) argue that the large number of intersections in ATSC makes centralized training impractical. Agents are therefore trained and executed independently. Agents do not model others. The agent module follows the default orchestration role described in Section 4.8.

The system is evaluated during execution using three measures: average travel time, average queue length, and the number of vehicles arriving at their destination.

## 4.10.4 Summary of the Modular Analysis

We summarize our analysis with a core module table (see Table 2). This table illustrates the process of applying the modular perspective to existing MARL approaches by making explicit the choices associated with each core module.

<table><tr><td>Core Module</td><td>Lowe et al. (2017)</td><td>Yang et al. (2023)</td><td>Zhang et al. (2022)</td></tr><tr><td>Environment</td><td>Cooperative navigation</td><td>Power market game</td><td>Networked traffic signal control</td></tr><tr><td>Agent</td><td>Default</td><td>Default</td><td>Default</td></tr><tr><td>Learning</td><td>DDPG</td><td>WoLF-PHC</td><td>HDQN</td></tr><tr><td>Training and Execution Modes</td><td>CTDE (Centralized Critic)</td><td>DTE</td><td>DTE</td></tr><tr><td>Reward Modeling</td><td>Fully cooperative</td><td>Mixed</td><td>Neighborhood Cooperative (NC)</td></tr><tr><td>Communication</td><td>None</td><td>None</td><td>Networked. Observation sharing</td></tr><tr><td>MOA</td><td>Other Policies Directly Available</td><td>None</td><td>None</td></tr><tr><td>Evaluation Criteria</td><td>Landmark distance + collisions</td><td>Power output + revenue</td><td>Travel time + queue length + throughput</td></tr></table>

Table 2: Modular perspective on selected MARL approaches

This analysis shows that substantially diferent approaches can be described using the same decomposition into core modules. By separating the core modules and making their respective implementations explicit, this perspective provides a clearer view of each proposed solution. It also helps identify the specific contribution of a new approach by making explicit which components have changed compared with existing approaches. In addition to describing the proposed solution, the decomposition specifies the context in which it was tested: the task being addressed, the reward model used, and the evaluation criteria used for the solution.

For instance, Lowe et al. (2017) extend DDPG by using centralized critics during training. This approach is evaluated in diferent experiments, including a fully cooperative navigation task where distance to landmarks and collisions were the main evaluation criteria. The strategy proposed by Yang et al. (2023) was to modify how a standard PHC learning algorithm updates its policy by using WoLF, an adaptive learning-rate mechanism.

Overall, such a decomposition provides a clearer view of a proposed solution, and helps identify which components could be modified to explore alternative configurations. It ofers a common structure for presenting MARL approaches, making it easier to identify the key diferences between solutions and between the experimental configurations in which they are evaluated.

## 5 Analysis of MARL Tools

Having established the theoretical relevance of the modular perspective in the previous section, we now examine existing MARL tools through this perspective. We first describe their general organization and main functionalities, before analyzing how they represent the core modules identified in this paper and to what extent they support varying their implementations independently.

Several software tools have been proposed to support MARL experimentation, but they do not all address the same level of abstraction. In the following, we focus on a representative set of widely used tools: PettingZoo (Terry et al., 2021), OpenSpiel (Lanctot et al., 2019), RLlib (Liang et al., 2018), MARLlib (Hu et al., 2023), and EPyMARL (Papoudakis et al., 2021b). This selection is not intended to be exhaustive. Rather, it covers complementary roles in the MARL ecosystem, especially environment standardization and MARL experimentation with diferent algorithms. This diversity makes these tools useful for assessing how current MARL software supports, or fails to support, the explicit modular exploration of MARL implementations.

## 5.1 Tools Rationales

Among MARL tools, some are especially useful for defining and accessing multi-agent environments. PettingZoo provides a multi-agent alternative to Gymnasium environments, with standardized interfaces that make environments easier to reuse across diferent learning frameworks.

OpenSpiel also provides a large collection of multi-agent environments, but its scope is broader: it is designed around game-theoretic modeling and includes both environments and algorithms for studying diferent classes of games, as well as tools for analyzing learning dynamics and other evaluation metrics.

RLlib provides a general scalable reinforcement learning framework with multi-agent support. RLlib can run multiple copies of the environment in parallel to collect training data faster. RLlib also supports flexible mappings between agents and policies, which makes it possible to implement algorithm self-play, policy self-play, or mixed-play configurations (see Section 3.5.1). In addition, for components that rely on neural networks, such as policies and critics, RLlib allows the underlying neural architecture to be configured or replaced without modifying the surrounding execution pipeline, provided that the new architecture satisfies the interface and outputs expected by the learning algorithm.

MARLlib builds on top of RLlib<sup>15</sup> and provides a higher-level interface dedicated to MARL experimentation. Its primary rationale is to retain RLlib’s scalable execution infrastructure while providing a more structured and accessible framework for multi-agent tasks. MARLlib organizes supported algorithms into three groups: independent learning, centralized critic, and value decomposition. This allows algorithms within the same group to base their learning on common data structures. The data consumed by MARLlib is harmonized rather than being directly based on the heterogeneous structures exposed by external environments. Because some algorithms are not suited to certain task settings, MARLlib checks the suitability of the selected algorithm for the task. Its flexible policymapping strategy allows policies and their parameters to be shared by all agents, shared within groups, or kept separate for each agent.

EPyMARL also focuses on MARL experimentation, with an emphasis on comparing algorithms across environments coming from diferent sources, such as PettingZoo, Gymnasium, and SMAC. It defines a common multi-agent environment interface and relies on wrappers to use environments from these diferent sources. Building on PyMARL<sup>16</sup>, EPy-MARL extends the available set of algorithms and provides configurable implementation choices, such as parameter sharing, reward standardization, and network architecture<sup>17</sup>. As noted by Papoudakis et al. (2021b), implementation details can significantly afect algorithm performance. Exposing these choices facilitates their controlled and consistent evaluation.

## 5.2 Analysis of MARL Tools Through the Core Module Perspective

In this section, we analyze the MARL tools from the perspective of our core modules. We also examine the extent to which their architectures follow the single-responsibility principle.

## 5.2.1 Environment

These tools all provide interchangeable and extensible environment abstractions and therefore have modular implementation of the environment.

PettingZoo defines diferent types of environment, and is designed to facilitate the integration of diferent multi-agent environments with external training frameworks.

RLlib defines a common multi-agent environment interface through which the rest of its multi-agent pipeline interacts with environments. Although RLlib provides some of its own environments, its design primarily supports custom environments and the adaptation of environments from external tools, such as Gymnasium, PettingZoo, and OpenSpiel, through wrappers.

MARLlib works in a similar way to RLlib and relies on the same multi-agent environment interface. However, unlike RLlib, which uses one generic environment wrapper for each external tool (PettingZoo, OpenSpiel, etc.), MARLlib generally wraps each environment suite separately.

EPyMARL also defines a common multi-agent environment interface and provides wrappers that standardize access to environments from external sources, such as SMAC<sup>18</sup> and PettingZoo. Unlike RLlib, EPyMARL does not implement environment dynamics and relies on wrapped external environments.

OpenSpiel directly implements a large collection of games and exposes them through a common reinforcement learning interface. A general component orchestrates the interaction with the selected game and exposes it through a reinforcement learning interface. The game describes its general properties, while its state implementation defines how actions modify the current state and provides the resulting rewards.

Although these tools provide modular and interchangeable environment implementations, they do not separate the responsibilities identified in our modular perspective. In particular, the same environment implementation generally defines both how the system reacts to the agents’ actions and how rewards are computed. It therefore combines the Reaction and Reward modules within the same component and does not respect the singleresponsibility principle. This coupling leads to the issues discussed in the following section.

## 5.2.2 Reward

In these tools, rewards are computed by the environment<sup>19</sup>. Reward modeling is not exposed as an independent, reusable module that can transform, share, or redistribute rewards across diferent environments.

Consequently, testing the same environment dynamics under a diferent reward structure, for example changing from individual rewards to a fully cooperative reward, usually requires modifying or wrapping the environment separately for each experiment. An independent reward-modeling module would instead allow researchers to reuse reward structures across environments and compare diferent reward models while keeping the environment dynamics unchanged.

MARLlib illustrates this lack of modularity particularly well. To support fully cooperative rewards across diferent environments, it defines separate cooperative versions of these environments, each overriding the original reward logic in the step method. Supporting another reward model would similarly require implementing environment-specific variants rather than adding a single reusable reward component.

## 5.2.3 Training and Execution Modes

During training, execution, or both, a centralized process may influence how agents learn or behave. A common centralized technique combines an actor critic learning algorithm with a centralized critic. Other common techniques include value decomposition and parameter sharing managed by a centralized component.

OpenSpiel does not provide dedicated support for any centralized technique. Using them requires implementing their logic as part of a specific algorithm.

PettingZoo is an environment tool and does not implement training or execution modes. This core module is therefore outside its scope.

RLlib directly supports parameter sharing as a centralized training approach: multiple agents can be mapped to the same policy module and therefore trained using a shared set of parameters. Agents can alternatively be mapped to separate policy instances with distinct parameters. Other centralized training techniques, such as centralized critics and value decomposition, are not supported through generic, reusable components. RLlib provides specific implementations of these techniques, but their centralized training logic is embedded in these implementations and must be adapted or reimplemented for use with other algorithms.

EPyMARL directly supports parameter sharing as a centralized approach. It also provides modular support for centralized critics and value decomposition within compatible algorithm families. For instance, the same Q-learning learner can use diferent value decomposition mixers.

MARLlib distinguishes three families of algorithms: (1) independent learning, (2) centralized critic, and (3) value decomposition. Its modular organization allows compatible learning algorithms to reuse centralized critic or value decomposition mechanisms instead of implementing the complete centralized training pipeline independently for each algorithm. MARLlib also directly supports diferent parameter sharing strategies. Policy parameters can be shared among all agents, shared within groups of agents, kept separate for each agent, or assigned through a custom policy-mapping strategy (Hu et al., 2023).

## 5.2.4 Learning

The learning module is implemented as an independent component across these tools, except PettingZoo which is not concerned. It is possible to test diferent learning algorithm and policy on diferent environments. However, the two sub components: learning algorithm and policy are not always separated. These two components cannot be completely independent, however, if they are separated enough, it allows for learning algorithms to be used with diferent policies without reimplementing everything. For example, a Q-table based policy could be updated using learning algorithms such as SARSA or Q-learning.

OpenSpiel does not systematically separate these two components. It does not provide dedicated abstractions for policies and learning algorithms that would allow compatible implementations to be combined independently. Moreover, in some algorithms, the same component is responsible for both selecting actions and updating the action selection process.

RLlib has two specific components for the learning algorithm and the policy. It allows for using learners with diferent policies that are compatibles.

MARLlib separates policy networks from optimization code, but it does not expose policies and learning algorithms as two fully independent components. This is partly due to its reliance on an older version of RLlib. In contrast, the redesign of RLlib explicitly aimed at improving the separation of concerns through a more modular architecture.

EPyMARL separates these two components. However, the policy itself is distributed across two components: a centralized controller that calls the agents’ neural network and converts its outputs into actions, and the neural network that computes the outputs used for action selection.

## 5.2.5 Agent

Having an agent component that manages policies, learning, and potentially other functions, such as communication, enables the implementation of approaches in which the agent itself performs additional operations before selecting an action or initiating learning (see Section 4.8). It also prevents other components from being overloaded with additional responsibilities. However, most of the analyzed tools provide limited support for the agent as an independent core module.

In PettingZoo, agents are represented only by identifiers within the environment.

RLlib and MARLlib directly associate agent identifiers with policies, without introducing an intermediate agent component responsible for coordinating the decision process.

EPyMARL instead uses a centralized multi-agent controller that manages action selection for all agents. Consequently, it does not provide each agent with an individual orchestration component that can be extended independently.

OpenSpiel provides a more explicit agent abstraction: the execution loop delegates action selection to the Agent. This design makes it possible to extend the Agent with additional behaviors before returning the final action, for example by selecting among several policies or applying a specialized decision mechanism.

## 5.2.6 Communication

Communication requires messages to be generated, transmitted, and received by agents. Even when communication is available in the analyzed tools, it is not handled by an independent module that can be activated or deactivated as needed.

Some environments provided by PettingZoo include communication capabilities. Agents generate messages during action selection, either as their complete action or as one component of it, while the environment transmits these messages to the receiving agents through their observations.

OpenSpiel also provides games involving communication. The communication setting is defined by the game itself, and communicating generally constitutes an action of the agent rather than a secondary protocol executed in parallel with its action.

RLlib, MARLlib, and EPyMARL do not provide any native implementation or example of an agent communication system. This absence does not mean that communication cannot be implemented using these tools. For example, an external extension of RLlib<sup>20</sup> has been developed to implement a diferentiable communication channel between agents, demonstrating that such mechanisms can be added despite not being natively supported (Blumenkamp and Prorok, 2021).

## 5.2.7 MOA

MOA is poorly represented in the MARL tools considered above. Some specialized MARL tools exist, such as LeMOL, but MOA has not been generalized within the main MARL frameworks.

OpenSpiel provides limited support for MOA through methods such as best response and minimax. However, these mechanisms are tied to specific algorithms and require the agent to have access to a model of the environment dynamics<sup>21</sup>.

## 5.2.8 System Evaluator

The analyzed tools provide diferent levels of support for system evaluation.

PettingZoo does not include a dedicated evaluation system, which is consistent with its role as an environment API rather than a complete MARL framework.

OpenSpiel provides several evaluation functions, particularly for game-theoretic criteria such as Nash equilibrium (see Section 4.9).

RLlib and MARLlib provide more flexible support for experiment-specific evaluation. Users can define metrics beyond rewards and returns by processing information exposed by the environment or generated during training. This flexibility makes system evaluation a well-supported core module in these frameworks.

EPyMARL follows a similar approach and can automatically record statistics reported by the environment. However, computing more complex metrics across multiple steps may require modifying the environment or the execution scheduler, notably the ParallelRunner.

## 5.3 BenchMARL

The preceding analysis examined existing MARL tools from the perspective of the core modules identified in this paper. However, other studies have highlighted complementary limitations of current MARL tools and experimental practices. In particular, recent work has emphasized the need for more standardized and reproducible evaluation protocols in MARL (Gorsane et al., 2022). BenchMARL was introduced as a framework specifically designed to address this problem (Bettini et al., 2024). It enables users to systematically compare diferent combinations of tasks, algorithms, and models within a common experimental pipeline. This represents an important step towards better experimental practices in MARL experiments.

BenchMARL itself adopts a modular perspective by distinguishing several components of a MARL experiment, including tasks, algorithms, models, experiment configurations, and benchmark. A task is an environment scenario that defines the MARL problem to be solved. An algorithm defines all the components that determine the training strategy. A model defines the neural network architecture used by components such as the policy for action selection or the critic for value estimation. An experiment is a training run in which the task, algorithm and model are fixed. Finally, a benchmark component corresponds to a collection of experiments.

However, this modular decomposition does not explicitly expose several of the core modules identified in this paper, such as communication, MOA, reward modeling, or training and execution modes, despite their importance in the design of MARL systems. These mechanisms consequently remain integrated into the components defined by BenchMARL rather than being independently configurable.

A finer decomposition of such approaches would make it easier to study the contribution of these specific modules in isolation. Such a decomposition would make it possible to test an algorithm while independently varying the modules presented in Section 4.

## 6 MARLKIT

Having examined the extent to which existing MARL tools support the core module perspective, we now examine the practical implications of the proposed decomposition through its implementation in our MARL framework, MARLKIT.

MARLKIT is a Java-based MARL framework that supports the exploration and comparison of MARL methods through experiments involving diferent implementations of the core modules. The modules presented in Section 4 are explicitly represented in MARLKIT, making it possible to add new implementations of a module and use them with the rest of the framework. Figure 7 illustrates the core modules and how they interact in MARLKIT.

MARLKIT was designed according to two main objectives. First, its architecture should be organized around the core modules, exposing them as first-class components. Second, new module implementations should be easy to create, interchangeable with existing implementations, and reusable. Particular attention was also paid to the single responsibility principle, so that each class has a clearly identified responsibility.

MARLKIT serves as a proof of concept illustrating the practical benefits of structuring MARL tools around the proposed core module decomposition. Examining how easily module implementations can be added, exchanged, and compared across experiments provides a practical indication of the relevance of the proposed decomposition. In particular, it enables controlled comparisons in which the implementation of a specific module is varied while the remaining experimental configuration is kept unchanged.

Experiments varying representative choices across the core modules provide an initial assessment of how the framework accommodates distinct MARL configurations.

In MARLKIT, each component has a clearly separated responsibility. The framework relies on MaDKit for the simulation and execution of the multi-agent system.

MARLKIT was implemented in Java, despite most MARL tools being python-based. Java ofers a stricter class-based architecture with more constraints that help ensure certain design rules are followed. The main purpose of our methodology and decomposition is to provide a clearer and more precisely defined view of concepts such as core modules. In our Java implementation, these concepts are represented by dedicated packages and classes, making their respective responsibilities, operations, and components more explicit. Python bindings could be developed to make the project accessible from Python for the MARL community. It should also be noted that MARLKIT is primarily a proof of concept. The proposed modular approach is not language-specific and could therefore be implemented as a Python-based framework.

![](images/c488a9543e716fc9870cdf07d2a89d26cb0ebb701a377b3e7092b049d84fc546.jpg)  
Figure 7: MARLKIT core modules.

The scheduler (MLKScheduler) orchestrates the overall MARL interaction loop by coordinating calls to the environment, the agents and their associated update methods in a predefined sequence. By default, it enforces a fixed observation–influence-reaction–reward–learning cycle. It can be redefined partly or completely.

In the following sections, we describe how the core modules are implemented in MAR-LKIT and illustrate their use in experimental settings. The purpose of these experiments is not to achieve state-of-the-art performance, but to examine the decomposition in practice.

However, exploring alternative solutions in realistic use cases often requires considering multiple implementations across several core modules. Section 6.9 describes how MARLKIT supports batches of simulations involving diferent combinations of core module implementations, and how the resulting performances can be compared.

## 6.1 Agent

An agent performs actions and influences the environment. MLKAgent defines the methods required for an agent to participate in the MARL learning loop.

These methods allow the agent to orchestrate its internal core modules. In particular, the agent calls its learning components, such as the policy and the learning algorithm, to select actions and update its behavior from experience. Depending on the considered implementation, the agent may also orchestrate other core modules, such as communication and MOA.

AgentStandard is a default implementation of MLKAgent. It implements the standard methods of the MARL learning loop by delegating action selection and learning updates to the learning module, without introducing additional agent-level mechanisms.

## 6.2 Environment

In MARLKIT, the environment manages the states and dynamics of the system. The implementation of the transition function respect the Influence/Reaction paradigm defined in Section 4.2. It receives the agents’ influences (that is joint action) and updates its state according to the reaction.

![](images/2c6b641d7543b11482f556a916d984ab686689aa2e25f11306b68b2052c5902d.jpg)  
Figure 8: Production of the rewards

An environment is represented by the MLKEnvironment interface, which defines the core methods required by the MLKScheduler. MARLKIT provides the abstract class EnvironmentStandard, which implements common methods while leaving the experimentspecific components, namely the states and dynamics, to the user.

## 6.2.1 State

The State represents the current configuration of the environment. It provides access to diferent types of data, such as numerical values, and object positions.

The implementation of State is expected to vary across experiments. However, a common type of state representation is a 2D grid. MARLKIT provides two ready-to-use implementation of State: State2DGridInt, which stores integer values in a 2D grid and records agent positions.

## 6.2.2 Reaction Event

The ReactionEvents are produced during the reaction phase (2) (see Figure 4).

ReactionEvents represent all meaningful events that occur in the environment as a result of the agents’ actions.

After agents take actions and the environment transitions to a new state, the latter generates a set of ReactionEvents for each agent. These ReactionEvents represent all events that may influence the reward of the agents or the evaluation of the system. For example, in a foraging environment, a ReactionEvent can be ”You collected a resource” or ”You moved”.

These ReactionEvents are not computed independently for each agent, it depends on the joint action and the state transition. For instance, if in the foraging environment, two agents are needed for collecting a resource, then they will receive the ReactionEvent ”You collected a resource” only if the other agent also take the action needed for collecting the resource.

ReactionEvents are used by the reward model to compute the rewards returned to each agent (see Section 6.3) and by the system evaluation module to compute diferent evaluation measures (see Section 6.8).

## 6.3 Reward Modeling

Reward modeling is an independent core module, used by the environment to compute rewards during the reward phase (3) (Figure 4).

The reward model has reaction events as input and a final reward vector $( r _ { 1 } , \ldots , r _ { N } )$ as output, where $r _ { i }$ is the reward received by an agent i.

## 6.3.1 Reward Notation

In standard RL, the reward is typically represented as a single scalar value. To support the modular architecture design, we introduce three reward-related quantities. This design allows generic reward models to operate on reward contributions without depending on environment-specific details.

Each reaction event e carries a default event reward $\rho ( e ) . \ \rho ( e )$ represents an eventlevel reward contribution and serves as an abstract intermediary between the environment and reward models.

For each agent $i ,$ the agent-level raw reward $\bar { r } _ { i }$ is obtained by summing the default event rewards associated with that agent. Reward models based on reward redistribution can manipulate directly this concept in order to produce the final reward.

The final reward $r _ { i }$ is the reward actually returned to agent i after applying the RewardModel.

The three reward model presented in Section 4.3 are good examples of reward models that do not depend on environment-specific details. Their implementations are discussed in the following section.

## 6.3.2 Reward Model

The RewardModel is a module that is responsible for computing the reward of each agent. It maps the generated ReactionEvents to the final reward received by each agent (see Figure 8). It may aggregate the default event rewards associated with each agent, redistribute rewards across agents, or transform them using additional information from the transition.

Depending on the RewardModel used, the reward can be computed in diferent ways. The main advantage of this design is that it allows to try diferent reward models without changing the environment dynamics nor having to redesign the reward associated to each ReactionEvent. The three main possible reward models presented in Section 4.3 are im plemented in MARLKIT: (1) Mixed Reward Model, (2) Fully Cooperative Reward Model, (3) Fully Competitive Reward Model.

MixedRewardModel is the default reward model class. With this reward, one agent final reward is the equal to its raw reward $( \forall i , r _ { i } = \bar { r } _ { i } )$ .

When the system uses FullyCooperativeReward, it operates in a fully cooperative setti $\mathrm { n g ^ { 2 2 } }$

ZeroSumReward implements a fully competitive reward model. With this reward model, every times an agent receive a reward value, the same value is taken from the total of the others. This assure the experience is a zero-sum game between the agents.

## 6.3.3 Example

To illustrate how reward models can afect the dynamics of the system on their own, we conduct a collecting resources experiment. In this experiment, agents collect resources from production units with limited stock. We run this experiment with four diferent RewardModels while keeping the rest of the core modules identical. In the collectingresource package, four diferent launchers instantiate the same core modules except for the RewardModel.

The selected evaluation criteria are: total reward, which indicates whether the system reaches a high collective value; lowest individual reward, which is used as a fairness indicator; and the total number of collected resources, which is negatively correlated with the competitiveness of the system.

![](images/5296157780ab86784a94fc5f8c7b3f506f0b4174345391c61f9b36ade0e68edd.jpg)  
Figure 9: Total of the raw rewards of all agents added together. Green is independent reward, Blue is fully cooperative reward, red is fair mixed reward and orange is logistical reward.

Figure 9 illustrates how the resulting system dynamics can strongly difer depending only on the selected reward model. For conciseness, only one evaluation criterion is dis played here, as the purpose of this experiment is to illustrate the modular comparison enabled by MARLKIT rather than to provide a comprehensive evaluation of reward models. The complete results are provided in Appendix A.1.

The figure shows that changing only the reward model is suficient to produce substantially diferent system-level outcomes.

Additional details about the experimental setup and results are provided in the appendix A.1 and also in the README.md file of the collectingresource package. The latter also explains how to run this experiment. A second experiment, detailed in the README.md of the foraging package, applies two of the reward models evaluated in this article without modifying their implementation, in order to verify their reusability.

## 6.4 Learning

The diference between MARLKIT and other MARL tools regarding learning algorithms is that MARLKIT tends to assign less responsibility to the learning algorithm. Following the definition of a learning algorithm introduced in Section 2.6, the Algorithm is not responsible for generating data, but instead uses available data to update the policy. Therefore, in MARLKIT, the learning algorithm and the policy are two separate classes. Policy is responsible for action selection as it maps observations to actions, and Algorithm manages the update of the policy based on the agent’s experience.

Policies and learning algorithms remain interdependent. Some learning algorithms may require specific policy architectures, and conversely, some policies may require specific learning algorithms in order to be properly updated. This separation is designed to be flexible and avoid the need to implement the same policy multiple times. For example, SARSA (Sarsa) and Q-learning (QLearning) can be implemented as two distinct learning algorithms that both update the same Q-table based policy (QValueBasedPolicy).

## 6.5 Training and Execution Modes

The training and execution mode determines whether agents operate in a centralized or decentralized manner during each phase. In MARLKIT this is managed by a dedicated module: TrainingExecutionStrategy. The scheduler invokes agent operations through this module, which determines whether they are performed in a centralized or decentralized manner.

CTE implies that agents are trained and executed using centralized information. It is the counterpart of DTE, where agents operate in a decentralized manner during both training and execution. CTDE consists of training agents with centralized information while executing them without access to such information.

## 6.5.1 DTE

DTE is the default mode in MARLKIT. The default TrainingExecutionStrategy implements decentralized agent operations during both training and execution. DTE therefore serves as the baseline configuration, while other strategies can introduce centralized operations without modifying the scheduler.

## 6.5.2 CTE and CTDE

CTDE and CTE require the introduction of centralization mechanisms, either during training only (CTDE) or both training and execution (CTE). The mechanisms used to introduce centralization may be identical in both CTDE and CTE<sup>23</sup>. They difer in the phase during which centralized information is accessible: training only for CTDE, and both training and execution for CTE.

In MARLKIT, these mechanisms are implemented through dedicated TrainingExecutionStrategy implementations, which are used by the scheduler without requiring modifications to it

For instance, CentralizedCriticTrainingExecutionStrategy implements centralized training by providing critics with centralized information from multiple agents. Other agent operations can also be centralized, such as observation generation or policy updates.

## 6.5.3 Example

To illustrate how training and execution modes can vary, we conduct a prey hunter experiment. Two hunters must jointly capture a non-learning prey by both moving within close range of it. We run the experiment using two centralization configurations while keeping all other core modules identical. In the preyhunter package, separate launchers instantiate the same core modules, difering only in their centralization mechanism.

The selected evaluation criteria are the capture rate, defined as the proportion of episodes in which the hunters successfully capture the prey; the capture time, measuring the time required to achieve capture; and the total reward, reflecting the overall performance of the hunter team.

Figure 10 illustrates how the selected training mode afects the resulting system dynamics. A more comprehensive evaluation is presented in Appendix A.2. The appendix also provides details about the experimental setup, while the README.md file of the preyhunter package provides additional implementation details and instructions for running the experiment.

## 6.6 Communication

Communication provides agents with an additional mechanism for interaction by allowing them to exchange information. Communication in MARLKIT is designed as an optional module. Rather than embedding a fixed communication mechanism directly into the agents, the framework separates the ability to communicate from the communication strategy itself. This separation allows diferent communication protocols to be integrated, replaced, or compared without modifying the core agent implementation nor the simulation scheduler.

![](images/dfd720edcfa6ceb0ed85e5e8c71c6f7a0f762d4304369b8dc301f042a5af307a.jpg)  
Figure 10: Total of the raw rewards of all agents added together. Blue is DDPG and orange is MADDPG.

## 6.6.1 Communication Model

CommunicationModel implements a specific communication strategy. It determines which information is transmitted, which agents are targeted, and how incoming messages are interpreted.

This design makes communication strategies interchangeable: the same agent can be evaluated with diferent communication mechanisms, while the scheduler and the learning components remain unchanged.

For example, BroadcastObservation implements a simple observation-sharing mechanism. During the communication step, it broadcasts the agent’s local observation to all other agents. During the reception step, it enhances the agent’s observation using the observations received from other agents. Such a mechanism can help mitigate partial observability issues by allowing agents to share their local observations.

## 6.6.2 MLKAgentCommunicating

Similarly to MLKAgent, which defines the methods used to connect an agent with its learning module, MLKAgentCommunicating defines the methods which are useful for using the CommunicationModel.

Agents that implement the MLKAgentCommunicating interface expose the minimal capabilities required to participate in communication. Such agents rely on CommunicationModel associated with it.

The communicating agent itself does not define a specific protocol. Instead, it delegates two operations to its communication model. The first operation corresponds to the emission phase, where the agent sends messages to other agents. The second operation corresponds to the reception phase, where the agent handles the messages received from other agents and updates its internal state accordingly.

## 6.6.3 Example

To illustrate how communication mechanisms can afect the dynamics of the system on their own, we conduct a prey hunter experiment. This experiment and the evaluation criteria are the same as the ones described in Section 6.5.3.

The four tested communication configurations are: no communication, observation sharing, policy parameter sharing, and both observation and policy parameter sharing.

Figure 11 illustrates how the resulting system dynamics difer depending only on the selected communication mechanism.

Additional details about the experimental setup and the tested configurations are pro vided in the appendix A.3 and in the README.md file of the preyhunter package.

![](images/42676730fa3a008790b09b3ff4c7260501f400f0165de4fccecbab3d1985c942.jpg)  
Figure 11: Total of the raw rewards of all agents added together. Orange is no communication, green is policy parameter sharing, blue is observation sharing and red is both observation and policy parameter sharing.

## 6.7 MOA

Modeling other agents can help an agent predict others’ behavior and better account for their influence on the environment dynamics (see Section 4.7).

MARLKIT also provides support for agents that maintain explicit models of other agents. This functionality is treated as an optional component of the agent architecture: agents that need such models can include them, while simpler agents can ignore this mech anism entirely. This separation allows diferent modeling approaches to be integrated, replaced, or compared without modifying the core agent implementation. However, policies that exploit these predictions may need to be adapted to incorporate them into the action-selection process (see Figure 5).

## 6.7.1 ModelsManager

ModelsManager is responsible for managing the models. It provides an interface for accessing and updating the models used to reason about other agents.

The predictions exposed by the ModelsManager can then be used by diferent components of the framework. In particular, compatible policies may use them during action selection. For example, joint-action policies can evaluate candidate actions together with predicted actions of the other agents.

PredictionModelsManager is a specific class that implements ModelsManager. It is designed to manage predictive models of others’ actions.

This structure supports diferent ways of predicting other agents’ actions. For instance, it is possible to maintain one model per agent like in fictitious play. Alternatively, a second approach consists in modeling the other agents at the group level, by directly predicting a joint action. Minimax-based approaches follow this second approach by predicting the worst-case joint action of the opponents.

## 6.7.2 MLKAgentModelingOthers

Similarly to both MLKAgent and MLKAgentCommunicating, that define methods that manip ulate, respectively, the learning module and the communication model, MLKAgentModelingOthers defines the methods which are useful for using the a ModelsManager. The responsibility of managing the models is delegate to the ModelsManager.

A model can be used for diferent purposes, one of them is to predict the behavior or actions of other agents. MLKAgentModelingOthersAction is an interface for agents that maintain predictive models of other agents’ actions. This interface relies on PredictionModelsManager, a specific type of ModelsManager that manages predictive models.

![](images/72e0124c03d7361b2aaecf802c3db3e3d689509d79a2cd64a80b96184ede3a5d.jpg)  
Figure 12: Proportion of episodes in which one car passes, with one agent selecting Go and the other Yield.

## 6.7.3 Example

To illustrate how MOA implementations can afect the dynamics of the system on their own, we conduct a Go or Yield experiment. In this experiment, two agents (cars) repeatedly choose between Go and Yield. We test one agent with diferent MOA implementations while keeping the implementations of all other modules unchanged.

The tested MOA configurations are:

• Independent: the agent does not model the opponent;

• JAL-AM with Action Frequencies: the agent evaluates joint actions and predicts the opponent’s action based on the opponent’s recent actions;

• JAL-GT with Minimax: the agent evaluates joint actions and assumes the worst-case action from the opponent.

The selected evaluation criteria measure the proportion of episodes associated with each possible joint action: both agents selecting Go, resulting in a collision; both selecting Yield, resulting in both cars stopping; or one selecting Go while the other selects Yield, allowing one car to pass.

Figure 12 illustrates how changing only the MOA implementation leads to diferent learning dynamics. A more comprehensive analysis of these dynamics, along with further details about the experimental setup, is provided in Appendix A.4. The README.md file of the gooryield package also provides details about the setup and explains how to run the experiment.

## 6.8 System Evaluation

As explained in Section 4.9, MARL experiments should be evaluated at the system level by considering the overall performance of the agents as a collective. Because MARLKIT aims to support the exploration of diferent solutions, comparisons between solutions should be based on explicit metrics. MARLKIT therefore provides a dedicated mechanism for facilitating such evaluation.

Similarly to reward computation, the default system evaluation mechanism is based on ReactionEvents (see Section 6.2.2) because ReactionEvents describe relevant events that occur in the environment (see Figure 13).

The Measure used for system evaluation depend on the environment and on the objectives of the experiment. As discussed in Section 4.9, diferent MARL settings may require diferent evaluation criteria. For instance, the sum of all agents’ rewards can be used to evaluate global performance, fairness measures<sup>24</sup> can assess how equitable the distribution of outcomes is among agents, and episode duration can capture the eficiency with which the collective task is completed.

![](images/8b4c1c199d685000a433eda51ac2bc4c02ca4715498df621147e1145e1c412c1.jpg)  
Figure 13: System evaluation process.

At the end of each episode, the Measures are computed and automatically added to a log file. The default SystemEvaluator updates these measures from the ReactionEvents produced by the environment, which allows the evaluation procedure to remain independent of the other modules. This makes it possible to compare diferent implementations of reward models, learning algorithms, communication mechanisms, or models of other agents while keeping the same evaluation procedure. However, this design is not restricted to eventbased measures: the SystemEvaluator can be extended to update measures from other sources, such as the number of communication messages exchanged during an episode.

## 6.9 Exploring Solutions

The previous examples illustrate how diferent implementations of a core module can be explored and compared while keeping the remaining modules unchanged. However, each experimental configuration was defined and executed manually. One important benefit of BenchMARL is that it facilitates the exploration of diferent systems by allowing multiple experiments with diferent configurations to be launched together. Such functionality significantly improves the exploration of possible configurations. MARL frameworks designed for this purpose should therefore make it easy to define, execute and compare multiple experimental configurations. MARLKIT provides this capability.

For each core module identified in this paper, MARLKIT provides a corresponding configuration class, such as LearningModule or CommunicationModule. Together, these config urable modules define the complete setup of an experiment through an ExperimentConfiguration. Multiple experiment configurations can then be executed by a dedicated runner, which collects the measures produced by the selected system evaluator for each experiment.

The package multipleexperiences provides examples of such configurations, along with a README.md file explaining how to define and run them.

## 7 Conclusion

As prior research has highlighted, there are dificulties in comparing MARL systems and experiments in a clear and structured way. Considering this issue, in this paper we argue that a key obstacle lies in the tendency to describe MARL approaches solely through their proposed algorithms. This narrow framing obscures the underlying architectural aspects that shape the actual behavior of MARL systems. Consequently, this monolithic perspective hinders our ability to trace how specific modeling choices afect the dynamics of the system.

To address this issue, this paper advocated the adoption of a modular perspective in MARL system modeling. In particular, beyond standard software engineering benefits, our primary contribution is the formulation of a modular perspective that (1) makes all underlying modeling choices explicit when designing MARL systems and (2) facilitates the structured comparison and exploration of diverse modeling choices.

To this end, in this paper we have introduced a modular perspective for modeling MARL systems through a set of core modules. We argue that MARL experimentation would benefit from such a modular design. This decomposition provides a common language to discuss MARL systems and facilitates the comparison of MARL approaches. The core modules identified in this paper are: (1) agent, (2) environment, (3) learning, (4) reward modeling, (5) training and execution modes, (6) communication, (7) modeling of other agents, and (8) evaluation criteria.

Many MARL experiment exhibit high variance in their implementations and inconsistent baseline, making approaches dificult to compare. Existing tools have made important progress toward more standardized experimentation, but they do not always expose MARLspecific design choices as independent and reusable components. To address this limitation, we introduced MARLKIT, a proof of concept tool for MARL experimentation that imple ments the core modules identified in this paper as independent first-class components.

MARLKIT was designed to be modular and extensible. It allows users to easily design new module implementations and compare them with existing ones. By facilitating the isolation, reuse, and comparison of core modules, this tool supports a more systematic exploration of design choices in multi-agent systems. These capabilities provide practical evidence for the relevance of organizing MARL modeling choices around the proposed core modules.

At this stage, MARLKIT already provides several implementations for each core module, but it is not intended to be complete. In particular, many MARL algorithms and many module implementations are not yet available. MARLKIT should therefore be understood as a proof of concept demonstrating the relevance of the proposed modular decomposition, rather than as an exhaustive MARL library equipped with state-of-the-art algorithms.

Although MARLKIT currently serves primarily as a proof of concept, we believe it provides a promising foundation for a more mature and practically useful framework. Future work should therefore extend its capabilities and robustness, while using these developments to further substantiate the relevance of a modular perspective for MARL tools. Future work should first focus on using MARLKIT to conduct detailed experimental studies for each core module, in order to assess the advantages, limitations, and interactions of diferent implementations. Such studies would help clarify how specific design choices afect MARL performance, reproducibility, and robustness.

Another important direction is to enrich MARLKIT with additional module implementations, especially learning algorithms, so as to progressively turn the current proof of concept into a more complete experimental framework.

This paper establishes initial links between the proposed core modules and the main MARL challenges, but these relationships should be further investigated in future work. Non-stationarity, partial knowledge, credit assignment, and scalability do not afect all modules in the same way. Making these relationships more explicit would help researchers identify which parts of a MARL system should be modified, extended, or compared depending on the dificulties encountered in a given use case.

## 8 Thanks

This work is part of the Slam-B project within the exploratory PEPR FairCarboN and has benefited from funding provided by the French government, managed by the National Research Agency under the France 2030 initiative, with reference ANR-22-PEXF-0003.

This work received support from the French government, managed by the National Research Agency, under the France 2030 program, reference ANR-16-CONV-0004.

## References

Stefano V Albrecht and Peter Stone. Autonomous agents modelling other agents: A comprehensive survey and open problems. Artificial Intelligence, 258:66–95, 2018.

Stefano V Albrecht, Filippos Christianos, and Lukas Sch¨afer. Multi-agent reinforcement learning: Foundations and modern approaches. MIT Press, 2024.

Christopher Amato. An introduction to centralized training for decentralized execution in cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2409.03052, 2024.

Zenith Arnejo, Benoit Gaudou, Mehdi Saqalli, and Nathaniel Bantayan. Communicating agent-based models to stakeholders: A scoping review. Journal of Artificial Societies and Social Simulation, 28(2):2, 2025. ISSN 1460-7425. doi: 10.18564/jasss.5623.

Markel Sanz Ausin, Mehak Maniktala, Tifany Barnes, and Min Chi. Tackling the credit assignment problem in reinforcement learning-induced pedagogical policies with neural networks. In International Conference on Artificial Intelligence in Education, pages 356–368. Springer, 2021.

Roderick JS Baker and Peter I Cowling. Bayesian opponent modeling in a simple poker environment. In 2007 IEEE Symposium on Computational Intelligence and Games, pages 125–131. IEEE, 2007.

Parasumanna Gokulan Balaji and Dipti Srinivasan. An introduction to multi-agent systems. In Innovations in multi-agent systems and applications-1, pages 1–27. Springer, 2010.

Samuel Barrett and Peter Stone. Cooperating with Unknown Teammates in Complex Domains: A Robot Soccer Case Study of Ad Hoc Teamwork. In AAAI Conference on Artificial Intelligence, pages 2010–2016, 2015. doi: 10.1609/AAAI.V29I1.9428. URL https://mlanthology.org/aaai/ 2015/barrett2015aaai-cooperating/.

Samuel Barrett, Peter Stone, Sarit Kraus, and Avi Rosenfeld. Teamwork with limited knowledge of teammates. Proceedings of the AAAI conference on artificial intelligence, 27(1):102–108, 2013.

Ulrich Berger. Fictitious play in 2 x n games. Journal of Economic Theory, 120(2):139–154, 2005.

Matteo Bettini, Amanda Prorok, and Vincent Moens. Benchmarl: Benchmarking multi-agent reinforcement learning. Journal of Machine Learning Research, 25(217):1–10, 2024.

Jan Blumenkamp and Amanda Prorok. The emergence of adversarial communication in multi-agent reinforcement learning. In Jens Kober, Fabio Ramos, and Claire Tomlin, editors, Proceedings of the 2020 Conference on Robot Learning, volume 155 of Proceedings of Machine Learning Research, pages 1394–1414. PMLR, 16–18 Nov 2021. URL https://proceedings.mlr.press/v155/ blumenkamp21a.html.

Hayley Borck, Justin Karneeb, Ron Alford, and David W Aha. Case-based behavior recognition in beyond visual range air combat. In FLAIRS, pages 379–384, 2015.

Vivek S. Borkar. Stochastic approximation with two time scales. Systems & Control Letters, 29(5): 191–194, 1997.

Vivek S Borkar. Reinforcement learning in markovian evolutionary games. Advances in Complex Systems, 5(01):55–72, 2002.

Albert Bou, Matteo Bettini, Sebastian Dittert, Vikash Kumar, Shagun Sodhani, Xiaomeng Yang, Gianni De Fabritiis, and Vincent Moens. [torchrl: A data-driven decision-making library for pytorch]. In International conference on learning representations, pages 1778–1811, 2024.

Michael Bowling. Convergence and No-Regret in Multiagent Learning. In Neural Information Processing Systems, pages 209–216, 2004. URL https://mlanthology.org/neurips/2004/ bowling2004neurips-convergence/.

Michael Bowling and Manuela Veloso. An analysis of stochastic game theory for multiagent reinforcement learning. Technical report, School of Computer Science, Carnegie Mellon University, Pennsylvania, 2000.

Michael Bowling and Manuela Veloso. Multiagent learning using a variable learning rate. Artificial intelligence, 136(2):215–250, 2002.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

George W. Brown. Iterative solution of games by fictitious play. In T. Koopmans, editor, Activity Analysis of Production and Allocation, pages 347–376. Wiley, 1951.

Lucian Bu¸soniu, Bart De Schutter, and Robert Babuska. Decentralized reinforcement learning control of a robotic manipulator. In 2006 9th International Conference on Control, Automation, Robotics and Vision, pages 1–6. IEEE, 2006.

Lucian Bu¸soniu, Robert Babuska, and Bart De Schutter. A comprehensive survey of multiagent reinforcement learning. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 38(2):156–172, 2008.

Lucian Bu¸soniu, Robert Babuˇska, and Bart De Schutter. Multi-agent reinforcement learning: An overview. In Dipti Srinivasan and Lakhmi C. Jain, editors, Innovations in Multi-Agent Systems and Applications – 1, volume 310 of Studies in Computational Intelligence, chapter 7, pages 183– 221. Springer, Berlin, Germany, 2010.

Lorenzo Canese, Gian Carlo Cardarilli, Luca Di Nunzio, Rocco Fazzolari, Daniele Giardino, Marco Re, and Sergio Span\`o. Multi-agent reinforcement learning: A review of challenges and applications. Applied Sciences, 11(11):4948, 2021.

Jim Martin Catacora Ocana, Francesco Riccio, Roberto Capobianco, and Daniele Nardi. Cooperative multi-agent deep reinforcement learning in soccer domains. In Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS ’19, pages 1865–1867, Richland, SC, 2019. International Foundation for Autonomous Agents and Multia gent Systems. ISBN 9781450363099.

Gang Chen. A new framework for multi-agent reinforcement learning: Centralized training and exploration with decentralized execution via policy distillation. In Proceedings of the 19th International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’20, pages 1784–1786, Auckland, New Zealand, 2020. International Foundation for Autonomous Agents and Multiagent Systems.

Alexander Chernyavskiy, Alexey Skrynnik, and Aleksandr Panov. Applying opponent and environment modelling in decentralised multi-agent reinforcement learning. Cognitive Systems Research, 89:101306, 2025.

Caroline Claus and Craig Boutilier. The dynamics of reinforcement learning in cooperative multiagent systems. In AAAI Conference on Artificial Intelligence, pages 746–752, 1998.

Vincent Conitzer and Tuomas Sandholm. Awesome: A general multiagent learning algorithm that converges in self-play and learns a best response against stationary opponents. Machine Learning, 67(1):23–43, 2007.

Abhishek Das, Th´eophile Gervet, Joshua Romof, Dhruv Batra, Devi Parikh, Mike Rabbat, and Joelle Pineau. Tarmac: Targeted multi-agent communication. In International Conference on machine learning, pages 1538–1546. PMLR, 2019.

Aaron Davidson, Darse Billings, Jonathan Schaefer, and Duane Szafron. Improved opponent modeling in poker. In International Conference on Artificial Intelligence, ICAI’00, pages 1467–1473, 2000.

Christian Schroeder De Witt, Tarun Gupta, Denys Makoviichuk, Viktor Makoviychuk, Philip HS Torr, Mingfei Sun, and Shimon Whiteson. Is independent learning all you need in the starcraft multi-agent challenge? arXiv preprint arXiv:2011.09533, 2020.

Badr Ben Elallid, Nabil Benamar, Abdelhakim Senhaji Hafid, Tajjeeddine Rachidi, and Nabil Mrani. A comprehensive survey on the application of deep and reinforcement learning approaches in autonomous driving. Journal of King Saud University-Computer and Information Sciences, 34 (9):7366–7390, 2022.

Jacques Ferber. Multi-agent systems: an introduction to distributed artificial intelligence, volume 1. Addison-Wesley, 1999.

Jacques Ferber and Jean-Pierre M¨uller. Influences and reaction: a model of situated multiagent systems. In Proceedings of second international conference on multi-agent systems (ICMAS-96), pages 72–79, 1996.

Jakob N. Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual Multi-Agent Policy Gradients. In AAAI Conference on Artificial Intelligence, pages 2974–2982, 2018. doi: 10.1609/AAAI.V32I1.11794. URL https://mlanthology.org/ aaai/2018/foerster2018aaai-counterfactual/.

Rihab Gorsane, Omayma Mahjoub, Ruan John de Kock, Roland Dubb, Siddarth Singh, and Arnu Pretorius. Towards a standardised performance evaluation protocol for cooperative marl. In Advances in Neural Information Processing Systems, volume 35, pages 5510–5521, 2022.

Sven Gronauer and Klaus Diepold. Multi-agent deep reinforcement learning: a survey. Artificial Intelligence Review, 55(2):895–943, 2022.

Zahia Guessoum, Lilia Rejeb, and Rodolphe Durand. Using adaptive multi-agent systems to simulate economic models. In Proceedings of the Third International Joint Conference on Autonomous Agents and Multiagent Systems, 2004. AAMAS 2004., pages 68–75. IEEE, 2004.

Jayesh K Gupta, Maxim Egorov, and Mykel Kochenderfer. Cooperative multi-agent control using deep reinforcement learning. In International conference on autonomous agents and multiagent systems, pages 66–83. Springer, 2017.

Johannes Heinrich and David Silver. Deep reinforcement learning from self-play in imperfectinformation games. arXiv preprint arXiv:1603.01121, 2016.

Pablo Hernandez-Leal, Matthew E. Taylor, Benjamin Rosman, L. Enrique Sucar, and Enrique Munoz de Cote. Identifying and tracking switching, non-stationary opponents: A bayesian approach. In Workshop on Multiagent Interaction without Prior Coordination, MIPC at AAAI-16, pages 560–566, Phoenix, Arizona, USA, February 2016.

Pablo Hernandez-Leal, Michael Kaisers, Tim Baarslag, and Enrique Munoz De Cote. A survey of learning in multiagent environments: Dealing with non-stationarity. arXiv preprint, 2017.

Yaqing Hou, Mingyang Sun, Wenxuan Zhu, Yifeng Zeng, Haiyin Piao, Xuefeng Chen, and Qiang Zhang. Behavior reasoning for opponent agents in multi-agent learning systems. IEEE Transactions on Emerging Topics in Computational Intelligence, 6(5):1125–1136, 2022.

Junling Hu and Michael P Wellman. Nash q-learning for general-sum stochastic games. Journal of machine learning research, 4:1039–1069, 2003.

Siyi Hu, Yifan Zhong, Minquan Gao, Weixun Wang, Hao Dong, Xiaodan Liang, Zhihui Li, Xiaojun Chang, and Yaodong Yang. Marllib: A scalable and eficient multi-agent reinforcement learning library. Journal of Machine Learning Research, 24(315):1–23, 2023.

Dom Huh and Prasant Mohapatra. Multi-agent reinforcement learning: A comprehensive survey. arXiv preprint arXiv:2312.10256, 2023.

Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, and Shimon Whiteson. Deep variational reinforcement learning for pomdps. In International conference on machine learning, pages 2117– 2126. PMLR, 2018.

Beakcheol Jang, Myeonghwi Kim, Gaspard Harerimana, and Jong Wook Kim. Q-learning algorithms: A comprehensive classification and applications. IEEE access, 7:133653–133667, 2019.

Natasha Jaques, Angeliki Lazaridou, Edward Hughes, Caglar Gulcehre, Pedro Ortega, DJ Strouse, Joel Z Leibo, and Nando De Freitas. Social influence as intrinsic motivation for multi-agent deep reinforcement learning. In International conference on machine learning, pages 3040–3049. PMLR, 2019.

Jiechuan Jiang and Zongqing Lu. Learning Attentional Communication for Multi-Agent Cooperation. In Neural Information Processing Systems, pages 7254–7264, 2018. URL https: //mlanthology.org/neurips/2018/jiang2018neurips-learning/.

Soummya Kar, Jos´e MF Moura, and H Vincent Poor. Qd-learning: A collaborative distributed strategy for multi-agent reinforcement learning through consensus + innovations. IEEE Transactions on Signal Processing, 61(7):1848–1862, 2013.

Vijay R. Konda and John N. Tsitsiklis. Actor-critic algorithms. In Sara A. Solla, Todd K. Leen, and Klaus-Robert M¨uller, editors, Advances in Neural Information Processing Systems, volume 12, pages 1008–1014. MIT Press, 1999. URL https://proceedings.neurips.cc/paper\_files/ paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf.

Ezgi Korkmaz. A survey analyzing generalization in deep reinforcement learning. arXiv preprint arXiv:2401.02349, 2024. doi: 10.48550/arXiv.2401.02349. URL https://arxiv.org/abs/2401. 02349.

Landon Kraemer and Bikramjit Banerjee. Multi-agent reinforcement learning as a rehearsal for decentralized planning. Neurocomputing, 190:82–94, 2016.

Harold W Kuhn. Extensive games and the problem of information. In Contributions to the Theory of Games, volume 2, pages 193–216. Princeton University Press, 1953.

Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius Zambaldi, Satyaki Upadhyay, Julien P´erolat, Sriram Srinivasan, Finbarr Timbers, Karl Tuyls, Shayegan Omidshafiei, Danie Hennes, Dustin Morrill, Paul Muller, Timo Ewalds, Ryan Faulkner, J´anos Kram´ar, Bart De Vylder, Brennan Saeta, James Bradbury, David Ding, Sebastian Borgeaud, Matthew Lai, Julian Schrittwieser, Thomas Anthony, Edward Hughes, Ivo Danihelka, and Jonah Ryan-Davis. Open-Spiel: A framework for reinforcement learning in games. CoRR, abs/1908.09453, 2019. URL http://arxiv.org/abs/1908.09453.

Martin Lauer and Martin A Riedmiller. An algorithm for distributed reinforcement learning in cooperative multi-agent systems. In Proceedings of the seventeenth international conference on machine learning, pages 535–542, 2000.

Ken Ming Lee, Sriram Ganapathi Subramanian, and Mark Crowley. Investigation of independent reinforcement learning algorithms in multi-agent environments. Frontiers in artificial intelligence, 5:805823, 2022.

David S. Leslie and Edmund J. Collins. Convergent multiple-timescales reinforcement learning algorithms in normal form games. The Annals of Applied Probability, 13(4):1231–1251, 2003.

Eric Liang, Richard Liaw, Robert Nishihara, Philipp Moritz, Roy Fox, Ken Goldberg, Joseph Gonzalez, Michael Jordan, and Ion Stoica. Rllib: Abstractions for distributed reinforcement learning. In International conference on machine learning, pages 3053–3062. PMLR, 2018.

Jiaxin Liang, Haotian Miao, Kai Li, Jianheng Tan, Xi Wang, Rui Luo, and Yueqiu Jiang. A review of multi-agent reinforcement learning algorithms. Electronics, 14(4):820, 2025.

Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In Machine learning proceedings 1994, pages 157–163. Elsevier, 1994.

Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actorcritic for mixed cooperative-competitive environments. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pages 6382–6393, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

La¨etitia Matignon, Guillaume J Laurent, and Nadine Le Fort-Piat. Hysteretic q-learning: an algorithm for decentralized reinforcement learning in cooperative multi-agent teams. In 2007 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 64–69. IEEE, 2007.

Laetitia Matignon, Guillaume J Laurent, and Nadine Le Fort-Piat. Independent reinforcement learners in cooperative markov games: a survey regarding coordination problems. The Knowledge Engineering Review, 27(1):1–31, 2012.

Fabien Michel. The IRM4S Model: The Influence/Reaction Principle for Multiagent Based Simulation (short paper). In Edmund H. Durfee, Makoto Yokoo, Michael N. Huhns, and Onn Shehory, editors, Proceedings of the 6th International Joint Conference on Autonomous Agents and Multiagent Systems (AAMAS 2007), Honolulu, Hawaii, USA, May 14-18, 2007, AAMAS ’07, pages 903– 905, New York, NY, USA, 2007. ACM. ISBN 978-81-904262-7-5. doi: 10.1145/1329125.1329289.

Fabien Michel, Jacques Ferber, and Alexis Drogoul. Multi-Agent Systems and Simulation: a Survey From the Agents Community’s Perspective. In Adelinde Uhrmacher and Danny Weyns, editors, Multi-Agent Systems: Simulation and Applications, Computational Analysis, Synthesis,

and Design of Dynamic Systems, pages 3–52. CRC Press - Taylor & Francis, June 2009. ISBN 9781420070231.

John E Moody, Yufeng Liu, Matthew Safell, and Kyoungju Youn. Stochastic direct reinforcement: Application to simple games with recurrence. In AAAI Technical Report (2), pages 23–34, 2004.

J. F. Nash. Non-cooperative games. Annals of Mathematics, 54:286–295, 1951.

Zepeng Ning and Lihua Xie. A survey on multi-agent reinforcement learning and its application. Journal of Automation and Intelligence, 3(2):73–91, 2024.

Sindhu Padakandla, Prabuchandran KJ, and Shalabh Bhatnagar. Reinforcement learning algorithm for non-stationary environments. Applied Intelligence, 50(11):3590–3606, 2020.

Georgios Papoudakis, Filippos Christianos, and Stefano Albrecht. Agent modelling under partial observability for deep reinforcement learning. In Advances in Neural Information Processing Systems, volume 34, pages 19210–19222, 2021a.

Georgios Papoudakis, Filippos Christianos, Lukas Sch¨afer, and Stefano V. Albrecht. Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks. In Joaquin Vanschoren and Serena Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021b. URL https://datasets-benchmarks-proceedings.neurips.cc/paper\_files/paper/2021/file/ a8baa56554f96369ab93e4f3bb068c22-Paper-round1.pdf.

Steven Perkins, Panayotis Mertikopoulos, and David S Leslie. Mixed-strategy learning with continuous action sets. IEEE Transactions on Automatic Control, 62(1):379–384, 2015.

H. L. Prasad, L. A. Prashanth, and Shalabh Bhatnagar. Two-timescale algorithms for learning nash equilibria in general-sum stochastic games. In Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent Systems, pages 1371–1379, 2015.

Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

Antonin Rafin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of machine learning research, 22(268):1–8, 2021.

Herbert Robbins and Sutton Monro. A Stochastic Approximation Method. The Annals ofMathematical Statistics, 22(3):400–407, 1951. ISSN 0003-4851, 2168-8990. doi: 10.1214/aoms/1177729586. URL https://projecteuclid.org/journals/annals-of-mathematical-statistics/ volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586. full.

Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim G. J. Rudner, Chia-Man Hung, Philip H. S. Torr, Jakob Foerster, and Shimon Whiteson. The starcraft multi-agent challenge. In Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS ’19, pages 2186–2188, Richland, SC, 2019. International Foundation for Autonomous Agents and Multiagent Systems. ISBN 9781450363099.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Ashish Kumar Shakya, Gopinatha Pillai, and Sohom Chakrabarty. Reinforcement learning algorithms: A brief survey. Expert Systems with Applications, 231:120495, 2023.

Lloyd S Shapley. Stochastic games. Proceedings of the national academy of sciences, 39(10):1095– 1100, 1953.

Piyush K Sharma, Rolando Fernandez, Erin Zaroukian, Michael Dorothy, Anjon Basak, and Derrik E Asher. Survey of recent multi-agent reinforcement learning algorithms utilizing centralized training. In Artificial intelligence and machine learning for multi-domain operations applications III, volume 11746, pages 665–676. SPIE, 2021.

Zhenyu Shou and Xuan Di. Reward design for driver repositioning using multi-agent reinforcement learning. Transportation research part C: emerging technologies, 119:102738, 2020.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587):484–489, 2016. ISSN 1476-4687. doi: 10.1038/nature16961. URL https://doi.org/10.1038/nature16961.

Sriram Ganapathi Subramanian, Pascal Poupart, Matthew E. Taylor, and Nidhi Hegde. Multi type mean field reinforcement learning. In Amal El Fallah Seghrouchni, Gita Sukthankar, Bo An, and Neil Yorke-Smith, editors, Proceedings of the 19th International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’20, Auckland, New Zealand, May 9-13, 2020, pages 411–419. International Foundation for Autonomous Agents and Multiagent Systems, 2020. doi: 10.5555/3398761.3398813.

Sriram Ganapathi Subramanian, Matthew E. Taylor, Mark Crowley, and Pascal Poupart. Partially observable mean field reinforcement learning. In Frank Dignum, Alessio Lomuscio, Ulle Endriss, and Ann Now´e, editors, AAMAS ’21: 20th International Conference on Autonomous Agents and Multiagent Systems, Virtual Event, United Kingdom, May 3-7, 2021, pages 537–545. ACM, 2021. doi: 10.5555/3463952.3464019. URL https://www.ifaamas.org/Proceedings/ aamas2021/pdfs/p537.pdf.

Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2nd edition, 2018.

Richard Stuart Sutton. Temporal credit assignment in reinforcement learning. University of Massachusetts Amherst, 1984.

Ardi Tampuu, Tambet Matiisen, Dorian Kodelja, Ilya Kuzovkin, Kristjan Korjus, Juhan Aru, Jaan Aru, and Raul Vicente. Multiagent cooperation and competition with deep reinforcement learning. PloS one, 12(4):e0172395, 2017.

Ming Tan. Multi-agent reinforcement learning: independent versus cooperative agents. In Proceedings of the Tenth International Conference on Machine Learning, ICML’93, pages 330–337, San Francisco, CA, USA, 1993. Morgan Kaufmann Publishers Inc. ISBN 1558603077.

J Terry, Benjamin Black, Nathaniel Grammel, Mario Jayakumar, Ananth Hari, Ryan Sullivan, Luis S Santos, Clemens Diefendahl, Caroline Horsch, Rodrigo Perez-Vicente, et al. Pettingzoo: Gym for multi-agent reinforcement learning. In Advances in Neural Information Processing Systems, volume 34, pages 15032–15043, 2021.

Gerald Tesauro. Temporal diference learning and td-gammon. Communications of the ACM, 38(3): 58–68, 1995.

Mark Towers, Ariel Kwiatkowski, John Balis, Gianluca De Cola, Tristan Deleu, Manuel Goul˜ao, Kallinteris Andreas, Markus Krimmel, Arjun KG, Rodrigo Perez-Vicente, J Terry, Andrea Pierr´e, Sander Schulhof, Jun Jet Tai, Hannah Tan, and Omar G. Younis. Gymnasium: A standard interface for reinforcement learning environments. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference. Curran Associates, Inc., 2025. doi: 10.52202/085713-4916. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ file/d7ff1795e8527f6443371c3933bdb52b-Paper-Datasets\_and\_Benchmarks\_Track.pdf.

Karl Tuyls and Gerhard Weiss. Multiagent learning: Basics, challenges, and prospects. Ai Magazine, 33(3):41–41, 2012.

Branislav Ulicny and Daniel Thalmann. Towards interactive real-time crowd behavior simulation. Computer Graphics Forum, 21(4):767–775, 2002. doi: https://doi.org/10.1111/1467-8659.00634. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-8659.00634.

Bingyan Wang, Yuling Yan, and Jianqing Fan. Sample-eficient reinforcement learning for linearlyparameterized mdps with a generative model. In Advances in Neural Information Processing Systems, volume 34, pages 23009–23022, 2021.

Michael Winikof and Lin Padgham. Agent oriented software engineering. In Gerhard Weiss, editor, Multiagent Systems, chapter 15, pages 695–757. MIT Press, 2<sup>nd</sup> edition, 2013. ISBN 978-0-262- 01889-0.

Annie Wong, Thomas B¨ack, Anna V Kononova, and Aske Plaat. Deep multiagent reinforcement learning: Challenges and directions. Artificial Intelligence Review, 56(6):5023–5056, 2023.

Dehai Yang, Yong Wei, Xuelei Nie, Liqiang Zhao, and Guiyang Xu. Reinforcement learning-based market game model considering virtual power plants. In 2023 5th International Conference on Power and Energy Technology (ICPET), pages 1159–1164. IEEE, 2023.

Yaodong Yang and Jun Wang. An overview of multi-agent reinforcement learning from game theoretical perspective. arXiv preprint arXiv:2011.00583, 2020.

Yaodong Yang, Rui Luo, Minne Li, Ming Zhou, Weinan Zhang, and Jun Wang. Mean field multiagent reinforcement learning. In International conference on machine learning, pages 5571–5580. PMLR, 2018.

Chengwei Zhang, Yu Tian, Zhibin Zhang, Wanli Xue, Xiaofei Xie, Tianpei Yang, Xin Ge, and Rong Chen. Neighborhood cooperative multiagent reinforcement learning for adaptive trafic signa control in epidemic regions. IEEE transactions on intelligent transportation systems, 23(12): 25157–25168, 2022.

Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Basar. Fully decentralized multiagent reinforcement learning with networked agents. In International conference on machine learning, pages 5872–5881. PMLR, 2018.

Kaiqing Zhang, Zhuoran Yang, and Tamer Ba¸sar. Multi-agent reinforcement learning: A selective overview of theories and algorithms. In Kyriakos G. Vamvoudakis, Yan Wan, Frank L. Lewis, and Derya Cansever, editors, Handbook of Reinforcement Learning and Control, pages 321–384. Springer International Publishing, Cham, 2021. ISBN 978-3-030-60990-0. doi: 10.1007/ 978-3-030-60990-0 12. URL https://doi.org/10.1007/978-3-030-60990-0\_12.

Ruixiaoxiao Zhang, Minhyun Lee, Lijie Huang, and Shiqi Mei. Multi-agent-based modeling of stakeholder dynamics in renewable energy markets: Towards enhanced policy-making and sustainable energy transition. Renewable Energy, 262:125369, 2026. ISSN 0960-1481. doi: doi.org/10.1016/j.renene.2026.125369.

Yan Zheng, Zhaopeng Meng, Jianye Hao, Zongzhang Zhang, Tianpei Yang, and Changjie Fan. A Deep Bayesian Policy Reuse Approach Against Non-Stationary Agents. In Neural Information Processing Systems, volume 31, pages 954–964, 2018. URL https://mlanthology.org/neurips/ 2018/zheng2018neurips-deep/.

Changxi Zhu, Mehdi Dastani, and Shihan Wang. A survey of multi-agent deep reinforcement learning with communication. Autonomous Agents and Multi-Agent Systems, 38(1):4, 2024.

## Appendix A. Experimentation Details

All the experiments presented in Section 6 was run with ten diferent seeds. The following give more details about each experiments.

In all these experiment, the agents’ behavior follows the default behavior presented in Section 4.8.1. Each agent has one policy and one learning algorithm, which are called when required.

## A.1 Reward Model Experimentation

The example presented in Section 6.3.3 consists of four variants that difer only in the implementation of the reward model module. The implementations of the remaining modules are detailed below.

The environment dynamics follow those of the collecting resource experiment. Entities produce resources, and agents can collect one unit of a resource at each timestep from their selected production unit. In MARLKIT, this is implemented in the EnvCollectingResource class.

For the learning module, the learning algorithm is Q-learning, a value-based algorithm based on a Q-table (Jang et al., 2019). For each state-action pair, the table stores a value corresponding to the expected return. The update follows the equation: $Q _ { t + 1 } =$ $Q _ { t } + \alpha \cdot ( r _ { t } + \gamma \cdot Q _ { \operatorname* { m a x } } - Q _ { t } )$ , where Q is the table value, α is the learning rate, and γ is the discount factor. The value of α is set to 0.2, and γ is set to 0.95. The policy is based on the Q-table and selects the action associated with the highest value. The policy is epsilon-greedy, and the ϵ parameter follows the equation:

$$
\epsilon = ( 1 - d ) ^ { t }\tag{15}
$$

where d is the decay rate, set to $3 \cdot 1 0 ^ { - 3 } $ , and t is the number of episodes. Thus, the ϵ parameter decreases exponentially.

Training and execution are completely decentralized (DTE). There is no communication between agents, and agents do not build or use models of other agents.

To evaluate the system, three metrics were selected. The evaluation criteria are the total reward, which indicates whether the system reaches a high collective value; the lowest individual reward, which is used as a fairness indicator; and the total number of collected resources, which is negatively correlated with the competitiveness of the system.

The four tested RewardModels are:

• Independent reward (MixedReward): the final reward received by each agent is equal to its raw reward, see Section 4.3.3.

• Fully cooperative reward (FullyCooperativeReward): rewards are averaged and returned to all agents, see Section 4.3.1.

• Fair mixed reward (FairMixedReward): extends the independent reward model with a fairness term. It first computes the raw reward of each agent and identifies the lowest raw reward ${ \bar { r } } _ { \operatorname* { m i n } } .$ Then, the final reward for an agent i is computed as follows: $r _ { i } = \left( 1 - \delta \right) \times \bar { r } _ { i } + \delta \times \bar { r } _ { \mathrm { m i n } }$ , with $\delta \in [ 0 , 1 ]$ the fairness parameter.

• Logistical reward (LogisticalRewardModel): an environment-specific reward model that applies a penalty when agents request too many resources.

Reward models manipulate the default event reward $\rho$ to generate the final reward r for each agent (see Section 6.3). The following list presents the possible events in this experiment and their associated default rewards:

• Collect Resource : This event is associated with an agent when it collects a resource. The corresponding reward is $\rho ( e _ { \mathrm { c o l l e c t - r e s o u r c e } } ) \in \{ 0 , 3 , 5 , 3 0 \}$ , depending on the resource acquired. The distribution of the collected resources’ values is highly unequal. Because high-value resources are produced in limited quantities, this creates incentives for greater competition.

• Distance Penalty : This event is associated with an agent when it travels to collect a resource. The value of $\rho ( e _ { \mathrm { d i s t a n c e } } )$ is negative and depends on the distance traveled. In this scenario, this penalty cannot be lower than −1.5.

Figure 14 presents the results obtained with the diferent reward models according to the selected evaluation criteria. The curves represent the median across all runs, while the colored areas indicate the range between the first and third quartiles. The data are smoothed using a moving average with a window size of 100.

The fully cooperative setting results in the highest total reward, which was expected because this reward model makes this metric the common objective of all agents. Therefore, it also maximizes the total number of collected resources because global reward maximiza tion eliminates competition between agents for resource collection. However, this reward model also increases inequality in this experiment: one agent ends up always collecting the resource associated with the smallest positive reward.

In this experiment, the independent and fair reward models produce similar results: the lowest total reward and the smallest number of collected resources, but a better balance between agents’ rewards. Completely egoistic behaviors lead agents to converge toward a balanced distribution of rewards, which explains why the independent and fair reward models produce similar results.

The logistical reward model penalizes competition. It leads to intermediate results between the fully cooperative and independent reward models. Two agents ended up competing for the highest value resource, while the remaining agent collects the second most valuable resource.

## A.2 Training and Execution Mode Experimentation

Section 6.5.3 presents an example in which decentralized and centralized training and execution are compared. The two presented algorithms are DDPG and MADDPG.

The environment dynamics follow a prey hunter experiment. One prey moves randomly, and two hunters must both be close enough to catch it. Each hunter’s action is a continuous two-dimensional movement vector generated by its policy. The hunters have a limited viewing range. The episode ends when either the prey has been caught or 100 timesteps have passed without a capture.

The selected reward model is the mixed reward model described in Section 4.3.3. This reward model aggregates the default rewards $\rho$ associated with the events experienced by an agent to determine its final reward. The possible events and their corresponding default rewards are defined as follows:

![](images/6cdcc05e324cf0a3eb2a39d835101654bbbe2a9fca7dfa69241d370b516bd72b.jpg)

(a) Total of the raw rewards of all agents added together.  
![](images/5eaa01e3c7899d58cc5846308640619028a623d7fcf2c1a51502cfaa5c8b8acf.jpg)

(b) Lowest individual raw reward received by an agent.  
![](images/d60b299614178644b3d2a8885f286c7bd33d72c4acff279e39788e26b6fc2013.jpg)  
(c) Number of collected resources.  
Figure 14: All collecting resources results. Green is independent reward, blue is fully cooperative reward, red is fair mixed reward and orange is logistical reward.

• Prey Caught: This event is associated with an agent when it catches the prey. The corresponding event default reward is $\rho ( e _ { \mathrm { p r e y \mathrm { - } c a u g h t } } ) = 3 0 0$

• Distance: This event is associated with an agent and depends on its distance from the prey. When the agent is close to the prey, the associated reward satisfies $0 \leq \rho ( e _ { \mathrm { d i s t a n c e } _ { - } } ) \leq 2$ . When it is too far from the prey, it receives a negative reward given by $\rho ( e _ { \mathrm { d i s t a n c e } _ { - } } ) = - 0 . 5 \times d$ where d denotes the distance between the agent and the prey.

Agents do not communicate information through messages, the communication module is set to no communication.

To evaluate the system, three metrics are used:

• Total reward: The sum of the raw rewards received by all agents.

• Capture success rate: The proportion of episodes in which the prey is caught.

• Capture time: The number of timesteps required to catch the prey. This metric is set to 100 when the prey is not caught.

The learning algorithm is Deep Deterministic Policy Gradient (DDPG), an actor-critic learning algorithm (see Section 2.6.1) designed for continuous action spaces. DDPG learns a deterministic policy and trains its actor and critic from experiences sampled from a replay bufer. MADDPG is a variation of DDPG based on centralized critics. To train the critic, agents also need models of the other agents’ policies to predict their actions. More details about MADDPG are provided in Section 4.10.1. In MARLKIT, both algorithms use the MLPDeterministicPolicy class as their policy.

DDPG and MADDPG use the same hyperparameters. The actor has two hidden layers of 64 neurons, while the critic has one hidden layer of 64 neurons. The actor and critic learning rates are set to $1 0 ^ { - 4 }$ and $1 0 ^ { - 3 }$ , respectively. The discount factor γ is set to 0.99, and the target-network update coeficient τ to 0.005. Training uses batches of 64 experiences sampled from a replay bufer with a capacity of 100000 experiences. Exploration is performed by adding Gaussian noise to the actor’s actions. The standard deviation for the Gaussian noise is set 0.4.

Figure 15 presents the results obtained with DDPG and MADDPG, corresponding respectively to decentralized and centralized training. The curves represent the median across all runs, while the colored areas indicate the range between the first and third quartiles. The data are smoothed using a moving average with a window size of 100.

In this experiment, DDPG and MADDPG present similar median performance across the selected evaluation criteria. However, MADDPG produces less dispersed results, as indicated by the smaller distance between the first and third quartiles.

## A.3 Communication Experimentation

Section 6.6.3 presents an example in which four diferent communication modes are compared.

The environment dynamics follow the same prey hunter experiment as the one described in Appendix A.2. The reward model and evaluation criteria are also unchanged.

For the learning module, the learning algorithm is PPOCategorical, an implementation of Proximal Policy Optimization (PPO), a policy-based algorithm (Schulman et al., 2017). PPO updates the policy by increasing the probability of successful actions and decreasing the probability of unsuccessful ones. It limits how much the policy can change during each update, making learning more stable. ϵ is the clipping parameter and is set to 0.2. The learning rate is set to 0.001, and the discount factor γ is set to 0.95. The policy is updated over 4 epochs at each learning step. The policy is implemented by the NeuralNetworkCategoricalPolicy class and is based on a neural network composed of two hidden layers containing 32 neurons each. The network produces a categorical probability distribution over the possible actions. Actions are selected using a softmax function with a temperature parameter set to 1.0. The PPO algorithm improves the policy while preventing excessively large updates through the clipped objective.

![](images/d6dfe9356d4bdcdceae817c2c81c18a7de7f5e2b757a59d3d401666c34fa8dc6.jpg)

(a) Total of the raw rewards of all agents added together.  
![](images/1f207155070d415484747995165dd3eef79475353d1c5f9e46323c4921670a0e.jpg)

(b) Proportion of episodes in which the prey is caught  
![](images/47fbfd6c1428fedbe8a540824525c1c96bb8358bfe4b088727df8413bed0d99b.jpg)  
(c) Number of timesteps required to catch the prey.  
Figure 15: Results of the prey-hunter experiment. Blue is DDPG and orange is MADDPG.

Training and execution are completely decentralized (DTE), and agents do not build or use models of other agents.

This experiment compares diferent communication modes. The four communication modes considered are:

• No communication: The agents do not communicate with one another.

• Observation sharing: The agents broadcast their observations to the other agents. They then use the received messages to enrich their own observations.

• Policy parameter sharing: The agents broadcast their policy parameters. Each agent then averages its own parameters with those received from the other agents and uses the resulting averaged parameters.

• Observation and policy parameter sharing: This mode combines observation sharing with policy parameter sharing.

Figure 16 presents the results obtained for the diferent communication modes. The curves represent the median across all runs, while the colored areas indicate the range between the first and third quartiles. The data are smoothed using a moving average with a window size of 500.

In this experiment, even with a well trained policy, the agents may fail to catch the prey because they must both have it within their observation range and be close enough to capture it. Observation sharing leads to higher rewards because it increases the likelihood that these conditions are met. With well trained policies, only one agent needs to locate the prey for the agents to coordinate and capture it. Therefore, agents that share their observations generally achieve better performance.

Sharing policy parameters also facilitates learning. When one agent improves its policy, this improvement propagates to the other agents through parameter sharing. In this experiment, policy parameter sharing not only leads to better median performance but also produces more consistent results across runs.

## A.4 MOA Experimentation

Section 6.7.3 presents an example comparing three diferent modes for modeling other agents. The experimental configuration and results are detailed below.

The environment dynamics follow a Go or Yield experiment. Both agents represent cars that can either Yield or Go. Each episode consists of a single timestep in which both agents select an action, resulting in a collision, one car passing, or both cars stopping.

The selected reward model is the mixed reward model (see Section 4.3.3). The possible joint actions and their corresponding default rewards ρ are defined by the reward matrix presented in Figure 17.

The tested MOA configurations are:

• Independent: the agent does not model the opponent;

• JAL-AM with Action Frequencies: the agent evaluates joint actions and predicts the opponent’s action based on the opponent’s 1000 most recent actions;

![](images/92ff1e51e356e960cb69148698e0ff6f712588d786b659553e7a79d6b37d6509.jpg)

(a) Total of the raw rewards of all agents added together.  
![](images/86221a6b3bdc0928ba7dadc660e3b21c29b7c83f9c418d2be30528ae7e3e18ba.jpg)

(b) Proportion of episodes in which the prey is caught  
![](images/6968da3bc29bf8f681dad1376a89686092ff34c89648ae5ec63fee2d3257d342.jpg)  
(c) Number of timesteps required to catch the prey.  
Figure 16: Results of the prey-hunter experiment. Orange is no communication, green is policy parameter sharing, blue is observation sharing and red is both observation and policy parameter sharing.

Agent 2
<table><tr><td colspan="3">Go Yield</td></tr><tr><td rowspan="2">Agen 1 Go Yield</td><td>(−10,-10)</td><td>(5,0)</td></tr><tr><td>(0,5)</td><td>(−2,−2)</td></tr></table>

Figure 17: Go or Yield reward matrix. In each cell, the reward on the left is received by Agent 1 and the reward on the right by Agent 2.

• JAL-GT with Minimax: the agent evaluates joint actions and assumes the worst-case action from the opponent.

In this experiment, one agent is evaluated while the other serves as its opponent. The learning agent uses a Q-learning algorithm (QLearning) and selects actions using a Q-table. The parameters α and γ are respectively set to 0.2 and 0.95 The policy is epsilon greedy, following the Equation 15. where d is the decay rate, set to 10<sup>−4</sup>, and t is the number of episodes.

The only diference between the learning module configurations in this experiment lies in the policy implementation. Agents that consider the opponent’s actions use QValueBasedJALPolicy, which relies on the MOA module and maps joint actions to expected returns. By contrast, the standard QValueBasedPolicy evaluates only the agent’s own actions.

The opponent is preconfigured with five policies, which it uses sequentially. Each policy defines fixed probabilities of selecting Go or Yield. Every 5000 episodes, the opponent switches to another policy with diferent action probabilities. This setup evaluates whether the tested agent can adapt to changes in the opponent’s behavior. The simulation lasts for 100000 episodes.

Training and execution are decentralized, and no communication mechanism is used in this experiment.

The selected evaluation criteria measure the proportion of episodes resulting in each possible outcome: a collision, both cars stopping, or one car passing.

![](images/aecc5596c537a9afb19918b2cf2906a5e5ff83158fdfb93a4625b866f7407cc5.jpg)

(a) Proportion of episodes resulting in a collision, with both agents selecting Go.  
![](images/a0d0d94e1fd08693314297244f8380c967cc84c9cbdb7edf53842d32c1294d5c.jpg)

(b) Proportion of episodes in which one car passes, with one agent selecting Go and the other Yield.  
![](images/0276857a0a92d11cf5f1bdc84911d69545a8b13fcef355c2a2693c3a4bcec842.jpg)  
(c) Proportion of episodes in which both cars stop, with both agents selecting Yield.

Figure 18: Results of Go or Yield experiment. Green is independent, orange is minimax and blue is action frequencies modeling.

Figure 18 presents the results obtained with the three diferent modeling modes. The curves represent the median across all runs, while the colored areas indicate the range between the first and third quartiles. The data are smoothed using a moving average with a window size of 100.

In this experiment, the three modeling modes lead to significantly diferent outcomes. Minimax assumes the worst-case action from the opponent and therefore tends to select Yield to avoid collisions. Action-frequency modeling maximizes the proportion of outcomes in which one agent selects Go and the other selects Yield. It adapts to the opponent’s recent actions and can therefore exploit policies that frequently select Yield. However, abrupt changes in the opponent’s behavior temporarily increase the collision rate because the model requires time to adapt. The independent configuration exhibits intermediate behavior between the two MOA based configurations.