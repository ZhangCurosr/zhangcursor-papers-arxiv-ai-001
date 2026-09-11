# Distributed Optimization of Modular Production Systems using Model-based Reinforcement Learning with Inverse Models

Andreas Schwung<sup>∗</sup>, Steve Yuwono <sup>∗</sup>, Sofiene Lassoued <sup>∗</sup>, Dorothea Schwung<sup>†</sup>

<sup>∗</sup>South Westphalia University of Applied Sciences, Soest 59494, Germany

{schwung.andreas, yuwono.steve, lassoued.sofiene }@fh-swf.de

<sup>†</sup> Hochschule Dusseldorf University of Applied Sciences, D ¨ usseldorf 40476, Germany¨ dorothea.schwung@hs-duesseldorf.de

Abstract—This paper presents a novel approach for data-driven self-learning control of highly flexible, modular manufacturing systems. Specifically, we employ a novel framework for modelbased reinforcement learning which introduces approximate inverse process models within the training of reinforcement policies. This approach disentangles the learning of actuation dynamics and the dynamics in state space, resulting in RL-based training solely within the task space. We propose a lightweight feedforward architecture for approximate inverse models and integrate them within the policy network of standard RL algorithms. We apply the approach to a laboratory modular production testbed with heterogeneous production modules. The results underline the efficiency improvements for modular manufacturing units in terms of both performance and training speed, particularly for off-policy algorithms.

Index Terms—Reinforcement Learning, Model-based Reinforcement Learning, inverse models, distributed optimization

## I. INTRODUCTION

Complex demands on modern manufacturing processes, specifically the requirements for small lot size production of individualized products, require both highly digitalized production units as well as self-learning and self-reconfiguration capabilities. Such fast-adapting systems offer plug-and-play functions [1] such that near-zero reconfiguration and fast adjustment to different production goals is achievable.

A fundamental prerequisite for the realization of such production systems is a consistently modular system architecture encompassing both the hardware and software levels. While this modularity significantly enhances the flexibility and scalability of the system, it also increases the complexity of its control. Consequently, decentralized and distributed control and communication concepts are required, which are commonly based on the paradigm of Multi-Agent Systems (MAS) [2]. Furthermore, dynamic production environments, continuously evolving requirements, and the demand for resource-efficient and highly productive manufacturing necessitate the integration of self-learning capabilities into the control architecture. To realize such self-learning functionality, various machine learning approaches can be employed, which are generally categorized into supervised learning, unsupervised learning, and reinforcement learning (RL), see [3], [4] for recent overviews.

Among these techniques, RL has emerged as a particularly promising approach for equipping dynamic systems with selflearning capabilities. In RL, an autonomous agent learns to select actions through repeated interactions with its environment, with the objective of maximizing the cumulative reward obtained over time [5]. Building upon the traditional single-agent framework, reinforcement learning has been extended to multi-agent settings, resulting in the field of Multi-Agent Reinforcement Learning (MARL) [6]. Since flexible manufacturing systems are typically characterized by distributed decision-making and decentralized control structures, the MARL paradigm provides a natural and well-suited framework for enabling autonomous learning and coordination in such environments.

Despite the significant progress achieved in RL, its adoption in industrial applications remains limited. This is primarily due to inherent challenges that restrict its practical deployment. One major limitation is the extensive training effort required by most RL algorithms [7]. Moreover, state-of-theart deep RL methods require large amounts of training data before satisfactory performance can be achieved. In industrial settings, acquiring such comprehensive datasets from real production processes is impractical due to limited operating time, associated costs, and the risk of production interruptions.

One approach to reduce both data and training times is modelbased RL (MBRL) in which a model of the system is first learned in a supervised manner using a comparably small dataset from random interactions with the environment. The RL agent is then trained using the model instead of the real environment, thereby significantly reducing the environment interactions. While MBRL is widely applied, the potential of inverting the learned system models to predict appropriate actions given a desired state trajectory has rarely been explored.

In this paper, we bridge this gap by proposing the integration of (approximately) inverse models of the forward system model into policy networks. This allows the use of trained relations between states and actions in both directions, resulting in a partly pretrained policy network. Specifically, this allows us to restrict the learning behavior of the agent to learn only in the task space, meaning the RL agent mainly trains desired state trajectories instead of the more complex to train action trajectories.

The contributions of the paper can be summarized as follows:

• We propose a novel approach to MBRL which employs inverse models within policy networks, thereby disentangling the actuation dynamics from task space learning.

• We propose a lightweight network architecture for (approximate) model inversion based on feedforward neural networks.

• We present the application of the novel approach on a modular production system with very encouraging results in terms of both performance and learning speed.

The paper is structured as follows: Section II discusses the related work. Section III provides the problem statement and presents the proposed framework. Section IV provides results and comparisons while Section V concludes the paper.

## II. RELATED WORK

We review literature on both distributed learning in manufacturing systems and subsequently MARL and MBRL approaches.

## A. Distributed Learning in Manufacturing Systems

Distributed manufacturing systems (DMS) are production processes coordinated in multiple decentralized locations [8]. Unlike traditional systems that are limited to a single location, DMS distribute decision-making and production processes among multiple nodes. Conventional approaches for distributed optimization and control include distributed optimal control [9], distributed Model Predictive Control [10], consensus-based distributed control [11], and Fuzzy Control [12]. However, these approaches typically lack selflearning capabilities and hence, cannot be applied to fastchanging manufacturing systems with varying behavior. Specifically, as they typically rely on fixed models, they are inherently exposed to differences between model and real world.

Recently, self-learning systems like MARL and game theorybased methods have gained significant attention in DMS. While [13], [14] propose model-free and model-based statebased potential games, the works [15], [16] extend the approaches to leader-follower games. However, all these approaches rely on rather simple learning strategies without considering the potential of inverse models.

## B. Multi-Agent and Model-based Reinforcement Learning

As we consider modular systems controlled by distributed controllers, the learning algorithm also has to be chosen in a distributed setting. However, extending single-agent to MARL [6] is not straightforward due to different obstacles discussed in [17]. Specifically, individually learning agents result in a non-Markovian environment for all agents, rendering the Markov decision process (MDP) only partially observable. Early approaches use cooperative settings based on Q-learning, assuming that the actions of other agents improve the collective reward [18]. In recent years, various versions of deep MARL with a centralized critic have been proposed [19]– [22] resulting in the centralized training, distributed execution paradigm, see [23] for an overview. However, this requires extensive communication between the agents. In [24], [25], all agents share a common reward function, which requires constant inter-agent communication between all agents. In contrast, we consider fully distributed learning which allows for plug-and-play of modules in changing environments.

While the previously discussed works use model-free RL, the MBRL approach provides a promising alternative to reduce real-world interactions. MBRL frameworks can be roughly distinguished into world models [26] and planning approaches [27]. While the latter derives control policies similar to model predictive control using Monte-Carlo optimization, MBRL based on world models simultaneously learns an environment model and an optimal policy using standard RL algorithms. However, none of these approaches combine forward and inverse models to incorporate both the world model and policy network. The closest work to ours can be found in [28], where inverse kinematic models have been incorporated into the policy networks of MBRL. However, the approach is restricted to systems modeled by kinematics, which is unsuitable for general dynamic systems.

## III. DISTRIBUTED MBRL WITH INVERSE MODELS

In this section, we present the novel approach for distributed MBRL with inverse models. To this end, we first state the problem and give a system overview, followed by the inverse model architectures and the incorporation into RL algorithms.

## A. Problem Statement

We consider distributed systems consisting of several subsystems which can potentially be left out, included, or interchanged with other modules as illustrated in Fig. 1. Each module is equipped with its own local control system, which are able to communicate with neighboring modules. Note that such systems are common in modern manufacturing environments.

![](images/3635b379191a59a63194671f0980f1a8985af8deaf475e774619f7cb77cc4867.jpg)  
Fig. 1: Considered system structure consisting of several subsystems with their own control system.

Based on the above system description, we state the following problem. Consider the distributed system $s$ with $i = 1 , \ldots , n$ subsystems $S _ { i }$ equipped with local actuation $a _ { i }$ as illustrated in Fig. 1. We define a number of production goals $m _ { i }$ for each module indicated by $e _ { i , j }$ with $i = 1 , \ldots , l , \ j = 1 , \ldots , m _ { i }$ Then, find the optimal production control over a given production episode $t = 0 , \ldots , T$ by optimizing

$$
e ^ { * } ( t ) = \operatorname* { m a x } _ { a _ { i } } \sum _ { t = 0 } ^ { T } \sum _ { i , j } e _ { i , j } ( t ) .\tag{1}
$$

The production goals $e _ { i , j }$ can be arbitrarily defined, with typical examples including product concentrations in chemical reactors, mass flows in bulk good plants or processing times in manufacturing systems.

## B. Framework Overview

After stating the problem, we now present our novel framework for distributed model-based RL, which is illustrated in Fig. 2. First, we set up individual learning agents for each module or actuated system, resulting in n independent agents that only share local state information with their neighboring agents, i.e., agent i receives $s _ { i - 1 } , s _ { i + 1 }$ in addition to its own state information only. The learning agents are implemented within an MBRL architecture using the world model approach [26]. In this setting, each agent first collects data by injecting sufficiently excited input signals and collects data sets $\mathcal { D } _ { i } = \{ s _ { i } ( k ) , s _ { i - 1 } ( k ) , s _ { i + 1 } ( k ) , a _ { i } ( k ) , s _ { i } ( k + 1 ) \} _ { k = 0 } ^ { N }$ consisting of states $s ( k )$ , actions $a _ { i } ( k )$ and state in the next time step $s _ { i } ( k + 1 )$ . Using $\mathcal { D } _ { i }$ , we train a neural network model $f _ { \theta _ { i } }$ with parameters $\theta _ { i }$ for the forward dynamics of the environment, i.e. we train $s _ { i } ( k + 1 ) = f _ { \theta _ { i } } ( s _ { i } ( k ) , s _ { i - 1 } ( k ) , s _ { i + 1 } ( k ) , a _ { i } ( k ) )$ using standard MSE-loss. In vanilla MBRL, after the forward model reaches sufficient accuracy, the policy network is trained on the forward model instead of training with the real environment, thereby strongly reducing real-world interactions.

The core novelty of the approach lies in the design and the training of the policy network. As illustrated in Fig. 2, the policy network is split into two independent networks, namely an inverse model network and a state governor network. The inverse model network $g _ { \gamma _ { i } }$ with parameters $\gamma _ { i }$ outputs actions $a _ { i n v , i }$ based on actual and next states, i.e. $a _ { i n v , i } ( k ) =$ $g _ { \gamma _ { i } } ( s _ { i } ( k + 1 ) , s _ { i } ( k ) , s _ { i - 1 } ( k ) , s _ { i + 1 } ( k ) )$ ). Note that we use the same data set $\mathcal { D } _ { i }$ for the inverse model learning. Further, the inverse model is trained during the ramp-up phase together with the world model and is subsequently frozen during the MBRL learning stage. The state governor network corresponds to the policy network in the original MBRL setting, but with a different purpose. While in vanilla MBRL, the policy network learns to derive optimal actions to the environment, the state governor learns to derive the optimal next state to be visited by the agent. This optimal next state is subsequently sent to the inverse network to derive the required action to reach this state. Hence, this setting disentangles the learning of the task typically defined in the state space from deriving actions required to follow the state trajectories. In parallel, the state governor network also outputs a corrective action component $a _ { p o l , i }$ <sub>i</sub> which is added to $a _ { i n v , i }$ to obtain the final action $a _ { i }$

![](images/1442827f894b23e66f7e6ac7930b8613f6be3ccd870cce6ad081a47a552e11d8.jpg)  
Fig. 2: System overview of the proposed distributed modelbased RL with inverse models.

## C. World models and Inverse world models

After discussing the general framework, we present details about the design and training of the world models. In general, the used network types can be chosen arbitrarily depending on the complexity of the system and inverse dynamics, i.e., we can potentially use standard MLPs, recurrent networks such as LSTMs, or autoregressive architectures like transformers.

For the considered distributed MBRL set-up, where the individual dynamics of the subsystems are typically less complex, we found MLPs for both the forward world model and the inverse model to be sufficient to obtain both training stability and sufficient accuracy. Specifically, we define small decentralized networks, one for each actuator. Each network has a 4 → 128 → 128 → 1 architecture with SiLU activations and a linear output, which is clamped to the range [0, 1]. Both models are trained using MSE regression, i.e., we train

$$
\theta _ { i } ^ { * } \mathrm { = m i n } \mathrm { E } _ { \mathcal { D } _ { i } } ( \| s _ { i } ( k + 1 ) - f _ { \theta _ { i } } ( s _ { i } ( k ) , s _ { i - 1 } ( k ) , s _ { i + 1 } ( k ) , a _ { i } ( k ) ) \| ^ { 2 } )\tag{2}
$$

for the world model network and

$$
\gamma _ { i } ^ { * } \underset { \gamma _ { i } } { = } \underset { \gamma _ { i } } { \operatorname* { m i n } } \mathrm { E } _ { \mathcal { D } _ { i } } \left( \| a _ { i } ( k ) - g _ { \gamma _ { i } } ( s _ { i } ( k + 1 ) , s _ { i } ( k ) , s _ { i - 1 } ( k ) , s _ { i + 1 } ( k ) ) \| ^ { 2 } \right)\tag{3}
$$

for the inverse model. We remark that $\mathcal { D } _ { i }$ only consists of one-step predictions. An extension to multi-step predictions is straightforward using e.g. sequence-to-sequence network architectures and will be considered in future work. We also remark that the training of both networks is done during the ramp-up phase only. Thereafter, both network parameters are kept fixed. We note that we experimented with keeping the inverse model parameters trainable, but with lower performance.

## D. Model Based RL with inverse models

After the model training, we now discuss the training of the policy network using distributed MBRL. To this end, first note that the core difference from the vanilla MBRL algorithm lies in the specific definition of the policy networks. In fact, the novel proposed policy networks essentially consist of the inverse model network $g _ { \gamma _ { i } }$ and the state governor network $h _ { \theta _ { i } } ( s _ { i } ( k ) , s _ { i - 1 } ( k ) , s _ { i + 1 } ( k ) )$ . Let $h _ { \theta _ { i } } ^ { s }$ and $h _ { \theta _ { i } } ^ { a }$ denote the two subnetworks with outputs $s _ { i } ( k + 1 )$ and $a _ { p o l , i } ( k )$ respectively, the resulting policy network $\pi _ { \boldsymbol { \theta } _ { i } }$ is given by (see Fig. 2):

![](images/ab6b37fafc75e70a07a02779613787467811038c6acee71e9df5fac74dbd1e27.jpg)  
Fig. 3: The Bulk Good Laboratory Plant, including the details of its modules, actuators, and reservoirs.

$$
\pi _ { \theta _ { i } } ( s ) = h _ { \theta _ { i } } ^ { a } + g _ { \gamma _ { i } } ( h _ { \theta _ { i } } ^ { s } )\tag{4}
$$

Note that in the above, we assume the inverse model network to be fixed, such that $\pi _ { \boldsymbol { \theta } _ { i } }$ only depends on the learnable parameters $\theta _ { i }$

Applying the specifically defined policy networks, we can straightforwardly employ arbitrary RL algorithms, online and offline, as well as value function-based, actor-critic approaches or the various proposed subversions to the above policy network training [29]. In the results, we will present a comparison of TD3 [30], SAC [31], and DDPG [32] for the considered system application. We focus on off-policy actorcritic methods, where the action-space pathologies the inverse model addresses are most pronounced.

Finally, Algorithm 1 provides the pseudocode of the implementation and learning procedure of MBRL with inverse models.

## IV. EXPERIMENTS AND RESULTS

We apply our methodology to a Bulk Good Laboratory Plant (BGLP). This section contains an overview of the BGLP, the training setup, experimental findings, and comparisons.

## A. Testing Environment: The Bulk Good Laboratory Plant

The BGLP [13] is a distributed manufacturing system for bulk good transport through a network of various actuators and reservoirs. The system has four operational modules: loading, storage, weighing, and filling, as depicted in Fig. 3.

In Module 1, the control parameter for the conveyor belt is the motor rotation speed in rpm, similar to the rotary feeder in Module 3. The vibratory conveyor in Module 2 uses a binary value to indicate whether it is fully operational. For the vacuum pumps in Modules 2, 3, and 4, the control parameter is the turn-on duration in seconds. The simulation of the BGLP is available in the MLPro framework [33]. The simulation is useful for training the proposed MBRL framework before deployment on the real system.

Algorithm 1: Basic Approach of MBRL with inverse   
models.   
Environment: Real environment, World mode   
Data: Actual episode for world model learning and   
MBRL, max. episode for world model learning   
and max. episode for MBRL, max. time duration   
per episode   
Result: Trained $f _ { \theta _ { i } } , g _ { \gamma _ { i } } , \forall i = 1 , \ldots , N ,$ selected action(s)   
/<sub>\*</sub> Model-based learning to derive world   
and inverse model \*/   
initialize hyperparameters, $f _ { \theta _ { i } } , g _ { \gamma _ { i } } , \forall i = 1 , \ldots , N ;$   
while max. episode has not been reached do   
calculate initial states;   
begin an episode;   
while max. time duration has not reached do   
obtain actual states from $f _ { \theta _ { i } } ;$   
calculate subsequent states and rewards;   
generate new action(s) from $A _ { g e n } ;$   
store experience   
$( s _ { i } ( k ) , s _ { i - 1 } ( k ) , s _ { i + 1 } ( k ) , a _ { i } ( k ) , s _ { i } ( k + 1 ) )$ in   
buffer $\mathcal { D } _ { i } ;$   
if buffer $\mathcal { D } _ { i }$ is full then   
sample random minibatch of experiences from   
buffer $H ;$   
train $f _ { \theta _ { i } } , g _ { \gamma _ { i } } , \forall i = 1 , \ldots , N .$ , in a supervised   
manner;   
end   
end   
end   
/<sub>\*</sub> Training MBRL agent in world model   
environment $\star /$   
Run an arbitrary RL algorithm to train state governor   
networks by replacing the real environment with world   
models $f _ { \theta _ { i } } , \forall i = 1 , \hdots , N .$

## B. Training Setup on the BGLP

Within the BGLP, each player i corresponds to an individual actuator and has two pieces of state information, which are the fill levels of the preceding and subsequent reservoirs. Further, two objectives are defined: First, maintaining the fill level of the preceding and subsequent reservoirs to prevent bottlenecks and overflow, $r _ { V } ^ { i }$ , and second, minimizing power consumption, $r _ { P } ^ { i }$ . Additionally, the final player is tasked with an additional objective, such as meeting production demand, $r _ { D } ^ { i }$

The objectives are evaluated at each time step t, resulting in

reward functions:

$$
r _ { V } ^ { i } = \frac { 1 } { 1 + V _ { p } ^ { i } } + \frac { 1 } { 1 + V _ { s } ^ { i } } ; \quad r _ { P } ^ { i } = \frac { 1 } { 1 + P ^ { i } } ; \quad r _ { D } ^ { i } = \frac { 1 } { 1 - V _ { D } } ;\tag{5}
$$

where $V _ { D }$ is production demand fulfilment, $P ^ { i }$ power consumption, and $V _ { p } ^ { i } , V _ { s } ^ { i }$ local constraints to prevent bottleneck and overflow in the buffers. The production demand fulfilment $V _ { D }$ is calculated as

$$
V _ { D } = \int _ { 0 } ^ { T _ { I } } \dot { D } _ { t } d t , \quad \dot { D } _ { t } = \left\{ \begin{array} { l l } { \dot { V } _ { N , o u t } - \dot { V } _ { N , i n } , } & { \mathrm { i f ~ } h ^ { N } = 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{6}
$$

where $\dot { V } _ { N , o u t } , ~ \dot { V } _ { N , i n }$ denote the outflow and inflow to the buffer, $h ^ { N }$ is the normalized value of the fill level, and $T _ { I }$ represents the duration of an iteration. The demand is considered fulfilled if the fill level of the last hopper exceeds the specified production demand. Meanwhile, the mathematical expressions for $V _ { p } ^ { i }$ and $V _ { s } ^ { i }$ are provided as follows:

$$
V _ { p } ^ { i } = \int \displaylimits _ { 0 } ^ { T _ { I } } \mathbb { 1 } _ { q _ { p } ^ { i } < Q _ { p } ^ { i } } ( q _ { p } ^ { i } ) d t , \quad V _ { s } ^ { i } = \int \displaylimits _ { 0 } ^ { T _ { I } } \mathbb { 1 } _ { q _ { s } ^ { i } > Q _ { s } ^ { i } } ( q _ { s } ^ { i } ) d t ,\tag{7}
$$

where $Q _ { s } ^ { i }$ and $Q _ { p } ^ { i }$ represent the upper and lower limits of the associated fill levels.

Based on the above derivations, we apply reward functions adopted from [13], which are outlined as follows:

$$
\begin{array} { r } { R _ { t } ^ { i } = \left\{ \begin{array} { l l } { \omega _ { v } \cdot r _ { V } ^ { i } + \omega _ { p } \cdot r _ { P } ^ { i } + \omega _ { d } \cdot r _ { D } ^ { i } } & { \mathrm { i f ~ } i = N , } \\ { \omega _ { v } \cdot r _ { V } ^ { i } + \omega _ { p } \cdot r _ { P } ^ { i } } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{8}
$$

where $\omega _ { v } , \omega _ { p } , \omega _ { d }$ are pre-defined weights of each objective. In this paper, we apply $\omega _ { v } = 1 . 0 , \omega _ { p } = 0 . 0 0 1 , \omega _ { d } = 4 . 0$

## C. Experimental Results

We now present the experimental results on both the inverse model training as well as the MBRL approach with inverse models. Every experiment is conducted under identical settings and scenarios within the BGLP, where we conduct 100 training episodes, with each episode spanning 10,000 seconds of production time (1,000 cycles) and targeting a production rate of 0.125 and 0.15 L/s. Then, we conduct an additional testing episode utilizing the reward function outlined in Eq. (8).

1) Inverse model training: We start by reporting the results of the inverse model training during the ramp-up phase. Note that training results of typical world model training have already been reported in [14] and are omitted due to space restrictions. The trajectories of the training losses for each actuator are presented in Fig. 4. The loss curves show a consistent decrease, indicating stable training with sufficient accuracy. Fig. 5 shows a comparison of predicted action and ground truth action using scatter plots. As can be seen, all actuators show a very good agreement with their ground truth value (along the straight line) also on unseen data. Note that actuator 3 is discrete operating, resulting in a step function reference.

![](images/048b8883f5b773a8d47685db81d544536287d80475384e98d434350bf9776041.jpg)  
Fig. 4: Training loss of the inverse models for each agent.

![](images/351cc933a787e08e88142157e380969d39720cf033510fcc06bb1e10c85a5feb.jpg)  
Fig. 5: Comparison of inverse model outputs and target value for each actuator.

2) Distributed MBRL with inverse models: Fig. 6 compares the distributed MBRL approaches with and without inverse models for TD3, SAC, and DDPG at production rates of 0.125 L/s and 0.15 L/s. The reference ”without inverse models” corresponds to the same MBRL setting in which the policy outputs actuator commands directly, i.e., the state governor and the inverse model are replaced by a standard policy network. We report demand fulfilment, overflow, and power consumption together with the overall reward and the required training cycles. Two effects are consistent across all six settings: the inverse models reduce the overflow (e.g., from 0.137 L to 0.009 L for TD3 at 0.15 L/s) and they reduce the required training cycles (e.g. from 100k to 30k for DDPG at 0.15 L/s). The effect on the overall reward is algorithm-dependent.

TD3 and DDPG do not solve the task without inverse models: they fail to meet the production demand and overflow heavily, which also renders their apparently low power consumption meaningless, since hardly any material is transported. With inverse models, both meet the demand and improve the reward from 8.0-9.0 to approximately 10.1. SAC, in contrast, already solves the task without inverse models. Here the reward is comparable, while overflow and power consumption are clearly reduced. As expected, the inverse model acts as a form of pretraining of the policy, which makes the overall training easier and results in shorter training times. We attribute the pronounced benefit for TD3 and DDPG to off-policy learning, such as bootstrapping from a replay buffer of raw actions, which makes these deterministic policy-gradient methods latch onto unstable action combinations, whereas learning desired next states delegates action generation to the frozen inverse model.

Regarding computational overhead and sensitivity, the inverse models add negligible cost: each is a small decentralized network (4 → 128 → 128 → 1, roughly 17k parameters per actuator) trained only once during the ramp-up phase and kept frozen thereafter, so at policy-training and deployment time it contributes only a single additional forward pass per step. Regarding sensitivity to model inaccuracies, because the policy operates in task space and the inverse model only decodes a desired next state into an action, a bounded inverse-model error acts as a bounded actuation perturbation that the policy compensates for during training; the clamping of the output to [0,1] additionally limits its effect. This is consistent with our observation that freezing the inverse model outperforms keeping it trainable, indicating robustness to moderate inverse inaccuracies.

![](images/7fb7b4da2325d86929697e698db506e11114795010356157542c55bc21c424fb.jpg)

![](images/ee84be3037bb311f1aa9e296edfe5007c9f3f719e7a485b1514bd2312fbcfde7.jpg)

![](images/607bec406a30ab4ee6e1ecf30438018d2c684a2f30c24e249684492dd84f8280.jpg)

![](images/830b76ec52110f41be445cace61e0fd295b15b8ed73568f9b56b682a0942f2e5.jpg)

![](images/0d34535c15aedb3cb2831162e6be7cbc5a0d1c87e9def99970e29f8e1234516e.jpg)

![](images/120cac51142f6bb6a5e8be35db1460eee2a5112345611d9ae2b6f7a280eda06c.jpg)  
(a) Production demand = 0.125 L/s.  
(b) Production demand = 0.15 L/s.  
Fig. 6: Comparison of distributed MBRL approaches with and without inverse models on the BGLP for TD3, SAC, and DDPG.

## V. CONCLUSION AND FUTURE WORK

This paper introduced a novel data-driven self-learning control framework for modular manufacturing systems based on MBRL. By incorporating inverse process models into the policy learning process, the proposed approach separates the learning of actuation dynamics from the dynamics in the task space, enabling RL to focus on task-space optimization, leading to more efficient policy training. The proposed framework was evaluated on a laboratory-scale testbed. The experimental results demonstrate that the integration of inverse models improves both learning efficiency and performance compared to conventional RL approaches, particularly for off-policy methods. Future work will focus on investigating alternative network architectures, specifically sequence-to-sequence architectures for inverse models, and analyzing the robustness of the approach under changing operating conditions and model uncertainties.

Beyond this, we plan to evaluate scalability on larger lineshaped modular plants and, ultimately, on the real testbed, and to complement the reported results with a statistical evaluation across multiple random seeds to establish confidence intervals on the observed improvements.

## REFERENCES

[1] M. Schleipen, A. Luder, O. Sauer, H. Flatt, and J. Jasperneite, “Re-¨ quirements and concept for plug-and-work – adaptivity in the context of industry 4.0,” at - Automatisierungstechnik, vol. 63, no. 10, pp. 801– 820, 2015.

[2] F. Chen and W. Ren, “On the control of multi-agent systems: A survey,” vol. 6, no. 4, pp. 339–499, Jul. 2019.

[3] J. Wang, Y. Ma, L. Zhang, R. X. Gao, and D. Wu, “Deep learning for smart manufacturing: Methods and applications,” Journal of Manufacturing Systems, 2018, in Press.

[4] T. Wuest, D. Weimer, C. Irgens, and K.-D. Thoben, “Machine learning in manufacturing: Advantages, challenges, and applications,” Production & Manufacturing Research, vol. 4, no. 1, pp. 23–45, 2016.

[5] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. Cambridge, MA, USA: MIT Press, 1998.

[6] L. Busoniu, R. Babuska, and B. De Schutter, “A comprehensive survey of multi-agent reinforcement learning,” IEEE Transactions on Systems, Man, and Cybernetics Part C: Applications and Reviews, vol. 38, no. 2, pp. 156–172, 2008.

[7] D. Schwung, J. N. Reimann, A. Schwung, and S. X. Ding, “Self learning in flexible manufacturing units: A reinforcement learning approach,” in Proceedings of the 9th IEEE International Conference on Intelligent Systems (IS), Funchal, Portugal, 2018.

[8] M. Bortolini, F. G. Galizia, and C. Mora, “Reconfigurable manufacturing systems: Literature review and research trend,” Journal of Manufacturing Systems, vol. 49, pp. 93–106, 2018.

[9] Q. Zhang, F. Chen, J. Yuan, Z. Liu, and G. Feng, “Distributed optimal control of multiagent systems under control variation cost,” IEEE Transactions on Control ofNetwork Systems, vol. 12, no. 1, pp. 104–113, 2025.

[10] A. Aboudonia, A. Eichler, F. Cordiano, G. Banjac, and J. Lygeros, “Distributed model predictive control with reconfigurable terminal ingredients for reference tracking,” IEEE Transactions on Automatic Control, vol. 67, no. 11, pp. 6263–6270, 2022.

[11] J. Gonzalez Silva, T. Keijzer, A. Julian Gallo, R. Ferrari, and J.-W. van Wingerden, “Multirate consensus-based distributed control for largescale wind farms,” IEEE Transactions on Control Systems Technology, vol. 33, no. 5, pp. 1572–1585, 2025.

[12] M. Cui, W. Zhang, and S. Tong, “Adaptive fuzzy distributed optimal event-triggered control for high-order nonlinear multiagent systems,” IEEE Transactions on Fuzzy Systems, vol. 33, no. 9, pp. 3318–3330, 2025.

[13] D. Schwung, A. Schwung, and S. X. Ding, “Distributed self-optimization of modular production units: A state-based potential game approach,” IEEE Transactions on Cybernetics, vol. 52, no. 4, pp. 2174–2185, 2020.

[14] S. Yuwono and A. Schwung, “Model-based learning on state-based potential games for distributed self-optimization of manufacturing systems,” Journal of Manufacturing Systems, vol. 71, pp. 474–493, 2023.

[15] S. Yuwono, A. K. Hussain, D. Schwung, and A. Schwung, “Selfoptimization in distributed manufacturing systems using modular statebased stackelberg games,” Journal of Manufacturing Systems, vol. 80, pp. 578–594, 2025.

[16] S. Yuwono, D. Schwung, and A. Schwung, “Distributed stackelberg strategies in state-based potential games for autonomous decentralized learning manufacturing systems,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2025.

[17] S. Kapoor, “Multi-agent reinforcement learning: A report on challenges and approaches,” 2018.

[18] M. Lauer and M. Riedmiller, “An algorithm for distributed reinforcement learning in cooperative multi-agent systems,” in Proceedings of the 17th International Conference on Machine Learning (ICML), Stanford, CA, USA, 2000.

[19] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, “Multiagent actor-critic for mixed cooperative-competitive environments,” in Advances in Neural Information Processing Systems (NeurIPS), Long Beach, CA, USA, 2017.

[20] J. N. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson, “Counterfactual multi-agent policy gradients,” in Proceedings of the AAAI Conference on Artificial Intelligence, New Orleans, LA, USA, 2018.

[21] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, “The surprising effectiveness of ppo in cooperative multi-agent games,” in Proceedings ofthe 36th International Conference on Neural Information Processing Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

[22] T. Rashid, M. Samvelyan, C. S. De Witt, G. Farquhar, J. Foerster, and S. Whiteson, “Monotonic value function factorisation for deep multiagent reinforcement learning,” J. Mach. Learn. Res., vol. 21, no. 1, Jan. 2020.

[23] C. Amato, “An introduction to centralized training for decentralized execution in cooperative multi-agent reinforcement learning,” 2024. [Online]. Available: https://arxiv.org/abs/2409.03052

[24] P. Sunehag, G. Lever, A. Gruslys, W. M. Czarnecki, V. Zambaldi, M. Jaderberg, M. Lanctot, N. Sonnerat, J. Z. Leibo, K. Tuyls, and T. Graepel, “Value-decomposition networks for cooperative multi-agent

learning based on team reward,” in Proceedings ofthe 17th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), Stockholm, Sweden, 2018.

[25] S. Omidshafiei, J. Pazis, C. Amato, J. P. How, and J. Vian, “Deep decentralized multi-task multi-agent reinforcement learning under partial observability,” in Proceedings of the 34th International Conference on Machine Learning (ICML), Sydney, Australia, 2017.

[26] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, 2018.

[27] C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek, and G. Martius, “Sample-efficient cross-entropy method for real-time planning,” in Conference on Robot Learning, 2020.

[28] M. R. Diprasetya, J. Poppelbaum, and A. Schwung, “Kinenn: Kinematic¨ neural network for inverse model policy based on homogeneous transformation matrix and dual quaternion,” Robotics and Computer-Integrated Manufacturing, vol. 94, p. 102945, 2025.

[29] F. AlMahamid and K. Grolinger, “Reinforcement learning algorithms: An overview and classification,” in 2021 IEEE Canadian Conference on Electrical and Computer Engineering (CCECE), 2021, pp. 1–7.

[30] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in International Conference on Machine Learning, 2018.

[31] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor.” in ICML, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 1856–1865.

[32] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” in International Conference on Learning Representations (ICLR), San Juan, Puerto Rico, 2016.

[33] D. Arend, S. Yuwono, M. R. Diprasetya, and A. Schwung, “Mlpro 1.0-standardized reinforcement learning and game theory in python,” Machine Learning with Applications, vol. 9, p. 100341, 2022.