# Reinforcement Learning-Based Control of CAV Platoon Joining Maneuvers in Mixed Trafic

Biao Yin<sup>a,∗</sup>, Abderrahmane Kasmi<sup>b</sup>, Nadir Farhi<sup>c</sup>

<sup>a</sup>Universit´e Gustave Eifel, SATIE, Gif-sur-Yvette, 91190, France.

<sup>b</sup>Universit´e Paris Dauphine-PSL, Paris, 75016, France.

<sup>c</sup>Cosys-Grettia, Univ Gustave Eifel, F-77454 Marne-la-Vallee, France.

## Abstract

Connected and automated vehicle (CAV) platooning ofers a promising approach to improving road safety and trafic capacity. However, platoon control in real-world trafic is challenging due to uncertainty and heterogeneous driving behaviors. Reinforcement learning (RL) has strong potential for addressing such control problems, but its practical deployment raises challenges related to safety and learning eficiency. This paper proposes a generic modeling and simulation framework for investigating CAV platoon joining maneuvers and comparing deep reinforcement learning (DRL)-based control algorithms. The problem is particularly challenging in mixedtrafic environments, where CAVs coexist with human-driven vehicles exhibiting heterogeneous longitudinal and lateral behaviors. The objective is to achieve safe and eficient joining maneuvers by either incorporating penalties for risky behaviors into the learning process or using an external safety controller to constrain the learned policy. An agent-based modeling framework coupled with the Simulation of Urban MObility (SUMO) simulator is used to evaluate Deep Q-Network (DQN), Double Deep Q-Network (DDQN), and Proximal Policy Optimization (PPO). Results show that PPO outperforms DQN and DDQN, achieving a joining success rate of approximately 98 % and a collision rate below 1 %, largely due to risk-related penalties incorporated into the reward function. However, this improved performance requires more decision steps to complete the maneuver, revealing a trade-of between safety, joining efectiveness, and decision eficiency. An external safety controller efectively prevents collisions, although its interventions may reduce joining eficiency. The results highlight the importance of jointly considering safety and eficiency when designing RL-based controllers for CAV platoon joining in mixed trafic. The source code is available at:

https://github.com/biaoyin/platoon-drl/tree/platoon join

Keywords: Vehicle platooning, Deep reinforcement learning, CAVs, Mixed trafic, Platoon joining, Collision avoidance.

## 1. Introduction

CAVs enable vehicle-to-vehicle and vehicle-to-infrastructure communications to achieve cooperative control and mitigate undesirable human driving behavior (e.g., aggression and stopand-go instability) [1][2]. The benefits of CAVs extend to trafic flow, yielding improvements in safety and eficiency, as well as contributing to environmental sustainability. Abundant studies have focused on longitudinal speed coordination control of CAVs within a platoon, in which a group of CAVs moves at a consensual speed while maintaining a small, nearly constant distance between adjacent vehicles, such as cooperative adaptive cruise control (CACC). It has been demonstrated that CAV platoons have enormous potential to enhance road capacity [3], trafic stability [4], and energy eficiency [5].

Platooning optimization typically involves several key operational phases in the lifecycle of a platoon. These phases structure how vehicles form, operate within, and leave a platoon. Each phase requires sophisticated mechanisms to ensure both safety and eficiency. Although carfollowing dynamics (such as CACC) have been extensively studied, the initial platoon joining phase is fundamentally important, enabling a CAV to merge into a platoon at a designated position. This requires precise speed control and well-timed lane-changing maneuvers. A key challenge lies in managing longitudinal and lateral control that avoids trafic disturbances, such as congestion or collisions. This challenge is further intensified in mixed trafic, where CAVs coexist with human-driven vehicles in the near future.

In previous work, conventional approaches for platoon joining have been mainly explored based on rules, optimization models, or conceptualized techniques such as virtual platoons [6],[7],[8]. Rule-based approaches enable coordinated merging behavior with low computational complexity, but their performance may be limited in highly dynamic environments [6]. Optimization-based approaches, such as model predictive control [7], can improve flexibility and performance by formulating the problem as a constrained optimization problem, while requiring higher computational efort and more accurate modeling. Virtual platooning introduces a higher level of cooperation by organizing vehicles into a logical platoon before physical merging occurs. This enables early synchronization of speeds, precise position adjustment, and proactive gap creation with the target platoon. The technique is mostly used in on-ramp joining scenarios, where the merging vehicle is virtually integrated into the main-road platoon in advance, as demonstrated in [5] and [8].

Recently, reinforcement learning (RL) algorithms have been increasingly adopted to tackle the complexities of platoon-joining tasks. Early foundational work primarily focused on singleagent settings. In the value-based RL domain, [9] introduced a deep Q-network (DQN) solution for platoon merging within a discrete-cell road environment. The results show that the proposed DQN approach requires less merging travel time and fewer vehicle lane-change times than the rule-based approach. Concurrently, policy-based RL methods were explored in [10], which applied proximal policy optimization (PPO) to automated lane-changing maneuvers and achieved a 95% success rate in dense trafic. To address the inherently multi-agent nature of platoon coordination, subsequent research extended RL frameworks to handle inter-agent competition and cooperation. A series of multi-agent RL algorithms have been applied to CAV platoon formation, as demonstrated across studies [11], [12], [13], [14], [15]. However, a critical step toward real-world deployment lies in navigating mixed-trafic environments, where CAVs and humandriven vehicles (HDVs) coexist. In this context, [16] proposed a two-stage framework: the first stage enumerates all feasible platoon formations given a specific vehicle sequence, while the second stage employs a multi-agent policy to guide each vehicle—both automated and humandriven—into its designated position according to the formation plan. To further accelerate convergence, hybrid approaches have also been investigated, such as combining genetic algorithms with deep RL for smart platooning [17]. Beyond formation planning and convergence speed, collision during the joining maneuver remains a persistent and fundamental challenge. The literature reveals two predominant safety strategies. The first appends an independent safety controller—an external, rule-based module—to override potentially hazardous decisions and ensure collision-free execution [18]. The second incorporates internal risky-joining penalties directly within the reward function, thereby discouraging unsafe behaviors intrinsically during the training process [10]. A recent study integrates a control barrier function (CBF) into an RL framework to guarantee a safe joining process in the formation of mixed platoons (including CAVs and HDVs), demonstrating promising results for longitudinal and lateral control [25].

Despite these advances, existing studies indicate that most frameworks and simulations rely on oversimplified or abstracted environments that fail to capture realistic microscopic trafic dynamics. The transition from mixed-trafic lanes to dedicated CAV lanes—a highly practical scenario—has received limited systematic investigation, which was deemed eficient for flow organization with mixed trafic [19]. Most critically, while safety modules (external shields) and penalty-based rewards (internal costs) have been individually explored, there is a lack of comprehensive comparative analysis that evaluates their distinct roles, trade-ofs, and combined efectiveness under diverse and dynamic trafic conditions. To bridge these gaps, this paper proposes a generic RL-based modeling framework built within the SUMO simulation environment to systematically study CAV platoon joining maneuvers.

Our main contributions therefore are threefold: 1) developing a system framework that enables CAVs to join a platoon when operating from a mixed-trafic lane to a dedicated CAV lane on the highway; 2) implementing and comparing three distinct types of deep RL algorithms to learn optimal joining policies; and 3) conducting comprehensive performance analyses under diverse trafic conditions, with a specific emphasis on assessing the roles of risky penalties during training versus the independent safety shield for collision avoidance, thereby providing critical insights into the design of robust and reliable platoon-joining controllers.

The rest of the paper is organized as follows. We introduce the fundamentals of RL and typical deep RL algorithms in Section 2. The detailed methodology, including the simulation system architecture, platooning task modeling, and algorithm implementation, is presented in Section 3. Afterwards, we present the simulation experiments and results analysis. The conclusion is drawn in the last section.

## 2. Background Knowledge

## 2.1. Fundamentals of RL

RL is designed for an agent to learn by interacting with the environment. The agent learns how to make decisions by taking actions and receiving feedback from the environment. Specifically, the decision-making process is modeled as a Markov decision process (MDP), defined by the tuple $( S , { \mathcal { A } } , P , r , \gamma )$ , where S and A denote the state and action spaces, respectively, $P ( s ^ { \prime } | s , a )$ represents the state transition probability, $r ( s , a , s ^ { \prime } )$ is the reward function, and $\gamma \in ( 0 , 1 ]$ is the discount factor for future reward. At each time step t, the agent selects action $a _ { t }$ based on the current state of the environment $s _ { t }$ . The environment then responds by transitioning to a new state $s _ { t + 1 } \sim P ( \cdot | s _ { t } , a _ { t } )$ and returns a reward $r _ { t + 1 } = r ( s _ { t } , a _ { t } , s _ { t + 1 } )$ The goal of RL is to learn a policy $\pi : { \mathcal { S } }  A$ in the form of a condition probability $\pi ( a _ { t } | s _ { t } )$ to maximize the expected cumulative discounted reward in infinite or over a finite horizon $T .$ defined as $J ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t + 1 } \right]$ . Q-learning is a classical RL method based on the Bellman Equation and seeks to maximize the objective $J ( \pi )$ by updating the estimate of $Q ( s , a )$ iteratively.

## 2.2. DQN and DDQN

When deep neural networks are used to approximate the value function $( \mathrm { e . g . }$ , the Q-function) or the policy, the approach is known as deep RL. As one of the deep RL algorithms, DQN uses the same Q-value maximization for both action selection and evaluation, although the online and target networks are maintained separately. The target value function is given by,

$$
y ^ { \mathrm { D Q N } } = r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s ^ { \prime } , a ^ { \prime } ; \theta ^ { - } )\tag{1}
$$

where θ<sup>−</sup> are the weights of the target network. The goal is to find θ in the Q-network that estimates the best $Q$ function, i.e., $Q ( s , a ; \theta ) \approx Q ^ { * } ( s , a )$ by minimizing the following loss function:

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } \left[ ( y ^ { \mathrm { D Q N } } - Q ( \boldsymbol { s } , \boldsymbol { a } ; \boldsymbol { \theta } ) ) ^ { 2 } \right]\tag{2}
$$

where $\theta$ can be optimized using gradiant decent and $\theta ^ { - }$ is periodically cloned from $\theta \ \mathrm { ( e . g . , \ a \ }$ soft update strategy) to maintain stability.

Due to noise in the estimated $Q ( s , a ; \theta )$ , the max operator can lead to a systematic overestimation bias. Therefore, DDQN [21] is proposed to use two diferent networks (with the same architecture), where the online network picks the best action, and the target network evaluates its value, as shown in Eq.(3).

$$
y ^ { \mathrm { D D Q N } } = r + \gamma Q ( s ^ { \prime } , \mathrm { a r g m a x } Q ( s ^ { \prime } , a ^ { \prime } ; \theta ) ; \theta ^ { - } )\tag{3}
$$

where argmax uses the online weight θ and the Q value uses the target weights $\theta ^ { \cdot }$ <sup>−</sup>. The loss function in Eq. (2) then uses $y ^ { \mathrm { D D Q N } }$ instead of $y ^ { \mathrm { D Q N } }$

Normally, the ϵ-greedy policy is used for action selection. The best action is chosen with probability $1 - \epsilon$ and the random action selection with probability ϵ. Specifically, we adopt the time-dependent $\epsilon ( t )$ determined by an exponential decay function as shown in Eq. (4). This $\epsilon ( t ) = \epsilon _ { \mathrm { m i n } }$ is satisfied after a large number of simulation steps of $\epsilon _ { \mathrm { d e c a y } }$ , which means the best actions are mostly chosen in the final convergence.

$$
\epsilon ( t ) = \exp \bigg ( \log \bigl ( \epsilon _ { \mathrm { s t a r t } } \bigr ) + \frac { t } { \epsilon _ { \mathrm { d e c a y } } } \bigl ( \log \bigl ( \epsilon _ { \mathrm { m i n } } \bigr ) - \log \mathopen { } \mathclose \bgroup \left( \epsilon _ { \mathrm { s t a r t } } \bigr ) \aftergroup \egroup \right) \bigg )\tag{4}
$$

For both algorithms, a replay bufer is used, in which a batch of stored experience quadruplets (state, action, reward, new state) is randomly sampled for θ updates.

## 2.3. PPO

PPO does not use a Q-target network. It updates its policy using an advantage function in Eq.(5) (usually computed via GAE - Generalized Advantage Estimation) to tell the agent how much better an action is compared to the average.

$$
\hat { A } _ { t } = \delta _ { t } + ( \gamma \lambda ) \delta _ { t + 1 } + . . . + ( \gamma \lambda ) ^ { T - t + 1 } \delta _ { T - 1 }\tag{5}
$$

where $\delta _ { t } = r _ { t } + \gamma V ( s _ { t + 1 } ) - V ( s _ { t } )$ and λ is the GAE parameter. The objective in PPO is to maximize a surrogate objective function (see Eq.(6)) by using importance sampling to safely reuse old data to update the neural network’s weights θ.

$$
L ^ { \mathrm { P P O } } ( \theta ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } ( r _ { t } ( \theta ) \hat { A } _ { t } , \operatorname { c l i p } ( r _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t } ) \right]\tag{6}
$$

The related PPO algorithm can be seen in [10].

## 3. Methodology

## 3.1. System architecture

To realize the agent-based learning process in a trafic dynamic simulation environment, we illustrate the built system architecture in Fig. 1.

The system consists of two main components: the environment and the agent. For the environment, two-lane highway trafic is modeled in SUMO, along with vehicle characteristics and trafic load variations over time. In the network configuration, the inner lane is dedicated exclusively to CAV platoons, and the outer lane is a mixed lane that includes both HDVs and CAVs. The speed limits of these two lanes can be diferent. Besides the embedded longitudinal and lateral control models in SUMO for HDVs, the environment includes a CAV control module – PLEXE [22], which is a cooperative driving framework extended in SUMO to enable automated driving behavior simulation, such as ACC and CACC. To facilitate communication with the learning agent, the environment is encapsulated as a Gym environment, which is a standard open-source library used for many RL applications. The current environmental information around the ego CAV is transmitted via the SUMO Traci API to construct the state representation, which is used as input for the agent-based learning model. The model predicts values of possible actions in terms of longitudinal or lateral driving control, and the selected action is sent back to the ego CAV for maneuver execution. Notably, the built system architecture is supposed to be transferable for other agent-based trafic control problems, such as trafic signal control [23] and on-ramp merging [24].

![](images/6b5726a956ef63aa2bd5efa0750a4661e99d113308452cea8b0b90e1656e0419.jpg)  
Figure 1: System architecture for the RL-based agent modeling and simulation

## 3.2. CAV platoon formation problem

In this study, we assume that the joining CAV enters at the rear of the target platoon and leaves from the rear position. All members in the platoon from the head to the last are organized in increasing order according to their destinations (i.e., exits of the highway). This strategy reduces the need for internal reordering and mitigates trafic disturbances. Focusing on our CAV platoon formation, the control procedure consists of three phases: 1) CAV joiner and platoon selection, 2) joining maneuver, and 3) gap closing. Fig. 2 illustrates the workflow of platoon formation. Regarding the life cycle of the platoon, an additional process for exiting platoons should be integrated. This process is not in the scope of this paper. The above three main phases for the platoon formation are introduced below.

Joiner and target platoon selection.. A CAV is randomly selected from the set of joiner candidates on the mixed-trafic lane (i.e., outer lane). The set of joiner candidates consists of CAVs that must have the possibility to choose one of the existing platoons within their perception range. Here, the perception range is not the same as the communication zone, in which joining requests can be reached as far as possible. We define the perception range as follows: if the joiner is at position x meters (m), the last member of the target platoon should be located in the range of $[ x - \delta _ { 1 } , x + \delta _ { 2 } ]$ $\delta _ { 1 }$ is the distance behind the joiner vehicle to ensure the maximum platooning size, and it must be less than the radius of the communication zone R. $\delta _ { 2 }$ is a short distance ahead of the joiner vehicle for following the target platoon in a short time. In our study, we set $\delta _ { 1 } = 1 5 0$ m, $\delta _ { 2 } = 5$ m and $R = 2 0 0 ~ \mathrm { m }$ . When platoons receive the joining request from the selected CAV joiner, they may accept or refuse the joining request, subject to their availability. The platoon’s availability depends on two conditions: 1) no ongoing joining/splitting process by another CAV, and 2) satisfying the ordered destinations regarding the existing CAV members in the platoon. Under these constraints, a joiner vehicle will be finally confirmed to join one of the available platoons in its neighborhood. If there is no platoon in the zone, a CAV can move to the inner lane under the default lane-change model to form an initial platoon containing only one vehicle.

![](images/f7b1c562e820bc3374f2c78b54a6f5fef468b8a54d8448e195d0bae9c263bae8.jpg)  
Figure 2: Workflow of platooning control

Joining maneuver.. Once the CAV joiner and the target platoon are selected, the joining maneuver (either lane change or speed adjustment) starts under control by the agent learning model. This phase mainly focuses on implementing the deep RL algorithms (i.e., DQN, Double DQN, and PPO). The CAV joiner will learn from the surrounding trafic conditions (i.e., state), intending to join the target platoon by executing an appropriate driving maneuver (i.e., action), through either adjusting longitudinal speed or switching to the inner lane, under the assessment of the cumulative reward. The configuration of each component of the deep RL algorithms will be detailed in Section 3.3. To avoid collisions, related safety constraints such as time-to-collision (TTC) can be applied during the algorithm training or before the execution of lane changes. We will initially take TTC-related collision risks into account during the model training for the purpose of assessing the learning performance. A TTC-related safety shield will be considered for a comparison as needed.

Gap closing.. After the ego vehicle $V _ { e }$ arrives at the target platoon lane, it should approach the target platoon and minimize the coordinated distance $d _ { c r d } ~ ( \mathrm { e . g . , ~ 5 ~ m } )$ to its predecessor. To achieve this, we use the embedded PLEXE package for the CACC implementation, which allows the ego vehicle to synchronize its behavior to the target platoon members and complete its formation. Since the inner lane is dedicated exclusively to platoons composed of CAVs and the ego vehicle is correctly positioned after the lane-change maneuver, this simple mechanism is suficient to maintain the desired headway among platoon members.

## 3.3. RL for CAV agent modeling

State representation.. For a CAV joining a platoon at the right moment, it should have a vision of the surrounding vehicles. This is crucial for the ego vehicle to adapt speed and know when to change lanes without causing collisions. Fig. 3 illustrates the CAV joiner and its surrounding vehicles as follows:

$V _ { e } { : }$ the CAV ego vehicle, i.e., the joiner.

$V _ { p m } \mathrm { : }$ : the predecessor of the ego vehicle on the mixed lane $m$

$V _ { f m } \colon$ the follower of the ego vehicle on the mixed lane m.

$V _ { p p } \mathrm { : }$ the predecessor of the ego vehicle on the platoon lane $p .$

![](images/aca3e01bf8318e349f6caaa1f63da75501b3037cd89e9eb7f4655995bcd2b35c.jpg)  
Figure 3: Ego vehicle and surrounding vehicles

$V _ { f p } \mathrm { : }$ : the follower of the ego vehicle on the platoon lane $p .$

$V _ { l t } \mathbf { : }$ the leader of the target platoon.

$V _ { t a r g e t } \mathrm { : }$ the last member of the target platoon. It can be the same as $V _ { l t }$ if the platoon initially consists of a single vehicle.

$V _ { l f } { \mathrm { : } }$ the leader of the following platoon.

Here, the state is a vector of 15 variables given by Eq. (7).

$$
\begin{array} { r l } { s = \{ } & { { } v _ { e } , v _ { p m } , v _ { f m } , v _ { p p } , v _ { f p } , v _ { t a r g e t } , v _ { l f } , d _ { p m } , d _ { f m } , d _ { p p } , d _ { f p } , d _ { t a r g e t } , d _ { i n t } , l _ { e } , l _ { p } \} } \end{array}\tag{7}
$$

where $v _ { e } , v _ { p m } , v _ { f m } , v _ { p p } , v _ { f p } , v _ { t a r g e t }$ , and $v _ { l f }$ are respectively the speeds of the ego vehicle $V _ { e }$ , its predecessor and follower vehicles on the mixed lane $( \mathrm { i . e } , V _ { f m } , V _ { f m } )$ , its predecessor and follower vehicles on the platoon lane $( \mathrm { i . e . , } V _ { p p } , V _ { f p } )$ , and the predecessor $V _ { l t }$ and follower $V _ { l f }$ of the target joining position; $d _ { p m } , d _ { f m } , d _ { p p } .$ , and $d _ { f p }$ are the absolute distances between the ego vehicle and the predecessor/follower vehicles on the two lanes; $d _ { t a r g e t }$ is the directional distance between the ego vehicle and the last vehicle in the target platoon (notice that $d _ { t a r g e t }$ is negative when $V _ { t a r g e t }$ is behind of $V _ { e } ,$ otherwise.); $d _ { i n t }$ is the absolute value of the distance between the rear of $V _ { t a r g e t }$ and the head of $V _ { l f }$ . Except for $d _ { i n t }$ , for simplification, all distances are calculated as head distances between the ego vehicle and the related surrounding vehicles. $l _ { e }$ and $l _ { p }$ are the indices of the ego vehicle’s located lane and the platoon lane. We assume that related vehicle information can be acquired in time within the communication zone. In a low-density trafic situation, the predecessor and follower vehicles may be partially missing. We adopt the default limit speed on the lane and the communication range $R$ as substitutes, $\mathrm { e . g . } , v _ { p m } = 1 0 0$ km/h and $d _ { p m } = 2 0 0$ m when no vehicle is ahead of $V _ { e }$

Action space.. Our action space consists of six discrete actions, including the lane change action $a _ { 0 }$ with a current speed and five actions for speed adjustements without lane changes, which are defined as follows: $a _ { 1 } = 2 \mathrm { m } / \mathrm { s } ^ { 2 } , a _ { 2 } = 1 \mathrm { m } / \mathrm { s } ^ { 2 } , a _ { 3 } = 0 \mathrm { m } / \mathrm { s } ^ { 2 } , a _ { 4 } = - 1 \mathrm { m } / \mathrm { s } ^ { 2 }$ , and $a _ { 5 } = - 2 \mathrm { m } / \mathrm { s } ^ { 2 }$ To simulate an action in a reasonable implementation, we set a decision-action interval $\Delta t$ consisting of five simulation steps (i.e., a total of 0.5 s due to the simulation time step of 0.1 s), meaning that once an action is chosen, the next action can only be taken after five simulation steps. The decision-action interval should ensure the time delay from the decision-making process and the communication between vehicles.

Reward function.. To achieve an eficient and safe joining maneuver, we propose a reward function with sub-rewards in Eq. (8), implying a bonus of a successful joining and penalties in other cases such as a failure joining, a joining with a collision, a delayed joining, and unstable joinings (either being risky as too close to the last member of the target platoon or being ineficient as farway behind). The reward formulation is presented as follows:

$$
\boldsymbol { r } = \alpha _ { s } ( \boldsymbol { r } _ { s } + \boldsymbol { r } _ { r } + \boldsymbol { r } _ { u } ) + \boldsymbol { \alpha } _ { f } \cdot \boldsymbol { r } _ { f } + \boldsymbol { \alpha } _ { c } \cdot \boldsymbol { r } _ { c } + \boldsymbol { \alpha } _ { k } ( \boldsymbol { r } _ { r } + \boldsymbol { r } _ { j } )\tag{8}
$$

where:

$\alpha _ { s } , \alpha _ { f } , \alpha _ { c }$ , and $\alpha _ { k }$ are dummy variables. For each variable, a value of 1 indicates that the corresponding status (success, failure, collision, or lane keeping) occurs, while 0 indicates otherwise.

• $r _ { s }$ is a success bonus. A successful joining maneuver leads the ego vehicle $V _ { e }$ to the target position behind the predecessor $V _ { t a r g e t }$ in the target platoon. The success bonus is set to 100.

$r _ { r }$ is the risky maneuver penalty, although the joiner gets a successful join. It is related to two alternative penalties for a risky situation of either changing to the platoon lane $( \alpha _ { s } = 1 )$ or staying on the mixed lane with speed adjustments $( \alpha _ { k } = 1 )$ , according to the measures of rear and front TTC. We set it as:

$$
r _ { r } = \left\{ \begin{array} { l l } { - 1 0 0 , } & { i f \operatorname* { m i n } ( T T C _ { ( V _ { e } , V _ { i f } ) } , T T C _ { ( V _ { e } , V _ { t a r g e t } ) } ) < 2 } \\ & { a n d \alpha _ { s } = 1 , } \\ { - 5 0 , } & { i f \operatorname* { m i n } ( T T C _ { ( V _ { e } , V _ { f m } ) } , T T C _ { ( V _ { e } , V _ { p m } ) } ) < 1 } \\ & { a n d \alpha _ { k } = 1 , } \\ { 0 , } & { o t h e r w i s e . } \end{array} \right.\tag{9}
$$

$r _ { u }$ is the unstable maneuver penalty. This is related to the distance (either too close or too far) to the target platoon after joining. We consider the range of safe positions between 10 m and 20 m without any penalty. Eq.(10) represents how this sub-reward is formulated:

$$
r _ { u } ( d ) = \left\{ \begin{array} { l l } { - 2 ( 1 0 - d ) , } & { i f \ d < 1 0 , } \\ { 0 , } & { i f \ 1 0 \leq d \leq 2 0 , } \\ { - 0 . 5 ( d - 2 0 ) , } & { o t h e r w i s e . } \end{array} \right.\tag{10}
$$

where d is the immediate distance between the joiner $V _ { e }$ and the vehicle $V _ { t a r g e t }$ after a joining maneuver (i.e., only completing a lane change before the gap closing).

• $r _ { f }$ is the failure penalty. Any joining maneuver by which the joiner $V _ { e }$ is not placed after the predecessor $V _ { t a r g e t }$ is considered a failure. A failure also occurs when the joiner exits the network before performing the joining maneuver. We set the failure penalty to −50.

• $r _ { c }$ is the collision penalty. To avoid collisions, we set a heavy penalty of -10,000 in a collision case that occurs either on the platoon lane or on the mixed lane.

• $r _ { j }$ is the jerk penalty. This penalty relates to excessive longitudinal acceleration and jerk when actions are taken to keep the current lane with the speed adjustment. We penalize the jerk if it satisfies $| \dot { a } | > 4 \mathrm { m } / \mathrm { s } ^ { 3 }$ (when $\alpha _ { k } = 1 )$ , reflecting an uncomfortable driving experience. In our study, the penalty is set as:

$$
r _ { j } = \left\{ \begin{array} { l l } { - \left| \dot { a } \right| , } & { i f \left| \dot { a } \right| > 4 , } \\ { 0 , } & { o t h e r w i s e . } \end{array} \right.\tag{11}
$$

![](images/4488d193661a50d092681a6239149672480501a0f30311e98b3869354f41b905.jpg)  
Figure 4: Reward evolutions

## 4. Simulation and Results

## 4.1. Experimental setup

To build a mixed CAV trafic environment in SUMO, we adopt a CAV penetration ratio of 50% on the mixed lane coexisting with HDVs and set 100% CAVs on the platoon lane. The default car-following model (IDM) and lane-change model (LC2013) are applied only for HDVs. Thanks to the package PLEXE embedded in SUMO, for CAVs not in platoons (i.e., on the mixed lane) and platoon leaders (i.e., on the platoon lane), the ACC car-following model is adopted; on the platoon lane, the CAV ego vehicle adopts the CACC car-following model to catch up to the target platoon. Once a CAV is selected as a platoon joiner, its ACC or CACC is deactivated, and lane changes and speed adjustments are governed by the deep RL algorithms. Table A.1 and Table A.2 show the hyperparameter settings for training/testing the value-based models of DQN and DDQN, and the policy-based PPO model, respectively. Table A.3 gives the configuration for simulating trafic scenarios. All experiments were performed on a server featuring a 2.4 GHz CPU with 64 processors and 512 GB of RAM.

## 4.2. Model training performance

Before training, a warm-up phase of 1,200 simulation steps is launched in SUMO to allow vehicles to spread out in the network, and platoons are also initialized during the phase. Specifically, in our configuration, we set event as the unit of executed joining status (i.e., success, failure, or collision). Each training episode consists of 1,000 decision-making steps, thus resulting in fewer than 1,000 events. That is to say, one event could occur within multiple decision steps, as a selected joiner can either stay on its current lane due to an adopted action for just speed adjustment, or give up its platoon joining if potential collision risks are detected (see safety shield tests in Section 4.4). We train the model by 10<sup>6</sup> decision steps with incremental levels of random trafic. The training time costs are: 46.8 hours for the DQN, 18.9 hours for the DDQN, and 12.1 hours for the PPO.

The generated average rewards per recording log iteration of the last 100 decision steps are shown in Fig. 4. To make the comparison observable, we smoothed the rewards to represent their evolution, which increases over iterations. Overall, PPO outperforms the other two algorithms, especially during the two periods before 4,000 iterations and after 8,000 iterations, which almost correspond to low and high trafic load scenarios. Compared to DQN, the PPO and DDQN algorithms have relatively more stable and larger mean rewards (<-50) for the last periods. Small negative rewards indicate that collisions, failures, or risky joining occur very occasionally.

We present the operational performances of the three algorithms in Fig. 5. The evolutions of success, collision, and failure rates are shown respectively in Fig. 5(a), (b), and (c). By using a smoothing spline fit, they exhibit similar tendencies where the success rates increase, while the failure and collision rates decrease over time. Compared to DQN and DDQN, obviously, PPO achieves superior performance, which has not only the fastest convergence (around 3,000 iterations) but also the most eficiency with the highest success rate and lowest failure/collision rates. DDQN has relatively lower collision rates than DQN. On the contrary, its performance in terms of success rates and failure rates is a bit worse than DQN. Fig. 5(d) represents the convergence of decision steps per event that occurred. PPO decreases the number of decision steps for convergence, while DQN and DDQN are opposites. They all fall in the range of 10 ∼ 15 decision steps, namely between 5 and 7.5 seconds (as mentioned, one decision step is 0.5 seconds). Among them, PPO seems more conservative for platoon joining by taking a few more steps. This can also be seen in Fig. 5(e), where its cumulative number of events is less than the other two algorithms entirely. From this, it explains well why PPO consumed so much less time to complete the training.

Regarding the value-based DQN and DDQN algorithms, initially, the agent randomly selects actions to explore the environment with a strong possibility of a high exploration rate ϵ. In cases where the selected vehicle’s target position is behind, for instance, any early lane-change manoeuvre will lead to either a collision or a join in the wrong position. This is why, at the beginning of training, the failure rate is very high (> 70%), and the collision rate reaches 10%. Successful join maneuvers occur occasionally (20%). With ϵ decreasing exponentially over time, the probability (i.e., 1 − ϵ) of selecting a greedy action gets higher. The agent learns to successfully join the platoon, reaching approximately 98%, and the failure and collision rates tend to be smaller near zero. We should mention that at the end of training, the success rate still oscillates because we keep a small exploration rate $( \epsilon _ { \mathrm { m i n } } = 0 . 0 1 )$ to ensure continuous learning. This means the agent selects random actions with a probability of 1%, which may lead to collisions or failures occasionally.

## 4.3. Model testing analysis

## 4.3.1. Joining manoeuvre and speed synchronization

We analyze the joining maneuver based on the speed profiles of the joiner samples, as illustrated in Fig. 6. The solid lines represent the joiners staying on the mixed lane or starting the lane change (controlled by the deep RL algorithm), and the dotted lines represent ongoing lane changes and catching up with the last members of the platoons (controlled by CACC within the PLEXE). In most cases, CAVs on the mixed lane are ahead of the target platoons when they are selected as joiners. That is why the joiner briefly decelerates to allow the target platoon to pass. Once the last platoon member has moved ahead, the joiner initiates a lane change and accelerates to catch up with the platoon. The joiner slows down again to synchronize with the platoon speed by adopting the CACC mechanism.

## 4.3.2. Joining quality

We first investigate joining quality regarding the average number of decision steps until completing the joining, as shown in Fig. 7. Boxplots use the standard 1.5×IQR whisker definition. Outlier markers are omitted for clarity. As previously shown in Fig. 5(d) PPO has more decision steps per event than the other two algorithms in convergence, we show in detail the distributions for diferent trafic load levels. With higher symmetric trafic loads in Fig. 7 (a), the number of decision steps gets smaller, especially for the DDQN. This may be explained by the decreased vehicle speeds due to high trafic density on both lanes. In cases with asymmetric loads (see Fig. 7 (b)), PPO takes many more decision steps for platoon joining than DQN and DDQN do when the trafic on the platoon lane is much more than that on the mixed lane, i.e., (500, 2000) and (1000, 2000). This diference mitigates a lot for the opposite demand cases. This implies that PPO’s control of platoon joining is more conservative and dynamic when the platoon lane has much higher trafic density than the mixed lane. To some extent, DQN and DDQN both reflect their joining possibilities within stable decision steps when meeting largely asymmetric trafic loads on highways.

![](images/1f3302084eb0045f726a16c2ebf64ae2e5e08edc2b0715cd0b567c31acdde9c3.jpg)  
(a) Success rate

![](images/0d2f28834192d93d72ff7b3e7db9d2b6e19671741eb792259a98d4713126f45d.jpg)  
(b) Failure rate

![](images/3c33a2f65d0574e8e164f058e668f470f7a5d5aa5e093d8a444225a63874f399.jpg)  
(c) Collision rate

![](images/87708718c156990cd36506d4784996dcbccb9718aaabc8d40e0f9f31912aa32c.jpg)  
(d) Decision steps per event

![](images/8636261ccd820b7cd818f11d40d969a4c1be12c739093aad9c26742b6b985a7b.jpg)  
(e) Events  
Figure 5: Comparisons of training performance.

We then check the joining distance between the joiner and its vehicle ahead (i.e., the last member of the target platoon), as shown in Fig. 8. As a whole, the distributions of average joining distances seem very stable in symmetric loads, and they are located between 10 and 20 meters. In asymmetric loads, all three algorithms generate a bit longer distances for the cases of low trafic on the platoon lane, i.e., (1500, 500) and (2000, 500). No matter what, the observations are aligned with our previous stable-maneuver setting, where no penalty is valued in this range during the training procedure (see Eq. (10)).

![](images/aa7292562cbae2901880432cb4e648c6b016d8ea87d18dfd8c038461938f11ab.jpg)  
Figure 6: Samples of joiners’ speed evolutions

## 4.3.3. Robustness analysis

The success, failure, and collision percentages of platoon joining under diferent trafic loads were recorded to assess the robustness of the three types of deep RL approaches, as shown in Fig. 9. The solid lines refer to the performance of the designed algorithms, whereas the dashed lines are those without taking the TTC penalty (i.e., r<sub>r</sub> in Eq. (9)) into account. Obviously, PPO presents the best performance with more than 97.5% success rate (see Fig. 9(a)) and less than 1.0% collision rate (see Fig. 9(c)) for all levels of trafic loads. Unexpectedly, the success rates of DDQN and DQN drop a lot, even under 90.0% in the four trafic demand pairs of (500, 2000), (1000, 2000), (1500, 2000), and (2000, 2000). This can be explained by the large rates of failed joinings (including wrong joining positions and uncompleted joinings) due to the platoon lane with the highest trafic of 2000 veh/h (see Fig. 9(b)). Compared to the variants of the three algorithms without TTC penalty, there are no surprisingly large failures because of the removed TTC constraints. However, this causes most of their collision rates to be relatively larger than those of the designed algorithms with the TTC risky penalty, as shown in Fig. 9(c). Based on these results, the policy-based PPO algorithm seems more robust than the value-based DDQN and DQN algorithms, and PPO is even better when the TTC risky penalty is included in the reward function.

## 4.4. Safety shield for lane changes

The importance of the risky maneuver penalty designed in the reward function was demonstrated above. Diferently regarding this ”inner” safety mode, we also intend to investigate external protection when the algorithms are designed without the inner TTC conditions, causing relatively high collision rates for the high trafic demands on the platoon lane (see Fig. 9(c)). A safety shield (noted as ”SS”) beyond the lane-changing decisions made by the deep RL algorithms was implemented. The rear and front time-to-collision (TTC) conditions are both applied before platoon joining or speed adjustment on the mixed lane, following a decision by the learning algorithm, with the same settings as in the reward function design. Specifically, they are min $( T T C _ { ( V _ { e } , V _ { l f } ) } , T T C _ { ( V _ { e } , V _ { t a r g e t } ) } ) \ > = \ 2$ for the lane-change execution; and 2) min $( T T C _ { ( V _ { e } , V _ { f m } ) } , T T C _ { ( V _ { e } , V _ { p m } ) } ) ^ { \prime } > = 1$ for speed adjustement actions. The obtained results of metrics are given in Table 1. It shows that collision rates are almost reduced to zero $( < 0 . 0 0 1 )$ and they are transferred as abandonment. Notice that very small parts of previous successful joinings are also abandoned. The abandoned joinings are related to previous dangerous but successful joinings with a small TTC $( \mathrm { i . e . , < 2 ~ s ) }$ . Again, PPO shows relatively higher success joining rates $( > = 9 6 . 8 \% )$ with higher density trafic, followed by DDQN, which even demonstrates the best joining eficiency reaching 97.1% in the asymmetric trafic case of (500, 2000). Mention that the safe PPO approaches proposed in [18] only achieved lane-changing goals of 75% to 88% with free collisions.

![](images/0ad5d5014092441fdecd7002fa077e94bd3db580f36a4f5620fbbcf15a834bbb.jpg)  
Traffic loads on mixed lane and platoon lane

(a) Symmetric loads  
![](images/d31e045b8e773d7ad095fde399cdd45ddc88eb2bbe734272a29d13add6aa6368.jpg)  
(b) Asymmetric loads  
Figure 7: Number of decision steps per event.

Table 1: Joining performance of algorithms after applying a safety shield (SS)
<table><tr><td></td><td>Metrics (%)</td><td>(500, 2000)</td><td>(1000, 2000)</td><td>(1500, 2000)</td><td>(2000, 2000)</td></tr><tr><td rowspan="4">DQN (SS instead of TTC penalty)</td><td>Success</td><td>93.7</td><td>93.3</td><td>92.8</td><td>91.6</td></tr><tr><td>Failure</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.1</td></tr><tr><td>Collision</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Abandonment</td><td>6.1</td><td>6.5</td><td>7.0</td><td>8.3</td></tr><tr><td rowspan="3">DDQN (SS instead of TTC penalty)</td><td>Success</td><td>97.1</td><td>96.3</td><td>96.3</td><td>95.5</td></tr><tr><td>Failure Collision</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.2</td></tr><tr><td>Abandonment</td><td>0.0 2.8</td><td>0.0 3.4</td><td>0.0</td><td>0.0</td></tr><tr><td rowspan="4">PPO (SS instead of TTC penalty)</td><td>Success</td><td>94.4</td><td>96.1</td><td>3.6 96.9</td><td>4.3 96.8</td></tr><tr><td>Failure</td><td>0.4</td><td>0.4</td><td>0.3</td><td></td></tr><tr><td>Collision</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.3</td></tr><tr><td>Abandonment</td><td>5.2</td><td>3.5</td><td>2.8</td><td>0.1 2.8</td></tr></table>

![](images/462f9ce6b1ad826c09b89958e0e91af92a467e944dc71ba4a664add531ca9d99.jpg)  
Traffic loads on mixed lane and platoon lane

(a) Symmetric loads  
![](images/d0d23cad946e70f3ff5abda5d831de6e32ab786cd7d647ff101194976d12669b.jpg)  
Traffic loads on mixed lane and platoon lane  
(b) Asymmetric loads  
Figure 8: Joining distance between $V _ { e }$ and $V _ { t a r g e t }$

## 5. Conclusion

In this paper, we proposed a deep RL-based framework for optimizing platoon joining on highways with mixed trafic. We primarily implemented the typical value-based DQN and DDQN algorithms and the policy-based PPO algorithm for this control. Compared to the DQN and DDQN, the PPO achieves superior performance on model training and testing, although it takes a few more decision steps (about +10%) for platoon joining and longer distances $( + \ 0 . 9 \sim 1 . 2 \ \mathrm { m . } )$ to target platoons on average for the model tests. Concretely, the PPO performs excellently in all cases of trafic load testing, with about 98% of successful joinings and less than 1% for both failures and collisions. Regarding some quite degraded results of DQN and DDQN when the platoon load is set to 2,000 veh/h where a very short trafic head time is generated, we adopted strict conditions set by the rear/front TTC constraints before implementing the joining decisions that were learned by our approaches. The success rates under this safety shield for those specific cases are eventually over 92% for DQN and 96% for DDQN with guarantees of free collisions.

Despite the generality of the proposed deep RL modeling framework for CAV platoon joining and results comparison by the proposed deep RL algorithms, a limitation of this study is related to the definite discrete action space for speed adjustments and its lack of optimization. Although driving comfort was considered through the inclusion of a jerk penalty in the reward function, joining eficiency is not confirmed as the ego vehicle’s speed trajectory is not optimized explicitly. Future work will focus on enhancing the reward design by incorporating speed trajectory optimization objectives, such as energy eficiency and time-eficient platoon joining globally. Furthermore, the proposed framework will be extended by developing an eficient strategy for CAVs to safely and efectively exit platoons.

![](images/32fa21a06dc849d97d78017a30401e2456c3dbf8c7c9d7a8c5828d28039635fa.jpg)

(a) Success rate  
![](images/6a2fa3c8600d4a6efd889c4b79a22ffab0ce597b4bf25d490723f752809038ac.jpg)

(b) Failure rate  
![](images/10846a1da3d1d4042d74bd3f6577404d99e232f9f8b7690d988887263f9c7f9a.jpg)  
(c) Collision rate  
Figure 9: Model tests of algorithms with or without TTC penalty (r<sub>r</sub>) in diferent trafic loads.

## References

[1] Ahmed, H. U., Huang, Y., Lu, P., Bridgelall, R., 2022. Technology developments and impacts of connected and autonomous vehicles: An overview. Smart Cities, 5(1), pp. 382-404.

[2] Jiang, L., Xie, Y., Wen, X., Chen, D., Li, T., Evans, N. G., 2021. Dampen the stop-andgo trafic with connected and automated vehicles–a deep reinforcement learning approach. In: 2021 IEEE 7th International Conference on Models and Technologies for Intelligent Transportation Systems (MT-ITS), pp. 1-6.

[3] Sala, M., Soriguera, F., 2021. Capacity of a freeway lane with platoons of autonomous vehicles mixed with regular trafic. Transportation research part B: methodological, 147, pp. 116-131.

[4] Yang, B., Chen, J., Zhang, J., Zhou, B., Zhang, J., Ji, H., 2025. The efect of local platoon control strategy on the stability of mixed trafic flow. Transportmetrica B: Transport Dynamics, 13(1), pp. 2496828.

[5] Li, W., Ding, H., Xu, N., Song, Z., Zhang, J., 2024. A time and energy eficient merging control for platoon formation of connected and automated electric vehicles at on-ramps. Nonlinear Dynamics, 112, pp. 1–24. DOI: 10.1007/s11071-023-09238-4.

[6] Ding, J., Li, L., Peng, H., Zhang, Y., 2019. A rule-based cooperative merging strategy for connected and automated vehicles. IEEE Transactions on Intelligent Transportation Systems, 21(8), pp. 3436-3446. DOI: 10.1109/TITS.2019.2928969

[7] Liu, P., Kurt, A., Ozguner, U., 2018. Distributed model predictive control for cooperative and flexible vehicle platooning. IEEE Transactions on Control Systems Technology, 27(3), pp. 1115-1128. DOI: 10.1109/TCST.2018.2808911

[8] Huang, Z., Zhuang, W., Yin, G., Xu, L., Luo, K., 2019. Cooperative merging for multiple connected and automated vehicles at highway on-ramps via virtual platoon formation. In: Proceedings of the 2019 Chinese Control Conference (CCC), pp. 6709–6714. DOI: 10.23919/ChiCC.2019.8866378.

[9] Wang, J., Hu, C., Zhao, J., Zhang, L., Han, Y., 2024. Deep Q-Network-enabled platoon merging approach for autonomous vehicles. Transportation Research Record, 2678(7), pp. 17–31. DOI: 10.1177/03611981231203229.

[10] Ye, F., Cheng, X., Wang, P., Chan, C.-Y., 2020. Automated lane change strategy using proximal policy optimization-based deep reinforcement learning. CoRR, abs/2002.02667. Available at: https://arxiv.org/abs/2002.02667.

[11] Kolat, M., B´ecsi, T., 2024. Cooperative MARL-PPO approach for automated highway platoon merging. Electronics, 13(15). DOI: 10.3390/electronics13153102.

[12] Zhou, W., Chen, D., Yan, J., Li, Z., Yin, H., Ge, W., 2022. Multi-agent reinforcement learning for cooperative lane changing of connected and autonomous vehicles in mixed trafic. Autonomous Intelligent Systems, 2(1), 5.

[13] Zhang, J., Chang, C., Zeng, X., Li, L., 2022. Multi-agent DRL-based lane change with right-of-way collaboration awareness. IEEE Transactions on Intelligent Transportation Systems, 24(1), pp. 854–869.

[14] Wang, S., Wang, Z., Jiang, R., Zhu, F., Yan, R., Shang, Y., 2024. A multi-agent reinforcement learning-based longitudinal and lateral control of CAVs to improve trafic eficiency in a mandatory lane change scenario. Transportation Research Part C: Emerging Technologies, 158, p. 104445.

[15] Zhou, W., Chen, D., Yan, J., Li, Z., Yin, H., Ge, W., 2021. Multi-agent reinforcement learning for cooperative lane changing of connected and autonomous vehicles in mixed trafic. CoRR, abs/2111.06318. Available at: https://arxiv.org/abs/2111.06318.

[16] Shi, Y., Dong, H., He, C. R., Chen, Y., Song, Z., 2025. Mixed vehicle platoon forming: A multi-agent reinforcement learning approach, 12(11), pp. 16886-16898. IEEE Internet of Things Journal.

[17] Prathiba, S. B., Raja, G., Dev, K., Kumar, N., Guizani, M., 2021. A hybrid deep reinforcement learning for autonomous vehicles smart-platooning. IEEE Transactions on Vehicular Technology, 70(12), pp. 13340–13350.

[18] Krasowski, H., Wang, X., Althof, M., 2020. Safe reinforcement learning for autonomous lane changing using set-based prediction. IEEE 23rd international conference on Intelligent Transportation Systems, pp. 1-7.

[19] Kim, J., Lim, D., Seo, Y., So, J., Kim, H., 2023. Influence of dedicated lanes for connected and automated vehicles on highway trafic flow. IET Intelligent Transport Systems, 17(4), pp. 678-690.

[20] Zhang, X., Wu, L., Liu, H., Wang, Y., Li, H., Xu, B., 2023. High-speed ramp merging behavior decision for autonomous vehicles based on multi-agent reinforcement learning. IEEE Internet of Things Journal, 10(24), pp. 22664–22672.

[21] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M. A., 2013. Playing Atari with Deep Reinforcement Learning. CoRR, abs/1312.5602. Available at: http://arxiv.org/abs/1312.5602.

[22] Segata, M., Cigno, R. L., Hardes, T., Heinovski, J., Schettler, M., Bloessl, B., Dressler, F., 2022. Multi-technology cooperative driving: An analysis based on PLEXE. IEEE Transactions on Mobile Computing, 22(8), pp. 4792-4806.

[23] Ducrocq, R., Farhi, N., 2023. Deep reinforcement Q-learning for intelligent trafic signal control with partial detection. International journal of intelligent transportation systems research, 21(1), pp. 192-206.

[24] Dinneweth, J., Boubezoul, A., Mandiau, R., Espi´e, S., 2026. Archicool: Driver model based on the selective empathy. IEEE Transactions on Intelligent Transportation Systems, 27(3), pp. 3357-3368.

[25] Zhou, J., Yan, L., Yang, K., 2024. Enhancing system-level safety in mixed-autonomy platoon via safe reinforcement learning. IEEE Transactions on Intelligent Vehicles, pp. 1-13. DOI: 10.1109/TIV.2024.3373512

## Appendix A. Configuration

Table A.1: Configuration for the DQN and DDQN algorithms
<table><tr><td>Hyperparameter</td><td>Value/Definition</td></tr><tr><td>Model type</td><td>Fully connected online/target neural networks (MLP)</td></tr><tr><td>Hidden layers</td><td>4</td></tr><tr><td>Hidden units</td><td>[32, 64, 64, 32]</td></tr><tr><td>Activation function</td><td>LeakyReLU</td></tr><tr><td>Output layer</td><td>Linear</td></tr><tr><td>Input dimension</td><td>15</td></tr><tr><td>Output dimension</td><td>6</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Loss function</td><td>SmoothL1Loss</td></tr><tr><td>Learning rate α</td><td>0.0002</td></tr><tr><td>Discount factor γ Soft update coefficient τ</td><td>0.9</td></tr><tr><td>Soft update frequency C</td><td>0.001</td></tr><tr><td>Decay(€)</td><td>103</td></tr><tr><td>€start</td><td>Exponential decay</td></tr><tr><td>€min</td><td>1</td></tr><tr><td>€decay</td><td> $_ { 0 . 0 1 }$   $6 \times 1 0 ^ { 5 }$ </td></tr><tr><td>Mini-batch size M</td><td>64</td></tr><tr><td>Replay buffer size</td><td> $2 \times 1 0 ^ { 4 }$ </td></tr><tr><td>Training warm-up steps</td><td> $1 . 2 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Maximum training steps</td><td> $1 0 ^ { 6 }$ </td></tr><tr><td>Maximum steps per episode</td><td> $1 0 ^ { 3 }$ </td></tr><tr><td>Decision step duration</td><td>0.5s (5 simulation steps)</td></tr><tr><td></td><td></td></tr></table>

Table A.2: Configuration for the PPO algorithm
<table><tr><td>Hyperparameter</td><td>Value/Definition</td></tr><tr><td>Model type</td><td>Fully connected actor/critic neural networks (MLP)</td></tr><tr><td>Hidden layers</td><td>4</td></tr><tr><td>Hidden units</td><td>[32, 64, 64, 32]</td></tr><tr><td>Activation function</td><td>LeakyReLU</td></tr><tr><td>Output layer</td><td>Linear</td></tr><tr><td>Input dimension (actor network)</td><td>15</td></tr><tr><td>Output dimension (actor network)</td><td>6</td></tr><tr><td>Input dimension (critic network)</td><td>15</td></tr><tr><td>Output dimension (critic network)</td><td>1</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Actor learning rate αactor critic learning rate αcritic</td><td>0.0001</td></tr><tr><td>Discount factor γ</td><td>0.001</td></tr><tr><td>Clipping coefficient</td><td>0.9</td></tr><tr><td> $\epsilon _ { \mathrm { c l i p } }$  GAE coefficient λ</td><td>0.1</td></tr><tr><td>Critic loss coefficient c1</td><td>0.95</td></tr><tr><td>Entropy coefficient  $c _ { 2 }$ </td><td>0.5</td></tr><tr><td>Epochs K</td><td>0.01</td></tr><tr><td>Mini-batch size M</td><td>5 256</td></tr><tr><td>Rollout size</td><td>2048</td></tr><tr><td>Training warm-up steps</td><td> $1 . 2 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Maximum training steps</td><td> $1 0 ^ { 6 }$ </td></tr><tr><td>Maximum steps per episode</td><td> $1 0 ^ { 3 }$ </td></tr><tr><td>Decision step duration</td><td></td></tr><tr><td></td><td>0.5s (5 simulation steps)</td></tr></table>

Table A.3: Configuration for trafic scenarios
<table><tr><td>Hyperparameter</td><td>Value/Definition</td></tr><tr><td>Traffic load per lane</td><td>Varying from 500 veh/h to 2,000 veh/h, with an increment of 500 veh/h</td></tr><tr><td>CAV penetration rate</td><td></td></tr><tr><td>(on the mixed lane) Communication radius R</td><td>50%</td></tr><tr><td>Human driver imperfection</td><td>200 m 0.5</td></tr><tr><td>Rear distance search δ1</td><td>150 m</td></tr><tr><td>Front distance search δ2</td><td>5 m</td></tr><tr><td>Intervehicle distance (CACC)</td><td>5 m</td></tr><tr><td>Speed limits on lane</td><td> $v _ { m } = 1 0 0 ~ \mathrm { k m / h } ; v _ { p } = 1 2 0 ~ \mathrm { k m / h }$ </td></tr></table>