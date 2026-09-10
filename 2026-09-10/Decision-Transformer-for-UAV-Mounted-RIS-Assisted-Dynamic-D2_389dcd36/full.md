# Decision Transformer for UAV-Mounted RIS-Assisted Dynamic D2D Communications

Yaxuan Liu

Abstract—This paper studies unmanned aerial vehicle (UAV)- mouted reconfigurable intelligent surface (RIS)-assisted deviceto-device (D2D) communication with stochastic link activation. It models UAV motion and attitude, time-varying Rician angles, and angle-dependent RIS reflection. A joint optimization of UAV trajectory, attitude, and RIS phases is formulated to maximize average sum rate under mobility, energy, and hardware constraints. The problem is addressed using deep reinforcement learning and a Decision Transformer trained on expert trajectories from multiple scenarios. Results demonstrate effective crossscenario generalization, with zero-shot transfer outperforming direct DRL transfer and online fine-tuning achieving competitive performance with fewer interactions.

Index Terms—unmanned aerial vehicle, reconfigurable intelligent surface, deep reinforcement learning, joint optimization, decision transformer, generalization performance

## I. INTRODUCTION

Reconfigurable intelligent surface (RIS) has emerged as a promising low-cost and energy-efficient technology for enhancing wireless signal coverage and communication quality [1]. Conventionally, RIS devices are fixedly deployed on walls or stationary infrastructures, which severely limits their flexibility and adaptability in dynamic communication scenarios such as vehicular networks and industrial wireless systems. Such fixed deployment fails to cope with time-varying channel conditions and frequently occurring link blockage, resulting in degraded beamforming performance and unreliable transmission. To address these limitations, the integration of RIS with unmanned aerial vehicles (UAVs) provides an effective solution. Benefiting from the high mobility and flexible maneuverability of UAVs, the UAV-mounted RIS system can dynamically adjust its spatial position and attitude in real time. Different from static RIS, this integrated framework can actively avoids wireless link blockage to maintain stable lineof-sight (LoS) propagation paths.

In [4], [5], the authors jointly optimize UAV trajectory and RIS phase shifts to enhance the sum rate, physicallayer security and energy efficiency. However, These studies either assumed that the RIS is deployed on a stationary planar surface, or considered UAV-mounted RIS systems while ignoring the three-dimensional rotational attitudes of UAV. Several other works incorporate the three-dimensional rotational angles of UAVs as random jitter disturbances, without treating RIS orientation as an adjustable optimization variable and still constraining the RIS to a fixed plane. Several studies verified from an electromagnetic perspective that the reflection coefficients of RIS elements are dependent on incident angles [2], [3]. Though [6], [7] integrated this angle-variant reflection property into UAV-mounted RIS modeling, [7] focused on the statistic scenarios, while [6] relied on an oversimplified uniform linear array (ULA) steering model and neglected the intricate coupling among UAV pose, time-varying Rician channels, and element-wise local incident angles, which is critical for indoor propagation environments.

Notably, since the optimization problem in UAV assisted communications are non-convex and coupling, deep reinforcement learning (DRL) algorithms are always adopted to resolve theses complicated problems. For example, deep deterministic policy gradient (DDPG) algorithm was used in [4] to jointly optimize the UAV trajectories and RIS phase shift. Authors in [8] employed soft actor-critic (SAC) to jointly optimize UAV trajectory and resource allocation for maximizing computation bits under a fairness constraint. However, conventional DRL policies often lack robustness to changes in communication environments, with performance degrading in unseen scenarios and adaptation requiring costly online interactions [9]. The Decision Transformer (DT) addresses this issue by framing reinforcement learning as conditional sequence modeling [10]. Based on the Transformer architecture, it predicts actions from returns-to-go, states, and interaction history, enabling efficient policy learning from offline trajectories without value estimation or policy-gradient updates.

Motivated by these, this work studies a dynamic indoor communication system assisted by a UAV-mounted RIS configured as a uniform planar array (UPA), in which communication links between users are stochastically established over time. The UAV trajectory, three-dimensional attitude, and RIS phase shifts are jointly optimized while accounting for the time-varying arrival and departure angles of the Rician LoS components and the incident-angle-dependent responses of the RIS elements. A Decision Transformer is pre-trained offline on high-quality DRL trajectories across multiple scenarios, enabling zero-shot control and efficient online adaptation in unseen deployments with fewer interactions.

## II. SYSTEM MODEL

We focus on the dense industrial manufacturing scenario where exist K single-antenna device-to-device (D2D) communication devices, denoted by the set $\begin{array} { r l r } { { \cal D } } & { { } = } & { \left\{ { \cal D } U _ { 1 } , { \cal D } U _ { 2 } , . . . , { \cal D } U _ { K } \right\} } \end{array}$ . Pairwise communication is established among these D2D devices by reusing spectrum resources on licensed frequency bands. Within each communication time interval, at most $\lfloor K / 2 \rfloor$ D2D pairs are randomly activated with random user pairing. A

![](images/fc3c02177e5d6a66880d806e298b9f953cd98c1b3eb659d112b7cf8d941fb391.jpg)  
Fig. 1: UAV-Mounted RIS-Assisted Dynamic D2D Communication System.

UAV-mounted RIS is deployed to assist communications, where both desired signals and interference propagate via RIS reflection.

We consider that all D2D devices possess different altitudes, and the UAV supports flexible altitude adjustment rather than hovering at a constant height. For this practical scenario, we adopt an incident-angle-dependent RIS reflection coefficient model. To accurately characterize the spatial incident angles, the RIS is equipped with a UPA. In contrast to conventional RIS models with ideal unitary reflection coefficients, the vertical height differences among D2D devices lead to dynamic variations in the incident angle impinging on the RIS, which further alters the resultant reflection coefficients. Accordingly, it is essential to dynamically adjust the three-dimensional attitude angles of the RIS-mounted UAV to deliberately regulate the RIS incident angle. Based on this physical mechanism, we jointly optimize the UAV flight trajectory, three-dimensional attitude angles, and RIS phase shifts to enhance the overall communication performance.

## A. Channel Model

Mutual interference emerges when multiple D2D pairs transmit concurrently within the same time slot. Severe ground blockages are assumed in this work, such that the direct paths between D2D users can be neglected. Since the UAV operates at altitude, LoS links are maintained between ground users and the RIS. Accordingly, the channel between D2D user $D U _ { k }$ and the RIS is modeled by Rician fading, which yields

$$
\mathbf { h } _ { k , R } = \sqrt { \frac { \kappa } { \kappa + 1 } } \bar { \mathbf { h } } _ { k , R } + \sqrt { \frac { 1 } { \kappa + 1 } } \tilde { \mathbf { h } } _ { k , R } ,\tag{1}
$$

where κ denotes the Rician factor, $\bar { \mathbf { h } } _ { k , R } = \sqrt { d _ { k , R } ^ { - \alpha _ { L } } } \bar { \mathbf { g } } _ { k , R }$ stands for the LoS component where $d _ { k , R }$ is the distance between $D U _ { k }$ and RIS, $\alpha _ { L }$ represents the path-loss exponent for the LoS path, and $\bar { g } _ { k , R }$ denotes the receiving array response of the LoS component. Since UPA is considered for RIS, we denote the numbers of elements along the x-axis and y-axis as $N _ { x }$ and $N _ { v }$ , respectively, yielding a total of $N = N _ { x } \times N _ { y }$ units. The angle of arrival (AoA) at the RIS is $\left( \phi _ { k , R } , \vartheta _ { k , R } \right)$ where $\phi _ { k , R }$ and $\vartheta _ { k , R }$ represent the azimuth and elevation angles of the incident electromagnetic wave with respect to the RIS local coordinate system. Therefore, the receiving array response at the UAV-mounted RIS is expressed by $\mathbf { \bar { g } } _ { k , R } = \mathbf { a } _ { y } \left( \phi _ { k , R } , \vartheta _ { k , R } \right) \otimes \mathbf { a } _ { x } \left( \phi _ { k , R } , \vartheta _ { k , R } \right)$ where

$$
\mathbf { a } _ { x } \left( \phi _ { k , R } , \vartheta _ { k , R } \right) = \Big [ 1 , \ldots , e ^ { - j \frac { 2 \pi } { \lambda } d \left( N _ { x } - 1 \right) u _ { k , R } } \Big ] ^ { T } ,
$$

and

(2)

$$
\mathbf { a } _ { y } \left( \phi _ { k , R } , \vartheta _ { k , R } \right) = \Big [ 1 , \ldots , e ^ { - j \frac { 2 \pi } { \lambda } d ( N _ { y } - 1 ) v _ { k , R } } \Big ] ^ { T } ,\tag{3}
$$

where $u _ { k , R } = \sin \vartheta _ { k , R }$ cos $\phi _ { k , R }$ and $v _ { k , R } = \sin \vartheta _ { k , R }$ sin $\phi _ { k , R }$ represent the spatial frequencies along the local x-axis and y-axis, respectively, d denotes the inter-element spacing, λ denotes the carrier wavelength and $\otimes$ is the Kronecker product operation. Similarly, the transmit array response, defined by its angle of departure (AoD), can be calculated and denoted as $\tilde { \bf g } _ { R , k }$ . In addition, $\tilde { \mathbf { h } } _ { k , R } = \sqrt { d _ { k , R } ^ { - \alpha _ { L } } } \tilde { \mathbf { g } } _ { k , R }$ denotes the non-lineof-sight (NLoS) component where $\widetilde { \mathbf { g } } _ { k , R }$ follows a circularly symmetric complex Gaussian distribution with zero mean and unit variance. Similarly, the channel from the RIS to $D U _ { i }$ is denoted as $\mathbf { h } _ { R , i }$ , which possesses the identical mathematical structure as (1).

## B. Angle Dependent RIS Reflection Model

Specially, in this paper we adopts the practical reflection coefficient modeling framework for RIS proposed in [11]. We first define Θ as the phase-shift matrix of the RIS, which is formulated as

$$
\Theta = \mathrm { d i a g } \left\{ ( r _ { 1 } ^ { p r e } , . . . , r _ { N } ^ { p r e } ) \right\} ,\tag{4}
$$

where $r _ { n } ^ { p r e } = \left| r _ { n } ^ { p r e } \right| e ^ { j \angle r _ { n } ^ { p r e } } \left( n = 1 , . . . , N \right)$ denotes the preset complex reflection coefficient. According to [11], the practical reflection coefficient depending on the incident angle is obtained as

$$
r _ { n } ^ { p r a } \left( \theta _ { i n } \right) = \frac { - \eta \lambda _ { 1 } } { 2 \cos { \theta _ { i n } } + \eta \lambda _ { 1 } } + \frac { \lambda _ { 2 } \cos { \theta _ { i n } } } { \lambda _ { 2 } \cos { \theta _ { i n } } + 2 \eta } ,\tag{5}
$$

where $\theta _ { i n }$ is the electromagnetic wave incident angle, and

$$
\lambda _ { 1 } = \frac { 2 } { \eta } \frac { 1 - r _ { n } ^ { p r e } } { 1 + r _ { n } ^ { p r e } } , \lambda _ { 2 } = \frac { 2 } { \eta } \frac { 1 + r _ { n } ^ { p r e } } { 1 - r _ { n } ^ { p r e } } ,\tag{6}
$$

with the free-space wave impedance $\eta = 1 2 0 \pi \Omega$ . We denote $\Theta ^ { p r a } =$ diag $\bar { \{ ( r _ { 1 } ^ { p r a } , . . . , r _ { N } ^ { p r a } ) \} }$ as the practical phase matrix of RIS. This model characterizes how the preset reflection amplitude and phase of RIS elements vary with the incident angle of electromagnetic waves, and outputs the actual reflection coefficient under specific propagation scenarios. The calculation method of $\theta _ { i n }$ will be given in Remark 1.

## C. Signal Model

Based on the channel model and the RIS reflection model, the received signal at user $D U _ { i }$ can be formulated as

$$
\begin{array} { l } { { { y } _ { i } } = \sqrt { P } \left( { \bf { h } } _ { k , R } ^ { H } { \Theta } ^ { p r a } { \bf { h } } _ { R , i } \right) { x } _ { k } + } \\ { \displaystyle \sum _ { j = 1 , j \neq k } ^ { K } \beta _ { j } \sqrt { P } \left( { \bf { h } } _ { j , R } ^ { H } { \Theta } ^ { p r a } h _ { R , i } \right) { x } _ { j } + n _ { i } } \end{array}\tag{7}
$$

where P denotes the transmit power of each D2D user, and we assume identical transmit power for all users in this paper, $\beta _ { j } \in \{ 0 , 1 \}$ is the control coefficient, which indicates whether $D U _ { j }$ performs transmission in the current time slot. $x _ { j }$ and $x _ { k }$ represent the transmitted signals of $D U _ { j }$ and $D U _ { k }$ respectively; $n _ { i }$ denotes the additive white Gaussian noise (AWGN) at receiver $D U _ { i }$ . Accordingly, the achievable rate at $D U _ { i }$ can be formulated as

$$
R _ { i } = \log _ { 2 } \left( 1 + \frac { P \left( \mathbf { h } _ { k , R } ^ { H } \mathbf { \Theta } \Theta \mathbf { h } _ { R , i } \right) } { \sum _ { j = 1 , j \neq k } ^ { K } \beta _ { j } P \left( \mathbf { h } _ { j , R } ^ { H } \mathbf { \Theta } \Theta \mathbf { h } _ { R , i } \right) + \sigma ^ { 2 } } \right)\tag{8}
$$

where $\sigma ^ { 2 }$ is the noise power.

## D. UAV Motion Model

Let the position coordinate of the UAV at time t be ${ \bf q } _ { U } \left( t \right) = \left( x _ { U } \left( t \right) , y _ { U } \left( t \right) , z _ { U } \left( t \right) \right)$ , and its velocity and acceleration vectors along the $X , ~ Y , ~ Z$ axes be denoted as $\mathbf { v } _ { U } = \left[ v _ { x } , v _ { y } , v _ { z } \right]$ and $\mathbf { a } _ { U } = \left[ a _ { x } , a _ { y } , a _ { z } \right]$ , respectively. After a time slot interval $\Delta t .$ , the UAV position at time $t + 1$ is updated via the uniformly accelerated motion model that

$$
{ \bf q } _ { U } \left( t + 1 \right) = { \bf q } _ { U } \left( t \right) + { \bf v } _ { U } \left( t \right) \Delta t + 0 . 5 { \bf a } _ { U } \left( t \right) \Delta t ^ { 2 } ,\tag{9}
$$

where $\Delta t$ refers to the duration of a single time slot.

Since the electromagnetic reflection characteristics of RIS are highly dependent on the incident angle of electromagnetic waves, three attitude angles, including yaw $\zeta _ { y } ,$ pitch $\zeta _ { p }$ and roll $\zeta _ { r } ,$ , are modeled for the aerial RIS mounted on the UAV. Considering the indoor scenario adopted in this work, the three-dimensional rotation angles of the UAV are precisely controlled without external disturbances such as wind-induced jitter. The quadrotor UAV is considered in this paper, whose translational moving direction is decoupled from the heading direction of the nose. Therefore, a 3D rotation matrix is constructed from the above three attitude angles as

$$
\begin{array} { r l r } {  { { \bf R } = { \bf R } _ { y a w } { \bf R } _ { p i t c h } { \bf R } _ { r o l l } } } \\ & { } & { = [ \begin{array} { c c c } { \cos \zeta _ { y } } & { - \sin \zeta _ { y } } & { 0 } \\ { \sin \zeta _ { y } } & { \cos \zeta _ { y } } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} ] [ \begin{array} { c c c } { \cos \zeta _ { p } } & { 0 } & { \sin \zeta _ { p } } \\ { 0 } & { 1 } & { 0 } \\ { - \sin \zeta _ { p } } & { 0 } & { \cos \zeta _ { p } } \end{array} ] } \\ & { } & { \times [ \begin{array} { c c c } { 1 } & { 0 } & { 0 } \\ { 0 } & { \cos \zeta _ { r } } & { - \sin \zeta _ { r } } \\ { 0 } & { \sin \zeta _ { r } } & { \cos \zeta _ { r } } \end{array} ] . } \end{array}\tag{10}
$$

Remark 1. For an RIS arranged parallel to the xOy plane, the corresponding normal vector, x-axis directional vector and y-axis directional vector are defined as $\vec { \bf n } _ { R I S } = \left[ 0 , 0 , 1 \right] ^ { T }$ $\mathbf { \check { e } } _ { x } = \left[ 1 , 0 , 0 \right] ^ { T }$ and ${ \bf e } _ { y } = { [ 0 , 1 , 0 ] } ^ { T }$ . After the UAV undergoes three-axis attitude deflections characterized by $\zeta _ { y } , \zeta _ { p } , \zeta _ { r }$ , the update process of RIS vectors is obtained through coordinate rotation: $\left( \mathbf { n } _ { R I S } , \mathbf { e } _ { x } , \mathbf { e } _ { y } \right) _ { t + 1 } = \mathbf { R } _ { t } \left( \mathbf { n } _ { R I S } , \mathbf { e } _ { x } , \mathbf { e } _ { y } \right)$

Combining the established UAV motion and attitude models, the precise incident angle of electromagnetic waves impinging on RIS from transmitter $D U _ { k }$ is calculated as

$$
\theta _ { k , i n } = \operatorname { a r c c o s } \left( \frac { \left. \mathbf { q } _ { U } - \mathbf { q } _ { k } , \vec { \mathbf { n } } _ { R I S } \right. } { \left| \mathbf { q } _ { U } - \mathbf { q } _ { k } \right| \left| \vec { \mathbf { n } } _ { R I S } \right| } \right) ,\tag{11}
$$

where $\mathbf { q } _ { k }$ denotes the spatial coordinate of transmitter $D U _ { k }$ Similarly, the spatial frequencies at RIS is calculated as

$$
u _ { k , R } = \frac { \left. \mathbf { q } _ { U } - \mathbf { q } _ { k } , \mathbf { e } _ { x } \right. } { \left| \mathbf { q } _ { U } - \mathbf { q } _ { k } \right| \left| \mathbf { e } _ { x } \right| } , v _ { k , R } = \frac { \left. \mathbf { q } _ { U } - \mathbf { q } _ { k } , \mathbf { e } _ { y } \right. } { \left| \mathbf { q } _ { U } - \mathbf { q } _ { k } \right| \left| \mathbf { e } _ { y } \right| } .\tag{12}
$$

It is worth noting that the calculated incident angle accounts for the three-dimensional rotational attitudes of the UAV, representing the true incidence angle of transmitted signals impinging on the RIS plane, and corresponds to AOA and AOD. This angular parameter is substituted into (1) and (5) to acquire the updated Rician channel as well as the practical reflection coefficient of RIS.

## III. PROBLEM FORMULATION

In this work, we investigate UAV-mounted RIS assisted communications for multiple D2D pairs, where the practical RIS reflection coefficient varies with wave incident angles. Accordingly, we aim to maximize the sum achievable rate of all D2D pairs via jointly optimizing UAV trajectories, UAV attitudes, and preset RIS reflection coefficient. The optimization problem is formulated as follows:

$$
\operatorname* { m a x } _ { \{ \mathbf { Q } ( t ) , \boldsymbol { \Theta } ( t ) \} } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { k = 1 } ^ { K } R _ { k }
$$

$$
{ \mathbf { \mathcal { s } } } . t . { \mathbf { q } } _ { U } \left( t \right) \in \Omega _ { U }\tag{13}
$$

(13a)

$$
\left| \mathbf { v } _ { U } \left( t \right) \right| \leqslant v _ { \operatorname* { m a x } }\tag{13b}
$$

$$
\left. \mathbf { a } _ { U } \left( t \right) \right. \leqslant a _ { \mathrm { m a x } }
$$

$$
| r _ { n } | \leqslant 1 , \angle r _ { n } \in [ 0 , 2 \pi )\tag{13c}
$$

$$
\zeta _ { y } \left( t \right) , \zeta _ { p } \left( t \right) , \zeta _ { r } \left( t \right) \in \left[ 0 , 2 \pi \right)\tag{13d}
$$

$$
\beta _ { k } \left( t \right) \in \left\{ 0 , 1 \right\}\tag{13e}
$$

(13f)

$$
\sum _ { t = 1 } ^ { T } P _ { U } \left( t \right) \leqslant E _ { U } ^ { \operatorname* { m a x } }\tag{13g}
$$

where $T$ is the number of total time slots, $\begin{array} { r l } { \mathbf { Q } \left( t \right) } & { { } = } \end{array}$ $\left\{ \mathbf { q } _ { U } \left( t \right) , \zeta _ { y } \left( t \right) , \zeta _ { p } \left( t \right) , \zeta _ { r } \left( t \right) \right\}$ contains the UAV trajectories and UAV attitudes. (13a) constrains the UAV position, where $\Omega _ { U }$ denotes the feasible flight region. (13b) and (13c) constrain the resultant velocity and acceleration, respectively. The amplitudes and phase shifts limitation is presented in (13d). (13e) describes the full rotational range for three-axis UAV attitude adjustment at each time slot. (13f) is the binary indicator constrain for the k-th D2D pair. $\beta _ { k } \left( t \right) = 1$ means the k-th D2D link is served at time slot t, while $\beta _ { k } \left( t \right) = 0$ means this link remains inactive in the current time slot. Constraint (15d) limits the UAV’s total propulsion energy consumption to $E _ { U } ^ { \mathrm { m a x } }$

## IV. DECISION TRANSFORMER OPTIMIZATION FRAMEWORK

To address the formulated joint optimization problem and improve the adaptability of the learned policy to different UAV-RIS deployment scenarios, we develop a generalizable Decision Transformer (DT) framework. The overall framework consists of four main components: MDP formulation, DRLbased expert algorithm selection, offline DT pre-training, and online fine-tuning for unseen scenarios.

## A. MDP Formulation

We first model the system as a MDP, which consists of a state space $s ,$ , an action space $\mathbfcal { A } ,$ a state transition probability ${ \mathcal { P } } ,$ , and a reward function R, and denoted by the tuple $\langle s , \mathcal { A } , \mathcal { P } , \mathcal { R } \rangle$ , which are defined as follows.

![](images/81cb15cce478adc99f5b03c9a47a1ce47b75f2ef6e8d83baf478cd7250c34ed1.jpg)  
Fig. 2: Decision Transformer Framework with Offline Pre-Training and Online Fine-Tuning

1) State: The state space at timestep t is defined by $S _ { t } = \left\{ \mathbf { q } _ { U } \left( t \right) , \mathbf { v } _ { U } \left( t \right) , \zeta _ { y } \left( t \right) , \zeta _ { p } \left( t \right) , \zeta _ { r } \left( t \right) , \mathbf { H } \left( t \right) , \mathbf { P } \left( t \right) \right\}$ where ${ \bf H } \left( t \right) \in \mathbb { R } ^ { N \times K }$ denotes the Rician fading matrix and $\mathbf { P } \left( t \right) \in \mathbb { R } ^ { K \times K }$ represents the D2D pair state in time t.

2) Action: The action space is defined as: $\begin{array} { r l r } { \mathcal { A } _ { t } } & { { } = } & { \{ \mathbf { a } _ { U } ( t ) , \Theta ( t ) , \Delta \zeta _ { y } \left( t \right) , \Delta \zeta _ { p } \left( t \right) , \Delta \zeta _ { r } \left( t \right) \} } \end{array}$ , where $\Delta \zeta _ { y } ( t ) , \Delta \zeta _ { p } ( t )$ , and $\Delta \zeta _ { r } ( t )$ denote the incremental changes of 3D rotation angles.

3) Transition probability: After taking action $\boldsymbol { A } _ { t }$ in the current state $S _ { t }$ , the system transits to the next state $S _ { t + 1 }$ . The transition probability is denoted as $\mathcal { P } \left( S _ { t + 1 } \mid S _ { t } , \mathcal { A } _ { t } \right)$ , which represents the probability of moving from $S _ { t }$ to $\boldsymbol { S } _ { t + 1 }$ after executing $A _ { t } .$

4) Reward: At each time slot t, the reward is defined as $\begin{array} { r } { r ( t ) = \sum _ { k = 1 } ^ { K } R _ { k } ( t ) } \end{array}$ . The reward return is defined as $\mathcal { R } _ { t } ~ =$ $\scriptstyle \sum _ { \tau = t } ^ { T ^ { \prime } } \gamma ^ { \tau - t } r ( \tau )$ , where γ is the discount factor. The objective of the MDP is to learn an optimal policy $\pi ^ { * }$ that maximizes the expected cumulative reward $\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } E _ { \pi } \left[ \mathcal { R } _ { 1 } \right]$

## B. DRL-Based Expert Algorithm Selection

To construct a high-quality offline dataset, several representative DRL algorithms for continuous control are first trained and evaluated under the same simulation settings, including proximal policy optimization (PPO) [12], DDPG [13], SAC [14], and twin delayed deep deterministic policy gradient (TD3) [15]. Their convergence behavior, average sum-rate performance, and policy stability are compared, and the bestperforming algorithm is selected as the expert. The detailed comparison is presented in Section V. This empirical selection procedure avoids assuming in advance that a particular DRL algorithm is optimal for the considered environment.

## C. Offline Pre-Training

First, the selected DRL algorithm is independently trained in M different UAV-RIS scenarios to obtain high-quality expert policies. The i-th trajectory generated by DRL policy m can be represented as

$$
\tau _ { m } ^ { i } = \left\{ S _ { m , 1 } ^ { i } , \mathcal { A } _ { m , 1 } ^ { i } , \hat { \mathcal { R } } _ { m , 1 } ^ { i } , \ldots , S _ { m , T } ^ { i } , \mathcal { A } _ { m , T } ^ { i } , \hat { \mathcal { R } } _ { m , T } ^ { i } \right\} ,\tag{14}
$$

where $\hat { \mathcal { R } } _ { m , t }$ <sub>t</sub> denotes the return-to-go (RTG) from time slot t, which is calculated as

$$
\hat { \mathcal { R } } _ { m , t } = \sum _ { t ^ { \prime } = t } ^ { T } r _ { m } ( t ^ { \prime } ) .\tag{15}
$$

Multiple trajectories can be generated by each of the M expert policies and added to the dataset D. During offline training, DT takes the historical states, actions and RTGs as input and predicts the action at the current time slot. The model parameters are optimized by minimizing the mean squared error between the predicted action $\mathcal { A } _ { m , t } ^ { i , * }$ and the expert action $\mathcal { A } _ { m , t } ^ { i } , \mathrm { i . e . }$ •,

$$
L _ { \mathrm { O F F } } = \mathbb { E } _ { \tau _ { m } ^ { i } \sim \mathcal { D } } \left[ \left\| \boldsymbol { A } _ { m , t } ^ { i } - \boldsymbol { A } _ { m , t } ^ { i , * } \right\| _ { 2 } ^ { 2 } \right] .\tag{16}
$$

Through multi-scenario offline pre-training, DT learns the relationship between system states, target returns, and highquality control actions, thereby obtaining an initialized policy with cross-scenario decision-making capability.

## D. Online Fine-Tuning

When the UAV-RIS system encounters a new scenario $M { + 1 }$ that is not contained in the offline training dataset, the pretrained DT is first directly deployed to generate a zero-shot control policy. Given the initial target return $\hat { \mathcal { R } } _ { M + 1 , 1 }$ and the observed state $\mathcal { S } _ { M + 1 , 1 }$ , DT predicts the corresponding action $\hat { \mathcal { A } } _ { M + 1 , 1 } .$ . After executing the action, the RTG is updated according to $\hat { \mathcal { R } } _ { M + 1 , t + 1 } = \hat { \mathcal { R } } _ { M + 1 , t } - r ( t )$ , and the updated RTG, system state, and historical actions are subsequently used to generate the next control action. Notably, to encourage exploration in the new scenario, Gaussian noise with standard deviation σ is added to the action predicted by DT. Thus, the action executed during online interaction is given by $\mathcal { A } _ { M + 1 , t } = \hat { \mathcal { A } } _ { M + 1 , t } + \epsilon _ { M + 1 , t }$ , where $\epsilon _ { M + 1 , t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbf { I } )$

After completing multiple episodes, the newly collected samples are organized into a set of candidate trajectories, where the i-th trajectory of new scenario is represented as

$$
\begin{array} { c } { \tau _ { M + 1 } ^ { i } = \left\{ S _ { M + 1 , 1 } ^ { i } , \mathcal { A } _ { M + 1 , 1 } ^ { i } , \hat { \mathcal { R } } _ { M + 1 , 1 } ^ { i } , \ldots , \right. } \\ { \left. S _ { M + 1 , T } ^ { i } , \mathcal { A } _ { M + 1 , T } ^ { i } , \hat { \mathcal { R } } _ { M + 1 , T } ^ { i } \right\} . } \end{array}\tag{17}
$$

Rather than using all collected trajectories for fine-tuning, the candidate trajectories are evaluated according to their cumulative returns. Only those with the top-K returns are selected to construct an elite trajectory set

$$
\mathcal { D } _ { M + 1 } ^ { \mathrm { e l i t e } } = \mathrm { T o p K } _ { \tau _ { M + 1 } ^ { i } } \left( \sum _ { t = 1 } ^ { T _ { i } } r _ { i } ( t ) \right) .\tag{18}
$$

The selected trajectories serve as high-quality references for fine-tuning the pre-trained DT, whereas trajectories with relatively poor performance are discarded. Specifically, the model

Algorithm 1 DT Offline Pre-Training and Online Fine-Tuning   
Require: Offline scenarios M, online iterations H, candidate   
trajectories N, elite trajectories K, exploration standard   
deviation σ   
Ensure: Fine-tuned DT policy π<sub>θ</sub>   
Stage 1: Offline pre-training   
1: Initialize offline dataset $\mathcal { D }  \emptyset$   
2: for $m = 1$ to M do   
3: Train DRL expert policy $\pi _ { m } ^ { \mathrm { D R L } }$   
4: Generate multiple trajectories using $\pi _ { m } ^ { \mathrm { D R L } }$   
5: Compute their RTGs and add them to $\mathcal { D }$   
6: end for   
7: Pre-train DT on D using the loss in (16)   
Stage 2: Online fine-tuning   
8: for $h = 1$ to H do   
9: Initialize candidate set $\mathcal { D } _ { M + 1 } ^ { \mathrm { c a n d } }  \mathcal { O }$   
10: for $i = 1$ to N do   
11: Generate $\tau _ { M + 1 } ^ { i }$ using $\mathcal { A } _ { i , t } = \hat { \mathcal { A } } _ { i , t } + \epsilon _ { i , t } , \epsilon _ { i , t } \sim$   
${ \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$   
12: Add $\tau _ { M + 1 } ^ { i }$ to D<sup>cand</sup>   
13: end for   
14: Select the top-K trajectories to construct $\mathcal { D } _ { M + 1 } ^ { \mathrm { e l i t e } }$   
15: Fine-tune DT on $\mathcal { D } _ { M + 1 } ^ { \mathrm { e l i t e } }$ using the loss in (19)   
16: end for   
17: return $\pi _ { \theta }$

is updated by minimizing the action prediction error over the elite trajectory set:

$$
L _ { \mathrm { F T } } = \mathbb { E } _ { \tau _ { M + 1 } ^ { i } \sim \mathcal { D } _ { M + 1 } ^ { \mathrm { e l i t e } } } \left[ \left\| \boldsymbol { \mathcal { A } } _ { M + 1 , t } ^ { i } - \hat { \boldsymbol { \mathcal { A } } } _ { M + 1 , t } ^ { i } \right\| _ { 2 } ^ { 2 } \right] ,\tag{19}
$$

where $\mathcal { A } _ { i , t }$ is the action recorded in a selected trajectory and $\hat { \mathcal { A } } _ { i , t }$ is the action predicted by DT. Through repeated trajectory collection, quality-based selection, and fine-tuning, the pretrained model gradually adapts to the new UAV-RIS scenario while avoiding performance degradation caused by low-quality online trajectories and retaining the knowledge learned from the offline dataset.

The overall DT optimization framework based on DRL expert trajectories is summarized in Algorithm 1.

## V. NUMERICAL RESULTS

In the simulations, we consider a UAV-mounted RISassisted communication system deployed within a threedimensional area of $4 0 \times 4 0 \times 8 \mathrm { m } ^ { 3 }$ . The system consists of four D2D users located at [15, 5, 1.0], [20, 8, 1.5], [30, 20, 0.1], and [15, 35, 0.6], respectively, where all coordinates are measured in meters. RIS is configured as a $4 \times 4$ uniform planar array comprising 16 reflecting elements. The transmit power of each user is set to 1W. The Rician κ-factor and path-loss exponent set to $\kappa = 8$ and $\alpha \ = \ 2 . 4 .$ , respectively. Unless otherwise specified, these parameter settings are adopted throughout the subsequent simulations.

Fig. 3 compares the training performance of four representative DRL algorithms, including DDPG, PPO, SAC, and TD3. In this figure, the initial position of UAV-RIS is set to [15, 20, 4.5]. We can observe that DDPG improves rapidly after an initial exploration stage, and gradually stabilizes after approximately 25,000 episodes, ultimately achieving an average sum rate of approximately 29.5 bps/Hz. This value is significantly higher than those obtained by PPO, TD3, and SAC, which converge to approximately 25.0, 20.5, and 17.0 bps/Hz, respectively.

![](images/eddc59c5dfc4577ab2af2d7788e5cf073927dfd85f9845b14457465834525e49.jpg)  
Fig. 3: Convergence performance comparison of different DRL algorithms in the UAV mounted RIS-assisted communication environment.

The action space of the proposed environment consists of three-dimensional UAV motion control, UAV attitude adjustment, and the phase control of multiple RIS elements. Therefore, the resulting optimization problem is characterized by a continuous, high-dimensional, and strongly coupled action space. DDPG employs a deterministic policy gradient and can directly generate continuous control variables, thereby avoiding the dimensional expansion and loss of control accuracy caused by discretizing the action space. Moreover, its experience replay mechanism allows historical interaction data to be reused, improving sample efficiency, while the target networks help mitigate severe fluctuations during valuefunction estimation. These characteristics make DDPG well suited to the continuous joint optimization problem considered in this study.

Based on the above comparison, the DDPG algorithm is selected as the expert policy for DT. Specifically, 8 training scenarios are considered, each characterized by a different initial horizontal position of the UAV-mounted RIS. The initial altitude is fixed at $z ~ = ~ 4 . 5$ m, while the corresponding [x, y] coordinates are set to [10, 10], [10, 30], [15, 20], [20, 20], [20, 30], [25, 20], [30, 10], and [30, 30], respectively. For each scenario, the converged DDPG expert policy is employed to generate 500 trajectories without additional exploration noise. These trajectories are subsequently aggregated to form the offline training dataset for DT. During the fine-tuning stage, 80 trajectories are collected through online interactions with the environment, from which the top 50 trajectories are selected for policy fine-tuning.

Fig. 4 compares the evaluation performance of different policies in the unseen scenario with the initial UAV-mounted RIS position set to [15, 15, 4.5] in Fig. 4a and [25, 25, 4.5] in Fig. 4b. This scenario is excluded from the offline dataset used to train DT and is therefore employed to evaluate policy generalization. The horizontal axis represents the evaluation episodes rather than the training process.

![](images/143e1e3673013e741609f4192b16ab9467b8b2be9ba98a698ec8a3436057adf9.jpg)  
(a)

![](images/57bf59515d874afb938f8c8549660ff9cfe10da14d759ca8a86c4081a95905d4.jpg)  
(b)  
Fig. 4: Evaluation results of the DDPG expert, fine-tuned DT, zero-shot DT, and transferred DDPG policies in the unseen scenario. (a) Initial UAV-mounted RIS position at [15,15,4.5]. (b) Initial UAV-mounted RIS position at [25,25,4.5].

As shown in Fig. 4, the fine-tuned DT achieves nearly the same performance as the scenario-specific DDPG expert. The zero-shot DT also obtains a high average sum rate and consistently outperforms the DDPG policies transferred from neighboring scenarios. These results demonstrate that multiscenario offline training enables DT to generalize effectively to an unseen UAV–RIS initial position, while fine-tuning further improves its performance to the expert level. In addition, we can also observe that the zero-shot DT and the transferred DDPG exhibit larger fluctuations because they have not been optimized directly for the target scene and are therefore more sensitive to distribution shifts in channel states, UAV trajectories, and D2D pairings.

## VI. CONCLUSIONS

This paper studied a UAV-mounted RIS-assisted D2D communication system with stochastic link activation, considering UAV motion, three-dimensional attitude, time-varying Rician LoS angles, and incident-angle-dependent RIS responses. A joint optimization problem was formulated to maximize the average sum rate through UAV trajectory, attitude, and RIS phase control. Among several DRL methods, DDPG generated expert trajectories for multiple scenarios. A Decision Transformer was pre-trained on this offline dataset and adapted to unseen scenarios through zero-shot deployment and online fine-tuning. Results showed that zero-shot DT outperformed direct DDPG transfer, while fine-tuned DT approached scenario-specific DDPG performance with fewer interactions, demonstrating the potential of expert-data-driven sequence modeling for generalizable and efficient UAV-RIS control.

## REFERENCES

[1] M. Ahmed et al., “A comprehensive survey of artificial intelligence advances in reconfigurable intelligent surfaces-assisted wireless networks,” Engineering Applications of Artificial Intelligence, vol. 176, Art. no. 114762, Jul. 2026.

[2] B. Xu, T. Zhou, F. Gao, T. Xu, and H. Hu, “RIS-aided MIMO communications: Angle-dependent amplitude-phase response model and capacity analysis,” IEEE Wireless Commun. Lett., vol. 14, no. 2, pp. 290–294, Feb. 2025.

[3] Y. Chen, Y. Guo, and H. Zhang, “Angle-sensitive effect of reconfigurable intelligent surface,” IEEE Trans. Wireless Commun., vol. 24, no. 11, pp. 9556–9568, Nov. 2025.

[4] H. Zhao, W. Sun, Y. Ni, W. Xia, G. Gui, and C. Zhu, “Deep deterministic policy gradient-based rate maximization for RIS-UAV-assisted vehicular communication networks,” IEEE Trans. Intell. Transp. Syst., vol. 25, no. 11, pp. 15732–15744, Nov. 2024.

[5] W. Chen, Y. Zou, J. Zhu, and L. Zhai, “Joint trajectory design and phase shift optimization for multi-RIS-assisted UAV relay network using deep reinforcement learning,” IEEE Internet Things J., vol. 12, no. 8, pp. 9759–9774, Apr. 2025.

[6] M. M. Salim, K. M. Rabie, and A. H. Muqaibel, “Robust energy-efficient DRL-based optimization in UAV-mounted RIS systems with jitter,” IEEE Commun. Lett., vol. 29, no. 12, pp. 2780–2784, Dec. 2025.

[7] C. Liu, W. Mei, and Z. Chen, “Joint 3D orientation and location optimization for UAV-mounted intelligent reflecting surface,” in Proc. IEEE Global Commun. Conf. (GLOBECOM), Cape Town, South Africa, Dec. 2024, pp. 2725–2730.

[8] X. Zhou, L. Huang, T. Ye, and W. Sun, “Computation bits maximization in UAV-assisted MEC networks with fairness constraint,” IEEE Internet Things J., vol. 9, no. 21, pp. 20997–21009, Nov. 2022.

[9] J. Zhang et al., “Decision transformers for wireless communications: A newparadigm of resource management,” IEEE Wireless Commun., vol. 32, no. 2, pp. 180–186, Apr. 2025.

[10] L. Chen et al., “Decision transformer: Reinforcement learning via sequence modeling,” in Proc. Int. Conf. Neural Inf. Process., 2021, vol. 34, pp. 15084–15097.

[11] Y. Zhang et al., “A unified deterministic channel model for multi-type RIS with reflective, transmissive, and polarization operations,” IEEE Trans. Veh. Technol., vol. 75, no. 2, pp. 2821–2833, Feb. 2026.

[12] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[13] T. P. Lillicrap et al., “Continuous control with deep reinforcement learning,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2016.

[14] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in Proc. 35th Int. Conf. Mach. Learn. (ICML), vol. 80, pp. 1861–1870, 2018.

[15] Y. Hou, H. Hong, Z. Sun, D. Xu, and Z. Zeng, “The control method of twin delayed deep deterministic policy gradient with rebirth mechanism to multi-DOF manipulator,” Electronics, vol. 10, no. 7, Art. no. 870, Apr. 2021.