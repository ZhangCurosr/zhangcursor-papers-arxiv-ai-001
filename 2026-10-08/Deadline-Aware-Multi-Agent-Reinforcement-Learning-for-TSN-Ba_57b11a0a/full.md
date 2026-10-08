# Deadline-Aware Multi-Agent Reinforcement Learning for TSN-Based Vehicular Edge Networks

Bernardo A. C. Pereira<sup>∗</sup>, Marcos Carvalho<sup>∗</sup>, Fatih Temiz<sup>†</sup>, Shavbo Salehi<sup>†</sup>, Melike Erol-Kantarci<sup>†</sup>, Fellow, IEEE, Andreas Gavrielides<sup>‡</sup>, Johann M. Marquez-Barja<sup>‡</sup>, Daniel F. Macedo<sup>∗</sup>

<sup>∗</sup>Universidade Federal de Minas Gerais, Brazil

<sup>†</sup>School of Electrical Engineering and Computer Science, University of Ottawa, Ottawa, Canada

<sup>‡</sup>University of Antwerp - imec, IDLab - Faculty of Applied Engineering, Belgium

E-mails: bernardoalvescosta10@gmail.com,{marcoscarvalho, damacedo}@dcc.ufmg.br

{ftemi033, ssale038, melike.erolkantarci}@uottawa.ca

{andreas.gavrielides, johann.marquez-barja}@imec.be

Abstract—Vehicular edge computing (VEC) enables latencysensitive applications by bringing computing and networking resources closer to vehicles. However, existing approaches often overlook network contention among co-located services with heterogeneous and dynamic latency requirements. While timesensitive networking (TSN) provides bounded-latency communication, conventional and reinforcement learning-based schedulers struggle to adapt to highly dynamic vehicular environments and inter-queue dependencies. To address these limitations, we propose a multi-agent reinforcement learning (MARL) approach for queue-level scheduling in TSN-enabled VEC. Each TSN queue is assigned an autonomous agent that jointly learns the queue service order and time-slot duration to minimize deadline misses under speed-dependent latency requirements. We employ multi-agent proximal policy optimization (MAPPO) to enable coordinated yet autonomous scheduling decisions. Evaluation against single-agent, multi-agent, and non-learning-based baselines shows that MAPPO provides robust performance across different traffic profiles. Compared with centralized single-agent methods, it reduces service latency by up to 66.2% and improves reliability by up to 271.8%. Furthermore, unlike urgency-based heuristics, MAPPO ensures balanced scheduling while achieving lower inference times compared to other MARL methods.

Index Terms—Extended Reality, Multi-agent Reinforcement Learning, Time-Sensitive Networking, Vehicular Edge Computing

## I. INTRODUCTION

Vehicular edge computing (VEC) has emerged as a key infrastructure for enabling a wide range of vehicular applications, including autonomous driving support and immersive applications such as augmented and virtual reality (AR/VR) [1]. These emerging applications introduce stringent quality of service requirements, demanding ultra-low latency and high reliability to ensure responsive operation. To meet these requirements, VEC deploys computing and networking resources at roadside units (RSUs), bringing them closer to vehicles [2]. In this architecture, RSUs can cache and deliver application content locally, improving service responsiveness. While existing VEC research primarily focuses on managing computational resources at edge servers, it generally assumes that edge servers operate under a best-effort Ethernet communication model.

Time-sensitive networking (TSN) has become a key technology for supporting latency-sensitive services over Ethernet networks. Developed by the IEEE 802.1 working group, TSN comprises a set of standards that provide bounded latency and high reliability for time-critical services [3]. Among these mechanisms, the IEEE 802.1Qbv Time-Aware Shaper (TAS) enables time-based traffic scheduling, allowing the management of co-located latency-sensitive flows [4]. Nevertheless, enabling the management of mixed deadline-critical traffic in highly dynamic edge environments remains an open challenge [5]. This challenge becomes even more complex in vehicular networks due to the high mobility of vehicles. To mitigate this issue, we argue that the latency budget for critical services at the edge server must be tightened as vehicle speeds increase, ensuring that packets can be successfully delivered before connectivity degradation occurs.

In this context, conventional TSN schedulers based on optimization techniques, such as satisfiability modulo theories and integer linear programming, face challenges in rapidly adapting schedules to highly dynamic environments [6]. Conversely, reinforcement learning (RL) has been used for TSN scheduling by enabling adaptive decision-making through continuous interaction with the environment. Nevertheless, existing studies primarily optimize intermediate network devices while overlooking the traffic sources. Furthermore, existing RL-based approaches rely on a centralized scheduling policy that determines the transmission schedule for all queues, rather than allowing each queue to make autonomous scheduling decisions [7]. As a result, they are limited in capturing interqueue dependencies in dynamic environments, particularly when latency deadlines vary over time.

To address these limitations, inspired by the approach in [8], we propose a multi-agent reinforcement learning (MARL) approach in which each TSN queue is assigned an autonomous agent. The agents learn scheduling policies that jointly determine the service order and time-slot duration of each queue to minimize deadline misses. We also leverage multi-agent proximal policy optimization (MAPPO) [9], which employs centralized training with decentralized execution (CTDE).

![](images/eb633adc5b9b48b003df2e33113074e563d6ec5861452e971276f9fc530d2349.jpg)  
Figure 1: Overall system model.

During training, the centralized critic exploits global state information to produce more informative policy updates, encouraging coordinated scheduling decisions while enabling decentralized execution. We evaluate the effectiveness of the proposed MARL approach against centralized single-agent and non-learning-based baselines. The results demonstrate that MAPPO outperforms the considered approaches, reducing service latency by up to 66.2% and improving reliability by up to 271.8%, while achieving lower inference times than the other MARL methods.

## II. RELATED WORKS

The authors in [10] employ proximal policy optimization (PPO) to dynamically optimize scheduling parameters, thereby reducing end-to-end latency in industrial network environments. Despite its effectiveness in ensuring bounded latency, the proposed approach is evaluated in a limited scenario with only two queues. This limitation may hinder its scalability and effectiveness in more complex network environments, where scheduling decisions for one queue can directly influence the performance of others. Conversely, the authors in [11] introduce a deep deterministic policy gradient (DDPG)-based method for adapting time-slot durations across multiple queues, aiming to mitigate the impact of transmission failures caused by external factors (i.e., time-synchronization errors) on application latency. Although the approach ensures bounded latency for critical applications, it does not explicitly capture the interdependence among queues, which may limit its ability to coordinate resource allocation in highly dynamic network environments with mixed and heterogeneous latency requirements. Recent studies have explored MARL solutions with different agent designs to address the complexity of TSN scheduling. For instance, [12] divides the scheduling problem into dedicated routing and timing agents, enabling adaptive traffic management at the device level. Likewise, [13] introduces a multi-agent architecture within a centralized control framework, in which routing and scheduling agents collaborate to minimize latency in dynamic network environments. However, these approaches focus on network-level coordination rather than cooperative optimization of individual host-level queues. Our work learns coordinated scheduling policies that account for queue states, heterogeneous latency requirements, and dynamic edge traffic.

## III. SYSTEM MODEL AND PROBLEM FORMULATION

As illustrated in Figure 1, we consider RSUs co-located with MEC servers and deployed along a highway segment. Let $\mathcal { U } = \{ u _ { 1 } , u _ { 2 } , \ldots , u _ { N } \}$ denote the set of N vehicles within the coverage range of the RSU, and let $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ denote their respective speeds. The MEC server hosts two classes of applications: traditional AR and semantic AR (SeAR) applications [14]. Unlike traditional AR, which transmits the entire video content, SeAR transmits compact data units containing masks that capture the semantic content of the video. This allows SeAR to convey the relevant semantic information while reducing the amount of data transmitted over the network. Furthermore, to ensure strict temporal isolation, the server is equipped with a TSN-capable network interface card (NIC) with a set of queues, denoted as $\mathcal { Q } = \{ q _ { 0 } , q _ { 1 } , . . . , q _ { M } \}$ , where each XR application is mapped to a dedicated queue. Since our focus is on queue-level scheduling at the MEC layer, we model only the downstream traffic generated by the considered applications at the edge server.

## A. XR Traffic, Latency and Dynamic Deadline Model

a) XR Traffic: We adopt the statistical traffic model for XR applications presented in [14] and incorporate the protocol data unit (PDU)-set concept as in [15]. The traffic stream for each vehicle $u _ { n } \in \mathcal { U }$ is modeled as a sequence of PDU-sets, where each PDU-set is indexed by $f \in \mathcal { F } _ { u _ { n } }$ and consists of a group of closely related PDUs that carry a single applicationlevel payload. The individual PDUs within a given PDU-set are indexed by $j \in \mathcal { I } _ { u _ { n } , f }$ . The temporal characteristics of the XR traffic are determined by the arrival times of PDU-sets and their constituent PDUs. Let $t _ { u _ { n } , f , q _ { m } }$ denote the arrival timestamp of PDU-set $f$ at TSN queue $q _ { m }$ of vehicle $u _ { n }$ . Each PDU-set consists of $N _ { \mathbb { I P } } ^ { f }$ PDUs, where $t _ { u _ { n } , f , j , q _ { m } }$ denotes the arrival timestamp of PDU j within PDU-set $f$ at the queue $q _ { m }$ The PDUs are enqueued at the TSN queue according to their arrival times and served following a first-in, first-out (FIFO) policy during each transmission opportunity.

b) Latency Model: PDUs experience queueing latency while waiting for a transmission opportunity. Let $\begin{array} { c } { t _ { u _ { n } , f , j , q _ { m } } ^ { s } } \\ { { \bf { t } } \hat { \bf { \Delta } } _ { \mathrm { { \bf { P } } } } \hat { \bf { \Delta } } _ { \mathrm { { D } } \mathrm { { D I } } } ^ { s } ] . } \end{array}$ denote the service start time of PDU j belonging to PDUset $f$ of vehicle $u _ { n }$ at queue $q _ { m }$ . To capture the worst-case waiting time at each queue, we define the age of the oldest PDU (AOPDU), representing the waiting time of the oldest

PDU currently in queue $q _ { m }$ at observation time step t, denoted by $g _ { m } ( t )$ . Let $B _ { m } ( t )$ denote the set of PDUs currently waiting in queue $q _ { m }$ . The AOPDU is given by

$$
g _ { m } ( t ) = \left\{ \begin{array} { l l } { \underset { j \in \mathcal { B } _ { m } ( t ) } { \operatorname* { m a x } } \left( t - t _ { u _ { n } , f , j , q _ { m } } \right) , } & { \mathrm { i f } \mathcal { B } _ { m } ( t ) \neq \emptyset , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

Due to the interdependencies among the PDUs within a PDUset, the delayed delivery of a single PDU may prevent the timely reconstruction of an XR video frame or semantic mask. The total latency experienced by PDU-set $f ,$ denoted by $\Gamma _ { u _ { n } , f , q _ { m } }$ , is defined as the time elapsed between the arrival of its first PDU at the queue $q _ { m }$ and the completion of its last PDU, expressed as $\Gamma _ { u _ { n } , f , q _ { m } } = t _ { u _ { n } , f , q _ { m } } ^ { c } - t _ { u _ { n } , f , 1 , q _ { m } }$ , where $t _ { u _ { n } , f , q _ { m } } ^ { c }$ denotes the completion time of the last PDU in PDUset f, and $t _ { u _ { n } , f , 1 , q _ { m } }$ denotes the arrival time of its first PDU at queue $q _ { m }$

c) Dynamic PDU-Set Deadline Model: The sequence of PDU-sets $\mathcal { F } _ { u _ { n } }$ requested by vehicle $u _ { n }$ is mapped to a tuple $\Phi _ { u _ { n } , f } ~ = ~ \langle s r c _ { u _ { n } } , d s t _ { u _ { n } } , N _ { \mathrm { I P } } ^ { f } , q _ { m } , D _ { u _ { n } , f } \rangle$ , where $s r c _ { u _ { n } }$ and $d s t _ { u _ { n } } = u _ { n }$ denote the source RSU and destination vehicle, respectively, $N _ { \mathrm { I P } } ^ { f }$ is the number of PDUs in PDU-set $f , q _ { m }$ is the associated TSN queue, and $D _ { u _ { n } , f }$ is its dynamic deadline. The deadline is defined as $D _ { u _ { n } , f } = \alpha D _ { \mathrm { b a s e } }$ , where $D _ { \mathrm { b a s e } }$ is the fixed base deadline of the corresponding XR application. For traditional AR applications, $D _ { \mathrm { b a s e } }$ is determined by the generation rate; for example, a 90 Hz application has a base deadline of 11.11 ms. For SeAR applications, we consider a tighter base deadline of 2 ms due to the importance of timely semantic information. The speed-dependent scaling factor α is given by

$$
\alpha = \alpha _ { \operatorname* { m a x } } - \frac { v _ { a v g } - v _ { a v g } ^ { \operatorname* { m i n } } } { v _ { a v g } ^ { \operatorname* { m a x } } - v _ { a v g } ^ { \operatorname* { m i n } } } ( \alpha _ { \operatorname* { m a x } } - \alpha _ { \operatorname* { m i n } } ) .\tag{2}
$$

where $\alpha _ { \mathrm { m a x } }$ and $\alpha _ { \mathrm { { m i n } } }$ are the maximum and minimum deadline adjustment factors, respectively, $v _ { a v g }$ denotes the average speed of $u ,$ and $v _ { a v g } ^ { \mathrm { m i n } }$ and $v _ { a v g } ^ { \operatorname* { m a x } }$ denote the minimum and maximum average speeds considered. Thus, α dynamically adjusts the application deadline according to vehicle speed. As speed increases, the vehicle remains within the RSU coverage area for less time, reducing the time available for transmission. To satisfy the XR applications’ requirements, the total latency of each PDU-set must not exceed its corresponding dynamic deadline $D _ { u _ { n } , f }$ . Accordingly, we define the indicator of a successful on-time transmission of PDU-set f from vehicle $u _ { n }$ as

$$
\mathbb { I } _ { u _ { n } , f } = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \Gamma _ { u _ { n } , f , q _ { m } } \leq D _ { u _ { n } , f } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{3}
$$

## B. TSN Schedule Model

The network execution time is divided into fixed-duration scheduling cycles, indexed by $c ,$ with $\Delta t _ { \mathrm { c y c l e } } ~ = ~ 1$ ms. At the beginning of each cycle, a schedule $S _ { c }$ partitions the cycle into M time slots, where $w _ { c , m }$ denotes the duration (in µs) allocated to queue $q _ { m }$ . The allocated durations satisfy $\begin{array} { r } { \sum _ { m = 0 } ^ { M } w _ { c , m } = \Delta t _ { \mathrm { c y c l e } } } \end{array}$ . The second step consists of determining the service order of the queues within the cycle. To govern this, we introduce a continuous priority score $p _ { c , k } \in [ 0 , 1 ]$ Queues are scheduled in descending order of their assigned priority score $p _ { c , m }$ . Furthermore, two or more queues may be assigned the same score, indicating equal urgency. In such cases, ties are resolved by granting higher precedence to queues with larger indices (i.e., $q _ { M } > q _ { M - 1 } > \cdot \cdot \cdot > q _ { 0 } )$ reflecting standard hardware driver-level queue prioritization. Consequently, the resulting service order is defined as a permutation $\pi _ { c } ~ : ~ \{ 1 , \ldots , M \} ~  ~ \{ 0 , \ldots , M - 1 \}$ , where $\pi _ { c } ( k )$ denotes the queue index placed in the k-th position of the schedule. This is obtained by sorting the queues in descending order of $p _ { c , m } .$ , and breaking ties by selecting the largest index m, as $\pi _ { c } = \mathrm { a r g s o r t } _ { m \in \{ 0 , . . . , M \} } ( - p _ { c , m } , - m )$ Given the formalization above, the final schedule $S _ { c }$ is defined as an ordered sequence of tuples, given by $\begin{array} { r l } { S _ { c } } & { { } = } \end{array}$ $\langle ( w _ { c , \pi _ { c } ( 1 ) } , \pi _ { c } ( 1 ) ) , \ldots , ( w _ { c , \pi _ { c } ( M ) } , \pi _ { c } ( M ) ) \rangle$ , where $w _ { c , \pi _ { c } ( m ) }$ is the corresponding assigned time slice duration and $\pi _ { c } ( k )$ denotes the index of the queue scheduled in the k-th position of the cycle.

## C. Problem Formulation

The primary objective of the scheduler is to maximize the aggregate number of on-time PDU-sets across all co-located, mixed-criticality XR applications. Let W denote the set of time slot duration decisions and P denote the set of priority scores. The scheduling problem is formulated as

$$
\underset { \mathbf { W } , \mathbf { P } } { \operatorname* { m a x } } \quad \sum _ { u \in \mathcal { U } } \sum _ { f \in \mathcal { F } _ { u } } \mathbb { I } _ { u , f }\tag{4}
$$

$$
\mathrm { s . t . } \sum _ { m = 1 } ^ { M } w _ { c , m } = \Delta t _ { \mathrm { c y c l e } } , \quad \forall c ,\tag{5}
$$

$$
D _ { u , f } \leq D _ { \mathrm { b a s e } } , \quad \forall u , \forall f ,\tag{6}
$$

$$
D _ { u , f } ^ { \mathrm { S e A R } } < D _ { u , f } ^ { \mathrm { A R } } , \quad \forall u .\tag{7}
$$

Constraint (5) guarantees that the sum of all allocated time slots equals the predefined cycle duration $\Delta t _ { \mathrm { c y c l e } }$ . Constraint (6) dictates that the dynamic deadline $D _ { u _ { n } , f }$ of any application cannot exceed its base deadline. Finally, constraint (7) strictly enforces the tighter delay requirements for SeAR applications compared to traditional AR applications.

## D. MDP Formulation

We formulate the TSN scheduling problem as a multi-agent partially observable Markov decision process. Since each agent controls exactly one queue, we index agents by their queue index $m \in \{ 0 , \ldots , M \}$ . Below, we describe the three essential components of each agent.

• Observation Space: At the start of scheduling cycle $c ,$ each queue agent m receives a local observation $_ { O _ { c , m } }$ that captures the traffic state and queue dynamics, given by $o _ { c , m } = \langle B _ { c , m } , \mathrm { M } _ { c , m } , g _ { c , m } , d _ { c , m } \rangle$ , where $B _ { c , m }$ represents the backlog, defined as the total number of PDUs currently waiting for transmission in queue $q _ { m } .$ $M _ { c , m }$ denotes the number of missed PDU-sets waiting in queue $q _ { m } . ~ g _ { c , m }$ represents the AOPDU waiting time in queue $q _ { m }$ , while $d _ { c , m }$ denotes the urgency rate, defined as $\begin{array} { r } { d _ { c , m } = \frac { g _ { c , m } } { D _ { u n , f } } } \end{array}$

• Action Space: At each scheduling cycle $c ,$ each agent m generates a two-dimensional action vector. Specifically, the action for agent m is defined as $a _ { c , m } = \langle a _ { c , m } ^ { d u r } , a _ { c , m } ^ { o r d } \rangle$ where $a _ { c , m } ^ { d u r } \in [ 0 , 1 ]$ represents the agent’s desired fraction of the time slice duration. The actual time slice duration $w _ { c , m }$ (in $\mu \mathrm { s } )$ allocated to queue $q _ { m }$ is computed as $\begin{array} { r } { w _ { c , m } = \frac { { a _ { c , m } ^ { d u r } } } { \sum _ { m = 0 } ^ { M - 1 } { a _ { c , m } ^ { d u r } } } \cdot \Delta t _ { \mathrm { c y c l e } } } \end{array}$ . In contrast, $a _ { c , m } ^ { o r d } \in [ 0 , 1 ]$ determines the service order $\pi _ { c } ( m )$ of the queue within the scheduling cycle.

• Reward Function: The reward is driven by the urgency rate $d _ { c , m } .$ , which measures the AOPDU relative to its dynamic deadline, and by PDU-set deadline satisfaction. While eq. 4 directly captures the objective of maximizing PDU-set deadline satisfaction through the binary indicator in eq. 3, this indicator provides limited intermediate feedback for RL-based scheduling. Therefore, Eq. (8) uses a dense, bottleneck-aware surrogate reward based on queue urgency and accumulated deadline violations, providing continuous feedback to guide the policy toward higher PDU-set deadline satisfaction:

$$
\Psi _ { c , m } = \left\{ { \begin{array} { l l } { d _ { c , m } , } & { { \mathrm { i f ~ } } d _ { c , m } \leq 1 } \\ { 1 + \lambda \cdot M _ { c , m } \cdot ( d _ { c , m } - 1 ) , } & { { \mathrm { i f ~ } } d _ { c , m } > 1 } \end{array} } \right.\tag{8}
$$

where λ is a scaling factor. When a queue operates safely by transmitting each PDU within its dynamic deadline $( d _ { c , m } ~ \leq ~ 1 )$ , the penalty simply reflects its current urgency rate. However, once a deadline is violated $( d _ { c , m } > 1 )$ , the penalty is amplified proportionally to both the violation magnitude and the number of missed PDU-sets $M _ { c , m }$ . The global team reward $R _ { c }$ is determined by the bottleneck queue, defined as the negative maximum penalty across all TSN queues, given by $R _ { c } = - \operatorname* { m a x } _ { m \in \{ 0 , . . . , M \} } \Psi _ { c , m }$

## IV. MAPPO FOR TSN-BASED VEC

## A. MAPPO Framework

To efficiently address the optimization problem, we employ the MAPPO algorithm [9]. As illustrated in fig. 1, M cooperative agents control an individual TSN queue. MAPPO extends the single-agent PPO framework to the multi-agent setting by adopting a CTDE paradigm. Specifically, each agent is equipped with an actor policy that determines its local scheduling action, while a centralized critic estimates the value of the global system state during training. The critic network has access to global information, including the observations and actions of all agents, allowing it to capture the inter-queue dependencies and coupling among the queues. In contrast, each actor receives only its corresponding local observation and learns a policy for selecting its action. The centralized critic network provides a more informative value estimate for computing the policy gradient and advantage function, thereby reducing the non-stationarity and variance that arise when multiple agents simultaneously adapt their policies. Consequently, the actor can learn scheduling decisions that implicitly account for the impact of its actions on the overall system performance.

## B. Training and Inference Procedure

In our model, each TSN queue has a local actor policy $\pi _ { \theta _ { m } }$ while a centralized critic $V _ { \phi }$ is used during training. During data collection, each agent observes its local queue state $o _ { c , m } ^ { t }$ and selects an action according to its policy $\pi _ { \theta _ { i } } ( a _ { c , m } | o _ { c , m } ^ { t } )$ The environment then executes the joint actions of all agents, determining the corresponding TSN scheduling decisions, and returns a global team reward $R _ { c }$ and the next system state $s _ { t + 1 }$ These transitions are stored in the buffer D for subsequent policy updates. After trajectory collection, the centralized critic uses the global system information to estimate the value function, which is used to compute the advantages $( \mathrm { i . e . }$ , how good each action was relative to expectations) and returns for policy optimization. The actor policies are then updated using the PPO objective, while the centralized critic is updated by minimizing the value-function loss. During inference, the centralized critic is no longer required, and each queue independently selects its scheduling action using only its local observation and the corresponding trained actor policy.

## V. PERFORMANCE EVALUATION

## A. Experimental Setup

We use vehicular mobility data collected on the Smart Highway testbed [16], providing vehicle counts and average speeds for different highway segments. The TSN network has eight queues supporting five XR applications, categorized as AR or SeAR, with traffic modeled using Johnson’s $S _ { U }$ distribution [14]. As shown in Table I, queues q<sub>0</sub>, q<sub>1</sub>, q<sub>3</sub>, $q _ { 4 }$ , and $q _ { 7 }$ handle traditional AR traffic, while the others support different SeAR resolutions with a 2 ms base deadline. All deadlines are scaled by $\alpha \in [ 0 . 5 0 , 0 . 9 0 ]$ , reducing the MEC delay budget from 90% at low speeds $( v _ { a v g } ^ { \operatorname* { m i n } } = 1 0 )$ to 50% at high speeds $( v _ { a v q } ^ { \operatorname* { m a x } } = 1 2 0 )$ to account for speeddependent latency deadlines. Table I also reports the MAPPO hyperparameters. Each episode contains 150 steps, with the number of connected vehicles and their average speed sampled from the dataset at its start. Average speed and vehicle count are clipped to 10–120 and 1–100, respectively. Clients are then distributed among queues using a uniform multinomial distribution, with at least one active user per queue. We set the reward penalty weight to $\lambda = 1 . 0$ . Results are reported as the mean and standard error across three random seeds.

Baselines: To evaluate the performance of our MAPPO approach, we compare it against a set of baselines, including alternative MARL architectures, single-agent algorithms, and a heuristic-based scheduling policy, as described below:

• MARL: Multi-agent transformer (MAT) [17] and heterogeneous-agent proximal policy optimization (HAPPO) [18], where we also model each queue as an independent agent.

Table I: VEC and MAPPO parameters
<table><tr><td>VEC</td><td>Value</td><td>MAPPO parameter</td><td>Value</td></tr><tr><td>Network speed (Gbps)</td><td>10</td><td>Learning rate</td><td>10⁻4</td></tr><tr><td>Max packet size (B)</td><td>1428</td><td>gamma</td><td>0.99</td></tr><tr><td>Number of vehicles</td><td>1-100</td><td>Episode length</td><td>150</td></tr><tr><td>Average speed (km/h)</td><td>10-120</td><td>PPO epochs</td><td>5</td></tr><tr><td colspan="2">AR setting/Base deadline</td><td colspan="2">SeAR setting/Base deadline</td></tr><tr><td colspan="2">q0, q3: 3840× 1920@90 Hz / 11.11 ms</td><td colspan="2">q2: 1920×720@60 Hz / 2 ms</td></tr><tr><td colspan="2">q1, q4, q7: 3840× 1920 @72 Hz / 13.80 ms</td><td colspan="2"></td></tr><tr><td colspan="2"></td><td colspan="2">q5: 1280×480@60 Hz / 2 ms q6: 2560×960@60 Hz / 2 ms</td></tr></table>

• Single-agent RL: PPO and advantage actor-critic (A2C) for centralized control, where a single agent has control over all queues.

• Urgency-based heuristic: A non-learning policy that prioritizes queues approaching their deadlines based on the urgency rate. The time slot allocation is given by $\begin{array} { r } { t _ { i } = \frac { \tilde { d _ { c , i } } } { \sum _ { m = 0 } ^ { M - 1 } { d _ { c , m } } } \cdot \Delta t _ { \mathrm { c y c l e } } . } \end{array}$

## B. Experimental Results

Learning performance: As shown in fig. 2, MARL methods converge faster than single-agent methods, reaching stable reward levels in fewer training steps, whereas PPO and A2C require longer training to converge. By learning from multiple

![](images/5f873d55bd892891ac6f02ed5b47adcdb4460def9d19739fb1f3b59318e2c1ad.jpg)  
Figure 2: Average cumulative reward.

agents, MARL methods can better capture the impact of queue-level decisions on overall network performance, potentially improving convergence over single-agent methods. Nevertheless, MAPPO converges slightly faster than MAT and HAPPO, likely because it jointly updates all agents using a shared objective, whereas MAT and HAPPO rely on sequential mechanisms that introduce additional dependencies during learning. Despite faster MARL convergence, PPO achieves the highest and most stable final reward. However, higher training rewards do not necessarily imply better network performance under heterogeneous queue deadlines, as shown next.

Service Latency Analysis: Fig. 3 presents the average and p99 PDU-level service latency for each queue. As observed, MARL methods achieve lower latency across all queues, reducing the average PDU latency by at least 19.5% compared with A2C and 14.9% compared with PPO for the AR queues. Similarly, MARL methods reduce worst-case (p99) latency, with reductions of up to 38.6% compared with the single-agent methods. This performance gap is even more pronounced for

SeAR queues, where MARL methods reduce average and p99 latency by up to 66.2% and 68.3%, respectively, compared with single-agent methods.

![](images/9773d3ad88d6bb95d35635da61f48d0a49f215a58209a090742953faaf2037c1.jpg)  
Figure 3: Average (solid fill) and 99th-percentile (transparent fill) PDU-level service latency.

The results suggest that single-agent policies do not fully capture the interactions and resource trade-offs among queues with heterogeneous requirements, which may explain why PPO achieves higher training rewards than the MARL methods. In particular, PPO and A2C appear to prioritize AR queues, whose higher traffic intensity contributes more frequently to the reward, while postponing service for lowerdemand SeAR queues. Although this behavior can increase the number of satisfied AR deadlines and thus the training reward, it may degrade overall network performance by increasing SeAR delay. In contrast, MAPPO, HAPPO, and MAT explicitly model the scheduling problem in a multiagent setting, enabling more coordinated resource allocation across queues. Consequently, they achieve better execution performance despite lower cumulative training rewards. Nevertheless, MAPPO and MAT slightly outperform HAPPO, exhibiting lower variance across the queues, as indicated by the standard deviation. This difference may be attributed to the distinct coordination mechanisms employed by each method. HAPPO updates the agents sequentially using a randomly shuffled update order, which can introduce variability in the learning process. In addition, urgency-based heuristics generally outperform all learning-based methods for SeAR queues, reducing latency by at least 29.1%. However, this gain comes at the cost of higher latency in AR queues, with average latency increasing by up to 315% and worst-case latency reaching 80 ms, corresponding to an 373% increase compared with learning-based methods.

PDU-Set Reliability: Fig. 4 shows the PDU-sets deadline satisfaction rate (i.e., reliability). To achieve high reliability, scheduling policies must ensure that all PDUs belonging to a PDU-set are delivered within its dynamic deadline, as designed by eq. 3. Accordingly, the reliability results are consistent with the latency analysis, with MARL methods achieving higher reliability across the AR queues, exceeding 70% compared with approximately 60% for single-agent methods, corresponding to a 16.66% improvement. As observed previously, the urgency-based method tends to prioritize SeAR queues due to their stricter deadlines. This urgency-driven approach explains its higher reliability for SeAR traffic, exceeding 90% for SeAR queues, particularly for q<sub>2</sub> and q<sub>5</sub>. Despite this improvement, they perform poorly across the AR queues, achieving reliability of only around 30%.

![](images/2fa51fc67b498e593d94cf78b02ff2d43d467ecfa29efd7a68a1f91ff58ea3a8.jpg)  
Figure 4: PDU-set reliability (deadline satisfaction).

Finally, Table II presents the inference time of the learningbased methods on an RTX 4060 Ti. MAPPO and the singleagent methods exhibit lower inference times, whereas HAPPO and MAT exceed the scheduling cycle by more than 5 and 10 times, respectively. This gap results from differences in action generation: MAPPO and single-agent methods generate actions in a single forward pass, while MAT generates them sequentially through an autoregressive policy. HAPPO also incurs higher inference time due to its separate actor network for each agent.

Table II: Inference Time (ms)
<table><tr><td>Method</td><td>Inference Time</td><td>Method</td><td>Inference Time</td></tr><tr><td>MAPPO</td><td> $0 . 6 9 \pm 0 . 1 8$ </td><td>MAT</td><td> $1 0 . 9 0 \pm 1 . 2 9$ </td></tr><tr><td>HAPPO</td><td> $5 . 4 8 \pm 0 . 6 1$ </td><td>PPO</td><td> $0 . 7 8 \pm 0 . 2 2$ </td></tr><tr><td>A2C</td><td> $0 . 7 9 \pm 0 . 2 7$ </td><td></td><td></td></tr></table>

Therefore, although HAPPO and MAT achieve competitive performance, their higher inference latency may limit realtime deployment under a 1 ms control cycle. In contrast, MAPPO achieves comparable performance with significantly lower inference latency.

## VI. CONCLUSION

This paper addresses co-located XR traffic scheduling in VEC-TSN integration by considering vehicle speed-dependent latency deadlines. We leverage MAPPO to model queue-level interdependencies under dynamic deadlines, using a centralized critic to learn cooperative scheduling policies. Simulation results show that MAPPO outperforms the considered baselines by reducing latency, improving PDU-set reliability, and maintaining low inference time, making it suitable for realtime scheduling. In future work, we intend to evaluate the impact of wireless networks on the proposed framework.

## ACKNOWLEDGMENT

This work was financed by the Coordenac¸ao de˜ Aperfeic¸oamento de Pessoal de N´ıvel Superior - Brasil (CAPES) - Finance Code 001, CNPq (funding agency from the Brazilian federal government), FAPEMIG (Minas Gerais State Funding Agency), and Sao Paulo Research Founda-˜ tion (FAPESP) with Brazilian Internet Steering Committee (CGI.br), grants 2018/23097-3.

## REFERENCES

[1] J. Feng and J. Zhao, “Resource allocation for augmented reality empowered vehicular edge metaverse,” IEEE Transactions on Communications, vol. 73, no. 3, pp. 1987–2001, 2023.

[2] A. A. Barakabitze, A. Ahmad, R. Mijumbi, and A. Hines, “5G Network Slicing Using SDN and NFV: A Survey of Taxonomy, Architectures and Future Challenges,” Computer Networks, vol. 167, p. 106984, 2020.

[3] J. Farkas, B. Varga, G. Miklos, and J. Sachs, “5G-TSN Integration´ Meets Networking Requirements for Industrial Automation,” Ericsson Technology Review, vol. 96, no. 7, pp. 1–8, 2019.

[4] M. Carvalho and D. F. Macedo, “Performance analysis of the integration of dynamic cloud computing environments and tsn networks,” in 2025 IEEE 11th International Conference on Network Softwarization (NetSoft). IEEE, 2025, pp. 406–414.

[5] V. Gavrilut¸ and P. Pop, “Traffic-type assignment for tsn-based mixedcriticality cyber-physical systems,” ACM Transactions on Cyber-Physical Systems, vol. 4, no. 2, pp. 1–27, 2020.

[6] T. Stuber, L. Osswald, S. Lindner, and M. Menth, “A survey of schedul-¨ ing algorithms for the time-aware shaper in time-sensitive networking (tsn),” Ieee Access, vol. 11, pp. 61 192–61 233, 2023.

[7] N. Zhou, Y. Yao, X. Liang, S. Yao, J. Wan, C. Yang, W. Gui, and W. Gao, “Convergence of reinforcement learning and time-sensitive networking for future industrial ai agent communication: Fundamentals, challenges, and opportunities,” IEEE Communications Surveys & Tutorials, 2026.

[8] M. Carvalho, F. Temiz, S. Salehi, M. Erol-Kantarci, and D. F. Macedo, “Multi-agent reinforcement learning for online traffic scheduling in timesensitive application,” arXiv preprint arXiv:2608.05346, 2026.

[9] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, “The surprising effectiveness of ppo in cooperative multi-agent games,” Advances in neural information processing systems, vol. 35, pp. 24 611– 24 624, 2022.

[10] A. Roberty, Q. Besnard, S. B. H. Said, F. Ridouard, H. Bauer, and A. Geniet, “Configuring the ieee 802.1 q time-aware shaper with deep reinforcement learning,” in NOMS 2024-2024 IEEE Network Operations and Management Symposium. IEEE, 2024, pp. 1–7.

[11] B. Zhou and L. Cheng, “Mitigation of scheduling violations in timesensitive networking using deep deterministic policy gradient,” in Proceedings of the 4th FlexNets Workshop on Flexible Networks Artificial Intelligence Supported Network Flexibility and Agility, 2021, pp. 32–37.

[12] M. Guo, S. He, C. Gu, X. Guo, J. Chen, T. Gao, and T. Wang, “Towards distributed flow scheduling in ieee 802.1 qbv time-sensitive networks,” ACM Transactions on Sensor Networks, vol. 20, no. 5, pp. 1–30, 2024.

[13] S. Garcia-Canton, C. Ruiz de Mendoza, C. Cervell´ o-Pastor, and S. Sal-´ lent, “Multi-agent reinforcement learning-based routing and scheduling models in time-sensitive networking for internet of vehicles communications between transportation field cabinets,” Applied Sciences, vol. 15, no. 3, p. 1122, 2025.

[14] D. G. Mor´ın, D. Medda, A. Iossifides, P. Chatzimisios, A. G. Armada, A. Villegas, and P. Perez, “An extended reality offloading ip traffic´ dataset and models,” IEEE Transactions on Mobile Computing, vol. 23, no. 6, pp. 6820–6834, 2023.

[15] P. Paymard, S. Paris, A. Amiri, T. E. Kolding, F. S. Moya, and K. I. Pedersen, “Pdu-set scheduling algorithm for xr traffic in multi-service 5g-advanced networks,” in ICC 2024 - IEEE International Conference on Communications, 2024, pp. 758–763.

[16] R. C. Bello, M. Camelo, J. Marquez-Barja, and N. Slamnik-Krijestorac,ˇ “Towards sustainable 6g compute continuum: Energy-aware zero-touch network and service management used for dynamic vehicular service deployments,” in 2026 IEEE 23rd Consumer Communications & Networking Conference (CCNC). IEEE, 2026, pp. 1–6.

[17] M. Wen, J. Kuba, R. Lin, W. Zhang, Y. Wen, J. Wang, and Y. Yang, “Multi-agent reinforcement learning is a sequence modeling problem,” Advances in Neural Information Processing Systems, vol. 35, pp. 16 509–16 521, 2022.

[18] J. G. Kuba, R. Chen, M. Wen, Y. Wen, F. Sun, J. Wang, and Y. Yang, “Trust region policy optimisation in multi-agent reinforcement learning,” arXiv preprint arXiv:2109.11251, 2021.