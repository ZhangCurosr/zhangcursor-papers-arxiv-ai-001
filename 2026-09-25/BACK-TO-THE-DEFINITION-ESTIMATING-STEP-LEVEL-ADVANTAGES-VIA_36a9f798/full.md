# BACK TO THE DEFINITION: ESTIMATING STEP-LEVEL ADVANTAGES VIA TRAJECTORY GRAPHS FOR AGENTIC REINFORCEMENT LEARNING

Xincheng Yao<sup>1∗</sup>, Haobo Fu<sup>2</sup>, Weiming Liu<sup>2</sup>, Chongyang Zhang<sup>1,3†</sup>

<sup>1</sup>School of Information Science and Electronic Engineering, Shanghai Jiao Tong University.   
<sup>2</sup>Tencent AI Platform Department. <sup>3</sup>MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University. {i-dover, sunny zhang}@sjtu.edu.cn<sup>1</sup>   
{haobofu, weimingliu}@tencent.com<sup>2</sup>

## ABSTRACT

Group-based reinforcement learning (RL) methods, such as GRPO and its variants, have become a leading paradigm for training reasoning and agentic large language models (LLMs). While their group-normalized advantage estimation is reliable at the response level, it becomes systematically biased at the step level, since coarsegrained trajectory-level advantages are hard to accurately reflect the contribution of individual steps (i.e, failed trajectories may contain valuable steps). Revisiting the foundational RL definition, we notice that GRPO’s success on single-turn tasks stems from its advantage estimation strategy, which adheres to the basic definition: the mean reward of multiple actions sampled from the same state constitutes a credible state-value estimate. Extending the faithful estimation to step-level would in principle demand sampling multiple actions from each intermediate state, which is too costly on a per-state basis. To mitigate this issue, we can aggregate similar states across trajectories to better leverage global information. Since each trajectory is a chain of state-action transitions, cross-trajectory information flow requires modeling state-action transitions across trajectories, which naturally forms a directed graph. Building on this insight, we propose a Graph-based Faithful sTeplevel credit-assignment framework (GRAFT) that grafts all rollout trajectories into a trajectory graph, recovering node state-values via Bellman iteration on the graph, and assigning credit to each edge by the node value difference. Theoretically, the estimated step-level advantage faithfully adheres to the basic advantage definition in RL. To further ensure the reliability of step-level advantage estimation, we further propose Graph GAE, which extends GAE to the trajectory graph for reducing the impact of state-value estimation bias. Experiments across a range of multi-turn agentic benchmarks show consistent gains over GRPO and superior performance compared to recent agentic RL algorithms. Code will be available at https: //github.com/xcyao00/GRAFT.

## 1 INTRODUCTION

Reinforcement learning (RL) has emerged as a cornerstone for training reasoning and agentic large language models (LLMs) (OpenAI, 2024; DeepSeek-AI, 2025; Team, 2026a;c; 2025; 2026b). Among the various RL algorithms (Schulman et al., 2017; Ahmadian et al., 2024; Rafailov et al., 2023; Williams, 1992; Shao et al., 2024) adapted to the LLM domain, Group Relative Policy Optimization (GRPO) (Shao et al., 2024) and its variants (Seed, 2025; Zheng et al., 2025; Gao et al., 2025; Yao et al., 2026) have gained the most significant traction, primarily due to their critic-free design. For each prompt, GRPO samples a group of N responses and estimates the advantage of each response by normalizing its reward against the group-level statistics. Within the standard RL framework, this strategy is theoretically well-grounded at the response level: when the entire response is treated as a single action and the prompt q as the initial state, the group mean exactly serves as a valid estimate of the state value $V _ { \pi } ( \boldsymbol { \dot { q } } )$ , and the resulting normalized advantage aligns with the standard definition $A _ { \pi } ( q , a ) = Q _ { \pi } ( q , a ) { \overset {  } { - } } V _ { \pi } ( q )$ (see Sec.2).

However, this mechanism shifts fundamentally in multi-turn tasks, where trajectory-level advantages are too coarse-grained to accurately capture the contribution of individual steps. A successful trajectory may contain redundant or erroneous steps that receive unwarranted credit, while a failed trajectory may include valuable steps whose benefits are obscured by subsequent mistakes. A common remedy is to employ a Process Reward Model (PRM) to assign step-wise rewards and extend the group-based advantage estimation at each step index t (Shao et al., 2024; Cui et al., 2025; Xi et al., 2025). However, training PRMs requires costly step-level annotations and often suffers from distributional shift when applied to out-of-distribution reasoning traces. Moreover, as analyzed in Sec. 2, applying group normalization per step t incurs systematic bias. Because $s _ { i , t }$ is a trajectory prefix, different trajectories usually occupy different intermediate states at the same step index t; consequently, the “group” used to estimate $V _ { \pi } ( s _ { i , t } )$ aggregates samples from distinct states rather than a single shared state. The resulting bias is systematic and grows with trajectory diversity. The theoretically faithful solution is to resample $N - 1$ 1 additional continuations from each intermediate state, but this recovers unbiasedness at an $\mathcal { O } ( N ^ { 2 } T )$ rollout cost that is essentially unaffordable for long-horizon multi-turn training.

To mitigate this issue while preserving theoretically faithful advantage estimation, we should aggregate similar states across trajectories to better leverage global information. Since each trajectory can be regarded as a chain of state-action transitions, cross-trajectory information flow requires modeling state-action transitions across trajectories, which naturally forms a directed graph, where nodes represent states and edges represent actions. With this insight, we can reorganize N trajectories from isolated chains into a unified trajectory graph. Each node in this graph aggregates all actions executed from the same state across all trajectories, providing the on-state action groups required for a more credible step-level advantage estimation at no additional cost.

Building on the trajectory graph insight, we propose a Graph-based Faithful sTep-level credit assignment (GRAFT) framework, which grafts N rollout trajectories into a unified trajectory graph so that more faithful step-level advantages can be estimated in accordance with the foundational RL definition. First, graph construction canonicalizes states via exact matching or embedding-based similarity matching, merging semantically identical states across trajectories into a single node with a shared on-state action group. Second, to estimate the state-value for every node, value propagation applies Bellman iteration over the graph using only the rewards attached to the terminal nodes. In sparse reward settings, the converged value estimation is a direct empirical approximation of the definition of a standard value function. Third, given these node values, we define the step advantage as $A _ { t } = \gamma V ( s _ { t + 1 } ) - V ( s _ { t } )$ , which coincides with the standard advantage definition (see Sec.3). Additionally, to further ensure the reliability of step-level advantage estimation, we take advantage of Generalized Advantage Estimation (GAE), which is the de facto in RL that can reduce the impact of state-value estimation bias. Specifically, we propose Graph GAE, which extends GAE to the trajectory graph by computing a weighted sum of the subsequent multi-hop average advantage estimates. Compared to prior methods, the step-level advantage estimates obtained by our method are theoretically sound and exactly aligned with the fundamental definition. Furthermore, GRAFT only requires outcome rewards, eliminating the reliance on PRM-assigned step-level rewards. Experiments across a range of multi-turn agentic benchmarks show consistent gains over GRPO and superior performance compared to recent agentic RL algorithms.

## 2 PRELIMINARY ANALYSIS

In this section, we revisit the advantage estimation problem from the perspective of standard definitions in reinforcement learning. We begin by formally defining the state-action (Q) function $Q _ { \pi }$ , the value function $V _ { \pi }$ , and the advantage function $A _ { \pi }$

$$
\begin{array} { l l } { { Q _ { \pi } ( s _ { t } , a _ { t } ) = \mathbb { E } _ { s _ { t + 1 } , a _ { t + 1 } , \ldots } \biggl [ \displaystyle \sum _ { l = 0 } ^ { \infty } \gamma ^ { l } r ( s _ { t + l } ) \biggr ] \qquad } } & { { V _ { \pi } ( s _ { t } ) = \mathbb { E } _ { a _ { t } , s _ { t + 1 } , a _ { t + 1 } , \ldots } \biggl [ \displaystyle \sum _ { l = 0 } ^ { \infty } \gamma ^ { l } r ( s _ { t + l } ) \biggr ] } } \\ { { A _ { \pi } ( s _ { t } , a _ { t } ) = Q _ { \pi } ( s _ { t } , a _ { t } ) - V _ { \pi } ( s _ { t } ) } } & { { } } \end{array}\tag{1}
$$

where $a _ { t } \sim \pi ( a _ { t } \mid s _ { t } )$ and $s _ { t + 1 } \sim P ( s _ { t + 1 } \mid s _ { t } , a _ { t } )$ . The $Q _ { \pi } ( s _ { t } , a _ { t } )$ denotes the expected return when taking an action $a _ { t }$ at state $s _ { t } , Q _ { \pi } ( s _ { t } , a _ { t } ) = \mathbb { E } _ { s _ { t + 1 } , a _ { t + 1 } , \dots } [ r ( s _ { t } ) + \gamma r ( s _ { t + 1 } ) + \gamma ^ { 2 } r ( s _ { t + 2 } ) + \dots ] ^ { 1 }$ where $r ( s _ { t } )$ is the reward for transition $( s _ { t } , a _ { t } , s _ { t + 1 } )$ . Compared to $Q _ { \pi } ( s _ { t } , a _ { t } )$ , the only difference in $V _ { \pi } ( s _ { t } )$ is that the action $a _ { t }$ needs to be expected, which means the expected return starting from state $s _ { t } .$ . Then, the connection between $Q _ { \pi } ( s _ { t } , a _ { t } )$ and $V _ { \pi } ( s _ { t } )$ is that $V _ { \pi } ( s _ { t } ) = \mathbb { E } _ { a _ { t } } [ Q _ { \pi } ( s _ { t } , a _ { t } ) ]$ . Thus, the advantage can be understood as “the benefit of performing a specific action $a _ { t }$ in state $s _ { t }$ compared to the average effect of all actions at that $\mathrm { s t a t e } ^ { \prime \prime }$

In LLMs, the input prompt q can be regarded as the initial state, denoted as $s _ { 1 }$ . For action, it can be defined at various levels of granularity: as the whole response $^ { O , }$ individual tokens, or intermediate reasoning steps. In GRPO (Shao et al., 2024), the whole response is treated as a single action, for which a scalar reward is assigned. Since the generation process is terminated, there will be no subsequent $a _ { t + 1 }$ . Moreover, state transitions in LLMs are inherently deterministic $( i . e . , s _ { t + 1 }$ is obtained by concatenating the generated tokens $a _ { t }$ to the previous context $s _ { t } )$ . Therefore, the expectation over $s _ { t + 1 }$ in estimating $Q _ { \pi } ( s _ { t } , a _ { t } )$ can be omitted, also the $a _ { t + 1 }$ and subsequent states and actions can all be omitted. Thus, the Q-value of $( q , o )$ simplifies to $Q ( q , o ) = r$ . Accordingly, the value of $q$ can be defined as $\begin{array} { r } { V ( q ) = \mathbb { E } _ { o } [ Q ( q , o ) ] = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Q ( q , o _ { i } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r _ { i } } \end{array}$

We now revisit the advantage estimation strategy in GRPO, where the advantage is calculated as:

$$
\hat { A } _ { i } = \frac { r _ { i } - \operatorname * { m e a n } ( \{ r _ { 1 } , r _ { 2 } , \ldots , r _ { N } \} ) } { \mathrm { s t d } ( \{ r _ { 1 } , r _ { 2 } , \ldots , r _ { N } \} ) }\tag{2}
$$

It can be found that this estimator aligns precisely with the standard definition of the advantage function in $\operatorname { E q . } ( 1 )$ , as $r _ { i } = Q ( q , o _ { i } )$ and mean $( \{ r _ { i } , r _ { 2 } , \dots , r _ { N } \} ) = V ( q )$ (GRPO has additional normalization). However, a critical mismatch arises: while the derivation treats the entire response as a single action, GRPO assigns ${ \hat { A } } _ { i }$ uniformly to each token within the response.

To correctly utilize the advantage in $\operatorname { E q . } ( 2 )$ , we need to treat the whole response as an action and compute the importance ratio at the sequence level. The sequence-level importance ratio can be derived from the likelihood decomposition, $\begin{array} { r } { \frac { \pi _ { \theta } ( o _ { i } | q ) } { \pi _ { \theta _ { o l d } } ( o _ { i } | q ) } = \frac { \pi _ { \theta } ( o _ { i , 1 } | q ) \pi _ { \theta } ( o _ { i , 2 } | q , o _ { i , < 2 } ) \dots } { \pi _ { \theta _ { o l d } } ( o _ { i , 1 } | q ) \pi _ { \theta _ { o l d } } ( o _ { i , 2 } | q , o _ { i , < 2 } ) \dots } } \end{array}$ . To avoid numerical instability caused by the cumulative multiplication, we can adopt the geometric mean. Accordingly, the importance ratio $w _ { i } ( \theta )$ for the response $o _ { i }$ is defined as follows:

$$
w _ { i } ( \theta ) = \bigg ( \frac { \pi _ { \theta } ( o _ { i } | q ) } { \pi _ { \theta _ { o l d } } ( o _ { i } | q ) } \bigg ) ^ { \frac { 1 } { | o _ { i } | } } = \exp \bigg ( \frac { 1 } { | o _ { i } | } \sum _ { t = 1 } ^ { | o _ { i } | } \log \frac { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta _ { o l d } } ( o _ { i , t } | q , o _ { i , < t } ) } \bigg )\tag{3}
$$

Then, the calibrated GRPO optimization objective is defined as follows:

$$
\mathcal { I } _ { G R P O - C } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } , \{ \sigma _ { i } \} _ { i = 1 } ^ { N } \sim \pi _ { \theta _ { o l d } } ( \cdot | q ) } \bigg [ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m i n } \Big ( w _ { i } ( \theta ) \hat { A } _ { i } , \mathrm { c l i p } ( w _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i } \Big ) \bigg ]\tag{4}
$$

However, assigning a single advantage to the whole response still suffers from coarse granularity, especially in multi-turn agentic tasks that are inherently step-by-step. To this end, we further analyze how to perform step-level advantage estimation that can adhere to the standard definition. For $s _ { 1 }$ (the prompt) and $a _ { i , 1 }$ (the first step), the advantage is $\begin{array} { r } { \hat { A } _ { i , 1 } = Q ( s _ { 1 } , a _ { i , 1 } ) - V ( s _ { 1 } ) = r _ { i , 1 } - \frac { 1 } { N } \sum _ { j = 1 } ^ { N } r _ { j , 1 } , } \end{array}$ where $r _ { i , 1 }$ is the reward for the first step of the i-th response. Here, we use the reward to approximate the return, strictly according to the definition, it should be $Q ( s _ { 1 } , a _ { i , 1 } ) = r _ { i , 1 } + \gamma r _ { i , 2 } + . . . .$ Then, for $s _ { 1 }$ and $a _ { 1 }$ , the advantage estimation can align with GRPO. However, the situation diverges from the second step onward. Since $s _ { 2 }$ is the concatenation of $s _ { 1 }$ and $a _ { 1 }$ , there is no guarantee that $s _ { i , 2 }$ remains identical across all $N$ responses. The worst case is that if every $a _ { i , 1 }$ differs, each $s _ { i , 2 }$ will be distinct. For $s _ { i , 2 }$ and $a _ { i , 2 } ,$ the value $V ( s _ { 2 } )$ cannot be reliably estimated because there are not $N$ samples of $_ { a _ { i , 2 } }$ conditioned on the same $s _ { i , 2 }$ . Despite this, in GRPO with process supervision (Shao et al., 2024; Cui et al., 2025), the advantage is directly calculated by $\begin{array} { r } { \hat { A } _ { i , t } = \frac { r _ { i , t } - \mathrm { m e a n } \left( r _ { 1 , t } , r _ { 2 , t } , \dots , r _ { N \left( t \right) , t } \right) } { \mathrm { s t d } \left( r _ { 1 , t } , r _ { 2 , t } , \dots , r _ { N \left( t \right) , t } \right) } } \end{array}$ where $N ( t )$ is the number of responses that reach step t. This introduces a systematic bias: actions $a _ { j , t } , j \neq i$ that don’t belong to $s _ { i , t }$ are also incorrectly used to estimate $V ( s _ { t } )$ . To mitigate this bias, a straightforward solution is to adopt the Monte Carlo sampling. For $s _ { i , t } ,$ we treat it as the prefix and sample $N - 1$ extra $\{ a _ { i , t } ^ { j } \} _ { j = 1 } ^ { N - 1 }$ from the LLM. With the existing $a _ { i , t }$ , these $N$ actions enable an unbiased estimation of the advantage $\hat { A } _ { i , t }$

## 3 PROPOSED METHOD

![](images/fdeb54d953acbb3f91ef46f7487a64014c3dc2c8d4f798016256477b1cdaee79.jpg)  
Figure 1: Method overview. Given N rollouts for each prompt, GRAFT merges trajectories into a trajectory graph. Terminal rewards are then propagated backward via Bellman iteration to estimate node state-values. The state-value difference between two nodes $\gamma V ( s _ { t + 1 } ) - V ( s _ { t } )$ is assigned as the step-level advantage, which coincides with the standard advantage function definition (see Sec.3.3).

The analysis in Sec.2 reveals a fundamental challenge in applying GRPO-style advantage estimation to agentic RL: the value function $V _ { \pi } ( s _ { i , t } )$ at an intermediate state $s _ { i , t }$ cannot be reliably estimated unless multiple actions are sampled from the same state. The sampling-based remedy outlined in Sec.2 is theoretically sound but computationally prohibitive, requiring $\breve { \mathcal { O } } ( N ^ { 2 } T )$ rollout cost.

To mitigate this issue while preserving definition-adherent estimation, we should aggregate similar states across trajectories to better leverage global information. Crucially, different trajectories are not fully independent, different actions can also reach semantically identical intermediate states (see experiments in Appendix D.2). This means all rollout data from a prompt can form a directed graph, where nodes represent states and edges represent actions, rather than being regarded as N isolated chains. By aggregating all edges outgoing from the same node, we obtain a natural group of actions sampled from the same state, enabling more faithful step-level advantage estimation at no additional rollout cost. The full framework is illustrated in Fig.1. Note that GiGPO (Feng et al., 2025) has employed the way of aggregating cross-trajectory steps sharing the same state into a group to compute step-level advantages within the group. Compared to this plain method of aggregating all steps into multiple independent groups, we further discuss that our method would have more advantages over GiGPO in Appendix A.1.

## 3.1 GRAPH CONSTRUCTION

Formal definition. Let $\mathcal { T } = \{ ( s _ { i , t } , a _ { i , t } , s _ { i , t + 1 } ) \}$ denotes the set of all transition tuples collected from N parallel rollouts for a given task with prompt $q .$ . We construct a task-specific directed graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where each node $u , v \in \mathcal { V }$ corresponds to a canonicalized state, and each edge $( u , v , a ) \in \mathcal { E }$ corresponds to a transition $s _ { t } \stackrel { a } { \to } s _ { t + 1 }$ such that $u = \phi ( s _ { t } )$ and $v = \phi ( s _ { t + 1 } )$

State canonicalization. In our framework, the state represents both the predefined environment state and the LLM response prefix up to a specific step (as discussed in Sec.2). Two states s and $s ^ { \prime }$ will be mapped to the same node if $\bar { \phi ( s ) } = \bar { \phi ( s ^ { \prime } ) }$ . In practice, we can employ two canonicalization modes: (i) exact matching, applied when all states are predefined strings provided by the environment, here $\phi ( s )$ is a deterministic hash of the state string; and (ii) similarity-based matching, applied when states are derived from raw LLM contexts, here s and $s ^ { \prime }$ are merged into a node if the cosine similarity between their encoded embedding vectors exceeds a threshold $\tau _ { \ast }$ , and $\phi ( s )$ returns a unique identifier (UID) for the resulting cluster. The latter handles a key practical reality: language models frequently introduce minor lexical variations even when generating semantically identical content, rendering exact string matching insufficient for reliable state identification in unstructured settings.

Sink nodes. For each trajectory i, the final state $s _ { i , T }$ (the state terminated or reached the max turns) is designated as a sink node with terminal reward $\dot { R } _ { i } \in \{ 0 , 1 \}$ , where 0 and 1 indicate the failure and success of the trajectory, respectively.

## 3.2 STATE-VALUE ESTIMATION VIA BELLMAN ITERATION

Given the graph $\mathcal { G }$ and the rewards in the terminal nodes, our goal is to assign a state-value estimate $V ( u )$ to every node $u \in \mathcal V$ . To leverage the graph’s transition structure, we can perform the classical Bellman value iteration on the graph: starting from the terminal nodes with their outcome rewards, we iteratively propagate value estimates backward through the graph until convergence.

Bellman equation. For sink nodes $f \in S _ { \mathrm { s i n k } }$ , the value is fixed to the terminal reward: $V ( f ) =$ $R ( f ) , f \in \bar { \cal S } _ { \mathrm { s i n k } }$ . For non-sink nodes, we adopt the “action-level” aggregation scheme, where the node value is computed as the mean over distinct actions of the expected next-state value:

$$
V ( u ) = \sum _ { a \in \mathcal { A } ( u ) } \hat { \pi } ( a \mid u ) \sum _ { v \in \mathcal { V } } \hat { P } ( v \mid u , a ) \cdot \gamma \cdot V ( v )\tag{5}
$$

where $\boldsymbol { \mathcal { A } } ( \boldsymbol { u } )$ is the set of distinct actions executed from node u, and ${ \hat { \pi } } ( a \mid u )$ and $\hat { P } ( v \mid u , a )$ are the empirical transition probability estimated from edge counts:

$$
{ \hat { \pi } } ( a \mid u ) = { \frac { \operatorname { c o u n t } ( u \stackrel { a } { \to } \cdot ) } { \sum _ { a ^ { \prime } } \operatorname { c o u n t } ( u \stackrel { a ^ { \prime } } { \to } \cdot ) } } , \qquad { \hat { P } } ( v \mid u , a ) = { \frac { \operatorname { c o u n t } ( u \stackrel { a } { \to } v ) } { \sum _ { v ^ { \prime } } \operatorname { c o u n t } ( u \stackrel { a } { \to } v ^ { \prime } ) } }\tag{6}
$$

Theoretical connection. The action-level Bellman equation is a direct empirical approximation of the standard value function $( \mathrm { E q . } ( 1 ) )$ . Under sparse reward settings (rewards are assigned only at sink nodes), the state-value satisfies $V ( s _ { t } ) = \bar { \mathbb { E } } _ { a _ { t } } [ Q ( s _ { t } , a _ { t } ) ]$ and $\bar { Q ( s _ { t } , a _ { t } ) } = \mathbb { E } _ { s _ { t + 1 } } [ \gamma \bar { V } ( s _ { t + 1 } ) ]$ ] (see Sec.2). Combining these yields:

$$
V ( s _ { t } ) = \mathbb { E } _ { a _ { t } } \left[ \mathbb { E } _ { s _ { t + 1 } } [ \gamma V ( s _ { t + 1 } ) ] \right] = \sum _ { a \in A ( s _ { t } ) } { \hat { \pi } } ( a \mid u ) \sum _ { v } { \hat { P } } ( v \mid s _ { t } , a ) \cdot \gamma \cdot V ( v )\tag{7}
$$

which is precisely the action-level update rule defined in Eq.(5).

Iterative solver. We solve the Bellman equation via Gauss-Seidel value iteration (Wikipedia, 2006). To accelerate convergence, we determine the updating order by computing the reverse-BFS distance from each node to the nearest sink node: nodes closer to a sink node are updated earlier, so information propagates from sinks back to source nodes in as few sweeps as possible. Nodes that cannot reach any sink node retain $V ( u ) = 0$ , which constitutes the correct fixed point under sparse rewards. The iteration terminates when max<sub>u</sub> $| V ^ { ( k + 1 ) } ( u ) - V ^ { ( k ) } ( u ) | < \epsilon \mathrm { ( d e f a u l t } \epsilon = 1 0 ^ { - 8 } \big )$ . Since $\gamma < 1$ , the Bellman operator is a contraction mapping and convergence is guaranteed. The maximum number of iterations is set adaptively as $k _ { \mathrm { m a x } } = \lceil \log ( \epsilon ) / \log ( \gamma ) \rceil + 5 0$ to ensure convergence for any $\gamma \in ( 0 , 1 )$

## 3.3 STEP-LEVEL ADVANTAGE ESTIMATION

Given the converged value estimates $\{ V ( u ) \} _ { u \in \mathcal { V } }$ , we define the step-level advantage for each executed transition $\left( { { s _ { t } } , { a _ { t } } , { s _ { t + 1 } } } \right)$ as:

$$
A _ { t } = \gamma \cdot V ( \phi ( s _ { t + 1 } ) ) - V ( \phi ( s _ { t } ) )\tag{8}
$$

Notably, as analyzed in Sec.2, we know that $A ( s _ { t } , a _ { t } ) = Q ( s _ { t } , a _ { t } ) - V ( s _ { t } ) = r ( s _ { t } ) + \gamma V ( s _ { t + 1 } ) -$ $V ( s _ { t } )$ . In sparse-reward settings where the environment step reward $r ( s _ { t } )$ is zero, the $A _ { t }$ coincides exactly with the standard advantage function definition $( i . e .$ , which is also called TD(0) in RL). Intuitively, $A _ { t } > 0$ indicates that action $a _ { t }$ moves the agent toward a higher-value state (closer to success), and $A _ { t } < 0$ signals a transition toward a lower-value state.

Proposition 1 (Lower Advantage Estimation Error). Given rollouts sampled from a fixed policy π in afinite-horizon deterministic environment with sparse rewards. Define

$$
\Omega = \{ ( s , a ) : 0 < \operatorname* { P r } ( R ( \tau ) > 0 \mid S _ { t } = s , A _ { t } = a ) < 1 \}
$$

as the set ofstate-action pairs that admit both successful andfailed continuations. Let

$$
X _ { t } ^ { G } = r _ { t } + \gamma V ^ { \pi } ( S _ { t + 1 } ) - V ^ { \pi } ( S _ { t } )
$$

denote the graph-based step-level credit, and let

$$
X _ { t } ^ { S } = G _ { t } - V ^ { \pi } ( S _ { t } ) , \qquad X _ { t } ^ { E } = R ( \tau ) - \mathbb { E } _ { \tau \sim \pi } [ R ( \tau ) \mid S _ { 0 } = s _ { 0 } ]
$$

denote the population counterparts of the step-level credit of GiGPO and the trajectory-level credit ofGRPO, respectively. Then

$$
\mathcal { E } ( X ^ { G } ) \leq \operatorname* { m i n } \left\{ \mathcal { E } ( X ^ { S } ) , \mathcal { E } ( X ^ { E } ) \right\}
$$

where $\mathcal { E } ( X ) = \mathbb { E } _ { \tau \sim \pi } \left\lceil \left( X _ { t } - A ^ { \pi } ( S _ { t } , A _ { t } ) \right) ^ { 2 } \right\rceil$ is the population estimation error of a credit signal X.   
Moreover, the inequality is strict whenever Ω has positive visitation probability under π.

The proof of Proposition 1 is provided in Appendix C. This proposition identifies the structural benefit of our graph-based Bellman credit assignment. At the population level, graph-based TD credit replaces the trajectory-specific sampled continuation with the policy-averaged value of the successor state, thereby producing a Bellman-aligned credit signal for the current decision. Thus, the superiority of our GRAFT arises from merging trajectories at shared states and aggregating information from their different continuations via Bellman backups into a lower-error, state-specific credit signal.

![](images/e832c530a499b547b2eb12e181c9bbe36b489eb2fae9686e3d804d5f355bc1c9.jpg)  
Figure 2: Intuitive comparison of advantage estimation methods: trajectory-level advantage in GRPO, state-grouping-based advantage in GiGPO, and our proposed graph-bootstrapped advantage.

To more intuitively illustrate our method’s advantages, in Fig.2, we compare our graph-based advantage estimation with the trajectory-level advantage in GRPO and the state-groupingbased approach in GiGPO. By bootstrapping (Bellman backups) over the graph, our method estimates state values more accurately, thus effectively identifying valuable steps within failed trajectories $( e . g . , \bar { S }  D$ in T ) and erroneous steps within successful ones $( e . g . , S  L$ in $T _ { 1 } )$ In contrast, advantage estimation in GRPO and GiGPO is heavily influenced by terminal outcomes $( i . e .$ , both $\mathbf { \dot { \Gamma } } T _ { 1 } \mathbf { \dot { s } } S  L$ and $T _ { 2 } { ' } s \ S  D$ are affected by the outcome of the belonging trajectory, resulting wrong advantages). Thus,

they fail to properly credit beneficial actions in failed rollouts or penalize detrimental actions in successful ones. In Appendix D.5, we also provide case studies to demonstrate the advantages of our method.

However, since this single-step TD residual $A _ { t }$ relies solely on the state-value difference between two consecutive nodes, any bias in the state-value estimation directly propagates to the advantage estimation. To obtain a more robust advantage estimation, we further take the advantage of Generalized Advantage Estimation (GAE), which is the de facto in RL that can reduce the impact of state-value estimation bias by computing a weighted sum of the subsequent multi-step advantage estimates. Specifically, we propose Graph ${ \mathrm { G A } } { \bar { \mathrm { E } } } ,$ , which extends GAE to the trajectory graph. For each executed transition $\left( { { s _ { t } } , { a _ { t } } , { s _ { t + 1 } } } \right)$ , we define the Graph GAE advantage as:

$$
A _ { t } ^ { \mathrm { G A E } } = \underbrace { \gamma V ( \phi ( s _ { t + 1 } ) ) - V ( \phi ( s _ { t } ) ) } _ { A _ { t } } + \sum _ { k = 1 } ^ { T - t } ( \gamma \lambda ) ^ { k } \cdot \bar { A } _ { t + k }\tag{9}
$$

where $\bar { A } _ { t + k }$ is the mean k-hop TD residual, defined as the average single-step TD residual over all raw edge instances $( u ^ { \prime } , a ^ { \prime } , v ^ { \prime } )$ reachable from $a _ { t }$ in exactly k hops on the graph G:

$$
\bar { A } _ { t + k } = \frac { 1 } { | \mathcal { E } _ { k } ( a _ { t } ) | } \sum _ { ( u ^ { \prime } , a ^ { \prime } , v ^ { \prime } ) \in \mathcal { E } _ { k } ( a _ { t } ) } \left[ \gamma V ( v ^ { \prime } ) - V ( u ^ { \prime } ) \right]\tag{10}
$$

where $\mathcal { E } _ { k } ( a _ { t } )$ denotes the set of all directed edges reachable from $a _ { t }$ in exactly k hops. The parameter $\lambda \in [ 0 , 1 ]$ controls the bias-variance trade-off: when $\lambda = 0 , A _ { t } ^ { \mathrm { G A E } }$ reduces to the one-step estimate $A _ { t }$ with low variance but high bootstrap bias. As λ increases, the effective horizon of the estimator expands, thereby reducing the reliance on intermediate state-value bootstrapping, this mitigates bootstrap bias at the cost of higher variance. Intuitively, if the edges reachable from $a _ { t }$ have overall high advantages, this means that starting from $a _ { t }$ will be easier to reach good states, thereby contributing to increase the advantage estimate for $a _ { t } ;$ Conversely, if the reachable edges yield predominantly low advantages, the advantage of $a _ { t }$ is correspondingly reduced. The characteristic of Graph GAE is that it smoothly interpolates between two extremes—the single-step advantage and the Monte Carlo advantage $\begin{array} { r } { \sum _ { k = 0 } ^ { T - t } \gamma ^ { k } r \dot { ( s _ { t + k } ) } - V ( s _ { t } ) . } \end{array}$ —via λ, achieving a better bias-variance trade-off.

Table 1: Performance on ALFWorld and WebShop. Results are averaged over 3 random seeds. For ALFWorld, we report the average success rate (%) for each subtask as well as the overall result. For WebShop, we report both the average score and the average success rate (%). Best results are bolded.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td></td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>All</td><td>Score</td><td>Succ.</td></tr><tr><td colspan="10">Qwen2.5-1.5B-Instruct</td></tr><tr><td>Prompting</td><td>Qwen2.5</td><td>5.9</td><td>5.5</td><td>3.3</td><td>9.7</td><td>4.2</td><td>0.0</td><td>4.1</td><td>23.1</td><td>5.2</td></tr><tr><td>Prompting</td><td>ReAct</td><td>17.4</td><td>20.5</td><td>15.7</td><td>6.2</td><td>7.7</td><td>2.0</td><td>12.8</td><td>40.1</td><td>11.3</td></tr><tr><td>Prompting</td><td>Reflexion</td><td>35.3</td><td>22.2</td><td>21.7</td><td>13.6</td><td>19.4</td><td>3.7</td><td>21.8</td><td>55.8</td><td>21.9</td></tr><tr><td>RL Training</td><td>PPO</td><td>64.8±3.5</td><td>40.5±6.9</td><td> $5 7 . 1 { \pm } 4 . 9$ </td><td>60.6±6.6</td><td>46.4±4.0</td><td>47.4±1.9</td><td> $5 4 . 4 { \pm } 3 . 1 $ </td><td>73.8±3.0</td><td>51.5±2.9</td></tr><tr><td>RL Training</td><td>RLOO</td><td>88.3±3.0</td><td>52.8±8.6</td><td>71.0±5.9</td><td>62.8±8.7</td><td>66.4±5.5</td><td>56.9±4.7</td><td> $6 9 . 7 { \pm } 2 . 5 $ </td><td>73.9±5.6</td><td>52.1±6.7</td></tr><tr><td>RL Training</td><td>GRPO</td><td>85.3±1.5</td><td> $5 3 . 7 { \pm } 8 . 0 $ </td><td>84.5±6.8</td><td> $7 8 . 2 \pm 7 . 9$ </td><td> $5 9 . 7 { \pm } 5 . 0 $ </td><td> $5 3 . 5 { \pm } 5 . 6 $ </td><td> $7 2 . 1 \pm 3 6$   $7 2 . 8 { \pm } 3 . 6 $ </td><td>75.8±3.5</td><td>56.8±3.8</td></tr><tr><td>RL Training</td><td>GiGPO</td><td> $9 4 . 4 \pm 5 . 9$ </td><td> $6 7 . 5 { \pm } 4 . 6 $ </td><td> $9 4 . 8 { \pm } 3 . 8 $ </td><td> $9 4 . 4 { \pm } 7 . 8 $ </td><td> $7 9 . 8 { \pm } 4 . 7 $ </td><td> $7 6 . 4 \pm 5 . 4$ </td><td> $8 6 . 7 { \pm } 1 . 7$ </td><td>83.1±1.6</td><td>65.0±3.2</td></tr><tr><td>RL Training</td><td>SALT</td><td> $9 6 . 2 { \pm } 1 . 7 $ </td><td> $6 5 . 2 { \pm } 1 0 . 8 $ </td><td> ${ \overline { { 9 3 . 1 } } } \pm 4 . 7$ </td><td>81.8±8.3</td><td> $8 5 . 0 { \pm } 6 . 9$ </td><td> $7 7 . 0 { \pm } 4 . 7 $ </td><td> $8 5 . 2 { \pm } 2 . 5 $ </td><td>86.9±0.6</td><td>74.7±2.4</td></tr><tr><td>RL Training</td><td>GraphGPO</td><td> $9 5 . 2 { \pm } 1 . 6 $ </td><td> $8 5 . 7 { \pm } 5 . 8 $ </td><td> ${ \bf 1 0 0 . 0 { \pm } 0 . 0 }$ </td><td> $9 6 . 3 { \pm } 2 . 6 $ </td><td> $8 5 . 3 { \pm } 2 . 6 $ </td><td> $9 3 . 7 { \pm } 2 . 2 $ </td><td> $9 2 . 7 { \pm } 1 . 3 $ </td><td>89.3±1.5</td><td>78.7±3.9</td></tr><tr><td>RL Training</td><td>GRAFT (Ours)</td><td>99.2±1.2</td><td>98.3±2.4</td><td>100.0±0.0</td><td>100.0±0.0</td><td>93.3±3.7</td><td>97.6±3.3</td><td>97.4±0.4</td><td>90.8±0.4</td><td>82.3±1.0</td></tr><tr><td colspan="10">Qwen2.5-7B-Instruct</td></tr><tr><td>Prompting</td><td>Qwen2.5</td><td>33.4</td><td>21.6</td><td>19.3</td><td>6.9</td><td>2.8</td><td>3.2</td><td>14.8</td><td>26.4</td><td>7.8</td></tr><tr><td>Prompting</td><td>ReAct</td><td>48.5</td><td>35.4</td><td>34.3</td><td>13.2</td><td>18.2</td><td>17.6</td><td>31.2</td><td>46.2</td><td>19.5</td></tr><tr><td>Prompting</td><td>Reflexion</td><td>62.0</td><td>41.6</td><td>44.9</td><td>30.9</td><td>36.3</td><td>23.8</td><td>42.7</td><td>58.1</td><td>28.8</td></tr><tr><td>RL Training</td><td>PPO</td><td>92.3±4.0</td><td>64.0±8.4</td><td>92.5±2.4</td><td>89.5±7.0</td><td>80.3±2.0</td><td>68.8±8.3</td><td>80.4±2.7</td><td>81.4±3.1</td><td>68.7±5.1</td></tr><tr><td>RL Training</td><td>RLOO</td><td>87.6±4.3</td><td>78.2±8.3</td><td>87.3±5.8</td><td>81.3±7.6</td><td>71.9±5.2</td><td>48.9±8.4</td><td> $7 5 . 5 { \pm } 4 . 6 $ </td><td>80.3±3.2</td><td>65.7±4.0</td></tr><tr><td>RL Training</td><td>GRPO</td><td>90.8±5.1</td><td>66.1±6.7</td><td>89.3±5.4</td><td>74.7±6.9</td><td>72.5±5.4</td><td>64.7±7.3</td><td> $7 7 . 6 { \pm } 5 . 2 $ </td><td>79.3±2.8</td><td>66.1±3.7</td></tr><tr><td>RL Training</td><td>GiGPO</td><td>97.7±1.6</td><td>82.7±7.9</td><td> $9 8 . 8 { \pm } 1 . 6 $ </td><td>83.7±7.2</td><td>89.3±8.2</td><td> $7 9 . 2 { \pm } 6 . 6 $ </td><td> $9 0 . 8 { \pm } 1 . 3 $ </td><td>84.4±2.9</td><td>72.8±3.2</td></tr><tr><td>RL Training</td><td>SALT</td><td>93.4±2.1</td><td>72.6±7.5</td><td> $9 1 . 5 { \pm } 3 . 2 $ </td><td>90.3±3.5</td><td> $7 8 . 1 \pm 2 . 7$ </td><td> $7 6 . 4 \pm 4 . 4$ </td><td> $8 7 . 3 { \pm } 4 . 4 $ </td><td>83.1±3.8</td><td>75.2±5.5</td></tr><tr><td>RL Training</td><td>GraphGPO</td><td>100.0±0.0</td><td>92.9±5.8</td><td> ${ \bf 1 0 0 . 0 { \pm } 0 . 0 }$ </td><td> $9 4 . 4 { \pm } 0 . 0 $ </td><td> $9 1 . 4 { \pm } 1 . 5 $ </td><td> $9 2 . 1 { \pm } 2 . 2 $ </td><td> $9 5 . 3 { \pm } 1 . 1 $ </td><td>86.9±0.7</td><td>80.3±1.3</td></tr><tr><td>RL Training</td><td>GRAFT (Ours)</td><td>99.0±1.5</td><td>100.0±0.0</td><td>98.4±2.3</td><td>100.0±0.0</td><td> ${ \bf 9 7 . 2 } { \pm } 2 . 9$ </td><td>96.8±2.6</td><td>98.4±0.6</td><td>91.5±1.3</td><td>82.8±0.6</td></tr></table>

Advantage normalization. Although $A _ { t } ^ { G A E }$ itself can serve as a robust step-level advantage estimate, to further eliminate scale sensitivity, we additionally apply advantage normalization within each node’s on-state group (please see the experiments in Tab.3). For each node u, all outgoing edges $( u , a , v )$ form a natural step group corresponding to the set of actions sampled from the same state $\dot { s } = \phi ^ { - 1 } ( u )$ . Within this group, we apply advantage normalization:

$$
\hat { A } _ { \mathrm { s t e p } } ( s _ { t } , a _ { t } ) = \frac { A _ { t } ^ { G A E } - \mu _ { \mathcal { G } ( s _ { t } ) } } { \sigma _ { \mathcal { G } ( s _ { t } ) } + \epsilon }\tag{11}
$$

where $\mu _ { \mathcal { G } ( s _ { t } ) }$ and $\sigma _ { \mathscr { G } ( s _ { t } ) }$ are the mean and standard deviation of step advantages within the step group $\mathcal { G } ( s _ { t } ) = \{ A _ { t } ^ { G A E } : ( s _ { t } , \cdot , \cdot ) \in \mathcal { E } \}$ . For groups containing only one edge, to avoid yielding a normalized advantage of zero, we perform normalization by setting the mean to 0 and using the absolute value as the standard deviation. The whole GRAFT algorithm is in Alg. 1 in Appendix B.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmarks. Following the agentic RL baseline GiGPO (Feng et al., 2025), we evaluate our method on a suite of challenging multi-turn agentic benchmarks, including ALFWorld (Shridhar et al., 2021), WebShop (Yao et al., 2022a), and SearchQA (Jin et al., 2025). The introduction of these benchmarks is deferred to Appendix E. In SearchQA, the training set is drawn from NQ and HotpotQA, making these two datasets for in-domain evaluation, while the remaining datasets are used to assess out-of-domain generalization.

Baselines. To ensure a standardized and rigorous comparison, we adopt the baseline results directly from the original GiGPO (Feng et al., 2025) and GraphGPO (Cheng et al., 2026) papers. For ALFWorld and WebShop, the baselines include: (1) Prompting-based agents: ReAct (Yao et al., 2022b) and Reflexion (Shinn et al., 2023); and (2) RL training methods: PPO (Schulman et al., 2017), a widely used critic-based RL algorithm, group-based RL methods including RLOO (Ahmadian et al., 2024), GRPO (Shao et al., 2024), and GiGPO (Feng et al., 2025), as well as graph-based policy optimization methods, SALT (Li et al., 2025a), GraphGPO (Cheng et al., 2026) (In appendix A.2, A.3, we provide detailed discussions about the differences between our method and these two methods). For searchQA tasks, we also follow the experimental protocol in GiGPO (Feng et al., 2025) and compare GRAFT against a specific suite of baselines including R1-Instruct, Search-R1 (Jin et al., 2025), ZeroSearch (Sun et al., 2025), StepSearch (Wang et al., 2025b) and the agentic RL methods: GiGPO, GraphGPO.

Implementation Details. To ensure a direct and fair comparison with the prior methods, we utilize Qwen2.5-Instruct series (1.5B, 3B, 7B) as our base models (the compared methods report results based on Qwen2.5-Instruct series). For all benchmarks, we follow GiGPO’s training and evaluation configurations as our basic configurations, including rollout batch size, group size, mini-batch size, learning rate, and sampling temperature, etc. Additionally, the specific hyperparameters of our method are: the value discount factor $\gamma = 0 . 9 9$ , the advantage weighting factor λ = 0.95 for ALFWorld, 0.8 for WebShop, and 0.6 for SearchQA. Further implementation details are provided in Appendix E.

## 4.2 MAIN RESULTS

As shown in Tab.1, GRAFT achieves significant gains over the trajectory-level baseline GRPO and demonstrates superior performance compared to the state-of-the-art GiGPO and GraphGPO across both ALFWorld and WebShop. On ALFWorld, GRAFT achieves improvements on nearly all subtasks, resulting in average success rate gains of 24.6% and 20.8% over GRPO for the 1.5B and 7B models, respectively. On WebShop, GRAFT not only attains higher task scores but also improves the average success rate over GRPO by 25.5% and 16.7% for the 1.5B and 7B models, respectively. These results highlight that our method effectively overcomes the limitations of coarse-grained trajectory-level advantages. In addition, GRAFT also surpasses both step-group-based GiGPO and graph-based GraphGPO by 10.7%/17.3% and 4.7%/3.6% on ALFWorld and WebShop, respectively, both of which are step-level credit assignment algorithms. As the advantage estimation in our method closely revolves around the standard definition, compared to heuristic advantage estimation approaches in GiGPO, SALT, and GraphGPO, our method can provide more credible step-level advantage estimates.

Tab.2 presents the results on searchQA tasks. We observe that GRAFT achieves strong and consistent gains across both single-hop and multi-hop reasoning datasets. Notably, GRAFT reaches an average success rate of 48.6% at 7B, outperforming prior baselines such as Search-R1 and StepSearch, and also outperforms the strong step-level credit assignment methods, GiGPO and GraphGPO. In Appendix D.4, to demonstrate the generalization of our method, we further provide results on the Qwen3 model series.

Table 2: Performance on searchQA tasks. † and ⋆ indicate in-domain and out-of-domain datasets, respectively. Bold indicates the best performance in each category.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="3">Single-Hop QA</td><td colspan="4">Multi-Hop QA</td><td rowspan="2">Avg.</td></tr><tr><td>NQ†</td><td>TriviaQA*</td><td>PopQA*</td><td>HotpotQA†</td><td>2Wiki*</td><td>MuSiQue*</td><td>Bamboogle*</td></tr><tr><td colspan="10">Qwen2.5-3B-Instruct</td></tr><tr><td>RL Training</td><td>R1-Instruct</td><td>27.0</td><td>53.7</td><td>19.9</td><td>23.7</td><td>29.2</td><td>7.2</td><td>29.3</td><td>27.1</td></tr><tr><td>RL Training</td><td>Search-R1</td><td>34.1</td><td>54.5</td><td>37.8</td><td>32.4</td><td>31.9</td><td>10.3</td><td>26.4</td><td>32.5</td></tr><tr><td>RL Training</td><td>ZeroSearch</td><td>41.4</td><td>57.4</td><td>44.8</td><td>27.4</td><td>30.0</td><td>9.8</td><td>11.1</td><td>31.7</td></tr><tr><td>RL Training</td><td>StepSearch</td><td></td><td></td><td></td><td>34.5</td><td>32.0</td><td>17.4</td><td></td><td>34.4</td></tr><tr><td>RL Training</td><td>GiGPO</td><td>42.0</td><td>59.5</td><td>42.4</td><td>36.9</td><td>37.0</td><td>12.6</td><td>64.1</td><td>42.1</td></tr><tr><td>RL Training</td><td>GraphGPO</td><td>44.5</td><td>59.7</td><td>46.2</td><td>36.2</td><td>37.2</td><td>12.1</td><td>63.7</td><td>44.0</td></tr><tr><td>RL Training</td><td>GRAFT (Ours)</td><td>45.3</td><td>61.8</td><td>46.4</td><td>37.2</td><td>37.3</td><td>12.4</td><td>64.9</td><td>45.2</td></tr><tr><td colspan="10">Qwen2.5-7B-Instruct</td></tr><tr><td>RL Training</td><td>R1-Instruct</td><td>21.0</td><td>44.9</td><td>17.1</td><td>20.8</td><td>27.5</td><td>6.0</td><td>19.2</td><td>22.4</td></tr><tr><td>RL Training</td><td>Search-R1</td><td>39.3</td><td>61.0</td><td>39.7</td><td>37.0</td><td>40.1</td><td>14.6</td><td>36.8</td><td>38.5</td></tr><tr><td>RL Training</td><td>ZeroSearch</td><td>43.6</td><td>61.8</td><td>51.5</td><td>34.6</td><td>35.2</td><td>18.4</td><td>27.8</td><td>39.1</td></tr><tr><td>RL Training</td><td>StepSearch</td><td></td><td></td><td></td><td>38.6</td><td>36.6</td><td>22.6</td><td></td><td>40.0</td></tr><tr><td>RL Training</td><td>GiGPO</td><td>46.4</td><td>64.7</td><td>46.1</td><td>41.6</td><td>43.6</td><td>18.9</td><td>68.9</td><td>47.2</td></tr><tr><td>RL Training</td><td>GraphGPO</td><td>46.8</td><td>65.4</td><td>47.7</td><td>41.1</td><td>42.7</td><td>17.0</td><td>69.0</td><td>48.0</td></tr><tr><td>RL Training</td><td>GRAFT (Ours)</td><td>46.9</td><td>65.6</td><td>47.8</td><td>41.9</td><td>43.4</td><td>17.7</td><td>70.2</td><td>48.6</td></tr></table>

## 4.3 ABLATION STUDIES & FURTHER ANALYSIS

Tab. 3 isolates the effects of the key components of our GRAFT. We start with a baseline $\mathrm { { G R A F T ^ { \dagger } } }$ that uses the step-level advantage defined in Eq.(8). Removing advantage normalization suffers a substantial performance drop across both benchmarks (e.g., ALFWorld’s “All” declines from 96.10 to 71.33, and WebShop success rate falls to 72.93), demonstrating that normalization is critical for stabilizing gradient updates and preventing scale-induced optimization instability. When we replace the original GRPO objective with the GRPO-C objective (Eq.(4)), performance surpasses the baseline across nearly all metrics. This confirms that aligning the optimization granularity with the advantage estimator (treating the entire step as a single action to match the step-level advantage) is superior to $\mathrm { G R P O ^ { \circ } s }$ mismatched token-level assignment. Integrating Graph GAE further delivers consistent gains, analogous to the bias-variance trade-off in standard GAE, our Graph GAE effectively fuses multiple k-step estimators to produce a more faithful and robust advantage estimate than the single-step estimator, leading to improved policy optimization. The complete GRAFT framework achieves the highest performance, confirming that these components are not independent but can synergistically enhance policy learning in long-horizon agentic tasks.

Table 3: Ablation study results. “w/o adv normalization” denotes directly using the step-level advantage defined in Eq.(8) without further advantage normalization. $\mathrm { { G R A F T } ^ { \dagger } }$ is a variant of GRAFT without GRPO-C objective (Eq.(4)) and Graph GAE, serving as the baseline for ablation studies. The results are based on Qwen2.5-1.5B-Instruct.
<table><tr><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">Webshop</td></tr><tr><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>All</td><td>Score</td><td>Succ.</td></tr><tr><td>GRAFT†</td><td>99.23±1.08</td><td>100.0±0.00</td><td>96.73±2.31</td><td>97.23±3.15</td><td>92.57±5.35</td><td>89.60±2.46</td><td>96.10±1.13</td><td>89.53±0.76</td><td>80.67±1.68</td></tr><tr><td>w/o adv normalization</td><td>75.37±1.51</td><td>75.47±3.53</td><td>72.50±4.76</td><td>63.10±3.53</td><td>72.20±5.67</td><td>60.90±4.54</td><td>71.33±2.87</td><td>85.67±0.74</td><td>72.93±1.46</td></tr><tr><td>w/ GRPO-C objective (4)</td><td>100.0±0.00</td><td>98.03±2.78</td><td>96.00±3.39</td><td>100.0±0.00</td><td>96.20±2.53</td><td>90.73±3.78</td><td>96.87±1.08</td><td>90.63±1.51</td><td>81.30±1.31</td></tr><tr><td>w/ Graph GAE (9)</td><td> $9 7 . 5 0 { \pm } 2 . 1 8 $ </td><td>100.0±0.00</td><td>100.0±0.00</td><td>100.0±0.00</td><td>97.10±2.11</td><td>91.53±4.13</td><td>97.17±0.75</td><td>90.23±0.70</td><td>81.73±0.75</td></tr><tr><td>GRAFT</td><td> $9 9 . 1 7 { \pm } 1 . 1 7$ </td><td>98.33±2.35</td><td>100.0±0.00</td><td>100.0±0.00</td><td>93.33±3.71</td><td>97.63±3.34</td><td>97.43±0.38</td><td>90.83±0.37</td><td>82.27±0.99</td></tr></table>

Task Efficiency Analysis. In Fig.3, we further illustrate the average number of interaction turns required by the trained models to complete tasks on WebShop and ALF-World. Across both benchmarks, GRAFT-trained models consistently require fewer turns than those trained with GRPO or GiGPO. This indicates that our method can yield more faithful step-level advantage estimates, which enable the agent to avoid redundant actions and make more efficient decisions. Fewer interaction turns inherently translate to lower model inference costs and reduced environment API calls. Together with Tab.1, these results demonstrate that our GRAFT achieves substantial advantages over existing methods in both task accuracy and cost efficiency.

![](images/9047cc0349770de0cd39df6bb2123f4d87a6b5b65ff564cdaaf8f2f43dd55a77.jpg)  
Figure 3: The average number of execution turns required to finish tasks.

In appendix D, we further provide additional experimental analysis, including training dynamics, hyperparameter experiments, method mechanism analysis, additional results, and case studies.

Computional Cost. Our method incurs no additional GPU memory overhead, as we have not introduced any extra models or modules. Due to inconvenience in comparing FLOPs between advantage estimation and model computation, we analyze the time consumption per training step. We use Qwen2.5-1.5B-Instruct as the base model and train it on the ALFWorld benchmark. The per-step runtime breakdown is shown in Fig.4. As illustrated, a typical training step involves a rollout stage, advantage estimation, probability computation for sampled responses under both the reference and old policies, and policy update. Our method operates specifically within the advantage estimation stage. The additional costs introduced by GRAFT stem from graph construction and graph-based advantage estimation. These costs are negligible compared to rollout and policy update, occupying merely 0.43% of the total RL training time and introducing virtually no extra computational burden.

![](images/3153f4c0c55a468c18f8ff8bfd21ab500ea84e88a87a0877cdbc0a8da9b5602d.jpg)  
Figure 4: Breakdown of time consumption per training step.

## 5 RELATED WORK

Reinforcement Learning for LLMs. RL for LLMs has evolved from preference alignment to reasoning enhancement. Early PPO-based RLHF methods (Schulman et al., 2017; Bai et al., 2022; Ouyang et al., 2022), while effective, were computationally burdensome due to the need for separate reward and value models, prompting a shift toward alternatives like DPO (Rafailov et al., 2023) and its variants (e.g., SimPO (Meng et al., 2024), KTO (Ethayarajh et al., 2024), ORPO (Hong et al., 2024)). However, the offline characteristics of the DPO series result in inherently performancebounded compared to online methods. The recent success of DeepSeek-R1 (DeepSeek-AI, 2025) established RL with Verifiable Rewards (RLVR) as a new paradigm, demonstrating that strong reasoning can emerge via online outcome-based algorithms (e.g., GRPO (Shao et al., 2024), DAPO (Seed, 2025), GSPO (Zheng et al., 2025)) without SFT cold-start or process supervision. Although such group-based methods enable scalable, memory-efficient training by eliminating critic models, they predominantly rely on sparse outcome rewards and suffer from the credit assignment problem at the step level. While Process Reward Models (Cui et al., 2025; Cao et al., 2025; Xi et al., 2025) offer finer-grained guidance, they incur prohibitive annotation costs and are prone to reward hacking. Consequently, recent efforts (Feng et al., 2025; Tan et al., 2026; Cheng et al., 2026) have pivoted on deriving dense, multi-granularity advantages directly from sampling statistics or state transitions, aiming to achieve precise credit assignment without the overhead of learned critics or expensive process annotations.

Agentic Reinforcement Learning. RL has become a cornerstone for empowering LLM agents in dynamic, open-ended environments (Zhang et al., 2025; Wang et al., 2025b; Team, 2025), evolving from early value-based methods (Peng et al., 2019; Zha et al., 2021) to modern policy gradient approaches for complex tasks like web navigation (Shi et al., 2025), tool use (Jiang et al., 2025), and software engineering (Da et al., 2025). Recent works such as Search-R1 (Jin et al., 2025) and WebSailor (Li et al., 2025b) demonstrate that agentic RL can effectively instill multi-step information-seeking and long-horizon planning capabilities without human-curated supervision. Despite these advances, training efficacy remains limited by sparse outcome-based rewards, which hinder the agent’s ability to correct intricate intermediate errors. To address this credit assignment problem, Process Reward Models offer step-level supervision but incur prohibitive annotation costs. Alternatively, intrinsic reward mechanisms such as EMPG (Wang et al., 2025a) utilize dynamic entropy to encourage exploration. More recently, GiGPO (Feng et al., 2025) introduces state-based anchoring to aggregate all states into step-wise groups for normalized advantage estimation in the aggregated groups. HCAPO (Tan et al., 2026) proposes to leverage the LLM itself as a post-hoc critic to refine step-level credit through hindsight reasoning. A recent work, GraphGPO (Cheng et al., 2026), employs the graph structure to model the state-action transition relationship and achieves step-level credit assignment via heuristic reward shaping. Although both use the graph structure, our method differs fundamentally from GraphGPO in motivation, implementation, and effectiveness; we provide a detailed comparative discussion in Appendix A.2.

## 6 CONCLUSION

In this work, we revisit the basic advantage estimation problem and group-based RL, and identify a critical gap: GRPO-style advantage estimation is reliable at the response level, but systematically biased when extending the group-based estimation strategy to the step-level. To address this, we propose GRAFT, a graph-based credit-assignment framework that grafts all rollout trajectories into a unified trajectory graph, recovers node state-values via Bellman iteration on the graph, and assigns credit to each edge by the node value difference. The resulting step-level advantage is not a heuristic surrogate but faithfully adheres to the basic advantage definition in RL, and asymptotically converges to the true target as the graph reaches sufficiently saturated. Across diverse multi-turn agentic benchmarks, GRAFT consistently outperforms GRPO and demonstrates superior performance compared to recent SOTA agentic RL algorithms, while remaining critic-free and PRM-free.

## ACKNOWLEDGMENTS

This work was completed during the internship at Tencent AI Platform Department. We gratefully acknowledge Tencent Inc. for providing computing resources that greatly supported the completion of this work. This work was also supported in part by the National Natural Science Fund of China (No.62371295), the Shanghai Jiao Tong University AI for Engineering Initiative (No.WH410263001/001), and the Science and Technology Commission of Shanghai Municipality (No.22DZ2229005).

## REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin,´ Ahmet Ust <sup>¨</sup> un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning¨ from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, and et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Lang Cao, Renhong Chen, Yingtian Zou, Chao Peng, Huacong Xua, Yuxian Wang, Wu Ning, Qian Chen, Mofan Peng, Zijie Chen, and et al. More bang for the buck: Process reward modeling with entropy-driven uncertainty. arXiv preprint arXiv:2503.22233, 2025.

Xin Cheng, Shuo He, Lang Feng, HaiYang Xu, Ming Yan, Lei Feng, and Bo An. Beyond trajectorylevel attribution: Graph-based credit assignment for agentic reinforcement learning. In ICML, 2026.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Jeff Da, Clinton Wang, Xiang Deng, Yuntao Ma, Nikhil Barhate, and Sean Hendryx. Agent-rlvr: Training software engineering agents via guidance and environment rewards. arXiv preprint arXiv:2506.11425, 2025.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. In ICML, 2024.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. In NeurIPS, 2025.

Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization. arXiv preprint arXiv: 2511.20347, 2025.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings ofthe 28th International Conference on Computational Linguistics, 2020.

Jiwoo Hong, Noah Lee, and James Thornen. Orpo: Monolithic preference optimization without reference model. arXiv preprint arXiv: 2403.07691, 2024.

Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, Tianyu Pang, and Wenhu Chen. Verltool: Towards holistic agentic reinforcement learning with tool use. arXiv preprint arXiv:2509.01055, 2025.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Mandar Joshi, Eunsol Choi, Daniel SWeld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting ofthe Associationfor Computational Linguistics, 2017.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and et al. Natural questions: a benchmark for question answering research. Transactions ofthe Associationfor Computational Linguistics, 2019.

Jiazheng Li, Yawei Wang, David Yan, Yijun Tian, Zhichao Xu, Huan Song, Panpan Xu, and Lin Lee Cheong. Salt: Step-level advantage assignment for long-horizon agents via trajectory graph. arXiv preprint arXiv:2510.20022, 2025a.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025b.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st annual meeting of the association for computational linguistics, 2023.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a referencefree reward. In NeurIPS, 2024.

OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and et al. Training language models to follow instructions with human feedback. In NeurIPS, 2022.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv: 1707.06347, 2017.

ByteDance Seed. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv: 2503.14476, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yucheng Shi, Wenhao Yu, Zaitang Li, Yonglin Wang, Hongming Zhang, Ninghao Liu, Haitao Mi, and Dong Yu. Mobilegui-rl: Advancing mobile gui agent through reinforcement learning in online environment. arXiv preprint arXiv:2507.05720, 2025.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In NeurIPS, 2023.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk´ ‡ Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In ICLR, 2021.

Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms without searching. arXiv preprint arXiv:2505.04588, 2025.

Huize Tan, Xiaowen Yang, Hao Chen, Jiejing Shao, Yi Wen, Yuteng Shen, Weihong Luo, Xiku Du, Lanzhe Guo, and Yufeng Li. Hindsight credit assignment for long-horizon llm agents. arXiv preprint arXiv:2603.08754, 2026.

Kimi Team. Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026a.

Qwen Team. Qwen-agentworld: Language world models for general agents. arXiv preprint arXiv:2606.24597, 2026b.

Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents. 2026c. URL https://qwen.ai/blog?id=qwen3.5.

Tongyi DeepResearch Team. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multi-hop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 2022.

Jiawei Wang, Jiacai Li, Yuqian Fu, Yingru Li, Xintao Wang, Yuan Lin, Yu Yue, Lin Zhang, Yang Wang, and Ke Wang. Harnessing uncertainty: Entropy-modulated policy gradients for long-horizon llm agents. arXiv preprint arXiv:2509.09265, 2025a.

Tao Wang, Suhang Zheng, and Xiaoxiao Xu. Rtmc: Step-level credit assignment via rollout trees. arXiv preprint arXiv:2604.11037, 2026.

Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, and Yichao Wu. Stepsearch: Igniting llms search ability via step-wise proximal policy optimization. arXiv preprint arXiv:2505.15107, 2025b.

Wikipedia. Gauss–seidel method. https://en.wikipedia.org/wiki/Gauss-Seidel method, 2006.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn, 1992.

Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, and Xuanjing Huang. Agentprm: Process reward models for llm agents via step-wise promise and progress. arXiv preprint arXiv:2511.08325, 2025.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, 2018.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. In NeurIPS, 2022a.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICLR, 2022b.

Xincheng Yao, Ruoqi Li, Cheng Chen, Daoxin Zhang, Yi Wu, and Yao Hu. Htpo: Towards exploration-exploitation balanced policy optimization via hierarchical token-level objective control. arXiv preprint arXiv:2605.08283, 2026.

Daochen Zha, Jingru Xie, Wenye Ma, Sheng Zhang, Xiangru Lian, Xia Hu, and Ji Liu. Douzero: Mastering doudizhu with self-play deep reinforcement learning. In ICML, 2021.

Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou, Zehan Qi, Hanyu Lai, Yifan Xu, and et al. Agentrl: Scaling agentic reinforcement learning with a multi-turn, multi-task framework. arXiv preprint arXiv:2510.04206, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

## APPENDIX

## A MORE DISCUSSIONS

## A.1 ADVANTAGES OVER GIGPO

As stated in Sec.3, GiGPO (Feng et al., 2025) has proposed to estimate step-level advantages by grouping steps that share the same state and normalizing their discounted Monte Carlo returns $\begin{array} { r } { \dot { R } _ { t } = \dot { \sum _ { l > 0 } \gamma ^ { l } } r _ { t + l } } \end{array}$ within each group. Since our method relies on aggregating cross-trajectory steps into graph nodes, we discuss the advantages of our method over the plain state-grouped advantage estimation approach in GiGPO.

The design in GiGPO has a fundamental limitation: the value signal for state s is derived only from trajectories that pass through s, making the estimate high-variance and sample-inefficient, especially for states visited infrequently. Our method addresses this by constructing an empirical transition graph $\mathcal { G }$ over the rollout group and estimating $V ( s )$ via Bellman value iteration (see Sec.3). The step advantage for each executed edge $( s \stackrel { a } {  } s ^ { \prime } )$ is then defined as $A ( s , a , s ^ { \prime } ) = \gamma V ( s ^ { \prime } ) - V ( s )$ which exactly aligns with the standard advantage function in RL. This formulation offers two key advantages over GiGPO’s vanilla state-grouping approach:

1. Broader information aggregation. Because $V ( s )$ is computed by backward propagation through the graph, it implicitly aggregates reward signals from all trajectories reachable from the successors of s, rather than only those passing through s itself. Concretely, if s branches into $s _ { 1 } ^ { \prime }$ and $s _ { 2 } ^ { \prime }$ , and these successors are collectively visited by K downstream trajectories, then $V ( s )$ incorporates rewards from all K trajectories, whereas GiGPO’s all $R _ { t }$ for s are confined to the trajectories that directly visit s. In Appendix.D.3, we compare the information propagation degree (the number of distinct trajectories contributing reward information to the value estimation of a node (or state)) between our method and GiGPO. The results show that our method consistently achieves a higher information propagation degree than GiGPO throughout training.

2. Theoretical soundness. The step advantage $\gamma V ( s ^ { \prime } ) - V ( s )$ is precisely aligned with the standard definition of the advantage function $A ( s , a ) = Q ( s , a ) - V ( s )$ , since under a deterministic transition $Q ( s , a ) = \gamma V ( s ^ { \prime } )$ . This theoretical grounding ensures that the step-level signal faithfully measures how much better action a is relative to the expected value of state s, whereas GiGPO’s Monte Carlo return $R _ { t }$ introduces high variance from stochastic future trajectories.

3. GiGPO can be viewed as a special case of our method in which each trajectory is treated as an independent graph, and Bellman iteration is performed separately on these isolated graphs to estimate node values. Subsequently, step-level groups are constructed across trajectories, and values are normalized within each group to obtain step-level advantages. Compared to graph-based value estimation, single-trajectory value estimation is inherently less accurate due to the stochasticity of individual trajectories, resulting in higher variance.

Together, these properties make our graph-based estimator both more sample-efficient and more theoretically grounded than GiGPO’s state-grouping baseline. As for performance, the results in Tab.1 and 2 show that our method can achieve significantly better results than GiGPO, validating the superiority of our method over GiGPO in estimating step-level advantages.

## A.2 DISCUSSION WITH GRAPHGPO

In GraphGPO (Cheng et al., 2026), the authors also utilized the graph structure to model state-action transitions across trajectories, which coincides with the idea in our work. Therefore, it is necessary to clarify the distinctions between our work and GraphGPO. Although both methods use the graph structure, our method differs fundamentally from GraphGPO in motivation, implementation, and performance. We elaborate on these differences below:

1. In terms of motivation, GraphGPO is driven by empirical observations: it finds that 65% of steps in successful trajectories do not meaningfully advance the task, while 22% of steps in failed trajectories contribute to task progress. Arguing that trajectory-level credit attribution is too coarse, GraphGPO aggregates all rollout trajectories into a state transition graph to leverage global connectivity for finer-grained credit assignment. In contrast, our motivation stems from the preliminary analysis in Sec.2, which demonstrates that extending GRPO-style advantage estimation to the step level introduces systematic bias. To obtain theoretically grounded estimates while not incurring prohibitive Monte Carlo sampling costs, we should aggregate similar states across trajectories to better leverage global information. To this end, the graph structure serves as a natural representation for modeling the cross-trajectory state-action transition relationship.

2. In terms of implementation, GraphGPO adopts a heuristic approach that defines the step-level rewards based on the shortest-path distance from each node to any successful nodes $r ( s , a , s ^ { \prime } ) =$ $r _ { s u c c } w ^ { d ( s ) }$ , where $d ( s )$ is the shortest distance from s to any successful states and w $\in \left( 0 , 1 \right)$ is a distance discount factor. Then, the rewards are normalized within each node group to obtain step-level advantages. This heuristic reward assignment approach is based on the assumption that beneficial actions can reduce the distance to the task goal. In contrast, our method is grounded in fundamental RL theory. We treat the constructed graph as an empirical Markov Decision Process (MDP) and apply Bellman iteration $\begin{array} { r } { ( V ( u ) = \sum _ { a \in \mathcal { A } ( u ) } \hat { \pi } ( a \mid u ) \sum _ { v \in \mathcal { V } } \hat { P } ( v \mid u , a ) { \cdot } \gamma \cdot V ( v ) ) } \end{array}$ to estimate the state-value of each node. The step-level advantage between two nodes is then defined as $A _ { t } = \gamma \cdot V ( u ) - V ( v )$ Theoretically, the estimated node values conform to the standard definition of state-value in RL (see Eq.(7)), and the derived step-level advantages coincide exactly with the standard advantage function (see Sec.3.3). Rather than relying on hand-crafted heuristics, our approach is principled and theory-driven.

3. In GraphGPO, after obtaining the step-level advantages, these values are uniformly allocated to individual tokens within each step and the GRPO optimization objective is applied. However, as our analysis in Sec.2 demonstrates, this token-level allocation introduces systematic bias. To this end, we employ the calibrated GRPO optimization objective defined in Eq.(4) in our method, which can further outperform the original objective formulation as shown in Tab.3.

4. As for method performance, Tab.1 and 2 both demonstrate that our method consistently outperforms GraphGPO across multiple multi-turn agentic benchmarks, even though both methods are based on graph representation. This indicates that our advantage estimation method, grounded in standard RL definitions, yields more accurate step-level advantages than GraphGPO’s heuristic approach. Furthermore, we provide theoretical guarantees showing that as the graph becomes saturated, the estimated step-level advantages in our method converge to the true advantages.

## A.3 DISCUSSION WITH SALT

In SALT (Li et al., 2025a), the authors also claimed to propose a step-level advantage assignment method based on the trajectory graph. However, their approach underutilizes the graph structure, which actually can be reduced to the plain state aggregation method akin to GiGPO. Specifically, SALT first computes trajectory-level advantages via GRPO, then identifies identical transitions $( s , a , s ^ { \prime } )$ across trajectories into a group and averages their inherited advantages (i.e., the trajectorylevel advantages) within the group. This mechanism mirrors the state-based grouping in GiGPO, but differently SALT groups actions (along with their adjacent states) and assigns the group-averaged advantage to each action. Consequently, SALT inherits GiGPO’s fundamental limitations as discussed in Appendix A.1: information propagation also remains limited. That is, advantage averaging within each group solely utilizes trajectory-level signals from the corresponding trajectories containing the specific action. Although this heuristic advantage adjustment method can mitigate trajectory-level noise (e.g., if a good action receives a misleading trajectory-level advantage due to subsequent failures, averaging across multiple trajectories may mitigate the credit misalignment issue), it cannot adjust advantages for actions that appear only once across all trajectories. In contrast, our method applies Bellman iteration to estimate the state-value of each node on the graph, deriving step-level advantages from the value differences between two nodes. This formulation does not depend on the occurrences number of the action, and even for actions appearing only once, our method can take advantage of bootstrapping on the graph to achieve a good step-level advantage estimation.

As for performance, the results in Tab.1 show that our method can achieve significantly better results than SALT, validating the superiority of our method over SALT in estimating step-level advantages.

## A.4 DISCUSSION WITH RTMC

We further discuss with RTMC (Wang et al., 2026), the authors proposed Rollout-Tree Monte Carlo advantage estimation, which aggregates return statistics from group rollouts over a tree structure to compute per-step Q-values and advantages, enabling fine-grained credit assignment. Specifically, for each step t, the discounted return is computed as $\begin{array} { r } { R _ { t } = \sum _ { l = 0 } ^ { T - t } \gamma ^ { l } r _ { t + l } } \end{array}$ . Then, for each state-action pair $( s , a )$ , the Q-value is estimated by averaging returns across rollouts visiting $\begin{array} { r } { ( s , a ) \colon \widehat { Q } ( s , a ) = \frac { 1 } { N ( s , a ) } \sum _ { i : ( s _ { t } ^ { i } , a _ { t } ^ { i } ) = ( s , a ) } R _ { t } ^ { i } } \end{array}$ , where $N ( s , a )$ counts the number of rollouts that visit state s and take action a. The state value for s is estimated as $\begin{array} { r } { \widehat V ( s ) = \frac { 1 } { N ( s ) } \sum _ { i : s _ { t } ^ { i } = s } R _ { t } ^ { i } } \end{array}$ , where $\begin{array} { r } { N ( s ) = \sum _ { a } N ( s , a ) } \end{array}$ . The per-step advantage is finally derived as the difference between the action value and the state value: $\widehat { A } ( s , a ) = \widehat { Q } ( s , a ) - \widehat { V } ( s )$

Fundamentally, the advantage estimation in RTMC is equivalent to that in GiGPO. First, both methods compute discounted returns for each step and then group steps that share the same state into a group and normalize the discounted returns within each group. While RTMC additionally averages the returns for each $( s , a )$ pair within the state-action group before state-group normalization, this does not introduce any essential difference with GiGPO’s step-level advantage estimation. Consequently, the advantages of our method over GiGPO discussed in Appendix A.1 apply equally to RTMC.

## B THE OVERALL ALGORITHMIC PROCEDURE FOR GRAFT

We present the overall algorithmic procedure of our GRAFT in Algorithm 1. Our method utilizes graph as the basic representation structure to model state-action transition relationship. Its key characteristic is that it is grounded in reinforcement learning theory rather than heuristics: we perform Bellman iteration on the graph to estimate the state-value of each node, and then derive the step-level advantage from the difference in node values. Furthermore, we extend GAE on the graph to obtain more robust advantage estimation. Additionally, for loss computation, we identify the misalignment between the step-level advantage and the optimization objective in GRPO, thus we further adopt a calibrated optimization objective (Eq.(4)) to ensure better alignment.

Algorithm 1 GRAFT: GRAph-based Faithful STep-level Credit Assignment   
Require: Policy $\pi _ { \theta } ,$ environment E, dataset D, group size N, discount factor $\gamma _ { - }$ , weighting factor λ   
1: for each training iteration do   
2: Sample tasks $\left\{ q _ { k } \right\}$ from $\mathcal { D } ;$ collect N rollouts per task under $\pi _ { \boldsymbol { \theta } } $ batch $\tau$   
3: // Graph construction   
4: for each task $q _ { k }$ do   
5: Build trajectory graph $\mathcal { G } _ { k }$ from $\{ ( s _ { t } , a _ { t } , s _ { t + 1 } ) \} \subset \mathcal { T }$ via state canonicalization $\phi$   
6: Identify sink nodes; assign terminal reward $\dot { R _ { i } } \in \{ 0 , 1 \}$   
7: end for   
8: // State-value estimation   
9: for each task $q _ { k }$ do   
10: Solve the Bellman equation (5) on $\mathcal { G } _ { k }$ via value iteration $ \{ V ( u ) \}$   
11: end for   
12: // Step-level advantage computation   
13: for each step $( s _ { t } , a _ { t } , s _ { t + 1 } ) \in \mathcal { T }$ do   
14: $A _ { t }  \dot { \gamma } \dot { \cdot } \dot { V } ( \phi ( s _ { t + 1 } ) ) \dot { - } V ( \phi ( s _ { t } ) )$   
15: $\begin{array} { r } { A _ { t } ^ { G A E } \gets A _ { t } + \sum _ { k = 1 } ^ { \bar { T } - t } ( \gamma \lambda ) ^ { k } \cdot \bar { A } _ { t + k } , \quad \bar { A } _ { t + k } = \frac { 1 } { | \mathcal { E } _ { k } ( a _ { t } ) | } \sum _ { ( u ^ { \prime } , a ^ { \prime } , v ^ { \prime } ) \in \mathcal { E } _ { k } ( a _ { t } ) } \left[ \gamma V \bigl ( \phi ( v ^ { \prime } ) \bigr ) - \phi ( x ^ { \prime } ) \bigr ) \right] } \end{array}$   
$V ( \phi ( u ^ { \prime } ) ) ]$   
16: $\hat { A } _ { \mathrm { s t e p } } ( s _ { t } , a _ { t } ) \gets \mathrm { G r o u p N o r m } ( A _ { t } ^ { G A E } ; \mathcal { G } ( \phi ( s _ { t } ) ) )$   
17: end for   
18: // Policy update   
19: Update $\pi _ { \theta }$ via the calibrated optimization objective in Eq.(4) on batch T using $\{ \hat { A } _ { i , t } \}$   
20: end for

## C PROOF OF PROPOSITION 1

Proof. We first formalize the distribution under which the advantage estimation error is evaluated. Let $d ^ { \pi }$ denote the state-action visitation distribution induced by policy π over the time steps used for policy optimization. For a step-level credit signal X, its population estimation error is

$$
\mathcal { E } ( X ) = \mathbb { E } _ { \tau \sim \pi } \left[ \left( X _ { t } - A ^ { \pi } ( S _ { t } , A _ { t } ) \right) ^ { 2 } \right]\tag{12}
$$

where the expectation is taken over both the visitation of $( S _ { t } , A _ { t } )$ and the stochastic continuation of the trajectory after time t.

Graph-based TD credit. Because the environment is deterministic, each state-action pair (s, a) has a unique successor state

$$
s ^ { \prime } = T ( s , a )\tag{13}
$$

The Bellman equation therefore gives

$$
Q ^ { \pi } ( s , a ) = r ( s , a , s ^ { \prime } ) + \gamma V ^ { \pi } ( s ^ { \prime } )\tag{14}
$$

Using the standard definition of the policy advantage,

$$
A ^ { \pi } ( s , a ) = Q ^ { \pi } ( s , a ) - V ^ { \pi } ( s )\tag{15}
$$

we obtain

$$
A ^ { \pi } ( s , a ) = r ( s , a , s ^ { \prime } ) + \gamma V ^ { \pi } ( s ^ { \prime } ) - V ^ { \pi } ( s ) = X _ { t } ^ { G }\tag{16}
$$

Hence, the population estimation error of our graph-based step-level credit $X _ { t } ^ { G }$ follows that

$$
\mathcal { E } ( X ^ { G } ) = \mathbb { E } _ { \tau \sim \pi } \left[ \left( X _ { t } ^ { G } - A ^ { \pi } ( S _ { t } , A _ { t } ) \right) ^ { 2 } \right] = 0\tag{17}
$$

State-grouped step-level credit. Recall that the population counterpart of the state-grouped steplevel signal in GiGPO (Feng et al., 2025) is

$$
X _ { t } ^ { S } = G _ { t } - V ^ { \pi } ( S _ { t } )\tag{18}
$$

where $\begin{array} { r } { G _ { t } = \sum _ { k = t } ^ { T _ { \tau } } \gamma ^ { k - t } r _ { k } } \end{array}$ is the sampled return from time t.

For a fixed state-action pair $( s , a )$ , subtracting the true policy advantage from Equation (18) yields

$$
\begin{array} { c } { { X _ { t } ^ { S } - A ^ { \pi } ( s , a ) = G _ { t } - V ^ { \pi } ( s ) - [ Q ^ { \pi } ( s , a ) - V ^ { \pi } ( s ) ] } } \\ { { { } } } \\ { { { } = G _ { t } - Q ^ { \pi } ( s , a ) } } \end{array}\tag{19}
$$

By the definition of the action-value function,

$$
Q ^ { \pi } ( s , a ) = \mathbb { E } [ G _ { t } \mid S _ { t } = s , A _ { t } = a ]\tag{20}
$$

Therefore,

$$
\begin{array} { r l r } & { } & { { \mathbb { E } } \left[ \left( X _ { t } ^ { S } - A ^ { \pi } ( s , a ) \right) ^ { 2 } | S _ { t } = s , A _ { t } = a \right] = { \mathbb { E } } \left[ \left( G _ { t } - Q ^ { \pi } ( s , a ) \right) ^ { 2 } | s , a \right] } \\ & { } & { = \operatorname { V a r } \left( G _ { t } | S _ { t } = s , A _ { t } = a \right) \quad } \end{array}\tag{21}
$$

Taking the expectation over the visitation distribution gives

$$
\mathcal { E } ( X ^ { S } ) = \mathbb { E } _ { ( s , a ) \sim d ^ { \pi } } \left[ \mathrm { V a r } \left( G _ { t } \ | \ S _ { t } = s , A _ { t } = a \right) \right]\tag{22}
$$

In particular, $\mathcal { E } ( X ^ { S } ) \geq 0$ . We next establish when the inequality is strict. Under sparse rewards, suppose that

$$
r _ { k } = 0 \quad \mathrm { f o r } k < T _ { \tau } , \qquad R ( \tau ) = r _ { T _ { \tau } }\tag{23}
$$

where a failed trajectory receives $R ( \tau ) = 0$ and a successful trajectory receives $R ( \tau ) > 0$ . The return from time t is then

$$
G _ { t } = \gamma ^ { T _ { \tau } - t } R ( \tau )\tag{24}
$$

For every $( s , a ) \in \Omega .$ , both successful and failed continuations occur with positive probability. Consequently,

$$
\operatorname* { P r } ( G _ { t } = 0 \mid s , a ) > 0 \qquad \mathrm { a n d } \qquad \operatorname* { P r } ( G _ { t } > 0 \mid s , a ) > 0\tag{25}
$$

Thus, $G _ { t }$ is not constant conditional on $( s , a )$ , which implies

$$
\operatorname { V a r } ( G _ { t } \mid s , a ) > 0 , \qquad ( s , a ) \in \Omega\tag{26}
$$

If Ω has positive visitation probability $d ^ { \pi } ( \Omega ) > 0$ , then Equations (22) and (26) imply $\mathcal { E } ( X ^ { S } ) > 0$

Trajectory-level credit. Let

$$
b _ { E } = \mathbb { E } _ { \tau \sim \pi } [ R ( \tau ) \ | \ S _ { 0 } = s _ { 0 } ]\tag{27}
$$

denotes the population trajectory-level baseline for the fixed task. The population counterpart of the trajectory-level credit credit in GRPO is

$$
X _ { t } ^ { E } = R ( \tau ) - b _ { E }\tag{28}
$$

Although the same trajectory-level signal is assigned to every step in the trajectory, the desired target at step t is the state-dependent advantage $A ^ { \pi } ( S _ { t } , A _ { t } )$ . For a fixed state-action pair $( s , a )$ , define

$$
\mu _ { E } ( s , a ) = \mathbb { E } [ X _ { t } ^ { E } \mid S _ { t } = s , A _ { t } = a ]\tag{29}
$$

The estimation error can be decomposed as

$$
X _ { t } ^ { E } - A ^ { \pi } ( s , a ) = \big [ X _ { t } ^ { E } - \mu _ { E } ( s , a ) \big ] + [ \mu _ { E } ( s , a ) - A ^ { \pi } ( s , a ) ]\tag{30}
$$

Squaring both sides and taking the conditional expectation gives

$$
\begin{array} { r } { \mathbb { E } \left[ \left( X _ { t } ^ { E } - A ^ { \pi } ( s , a ) \right) ^ { 2 } | S _ { t } = s , A _ { t } = a \right] = \mathrm { V a r } ( X _ { t } ^ { E } \mid s , a ) + \left[ \mu _ { E } ( s , a ) - A ^ { \pi } ( s , a ) \right] ^ { 2 } } \end{array}\tag{31}
$$

The above derivation is because the cross term has conditional expectation zero.

Since $b _ { E }$ is a constant for the fixed task, thus

$$
\operatorname { V a r } ( X _ { t } ^ { E } \mid s , a ) = \operatorname { V a r } ( R ( \tau ) \mid s , a )\tag{32}
$$

Substituting Equation (32) into Equation (31) yields

$$
\begin{array} { r } { \mathbb { E } \left[ \left( X _ { t } ^ { E } - A ^ { \pi } ( s , a ) \right) ^ { 2 } | s , a \right] = \mathrm { V a r } ( R ( \tau ) | s , a ) + \left[ \mu _ { E } ( s , a ) - A ^ { \pi } ( s , a ) \right] ^ { 2 } } \end{array}\tag{33}
$$

Taking the expectation over $d ^ { \pi }$ gives

$$
\begin{array} { r } { \mathcal { E } ( X ^ { E } ) = \mathbb { E } _ { ( s , a ) \sim d ^ { \pi } } \left[ \mathrm { V a r } ( R ( \tau ) \mid s , a ) \right] + \mathbb { E } _ { ( s , a ) \sim d ^ { \pi } } \left[ ( \mu _ { E } ( s , a ) - A ^ { \pi } ( s , a ) ) ^ { 2 } \right] } \end{array}\tag{34}
$$

Both terms on the right-hand side are non-negative, and hence $\mathcal { E } ( X ^ { E } ) \geq 0$

For every $( s , a ) \in \Omega$ , both successful and failed trajectories occur with positive probability. Therefore,

$$
\operatorname* { P r } ( R ( \tau ) = 0 \mid s , a ) > 0 \qquad \mathrm { a n d } \qquad \operatorname* { P r } ( R ( \tau ) > 0 \mid s , a ) > 0\tag{35}
$$

It follows that

$$
\operatorname { V a r } ( R ( \tau ) \mid s , a ) > 0 , \qquad ( s , a ) \in \Omega\tag{36}
$$

If $d ^ { \pi } ( \Omega ) > 0$ , Equation (31) therefore implies $\mathcal { E } ( X ^ { E } ) > 0$

Comparison. Equation (17) establishes that $\mathcal { E } ( X ^ { G } ) = 0$ , Equations (22) and (31) establish that $\mathcal { E } ( X ^ { \hat { S } } ) \geq 0 , \mathcal { E } ( X ^ { \hat { E } } ) \geq 0$ . Therefore,

$$
\mathcal { E } ( X ^ { G } ) \leq \operatorname* { m i n } \left\{ \mathcal { E } ( X ^ { S } ) , \mathcal { E } ( X ^ { E } ) \right\}\tag{37}
$$

Furthermore, if Ω has positive visitation probability under $\pi ,$ Equations (26) and (32) give

$$
\mathcal { E } ( X ^ { S } ) > 0 , \qquad \mathcal { E } ( X ^ { E } ) > 0\tag{38}
$$

Combining Equations (37) and (38), we obtain the strict inequality

$$
\mathcal { E } ( X ^ { G } ) < \operatorname* { m i n } \left\{ \mathcal { E } ( X ^ { S } ) , \mathcal { E } ( X ^ { E } ) \right\}\tag{39}
$$

This completes the proof.

Remark. Proposition 1 compares the population credit signals underlying the three methods. In practice, GRAFT replaces $V ^ { \pi }$ with the graph-based estimator $\widehat { V }$ . For

$$
\widehat { X } _ { t } ^ { G } = r _ { t } + \gamma \widehat { V } ( S _ { t + 1 } ) - \widehat { V } ( S _ { t } )
$$

its deviation from the population graph-based TD credit satisfies

$$
\widehat X _ { t } ^ { G } - X _ { t } ^ { G } = \gamma \left[ \widehat V ( S _ { t + 1 } ) - V ^ { \pi } ( S _ { t + 1 } ) \right] - \left[ \widehat V ( S _ { t } ) - V ^ { \pi } ( S _ { t } ) \right]
$$

Thus, the empirical graph-based step-level advantage estimator approaches the zero-error population target as the graph-based value estimates become more accurate.

Proposition 2 (Lower Finite-Sample Advantage Estimation Error). Consider $N \geq 2$ independent length- $\mathbf { \nabla } \cdot T$ rolloutsfrom the same initial state under a fixed policy π in a deterministic environment with sparse terminal-state rewards $R \in [ 0 , 1 ]$ , and $\gamma \in ( 0 , 1 \check { } )$ ). Let

$$
X _ { t } ^ { G } = \gamma \widehat { V } ( S _ { t + 1 } ) - \widehat { V } ( S _ { t } ) , \quad X _ { t } ^ { S } = G _ { t } - \overline { { G } } ( S _ { t } ) , \quad X _ { t } ^ { E } = R - \overline { { R } }
$$

be the unnormalized graph-based, state-grouped, and trajectory-level credits of GRAFT, GiGPO, and GRPO, respectively. $\widehat { V }$ is the empirical Bellman solution with ${ \widehat { V } } ( S _ { T } ) = R , G _ { t } = \gamma ^ { T - t } R ,$ G(u) averages over all visits to u across trajectories and time steps, and R is the whole-group mean. Let $e _ { t } = \widehat { V } ( S _ { t } ) - V ^ { \pi } ( S _ { t } )$ denotes the state-value estimation error. Define the mean-squared advantage estimation error as $\begin{array} { r } { \dot { \mathcal { E } } ( X ) = \mathbb { E } _ { \tau \sim \pi } [ ( X _ { t } - A ^ { \pi } ( S _ { t } , A _ { t } ) ) ^ { 2 } ] } \end{array}$ , where the expectation is taken over the sampled rollout group and an action selected uniformly from that group. If

$$
\| \gamma e _ { t + 1 } - e _ { t } \| _ { 2 } + \operatorname* { m a x } \left\{ \| \overline { { G } } ( S _ { t } ) - V ^ { \pi } ( S _ { t } ) \| _ { 2 } , \sqrt { \frac { \mathrm { V a r } ( R ) } { N } } \right\} \leq \sqrt { \mathbb { E } [ \mathrm { V a r } ( G _ { t } \mid S _ { t } , A _ { t } ) ] }\tag{40}
$$

then

$$
\mathcal { E } ( X ^ { G } ) \leq \operatorname* { m i n } \{ \mathcal { E } ( X ^ { S } ) , \mathcal { E } ( X ^ { E } ) \}\tag{41}
$$

Both comparisons are strict if Eq. 40 is strict.

Proof. We evaluate all three credit signals on the same rollout group. Independently of the group, select i uniformly from $\{ 1 , \ldots , N \}$ and t uniformly from $\{ \bar { 0 } , \ldots , T ^ { \mathrm { ~ - ~ } 1 } \}$ , and write $( S _ { t } , A _ { t } , S _ { t + 1 } , R ) = ( S _ { i , t } , A _ { i , t } , S _ { i , t + 1 } , R _ { i } )$ . For any random variable Z, let $\| Z \| _ { 2 } = ( \bar { \mathbb { E } } [ Z ^ { 2 } ] ) ^ { 1 / 2 }$

Following the terminal-value convention, $V ^ { \pi } ( S _ { i , T } ) = R _ { i }$ and

$$
G _ { i , t } = \gamma ^ { T - t } R _ { i } , \qquad Q ^ { \pi } ( S _ { t } , A _ { t } ) = \mathbb { E } [ G _ { t } \mid S _ { t } , A _ { t } ] , \qquad A ^ { \pi } = Q ^ { \pi } - V ^ { \pi }\tag{42}
$$

Define the state visit group

$$
\mathcal { T } ( u ) = \{ ( j , k ) : 1 \leq j \leq N , 0 \leq k < T , S _ { j , k } = u \}\tag{43}
$$

The empirical baselines are

$$
\overline { { G } } ( u ) = \frac { 1 } { | \mathcal { T } ( u ) | } \sum _ { ( j , k ) \in \mathcal { Z } ( u ) } G _ { j , k } , \qquad \overline { { R } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } R _ { j }\tag{44}
$$

All occurrences are included, including the selected occurrence and repeated visits within one trajectory. The evaluated group is always nonempty. For brevity in the proof, set

$$
\begin{array} { r l r } & { \sigma _ { G } = \sqrt { \mathbb { E } [ \mathrm { V a r } ( G _ { t } \mid S _ { t } , A _ { t } ) ] } , } & \\ & { \sigma _ { R } = \sqrt { \mathbb { E } [ \mathrm { V a r } ( R \mid S _ { t } , A _ { t } ) ] } , } \\ & { \eta _ { S } = \| \overline { { G } } ( S _ { t } ) - V ^ { \pi } ( S _ { t } ) \| _ { 2 } , } & { \eta _ { E } = \sqrt { \mathrm { V a r } ( R ) / N } } \end{array}\tag{45}
$$

Graph-based TD credit. Because the environment is deterministic, the true Bellman equation at an executed transition gives

$$
A ^ { \pi } ( S _ { t } , A _ { t } ) = \gamma V ^ { \pi } ( S _ { t + 1 } ) - V ^ { \pi } ( S _ { t } )\tag{46}
$$

Consequently,

$$
\begin{array} { r l r } {  { X _ { t } ^ { G } - A ^ { \pi } ( S _ { t } , A _ { t } ) = \gamma \big [ \widehat { V } ( S _ { t + 1 } ) - V ^ { \pi } ( S _ { t + 1 } ) \big ] - \big [ \widehat { V } ( S _ { t } ) - V ^ { \pi } ( S _ { t } ) \big ] } } \\ & { } & { = \gamma e _ { t + 1 } - e _ { t } } \end{array}\tag{47}
$$

Thus the finite-sample graph-credit error is exactly

$$
\sqrt { \mathcal E ( X ^ { G } ) } = \| \gamma e _ { t + 1 } - e _ { t } \| _ { 2 }\tag{48}
$$

State-grouped step-level credit. Subtracting the true advantage yields

$$
X _ { t } ^ { S } - A ^ { \pi } ( S _ { t } , A _ { t } ) = \left[ G _ { t } - Q ^ { \pi } ( S _ { t } , A _ { t } ) \right] - \left[ \overline { { { G } } } ( S _ { t } ) - V ^ { \pi } ( S _ { t } ) \right]\tag{49}
$$

By the definition of $Q ^ { \pi }$

$$
\| G _ { t } - Q ^ { \pi } ( S _ { t } , A _ { t } ) \| _ { 2 } ^ { 2 } = \mathbb { E } [ \operatorname { V a r } ( G _ { t } \mid S _ { t } , A _ { t } ) ] = \sigma _ { G } ^ { 2 }\tag{50}
$$

Applying the reverse triangle inequality gives

$$
\sqrt { \mathcal { E } ( X ^ { S } ) } \geq ( \sigma _ { G } - \eta _ { S } ) _ { + } , \qquad ( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}\tag{51}
$$

The argument permits arbitrary dependence between the sampled return and its empirical stategrouped baseline. In particular, it applies to cross-time groups and repeated visits without treating them as independent samples.

Trajectory-level credit. Let $b _ { E } = \mathbb { E } [ R ]$ and define the population-centered signal $Y _ { t } ^ { E } = R - b _ { E }$ Its conditional bias–variance decomposition is

$$
\begin{array} { r l } & { \mathbb { E } [ ( Y _ { t } ^ { E } - A ^ { \pi } ( S _ { t } , A _ { t } ) ) ^ { 2 } ] = \mathbb { E } [ \mathrm { V a r } ( R \mid S _ { t } , A _ { t } ) ] } \\ & { \qquad + \mathbb { E } \Big [ \big ( \mathbb { E } [ R \mid S _ { t } , A _ { t } ] - b _ { E } - A ^ { \pi } ( S _ { t } , A _ { t } ) \big ) ^ { 2 } \Big ] } \end{array}\tag{52}
$$

Hence $\| Y _ { t } ^ { E } - A ^ { \pi } ( S _ { t } , A _ { t } ) \| _ { 2 } \geq \sigma _ { R }$ . Since the true state includes t and $G _ { t } = \gamma ^ { T - t } R$

$$
\sigma _ { G } ^ { 2 } = \mathbb { E } [ \gamma ^ { 2 ( T - t ) } \operatorname { V a r } ( R \mid S _ { t } , A _ { t } ) ] \leq \sigma _ { R } ^ { 2 }\tag{53}
$$

The empirical baseline includes the selected trajectory, so

$$
X _ { t } ^ { E } = R _ { i } - \overline { { R } } = \left( 1 - \frac { 1 } { N } \right) R _ { i } - \frac { 1 } { N } \sum _ { j \neq i } R _ { j }\tag{54}
$$

The $N$ complete rollouts are independent and identically distributed. Therefore,

$$
\| X _ { t } ^ { E } - Y _ { t } ^ { E } \| _ { 2 } ^ { 2 } = \mathbb { E } [ ( \overline { { R } } - b _ { E } ) ^ { 2 } ] = \frac { \mathrm { V a r } ( R ) } { N } = \eta _ { E } ^ { 2 }\tag{55}
$$

A further application of the reverse triangle inequality gives

$$
\sqrt { \mathcal { E } ( X ^ { E } ) } \geq ( \sigma _ { R } - \eta _ { E } ) _ { + } \geq ( \sigma _ { G } - \eta _ { E } ) _ { + }\tag{56}
$$

No independence between $R _ { i }$ and R is assumed.

Comparison. Set $\eta = \operatorname* { m a x } \{ \eta _ { S } , \eta _ { E } \}$ . Equations 51 and 56 imply

$$
\operatorname* { m i n } \biggl \{ \sqrt { \mathcal E ( X ^ { S } ) } , \sqrt { \mathcal E ( X ^ { E } ) } \biggr \} \geq ( \sigma _ { G } - \eta ) _ { + }\tag{57}
$$

Under condition 40,

$$
\sqrt { \mathcal { E } ( X ^ { G } ) } = \| \gamma e _ { t + 1 } - e _ { t } \| _ { 2 } \leq \sigma _ { G } - \eta\tag{58}
$$

The right-hand side is nonnegative. Squaring proves

$$
\mathcal { E } ( X ^ { G } ) \leq \operatorname* { m i n } \{ \mathcal { E } ( X ^ { S } ) , \mathcal { E } ( X ^ { E } ) \}\tag{59}
$$

If condition 40 is strict, the preceding comparison of square roots is strict and $\sigma _ { G } - \eta > 0$ , yielding strict inequalities against both baselines. This completes the proof. □

Remark. Condition 40 relates the admissible graph-credit error to the accuracy of the empirical baselines. For a fixed level of conditional return variability, smaller baseline errors increase the margin $\sigma _ { G } - \operatorname* { m a x } \{ \eta _ { S } , \eta _ { E } \}$ , allowing a larger graph-credit estimation error while preserving the comparison. With correct state aggregation and adequate coverage of relevant states, additional independent rollouts can improve both the empirical baselines and the transition statistics used for graph-value estimation. These improvements act in complementary directions: more accurate baselines enlarge the admissible error margin, while more accurate graph values can reduce the graph-credit error. Thus, the condition describes a regime in which reliable cross-trajectory statistics support more accurate credit assignment, without requiring exact value estimates.

## D ADDITIONAL EXPERIMENTAL ANALYSIS

## D.1 HYPERPARAMETER ABLATION STUDIES

Tab.4 investigates the sensitivity of our method to the discount factor γ and the weighting factor λ. For γ, we observe that larger values yield better performance. This is expected in long-horizon agentic tasks, where a higher γ facilitates the backward propagation of sparse terminal rewards to earlier steps; conversely, lower values may cause initial steps to receive insufficient reward signals for effective learning. Regarding λ, a larger value indicates a longer backward-looking horizon during Graph GAE estimation, incorporating more k-hop step estimators to refine the single-step estimator. Since the optimal lookback depth varies with task complexity, λ is typically tuned per benchmark. Overall, the results demonstrate robust performance across a reasonable range of hyperparameters.

Table 4: Hyperparameter ablation studies on the discount factor γ and the weighing factor λ in Graph GAE (Eq.(9)).
<table><tr><td>γ</td><td>ALFWorld</td><td>WebShop</td><td>λ</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>0.8</td><td>95.6</td><td>76.3</td><td>0.5</td><td>95.6</td><td>79.7</td></tr><tr><td>0.9</td><td>96.1</td><td>79.2</td><td>0.6</td><td>97.2</td><td>81.2</td></tr><tr><td>0.95</td><td>96.4</td><td>78.1</td><td>0.8</td><td>96.4</td><td>82.8</td></tr><tr><td>0.97</td><td>96.9</td><td>80.5</td><td>0.9</td><td>96.4</td><td>77.3</td></tr><tr><td>0.99</td><td>96.9</td><td>81.0</td><td>0.95</td><td>97.7</td><td>81.2</td></tr></table>

## D.2 NODE GROUP SIZE

![](images/fede522dd435b9adc666b31e319664bb77d691ae3da51f0b14bb99784bca55ae.jpg)  
Figure 5: Dynamics of average node group size during the training process.

![](images/28f4eee5e001cb4a16620c142add48cb9b6f77ce0d657b24eb73a416771b06c7.jpg)

![](images/65ecb3d2b54484146a4333f4ba0cf17d85c47029bf4848334ea8dc798bea2669.jpg)  
Figure 6: The distributions of node group size during training.

Here, we present the dynamics of the average node group size (i.e., the number of outgoing edges per node) during training and the distributions of group size. As shown in Fig.5, the average group size for WebShop and ALFWorld is around 4 and 5. This indicates that many environmental observations recur in distinct trajectories yet lead to divergent outcomes, underscoring the necessity of aggregating trajectories into a unified trajectory graph. In WebShop, the average group size initially increases, suggesting that the agent’s decisions converge during early training, leading to more frequent state revisits. Subsequently, the group size decreases because fine-grained credit assignment helps the agent to eliminate redundant steps, leading to shorter trajectories and fewer total observations. In ALFWorld, the average group size increases, this is because that ALFWorld tasks inherently require repetitive actions, forcing the agent to revisit identical states. This inherent repetition amplifies the upward trend in group size, outweighing the downward trend caused by redundant step removal. Furthermore, as shown in Fig.6, on both WebShop and ALFWorld benchmarks, more than 60% of steps fall into groups of 2 or larger, meaning the more faithful step-level advantage estimation benefit from the shared state group-aggregation mechanism.

## D.3 INFORMATION PROPAGATION DEGREE

We further analyze the dynamics and distributions of the information propagation degree during training. The information propagation degree is defined as the number of distinct trajectories contributing reward information to the value estimation of a node (or state), as starting from one node in the graph can lead to multiple different trajectories. As shown in Fig.7, our method consistently achieves a higher information propagation degree than GiGPO across training. This indicates that graph-based state-value estimation will aggregate reward signals from a broader set of trajectories for each node. In contrast, GiGPO’s simple state grouping restricts reward propagation to the existing trajectory, preventing cross-trajectory credit flow. The results further support the analysis in Appendix A.1 that: graph-based state-value estimation will be more reliable as it can leverage more reward information. Furthermore, in Fig.8, up to 60% of the steps in GiGPO receive reward information from only a single trajectory, whereas in our GRAFT, this proportion significantly drops to 20%.

![](images/66f30178749381a6aa4ad25ca7d0d705feb6faabb6842b02bc1011a8aaa654ab.jpg)  
Figure 7: Dynamics of average information propagation degree during training.

![](images/33af27a49c9fb671d34724625137286b5e0ebe213d8a81d9b4ba1c21545b935b.jpg)  
Figure 8: The distributions of information propagation degree.

## D.4 ADDITIONAL RESULTS

Performance based on Qwen3 model series. In Tab.1, for the convenience of direct and fair comparison with prior methods, we conducted experiments based on the Qwen2.5-Instruct model series. Here, to further validate generalizability, we further extend our evaluation to the Qwen3 family (e.g., Qwen3-4B and Qwen3-8B, we reproduce GiGPO and GraphGPO for comparison). The results in Tab.5 show that our method significantly outperforms GiGPO and GraphGPO on both Qwen3-4B and Qwen3-8B models. These evaluations confirm that our method is not tied to a specific base model, but can be effectively transferred to other base models.

Table 5: Performance on ALFWorld and WebShop. These results are based on the Qwen3 model series, e.g., Qwen3-4B and Qwen3-8B.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>All</td><td>Score</td><td>Succ.</td></tr><tr><td colspan="10"></td></tr><tr><td>RL Training</td><td>GiGPO</td><td>100.0</td><td>81.8</td><td>Qwen3-4B 77.3</td><td>54.5</td><td>72.0</td><td>84.0</td><td>82.0</td><td>84.1</td><td>70.6</td></tr><tr><td>RL Training</td><td>GraphGPO</td><td>100.0</td><td>70.0</td><td>83.3</td><td>78.6</td><td>92.9</td><td>69.6</td><td>85.9</td><td>85.8</td><td>78.9</td></tr><tr><td>RL Training</td><td>GRAFT (Ours)</td><td>100.0</td><td>90.0</td><td>91.7</td><td>92.9</td><td>100.0</td><td>91.3</td><td>95.3</td><td>89.6</td><td>82.8</td></tr><tr><td colspan="10">Qwen3-8B</td></tr><tr><td>RL Training</td><td>GiGPO</td><td></td><td></td><td></td><td>85.7</td><td>89.2</td><td>81.2</td><td>86.7</td><td>82.7</td><td>71.1</td></tr><tr><td>RL Training</td><td>GraphGPO</td><td>91.7 95.3</td><td>73.3 90.0</td><td>82.8 83.3</td><td>78.6</td><td>100.0</td><td>95.7</td><td>91.4</td><td>88.0</td><td>78.9</td></tr><tr><td>RL Training</td><td>GRAFT (Ours)</td><td>100.0</td><td>100.0</td><td>100.0</td><td>87.5</td><td>90.5</td><td>94.7</td><td>96.1</td><td>90.9</td><td>80.5</td></tr></table>

## D.5 CASE STUDY

To provide a more intuitive understanding of why our method can yield more faithful step-level advantage estimates, we visualize some case graphs constructed during training on ALFWorld. The graphs are shown in Fig.9 and 10. In Fig.9, all trajectories ultimately lead to success, whereas Fig.10 contains a mix of successful and failing trajectories. Node color reflects state-value: red indicates lower values, while green indicates higher values. The visualization reveals that state-values increase with proximity to successful terminal states and remain low for nodes from which success is unreachable. Furthermore, nodes with mixed outcomes—where some trajectories succeed and others fail—exhibit intermediate state-values, reflecting the expected value under the current policy.

In Fig.9, the task is to clean some cloth and put it in dresser. Although all trajectories ultimately succeed, our method still effectively identifies specific erroneous steps. Consider the actions highlighted by the red boxes, the preceding state is “you pick up the cloth 3 from the countertop 1”.

![](images/a366659023283d86f65d9039725091b551e59caf6869a2ce6ee14b5194299e64.jpg)  
Figure 9: A case study illustrating the constructed graph during training on ALFWorld. In this task, all trajectories finally reach success.

According to the task goal, after obtaining the cloth, it needs to be cleaned first and then placed in the dresser. However, the action “go to dresser 1” is to directly put the cloth in the dresser, which results in deviating from the goal. Subsequently, with making multiple corrective actions, the trajectory ultimately succeeds. In the graph, this action leads to a lower-value state, resulting in a negative advantage to successfully capture this erroneous action. The action “go to sinkbasin $1 ^ { \sqrt { 3 } }$ instead results in successful cleaning before placing in the dresser, thereby transitioning the agent into a higher-value state. Meanwhile, “go to bathroom sinkhole 1” is an invalid action, and it takes an additional step to reach the “sinkhole 1” state. It can be found that the computed advantage of “go to bathroom sinkhole 1” is also slightly lower than that of “go to sinkhole 1”. Overall, this qualitative analysis demonstrates that our method can discriminate among individual actions based on their true contributions, whereas trajectory-level advantage estimation assigns uniform credit to all steps in a successful trajectory, failing to capture fine-grained action quality.

In Fig.10, the task is to put two kettle in cabinet. Let’s first focus on the actions marked in the upper red boxes. The preceding state is “you move the kettle 1 to the cabinet 2”. Now that a kettle has been placed in the cabinet, when action “go to stoveburner 3” is taken, it enables the agent to locate the second kettle. Consequently, this action leads to a significant increase in the subsequent state values. In contrast, “go to countertop 1” returns the agent to a location where kettle 1 was originally found; since the kettle has already been moved, no new kettle can be obtained there. This results in a marked decrease in downstream state values. A similar situation appears in the lower red boxes: “go to stoveburner 1” enters a state without a kettle, resulting in a low value, whereas “go to stoveburner 4” allows the agent to find kettle 3, again producing a substantial increase in value. This example demonstrates that, in tasks containing both successful and failed trajectories, our method effectively identifies critical decision points, assigning higher advantages to steps pivotal to success and lower advantages to those leading toward failure.

## E EXPERIMENT DETAILS

## E.1 DESCRIPTIONS OF BENCHMARKS

ALFWorld is a text-based environment aligned with the ALFRED embodied AI benchmark, which is designed to assess an agent’s ability to perform long-horizon, multi-step decision-making tasks. WebShop is a complex web-based interactive environment that tests LLM agents in realistic online shopping scenarios, requiring them to navigate a realistic web interface to purchase items matching user specifications. The SearchQA benchmark comprises several widely-used search-augmented QA datasets, including single-hop QA (NQ (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), PopQA (Mallen et al., 2023)) and multi-hop QA (HotpotQA (Yang et al., 2018), 2Wiki (Ho et al., 2020), MuSiQue (Trivedi et al., 2022), Bamboogle (Press et al., 2023)).

## E.2 TRAINING DETAILS

To ensure a direct and fair comparison with previous methods, we follow the training and evaluation configurations from GiGPO (Feng et al., 2025) across all benchmarks. The detailed settings for each benchmark are described below:

![](images/ae9bb2f3f81468376e9e79e7aca96c9a0846a1bc9835a8aab14e2dfd3fabf474.jpg)  
Figure 10: A case study illustrating the constructed graph during training on ALFWorld. This graph contains a mix of successful and failing trajectories.

Hyperparameters for ALFWorld. In ALFWorld, since states are predefined strings provided by the environment, state aggregation for building graph node is based on exact matching. We set the maximum prompt and response lengths to 2048 and 512 tokens, respectively. Each trajectory allows up to 50 environment steps. Training runs for 150 steps with a policy learning rate of $1 e ^ { - \bar { 6 } }$ We employ a rule-based reward scheme: +10 for success, 0 for failure. To handle invalid actions generated by the agent, we apply a reward penalty of −0.1. For all group-based RL methods (GRPO, GiGPO, HCAPO, GraphGPO), we use a group size of $G = 8$ and sample 16 different tasks per rollout step, yielding $1 6 \times 8 = 1 2 8$ parallel environments. In contrast, PPO uses 128 independent environments for fair comparison. The rollout and validation temperatures are set to 1.0 and 0.4, respectively. The mini-batch size is 256, and the KL-divergence loss coefficient $\beta _ { K L }$ is 0.01.

Hyperparameters for WebShop. In WebShop, state aggregation is also based on exact matching. We set the maximum prompt and response lengths to 5120 and 512 tokens, respectively, with each episode capped at 15 environment steps. Training runs for 150 steps with a policy learning rate of $1 \dot { e } ^ { - 6 }$ . We adopt a rule-based reward scheme: assigning +10 for success and 0 for failure. Invalid actions are penalized with a reward of −0.1. Consistent with ALFWorld, all group-based RL methods use a group size of $G = 8$ and sample 16 tasks per rollout step, yielding 128 parallel environments. PPO uses 128 distinct environments for rollouts. The rollout and validation temperatures are set to 1.0 and 0.4, respectively. The mini-batch size is 64, and $\beta _ { k L }$ is 0.01.

Hyperparameters for SearchQA. In SearchQA, as there are no predefined state strings provided by the environment, the state aggregation is based on similarity-based matching, with the similarity threshold τ set to 0.9. state aggregation for building graph node is based on exact matching. We set the maximum prompt and response lengths to 4096 and 512 tokens, respectively. The maximum number of turns is set to 4. The learning rate is $1 e ^ { - 6 }$ for the policy model. We employ a rule-based reward scheme: assigning +1 for success and 0 for failure. Invalid actions are penalized with a reward of −0.01. We set the training data size to 256 and use a group size of $G = 5 \mathrm { . }$ . The rollout and validation temperatures are set to 1.0 and 0.0, respectively. The mini-batch size is 512, and $\beta _ { K L }$ is 0.001.

## E.3 TRAINING PROMPTS

The prompts we use for LLM agents are constructed using Python-style string formatting, where placeholders enclosed in curly braces (e.g., task description, step count, and current observation) serve as semantic slots dynamically populated at runtime via Python’s .format() function. To enrich the agent’s context, we incorporate historical information and set the history length to 2 for ALFWorld and WebShop, while retaining the full history for search-augmented QA experiments.

The <think> </think> block instructs the agent to perform explicit step-by-step reasoning, thereby facilitating Chain-of-Thought (CoT) deliberation. The <action> </action> block is used to clearly indicate the final action decision. For the search agent, reasoning traces are enclosed in <think> </think>, search queries in <search> </search>, and final answers in <answer> </answer> tags. Retrieved evidence from the search engine is presented within <information> </information> tags. The detailed prompt templates for each benchmark are provided in Fig.11 (ALFWorld), Fig.12 (Webshop), Fig.13 (SearchQA).

![](images/2c38a4ace6abfab9559488b51bd7dd77c5e04e2829cef8356ce880b037a340f8.jpg)  
Figure 11: Prompt template for ALFWorld.

![](images/789f1b4bb1f951a0250b25f0342b4af1abc0cc68f31fa70a00b8b3ad9a98ca7d.jpg)  
Figure 12: Prompt template for WebShop.

![](images/da6558771335af4f6dcd87c8df4d1ab13035440cbbd517c32dada4beeaa09018.jpg)  
Figure 13: Prompt template for SearchQA.