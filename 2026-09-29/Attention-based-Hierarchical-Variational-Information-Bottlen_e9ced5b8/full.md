# Attention-based Hierarchical Variational Information Bottleneck for Robust Multi-Agent Communication under Variable Bandwidth

Lukas Koch Vindbjerg<sup>1</sup>, Qi Zhang<sup>1</sup>, Yury Brodskiy<sup>3</sup> and Lukas Esterle<sup>1,2</sup>.

Abstract— Learning-based multi-agent communication under limited bandwidth does not only require deciding what to communicate, but also structuring messages so that partial transmissions remain useful. We study this problem under prefix truncation, where only the first part of each message is received. To address it, we propose AH-VIB, an attention-based autoregressive variational communication model that combines a variational information bottleneck (VIB) with sequential message generation and a hierarchical robustness loss. We evaluate AH-VIB on a custom cooperative object-inspection and occupancy-mapping task, where agents equipped with a limited field-of-view sensor coordinate to scan inspection objects in an occupancy-grid world, under variable and fixed bandwidth conditions, and compare it against MADDPG, CommNet, a flat VIB baseline, and an autoregressive MLP ablation. AH-VIB achieves competitive mean return while improving performance reliability under the most constrained bandwidth conditions.

These results indicate that AH-VIB improves the reliability and graceful degradation of learned communication under bandwidth constraints.

## I. INTRODUCTION

Multi-robot systems increasingly support tasks such as infrastructure inspection, environmental monitoring, and operation in hazardous environments [1]. Effective coordination in such settings depends on communication, yet available bandwidth can vary with range, interference, and environmental occlusion [2], [3]. Robots must therefore coordinate even when only part of a teammate’s intended message is received.

We consider cooperative object inspection as a representative use case. Agents with limited fields of view must exchange compact information about their observations, relative state, and remaining viewpoints to avoid redundant coverage. As illustrated in Figure 1, a useful communication protocol should support complementary inspection even when individual agents cannot observe the complete environment. When the available communication budget changes, the received portion of a message should provide a useful coarse description rather than an arbitrary fragment.

Multi-agent reinforcement learning (MARL) has produced many learned communication protocols [4], [5], [6], while information-theoretic methods such as IMAC [7] use the Variational Information Bottleneck (VIB) [8] to learn compact messages. However, many learned communication methods assume that messages are either fully delivered or entirely lost [3]. We instead model variable bandwidth through prefix truncation. Only the first k dimensions of a message are received before transmission ends. This is a controlled abstraction of progressive communication degradation; practical channels may additionally involve packet loss, delay, asymmetry, and retransmission, which are outside the present scope.

![](images/004c26644a64de33ae6ceacfbec2904ca65cf0342ad54207b2663768a87ad5da.jpg)  
(a) Step 1

![](images/924636ef72e4cda8bba52d8d9c09f95f871b44dbf3ae8128b870905488c275a5.jpg)  
(b) Step 50  
Fig. 1. Cooperative object inspection under partial observability. Agents with limited fields of view progressively reveal free space and inspectionobject surfaces while coordinating their coverage.

Prefix truncation poses an ordering problem: what should an agent say first? For a message to be useful, it should place task-relevant information in early dimensions, so that short prefixes remain useful and later dimensions provide refinement. We call this property progressive information encoding. Thus, our objective is not merely to compress communication, but to learn ordered, task-oriented message representations.

To achieve this, we propose AH-VIB (Attention-based Hierarchical Variational Information Bottleneck), which combines two complementary components. First, an autoregressive attention-based encoder generates each message dimension conditioned on the preceding prefix, making truncated prefixes structurally consistent. Second, a hierarchical robustness loss penalizes reductions in the centralized critic’s estimate of team return when messages are truncated during training. Together, these components encourage messages that remain useful under changing bandwidth.

We evaluate AH-VIB on a custom cooperative objectinspection and occupancy-mapping task, in which agents with limited fields of view coordinate to scan relevant inspection objects. We compare our communication approach, used to enable the coordination among agents, against MAD-DPG [9], CommNet [4], a flat VIB baseline [7], and an autoregressive MLP ablation (HVIB). AH-VIB achieves its strongest performance at the most constrained bandwidths and provides more reliable behavior than the VIB and MADDPG baselines when only short message prefixes are available.

Our contributions can be summarized as follows:

• We formulate variable-bandwidth MARL as prefix truncation of learned message vectors, as a controlled abstraction of progressively available communication [2], [3].

• We propose an autoregressive attention-based message encoder that imposes a causal ordering over message dimensions, combined with a hierarchical robustness objective that encourages useful message prefixes.

• We evaluate AH-VIB in a cooperative object-inspection task under fixed and variable bandwidth, including return distributions and representation-level analysis of learned message usage.

## II. RELATED WORK

Prior MARL communication work has primarily studied what to communicate and to whom, through learned protocols, recipient targeting, and information-theoretic compression. We instead study how information should be ordered within a message when only a prefix may be received.

## A. Learned Communication in MARL

CommNet [4] was one of the earliest demonstrations of fully differentiable multi-agent communication. CommNet specifically proposed continuous communication via fixed mean pooling of hidden states over K rounds, learning both a policy and a communication protocol end-to-end. While effective on cooperative tasks, CommNet treats all agents messages symmetrically and provides no mechanism for selective or bandwidth-adaptive encoding.

MADDPG [9] popularized the Centralized Training with Decentralized Execution (CTDE) paradigm, where a centralized critic evaluates during training with decentralized execution of agents. This enables stable multi-agent policy gradient updates. TarMAC [5] extended learned communication with a signature-based soft attention mechanism, allowing agents to learn what to communicate and to whom. Our work is complementary. Where TarMAC addresses targeting (who receives a message), we address content encoding under bandwidth constraints (what survives when the channel is limited).

## B. Information-Theoretic Communication

The Variational Information Bottleneck (VIB) [8]provides a principled framework for learning compressed representations that retain only task-relevant information, originally developed outside the multi-agent setting. IMAC [7] adapted this framework to multi-agent communication, applying VIB to message encoding and using KL regularization to learn compact representations that balance task performance against communication cost under bandwidth constraints. Our work builds directly on IMAC’s VIB formulation but identifies its key limitation under prefix truncation. A flat VIB encoder produces all message dimensions simultaneously from a shared hidden state, imposing no ordering across dimensions. AH-VIB instead uses an autoregressive factorization that makes each prefix causally consistent, while the hierarchical robustness loss encourages prefixes to retain task utility.

## C. Bandwidth-Constrained Multi-Robot Communication

Outside the MARL framework, bandwidth constraints in real multi-robot systems have been studied extensively in the robotics literature. ACHORD [2] demonstrates that real multi-robot deployments in non-line-of-sight environments experience intermittent connectivity, with robots accumulating data backlogs in on-board queues that must be transmitted when a connectivity window is available. This stands in stark contrast to the full-delivery assumption of most learned communication methods.

## D. Hierarchical and Progressive Message Encoding

Progressive or embedded source coding produces representations whose initial bits provide a useful coarse description and whose later bits refine it. Embedded zerotree wavelet coding [10], for example, orders coded information by importance so that encoding or decoding can stop at an arbitrary rate. Related ideas appear in learned ordered representations: nested dropout [11] and Matryoshka Representation Learning [12] train nested latent prefixes for adaptive compression or variable representation budgets. We adapt this principle to learned multi-agent communication. The received prefix models the portion of a message that survives a variablebandwidth channel, and AH-VIB combines autoregressive generation with a hierarchical robustness loss to retain task utility under truncation.

## III. SYSTEM MODEL

We consider a cooperative Multi-Agent Reinforcement Learning (MARL) setup where n agents coordinate their actions to achieve a shared objective under partial observability. Each agent relies both on its local observations and also on explicit communication with other agents [13].

## A. Dec-POMDP Formulation

We formally describe the problem as a communicationaugmented Decentralized Partially Observable Markov Decision Process (Dec-POMDP), defined by the tuple:

$$
\langle N , S , \{ \hat { A } _ { i } \} _ { i = 1 } ^ { n } , \{ O _ { i } \} _ { i = 1 } ^ { n } , M , T , R , \gamma \rangle \ .
$$

Here, $N = \{ 1 , \ldots , n \}$ is the set of agents and S is the global state space of the environment. Each agent i has a composite action space $\hat { A } _ { i } = A _ { i } \times M \subset \mathbb { R } ^ { d _ { i } ^ { a } + d _ { m } ^ { - } }$ , so that a composite action is the pair $\left( { { a _ { i } } , m _ { i } } \right)$ . Here, $a _ { i } \in A _ { i } \subset \mathbb { R } ^ { d _ { i } ^ { a } }$ is the physical environment action and $m _ { i } \in M \subset \mathbb { R } ^ { d _ { m } }$ is the message, where $d _ { i } ^ { a }$ and $d _ { m }$ denote the physical-action and message dimensionalities, respectively. Agent i also has a local observation space $o _ { i } \in O _ { i } \subset \mathbb { R } ^ { d _ { i } ^ { o } }$ . The shared message space M is common to all agents. Each agent i sends $m _ { i }$ and receives messages from all other agents $\{ m _ { j } \} _ { j \in N _ { - i } } ,$ where $N _ { - i } = N \setminus \{ i \}$ . The state transition function $T : S \times A _ { 1 } \times$ $\cdots \times A _ { n } \to \Delta ( S )$ maps joint physical actions to a probability distribution over next states. Messages do not directly affect the world state. Finally, $R : S \times A _ { 1 } \times \cdots \times A _ { n } \times S $ R is the shared team reward function, and $\gamma \in [ 0 , 1 )$ is the discount factor.

![](images/c7c40488e48c840398dbc27e7ddb07be4903b227420fa9445495769bfc347646.jpg)  
Fig. 2. Execution flow of the multi-agent system at timestep t. Agent i receives local observation $o _ { i } ^ { t }$ and the message set $\mathbf { m } _ { - i } ^ { t - 1 } = \{ m _ { i } ^ { t - 1 } \} _ { j \in N }$ <sub>N</sub> sent by the other agents at the previous timestep. From these inputs, it produces a physical action $a _ { i } ^ { t }$ and a message m<sup>t</sup>. Before delivery, m<sup>t</sup> is truncated by the channel, retaining only the first k dimensions, and is then broadcast to the other agents for use at timestep t + 1.

At each timestep $t ,$ agent i receives a local observation $o _ { i } ^ { t }$ and the tuple of messages sent by all other agents at the previous timestep,

$$
\mathbf { m } _ { - i } ^ { t - 1 } = \{ m _ { j } ^ { t - 1 } \} _ { j \in N _ { - i } } , \qquad N _ { - i } = N \setminus \{ i \} .\tag{1}
$$

At episode start, all messages are initialized to zero. The agent then selects a composite action $\hat { a } _ { i } ^ { t } = ( a _ { i } ^ { t } , m _ { i } ^ { t } ) \in \hat { A } _ { i }$ according to its policy $\pi _ { \theta }$ as,

$$
\left( a _ { i } ^ { t } , m _ { i } ^ { t } \right) = \pi _ { \theta } ^ { i } \left( o _ { i } ^ { t } , { \bf m } _ { - i } ^ { t - 1 } \right) .\tag{2}
$$

Figure 2 illustrates this execution flow, including the one-step delay between message transmission and reception.

The objective is to learn a joint policy $\pmb { \pi } = ( \pi _ { 1 } , \ldots , \pi _ { n } )$ that maximizes the expected discounted return:

$$
J ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ^ { t } \right] \mathrm { ~ , ~ }\tag{3}
$$

where $r ^ { t } = r _ { i } ^ { t }$ for all $i \in N$ is the shared team reward emitted by the environment at timestep t. All agents receive an identical scalar reward, reflecting the fully cooperative nature of the task.

## B. Centralized Training with Decentralized Execution (CTDE)

We use the CTDE paradigm [9]. During training, a single centralized critic receives the pooled observations and contexts of all agents as input:

$$
\mathcal { Q } ( \mathbf { s } , \mathbf { a } ) = Q ( [ o _ { 1 } \rVert c _ { 1 } , . . . , o _ { n } \rVert c _ { n } ] , [ a _ { 1 } \rVert m _ { 1 } , . . . , a _ { n } \rVert m _ { n } ] )\tag{4}
$$

where $\mathbf { s } = \left[ o _ { 1 } \rVert c _ { 1 } , \ldots , o _ { n } \rVert c _ { n } \right]$ concatenates each agent’s local observation and aggregated context, and ${ \bf a } =$ $[ a _ { 1 } \| m _ { 1 } , \ldots , a _ { n } \| m _ { n } ]$ concatenates all agents’ environment actions and messages. The critic is not used during execution time, where each agent operates using only its local observation $o _ { i }$ and context $c _ { i } .$ , making the operation fully decentralized.

## C. Communication Model

We model bandwidth-limited communication as prefix truncation of the message vector. Only the first k dimensions of a learned message are delivered, while the remaining dimensions are not available to the receiving agents.

Formally, given a message $m _ { i } \in \mathbb { R } ^ { d _ { m } }$ and prefix length $k \in$ $\{ 1 , \ldots , d _ { m } \}$ , the received message $\tilde { m } _ { i }$ is

$$
\tilde { m } _ { i , d } = \left\{ \begin{array} { l l } { m _ { i , d } , } & { d \leq k , } \\ { 0 , } & { d > k , } \end{array} \right. \quad d = 1 , \ldots , d _ { m } ,\tag{5}
$$

where $d$ denotes the index in the message vector.

The same k is applied symmetrically to all agents. We define the bandwidth fraction as $\rho = k / d _ { m } \in ( 0 , 1 ]$ . At the start of each episode, $\rho$ is sampled from a Beta distribution, with $k \geq 1$ enforced by the episode-level sampler (dynamic within-episode bandwidth schedules used at evaluation may reach $k = 0 )$ . This formulation gives a structural requirement on the message encoder. The earliest dimensions should carry the most task-relevant information, since they are the most likely to survive at low bandwidth. We therefore study how agents should order information across message dimensions so that any received prefix remains useful for coordination, rather than treating the channel as either fully available or completely unavailable.

## IV. ATTENTION-BASED HIERARCHICAL VARIATIONAL INFORMATION BOTTLENECK

Agents must learn to encode messages such that the first dimensions contain the most task-relevant information, since later dimensions may be truncated and lost entirely. Here, we present our approach to achieving this behavior. The core contribution is two-fold: (i) an autoregressive, attention-based message encoder that constructs each message dimension conditioned on all previous ones, creating a natural information priority ordering. And (ii) a hierarchical robustness loss that actively penalizes degradation in team performance under prefix truncation during training. Together, these make up the Attention-based Hierarchical Variational Information Bottleneck (AH-VIB) algorithm.

The autoregressive construction provides a structural guarantee as a received prefix is generated according to the same causal factorization as the corresponding prefix of the full message. It does not, by itself, guarantee that short prefixes preserve high task value. That property is encouraged separately by the hierarchical robustness loss. Thus, the two components address different failure modes. Autoregressive generation makes prefixes structurally coherent, whereas the hierarchical loss makes them useful for the task. This distinction is important when interpreting the ablations and the relative contributions of AH-VIB’s components.

## A. Variational Information Bottleneck

The Variational Information Bottleneck (VIB) [8] provides a principled framework for learning compressed representations that retain only task-relevant information. Rather than outputting a deterministic representation, VIB trains a stochastic encoder that maps its input to a distribution from which a sample is drawn at each forward pass. IMAC [7] applies this framework to multi-agent communication by applying VIB to message encoding such that agents learn to produce compact, informative messages under bandwidth constraints. We adopt this approach as the probabilistic foundation for our encoder, and extend it to address the prefix-truncation setting.

Concretely, each agent i encodes its input into a mean $\mu _ { d }$ and standard deviation $\sigma _ { d }$ for each message dimension $d ,$ from which a sample is drawn as:

$$
\begin{array} { r } { z _ { d } = \mu _ { d } + \sigma _ { d } \cdot \pmb { \varepsilon } , \qquad \pmb { \varepsilon } \sim \mathcal { N } ( 0 , 1 ) \ , } \end{array}\tag{6}
$$

giving the full message $m _ { i } = [ z _ { 1 } , \ldots , z _ { d _ { m } } ]$

The encoder is regularized by minimizing the Kullback–Leibler (KL) divergence from a standard normal prior across all dimensions and agents:

$$
\mathcal { L } _ { \mathrm { V I B } } = \sum _ { i = 1 } ^ { n } \sum _ { d = 1 } ^ { d _ { m } } D _ { \mathrm { K L } } \big ( \mathcal { N } ( \mu _ { d } , \pmb { \sigma } _ { d } ^ { 2 } ) \| \mathcal { N } ( \mathbf { 0 } , \mathbf { 1 } ) \big ) \enspace .\tag{7}
$$

Traditional VIB encoders predict all $( \mu _ { d } , \sigma _ { d } ^ { 2 } )$ simultaneously from a shared hidden state, which provides no intentional structure on how information is distributed across dimensions. Consequently, the first k dimensions are not necessarily optimized as a self-contained representation for every k. As a result, any given dimension may implicitly depend on information encoded in later dimensions, meaning a truncated prefix is not necessarily a coherent or selfsufficient representation. The order in which information is placed across message dimensions must therefore be learned.

## B. Autoregressive Message Encoding

To enforce message-order priority, we model the message dimensions autoregressively, so that each dimension can depend on those generated before it. This directly addresses the cross-dimensional dependency problem identified above. Each dimension $z _ { d }$ is conditioned on the previously generated prefix $\mathbf { z } _ { < d } = \left( z _ { 1 } , \ldots , z _ { d - 1 } \right)$ :

$$
q ( z _ { d } \mid \mathbf { z } _ { < d } , o _ { i } , c _ { i } ) , \qquad d = 1 , \ldots , d _ { m } .\tag{8}
$$

Since no dimension conditions on any later $z _ { \hat { d } }$ with $\hat { d } > d ,$ any truncated prefix remains self-consistent by construction.

We implement this using a multi-head attention mechanism. At each iteration over $d ,$ the encoder hidden state $h _ { i }$ is projected into attention space and combined with a position embedding to form the query, while the key-value context grows with each previously sampled dimension:

$$
\mathbf { q } _ { d } = W _ { \mathrm { p r o j } } h _ { i } + \mathbf { p o s } ( d ) \mathrm { ~ , ~ }\tag{9}
$$

$$
\mathbf { K } \mathbf { V } _ { d } = \left[ W _ { K V } h _ { i } , \ W _ { z } z _ { 1 } , \ . . . , \ W _ { z } z _ { d - 1 } \right] \ .\tag{10}
$$

![](images/307005f85aa941312ddab28ecbb6f8c2bb7dbdeb15870e4032c111bb3aad653a.jpg)  
Fig. 3. AH-VIB architecture. Top: Full agent diagram showing how observation o<sub>i</sub> and aggregated context c<sub>i</sub> are processed by the Contextualizerencoder MLP into hidden representation $[ o ^ { t } , c ^ { t } ]$ , which drives both the action head $\pi _ { \theta } ^ { a c t }$ to produce $a _ { i } ^ { \mathrm { n o r m } }$ and together with $a _ { i } ^ { \mathrm { n o r m } }$ , the autoregressive message encoder to produce $m _ { i } = \widehat { [ } \widetilde { z } _ { 1 } , \ldots , \widetilde { z } _ { d _ { m } } ] ^ { \top }$ . Bottom: Detail of the autoregressive message encoder. The dashed box is executed sequentially for $d \bar { = } 1 , \ldots , d _ { m }$ . At each step, the key-value context $\mathrm { K V } _ { d }$ is constructed by stacking a projection of $h _ { i }$ with projections of all previously sampled dimensions $\tilde { z } _ { 1 : d - 1 } ,$ growing by one token each iteration. The query q<sub>d</sub> is formed from $W _ { \mathrm { p r o j } } h _ { i }$ and position embedding $\scriptstyle \operatorname { p o s } ( d )$ . The multi-head attention output is concatenated with $h _ { i }$ and passed through the head MLP to produce $( \bar { \mu } _ { d } , \sigma _ { d } ^ { 2 } )$ , from which $\tilde { z } _ { d }$ is sampled and bounded by tanh. The sampled $\tilde { z } _ { d }$ is fed back into $\mathrm { K V } _ { d + 1 }$ and accumulated.

The attention output is concatenated with $h _ { i }$ and passed through a small head network to produce $( \mu _ { d } , \sigma _ { d } ^ { 2 } )$ ), from which $z _ { d }$ is sampled and bounded:

$$
z _ { d } = \mu _ { d } + \sigma _ { d } \cdot \pmb { \varepsilon } , \quad \pmb { \varepsilon } \sim \mathcal { N } ( 0 , 1 ) , \qquad \tilde { z } _ { d } = \operatorname { t a n h } ( z _ { d } ) ~ .\tag{11}
$$

Note that $W _ { z }$ is shared across all steps to reduce the number of parameters and improve generalization. The attention mechanism allows each dimension to focus on the most relevant previously generated dimensions, rather than receiving a fixed concatenation.

The full message $m _ { i } = [ \tilde { z } _ { 1 } , \dots , \tilde { z } _ { d _ { m } } ]$ is assembled sequentially and transmitted through the channel, where prefix truncation is applied before message transmission. Figure 3 illustrates the complete AH-VIB architecture, with the bottom panel illustrating a single autoregressive step.

## C. Hierarchical Robustness Loss

The autoregressive structure ensures representational consistency under truncation, but does not directly enforce strong task performance at reduced bandwidth. We address this with a hierarchical robustness loss that directly penalizes drops in team value under prefix truncation.

TABLE I  
TRAINING HYPERPARAMETERS USED IN THE EXPERIMENTS, SHARED ACROSS ALL ALGORITHMS UNLESS NOTED.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Environment &amp; Training</td><td></td><td>Optimisation</td><td></td><td>Exploration Noise</td><td></td></tr><tr><td>Total env. steps</td><td> $5 \times 1 0 ^ { 5 }$ </td><td>Actor LR</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>Initial noise  $\sigma _ { 0 }$ </td><td>0.20</td></tr><tr><td>Episode length T</td><td>200</td><td>Critic LR</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>Decay factor</td><td>0.80</td></tr><tr><td>Replay buffer size</td><td> $2 \times 1 0 ^ { 5 }$ </td><td>Discount γ</td><td>0.95</td><td>Decay over (steps)</td><td> $1 \times 1 0 ^ { 5 }$ </td></tr><tr><td>Batch size</td><td> $^ { 2 5 6 }$ </td><td>Target network τ</td><td>0.005</td><td>Min. noise  $\sigma _ { \mathrm { m i n } }$ </td><td>0.05</td></tr><tr><td>Random exploration steps</td><td>4,000</td><td>Actor hidden layers</td><td>[512, 256, 256, 128]</td><td></td><td></td></tr><tr><td>Update interval</td><td></td><td>Critic hidden layers</td><td>[512, 256, 256, 128]</td><td></td><td></td></tr><tr><td>Gradient steps / update</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random seeds</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Communication Channel</td><td></td><td>VIB &amp; Hier. Loss</td><td></td><td>Algorithm-Specific</td><td></td></tr><tr><td>Message dim.  $d _ { m }$ </td><td>32</td><td>VIB coeff  $. \beta$ </td><td>0.001</td><td>Attn. dim.  $d _ { \mathrm { a t t n } }$ </td><td>64</td></tr><tr><td>BW sampling (α, β)</td><td> $( 3 . 0 , 7 . 0 )$ </td><td>Hier. loss λ</td><td>0.05</td><td>Attention heads (AH-VIB)</td><td>4</td></tr><tr><td>Comm regimes</td><td>full / zero / beta</td><td>Hier. samples / update</td><td></td><td>CommNet rounds K</td><td>3</td></tr><tr><td>Loss probability</td><td>0.0 (1.0 in zero comms)</td><td> ${ p } _ { \mathrm { f u l l } }$  (hier. sampler)</td><td>1 0.10</td><td></td><td></td></tr><tr><td>Topology</td><td>broadcast</td><td>Stop-grad.  $Q _ { \mathrm { f u l l } }$ </td><td>√</td><td></td><td></td></tr></table>

During each actor update, we sample a truncation ratio ρ for the robustness loss and convert it to a prefix length k as:

$$
k = \mathrm { r o u n d } ( \rho d _ { m } ) \qquad \rho \sim q ( \rho ) ,\tag{12}
$$

where

$$
\begin{array} { r } { q ( \rho ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } p _ { \mathrm { f u l l } } , } \\ { \mathrm { B e t a } ( \alpha , \beta ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{13}
$$

This truncation sampler is independent of the actual channel bandwidth applied during environment interaction. By decoupling the hierarchical loss from the environment’s specific bandwidth dynamics, we encourage the encoder to learn a progressive message structure that generalizes across varying bandwidth conditions, rather than overfitting to the particular degradation schedule of the training environment.

We apply the prefix mask to the joint action vector and evaluate the centralized critic under truncation:

$$
Q _ { k } = Q ( \mathbf { s } , \tau _ { k } ( \mathbf { a } ) ) , \qquad Q _ { \mathrm { f u l l } } = Q ( \mathbf { s } , \mathbf { a } ) .\tag{14}
$$

The hierarchical loss penalizes the gap between the full and truncated team value using a hinge:

$$
\mathcal { L } _ { \mathrm { h i e r } } = \mathbb { E } _ { k } \left[ \operatorname* { m a x } \left( 0 , \overline { { Q } } _ { \mathrm { f u l l } } - Q _ { k } \right) \right] \ ,\tag{15}
$$

where $\overline { { Q } } _ { \mathrm { f u l l } }$ is a stop-gradient copy of $Q _ { \mathrm { f u l l } } ,$ , i.e., it is treated as a constant when computing gradients for ${ \mathcal { L } } _ { \mathrm { h i e r } }$ . This makes the loss increase $Q _ { k }$ toward the full-bandwidth reference, rather than reducing $Q _ { \mathrm { f u l l } }$ to satisfy the objective.

The complete actor loss for AH-VIB is:

$$
\mathcal { L } _ { \mathrm { a c t o r } } = \mathbb { E } [ - Q _ { \mathrm { f u l l } } + \beta \mathcal { L } _ { \mathrm { V I B } } + \lambda \mathcal { L } _ { \mathrm { h i e r } } ] \ .\tag{16}
$$

The three terms have complementary roles: $- Q _ { \mathrm { f u l l } }$ drives task performance at full bandwidth; ${ \mathcal { L } } _ { \mathrm { V I B } }$ enforces message compression; ${ \mathcal { L } } _ { \mathrm { h i e r } }$ enforces graceful degradation under truncation. Thus, the two components address different failure modes. Autoregressive generation makes prefixes structurally coherent, whereas the hierarchical loss makes them useful for the task. This distinction is important when interpreting the ablations and the relative contributions of AH-VIB’s components.

## V. EXPERIMENTS & RESULTS

## A. Environment

We evaluate all algorithms on a custom multi-agent inspection SLAM task implemented in the semcomms framework as a PettingZoo ParallelEnv [14]. A team of $n = 3$ agents explores a $5 0 \times 5 0$ occupancy grid containing three randomly placed $6 \times 4$ rectangular inspection objects. At each step, agents select a continuous 2D velocity action u ∈ $[ - 1 , 1 ] ^ { 2 }$ (maximum displacement: two cells); the resulting heading defines a $6 0 ^ { \circ }$ field-of-view sensor with 16 rays and a range of eight cells. Figure 1 illustrates the environment at the start of an episode and after 50 steps of exploration.

Each agent maintains its own personal occupancy grid, updated by ray-casting. Object cells occlude scans, so only surface cells can be revealed. A shared team grid (union of all personal grids) drives a team-shared reward with three additive terms:

$$
r ^ { t } = w _ { \mathrm { o b j } } \frac { n _ { \mathrm { o b j } } ^ { t } } { | \mathcal { O } | } + w _ { \mathrm { f r e e } } \frac { n _ { \mathrm { f r e e } } ^ { t } } { G ^ { 2 } } - \lambda _ { \mathrm { d u p } } c _ { \mathrm { d u p } } ^ { t } .\tag{17}
$$

Here, $n _ { \mathrm { o b j } } ^ { t }$ is the number of inspection-object surface cells newly revealed in the team grid, $| \mathcal { O } |$ is the total number of object cells (3 objects $\times 6 \times 4 = 7 2$ cells), $n _ { \mathrm { f r e e } } ^ { t }$ is the number of free cells newly revealed, $G = 5 0$ is the grid size, $c _ { \mathrm { d u p } } ^ { t }$ counts agent pairs within a proximity threshold of 3 cells of each other, $\lambda _ { \mathrm { d u p } } = 0 . 5 ,$ with weights $w _ { \mathrm { o b j } } = 1 . 0$ and $w _ { \mathrm { f r e e } } =$ 0.01. The second term is a deliberately small exploration bonus, so discovering unoccupied geometry is non-trivially incentivized but object scanning dominates.

Each agent receives a partial observation consisting of a $1 7 \times 1 7$ local patch from its personal occupancy grid (normalized to $0 / 0 . 5 / 1$ for unknown/free/occupied) concatenated with its own normalized pose $[ x / G , y / G ,$ , cos θ, sin θ]; the observation dimension is $1 7 ^ { 2 } + 4 = 2 9 3$ . Communication context is not part of the observation (all comm passes through the external channel module). Episodes run for $T = 2 0 0$ timesteps.

TABLE II  
BASELINE DESIGN COMPARISON.
<table><tr><td>Algorithm</td><td>VIB</td><td>AR</td><td>Hier.</td><td>Attn.</td></tr><tr><td>MADDPG</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>CommNet</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>VIB</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>HVIB</td><td>√</td><td>√</td><td>√</td><td>×</td></tr><tr><td>AH-VIB</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

The communication channel is configured as described in Section III-C, with synchronous broadcast, message dimension $d _ { m } = 3 2$ , and episode-level bandwidth sampling from Beta $( 3 . 0 , 7 . 0 )$ . Prefix truncation is applied uniformly to all agents. For the robustness loss, we use a separate truncation sampler with the same Beta parameters and $p _ { \mathrm { f u l l } } = 0 . 1 0$ . Furthermore, the training-time comm regime is ablated across three conditions: full (forced $\rho = 1 )$ , zero (’loss prob: 1’ hard drop), and beta (the default Beta-sampling truncation).

## B. Baselines

For clarity, we include both standard communication baselines and structured ablations of our own method. Alongside the full AH-VIB model, we include (i) a flat VIB-based communication model with simultaneous message generation and (ii) an autoregressive MLP-based variant, which we refer to as HVIB. These baselines provide progressively structured comparisons. The flat VIB baseline tests the effect of stochastic compression without ordered message generation. HVIB adds autoregressive generation and the hierarchical loss but replaces attention with an MLP. The comparison between HVIB and AH-VIB therefore evaluates whether attention provides benefits beyond autoregressive ordering and truncation-aware training.

We compare AH-VIB against four alternative approaches:

• MADDPG [9]: a deterministic CTDE baseline with learned continuous messages, but without a stochastic bottleneck, explicit dimensional ordering, or robustness loss.

• CommNet [4]: a continuous communication baseline based on differentiable mean pooling over $K = 3$ rounds, with no explicit bottleneck or dimensional ordering.

• IMAC-style VIB [7]: a flat variational communication baseline inspired by IMAC, where all message dimensions are generated simultaneously from a shared encoder. We implement only the communication component relevant to our setting.

• HVIB: our autoregressive MLP ablation. HVIB retains the stochastic bottleneck and hierarchical robustness loss, but replaces the attention-based autoregressive message mechanism of AH-VIB with a standard perdimension MLP-based construction.

![](images/260813a9438f5ecef227062a71cb20a40c6e582e85656b9c24e5f1be69bb18d8.jpg)  
Fig. 4. Mean episodic return over training steps for AH-VIB, HVIB, VIB, MADDPG, and CommNet, averaged across three seeds. Shaded regions denote ±1 standard deviation. All methods are trained under the same variable-bandwidths, with $\rho \sim \mathrm { B e t a } ( 3 . 0 , 7 . 0 )$

## C. Evaluation Metrics

a) Episode reward: Our first metric is the mean undiscounted episode return $\begin{array} { r } { \bar { R } = \sum _ { t = 0 } ^ { T - 1 } r ^ { t } } \end{array}$ , where higher (less negative) values indicate better performance. For training curves, returns are smoothed across episodes for visual clarity.

b) Robustness at constrained bandwidth: At inference time, each trained policy is evaluated at fixed bandwidth fractions $\rho ,$ with truncation cutoff $k = \mathrm { r o u n d } ( \rho d _ { m } )$ . This produces a reward–bandwidth curve that shows team performance when communication is restricted.

c) Intra-episode bandwidth dynamics: To test outof-distribution robustness, we also evaluate policies under within-episode bandwidth schedules, where $\rho$ either decreases from 1.0 to 0.0 or increases from 0.0 to 1.0. Performance is measured by episode reward.

## D. Training Configuration

All algorithms share a common base training configuration. Each method is trained for 500,000 environment steps with episode length T = 200, replay buffer size $2 \times 1 0 ^ { 5 }$ , batch size 256, and random exploration for the first 4,000 steps. Updates are then performed every 50 environment steps using 5 gradient steps per update. Actor and critic learning rates are $3 \times 1 0 ^ { - 4 }$ and $\bar { 3 } \times 1 \bar { 0 } ^ { - 5 }$ , respectively, with discount factor $\gamma = 0 . 9 5$ and target-network update rate $\tau = 0 . 0 0 5$

For the VIB-based variants, we use bottleneck coefficient $\beta = 0 . 0 0 1$ . For the hierarchical variants, we use robustnessloss weight $\lambda = 0 . 0 5 .$ , a single sampled prefix per update, and a stop-gradient target on $Q _ { \mathrm { f u l l } }$ . The full hyperparameter details with algorithm-specific configurations are given in Table I.

TABLE III  
EPISODE RETURN AT SELECTED BANDWIDTH FRACTIONS (MEAN ± STANDARD DEVIATION). BOLD INDICATES THE BEST MEAN AND LOWEST STANDARD DEVIATION IN EACH COLUMN.
<table><tr><td></td><td>ρ=0.10</td><td> $\rho { = } 0 . 2 5$ </td><td> $\rho { = } 0 . 7 5$ </td><td>ρ=1.0</td></tr><tr><td>AH-VIB</td><td> ${ \bf - 0 . 2 \pm 0 . 5 }$ </td><td> ${ \bf - 0 . 2 \pm 0 . 5 }$ </td><td> $- 0 . 4 \pm 1 . 9$ </td><td> $- 0 . 4 \pm 2 . 2$ </td></tr><tr><td>VIB</td><td> $- 7 . 7 \pm 2 4 . 3$ </td><td> $- 6 . 2 \pm 2 1 . 6$ </td><td> $- 6 . 1 \pm 2 1 . 6$ </td><td> $- 7 . 3 \pm 2 3 . 0$ </td></tr><tr><td>MADDPG</td><td> $- 5 . 4 \pm 1 6 . 7$ </td><td> $- 5 . 3 \pm 1 6 . 9$ </td><td> $- 1 3 . 7 \pm 2 9 . 2$ </td><td> $- 1 . 1 \pm 8 . 4$ </td></tr><tr><td>COMMNET</td><td> $- 1 . 8 \pm 1 0 . 1$ </td><td> $- 2 . 1 \pm 1 2 . 2$ </td><td> ${ \bf - 0 . 1 \pm 0 . 3 }$ </td><td> ${ \bf - 0 . 1 \pm 0 . 3 }$ </td></tr></table>

![](images/a436216f329fb62a9c7a75c1e559c8aa19da49c462d766fbc9fe623516ff7a4e.jpg)  
Fig. 5. Episode return distributions at selected bandwidth fractions $\rho \in \{ 0 . 1 0 , 0 . 2 5 , 0 . 7 5 , 1 . 0 0 \}$ for AH-VIB, VIB, MADDPG, and CommNet. Violin width indicates return density, and horizontal bars mark the mean.

## E. Training Dynamics

We first examine whether the additional bottleneck and robustness terms in AH-VIB destabilize training or improve convergence. Figure 4 shows the mean episodic return over training steps, with shaded regions indicating one standard deviation.

AH-VIB, HVIB, VIB, and MADDPG converge to similar final returns, indicating that the VIB and hierarchical robustness objectives do not impede training. CommNet converges more slowly and remains more variable. Since asymptotic mean return only weakly separates the stronger methods, we focus on reliability under restricted bandwidth.

a) Statistical Reliability: To formalize the observation of tighter return distributions, we perform bootstrap resampling $( N = 1 0 , 0 0 0 )$ and compare the standard deviation of episode returns against the closest VIB-based baseline via $\Delta \sigma = \sigma _ { \mathrm { A H - V I B } } - \sigma _ { \mathrm { V I B } }$ , where $\Delta \sigma < 0$ indicates more consistent behavior. Across all evaluated bandwidth fractions $\rho \in \{ 0 . 1 0 , 0 . 2 5 , 0 . 7 5 , 1 . 0 0 \}$ , ∆σ is strongly negative (between −19.7 and −23.8) and the bootstrap probability that AH-VIB has lower standard deviation than VIB is 100% at every bandwidth. The same holds against MADDPG $( P \ge$ 86.8% at all bandwidths). Against CommNet, AH-VIB is significantly more consistent at the two tightest bandwidths $( P \ge 9 9 . 3 \%$ at $\rho \in \{ 0 . 1 0 , 0 . 2 5 \} _  \mathrm  \it  \ /  \ /  \ / { \ / { \ / { \ / { \ / { \ / { \ / { \ / { \ / { \ / { \ / { \ / { \ / { \ / } { \ / { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / { \ / } { \ / } { \ / { \ / } { \ / } { \ / { \ / } { \ / } { \ / { \ / } { \ / } { \ / { \ / } { \} } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }$ ), while CommNet is the most consistent method at $\rho \ge 0 . 7 5$ . This supports the concrete claim that AH-VIB’s advantage under bandwidth constraints is improved reliability — and that this advantage is most pronounced precisely when communication is scarce.

## F. Performance Under Variable Bandwidth

We evaluate inference performance under fixed bandwidth budgets. Figure 5 shows episode-return distributions at $\rho \in \{ 0 . 1 0 , 0 . 2 5 , 0 . 7 5 , 1 . 0 0 \}$ , and Table III reports the corresponding mean and standard deviation.

AH-VIB achieves the highest mean return at the two tightest bandwidth settings, $\rho = 0 . 1 0$ and $\rho = 0 . 2 5$ , where reliable use of a short message prefix is most important. At the larger bandwidth fractions, $\rho = 0 . 7 5$ and $\rho = 1 . 0 0 ,$ CommNet obtains a slightly higher mean return and lower variation. This suggests that when most or all of the message is available, the additional structure imposed by AH-VIB is less important.

The principal advantage of AH-VIB is therefore most evident under restricted communication, often encountered in harsh environmental conditions. As shown in Figure 5, AH-VIB produces a compact return distribution with short lower-return tails at $\rho = 0 . 1 0$ and $\rho = 0 . 2 5$ . In contrast, VIB and MADDPG exhibit substantially broader distributions and more severe low-return episodes under these constrained conditions. CommNet remains competitive at larger bandwidth fractions, but does not provide the same mean-return performance as AH-VIB when only a small message prefix is received.

Table III shows the same bandwidth-dependent pattern. AH-VIB provides the best mean return and lowest return variation at the two tightest bandwidth settings, whereas CommNet performs best when bandwidth is less restrictive. Thus, AH-VIB is not intended to uniformly dominate every communication method at full bandwidth; rather, it improves the reliability of coordination when the available bandwidth is limited.

## G. Per-Dimension KL Divergence

To better understand how the VIB-based methods use the information bottleneck, we analyse the per-dimension KL divergence between the encoder posterior and the unit-Gaussian prior:

$$
D _ { \mathrm { K L } } \big ( \mathcal { N } ( \mu _ { d } , \pmb { \sigma } _ { d } ^ { 2 } ) \big | \big | \mathcal { N } ( 0 , 1 ) \big ) .\tag{18}
$$

The values are averaged over agents and timesteps within each evaluation episode, yielding one KL profile per episode.

We report this diagnostic analysis on MPE Simple Spread, which provides a controlled setting for comparing latent message usage across the VIB-based methods. All profiles are evaluated at full bandwidth $( \rho = 1 . 0 )$ , so they reflect native dimension usage rather than the direct effect of masking. While the main task-level evaluation in this paper uses the object-inspection and occupancy-mapping task, the KL analysis is included as complementary evidence about the learned communication representations.

![](images/0f0608c47c69d118625804fa3b3815d8f7758667b9236360cf6ccb3793ec2def.jpg)  
Fig. 6. Mean per-dimension KL divergence between the encoder posterior and the unit-Gaussian prior for AH-VIB, HVIB, and VIB on MPE Simple Spread. Results are averaged over evaluation episodes at full bandwidth $( \rho = 1 . 0 ) ;$ error bars show the standard error of the mean (SEM). Vertical dashed lines indicate the prefix lengths corresponding to $\rho = 0 . 1 0 , \rho = 0 . 2 5$ and $\rho = 0 . 5 0$ for visual reference.

Figure 6 shows that AH-VIB obtains lower mean KL divergence than HVIB and VIB across message dimensions, indicating a more compact latent communication representation [7]. AH-VIB also exhibits tighter SEM bands, indicating more stable dimension-wise latent usage across evaluation episodes.

## H. Discussion and Future Work

While AH-VIB demonstrates superior performance under bandwidth constraints the simple communication setup presents a limitation. Furthermore, agents broadcast messages at every timestep, no advanced communication media access protocol, and scheduling mechanisms and resource allocation are integrated. The communication channel model is also simplified without considering packet loss, propagation delay, and asymmetric links, which could have a significant impact for multi-agent collaboration, in particular, in underwater robot collaboration. In addition, the inspection environment remains a simplified simulation. It uses rectangular inspection objects, idealized ray-casting sensors, simplified motion dynamics, and perfect localization. These assumptions limit the conclusions that can be drawn about transfer to physical multi-robot inspection systems.

A natural direction for future work is to combine taskoriented communication with advanced scheduling and resource allocation, so that agents can learn both what information should appear early in a message and which agent can communicate at which slot. Another direction is to evaluate the method on more challenging tasks and more realistic multi-robot settings with practical channel modeling.

## VI. CONCLUSION

We introduced AH-VIB, a novel autoregressive variational communication method for multi-agent coordination under variable bandwidth and prefix truncation. AH-VIB combines causal message generation with a hierarchical robustness loss that promotes retention of task performance when only a message prefix is received.

On our cooperative object-inspection task, AH-VIB achieves competitive mean return while improving the consistency of performance under bandwidth restriction. These results support progressive message encoding as a useful design principle for bandwidth-constrained multi-robot communication.

## REFERENCES

[1] M. Dorigo, G. Theraulaz, and V. Trianni, “Reflections on the future of swarm robotics,” Science robotics, vol. 5, no. 49, p. eabe4385, 2020.

[2] M. Saboia, L. Clark, V. Thangavelu, J. A. Edlund, K. Otsu, G. J. Correa, V. S. Varadharajan, A. Santamaria-Navarro, T. Touma, A. Bouman et al., “ACHORD: Communication-aware multi-robot coordination with intermittent connectivity,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10 184–10 191, 2022.

[3] J. Gielis, A. Shankar, and A. Prorok, “A critical review of communications in multi-robot systems,” Current robotics reports, vol. 3, no. 4, pp. 213–225, 2022.

[4] S. Sukhbaatar, R. Fergus et al., “Learning multiagent communication with backpropagation,” Advances in Neural Information Processing Systems, vol. 29, 2016.

[5] A. Das, T. Gervet, J. Romoff, D. Batra, D. Parikh, M. Rabbat, and J. Pineau, “TarMAC: Targeted multi-agent communication,” in Proceedings of the International Conference on Machine Learning, 2019, pp. 1538–1546.

[6] K. Zhang, Z. Yang, and T. Bas¸ar, “Multi-agent reinforcement learning: A selective overview of theories and algorithms,” in Handbook of Reinforcement Learning and Control. Springer, 2021, pp. 321–384.

[7] R. Wang, X. He, R. Yu, W. Qiu, B. An, and Z. Rabinovich, “Learning efficient multi-agent communication: An information bottleneck approach,” in Proceedings of the International Conference on Machine Learning, 2020, pp. 9908–9918.

[8] A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy, “Deep variational information bottleneck,” arXiv preprint arXiv:1612.00410, 2016.

[9] R. Lowe, Y. WU, A. Tamar, J. Harb, O. Pieter Abbeel, and I. Mordatch, “Multi-agent actor-critic for mixed cooperative-competitive environments,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[10] J. M. Shapiro, “Embedded image coding using zerotrees of wavelet coefficients,” IEEE Transactions on Signal Processing, vol. 41, pp. 3445–3462, 2002.

[11] O. Rippel, M. Gelbart, and R. Adams, “Learning ordered representations with nested dropout,” in International Conference on Machine Learning. PMLR, 2014, pp. 1746–1754.

[12] A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder, K. Chen, S. Kakade, P. Jain et al., “Matryoshka representation learning,” Advances in Neural Information Processing Systems, vol. 35, pp. 30 233–30 249, 2022.

[13] L. Esterle and D. W. King, “Loosening control—a hybrid approach to controlling heterogeneous swarms,” ACM Transactions on Autononous and Adaptive Systems, vol. 16, no. 2, Mar. 2022.

[14] J. Terry, B. Black, N. Grammel, M. Jayakumar, A. Hari, R. Sullivan, L. S. Santos, C. Dieffendahl, C. Horsch, R. Perez-Vicente et al., “PettingZoo: Gym for multi-agent reinforcement learning,” Advances in Neural Information Processing Systems, vol. 34, pp. 15 032–15 043, 2021.