PREPRINT

# Map-Distribution Mixture-of-Experts for Multi-Agent Pathfinding

Fumimaro Suzuki<sup>1,2</sup>, Keisuke Okumura<sup>2</sup>

<sup>1</sup>University of Tsukuba, Tsukuba, Japan

<sup>2</sup>National Institute of Advanced Industrial Science and Technology (AIST), Tokyo, Japan s2312431@u.tsukuba.ac.jp, okumura.k@aist.go.jp

## ABSTRACT

Designing capable neural policies for multi-agent pathfinding (MAPF), which map local observations to action distributions, represents an important challenge in multi-agent learning, with practical relevance to scalable collective automation problems such as warehouse logistics. In contrast to prior studies focusing primarily on individual policy design, this study explores a meta-level Mixture-of-Experts (MoE) approach to learning-based MAPF. Specifically, given a collection of compact yet high-performing experts specialised for diferent map distributions, our architecture learns a router that selectively fuses their outputs according to each agent’s local situation. We instantiate and validate this MoE approach with a commonly used graph neural network-based policy as well as a newly developed recursive action-token refinement model. Combined with supervised and reinforcement learning fine-tuning of the experts, our experiments show that MoE generally improves upon the pretrained policy and outperforms existing leading methods. At the same time, our findings reveal that appropriately trained, parameter-matched monolithic models still outperform MoE, highlighting a nuanced role for map-distribution MoE for MAPF.

Code – https://github.com/romanohu/MDMoE-MAPF

## 1 Introduction

Multi-agent pathfinding (MAPF) has emerged as a challenging testbed for large-scale multi-agent learning, requiring dense coordination among agents, while also being closely connected to real-world applications such as logistics automation. Among the various paradigms for integrating learning into planning, one central topic is the development of capable neural MAPF policies, in which each agent selects actions based on local observations and interactions [1]. In fact, various policy architectures have been developed, together with eficient training pipelines based primarily on reinforcement learning (RL) and imitation learning (IL). Recent studies have further shown that integrating neural MAPF policies with search can outperform purely search-based MAPF solvers in certain problem regimes [2, 3].

Existing architectures, however, largely rely on monolithic policies, where a single policy is expected to handle all situations encountered during execution. In contrast, Mixture-of-Experts (MoE) architectures [4] have proven efective across other domains. At its core, MoE combines multiple specialised experts through a learned mechanism that selectively uses their outputs according to the given input. A notable example is the development of large language models (LLMs), where modern architectures increasingly adopt sparse MoE to eficiently leverage specialised experts [5, 6]. Related ideas of combining specialised skills have also been explored in robotics, enabling increasingly versatile and complex behaviours [7, 8].

This naturally raises the question: can we exploit the power ofMoEfor neural MAPF policies, and ifso, how? This work addresses this question through a systematic investigation.

More precisely, two key considerations arise: (i) how to prepare the experts and (ii) how to combine them. For the former, inspired by the observation in [3] that fine-tuning a neural MAPF policy for a specific map can substantially enhance its performance, we prepare a collection of experts specialised for diferent map distributions. To this end, we adopt a training strategy inspired by recent LLM development: starting from a pretrained base model, we perform supervised fine-tuning (SFT) for each target map distribution, followed by RL fine-tuning using Group Relative Policy Optimisation (GRPO) [9]. For the latter, we learn a routing function that takes the same observations as the experts and assigns agent-wise weights to fuse their outputs, as shown in Fig. 1. Through this mechanism, the model can selectively leverage diferent experts according to the local geometric features encountered at test time.

![](images/123769af7a949c4614e19506c36d2de084adceb4faf53803271af8ec58c78dd5.jpg)  
Figure 1: MoE framework. A learned router fuses map-distribution experts taking each agent’s local observation.

We show that this simple recipe works efectively with two distinct neural architectures for the expert policies. The first is a graph neural network (GNN)-based architecture called MAGAT (Message-Aware Graph Attention Network) [3, 10], which has served as a common architectural backbone in the literature. The second is our newly developed lightweight architecture, ReCAT (Recursive Collective Action Token Network), which recursively refines action-token representations through GNN-based token interactions within and between agents. In particular, ReCAT-MoE outperforms existing leading methods [11, 12], despite having fewer trainable parameters, across both in-distribution and out-of-distribution scenarios. Meanwhile, applying the same SFT+RL recipe to parameter-matched monolithic counterparts produces remarkably strong policies, which remain competitive with or even outperform their MoE counterparts. Together, these results demonstrate both the efectiveness and limitations of map-distribution MoE, revealing a nuanced role for this previously under-explored paradigm in neural MAPF.

The remainder of this paper is organised as follows. We first introduce the problem formulation and basic assumptions underlying neural MAPF policies in Sec. 2. We then present map-wise expert training (Sec. 3.1), MoE training (Sec. 3.2), and ReCAT (Sec. 4), followed by their empirical evaluation in Sec. 5. Finally, we position our study within the broader literature and conclude the paper.

## 2 Preliminaries

## 2.1 Problem Formulation

We consider a commonly used MAPF formulation [13], defined by a set of agents $A = \{ 1 , 2 , \dots , n \}$ operating on a four-connected grid $G = ( V , E )$ , where each agent $i \in A$ has distinct start and goal vertices, denoted by $s _ { i } , g _ { i } \in V$ . At each timestep, each agent either stays at its current vertex or moves to an adjacent vertex. Vertex and edge collisions are prohibited; that is, two agents cannot occupy the same vertex at the same timestep or traverse the same edge in opposite directions within one timestep. The objective is to derive a collision-free path $[ v _ { i } ^ { 0 } = s _ { i } , v _ { i } ^ { 1 } , \ldots , v _ { i } ^ { T } = g _ { i } ]$ for each agent $i \in A$ , where the plan length $T$ is called makespan. Equivalently, using a configuration $Q \in V ^ { n }$ that represents the locations of all agents, MAPF seeks a sequence of transitionable collision-free configurations that moves every agent from its start to its goal. The solution cost is measured by the sum-of-costs (SoC), defined as the sum of the travel times of all agents until they reach their goals and remain there. Following common practice, we use SoC normalised by $\textstyle \sum _ { i \in A }$ dist $( s _ { i } , g _ { i } )$ as cost, where dist $( u , v )$ denotes the length of a shortest path between two vertices $u , v \in V$

## 2.2 Neural MAPF Policies

This study regards MAPF as a collective sequential decision-making problem, where each agent i receives a local observation $o _ { i }$ from the current configuration Q and selects its action according to the action distribution $\pi ( o _ { i } )$ over five possible actions, including staying in place. Our objective is then to learn a policy π that solves MAPF while achieving low SoC. Such a policy is typically trained via either RL or IL from expert search-based MAPF planners. Many policy architectures have been developed, with representative examples including PRIMAL [14], MAPF-GPT [11, 15], and HMAGAT [12]; see [1] for a comprehensive review. Beyond improving individual policy architectures, this study explores an orthogonal, architecture-agnostic direction: how to efectively construct experts for target domains, and fuse them through MoE.

## 2.2.1 Observation

While the proposed MoE framework itself does not assume a specific form of observation, following common practice in the literature $[ 3 , 1 0 , 1 4 ] .$ , we use an observation tensor $o _ { i }$ centred around agent i’s current position $Q [ i ] \in V$ , with shape $4 \times ( 2 R _ { \mathrm { o b s } } + 1 ) ^ { 2 }$ , where $R _ { \mathrm { 0 b s } } \in \mathbb { N } _ { > 0 }$ determines the field-of-view (FOV) size. The four channels represent: (i) the presence of obstacles, (ii) that of other agents, (iii) a projection of the goal direction, all represented by binary values, and (iv) cost-to-go values, where the value is normalised as $( \mathsf { d i s t } ( v , g _ { i } ) - \mathsf { d i s t } ( Q [ i ] , g _ { i } ) ) / ( 2 R _ { \mathrm { o b s } } )$ for each location $v \in V$ within the FOV.

## 2.2.2 GNN-based Policy and Communication

To validate the eficacy of MoE, this study conducts expert fusion over two distinct neural MAPF policy architectures. One is a newly developed architecture designed to achieve high performance with a compact policy, described in detail in Sec. 4, while the other follows a standard GNN-based policy, MAGAT [3, 10], as described below.<sup>1</sup>

Given a configuration $Q ,$ , MAGAT assumes a communication graph $G _ { \mathrm { c o m m } } .$ , where each vertex corresponds to an agent and an edge $( i , j )$ exists if agents i and $j$ are within a communication radius $R _ { \mathrm { c o m m } } \in \mathbb { N } ^ { + }$ of each other. MAGAT then consists of (i) a CNN-based encoder that maps the observation $o _ { i }$ to an initial latent embedding $z _ { i } ^ { 0 }$ , (ii) L communication layers that perform GNN message passing over $G _ { \mathrm { c o m m } } .$ , where each layer l produces $z _ { i } ^ { l }$ from the embeddings of the previous layer, and (iii) a multilayer perceptron (MLP) decoder that maps $z _ { i } ^ { L }$ to an action distribution. The communication layers use a three-dimensional edge feature $\omega _ { i j }$ , consisting of the relative position in the x- and y-directions and the Manhattan distance between the corresponding agents.

## 2.2.3 Collision Shielding

A neural MAPF policy alone outputs an action distribution, leaving a design choice in how to decode it into collision-free actions. Earlier work stochastically samples actions from the distribution and then resolves inter-agent collisions by simply stopping one of the conflicting agents, an approach known as naive collision shielding (CS) [17, 18]. Meanwhile, recent studies have shown that leveraging PIBT [19] to sample collision-free actions substantially improves the learned policy, an approach called CS-PIBT [2, 3, 12, 20, 21]; we mainly follow this strategy.

PIBT is a popular MAPF algorithm underlying modern scalable MAPF planners in both learning-free and learning-based settings [2, 12, 22]. It receives a configuration and an ordered action list for each agent, called a preference, and generates a collision-free next configuration while respecting the agent-wise preferences.

The original PIBT constructs preferences using the cost-to-go values, $\mathsf { d i s t } ( \cdot , g _ { i } )$ , while CS-PIBT instead uses the learned policy for this purpose by either stochastically or deterministically sampling an ordered action list according to the assigned action probabilities.

## 3 MoE over Map-Specific Experts

A key consideration in constructing an MoE is how to prepare a diverse and complementary set of expert policies. Two general strategies are commonly used: (i) training the experts first and subsequently learning a routing function to combine them, and (ii) jointly training the experts and router. We adopt the first strategy, motivated by prior work [3] showing that supervised fine-tuning of a pretrained policy on one target map can substantially improve its performance on that map. Following this insight, we first train a base policy and then independently fine-tune copies of it into a collection of experts, each specialised for a particular map distribution, before learning the router that combines their outputs. Such a training recipe has been used in MoE-based LLMs [23].

Based on common benchmark categorisations [18, 24], this study considers four distinct map distributions: Maze, Warehouse, Room, and Random. We select their instance configurations while accounting for dificulty, such that they provide useful training signals for high-performing policies. These map distributions are currently selected manually, leaving their automatic selection to future work.

## 3.1 Expert Training

Starting from a pretrained policy, our reproducible fine-tuning pipeline for each target map distribution, applicable across diferent neural architectures, consists of supervised fine-tuning (SFT) followed by RL-based fine-tuning.

## 3.1.1 Pretraining and Supervised Fine-Tuning (SFT)

We pretrain a policy to imitate teacher trajectories generated by an expert MAPF solver. All agents share the policy parameters $\theta ,$ and training minimises the cross-entropy between the predicted action distributions and the teacher actions, as commonly adopted in IL-based MAPF policies. The resulting parameters $\theta _ { 0 }$ provide a common initialisation for adaptation to the target map distributions.

Next, for each target map distribution k, we collect additional teacher trajectories by solving instances sampled from that distribution by the expert MAPF solver. We initialise the corresponding expert with $\theta _ { k } = \theta _ { 0 }$ and fine-tune it on these trajectories using the same cross-entropy objective.

## 3.1.2 Reinforcement Learning Fine-Tuning

While SFT alone provides decent performance, we empirically observe that RL fine-tuning further improves each SFT expert $\pi _ { k }$ using rewards from its own rollouts on the target map distribution. Our approach consists of two key components: PIBT-based safeguarded rollouts and GRPO-based [9] policy updates.

Rollout Procedure For each instance, we collect B rollouts from the same initial configuration, ending each rollout when all agents reach their goals or the timestep limit $T _ { \mathrm { m a x } }$ is reached. At each timestep, CS-PIBT [20] converts the policy’s output distributions into a collision-free joint move. For each agent, it samples an ordering of legal actions without replacement according to the policy probabilities, which serves as the preference ordering for PIBT. PIBT then sequentially attempts to assign compatible moves according to these preferences, trying subsequent candidates when necessary. For the policy update, we record the prefix of each ordering up to the last action examined; i.e., $\boldsymbol { u } _ { t , i } = [ \boldsymbol { v } ^ { ( 1 ) } , \boldsymbol { v } ^ { ( 2 ) } , \dots , \boldsymbol { v } ^ { ( m ) } ]$ denotes the sequence of candidate vertices attempted for agent i at timestep t during PIBT. This prefix is subsequently used to compute the likelihood for the policy update.

![](images/60955a52248912746ca128fdfffe36a6bf5df3af23af30a94e16ba75763897f5.jpg)  
Figure 2: ReCAT architecture. Following the spatial encoding (left), an action token is refined through message passing from other tokens of agent i and matching action tokens of neighbouring agents (right). Tensor shapes use $H = W = 2 R _ { \mathrm { o b s } } + 1$ , C feature channels, token dimension d, and action embedding dimension $d _ { a }$

Reward Design Our reward design prioritises solving the instance, with secondary terms favouring shorter successful rollouts and a larger fraction of agents reaching their goals in unsuccessful rollouts. More concretely, for rollout b, the reward is computed based on its length $T _ { b }$ and the fraction $q _ { b }$ of agents that have reached their goals at termination, with a hyperparameter $\eta \in \mathbb { R } ^ { + }$

$$
R _ { b } = \left\{ \begin{array} { l l } { 1 + \eta ( 1 - T _ { b } / T _ { \mathrm { m a x } } ) , } & { \mathrm { i f ~ a l l ~ a g e n t s ~ r e a c h ~ t h e i r ~ g o a l s , } } \\ { - 1 + \eta q _ { b } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

Following GRPO, we next compute an advantage $A _ { b } = ( R _ { b } - \bar { R } ) / \sigma _ { R } .$ where R¯ and $\sigma _ { R }$ are the mean and standard deviation of rewards across the B rollouts for the same instance. This advantage is shared by all recorded decisions in rollout b.

Preference Likelihood To use these rollout-level advantages for policy optimisation, we next quantify the likelihood of each recorded preference under the policy. For agent i at timestep t, let $P _ { \theta } ( u _ { t , i } \mid o _ { t , i } )$ denote the likelihood of the recorded prefix $u _ { t , i }$ given observation $o _ { t , i }$ . Since the preference ordering is sampled without replacement, this likelihood follows the Plackett–Luce model [25], given by the product of successive sampling probabilities, each renormalised over the remaining legal actions. We compare this likelihood under the current parameters θ and the rollout parameters $\theta _ { \mathrm { o l d } }$ through the importance ratio,

$$
r _ { t , i } = \frac { P _ { \theta } ( u _ { t , i } \mid o _ { t , i } ) } { P _ { \theta _ { \mathrm { o l d } } } ( u _ { t , i } \mid o _ { t , i } ) } .\tag{2}
$$

Loss Design We minimise a clipped policy loss [26] with a Kullback–Leibler (KL) penalty relative to the frozen supervised expert $\pi _ { \mathrm { r e f } }$ and an entropy bonus, which prevents the policy from becoming deterministic too early, as commonly adopted in PPO training, with hyperparameters $\epsilon , \beta ,$ and α, as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R L } } ( \theta ) = { \bf \mathcal { \nu } } - \mathbb { E } [ \operatorname* { m i n } ( r _ { t , i } A _ { b } , \mathrm { c l i p } ( r _ { t , i } , 1 - \epsilon , 1 + \epsilon ) A _ { b } ) ] + \beta \mathbb { E } [ D _ { \mathrm { K L } } ( \pi _ { \mathrm { r e f } } | | \pi _ { \theta } ) ] - \alpha \mathbb { E } [ \mathcal { H } ( \pi _ { \theta } ) ] . } \end{array}\tag{3}
$$

The KL divergence and entropy are evaluated at $o _ { t , i }$ after restricting the action distributions to legal actions.   
The expectations are taken over all recorded agent–timestep pairs (i, t) across the B rollouts.

## 3.2 Expert Fusion

Through the training pipeline in Sec. 3.1, suppose that we obtain K expert policies $\{ \pi _ { \theta _ { 1 } } , \ldots , \pi _ { \theta _ { K } } \}$ . For agent i at timestep t, expert k produces a vector of five action logits $z _ { t , i } ^ { ( k ) }$ , which would normally be converted into an action distribution via softmax. We instead combine these specialised policies through an observation-based MoE, as illustrated in Fig. 1.

Router Design To weight the expert outputs, an MLP router $f _ { \phi }$ maps the flattened local observation $o _ { t , i }$ for agent i at timestep t to one score per expert. A softmax with temperature τ converts the scores into mixing weights $w _ { t , i } = \mathrm { s o f t m a x } ( f _ { \phi } ( o _ { t , i } ) / \tau )$ . The k-th component $w _ { t , i , k }$ weights the logits of expert $k ,$ and the weighted sum of the logits is converted to the action distribution with:

$$
\pi _ { \phi } ( o _ { t , i } ) = \mathrm { s o f t m a x } \left( \sum _ { k = 1 } ^ { K } w _ { t , i , k } z _ { t , i } ^ { ( k ) } \right) .\tag{4}
$$

This defines our MoE policy, where the contribution of each expert is determined independently for each agent and timestep from its local observation.

Training Once the experts have been trained, we freeze their parameters and optimise only the router parameters ϕ on the concatenation of the supervised fine-tuning datasets across all map distributions. We train the combined policy using the same cross-entropy objective as in supervised fine-tuning, with the teacher action labels. Optionally, we can further fine-tune the router and experts jointly with RL, following the same procedure as in Sec. 3.1.2.

## 4 Lightweight yet Capable Architecture: ReCAT

Our MoE design is highly flexible, allowing, for example, experts with diferent architectures to be combined within a single policy. However, this flexibility comes at the cost of increased inference overhead. Unlike typical MoE architectures used in LLMs, where only a subset of experts is activated for each token, our design usually requires all experts to be evaluated for every agent, as each expert may employ its own interaction modelling among agents. Consequently, the inference cost grows with the number and complexity of experts, making compact yet capable expert architectures particularly desirable.

This motivates us to develop a new architecture that remains compact yet highly performant. To this end, this section introduces Recursive Collective Action Token Network (ReCAT), illustrated in Fig. 2, which integrates prominent architectural components that have proven efective across diferent domains. As indicated by its name, ReCAT represents each candidate action as a token. A spatial encoder first encodes the local observation into each action token, after which a recurrent GNN iteratively refines these tokens by exchanging messages both within and between agents. Finally, a linear projection converts the refined tokens into action logits.

## 4.1 Spatial Encoding

The spatial encoder transforms each observation $o _ { i }$ into a C-channel feature map while preserving its spatial resolution, thereby retaining the correspondence between individual FOV cells and their feature representations. We first apply a $3 \times 3$ convolution, followed by group normalisation (GN) [27], which normalises each agent’s features independently of other agents and the batch size, and a SiLU activation [28, 29], to obtain the initial feature map $\bar { F } _ { i } ^ { ( 0 ) }$

$$
F _ { i } ^ { ( 0 ) } = \mathrm { S i L U } ( \mathrm { G N } ( \mathrm { C o n v } _ { 3 \times 3 } ( o _ { i } ) ) ) .\tag{5}
$$

Inspired by lightweight CNN architectures [30], our encoder next applies $B _ { \mathrm { e n c } }$ residual blocks, each consisting of a depthwise $3 \times 3$ convolution, $\mathrm { D W } _ { d _ { b } }$ , with dilation $d _ { b }$ to enlarge the receptive field without reducing the

spatial resolution [31], followed by a pointwise $1 \times 1$ convolution (PW). More concretely,

$$
\begin{array} { r l } & { U _ { i } ^ { ( b ) } = \mathrm { S i L U } \Big ( \mathrm { G N } ( \mathrm { D W } _ { d _ { b } } ( F _ { i } ^ { ( b - 1 ) } ) ) \Big ) , } \\ & { F _ { i } ^ { ( b ) } = \mathrm { S i L U } \Big ( F _ { i } ^ { ( b - 1 ) } + \mathrm { G N } ( \mathrm { P W } ( U _ { i } ^ { ( b ) } ) ) \Big ) . } \end{array}\tag{6}
$$

The final feature map $F _ { i } = F _ { i } ^ { ( B _ { \mathrm { e n c } } ) }$ assigns a C-dimensional vector $F _ { i } ( v )$ to each cell v in the FOV.

## 4.2 Action Tokenisation

For each action a, let $v _ { a } \in V$ denote the location reached by applying a from the current location $Q [ i ]$ We tokenise each action into an initial action token $h _ { i , a } ^ { 0 } \in \mathbb { R } ^ { d }$ by applying an MLP to the concatenation of the spatial features at the current and destination cells, the resulting change in cost-to-go $\Delta c _ { i , a } : =$ dist $( v _ { a } , g _ { i } ) { - } \mathsf { d i s t } ( Q [ i ] , g _ { i } )$ , and a learned action embedding:

$$
\begin{array} { r } { h _ { i , a } ^ { 0 } = \mathrm { M L P } ( [ F _ { i } ( Q [ i ] ) , F _ { i } ( v _ { a } ) , \Delta c _ { i , a } , e _ { a } ] ) , } \end{array}\tag{7}
$$

where $e _ { a } \in \mathbb { R } ^ { d _ { a } }$ is the action embedding. The MLP comprises two linear layers of width $d ,$ with a SiLU activation between them and layer normalisation [32] after the second layer.

## 4.3 Recurrent Token Update

We next recursively refine the action tokens while capturing interactions among agents. Inspired by looped Transformers [33, 34] and recurrent GNNs [35], we repeatedly apply a shared update module, enabling multiple rounds of refinement without increasing the number of parameters.

As illustrated in Fig. 2, the refinement is performed through GNN message passing. For this purpose, we expand the communication graph $G _ { \mathrm { c o m m } }$ used in MAGAT into an action-level graph $G _ { \mathrm { a c t } } = ( V _ { \mathrm { a c t } } , E _ { \mathrm { a c t } } )$ whose nodes correspond to the five action tokens of each agent, $\mathrm { i . e . , } | V _ { \mathrm { a c t } } | = 5 n$ . Within each agent, all five action tokens are mutually connected, including self-edges, with zero edge features, whereas across agents, tokens corresponding to the same action are connected according to $G _ { \mathrm { c o m m } }$ with edge features $\omega _ { j i } ,$ , as in MAGAT. Thus, $| E _ { \mathrm { a c t } } | = 2 5 n + 5 | E _ { \mathrm { c o m m } } |$ . GATv2 [16] then performs message passing over $G _ { \mathrm { a c t } }$ , and a gated recurrent unit (GRU) [36] updates each token using the aggregated message and its hidden state from the previous round.

More formally, let $h ^ { \ell } \in \mathbb { R } ^ { 5 n \times d }$ denote the stacked token states at round ℓ and LN denote layer normalisation. We apply the following update for L rounds with shared parameters:

$$
m ^ { \ell } = \mathrm { G A T v 2 } \Big ( \mathrm { L N } ( h ^ { \ell } ) , G _ { \mathrm { a c t } } \Big ) , h ^ { \ell + 1 } = \mathrm { L N } \Big ( \mathrm { G R U } ( m ^ { \ell } , h ^ { \ell } ) \Big ) .\tag{8}
$$

## 4.4 Action Logits

Finally, two distinct linear projections $f _ { 0 } , f _ { L } : \mathbb { R } ^ { d }  \mathbb { R }$ map the initial token $h _ { i , a } ^ { 0 }$ and the final token $h _ { i , a } ^ { L }$ to scalar scores, with a learned coeficient λ weighting the latter:

$$
z _ { i , a } = f _ { 0 } ( h _ { i , a } ^ { 0 } ) + \lambda f _ { L } ( h _ { i , a } ^ { L } ) .\tag{9}
$$

These logits yield the action distribution via softmax.

## 5 Evaluation

We evaluate our MoE pipeline with both MAGAT and ReCAT. All training and evaluation runs are conducted on a single NVIDIA H200 GPU using the POGEMA toolkit [18].

![](images/039be57163a2e62a8a66425cfbba0df44881ddc80e5af78d433cc47c090bb425.jpg)

![](images/6283859e75aa643d33c4519275ed52d619044b5f6702439a278016939eb62751.jpg)

![](images/e7885f7e883056aaa22f5a17b60fbb51336256804290265fe537a0084530bd5e.jpg)  
Figure 3: Main results. Top: success rates (%) for MAGAT (left) and ReCAT (right); values in parentheses indicate percentage-point diferences from the base pretrained policy, the Avg column denotes the unweighted mean across the six map distributions, and the Uniform row combines expert logits with equal weights. Bottom: performance across seven map distributions and varying numbers of agents, comparing base pretrained policies, MoE policies, larger monolithic policies with SFT+RL, HMAGAT, and MAPF-GPT-DDG. Shaded bands indicate 95% bootstrap confidence intervals for SoC.

## 5.1 Experimental Setup

## 5.1.1 Dataset

Following prior GNN-based policy training studies [3, 12], the pretraining dataset contains 21,000 instances with 20% Random and 80% Maze instances, where widths and heights are between 17 and 21 cells and contain 16, 32, or 48 agents. We generate teacher trajectories with LaCAM3 [22]. The supervised fine-tuning datasets contain approximately 3,000 instances per target distribution, with 48, 64, or 80 agents. For router training, we combine the four SFT datasets, resulting in approximately 3,000 × 4 instances in total. To exclude low-quality demonstrations, we retain solutions whose makespan, normalised by $\mathrm { m a x } _ { i \in A } { \mathsf { d i s t } } ( s _ { i } , g _ { i } )$ , is at most 1.4 for each target distribution.

For all training stages, we split the instances into training and validation sets in an 8:2 ratio. As teacher trajectories contain many wait actions from agents that have already reached their goals, which may weaken the learning signal, we follow [15] and mask 80% of these action labels in the cross-entropy loss while retaining all other action labels. The corresponding agents remain in the communication graph. This masking excludes approximately 38% of all action labels in the initial pretraining dataset.

## 5.1.2 Training Procedure

We use AdamW [37] for pretraining, SFT, and router training, with weight decay of 0.05 and cosine annealing of the learning rate to $1 0 ^ { - 6 }$ . For pretraining and SFT, we adjust the batch size according to the fraction of retained labels to maintain an efective batch size of approximately 48.

Pretraining Pretraining runs for 200 epochs with an initial learning rate of $1 0 ^ { - 3 } .$ . Following [12], we perform on-demand data aggregation every four epochs by running the current policy with naive collision shielding on 500 instances. For each failed rollout, we invoke LaCAM3 from its final configuration and add the resulting teacher trajectory to the training dataset. Furthermore, inspired by [11], once the rollout success rate reaches 80%, we additionally revisit successful rollouts whose makespan is at least 1.2 times that of the teacher trajectory. For up to 30 such instances per aggregation round, we sample configurations at 16-timestep intervals and invoke LaCAM3 from each sampled configuration. We add the resulting teacher trajectory if its makespan is at most 1/1.2 of the remaining policy rollout. Pretraining takes approximately 14 hours and 21 hours for the base MAGAT and ReCAT models, respectively, including validation and on-demand data aggregation.

SFT It runs for 52 epochs with an initial learning rate of $1 0 ^ { - 5 }$ , using only the pre-collected trajectories for the target distribution without on-demand data aggregation, taking a few hours per expert.

RL Fine-Tuning Each supervised expert undergoes 100 epochs of RL on its target map distribution. In each epoch, we sample 25 instances with 48, 64, or 80 agents and collect B=4 rollouts per instance, each capped at $T _ { \mathrm { m a x } } { = } 1 2 8$ timesteps. We then perform three optimisation passes over minibatches of the collected rollouts using Adam [38] with a learning rate of 10−6, a clipping threshold $\epsilon = 0 . 2$ , reference-policy KL coeficient $\beta = 0 . 0 1$ , entropy coeficient $\alpha = 0 . 0 0 1$ , and the gradient norm clipped at 0.5.

Router Training With the expert weights frozen, we train the router for 30 epochs with an initial learning rate of $1 0 ^ { - 3 }$ and a fixed routing temperature of $\tau = 1$ . Router training takes approximately 2.5 hours for MAGAT-MoE and 5.5 hours for ReCAT-MoE.

## 5.1.3 Model Configurations

We use the same observation representation and communication graph for MAGAT and ReCAT, with an observation radius of $R _ { \mathrm { { o b s } } } = 5$ and a communication radius of $R _ { \mathrm { c o m m } } = 5$ . Our base MAGAT has 485K parameters and consists of three stacked GNN layers. Our base ${ \mathrm { R e C A T } }$ has 121K parameters, with $B _ { \mathrm { e n c } } = 4$ residual spatial encoder blocks with $C = 6 4$ channels, action tokens of dimension $d = 9 6$ , and $L = 4$ communication rounds. The MLP router has 190K parameters.

## 5.1.4 Baselines

We compare MAGAT-MoE (2.1M parameters) and ReCAT-MoE (676K), with routers trained by supervised learning while the experts remain frozen, against their respective pretrained policies and distribution-specific experts obtained through SFT and RL fine-tuning, to assess how well the mixture preserves specialist performance. We also evaluate Uniform, which combines the same four frozen $\mathrm { S F T { + } R I }$ experts by averaging their action logits before applying softmax to rate the learned routing eficacy.

For comparison with existing methods, we consider two leading neural MAPF policies. HMAGAT (1M) [12] is an IL policy that models agent interactions using hypergraph neural layers and employs RL to adapt each agent’s action-sampling temperature. MAPF-GPT-DDG (2M) [11] is a compact variant of the Transformer-based MAPF-GPT policy [15] that achieves comparable performance with a smaller model through targeted expert data augmentation. For these, we use the authors’ oficial code repositories.

We also prepare parameter-matched monolithic versions of MAGAT (2.1M) and ReCAT (782K) by increasing their latent dimensions accordingly. We first pretrain these models on the same instances as the smaller pretrained base models, taking approximately 15 hours for MAGAT and 30 hours for ReCAT. For a fair comparison, as the MoE policies are exposed to instances with larger numbers of agents during training, these monolithic models subsequently undergo SFT on the concatenation of the four SFT datasets, using the same settings as expert training. RL fine-tuning follows the expert training procedure, with the four training distributions sampled with equal probability. SFT takes approximately 5 hours for MAGAT and 10 hours for ReCAT, while RL fine-tuning takes approximately 8 hours per policy.

All main evaluations use CS-PIBT, given its substantial performance improvement over naive collision shielding.

## 5.1.5 Metrics and Instances

We report the success rate (all agents reaching their goals) and mean relative SoC with respect to LaCAM3 solutions obtained with a 60 s time budget. For unsuccessful instances, agents that have not reached their goals are assigned the timestep limit as their individual costs when computing SoC. For in-distribution evaluation (Maze, Warehouse, Room, and Random), there is no overlap between training and test instances. For out-of-distribution evaluation, we use a fixed map layout for each of Sortation, derived from [39], and Stripes, from [3]. We also generate obstacle-free Empty maps with widths and heights between 17 and 21 cells; see Fig. 3. For each agent count, we use the same 100 fixed instances, each with a limit of 128 timesteps.

## 5.2 Main Results

Figure 3 summarises success rates and mean relative SoC, while Table 1 reports aggregated scores over six map distributions with and without CS-PIBT. Several key observations highlight the benefits of MoE across both underlying architectures.

Map-distribution Experts The success-rate heatmaps show that our fine-tuning strategy substantially improves policy performance when the training and test distributions match, but these gains do not transfer uniformly to other distributions. For example, for both architectures, the Warehouse expert achieves a higher success rate than the pretrained policy on Warehouse, but lower success rates on Maze and Random. This indicates that the resultingfine-tuned policies indeed become specialistsfor specific map distributions.

Eficacy ofMoE Unlike individual experts, whose success rates can drop substantially on certain distributions, MoE policies retain the experts’ strong in-distribution performance while generally improving upon the pretrained models across distributions. MoE generally outperforms Uniform too, indicating that the learned router efectively exploits complementary strengths across experts rather than merely averaging their outputs. Furthermore, despite using fewer parameters, ReCAT-MoE outperforms the leading HMAGAT and MAPF-GPT-DDG baselines across all distributions when considering success rate and solution cost jointly. MAGAT-MoE also achieves comparable aggregate performance to both baselines in terms of success rate and solution cost, further demonstrating the efectiveness of our training pipeline. Table 1 also shows that the benefits of the learned mixtures extend to the setting without CS-PIBT. Nevertheless, as seen for ReCAT on Stripes, the MoE policy does not always match the best-performing expert for each map distribution, leaving room for improvement in expert utilisation.

Expert Usage Analysis Table 2 reports the mean mixing weight assigned to each expert during evaluation. For both architectures, the expert trained on the corresponding distribution receives the largest weight on each in-distribution map, even though the router is not given a map label. On all three out-of-distribution maps, the Maze expert receives the largest weight. Still, the router assigns weight to the other experts too, combining their specialised outputs rather than relying exclusively on a single expert.

Table 1: Aggregated scores with and without CS-PIBT over the same 2,800 instances across six map distributions. SR denotes success rate; relative SoC and runtime are means over all instances, including failures.
<table><tr><td></td><td colspan="3">CS-PIBT</td><td colspan="3">w/o CS-PIBT</td></tr><tr><td>Model</td><td>SR (%)</td><td>Rel. SoC</td><td>Time (s)</td><td>SR (%)</td><td>Rel. SoC</td><td>Time (s)</td></tr><tr><td>MAGAT (485K)</td><td>78.4</td><td>1.752</td><td>0.879</td><td>62.4</td><td>2.161</td><td>0.501</td></tr><tr><td>MAGAT Uniform</td><td>82.6</td><td>1.578</td><td>1.073</td><td>67.6</td><td>2.017</td><td>0.778</td></tr><tr><td>MAGAT MoE</td><td>83.4</td><td>1.555</td><td>1.173</td><td>67.8</td><td>1.984</td><td>0.821</td></tr><tr><td>MAGAT (2.1M, SFT+RL)</td><td>87.3</td><td>1.470</td><td>0.733</td><td>73.6</td><td>1.829</td><td>0.412</td></tr><tr><td>ReCAT (121K)</td><td>81.4</td><td>1.664</td><td>0.921</td><td>69.5</td><td>1.981</td><td>0.535</td></tr><tr><td>ReCAT Uniform</td><td>90.3</td><td>1.414</td><td>1.252</td><td>76.9</td><td>1.729</td><td>1.042</td></tr><tr><td>ReCAT MoE</td><td>91.6</td><td>1.402</td><td>1.241</td><td>78.0</td><td>1.730</td><td>1.061</td></tr><tr><td>ReCAT (782K, SFT+RL)</td><td>94.9</td><td>1.313</td><td>0.723</td><td>85.6</td><td>1.520</td><td>0.437</td></tr><tr><td>HMAGAT</td><td>80.1</td><td>1.483</td><td>2.069</td><td>37.4</td><td>2.191</td><td>2.432</td></tr><tr><td>MAPF-GPT-DDG</td><td>80.2</td><td>1.749</td><td>1.310</td><td>55.0</td><td>2.455</td><td>1.072</td></tr></table>

Table 2: Mean expert mixture weights (%). Columns correspond to evaluation distributions.
<table><tr><td></td><td>Expert</td><td>Maze</td><td>WH.</td><td>Room</td><td>Random</td><td>Sort.</td><td>Stripes</td><td>Empty</td><td>Avg.</td></tr><tr><td></td><td>Maze</td><td>97.4</td><td>11.0</td><td>12.4</td><td>10.1</td><td>63.3</td><td>94.5</td><td>65.5</td><td>50.6</td></tr><tr><td>MA-MOE</td><td>WH.</td><td>0.3</td><td>78.9</td><td>0.5</td><td>1.1</td><td>0.7</td><td>0.0</td><td>12.1</td><td>13.4</td></tr><tr><td></td><td>Room</td><td>1.0</td><td>1.7</td><td>83.3</td><td>2.5</td><td>18.4</td><td>5.4</td><td>0.6</td><td>16.1</td></tr><tr><td></td><td>Random</td><td>1.2</td><td>8.4</td><td>3.9</td><td>86.3</td><td>17.5</td><td>0.1</td><td>21.8</td><td>19.9</td></tr><tr><td></td><td>Maze</td><td>70.4</td><td>22.0</td><td>28.2</td><td>25.0</td><td>49.7</td><td>69.5</td><td>42.4</td><td>43.9</td></tr><tr><td>RCT-TME</td><td>WH.</td><td>8.7</td><td>72.6</td><td>8.1</td><td>6.5</td><td>9.0</td><td>9.1</td><td>18.9</td><td>19.0</td></tr><tr><td></td><td>Room</td><td>11.7</td><td>1.4</td><td>57.0</td><td>5.0</td><td>26.3</td><td>18.6</td><td>0.5</td><td>17.2</td></tr><tr><td></td><td>Random</td><td>9.2</td><td>4.0</td><td>6.8</td><td>63.5</td><td>15.0</td><td>2.8</td><td>38.2</td><td>19.9</td></tr></table>

Does MoE ofer benefits over monolithic policies? While both MoE policies achieve strong performance, their parameter-matched monolithic counterparts reveal a more nuanced picture; the monolithic policies, trained using the same pretraining–SFT–RL recipe, outperform the MoE models. In particular, the larger ReCAT, which incorporates efective components adopted across the literature, achieves the best performance among all evaluated methods by a substantial margin. This result suggests that designing an efective underlying architecture under the same parameter budget can yield more powerful policies than map-distribution-based MoE.

## 5.3 Large-Scale Experiment

We next evaluate these models on substantially larger instances with up to 100,000 agents on larger 512×512 Random maps with 20% obstacles, with start–goal distances at most 64. Table 4 shows that both {MAGAT, ReCAT}-MoE achieve substantially faster inference than the baseline MAPF-GPT-DDG, while HMAGAT

Table 3: Expert performance on Warehouse and Maze across diferent training strategies. PL denotes the Plackett–Luce likelihood of the candidate sequence examined by PIBT; w/o PL uses only the selected action’s probability for RL updates.
<table><tr><td colspan="2"></td><td colspan="2">Warehouse</td><td colspan="2">Maze</td></tr><tr><td></td><td>Training</td><td>SR (%) ↑</td><td>Rel. SoC↓</td><td>SR (%) ↑</td><td>Rel. SoC ↓</td></tr><tr><td rowspan="5">MA)GG5)</td><td>Pretrain</td><td>78.8</td><td>1.900</td><td>70.8</td><td>1.826</td></tr><tr><td>SFT</td><td>85.6</td><td>1.646</td><td>74.4</td><td>1.711</td></tr><tr><td>RL (w/o PL)</td><td>83.8</td><td>1.928</td><td>75.0</td><td>1.854</td></tr><tr><td>RL (PL)</td><td>87.2</td><td>1.792</td><td>77.4</td><td>1.707</td></tr><tr><td>SFT+RL (w/o PL)</td><td>90.0</td><td>1.649</td><td>80.4</td><td>1.708</td></tr><tr><td rowspan="6">RC1)</td><td>SFT+RL (PL)</td><td>90.0</td><td>1.602</td><td>75.8</td><td>1.630</td></tr><tr><td>Pretrain</td><td>75.0</td><td>1.838</td><td>76.4</td><td>1.738</td></tr><tr><td>SFT RL (w/o PL)</td><td>90.4 94.2</td><td>1.522 1.649</td><td>82.4 81.8</td><td>1.568</td></tr><tr><td>RL (PL)</td><td>89.2</td><td>1.645</td><td>85.2</td><td>1.642</td></tr><tr><td>SFT+RL (w/o PL)</td><td>95.4</td><td>1.438</td><td>85.4</td><td>1.549 1.607</td></tr><tr><td>SFT+RL (PL)</td><td>95.2</td><td>1.400</td><td>88.2</td><td>1.472</td></tr></table>

Table 4: Scalability assessment. Times report network inference. The versions with CS-PIBT is shown with parentheses (in Python unlike [2, 3] using C++). Goal denotes the percentage of agents at their goals. SoC is normalised by $\textstyle \sum _ { i \in A } \operatorname { d i s t } ( s _ { i } , g _ { i } )$
<table><tr><td rowspan="2"># Agents</td><td colspan="2">MAGAT-MoE (2.1M)</td><td colspan="2">MAGAT (2.1M)</td><td colspan="2">ReCAT-MoE (676K)</td><td colspan="2">ReCAT (782K)</td><td colspan="2">MAPF-GPT-DDG (2M)</td></tr><tr><td>ms/step Goal (%)</td><td>SoC</td><td></td><td>ms/step Goal (%) SoC</td><td>ms/step Goal (%)</td><td>SoC</td><td></td><td>ms/step Goal (%) SoC</td><td>ms/step Goal (%)</td><td>SoC</td></tr><tr><td>1,000</td><td>5.9 (79.9)</td><td>100 1.017</td><td>1.8 (81.8)</td><td>100 1.015</td><td>11.4 (83.5)</td><td>100 1.019</td><td>6.3 (85.7)</td><td>100 1.013</td><td>40.0 (115.3)</td><td>100 1.032</td></tr><tr><td>10,000</td><td>13.1 (727.7)</td><td>100 1.095</td><td>5.0 (707.1)</td><td>100 1.090</td><td>97.7 (809.9)</td><td>1001.080</td><td>60.3 (771.1)</td><td>100 1.079</td><td>424.2 (1145.2)</td><td>100 1.100</td></tr><tr><td>100,000 620.6 (7402.2)</td><td></td><td>97.122 2.318 285.6 (7068.5)</td><td></td><td>98.335 2.233 2189.8 (9022.2)</td><td></td><td></td><td>99.466 2.104 1227.5 (8191.5)</td><td>99.711 2.020 4476.2 (11435.6)</td><td></td><td>96.555 2.415</td></tr></table>

cannot operate at this scale due to the computational overhead of hypergraph construction. Moreover, ReCAT-MoE achieves superior solution performance while retaining this computational scalability. Still, the monolithic policies achieve faster inference and better solution quality.

## 5.4 Expert Training Ablation

The performance of our MoE policies naturally depends on the quality of the underlying experts, making an efective expert training recipe as important as the MoE training itself. Table 3 investigates this aspect by focusing on Warehouse and Maze. For both architectures, either SFT or RL improves policy performance in terms of both success rate and solution quality, while combining them yields the best overall results, supporting our expert training strategy. The table also ablates the Plackett–Luce likelihood used during GRPO fine-tuning, showing that it consistently reduces SoC compared with using only the selected action’s probability, although its efect on success rate varies across architectures and map distributions.

## 5.5 MoE Training Ablation

We next examine the MoE training pipeline, using ReCAT. Thus far, our MoE models use frozen experts and train only the router through supervised learning. However, the experts can also be jointly optimised during router training, and the resulting MoE can be further fine-tuned with RL. Table 5 compares these alternative training strategies after expert preparation.

Table 5: Performance across diferent MoE training strategies w.r.t. success-rate gains over the pretrained ReCAT (121K), in percentage points. Icons: frozen or updated parameters.
<table><tr><td colspan="2">Supervised training</td><td colspan="2">RL</td><td rowspan="2">Maze</td><td rowspan="2">WH.</td><td rowspan="2">Room</td><td rowspan="2">Random</td><td rowspan="2">Sort.</td><td rowspan="2">Stripes</td><td rowspan="2">Avg.</td></tr><tr><td>Experts</td><td>Router</td><td>Experts</td><td>Router</td></tr><tr><td></td><td>s</td><td>N/A</td><td></td><td>+10.6</td><td>+20.6</td><td>+11.8</td><td>+6.0</td><td>+0.0</td><td>+13.3</td><td>+10.4</td></tr><tr><td>日</td><td>s</td><td>N/A</td><td></td><td>+6.0</td><td>+15.8</td><td>+9.8</td><td>+3.8</td><td>-0.4</td><td>-18.3</td><td>+2.8</td></tr><tr><td>长</td><td>s</td><td></td><td>s</td><td>+9.0</td><td>+21.2</td><td>+12.6</td><td>+6.6</td><td>-0.2</td><td>+16.0</td><td>+10.9</td></tr><tr><td>长</td><td>●</td><td>●</td><td>0</td><td>+13.0</td><td>+21.6</td><td>+12.8</td><td>+8.4</td><td>+0.2</td><td>+24.3</td><td>+13.4</td></tr><tr><td>0</td><td>●</td><td>长</td><td>0</td><td>+3.6</td><td>+17.0</td><td>+9.2</td><td>+5.4</td><td>-0.4</td><td>-18.3</td><td>+2.7</td></tr><tr><td>日</td><td>s</td><td>8</td><td>●</td><td>+8.0</td><td>+19.0</td><td>+11.0</td><td>+6.4</td><td>+0.2</td><td>-5.3</td><td>+6.5</td></tr><tr><td>ReCAT (782K, SFT+RL)</td><td></td><td></td><td></td><td>+12.8</td><td>+20.8</td><td>+14.2</td><td>+7.6</td><td>+0.0</td><td>+33.3</td><td>+14.8</td></tr></table>

The result shows that keeping the experts frozen during supervised training generally yields larger gains than updating them jointly with the router. Among the evaluated variants, supervised router training with frozen experts, followed by RL for both the router and experts, yields the largest gains; this matches the monolithic model on most distributions, although it still falls behind on average, mainly due to the out-of-distribution scenario Stripes.

## 6 Discussion

Across two distinct policy architectures, this study demonstrates that MoE over map-distribution-specialised experts provides a capable approach to neural MAPF. In particular, despite its compact construction, ReCAT MoE achieves strong performance in both in-distribution and out-of-distribution settings. However, we also find that, under the same parameter budget, monolithic policies still provide stronger performance, suggesting that realising the benefits of MoE—a paradigm that has proven highly successful in recent LLMs—remains challenging in the MAPF context. We believe that alternative MoE architectures warrant further investigation. In the following, we position our study within the broader literature and discuss its limitations and future directions.

MoE MoE [4] is a long-established concept in machine learning that combines multiple experts through a routing function. Broadly, MoE training can follow two approaches: learning a router over frozen experts or jointly learning the router and experts. While recent LLMs predominantly follow the latter approach [6, 40], our approach largely falls into the former, leveraging the fact that map-wise fine-tuning provides strong specialised experts for MAPF [3]. Meanwhile, the superior performance obtained by jointly fine-tuning the experts and router with RL suggests that expert specialisation may also emerge implicitly through joint learning. This points to a key limitation of our current approach, i.e., the need to explicitly construct experts for predefined map distributions, while suggesting that learning expert specialisation implicitly may open a diferent avenue for MoE in neural MAPF.

Multiple Algorithms and Policies in MAPF We clarify that this is not the first study to utilise multiple algorithms or policies in MAPF. For example, algorithm selection [41, 42] aims to choose a suitable algorithm from a portfolio of MAPF algorithms for a given target instance; by applying the selected solver, this meta-scheme aims to find solutions eficiently. Tang et al. [43] adopt best-trajectory sampling over multiple policies configured with diferent hyperparameters. Other approaches switch between policies according to predefined rules [3, 19, 44], or combine collision-avoidance and goal-reaching policies by additively combining their outputs, similarly to potential-field methods [45, 46]. In contrast, our contribution lies in an explicit MoE formulation that uses a learned router to adaptively fuse expert outputs according to each agent’s local situation, together with an efective recipe for creating strong specialised experts. This adaptive fusion during rollout is critical, as demonstrated by the results in Fig. 3.

ReCAT As described earlier, our newly designed compact neural MAPF policy, ReCAT, integrates several techniques that have proven efective in other domains. Meanwhile, concurrent preprints have also explored recursive refinement for interaction modelling in neural MAPF policies [47, 48], similarly showing promising results.

Automatic Map Generation Our MoE analysis implicitly suggests that certain map distributions are more useful than others for training capable policies, motivating another interesting direction: automatically generating map distributions for training. Such automatic curriculum generation has proven efective in RL [49], while map generation itself is an active area of MAPF research [50]. Integrating these techniques into neural MAPF policy training represents an exciting direction for future work.

## Acknowledgments

This research was conducted by AIST and under FRONTia, a Japanese national program led by the Ministry of Economy, Trade and Industry of Japan (METI) and the New Energy and Industrial Technology Development Organization (NEDO), aimed at developing a domestic multimodal foundation model for AI robots and physical AI.

## References

[1] Jean-Marc Alkazzi and Keisuke Okumura. A comprehensive review on leveraging machine learning for multi-agent path finding. IEEE Access, 2024.

[2] He Jiang, Yutong Wang, Rishi Veerapaneni, Tanishq Duhan, Guillaume Sartoretti, and Jiaoyang Li. Deploying ten thousand robots: Scalable imitation learning for lifelong multi-agent path finding. In Proceedings ofIEEE International Conference on Robotics and Automation (ICRA), 2025.

[3] Rishabh Jain, Keisuke Okumura, Michael Amir, and Amanda Prorok. Graph attention-guided search for dense multi-agent pathfinding. In Proceedings ofAAAI Conference on Artificial Intelligence (AAAI), 2026.

[4] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geofrey E Hinton. Adaptive mixtures of local experts. Neural computation, 1991.

[5] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research (JMLR), 2022.

[6] Damai Dai et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

[7] Chuanyu Yang, Kai Yuan, Qiuguo Zhu, Wanming Yu, and Zhibin Li. Multi-expert learning of adaptive legged locomotion. Science Robotics, 2020.

[8] David Hoeller, Nikita Rudin, Dhionis Sako, and Marco Hutter. Anymal parkour: Learning agile navigation for quadrupedal robots. Science Robotics, 2024.

[9] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint, 2024.

[10] Qingbiao Li, Weizhe Lin, Zhe Liu, and Amanda Prorok. Message-aware graph attention networks for large-scale multi-robot path planning. IEEE Robotics and Automation Letters (RA-L), 2021.

[11] Anton Andreychuk, Konstantin Yakovlev, Aleksandr Panov, and Alexey Skrynnik. Advancing learnable multi-agent pathfinding solvers with active fine-tuning. In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.

[12] Rishabh Jain, Keisuke Okumura, Michael Amir, Pietro Liò, and Amanda Prorok. Pairwise is not enough: Hypergraph neural networks for multi-agent pathfinding. In Proceedings of the International Conference on Learning Representations (ICLR), 2026.

[13] Roni Stern, Nathan R. Sturtevant, Ariel Felner, Sven Koenig, Hang Ma, Thayne T. Walker, Jiaoyang Li, Dor Atzmon, Liron Cohen, T. K. Satish Kumar, Eli Boyarski, and Roman Barták. Multi-agent pathfinding: Definitions, variants, and benchmarks. In Proceedings ofAnnual Symposium on Combinatorial Search (SoCS), 2019.

[14] Guillaume Sartoretti, Justin Kerr, Yunfei Shi, Glenn Wagner, TK Satish Kumar, Sven Koenig, and Howie Choset. Primal: Pathfinding via reinforcement and imitation multi-agent learning. IEEE Robotics and Automation Letters (RA-L), 2019.

[15] Anton Andreychuk, Konstantin Yakovlev, Aleksandr Panov, and Alexey Skrynnik. MAPF-GPT: Imitation learning for multi-agent pathfinding at scale. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), 2025.

[16] Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks? In Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[17] Qingbiao Li, Fernando Gama, Alejandro Ribeiro, and Amanda Prorok. Graph neural networks for decentralized multi-robot path planning. In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020.

[18] Alexey Skrynnik, Anton Andreychuk, Anatolii Borzilov, Alexander Chernyavskiy, Konstantin Yakovlev, and Aleksandr Panov. POGEMA: A benchmark platform for cooperative multi-agent pathfinding. In Proceedings of the International Conference on Learning Representations (ICLR), 2025.

[19] Keisuke Okumura, Manao Machida, Xavier Défago, and Yasumasa Tamura. Priority inheritance with backtracking for iterative multi-agent path finding. Artificial Intelligence (AIJ), 2022.

[20] Rishi Veerapaneni, Qian Wang, Kevin Ren, Arthur Jakobsson, Jiaoyang Li, and Maxim Likhachev. Improving learnt local mapf policies with heuristic search. In Proceedings ofInternational Conference on Automated Planning and Scheduling (ICAPS), 2024.

[21] Rishi Veerapaneni, Arthur Jakobsson, Kevin Ren, Samuel Kim, Jiaoyang Li, and Maxim Likhachev. Work smarter not harder: Simple imitation learning with cs-pibt outperforms large-scale imitation learning for mapf. In Proceedings of IEEE International Conference on Robotics and Automation (ICRA), 2025.

[22] Keisuke Okumura. Engineering lacam∗: Towards real-time, large-scale, and near-optimal multi-agent pathfinding. In Proceedings ofInternational Conference on Autonomous Agents & Multiagent Systems (AAMAS), 2024.

[23] Sainbayar Sukhbaatar, Olga Golovneva, Vasu Sharma, Hu Xu, Xi Victoria Lin, Baptiste Rozière, Jacob Kahn, Daniel Li, Wen-tau Yih, Jason Weston, and Xian Li. Branch-train-mix: Mixing expert llms into a mixture-of-experts llm. In Conference on Language Modeling (COLM), 2024.

[24] Bojie Shen, Zhe Chen, Muhammad Aamir Cheema, Daniel D Harabor, and Peter J Stuckey. Tracking progress in multi-agent path finding. In Proceedings of International Conference on Automated Planning and Scheduling (ICAPS), 2023.

[25] Robin L Plackett. The analysis of permutations. Journal of the Royal Statistical Society Series C: Applied Statistics, 1975.

[26] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint, 2017.

[27] Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European Conference on Computer Vision (ECCV), 2018.

[28] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 2018.

[29] Mingxing Tan and Quoc Le. Eficientnet: Rethinking model scaling for convolutional neural networks. In Proceedings ofthe International Conference on Machine Learning (ICML), 2019.

[30] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Eficient convolutional neural networks for mobile vision applications. arXiv preprint, 2017.

[31] Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. In Proceedings of the International Conference on Learning Representations (ICLR), 2016.

[32] Jimmy Lei Ba, Jamie Ryan Kiros, and Geofrey E Hinton. Layer normalization. arXiv preprint, 2016.

[33] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. In Proceedings of the International Conference on Learning Representations (ICLR), 2019.

[34] Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. In Proceedings of the International Conference on Learning Representations (ICLR), 2024.

[35] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2016.

[36] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. In Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.

[37] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of the International Conference on Learning Representations (ICLR), 2019.

[38] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2015.

[39] Zhe Chen, Daniel Harabor, Jiaoyang Li, and Peter J Stuckey. Trafic flow optimisation for lifelong multi-agent path finding. In Proceedings ofAAAI Conference on Artificial Intelligence (AAAI), 2024.

[40] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2017.

[41] Omri Kaduri, Eli Boyarski, and Roni Stern. Algorithm selection for optimal multi-agent pathfinding. In Proceedings of International Conference on Automated Planning and Scheduling (ICAPS), 2020.

[42] Jingyao Ren, Vikraman Sathiyanarayanan, Eric Ewing, Baskin Senbaslar, and Nora Ayanian. Mapfast: A deep algorithm selector for multi agent path finding using shortest path embeddings. In Proceedings of International Conference on Autonomous Agents & Multiagent Systems (AAMAS), 2021.

[43] Huijie Tang, Federico Berto, and Jinkyoo Park. Ensembling prioritized hybrid policies for multi-agent pathfinding. In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

[44] Jianqi Gao, Yanjie Li, Xiaoqing Yang, and Mingshan Tan. Rde: A hybrid policy framework for multi-agent path finding problem. arXiv preprint, 2023.

[45] Alexey Skrynnik, Alexandra Yakovleva, Vasilii Davydov, Konstantin Yakovlev, and Aleksandr I Panov. Hybrid policy learning for multi-agent pathfinding. IEEE Access, 2021.

[46] Arseni Pertzovskiy, Roni Stern, Ariel Felner, and Roie Zivan. Enhancing lifelong multi-agent path-finding by using artificial potential fields. In Proceedings of International Conference on Automated Planning and Scheduling (ICAPS), 2026.

[47] Valeriy Vyaltsev, Alsu Sagirova, Anton Andreychuk, Oleg Bulichev, Yuri Kuratov, Konstantin Yakovlev, Aleksandr Panov, and Alexey Skrynnik. Learning to communicate locally for large-scale multi-agent pathfinding. arXiv preprint, 2026.

[48] Valeriy Vyaltsev, Anton Andreychuk, Taisia Zlotnikova, Konstantin Yakovlev, Aleksandr Panov, and Alexey Skrynnik. Decentralized master-mind: Joint action refinement through iterative intent denoising in multi-agent pathfinding. arXiv preprint, 2026.

[49] Thomy Phan, Timy Phan, and Sven Koenig. Generative curricula for multi-agent path finding via unsupervised and reinforcement learning. Journal ofArtificial Intelligence Research (JAIR), 2025.

[50] Cheng Qian, Yulun Zhang, Varun Bhatt, Matthew Christopher Fontaine, Stefanos Nikolaidis, and Jiaoyang Li. Qd-mapper: A quality diversity framework to automatically evaluate multi-agent path finding algorithms in diverse maps. In Proceedings of International Conference on Autonomous Agents & Multiagent Systems (AAMAS), 2026.

Maze

## Appendix

## A Map Distributions

Figure 4 shows four layouts from each in-distribution map generator. Start–goal pairs are sampled within connected components. Maps that cannot accommodate the requested number of agents are resampled. Generator parameters are sampled uniformly within the ranges below.

Maze We generate maze-like layouts by growing walls from randomly selected cells in two-cell steps, filling the intervening cell. Even map dimensions are rounded up to the next odd integer. Each wall seed receives 1–8 extension attempts, with a straight-continuation weight of 0.75–0.85 to favour long corridors. The density parameter, sampled between 0.2 and 1.0, controls the number of wall-growth attempts.

Warehouse We arrange rectangular shelves in a regular grid, with shelf widths of 2–8 cells and heights of 2–6 cells. Aisles span 1–2 cells between shelf columns and one cell between rows, with side padding of 1–3 cells. Map dimensions are determined from an initial arrangement of 1–4 shelf rows and columns.

Room We generate a grid with 3–6 rows and columns of rooms, whose shared width and height are independently sampled between three and five cells. Adjacent rooms are separated by one-cell-thick walls, each with a randomly placed doorway. A one-cell-wide perimeter remains free in each room, while interior cells are independently blocked with a probability sampled between 0.8 and 1.0 per map.

Random Obstacles are placed randomly, with an obstacle density sampled between 20% and 40%.

Warehouse  
Room  
Random  
![](images/1f03bb25470c32b51333037c31345750416f59aa113f82157fb202f66a18cf12.jpg)  
Figure 4: Examples from the four in-distribution map generators. Each column shows four diferent layouts; black cells denote obstacles and white cells are traversable.

## B Model Details

Table 6 lists the exact parameter counts.

Table 6: Parameter counts of the evaluated MAGAT and ReCAT configurations, including all expert parameters in each mixture.

<table><tr><td>Model Parameters</td></tr><tr><td>MAGAT (485K) 484,933</td></tr><tr><td>MAGAT (2.1M) 2,154,181</td></tr><tr><td>ReCAT (121K) 121,299</td></tr><tr><td>ReCAT (782K) 781,843</td></tr><tr><td>MLP router (190K) 190,980</td></tr><tr><td>MAGAT-MoE (2.1M) 2,130,712</td></tr><tr><td>ReCAT-MoE (676K) 676,176</td></tr><tr><td>MAGAT Uniform 1,939,732</td></tr><tr><td>ReCAT Uniform 485,196</td></tr></table>

## B.1 MAGAT

The encoder uses five $3 \times 3$ convolutions with channel widths (32, 32, 64, 64, 128) and ReLU, with $2 \times 2$ max pooling after the first, third, and fifth layers. A linear projection produces 256-dimensional agent embeddings. For MAGAT (485K), the first two GATv2 layers concatenate four 16-dimensional heads, and the final layer averages four 256-dimensional heads. The encoder embedding is added to the GAT output before an MLP decoder with two 256-dimensional hidden layers and ReLU. MAGAT (2.1M) increases the head dimension in the first two GATv2 layers from 16 to 128, leaving the remaining architecture unchanged.

## B.2 ReCAT

ReCAT (121K) and ReCAT (782K) use depthwise-convolution dilation factors (1, 2, 1, 2), eight groups for group normalisation, and action embeddings of dimension $d _ { a } = 1 6 . \mathrm { G A T v } 2$ concatenates four attention heads of dimension $d / 4 ,$ , with attention dropout set to zero. The learned coeficient λ is initialised to 0.05. The larger ReCAT (782K) increases the spatial channels and action-token dimension to $C = d = 2 2 4$ , retaining the remaining settings of the base model.

## C Evaluation without CS-PIBT

Figure 5 reports results with naive collision shielding on the same test instances as Fig. 3.  
![](images/ace12255b865a9103efdc6c5012ba8672c8937a9b6885cd75b4279b1bb6fe44f.jpg)  
Figure 5: Success rate and mean relative SoC without CS-PIBT across seven map distributions. Each point averages 100 instances; SoC is reported relative to LaCAM3 with a 60 s time limit. Shaded bands indicate 95% bootstrap confidence intervals for SoC.

## D Expert Exclusion

We examine whether each expert contributes to MoE performance on its target distribution. We remove one distribution-specific expert and train a new router on the SFT data from all four distributions, keeping the remaining experts frozen. Table 7 compares these variants with the full four-expert MoE. Removing an expert reduces success rates on its target distribution for all four ReCAT experts and for the Warehouse and Room experts in MAGAT, highlighting the contribution of distribution-specific experts.

Table 7: Success rates (%) with CS-PIBT after excluding one expert while retaining router-training data from all four distributions. Parentheses give changes in percentage points relative to the full four-expert MoE.
<table><tr><td></td><td>Variant</td><td>Maze</td><td>WH.</td><td>Room</td><td>Random</td><td>Sort.</td><td>Stripes</td><td>Empty</td><td>Avg.</td></tr><tr><td></td><td>Maze</td><td>75.0 (+0.2)</td><td>89.2 (-2.0)</td><td>88.2 (-1.4)</td><td>87.4 (+3.0)</td><td>99.0 (+0.0)</td><td>45.3 (-1.0)</td><td>100.0 (+0.0)</td><td>83.4 (-0.2)</td></tr><tr><td></td><td>WH.</td><td>77.0 (+2.2)</td><td>87.6 (-3.6)</td><td>87.6 (-2.0)</td><td>87.2 (+2.8)</td><td>99.0 (+0.0)</td><td>43.7 (-2.7)</td><td>100.0 (+0.0)</td><td>83.2 (-0.5)</td></tr><tr><td></td><td>Room</td><td>74.8 (+0.0)</td><td>89.4 (-1.8)</td><td>83.8 (-5.8)</td><td>87.0 (+2.6)</td><td>98.8 (-0.2)</td><td>43.7 (-2.7)</td><td>100.0 (+0.0)</td><td>82.5 (-1.1)</td></tr><tr><td>MAG-MOE</td><td>Random</td><td>77.2 (+2.4)</td><td>89.8 (-1.4)</td><td>88.8 (-0.8)</td><td>84.4 (+0.0)</td><td>98.6 (-0.4)</td><td>46.7 (+0.3)</td><td>100.0 (+0.0)</td><td>83.6 (+0.0)</td></tr><tr><td></td><td>Full MoE</td><td>74.8</td><td>91.2</td><td>89.6</td><td>84.4</td><td>99.0</td><td>46.3</td><td>100.0</td><td>83.6</td></tr><tr><td></td><td>\Maze</td><td>85.4 (-1.6)</td><td>95.4 (-0.2)</td><td>96.4 (-1.0)</td><td>93.0 (+0.4)</td><td>100.0 (+0.2)</td><td>66.0 (-1.3)</td><td>100.0 (+0.0)</td><td>90.9 (-0.5)</td></tr><tr><td>RCT-TME</td><td>WH.</td><td>86.0 (-1.0)</td><td>90.8 (-4.8)</td><td>97.6 (+0.2)</td><td>91.4 (-1.2)</td><td>100.0 (+0.2)</td><td>73.3 (+6.0)</td><td>100.0 (+0.0)</td><td>91.3 (-0.1)</td></tr><tr><td></td><td>Room</td><td>88.0 (+1.0)</td><td>96.2 (+0.6)</td><td>95.4 (-2.0)</td><td>93.2 (+0.6)</td><td>99.8 (+0.0)</td><td>70.3 (+3.0)</td><td>100.0 (+0.0)</td><td>91.8 (+0.5)</td></tr><tr><td></td><td>Random</td><td>87.2 (+0.2)</td><td>96.0 (+0.4)</td><td>97.8 (+0.4)</td><td>89.6 (-3.0)</td><td>99.8 (+0.0)</td><td>71.3 (+4.0)</td><td>100.0 (+0.0)</td><td>91.7 (+0.3)</td></tr><tr><td></td><td>Full MoE</td><td>87.0</td><td>95.6</td><td>97.4</td><td>92.6</td><td>99.8</td><td>67.3</td><td>100.0</td><td>91.4</td></tr></table>