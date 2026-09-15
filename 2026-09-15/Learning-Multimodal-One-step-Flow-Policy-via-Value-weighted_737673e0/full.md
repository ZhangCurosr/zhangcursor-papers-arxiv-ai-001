# Learning Multimodal One-step Flow Policy via Value-weighted Optimal Transport

Jaehun Shon<sup>\*</sup> Jinha Choi<sup>\*</sup> Jongwook Jeon Jongmin Lee<sup>†</sup> Department of Artificial Intelligence, Yonsei University {manfromearth, danielc174, jjwook2602, jongminlee}@yonsei.ac.kr <sup>\*</sup>Equal contribution <sup>†</sup>Corresponding author

## Abstract

Offline reinforcement learning aims to learn a policy solely from fixed datasets, which often contain multimodal action distributions. Flow policies can naturally represent such multimodal behaviors, but learning an efficient one-step flow policy remains challenging: standard value guidance often leads to mode collapse or exploits overestimation bias in out-of-distribution regions. To address this, we introduce One-step Flow policy via Optimal Transport (OptiFlow), a framework for one-step flow policy learning as a structured sample-allocation problem. OptiFlow jointly trains a value-aware reference flow policy and an efficient onestep policy, coupling their action samples through state-wise entropic optimal transport. For each state, critic-estimated values define the priority of distillation target actions, while the action-distance cost ensures geometrically compatible pairings. By avoiding direct critic maximization, our transport-guided approach enables in-distribution exploitation by anchoring the one-step policy to high-value, dataset-supported modes without the risk of out-of-distribution divergence. Experimental results demonstrate that OptiFlow effectively captures optimal multimodal behaviors and achieves strong performance across diverse offline RL benchmarks. Our code is available at https://github.com/Yonsei-DILLab/OptiFlow.

## 1 Introduction

Offline reinforcement learning (RL) aims to learn decision-making policies from fixed datasets without further environment interaction [1, 2]. This setting is crucial in domains such as robotics and control, where online exploration can be costly or unsafe. A central challenge is distribution shift: when a learned policy selects actions poorly supported by the offline data, value estimates can become unreliable and be amplified during policy optimization. Standard methods mitigate this through conservative value estimation [3, 4], behavior-constrained policy learning [5, 6], or distribution correction via dual optimization [7, 8].

Beyond distribution shift, modern offline datasets often exhibit multimodal behavior: for a given state, multiple distinct actions may be valid and lead to high return. Standard unimodal policies can struggle in such settings, often collapsing toward a mean action that is suboptimal or physically invalid [6, 9]. This issue is especially important in high-dimensional manipulation and control tasks, where multiple successful strategies may coexist. Expressive generative policies, including diffusionbased [10–12] and flow-based [13] policies, provide a natural way to model such multimodal action distributions [14–18]. Flow policies are particularly appealing because their deterministic probability paths make one-step policy learning a natural target for efficient deployment [11, 12].

However, learning an effective one-step flow policy remains challenging. Multi-step sampling is often too expensive for real-time control, but compressing a reference policy into one step can lose multimodal structure [19]. Pointwise distillation, which matches the same noise input to a target action, can force a one-step model to memorize complex paths and collapse distinct behaviors into compromised averages [20]. Direct critic maximization introduces a different failure mode: the policy may exploit local optima or overestimated out-of-distribution actions instead of capturing high-return in-distribution behavior.

We address these limitations by reframing one-step flow policy learning as a structured sampleallocation problem. We propose Flow Policy via Optimal Transport (OptiFlow), which couples samples from a value-aware reference flow policy and an efficient one-step policy through entropic optimal transport [21]. OptiFlow separates value from geometry: critic-estimated values determine the supervision mass by prioritizing high-return reference actions, while the transport cost encourages geometrically compatible pairings between one-step-policy samples and reference actions.

This transport-guided update avoids direct critic maximization of the deployed policy. Instead of following unreliable critic gradients into OOD regions or local optima, the one-step policy is anchored to high-value actions proposed by the behavior-regularized reference policy. At the same time, by relaxing strict pointwise matching OptiFlow lets the one-step policy flexibly map noise to actions and preserve multimodal reference-policy structure. We empirically show that OptiFlow captures multimodal behavior in controlled diagnostics and achieves strong performance across offline RL benchmarks.

## 2 Related Work

Expressive generative policies for offline RL. Recent work has introduced diffusion and flowbased policies as expressive alternatives to unimodal actors for modeling multimodal action distributions in offline RL [14–18]. FQL [16] learns flow-based policies and uses critic-guided policy improvement to obtain strong offline RL performance. GFP [17] introduces value-aware behavior cloning for flow policies together with one-step distillation. OptiFlow builds on this line of work, but addresses a different question: how should value-guided supervision be allocated across multiple action samples when the reference flow policy contains several high-value modes? Rather than directly optimizing the one-step policy against the critic or applying guidance only at inference time, OptiFlow uses critic-estimated values to define a reference-policy marginal in an optimal-transport coupling.

Value-guided policy learning and behavior constraints. A central theme in offline RL is to improve policy quality while avoiding unsupported actions. Prior methods address this through conservative value estimation [3, 4], behavior regularization [5, 6], or value-weighted policy extraction [22, 23]. These approaches provide different mechanisms for balancing value improvement and support preservation. OptiFlow follows the same broad principle, but uses value information differently: critic-estimated values do not directly optimize the deployed one-step policy, but instead shape the reference-policy marginal used for transport-guided supervision.

Matching, retrieval, and optimal transport. Several offline RL and imitation learning methods use matching or retrieval mechanisms to keep learned policies close to data-supported actions. For example, retrieval-based policy constraints regularize the policy toward nearby dataset actions [24]. Optimal transport has also been used to compare or align behavior distributions in imitation learning and offline decision making [25–27]. OptiFlow differs in both purpose and construction. We do not use optimal transport merely as a distributional distance or trajectory-level alignment objective. Instead, OptiFlow solves a state-wise entropic optimal transport problem between one-step-policy and reference-policy action samples. The resulting coupling combines a value-weighted reference-policy marginal with action-space geometry and produces transport-guided regression targets for efficient one-step flow-policy learning.

## 3 Preliminaries

## 3.1 Offline Reinforcement Learning

We consider a discounted Markov decision process $\mathcal { M } = \langle \boldsymbol { S } , \mathcal { A } , P , r , \rho _ { 0 } , \gamma \rangle$ [28], where $s$ and A denote the state and action spaces, $P : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ is the transition kernel, $r : S \times \mathcal { A } $ R is the reward function, $\rho _ { 0 } \in \Delta ( \bar { \mathcal { S } } )$ is the initial-state distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. In offline RL, the learner is given a fixed dataset $\mathcal { D } = \{ ( s _ { i } , a _ { i } , r _ { i } , s _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { | \mathcal { D } | }$ collected by potentially diverse and unknown behavior policies and cannot collect additional data from environment interactions.

For a policy π, we write $Q ^ { \pi } ( s , a )$ and $V ^ { \pi } ( s )$ for its action-value and value functions, $J ( \pi ) =$ $\mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \dot { \gamma ^ { t } } r ( s _ { t } , a _ { t } ) \right]$ for its discounted return, and $\begin{array} { r } { d ^ { \pi } ( s ) = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \operatorname* { P r } _ { \pi } ( s _ { t } = s ) } \end{array}$ for its discounted state occupancy. OptiFlow learns a critic estimate $Q _ { \phi } ( s , a )$ from $\mathcal { D }$ and uses criticestimated values to construct a value-weighted transport problem for updating a one-step flow policy.

## 3.2 Flow Policies

Flow matching [13, 29] learns a time-dependent vector field that transports samples from a simple base distribution to a target distribution. In offline RL, we apply this to conditional action generation, where the vector field $v _ { \theta } ( s , x _ { t } , t )$ is conditioned on the state $s .$ For behavioral cloning, we set the base sample as noise $z \sim \mathcal { N } ( 0 , I )$ and the target action as an action $a \sim \mathcal { D }$ from the dataset. Defining the linear interpolation $x ^ { t } \overset { \cdot } { = } ( 1 - t ) x ^ { 0 } + t \overset { \cdot } { x } ^ { 1 }$ for $t \in [ 0 , 1 ] , z = x ^ { 0 }$ , and $a = x ^ { 1 }$ , the most basic flow-matching objective for BC is as follows:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { B C } } ( \theta ) = \mathbb { E } _ { \mathbf { \Phi } _ { ( s , a = x ^ { 1 } ) \sim \mathcal { D } , \atop z = x ^ { 0 } \sim \mathcal { N } ( 0 , I _ { d } ) , \atop t \sim \mathrm { U n i f } ( [ 0 , 1 ] ) } } \left[ \left\| v _ { \theta } ( s , x ^ { t } , t ) - ( x ^ { 1 } - x ^ { 0 } ) \right\| _ { 2 } ^ { 2 } \right] .\tag{1}
$$

At inference time, an action is generated by sampling $z \sim \mathcal { N } ( 0 , I _ { d } )$ and integrating the ODE from $t = 0$ to $\begin{array} { r } { t \stackrel { } { = } 1 : \frac { d x ^ { t } } { d t } \stackrel { } { = } v _ { \theta } ( s , x ^ { t } , t ) } \end{array}$ , starting from $z = x ^ { 0 }$ , to obtain $a = x ^ { 1 }$ . In practice, this integration is approximated by a finite number of Euler steps:

$$
\begin{array} { r } { x ^ { k + 1 } = x ^ { k } + \frac { 1 } { K } v _ { \theta } \left( s , x ^ { k } , \frac { k } { K } \right) \mathrm { ~ f o r ~ } k = 0 , \dots , K - 1 . } \end{array}\tag{2}
$$

Larger K gives an expressive multi-step flow policy but requires iterative action generation. To enable efficient deployment, a one-step flow policy directly learns a mapping $a = \mu _ { \theta } ( s , z )$ from noise to action, which induces a stochastic policy $\pi _ { \boldsymbol { \theta } } ( a | \boldsymbol { s } )$ due to the stochasticity of the base noise z. OptiFlow uses a value-aware multi-step reference flow policy to model multimodal behavior and learns an efficient one-step flow policy $\mu _ { \boldsymbol { \theta } } ( s , z )$ through transport-guided updates.

## 3.3 Entropic Optimal Transport

Let $p \in \Delta ^ { N }$ and $q \in \Delta ^ { M }$ be discrete probability distributions over two finite sets, and let $C \in \mathbb { R } ^ { N \times M }$ be a cost matrix. The discrete optimal transport problem seeks a coupling $P \in \mathbb { R } _ { + } ^ { N \times M }$ whose marginals match p and q:

$$
\Pi ( p , q ) = \left\{ P \geq 0 \mid P \mathbf { 1 } _ { M } = p , \ P ^ { \top } \mathbf { 1 } _ { N } = q \right\} .\tag{3}
$$

Exact optimal transport requires solving a linear program, which scales poorly and cannot efficiently leverage GPU parallelization. For efficient and scalable computation, we consider the entropyregularized optimal transport problem [21]:

$$
P ^ { * } = \underset { P \in \Pi ( p , q ) } { \arg \operatorname* { m i n } } \Big [ \langle P , C \rangle + \varepsilon \sum _ { i , j } P _ { i j } ( \log P _ { i j } - 1 ) \Big ] ,\tag{4}
$$

where $\varepsilon > 0$ controls the smoothness of the transport plan. This regularized problem admits a unique solution that can be computed efficiently using the Sinkhorn algorithm [21], an iterative matrix scaling procedure. The detailed formulation and the corresponding pseudocode are provided in Appendix C. In the following section, we detail how OptiFlow leverages this optimal transport framework state-wise to align an efficient one-step flow policy with a multimodal reference policy.

## 4 Flow Policy via Optimal Transport

We introduce Flow Policy via Optimal Transport (OptiFlow), a framework that learns efficient one-step flow policies for offline RL through a value-weighted optimal transport formulation. While one-step flow policies are expressive enough to represent multimodal action distributions in principle, existing approaches suffer from two distinct failure modes: (i) Mode collapse during mapping, where pointwise noise-to-action matching is prone to collapsing into compromised averages, (ii) Prone to local optima or OOD actions, where direct critic maximization can pull the one-step policy toward local optima or out-of-distribution regions. OptiFlow addresses these challenges by reframing one-step policy learning as a structured sample-allocation problem. We decouple the policy improvement process into three mechanisms: (i) learning a value-aware multi-step reference flow policy anchored in the dataset, (ii) constructing a value-aware geometric coupling via entropic OT, and (iii) extracting the one-step policy through transport-guided target selection. An overview is illustrated in Figure 1.

![](images/038fbffa7848b2e6df12a8387618ba94d3f0805dbeac4b836aab9c291a428727.jpg)  
Figure 1: Overview of OptiFlow. OptiFlow learns a critic and a reference flow policy from offline data, then couples reference and one-step policy action samples with entropic optimal transport. Critic values define a reference marginal over reference actions, and the resulting transport plan provides structured distillation targets for the one-step policy.

Step 1: Learning the critic. The first step is to train an action-value function $Q _ { \phi } ( s , a )$ from the offline dataset $\mathcal { D } = \{ ( s _ { i } , a _ { i } , r _ { i } , s _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { | \mathcal { D } | }$ using a standard temporal-difference objective for the one-step policy µ<sub>θ</sub>:

$$
\operatorname* { m i n } _ { \phi } \mathcal { L } _ { Q } ( \phi ) = \mathbb { E } _ { ( s , a , r , s ^ { \prime } ) \sim \mathcal { D } , z ^ { \prime } \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ \Big ( Q _ { \phi } ( s , a ) - \big ( r + \gamma Q _ { \bar { \phi } } ( s ^ { \prime } , \mu _ { \theta } ( s ^ { \prime } , z ^ { \prime } ) ) \big ) \Big ) ^ { 2 } \right] ,\tag{5}
$$

where $Q _ { \bar { \phi } }$ denotes a target critic network [30]. In OptiFlow, the critic is used in two ways: it defines the priority weights for training the reference flow policy, and it constructs the reference-side marginal for the optimal transport problem. Additional implementation variants of the TD target are described in Appendix D.

Step 2: Learning the multi-step reference flow policy. The next step of OptiFlow is to establish a data-supported, high-value reference policy $\mu _ { \omega }$ . To this end, we adopt the Value-aware behavior cloning (VaBC) [17] objective to train a multi-step reference flow policy. Given a dataset action a and noise $\epsilon \sim \mathcal { N } ( 0 , I _ { d } )$ , the velocity field $v _ { \omega }$ is trained via:

$$
\operatorname* { m i n } _ { \omega } \mathcal { L } _ { \mathrm { r e f } } ( \omega ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } , \epsilon \sim \mathcal { N } ( 0 , I _ { d } ) , t \sim \mathrm { U n i f } ( [ 0 , 1 ] ) } \left[ g _ { \eta } ( s , a ) \| v _ { \omega } ( s , a _ { t } , t ) - ( a - \epsilon ) \| _ { 2 } ^ { 2 } \right]\tag{6}
$$

$$
\begin{array} { r } { \exp \left( \frac { \lambda } { \eta } Q _ { \phi } ( s , a ) \right) } \end{array}
$$

$$
\begin{array} { r } { \mathrm { w h e r e ~ } g _ { \eta } ( s , a ) = \frac { \star \ \setminus \ \cup \ \textnormal { \textsf { s v a r } } ^ { \prime } \ { } ^ { \prime } J } { \exp \Big ( \frac { \lambda } \eta Q _ { \phi } ( s , a ) \Big ) + \exp \Big ( \frac { \lambda } \eta Q _ { \phi } ( s , \mu _ { \theta } ( s , z ) ) \Big ) } , \quad z \sim  { \mathcal { N } ( 0 , I _ { d } ) } . } \end{array}\tag{7}
$$

where $\begin{array} { r } { \lambda = \left( \frac { 1 } { B } \sum _ { i = 1 } ^ { B } | Q _ { \phi } ( s _ { i } , a _ { i } ) | + \epsilon _ { q } \right) ^ { - } } \end{array}$ 1 normalizes the critic scale within a minibatch of size B, with a small constant ${ \epsilon } _ { q } > 0$ . Here, $g _ { \eta } ( s , a )$ assigns higher weights to dataset actions that outperform the current one-step policy $\mu _ { \boldsymbol { \theta } } ( s , z )$ as estimated by the critic $Q _ { \phi }$ . This objective clones dataset actions with high values. Consequently, $\mu _ { \omega }$ successfully captures the multi-modal structure of the data while concentrating probability density on high-return regions Importantly, the reference multi-step flow policy $\mu _ { \omega }$ serves as a support-preserving proposal distribution that provides a high-quality candidate action set. This ensures that the subsequent OptiFlow’s sample allocation is restricted to regions that are both in-distribution and high-performing, effectively filtering out suboptimal or unsupported actions before the final one-step policy extraction.

Step 3: Value-weighted transport coupling. With the multi-step reference policy $\mu _ { \omega }$ providing a set of high-quality, in-distribution candidate actions, the next challenge is to allocate these targets to the one-step policy $\mu _ { \theta }$ without inducing mode collapse or geometric misalignment. For each state s, we sample N actions from the one-step policy and M actions from the reference flow policy:

$$
a _ { \theta , i } ^ { s } = \mu _ { \theta } ( s , z _ { i } ) , \qquad z _ { i } \sim \mathcal { N } ( 0 , I _ { d } ) , \qquad i = 1 , \ldots , N ,\tag{8}
$$

$$
\tilde { a } _ { \omega , j } ^ { s } = \mu _ { \omega } ( s , z _ { j } ) , \qquad z _ { j } \sim \mathcal { N } ( 0 , I _ { d } ) , \qquad j = 1 , \ldots , M .\tag{9}
$$

To decouple the value guidance from geometric alignment, OptiFlow formulates the sample allocation as an optimal transport problem with asymmetric marginal constraints. Specifically, we impose a uniform marginal constraint $\boldsymbol { p } \in \Delta ^ { \tilde { N } }$ , with $\begin{array} { r } { { p } _ { i } \ = \ \frac { \overline { { 1 } } } { N } } \end{array}$ , on the one-step policy samples, and a value-weighted marginal constraint $q ^ { s } \in \Delta ^ { M }$ on the reference policy actions:

$$
q _ { j } ^ { s } = \frac { \exp ( Q _ { \phi } ( s , \tilde { a } _ { \omega , j } ^ { s } ) / \tau ) } { \sum _ { k = 1 } ^ { M } \exp ( Q _ { \phi } ( s , \tilde { a } _ { \omega , k } ^ { s } ) / \tau ) } .\tag{10}
$$

where $\tau > 0$ is a temperature parameter. This asymmetric marginal design is essential for learning a valid one-step flow policy. The uniform constraint p ensures that all base noise samples $z _ { i } \sim \mathcal { N } ( 0 , I _ { d } )$ are used equally, preserving the base distribution. Conversely, the value-weighted constraint q injects the critic’s guidance directly into the allocation process. By assigning larger probability mass to high-return reference actions, the transport plan maps the majority of one-step samples to high-value modes.

Then, to complete this decoupling, we define the optimal transport cost solely based on geometric proximity in the action space, without any value-based terms.

$$
C _ { i j } ^ { s } = \frac { 1 } { \bar { c } ^ { s } } \| a _ { \theta , i } ^ { s } - \tilde { a } _ { \omega , j } ^ { s } \| _ { 2 } ^ { 2 } ,\tag{11}
$$

where $\bar { c } ^ { s }$ denotes the per-state mean squared distance over all action pairs. With the asymmetric marginals $( p , q ^ { s } )$ that determines the target weights and the cost matrix $C$ that encodes the geometry, we compute the entropic optimal transport plan:

$$
P ^ { s , * } = \underset { P \in \Pi ( p , q ^ { s } ) } { \arg \operatorname* { m i n } } \left[ \langle P , C ^ { s } \rangle + \varepsilon \sum _ { i , j } P _ { i j } ( \log P _ { i j } - 1 ) \right] ,\tag{12}
$$

where $\begin{array} { r } { \Pi ( p , q ^ { s } ) = \{ P \ge 0 \mid P \mathbf { 1 } _ { M } = \frac { 1 } { N } \mathbf { 1 } _ { N } , P ^ { \top } \mathbf { 1 } _ { N } = q ^ { s } \} } \end{array}$ denotes the set of valid couplings, and $\varepsilon > 0$ is the entropic regularization hyperparameter. The optimal transport plan $P ^ { s , * }$ is efficiently computed by Sinkhorn algorithm (see Appendix C for more details). By keeping the critic’s influence isolated within the $q ^ { s }$ -marginal constraints and the spatial structure isolated within the cost matrix, OptiFlow ensures that probability mass is allocated proportionally to high-value action modes without compromising the geometric shape of the action distribution. This is in contrast to existing behaviorregularized offline RL methods that rely on an explicit behavioral penalty term. While strong regularization prevents out-of-distribution actions, it forces the policy to cover suboptimal dataset actions. OptiFlow circumvents this trade-off by leveraging the optimal transport coupling for one-step policy extraction. Rather than relying on a global behavioral penalty, the transport plan directly map the one-step policy only to high-value modes, naturally discarding suboptimal candidates.

Step 4: Transport-guided distillation. Given the optimal coupling $P ^ { s , * }$ , the final step is to learn the efficient one-step policy through a structured, transport-guided distillation. While the optimal coupling provides a soft assignment, practical one-step policy learning requires explicit, unambiguous regression targets. Thus, we extract a per-row reference action index via maximum assignment:

$$
j _ { i } ^ { * } \in \arg \operatorname* { m a x } _ { j } P _ { i j } ^ { s , * } , \qquad i = 1 , \ldots , N .\tag{13}
$$

Algorithm 1 OptiFlow training   
1: for each training iteration do   
2: Sample a mini-batch $\boldsymbol { B } = \{ ( s , a , r , s ^ { \prime } ) \}$ from D   
3: Update critic $Q _ { \phi }$ using $\nabla _ { \phi } \dot { \mathcal { L } } _ { Q } ( \phi )$ (Eq. (5))   
4: Update reference flow policy $\mu _ { \omega }$ using $\bar { \nabla } _ { \omega } \mathcal { L } _ { \mathrm { r e f } } ( \omega )$ (Eq. (6))   
5: for each state $s \in B$ do   
6: Sample one-step actions $\{ a _ { \theta , i } ^ { s } \} _ { i = 1 } ^ { N }$ from $\mu _ { \boldsymbol { \theta } } ( s , \cdot )$   
7: Sample reference actions $\{ \tilde { a } _ { \omega , j } ^ { s } \} _ { j = \mathrm { : } } ^ { M }$ <sub>1</sub> from $\mu _ { \omega } ( s , \cdot )$   
8: Set $\begin{array} { r } { p _ { i } = \frac { 1 } { N } } \end{array}$ and compute the value-weighted reference-policy marginal $q _ { j } ^ { s }$ by Eq. (10)   
9: Construct $\begin{array} { r } { \dot { \boldsymbol { C } } _ { i j } ^ { s } = \frac { 1 } { \bar { c } ^ { s } } \| \boldsymbol { a } _ { \theta , i } ^ { s } - \tilde { \boldsymbol { a } } _ { \omega , j } ^ { s } \| _ { 2 } ^ { 2 } } \end{array}$   
10: Compute $P ^ { s , * } \boldsymbol { \mathrm { b y } }$ Sinkhorn on $( \ddot { C } ^ { s } , p , q , \varepsilon , T )$ (Alg. 2)   
11: Set $j _ { i } ^ { * } = \arg$ max<sub>j</sub> $P _ { i j } ^ { s , * }$   
12: end for   
13: Update one-step policy $\mu _ { \theta }$ using $\nabla _ { \theta } \mathcal { L } _ { \mathrm { d i s t i l l } } ( \theta )$ (Eq. (14))   
14: Update target critic $Q _ { \bar { \phi } }$   
15: end for

The one-step policy is then updated by minimizing the mass-weighted distillation loss:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { d i s t i l l } } ( \theta ) = \mathbb { E } _ { s \sim \mathcal { D } } \left[ \sum _ { i = 1 } ^ { N } P _ { i , j _ { i } ^ { * } } ^ { s , * } \left\| \mu _ { \theta } ( s , z _ { i } ) - \tilde { a } _ { \omega , j _ { i } ^ { * } } ^ { s } \right\| _ { 2 } ^ { 2 } \right] .\tag{14}
$$

During this update, the reference actions $\tilde { a } _ { \omega , j _ { i } ^ { * } } ^ { s }$ serve as fixed distillation targets, while the transport weights $P _ { i , j _ { i } ^ { * } } ^ { s , * }$ and hard assignments $j _ { i } ^ { * }$ are treated as constants; gradients flow exclusively through the one-step-policy actions $\mu _ { \boldsymbol { \theta } } ( s , z _ { i } )$ . Unlike pointwise distillation that simply matches shared noise inputs and action outputs and often collapses multimodal behaviors into a compromised mean, each distillation target in OptiFlow is explicitly routed to a geometrically aligned, distinct high-value mode via optimal transport, preserving the multimodal structure.

Algorithm Summary. To sum up, the full OptiFlow (Algorithm 1) extracts an effective one-step policy that preserves the multimodal structure of high-value behaviors. It leverages a learned critic (Step 1) and a reference multi-step flow policy (Step 2) to solve per-state optimal transport problems (Step 3). By decoupling value guidance (target marginals) from behavioral geometry (cost matrix), the final transport-guided distillation (Step 4) explicitly maps noise to distinct, in-distribution high-value action modes. Notably, OptiFlow avoids direct critic maximization, which often exploits unreliable out-of-distribution regions. Instead, it anchors the policy to high-return in-distribution targets via transport plan. This structural separation mitigates the need for explicit behavior regularization while ensuring stable and mode-preserving one-step policy extraction.

## 5 Experiments

Our experiments are designed to answer three questions: (i) whether OptiFlow preserves multimodal action structure in controlled settings, (ii) which design choices are responsible for the gains, including the use of a value-weighted reference-policy marginal and an OT coupling, and (iii) whether valueweighted, transport-guided distillation improves offline RL performance over conventional and flow-based baselines.

## 5.1 Controlled Multimodal Experiments

Bandit diagnostic. Figure 2 illustrates three requirements for offline policy distillation: preserving multimodality, avoiding suboptimal target assignments, and remaining within behavior support. Experimental results of FQL [16] highlight the trade-off between behavior matching and value maximization: emphasizing the dataset regularization term (FQL-D in Figure 2d) biases the policy toward the data distribution but under-prioritizes high-value modes, whereas emphasizing the critic maximization term (FQL-Q in Figure 2e) can deviate from behavior support due to out-of-distribution value overestimation. Gaussian-based policies (Figure 2c) with direct critic maximization can similarly drift outside behavior support under value overestimation and fail to capture multimodality.

![](images/22bf33a4b2cc26c145515a1b71c65520949cb6fe75b8a9e2c82bcc09aac71525.jpg)  
(a) True Reward

![](images/f32606597cee597d59b7c5619f198b09bf861b590225e78eff1e183e268dd84e.jpg)  
(b) BC

![](images/aed6a2430a124f8ed6e5a3c825514cba0d460b7748d1ee2372a4f1f499f554d1.jpg)  
(c) ReBRAC  
w/ Gaussian Policy

![](images/ccd32c54720411280075d9c025b522fcc605c66f4df9b2e5ce946cf03de868ed.jpg)  
(d) FQL-D

![](images/ce5e7ab8ef23387629268b842ecf11223d0a609219dd471e49375e47062da0b5.jpg)  
(e) FQL-Q

![](images/ac5fbd0afea8be5804ff3361e548e39fd8ec0d112e7306506d7581412929a0fc.jpg)  
(f) OptiFlow (ours)

Figure 2: Bandit multimodal diagnostic. A bimodal Reward landscape illustrating two highvalue modes and a suboptimal region. The figure evaluates whether different methods preserve multimodality, avoid suboptimal assignments, and remain within behavior support. The experimental setup is detailed in Appendix D

In contrast, OptiFlow uses a mechanism in which the critic shapes the reference-policy marginal, while the OT coupling determines pairings according to the action-distance cost. This combination enables structured supervision that preserves multimodality (Figure 2f). Even when suboptimal actions appear among the reference samples, they are rarely selected after the row-wise maximummass assignment. Finally, because the targets are drawn from the behavior-regularized flow reference policy, OptiFlow remains grounded in dataset support while preserving multiple high-value modes.

Why OT coupling and value-weighted marginals? Figure 3 compares several alternatives for offline policy distillation under asymmetric and bimodal value landscapes. Value-weighted distillation (VWD) guides the one-step policy toward high-value targets, but collapses toward a single dominant mode (Figure 3b,h). Top-K nearest-neighbor distillation partially preserves multimodality by selecting multiple reference actions, but still struggles to fully capture the multimodal structure (Figure 3c,d,i,j). These results suggest that local or row-wise matching alone is insufficient for preserving the distribution over optimal modes.

![](images/4392cf97568494891ae814c0998f5a99ee8cb505f399368b6e2b3ee109bca60f.jpg)  
(a) Asym. Q

![](images/8db23f2f59b3946c7fb9a38afa3cebc8eea39051761bebba5dabdb8010f153a5.jpg)  
(b) VWD

![](images/7bafdb1daa2201486c2ecceee3b3a468ffcc22a86dc06ab1eab06eda4ae95c77.jpg)  
(c) Top1-NN

![](images/d89c11a1047f8db8e902f825ab6ec230aea3ae937a0755bf624d4a1de62377ad.jpg)  
(d) Top3-NN

![](images/a82fc43a2d501b3744f16bc3617406496c5ef2f5a652156bf85e01d7756e9e96.jpg)  
(e) Q-cost OT

![](images/405b28ee8ee3eb3f44b587ed4b7e68605c5d0dd8aa776de3b6954a758b2d7ec5.jpg)  
(f) OptiFlow (ours)

![](images/c2e5f32ff0989446e72269cd2c50547c6d2a8ce858ecadd2947ea18440b7d840.jpg)  
(g) Bimodal Q

![](images/1e508bf260d9510cf3abf6addc2ae02384c236c49899b6a1332cbb73ecf7cfa5.jpg)  
(h) VWD

![](images/9990e81d90413bfd6613a24e8e102cef279c4e6ab66d0a43be10bd8f37864055.jpg)  
(i) Top1-NN

![](images/0a33971243fac1c9cb8ba08f810575c672a87951449e4dcd1a246bafc4a2ec02.jpg)  
(j) Top3-NN

![](images/5ec8f5ff2d6e5d7cbf659a62e86cc2c0c9c300dc73ba73d6e7082a93dfbe6035.jpg)  
(k) Q-cost OT

![](images/df12b926d776255a2d90f8e9a76464ef1a9505140e25ac12d607ad301dbabe95.jpg)  
(l) OptiFlow (ours)

Figure 3: Bandit diagnostic for value-aware transport. (a,g) show asymmetric and bimodal value landscapes. The figure evaluates whether different distillation strategies can capture high-value regions while preserving the distribution over optimal modes.

A more structured alternative is to introduce OT coupling between one-step-policy and referencepolicy samples. However, not all OT formulations behave similarly. One natural alternative approach of our method is to keep the reference-policy marginal uniform while injecting value directly into the transport cost, $C _ { i j } = \bar { f } ( d _ { i j } , Q ( s , a _ { \theta , i } ^ { s } ) , Q ( \bar { s } , \tilde { a } _ { \omega , j } ^ { s } ) )$ . Although this formulation trades off value and distance, Figure 3e,k shows that it still misallocates samples because the uniform reference marginal forces every reference action, including low-value ones, to receive equal total mass.

OptiFlow instead places value into the reference-policy marginal itself: high-value reference actions receive more supervision mass, while OT coupling determines geometrically compatible assignments between one-step and reference-policy samples. As shown in Figure 3f,l, this asymmetric marginal de sign preserves multimodal structure while concentrating supervision on high-value, dataset-supported regions.

We additionally provide a counterexample illustrating a structural limitation of uniform-marginal OT, generalize this limitation beyond the counterexample, and empirically demonstrate the importance of the reference marginal on real-world benchmarks in Appendix A.2. We describe the value-weighted, Top-K nearest-neighbor distillation formulations and more general value-, distance-aware distillation formulation in Appendix A.3, and discuss their failure modes on mode collapse to motivate the need for OptiFlow formulation.

![](images/8137dc8fad5ed9d55d8fc45352ba6b89c0d0258278e62eea51b77c6aa3f09f9a.jpg)  
(a) Dataset Distribution

![](images/b57b932700edfdf3daed349b48cdc054ad4a461abe6ecb4b8114491720bf8b7c.jpg)  
(b) FQL (α = 0.1)

![](images/480df8fc4b392aea598c56eb4325068fe18648b01e67f441e6376518079c6aaf.jpg)  
(c) FQL (α = 1)

![](images/1b2a5ba5fec1038e9a617cb9714913be7b954eb1720c0b0b87b34ec92521548d.jpg)  
(d) FQL (α = 10)

![](images/0736809bcfa783267441a4886b289a16cb320b149c157cf126060ff4a3e2f421.jpg)  
(e) OptiFlow-BC  
Figure 4: Four-mode environment for evaluating multimodal behavior. (a) Offline dataset distribution with intentionally uneven coverage across the four goal directions. (b–e) Learned policy distributions overlaid on the reward landscape. (b–d) FQL with increasing distillation strength α, exhibiting a diversity–quality trade-off. (e) OptiFlow-BC, maintaining balanced coverage across the four equally optimal modes.

Beyond the bandit setting. We further examine the preservation of multimodal behavior in a four-goal MDP introduced by [31], as illustrated in Figure 4, where the four goals constitute equally optimal modes but are represented unevenly in the offline dataset (Figure 4a). This setting tests whether the one-step policy extraction preserves multiple high-value modes despite imbalanced data coverage. To isolate the effect of transport-guided distillation from value-aware reference-policy training, we use OptiFlow-BC, a controlled variant of OptiFlow that replaces the VaBC reference policy with a standard BC reference while retaining the same value-weighted OT coupling and transport-guided distillation. OptiFlow-BC (Figure 4e) achieves balanced coverage of the four optimal modes while attaining the highest return, demonstrating that the policy extraction mechanism can preserve multimodal behavior despite imbalanced mode frequencies in the offline dataset. In contrast, FQL exhibits a clear diversity–quality trade-off (Figure 4b–d): stronger distillation improves multimodal coverage but progressively degrades policy return.

We also observe that the mode-level transport assignments remain consistent across repeated resampling of the reference actions, suggesting that the stochastic candidate set does not induce frequent assignment switches across distinct modes. The environment and reference-resampling protocols for assignment stability experiment are detailed in Appendix D.

## 5.2 Benchmark Experimental Setup

Benchmarks. We evaluate OptiFlow on reward-based single-task variants of OGBench [32] and on D4RL AntMaze and Adroit [33]. OGBench includes navigation and manipulation tasks with complex and often multimodal action distributions, while D4RL provides sparse-reward navigation and dexterous manipulation benchmarks. Full per-task results are provided in Appendix E.

Baselines. We compare against both conventional offline RL baselines and expressive generativepolicy baselines. BC, IQL [4], ReBRAC [34], serve as standard offline RL references, while IDQL [14], IFQL, FQL [16] and GFP [17] represent the most closely related baselines based on expressive generative policies.

Training and evaluation. For state-based OGBench tasks, all methods are trained for 1M gradient steps. For D4RL and OGBench visual tasks, all methods are trained for 500K gradient steps. We report final-checkpoint performance averaged over 8 random seeds unless otherwise noted. For visual tasks, due to high computational cost, we report performance averaged over 4 random seeds and over checkpoints at 300K, 400K, and 500K steps, following the evaluation protocol of FQL [16]. Full training and evaluation details are provided in Appendix D and the result-table captions.

Table 1: Aggregate performance across OGBench families and D4RL domains. For state-based OGBench, each family row reports the mean ± standard error over the five single-task variants in that family. Visual OGBench rows report the corresponding visual single-task environments. D4RL rows aggregate over all tasks in the corresponding domain. Values within 95% of the best score in each row are in bold. Dashes denote unreported entries. Italicized entries indicate results taken from prior work [16], with complete per-task result tables in Appendix E.
<table><tr><td>Family / Domain</td><td>BC</td><td>IQL</td><td>ReBRAC</td><td>IDQL</td><td>IFQL</td><td>FQL</td><td>GFP</td><td>II OptiFlow</td></tr><tr><td colspan="9">OGBench —Navigation</td></tr><tr><td>AntMaze-Large {1-5}</td><td> $1 . 7 { \pm } 0 . 2 $ </td><td> $5 3 . 0 { \pm } 2 . 3 $ </td><td> $8 6 . 0 { \pm } 1 . 5 $ </td><td> $2 6 . 8 { \pm } 1 . 4 $ </td><td> $2 8 . 0 { \pm } 2 . 5 $ </td><td> $7 9 . 6 { \pm } 1 . 0$ </td><td> ${ \bf 9 3 . 9 2 0 . 5 }$ </td><td> ${ \bf 9 1 . 9 { \pm } 0 . 5 }$ </td></tr><tr><td>AntMaze-Giant {1-5}</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $4 . 3 { \pm } 0 . 8$ </td><td> $3 2 . 4 \pm 3 . 3$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 9 { \pm } 0 . 4$ </td><td> $9 . 5 { \pm } 2 . 3 $ </td><td> $1 8 . 9 2 2 . 5$ </td><td> $2 6 . 5 { \pm } 1 . 5$ </td></tr><tr><td>Humanoid-Medium {1-5}</td><td> $1 . 4 \pm 0 . 2$ </td><td> $3 3 . 8 { \pm } 1 . 4 $ </td><td> $1 9 . 1 { \pm } 2 . 4$ </td><td> $3 . 0 { \pm } 0 . 3$ </td><td> $5 8 . 0 { \pm } 4 . 2 $ </td><td> $5 7 . 4 { \pm } 1 . 6 $ </td><td> ${ \bf 7 1 . 8 \pm 1 . 0 }$ </td><td> $7 4 . 4 \pm 1 . 2$ </td></tr><tr><td>Humanoid-Large {1-5}</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $2 . 3 { \pm } 0 . 3$ </td><td> $2 . 7 { \pm } 0 . 5$ </td><td> $0 . 4 \pm 0 . 1$ </td><td> $1 2 . 7 { \pm } 0 . 9$ </td><td> $3 . 6 \pm 0 . 7$ </td><td> $1 1 . 0 { \pm } 1 . 8$ </td><td> ${ \bf 1 9 . 1 \pm 1 . 1 }$ </td></tr><tr><td>AntSoccer-Arena {1-5}</td><td> $0 . 2 { \pm } 0 . 1$ </td><td> $9 . 4 \pm 0 . 6$ </td><td> $0 . 1 { \pm } 0 . 1$ </td><td> $3 . 5 { \pm } 0 . 5$ </td><td> $2 9 . 1 { \pm } 3 . 1 $ </td><td> ${ \bf 6 1 . 4 \pm 1 . 5 }$ </td><td> ${ \bf 6 1 . 1 \pm 1 . 3 }$ </td><td> $5 2 . 7 { \pm } 1 . 6 $ </td></tr><tr><td>Navigation Avg.</td><td> $0 . 7 { \pm } 0 . 1$ </td><td> $2 0 . 6 { \pm } 0 . 6 $ </td><td> $2 8 . 1 { \pm } 0 . 9$ </td><td> $6 . 7 { \pm } 0 . 3$ </td><td> $2 5 . 8 { \pm } 1 . 2$ </td><td> $4 2 . 3 { \pm } 0 . 7$ </td><td> ${ \bf 5 1 . 3 2 0 . 7 }$ </td><td> ${ \bf 5 2 . 9 2 0 . 5 }$ </td></tr><tr><td colspan="9">OGBench — Manipulation</td></tr><tr><td>Cube-Single {1-5}</td><td> $2 . 8 \pm 1 . 0$ </td><td> $5 0 . 1 \pm 2 . 5$ </td><td> $8 7 . 8 { \pm } 1 . 2 $ </td><td> $9 1 . 9 { \pm } 0 . 5 $ </td><td> $8 3 . 9 { \pm } 0 . 8 $ </td><td> $9 1 . 0 { \pm } 1 . 1$ </td><td> ${ \bf 9 6 . 3 { \pm 0 . 5 } }$ </td><td> ${ \bf 9 7 . 6 { \pm } } 0 . 3$ </td></tr><tr><td>Cube-Double {1-5}</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $2 . 4 \pm 0 . 2$ </td><td> $8 . 1 \pm 1 . 3$ </td><td> $2 . 8 { \pm } 0 . 7$ </td><td> $1 0 . 5 { \pm } 0 . 7 $ </td><td> $2 5 . 2 { \pm } 1 . 5$ </td><td> $4 3 . 3 { \pm } 1 . 6 $ </td><td> ${ \bf 4 6 . 1 \pm 1 . 3 }$ </td></tr><tr><td>Cube-Triple {1-5}</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 5 { \pm } 0 . 2$ </td><td> $2 . 2 { \pm } 0 . 4$ </td><td> $0 . 4 \pm 0 . 2$ </td><td> $0 . 2 { \pm } 0 . 1$ </td><td> $1 . 1 { \pm } 0 . 4$ </td><td> $1 . 8 { \pm } 0 . 3 $ </td><td> ${ \bf 3 . 6 \pm 0 . 2 }$ </td></tr><tr><td>Puzzle-3×3 {1-5}</td><td> $3 . 9 { \pm } 0 . 6 $ </td><td> $4 . 3 { \pm } 0 . 5$ </td><td> $2 2 . 3 { \pm } 0 . 5 $ </td><td> $1 4 . 0 { \pm } 0 . 7 \ $ </td><td> $1 9 . 3 { \pm } 0 . 4$ </td><td> ${ \bf 2 9 . 5 { \pm 0 . 8 } }$ </td><td> $2 4 . 4 { \pm } 1 . 1$ </td><td> ${ \bf 2 8 . 1 \pm 2 . 3 }$ </td></tr><tr><td>Puzzle-4×4 {1-5}</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 7 { \pm } 0 . 2 $ </td><td> $1 4 . 0 { \pm } 0 . 5 $ </td><td> $2 2 . 1 \pm 1 . 2$ </td><td> $2 6 . 8 { \pm } 1 . 1$ </td><td> $1 5 . 0 { \pm } 0 . 8 $ </td><td> $2 5 . 8 { \pm } 1 . 1$ </td><td> ${ \bf 3 2 . 1 \pm 1 . 5 }$ </td></tr><tr><td>Scene-Play {1-5}</td><td> $0 . 4 \pm 0 . 1$ </td><td> $2 0 . 0 { \pm } 0 . 8$ </td><td> $4 2 . 0 { \pm } 1 . 5 $ </td><td> $3 9 . 1 { \pm } 0 . 3 $ </td><td> $4 4 . 7 { \pm } 1 . 7$ </td><td> $5 5 . 8 { \pm } 1 . 4 $ </td><td> $5 3 . 1 \pm 1 . 6$ </td><td> ${ \bf 5 9 . 4 } \pm \mathrm { 0 . 3 }$ </td></tr><tr><td>Manipulation Avg.</td><td> $1 . 2 { \pm } 0 . 2 $ </td><td> $1 3 . 2 { \pm } 0 . 5 $ </td><td> $2 9 . 4 { \pm } 0 . 4$ </td><td> $2 8 . 4 { \pm } 0 . 3 $ </td><td> $3 0 . 9 { \pm } 0 . 4$ </td><td> $3 6 . 3 { \pm } 0 . 4$ </td><td> $4 0 . 8 { \pm } 0 . 5 $ </td><td> $4 4 . 5 { \pm } 0 . 5 $ </td></tr><tr><td colspan="9">OGBench — Visual</td></tr><tr><td>Visual-Cube-Single</td><td></td><td> $\gamma \boldsymbol { O } \pm \boldsymbol { \epsilon }$ </td><td> $g \mathcal { 3 } \pm \mathcal { 3 }$ </td><td></td><td> $\mathbf { \nabla } _ { 4 } 9 \pm \mathbf { \nabla } _ { 3 } . 5$ </td><td> $\boldsymbol { 8 1 \pm \sigma }$ </td><td> $7 2 . 2 { \pm } 3 . 6 $ </td><td> ${ \bf 8 6 . 6 \pm 1 . 3 }$ </td></tr><tr><td>Visual-Cube-Double</td><td></td><td> $\mathcal { B } \mathcal { 4 } \pm \pi \ : 1 \ : . 5$ </td><td> $\not = 2$ </td><td></td><td> $\delta \pm \it { 3 }$ </td><td> $\textit { 2 1 } \pm 5 . 5$ </td><td> $1 2 . 0 { \pm } 2 . 6 $ </td><td> ${ \bf 6 6 . 7 \pm 3 . 5 }$ </td></tr><tr><td>Visual-Scene-Play</td><td></td><td> ${ \pm } 1$ </td><td> ${ \pm } 2$ </td><td></td><td> $\boldsymbol { 8 6 \pm 5 }$ </td><td> $g _ { 8 \pm 1 . 5 }$ </td><td> $\mathbf { 9 8 . 7 \pm 0 . 3 }$ </td><td> $9 2 . 8 { \pm } 0 . 8 $ </td></tr><tr><td>Visual-Puzzle-3×3</td><td></td><td> $\gamma _ { \pm 7 , 5 }$ </td><td> $g \delta \pm 2$ </td><td></td><td> ${ \mathbf { } } I \pmb { 0 } \pmb { \mathrm { \pm } } \mathbf { \mathrm { \Omega } }$ </td><td> $g _ { 4 } \pm 0 . 5$ </td><td> $8 6 . 5 { \pm } 1 . 1$ </td><td> $9 3 . 8 { \pm } 1 . 2 $ </td></tr><tr><td>Visual-Puzzle-4×4</td><td></td><td> $O \pm o$ </td><td> ${ \mathcal { Q } } \delta \pm { \mathcal { s } }$ </td><td></td><td> $\delta \pm \tau . 5$ </td><td> ${ 3 3 \pm 3 }$ </td><td> $1 9 . 3 { \pm } 1 . 4 $ </td><td> ${ \bf 3 9 . 7 \pm 4 . 8 }$ </td></tr><tr><td colspan="9">Overall</td></tr><tr><td>OGBench Visual  $\operatorname { A v g } .$ </td><td></td><td> $\textit { 4 1 . 4 } \pm \textit { 3 . 0 }$ </td><td> $5 9 . 8 \pm { } 1 . { } 1$ </td><td></td><td> $5 0 . 2 \pm 2 . 0$ </td><td> $6 5 . 4 \pm { \ : } 1 . 8$ </td><td> $5 7 . 7 { \pm } 1 . 0$ </td><td> ${ \bf 7 5 . 9 \pm 1 . 3 }$ </td></tr><tr><td>OGBench State-Based  $\operatorname { A v g } .$ </td><td> $0 . 9 { \pm } 0 . 1$ </td><td> $1 6 . 5 { \pm } 0 . 4 $ </td><td> $2 8 . 8 { \pm } 0 . 5$ </td><td> $1 8 . 5 { \pm } 0 . 2 $ </td><td> $2 8 . 6 { \pm } 0 . 6 $ </td><td> $3 9 . 0 { \pm } 0 . 4$ </td><td> $4 5 . 6 { \pm } 0 . 4$ </td><td> ${ \bf 4 8 . 3 { \scriptstyle \pm 0 . 4 } }$ </td></tr><tr><td>AntMaze (D4RL) Avg.</td><td> $_ { 1 6 . 7 }$ </td><td>58.4</td><td> $\ 7 6 . 9$ </td><td> $\gamma 9 . 1$ </td><td> $6 1 . 0 { \pm } 2 . 5$ </td><td> ${ \bf 8 2 . 3 \pm 1 . 3 }$ </td><td> ${ \bf 8 2 . 6 \pm 1 . 5 }$ </td><td> ${ \bf 8 6 . 6 { \pm } } 0 . 7$ </td></tr><tr><td>Adroit (D4RL) Avg.</td><td>36.7</td><td>53.5</td><td>58.6</td><td> $5 0 . 2 { \pm } 0 . 4 $ </td><td> $4 8 . 2 { \pm } 0 . 5 $ </td><td> $5 0 . 7 { \pm } 0 . 5$ </td><td> $5 2 . 1 { \pm } 0 . 5 $ </td><td> $5 0 . 5 { \pm } 0 . 4 $ </td></tr></table>

Method details. OptiFlow uses a value-aware reference flow policy and a one-step flow policy. For each state, we sample $N = 1 6$ one-step-policy actions and $M = 6 4$ reference-policy actions, construct a value-weighted reference-policy marginal, compute an entropic Optimal Transport plan with 30 Sinkhorn iterations, select a reference action for each one-step-policy sample via the maximum transport assignment in the corresponding row, and distill the resulting assignments into the one-step policy. Additional implementation details and full hyperparameters are given in Appendix D.

## 5.3 Main Results

Aggregate benchmark performance. Table 1 summarizes performance across OGBench families and D4RL domains. OptiFlow achieves the best overall state-based OGBench average, with $4 8 . 3 { \pm } 0 . 4 $ compared to $4 5 . 6 \pm 0 . 4$ for GFP and $3 9 . 0 \pm 0 . 4$ for FQL. The gains are most pronounced on OGBench manipulation, where OptiFlow obtains the strongest aggregate performance, improving the manipulation average to 44.5 ± 0.5 compared to $4 0 . 8 \pm 0 . 5$ for GFP and $3 6 . 3 \pm 0 . 4$ for FQL. These results show that value-weighted transport-guided distillation improves over the closest flow-based baselines in the setting most aligned with OptiFlow’s motivation: learning efficient one-step policies from complex and multimodal action distributions.

OGBench manipulation and navigation. OptiFlow obtains the best aggregate score on five of the six OGBench manipulation families: Cube-Single, Cube-Double, Cube-Triple, Puzzle-4×4, and Scene-Play. On OGBench navigation, OptiFlow remains competitive: it improves over FQL on AntMaze-Giant, Humanoid-Medium, and Humanoid-Large. Overall, OptiFlow’s strongest gains appear in tasks where multimodal reference-policy structure is useful for one-step policy learning.

D4RL results. On D4RL, OptiFlow achieves the best AntMaze average, improving over GFP by 4.0 points and FQL by 4.3 points. On Adroit, OptiFlow is competitive with flow-based baselines. These results suggest that OptiFlow transfers beyond OGBench, with its clearest advantage in settings where useful multimodal reference actions can be transferred to a one-step policy through structured assignment.

Sensitivity analysis. We provide additional sensitivity analyses of OptiFlow with respect to the value-weighting temperatures τ and η (Appendix F.1), the entropic regularization coefficient and the number of Sinkhorn iterations (Appendix F.2), and the numbers of one-step and reference-policy action samples (Appendix F.3).

Additional ablations and diagnostics. We first control for OptiFlow’s use of multiple action samples per state and show that increased sampling alone does not explain its gains (Appendix F.4). We then disentangle the contributions of the reference policy and transport-guided policy extraction (Appendix F.5) and compare our default VaBC weighting with an AWR-style alternative (Appendix F.6). Finally, we examine whether direct critic maximization provides benefits beyond transport-based supervision (Appendix F.7) and demonstrate strong offline-to-online fine-tuning performance (Appendix G).

## 6 Conclusion

We presented One-step Flow Policy via Optimal Transport (OptiFlow), a value-weighted optimal transport method to learn efficient one-step flow policies for offline reinforcement learning. OptiFlow uses a value-aware reference flow policy to represent multimodal, data-supported behavior, and trains a one-step flow policy through transport-guided distillation. By constructing an entropic OT coupling between one-step-policy and reference-policy samples, OptiFlow separates value and geometry: critic-estimated values determine how much supervision mass each reference action receives, while action-space distance determines how one-step samples are assigned. This provides a structured alternative to independent noise-wise distillation and direct critic maximization.

Experimental results show that OptiFlow preserves multimodal action structure in controlled diagnostics and achieves strong performance across offline RL benchmarks, with especially clear gains on OGBench manipulation tasks. These results support the central claim that value-guided one-step policy learning benefits from transport-based distillation.

More broadly, OptiFlow suggests that critic information in offline RL need not be used only through direct critic maximization. It can instead shape structured supervision for policy learning, opening a path toward value-guided generative policy learning methods that preserve multimodal behavior while retaining efficient one-step policy deployment.

OptiFlow also has several limitations. Compared to standard offline RL methods, especially existing flow-based policies, it introduces additional computational overhead from repeated policy sampling and Sinkhorn-based optimal transport updates. However, these operations mainly consist of matrixbased computations that can be efficiently parallelized on modern GPUs (see Appendix A.4 for detailed analysis of computational overhead of OptiFlow).

## Acknowledgments and Disclosure of Funding

This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (No. RS-2020-II201361, Artificial Intelligence Graduate School Program (Yonsei University), RS-2024-00457882, AI Research Hub Project) and the National Research Foundation of Korea (NRF) grants funded by the Korea government(MSIT) (RS-2026-25505230, RS-2026-25619452). This research was also supported by a grant from the Institute for AI and Social Innovation at Yonsei University and the Yonsei University Research Fund of 2026-22-0204.

## References

[1] Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch Reinforcement Learning, pages 45–73. Springer Berlin Heidelberg, Berlin, Heidelberg, 2012. ISBN 978-3-642-27645-3. doi: 10.1007/978-3-642-27645-3\_2.

[2] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. CoRR, abs/2005.01643, 2020.

[3] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in neural information processing systems, 2020.

[4] Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. In International Conference on Learning Representations, 2022.

[5] Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning. arXiv preprint arXiv:1911.11361, 2019.

[6] Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. In Thirty-Fifth Conference on Neural Information Processing Systems, 2021.

[7] Jongmin Lee, Wonseok Jeon, Byung-Jun Lee, Joelle Pineau, and Kee-Eung Kim. Optidice: Offline policy optimization via stationary distribution correction estimation, 2021. URL https: //arxiv.org/abs/2106.10783.

[8] Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. Algaedice: Policy gradient from arbitrary experience, 2019. URL https://arxiv.org/abs/ 1912.02074.

[9] Seohong Park, Kevin Frans, Sergey Levine, and Aviral Kumar. Is value learning really the main bottleneck in offline rl? Advances in Neural Information Processing Systems, 37:79029–79056, 2024.

[10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[12] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[13] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[14] Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine. Idql: Implicit q-learning as an actor-critic method with diffusion policies, 2023.

[15] Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. arXiv preprint arXiv:2208.06193, 2022.

[16] Seohong Park, Qiyang Li, and Sergey Levine. Flow q-learning. In International Conference on Machine Learning (ICML), 2025.

[17] Franki Nguimatsia Tiofack, Theotime Le Hellard, Fabian Schramm, Nicolas Perrin-Gilbert, and Justin Carpentier. Guided flow policy: Learning from high-value actions in offline reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026.

[18] Yejun Jang, Hong Chul Nam, Jeong Min Park, GIMIN BAE, and Hyun Kwon. Q-guided flow q-learning. In CoRL 2025 Workshop RemembeRL, 2025.

[19] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

[20] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024.

[21] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

[22] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

[23] Ziyu Wang, Alexander Novikov, Konrad Zolna, Josh S Merel, Jost Tobias Springenberg, Scott E Reed, Bobak Shahriari, Noah Siegel, Caglar Gulcehre, Nicolas Heess, et al. Critic regularized regression. Advances in Neural Information Processing Systems, 33:7768–7778, 2020.

[24] Yuhang Ran, Yi-Chen Li, Fuxiang Zhang, Zongzhang Zhang, and Yang Yu. Policy regularization with dataset constraint for offline reinforcement learning. In International conference on machine learning, pages 28701–28717. PMLR, 2023.

[25] Yicheng Luo, Zhengyao Jiang, Samuel Cohen, Edward Grefenstette, and Marc Peter Deisenroth. Optimal transport for offline imitation learning. arXiv preprint arXiv:2303.13971, 2023.

[26] Wei-Di Chang, Scott Fujimoto, David Meger, and Gregory Dudek. Imitation learning from observation through optimal transport. arXiv preprint arXiv:2310.01632, 2023.

[27] Arip Asadulaev, Rostislav Korst, Alexander Korotin, Vage Egiazarian, Andrey Filchenkov, and Evgeny Burnaev. Rethinking optimal transport in offline reinforcement learning. Advances in Neural Information Processing Systems, 37:123592–123607, 2024.

[28] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018.

[29] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.

[30] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

[31] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International conference on machine learning, pages 1352–1361. PMLR, 2017.

[32] Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goal-conditioned rl. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 94937–94982, 2025.

[33] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

[34] Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, and Sergey Kolesnikov. Revisiting the minimalist approach to offline reinforcement learning. Advances in Neural Information Processing Systems, 36:11592–11620, 2023.

[35] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

[36] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[37] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

[38] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. In International Conference on Machine Learning (ICML), 2018.

[39] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

## Appendix Contents

A Discussion 14   
A.1 Limitations 14   
A.2 Why use a value-aware reference marginal in optimal transport? 15   
A.3 Alternative distillation methods and why OptiFlow? 16   
A.4 Computational cost 18   
B Broader Impacts 19   
C Sinkhorn Algorithm 20   
C.1 Algorithm 20   
D Implementation Details 20   
E Complete Experimental Results 24   
F Ablation Studies 26   
F.1 Sensitivity to value-weighting temperatures 26   
F.2 Sensitivity to Sinkhorn hyperparameters 26   
F.3 Reference and one-step action sampling . 27   
F.4 Controlling for the action-sampling budget 27   
F.5 Disentangling reference policy quality and distillation 28   
F.6 AWR-style weighting for value-aware behavior cloning . 28   
F.7 Direct Q-maximization 29   
G Offline-to-Online Fine-tuning 30

## A Discussion

In this section, we provide additional analysis and discussion of OptiFlow. We first discuss its limitations and then examine two key aspects of its transport-based formulation. We show why value information should be encoded in the reference marginal rather than solely in the transport cost, and analyze alternative distillation strategies to clarify the role of global transport-based matching in preserving multimodality. Finally, we characterize the computational cost of OptiFlow, including its scaling with the number of Sinkhorn iterations and policy samples.

## A.1 Limitations

OptiFlow introduces additional training-time computation because it solves a state-wise entropic optimal transport problem over sampled one-step-policy and reference-policy actions. This requires Sinkhorn iterations during training, making OptiFlow more expensive than purely pointwise distillation or direct actor updates. However, this overhead does not affect deployment: after training, OptiFlow uses only the one-step flow policy for action generation, so inference remains as efficient as other one-step policies. Appendix A.4 discusses the trade-off between Sinkhorn iterations, training cost, and policy performance.

OptiFlow is most beneficial when the reference flow policy provides multiple useful candidate actions and the supervision mass get allocated to advantage. When the reference action distribution is effectively simple, or when a single local target is already sufficient for policy extraction, the advantage over simpler one-step distillation methods may be smaller.

## A.2 Why use a value-aware reference marginal in optimal transport?

We construct a simple $2 \times 2$ example showing that, under a uniform reference marginal, incorporating value information solely through the transport cost is insufficient: regardless of the cost design, the induced transport plan cannot guarantee policy improvement.

Counterexample. Consider a $. 2 \times 2$ transport problem with uniform student and teacher marginals in certain state $s \in S .$

$$
p = \left( { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } \right) , \qquad q = \left( { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } \right) .
$$

The transport plan P must satisfy

$$
\sum _ { j } { P _ { i j } } = p _ { i } , \qquad \sum _ { i } { P _ { i j } } = q _ { j } .
$$

Suppose both target policy’s actions have the same value,

$$
Q ^ { \pi } ( s , a _ { \theta , 1 } ^ { s } ) = Q ^ { \pi } ( s , a _ { \theta , 2 } ^ { s } ) = 7 ,
$$

while the reference policy’s actions have highly asymmetric values,

$$
Q ^ { \pi } ( s , \tilde { a } _ { \omega , 1 } ^ { s } ) = 1 0 , \qquad Q ^ { \pi } ( s , \tilde { a } _ { \omega , 2 } ^ { s } ) = 1 .
$$

Even if the cost function is designed so that both target policy actions prefer the first reference policy’s action, $\mathrm { e . g . }$

$$
c _ { 1 1 } < c _ { 1 2 } , \qquad c _ { 2 1 } < c _ { 2 2 } ,
$$

the uniform reference-side marginal constraint still imposes

$$
P _ { 1 1 } + P _ { 2 1 } = \frac { 1 } { 2 } , \qquad P _ { 1 2 } + P _ { 2 2 } = \frac { 1 } { 2 } .
$$

Therefore, at least half of the total mass must be transported to the low-value reference action $a _ { 2 } ^ { \omega }$ This is independent of how the cost is constructed, as long as the reference-side marginal follows uniform distribution.

Indeed, the expected value induced by any feasible transport plan is fixed by the reference-side marginal:

$$
\sum _ { i , j } P _ { i j } Q ^ { \pi } ( s , \tilde { a } _ { \omega , j } ^ { s } ) = \sum _ { j } q _ { j } Q ^ { \pi } ( s , \tilde { a } _ { \omega , j } ^ { s } ) = \frac { 1 } { 2 } \cdot 1 0 + \frac { 1 } { 2 } \cdot 1 = 5 . 5 .
$$

Thus, any cost formulation cannot improve the target policy in this example, unless the uniform marginal constraint can be mitigated or changed.

In contrast, our softmax-marginal formulation sets the reference-side marginal according to the value,

$$
w _ { j } = \frac { \exp ( Q ^ { \pi } ( s , \tilde { a } _ { \omega , j } ^ { s } ) / \tau ) } { \sum _ { k } \exp ( Q ^ { \pi } ( s , \tilde { a } _ { \omega , k } ^ { s } ) / \tau ) } .
$$

When $\tau$ is small enough, the marginal concentrates on the high-value reference action:

$$
q ^ { \tau ^ { * } } \approx ( 1 , 0 ) .
$$

Then the transport plan

$$
P \approx { \left( \begin{array} { l l } { { \frac { 1 } { 2 } } } & { 0 } \\ { { \frac { 1 } { 2 } } } & { 0 } \end{array} \right) }
$$

is feasible, and both student actions can be guided toward the optimal teacher action. The induced teacher-side value becomes approximately

$$
\sum _ { j } \boldsymbol { q } _ { j } ^ { \tau ^ { * } } \boldsymbol { Q } ^ { \pi } ( s , \tilde { a } _ { \omega , j } ^ { s } ) \approx 1 0 > 7 ,
$$

rather than the uniform-marginal value 5.5. This counterexample supports why the value information should be encoded in the reference-side marginal, not in the transport cost.

Generalization beyond the $2 \times 2$ example. The observation above extends to an arbitrary number of sampled actions. For any state s, let $\{ \hat { \tilde { a } } _ { \omega , j } ^ { s } \} _ { j = 1 } ^ { M }$ denote the reference actions and $q ^ { s }$ their prescribed marginal. For any feasible coupling $P \in \tilde { \Pi } ( \bar { p ^ { s } } , q ^ { s } )$ , we have

$$
\sum _ { i , j } P _ { i j } Q ^ { \pi } ( s , \tilde { a } _ { \omega , j } ^ { s } ) = \sum _ { j } q _ { j } ^ { s } Q ^ { \pi } ( s , \tilde { a } _ { \omega , j } ^ { s } ) ,
$$

since $\textstyle \sum _ { i } P _ { i j } = q _ { j } ^ { s }$ . Thus, the expected value of the reference actions assigned through the coupling is determined solely by the reference-side marginal $q ^ { s } .$ , regardless of the transport cost. Under a uniform reference marginal, this quantity reduces to the average value of the sampled reference actions, regardless of the cost matrix.

In contrast, the value-weighted marginal assigns more mass to higher-value reference actions, where the policy improvement is controlled $\mathbf { b y } \ \tau \colon$ the expected assigned value increases monotonically as τ decreases, from the candidate average at $\tau  \infty$ to the best candidate value at $\tau  0 ,$ , so a sufficiently small τ guarantees improvement whenever some candidiates beats the current policy. We use softmax rater than an argmax for $q ^ { s }$ since the latter concentrates all supervision on one candidate and collapses the multimodal structure, so τ trades assigned value against mode coverage.

Empirical effect of the reference marginal. We further evaluate whether the choice of reference marginal has a corresponding empirical effect. We compare full OptiFlow, which combines VaBC with the value-weighted marginal, against (i) a variant that retains VaBC but replaces the valueweighted marginal with a uniform marginal, and (ii) OptiFlow-BC, which replaces VaBC with standard BC while retaining the value-weighted marginal.

Table 2: Effect of the reference marginal. We compare full OptiFlow against using a uniform reference marginal and against replacing VaBC with standard BC while retaining the value-weighted marginal. We follow the same evaluation protocol as Table 9.
<table><tr><td>Task</td><td>OptiFlow</td><td>VaBC + Uniform</td><td>BC + Value-weighted</td></tr><tr><td>antmaze-large-navigate{1}</td><td> ${ \bf 9 4 . 0 { \scriptstyle \pm 0 . 9 } }$ </td><td> $0 . 2 { \pm } 0 . 2$ </td><td> ${ \bf 9 1 . 0 { \pm } 2 . 0 }$ </td></tr><tr><td>antmaze-giant-navigate{1}</td><td> ${ \pm } 2 . 4 { \pm } 2 . 4$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 9 { \pm } 0 . 3$ </td></tr><tr><td>humanoidmaze-medium-navigate{1}</td><td> $\mathbf { 8 } 2 . 4 \pm 1 . 2$ </td><td> $3 . 6 { \pm } 0 . 5$ </td><td> $5 0 . 5 { \pm } 8 . 3 $ </td></tr><tr><td>humanoidmaze-large-navigate{1}</td><td> $4 8 . 3 { \scriptstyle \pm 2 . 2 }$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $4 3 . 4 { \pm } 4 . 1 $ </td></tr><tr><td>antsoccer-arena-navigate{4}</td><td> ${ \bf 4 6 . 8 { \scriptstyle \pm 4 . 1 } }$ </td><td> $1 1 . 6 { \pm } 2 . 2 $ </td><td> $4 3 . 6 { \pm } 3 . 4 $ </td></tr><tr><td>cube-single-play{2}</td><td> ${ \bf 9 8 . 5 { \scriptstyle \pm 0 . 5 } }$ </td><td> $2 5 . 9 { \pm } 3 . 6 $ </td><td> ${ \bf 9 7 . 5 { \pm 1 . 3 } }$ </td></tr><tr><td>cube-double-play{2}</td><td> $5 2 . 4 2 . 9$ </td><td> $0 . 6 { \pm } 0 . 2$ </td><td> $3 9 . 1 { \pm } 3 . 5 $ </td></tr><tr><td>scene-play{2}</td><td> ${ \bf 9 7 . 5 { \pm 1 . 1 } }$ </td><td> $4 7 . 4 { \pm } 6 . 3$ </td><td> $9 8 . 2 { \scriptstyle \pm 0 . 8 }$ </td></tr><tr><td>puzzle-3x3-play{4}</td><td> $2 5 . 3 { \pm } 6 . 8 $ </td><td> $1 . 8 { \pm } 0 . 4$ </td><td> $1 . 1 { \pm } 0 . 7$ </td></tr><tr><td>puzzle-4x4-play{4}</td><td> $2 2 . 8 { \pm } 2 . 9$ </td><td> $5 . 5 { \pm } 0 . 9$ </td><td> $2 0 . 9 { \pm } 4 . 2 $ </td></tr><tr><td>Average</td><td> $5 7 . 3 { \pm } 1 . 0 $ </td><td> $9 . 7 { \pm } 0 . 7 $ </td><td> $4 8 . 6 { \pm } 1 . 3 $ </td></tr></table>

Table 2 shows that the value-weighted marginal is critical to the performance of OptiFlow. Replacing it with a uniform marginal while retaining VaBC reduces the average performance from 57.3 to 9.7. In contrast, replacing VaBC with standard BC while retaining the value-weighted marginal results in a substantially smaller decrease, from 57.3 to 48.6. These results are consistent with the analysis above: the reference marginal directly controls how much transport mass is allocated to high-value reference actions, whereas the transport cost alone cannot alter the total mass assigned to each reference action.

## A.3 Alternative distillation methods and why OptiFlow?

Here, we analyze alternative distillation strategies such as value-weighted and nearest-neighbor matching, and discuss their failure modes illustrated in Figure 5.

Our method can be viewed as combining ideas from value-weighted distillation (VWD) and Top-K nearest-neighbor (Top-K NN) matching, while addressing their limitations. Define referencepolicy actions $\{ \tilde { a } _ { \omega , j } ^ { s } \} _ { j = 1 } ^ { N }$ , where $\tilde { a } _ { \omega , j } ^ { s } = \mu _ { \omega } ( s , z _ { j } )$ , and one-step policy actions $\{ a _ { \theta , i } ^ { s } \} _ { i = 1 } ^ { N }$ , where

![](images/90215d8c19228e2adfbd7b9af65ba53206af929672cc7eb199c0d7231a121c42.jpg)  
(a) Initialization

![](images/9c8f7eba39678ff2ae80bd34f8849e29ca63938e5574cdba8d0b1c4d99658aa8.jpg)  
(b) VDWM

![](images/d2d27a6ced06ad7490e6ff505dfb6109624b81deefa7659fec044b2a2e291c24.jpg)  
(c) Top-2 NN

![](images/c440e69f328e7cb649027ed8ae31221c2054df286b75ef8b0d0c1d99e75c204e.jpg)  
(d) OptiFlow (ours)  
Figure 5: Initialization-dependent matching in an equal-value bimodal bandit. The reference set contains two equal-value optimal anchors, while all generated particles are initialized in the left half of the support. Value-distance weighted matching (VDWM) selects an anchor independently for each particle and therefore assigns all particles to the closer left mode. Top-2 nearest-neighbor matching still assigns each particle independently to its nearest anchor, again collapsing to the left mode. In contrast, OptiFlow preserves multimodality since the marginal constraint assigns equal mass to both modes, preserving the bimodal allocation.

$a _ { \theta , i } ^ { s } = \mu _ { \theta } ( s , z _ { i } )$ . We use the same noise bank for both policies, i.e., $z _ { i } = z _ { j }$ when $i = j$ , with $z _ { i } \sim \mathcal { N } ( 0 , I )$ . VWD assigns a value-based weight to each reference action,

$$
w _ { j } = \frac { f ( Q ( s , \tilde { a } _ { \omega , j } ^ { s } ) ) } { \sum _ { \ell = 1 } ^ { N } f ( Q ( s , \tilde { a } _ { \omega , \ell } ^ { s } ) ) } .
$$

It then minimizes

$$
\mathcal { L } _ { \mathrm { V W D } } = \sum _ { j = 1 } ^ { N } w _ { j } \left\| a _ { j } ^ { \theta } - \tilde { a } _ { \omega , j } ^ { s } \right\| _ { 2 } ^ { 2 } .
$$

In practice, we consider $w _ { j }$ as:

$$
w _ { j } = \frac { \exp ( Q ( s , \tilde { a } _ { \omega , j } ^ { s } ) / \tau ) } { \sum _ { \ell = 1 } ^ { N } \exp ( Q ( s , \tilde { a } _ { \omega , \ell } ^ { s } ) / \tau ) } .
$$

Top-K NN first selects the K highest-value reference samples,

$$
\begin{array} { r } { \mathcal { T } _ { K } ( s ) = \mathrm { T o p K } _ { j \in \{ 1 , . . . , M \} } Q ( s , \tilde { a } _ { \omega , j } ^ { s } ) , } \end{array}
$$

and then assigns each one-step policy’s action to its nearest reference action within this set,

$$
j _ { i } ^ { \ast } = \arg \operatorname* { m i n } _ { j \in \mathcal { T } _ { K } ( s ) } \left\| a _ { \theta , i } ^ { s } - \tilde { a } _ { \omega , j } ^ { s } \right\| _ { 2 } ^ { 2 } .
$$

The corresponding distillation loss is

$$
\mathcal { L } _ { \mathrm { T o p K } } = \sum _ { i = 1 } ^ { N } \left. a _ { \theta , i } ^ { s } - \tilde { a } _ { \omega , j _ { i } ^ { * } } ^ { s } \right. _ { 2 } ^ { 2 } .
$$

In the bandit example in Figure 2, VWD collapses to the middle point, a well-known failure mode when distilling multi-step flow models into a one-step policy with naive $\ell _ { 2 }$ regression [20]. Top-K NN is also sensitive to the choice of K. When $K = 1$ , the policy collapses to a single selected mode. When $K = 3 .$ , it can preserve some multimodality, but it still cannot suppress probability mass in the intermediate region between modes. Although Top-K NN shows potential for capturing multimodality, it relies on hard selection and does not specify how much mass each selected reference action should receive, making it overly restrictive and sensitive to the choice of K.

Now let’s consider Value-Distance Weighted Matching(VDWM) distillation, more general formulation which is closer to our algorithm. Define a score

$$
\xi _ { i j } = f ( d _ { i j } , Q ( s , a _ { \theta , i } ^ { s } ) , Q ( s , \tilde { a } _ { \omega , j } ^ { s } ) ) ,
$$

and perform weighted distillation:

$$
\mathcal { L } _ { \mathrm { V D W M } } = \sum _ { i } \xi _ { i j _ { i } ^ { * } } \| a _ { \theta , i } ^ { s } - \tilde { a } _ { \omega , j _ { i } ^ { * } } ^ { s } \| ^ { 2 } ,
$$

where $j _ { i } ^ { * } = \arg \operatorname* { m a x } _ { j } s _ { i j }$ . For instance, one may choose

$$
\xi _ { i j } = \frac { \exp \big ( - d _ { i j } + \alpha ( Q ( s , \tilde { a } _ { \omega , j } ^ { s } ) - Q ( s , a _ { \theta , i } ^ { s } ) ) \big ) } { \sum _ { k = 1 } ^ { M } \exp \big ( - d _ { i k } + \alpha ( Q ( s , \tilde { a } _ { \omega , k } ^ { s } ) - Q ( s , a _ { \theta , i } ^ { s } ) ) \big ) }
$$

However, this approach performs independent selection for each i. As a result, it does not impose any global mass-allocation constraint across reference actions, and therefore has probability of collapse onto a subset of modes. In particular, when multiple target modes have comparable value, small differences in initialization, distance, or estimated value can break the symmetry arbitrarily, causing many samples to select the same mode.

In contrast, our method constructs a global coupling with value-conditioned marginals. This induces a structured allocation of mass across reference actions, so multiple high-value modes can be represented proportionally rather than selected independently. As a result, our method mitigates mode collapse and better preserves multimodality. We illustrate this behavior with the bandit example in Figure 5.

## A.4 Computational cost

OptiFlow introduces additional computational cost from three sources: (i) per-state Sinkhorn updates for optimal transport, (ii) action generation from the K-step reference flow model, and (iii) action generation from the one-step policy. We analyze how these factors contribute to the computation cost.

First, we identify the training-time bottleneck by varying the number of Sinkhorn iterations and analyzing the computational overhead introduced by OT updates.

Second, we analyze how the computational cost scales with the number of one-step flow policy and reference K-step flow policy samples (N, M), which affect reference action generation and transport matching. The corresponding experiments are reported in Appendix F.3.

Finally, we compare OptiFlow against alternative baselines in terms of both training and inference runtime.

Effect of Sinkhorn iterations on runtime. We use 30 Sinkhorn iterations as the default setting. To assess whether this choice introduces unnecessary computational overhead, we vary the number of iterations and measure the average computation time per training step. As shown in Table 3, reducing the number of iterations from 30 to 20 or 10 yields only marginal runtime improvements, suggesting that Sinkhorn updates are not a dominant training-time bottleneck. We analyze the corresponding performance behavior in Table 12b.

<table><tr><td>Sinkhorn Iterations</td><td>10</td><td>20</td><td>30 (default)</td><td>40</td></tr><tr><td>ms/step</td><td>5.14±0.04</td><td>5.26±0.04</td><td>5.37±0.04</td><td>5.47±0.04</td></tr></table>

Table 3: Effect of the number of Sinkhorn iterations on runtime. We report the average computation time per training step (ms/step) over 15 tasks measured on a single NVIDIA H100 GPU.

Computational scaling with the number of samples (N, M). We analyze how the computational cost scales with the number of one-step flow policy samples and reference K-step flow policy samples (N, M), which jointly affect both reference action generation and transport matching. As shown in Table 4, the computational cost is more sensitive to the number of reference-policy samples M than to the number of one-step-policy samples N. In particular, increasing M introduces substantially higher runtime overhead due to the increased cost of reference action generation and transport matching, whereas increasing N results in comparatively moderate overhead. We analyze the corresponding performance behavior in Appendix F.3.

<table><tr><td> $N \backslash M$ </td><td>4</td><td>16</td><td> ${ \bf 6 4 } \left( \mathrm { d e f a u l t } \right)$ </td><td>256</td><td> $\mathbf { A v g } ( M )$ </td></tr><tr><td>4</td><td> $3 . 7 3 { \pm } 0 . 1 3$ </td><td> $3 . 5 3 { \pm } 0 . 0 1$ </td><td> $4 . 9 9 2 0 . 0 4$ </td><td> $1 6 . 7 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $7 . 2 5 { \pm } 0 . 9 3 $ </td></tr><tr><td>8</td><td> $3 . 3 9 { \pm } 0 . 0 7$ </td><td> $3 . 6 5 { \pm } 0 . 0 1$ </td><td> $5 . 1 0 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 6 . 5 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $7 . 1 8 { \pm } 0 . 9 2$ </td></tr><tr><td>16 (default)</td><td> $3 . 4 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 . 7 4 { \pm } 0 . 0 2$ </td><td> $5 . 3 6 { \pm } 0 . 0 4$ </td><td> $1 6 . 8 3 { \pm } 0 . 0 9$ </td><td> $7 . 3 4 \pm 0 . 9 4$ </td></tr><tr><td>32</td><td> $3 . 5 0 { \pm } 0 . 0 2$ </td><td> $3 . 8 5 { \pm } 0 . 0 1$ </td><td> $5 . 7 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 7 . 4 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $7 . 6 3 { \pm } 0 . 9 7$ </td></tr><tr><td>Avg(N)</td><td> $3 . 5 1 { \pm } 0 . 0 4$ </td><td> $3 . 6 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $5 . 3 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 6 . 8 9 { \scriptstyle \pm 0 . 0 7 }$ </td><td></td></tr></table>

Table 4: Effect of the number of samples (N, M) on runtime. We report the average computation time per training step (ms/step) over 15 tasks measured on a single NVIDIA H100 GPU.

Training and inference runtime comparison. Finally, we compare the training and inference runtime of OptiFlow with alternative baselines. As shown in Figure 6, OptiFlow introduces additional training-time overhead compared to standard one-step methods due to the need to generate and match multiple samples (N, M) from the one-step and reference flow policies. In contrast, inference remains efficient since the learned policy requires only a single forward pass at test time. Combined with our Sinkhorn iteration analysis, these results suggest that the dominant computational cost arises from sample generation and matching rather than from the OT updates themselves. Reducing the sampling cost required for effective transport-based distillation is an important direction for future work.

![](images/2139a2f498e67287cdfa169f292a43b2fffc7390ab466ad8ad2d1a0ca2877411.jpg)  
Figure 6: Comparison of training and inference time across methods. We report the average training time per gradient step (ms/step), averaged over 15 tasks and measured on a single NVIDIA H100 GPU.

## B Broader Impacts

This work studies offline reinforcement learning in simulated control benchmarks. The main potential benefit is improved policy learning from fixed datasets, which may reduce the need for unsafe or expensive online exploration when applying RL to physical systems. OptiFlow also produces a one-step policy at deployment time, which can make generative-policy methods more practical in latency-sensitive control settings.

The main risk is the usual risk of offline RL: a learned policy may behave poorly when deployed outside the coverage of the dataset, especially if the critic assigns inaccurate values to unsupported actions. While OptiFlow is designed to reduce direct critic exploitation by using value information through transport-based target construction, it does not remove the need for careful validation before deployment. In safety-critical applications, the method should be combined with dataset auditing, environment-specific safety constraints, and out-of-distribution monitoring.

## C Sinkhorn Algorithm

## C.1 Algorithm

Algorithm 2 Sinkhorn Algorithm   
1: Initialize $K = \exp ( - C / \varepsilon )$   
2: Initialize $u = \mathbf { 1 } _ { N } , v = \mathbf { 1 } _ { M }$   
3: for $t = 1 , \dots , T$ do   
4: $u _ { i } \gets p _ { i } / ( K \underline { { v } } ) _ { i }$   
5: $v _ { j }  q _ { j } / ( K ^ { \top } u ) _ { j }$   
6: end for   
7: $P = \mathrm { d i a g } ( u ) K \mathrm { d i a g } ( v )$

For completeness, we provide the Sinkhorn procedure [21] used to solve the entropy-regularized optimal transport problem in Step 3 of our method. Given the cost matrix $C ,$ reference-side marginal p, target-side marginal $q ,$ and entropic regularization parameter ε, the algorithm alternates row and column scaling updates to compute the transport plan.

In all experiments, we use a fixed number of $T = 3 0$ Sinkhorn iterations.

## D Implementation Details

Architectures. The critic, reference flow policy, and one-step policy all use [512, 512, 512, 512] MLPs with GeLU [35] activations. LayerNorm [36] is applied to the critic only. The critic is a 2-ensemble; we aggregate with the mean by default, unless otherwise stated as minimum. The one-step policy takes concat(s, z) through the MLP and a linear head (init scale $1 0 ^ { - 2 } )$ ; policy outputs are clipped to [−1, 1] at use time. All methods are implemented in JAX [37].

Image processing. For pixel-based OGBench environments, we use a smaller variant of the IMPALA encoder [38] and apply random-shift augmentation with probability 0.5, following the official implementation of [16]. We use 3-frame stacking, prestacked at dataset load time.

Flow teacher. We use linear-interpolant flow matching with uniform time sampling and integrate with K = 10 Euler steps. Time t is concatenated directly into the MLP input.

Value-aware BC. The reference action in the VaBC weight $g _ { \eta }$ is the stop-graded one-step policy action, and the minibatch scale $\begin{array} { r } { \lambda = ( B ^ { - 1 } \sum _ { i } | Q _ { \phi } ( s _ { i } , a _ { i } ) | ) ^ { - 1 } } \end{array}$ is also stop-graded. In practice, $g _ { \eta }$ is implemented as a 2-class softmax between $Q _ { \phi } ( s , a )$ and $Q _ { \phi } ( s , \mu _ { \theta } ( s , z ) ) ,$ ). Gradients from this loss flow only into the reference policy.

Transport and distillation. Per state, we sample N = 16 one-step actions and M = 64 reference actions. The cost is squared Euclidean distance normalized by its batchwise mean. The one-step marginal is uniform, and the reference marginal is a softmax with temperature τ applied to reference critic values. We run balanced Sinkhorn in the log domain for T = 30 iterations, with entropic regularization fixed at $\varepsilon = 0 . 0 5$ across all tasks. We use hard row-wise anchors $j _ { i } ^ { * } = \arg \operatorname* { m a x } _ { j } \bar { P } _ { i j } ^ { * }$ with distillation weights $P _ { i , j _ { i } ^ { * } } ^ { * }$ . The actor objective combines the VaBC and distillation losses with equal weights.

Training and evaluation. We train for 1M gradient steps on state-based OGBench tasks and 500K steps on D4RL. Every 100K steps we evaluate with 100 episodes and reported scores are taken at the final training step and averaged over 8 seeds unless otherwise stated. For OGBench visual tasks, we report the average success rates across the last three evaluation epochs (300K, 400K, 500K), each evaluated over 50 episodes, following the protocol of FQL [16] due to computational constraints.

Bandit multimodal diagnostic. For Figure 2, FQL-D uses α = 15 and FQL-Q uses α = 1, where larger α places greater relative weight on the distillation term. For reference, at $\alpha = 1 0 0 0$ , the resulting policy becomes nearly indistinguishable from BC. The reward landscape contains two high-value modes with maximum reward +20 and a suboptimal band across the middle with reward +10 (the faint red region).

Four-mode environment diagnostic We consider a two-dimensional point-mass environment adapted from the multi-goal environment of [31]. The state and action spaces are bounded by $[ - 7 , 7 ] ^ { 2 }$ and $[ - 1 , 1 ] ^ { 2 }$ , respectively, and the dynamics are given by $s ^ { \prime } = \mathrm { c l i p } ( s + \bar { a } )$ . Four symmetric goals are located at $\mathcal { G } = \{ ( 5 , 0 ) , ( - 5 , 0 ) , ( 0 , 5 ) , ( 0 , - 5 ) \}$ . The transition reward is

$$
r ( s , a , s ^ { \prime } ) = - 3 0 \| a \| _ { 2 } ^ { 2 } - \operatorname* { m i n } _ { g \in \mathcal { G } } \| s ^ { \prime } - g \| _ { 2 } ^ { 2 } + 1 0 \cdot \mathbb { I } \biggl [ \operatorname* { m i n } _ { g \in \mathcal { G } } \| s ^ { \prime } - g \| _ { 2 } < 1 \biggr ] ,
$$

and an episode terminates upon reaching any goal or after 30 steps. The reward contours in Figure 4 visualize the state-dependent term $- \operatorname* { m i n } _ { g \in { \mathcal { G } } } \| { \bar { s } } - g \| _ { 2 } ^ { 2 } \colon$ ; consequently, all four goals are equally optimal from the symmetric initial region.

We first collect 100,000 transitions using a uniform-random behavior policy. To introduce asymmetric coverage while retaining all four directional modes, we independently remove 80% of transition tuples $( s , a , s ^ { \prime } )$ for which either s or $s ^ { \prime }$ lies below the diagonal $y \ = \ x .$ This leaves 55,044 transitions and changes the initial action proportions from approximately uniform to $\mathrm { R } / \mathrm { L } / \mathrm { T } / \mathrm { B } = 1 6 . 2 / 3 5 . 4 / 3 4 . 7 / 1 3 . 7 \%$ . For FQL, we sweep $\alpha \in \{ 0 . 0 \bar { 1 } , 0 . 1 , 0 . 3 , 0 . 5 , 1 , 3 , 5 , 1 0 \}$ where larger α places greater relative weight on flow-policy distillation. The figure presents represen tative results for $\alpha \in \{ 0 . 1 , 1 , 1 0 \}$ . For OptiFlow-BC, we use a standard flow-BC reference policy and set the value-weighted marginal temperature to $\tau = 0 . 1 ;$ no VaBC temperature η is used.

OT assignment stability diagnostic Using the same four-mode environment, we examine whether OT assignments remain directionally consistent when the reference actions are resampled. With all networks frozen, we fix four one-step actions at a given state, with one action pointing toward each of the right, left, top, and bottom goals. We then independently resample $M = 6 4$ reference actions 100 times and recompute the value-weighted marginal, Sinkhorn transport plan, and hard assignments at every round. Each assigned reference action is labeled according to the goal direction toward which it moves. We perform this diagnostic at the symmetric origin, where all four directions are equally valuable, and at nearby states where the values are no longer tied.

Online fine-tuning. For offline-to-online RL, we extend offline OptiFlow training with 1M additional gradient steps starting from the 1M-step offline checkpoint, adding online transitions to a unified replay buffer that already contains the offline dataset. Online data are collected using the deterministic one-step policy with no exploration noise, and we continue to train all components of OptiFlow with the same objective as in offline training.

Hyperparameters. We provide complete lists of hyperparameters used across all tasks and methods in Tables 5 to 8. We use the hyperparameters setup shown in [16] for the baseline experiments and those presented in [17].

Modified TD target. Standard TD target takes the form $y = r + \gamma Q ( s ^ { \prime } , a ^ { \prime } )$ . [17] presents a variant of the TD target $\begin{array} { r } { y ^ { \mathrm { v a B C } } ( s , r , s ^ { \prime } ) \stackrel { } { = } r + \frac { \gamma } { 2 } \big ( Q _ { \bar { \phi } } ( s ^ { \prime } , \mu _ { \theta } ( s ^ { \prime } , z ) ) + Q _ { \bar { \phi } } ( s ^ { \prime } , \mu _ { \omega } ( s ^ { \prime } , z ) ) \big ) } \end{array}$ where $z \sim \mathcal { N } ( 0 , I _ { d } )$ . Here $\mu _ { \boldsymbol { \theta } } \big ( \boldsymbol { s } ^ { \prime } , \boldsymbol { z } \big )$ denotes the action from the one-step policy and $\mu _ { \omega } ( s ^ { \prime } , z )$ denotes the action from the reference policy. This TD target is formulated as the average of the two estimates of Q-value. This is a more conservative variant of the TD target, and we report results of OptiFlow and GFP on the specified tasks in Table 6

Table 5: Shared hyperparameters used across all tasks and methods unless otherwise noted in Table 6. We evaluated across all tasks and methods equally according to Table 5 and 6 where relevant.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>MLP hidden dims</td><td>[512, 512, 512, 512]</td></tr><tr><td>Activation</td><td>GeLU [35]</td></tr><tr><td>LayerNorm [36]</td><td>Critic only</td></tr><tr><td>Clipped double Q-learning</td><td>False (default), True (antmaze-{large, giant} -navigate, adroit)</td></tr><tr><td>Optimizer</td><td>Adam [39]</td></tr><tr><td>Learning rate</td><td>3 × 10−4</td></tr><tr><td>Discount γ</td><td>0.99 (default), 0.995 (antmaze-giant, humanoidmaze, antsoccer)</td></tr><tr><td>TD target</td><td>Standard</td></tr><tr><td>Target network smoothing coefficient</td><td>0.005</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Gradient steps</td><td>1M (OGBench), 500k (OGBench visual tasks, D4RL)</td></tr><tr><td>Critic update</td><td>Every step (default), every 5 steps (antsoccer for OptiFlow only)</td></tr><tr><td>Euler integration steps K</td><td>10</td></tr><tr><td>Student samples N</td><td>16</td></tr><tr><td>Teacher samples M</td><td>64</td></tr><tr><td>Sinkhorn iterations T</td><td>30</td></tr><tr><td>Entropic regularization ε</td><td>0.05</td></tr></table>

Table 6: Per-task overrides to the shared defaults in Table 5 for OptiFlow and GFP only. We bring these overrides exactly from GFP setup. All other baselines and tasks not listed use the shared defaults. We make these overrides in order to maintain consistency with all baselines following the convention, and to provide accurate comparison against GFP which makes the following exact overrides due to apparently significant performance improvements. OptiFlow also demonstrates slightly improved performance with this setting. We report OptiFlow and GFP results on the following tasks with these overrides and all other tasks not listed follow the shared defaults in Table 5.

<table><tr><td>Task</td><td>Discount γ TD target</td><td></td></tr><tr><td>antmaze-large-navigate-singletask-task{1-5}-v0</td><td>0.995</td><td>一</td></tr><tr><td>humanoidmaze-large-navigate-singletask-task{1-5}-v0</td><td>0.999</td><td></td></tr><tr><td>humanoidmaze-medium-navigate-singletask-task{1-5}-v0</td><td></td><td>VaBC y</td></tr><tr><td>cube-single-play-singletask-task{1-5}-v0</td><td></td><td>VaBC y</td></tr><tr><td>cube-double-play-singletask-task{1-5}-v0</td><td></td><td>y VaBC</td></tr><tr><td>cube-triple-play-singletask-task{1-5}-v0</td><td></td><td>VaBC y</td></tr></table>

Table 7: Task-specific hyperparameters for offline RL on OGBench. We use the hyperparameters used by For OptiFlow, τ is the teacher-marginal softmax temperature and η is the value-aware BC temperature. "-" indicates that the experimental result is taken from prior work (or does not exist).
<table><tr><td>Task</td><td>BC</td><td>IQL α</td><td>ReBRAC  $\left( \alpha _ { 1 } , \alpha _ { 2 } \right)$ </td><td>IDQL N</td><td>IFQL N</td><td>FQL α</td><td>GFP (α, η)</td><td>OptiFlow (τ, η)</td></tr><tr><td>antmaze-large-navigate-singletask{1-5}-v0</td><td></td><td>10</td><td>(1e-3, 1e-2)</td><td>32</td><td>32</td><td>10</td><td>(0.3,1e-4)</td><td>(2.0, 1e-2)</td></tr><tr><td>antmaze-giant-navigate-singletask{1-5}-v0</td><td></td><td>10</td><td>(1e-3, 1e-2)</td><td>32</td><td>32</td><td>10</td><td>(0.1,1e-1)</td><td>(1.0, 1e-3)</td></tr><tr><td>humanoidmaze-medium-navigate-singletask{1-5}-v0</td><td></td><td>10</td><td>(1e-2,1e-2)</td><td>32</td><td>32</td><td>30</td><td>(0.3,1e-3)</td><td>(1.0, 1e-3)</td></tr><tr><td>humanoidmaze-large-navigate-singletask{1-5}-v0</td><td></td><td>10</td><td>(1e-2,1e-2)</td><td>32</td><td>32</td><td>30</td><td>(0.3,1e-4)</td><td>(2.0, 1e-1)</td></tr><tr><td>antsoccer-arena-navigate-singletask{1-5}-v0</td><td></td><td>1</td><td>(1e-2,1e-2)</td><td>32</td><td>64</td><td>10</td><td>(0.1,1e-2)</td><td>(1.0, 1e-2)</td></tr><tr><td>cube-single-play-singletask{1-5}-v0</td><td></td><td>1</td><td>(1,0)</td><td>32</td><td>32</td><td>300</td><td>(10,1e-1)</td><td>(2.0, 1e-1)</td></tr><tr><td>cube-double-play-singletask{1-5}-v0</td><td></td><td>0.3</td><td>(1e-1,0)</td><td>32</td><td>32</td><td>300</td><td>(1.0,1e-2)</td><td>(2.0, 1e-2)</td></tr><tr><td>cube-triple-play-singletask{1-5}-v0</td><td></td><td>1</td><td>(1e-1,0)</td><td>64</td><td>128</td><td>300</td><td>(0.1,1e-5)</td><td>(1.0, 1e-5)</td></tr><tr><td>scene-play-singletask{1-5}-v0</td><td></td><td>10</td><td>(1e-1,1e-2)</td><td>32</td><td>32</td><td>300</td><td>(10,1e-3)</td><td>(1.0, 1e-2)</td></tr><tr><td>puzzle-3x3-play-singletask{1-5}-v0</td><td></td><td>10</td><td>(3e-1,1e-2)</td><td>32</td><td>32</td><td>1000</td><td>(3.0,1e-3)</td><td>(1.0, 1e-3)</td></tr><tr><td>puzzle-4x4-play-singletask{1-5}-v0</td><td></td><td>3</td><td>(3e-1,1e-2)</td><td>32</td><td>32</td><td>1000</td><td>(3.0,1e-5)</td><td>(2.0,1e-2)</td></tr><tr><td>visual-cube-single-play-singletask1-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>(10,1e-1)</td><td>(1.0,1e-1)</td></tr><tr><td>visual-cube-double-play-singletask1-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>(0.3,1e-2)</td><td>(2.0,1e-2)</td></tr><tr><td>visual-scene-play-singletask1-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>(10,1e-3)</td><td>(1.0,1e-1)</td></tr><tr><td>visual-puzzle-3x3-play-singletask1-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>(3.0, 1e-2)</td><td>(1.0,1e-2)</td></tr><tr><td>visual-puzzle-4x4-play-singletask1-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>(1.0, 1e-4)</td><td>(1.0,1e-1)</td></tr></table>

Table 8: Task-specific hyperparameters for offline RL on D4RL. For OptiFlow, τ is the teachermarginal softmax temperature and η is the value-aware BC temperature. "-" indicates that the experimental result is taken from prior work (or does not exist).
<table><tr><td>Task</td><td>BC</td><td>IQL α</td><td>ReBRAC  $\left( \alpha _ { 1 } , \alpha _ { 2 } \right)$ </td><td>IDQL N</td><td>IFQL N</td><td>FQL α</td><td>GFP (α, η)</td><td>OptiFlow  $( \tau , \eta )$ </td></tr><tr><td>antmaze-umaze</td><td></td><td></td><td></td><td></td><td>32</td><td>10</td><td>(0.1,1e-3)</td><td>(1.0, 1e-3)</td></tr><tr><td>antmaze-umaze-diverse</td><td>1</td><td></td><td></td><td></td><td>32</td><td>10</td><td>(0.1,1e-3)</td><td>(1.5, 1e-6)</td></tr><tr><td>antmaze-medium-play</td><td>1</td><td></td><td></td><td></td><td>32</td><td>10</td><td>(0.03,1e-3)</td><td>(0.8, 1e-3)</td></tr><tr><td>antmaze-medium-diverse</td><td>一</td><td>一</td><td>一</td><td>一</td><td>32</td><td>10</td><td>(0.03,1e-3)</td><td>(2.0, 1e-4)</td></tr><tr><td>antmaze-large-play</td><td>一</td><td>一</td><td>一</td><td>一</td><td>32</td><td>3</td><td>(0.03,1e-5)</td><td>(0.5, 1e-5)</td></tr><tr><td>antmaze-large-diverse</td><td>一</td><td>一</td><td></td><td>一</td><td>32</td><td>3</td><td>(0.03,1e-5)</td><td>(0.8, 1e-4)</td></tr><tr><td>pen-human-v1</td><td></td><td></td><td></td><td>32</td><td>32</td><td>10000</td><td>(3.0,1e-4)</td><td>(5.0, 1e-3)</td></tr><tr><td>pen-cloned-v1</td><td></td><td></td><td></td><td>32</td><td>32</td><td>10000</td><td>(3.0,1e-5)</td><td>(5.0, 1e-3)</td></tr><tr><td>pen-expert-v1</td><td></td><td></td><td></td><td>32</td><td>32</td><td>3000</td><td>(1.0,1e-3)</td><td>(3.0, 1e-3)</td></tr><tr><td>door-human-v1</td><td></td><td></td><td></td><td>32</td><td>32</td><td>30000</td><td>(10,1e-2)</td><td>(3.0, 1e-5)</td></tr><tr><td>door-cloned-v1</td><td></td><td></td><td></td><td>32</td><td>128</td><td>30000</td><td>(10,1e-2)</td><td>(1.0, 1e-6)</td></tr><tr><td>door-expert-v1</td><td></td><td></td><td></td><td>32</td><td>32</td><td>30000</td><td>(10,1e-2)</td><td>(10, 1e-3)</td></tr><tr><td>hammer-human-v1</td><td></td><td></td><td></td><td>128</td><td>32</td><td>30000</td><td>(10,1e-5)</td><td>(5.0, 1e-5)</td></tr><tr><td>hammer-cloned-v1</td><td></td><td></td><td></td><td>32</td><td>32</td><td>10000</td><td>(10,1e-5)</td><td>(50, 1e-3)</td></tr><tr><td>hammer-expert-v1</td><td></td><td></td><td></td><td>32</td><td>32</td><td>30000</td><td>(10,1e-2)</td><td>(5.0, 1e-2)</td></tr><tr><td>relocate-human-v1</td><td></td><td></td><td></td><td>32</td><td>128</td><td>10000</td><td>(10,1e-4)</td><td>(10, 1e-3)</td></tr><tr><td>relocate-cloned-v1</td><td></td><td></td><td></td><td>64</td><td>32</td><td>30000</td><td>(10,1e-4)</td><td>(5.0, 1e-4)</td></tr><tr><td>relocate-expert-v1</td><td>一</td><td>一</td><td></td><td>32</td><td>32</td><td>30000</td><td>(10,1e-4)</td><td>(5.0, 1e-4)</td></tr></table>

## E Complete Experimental Results

This section presents experimental results spanning diverse OGBench (Table 9, 10) and D4RL (Table 11) benchmarks tasks. Table 5 and Table 6 indicate the hyperparameters used across all tasks and methods. All results are averaged over 8 random seeds except for the visual tasks which are averaged over 4 random seeds due to high computational cost.

Table 9: OGBench results part 1/2. All methods are trained for 1M gradient steps (500K for visual tasks) and evaluated at the final step over 100 episodes. We report success rate as mean ± standard error across 8 random seeds for state-based tasks and 4 random seeds for visual tasks. Bold indicates results within 95% of the best score on each task. Italics indicate results taken from prior work [16].
<table><tr><td>Task</td><td>BC</td><td>IQL</td><td>ReBRAC</td><td>IDQL</td><td>IFQL</td><td>FQL</td><td>GFP</td><td>OptiFlow</td></tr><tr><td colspan="9">antmaze-large-navigate-singletask</td></tr><tr><td colspan="9"></td></tr><tr><td>task1-v0</td><td>0.0±0.0</td><td>51.4±4.3 40.3±5.3</td><td>93.3±1.5 87.7±1.1</td><td>58.1±3.1 0.0±0.0</td><td>31.9±6.0 8±1.9</td><td>79.5±2.1 61.4±3.6</td><td>94.5±1.4 91.1±1.4</td><td>94.0±0.9 87.6±1.1</td></tr><tr><td>task2-v0</td><td>0.0±0.0</td><td>71.9±3.1</td><td>75±6.4</td><td>27.5±4.9</td><td>34.5±8.4</td><td>94.6±1.2</td><td>97.2±0.7</td><td>93.8±1.0</td></tr><tr><td>task3-v0</td><td>0.9±0.4</td><td>48.5±3.9</td><td>87.4±1.6</td><td>0.0±0.0</td><td>18.9±5.2</td><td>81.4±1.9</td><td>91.3±0.8</td><td>88.8±1.1</td></tr><tr><td>task4-v0 task5-v0</td><td>6.3±0.7 1.3±0.5</td><td>53.1±7.9</td><td>86.5±2.4</td><td>48.5±3.8</td><td>46.6±4.6</td><td>81.3±1.2</td><td>95.6±0.9</td><td>95.1±1.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">antmaze-giant-navigate-singletask</td></tr><tr><td>task1-v0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>39.4±9.9</td><td>0.0±0.0</td><td>0.0±0.0</td><td>10.5±2.9</td><td>8.6±2.5</td><td>5.4±2.4</td></tr><tr><td>task2-v0</td><td>0.0±0.0</td><td>0.1±0.1</td><td>16.6±9.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>16.4±4.9</td><td>36.6±4.6</td><td>25.9±3.4 0.4±0.2</td></tr><tr><td>task3-v0</td><td>0.0±0.0</td><td>0.1±0.1</td><td>53.5±8.8 0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0 0.0±0.0</td><td>3.3±0.7</td><td>43.0±4.3</td></tr><tr><td>task4-v0 task5-v0</td><td>0.0±0.0</td><td>0.0±0.0 21.4±4.1</td><td>52.6±4.8</td><td>0.0±0.0 0.0±0.0</td><td>0.0±0.0 4.5±2.2</td><td>20.6±9.8</td><td>28.1±7.8 17.9±8.3</td><td>57.9±4.6</td></tr><tr><td></td><td>0.1±0.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">humanoidmaze-medium-navigate-singletask</td></tr><tr><td>task1-v0</td><td>0.1±0.1</td><td>32±3.9</td><td>13.9±6.5</td><td>2.8±0.5</td><td>61.9±8.4</td><td>19±4.2</td><td>84.8±2.1</td><td>82.4±1.2</td></tr><tr><td>task2-v0</td><td>0.4±0.3</td><td>39.5±2.9</td><td>10.6±3.2</td><td>3.9±0.4</td><td>84.3±8.8</td><td>94±1.1</td><td>92.3±1.6</td><td>92.9±1.7</td></tr><tr><td>task3-v0 task4-v0</td><td>4.5±0.3</td><td>30.1±1.9</td><td>47.6±6.2 10.5±4.4</td><td>1.6±0.7 0.6±0.3</td><td>45.9±17.0</td><td>74±6.4 3±1.4</td><td>85.1±4.0</td><td>82.1±1.8</td></tr><tr><td>task5-v0</td><td>0.6±0.2</td><td>0.3±0.2</td><td>12.8±5.6</td><td>5.9±0.9</td><td>0.0±0.0 98.1±0.4</td><td>97±0.7</td><td>0.1±0.1 96.5±1.4</td><td>19.1±5.4</td></tr><tr><td></td><td>1.4±0.6</td><td>66.9±4.7</td><td></td><td></td><td></td><td></td><td></td><td>95.5±0.7</td></tr><tr><td colspan="9">humanoidmaze-large-navigate-singletask</td></tr><tr><td>task1-v0</td><td>0.0±0.0</td><td>2.3±0.6</td><td>0.8±0.4 0.0±0.0</td><td>0.1±0.1 0.0±0.0</td><td>12±2.3</td><td>1±0.5</td><td>36±5.8</td><td>48.3±2.2</td></tr><tr><td>task2-v0 task3-v0</td><td>0.1±0.1</td><td>0.0±0.0</td><td>9.9±1.6</td><td>1.9±0.5</td><td>0.0±0.0 51.1±3.7</td><td>0.0±0.0 15.4±3.0</td><td>0.6±0.6 8.1±2.3</td><td>0.0±0.0 20.9±1.8</td></tr><tr><td>task4-v0</td><td>0.1±0.1</td><td>6.5±1.5</td><td>1.8±1.5</td><td>0.0±0.0</td><td>0.3±0.2</td><td>0.1±0.1</td><td>3.8±3.7</td><td>0.1±0.1</td></tr><tr><td>task5-v0</td><td>0.0±0.0</td><td>1±0.3</td><td>0.9±0.5</td><td>0.0±0.0</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.0±0.0</td><td>1.9±0.5</td><td></td><td></td><td>0.0±0.0</td><td>1.5±1.2</td><td>6.7±4.9</td><td>26.3±4.4</td></tr><tr><td colspan="9">antsoccer-arena-navigate-singletask</td></tr><tr><td>task1-v0</td><td>0.3±0.2</td><td>15.6±0.8</td><td>0.1±0.1</td><td>15.8±2.6</td><td>66.9±7.3</td><td>79.3±2.0</td><td>79.1±2.5</td><td>67.9±3.3</td></tr><tr><td>task2-v0</td><td>0.5±0.2</td><td>17.3±1.6</td><td>0.3±0.3</td><td>0.0±0.0</td><td>46.3±12.7</td><td>92.8±1.5</td><td>94.1±1.9</td><td>81.5±1.6</td></tr><tr><td>task3-v0</td><td>0.0±0.0</td><td>5.8±1.9</td><td>0.0±0.0</td><td>1.8±0.6</td><td>5.8±3.7</td><td>54.8±2.8</td><td>52.5±2.4</td><td>43.3±3.9</td></tr><tr><td>task4-v0</td><td>0.0±0.0</td><td>3.1±0.9</td><td>0.3±0.3</td><td>0.0±0.0</td><td>26.4±2.8</td><td>38.1±4.4</td><td>42.5±3.8</td><td>46.8±4.1</td></tr><tr><td>task5-v0</td><td>0.0±0.0</td><td>5.4±1.4</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.3±0.3</td><td>42.0±5.1</td><td>37.2±3.0</td><td>23.8±4.2</td></tr><tr><td colspan="9">cube-single-play-singletask</td></tr><tr><td>task1-v0</td><td>2.5±1.3</td><td>56.3±4.3</td><td>92.1±1.3</td><td>92±0.8</td><td>79.1±1.5</td><td>93.5±1.8</td><td>96±0.9</td><td>99.1±0.3</td></tr><tr><td>task2-v0</td><td>6.1±4.3</td><td>53.5±6.1</td><td>86.7±4.2</td><td>98.5±0.5</td><td>89.5±0.8</td><td>96.1±0.6</td><td>97.7±1.1</td><td>98.5±0.5</td></tr><tr><td>task3-v0</td><td>4.9±1.9</td><td>57.9±6.2</td><td>89.7±2.1</td><td>99.6±0.2</td><td>89.3±0.9</td><td>95.1±1.8</td><td>98.8±0.6</td><td>99.0±0.6</td></tr><tr><td>task4-v0 task5-v0</td><td>0.4±0.4</td><td>42±6.1</td><td>87.3±2.5 83.1±2.7</td><td>86.5±0.5</td><td>79±2.1 82.5±2.7</td><td>84.5±3.4 85.8±3.4</td><td>95.7±1.0 93.5±1.7</td><td>95.8±0.9</td></tr><tr><td></td><td>0.0±0.0</td><td>40.8±5.5</td><td></td><td>82.9±2.1</td><td></td><td></td><td></td><td>95.4±1.2</td></tr><tr><td colspan="9">cube-double-play-singletask</td></tr><tr><td>task1-v0</td><td>0.0±0.0</td><td>11.6±1.1</td><td>27.2±6.3</td><td>3.9±1.2</td><td>27.5±2.8</td><td>50.1±4.1</td><td>73.5±3.6</td><td>80.4±2.3</td></tr><tr><td>task2-v0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>6.4±1.2</td><td>4.4±1.9</td><td>9.8±1.1</td><td>33.8±5.1</td><td>48.3±3.7</td><td>52.4±2.9</td></tr><tr><td>task3-v0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>3.3±0.8</td><td>0.5±0.2</td><td>5.3±0.8</td><td>21.1±2.2</td><td>40.6±3.7</td><td>46.3±3.3 16.8±1.3</td></tr><tr><td>task4-v0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>1.6±0.7</td><td>0.0±0.0</td><td>0.8±0.3</td><td>4.6±0.9</td><td>5.2±1.2</td><td></td></tr><tr><td>task5-v0</td><td>0.0±0.0</td><td>0.4±0.3</td><td>1.9±0.6</td><td>5.4±2.5</td><td>8.9±1.0</td><td>16.6±3.1</td><td>48.7±4.3</td><td>34.6±3.9</td></tr><tr><td colspan="9">cube-triple-play-singletask</td></tr><tr><td>task1-v0</td><td>0.2±0.1</td><td>2.7±1.0</td><td>10.8±1.8</td><td>2.1±1.0</td><td>1.2±0.6</td><td>5.7±2.1</td><td>9.0±1.6</td><td>17.9±0.8</td></tr><tr><td>task2-v0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td></tr><tr><td>task3-v0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.3±0.1</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0</td></tr><tr><td>task4-v0</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.0±0.0 0.0±0.0</td><td>0.0±0.0 0.0±0.0</td><td>0.0±0.0 0.0±0.0</td><td>0.0±0.0 0.0±0.0</td><td>0.0±0.0 0.0±0.0</td><td>0.0±0.0 0.0±0.0</td></tr><tr><td>task5-v0</td><td>0.0±0.0</td><td>0.0±0.0</td></table>

Table 10: OGBench results 2/2. We follow the same evaluation protocol as Table 9. Dashes denote unreported entries.
<table><tr><td>Task</td><td>BC</td><td>IQL</td><td>ReBRAC</td><td>IDQL</td><td>IFQL</td><td>FQL</td><td>GFP</td><td>OptiFlow</td></tr><tr><td colspan="9">puzzle-3x3-play-singletask</td></tr><tr><td>task1-v0</td><td> $1 9 . 1 { \pm } 2 . 9$ </td><td> $1 2 . 1 { \pm } 2 . 4$ </td><td> ${ \bf 9 6 . 9 2 0 . 9 }$ </td><td> $6 9 . 9 { \scriptstyle \pm 3 . 7 }$ </td><td> $9 2 \pm 1 . 3$ </td><td> $9 0 . 4 { \pm } 1 . 4 $ </td><td> ${ \bf 9 6 . 1 \pm 1 . 2 }$ </td><td> ${ \bf 9 5 . 3 2 1 . 5 }$ </td></tr><tr><td> $\mathrm { t a s k } 2 \mathrm { - v } 0$ </td><td> $0 . 1 { \pm } 0 . 1$ </td><td> $3 . 3 { \pm } 0 . 6 $ </td><td> $1 . 8 { \pm } 0 . 9$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $2 . 4 \pm 1 . 1$ </td><td> ${ \bf 1 6 . 9 \pm 2 . 3 }$ </td><td> $0 . 2 { \pm } 0 . 1$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td></tr><tr><td>task3-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 4 \pm 0 . 4$ </td><td> $1 . 9 { \pm } 0 . 6 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $\pm 0 . 3$ </td><td> ${ \bf 1 } 2 . { \bf 0 } { \pm } 1 . 3$ </td><td> $0 . 7 { \pm } 0 . 2 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td></tr><tr><td>task4-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $2 . 5 { \pm } 0 . 5$ </td><td> $5 . 0 { \pm } 1 . 8 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 1 { \pm } 0 . 1$ </td><td> $1 1 . 1 { \pm } 1 . 8 $ </td><td> $5 . 3 { \pm } 1 . 2 $ </td><td> $2 5 . 3 { \pm } 6 . 8 $ </td></tr><tr><td>task5-v0</td><td> $0 . 3 { \pm } 0 . 3$ </td><td> $2 . 3 { \pm } 0 . 7$ </td><td> $6 . 1 \pm 1 . 3$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 9 { \pm } 0 . 3$ </td><td> $1 7 . 3 { \pm } 1 . 8$ </td><td> $\mathbf { 1 9 . 6 \pm 5 . 3 }$ </td><td> ${ \bf 2 0 . 1 \pm 8 . 9 }$ </td></tr><tr><td colspan="9">puzzle-4x4-play-singletask</td></tr><tr><td>task1-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $3 . 3 { \pm } 0 . 9$ </td><td> $2 3 . 8 { \pm } 1 . 1$ </td><td> $2 6 . 1 \pm 2 . 5$ </td><td> $4 3 . 9 2 2 . 5$ </td><td> $2 8 . 9 { \pm 2 . 7 }$ </td><td> ${ \pm } 3 . 3 { \pm } 3 . 1 $ </td><td> ${ \bf 5 1 . 8 \pm 3 . 6 }$ </td></tr><tr><td>task2-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 6 \pm 0 . 4$ </td><td> $1 0 . 0 { \pm } 0 . 6 $ </td><td> $3 . 5 { \pm } 1 . 4 $ </td><td> ${ \bf 1 6 . 9 \pm 3 . 5 }$ </td><td> $1 3 . 3 { \pm } 1 . 3 $ </td><td> $7 . 2 { \pm } 1 . 2$ </td><td> $1 1 . 6 { \pm } 3 . 7 $ </td></tr><tr><td>task3-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 3 { \pm } 0 . 3$ </td><td> $1 6 . 8 { \pm } 1 . 1 $ </td><td> $4 2 . 9 { \pm } 3 . 2 $ </td><td> $4 0 \pm 2 . 2$ </td><td> $1 6 . 3 { \pm } 1 . 6 $ </td><td> $4 6 . 0 { \pm } 3 . 9$ </td><td> ${ \bf 6 3 . 1 \pm 3 . 6 }$ </td></tr><tr><td> $\mathrm { t a s k 4 – v 0 }$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 0 { \pm } 0 . 5$ </td><td> $1 1 . 0 { \pm } 1 . 6 $ </td><td> $3 3 . 3 { \scriptstyle \pm 4 . 2 }$ </td><td> $2 2 . 9 { \pm } 2 . 5$ </td><td> $8 . 8 \pm 1 . 1$ </td><td> $1 6 . 2 { \pm } 1 . 7 $ </td><td> $2 2 . 8 { \pm } 2 . 9$ </td></tr><tr><td>task5-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 3 { \pm } 0 . 4 $ </td><td> $8 . 6 { \pm } 0 . 7 $ </td><td> $4 . 5 { \pm } 0 . 4 $ </td><td> $1 0 . 5 { \pm } 1 . 1$ </td><td> $7 . 8 \pm 1 . 2$ </td><td> $6 . 3 { \pm } 1 . 2$ </td><td> $\mathbf { 1 1 . 1 \pm 2 . 7 }$ </td></tr><tr><td colspan="9">scene-play-singletask</td></tr><tr><td>task1-v0</td><td> $1 . 8 \pm 0 . 7$ </td><td> $6 5 \pm 2 . 6 $ </td><td> ${ \bf 9 5 . 6 { \pm 1 . 4 } }$ </td><td> ${ \bf 9 9 . 8 \pm 0 . 2 }$ </td><td> ${ \bf 9 9 . 4 } { \bf \pm 0 . 4 }$ </td><td> $\mathbf { 1 0 0 } \pm \mathrm { 0 }$ </td><td> ${ \bf 9 9 . 6 { \pm 0 . 2 } }$ </td><td> $\mathbf { 1 0 0 } \pm \mathrm { 0 }$ </td></tr><tr><td> $\mathrm { t a s k } 2 \mathrm { - v } 0$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 0 . 9 { \pm } 1 . 9$ </td><td> $4 6 . 1 \pm 5 . 5$ </td><td> $0 . 3 { \pm } 0 . 2$ </td><td> $4 3 . 4 { \pm } 8 . 0 $ </td><td> $7 8 . 1 \pm 6 . 4$ </td><td> $9 1 . 0 { \pm } 2 . 9$ </td><td> ${ \bf 9 7 . 5 \pm 1 . 1 }$ </td></tr><tr><td>task3-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $2 3 \pm 2 . 1$ </td><td> $6 7 . 5 { \pm } 5 . 2 $ </td><td> ${ \bf 9 5 . 3 \pm 1 . 3 }$ </td><td> $8 0 . 8 { \pm } 2 . 4 $ </td><td> ${ \bf 9 5 . 1 \pm 0 . 8 }$ </td><td> $7 5 . 0 { \pm } 7 . 6 $ </td><td> ${ \bf 9 8 . 0 { \pm 0 . 6 } }$ </td></tr><tr><td>task4-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 1 { \pm } 0 . 5$ </td><td> $0 . 9 { \pm } 0 . 4$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 1 { \pm } 0 . 1$ </td><td> ${ \bf 6 . 0 \pm 2 . 3 }$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $1 . 5 { \pm } 0 . 8 $ </td></tr><tr><td>task5-v0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td></tr><tr><td colspan="9">visual-{-}-play-singletask-task1-v0</td></tr><tr><td>cube-single</td><td></td><td> $\gamma \boldsymbol { O } \pm \boldsymbol { \epsilon }$ </td><td> $g 3 \pm 3$ </td><td></td><td> $\mathbf { \nabla } _ { 4 } 9 \pm \mathbf { \nabla } 3 . 5$ </td><td> $\delta \boldsymbol { 1 } \pm \boldsymbol { \delta }$ </td><td> $7 2 . 2 { \pm } 3 . 6 $ </td><td> ${ \bf 8 6 . 6 \pm 1 . 3 }$ </td></tr><tr><td>cube-double</td><td></td><td> $\mathcal { B } \mathcal { 4 } \pm \ : \tau 1 \ : . 5$ </td><td> $\not = 2$ </td><td></td><td> $\delta \pm \it { 3 }$ </td><td> $\textit { 2 1 } \pm 5 . 5$ </td><td> $1 2 . 0 { \pm } 2 . 6 $ </td><td> ${ \bf 6 6 . 7 \pm 3 . 5 }$ </td></tr><tr><td>scene</td><td></td><td> ${ \pm } _ { { \bf \delta } }$ </td><td> ${ \pm } 2$ </td><td></td><td> $\boldsymbol { 8 6 \pm 5 }$ </td><td> ${ \bf 9 8 \pm 1 . 5 }$ </td><td> $\mathbf { 9 8 . 7 \pm 0 . 3 }$ </td><td> $9 2 . 8 { \pm } 0 . 8 $ </td></tr><tr><td> $\mathrm { p u z z l e } { - 3 \mathrm { x } 3 }$ </td><td></td><td> $\gamma _ { \pm 7 , 5 }$ </td><td> $\boldsymbol { 8 8 \pm 2 }$ </td><td></td><td> ${ \mathbf { } } I \pmb { 0 } \pmb { \mathrm { \pm } } \mathbf { \mathrm { \Omega } }$ </td><td> $g _ { \mathscr { 4 } } \pm o . 5$ </td><td> $8 6 . 5 { \pm } 1 . 1$ </td><td> $9 3 . 8 { \pm } 1 . 2 $ </td></tr><tr><td>puzzle-4x4</td><td></td><td> $\it { 0 . 0 \pm 0 . 0 }$ </td><td> ${ \boldsymbol { \mathcal { Q } } } { \boldsymbol { 6 } } \pm { \boldsymbol { 3 } }$ </td><td></td><td> $\delta \pm \tau . 5$ </td><td> ${ 3 3 \pm 3 }$ </td><td> $1 9 . 3 { \pm } 1 . 4 $ </td><td> ${ \bf 3 9 . 7 \pm 4 . 8 }$ </td></tr></table>

Table 11: D4RL results across AntMaze navigation and Adroit dexterous manipulation tasks. All methods are trained for 500K gradient steps and evaluated at the final step over 100 episodes. We report scores as mean ± standard error across 8 random seeds, following D4RL conventions: success rate for AntMaze and normalized return for Adroit. Bold indicates results within 95% of the best score on each task. Italics indicate results taken from prior work [14, 34].
<table><tr><td>Task</td><td>BC</td><td>IQL</td><td>ReBRAC</td><td>IDQL</td><td>IFQL</td><td>FQL</td><td>GFP</td><td>OptiFlow</td></tr><tr><td>antmaze-umaze</td><td>54.6</td><td>83.3</td><td>97.8</td><td>94.0</td><td> $8 6 . 5 { \pm } 2 . 1 $ </td><td> ${ \bf 9 5 . 0 \pm 1 . 8 }$ </td><td> ${ \bf 9 6 . 5 \pm 0 . 5 }$ </td><td> ${ \bf 9 6 . 0 { \pm } 0 . 9 }$ </td></tr><tr><td>antmaze-umaze-diverse</td><td>45.6</td><td>70.6</td><td>88.3</td><td>80.2</td><td> $5 8 . 5 { \pm } 5 . 3 $ </td><td> $8 3 . 3 { \pm } 4 . 8$ </td><td> ${ \bf 9 1 . 8 \pm 1 . 6 }$ </td><td> $8 4 . 9 2 1 . 6 $ </td></tr><tr><td>antmaze-medium-play</td><td>0.0</td><td>64.6</td><td>84.0</td><td>84.2</td><td> $5 4 . 8 { \pm } 6 . 5 $ </td><td> $7 6 . 5 { \pm 2 . 7 }$ </td><td> $8 3 . 2 { \pm } 3 . 2 $ </td><td> ${ \bf 8 8 . 0 \pm 1 . 4 }$ </td></tr><tr><td>antmaze-medium-diverse</td><td>0.0</td><td>61.7</td><td>76.3</td><td>84.8</td><td> $6 0 . 3 { \pm } 8 . 8 $ </td><td> $7 0 . 0 { \pm } 4 . 2 $ </td><td> $5 7 . 5 { \pm } 8 . 2 $ </td><td> ${ \bf 8 7 . 4 \pm 1 . 6 }$ </td></tr><tr><td>antmaze-large-play</td><td>0.0</td><td>42.5</td><td>60.4</td><td>63.5</td><td> $4 6 . 3 { \pm } 4 . 8 $ </td><td> ${ \bf 8 7 . 0 \pm 1 . 7 }$ </td><td> $8 2 . 1 { \pm } 1 . 8 $ </td><td> $8 2 . 6 { \pm } 2 . 4$ </td></tr><tr><td>antmaze-large-diverse</td><td>0.0</td><td>27.6</td><td>54.4</td><td>67.9</td><td> $5 9 . 5 { \pm } 7 . 4 $ </td><td> $\mathbf { 8 1 . 7 5 \pm 1 . 4 }$ </td><td> $\mathbf { 8 4 . 5 \pm 1 . 2 }$ </td><td> ${ \bf 8 0 . 6 \pm 1 . 8 }$ </td></tr><tr><td>pen-human-v1</td><td>34.4</td><td>81.5</td><td>103.5</td><td>56.1±2.3</td><td> $6 9 . 8 { \pm } 2 . 4 $ </td><td> $4 0 . 0 { \pm } 3 . 5 $ </td><td> $5 9 . 1 \pm 3 . 6 $ </td><td> $6 1 . 1 { \pm } 2 . 3 $ </td></tr><tr><td>pen-cloned-v1</td><td>56.9</td><td>77.2</td><td>91.8</td><td> $7 4 . 2 { \pm } 1 . 9$ </td><td> $7 6 . 9 { \pm } 2 . 8 $ </td><td> $7 4 . 9 { \pm } 4 . 5 $ </td><td> $7 8 . 7 \pm 2 . 7$ </td><td> $6 7 . 4 { \pm } 2 . 5$ </td></tr><tr><td>pen-expert-v1</td><td>85.1</td><td>133.6</td><td>154.1</td><td> $1 3 3 . 7 { \pm } 1 . 2 $ </td><td> $1 3 5 . 7 { \pm } 1 . 9$ </td><td> $1 4 4 . 8 { \pm } 1 . 2 $ </td><td> $1 4 1 . 2 { \pm } 2 . 0 $ </td><td> $1 2 7 . 6 { \pm } 2 . 2 $ </td></tr><tr><td>door-human-v1</td><td>0.5</td><td>3.1</td><td>0.0</td><td> $3 . 7 \pm 0 . 6$ </td><td> $2 . 9 { \pm } 0 . 6 $ </td><td> $0 . 1 { \pm } 0 . 1$ </td><td> $0 . 1 { \pm } 0 . 0$ </td><td> $2 . 0 { \pm } 0 . 4 $ </td></tr><tr><td>door-cloned-v1</td><td>-0.1</td><td>0.8</td><td>1.1</td><td> ${ \bf 6 . 1 \pm 1 . 3 }$ </td><td> $0 . 2 { \pm } 0 . 2$ </td><td> $1 . 7 { \pm } 0 . 5 $ </td><td> $0 . 3 { \pm } 0 . 2$ </td><td> $3 . 0 { \pm } 1 . 6 $ </td></tr><tr><td>door-expert-v1</td><td>34.9</td><td>105.3</td><td>104.6</td><td> $\mathbf { 1 0 3 . 9 2 0 . 3 }$ </td><td> $9 7 . 9 { \pm 2 . 6 }$ </td><td> ${ \bf 1 0 4 . 0 { \pm } 0 . 3 }$ </td><td> ${ \bf 1 0 4 . 2 { \scriptstyle \pm 0 . 2 } }$ </td><td> $\mathbf { 1 0 4 . 3 { \scriptstyle \pm 0 . 4 } }$ </td></tr><tr><td>hammer-human-v1</td><td>1.5</td><td>2.5</td><td>0.2</td><td> $1 . 4 \pm 0 . 3$ </td><td> $1 . 5 { \pm } 0 . 3 $ </td><td> $2 . 3 { \pm } 0 . 6 $ </td><td> $5 . 7 \pm 2 . 2$ </td><td> $5 . 1 { \pm } 1 . 0 $ </td></tr><tr><td>hammer-cloned-v1</td><td>0.8</td><td>1.1</td><td>6.7</td><td> $1 . 8 { \pm } 0 . 3$ </td><td> $0 . 2 { \pm } 0 . 2$ </td><td> $7 . 5 \pm 2 . 7$ </td><td> ${ \bf 7 . 4 \pm 1 . 5 }$ </td><td> $2 . 1 { \pm } 0 . 5$ </td></tr><tr><td>hammer-expert-v1</td><td>125.6</td><td>129.6</td><td>133.8</td><td> $1 1 3 . 7 { \scriptstyle \pm 3 . 4 }$ </td><td> $9 7 . 9 { \pm 2 . 6 }$ </td><td> $1 2 5 . 7 { \pm } 0 . 4 $ </td><td> $1 2 2 . 4 { \pm } 1 . 2$ </td><td> ${ \bf 1 2 8 . 8 { \scriptstyle \pm 0 . 2 } }$ </td></tr><tr><td>relocate-human-v1</td><td>0.0</td><td>0.1</td><td>0.0</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 2 { \pm } 0 . 1$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> ${ \bf 0 . 6 \pm } 0 . 2$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td></tr><tr><td>relocate-cloned-v1</td><td>-0.1</td><td>0.2</td><td>0.9</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $- 0 . 0 { \pm } 0 . 0 $ </td><td> $0 . 2 { \pm } 0 . 1$ </td><td> $1 . 5 { \pm } 0 . 3$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td></tr><tr><td>relocate-expert-v1</td><td>101.3</td><td>106.5</td><td>106.6</td><td> $\mathbf { 1 0 7 . 5 } 2 0 . 3$ </td><td> $9 5 . 6 { \pm } 2 . 8 $ </td><td> ${ \bf 1 0 6 . 8 \pm 0 . 8 }$ </td><td>104.2±2.1</td><td> $\mathbf { 1 0 4 . 4 \pm 1 . 3 }$ </td></tr></table>

## F Ablation Studies

We conduct a comprehensive set of ablation studies to validate the design choices and probe the behavior of OptiFlow. We first analyze sensitivity to the task-specific temperatures τ and η, as well as to the Sinkhorn regularization coefficient and iteration count. We then study the effect of the reference and one-step action sample counts (M, N) and show that the gains of OptiFlow cannot be explained by a larger action-sampling budget alone. Next, we disentangle the contributions of reference-policy quality and transport-based policy extraction, and compare our default value-aware behavior cloning objective with an AWR-style weighting alternative. Finally, we examine whether direct Q-maximization provides additional benefits beyond transport-based supervision and evaluate OptiFlow under offline-to-online fine-tuning.

## F.1 Sensitivity to value-weighting temperatures

OptiFlow introduces two task-specific hyperparameters: the reference-marginal softmax temperature τ , which controls how sharply the transport plan is biased toward high-value reference actions, and the value-aware BC temperature η, which controls the strength of value weighting in reference policy training.

![](images/db404320c8a0624cc84036fb03e95a6e6fb3ec773b0df12d2f896c94c6b24b6d.jpg)  
(a) OptiFlow antmaze-large

![](images/5f245b4290025f8a1781a2d4ac1c1d3b165cb2a108389b006e6385c548e448f1.jpg)  
(b) OptiFlow cube-double

![](images/4ebecda3be3ccb40192bd76fae0f271b9d16e112fa7e7588ba04afd3e270498b.jpg)  
(c) OptiFlow scene  
Figure 7: Sensitivity analysis to the reference-marginal temperature $\tau ,$ and to the value-aware BC temperature η. We evaluate the sensitivity by testing variations around task-specific hyperparameters $( \tau ^ { * } , \eta ^ { * } )$ from Table 7, using 8 random seeds. Figures (a), (b), and (c) report OptiFlow’s sensitivity to $\tau ^ { * }$ and $\eta ^ { * }$ on antmaze-large-navigate-task1, cube-double-play-task2, and scene-playtask2, respectively.

Result figures show that OptiFlow’s performance varies smoothly around the tuned settings $( \tau ^ { * } , \eta ^ { * } )$ with no single hyperparameter dominating the result. It is simple to find appropriate task-specific $( \tau ^ { * } , \eta ^ { * } )$ values via minor tuning with an understanding of how these parameters affect the learning curve depending on the difficulty and the nature of tasks.

## F.2 Sensitivity to Sinkhorn hyperparameters

We normalize each per-state cost matrix by its mean (Eq. 11), so the entropic regularization coefficient ϵ has a comparable scale across tasks. We use $\epsilon = 0 . 0 5$ and 30 Sinkhorn iterations by default, and sweep both choices on the three tasks from Figure 7.

(a) Regularization coefficient ϵ  
(b) Sinkhorn iterations
<table><tr><td>Task</td><td>.01</td><td>.05</td><td>.1</td><td>.5</td></tr><tr><td>Antmaze</td><td> $9 1 . 6 { \pm } 2 . 7 \ $ </td><td> ${ \bf 9 4 . 0 { \scriptstyle \pm 0 . 9 } }$ </td><td> $9 3 . 8 { \pm } 1 . 2 $ </td><td> $9 3 . 9 { \pm } 0 . 9$ </td></tr><tr><td>Cube</td><td> ${ \bf 6 2 . 1 \pm 3 . 7 }$ </td><td> $5 2 . 4 { \pm } 2 . 9$ </td><td> $5 7 . 0 { \pm } 5 . 0$ </td><td> $4 9 . 9 { \pm } 2 . 9$ </td></tr><tr><td>Scene</td><td> $9 3 . 7 { \pm 2 . 4 }$ </td><td> ${ \bf 9 7 . 5 \pm 1 . 1 }$ </td><td> $9 6 . 0 { \pm } 1 . 2 $ </td><td> $8 2 . 9 { \pm } 3 . 6 $ </td></tr></table>

<table><tr><td>Task</td><td>10</td><td>20</td><td>30</td><td>40</td></tr><tr><td>Antmaze</td><td> $9 1 . 8 { \pm } 1 . 9$ </td><td> $9 2 . 9 { \pm } 1 . 9$ </td><td> ${ \bf 9 4 . 0 { \scriptstyle \pm 0 . 9 } }$ </td><td> $9 3 . 3 { \pm } 1 . 9$ </td></tr><tr><td>Cube</td><td> $5 0 . 3 { \pm } 3 . 7 $ </td><td> $4 8 . 9 { \pm } 3 . 2 $ </td><td> ${ \pm } 2 . 4 { \pm } 2 . 9$ </td><td> $5 1 . 8 { \pm } 3 . 9$ </td></tr><tr><td>Scene</td><td> $9 6 . 4 { \pm } 1 . 5 $ </td><td> $9 4 . 8 { \pm } 2 . 2 $ </td><td> ${ \bf 9 7 . 5 \pm 1 . 1 }$ </td><td> $9 5 . 9 { \pm } 1 . 4 $ </td></tr></table>

Table 12: Sensitivity to Sinkhorn hyperparameters. Antmaze, Cube, and Scene denote antmazelarge-navigate-task1, cube-double-play-task2, and scene-play-task2. Evaluation protocol follows Table 9.

The sweeps show a mild task-dependent trade-off. The fixed $\epsilon = 0 . 0 5$ is a robust intermediate point on the normalized cost scale, while 30 iterations performs best on all three tasks and adds little overhead (Appendix A.4).

## F.3 Reference and one-step action sampling

The transport plan in OptiFlow is constructed over N one-step actions and M reference actions sampled per state, and the quality of the resulting supervision depends on whether these sample sets adequately cover the conditional action distribution. We sweep $( N , M )$ over $\left\{ 4 , 8 , 1 6 , 3 2 \right\} \times$ $\{ 4 , 1 6 , 6 \bar { 4 } , 2 5 6 \}$ , holding all other hyperparameters fixed, to identify how many samples are needed for stable transport-based distillation and show that we only need an adequate number of samples.

Table 13: Antmaze-large-navigate-task1. We follow the same evaluation protocol as Table 9, varying one-step samples N and reference samples M.
<table><tr><td> $N \backslash M$ </td><td>4</td><td>16</td><td>64 (default)</td><td>256</td></tr><tr><td>4</td><td> $2 . 8 { \pm } 0 . 8$ </td><td> $8 0 \pm 2 . 3$ </td><td> $9 2 . 2 { \pm } 2 . 2 $ </td><td> ${ \bf 9 7 . 2 { \pm 0 . 3 } }$ </td></tr><tr><td>8</td><td> $7 . 8 { \pm } 2 . 8 $ </td><td> $3 8 . 7 { \pm } 4 . 0 $ </td><td> $9 0 . 7 { \scriptstyle \pm 0 . 7 }$ </td><td> ${ \bf 9 6 . 7 \pm 0 . 6 }$ </td></tr><tr><td>16 (default)</td><td> $1 8 . 5 { \pm } 4 . 3 $ </td><td> $3 9 . 8 { \pm } 3 . 3 $ </td><td> ${ \bf 9 4 . 0 { \scriptstyle \pm 0 . 9 } }$ </td><td> $\mathbf { 9 4 . 2 \bot 0 . 6 }$ </td></tr><tr><td>32</td><td> $4 . 8 { \pm } 0 . 7$ </td><td> $2 9 . 8 { \pm } 3 . 1 $ </td><td> $8 1 . 8 { \pm } 3 . 0 $ </td><td> $8 3 . 0 { \pm } 2 . 2 $ </td></tr></table>

It can seen that performance depends mainly on M and is largely insensitive to N. The default $( N , M ) = ( 1 6 , 6 4 )$ matches or achieves the best result on each task: smaller M degrades performance sharply, while $M = 2 5 6$ gives consistent gain as can be expected. Sampling larger number of reference actions may be advantageous for distill target actions selection. Due to performance vs computation trade off, we use $( N , M ) = ( 1 6 , 6 4 )$ by default, concluding that OptiFlow only needs sufficient number of one-step and reference action samples, and excessive number of samples are not critical.

## F.4 Controlling for the action-sampling budget

OptiFlow samples multiple actions per state to construct the transport coupling, raising the possibility that its gains arise simply from a larger action-sampling budget rather than from transport-based supervision. To control for this factor, we modify FQL and $\bar { \mathrm { G F P } }$ to draw either $N = 1 6$ or $N = 6 4$ one-step actions per state and average their actor gradients over the sampled actions, while leaving the remaining training procedure unchanged.

Table 14: Controlling for the action-sampling budget. FQL and GFP with $N = 1 6$ or $N = 6 4$ draw multiple actions per state and average the actor gradient over the sampled actions. OptiFlow uses $N = \bar { 1 6 }$ one-step policy samples per state. We follow the same evaluation protocol as Table 9.
<table><tr><td></td><td>FQL</td><td>FQL</td><td>FQL</td><td>GFP</td><td>GFP</td><td>GFP</td><td></td></tr><tr><td>Task</td><td>(original)</td><td> $( N = 1 6 )$ </td><td> $( N = 6 4 )$ </td><td>(original)</td><td> $( N = 1 6 )$ </td><td> $( N = 6 4 )$ </td><td>OptiFlow</td></tr><tr><td>antmaze-large-navigate{1}</td><td> $7 9 . 5 { \pm 2 . 1 }$ </td><td> $7 9 . 5 { \pm } 1 . 8$ </td><td> $7 6 . 3 { \pm } 4 . 0$ </td><td> $\mathbf { 9 4 . 5 \pm 1 . 4 }$ </td><td> ${ \bf 9 5 . 2 \pm 0 . 6 }$ </td><td> ${ \bf 9 4 . 4 \pm 0 . 7 }$ </td><td> ${ \bf 9 4 . 0 { \pm } 0 . 9 }$ </td></tr><tr><td>antmaze-giant-navigate{1}</td><td> $1 0 . 5 { \pm } 2 . 9$ </td><td> $8 . 0 { \pm } 2 . 5$ </td><td> $1 2 . 3 { \pm } 3 . 5 $ </td><td> $8 . 6 \pm 2 . 5$ </td><td> $1 5 . 9 { \pm } 7 . 0 $ </td><td> $1 7 . 5 { \pm } 6 . 8 $ </td><td> $5 . 4 \pm 2 . 4$ </td></tr><tr><td>humanoidmaze-medium- navigate{1}</td><td> $1 9 . 0 { \pm } 4 . 2 $ </td><td> $2 5 . 0 { \pm } 1 0 . 6 $ </td><td> $1 0 . 3 { \pm } 4 . 1$ </td><td> ${ \bf 8 4 . 8 \pm 2 . 1 }$ </td><td> ${ \bf 8 3 . 0 \pm 1 . 8 }$ </td><td> ${ \bf 8 4 . 7 \pm 1 . 0 }$ </td><td> ${ \bf 8 2 . 4 \pm 1 . 2 }$ </td></tr><tr><td>humanoidmaze-large-navigate{ 1}</td><td> $1 . 0 { \pm } 0 . 5 $ </td><td> $2 . 0 { \pm } 0 . 7$ </td><td> $3 . 3 { \pm } 1 . 8 $ </td><td> $3 6 . 0 { \pm } 5 . 8 $ </td><td> $2 2 . 5 { \pm } 7 . 4$ </td><td> $1 7 . 5 { \pm } 7 . 4 $ </td><td> ${ \bf 4 8 . 3 \pm 2 . 2 }$ </td></tr><tr><td>antsoccer-arena-navigate{4}</td><td> $3 8 . 1 \pm 4 . 4$ </td><td> $3 4 . 8 { \pm } 1 . 7$ </td><td> $4 3 . 5 { \pm } 2 . 4$ </td><td> $4 2 . 5 { \pm } 3 . 8 $ </td><td> $4 2 . 5 { \pm } 1 . 8$ </td><td> $\pm 4 . 6 { \pm } 1 . 4$ </td><td> ${ \bf 4 6 . 8 \pm } 4 . 1$ </td></tr><tr><td>cube-single-play{2}</td><td> ${ \bf 9 6 . 1 \pm 0 . 6 }$ </td><td> $\mathbf { 9 7 . 8 \pm 1 . 0 }$ </td><td> ${ \bf 9 8 . 0 { \pm } 0 . 8 }$ </td><td> $\mathbf { 9 7 . 7 \pm 1 . 1 }$ </td><td> ${ \bf 9 9 . 3 { \pm 0 . 2 } }$ </td><td> ${ \bf 9 8 . 8 \pm 0 . 3 }$ </td><td> ${ \bf 9 8 . 5 \pm 0 . 5 }$ </td></tr><tr><td>cube-double-play{2}</td><td> $3 3 . 8 { \pm } 5 . 1$ </td><td> $3 6 . 0 { \pm } 5 . 2 $ </td><td> $3 6 . 3 { \pm } 4 . 0$ </td><td> $4 8 . 3 { \pm } 3 . 7 \ $ </td><td> $4 0 . 8 { \pm } 2 . 0 $ </td><td> $4 1 . 8 { \pm } 1 . 9$ </td><td> ${ \pm } 2 . 4 { \pm } 2 . 9$ </td></tr><tr><td>scene-play{2}</td><td> $7 8 . 1 \pm 6 . 4$ </td><td> $7 9 . 8 { \pm } 3 . 5 $ </td><td> $7 8 . 8 \pm 5 . 6$ </td><td> $9 1 . 0 { \pm } 2 . 9$ </td><td> $9 0 . 6 { \pm } 2 . 2 $ </td><td> $8 9 . 7 { \pm } 1 . 5 $ </td><td> ${ \bf 9 7 . 5 \pm 1 . 1 }$ </td></tr><tr><td> $\mathrm { p u z z l e } . 3 \mathrm { x } 3 \mathrm { - p l a y } \{ 4 \}$ </td><td> $1 1 . 1 { \pm } 1 . 8 $ </td><td> $1 2 . 5 { \pm } 1 . 2$ </td><td> $1 3 . 3 { \pm } 1 . 9$ </td><td> $5 . 3 { \pm } 1 . 2 $ </td><td> $6 . 2 { \pm } 0 . 8 $ </td><td> $7 . 9 { \pm } 1 . 6 $ </td><td> $2 5 . 3 2 6 . 8$ </td></tr><tr><td> $\mathrm { p u z z l e } { - } 4 \mathrm { x } 4 { - } \mathrm { p l a y } \{ 4 \}$ </td><td> $8 . 8 \pm 1 . 1$ </td><td> $9 . 8 \pm 1 . 2$ </td><td> $9 . 3 { \pm } 1 . 2 $ </td><td> $1 6 . 2 { \pm } 1 . 7 $ </td><td> $1 7 . 1 { \pm } 1 . 4$ </td><td> $1 5 . 4 \pm 1 . 2$ </td><td> $2 2 . 8 { \pm } 2 . 9$ </td></tr><tr><td>Average</td><td> $3 7 . 6 { \pm } 1 . 1$ </td><td> $3 8 . 5 { \pm } 1 . 5 $ </td><td> $3 8 . 1 \pm 1 . 3$ </td><td> $5 2 . 5 { \pm } 0 . 9$ </td><td> $5 1 . 3 { \pm } 0 . 8 $ </td><td> $5 1 . 2 { \pm } 1 . 2 $ </td><td> ${ \pm } 7 . 3 { \pm } 1 . 0$ </td></tr></table>

Increasing the number of sampled actions does not yield a consistent improvement for either baseline. FQL obtains average scores of 37.6, 38.5, and 38.1 with $N = 1$ , 16, and 64, respectively, while GFP obtains 52.5, 51.3, and 51.2. All variants remain below OptiFlow’s 57.3. Notably, the $N = 6 4$ baseline variants use more one-step action samples per state than OptiFlow $( N = 1 6 )$ , yet do not close the performance gap. Thus, the performance improvement cannot be explained by the number of one-step actions sampled during training alone.

## F.5 Disentangling reference policy quality and distillation

To disentangle whether OptiFlow’s performance gains arise from a stronger reference policy or from the distillation procedure itself, we separately evaluate the reference and one-step policies and introduce OptiFlow-BC, which replaces the VaBC reference policy with a standard BC reference while keeping the OT distillation procedure unchanged.

Table 15: Reference and one-step policy performance. We report the performance of the reference (Ref.) and distilled one-step policies for FQL, OptiFlow-BC, GFP, and OptiFlow. We follow the same evaluation protocol as Table 9.
<table><tr><td></td><td colspan="2">FQL</td><td colspan="2">OptiFlow-BC</td><td colspan="2">GFP</td><td colspan="2">OptiFlow</td></tr><tr><td>Task</td><td>BC Ref.</td><td>one-step</td><td>BC Ref.</td><td>one-step</td><td>VaBC Ref. one-step</td><td></td><td>VaBC Ref. one-step</td><td></td></tr><tr><td>antmaze-large-navigate{1}</td><td> $1 . 5 { \pm } 1 . 0 $ </td><td> $7 9 . 5 { \pm 2 . 1 }$ </td><td> $1 . 1 { \pm } 0 . 5$ </td><td> ${ \bf 9 1 . 0 { \pm } } 2 . 0$ </td><td> $8 8 . 5 { \pm } 2 . 3 $ </td><td> $\mathbf { 9 4 . 5 \pm 1 . 4 }$ </td><td> $1 8 . 4 { \pm } 3 . 9$ </td><td> ${ \bf 9 4 . 0 { \scriptstyle \pm 0 . 9 } }$ </td></tr><tr><td>antmaze-giant-navigate{1}</td><td>0.0±0.0</td><td> ${ \bf 1 0 . 5 \pm 2 . 9 }$ </td><td>0.0±0.0</td><td> $0 . 9 { \pm } 0 . 3$ </td><td> $0 . 3 { \pm } 0 . 2$ </td><td> $8 . 6 \pm 2 . 5$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $5 . 4 \pm 2 . 4$ </td></tr><tr><td>humanoidmaze-medium-navigate{1}</td><td>0.5±0.3</td><td> $1 9 . 0 { \pm } 4 . 2 $ </td><td>0.8±0.3</td><td> $5 0 . 5 { \pm } 8 . 3 $ </td><td> $2 8 . 8 { \pm } 2 . 4 $ </td><td> $\mathbf { 8 4 . 8 \pm 2 . 1 }$ </td><td> $4 6 . 1 \pm 1 . 1$ </td><td> ${ \bf 8 2 . 4 \pm 1 . 2 }$ </td></tr><tr><td>humanoidmaze-large-navigate{1}</td><td>0.0±0.0</td><td> $1 . 0 { \pm } 0 . 5$ </td><td>0.0±0.0</td><td> $4 3 . 4 \pm 4 . 1$ </td><td> $2 . 1 \pm 1 . 1$ </td><td> $3 6 . 0 { \pm } 5 . 8 $ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> ${ \bf 4 8 . 3 \pm 2 . 2 }$ </td></tr><tr><td>antsoccer-arena-navigate{4}</td><td>1.4±0.2</td><td> $3 8 . 1 \pm 4 . 4$ </td><td> $1 . 4 \pm 0 . 5$ </td><td> $4 3 . 6 { \pm } 3 . 4 $ </td><td> $7 . 8 \pm 1 . 6$ </td><td> $4 2 . 5 { \pm } 3 . 8 $ </td><td> $1 1 . 8 { \pm } 2 . 2 $ </td><td> ${ \bf 4 6 . 8 \pm 4 . 1 }$ </td></tr><tr><td>cube-single-play{2}</td><td> $1 0 . 4 \pm 2 . 6$ </td><td> ${ \bf 9 6 . 1 \pm 0 . 6 }$ </td><td>8.6±1.5</td><td> ${ \bf 9 7 . 5 } \pm 1 . 3$ </td><td> $4 4 . 4 { \pm } 6 . 0$ </td><td> $\mathbf { 9 7 . 7 \pm 1 . 1 }$ </td><td> $4 5 . 1 { \pm } 7 . 1$ </td><td> ${ \bf 9 8 . 5 { \scriptstyle \pm 0 . 5 } }$ </td></tr><tr><td>cube-double-play{2}</td><td> $0 . 3 { \pm } 0 . 2$ </td><td>33.8±5.1</td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $3 9 . 1 { \pm } 3 . 5 $ </td><td> $2 . 3 { \pm } 0 . 6 $ </td><td> $4 8 . 3 { \pm } 3 . 7 \ $ </td><td>1.4±0.4</td><td> ${ \pm } 2 . 4 { \pm } 2 . 9$ </td></tr><tr><td>scene-play{2}</td><td> $2 . 4 \pm 0 . 8$ </td><td> $7 8 . 1 \pm 6 . 4$ </td><td> $2 . 8 { \pm } 0 . 9$ </td><td> $\mathbf { 9 8 . 2 \pm 0 . 8 }$ </td><td> $9 4 . 0 { \pm } 2 . 5 $ </td><td> $9 1 . 0 { \pm } 2 . 9$ </td><td> $6 9 . 8 { \pm } 3 . 7 \ $ </td><td> $\mathbf { 9 7 . 5 } { \pm 1 . 1 }$ </td></tr><tr><td>puzzle-3x3-play{4}</td><td> $0 . 4 \pm 0 . 2$ </td><td> $1 1 . 1 { \pm } 1 . 8 $ </td><td> $0 . 1 { \pm } 0 . 1$ </td><td> $1 . 1 { \pm } 0 . 7$ </td><td> $1 0 . 3 { \pm } 1 . 6 $ </td><td> $5 . 3 { \pm } 1 . 2 $ </td><td> $4 . 3 { \pm } 0 . 5$ </td><td> $2 5 . 3 { \pm } 6 . 8 $ </td></tr><tr><td>puzzle-4x4-play{4}</td><td> $0 . 3 { \pm } 0 . 2$ </td><td> $8 . 8 \pm 1 . 1$ </td><td> $0 . 0 { \pm } 0 . 0 $ </td><td> $2 0 . 9 \pm 4 . 2$ </td><td> $5 . 1 \pm 1 . 6$ </td><td> $1 6 . 2 { \pm } 1 . 7 $ </td><td> $1 . 5 { \pm } 0 . 3 $ </td><td> $2 2 . 8 { \pm } 2 . 9$ </td></tr><tr><td>Average</td><td> $1 . 7 { \pm } 0 . 3$ </td><td> $3 7 . 6 { \pm } 1 . 1$ </td><td> $1 . 5 { \pm } 0 . 2 $ </td><td> $4 8 . 6 { \pm } 1 . 3 $ </td><td> $2 8 . 4 \pm 3 . 4$ </td><td> $5 2 . 5 { \pm } 0 . 9$ </td><td> $1 9 . 8 { \pm } 2 . 8 $ </td><td> ${ \pm } 7 . 3 { \pm } 1 . 0$ </td></tr></table>

As shown in the Table 15, FQL and OptiFlow-BC show comparable reference performance, yet OptiFlow-BC yields a better one-step policy on 8 of 10 tasks, improving the average from 37.6 to 48.6. Since the reference policy is matched by construction, this indicates that a large part of OptiFlow’s gain comes from the distillation process itself. Comparing OptiFlow-BC with full OptiFlow demonstrates that the Value-aware BC reference policy and the OT distillation are complementary: replacing the BC reference policy with a VaBC one improves the one-step average further, from 48.6 to 57.3. VaBC would become unnecessary only with a very large M, since the marginal can only select within the sampled pool. However, larget M increases compute cost, so VaBC is what makes a moderate M sufficient, by improving the quality of the candidate pool.

Notably, OptiFlow’s reference policy is weaker than $\mathrm { G F P ^ { \circ } s }$ on average (19.8 vs 28.4) while its one-step policy is stronger (57.3 vs 52.5), which further indicates that reference performance alone does not determine one-step performance.

## F.6 AWR-style weighting for value-aware behavior cloning

Our reference flow policy is trained using the value-aware behavior cloning (VaBC) objective in Eq. (6), where the weight $g _ { \eta }$ is implemented as a two-class softmax between the value of the dataset action and that of the current one-step policy action. As an alternative, we consider the AWR-style exponential weighting used in GFP [17]. Specifically, OptiFlow-AWR replaces the default weight $g _ { \eta }$ with

$$
g _ { \eta } ^ { \mathrm { A W R } } ( s , a , z ) = \exp \left( \frac { \lambda } { \eta } \left[ Q _ { \phi } ( s , a ) - Q _ { \phi } ( s , \mu _ { \theta } ( s , z ) ) \right] \right) ,\tag{15}
$$

Table 16: AWR-style weighting for value-aware behavior cloning. OptiFlow-AWR replaces the default softmax weight $g _ { \eta }$ in the reference-policy objective with an AWR-style exponential advantage weight, while leaving all other components unchanged. We follow the same evaluation protocol as Table 9.
<table><tr><td>Task</td><td>OptiFlow (default)</td><td>OptiFlow-AWR</td></tr><tr><td>antmaze-large-navigate{1}</td><td> $9 4 . 0 { \pm } 0 . 9$ </td><td> $9 7 . 6 { \pm } 0 . 8 $ </td></tr><tr><td>antmaze-giant-navigate{1}</td><td> $5 . 4 \pm 2 . 4$ </td><td> $4 . 4 { \pm } 1 . 3$ </td></tr><tr><td>humanoidmaze-medium-navigate{1}</td><td> $8 2 . 4 { \pm } 1 . 2 $ </td><td> $8 8 . 6 { \pm } 2 . 4 $ </td></tr><tr><td>humanoidmaze-large-navigate{1}</td><td> $4 8 . 3 { \pm } 2 . 2 $ </td><td> $3 4 . 2 { \pm } 9 . 2 $ </td></tr><tr><td>antsoccer-arena-navigate{4}</td><td> $4 6 . 8 { \pm } 4 . 1$ </td><td> $3 6 . 8 { \pm } 3 . 9$ </td></tr><tr><td>cube-single-play{2}</td><td> $9 8 . 5 { \pm } 0 . 5 $ </td><td> $9 8 . 8 { \pm } 0 . 6 $ </td></tr><tr><td>cube-double-play{2}</td><td> $5 2 . 4 { \pm } 2 . 9$ </td><td> $4 8 . 8 { \pm } 6 . 3 $ </td></tr><tr><td>scene-play{2}</td><td> $9 7 . 5 { \pm } 1 . 1$ </td><td> $9 3 . 4 { \pm } 3 . 4 $ </td></tr><tr><td>puzzle-3x3-play{4}</td><td> $2 5 . 3 { \pm } 6 . 8 $ </td><td> $1 . 8 { \pm } 1 . 8 $ </td></tr><tr><td>puzzle-4x4-play{4}</td><td> $2 2 . 8 { \pm } 2 . 9$ </td><td> $2 0 . 6 { \pm } 3 . 9$ </td></tr><tr><td>Average</td><td> $5 7 . 3 { \pm } 1 . 0 $ </td><td> $5 2 . 5 { \pm } 1 . 4 $ </td></tr></table>

As shown in Table 16, AWR-style weighting improves performance on some tasks, including antmaze-large and humanoidmaze-medium, but does not provide a consistent improvement across tasks. Averaged over the 10 evaluated tasks, OptiFlow-AWR achieves $5 2 . 5 \pm 1 . { \overset { . } { 4 } } .$ , compared with $5 7 . 3 \pm 1 . 0$ for the default weighting. We therefore retain the default two-class softmax weighting for the reference-policy objective.

## F.7 Direct Q-maximization

A standard alternative to transport-based supervision is to optimize the one-step policy directly against the critic. We add a Q-maximization term weighted by $\lambda _ { q }$ (default $\lambda _ { q } = 0 .$ , recovering pure OptiFlow) to the actor objective and sweep its weight to test whether direct critic guidance complements or substitutes for transport-based distillation.

Table 17: Direct Q-maximization ablation. We follow the same evaluation protocol as Table 9. $\lambda _ { q } = 0$ is the OptiFlow default reported in our main results.
<table><tr><td>Task</td><td> $\lambda _ { q } = 0$  (default)</td><td> $\lambda _ { q } = 1 0 ^ { - 3 }$ </td><td> $\lambda _ { q } = 1 0 ^ { - 2 }$ </td><td> $\lambda _ { q } = 1 0 ^ { - 1 }$ </td></tr><tr><td>humanoidmaze-medium-navigate-task1</td><td> ${ \pm 4 . 4 \pm 1 . 2 }$ </td><td> $8 1 . 6 { \pm } 1 . 7$ </td><td> $8 1 . 4 { \pm } 2 . 4 $ </td><td> ${ \bf 8 8 . 8 \pm 1 . 2 }$ </td></tr><tr><td>antsoccer-arena-navigate-task1</td><td> $6 7 . 9 { \pm } 2 . 4 $ </td><td> $4 2 . 0 { \pm } 3 . 0 $ </td><td> $4 5 . 3 { \pm } 2 . 1 $ </td><td> ${ 7 6 . 4 \pm 3 . 3 }$ </td></tr><tr><td>cube-double-play-task2</td><td> $5 2 . 4 2 . 9$ </td><td> ${ \pm 4 . 3 \pm 3 . 3 }$ </td><td> $\mathbf { 5 2 . 9 2 } \mathbf { \pm 3 . 9 }$ </td><td> $7 . 5 { \pm } 1 . 2 $ </td></tr><tr><td>puzzle-4x4-play-task1</td><td> ${ \bf 5 1 . 8 { \pm } 3 . 6 }$ </td><td> $3 9 . 4 { \pm } 5 . 2 $ </td><td> $1 0 . 4 { \pm } 2 . 1 $ </td><td> $0 . 9 { \pm } 0 . 4$ </td></tr></table>

The effect of direct critic maximization is mixed and task-dependent, likely because $\lambda _ { q }$ interacts with τ and η, which control the sharpness of value weighting in the transport plan and reference-policy training. Although $\lambda _ { q } = 0 . 1$ improves humanoidmaze-medium and antsoccer-arena under our setting, it sharply degrades cube-double and puzzle-4×4, and larger values can collapse performance. Jointly tuning $\lambda _ { q }$ with τ and η may improve some tasks, but it introduces another sensitive hyperparameter.

OptiFlow already achieves strong performance across diverse benchmarks with $\lambda _ { q } = 0 ,$ , showing that value-aware transport-based distillation is sufficient in our setting. Together with the diagnostics in Figure. 2 from the main experiment section, this supports using the critic primarily for transport-based target construction rather than direct one-step policy maximization. Accordingly, all main OptiFlow results use $\lambda _ { q } = 0 ,$ , isolating the effect of value-weighted OT distillation and avoiding an additional task-sensitive tuning parameter. This matches our design choice of avoiding direct critic maximization to address mode collapse and overestimation bias in out-of-distribution regions.

## G Offline-to-Online Fine-tuning

While OptiFlow is designed for the offline setting, the structure of OT coupling between the reference policy and one-step target policy is naturally compatible with continued learning from newly collected data. We initialize from an offline OptiFlow checkpoint at 1M training steps and continue training with online interaction, retaining the same reference policy, transport-based target construction, and one-step policy distillation. This study examines whether OptiFlow can convert offline gains into further improvements online without algorithmic modification.

Table 18: Task-specific OptiFlow hyperparameters for offline-to-online RL. We individually tune these hyperparameters for each task. “–” indicates that the default values from Table 5 were used.
<table><tr><td>Task</td><td>Q-agg</td><td>(τ, η)</td></tr><tr><td>humanoidmaze-medium-navigate-singletask-task1-v0</td><td></td><td></td></tr><tr><td>antsoccer-arena-navigate-singletask-task4-v0</td><td></td><td></td></tr><tr><td>cube-double-play-singletask-task2-v0</td><td></td><td></td></tr><tr><td>scene-play-singletask-task2-v0</td><td></td><td></td></tr><tr><td>puzzle-4x4-play-singletask-task4-v0</td><td></td><td></td></tr><tr><td>antmaze-umaze-v2</td><td></td><td></td></tr><tr><td>antmaze-umaze-diverse-v2</td><td></td><td></td></tr><tr><td>antmaze-medium-play-v2</td><td></td><td></td></tr><tr><td>antmaze-medium-diverse-v2</td><td></td><td></td></tr><tr><td>antmaze-large-play-v2</td><td></td><td></td></tr><tr><td>antmaze-large-diverse-v2</td><td></td><td></td></tr><tr><td>pen-cloned-v1</td><td>qmean</td><td>(1.0,1e-2)</td></tr><tr><td>door-cloned-v1</td><td>qmean</td><td>(1.0,1e-5)</td></tr><tr><td>hammer-cloned-v1</td><td>qmean</td><td>(2.0,1e-4)</td></tr><tr><td>relocate-cloned-v1</td><td>qmean</td><td>(1.0,1e-4)</td></tr></table>

Table 19: Offline-to-online RL results. Each cell reports the offline → online performance, and we follow the same evaluation protocol as Table 9. Bold indicates online results within 95% of the best online score on each task. Italics indicate results taken from FQL [16].
<table><tr><td>Task</td><td>IQL</td><td>ReBRAC</td><td>IFQL</td><td>FQL</td><td>Ⅱ OptiFlow</td></tr><tr><td colspan="6">OGBench</td></tr><tr><td>humanoidmaze-medium-navigate{1}</td><td> $2 1 \pm 1 3  1 6 \pm \delta$ </td><td> $1 6 \pm 2 0  1 \pm { \mathit { 1 } }$ </td><td> $7 0 . 5 { \pm } 2 . 3  4 3 . 0 { \pm } 3 . 4$ </td><td></td><td> $1 8 . 1 \pm 1 . 4  2 4 . 6 \pm 3 . 6$ </td><td>79.0±2.0 → 97.1±0.7</td></tr><tr><td>antsoccer-arena-navigate{4}</td><td> $ { \mathcal { Q } } \pm  { \boldsymbol { { 1 } } } \to  { \boldsymbol { { \ O } } }  { \boldsymbol { { \ O } } } \pm  { \boldsymbol { { \ O } } }$ </td><td></td><td> $\ 0 { \pm } o  \ 0 { \pm } o$ </td><td> $4 0 . 8 { \pm } 3 . 5 \to 3 6 . 0 { \pm } 5 . 3$ </td><td> $3 3 . 8 { \pm } 2 . 8  8 2 . 8 { \pm } 2 . 1$ </td><td>41.0±5.1 → 71.5±3.5</td></tr><tr><td>cube-double-play{2}</td><td> $\theta \pm \imath  \theta \pm \sigma$ </td><td></td><td> $6 \pm { \it 5 }  2 8 \pm 2 8$ </td><td> $1 1 . 8 { \pm } 1 . 7  5 5 . 2 { \pm } 5 . 1$ </td><td> $3 5 . 1 \pm 3 . 2  9 4 . 0 \pm 1 . 1$ </td><td> $4 9 . 1 { \pm } 5 . 1  9 4 . 9 { \pm } 1 . 3$ </td></tr><tr><td>scene-play{2}</td><td> $1 4 \pm \imath \imath \to / 0 \pm \jmath$ </td><td></td><td> $5 5 { \pm } 1 0 \to 1 0 0 { \pm } \sigma$ </td><td> $2 9 . 0 { \pm } 4 . 8 \to 9 4 . 2 { \pm } 1 . 7$ </td><td> $7 9 . 5 { \pm } 4 . 2  9 9 . 8 { \pm } 0 . 2$ </td><td> $9 4 . 8 { \pm } 1 . 8 \to 1 0 0 . 0 { \pm } 0 . 0$ </td></tr><tr><td>puzzle-4x4-play{4}</td><td> $5 \pm 2  1 \pm { \mathit { 1 } }$ </td><td></td><td> $8 \pm 4  1 4 \pm 3 5$ </td><td> $2 3 . 5 { \pm } 2 . 3  2 5 . 0 { \pm } 1 6 . 4$ </td><td> $9 . 2 { \pm } 1 . 0  1 0 0 . 0 { \pm } 0 . 0$ </td><td> $1 3 . 0 { \pm } 2 . 5  6 4 . 1 { \pm } 1 7 . 5$ </td></tr><tr><td colspan="6">AntMaze (D4RL)</td><td></td></tr><tr><td>antmaze-umaze</td><td>77 → 96</td><td></td><td>98 → 75</td><td> $9 1 . 5 { \pm } 1 . 5  9 6 . 0 { \pm } 0 . 8$ </td><td>96.8±0.6 → 99.5±0.3</td><td>95.8±1.0 → 99.5±0.3</td></tr><tr><td>antmaze-umaze-diverse</td><td>60 → 64</td><td></td><td>74 → 98</td><td> $6 1 . 0 { \pm } 9 . 8  9 5 . 2 { \pm } 1 . 0$ </td><td>89.0±1.9 → 99.4±0.3</td><td>80.1±2.3 → 99.0±0.4</td></tr><tr><td>antmaze-medium-play</td><td>72 → 90</td><td></td><td>88 → 98</td><td> $3 6 . 2 { \pm } 9 . 8 \to 9 0 . 5 { \pm } 1 . 9$ </td><td>75.8±2.5 → 97.5±0.3</td><td>82.8±2.0 → 98.1±0.5</td></tr><tr><td>antmaze-medium-diverse</td><td>64 → 92</td><td></td><td>85 → 99</td><td> $5 9 . 5 { \pm } 8 . 9  9 2 . 5 { \pm } 1 . 3$ </td><td>65.6±5.7 → 96.0±0.7</td><td>76.5±4.5 → 97.6±0.7</td></tr><tr><td>antmaze-large-play</td><td>38 → 64</td><td></td><td>68 → 32</td><td> $6 7 . 5 { \pm } 2 . 5  8 6 . 5 { \pm } 1 . 6$ </td><td>84.0±2.4 → 93.2±0.6</td><td>83.0±1.6 → 94.2±0.6</td></tr><tr><td>antmaze-large-diverse</td><td> $2 7  6 4$ </td><td></td><td> $6 7  7 2$ </td><td> $6 6 . 5 { \pm } 3 . 0 \to 8 3 . 8 { \pm } 2 . 1$ </td><td> $8 7 . 0 { \pm } 1 . 9  9 5 . 1 { \pm } 0 . 7$ </td><td>83.0±1.7 → 95.8±0.9</td></tr><tr><td colspan="6">Adroit (D4RL)</td><td></td></tr><tr><td>pen-cloned</td><td>84 → 102</td><td></td><td>74 → 138</td><td> $7 1 . 6 { \pm } 2 . 6  1 0 6 . 0 { \pm } 3 . 4$ </td><td>54.4±4.9 → 124.0±1.9</td><td>45.2±2.7 → 132.0±1.9</td></tr><tr><td>door-cloned</td><td> $1  2 0$ </td><td></td><td> $\bf { \Lambda } 0  { \bf 1 0 2 }$ </td><td> $0 . 3 { \pm } 0 . 2  3 5 . 2 { \pm } 3 . 1$ </td><td> $0 . 3 { \pm } 0 . 3 \to 7 2 . 7 { \pm } 4 . 4$ </td><td> $- 0 . 1 { \pm } 0 . 1  7 8 . 0 { \pm } 3 . 2$ </td></tr><tr><td>hammer-cloned</td><td>1 → 57</td><td></td><td> $\gamma \to 1 2 5$ </td><td> $2 . 0 { \pm } 0 . 7  5 2 . 1 { \pm } 5 . 3$ </td><td> $1 . 2 { \pm } 0 . 5  1 1 0 . 0 { \pm } 9 . 8$ </td><td> $2 . 2 { \pm } 1 . 0 \to 9 7 . 6 { \pm } 9 . 9$ </td></tr><tr><td>relocate-cloned</td><td>0 → 0</td><td></td><td>1 → 7</td><td> $- 0 . 1 { \pm } 0 . 0  3 . 2 { \pm } 2 . 1$ </td><td> $0 . 5 { \pm } 0 . 2  7 3 . 6 { \pm } 3 . 8$ </td><td> $- 0 . 1 { \pm } 0 . 0  - 0 . 0 { \pm } 0 . 1$ </td></tr></table>

![](images/c2319670067e9e4c8d1840bd44e7dd4ed476e3c5c8c6c1640fec06fe18ba04b2.jpg)  
Figure 8: Offline-to-online RL results. Online fine-tuning starts at 1M steps. The results are averaged over 8 random seeds and plotted every 200K steps.

We report these 15 tasks to match the offline-to-online task set of FQL [16]. Across the evaluated OGBench and D4RL tasks, online fine-tuning improves over the offline checkpoint without modifying the OptiFlow training procedure. Newly collected transitions are incorporated directly into critic updates, VaBC reference-policy training, and transport-based target construction. These results suggest that the OT coupling structure is compatible with online fine-tuning and can extend beyond the strictly offline setting.

Some exceptions are puzzle-4×4-play-singletask-task4-v0, which exhibits high seed variance under the default offline hyperparameters: five of eight seeds reach a score of 100, while three remain at 0; and adroit tasks that require hyperparameter tuning. Relocate-cloned task can be further tuned to show improving performance. We report this result without task-specific online tuning unless otherwise stated in Table 18. Overall, OptiFlow shows robust offline-to-online fine-tuning behavior across the evaluated tasks.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

• You should answer [Yes], [No], or [N/A].

• [N/A] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.

• Please provide a short (1–2 sentence) justification right after your answer (even for [N/A]).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will also be asked to include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While [Yes] is generally preferable to [No], it is perfectly acceptable to answer [No] provided a proper justification is given (e.g., error bars are not reported because it would be too computationally expensive” or “we were unable to find the license for the dataset we used”). In general, answering [No] or [N/A] is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist",

• Keep the checklist subsection headings, questions/answers and guidelines below.

• Do not modify the questions and only use the provided macros for your answers.

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: We clearly state the contribution points and the scope in the abstract and introduction that are discussed and demonstrated throughout the paper.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide a limitations section before the conclusion of the main paper in section A.1.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not present formal theoretical results such as theorems, propositions, or convergence guarantees.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide the full code-base needed to exactly replicate the benchmark experiments. We include all hyperparameters and protocols in the Table 5. The provided code-base can reproduce the experiments with simple guideline in the README.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We include a git repository of the code-base which contains all experiment setups needed to run the experiments as we denote in the details within the Table 5 in the appendix.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: We specify every eval protocol, setup, and hyperparameters in cleanly organized tables starting with the Table 5.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report errors for all experimental results where appropriate. We state for each experimental table that the results are reported in standard error.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide the details on compute in the discussion section A.4.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: Yes, we understand and abide by rules and conform with the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We include our statement as a separate section in Appendix B.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [N/A]

Justification: We release our code-base, but it is not subject to safeguards requirement.

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly cite all calls or uses of existing assets whenever appropriate.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We provide our code-base as supplement materials as a link to anonymized repository with self-contained, explanatory README.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: Our work does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: Our work does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: We do not use LLM for any important, original, or non-standard components. Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.