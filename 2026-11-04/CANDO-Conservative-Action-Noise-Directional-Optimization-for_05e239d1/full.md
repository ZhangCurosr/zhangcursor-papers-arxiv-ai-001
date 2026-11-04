# CANDO: Conservative Action-Noise Directional Optimization for Residual Reinforcement Learning

Zeyu Gao<sup>1,2∗</sup>, Chengkai Hou<sup>3∗</sup>, Chaoyue Li<sup>4</sup>, Zhuangzhe Wu<sup>3,4</sup>, Jiaming Liu<sup>3,4</sup>, Yao Mu<sup>5</sup>, Peng Jia<sup>4</sup>, Li Zhang<sup>1,2†</sup> and Shanghang Zhang<sup>3†</sup>

Abstract— Pretrained generative policies provide strong behavior priors for robot control, but adapting them efficiently with online reinforcement learning remains difficult. A central challenge is how to improve a frozen generative policy without destroying the structure learned from offline demonstrations. Noise-space residual methods address this by modifying the policy’s input noise, but existing noise-space methods typically regulate adaptation through fixed latent feasible regions, which may exclude valuable noise edits. In contrast, action-space residual methods are more sample-efficient, but their efficiency can still be further improved. We propose Conservative Noise Directional Optimization (CNDO), a noise-space residual RL method that improves a frozen generative policy by learning state- and noise-conditioned directions around Gaussian latent samples. CNDO removes predefined latent feasible regions and instead regulates noise-space adaptation through the soft conservative constraint, which consists of conservative value estimation and gated execution. We further introduce Conservative Action-Noise Directional Optimization (CANDO), which adds an action-space residual, combining structured latentspace improvement with local action correction. Experiments show that, once CNDO narrows the gap between noise-space and action-space residual learning, CANDO further improves over single-space optimization, achieving 76.1 NAUC and a 91.4% final success rate, surpassing the best single-space results by 6.6 NAUC and 4.1% success rate. More demos and info are available on https://cando-rl.github.io/.

## I. INTRODUCTION

Learning robot policies from offline demonstrations provides an effective way to acquire complex manipulation skills without manually designing task-specific controllers. In particular, generative policies based on diffusion models and conditional flow matching can represent complex and multi-modal action distributions and have demonstrated strong performance across a broad range of robotic tasks [1], [2]. However, behavior cloning is inherently limited by the coverage of demonstration data and is susceptible to covariate shift during closed-loop deployment [3]. Online reinforcement learning (RL) provides a natural means to further improve pretrained policies through task feedback, but directly finetuning high-capacity generative policies typically requires optimization through the full generation process, which can introduce substantial computational cost and training instability [4], [5]. This motivates lightweight adaptation methods that keep the pretrained policy frozen while learning only a task-specific adaptation module.

Residual reinforcement learning provides a natural framework for this setting by keeping a base policy fixed and learning only an additional residual correction [6]. For a frozen generative policy, residual adaptation can operate in two distinct spaces. In noise space, the input noise is modified before generation to alter the behavior produced by the frozen policy; in action space, the generated action is directly corrected after decoding. Diffusion Steering via Reinforcement Learning (DSRL) represents the former approach by treating the input noise as the action of a latent-space Markov decision process [5]. In contrast, Distribution Contractive Reinforcement Learning (DICE) learns a lightweight residual over the generated action and combines residual optimization with value-guided action selection [7]. Although both spaces provide viable interfaces for online policy improvement, two important questions remain: can noise-space residual learning achieve sample efficiency comparable to strong action-space residual learning, and can the two spaces be jointly optimized to further improve policy learning?

Existing noise-space methods are substantially less sampleefficient than DICE on most of the tasks considered in our experiments. We find that this gap is partly associated with how noise-space policy updates are regulated. DSRL restricts latent actions to a bounded region, Latent Policy Steering (LPS) projects them onto the typical Gaussian hypersphere, and Lagrangian Perturbation Diffusion Steering (LP-DS) retains Gaussian noise samples while constraining their residual perturbations through a Lagrangian trust region [5], [8], [9]. Such mechanisms improve training stability by explicitly limiting the region that the noise policy can explore. However, latent displacement is only indirectly related to the quality of the behavior produced after decoding. Consequently, a predefined geometric region may reject large but valuable latent edits while still admitting small edits that do not improve the resulting behavior. This suggests that noise-space adaptation should be regulated according to the quality of the decoded behavior, rather than solely by a predefined geometric region in latent space.

To address this gap, we propose Conservative Noise Directional Optimization (CNDO), which removes the predefined latent feasible region and instead evaluates and regulates noise edits through their decoded behaviors. Removing hard geometric restrictions enables broader noisespace exploration, while increasing the risk of unstable optimization. CNDO introduces a soft conservative constraint consisting of two components: a conservative critic and gated execution. The conservative critic suppresses unsupported value overestimation when optimizing noise directions, while the gate selectively executes proposed edits according to target-critic value estimates. In this way, CNDO expands the effective optimization range in noise space while preventing unrestricted exploration from destabilizing policy learning.

![](images/adeefa8c27425ab669054c95ed5e6d84ac36f231c92335594c55690da2c60694.jpg)

![](images/b316deeff50a8161b6a456bed64481f837ee785acfbf31815e8d5a8558317cc4.jpg)  
Fig. 1. Motivation for residual learning in noise and action spaces. (a) Noise-space residual RL edits the input noise before frozen-policy decoding, while action-space residual RL corrects decoded actions. (b) Existing noise-space methods show a clear sample-efficiency gap relative to DICE. (c) Naively combining DSRL and DICE provides no additional improvement.

We further combine CNDO with action-space residual learning and introduce Conservative Action-Noise Directional Optimization (CANDO). The two residual branches affect the final policy through different pathways: the noisespace residual modifies the behavior generated through the frozen policy, whereas the action-space residual directly corrects the generated action. However, having access to both adaptation spaces does not necessarily guarantee additional gains from joint training. Our experiments show that the benefit of joint learning depends on the capabilities of both residual learners. When their standalone performance differs substantially, combining the two provides only limited improvement over the stronger learner. Once noise-space learning is improved to a level comparable to action-space residual learning, joint training can more effectively exploit policy improvements from both spaces.

Our main contributions can be summarized as follows:

• We identify a mismatch between hard latent-geometric constraints and decoded behavioral quality in noise-space residual learning, and propose CNDO to regulate noisespace adaptation through the soft conservative constraint rather than a fixed latent feasible region.

• We further introduce CANDO, which jointly optimizes CNDO and action-space residual learning and achieves higher overall sample efficiency than the best singlespace method.

• Through experiments, we reveal the sample-efficiency gap between noise-space and action-space residual learn ing and further show that joint training provides limited additional benefit when the standalone capabilities of the two residual learners are substantially mismatched.

## II. RELATED WORK

## A. Residual Reinforcement Learning

Residual reinforcement learning improves an existing controller or policy by preserving its effective behavioral prior and learning only the corrections that are missing from the base policy. Early works on Residual Policy Learning and Residual Reinforcement Learning for Robot Control add trainable residual actions on top of hand-designed or model-based controllers, allowing reinforcement learning to avoid searching over the entire control policy from scratch [6], [10]. This paradigm was later extended to contactrich manipulation with visual observations [11], shared autonomy [12], and settings where the base controller is obtained from demonstrations or motion primitives rather than manually designed [13], [14].

## B. Noise-Space Policy Steering

Using latent representations to structure policy optimization predates generative robot policies. PLAS optimizes policies in a VAE-derived latent action space, while OPAL and SPiRL learn temporally extended latent primitives or skill priors from offline data [15]–[17]. LAPO further learns a latent-variable policy through advantage-weighted optimization to generate high-value actions [18]. With diffusion policies, the Gaussian noise input provides a native latent interface for modifying generated behavior. To the Noise and Back demonstrates that manipulating the diffusion process can effectively incorporate external guidance in shared autonomy [19]. More closely related to our setting, DSRL formulates the input noise of a frozen diffusion policy as the action of a latent-space MDP and optimizes it using reinforcement learning [5]. LPS directly optimizes a latent actor using gradients from an originalaction-space critic, backpropagated through a differentiable one-step MeanFlow policy, and constrains the actor output to the Gaussian hypersphere [8]. More recently, LP-DS retains samples from the Gaussian prior and learns residual noise perturbations while regulating their magnitude through a Lagrangian trust-region constraint [9].

## C. Action-Space Residual Reinforcement Learning

In contrast to noise-space steering before policy decoding, action-space residual learning directly corrects the output action. Recent work has extended the classical residual formulation to high-capacity policies pretrained from offline demonstrations. ResiP learns closed-loop per-step residual corrections over frozen action-chunking policies to compensate for execution errors in precise assembly tasks [20]. Policy Decorator introduces a model-agnostic residual adapter for online refinement of large pretrained policies [21], while ResFiT applies off-policy RL to learn lightweight per-step corrections over behavior-cloning policies [22]. Related approaches also use value functions to coordinate pretrained behaviors with online policy improvement. IBRL treats actions proposed by an imitation policy as candidates for both exploration and temporal-difference bootstrapping [23]. DAWN further studies critic cold-start and scale mismatch between base and residual actions as key bottlenecks for value learning in residual RL [24]. Most closely related to our action-space component, DICE combines residual action optimization with selective behavioral regularization and value-guided multicandidate action selection, providing a strong sample-efficient action-space baseline [7]. USR unifies noise-space steering and action-space residual refinement within a single actor, jointly optimizing latent noise and residual actions, but does not thoroughly investigate the interactions and complementary roles of these two optimization strategies [25].

## III. METHOD

We develop our method in two variants. First, we introduce Conservative Noise Directional Optimization (CNDO) to improve noise-space reinforcement learning without imposing a fixed feasible region on the edited latent. We then augment CNDO with an action residual to obtain Conservative Action-Noise Directional Optimization (CANDO), jointly exploiting policy improvement in noise and action spaces. Finally, we summarize the overall framework of CANDO.

![](images/c7a6302b1dbb7d0f190669b504c3f52349637c05139c001cc14942818fbc8553.jpg)

## A. Conservative Noise Directional Optimization

We consider an action-chunk Markov decision process. Given the observation s, a frozen Flow Matching policy generates a base action chunk $a _ { b }$ from Gaussian noise:

$$
w \sim \mathcal { N } ( 0 , I ) , \qquad a _ { b } = \pi _ { \phi } ( s , w ) ,\tag{1}
$$

where $\pi _ { \phi }$ remains frozen throughout online training. We learn an action-value critic $Q _ { \omega } ( s , a )$ directly over decoded action chunks, such that noise-space policy improvements are ultimately evaluated by their induced behaviors.

Existing noise-space methods are shown in Fig. 2. These constraints stabilize policy optimization by explicitly restricting latent exploration. DSRL bounds each latent dimension through a tanh actor, LPS normalizes latent variables onto the typical hypersphere of a high-dimensional Gaussian, and LP-DS combines residual noise perturbations with explicit bounds and a Lagrangian trust-region constraint [5], [8], [9]. Although these constraints improve stability, latent displacement is only an indirect indicator of decoded behavior quality. Large latent changes may produce valuable behaviors, while small changes do not necessarily imply better actions. Geometric constraints may therefore exclude useful noise-induced policy improvements and limit sample efficiency.

CNDO removes the fixed feasible-region constraint and instead learns a residual direction:

$$
\begin{array} { l l } { { \delta w = \pi _ { \theta } ^ { w } ( s , w ) , } } & { { \quad \tilde { w } = w + \delta w , } } \\ { { a _ { w } = \pi _ { \phi } ( s , \tilde { w } ) , } } & { { \quad \delta a _ { w } = a _ { w } - a _ { b } . } } \end{array}\tag{2}
$$

This Gaussian-anchored residual parameterization follows the same general form as LP-DS [9]. Instead, CNDO focuses on how such directions can be reliably optimized and executed after removing the explicit latent constraint.

Removing the hard constraint enables broader noise-space exploration, but also exposes the critic to decoded actions outside the available behavioral support. Critic extrapolation errors can assign spuriously high values to such actions and drive the direction actor toward increasingly unreliable latent regions. We address this issue with a soft conservative constraint consisting of two complementary components: conservative critic for reliable direction learning and gated execution for selective deployment.

For conservative critic, we maintain an ensemble of M critics and use their minimum whenever a conservative policy or target value is required. The critics are trained from expert and online replay using a multi-sample n-step TD objective:

![](images/807b13c2b9bef5fcd31a34351d33b2d1c1e158dcf57b7551e69982c7bdbf1a75.jpg)

![](images/e5a0f25ab5a903f0c2d341db6b8a9a9c0710eff6c2ea921ed93e6741ff521623.jpg)

![](images/e2896fe611a6cb3cf817fd5f97f5cca5a831992c4ad05018a88b10480a0b9531.jpg)  
Fig. 2. Comparison of noise-space regulation strategies. (a) DSRL bounds latent variables with tanh squashing. (b) LPS projects latents onto the Gaussian hypersphere. (c) LP-DS constrains residual noise with clipping and a trust region. (d) CNDO removes fixed geometric constraints and instead uses conservative value estimation and gated execution.

![](images/b4f0bc7cffbc60366a684e989f6b8d88ab2a729b10c41a81e537204287f28c10.jpg)  
Fig. 3. Overview of CANDO. During online inference, a noise direction modifies the frozen-policy output, the gate selects whether to use the edit, and an action residual further refines the action. During off-policy learning, the critic is trained conservatively, the gate is supervised by the target critic, and both residual branches are jointly optimized. The pretrained policy remains frozen throughout.

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { T D } } = \sum _ { j = 1 } ^ { M } \mathbb { E } [ ( Q _ { \omega _ { j } } \big ( s _ { t } , a _ { t } \big ) - y _ { t } ) ^ { 2 } ] , } } \\ & { } & { y _ { t } = R _ { t } ^ { ( n ) } + \gamma ^ { n } ( 1 - d _ { t } ) \frac { 1 } { K _ { \mathrm { n e x t } } } \sum _ { k = 1 } ^ { K _ { \mathrm { n e x t } } } Q _ { \bar { \omega } } \big ( s _ { t + n } , a _ { k } ^ { \prime } \big ) . ~ } \end{array}\tag{3}
$$

To further suppress extrapolation errors caused by unrestricted noise edits, we apply conservative regularization only to candidates outside behavioral support:

$$
\begin{array} { r l } & { \mathcal { L } _ { Q } = \mathcal { L } _ { \mathrm { T D } } + \alpha _ { \mathrm { C } } \mathcal { L } _ { \mathrm { C } } , } \\ & { \mathcal { L } _ { \mathrm { C } } = \displaystyle \sum _ { j = 1 } ^ { M } \frac { \sum _ { k } m _ { k } T \mathrm { s o f t p l u s } \left( \frac { Q _ { \omega _ { j } } ( s _ { t } , a _ { k } ^ { \pi } ) - Q _ { \omega _ { j } } ( s _ { t } , a _ { t } ) } { T _ { C } } \right) } { \operatorname* { m a x } ( \sum _ { k } m _ { k } , 1 ) } , } \\ & { m _ { k } = \mathbb { I } \left[ \operatorname* { m i n } \left( \lVert a _ { k } ^ { \pi } - a _ { k } \rVert _ { 2 } , \lVert a _ { k } ^ { \pi } - a _ { t } \rVert _ { 2 } \right) > \epsilon _ { \mathrm { s u p } } \right] . } \end{array}\tag{4}
$$

Here, $a _ { k } ^ { \pi }$ is a detached candidate generated by the current policy, while $a _ { k }$ is the frozen-policy action generated from the same Gaussian anchor. Thus, candidates supported by either the pretrained behavior or replay data remain unpenalized, while unsupported value overestimation is selectively suppressed.

The direction network directly optimizes the complete edited behavior:

$$
\mathcal { L } _ { \mathrm { n o i s e } } = - \mathbb { E } _ { s , w } \left[ Q _ { \omega } ( s , a _ { w } ) \right] + \beta _ { w } \mathbb { E } _ { s , w } \left[ \| \delta w \| _ { 2 } ^ { 2 } \right] .\tag{5}
$$

Gradients are propagated through the frozen generative policy to $\pi _ { \theta } ^ { w }$ , while the critic and pretrained-policy parameters remain fixed. The second term only provides a weak preference for smaller edits and does not impose a fixed latent feasible region.

Conservative critic reduces unreliable direction optimization, but critic estimates can still be inaccurate, especially early in training. CNDO therefore introduces a chunk-level binary gate $G _ { \eta }$ to regulate the execution of noise edits. Given $( s , w , \tilde { w } )$ , the gate predicts a distribution $p _ { \eta } ( b \mid s , w , \tilde { w } )$ over $b \in \{ 0 , 1 \}$ , and the deployed decision is

$$
a _ { \mathrm { C N D O } } = \pi _ { \mathrm { C N D O } } ( s , w ) = a _ { b } + b \delta a _ { w } .\tag{6}
$$

The gate is trained by distilling the relative values of these two behaviors from the target critic:

$$
\mathcal { L } _ { \mathrm { g a t e } } = - \mathbb { E } _ { s , w } \sum _ { b \in \{ 0 , 1 \} } q _ { \bar { \omega } } ( b \mid s , w , \tilde { w } ) \log p _ { \eta } ( b \mid s , w , \tilde { w } ) ,\tag{7}
$$

where $q _ { \bar { \omega } }$ is the teacher distribution obtained by normalizing and temperature-scaling the target-critic scores $Q _ { \bar { \omega } } ( s , a _ { b } +$ $b \delta a _ { w } ) - \beta _ { \lambda } b ^ { 2 }$ , where $\beta _ { \lambda }$ is the gate-opening penalty factor, such that the noise-edited behavior is favored only when its estimated value improvement is sufficient to offset the gate-opening penalty. Furthermore, we use the more stable target critic to distill the gate, yielding more stable gating decisions than using the online critic either for distillation or for direct action selection.

## B. Action-Noise Joint Residual Training

CNDO adapts the frozen policy before decoding by editing its input noise. To additionally exploit the post-decoding action space, we introduce an action residual conditioned on the state and action. Specifically, CANDO is defined as

$$
\begin{array} { r l } & { a _ { \mathrm { C A N D O } } = \pi _ { \mathrm { C A N D O } } ( s , w , a _ { b } ) } \\ & { \phantom { a a a a a a a a a a a a } = \pi _ { \mathrm { C N D O } } ( s , w ) + a _ { r } } \\ & { \phantom { a a a a a a a a a } = a _ { b } + b \delta a _ { w } + a _ { r } , } \\ & { a _ { r } = \pi _ { \rho } ^ { r } ( s , a _ { b } ) , } \end{array}\tag{8}
$$

where $a _ { r }$ denotes the action-space residual.

The two branches are jointly optimized with respect to the final CANDO action:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { j o i n t } } = - \mathbb { E } _ { s , w } \left[ Q _ { \omega } ( s , a _ { C A N D O } ) \right] } \\ & { \quad \quad \quad + \beta _ { w } \mathbb { E } _ { s , w } \left[ \| \delta w \| _ { 2 } ^ { 2 } \right] + \beta _ { r } \mathbb { E } _ { s , w } \left[ \| a _ { r } \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{9}
$$

![](images/4c93568e2ed80729734f61db8736c989b8a8481de5d65a9e28fb179f264b14b3.jpg)  
Fig. 4. Noise-space policy improvement on five tasks. Lines and shaded regions show the mean and sample standard deviation over three seeds; the gray dashed line is the frozen base policy.

The first term jointly improves the noise direction and action residual, while the remaining terms weakly regularize their respective magnitudes. CANDO retains CNDO’s gate exclusively for noise-space edits, as the gate is designed to regulate noise-space adaptation rather than the actionspace residual. The action residual is applied after gating and therefore remains active when $b = 0$ , in which case the executed action is $a _ { b } + a _ { r }$ . We also evaluated alternative conditioning variables for the action residual.

CANDO consequently combines two distinct policyimprovement paths: the noise branch modifies behavior through the frozen generator, whereas the action branch directly refines the decoded action.

## C. Overall CANDO Framework

We summarize CANDO in Alg. 1. CANDO alternates between off-policy parameter updates and online data collection. During each update, we use RLPD [26] to sample states and transitions from a mixture of expert and online replay, and sample fresh Gaussian latents to construct the base behavior $^ { a _ { b } , }$ the noise-edited behavior $a _ { w } ,$ and the action residual $a _ { r } .$ The complete CANDO policy is used to generate next-policy actions and conservative candidates for critic learning. We first update the critics using Eq. (4), distill the gate from the target critic using Eq. (7), and jointly optimize the noise-direction and action-residual networks using Eq. (9). Finally, the target critics are updated by Polyak averaging. The pretrained Flow Matching policy remains frozen throughout training.

## IV. EXPERIMENTS

We design the experiments to answer the following questions:

• Q1: Can CNDO improve noise-space reinforcement learning to a level comparable to strong action-space residual methods?

• Q2: How do conservative critic and gated execution contribute to CNDO, and how does CNDO compare with hard constrained directional noise-space methods?

• Q3: How does the standalone capability of noise-space and action-space learners affect the benefit of joint residual training?

## A. Experimental Setup

Tasks. We use state observations on five manipulation tasks from two benchmarks: Can, Square, and Transport from RoboMimic [27], and PegInsertionSide and StackCube from ManiSkill [28].

Algorithm 1 Conservative Action-Noise Directional Opti  
mization (CANDO)   
1: Input: Expert replay ${ \mathcal { D } } _ { E } ,$ frozen policy $\pi _ { \phi } ,$ update steps   
U, and interaction steps $T$   
2: Initialize: Online replay $\mathcal { D } _ { O }$ , direction network $\pi _ { \theta } ^ { w } .$   
action-residual network $\pi _ { \rho } ^ { r } ,$ gate network $G _ { \eta } ,$ critics $Q _ { \omega } ,$   
and target critics $Q _ { \bar { \omega } }$   
3: while not converged do   
//Off-policy parameter learning   
4: for update step $u = 1$ to U do   
5: Sample a batch $\boldsymbol { \mathcal { B } } \sim \mathrm { R L P D } ( \mathcal { D } _ { E } , \mathcal { D } _ { O } )$   
6: Sample $w \sim \mathcal { N } ( 0 , I )$ and compute $a _ { b } = \pi _ { \phi } ( s , w )$   
7: Compute $\delta w = \pi _ { \theta } ^ { w } ( s , w ) , a _ { w } = \pi _ { \phi } ( s , w + \delta w )$ , and   
$\delta { a } _ { w } = { a } _ { w } - { a } _ { b }$   
8: Compute $a _ { r } = \pi _ { \rho } ^ { r } ( s , a _ { b } )$   
9: Infer $b = \arg \operatorname* { m a x } _ { b \in \{ 0 , 1 \} } p _ { \eta } ( b \mid s , w , \tilde { w } )$   
10: Construct the deployable CANDO action $a = a _ { b } +$   
$b \delta a _ { w } + a _ { r }$   
11: Generate next-policy actions and conservative can  
didates using the complete CANDO policy   
12: Update critics ω using Eq. (4)   
13: Update gate $\eta$ using Eq. (7)   
14: Update actors θ and $\rho$ using Eq. (9)   
15: Update target critics by Polyak averaging $\bar { \omega } = ( 1 -$   
$\tau ) \bar { \omega } + \tau \omega$   
16: end for   
//Online data collection   
17: for interaction step $t = 1$ to $T$ do   
18: Infer and execute CANDO action $a _ { t }$   
19: Add $( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } , d _ { t } )$ to $\mathcal { D } _ { O }$   
20: end for   
21: end while

Online training and evaluation. Training uses 16 environments and batch 256 with sparse rewards. Hyperparameters are: updates per collection $U = 1 0 ;$ return steps $n = 3 ;$ discount $\gamma = 0 . 9 9 ;$ critic count $M = 1 0 ;$ next-policy samples $K _ { \mathrm { n e x t } } = 4 ;$ Polyak rate $\tau = 0 . 0 1 ;$ conservative regularization weight $\alpha _ { \mathrm { C } } = 0 . 1$ ; conservative regularization temperature $T _ { C } ~ = ~ 0 . 0 1 ;$ support threshold $\begin{array} { r l r } { \epsilon _ { \mathrm { s u p } } } & { { } = } & { 1 ; } \end{array}$ noise penalty $\beta _ { w } = 1 0 ^ { - 4 }$ ; gate penalty $\beta _ { \lambda } = 3 \times \mathrm { { 1 \bar { 0 } ^ { - 3 } } }$ and residual action penalty $\beta _ { r } = 1 0 0$ (RoboMimic) and 3000 (ManiSkill). We run three seeds (42, 43, and 44) and report the mean and sample standard deviation.

![](images/2f43730528c0c48c1b49572d7e5f2c29340f4a27df9a7acfdc82612f7e751a2a.jpg)  
Base policy DSRL LP-DS CNDO w/o conservative critic CNDO w/o gated execution CNDO  
Fig. 5. CNDO component analysis on five tasks. CNDO “w/o gated execution” always executes the full latent edit, and “w/o conservative critic” sets the behavior-support conservative loss weight to zero. Lines and shaded regions denote the mean and sample standard deviation over three seeds; the gray dashed line is the frozen base policy.

Metrics. Let B be the max environment steps of each task, and let $\mathrm { S R } ( n ) \in [ 0 , 1 ]$ denote success rate after n environment transitions. We measure sample efficiency by

$$
\mathrm { N A U C } = \frac { 1 0 0 } { B } \int _ { 0 } ^ { B } \mathrm { S R } ( n ) \mathrm { d } n ,\tag{10}
$$

using linear interpolation between evaluation points and trapezoidal integration. We also report 100 SR(B) as Final SR. The overall score gives every task equal weight. The frozen behavior-policy result is treated as a constant curve.

Baseline algorithms and proposed methods. We compare the following frozen-policy adaptation methods:

1) Base policy: the pretrained frozen Flow Matching behavior policy without online adaptation.

2) DSRL [5]: a noise-space method that learns latent inputs for the frozen generative policy.

3) LPS [8]: a latent-policy method that directly propagates action-space value gradients through a differentiable flow decoder while constraining the latent input to the typical Gaussian hypersphere.

4) LP-DS [9]: residual noise steering with an explicit latent bound and a Lagrangian trust-region constraint.

5) CNDO: our noise-space directional learner with conservative value estimation and gated execution.

6) DICE [7]: an action-space residual learner with valueguided candidate selection.

7) DSRL+DICE: Combination of DSRL noise learning and DICE action residual learning, following a formulation similar to USR [25].

8) LP-DS+DICE: combination of LP-DS noise learning and DICE action residual learning.

9) CANDO: our joint CNDO and action-residual learner.

## B. Noise-Space Policy Improvement

To address Q1, we compare CNDO with existing noisespace methods and use DICE as the action-space reference. Fig. 4 and Tab. I show that CNDO consistently improves over the frozen policy and is the strongest noise-space method on average. Relative to DSRL, CNDO raises average NAUC from 57.9 to 69.5 and Final SR from 66.2% to 81.5%; relative to LP-DS, the gains are 9.6 NAUC points and 21.0 Final SR points. Relative to the LPS, the gains are 27.8 NAUC points and 31.3 Final SR points. The improvement is particularly large on Transport, PegInsertionSide, and StackCube. Can is the exception in NAUC: its pretrained success rate is already 87.4%, and LP-DS reaches 93.0 NAUC compared with 91.6 for CNDO and 91.4 for LPS. The LPS port obtains 41.7 average NAUC and 50.2% average Final SR. The results indicate that spherical latent geometry is not an effective constraint across these five tasks.

CNDO closes the sample-efficiency gap to the action-space reference. The average NAUC of CNDO is 69.5, compared with 68.5 for DICE. But DICE has the higher average Final SR (87.3% versus 81.5%), mostly because it reaches 94.1% on StackCube. Thus, CNDO is the strongest noise-space method and brings noise-space learning to an overall sampleefficiency level comparable to DICE.

## C. Ablation Studies of Soft Conservative Constraint

This experiment addresses Q2. We compare full CNDO with different components in Fig. 5 that set the conservative critic weight to zero or force the execution gate to one. We additionally include LP-DS as a geometrically constrained directional-optimization reference; unlike CNDO, LP-DS enforces a fixed latent bound.

Removing conservative critic lowers average NAUC from 69.5 to 58.2. It prevents a severe failure on PegInsertionSide, where Final SR falls from 83.1% to 0.1% without it. This supports using the conservative critic to improve stability when the fixed latent feasible region is removed.

Always executing the edit lowers average NAUC to 57.5 and Final SR to 72.2%. The largest NAUC losses occur on Transport (71.2 to 46.4), PegInsertionSide (65.8 to 49.7), and StackCube (50.9 to 36.6). The gate primarily improves sample efficiency and cross-task robustness, rather than guaranteeing a higher endpoint on every task. Finally, full CNDO exceeds the geometrically constrained LP-DS comparator by 9.6 average NAUC points, consistent with the hypothesis that a fixed latent region can reject useful decoded improvements.

## D. Action-Noise Joint Policy Improvement

To answer Q3, we compare the standalone and joint versions of the noise-space and action-space learners, with the results shown in Fig. 6. In particular, we contrast the direct combination DSRL+DICE and LP-DS+DICE with CANDO to determine whether improving the standalone noise learner changes the benefit of joint training.

The standalone DSRL and LP-DS learners trail DICE by 10.6 and 8.6 average NAUC points, respectively. Directly

TABLE I  
MAIN COMPARISON RESULTS. FOR EACH TASK, NAUC AND FINAL SR ARE REPORTED IN SEPARATE COLUMNS, IN PERCENT, AVERAGED OVER THREE SEEDS. BOLD AND UNDERLINED VALUES DENOTE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Space</td><td colspan="2">Can</td><td colspan="2">Square</td><td colspan="2">Transport</td><td colspan="2">PegInsertionSide</td><td colspan="2">StackCube</td><td colspan="2">Average</td></tr><tr><td>NAUC</td><td>Final SR</td><td>NAUC</td><td>Final SR</td><td>NAUC</td><td>Final SR</td><td>NAUC</td><td>Final SR</td><td>NAUC</td><td>Final SR</td><td>NAUC Final SR</td><td></td></tr><tr><td>Base policy</td><td>Base</td><td>1</td><td>87.4</td><td>1</td><td>57.8</td><td>1</td><td>54.0</td><td>1</td><td>49.7</td><td>1</td><td>40.0</td><td>1</td><td>57.8</td></tr><tr><td>DSRL [5]</td><td>Noise</td><td>90.6</td><td>95.8</td><td>55.7</td><td>66.1</td><td>58.5</td><td>75.7</td><td>46.4</td><td>47.9</td><td>38.5</td><td>45.4</td><td>57.9</td><td>66.2</td></tr><tr><td>LPS [8]</td><td>Noise</td><td>91.4</td><td>90.0</td><td>58.0</td><td>69.4</td><td>22.5</td><td>55.4</td><td>9.1</td><td>1.4</td><td>27.3</td><td>34.8</td><td>41.7</td><td>50.2</td></tr><tr><td>LP-DS [9]</td><td>Noise</td><td>93.0</td><td>94.4</td><td>57.5</td><td>59.4</td><td>55.2</td><td>55.0</td><td>52.5</td><td>51.8</td><td>41.2</td><td>41.9</td><td>59.9</td><td>60.5</td></tr><tr><td>CNDO</td><td>Noise</td><td>91.6</td><td>96.0</td><td>68.1</td><td>80.6</td><td>71.2</td><td>87.7</td><td>65.8</td><td>83.1</td><td>50.9</td><td>60.1</td><td>69.5</td><td>81.5</td></tr><tr><td>DICE [7]</td><td>Action</td><td>92.9</td><td>97.4</td><td>63.8</td><td>82.2</td><td>68.4</td><td>89.8</td><td>47.5</td><td>72.9</td><td>70.0</td><td>94.1</td><td>68.5</td><td>87.3</td></tr><tr><td>DSRL+DICE</td><td>Joint</td><td>90.7</td><td>97.1</td><td>66.0</td><td>78.9</td><td>67.2</td><td>83.4</td><td>57.9</td><td>70.3</td><td>58.2</td><td>83.3</td><td>68.0</td><td>82.6</td></tr><tr><td>LP-DS+DICE</td><td>Joint</td><td>90.4</td><td>96.8</td><td>55.0</td><td>72.9</td><td>59.0</td><td>80.4</td><td>60.9</td><td>72.0</td><td>72.7</td><td>91.8</td><td>67.6</td><td>82.8</td></tr><tr><td>CANDO</td><td>Joint</td><td>93.2</td><td>97.2</td><td>72.2</td><td>89.4</td><td>74.1</td><td>91.7</td><td>70.9</td><td>88.2</td><td>70.4</td><td>90.6</td><td>76.1</td><td>91.4</td></tr></table>

![](images/abba597de45b642dc9fbebd0e067f67bcb07389e7f3c9f9c41e624a93dda2407.jpg)  
Base policy DSRL CNDO DICE DSRL+DICE LP-DS+DICE CANDO

Fig. 6. Joint noise-action residual learning on five tasks. Lines and shaded regions denote the mean and sample standard deviation over three seeds; the gray dashed line is the frozen base policy.

combining either noise-space learner with DICE does not improve the stronger action-space branch: DSRL+DICE obtains 68.0 average NAUC and 82.6% Final SR, while LP-DS+DICE obtains 67.6 average NAUC and 82.8% Final SR, both below DICE at 68.5 NAUC and 87.3% Final SR. At the task level, although joint training provides only marginal or no NAUC improvements on the other tasks, PegInsertionSide deserves particular attention. On this task, DSRL+DICE substantially improves NAUC when DSRL and DICE have comparable standalone performance, and LP-DS+DICE exhibits a similar pattern.

After improving the standalone noise learner, CNDO and DICE are closely matched in average NAUC. Their joint model CANDO reaches 76.1 average NAUC, gains of 6.6 points over CNDO and 7.6 points over DICE, and raises average Final SR to 91.4%. CANDO ranks first or second on every reported metric except StackCube Final SR, but it does not consistently outperform the single-space methods on every task. This pattern is particularly clear on PegInsertionSide and StackCube. On PegInsertionSide, CNDO substantially outperforms DICE, while CANDO performs similarly to CNDO. The opposite pattern is observed on StackCube, where CANDO performs similarly to DICE.

These results empirically support the proposed relationship: joint training is most useful here after the two standalone residual learners have comparable capability.

## E. Real-world Experiments

We validate DSRL, DICE, CNDO, and CANDO on four tasks using the Simplexity Robotics i7PRO dual-arm robot shown in Fig. 7, equipped with two Wuji hands, one Gemini 335Lg head camera, and two Gemini 305g wrist cameras. All tasks are performed at fixed positions without position randomization. After each policy rollout an episode, we perform 50 residual reinforcement learning update steps. We roll out 20 episodes for each policy and task. The online training process is provided in the supplementary video and on our project website.

As shown in Fig. 8, CANDO achieves the highest average success rate of 87.5%, followed closely by CNDO at 86.25%, while DSRL, DICE, and the frozen base policy achieve 55.0%, 42.5%, and 40.0%, respectively. The success rate of each task in Fig. 8 is measured over 20 independent evaluation trials after residual RL training. The online adaptation process shows that CANDO improves rapidly from the early stage and CNDO continues to improve steadily, whereas DICE exhibits a less consistent trend over prolonged training. Overall, the real-world results resemble the pattern observed on PegInsertionSide: CNDO consistently outperforms DSRL and DICE, while CANDO provides only a small additional gain over CNDO. This result is consistent with our simulation findings: once the noise-space learner itself becomes sufficiently strong, joint training tends to achieve performance comparable to the stronger single-space method, rather than consistently surpassing the best single-space method on every task.

![](images/1908be7c0e908f27892f1669408d1284da10d4e2cd7b57abbb046b0f17f17f29.jpg)  
Fig. 7. Real-world snapshots of our platform and the execution process of tasks.

![](images/ae3733c57bb51277a57b193578d7ae0260988861c034894b2a10e4dc52e4c7af.jpg)  
Fig. 8. Success rates on real-world tasks. We report the success rates measured over 20 evaluation trials for each task.

## V. CONCLUSION

This work introduced Conservative Noise Directional Optimization (CNDO) and Conservative Action-Noise Directional Optimization (CANDO) for sample-efficient residual RL. CNDO improves noise-space residual learning without imposing a fixed latent feasible region. It optimizes Gaussiananchored residual directions with conservative value estimation and uses a target-critic supervised gate to selectively execute trusted decoded edits. Across simulation and realworld experiments, CNDO substantially outperforms existing noise-space methods and achieves overall sample efficiency comparable to the strong action-space baseline DICE. Building on this stronger noise-space learner, CANDO jointly optimizes pre-decoding noise directions and post-decoding action residuals at the complete edited action which achieves the highest average sample efficiency. We also figure out that access to both adaptation spaces is not sufficient by itself: joint residual training is most beneficial when the standalone noisespace and action-space learners have comparable capability.

## REFERENCES

[1] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[2] E. Chisari, N. Heppert, M. Argus, T. Welschehold, T. Brox, and A. Valada, “Learning robotic manipulation policies from point clouds with conditional flow matching,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270. PMLR, 2025, pp. 982–993.

[3] S. Ross, G. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in Proceedings of the fourteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2011, pp. 627–635.

[4] A. Ren, J. Lidard, L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz, “Diffusion policy policy optimization,” in International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, Eds., vol. 2025, 2025, pp. 77 288–77 329.

[5] A. Wagenmaker, Y. Zhang, M. Nakamoto, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine, “Steering your diffusion policy with latent space reinforcement learning,” in Proceedings of the 9th Conference on Robot Learning, vol. 305. PMLR, 2025, pp. 258–282.

[6] T. Johannink, S. Bahl, A. Nair, J. Luo, A. Kumar, M. Loskyll, J. A. Ojea, E. Solowjow, and S. Levine, “Residual reinforcement learning for robot control,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 6023–6029.

[7] Z. Sun and S. Song, “From prior to pro: Efficient skill mastery via distribution contractive RL finetuning,” in Forty-third International Conference on Machine Learning, 2026, pp. 1–17.

[8] H. Im, A. Kolobov, J. Fu, and Y. Lee, “Latent policy steering through one-step flow policies,” in Proceedings of Robotics: Science and Systems, 2026, pp. 1–14.

[9] H. Simsir and O. S. Oguz, “Lagrangian perturbation diffusion steering: Latent reinforcement learning for generative policies,” arXiv preprint arXiv:2606.01151, 2026.

[10] T. Silver, K. R. Allen, J. B. Tenenbaum, and L. P. Kaelbling, “Residual policy learning,” arXiv preprint arXiv:1812.06298, 2018.

[11] G. Schoettler, A. Nair, J. Luo, S. Bahl, J. A. Ojea, E. Solowjow, and S. Levine, “Deep reinforcement learning for industrial insertion tasks with visual inputs and natural rewards,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 5548–5555.

[12] C. Schaff and M. R. Walter, “Residual policy learning for shared autonomy,” in Proceedings of Robotics: Science and Systems, Jul. 2020, pp. 1–10.

[13] M. Alakuijala, G. Dulac-Arnold, J. Mairal, J. Ponce, and C. Schmid, “Residual reinforcement learning from demonstrations,” arXiv preprint arXiv:2106.08050, 2021.

[14] T. Davchev, K. S. Luck, M. Burke, F. Meier, S. Schaal, and S. Ramamoorthy, “Residual learning from demonstration: Adapting DMPs for contact-rich manipulation,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4488–4495, 2022.

[15] W. Zhou, S. Bajracharya, and D. Held, “PLAS: Latent action space for offline reinforcement learning,” in Proceedings of the 2020 Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Kober, F. Ramos, and C. Tomlin, Eds., vol. 155. PMLR, 2021, pp. 1719–1735.

[16] A. Ajay, A. Kumar, P. Agrawal, S. Levine, and O. Nachum, “OPAL: Offline primitive discovery for accelerating offline reinforcement learning,” in International Conference on Learning Representations, 2021, pp. 1–23.

[17] K. Pertsch, Y. Lee, and J. J. Lim, “Accelerating reinforcement learning with learned skill priors,” in Proceedings of the 2020 Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Kober, F. Ramos, and C. Tomlin, Eds., vol. 155. PMLR, 2021, pp. 188–204.

[18] X. Chen, A. Ghadirzadeh, T. Yu, J. Wang, A. Y. Gao, W. Li, L. Bin, C. Finn, and C. Zhang, “LAPO: Latent-variable advantage-weighted policy optimization for offline reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 36 902– 36 913.

[19] T. Yoneda, L. Sun, G. Yang, B. C. Stadie, and M. R. Walter, “To the noise and back: Diffusion for shared autonomy,” in Proceedings of Robotics: Science and Systems, 2023, pp. 1–18.

[20] L. Ankile, A. Simeonov, I. Shenfeld, M. Torne, and P. Agrawal, “From imitation to refinement: Residual RL for precise assembly,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 1–8.

[21] X. Yuan, T. Mu, S. Tao, Y. Fang, M. Zhang, and H. Su, “Policy decorator: Model-agnostic online refinement for large policy model,” in International Conference on Learning Representations, 2025, pp. 1–36.

[22] L. Ankile, Z. Jiang, R. Duan, G. Shi, P. Abbeel, and A. Nagabandi, “Residual off-policy RL for finetuning behavior cloning policies,” arXiv preprint arXiv:2509.19301, 2025.

[23] H. Hu, S. Mirchandani, and D. Sadigh, “Imitation bootstrapped reinforcement learning,” in Proceedings of Robotics: Science and Systems, 2024, pp. 1–18.

[24] G. Ma, L. Li, H. Wang, Z. Liu, P.-L. Bacon, and D. Tao, “What makes value learning efficient in residual reinforcement learning?” arXiv preprint arXiv:2602.10539, 2026.

[25] Z. Zhu, Z. Li, X. Yuan, H. Zhang, Y. Liu, C. Zhang, Y. Yu, W. Zhang, and M. Liu, “Unified latent steering and residual refinement for online improvement of diffusion policy models,” in ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation, 2026.

[26] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, “Efficient online reinforcement learning with offline data,” in International Conference on Machine Learning. PMLR, 2023, pp. 1577–1594.

[27] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín, “What matters in learning from offline human demonstrations for robot manipulation,” in 5th Annual Conference on Robot Learning. PMLR, 2021, pp. 1–13.

[28] T. Mu, Z. Ling, F. Xiang, D. C. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su, “Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, vol. 1, 2021, pp. 1–14.