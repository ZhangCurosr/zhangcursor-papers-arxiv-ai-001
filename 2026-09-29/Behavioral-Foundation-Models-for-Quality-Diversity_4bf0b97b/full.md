# Behavioral Foundation Models for Quality Diversity

Nazim Bendib   
Sorbonne Université, ISIR   
Paris, France   
bendib@isir.upmc.fr

Nicolas Perrin-Gilbert Sorbonne Université, ISIR Paris, France nicolas.perrin@isir.upmc.fr

Olivier Sigaud Sorbonne Université, ISIR Paris, France olivier.sigaud@isir.upmc.fr

## Abstract

Behavioral Foundation Models (BFMs) are an emerging paradigm in reinforcement learning, playing a role analogous to large language models in natural language processing: they have shown remarkable versatility, enabling zero-shot performance, fast imitation, and online adaptation, all by exploiting the structure of a latent space. In this work, we investigate whether the latent behavioral space induced by BFMs can serve as an effective search space to discover large repertoires of behaviorally diverse and high-performing policies through Quality-Diversity (QD) methods. While QD methods generally search directly in high-dimensional policy parameter space, in this paper, we present BFM-QD, a framework that performs QD search in the compact latent space of a BFM. We further show that the BFM-QD framework provides a closed-form, gradient-free policy improvement operator that approximates a policy gradient update, but requires no critic training and no backpropagation. Across continuous-control benchmarks spanning dense locomotion, sparse navigation, and contact-rich manipulation, BFM-QD consistently outperforms parameter-space baselines, with particularly stark gains in sparse and deceptive settings, where all tested parameter-space QD methods collapse to near-zero performance. These results show the effectiveness of the BFM-QD framework, benefiting from the synergy between dimensionality reduction of the search space and offline pretraining from diverse behavioral data. This positions BFMs as a general-purpose backbone for QD optimization, extending their utility beyond zero-shot task solving to the discovery of diverse behavioral repertoires.

## 1 Introduction

The history of machine learning is marked by a recent and decisive shift: rather than training models from scratch for each new task, the field has converged on the paradigm of large-scale pretraining followed by downstream reuse. In natural language processing, this inflection point arrived with large language models: pretrained representations so rich that a single backbone could be adapted prompted, or searched over to solve tasks that no individual fine-tuned model could reach alone. Reinforcement Learning (RL) appears to be approaching an analogous moment.

Two families are emerging as the RL analogue of this paradigm. On one hand, Vision-Language-Action models (VLAs) [25] benefit from large-scale pretraining on expert demonstrations mainly for text-conditioned robotic manipulation tasks. On the other hand, Behavioral Foundation Models (BFMs) [3, 35, 34] take a complementary approach. Trained once on large offline non-expert datasets of diverse agent behavior, a BFM encodes a diverse behavioral manifold into a compact latent space $\mathcal { Z } \subset \mathbb { R } ^ { d } .$ , such that varying a single continuous vector z is sufficient to instantiate a wide spectrum of motor skills, navigation strategies, and manipulation primitives, all without any additional training. This representational efficiency has enabled remarkable results in zero-shot task generalization [35, 36], fast imitation [30], and online adaptation [32], placing BFMs at the frontier of generalist agent design. Yet a striking limitation unites all existing work on BFMs: they all extract exactly one optimal behavior from the model. Given a task, the latent code z\* that maximizes expected return is inferred or optimized, and the resulting single policy is deployed. The vast behavioral structure embedded in its latent space, containing diverse, distinct, and potentially complementary behaviors, is underexploited. From this perspective, this paper asks a simple but consequential question: rather than extracting a single optimal behavior from a BFM, can we discover a large repertoire of diverse, high-performing behaviors from its latent space?

Quality-Diversity (QD) [26] is the natural framework to unlock this diversity: it discovers large repertoires of diverse, high-performing behaviors. Such repertoires matter because a single policy is optimal only under fixed assumptions. An archive enables, for example, damage recovery (selecting a policy that avoids a damaged leg without retraining), sim-to-real transfer (testing several candidates on hardware), and skill libraries for hierarchical control. Yet standard QD methods search directly in the policy parameter space $\Theta \subset \mathbb { R } ^ { D }$ , which is high-dimensional, unstructured, and almost entirely composed of incoherent behaviors (random perturbations of network weights almost never produce rewarding behaviors). This is not a limitation of the QD objective, but of the search space it is applied to. The key insight is that in a well-trained BFM, the parameter space Z is behaviorally rich: points $z \in { \mathcal { Z } }$ in the latent space decode to behaviors that cover a wide spectrum of task-relevant features. So, by design, Z is exactly the kind of search space QD needs.

We propose BFM-QD, a framework that exploits this synergy by relocating QD search from the policy parameter space to the latent space of a pretrained BFM, providing an ideal search space for QD without modifying its objective. Across a benchmark spanning dense locomotion (Walker, HalfCheetah), long-horizon sparse navigation (AntMaze-medium), and contact-rich manipulation (Cube-single), BFM-QD consistently outperforms all parameter-space baselines. On easy and dense tasks, the margin is modest, but convergence is faster. On hard and sparse tasks, the gap is stark: all parameter-space methods fail, while BFM-QD maintains strong behavioral coverage and high performance. Together, these results position BFMs not merely as tools for single-task policy extraction, but as general-purpose backbones for QD, extending their utility from zero-shot and few-shot task solving to the generation of large repertoires of efficient policies.

Contributions: In this paper:

• We present BFM-QD, the first framework to use the latent space of a BFM as the search space for QD optimization.

• We show that BFMs provide a closed-form, gradient-free policy improvement operator that matches a policy-gradient variation operator, requiring no critic training and no backpropagation.

• Through an extensive benchmark spanning dense locomotion, sparse navigation, and contact-rich manipulation, we show that BFM-QD consistently outperforms all parameter-space baselines, with stark gains in sparse and deceptive settings where all tested parameter-space QD methods fail.

• We characterize two key properties of the BFM latent space: behavioral richness, where random latent vectors decode to diverse efficient behaviors, and a structure where interpolation between solutions produces meaningful offspring.

• We show that QD performance saturates early in BFM pretraining, well before zero-shot performance converges, revealing BFM-QD does not require a fully trained BFM.

## 2 Related Work

## 2.1 Behavioral Foundation Models

Behavioral Foundation Models are policies trained on large offline datasets of diverse behaviors, providing a pretrained latent space that can be adapted to new tasks without additional training. Successor Features [10, 3, 6] provide an early instance of this paradigm, decomposing the value function into task-independent state features and task-specific reward weights, enabling zero-shot transfer across reward functions. Forward-Backward representations [35] extend this idea by jointly learning a forward map over successor measures and a backward map that projects reward functions into the latent space, enabling closed-form task inference and zero-shot policy retrieval. TD-JEPA [2] further advances this direction by leveraging latent-predictive representations that capture long-term, policy-conditioned dynamics from offline reward-free data. Beyond zero-shot task solving, BFMs have demonstrated broad utility as reusable behavioral backbones. They have been applied to fast imitation learning [30] using a small set of expert demonstrations, and to fast online adaptation [32], where the latent space enables rapid fine-tuning to new tasks with minimal environment interaction. More recently, BFMs have been scaled to humanoid robot control [34], demonstrating that the learned latent behavioral structure transfers to high-dimensional, contact-rich settings. Our work extends this line of research in a new direction: rather than extracting a single optimal behavior from the BFM, we use its latent space as the search space for quality-diversity optimization, discovering entire repertoires of diverse and high-performing behaviors.

## 2.2 Quality-Diversity algorithms

Quality-Diversity algorithms aim to discover large collections of diverse, high-performing solutions rather than a single optimum. MAP-Elites (ME) [26] is the canonical algorithm, maintaining an archive of elite solutions indexed by behavioral descriptors and evolving them through random mutations. CMA-ME [15] improves upon this by replacing Gaussian mutation with Covariance Matrix Adaptation (CMA-ES) [20], enabling more structured and efficient parameter-space search. However, these methods rely purely on undirected exploration and struggle in environments where random parameter perturbations rarely produce rewarding behaviors. To address this, policy gradient methods have been integrated into QD. PGA-ME [27] augments ME with TD3 [16] actor updates, using a learned critic to guide mutations toward higher-fitness regions. QDRL [24] similarly combines QD with deep RL, alternating between diversity-seeking and reward-maximizing updates. These approaches improve performance on dense-reward tasks but rely on a well-trained critic, which degrades under sparse or deceptive rewards. A complementary line of work augments the gradientbased variation operator with explicit descriptor conditioning. DCG-ME [12] enhances the policy gradient variation operator with a descriptor-conditioned critic, providing gradients that jointly optimize fitness and target descriptors. DCRL-ME [13] builds on this by additionally exploiting the descriptor-conditioned actor as a generative model, injecting diverse solutions directly into the offspring batch at each generation. Our work is complementary to these: rather than improving the descriptor or critic, we relocate the search space entirely to the latent space of a pretrained BFM. Another line of work searches learned latent spaces. Latent space illumination [14] runs QD in the latent space of a GAN to generate game levels; it is a content-generation task without policies or control. Policy Manifold Search [31] and DDE-Elites [17] encode archive solutions with an autoencoder over policy parameters, so the latent space is learned online from the archive. In contrast, we search the latent space of a policy-generating model that is pretrained offline without rewards.

## 3 Background

## 3.1 Behavioral Foundation Models

A Behavioral Foundation Model (BFM) is an agent pretrained on a reward-free offline dataset of environment transitions, such that it can produce optimal policies for a large class of reward functions specified at test time, without any additional training or planning. Most BFMs are built on top of the successor measure [5, 10] framework. The successor measure of a policy π at state-action $( s , a )$ is the discounted distribution of future states:

$$
M ^ { \pi } ( X \mid s , a ) : = \sum _ { t \geq 0 } \gamma ^ { t } \operatorname* { P r } ( s _ { t } \in X \mid s _ { 0 } = s , a _ { 0 } = a , \pi ) , \quad \forall X \subset { \mathcal { S } } .\tag{1}
$$

Successor measures disentangle environment dynamics from the reward function: for any reward r and policy π, the Q-function decomposes as $Q _ { r } ^ { \pi } ( \bar { s } , a ) = M ^ { \pi } r ( s , a )$ . Given a feature map $\phi : { \mathcal { S } }  \mathbb { R } ^ { d }$ the corresponding successor features are $\begin{array} { r } { \dot { \psi } ^ { \pi } ( s , a ) = \sum _ { t > 0 } \dot { \gamma } ^ { t } \mathbb { E } [ \phi ( s _ { t + 1 } ) \ | \ s , a , \pi ] } \end{array}$ , which allow expressing the Q-function compactly as $Q _ { r } ^ { \pi } ( s , a ) = \psi ^ { \pi } ( s , a ) ^ { \top } z$ for any reward $r _ { z } ( s ) = \phi ( s ) ^ { \top } z$ BFMs instantiate this framework by learning a family of policies $\{ \pi _ { z } \} _ { z \in \mathcal { Z } }$ parameterized by a latent code $z \in \mathcal { Z } \subset \mathbb { R } ^ { d }$ , alongside their successor features $\psi ( s , a , z )$ . At test time, given a reward function r, the optimal latent code is inferred by projecting r onto the learned features:

![](images/3c13bf0fc6269f0375b2f683ea403ea19b3f25d2ec2b5430f4e0fc22e518bee8.jpg)  
Figure 1: Overview of BFM-QD. A BFM is pretrained once on an offline dataset and then frozen. The loop is one QD generation: latent codes z are sampled from the archive, mutated into $z ^ { \prime } .$ , and evaluated by rolling out $\pi _ { \mathrm { B F M } } ( \cdot | z ^ { \prime } )$ . Offspring update the archive.

$$
z _ { r } = \underset { z \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } \mathbb { E } _ { s \sim \rho } \big [ ( r ( s ) - \phi ( s ) ^ { \top } z ) ^ { 2 } \big ] = \mathbb { E } _ { s \sim \rho } [ \phi ( s ) \phi ( s ) ^ { \top } ] ^ { - 1 } \mathbb { E } _ { s \sim \rho } [ \phi ( s ) r ( s ) ] ,\tag{2}
$$

and the policy $\pi _ { z _ { r } }$ is deployed zero-shot. In practice $z _ { r }$ is projected to a hypersphere of radius $\sqrt { d _ { z } }$ with $d _ { z } = \dim ( { \mathcal { Z } } )$

Different BFMs differ in how they learn φ. Forward-Backward (FB) representations [35] jointly learn φ and ψ through a successor measure consistency loss, enabling closed-form task inference at test time. TD-JEPA [2] instead learns $\phi$ and ψ through a temporal-difference latent-predictive loss, training a policy-conditioned predictor that approximates successor features directly in latent space. Other BFMs are presented in Appendix B. In all cases, the result is a single pretrained model πBFm(.|z) whose latent space Z encodes a wide range of behaviors, all accessible by varying z.

## 3.2 Quality-Diversity algorithms

Quality-Diversity (QD) algorithms aim to discover a large collection of diverse, high-performing solutions rather than converging to a single optimum. A QD problem is defined by a fitness function $r : \mathcal { S }  \mathbb { R }$ , measuring how good a solution is, and a behavioral descriptor ${ \dot { \beta } } : { \mathcal { S } }  B ,$ a lowdimensional summary of how a solution behaves (e.g., the foot contact pattern of a walking robot). The descriptor space B is discretized into a finite grid of cells, and the goal is to find the best possible solution for every cell, that is, to fill the grid with a diverse repertoire of high-performing policies, each with a distinct behavior. Solutions are maintained in an archive C storing at most one elite per cell; a candidate replaces the existing elite only if it achieves higher fitness in the same cell.

ME [26] is the canonical QD algorithm: parents are sampled from the archive, perturbed by a variation operator, evaluated, and inserted if they improve their cell. The typical variation operator is a Gaussian mutation, but it can be replaced with more sophisticated methods, such as covariance-based adaptation [15] or policy gradient updates that leverage a learned critic to guide mutations toward higher-fitness regions [27, 24, 12, 13]. See Appendix C for more details. Each generation thus repeats four steps: select parents from the archive, vary them, evaluate fitness and descriptor with a rollout, and insert offspring into their cells (Algorithm 1).

The performance of QD methods is measured by three metrics: coverage, the fraction of cells populated in the archive; the QD-score, the sum of fitness values across all filled cells, which jointly captures whether the repertoire is both diverse and high-performing; and maximum fitness, the fitness of the single best solution found, measuring peak performance regardless of diversity.

## 4 Method

Standard QD methods parameterize policies as neural networks $\theta \in \Theta$ and mutate directly in Θ. We instead leverage a pretrained BFM, specifically its pretrained actor πBFM : $s \times \mathcal { Z } \to A$ , which produces diverse behaviors by just varying $z \in { \mathcal { Z } }$ . The BFM is trained once on an offline dataset and then kept frozen throughout all QD runs.

## 4.1 Latent Space Quality-Diversity

The central observation of our work is that Z is a far more efficient search space for QD than Θ: it is low-dimensional $( d _ { z } \ll \vert \theta \vert )$ and behaviorally rich. We therefore propose to run QD directly in Z. The archive stores latent codes z rather than policy weights θ, and all variation operators act on z.

Each candidate is evaluated by rolling out $\pi _ { \mathrm { B F M } } ( \cdot \mid z )$ in the environment and computing its fitness and behavioral descriptor. The QD loop is otherwise standard: solutions compete for cells in the archive based on fitness, and the archive is used to seed the next generation of mutations. Since the BFM is trained with a normalization constraint, we re-project every mutated candidate back onto the sphere after perturbation. See Figure 1 for an overview. See Algorithm 2 for pseudocode.

This constitutes our framework, which we refer to as BFM-QD. A key property of the BFM-QD framework is that it is modular by design, meaning any BFM can be paired with any QD optimizer, yielding variants such as FB-ME, FB-CMA-ME, TD-JEPA-ME, or TD-JEPA-CMA-ME.

## 4.2 Closed-Form Policy Improvement via Backward Inference

Standard QD methods rely on gradient-free exploration, which is sample-inefficient. Recent work has addressed this by integrating policy gradient methods into QD [27, 13]: a critic is trained on the fitness function, and policies are updated to maximize it, improving solution quality while the archive maintains diversity. However, this requires training and maintaining a critic online, and doing multiple backpropagations to update the solutions.

BFMs enable sidestepping this limitation. A key advantage of BFMs is their zero-shot capabilities: Given a fitness function and a set of evaluated transitions, the latent code $z ^ { * }$ corresponding to the optimal policy for the observed reward function can be estimated in closed form following Eq 2. This inference requires no gradient computation, no critic training, and no additional environment interactions beyond those already collected during archive evaluation. We use this inference, which we call the Backward Inference (BI) operator, as a directed variation operator. For each solution z selected from the archive, we then produce an offspring by interpolating toward this global $z ^ { * }$

$$
z ^ { \prime } \gets ( 1 - \alpha ) \cdot z + \alpha \cdot z ^ { * } ,\tag{3}
$$

where $\alpha \in ] 0 , 1 [$ controls the step size toward the inferred optimum. $z ^ { \prime }$ is then normalized and projected back onto the BFM latent sphere. Note that we do not set $z ^ { \prime } \gets z ^ { * }$ directly: doing so for every solution would collapse the entire archive toward a single behavior, destroying the diversity that QD is designed to maintain. Instead, we push each solution individually toward higher fitness while preserving its distinctiveness. At each generation, we (i) sample state-reward pairs $( s , r ( s ) )$ from the replay buffer, which stores all past evaluation trajectories; (ii) compute $z ^ { * }$ with Eq. 2; and (iii) for each parent z selected from the archive, produce an offspring with Eq. 3 and project it onto the sphere. Following PGA-ME, half of the offspring are produced by BI $( \alpha = 0 . 0 2 )$ and half by Gaussian mutation (σ = 1). A pseudo-algorithm is presented in Algorithm 3.

Interestingly, this interpolation is a Newton step on a local quadratic surrogate of the policy improvement objective, moving z toward $z ^ { * }$ (Appendix A). The step itself is generic; the BFM makes it practical by providing $z ^ { * }$ in closed form, with no critic training and no backpropagation. The approximation error is proportional to the reward projection residual.

## 5 Experiments

We evaluate BFM-QD on a suite of continuous control tasks spanning a spectrum of reward densities, from dense locomotion to sparse navigation and contact-rich manipulation. Our experiments are designed to answer four questions: (1) Does BFM-QD outperform standard QD baselines across task types and reward densities? (2) What properties of the FB latent space drive BFM-QD's performance? (3) Can Backward Inference in BFMs replace policy gradient updates? (4) Does BFM-QD require a large pretraining budget to be effective?

## 5.1 Experimental Setup

Environments. We evaluate on 18 tasks spanning four environments and a spectrum of difficulty levels. Walker2D\_uni and HalfCheetah\_uni are standard continuous control benchmarks from QDax [8], representing dense, simple locomotion, with pretraining data collected via Random Network Distillation (RND) [7]. AntMaze-medium from OGBench [28] uses the antmaze-medium-explore dataset collected by random policies and introduces long-horizon navigation with both dense and sparse rewards. Cube-single from OGBench uses the cube-single-noisy dataset, consisting of noisy trajectories, and introduces manipulation under sparse rewards (hardest setting). Full details of environments, behavioral descriptors, and fitness functions are provided in Appendix D.1.

![](images/ff939d068d73d84bd3e710af1909c495a554d4dbdd84a7ca533405bd3e8f0bc7.jpg)  
Figure 2: Normalized QD-score, coverage, and normalized max fitness across all environments. BFM-QD variants consistently match or outperform all baselines, especially on harder tasks.

Baselines. We compare BFM-QD against five parameter-space baselines. ME [26] is the standard QD algorithm, operating in MLP parameter space with Gaussian mutation. CMA-ME [15] augments ME with CMA-ES. DCRL-ME [13] combines ME with TD3 policy gradient using a descriptorconditioned policy. DDE-Elites [17] iteratively trains a Variational Auto-Encoder (VAE) [29] on the archive solutions to learn a compact latent space for more sample-efficient search. This baseline isolates the effect of searching in a low-dimensional space. ME+Pretrain initializes MLP policies via offline TD3 trained on the same dataset as the BFM, using the task fitness function as a reward. This baseline isolates the effect of pretraining on diverse data. All baselines use the same MLP architecture for a fair comparison. For more details, see Appendix D.2.

BFM-QD Variants. We instantiate three BFM-QD variants: FB-ME pairs FB with ME using Gaussian mutation; FB-CMA-ME pairs FB with CMA-ME; TD-JEPA-CMA-ME pairs TD-JEPA with CMA-ME. Other variants of BFM-QD are presented in Appendix E.4.

Implementation Details. The FB model is pretrained once per environment on the offline dataset and kept frozen throughout all QD runs (See Appendix E.11 for reported compute time). The latent space has dimension $\bar { d } _ { z } = 5 0$ . The archive is a 50 × 50 grid. Each run consists of 500 generations of 400 parallel rollouts, for 200k evaluations in total. A rollout is one full episode of 500 steps (Walker, HalfCheetah) or 1000 steps (AntMaze, Cube-single), so a run collects 100M or 200M transitions, respectively (200k or 400k per generation). Since each environment contains multiple tasks with different reward scales, we normalize QD-scores using min-max normalization before aggregating across tasks (see Appendix D.1 for details). For parameter-based methods, we use MLPs of the shape (state\_dim, 128, 128, action\_dim). All results are averaged over 5 random seeds unless stated otherwise. Standard deviations are reported.

## 5.2 Main Results

Figure 2 reports the QD-score, coverage, and maximum fitness across all methods and environments. BFM-QD variants consistently achieve the highest QD-score across all tasks. On dense locomotion tasks (Walker, HalfCheetah), the margin over parameter-space baselines is modest, but BFM-QD variants converge faster, confirming that the behavioral prior accelerates search even when the task is easy. Notably, BFM-QD variants achieve this despite operating in a far lower-dimensional space $( d _ { z } = 5 0$ versus \~20k parameters), demonstrating that the BFM latent space is sufficiently expressive to reproduce near-optimal behaviors across a wide range of fitness functions.

![](images/d5eed9e5d93eefb05a80a18fedfe27f43e900f3602878805d9aa912ad03720f6.jpg)

![](images/e3879402dfaa7a40aa5aa4756eff97743546b0c60732f49de23d38e958736741.jpg)  
(a) Random sampling. Coverage and QD- (b) Interpolation operator. QD-score for ME and score of 10,000 random FB latent samples vs. CMA-ME, with and without the isoline operator, in parandom MLP samples. rameter space and FB latent space.  
Figure 3: The FB latent space is expressive and behaviorally rich, and supports interpolation between behaviors. (a) Random FB latent codes yield higher coverage and QD-score compared to random MLP policies. (b) Interpolation between solutions is only beneficial in the FB latent space, not in parameter space.

The advantage of BFM-QD grows substantially as task difficulty increases. On AntMaze-medium and Cube-single, all parameter-space baselines collapse to near-zero QD-score and low coverage, while BFM-QD variants maintain high performance across all metrics. Crucially, BFM-QD variants also achieve substantially higher maximum fitness on hard tasks. Two additional comparisons sharpen this finding. DDE-Elites, which also searches in a compact latent space, fails to match the BFM-QD performance, demonstrating that dimensionality reduction alone is not sufficient: what matters is the behavioral richness of the latent space, not its dimensionality. Similarly, ME+Pretrain, which benefits from offline pretraining, also fails, demonstrating that the benefit of BFM-QD is not simply a matter of pretraining on diverse data, but of searching in the structured latent space the BFM provides. Two further baselines that combine offline pretraining with latent-space search (DDE-Elites+Pretrain with TD3 or BC policies) and Policy Manifold Search also fall short of FB-ME (Appendix E.7).

Takeaway: Searching in the BFM latent space consistently outperforms searching in the parameter space, with stark gains in sparse and deceptive settings. The gains do not come from dimensionality reduction or pretraining alone, but from the compact behavioral representation of the BFM.

## 5.3 Expressivity of the latent space

We investigate two properties of the FB latent space that underpin the success of BFM-QD: the expressivity of the learned latent space and its smooth structure, where linear interpolations between solutions produce behaviorally intermediate policies.

## 5.3.1 The latent space is behaviorally rich by construction

To isolate the contribution of the behavioral prior from the QD optimizer, we remove all optimization entirely and evaluate the quality of random sampling alone.

Experiment We sample N = 10,000 random latent vectors from (1) a trained FB latent space, (2) the MLP parameter space, and (3) an untrained FB latent space (same architecture, random weights) to further isolate the contribution of pretraining from that of architecture. We report the resulting coverage and QD-score for each environment. No optimization is performed in any condition.

Results Results are shown in Figure 3a. Even without optimization, random FB samples already achieve higher coverage and QD-score than random MLP samples across all environments, with the gap growing monotonically with task difficulty (AntMaze-medium and Cube-single), where random MLP samples produce near-zero QD-score (see Appendix E.3 for archive visualization). Crucially, the untrained FB model performs significantly worse than the trained FB, confirming that this behavioral richness stems specifically from offline pretraining, not from the architecture or the geometry of the latent sphere.

![](images/38188f90b277db8432823fbc1d75769feb64594e839fc75a134620fab2bac7c3.jpg)  
Figure 4: The BFM Backward Inference (BI) matches a policy gradient update. FB-BI (backward inference) matches FB-PGA-ME across all environments, while PGA-ME fails on hard tasks.

Takeaway: Through pretraining, the FB latent space naturally produces behaviors that cover a wide spectrum of task-relevant features.

## 5.3.2 The latent space structure makes interpolation meaningful

A key property of the BFM latent space is its geometric smoothness: nearby latent codes induce behaviorally similar policies. We investigate whether this smoothness is sufficient to make interpolation between archive solutions semantically meaningful, enabling a class of efficient crossover-based variation operators. By contrast, such operators can be destructive in parameter space as the interpolant typically collapses to an incoherent solution [11]. We introduce an isoline operator (Iso) that, given two parent solutions $x _ { 1 } , x _ { 2 }$ from the archive, produces an offspring by linear interpolation:

$$
x _ { \mathrm { o f f s p r i n g } } = \left( 1 - \alpha \right) x _ { 1 } + \alpha x _ { 2 } , \quad \alpha \sim \mathcal { U } [ 0 , 1 ] .
$$

A pseudo-algorithm is presented in Algorithm 4.

Experiment We compare ME and CMA-ME with and without the interpolation operator, both in parameter space (ME-Iso, CMA-ME-Iso) and the FB latent space (FB-ME-Iso, FB-CMA-ME-Iso), isolating whether the gain comes from the operator itself or the smoothness of the latent space.

Results Results are shown in Figure 3b. Adding the isoline operator to parameter-space methods (ME-Iso, CMA-ME-Iso) yields no consistent improvement, confirming that interpolation in parameter space can be destructive. In contrast, FB-ME-Iso and FB-CMA-ME-Iso consistently improve over their non-Iso counterparts, especially on AntMaze-medium and Cube-single. This extra gain can be attributed to the structure of the latent space: interpolated codes produce behaviorally intermediate policies, enabling a better exploration of the descriptor space that random mutation cannot replicate.

Takeaway: Interpolation between solutions is only beneficial in the BFM latent space due to its structure and smoothness, unlike in the neural-network parameter space.

## 5.4 Can Backward Inference in BFMs replace policy gradient updates?

A natural question is whether gradient-based variation operators can further improve BFM-QD. PGA-ME [27] augments ME with TD3 policy gradient updates, using a learned critic to guide mutations toward higher-fitness regions. However, BFMs and typically FB provide a closed-form estimate of the task vector that best explains an observed reward signal (Equation 2), requiring no critic training and no additional environment interactions. We investigate whether this zero-shot inference approach can substitute for a full policy gradient update.

Experiment We compare three methods: PGA-ME, which runs TD3 policy gradient updates in parameter space; FB-PGA-ME, which runs the same gradient updates but in the FB latent space; and FB-BI, which replaces gradient updates entirely with the Backward Inference (BI) operator (Equation 3). See Appendix D.2 for more details.

![](images/7d03f5d533445ad8ec0f7007422fadb48d37c3b7b13f37da21d5322a30e743b4.jpg)

![](images/aa3c2a8c7bb002575f25c06c49f126a78dc3aefd0cb7c9c5366e6fecea7af973.jpg)  
Figure 5: BFM-QD does not require a fully trained BFM. QD-score (blue) and zero-shot performance (red) as a function of FB training budget on AntMaze-medium and Cube-single

Results Results are shown in Figure 4. PGA-ME matches FB-PGA-ME and FB-BI methods on locomotion tasks but drastically fails on AntMaze-medium and Cube-single, where the reward signal is too deceptive and sparse to train an accurate critic. FB-BI matches FB-PGA-ME across all environments, achieving the same QD-score without any critic training. This shows that closed-form backward inference can substitute for a learned critic.

Takeaway: The BFM Backward Inference (BI) operator matches a policy-gradient variation operator, without critic training or backpropagation.

## 5.5 BFM-QD does not require a large pretraining budget

We investigate whether BFM-QD requires a large BFM pretraining budget to be effective for QD search. A priori, one might expect that QD performance follows zero-shot performance and thus requires a fully trained BFM. We test whether this holds by measuring QD performance and zero-shot performance at different stages of BFM training.

Experiment We train FB and extract checkpoints every 40k steps. At each checkpoint, we run FB-ME and report the resulting QD-score and the zero-shot performance of the checkpoint on the same task. We do this for (energy efficiency |ant xy) from AntMaze-medium and (energy efficiency |cube xz) from Cube-single.

Results Results are shown in Figure 5. QD performance rises sharply in the first 40k training steps and plateaus early, well before zero-shot performance converges. In both environments, a partially trained BFM already enables strong QD performance, while zero-shot performance continues to improve long after the QD-score has saturated. This decoupling confirms that the latent space acquires sufficient behavioral structure for QD search early in training, even before it can reliably solve individual tasks zero-shot.

We distinguish two properties of the BFM latent space: behavioral diversity, the repertoire of distinct behaviors it contains, and value precision, the accuracy of (φ, ψ) for computing $z _ { r }$ in closed form (Equation 2). Zero-shot solving requires both: diversity provides good candidates, and precision selects the best one. QD search requires only diversity. We attribute the decoupling in Figure 5 to the two properties emerging at different times: diversity is acquired early in pretraining, while value precision keeps improving later.

Takeaway: A partially trained BFM already provides a latent space rich enough for strong QD performance, amortizing pretraining cost further.

## 5.6 Additional Analyses

Beyond the main experiments, we conduct seven additional analyses in Appendix $\mathrm { E , }$ whose findings we summarize here:

1. The choice of a BFM matters at scale (E.4): Across five BFMs (Laplacian [37], BYOLγ [22], FB, BTD-FB [4], and TD-JEPA), zero-shot performance is a reliable proxy for QD performance; the gap between strong and weak BFMs widens sharply on hard tasks, making BFM quality a critical design decision in sparse settings, while weaker BFMs can still perform well on easy tasks.

2. Diversity aids optimization, not just coverage (E.5): FB-CMA-ME outperforms a singleobjective CMA-ES [19] in the same latent space even on maximum fitness, demonstrating that maintaining a diverse population acts as an implicit exploration bonus that prevents premature convergence to local optima in the latent space.

3. Exploration data dominates dataset quality (E.6): Random exploration datasets consistently outperform expert datasets for BFM pretraining, even when spatially restricted, because limited coverage in the behavioral space in the offline data results in bounded latent space diversity

4. Pretraining and latent-space search are not sufficient (E.7): DDE-Elites initialized with policies pretrained offline on the BFM dataset (TD3 or behavioral cloning) and Policy Manifold Search do not match FB-ME, showing that the gains come from the structure of the BFM latent space and not from combining pretraining with a learned latent space.

5. Backward Inference outperforms sampling around z\* (E.8): Sampling offspring around the inferred optimum confines the search to a small region of the latent space, whereas BI moves each parent individually toward z\* and preserves diversity.

6. QD-collected data is a worse pretraining source than RND (E.9): A BFM pretrained on data collected by a QD search performs worse than one pretrained on RND data, because QD refines solutions near existing elites instead of covering new transitions.

7. Backward Inference is robust to its step size (E.10): FB-BI is stable for $\alpha \in [ 0 . 0 0 5 , 0 . 1 ]$ and degrades for larger steps, which pull offspring toward the single point z\* and collapse diversity.

## 6 Limitations

BFM-QD requires a behaviorally diverse offline dataset and a sufficiently pretrained BFM before any QD search can begin, though Section 5.5 shows that full convergence of the BFM is not necessary. Besides, a single BFM can be reused across all subsequent tasks and QD runs in the same environment, amortizing this cost (see Appendix E.11 for training time). Furthermore, performance degrades with narrower datasets (Appendix E.6) and with data collected by QD instead of RND (Appendix E.9), as the diversity of the latent space is bounded by the diversity of the training data. A deeper limitation is that the frozen BFM imposes a hard ceiling on behavioral expressivity: novel behaviors outside the pretrained latent space remain inaccessible. Finetuning the BFM during the QD run is a promising direction we leave to future work. Moreover, each BFM is trained for a single environment, so environment-agnostic QD remains open. Finally, as in standard QD, we assume the reward and behavioral descriptor are given; BFM-QD can only produce behaviors captured by the learned features φ(s), and we do not measure how far the discovered behaviors lie outside the pretraining data.

## 7 Conclusion

We proposed BFM-QD, a framework that replaces parameter-space exploration in QD search with exploration in the latent space of a pretrained BFM. BFM-QD sidesteps the fundamental bottleneck of standard QD methods by searching in a compact, behaviorally dense latent space, consistently outperforming parameter-space baselines across dense locomotion, sparse navigation, and contactrich manipulation, with stark gains in sparse and deceptive settings where all tested parameter-space methods collapse. These results establish BFM latent space exploration as a principled foundation for the next generation of QD algorithms, and position BFMs as general-purpose backbones for diversity-seeking optimization beyond their original use case of single-task policy extraction.

## Acknowledgments

Experiments presented in this paper were carried out using the HPC resources of IDRIS under the allocation 2025-[AD011016374] made by GENCI. This work was supported by the Sorbonne Center for Artificial Intelligence (SCAI)

Competing interests. We declare no competing interests.

## References

[1] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

[2] Marco Bagatella, Matteo Pirotta, Ahmed Touati, Alessandro Lazaric, and Andrea Tirinzoni. TD-JEPA: Latent-predictive representations for zero-shot reinforcement learning. arXiv preprint arXiv:2510.00739, 2025.

[3] André Barreto, Will Dabney, Rémi Munos, Jonathan J Hunt, Tom Schaul, Hado P Van Hasselt, and David Silver. Successor features for transfer in reinforcement learning. Advances in neural information processing systems, 30, 2017.

[4] Nazim Bendib, Nicolas Perrin-Gilbert, and Olivier Sigaud. Improving zero-shot offline RL via behavioral task sampling. arXiv preprint arXiv:2604.25496, 2026.

[5] Léonard Blier, Corentin Tallec, and Yann Ollivier. Learning successor states and goal-dependent values: A mathematical viewpoint. arXiv preprint arXiv:2101.07123, 2021.

[6] Diana Borsa, André Barreto, John Quan, Daniel Mankowitz, Rémi Munos, Hado Van Hasselt, David Silver, and Tom Schaul. Universal successor features approximators. arXiv preprint arXiv:1812.07626, 2018.

[7] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.

[8] Felix Chalumeau, Bryan Lim, Raphael Boige, Maxime Allard, Luca Grillotti, Manon Flageat, Valentin Macé, Guillaume Richard, Arthur Flajolet, Thomas Pierrot, et al. QDAX: A library for quality-diversity and population-based algorithms with hardware acceleration. Journal of Machine Learning Research, 25(108):1–16, 2024.

[9] Shuangshuang Chen and Wei Guo. Auto-encoders in deep learning: A review with new perspectives. Mathematics, 11(8):1777, 2023.

[10] Peter Dayan. Improving generalization for temporal difference learning: The successor representation. Neural computation, 5(4):613–624, 1993.

[11] Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. arXiv preprint arXiv:2110.06296, 2021.

[12] Maxence Faldor, Félix Chalumeau, Manon Flageat, and Antoine Cully. Map-elites with descriptor-conditioned gradients and archive distillation into a single policy. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 138–146, 2023.

[13] Maxence Faldor, Félix Chalumeau, Manon Flageat, and Antoine Cully. Synergizing quality-diversity with descriptor-conditioned reinforcement learning. ACM Transactions on Evolutionary Learning, 5(1):1–35, 2025.

[14] Matthew C Fontaine, Ruilin Liu, Ahmed Khalifa, Jignesh Modi, Julian Togelius, Amy K Hoover, and Stefanos Nikolaidis. Illuminating mario scenes in the latent space of a generative adversarial network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 5922–5930, 2021.

[15] Matthew C Fontaine, Julian Togelius, Stefanos Nikolaidis, and Amy K Hoover. Covariance matrix adaptation for the rapid illumination of behavior space. In Proceedings of the 2020 genetic and evolutionary computation conference, pages 94–102, 2020.

[16] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pages 1587–1596. PMLR, 2018.

[17] Adam Gaier, Alexander Asteroth, and Jean-Baptiste Mouret. Discovering representations for black-box optimization. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, pages 103–111, 2020.

[18] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

[19] Nikolaus Hansen. The CMA evolution strategy: A tutorial. arXiv preprint arXiv: 1604.00772, 2016.

[20] Nikolaus Hansen, Sibylle D Müller, and Petros Koumoutsakos. Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES). Evolutionary computation, 11(1):1–18, 2003.

[21] Michael Laskin, Denis Yarats, Hao Liu, Kimin Lee, Albert Zhan, Kevin Lu, Catherine Cang, Lerrel Pinto, and Pieter Abbeel. URLB: Unsupervised reinforcement learning benchmark. arXiv preprint arXiv:2110.15191, 2021.

[22] Daniel Lawson, Adriana Hugessen, Charlotte Cloutier, Glen Berseth, and Khimya Khetarpal. Self-predictive representations for combinatorial generalization in behavioral cloning. arXiv preprint arXiv:2506.10137, 2025.

[23] Yann LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.

[24] Bryan Lim, Manon Flageat, and Antoine Cully. Understanding the synergies between qualitydiversity and deep reinforcement learning. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 1212–1220, 2023.

[25] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on visionlanguage-action models for embodied ai. arXiv preprint arXiv:2405.14093, 2024.

[26] Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909, 2015.

[27] Olle Nilsson and Antoine Cully. Policy gradient assisted map-elites. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 866–875, 2021.

[28] Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goal-conditioned RL. arXiv preprint arXiv:2410.20092, 2024.

[29] Lucas Pinheiro Cinelli, Matheus Araújo Marins, Eduardo Antúnio Barros da Silva, and Sérgio Lima Netto. Variational autoencoder. In Variational methods for machine learning with applications to deep networks, pages 111–149. Springer, 2021.

[30] Matteo Pirotta, Andrea Tirinzoni, Ahmed Touati, Alessandro Lazaric, and Yann Ollivier. Fast imitation via behavior foundation models. In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023.

[31] Nemanja Rakicevic, Antoine Cully, and Petar Kormushev. Policy manifold search: Exploring the manifold hypothesis for diversity-based neuroevolution. In Genetic and Evolutionary Computation Conference, 2021.

[32] Harshit Sikchi, Andrea Tirinzoni, Ahmed Touati, Yingchen Xu, Anssi Kanervisto, Scott Niekum, Amy Zhang, Alessandro Lazaric, and Matteo Pirotta. Fast adaptation with behavioral foundation models. arXiv preprint arXiv:2504.07896, 2025.

[33] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

[34] Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto, Yingchen Xu, Alessandro Lazaric, and Matteo Pirotta. Zero-shot whole-body humanoid control via behavioral foundation models. arXiv preprint arXiv:2504.11054, 2025.

[35] Ahmed Touati and Yann Ollivier. Learning one representation to optimize all rewards. Advances in Neural Information Processing Systems, 34:13–23, 2021.

[36] Ahmed Touati, Jérémy Rapin, and Yann Ollivier. Does zero-shot reinforcement learning exist? arXiv preprint arXiv:2209.14935, 2022.

[37] Yifan Wu, George Tucker, and Ofir Nachum. The laplacian in RL: Learning representations with efficient approximations. arXiv preprint arXiv:1810.04586, 2018.

A Proof of the Backward Inference Operator 15   
B Background on Behavioral Foundation Models 17   
B.1 Successor Features 17   
B.2 Forward-Backward Representations 18   
B.3 TD-JEPA 19   
B.4 BTD-FB 19   
C Background on Quality-Diversity Methods 20   
C.1 MAP-Elites 20   
C.2 CMA-ME 20   
C.3 PGA-ME 20   
C.4 DCRL-ME 21   
C.5 DDE-Elites 21   
C.6 ME+Pretrain 21   
D Experimental Details 22   
D.1 Environments and Tasks 22   
D.2 Baselines Implementation Details 23   
D.3 BFM Implementation Details . 26   
E Additional Results 31   
E.1 Expanded main results 31   
E.2 Archive Visualization 32   
E.3 The latent space is behaviorally rich by construction . 32   
E.4 Effect of the Behavioral Foundation Model 33   
E.5 Diversity as an Optimization Strategy 33   
E.6 Sensitivity to the Offline Dataset Quality 34   
E.7 Additional Latent-Space Search Baselines 35   
E.8 Backward Inference vs. Sampling Around the Inferred Optimum 36   
E.9 Pretraining Data Collected by QD 36   
E.10 Sensitivity to the Backward Inference Step Size 36   
E.11 Compute Resources 37   
E.12 Results Table 38

## A Proof of the Backward Inference Operator

In the following, we denote $\pi _ { z } ( \cdot ) : = \pi _ { \mathrm { B F M } } ( \cdot \mid z )$ the policy instantiated by the BFM at latent code $z \in { \mathcal { Z } }$ .Consider the policy improvement objective:

$$
J ( z ) = \mathbb { E } _ { s \sim \rho } \bigl [ { - Q ^ { \pi _ { z } } ( s , \pi _ { z } ( s ) ) } \bigr ] ,\tag{4}
$$

where $\begin{array} { r } { Q ^ { \pi _ { z } } ( s , a ) = \mathbb { E } \bigl [ \sum _ { t > 0 } \gamma ^ { t } r ( s _ { t } ) \mid s _ { 0 } = s , a _ { 0 } = a , \pi _ { z } \bigr ] } \end{array}$ is the action-value function of policy $\pi _ { z }$ under reward r, and $\rho$ is thē mixture state distribution induced by the archive policies (equivalent to a replay buffer). A standard first-order policy gradient step on J takes the form:

$$
z \gets z - \alpha \nabla _ { z } J ( z ) .\tag{5}
$$

We will show that the BI operator (Equation 3) actually approximates a second-order Newton step of J, which subsumes this update.

By construction of $z ^ { * }$ (Equation 2), the task vector $z ^ { * }$ is the least-squares projection of r onto the feature space of φ:

$$
\begin{array} { r } { z ^ { * } = \underset { z \in \mathbb { R } ^ { d _ { z } } } { \arg \operatorname* { m i n } } ~ \mathbb { E } _ { \rho } \big [ ( r ( s ) - \phi ( s ) ^ { \top } z ) ^ { 2 } \big ] , } \end{array}\tag{6}
$$

so that $r ( s ) \approx \phi ( s ) ^ { \top } z ^ { * }$ for all $s ^ { 1 }$ . Substituting into the definition of $Q ^ { \pi _ { z } }$

$$
Q ^ { \pi _ { z } } ( s , a ) = \mathbb { E } \left[ \sum _ { t \geq 0 } \gamma ^ { t } r ( s _ { t } ) \Bigm | s , a , \pi _ { z } \right]\tag{7}
$$

$$
\approx \mathbb { E } \left[ \sum _ { t \geq 0 } \gamma ^ { t } \phi ( s _ { t } ) ^ { \top } z ^ { * } \Bigm | s , a , \pi _ { z } \right]\tag{8}
$$

$$
= \underbrace { \mathbb { E } \left[ \sum _ { t \geq 0 } \gamma ^ { t } \phi ( s _ { t } ) \Bigm | s , a , \pi _ { z } \right] ^ { \top } z ^ { * } \ = \ \psi ^ { \pi _ { z } } ( s , a ) ^ { \top } z ^ { * } \ = \ Q _ { z ^ { * } } ^ { \pi _ { z } } ( s , a ) , } _ { \psi ^ { \pi _ { z } } ( s , a ) }\tag{9}
$$

where $\psi ^ { \pi _ { z } } ( s , a )$ are the successor features of $\pi _ { z } \left[ 3 \right]$ . The policy gradient objective therefore becomes:

$$
J ( z ) \approx \mathbb { E } _ { s \sim \rho } \big [ { - Q _ { z ^ { * } } ^ { \pi _ { z } } ( s , \pi _ { z } ( s ) ) } \big ] = \mathbb { E } _ { s \sim \rho } \big [ { - \psi ^ { \pi _ { z } } ( s , \pi _ { z } ( s ) ) ^ { \top } z ^ { * } } \big ] .\tag{10}
$$

Denote this surrogate objective $\tilde { J } ( z ) : = \mathbb { E } _ { s \sim \rho } \bigl \lceil - \psi ^ { \pi _ { z } } ( s , \pi _ { z } ( s ) ) ^ { \top } z ^ { * } \bigr \rceil$ , so that $J ( z ) \approx \tilde { J } ( z )$ with error proportional to the reward projection residual $\| \boldsymbol { r } - \boldsymbol { \phi } ^ { \top } \boldsymbol { z } ^ { * } \| _ { 2 }$ . By definition, $\pi _ { z ^ { * } }$ is the greedy policy with respect to $Q _ { z ^ { * } } ^ { \pi _ { z ^ { * } } }$

$$
\pi _ { z ^ { * } } \left( s \right) = \arg \operatorname* { m a x } _ { a } \psi ^ { \pi _ { z ^ { * } } } \left( s , a \right) ^ { \top } z ^ { * } ,\tag{11}
$$

and is therefore optimal for task $z ^ { * }$ . By Bellman optimality [33], for any other policy $\pi _ { z }                   \mathrm { : }$

$$
Q _ { z ^ { * } } ^ { \pi _ { z ^ { * } } } ( s , a ) \geq Q _ { z ^ { * } } ^ { \pi _ { z } } ( s , a ) \quad \forall z , s , a ,\tag{12}
$$

which implies $z ^ { * } = \arg \operatorname* { m i n } _ { z } \tilde { J } ( z )$ and therefore $\nabla _ { z } \tilde { J } \big | _ { z = z ^ { * } } = 0$

A second-order Taylor expansion of $\tilde { J }$ around $z ^ { * }$ gives:

$$
\tilde { J } ( z ) = \tilde { J } ( z ^ { * } ) + \underbrace { \nabla _ { z } \tilde { J } \big | _ { z ^ { * } } ^ { \top } } _ { = 0 } ( z - z ^ { * } ) + \frac 1 2 ( z - z ^ { * } ) ^ { \top } H \left( z - z ^ { * } \right) + O \left( \| z - z ^ { * } \| ^ { 3 } \right) ,\tag{13}
$$

where $H = \nabla _ { z } ^ { 2 } \tilde { J } \big | _ { z ^ { * } } \succeq 0$ . Differentiating gives $\nabla _ { z } \tilde { J } ( z ) \approx H ( z - z ^ { * } )$ . Assuming a strict positive definiteness of the hessian $( H \succ 0 )$ , a Newton step on Î yields:

$$
z  z - \alpha H ^ { - 1 } \nabla _ { z } \tilde { J } ( z )\tag{14}
$$

$$
= z - \alpha H ^ { - 1 } H ( z - z ^ { * } )\tag{15}
$$

$$
= ( 1 - \alpha ) z + \alpha z ^ { * } .\tag{16}
$$

Projecting back onto the latent sphere $\| z \| = \sqrt { d _ { z } }$ recovers the BI operator (Equation 3). A Newton step on $\tilde { J }$ therefore approximates a Newton step on the true objective J, with residual error proportional to the reward projection quality $\| r - \phi ^ { \top } z ^ { * } \| _ { 2 }$

Remark. The BI operator constitutes a second-order policy gradient update in the BFM latent space. The critic gradient direction of PGA-ME is replaced by the analytically inferred $z ^ { * }$ , obtained in closed form from archive rollouts with no critic training and no backpropagation. The proof extends to transition-dependent rewards $r ( s , a )$ by replacing $\phi ( s )$ with $\phi ( s , a )$ throughout.

## B Background on Behavioral Foundation Models

Behavioral foundation models (BFMs) aim to pretrain a single agent from reward-free interactions such that it can produce optimal policies for any downstream reward at test time, with no or minimal additional learning. The central challenge is to learn a compact representation of the environment's long-term dynamics that is simultaneously expressive enough to cover all tasks and structured enough to support efficient zero-shot policy extraction.

## B.1 Successor Features

The standard $Q \cdot$ function conflates environment dynamics and task rewards, making it expensive to reuse across tasks. [3] propose Successor Features (SF) to decouple them. They assume that the reward on each transition can be linearly decomposed as

$$
r ( s , a , s ^ { \prime } ) = \phi ( s , a , s ^ { \prime } ) ^ { \top } { \bf w } ,\tag{17}
$$

where $\phi : \mathcal { S } \times \mathcal { A } \times \mathcal { S }  \mathbb { R } ^ { d }$ is a feature map shared across all tasks and $\mathbf { w } \in \mathbb { R } ^ { d }$ is a task-specific reward weight vector. Under this factorization, the action-value function of any policy π on task w decomposes as

$$
\begin{array} { r l } & { Q _ { \mathbf { w } } ^ { \pi } ( s , a ) \ = \ \underbrace { { \mathbb E } \left[ \underset { t = 0 } { \sum } \gamma ^ { t } \phi ( s _ { t } , a _ { t } , s _ { t + 1 } ) \ \middle | \ s _ { 0 } = s , a _ { 0 } = a , \pi \right] } _ { \psi ^ { \pi } ( s , a ) } \top _ { \mathbf { w } } \ = \ \psi ^ { \pi } ( s , a ) ^ { \top } \mathbf { w } , } \end{array}\tag{18}
$$

where $\psi ^ { \pi } ( s , a ) \in \mathbb { R } ^ { d }$ are the successor features of $\pi .$

Training. Successor features satisfy a Bellman equation and can therefore be estimated via temporal difference (TD) learning [10]. Given a replay buffer $\mathcal { D } = \{ ( s , a , s ^ { \prime } ) \}$ and a fixed target policy $\bar { \pi } .$ the standard TD loss is

$$
\mathcal { L } _ { \mathrm { S F } } ( \psi ) = \mathbb { E } _ { ( s , a , s ^ { \prime } ) \sim \mathcal { D } } \left\| \psi ( s , a ) - \phi ( s , a , s ^ { \prime } ) - \gamma \bar { \psi } ( s ^ { \prime } , \bar { \pi } ( s ^ { \prime } ) ) \right\| ^ { 2 } ,\tag{19}
$$

where $\bar { \psi }$ denotes a stop-gradient (target network) copy of $\psi .$ Once $\psi ^ { \pi }$ has been learned under any exploration policy, adapting to a new task $\mathbf { w } ^ { \prime }$ only requires re-fitting the scalar weights using linear regression on a handful of evaluated samples:

$$
\mathbf { w } ^ { * } = \underset { \mathbf { w } \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } \mathbb { E } _ { s \in \mathcal { D } } \big [ ( \phi ( s ) ^ { \top } \mathbf { w } - r ( s ) ) ^ { 2 } \big ] ,
$$

and the zero-shot policy is $\begin{array} { r } { \pi _ { \mathbf { w } ^ { * } } ( s ) = \arg \operatorname* { m a x } _ { a } \psi ( s , a , \mathbf { w } ^ { * } ) ^ { \top } \mathbf { w } ^ { * } } \end{array}$ , requiring no additional environment interaction.

The central limitation is that the feature map $\phi$ must be specified in advance (hand-coded or learned), and the expressivity of SF is bounded by the linear span of $\phi ,$ making the quality of the representation a critical bottleneck. Some prominent approaches to learn $\phi$ directly from data are described below:

• Autoencoder [9] learns a compact representation $\phi ( s )$ by minimizing the reconstruction error of the input state through an encoder-decoder architecture, capturing the most salient features of the state space. Here, φ is the encoder $\phi ( s )$ and $f$ is the decoder:

$$
\operatorname* { m i n } _ { f , \phi } \mathbb { E } _ { s \sim \mathcal { D } } \left[ ( f ( \phi ( s ) ) - s ) ^ { 2 } \right] .
$$

• Transition Model learns representations by training a model to predict the next state $s _ { t + 1 }$ given the current state-action pair $\left( { { s _ { t } } , { a _ { t } } } \right)$ , thereby encoding the environment's local dynamics into the latent space. Here, φ is the encoder $\phi ( s )$ and $f$ is the latent transition predictor:

$$
\operatorname* { m i n } _ { f , \phi } \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } ) \sim \mathcal { D } } \left[ ( f ( \phi ( s _ { t } ) , a _ { t } ) - s _ { t + 1 } ) ^ { 2 } \right] .
$$

• Laplacian (Lap) [37] learns representations by approximating the eigenfunctions of the graph Laplacian induced by an exploratory policy, encouraging temporally adjacent states to have similar representations while keeping all state representations globally spread apart via an orthonormality regularization. Here, $\bar { \boldsymbol { f } }$ is the state encoder:

$$
\operatorname* { m i n } _ { f } { \underset { ( s _ { t } , s _ { t + 1 } ) \sim \mathcal { D } } { \mathbb { E } } } \left[ \| f ( s _ { t } ) - f ( s _ { t + 1 } ) \| ^ { 2 } \right] + \mathbb { E } _ { \underset { s ^ { \prime } \sim \mathcal { D } } { s \sim \mathcal { D } } } \left[ ( f ( s ) ^ { \top } f ( s ^ { \prime } ) ) ^ { 2 } - \| f ( s ) \| ^ { 2 } - \| f ( s ^ { \prime } ) \| ^ { 2 } \right] .
$$

• Lower-Rank Approximation of the transition probability [36] factorizes the one-step transition probability $P ( s ^ { \prime } | s , a )$ into a low-rank product $f ( s , \bar { a } ) ^ { \top } \phi ( s ^ { \prime } )$ , effectively performing a spectral decomposition of the transition operator. Here, $f$ and φ represent the left and right singular vectors of the transition matrix:

$$
\operatorname* { m i n } _ { f , \phi } \frac { 1 } { 2 } \mathbb { E } _ { ( s _ { t } , a _ { t } ) \sim \mathcal { D } } \left[ \left( ( f ( s _ { t } , a _ { t } ) ^ { \top } \phi ( s ^ { \prime } ) ) \right) ^ { 2 } \right] - \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } ) \sim \mathcal { D } } \left[ f ( s _ { t } , a _ { t } ) ^ { \top } \phi ( s _ { t + 1 } ) \right] .
$$

• Lower-Rank Approximation of the successor measure [36] learns a low-rank factorization of the successor measure $M ( s , s ^ { \prime } )$ using temporal difference learning. Here, $f$ and $\phi$ are the learned factors that represent the state and its temporally extended features:

$$
\operatorname* { m i n } _ { f , \phi } \mathbb { E } _ { ( s _ { t } , s _ { t + 1 } ) \sim \mathcal { D } } \left[ \left( f ( s _ { t } ) ^ { \top } \phi ( s ^ { \prime } ) - \gamma f ( s _ { t + 1 } ) ^ { \top } \bar { \phi } ( s ^ { \prime } ) \right) ^ { 2 } \right] - 2 \mathbb { E } _ { ( s _ { t } , s _ { t + 1 } ) \sim \mathcal { D } } \left[ f ( s _ { t } ) ^ { \top } \phi ( s _ { t + 1 } ) \right] .
$$

• Bootstrap Your Own Latent (BYOL) [18] learns a latent space by predicting the next state representation from the current state-action pair using a predictor and a target network Here, φ is the encoder and ψ is the online predictor:

$$
\operatorname* { m i n } _ { \phi , \psi } \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } ) \sim \mathcal { D } } \left[ \| \psi ( \phi ( s _ { t } ) , a _ { t } ) - \mathbf { s g } ( \phi ( s _ { t + 1 } ) ) \| _ { 2 } ^ { 2 } \right] .
$$

• Bootstrap Your Own Latent with Bidirectional Prediction (BYOLγ) [22] captures longrange temporal consistency by predicting future states at horizons sampled geometrically. Here, $\psi _ { f }$ is the forward predictor towards future states and $\psi _ { b }$ is the backward predictor towards past states:

$$
\operatorname* { m i n } _ { \phi , \psi _ { f } , \psi _ { b } } \mathbb { E } _ { \mathrm { \textit { \textbf { k } } } \sim \mathrm { G e c o m } ( 1 - \gamma ) } \left[ \| \psi _ { f } ( \phi ( s _ { t } ) , a ) - \mathrm { s g } ( \phi ( s _ { t + k } ) ) \| _ { 2 } ^ { 2 } + \| \psi _ { b } ( \mathrm { s g } ( \phi ( s _ { t + k } ) ) ) - \phi ( s _ { t } ) \| _ { 2 } ^ { 2 } \right] .
$$

## B.2 Forward-Backward Representations

Successor features require φ to be designed before training and restrict expressivity to the linear span of those features. [35] address both shortcomings by replacing the feature-based decomposition with a direct low-rank factorization of the Successor Measure [10, 5]. For a family of policies $\{ \pi _ { z } \} _ { z \in \mathcal { Z } }$ parameterized by task embeddings $z \in \mathcal { Z } \subseteq \mathbb { R } ^ { d }$ , the discounted successor measure of $\pi _ { z }$ starting from $( s , a )$ is

$$
M ^ { \pi _ { z } } ( s , a , X ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \operatorname* { P r } \bigl ( s _ { t + 1 } \in X \mid s _ { 0 } = s , a _ { 0 } = a , \pi _ { z } \bigr ) , \qquad X \subseteq { \mathcal { S } } .\tag{20}
$$

The action-value function for any reward r can then be recovered via the linear functional $Q _ { r } ^ { \pi _ { z } } ( s , a ) = \mathbb { E } _ { s ^ { + } \sim M ^ { \pi _ { z } } ( \cdot | s , a ) } [ r ( s ^ { + } ) ]$ , which makes the successor measure a sufficient statistic for all tasks simultaneously.

Training. Forward-Backward representations factorize the normalized successor measure into a bilinear product via two neural networks: a Forward map $F : \mathcal { S } \times \mathcal { A } \times \mathcal { Z }  \mathbb { R } ^ { d }$ and a Backward map $B : \mathcal { S }  \mathbb { R } ^ { d }$ , trained so that $F ( s , a , z ) ^ { \top } B ( s ^ { \prime } ) \approx M ^ { \dot { \pi } _ { z } } ( s , a , \{ s ^ { \prime } \} ) / \rho ( s ^ { \prime } )$ , where ρ is a reference state distribution. The Bellman consistency of the successor measure, $\dot { M } ^ { \pi _ { z } } ( s , a , \cdot ) = P ( \cdot | s , a ) +$ $\gamma \mathbb { E } _ { a ^ { \prime } \sim \pi _ { z } ( \cdot | s ^ { \prime } ) } [ M ^ { \pi _ { z } } ( s ^ { \prime } , a ^ { \prime } , \cdot ) ]$ , yields a tractable TD objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { F B } } ( F , B ) = \mathbb { E } _ { ( s , a , s ^ { \prime } ) \sim \mathcal { D } , z \sim \mathcal { Z } } \left( F ( s , a , z ) ^ { \top } B ( s ^ { + } ) - \gamma F ( s ^ { \prime } , a ^ { \prime } , z ) ^ { \top } B ( s ^ { + } ) \right) ^ { 2 } } \\ & { \qquad a ^ { \prime } \sim \pi _ { z } ( \cdot | s ^ { \prime } ) , s ^ { + } \sim \mathcal { D } } \\ & { \phantom { \mathcal { L } } - 2 \mathbb { E } _ { ( s , a , s ^ { \prime } ) \sim \mathcal { D } , z \sim \mathcal { Z } } \bigl [ F ( s , a , z ) ^ { \top } B ( s ^ { \prime } ) \bigr ] , } \end{array}\tag{21}
$$

where the first term enforces Bellman consistency and the second anchors the diagonal of the successor measure (i.e., forces $F ( s , a , z ) ^ { \top } B ( s ^ { \prime } ) = 1$ on transitions actually visited under $\pi _ { z } )$ . At test time, a new reward r is encoded into a task embedding $\boldsymbol { z } _ { r } = \mathbb { E } [ r ( \boldsymbol { s } ) \dot { \boldsymbol { B } } ( \boldsymbol { s } ) ]$ and the zero-shot policy is $\begin{array} { r } { \pi _ { z _ { r } } ( s ) = \arg \operatorname* { m a x } _ { a } F ( s , a , z _ { r } ) ^ { \top } z _ { r } } \end{array}$ , requiring no additional environment interaction.

## B.3 TD-JEPA

FB learns the successor measure factorization directly in the state space, without any explicit state encoder. This makes it difficult to scale to raw pixel inputs, where compressed representations are essential, and prevents the reuse of learned features for representation-based transfer. Furthermore, FB offers no mechanism to learn a state encoder that is aligned with long-term, policy-conditioned dynamics. [2] introduce TD-JEPA, which replaces the explicit successor measure factorization of FB with a latent-predictive objective inspired by the Joint-Embedding Predictive Architecture (JEPA) paradigm of [23].

Training TD-JEPA trains four components end-to-end: a state encoder $\phi : \mathcal { S }  \mathbb { R } ^ { d _ { \phi } }$ , a task encoder $\overline { { \psi } } : \mathcal { S }  \mathbb { R } ^ { d _ { \psi } }$ , a policy-conditioned predictor $T _ { \phi } : \mathbb { R } ^ { d _ { \phi } } \times \mathcal { A } \times \mathcal { Z }  \mathbb { R } ^ { d _ { \psi } }$ , and a family of parameterized policies $\{ \pi _ { z } \} _ { z \in \mathcal { Z } }$

The predictor is trained to approximate multi-step, policy-conditioned latent dynamics. TD-JEPA uses the Bellman equation for successor features in latent space, yielding the off-policy, single-step TD-JEPA loss:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { T D - J E P A } } ( \phi , T _ { \phi } ) = { \mathbb { E } } _ { ( s , a , s ^ { \prime } ) \sim \mathcal { D } , \ z \sim \mathcal { Z } } \left\| T _ { \phi } \big ( \phi ( s ) , a , z \big ) - \bar { \phi } ( s ^ { \prime } ) - \gamma T _ { \phi } \big ( \bar { \phi } ( s ^ { \prime } ) , a ^ { \prime } , z \big ) \right\| ^ { 2 } , } \\ & { \qquad a ^ { \prime } \sim \pi _ { z } ( \cdot | s ^ { \prime } ) } \end{array}\tag{22}
$$

where $\bar { \phi }$ denotes a stop-gradient copy of $\phi$ (target encoder).

Zero-shot policy extraction follows the FB framework: at test time a, reward r is projected onto the task encoder as $z _ { r } = \mathbb { E } [ r ( s ) \psi ( s ) ]$ , and the policy $\begin{array} { r } { \pi _ { z _ { r } } ( s ) = \arg \operatorname* { m a x } _ { a } T _ { \phi } ( \phi ( s ) , a , z _ { r } ) ^ { \top } z _ { r } } \end{array}$ is executed without any environment interaction.

## B.4 BTD-FB

Forward-Backward representations learn a rich, task-agnostic successor measure factorization, but their policy training component still relies on uniformly sampled task vectors $z \sim \operatorname { U n i f } ( S ^ { d - 1 } )$ . [4] argue that this uniform sampling strategy suffers from a signal dilution effect: in high-dimensional latent spaces, uniformly drawn task vectors are nearly orthogonal to the behavioral space, causing the variance of returns across policies to vanish and the learning signal to collapse. BTD-FB addresses this by replacing the uniform task prior in FB with the Behavioral Task Distribution (BTD), a data-driven distribution fitted directly to the offline dataset.

Training. BTD-FB first trains Forward and Backward maps F and B using the standard FB objective (Eq. 21). Following training, the state embedding is fixed as $\begin{array} { r } { \phi ( s ) = \mathbb { E } [ B B ^ { \top } ] ^ { - 1 } B ( s ) } \end{array}$ . To construct the BTD, $N _ { \tau }$ sub-trajectories of random lengths are sampled from the offline dataset, and their empirical feature occupancies are computed and normalized to yield task vectors:

$$
z _ { \tau } = \frac { \tilde { \psi } ^ { \tau } } { \lVert \tilde { \psi } ^ { \tau } \rVert _ { 2 } } , \qquad \tilde { \psi } ^ { \tau } = \sum _ { t = 0 } ^ { | \tau | } \gamma ^ { t } \phi ( s _ { t } ) .\tag{23}
$$

A Gaussian Mixture Model $p _ { \theta } ( z )$ is then fitted by maximum likelihood to the resulting empirical task set. The conditional policy is trained from scratch on these extracted tasks instead.

## C Background on Quality-Diversity Methods

Quality-Diversity (QD) algorithms discover large collections of diverse, high-performing solutions simultaneously. A QD problem is defined by a fitness function $r : S  \mathbb { R }$ and a behavioral descriptor $\beta : { \cal S }  B .$ where B is discretized into a finite grid of cells. An archive C stores at most one elite per cell, replaced only when a candidate achieves strictly higher fitness in the same cell. Performance is measured by coverage (fraction of populated cells) and QD-score (sum of all elite fitnesses). Most QD variants share this skeleton and differ only in their choice of variation operator, as summarized in Algorithm 1.

Algorithm 1 Base Quality-Diversity Algorithm   
Require: Fitness function r, descriptor $\phi ,$ budget $T$   
1: Initialize archive $c \gets \emptyset$   
2: for $t = 1$ to $T$ do   
3: Sample parent $\theta \sim \mathcal { C }$ randomly if ${ \mathcal { C } } = \emptyset$   
4: θ′ ← OPERATOR(θ, C) differs across methods   
5: Evaluate fitness $f \gets r ( \theta ^ { \prime } )$ and descriptor $b \gets \phi ( \theta ^ { \prime } )$   
6: if $f > r ( \mathcal { C } [ b ] )$ then   
7: ${ \mathcal C } [ b ] \stackrel { \cdot } {  } ( \stackrel { \cdot } { \theta ^ { \prime } } , f )$   
8: end if   
9: end for   
10: return $\mathcal { C }$

## C.1 MAP-Elites

MAP-Elites (ME) [26] is the canonical QD algorithm. It addresses the exploration-exploitation tension in a single-objective optimizer by explicitly maintaining one best solution per behavioral cell, so that the search covers the descriptor space rather than collapsing onto a single high-fitness region

Operator. Parents are sampled from the archive and perturbed by an isotropic Gaussian mutation:

$$
\theta ^ { \prime } = \theta + \varepsilon , \qquad \varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) .\tag{24}
$$

Each offspring is evaluated and inserted into its behavioral cell if it improves the current elite.

ME is simple and highly parallelizable. But because mutation is isotropic, ME does not exploit any structure of the fitness landscape: every direction in parameter space is equally likely to be explored, regardless of which directions have previously led to archive improvements.

## C.2 CMA-ME

CMA-ME [15] addresses the sample-inefficiency of isotropic mutation by replacing it with CMA-ES [20], which maintains an adaptive covariance matrix C estimated from directions that previously improved the archive.

Operator. Offspring are sampled from a multivariate Gaussian whose covariance is updated from archive improvement signals:

$$
\theta ^ { \prime } = \theta + \varepsilon , \qquad \varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbf { C } ) .\tag{25}
$$

Despite adapting the mutation distribution, CMA-ME still relies purely on undirected sampling: it does not use any task-specific gradient information to steer offspring towards higher-fitness regions.

## C.3 PGA-ME

PGA-ME [27] introduces a gradient-based variation operator to guide the search towards high-fitness regions, complementing random mutation with directed improvement steps.

Operator. Half of each generation's offspring are produced by Gaussian mutation as in ME; the other half by gradient ascent on a learned critic $Q _ { \phi }$ trained via TD3:

$$
\theta  \theta + \alpha \nabla _ { \theta } Q _ { \phi } ( s , \pi _ { \theta } ( s ) ) .\tag{26}
$$

The critic is trained on transitions collected during archive evaluation.

The limitation with PGA-ME is that gradient steps optimize fitness without any awareness of the behavioral descriptor, so the method cannot explicitly target underpopulated regions of the archive, and exploration can collapse on the optimal behavior.

## C.4 DCRL-ME

Motivation. DCRL-ME [13] extends PGA-ME by conditioning both actor and critic on a target behavioral descriptor $b ^ { * }$ , so that gradient steps jointly optimize fitness and steer the offspring toward a desired region of the descriptor space.

Operator. The policy gradient update becomes descriptor-conditioned:

$$
\theta  \theta + \alpha \nabla _ { \theta } Q _ { \phi } \big ( s , \pi _ { \theta } ( s , b ^ { * } ) , b ^ { * } \big ) ,\tag{27}
$$

and the conditioned actor $\pi _ { \boldsymbol { \theta } } \big ( \cdot , b ^ { * } \big )$ is also used as a generative model, injecting candidates directly into target archive cells.

Both the actor and critic in DCRL-ME are trained from scratch using only transitions collected during archive evaluation, making them dependent on the quality and density of the reward signal encountered online.

## C.5 DDE-Elites

## Motivation.

DDE-Elites [17] addresses the problem of high dimensionality of the policy parameter space by learning a compact Data-Driven Encoding (DDE) from the archive elites using a VAE, and using it as a variation operator within ME.

Operators. At each generation, a VAE is retrained on all current archive solutions by minimzing this loss (maximizing the Evidence Lower Bound):

$$
{ \mathcal { L } } _ { \mathrm { V A E } } = - \mathbb { E } _ { q _ { \theta _ { e } } ( z \mid \theta ) } [ \log p _ { \theta _ { d } } ( \theta \mid z ) ] + \mathrm { K L } ( q _ { \theta _ { e } } ( z \mid \theta ) \parallel \mathcal { N } ( 0 , I ) ) .\tag{28}
$$

Three variation operators are combined at each generation: (i) Isometric mutation applies Gaussian noise directly in parameter space; (ii) line mutation applies directional Gaussian noise, where the variance per dimension scales with the difference between two archive elites, biasing perturbations toward directions of known diversity; (iii) reconstructive crossover shifts a solution toward the VAE-learned distribution without adding noise:

$$
\begin{array} { r } { \theta ^ { \prime } = \frac { 1 } { 2 } \big ( \theta + D _ { \theta _ { d } } ( E _ { \theta _ { e } } ( \theta ) ) \big ) . } \end{array}\tag{29}
$$

The mix of operators is selected at each generation by a UCB1 bandit algorithm [1], which tracks the success rate of each operator and balances exploration and exploitation automatically.

The compact latent space simplifies search, but the VAE is trained only on solutions already in the archive, so the latent space does not necessarily reflect task-relevant structure.

## C.6 ME+Pretrain

This is a baseline that we introduce to isolate the contribution of offline initialization from the structure of a behavioral latent space. In ME+Pretrain, MLP policies are warm-started via offline TD3 on the same dataset as the BFM, using the task fitness as reward, before running standard ME.

Operator. The pretraining objective is:

$$
\theta ^ { \star } = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } \left[ Q _ { \omega } { \big ( } s , \pi _ { \theta } ( s ) { \big ) } \right]\tag{30}
$$

after which ME runs from $\theta ^ { \star }$

## D Experimental Details

## D.1 Environments and Tasks

![](images/b190ebf2ce95e978eced562c0bc77447e795b5ba73f9612836aa82c3cb2870dd.jpg)  
Figure 6: Evaluation environments. The four benchmarks used in this work: HalfCheetah\_uni, Walker2d\_uni, AntMaze-medium, and Cube-single (from left to right). Difficulty increases from left to right, with AntMaze-medium and Cube-single introducing long-horizon tasks and contact-rich interaction, respectively.

## HalfCheetah\_uni:

A planar 2D robot from QDax [8]. It has a state dimension of 17 and an action dimension of 6. It represents the simplest setting with dense rewards and straightforward dynamics.

Behavioral descriptor:

• Foot contact pattern (2D): The ratio of the contact of the front/back feet with the ground.

## Fitness functions:

• Walk: Reward is linearly proportional to forward velocity, capped at 2 m/s.

• Run: Reward is linearly proportional to forward velocity, capped at 5 m/s.

• Walk backward: Reward is linearly proportional to backward velocity, capped at -2 m/s.

• Run backward: Reward is linearly proportional to backward velocity, capped at -5 m/s.

## Walker2D\_uni:

A bipedal walker from QDax, with slightly more complex dynamics than HalfCheetah\_uni. It has a state dimension of 17 and an action dimension of 6.

## Behavioral descriptor:

• Foot contact pattern (2D): The ratio of the contact of the front/back feet with the ground.

## Fitness functions:

• Run forward: Maximize the velocity of the center-of-mass of the Walker agent.

• Run backward: Maximize the backward velocity of the point-of-mass of the Walker agent.

• Upright walk: Maximize forward velocity while maintaining an upright torso above 0.3.

• Jump: Maintain a center-of-mass height above 1.2.

## AntMaze-medium:

A quadrupedal Ant robot navigating a medium-sized maze from OGBench [28]. It has a state dimension of 29 and an action dimension of 8, and introduces long-horizon navigation with deceptive reward landscapes.

## Behavioral descriptor:

• (x, y) position: the coordinate of the position of the ant in the maze.

• Foot contact pattern (4D): The ratio of the contact of the ant's feet with the ground.

## Fitness functions:

• Energy efficiency: Minimize arm energy consumption (norm of the actions), i.e. maximize the fitness function $f = \sqrt { d _ { A } } - \| a _ { t } \| _ { 2 } , \mathrm { w i t h } d _ { A } \bar { = } 8 .$

• Reach center: Reach the center of the maze in position (12, 12) with a radius of 1.

• Top wall: Reach the top wall of the maze after $x = 2 0$

• Right wall: Reach the right wall of the maze after $y = 2 0$

• Top wall dense: The dense version of top wall, with $\mathrm { f i t n e s s } = e ^ { ( \mathrm { a n t . x } - 2 0 ) / 3 }$

• Right wall dense: The dense version of right wall, with fitness $= e ^ { ( \mathrm { a n t . y - 2 0 } ) / 3 }$

Cube-single: A 6-DoF robotic arm from OGBench. The state space dimension is 28. The action space is 5-dimensional $( \Delta x , \Delta y , \Delta z , \Delta \mathrm { y a w } ,$ gripper speed), and it represents the hardest setting requiring temporally extended contact-rich manipulation.

## Behavioral descriptor:

$( x , y )$ position: the $( x , y )$ position of the cube.

• (x, z) position: replaces the y position with the height of the cube.

• Grasp style: (Relative grasp angle, how wide the grip is).

• Wrist configuration: (the angle in joint\_3, the angle in joint\_4)

## Fitness functions:

• Energy efficiency: Minimize arm energy consumption (norm of the actions), i.e. maximize the fitness function $f = \sqrt { d _ { A } } - \| a _ { t } \| _ { 2 } ,$ with $d _ { A } = 5$

• Pick-and-place: Binary success of lifting and placing the cube at a target location.

QD-score normalization. Each environment contains multiple tasks with different reward scales, making raw QD-scores incomparable across tasks. To aggregate results within an environment, we apply per-task min-max normalization: for each task, we compute the minimum and maximum QDscore observed across all methods and all seeds, and rescale each method's score to [0, 1] accordingly. Normalized scores are then averaged across all tasks within an environment to produce the aggregate curves shown in Figure 2. This normalization is applied identically to QD-score and max-fitness curves. Unnormalized per-task QD-scores are reported in Appendix E.12.

## D.2 Baselines Implementation Details

All baselines share the same environment setup, archive configuration, training budget, and random seeds as BFM-QD for a fair comparison.

Table 2: Shared hyperparameters across all methods.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Archive grid size</td><td>50 × 50</td></tr><tr><td>Number of generations</td><td>500</td></tr><tr><td>Parallel environments</td><td>400</td></tr><tr><td>Episode length (Brax) Episode length (OGBench)</td><td>500 steps 1 000 steps</td></tr></table>

MAP-Elites. Offspring are produced by isotropic Gaussian mutation. Five operator instances of Gaussian mutation with a sigma ladder run in parallel, each handling an equal share of candidates per generation. Archive policies are 2-hidden-layer MLPs with ReLU activations: input $ ~ 1 2 8 ~ $ relu → 128 → output → tanh, taking the state observation as input and producing actions through a tanh output layer.

Table 1: Summary of environments, behavioral descriptors, and fitness functions used in our evaluation. Difficulty combines reward density and task complexity: Easy = dense reward, simple dynamics;Medium= dense reward, complex dynamics or navigation;Hard = dense but deceptive or long-horizon;Very Hard = sparse reward, contact-rich manipulation.
<table><tr><td>Environment</td><td>Behavioral Descriptor</td><td>Fitness Function</td><td>Density</td><td>Difficulty</td></tr><tr><td rowspan="4">HalfCheetah</td><td rowspan="4">Foot contact</td><td>Run backward</td><td>Dense</td><td>Easy</td></tr><tr><td>Walk backward</td><td>Dense</td><td>Easy</td></tr><tr><td>Walk forward</td><td>Dense</td><td>Easy</td></tr><tr><td>Run forward</td><td>Dense</td><td>Easy</td></tr><tr><td rowspan="4">Walker</td><td rowspan="4">Foot contact</td><td>Run backward</td><td>Dense</td><td>Easy</td></tr><tr><td>Run forward</td><td>Dense</td><td>Easy</td></tr><tr><td>Upright walk</td><td>Dense</td><td>Medium</td></tr><tr><td>Jump</td><td>Dense</td><td>Medium</td></tr><tr><td rowspan="4">AntMaze-medium</td><td rowspan="2">(x, y) position</td><td>Energy efficiency</td><td>Dense</td><td>Medium</td></tr><tr><td>Reach center</td><td>Dense</td><td>Hard</td></tr><tr><td rowspan="4">Foot contact</td><td>Reach right wall</td><td>Dense</td><td>Hard</td></tr><tr><td>Reach top wall</td><td>Dense</td><td>Hard</td></tr><tr><td>Reach right wall</td><td>Sparse</td><td>Very Hard</td></tr><tr><td>Reach top wall</td><td>Sparse</td><td>Very Hard</td></tr><tr><td rowspan="4">Cube-single</td><td>(x, y) position of cube</td><td>Energy efficiency</td><td>Dense</td><td>Hard</td></tr><tr><td>(x, z) position of cube</td><td>Energy efficiency</td><td>Dense</td><td>Hard</td></tr><tr><td>Grasp style</td><td>Pick-and-place</td><td>Sparse</td><td>Very Hard</td></tr><tr><td>Wrist configuration</td><td>Pick-and-place</td><td>Sparse</td><td>Very Hard</td></tr></table>

Table 3: MAP-Elites hyperparameters.
<table><tr><td>Parameter</td><td> Value</td></tr><tr><td>Mutation σ</td><td>{0.1, 0.5, 1.0, 1.0, 5.0}</td></tr></table>

CMA-ME. Uses the same emitter configuration and archive policy architecture as ME, but replaces isotropic mutation with CMA-ES, adapting the covariance matrix from archive improvement signals at each generation.

PGA-ME. Archive policies are 2-hidden-layer MLPs with 128 units per layer, ReLU activations, and a tanh output layer. Half of each generation's offspring are produced by Gaussian mutation; the other half by one step of gradient ascent on a TD3 critic. The critic is a twin Q-network with two hidden layers of 256 units and ReLU activations, taking the concatenation of state and action as input. The TD3 actor has two hidden layers of 128 units with ReLU activations and produces a deterministic action through a tanh output layer. Both are trained online from transitions collected during archive evaluation. A warmup phase is applied before any PG variation, allowing the replay buffer to fill before critic-guided mutations begin.

Table 4: PGA-ME hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Archive policy MLP Critic hidden width Critic learning rate Actor learning rate Replay buffer size</td><td>2 × 128 hidden units 256  $3 \times 1 0 ^ { - 4 }$   $3 \times 1 0 ^ { - 4 }$  1 000 000</td></tr><tr><td>Batch size Discount γ Target smoothing τ TD3 policy noise σ</td><td>256 0.99</td></tr><tr><td>TD3 noise clip Target update frequency</td><td>0.005</td></tr><tr><td>Critic updates per gen. Warmup</td><td>0.2 0.5 every 2 critic steps 256</td></tr></table>

DCRL-ME. Extends PGA-ME with a descriptor-conditioned actor and critic, both taking the normalized 2-dimensional behavioral descriptor concatenated to their respective inputs (state for the actor, state-action for the critic). The archive policy MLP, actor, and critic retain the same widths and activations as ME. The offspring batch is split across three variation operators: Gaussian mutation (GA), descriptor-conditioned PG, and Actor Injection (AI). For AI, a target descriptor $b ^ { * }$ is sampled uniformly from the descriptor space and the actor's first layer is analytically specialized to it: since the first layer computes $W [ s ; b ^ { * } ] + b ,$ fixing $b ^ { * }$ reduces it to a standard affine map $W _ { s } s + ( W _ { b ^ { * } } b ^ { * } + b )$ yielding a valid archive policy with no additional forward pass. TD3 training hyperparameters are otherwise identical to PGA-ME (Table 4).

Table 5: DCRL-ME hyperparameters. TD3 parameters are identical to PGA-ME (Table 4).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Offspring split (GA / PG / AI)</td><td>50% / 25% / 25%</td></tr><tr><td>Similarity length scale L</td><td>0.1</td></tr></table>

DDE-Elites. Archive policies use the same MLP architecture as ME. The VAE has a symmetric encoder and decoder, each consisting of two hidden layers of 256 units with ReLU activations. The encoder maps a flattened policy parameter vector to a 50-dimensional latent space, producing a mean and log-variance from which a latent code is sampled via the reparameterization trick. The VAE is retrained every generation on all current archive weight vectors. Three variation operators are used: isometric mutation, line mutation, and reconstructive crossover (averaging a solution with its VAE reconstruction). Their mix is selected each generation by a UCB1 bandit. When fewer than the minimum number of elites are present, only isometric and line mutation are used.

Table 6: DDE-Elites hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Archive policy MLP</td><td>2 × 128 hidden units</td></tr><tr><td>VAE latent dimension</td><td>50</td></tr><tr><td>VAE encoder/decoder width</td><td>256</td></tr><tr><td>VAE learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>VAE training epochs</td><td>5</td></tr><tr><td>VAE KL weight</td><td>0.01</td></tr><tr><td>VAE retraining frequency</td><td>every generation</td></tr><tr><td>Min. elites to retrain</td><td>50</td></tr><tr><td>σ isometric mutation</td><td>0.003</td></tr><tr><td>σ line mutation</td><td>0.1</td></tr><tr><td>Bandit sliding window</td><td>1000</td></tr></table>

ME+Pretrain. MLP policies are initialized by offline TD3 training on the same dataset used to train the BFM, using the task fitness as reward. The offline actor and critic use the same architecture as in PGA-ME and DCRL-ME. The pretrained weights are then used as the starting point for ME with the same Gaussian mutation and emitter configuration as the ME baseline.

Table 7: ME+Pretrain hyperparameters (offline TD3 phase).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Offline training steps</td><td>2 000 000</td></tr><tr><td>Batch size</td><td>2048</td></tr><tr><td>Learning rate (actor, critic)</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Target Polyak τ</td><td>0.01</td></tr><tr><td>Hidden width</td><td>1024</td></tr><tr><td>Feature dimension</td><td>512</td></tr></table>

## D.3 BFM Implementation Details

Laplacian (Lap). Lap learns only one state encoder $f \colon | S | \ { \xrightarrow { \mathrm { \ n t a n h } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { \ R e L U } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { \ R e L U } } } \ d _ { z }$

$\mathbf { B Y O L } \gamma$ representation. $\operatorname { B Y O L } \gamma$ learns three networks:

• The state encoder $g \colon \left| { \cal S } \right| \xrightarrow { \mathrm { n t a n h } } \quad$ 1024 $\xrightarrow { \mathrm { R e L U } } d _ { z } ^ { 2 }$ , with a target network ē updated by EMA.

• The forward predictor $f \colon ( d _ { z } + | { \mathcal { A } } | ) \ { \xrightarrow { \mathrm { n t a n h } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { R e L U } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { R e L U } } } \ d _ { z }$

• The backward predictor b: $d _ { z } \ { \xrightarrow { \mathrm { \ n t a n h } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { \ R e L U } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { \ R e L U } } } \ d _ { z } .$

All three are trained jointly with an orthogonality regularizer on $g .$

Table 8: SF-TD3 hyperparameters (shared across Lap and $\operatorname { B Y O L } \gamma$ variants).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Latent dimension  $d _ { z }$ </td><td>50</td></tr><tr><td>SF training steps</td><td>2 000 000</td></tr><tr><td>SF batch size</td><td>1024</td></tr><tr><td>SF learning rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>SF target Polyak τ</td><td>0.01</td></tr><tr><td>Actor/critic training steps</td><td>2 000 000</td></tr><tr><td>Actor/critic learning rate</td><td>10⁻⁴</td></tr><tr><td>TD3 batch size</td><td>2 048</td></tr><tr><td>Discount γ</td><td>0.99</td></tr><tr><td>Target EMA τ</td><td>0.01</td></tr><tr><td>z-inference batch size</td><td>10000</td></tr></table>

Forward-Backward (FB) model. The FB model follows [35]. The forward map $F$ takes $( s , a , z )$ through two parallel streams:

• obs-action: $( | S | + | A | ) \xrightarrow { \mathrm { ~ n t a n h } } 5 1 2 \left[ \mathrm { R e L U } \right]$

• obs-z: (|S| + dz) ntanb> 512 [ReLU].

Their concatenation passes through a shared trunk 1024 $\xrightarrow { \mathrm { R e L U } } 1 0 2 4 \xrightarrow { \mathrm { R e L U } } d _ { z }$

The backward network B is a 4-hidden-layer ${ \mathrm { M L P } } \colon | S | \ { \xrightarrow { \mathrm { \ n t a n h } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { \ R e L U } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { \ R e L U } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { \ R e L U } } } \ { d } _ { z } ,$ l2-normalized and rescaled to radius $\sqrt { d _ { z } } .$ The actor shares the same preprocessed trunk as $F$ (obs-only and obs-z streams) with a tanh action head.

Table 9: FB hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Latent dimension  $d _ { z }$ </td><td>50</td></tr><tr><td>Training steps</td><td>2 000 000</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Learning rate</td><td>10⁻4</td></tr><tr><td>Discount γ EMA τ</td><td>0.99</td></tr><tr><td></td><td>0.01</td></tr><tr><td>z-inference batch size</td><td>10000</td></tr></table>

TD-JEPA. TD-JEPA learns four components:

• State encoder $\phi \colon | S | \ { \xrightarrow { \mathrm { n t a n h } } } \ 1 0 2 4 \ { \xrightarrow { \mathrm { R e L U } } } \ d _ { \phi }$ , where $d _ { \phi } = 5 1 2 \ /$

• Task encoder $\psi \colon | S | \xrightarrow { \mathrm { \scriptscriptstyle ~ n t a n h } } 1 0 2 4 \xrightarrow { \mathrm { \scriptscriptstyle ~ R e L U } } d _ { z } , \ell _ { 2 }$ -normalized and rescaled to radius $\sqrt { d _ { z } }$

• Forward predictor $T _ { \phi }$ (predicting future ψ from current $\phi ,$ twin MLP): $( d _ { \phi } + d _ { z } + \vert \mathcal { A } \vert ) \xrightarrow { \mathrm { ~ n t a n h } }$ 1024 $\xrightarrow { \mathrm { R e L U } } d _ { z }$

• Backward predictor $T _ { \psi }$ (predicting future $\phi$ from current $\psi ,$ twin MLP): $( d _ { z } + d _ { z } + \vert \mathcal { A } \vert ) \xrightarrow { \mathrm { ~ n t a n h } }$ 1024 $\xrightarrow { \mathrm { R e L U } } d _ { \phi }$

Each twin predictor averages its two heads before computing the TD-JEPA loss. Target networks are maintained for $\phi , \psi , T _ { \phi }$ , and $T _ { \psi }$ via EMA.

• The actor: $\left( d _ { \phi } + d _ { z } \right) \xrightarrow { \mathrm { \scriptscriptstyle { n t a n h } } } 1 0 2 4 \xrightarrow { \mathrm { \scriptscriptstyle { R e L U } } } 1 0 2 4 \xrightarrow { \mathrm { \scriptscriptstyle { R e L U } } } | A | .$ with tanh output.

Table 10: TD-JEPA hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Latent dimension  $d _ { z }$ </td><td>50</td></tr><tr><td>Training steps</td><td>2 000 000</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Target EMA τ Actor std. dev. σ</td><td>0.01</td></tr><tr><td>Actor std. dev. clip</td><td>0.2</td></tr><tr><td>Ortho. coeff.</td><td>0.3</td></tr><tr><td> $\lambda _ { \phi }$ </td><td>1.0</td></tr><tr><td>Ortho. coeff.  $\lambda _ { \psi }$ </td><td>1.0</td></tr><tr><td>z-inference batch size</td><td>10000</td></tr></table>

Backward Inference (BI) Operator. At each generation, half of the offspring are produced by Gaussian mutation and half by the BI operator. For the BI operator, state-reward pairs are sampled from the replay buffer and used to compute $z ^ { * }$ in closed form via Equation 2. Each parent is then interpolated toward $z ^ { * }$ with step size α and projected back onto the latent sphere.

Table 11: Backward Inference (BI) operator hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Step size α</td><td>0.02</td></tr><tr><td>BI probability</td><td>0.5</td></tr><tr><td>Gaussian mutation σ</td><td>1.0</td></tr><tr><td>z-inference batch size</td><td>10000</td></tr></table>

Interpolation (Iso) Operator. At each generation, half of the offspring are produced by Gaussian mutation and half by the interpolation operator. For the interpolation operator, two parents are sampled from the archive and linearly interpolated with a coefficient $\alpha \sim \mathcal { \bar { U } } [ 0 , 1 ]$ , before projecting back onto the latent sphere.

Table 12: Interpolation (Iso) operator hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Interpolation probability</td><td>0.5</td></tr><tr><td>α distribution</td><td> $\boldsymbol { \mathcal { U } } [ 0 , 1 ]$ </td></tr><tr><td>Gaussian mutation σ</td><td>1.0</td></tr></table>

Dataset collection for BFM-QD For HalfCheetah\_uni and Walker2D\_uni, we collect our own dataset using RND [7]. We use the official implementation from [21]. We keep the exact same hyperparameters.

Pseudo-algorithm of BFM-QD Algorithm 2 summarizes the BFM-QD framework. The key difference from standard QD (Algorithm 1) is that the archive stores latent codes z rather than policy weights $\theta ,$ and all variation operators act directly in Z. The BFM policy πBFM(· | z) is called at evaluation time and remains frozen throughout.

Algorithm 2 BFM-QD   
Require: Pretrained BFM policy πBFM(· | z)   
Require: Latent space Z, archive C   
Require: Number of iterations T, batch size $N$   
1: Initialize archive $c \gets \emptyset$   
2: for $t = 1$ to $T$ do   
3: if C is empty then   
4: Sample N latent vectors $\{ z _ { i } \} _ { i = 1 } ^ { N } \sim \mathcal { Z }$   
5: else   
6: Sample parent latent vectors $\{ z _ { i } \} _ { i = 1 } ^ { N }$ from C   
7: end if   
8: Generate offspring latents: $z _ { i } ^ { \prime } = \mathrm { P }$ erturbation(zi)   
9: for each latent vector $z _ { i } ^ { \prime }$ do   
10: Roll out policy πBFM $( \cdot \mid z _ { i } ^ { \prime } )$   
11: Evaluate fitness $f _ { i }$ and descriptor $b _ { i }$   
12: Attempt insertion into archive C using $( z _ { i } ^ { \prime } , f _ { i } , b _ { i } )$   
13: end for   
14: end for   
15: return archive C

Algorithms 3 and 4 detail two variation operators: (1) The backward Inference defined in 4.2; (2) Interpolation defined in 5.3.2.

Algorithm 3 BFM-QD with Backward Inference (BI) Operator   
Require: Pretrained BFM policy $\pi _ { \mathrm { B F M } } ( \cdot \mid z )$ , feature map $\phi$   
Require: Latent space Z, archive C, replay buffer D   
Require: Number of iterations T, batch size N, step size $\alpha ,$ latent dimension $d _ { z }$   
1: Initialize archive ${ \mathcal { C } } \gets \emptyset ,$ replay buffer $\mathcal { D }  \emptyset$   
2: for $t = 1$ to $T$ do   
3: if C is empty then   
4: Sample N latent vectors $\{ z _ { i } ^ { \prime } \} _ { i = 1 } ^ { N } \sim \mathcal { Z }$   
5: else   
6: Sample parent latent vectors $\{ z _ { i } \} _ { i = 1 } ^ { N }$ from C   
7: Sample $\dot { \mathcal { D } } _ { r } = \{ ( s _ { t } , r ( s _ { t } ) ) \}$ from D   
8: $z ^ { \ast } \gets \left( \mathbb { E } _ { s \sim \mathcal { D } _ { r } } [ \phi ( s ) \phi ( s ) ^ { \top } ] \right) ^ { - 1 } \mathbb { E } _ { s \sim \mathcal { D } _ { r } } [ \phi ( s ) r ( s ) ]$   
9: for each $z _ { i }$ do   
10: if with probability 0.5 then   
11: $z _ { i } ^ { \prime } \stackrel { . } {  } z _ { i } + \varepsilon , \stackrel { . } { \varepsilon } \sim \mathcal N ( 0 , \sigma ^ { 2 } I )$ Gaussian mutation   
12: else   
13: $z _ { i } ^ { \prime } \gets ( 1 - \alpha ) \cdot z _ { i } + \alpha \cdot z ^ { * }$ Backward Inference   
14: end if   
15: $z _ { i } ^ { \prime }  z _ { i } ^ { \prime } / \| z _ { i } ^ { \prime } \| \cdot \sqrt { d _ { z } }$ project all offspring onto sphere   
16: end for   
17: end if   
18: $z _ { i } ^ { \prime }  z _ { i } ^ { \prime } / \| z _ { i } ^ { \prime } \| \cdot \sqrt { d _ { z } }$ project onto latent sphere   
19: for each latent vector $z _ { i } ^ { \prime }$ do   
20: Roll out policy πBFM( $\cdot \mid z _ { i } ^ { \prime } )$ , collecting trajectory $\tau _ { i } = \{ ( s _ { t } , r ( s _ { t } ) ) \}$   
21: Evaluate fitness $f _ { i }$ and descriptor $b _ { i }$   
22: Store trajectory τi in replay buffer D   
23: Attempt insertion into archive C using $( z _ { i } ^ { \prime } , f _ { i } , b _ { i } )$   
24: end for   
25: end for   
26: return archive C

Algorithm 4 BFM-QD with Interpolation (Iso) Operator   
Require: Pretrained BFM policy $\pi _ { \mathrm { B F M } } ( \cdot \mid z )$   
Require: Latent space Z, archive $\mathcal { C }$   
Require: Number of iterations T, batch size $N _ { \cdot }$ , latent dimension $d _ { z }$   
1: Initialize archive $c \gets \emptyset$   
2: for $t = 1$ to $T$ do   
3: if C is empty then   
4: Sample N latent vectors $\{ z _ { i } ^ { \prime } \} _ { i = 1 } ^ { N } \sim \mathcal { Z }$   
5: else   
6: Sample parent latent vectors $\{ z _ { i } \} _ { i = 1 } ^ { N }$ from C   
7: for each $z _ { i }$ do   
8: if with probability 0.5 then   
9: $z _ { i } ^ { \prime } \stackrel { - } {  } z _ { i } + \varepsilon , \stackrel { \cdot } { \varepsilon } \sim \mathcal N ( 0 , \sigma ^ { 2 } I )$ Gaussian mutation   
10: else   
11: Sample second parent $z _ { j } \sim \mathcal { C }$   
12: Sample $\alpha \sim \mathcal { U } [ 0 , 1 ]$   
13: $z _ { i } ^ { \prime } \gets ( 1 - \alpha ) \cdot \dot { \cdot } z _ { i } \dot { + } \alpha \cdot z _ { j }$ Interpolation   
14: end if   
15: $z _ { i } ^ { \prime }  z _ { i } ^ { \prime } / \| z _ { i } ^ { \prime } \| \cdot \sqrt { d _ { z } }$ project all offspring onto sphere   
16: end for   
17: end if   
18: $z _ { i } ^ { \prime }  z _ { i } ^ { \prime } / \| z _ { i } ^ { \prime } \| \cdot \sqrt { d _ { z } }$ project onto latent sphere   
19: for each latent vector $z _ { i } ^ { \prime }$ do   
20: Roll out policy πBFM $( \cdot \ | \ z _ { i } ^ { \prime } )$   
21: Evaluate fitness $f _ { i }$ and descriptor $b _ { i }$   
22: Attempt insertion into archive C using $( z _ { i } ^ { \prime } , f _ { i } , b _ { i } )$   
23: end for   
24: end for   
25: return archive C

## E Additional Results

## E.1 Expanded main results

Figure 7 reports the per-task, unnormalized QD-score curves for all methods across all 18 tasks and four environments. The results confirm the aggregated findings in Figure 2: BFM-QD variants consistently lead across tasks, with the performance gap over parameter-space baselines widening sharply as reward density decreases and task complexity increases.

![](images/19fcd0bbb64dade76e93cf2a7ab220f7f25369e7c81074d08fc29287fcb83ac4.jpg)  
Figure 7: QD-score across all environments and methods, averaged over 5 seeds. BFM-QD variants (FB-ME, FB-CMA-ME, TD-JEPA-CMA-ME) consistently outperform parameter-space baselines (ME, CMA-ME, DCRL-ME, DDE-Elites), with margins that grow substantially in sparse and deceptive tasks (AntMaze-medium, Cube-single) where all parameter-space methods collapse to near-zero performance.

## E.2 Archive Visualization

![](images/f05c20e5784e29685fdd0c585ee02b63398a03739272853e494a59e9a3b4a7e0.jpg)  
Figure 8: Archive coverage. Heatmaps of the descriptor space populated by FB-CMA-ME versus CMA-ME, ME, DCRL-ME, DDE-Elites, and ME+Pretrain, on AntMaze-medium (descriptor: xyposition, fitness: energy efficiency) and Cube-single (descriptor: cube [x, height], fitness: energy efficiency). FB-CMA-ME achieves dense, broad coverage while parameter-space methods leave large regions entirely unpopulated.

Figure 8 visualizes the archive heatmaps for FB-CMA-ME, CMA-ME, ME, DCRL-ME, and DDE-Elites on AntMaze-medium and Cube-single, making the collapse of parameter-space methods concrete. On AntMaze-medium, CMA-ME populates only a small cluster near the origin. On Cube-single, the CMA-ME archive is almost entirely empty, except for moving the cube on the surface of the table, because successfully holding the cube requires a temporally extended sequence of approach, grasp, and transport that random parameter perturbations cannot discover. FB-CMA-ME sidesteps both failure modes entirely, achieving dense and uniform coverage across almost the full descriptor space in both environments.

## E.3 The latent space is behaviorally rich by construction

Figure 9 shows the archive coverage heatmaps obtained by random sampling from the trained FB latent space and the MLP parameter space, across all four environments.

![](images/1179933851c668f994ebeaf19c80cc9a59b46927e6fa4430b3b6d94fefb1705d.jpg)  
Figure 9: Archive coverage from random sampling.

![](images/d6f7fb53f846f173fe5527bcd5f97d30acbb82f996af72456d060c3cebd054dc.jpg)

![](images/163731fb93631839e2facc93939f7b6562ee9235b9ec4db7e5e1389732c3f5e4.jpg)  
Figure 10: BFM-QD does not require high zero-shot performance. (Left) Methods with modest zero-shot returns still attain high QD-scores, suggesting latent space diversity matters more than zero-shot accuracy. (Right) FB-CMA-ME achieves the highest average QD-score, closely followed by BTD-FB-CMA-ME and TD-JEPA-CMA-ME.

## E.4 Effect of the Behavioral Foundation Model

We investigate how the choice of BFM affects QD performance. Different BFMs learn the feature map φ and latent space Z through different objectives, which may yield latent spaces of varying quality and behavioral coverage. Understanding this sensitivity is important for practitioners choosing a BFM backbone for QD search.

Experiment We fix the QD optimizer to CMA-ME and vary the BFM across five methods: Laplacian (Lap) [37], BYOLγ [22], TD-JEPA, BTD-FB [4], and FB. For each BFM, we report its zero-shot performance on the task fitness function alongside the resulting QD-score to assess whether zero-shot performance is a reliable predictor of QD performance.

Results Results are shown in Figure 10. FB achieves the highest average QD-score, closely followed by BTD-FB and TD-JEPA, while Lap and BYOLγ perform worse, particularly on AntMaze-medium and Cube-single. Overall, zero-shot performance is a good predictor of QD-score, and the regression confirms a clear positive correlation. However, on easy tasks, even BFMs with low zero-shot performance can achieve strong QD-scores, suggesting that a highly accurate BFM is not strictly necessary when tasks are dense and simple.

Takeaway: Zero-shot performance is a reliable proxy for QD performance: stronger BFMs consistently yield higher QD-scores. While weaker BFMs can still perform well on easy tasks, the choice of the BFM becomes critical as task difficulty increases.

## E.5 Diversity as an Optimization Strategy

We investigate whether diversity pressure in QD search provides benefits beyond single-objective optimization in the BFM latent space. A natural question is whether simply optimizing for fitness in Z is sufficient, or whether maintaining a diverse archive of solutions actively helps find better optima

Experiment We compare three methods that all operate within the FB latent space. FB Zero-shot uses the closed-form backward inference (Equation 2) to obtain $z ^ { * }$ in a single pass without any additional environment interaction. FB-CMA-ES runs a single CMA-ES [19] optimizer directly in ${ \mathcal { Z } } ,$ optimizing for maximum fitness without diversity pressure. FB-CMA-ME adds QD search on top of the same latent space.

![](images/7cafc32b4093dbd4142e67d4bac2815bf149e14327e62ad42f1c341d42f8cace.jpg)  
Figure 11: Diversity as an optimization strategy. Comparison of FB Zero-shot, FB-CMA-ES (single-objective optimization in Z), and FB-CMA-ME (QD search in Z). FB-CMA-ME outperforms FB-CMA-ES on maximum fitness, demonstrating that diversity pressure helps escape local optima during optimization.

Results Results are shown in Figure 11. FB-CMA-ME outperforms both FB-CMA-ES and FB Zero-shot consistently across all tasks. The gap between FB Zero-shot and FB-CMA-ES confirms that closed-form inference provides a useful initialization but not an optimal solution, due to approximation errors. More strikingly, FB-CMA-ME outperforms FB-CMA-ES even on maximum fitness: by maintaining solutions across diverse behavioral niches, diversity pressure helps escape local optima that a focused single-objective optimizer converges to.

Takeaway: Diversity pressure is not only beneficial for coverage: it also improves maximum fitness by preventing premature convergence to local optima in the latent space.

## E.6 Sensitivity to the Offline Dataset Quality

We investigate how the quality and composition of the offline dataset used to pretrain the BFM affect QD performance. Since the diversity of the latent space is ultimately bounded by the diversity of the training data, dataset composition is a critical factor for BFM-QD practitioners.

Experiment We vary the dataset composition on AntMaze-medium across six conditions: a Biased dataset restricted to the lower-left region of the maze, and five mixtures interpolating between fully random and expert data (Random, 75R/25E, 50R/50E, 25R/75E, Expert). All other hyperparameters are fixed, and the QD-score is averaged across all AntMaze-medium tasks.

![](images/4b02cd965b5754b210664cb1b0babeba097a215b64c8f40b7e8bd5ddb2d7061c.jpg)  
Figure 12: Sensitivity to offline dataset quality. Normalized QD-score on AntMaze-medium for FB-CMA-ME trained on datasets of varying composition: a spatially biased dataset (lower-left region only), and five mixtures of random and expert data. Performance degrades as expert data replaces random data, confirming that behavioral coverage matters more than trajectory quality for BFM-QD. Even the biased dataset substantially outperforms the CMA-ME baseline (dashed line).

Results Results are shown in Figure 12. Performance degrades monotonically as expert data replaces random data: expert trajectories cover only a narrow set of optimal paths, reducing the behavioral diversity of the latent space. This confirms that coverage matters more than trajectory quality for BFM-QD. Notably, even the spatially biased dataset does not collapse and substantially outperforms CMA-ME, suggesting that the BFM learns environment dynamics and motor primitives that generalize beyond the covered region.

Takeaway: Behavioral coverage in the offline dataset matters more than trajectory quality: random exploration datasets provide a richer and more diverse behavioral prior than expert ones. Even a spatially restricted random-exploration dataset yields a functional latent space that supports strong QD performance.

## E.7 Additional Latent-Space Search Baselines

DDE-Elites searches a learned latent space but learns it online from the archive, and ME+Pretrain uses offline data but searches parameter space. We test baselines that combine both, and Policy Manifold Search (PoMS) [31], which runs ME in the latent space of an autoencoder over policy parameters.

Experiment We train three baselines:

• DDE-Elites+Pretrain (TD3) pretrains 10 independent TD3 policies offline on the BFM dataset, using the task fitness as reward (as in ME+Pretrain). A single converged TD3 run collapses to one behavior and gives the VAE nothing to encode diversity from, so we keep 10 intermediate checkpoints per run (100 policies). This population initializes the archive, and we then run DDE-Elites.

• DDE-Elites+Pretrain (BC) follows the same protocol, but policies are pretrained by behavioral cloning on dataset trajectories, since different trajectories represent different behaviors.

• PoMS is closely related to DDE-Elites, as both encode archive solutions with an autoencoder over policy parameters.

We evaluate on Walker and Cube-single over 5 seeds.

Results Table 13 reports normalized QD-score. FB-ME outperforms all baselines on both environments. On Walker, pretraining does not help: DDE-Elites+Pretrain reaches 0.61 (BC) and 0.76 (TD3), versus 0.74 for DDE-Elites, and none of the DDE-Elites or PoMS variants reaches ME. On Cube-single, pretraining raises DDE-Elites from 0.09 to 0.15 (BC) and 0.14 (TD3), which remains far below FB-ME (0.58). PoMS falls below DDE-Elites on both environments (0.63 and 0.07). These baselines imitate trajectories, optimize a single reward, or learn the latent space from the policy parameters in the archive, so their policies are bounded by the dataset, by one objective, or by the archive itself. A BFM instead learns the environment dynamics and trains policies to explore the resulting feature space.

Table 13: Normalized QD-score (mean ± std, 5 seeds).
<table><tr><td>Method</td><td>Walker</td><td>Cube</td></tr><tr><td>ME</td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td></tr><tr><td>DDE-Elites</td><td> $0 . 7 4 \pm 0 . 2 1$ </td><td> $0 . 0 9 \pm 0 . 0 3$ </td></tr><tr><td>ME+Pretrain</td><td> $0 . 8 1 \pm 0 . 1 1$ </td><td> $0 . 0 2 \pm 0 . 0 0$ </td></tr><tr><td>DDE-Elites+Pretrain (BC)</td><td> $0 . 6 1 \pm 0 . 0 9$ </td><td> $0 . 1 5 \pm 0 . 0 4$ </td></tr><tr><td>DDE-Elites+Pretrain (TD3)</td><td> $0 . 7 6 \pm 0 . 1 3$ </td><td> $0 . 1 4 \pm 0 . 0 3$ </td></tr><tr><td>PoMS</td><td> $0 . 6 3 \pm 0 . 1 3$ </td><td> $0 . 0 7 \pm 0 . 0 3$ </td></tr><tr><td>FB-ME (ours)</td><td> $\mathbf { 0 . 9 8 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 5 8 \pm 0 . 1 9 }$ </td></tr></table>

Takeaway: Neither offline pretraining nor a latent space learned over policy parameters is sufficient. The gains come from the structure of the BFM latent space.

## E.8 Backward Inference vs. Sampling Around the Inferred Optimum

BI moves each parent toward $z ^ { * }$ . A simpler alternative is to sample offspring around $z ^ { * }$

Experiment We sample $z ^ { \prime } = z ^ { * } + \epsilon$ with $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ for $\sigma \in \{ 0 . 5 , 1 . 0 \}$ , with $z ^ { * }$ computed from the pretraining dataset to isolate the effect of data quality early in the search. We compare to FB-BI on Walker and Cube-single over 3 seeds.

Results Table 14 reports normalized QD-score. Sampling around a fixed point confines the search to a small region of the latent space with behaviors close to that of $z ^ { * }$ , instead of exploring the whole behavioral space. FB-BI performs far better because each parent keeps its own behavior and moves only a small step toward $z ^ { * }$

Table 14: Normalized QD-score (mean ± std, 3 seeds)
<table><tr><td>Method</td><td>Walker</td><td>Cube</td></tr><tr><td> $z ^ { * } + \epsilon , \sigma = 0 . 5$ </td><td> $0 . 1 6 \pm 0 . 0 3$ </td><td> $0 . 0 9 \pm 0 . 0 1$ </td></tr><tr><td> $z ^ { * } + \epsilon , \sigma = 1 . 0$ </td><td> $0 . 1 8 \pm 0 . 0 2$ </td><td> $0 . 0 9 \pm 0 . 0 2$ </td></tr><tr><td> $\mathrm { F B - B I }$ </td><td> ${ \bf 0 . 9 2 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 9 6 \pm 0 . 0 5 }$ </td></tr></table>

Takeaway: BI works because it moves each solution individually toward $z ^ { * }$ while preserving diversity. Sampling around $z ^ { * }$ does not.

## E.9 Pretraining Data Collected by QD

We use RND to collect pretraining data. QD could also collect it, but this commits the dataset to a descriptor, whereas RND is descriptor-agnostic.

Experiment We retrain FB on data collected by our own FB-QD search, and compare the resulting QD-score to FB trained on RND data. We evaluate on Walker (forward velocity) and AntMaze (top wall).

Results Table 15 shows that QD-collected data reduces performance. Once QD finds a high-fitness elite for a cell, it refines solutions near that elite instead of covering new transitions. RND instead visits states with high prediction error and covers the state space. BFM pretraining needs data that covers the transition dynamics broadly, not only along the descriptor axes.

Table 15: QD-score of FB-ME by pretraining data source (mean ± std).
<table><tr><td>Pretraining data</td><td>Walker</td><td>AntMaze</td></tr><tr><td>RND</td><td> $\mathbf { 1 6 4 7 1 \pm 1 7 5 }$ </td><td> ${ \bf 6 1 \pm 4 5 }$ </td></tr><tr><td>QD (FB-QD)</td><td> $5 6 5 4 \pm 1 2$ </td><td> $4 6 \pm 9$ </td></tr></table>

Takeaway: Broad coverage of the dynamics matters more than diversity along the test descriptor.   
Exploration data beats goal-directed data.

## E.10 Sensitivity to the Backward Inference Step Size

Experiment We run FB-BI on Walker with step size $\alpha \in \{ 0 , 0 . 0 0 5 , 0 . 0 2 , 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 5 , 1 \}$ over 3 seeds. $\alpha = 0$ recovers FB-ME, and $\alpha = 1$ replaces each solution with z\*. Our default is $\alpha = 0 . 0 2$

Results Figure 13 reports normalized QD-score. Performance is stable for $\alpha \in [ 0 . 0 0 5 , 0 . 1 ]$ , and our default lies inside this plateau. Beyond $\alpha = 0 . 1$ it degrades, since larger α pulls offspring toward the single point $z ^ { * }$ and collapses diversity (Section 4.2).

Takeaway: FB-BI is robust to α over a wide range. Large steps collapse diversity.

![](images/65fae354838e6572edfb966111fb3cd9c631862ce7ee29caef74739cb7a9acc6.jpg)  
Figure 13: Normalized QD-score of FB-BI on Walker vs. step size α (mean ± std, 3 seeds).

## E.11 Compute Resources

All experiments were run on NVIDIA H100 GPUs. Table 16 reports wall-clock times per run. BFM pretraining is a one-time cost shared across all tasks within a given environment, so the amortized cost per QD run is dominated by the search itself.

Table 16: Compute cost per run on a single NVIDIA H100 GPU. BFM pretraining is performed once per environment and reused across all tasks and seeds.
<table><tr><td>Phase</td><td>Method</td><td>GPU-hours</td></tr><tr><td rowspan="3">BFM Pretraining (one-time)</td><td>BYOLγ Lap</td><td>2.67</td></tr><tr><td>TD-JEPA FB</td><td>2.67 2.50</td></tr><tr><td>BTD-FB</td><td>2.32 3.17</td></tr><tr><td rowspan="6">QD Search (per run)</td><td>FB-ME</td><td>1.63</td></tr><tr><td>FB-CMA-ME</td><td>1.83</td></tr><tr><td>TD-JEPA-CMA-ME</td><td>1.87</td></tr><tr><td>ME</td><td>1.28</td></tr><tr><td>CMA-ME</td><td>1.47</td></tr><tr><td>DCRL-ME</td><td>2.33</td></tr><tr><td rowspan="3"></td><td></td><td>2.25</td></tr><tr><td>DDE-Elites ME+Pretrain</td><td></td></tr><tr><td></td><td>1.28</td></tr></table>

## E.12 Results Table

Tables 17–20 report the final unnormalized QD-scores (mean ± std over 5 seeds) for all methods across Walker2D, HalfCheetah, AntMaze-medium, and Cube-single.

Table 17: QD-scores on Walker2D (mean ± std over 5 seeds). Bold: best; underline: second best.
<table><tr><td>Method</td><td>Backward Vel</td><td>Forward Vel</td><td>Upright Walk</td><td>Jump</td></tr><tr><td colspan="5">Baselines</td></tr><tr><td>MAP-Elites</td><td> $1 4 8 0 1 { \pm } 5 1 4$ </td><td> $1 5 2 5 3 { \pm } 5 3 8$ </td><td> $1 3 7 4 6 { \pm } 4 9 3$ </td><td> $1 3 7 7 2 { \pm } 4 6 5$ </td></tr><tr><td>CMA-ME</td><td> $1 4 5 8 0 { \pm } 4 8 9$ </td><td> $1 4 1 7 7 { \pm } 5 0 7$ </td><td> $1 3 2 4 1 { \pm } 5 1 1$ </td><td> $1 3 2 6 2 { \pm } 4 8 1$ </td></tr><tr><td>PGA-ME</td><td> $1 2 4 1 0 { \pm } 4 9 1$ </td><td> $1 3 0 8 1 { \pm } 5 3 5$ </td><td> $1 2 1 2 4 { \pm } 4 7 4$ </td><td> $1 2 4 5 6 { \pm } 4 7 5$ </td></tr><tr><td>DCRL-ME</td><td> $1 3 4 1 0 { \pm } 4 1 3$ </td><td> $1 4 0 8 9 { \pm } 4 5 6 $ </td><td> $1 3 7 1 4 { \pm } 4 3 6$ </td><td> $1 3 2 8 6 { \pm } 4 6 2$ </td></tr><tr><td>ME+Pretrain</td><td> $1 3 4 6 5 { \pm } 2 6 8$ </td><td> $1 3 5 7 6 { \pm } 2 6 4$ </td><td> $1 2 3 2 8 { \pm } 2 8 0$ </td><td> $1 2 5 9 5 { \pm } 2 6 8$ </td></tr><tr><td>DDE-Elites</td><td> $1 3 0 9 7 { \pm } 1 0 5$ </td><td> $1 3 7 2 3 { \pm } 1 3 2$ </td><td> $1 0 7 4 2 { \pm } 2 1 0$ </td><td> $9 3 8 8 { \pm } 1 6 5$ </td></tr><tr><td colspan="5"> $B F M { \cdot } Q D \left( o u r s \right)$ </td></tr><tr><td>FB-ME</td><td> $1 5 3 0 7 { \pm } 2 1 9$ </td><td> $1 6 4 7 1 { \pm } 1 7 5$ </td><td> $1 4 5 3 3 { \pm } 1 9 8$ </td><td> $1 4 6 4 1 { \pm } 2 2 8$ </td></tr><tr><td>FB-CMA-ME</td><td> $1 3 6 7 3 { \pm } 1 6 9$   $1 5 3 0 2 { \pm } 2 7 2$ </td><td> $1 5 9 8 6 { \pm } 1 6 9$   $1 5 2 9 1 { \pm } 2 7 7$ </td><td> $1 4 0 5 1 { \pm } 2 3 0$ </td><td> $1 4 1 1 5 { \pm } 1 8 8$ </td></tr><tr><td> $\mathrm { T D \mathrm { - } J E P A \mathrm { - } C M A \mathrm { - } M E }$   $\mathrm { L a p \mathrm { - } C M A \mathrm { - } M E }$ </td><td> $9 3 5 8 { \pm } 1 6 6$ </td><td> $1 0 3 9 1 { \pm } 1 5 4$ </td><td> $1 4 2 6 8 { \pm } 2 2 5$   $9 9 9 4 { \pm } 2 2 7$ </td><td> $1 3 7 4 7 { \pm } 2 2 5$ </td></tr><tr><td> $\mathbf { B } \bar { \mathrm { Y O L } } \gamma \mathbf { - C M A - M E }$ </td><td> $9 4 0 1 { \pm } 1 7 6 $ </td><td> $9 9 3 2 { \pm } 1 8 7$ </td><td> $1 0 2 2 5 { \pm } 2 2 0$ </td><td> $1 0 0 4 3 { \pm } 1 9 2$ </td></tr><tr><td> $\mathbf { B T D - F B - C M A - M E }$ </td><td> $1 6 1 3 5 { \pm } 2 2 0$ </td><td> $1 7 5 4 6 { \pm } 1 8 3$ </td><td> $1 3 3 5 0 { \pm } 2 0 0$ </td><td> $9 4 7 3 { \pm } 2 0 4$ </td></tr><tr><td> $\mathrm { F B - C M A - M E - I s o }$ </td><td> $1 3 4 3 2 { \pm } 1 7 4$ </td><td> $1 7 0 8 7 { \pm } 1 7 5$ </td><td> $1 5 3 6 5 { \pm } 2 2 0$ </td><td> $1 3 3 0 7 { \pm } 2 2 1$ </td></tr><tr><td></td><td> $1 7 8 0 7 { \scriptstyle \pm 2 2 7 }$ </td><td> $\mathbf { 1 8 9 0 8 } { \pm } \mathbf { 1 8 0 }$ </td><td></td><td> $1 4 2 9 1 { \pm } 1 9 4$ </td></tr><tr><td>FB-BI</td><td></td><td></td><td> $1 5 4 7 4 { \pm } 1 8 8$ </td><td> $1 5 3 2 8 { \pm } 2 2 5$ </td></tr><tr><td> $\mathrm { F B - P G A - M E }$ </td><td> $1 6 2 1 2 { \pm } 2 1 8$ </td><td> $1 7 9 0 6 { \pm } 1 8 6 $ </td><td> $\overline { { 1 6 3 6 8 \pm 1 8 7 } }$ </td><td> $\overline { { { \bf 1 6 7 1 4 } \pm 2 2 5 } }$ </td></tr></table>

Table 18: QD-scores on HalfCheetah (mean ± std over 5 seeds). Bold: best; underline: second best.
<table><tr><td>Method</td><td>Backward Run</td><td>Backward Walk</td><td>Forward Walk</td><td>Forward Run</td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>MAP-Elites</td><td> $1 1 3 3 1 \pm 5 2 4$ </td><td> $1 1 8 7 4 { \pm } 5 3 4$ </td><td> $1 1 4 0 2 { \pm } 4 9 0$ </td><td> $1 1 2 3 0 { \pm } 5 2 0$ </td></tr><tr><td>CMA-ME</td><td> $1 2 3 8 1 { \pm } 5 0 7$ </td><td> $1 3 2 0 0 { \pm } 5 1 4$ </td><td> $1 3 7 2 4 { \pm } 4 6 9$ </td><td> $1 2 3 7 1 { \pm } 5 3 4$ </td></tr><tr><td>PGA-ME</td><td> $1 0 6 8 5 { \pm } 4 6 3$ </td><td> $1 4 3 2 6 { \pm } 4 7 3$ </td><td> $1 2 5 2 3 { \pm } 4 7 8$ </td><td> $1 2 6 3 6 { \pm } 4 9 8$ </td></tr><tr><td>DCRL-ME</td><td> $1 2 4 6 6 { \pm } 4 6 0$ </td><td> $\overline { { 1 3 1 0 3 \pm 4 7 9 } }$ </td><td> $1 4 0 8 2 { \pm } 4 5 6$ </td><td> $1 2 3 2 4 { \pm } 4 7 2$ </td></tr><tr><td>ME+Pretrain</td><td> $1 2 3 0 2 { \pm } 2 9 1$ </td><td> $1 3 2 9 3 { \pm } 3 3 8$ </td><td> $1 3 6 1 8 { \pm } 3 1 5$ </td><td> $1 2 4 2 5 { \pm } 3 0 9$ </td></tr><tr><td>DDE-Elites</td><td> $7 9 3 1 { \pm } 1 2 0$ </td><td> $8 3 1 1 \pm 1 2 5$ </td><td> $7 9 8 1 { \pm } 2 1 6$ </td><td> $7 8 6 0 { \pm } 1 9 8$ </td></tr><tr><td> $B F M { \cdot } Q D \left( o u r s \right)$ </td><td></td><td></td><td></td><td></td></tr><tr><td>FB-ME</td><td> $1 1 5 8 9 { \pm } 2 2 8$ </td><td> $1 1 2 4 6 { \pm } 2 2 8$ </td><td> $1 1 7 8 1 { \pm } 2 2 4$ </td><td> $1 1 5 9 0 { \pm } 2 3 9$ </td></tr><tr><td>FB-CMA-ME</td><td> $1 2 5 7 4 { \pm } 1 6 5$ </td><td> $1 3 6 1 7 { \pm } 2 0 1$ </td><td> $1 4 0 9 6 { \pm } 2 0 3$ </td><td> $1 2 4 7 0 { \pm } 2 2 3$ </td></tr><tr><td> $\mathrm { T D \mathrm { - } J E P A \mathrm { - } C M A \mathrm { - } M E }$ </td><td> $1 2 5 0 2 { \pm } 2 3 4$ </td><td> $1 3 3 8 7 { \pm } 2 2 9$ </td><td> $1 4 0 0 2 { \pm } 2 2 0$ </td><td> $1 2 4 9 5 { \pm } 2 8 2$ </td></tr><tr><td> $_ \mathrm { L a p - C M A - M E }$ </td><td> $7 3 4 1 { \pm } 1 5 6$ </td><td> $8 0 5 1 { \pm } 1 8 9$ </td><td> $9 2 8 1 { \pm } 1 8 4$ </td><td> $7 9 5 2 { \pm } 2 3 2$ </td></tr><tr><td> $\mathbf { B Y O L } \gamma \mathbf { - C M A - M E }$ </td><td> $9 0 5 5 { \pm } 1 5 9$ </td><td> $9 8 4 9 { \pm } 2 1 5$ </td><td> $1 1 2 1 4 { \pm } 1 9 5$ </td><td> $8 5 5 7 { \pm } 2 3 9$ </td></tr><tr><td> $\mathbf { B T D - F B - C M A - M E }$ </td><td> $1 0 5 7 8 { \pm } 2 2 0$ </td><td> $1 0 4 8 8 { \pm } 2 2 0 $ </td><td> $1 0 5 7 3 { \pm } 2 1 7$ </td><td> $1 1 3 9 4 \pm 2 3 0$ </td></tr><tr><td> $\mathrm { F B - C M A - M E - I s o }$ </td><td> $\mathbf { 1 4 3 1 5 } \pm \mathbf { 1 6 5 }$ </td><td> $\mathbf { 1 4 8 3 9 } \pm 2 \mathbf { 0 0 }$ </td><td> $\mathbf { 1 4 7 4 5 } { \pm 2 1 2 }$ </td><td> $1 3 2 3 2 { \pm } 2 3 0 $ </td></tr><tr><td> $\mathrm { F B - B I }$ </td><td> $1 2 8 9 0 { \pm } 2 3 3$ </td><td> $1 2 5 1 4 { \pm } 2 2 1$ </td><td> $1 2 3 6 0 { \pm } 2 2 4$ </td><td> $\overline { { 1 3 0 7 7 } } \pm 2 4 7$ </td></tr><tr><td> $\mathrm { F B - P G A - M E }$ </td><td> $1 4 1 0 5 { \pm } 2 3 3 $ </td><td> $1 2 5 2 0 { \pm } 2 1 9$ </td><td> $1 2 9 6 0 { \pm } 2 2 3$ </td><td> $\mathbf { 1 3 7 4 5 } \pm 2 4 0$ </td></tr></table>

Table 19: QD-scores on AntMaze-Medium (mean ± std over 5 seeds). Bold: best; underline: second best. SuperscriptD denotes the dense-reward variant.
<table><tr><td>Method</td><td>Energy Eff.</td><td>Top Wall</td><td>Right Wall</td><td>Reach Center</td><td> $\mathbf { T o p \ W a l l } ^ { \mathbf { D } }$ </td><td> $\mathbf { R i g h t W a l l } ^ { \mathbf { D } }$ </td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAP-Elites</td><td>6±0</td><td>0±0</td><td>0±0</td><td>0±0</td><td> $2 3 { \pm } 0$ </td><td>14±0</td></tr><tr><td>CMA-ME</td><td>6±0</td><td>0±0</td><td>0±0</td><td>0±0</td><td> $2 9 3 { \pm } 0$ </td><td> $1 7 5 { \pm } 0$ </td></tr><tr><td>PGA-ME</td><td>22±2</td><td>0±0</td><td>0±0</td><td>0±0</td><td> $7 6 { \pm } 0$ </td><td> $5 { \pm } 1$ </td></tr><tr><td>DCRL-ME</td><td>61±0</td><td>0±0</td><td>0±0</td><td>0±0</td><td> $1 5 3 { \pm } 0$ </td><td> $1 7 { \pm } 0$ </td></tr><tr><td>ME+Pretrain</td><td>5±0</td><td>0±0</td><td>0±0</td><td>0±0</td><td> $2 0 { \pm } 0$ </td><td> $1 3 { \pm } 0$ </td></tr><tr><td>DDE-Elites</td><td>509±0</td><td>14±3</td><td>7±4</td><td>0±0</td><td> $_ { 8 \pm 1 }$ </td><td> $7 { \pm } 4$ </td></tr><tr><td>BFM-QD (ours)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FB-ME</td><td> $3 4 0 6 { \pm } 5 2$ </td><td> $6 1 \pm 4 5$ </td><td> $4 4 6 { \pm } 5 1$ </td><td> $1 6 1 \pm 5 1$ </td><td> $4 4 1 \pm 5 2$ </td><td>1018±57</td></tr><tr><td>FB-CMA-ME</td><td> $3 6 5 7 { \pm } 5 1$ </td><td> $6 1 1 { \pm } 5 0 $ </td><td> $4 9 8 \pm 4 9$ </td><td> $5 5 2 \pm 4 9$ </td><td> $9 0 9 \pm 4 9$ </td><td>1099±49</td></tr><tr><td>TD-JEPA-CMA-ME</td><td> $3 4 0 7 { \pm } 8 8$ </td><td> $\overline { { 4 3 6 \pm 1 0 } }$ </td><td> $5 6 5 { \pm } 5 6 $ </td><td> $1 9 2 \pm 2 1$ </td><td> $6 3 8 { \pm } 6 6$ </td><td>900±50</td></tr><tr><td> $_ \mathrm { L a p - C M A - M E }$ </td><td> $2 7 4 \pm 6 0$ </td><td> $1 4 \pm 3 7$ </td><td> $\overline { { 1 0 5 \pm 5 3 } }$ </td><td> $1 4 \pm 4 2$ </td><td> $9 4 \pm 4 6$ </td><td>133±66</td></tr><tr><td> $\mathbf { B Y O L } \gamma \mathbf { - C M A - M E }$ </td><td> $9 4 7 { \pm } 5 4 $ </td><td> $1 0 { \pm } 5 1$ </td><td> $5 9 2 5 1$ </td><td> $3 6 \pm 5 0$ </td><td> $6 4 \pm 5 2$ </td><td>151±58</td></tr><tr><td> $\mathbf { B T D - F B - C M A - M E }$ </td><td> $3 6 3 7 { \pm } 5 5$ </td><td> $5 2 8 \pm 4 8$ </td><td> $4 9 3 { \pm } 4 2$ </td><td> $4 7 0 \pm 3 9$ </td><td> $7 8 3 \pm 5 7$ </td><td>1098±46</td></tr><tr><td> $\mathrm { F B - C M A - M E - I s o }$ </td><td> $3 6 0 3 { \pm } 4 4$ </td><td> ${ \bf 6 6 7 \pm 5 8 }$ </td><td> ${ \pm } 6 9 \pm 5 5$ </td><td> ${ \bf 6 4 1 \pm 5 6 }$ </td><td> $\mathbf { 1 0 4 0 } { \pm } 4 3$ </td><td> $1 2 0 8 { \pm } 4 0 $ </td></tr><tr><td>FB-BI</td><td> $\mathbf { 4 0 7 4 } \pm \mathbf { 6 0 }$ </td><td> $7 0 \pm 4 8$ </td><td> $5 1 0 { \pm } 5 2 $ </td><td> $1 6 3 { \pm } 4 4$ </td><td> $4 5 7 { \pm } 4 7$ </td><td> $\overline { { 1 1 8 5 \pm 5 4 } }$ </td></tr><tr><td> $\mathrm { F B - P G A - M E }$ </td><td></td><td> $6 3 \pm 4 3$ </td><td> $5 1 4 \pm 4 5$ </td><td> $1 7 6 { \pm } 4 3$ </td><td> $4 7 5 { \pm } 4 9$ </td><td></td></tr><tr><td></td><td> $3 8 9 6 { \pm } 6 6 $ </td><td></td><td></td><td></td><td></td><td>1278±57</td></tr></table>

Table 20: QD-scores on Cube-Single (mean ± std over 5 seeds). Bold: best; underline: second best.
<table><tr><td>Method</td><td>Energy Eff.  $( x , y )$ </td><td>Energy Eff. (x, z)</td><td>Wrist Config</td><td>Grasp Style</td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>MAP-Elites</td><td>0±0</td><td>2±0</td><td>305±0</td><td>6±0</td></tr><tr><td>CMA-ME</td><td>4±0</td><td>2±0</td><td> $4 4 3 { \pm } 0$ </td><td>29±0</td></tr><tr><td>PGA-ME</td><td>0±0</td><td>89±0</td><td> $6 8 0 { \pm } 5$ </td><td>8±1</td></tr><tr><td>DCRL-ME</td><td>0±0</td><td> $1 4 1 { \pm } 0$ </td><td> $5 2 1 { \pm } 0$ </td><td>6±0</td></tr><tr><td>ME+Pretrain</td><td>3±0</td><td>1±0</td><td> $1 5 2 { \pm } 0$ </td><td>4±0</td></tr><tr><td>DDE-Elites</td><td>875±4</td><td>174±0</td><td> $_ { 0 \pm 0 }$ </td><td>0±0</td></tr><tr><td>BFM-QD (ours)</td><td></td><td></td><td></td><td></td></tr><tr><td>FB-ME FB-CMA-ME</td><td> $2 4 8 6 { \pm } 8 3$   $5 3 6 8 { \pm } 1 0 6 $ </td><td> $4 2 4 0 { \pm } 8 9$   $5 0 9 5 { \pm } 9 3$ </td><td> $1 7 8 3 { \pm } 1 0 5$ </td><td>229±103</td></tr><tr><td> $\mathrm { T D \mathrm { - } J E P A \mathrm { - } C M A \mathrm { - } M E }$ </td><td> $\overline { { 5 0 1 8 \pm 1 2 1 } }$ </td><td> $3 8 8 8 { \pm } 1 0 6 $ </td><td> $\mathbf { 1 9 5 3 \pm 1 2 0 }$ </td><td>1073±111</td></tr><tr><td></td><td></td><td></td><td> $1 6 9 0 { \pm } 8 7$ </td><td> $2 9 0 \pm 1 4 4$ </td></tr><tr><td> $\mathrm { L a p \mathrm { - } C M A \mathrm { - } M E }$ </td><td> $6 1 3 { \pm } 7 3$ </td><td> $4 0 7 { \pm } 9 3$ </td><td> $3 2 0 { \pm } 1 1 4$ </td><td> $1 7 { \pm } 1 0 0 \ $ </td></tr><tr><td> $\mathbf { B Y O L } \gamma \mathbf { - C M A - M E }$ </td><td> $6 5 7 { \pm } 7 7$ </td><td> $1 1 1 5 { \pm } 9 6$ </td><td> $3 1 7 { \pm } 1 0 9 $ </td><td> $5 7 { \pm } 1 1 0 $ </td></tr><tr><td>BTD-FB-CMA-ME</td><td> $5 1 9 2 { \pm } 1 0 7$ </td><td> $4 9 6 3 { \pm } 8 4$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td> $1 9 0 6 { \pm } 1 1 3$ </td><td> $9 1 9 { \pm } 1 1 0 $ </td></tr><tr><td> $\mathrm { F B - C M A - M E - I s o }$ </td><td> ${ \pm } 9 6 5 { \pm } 1 1 4$ </td><td> ${ \bf 5 9 5 7 \pm 9 3 }$ </td><td> $1 9 4 5 { \pm } 1 2 1$ </td><td> ${ \bf 1 } 2 2 7 { \pm } { \bf 1 0 8 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $2 5 5 { \pm } 9 3$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $2 7 8 7 { \pm } 8 0$ </td><td> $4 8 9 2 { \pm } 8 0$ </td><td></td><td></td></tr><tr><td>FB-BI</td><td></td><td></td><td> $1 9 2 8 { \pm } 9 7$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { F B - P G A - M E }$ </td><td> $2 8 3 5 { \pm } 7 4$ </td><td> $4 8 3 2 \pm 7 3$ </td><td> $1 8 8 7 { \pm } 9 4 $ </td><td> $2 5 5 { \pm } 1 0 1$ </td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately state all contributions. Claims are matched by the experiments in Section 5.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: Section 6 explicitly discusses three limitations: (i) requirement of a diverse offline dataset; (ii) performance degradation with narrower datasets; (iii) the frozen BFM imposes a hard ceiling on behavioral expressivity.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations" section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: See 4.2, and a complete proof is provided in A.

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

Justification: Appendix D provides full experimental details: environments, datasets, archive configuration, evaluation budget, MLP architecture, latent dimension, all hyperparameters, number of seeds, and the variation operators.

Guidelines:

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

## Answer: [No]

Justification: All datasets used are publicly available, and Appendix D provides sufficient detail for reimplementation.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https: //neurips . cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https : //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: All training and evaluation details are reported in Appendix D.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All results are averaged over 5 (3 in two additional experiments) independent random seeds. Shaded regions in learning curves and error bars in bar plots denote standard deviation across seeds.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors)

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Compute resources mentioned in the Appendix E.11

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research involves only simulated benchmarks. No human subjects, sensitive data, or dual-use concerns are present

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: This is foundational research on reinforcement learning algorithms using simulated environments. No involvement of real deployments, sensitive data, or high-risk technology.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: All datasets used are existing public benchmarks. No safeguards are required. Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All assets (libraries and baselines) are properly cited.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode. com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The contribution is a framework and algorithmic approach.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper involves no crowdsourcing and no research with human subjects.   
All experiments are conducted entirely in simulation.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: No human subjects are involved. IRB approval is not required.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: No LLMs needed in our research.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.