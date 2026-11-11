# Recurrent Modular Structure in Spiking Neural Network Reinforcement Learning

Vikram Ramavarapu   
Siebel School of Computing   
and Data Science   
University of Illinois   
Urbana-Champaign   
Champaign, IL, USA   
vikramr2@illinois.edu   
Himanshu Udupi   
Siebel School of Computing   
and Data Science   
University of Illinois   
Urbana-Champaign   
Champaign, IL, USA   
hudupi2@illinois.edu   
Ashish Gautam   
Computer Science and   
Mathematics Division   
Oak Ridge National Laboratory   
Oak Ridge, TN, USA   
gautama@ornl.gov

![](images/53c61004b915a20bfc3c0795a697f768771c8c8213d226708dd3896b1df9d381.jpg)  
Fig. 1. Concept overview. (a) A feed-forward network passes information layer by layer and is usually trained by backpropagation. Each action is a singl output neuron. (b) A community recurrent SNN (CoRe SNN) organises neurons into communities with distinct roles instead of layers. Each action is an entir community, and information flows through recurrent connections within and between communities. (c–e) Three ways to obtain the communities: (c) dynamic modularity, where Map-STDP encourages modular structure during learning, (d) partial modularity, where a stochastic block model (SBM) initializatio imposes it partially, and (e) full modularity, where sealing imposes it fully. (f) The full model with CoRe inserted on a TD-STDP actor-critic backbone. An interactive visualization of the CoRe SNN variants is at https://vikramramavarapu.org/CoRe.

Abstract—Biological neural circuits are both modular and recurrent, with dense communities sparsely linked to one another. Spiking neural networks (SNNs) for reinforcement learning are usually feed-forward, and it is unclear how to obtain functional modules in a recurrent SNN or what they buy. In this work, we introduce a class of spiking networks, called Community Recurrent SNNs (CoRe SNNs): recurrent networks partitioned into communities with fixed roles. A controller community receives the stimulus, each action has its own community, and latent communities carry internal state. Spiking activity is then a walk between communities, and the action is the community the walk reaches. All learning is local and three-factor, and the recurrent communities are closer to cortical microcircuits than a feed-forward actor. We compare three regimes for obtaining the communities: dynamic modularity, in which a new learning rule we introduce, Map-STDP, encourages modular structure during learning; partial modularity, in which a stochastic block model (SBM) initialization enforces it partly; and full modularity, in which sealing enforces it fully. On CartPole, across three spiking neuron models, Map-STDP improves learning over the same network without it (+65 to +87 mean return over training), matches full modularity (sealed communities) on two of the three models, and makes the observed spike walk more compressible. Replacing the feed-forward actor of a TD-STDP actor-critic with recurrent action communities improves on the feed-forward network on CartPole (+25 to +45) and LunarLander (+36 to +41 final return), and matches it on Acrobot. How strongly to enforce modularity depends on the task: full modularity helps most on Acrobot, while on LunarLander the dynamic, partial and full regimes perform alike. Overall, these recurrent modular networks can match or even improve upon feed-forward approaches, with improvements in bio-plausibility and energy efficiency.

Index Terms—spiking neural networks, reinforcement learning, spike-timing-dependent plasticity, three-factor learning, modular networks, map equation, community detection, neuromorphic computing

## I. INTRODUCTION

The cortex is organized into columns [27], and its wiring forms densely connected communities that are sparsely linked to one another [28], [29]. Modularity lets a circuit divide its work among parts that can be specialized and reused. Spiking neural networks (SNNs) are attractive models of such circuits and, on neuromorphic hardware, an energy-efficient way to run them [2], [3]. Yet SNNs for reinforcement learning (RL) are usually feed-forward stacks of layers, trained by backpropagation or by local reward-modulated rules [39]. It is unclear how a recurrent SNN could acquire functional modules using only local plasticity, or what those modules would buy.

A guiding principle comes from compression. Hinton and van Camp argued that a good network is a simple one and trained networks by minimizing the description length of their weights [25]. Efficient-coding theories make a similar claim for neural activity: neural systems should transmit information at low cost [35], and the same accuracy–complexity tradeoff underlies the free-energy principle [36]. As such, the map equation gives the description length of a random walk on a graph split into communities [24], and it is shortest when the communities contain the walk. Spiking activity is a walk over synapses, so a network whose activity is compressible has activity confined within modules and sparse traffic between them.

This view recalls the self-organizing map (SOM) [47], [48], the classic example of structure that emerges from local, unsupervised learning. Both approaches replace a global error signal with a local rule whose collective effect is organization. They differ in what is organized and why. A SOM arranges a layer of prototype units on a fixed lattice so that it preserves the topology of the input. Our network organizes its own recurrent connectivity into communities with distinct roles, using spikes, a reward-modulated rule, and the description length of its activity as the organizing objective. We see this as a modern descendant of the SOM idea, brought to recurrent SNNs, reinforcement learning, and information-theoretic compressibility.

We introduce Community Recurrent SNNs (CoRe SNNs), a class of recurrent networks partitioned into communities with fixed roles: a controller community that receives the stimulus, one community per action, and latent communities that carry internal state. We make four contributions.

• CoRe SNNs. Spiking activity is a walk between communities, and the action is the community the walk reaches.

• Map-STDP. A local, three-factor learning rule that encourages the communities, and lowers description length during learning.

• Modularity regimes. A comparison of dynamic, partial and full modularity across four neuron engines.

• A drop-in actor. CoRe SNNs replace the feed-forward actor of a TD-STDP actor–critic and are evaluated on CartPole, Acrobot and LunarLander, together with a sweep of cross-module connectivity.

## II. BACKGROUND

## A. Spiking Neural Networks

SNNs communicate through discrete spike events, enabling sparse, temporally-coded computation that maps naturally to neuromorphic hardware [1]. This makes them a closer model of cortical computation than conventional artificial neural networks. Because computation is event-driven, a neuron consumes energy mainly when it spikes or receives a spike, and can therefore run at a fraction of the energy of dense matrix multiplication on digital neuromorphic chips [2], [3] and on analog memristive crossbars. In a crossbar, each synapse is a conductance, and multiply–accumulate happens in place [4]. These savings depend on activity staying sparse [5]. Training SNNs is difficult because the spike function is non-differentiable. Surrogate-gradient backpropagation through time (BPTT) [6] works well but is non-local, needing weight transport and the full activity history, which have no known biological counterpart [7] and are hard to implement with the local plasticity that neuromorphic hardware supports, so such networks are typically trained offline and then deployed [3], [6].

Spike-timing-dependent plasticity (STDP) is a local alternative to BPTT. A synapse is potentiated when a presynaptic spike shortly precedes a postsynaptic spike and depressed when the order is reversed [8], [9]. The update depends only on quantities available at the synapse, and the pairing of pulses across a device maps directly onto memristive hardware [4]. STDP alone is unsupervised, however. STDP alone is unsupervised: it strengthens whatever correlations the input contains. In networks that respect Dale’s law, where each neuron is either excitatory or inhibitory, it can still learn useful features, for example digit prototypes shaped by lateral inhibition [10]. It cannot, however, tell behaviorally useful correlations from irrelevant ones, so on its own it cannot learn a task defined by reward.

Three-factor rules add a global modulatory signal to the two local factors of STDP [30], [31]. Rewards arrive seconds after the spikes that caused them, far outside the tens-ofmilliseconds STDP window. Each synapse therefore keeps an eligibility trace $e _ { i j }$ . The trace accumulates pre–post coincidences and decays with a time constant matched to the delay of the outcome. A broadcast third factor $M ( t )$ then turns the trace into a weight change:

$$
\dot { e } _ { i j } = - \frac { e _ { i j } } { \tau _ { e } } + \mathrm { S T D P } _ { i j } ( t ) , \qquad \Delta W _ { i j } \propto M ( t ) e _ { i j } ( t ) .
$$

Chung and Kozma [39] introduced a three-factor rule in which the temporal-difference (TD) reward prediction error $\delta = r + \gamma V ( s ^ { \prime } ) - V ( s )$ [32] is the third factor. In a spiking actor–critic network, this TD-modulated STDP was applied to standard control tasks such as CartPole and LunarLander.

## B. Neuron Models

We use three spiking neuron models, which we call engines. In every engine, a spike of presynaptic neuron j gives

postsynaptic neuron i the output-normalized kick

$$
b _ { i j } = \mathrm { s g n } _ { j } w _ { 0 } \frac { W _ { i j } } { d _ { j } } , \qquad d _ { j } = \sum _ { k } W _ { k j } .\tag{1}
$$

1) Branching Neuron Model: The branching model is the standard neuron model of the SuperNeuroMAT simulator [11] and follows the branching-process description of cortical activity [12]. The membrane forgets after every step, so one step is one hop of the walk:

$$
\begin{array} { c } { { \displaystyle u _ { i } ( t + 1 ) = \sum _ { j } b _ { i j } x _ { j } ( t ) + 2 \theta \mu _ { i } ( t ) , } } \\ { { \displaystyle x _ { i } ( t + 1 ) \sim \mathrm { B e r n o u l l i } \left( \mathrm { c l i p } \left( \frac { u _ { i } ( t + 1 ) } { 2 \theta } , 0 , 1 \right) \right) , } } \end{array}\tag{2}
$$

where $x _ { i } ( t ) \in \{ 0 , 1 \}$ is the spike indicator, θ the threshold and $\mu _ { i }$ the external (stimulus) drive.

2) Leaky Integrate-and-Fire (LIF): The LIF neuron [13] is driven by background noise as in Brunel [14]:

$$
\begin{array} { c } { { \displaystyle \tau _ { m } \frac { d V _ { i } } { d t } = - ( V _ { i } - E _ { L } ) + \mu _ { i } ( t ) + \sigma _ { V } \sqrt { 2 \tau _ { m } } \xi _ { i } ( t ) } } \\ { { + \tau _ { m } \displaystyle \sum _ { j } b _ { i j } \sum _ { f } \delta \left( t - t _ { j } ^ { f } \right) , } } \end{array}\tag{3}
$$

with a spike and reset to $V _ { r }$ when $V _ { i }$ reaches $V _ { t h }$ . Here $\xi _ { i }$ is unit white noise standing for background input, and $t _ { j } ^ { f }$ are the spike times of $j .$

3) Quadratic Integrate-and-Fire (QIF): The QIF neuron [15] is written in the normalization of Montbrio et al. [16]:´

$$
\begin{array} { c } { { \displaystyle \tau \frac { d V _ { i } } { d t } = V _ { i } ^ { 2 } + \eta _ { i } + I _ { i } ^ { e x t } ( t ) + \sigma \sqrt { \tau } \xi _ { i } ( t ) } } \\ { { + \tau \displaystyle \sum _ { j } b _ { i j } \sum _ { f } \delta \left( t - t _ { j } ^ { f } \right) , } } \end{array}\tag{4}
$$

with a spike at $V _ { i } = + V _ { p e a k }$ and reset $t 0 ~ - V _ { p e a k }$ . Here $\eta _ { i }$ is the excitability.

## C. Random Walks, Information Theory, and Communities

A random walk on a weighted directed graph is a process in which a walker at node j moves to node i with probability proportional to the weight of the edge $j \to i$ . It is a Markov chain with transition probabilities $T _ { i j } = W _ { i j } / \sum _ { k } W _ { k j }$ , and over time the fraction of steps spent at each node converges to a stationary distribution π that measures how much flow each node carries. This is the idea behind PageRank [17]. In a graph with communities, a walker that has entered a densely connected group tends to stay there for many steps before crossing a sparse link, so the walk’s behavior reveals the community structure. In a spiking network, a natural walk follows spikes forward from each neuron to the neurons it excites.

Info-theoretic entropy can be measured by the description length of a random walk, following the core principle of information theory: the more predictable a process, the shorter its description. The map equation [19] measures the description length of a random walk on a graph partitioned into communities:

$$
L ( M ) = q _ {  } H ( \mathcal { Q } ) + \sum _ { i = 1 } ^ { m } p _ { \odot } ^ { i } H ( \mathcal { P } ^ { i } ) .\tag{5}
$$

Each step of the walk is given a binary codeword, shorter for more frequent steps, and the entropy of a codebook is its average codeword length in bits. The map equation uses a twolevel code. An index codebook Q names the module entered at each exit, which happens with probability $q _ { \curvearrowright }$ per step. Each module i has its own codebook ${ \mathcal { P } } ^ { i }$ for naming the nodes inside it, used at rate $p _ { \odot } ^ { i }$ . Leaving a community costs extra bits, so the map equation is shorter for partitions that capture the graph’s community structure. The partition that minimizes $L ( M )$ is the one that best compresses the walk and reveals the communities, hence making the map equation the objective function of the community detection algorithm, Infomap [24]. The map equation has been used to detect communities in networks from social [19], biological [20], and to study the modularity of cortical networks [29].

A basic model that generates networks with community structure is the stochastic block model (SBM) [21]. In an SBM, each node belongs to one of K communities, and the probability of an edge between two nodes depends on a $K \times K$ matrix of connection probabilities. The SBM can generate networks with strong communities, weak communities, or no communities at all, depending on the choice of the probability matrix.

## D. Other Graph-Theoretic Approaches to SNNs

Understanding how the structure of a network shapes its dynamics in the brain has been a long standing question in the neuroscience community [49]. For threshold-linear rate networks it has a concrete answer. A CTLN takes a directed graph, assigns the weights from it, and its fixed point supports, stable states and dynamic attractors are then determined by the graph [50]–[53]. Brief external kicks switch the network from one attractor to the next, and this has been enough to assemble counters, sequences and gait controllers by hand [54], [55]. Our SNN work sits next to this. We ask the same thing of a spiking network, and in both settings the answer takes the same shape, a collection of attractors pinned down by connectivity plus a set of input-triggered jumps between them, which is also how a finite state machine lives inside a recurrent network [61].

Part of the CTLN picture has been shown to survive the move to spikes, but only in a restricted setting. In clustered E-I networks of Hawkes neurons the excitatory rates obey a CTLN in the mean-field limit, and the CTLN fixed points become metastable states of the finite network [59]. CTLNs also come with a modular story, where the attractors of a graph assembled from smaller pieces are inherited from the pieces [56], [57], though no such result yet exists for spiking networks. Whether either property carries over to the SNNs we study is open. If it does, it would suggest a use for these rules in learning. A reward machine turns a history-dependent RL task into one with a finite set of states [60], and if those states could be planted as the attractors of a glued graph, a spiking network started from that graph would have the states in place and would only need to learn the jumps [62]. Some known initialisations of structure are detailed below.

Columnar organization has also been used for supervised learning in spiking networks. CoLaNET [22] gives each target class its own column, built from several layers of neurons and sub-class microcolumns, and trains it with local anti-Hebbian and dopamine-modulated plasticity. It was evaluated on classifying MNIST images [23] and Pong states by their proximity to reward, a task derived from model-based reinforcement learning. Its signal path is feed-forward, with only local winner-take-all competition and gating inside each column. The columns are fixed by design as class-specific classifiers, and the network evaluates states without selecting actions. In contrast, our work learns recurrent modular structure with Map-STDP and studies the effect of modularity on reinforcement learning.

## III. MAP-STDP LEARNING RULE

Community detection solves a forward problem: given a graph, find the partition into communities that minimizes the description length of a random walk on it, the map equation [24]. Map-STDP solves the inverse problem. The partition is fixed in advance (one controller module, one module per action, and a few latent modules), and plasticity shapes the synapses until that partition is the minimum-description-length one. This section states the rule and what each part does. The derivation from the map equation is in Appendix B.

## A. Ingredients

Each synapse from neuron $j$ to neuron i has a magnitude $W _ { i j } \ \geq \ 0$ and the sign of its presynaptic type, $\mathrm { s g n } _ { j } = \pm 1$ Each neuron belongs to one module $m ( j ) { \mathrm { ; } }$ the controller ${ \mathcal { C } } ,$ an action module in A, or a latent module in ${ \mathcal { L } } .$

• The walk. A step of the walk from $j$ to i is a spike of i within a short window after a spike of $j .$ Its flow $J _ { i j }$ , coarse-grained to modules, is a small Markov chain over the network’s own states. The policy is the flow into the action modules, $P ( a \mid \mathbf { o } _ { f } ) \propto p _ { a } ( f ) ^ { \beta }$ , where $\mathbf { o } _ { f }$ is the stimulus at frame $f , p _ { a } ( f )$ is the flow into action module a in that frame, and $\beta$ is a temperature parameter.

• Description length. $D ( W ) ~ = ~ \mathbb { E } _ { f } [ L ( J _ { f } ) ]$ is the code length of the walk averaged over frames. Leaving module m costs $M _ { m } ^ { * } = \partial L / \partial q _ { m } \geq 0$ bits per unit of exit flow $q _ { m }$ . It is computed from $K$ module-level statistics and broadcast to module m: it is the rule’s neuromodulator.

• Signed counter. $c _ { i j }$ is the change in i’s spikes, within the window, caused by $j ^ { \flat } \boldsymbol { \mathrm { s } }$ kicks alone. It is the local pre-synaptic activity times post-synaptic sensitivity, and carries sgn<sub>j</sub>.

## B. The rule

The per-frame update of each synapse is

$$
\begin{array} { r l } & { \Delta W _ { i j } = \underbrace { \underset { \mathrm { t a k } } { \eta \delta _ { f } e _ { i j } } } _ { \mathrm { t a k } } } \\ & { \qquad - \underbrace { \lambda M _ { m ( j ) } ^ { * } \left( \mathbb { I } [ i \notin m ( j ) ] - \bar { e } _ { j } ^ { o b s } \right) c _ { i j } } _ { \mathrm { s t u c t u e } } } \\ & { \qquad - \underbrace { \epsilon _ { h } ( d _ { j } - 1 ) \frac { \pi _ { j } W _ { i j } } { d _ { j } } } _ { \mathrm { h o m e o s t a s i s } } . } \end{array}\tag{6}
$$

The rule has three terms:

• Task. $\delta _ { f }$ is a global TD error and $e _ { i j }$ an action-gated eligibility trace built from $c _ { i j } \ ( { \mathrm { E q . } } \ ( 2 4 ) )$

• Structure (the map term). $\bar { e } _ { j } ^ { o b s }$ is the share of $j ^ { \circ } \mathbf { s }$ observed transitions that leave its module. Transitions that leave the module are pushed down when the module leaks, and those that stay are pushed up, scaled by $M _ { m ( j ) } ^ { * } .$

• Homeostasis. $d _ { j }$ is $j ^ { \dagger } \mathbf { s }$ total output weight, held near 1. It rescales a column uniformly, so it does not change the walk.

λ sets the strength of the map term. In partial and full modularity, $\lambda = 0$ and the map term is not applied. In dynamic modularity, $\lambda > 0$ and the map term is applied. Inhibitory synapses reverse the signs of the structural term, through sgn<sub>j</sub> in $c _ { i j }$ . On average the structural term is gradient descent on $D ,$ up to a rescaling that cannot change D (Appendix B-F). The controller’s synapses onto the action and latent modules carry the policy, so they are exempt from the map term (Appendix B-G).

## C. Three regimes of modularity

The experiments compare three regimes:

• partial modularity: $\lambda = 0$ on a network initialized with SBM connectivity $( p _ { c r o s s } = 0 . 0 5 )$ , so communities are only partly enforced and the task term alone shapes the structure;

• full modularity: $\lambda = 0$ on a network built without crossmodule synapses $( p _ { c r o s s } = 0$ , sealed);

• dynamic modularity: the complete rule (6), which encourages the communities during learning.

## IV. MODEL ARCHITECTURE

## A. Standalone Map-STDP Agent

The controller has no recurrent synapses of its own. In each environment frame the network is simulated on the current weights:

• 100 steps for the branching engine;

• 100 ms for the LIF and QIF engines;

• 20 hops of power iteration for the flow engine, which has no spikes.

The spike raster gives the module flow (spike shares), the pair counts and the signed counters. Each spiking agent runs 8 parallel heads on shared weights and pools their statistics.

## B. TD-STDP Actor-Critic Backbone

The backbone is the TD-STDP actor–critic of Chung and Kozma. Its neurons are noisy LIF $( \tau _ { m } = 2 0 ~ \mathrm { m s } )$ on a 1 ms grid, and each environment step is a 20 ms window. Population sizes follow the authors’ configurations on CartPole, and in Acrobot the critic has 40 excitatory neurons and the actor has 20 per action; on LunarLander each excitatory group is paired with an inhibitory group of equal size (128 critic, 32 per action). CoRe replaces the feed-forward actor with recurrent communities, one per action, among the actor’s excitatory neurons. Actions are read out by a race between the action communities’ excitatory spike counts instead of the original softmax.

## C. SBM Initialization

We initialize the network with an SBM with $K = | { \mathcal { C } } | + | { \mathcal { A } } | +$ |L| communities. The intra-community connection probability, $p _ { i n }$ is consistently set to 0.5. In the sealed regime, the intercommunity connection probability, $p _ { c r o s s } = 0$ with all-toall connections only between the controller and the other communities. In most experiments, the partial regime has $p _ { c r o s s } = 0 . 0 5$ , but in the inter-module connectivity study, we vary $p _ { c r o s s }$ from 0 (full modularity) to 0.5 (no communities).

## V. IMPLEMENTATION DETAILS

Models are implemented in NumPy, with Numba kernels for the LIF and QIF engines. The branching engine uses SuperNeuroMAT’s [11] branching mode (infinite leak, uniform threshold noise), so each simulation step is one hop of the walk. Environments come from Gymnasium [26].

## VI. TRAINING

a) Standalone agent.: Agents are trained on CartPole for 1500 episodes, with 20 seeds per condition. Each condition is one engine (flow, branching, LIF, QIF) in one of three regimes: dynamic $( \lambda = 0 . 0 5 )$ , partial $( \lambda = 0 )$ , or full (sealed: $\lambda = 0$ , no cross-module synapses). Map-STDP and the task update are applied once per frame. All conditions use TDerror normalization and an adaptive readout sharpness. In the spiking engines, λ and the task rate are rescaled to match the flow engine’s update size at initialization.

b) Backbone.: The TD-STDP and Map-STDP updates are applied once per 20 ms window, without Adam. Input scales are calibrated so that every arm starts at the same 40 Hz rate. The arms are the original feed-forward network, the feedforward network with the race readout, and CoRe SNNs in the partial, dynamic and full regimes. Each arm has 20 seeds. Per task:

• CartPole: 400 episodes, one environment copy, race bound 16.

• Acrobot: 500 episodes, one environment copy, race bound 16, one action every 40 ms, plus a firing-rate cost and floor on the actor.

• LunarLander: 10,000 episodes pooled over 16 environment copies on shared weights, race bound 26, one action every 40 ms, plus the same rate cost and floor.

## VII. RESULTS

## A. Minimal spiking policy network

1) Results on CartPole: Figure 2 summarizes the compari son. In panel (a) of the figure, the trained networks of one seed are shown for each regime of the flow engine with a central controller, two latent modules, and two action modules. In the dynamic and full regimes, the modules are clearly apparent, indicating that qualitatively, on this engine, the map term is effective in shaping the network into a modular structure. In the partial regime, the modules are less distinct, indicating that the SBM initialization alone leaves room for inter-module connectivity. In panel (b), the mean return over training is shown for each engine and regime. The dynamic and full regimes outperform the partial regime across all engines, however overall, flow outperforms the other engines, albeit with a larger spread. LIF comes next, followed by QIF and branching. This may be due to the fact that in the base CoRe architecture, the random walk entropy formulation is most closely aligned with the flow engine, and spiking engines approximate walk behavior. In panel (c), the effective synaptic operations per decision (spikes × outgoing synapses above the weight floor, averaged over 50 held-out stimuli with W frozen) is plotted against $L _ { o b s } ,$ , the held-out description length of the observed walk. As there is a direct exponential correlation, this may hint at the fact that the more compressible the walk, the fewer synaptic operations are needed to make a decision, and therefore the more energy efficient the network is. In panel (d), $L _ { o b s }$ is shown for each engine and regime at episode 1500. The dynamic and full regimes have lower $L _ { o b s }$ than the partial arm, indicating that the modularity is effective in shaping the network into a more compressible structure. Compressibility largely follows from modularity itself: the full regime, which has no map term, has a lower $L _ { o b s }$ than the partial regime on all four engines, and on flow and branching it is within 0.2 bits of the dynamic regime (Table II). On QIF and LIF the map term lowers $L _ { o b s }$ further. Overall, these results suggest that the map term is effective in shaping the network into a modular structure that is more compressible and energy efficient, and that this leads to improved performance on the CartPole task.

2) Latent Ablation: To find out what the latent modules contribute, we ablated them in the final CartPole models (episode 1500) of every engine and regime (240 models in all) and measured the change in return with plasticity off. The design was pre-registered. Each network has a 36-neuron controller C, two latent modules $\mathcal { L } = \mathcal { L } _ { 1 } \cup \mathcal { L } _ { 2 }$ and two 16- neuron action modules A. Because every engine normalizes outgoing weights, $T _ { i j } = W _ { i j } / d _ { j }$ with $\begin{array} { r } { d _ { j } = \sum _ { k } W _ { k j } } \end{array}$ , a cut either loses the drive the cut synapses carried or reroutes it, so we test both cases. A lesion cuts the latents’ outgoing synapses and leaves every other neuron’s $d _ { j }$ unchanged, so the drive into the latents is lost. It tests whether the latents relay anything the policy uses. A removal disconnects the latents entirely, and each remaining neuron’s outflow renormalizes onto its other targets, $T _ { i j } ^ { \prime } = T _ { i j } / ( 1 - \ell _ { j } )$ with $\begin{array} { r } { \ell _ { j } \ = \ \sum _ { k \in \mathcal { L } } T _ { k j } } \end{array}$ . It tests whether the latents act as a sink that sets the action modules share of the controller’s drive. In the latent path only condition the direct controller-to-action synapses are cut, which tests whether the latents alone can carry the policy. Each model is evaluated intact and under each ablation for 50 episodes with common random numbers, so every model is its own control. QIF’s warm state is not saved, so its intact returns differ slightly from training.

![](images/7363a95e292841d8ac2e79f602f3e185c5c262894e3b977be1d832a550e4fea7.jpg)

![](images/f40732b759f4978430e36625e8c3930444cc279a3b882ce256ef63191fcb4744.jpg)

![](images/c11df2876a2c1e22c4e527c9cd84aa1c15627253f5d486e081644c240996cb3b.jpg)

![](images/1b189b4997b097550e6a8132c7663859cc5fbac3f9c3031fe074e8f08c086a06.jpg)  
Fig. 2. (a) Trained CartPole networks on the flow engine, one seed (2000), from the same initial network. C is the controller, L1–L2 the latent modules and A1–A2 the action modules. Edges of at least 2% of the maximum weight are drawn, with width and opacity proportional to weight. Below each network: the share of weight within modules and across modules (excluding controller routing), and $L _ { W }$ , the map equation (bits) of W’s own walk. (b) AUC, the mean return over 1500 training episodes, for each engine and regime; 20 seeds per condition. (c) Effective synaptic operations per decision (spikes × outgoing synapses above the weight floor, averaged over 50 held-out stimuli with W frozen) against $L _ { o b s } .$ , for the final networks of the three spiking engines. Color: engine; shape: regime. Dashed lines are least-squares fits of $\log _ { 1 0 }$ effective SOPs on $L _ { o b s } ; \rho$ is Spearman’s. Post-hoc, exploratory analysis. (d) $L _ { o b s } .$ , the held-out description length (bits) of the engine’s observed walk, at episode 1500. In (b) and (d), boxes show the median and interquartile range, whiskers the full range, and dots individual runs.

The policy is carried by the direct routing, not by the latents. With it cut, every group fell to about random play (returns of 10–32 against 22.8 for a random policy; Table I, last column). Removing the latents lowered return most in flow (−206 to −287), branching in the dynamic regime (−118) and LIF in the dynamic and full regimes (−112, −80), and by −31 to −65 in the other groups. Without the latents the controller’s whole outflow reaches the action modules, and their share of the drive rose from 0.08–0.49 to 0.57–0.80. The latents therefore act as a sink that sets how much of the controller’s drive reaches the actions, and the trained policy depends on that setting.

Their output is not a relay the policy needs. In fullmodularity models the lesion left returns identical to intact, since they have no latent-to-action synapses, and in the dynamic regime it changed almost nothing on QIF and LIF. Where the latents still drive the actions, cutting their output improved return (branching dynamic +47, LIF partial +38), so it acted as interference. Flow in the dynamic regime is the exception: lesioning one latent module cost return $( { \mathcal { L } } _ { 2 } { : }$ −117, $\mathcal { L } _ { 1 } \colon - 7 7 )$ , but lesioning both did not (−3), which suggests that the two modules’ small outputs balance each other. Differences between regimes were small. Only LIF (partial against dynamic) and branching (full against dynamic) differed, and the remaining contrasts did not.

When proceeding with the TD-STDP backbone, we test models without latents.

## B. TD-STDP actor-critic backbone

1) Performance: In the previous subsection, we showed that the minimal spiking policy network can be significantly improved with CoRe. However, in order to show that advancements can be made with this abstraction, we now demonstrate how CoRe can be applied to an existing TD-STDP actorcritic backbone to improve its performance. Figure 3 shows the mean return over training episodes for each environment and arm. The results indicate that when CoRe is applied to the TD-STDP backbone, the performance of the network improves cleanly in CartPole and LunarLander. However, the results on Acrobot are a bit more complicated, when the race readout first replaced the softmax, the performance dropped, but when CoRe was applied, the performance improved and slightly exceeded the original softmax performance in the full regime. Overall, these results suggest that CoRe can be

![](images/9d9ce647f07f925ebda4c64d1f2408bdb7902d12a1ab71fcbe33661f144bba74.jpg)

![](images/2b2ce478293b4eaa3d88cda98ee7898a313db39fe92c48faddc539df48ad1ab1.jpg)

![](images/3f2bbb86a978c1adc41ea94443bcd769829002edf16453f17499e2ff6185f091.jpg)  
Fig. 3. TD-STDP actor-critic backbone. The actor network generates actions based on the current state, while the critic network evaluates the actions. The TD-STDP learning rule updates the weights based on the prediction error and the eligibility trace. In panels (a-c), the mean return over training episodes is shown for each environment and arm; 20 seeds per arm. Lunar Lander shows mean return for the last 100 episodes.  
TABLE I

CHANGE IN RETURN FROM THE INTACT BASELINE WHEN THE LATENTS ARE ABLATED (MEDIAN OVER THE 20 MODELS IN EACH ROW). ARE ABLATED (MEDIAN OVER THE 20 MODELS IN EACH ROW),

<sup>†</sup>IDENTICAL TO INTACT ON EVERY MODEL. INTACT: MEAN RETURN OF THE INTACT MODELS.
<table><tr><td colspan="3"></td><td colspan="3">Lesion</td><td rowspan="2">Removal</td><td rowspan="2">Latent L path only</td></tr><tr><td>Engine</td><td>Regime Intact</td><td></td><td>L</td><td> $\mathcal { L } _ { 1 }$ </td><td> $\mathcal { L } _ { 2 }$ </td></tr><tr><td rowspan="3">Flow</td><td>dynamic</td><td>462</td><td></td><td>-3-77-117</td><td></td><td>-206</td><td>-455</td></tr><tr><td>partial</td><td>414-18</td><td></td><td>+1</td><td>0</td><td>-287</td><td>-418</td></tr><tr><td>full</td><td>481</td><td>0†</td><td>0†</td><td>0†</td><td>-254</td><td>-478</td></tr><tr><td rowspan="3"></td><td>dynamic</td><td></td><td>199 +47 +18</td><td></td><td>+16</td><td>-118</td><td>-167</td></tr><tr><td>Branching partial</td><td></td><td>124 +18 +12</td><td></td><td>+8</td><td>-65</td><td>-105</td></tr><tr><td>full</td><td>178</td><td>0†</td><td>0†</td><td>0†</td><td>-41</td><td>-137</td></tr><tr><td rowspan="3">QIF</td><td>dynamic</td><td>150</td><td>0</td><td>+1</td><td>+1</td><td>-31</td><td>-140</td></tr><tr><td>partial</td><td></td><td>96 +13</td><td>+6</td><td>+13</td><td>-50</td><td>-87</td></tr><tr><td>full</td><td>192</td><td>0†</td><td>0†</td><td>0†</td><td>-32</td><td>-179</td></tr><tr><td rowspan="3">LIF</td><td>dynamic</td><td>327</td><td>+2</td><td>+1</td><td>-2</td><td>-112</td><td>-311</td></tr><tr><td>partial</td><td></td><td>199 +38 +18</td><td></td><td>+17</td><td>-42</td><td>-181</td></tr><tr><td>full</td><td>305</td><td>0†</td><td>0†</td><td>0†</td><td>-80</td><td>-281</td></tr></table>

effectively applied to existing TD-STDP actor-critic backbones to improve their performance on reinforcement learning tasks.

2) Inter-module Connectivity Effects: In the previous sections, we have explored comparing the three regimes of modularity: partial (SBM with $p _ { c r o s s } = 0 . 0 5 )$ , full (sealed, $p _ { c r o s s } ~ = ~ 0 )$ , and dynamic (learned with the map term). However, it has been left open how much cross-module connectivity is optimal for performance, and what happens when $p _ { c r o s s }$ is varied. To address this, we conducted a sweep of $p _ { c r o s s }$ values from 0 (sealed) to 0.5 (no communities) in the TD-STDP actor-critic backbone, with both the map term on and off. When $p _ { c r o s s } = 0 .$ , while the map-term is constrained to act upon intra-module connectivity, this still yields a difference in performance compared to when the map term is off.

The results, shown in Figure 4, indicate that optimal intermodule connectivity is task-dependent, and setting-dependent. Across all three environments, the most optimal $p _ { c r o s s }$ lies in its lower range. In CartPole, the performance is highest at $p _ { c r o s s } = 0$ with the map term off. In Acrobot, the performance is highest at $p _ { c r o s s } = 0$ with the map term on. In LunarLander, the performance is highest when $p _ { c r o s s } ~ = ~ 0 . 0 2 5$ with the map term on. Overall, this suggests that in some tasks, actions may be dependent on each other, and therefore some crossmodule connectivity is beneficial, while in other tasks, actions may be independent of each other, and therefore cross-module connectivity is detrimental.

The trends in the results are also task and setting dependent. While most results show a downward trend in performance as $p _ { c r o s s }$ increases, there are several exceptions and nuances. Many of the correlation coefficients are brought down by nonmonotonicity in the trends, with many trends following a drop off, then a rise, then a drop again, and sometimes another rise. This is especially apparent with CartPole with the map term on, and LunarLander with the map term off.

Overall, these results suggest that the optimal inter-module connectivity is task-dependent, and setting-dependent, and that the map term can have a significant impact on performance. Future work should explore dynamically setting the $K \times K$ inter-module connectivity matrix, rather than a single $p _ { c r o s s }$ value, to allow for more flexibility in the network’s structure and performance.

## VIII. DISCUSSION: BIOLOGICAL PLAUSIBILITY

CoRe SNNs share two features of cortex. First, as motivated in the Introduction, their connectivity is modular: Map-STDP learns weights that are dense within communities and sparse between them, and partial and full modularity impose the same structure directly. Second, all CoRe models learn with three-factor rules, in which a local eligibility changes a weight only when a neuromodulatory signal arrives. Such rules have direct experimental support [30], [31], our task term follows the dopaminergic temporal-difference account of reward learning [32], and the structural modulator is a single scalar per module, in line with the regional specificity of neuromodulation [33]. Two parts remain abstractions. The structural modulator is indexed by the presynaptic module, which requires a roughly uniform or projection-specific signal, and the per-neuron baseline $\bar { e } _ { j } ^ { o b s }$ has no established biological counterpart.

![](images/6fdeb098548f1a6c57b7d5d9784677b77e0fa1239d2e0ee2d2722314434852a0.jpg)

b Acrobot  
![](images/582405a93bcda003402a91e811ad92db7169d94cc8f5ad50e16b9e493bff21a5.jpg)

c LunarLander  
![](images/806c41e4cff8431270b0389e49e864d6b409feb91612fa5f41fc4fd016abab9f.jpg)  
p<sub>cross</sub>: cross-column connection probability  
Fig. 4. Inter-module connectivity effects in the TD-STDP actor-critic backbone. The figure illustrates how the connectivity between different modules influences the performance and learning dynamics of the network. The Spearman coefficients are shown to the right of each plot, indicating the strength and direction of the relationship between inter-module connectivity and performance.

Map-STDP also fits the efficient-coding view of the Introduction. Communication accounts for much of the brain’s energy budget [34], and compressing the network’s activity prunes cross-module traffic. In exploratory measurements, a lower description length coincided with fewer effective synaptic operations per decision (Spearman $\rho = 0 . 6 1 – 0 . 9 5$ across engines). The task term keeps the network useful for acting, and the structural term keeps its activity compressible. Finally, our TD-STDP actor-critic [39] is more biologically plausible than the original. Its actors are recurrent communities, closer to cortical microcircuits and to the channel organization of the basal ganglia [37], [38], it runs in a noisy leaky integrate-andfire regime at lower firing rates, and it selects actions by a race between action communities, which is equivalent to softmax sampling under constant hazards [40].

## IX. LIMITATIONS AND FUTURE WORK

Limitations. Map-STDP relies on a random-walk abstraction of inference. Random walks, and branching processes in particular, are inherently sequential, so estimating the walk statistics per frame becomes a bottleneck as networks scale; a more parallelizable abstraction of inference is needed. Second, the map equation is only one principle that can drive module formation. Alternatives such as the Constant Potts Model [41], modularity maximization [42], attractor models [43], and the dynamic SBM matrices discussed earlier remain unexplored here. Our results suggest that such alternatives are worth testing, because compressibility largely accompanied modularity itself rather than requiring an explicit information-theoretic objective. Objectives that reward modularity directly could therefore yield similar structure without the map equation. Third, as Table I shows, the latent communities act largely as sinks and, at best, routers; we do not yet train them to perform useful computation of their own.

CoRe as a starting point. We view CoRe as a base architecture that opens many directions. Its modular structure decomposes functionality into smaller, separately trainable pieces, and the TD-STDP backbone experiment shows that a CoRe module can be inserted into and coupled with existing, performant architectures and improve them. This opens several directions. Latents could be trained to carry out subtasks or store memory, through persistent activity across frames or through coupling with memory-oriented architectures such as Hopfield networks [44] or Boltzmann machines [45]. Coupling with architectures built for binding and unbinding [46] in the controller could let different latents attend to different objects in the environment. Finally, the same modular interface could host other backbones and other module-forming objectives, so that CoRe serves as a common substrate for these extensions.

## ACKNOWLEDGMENT

We thank Dr. Ramakrishnan (Ramki) Kannan at Oak Ridge National Laboratory for providing access to his local cluster. We also thank the AI Interpretability @ Illinois group for providing the collaborative network to make this work possible.

The authors used Claude Code (Anthropic) with the Claude Sonnet 5.5, Opus 5.5, and Fable 5.1 models during this work. The AI generated software and experimental scripts used for the results in Section VII; assisted with the derivation in Appendix B; assisted in the iterative design of the figures; and provided language editing throughout the manuscript. All code was reviewed, tested, and validated by the authors against the source data; all derivations were independently checked by the authors; and all text was reviewed and revised by the authors. The authors take full responsibility for the content of this paper, including the integrity of the analysis and results.

[1] K. Malcolm and J. Casco-Rodriguez, “A comprehensive review of spiking neural networks: Interpretation, optimization, efficiency, and best practices,” arXiv:2303.10780 [cs.NE], 2023.

[2] P. A. Merolla et al., “A million spiking-neuron integrated circuit with a scalable communication network and interface,” Science, vol. 345, no. 6197, pp. 668–673, 2014, doi: 10.1126/science.1254642.

[3] M. Davies et al., “Loihi: A neuromorphic manycore processor with onchip learning,” IEEE Micro, vol. 38, pp. 82–99, 2018.

[4] M. Prezioso et al., “Training and operation of an integrated neuromorphic network based on metal-oxide memristors,” Nature, vol. 521, pp. 61–64, 2015, doi: 10.1038/nature14441.

[5] M. Davies et al., “Advancing neuromorphic computing with Loihi: A survey of results and outlook,” Proc. IEEE, vol. 109, no. 5, pp. 911–934, 2021.

[6] E. O. Neftci, H. Mostafa, and F. Zenke, “Surrogate gradient learning in spiking neural networks,” arXiv:1901.09948 [cs.NE], 2019.

[7] T. P. Lillicrap, A. Santoro, L. Marris, C. J. Akerman, and G. E. Hinton, “Backpropagation and the brain,” Nat. Rev. Neurosci., vol. 21, pp. 335– 346, 2020.

[8] H. Markram, J. Lubke, M. Frotscher, and B. Sakmann, “Regulation¨ of synaptic efficacy by coincidence of postsynaptic APs and EPSPs,” Science, vol. 275, no. 5297, pp. 213–215, 1997, doi: 10.1126/science.275.5297.213.

[9] G. Bi and M. Poo, “Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type,” J. Neurosci., vol. 18, no. 24, pp. 10464–10472, 1998.

[10] P. U. Diehl and M. Cook, “Unsupervised learning of digit recognition using spike-timing-dependent plasticity,” Front. Comput. Neurosci., vol. 9, p. 99, 2015, doi: 10.3389/fncom.2015.00099.

[11] P. Date et al., “SuperNeuroMAT: An efficient matrix-based simulator for spiking neural networks,” arXiv:2608.08479 [cs.NE], 2026.

[12] J. M. Beggs and D. Plenz, “Neuronal avalanches in neocortical circuits,” J. Neurosci., vol. 23, no. 35, pp. 11167–11177, 2003.

[13] L. Lapicque, “Recherches quantitatives sur l’excitation electrique des´ nerfs traitee comme une polarization,” J. Physiol. Pathol. Gen., vol. 9,´ pp. 620–635, 1907.

[14] N. Brunel, “Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons,” J. Comput. Neurosci., vol. 8, no. 3, pp. 183–208, 2000.

[15] G. B. Ermentrout and N. Kopell, “Parabolic bursting in an excitable system coupled with a slow oscillation,” SIAM J. Appl. Math., vol. 46, no. 2, pp. 233–253, 1986.

[16] E. Montbrio, D. Paz´ o, and A. Roxin, “Macroscopic description for´ networks of spiking neurons,” Phys. Rev. X, vol. 5, no. 2, p. 021028, 2015.

[17] S. Brin and L. Page, “The anatomy of a large-scale hypertextual web search engine,” Comput. Netw., vol. 30, pp. 107–117, 1998.

[18] L. Buesing, J. Bill, B. Nessler, and W. Maass, “Neural dynamics as sampling: A model for stochastic computation in recurrent networks of spiking neurons,” PLoS Comput. Biol., vol. 7, no. 11, p. e1002211, 2011, doi: 10.1371/journal.pcbi.1002211.

[19] M. Rosvall, D. Axelsson, and C. T. Bergstrom, “The map equation,” Eur. Phys. J. Spec. Top., vol. 178, no. 1, pp. 13–23, 2009, doi: 10.1140/epjst/e2010-01179-1.

[20] M. Rosvall and C. T. Bergstrom, “Mapping change in large networks,” PLoS ONE, vol. 5, no. 1, p. e8694, 2010, doi: 10.1371/journal.pone.0008694.

[21] P. W. Holland, K. B. Laskey, and S. Leinhardt, “Stochastic blockmodels: First steps,” Soc. Netw., vol. 5, no. 2, pp. 109–137, 1983.

[22] M. Kiselev, “CoLaNET – A spiking neural network with columnar layered architecture for classification,” arXiv:2409.01230 [cs.NE], 2025.

[23] M. Kiselev, “Classifying images with CoLaNET spiking neural network – the MNIST example,” arXiv:2409.07833 [cs.NE], 2024.

[24] M. Rosvall and C. T. Bergstrom, “Maps of random walks on complex networks reveal community structure, Proc. Natl. Acad. Sci. USA, vol. 105, no. 4, pp. 1118–1123, 2008.

[25] G. E. Hinton and D. van Camp, “Keeping the neural networks simple by minimizing the description length of the weights,” in Proc. 6th Annu. Conf. Comput. Learn. Theory (COLT), Santa Cruz, CA, USA, 1993, pp. 5–13, doi: 10.1145/168304.168306.

[26] M. Towers et al., “Gymnasium: A standard interface for reinforcement learning environments,” arXiv:2407.17032 [cs.LG], 2024.

[27] V. B. Mountcastle, “The columnar organization of the neocortex,” Brain, vol. 120, no. 4, pp. 701–722, 1997, doi: 10.1093/brain/120.4.701.

[28] D. Meunier, R. Lambiotte, and E. T. Bullmore, “Modular and hierarchically modular organization of brain networks,” Front. Neurosci., vol. 4, p. 200, 2010, doi: 10.3389/fnins.2010.00200.

[29] O. Sporns and R. F. Betzel, “Modular brain networks,” Annu. Rev. Psychol., vol. 67, pp. 613–640, 2016, doi: 10.1146/annurev-psych-122414-033634.

[30] W. Gerstner, M. Lehmann, V. Liakoni, D. Corneil, and J. Brea, “Eligibility traces and plasticity on behavioral time scales: Experimental support of neoHebbian three-factor learning rules,” Front. Neural Circuits, vol. 12, p. 53, 2018, doi: 10.3389/fncir.2018.00053.

[31] N. Fremaux and W. Gerstner, “Neuromodulated spike-timing-dependent´ plasticity, and theory of three-factor learning rules,” Front. Neural Circuits, vol. 9, p. 85, 2016, doi: 10.3389/fncir.2015.00085.

[32] W. Schultz, P. Dayan, and P. R. Montague, “A neural substrate of prediction and reward,” Science, vol. 275, no. 5306, pp. 1593–1599, 1997, doi: 10.1126/science.275.5306.1593.

[33] B. Engelhard et al., “Specialized coding of sensory, motor and cognitive variables in VTA dopamine neurons,” Nature, vol. 570, no. 7762, pp. 509–513, 2019, doi: 10.1038/s41586-019-1261-9.

[34] D. Attwell and S. B. Laughlin, “An energy budget for signaling in the grey matter of the brain,” J. Cereb. Blood Flow Metab., vol. 21, no. 10, pp. 1133–1145, 2001, doi: 10.1097/00004647-200110000-00001.

[35] H. B. Barlow, “Possible principles underlying the transformations of sensory messages,” in Sensory Communication, W. A. Rosenblith, Ed. Cambridge, MA, USA: MIT Press, 1961, pp. 217–234.

[36] K. Friston, “The free-energy principle: A unified brain theory?” Nat. Rev. Neurosci., vol. 11, no. 2, pp. 127–138, 2010, doi: 10.1038/nrn2787.

[37] R. J. Douglas and K. A. C. Martin, “Neuronal circuits of the neocortex,” Annu. Rev. Neurosci., vol. 27, pp. 419–451, 2004, doi: 10.1146/annurev.neuro.27.070203.144152.

[38] P. Redgrave, T. J. Prescott, and K. Gurney, “The basal ganglia: A vertebrate solution to the selection problem?” Neuroscience, vol. 89, no. 4, pp. 1009–1023, 1999, doi: 10.1016/s0306-4522(98)00319-4.

[39] S. Chung and R. Kozma, “Reinforcement learning with feedbackmodulated TD-STDP,” arXiv:2008.13044 [cs.LG], 2020.

[40] J. I. Yellott, “The relationship between Luces choice axiom, Thurstone´ s´ theory of comparative judgment, and the double exponential distribution, J. Math. Psychol., vol. 15, no. 2, pp. 109–144, 1977.

[41] V. A. Traag, P. Van Dooren, and Y. Nesterov, “Narrow scope for resolution-limit-free community detection,” Phys. Rev. E, vol. 84, no. 1, p. 016114, Jul. 2011.

[42] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, “Fast unfolding of communities in large networks,” J. Stat. Mech.: Theory Exp., vol. 2008, no. 10, p. P10008, Oct. 2008.

[43] C. Curto, J. Geneson, and K. Morrison, “Fixed points of competitive threshold-linear networks,” arXiv:1804.00794, 2018.

[44] J. J. Hopfield, “Neural networks and physical systems with emergent collective computational abilities,” Proc. Natl. Acad. Sci. USA, vol. 79, no. 8, pp. 2554–2558, 1982.

[45] D. H. Ackley, G. E. Hinton, and T. J. Sejnowski, “A learning algorithm for Boltzmann machines,” Cogn. Sci., vol. 9, no. 1, pp. 147–169, 1985.

[46] E. P. Frady, S. J. Kent, B. A. Olshausen, and F. T. Sommer, “Resonator networks, 1: An efficient solution for factoring high-dimensional, distributed representations of data structures,” Neural Comput., vol. 32, no. 12, pp. 2311–2331, 2020.

[47] T. Kohonen, “Self-organized formation of topologically correct feature maps,” Biol. Cybern., vol. 43, no. 1, pp. 59–69, 1982.

[48] T. Kohonen, “The self-organizing map,” Proc. IEEE, vol. 78, no. 9, pp. 1464–1480, 1990.

[49] C. Curto and K. Morrison. Relating network connectivity to dynamics: opportunities and challenges for theoretical neuroscience. Current Opinion in Neurobiology, 58:11–20, 2019. doi:10.1016/j.conb.2019.06.003.

[50] C. Curto, J. Geneson, and K. Morrison. Fixed points of competitive threshold-linear networks. Neural Computation, 31(1):94–155, 2019. doi:10.1162/neco a 01151.

[51] C. Curto and K. Morrison. Graph rules for recurrent neural network dynamics. Notices of the American Mathematical Society, 70(4):536– 551, 2023. doi:10.1090/noti2661.

[52] C. Curto, J. Geneson, and K. Morrison. Stable fixed points of combinatorial threshold-linear networks. Advances in Applied Mathematics, 154:102652, 2024. doi:10.1016/j.aam.2023.102652.

[53] C. Parmelee, S. Moore, K. Morrison, and C. Curto. Core motifs predict dynamic attractors in combinatorial threshold-linear networks. PLOS ONE, 17(3):e0264456, 2022. doi:10.1371/journal.pone.0264456.

[54] C. Parmelee, J. Londono Alvarez, C. Curto, and K. Morrison. Sequential attractors in combinatorial threshold-linear networks. SIAM Journal on Applied Dynamical Systems, 21(2):1597–1630, 2022. doi:10.1137/21M1445120.

[55] J. Londono Alvarez, K. Morrison, and C. Curto. Attractorbased models for sequences and pattern generation in neural circuits. Neural Computation, 38(3):257, 2026. Preprint: bioRxiv, doi:10.1101/2025.03.07.642121.

[56] D. Egas Santander, S. Ebli, A. Patania, N. Sanderson, F. Burtscher, K. Morrison, and C. Curto. Nerve theorems for fixed points of neural networks. In Research in Computational Topology 2, Association for Women in Mathematics Series, vol. 30. Springer, 2022. doi:10.1007/978- 3-030-95519-9 6.

[57] J. Londono Alvarez. Fixed point compositionality via low-rank gluing rules in inhibition-dominated threshold-linear networks. arXiv:2606.07336, 2026.

[58] C. Curto, C. Langdon, and K. Morrison. Combinatorial geometry of threshold-linear networks. arXiv:2008.01032, 2020.

[59] C. Lienkaemper and G. K. Ocker. Diverse mean-field dynamics of clustered, inhibition-stabilized Hawkes networks via combinatorial threshold-linear networks. PRX Life, 2025. doi:10.1103/m8hf-q5pl. arXiv:2506.06234.

[60] R. Toro Icarte, T. Q. Klassen, R. Valenzano, and S. A. McIlraith. Reward machines: exploiting reward function structure in reinforcement learning. Journal of Artificial Intelligence Research, 73:173–208, 2022. doi:10.1613/jair.1.12440.

[61] C. W. Omlin and C. L. Giles. Constructing deterministic finite-state automata in recurrent neural networks. Journal of the ACM, 43(6):937– 972, 1996. doi:10.1145/235809.235811.

[62] M. Rigotti, D. D. Ben Dayan Rubin, X.-J. Wang, and S. Fusi. Internal representation of task rules by recurrent dynamics: the importance of the diversity of neural responses. Frontiers in Computational Neuroscience, 4:24, 2010. doi:10.3389/fncom.2010.00024.

## APPENDIX A FULL RESULTS

TABLE II  
CARTPOLE, MODEL V4 WITH THE OBSERVED-WALK RULE: DYNAMIC (λ = 0.05), PARTIAL (λ = 0) AND FULL (SEALED: NO CROSS-MODULE SYNAPSES, $\lambda = 0 )$ MODULARITY, ON FOUR ENGINES. $\mathrm { M E A N \pm S . D }$ . OVER 20 SEEDS (2000–2019), 1500 EPISODES PER RUN. AUC: MEAN RETURN OVER ALL EPISODES. RETURN: MEAN OF THE LAST 100 EPISODES. FROZEN: MEAN RETURN OF 20 EPISODES WITH W FROZEN AT THE END. $L _ { o b s } \mathrm { : }$ : HELD-OUT DESCRIPTION LENGTH (BITS) OF THE ENGINE’S OBSERVED WALK AT EPISODE 1500. SOLVED: RUNS WHOSE TRAILING-100 MEAN REACHES 475. p: TWO-SIDED MANN–WHITNEY, HOLM-ADJUSTED OVER THE FOUR ENGINES, OF EACH ARM AGAINST THE MAP-TERM-ON ARM OF THE SAME ENGINE.
<table><tr><td>Engine</td><td>Regime</td><td>AUC</td><td>Return</td><td>Frozen</td><td> $L _ { o b s } ~ ( \mathrm { b i t s } )$ </td><td>Solved</td><td>pAUC</td><td> $p _ { L _ { o b s } }$ </td></tr><tr><td>Flow</td><td>Dynamic</td><td> $3 5 6 . 9 \pm 9 6 . 7$ </td><td> $4 4 2 . 4 \pm 1 0 3 . 5$ </td><td> $4 6 3 . 9 \pm 9 4 . 3$ </td><td> $4 . 5 4 \pm 0 . 0 1$ </td><td>15/20</td><td></td><td></td></tr><tr><td></td><td>Partial</td><td> $3 3 5 . 9 \pm 6 4 . 9$ </td><td> $4 3 1 . 6 \pm 1 0 5 . 6$ </td><td> $4 1 2 . 8 \pm 1 2 2 . 8$ </td><td> $6 . 0 0 \pm 0 . 1 4$ </td><td>20/20</td><td>0.17</td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td>Full</td><td> $3 8 8 . 3 \pm 6 8 . 8$ </td><td> $4 7 5 . 9 \pm 8 0 . 3$ </td><td> $4 8 1 . 4 \pm 8 0 . 1$ </td><td> $4 . 3 4 \pm 0 . 0 1$ </td><td>20/20</td><td>1.0</td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td>Branching</td><td>Dynamic</td><td> $2 0 4 . 1 \pm 1 4 . 1$ </td><td> $2 1 3 . 2 \pm 5 0 . 4$ </td><td> $1 9 8 . 9 \pm 9 0 . 2 $ </td><td> $6 . 6 9 \pm 0 . 0 3$ </td><td>0/20</td><td>一</td><td></td></tr><tr><td></td><td>Partial</td><td> $1 1 6 . 8 \pm 3 7 . 5$ </td><td> $1 2 8 . 7 \pm 6 8 . 1$ </td><td> $1 2 6 . 5 \pm 1 0 8 . 8$ </td><td> $7 . 0 6 \pm 0 . 1 4$ </td><td>0/20</td><td> $< 1 0 ^ { - 6 }$ </td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td>Full</td><td> $2 0 1 . 4 \pm 3 9 . 0$ </td><td> $1 7 6 . 2 \pm 6 7 . 9$ </td><td> $1 7 7 . 5 \pm 7 8 . 3$ </td><td> $6 . 7 5 \pm 0 . 0 4$ </td><td>3/20</td><td>1.0</td><td> $5 . 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>QIF</td><td>Dynamic</td><td> $1 6 8 . 3 \pm 1 8 . 8$ </td><td> $1 7 4 . 3 \pm 3 9 . 7$ </td><td> $1 5 1 . 6 \pm 4 3 . 6$ </td><td> $6 . 4 2 \pm 0 . 1 1$ </td><td>0/20</td><td>二</td><td></td></tr><tr><td></td><td>Partial</td><td> $1 0 3 . 4 \pm 1 7 . 1$ </td><td> $1 2 4 . 5 \pm 9 1 . 1$ </td><td> $9 7 . 5 \pm 8 2 . 8$ </td><td> $7 . 0 5 \pm 0 . 0 5$ </td><td>0/20</td><td> $< 1 0 ^ { - 6 }$ </td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td>Full</td><td> $1 9 6 . 0 \pm 2 4 . 3$ </td><td> $1 9 8 . 0 \pm 4 3 . 2$ </td><td> $1 8 9 . 4 \pm 3 0 . 7$ </td><td> $6 . 7 4 \pm 0 . 0 6$ </td><td>0/20</td><td>0.0023</td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td>LIF</td><td>Dynamic</td><td> $2 7 6 . 2 \pm 1 3 . 5$ </td><td> $3 2 1 . 6 \pm 8 0 . 8$ </td><td> $3 1 0 . 2 \pm 9 4 . 0$ </td><td> $7 . 4 4 \pm 0 . 0 2$ </td><td>0/20</td><td>二</td><td></td></tr><tr><td></td><td>Partial</td><td> $1 9 4 . 0 \pm 7 . 8$ </td><td> $2 0 8 . 4 \pm 4 0 . 4$ </td><td> $1 9 9 . 7 \pm 5 4 . 1 $ </td><td> $7 . 7 4 \pm 0 . 0 1$ </td><td>0/20</td><td> $< 1 0 ^ { - 6 }$ </td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td>Full</td><td> $2 7 0 . 8 \pm 1 2 . 7$ </td><td> $3 0 6 . 5 \pm 7 2 . 0$ </td><td> $2 9 4 . 6 \pm 8 0 . 6$ </td><td> $7 . 6 6 \pm 0 . 0 1$ </td><td>0/20</td><td>0.6</td><td> $< 1 0 ^ { - 6 }$ </td></tr></table>

## APPENDIX B DERIVATION OF MAP-STDP

This appendix derives the rule in Section III. All logarithms are base 2 (description lengths in bits).

## A. Setup and notation

Each synapse from presynaptic neuron j to postsynaptic neuron i has a magnitude $W _ { i j } \geq 0$ . The type of $j$ fixes its sign, $\mathrm { s g n } _ { j } = + 1$ (excitatory) $\mathrm { o r } - 1$ (inhibitory), as in Dale’s law. We write

$$
d _ { j } = \sum _ { k } W _ { k j } , \qquad T _ { i j } = \frac { W _ { i j } } { d _ { j } } ,\tag{7}
$$

for $j ^ { \dagger } \mathbf { s }$ total output weight and the share of it that goes to i. Each neuron belongs to one module $m ( j ) \in \{ 1 , \ldots , K \}$ . The module types are the controller ${ \mathcal { C } } ,$ the action modules A (one per action) and the latent modules $\mathcal { L } .$ Environment frames are indexed by $f ,$ and the stimulus of frame f is $\mathbf { o } _ { f }$

T is column-stochastic: column j says where $j ^ { \circ } \mathrm { s }$ output goes. A random walk that follows spikes forward moves from j to i with probability $T _ { i j }$

## B. The observed walk

A step of the walk from j to i is a spike of i within a short window w after a spike of $j .$ . The expected number of such spikes splits into three parts:

$$
\begin{array} { r } { E _ { i j } = \underbrace { a _ { i } } _ { \mathrm { c h a n c e } } + \underbrace { D _ { i j } } _ { \mathrm { d i r e c t } } + \underbrace { X _ { i j } } _ { \mathrm { i n d i r e c t } } \ge 0 , } \\ { a _ { i } = r _ { i } w , \qquad D _ { i j } = \mathrm { s g n } _ { j } \tilde { \Lambda } _ { i j } . } \end{array}\tag{8}
$$

Here $r _ { i }$ is i’s firing rate. $\tilde { \Lambda } _ { i j } = h _ { i } T _ { i j }$ is the direct effect of $j ^ { \circ } \mathrm { s }$ kick on i, where $h _ { i }$ is i’s windowed sensitivity (gain times kick size). $X _ { i j }$ collects chains $j \to k \to i$ and common input. The observed transition probabilities and link flows are

$$
E _ { j } = \sum _ { i } E _ { i j } , \qquad T _ { i j } ^ { o b s } = \frac { E _ { i j } } { E _ { j } } , \qquad J _ { i j } = \pi _ { j } T _ { i j } ^ { o b s } ,\tag{9}
$$

where $\pi _ { j }$ is $j ^ { \circ } \mathbf { s }$ share of all spikes in the frame.

a) Intuition.: $E _ { i j }$ is the mean of a count, so it is non-negative whatever the signs of the synapses. Inhibition does not create negative transitions; it pushes $E _ { i j }$ below chance.

b) Pair counts measure the walk.: With $\begin{array} { r } { \kappa _ { i j } = \sum _ { t } x _ { j } ( t ) N _ { i } ( t , t + w ] } \end{array}$ , where $x _ { j } ( t )$ marks a spike of $j$ at step t and $N _ { i }$ counts spikes of i, stationarity within the frame gives $\mathbb { E } [ \kappa _ { i j } ] = \bar { n } _ { j } E _ { i j }$ . So

$$
\hat { T } _ { i j } ^ { o b s } = \frac { \kappa _ { i j } } { \sum _ { k } \kappa _ { k j } } , \qquad \hat { \bar { e } } _ { j } ^ { o b s } = \frac { \sum _ { i \notin m ( j ) } \kappa _ { i j } } { \sum _ { i } \kappa _ { i j } } .\tag{10}
$$

$\bar { e } _ { j } ^ { o b s }$ is the share of $j ^ { \circ } \mathbf { s }$ observed transitions that leave its module. A neuron can estimate it from its own spike times and two population signals: activity in its own module, and in the whole network.

c) The flow engine is the special case.: Without spikes there is no chance $( a = 0 )$ and no indirect term $( X = 0 )$ , and all gains are equal, so $T ^ { o b s } = T$ . The stimulus v(o) re-injects a share α of the activity at each step:

$$
\pi = ( 1 - \alpha ) T \pi + \alpha v ( \mathbf { o } ) , \qquad \sum _ { i } \pi _ { i } = 1 .\tag{11}
$$

For $\alpha > 0$ the solution is unique. Starting from $x _ { 0 } = v ,$ n power-iteration hops give $\| x _ { n } - \pi \| _ { 1 } \leq 2 ( 1 - \alpha ) ^ { n }$ , whatever the network size.

## C. Communities as states and the policy

Coarse-graining the frame’s walk to modules gives a K-state Markov chain, with tables indexed [post, pre]:

$$
p _ { a } = \sum _ { j \in a } \pi _ { j } , \qquad J _ { b a } = \sum _ { i \in b } \sum _ { j \in a } J _ { i j } , \qquad T _ { b a } ^ { K } ( \mathbf { o } _ { f } ) = \frac { J _ { b a } } { p _ { a } } .\tag{12}
$$

Because the stimulus is concentrated in the controller, the controller’s column of $T ^ { K }$ is the stimulus-dependent one. The action is read from the flow into the action modules,

$$
P ( a \mid \mathbf { o } _ { f } ) = { \frac { p _ { a } ( f ) ^ { \beta } } { \sum _ { a ^ { \prime } \in { \mathcal { A } } } p _ { a ^ { \prime } } ( f ) ^ { \beta } } } , \qquad a \in { \mathcal { A } } ,\tag{13}
$$

with $\beta = 1$ the plain readout. Model v4 adapts $\beta$ to the trend of episode return. In spiking engines the readout is a race between the action modules’ spikes, which samples (13).

## D. The map equation

Describe the walk with a two-level code. Inside a module, name the next neuron, or “exit”, with a module-local codeword. On exit, name the next module from a global index. The average code length per step is [24]

$$
\begin{array} { l } { { \displaystyle { \cal L } = q _ { \sim } \log q _ { \sim } - 2 \sum _ { m } q _ { m } \log q _ { m } } } \\ { { \displaystyle ~ - \sum _ { j } \pi _ { j } \log \pi _ { j } + \sum _ { m } ( p _ { m } + q _ { m } ) \log ( p _ { m } + q _ { m } ) } , } \end{array}\tag{14}
$$

where $\begin{array} { r } { q _ { m } = \sum _ { j \in m } \sum _ { i \not \in m } J _ { i j } } \end{array}$ is module m’s exit flow and $\begin{array} { r } { q _ { \curvearrowright } = \sum _ { m } q _ { m } } \end{array}$ . With π held fixed, the weights enter $L$ only through the exit flows, and

$$
M _ { m } ^ { * } : = \frac { \partial L } { \partial q _ { m } } = \log \frac { q _ { \cap } ( p _ { m } + q _ { m } ) } { q _ { m } ^ { 2 } } \geq 0 .\tag{15}
$$

a) Intuition.: Leakage out of a module always costs bits. $M _ { m } ^ { * }$ is the marginal number of bits per unit of exit flow from m. It is computed from K module-level statistics and broadcast to module m: this is the rule’s neuromodulator.

b) The inverse problem.: Map-STDP pins the partition and minimizes over the weights, averaged over the frames the environment produces:

$$
\operatorname* { m i n } _ { W } D ( W ) , \qquad D ( W ) = \mathbb { E } _ { f } \left[ L ( J _ { f } ) \right] .\tag{16}
$$

c) Check.: Differentiate (14) term by term. The derivative of x log x is log x + c for a constant $c ,$ and the constants cancel $( c - 2 c + c = 0 )$ . The result is log $q _ { \curvearrowright } - 2 \log q _ { m } + \log ( p _ { m } + q _ { m } )$ . It is non-negative because $q _ { \cap { } } \geq q _ { m }$ and $p _ { m } + q _ { m } \ge q _ { m }$

## E. The structural rule

Let $c _ { i j }$ be the signed counter: the change in i’s spikes within the window caused by $j ^ { \dagger } \mathbf { s }$ kicks alone, with $\mathbb { E } [ c _ { i j } ] = \bar { n } _ { j } D _ { i j }$ The structural update is

$$
\Delta W _ { i j } = - \lambda \left( g _ { i j } - \bar { g } _ { j } ^ { o b s } \right) c _ { i j } ,
$$

$$
g _ { i j } = \frac { \partial D } { \partial J _ { i j } } , \qquad \bar { g } _ { j } ^ { o b s } = \sum _ { k } T _ { k j } ^ { o b s } g _ { k j } .\tag{17}
$$

For the map equation, $q _ { m }$ is a sum of the $J _ { i j }$ with $j \in m$ and $i \notin m ,$ so

$$
g _ { i j } = M _ { m ( j ) } ^ { * } \mathbb { I } [ i \notin m ( j ) ] , \qquad \bar { g } _ { j } ^ { o b s } = M _ { m ( j ) } ^ { * } \bar { e } _ { j } ^ { o b s } ,\tag{18}
$$

and the rule becomes

$$
\Delta W _ { i j } = - \lambda M _ { m ( j ) } ^ { * } \left( \mathbb { I } [ i \notin m ( j ) ] - \bar { e } _ { j } ^ { o b s } \right) c _ { i j } .\tag{19}
$$

a) Two statistics, two $j o b s .$ : The baseline $\bar { e } _ { j } ^ { o b s }$ describes the walk, so it comes from pair counts, chance included. The step must say how the walk changes when $W _ { i j }$ changes, which is the direct effect, so it comes from the signed counter. Using pair counts for the step fails: they are mostly chance, and renormalization cancels them.

b) Sign pattern.: The factor $\mathbb { I } [ i \notin m ( j ) ] - \bar { e } _ { j } ^ { o b s }$ is positive across modules and negative within. The counter carries $\mathrm { s g n } _ { j }$ Hence:

<table><tr><td>Synapse</td><td>Across modules</td><td>Within a module</td></tr><tr><td>Excitatory</td><td>weakened</td><td>strengthened</td></tr><tr><td>Inhibitory</td><td>strengthened</td><td>weakened</td></tr></table>

F. Why the rule descends the description length

a) Local split.: While differentiating with respect to one weight, hold fixed everything a single synapse does not directly touch: the flow $\pi ,$ chance a, the indirect term X and the gains. Then $W _ { i j }$ changes only column $j ^ { \circ } \mathrm { s }$ direct part, $D _ { k j } =$ $\mathrm { s g n } _ { j } h _ { k } W _ { k j } / d _ { j }$ (through $W _ { i j }$ itself and through $d _ { j } )$ .

Lemma 1 (Lemma E). If D depends on W only through the link flows J, then under the local split

$$
\begin{array} { c } { { W _ { i j } \displaystyle \frac { \partial { \cal D } } { \partial W _ { i j } } = \displaystyle \frac { \pi _ { j } \mathrm { s g n } _ { j } } { E _ { j } } \left[ \tilde { \Lambda } _ { i j } \left( g _ { i j } - \bar { g } _ { j } ^ { o b s } \right) - T _ { i j } C _ { j } \right] , } } \\ { { C _ { j } = \displaystyle \sum _ { k } \tilde { \Lambda } _ { k j } \left( g _ { k j } - \bar { g } _ { j } ^ { o b s } \right) . } } \end{array}\tag{20}
$$

Proof sketch. Under the split, $\partial E _ { k j } / \partial W _ { i j } = \mathrm { s g n } _ { j } ( h _ { k } / d _ { j } ) ( \delta _ { k i } - T _ { k j } )$ , and $J _ { k j } = \pi _ { j } E _ { k j } / E _ { j }$ . By the quotient rule,

$$
\frac { \partial J _ { k j } } { \partial W _ { i j } } = \frac { \pi _ { j } } { E _ { j } } \left( \frac { \partial E _ { k j } } { \partial W _ { i j } } - T _ { k j } ^ { o b s } \frac { \partial E _ { j } } { \partial W _ { i j } } \right) .
$$

Summing against $g _ { k j }$ , and using $\begin{array} { r } { \partial E _ { j } / \partial W _ { i j } = \sum _ { k } \partial E _ { k j } / \partial W _ { i j } } \end{array}$

$$
\begin{array} { c } { { \displaystyle \frac { \partial D } { \partial W _ { i j } } = \frac { \pi _ { j } } { E _ { j } } \sum _ { k } \left( g _ { k j } - \bar { g } _ { j } ^ { o b s } \right) \displaystyle \frac { \partial E _ { k j } } { \partial W _ { i j } } } } \\ { { = \displaystyle \frac { \pi _ { j } \mathrm { s g n } _ { j } } { E _ { j } d _ { j } } \Big [ h _ { i } \left( g _ { i j } - \bar { g } _ { j } ^ { o b s } \right) } } \\ { { - \sum _ { k } h _ { k } T _ { k j } \left( g _ { k j } - \bar { g } _ { j } ^ { o b s } \right) \Big ] . } } \end{array}
$$

Multiplying by $W _ { i j }$ , with $W _ { i j } / d _ { j } = T _ { i j }$ and $h _ { i } T _ { i j } = \tilde { \Lambda } _ { i j }$ , gives (20).

b) Intuition.: $C _ { j }$ is the same for every target i of $j ,$ so $T _ { i j } C _ { j }$ rescales column $j$ uniformly. That cannot change $T ,$ so it cannot change $D ;$ hence $\textstyle \sum _ { i } W _ { i j } \partial D / \partial W _ { i j } = 0$ . The other factors read as follows:

$\pi _ { j } / E _ { j }$ is a per-neuron step size;

$\mathrm { s g n } _ { j }$ flips the direction for inhibitory synapses;

• $\Lambda _ { i j }$ scales the step by how much j actually moves $i ;$

$g _ { i j } - \bar { g } _ { j } ^ { o b s }$ compares this transition’s marginal cost with the average over $j ^ { \circ } \mathbf { s }$ observed transitions.

Theorem 1 (Theorem E). If the counter is unbiased, $\mathbb { E } [ c _ { i j } ] = T _ { f } r _ { j } \mathrm { s g n } _ { j } \tilde { \Lambda } _ { i j }$ over a frame of duration $T _ { f }$ , then

$$
\begin{array} { c l c r } { { \mathbb { E } [ \Delta W _ { i j } ] = - \lambda \zeta E _ { j } W _ { i j } \frac { \partial D } { \partial W _ { i j } } - \lambda T _ { f } r _ { j } \mathrm { s g n } _ { j } C _ { j } T _ { i j } , } } \\ { { \Delta D = - \lambda \zeta \displaystyle \sum _ { i j } E _ { j } W _ { i j } \left( \frac { \partial D } { \partial W _ { i j } } \right) ^ { 2 } \leq 0 , } } \end{array}\tag{21}
$$

with $\zeta = T _ { f } \sum _ { k } r _ { k }$

Proof sketch. Solve (20) for $\mathrm { s g n } _ { j } \tilde { \Lambda } _ { i j } ( g _ { i j } - \bar { g } _ { j } ^ { o b s } )$ , substitute it into $- \lambda ( g _ { i j } - \bar { g } _ { i } ^ { o b s } ) \mathbb { E } [ c _ { i j } ]$ , and use $\pi _ { j } = r _ { j } / \sum _ { k } r _ { k }$ . To first order, $\begin{array} { r } { \Delta D = \sum _ { i j } ( \partial D / \partial W _ { i j } ) \Delta \bar { W } _ { i j } } \end{array}$ . The column-rescaling term contributes ∝ $\textstyle \sum _ { i } W _ { i j } { \bar { \partial } } D / \partial W _ { i j } = 0$ □

c) Intuition.: On average the rule is gradient descent on $D ,$ preconditioned by the positive diagonal factor $E _ { j } W _ { i j }$ (which never has to be computed), plus a column rescaling that cannot change D.

d) Flow-engine case.: With $a = X = 0$ , excitation only and equal gains, $T ^ { o b s } = T$ and $C _ { j } = 0$ , and Lemma E reduces to

$$
\frac { \partial D } { \partial W _ { i j } } = \frac { \pi _ { j } } { d _ { j } } \left( g _ { i j } - \bar { g } _ { j } \right) , \qquad \bar { g } _ { j } = \sum _ { k } T _ { k j } g _ { k j } .\tag{22}
$$

The flow engine replaces the counter by its expectation $J _ { i j }$ , giving $\mathbb { E } [ \Delta W _ { i j } ] = - \lambda W _ { i j } \partial D / \partial W _ { i j }$ : gradient descent with each coordinate scaled by its own weight.

## G. Routing exemption

The policy is carried by cross-module flow: controller → action and controller → latent. Applied to those synapses, the map term would drive the controller’s exit flow to zero and starve the action modules. The routing pairs are therefore exempt: $g _ { b a } = 0$ for $a = { \mathcal { C } }$ and $b \in \mathcal { A } \cup \mathcal { L }$ . The controller then acts as an input layer, and the map term seals the latent and action modules.

## H. Task term

The task is learned by a TD-modulated, action-gated eligibility. Take the surrogate policy $\begin{array} { r } { \tilde { P } ( a ) = J _ { a } ^ { i n } / \sum _ { a ^ { \prime } \in \mathcal { A } } J _ { a ^ { \prime } } ^ { i n } } \end{array}$ , with $\begin{array} { r } { J _ { b } ^ { i n } = \sum _ { i \in b } \sum _ { j } J _ { i j } } \end{array}$ . Its score with respect to $J _ { i j }$ depends only on the module of i:

$$
h _ { b } = \frac { \mathbb { I } [ b = a ^ { * } ] } { J _ { a ^ { * } } ^ { i n } } - \frac { \mathbb { I } [ b \in \mathcal { A } ] } { \sum _ { a ^ { \prime } \in \mathcal { A } } J _ { a ^ { \prime } } ^ { i n } } , \qquad \bar { h } _ { j } = \sum _ { k } T _ { k j } h _ { m ( k ) } ,\tag{23}
$$

where $a ^ { * }$ is the chosen action. The eligibility trace and the update are

$$
\begin{array} { r } { e _ { i j }  ( 1 - \frac { 1 } { \tau _ { e } } ) e _ { i j } + c _ { i j } ( h _ { m ( i ) } - \bar { h } _ { j } ) , \quad \quad \Delta W _ { i j } = \eta \delta _ { f } e _ { i j } , } \end{array}\tag{24}
$$

with the TD error and a linear critic trained by normalized LMS:

$$
\begin{array} { c } { \delta _ { f } = r _ { f } + \gamma V _ { f + 1 } - V _ { f } , \qquad V _ { f } = u \cdot x _ { f } , } \\ { u  u + \eta _ { V } \delta _ { f } \frac { x _ { f } } { \vert \vert x _ { f } \vert \vert ^ { 2 } } . } \end{array}\tag{25}
$$

In the flow engine, by the same Lemma, the increment of (24) is $W _ { i j } \partial \log \tilde { P } ( a ^ { * } ) / \partial W _ { i j }$ : a one-hop policy-gradient score.

## I. Homeostasis

A presynaptic homeostasis term holds each neuron’s output weight near 1:

$$
\Delta W _ { i j } = - \epsilon _ { h } ( d _ { j } - 1 ) \frac { \pi _ { j } W _ { i j } } { d _ { j } } .\tag{26}
$$

It rescales column j uniformly, so it leaves $T ,$ the walk and D unchanged. The full per-frame update, Eq. (6), is the sum of the task, structure and homeostasis terms.

## J. Computing the counter in each engine

Each engine uses the same rule. What differs is how $c _ { i j }$ is computed.

a) Branching.: The neuron is $\begin{array} { r } { u _ { i } ( t + 1 ) = \sum _ { i } b _ { i j } x _ { j } ( t ) } \end{array}$ + noise, which spikes with probability $p ( u ) = \mathrm { c l i p } ( u / 2 \theta , 0 , 1 )$ The kick $b _ { i j } = \mathrm { s g n } _ { j } w _ { 0 } T _ { i j }$ is output-normalized. Removing $j ^ { \dagger } \mathbf { s }$ kick changes i’s firing probability by a known amount, so the counter is exact:

$$
c _ { i j } = \sum _ { t } x _ { j } ( t ) \left[ p \left( u _ { i } ( t + 1 ) \right) - p \left( u _ { i } ( t + 1 ) - b _ { i j } \right) \right] .\tag{27}
$$

b) LIF and QIF.: A kick of size b is an input pulse of area ${ \boldsymbol { \tau } } b .$ In linear response it changes $i \mathbf { \ ' } _ { \mathbf { S } }$ spike count by $\hat { G } _ { i } \tau b .$ where $\hat { G } _ { i }$ is i’s gain (the slope of its rate transfer). So

$$
c _ { i j } = \tau \hat { G } _ { i } b _ { i j } n _ { j } ,\tag{28}
$$

an outer product of the presynaptic spike count and the postsynaptic gain. For LIF the gain comes from the Siegert rate formula, or from the exact Markov transfer of the discretized neuron. For QIF it comes from regressing the rate on the injected noise.

## K. Stability

The walk always exists, but the counters rely on linear response around a stable operating point. With Λ the signed offspring matrix (extra spikes of i per spike of $j )$ , the conditions are

$$
\mathrm { b r a n c h i n g \ ( m a p ) } \colon \operatorname* { m a x } _ { k } | \lambda _ { k } ( \Lambda ) | < 1 ,
$$

$$
\mathrm { L I F } , \mathrm { Q I F ~ ( c o n t i n u o u s ) } \colon \operatorname* { m a x } _ { k } \mathrm { R e } \lambda _ { k } ( \Lambda ) < 1 .\tag{29}
$$

The network monitors them through a module-level offspring table estimated from the counters.

## L. Map-STDP communities in the TD-STDP backbone

In the hybrid model, the actor of the TD-STDP actor-critic of Chung and Kozma [39] is organized into modules:

• Controller. The controller C is the Fourier input population.

• Action communities. Each action’s excitatory actor population is one module.

• Recurrence. Recurrent synapses R connect all actor excitatory neurons, with output-normalized kicks $b _ { i j } = \chi _ { r e c } R _ { i j } / d _ { j }$

• Routing. The input → actor weights are exempt routing, learned by the TD-STDP rule alone.

• Where the map term acts. The structural term acts on R only, with the LIF counter $c _ { i j } = \tau \hat { G } _ { i } b _ { i j } n _ { j }$ accumulated per window.

The action is sampled from a softmax over community rates, $p _ { k } \propto \exp ( \alpha _ { a } \bar { \rho } _ { k } )$ . This is equivalent to a race of escapenoise decision units with constant hazards $\exp ( \alpha _ { a } \bar { \rho } _ { k } )$ , in which the first unit to spike wins [40]. Each unit’s local score $\mathbb { I } [ k = a ] - \Lambda _ { k } ( t ^ { * } )$ , with $\Lambda _ { k }$ its integrated hazard up to the decision time $t ^ { * }$ , has expectation $A _ { k } - p _ { k }$ given the winner, where A is the one-hot chosen action. So the community credit $h _ { b } \propto A _ { b } - p _ { b }$ of the backbone is available locally. The task term on R uses the one-hop credit $h _ { m ( i ) } - \bar { h } _ { j }$ of (24).

## APPENDIX C

## RANDOM WALK ENTROPY AND SNN ENERGY USAGE

In this section we discuss in more depth how entropy of a random walk abstraction of SNN inference relates to the energy usage of the network (Fig. 5 and Table III).

## A. Energy Usage

In this supplementary study, we measure energy usage through spiking operations (SOPs). An SOP is one spike delivered across one synapse. Each spike is weighted by the number of outgoing synapses of the neuron that fired it:

$$
\mathrm { S O P s } = \sum _ { j } n _ { j } \cdot | \{ i : W _ { i j } > 2 w _ { \operatorname* { m i n } } \} |
$$

Here $n _ { j }$ is the spike count of neuron $j .$ The count only includes synapses above twice the weight floor $( w _ { \mathrm { m i n } } = 1 0 ^ { - 3 } )$ , the “effective” synapses. Synapses the rule has pushed down to the floor are treated as pruned, so an SOP is only counted where a synapse effectively exists. SOPs are computed on frozen weight models without any continual learning.

## B. L<sub>W</sub> and $L _ { o b s }$

There are two measurements of info-theoretic entropy in the random-walk abstraction: the map equation of the observed walk $( L _ { \mathrm { o b s } } )$ and the map equation of the walk on the trained weights $( L _ { W } )$ . Both description lengths are the map equation of a walk with visit rates π and transitions $T$ over the modules m (see Equation (5)). They differ in the walk they use:

$$
{ \cal L } _ { W } = { \cal L } ( \pi ^ { W } , T ^ { W } ) ,
$$

$$
\begin{array} { r } { T _ { i j } ^ { W } = W _ { i j } \Big / \sum _ { k } W _ { k j } , } \end{array}\tag{30}
$$

$$
\begin{array} { r } { \pi ^ { W } = ( 1 - \alpha ) T ^ { W } \pi ^ { W } + \frac { \alpha } { N } \mathbf { 1 } , \quad \alpha = 0 . 2 , } \end{array}
$$

$$
L _ { \mathrm { o b s } } = \left. L ( \pi _ { f } , T _ { f } ^ { \mathrm { o b s } } ) \right. _ { f } ,
$$

$$
\begin{array} { r } { T _ { i j } ^ { \mathrm { o b s } } = \kappa _ { i j } \bigg / \sum _ { k } \kappa _ { k j } , } \end{array}\tag{31}
$$

$$
\begin{array} { r } { \pi _ { f , j } = n _ { j } \Big / \sum _ { k } n _ { k } . } \end{array}
$$

L<sub>W</sub> is computed from the weights alone (the recurrent weights R for TD-STDP) under uniform teleportation. $L _ { \mathrm { o b s } }$ is computed from spikes and averaged over frames $f .$ In it, $\kappa _ { i j }$ counts spikes of i within the engine’s window after a spike of $j ,$ and $n _ { j }$ is the spike count of neuron $j$ in the frame.

## C. Info-theoretic entropy and downstream energy use

No experiment was run on the flow model as there are no spiking operations in the flow model. The full correlation table can be found in Table III. Overall, there is general agreement between energy usage and description length (Spearman correlations: +0.64 overall SOPs vs $L _ { o b s }$ , +0.53 overall SOPs vs $L _ { W } )$ , showing strong evidence for lower description length yielding lower energy usage. Moreover, there is even a weak negative correlation between performance and the two description length quantities (-0.19 Return vs $L _ { o b s }$ and vs $L _ { W } ,$ ), providing a small amount of evidence for Hinton’s compressibility argument. However, the agreement between info-theoretic entropy via description length and downstream energy usage, as well as the two measures of description length, depends on the model and the task.

The correlation between SOPs and $L _ { o b s }$ becomes weak and even goes slightly negative in the full modularity regime in the TD-STDP backbone on Acrobot (-0.13 Spearman correlation). The correlation between SOPs and $L _ { W }$ , and between $L _ { o b s }$ and $L _ { W }$ drops heavily on LunarLander, indicating that preemptive info-theoretic analysis holds weaker on this task.

In Figure 5, showing the correlation between SOPs and the two description lengths across $p _ { c r o s s }$ values, we see a similar trend. There is a positive correlation between SOPs and $L _ { o b s }$ across all three tasks (CartPole: +0.93, Acrobot: +0.43, LunarLander: $+ 0 . 4 1 )$ . The correlation between all three quantities are strongest in CartPole (SOPs vs $L _ { o b s } \colon + 0 . 9 3$ , SOPs vs $L _ { W } \colon + 0 . 8 3$ $L _ { o b s } \ \mathrm { v s } \ L _ { W } \colon + 0 . 7 8 )$ . And LunarLander holds the weakest correlation, with theoretical description length analysis even serving counterintuitively (SOPs vs $L _ { o b s } \colon + 0 . 4 1$ , SOPs vs $L _ { W } \colon - 0 . 4 3 , L _ { o b s }$ vs $L _ { W } \colon - 0 . 2 4 )$ . Across the $p _ { c r o s s }$ values, in the plots on the rightmost column, correlations hold positive consistently in CartPole and Acrobot, and while SOPs and $L _ { o b s }$ hold positive, the correlations with $L _ { W }$ drop negative consistently. This opens the question of what attributes the LunarLander task has for preemptive info-theoretic analysis to hold so counterintuitively. This question is explored in the following subsection.

## D. The Special Case of LunarLander

Synaptic operations (SOPs) here track the number of surviving synapses, not activity. Within each task, column spikes per decision barely change across regimes: about 5 in CartPole, 92 in Acrobot and 211–223 in LunarLander, since the rate-matching settings of these studies hold firing rates fixed. SOPs therefore vary only through the effective fan-out, the number of a neuron’s recurrent synapses still above threshold (Table IV).

$L _ { W }$ , in contrast, measures how much of the weight crosses between columns. The two usually move together because one process drives both: the map term removes cross-column synapses, lowering the synapse count and $L _ { W }$ at once. Without the map term, each task treats the cross-column synapses differently (Table V).

In CartPole the task rule removes the cross-column synapses by itself within about 200 episodes, so the partial arm ends up equal to the full one. In Acrobot the cross-column weights double, but the columns survive. In LunarLander the cross-column weights triple and most within-column synapses fall below threshold, mainly between episodes 2,500 and 5,500. The final within-column share of 0.31 is below the 0.77 of the random initial wiring, so the network ends up less modular than chance.

In LunarLander’s partial arm, then, the pruned synapses are the within-column ones, and fewer synapses go with more cross-column flow. This makes SOPs and $L _ { W }$ negatively correlated $( \rho = - 0 . 4 7$ over all regimes, −0.69 within the partial arm). With the map term on, the cross-column synapses go and all 15.6 within-column synapses stay, which is why the dynamic and full arms end up with the same SOPs and $L _ { W }$

Why the within-column synapses decay is not yet established. Candidates are the rate cost, homeostasis, or the task rule over LunarLander’s much longer training (10,000 episodes, against 400 for CartPole and 500 for Acrobot).

A fixed partition thus appears too rigid for this task, and unconstrained coupling too loose. The map term alone has limitations expressing a middle ground: its modulator $g _ { b a } = M _ { a } ^ { * } \mathbb { I } [ b \neq a ]$ charges every exit from column a the same marginal cost, whatever the target column b. A natural extension is an adaptive $K \times K$ module-pair table learned for instance from the task’s credit, so that useful directed links between action columns are kept while the rest are pruned. We leave this to future work.

<table><tr><td></td><td></td><td></td><td>p_cross</td><td></td><td></td><td></td></tr><tr><td>● 0</td><td>● 0.0125</td><td>● 0.025</td><td>● 0.05</td><td>● 0.1</td><td>● 0.2</td><td>● 0.5</td></tr></table>

<table><tr><td colspan="2">Arm</td></tr><tr><td>Map term on (dynamic)</td><td>▲Map term off (partial; full at 0)</td></tr></table>

a CartPole · TD-STDP race readout, recurrent action columns  
![](images/5f205349582ca9ab8a2501f25d74831215a12549a51f2ebd2778fce5720a187a.jpg)

![](images/f0aa2a559cfb749055efb9a8b3488abf12dcd449f605ab5c22e50125aa6686f2.jpg)

b Acrobot · TD-STDP race readout, recurrent action columns  
![](images/1f24238f1e2e955816cefd9f50b43349b0f9eb02e2758b02b094b3349496a084.jpg)

![](images/91d69f78f69741e21780d11d031076b8d5e93fa4c93e4bc43d46cf8dfcd624b1.jpg)

![](images/c131bbf3c40790738f982974f2c9f19e6ed7e1a6af600e1bed68b95f0ed65f13.jpg)

![](images/bb151e56502f498ff4021fa25002d29751000486f8e5cf7b4ec30642ea30147a.jpg)

![](images/ce59a1b18370ce00f52d18f546b660f502c0cddcc4c7fb91c3e7a4b017421076.jpg)

![](images/a24b6fa368d368130eb87d92f5122c9ba25bc6c36fb74906aa037a7bca73fcff.jpg)

c LunarLander · TD-STDP race readout, recurrent action columns  
![](images/98683fddbf1d5e7db5e4edcb4d562b9717a66335b6a22fa18b2416b219857687.jpg)

![](images/9037e203fc5df4542b73824b398a33e13d391dae9c988e4ea10bbafffedbdd15.jpg)

![](images/fefeaffb26599eb0cc83ea6d792471a7a8d7bb5c59cdd9fe56d37a46e75d9429.jpg)

![](images/8813c176204c907df9c591445db2dd55bece52a5ab554220470f602a6eb2257d.jpg)  
Fig. 5. Energy use and description length across the inter-module connectivity sweep in the TD-STDP actor-critic backbone, for (a) CartPole, (b) Acrobot and (c) LunarLander. Each point is one final network $( 7$ values of $p _ { c r o s s } \ \times \ 2$ map-term settings $\times ~ 2 0$ seeds, $n = 1 4 0 ) $ ; color gives $p _ { c r o s s }$ and shape the map-term setting (dynamic with the term on, partial with it off, full at $p _ { c r o s s } = 0$ with it off). Columns 1–3: effective synaptic operations (SOPs) per decision against $L _ { o b s }$ and $L _ { W } ,$ , and $L _ { o b s }$ against $L _ { W }$ , with the Spearman $\rho$ over all points above each panel. Column 4: Spearman $\rho$ within each $p _ { c r o s s }$ value $( n = 2 0$ per setting), with bootstrap intervals.

TABLE III  
SPEARMAN CORRELATION BETWEEN ENERGY USE, RANDOM-WALK DESCRIPTION LENGTH AND PERFORMANCE OF THE FINAL NETWORKS: EFFECTIVE SYNAPTIC OPERATIONS (SOPS) PER DECISION, THE MAP EQUATION OF THE OBSERVED WALK $\left( L _ { \mathrm { o b s } } \right)$ AND OF THE WALK ON THE TRAINED WEIGHTS (L ), AND THE MEAN RETURN OF THE FROZEN FINAL NETWORK. ρ WITH ITS BOOTSTRAP 95% INTERVAL (2000 RESAMPLES OVER SEEDS) BELOW; <sup>∗</sup> p < 0.05. MAP-STDP V4: PER SPIKING ENGINE, POOLED OVER THE THREE MODULARITY REGIMES. TD-STDP: ALL REGIMES POOLED, THEN WITHIN EACH REGIME. ALL SETTINGS: EACH VARIABLE RANKED WITHIN ITS SETTING (THREE V4 ENGINES, THREE TD-STDP TASKS) AND POOLED; BOOTSTRAP STRATIFIED BY SETTING. —: EVERY FROZEN TD-STDP CARTPOLE NETWORK REACHES THE 500 CEILING, SO ITS RETURN IS CONSTANT; THAT SETTING IS LEFT OUT OF THE ALL-SETTINGS RETURN COLUMNS (n = 300).
<table><tr><td></td><td></td><td colspan="2">SOPs vs</td><td> $L _ { \mathrm { o b s } }$  VS</td><td colspan="2">Return vs</td></tr><tr><td>Group</td><td>n</td><td> $L _ { \mathrm { o b s } }$ </td><td> $L _ { W }$ </td><td> $L _ { W }$ </td><td> $L _ { \mathrm { o b s } }$ </td><td> $L _ { W }$ </td></tr><tr><td colspan="7">Map-STDP v4, CartPole</td></tr><tr><td>Branching</td><td>60</td><td> $+ 0 . 6 1 ^ { * }$  [+0.35, +0.77]</td><td> $+ 0 . 7 1 ^ { * }$  [+0.52, +0.81]</td><td> $+ 0 . 4 9 ^ { * }$  [+0.21,+0.67]</td><td> $- 0 . 4 9 ^ { * }$  [-0.70,-0.24]</td><td>-0.19 [−0.44, +0.08]</td></tr><tr><td>QIF</td><td>60</td><td>+0.95* [+0.90, +0.97]</td><td>+0.93* [+0.87, +0.96]</td><td>+0.92* [+0.86, +0.95]</td><td>-0.09 [−0.35, +0.18]</td><td>-0.11</td></tr><tr><td>LIF</td><td>60</td><td>+0.91* [+0.86, +0.94]</td><td>+0.91* [+0.86, +0.93]</td><td>+0.92* [+0.86, +0.95]</td><td>-0.50* [−0.68, -0.28]</td><td>[−0.37, +0.17] -0.45* [−0.64, −0.23]</td></tr><tr><td colspan="7">TD-STDP, CartPole</td></tr><tr><td>All regimes</td><td>60</td><td>+0.88* [+0.78, +0.94]</td><td>+0.72* [+0.53, +0.84]</td><td>+0.71* [+0.52, +0.84]</td><td></td><td></td></tr><tr><td>Dynamic</td><td>20 20</td><td>+0.83* [+0.56, +0.95] +0.65*</td><td>+0.55* [+0.11,+0.85] -0.23</td><td>+0.66* [+0.20, +0.93] -0.29</td><td></td><td></td></tr><tr><td>Partial Full</td><td>20</td><td>[+0.24,+0.87] +0.65*</td><td>[−0.61,+0.21] +0.25</td><td>[−0.73, +0.24] +0.02</td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>All regimes</td><td>60</td><td>+0.02 [−0.24, +0.28]</td><td>+0.38* [+0.11, +0.63]</td><td>+0.07 [−0.19, +0.31]</td><td>-0.04 [−0.33,+0.24]</td><td>-0.10 [−0.34,+0.14]</td></tr><tr><td>Dynamic</td><td>20</td><td>+0.32 [−0.21,+0.77] +0.18</td><td>+0.46* [+0.03, +0.80] -0.15</td><td>+0.61* [+0.24, +0.82]</td><td>+0.02 [−0.44, +0.45]</td><td>+0.26 [−0.24, +0.66]</td></tr><tr><td>Partial Full</td><td>20</td><td>[−0.32, +0.61] -0.13</td><td>[−0.59,+0.41] +0.18</td><td>-0.64* [−0.87, −0.21] +0.06</td><td>+0.13 [−0.36, +0.59] -0.39</td><td>-0.30 [−0.71,+0.20] -0.02</td></tr><tr><td></td><td>20</td><td>[−0.59, +0.40]</td><td>[−0.34, +0.60]</td><td>[−0.41, +0.51]</td><td>[−0.75, +0.08]</td><td>[−0.52, +0.41]</td></tr><tr><td colspan="7">TD-STDP, LunarLander All regimes 60 +0.49*</td></tr><tr><td>Dynamic</td><td>20</td><td>[+0.25, +0.69] +0.11</td><td>[-0.69,-0.17] +0.29</td><td>[−0.30, +0.30] +0.26</td><td>[−0.09, +0.43] +0.07</td><td>[−0.37,+0.16] -0.09</td></tr><tr><td></td><td>20</td><td>[−0.44, +0.55]</td><td>[−0.14, +0.65] -0.69*</td><td>[−0.27, +0.71] -0.39</td><td>[−0.50, +0.54] +0.65*</td><td>[−0.52, +0.41] -0.79*</td></tr><tr><td>Partial Full</td><td>20</td><td>+0.58* [+0.13, +0.79] +0.26</td><td>[−0.87, -0.35] +0.49*</td><td>[−0.72,+0.12] -0.14</td><td>[+0.22, +0.88] +0.18</td><td>[−0.93, -0.50] -0.08</td></tr><tr><td>All settings</td><td></td><td>[−0.24, +0.69]</td><td>[+0.03, +0.79]</td><td>[−0.60,+0.37]</td><td>[−0.30, +0.60]</td><td>[−0.54, +0.41]</td></tr></table>

TABLE IV  
EFFECTIVE FAN-OUT (RECURRENT SYNAPSES PER NEURON ABOVE $2 w _ { \mathrm { m i n } } )$ OF THE FINAL NETWORKS, MEAN OVER SEEDS.
<table><tr><td>Task</td><td>Dynamic</td><td>Full</td><td>Partial</td></tr><tr><td>CartPole</td><td>3.3</td><td>9.5</td><td>9.5</td></tr><tr><td>Acrobot</td><td>7.1</td><td>9.6</td><td>8.3</td></tr><tr><td>LunarLander</td><td>15.6</td><td>15.6</td><td>6.2</td></tr></table>

TABLE V  
RECURRENT WEIGHTS OF THE PARTIAL ARM (MAP TERM OFF) FROM THE START TO THE END OF TRAINING, MEAN OVER SEEDS. WITHIN-COLUMN SYNAPSES: PER NEURON, ABOVE $2 w _ { \mathrm { m i n } } .$ . WITHIN-COLUMN SHARE: FRACTION OF THE TOTAL RECURRENT WEIGHT INSIDE COLUMNS.
<table><tr><td>Task</td><td>Mean cross-column weight</td><td>t Within-column synapses</td><td>Within-column share</td></tr><tr><td>CartPole</td><td> $0 . 0 9 4  0 . 0 0 1$ </td><td> $9 . 5  9 . 5 $ </td><td> $0 . 9 1  1 . 0 0$ </td></tr><tr><td>Acrobot</td><td> $0 . 0 8 4  0 . 1 8 7$ </td><td> $9 . 5  6 . 9$ </td><td> $0 . 8 3  0 . 6 2$ </td></tr><tr><td>LunarLander</td><td> $0 . 0 4 8  0 . 1 4 0$ </td><td> $\mathbf { 1 5 . 6  4 . 5 }$ </td><td> $\mathbf { 0 . 7 7  0 . 3 1 }$ </td></tr></table>