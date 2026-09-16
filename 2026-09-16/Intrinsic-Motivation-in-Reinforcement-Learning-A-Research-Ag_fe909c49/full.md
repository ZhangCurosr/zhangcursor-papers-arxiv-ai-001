# Intrinsic Motivation in Reinforcement Learning: A Research Agenda for Adaptive Self-Organisation

Anatoly Belikov<sup>a</sup>

<sup>a</sup>SingularityNET Foundation,

## Abstract

Biological cells can be viewed as individual, interacting agents whose collective dynamics give rise to adaptive behaviour at multiple levels of organisation, from individual cells through tissues to whole multicellular organisms. In this perspective and tutorial article we discuss whether intrinsic rewards in artificial neural systems can support adaptation, functional specialisation and higher-level selforganisation without a shared external objective. We review empowerment, curiosity, learning progress, information gain, unsupervised skill discovery, mutual information estimation and the use of world models for intrinsic reward computation. Particular attention is given to failure modes showing when such objectives do not produce sustained exploration or increasingly complex behaviour. We argue that more capable systems may require complementary objectives, communication, memory, learning at multiple temporal scales and environmental constraints. Based on this perspective, we outline three experimental directions. These include a resource-constrained environment in which otherwise stable behavioural attractors become unsustainable, allowing us to test whether environmental constraints can mitigate characteristic failure modes of intrinsic objectives. The network of recurrent agents with per-agent intrinsic rewards, and a hierarchical world-model agent in which exploratory motor competence develops before goal-directed behaviour. These experiments are intended to test whether intrinsic learning can lead to adaptive organisation at progressively higher levels.

Keywords: intrinsic motivation, mutual information, empowerment, curiosity, unsupervised skill discovery, world models, hierarchical reinforcement learning

## 1. Introduction

In this article, we consider intrinsic motivation in reinforcement learning. RL methods are well developed both theoretically and at the implementation level. They are highly efective when provided with a dense reward function. Some progress has been made in defining universal intrinsic reward functions that are applicable across diferent environments. Here, intrinsic means that the reward is computed only from what an agent observes during its lifetime, without using hidden properties of the environment. This is especially interesting given that nature does not define a reward function for biological organisms; instead, evolution must have led to the development of some kind of intrinsic preferences.

Existing intrinsic objectives exhibit characteristic failure modes, but some of them are naturally complementary and may mitigate one another’s limitations. We also analyse environmental constraints, such as limited energy, and structural features of biological systems, such as the existence of relatively global communication channels, that could be introduced into artificial environments to support the development and self-organisation of intrinsically motivated agents.

## 1.1. Article structure and contribution:

We propose a research direction focused on adaptability in artificial neural networks, with the goal of developing learning systems that could demonstrate adaptive behaviour similar to that observed in biological networks. We ask whether some of their functional properties—adaptation, memory, communication, functional diferentiation and the formation of higher-level organisation—can emerge in non-biological recurrent neural networks.

A broader question motivating the research agenda is whether such a learning system can produce progressively more complex adaptive behaviour and organisation at higher levels of organisation, similar to the formation of multicellular organisms.

The main contribution of the article is a comparison of what existing objectives optimise, an analysis of their failure modes, and a research agenda for studying how complementary per-agent objectives may produce higher-level adaptation.

We also propose a biologically inspired recurrent architecture and a specific environment suitable for evaluating intrinsic motivation methods.

We begin with motivation from biology. The next few sections provide a tutorial-style overview and comparison of empowerment, DIAYN (skill discovery), learning progress and information gain.

![](images/8855e61b8f35ffc2224716097d67049afc8390bb546969fbf7063c2241c7d3c1.jpg)  
Figure 1: Kidney tubules in the newt are made with a constant size, whereas cell size can vary drastically under polyploidy. The same shape achieved through diferent molecular mechanisms: cell to cell communication vs cytoskeleton bending. Adapted from [1].

The tutorial part, together with the appendix, contains derivations of the main identities and discusses interesting corner cases, including an analysis of why meaningful intrinsic rewards may fail to produce useful behaviour.

We provide an overview of some methods of world modelling and their possible use for intrinsic reward computation, and conclude with a biologically motivated research agenda: experiments intended to improve our understanding of the intrinsic motivation methods discussed.

This article is inspired by the work of Jürgen Schmidhuber, who developed the first formal computational methods for intrinsic motivation and artificial curiosity in reinforcement learning, as well as by Michael Levin’s work on adaptive behaviour in biological systems across diferent scales.

## 2. Motivation from biology

AI as a field often draws inspiration from biology. Recent discoveries point that there is at least some intelligence on diferent levels of organization. Molecular networks have memory including pavlovian conditioning, associative learning etc.

Also there is no large diference between neural cells and somatic cells. Somatic cells exhibit learning, problem solving, they also use electricity for communication.

Some notable examples of plasticity and problem solving in image 2

A  
![](images/efd5ca8b079d47eef68133f40de2ad7456b2a991edfa8c5be62467721fd3b3d4.jpg)  
Figure 2: Classical conditioning in biological network, for example drug-drug conditioning. Adapted from Figure 1 in [2].

![](images/d79d474504f8bbdcb64455a7feea6d9fe0b8103363573d68c963dbc779b93b86.jpg)  
Figure 3: Adaptation of neurons in planaria to barium chloride. After exposure to BaCL2 neurons in planaria’s head die of. New head has resistant neurons. It’s unlikely that planaria has ever been exposed to BaCL2 before. Adapted from Figure 5 in [3].

But perhaps the most striking and important case is cancer plasticity. Tumors switch between metabolic pathways (e.g., glycolysis to oxidative phosphorylation) to survive fluctuating conditions, a form of tissue-level adaptability.

For more detailed overviews, see Michael Levin’s interview Michael Levin: Intelligence Beyond the Brain and his presentation Bioelectricity: A Bridge between Physics and Cognition, by Way of Biology [4, 5].

## evolutionary trend hypothesis

There appears to be a trend in growing intelligence across biosphere. Both - upper cap and average(measured by total biomass). For insects it’s estimated that eusocial species make now around 50% [6] of insects biomass while paradoxically constituting only about 2% of species. Modern colony size are relatively recent invention [7] and there is growing evidence that colony size is a primary drive of specialisation [8]. This might appear obvious in hindsight given growing specialisation of people in modern economy. This transition provides evidence that evolution may increase not only the upper bound of intelligence, but also the weighted by biomass prevalence of complex adaptive organization.

$$
\begin{array} { r } { { \bf \dot { C } } _ { b a r } ( t ) = \frac { \sum _ { i } B _ { i } ( t ) \hat { C } _ { i } ( t ) } { \sum _ { i } B _ { i } ( t ) } } \end{array}
$$

here � is intelligence measure

## � - biomass

## i - cells, organisms or colonies.

Possible mechanism is random specialization of some species in intelligencebased adaptation with later evolutionary arms race.

## 2.1. Learning hierarchy

We can very roughly sort diferent learning types from more simple to more advanced forms, that likely appeared later in evolutionary history.

## 1. Non-associative learning

Habituation - decrease of response to non-harmful repeated stimulus

Sensitization - increase of response after exposure to strong or harmful stimulus.

## 2. Associative learning

Classical conditioning: learning of stimulus-outcome association .

Operant Conditioning: learning of action-outcome association.

## 3. Flexible individual ans socially mediated learning

Metacognition. This includes capacity to model agent’s own understanding e.g. being uncertain, seek more information, being able to assess own likelihood of error.

Spatial Learning & Navigation

Insightful Problem Solving, probably based on some internal modelling as opposed to trial and error.

Teaching & Pedagogy

Play

## 4. Cumulative Culture:

Cultural learning over generations, learning from purely symbolic input e.g. reading instructions.

There is no universally accepted hierarchy covering all forms of learning and cognition. So this classification is rather a heuristic. But there is a good evidence that non-associative mechanisms are evolutionarily ancient, whereas flexible planning, metacognitive control, teaching, and cumulative culture appeared much later.

Level 2 and Level 3 learning types can be observed already in insects. Bumblebee for example would play with appropriately sized balls without any external reward from only intrinsic motivation [10]. Another example is Portia spiders which employ a sophisticated hunting strategy involving mimicry, detours, and ambush tactics when hunting on other spiders. Portia often attacks other spiders in their own web, so it has to be clever. Sometimes it will literally lure the prey by mimicking vibrations of an insect being stuck. It has been reported that Portia can learn to hunt a new, never seen before spider by trial and error.

Portia is a very good example since it has a very small brain. Bumblebees have approximately 950k - 1 mln neurons vs Portia around 100k.

From observations we can conclude that it has:

Internal Representation: The ability to take a detour where the prey is out of sight for extended periods implies the spider is not just reacting to the prey's presence. It must be operating from a stored representation of the environment and the prey's location within it.

Object Permanence and Spatial Memory: Portia's behavior indicates it has a grasp of object permanence (knowing the prey still exists even when it can't be seen) and strong spatial memory to navigate the planned route.

![](images/800d9b1c4c4e7f0ac6fdaa9b09e04860cb1391055ca467bd1852c7bbec06887a.jpg)  
Figure 4: Xenobots - small, self-organizing robots made from frog embryonic cells (Xenopus laevis). This one is made of skin and cardiac cells. Image adapted from [9], Wikimedia Commons, licensed under CC BY 4.0.

Systematic Scanning: The process of systematically scanning its surroundings before a hunt is interpreted as the spider building a detailed mental map of the area, which it then uses to execute its plan.

Expectancy Violation: When a spider takes a detour and finds wrong number of pray it spent more time inspecting the scene. This implies it had an expectation of what it should find, based on its internal model, and that expectation was violated.

Also there is growing evidence about tool use by insects. Image 5 shows the experiment: researchers gave hungry ants containers with sugar water. Researchers then altered the surface tension of the sugar water by adding surfactant. When surfactant concentrations were over 0.05%, representing considerable drowning risk, ants were observed building the sand structures to syphon sugar water out of the container. These structures were never observed when ants foraged in containers of pure sugar water, indicating an adaptable approach to this novel tool use.

1. Biological organisms are adaptive on diferent levels of organization

2. Learn without backpropagation

3. All cells types can communicate with each other

4. Demonstrate goal-directed behaviour

Current neural architectures are more fragile than biological ones.

Our motivation is to determine if current intrinsic motivation methods would lead to learning and behaviour types observable in insects. Whether these types of behaviours are reproducible with intrinsic rewards.

Could intrinsic motivation in RL lead to the development of level 2 and level 3 learning types:

1) Does it lead to emergent communication and cooperation?

2) Lead to observational learning?

3) Does it lead to play and episodic learning?

![](images/cd6fb72ee8d4ec5d69911e2bd4903014555866930e8c363581e7deffa62eb46c.jpg)

![](images/dcf4fc26b18dae5a4515af2db12b1d547b1e97ee86a35727b810522cc2de2e86.jpg)

![](images/4429c35af243af253176e199abc21bb90e54e97ba9535893685690bce95f6da1.jpg)

![](images/bc6fa66801fe6c00309a485956e0e85b5965bba4a12ea19429ce110d2d742ba0.jpg)

![](images/3192f27d91d4741f3abfeb20695608639a48a56a4ac2fb656fbaea9583ad6ad3.jpg)  
Figure 5: Examples of learned behaviour, play, complex hunting strategy and tool use in arthropods. The trained bumblebee ball-rolling image is a generated illustration of the experiment reported by Loukola et al. [11]. The tool-selection image is a frame from the supplementary video by Chow et al. [12], licensed under CC BY 4.0. The spontaneous bumblebee ball-rolling image is adapted from Galpayage Dona et al. [10] under CC BY 4.0. The Portia image, “Male Portia waiting for opportunity to predate buttonspider,” is by i\_c\_riddell under CC BY. The ant tool-use image is a generated illustration of the experiment reported by Zhou et al. [13]. Each image links to its source or related publication.

## 3. Basic mathematical definitions

Entropy

For a discrete random variable � with probability mass function $p ( y )$ we define entropy H:

$$
H ( Y ) = - \sum _ { y } p ( y ) \log p ( y ) .\tag{1}
$$

$$
H ( Y | Z ) = - \sum _ { y , z } p ( y , z ) \log p ( y | z ) .\tag{2}
$$

Kullback–Leibler divergence

For two probability distributions � and � over the same sample space,

$$
D _ { \mathrm { K L } } ( P \Vert Q ) = \sum _ { x } p ( x ) \log { \frac { p ( x ) } { q ( x ) } } = \mathbb { E } _ { x \sim P } \left[ \log p ( x ) - \log q ( x ) \right] .\tag{3}
$$

Mutual information

Mutual information of two random variables Y, Z

$$
I ( Y ; Z ) : = D _ { K L } \big ( P ( Y , Z ) \| P ( Y ) P ( Z ) \big )\tag{4}
$$

Mutual information defined via entropy

$$
I ( Y ; Z ) : = H ( Y ) - H ( Y \mid Z ) = H ( Z ) - H ( Z \mid Y )\tag{5}
$$

Pointwise mutual information $\begin{array} { l } { \mathrm { P M I } ( x , y ) ~ = ~ \log \frac { p ( x , y ) } { p ( x ) p ( y ) } ~ } \end{array}$ , where x and $\mathsf { y }$ are events, not random variables.

It’s possible for PMI to be negative, but it’s expectation is always non-negative.

$$
I ( X ; Y ) = \mathbb { E } _ { ( x , y ) \sim p ( x , y ) } [ \mathrm { P M I } ( x , y ) ]
$$

## 3.1. Reinforcement learning

Reinforcement learning objective - maximising expected discounted reward

$$
J ( \pi ) : = \mathbb { E } _ { p _ { \pi } ( \tau ) } \left[ \sum _ { t } \gamma ^ { t } r _ { t } \right]\tag{6}
$$

� - policy.

$\gamma$ - discount factor in range (0, 1]

$s _ { t }$ - state at time t.

$p _ { \pi } ( \tau )$ - trajectory distribution under policy.

For a concise introduction to policy-gradient methods, see [14].

## 4. Empowerment in reinforcement learning

Mutual information between agent’s action and observable states can be used as reward. In this section, we derive such a reward and discuss its relation to empowerment. Empowerment is defined as channel capacity from agent’s action to subsequent state [15]:

$$
{ \mathcal { E } } ( s ) = \operatorname* { m a x } _ { p ( a \mid s ) } I ( A ; S ^ { \prime } \mid S = s ) .\tag{7}
$$

We can optimise policy to increase $I _ { \pi } ( A ; S ^ { \prime } | s ) ; I _ { \pi } ( A ; S ^ { \prime } | s ) \le \mathcal { E } ( s )$ gives us lower-bound estimation of empowerment.

We will refer to this quantity as on-policy empowerment.

In order to gain some intuition we will consider diferent formulations of the same quantity:

$$
I ( A ; S ^ { \prime } \mid s ) = D _ { K L } \big ( p ( a , s ^ { \prime } \mid s ) \| p ( a \mid s ) p ( s ^ { \prime } \mid s ) \big ) .\tag{8}
$$

$$
I ( A ; S ^ { \prime } \mid s ) = H ( S ^ { \prime } \mid s ) - H ( S ^ { \prime } \mid A , s ) .\tag{9}
$$

$$
I ( A ; S ^ { \prime } \mid s ) = H ( A \mid s ) - H ( A \mid S ^ { \prime } , s ) .\tag{10}
$$

We can rewrite $I ( A ; S ^ { \prime } \mid s )$ as expectation over events: First expand $H ( S ^ { \prime } \mid s )$

$$
H ( S ^ { \prime } \mid s ) = - \sum _ { s ^ { \prime } } p ( s ^ { \prime } \mid s ) \log p ( s ^ { \prime } \mid s ) .\tag{11}
$$

Since $\begin{array} { r } { p ( s ^ { \prime } \mid s ) = \sum _ { a } p ( a , s ^ { \prime } \mid s ) } \end{array}$

$$
H ( S ^ { \prime } \mid s ) = - \sum _ { s ^ { \prime } , a } p ( a , s ^ { \prime } \mid s ) \log p ( s ^ { \prime } \mid s ) = - \mathbb { E } _ { p ( a , s ^ { \prime } \mid s ) } \log p ( s ^ { \prime } \mid s ) .\tag{12}
$$

Second term $H ( S ^ { \prime } \mid A , s )$

$$
H ( S ^ { \prime } \mid A , s ) = - \sum _ { a , s ^ { \prime } } p ( a , s ^ { \prime } \mid s ) \log p ( s ^ { \prime } \mid a , s ) = - \mathbb { E } _ { p ( a , s ^ { \prime } \mid s ) } \log p ( s ^ { \prime } \mid a , s ) .\tag{13}
$$

Substituting Eqs. (12) and (13) into Eq. (9) gives

$$
\begin{array} { r l } & { I ( A ; S ^ { \prime } \mid s ) = - \mathbb { E } _ { p ( a , s ^ { \prime } \mid s ) } \log p ( s ^ { \prime } \mid s ) + \mathbb { E } _ { p ( a , s ^ { \prime } \mid s ) } \log p ( s ^ { \prime } \mid a , s ) } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } = \mathbb { E } _ { p ( a , s ^ { \prime } \mid s ) } \left[ \log p ( s ^ { \prime } \mid a , s ) - \log p ( s ^ { \prime } \mid s ) \right] . } \end{array}\tag{14}
$$

Here, s is the current state, $S ^ { \prime }$ is the next state and � is the action taken in state s.

� and $S ^ { \prime }$ are random variables but � is not.

It’s important to note that removing conditioning on the current state �, as in $I ( A ; S ^ { \prime } )$ or $I ( A ; S )$ will lead to a diferent, possibly degenerate solution.

Here $I ( A ; { \cal S } ^ { \prime } )$ is action - future state mutual information and $I ( A ; S )$ - action - current state.

Maximising $H ( S ^ { \prime } ) - H ( S ^ { \prime } | A )$ means that future state $S ^ { \prime }$ should be predictable from action alone, and $I ( A ; S ) = H ( S ) - H ( A \mid S )$ means the agent should directly map current state to action. For example always choose turn left in one room and turn right in an another room. In this case $I ( A ; S )$ will be high, but $I ( A ; { \cal S } ^ { \prime } | s )$ close to zero.

What is the behaviour that is encouraged by this type of reward?

First term that is maximised $H ( S ^ { \prime } \mid s )$ is conditional entropy of future state $S ^ { \prime }$

Entropy is low for spiky, concentrated probability distributions and high for more even distributions.

A binary random variable with probabilities (0.5,0.5) has an entropy of one bit. A variable with 10 possible outcomes with probability $p ( s _ { i } ) ~ = ~ 0 . 1$ entropy is ${ \sim } 3 . 3 2$ bits.

So $H ( S ^ { \prime } \mid s )$ is high then our agent visits a large and diverse set of states. To be more precise it means for given state � policy could achieve diverse set of future states. Global state visitation entropy $H ( S ^ { \prime } )$ is related to conditional as $H ( S ^ { \prime } ) =$ $H ( S ^ { \prime } \mid S ) + I ( S ^ { \prime } ; S )$ , so $H ( S ^ { \prime } )$ is at least as large as $H ( S ^ { \prime } \mid S )$ . What it means in practice agent could stay in an relatively small area with a lot of achievable states it could switch between.

Second term is minimised: $H ( S ^ { \prime } \mid A , s )$ , this is the entropy of the future state given the current state and action. This term encourages policy to take actions that have predictable outcomes. In a deterministic environment we have $H ( S ^ { \prime } \mid A , s ) =$ 0, so only the first term is optimised.

Consider this grid world, with actions up, down, left, right:

![](images/f3a3850d52f2f7bdd0ee3b7f6e87bd689d542cf9a4bfc5b552b42f110375d5c1.jpg)

Being in the central cell will have highest value of $\begin{array} { r } { H ( S ^ { \prime } \mid s ) , - \sum _ { i = 0 } ^ { 3 } 0 . 2 5 l o g _ { 2 } 0 . 2 5 = } \end{array}$ 2 bits.

We can formalise empowerment use as intrinsic reward: $\begin{array} { r l } { \sum _ { t } \gamma ^ { t } \operatorname { P M I } ( s _ { t + 1 } , a | s _ { t } ) } \end{array}$ following the objective in Eq. (6).

Since MI is an expected value of PMI it’s close to original definition:

$$
\mathbb { E } \left[ r _ { t } ^ { \operatorname { P M I } } | S _ { t } = s \right] = I _ { \pi } \left( A ; S ^ { \prime } | S = s \right) \le \mathcal { E } \left( s \right)
$$

Technical note: Empowerment is a property of a state, environment dynamics and the action space, not of our policy. It’s already defined as maximum capacity and we can’t literally optimise it. A behavioural policy may be trained both to approach this capacity and to visit states in which the capacity is high. We use the term ”empowerment maximisation” for such cases.

## 4.1. Inverse model trick

We can train a model to predict the action that caused a transition from state $s _ { t }$ to state $s _ { t + 1 }$ and then use it as empowerment estimator. Here is how it works:

We can then estimate empowerment with equation 10.

$$
I ( A ; S ^ { \prime } \mid s ) = H ( A \mid s ) - H ( A \mid S ^ { \prime } , s )
$$

By definition we have for entropies:

$$
H ( A \mid s ) = - \mathbb { E } _ { a \sim \pi ( \cdot \mid s ) } \log \pi ( a \mid s ) .\tag{15}
$$

$$
\begin{array} { r } { H ( A \mid S ^ { \prime } , s ) = - \mathbb { E } _ { \underset { a \sim \pi ( \cdot \mid s ) } { \mathrm { \ell } } } \log p ( a \mid S ^ { \prime } , s ) . } \end{array}\tag{16}
$$

The distribution $p ( a \mid s )$ is known. It is the action distribution under the current policy $\pi ( a | s )$ . Distribution of actions given previous and future states $p ( a \mid s ^ { \prime } , s )$ is not known, but we can approximate it with additional distribution $q ( a \mid s ^ { \prime } , s )$ First we plug it in conditional entropy $H ( A \mid S ^ { \prime } , s )$

$$
H ( A \mid S ^ { \prime } , s ) = - \mathbb { E } _ { \underset { s ^ { \prime } \sim p ( \cdot \mid a , s ) } { { \mathrm { \textsc { g } } } } } \log \frac { p ( a \mid S ^ { \prime } , s ) q ( a \mid S ^ { \prime } , s ) } { q ( a \mid S ^ { \prime } , s ) } .\tag{17}
$$

Splitting the logarithm gives

$$
\begin{array} { r l } & { H ( A \mid S ^ { \prime } , s ) = - \mathbb { E } _ { \mathbf { \phi } _ { a \sim \pi ( \cdot \mid s ) } } \log q ( a \mid S ^ { \prime } , s ) } \\ & { \qquad \quad s ^ { \prime } { \sim } p ( \cdot \mid a , s ) } \\ & { \qquad - \mathbb { E } _ { \mathbf { \phi } _ { a \sim \pi ( \cdot \mid s ) } } \log \frac { p ( a \mid S ^ { \prime } , s ) } { q ( a \mid S ^ { \prime } , s ) } . } \end{array}\tag{18}
$$

The second term is, by definition, the KL divergence. Since $S ^ { \prime }$ is itself random, this divergence must subsequently be averaged over $p ( s ^ { \prime } | s )$

$$
D _ { K L } \big ( p ( a \mid S ^ { \prime } , s ) , \| q ( a \mid S ^ { \prime } , s ) \big ) = \mathbb { E } _ { p ( s ^ { \prime } \mid s ) } D _ { K L } \left( p ( a | s ^ { \prime } , s ) \| q ( a | s ^ { \prime } , s ) \right) .\tag{19}
$$

Substitute in Eq. 18:

$$
H ( A \mid S ^ { \prime } , s ) = - \mathbb { E } _ { \stackrel { a \sim \pi ( \cdot \mid s ) } { s ^ { \prime } \sim p ( \cdot \mid a , s ) } } \log q ( a \mid S ^ { \prime } , s ) - D _ { K L } \big ( p ( a \mid S ^ { \prime } , s ) , \| q ( a \mid S ^ { \prime } , s ) \big ) .\tag{20}
$$

Substituting Eq. (20) into Eq. (10) gives

$$
\begin{array} { r l } & { I ( A ; S ^ { \prime } \mid s ) = \operatorname { \mathbb { E } } _ { a \sim \pi ( \cdot \mid s ) } \log q ( a \mid S ^ { \prime } , s ) - \operatorname { \mathbb { E } } _ { a \sim \pi ( \cdot \mid s ) } \log \pi ( a \mid s ) } \\ & { \qquad \quad s ^ { \prime } { \sim } p ( \cdot \mid a , s ) } \\ & { \qquad + D _ { K L } \big ( p ( a \mid S ^ { \prime } , s ) \mid q ( a \mid S ^ { \prime } , s ) \big ) . } \end{array}\tag{21}
$$

$\mathrm { p } ( \mathrm { a } | \mathrm { s } )$ is defined by our policy �:

$$
\begin{array} { r l } & { I _ { \boldsymbol \pi } ( { \boldsymbol A } ; { \boldsymbol S } ^ { \prime } | s ) = \mathbb { E } _ { \mathbf \Delta _ { a } \sim \boldsymbol \pi ( \cdot | s ) } \left[ \log q ( a | s ^ { \prime } , s ) - \log \pi ( a | s ) \right] } \\ & { \qquad \quad s ^ { \prime } { \sim } p ( \cdot | a , s ) } \\ & { \qquad + \mathbb { E } _ { p ( s ^ { \prime } | s ) } D _ { \mathrm { K L } } \left( p ( a | s ^ { \prime } , s ) \lVert q ( a | s ^ { \prime } , s ) \right) . } \end{array}\tag{22}
$$

This gives us a variational lower bound on on-policy empowerment, and therefore lower bound on empowerment. This construction is similar to ELBO objective in variational autoencoder [16, 17].

$$
\begin{array} { r } { I _ { \pi } ( A ; S ^ { \prime } | s ) \geq \mathbb { E } _ { \mathbf { \phi } _ { a \sim \pi ( \cdot | s ) } } \left[ \log q ( a | s ^ { \prime } , s ) - \log \pi ( a | s ) \right] } \\ { s ^ { \prime } { \sim } p ( \cdot | a , s ) \qquad } \end{array}\tag{23}
$$

We can minimise divergence $D _ { K L }$ term by maximising log-likelihood log $q ( a | s , s ^ { \prime } )$ on the on-policy samples $( a , s , s ^ { \prime } )$ . We need only positive samples for this case, unlike MINE or InfoNCE.

For example if actions are continuous

$$
q _ { \theta } ( a \mid s , s ^ { \prime } ) = \mathcal { N } \big ( a ; \mu _ { \theta } ( s , s ^ { \prime } ) , \sigma _ { \theta } ( s , s ^ { \prime } ) \big )\tag{24}
$$

Then our reward function is:

$$
r _ { t } ^ { \mathrm { e m p } } = \log q _ { \theta } ( a _ { t } | s _ { t } , s _ { t + 1 } ) - \log \pi ( a _ { t } | s _ { t } )\tag{25}
$$

Our new reward function has this relation with empowerment and on-policy empowerment $I _ { \pi }$ :

$$
\mathbb { E } [ r _ { t } ^ { \mathrm { e m p } } | S _ { t } = s ] \le I _ { \pi } ( A ; { S ^ { \prime } } | s ) \le \mathcal { E } ( s )\tag{26}
$$

Left bound becomes tight when $q ( a | s , s ^ { \prime } ) = p ( a | s , s ^ { \prime } )$ , right bound becomes tight when policy achieves maximum capacity for given state.

We can obtain similar results for forward model that predicts future state from the equation

$$
I ( A ; S ^ { \prime } \mid s ) = H ( S ^ { \prime } \mid s ) - H ( S ^ { \prime } \mid A , s )
$$

Though in this case we have to learn distribution of future state given current $p ( S ^ { \prime } | s )$ which is usually a much harder problem than learning the inverse model.

In most environments there are much less options for an action that have caused transition between states $( s , s ^ { \prime } )$ than there are possible future states under $p ( s ^ { \prime } | s )$ For example in our grid world it’s always only one possible action that causes transition between two cells, but there are 4 possible future states for the central cell.

The dificulty comes from simple models such as $q ( s ^ { \prime } , s ) = \mathcal { N } ( s ^ { \prime } ; \mu _ { \theta } ( s ) , I )$ being inherently unimodal, not suitable for modelling a multimodal distribution. It’s possible to make $\mu _ { \theta } ( s )$ output parameter for e.g. mixture of several Gaussians, or use more advanced and dificult techniques.

![](images/8cc16e3f436757b131a7f9fb5a17c71e4931b64fc7370fa28b48c250d8c3ab3a.jpg)  
Figure 6: Maximum-likelihood estimation failure with a single normal distribution.

A unimodal predictor will place mean between modes and give difuse(high variance) estimation for future state S as illustrated in Fig. 6.

## 4.2. Continuous(diferential) entropy

Let’s start with a definition of continuous entropy, it’s usually noted as lower ℎ to distinguish from discrete case:

$$
h ( X ) = - \int p ( x ) \log p ( x ) d x\tag{27}
$$

Diferential entropy of normal distribution:

$$
h ( A ) = { \frac { 1 } { 2 } } \log _ { 2 } ( 2 \pi e \sigma ^ { 2 } ) .\tag{28}
$$

Continuous entropy behaves quite diferently from discrete, in fact for a variable � with Delta-function density we have $h _ { X } = - \infty$

Mutual information on the other hand stay non-negative, but there is another issue. Consider this transition function: $S ^ { \prime } = A + \epsilon$

T = 0 T = 1 T = 2, return a0 xor a1   
observe 0→a0 →observe 0 → a1 → observe 1   
observe 0→a0 →observe 0 → a0 → observe 0   
observe 0→a1 →observe 0 → a0 → observe 1   
observe 0→a1 →observe 0 → a1 → observe 0

With action and noise following normal distribution: $A \sim \mathcal { N } ( 0 , \sigma )$ ， � ∼ $\mathcal { N } ( 0 , N )$ . If action reconstruction becomes possible to arbitrary precision, in such deterministic environments empowerment may become infinite.

For our inverse model we will have:

$$
\begin{array} { r } { h ( A | S ^ { \prime } , s ) = \frac 1 2 \log ( 2 \pi e N ^ { 2 } ) \longrightarrow - \infty \qquad \mathrm { a s ~ } N \to 0 } \end{array}
$$

$$
I ( A ; S ^ { \prime } \mid s ) = h ( A \mid s ) - h ( A \mid S ^ { \prime } , s ) = h ( A \mid s ) - ( - \infty ) = \infty\tag{29}
$$

Also for continuous actions it’s possible to maximise empowerment by increasing action magnitude:

$$
\begin{array} { r } { h ( a | s ) = \frac { 1 } { 2 } \log ( 2 \pi e \sigma ^ { 2 } ) \longrightarrow \infty \qquad \mathrm { a s ~ } \sigma  \infty } \end{array}
$$

Thus, the policy will learn to maximise $h ( A \mid s )$ by increasing the action range.

A natural solution is to constrain the range to $A ~ \in ~ [ a _ { \operatorname* { m i n } } , ~ a _ { \operatorname* { m a x } } ]$ or restrict covariance. To prevent entropy explosion in inverse model $\mathrm { p } ( \mathrm { a } | \mathrm { s } ^ { \prime } , \mathrm { s } )$ we could add noise to observations or actions:

$p ( a \mid s ^ { \prime } + \epsilon , s )$ cannot perfectly recover � anymore keeping entropy and mutual information finite.

For a more detailed treatment of continuous entropy and mutual information, see [18].

## 5. Multistep and single-step empowerment

Consider this simple 2 step environment. Two actions are possible $\scriptstyle \mathrm { a 0 = 0 }$ , and a1=1. The environment always returns 0 at t=0 and t=1, and xor of two actions at t=2:

The first action produces no visible one-step change: $I ( A _ { 0 } ; S _ { 1 } \mid S _ { 0 } ) = 0$

$O _ { 2 }$ is independent from $O _ { 1 }$ and $A _ { 1 }$ and $I ( A _ { 1 } ; S _ { 2 } | s _ { 1 } ) = 0$

On the other hand if agent is recurrent it(and inverse model) remembers it’s actions. In such case we have $I ( A _ { 0 } , A _ { 1 } ; S _ { 2 } \mid S _ { 0 } ) = 1$ bit.

Note that final state doesn’t identify each action individually:

$S _ { 2 } = 0$ implies $( A _ { 0 } , A _ { 1 } ) \in 0 0 , 1 1 $

$S _ { 2 } = 1$ implies $( A _ { 0 } , A _ { 1 } ) \in { 0 1 , 1 0 }$

This example shows that summing local quantities, $\textstyle \sum _ { t } I ( A _ { t } ; S _ { t + 1 } \mid S _ { t } )$ can miss information carried jointly by multiple actions.

$I ( A _ { 0 } A _ { 1 } ; S _ { 2 } \ | \ s _ { 0 } )$ in our example would be two-step empowerment. We can define k-step empowerment as:

$$
\begin{array} { r } { I _ { k } ( s 0 ) \triangleq \operatorname* { m a x } _ { p ( a _ { 0 : k - 1 } \mid s _ { 0 } ) } I ( A _ { 0 : k - 1 } ; S _ { k } \mid s _ { 0 } ) } \end{array}
$$

With intermediate states added it is called trajectory empowerment.

$$
\begin{array} { r } { I _ { k } ( s _ { 0 } ) \stackrel { \Delta } { = } \operatorname* { m a x } _ { p ( a _ { 0 : H - 1 } \mid s _ { 0 } ) } I ( A _ { 0 : k - 1 } ; S _ { 1 : k } \mid s _ { 0 } ) } \end{array}
$$

Knowing intermediate states makes it easier to recover actions so $I ( A _ { 0 : k - 1 } ; S _ { k } \mid$ $S _ { 0 } ) \leq I ( A _ { 0 : k - 1 } ; S _ { 1 : k } \mid S _ { 0 } )$

Let’s expand first equation with inverse model:

$$
\begin{array} { r } { I ( A _ { 0 : k - 1 } ; S k \mid s _ { 0 } ) \ge \mathbb { E } \log q ( A _ { 0 : k - 1 } | S _ { k } , s _ { 0 } ) - \log p _ { \pi } ( A _ { 0 : k - 1 } \mid s _ { 0 } ) } \end{array}\tag{30}
$$

## 5.1. Marginal and Causal objective

We can sample all actions at s0 and then execute them one by one. In this case actions probability term is easy to compute: $\begin{array} { r } { p _ { \pi } ( { a } _ { 0 : H - 1 } \mid s _ { 0 } ) = \prod _ { t = 0 } ^ { H - 1 } \pi ( a _ { t } \mid } \end{array}$ $s _ { 0 } , a _ { 0 : t - 1 } )$

Or we are sampling actions one by one, a new action at each new state. In this case to compute actions probability we need expectation over all possible trajectories between $s _ { 0 }$ and $s _ { T }$

$$
\begin{array} { r } { p _ { \pi } ( a _ { 0 : H - 1 } \mid s _ { 0 } ) = \int p _ { \pi } ( a _ { 0 : H - 1 } , s _ { 1 : H - 1 } \mid s _ { 0 } ) d s _ { 1 : H - 1 } } \end{array}
$$

This is intractable to compute besides most simple cases. If we have a model of the environment(see sec. 9 ) we can compute Monte-Carlo approximation by doing virtual rollouts with fixed initial state and fixed actions. Then approximation is average probability of sequence of actions over all virtual episodes. We can call it marginal MI objective since it marginalises over trajectory.

$$
\begin{array} { r } { { w _ { j } = \displaystyle \prod _ { t = 0 } ^ { H - 1 } \pi ( { { { \bar { a } } } _ { t } } \mid { C _ { t } } ^ { ( j ) } ) } } \\ { { { \hat { p } } _ { \pi } ( { { \bar { a } } } _ { 0 : H - 1 } \mid s _ { 0 } ) = \displaystyle \frac { 1 } { N } \sum _ { j = 1 } ^ { N } w _ { j } } } \end{array}\tag{31}
$$

Here $C _ { t } ^ { ( j ) }$ is all relevant inputs to the policy for given episode $j .$

Another option is to plug in on-policy likelihood:

$$
I ( A _ { 0 : k - 1 } ; S k \mid s _ { 0 } ) \geq \mathbb { E } \log q ( A _ { 0 : H - 1 } ; S _ { H } | s _ { 0 } ) - \sum _ { t = 0 } ^ { H - 1 } \log \pi ( A _ { t } \mid S _ { 0 : t } , A _ { 0 : t - 1 } )\tag{32}
$$

This objective can be named causal since it uses causal action entropy $\textstyle \sum _ { t } H _ { \mathcal { \pi } } ( A _ { t } | S _ { t } , A _ { i < t } )$ term.

## 5.2. cards and notebook environment

Consider k-step environment:

At each time dealer draws a random card $C _ { t }$ and stacks it on the table. The robot performs action $A _ { t } = C _ { t }$ such as:

• wave hand left if the card is red and right if card is black

• write what card it sees to the notebook

After H cards are drawn it’s possible to examine stacked cards and determine which actions the robot took. There are total $2 ^ { k }$ possible actions sequences. So we will get up to k bits of information

$$
\begin{array} { r l } & { r = \log q ( \bar { a } _ { 0 : k - 1 } \mid S _ { k } , s _ { 0 } ) - \log \hat { p } _ { \pi } ( \bar { a } _ { 0 : k - 1 } \mid s _ { 0 } ) } \\ & { \quad \to 0 - \log 2 ^ { - H } } \\ & { \quad = H \log 2 . } \end{array}
$$

On other hand entropy term in causal objective is always zero since each action is determined by the card. So copying environment is not rewarded. Thus marginal objective ≥ causal. It’s important to note that causal objective is not a mutual information, it can be negative. Consider our card environment but this time cards are removed after being shown to the robot, $S _ { 2 } = b l a n k$ . Policy still copies actions so $\pi ( A _ { 1 } = C | S _ { 1 } = C ) = 1$ . Then causal entropy is $H ( A _ { 1 } \mid S _ { 0 } , S _ { 1 } ) = 0$ . Inverse model can’t reconstruct $A _ { 1 }$ from $S _ { 0 } , S _ { 2 }$ , so it assigns equal probability 1/2 and $\log _ { 2 } \pi ( A _ { 1 } \mid S _ { 0 } , S _ { 1 } ) = - 1$ . Thus we have $I _ { c } a u s a l = - 1 - 0 = - 1$

On other hand for an agent with a notebook maximum is achieved with uniform policy $H ( A _ { t } \mid S _ { t } ) = \log K$ and $J = k \log K$ . Thus the policy might learn to write down random symbols to notebook. Similar failure modes exist for all informationbased rewards. This issue is discussed in more details in the next section.

We can also sample so-called latent plan �, or skill at the begging and optimise for this objective

with objective $\begin{array} { r } { \mathcal { E } _ { H } ^ { U } ( s _ { 0 } ) = \operatorname* { m a x } _ { p ( u \mid s _ { 0 } ) } I ( U ; S _ { H } \mid s _ { 0 } ) } \end{array}$

with variational lower bound

$$
I ( U ; S _ { H } \mid s _ { 0 } ) \ge \mathbb { E } \left[ \log q ( U \mid S _ { H } , S _ { 0 } ) - \log p ( U \mid s _ { 0 } ) \right]
$$

This option is discussed in sec. 10 and 15.

## 6. Local Optima

Many people enjoy computer games, games can even hook some humans into addictive patterns. Empowerment is suitable for modelling this efect. The reward is high where there are many visited states, but the environment is controllable. An agent might get obsessed with flipping a switch that has a lot of settings because it’s easy to manipulate and ofers immediate, predictable feedback. Prediction-error driven agents can get stuck observing a noise source. Similarly, empowermentdriven agent can stuck interacting with a controllable objects with many states.

This problem arises for all information-based rewards and it’s likely not solvable on agent’s side.. People like playing video games, but even ones with addiction won’t stick there because of resource constraints. Any real environment has resource constraints and energy and resource sources. We can introduce such constraints into virtual environments.

Lets add to the state energy coordinate $e \in [ 0 , 1 ]$

$$
S = ( s , e )
$$

Every action consumes $\Delta e > 0$ . When $e < e _ { c r i t }$ the motor controller becomes noisy or certain actions are disabled. Formally, $\mathrm { p } ( \mathrm { s } ^ { \prime } | \mathrm { s } , \mathrm { a } )$ grows broader $\Rightarrow \mathrm { H } ( \mathrm { S } ^ { \prime } | \mathrm { a } , \mathrm { s } )$ increases, thus lowering empowerment. There are recharge states $s _ { c h a r g e }$ (pickups, charging pads, food) that reset e upward.

Therefore empowerment maximiser can obtain an intrinsic drive to:

1. stay away from low-energy states

2. periodically reach recharge regions

3. optimise its action sequence for long-term discounted reward

I.e. the agent behaves like a living organism: forage → play/act → return to base → repeat.

It is easy to introduce to an environment with direct biological analogy, like predator-prey, but harder for other cases. On the other hand in a multi-agent world it might not be necessary. Other agents can provide a natural brake against ”camping on a toy” behaviour.

• Disturbance: other agents policy injects variability into transition probability $p ( s _ { t + 1 } | a _ { t } , s _ { t } )$ leading to increase of entropy of future state $H ( S ^ { \prime } \mid A , s )$ conditional on action.

• resource competition: agent must keep discovering new high-control niches or defend the old one.

• Non-stationary dynamics: camping behaviour might become impossible as time goes due to the non-stationary environment itself.

When the multi-agent cure might fail:

• Predictable opponents

If other agents adopt highly regular or submissive policies $H ( S ^ { \prime } \mid A , s )$ can still be low → empowerment can stay high.

• Collusion / territorial partition

Agents may implicitly agree on “you keep toy #1, I keep toy $\# 2 ^ { \bullet }$ . Each finds a private controllable niche. This has analogies in biology.

Other agents can destabilise simple controllable niches, but they can also create new intrinsically rewarding attractors. Their efect therefore depends on both their behaviour, environment constraints and the intrinsic objective.

## 7. Communication and collective intrinsic motivation

Communication extend an agent’s action channel beyond its own actuators. Other agents may disrupt simple controllable niches, but communication also creates a new controllable subsystem. Empowerment can therefore be defined not only through direct physical actions, but also through causal influence mediated by other agents. As mentioned in seq. 2 communication is ubiquitous in multicellular organisms so it’s important to consider how it afects agents with intrinsic motivation.

In the simplest case there are two agents � and � that can send messages to each other from some alphabet M. They take turns: � sends a message and waits for a response from �. Then � sends its message and waits for response. High empowerment reward is achievable for both agents in such a situation. Agent � selects message m randomly and sends it to �. If � responds predictably, e.g. $f ( m ) = ( m { + } 1 )$ ) mod $\lvert \mathcal { M } \rvert$ then we have high accuracy for $P ( S ^ { \prime } \mid m , s _ { 0 } )$ but also high $H ( S ^ { \prime } \mid s _ { 0 } )$ (since response can’t be predicted without knowing the message). Since next state now is completely predictable we get empowerment equal to $H ( A \mid s )$ In this case both agents will achieve maximum reward with uniform distribution over M.

## Achievable empowerment is log|A|

Thus agents may obtain intrinsic reward by exchanging arbitrary messages without producing useful collective behaviour.

Prediction error could reward unexpected messages.

Learning progress - learning of changes in other agent’s policy.

Information gain - messages exposing other agents internal state

diversity objectives - role and protocol specialisation.

With a shared channel in which simultaneous messages interfere, agents may develop turn-taking schedules. But with energy constraints agents might refuse sending a response, lowering other agent’s empowerment.

We can think of diferent levels and objectives in such systems:

• self, or per-agent empowerment - increasing each own control

• transfer empowerment - increasing control of other agents

• assistance empowerment - increasing own influence on other agents [19]

• joint empowerment - group of communicating agents constitute an organism, or meta-agent and can optimise their joint objective

Joint empowerment is not automatically optimised when every agent independently optimises its own empowerment. From biology we can guess that local self-empowerment may lead to competition, domination, no-communication and possibly even to organism level cooperation. Communication predates complex multicellular organisms, but the emergence of integrated multicellular organisms required communication to be combined with adhesion, functional diferentiation and mechanisms that limit conflict between cells. The long evolutionary delay, possibly billions of years, between early life and complex multicellular organisation suggests that local communication and adaptive behaviour are not by themselves suficient to produce a stable higher-level organisation. Synchronisation of organism state across many cells using signals between adjacent cells may face scaling limits due to delays and error accumulation.

There are examples of relatively broad low-capacity channels in biology. Quorum sensing(QS) is a widespread mechanism of cell-to-cell communication and coordination using signalling molecules. It is based on release of signal molecules to the outside of cells. The phenomenon has not only been described between cells of the same species (intraspecies), but also between species (interspecies) and between bacteria and higher organisms (inter-kingdom) [20]. Hormones are later development of the same mechanism.

There’s also communication based on electricity. Both cell-to-cell using ion pumps embedded in cell membranes [21]. And using relatively broad, macroscopic electric fields. Important example is electric gradient that guides body formation in embryogenesis. Optical signalling may represent another possibility, although its functional role in cell-to-cell communication remains less established [22].

We can conclude that global communication channel might enable more complex behaviour development in agents optimising self-empowerment. Analogously to chemical communication we can introduce global message $g _ { t }$ that aggregates individual messages:

$$
\begin{array} { r } { g _ { t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } m _ { i , t } } \end{array}
$$

with aggregated message broadcasted to all agents:

$$
o _ { i , t + 1 } = \left( o _ { i , t + 1 } ^ { \mathrm { l o c a l } } , g _ { t } \right)
$$

## 8. MINE and InfoNCE

## 8.1. Mutual information neural estimation is defined as this objective:

Mutual Information Neural Estimation (MINE) [23] uses the Donsker–Varadhan representation to estimate mutual information with a neural network.

$$
I ( X ; Y ) = D _ { \mathrm { K L } } \left( P _ { X Y } \parallel P _ { X } \otimes P _ { Y } \right) = \operatorname* { s u p } _ { T : \Omega \to \mathbb { R } } \left[ \mathbb { E } _ { ( X , Y ) \sim P _ { X Y } } \left[ T ( X , Y ) \right] - \log \mathbb { E } _ { X \sim P _ { X } } \left[ e ^ { T ( X , Y ) } \right] \right] .\tag{33}
$$

Here first expectation is over joint distribution and second is over marginal distributions of X and Y.

T is a function returning real number $T ( X , Y ) \to R$ . We approximate T by neural network $T _ { \theta } ( X , Y )$ and use gradient ascend to find supremum.

At training time we just need Monte-Carlo samples for both terms.

$$
\begin{array} { r } { \hat { I } _ { \mathrm { M I N E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } T _ { \theta } ( x _ { i } , y _ { i } ) - \log \left\lceil \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \exp T _ { \theta } ( \tilde { x } _ { i } , \tilde { y } _ { i } ) \right\rceil } \end{array}
$$

,̃� ̃� - samples from marginal distributions, to obtain them we can draw separate batches for x and y or just shufle one of them e.g. y from the same batch that is used in $T _ { \theta } ( x , y )$

It can be proofed that supremum achieved when

$$
T ^ { * } ( x , y ) = \log \frac { p _ { X Y } ( x , y ) } { p _ { X } ( x ) p _ { Y } ( y ) } + C\tag{34}
$$

log ${ \frac { P _ { X , Y } } { Q _ { X , Y } } } \ = \ \log P - l o g \ Q \ = \ l o g \ p ( X , Y ) - \log p ( X ) P ( Y )$ is just pointwise mutual information.

For empowerment estimation we have two random variables A and S, but this is conditioned on current state s, so function T will have 3 inputs.

Plugging A and S in T gives

$$
T ( A ; S ^ { \prime } | s ) = l o g P ( A ; S ^ { \prime } | s ) - l o g ( P ( A | s ) P ( S ^ { \prime } | s ) + c
$$

$$
T ( A ; S ^ { \prime } | s ) = l o g ( p ( S ^ { \prime } | A , s ) p ( A | s ) ) - l o g ( P ( A | s ) P ( S ^ { \prime } | s ) + c
$$

$$
{ \cal T } ( A ; S ^ { \prime } | s ) ~ = ~ l o g ~ p ( S ^ { \prime } | A , s ) ~ + ~ l o g ~ p ( A | s ) ~ - ~ l o g ~ p ( A | s ) ~ - ~ l o g ~ p ( S ^ { \prime } | s ) ~ + ~ c ~ 
$$

$$
T ( A ; S ^ { \prime } | s ) \ = \ l o g p ( S ^ { \prime } | A , s ) \ - \ l o g p ( S ^ { \prime } | s ) \ + \ c
$$

$$
I ( A ; S ^ { \prime } \mid s ) = H ( S ^ { \prime } \mid s ) - H ( S ^ { \prime } \mid A , s ) \ = - E \left[ l o g P ( S ^ { \prime } \mid s ) \right] - \left( - E \left[ l o g p ( S ^ { \prime } \mid A , s ) \right] \right) .
$$

$$
{ \cal I } _ { A , S ^ { \prime } ~ { \sim } ~ \pi } ( A ; S ^ { \prime } ~ \vert ~ s ) ~ = ~ E [ ~ l o g ~ p ( S ^ { \prime } \vert A , ~ s ) ~ - ~ l o g ~ P ( S ^ { \prime } ~ \vert ~ s ) ~ ]
$$

Thus trained T-function directly gives a point estimate of mutual information up to an additive constant. However it wont give empowerment estimation when trained on shufled states, actions pairs. Empowerment requires outcomes from distribution conditioned on current state $p ( S ^ { \prime } | s = s _ { t } )$ . Shufling batch will give us $I ( S ^ { \prime } ; ( S , A ) )$ because shufling will approximate $S ^ { \prime } { \sim } p ( S ^ { \prime } )$ - unconditional distribution instead of $P ( S ^ { \prime } \mid s )$ . Compare these equations with explicitly written expectations:

$$
\begin{array} { r } { I ( ( S , A ) ; S ^ { \prime } ) = E _ { S \sim p ( s ) } E _ { A \sim \pi ( \cdot \vert S ) } E _ { S ^ { \prime } \sim p ( \cdot \vert S , A ) } l o g \frac { p ( S ^ { \prime } \vert S , A ) } { p ( S ^ { \prime } ) } } \end{array}
$$

$$
\begin{array} { r } { I ( S ^ { \prime } ; A | s ) = E _ { A \sim \pi ( \cdot | S ) } E _ { S ^ { \prime } \sim p ( \cdot | S , A ) } l o g \frac { p ( S ^ { \prime } | S = s , A ) } { p ( S ^ { \prime } | S = s ) } } \end{array}
$$

$$
\operatorname { W i t h } I ( ( S , A ) ; S ^ { \prime } ) = I ( S ; S ^ { \prime } ) + I ( A ; \bar { S ^ { \prime } } \mid S )
$$

$$
\operatorname { C o m p a r e } I ( S ; S ^ { \prime } ) { \mathrm { ~ t o ~ } } I ( A ; S ^ { \prime } \mid s ) \colon
$$

$$
{ \cal I } ( S ; S ^ { \prime } ) = { \cal H } ( S ^ { \prime } ) - { \cal H } ( S ^ { \prime } | S )
$$

$$
I ( A ; S ^ { \prime } \mid s ) = H ( S ^ { \prime } \mid s ) - H ( S ^ { \prime } \mid A , s )
$$

So we have conflicting term $H ( S ^ { \prime } | S )$ that cancels out.

$$
I ( ( S , A ) ; S ^ { \prime } ) = H ( S ^ { \prime } ) - H ( S ^ { \prime } \mid A , s )
$$

Policy-level failure mode. Using this term as a reward is problematic. $H ( S ^ { \prime } )$ is fixed for all states in a minibatch, so our reward function does not distinguish if concrete action leads to better state space exploration. With all episodes being equal in this term the only option to optimise remains inverse model. This might lead to policy visiting just a few states that are easy to predict. This is an example when an intrinsic reward may correlate positively with exploration under the current policy, yet optimising that reward may fail to shift the policy-induced state-visitation distribution towards broader coverage.

## 8.2. InfoNCE

Another method for mutual information estimation is Information Noise-Contrastive Estimation.

$$
L _ { \mathrm { I n f o N C E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp f _ { \theta } ( x _ { i } , y _ { i } ) } { \sum _ { j = 1 } ^ { N } \exp f _ { \theta } ( x _ { i } , y _ { j } ) } .\tag{35}
$$

Here $( x _ { i } , y _ { i } ) -$ positive pair, $y _ { j } , j \ \ne i , - \mathrm { n e g a t i v e s } .$ , sampled from marginal distribution. Function $f _ { \theta }$ is analogous to $T _ { \theta }$ in MINE. It should return high value for samples coming from joint distribution and low values for marginal distribution. It is identical to softmax operator applied to positive and negative samples and can be viewed as a standard multi-class classification problem. This estimator is often more stable than MINE. It defines lower bound on mutual information: $I ( X ; Y ) \ge \log N - L _ { \mathrm { I n f o N C E } }$

## 9. WORLD MODELS

World models are models that infer world state from observations and predict how this state evolves given actions. This allows to train policy in virtual episodes.

Here we describe prominent models Dreamer v3 and TWISTER and their components.

Contrastive Predictive Coding [24]

DreamerV3 [25]

TWISTER [26]

Key components of DREAMER/TWISTER

$\boldsymbol { h _ { t } } \ = \boldsymbol { F } \boldsymbol { \phi } ( \boldsymbol { h _ { t - 1 } } , \boldsymbol { z _ { t - 1 } } , \boldsymbol { a _ { t - 1 } } )$ - context encoder e.g. RNN or transformer.

$\hat { z } _ { t } ~ \sim ~ d _ { \phi } ( \hat { z } _ { t } \mid h _ { t } ) \cdot \mathrm { d y }$ namics predictor: models distribution of world state given context. This predicts how world changes given previous state and action $a _ { t - 1 }$ which is also encoded in $h _ { t }$

$\begin{array} { r c l } { { z _ { t } } } & { { \sim } } & { { q _ { \phi } ( z _ { t } \mid o _ { t } , \ldots ) ~ . } } \end{array}$ - observation encoder, models distribution of $z _ { t }$ given current observation $o _ { t } \in . \mathsf { g }$ . VAE encoder. In dreamer v3 its $z _ { t } ~ \sim ~ q _ { \phi } ( * \mid o _ { t } , h _ { t } )$ $s _ { t } ~ = ~ \{ h _ { t } , ~ z _ { t } \}$ - concatenation of context and world state.

$\hat { o } _ { t } ~ \sim ~ p _ { \phi } ( \hat { o } _ { t } \mid z _ { t } )$ - observation decoder, models distribution for observations e.g. VAE decoder, in dreamer it’s $\hat { o } _ { t } ~ \sim ~ p _ { \phi } ( \hat { o } _ { t } \mid s _ { t } )$

$\hat { e } _ { t } = u ( z _ { t } )$ - model that projects world states to embedding space.

$\hat { e } _ { t : t + k } ~ = ~ w _ { \phi } ~ ( s _ { t } , ~ a _ { t : t + k } )$ - predicted embeddings given state and actions

$\boldsymbol { \hat { r } } _ { t } ~ \sim ~ r _ { \phi } ( \boldsymbol { \hat { r } } _ { t } \mid \boldsymbol { s } _ { t } )$ - predicted reward for current transition

$\hat { c } _ { t } ~ \sim ~ c _ { \phi } ( \hat { c } _ { t } \mid s _ { t } )$ - predicted episode continuation/termination

![](images/a9402d413ff310f1ac712b7bc60003ebdcc5b65605f692a7b9cf5523bbd60f4d.jpg)  
Figure 7: Key components of DREAMER/TWISTER

losses:

$\hat { z } _ { t } \sim d _ { \phi }$ is learnt with KL divergence with observation encoder $q _ { \phi } ( z _ { t } \mid o _ { t } , \ldots )$ This is very similar to recursive Bayesian estimation or Bayesian filtering, very similar to the inference in the hidden-markov model.

$q _ { \phi }$ and� are trained with whatever encoder-decoder loss is suitable e.g. ELBO $p _ { \phi }$ loss.

$w _ { \phi }$ is trained with constructive loss with projection $u ( z _ { t } )$ of observed $z _ { t }$ to predicted $\hat { e } _ { t : t + k }$

It is also possible to learn other useful signals such as rewards and episode termination.

## 9.1. Action Conditioned (AC-CPC)

This is TWISTER component that is responsible for training $\boldsymbol { \hat { e } } _ { t } = \boldsymbol { u } ( \boldsymbol { z } _ { t } )$

This is an architecture that allows compressing high-dimensional observations into useful representations.

RSSM stands for Recurrent State-Space Model. This is a part of dreamer that constitutes the world model. In TWISTER it is TSSM - transformer instead of recurrent model.

There are two key diferences:

1) Dreamer updates distribution $\hat { z } _ { t } ~ \sim ~ d _ { \phi } ( * | h _ { t } )$ to match diferent encoder: $z _ { t } ~ \sim ~ q _ { \phi } ( * \mid o _ { t } , h _ { t } )$ . Unlike TWISTER which uses $z _ { t } ~ \sim ~ q _ { \phi } ( * \mid o _ { t } )$

2) AC-CPC/TWISTER contrastive loss for embeddings additionally to reconstructive loss in Dreamer.

Another minor diference decoder in dreamer is $\hat { o } _ { t } ~ \sim ~ p _ { \phi } ( \hat { o } _ { t } \mid s _ { t } )$

Having said that it helps to concatenate many observations $o _ { t }$ or use dreamer style encoder $q _ { \phi } ( z _ { t } \mid o _ { t } , h _ { t } )$ for AC-CPC.

Both methods are reported to work best with discrete distribution of world states $z _ { t }$ . though can be used with diferent distributions.

$o _ { 0 } , a _ { 0 } , o _ { 1 } , a _ { 1 } , o _ { 2 } , a _ { 2 } , o _ { 3 } , a _ { 3 } . . . \cdot \mathrm { s t a t e s }$ , action sequence

$$
h _ { 0 } = 0 , \mathrm { p o s t e r i o r } z _ { 0 } = q _ { \phi } ( h _ { 0 } , o _ { 0 } )
$$

$$
s _ { 0 } = [ h _ { 0 } , z _ { 0 } ] s _ { 0 } - > a _ { 0 }
$$

$$
h _ { 1 } = F _ { \phi } ( z _ { 0 } , h _ { 0 } , a _ { 0 } )
$$

$$
\mathrm { p r i o r } \widehat { z _ { 1 } } = d _ { \phi } ( h _ { 1 } )
$$

posterior $z _ { 1 } = q _ { \phi } ( h _ { 1 } , O _ { 1 } )$

Conceptually there information flow:

$$
z _ { 0 } = q ( h _ { 0 } , o _ { 0 } )  h _ { 1 } = F ( h _ { 0 } , z _ { 0 } , a _ { 0 } )  a _ { 1 }
$$

So $a _ { 1 }$ is generated from $o _ { 0 } , a _ { 0 } , o _ { 1 }$

$s _ { 0 }$ maps to $a _ { 0 }$ ,

$s _ { 1 }$ maps to $a _ { 1 }$ .

and for ac-cpc starting from h1:

$$
e _ { 1 0 } = f ( h _ { 1 } ) \mathrm { ~ - c o n d i t i o n e d ~ b y ~ } a _ { 0 } \mathrm { ~ v i a ~ } h _ { 1 }
$$

$$
e _ { 1 2 } = f ( h _ { 1 } , \widehat { z _ { 1 } } , a _ { 1 } )
$$

$$
e _ { 1 3 } = f ( h _ { 1 } , \widehat { z _ { 1 } } , a _ { 1 } , a _ { 2 } )
$$

$$
\mathrm { w i t h \ p o s i t i v e s \ e l { 0 } <  \ z l , e l 2 <  \ z 2 , e l 3 <  \ z 3 . . . }
$$

TWISTER applies InfoNCE objective introduced in sec. 8.2 to action-conditioned predictions of future latent representations:

$$
L _ { I n f o N C E } \ = \ - \frac 1 N \sum _ { i = 0 } ^ { N } l o g \ \frac { e ^ { s ^ { k } _ { i i } } } { e ^ { s ^ { k } _ { i i } } + \sum _ { j \neq i } e ^ { s ^ { k } _ { i j } } }\tag{36}
$$

Here s being similarity measure between embedding and latent variable z(z is being projected to the embedding space with a multi-layer perception). N is a batch size.

Mutual information between matching embeddings e and latent states z then:

$$
{ \cal I } ( e ; ~ z ) ~ > = ~ l o g ~ N ~ - ~ L _ { I n f o N C E }
$$

In twister this is used only for descriptor training, but what if we use this MI estimation for reward? Is it similar to empowerment? If we assume that e and z are good enough representations of it’s inputs we could write:

$$
I ( e ; z ) \approx I ( ( h _ { t } , a _ { t : t + k } ) , h _ { t + k } ) \approx I ( ( s _ { t } , A ) , S ^ { \prime } )\tag{37}
$$

By chain rule for MI(see appendix)

$$
I ( ( s , A ) ; S ^ { \prime } ) = I ( s ; S ^ { \prime } ) + I ( A ; S ^ { \prime } \mid s )\tag{38}
$$

As you can see this quantity can reward behaviour when future trajectory is predictable even without knowing action.

Expand:

$$
I ( s ; s ^ { \prime } ) = H ( S ^ { \prime } ) \ - \ H ( S ^ { \prime } | s )
$$

$$
I ( A ; S ^ { ' } \mid s ) ~ = ~ H ( S ^ { ' } \mid s ) - H ( S ^ { ' } \mid A , s ) ~ { \mathrm { - m u l t i s t e p ~ e m p o w e r m e n t } }
$$

$$
H ( S ^ { \prime } ) \ - \ H ( S ^ { \prime } | s ) + H ( S ^ { \prime } \mid s ) - H ( S ^ { \prime } \mid A , s ) = H ( S ^ { \prime } ) - H ( S ^ { \prime } \mid A , s )
$$

$$
I ( ( s , A ) ; S ^ { \prime } ) = H ( S ^ { \prime } ) - H ( S ^ { \prime } \mid A , s )
$$

This is diference $H ( S ^ { \prime } )$ vs $H ( S ^ { \prime } \mid s )$ tells us what we are trying to distinguish/predict from s and A. AC-CPC uses for negatives all possible states S in a batch. While $H ( S ^ { \prime } \mid s ) ( \mathbf { a s }$ in empowerment) would require us to construct negatives only from states we could arrive from the current state �!

Despite equation 38 being sound using global pool of states(or their embeddings) has issues. One issue mentioned in sec. 8.1 is that $H ( S ^ { \prime } )$ will be the same for all episodes in the minibatch. Another issue is first term $I ( S ; S ^ { \setminus } )$ doesn’t involve actions. Suppose the positive pair from a RTS-game environment is: current frame: your base, five workers, daytime;

future frame: nearly the same scene after a few seconds.

A global negative might include completely diferent map locations, enemy type, time elapsed from start, diferent army and resource count. It’s easy for discriminator to distinguish even without using actions. At this point $L _ { n c e }$ is close to zero and learning stops. Empowerment formulation on other hand requires ”hard negatives”. That is states produced from the same initial condition, but under alternative actions. The good thing is that world models allow us to generate realistic virtual episodes!

## 10. Diversity is all you need

The goal of Diversity Is All You Need (DIAYN) [27] is to make an agent learn diferent skills in an unsupervised manner.

In this method we pass to the policy additional parameter z that is sampled from a random distribution(categorical or normal). $Z \sim { \mathfrak { p } } ( z )$ . Policy conditioned on z is called skill. Training objective has tree parts:

1) Mutual information between states and skills ${ \mathrm { I } } ( \mathbf { S } { ; } \mathbf { Z } )$ is maximised

2) MI between actions and skills given the state is minimised $\scriptstyle \operatorname { I } ( \mathrm { A } ; Z \mid S )$

3) Entropy of actions given state is maximised $\mathrm { H } [ \mathbf { A } \mid \mathbf { S } ]$

$$
\begin{array} { l } { \operatorname* { m a x } F ( \theta ) \triangleq I ( S ; Z ) + H [ A | S ] - I ( A ; Z | S ) } \\ { \qquad = ( H [ Z ] - H [ Z | S ] ) + H [ A | S ] - ( H [ A | S ] - H [ A | S , Z ] ) } \\ { \qquad = H [ Z ] - H [ Z | S ] + H [ A | S , Z ] \ } \end{array}\tag{39}
$$

�[�] is entropy of skill distribution - constant.

Conditional entropy of skill distribution can be rewritten using chain rule for conditional entropy as:

$$
H [ Z | S ] = H [ S , Z ] - H [ S ] = H [ S | Z ] - H [ S ] + H [ Z ]
$$

This will help us to understand when exactly the reward will be high and when low.

Substituting it into original equation:

�(�) = �[�] − (�[�, �] − �[�]) + �[�|�, �]   
= �[�] − �[�, �] + �[�] + �[�|�, �]   
Expand joint entropy of skills and states:   
�[�, �] = �[�|�] + �(�) = �(�|�) + �(�)   
�(�) = �[�] − (�(�|�) + �(�)) + �(�) + �[�|�, �]   
= �[�] − �(�|�) − �(�) + �(�) + �[�|�, �]   
= �(�) + �[�|�, �] − �[�|�]   
This form helps to understand what behaviour is encouraged by this reward.   
First term H(S) maximises entropy of states = encourages policy to visit many   
states.

�[� | �, �] - conditional entropy of action given state and skill - encourages policy to apply diferent actions given skill and state. Equivalently this means it should be hard to guess skill from action alone.

Also maximising this term makes it so knowing state + action doesn’t give more information about skill than knowing just state alone.

H(S|Z) is minimised. If it is low we have a small number of states which are visited by given skill.

Using definition I(Z;A∣S)=H(A∣S)−H(A∣S,Z)   
H(A∣S,Z) = H(A∣S) - I(Z;A∣S)   
So, �(�) = �(�) + �(� ∣ �) − �(�|�) − �(�;� ∣ �) =   
�(�) + �(� ∣ �) − �(� | �) − �(�;� ∣ �)   
Max H(S) = maximise the number of visited states.   
Max H(A∣S) = maximise the number of actions taken in any particular state.   
Min H(S|Z) = minimise the number of states visited by a particular skill.   
Min I(Z;A∣S) = all actions are more or less the same for all skills.   
Min H(S|Z) = skill can be used to predict S

$$
\begin{array} { r l } & { \mathrm { T h i s ~ o b j e c t i v e ~ c a n ~ b e ~ i m p l e m e n t e d ~ a s : } } \\ & { \qquad H ( Z | S ) = - \sum p ( s , \ z ) l o g \frac { p ( s , z ) } { p ( s ) } = { E } [ - l o g p ( z | s ) ] } \\ & { \qquad H ( Z ) = - \sum p ( z ) \log p ( z ) = \mathrm { E } [ - \log p ( z ) ] } \\ & { \qquad H ( A | \ S , Z ) = - \sum p ( s , z ) \sum p ( a | s , z ) l o g p ( a | s , z ) \ = { E } _ { s , z } [ - \sum p ( a | s , z ) l o g p ( a | s , z ) \ ] } \\ &  = - { E } _ { s , z , a } [ l o g p ( a | s , z ) \} \\ & { \qquad F ( \theta ) \ = \ E [ - l o g p ( z ) ] - E [ - l o g p ( z | s ) ] + E [ - l o g p ( a | s , z ) \ ] = { E } [ \ l o g p ( z | s ) \ - } \\ & { l o g p ( z ) \ - l o g p ( a | s , z ) \qquad ] } \end{array}
$$

Probability of skill given state ${ \sf p } ( { \bf z } | { \bf s } )$ is approximated by a neural network predictor $\phi _ { \theta } ( s ) - > z$

log $p ( a | \ s , z )$ is just policy entropy log $\pi \theta ( \mathbf { a } | \ \mathbf { s } , \mathbf { Z } )$ . It can be included in the reward or treated separately, like entropy reguliser as in soft actor-critic.

$$
\operatorname { r } ( \mathrm { s } , \mathrm { a } | \mathrm { z } ) = \log \phi ( \mathrm { z } | \mathrm { s } ) - \log \mathrm { p } ( \mathrm { z } )
$$

In principle it’s possible to use forward density p(s|z), but practically choice p(z|s) is much easier to implement, since we don’t know the ground truth distribution of S given skill.

Estimating $\phi ( \mathbf { z } | \mathbf { s } )$ is just a classification/regression problem; the target distribution over z is simple (uniform or Gaussian).

Estimating ${ \sf p } ( { \bf s } | z )$ is often a high-dimensional density problem (very hard for image observations).

Side by Side comparison with Empowerment
<table><tr><td></td><td>Empowerment</td><td>DIAYN</td></tr><tr><td>pushed up</td><td> $p ( s ^ { \cdot } | a , s )$ </td><td> $p ( z | s )$ </td></tr><tr><td>variance pushed up</td><td> $p ( s ^ { \prime } \mid s )$ </td><td> $p ( a | s , z )$ </td></tr><tr><td>Encouraged property</td><td>Predictable consequences</td><td>States discriminate skills;</td></tr><tr><td>Discouraged</td><td>per action Small marginal next-state</td><td>actions stay diverse Peaky action distribution</td></tr><tr><td>property</td><td>variability</td><td>inside a skill</td></tr></table>

In equation 1 variable Z is sampled ones per episode, S is picked uniformly from the whole trajectory, so the term $H ( Z \mid S )$ is computed with respect to the entire trajectory. But there are situations when examining multiple states might be desirable. We might be interested in achieving the same state by diferent means, for example robot can place a spoon in a mug either by taking a spoon and placing it or by trying to scoop the spoon. In this case intermediate states are clearly distinguish skill vector, but the end state is the same. By default DIAYN will try to avoid visiting the same state from two skills. We could counter it by either by adding external reward, curiosity, or adding structure to the skill vector as discussed below:

## Structural DIAYN

For example let vector z be concatenation of vectors $z _ { i }$

If we choose to use just two vectors we will have:

• two latents z(1) and z(2) are concatenated $z = [ z ^ { ( 1 ) } , z ^ { ( 2 ) } ]$

• the policy $\pi ( a \mid s , z ^ { ( 1 ) } , \ z ^ { ( 2 ) } )$ can use both parts all the time;

• the discriminator at an early time slice tries to predict $z ^ { ( 1 ) }$ only, ignoring $z ^ { ( 2 ) }$ ;

• the discriminator at the final state (or any late slice) predicts $z ^ { ( 2 ) }$ only.

Any two skills that difer only in $z ^ { ( 1 ) }$ therefore must converge to (almost) the same final state, reached via recognisably distinct trajectories.

We can also try to reconstruct all $\boldsymbol { z } ^ { ( i \mathrm { ~ < ~ } t ) }$ . In this case policy will be encouraged to treat early $z ^ { i }$ as a high-level coarse plan or “style” and later $z ^ { i }$ further nuancing the behaviour. Other options such as using a sliding window are possible.

## 11. Curiosity and Learning progress

Curiosity reward can be a combination of a few things: surprise, for example in the form of prediction error, novelty and learning progress. The theoretical foundations of curiosity and learning progress were developed by Jürgen Schmidhuber; see [28], [29].

## 11.1. Prediction error and Novelty

Prediction error can be estimated from forward model: $f ( s , a ) \mathrm { ~ - ~ } > \hat { s } _ { t + 1 }$

Reward then is $| | s _ { t + 1 } \rrangle - f ( s , a ) | |$

Novelty is inversely related to state visitation count: it is high for new states and low for frequently visited states. It encourages large entropy H(S) of the state visitation distribution.

Implementation note. In practice, novelty is often computed from embeddings of observed states. As discussed in the Policy-level failure mode paragraph and Section 9, some novelty-based rewards may fail to distinguish episodes that produce broader exploration.

A second failure mode can occur when the score lacks a persistent scale across policy updates. If novelty is normalised using statistics of the current batch, small diferences between increasingly similar episodes are rescaled. The normalised reward can therefore continue to rank episodes within each batch even as the absolute diversity and coverage decreases.

## 11.2. Learning progress

Learning progress is a bit more complex. We have forward model $f _ { \theta } ( s _ { t + 1 } | s , a )$ and distribution over its weights $p ( \theta | O _ { t } )$ given the history of transitions; $\begin{array} { r l } { O _ { t } } & { { } = } \end{array}$ $\{ ( s _ { \tau } , a _ { \tau } , s _ { \tau } ^ { \prime } ) \}$ for $\tau < t$

Learning progress is modeled as a change of distribution over parameters of forward model $f _ { \theta } ( s _ { t + 1 } | s , a )$ given a new observation.

$$
L P _ { t } \triangleq { \mathit { K L } } [ p ( \theta \mid O _ { t } \cup O _ { t + 1 } ) \parallel p ( \theta \mid O _ { t } ) ]
$$

This can be approximated simply with improvement in prediction error.

$$
\mathrm { T h a t ~ i s } \ : r = \vert \vert f _ { \theta _ { k } } ( s , a ) - s _ { t + 1 } \vert \vert - \vert \vert f _ { \theta _ { k + 1 } } ( s , a ) - s _ { t + 1 } \vert \vert
$$

Novelty and prediction error, unlike learning progress are prone to noise-staring behaviour. That is, the reward is high when the agent observes pure noise. But there are easy workarounds. We can train a model that predicts action given previous and current states. $g _ { \theta } ( s _ { t + 1 } , s _ { t } ) \mathrel { - } > a$ . Higher layers of this model can be used as feature extractors, that keep information only about controllable aspects of the environment. These features can be used then instead of raw states in novelty or surprise rewards.

Let’s compare curiosity with empowerment.

Empowerment $\mathrm { I ( A { ; } S { ' } | s ) { = } H ( S { ' } | s ) { - } H ( S { ' } | s , a ) }$

Prediction error $\mathrm { H } ( \mathbf { S } ^ { \prime } | \mathbf { s } , \mathbf { a } )$

State novelty H(S)

Given identity $H ( S ^ { \prime } ) = I ( S ^ { \prime } ; S ) + H ( S ^ { \prime } \mid S )$ we have $\mathrm { H } ( \mathbf { S } ^ { \prime } | \mathbf { s } )$ is less or equal to $H ( S ^ { \prime } )$ .

That is possible to have large $H ( S ^ { \prime } )$ and but small $H ( S ^ { \prime } \mid S )$ . Reverse is not true:

High $H ( S ^ { \prime } \mid S )$ implies that $H ( S ^ { \prime } )$ at least that large.

So maximising $H ( S ^ { \prime } )$ is not identical to maximizing $H ( S ^ { \prime } \mid S ) . \ H ( S ^ { \prime } \mid S )$ encourages rewards states from which many diferent states can occur immediately(or in k-steps with k-steps empowerment). But $H ( S ^ { \prime } )$ rewards visiting the whole state space.

## 12. Information gain

Assume our world model estimates world state with z and tracks history in h. Then we can define (point) information gain about world state as

$$
K L ( q ( z \mid o , h ) \parallel p ( z \mid h ) ) = I _ { p m i } ( o ; z \mid h )
$$

That is how much have we learned about world given a new observation o.

Note: term ”information gain” is applicable to diferent things, including model parameters. In this case we have change in distribution over model parameters which is directly related to learning progress.

This quantity could be used as addition to prediction error. We have this relation:

$$
H ( O \mid h ) = E _ { z \sim p ( z \mid h ) } [ H ( O \mid z , h ) ] + I ( O ; Z \mid h )
$$

The first quantity is unreducible or so called aleatoric uncertainty. This is uncertainty that remains even if we have good state estimation. The second term is the uncertainty caused by not knowing z. If agent encounters a source of noise prediction entropy(and error) will be high, but information gain small hinting at large aleatoric uncertanty. On other case consider agent exploring unknown part of the map, in this case prediction error will be high, but information gain also large. We could use this to reward useful exploration much more than just watching random events.

Similar to other information-based rewards IG is prone to ”camping on a toy” issue. For example with fair dice we have $I ( O ; Z | H ) = H ( Z | H ) - H ( Z | O , H ) =$ $l n ( 6 ) - 0 = l n ( 6 )$ nats for each throw.

## 13. SFA

Invented by Laurenz Wiskott and Terrence Sejnowski [30], Slow Feature Analysis is an unsupervised learning rule that extracts features whose values change as slowly as possible over time, although they are computed from an input stream that may itself vary quickly.

Suppose that an encoder neural network transforms each observation $o _ { t }$ into a feature vector $y _ { t }$

$$
y _ { t } = g ( o _ { t } )
$$

Slowness is achieved with loss

$$
L _ { \mathrm { s l o w n e s s } } = \frac { 1 } { T - 1 } \sum _ { t = 2 } ^ { T } \| y _ { t } - y _ { t - 1 } \| _ { 2 } ^ { 2 } .\tag{40}
$$

In order to avoid degenerate solutions such as encoding each observation as zeros we require decorrelation, zero mean and unit variance for $\mathrm { y . }$

Variance loss is defined as:

$\begin{array} { r } { C _ { Y } = \frac { 1 } { N } Y ^ { T } Y } \end{array}$ where Y is a concatenation of zero-centred vectors y.

$$
L _ { \mathrm { v a r i a n c e } } = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \left( C _ { Y } [ j , j ] - 1 \right) ^ { 2 } .\tag{41}
$$

The correlation loss is the normalised squared Frobenius norm of the of-diagonal part of $C _ { Y }$ :

$$
L _ { \mathrm { c o r r e l a t i o n } } = \frac { 1 } { d ( d - 1 ) } \sum _ { i \neq j } C _ { Y } [ i , j ] ^ { 2 } .\tag{42}
$$

Then objective is

$$
{ \cal L } _ { s f a } \ = \ { \cal L } _ { s l o w n e s s } \ + \ { \cal L } _ { \nu a r i a n c e } \ + \ { \cal L } _ { c o r r e l a t i o n }
$$

This is potentially useful augmentation for action conditioned state embeddings. AC-CPC in TWISTER must store relatively fast details needed for next observation prediction. We could extract slow-changing features from action-conditioned embeddings. Action conditioning is important: it may help distinguish controllable features from features that do not depend on the agents actions. Applying SFA on top of AC-CPC might therefore extract slow, controllable features suitable for computing of intrinsic reward over longer time scales.

## 14. Predictive information bonus and MDL

Imagine an agent that can see part of an image, can move on it and can change pixels. Empowerment alone won’t produce interesting non-random looking images, it does not express a preference for regularity, coherence, or semantic content. It encourages diverse, but predictive states, so the agent might change many pixels randomly producing images visually similar to noise. We can use MI between patches of the image $I ( t o p - l e f t , t o p - r i g h t ) = H ( t o p - l e f t ) - H ( t o p -$ $l e f t | t o p - r i g h t )$ as reward in this case.

This distinguishes structured images from two trivial solutions:

Blank images: both patches are predictable, but there is no variation so H(top-left) $= 0$

Independent random noise: the patches vary, but one does not predict the other. So $H ( t o p - l e f t | t o p - r i g h t ) \approx H ( t o p - l e f t )$ again giving 0 mutual information. Structured and variable images: patches vary across images but share regularities, giving positive mutual information.

The objective can be extended to many random partitions and multiple spatial scales:

$$
R _ { \mathrm { s t r u c t u r e } } ( x ) = \mathbb { E } _ { ( U , V ) \sim \mathcal { M } } \left[ I ( X _ { U } ; X _ { V } ) \right] ,
$$

For discrete pixels or image tokens, one can train:

\- a marginal model $p _ { \phi } ( X _ { V } )$ , and - a conditional model $q _ { \psi } ( X _ { V } \mid X _ { U } )$

A sample-level structure reward is then

$$
r _ { \mathrm { s t r u c t u r e } } ( x ) = \log q _ { \psi } ( x _ { V } \mid x _ { U } ) - \log p _ { \phi } ( x _ { V } ) .
$$

The first term rewards predictability from context, while the second prevents the predictor from receiving high reward merely because the patch is constant everywhere.

log $p _ { \phi } ( x _ { V } )$ term corresponds to minimum description length(see the section below).

Predictive information can still collapse to a small family of highly regular images—for example, the same checkerboard in every episode. We therefore need to distinguish within-image structure from across-image diversity.

A diversity objective can reward entropy in a global image representation $H ( f _ { \theta } ( X ) )$ where f should preferably be a frozen or slowly changing perceptual encoder. $\mathbf { A n - }$ other option is to sample a latent intention $Z$ at the beginning of an episode and maximize $I ( C ; X _ { T } )$ . It is closely related to unsupervised skill discovery: each latent code corresponds to a diferent controllable mode of image generation.

The two objectives are complementary: predictive information discourages noise generation, global diversity discourages generation of single or a few patterns.

Therefore we could combine these terms into one reward:

$$
R = \alpha R _ { e } m p o w e r m e n t + \beta R _ { s } t r u c t u r e + \gamma R _ { d } i \nu e r s i t y - \lambda R _ { c } o s t
$$

New term $R _ { c } o s t$ here represents action or complexity cost. This term is needed to avoid useless image modification e.g. changing one pixel black -> white -> black.

This objective does not formally guarantee aesthetically or semantically interesting images. For example, repeated textures, barcodes, or hidden high-frequency signals may score highly despite looking uninteresting to humans.

The term used for a processes that could achieve more and more diverse and complex artifacts is ”open-endedness”. Our reward combination does not guarantee an open-ended process. Fixed intrinsic objectives can still be exhausted or exploited. Once the agent discovers a finite family of highly controllable, structured images, it may cycle among them indefinitely without producing genuinely new organization. Open-endedness additionally requires a continually expanding space of challenges, or niches — for example through co-evolving competing agents, procedurally generated environments, or learned objectives that change as previous behaviours become common. Possible formalisation is given in [31].

## 14.1. VAE and MDL

VAE could be used as an approximator for description length. According to Shannon’s source coding theorem optimal code length(for a prefix code) that one can assign to a datapoint � is its negative log-likelihood $- l o g p ( x )$

VAE models empirical distribution $p _ { t r u e } ( x )$ as $\begin{array} { r } { p _ { \theta } ( x ) = \int _ { z } p _ { \theta } ( x | z ) p ( z ) d z , } \end{array}$ , approximating the intractable posterior $p _ { \theta } ( z | x )$ with the variational posterior $q ( \boldsymbol { z } | \boldsymbol { x } )$ For detailed derivation see [16, 17].

ELBO objective

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { E L B O } } ( x ) = \log p _ { \theta } ( x ) - \mathbb { D } _ { K L } ( q _ { \phi } ( z | x ) | | p _ { \theta } ( z | x ) ) } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } = \mathbb { E } _ { q _ { \phi } ( z | x ) } [ \log p _ { \theta } ( x | z ) ] - \mathbb { D } _ { K L } ( q _ { \phi } ( z | x ) \parallel p ( z ) ) } \end{array}\tag{43}
$$

Thus we have

$$
\log p _ { \theta } ( x ) \geq \mathcal { L } _ { \mathrm { E L B O } } ( x )
$$

for unknown true distribution this inequality holds in expectation:

$$
E _ { x \sim p _ { t r u e } } \log p _ { \theta } ( x ) \geq E _ { x \sim p _ { t r u e } } \mathcal { L } _ { \mathrm { E L B O } } ( x )
$$

## 15. Problems with Empowerment and DIAYN

Single-step empowerment is short-sighted. It can even be zero for obviously easy environments as shown in the xor environment example. Multistep empowerment is better, but it can’t be applied without modifications to high-frequency long-term environments like RTS games. On the other hand DIAYN objective is global in a sense that we can sample individual states from a trajectory, and it will provide valid lower bound estimation $I ( Z ; s i n g l e s t a t e ) \le I ( Z ;$ �ℎ��� ����������). But empowerment can be zero given one-step or two step estimation.

DIAYN’s objective is limited in a sense that it is just for learning skills. Humans or animals use skills to achieve certain goals, skills are combined and used in certain order. DIAYN objective says nothing about how skills should be combined. In more realistic architecture an agent would pursue long-term goals switching between skills as the situation evolves. It’s possible to aggregate multiple steps to enable DIAYN to develop diferent temporal patters, for example achieving the same goal with diferent locomotion gaits. However it’s become very easy for policy to develop pathological solution for skill discrimination e.g. just waiving manipulators in diferent pattern that are easy to classify. Another example of valid, but uninteresting optimum is in Figure 8. Assume out agent can move in 2d circle, starting from the centre and observe it’s position. It’s possible for policy to slice circle along radius and thus recover ”skill” vector z while staying near the start. Other rewards such as novelty, e.g. euclidean distance from all states in minibatch could drive agent to explore much larger state-space. Our example shows that combination of intrinsic rewards can help counter each-others failure modes in some cases.

Issues with empowerment can be addressed by introducing diferent time scales and state compression. In such case empowerment would be used to reward highlevel actions that take many environmental steps to execute.

DIAYN skills can be such low-level actions. And skill selection can be trained by n-step empowerment.

If we choose to use empowerment for only high-level action selection we have to decide which states to use and which to omit. Consider RTS game with 640x480 video frames as observations. Suppose we have high-level actions pursue, build expand, build unit, attack, flee etc. We can’t use just pixel observations at skill boundary. Both policy and empowerment estimator would require a feature vector that encodes strategic situations in the game + more detailed description of the current situation, that is exact unit position, recent events such as “hero just died”.

Multi-layer SFA is a possible front-end to obtain the macro strategic state on which we measure empowerment or DIAYN MI, but exact architecture remains an open research problem.

![](images/bff098096a8405d05cae21b8ea7abb486971514e00f64db89ee363f14208aa71.jpg)

![](images/82a5c89e0774b9f9bfcaea3ffa30192147974206468612fd11194bb1c15a0290.jpg)  
Figure 8: State-space partition and traversal. Left: DIAYN partitions the state space into skillconditioned regions but does not guarantee that an arbitrary goal $g _ { 0 }$ can be reached; increasing the number of skills may only make these regions thinner. Right: a sequence of reachable goalconditioned regions $g _ { 0 } , g _ { 1 }$ and $g _ { 2 }$ can support traversal through the state space.

## 16. Research agenda

We conclude with discussion of possible experiments intended to test if existing intrinsic motivation methods can produce biological adaptability on diferent levels.

The proposed direction is related to Schmidhuber’s formal theory of creativity and intrinsic motivation, in which agents actively generate experiments and receive intrinsic reward for discovering novel but learnable regularities that improve prediction or data compression [29].

## 16.1. Environmentfor intrinsic reward evaluation

Based on previous discussion of MI-based rewards and their ”camping on a toy” limitations we propose an environment with natural constraints on camping(or other similarly useless) behaviour. There are two properties of biological environments that counter failure modes of information-based rewards:

1. World is adversarial.

2. Energy is scarce.

We design an environment where agents have an energy level. Analogously to physiological deficits, a low energy level reduces action amplitude and/or makes actions less predictable.

Thus energy level afects the capacity of action to future state channel.

We propose a predator–prey environment with the following observed state: $o _ { t } = ( s e n s o r _ { t } , e _ { t } )$ where $e _ { t }$ is the energy level.

Agents have a finite storage capacity $e _ { m a x }$ and each action consumes energy:

$$
e _ { t + 1 } = c l i p ( e _ { t } - c ( a _ { t } ) + f o o d _ { t } , 0 , e _ { m a x } )
$$

Energy level afects actions:

$$
\begin{array} { r } { a _ { e f f e c t i \nu e } = m ( e _ { t } ) a _ { t } + \sigma ( e _ { t } ) \epsilon _ { t } } \end{array}
$$

Here � is a magnitude function, might be as simple as $m ( e _ { t } ) = \operatorname* { m a x } ( e _ { t } , m _ { \operatorname* { m i n } } )$ The function $\sigma$ determines the magnitude of motor noise and increases as $e _ { t }$ decreases. for example we can define

$$
\begin{array} { r l } & { \sigma ( e _ { t } ) = \operatorname* { m i n } ( \sigma _ { m a x } , - \ln ( e _ { t } / e _ { m a x } ) ) } \\ & { \epsilon _ { t } \sim \mathcal { N } ( 0 , I ) } \end{array}
$$

Finding food or capturing prey increases the energy level without providing direct reward signal.

Note: as discussed earlier continuous actions requires non-zero observation or action uncertainty to keep mutual information finite.

This environment would allow to test directly whether diferent combinations of intrinsic rewards lead to behaviour that maintains a stable energy level, analogous to energy homeostasis in biological organisms.

Possible tests include environments with:

1. controllable toy;

2. ”noisy-tv”;

3. communication channel between predators;

4. adversarial prey;

Important ablations include disabling the efects of energy on actions and comparing stationary food with adversarial prey. This comparison is necessary because an intrinsic reward may cause the agent to follow prey for reasons unrelated to energy regulation.

Primary metrics to monitor:

1. mean energy level

2. time spent on toys

3. spatial coverage

4. energy level at which the agent starts sustained movement towards prey

5. energy recovery time after reaching the minimum energy level

Interesting extension is test conditions under which communication might develop e.g. when it’s hard for an agent to catch the prey on it’s own.

## 16.2. Recurrent model

We propose to study a network of recurrent modules where every module is treated as a local agent. Each agent has its own hidden state, observations, incoming messages, actions and intrinsic reward. There is no reward defined for the network as a whole. The modules can communicate during the forward pass, but messages are detached before being passed between agents. Consequently, gradients from one agent cannot propagate through the internal computations of another agent.

For a system of � agents communicating with messages � the shared recurrent update can be written as

$$
h _ { i , t + 1 } = F _ { \theta } ( h _ { i , t } , o _ { i , t } , m _ { i , t } ) , \qquad i \in \{ 1 , \ldots , N \} ,\tag{44}
$$

where all agents use the same parameters $\theta ,$ , while $h _ { i , t } , o _ { i , t }$ and the position of an agent in the communication graph are diferent. The shared parameters are analogous to a common genome, while diferent hidden states, inputs and network positions provide diferent local contexts. Functional specialisation may therefore emerge without assigning a permanent identity or a separate set of parameters to every module.

Every agent computes an intrinsic reward $r _ { i , i }$ <sub>�</sub> only from its own interaction history.

The primary experiment is to test whether the recurrent modules develop stable and complementary roles and whether their joint dynamics exhibit adaptive behaviour that is not explicitly rewarded at the system level.

The basic ablation removes weight sharing. In this condition each module has an independent recurrent function

$$
h _ { i , t + 1 } = F _ { \theta _ { i } } ( h _ { i , t } , o _ { i , t } , m _ { i , t } ) .\tag{45}
$$

Independent parameters may make specialisation easier, because diferent roles can be stored directly in $\theta _ { i } .$ . Shared weights provide a stronger test: diferent roles must emerge from local state, experience and network context. As discussed in section 7, a global communication channel might facilitate the formation of collective behaviour. The role of communication can be tested through the following ablations:

1. no communication;

2. local communication;

3. global low-capacity broadcast;

4. local + global communication;

This type of agent may be evaluated in a collectively embodied task e.g., sensor + motor policies jointly controlling motion in a maze. In such an environment communication is mandatory. The architecture can also be tested with independently embodied agents. A good example is a predator-prey environment where each recurrent module controls one predator. This type of environment is especially suitable for a communication ablations because the agents can remain independently function when message passing is disabled.

This experiment is not intended to prescribe a final architecture. Its purpose is to test whether local intrinsic objectives, recurrent memory, communication and a shared learning rule are suficient ingredients for the emergence of functional diferentiation and higher-level adaptive organisation.

## 16.3. Automatic curriculum in a two-level world model agent

The limitations discussed in Section 15 suggest an experiment in which motor control and long-term goal selection are learned by separate agents potentially operating at diferent time scales. The proposed architecture consists of a low-level agent, a high-level agent and a world model. The world model supplies learned state representations and intrinsic signals for training both agents.

Training starts with the low-level agent acting without a valid goal. Its reward is a mixture of learning progress, prediction error and diversity in the learned embedding space. This stage has two purposes: to collect diverse experience for the world model and to train a low-level policy capable of producing non-trivial transitions before it is asked to follow goals. For initial goal conditioning training obvious choices are hindsight training and virtual hindsight training on wm-generated episodes once wm is stable.

After this initial stage, the high-level agent generates a target embedding $g _ { t }$ The low-level policy receives both the current representation $s _ { t }$ and the target $g _ { t }$ and is rewarded for reducing their distance. A simple progress reward is

$$
r _ { t } ^ { \mathrm { l o w } } = d ( z _ { t } , g _ { t } ) - d ( z _ { t + 1 } , g _ { t } ) .\tag{46}
$$

The high-level action is held for several environment steps(determined by separate switch head), so selecting one target embedding corresponds to a temporally extended action. The two agents can therefore learn diferent functions: the lowlevel agent learns how to realise changes in the representation space, while the high-level agent learns which changes are informative, reachable and useful for continued exploration.

The target space should discard high-frequency details that cannot be controlled over the selected horizon. Slowly varying representations, including representations obtained with SFA-like objectives, are a possible goal space. Long-term empowerment may then be estimated between high-level actions and future states in this compressed representation rather than between individual motor commands and raw observations.

Overall process should achieve state-space traversal similar to one in Figure 8.

This training process resembles developmental motor learning: initially unstructured self-generated actions establish sensorimotor regularities, after which achieved outcomes can become goals for increasingly directed behaviour. Motor competence and goal selection may then form a coupled curriculum in which each expands the learning opportunities of the other.

This experiment tests whether temporal hierarchy addresses two complementary limitations. Skill-discovery objectives such as DIAYN do not specify how independently learned skills should be ordered, while short-horizon empowerment does not represent consequences separated from motor actions by many environment steps. A two-level agent instead treats goal-conditioned behaviour as the low-level action space of a slower decision process.

Code availability. Code for the proposed experiments will be published as the implementations are developed at https://github.com/noskill/reinf.

## 17. Mathematical Appendix

## 17.0.1. Chain rule for Mutual information

$$
I ( X , Y ; Z ) = I ( X ; Z ) + I ( Y ; Z \mid X )\tag{47}
$$

By definition we have

$$
I ( X , Y ; Z ) = H ( Z ) - H ( Z \mid X , Y )
$$

Expand �(�; �) and $I ( Y ; Z \mid X )$ to entropies:

$$
I ( X ; Z ) = H ( Z ) - H ( Z \mid X )
$$

$$
I ( Y ; Z \mid X ) = H ( Z \mid X ) - H ( Z \mid X , Y )
$$

Compute sum:

$$
I ( X ; Z ) + I ( Y ; Z \mid X ) = H ( Z ) - H ( Z \mid X ) + H ( Z \mid X ) - H ( Z \mid X , Y )
$$

$$
I ( X ; Z ) + I ( Y ; Z \mid X ) = H ( Z ) - H ( Z \mid X , Y ) = I ( X , Y ; Z )
$$

## 17.0.2. Donsker-Varadhan MI

We are starting from KL divergence $\begin{array} { r } { D _ { K L } ( P | | Q ) = \int p ( u ) l o g \frac { p ( u ) } { q ( u ) } d u } \end{array}$ Step 1. Define Gibbs density.

Let ${ \sf q } ( { \sf u } )$ be a probability density and T(u) be our arbitrary function. We create a new valid probability density, $g ( u )$ , by weighting q(u) with $e ^ { T ( u ) }$ . To ensure g(u) integrates to 1, we must divide by a normalizing constant $Z { : }$

$$
g ( u ) \ = \frac { e ^ { T ( u ) } q ( u ) } { Z }
$$

Where the constant Z is just the expected value over the distribution Q:

$$
\begin{array} { r } { Z = \int e ^ { T ( u ) } q ( u ) d u = E _ { O } e ^ { T ( u ) } } \end{array}
$$

Step 2: Use the Non-Negativity of KL Divergence

$$
\begin{array} { r } { D _ { K L } ( P | | G ) = \int p ( u ) l o g \frac { p ( u ) } { g ( u ) } d x \geq 0 } \end{array}
$$

Step 3: Multiply by $\frac { q ( u ) } { q ( u ) }$

$$
\begin{array} { r } { l o g \frac { p ( u ) } { g ( u ) } = l o g \frac { p ( u ) } { q ( u ) } \frac { q ( u ) } { g ( u ) } = l o g \frac { p ( u ) } { q ( u ) } + l o g \frac { q ( u ) } { g ( u ) } } \end{array}
$$

$$
g ( u ) ~ = ~ \frac { e ^ { T ( u ) } q ( u ) } { Z } ~ = > ~ \frac { g ( u ) } { q ( u ) } ~ = ~ \frac { e ^ { T ( u ) } } Z
$$

$$
\begin{array} { r } { \frac { q ( u ) } { g ( u ) } \ = \frac { Z } { e ^ { T ( u ) } } } \end{array}
$$

$$
\begin{array} { r } { l o g \ \frac { q ( u ) } { g ( u ) } \ = l o g \ \frac { Z } { e ^ { T ( u ) } } \ = \ l o g Z \ - \ T ( u ) } \end{array}
$$

Step 4: Substitute and Solve

$$
\begin{array} { r } { \int p ( u ) ~ l o g ~ \frac { p ( u ) } { q ( u ) } \frac { q ( u ) } { g ( u ) } d u = \int p ( u ) \big ( \log \frac { p ( u ) } { q ( u ) } + \log Z - T ( u ) \big ) d u \geq 0 } \end{array}
$$

$$
\begin{array} { r l } { \mathrm { F i r s t ~ t e r m } \int p ( u ) \log { \frac { p ( u ) } { q ( u ) } } d u \mathrm { ~ i s ~ } D _ { K L } ( P \parallel Q ) } & { { } } \end{array}
$$

Second term $\begin{array} { r } { \int p ( u ) \log Z d u = \log Z \int p ( u ) d u = \log Z * 1 = \log Z } \end{array}$

$$
\begin{array} { r } { \mathrm { T h i r d ~ t e r m } \int p ( u ) T ( u ) ~ d u ~ \mathrm { i s } { E _ { P } } T ( u ) } \end{array}
$$

$$
\begin{array} { r } { D _ { K L } ( P \parallel Q ) + l o g E _ { O } e ^ { T ( u ) } - E _ { P } T ( u ) \ \geq 0 } \end{array}
$$

$$
D _ { K L } ( P \parallel Q ) \ge E _ { P } T ( u ) - l o g E _ { Q } e ^ { T ( u ) }
$$

$$
{ \mathrm { B y ~ d e f i n i t i o n : ~ } } I ( X ; Y ) = D _ { K L } { \big ( } P ( X , Y ) \parallel P ( X ) P ( Y ) { \big ) }
$$

We get mutual information(from definition) with P having density p(x, y) - joint density; and Q having density p(x)p(y) - the product of the marginal densities:

$$
I ( X ; ~ Y ) ~ \geq ~ E _ { P ( X , Y ) } T ( x , y ) ~ - ~ l o g ~ E _ { P ( X ) P ( Y ) } ~ e ^ { T ( x , y ) }
$$

We got lower bound for KL and mutual information.

We can optimise this estimator by finding better function � e.g. with gradient ascent.

## References

[1] M. Levin, Self-improvising memory: A perspective on memories as agential, dynamically reinterpreting cognitive glue, Entropy 26 (2024) 481. URL: https://doi.org/10.3390/e26060481. doi:10.3390/e26060481.

[2] S. Biswas, W. Clawson, M. Levin, Learning in transcriptional network models: Computational discovery of pathway-level memory and effective interventions, International Journal of Molecular Sciences 24 (2023) 285. URL: https://doi.org/10.3390/ijms24010285. doi:10. 3390/ijms24010285.

[3] R. Chis-Ciure, M. Levin, Cognition all the way down 2.0: Neuroscience beyond neurons in the diverse intelligence era, Synthese 206 (2025) 257. URL: https://doi.org/10.1007/s11229-025-05319-6. doi:10.1007/ s11229-025-05319-6.

[4] M. Levin, Michael levin: Intelligence beyond the brain, YouTube video, Principles of Intelligence, 2022. URL: https://youtu.be/RwEKg5cjkKQ, published September 11, 2022; accessed September 3, 2026.

[5] M. Levin, Bioelectricity: A bridge between physics and cognition, by way of biology, YouTube video, Michael Levin’s Academic Content, 2025. URL: https://www.youtube.com/watch?v=GiL6wtg3U0I, published September 23, 2025; accessed September 15, 2026.

[6] E. O. Wilson, The Insect Societies, Belknap Press of Harvard University Press, Cambridge, Massachusetts, 1971.

[7] T. Vida, Z. T. Calamari, P. Barden, Post K–Pg rise in ant and termite prevalence underlies convergent dietary specialization in mammals, Evolution 79 (2025) 2315–2324. URL: https://pubmed.ncbi.nlm.nih.gov/ 40455576/. doi:10.1093/evolut/qpaf121.

[8] L. Bell-Roberts, The Evolution of Division of Labour in Social Insects, Ph.D. thesis, University of Oxford, 2024. URL: https://ora.ox.ac.uk/ objects/uuid:781db1e5-929a-485e-a1bc-5bdf0be99d3a.

[9] S. Kriegman, D. Blackiston, M. Levin, J. Bongard, Xenobot—a tall quadruped, Wikimedia Commons, 2020. URL: https://commons. wikimedia.org/wiki/File:Xenobot\_-\_A\_tall\_quadruped.jpg, licensed under Creative Commons Attribution 4.0 International (CC BY 4.0); accessed September 15, 2026.

[10] H. S. Galpayage Dona, C. Solvi, A. Kowalewska, K. Mäkelä, H. MaBouDi, L. Chittka, Do bumble bees play?, Animal Behaviour 194 (2022) 239–251. URL: https://doi.org/10.1016/j.anbehav.2022.08.013. doi:10.1016/j.anbehav.2022.08.013, creative Commons Attribution 4.0 International License.

[11] O. J. Loukola, C. J. Perry, L. Coscos, L. Chittka, Bumblebees show cognitive flexibility by improving on an observed complex behavior, Science 355 (2017) 833–836. URL: https://doi.org/10.1126/science.aag2360. doi:10.1126/science.aag2360.

[12] P. K. Y. Chow, T. K. Lehtonen, V. Näreaho, O. J. Loukola, Prior associations afect bumblebees’ generalization performance in a tool-selection task,

iScience 25 (2022) 105466. URL: https://doi.org/10.1016/j.isci. 2022.105466. doi:10.1016/j.isci.2022.105466, creative Commons Attribution 4.0 International License.

[13] A. Zhou, Y. Du, J. Chen, Ants adjust their tool use strategy in response to foraging risk, Functional Ecology 34 (2020) 2524–2535. URL: https:// doi.org/10.1111/1365-2435.13671. doi:10.1111/1365-2435.13671.

[14] J. Peters, Policy gradient methods, Scholarpedia 5 (2010) 3698. URL: http: //www.scholarpedia.org/article/Policy\_gradient\_methods. doi:10.4249/scholarpedia.3698.

[15] A. S. Klyubin, D. Polani, C. L. Nehaniv, Empowerment: A universal agentcentric measure of control, in: 2005 IEEE Congress on Evolutionary Computation, volume 1, IEEE, 2005, pp. 128–135. URL: https://doi.org/ 10.1109/CEC.2005.1554676. doi:10.1109/CEC.2005.1554676.

[16] D. P. Kingma, M. Welling, Auto-encoding variational bayes, in: International Conference on Learning Representations, 2014. URL: https: //arxiv.org/abs/1312.6114. doi:10.48550/arXiv.1312.6114.

[17] C. Doersch, Tutorial on variational autoencoders, arXiv preprint arXiv:1606.05908 (2016). URL: https://arxiv.org/abs/1606.05908. doi:10.48550/arXiv.1606.05908.

[18] J. V. Michalowicz, J. M. Nichols, F. Bucholtz, Handbook of Diferential Entropy, Chapman and Hall/CRC, Boca Raton, Florida, 2013. doi:10.1201/ b15991.

[19] Y. Du, S. Tiomkin, E. Kiciman, D. Polani, P. Abbeel, A. Dragan, AvE: Assistance via empowerment, 2020. URL: https://arxiv.org/abs/2006. 14796. doi:10.48550/arXiv.2006.14796. arXiv:2006.14796.

[20] S. P. Diggle, A. Gardner, S. A. West, A. S. Grifin, Evolutionary theory of bacterial quorum sensing: When is a signal not a signal?, Philosophical Transactions of the Royal Society B: Biological Sciences 362 (2007) 1241–1249. URL: https://pmc.ncbi.nlm.nih.gov/articles/ PMC2435587/. doi:10.1098/rstb.2007.2049.

[21] A. Prindle, J. Liu, M. Asally, S. Ly, J. Garcia-Ojalvo, G. M. Süel, Ion channels enable electrical communication in bacterial communities,

Nature 527 (2015) 59–63. URL: https://www.nature.com/articles/ nature15709. doi:10.1038/nature15709.

[22] J. Bódis, J. Berke, B. Nagy, I. Gulyas, P. Hersics, Á. Várnagy, K. Kovács, Ultra-weak photon emission: From oxidative metabolism to DNA-based communication: A review of biochemical, biophysical and quantum biological perspectives, Frontiers in Endocrinology 17 (2026) 1861061. URL: https://www.frontiersin.org/journals/ endocrinology/articles/10.3389/fendo.2026.1861061/full. doi:10.3389/fendo.2026.1861061.

[23] M. I. Belghazi, A. Baratin, S. Rajeshwar, S. Ozair, Y. Bengio, A. Courville, R. D. Hjelm, Mutual information neural estimation, in: Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, PMLR, 2018, pp. 531–540. URL: https://proceedings.mlr.press/v80/belghazi18a.html.

[24] A. van den Oord, Y. Li, O. Vinyals, Representation learning with contrastive predictive coding, arXiv preprint arXiv:1807.03748 (2018). URL: https: //arxiv.org/abs/1807.03748. doi:10.48550/arXiv.1807.03748.

[25] D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, Mastering diverse domains through world models, arXiv preprint arXiv:2301.04104 (2023). URL: https://arxiv.org/abs/2301.04104. doi:10.48550/ arXiv.2301.04104.

[26] M. Burchi, R. Timofte, Learning transformer-based world models with contrastive predictive coding, arXiv preprint arXiv:2503.04416 (2025). URL: https://arxiv.org/abs/2503.04416. doi:10.48550/ arXiv.2503.04416.

[27] B. Eysenbach, A. Gupta, J. Ibarz, S. Levine, Diversity is all you need: Learning skills without a reward function, arXiv preprint arXiv:1802.06070 (2018). URL: https://arxiv.org/abs/1802.06070. doi:10.48550/arXiv.1802.06070.

[28] J. Schmidhuber, Formal theory of creativity & fun & intrinsic motivation (1990–2010), IDSIA webpage, 2010. URL: https://people.idsia.ch/ \~juergen/creativity.html, accessed September 15, 2026.

[29] J. Schmidhuber, Formal theory of creativity, fun, and intrinsic motivation (1990–2010), IEEE Transactions on Autonomous Mental Development 2 (2010) 230–247. URL: https://doi.org/10.1109/TAMD.2010. 2056368. doi:10.1109/TAMD.2010.2056368.

[30] L. Wiskott, T. J. Sejnowski, Slow feature analysis: Unsupervised learning of invariances, Neural Computation 14 (2002) 715–770. doi:10.1162/ 089976602317318938.

[31] A. Adams, H. Zenil, P. C. W. Davies, S. I. Walker, Formal definitions of unbounded evolution and innovation reveal universal mechanisms for open-ended evolution in dynamical systems, Scientific Reports 7 (2017) 997. URL: https://www.nature.com/articles/ s41598-017-00810-8. doi:10.1038/s41598-017-00810-8.