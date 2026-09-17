# Multitask Reinforcement Learning for Assisting Choice Model Specification

Gabriel Nova\*<sup>1</sup>, Stephane Hess<sup>1,2</sup>, and Sander van Cranenburgh<sup>1</sup>

<sup>1</sup>CityAI Lab, Transport and Logistics Group, Delft University of Technology, The Netherlands <sup>2</sup>Institute for Transport Studies and Choice Modelling Centre, University of Leeds, UK

Discrete choice model specification is a time-consuming task in which modellers often specify and estimate multiple models while balancing goodness-of-fit, parsimony, and behavioural plausibility. We present Delphos, a multitask reinforcement learning framework that learns transferable specification strategies across transport choice datasets. Delphos frames model specification as a sequential decision-making problem in which it applies a sequence of modelling actions and receives feedback from an estimation environment based on model performance and convergence. To transfer modelling decisions across datasets with diferent sets of variables, Delphos represents utility specifications as sets of modelling terms using a DeepSet-Q architecture, allowing a shared specification pol icy to learn across multiple datasets. Trained on nine transport choice datasets, Delphos consistently outperforms independently trained single-task agents, indicating that sharing modelling experience improves learning eficiency and helps identify promising sequences of modelling decisions with fewer unsuccessful estimation attempts. When applied without further training to the unseen Swissmetro and Decisions datasets, the same agent identifies competitive specifications in less than 20 minutes on a standard CPU. It achieves a higher log-likelihood per observation than the VNS metaheuristic on Swissmetro and performance comparable to a published MNL specification developed by expert modellers on Decisions. These findings show that accumulating and reusing modelling experience enables Delphos to function as an intelligent assistant for discrete choice model specification. It reduces manual trial-and-error while allowing modellers to retain control over model diagnosis, refinement, and final selection.

Keywords: Assisted choice model specification; Deep reinforcement learning; Artificial intelligence

## 1 Introduction

Discrete choice models (DCMs) are econometric frameworks widely used to understand and analyse individual choice behaviour, forecast demand, and evaluate policies across a wide range of application domains Buckell et al. (2019); Mariel et al. (2021); Hess & Daly (2024); de Vries et al. (2025). By representing the decision-making process underlying observed choices, DCMs provide insights into individuals’ preferences and the trade-ofs they make between alternatives. These models can then be used to derive behavioural measures such as marginal utilities, elasticities, willingness-to-pay, and values of travel time. To obtain such behavioural insights, modellers typically specify competing choice models by making various modelling decisions, such as selecting explanatory variables, defining functional forms, and representing preference heterogeneity that reflect diferent behavioural assumptions (Van Cranenburgh et al., 2022). In practice, they estimate and evaluate specifications by balancing goodness-of-fit, parsimony, and behavioural plausibility, and then refine them through successive iterations (Nova et al., 2025). Consequently, specifying discrete choice models remains an iterative, cognitively demanding, and time-consuming process.

This trial-and-error workflow has motivated the development of assisted specification methods, which seek to support, speed up, or partially automate the specification process. Two broad classes of approaches can be distinguished. The first comprises metaheuristic methods that formulate model specification as a combinatorial search problem and use optimisation techniques to explore the space of possible models (Páez & Boisjoly, 2022; Rodrigues et al., 2020; Ortelli et al., 2021). These approaches iteratively construct candidate specifications from previously explored models and evaluate them using objective functions based on model fit and parsimony. Promising specifications are retained and modified through predefined search operators to generate new candidates. More recent approaches have incorporated behavioural constraints and grammar-based representations to improve the behavioural plausibility of the resulting specifications (Beeramoole et al., 2023; Haj-Yahia et al., n.d.; Ghorbani et al., 2025). Although these methods can eficiently explore model spaces, they rely on predefined search operators and do not explicitly accumulate knowledge about modelling decisions across specification problems.

The second class comprises machine-learning-based approaches that generate model specifications by learning from previous modelling exercises. Instead of relying on predefined search operators, these methods learn patterns from previously estimated specifications and use them to propose candidates that are likely to satisfy behavioural expectations and provide a good fit. For instance, Sfeir et al. (2025) explore the use of pre-trained large language models to generate utility specifications by drawing on knowledge and reasoning capabilities acquired from large text corpora. Similarly, Nova et al. (2025) formulate model specification as a sequential decision-making problem in which a reinforcement learning agent learns specification strategies through repeated interaction with an estimation environment. These approaches shift the focus from predefined search operators to learning-based systems that can use accumulated knowledge to assist the specification process.

However, existing assisted model specification methods struggle to reuse modelling experience across datasets. Human modellers, by contrast, rarely approach each specification problem independently. They draw on experience from previous modelling exercises and adapt it to new contexts. This is possible because many modelling decisions are informed by behavioural principles that generalise across datasets and application domains. For example, travel cost is expected to have a negative marginal utility, income interactions are commonly used to capture heterogeneity in cost sensitivity, and total, in-vehicle, and out-of-vehicle travel time can be treated as alternative representations of the same behavioural construct. Despite the transferability of this knowledge, metaheuristic methods typically discard information about previously explored specifications once the search terminates, ofering little scope for cumulative learning. Machine-learning-based approaches retain knowledge from previous modelling tasks, but usually encode it in dataset-specific representations. Consequently, the specification process must efectively be repeated for each new dataset, even when relevant modelling knowledge could be transferred from related contexts.

To leverage modelling experience across datasets, we propose a multitask reinforcement learning framework for assisting choice model specification. By extending Delphos (Nova et al. (2025)) to a multitask setting, we enable a single agent to learn a transferable specification policy from multiple choice datasets. To support transfer across datasets with diferent attributes and socio-demographic variables, Delphos represents modelling decisions independently of the dataset-specific variables and contexts in which they are applied. Specifically, we extend Delphos with a DeepSet-Q network (Hügle et al., 2020) that encodes utility specifications as sets of modelling terms and learns a shared latent representation. Conditioned on the current modelling context, this representation enables the agent to select decisions that generalise across related datasets. Through training across datasets, the agent learns which modelling decisions tend to produce high-performing specifications in similar contexts and reuses this knowledge when specifying models for new datasets.

The main aim of the framework is to support modellers by using experience acquired across datasets to provide automated, data-driven suggestions for utility specifications. By directing the search towards high-performing and behaviourally plausible candidates, Delphos can reduce the cognitive and computational burden of manual trial-and-error and reduce the risk of misspecification. In practice, the modeller defines a catalogue of attributes, socio-demographic characteristics, non-linear transformations, and taste structures. Delphos then proposes and estimates candidate specifications and presents the successfully estimated models on the Pareto front. The modeller can compare, refine, and validate these candidates while retaining control over their behavioural interpretation and final selection. Overall, Delphos functions as an intelligent assistant that automates part of the specification workflow. It supports the development of utility functions from which willingness-to-pay measures, such as values of travel time savings used in cost–benefit analysis, can be derived.

The remainder of the paper is organised as follows. Section 2 introduces the reinforcement learning concepts used in the paper. Section 3 presents our framework for assisted choice model specification across multiple datasets. Section 3.1 formulates the specification task as a Markov decision process, Section 3.2 describes the multitask extension for sharing modelling decisions across datasets, and Section 3.3 presents the training and evaluation protocol. Section 4 describes the experimental design and evaluation studies. Finally, Section 5 presents the main results, and Section 6 concludes the paper.

## 2 Background

## 2.1 Markov decision process

A Markov decision process (MDP) represents a sequential decision-making problem in which an agent interacts with an environment over time. The agent repeatedly observes the current state, selects an action, receives feedback from the environment, and transitions to a new state. Formally, an MDP is defined as the tuple $( \mathbf { S } , \mathbf { A } , \mathbf { P } , \mathbf { R } , \gamma )$ , where S represents the possible states of the environment and A represents the actions available to the agent. The transition function $\mathbf { P } ( s ^ { \prime } \mid s , a )$ defines the probability of moving from state s to a subsequent state $s ^ { \prime }$ after taking action a. The reward function $\mathbf { R } ( s , a )$ evaluates the outcome of taking action a in state s. Finally, the discount factor $\gamma$ determines the relative importance of future and immediate rewards.

$$
V _ { \pi } ( s ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { t + 1 } \Bigg | s _ { 0 } = s \right]\tag{1}
$$

The agent aims to maximise the cumulative reward obtained through its interactions with the environment. It follows a policy $\pi ( a | s ; \theta )$ that defines the probability of selecting action a when the environment is in state s. The state-value function represents the expected cumulative reward obtained when starting from state s and following policy π, as shown in Eq. (1). The optimal value function, denoted by $V ^ { * } ( s )$ , represents the maximum return achievable from state s.

## 2.2 Reinforcement learning

While an MDP formulates a sequential decision-making problem, reinforcement learning (RL) addresses that problem by learning a policy that maximises expected long-term rewards (Sutton & Barto, 2018). Through repeated interaction with the environment, the agent balances exploration and exploitation. Exploration provides additional knowledge about the environment, whereas exploitation uses that knowledge to select actions with high expected returns. At each decision step, the agent stores a transition containing the current state, selected action, received reward, and next state. These transitions are then used to refine the policy towards actions that are expected to yield higher returns.

Q-learning is commonly used to learn such a policy (Watkins & Dayan, 1992). The agent estimates an action-value function, $Q ( s , a )$ , which represents the expected cumulative reward after taking action a in state s, as shown in Eq. (2). These action values define a policy that selects the feasible action with the highest expected return in each state. Formally, the optimal policy maximises the action-value function over the set of feasible actions $\mathcal { A } ( s )$ , as shown in Eq. (3).

$$
Q ( s , a ) = \mathbb { E } \left[ r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s ^ { \prime } , a ^ { \prime } ) \mid s , a \right]\tag{2}
$$

$$
\pi ^ { * } ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } Q ^ { * } ( s , a )\tag{3}
$$

In practice, the action-value function cannot be represented explicitly when the state space is large. Deep Q-networks (DQNs; Mnih et al., 2015) address this limitation by approximating $Q ( s , a ; \theta )$ with a neural network parameterised by θ. Rather than storing a value for every state–action pair, the network predicts action values directly from the state representation. To improve training stability, a DQN also maintains a target network $Q ( s , a ; \theta ^ { - } )$ whose parameters are periodically updated from the policy network and used to compute stable learning targets. This approach enables RL agents to operate in complex decision spaces where traditional tabular methods are infeasible (Plaat, 2022).

## 3 Methodological framework

This section introduces the proposed multitask reinforcement learning framework for assisting the choice model specification process across multiple datasets. We first formulate the specification process as a Markov decision process. We then extend the formulation to multiple datasets through a shared representation of utility specifications that enables transferable modelling decisions. Finally, we present the training and evaluation protocol used to assess learning and transferability.

## 3.1 Problem formulation

We formulate discrete choice model specification as a Markov decision process in which an agent learns to build utility specifications through interaction with an estimation environment (Figure 1). Delphos is the reinforcement learning agent that learns a policy for specifying high-performing choice models (Nova et al. (2025)). At the start of each episode, Delphos uses a linear additive specification in which the available attributes enter linearly with generic coeficients and no covariate interactions. It then applies a sequence of modelling actions to modify these terms and propose a final candidate. The episode ends when the agent selects the terminate action. The environment then estimates the candidate and returns modelling outcomes used to compute a reward based on goodnessof-fit, convergence, and behavioural expectations.

![](images/f1a1037d18511b6cb7be24645b9ac7c9f5389679445c86e08783dca9e96e6c9f.jpg)  
Figure 1: Delphos: a multitask reinforcement learning framework for assisting choice model specification across datasets. The agent learns to take a sequence of modelling actions to propose utility specifications, while the environment returns feedback based on model fit, complexity, and behavioural plausibility.

The previous formulation considers a single specification problem, whereas our objective is to learn modelling strategies that transfer across related datasets. We therefore treat each choice dataset as a separate multinomial logit utility specification task τ sampled from a domain, such that $\tau \sim P ( \tau )$ (McFadden, 1978). The domain is represented by a catalogue of modelling components that can be shared across tasks, including attributes, transformations, taste structures, and covariates. Each dataset τ uses a task-specific subset that is compatible with its available variables and feasible modelling terms. Consequently, each dataset induces its own MDP, with a task-specific state space $\boldsymbol { S } _ { \tau } ,$ action space $\boldsymbol { A } _ { \tau } .$ transition function $P ( s _ { \tau , e + 1 } \mid s _ { \tau , e } , a _ { \tau , e } )$ , and reward function $\mathcal { R } _ { \tau }$

To describe modelling decisions across specification problems, we define a domain catalogue containing all modelling components that may appear across the considered datasets. Formally, the domain catalogue is defined as

$$
{ \mathcal { C } } = \{ { \mathcal { K } } , { \mathcal { T } } , { \mathcal { G } } , { \mathcal { V } } \} ,
$$

where K is the set of available attributes that may enter the utility functions, such as travel time and travel cost. $\tau$ is the set of transformations that can be applied to these attributes, including linear, logarithmic, and Box–Cox transformations. G is the set of taste structures, which determine whether a parameter is generic across alternatives or alternative-specific. V is the set of covariates used to capture observed heterogeneity. Each dataset uses a subset $\mathcal { C } _ { \tau }$ and induces an MDP with a dataset-specific state space ${ \cal { S } } _ { \tau } ,$ action space $\mathbf { \mathcal { A } } _ { \tau }$ transition function $\mathcal { P } _ { \tau }$ , and reward function $\mathcal { R } _ { \tau }$

i) State space $( S _ { \tau } )$ . The state space comprises all feasible utility specifications that can be constructed from the dataset-specific catalogue $\mathcal { C } _ { \tau }$ . Each specification is represented as a set of modelling terms. Each term encodes a decision about how a variable enters the utility function. Specifically, a modelling term combines an attribute $k _ { l } \in \mathcal { K } _ { \tau }$ with its transformation $t _ { l } \in \mathcal { T } _ { \tau }$ , taste structure $g _ { l } \in \mathcal { G } _ { \tau }$ , and covariate interaction $v _ { l } \in \mathcal { V } _ { \tau }$ . Formally, the state at interaction e is

$$
\begin{array} { r } { s _ { e } ^ { \tau } = \{ x _ { l } \} _ { l = 1 } ^ { L _ { e } } , \quad x _ { l } = ( k _ { l } , t _ { l } , g _ { l } , v _ { l } ) , } \end{array}\tag{4}
$$

where $L _ { e }$ denotes the number of modelling terms in the specification. Attributes that are not included in the current specification are encoded as $t _ { l } = g _ { l } = v _ { l } = 0$

To illustrate this representation, consider the following utility specification for the bus alternative. It contains four modelling terms: an alternative-specific constant; travel time with a logarithmic transformation and an alternative-specific coeficient; travel cost entering linearly with a generic coeficient that interacts with the two levels of the covariate cov1; and headway, which is excluded from the specification.

V[["bus"]]= asc\_bus #(1, linear, specific, none)   
+ b\_tt\_bus \* log(TT\_bus) #(2, log, specific, none)   
+ b\_tc\_female \* Female \* TC\_bus #(3, linear, generic, female)   
+ b\_tc\_male \* (1-Female) \* TC\_bus #(3, linear, generic, male)

Using the catalogue indices (e.g., attributes: 0=None, 1=ASC, 2=TT, 3=TC, 4=HE; transformations: 0=None, 1=Linear, 2=Log, 3=Box–Cox; taste structures: 0=None, 1=Generic, 2=Alternative-specific; covariates: 0=None, 1=COV1), the corresponding state is

$$
s _ { e } ^ { \tau } = \{ ( 1 , 1 , 2 , 0 ) , ( 2 , 2 , 2 , 0 ) , ( 3 , 1 , 1 , 1 ) , ( 4 , 0 , 0 , 0 ) \}\tag{5}
$$

This representation preserves the behavioural decisions that define a utility specification while separating them from dataset-specific variable labels and availability. Consequently, variables that capture the same behavioural construct can be encoded consistently across datasets. Examples include total and in-vehicle travel time, or fare and travel cost. This shared representation provides the basis for transferring modelling experience from one specification problem to related tasks.

ii) Action space $( \mathcal { A } _ { \tau } )$ . The action space defines the feasible operations that the agent can use to modify the current utility specification. Because each state is represented as a set of modelling terms, the agent operates directly on these terms. It can add a term, modify an existing term, or terminate the specification process. Formally, the action space is

$$
a \in \{ \mathrm { a d d } ( k , t , g , v ) , \ \mathrm { c h a n g e } ( k , t , g , v ) , \ \mathrm { t e r m i n a t e } \} ,\tag{6}
$$

where add introduces a new modelling term, change updates the transformation, taste structure, or covariate interaction of an existing term, and terminate ends the specification process and triggers model estimation.

Continuing the example, suppose that the agent first selects add(HE, linear, generic, none), which introduces headway as a new modelling term. The updated specification is

```diff
s<sup>0</sup><sub>e</sub> → add(HE, linear, generic, none) → s<sup>1</sup><sub>e</sub> 1
V[["bus"]]= asc_bus #(1, linear, specific, none)
+ b_tt_bus * log(TT_bus) #(2, log, specific, none)
+ b_tc_female * Female * TC_bus #(3, linear, generic, female)
+ b_tc_male * (1-Female) * TC_bus #(3, linear, generic, male)
+ b_he * HE_bus #(4, linear, generic, none)
```

The agent may then choose change(TT, Box-Cox, specific, none). This action replaces the logarithmic transformation of travel time with a Box–Cox transformation while preserving the remaining modelling decisions. The resulting specification is

```diff
s<sup>1</sup><sub>e</sub> → change(TT, box-cox, specific, none) → 2
V[["bus"]]= asc_bus #(1, linear, specific, none)
+ b_tt_bus * BoxCox(TT_bus) #(2, box-cox, specific, none)
+ b_tc_female * Female * TC_bus #(3, linear, generic, female)
+ b_tc_male * (1-Female) * TC_bus #(3, linear, generic, male)
+ b_he * HE_bus #(4, linear, generic, none)
```

This formulation decomposes utility specification into a sequence of modelling decisions, whereby the agent progressively constructs a utility specification by adding and modifying modelling terms until it selects the terminate action. The previous example thus corresponds to a trajectory of state transitions, in which the agent progressively updates the specification through successive modelling actions before submitting the final candidate for estimation.

s<sub>0</sub>   
,→ add(HE, linear, generic, none) → s<sub>1</sub>   
,→ change(TT, box–cox, specific, none) → s<sub>2</sub>   
,→ terminate → s<sub>3</sub>   
,→ environment estimates s<sub>3</sub> → reward

To ensure that each decision contributes meaningfully to the specification process, the available actions are dynamically restricted. The agent cannot select modelling components that are unavailable in the dataset, actions that immediately reverse previous modifications, or actions already selected in the current episode. An action-masking mechanism implements these restrictions and ensures that the agent explores only feasible and nonredundant specification trajectories (Huang & Ontañón, 2020).

iii) Reward signal (R<sub>τ</sub>). The reward signal evaluates the quality of the utility specification at the end of an episode. Through delayed credit assignment, this feedback is propagated to the preceding sequence of modelling decisions. The agent can therefore learn which decisions improve model performance over repeated trials. In a multitask setting, however, goodness-of-fit measures such as log-likelihood cannot be used directly because their magnitude depends on the sample size, model complexity, and baseline fit of each dataset. Similar modelling decisions may therefore produce substantially diferent rewards across datasets. This could bias learning towards datasets with larger absolute improvements in log-likelihood. To provide a stable signal across heterogeneous datasets, we define the reward as the per-observation improvement in log-likelihood relative to a baseline model:

$$
r = \operatorname { t a n h } \Bigl ( \frac { L L _ { \hat { \beta } } - L L _ { b a s e l i n e } } { N _ { o b s } } \Bigr ) ,\tag{7}
$$

Normalising by the number of observations makes the reward comparable across datasets of diferent sizes. The tanh transformation bounds the reward within (−1, 1) and improves numerical stability in a similar way to reward clipping (Mnih et al., 2015; Hessel et al., 2019). We use the null model as the baseline because it provides a lower bound for theoretically valid maximum-likelihood specifications (Mokhtarian, 2016). Specifications that fail to converge receive a fixed penalty of −1. The reward can also accommodate richer modelling objectives through penalties or incentives based on behavioural expectations. For example, it could discourage specifications with implausible parameter signs, such as positive travel-time or travel-cost sensitivities, statistically insignificant parameters, or estimation issues (Nova et al. (2025)).

iv) Environment. The environment provides the interface between the reinforcement learning agent and the choice model estimation framework. After the agent selects the terminate action, the environment translates the final state into a complete utility specification and estimates it on the corresponding dataset using Apollo (Hess & Palma, 2019). The resulting outcomes, including goodness-of-fit measures, convergence diagnostics, and parameter estimates, are used to compute the reward in Eq. (7).

## 3.2 Learning transferable modelling decisions

To enable a single agent to learn and reuse modelling experience across choice datasets, we extend Delphos to a multitask reinforcement learning framework. Instead of training a separate agent for each specification problem, Delphos learns jointly across related problems. It learns both which modelling decisions tend to produce high-performing utility specifications and the contexts in which those decisions are appropriate. A single policy, however, requires a common representation of utility specifications despite diferences in the number and composition of their modelling terms.

To address this heterogeneity, we implement Delphos as a DeepSet-Q network (Hügle et al., 2020). DeepSets encode unordered sets with varying numbers of elements as fixedlength latent representations (Zaheer et al., 2017). They are therefore well suited to utility specifications, whose modelling terms have no intrinsic order and may difer in number and composition. In the single-task framework of Nova, Hess, & van Cranenburgh (2025), each specification is represented as a fixed-length vector. By contrast, we represent utility specifications as sets of modelling terms. Specifications from diferent datasets can therefore be represented consistently and processed by a single policy.

![](images/b785c1bfc4e3d471856c310915e3d3b2d18a5ec875eb6ac5f2f0598cc10213ae.jpg)  
Figure 2: DeepSet-Q network (adapted from Hügle et al. (2020)). It combines a specification encoder with a context-dependent Q-network. Each modelling term is embedded by a shared network $\phi ( \cdot )$ and then aggregated through pooling $\rho ( \cdot )$ to obtain a latent representation of the specification.

Figure 2 shows the proposed architecture, which combines a specification encoder with a context-dependent Q-network. The architecture separates the representation of a utility specification from the modelling context in which it is applied. The first component is the DeepSet encoder (Zaheer et al., 2017), which maps the set of modelling terms $s _ { e } ^ { \tau } = \{ x _ { l } \} _ { l = 1 } ^ { L _ { e } }$ to a fixed-length latent representation. Each term $x _ { l }$ is embedded using a shared neural network $\phi ( \cdot )$ and then aggregated through a permutation-invariant pooling operation,

$$
Z ( s _ { e } ^ { \tau } ) = \rho \left( \sum _ { l = 1 } ^ { L _ { e } } \phi ( x _ { l } ) \right) ,\tag{8}
$$

where $Z ( s _ { e } ^ { \tau } )$ is the latent representation of the utility specification. Because the aggregation is invariant to the order and number of modelling terms, specifications from diferent datasets are mapped to a common latent space.

The specification embedding alone does not identify which modelling terms are available for the current dataset. Delphos therefore uses a task-specific context vector $x ^ { \tau }$ , which encodes the availability of attributes, transformations, taste structures, and covariates in the dataset-specific catalogue $\mathcal { C } _ { \tau }$ . The specification embedding $Z ( s _ { e } ^ { \tau } )$ and context vector $x ^ { \tau }$ are concatenated and provided as input to a deep Q-network (Mnih et al., 2015),

$$
q = Q ( Z ( s _ { e } ^ { \tau } ) | | x ^ { \tau } ) ,\tag{9}
$$

which estimates the value of each feasible modelling decision in the current state. By combining a shared specification representation with a task-specific context, Delphos learns a single policy that can be applied across related datasets.

## 3.3 Multitask training and evaluation

Delphos is trained jointly across all specification tasks using a single DeepSet-Q network whose parameters are shared across datasets. During each episode, a dataset $\tau$ is sampled at random, and the agent selects modelling actions until it reaches the terminate action. The environment estimates the resulting utility specification and assigns a reward according to $\operatorname { E q } .$ (7). Each interaction generates a transition $( s _ { e } ^ { \tau } , a _ { e } ^ { \tau } , r _ { e } ^ { \tau } , s _ { e + 1 } ^ { \tau } )$ , which is stored in a shared experience replay bufer. The bufer therefore contains modelling experience from all training datasets and is used to update the network.

The network parameters are updated following the deep Q-learning framework (Mnih et al., 2015). At each optimisation step, a mini-batch B of transitions is sampled from the bufer, and the temporal-diference loss is minimised,

$$
\mathcal { L } ( \theta ) = \frac { 1 } { | \mathcal { B } | } \sum _ { b = 1 } ^ { | \mathcal { B } | } \left[ Q _ { \mathrm { t a r g e t } } ( s ^ { \prime } , a ^ { \prime } ; \theta ^ { - } ) - Q ( s , a ; \theta ) \right] ^ { 2 } ,\tag{10}
$$

where $\theta$ and $\theta ^ { - }$ denote the parameters of the policy and target networks, respectively. The loss compares the estimated value of a modelling decision with a target that combines the observed reward and expected value of subsequent decisions. By minimising this error, Delphos learns to favour decisions that improve the current utility specification and lead to more promising sequences of decisions over time.

To prevent datasets with longer specification trajectories from dominating optimisation, we sample balanced mini-batches from the replay bufer so that each task contributes similarly to the parameter updates (Sutton & Barto, 2018). During training, ϵ-greedy exploration is gradually reduced, shifting the agent from exploration towards decisions associated with high-performing specifications. This design follows the purpose of experience replay in deep reinforcement learning (Schaul et al., 2015) and provides balanced (Sodhani et al., 2021) and diverse (Ross & Bagnell, 2010) exposure across tasks.

During inference, the trained agent is applied directly to unseen datasets without further training. Given the task-specific catalogue $\mathcal { C } _ { \tau }$ , Delphos selects a sequence of modelling decisions before submitting a utility specification for estimation. By repeating this process, Delphos proposes and estimates a set of high-performing candidates. Modellers can then inspect and compare the resulting models using criteria such as goodness-of-fit, behavioural plausibility, and parameter significance. Delphos thus acts as a specification assistant that reuses experience from related problems while adapting its decisions to previously unseen datasets.

## 4 Experiments

This section evaluates the proposed multitask reinforcement learning framework. We first introduce the choice datasets used for training and inference (Subsection 4.1). We then describe the metrics used to assess learning and transferability (Subsection 4.2). Finally, we present the studies used to evaluate whether modelling experience transfers across specification problems and whether the specifications generated for unseen datasets are behaviourally plausible (Subsection 4.3).

## 4.1 Datasets

Table 1 summarises the transport choice datasets used in this paper. The first nine are used for training, while Decisions and Swissmetro are held out for inference. Each dataset defines a diferent specification problem, with its own choice context, alternatives, available attributes, socio-demographic variables, and sample size. Although the datasets difer in their observed variables and experimental settings, they share behavioural concepts such as travel time and travel cost, among others.

## 4.2 Evaluation metrics

The experiments evaluate two aspects of the proposed framework: (i) its ability to learn sequences of modelling decisions that produce high-performing utility specifications during multitask training and (ii) its ability to transfer these strategies to unseen specification problems during inference.

During training, Delphos sequentially proposes utility specifications and receives feedback from the estimation environment. We evaluate learning performance using the average reward, maximum reward, and area under the learning curve. These metrics measure the average quality of the modelling trajectories, the quality of the best specification found, and overall learning eficiency, respectively. To characterise exploration, we also report the convergence rate, defined as the proportion of estimated specifications that converge successfully, and the proportion of novel specifications explored during training.

During inference, the trained agent is applied to unseen datasets without further training. We use the same metrics to evaluate the generated specifications and the transferability of the learnt policy. We also derive the Pareto front to examine the trade-of between model fit and complexity and to characterise the models that Delphos tends to propose.

## 4.3 Evaluation studies

To evaluate Delphos’ ability to learn transferable modelling strategies, we examine whether it can use experience across datasets to improve sample eficiency and whether the generated specifications balance model fit, parsimony, and behavioural plausibility.

First, we evaluate whether the proposed multitask framework improves Delphos’ ability to learn and transfer modelling experience across specification problems. We compare the proposed DeepSet-Q architecture with the single-task DQN framework introduced by Nova et al. (2025). All agents are trained for 10,000 episodes using the hyperparameter configuration reported in Appendix 5. The multitask agent is trained jointly across all training datasets, whereas each single-task agent is trained independently on its corresponding dataset. We evaluate learning and transfer using the metrics introduced in Subsection 4.2.

Table 1: Choice modelling tasks used for multitask training and inference.
<table><tr><td>Task τ</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td></td><td>T8 T9</td><td>T10</td><td>T11</td><td></td></tr><tr><td>Training datasets</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Choice attributes Travel time</td><td></td><td>√</td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Travel cost</td><td>√ √</td><td>√</td><td>√ √</td><td>√</td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√ √</td></tr><tr><td>Out-of-vehicle time</td><td>√</td><td></td><td></td><td></td><td>√</td><td>√ √</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Transfers</td><td></td><td>√</td><td>√</td><td></td><td>√ √</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Service quality</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td><td>√</td><td></td></tr><tr><td>Reliability</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Socio-demographic variables Gender</td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Income</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td>√ √</td><td>√ √</td><td>√ √</td><td>√</td></tr><tr><td>Age</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td></td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Purpose</td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td><td>√ √</td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Car access</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Business</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>Education</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Dataset characteristics</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Panel data</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>Alternatives</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td>2</td><td></td><td></td><td></td></tr><tr><td>Sample size</td><td>4</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td><td></td><td>4</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>7000</td><td>3492</td><td>1511</td><td>52488</td><td>6570</td><td>1576</td><td></td><td>81086</td><td>1521</td><td>850</td><td>9356</td><td>5409</td></tr><tr><td>LL0 LLlinear</td><td>-8196 -5761-1665</td><td>-2420</td><td>-1047</td><td>-36381 -33026</td><td>-5885</td><td>-1731</td><td></td><td>-112409</td><td>-1054</td><td>-966</td><td>-11233</td><td>-5548</td></tr><tr><td></td><td></td><td></td><td>-732</td><td></td><td></td><td>-4734</td><td>-1121</td><td>-71099</td><td>-812</td><td>-924</td><td>-4294</td><td>-4346</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>(00%)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>. et</td><td></td><td></td></tr><tr><td></td><td></td><td>(2000)</td><td></td><td></td><td>(200)</td><td></td><td></td><td></td><td></td><td>Cramggh</td><td>(202)</td><td>(200)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>(20)</td><td>(208)</td><td></td><td></td><td></td><td>al</td></tr><tr><td></td><td></td><td>A  xae.</td><td>(208)</td><td></td><td></td><td></td><td></td><td></td><td>(00)</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>a.</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>t a.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Birai)</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> e  ma</td><td></td><td>Biraaie</td><td>Ra  a.</td><td>AIrn l  13)</td><td>eseet</td><td>Hil</td><td></td><td>Van</td><td>Ca tatsta.</td><td>Bieeaet</td></tr><tr><td></td><td></td></table>

Second, we assess the quality of the utility specifications generated by Delphos for unseen datasets. We compare the Pareto-optimal specifications identified by Delphos with benchmark models from the original studies and with other assisted specification approaches. We consider model performance, parsimony, and behavioural plausibility to determine whether the transferred policy produces competitive models that are consistent with established choice modelling practice.

## 5 Results

This section presents the results of the two evaluation studies. We first examine learning during multitask training and then assess the utility specifications generated for unseen datasets.

## 5.1 Learning transferable specification strategies

Figure 3 compares the learning curves of the proposed multitask DeepSet-Q agent with those of the single-task DQN agents across the training datasets. For most datasets, both approaches achieve progressively higher rewards during training. However, the multitask agent generally learns faster and reaches higher reward levels. This suggests that updating a single policy with experience from multiple specification problems improves its ability to generalise across datasets.

![](images/e812588baf1469a8bc235791460f091b80bbc5c97f69d7938a9d02abc6a5169b.jpg)  
Figure 3: Learning curves of the single-task DQN agents and the proposed multitask DeepSet-Q agent across the training datasets.

Learning performance varies across datasets. The gains are larger for LondonModeChoice, SpainParkingChoice, and ApolloModeChoice, and the diferences are smaller for NorwayVTT and VanCranenburghVOT. One possible explanation is the complexity of the underlying specification problem. Datasets with more attributes and feasible covariate interactions define larger specification spaces and require longer sequences of modelling decisions. In these settings, experience from related datasets provides greater guidance during specification. By contrast, NorwayVTT and VanCranenburghVOT contain only two attributes: travel time and travel cost. Their smaller specification and action spaces make transferred modelling experience less beneficial.

Table 2 summarises the performance of the single-task DQN agents and the proposed multitask DeepSet-Q agent across the training datasets. The multitask agent generally achieves higher mean rewards, maximum rewards, and areas under the learning curve. These improvements indicate that sharing modelling experience across specification problems enables Delphos to learn more efective policies. The main exception is VanCranenburghVOT, for which the two approaches perform similarly. This result is consistent with its smaller specification space.

From a reinforcement learning perspective, the higher rewards are accompanied by larger areas under the learning curve, which suggests that Delphos reaches high-performing specifications earlier during training. Notably, these improvements are obtained while exploring a slightly smaller proportion of novel specifications. This suggests that modelling experience acquired from related datasets makes the search more sample-eficient, which allows the agent to identify promising sequences of modelling decisions with fewer unsuccessful specification trials.

Table 2: Learning performance on the training datasets.
<table><tr><td>Dataset</td><td>Agent</td><td>Mean Reward</td><td>Max Reward</td><td>AUC</td><td>Novel Specs (%)</td><td>Estimable Specs (%)</td></tr><tr><td rowspan="2">ApolloModeChoice</td><td>DQN</td><td>0.151</td><td>0.334</td><td>1466</td><td>93.66</td><td>82.74</td></tr><tr><td>DeepSet-Q</td><td>0.187</td><td>0.434</td><td>1839</td><td>91.75</td><td>85.56</td></tr><tr><td rowspan="2">SwissmetroRouteChoice</td><td>DQN</td><td>0.033</td><td>0.219</td><td>291</td><td>95.86</td><td>84.62</td></tr><tr><td>DeepSet-Q</td><td>0.082</td><td>0.220</td><td>761</td><td>92.87</td><td>88.52</td></tr><tr><td rowspan="2">NLModeChoice</td><td>DQN</td><td>-0.111</td><td>0.216</td><td>-1161</td><td>96.27</td><td>72.80</td></tr><tr><td>DeepSet-Q</td><td>-0.042</td><td>0.205</td><td>-497</td><td>89.54</td><td>78.61</td></tr><tr><td rowspan="2">NorwayVTT</td><td>DQN</td><td>0.130</td><td>0.168</td><td>1300</td><td>26.24</td><td>99.84</td></tr><tr><td>DeepSet-Q</td><td>0.140</td><td>0.173</td><td>1394</td><td>22.66</td><td>99.84</td></tr><tr><td rowspan="2">Arentze2013</td><td>DQN</td><td>-0.273</td><td>0.008</td><td>-2825</td><td>77.90</td><td>62.01</td></tr><tr><td>DeepSet-Q</td><td>-0.199</td><td>0.111</td><td>-2096</td><td>63.32</td><td>68.35</td></tr><tr><td rowspan="2">SpainParkingChoice</td><td>DQN</td><td>-0.237</td><td>0.176</td><td>-2473</td><td>17.99</td><td>54.91</td></tr><tr><td>DeepSet-Q</td><td>-0.058</td><td>0.270</td><td>-698</td><td>13.64</td><td>68.21</td></tr><tr><td rowspan="2">LondonModeChoice</td><td>DQN</td><td>-0.196</td><td>0.518</td><td>-2086</td><td>96.74</td><td>52.62</td></tr><tr><td>DeepSet-Q</td><td>-0.056</td><td>0.497</td><td>-716</td><td>91.81</td><td>61.22</td></tr><tr><td rowspan="2">Optima</td><td>DQN</td><td>-0.297</td><td>0.181</td><td>-3060</td><td>96.47</td><td>57.61</td></tr><tr><td>DeepSet-Q</td><td>-0.228</td><td>0.183</td><td>-2376</td><td>89.27</td><td>63.83</td></tr><tr><td rowspan="2">VanCranenburghVOT</td><td>DQN</td><td>0.064</td><td>0.086</td><td>635</td><td>27.77</td><td>99.73</td></tr><tr><td>DeepSet-Q</td><td>0.060</td><td>0.089</td><td>597</td><td>25.20</td><td>99.72</td></tr></table>

This behaviour is also reflected from a choice modelling perspective. Although the multitask agent explores fewer unique specifications, it consistently produces a larger proportion of successfully estimated models across almost all datasets. This suggests that the transferred modelling experience not only improves model fit but also guides the search towards specifications that are more likely to be successful estimated.

## 5.2 Application to unseen datasets

## Swissmetro

When applied to the unseen Swissmetro dataset (τ<sub>11</sub>; (Bierlaire et al., 2001)), Delphos proposes 283 candidate models within a computational budget of 20 minutes on a laptop CPU. Figure 4 shows all converged candidates and the resulting Pareto front, whose nondominated models balance model performance and parsimony. Thirteen of the converged specifications are non-dominated. Within the computational budget, Delphos identifies a best specification with a log-likelihood of −0.681 per observation. This result improves on both the linear baseline $( L L _ { \mathrm { l i n e a r } } = - 1 . 0 2 9 )$ and the value of −0.72 per observation reported for the VNS assisted specification method by Ortelli et al. (2021).

![](images/5c2c1dc16ecf94401dde3fda0cbe0624e1e0efa5604f5f5a0d60d56b48a6b480.jpg)  
Figure 4: Pareto front of Delphos candidates on the Swissmetro dataset. Grey dots show all converged specifications proposed during inference. Stars indicate the non-dominated candidates, and circles show the specifications with the best AIC and log-likelihood values.

Table 3: Best specification identified by Delphos on Swissmetro.
<table><tr><td>Modelling term</td><td>Included</td><td>Taste</td><td>Transformation</td><td>Interaction</td></tr><tr><td>ASC</td><td>√</td><td>Specific</td><td></td><td></td></tr><tr><td>Travel time</td><td>√</td><td>Specific</td><td>Log</td><td>Age</td></tr><tr><td>Travel cost</td><td>V</td><td>Specific</td><td>Log</td><td>Gender</td></tr><tr><td>Headway</td><td>V</td><td>Generic</td><td>Linear</td><td>Gender</td></tr><tr><td>Seat type</td><td>V</td><td>Generic</td><td>Linear</td><td>Income</td></tr><tr><td>LL(0)</td><td></td><td></td><td></td><td>-6,964.66</td></tr><tr><td>LL(final)</td><td></td><td></td><td></td><td>-4,603.96</td></tr><tr><td>AIC</td><td></td><td></td><td></td><td>9,265.92</td></tr><tr><td>BIC</td><td></td><td></td><td></td><td>9,463.70</td></tr><tr><td>Rho-squared</td><td></td><td></td><td></td><td>0.34</td></tr><tr><td>Adj. Rho-squared</td><td></td><td></td><td></td><td>0.34</td></tr><tr><td>Observations</td><td></td><td></td><td></td><td>6,768</td></tr><tr><td>Number of parameters</td><td></td><td></td><td></td><td>29</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LL/N - Delphos LL/N – VNS Ortelli et al. (2021)</td><td></td><td></td><td></td><td>-0.68 -0.72</td></tr></table>

Table 3 reports the specification identified by Delphos and its fit statistics. The model includes constants specific to each alternative and relevant attributes with either generic parameters or parameters that vary across alternatives. It also introduces non-linear transformations and meaningful covariate interactions. These results suggest that Delphos can transfer modelling decisions to an unseen dataset and identify a competitive, behaviourally plausible specification within a limited computational budget. Inference runs on a standard laptop CPU, making the approach practical for analysts seeking to reduce specification effort and for researchers studying reusable modelling strategies.

## Decisions

We apply the same trained multitask agent to the unseen Decisions dataset (τ<sub>10</sub>; (Calastri et al., 2020)) using the same 20-minute computational budget. Figure 5 shows all converged specifications and the resulting Pareto front. Without further training, Delphos identifies a set of non-dominated models. This result suggests that the learnt policy continues to select efective modelling actions for a diferent and unseen specification problem. The specification with the best AIC has a log-likelihood of −0.37 per observation, improving on the linear additive model $( L L _ { \mathrm { l i n e a r } } = - 0 . 4 6 )$ . Its performance is also comparable to the value of −0.36 reported for the MNL specification developed by expert modellers in Tsoleridis et al. (2022), as shown in Table 4. The Delphos specification includes alternativespecific constants, non-linear transformations, and interactions with socio-demographic variables.

![](images/a9c612bb2e98ae1ccd952b84348075387dc5c68ca0a2e34c18deb183c3350594.jpg)  
Figure 5: Pareto front of Delphos candidates on the Decisions dataset. Grey dots show all converged specifications proposed during inference. Stars indicate the non-dominated candidates, and circles show the specifications with the best AIC and log-likelihood values.

Although the two datasets difer substantially in sample size, choice-set structure, and available variables, the same transferred policy identifies competitive specifications for both. Delphos achieves a higher log-likelihood per observation than the VNS benchmark on Swissmetro and performance comparable to a published MNL specification developed by expert modellers on Decisions.

Table 4: Best specification identified by Delphos on Decisions.
<table><tr><td>Modelling term</td><td>Included</td><td>Taste</td><td>Transformation</td><td>Interaction</td></tr><tr><td>ASC</td><td>√</td><td>Specific</td><td></td><td>Purpose</td></tr><tr><td>Travel time</td><td>V</td><td>Specific</td><td>Linear</td><td>N. Car</td></tr><tr><td>Travel cost</td><td>V</td><td>Generic</td><td>Box-Cox</td><td>N. Car</td></tr><tr><td>Headway</td><td>V</td><td>Specific</td><td>Log</td><td>Gender</td></tr><tr><td>LL(0)</td><td></td><td></td><td></td><td>-11,233.44</td></tr><tr><td>LL(final)</td><td></td><td></td><td></td><td>-3,481.79</td></tr><tr><td>AIC</td><td></td><td></td><td></td><td>7,055.59</td></tr><tr><td>BIC</td><td></td><td></td><td></td><td>7,384.20</td></tr><tr><td>Rho-squared</td><td></td><td></td><td></td><td>0.69</td></tr><tr><td>Adj. Rho-squared</td><td></td><td></td><td></td><td>0.69</td></tr><tr><td>Observations</td><td></td><td></td><td></td><td>9,356</td></tr><tr><td>Number of parameters</td><td></td><td></td><td></td><td>46</td></tr><tr><td>LL/N – Delphos</td><td></td><td></td><td></td><td>-0.37</td></tr><tr><td>LL/N – Tsoleridis et al. (2022)</td><td></td><td></td><td></td><td>-0.36</td></tr></table>

## 6 Conclusions

This paper extends Delphos to a multitask reinforcement learning setting for assisted discrete choice model specification across related transport datasets. We formulate specification as a sequential decision problem in which an agent builds a candidate through modelling actions and submits it to an estimation environment. A DeepSet-Q architecture represents each candidate as a set of modelling terms and combines this representation with dataset-specific context. This design allows the same decision rule to be applied across datasets with diferent variables.

Our results show that Delphos learns reusable modelling strategies that transfer across specification problems. During training, the multitask agent consistently outperforms independently trained single-task agents. It achieves higher rewards, greater sample eficiency, and a larger proportion of successfully estimated specifications while exploring fewer candidates. These results suggest that shared modelling experience guides the search towards promising specification trajectories rather than simply encouraging broader exploration. When applied without further training to the unseen Swissmetro and Decisions datasets, the same agent identifies competitive and behaviourally meaningful specifications in less than 20 minutes on a standard CPU. It achieves a higher log-likelihood per observation than the VNS benchmark on Swissmetro and performance comparable to a published MNL specification developed by expert modellers on Decisions. Overall, these findings highlight the potential of reinforcement learning to accumulate and reuse experience for assisted choice model specification.

Beyond these results, the framework can reduce the cognitive and computational burden of manual trial and error and provide suggestions for behaviourally sound utility specifications. This support may help reduce the risk of misspecification, which can bias parameter estimates, weaken forecasts, and lead to misleading welfare conclusions. Delphos may therefore help choice modellers build utility functions that can be used to derive marginal efects and willingness-to-pay measures, including values of travel time savings for costbenefit analysis.

The multitask training setup has several limitations. First, Delphos is trained on only nine datasets and evaluated on two unseen datasets. Although these datasets have distinct catalogues, broader validation across additional problems is needed to assess the generality of the learnt policy. Second, the reward function is based mainly on goodness-of-fit and convergence. It could be extended to include explicit behavioural constraints, as in (Nova,

Hess, & van Cranenburgh, 2025). Finally, the current framework transfers a fixed policy without adapting it to new datasets during inference.

Delphos can be then used as a software package for assisted choice model specification and estimation. The modeller first defines a catalogue of attributes, socio-demographic characteristics, non-linear transformations, and taste structures. The agent then proposes and returns successfully estimated utility specifications on the Pareto front. The modeller can compare these models and select promising candidates for further diagnosis, refinement, and validation. This workflow automates part of the search while retaining modeller control over interpretation and final selection.

Future work will extend the experimental setup to a larger set of datasets to assess transferability across a broader range of modelling contexts. We will also investigate context representations that describe each problem using dataset statistics and behavioural features, rather than only the availability of catalogue components. These representations may enable more informed decisions across heterogeneous datasets. Finally, we will evaluate few-shot adaptation strategies to determine when a shared policy can be eficiently specialised to a new dataset using only a small number of additional training episodes.

## Acknowledgements

Stephane Hess acknowledges support from the European Research Council through the Advanced Grant 101020940-SYNERGY.

Table 5: Hyperparameter configuration used for DeepSet-Q.
<table><tr><td colspan="2">Domain catalogue</td></tr><tr><td>Attributes (|K|)</td><td>7</td></tr><tr><td>Transformations (|T|)</td><td>3</td></tr><tr><td>Taste structures (|9|)</td><td>2</td></tr><tr><td>Covariates (|ν|)</td><td>7</td></tr><tr><td>Training datasets</td><td>9</td></tr><tr><td colspan="2">DeepSet encoder</td></tr><tr><td>Input modelling term Embedding dimensions Hidden layer 1</td><td>(k, t, g, v) (16, 8, 8, 16)</td></tr><tr><td></td><td></td></tr><tr><td>Hidden layer 2</td><td>128</td></tr><tr><td>Modelling term (dterm)</td><td>64</td></tr><tr><td>Pooling</td><td>64</td></tr><tr><td></td><td>Mean</td></tr><tr><td>Specification embedding (Z(s))</td><td>64</td></tr><tr><td>Activation</td><td>ReLU</td></tr><tr><td>Weight initialisation</td><td>Kaiming uniform</td></tr><tr><td>Trainable parameters</td><td>27,384</td></tr><tr><td>Deep Q-network</td><td></td></tr><tr><td>Specification embedding</td><td>64</td></tr><tr><td>Context vector Input dimension</td><td>19 83</td></tr><tr><td>Hidden layer 1</td><td>256</td></tr><tr><td>Hidden layer 2</td><td></td></tr><tr><td>Output actions</td><td>256</td></tr><tr><td></td><td>297</td></tr><tr><td>Activation</td><td>ReLU</td></tr><tr><td>Weight initialisation</td><td>Kaiming uniform</td></tr><tr><td>Trainable parameters</td><td>158,761</td></tr><tr><td>Training</td><td>Value</td></tr><tr><td>Episodes per task</td><td></td></tr><tr><td>Learning updates</td><td>10,000</td></tr><tr><td></td><td>10,000</td></tr><tr><td>Replay buffer sampling</td><td>10,000</td></tr><tr><td>Mini-batch size</td><td>64</td></tr><tr><td>Discount factor (γ)</td><td>0.90</td></tr><tr><td>Learning rate</td><td>1 × 10−3</td></tr><tr><td></td><td>0.005</td></tr><tr><td>Target update (τ) €-greedy exploration</td><td>Linear decay</td></tr></table>

## References

Arentze, T. A., & Molin, E. J. (2013). Travelers’ preferences in multimodal networks: Design and results of a comprehensive series of choice experiments. Transportation Research Part A: Policy and Practice, 58, 15–28.

Axhausen, K. W., Hess, S., König, A., Abay, G., Bates, J. J., & Bierlaire, M. (2008). Income and distance elasticities of values of travel time savings: New swiss results. Transport Policy, 15(3), 173–185.

Beeramoole, P. B., Arteaga, C., Pinz, A., Haque, M. M., & Paz, A. (2023). Extensive hypothesis testing for estimation of mixed-logit models. Journal of choice modelling, 47 , 100409.

Bierlaire, M. (2018). Description of the netherlands mode choice dataset. Transport and Mobility Laboratory, EPFL. Retrieved from https://transp-or.epfl.ch/documents/ technicalReports/CS\_ParkingDescription.pdff (Biogeme example dataset)

Bierlaire, M., Axhausen, K., & Abay, G. (2001). The acceptance of modal innovation: The case of swissmetro. In Swiss transport research conference.

Buckell, J., Marti, J., & Sindelar, J. L. (2019). Should flavours be banned in cigarettes and ecigarettes? evidence on adult smokers and recent quitters from a discrete choice experiment. Tobacco control, 28 (2), 168–175.

Calastri, C., Crastes dit Sourd, R., & Hess, S. (2020). We want it all: experiences from a survey seeking to capture social network structures, lifetime events and short-term travel and activity planning. Transportation, 47 , 175–201.

de Vries, M. O., Hernández, J. I., & Mouter, N. (2025). The value of accessibility, health, safety, inclusion and sustainability: a public willingness to pay experiment. Research in Transportation Economics, 113, 101605.

Ghorbani, A., Nassir, N., Lavieri, P. S., Beeramoole, P. B., & Paz, A. (2025). Enhanced utility estimation algorithm for discrete choice models in travel demand forecasting. Transportation, 1–28.

Haj-Yahia, S., Mansour, O., & Toledo, T. (n.d.). Grammar-based approach to data-driven utility specification for discrete choice models. Available at SSRN 5195530.

Hess, S., & Daly, A. (2024). Handbook of choice modelling. Edward Elgar Publishing.

Hess, S., & Palma, D. (2019). Apollo: A flexible, powerful and customisable freeware package for choice model estimation and application. Journal of choice modelling, 32, 100170.

Hessel, M., Soyer, H., Espeholt, L., Czarnecki, W., Schmitt, S., & Van Hasselt, H. (2019). Multi task deep reinforcement learning with popart. In Proceedings of the aaai conference on artificial intelligence (Vol. 33, pp. 3796–3803).

Hillel, T., Elshafie, M. Z., & Jin, Y. (2018). Recreating passenger mode choice-sets for transport simulation: A case study of london, uk. Proceedings of the Institution of Civil Engineers-Smart Infrastructure and Construction, 171 (1), 29–42.

Huang, S., & Ontañón, S. (2020). A closer look at invalid action masking in policy gradient algorithms. arXiv preprint arXiv:2006.14171 .

Hügle, M., Kalweit, G., Werling, M., & Boedecker, J. (2020). Dynamic interaction-aware scene understanding for reinforcement learning in autonomous driving. In 2020 ieee international conference on robotics and automation (icra) (pp. 4329–4335).

Ibeas, A., Dell’Olio, L., Bordagaray, M., & Ortúzar, J. d. D. (2014). Modelling parking choices considering user heterogeneity. Transportation Research Part A: Policy and Practice, 70, 41–49.

Mariel, P., Hoyos, D., Meyerhof, J., Czajkowski, M., Dekker, T., Glenk, K., . . . others (2021). Environmental valuation with discrete choice experiments: Guidance on design, implementation and data analysis. Springer Nature.

McFadden, D. (1978). Modelling the choice of residential location. In Spatial interaction theory and planning models (pp. 75–96). North-Holland.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., . . . others (2015). Human-level control through deep reinforcement learning. nature, 518(7540), 529–533.

Mokhtarian, P. L. (2016). Discrete choice models’ ρ2: A reintroduction to an old friend. Journal of choice modelling, 21 , 60–65.

Nova, G., Hess, S., & van Cranenburgh, S. (2025). Delphos: A reinforcement learning framework for assisting discrete choice model specification. arXiv preprint arXiv:2506.06410.

Nova, G., van Cranenburgh, S., & Hess, S. (2025). Understanding the decision-making process of choice modellers. Journal of choice modelling, 56 , 100562.

Ortelli, N., Hillel, T., Pereira, F. C., de Lapparent, M., & Bierlaire, M. (2021). Assisted specification of discrete choice models. Journal of choice modelling, 39, 100285.

Páez, A., & Boisjoly, G. (2022). Discrete choice analysis with r. Springer.

Plaat, A. (2022). Deep reinforcement learning (Vol. 10). Springer.

Ramjerdi, F., Flügel, S., Samstad, H., & Killi, M. (2010). Value of time, safety and environment in passenger transport–time. TØI report B, 1053.

Rodrigues, F., Ortelli, N., Bierlaire, M., & Pereira, F. C. (2020). Bayesian automatic relevance determination for utility function specification in discrete choice models. IEEE Transactions on Intelligent Transportation Systems, 23 (4), 3126–3136.

Ross, S., & Bagnell, D. (2010). Eficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 661–668).

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.

Sfeir, G., Nova, G., Hess, S., & van Cranenburgh, S. (2025). Can large language models assist choice modelling? insights into prompting strategies and current models capabilities. arXiv preprint arXiv:2507.21790 .

Sodhani, S., Zhang, A., & Pineau, J. (2021). Multi-task reinforcement learning with context-based representations. In International conference on machine learning (pp. 9767–9779).

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. A Bradford Book.

Tsoleridis, P., Choudhury, C. F., & Hess, S. (2022). Deriving transport appraisal values from emerging revealed preference data. Transportation Research Part A: Policy and Practice, 165 , 225–245.

Van Cranenburgh, S., & Alwosheel, A. (2019). An artificial neural network based approach to investigate travellers’ decision rules. Transportation Research Part C: Emerging Technologies, 98, 152–166.

Van Cranenburgh, S., Wang, S., Vij, A., Pereira, F., & Walker, J. (2022). Choice modelling in the age of machine learning-discussion paper. Journal of choice modelling, 42, 100340.

Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8, 279–292.

Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017). Deep sets. Advances in neural information processing systems, 30.