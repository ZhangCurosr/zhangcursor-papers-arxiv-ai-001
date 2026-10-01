# Belief-Aware Multi-Agent Path Finding under Map Uncertainty

Viraj Parimi∗<sup>1</sup>, Shao-Hung Chan<sup>2</sup>, Han Zhang<sup>2</sup>, Jingkai Chen<sup>2</sup> and Brian Williams<sup>1</sup>

Abstract— Multi-Agent Path Finding (MAPF) aims to find collision-free paths for multiple agents in a shared environment. Classical MAPF assumes that all static obstacles are known in advance, but real-world environments can change unexpectedly due to fallen objects, spills, or other local disturbances. When such changes are spatially correlated, an observation can inform traversability estimates beyond the observed location. Prior approaches address uncertainty in traversability through contingent plans or replanning based on direct observations, but do not leverage this spatial dependence to infer the traversability of nearby unobserved locations. As a result, they cannot use one observation to anticipate nearby unobserved obstacles that may cause costly rerouting later. We focus on Belief-Aware MAPF, where map discrepancies are fixed during execution but initially unknown, and observations can be informative beyond the observed location. We propose Multi-Agent Gaussian Belief Inference for Coordination (MAGIC), a framework that updates a shared belief about traversability online based on agents’ observations. MAGIC uses a Gaussian Markov Random Field and Gaussian Belief Propagation to approximately infer traversability and construct detour-aware costs for standard MAPF planners. Our experiments on MAPF benchmarks show that MAGIC reduces the executed sum of costs compared to existing approaches on 96.3% of instances, across several planner families and teams of up to 800 agents, demonstrating its applicability to large-scale MAPF problems.

## I. INTRODUCTION

Multi-Agent Path Finding (MAPF) is the problem of finding a set of collision-free paths, one for each agent, from its start to its goal location in a shared environment [1]. It provides a common abstraction for coordinating robots in warehouses and other shared spaces [2]. Most MAPF planners assume a fixed and accurately known environment. In practice, the planner’s representation may not accurately reflect the environment at execution time. For example, during automated warehouse operations, a fallen pallet, a closed passage, or a spill may make nearby locations inaccessible and invalidate planned paths. In this work, we consider initially unknown map discrepancies that remain fixed during execution and are discovered through local observations. Planning must therefore account not only for interactions among agents, but also for uncertainty about which parts of the environment remain traversable.

Recent approaches have begun to relax the assumption that the environment is completely known. MAPF under obstacle uncertainty (MAPF-OU) constructs contingent plans that branch on future observations [3] but becomes computationally costly as uncertainty grows. MAPF with imperfect maps (MAPF-IM) instead interleaves planning and execution and replans as uncertain parts of the environment are directly observed [4]. However, certain environmental changes can exhibit spatial structure [5]. A blocked aisle or a displaced object may affect several nearby locations, so an observation at one location can provide information about the unobserved parts of the environment. This motivates an online MAPF framework that uses spatial dependence to infer traversability beyond directly observed locations.

We formulate this setting as Belief-Aware MAPF, where agents plan under a shared probabilistic belief over initial uncertain traversability and update that belief through agents observations. To address this problem, we propose Multi-Agent Gaussian belief Inference for Coordination (MAGIC), a framework that integrates spatial inference with online MAPF planning. MAGIC represents spatial correlations among uncertain locations with a Gaussian Markov Random Field (GMRF) and incorporates new observations from the agents using Gaussian Belief Propagation (GaBP) [6]. Each observation directly resolves the states of the observed locations while also updating beliefs about nearby unobserved locations. We combine inferred traversability with local edgebypass distances to construct detour-aware planning costs. This allows a planner to distinguish uncertain transitions with inexpensive alternatives from those that would require a substantial detour. We apply these traversal costs to MAPF planners and update them online as agents gather new observations, following an interleaved planning-and-execution pipeline. Fig. 1 shows the overall framework of MAGIC.

Our contributions are threefold:

We formulate Belief-Aware MAPF, a one-shot MAPF problem in a fixed environment where agents lack a priori knowledge of the traversability of some locations. We frame it as a planning problem under a shared probabilistic belief that captures the spatial dependence among uncertain locations and quantifies their traversability.

We propose MAGIC, a framework that leverages this shared probabilistic belief to construct detour-aware traversal costs, enabling MAPF planners to weigh the likelihood that a planned transition remains available against the cost of an alternative route if it does not.

We evaluate MAGIC across standard MAPF benchmark maps, planner families, and teams of up to 800 agents, measuring executed cost, success rate, and sensitivity to uncertainty and prior information. Compared with existing approaches, MAGIC achieves lower executed SoC on 96.3% of instances.

![](images/b86cc5280d448a2360e1d738ca68e86da83e62997829a4293da9188435746bcd.jpg)  
Fig. 1. Overview of MAGIC. White and black locations are known to be traversable and blocked, respectively. Other colors show traversability probabilities at uncertain locations. Filled circles mark the positions of agents, and crosshairs mark goals, with Agent 1 in pink and Agent 2 in blue. (a) Agents maintain a shared posterior over locations with uncertain traversability, with representative probabilities shown. (b) Solid borders mark directly observed locations, while dashed borders mark unobserved locations whose beliefs change through GaBP. (c) Updated beliefs and detour distances define traversal costs for a standard MAPF planner. Solid and thick dashed lines show the executed paths and planned paths, respectively. Under optimistic replanning, which treats uncertain locations as traversable, Agent 1 would instead be at the hollow marker, where it is about to observe the hidden obstacle marked by the red cross and replan. MAGIC instead routes it through the lower corridor, avoiding that later reroute.

## II. RELATED WORK

## A. MAPF Under Map Uncertainty and Partial Observability

MAPF-OU constructs contingent plans that branch on observations of initially unknown obstacles, with computational cost increasing as that uncertainty grows [3]. MAPF-IM instead interleaves planning and execution, replanning as direct observations revise assumed traversability [4]. Neither approach, however, uses spatial dependence to infer traversability of unobserved locations.

## B. Adaptive Guidance and Incremental Execution

Guidance graph methods steer multi-agent traffic through edge weights without replacing the underlying planner. One such method, Guidance Graph Optimization, optimizes these weights to improve lifelong MAPF throughput [7]. Such methods establish weighted graphs as a practical interface for influencing the routes produced by MAPF planners. However, their weights shape traffic rather than encode uncertainty in environmental traversability. Interleaved planning and execution is likewise well studied. Rolling-Horizon Collision Resolution repeatedly replans in lifelong settings while resolving collisions only within a bounded horizon [8]. Similarly, MAPF-IM defers distant conflicts that may become irrelevant after new observations [4]. These approaches show that repeated replanning can avoid committing computation to portions of a plan that may change before execution.

## C. Planning With Uncertain Traversability

Single-agent planning on uncertain graphs provides an important foundation for reasoning about decisions whose outcomes depend on unknown traversability. The Canadian Traveller Problem seeks policies that minimize expected travel cost when edge states are revealed during execution [9]. Such works show how uncertainty and the cost of recovery shape route selection, but for a single agent rather than for collision-free coordination among many.

Planning under uncertain traversability has also considered information sharing and the construction of alternative routes. Stadler et al. [10] build team policies from macroactions that combine movement, sensing, and waiting for information from other agents. Veys et al. [11] generate sparse probabilistic roadmaps that retain uncertain shortcuts and recovery paths supporting low-expected-cost policies. These methods reason about uncertain traversability through specialized policies or graph representations, rather than cost-based guidance for existing MAPF planners.

## D. Belief-Space Planning and Spatial Map Inference

Partially observable Markov Decision Processes reason jointly about actions, observations, and evolving beliefs. Online planners such as Partially Observable Monte Carlo Planning use sampling to limit the search over future actions and observations [12]. Multi-robot belief-space planning considers future observations and collaboration when selecting trajectories [13], but the joint belief and action spaces grow rapidly with the number of agents, making large-team planning computationally demanding.

Probabilistic mapping instead exploits spatial dependencies to infer occupancy at unobserved locations. Gaussian-Process occupancy maps use spatial correlations [5], while MRFMap models dependencies among occupancy variables using a Markov Random Field and Loopy Belief Propagation [14]. Related single-agent work updates beliefs over spatially correlated blockage states during planning [15]. These approaches demonstrate that spatial structure can make local observations informative about unobserved parts of the environment. Using such beliefs in scalable MAPF planners with explicit collision resolution remains less explored.

## III. BELIEF-AWARE MAPF

## A. Classical MAPF

A classical MAPF instance consists of an undirected graph $\mathcal { G } = ( \nu , \mathcal { E } )$ representing the environment and a set of n agents $\{ a _ { 1 } , \ldots , a _ { n } \}$ [1]. Vertices represent locations, and edges represent allowed moves. Each agent $a _ { i }$ has a start location $s _ { i } ~ \in ~ \mathcal { V }$ and a goal location $g _ { i } \in \mathcal { V } .$ . Time is discretized into timesteps. We write $q _ { i } ^ { t } \in \mathcal { V }$ for the location of agent $a _ { i }$ at timestep t, with $q _ { i } ^ { 0 } = s _ { i }$ . At each timestep, an agent either waits at its current location or moves to an adjacent location in G. A joint plan must avoid vertex and edge conflicts [1]. A vertex conflict occurs when $q _ { i } ^ { t } = q _ { j } ^ { t }$ for some timestep t and some $i \neq j ,$ and an edge conflict occurs when $q _ { i } ^ { t } \stackrel { \cdot } { = } q _ { j } ^ { t + 1 }$ and $q _ { i } ^ { t + 1 } = \stackrel { \cdot } { q _ { j } ^ { t } }$ for some $i \neq j$ . Let $T _ { i }$ denote the final arrival time after which agent $a _ { i }$ remains at $g _ { i }$ . The sum of costs (SoC) of a joint plan is $\textstyle \sum _ { i = 1 } ^ { n } T _ { i }$

## B. Fixed Environment with Unknown Obstacles

Belief-Aware MAPF extends classical MAPF by allowing some locations to have initially unknown traversability. Let $\mathcal { Z } \subseteq \mathcal { V }$ denote this initially uncertain set. Each $z \in { \mathcal { Z } }$ carries a binary random variable $X _ { z }$ , where $X _ { z } = 1$ indicates that z is traversable and $X _ { z } ~ = ~ 0$ that it is blocked. We write $\mathbf { X } = ( X _ { z } ) _ { z \in \mathcal { Z } }$ for the joint hidden traversability state and $\mathbf { x } = ( x _ { z } ) _ { z \in \mathcal { Z } }$ for a particular realization, i.e., the true traversability state of the uncertain locations. Every location in $\mathcal { V } \backslash \mathcal { Z }$ is known to be traversable. For these locations, we set $X _ { v } = 1$ and $x _ { v } = 1$ . Obstacles already represented in the initial graph specification are excluded from V altogether. We require $s _ { i } , g _ { i } \in \mathcal { V } \setminus \mathcal { Z }$ for all i, so starts and goals are never placed at uncertain locations. The true traversable graph induced by a particular realization x is denoted by $\mathcal { G } ^ { \star } ( \mathbf { x } ) = ( \mathcal { V } ^ { \star } ( \mathbf { x } ) , \mathcal { E } ^ { \star } ( \mathbf { x } ) )$ with

$$
\begin{array} { r l } & { \mathcal { V } ^ { \star } ( \mathbf { x } ) = \lbrace v \in \mathcal { V } \mid x _ { v } = 1 \rbrace } \\ & { \mathcal { E } ^ { \star } ( \mathbf { x } ) = \lbrace ( u , v ) \in \mathcal { E } \mid u , v \in \mathcal { V } ^ { \star } ( \mathbf { x } ) \rbrace } \end{array}\tag{1}
$$

Let $\mathcal { P }$ denote the environment distribution over X, with support restricted to realizations x for which $\mathcal G ^ { \star } ( \mathbf x )$ admits a collision-free solution. As part of the problem setup, a realization $\mathbf { x } \sim \mathcal { P }$ is sampled and remains fixed throughout execution. Agents know $\bar { z }$ and are given a prior distribution $p _ { 0 } ( \mathbf { X } )$ , but they do not know $\mathbf { x } .$ The planner’s prior $p _ { 0 }$ need not match the environment distribution $\mathcal { P } .$ We defer the specification of $p _ { 0 }$ to Sec. IV and the construction of $\mathcal { Z }$ to Sec. V.

## C. Observations and Online Objective

We assume that sensing is local and incurs no additional cost. Before each joint move, every agent noiselessly observes the states of uncertain locations adjacent to its current position, and immediately shares these observations with the team. Let $\mathcal { H } _ { t }$ denote all observations collected through timestep t, so every agent conditions on the same history. The shared posterior probability that a location $v \in \mathcal V$ is traversable is

$$
b _ { t } ( v ) = \operatorname* { P r } _ { p _ { 0 } } [ X _ { v } = 1 \mid \mathcal { H } _ { t } ]\tag{2}
$$

When $p _ { 0 }$ couples nearby locations, an observation generally shifts this posterior at locations that have not been observed yet. A solution is a centralized online policy $\pi$ that maps the current configuration $( q _ { 1 } ^ { t } , \ldots , q _ { n } ^ { t } )$ and the observation history $\mathcal { H } _ { t }$ to a joint action, eventually bringing every agent to its goal while avoiding vertex and edge conflicts. Executed moves must lie in $\mathcal G ^ { \star } ( \mathbf x )$ . Let $\begin{array} { r } { \mathrm { S o C } ( { \boldsymbol \pi } , { \bf x } ) = \sum _ { i = 1 } ^ { n } T _ { i } } \end{array}$ denote the executed SoC when policy π operates on $\mathcal G ^ { \star } ( \mathbf x )$ . We seek

$$
\pi ^ { \star } \in \arg \operatorname* { m i n } _ { \pi } \ \mathbb { E } _ { \mathbf { X } \sim \mathcal { P } } \left[ \operatorname { S o C } ( \pi , \mathbf { X } ) \right] .\tag{3}
$$

Computing $\pi ^ { \star }$ requires contingent reasoning over possible unknown location states and observation histories, and $\mathcal { P }$ is unknown to the planner. Even related single-agent routing problems with uncertain edge states are computationally hard [9], and the multi-agent setting additionally requires joint conflict resolution. For this paper, we restrict attention to one-shot MAPF with fixed start and goal positions and unknown location states.

## IV. MAGIC

Directly optimizing Eq. (3) would require knowledge of P and contingent reasoning over possible hidden states and future observation histories. MAGIC instead approximates this online decision problem by repeatedly solving a deterministic MAPF instance informed by the observations collected so far. At each update, we first infer a shared belief over the traversability of locations that remain unobserved. We then translate these beliefs into traversal costs that reflect the consequence of using uncertain parts of the environment, and provide the resulting weighted graph to a standard MAPF planner. The resulting plan is executed until new observations trigger another belief and planning update.

## A. Shared Traversability Belief

We model the shared belief over the hidden traversability state X using a GMRF over latent scores $\textbf { f } = \mathbf { \Psi } ( f _ { v } ) _ { v \in \mathcal { V } }$ where $f _ { v } \in \mathbb { R } .$ . Conditioned on f, we model the traversability state of each uncertain location $\textit { v } \in \textit { Z }$ independently as $X _ { v } \mid f _ { v } \sim \operatorname { B e r n o u l l i } ( \Phi ( f _ { v } ) )$ , where Φ denotes the standard normal cumulative distribution function. A larger latent score $f _ { v }$ indicates a higher probability that location v is traversable. This latent field has density

$$
p ( \mathbf { f } ) \propto \exp \left( - \frac { \lambda _ { 0 } } { 2 } \sum _ { v \in \mathcal { V } } ( f _ { v } - \mu _ { 0 } ) ^ { 2 } - \frac { \lambda _ { 1 } } { 2 } \sum _ { ( u , v ) \in \mathcal { E } } ( f _ { u } - f _ { v } ) ^ { 2 } \right)\tag{4}
$$

The first term anchors each latent score at the prior mean $\mu _ { 0 } .$ while the second penalizes differences between adjacent scores, thereby inducing spatial correlation. The parameters $\lambda _ { 0 } > 0$ and $\lambda _ { 1 } \geq 0$ control the strength of the prior anchoring and spatial coupling, respectively. Since locations in $\mathcal { V } \backslash \mathcal { Z }$ are known to be traversable, we do not infer their states. Instead, we fix their latent scores to a positive constant c, so they act as known traversable locations in the belief model. Together, the resulting GMRF and the Bernoulli probit model induce agents’ initial belief $p _ { 0 } ( \mathbf { X } )$ over the unknown traversability states of locations.

Let $\mathcal { Z } _ { t } \subseteq \mathcal { Z }$ denote the locations whose traversability remains unknown after observations at timestep t. When a location $z \in \mathcal { Z } _ { t }$ is observed by an agent, it is removed from $\mathcal { Z } _ { t }$ with its traversability belief $b _ { t } ( z )$ set to 1 if traversable and 0 otherwise. To incorporate this observation while retaining Gaussian inference, we approximate the effect of the binary observation by fixing $f _ { z } = + c$ when z is traversable and $f _ { z } ~ = ~ - c$ when it is blocked. Through the pairwise terms in Eq. (4), these fixed values update beliefs at nearby unobserved locations.

Let ${ \mathcal { O } } _ { t }$ contain the uncertain locations observed at timestep t, together with their observed states. We incorporate all observations in ${ \mathcal { O } } _ { t }$ into a single batch and use GaBP to estimate the latent marginals over $\mathcal { Z } _ { t }$ . Let $\mu _ { v } ^ { t }$ and $( \sigma _ { v } ^ { t } ) ^ { 2 }$ denote the resulting marginal mean and variance of $f _ { v } .$ For $v \in \mathcal Z _ { t }$ , integrating $\Phi ( f _ { v } )$ over the inferred Gaussian marginal gives the closed-form traversability estimate [16]

$$
b _ { t } ( v ) \approx \Phi \left( \frac { \mu _ { v } ^ { t } } { \sqrt { 1 + ( \sigma _ { v } ^ { t } ) ^ { 2 } } } \right)\tag{5}
$$

Observed locations retain their known binary beliefs, while $b _ { t } ( v ) ~ = ~ 1$ for $v \in \mathcal { V } \setminus \mathcal { Z }$ . These approximate estimates guide subsequent planning. For a fixed marginal mean, a large marginal variance shifts the estimate toward 0.5. Thus, poorly determined latent scores produce less confident traversability estimates.

After initial inference, we rerun GaBP only when new observations are obtained and reuse the previous estimates otherwise. We use message damping, which blends new and previous message parameters, to aid convergence [17]. Each update stops when the largest message residual falls below a tolerance or after K sweeps, using the final marginal estimates. At convergence, GaBP gives exact means for the clamped Gaussian surrogate, while variances may be approximate on graphs with cycles [6]. Each sweep is linear in the number of variables and pairwise terms processed, with known locations contributing fixed evidence.

## B. Detour-Aware Traversal Costs

The shared belief provides traversability estimates for individual locations. We next convert these estimates into traversal costs for use by a MAPF planner. Let $\begin{array} { r l } { \mathcal { G } _ { t } } & { { } = } \end{array}$ $( \nu _ { t } , \mathcal { E } _ { t } )$ denote the graph the planner searches at timestep t. It contains every location that has not yet been observed to be blocked. An observed blocked location is removed together with its incident edges, while locations in $\mathcal { Z } _ { t }$ remain available. Since unobserved locations are never removed, the realized traversable graph satisfies ${ \mathcal { G } } ^ { \star } ( \mathbf { x } ) \subseteq { \mathcal { G } } _ { t }$

We next map location beliefs to an edge-traversability estimate. For an edge $e = ( u , v ) \in \mathcal { E } _ { t }$ , we define

$$
\hat { b } _ { t } ( e ) = \operatorname* { m i n } \{ b _ { t } ( u ) , b _ { t } ( v ) \}\tag{6}
$$

The marginal traversability beliefs $b _ { t } ( u )$ and $b _ { t } ( v )$ do not determine the probability that both locations are traversable. Their product assumes independence, while computing the pairwise probability requires the joint posterior of the two latent variables. Instead, we use the smaller marginal as a lightweight edge-traversability estimate that only requires the location beliefs already produced by GaBP. For each edge $e = ( u , v ) \in \mathcal { E } _ { t }$ incident to a location in $\mathcal { Z } _ { t } ,$ we compute

$$
C _ { \mathrm { d e t } } ^ { t } ( e ) = d _ { \mathcal { G } _ { t } \backslash \{ e \} } ( u , v )\tag{7}
$$

where d denotes the unweighted single-agent shortest-path distance. Thus, $C _ { \mathrm { d e t } } ^ { t } ( e )$ is the length of the shortest edgebypass distance from u to v. This quantity depends only on the current graph and the edge, and is shared across the agents. If removing e disconnects u from v, we set $C _ { \mathrm { d e t } } ^ { t } ( e ) =$ |V|. Since any finite unweighted shortest path in $\mathcal { G } _ { t }$ does not need to revisit a location, its length is at most $| \nu | - 1$ . Thus, this value exceeds every finite detour length and provides a finite penalty when no detour exists.

Algorithm 1 MAGIC execution loop   
1: Initialize the planning graph from ${ \mathcal { G } } ,$ and the belief and   
costs from the prior $p _ { 0 }$   
2: $\mathbf { q } ^ { 0 }  ( s _ { 1 } , \ldots , s _ { n } ) , \mathcal { I }  \emptyset , t  0$ ▷ current joint plan   
3: while $\mathbf { q } ^ { t } \neq \mathbf { g }$ do   
4: $\mathcal { O } _ { t } \gets$ newly observed adjacent uncertain locations   
and their shared states   
5: if $\mathcal { O } _ { t } \neq \mathcal { O }$ then   
$_ { 6 : }$ Incorporate ${ \mathcal { O } } _ { t }$ and update the posterior with   
$\mathrm { G a B P }$   
7: Update $\mathcal { G } _ { t } ,$ detours, and $c _ { t }$   
8: $\mathcal { I }  \emptyset$ ▷ planner input changed   
9: end if   
10: if $\mathcal { I }$ has no valid next action then   
11: $\mathcal { I }  \mathbf { M A P F } ( \mathcal { G } _ { t } , c _ { t } , \mathbf { q } ^ { t } , \mathbf { g } )$   
12: if J has no valid next action then   
13: return Failure   
14: end if   
15: end if   
16: Execute the first joint action of $\mathcal { I }$   
17: Update $\mathbf { q } ^ { t + 1 }$   
18: Remove the executed joint action from $\mathcal { I }$   
19: $t \gets t + 1$   
20: end while

We combine the estimated traversability with the detour distance to define the detour-aware edge cost for each edge $e \in { \mathcal { E } } _ { t }$ incident to a location in $\mathcal { Z } _ { t }$

$$
c _ { t } ( e ) = \hat { b } _ { t } ( e ) \cdot 1 + \left( 1 - \hat { b } _ { t } ( e ) \right) C _ { \mathrm { d e t } } ^ { t } ( e )\tag{8}
$$

This cost interpolates between the unit traversal cost and the local detour distance. All remaining edges in $\mathcal { E } _ { t }$ have $\hat { b } _ { t } ( e ) = 1$ and therefore retain unit cost, so no detour needs to be computed for them. Wait actions also retain unit cost. An edge incident to an unobserved, uncertain location with a short alternative remains inexpensive, while an edge with a long detour between its endpoints receives a higher cost. Since both terms are measured in steps, the construction introduces no additional weighting parameter. Because $\hat { b } _ { t } ( e )$ is an edge-level estimate and $C _ { \mathrm { d e t } } ^ { t } ( e )$ considers only removal of $e , c _ { t } ( e )$ is a local detour-aware surrogate rather than the exact expected cost of future execution.

We cache each detour length along with the corresponding shortest detour path, when one exists. The detour lengths depend only on $\mathcal { G } _ { t }$ and not on the posterior beliefs, so a traversable observation leaves them unchanged. When an uncertain location is observed to be blocked, its removal invalidates only the cached detour paths that use one of the newly removed edges. We recompute invalidated detours and reuse surviving cached detour paths, which remain shortest because $\mathcal { G } _ { t }$ only loses edges and vertices. Without such reuse, computing all detours requires one unweighted shortest-path query for each edge incident to a location in $\mathcal { Z } _ { t }$

## C. Planning and Execution

The resulting weighted graph defines the deterministic MAPF instance solved at each replanning step. At each step, the MAPF planner receives the current configuration $\mathbf { q } ^ { t } = ( q _ { 1 } ^ { t } , \ldots , q _ { n } ^ { t } )$ , the goals $\mathbf { g } = ( g _ { 1 } , \ldots , g _ { n } )$ , the graph $\mathcal { G } _ { t } ,$ and the edge costs $c _ { t }$ . Search-based MAPF planners such as CBS [18] use $c _ { t }$ as edge costs to find paths for agents. MAPF planners such as PIBT [19] and LaCAM [20] use $c _ { t }$ to estimate the weighted distance-to-go for selecting candidate moves. In general, the same belief and cost model can be applied to different MAPF planners.

Algorithm 1 summarizes the execution loop. Let $\mathcal { I }$ denote the current joint plan, represented as an ordered sequence of joint actions returned by the underlying MAPF planner. Replanning is triggered by any new observation, not only by a blocked one, because even a traversable observation can update posterior beliefs and, in turn, the edge costs. A MAPF planner that returns a single joint action rather than a full joint plan is queried again at the next timestep. Execution terminates with failure when a time limit is reached.

A valid joint action consists of waits or adjacent moves that avoid known blocked locations and vertex and edge conflicts. Because adjacent uncertain locations are observed before moving, noiseless observations reveal any blocked destination before entry. With a sound planner, executed actions are therefore valid in $\mathcal G ^ { \star } ( \mathbf x )$

## V. EXPERIMENTAL EVALUATION

We organize our evaluations around three questions.

Q1 Does MAGIC improve execution under uncertain traversability?

Q2 Does the benefit persist across planner families and problem scales?

Q3 How sensitive is MAGIC to the spatial information encoded by the prior $p _ { 0 } ( \mathbf { X } ) \mathbf { \varPsi }$

## A. Experimental Setup

a) Benchmark maps: We evaluate on the standard MAPF benchmark [1] using maps spanning open, random, maze, room, warehouse, game, and city layouts, grouped into four scale tiers, as summarized in Table I.

b) Uncertain environment generation: We instantiate the environment distribution $\mathcal { P }$ using spatially structured hidden obstacles. We first sample uncertainty centers from the traversable locations of each benchmark map. For environment generation, let $d ( v )$ denote the unweighted distance in G from location v to its nearest uncertainty center. For an uncertainty radius R, locations other than starts and goals satisfying $d ( v ) ~ \leq ~ R$ form the uncertain set $\mathcal { Z } .$ . Before feasibility filtering, the hidden state of each $v \in { \mathcal { Z } }$ is sampled independently, conditioned on the sampled centers, with

$$
\operatorname* { P r } [ X _ { v } = 0 \mid { \mathrm { s a m p l e d c e n t e r s } } ] = \exp \left( - { \frac { d ( v ) ^ { 2 } } { 2 \ell ^ { 2 } } } \right)\tag{9}
$$

BENCHMARK MAPS AND TEAM SIZES n USED FOR EVALUATION.  
TABLE I
<table><tr><td>Tier</td><td>Maps</td><td>n</td></tr><tr><td>Small</td><td> $\mathrm { e m p t y } - 3 2 - 3 2 , \mathrm { r a n d o m } - 3 2 - 3 2 - 1 0 ,$   $\mathtt { m a z e - 3 2 - 3 2 - 4 , ~ r a n d o m - 3 2 - 3 2 - 2 0 , }$   $\mathtt { r o o m } - 3 2 - 3 2 - 4$ </td><td>10</td></tr><tr><td>Medium</td><td> $\mathrm { d e n } 3 1 2 \mathrm { d } , \mathrm { e m p t y } - 4 8 - 4 8 ,$   $\mathtt { r o o m } - 6 4 - 6 4 - 8 , \mathtt { r a n d o m } - 6 4 - 6 4 - 2 0$ </td><td>50</td></tr><tr><td>Large</td><td>warehouse-10-20-10-2-2,  $\mathtt { d e n 5 2 0 d , m a z e - 1 2 8 - 1 2 8 - 1 0 , }$ </td><td>200</td></tr><tr><td>Huge</td><td>lt_gallowstemplar_n  $\mathsf { w a r e h o u s e { - } } 2 0 \mathsf { - } 4 0 \mathsf { - } 1 0 \mathsf { - } 2 \mathsf { - } 2 .$  brc202d, Berlin_1_256,  $\mathtt { \odot \vec { r } \vec { z } \vec { 9 } 0 0 d }$ </td><td>800</td></tr></table>

![](images/ff427c4454706218b6534b7de2f287a29b82a80110b1a5895edf8544a2d7d9bc.jpg)  
Fig. 2. Illustration of the uncertainty parameters used in the experiments. Orange locations have initially uncertain traversability, and inset dark squares indicate locations that are blocked in the hidden realization. (a) Increasing ϕ increases the portion of the map whose traversability is initially unknown. (b) A higher $\rho$ corresponds to a larger fraction of uncertain locations being blocked.

where $\ell \ > \ 0$ controls how quickly blockage probability decreases with distance from an uncertainty center. We characterize each realization by the uncertain fraction $\begin{array} { r } { \phi = \frac { | { \mathcal Z } | } { | { \mathcal V } | } } \end{array}$ and blockage rate $\begin{array} { r } { \rho = \frac { | \{ v \in \mathcal { Z } : x _ { v } = 0 \} | } { | \mathcal { Z } | } } \end{array}$ . We choose the number of uncertainty centers and ℓ to target specified values of $\phi$ and $\rho .$ For settings where $\rho = 0$ , we retain Z but set every location in it to be traversable. Additionally, when $\phi = 0 .$ there are no uncertain locations and $\rho$ is not used. Fig. 2 illustrates the effect of these parameters. We retain only realizations for which ${ \mathcal G } ^ { \star } ( \mathbf { x } )$ admits a collision-free solution, and the accepted realization x remains fixed throughout execution. In the standard benchmark runs, agents know Z but not the uncertainty centers, the blockage probabilities in Eq. (9), or the realization x.

c) Experimental conditions: Unless varied, the generator targets an uncertain fraction $\phi = 0 . 2 0 \ : \mathrm { { z } }$ , a blockage rate $\rho = 0 . 3 0$ , and an uncertainty radius $R = 3$ . To answer $\mathbf { Q 1 }$ the small-tier experiments vary the target values of $\phi$ and $\rho$ from 0.00 to 0.30 in increments of 0.05. The small-tier team size study varies $n ~ \in ~ \{ 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ while holding the uncertainty parameters fixed. Each small-tier setting uses 25 benchmark scenarios per map and three independently sampled realizations x per scenario, yielding 375 instances across the five maps. Each medium-, large-, and huge-tier setting uses 25 scenarios per map and one realization per scenario, yielding 100 instances. To answer Q2, we evaluate n = 10, 50, 200, and 800 agents on the small-, medium-, large-, and huge-tiers, respectively.

![](images/a5abbd65fa8d49cf535beedaf8d3959398adf802b8d142f8578cfbc2a4d9d60e.jpg)  
Fig. 3. Executed SoC and success rate under uncertain traversability. MAGIC and optimistic replanning use CBS with $n = 1 0$ in panels (a) and (b), and PBS in (c). Cost ratios use optimal full-information SoC obtained by CBS on $\mathcal { G } ^ { \star } ( \mathbf { \bar { x } } )$ as the reference in (a,b), and optimistic replanning in (c). Panel (a) varies ϕ and (b) varies $\rho .$ Each cost curve uses instances completed by the method and its reference, whereas success rates include all attempts. Shading denotes 95% confidence intervals. MAPF-IM has no successful runs at $n = 1 0 0 .$ . Lower executed SoC and higher success rate are better.

d) Planners and comparisons: We compare MAGIC with location-adapted MAPF-IM and with optimistic replanning. The MAPF-IM comparison evaluates MAGIC against the closest prior approach to our setting. Optimistic replanning uses the same realization $\mathbf { x } ,$ start-goal assignment, and underlying MAPF planner as MAGIC, but treats every unresolved location $v \in \mathcal Z _ { t }$ as traversable with unit edge costs. This comparison evaluates belief-guided planning together with its observation-triggered replanning policy. For planners that maintain a joint plan, optimistic replanning replans when a newly observed location is blocked, whereas MAGIC replans after every new observation. PIBT, which returns a single joint action, is queried at each timestep.

The small-tier experiments use CBS, EECBS with suboptimality bounds $w = 1 . 0 5$ and $w = 1 . 1 0 [ 2 1 ]$ , and PBS [22]. The medium- and large-tiers use PBS, PIBT, PIBT+ [19], and LaCAM. The huge-tier uses PIBT, PIBT+, and LaCAM. For each small-tier instance, we also run CBS with full knowledge of $\mathcal G ^ { \star } ( \mathbf x )$ . When the full-information CBS run succeeds, its optimal SoC provides a common lower bound for the planners evaluated on that instance. For the medium-, large-, and huge-tiers, our primary cost comparison is instead between MAGIC and optimistic replanning using the same underlying planner. We additionally compare against MAPF-IM adapted to location-based uncertainty [4]. During local conflict resolution, its search adds a fixed penalty to the distance estimate for transitions incident to $\mathcal { Z } _ { t } .$ , rather than using $b _ { t } ( v )$ maintained by MAGIC.

e) Inference and limits: Unless otherwise stated, we set $\mu _ { 0 } = 0 , \lambda _ { 0 } = \lambda _ { 1 } = 1$ , and $c = 2$ for MAGIC. For GaBP, we use a message damping factor of 0.5, a residual tolerance of $1 0 ^ { - 8 }$ , and a maximum of $K \ : = \ : 2 0 0$ sweeps per belief update. The small-, medium-, large-, and hugetiers use per-call planner time limits of 60, 120, 120, 300 s, total runtime limits of 300, 3600, 3600, 14400 s, and execution limits of 500, 3000, 3000, 12000 timesteps, respectively. Methods compared within the same experimental condition are subject to the same limits. All experiments were run on a workstation with an Intel Core i9-14900K CPU, 32 logical CPUs, and 62 GiB of usable memory. Independent runs were executed in parallel.

f) Evaluation metrics: We report the executed SoC and the success rate separately. Cost ratios are computed per instance and averaged over instances completed by both the method and its stated reference. Each figure or table specifies that reference. Success rates include all attempted instances, counting runs that fail to complete within the stated limits or violate traversability or collision constraints as failures. We additionally report 95% confidence intervals.

## B. Results and Discussion

a) Q1: Fig. 3 shows how executed SoC and success rate vary with $\phi , \rho ,$ and n. As the uncertain fraction ϕ increases, optimistic replanning moves farther from the full-information optimum. $\mathbf { A } \mathbf { t } ~ \phi = 0$ , MAGIC and optimistic replanning both recover the optimal full-information SoC because $\mathcal { Z } = \mathcal { D } . \mathrm { A t }$ $\phi = 0 . 3 0$ , optimistic replanning is 9.4% above the optimum, whereas MAGIC is $3 . 6 \%$ above it. MAPF-IM is 14.9% above the optimum at the same setting. Thus, as more locations are uncertain to traverse, MAGIC recovers a substantial portion of the execution-cost gap between optimistic replanning and

TABLE II  
PERFORMANCE ACROSS PLANNER FAMILIES AND PROBLEM SCALES AT ϕ = 0.20, ρ = 0.30, AND R = 3.
<table><tr><td></td><td colspan="2">Small (n = 10)</td><td colspan="2">Medium  $( n = 5 0 )$ </td><td colspan="2">Large (n = 200)</td><td colspan="2">Huge  $( n = 8 0 0 )$ </td></tr><tr><td>Planner</td><td>SoC ratio ↓</td><td>Success ↑</td><td>SoC ratio ↓</td><td>Success ↑</td><td>SoC ratio ↓</td><td>Success ↑</td><td>SoC ratio ↓</td><td>Success ↑</td></tr><tr><td>CBS</td><td> $0 . 9 6 6 \pm 0 . 0 0 4$ </td><td>91.2/88.8</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EECBS  $( w = 1 . 0 5 )$ </td><td> $0 . 9 6 7 \pm 0 . 0 0 5$ </td><td>96.8/94.9</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EECBS (w = 1.10)</td><td> $0 . 9 6 9 \pm 0 . 0 0 5$ </td><td>97.6/96.5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PBS</td><td> $0 . 9 6 3 \pm 0 . 0 0 6$ </td><td>100.0/99.7</td><td> $0 . 9 8 7 \pm 0 . 0 0 4$ </td><td>100/99</td><td> $0 . 9 9 0 \pm 0 . 0 0 2$ </td><td>95/74</td><td></td><td></td></tr><tr><td>LaCAM</td><td></td><td></td><td> $0 . 9 8 6 \pm 0 . 0 0 4$ </td><td>100/100</td><td> $0 . 9 8 9 \pm 0 . 0 0 1$ </td><td>100/100</td><td> $0 . 9 9 5 \pm 0 . 0 0 1$ </td><td>100/94</td></tr><tr><td>PIBT+</td><td></td><td></td><td> $0 . 9 9 5 \pm 0 . 0 0 9$ </td><td>100/100</td><td> $0 . 9 9 1 \pm 0 . 0 0 3$ </td><td>100/100</td><td> $0 . 9 9 5 \pm 0 . 0 0 1$ </td><td>100/100</td></tr><tr><td>PIBT</td><td></td><td></td><td> $0 . 9 9 1 \pm 0 . 0 0 8$ </td><td>96/96</td><td> $0 . 9 9 2 \pm 0 . 0 0 3$ </td><td>95/94</td><td> $0 . 9 9 3 \pm 0 . 0 0 2$ </td><td>79/55</td></tr><tr><td>MAPF-IM†</td><td> $1 . 0 5 2 \pm 0 . 0 0 9$ </td><td>91.2/98.4</td><td> $1 . 1 0 3 \pm 0 . 0 1 0$ </td><td>100/66</td><td> $1 . 2 0 9 \pm 0 . 0 2 3$ </td><td>100/11</td><td></td><td></td></tr></table>

the full-information optimum.

The blockage rate sweep shows when the additional caution introduced by MAGIC becomes useful. At $\rho = 0 ,$ , every location in $\mathcal { Z }$ is traversable in the realized environment, so optimistic replanning matches the full-information optimum. MAGIC is 2.9% above the optimum because it still penalizes transitions involving unresolved locations. As the blockage rate increases, this cost is offset by avoiding routes through locations that are more likely to be blocked. At the largest tested ρ, MAGIC is 2.7% above the optimum, compared with 6.7% for optimistic replanning and 12.4% for MAPF-IM.

The executed SoC advantage of MAGIC decreases as n increases. This trend is consistent with two effects. Larger teams can gather observations in parallel, shortening the period during which inference about $\mathcal { Z } _ { t }$ can influence route selection, while increased coordination can also leave fewer alternative routes around uncertain locations. Accordingly, MAGIC and optimistic replanning approach parity as n grows, while MAPF-IM exhibits a sharper decline in success and has no successful run at $n = 1 0 0$

b) Q2: Table II evaluates Q2 across the planner families and four scale tiers. For each planner and tier, the table reports the mean per-instance executed-SoC ratio and reference/method success rates over all attempts. Ratios below one favor the evaluated method and above one favor the reference. For each non-daggered row, the SoC ratio compares MAGIC with optimistic replanning using the same underlying planner. The ± value reports the wider side of the 95% bootstrap intervals. For MAPF-IM†, the reference is optimistic replanning with CBS for the small tier and with LaCAM for the other tiers.

Across the small-, medium-, and large-tiers, MAGIC achieves lower executed SoC than MAPF-IM on 96.3%. All 15 MAGIC configurations have mean ratios below one. The smaller gains in the larger tiers are consistent with the trend discussed in Q1, but maps, team size, and resource limits also differ across tiers. ${ \mathrm { A t ~ } } n = 8 0 0 .$ , PIBT+ retains 100% success under both methods with a mean SoC ratio of 0.995. The success rate results show that this cost benefit does not imply uniform reliability among planners. MAGIC solves the same underlying instances but performs additional GaBP inference and belief-dependent planning, while planners that return complete plans may also fully replan after traversable observations. This more demanding online workload can cause difficult runs to exhaust planner or total-runtime limits before all agents reach their goals. Thus, the lower success rates in some settings primarily reflect a computational tradeoff. MAPF-IM exhibits a different tradeoff. Its higher success rate on the small-tier is consistent with its impactdetection and localized conflict resolution, but its success rate decreases substantially at larger scales.

![](images/d0ce242cb223e7e7adbe59bae677a1a3371686f7ad7d949e87952d9ebfcf8d38.jpg)

![](images/905ae2f7a571fec0cc9c70a2463d8872a4542a7f6525984fa6a2cf3eb8c200dc.jpg)  
Fig. 4. Sensitivity to the spatial allocation of the initial prior (a) Illustration of reversed, uniform, and aligned prior traversability probabilities. A higher prior blockage probability corresponds to a lower initial traversability belief. (b) Executed SoC relative to optimistic replanning as the prior allocation varies from reversed to aligned. (c) Success rate over all attempted instances. Shading denotes 95% confidence intervals. Lower executed SoC and higher success rates are better.

We additionally varied the uncertainty radius R and found little aggregate sensitivity to it. On the large warehouse map, however, MAGIC incurred a higher executed SoC than optimistic replanning at R = 1, but for $R \ \geq \ 2$ the trend reversed. This suggests that very localized uncertainty may provide too little spatial evidence to offset conservative detours in narrow aisles, whereas larger regions of uncertainty make neighboring observations more informative.

Further, medium- and large-tier sweeps over $\phi$ and ρ broadly reproduced similar trends.

c) Q3: Q1 and Q2 use the same prior model and parameter settings. To answer Q3, we now vary which locations in Z receive higher or lower prior traversability probabilities. Using the same small-tier instances at $\phi = 0 . 2 0 , \rho = 0 . 3 0$ and $R \ = \ 3 ,$ we construct five initial priors indexed by $\alpha \in \{ - 1 , - 0 . 5 , 0 , 0 . 5 , 1 \}$ . The aligned prior $\alpha = 1$ assigns each location a prior traversability probability consistent with its probability under the environment generator. The reversed prior $\alpha = - 1$ uses the same probability values but assigns them in reverse order, so locations that are more likely to be traversable under the generator receive lower prior traversability probabilities, and vice versa. At $\alpha = 0 ,$ every location in Z receives the same average traversability probability. The intermediate settings $\alpha = \pm 0 . 5$ shift the corresponding aligned or reversed probabilities by half toward this average. To obtain these probabilities, we replace the shared mean $\mu _ { 0 }$ in Eq. (4) with location-specific prior means chosen to produce the corresponding initial traversability probabilities, while keeping the other settings unchanged. Fig. 4(a) illustrates these effects.

Fig. 4(b) shows a consistent reduction in executed SoC as the spatial allocation becomes better aligned with the generating probability profile. When comparing the aligned and reversed priors directly, the aligned prior reduces executed SoC by 3.40% for successful instances. The success rate results show a similar, though small, trend. It increases from 88.5% under the reversed prior to 90.9% under the aligned prior, compared with 91.2% for the optimistic replanning on the same benchmark set. The aligned prior does not improve every individual instance. Still, its aggregate advantage is strongest on maps with constrained routes such as rooms, where assigning a low prior traversability probability to useful passages can make them artificially expensive. The result, therefore, shows that informative spatial structure in p<sub>0</sub>(X) can improve execution.

d) Physical robot demonstration: We also demonstrate MAGIC on a team of two TurtleBots in a fixed indoor environment. Planning observations are restricted to adjacent locations to match the observation model. In the demonstration, one robot observes a blocked, uncertain location, which updates the shared belief at a nearby location that remains unobserved by either robot. This causes the second robot to change its route before directly observing that location. A supplemental video shows the complete executions together with additional qualitative examples.

## VI. CONCLUSION

We formulated Belief-Aware MAPF for fixed environments with initially uncertain traversability and introduced MAGIC, which updates a shared spatial belief from local observations and converts inferred traversability into detouraware costs for standard MAPF planners. Across different tiers, MAGIC achieves lower executed SoC than existing approaches on 96.3% of instances. The benefit persists across several planner families and scales to teams of 800 agents.

We further find that execution depends on how the prior $p _ { 0 } ( \mathbf { X } )$ assigns traversability probabilities across Z. Future work will extend Belief-Aware MAPF to lifelong tasks and environments whose traversability changes during execution.

## REFERENCES

[1] R. Stern, N. R. Sturtevant, A. Felner, S. Koenig, H. Ma, T. T. Walker, J. Li, D. Atzmon, L. Cohen, T. K. S. Kumar, R. Bartak,´ and E. Boyarski, “Multi-agent pathfinding: Definitions, variants, and benchmarks,” in Proc. Symp. Combin. Search (SoCS), 2019, pp. 151– 158.

[2] P. R. Wurman, R. D’Andrea, and M. Mountz, “Coordinating hundreds of cooperative, autonomous vehicles in warehouses,” AI Mag., vol. 29, no. 1, pp. 9–19, 2008.

[3] B. Shofer, G. Shani, and R. Stern, “Multi agent path finding under obstacle uncertainty,” in Proc. Int. Conf. Autom. Plan. Sched. (ICAPS), vol. 33, 2023, pp. 402–410.

[4] N. Malka, G. Shani, and R. Stern, “Online planning for multi agent path finding in inaccurate maps,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2024, pp. 10 214–10 221.

[5] S. T. O’Callaghan and F. T. Ramos, “Gaussian process occupancy maps,” Int. J. Robot. Res., vol. 31, no. 1, pp. 42–62, 2012.

[6] Y. Weiss and W. T. Freeman, “Correctness of belief propagation in Gaussian graphical models of arbitrary topology,” Neural Comput., vol. 13, no. 10, pp. 2173–2200, 2001.

[7] Y. Zhang, H. Jiang, V. Bhatt, S. Nikolaidis, and J. Li, “Guidance graph optimization for lifelong multi-agent path finding,” in Proc. Int. Joint Conf. Artif. Intell. (IJCAI), 2024, pp. 311–320.

[8] J. Li, A. Tinka, S. Kiesel, J. W. Durham, T. K. S. Kumar, and S. Koenig, “Lifelong multi-agent path finding in large-scale warehouses,” in Proc. AAAI Conf. Artif. Intell., vol. 35, no. 13, 2021, pp. 11 272–11 281.

[9] E. Nikolova and D. R. Karger, “Route planning under uncertainty: The Canadian Traveller Problem,” in Proc. AAAI Conf. Artif. Intell., 2008, pp. 969–974.

[10] M. Stadler, J. Banfi, and N. Roy, “Approximating the value of collaborative team actions for efficient multiagent navigation in uncertain graphs,” in Proc. Int. Conf. Autom. Plan. Sched. (ICAPS), vol. 33, no. 1, 2023, pp. 677–685.

[11] Y. Veys, M. S. Kurtz, and N. Roy, “Generating sparse probabilistic graphs for efficient planning in uncertain environments,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 133–139.

[12] D. Silver and J. Veness, “Monte-Carlo planning in large POMDPs,” in Adv. Neural Inf. Process. Syst., vol. 23, 2010, pp. 2164–2172.

[13] T. Regev and V. Indelman, “Multi-robot decentralized belief space planning in unknown environments via efficient re-evaluation of impacted paths,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2016, pp. 5591–5598.

[14] K. S. Shankar and N. Michael, “MRFMap: Online probabilistic 3D mapping using forward ray sensor models,” in Proc. Robotics: Science and Systems (RSS), 2020.

[15] L. Zhou and E. Ceyhan, “Stochastic path planning in correlated obstacle fields,” 2025, arXiv:2509.19559.

[16] C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning. The MIT Press, 2006.

[17] Q. Su and Y.-C. Wu, “On convergence conditions of Gaussian belief propagation,” IEEE Trans. Signal Process., vol. 63, no. 5, pp. 1144– 1155, 2015.

[18] G. Sharon, R. Stern, A. Felner, and N. R. Sturtevant, “Conflict-based search for optimal multi-agent pathfinding,” Artif. Intell., vol. 219, pp. 40–66, 2015.

[19] K. Okumura, M. Machida, X. Defago, and Y. Tamura, “Priority´ inheritance with backtracking for iterative multi-agent path finding,” Artif. Intell., vol. 310, p. 103752, 2022.

[20] K. Okumura, “LaCAM: Search-based algorithm for quick multi-agent pathfinding,” in Proc. AAAI Conf. Artif. Intell., vol. 37, no. 10, 2023, pp. 11 655–11 662.

[21] J. Li, W. Ruml, and S. Koenig, “EECBS: A bounded-suboptimal search for multi-agent path finding,” in Proc. AAAI Conf. Artif. Intell., vol. 35, no. 14, 2021, pp. 12 353–12 362.

[22] H. Ma, D. Harabor, P. J. Stuckey, J. Li, and S. Koenig, “Searching with consistent prioritization for multi-agent path finding,” in Proc. AAAI Conf. Artif. Intell., vol. 33, no. 1, 2019, pp. 7643–7650.