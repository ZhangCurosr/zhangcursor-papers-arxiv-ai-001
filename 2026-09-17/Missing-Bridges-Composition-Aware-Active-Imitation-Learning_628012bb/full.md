# Missing Bridges: Composition-Aware Active Imitation Learning

Maxwell J. Jacobson, Ahmed H Qureshi, Yexiang Xue

Department of Computer Science, Purdue University

West Lafayette, Indiana, USA

jacobs57@purdue.edu, qureshi7@purdue.edu, yexiang@purdue.edu

## Abstract

Active imitation learning reduces expert efort by allowing a learner to request the demonstrations it needs. Existing methods typically select these requests for their expected information gain about the expert policy. In structured multi-task domains, however, the number ofstart–goal tasks may grow combinatorially despite their solutions sharing reusable behavior. This makes composable behaviors especially valuable, since a single demonstration may help solve many tasks at once. Prior methods do not explicitly account for this value when selecting which demonstration to request. We introduce Adaptive Agents via Latent Topologies (AALT), which requests demonstrations that maximize expected gains in start–goal connectivity. We further show that this objective is formally tied to information gain about task reachability. AALT organizes existing demonstrations into a topology of latent hub states connected by learned behaviors, identifies high-value bridge demonstrations that are likely to enable many tasks at once, and grounds each to an expert query. At inference, it plans through the resulting topology and conditions a difusion policy on each successive hub transition. In a simulated UR5e robot ordered-retrieval domain with 72 tasks, AALT improved from 42/72 to 72/72 (100%) successful tasks consistently using only 3 demonstrations totaling 5 transitions beyond the initial dataset. After 20 demonstrations, the strongest baseline averaged 88.6% success using 98 transitions.

## Introduction

We address goal-based active imitation learning over large, structured sets of start–goal tasks. The learner begins with an incomplete demonstration dataset and may request additional expert demonstrations under a limited budget. Each start– goal task pairs an initial state with a requested goal, and the number of tasks can grow combinatorially as more start states and goals are introduced. Consider a robotic retrieval task in which a robot arm fetches part canisters from a shelf to fulfill work orders (see Fig. 1). Diferent shelf organizations and misplaced canisters produce diferent starts, while each work order defines a goal by specifying which canisters to retrieve and in what order. Combining these starts and goals yields many tasks, yet historical demonstrations may cover only a few, such as shelves already arranged for expected work orders. Demonstrating every remaining pair would therefore require efort that grows with the task space.

Existing active imitation learning methods often select demonstrations according to how much they are expected to improve learning of the expert policy. We refer to this broad family as policy-centric active IL. These methods provide a strong means to direct limited expert efort toward the most informative behavior. Active Multi-task Fine-tuning (AMF) is a representative example, selecting complete start–goal tasks by their expected information gain about the expert policy (Bagatella et al. 2025).

Existing active imitation learning objectives do not explicitly value the compositional structure learned from a demonstration. This distinction matters when many start– goal tasks share portions of their solutions. A short demonstration may connect two previously demonstrated behaviors and thereby enable many unsupported start–goal tasks with just one “bridge”, even if it provides relatively little new information about the expert policy. In our robot example, a policy-centric method might prioritize an uncertain maneuver useful for only a few work orders over one that connects many starts to already learned retrieval behaviors. The expert may therefore be asked for many additional demonstrations whose solutions substantially overlap.

We introduce Adaptive Agents via Latent Topologies (AALT), a composition-aware active imitation learning method that selects demonstrations according to their expected increase in start-to-goal connectivity. This objective is formally connected to information gain about task reachability rather than the expert policy. We prove that this information gain factorizes into AALT’s connectivity gain and uncertainty about bridge acquisition when acquired behaviors are treated as perfectly reliable, with a corresponding lower-bound relationship when execution may fail.

AALT encodes demonstrated states into a learned latent space and identifies hub states where demonstrated trajectories converge or diverge. Demonstrated behavior between hub states forms directed edges in a latent topology. The topology represents which behaviors are currently available, how they can be composed, and which missing bridges prevent additional start–goal tasks from being supported. In the robotic retrieval example, a hub may correspond to a shelf state reached after retrieving the first canister shared by three work orders, with demonstrated continuations for each order’s remaining canisters. A single bridge from another reachable shelf state to this hub can therefore unlock all three work orders.

![](images/26dde59cff3fb00ee2709904510b192ceb6c992aafb594a4e912da8df1ed3f7d.jpg)  
Figure 1: AALT requests demonstrations for composability. An agent begins with learned behaviors (black arrows) covering only some tasks between start states ( S ) and goal states ( A , B , C ), and may request additional expert demonstrations (orange arrows). (Left) Standard policy-centric active IL selects requests for their expected information about the expert policy. Because it does not explicitly value composability, it may request separate full-task demonstrations from S to A , B , and C , each of which may help the agent understand only one task. (Center) AALT instead organizes demonstrated behaviors into a topology connected through reusable latent hub states (blue), then requests short bridge demonstrations that maximize expected start–goal connectivity. Here, known behaviors reach h<sub>2</sub> , while other known behaviors lead from h<sub>1</sub> to A , B , and C . Requesting the bridge $\textcircled { h _ { 2 } } \textcircled { \div } \textcircled { h _ { 1 } }$ creates routes to all three goals with one short demonstration. During execution, the topology specifies a route through these hubs, while a learned policy generates actions from the current observations for each successive hub transition. (Right) In the simulated retrieval task, the bridge moves the black canister to transform h<sub>2</sub> into h<sub>1</sub> , from which the policy can pursue all three goals. Video of the simulation is provided in the supplementary material.

AALT estimates the execution reliability of each demonstrated edge and measures support for a start–goal task using the reliability of its best path through the topology. It then evaluates absent edges by how much adding each one would increase expected start–goal connectivity over the target task distribution. This objective is essentially a surrogate for reachability information gain – or how much information a bridge query reveals about which target tasks will be reachable after the query. The highest-value edge is requested as a bridge demonstration. In the robotic retrieval example, this might be requesting a demonstration from a reliably reachable shelf configuration to one with strongly demonstrated continuations to goal states. After a request, AALT adds the demonstration to the dataset, updates the shared policy and edge reliabilities, and recomputes the value of the remaining bridge candidates. Acquisition stops when the expert budget is exhausted or no bridge provides enough expected gain.

At inference, AALT matches the current state to the latent topology and plans a reliable path to a hub state satisfying the requested goal. A single difusion policy (Chi et al. 2023) executes this path one edge at a time, conditioned on the current and next hub states. The topology supplies compositional guidance, while the difusion policy learns the low-level behavior across all edges.

We evaluate AALT in a simulated UR5e retrieval environment containing 72 start–goal tasks, where a robot must retrieve three ordered part canisters from a shelf without disturbing obstructing objects. AALT improves from 42/72 tasks to $\bar { 1 } 0 0 . 0 \pm 0 . 0 \bar { \% }$ using three demonstrations totaling five transitions, versus 88. $6 \pm 9 . 9 \%$ for the strongest baseline using 20 demonstrations totaling 98.0 ± 9.8 transitions. Our baselines include policy-centric AMF (Bagatella et al. 2025), which requests complete task demonstrations (46.9 ± 2.7%); a variant given the same topology and bridge candidates as AALT $( 8 \bar { 2 } . 5 \pm 4 . 0 \% ) $ ; and uniform random selection from those bridges $( 8 8 . 6 \pm 9 . 9 \% )$ ). AALT’s three bridges enable 12, 10, and 8 additional tasks and are reused by 18, 10, and 8 final task routes. Failure analysis further shows that topologyenabled AMF more often fails because no route was acquired $( 8 . 4 \pm 5 . 9$ tasks) than because an available route fails during execution (4.2±4.3), consistent with policy information gain leaving important connections unrequested.

## Problem Definition

We consider goal-based active imitation learning over large, structured families of start–goal tasks. The agent must reach diferent requested goals from many possible initial states, but begins with demonstrations for only some of these combinations. It may then ask an expert to demonstrate how to move between selected source and destination states, with the aim of using a limited number of queries to improve performance across the full task family. Let S, A, and G denote the state, action, and goal spaces, and let ${ \cal S } _ { 0 } \subseteq { \cal S }$ denote the possible initial states. Each task is a pair $( s _ { 0 } , g ) \in S _ { 0 } \times \mathcal { G }$ drawn from a target distribution $\rho .$ The agent receives the state as input, and must output actions which change the state such that it eventually reaches a state compatible with the goal (call this set $\boldsymbol { \mathcal { S } } _ { g } )$

The agent initially receives an incomplete expert dataset $\mathcal { D } _ { 0 } = \{ \bar { \tau _ { i } } \} _ { i = 1 } ^ { N }$ , where each trajectory contains states, actions, its intended goal, and a success indicator. The agent may use this dataset in a pre-training phase. This is followed by the acquisition phase, where the agent may create a query and receive an expert demonstration to add to the dataset for more training. Given a budget of B queries, the objective is to maximize expected task success under $\rho ,$ including for start–goal pairs absent from $\mathcal { D } _ { 0 }$

Although the task space may be large or combinatorial, its solutions need not be independent: tasks may share prefixes, sufixes, or intermediate behaviors. Demonstrating each complete task separately may therefore repeat substantial expert efort. We target settings where such compositional structure may exist but is not given to the agent.

## Adaptive Agents via Latent Topologies (AALT)

AALT seeks to eficiently expand the set of tasks an agent can solve from limited expert demonstrations. It does this by requesting missing bridge demonstrations that maximize start–goal connectivity. AALT has three components: acquisition, pre-training, and inference. Acquisition operates on a latent topology, an abstract map of reusable states and the demonstrated behaviors connecting them. Its hub states represent points where demonstrations begin, end, converge, or diverge, while its directed edges represent behaviors that move between those hubs. AALT evaluates missing edges in this topology and requests the bridge demonstration expected to produce the greatest increase in start–goal connectivity. During pre-training, AALT learns the topology from the initial dataset, along with a matcher for associating observations with hubs and a shared difusion policy for executing its edges. During inference, AALT matches the current state to the topology, plans a route to a hub satisfying the requested goal, and executes the route one edge at a time using the shared policy.

In the example in Fig. 1, suppose hub $h _ { 1 }$ represents a configuration in which the center canisters have been moved aside, exposing the pink, white, and blue canisters. Existing behaviors from $h _ { 1 }$ can then complete several work orders, such as retrieving $p i n k ~  ~ w h i t e ~  ~ b l u e$ , or $b l u e  p i n k  w h i t e .$ . However, the initial topology may contain no route to $h _ { 1 } .$ , even though it can already reach a nearby hub $h _ { 2 }$ . During acquisition, AALT evaluates a bridge from $h _ { 2 } \tan h _ { 1 }$ according to how many task routes it would create. If this bridge has the greatest expected gain and the expert demonstrates it successfully, AALT adds the demonstration to the topology and updates the policy. The bridge might simply move the remaining black canister out of the way. At inference, the agent follows existing edges to $h _ { 2 } .$ , executes the acquired bridge to $h _ { 1 }$ , and then uses the existing outgoing behaviors to complete the requested work order. One short demonstration can make several known goal-reaching behaviors available from new starts.

## Acquisition: Selecting Expert Queries to Gain Task Connectivity

Given a learned topology $\Gamma = ( \mathcal { H } , \mathcal { E } ) , \mathrm { A A L T }$ selects additional demonstrations that improve connectivity between the target start and goal states. Each directed edge $e \in { \mathcal { E } }$ represents a demonstrated behavior and has an estimated reliability $r _ { e } \in [ 0 , 1 ]$ . Acquisition evaluates missing edges according to how much their addition would increase reliable start–goal connectivity over the target task distribution.

Once we have this topology, AALT uses it to decide which additional demonstration would be most useful. The central idea is to request a missing behavior that connects existing parts of the topology, allowing already demonstrated behaviors to be composed into solutions for additional tasks. To make this decision, AALT first estimates how reliably each existing edge can be executed, then measures how well the current topology supports the target task distribution, and finally evaluates how much each possible new edge would improve that support.

Not every demonstrated behavior can be executed equally reliably. An edge that repeatedly succeeds should contribute more confidence to a composed solution than one that often fails. AALT therefore maintains an execution-success estimate for each edge $e .$ We represent uncertainty about its execution-success probability with $\mathrm { ~ a ~ } \operatorname { B e t a } ( \alpha _ { e } , \beta _ { e } )$ distribution and use the posterior mean $r _ { e } = \alpha _ { e } / ( \alpha _ { e } + \beta _ { e } )$ as the edge reliability. Every edge begins with a shared prior $\mathrm { B e t a } ( \alpha _ { 0 } , \beta _ { 0 } )$ and corresponding initial reliability $r _ { 0 } =$ $\alpha _ { 0 } / ( \alpha _ { 0 } + \beta _ { 0 } )$ . After observing an execution outcome $y _ { e , k } ^ { \mathrm { e x e c } } \in$ {0, 1}, where $y _ { e , k } ^ { \mathrm { e x e c } } = 1$ denotes success, AALT updates $\alpha _ { e }  \alpha _ { e } + y _ { e , k } ^ { \mathrm { e x e c } }$ and $\beta _ { e }  \beta _ { e } + ( 1 - y _ { e , k } ^ { \mathrm { e x e c } } )$ . The reliability estimate therefore increases with successful executions and decreases with failures. All experiments use this soft-reliability form, reflecting that learned behaviors may succeed only some of the time. For the theoretical analysis in the Theory section, we also consider a binary-reliability version of AALT in which acquired edges are treated as perfectly reliable $( r _ { e } \in \{ 0 , 1 \}$ for every edge, including initial reliability $r _ { 0 } )$

A start–goal task is supported when the topology contains a path from the hub matching its initial state to a terminal hub satisfying its goal. Because a path may compose several learned behaviors, its reliability is estimated by multiplying the reliabilities of its edges. When several paths are available, AALT uses the most reliable one. Formally,

$$
R _ { \Gamma } ( s _ { 0 } , g ) = \operatorname* { m a x } _ { h _ { g } \in \mathcal { H } _ { g } } \prod _ { e \in P } r _ { e } ,\tag{1}
$$

where $\boldsymbol { \Gamma } = ( \mathcal { H } , \mathcal { E } )$ is the current topology, $h _ { s }$ is the hub matched to $s _ { 0 } , \mathcal { H } _ { g }$ is the set of terminal hubs satisfying $^ { g , }$ and $P : h _ { s }  h _ { q }$ is a directed path between them. If no such path exists, then $R _ { \Gamma } ( s _ { 0 } , g ) = 0$

AALT summarizes the support provided by the entire topology by averaging this value over the target task distribution. Its soft connectivity is

$$
C ( \Gamma ) = \mathbb { E } _ { ( s _ { 0 } , g ) \sim \rho } \left[ R _ { \Gamma } ( s _ { 0 } , g ) \right] .\tag{2}
$$

![](images/8cd01a5a9f285c34f62f6c6e0a33b2c40d454029fe0a66e38d74cee11d52756f.jpg)  
Figure 2: The AALT method. (Left) latent states are clustered, and may become hub states if they converge or diverge between diferent demos. (Center) hubs are linked by demos to form a topology of composable behaviors. New demonstration requests are selected to optimize start–goal connectivity. Here, $h _ { 3 }  h _ { 1 }$ would connect the right-side starts to the left side goals. (Right) during inference, we plan from the start state to a goal hub. A difusion policy generates actions to reach the goal, conditioned on $( h _ { i } , h _ { j } )$ from each edge.

Thus, C(Γ) is the expected reliability of the best available solution for a task drawn from $\rho .$ Tasks with greater probability under $\rho$ contribute more strongly to the acquisition objective.

AALT next forms demonstration queries for behaviors that are currently absent from the topology. Each candidate is a directed edge between two existing hubs that are not already connected, giving $\mathcal { E } _ { \mathrm { c a n d } } \subseteq ( \mathcal { H } \times \breve { \mathcal { H } } ) \backslash \mathcal { E } .$ . These queries must be grounded in concrete states available in the demonstration dataset. The expert is therefore shown a source state associated with the first hub and a destination state associated with the second hub, rather than being asked to interpret the latent representations themselves.

To evaluate a candidate edge, AALT temporarily adds it to the topology with the initial reliability $r _ { 0 }$ and measures how much the resulting topology would improve expected task connectivity. Let $\mathbf { \breve { r } } + \mathbf { \breve { e } } = ( \varkappa , \mathcal { E } \cup \{ \dot { e } \} )$ denote this hypothetical topology. The value of candidate e is

$$
\Delta ( e ) = C ( \Gamma + e ) - C ( \Gamma ) .\tag{3}
$$

Because C(Γ) averages over the complete target distribution, a short bridge can receive high value when it completes useful paths for many start–goal tasks, even when the demonstrated behavior itself is brief.

AALT requests the candidate with the greatest expected gain, $e ^ { * } = \arg \operatorname* { m a x } _ { e \in { \mathcal { E } } _ { \mathrm { c a n d } } } \Delta ( e )$ . If the expert returns a feasible trajectory connecting its grounded source and destination states, the demonstration is added to the training dataset and the corresponding edge is inserted into the topology. If no feasible trajectory is returned, no edge is added, and the candidate is removed from further consideration. After each successful query, AALT updates the learned behavior and recomputes the values of the remaining candidates using the expanded topology. Acquisition stops when the query budget B is exhausted, no candidate remains, or the best remaining candidate satisfies $\Delta ( e ^ { * } ) < \delta$

This greedy rule maximizes the expected value of the next query, but need not maximize connectivity after the entire query budget. Given a model of query outcomes, exhaustive lookahead over the remaining budget could recover the optimal acquisition policy under that model, while a shorter receding-horizon lookahead or beam search could provide a more computationally practical approximation. We use onestep greedy selection in this work.

## Pre-training: Learning a Latent Topology, Policy, & Matcher from Initial Dataset

To construct the topology assumed above, AALT identifies reusable states where demonstrated trajectories begin, end, converge, or diverge. Here, we borrow the concepts of convergence hubs and divergence hubs (Jacobson and Xue 2026). Convergence hubs are states reached by multiple distinct preceding trajectories (like a blocking canister being moved aside from any starting shelf), while divergence hubs are states from which trajectories branch into multiple distinct continuations (like a canister used in multiple work orders being accessed). Both are visualized in the left panel of Figure 2.

A useful topology must organize states by what the agent can do from them, rather than by visual similarity alone. Two state images may look mostly similar but support different actions, while diferent-looking states may support the same next behavior. We therefore shape the latent space around the efects of actions. Following a common pattern in latent world models (Ha and Schmidhuber 2018; Hafner et al. 2019, 2020), an encoder maps each observation to a compact state, an action-conditioned dynamics model predicts how that state changes, and a decoder preserves the observation information needed to make these predictions. Formally, $\mathrm { E n c } _ { \theta _ { \mathrm { e n c } } } ( s _ { t } )  z _ { t } , M _ { \theta _ { \mathrm { d y n } } } ( z _ { t } , a _ { t } )  \hat { z } _ { t + 1 }$ , and $\mathrm { D e c } _ { \theta _ { \mathrm { d e c } } } ( \hat { z } _ { t + 1 } )  \hat { s } _ { t + 1 }$ . AALT additionally trains an inverse dynamics model to predict which action connected two consecutive states, $\operatorname { I n v } _ { \theta _ { \mathrm { i n v } } } ( z _ { t } , z _ { t + 1 } )  \hat { a } _ { t }$ . The forward model encourages the encoder to distinguish states with diferent action-conditioned outcomes, while the inverse model requires consecutive state embeddings to retain enough information to identify the action between them.

AALT constructs the topology by joining pairs of demonstrated state embeddings satisfying $\| z _ { i } - z _ { j } \| _ { \infty } \leq \epsilon$ and taking the connected components of the resulting graph as state clusters, each represented by its mean embedding. Clusters occurring at demonstration starts or terminal states become hubs, as do clusters with multiple distinct predecessors or successors. Consecutive hub visits then define the directed edges in E. To match new observations during execution, AALT trains a symmetric binary matcher $\mathrm { M } \mathrm { \bar { a } t c h } _ { \theta _ { \mathrm { m a t c h } } } ( z _ { i } , z _ { j } ) \in [ 0 , 1 ]$ from the cluster identities and augmented observations, accepting matches above a threshold η. AALT also trains a categorical difusion policy on the demonstrated action segments between consecutive hubs. The policy reconstructs masked pick-and-place action sequences conditioned on the current image, recent observation history, and source and target hub embeddings. We write $\pi _ { \theta _ { \pi } } ( \mathbf { a } _ { t : t + K - 1 } \ | \ s _ { t }$ , history<sub>t</sub>, z¯<sub>h</sub> , z¯<sub>h</sub> ), where the segment connects $h _ { i } \ \mathrm { t o } \ h _ { j }$ and K is the prediction horizon.

## Inference: Composing Behaviors to Solve Tasks

At inference, AALT first grounds the requested task in the learned topology – it encodes the current observation and uses the learned matcher to identify a corresponding start hub. The requested goal determines the set of terminal hubs whose demonstrated states satisfy that goal. AALT then applies Dijkstra’s algorithm to find a route from the start hub to any goal-satisfying hub, assigning each edge e the cost $- \log r _ { e }$ . The resulting minimum-cost route is the one with the greatest product of edge reliabilities. Equivalently, this favors routes that remain reliable across all of their constituent behaviors rather than routes containing a particularly unreliable edge. If the current state cannot be matched to a hub or no route reaches a goal-satisfying hub, the task is not currently supported by the topology. AALT can warn the user, or select the closest state.

AALT executes the selected route one edge at a time using the difusion policy. For each current edge $( h _ { i } , h _ { j } )$ , the policy is conditioned on the latest observation, recent observation history, and the latent representations of its source and target hubs. The topology therefore specifies which intermediate state should be reached next, while the policy generates the low-level actions needed to reach it. After each action, AALT encodes the new observation and uses the matcher to determine whether the target hub $h _ { j }$ has been reached. Once it has, AALT advances to the next edge and conditions the same policy on the next hub pair. Otherwise, it continues acting toward the current target from the updated observation. Execution succeeds when a goal-satisfying terminal hub is reached. Each attempted edge is recorded as successful if its target hub is reached and as failed otherwise, and these outcomes update its reliability between episodes.

## AALT as a Surrogate for Reachability Information Gain

AALT uses connectivity gain (Eqn 3) as a surrogate for weighted reachability information gain. This quantity measures how much a bridge query reveals about which target tasks will be reachable after the query, weighted by their probability under the target distribution.

Definition 1 (Weighted reachability information gain). At acquisition round $b ,$ consider a candidate bridge e. Let $Y _ { e } ^ { \mathrm { a c q } } \in \{ 0 , 1 \}$ indicate whether the expert successfully provides the bridge, and let $Z _ { x }$ denote the resulting reachability of target task x: $Z _ { x } = R _ { \Gamma _ { b } + e } ( x )$ if $Y _ { e } ^ { \mathrm { a c q } } = 1$ , and $Z _ { x } \ \dot { = } \ R _ { \Gamma _ { b } } \bar { ( } x )$ otherwise. Holding the learned hub set H fixed, the weighted reachability information gain of e is

$$
\mathrm { R I G } _ { \rho } ( e ) = \sum _ { x \in \mathrm { s u p p } ( \rho ) } \rho ( x ) I ( Z _ { x } ; Y _ { e } ^ { \mathrm { a c q } } \mid \mathcal { D } _ { b } , \mathcal { H } ) .\tag{4}
$$

Here, $\mathcal { D } _ { b }$ is the information available before the query, and the mutual information measures how much observing its outcome resolves uncertainty about the resulting reachability of task x. Weighting by $\rho ( x )$ gives greater importance to tasks that occur more often under the target distribution. In Fig. 1, acquiring the bridge from $h _ { 2 }$ to $h _ { 1 }$ determines the reachability of the three tasks from start $S$ to goals $A , B ,$ and C. The query is informative about all three because its outcome determines whether the known route from $S$ to $h _ { 2 }$ can be composed with the ones from $h _ { 1 }$ to each goal.

To relate this quantity to $\mathbf { A A L T \vec { s } }$ score, let $p _ { e } = \mathbb { P } ( Y _ { e } ^ { \mathrm { a c q } } =$ $1 \mid \mathcal { D } _ { b } , \mathcal { H } )$ denote the learner’s pre-query probability that the expert can provide a feasible demonstration for bridge e. Let $\mathrm { E n t } ( p _ { e } ) \stackrel { - } { = } - p _ { e } \log p _ { e } - ( 1 - p _ { e } ) \log ( 1 - p _ { e } )$ denote the corresponding uncertainty. The following result separates $\mathrm { R I G } _ { \rho } ( \boldsymbol { e } )$ into the value created when acquisition succeeds and the uncertainty over whether it will succeed.

Proposition 1 (Binary-reliability factorization). For binaryreliability AALT, in which every acquired edge is treated as perfectly reliable,

$$
\mathrm { R I G } _ { \rho } ( e ) = \mathrm { E n t } ( p _ { e } ) \Delta ( e ) .\tag{5}
$$

Under binary reliability, every task whose reachability depends on the query outcome changes from unreachable to reachable when acquisition succeeds. Its post-query reachability therefore reveals the acquisition outcome exactly. A task with the same reachability under success and failure reveals nothing about that outcome. Because every changed task gains exactly one, their total target-distribution weight is precisely AALT’s connectivity gain $\Delta ( e )$

The factorization separates two values of the query. Connectivity gain measures what successful acquisition would add to the agent’s capabilities, while the entropy term measures uncertainty about whether that acquisition will occur. In Fig. 1, connectivity gain values the new routes to $A , B ,$ and $C$ created when the bridge from $h _ { 2 }$ to $h _ { 1 }$ is supplied. Learning that this bridge cannot be supplied is informative, but it creates none of those routes. AALT therefore optimizes the successful-acquisition factor because only that outcome directly increases the set of tasks the agent can solve. $\mathsf { A p - }$ pendix A proves Proposition 1.

Proposition 2 (Soft-reliability bound). For soft-reliability AALT,

$$
\begin{array} { r } { \mathrm { E n t } ( p _ { e } ) \Delta ( e ) \le \mathrm { R I G } _ { \rho } ( e ) . } \end{array}\tag{6}
$$

The bound is exact whenever every task whose reachability changes movesfrom zero to one.

With soft reliability, acquiring a bridge may improve a task without making its solution perfectly reliable. Reachability information gain counts the task’s full weight whenever its reachability difers between query success and failure, regardless of the size of that diference. Connectivity gain instead scales the task’s weight by the magnitude of its improvement. It therefore gives less value to a bridge that only slightly improves a task and more directly reflects the expected increase in task-solving capability. For example, if the bridge from $h _ { 2 }$ to $h _ { 1 }$ makes the three tasks leading to A, B, and C fully reachable, each contributes its full task weight and the bound becomes exact. Proof in Appendix A.

Together, these results specify the sense in which connectivity gain is a surrogate for reachability information gain. It is exactly the successful-acquisition factor under binary reliability, and its entropy-scaled value lower-bounds reachability information gain under soft reliability. This is a pointwise relationship rather than a guarantee that the two objectives rank all bridges identically when their acquisition uncertainties difer. AALT isolates the component that directly measures added task-solving capability and can be evaluated from the learned topology.

## Related Work

Active & Imitation Learning. Active learning reduces annotation cost by allowing a learner to select the examples for which supervision is expected to be most valuable (Settles 2009; Ren et al. 2021; Zhan et al. 2022; Li et al. 2024). Imitation learning similarly seeks to learn behavior from expert demonstrations rather than through direct reward optimization (Hussein et al. 2017; Osa et al. 2018; Zare et al. 2024; Correia and Alexandre 2024). Active imitation learning combines these ideas by selectively requesting demonstrations for chosen portions of a task, ranging from individual decisions or state-to-state segments to complete task executions (Ross, Gordon, and Bagnell 2011; Judah et al. 2014). Most such methods are policy-centric, selecting demonstrations expected to provide the most information about the expert policy, often in states where the learner is likely to disagree with the expert or fail (Zhang and Cho 2017; Hoque et al. 2022; Silver, Bagnell, and Stentz 2012). Active Multi-task Fine-tuning (AMF), for example, selects complete start–goal tasks according to their expected information gain about a shared expert policy (Bagatella et al. 2025).

Latent Models & Compositionality. Latent representations provide compact encodings of high-dimensional modalities such as images or text, allowing models to predict, reconstruct, or plan within a lower-dimensional space (Ha and Schmidhuber 2018; Hafner et al. 2019, 2020; Hu et al. 2022). Latent variables can also represent temporally extended behaviors, enabling long-horizon tasks to be composed from reusable behavioral units. SeCTAR learns latent trajectory representations for hierarchical planning over temporally extended behaviors (Co-Reyes et al. 2018), while SkiMojointly learns latent skills and skill-level dynamics for long-horizon skill composition (Shi, Lim, and Lee 2023). Most directly inspiring this work, ZALT constructs a topology over latent hub states from a fixed demonstration dataset, learns policies over its edges, and composes hub-to-hub behaviors to solve unseen start–goal tasks zero-shot (Jacobson and Xue 2026). AALT utilizes a similar latent behavioral topology in active imitation learning, using the topology to identify demonstrations whose missing edges provide the greatest increase in start–goal connectivity.

## Experiments

## Setup

Existing active IL objectives do not explicitly reward demonstrations for the new tasks they enable through composition; AALT instead requests short missing behaviors that maximize start–goal connectivity. We have designed this simulated robot experiment to evaluate this.

Toolshelf domain. We evaluate in a simulated UR5e robot arm servicing cell where the robot retrieves ordered sets of 3 colored canisters for assembly and maintenance work orders. Movable blockers control access to diferent shelf regions, so tasks may require rearrangement before retrieval. Agents receive top-down RGB observations and an ordered three-canister goal vector and operate over 465 discrete pickand-place actions. During execution, all methods receive the same environment-provided action mask, which removes invalid pick-and-place actions. More information on this environment can be found in Appendix B.

Tasks and initial data. Each task pairs a shelf layout (start) with an ordered three-canister work order (goal). The six layouts fall into three access modes in which movable guards leave the left, center, or right shelf region open. The twelve work orders are likewise divided into three families of four goals whose requested canisters are stored primarily in one of these regions. Crossing all six layouts with all twelve work orders produces 72 possible tasks. The 24 initial demonstrations contain 196 transitions and model historical operation in which each shelf was already staged for the corresponding work-order family. Thus, every layout and every work order appears in the initial data, but 48 cross-family combinations are omitted. Each method receives only the demonstrations and has no prior information about work-order families.

Compared methods. For this experiment, AALT constructs its latent topology from the 24 initial demonstrations. Acquisition stops when the best remaining gain falls below the fixed threshold δ = 0.08. We compare AALT with two variants of Active Multi-task Fine-tuning (AMF) and a random bridge-selection baseline. AMF selects the demonstration expected to provide the most information about the shared policy. Intuitively, it favors queries that reduce uncertainty about which actions the expert would take. AMF-Full follows the original setting and selects among complete start–goal task demonstrations. AMF-Bridge instead selects among the same grounded bridge queries available to AALT, but ranks them using AMF information gain. Random-Bridge samples uniformly from the same set of admissible grounded bridge queries. It tests whether improvements result from AALT’s connectivity-based acquisition objective rather than simply from allowing bridge demonstrations. AALT, AMF-Bridge, and Random-Bridge use the same learned topology and controller, difering only in how they select queries. All methods use categorical difusion policies with the same architecture, initial demonstrations, expert, accumulated replay, and environment-provided action mask during execution. Additional baseline details are provided in Appendix C.

Expert and query grounding. The expert uses A\* search over the symbolic shelf state (symbolic states are only available to the expert, not the evaluated methods). Full-task queries request a plan from a task’s initial state to its ordered retrieval goal. Bridge queries instead provide concrete source and destination states associated with two hubs and request a trajectory connecting them. A query is unsuccessful when no feasible trajectory is found.

![](images/7f306098fd36853d21317248d44a0a268a96bda307ca1d6a9020c7fedf6ddf3b.jpg)

![](images/fa8199c015c0a8b9c67d2442c2a11b5ca9075ab826a1eb1f7456bcc77aac8dac.jpg)  
Figure 3: AALT solves all 72 tasks after only three expert calls totaling five transitions, whereas after 20 calls, the most efective baseline Random-Bridge reaches 88.6±9.9% using $9 8 . 0 \pm 9 . 8 $ transitions. (a) Target-task success under increasing expert-call budgets (↑). (b) Cumulative expert transitions requested (↓).

Learning protocol. All methods are trained on the same initial dataset before acquisition. After each successful query, the returned demonstration is added to the accumulated dataset and the shared policy is updated before the next evaluation. Failed queries provide no policy-training trajectory. Models remain fixed during each rollout, and no evaluation experience is added to the policy dataset.

## Results

Figure 3(a) reports task success and Figure 3(b) reports cumulative expert transitions, with means and sample standard deviations over five adaptation seeds. AALT begins by solving 42 of the 72 tasks, then improves to 54 after its first query, 64 after its second, and 72 after its third. These demonstrations contain two, one, and two transitions, respectively, for five total. The best remaining connectivity gain then falls below the fixed threshold $\delta = 0 . 0 8$ , so acquisition stops. Random-Bridge reaches $8 8 . 6 \pm 9 . 9 \%$ success after 20 demonstrations containing $9 8 . 0 \pm 9 . 8 $ transitions. AMF-

Bridge reaches $8 2 . 5 \pm 4 . 0 \%$ using $1 1 8 . 6 \pm 1 1 . 3$ transitions across 20 demonstrations. AMF-Full begins at 33 successful tasks and reaches $4 6 . 9 \pm 2 . 7 \%$ after 20 demonstrations containing $1 7 1 . 6 { \pm } 1 2 . 4 $ transitions. AALT therefore achieves the highest success while requesting the fewest demonstrations and expert transitions.

The final failures of the bridge methods separate into missing routes and failures to execute available routes. AMF-Bridge ends with $8 . 4 \pm 5 . 9$ tasks lacking a route and $4 . 2 \pm 4 . 3$ tasks for which a route exists but the policy fails to complete it. Random-Bridge ends with $3 . 8 \pm 5 . 5$ tasks lacking a route and $4 . 4 \pm 2 . 7$ unsuccessful route executions. AALT has neither type of failure and never loses a previously successful task after a policy update. By the final evaluation, AMF-Bridge has lost $3 . 0 \pm 2 . 4$ of its 42 initial successes, Random-Bridge has lost $2 . 6 \pm 2 . 6 $ , and AMF-Full has lost $8 . 0 \pm 1 . ( $ 0 of its 33 initial successes. AMF-Full gains $8 . 8 \pm 1 . 5$ previously unsuccessful tasks but loses eight initial successes, leaving a net improvement of only $0 . 8 \pm 1 . 9$ tasks.

The acquired AALT bridges are reused across many solutions. The three queries immediately enable 12, 10, and 8 additional tasks and appear in 18, 10, and 8 final task routes, respectively. Every one of the 30 newly solved tasks uses at least one acquired bridge, and six compose two of them. Across seeds, 67 of the 100 AMF-Bridge queries lead directly to terminal hubs. These demonstrations average 7.94 transitions and are followed by a net loss of 19 solved tasks, whereas its 21 queries into start-class hubs average 1.43 transitions and are followed by a net gain of 98. AMF-Full repeats an already queried task in 49 of its 100 calls, consuming 47.8% of its expert transitions on repeated demonstrations.

These results show that access to bridge queries alone does not explain AALT’s performance. Random-Bridge samples from the same candidate set and eventually discovers several useful access bridges, but it finds them late and inconsistently. AMF-Bridge also uses the same bridge format, but its policy-information objective often selects longer demonstrations that support fewer tasks. AMF-Full further shows that complete-task acquisition can spend substantial expert efort repeating individual tasks while producing little improvement over the full task distribution. AALT instead identifies three short missing connections that make existing behaviors available to many additional start–goal tasks. Its gains therefore follow from selecting bridges according to the connectivity they create, not just the bridge format by itself.

## Conclusion

This work has introduced AALT, a composition-aware active imitation learning method that requests missing bridge demonstrations according to the start–goal connectivity they create. AALT expanded from 42/72 to 72/72 tasks using just three expert demonstrations. It optimizes connectivity gain (a surrogate for information gain about task reachability). We discuss the scope and limitations of our method, including its reliance on hub identification and assumptions on the setting, in Appendix E.

## References

Bagatella, M.; Hübotter, J.; Martius, G.; and Krause, A. 2025. Active Fine-Tuning of Multi-Task Policies. In Singh, A.; Fazel, M.; Hsu, D.; Lacoste-Julien, S.; Berkenkamp, F.; Maharaj, T.; Wagstaf, K.; and Zhu, J., eds., Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, 2409–2441. PMLR.

Chi, C.; Feng, S.; Du, Y.; Xu, Z.; Cousineau, E.; Burchfiel, B. C.; and Song, S. 2023. Difusion Policy: Visuomotor Policy Learning via Action Difusion. In Proceedings of Robotics: Science and Systems. Daegu, Republic of Korea.

Co-Reyes, J.; Liu, Y.; Gupta, A.; Eysenbach, B.; Abbeel, P.; and Levine, S. 2018. Self-Consistent Trajectory Autoencoder: Hierarchical Reinforcement Learning with Trajectory Embeddings. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, 1009–1018. PMLR.

Correia, A.; and Alexandre, L. A. 2024. A survey of demonstration learning. Robotics and Autonomous Systems, 182: 104812.

Ha, D.; and Schmidhuber, J. 2018. Recurrent World Models Facilitate Policy Evolution. In Advances in Neural Information Processing Systems, volume 31, 2450–2462.

Hafner, D.; Lillicrap, T.; Ba, J.; and Norouzi, M. 2020. Dream to Control: Learning Behaviors by Latent Imagination. In International Conference on Learning Representations.

Hafner, D.; Lillicrap, T.; Fischer, I.; Villegas, R.; Ha, D.; Lee, H.; and Davidson, J. 2019. Learning Latent Dynamics for Planning from Pixels. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, 2555–2565. PMLR.

Hoque, R.; Balakrishna, A.; Novoseller, E.; Wilcox, A.; Brown, D. S.; and Goldberg, K. 2022. ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning. In Proceedings ofthe 5th Conference on Robot Learning, volume 164 of Proceedings ofMachine Learning Research, 598–608. PMLR.

Hu, A.; Corrado, G.; Grifiths, N.; Murez, Z.; Gurau, C.; Yeo, H.; Kendall, A.; Cipolla, R.; and Shotton, J. 2022. Model-Based Imitation Learning for Urban Driving. In Advances in Neural Information Processing Systems, volume 35, 20703– 20716.

Hussein, A.; Gaber, M. M.; Elyan, E.; and Jayne, C. 2017. Imitation learning: A survey of learning methods. ACM Computing Surveys (CSUR), 50(2): 1–35.

Jacobson, M. J.; and Xue, Y. 2026. Zero-shot Imitation Learning by Latent Topology Mapping. arXiv preprint arXiv:2605.08450.

Judah, K.; Fern, A. P.; Dietterich, T. G.; and Tadepalli, P. 2014. Active Imitation Learning: Formal and Practical Reductions to I.I.D. Learning. Journal of Machine Learning Research, 15(120): 4105–4143.

Li, D.; Wang, Z.; Chen, Y.; Jiang, R.; Ding, W.; and Okumura, M. 2024. A survey on deep active learning: Recent advances

and new frontiers. IEEE Transactions on Neural Networks and Learning Systems, 36(4): 5879–5899.

Osa, T.; Pajarinen, J.; Neumann, G.; Bagnell, J. A.; Abbeel, P.; and Peters, J. 2018. An algorithmic perspective on imitation learning. Foundations and Trends® in Robotics, 7(1-2): 1–179.

Ren, P.; Xiao, Y.; Chang, X.; Huang, P.-Y.; Li, Z.; Gupta, B. B.; Chen, X.; and Wang, X. 2021. A survey of deep active learning. ACM computing surveys (CSUR), 54(9): 1–40.

Ross, S.; Gordon, G. J.; and Bagnell, J. A. 2011. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, 627–635. PMLR.

Settles, B. 2009. Active learning literature survey.

Shi, L. X.; Lim, J. J.; and Lee, Y. 2023. Skill-Based Model-Based Reinforcement Learning. In Proceedings of the 6th Conference on Robot Learning, volume 205 of Proceedings ofMachine Learning Research, 2262–2272. PMLR.

Silver, D.; Bagnell, J. A.; and Stentz, A. 2012. Active Learning from Demonstration for Robust Autonomous Navigation. In 2012 IEEE International Conference on Robotics and Automation, 200–207. IEEE.

Zare, M.; Kebria, P. M.; Khosravi, A.; and Nahavandi, S. 2024. A survey of imitation learning: Algorithms, recent developments, and challenges. IEEE Transactions on Cybernetics, 54(12): 7173–7186.

Zhan, X.; Wang, Q.; Huang, K.-h.; Xiong, H.; Dou, D.; and Chan, A. B. 2022. A comparative survey of deep active learning. arXiv preprint arXiv:2203.13450.

Zhang, J.; and Cho, K. 2017. Query-Eficient Imitation Learning for End-to-End Simulated Driving. In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, volume 31, 2891–2897.

## A. Proofs for AALT as a Surrogate for Reachability Information Gain

This appendix proves the binary-reliability factorization in Proposition 1 and the soft-reliability bound in Proposition 2. Intuitively, we show that AALT’s connectivity gain is a surrogate of information gain about task reachability. In a binaryreliability version of AALT, weighted reachability information gain factorizes exactly into AALT’s connectivity gain and the uncertainty about whether the queried bridge can be acquired. For soft reliabilities, we derive a lower-bound relationship and characterize the resulting gap. We first formalize the acquisition setting and derive two auxiliary results relating weighted reachability information gain to AALT’s connectivity gain. We then restate and prove the two propositions from the main text.

Definition A.1 (Fixed-hub acquisition setting). The analysis applies to AALT’s acquisition phase after the latent hub set H has been learned from the initial dataset $\mathcal { D } _ { 0 }$ . Pretraining produces the initial topology $\Gamma _ { 0 } = ( \mathcal { H } , \mathcal { E } _ { 0 } )$ . Let $b \in \{ 0 , \ldots , B \}$ denote the number of completed queries. After b queries, the topology is

$$
\Gamma _ { b } = ( \mathcal { H } , \mathcal { E } _ { b } ) , \qquad \mathcal { E } _ { 0 } \subseteq \mathcal { E } _ { b } \subseteq \mathcal { H } \times \mathcal { H } .
$$

The hub set H remains fixed throughout acquisition, while successful queries add edges to $\mathcal { E } _ { b }$ . The information available after the first b queries is denoted by $\mathcal { D } _ { b } .$ , including failed query outcomes that return no trajectory for policy training.

At acquisition round $b ,$ let $\mathcal { E } _ { \mathrm { c a n d } , b }$ be a finite subset of the missing directed edges $( { \mathcal { H } } \times { \mathcal { H } } ) \setminus { \mathcal { E } } _ { b }$ . For a candidate edge $e = ( \bar { h _ { i } } , h _ { j } )$ , let $Y _ { e } ^ { \mathrm { { \bar { a } c q } } } \in \{ 0 , 1 \}$ denote the query outcome. The outcome is one if the expert returns a feasible trajectory connecting $h _ { i }$ to $h _ { j }$ and zero otherwise. The resulting edge set is

$$
\mathcal { E } _ { b + 1 } = \left\{ \begin{array} { l l } { \mathcal { E } _ { b } \cup \{ e \} , } & { Y _ { e } ^ { \mathrm { a c q } } = 1 , } \\ { \mathcal { E } _ { b } , } & { Y _ { e } ^ { \mathrm { a c q } } = 0 . } \end{array} \right.
$$

The learner’s pre-query probability that the expert can provide a feasible demonstration for e is

$$
p _ { e } = \mathbb { P } ( Y _ { e } ^ { \mathrm { a c q } } = 1 \mid \mathcal { D } _ { b } , \mathcal { H } ) .
$$

For $p \in [ 0 , 1 ]$ , binary entropy is

$$
\operatorname { E n t } ( p ) = - p \log p - ( 1 - p ) \log ( 1 - p ) ,
$$

where 0 log $0 : = 0 .$ . Thus, Ent $\left( p _ { e } \right)$ is the uncertainty associated with the acquisition outcome.

Intuition. Fixing H isolates the efect of acquisition. A query either adds one specified bridge or leaves the topology unchanged. The uncertainty represented by $p _ { e }$ concerns which of these two outcomes will occur.

Remark A.1 (Acquisition feasibility and execution reliability). Acquisition feasibility and execution reliability are distinct. The probability $p _ { e }$ concerns whether the expert can provide any feasible trajectory for a missing bridge. The edge reliability $r _ { e }$ concerns whether the learned policy can execute that bridge after it has been acquired. Current AALT estimates $r _ { e }$ from execution outcomes but does not estimate or use $p _ { e }$

Definition A.2 (Post-query task reachability). For a topology $\boldsymbol { \Gamma } = ( \mathcal { H } , \mathcal { E } )$ and task $\boldsymbol { x } = \left( \boldsymbol { s } _ { 0 } , \boldsymbol { g } \right)$ , let $h _ { s } \in \mathcal { H }$ be the hub matched to $s _ { 0 } .$ , and let $\mathcal { H } _ { g } \subseteq \mathcal { H }$ be the set of terminal hubs satisfying g. The reachability of x is

$$
R _ { \Gamma } ( x ) = \operatorname* { m a x } _ { h _ { g } \in \mathcal { H } _ { g } \atop P : h _ { s }  h _ { g } } \prod _ { e ^ { \prime } \in P } r _ { e ^ { \prime } } .
$$

If no such path exists, then $R _ { \Gamma } ( x ) = 0 .$

Let

$$
\Gamma _ { b } + e = ( \mathcal { H } , \mathcal { E } _ { b } \cup \{ e \} )
$$

denote the topology produced by successfully acquiring candidate $e _ { \cdot }$ The candidate edge is assigned the initial execution reliability $r _ { 0 } ,$ , while all existing edge reliabilities remain unchanged. The post-query reachability of task x is the random variable

$$
Z _ { x } = \left\{ \begin{array} { l l } { R _ { \Gamma _ { b } + e } ( x ) , } & { Y _ { e } ^ { \mathrm { a c q } } = 1 , } \\ { R _ { \Gamma _ { b } } ( x ) , } & { Y _ { e } ^ { \mathrm { a c q } } = 0 . } \end{array} \right.
$$

The dependence of $Z _ { x }$ on candidate e is left implicit.

Intuition. The query outcome selects between two fixed reachability values. If acquisition succeeds, the task is evaluated in the topology containing the bridge. If acquisition fails, it retains its previous reachability.

Definition A.3 (Weighted reachability information gain). Let $\mathcal { X } \ = \ \operatorname { s u p p } ( \rho )$ be the finite set of target tasks. The weighted reachability information gain of candidate edge e is

$$
\mathrm { R I G } _ { \rho } ( e ) = \sum _ { x \in \mathcal { X } } \rho ( x ) I ( Z _ { x } ; Y _ { e } ^ { \mathrm { a c q } } \mid \mathcal { D } _ { b } , \mathcal { H } ) .
$$

Define the set of tasks whose reachability difers between successful and unsuccessful acquisition as

$$
\begin{array} { r } { \mathcal { T } _ { e } : = \left\{ x \in \mathcal { X } \vert R _ { \Gamma _ { b } + e } ( x ) \neq R _ { \Gamma _ { b } } ( x ) \right\} . } \end{array}
$$

Their total probability under the target distribution is

$$
w _ { \rho } ^ { \prime } ( e ) : = \sum _ { x \in \mathcal { T } _ { e } } \rho ( x ) .
$$

Intuition. A task contributes information only when the query outcome changes its reachability. The quantity $w _ { \rho } ^ { \prime } ( e )$ counts the complete target-distribution weight of all such tasks.

Definition A.4 (Connectivity gain). The connectivity gain assigned to candidate edge e is

$$
\begin{array} { l } { \Delta ( e ) : = C ( \Gamma _ { b } + e ) - C ( \Gamma _ { b } ) } \\ { = \displaystyle \sum _ { x \in \mathcal { X } } \rho ( x ) R _ { \Gamma _ { b } + e } ( x ) - \sum _ { x \in \mathcal { X } } \rho ( x ) R _ { \Gamma _ { b } } ( x ) } \\ { = \displaystyle \sum _ { x \in \mathcal { X } } \rho ( x ) \left[ R _ { \Gamma _ { b } + e } ( x ) - R _ { \Gamma _ { b } } ( x ) \right] . } \end{array}
$$

Intuition. Both $w _ { \rho } ^ { \prime } ( e )$ and $\Delta ( e )$ concern tasks whose reachability changes. The diference is that $w _ { \rho } ^ { \prime } ( e )$ counts each task’s full probability weight, while $\Delta ( e )$ scales that weight by the size of the reachability improvement.

Lemma A.1 (Reachability-information decomposition). For any assignment ofedge reliabilities,

$$
\mathrm { R I G } _ { \rho } ( e ) = \mathrm { E n t } ( p _ { e } ) w _ { \rho } ^ { \prime } ( e ) .
$$

Proof. Conditioning on $\mathcal { D } _ { b }$ and H fixes the two possible values of $Z _ { x }$

For $x \in \mathcal { T } _ { e }$ , successful and unsuccessful acquisition produce distinct reachability values. Because $Z _ { x }$ is a deterministic function of the binary variable $Y _ { e } ^ { \mathrm { a c q } }$ and takes a distinct value under each outcome, observing $Z _ { x }$ determines $Y _ { e } ^ { \mathrm { a c q } }$ Therefore,

$$
I ( Z _ { x } ; Y _ { e } ^ { \mathrm { a c q } } \mid { \mathcal D } _ { b } , { \mathcal H } ) = \operatorname { E n t } ( p _ { e } ) .
$$

For x $\notin \mathcal { T } _ { e }$ , successful and unsuccessful acquisition produce the same reachability value. The variable $\hat { Z _ { x } }$ is therefore constant with respect to $\dot { Y } _ { e } ^ { \mathrm { a c q } }$ , giving

$$
I ( Z _ { x } ; Y _ { e } ^ { \mathrm { a c q } } \mid { \mathcal { D } } _ { b } , { \mathcal { H } } ) = 0 .
$$

Substituting these two cases into Definition A.3 gives

$$
\begin{array} { r l } & { { \mathrm { \tiny ~ R I G } } _ { \rho } ( e ) = \displaystyle \sum _ { x \in \mathcal { X } } \rho ( x ) I ( Z _ { x } ; Y _ { e } ^ { \mathrm { a c q } } \mid \mathcal { D } _ { b } , \mathcal { H } ) } \\ & { \quad \quad \quad = \displaystyle \sum _ { x \in \mathcal { T } _ { e } } \rho ( x ) \mathrm { E n t } ( p _ { e } ) } \\ & { \quad \quad \quad = \mathrm { E n t } ( p _ { e } ) \displaystyle \sum _ { x \in \mathcal { T } _ { e } } \rho ( x ) } \\ & { \quad \quad \quad = \mathrm { E n t } ( p _ { e } ) w _ { \rho } ^ { \prime } ( e ) . } \end{array}
$$

Intuition. Reachability information gain counts every task whose reachability reveals the query outcome. Each such task contributes the full entropy of that outcome, regardless of whether its reachability changes slightly or moves from zero to one.

Lemma A.2 (Connectivity-gain bound). For any assignment ofedge reliabilities,

$$
\Delta ( e ) \leq w _ { \rho } ^ { \prime } ( e ) .
$$

Equality holds if and only if

$$
R _ { \Gamma _ { b } + e } ( x ) - R _ { \Gamma _ { b } } ( x ) = 1
$$

for every $x \in \mathcal { T } _ { e }$

Proof. The topology $\Gamma _ { b } + e$ contains every edge and path available in $\Gamma _ { b }$ with unchanged reliability. It also contains the additional candidate edge e. Because task reachability is the maximum reliability over all available paths,

$$
R _ { \Gamma _ { b } + e } ( x ) \geq R _ { \Gamma _ { b } } ( x )
$$

for every $x \in \mathcal X$

For each $x \in \mathcal { T } _ { e }$ , the two reachability values difer. Reachability monotonicity and the range $\dot { R _ { \Gamma } } ( x ) \in [ 0 , 1 ]$ therefore give

$$
0 < R _ { \Gamma _ { b } + e } ( x ) - R _ { \Gamma _ { b } } ( x ) \leq 1 .
$$

Tasks outside $\mathcal { T } _ { e }$ have zero reachability change. Definition A.4 therefore gives

$$
\begin{array} { l } { \displaystyle \Delta ( e ) = \sum _ { x \in { \mathcal T } _ { e } } \rho ( x ) \left[ R _ { \Gamma _ { b } + e } ( x ) - R _ { \Gamma _ { b } } ( x ) \right] } \\ { \displaystyle \qquad \leq \sum _ { x \in { \mathcal T } _ { e } } \rho ( x ) } \\ { \displaystyle \qquad = w _ { \rho } ^ { \prime } ( e ) . } \end{array}
$$

Because $\mathcal { X } = \mathrm { s u p p } ( \rho )$ , every task in X has positive probability. Equality therefore holds exactly when

$$
R _ { \Gamma _ { b } + e } ( x ) - R _ { \Gamma _ { b } } ( x ) = 1
$$

for every $x \in \mathcal { T } _ { e }$

Intuition. Connectivity gain cannot count more than the full probability weight of a task whose reachability changes. It equals that full weight only when acquisition moves the task completely from reachability zero to reachability one.

Proposition 1 (Binary-reliability factorization). For binaryreliability $A A L T ,$ in which every acquired edge is treated as perfectly reliable,

$$
\mathrm { R I G } _ { \rho } ( e ) = \mathrm { E n t } ( p _ { e } ) \Delta ( e ) .
$$

ProofofProposition 1. Under binary reliability, every available edge has reliability one. Task reachability therefore satisfies

$$
R _ { \Gamma } ( x ) \in \{ 0 , 1 \}
$$

for every topology Γ and task x.

If $x \in \tau _ { e } ,$ acquiring e changes the reachability of x. Because reachability is binary and cannot decrease, this change must be

$$
R _ { \Gamma _ { b } } ( x ) = 0 , \qquad R _ { \Gamma _ { b } + e } ( x ) = 1 .
$$

Every task in $\mathcal { T } _ { e }$ therefore has reachability improvement equal to one. Lemma A.2 gives

$$
\Delta ( e ) = w _ { \rho } ^ { \prime } ( e ) .
$$

Substituting this equality into Lemma A.1 gives

$$
\begin{array} { r } { \mathrm { R I G } _ { \rho } ( e ) = \mathrm { E n t } ( p _ { e } ) w _ { \rho } ^ { \prime } ( e ) } \\ { = \mathrm { E n t } ( p _ { e } ) \Delta ( e ) . } \end{array}
$$

Intuition. Under binary reliability, an acquired bridge either creates a complete route for a task or does not change that task. Reachability information gain therefore separates exactly into uncertainty about successful acquisition and the total task-solving capability created when acquisition succeeds.

Proposition 2 (Soft-reliability bound). For soft-reliability AALT,

$$
\begin{array} { r } { \mathrm { E n t } ( p _ { e } ) \Delta ( e ) \le \mathrm { R I G } _ { \rho } ( e ) . } \end{array}
$$

The bound is exact whenever every task whose reachability changes movesfrom zero to one.

Proof of Proposition 2. Lemma A.2 gives

$$
\Delta ( e ) \leq w _ { \rho } ^ { \prime } ( e ) .
$$

Because $\mathrm { E n t } ( p _ { e } ) \ge 0$ , multiplying both sides by Ent $\left( p _ { e } \right)$ preserves the inequality:

$$
\begin{array} { r } { \operatorname { E n t } ( p _ { e } ) \Delta ( e ) \le \operatorname { E n t } ( p _ { e } ) w _ { \rho } ^ { \prime } ( e ) . } \end{array}
$$

Lemma A.1 gives

$$
\mathrm { E n t } ( p _ { e } ) w _ { \rho } ^ { \prime } ( e ) = \mathrm { R I G } _ { \rho } ( e ) .
$$

Combining these relations yields

$$
\begin{array} { r } { \mathrm { E n t } ( p _ { e } ) \Delta ( e ) \le \mathrm { R I G } _ { \rho } ( e ) . } \end{array}
$$

If every task whose reachability changes moves from zero to one, then every $x \in \mathcal { T } _ { e }$ has reachability improvement equal to one. Lemma A.2 then holds with equality, which gives

$$
\operatorname { E n t } ( p _ { e } ) \Delta ( e ) = \operatorname { R I G } _ { \rho } ( e ) .
$$

Intuition. With soft reliability, acquiring a bridge may improve a task without making its route perfectly reliable. Reachability information gain still counts the task’s full probability weight because the query outcome changes its reachability. Connectivity gain discounts the task according to the size of the improvement. Its entropy-scaled value is therefore a lower bound.

Remark A.2 (Equality conditions and scope). When $0 ~ <$ $p _ { e } ~ < ~ 1$ , the soft-reliability bound is exact if and only if every task in $\mathcal { T } _ { e }$ moves from reachability zero to one. When $p _ { e } \in \{ 0 , 1 \}$ , the acquisition outcome has zero entropy, so both sides of the bound are zero regardless of the reachability improvements.

The factorization and bound are pointwise relationships for each candidate bridge. They do not imply that connectivity gain and reachability information gain rank all candidates identically when candidates have diferent acquisition probabilities. AALT optimizes connectivity gain because it directly measures the capability created by successful acquisition and can be evaluated from the learned topology. Learning that a bridge is infeasible resolves uncertainty, but it does not add an executable behavior or increase any task’s reachability.

## B. Robotic Retrieval Environment

This section describes the retrieval environment used in our experiment. Code for this environment will be released on publication of this work.

## Physical Setup

The environment models a robotic servicing cell containing a UR5e arm with a Robotiq 2F-85 parallel-jaw gripper. The robot is mounted in front of an open cabinet containing 15 color-coded cylindrical canisters. The robot pedestal is centered relative to the cabinet, and the cabinet’s open face begins 0.28 m in front of the center of the pedestal.

The cabinet is 1.40 m wide, 0.70 m deep, and 0.545 m high. Its front is open, while its floor, back, sides, and roof are closed. The cabinet floor contains a 4 × 7 grid of discrete canister locations. Columns A through G run from left to right, and rows 1 through 4 run from the front of the cabinet toward the back.

<table><tr><td>Row</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td></tr><tr><td>1</td><td>一</td><td>guard</td><td></td><td>guard</td><td>一</td><td>guard</td><td>一</td></tr><tr><td>2</td><td>一</td><td>red</td><td>orange</td><td>teal</td><td>brown</td><td>pink</td><td></td></tr><tr><td>3</td><td>一</td><td>yellow</td><td>lime</td><td>cyan</td><td>purple</td><td>green</td><td>一</td></tr><tr><td>4</td><td>一</td><td></td><td>blue</td><td>white</td><td>magenta</td><td></td><td>一</td></tr></table>

Table 1: Canister planogram. Two ofthe three indicated guard locations are occupied in each starting layout.
<table><tr><td>Starting layout</td><td>Black guard</td><td>Gray guard</td></tr><tr><td>LEFT_A</td><td>D1</td><td>F1</td></tr><tr><td>LEFT_B</td><td>F1</td><td>D1</td></tr><tr><td>CENTER_A</td><td>B1</td><td>F1</td></tr><tr><td>CENTER_B</td><td>F1</td><td>B1</td></tr><tr><td>RIGHT_A</td><td>B1</td><td>D1</td></tr><tr><td>RIGHT_B</td><td>D1</td><td>B1</td></tr></table>

Table 2: The six starting layouts. All non-guard canisters retain the positions in Table 1.

The centers of the first and last columns are each 0.14 m from the corresponding side wall. Adjacent columns are approximately 0.187 m apart. The centers of the first and last rows are each 0.14 m from the front and back of the cabinet. Adjacent rows are 0.14 m apart.

Each canister is a vertical cylinder with radius 0.035 m, height 0.16 m, and mass 0.10 kg. Thirteen canisters have the fixed starting positions shown in Table 1. The black and gray canisters serve as movable access guards and vary between starting layouts.

Three retrieval zones, denoted R1, R2, and R3, are placed immediately outside the open front of the cabinet. Their centers are 0.10 m in front of the cabinet face and 0.15 m apart. Each retrieval platform is 0.14 × 0.14 m. The zones are visually distinguished by one, two, and three markers, respectively.

## Starting Layouts

The black and gray access guards occupy two of the three front-row locations B1, D1, and F1. The remaining location provides the clearest access corridor to the left, center, or right portion of the cabinet. Each access configuration has two variants that exchange the identities of the guards.

The A and B layouts represent the same general access configuration but difer in which guard occupies each blocking position. Changing between access configurations requires rearranging one or both guards.

## Work Orders

Each task specifies an ordered three-canister work order. The first requested canister must be delivered to R1, the second to R2, and the third to R3. There are three work-order families: MOTOR, SENSOR, and SERVICE. Each family contains four work orders formed from two possible first canisters and two possible second canisters. Every work order ends with the white canister.

<table><tr><td>Work order</td><td>R1</td><td>R2</td><td>R3</td></tr><tr><td>MOTOR_00</td><td>red</td><td>yellow</td><td>white</td></tr><tr><td>MOTOR_01</td><td>red</td><td>lime</td><td>white</td></tr><tr><td>MOTOR_10</td><td>orange</td><td>yellow</td><td>white</td></tr><tr><td>MOTOR_11</td><td>orange</td><td>lime</td><td>white</td></tr><tr><td>SENSOR_00</td><td>teal</td><td>blue</td><td>white</td></tr><tr><td>SENSOR_01</td><td>teal</td><td>purple</td><td>white</td></tr><tr><td>SENSOR_10</td><td>cyan</td><td>blue</td><td>white</td></tr><tr><td>SENSOR_11</td><td>cyan</td><td>purple</td><td>white</td></tr><tr><td>SERVICE_00</td><td>pink</td><td>green</td><td>white</td></tr><tr><td>SERVICE_01</td><td>pink</td><td>magenta</td><td>white</td></tr><tr><td>SERVICE_10</td><td>brown</td><td>green</td><td>white</td></tr><tr><td>SERVICE_11</td><td>brown</td><td>magenta</td><td>white</td></tr></table>

Table 3: The 12 ordered work orders. Each represents one goal.

The complete task set is the Cartesian product of the six starting layouts and 12 work orders, producing $6 \times 1 2 = 7 2$ start–goal tasks.

## Actions and Motion Constraints

The environment uses discrete pick-and-place actions. Each action selects one of the 15 canisters and one destination. A destination may be any of the 28 cabinet cells or one of the three retrieval zones. The complete action space therefore contains $1 5 ( 2 8 + 3 ) = 4 6 5$ actions.

A canister may be moved to any unoccupied cabinet cell if both its pick and placement paths are clear. This includes the black and gray guards as well as the canisters appearing in work orders. Cabinet rearrangement may therefore be used to clear an access corridor before retrieving a requested canister.

Each pick or placement is evaluated using a straight access corridor from a centered position 0.13 m in front of the cabinet to the selected source or destination. The corridor has a total width of 0.14 m, corresponding to four canister radii. A stationary canister blocks the path when its center lies within 0.105 m of the corridor centerline. Both the path to the source and the path to the destination must be clear. The corridor must also remain within the cabinet floor, roof, side walls, and back wall.

The experiments use abstract geometrically validated pickand-place execution. If the source, destination, and access corridors are valid, the selected canister is transferred directly to its destination. Continuous arm trajectories, grasp uncertainty, and contact dynamics are not included in the experimental outcome.

An action is invalid if the selected canister has already been retrieved, the destination is occupied, the source and destination are identical, either access corridor is blocked, or the action violates the ordered-retrieval rules. An invalid action does not change the arrangement but consumes one action from the episode horizon.

During evaluation, actions that are invalid under the current occupancy, retrieval-order, and corridor constraints are masked before action selection. This validity mask is computed from the simulator state and is not included in the visual observation.

## Ordered Retrieval Rules

For a work order $\boldsymbol { g } = \left( g _ { 1 } , g _ { 2 } , g _ { 3 } \right)$ , the required retrieval sequence is $g _ { 1 }  R 1 , g _ { 2 }  R 2 , g _ { 3 }  R 3$ . The agent may perform any number of valid cabinet rearrangements between these retrieval actions. However, it may not retrieve a later canister before completing the preceding retrieval. For example, $g _ { 2 }$ cannot be placed in R2 until $g _ { 1 }$ has been placed in R1. A canister not contained in the work order cannot be placed in a retrieval zone.

Retrieval is permanent. Once a canister is placed in R1, R2, or R3, it cannot be moved again during that episode. The task succeeds when all three requested canisters have been placed in their assigned retrieval zones in the correct order. An episode is considered a failure if this has not been completed within 30 actions.

## Observations

The agent receives a top-down RGB image of the complete cabinet and retrieval area after every action. Images are rendered at $3 2 0 \times 2 4 0$ pixels and resized to 128 × 128 before being supplied to the learned models.

The observation shows the physical canister arrangement but does not display the current work order. The ordered three-color goal is supplied separately.

## Initial Demonstration Coverage

The initial dataset contains 24 successful demonstrations comprising 196 actions. It includes the four MOTOR work orders from both LEFT layouts, the four SENSOR work orders from both CENTER layouts, and the four SERVICE work orders from both RIGHT layouts: $2 ( 4 ) + 2 ( 4 ) + 2 ( 4 ) =$ 24. Thus, every starting layout and every work order appears in the initial dataset, but only in its matched access region. The 48 cross-family combinations are omitted. In particular, no initial demonstration shows a MOTOR work order from a CENTER or RIGHT layout, a SENSOR work order from a LEFT or RIGHT layout, or a SERVICE work order from a LEFT or CENTER layout. The family and access-region groupings are used only to construct the task distribution and are not provided to the learner.

## C. Baselines

We compare AALT with Active Multi-task Fine-tuning (AMF) (Bagatella et al. 2025). AMF considers a pre-trained multi-task policy and sequentially selects complete tasks for additional expert demonstration. Recall that a task is $x = ( s _ { 0 } , g ) \in \mathcal { S } _ { 0 } \times \mathcal { G }$ , drawn from the target distribution $\rho ,$ and let $\mathcal { D } _ { b }$ denote the demonstration dataset available after acquisition round $b .$ Let $p ^ { \star } ( \tau \mid x )$ denote the expert trajectory distribution for task $x ,$ and let Π denote AMF’s uncertain policy model of the expert. For a candidate task $x ^ { \prime } ,$ , let $\tau ^ { \prime } \stackrel { . } { \sim } p ^ { \star } \dot { ( } \cdot | \ x ^ { \prime } )$ denote the expert demonstration that would be obtained by querying x<sup>′</sup>. AMF selects

$$
\arg \operatorname* { m a x } _ { x ^ { \prime } \in S _ { 0 } \times \mathcal { G } } \mathbb { E } \underset { \tau \sim p ^ { \star } ( \cdot | x ) } { \sim } \left[ \sum _ { t = 0 } ^ { | \tau | - 1 } I ( \Pi ( s _ { t } , x ) ; \tau ^ { \prime } \mid \mathcal { D } _ { b } ) \right] ,
$$

where $I ( U ; V \mid W )$ denotes conditional mutual information. The inner term measures how much the candidate demonstration $\tau ^ { \prime }$ is expected to reduce uncertainty about the expert’s action at state $s _ { t }$ for target task x. Averaging over $x \sim \rho$ and the corresponding expert trajectories values information that transfers across the complete target task distribution. Equivalently, AMF selects the task expected to minimize the remaining posterior entropy of the policy along target-task trajectories. This difers from AALT, which values a bridge according to the additional start–goal connectivity it creates.

Computing the AMF objective exactly would require knowing which states are likely to be visited while solving each target task and how observing every possible candidate demonstration would change the learner’s policy posterior. These quantities are not directly available. Following the practical neural-network formulation of AMF, we approximate the state occupancies using the expert trajectories already stored in $\mathcal { D } _ { b }$ . For each candidate context, importance weights give greater weight to stored trajectories that the current policy considers likely under that context and less weight to trajectories that it considers unlikely. A context is a complete task in AMF-Full and a grounded bridge in AMF-Bridge. We represent policy uncertainty using lossgradient embeddings. At each sampled state, we compute the exact cross-entropy gradient of the first-action output head, including its bias. Examples with similar gradients would produce similar policy updates, so their gradient inner products provide a measure of how much information can transfer between them. These inner products define a kernel that allows the policy to be approximated as a Gaussian process. AMF then estimates the posterior variance that would remain if each candidate were added as a demonstration, using the importance-weighted stored trajectories as possible demonstration outcomes. It selects the candidate with the lowest expected posterior variance, or equivalently the largest expected variance reduction.

AMF-Full follows the original AMF query format. Its candidate set contains all 72 complete start–goal tasks, and each accepted query returns a full expert demonstration from the selected starting layout to the selected work order. The target distribution in the AMF objective is uniform over the 72 tasks. The policy receives the RGB observation and a 51-dimensional factorized task descriptor containing one of six starting layouts and one of 15 colors for each of the three ordered goal positions. This factorization allows the policy to share information across previously unseen start–goal combinations without assigning an unrelated learned identifier to each task. AMF-Full may query the same task more than once, as permitted by the original AMF formulation.

AMF-Bridge controls for the diference between complete-task queries and short bridge queries. It uses the same learned topology, admissible grounded bridge set, hub matcher, global difusion controller, expert, and execution procedure as AALT. Its candidates are therefore the same feasible missing hub-to-hub bridges considered by AALT. However, it ranks these candidates using AMF posterior-variance reduction rather than connectivity gain. Policy uncertainty is averaged uniformly over the observed and admissible hubtransfer contexts. Once a bridge has been successfully acquired, it becomes part of the topology and is removed from the missing-bridge candidate set. Thus, the comparison between AALT and AMF-Bridge changes the acquisition objective while holding the query granularity and controller fixed.

Both baselines use the same categorical difusion configuration as AALT: a prediction horizon of eight actions, 12 denoising steps, one executed action before replanning, four observation-history frames, transformer width 192, four layers, four attention heads, and zero dropout. Inference uses deterministic sampling with temperature 1.0. Initial policy training uses 600 epochs with learning rate $1 0 ^ { - 3 }$ . After each successful query, the policy is adapted for 100 epochs with learning rate $2 \times 1 0 ^ { - 4 }$ , replaying all initial and acquired demonstrations. The baselines also use the same image augmentation, label smoothing, and valid-action mask as $\mathbf { A A L T }$

For the AMF approximation, we use posterior noise $1 0 ^ { - 2 }$ following the reference AMF setting. Up to four evenly spaced decision points are taken from each stored trajectory, preserving early, intermediate, and late behavior while limiting acquisition cost. The resulting output-head gradients are compressed to 2,048 dimensions. Trajectory likelihoods use temperature 1.0, and log importance weights are clipped at 30 for numerical stability. The action-validity mask is used during policy execution but not when computing AMF likelihoods, preventing symbolic validity information from entering the uncertainty estimate. Each baseline receives the same 24 initial demonstrations. We omit AMF’s adaptiveprior mechanism because all initial demonstrations remain available for full replay, allowing forgetting to be handled identically across methods.

## D. Compute Resources

All models were trained and evaluated on a single workstation with an Intel Core Ultra 9 275HX CPU, 32 GB of system memory, and an NVIDIA GeForce RTX 5080 Laptop GPU with 16 GB of video memory. Training used one GPU without distributed computation.

The complete model contains 6,325,123 trainable parameters. Table 4 gives the parameter count for each major component.

<table><tr><td>Component</td><td>Parameters</td></tr><tr><td>Latent dynamics model</td><td>3,469,233</td></tr><tr><td>Hub matcher</td><td>10,369</td></tr><tr><td>Global diffusion policy</td><td>2,845,521</td></tr><tr><td>Total</td><td>6,325,123</td></tr></table>

Table 4: Trainable parameter counts for the model components used in the reported experiments.

Initial difusion-policy training used 600 epochs over 64 demonstrated behavior segments, corresponding to 38,400 optimization steps. After each successful expert query, the shared policy was adapted for 100 epochs using the complete replay dataset. The three adaptation rounds required approximately 357, 366, and 356 seconds, respectively, for a total adaptation time of approximately 18 minutes. These rounds performed 6,500, 6,600, and 6,700 optimization steps as the dataset grew, giving 19,800 post-query optimization steps in total.

## E. Limitations, Scope & Future Work

AALT is designed for settings in which task failures primarily result from missing connections between otherwise reusable behaviors. It is most appropriate when tasks are defined by start states and recognizable goals, demonstrations share meaningful intermediate states, and reaching the same hub makes similar downstream behaviors available. When tasks share little behavior, or when their main dificulty is learning individual skills rather than connecting them, policy-centric active imitation learning may be more appropriate. A hybrid method could address a broader range of settings by weighting both connectivity gain and information gain about the expert policy, but we leave this combination to future work.

AALT can incorporate new start and goal states by adding them as hubs together with incident bridge demonstrations. However, it does not presently propose new internal hub states. Its available compositions therefore depend on the intermediate structure represented by the learned topology, as extracted from the initial dataset. The quality of this structure also depends on the encoder, clustering procedure, and runtime matcher. Incorrectly merging states with diferent future possibilities can create routes that are not executable, while separating equivalent states can hide useful compositions. The action-conditioned representation and learned matcher are intended to reduce these errors.

AALT selects bridges greedily according to their immediate connectivity gain. This provides a direct and computationally manageable acquisition rule, but it may miss complementary bridge sets. For example, two bridges may jointly connect many starts and goals even though neither completes a new route by itself. Both may then receive little individual gain. This limitation could be addressed through limited-horizon lookahead, selection over small bridge sets, or a planning procedure that values partially completed connections. The present method also assigns a common initial reliability to new edges. Candidate-specific feasibility and reliability models could reduce the value assigned to bridges that are dificult for the expert to provide or dificult for the policy to execute.

Finally, connectivity is not always the principal limitation on task success. A topology may contain a route for a task even when one of its behaviors remains poorly learned. In this case, a policy-centric method can request demonstrations in regions where the policy is uncertain, while AALT may find no valuable missing bridge. A combined objective could choose between adding a missing bridge and refining an unreliable existing behavior based on their expected efects on task success.