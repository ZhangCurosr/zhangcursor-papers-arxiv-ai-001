# LLM-Guided Contextual Action Evaluation for Operational Decisions in Industrial Processes

Youcheng Zong , Student Member, IEEE, Runda Jia ,

and Dakuo He

Abstract— Industrial actor–critic methods usually represent continuous actions as anonymous numerical coordinates. They must therefore learn from limited interactions which process variables each action affects, in which direction, and after what delay. Fixed industrial documents already describe part of these relations, but their open-text statements neither represent the current operating condition nor directly fit a numerical policy. This article presents LLM-Guided Contextual Action Evaluation for Operational Decisions in Industrial Processes (LCAE), which uses a large language model before training to normalize fixed documents into a frozen action–observation– direction–delay relation basis. Recent numerical action– response history then modulates the current strength of each relation, while the evaluated action forms a stateconditioned nonlinear action-effect field in the same basis. The critic evaluates actions through this field, and the actor uses the same relation gains to generate actions, making document semantics part of maximum-entropy policy learning. Neither the LLM nor the embedding model runs online during training or deployment; the deployed policy uses only frozen semantic artifacts and visible numerical history. The method states a falsifiable hypothesis: when documented relations are correct and recent history reflects their contextual strength, this action representation should provide a more useful decision bias than raw action coordinates.

Index Terms— large language models, reinforcement learning, industrial process decision-making, action representation, actor–critic.

## I. INTRODUCTION

Operational decision-making in process industries adjusts manipulated variables from continuous observations and improves long-term operating objectives through delayed process responses. These tasks are usually partially observable because available measurements provide an incomplete view of the internal process state, and an action may affect several later sampling instants [1]. Reinforcement learning provides a common interface for such closed-loop sequential decisions, and deep actor–critic methods have been used for continuous control and industrial process control [2]–[4]. These studies show that numerical policies can learn complex control laws from interaction, but their action interfaces still usually treat each manipulated variable as an anonymous coordinate.

Anonymous action coordinates leave the critic with a compound learning problem. To estimate action value, the critic must learn how the state evolves and rediscover which observations each action affects, in which direction, and after what delay. Twin critics, target networks, and maximumentropy objectives can improve numerical optimization [3], [5], but they do not explain the process meaning of an action coordinate. In industrial environments with costly interaction, coupled actions, or slow responses, rediscovering these relations consumes samples that could otherwise be used to learn the policy.

Industrial numerical modeling has extracted state and prediction information from numerical trajectories [6]–[9], but industrial systems also contain information outside those trajectories. Equipment descriptions, variable tables, operating procedures, and environment descriptions often state which visible variables an action affects through process paths, the typical direction of the effect, and the delay range in which a response may occur. These relations may span several paragraphs, use inconsistent terminology, and mix structured fields with open text. An LLM can use open knowledge and language reasoning to normalize these descriptions into consistent relations, but the documents themselves remain static. A document can describe a basic effect path, but it cannot state how strong that path is in the current operating condition or directly specify the optimal action.

Variable-name matching or fixed rules alone are often insufficient for this conversion. The same manipulated variable may appear as an equipment tag, a control-loop name, or a process term, while direction and delay statements may be distributed across different document locations. The necessary role of the LLM in this work is to resolve such open vocabulary and cross-sentence relations and then restrict the result to fields allowed by the task schema. This role differs from replacing the controller with a language model: the LLM constructs the relation interface, while the numerical policy still learns current decisions from interaction.

This distinction determines how this work separates language and numerical information. Fixed documents define action-relation coordinates that are expected to remain stable across operating conditions, while recent action–response history estimates their relative strength in the current state.

The evaluated action must also retain its own amplitude because different actions cannot have the same value under identical relation strengths. The required action representation must therefore depend jointly on fixed relation identity, recent numerical context, and the current candidate action.

This article presents LLM-Guided Contextual Action Evaluation (LCAE) to implement this division of roles. Before training, an LLM reads only fixed environment documents and a task schema and produces action–observation–direction– delay relation cards. A fixed renderer and embedding model convert the cards into a frozen semantic basis, while the relation fields determine action selection, observation selection, and valid delay bands. An online numerical module extracts relation-wise action–response events from recent history and computes relation gains centered at one. The current action and these gains form a state-conditioned action-effect field in the frozen relation basis. The critic uses this field to estimate long-term value, and the actor uses the same gains to generate the action that will be executed.

LCAE does not place an LLM in the control loop. The LLM does not read numerical trajectories, rewards, hidden simulator states, optimal actions, or test information, and it does not generate, rank, or validate online actions. The embedding model also runs only before training. Closed-loop decisions during training and deployment are made entirely by a compact numerical actor, separating semantic construction cost from per-step control latency.

This work focuses on a scientific hypothesis rather than presuming a performance improvement. If fixed documents provide stable action paths and recent history contains their contextual strength, LCAE should improve policy learning and closed-loop return under limited interaction or changing action effectiveness. Conversely, if a matched-budget rawaction baseline reaches the same performance, or shuffling action–effect correspondences does not change the result, then the documented relations do not provide the expected decision bias. This falsifiable boundary limits the claim to contextual action evaluation rather than causal-effect identification, fault diagnosis, or safety control.

The main contributions are as follows.

• We reformulate action learning in industrial actor–critic as contextual action evaluation: fixed documents define relation identity, recent numerical history defines current relation strength, and the candidate action defines relation amplitude.

• We introduce a state-conditioned action-effect field that combines relation semantics, historical gains, and continuous actions through a bounded nonlinearity, with explicit empty-history fallback, local sensitivity, and actioncoverage conditions.

• We give a complete maximum-entropy actor–critic interface, training objective, and deployment boundary so that frozen offline semantic relations organize both action generation and value learning without an online LLM.

The remainder of this article is organized as follows. Section II reviews action representation, language-informed reinforcement learning, and knowledge-guided industrial control.

Section III defines LCAE, and Section IV summarizes its claims and boundaries.

## II. RELATED WORK

## A. From Continuous Control to Industrial Actor–Critic

Reinforcement learning formulates control as maximizing long-term return through environment interaction [10]. DDPG extends deterministic policy gradients to deep continuous control, TD3 reduces function-approximation error through twin critics and delayed updates, and SAC uses a stochastic policy and maximum-entropy objective to improve exploration and training stability [2], [3], [5]. Industrial studies have applied deep reinforcement learning to process control and discussed sample efficiency, constraints, stability, and deployment risk [4], [11]. A continuous polymerization case further shows that a numerical actor can directly generate manipulated actions [12]. These routes mainly improve how policies and value functions are optimized, while the action usually still enters the critic as a raw numerical vector. LCAE builds on this interface: it retains maximum-entropy actor–critic learning but changes the action representation received by the critic and the relation context visible to the actor. This position also separates the roles of numerical state and action representations. The history encoder must still summarize load, inventory, setpoints, and other operating conditions, while the action-effect field specifically determines how a candidate action should enter the value function under that history. LCAE therefore does not assume that semantic relations replace dynamic state modeling, and it does not change the environment’s reward or transition definition.

## B. From Action Identity to Action Relations

Action-representation studies have shown that structure among actions can improve generalization across actions. Act2Vec extracts action relations from demonstration contexts and uses the learned vectors for state augmentation and valuefunction approximation [13]. Another approach jointly learns a low-dimensional action representation and its mapping to actual actions, allowing outcomes of similar actions to be reused in large finite action sets [14]. Know Your Action Set explicitly learns relations in variable action sets so that a policy can handle action combinations that were not jointly available during training [15]. Action-adaptive policies can also infer action impact from recent state changes and adjust when actions are missing or their effects are perturbed [16]. These studies learn action structure from demonstrations, action sets, or online state changes, but they do not combine action–observation–direction–delay relations from fixed industrial documents with the current numerical amplitude of a continuous action. LCAE retains both stable relation identity from documents and contextual strength from history and evaluates continuous actions in the same effect field. Compared with discrete action embeddings, continuous industrial actions also require amplitude and sign to be preserved. Mapping an increase and a decrease in valve opening to one static action vector would lose control meaning, while learning only local action impact would discard cross-variable paths stated in documents. The relation-wise nonlinearity in LCAE preserves both elements and allows one action coordinate to correspond to several observation paths and delay bands.

## C. From Language Knowledge to Decision Interfaces

Natural language can provide goals, policy advice, reward structure, and environment knowledge to reinforcement learning, and these interfaces have been summarized in prior work [17]. EMMA grounds entities and dynamics from freeform manuals to observations to support policy generalization to unseen tasks [18]. RLang uses a formal declarative language to describe partial knowledge about all components of an MDP and grounds it into an algorithm-independent partial world model and policy [19]. Reward machines represent task-reward structure as automata that reinforcement-learning algorithms can exploit [20]. These methods show that language or symbolic knowledge can change the information available to decision learning rather than serving only as a prediction label.

Foundation models can also transfer pretrained representations to zero-shot domain tasks [21], while LLMs further provide the ability to propose goals or construct policy priors from open descriptions. ELLM rewards pretraining behaviors with LLM-suggested goals, while GLAM updates an LLM policy through online reinforcement learning [22], [23]. Both routes place the language model closer to the online decision process. LCAE extends the offline semantic-interface principle from industrial forecasting [24] to action evaluation: the LLM only normalizes fixed industrial relations before training, while the online actor and critic remain numerical models. LCAE therefore studies how language knowledge defines actionevaluation coordinates that numerical history can activate, rather than studying a language policy itself. The offline interface also allows relation artifacts to be inspected manually or programmatically before training. Compared with online freetext prompting, this interface can freeze document versions, relation fields, rendered text, and embeddings, ensuring that different training runs receive exactly the same semantic input. Auditability does not guarantee that a relation is correct, but it makes erroneous relations identifiable and removable rather than hiding them inside per-step language-model calls.

## D. From Industrial Knowledge Guidance to Contextual Action Evaluation

Industrial reinforcement learning has used rules, simulators, and process knowledge to narrow the search space or constrain policies. Chemical-operation procedure synthesis combines external knowledge, dynamic simulation, and deep reinforcement learning to search for interpretable operating steps [25]. Knowledge–data-guided methods use process-causality structure to organize industrial reinforcement learning, and multiobjective graph reinforcement learning uses expert prior graphs to guide fermentation optimization [26], [27]. These studies show that industrial knowledge can provide structure beyond numerical interaction, but they usually require manual rules, explicit graphs, or task-specific knowledge encoding.

Another route places LLM agents directly in industrial decisions. Systems have used LLMs to interpret real-time events, generate production plans, and control automation operations [28], and other work has used them for human–AI collaborative industrial decision support [29]. Agent frameworks for fault-tolerant control also generate recovery actions and validate them with digital twins, knowledge graphs, and deterministic constraints [30]. These systems target planning, recovery, or human-decision layers, whose time scales and safety interfaces differ from a closed-loop policy that produces continuous actions with a compact actor at every sampling instant. LCAE therefore restricts the LLM to offline relation construction and focuses on contextual action evaluation jointly defined by document relations and recent numerical history. This distinction also limits the deployment claim. LCAE provides no safety guarantee for LLM-generated actions and does not place language-reasoning latency inside the realtime control budget because the deployed policy never calls the LLM. Its required engineering checks are offline relation correctness, action-space coverage, and standard validation of the online numerical module.

## III. METHOD

## A. Problem Formulation and Decision Coordinates

Consider a partially observable industrial decision process in environment $e .$ Let $\bar { o _ { t } ^ { \mathrm { p h y } } } \in \mathcal { O } _ { e } ^ { \mathrm { p h y } }$ and $a _ { t } ^ { \mathrm { p h y } } \in \mathcal { A } _ { e } ^ { \mathrm { p h y } }$ denote the physical observation and physical action, respectively. Fixed elementwise strictly increasing affine bijections $\mathcal { N } _ { e } ^ { o }$ and $\mathcal { N } _ { e } ^ { a }$ define the internal decision coordinates:

$$
o _ { t } = \mathcal { N } _ { e } ^ { o } \Big ( o _ { t } ^ { \mathrm { p h y } } \Big ) \in \mathbb { R } ^ { d _ { o } } , \qquad a _ { t } = \mathcal { N } _ { e } ^ { a } \Big ( a _ { t } ^ { \mathrm { p h y } } \Big ) \in \mathcal { A } _ { e } \subseteq \mathbb { R } ^ { d _ { a } } .\tag{1}
$$

Strict monotonicity preserves the signs of action and response directions stated in the documents. Except for quantities carrying the phy superscript at the environment interface, all observations, actions, histories, and policy distributions below use these decision coordinates.

The policy-visible history at decision instant t is

$$
\mathcal { H } _ { t } = \left( o _ { t - L : t } , a _ { t - L : t - 1 } , \Delta t _ { t - L : t - 1 } \right) ,\tag{2}
$$

where $\Delta t _ { i }$ is the physical time elapsed after executing action $a _ { i }$ . The policy generates $a _ { t } ,$ the environment executes $a _ { t } ^ { \mathrm { p h y } } = ( \mathcal { N } _ { e } ^ { a } ) ^ { - 1 } ( a _ { t } )$ , and then returns reward $r _ { t }$ and the next observation $o _ { t + 1 }$ . This history definition covers both fixed-step and irregularly timed decision processes.

For a maximum-entropy policy $\pi ,$ , the action value with continuous-time discounting is

$$
\begin{array} { r l } & { Q ^ { \pi } ( \mathcal { H } _ { t } , a _ { t } ) = \mathbb { E } _ { \pi } \Bigg [ r _ { t } + \displaystyle \sum _ { k = 1 } ^ { \infty } \exp \left( - \lambda \displaystyle \sum _ { j = 0 } ^ { k - 1 } \Delta t _ { t + j } \right) } \\ & { ~ \cdot \left( r _ { t + k } - \alpha \log \pi ( a _ { t + k } \mid \mathcal { H } _ { t + k } ) \right) ~ \Big \vert ~ \mathcal { H } _ { t } , a _ { t } \Bigg ] , } \end{array}\tag{3}
$$

where $\lambda > 0$ is the temporal discount rate and $\alpha > 0$ is the entropy temperature. When every $\Delta t _ { t }$ is fixed, $\exp ( - \lambda \Delta t _ { t } )$ reduces to the standard discrete discount factor γ. LCAE

retains this decision objective but does not use raw action a directly as the critic’s action input.

## B. Offline Action-Effect Relation Basis

For environment $e ,$ let $D _ { e }$ and $S _ { e }$ denote fixed industrial documents and the task schema. The LLM runs only before policy training and produces a set of relation cards

$$
\begin{array} { r } { \mathcal { K } _ { e } = F _ { \mathrm { L L M } } ( D _ { e } , \mathcal { S } _ { e } ) = \{ c _ { p } \} _ { p = 1 } ^ { P _ { e } } . } \end{array}\tag{4}
$$

Each card $c _ { p }$ specifies an action field, an affected visible observation, a typical effect direction, and a physical responsedelay band. A relation card may reference only actions and visible observations declared in the task schema. It contains no reward, hidden state, future measurement, test information, or recommended action.

Each relation card should also retain its document-evidence location and schema version so that the relation artifact can be traced to fixed inputs. Structural validation rejects unknown fields, invisible observations, undeclared actions, empty delay bands, and relations outside the allowed scope. Semantic validation checks whether direction and delay are supported by the recorded document span rather than freely completed by the model. These checks occur before embedding so that invalid cards cannot enter the frozen relation basis.

A fixed renderer R turns each card into canonical text, and a fixed embedding model $E _ { \psi }$ then produces a unit semantic direction:

$$
v _ { p } = \frac { E _ { \psi } ( \mathcal { R } ( c _ { p } ) ) } { \lVert E _ { \psi } ( \mathcal { R } ( c _ { p } ) ) \rVert _ { 2 } } \in \mathbb { R } ^ { k } , \qquad V _ { e } = \left[ \begin{array} { l } { v _ { 1 } ^ { \top } } \\ { \vdots } \\ { v _ { P _ { e } } ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { P _ { e } \times k } .\tag{5}
$$

Matrix $V _ { e }$ remains frozen during policy training and deployment. The unit normalization controls only embedding scale and does not change the field semantics of a relation card.

The cards also determine an action-selection matrix $M _ { e } ^ { a } \in \{ 0 , 1 \} ^ { P _ { e } \times d _ { a } }$ , an observation-selection matrix $M _ { e } ^ { o } ~ \in$ $\{ 0 , 1 \} ^ { P _ { e } \times d _ { o } }$ , and delay bands $\{ \mathcal { L } _ { p } \} _ { p = 1 } ^ { P _ { e } }$ . Row $p$ selects the action and visible response for relation $p ,$ while ${ \mathcal { L } } _ { p }$ gives the physical-time range in which a response may be collected after the action. The effect direction remains in the full relation card and its semantic direction $v _ { p } ,$ while the selection matrices perform only field routing. Together, these offline objects define the basic action-relation coordinates that remain fixed across episodes. One action may correspond to several cards, and one observation may receive several action paths, but duplicate cards should be merged before freezing. Every action coordinate must be covered by at least one relation, and the final relation order must remain identical across caches, matrices, and training runs. Thus, $\kappa _ { e } , V _ { e } ,$ , the two selection matrices, and the delay bands form one versioned offline artifact rather than a stochastic prompt result regenerated between episodes.

## C. History-Aligned Relation Strength

A numerical history encoder first produces the state representation

$$
h _ { t } = E _ { \theta } ( \mathcal { H } _ { t } ) \in \mathbb { R } ^ { d _ { h } } .\tag{6}
$$

This representation summarizes the overall history needed by the policy, while the relation-strength module explicitly preserves action–response events aligned with documented relations.

To handle irregular sampling, set $\tau _ { t - L } = 0$ and $\begin{array} { r l } { \tau _ { i + 1 } } & { { } = } \end{array}$ $\tau _ { i } + \Delta t _ { i }$ , and let $T _ { t } = \tau _ { t } - \tau _ { t - L }$ denote the window duration. Let $M _ { e , p } ^ { a }$ and $M _ { e , p } ^ { o }$ denote row p of the two selection matrices. The valid action–response index set for relation $p$ is

$$
\mathcal { T } _ { t , p } = \{ ( i , j ) : t - L \leq i < j \leq t , \tau _ { j } - \tau _ { i } \in \mathcal { L } _ { p } \} .\tag{7}
$$

This set permits only responses that are already visible at the current instant and satisfy the documented delay.

For every $( i , j ) \in \mathcal { T } _ { t , p } ,$ define the relation-aligned event

$$
z _ { t , i j } ^ { ( p ) } = \bigg [ M _ { e , p } ^ { a } a _ { i } , ~ M _ { e , p } ^ { o } \big ( o _ { j } - o _ { i } \big ) , ~ \frac { \tau _ { j } - \tau _ { i } } { T _ { t } } , ~ \frac { \tau _ { t } - \tau _ { j } } { T _ { t } } \bigg ] \in \mathbb { R } ^ { 4 } .\tag{8}
$$

The first two components are the corresponding historical action amplitude and visible response change, and the last two are normalized response delay and response age. An event describes observed action–response alignment and is not interpreted as an identified causal effect. Using change $o _ { j } - o _ { i }$ rather than an absolute observation makes the event focus on the local response after an action, while $h _ { t }$ retains responsibility for modeling the overall operating condition. Explicit physical time rather than index distance allows one documented delay to be used with irregular sampling. Response age distinguishes newly observed evidence from older evidence near the start of the window, while the shared map learns how to attenuate or retain this information.

One shared numerical map $\Phi _ { \xi } : \mathbb { R } ^ { 4 }  \mathbb { R } ^ { d _ { r } }$ encodes events from every relation, and mean aggregation forms the relationwise representation:

$$
R _ { t , p } = \operatorname* { M e a n } _ { ( i , j ) \in \mathscr { T } _ { t , p } } \Phi _ { \xi } \left( z _ { t , i j } ^ { ( p ) } \right) \in \mathbb { R } ^ { d _ { r } } , \qquad \mathrm { M e a n } _ { \emptyset } = 0 .\tag{9}
$$

Using a shared map avoids assigning an independent event network to every documented relation; relation differences are expressed by field selection, delay, and the semantic basis. Stacking all relation representations gives

$$
R _ { t } = \left[ \begin{array} { c } { R _ { t , 1 } ^ { \top } } \\ { \vdots } \\ { R _ { t , P _ { e } } ^ { \top } } \end{array} \right] = \mathcal { T } _ { e , \xi } ( \mathcal { H } _ { t } ; M _ { e } ^ { a } , M _ { e } ^ { o } , \{ \mathcal { L } _ { p } \} ) \in \mathbb { R } ^ { P _ { e } \times d _ { r } } .\tag{10}
$$

Numerical events and frozen semantic directions are projected into the same q-dimensional matching space. The relation-wise matching score is

$$
\ell _ { t } = \mathrm { d i a g } \big [ ( R _ { t } W _ { r } ) ( V _ { e } W _ { v } ) ^ { \top } \big ] \in \mathbb { R } ^ { P _ { e } } ,\tag{11}
$$

where $W _ { r } \in \mathbb { R } ^ { d _ { r } \times q }$ and $W _ { v } \in \mathbb { R } ^ { k \times q }$ are trainable projections. The diagonal operator matches numerical events for relation $p$ only with documented relation $p ,$ avoiding unconstrained cross-relation mixing.

The matching score is converted into relation gains through the centered map

$$
g _ { t } = 2 \sigma ( \ell _ { t } ) \in ( 0 , 2 ) ^ { P _ { e } } ,\tag{12}
$$

where $\sigma$ is the logistic function. $g _ { t , p } ~ = ~ 1$ denotes neutral documented strength, while $g _ { t , p } ~ < ~ 1$ and $g _ { t , p } ~ > ~ 1$ denote suppression and enhancement of relation p by recent history. When a relation has no valid historical event, $R _ { t , p } ~ = ~ 0$ so $\ell _ { t , p } ~ = ~ 0$ and $g _ { t , p } ~ = ~ 1$ . This default preserves the basic documented relation rather than deleting an action path because recent evidence is unavailable. The neutral fallback also avoids equating “no valid event has been observed” with “the relation is currently ineffective.” The former is missing evidence, whereas the latter requires recent events that support a smaller gain. This distinction prevents cold-start windows, long-delay relations, and temporary missing observations from automatically creating zeroed action channels.

## D. State-Conditioned Action-Effect Field

For any action $a \in A _ { e }$ currently evaluated by the critic, the relation-level action amplitude is

$$
u _ { e } ( a ) = M _ { e } ^ { a } a \in \mathbb { R } ^ { P _ { e } } .\tag{13}
$$

The same action coordinate may appear in several relations because one manipulated variable may affect several process observations or delay paths. Relation gains and the current action are jointly activated by the same bounded nonlinearity:

$$
\eta _ { t } ( a ) = \operatorname { t a n h } ( g _ { t } \odot u _ { e } ( a ) ) \in ( - 1 , 1 ) ^ { P _ { e } } .\tag{14}
$$

This operation preserves the action sign, allows relation gains to continuously modulate action sensitivity, and bounds the relation amplitude produced by large normalized actions.

The LCAE state-conditioned action-effect field is defined as

$$
\bigg | f _ { t } ( \boldsymbol { a } ) = \eta _ { t } ( \boldsymbol { a } ) ^ { \top } V _ { e } = \operatorname { t a n h } ( g _ { t } \odot M _ { e } ^ { a } \boldsymbol { a } ) ^ { \top } V _ { e } \in \mathbb { R } ^ { k } . \bigg |\tag{15}
$$

This field combines three sources with distinct roles: $V _ { e }$ gives relation identity, $g _ { t }$ gives history-conditioned strength, and $M _ { e } ^ { a } a$ gives the amplitude of the currently evaluated action. The same action therefore usually has different effect fields under different histories, while different actions remain distinct under the same history. This is not ordinary concatenation of a document embedding, history vector, and action vector because the three jointly determine the critic’s action input through a relation-wise nonlinearity.

The local sensitivity of the gain-modulated activation to relation-level action amplitude is

$$
\frac { \partial \eta _ { t , p } ( a ) } { \partial u _ { e , p } ( a ) } = g _ { t , p } \left( 1 - \eta _ { t , p } ( a ) ^ { 2 } \right) > 0 .\tag{16}
$$

Because this derivative is strictly positive, the gain does not reverse the action direction in decision coordinates. It changes local sensitivity, while tanh gradually reduces sensitivity in large-action regions. Relation direction is retained by the frozen semantic basis, and numerical history modulates only its current strength.

To prevent the effect field from collapsing valid action directions, its Jacobian with respect to the original decision action should have full column rank:

$$
\begin{array} { l } { { J _ { t } ( a ) = \displaystyle \frac { \partial { f _ { t } ( a ) } } { \partial { a } } } \qquad } \\ { { \quad = V _ { e } ^ { \top } \operatorname { D i a g } \left[ g _ { t } \odot \left( 1 - \eta _ { t } ( a ) ^ { 2 } \right) \right] M _ { e } ^ { a } , \qquad \mathrm { r a n k } \ : J _ { t } ( a ) = d _ { a } } } \end{array}\tag{17}
$$

Because every diagonal term is strictly positive, a sufficient structural condition is rank $\left( M _ { e } ^ { a } \right) = d _ { a }$ and rank $( V _ { e } ) = P _ { e }$ The first condition requires every valid action direction to be covered by an independent set of relations, and the second requires the semantic relation directions to have no rowrank degeneration. These conditions should be checked when building the relation basis; if they fail, relation cards must be corrected, merged, or added rather than allowing actiondirection loss to remain hidden during training.

## E. Decision Backbone and Training Objective

LCAE defines twin critics from the history representation and action-effect field:

$$
\begin{array} { r } { Q _ { m } ( \mathcal { H } _ { t } , a ; V _ { e } ) = C _ { \phi _ { m } } \big ( h _ { t } , f _ { t } ( a ) \big ) , \qquad m \in \{ 1 , 2 \} . } \end{array}\tag{18}
$$

The critic receives no raw-action bypass, so its action evaluation must pass through the relation field. The environmentspecific actor is

$$
a _ { t } \sim \pi _ { \omega } ( \cdot \mid \mathcal { H } _ { t } ) : = \pi _ { \omega } ( \cdot \mid h _ { t } , g _ { t } ) .\tag{19}
$$

The actor uses the overall history representation $h _ { t }$ and relation-wise gains $g _ { t }$ but does not receive an arbitrary concatenation of frozen semantic vectors. Relation identity enters policy learning through the gain indices and the critic’s action field.

For replay transition $( \mathcal { H } _ { t } , a _ { t } , r _ { t } , \Delta t _ { t } , \mathcal { H } _ { t + 1 } , d _ { t } )$ , where $d _ { t }$ is the termination indicator, the maximum-entropy Bellman target is

$$
\begin{array} { r l } & { y _ { t } = r _ { t } + ( 1 - d _ { t } ) \exp ( - \lambda \Delta t _ { t } ) } \\ & { \qquad \cdot \mathbb { E } _ { a ^ { \prime } \sim \pi _ { \omega } ( \cdot | \mathcal { H } _ { t + 1 } ) } \left[ \underset { m } { \operatorname* { m i n } } \bar { Q } _ { m } ( \mathcal { H } _ { t + 1 } , a ^ { \prime } ; V _ { e } ) - \alpha \log \pi _ { \omega } ( a ^ { \prime } \mid \mathcal { H } _ { t + 1 } ) \right] , } \end{array}\tag{20}
$$

where $\hat { Q } _ { m }$ denotes a target critic. The temporal discount directly uses the realized $\Delta t _ { t }$ in the transition, keeping different action durations consistent in the value target.

The twin critics are trained by minimizing

$$
\mathcal { L } _ { Q } = \mathbb { E } \left[ \sum _ { m = 1 } ^ { 2 } \left( Q _ { m } ( \mathcal { H } _ { t } , a _ { t } ; V _ { e } ) - y _ { t } \right) ^ { 2 } \right] .\tag{21}
$$

The reparameterized actor objective is

$$
\mathcal { L } _ { \pi } = \mathbb { E } _ { a \sim \pi _ { \omega } ( \cdot | \mathcal { H } _ { t } ) } \Big [ \alpha \log \pi _ { \omega } ( a \mid \mathcal { H } _ { t } ) - \operatorname* { m i n } _ { m } Q _ { m } ( \mathcal { H } _ { t } , a ; V _ { e } ) \Big ] .\tag{22}
$$

The actor gradient passes through the critic that depends on $f _ { t } ( a )$ , so the relation basis does more than provide extra context: it directly shapes action updates. At the same time, the actor explicitly receives $g _ { t }$ , allowing it to use the same historyconditioned relation strengths before generating an action.

The parameter sets $E _ { \theta } , \Phi _ { \xi } , W _ { r } , W _ { v } , C _ { \phi _ { 1 } } , C _ { \phi _ { 2 } } ,$ , and $\pi _ { \omega }$ are trained by the decision losses above. $F _ { \mathrm { L L M } } , E _ { \psi } , V _ { e } , M _ { e } ^ { a } , M _ { e } ^ { o }$ and $\{ { \mathcal { L } } _ { p } \}$ remain frozen during policy training. This separation prevents the policy loss from rewriting offline document relations into unauditable online semantic objects. It also allows .the same relation artifacts to be reused across random seeds and matched baselines under a fixed experimental protocol.

To separate method gains from extra model capacity, a rawaction baseline should share the history encoder, actor and critic widths, optimizer, replay data, environment-interaction budget, and seed path. Because LCAE adds relation-event and projection parameters, matching hidden widths alone does not constitute strict capacity matching. A parametercount-matched numerical baseline without semantic content is therefore also required, together with ablations that keep the architecture fixed while shuffling or removing relation content. These comparisons are not part of the method definition itself, but they are necessary interfaces for testing whether the actioneffect field actually uses documented relations.

## F. Deployment and Claim Boundaries

At deployment, the system first computes $h _ { t } , \ R _ { t }$ , and $g _ { t }$ from visible numerical history, after which the actor generates decision action $a _ { t } .$ . The action is mapped back to physical coordinates through $( \mathcal { N } _ { e } ^ { a } ) ^ { - 1 }$ and sent to the environment. The critic, LLM, and embedding model do not participate in per-step online execution. Online computation contains only history encoding, relation-event aggregation, fixed matrix operations, and one actor forward pass.

Different industrial environments share the relation-card schema, action-effect-field definition, and training interface, but they do not share one controller. Each environment retains its own observation space, action space, reward, coordinate transforms, history length, delay bands, encoder, actor, and critic parameters. This work therefore does not claim direct policy transfer across environments without retraining.

Relation gain $g _ { t }$ is a history-conditioned action-evaluation quantity, not a physical gain or a strict causal coefficient. Unobserved disturbances, closed-loop feedback, and sensor errors may all affect observation changes after an action. LCAE does not perform fault diagnosis, measurement correction, safety shielding, or causal identification; these capabilities require separate evidence and dedicated mechanisms. The method also assumes that fixed documents are sufficiently accurate and that relation cards cover every valid action direction. When documents are outdated, relations conflict, or action coverage is insufficient, the corresponding relation artifacts should be withheld and revalidated before deployment rather than left for the online policy to repair automatically. The deployed system should also bind an offline-artifact hash to the policy checkpoint so that the controller cannot load a mismatched relation order or coordinate definition. If a history window is incomplete, the system may use the defined neutral fallback; if the schema or coordinate transform does not match, execution must be rejected rather than silently degraded. This distinction separates recoverable data sparsity from configuration errors that would change action meaning.

## IV. CONCLUSION

This article introduced LCAE, which replaces anonymous continuous actions in industrial actor–critic learning with a state-conditioned action-effect field. Fixed industrial documents define action–observation–direction–delay relations through an offline LLM, recent numerical history modulates relation strength, and the current action provides the relation amplitude being evaluated. The same relation gains inform action generation, and the same semantic basis supports longterm value learning, so document knowledge enters policy optimization while the deployed loop remains free of LLM calls.

The central claim is a testable decision bias rather than an unmeasured performance advantage. When documented relations are correct and recent action–response history reflects contextual strength, LCAE should improve policy learning under limited interaction and changing action effectiveness; matched baselines and shuffled-relation ablations can directly falsify this hypothesis. Relation gains should not be interpreted as identifiable causal coefficients, and different environments still require their own controllers, rewards, and training processes. Subsequent validation should first examine relationcard correctness, action coverage, non-semantic baselines with matched information and capacity, and closed-loop outcomes after documented relations are removed or shuffled.

## REFERENCES

[1] L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, “Planning and acting in partially observable stochastic domains,” Artificial Intelligence, vol. 101, no. 1–2, pp. 99–134, 1998. [Online]. Available: https://doi.org/10.1016/S0004-3702(98)00023-X

[2] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” in Proceedings of the 4th International Conference on Learning Representations, 2016, openReview publication; no page range assigned. [Online]. Available: https://arxiv.org/abs/1509.02971

[3] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actorcritic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 1861–1870. [Online]. Available: https://proceedings.mlr.press/v80/haarnoja18b.html

[4] R. Nian, J. Liu, and B. Huang, “A review on reinforcement learning: Introduction and applications in industrial process control,” Computers & Chemical Engineering, vol. 139, p. 106886, 2020. [Online]. Available: https://doi.org/10.1016/j.compchemeng.2020.106886

[5] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 1587–1596. [Online]. Available: https://proceedings.mlr.press/v80/fujimoto18a.html

[6] Y. Zong, S. Hu, D. Qin, Z. Wang, C. Zhang, J. Chu, and L. Zhang, “Iron-tapping state recognition of blast furnace based on Bi-GRU composite model and post-processing classifier,” IEEE Sensors Journal, vol. 23, no. 18, pp. 22 006–22 018, 2023. [Online]. Available: https://doi.org/10.1109/JSEN.2023.3300123

[7] Y. Zong, Z. Wang, X. Liu, Y. Nian, J. Pan, C. Zhang, Y. Wang, J. Chu, and L. Zhang, “Judgment of blast furnace iron-tapping status based on data differential processing and dynamic window analysis algorithm,” Progress in Natural Science: Materials International, vol. 33, no. 4, pp. 450–457, 2023. [Online]. Available: https: //doi.org/10.1016/j.pnsc.2023.10.004

[8] Y. Zong, Y. Nian, C. Zhang, X. Tang, L. Wang, and L. Zhang, “Hybrid grid search and Bayesian optimization-based random forest regression for predicting material compression pressure in manufacturing processes,” Engineering Applications of Artificial Intelligence, vol. 141, p. 109580, 2025. [Online]. Available: https://doi.org/10.1016/j.engappai. 2024.109580

[9] Y. Zong, R. Jia, S. Wu, L. Zhang, and D. He, “A metacontrastive learning hybrid model for adaptive temperature trend prediction in variable ladle preheating,” Engineering Applications of Artificial Intelligence, vol. 162, p. 112750, 2025. [Online]. Available: https://doi.org/10.1016/j.engappai.2025.112750

[10] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA, USA: MIT Press, 2018. [Online]. Available: http://incompleteideas.net/book/the-book-2nd.html

[11] S. Spielberg, A. Tulsyan, N. P. Lawrence, P. D. Loewen, and R. B. Gopaluni, “Toward self-driving processes: A deep reinforcement learning approach to control,” AIChE Journal, vol. 65, no. 10, p. e16689, 2019. [Online]. Available: https://doi.org/10.1002/aic.16689

[12] Y. Ma, W. Zhu, M. G. Benton, and J. Romagnoli, “Continuous control of a polymerization system with deep reinforcement learning,” Journal of Process Control, vol. 75, pp. 40–47, 2019. [Online]. Available: https://doi.org/10.1016/j.jprocont.2018.11.004

[13] G. Tennenholtz and S. Mannor, “The natural language of actions,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 6196–6205. [Online]. Available: https://proceedings. mlr.press/v97/tennenholtz19a.html

[14] Y. Chandak, G. Theocharous, J. Kostas, S. Jordan, and P. Thomas, “Learning action representations for reinforcement learning,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 941–950. [Online]. Available: https://proceedings.mlr. press/v97/chandak19a.html

[15] A. Jain, N. Kosaka, K.-M. Kim, and J. J. Lim, “Know your action set: Learning action relations for reinforcement learning,” in Proceedings of the 10th International Conference on Learning Representations. OpenReview.net, 2022, openReview publication; no page range assigned. [Online]. Available: https://openreview.net/forum? id=MljXVdp4A3N

[16] K.-H. Zeng, L. Weihs, R. Mottaghi, and A. Farhadi, “Moving forward by moving backward: Embedding action impact over action semantics,” in Proceedings of the 11th International Conference on Learning Representations. OpenReview.net, 2023, openReview publication; no page range assigned. [Online]. Available: https: //openreview.net/forum?id=vmjctNUSWI

[17] J. Luketina, N. Nardelli, G. Farquhar, J. Foerster, J. Andreas, E. Grefenstette, S. Whiteson, and T. Rocktaschel, “A survey of¨ reinforcement learning informed by natural language,” in Proceedings of the 28th International Joint Conference on Artificial Intelligence, 2019, pp. 6309–6317. [Online]. Available: https://doi.org/10.24963/ ijcai.2019/880

[18] A. W. Hanjie, V. Y. Zhong, and K. Narasimhan, “Grounding language to entities and dynamics for generalization in reinforcement learning,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 4051–4062. [Online]. Available: https://proceedings.mlr.press/v139/hanjie21a.html

[19] R. Rodriguez-Sanchez, B. A. Spiegel, J. Wang, R. Patel, S. Tellex, and G. Konidaris, “RLang: A declarative language for describing partial world knowledge to reinforcement learning agents,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 29 161–29 178. [Online]. Available: https: //proceedings.mlr.press/v202/rodriguez-sanchez23a.html

[20] R. Toro Icarte, T. Q. Klassen, R. Valenzano, and S. A. McIlraith, “Reward machines: Exploiting reward function structure in reinforcement learning,” Journal of Artificial Intelligence Research, vol. 73, pp. 173–208, 2022. [Online]. Available: https://doi.org/10.1613/jair.1.12440

[21] W. Su, Y. Zong, R. Jia, J. Qin, and M. Li, “Zero-shot capillary segmentation in dermoscopy images via SAM2: A case study on oral mucosa,” IEEE Journal of Biomedical and Health Informatics, vol. 30, no. 5, pp. 4376–4387, 2026. [Online]. Available: https: //doi.org/10.1109/JBHI.2025.3628493

[22] Y. Du, O. Watkins, Z. Wang, C. Colas, T. Darrell, P. Abbeel, A. Gupta, and J. Andreas, “Guiding pretraining in reinforcement learning with large language models,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 8657–8677. [Online]. Available: https://proceedings.mlr.press/v202/du23f.html

[23] T. Carta, C. Romac, T. Wolf, S. Lamprier, O. Sigaud, and P.-Y. Oudeyer, “Grounding large language models in interactive environments with online reinforcement learning,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 3676–3713. [Online]. Available: https://proceedings.mlr.press/v202/carta23a.html

[24] Y. Zong, R. Jia, M. Ren, and D. He, “LLM-guided task-semantic field factorization for industrial process forecasting,” arXiv preprint arXiv:2607.06623, 2026. [Online]. Available: https://arxiv.org/abs/2607. 06623

[25] S. Kubosawa, T. Onishi, and Y. Tsuruoka, “Synthesizing chemical plant operation procedures using knowledge, dynamic simulation and deep reinforcement learning,” in Proceedings of the SICE Annual Conference, 2018, pp. 1376–1379. [Online]. Available: https://arxiv.org/abs/1903.02183

[26] X. Zhang, C. Song, J. Zhao, Z. Xu, and X. Deng, “Spatial-temporal causality modeling for industrial processes with a knowledge-data guided reinforcement learning,” IEEE Transactions on Industrial Informatics, vol. 20, no. 4, pp. 5634–5646, 2024. [Online]. Available: https://doi.org/10.1109/TII.2023.3333921

[27] D. Li and Z. Yang, “Prior-informed adaptive multi-objective graph reinforcement learning for lysine fed-batch fermentation process,” Journal of Process Control, vol. 156, p. 103578, 2025. [Online]. Available: https://doi.org/10.1016/j.jprocont.2025.103578

[28] Y. Xia, N. Jazdi, J. Zhang, C. Shah, and M. Weyrich, “Control industrial automation system with large language model agents,” in Proceedings of the 30th IEEE International Conference on Emerging Technologies and Factory Automation, 2025, pp. 1–8. [Online]. Available: https://doi.org/10.1109/ETFA65518.2025.11205539

[29] Y. Zong, R. Jia, K. Li, D. Xue, L. Zhang, and D. He, “LLM-driven human–AI collaborative decision support system for complex industrial processes: A case study in metallurgy,” Neural Networks, vol. 202, p. 109055, 2026. [Online]. Available: https: //doi.org/10.1016/j.neunet.2026.109055

[30] J. Vyas, M. S. Gill, A. Markaj, F. Gehlhoff, and M. Mercangoz, “From¨ detection to action: Using LLM agents for fault-tolerant control,” 2026. [Online]. Available: https://arxiv.org/abs/2606.28011