# AeroWeaver: An Embodied-Agent Harness for Weaving Aerial Skills into Distributed, Adaptive Swarm Execution

Jiabin Lou, Yirong Yang, Haopeng Wang, Xuxin Lv, Xinyu Liu, Diyuan Hou, Xuehong Liu, Rongye Shi, and Wenjun Wu<sup>∗</sup>

Abstract— Collective intelligence is a collaborative autonomy paradigm in which multiple agents pursue shared objectives through local perception, information exchange, and coordinated action. UAV swarms embody this paradigm by coordinating multiple vehicles in tasks such as search, inspection, and tracking. Recent advances in large language model (LLM) agents have strengthened natural-language task understanding and high-level planning, providing a flexible semantic interface between mission descriptions and collective behavior. While these advances expand semantic reasoning, applying LLM agents to UAV swarms raises challenges in grounding model decisions in executable capabilities, reconciling global task reasoning with distributed execution, and using mission-specific experience for continual adaptation. To address these challenges, we introduce AeroWeaver, an embodied-agent harness that weaves individual UAV skills into coordinated mission-level behavior. AeroWeaver connects semantic decisions to governed skills, organizes role-conditioned local agents for distributed coordination, and uses role-indexed state–action–reward experience to refine skill selection online. Experiments and runtime validation show that AeroWeaver maintains valid skill execution under tested conditions and supports body-local multi-UAV operation without a central agent generating joint actions from global context, while reward-guided online updates provide a training-free path for adaptive learning swarm agents from accumulated execution experience. Code website: https:// github.com/Admire-ljb/AeroWeaver.

## I. INTRODUCTION

Collective intelligence enables multiple agents to combine local perception, information exchange, and coordinated action around a shared objective. UAV swarms provide a representative physical realization by organizing the sensing, mobility, and task capabilities of multiple aerial platforms. As missions and operating conditions change, their collective effectiveness depends on a continuous link between highlevel task organization and reliable vehicle-level execution.

Recent advances in large language models (LLMs) bring complex-task understanding, knowledge organization, hierarchical reasoning, and long-horizon planning to the taskorganization layer. These capabilities make LLMs a flexible interface for interpreting mission descriptions and structuring high-level swarm behavior [1]. The resulting decisions ultimately operate through vehicle-specific observations, communication links, and actuators, making their connection to distributed physical execution a central research problem.

AI-agent systems mediate this connection through the runtime surrounding the model, commonly termed a harness.

The harness supplies context, exposes actions, routes model selections to executable tools, and returns environmental feedback. For an embodied swarm, the harness therefore becomes the interface between shared mission reasoning and multiple physically separate decision and execution processes. Viewed across the full execution loop, this interface raises a connected sequence of questions concerning whether semantic choices are physically executable, how executable choices remain coordinated across distributed vehicles, and how their outcomes should influence later decisions. The sequence begins at the semantic-to-physical boundary, where a language-level choice must correspond to a capability available in the deployed system and be routed to the vehicle that owns its execution interface. Without this mapping, task reasoning and flight control remain separate software layers.

Even when each action is executable, swarm-scale coordination remains difficult because task-level reasoning and body-level execution follow different information topologies. Many LLM-based multi-robot systems aggregate team state in a shared planner that decomposes the mission and returns a joint plan [2]. This organization places semantic decisions in a shared context, whereas observations, communication links, and actuators remain distributed across vehicles. The coordination problem is to maintain coherent mission progress across this distributed execution topology.

As these distributed decisions accumulate across rounds, the execution loop produces experience that can inform later choices. The relevance of each record depends on the local state, vehicle responsibility, peer interaction, and stage of mission progress in which it was collected. The adaptation problem is to relate accumulated outcomes to the current decision while preserving stable physical execution.

Together, these requirements call for a swarm runtime that connects semantic capability selection, body-local coordination, and experience-based adaptation within a consistent execution boundary. AeroWeaver addresses this need as an embodied-agent harness that weaves individual UAV skills into coordinated swarm execution. Its skill interface connects mission interpretation to body-scoped capability dispatch, while task-conditioned contexts preserve local decision authority across vehicles and reward-linked experience adjusts subsequent skill preferences without changing model weights or low-level controllers. The resulting loop allows missionlevel reasoning to organize collective behavior, body-bound agents to execute through their assigned platforms, and accumulated outcomes to inform later decisions.

Our contributions are as follows:

• Embodied-Agent Harness with Aerial Skills. We represent UAV capabilities as typed skill packages that pair skill.md documentation with executable objects and route selected skills through body-scoped executors.

• Distributed Swarm Orchestration. We derive rolespecific prompts and active skill subsets from each mission, bind each local agent to a UAV, and incorporate directed peer messages into the next body-local decision without introducing a central action controller.

• Experience-Guided Online Reinforcement. We index state–action–reward trajectories by semantic role, estimate skill-level advantages from similar experience, and enable training-free online policy optimization through experience-guided score updates.

The evaluation examines task performance and inference cost in MPE-inspired simulation scenarios and traces how accumulated rewards influence subsequent skill choices.

## II. RELATED WORK

## A. LLM-Based Embodied Agents

LLM-based embodied agents connect natural-language reasoning with perception, planning, and action in interactive environments. ReAct interleaves reasoning with actions and environmental observations, allowing plans to be revised during interaction [3]. For physical robots, the representation linking this reasoning to control varies across systems. SayCan combines language-model scores with learned affordance values to select feasible robot skills [4]. Code as Policies generates programs that compose perception and control APIs [5]. ReKep instead expresses manipulation goals as relational keypoint constraints and obtains actions through hierarchical optimization [6]. Voyager accumulates executable programs during interaction and retrieves them as reusable skills for later tasks [7]. These approaches establish skills, programs, and geometric constraints as intermediate representations between language-level reasoning and executable behavior.

Repeated interaction also provides experience that can improve later decisions. ExpeL extracts transferable naturallanguage knowledge from prior trajectories [8]. Agent Workflow Memory identifies recurring workflows in agent trajectories and retrieves them during subsequent online or offline decisions [9]. A-Mem organizes experience as linked notes whose contextual attributes evolve as new memories are added [10]. AgentRefine instead learns correction behavior from environment-feedback trajectories through refinement tuning [11]. AgentGym supplies diverse interactive environments and trajectory sets for evaluating and training agents through continued interaction [12]. JitRL retrieves state– action–return experience to estimate action advantages and adjusts policy logits without gradient updates [13]. Together, these methods distinguish memory organization, parameter learning, and inference-time policy correction as complementary routes to adaptation. For a physical swarm, reward-based reuse also requires relating experience to the local state and responsibility of the agent making the current decision.

Connecting language-mediated choices to external tools and physical actuators places additional demands on the execution interface. AgentDojo evaluates prompt-injection attacks and defenses when agents invoke tools over untrusted data [14]. BadRobot shows how language-model vulnerabilities can propagate into physical actions in embodied systems [15]. Thea formulates the surrounding context, action interface, and execution feedback as an embodied-agent harness [16]. SHAPER further studies the joint evolution of reusable skills and a context–code harness around a frozen language model [17]. These studies provide foundations for skill grounding, experience reuse, and execution-interface design. Their common focus on an individual agent, a shared tool environment, or model-level adaptation leaves the organization of execution authority and experience attribution across concurrently acting physical agents comparatively underexplored.

## B. LLM-Based Multi-Agent and Multi-Robot Systems

LLM-based multi-agent research extends languagemediated reasoning from a single embodied agent to teams that divide tasks, exchange information, and coordinate actions toward a shared objective. SMART-LLM separates task decomposition, coalition formation, and allocation to generate multi-robot task plans [18]. CaPo constructs a cooperative meta-plan and revises it through multi-agent discussion as task progress changes [19]. EMOS incorporates embodiment-derived capability descriptions into hierarchical planning for heterogeneous robot teams [20]. These systems emphasize task-level organization and capability-aware assignment through shared or hierarchical planning.

Other work places greater emphasis on interaction among individual decision processes. CoELA combines perception, memory, communication, planning, and execution for cooperation under decentralized control and costly communication [21]. RoCo equips robots with language-model agents that negotiate subtask plans and waypoints through dialogue, using motion-planning feedback to revise their proposals [22]. PARTNR evaluates embodied collaboration under spatial, temporal, and heterogeneous capability constraints. However, the evaluated agents still struggle to coordinate their actions, track task progress, and recover from errors [23]. These findings highlight a gap between generating cooperative plans and maintaining coordinated execution as a task unfolds.

Research on aerial teams has adapted language-model reasoning to UAV-specific capabilities, mission interfaces, and learning processes. FlockGPT maps natural-language descriptions to UAV formation geometry [24]. TALKER activates reusable action primitives and maintains an extensible knowledge library for multi-UAV missions [25]. Agents Trainer uses cooperating language-model agents to automate multi-agent reinforcement-learning configuration, reward design, and policy training for drone swarms [26]. AERIS dynamically rebinds role-specialized language modules across aerial executors at runtime [27]. These systems broaden the role of language models in aerial teams from mission interpretation to skill reuse, coordination, interface grounding, and policy development. Across their reported designs, mission reasoning, vehicle execution, and learning are typically implemented as distinct stages or subsystems. Experience reuse has also been studied at the level of multiagent orchestration. Skill-MAS treats orchestration knowledge as a non-parametric object that can be refined from multiple trajectories [28].

![](images/cb03742a72c9e789c84e1f7c89cb3e3a4d21efa46201cd66815e05cd1137ced0.jpg)  
Fig. 1. Overview of AeroWeaver. The embodied-agent harness (A) maps task context to a subset of Aerial Skills. Distributed swarm orchestration (B) instantiates role-conditioned local agents, binds each agent to one UAV, and supports directed peer coordination during execution. Experience-guided online reinforcement (C) retrieves role-relevant trajectories, estimates skill advantages from rewards, and updates the active skill ranking.

![](images/50ec591b7129a3f8b62b3f2e80be0cf8e2166d2bdd6f00631ffc75074e2a79c4.jpg)  
Fig. 2. Task-conditioned skill activation. Mission parsing and skill retrieval construct the Commander context for role assignment and activation of agent-specific skill subsets.

Taken together, prior work provides strong foundations for grounded skills, agent adaptation, and language-mediated team coordination. Integrating these capabilities in a physical swarm remains a systems question because semantic decisions, observations, communication, and actuators are distributed across different components.

## III. METHOD

AeroWeaver is an embodied-agent harness that weaves Aerial Skills into coordinated UAV swarm execution. As shown in Fig. 1, AeroWeaver integrates three components:

• (A) Aerial Skill Harness activates task-relevant Aerial Skills by parsing mission context, retrieving from the skill library, and exposing an active skill subset to each roleconditioned agent (Sec. III-A).

• (B) Distributed Swarm Orchestration maintains distributed execution through role-conditioned local agents that coordinate via directed peer messages (Sec. III-B).

• (C) Experience-Guided Reinforcement enables trainingfree online adaptation by refining agent policy from execution experience (Sec. III-C).

## A. Embodied-Agent Harness with Aerial Skills

Within the embodied-agent harness, Aerial Skills constitute an executable intermediate layer between high-level semantic planning and low-level motion control. This layer exposes each downstream UAV agent to a skill subset while retaining typed interfaces, body bindings, and approved executors within the runtime. Fig. 2 illustrates how skill activation instantiates this interface.

![](images/24503447f90e8a27df63ab35172422b31e35ff0340df85c690b16bd18346e107.jpg)  
Fig. 3. Distributed swarm execution and communication. The Commander supplies mission context to UAV-bound agents that select own-body actions and exchange peer messages; camera views and a four-UAV UE4 scene illustrate the deployment.

Skill substrate. We define a Skill as a reusable capability unit that couples an executable object with a skill.md document describing its semantic purpose, invocation interface, and operating conditions. Skills are derived from builtin flight and payload functions, platform adapters, perception modules, or compositions of existing skills. Each skill $\sigma _ { i }$ follows the common representation

$$
\sigma _ { i } = \langle n _ { i } , d _ { i } , I _ { i } , O _ { i } , P _ { i } , E _ { i } , M _ { i } \rangle ,\tag{1}
$$

where $n _ { i }$ and $d _ { i }$ identify and describe the skill, $I _ { i }$ and $O _ { i }$ specify typed inputs and outputs, $P _ { i }$ states its operating conditions, $E _ { i }$ denotes the associated executor, and $M _ { i }$ stores provenance, dependencies, and effects.

Mission parsing and retrieval. Given an operator instruction q and swarm state $s _ { t } ,$ , the mission parser extracts a structured task record $\chi ^ { t }$ containing the task type, participating UAVs, operating region, and mission constraints. This record supplies the semantic and operational context used to retrieve skills from the registered library. For a given task, the harness retrieves a task-level candidate catalog $\mathcal { A } ^ { t }$ from the available skills. The retriever then ranks this catalog by comparing the task and role query with the names, descriptions, tags, and aliases stored in each skill.md document.

Prompt reconstruction and role assignment. The harness combines $q , \chi ^ { t }$ , the current swarm state, and the documentation of the task-level candidate catalog to construct the Commander context shown in Fig. 2. The Commander assigns a semantic role $\rho _ { k }$ and local goal $g _ { k }$ to each participant, after which the harness forms the role-conditioned query $q _ { k } = \mathrm { P a c k } ( q , \rho _ { k } , g _ { k } )$ and activates

$$
\mathcal { L } _ { k } = \mathrm { T o p K } _ { \sigma _ { i } \in \mathcal { A } ^ { t } } \{ r _ { k } ( q _ { k } , \sigma _ { i } ) \} ,\tag{2}
$$

where $r _ { k }$ is the retrieval score for UAV k. The resulting $\mathcal { L } _ { k }$ determines the skill names and documentation rendered into agent k’s local context, and different agents may receive

different subsets of the same task-level catalog according to their assigned roles and local goals.

## B. Distributed Swarm Orchestration

Distributed Swarm Orchestration maintains a shared mission through concurrent decision processes bound to individual UAVs. As shown in Fig. 3, task context establishes their local objectives, directed peer messages support coordination, and local reports provide mission-progress feedback. The figure depicts n body-bound agents alongside a four-UAV UE4 scene, with front, left, right, rear, and downward camera views shown in the lower-left panel.

Body-bound local decisions. Following role assignment and skill activation in Sec. III-A, agent k is bound to UAV $u _ { k }$ and receives the active skill subset $\mathcal { L } _ { k }$ . Its context $c _ { k } ^ { t }$ contains the assigned role, local goal, mission constraints, skill documentation, and an isolated interaction history. At round $t ,$ the agent selects a parameterized skill invocation

$$
a _ { k } ^ { t } = \pi _ { \mathrm { s e m } } ( c _ { k } ^ { t } , o _ { k } ^ { t } , \{ m _ { \ell  k } ^ { t - 1 } \} _ { \ell \in \mathcal { N } _ { k } ^ { t } } , \mathcal { L } _ { k } ) ,\tag{3}
$$

where $o _ { k } ^ { t }$ is the body-local observation and $\mathcal { N } _ { k } ^ { t }$ identifies the currently reachable peers.

Directed peer communication. Communication supplements body-local sensing with information from reachable peers. The neighborhood $\mathcal { N } _ { k } ^ { t }$ contains other UAVs within the configured communication radius $r _ { \mathrm { c o m m } }$ and changes as the vehicles move. A message $m _ { k  \ell } ^ { t }$ names its sender and recipient and carries role status, local observations, intended motion, or a request for peer state. Received messages enter the recipient’s next-round decision context without transferring execution authority between agents.

Execution and progress feedback. Each world round refreshes local observations and communication neighborhoods, runs the agents’ decisions concurrently, and applies their selected invocations to the bound UAVs. In the pursuit runtime, each UAV retains its previous velocity until its own agent selects a new direction.

The Commander supplies task context and monitors mission progress, with no flight authority. Local role reports follow a single supervisory path through the Commander and Mission Console to the user. The local execution records supply role-indexed experience for the online reinforcement mechanism in Sec. III-C.

![](images/96955ec4b3a73f775d4d9aeb06f7363c57ecc06d775998a03d30324b58a82ee5.jpg)  
Fig. 4. Experience-guided online reinforcement. Retrieved reward evidence adjusts skill selection (upper stage), while executed transitions update role-indexed memory (lower stage).

## C. Experience-Guided Online Reinforcement

Execution experience provides reward evidence for adapting the skill preferences of local agents. As shown in Fig. 4, AeroWeaver combines an online update that adjusts current selector scores (upper stage) with a memory update that associates executed skills with their observed returns (lower stage). Dashed connectors denote execution feedback and subsequent experience reuse. The resulting reinforcement operates over the active skill set and accumulates across decision rounds through shared swarm memory.

Online update. For a local decision, let ρ denote the agent’s role, s its current local state, and $\mathcal { L }$ the active skill set supplied by the harness. The state summarizes local observations, received peer messages, execution progress, and relevant mission context. Each callable skill–parameter pair receives a single-token label; its provider-reported firsttoken log probability defines $z _ { i }$ . Parameterizations have separate base scores but share a skill advantage. Up to 20 alternative scores are requested. A corrected choice is used only when omitted candidates cannot exceed it under the returned probability bound; otherwise the highest observed base score is retained. Memory retrieval filters reusable records by exact task and role, then retains up to 32 by Jaccard overlap of tokenized state descriptions, breaking ties by recency. Baseline and per-skill returns are unweighted means; different bodies may contribute under the same role.

Let $\mathcal { N } _ { \rho } ( s )$ denote the retrieved neighborhood, $\bar { G } _ { \rho } ( s )$ its mean return, and $\bar { G } _ { i , \rho } ( s )$ the mean return of records associated with skill $\sigma _ { i }$ . For the current role, the estimated skill advantage is

$$
\widehat { A } ( s , \sigma _ { i } ) = \bar { G } _ { i , \rho } ( s ) - \bar { G } _ { \rho } ( s ) .\tag{4}
$$

A positive estimate raises the candidate’s score, while a negative estimate lowers it. When no matching experience is available for a candidate, its correction is set to zero.

The additive correction adjusts the scores of active skills:

$$
z _ { i } ^ { \prime } = z _ { i } + \beta \widehat { A } ( s , \sigma _ { i } ) , \qquad \beta \geq 0 ,\tag{5}
$$

where $\beta$ controls the influence of retrieved reward evidence on the selector’s initial preference. The agent selects the highest-scoring skill $\sigma _ { t }$ from the updated ranking and dispatches its invocation through the existing body-bound execution interface. The candidate set remains L throughout this update, and $\beta = 0$ recovers the base selector.

Memory update. Executing the selected skill produces a transition summarized by $\left( \rho _ { t } , s _ { t } , \sigma _ { t } , a _ { t } , r _ { t } \right)$ in the lower part of Fig. 4. Here $\sigma _ { t }$ identifies the skill, $a _ { t }$ is its concrete invocation with execution parameters, and $r _ { t }$ is the taskenvironment reward for the executing participant. The reward is stored unchanged, with benchmark-specific task and reward definitions given in Sec. IV. Rewards are accumulated along that participant’s trajectory to associate each skill choice with its subsequent outcomes,

$$
G _ { t } = \sum _ { u = t } ^ { T } \gamma ^ { u - t } r _ { u } ,\tag{6}
$$

where $\gamma \in \ [ 0 , 1 ]$ is the discount factor and $T$ is the last observed transition in the stored trajectory segment. As additional rewards arrive, the returns of earlier decisions are extended and are finalized when the trajectory ends. Each online selection therefore uses only reward evidence already observed before that decision.

Memory stores each outcome as a role-indexed record $\left( \rho _ { t } , s _ { t } , \sigma _ { t } , a _ { t } , r _ { t } , G _ { t } \right)$ . Vehicle and trajectory identifiers retain the provenance of each record, while role and state determine its relevance to later queries. This cycle supports trainingfree adaptation through accumulated execution experience while keeping model parameters, skill definitions, and flight executors fixed.

## IV. EXPERIMENTS

The evaluation covers nine closed-loop tasks, two LLM baselines, and three component controls. The comparison

covers 54 task–condition pairs with 10 episode-return observations per pair.

## A. Experimental Setup

Environment and implementation. We evaluate nine MPEinspired tasks [29], illustrated in Fig. 5. Coverage and circular/line formation assess spatial coordination, while guided navigation, private communication, and world communication involve role-dependent information exchange. Pursuitevasion tests coordinated pursuit, collection-delivery requires cooperative collection and delivery, and goal concealment combines target reaching with hiding the goal from an adversary. Task rewards are computed using MPE2 scenario functions evaluated on our simulation runtime states.

All methods share an AirSim [30] simulation runtime, task rewards, and fixed opponent policies. The area is 140×140 m at fixed altitude; default neighbor-sensing and communication ranges are 45 m and 55 m, with a 100 m sensing range for the world-communication leader. Local observations expose own state, visible entities, received messages, and callable skill–parameter pairs; private information remains role-restricted. Agents select from a frozen round state, then the world advances by 0.5 s. Invalid invocations produce an own-body hold. The evaluation uses 22 task skills; the deployment catalog in Fig. 1 additionally contains platformspecific capabilities. We use DeepSeek V4-flash [31] as the LLM backbone for the main comparisons. Experience correction uses $\gamma = 0 . 9 5$ and $\beta = 0 . 8$ , without return or advantage rescaling.

Episodes run for at most 24 rounds, with task-specific early termination. Coverage, navigation, and formation episodes end after three rounds within 3 m of all targets; pursuit requires two pursuers within 6 m of the evader. Private communication ends after three rounds, collection after all deliveries, and world communication and concealment at the horizon.

Baselines. We compare AeroWeaver with two LLM-based baselines. Centralized is an in-house planner that aggregates permitted team observations and directly selects skill calls for all controlled participants at each step. HMAS-2, adapted to our tasks from [32], first generates a central plan; local agents review their assigned actions using local observations, and the central planner revises the plan once based on their feedback before execution. Both baselines receive all 22 skills without reward memory, sharing AeroWeaver’s action constraints and skill executors. Component ablations separately disable skill activation, coordination reports, or reward correction.

Evaluation metrics. Task performance is measured by the undiscounted episode return, $\begin{array} { r } { R \ = \ | C | ^ { - 1 } \sum _ { i \in C } \sum _ { t = 1 } ^ { T } r _ { i , t } , } \end{array}$ where C is the set of controlled agents and $T$ is the episode length. We report the mean and sample standard deviation of raw returns over 10 episodes per task–condition pair. Inference cost is measured by provider-reported token consumption per episode, including decision calls and, where applicable, one full skill-activation setup.

![](images/a75558c83ae9202bc270066e810fda98aeff45ca7580b26aac0ea95051b0e42b.jpg)  
Fig. 5. MPE-inspired task forms and roles. Nine tasks span spatial coordination, communication, and cooperative–competitive interaction; solid and dashed arrows denote motion and information exchange.

TABLE I  
END-TO-END EPISODE REWARDS (MEAN ± SD; n = 10).
<table><tr><td>Task</td><td>Centralized</td><td>HMAS-2</td><td>AeroWeaver</td></tr><tr><td>Coverage</td><td> $- 1 4 . 8 1 \pm 1 . 0 2$ </td><td> $- 1 4 . 8 1 \pm 1 . 0 0$ </td><td> $\mathbf { - 1 4 . 2 0 \pm 0 . 9 5 }$ </td></tr><tr><td>Pursuit-evasion</td><td> $9 0 . 0 0 \pm 5 . 0 0$ </td><td> $7 0 . 0 0 \pm 4 . 6 0$ </td><td> $\mathbf { 1 3 0 . 0 0 \pm 4 . 8 0 }$ </td></tr><tr><td>Guided navigation</td><td> $- 8 . 5 3 \pm 0 . 3 6$ </td><td> $- 8 . 5 3 \pm 0 . 3 5$ </td><td> ${ \bf - 7 . 1 0 \pm 0 . 3 2 }$ </td></tr><tr><td>Private communication</td><td> $- 2 . 0 0 \pm 0 . 1 1$ </td><td> $- 2 . 0 0 \pm 0 . 1 0$ </td><td> $\mathbf { - 1 . 8 5 \pm 0 . 0 9 }$ </td></tr><tr><td>Circular formation</td><td> $- 8 . 4 5 \pm 0 . 1 2$ </td><td> $- 8 . 4 5 \pm 0 . 1 1$ </td><td> $\mathbf { - 8 . 2 0 \pm 0 . 1 0 }$ </td></tr><tr><td>Line formation</td><td> $- 1 3 . 9 9 \pm 0 . 2 7$ </td><td> $- 7 . 6 2 \pm 0 . 2 3$ </td><td> $- 7 . 3 0 \pm 0 . 2 2$ </td></tr><tr><td>World communication</td><td> $1 2 9 . 0 0 \pm 5 . 3 0 $ </td><td> $1 1 5 . 0 0 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $\mathbf { 1 3 5 . 0 0 \pm 5 . 2 0 }$ </td></tr><tr><td>Collection-delivery</td><td> $3 3 . 9 1 \pm 0 . 8 5$ </td><td> $3 8 . 7 0 \pm 0 . 7 8 $ </td><td> $\mathbf { 4 9 . 5 0 \pm 0 . 6 8 }$ </td></tr><tr><td>Goal concealment</td><td> $- 1 . 0 6 \pm 0 . 1 4$ </td><td> $- 0 . 4 1 \pm 0 . 1 0$ </td><td> $\mathbf { - 0 . 2 5 \pm 0 . 1 1 }$ </td></tr></table>

## B. Main Results

Task performance. AeroWeaver achieves the highest mean episode reward on all nine tasks in Table I, outperforming Centralized and HMAS-2 across spatial coordination, communication, and cooperative–competitive interaction. Centralized aggregates permitted team observations and the full skill catalog into a long context, then jointly assigns skills and body-specific parameters in one response, increasing the information and assignment burden of each decision. HMAS-2 adds local review, but its advantage over Centralized varies across tasks; neither baseline uses reward memory. AeroWeaver instead organizes selection around task-relevant skills and each agent’s executable options, reducing the number of capabilities and role-specific assignments considered in each decision. Peer reports inform complementary actions, while retrieved reward evidence updates skill preferences as execution proceeds. The consistently higher returns are compatible with this combination of focused selection, distributed coordination, and experience feedback, which connects mission objectives to local action choices throughout an episode.

![](images/74ec356672b8fba0e902acc8b6911b9cddf6ddf2c9a1468d0d84530db7a565d0.jpg)  
Fig. 6. Token consumption. Per-episode totals for Centralized, HMAS-2, and AeroWeaver include activation setup and decision calls; lower is better.

Token efficiency. As shown in Fig. 6, AeroWeaver consumes 52.0–70.0% fewer tokens than HMAS-2 across the nine tasks and fewer tokens than Centralized on four tasks. Task-conditioned activation limits the skill documentation processed at each decision, while local action selection eliminates the central planning and revision calls required by HMAS-2. Relative to Centralized, the shorter local contexts are offset by separate calls for individual agents, making the overall token advantage task-dependent. The reported totals include activation setup and all decision calls from one episode per condition, with task-dependent termination and episode lengths.

## C. Ablation Studies

Table II evaluates the removal of skill activation, coordination reports, and reward correction. The controls respectively expose the full catalog, suppress coordination reports while preserving task-required messages, and set $\beta = 0$ . For task t and variant m, the reported relative mean score is

$$
Q _ { t , m } = 1 + \frac { \bar { R } _ { t , m } - \bar { R } _ { t , \mathrm { f u l l } } } { K _ { t } } , \qquad K _ { t } > 0 ,\tag{7}
$$

where $\bar { R }$ is the mean episode return and $K _ { t }$ is the fixed task-specific scale of the existing affine score transformation, shared by all variants. AeroWeaver therefore scores 1, and lower scores indicate lower returns. Table II lists the scales, so raw means are recovered as $\bar { R } _ { t , m } = \bar { R } _ { t , \mathrm { f u l l } } + K _ { t } ( Q _ { t , m } - 1 )$

Removing coordination reports produces the largest decrease in five tasks: guided navigation, private communication, world communication, collection-delivery, and goal concealment. The remaining four tasks show the largest decrease when skill activation is removed. Goal concealment is particularly sensitive to coordination reports, with a score of 0.605 compared with 0.777 without activation and 0.780 without reward correction. Disabling reward correction yields scores of 0.764–0.874 and is the least detrimental ablation in eight tasks; line formation is the exception, where removing coordination reports retains a higher score. These comparisons indicate task-dependent contributions from activation and coordination, with reward correction providing an additional improvement across the suite.

TABLE II  
COMPONENT ABLATIONS ACROSS NINE TASKS.
<table><tr><td>Task</td><td>w/o skill activation</td><td>w/o coordination reports</td><td>w/o reward correction</td></tr><tr><td>Coverage</td><td>0.650</td><td>0.708</td><td>0.764</td></tr><tr><td>Pursuit-evasion</td><td>0.670</td><td>0.781</td><td>0.823</td></tr><tr><td>Guided navigation</td><td>0.734</td><td>0.632</td><td>0.802</td></tr><tr><td>Private communication</td><td>0.812</td><td>0.617</td><td>0.874</td></tr><tr><td>Circular formation</td><td>0.762</td><td>0.763</td><td>0.825</td></tr><tr><td>Line formation</td><td>0.722</td><td>0.821</td><td>0.798</td></tr><tr><td>World communication</td><td>0.790</td><td>0.633</td><td>0.854</td></tr><tr><td>Collection-delivery</td><td>0.632</td><td>0.628</td><td>0.814</td></tr><tr><td>Goal concealment</td><td>0.777</td><td>0.605</td><td>0.780</td></tr></table>

Relative mean scores (n = 10 per condition), normalized to AeroWeaver within each task; the full model has a reference score of 1. K in row order: 17.077, 17.606, 15.029, 15.969, 16.801, 14.280, 35.217, 10.881, and 13.505.

## D. Additional Experiments

The additional studies address experience reuse, dependence on the LLM backbone, and sensitivity to mission wording. Fig. 7(a)–(c) illustrates the measured results.

(a) Experience-guided adaptation. Fig. 7(a) shows rapid early improvement followed by gradual stabilization across the nine tasks. This trend is consistent with experienceguided online adaptation, as accumulated reward information refines subsequent skill selection without changing model weights or skill executors.

(b) Backbone sensitivity. In Fig. 7(b), GPT-5.6 Luna and GLM-5.3-Flash remain close to the DeepSeek-V4- Flash reference of 1 across the nine tasks, while qwen3.8- flash exhibits a moderate performance decrease. Overall, AeroWeaver maintains relatively stable performance across different LLM backbones, demonstrating robustness to backbone choice despite task-dependent variations.

(c) Prompt perturbation sensitivity. The prompt comparison varies only the mission text, keeping the activated skill set fixed. Paraphrasing preserves task meaning, reordering reverses sentence order, and distractor insertion appends unrelated report metadata without changing task requirements. With Original normalized to 1 within each task, Fig. 7(c) depicts smaller score reductions under paraphrasing and reordering than under distractor insertion, illustrating a stronger response to irrelevant context than to changes in wording or order.

## V. CONCLUSION

We introduced AeroWeaver, an embodied-agent harness for UAV swarms that integrates task-conditioned skill activation, distributed body-local coordination, and role-indexed experience reuse within a unified execution framework. Reward-derived evidence is incorporated into subsequent skill selection without modifying the underlying model parameters or low-level flight controllers. Across the nine evaluated tasks, AeroWeaver achieved higher mean episodic returns than Centralized and HMAS-2. It also required fewer tokens than HMAS-2 in the recorded episodes, while its efficiency relative to Centralized varied across tasks. The ablation results further show that skill activation, coordination reports, and reward-based correction contribute to overall performance.

![](images/d27845651c911862a5296eb84e55c3c5baac71454b9d1c5fda87379e2c9a03a9.jpg)  
(a) Experience-guided adaptation

![](images/4e1fa9e89c2dbfd8131cfe28f14372a513681b6dc3e59da9991af3fa9faf2833.jpg)  
(b) Backbone sensitivity

![](images/36fe6acdace48bd2554277d1ede7b15ea6b9a1dcc2366da9f9dcedea784abaf8.jpg)  
(c) Prompt perturbation sensitivity  
Fig. 7. Additional experiments. Panels illustrate adaptation trajectories and task-wise comparisons of LLM backbones and prompt perturbations.

The current evaluation is conducted with a predefined skill catalog and simulated execution conditions. Future work will extend AeroWeaver toward longer-term experience reuse, broader task transfer, and validation on physical UAV platforms.

## REFERENCES

[1] A. Iannoli, L. Gigli, L. Sciullo, A. Trotta, and M. Di Felice, “Say the mission, execute the swarm: Agent-enhanced LLM reasoning in the Web-of-Drones,” in Proc. IEEE Int. Symp. World Wireless, Mobile Multimedia Netw. (WoWMoM), 2026, pp. 139–148.

[2] P. Li, Z. An, S. Abrar, and L. Zhou, “Large language models for multirobot systems: A survey,” Auton. Robots, vol. 50, no. 3, 2026, Art. no. 30.

[3] S. Yao et al., “ReAct: Synergizing reasoning and acting in language models,” in Proc. ICLR, 2023.

[4] B. Ichter et al., “Do as I can, not as I say: Grounding language in robotic affordances,” in Proc. Conf. Robot Learn. (CoRL), ser. PMLR, vol. 205, 2023, pp. 287–318.

[5] J. Liang et al., “Code as policies: Language model programs for embodied control,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 9493–9500.

[6] W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei, “ReKep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation,” in Proc. Conf. Robot Learn. (CoRL), ser. PMLR, vol. 270, 2025, pp. 4573–4602.

[7] G. Wang et al., “Voyager: An open-ended embodied agent with large language models,” Trans. Mach. Learn. Res., 2024.

[8] A. Zhao, D. Huang, Q. Xu, M. Lin, Y.-J. Liu, and G. Huang, “ExpeL: LLM agents are experiential learners,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 17, 2024, pp. 19 632–19 642.

[9] Z. Z. Wang, J. Mao, D. Fried, and G. Neubig, “Agent workflow memory,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 267, 2025, pp. 63 897–63 911.

[10] W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang, “A-Mem: Agentic memory for LLM agents,” in Adv. Neural Inf. Process. Syst., vol. 38, 2025, pp. 20 004–20 031.

[11] D. Fu et al., “AgentRefine: Enhancing agent generalization through refinement tuning,” in Proc. ICLR, 2025.

[12] Z. Xi et al., “AgentGym: Evaluating and training large language model-based agents across diverse environments,” in Proc. Annu. Meeting Assoc. Comput. Linguistics (ACL), vol. 1, 2025, pp. 27 914– 27 961.

[13] Y. Li et al., “Just-in-time reinforcement learning: Continual learning in LLM agents without gradient updates,” 2026, arXiv:2601.18510.

[14] E. Debenedetti, J. Zhang, M. Balunovic, L. Beurer-Kellner, M. Fischer, and F. Tramer, “AgentDojo: A dynamic environment to evaluate\` prompt injection attacks and defenses for LLM agents,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 82 895–82 920.

[15] H. Zhang et al., “BadRobot: Jailbreaking embodied LLM agents in the physical world,” in Proc. ICLR, 2025.

[16] Q. Wang et al., “Towards the harness of embodied agents,” 2026, arXiv:2608.11246.

[17] P. Wang et al., “Self-evolving embodied agents via skill-harness evolution,” 2026, arXiv:2608.11350.

[18] S. S. Kannan, V. L. N. Venkatesh, and B.-C. Min, “SMART-LLM: Smart multi-agent robot task planning using large language models,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2024, pp. 12 140–12 147.

[19] J. Liu et al., “CaPo: Cooperative plan optimization for efficient embodied multi-agent cooperation,” in Proc. ICLR, 2025.

[20] J. Chen et al., “EMOS: Embodiment-aware heterogeneous multi-robot operating system with LLM agents,” in Proc. ICLR, 2025.

[21] H. Zhang et al., “Building cooperative embodied agents modularly with large language models,” in Proc. ICLR, 2024.

[22] Z. Mandi, S. Jain, and S. Song, “RoCo: Dialectic multi-robot collaboration with large language models,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 286–299.

[23] M. Chang et al., “PARTNR: A benchmark for planning and reasoning in embodied multi-agent tasks,” in Proc. ICLR, 2025.

[24] A. Lykov et al., “FlockGPT: Guiding UAV flocking with linguistic orchestration,” in Proc. IEEE Int. Symp. Mixed Augmented Reality Adjunct (ISMAR-Adjunct), 2024, pp. 485–488.

[25] J. Lou, R. Shi, Y. Lin, Q. Wang, and W. Wu, “TALKER: A taskactivated language model based knowledge-extension reasoning system,” IEEE Robot. Autom. Lett., vol. 10, no. 2, pp. 1026–1033, 2025.

[26] J. Lou et al., “Agents Trainer: Automatically training multi-agent reinforcement learning models for drone swarm using language modelbased agents,” IEEE Trans. Autom. Sci. Eng., vol. 23, pp. 8992–9006, 2026.

[27] J. Lou, H. Wang, X. Liu, Y. Zhang, R. Shi, and W. Wu, “AERIS: Aerial-edge role-driven intelligence at runtime via orchestrated language-model swarm,” 2026, arXiv:2606.30151.

[28] H. Lin, Q. Yang, and C. Qin, “Skill-MAS: Evolving meta-skill for automatic multi-agent systems,” 2026, arXiv:2606.18837.

[29] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, “Multi-agent actor-critic for mixed cooperative-competitive environments,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017.

[30] S. Shah, D. Dey, C. Lovett, and A. Kapoor, “Airsim: High-fidelity visual and physical simulation for autonomous vehicles,” in Field and Service Robotics, M. Hutter and R. Siegwart, Eds. Cham: Springer International Publishing, 2018, pp. 621–635.

[31] DeepSeek-AI, “DeepSeek-V4: Towards highly efficient million-token context intelligence,” 2026, arXiv:2606.19348.

[32] Y. Chen, J. Arkin, Y. Zhang, N. Roy, and C. Fan, “Scalable multi-robot collaboration with large language models: Centralized or decentralized systems?” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 4311–4317.