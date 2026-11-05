# Autonomy and Coordination in Intelligent Manufacturing: A Recursive Autonomous Agent Framework

Yuning Chen<sup>1,†</sup>, Hongda Zhang<sup>1</sup>, Linqiang Hu<sup>1</sup>, Ziqing Zhou<sup>1</sup>, Yi Liu<sup>1</sup>, Zhao Huang<sup>2</sup>, Ding Zhou<sup>2</sup>, Haining Xu<sup>2,\*</sup>

<sup>1</sup>Fudan University <sup>2</sup>Jiangsu Chuangling Intelligent Technology Co., Ltd.

## Abstract

Intelligent production requires systems to make decisions and organize work according to production goals and operating conditions. It also requires continuous adjustment through feedback. This paper analyzes intelligence in human production organizations from a complex adaptive systems (CAS) perspective. The analysis identifies two coupled mechanisms: individual autonomy and collective coordination. Individual autonomy enables actors to form local decisions using available information. Collective coordination aligns these decisions with shared goals and joint constraints. Learning and adaptation improve both mechanisms over time. A production-system model formalizes how local decision rules and coordination jointly form system decisions. On this basis, we propose the Recursive Autonomous Agent (RAA) framework, which combines goal-driven autonomous agents with a recursive organizational structure. Goal contracts define local objectives and operating boundaries, while capability con tracts support the selection and composition of production units. Common interfaces allow coordinated groups to participate as units in larger organizations, connecting equipment, production lines, and factories. RAA thus provides an engineering approach to next-generation intelligent manufacturing, translating the requirements of intelligence, openness, and system-wide integration into mechanisms for autonomous and coordinated production.

Keywords: intelligent manufacturing; complex adaptive systems; autonomous agent; Recursive Autonomous Agent

## 1 Introduction

A factory must turn changing orders into products that meet delivery, quality, and safety requirements. It must also deal with changes in equipment and resources. A fixed program can handle repeated operations under familiar conditions. A new order, a failed machine, or a material delay may require a diferent plan. People then have to assess the situation and organize a response. This need becomes more pressing as product variety grows and production batches become smaller [6, 7]. Digital systems make production easier to describe and monitor. Networks make information available across equipment and departments. Digital twins strengthen the link between physical operations and their digital representations [9]. These capabilities support decisions, but access to information does not itself decide how production should proceed. The practical question is how a system can use the information it has to choose actions and revise them as conditions change.

The question has both a local and a collective side. A workstation needs to understand its task, select suitable operations, and check the result. A production line must also manage shared resources and dependencies between workstations. A locally sensible choice can disrupt another task. The ability of each unit to act and the ability of the organization to coordinate those actions must therefore be developed together. Human production organizations ofer a useful reference. An operator observes conditions at a station and adjusts work within an assigned responsibility. A team leader handles issues that afect several stations. A manager connects these arrangements to broader production objectives. Problems and results move upward, while goals and work assignments move downward. This organization combines independent judgment with coordinated action. It provides a functional reference for engineering, rather than a requirement to reproduce human cognition.

We examine this reference through complex adaptive systems (CAS) theory [2, 4]. CAS focuses on interacting actors that adapt using local information and feedback. It also draws attention to outcomes that depend on how actors interact. This perspective connects the decisions made by production units to the performance of the organization as a whole. Our central account has two parts. Individual autonomy explains how production actors form and revise local decisions. Collective coordination explains how these decisions become joint action under shared constraints. Learning and adaptation develop both mechanisms. On this basis, we present the Recursive Autonomous Agent (RAA) framework. It combines goal-driven agents with a recursive structure for organizing their work.

The paper makes two contributions. First, it develops a CAS-based account of production intelligence and expresses the roles of autonomy and coordination in a production-system model. Second, it maps these roles to engineering requirements and a recursive agent framework. The design links local execution to broader production goals through contracts, feedback, and coordination across levels.

Section 2 explains the two mechanisms. Section 3 gives their mathematical description. Section 4 develops the engineering framework and its workflow. Section 5 relates the design to manufacturing paradigms and their practical requirements. Section 6 concludes the paper.

## 2 CAS Analysis of Production Intelligence

This section examines intelligence in human production organizations from a complex adaptive systems (CAS) perspective. The analysis identifies two coupled mechanisms: individual autonomy and collective coordination. Learning and adaptation improve both mechanisms over time. The following subsections explain why these mechanisms are needed and specify the functional requirements for their engineering implementation.

## 2.1 Individual Autonomy

Production information is distributed across diferent roles and organizational levels. Operators know the current condition of workpieces and equipment at their stations. Team leaders track progress across their teams. Factory managers oversee production as a whole, but cannot monitor every local detail in real time. Observations can also be delayed or inaccurate. A decision process that waits for a complete global picture may respond too slowly to changes on the shop floor.

Within the production organization considered here, actors use the information available where they work. They need enough authority to make local decisions and correct local problems. Their actions remain subject to assigned goals and operating boundaries. An issue outside those boundaries must be passed to a level that can address it.

A skilled worker illustrates the required cycle. The worker checks the equipment and workpiece, decides how to perform the operation, carries it out, and inspects the result. Feedback informs the next action. If a deviation can be corrected locally, the worker makes the correction. Otherwise, the issue is reported for coordination. The engineering requirement is a similar connection between observation, decision, execution, and feedback.

Individual autonomy is the ability of a production actor to use local information to pursue an assigned goal within defined boundaries. The actor maintains a local feedback loop and handles execution deviations within its authority.

Four requirements follow from this definition: AR1: Goal understanding. The actor receives and interprets its assigned objective.

AR2: Local closed loop. Observation, decision, execution, and feedback remain connected during operation.

AR3: Deviation handling. The actor corrects local execution problems within its authority.

AR4: Controlled learning. The actor improves its decisions through experience without bypassing operating boundaries.

These requirements concern more than the ability to issue an action. They connect the assigned goal to the observed result. They also distinguish a correction made during execution from a change to the model or policy used in later work.

## 2.2 Collective Coordination

Autonomous action alone does not organize a production system. Two units may need the same machine at the same time. A downstream station may depend on an upstream task that has not finished. A handof may fail even when both stations can perform their own operations. These dependencies make the relationship between actions part of the decision problem.

There are two reasons to coordinate. The first is joint feasibility. An action can be allowed at each station in isolation but disallowed when the stations act together. The second is the dependence of overall results on interaction. Having every machine available does not guarantee timely production. Task order, resource allocation, and handofs can change the outcome of the whole process.

Collective coordination is the ability of several autonomous actors to form jointly feasible system decisions in pursuit of a shared goal. It establishes the relationships needed for joint work and adjusts or ends them when conditions change.

This mechanism has four corresponding requirements:

CR1: Shared goals. The organization identifies a common objective and explains how local tasks contribute to it.

CR2: Capability matching. It brings together actors with suitable capabilities and specifies how they will work together.

CR3: Joint decisions. It coordinates local proposals into feasible system action and supports their execution.

CR4: Adaptive coordination. It uses feedback to assess collective results and revise the organization of work.

Coordination therefore includes both preparing an organization and managing its operation. Suitable participants must first be found. Their proposed actions must then satisfy shared constraints. Execution feedback may require changes to task assignments, resources, or coordination relationships.

![](images/63eaccdc24ce866192350e6a0d6c83596d56f1b4c3f85d1e8db986794b44bba6.jpg)  
Figure 1. Two connected decision processes. Each actor uses the local process shown on the left. Individual autonomy forms local proposals through a feedback loop. Collective coordination checks their combination against shared goals and joint constraints. Execution results support local recovery, controlled learning, and changes to coordination.

## 2.3 Coupled Mechanisms

The two mechanisms answer diferent questions. Individual autonomy asks what an actor should do using the information it has. Collective coordination asks how the proposed actions can work together. Figure 1 shows the connection. Local decisions become proposals for joint work. Shared constraints determine which combinations can be executed. The resulting feedback informs both the actors and their organization.

Without individual autonomy, a central system would have to guide each local response and could become slow when conditions change. Without collective coordination, actors could compete for resources or pursue incompatible local objectives. A complete account of production decisions must cover both the actors’ internal activity and their relationships.

At this level of description, the paper treats autonomy and coordination as a minimal and complete pair. The first covers the internal path from a local goal to an outcome. The second covers the relationships needed for a joint outcome. Sensing, planning, execution, and learning belong to the actor’s activity. Task allocation, negotiation, shared resources, and feedback aggregation concern relationships among actors. Learning and adaptation operate within both parts of this account. An actor can improve a local model or decision rule. An organization can improve participant selection or change a coordination relationship. These processes develop autonomy and coordination over time; they are not introduced here as a third mechanism alongside them.

## 3 Production Decision Model

The preceding account can be expressed in a common model. The model identifies the production state, the decisions available to the system, and the connection between local and collective decisions. It describes the roles that an implementation must realize.

## 3.1 System Formulation

Let a production system be

$$
\mathcal { P } = \langle \mathcal { A } , \mathcal { T } , \mathcal { R } , \mathcal { E } , \mathcal { X } , \mathcal { U } , \mathcal { D } , \mathcal { G } , \mathcal { B } , \Phi \rangle .\tag{1}
$$

Here, A denotes production actors, T denotes tasks, and R denotes resources. The set E describes relationships such as assignments and process dependencies. The remaining terms describe operation: X is the state space, U is the decision space, and D is the disturbance space. Goals belong to G, and operating boundaries belong to B. The function Φ describes state transitions.

At discrete time t, write the state as

$$
x _ { t } = \langle x _ { t } ^ { \mathrm { A } } , x _ { t } ^ { \mathrm { T } } , x _ { t } ^ { \mathrm { R } } , x _ { t } ^ { \mathrm { E } } \rangle \in \mathcal { X } .\tag{2}
$$

The four components record actor capabilities and availability, task progress, resource conditions, and current relationships. Resource conditions include location, suficiency, and occupancy. Relationships include task assignments, process dependencies, and cooperation between actors. Together, these components describe the production situation at the current time.

The system faces a goal $g _ { t } \in \mathcal G$ and boundaries $b _ { t } \in$ B. It selects a decision $u _ { t } \in \mathcal { U } .$ A disturbance $d _ { t } \in$ D, such as a machine failure or a material delay, also afects what happens next:

$$
x _ { t + 1 } = \Phi ( x _ { t } , u _ { t } , d _ { t } ; g _ { t } , b _ { t } ) .\tag{3}
$$

At this decision point, $x _ { t }$ has already arisen and the incoming disturbance is not chosen by the system. Goals and boundaries can be revised at a higher level, but they are given within the current cycle. Under these conditions, $u _ { t }$ is the quantity the system can actively select to influence its next state.

The system acts on an estimate $\hat { x } _ { t } ,$ , which may be incomplete or inaccurate. Its decision rule is

$$
\begin{array} { r l } & { u _ { t } = \Gamma ( \hat { x } _ { t } , g _ { t } , b _ { t } ) , } \\ & { u _ { t } \in \mathcal { U } ^ { \mathrm { a d m } } ( x _ { t } , b _ { t } ) . } \end{array}\tag{4}
$$

Here, Γ denotes the system-level decision rule. It maps the estimated state $\hat { x } _ { t } ,$ the production goal $g _ { t } ,$ , and the operating boundaries $b _ { t }$ to the decision $u _ { t }$ . The admissible set $\mathcal { U } ^ { \mathrm { a d m } }$ contains decisions allowed by the true state and current boundaries. Equation (4) states two requirements: form a goal-directed decision from available information, and ensure that the chosen action respects the applicable boundaries. It specifies what a decision process must achieve. The equation alone does not provide a way to guarantee admissibility under imperfect information.

In this model, production intelligence acts through decision formation and revision. Data, models, and communication support intelligence when they help choose an appropriate $u _ { t }$ and use execution feedback to improve subsequent choices.

## 3.2 Local Decisions and Joint Constraints

Actor $a _ { i }$ forms a proposed action using its local state estimate, objective, and boundaries:

$$
u _ { i , t } = \pi _ { i } ( \widehat { x } _ { i , t } , g _ { i , t } , b _ { i , t } ) , \qquad i = 1 , \ldots , n .\tag{5}
$$

The rule $\pi _ { i }$ represents local decision formation. The collection of proposals gives a candidate joint decision,

$$
\tilde { u } _ { t } = ( u _ { 1 , t } , u _ { 2 , t } , . . . , u _ { n , t } ) .\tag{6}
$$

These proposals are the local components from which the system decision is formed. Their direct combination is not necessarily executable.

Let $\mathcal { U } _ { i } ( x _ { i , t } )$ contain actions feasible for actor $a _ { i }$ in its local state. The Cartesian product of these sets contains all combinations of locally feasible actions. If shared constraints exclude some combinations, then the jointly feasible set satisfies

$$
\mathcal { U } ^ { \mathrm { J } } ( x _ { t } ) \subset \prod _ { i = 1 } ^ { n } \mathcal { U } _ { i } ( x _ { i , t } ) .\tag{7}
$$

For example, each of two stations may be able to use one resource, while simultaneous use is forbidden. Task precedence and spatial conflicts create similar restrictions. The final system action must account for these constraints rather than simply adopting $\tilde { u } _ { t }$

Overall outcomes also depend on the combination of actions. If $y _ { i , t }$ is a local outcome and $y _ { t }$ is the system outcome, in general $y _ { t } \ \ne \ \textstyle \sum _ { i } y _ { i , t }$ . Separate improvements at each station do not necessarily yield a corresponding production-line improvement. A poor handof or a resource conflict may ofset them. This dependence on interaction is the production meaning of emergence used in the paper.

## 3.3 System Decision Rule

Collect the local rules in π and use $\mu$ to represent coordination:

$$
\begin{array} { c } { { \pi = ( \pi _ { 1 } , \pi _ { 2 } , . . . , \pi _ { n } ) , } } \\ { { { \tilde { \bf J } } _ { \mathrm { P } } = \langle \pi , \mu \rangle , \qquad \Gamma = \mu \circ \pmb { \pi } . } } \end{array}\tag{8}
$$

The pair ${ \mathfrak { I } } _ { \mathrm { P } }$ expresses the two mechanisms. Local rules produce candidate actions; coordination determines how they form a system decision under shared constraints. The composition Γ is the rule in Equation (4). It connects local judgment to collective action.

Learning can change the local rules in π. Feedback about joint outcomes can also change $\mu ,$ for example by informing task allocation or coordination protocols. This gives a common description of improvement at the individual and organizational levels.

## 4 RAA Framework

The model separates two engineering tasks: building units that can act autonomously, and organizing those units into a coordinated system. RAA addresses both through agents that expose compatible interfaces at different levels.

The Recursive Autonomous Agent (RAA) framework organizes intelligent production through goal-driven agents operating within explicit constraints. Individual agents provide local autonomy, while recursive organizations provide collective coordination.

An agent associated with equipment is called a levelone agent. A level-two agent organizes several levelone agents. The level-two agent and its members can then appear as one unit to a higher level. The pattern connects equipment to production lines and production lines to factories. It also preserves a common way to state goals, describe capabilities, and report results.

Table 1 connects the requirements in Section 2 to the engineering design. These are functional requirements. Diferent models, planners, and communication systems may be used to implement them.

## 4.1 Autonomous Agent Design

Start with a goal contract. A conventional automated device receives an action command and follows a program. In RAA, an agent receives a goal and the conditions under which it may act. The goal contract states the desired result and its acceptance criteria. It also specifies timing, priority, and the safety, quality, and authority limits that apply. This makes goal understanding operational: the agent has a result to achieve and explicit boundaries for its choices.

Table 1. From decision requirements to engineering mechanisms.
<table><tr><td>Code</td><td>Requirement</td><td>Engineering mechanism</td><td>Role in production</td></tr><tr><td>AR1</td><td>Goal understanding</td><td>Goal contract with acceptance criteria and boundaries</td><td>Define the result to achieve and the allowed scope of action</td></tr><tr><td>AR2</td><td>Local closed loop</td><td>Sensing, local model, planning, checked skill execution, and feedback</td><td>Connect local conditions to action and outcome checks</td></tr><tr><td>AR3</td><td>Deviation handling</td><td>Recovery, replanning, safe stopping, and escalation</td><td>Resolve local problems or refer them to the appropriate level</td></tr><tr><td>AR4</td><td>Controlled learning</td><td>Evaluated and approved model or policy updates</td><td>Improve capability while preserving operating boundaries</td></tr><tr><td>CR1</td><td>Shared goals</td><td>Goal decomposition and subtask acceptance</td><td>Link local work to the overall production objective</td></tr><tr><td>CR2</td><td>Capability matching</td><td>Capability contracts and participant selection</td><td>Compose suitable units without binding tasks to fixed devices</td></tr><tr><td>CR3</td><td>Joint decisions</td><td>Shared-constraint checks and coordination protocols</td><td>Manage resource use, task order, and handoffs</td></tr><tr><td>CR4</td><td>Adaptive coordina- tion</td><td>Outcome evaluation and organizational adjustment</td><td>Revise assignments, resources, or relationships as conditions change</td></tr></table>

Connect observation to verified execution. Multimodal sensing captures equipment and environmental conditions. A local world model interprets this information. Planning and scheduling turn the goal into a sequence of skills and decide when to use them. The execution module invokes those skills through equipment interfaces. Results and execution evidence return to the agent, which compares them with the intended outcome before selecting its next action.

The design separates artificial intelligence (AI) proposals from physical execution. AI can interpret a task and propose a plan. Before an action reaches equipment, it must pass capability and physical-feasibility checks. The framework does not allow probabilistic model output to drive equipment without those checks. This separation connects flexible planning to the deterministic requirements of execution.

Recover within authority. A constraint engine monitors operation. When results depart from the plan, the agent first checks whether it can recover within its authority. It may retry an operation or revise its plan. Where permitted, it can use an alternative method or continue with reduced capability. If it cannot resolve the problem within its limits, it escalates the issue or stops safely. The event and the actions taken are recorded.

Recovery has a local scope. The agent does not resolve a cross-unit problem by exceeding its authority. Instead, it supplies the information needed by the coordinating level. The distinction allows quick local responses while keeping broader decisions with the organization responsible for them.

Control changes to learned capability. Operational correction and learning run on diferent time scales. The fast process uses existing models, skills, and boundaries to correct deviations or replan. The slower process collects experience and proposes changes to a model, rule, or policy. Proposed updates pass ofline evaluation, simulation, constraint checks, and version approval before use. Experience can therefore improve later decisions without allowing the update process to bypass physical operating limits.

## 4.2 Recursive Organization

A level-two agent receives a production objective and divides it into interdependent subtasks. Each subtask has an acceptance criterion, so its contribution to the overall result can be checked. Completion is assessed through the decomposition and the associated requirements, rather than from an informal report that each participant has finished.

The organization finds participants through capability contracts. Each contract states what an agent can do, the conditions it needs, the efects and quality it can provide, and the resources it uses. The level-two agent combines this information with current availability and state. It then selects participants, assigns roles, and establishes working relationships.

The planning question is expressed in terms of required capabilities. A task does not have to remain tied to a particular device. A replacement device can participate when its interfaces are adapted and its capabilities meet the relevant contract. Capability verification remains necessary; a matching name alone is not enough to establish that two devices can perform the same work.

Level-one agents propose local actions. The level-two agent checks their combination against resource restrictions, task order, and handof requirements. It forms an executable joint decision and coordinates the resulting work. Protocols specify how participants exchange state, hand over tasks, and confirm results. These mechanisms implement the role of µ in Equation (8).

During execution, participants report their states, results, exceptions, and supporting evidence. The level-two agent uses these reports in its own observation–decision–execution–feedback loop. It assesses progress and overall outcomes. It can revise assignments or resource allocation, change participants, or adjust coordination relationships. Completed coordination records provide experience for later selection and organization decisions.

![](images/c2251bfc1222ad20acc00411a4ad594d6de5f80e619ada39b7604df9271d78bb.jpg)  
Figure 2. RAA organization and operation in a functional production-line example. Left: a level-two agent coordinates processing, inspection, and material-handling agents through compatible goal, capability, and result interfaces. Together they can be used as one unit at a higher level. Right: the execution sequence, including feedback, escalation, and controlled improvement.

Recursion follows from the external interface. Both a level-one agent and a level-two organization can receive goals and constraints, describe capabilities, and report results. A higher-level agent can therefore call on a coordinated group as one production unit. At each level, goals are decomposed downward and results are brought together upward. The same organizational pattern can be repeated at a broader production scale.

## 4.3 Operational Workflow

Figure 2 illustrates the design with three equipment tasks: processing, inspection, and material handling. Each level-one agent maintains its local feedback loop. A level-two agent coordinates their work toward a production-line objective. Personnel or an existing management system set the overall goal and constraints.

The processing agent observes the workpiece and tool, chooses an operation, and checks quality and completion. The inspection agent compares measurements with acceptance criteria and returns pass or fail evidence. The material-handling agent checks locations and path conditions. It selects a route and handof, then confirms delivery. Each agent invokes validated skills within physical operating limits.

The operation has six stages:

1. Specify the outcome. Personnel or the management system supply the production goal and its boundaries.

2. Organize the work. The level-two agent decomposes the goal, matches subtasks to capabilities, checks joint constraints, and assigns work.

3. Execute locally. Each level-one agent acts within its boundaries and reports state, progress, and evidence of the result.

4. Handle exceptions at the right level. Local deviations are corrected locally. Issues afecting coordination go to the level-two agent. Matters beyond that agent’s authority go to personnel.

5. Check completion. The level-two agent brings together subtask results and acceptance evidence. It checks the overall production requirements and reports the outcome.

6. Improve under control. Execution and coordination records inform evaluated updates to individual agents and the organization.

This arrangement connects management decisions to physical action through feedback at each level. People set goals and boundaries, resolve escalated issues, and make high-risk decisions. They need not continuously direct every permitted local adjustment.

## 5 Advancing Intelligent Manufacturing

This section discusses the role of the proposed RAA framework in next-generation intelligent production. It places RAA within the evolution of manufacturing

![](images/5a04e6776e832c5379c1e9a9f5c0717d61e4270eb6c258e4150d75fd85398720.jpg)  
Figure 3. The three panels show the primary contributions of autonomous feedback, capability contracts, and recursive coordination to intelligence, openness, and system-wide integration, respectively.

paradigms and explains how its mechanisms support intelligence, openness, and system-wide integration.

## 5.1 Manufacturing Paradigms

The design addresses a broader development in manufacturing. Early mechanization and electrification increased the physical capacity of equipment. Computer and numerical-control technologies later made it possible to encode and repeat production operations. Digital tools also expanded into design and management. Computer-aided design (CAD) represented product information. Enterprise resource planning (ERP) supported the planning of orders, materials, and resources. Manufacturing execution systems (MES) organized and tracked shop-floor work [15]. These developments improved individual activities, even when information remained separated between systems.

Networks extended information exchange across equipment, departments, and enterprises. The Industrial Internet of Things, cyber-physical systems, and digital twins further connected field conditions to planning and control [9, 14]. Such connections make coordinated work possible, but they do not remove the need for a decision process when circumstances fall outside a fixed plan.

Zhou et al. [14] and Zang et al. [12] describe three basic paradigms: digital manufacturing, digital and networked manufacturing, and new-generation intelligent manufacturing. The first embeds existing knowledge in software and programs. The second connects information and resources. The third integrates newgeneration AI with manufacturing so that systems can learn, apply knowledge, and adapt decisions to operating conditions.

The first two paradigms leave two related dificulties. Separate digital systems may use incompatible formats or interfaces. People then transfer information and translate plans between them. Connected systems can reduce this separation, but a revised management plan may still fail to produce a timely change in equipment behavior [10, 5]. A system may detect a failed machine without being able to reorganize the afected work.

Recent work on large language model (LLM) agents illustrates ways to strengthen the link from production requirements to action. Xia et al. [11] connect task descriptions to planning and equipment skill invocation in a modular production setting. Fan et al. [1] examine LLM support for autonomous industrial robotics. These developments motivate a framework that connects local decision capability to the organization of multiple production units.

## 5.2 Alignment with Manufacturing Requirements

Intelligence, openness, and system-wide integration are key characteristics in Zhou’s account of next-generation intelligent manufacturing [13]. Figure 3 illustrates how the RAA framework supports intelligence, openness, and system-wide integration through its engineering mechanisms.

Intelligence. A production unit must interpret its objective, act using current conditions, and check whether the result is acceptable. When it detects a deviation, it adjusts the plan within its authority. Experience can improve later decisions through controlled updates. RAA connects these functions in the local agent. This is intended to reduce dependence on fixed programs and repeated manual intervention when conditions change [10].

Openness. An open production organization needs to know what a new unit can provide and under what conditions. Standardized descriptions, including the Asset Administration Shell, support a common representation of equipment information [3]. RAA uses capability contracts to connect such descriptions to task selection and execution requirements. Material, time, and resource needs must be understood before capabilities can be combined.

System Integration. Production units need to contribute to the overall delivery objective. They must also cooperate with activities such as design, logistics, and services [14, 5]. A station cannot plan only for its own convenience when another task depends on its output. Shared resources and handofs connect local choices to broader results. RAA carries goals downward and brings results upward through the same pattern at each level. A team-like organization resolves issues within its scope and escalates those that afect a broader group. This mirrors the functional links between operators, teams, workshops, and factories. Coordination gives local action an organizational context and helps prevent local optimization from undermining the overall task.

## 5.3 Deployment Scope

These three requirements reinforce one another. A unit that can adapt but cannot describe its capabilities is dificult to compose into larger tasks. Connected units that cannot decide locally still depend on external instructions. Autonomous units without coordination can interfere with one another. Local feedback, capability contracts, and recursive organization are intended to address these needs together.

Deployment can start with a workstation or production-line segment that ofers clear practical value. The enterprise can establish the sensing, connectivity, and decision capability needed for that task, then extend the organization using validated units. Factory-wide transformation need not be a prerequisite for every local application. This is relevant to enterprises with limited funding or technical staf [8].

## 6 Conclusion

Production intelligence concerns how a system selects actions under changing conditions and revises them through feedback. A CAS perspective brings two mechanisms into focus. Individual autonomy forms local decisions using available information. Collective coordination manages the dependencies that determine whether those decisions can work together. Learning and adaptation improve both mechanisms over time. The RAA framework turns this account into an organization of agents and coordinated groups. Goal contracts guide local work. Capability contracts support participant selection and composition. Recursive interfaces connect equipment, production lines, and factories, while feedback connects results to decisions at each level. Checked execution, local recovery, escalation, and controlled learning preserve the role of operating boundaries throughout this design. RAA provides an engineering approach to next-generation intelligent manufacturing. It connects local autonomy with recursive coordination to support intelligence, openness, and system-wide integration.

## References

[1] Haolin Fan, Xuan Liu, Jerry Ying Hsi Fuh, Wen Feng Lu, and Bingbing Li. Embodied intelligence in manufacturing: Leveraging large language models for autonomous industrial robotics. Journal of Intelligent Manufacturing, 36:1141–1157, 2025.

[2] John H. Holland. Studying complex adaptive systems. Journal of Systems Science and Complexity, 19:1–8, 2006.

[3] Industrial Digital Twin Association. Specification of the asset administration shell: Part 1, metamodel.

IDTA 01001-3-0, 2023. Version 3.0; Oficial specification.

[4] Simon A. Levin. Ecosystems and the biosphere as complex adaptive systems. Ecosystems, 1:431–436, 1998.

[5] Ministry of Industry and Information Technology of the People’s Republic of China. Transcript of the press conference on the 14th Five-Year Plan for Intelligent Manufacturing Development, 2021. In Chinese. December 28, 2021; Oficial ministry transcript.

[6] Behrouz Alizadeh Mousavi, Cathal Heavey, Chirine Millauer, Zhikang Tian, and Hans Ehm. Improvement of demand fulfillment in advanced planning system through decentralized decision support system. Journal of Industrial Information Integration, 35:100487, 2023.

[7] Qingwei Nie, Dunbing Tang, Changchun Liu, Liping Wang, and Jiaye Song. A multi-agent and cloud-edge orchestration framework of digital twin for distributed production control. Robotics and Computer-Integrated Manufacturing, 82:102543, 2023.

[8] OECD. The Digital Transformation of SMEs. OECD Studies on SMEs and Entrepreneurship. OECD Publishing, Paris, 2021.

[9] Mohsen Soori, Behrooz Arezoo, and Roza Dastres. Digital twin for smart manufacturing, a review. Sustainable Manufacturing and Service Economics, 2:100017, 2023.

[10] Baicun Wang, Jiyuan Zang, Xianming Qu, Jingchen Dong, and Yanhong Zhou. Research on new-generation intelligent manufacturing based on human-cyber-physical systems. Strategic Study of CAE, 20(4):29–34, 2018.

[11] Yuchen Xia, Manthan Shenoy, Nasser Jazdi, and Michael Weyrich. Towards autonomous system: Flexible modular production system enhanced with large language model agents. In 2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation (ETFA), pages 1–8, 2023.

[12] Jiyuan Zang, Baicun Wang, Liu Meng, and Yuan Zhou. Brief analysis on three basic paradigms of intelligent manufacturing. Strategic Study of CAE, 20(4):13–18, 2018.

[13] Ji Zhou. Accelerating the development of a manufacturing powerhouse with new-generation intelligent manufacturing as the main strategic direction. Keynote address, 2026 Conference on Intelligent Manufacturing System Solutions, 2026. In Chinese. June 11, 2026, Beijing; presentation reproduced online.

[14] Ji Zhou, Peigen Li, Yanhong Zhou, Baicun Wang, Jiyuan Zang, and Liu Meng. Toward new-generation intelligent manufacturing. Engineering, 4(1):11–20, 2018.

[15] Ji Zhou, Yanhong Zhou, Baicun Wang, and Jiyuan Zang. Human–cyber–physical systems (HCPSs) in the context of new-generation intelligent manufacturing. Engineering, 5(4):624–636, 2019.