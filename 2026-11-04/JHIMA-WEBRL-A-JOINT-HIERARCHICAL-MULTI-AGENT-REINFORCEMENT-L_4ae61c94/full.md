# JHIMA-WEBRL: A JOINT HIERARCHICAL MULTI-AGENT REINFORCEMENT LEARNING FRAMEWORK FOR LONG-HORIZON WEB TASKS

Shanka Subhra Mondal Jiayun Wang Center for Advanced AI, Accenture

## ABSTRACT

Multimodal web agents have shown promising capabilities for interacting with websites through graphical user interfaces (GUIs), yet realistic web tasks often require long-horizon, multi-step planning that remains challenging for monolithic policies. Such agents must jointly infer high-level plans, navigate through intermediate states, track progress over extended interaction histories, and execute visually grounded actions. Although prior work has explored modular multi-agent systems with a separate planner, the planner is typically not updated jointly with the actor. We investigate whether explicitly separating and jointly training planning and execution provides a stronger inductive bias for long-horizon web navigation. We introduce JHiMA-WebRL, a joint hierarchical multi-agent reinforcement learning framework for web agents that couples a task decomposer with an actor. To train the decomposer, we use a strong teacher model to construct structured subtask supervision from multimodal GUI trajectories, and use these annotations to fine-tune both the decomposer and the subtask-conditioned actor. We then extend Group Relative Policy Optimization (GRPO) to jointly optimize the decomposer and actor in the interactive WebArena environment through hierarchical rollout grouping and role-specific advantages. On WebArena-Lite-v2, JHiMA-WebRL-14B, consisting of a 7B actor and a 7B decomposer, achieves an average success rate of 37.2%, compared with 24.0% for a ScaleCUA-32B agent that we fine-tune with RL on the same training tasks and evaluate under the same protocol. Ablations show that jointly updating the decomposer during RL outperforms updating the actor alone. These results suggest that jointly learned role factorization with smaller models can be a competitive alternative to scaling monolithic agents for long-horizon web tasks.

![](images/2437a615882a735c6e692a2157fdb8dbe076c697664765496d7271b9f1b59f34.jpg)  
Figure 1: Overview of JHiMA-WebRL. Left: prior web agents use a single policy, or a separate planner that receives no RL updates while the actor is trained. Right: we generate subtask supervision from GUI trajectories, fine-tune the decomposer and actor, and jointly optimize both with hierarchical RL: K decompositions, M actor rollouts each, and role-specific advantages from terminal rewards.

## 1 INTRODUCTION

Multimodal web agents interact with websites by perceiving screenshots, interpreting page elements, and executing actions such as clicking, typing, scrolling, and navigating between webpages. Such agents could support workflows in online shopping, web-based administration, repository management, and information retrieval. Benchmarks such as WebArena (Zhou et al., 2024) and Visual-WebArena (Koh et al., 2024) evaluate this setting through tasks that require interactive execution across multiple steps. These tasks combine high-level planning with low-level visually grounded control: for example, to cancel an order, an agent must navigate intermediate pages, identify relevant interface elements, track progress over a long action history, and avoid repeated or irrelevant actions.

Many web agents use a monolithic policy that maps the task, observation, and action history directly to the next action, learning planning, state tracking, visual grounding, and execution within a single model. ScaleCUA (Liu et al., 2025b) and UI-TARS (Qin et al., 2025) demonstrate the value of large-scale GUI data for training such policies, while WebAgent-R1 (Wei et al., 2025) and WebRL (Qi et al., 2025) improve web interaction through online reinforcement learning (RL). Modular systems instead introduce a separate planner: Agent S uses a manager to generate subtasks from retrieved knowledge and experience for workers to execute (Agashe et al., 2025a), and Agent S2 adds proactive replanning as execution provides new observations (Agashe et al., 2025b). Even though an explicit planner module can support long-horizon behavior, the planner’s parameters are not updated from the outcomes of action execution. Recent studies (Erdogan et al., 2025; Wu et al., 2026; Kong et al., 2026; Lu et al., 2025) focus on training modular agents for long-horizon tasks, but either train the modules separately, apply supervised learning or RL to only one module while freezing the other, or alternate updates between modules. To our knowledge, prior work has not studied simultaneous RL training of a separately parameterized planner and actor from a shared rollout batch in web environments.

We therefore introduce JHiMA-WebRL, a joint hierarchical multi-agent reinforcement learning framework for long-horizon web interaction. Figure 1 presents an overview of the three stages in our framework: decomposer data generation, supervised fine-tuning, and joint hierarchical multiagent RL. First, a strong teacher model converts multimodal GUI trajectories into ordered subtasks, supplying planning annotations that existing GUI trajectory datasets typically lack. Second, these annotations support supervised fine-tuning of two Qwen2.5-VL-7B models: a decomposer that predicts subtasks from the instruction and starting screenshot, and an actor that predicts the next GUI action conditioned on those subtasks. Finally, we jointly train both modules using terminal task success rewards while interacting with web environments, so that the decomposer can adapt its plans to the actor’s capabilities while the actor adapts to executing them. We extend Group Relative Policy Optimization (GRPO) through the hierarchical rollout structure: the decomposer proposes K candidate decompositions, and the actor performs M complete environment rollouts for each decomposition. Actor-level advantages compare executions under the same plan, while decomposer-level advantages compare plans by the average terminal reward of their executions, and both policies are updated from this shared batch without dense process rewards.

We evaluate JHiMA-WebRL-14B, our instantiation with a 7B actor and a 7B decomposer, on WebArena-Lite-v2 (Liu et al., 2025b), a refined subset of WebArena tasks for purely visual agents. JHiMA-WebRL-14B achieves an average success rate of 37.2%, compared with 24.0% for a ScaleCUA-32B agent that we fine-tune with RL on the same training tasks and evaluate under the same protocol. Ablation studies further indicate that jointly updating the decomposer during RL is beneficial relative to updating the actor alone. These results suggest that joint training of role-factorized modules can be a competitive alternative to scaling monolithic agents.

## 2 RELATED WORK

WebArena, VisualWebArena, and WorkArena evaluate agents on web tasks requiring multi-step interaction, visual grounding, state tracking, and long-horizon planning (Zhou et al., 2024; Koh et al., 2024; Drouin et al., 2024). OSWorld, Windows Agent Arena, and MacArena study the broader setting of desktop and OS-level computer use (Xie et al., 2024; Bonatti et al., 2025; Muryn et al., 2026). Strong GUI agents such as UI-TARS and ScaleCUA train largely monolithic policies that map observations and histories directly to actions using large-scale GUI data (Qin et al., 2025; Liu et al., 2025b). These systems demonstrate the importance of data scale and GUI grounding, but require a single policy to learn planning, progress tracking, grounding, and execution jointly, whereas we factorize web interaction into a trainable decomposer and actor.

A broad line of work studies modular or multi-agent LLM systems, including AutoGen, CAMEL, ChatDev, MetaGPT, and MAP (Wu et al., 2023; Li et al., 2023; Qian et al., 2024; Hong et al., 2024; Webb et al., 2025). In computer use, Agent S combines manager–worker decomposition with retrieved experience, and Agent S2 updates plans as subtasks are completed (Agashe et al., 2025a;b). OpAgent coordinates a planner, grounder, reflector, and summarizer, with the reflector’s feedback guiding subsequent planning (Guo et al., 2026). These mechanisms concern the organiza tion and adaptation of inference-time behavior. Related methods such as ReAct, Reflexion, MiRA, SEAgent, and TreeCUA explore intermediate reasoning, subgoals, feedback, or structured exploration (Yao et al., 2023; Shinn et al., 2023; Wang et al., 2026; Sun et al., 2025; Jiang et al., 2026). JHiMA-WebRL focuses on learning the decomposition interface from annotated trajectories and jointly adapting it with the actor through task rewards.

A complementary line of work studies training across planner–executor interfaces. Plan-and-Act separately fine-tunes planner and executor models using synthetic plans and action trajectories, and adds dynamic replanning at inference time (Erdogan et al., 2025). Our approach also uses trajectoryderived planning supervision, then jointly adapts both modules through RL. Navigating WebAI pairs a supervised hierarchical planner with an RL-trained controller (Thil et al., 2024). Planner Matters! optimizes only the planner using trajectory rewards from a VLM judge, freezing the actor and memory manager (Wu et al., 2026). Its joint-training ablation shares parameters across roles, whereas our decomposer and actor are separate policies. In robotic manipulation, HiRoC aligns an executor with subgoals through supervised fine-tuning, then applies hierarchical GRPO to the executor with a frozen planner, combining task-level and subgoal-level advantages (Kong et al., 2026). SWIRL instead trains both modules through alternating RL phases (Lu et al., 2025). In contrast, JHiMA-WebRL computes actor advantages within plans and decomposer advantages across plans, and updates both policies from a shared rollout batch.

PPO and GRPO provide the basic optimization tools for recent LLM and agent RL, while multiagent RL methods such as MADDPG, COMA, QMIX, and MAPPO study credit assignment across interacting policies (Schulman et al., 2017; Shao et al., 2024; Lowe et al., 2017; Foerster et al., 2018; Rashid et al., 2018; Yu et al., 2022). For web and GUI agents, WebAgent-R1, WebRL, OpenWebRL, and OpAgent apply online multi-turn RL to browser-based tasks, optimizing a single agent policy from terminal or outcome-based rewards (Wei et al., 2025; Qi et al., 2025; Yang et al., 2026a; Guo et al., 2026), while WorkForceAgent-R1 applies rule-based RL to single-step web decisions (Zhuang et al., 2025). Other works improve credit assignment with process or progress signals, including Web-Shepherd, WebArbiter, TGPO, GraphGPO, and MiRA (Chae et al., 2025; Zhang et al., 2026; Chen et al., 2025; Cheng et al., 2026; Wang et al., 2026). We instead extend GRPO’s rollout grouping and advantage computation to two jointly trained roles, using only terminal rewards and no dense process rewards.

## 3 METHOD

We propose JHiMA-WebRL, a joint hierarchical multi-agent reinforcement learning framework for long-horizon web navigation. Our approach factorizes interaction into two specialized but jointly optimized modules: a task decomposer and an actor. The decomposer maps the task instruction and initial screenshot into an ordered sequence of executable subtasks. The actor then conditions on the task, current observation, interaction history, and generated subtasks to produce visually grounded actions in the web interface. Below, we first describe the problem formulation, followed by the three stages of our framework: data generation for training the decomposer, supervised fine-tuning of both the decomposer and actor, and a joint hierarchical multi-agent extension of GRPO.

## 3.1 PROBLEM FORMULATION

We consider an interactive web environment modeled as a partially observed sequential decision problem. Each task is specified by a natural-language instruction x. At time step t, the agent receives a screenshot $o _ { t }$ of the web interface and produces an action $a _ { t } \in A ,$ such as clicking a page element, typing, scrolling, navigating between webpages, or terminating with an answer. The environment then transitions to a new state and returns the next observation $o _ { t + 1 }$ . A trajectory is denoted by $\tau = ( o _ { 0 } , a _ { 0 } , o _ { 1 } , a _ { 1 } , \dots , o _ { T } )$ , where $T$ is the trajectory length. Each trajectory receives a terminal task reward $R ( \tau ) \in \{ 0 , 1 \}$ , where $R ( \tau ) = 1$ if the task is successfully completed and $R ( \tau ) = 0$ otherwise. Success is determined through environment-specific rule-based verification or an LLM judge.

Our modular policy consists of a decomposer policy $\pi _ { D } \equiv \pi _ { \theta _ { D } }$ and an actor policy $\pi _ { A } \equiv \pi _ { \theta _ { A } }$ parameterized by separate models. Given a task instruction x and an initial observation $o _ { 0 }$ , the decomposer samples a decomposition

$$
d = ( g _ { 1 } , g _ { 2 } , \ldots , g _ { L } ) \sim \pi _ { D } ( \cdot \mid x , o _ { 0 } ) ,\tag{1}
$$

where each $g _ { \ell }$ is a natural-language subtask and L is the number of generated subtasks. The actor then generates web-interface actions according to

$$
a _ { t } \sim \pi _ { A } ( \cdot \mid x , o _ { t } , h _ { t } , d ) ,\tag{2}
$$

where $h _ { t } = ( a _ { 0 } , \ldots , a _ { t - 1 } )$ denotes the action history.

## 3.2 DECOMPOSER DATA GENERATION

Existing GUI trajectory datasets contain task instructions, screenshots, and action traces, but typi cally do not provide explicit subtask annotations. To train a task decomposer, we construct a new decomposition dataset from trajectory annotations in ScaleCUA (Liu et al., 2025b). ScaleCUA provides large-scale multimodal GUI data across platforms, including grounding annotations and human or agent trajectories.

For each trajectory, we provide a strong teacher model, GPT-5.2-Codex (OpenAI, 2025), with the original task instruction x and starting screenshot $o _ { 0 }$ , together with the full sequence of user or agent actions $( a _ { 0 } , \ldots , a _ { T - 1 } )$ and corresponding screenshots $\big ( o _ { 0 } , \ldots , o _ { T } \big )$ . A subgoal-generation guideline prompt, included in Appendix A, guides the annotation process.

The teacher produces an ordered list of executable subtasks $d = ( g _ { 1 } , \dotsc , g _ { L } )$ and optional descriptions of unnecessary or redundant actions in the trajectory. This process yields approximately 76K decomposer training examples; dataset statistics are provided in Table 3, and qualitative examples are shown in Appendix C. The resulting automated pipeline converts low-level multimodal GUI demonstrations into high-level planning supervision, enabling us to train an open-source decomposer rather than relying on a proprietary teacher at inference time.

## 3.3 SUPERVISED FINE-TUNING

We first train both modules with supervised fine-tuning before reinforcement learning.

Decomposer SFT. The decomposer is initialized from the Qwen2.5-VL-7B model and trained on the generated subtask dataset. Given the task instruction and initial screenshot, the model is trained to predict the teacher-generated decomposition:

$$
\mathcal { L } _ { D } ^ { \mathrm { S F T } } = - \sum _ { n = 1 } ^ { N _ { d } } \log \pi _ { \theta _ { D } } \left( d ^ { ( n ) } \mid x ^ { ( n ) } , o _ { 0 } ^ { ( n ) } \right) ,\tag{3}
$$

where $N _ { d }$ is the number of decomposer examples.

Actor SFT. The actor is also initialized from Qwen2.5-VL-7B and fine-tuned using supervised learning on released ScaleCUA trajectories augmented with the teacher-generated subtasks described in Section 3.2. For each trajectory, the actor receives the generated subtask list together with the task, current screenshot, and previous actions. It is then trained to predict the demonstrated next action:

$$
\mathcal { L } _ { A } ^ { \mathrm { S F T } } = - \sum _ { n = 1 } ^ { N _ { A } } \sum _ { t = 0 } ^ { T _ { n } - 1 } \log \pi _ { \theta _ { A } } \left( \boldsymbol { a } _ { t } ^ { ( n ) } \mid \boldsymbol { x } ^ { ( n ) } , \boldsymbol { o } _ { t } ^ { ( n ) } , \boldsymbol { h } _ { t } ^ { ( n ) } , \boldsymbol { d } ^ { ( n ) } \right) ,\tag{4}
$$

where $N _ { A }$ is the number of actor trajectories. This trains the actor to use decompositions as an explicit planning context while learning GUI grounding and action generation from the released ScaleCUA trajectories.

## 3.4 JOINT HIERARCHICAL MULTI-AGENT GRPO

After supervised fine-tuning, the joint RL stage of JHiMA-WebRL optimizes the decomposer and actor using GRPO (Shao et al., 2024), extended to their hierarchical interaction. We retain the GRPO clipped policy objectives and compute role-specific advantages from hierarchical rollout groups to assign credit to planning and execution. Standard web-agent RL methods often optimize a single policy from terminal trajectory rewards (Wei et al., 2025; Qi et al., 2025; Yang et al., 2026a). In contrast, our setting contains two interacting policies with different roles and action spaces: the decomposer generates a high-level plan, and the actor executes low-level web-interface actions conditioned on that plan.

For a task x with initial observation $o _ { 0 }$ , the decomposer first samples K candidate decompositions $d _ { i } \sim \pi _ { \theta _ { D } } ( \cdot \mid x , o _ { 0 } ) , i = 1 , \ldots , K$ . For each $d _ { i }$ , the actor then generates M independent multiturn trajectories $\tau _ { i , j } \sim \pi _ { \theta _ { A } } ( \cdot \mid x , o _ { 0 } , d _ { i } ) , j = 1 , \ldots , M$ , by interacting with the environment. We denote the actions of $\tau _ { i , j } \ b \ y \ \left( a _ { i , j , 0 } , \ldots , a _ { i , j , T _ { i , j } - 1 } \right)$ , where $T _ { i , j }$ is the number of steps and $a _ { i , j , k }$ is the action generated at environment interaction step k. Because each action is produced autoregressively by the VLM, we further write $a _ { i , j , k } = \left( a _ { i , j , k , 1 } , \dotsc , a _ { i , j , k , | a _ { i , j , k } | } \right)$ , where u indexes the generated tokens $\boldsymbol { a } _ { i , j , k , u }$ within the action. Each actor trajectory receives a terminal binary reward $R _ { i , j } = R ( \tau _ { i , j } ) \in \{ 0 , 1 \}$

Actor-level advantage. For a fixed decomposition $d _ { i } ,$ , we compute the group-relative actor advantage by comparing the M downstream trajectories generated under the same decomposition:

$$
A _ { i , j } ^ { A } = \frac { R _ { i , j } - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } R _ { i , m } } { \mathrm { s t d } _ { m } \left[ R _ { i , m } \right] + \delta } ,\tag{5}
$$

where $\delta > 0$ is a small constant for numerical stability. Thus, all action tokens in trajectory $\tau _ { i , j }$ share the same trajectory-level advantage $A _ { i , j } ^ { A }$

Following the M-GRPO objective (Wei et al., 2025), which applies the loss per token, we define the actor importance-sampling ratio for token u of action $a _ { i , j , k }$ as

$$
r _ { i , j , k , u } ^ { A } ( \theta _ { A } ) = \frac { \pi _ { \theta _ { A } } \left( a _ { i , j , k , u } \mid c _ { i , j , k } ^ { A } , a _ { i , j , k , < u } \right) } { \pi _ { \theta _ { A } ^ { \mathrm { o l d } } } \left( a _ { i , j , k , u } \mid c _ { i , j , k } ^ { A } , a _ { i , j , k , < u } \right) } ,\tag{6}
$$

where $c _ { i , j , k } ^ { A } = ( x , o _ { i , j , k } , h _ { i , j , k } , d _ { i } )$ contains the task instruction, current observation, action history available at environment step $k ,$ and decomposition $d _ { i }$

The clipped token-level actor objective is

$$
\begin{array}{c} \mathcal { L } _ { A } = - \frac { 1 } { K M } \sum _ { i = 1 } ^ { K } \sum _ { j = 1 } ^ { M } \frac { 1 } { T _ { i , j } } \sum _ { k = 0 } ^ { T _ { i , j } - 1 } \frac { 1 } { | a _ { i , j , k } | } \sum _ { u = 1 } ^ { | a _ { i , j , k } | } \bigg [ \operatorname* { m i n } \big ( r _ { i , j , k , u } ^ { A } ( \theta _ { A } ) A _ { i , j } ^ { A } ,  \\ & { \mathrm { c l i p } ( r _ { i , j , k , u } ^ { A } ( \theta _ { A } ) , 1 - \epsilon _ { A } , 1 + \epsilon _ { A } ) A _ { i , j } ^ { A } \big ) } \\ & { - \beta _ { A } D _ { \mathrm { K L } } ^ { A } ( i , j , k , u ) \bigg ] , } \end{array}\tag{7}
$$

where $\epsilon _ { A }$ is the clipping parameter and $D _ { \mathrm { K L } } ^ { A } ( i , j , k , u )$ is the per-token KL divergence between $\pi _ { \theta _ { \angle } }$ and the reference (SFT) actor, weighted $\boldsymbol { \mathsf { b y } } \beta _ { A }$

Decomposer-level advantage. To evaluate decomposition quality, we first aggregate the down stream rewards of the M actor trajectories conditioned on $d _ { i } \mathbf { : }$

$$
S _ { i } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } R _ { i , j } .\tag{8}
$$

The decomposer advantage is then computed relative to the K candidate decompositions sampled for the same task:

$$
A _ { i } ^ { D } = { \frac { S _ { i } - { \frac { 1 } { K } } \sum _ { i ^ { \prime } = 1 } ^ { K } S _ { i ^ { \prime } } } { \mathrm { s t d } _ { i ^ { \prime } } [ S _ { i ^ { \prime } } ] + \delta } } .\tag{9}
$$

Because $d _ { i }$ is itself an autoregressively generated sequence, we define the token-level ratio $r _ { i , u } ^ { D } ( \theta _ { D } ) ~ = ~ \pi _ { \theta _ { D } } ( d _ { i , u } ~ \vert ~ x , o _ { 0 } , \bar { d _ { i , < u } } ) / \pi _ { \theta _ { D } ^ { \mathrm { o l d } } } ^ { \mathrm { ~ } } \bar { ( } d _ { i , u } ~ \vert ~ x , o _ { 0 } , \bar { d } _ { i , < u } )$ for its u-th token, analogously to Eq. equation 6. The token-level decomposer objective is

$$
\begin{array} { c } { { \displaystyle { \mathcal { L } _ { D } = - \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { 1 } { | d _ { i } | } \sum _ { u = 1 } ^ { | d _ { i } | } \Big [ \operatorname* { m i n } \big ( r _ { i , u } ^ { D } ( \theta _ { D } ) A _ { i } ^ { D } , } } } \\ { { \mathrm { c l i p } ( r _ { i , u } ^ { D } ( \theta _ { D } ) , 1 - \epsilon _ { D } , 1 + \epsilon _ { D } ) A _ { i } ^ { D } \big ) } } \\ { { - \beta _ { D } D _ { \mathrm { K L } } ^ { D } ( i , u ) \Big ] , } } \end{array}\tag{10}
$$

with $\epsilon _ { D } , \beta _ { D }$ , and $D _ { \mathrm { K I } } ^ { D }$ defined analogously to those for the actor.

Thus, the downstream execution quality of a decomposition determines a single group-relative advantage $A _ { i } ^ { D }$ , which is propagated to all generated tokens of that decomposition. This enables the decomposer to learn which planning outputs are most useful to the actor while retaining token-level policy optimization.

Finally, the two modules are optimized jointly according to

$$
\mathcal { L } _ { \mathrm { J H i M A } } = \mathcal { L } _ { A } + \mathcal { L } _ { D } .\tag{11}
$$

Since $\theta _ { A }$ and $\theta _ { D }$ are disjoint, this sum amounts to updating each module with its own objective, using the same rollout batch. Equations equation 5–equation 10 give the standard GRPO form; in our experiments we remove the standard-deviation and response-length normalization following Liu et al. (2025c) (Section 4.3).

Appendix B shows that, without standard-deviation normalization, $A _ { i } ^ { D }$ yields an unbiased (up to a positive scale) baselined policy-gradient estimate for the decomposer under the shared task reward.

## 4 EXPERIMENTS

## 4.1 DATASETS

The task decomposer was trained with supervised fine-tuning using approximately 76K taskdecomposition examples, which were generated from ScaleCUA trajectories using GPT-5.2-Codex as described in Section 3.2. The actor was fine-tuned on all ScaleCUA trajectories augmented with the teacher-annotated subtasks, along with grounding data spanning web, mobile, and desktop platforms, resulting in 2,535,552 training examples. Unlike standard ScaleCUA SFT, the actor thus receives a task decomposition in its context as a planning scaffold.

Both modules were then jointly trained through online reinforcement learning using the training tasks associated with WebArena-Lite, following the setup of WebAgent-R1 (Wei et al., 2025). WebArena contains 812 interactive web tasks spanning five domains—Reddit, GitLab, e-commerce administration, maps, and online shopping. WebArena-Lite (Liu et al., 2025a) selects a cleaned subset of 165 tasks for evaluation, leaving 647 tasks from the original WebArena benchmark for reinforcement-learning interaction. No WebArena-Lite evaluation task (and hence no WebArena-Lite-v2 task) is used during SFT or RL. We also employed the flexible evaluation protocol of Liu et al. (2025b), including LLM-based fuzzy matching using GPT-5.2-Codex, to compute the binary task outcome reward at the end of web interaction.

For evaluation, we used WebArena-Lite-v2, introduced by ScaleCUA (Liu et al., 2025b). WebArena-Lite-v2 is a refinement of the evaluation subset of WebArena-Lite designed specifically for evaluating purely visual GUI agents; it removes 11 tasks that require multi-tab interaction and revises task instructions and evaluators. It contains 154 tasks across five website environments: Shopping, ShoppingAdmin, Reddit, GitLab, and OpenStreetMap. The benchmark includes heterogeneous task objectives such as question answering, navigation, page-content matching, and statechanging operations, with success determined by the flexible evaluation criteria of Liu et al. (2025b).

## 4.2 BASELINES

We compared against native agents at several model scales ranging from 7B to 72B, allowing comparisons between our 14B modular system and both smaller and larger monolithic systems. We also compared against multi-agent systems consisting of a proprietary planner (GPT-4o) with a grounding module. These provide comparisons with systems that separate planning from visual grounding, but do not jointly train a learned decomposer and actor as in JHiMA-WebRL-14B. The UI-TARS and GPT-4o planner–grounder results are taken from ScaleCUA (Liu et al., 2025b) (15-step budget). The ScaleCUA-7B/32B SFT and GUI-Libra-8B results are taken from the re-evaluation of Yang et al. (2026b), which uses a 15-step budget, averages four runs, and replaces the GPT-4o fuzzy-match judge with GPT-5. Under its original evaluation setup, ScaleCUA reports higher scores for its own models (37.7% for ScaleCUA-7B and 44.2% for ScaleCUA-32B at 15 steps). We also trained a ScaleCUA-32B RL baseline, specifically fine-tuning the SFT model on the WebArena training set, using standard multi-turn GRPO with 7 rollouts and a maximum of 15 interaction steps. For a fair comparison with our method, we only included baselines that do not use HTML or DOM information.

## 4.3 EXPERIMENTAL DETAILS

We trained the 7B decomposer and 7B actor of JHiMA-WebRL-14B in two stages: SFT, followed by joint hierarchical RL. The decomposer was initialized from Qwen2.5-VL-7B and fine-tuned on the approximately 76K decomposition examples generated from ScaleCUA multimodal trajectories, on 8 NVIDIA H100 GPUs with 80 GB of memory each via full-parameter fine-tuning. We trained for 1 epoch with a learning rate of 1e-5 and a cosine learning-rate schedule with a warmup ratio of 0.03. The per-device batch size was 2, with gradient accumulation over 4 steps, resulting in an effective global batch size of 64. We used DeepSpeed (Rajbhandari et al., 2020) for distributed training with ZeRO-3, along with gradient checkpointing, bf16 mixed-precision training, gradient clipping to a maximum norm of 1, and a maximum token length of 30720. The actor was also initialized from Qwen2.5-VL-7B and fine-tuned with the same hyperparameters as above, using released ScaleCUA trajectories augmented with generated subtasks.

After SFT, both the actor and decomposer were jointly optimized through LoRA fine-tuning using GRPO, with standard-deviation and response-length normalization removed to reduce optimization bias, following Liu et al. (2025c). We used LoRA rank $r = 3 2$ and scaling factor $\alpha = 6 4$ for both modules, with a separate adapter for each. For each task, the decomposer sampled $K = 3$ candidate decompositions. For every decomposition, the actor generated M = 7 complete trajectories. For both, we used 0.8 as the sampling temperature, and we allowed a maximum of 15 interaction steps per trajectory. We optimized the LoRA parameters with a learning rate of 1e-5. The GRPO updates used clipping parameters $\epsilon _ { A } = \epsilon _ { D } = 0 . 2$ , KL coefficients $\beta _ { A } = \beta _ { D } = 0$ (i.e., no KL penalty), and 2 optimization steps per rollout batch. We used 32 gradient accumulation steps and trained for 1 epoch over the 647 training tasks. We used the efficient parallel trajectory rollout generation of M-GRPO (Wei et al., 2025) for each task decomposition, with training taking approximately 720 GPU-hours on a cluster of 8 H100 GPUs with 80 GB of memory each.

Evaluation. We report the task success rate on all 154 WebArena-Lite-v2 tasks, averaged over 4 independent evaluation runs. We sampled one decomposition at test time using a temperature of 0.0 and allowed a maximum of 15 interaction steps, consistent with the baselines.

## 4.4 RESULTS

As shown in Table 1, JHiMA-WebRL-14B achieved the highest average success rate among the listed methods, 37.2% over four runs, and the highest scores on GitLab (47.5%), Reddit (60.5%), Shopping (29.0%), and ShoppingAdmin (44.3%). The most controlled comparison is with ScaleCUA-32B RL, which we trained on the same RL tasks with the same step budget and evaluated under the same protocol: JHiMA-WebRL-14B outperformed it by 13.2 percentage points with fewer total parameters. Among results from prior work, JHiMA-WebRL-14B exceeded the GPT-4o + ScaleCUA-7B planner–grounder system (28.6%), GUI-Libra-8B (26.6%), and UI-TARS-72B-DPO (23.4%); however, these numbers come from evaluation setups that differ from ours, and ScaleCUA reports 44.2% for ScaleCUA-32B under its original setup (Section 4.2). The Map domain is the exception: JHiMA-WebRL-14B (12.5%) trails GUI-Libra-8B (17.3%) and ScaleCUA-7B SFT (15.4%). Figure 6 shows an example Map failure in which the agent repeatedly toggles the zoom controls until the 15-step budget is exhausted. Within our controlled comparisons, these results suggest that learned role factorization with smaller, jointly trained models is a competitive alternative to scaling a monolithic agent.

Table 1: Success rates (%) on WebArena-Lite-v2. Our results are averaged over 4 runs. <sup>∗</sup>Reported by ScaleCUA (Liu et al., 2025b); <sup>†</sup>evaluated by GUI-Libra (Yang et al., 2026b); <sup>‡</sup>trained and evaluated by us. All rows use a 15-step budget, but the fuzzy-match judge differs across sources (Section 4.2). Average is computed over all 154 tasks (task-weighted), not as the mean of the domain scores. Dashes indicate domain scores not reported.
<table><tr><td>Method</td><td>GitLab</td><td>Map</td><td>Reddit</td><td>Shopping</td><td>ShoppingAdmin</td><td>Average</td></tr><tr><td>UI-TARS-1.5-7B*</td><td>一</td><td>一</td><td>一</td><td></td><td></td><td>20.8</td></tr><tr><td>UI-TARS-72B-DPO*</td><td></td><td></td><td></td><td></td><td></td><td>23.4</td></tr><tr><td>ScaleCUA-7B SFT†</td><td>28.3</td><td>15.4</td><td>27.6</td><td>18.8</td><td>30.7</td><td>23.9</td></tr><tr><td>ScaleCUA-32B SFT†</td><td>34.2</td><td>10.6</td><td>26.3</td><td>16.5</td><td>33.6</td><td>24.0</td></tr><tr><td>ScaleCUA-32B RL‡</td><td>35.0</td><td>10.6</td><td>21.0</td><td>24.4</td><td>25.7</td><td>24.0</td></tr><tr><td>GUI-Libra-8B†</td><td>31.7</td><td>17.3</td><td>35.5</td><td>26.1</td><td>25.0</td><td>26.6</td></tr><tr><td>GPT-4o Planner + UI-TARS-1.5-7B Grounder*</td><td></td><td></td><td></td><td></td><td></td><td>22.6</td></tr><tr><td>GPT-4o Planner + ScaleCUA-7B Grounder*</td><td></td><td></td><td></td><td></td><td></td><td>28.6</td></tr><tr><td>JHiMA-WebRL-14B (Ours)</td><td>47.5</td><td>12.5</td><td>60.5</td><td>29.0</td><td>44.3</td><td>37.2</td></tr></table>

Table 2: Ablation success rates (%) on WebArena-Lite-v2, averaged over 4 runs (task-weighted average). “Rollouts” denotes the number of complete trajectories per training task.
<table><tr><td>Method</td><td>GitLab</td><td>Map</td><td>Reddit</td><td>Shopping</td><td>ShoppingAdmin</td><td>Average</td></tr><tr><td>JHiMA-WebRL-14B (Ours)</td><td>47.5</td><td>12.5</td><td>60.5</td><td>29.0</td><td>44.3</td><td>37.2</td></tr><tr><td>w/o decomposer RL</td><td>37.5</td><td>10.6</td><td>54.0</td><td>25.6</td><td>45.7</td><td>33.4</td></tr><tr><td>w/o decomposer (actor-only RL, 21 rollouts)</td><td>44.2</td><td>14.4</td><td>51.3</td><td>26.7</td><td>43.6</td><td>34.9</td></tr><tr><td>w/o decomposer (actor-only RL, 7 rollouts)</td><td>39.2</td><td>15.4</td><td>48.7</td><td>30.1</td><td>45.7</td><td>35.2</td></tr><tr><td>w/o RL (SFT decomposer + SFT actor)</td><td>43.3</td><td>8.6</td><td>59.2</td><td>25.6</td><td>46.4</td><td>35.1</td></tr></table>

## 4.5 ABLATION STUDY

We analyzed the contribution of the components of JHiMA-WebRL-14B through ablations (Table 2). First, we trained only the actor with RL while keeping the SFT decomposer frozen (‘w/o decomposer RL’), using the same number of rollouts per task (K=3, M=7) as JHiMA-WebRL-14B. This reduced the average success rate to 33.4%, a drop of 3.8 percentage points, indicating that updating the decomposer jointly with the actor matters. Second, to assess the role of the decomposer, we trained the actor alone using standard multi-turn GRPO without subtask conditioning. The actor was initialized from Qwen2.5-VL-7B and fine-tuned using supervised learning on released Scale-CUA trajectories and grounding data, without any subtasks. We trained two variants: one with the same total number of rollouts per task as JHiMA-WebRL-14B (21), and one with 7 rollouts, matching the per-decomposition group size of our actor. The two variants reached 34.9% and 35.2%, i.e., 2.3 and 2.0 percentage points below JHiMA-WebRL-14B. Finally, we removed RL entirely and evaluated the SFT decomposer and SFT actor together (‘w/o RL’), which reduced the average success rate by 2.1 percentage points. Notably, actor-only RL with a frozen SFT decomposer (33.4%) scored below this SFT-only system (35.1%), whereas joint RL improved on it; the SFT-only modular system also performed on par with the decomposer-free actors trained with RL (34.9% and 35.2%).

## 4.6 SUBGOAL QUALITY ANALYSIS

We also analyzed whether the generated decompositions provide meaningful intermediate structure. Following the subgoal-quality analysis of Wang et al. (2026), we measured whether partial subgoal completion was predictive of eventual task success and whether task success increased monotonically as more subgoals were completed.

For each WebArena-Lite-v2 trajectory from the four evaluation runs, we used GPT-5.2-Codex to judge whether each subgoal generated by the decomposer was completed, based on the sequence of predicted actions and resulting screenshots, and computed the fraction of generated subgoals that were completed. We first evaluated whether the fraction of completed subgoals discriminated successful trajectories from failed ones. We treated the fraction as a continuous prediction score for the binary task outcome and computed the area under the receiver operating characteristic curve (AUROC). As shown in Figure 2a, the subgoals generated by JHiMA-WebRL-14B achieved an AU-ROC of 0.838, well above chance (0.5) and comparable to the SFT-only system (‘w/o RL’, 0.836). Progress through the generated decompositions is therefore predictive of downstream task success, both before and after joint RL.

![](images/76bdcf427ee0e4646de40f59e987fab080079b21a53ca1504d00cfb6f2964f2a.jpg)  
(a) ROC curve using the fraction of completed subgoals as a score for predicting final task success.

![](images/3d630504f3339a819bc025d8a257d0e51b5ce91b45cc6c8b5ee0444e7c42502c.jpg)  
(b) Empirical task success rate as a function of subgoal completion.  
Figure 2: Subgoal quality analysis. (a) Discriminative power of subgoal completion for predicting terminal task success. (b) Empirical task success versus binned subgoal-completion fraction. Both panels compare JHiMA-WebRL-14B with the SFT-only system (‘w/o RL’).

A useful decomposition should also provide a monotone notion of progress, so we grouped trajectories by subgoal completion fraction and computed the empirical success rate per group. For JHiMA-WebRL-14B, empirical task success increases monotonically with subgoal completion fraction (Figure 2b), with Kendall’s rank correlation coefficient τ = 0.516, compared with τ = 0.503 without RL. Together with the AUROC analysis, these results suggest that the generated subgoals act as intermediate milestones associated with downstream task completion.

## 5 LIMITATIONS

Our study has several limitations. (i) Train–test overlap: RL is performed on the remaining WebArena tasks, which are disjoint from the evaluation tasks but can share intent templates and websites with them; results thus reflect in-environment generalization, not transfer to unseen websites. (ii) Supervision and cost: decomposition supervision is generated by a proprietary teacher (GPT-5.2- Codex) without human verification of annotation quality, and hierarchical rollouts require K×M complete episodes per task, which is costly in live environments. (iii) Open-loop planning: the decomposer plans once from the initial observation and cannot revise the plan during execution. (iv) Coarse credit: terminal rewards assign the same advantage to every token of a decomposition or trajectory, without identifying which subtask or action caused success or failure. (v) Evaluation scope: we evaluate on a single benchmark of 154 tasks, and the ablation differences (2.0–3.8 percentage points) are of a size where run-to-run variance matters.

## 6 CONCLUSION AND FUTURE DIRECTIONS

We introduced JHiMA-WebRL, a framework for jointly training a task decomposer and a visually grounded actor. The framework generates subtask supervision from multimodal GUI trajectories to train the decomposer and extends GRPO with hierarchical rollout grouping and role-specific advantages. JHiMA-WebRL-14B achieved an average success rate of 37.2% on WebArena-Litev2, 13.2 percentage points above a ScaleCUA-32B RL baseline trained and evaluated under the same protocol. In ablations, removing decomposer updates during RL, removing the decomposer together with subtask conditioning, or removing RL each lowered the average success rate, by 2.0– 3.8 percentage points.

Our work suggests several future directions. First, the decomposer could be made adaptive, revising subtasks as execution deviates from the plan. Second, joint optimization can be combined with intermediate progress signals such as milestone rewards (Wang et al., 2026) or graph-based credit assignment (Cheng et al., 2026). Finally, jointly trained decomposer–actor systems could be extended to cross-domain and live-web settings, where online RL has recently been shown to be feasible (Yang et al., 2026a). This would test whether modular role factorization transfers better than monolithic policies.

## AI USE STATEMENT

Generative AI is part of our method: GPT-5.2-Codex is used as a teacher model to generate subtask annotations from ScaleCUA trajectories (Section 3.2) and as a judge of subgoal completion in the analysis of Section 4.6; the evaluation protocol of Liu et al. (2025b) also uses an LLM for fuzzy answer matching. We used Claude Opus 4.8 for coding and GPT-5.6 and GPT-6 to assist with drafting the paper and reviewing the literature. We have reviewed all AI-assisted work and take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Section 3 specifies the full training objective, including the hierarchical advantages (Eqs. 5–9) and the clipped objectives for both modules. Section 4.1 describes the training and evaluation data, and Section 4.3 lists all SFT and RL hyperparameters. The exact prompt used to generate decomposition supervision is given in Appendix A. We will also release the code, the decomposition dataset, and model checkpoints.

## REFERENCES

Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent S: An open agentic framework that uses computers like a human. In International Conference on Learning Representations, 2025a. URL https://arxiv.org/abs/2410.08164.

Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents, 2025b. URL https: //arxiv.org/abs/2504.00906.

Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, Lawrence Keunho Jang, and Zack Hui. Windows agent arena: Evaluating multi-modal os agents at scale. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 4874–4910, 2025. URL https://arxiv.org/abs/2409.08264.

Hyungjoo Chae, Sunghwan Kim, Junhee Cho, Seungone Kim, Seungjun Moon, Gyeom Hwangbo, Dongha Lim, Minjin Kim, Yeonjun Hwang, Minju Gwak, Dongwook Choi, Minseok Kang, Gwanhoon Im, ByeongUng Cho, Hyojun Kim, Jun Hee Han, Taeyoon Kwon, Minju Kim, Beongwoo Kwak, Dongjin Kang, and Jinyoung Yeo. Web-shepherd: Advancing prms for reinforcing web agents, 2025. URL https://arxiv.org/abs/2505.15277. Advances in Neural Information Processing Systems (NeurIPS) 2025.

Ziyuan Chen, Zhenghui Zhao, Zhangye Han, Miancan Liu, Xianhang Ye, Yiqing Li, Hongbo Min, Jinkui Ren, Xiantao Zhang, and Guitao Cao. TGPO: Tree-guided preference optimization for robust web agent reinforcement learning, 2025. URL https://arxiv.org/abs/2509. 14172.

Xin Cheng, Shuo He, Lang Feng, HaiYang Xu, Ming Yan, Lei Feng, and Bo An. Beyond trajectorylevel attribution: Graph-based credit assignment for agentic reinforcement learning, 2026. URL

https://arxiv.org/abs/2605.26684. International Conference on Machine Learning (ICML) 2026.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks? In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 11642–11662. PMLR, 2024. URL https://proceedings.mlr.press/ v235/drouin24a.html.

Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-Act: Improving planning of agents for long-horizon tasks, 2025. URL https://arxiv.org/abs/2503.09572. International Conference on Machine Learning (ICML) 2025.

Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual multi-agent policy gradients. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. URL https://arxiv.org/abs/1705.08926.

Yuyu Guo, Wenjie Yang, Siyuan Yang, Ziyang Liu, Cheng Chen, Yuan Wei, Yun Hu, Yang Huang, Guoliang Hao, Dongsheng Yuan, Jianming Wang, Xin Chen, Hang Yu, Lei Lei, and Peng Di. Opagent: Operator agent for web navigation, 2026. URL https://arxiv.org/abs/2602. 13559.

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for a multiagent collaborative framework. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2308.00352.

Deyang Jiang, Jing Huang, Xuanle Zhao, Lei Chen, Liming Zheng, Fanfan Liu, Haibo Qiu, Peng Shi, and Zhixiong Zeng. Treecua: Efficiently scaling gui automation with tree-structured verifiable evolution, 2026. URL https://arxiv.org/abs/2602.09662.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024. URL https://arxiv.org/abs/ 2401.13649.

He Kong, Zengjue Chen, Qi Wang, Qianli Xing, Runliang Niu, Peidong Liu, Jiawei Li, Shiqi Wang, and Yi Chang. Beyond flat policies: Hierarchical post-training for embodied agents in robotic manipulation, 2026. URL https://arxiv.org/abs/2608.05999.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for “mind” exploration of large language model society. arXiv preprint arXiv:2303.17760, 2023. URL https://arxiv.org/abs/2303.17760.

Xiao Liu, Tianjie Zhang, Yu Gu, Iat Long Iong, Yifan Xu, Xixuan Song, Shudan Zhang, Hanyu Lai, Xinyi Liu, Hanlin Zhao, et al. VisualAgentBench: Towards large multimodal models as visual foundation agents. In International Conference on Learning Representations, 2025a. URL https://arxiv.org/abs/2408.06327.

Zhaoyang Liu, Jingjing Xie, Zichen Ding, Zehao Li, Bowen Yang, Zhenyu Wu, Xuehui Wang, Qiushi Sun, Shi Liu, Weiyun Wang, Shenglong Ye, Qingyun Li, Xuan Dong, Yue Yu, Chenyu Lu, YunXiang Mo, Yao Yan, Zeyue Tian, Xiao Zhang, Yuan Huang, Yiqian Liu, Weijie Su, Gen Luo, Xiangyu Yue, Biqing Qi, Kai Chen, Bowen Zhou, Yu Qiao, Qifeng Chen, and Wenhai Wang. Scalecua: Scaling open-source computer use agents with cross-platform data. arXiv preprint arXiv:2509.15221, 2025b. URL https://arxiv.org/abs/2509.15221.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025c. URL https: //arxiv.org/abs/2503.20783.

Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actorcritic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems, 2017. URL https://arxiv.org/abs/1706.02275.

Quanfeng Lu, Zhantao Ma, Shuai Zhong, Jin Wang, Dahai Yu, Michael K. Ng, and Ping Luo. SWIRL: A staged workflow for interleaved reinforcement learning in mobile gui control, 2025. URL https://arxiv.org/abs/2508.20018.

Victor Muryn, Maksym Shamrai, Sofiia Mazepa, and Yehor Khodysko. Macarena: Benchmarking computer use agents on an online macos environment, 2026. URL https://arxiv.org/ abs/2606.06560.

OpenAI. Introducing gpt-5.2-codex. https://openai.com/index/ introducing-gpt-5-2-codex/, December 2025. Accessed: 2026-09-25.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, Tianjie Zhang, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2411.02337.

Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Chatdev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024. URL https://arxiv.org/abs/2307.07924.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025. URL https://arxiv.org/abs/2501. 12326.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC), 2020. doi: 10.1109/ SC41405.2020.00024.

Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 4295–4304. PMLR, 2018. URL https://proceedings.mlr.press/v80/rashid18a.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxiv. org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2303.11366.

Zeyi Sun, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Tong Wu, Dahua Lin, and Jiaqi Wang. SEAgent: Self-evolving computer use agent with autonomous learning from experience, 2025. URL https://arxiv.org/abs/2508.04700.

Lucas-Andre¨ı Thil, Mirela Popa, and Gerasimos Spanakis. Navigating WebAI: Training agents to complete web tasks with large language models and reinforcement learning. In Proceedings ofthe 39th ACM/SIGAPP Symposium on Applied Computing, 2024. doi: 10.1145/3605098.3635903. URL https://arxiv.org/abs/2405.00516.

Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, and Edward Grefenstette. A subgoaldriven framework for improving long-horizon llm agents, 2026. URL https://arxiv.org/ abs/2603.19685.

Taylor Webb, Shanka Subhra Mondal, and Ida Momennejad. A brain-inspired agentic architecture to improve planning with LLMs. Nature Communications, 16:8633, 2025. doi: 10.1038/s41467-025-63804-5. URL https://www.nature.com/articles/ s41467-025-63804-5.

Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, and Lihong Li. WebAgent-R1: Training web agents via end-to-end multi-turn reinforcement learning. In The 2025 Conference on Empirical Methods in Natural Language Processing, 2025. URL https://arxiv.org/abs/2505.16421.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023. URL https://arxiv.org/abs/2308.08155.

Wenyi Wu, Sibo Zhu, Kun Zhou, and Biwei Huang. Planner Matters! an efficient and unbalanced multi-agent collaboration framework for long-horizon planning, 2026. URL https://arxiv. org/abs/2605.02168.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. URL https://arxiv. org/abs/2404.07972.

Rui Yang, Qianhui Wu, Yuxi Chen, Hao Bai, Wenlin Yao, Hao Cheng, Baolin Peng, Huan Zhang, Tong Zhang, and Jianfeng Gao. Openwebrl: Demystifying online multi-turn reinforcement learning for visual web agents, 2026a. URL https://arxiv.org/abs/2606.02031.

Rui Yang, Qianhui Wu, Zhaoyang Wang, Hanyang Chen, Ke Yang, Hao Cheng, Huaxiu Yao, Baolin Peng, Huan Zhang, Jianfeng Gao, and Tong Zhang. Gui-libra: Training native gui agents to reason and act with action-aware supervision and partially verifiable rl, 2026b. URL https: //arxiv.org/abs/2602.22190.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629.

Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative multi-agent games. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/2103.01955.

Yao Zhang, Shijie Tang, Zeyu Li, Zhen Han, and Volker Tresp. Webarbiter: A principle-guided reasoning process reward model for web agents, 2026. URL https://arxiv.org/abs/ 2601.21872. International Conference on Learning Representations (ICLR) 2026.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024. URL https://webarena.dev.

Yuchen Zhuang, Di Jin, Jiaao Chen, Wenqi Shi, Hanrui Wang, and Chao Zhang. Workforceagentr1: Incentivizing reasoning capability in llm-based web agents via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.22942.

## APPENDIX

Table 3: Subtask-decomposition dataset statistics.
<table><tr><td>Platform</td><td># Examples</td><td>Avg. # Subtasks</td><td>Frac. w/ Unnecessary Actions</td></tr><tr><td>Windows</td><td>5,873</td><td>2.68</td><td>1.2%</td></tr><tr><td>macOS</td><td>3,845</td><td>2.43</td><td>1.1%</td></tr><tr><td>Linux</td><td>25,122</td><td>2.20</td><td>1.8%</td></tr><tr><td>Android</td><td>17,245</td><td>2.84</td><td>1.6%</td></tr><tr><td>Web</td><td>24,311</td><td>2.73</td><td>1.5%</td></tr><tr><td>Total</td><td>76,396</td><td>2.56</td><td>1.6%</td></tr></table>

## A SUBTASK GENERATION PROMPT

The following prompt is provided together with the task instruction and interleaved trajectory screenshots. Images are inserted as multimodal image content immediately after the corresponding textual markers.

You are an AI assistant tasked with generating high level subtasks that highlight key subgoals   
necessary to complete a task.   
Task: {TASK}   
Starting Image URL:   
{STARTING\_IMAGE}   
Below are the actions taken by the user at each step to perform the task   
Step 1: {STEP\_1}   
Image URL after step 1:   
{IMAGE\_AFTER\_STEP\_1}   
Step 2: {STEP\_2}   
Image URL after step 2:   
{IMAGE\_AFTER\_STEP\_2}   
...   
Step N: {STEP\_N}   
Image URL after step N:   
{IMAGE\_AFTER\_STEP\_N}   
Guidelines for Subtask Generation   
1. Identify Essential High-Level Subgoals:   
- A subgoal should represent a significant step involving user action that leads to   
noticeable page transitions or meaningful changes in system state (i.e Image).   
- Consolidate closely related user actions (such as applying multiple filters or selecting   
several options) into a single subgoal, rather than separate subtasks for each action.   
- Prioritize only the most critical actions necessary for meaningful progression,   
avoiding the inclusion of minor or unnecessary steps (e.g., scroll, hover).   
2. Provide a Concise Subgoal Analysis:   
- Before creating the subtasks, offer a brief paragraph summarizing the main subgoals,   
emphasizing significant transitions or page-level actions.   
3. Ensure Clear Goal:   
- If multiple related actions occur (e.g., setting filters 1, 2, and 3), combine them into   
one   
subgoal with clear criteria verifying all required conditions.   
- The subtasks should contain only essential steps, explicitly excluding unnecessary   
actions, and should not exceed five critical subtasks. It is not necessary to use all five   
subtasks if fewer steps adequately represent the essential subgoals.   
Finally, based on the generated subtasks and subgoal analysis, report if there are any   
unnecessary or redundant actions which aren’t required to perform the task.   
If all the actions are important say <NONE>, else say <YES> alongwith the unnecessary   
actions.   
### Output Format   
Before generating the subtasks, first produce a concise subgoal analysis in a single

paragraph summarizing the required actions. Then, based on this, generate the subtasks   
following the format below:   
[SUBGOAL ANALYSIS]   
[One-paragraph summary explaining the key subgoals and their logical   
sequence in task completion.]   
[SUBTASKS]   
SUBTASK X:   
[Short title of the action/goal]   
- Goal:   
[Brief description of the subgoal at this stage, emphasizing the   
purpose of the action.]   
[UNNECESSARY ACTIONS]   
[NONE or YES alongwith the unnecessary actions]

## B POLICY-GRADIENT INTERPRETATION OF THE HIERARCHICAL ADVANTAGES

Both modules share the single objective $J ( \theta _ { D } , \theta _ { A } ) \ = \ \mathbb { E } _ { d \sim \pi _ { \theta _ { D } } } \mathbb { E } _ { \tau \sim \pi _ { \theta _ { A } } ( \cdot | x , o _ { 0 } , d ) } [ R ( \tau ) ]$ , so the setting is fully cooperative with a shared reward. By the policy-gradient theorem, $\nabla _ { \theta _ { D } } J ~ =$ $\mathbb { E } _ { d } \big [ \nabla _ { \theta _ { D } } \log \pi _ { \theta _ { D } } ( d \ | \ x , o _ { 0 } ) \left( Q ( d ) - b \right) \big ]$ with $Q ( d ) = \mathbb { E } _ { \tau } [ R ( \tau ) \ | \ d ]$ and any baseline b that does not depend on d. The score $S _ { i }$ in Eq. equation 8 is an unbiased Monte Carlo estimate of $Q ( d _ { i } )$ under the current actor; the M inner rollouts reduce its variance relative to scoring a decomposition by a single rollout. Because the group mean S<sup>¯</sup> includes $S _ { i }$ , we have $\begin{array} { r } { S _ { i } - \bar { S } = \frac { K ^ { \smile } - 1 } { K } ( S _ { i } - \bar { \bar { S } } _ { - i } ) } \end{array}$ , where the leave-one-out mean $\bar { S } _ { - i }$ depends only on the other $K - 1$ independently sampled decompositions and their rollouts, and is therefore a valid baseline for $d _ { i }$ . Hence, without standard-deviation normalization, the decomposer advantage gives an unbiased policy-gradient estimate up to the positive factor $( K - 1 ) / K$ ; dividing by the group standard deviation, as in standard GRPO, rescales the estimate per task and no longer preserves this property exactly. Likewise, conditioning the actor baseline on $d _ { i }$ (Eq. equation 5) removes the reward variance attributable to plan quality from the actor’s advantage, so that actor updates compare executions of the same plan. Because both policies change simultaneously, each module’s learning problem is non-stationary; we rely on the PPO-style clipping and a small number of optimization steps per batch to keep per-iteration policy changes small.

## C QUALITATIVE EXAMPLES

Figures 3 and 4 show teacher-generated annotations for two ScaleCUA trajectories, one containing an unnecessary action and one in which all actions are necessary. Figure 5 shows a decomposition produced by our decomposer from the task instruction and starting screenshot of a WebArena ShoppingAdmin task. Figure 6 shows a failure case for a Map task.

![](images/ab3393b406d36d8a1266c8f11a0fba345a2fa65ac284d5853e452acb85bbbb9d.jpg)

Figure 3: Teacher annotation of a ScaleCUA mobile trajectory with an unnecessary action (Step 3, highlighted). The teacher outputs three subtasks and flags the redundant scroll action.  
![](images/f065af14f38b63ce41ef2be01c4a12217529b2e85e9c9d76391830bdab513f7d.jpg)  
Figure 4: Teacher annotation of a ScaleCUA mobile trajectory in which all actions are necessary; the teacher outputs three subtasks and [NONE] for unnecessary actions.

![](images/48b2cabb4bf2f3605d7a3f7f82ffcceae9a5163e9f49c07039147d7023f4cb15.jpg)  
Figure 5: Decomposition generated by the JHiMA-WebRL decomposer for the ShoppingAdmin evaluation task “Cancel order 302” from the task instruction and starting screenshot.

![](images/b7e4d40be6051bbffaf15efa16a647305e8acb05673bf256a6a637124d8e4ca1.jpg)  
Figure 6: Failure of JHiMA-WebRL on a Map task: repeated irrelevant actions exhaust the 15-step budget.