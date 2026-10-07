# DECOUPLED MULTI-AGENT ORCHESTRATION

Xinle Wu, Yao Lu National University of Singapore {wuxl,luyao}@comp.nus.edu.sg

## ABSTRACT

Learned orchestration can automatically construct effective language-model multi-agent systems, but existing approaches couple planning to fixed worker pools and train decomposition and collaboration from the same terminal outcome, limiting transfer and obscuring credit assignment. We introduce DEORCH, which separates worker-agnostic planning from concrete worker selection. Its two-stage planner first decomposes the task without worker information, then chooses collaboration operations using compact, worker-identity-free matchability feedback from the pool, enabling conditional credit assignment to decomposition and collaboration decisions. A lightweight matcher estimates worker suitability from behavior on a fixed probe set and adapts online with a contextual bandit, allowing new workers to be incorporated without retraining the planner or matcher. Across diverse in- and out-of-distribution tasks, DEORCH outperforms prior automatic MAS orchestration methods with fewer worker calls than competing learned orchestrators, remains effective when transferred to an entirely unseen worker pool without retraining, and shows consistent gains from both components.

## 1 INTRODUCTION

A multi-agent system (MAS) tackles a task with several language-model agents rather than one alone, letting them divide the work, verify one another, and combine their complementary strengths. Studies have shown that on some complex tasks a MAS outperforms each of its constituent agents (Wang et al., 2025; Kim et al., 2025; Nielsen et al., 2026). An expert-designed collaboration structure, specifying how the task is divided and how the agents work on it, often performs well; hand-designing one for every task, however, is impractical. Recent work therefore learns to build it automatically (Zhang et al., 2024; Dang et al., 2025; Gao et al., 2025; Cui et al., 2026).

A recent line of work trains an orchestrator that, given a task, decides how to decompose it into subtasks and how each subtask is handled by worker agents in a fixed pool (Gao et al., 2025; Chi et al., 2025; Cui et al., 2026). For instance, Conductor (Nielsen et al., 2026) learns, over a pool of powerful frozen workers, to decompose a task, assign each subtask to a worker, and specify how intermediate outputs are shared, training the orchestrator end-to-end on final task correctness. Relative to Conductor, MAS-Orchestra (Ke et al., 2026) does not model worker selection; it instead uses a fixed underlying worker and learns only to decompose the task and choose an operation for each subtask (e.g., a single worker, an ensemble, or a debate). Trained on outcome reward alone, these orchestrators learn effective coordination: by directing a fixed set of strong workers, a small orchestrator makes them collectively surpass any individual worker, and even rival much larger models, at a fraction of the cost.

Despite these successes, current automatic-MAS methods suffer from two limitations that reduce their effectiveness and practicality in realistic settings. (1) The trained orchestrator struggles to exploit unseen workers. Existing methods either ignore concrete worker selection or learn it directly into the orchestrator’s policy (Ke et al., 2026; Nielsen et al., 2026; Gao et al., 2025). The for mer cannot exploit differences in workers’ capabilities, while the latter cannot generalize its learned worker-selection strategy to unseen workers. This limitation is increasingly relevant in practice as new and stronger agents continually emerge: as the worker pool evolves, the orchestrator is either blind to which workers are best suited to each subtask or must be adapted to exploit newly available workers. (2) A single outcome reward confounds credit assignment between decomposition and operation decisions. The orchestrator makes two coupled decisions: how to decompose the task and which operation to run on each subtask (Ke et al., 2026; Cui et al., 2026). A failure can stem from either one: the same decomposition may succeed when a subtask is handled by an ensemble but fail when that subtask is left to a single worker, yet a single outcome reward cannot isolate which decision is responsible, leaving both with entangled learning signals. Moreover, once the subtasks are determined, the appropriate operation for each one depends on how well the available workers match that specific subtask: several moderately matched workers may favor an ensemble, wherea one clearly dominant match may favor a single worker. Without such subtask-specific information about the worker pool, the orchestrator cannot reliably distinguish these cases.

In this paper, we introduce DEORCH, an automatic MAS orchestration framework that addresses both limitations by decoupling planning from concrete worker selection. Specifically, DEORCH consists of two parts. (1) A worker-agnostic, two-stage planner with matchability feedback. The planner makes two decisions: how to decompose the task into subtasks and which collaboration operation to run on each. Rather than making both decisions at once, it proceeds in two stages. It first decomposes the task without observing the workers; the matcher then summarizes, for each subtask, how well the worker pool can support it through a compact, worker-identity-free matchability profile. Conditioned on this feedback, the planner chooses a collaboration operation for each subtask while keeping the decomposition fixed. This factorization enables separate credit assignment to decomposition and operation decisions during training, rather than training both from the same confounded outcome signal. It also localizes worker dependence: task decomposition is performed without worker information, while collaboration decisions adapt to the current pool through matchability feedback. Concrete worker selection remains entirely in the matcher, allowing the same planner to be applied to changed worker pools without retraining. (2) A lightweight matcher combining probe-based profiling with online adaptation. For each (subtask, operation) pair, the matcher selects suitable workers using both probe-based prior estimates of their capabilities and execution feedback collected during deployment. It first evaluates each worker on a fixed probe set, producing a behavioral profile from which its suitability for a new subtask is estimated based on semantically related probes. This provides a transferable cold-start estimate for new workers. A contextual bandit then refines these estimates from observed execution outcomes during deployment. Because worker-specific knowledge is confined to the matcher, a new or upgraded worker can be incorporated simply by evaluating it on the probe set, without retraining the planner or matcher.

We evaluate DEORCH across diverse in- and out-of-distribution tasks spanning math, code, multihop QA, tool use, and planning. DEORCH consistently outperforms existing automatic MAS orchestration baselines while using fewer worker calls, and remains effective when transferred to an entirely unseen worker pool without retraining. Ablations further validate both components: twostage planning, matchability feedback, and conditional credit assignment improve orchestration, while probe-based matching, complementarity-aware worker selection, and online adaptation improve worker selection.

## 2 RELATED WORK

Automatic MAS Orchestration Automatic MAS design replaces hand-crafted collaboration with systems that automate decisions such as task decomposition, agent roles and prompts, and the flow of information among agents (Wang et al., 2026b; Li et al., 2026d;c; 2025; Shang et al., 2025). One line uses off-the-shelf LLMs to design or search over MAS structures, without training a dedicated orchestration model (Zhang et al., 2025b; Fourney et al., 2024; Li & Ramakrishnan, 2026; Lin et al., 2026; Liu et al., 2026; Yun et al., 2026; Zhang et al., 2025c; Wang et al., 2026d; Niu et al., 2025; Ke et al., 2025). For example, METAAGENT (Zhang et al., 2025c) asks an LLM to instantiate a set of agents and construct a finite-state machine that routes among them, while MASS (Zhou et al., 2026) optimizes both collaboration topologies and the prompts within them through a three-stage search pipeline. A second line uses non-LM controllers or optimization mechanisms to choose the MAS structure (Jiang et al., 2026; Li et al., 2026a; Xu et al., 2026; Li et al., 2026c;b; Wu et al., 2026; Wang et al., 2026a; Leong et al., 2025). For example, G-DESIGNER (Zhang et al., 2024) uses a GNN-based generator to decide which agents should communicate for a given task, while MAAS (Zhang et al., 2025a) learns a supernet containing many candidate agentic operators and activates only the parts needed for each query. These methods primarily optimize structural decisions rather than generating free-form task decompositions and subtask instructions in language. A third and more closely related line trains a language model specifically for orchestration (Dang et al., 2025;

Wang et al., 2026c; Cui et al., 2026; Chi et al., 2025; Ruan et al., 2026; Huang et al., 2026). For example, CONDUCTOR (Nielsen et al., 2026) uses reinforcement learning to train an LM that jointly writes the subtasks and assigns each subtask to a worker. MAS-ORCHESTRA (Ke et al., 2026) emits the entire orchestration in one decision step, instantiating for each subtask one of a fixed set of sub-agent types such as self-consistency, debate, or self-refine. FLOWREASONER (Gao et al., 2025) generates for each query a Python program defining the flow of a multi-agent system, with its subtasks handled by the same worker model under different roles. Unlike these approaches, which either omit concrete worker selection or couple it to planning, DEORCH keeps the planner worker-agnostic and delegates worker selection to a separate matcher.

LLM Routing and Model Selection LLM routing selects an appropriate model for each query from a candidate pool (Lu et al., 2024; Tsiourvas et al., 2025; Mei et al., 2025). Many existing routers are trained offline. One line trains a parametric router to predict which model best matches the input query (Ong et al., 2025; Chen et al., 2024; Song et al., 2025; Somerstep et al., 2025). For example, P2L (Frick et al., 2025) trains a language-model router whose output scores correspond to the candidate models, tying the router itself to the model pool. Another line avoids training such a router and instead estimates a model’s performance on a new query from its behavior on a reference set (Shadid et al., 2025; Jitkrittum et al., 2026; Li, 2025b). AVENGERS-PRO (Zhang et al., 2025d), for example, maps a new query to semantically similar query clusters and routes according to the models’ observed performance on them; a newly added model only needs to be evaluated on the reference set. A separate line studies online routing, treating model selection as a contextualbandit problem and updating the router from deployment feedback, for example with LinUCB or Thompson sampling (Nguyen et al., 2025; Li, 2025a).

Credit Assignment in RL for LLMs Credit assignment arises whenever a single outcome reflects multiple decisions (Pignatelli et al., 2023). In LLM reasoning, an incorrect final answer gives little indication of which intermediate reasoning steps were useful or harmful, motivating process reward models (Lightman et al., 2024). In cooperative multi-agent learning, a shared team reward similarly obscures the contribution of individual agents, motivating value decomposition and counterfactual agent-level credit (Liang et al., 2024; Zhao & Xie, 2025). The same ambiguity appears in learned MAS orchestration, where a terminal outcome can entangle planning with worker selection and, within the planner, can further entangle task decomposition with operation decisions; recent work such as LEMON (Chen et al., 2026) localizes credit by counterfactually editing individual orchestration fields. To our knowledge, DEORCH is the first orchestration framework to use two-stage planning for RL credit assignment, separating decomposition from collaboration so that they receive distinct learning signals.

## 3 METHOD

Figure 1 gives an overview of DEORCH, which separates MAS orchestration into a worker-agnostic task planner and a separate matcher responsible for worker selection. Given a task, the languagemodel planner first decomposes it into a DAG of subtasks, whose edges encode dependencies. The matcher then summarizes how well the current worker pool can support each subtask through worker-identity-free matchability features, which the planner uses in a second pass to assign collaboration operations while keeping the decomposition fixed. Finally, the matcher binds each planned operation to suitable workers using its probe-based prior and online adaptation, and the resulting MAS is executed; execution outcomes are used to train the planner and update the matcher.

## 3.1 PROBLEM FORMULATION

Given a query x and a pool of worker agents $\mathcal { W } = \{ w _ { 1 } , . . . , w _ { M } \}$ , automatic MAS orchestration constructs a multi-agent workflow for solving x. We represent an orchestration by

$$
\begin{array} { r } { \mathcal { A } = ( D = ( S , E ) , \{ ( o _ { s } , n _ { s } ) \} _ { s \in S } , \{ B _ { s } \} _ { s \in S } ) , } \end{array}
$$

where D is a task-decomposition DAG whose nodes S are subtasks decomposed from x and whose directed edges E specify their dependencies. For each subtask s, $o _ { s } \in \mathcal { O }$ denotes a collaboration operation from {SINGLE, ENSEMBLE, DEBATE, SELECT, REVIEW}, and $n _ { s }$ its required number of workers, while $\check { B } _ { s } \subseteq \mathcal { W }$ denotes the concrete worker team assigned to execute it, with $| B _ { s } | = n _ { s }$

![](images/4953102a706ca434710e075edc9a7b65d41ad60fa43613081e94dfae6895ebb4.jpg)  
Figure 1: Overview of DEORCH.

Executing A produces a final output y, whose quality is measured by an evaluator $Q ( x , y )$ . The orchestration problem therefore involves deciding how to decompose the task, how each subtask should be collaboratively executed, and which available workers should carry out that execution.

## 3.2 WORKER-AGNOSTIC TWO-STAGE PLANNER

Existing LM orchestrators often couple task planning with worker selection in the same policy (Nielsen et al., 2026; Cui et al., 2026), tying the orchestrator to a particular worker pool and requiring retraining when the pool changes. Our planner is instead worker-agnostic: it decides how to decompose the task and how each subtask should be collaboratively executed, while all concrete worker selection is delegated to the matcher. Workers can therefore be added to or removed from the pool without retraining the planner. The planner operates in two stages, separating task decomposition from collaboration planning for two reasons. First, a shared terminal reward entangles the two decisions: failure does not reveal whether the decomposition or collaboration strategy was responsible. Separating the two decisions allows collaboration choices to be sampled and compared while holding the decomposition fixed, yielding more targeted credit for collaboration decisions during RL training. Second, collaboration should depend on how well the current worker pool supports each subtask. For example, a subtask that is easily handled by the available workers may require only a single worker, whereas a poorly matched subtask may benefit from ensemble execution. The second stage therefore conditions operation selection on worker-identity-free matchability feedback, adapting collaboration to the pool without exposing worker identities.

Stage I: task decomposition. Given an input query x, the planner generates a task-decomposition DAG $D = ( S , E ) \ { \stackrel { \circ } { \sim } } \ \pi _ { \theta } ^ { \mathrm { d e c } } ( \cdot \mid x )$ , where $S = \{ s _ { 1 } , \ldots , s _ { L } \}$ contains natural-language subtasks and E specifies their dependencies. At this stage, the planner decomposes the task from the query alone, determining what should be solved and in what dependency order, while leaving collaboration planning and worker selection to subsequent stages

Stage II: collaboration planning. After generating the decomposition, the matcher evaluates how well the worker pool supports each subtask and returns a matchability feature $m _ { s } = m ( s ; \mathcal { W } )$ , a fixed-dimensional, worker-identity-free summary of pool support (Section 3.3). Conditioned on the query, fixed decomposition, and matchability features, the planner performs a second generation:

$$
\{ ( o _ { s } , n _ { s } ) \} _ { s \in S } \sim \pi _ { \theta } ^ { \mathrm { o p } } \left( \cdot \mid x , D , \{ m _ { s } \} _ { s \in S } \right) ,
$$

where $o _ { s } \in \mathcal { O }$ specifies the collaboration operation used for subtask s and $n _ { s }$ denotes the required team size. Stage II keeps the decomposition fixed and augments each subtask with $\left( o _ { s } , n _ { s } \right)$ , adapting collaboration to the current pool through matchability feedback while leaving concrete worker selection to the matcher.

Collaboration operations. We consider a compact operation set O = {SINGLE, ENSEMBL $\mathbf { \delta } . \mathrm { E } ,$ DEBATE, SELECT, REVIEW}. Each operation determines how the assigned worker or worker team interacts when solving s. Specifically, SINGLE uses one worker; ENSEMBLE aggregates responses from multiple workers; DEBATE lets multiple workers exchange and refine their responses; SELECT uses an adjudicator to choose among candidate responses; and REVIEW uses reviewers to critique and refine an initial response. By composing these operations over the subtask DAG, the planner can instantiate common MAS structures.

## 3.3 PROBE-BASED MATCHER WITH ONLINE ADAPTATION

The matcher combines a static probe-based matcher with an online contextual bandit. The static matcher estimates worker suitability for a subtask from performance on semantically related probes, while the online bandit refines these estimates from deployment feedback. The static matcher is used for planner training and deployment, while online adaptation is used only at deployment.

Probe-based matching with density correction. We construct a fixed probe set of N diverse tasks. Each worker w is evaluated once on the same probes, producing a behavioral signature $\mathbf { o } _ { w } \ = \ ( o _ { w , 1 } , \dotsc , o _ { w , N } ) \ \in \ \{ 0 , 1 \} ^ { N }$ , where $o _ { w , j } ~ = ~ 1$ if worker $w$ succeeds on probe $j$ and 0 otherwise. Given a subtask $s ,$ we estimate worker suitability from its outcomes on semantically related probes. Let $e _ { s }$ and $e _ { j }$ denote the embeddings of s and probe $j .$ . We define the subtask– probe relevance as $\kappa _ { j } ( s ) = \overset { \vartriangle } { \exp } ( ( \cos ( e _ { s } , e _ { j } ) - 1 ) / h _ { \mathrm { r e l } } )$ , where smaller $h _ { \mathrm { r e l } }$ focuses on more similar probes. Directly normalizing these weights can overemphasize semantic regions containing many near-duplicate probes. We therefore estimate the local probe density

$$
q _ { j } = \sum _ { \ell = 1 } ^ { N } \exp \left( \frac { \cos ( e _ { j } , e _ { \ell } ) - 1 } { h _ { \mathrm { d e n } } } \right) , \qquad { \bar { q } } _ { j } = \frac { q _ { j } } { \mathrm { m e d i a n } _ { \ell } q _ { \ell } } ,
$$

where $h _ { \mathrm { d e n } }$ controls the neighborhood scale. The final probe weights and worker suitability are

$$
\alpha _ { j } ( s ) = \frac { \kappa _ { j } ( s ) ( \bar { q } _ { j } + \epsilon ) ^ { - \nu } } { \sum _ { \ell = 1 } ^ { N } \kappa _ { \ell } ( s ) ( \bar { q } _ { \ell } + \epsilon ) ^ { - \nu } } , \qquad \hat { u } _ { w } ( s ) = \sum _ { j = 1 } ^ { N } \alpha _ { j } ( s ) o _ { w , j } .
$$

The density correction reduces the influence of locally redundant probes while preserving genuinely distinct coverage. Since $\alpha _ { j } ( s )$ depends only on the subtask and the fixed probe set, a new worker can be scored directly from its probe signature without retraining the matcher.

Matchability feedback. The second-stage planner should know how well the current worker pool can support a subtask without observing worker identities. We therefore summarize the pool with a fixed six-dimensional feature $m ( s ) = \overline { { [ } } B ( s ) , U ( s ) , L ( s ) , G ( s ) , D ( s ) , C ( s ) ]$ ]. The six coordinates capture complementary aspects of pool capability: B measures the strength of the best-matched single worker; $U$ and $\dot { L }$ describe the benefit and risk of adding a complementary second worker; G and $D$ summarize the amount and breadth of complementary capability available across additional workers; and $C$ measures how well the probe set supports the estimate for the current subtask. Importantly, these statistics describe the pool without encoding worker identities or its size. Hence the planner receives the same fixed interface as workers are added or removed, while the matcher recomputes the summary from the current pool.

Let $w _ { 1 } = \arg \operatorname* { m a x } _ { w } \hat { u } _ { w } ( s )$ be the best-matched worker, so that $B ( s ) = \hat { u } _ { w _ { 1 } } ( s )$ . Starting from $w _ { 1 }$ we greedily add workers according to their marginal increase in weighted probe coverage, up to a maximum team size $K$ . Let $g _ { k }$ denote the marginal coverage gain from the k-th addition. The first added worker, denoted $w _ { 2 } .$ , maximally covers the probe failures of $w _ { 1 }$ , giving

$$
U ( s ) = \sum _ { j } \alpha _ { j } ( s ) ( 1 - o _ { w _ { 1 } , j } ) o _ { w _ { 2 } , j } , \qquad L ( s ) = \sum _ { j } \alpha _ { j } ( s ) o _ { w _ { 1 } , j } ( 1 - o _ { w _ { 2 } , j } ) ,
$$

where U measures rescued failure mass and L measures converse spoilage mass. In particular, $g _ { 1 } = U ( s )$ . The resulting gains $g _ { 1 } , \ldots , g _ { K - 1 }$ are summarized as

$$
G ( s ) = \sum _ { k } g _ { k } , \qquad D ( s ) = \frac { G ( s ) ^ { 2 } } { \sum _ { k } g _ { k } ^ { 2 } } , \qquad C ( s ) = \sum _ { j } \alpha _ { j } ( s ) \cos ( e _ { s } , e _ { j } ) .
$$

Here, $G$ measures the total complementary coverage available beyond the best single worker, D reflects how broadly that coverage is distributed across additional workers, and $\bar { C }$ measures the semantic support provided by the probes used for the estimate.

Operation-specific worker selection. Once the planner specifies $\left( s , o _ { s } , n _ { s } \right)$ , the matcher selects workers using subtask suitability and behavioral complementarity. For SINGLE, we select the worker with the highest suitability score $\hat { u } _ { w } ( s )$ . For multi-worker operations, we use maximal marginal relevance (MMR) (Carbonell & Goldstein, 1998) for team construction. We initialize B with the highest-suitability worker and then greedily add workers until $| B | = n _ { s }$ . We define the directional complementarity of workers w and v and select each additional worker according to

$$
\delta _ { w  v } ( s ) = \sum _ { j = 1 } ^ { N } \alpha _ { j } ( s ) ( 1 - o _ { v , j } ) o _ { w , j } , \qquad w ^ { \star } = \arg \operatorname* { m a x } _ { w \notin B } [ \lambda _ { o } \hat { u } _ { w } ( s ) - ( 1 - \lambda _ { o } ) \operatorname* { m a x } _ { v \in B } ( 1 - \delta _ { w  v } ( s ) ) ] .
$$

This criterion balances competence with failure-region complementarity to the selected team $B .$ We use it for ENSEMBLE and DEBATE. Operations with asymmetric roles use specialized rules: SELECT first assigns the highest-suitability worker as the adjudicator and uses MMR to choose the remaining proposers, whereas REVIEW assigns the highest-suitability worker as the producer and greedily selects reviewers that best cover its failure region.

Online adaptation. The probe-based matcher provides a cold-start estimate from worker behavior, while deployment experience provides evidence of worker utility on the target distribution. We therefore augment the static matcher with contextual Thompson sampling. Each worker w has a probe-derived capability representation $g ( w )$ , and we maintain a Gaussian posterior over a shared parameter $\eta$ modeling context-dependent task–worker interactions. Before each MAS execution, we sample $\tilde { \eta }$ from the posterior and combine the probe-based estimate with the interaction between the subtask representation $z ( s )$ and worker representation $g ( w )$ to form the deployment-adapted suitability $\tilde { u } _ { w } \bar { ( } s ) \ = \ \hat { u } _ { w } ( s ) + \tilde { \eta } ^ { \top } ( z ( s ) \odot g \bar { ( w ) } )$ . At deployment, $\tilde { u } _ { w } ( s )$ replaces $\hat { u } _ { w } ( s )$ in the operation-specific worker-selection rules, while the probe-derived complementarity statistics remain unchanged. After the MAS is executed, its terminal utility is treated as noisy bandit feedback fo the selected worker assignment and used to update the shared posterior. The bandit does not directly attribute a single outcome to individual workers; rather, repeated feedback across tasks and worker assignments allows deployment-specific task–worker interactions to be learned over time.

## 3.4 PLANNER TRAINING WITH CONDITIONAL CREDIT ASSIGNMENT

We post-train the planner from terminal execution outcomes using nested conditional rollouts. For each query, we sample multiple decompositions and, for each fixed decomposition, multiple collaboration plans. Comparing collaboration plans under the same decomposition provides credit for Stage II, while averaging their outcomes estimates how good that decomposition is for Stage I. This yields separate learning signals for the two planning stages from the same terminal outcomes.

Conditional credit assignment. For each training query x, we sample J decompositions and, for each fixed decomposition, k collaboration plans:

$$
\begin{array} { r l r } & { D _ { j } \sim \pi _ { \theta } ^ { \mathrm { d e c } } ( \cdot  { \mid } x ) , } & { j = 1 , \ldots , J , } \\ & { O _ { j , l } \sim \pi _ { \theta } ^ { \mathrm { o p } } ( \cdot  { \mid } x , D _ { j } , \{ m _ { s } \} _ { s \in S _ { j } } ) , } & { l = 1 , \ldots , k . } \end{array}
$$

Each resulting plan is assigned concrete workers by the matcher and executed, yielding terminal quality $Q _ { j , l } = \stackrel { \cdot } { Q } ( x , y _ { j , l } )$ . For Stage II, all k rollouts under the same $D _ { j }$ share the same upstream decomposition, so their relative outcomes provide credit for the collaboration decisions. We first form a cost-adjusted reward and then compute its relative advantage within the group:

$$
R _ { j , l } ^ { \mathrm { o p } } = Q _ { j , l } \exp ( - \beta c _ { j , l } ) , \qquad A _ { j , l } ^ { \mathrm { o p } } = R _ { j , l } ^ { \mathrm { o p } } - \frac { 1 } { k } \sum _ { l ^ { \prime } = 1 } ^ { k } R _ { j , l ^ { \prime } } ^ { \mathrm { o p } } ,
$$

where $c _ { j , l }$ is the number of worker calls incurred by rollout $( j , l )$ and $\beta$ controls the cost penalty. For Stage I, we first average over downstream collaboration choices and then compare the resulting decomposition-level scores:

$$
\bar { Q } _ { j } = \frac { 1 } { k } \sum _ { l = 1 } ^ { k } Q _ { j , l } , \qquad A _ { j } ^ { \mathrm { d e c } } = \bar { Q } _ { j } - \frac { 1 } { J } \sum _ { j ^ { \prime } = 1 } ^ { J } \bar { Q } _ { j ^ { \prime } } .
$$

We use $A ^ { \mathrm { o p } }$ and $A ^ { \mathrm { d e c } }$ to optimize the corresponding Stage-II and Stage-I generations, respectively.

Cost-aware collaboration planning. Existing learned orchestrators such as CONDUCTOR and MAS-ORCHESTRA primarily optimize task quality without explicitly accounting for collaboration cost. Since more intensive collaboration typically incurs more worker calls and can also improve performance, a quality-only objective may systematically favor heavier operations. This cost–quality tradeoff is directly controlled by Stage II. In contrast, Stage I determines the task decomposition, and decomposition granularity by itself does not indicate solution quality: finer or coarser decompositions can each be preferable depending on the task. Our two-stage formulation therefore allows us to train Stage I with task quality alone while applying the cost-aware objective only to Stage II. The multiplicative reward further ensures that lower cost is valuable only when accompanied by task success: a failed rollout does not receive a higher reward merely because it uses fewer worker calls.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Models. We initialize the planner from Qwen3-8B and perform full-parameter RL post-training. Unless otherwise specified, all methods operate over the same heterogeneous pool of six worker models: Qwen3-4B-Instruct-2507, gemma-4-12B-it, NVIDIA-Nemotron-Nano-9B-v2, Qwen3.5-9B, granite-4.1-8b, and Ministral-3-14B-Instruct-2512. We use Qwen3-Embedding-8B as the text encoder for semantic matching and retrieval.

Data and benchmarks. We use disjoint data for probe construction, planner post-training, and final evaluation. We evaluate in-distribution on held-out MuSiQue and TravelPlanner examples, and out-of-distribution on Omni-MATH-2, LiveCodeBench v6, BBEH, NaturalPlan, PlanBenchmystery, and GAIA, spanning math, code, multi-hop reasoning, tool use, and planning. Probe and training datasets, data sources, splits, and sample counts are provided in Appendix B.1.

Baselines. We compare against several complementary baselines. BEST SINGLE AGENT uses gemma-4-12B-it, the best-performing worker in hindsight. As a fixed hand-designed MAS baseline, we include INDEPENDENT coordination from prior work (Kim et al., 2025). FRONTIER-PLANNER uses Qwen3.8-27B as an off-the-shelf planner, with the rest of the system following DEORCH. UNTRAINED PLANNER uses the same planner initialization as DEORCH but without orchestration-specific RL post-training. We further compare with learned automatic orchestrators, including CONDUCTOR (Nielsen et al., 2026) and MAS-ORCHESTRA (Ke et al., 2026), adapted to our worker setting. Additional baseline details are provided in Appendix B.3.

## 4.2 MAIN RESULTS

Table 1 summarizes the main results. DEORCH achieves the best overall performance across both indistribution and out-of-distribution benchmarks. It substantially outperforms BEST SINGLE AGENT and INDEPENDENT, showing that its gains cannot be explained by relying on the strongest worker or by simply executing all workers in parallel. Both FRONTIER-PLANNER and UNTRAINED PLANNER perform considerably worse, indicating that strong general-purpose planning or an untrained planner does not substitute for orchestration-specific post-training. Compared with learned automatic orchestrators, DEORCH consistently outperforms CONDUCTOR and MAS-ORCHESTRA, demonstrating the effectiveness and competitiveness of our approach. Moreover, DEORCH uses fewer worker calls than both learned orchestrators, indicating that its performance gains do not come from making more worker calls.

Table 1: Main results across in-distribution and out-of-distribution benchmarks.
<table><tr><td rowspan="2">Method</td><td colspan="2">In-Distribution</td><td colspan="6">Out-of-Distribution</td><td colspan="2">Overall</td></tr><tr><td>MuSiQue</td><td>TP</td><td>Omni</td><td>LCB</td><td>BBEH</td><td>NP</td><td>PB-M</td><td>GAIA</td><td>|Avg.</td><td>Calls↓</td></tr><tr><td>BEST SINGLE AGENT</td><td>72.8</td><td>13.6</td><td>42.5</td><td>69.9</td><td>38.4</td><td>51.1</td><td>19.4</td><td>6.1</td><td>39.2</td><td>1.00</td></tr><tr><td>INDEPENDENT</td><td>72.0</td><td>9.2</td><td>47.4</td><td>76.4</td><td>45.2</td><td>63.1</td><td>25.9</td><td>28.5</td><td>46.0</td><td>7.00</td></tr><tr><td>FRONTIER-PLANNER</td><td>59.1</td><td>3.5</td><td>38.9</td><td>71.0</td><td>37.4</td><td>45.5</td><td>9.1</td><td>28.5</td><td>36.6</td><td>4.67</td></tr><tr><td>UNTRAINED PLANNER</td><td>60.2</td><td>3.3</td><td>38.1</td><td>70.4</td><td>31.8</td><td>50.4</td><td>25.2</td><td>26.7</td><td>38.3</td><td>4.92</td></tr><tr><td>CONDUCTOR</td><td>70.3</td><td>9.9</td><td>41.2</td><td>75.0</td><td>37.1</td><td>52.9</td><td>38.0</td><td>27.9</td><td>44.0</td><td>6.16</td></tr><tr><td>MAS-ORCHESTRA</td><td>73.4</td><td>15.1</td><td>49.0</td><td>76.9</td><td>45.0</td><td>61.8</td><td>41.6</td><td>29.7</td><td>49.1</td><td>5.63</td></tr><tr><td>DEORCH</td><td>76.9</td><td>17.8</td><td>51.7</td><td>79.2</td><td>47.9</td><td>64.7</td><td>43.1</td><td>30.9</td><td>51.5</td><td>5.35</td></tr></table>

## 4.3 GENERALIZATION TO AN UNSEEN WORKER POOL

Table 2 evaluates generalization to an entirely unseen four-worker pool (Appendix B.3), with no planner or matcher retraining and online adaptation disabled. DEORCH remains effective, outper forming both BEST SINGLE AGENT and INDEPENDENT by clear margins. In contrast, CONDUC-TOR degrades from 44.0 to 40.9: its advantage over BEST SINGLE AGENT nearly disappears, its gap to INDEPENDENT widens, and its gap to DEORCH increases from 7.5 to 9.3 points. These results support the benefit of decoupling planning from concrete worker selection.

Table 2: Performance under an unseen worker pool.
<table><tr><td rowspan="2">Method</td><td colspan="2">In-Distribution</td><td colspan="6">Out-of-Distribution</td><td>Overall</td></tr><tr><td>MuSiQue</td><td>TP</td><td>Omni</td><td>LCB</td><td>BBEH</td><td>NP</td><td>PB-M</td><td>GAIA</td><td>Avg.</td></tr><tr><td>BEST SINGLE AGENT</td><td>69.0</td><td>6.2</td><td>45.9</td><td>75.9</td><td>47.7</td><td>40.3</td><td>29.0</td><td>12.1</td><td>40.8</td></tr><tr><td>INDEPENDENT</td><td>72.4</td><td>5.0</td><td>50.6</td><td>81.2</td><td>48.4</td><td>60.4</td><td>41.6</td><td>20.0</td><td>47.5</td></tr><tr><td>FRONTIER-PLANNER</td><td>61.4</td><td>3.1</td><td>44.4</td><td>76.3</td><td>46.8</td><td>52.2</td><td>18.1</td><td>21.8</td><td>40.5</td></tr><tr><td>CONDUCTOR</td><td>63.7</td><td>4.6</td><td>45.5</td><td>75.5</td><td>46.2</td><td>48.8</td><td>23.7</td><td>18.8</td><td>40.9</td></tr><tr><td>DEORCH</td><td>75.5</td><td>11.3</td><td>52.3</td><td>83.0</td><td>49.7</td><td>62.3</td><td>44.3</td><td>23.0</td><td>50.2</td></tr></table>

## 4.4 ANALYSIS OF THE PLANNER

We construct controlled variants to isolate the contributions of the planner design and training objective. JOINT PLANNING generates decomposition and collaboration decisions in one pass under a shared terminal reward. TWO-STAGE separates the two decisions but uses neither matchability feedback nor conditional credit. + MATCHABILITY provides the Stage-II planner with m(s), and + CONDITIONAL CREDIT further introduces our nested credit assignment. The full DEORCH additionally applies the cost-aware Stage-II utility. We also evaluate DECOMPOSITION + SINGLE, which preserves the learned decomposition but fixes every subtask to SINGLE, measuring how much performance remains without Stage-II collaboration choices. All trained variants use the same planner backbone, worker pool, and rollout budget. Table 3 shows consistent improvements as the planner components are introduced. Two-stage planning and matchability feedback each improve performance, while conditional credit assignment yields the largest gain, supporting separate credit for decomposition and collaboration decisions. DECOMPOSITION + SINGLE still achieves an overall average of 50.4, showing that the learned decomposition alone is already strong, while learned Stage-II collaboration provides additional gains. Finally, the cost-aware objective reduces worker calls from 7.78 to 5.35 with only a modest decrease in average performance from 51.9 to 51.5, substantially improving the quality–cost trade-off.

## 4.5 ANALYSIS OF THE MATCHER

Offline worker matching. We evaluate the static matcher independently of the learned planner. For in-distribution evaluation, examples from each probe source are randomly split 80/20 for probe construction and held-out evaluation. We additionally evaluate leave-one-source-out (LOSO) generalization on five representative sources by excluding the target source from the probe set. We compare against random routing, the strongest single worker, majority voting over all six workers, probe matching without density correction, and ORACLE SINGLE, which selects a successful worker in hindsight as an upper bound. Table 4 shows that the full probe matcher achieves the strongest nonoracle performance in both settings. Its advantage narrows under LOSO-OOD evaluation, where no probe examples from the target source are available, but it still outperforms the strongest singleworker reference. Density correction provides consistent gains in both settings, supporting its role in reducing the influence of locally redundant probes. Finally, the sizable gap to ORACLE SINGLE indicates substantial remaining headroom in worker selection.

Table 3: Ablation of the planner.
<table><tr><td>Variant</td><td>ID Avg.</td><td>OOD Avg.</td><td>Avg.</td><td>Calls ↓</td></tr><tr><td rowspan="3">Joint Planning Two-Stage</td><td>43.2</td><td>50.2</td><td>48.5</td><td>6.29</td></tr><tr><td>43.8</td><td>50.9</td><td>49.2</td><td>7.33</td></tr><tr><td>44.3</td><td>51.4</td><td>49.6</td><td>7.52</td></tr><tr><td>+ Matchability + Conditional Credit</td><td>47.7</td><td>53.4</td><td>51.9</td><td>7.78</td></tr><tr><td>Decomposition + Single</td><td>46.1</td><td>51.8</td><td>50.4</td><td>1.68</td></tr><tr><td>DEORCH</td><td>47.4</td><td>52.9</td><td>51.5</td><td>5.35</td></tr></table>

Table 4: Offline evaluation of worker matching.
<table><tr><td>Method</td><td>ID Avg. ↑</td><td>LOSO-OOD Avg. ↑</td><td>Calls ↓</td></tr><tr><td>Random</td><td>59.44</td><td>69.73</td><td>1</td></tr><tr><td>BEST SINGLE WORKER</td><td>68.33</td><td>75.26</td><td>1</td></tr><tr><td>Majority vote</td><td>67.92</td><td>74.59</td><td>6</td></tr><tr><td>Probe Matcher w/o Density</td><td>69.70</td><td>75.36</td><td>1</td></tr><tr><td>Probe Matcher</td><td>70.24</td><td>75.45</td><td>1</td></tr><tr><td>ORACLE SINGLE</td><td>82.68</td><td>87.17</td><td>1</td></tr></table>

Table 5: Online adaptation with increasing deployment feedback.
<table><tr><td>Adaptation Examples</td><td>0 3K</td><td>6K</td><td>9K</td><td>12K</td><td>15K</td></tr><tr><td>Overall Avg.</td><td>51.3</td><td>51.7 52.1</td><td>52.1</td><td>52.6</td><td>52.9</td></tr></table>

Online adaptation. We finally evaluate whether contextual bandit adaptation improves the matcher’s worker-selection ability during deployment. We split the evaluation data into an adaptation stream and a disjoint held-out evaluation set, with the split detailed in Appendix B.2. The planner and the probe-based component of the matcher remain fixed throughout the experiment. Adaptation examples are processed sequentially, and the shared Thompson-sampling posterior is updated after each execution using the observed terminal outcome. After every 3,000 adaptation examples, we evaluate the current matcher on the fixed held-out set. Performance is computed separately on each benchmark and macro-averaged across the eight benchmarks. The zero-example checkpoint corresponds to the static probe matcher without deployment feedback. Table 5 shows that performance generally improves as deployment feedback accumulates, increasing from 51.3 without online adaptation to 52.9 after 15K adaptation examples. This result demonstrates the effectiveness of our contextual bandit in progressively improving worker selection during deployment.

## 5 CONCLUSION

We introduced DEORCH, a multi-agent orchestration framework that decouples worker-agnostic planning from concrete worker selection and uses two-stage planning for separate credit assignment to decomposition and collaboration decisions. Across diverse tasks, DEORCH outperforms fixed and learned MAS baselines and remains effective under worker-pool changes without retraining.

## AI USE STATEMENT

Generative AI tools were only used to assist with language editing and polishing of the manuscript, including improving clarity, readability, grammar, and phrasing. All AI-assisted edits were reviewed by the authors. The authors take full responsibility for the final content of this work.

## REFERENCES

Jaime G Carbonell and Jade Goldstein. The use of mmr, diversity-based reranking for reordering documents and producing summaries. In SIGIR, 1998.

Shuhao Chen, Weisen Jiang, Baijiong Lin, James Kwok, and Yu Zhang. Routerdc: Query-based router by dual contrastive learning for assembling large language models. Advances in Neural Information Processing Systems, 37:66305–66328, 2024.

Xudong Chen, Yixin Liu, Hua Wei, and Kaize Ding. Lemon: Learning executable multi-agent orchestration via counterfactual reinforcement learning. arXiv preprint arXiv:2605.14483, 2026.

Zewen Chi, Li Dong, Qingxiu Dong, Yaru Hao, Xun Wu, Shaohan Huang, and Furu Wei. The era of agentic organization: Learning to organize with language models. arXiv preprint arXiv:2510.26658, 2025.

Zhiqing Cui, Haotong Xie, Jiahao Yuan, Cheng Yang, Hanqing Wang, Yuxin Wu, Yifan Wu, Siru Zhong, Tao Yu, Yifu Guo, et al. Uno-orchestra: Parsimonious agent routing via selective delegation. arXiv preprint arXiv:2605.05007, 2026.

Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, and Maosong Sun. Multi-agent collaboration via evolving orchestration. In Advances in Neural Information Processing Systems, volume 38, 2025.

Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, et al. Magentic-one: A generalist multi-agent system for solving complex tasks. arXiv preprint arXiv:2411.04468, 2024.

Evan Frick, Connor Chen, Joseph Tennyson, Tianle Li, Wei-Lin Chiang, Anastasios N Angelopoulos, and Ion Stoica. Prompt-to-leaderboard. arXiv preprint arXiv:2502.14855, 2025.

Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, and Tianyu Pang. Flowreasoner: Reinforcing query-level meta-agents. arXiv preprint arXiv:2504.15257, 2025.

Jing Huang, Lidong Zhang, Mutian Bao, Yadong Li, Xingzhong Xu, Jinjian Zhang, Jie Liu, Ming Kong, and Qiang Zhu. Mas-architect: Declarative multi-agent system design via separation of concerns. In Forty-third International Conference on Machine Learning, 2026.

Eric Hanchen Jiang, Levina Li, Rui Sun, Xiao Liang, Yubei Li, Yuchen Wu, Haozheng Luo, Hengli Li, Zhi Zhang, Zhaolu Kang, et al. Agent q-mix: Selecting the right action for llm multi-agent systems through reinforcement learning. arXiv preprint arXiv:2604.00344, 2026.

Wittawat Jitkrittum, Harikrishna Narasimhan, Ankit Singh Rawat, Jeevesh Juneja, Congchao Wang, Zifeng Wang, Alec Go, Chen-Yu Lee, Pradeep Shenoy, Rina Panigrahy, et al. Universal model routing for efficient llm inference. In International Conference on Learning Representations, volume 2026, pp. 10169–10218, 2026.

Zixuan Ke, Austin Xu, Yifei Ming, Xuan-Phi Nguyen, Ryan Chin, Caiming Xiong, and Shafiq Joty. Mas-zero: Designing multi-agent systems with zero supervision. arXiv preprint arXiv:2505.14996, 2025.

Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram, Jiayu Wang, Semih Yavuz, Caiming Xiong, and Shafiq Joty. Mas-orchestra: Understanding and improving multi-agent reasoning through holistic orchestration and controlled benchmarks. arXiv preprint arXiv:2601.14652, 2026.

Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Yun Liu, et al. Towards a science of scaling agent systems. arXiv preprint arXiv:2512.08296, 2025.

Hui Yi Leong, Yuheng Li, Yuqing Wu, Wenwen Ouyang, Wei Zhu, Jiechao Gao, and Wei Han. Amas: Adaptively determining communication topology for llm-based multi-agent system. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 2061–2070, 2025.

Ao Li, Yuexiang Xie, Songze Li, Fugee Tsung, Bolin Ding, and Yaliang Li. Agent-oriented planning in multi-agent systems. In International Conference on Learning Representations, volume 2025, pp. 19495–19517, 2025.

Haoran Li, Shulun Chen, Shaoyuan Sun, and Hanchen Wang. Multi-agent coordination adaptation via structure-guided orchestration. arXiv preprint arXiv:2605.25746, 2026a.

Sha Li and Naren Ramakrishnan. Experience as a compass: Multi-agent rag with evolving orchestration and agent prompts. arXiv preprint arXiv:2604.00901, 2026.

Shiyuan Li, Yixin Liu, Qingsong Wen, Chengqi Zhang, and Shirui Pan. Assemble your crew: Automatic multi-agent communication topology design via autoregressive graph generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 23142–23150, 2026b.

Shiyuan Li, Yixin Liu, Yu Zheng, Mei Li, Quoc Viet Hung Nguyen, and Shirui Pan. Ofa-mas: Onefor-all multi-agent system topology design based on mixture-of-experts graph generative models. In Proceedings of the ACM Web Conference 2026, pp. 1333–1344, 2026c.

Yang Li. Llm bandit: Cost-efficient llm generation via preference-conditioned dynamic routing. arXiv preprint arXiv:2502.02743, 2025a.

Yang Li. Rethinking predictive modeling for llm routing: When simple knn beats complex learned routers. arXiv preprint arXiv:2505.12601, 2025b.

Yu Li, Lehui Li, Zhihao Wu, Qingmin Liao, Jianye Hao, Kun Shao, and Fengli Xu. Agentswift: Efficient llm agent design via value-guided hierarchical search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 31843–31851, 2026d.

Yongheng Liang, Hejun Wu, Haitao Wang, and Hao Cai. Asynchronous credit assignment for multiagent reinforcement learning. arXiv preprint arXiv:2408.03692, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024.

Hehai Lin, Yu Yan, Zixuan Wang, Bo Xu, Sudong Wang, Weiquan Huang, Ruochen Zhao, Minzhi Li, and Chengwei Qin. Unified-mas: Universally generating domain-specific nodes for empowering automatic multi-agent systems. arXiv preprint arXiv:2603.21475, 2026.

Guangyi Liu, Haojun Lin, Huan Zeng, Heng Wang, and Quanming Yao. Mas-on-the-fly: Dynamic adaptation of llm-based multi-agent systems at test time. arXiv preprint arXiv:2602.13671, 2026.

Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren Zhou. Routing to the expert: Efficient reward-guided ensemble of large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1964–1974, 2024.

Kai Mei, Wujiang Xu, Minghao Guo, Shuhang Lin, and Yongfeng Zhang. Omnirouter: Budget and performance controllable multi-llm routing. ACM SIGKDD Explorations Newsletter, 27(2): 107–116, 2025.

Duy M. H. Nguyen, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. LASeR: Learning to adaptively select reward models with multi-arm bandits. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-3895.

Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, and Yujin Tang. Learning to orchestrate agents in natural language with the conductor. In International Conference on Learning Representations, volume 2026, pp. 135686–135724, 2026.

Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, and Tongliang Liu. Flow: Modularized agentic workflow automation. In International Conference on Learning Representations, volume 2025, pp. 74949–74977, 2025.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, Mohammed Kadous, and Ion Stoica. Routellm: Learning to route llms from preference data. In International Conference on Learning Representations, volume 2025, pp. 34433–34448, 2025.

Eduardo Pignatelli, Johan Ferret, Matthieu Geist, Thomas Mesnard, Hado van Hasselt, Olivier Pietquin, and Laura Toni. A survey of temporal credit assignment in deep reinforcement learning. arXiv preprint arXiv:2312.01072, 2023.

Jianhao Ruan, Zhihao Xu, Yiran Peng, Fashen Ren, Zhaoyang Yu, Xinbing Liang, Jinyu Xiang, Yongru Chen, Bang Liu, Chenglin Wu, et al. Aorchestra: Automating sub-agent creation for agentic orchestration. arXiv preprint arXiv:2602.03786, 2026.

Ahmad Shadid, Rahul Kumar, and Mohit Mayank. Ori: O routing intelligence. arXiv preprint arXiv:2502.10051, 2025.

Yu Shang, Yu Li, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, and Yong Li. Agentsquare: Automatic llm agent search in modular design space. In International Conference on Learning Representations, volume 2025, pp. 3841–3865, 2025.

Seamus Somerstep, Felipe Maia Polo, Allysson Flavio Melo de Oliveira, Prattyush Mangal, M´ırian Silva, Onkar Bhardwaj, Mikhail Yurochkin, and Subha Maity. Carrot: A cost aware rate optimal router. arXiv preprint arXiv:2502.03261, 2025.

Wei Song, Zhenya Huang, Cheng Cheng, Weibo Gao, Bihan Xu, GuanHao Zhao, Fei Wang, and Runze Wu. Irt-router: Effective and interpretable multi-llm routing via item response theory. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15629–15644, 2025.

Asterios Tsiourvas, Wei Sun, and Georgia Perakis. Causal LLM routing: End-to-end regret minimization from observational data. In Advances in Neural Information Processing Systems, volume 38, 2025.

Jiye Wang, Yu Wang, Jianbin Li, Shiduo Yang, Kenan Guo, and Yuanhe Zhao. Neuralfsm: Adaptive multi-agent coordination via learning finite-state execution policy. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 33414–33436, 2026a.

Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Y Zou. Mixture-of-agents enhances large language model capabilities. In International Conference on Learning Representations, volume 2025, pp. 33944–33963, 2025.

Kun Wang, Guibin Zhang, ManKit Ye, Xinyu Deng, Dongxia Wang, Xiaobin Hu, Jinyang Guo, Yang Liu, and Yufei Guo. MAS<sup>2</sup>: Self-generative, self-configuring, self-rectifying multi-agent systems. In International Conference on Learning Representations, volume 2026, pp. 113586– 113613, 2026b.

Siyu Wang, Ruotian Lu, Zhihao Yang, Yuchao Wang, Yanzhou Zhang, Lei Xu, Qimin Xu, Guojun Yin, Cailian Chen, and Xinping Guan. Agentconductor: Topology evolution for multi-agent competition-level code generation. arXiv preprint arXiv:2602.17100, 2026c.

Yimeng Wang, Jiaxing Zhao, Hongbin Xie, Hexing Ma, Yuzhen Lei, Shuangxue Liu, Xuan Song, Zichen Zhang, and Haoran Zhang. Metagen: Self-evolving roles and topologies for multi-agent llm reasoning. arXiv preprint arXiv:2601.19290, 2026d.

Tongtong Wu, Yanming Li, Ziye Tang, Chen Jiang, Linhao Luo, Guilin Qi, Shirui Pan, and Gholamreza Haffari. Card: Towards conditional design of multi-agent topological structures. arXiv preprint arXiv:2603.01089, 2026.

Chengdong Xu, Kaiqiang Ke, Ziheng Liu, Jiaqi Wei, Zibo Shao, Weile Guo, and Chao Yu. Evomas: Learning execution-time workflows for multi-agent systems. arXiv preprint arXiv:2605.08769, 2026.

Sukwon Yun, Jie Peng, Pingzhi Li, Wendong Fan, Jie Chen, James Y Zou, Guohao Li, and Tianlong Chen. Graph-of-agents: A graph-based framework for multi-agent llm collaboration. In International Conference on Learning Representations, volume 2026, pp. 19745–19760, 2026.

Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, and Dawei Cheng. G-designer: Architecting multi-agent communication topologies via graph neural networks. arXiv preprint arXiv:2410.11782, 2024.

Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent architecture search via agentic supernet. arXiv preprint arXiv:2502.04180, 2025a.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, et al. Aflow: Automating agentic workflow generation. In International Conference on Learning Representations, volume 2025, pp. 34040–34077, 2025b.

Yaolun Zhang, Xiaogeng Liu, and Chaowei Xiao. Metaagent: Automatically constructing multiagent systems based on finite state machines. arXiv preprint arXiv:2507.22606, 2025c.

Yiqun Zhang, Hao Li, Jianhao Chen, Hangfan Zhang, Peng Ye, Lei Bai, and Shuyue Hu. Beyond gpt-5: Making llms cheaper and better via performance-efficiency optimized routing. In Proceedings of the 2025 7th International Conference on Distributed Artificial Intelligence, pp. 122–129, 2025d.

Xutong Zhao and Yaqi Xie. Multi-level advantage credit assignment for cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2508.06836, 2025.

Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulic, Anna Korhonen,´ and Sercan Arik. Multi-agent design: Optimizing agents with better prompts and topologies. In International Conference on Learning Representations, volume 2026, pp. 15844–15872, 2026.

## A MATCHER DETAILS

## A.1 ONLINE MATCHER ADAPTATION

We implement deployment-time adaptation using linear contextual Thompson sampling with aggregate execution feedback. Online adaptation changes only concrete worker selection; the plannerfacing matchability features remain probe-derived.

Let $\boldsymbol { z } ( \cdot ) \in \mathbb { R } ^ { d }$ denote the frozen embedding produced by the text encoder for either a subtask or a probe. Using the probe-density statistic ${ \bar { q } } _ { j }$ defined in Section 3.3, we define a density-correction weight

$$
\rho _ { j } = ( \bar { q } _ { j } + \epsilon ) ^ { - \nu } , \qquad Z = \sum _ { j = 1 } ^ { N } \rho _ { j } .
$$

For each worker w, its probe-derived capability representation is

$$
g ( w ) = \frac { 1 } { Z } \sum _ { j = 1 } ^ { N } \rho _ { j } o _ { w , j } z ( p _ { j } ) ,
$$

where $p _ { j }$ denotes probe $j .$ The density correction reduces the influence of semantically redundant probe regions, while the worker-independent normalization $Z$ preserves differences in overall probe performance across workers.

For a subtask–worker pair, we define the dimension-aligned interaction

$$
\phi ( s , w ) = z ( s ) \odot g ( w ) .
$$

Rather than maintaining a separate online parameter for every worker, we use a single shared parameter $\eta \in \mathbb { R } ^ { d }$ . Given a posterior sample $\tilde { \eta } ,$ the deployment-adapted suitability is

$$
\tilde { u } _ { w } ( s ) = \hat { u } _ { w } ( s ) + \tilde { \eta } ^ { \top } \phi ( s , w ) .
$$

The adapted score replaces $\hat { u } _ { w } ( s )$ in the operation-specific worker-selection rules, while probederived complementarity statistics remain unchanged.

For execution $t ,$ let $B _ { t , i }$ <sub>s</sub> denote the workers assigned to subtask s and

$$
A _ { t } = \sum _ { s } | B _ { t , s } | .
$$

We aggregate the task–worker interactions and static suitability estimates of the selected assignment as

$$
x _ { t } = \frac { 1 } { A _ { t } } \sum _ { s } \sum _ { w \in B _ { t , s } } \phi ( s , w ) , \qquad \bar { u } _ { t } = \frac { 1 } { A _ { t } } \sum _ { s } \sum _ { w \in B _ { t , s } } \hat { u } _ { w } ( s ) .
$$

We model the terminal utility as

$$
r _ { t } = \bar { u } _ { t } + x _ { t } ^ { \top } \eta + \epsilon _ { t } , \qquad \epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) .
$$

We place a zero-mean Gaussian prior

$$
\eta \sim \mathcal { N } ( 0 , \Lambda _ { 0 } ^ { - 1 } ) .
$$

Treating ${ { \bar { u } } _ { t } }$ as a known probe-based offset, we define

$$
y _ { t } = r _ { t } - \bar { u } _ { t }
$$

and perform the standard Bayesian linear-regression update

$$
\Lambda _ { t } = \Lambda _ { t - 1 } + \sigma ^ { - 2 } x _ { t } x _ { t } ^ { \top } , \qquad b _ { t } = b _ { t - 1 } + \sigma ^ { - 2 } x _ { t } y _ { t } ,
$$

with posterior

$$
p ( \eta \mid \mathcal { H } _ { t } ) = \mathcal { N } \left( \Lambda _ { t } ^ { - 1 } b _ { t } , \Lambda _ { t } ^ { - 1 } \right) .
$$

Before each execution, Thompson sampling draws $\widetilde { \eta } \sim p ( \eta \mid \mathcal { H } _ { t } )$ for worker selection.

When a new worker joins, its probe signature immediately provides both $\hat { u } _ { w } ( s )$ and $g ( w )$ . No worker-specific online parameter is introduced, and the shared posterior over η remains unchanged.

## B EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

## B.1 DATASET STATISTICS.

Table 6 summarizes the datasets and splits used throughout our experiments. Note that the probe set contains 55,676 examples. Profiling a new worker therefore requires roughly 56K probe inferences, which is typically inexpensive in practice and is incurred only when the worker is introduced. Since worker updates are far less frequent than deployment queries, the profiling cost is amortized over the worker’s subsequent lifetime and adds little recurring overhead.

## B.2 ONLINE ADAPTATION SPLIT

For the online adaptation experiment, we randomly hold out examples from each evaluation benchmark to construct a fixed evaluation set. Because GAIA contains only 165 examples, we reserve the full benchmark for evaluation. The remaining examples form the adaptation stream used to update the contextual-bandit posterior.

Table 6: Data sources used for probe construction, planner post-training, and evaluation.
<table><tr><td>Usage</td><td>Dataset</td><td>Primary task type</td><td># Examples</td><td>Sampling</td></tr><tr><td rowspan="7">Probe set</td><td>APPS</td><td>Code generation</td><td>4,962</td><td>Full</td></tr><tr><td>BFCL</td><td>Tool / function use</td><td>3,151</td><td>Full</td></tr><tr><td>Big-Math</td><td>Mathematical reasoning</td><td>6,000</td><td>Sampled</td></tr><tr><td>FRAMES</td><td>Knowledge-intensive reasoning</td><td>824</td><td>Full</td></tr><tr><td>MBPP+</td><td>Code generation</td><td>378</td><td>Full</td></tr><tr><td>SuperGPQA</td><td>General knowledge / reasoning</td><td>26,529</td><td>Full</td></tr><tr><td>MMLU-Pro</td><td>General knowledge / reasoning</td><td>12,032</td><td>Full</td></tr><tr><td rowspan="7">RL training</td><td>ACPBench</td><td>Planning / agentic reasoning</td><td>1,800</td><td>Full</td></tr><tr><td>DAPO-Math-17K MuSiQue</td><td>Mathematical reasoning</td><td>1,000</td><td>Sampled</td></tr><tr><td>BigCodeBench</td><td>Multi-hop reasoning</td><td>675 415</td><td>Sampled</td></tr><tr><td></td><td>Code generation</td><td>400</td><td>Sampled</td></tr><tr><td>Logistics CodeContests</td><td>Planning</td><td>335</td><td>Sampled</td></tr><tr><td>TACO</td><td>Competitive programming</td><td></td><td>Sampled</td></tr><tr><td>BrowseComp-Plus</td><td>Code generation</td><td>335 400</td><td>Sampled</td></tr><tr><td rowspan="8">Evaluation</td><td>TravelPlanner</td><td>Retrieval / web reasoning Planning</td><td>225</td><td>Sampled Sampled</td></tr><tr><td>MuSiQue</td><td></td><td></td><td></td></tr><tr><td>TravelPlanner</td><td>Multi-hop reasoning</td><td>2,000</td><td>Sampled</td></tr><tr><td>Omni-MATH-2</td><td>Planning</td><td>1,000</td><td>Full</td></tr><tr><td>LiveCodeBench v6</td><td>Mathematical reasoning</td><td>4,428 611</td><td>Full</td></tr><tr><td>BBEH</td><td>Competitive programming</td><td></td><td>Full</td></tr><tr><td>NaturalPlan</td><td>Broad reasoning</td><td>4,520</td><td>Full</td></tr><tr><td>PlanBench-mystery</td><td>Constraint-based scheduling</td><td>3,600</td><td>Full</td></tr><tr><td>GAIA</td><td></td><td>Obfuscated symbolic planning General assistant / tool use</td><td>603 165</td><td>Full Full</td></tr></table>

Table 7: Dataset split for the online adaptation experiment. The held-out examples are used only for evaluation and never for posterior updates.
<table><tr><td>Benchmark</td><td>Total</td><td>Adaptation</td><td>Held-out</td></tr><tr><td>MuSiQue</td><td>2,000</td><td>1,750</td><td>250</td></tr><tr><td>TravelPlanner</td><td>1,000</td><td>800</td><td>200</td></tr><tr><td>Omni-MATH-2</td><td>4,428</td><td>4,128</td><td>300</td></tr><tr><td>LiveCodeBench v6</td><td>611</td><td>411</td><td>200</td></tr><tr><td>BBEH</td><td>4,520</td><td>4,220</td><td>300</td></tr><tr><td>NaturalPlan</td><td>3,600</td><td>3,300</td><td>300</td></tr><tr><td>PlanBench-mystery</td><td>603</td><td>391</td><td>212</td></tr><tr><td>GAIA</td><td>165</td><td>0</td><td>165</td></tr><tr><td>Total</td><td>16,927</td><td>15,000</td><td>1,927</td></tr></table>

## B.3 IMPLEMENTATION DETAILS

Planner training. We optimize the planner with a GRPO-style policy-gradient objective, applying $A ^ { \mathrm { d e c } }$ to the Stage-I generation and $A ^ { \mathrm { o p } }$ to the corresponding Stage-II generation, with the two losses weighted equally. We perform full-parameter RL post-training of the planner with AdamW using a learning rate of $1 \times \mathrm { { \bar { 1 0 } } ^ { - 6 } }$ under a cosine schedule with warmup ratio 0.03, a global batch size of 16 queries, and 400 optimization steps. For each training query, we sample $J \ = \ 8$ Stage-I decompositions and $k = 8$ Stage-II collaboration plans per decomposition, both at temperature 1.0. Group-relative advantages are mean-centered within each group and are not divided by the group standard deviation. We allow at most 8 subtasks per decomposition and cap the number of workers assigned to any single subtask at $K = 6 .$ . The Stage-II cost coefficient is set to $\beta = 0 . 0 1$

For planner post-training, terminal quality $Q ( x , y )$ is computed by the task-specific verifier of each training source. For every source except TravelPlanner, the verifier is binary, with $Q \in \{ 0 , 1 \}$ and $Q = 1$ iff the generated solution passes. TravelPlanner instead uses a denser gated reward derived from the benchmark’s own checkers: an invalid or undelivered plan receives zero reward, while a valid plan is scored as the equally weighted sum of five components—the fractions of commonsense and hard constraints satisfied, the corresponding two indicators of whether all such constraints are satisfied, and the benchmark’s final all-or-nothing success indicator. We use this denser reward only for training because TravelPlanner’s binary pass rate on our worker pool is near zero, causing rollout groups with uniformly zero rewards to provide no learning signal. Evaluation follows the standard task-specific grader of each benchmark, and all reported TravelPlanner results use the binary pass rate.

Static matcher. For probe-based matching, we use $h _ { \mathrm { r e l } } = 0 . 0 5 , h _ { \mathrm { d e n } } = 0 . 0 2 , \nu = 1 . 0$ , and $\epsilon = 0 . 0 1$ . For MMR-based worker selection, we set the trade-off coefficient to $\lambda _ { o } = 0 . 5$ for all multi-worker operations.

Online bandit. For online adaptation, we use the contextual Thompson-sampling formulation in Appendix A.1. We use the full 4096-dimensional Qwen3-Embedding-8B representation for both subtask embeddings $z ( s )$ and probe embeddings $z ( p _ { j } )$ . The shared online parameter η uses an isotropic Gaussian prior with precision 1.0, and the observation-noise variance is $\sigma ^ { 2 } = 0 . 0 1$ . Unless otherwise specified, all reported results use the static probe-based matcher without online updates; online adaptation is enabled only in the dedicated analysis in Table 5.

Unseen worker pool. For the worker-pool generalization experiment in Section 4.3, we replace the original six-worker pool with an entirely unseen pool of four models: gemma-4-26B-A4B-it, Qwen3-14B, granite-4.2-8b, and Ministral-3-8B-Instruct-2512. None of these workers is used during planner training. For DEORCH, each new worker is incorporated only by evaluating it on the fixed probe set; no planner or matcher retraining is performed. For CONDUCTOR, the four new workers are assigned the available worker indices and the orchestrator is informed that the current pool contains four workers, without further training. BEST SINGLE AGENT uses gemma-4-26B-A4B-it, the strongest worker in the new pool. INDEPENDENT executes all four workers independently and uses the strongest worker to aggregate their responses into the final answer. Because the current matcher forms teams from distinct workers, the default maximum team size $K = 6$ is clipped to the current pool size when fewer workers are available; the four-worker unseen pool therefore allows at most four workers per subtask. We omit MAS-Orchestra from the unseen-pool evaluation because it assumes a single underlying worker rather than selecting among a heterogeneous worker pool. Replacing this worker with one model from the new pool would test model substitution rather than adaptation to the pool, while introducing multi-worker selection would alter the original method.

Baseline adaptation. For CONDUCTOR (Nielsen et al., 2026), we replace its original worker pool with our six-worker pool and retrain the orchestrator following its original procedure. Although CONDUCTOR improves robustness to varying worker availability by training on randomized worker subsets, these subsets are drawn from the same fixed set of workers seen during training; its reported dynamic-pool evaluation therefore does not test transfer to entirely unseen workers. For MAS-ORCHESTRA (Ke et al., 2026), which uses a single underlying worker model, we replace that worker with gemma-4-12B-it, the strongest worker in our pool, while otherwise retaining its original orchestration mechanism. To approximately match the training rollout budget of DEORCH $( 4 0 0 \times$ $1 6 \times 8 \times 8 \approx 4 1 0 \mathrm { K }$ executed orchestrations), we train CONDUCTOR for 1,600 optimization steps and MAS-ORCHESTRA for 200 steps, accounting for their different numbers of rollouts per step.

All experiments are conducted on a single node with 4× NVIDIA H200 GPUs. Code is available at: https://anonymous.4open.science/r/DeOrch-code-F8D3.

## B.4 OFFLINE MATCHER EVALUATION PROTOCOL

For the in-distribution evaluation of the static matcher, examples from each probe source are randomly split 80/20, with 80% used to construct the probe set and the remaining 20% held out for evaluation. For leave-one-source-out (LOSO) evaluation, we consider five representative sources— ACPBench, MMLU-Pro, MBPP+, Big-Math, and BFCL—and exclude the corresponding source entirely from the probe set before evaluating on its held-out examples.

We compare against random routing, the strongest individual worker in the pool (gemma-4-12B-it), majority voting over all six workers, and probe matching without density correction. We additionally report ORACLE SINGLE, which selects a successful worker for each example in hindsight and serves only as an upper bound.

## B.5 COMPLEMENTARITY-AWARE TEAM SELECTION.

The previous experiment evaluates single-worker selection, while multi-worker operations additionally require constructing worker teams. To test whether probe-derived complementarity improves team selection, we construct a TOP-n SUITABILITY variant of DEORCH. Using the same planner outputs and team sizes as the full method, this variant selects the top-n workers solely according to their suitability scores $\hat { u } _ { w } ( s )$ , without using complementarity information. Table 8 shows that complementarity-aware selection consistently improves over selecting workers solely by individual suitability. Since both variants execute the same plans with the same team sizes, the improvement cannot be attributed to additional worker calls, but instead reflects the benefit of accounting fo worker complementarity when constructing multi-worker teams.

Table 8: Inference-time ablation of multi-worker selection.
<table><tr><td>Selection Rule</td><td>ID Avg. ↑</td><td>OOD Avg. ↑</td><td>Overall Avg. ↑</td></tr><tr><td>TOP-n SUITABILITY</td><td>45.9</td><td>52.3</td><td>50.7</td></tr><tr><td>Complementarity-Aware selection</td><td>47.4</td><td>52.9</td><td>51.5</td></tr></table>

## B.6 INFERENCE COST ANALYSIS

We further evaluate inference cost by counting total prompt and completion tokens across all autoregressive model calls, including planning, worker execution, and collaboration. Token usage is averaged within each benchmark and macro-averaged across the eight benchmarks. We additionally include BEST SINGLE + REFINEMENT to control for increased inference compute: starting from gemma-4-12B-it, it performs three sequential refinement rounds without ground-truth feedback, yielding four worker calls and a token budget comparable to DEORCH.

Table 9 shows that refinement improves BEST SINGLE AGENT from 39.2 to 42.4, but remains well below DEORCH’s 51.5 at a similar token budget (14.1K vs. 14.6K). DEORCH also outperforms INDEPENDENT, CONDUCTOR, and MAS-ORCHESTRA while using fewer tokens, indicating that its gains are not explained by increased inference compute alone.

## B.7 COMPARISON WITH COUNTERFACTUAL CREDIT ASSIGNMENT

LEMON addresses orchestration credit assignment through counterfactual edits to its generated orchestration. We do not include it as a direct baseline in Table 1 because LEMON assigns each role to one of three predefined capacity levels (small, medium, or large), whereas our setting requires selecting freely among six heterogeneous workers. Adapting LEMON to this setting would require redefining both its worker-selection representation and the corresponding counterfactual edits, making the comparison no longer a faithful reproduction of the original method.

Table 9: Inference cost across the eight evaluation benchmarks.
<table><tr><td>Method</td><td>Avg. ↑</td><td>Calls↓</td><td>Tokens / Query ↓</td></tr><tr><td>BEST SINGLE AGENT</td><td>39.2</td><td>1.00</td><td>1859</td></tr><tr><td>BEST SINGLE + REFINEMENT</td><td>42.4</td><td>4.00</td><td>14106</td></tr><tr><td>INDEPENDENT</td><td>46.0</td><td>7.00</td><td>25859</td></tr><tr><td>FRONTIER-PLANNER</td><td>36.6</td><td>4.67</td><td>11479</td></tr><tr><td>UNTRAINED PLANNER</td><td>38.3</td><td>4.92</td><td>12867</td></tr><tr><td>CONDUCTOR</td><td>44.0</td><td>6.16</td><td>22458</td></tr><tr><td>MAS-ORCHESTRA</td><td>49.1</td><td>5.63</td><td>17250</td></tr><tr><td>DEORCH</td><td>51.5</td><td>5.35</td><td>14577</td></tr></table>

We instead construct a controlled LEMON-inspired counterfactual-credit baseline in our orchestration space. This variant replaces our two-stage planner and conditional-credit objective with singlepass orchestration trained using LEMON-style global and localized counterfactual objectives, while keeping the worker pool, matcher, execution protocol, and training budget unchanged. Specifically, following LEMON’s counterfactual training procedure, we train the single-pass planner with global GRPO augmented by one localized counterfactual edit per rollout; we adapt its mutations to our orchestration space by editing either a decomposition dependency or a collaboration operation, reexecute the modified plan, and apply the resulting reward contrast only to the edited token span. This directly compares two approaches to credit assignment across decomposition and operation selection: counterfactual credit assignment versus our two-stage conditional formulation. As shown in Table 10, DEORCH improves the overall average from 49.2 to 51.5 with comparable worker calls, indicating that our two-stage conditional formulation provides more effective credit assignment in this setting.

Table 10: Comparison of credit-assignment formulations.
<table><tr><td>Variant</td><td>ID Avg.</td><td>OOD Avg. Avg.</td><td></td><td>Calls↓</td></tr><tr><td>Counterfactual Credit</td><td>43.1</td><td>51.2</td><td>49.2</td><td>5.17</td></tr><tr><td>Two-Stage Conditional Credit (DEORCH)</td><td>47.4</td><td>52.9</td><td>51.5</td><td>5.35</td></tr></table>

## B.8 DISCUSSION AND FUTURE WORK

Computational overhead. Compared with single-pass planning, our two-stage planner introduces one additional planner generation at inference time. This overhead is modest relative to the subsequent execution of multiple worker agents; indeed, our cost analysis shows that DEORCH still uses fewer inference tokens than competing learned MAS orchestrators. Training incurs a larger overhead because the nested conditional rollout procedure samples multiple Stage-II plans for each sampled decomposition. However, this additional training cost is incurred only during post-training: once trained, the same planner can be reused when the worker pool changes without further retraining. Improving the efficiency of this nested training procedure is an interesting direction for future work.

Extending the collaboration space. Our current multi-worker operations construct teams from distinct workers in the pool and therefore do not support homogeneous collaboration through repeated execution of the same worker. In practice, an ENSEMBLE-3 operation could either invoke three distinct workers or run three independent instances of the same strong worker, and which choice is preferable may depend on the task. A natural solution is to introduce both homogeneous and heterogeneous variants into the operation vocabulary, such as homogeneous-ENSEMBLE-3 and heterogeneous-ENSEMBLE-3, allowing the planner to learn which form of collaboration is more appropriate for each subtask. We also considered richer operations that are not explored in this work due to computational constraints. A recursive operation could further decompose a difficult subtask during execution, while an escalation operation could invoke an external worker outside the current pool when the available workers are insufficient. Incorporating such operations would extend the framework toward more adaptive and open-ended orchestration.

Scaling to larger worker pools. Our experiments consider heterogeneous pools of up to six workers. Although the planner interface is independent of worker identities and pool size, the behavior of the matcher at substantially larger scales remains to be evaluated. Evaluating and improving DE ORCH under substantially larger and more dynamic worker pools is therefore an important direction for future work.