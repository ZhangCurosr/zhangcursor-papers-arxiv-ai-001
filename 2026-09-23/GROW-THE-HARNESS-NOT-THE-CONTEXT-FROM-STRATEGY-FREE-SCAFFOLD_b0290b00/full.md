# GROW THE HARNESS, NOT THE CONTEXT: FROM STRATEGY-FREE SCAFFOLDS TO REUSABLE SPECIALIST AGENTS

A PREPRINT

Laizhen Li<sup>1,2</sup>, Jiarui Li<sup>1</sup>, Juanjuan Zhao<sup>1</sup>, Kejiang Ye<sup>1,3</sup>, Ye Li<sup>1</sup>, Cheng-zhong Xu<sup>4</sup>, Xitong Gao<sup>1,3∗</sup>

<sup>1</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences <sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>Shenzhen University of Advanced Technology <sup>4</sup>University of Macau

## ABSTRACT

Large language model (LLM) agents often handle streams of related tasks, yet standard harnesses repeatedly ask the model to reconstruct the same control decisions inside each task’s context. We study whether task feedback can instead turn recurring control into reusable executable code, while reserving LLM calls for task-specific semantic reasoning. We introduce Growing Harness, a failure-guided training paradigm that learns the agent harness itself from a strategy-free scaffold that exposes fixed model and tool interfaces but encodes no task-solving controller. Function-level execution traces localize each failure to a bounded code surface, an optimizer repairs a window of failures jointly, and a success-first held-out gate rolls back repair sequences that harm prior capability. Accepted edits accumulate in one shared harness, allowing its control structure to emerge from task feedback. Across BrowseComp-Plus and WebArena-Verified with three deployment models from 4B to 120B parameters, Growing Harness achieves the highest mean success in five of six benchmark–model settings and trails the best mean by 0.7 pp. in the sixth. Relative to a Tool-Calling agent, it reduces LLM calls by 76.0–91.8% and deployed-agent inference cost by 74.4–98.6%. On WebArena-Verified, its success remains 44.7–45.3% across model scales, whereas Tool-Calling falls to 6.7% with the 4B model. Ablations show that trace-local edits, joint repair, and gate-based rollback each improve final success. These results show that persistent program growth can move recurring control out of model context and into low-cost code, yielding reusable specialist agents that remain effective with smaller deployment models.

## 1 Introduction

Large language model (LLM) agents solve complex tasks by interleaving model reasoning with external tools. In many deployments, however, an agent does not face isolated tasks. It repeatedly handles instances from the same task family, using the same model and tool interfaces. Although goals and observations change across instances, the surrounding control often recurs: the agent must refine queries, filter observations, verify progress, recover from errors, and decide when to stop. Agent harnesses, the executable code that orchestrates model and tool calls, either specify such behavior in advance or delegate it to LLMs Wang et al. (2024); Yao et al. (2023). Repeated delegation preserves flexibility, but every decision incurs another inference call, grows context, and adds prompts, observations, and outputs to a growing execution history.

Repeated deployment within a task family therefore creates an opportunity for harness growth. Here, harness growth means using task feedback to turn recurring control into persistent executable behavior. Across tasks drawn from a common distribution, the harness can accumulate code that later executions reuse instead of reconstructing the same behavior through online inference. Prior agents reuse experience through retrieved workflows, executable skills, or synthesized tools Wang et al. (2025, 2023); Cai et al. (2023); Yuan et al. (2023). These approaches show that behavior learned from earlier tasks can benefit later executions. Harness growth uses the agent program itself as the persistent artifact: behavior acquired from earlier failures runs directly at low marginal cost, while the LLM remains available for task-dependent semantic reasoning. This leads to our central question: how can a scaffold with no predefined controller grow from task feedback so that it stops asking the LLM to reconstruct behavior that it can execute in code? Our thesis is that such growth can manage LLM context structurally: the harness acquires recurring control as persistent executable code, reserving model context for task-specific evidence and semantic reasoning. This design is especially useful for small LLMs deployed on mobile and other resource-constrained devices: moving recurring control from inference into code offers a path to capable agents when model scale and online inference are constrained.

![](images/dbba1c06ed4c0f59ea1a753dd637501884e31f37922bf2a767b667f5f7176848.jpg)

![](images/63f85c68c811d349058f61af482095c85018b3311cb376f7b3c121d47cf269a1.jpg)  
Figure 1: Success–cost trade-offs on BrowseComp-Plus and WebArena-Verified. Colors denote agents, and marker shapes denote deployment models. Each point reports the mean over three independent evaluation runs. Confidence intervals are omitted for visual clarity and reported in Table 1.

A harness grown from task feedback should meet three requirements. (a) reuse: behavior acquired from training tasks should help unseen tasks from the same distribution, rather than encode instance-specific solutions. (b) contextefficient execution: the harness should reduce repeated model calls and avoid placing routine control into a growing model context, while preserving LLM reasoning where semantics matter. Long histories increase input cost and can make relevant evidence harder to use Yao et al. (2023); Liu et al. (2024). (c) low prior commitment: harness growth should not require a complete controller at initialization. Recent methods optimize existing harnesses or synthesize code within prescribed interfaces and structures Lou et al. (2026); Lee et al. (2026). We call this starting point a strategy-free scaffold: it exposes the required task, model, and tool interfaces but encodes no task-solving controller, such as a ReAct-style tool-calling loop. This allows the executable structure itself to adapt to task feedback.

Three technical challenges emerge. (a) What should become code, and what should remain an LLM call? Executable code is cheap to reuse, but brittle rules cannot replace semantic judgment. The learner must identify deterministic, structured behavior that transfers across tasks, while reserving model calls for interpretation, synthesis, and other open-ended decisions. (b) How can an optimizer improve a growing program without processing the whole program at every step? As the harness accumulates capabilities, whole-program optimization makes the code context and search space grow with it. A task-level outcome identifies whether an execution succeeded, but not which function or control decision caused a failure. Harness growth therefore requires execution evidence that links each failure to a bounded local code surface, so per-step optimization depends on the active execution slice rather than the total program size. (c) How can growth avoid brittleness and regression? An edit that repairs one task may encode an instance-specific rule or damage behavior acquired earlier. This risk is acute during continual harness optimization, where successive local gains may fail to compound Wang, Kattakinda, and Feizi (2026). The learner must favor behavior shared across failures and test each update independently before adding it to the deployed harness.

![](images/48cd40c18cac241389aa3d18c184262bc231d480d6836707821e0b546cac9989.jpg)  
Figure 2: Overview of the Growing Harness. From a strategy-free scaffold h , each round traces a failed execution, locates the faulty functions, diagnoses the underlying control-policy failure, and repair function-level code. Accept candidate iff it fixes enough failures and preserves held-out success.

We introduce Growing Harness, a failure-guided training paradigm that naturally grows a cost-efficient agent harness from a strategy-free executable scaffold. The scaffold exposes the task entry point and fixed LLM and tool interfaces, but encodes no task-solving controller. Training runs the current harness on a task stream and maintains a bounded window of failures. For each failed execution, a function-level execution DAG records the code paths, model calls, tool calls, and errors involved. An offline optimizer uses these traces to synthesize a complete executable candidate, while trace scope and an edit budget limit changes to implicated functions and a small number of new helpers. We direct the optimizer to implement deterministic and reusable operations in code, while retaining LLM calls for taskdependent semantic reasoning. Candidates are tested on the active failures, and a success-first held-out gate rejects repair sequences that reduce aggregate gate-set success. Accepted edits update the shared harness, so executable behavior learned from one group of failures can serve later tasks. Over successive rounds, the scaffold grows into a code-first, LLM-assisted agent without committing to a predefined controller.

Figure 1 previews the resulting success–cost trade-off. Across BrowseComp-Plus and WebArena-Verified, the learned harnesses occupy the high-success, low-cost region, with the largest separation from inference-intensive agents on WebArena-Verified and with the 4B deployment model, highlighting the potential of harness growth for small LLMs in mobile and other constrained deployments.

Our contributions are:

• We formulate agent learning as reusable program growth from task feedback, starting from a strategy-free scaffold under fixed LLM and tool interfaces.

• We introduce trace-local program growth, which combines multi-failure training with function-level tracescoped edits so the global harness can grow while each optimizer step remains focused on a bounded execution slice. Held-out gate rollback guards against brittle or regressive updates.

• Across BrowseComp-Plus and WebArena-Verified and three deployment-model scales, Growing Harness achieves the highest mean success rate in five of six settings while reducing LLM calls by 76.0–91.8% and online cost by 74.4–98.6% relative to Tool-Calling. Ablations show complementary gains from function-level guidance, the failure-window curriculum, and gate-based rollback.

## 2 Related Work

Inference-time control in tool-using agents. LLM agents rely on control structures around the model to organize reasoning, tool use, observation handling, and stopping. ReAct interleaves reasoning and acting, Self-Ask decomposes questions into follow-up queries, and IRCoT alternates retrieval with multi-step reasoning Yao et al. (2023); Press et al. (2023); Trivedi et al. (2023). Reflexion further carries feedback from earlier attempts into subsequent executions Shinn et al. (2023). These methods establish that the control surrounding an LLM is a substantive part of agent behavior. Their loop structures are nonetheless specified before deployment, while many control decisions are represented in textual histories and recomputed by the model as each trajectory unfolds. Harness growth asks a different question: how can task feedback change the executable control itself, with the LLM and tools fixed, so later tasks reuse code rather than repeat inference?

Persistent experience across tasks. Experiential agents such as ExpeL and Agent Workflow Memory store naturallanguage insights, examples, or workflows for later retrieval Zhao et al. (2023); Wang et al. (2025). Voyager, LATM, and CRAFT instead distill experience into executable skills or tools that can be reused across tasks Wang et al. (2023); Cai et al. (2023); Yuan et al. (2023). Both lines of work show that experience can be amortized across related tasks. Textual artifacts, however, must be retrieved and interpreted within the model context, while skills package individual behaviors behind an existing agent structure. Our persistent artifact is the shared harness itself: accepted repairs become ordinary program paths that orchestrate model and tool use for all later tasks. The result is a growing controller, rather than a collection of experiences that the agent must retrieve and reinterpret online.

Optimizing language-model programs. A broad line of work improves LLM systems without updating model weights. APE, OPRO, and related methods optimize natural-language instructions Zhou et al. (2023); Yang et al. (2023); Pryzant et al. (2023), while DSPy, MIPRO, TextGrad, and GEPA optimize prompts and demonstrations across multi-stage language-model programs Khattab et al. (2023); Opsahl-Ong et al. (2024); Yuksekgonul et al. (2024); Agrawal et al. (2025). AFlow and ADAS search over code-represented workflows or broader agentic system designs using task feedback Zhang et al. (2024); Hu, Lu, and Clune (2024). These methods establish prompts, modules, and workflow structure as learnable objects. Our focus is continual program growth from a strategy-free scaffold: failed executions identify missing behavior, trace-local edits add that behavior to one shared harness, and accepted updates accumulate across a task stream.

Harness optimization. Recent work directly optimizes the code and configuration surrounding an LLM application. AutoHarness synthesizes code constraints or complete policies from environment feedback, and Meta-Harness searches over model-harness code using source code, candidate scores, and execution traces Lou et al. (2026); Lee et al. (2026). VeRO provides versioning, structured observations, and evaluation infrastructure for agents that optimize other agents Ursekar et al. (2026). Other methods learn from retrospective or retrieved experience, or adapt harnesses so smaller models can approach the performance of stronger models Pan et al. (2026); Huang et al. (2026); Yang et al. (2026). Recent evidence also shows that successive harness updates can lose earlier gains without explicit regression control Wang, Kattakinda, and Feizi (2026). Our distinction is not code synthesis alone, but open-ended global harness growth through local, failure-conditioned optimization, with training beginning from a strategy-free scaffold under fixed LLM and tool interfaces.

Cost-efficient LLM and agent execution. LLMLingua compresses prompts, whereas FrugalGPT and RouteLLM use cascades or learned routing to invoke expensive models selectively Jiang et al. (2023); Chen, Zaharia, and Zou (2023); Ong et al. (2025). AgentOccam and WebDreamer reduce planning or interaction overhead through more efficient agent architectures Yang et al. (2025); Gu et al. (2025), and agent evaluations increasingly report monetary cost alongside task success Kapoor et al. (2025). These approaches make calls shorter or cheaper, select a cheaper call, or improve a fixed inference-time strategy. Our approach instead manages context structurally: it grows executable code so recurring control need not be regenerated or carried through the model context on every task. The deployed LLM and tools remain fixed, and model context is reserved for task-specific evidence and semantic reasoning.

## 3 Method

## 3.1 Problem Formulation

An agent workflow can be represented as the composition of an executable harness, a language model backend, and a set of external tools. Let M denote a fixed LLM backend, let T denote a fixed collection of tool interfaces, and let h ∈ H denote executable harness code. Given a task x, the deployed agent harness is h(x; M, T), which specifies how the agent invokes M and $T ,$ maintains intermediate state, processes observations, recovers from errors, and decides when to terminate. In our paradigm, M and T remain fixed; the harness h is the only learnable object.

```latex
Algorithm 1: Failure-guided harness growth
Input: Training set $D _ { \mathrm { t r } } ,$ , gate set $G ,$ fixed model ${ \overline { { M } } } ,$ tools T, optimizer O, initial harness $h _ { 0 } ,$ window size K, repair threshold
Q, attempt limit $R _ { \operatorname* { m a x } } ,$ and edit budget L
Output: Learned harness $h ^ { \star }$
1 t ← 0; $W _ { t } \gets \emptyset$
2 while unseen tasks remain or $W _ { t } \neq \emptyset$ do
3 Run $h _ { t }$ on unseen tasks and retain failures and traces until $| W _ { t } | = K$ or the training stream ends
4 if $W _ { t } = \emptyset$ then
5 break
6 Using $h _ { t }$ and the traces in $W _ { t } , O$ produces a valid trace-scoped candidate $\widetilde { h } _ { t }$ satisfying $d _ { \mathrm { f u n } } ( h _ { t } , \widetilde { h } _ { t } ) \leq L$
7 Re-execute $\widetilde { h } _ { t }$ on every task in $W _ { t }$
8 $P _ { t } \gets \{ x _ { i } \in W _ { t } : o ( \widetilde { h } _ { t } , x _ { i } ) = 1 \}$
9 $U _ { t } \gets \dot { W } _ { t } \setminus P _ { t }$
10 if $| P _ { t } | < Q$ then
11 Discard $\widetilde { h } _ { t }$ and retain $h _ { t }$ $/ /$ rollback: insufficient repairs
12 Increment attempt counts in $W _ { t }$ and retire tasks reaching $R _ { \mathrm { m a x } }$
13 continue
14 if $\mathrm { S R } _ { G } ( \widetilde { h } _ { t } ) < \mathrm { S R } _ { G } ( h _ { t } )$ then
15 Discard $\widetilde { h } _ { t }$ and retain $h _ { t }$ $/ /$ rollback: gate regression
16 Increment attempt counts in $W _ { t }$ and retire tasks reaching $R _ { \mathrm { m a x } }$
17 continue
18 Accept the candidate: $h _ { t + 1 }  \widetilde { h } _ { t }$
19 Set $W _ { t + 1 }$ to the unresolved tasks $U _ { t } ,$ with fresh traces and incremented attempt counts; retire tasks reaching $R _ { \mathrm { m a x } }$
20 $t \gets t + 1$
21 $\dot { h ^ { \star } }  h _ { t } ;$ return $h ^ { \star }$
```

For a task set $D ,$ we denote the success rate of a harness h as $\mathrm { S R } _ { D } ( h )$ . We partition the available tasks into a training set $D _ { \mathrm { t r } }$ , a gate validation set $G ,$ , and a final evaluation set $D _ { \mathrm { t e } } .$ . The training set supplies execution experience, the gate set controls checkpoint acceptance, and the final set is used only for reporting. The optimizer never observes outcomes or traces from $D _ { \mathrm { t e } }$

Training starts from a minimal executable scaffold $h _ { 0 }$ . The scaffold exposes the required task entry point and fixed LLM and tool interfaces, but contains no complete hand-designed agent strategy. Because it retains generic access to M and T, an LLM-mediated agent remains representable within the harness hypothesis class. Training can therefore grow executable control from a weak initialization without committing to a predefined controller such as native toolcalling loop or Self-Ask.

## 3.2 Harness Execution and Function-Level Traces

For each task $x _ { i }$ , the runtime records an execution graph $\mathcal { G } _ { i } = ( \mathcal { V } _ { i } , \mathcal { E } _ { i } )$ . Nodes represent harness-function invocations, LLM and tool calls, returns, or runtime errors, and record the operation, inputs, outputs or errors, parent, and duration. An edge (u, v) indicates that v executes within the dynamic scope of u. The task trace is

$$
\tau _ { i } = ( x _ { i } , \mathcal { G } _ { i } , y _ { i } , o _ { i } , u _ { i } ) ,\tag{1}
$$

where $y _ { i }$ is the output, $o _ { i } = o ( h , x _ { i } ) \in \{ 0 , 1 \}$ is the benchmark outcome, and $u _ { i }$ contains the token, cost, and runtime measurements. To localize a failure to participating code, we extract

$$
\mathcal { F } ( \tau _ { i } ) = \{ f : f \mathrm { ~ i s ~ a ~ h a r n e s s ~ f u n c t i o n ~ i n v o k e d ~ i n ~ } \mathcal { G } _ { i } \} .\tag{2}
$$

This set later constrains which existing functions the optimizer may modify.

## 3.3 Failure-Window Curriculum

Rather than repeatedly optimizing tasks already solved, we maintain a bounded window $W _ { t }$ of current failures at round t. For the ordered training sequence $D _ { \operatorname { t r } } = ( x _ { 1 } , \dots , x _ { N } )$ , each entry stores a task, its latest trace, and its repair-attempt count $w _ { i } = ( x _ { i } , \tau _ { i } , a _ { i } )$ . The current harness fills the window with failures from unseen tasks, up to capacity K; direct successes leave the curriculum. The optimizer jointly repairs all window failures to encourage reusable behavior. After each repair, solved tasks $P _ { t }$ leave. Unresolved tasks $U _ { t } = W _ { t } \setminus P _ { t }$ retain fresh traces and incremented attempt counts; tasks whose updated count reaches $R _ { \mathrm { m a x } }$ are retired. New failures $F _ { t }$ then refill the window. Ignoring the stored trace metadata, its task membership evolves as

$$
W _ { t + 1 } = \{ x _ { i } \in U _ { t } : a _ { i } < R _ { \operatorname* { m a x } } \} \cup F _ { t } .\tag{3}
$$

Here, $a _ { i }$ is the updated attempt count. Before optimizing a full window, we checkpoint the complete state; if every task in that window exhausts its budget without a repair, we restore the checkpoint. This prevents an ineffective repair sequence from persisting when no single edit triggers an immediate gate regression.

## 3.4 Trace-Guided Function-Level Optimization

Given $h _ { t }$ and $W _ { t } ,$ an offline optimizer receives

$$
Z _ { t } = \left( h _ { t } , \{ \tau _ { i } : x _ { i } \in W _ { t } \} , E \right) ,\tag{4}
$$

where $E$ contains offline diagnostic artifacts such as evaluator feedback, retrieved-document identifiers, evidence, action statistics, and errors. These artifacts are unavailable to the deployed harness. An optimizer model $M _ { \mathrm { o p t } }$ , which may differ from M, produces a complete executable candidate

$$
\widetilde { h } _ { t + 1 } = O \left( Z _ { t } ; M _ { \mathrm { o p t } } \right) .\tag{5}
$$

To connect failures to edits and bound the search space, the optimizer may modify only the entry function and functions invoked by a supplied failure trace:

$$
\begin{array} { r } { \mathcal { A } _ { t } = \{ \operatorname* { m a i n } \} \cup \bigcup _ { x _ { i } \in W _ { t } } \mathcal { F } ( \tau _ { i } ) . } \end{array}\tag{6}
$$

It may also add reusable helpers, but must leave untraced existing functions unchanged. If $d _ { \mathrm { f u n } }$ counts modified and newly introduced functions, we require $d _ { \mathrm { f u n } } ( h _ { t } , \widetilde { h } _ { t + 1 } ) \leq L$ for edit budget L. The optimizer must preserve function signatures and runtime interfaces, may not delete existing functions, and may not encode task identifiers, expected answers, or fixed solutions.

We direct code toward deterministic, reusable control, $e . g .$ , parsing, validation, state updates, conditional query refinement, error recovery, and stopping, while retaining LLM calls for semantic interpretation, synthesis, fuzzy comparison, and answer generation. Before benchmark evaluation, each candidate must parse and compile, expose the required entry point, preserve the fixed model and tool interfaces, and satisfy the trace-scope and edit-budget constraints. Invalid candidates are rejected.

## 3.5 Candidate Evaluation and Window Update

A valid candidate is re-executed on the entire active window. The solved and unresolved subsets are

$$
P _ { t } = \left\{ x _ { i } \in W _ { t } : o ( \widetilde { h } _ { t + 1 } , x _ { i } ) = 1 \right\} , \quad U _ { t } = W _ { t } \setminus P _ { t } .\tag{7}
$$

The candidate becomes the provisional harness, $h _ { t + 1 }  \widetilde { h } _ { t + 1 } ; P _ { t }$ leaves the window, whereas $U _ { t }$ receives incremented attempt counts and fresh traces. Because each repair is a complete harness used on subsequent tasks, helpers and control logic accumulate across rounds rather than remaining task-specific patches. If synthesis or validation fails, optimization is retried without changing $h _ { t } ;$ exhausting the optimizer’s internal retry limit restores the latest accepted checkpoint and stops training. The harness, window, task cursor, counters, and checkpoint metadata are serialized for resumption.

## 3.6 Held-Out Gate Validation and Rollback

Repairs may solve the active window while damaging earlier behavior. We therefore evaluate $h _ { 0 }$ on held-out gate set G, then reevaluate the current harness after every $Q$ tasks solved through repair. Relative to the latest accepted checkpoint $h ^ { \mathrm { g a t e } } .$ , we accept $h _ { t } \ i f f \ S R _ { G } ( h _ { t } ) \geq \ S \mathrm { R } _ { G } ( h ^ { \mathrm { g a t e } } )$ . Once accepted, the complete current state becomes the new checkpoint $h ^ { \mathrm { g a t e } }  \bar { h } _ { t }$ . Otherwise, the entire repair sequence since the previous gate is rolled back. Rollback restores not only code, but also the task cursor, failure window, counters, and training records, yielding the same state from which the rejected sequence began.

After the training stream and window are exhausted, any changes made since the last checkpoint undergo one final gate evaluation. The current harness is selected only if its gate success is no lower than that of $h ^ { \mathrm { g a t e } }$ ; otherwise, the latest accepted checkpoint is restored. The resulting $h ^ { \star }$ is evaluated once on $D _ { \mathrm { t e } } .$ . The rule is success-first: lower online cost cannot compensate for lower gate success, although cost is recorded for all rollouts to compare success-preserving harnesses.

In summary, failures localize supervision for bounded program synthesis, repairs accumulate reusable control in one executable harness, and transactional gate rollback protects previously acquired capability while the fixed LLM continues to provide task-dependent semantic reasoning.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate Growing Harness on BrowseComp-Plus (Chen et al., 2025), a controlled benchmark for deepsearch agents under a fixed, human-verified retrieval corpus, and WebArena-Verified (hattami et al., 2025), an audited benchmark for reproducible evaluation of multi-step web agents with corrected tasks and deterministic evaluators. The two benchmarks cover complementary settings: open-domain retrieval and evidence synthesis in BrowseComp-Plus, and multi-step interaction in WebArena-Verified. For each benchmark, we use 200 training tasks, 50 held-out gate tasks, and 50 final-evaluation tasks.

Models. We evaluate the trained harnesses on three deployment LLMs: gpt-oss-120b and gpt-oss-20b (OpenAI et al., 2025), and Qwen3.5-4B (Qwen Team, 2026).

Baselines. We compare the trained harnesses with representative inference-time agent strategies adapted to each benchmark. Tool-Calling is a fixed, non-learning agent that repeatedly selects an environment tool based on the task context and interaction history. Its interleaved tool-calling loop follows a reasoning-and-acting execution pattern (Yao et al., 2023). For BrowseComp-Plus, we adapt IRCoT, which interleaves chain-of-thought reasoning with retrieval (Trivedi et al., 2023); Self-Ask, which decomposes a question into explicit follow-up questions that can be answered through search (Press et al., 2023). For WebArena-Verified, we adapt WebDreamer, which uses an LLM as a world model to predict and evaluate the outcomes of candidate browser actions (Gu et al., 2025), and AgentOccam, which aligns the browser observation and action spaces with the capabilities of the underlying LLM (Yang et al., 2025). Within each benchmark, all methods share deployment LLMs, task splits, tools, and evaluation protocol; complete baseline configurations and our hyperparameters are provided in the appendix.

Evaluation Metrics. For each method–model configuration, we conduct $R = 3$ independent evaluation runs. In each run, every final-evaluation task receives one single-attempt agent rollout.

Our primary task metric is Success Rate, defined as the percentage of successful task–run pairs:

$$
\begin{array} { r } { \widehat { \mathrm { S R } } = \frac { 1 } { R | \mathcal { D } | } \sum _ { r = 1 } ^ { R } \sum _ { x _ { i } \in \mathcal { D } } \mathbb { I } [ \mathrm { E v a l } ( x _ { i } , \hat { a } _ { i r } ) = 1 ] , } \end{array}\tag{8}
$$

where $\mathcal { D }$ is the final-evaluation set and $\hat { a } _ { i r }$ is the answer produced for task $x _ { i }$ in run $^ { r } \cdot$ We average results over all three runs and do not select the best of multiple attempts. BrowseComp-Plus uses an LLM-based equivalence judge, whereas WebArena-Verified uses the benchmark evaluator.

For efficiency, we report LLM Calls, Input Tokens, Output Tokens, Time, and Total Online Cost. Calls, token counts, and time are averaged over all task-run pairs. Token counts are reported in thousands per task, and Input Tokens include both uncached and cache-read tokens. Total Online Cost includes deployed-agent LLM calls only, excluding evaluator and offline-optimizer calls; it is computed separately for each run and then averaged across the three runs. For a run, we compute

$$
\begin{array} { r } { \mathrm { C o s t } = \sum _ { t = 1 } ^ { N _ { \mathrm { r e q } } } ( u _ { t } \pi _ { \mathrm { i n } } + c _ { t } \pi _ { \mathrm { c a c h e } } + v _ { t } \pi _ { \mathrm { o u t } } ) / 1 0 ^ { 6 } , } \end{array}\tag{9}
$$

where $u _ { t } , c _ { t }$ , and $v _ { t }$ are the uncached-input, cache-read, and output tokens for request $t ,$ respectively. The providerlisted prices used for cost accounting are reported in the appendix. We estimate cache-read tokens using the longest prefix shared with an earlier request from the same task and do not assume cache reuse across tasks.

To quantify uncertainty, we use a task-level cluster bootstrap with 10,000 replicates and a fixed random seed of 42. Each replicate samples $| \mathcal D |$ tasks with replacement while retaining all three runs for each sampled task. We report every metric as $\mathrm { \ m e a n { \pm } 1 . 9 6 \mathrm { S E _ { b o o t s t r a p } ; } }$ thus, each ± value is a normal-approximation 95% confidence-interval half-width, not a standard deviation.

## 4.2 Main Results

Learned control transfers to unseen tasks. Across two distinct task families, Growing Harness achieves the highest mean success rate in five of the six benchmark–model settings and falls only 0.7 pp. short of the highest mean in the remaining setting (Table 1). The breadth of this result suggests that the learned programs capture behavior shared within each family, rather than repairs specific to the training failures that produced them.

<table><tr><td>Model</td><td>Agent</td><td></td><td></td><td></td><td>Success Rate (%) ↑ LLM Calls ↓ Input Tokens (k) ↓ Output Tokens (k) ↓</td><td>Time (s) ↓</td><td>Cost ($) ↓</td></tr><tr><td rowspan="10">Bro-pus</td><td>Tool-Calling Self-Ask</td><td> $4 0 . 0 \pm 1 1 . 6$   $2 2 . 0 \pm 1 0 . 6$ </td><td> $3 2 . 7 \pm 3 . 4$ </td><td> $1 5 9 9 . 1 \pm 2 3 8 . 0$   $1 6 8 . 0 \pm 2 4 . 4$ </td><td> $1 2 . 5 \pm 1 . 9$   $1 5 . 2 \pm 2 . 0$ </td><td> $2 4 1 . 1 \pm 3 9 . 4$ </td><td> $6 . 7 0 \pm 0 . 9 6$ </td></tr><tr><td> $\mathrm { g p t - o s s - } 1 2 0 \mathrm { b }$ </td><td></td><td> $3 7 . 5 \pm 4 . 8$ </td><td></td><td></td><td> $5 2 2 . 3 \pm 7 6 . 8$ </td><td> $1 . 3 2 \pm 0 . 1 8$ </td></tr><tr><td>IRCoT</td><td> $2 6 . 0 \pm 1 1 . 0$ </td><td> $3 6 . 4 \pm 5 . 3$ </td><td> $3 1 2 . 1 \pm 4 8 . 2$ </td><td> $1 0 . 0 \pm 1 . 7$ </td><td> $5 2 1 . 2 \pm 8 9 . 2$ </td><td> $2 . 4 4 \pm 0 . 3 7$ </td></tr><tr><td>Ours</td><td> ${ \bf 4 9 . 3 \pm 1 1 . 7 }$ </td><td> ${ \bf 6 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 8 6 . 9 \pm 2 . 1 }$ </td><td> ${ \bf 7 . 5 \pm 0 . 7 }$ </td><td> $\mathbf { 1 7 6 . 9 \pm 1 1 . 2 }$ </td><td> ${ \bf 0 . 8 7 \pm 0 . 0 3 }$ </td></tr><tr><td></td><td>Tool-Calling</td><td> ${ \bf 4 0 . 0 \pm 1 1 . 7 }$ </td><td> $2 4 . 8 \pm 3 . 3$ </td><td> $9 5 2 . 8 \pm 1 9 6 . 0$ </td><td> ${ \bf 8 . 3 \pm 1 . 5 }$ </td><td> $4 0 7 . 0 \pm 7 1 . 3$ </td><td> $2 . 0 3 \pm 0 . 4 0$ </td></tr><tr><td>gpt-oss-20b</td><td>Self-Ask</td><td> $2 2 . 7 \pm 9 . 1$ </td><td> $3 4 . 5 \pm 4 . 2$ </td><td> $1 3 1 . 1 \pm 1 9 . 0$ </td><td> $1 9 . 8 \pm 2 . 5$ </td><td> $3 9 9 . 6 \pm 5 2 . 0$ </td><td> $0 . 6 5 \pm 0 . 0 8$ </td></tr><tr><td>IRCoT Ours</td><td></td><td> $2 7 . 3 \pm 1 1 . 2$ </td><td> $3 2 . 2 \pm 5 . 2$ </td><td> $2 7 5 . 3 \pm 4 7 . 4$ </td><td> $2 1 . 3 \pm 4 . 9$ </td><td> $4 1 5 . 4 \pm 8 9 . 6$ </td><td> $1 . 2 8 \pm 0 . 2 1$ </td></tr><tr><td></td><td></td><td> $3 9 . 3 \pm 1 2 . 6$ </td><td> ${ \bf 6 . 0 \pm 0 . 1 }$ </td><td> ${ \bf 8 2 . 9 \pm 2 . 5 }$ </td><td> $1 4 . 3 \pm 1 . 7$ </td><td> $\mathbf { 3 5 1 . 8 \pm 7 3 . 9 }$ </td><td> ${ \bf 0 . 5 2 \pm 0 . 0 3 }$ </td></tr><tr><td></td><td>Tool-Calling</td><td> $1 2 . 0 \pm 7 . 1$ </td><td> $2 9 . 6 \pm 3 . 3$ </td><td> $1 5 6 9 . 5 \pm 2 6 3 . 1$ </td><td> $3 . 3 \pm 0 . 5$ </td><td> $1 6 0 8 . 0 \pm 8 4 . 9$ </td><td> $\overline { { 8 . 3 3 \pm 1 . 3 6 } }$ </td></tr><tr><td rowspan="5">qwen3.5-4b</td><td>Self-Ask</td><td> $2 . 0 \pm 2 . 2$ </td><td> $6 . 7 \pm 1 . 9$ </td><td> ${ \bf 2 1 . 1 \pm 8 . 8 }$ </td><td> ${ \bf 1 . 1 \pm 0 . 3 }$ </td><td> ${ \bf 2 6 2 . 9 \pm 8 5 . 8 }$ </td><td>0.17 ± 0.06</td></tr><tr><td>IRCoT Ours</td><td> $7 . 3 \pm 7 . 0$ </td><td> $4 6 . 4 \pm 3 . 1$ </td><td> $3 6 9 . 0 \pm 3 3 . 0$ </td><td> $3 . 6 \pm 0 . 5$ </td><td> $5 2 9 . 4 \pm 8 4 . 2$ </td><td> $3 . 3 1 \pm 0 . 2 9$ </td></tr><tr><td></td><td> ${ \bf 2 9 . 3 \pm 1 1 . 8 }$ </td><td> ${ \bf 5 . 5 \pm 0 . 2 }$ </td><td> $8 1 . 5 \pm 4 . 9$ </td><td> $2 . 6 \pm 0 . 3$ </td><td> $3 9 0 . 2 \pm 7 3 . 7$ </td><td> $0 . 8 1 \pm 0 . 0 5$ </td></tr><tr><td>Tool-Calling</td><td> $3 0 . 0 \pm 1 1 . 4$ </td><td> $2 6 . 3 \pm 4 . 1$ </td><td> $6 2 4 . 1 \pm 1 5 8 . 4$ </td><td> $6 . 9 \pm 1 . 5$ </td><td> $6 6 7 . 5 \pm 1 2 2 . 7$ </td><td> $2 . 7 2 \pm 0 . 6 6$ </td></tr><tr><td>WebDreamer AgentOccam</td><td> $1 0 . 0 \pm 7 . 0$ </td><td> $1 0 . 7 \pm 1 . 3$ </td><td> $1 9 . 1 \pm 2 . 8$ </td><td> $6 . 3 \pm 0 . 9$ </td><td> $\mathbf { 1 9 9 . 0 \pm 3 6 . 4 }$ </td><td> $0 . 3 2 \pm 0 . 0 5$ </td></tr><tr><td rowspan="8">Web-ed</td><td>gpt-oss-120b</td><td> $6 . 7 \pm 6 . 0$ </td><td> $9 . 7 \pm 1 . 2$ </td><td> $2 0 . 8 \pm 3 . 2$ </td><td> $6 . 2 \pm 0 . 8$ </td><td> $2 4 7 . 7 \pm 7 5 . 1$ </td><td> $0 . 3 3 \pm 0 . 0 5$ </td></tr><tr><td>Ours</td><td> ${ \bf 4 5 . 3 \pm 1 3 . 4 }$ </td><td> ${ \bf 3 . 8 \pm 2 . 6 }$ </td><td> ${ \bf 8 . 0 \pm 6 . 0 }$ </td><td> ${ \bf 1 . 6 \pm 1 . 2 }$ </td><td> $2 9 9 . 9 \pm 1 1 9 . 1$ </td><td> ${ \bf 0 . 1 0 \pm 0 . 0 7 }$ </td></tr><tr><td>Tool-Calling</td><td> $1 2 . 7 \pm 7 . 3$ </td><td> $2 2 . 3 \pm 2 . 7$ </td><td> $\overline { { 1 5 8 . 5 \pm 2 5 . 1 } }$ </td><td> $3 . 1 \pm 0 . 4$ </td><td> $\overline { { 4 6 1 . 4 \pm 6 2 . 3 } }$ </td><td> $0 . 3 9 \pm 0 . 0 5$ </td></tr><tr><td>WebDreamer gpt-oss-20b</td><td> $4 . 0 \pm 3 . 9$ </td><td> $7 . 7 \pm 1 . 1$ </td><td> $1 3 . 3 \pm 2 . 1$ </td><td> $6 . 1 \pm 1 . 0$ </td><td> $1 7 4 . 0 \pm 3 0 . 7$ </td><td> $0 . 1 4 \pm 0 . 0 2$ </td></tr><tr><td>AgentOccam</td><td> $2 . 0 \pm 2 . 8$ </td><td> $6 . 2 \pm 0 . 9$ </td><td> $1 2 . 5 \pm 2 . 5$ </td><td> $4 . 9 \pm 0 . 8$ </td><td> ${ \bf 1 1 6 . 6 \pm 2 7 . 5 }$ </td><td> $0 . 1 2 \pm 0 . 0 2$ </td></tr><tr><td>Ours</td><td> ${ \bf 4 4 . 7 \pm 1 3 . 4 }$ </td><td> ${ \bf 1 . 8 \pm 1 . 0 }$ </td><td> ${ \bf 3 . 5 \pm 2 . 3 }$ </td><td> ${ \bf 1 . 6 \pm 1 . 1 }$ </td><td> $2 5 0 . 1 \pm 1 0 0 . 5$ </td><td> ${ \bf 0 . 0 3 \pm 0 . 0 2 }$ </td></tr><tr><td>Tool-Calling</td><td> $6 . 7 \pm 4 . 9$ </td><td> $3 7 . 8 \pm 2 . 7$ </td><td> $1 3 5 5 . 6 \pm 1 6 1 . 7$ </td><td> $4 . 4 \pm 0 . 5$ </td><td> $1 3 1 2 . 6 \pm 1 0 0 . 7$ </td><td> $7 . 2 6 \pm 0 . 8 5$ </td></tr><tr><td rowspan="4">qwen3.5-4b</td><td>WebDreamer</td><td> $0 . 0 \pm 0 . 0$ </td><td> ${ \bf 4 . 9 \pm 0 . 7 }$ </td><td> ${ \bf 1 0 . 0 \pm 1 . 5 }$ </td><td> $1 . 2 \pm 0 . 2$ </td><td> ${ \bf 1 7 4 . 9 \pm 3 0 . 1 }$ </td><td> $0 . 1 1 \pm 0 . 0 2$ </td></tr><tr><td>AgentOccam</td><td> $3 . 3 \pm 4 . 6$ </td><td> $7 . 0 \pm 0 . 8$ </td><td> $1 7 . 5 \pm 3 . 4$ </td><td> $3 . 2 \pm 0 . 5$ </td><td> $4 9 0 . 3 \pm 7 8 . 1 $ </td><td> $0 . 2 0 \pm 0 . 0 4$ </td></tr><tr><td>Ours</td><td> ${ \bf 4 5 . 3 \pm 1 3 . 4 }$ </td><td> $5 . 4 \pm 3 . 8$ </td><td> $1 3 . 7 \pm 1 0 . 1$ </td><td> ${ \bf 0 . 4 \pm 0 . 5 }$ </td><td> $3 8 7 . 5 \pm 1 4 8 . 6$ </td><td> ${ \bf 0 . 1 0 \pm 0 . 0 8 }$ </td></tr></table>

Table 1: Main results on BrowseComp-Plus and WebArena-Verified. Entries report means over three independent evaluation runs; ± values denote normal-approximation 95% confidence-interval half-widths estimated by task-level cluster bootstrap. Calls, token counts, and time are per-task averages, with token counts reported in thousands. Cost denotes the mean total deployed-agent LLM cost of one 50-task evaluation run in U.S. dollars. Best mean values within each benchmark–model block are shown in bold.

Persistent code replaces repeated online inference. Relative to Tool-Calling, Growing Harness reduces LLM calls by 76.0–91.8% and online cost by 74.4–98.6% across the six settings. Despite using fewer calls, it improves mean success over Tool-Calling in five settings and remains within 0.7 pp. in the sixth. The resulting success–cost frontier in Figure 1 supports the central premise of harness growth: recurring control can run as reusable code at low marginal cost, without replacing the LLM’s task-specific semantic reasoning.

The learned harness sustains success across model scales. On WebArena-Verified, its mean success remains within a narrow 44.7–45.3% range across all three deployment models, whereas Tool-Calling drops from 30.0% with gptoss-120b to 6.7% with Qwen3.5-4B. The largest gain therefore appears with the smallest deployment model. This pattern suggests that reusable code supplies recurring browser control that smaller models would otherwise need to reconstruct during each execution.

## 4.3 Harness Growth Convergence Analysis

Shared failures yield reusable control. The BrowseComp-Plus harness grows a single retrieval and verification pipeline that every task reuses for query generation, evidence collection and compression, and answer checking. Repairs to this common path therefore transfer across gate tasks instead of adding instance-specific branches, which explains the rapid early gain in Figure 3.

Local control lets capabilities accumulate. WebArena-Verified improves across the training run and rolls back only a few candidates. Its learned program adds specialized handlers for diverse Shopping, Reddit, and Map tasks around a general LLM-guided browser loop. These branches add new behavior without rewriting paths used by other task types, which is consistent with the steadier training curve.

The task family shapes the controller. Both harnesses start from the same strategy-free seed program, yet one becomes a shared pipeline and the other a set of specialized handlers. This contrast supports the low-prior-commitment goal: harness growth lets executable structure emerge from task feedback instead of fixing a controller in advance. Because the benchmarks use different training configurations, we treat the contrast as a description of these runs, not as a comparison of intrinsic difficulty.

Best accepted Accepted gate Rolled back

![](images/9623138af17cddd2e8cdac85581bc355afdd44bb82264d2a4867e2fda0930221.jpg)

![](images/b8a0eef336ffc20206580edb5b64c5ef6d9f1aaea3758a902340887f3490ce41.jpg)  
Figure 3: Validation convergence (training steps vs. task success rate in %) during harness training on BrowseComp-Plus and WebArena-Verified. Each point corresponds to a gate evaluation at the indicated optimization step.

<table><tr><td>Variant</td><td>Success (%) ↑</td></tr><tr><td>w/o Function-Level Guidance</td><td>18.0</td></tr><tr><td>w/o Gate Validation</td><td>22.0</td></tr><tr><td>w/o Failure-Window</td><td>28.0</td></tr><tr><td>Full Method</td><td>36.0</td></tr></table>

Table 2: Single-run ablation results on the 50-task BrowseComp-Plus final-evaluation set. Each configuration is optimized for 10 steps.

## 4.4 Ablation Studies

We isolate the three principal mechanisms in Algorithm 1: function-level guidance, the failure-window curriculum, and held-out gate validation with transactional rollback. All ablations are conducted on BrowseComp-Plus using gptoss-20b as the deployment model and GPT-5.6-terra (High) as the optimizer. Each configuration is optimized once for 10 optimization steps and evaluated on the same 50-task final-evaluation set. W/O FUNCTION-LEVEL GUIDANCE removes function-level traces, trace-derived edit localization, and the edit budget L, allowing unconstrained wholeprogram edits. W/O GATE VALIDATION disables gate-based acceptance and transactional rollback. W/O FAILURE-WINDOW sets the window capacity to K = 1, so each repair is conditioned on a single active failure. All remaining settings are held fixed.

Each removal lowers final-evaluation success, but the optimization trajectories reveal why the mechanisms complement one another (Table 2; Figure 4).

Trace locality makes program search effective. Removing function-level guidance halves final success from 36% to 18%. The whole-program variant also makes no gate progress for five steps before reaching a lower plateau. This stall supports the role of function-level traces in linking a task failure to a bounded edit surface.

Gate rollback lets gains accumulate. Without gate validation, gate success rises to 30% but then falls to 16%. The full method instead retains its best gate result once it reaches it. The contrast directly shows the failure mode that rollback targets: a repair can solve current failures while damaging behavior acquired earlier.

A window of failures favors reusable repairs. Conditioning each edit on one failure does not cause the same stall or regression, but it lowers final success by 8 pp. This gap suggests that joint repair helps the optimizer identify behavior shared across failures.

![](images/47cc8289f0961b0ac4626ebed39d3f8429c52a0918d18849bde0cbc54c508044.jpg)  
Figure 4: Validation success rates over 10 optimization steps on BrowseComp-Plus. Each variant removes one mechanism from the full method. Each trajectory is obtained from a single optimization run.

## 5 Conclusion

Agents that repeatedly serve one task family need not reconstruct all control through LLM inference. Growing Harness instead acquires recurring control as persistent code, while keeping the LLM for task-specific semantic decisions. Starting from a strategy-free scaffold, function-level traces focus edits on code implicated by current failures, a bounded failure window encourages repairs shared across tasks, and a success-first held-out gate rolls back changes that reduce prior capability. The method does not prescribe a controller; it lets executable structure adapt to task feedback.

Harness growth thus provides a structural way to manage model context: encode repeatable control in the harness and reserve model context for evidence and decisions that vary by task. Its value depends on reusing the learned harness enough to offset offline optimization. Real deployment also requires sandboxing, explicit permission boundaries, and validation of generated code.

## References

Agrawal, L. A.; Tan, S.; Soylu, D.; Ziems, N.; Khare, R.; Opsahl-Ong, K.; Singhvi, A.; Shandilya, H.; Ryan, M. J.; Jiang, M.; Potts, C.; Sen, K.; Dimakis, A. G.; Stoica, I.; Klein, D.; Zaharia, M.; and Khattab, O. 2025. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457.

Cai, T.; Wang, X.; Ma, T.; Chen, X.; and Zhou, D. 2023. Large Language Models as Tool Makers. arXiv:2305.17126.

Chen, L.; Zaharia, M.; and Zou, J. 2023. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176.

Chen, Z.; Ma, X.; Zhuang, S.; Nie, P.; Zou, K.; Liu, A.; Green, J.; Patel, K.; Meng, R.; Su, M.; Sharifymoghaddam, S.; Li, Y.; Hong, H.; Shi, X.; Liu, X.; Thakur, N.; Zhang, C.; Gao, L.; Chen, W.; and Lin, J. 2025. BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent. arXiv preprint arXiv:2508.06600.

Gu, Y.; Zhang, K.; Ning, Y.; Zheng, B.; Gou, B.; Xue, T.; Chang, C.; Srivastava, S.; Xie, Y.; Qi, P.; Sun, H.; and Su, Y. 2025. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents. Transactions on Machine Learning Research.

hattami, A. E.; Thakkar, M.; Chapados, N.; and Pal, C. 2025. WebArena Verified: Reliable Evaluation for Web Agents. In Workshop on Scaling Environments for Agents.

Hu, S.; Lu, C.; and Clune, J. 2024. Automated Design of Agentic Systems. arXiv:2408.08435.

Huang, Y.; Wang, W.; Bao, H.; Ma, Y.; Luo, X.; Nian, Y.; Zhuang, H.; Liu, Z.; Zhao, Y.; and Zhang, X. 2026. Memo-Harness: Agent Harnesses That Learn from Experience. arXiv:2607.14159.

Jiang, H.; Wu, Q.; Lin, C.-Y.; Yang, Y.; and Qiu, L. 2023. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. arXiv:2310.05736.

Kapoor, S.; Stroebl, B.; Kirgis, P.; Nadgir, N.; Siegel, Z. S.; Wei, B.; Xue, T.; Chen, Z.; Chen, F.; Utpala, S.; Ndzomga, F.; Oruganty, D.; Luskin, S.; Liu, K.; Yu, B.; et al. 2025. Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation. arXiv:2510.11977.

Khattab, O.; Singhvi, A.; Maheshwari, P.; Zhang, Z.; Santhanam, K.; Vardhamanan, S.; Haq, S.; Sharma, A.; Joshi, T. T.; Moazam, H.; Miller, H.; Zaharia, M.; and Potts, C. 2023. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. arXiv:2310.03714.

Lee, Y.; Nair, R.; Zhang, Q.; Lee, K.; Khattab, O.; and Finn, C. 2026. Meta-Harness: End-to-End Optimization of Model Harnesses. arXiv:2603.28052.

Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions ofthe Associationfor Computational Linguistics, 12: 157–173.

Lou, X.; Lazaro-Gredilla, M.; Dedieu, A.; Wendelken, C.; Lehrach, W.; and Murphy, K. P. 2026. AutoHarness:´ Improving LLM Agents by Automatically Synthesizing a Code Harness. arXiv:2603.03329.

Ong, I.; Almahairi, A.; Wu, V.; Chiang, W.-L.; Wu, T.; Gonzalez, J. E.; Kadous, M. W.; and Stoica, I. 2025. RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665.

OpenAI; :; Agarwal, S.; Ahmad, L.; Ai, J.; Altman, S.; Applebaum, A.; Arbus, E.; Arora, R. K.; Bai, Y.; Baker, B.; Bao, H.; Barak, B.; Bennett, A.; Bertao, T.; Brett, N.; Brevdo, E.; Brockman, G.; Bubeck, S.; Chang, C.; Chen, K.; Chen, M.; Cheung, E.; Clark, A.; Cook, D.; Dukhan, M.; Dvorak, C.; Fives, K.; Fomenko, V.; Garipov, T.; Georgiev, K.; Glaese, M.; Gogineni, T.; Goucher, A.; Gross, L.; Guzman, K. G.; Hallman, J.; Hehir, J.; Heidecke, J.; Helyar, A.; Hu, H.; Huet, R.; Huh, J.; Jain, S.; Johnson, Z.; Koch, C.; Kofman, I.; Kundel, D.; Kwon, J.; Kyrylov, V.; Le, E. Y.; Leclerc, G.; Lennon, J. P.; Lessans, S.; Lezcano-Casado, M.; Li, Y.; Li, Z.; Lin, J.; Liss, J.; Lily; Liu; Liu, J.; Lu, K.; Lu, C.; Martinovic, Z.; McCallum, L.; McGrath, J.; McKinney, S.; McLaughlin, A.; Mei, S.; Mostovoy, S.; Mu, T.; Myles, G.; Neitz, A.; Nichol, A.; Pachocki, J.; Paino, A.; Palmie, D.; Pantuliano, A.; Parascandolo, G.; Park, J.; Pathak, L.; Paz, C.; Peran, L.; Pimenov, D.; Pokrass, M.; Proehl, E.; Qiu, H.; Raila, G.; Raso, F.; Ren, H.; Richardson, K.; Robinson, D.; Rotsted, B.; Salman, H.; Sanjeev, S.; Schwarzer, M.; Sculley, D.; Sikchi, H.; Simon, K.; Singhal, K.; Song, Y.; Stuckey, D.; Sun, Z.; Tillet, P.; Toizer, S.; Tsimpourlas, F.; Vyas, N.; Wallace, E.; Wang, X.; Wang, M.; Watkins, O.; Weil, K.; Wendling, A.; Whinnery, K.; Whitney, C.; Wong, H.; Yang, L.; Yang, Y.; Yasunaga, M.; Ying, K.; Zaremba, W.; Zhan, W.; Zhang, C.; Zhang, B.; Zhang, E.; and Zhao, S. 2025. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925.

Opsahl-Ong, K.; Ryan, M. J.; Purtell, J.; Broman, D.; Potts, C.; Zaharia, M.; and Khattab, O. 2024. Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs. arXiv:2406.11695.

Pan, W.; Liu, S.; Lin, C.-Y.; Zeng, J.; Tang, X.; Zhou, X.; Lu, Y.; and Jia, X. 2026. Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference. arXiv:2606.05922.

Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N.; and Lewis, M. 2023. Measuring and Narrowing the Compositionality Gap in Language Models. In Bouamor, H.; Pino, J.; and Bali, K., eds., Findings of the Association for Computational Linguistics: EMNLP 2023, 5687–5711. Singapore: Association for Computational Linguistics.

Pryzant, R.; Iter, D.; Li, J.; Lee, Y. T.; Zhu, C.; and Zeng, M. 2023. Automatic Prompt Optimization with “Gradient Descent” and Beam Search. arXiv:2305.03495.

Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.

Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems, volume 36, 8634–8652.

Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal, A. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds., Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 10014–10037. Toronto, Canada: Association for Computational Linguistics.

Ursekar, V.; Shanker, A.; Chatrath, V.; Xue, Y.; and Denton, S. M. 2026. VeRO: A Harness for Agents to Optimize Agents. arXiv:2602.22480.

Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2023. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:2305.16291.

Wang, L.; Ma, C.; Feng, X.; Zhang, Z.; Yang, H.; Zhang, J.; Chen, Z.; Tang, J.; Chen, X.; Lin, Y.; Zhao, W. X.; Wei, Z.; and Wen, J.-R. 2024. A Survey on Large Language Model Based Autonomous Agents. Frontiers of Computer Science, 18(6): 186345.

Wang, W.; Kattakinda, P.; and Feizi, S. 2026. Do Agent Optimizers Compound? A Continual-Learning Evaluation on Terminal-Bench 2.0. arXiv:2607.14004.

Wang, Z. Z.; Mao, J.; Fried, D.; and Neubig, G. 2025. Agent Workflow Memory. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, 63897–63911. PMLR.

Yang, C.; Wang, X.; Lu, Y.; Liu, H.; Le, Q. V.; Zhou, D.; and Chen, X. 2023. Large Language Models as Optimizers. arXiv:2309.03409.

Yang, C.; Zhao, X.; Wu, T.; and Kastner, C. 2026. Better Harnesses, Smaller Models: Building 90arXiv:2607.08938.¨

Yang, K.; Liu, Y.; Chaudhary, S.; Fakoor, R.; Chaudhari, P.; Karypis, G.; and Rangwala, H. 2025. AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents. In The Thirteenth International Conference on Learning Representations.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations (ICLR).

Yuan, L.; Chen, Y.; Wang, X.; Fung, Y. R.; Peng, H.; and Ji, H. 2023. CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets. arXiv:2309.17428.

Yuksekgonul, M.; Bianchi, F.; Boen, J.; Liu, S.; Huang, Z.; Guestrin, C.; and Zou, J. 2024. TextGrad: Automatic “Differentiation” via Text. arXiv:2406.07496.

Zhang, J.; Xiang, J.; Yu, Z.; Teng, F.; Chen, X.; Chen, J.; Zhuge, M.; Cheng, X.; Hong, S.; Wang, J.; Zheng, B.; Liu, B.; Luo, Y.; and Wu, C. 2024. AFlow: Automating Agentic Workflow Generation. arXiv:2410.10762.

Zhao, A.; Huang, D.; Xu, Q.; Lin, M.; Liu, Y.-J.; and Huang, G. 2023. ExpeL: LLM Agents Are Experiential Learners. arXiv:2308.10144.

Zhou, Y.; Muresanu, A. I.; Han, Z.; Paster, K.; Pitis, S.; Chan, H.; and Ba, J. 2023. Large Language Models Are Human-Level Prompt Engineers. In International Conference on Learning Representations.

## A Experimental Details

## A.1 Datasets

For each benchmark, we partition tasks into train, gate, and final splits with 200 train tasks, 50 gate tasks, and 50 final tasks. The train split provides optimization traces, the gate split is used for checkpoint selection, and the final split is held out for reporting results. For WebArena-Verified, we restrict the benchmark to tasks involving the shopping, reddit, and map websites, and use stratified sampling over intent templates, task types, and websites to improve coverage and balance across task families, operation types, and websites. All dataset splits are constructed using a fixed random seed of 42.

## A.2 Shared Evaluation Configuration

All final-evaluation runs use the shared configuration in Table 3. The agent is not explicitly informed of the remaining call budget; the executable harness enforces the limit.

<table><tr><td>Parameter</td><td>Configuration</td></tr><tr><td>Sampling temperature</td><td>1.0</td></tr><tr><td>Reasoning effort</td><td>Medium</td></tr><tr><td>Maximum online LLM calls</td><td>50/task</td></tr><tr><td>Maximum output</td><td>8192 tokens/call</td></tr><tr><td>Task timeout</td><td>1,800 s</td></tr></table>

Table 3: Shared configuration used for final evaluation of all methods.

## A.3 Baseline Configurations

All baselines use the same seed interface, tool APIs, deployment models, and LLM-call budget as Growing Harness.

## BrowseComp-Plus.

All methods use the same search tool. Tool-Calling alternates LLM reasoning with search calls until the model emits a final answer. Self-Ask decomposes the task into follow-up search questions and intermediate answers before producing the final answer. IRCoT interleaves retrieval with evidence-sentence generation and then uses a final QA call.

## WebArena-Verified.

All methods use text-only browser tools without screenshots. Tool-Calling emits one browser action or final JSON answer per step. WebDreamer proposes candidate browser actions and selects among them using world-model and value-model scoring. AgentOccam uses compressed observations and short interaction history, with branch, prune, and note operations. WebDreamer and AgentOccam are adapted to the seed interface, accessibility-tree observation space, unified LLM budget, and shared deployment models.

## A.4 Token Prices

We compute online cost from the provider token prices in Table 4 and exclude judge calls.
<table><tr><td>Model</td><td>Provider</td><td>Input</td><td>Cache Read</td><td>Output</td></tr><tr><td>gpt-oss-120b</td><td>Groq</td><td>$0.15</td><td>$0.075</td><td>$0.60</td></tr><tr><td>gpt-oss-20b</td><td>Groq</td><td>$0.075</td><td>$0.0375</td><td>$0.30</td></tr><tr><td>qwen3.5-4b</td><td>Fireworks</td><td>$0.20</td><td>$0.10</td><td>$0.20</td></tr></table>

Table 4: Token prices (USD per million tokens).

## A.5 Hyperparameters

We select hyperparameters using only the training and gate splits; Table 5 reports the fixed configurations used for harness training.

<table><tr><td>Parameter</td><td>BrowseComp-Plus</td><td>WebArena-Verified</td></tr><tr><td>Training-time deployment LLM</td><td>gpt-oss-20b</td><td>gpt-oss-120b</td></tr><tr><td>Optimizer</td><td>GPT-5.6-terra</td><td>GPT-5.4</td></tr><tr><td>Reasoning effort</td><td>High</td><td>High</td></tr><tr><td>Deployment-LLM sampling seed</td><td>42</td><td>42</td></tr><tr><td>Failure-window capacity K</td><td>8</td><td>4</td></tr><tr><td>Per-task repair budget  $R _ { \mathrm { m a x } }$ </td><td>5</td><td>5</td></tr><tr><td>Gate interval Q</td><td>1</td><td>8</td></tr><tr><td>Function-level edit budget L</td><td>10</td><td>10</td></tr><tr><td>Candidates per optimization step</td><td>1</td><td>1</td></tr><tr><td>Maximum online LLM calls</td><td>50/task</td><td>50/task</td></tr><tr><td>Task timeout</td><td>900 s</td><td>1,800 s</td></tr></table>

Table 5: Benchmark-specific hyperparameters used for harness training. All configurations are selected without access to the final-evaluation sets.

![](images/b85001d5e214df6ee524654cb9ef8740b982226f7ac36811ff16f2a3d6d5210a.jpg)

Figure 5: Task-level comparison between Growing Harness and Tool-Calling. Each panel compares per-task LLM calls against one efficiency statistic on BrowseComp-Plus and WebArena-Verified using GPT-OSS-20B. Cost and token measurements exclude judge usage.  
![](images/2413e5383e7e8302f667ab4af73b305dd3e2a13c4c22948bdc050efa8b622165.jpg)

![](images/d77d379cd32e6806c4a752ce45054ff8d45920663ed5db070dadef14b4505455.jpg)  
Figure 6: Post-hoc abstraction of the learned executable harnesses $h ^ { \star }$ . BrowseComp-Plus induces a shared retrieval and evidence-verification pipeline, while WebArena-Verified induces a layered web-control program with deterministic resolvers, learned control routines, fallback browser control, and completion validation.

## B Additional Results

Figure 5 provides task-level efficiency measurements corresponding to the aggregate results in the main paper. The figure reports per-task calls, cost, and token usage for Growing Harness and Tool-Calling under the same 50-call evaluation budget.

## B.1 Discovered Harness Structures

To make the learned programs interpretable, Figure 6 abstracts the final harnesses $h ^ { \star }$ into their main executable control paths. These diagrams are post-hoc summaries of the discovered code, not hand-designed architectures used by the optimizer. They illustrate how failure-guided harness growth adapts the executable harness to the benchmark: BrowseComp-Plus induces a shared retrieval and evidence-verification pipeline, whereas WebArena-Verified induces a controller that combines deterministic resolvers, learned control routines, a generic guided browser loop, and an explicit completion-state validator. In both cases, the harness implements reusable control around the same fixed deployment LLM and tool interfaces.