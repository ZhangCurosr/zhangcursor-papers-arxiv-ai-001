# Loud Failures, Quiet Failures: Fault Detection and Recovery in Tool-Using Language Model Agents

Obada Kraishan

College of Media and Communication

Texas Tech University

Lubbock, TX, USA

omareikr@ttu.edu

Abstract—Tool-using agents are usually scored on whether they finish a task while the tools work. Deployments are less forgiving: services time out, endpoints disappear, parameter names change, and results come back well formed but wrong. We ask what agents do when the tool layer misbehaves. Prior work has shown that language models over-trust tool outputs that fail silently; we ask how that over-trust plays out across the stages of failure handling in multi-turn agents. Wrapping the executable environments of an established function-calling benchmark in a fault-injection layer, we inject one of four typed faults at a controlled point in the trajectory and record whether the agent notices, changes plan, recovers the task, or repeats itself. Six models from three families, half of them reasoning variants, ran 1,920 trials over 24 multi-step tasks. Agents treat a failure as a problem in 91.3% of trials when the tool returns an explicit error, but in 58.8% of trials when the tool returns a plausible wrong value, against a 26.8% rate of reporting problems when nothing was wrong (h = 0.80 between the two fault kinds). Reasoning models are not better placed here: paired against instruct siblings from the same family, they notice less (−9.3 points, p < .001) and change plan more (+10.4 points, p < .001), and the two effects leave recovery unchanged (p = .512 in a task-clustered model). Because agents are stochastic, two fault-free runs of the same task end in the same state only 63.3% of the time; against that baseline, only a missing tool clearly lowers recovery (39.9%), while timeouts, schema drift, and corruption stay within run-torun variation. After a fault, agents return to the same tool three or more times in a row in up to 22.2% of trials, though strictly identical repeats are rare. Adding a line to the prompt asking the agent to check each result did not move detection. The pattern points at a specific weakness: agents respond to the error channel rather than to the content of what a tool returns, so failures that stay inside the expected format pass through.

Index Terms—tool-using agents, fault injection, failure detection, agent evaluation, cognitive flexibility, reasoning models

## I. INTRODUCTION

When a language model calls tools, its mistakes no longer stay on the page. An agent books, transfers, writes, and deletes, and the record of what it did is the tool-call log rather than the prose around it. Benchmarks for these systems have grown quickly, and they measure something reasonable: given a working environment, does the agent reach the goal [1]–[3]. That question leaves out the part of deployment that operators actually worry about. Services time out. Endpoints are retired. A field gets renamed in a minor release. A cache returns a stale value that is correctly typed and quietly wrong. Evaluations that add this stress show how much a clean run hides: success measured once on working tools falls when the same tasks are repeated [10] or when faults are injected into the tool layer [28], and models tend to trust tool outputs that are wrong but carry no error signal [26], [29].

Handling those events is not a single skill. An agent must first register that something is off, then decide on a different course, then still reach the goal, and it must do all of this without treating ordinary friction as catastrophe. Those components separate cleanly in the human literature on monitoring and control of automation, where the failure to notice a malfunctioning system is studied apart from the failure to respond to it [4], [5], and where repeating an action that has stopped working is treated as its own phenomenon rather than as generic incompetence [6]. Agent evaluations rarely make the same separations. A trial that ends without the goal is scored as a failure whether the agent never saw the problem or saw it and could not route around it, and those two cases call for different fixes.

This paper measures the components separately. We take the multi-turn split of the Berkeley Function-Calling Leaderboard, whose environments execute locally as stateful Python objects, and wrap tool execution in a layer that injects one typed fault at a controlled point in the trajectory. Four faults are used: a timeout, a tool that disappears from the registry, a renamed parameter, and a returned value that is corrupted while keeping its shape. A no-fault condition provides the baseline. For every trial we record whether the agent treated the result as a problem, whether its next action differed from the failed one, whether the environment ended in the state the same model reaches on an independent clean run, and whether it called the same tool three or more times in a row.

The contrast that organises the results is not the name of the fault but its visibility. Three of the four faults surface an explicit error; the agent has only to read it. Corruption does not: the call succeeds, the payload is well formed, and only the content is wrong. That silent errors are harder to catch than explicit ones has been reported before, for single tool outputs judged on request [26] and for recovery in tasks with alternative tool paths [29]. What those studies leave open is how the stages of failure handling relate to one another within the same tasks and models: whether an agent that misses a failure also fails to recover, whether an agent that acts more also notices more, and how often agents report problems when

nothing is wrong.

The study is organised around three research questions, with a fourth, secondary analysis:

• RQ1. Does an agent’s detection of a tool failure depend on whether the failure surfaces an explicit error?

• RQ2. Do reasoning models handle tool failures better than matched instruct siblings from the same family?

• RQ3. What does a single tool failure cost in recovery, repeated calls, and effort?

The secondary analysis asks whether an instruction to check each tool result changes detection.

Our contributions are:

• a fault-injection layer for an existing executable agent benchmark, with typed faults that fire once at a controlled step and leave the environment consistent, released with the frozen task suite and the analysis code;

• a measurement scheme that separates noticing, replanning, recovery, and repetition, with recovery scored against the state the same model reaches without a fault, and judged against the agreement of two fault-free runs, so that neither baseline competence nor run-to-run variation is mistaken for a fault effect;

• evidence from 1,920 trials in multi-turn executable environments that extends earlier single-step findings on silent errors: detection follows the error channel against a nofault baseline, and reasoning variants notice less than their instruct siblings while acting more, with no net gain in recovery.

We release the harness, the frozen suite, the full trajectories, the scored dataset, and the analysis scripts.<sup>1</sup>

## II. RELATED WORK

Four lines of work meet in this study: how tool-using agents are built and scored, how faults in the tool layer have been studied, what reasoning variants of those models do, and how failure handling has been treated in cognitive and engineering research.

## A. Tool use and agent evaluation

Interleaving reasoning with tool calls became standard practice after ReAct [7], and models are now trained to call functions directly [1], [8]. Benchmarks followed. ToolLLM covers large API collections [2]; AgentBench spans several interactive environments [3]; WebArena places agents in realistic web tasks [9]; τ-bench adds users and domain rules to conversational tool use [10]. The Berkeley Function-Calling Leaderboard, which we build on, is unusual in that its multiturn split ships executable stateful environments rather than reference call strings, so an agent’s effect on the world can be checked directly [1].

Recent critiques of this literature focus on what the scores hide. Kapoor et al. argue that agent benchmarks reward narrow accuracy at the expense of cost and reliability, and that heldout evaluation practice is weaker than in ordinary machine learning [11]. Cemri et al. build a taxonomy of failure modes for multi-agent systems and find that most failures come from specification and coordination rather than from the base model’s limits [12]. Both point toward measuring the shape of failures rather than counting them, which is what we try to do here for the single-agent case.

TABLE I  
STUDIES THAT INJECT FAULTS INTO THE TOOL LAYER, BY WHAT THEY MEASURE.
<table><tr><td>Study</td><td>Setting</td><td>Faults</td><td>Primary outcome</td></tr><tr><td>Sun et al. [26]</td><td>calculator; embodied planning</td><td>silent tool errors</td><td>detection when asked to accept or reject</td></tr><tr><td>PALADIN [27]</td><td>ToolBench</td><td>execution errors</td><td>recovery rate after training</td></tr><tr><td>Reliability- Bench [28]</td><td>four simulated</td><td>timeouts, rate limits, partial</td><td>success under fault, by end</td></tr><tr><td></td><td>domains</td><td>responses, schema drift</td><td>state</td></tr><tr><td>ToolMaze [29]</td><td>multi-path tool tasks</td><td>explicit/implicit, tran- sient/permanent</td><td>recovery rate under perturbation</td></tr><tr><td>This work</td><td>BFCL multi-turn</td><td>timeout, missing tool, schema drift, silent corruption</td><td>detection, replanning, recovery, repetition, separately</td></tr></table>

## B. Faults in tool-using agents

A small set of studies injects failures into the tool layer itself, and they are the closest work to ours. Sun et al. showed that language models over-trust tools that fail silently: with a deliberately faulty calculator and in an embodied planning setting, models often accepted incorrect outputs that carried no error signal, and a short in-context disclaimer improved detection [26]. PALADIN injects execution errors from an existing failure taxonomy into ToolBench trajectories and trains agents to recover, raising recovery from 23.75% to 89.86% against untrained agents [27]. ReliabilityBench adapts chaos engineering to agents, injecting timeouts, rate limits, partial responses, and schema drift across four domains and scoring success by end-state equivalence [28]. ToolMaze crosses explicit and implicit perturbations with transient and permanent ones in tasks with alternative tool paths, and finds that recovery falls most under implicit failures, which it traces to over-trust in corrupted outputs [29].

Table I sets these studies beside ours. Their numbers are not directly comparable to ours, since tasks, faults, and outcome definitions differ, but their agreement on silent failures is itself informative, and our results add to it. Our study shares the fault-injection approach and the observation that silent failures are the hardest. It differs in what it measures: detection is coded separately from replanning and recovery and given a no-fault baseline, recovery is scored against each model’s own clean run, and reasoning and instruct models are compared within the same family.

## C. Reasoning models

Models that spend a controllable budget on an internal deliberation phase before answering now sit alongside their instruction-tuned siblings [13]–[15], and extra test-time computation buys real gains on tasks with verifiable answers [16], [17]. Whether the deliberation is what produces those gains is contested. Turpin et al. showed that chain-of-thought text can rationalise an answer driven by a cue the text never mentions [18]; Chen et al. found the same inside reasoning models, which name the hints that moved them only part of the time [19]; Shojaee et al. report reasoning effort that collapses as problems get harder [20]. Our design gives these models a different kind of test. Nothing in a fault trial rewards longer deliberation by itself, and the question is whether the extra phase helps an agent notice that its environment has stopped cooperating.

## D. Failure handling

Injecting faults to see how a system copes is old practice in dependable computing [21] and current practice in production engineering, where controlled failures are introduced into live services to expose weaknesses before customers find them [22]. What that tradition measures is the system’s recovery, not its awareness, because conventional software has no awareness to measure. Agents do, in the limited sense that their output states a reading of the situation, and that opens the door to the distinction the human factors literature has long drawn between detecting an automation failure and responding to it [4], [23]. Work on self-correction in language models sits on the response side of that line: Reflexion has agents write lessons for themselves after a failed attempt [24], and Self-Refine loops critique and revision [25]. Both assume the problem has been identified. The present study asks how often that assumption holds.

## III. METHOD

This section describes the tasks, the faults and how they were injected, the agent protocol, the measures, and the checks we ran on the resulting data.

## A. Tasks and environments

Tasks come from the multi-turn split of the Berkeley Function-Calling Leaderboard, which pairs each scenario with Python classes that hold state and expose tools as methods [1]. We drove those classes directly rather than through the official evaluation harness, which is what makes controlled injection possible: the tool layer is ours, so a fault can be placed at a chosen step and nowhere else.

Of 400 candidate tasks, 312 passed three filters: every environment class they use imports and runs locally, the task has between three and ten user turns, and an executable groundtruth call sequence is available. From those we sampled 24 tasks with a fixed seed, balanced across environment domains, and froze the suite before any trial was run. The frozen set spans 16 domain combinations, including a file system, a trading interface, travel booking, vehicle control, messaging, and a ticketing system, with a median of five user turns per task.

## B. Fault taxonomy

Four faults were chosen to separate failures an agent can read from failures it must infer, with a fifth no-fault condition as baseline (Table II). Timeout raises a timeout error. Missing tool removes a tool from the registry and reports it as unavailable. Schema drift rejects the call with an unknown-argument error, as if a parameter had been renamed. Silent corruption lets the call succeed and returns a value of the same type and shape as the real one, altered so that its content is wrong: a number moved by 20 to 60 percent, a list with an element dropped, a field replaced. When the real result is empty, a placeholder string is returned instead; this happened in 2 of the 374 analysed corruption trials. The first three place an error string in front of the agent. The fourth does not, and we refer to these two situations as loud and quiet failures.

## C. Injection procedure

The fault fires once, on the first eligible call at or after step ⌊T/2⌋, where T is the number of user turns in the task. In practice this placed faults early: the median firing step was 3, against a median of 7 calls in the reference solutions, and 42.1% of faults fired at step 1 or 2. Timing differs by fault type, because the two faults that target a specific tool wait for the agent to call it: timeouts and corruption fired at a median of step 2, missing tool and schema drift at step 5. Later calls behave normally, which means recovery is available by construction rather than blocked by the manipulation.

Two properties of the layer matter for interpretation. Faults do not leave the environment half-changed: corruption alters the returned value only, and the underlying state is what a successful call would have produced. Second, the two faults that must target a particular tool (missing tool and schema drift) take that tool from the task’s ground-truth call path rather than from a heuristic, because a fault attached to a tool the agent never calls is not a fault at all; even so, these two faults fired in 71% and 74% of their trials. Timeout fired in 99% of its trials and corruption in 97%, since neither depends on a specific tool. Trials in which the fault never fired are reported in Table II and excluded from analysis, leaving 1,694 of 1,920 trials.

A self-test accompanies the layer and is run before collection. It asserts, on every task, that the clean condition never faults, that each fault type fires exactly once and at or after its target step, that corruption changes the returned value while preserving its type (apart from the empty-result case above) and leaving environment state untouched, and that the same inputs produce the same fault behaviour on repeated runs.

## D. Agent protocol

Each trial follows the benchmark’s multi-turn structure. A user turn is appended to the conversation, the model acts until it replies without calling a tool, and the next turn follows. Tools are exposed as JSON schemas through the provider’s native function-calling interface, and every call passes through the injection layer. The system prompt tells the agent to work step by step, to call one tool at a time, and to keep working toward the goal if a tool behaves unexpectedly; it does not mention the experiment. Trials stop after 15 tool calls, a bound we report rather than hide, since exhausting it is one of the outcomes of interest.

TABLE II  
CONDITIONS, VISIBILITY, AND HOW OFTEN EACH FAULT FIRED.
<table><tr><td>Condition</td><td>Visibility</td><td>Trials</td><td>Fired</td><td>Analysed</td></tr><tr><td>clean</td><td>baseline</td><td>384</td><td>1</td><td>384</td></tr><tr><td>timeout</td><td>loud</td><td>384</td><td>99%</td><td>379</td></tr><tr><td>missing tool</td><td>loud</td><td>384</td><td>71%</td><td>273</td></tr><tr><td>schema drift</td><td>loud</td><td>384</td><td>74%</td><td>284</td></tr><tr><td>silent corruption</td><td>quiet</td><td>384</td><td>97%</td><td>374</td></tr></table>

Loud faults surface an explicit error; the quiet fault returns a well-formed but incorrect value. Missing tool and schema drift fire only when the agent calls the affected tool.

TABLE III  
MODEL PANEL. EACH FAMILY CONTRIBUTES A MATCHED REASONING AND INSTRUCT PAIR.
<table><tr><td>Model</td><td>Family</td><td>Type</td><td>Trials</td><td>Prompt arm</td></tr><tr><td>claude-haiku-4.5-thinking</td><td>claude</td><td>reasoning</td><td>480</td><td>yes</td></tr><tr><td>claude-haiku-4.5</td><td>claude</td><td>instruct</td><td>480</td><td>yes</td></tr><tr><td>deepseek-r1</td><td>deepseek</td><td>reasoning</td><td>240</td><td>一</td></tr><tr><td>deepseek-v3</td><td>deepseek</td><td>instruct</td><td>240</td><td></td></tr><tr><td>qwen3-thinking</td><td>qwen</td><td>reasoning</td><td>240</td><td></td></tr><tr><td>qwen3-instruct</td><td>qwen</td><td>instruct</td><td>240</td><td></td></tr></table>

1,920 trials over 24 tasks, five conditions, two seeds. The secondary scaffolding analysis doubles one pair.

Six models were used, forming three matched pairs so that the reasoning contrast never crosses model families (Table III). Reasoning variants ran with a 2,048-token thinking budget. All calls were routed through a single gateway to keep the interface uniform, sampling temperature was 1.0, and each cell was run twice; the provider’s sampling is not seeded, so the two runs are independent samples. The full design gives $2 4 \times 5 \times 2 = 2 4 0$ trials per model, plus a second arm for one matched pair described in Section IV-D, for 1,920 trials in total. A seventh and eighth model, a free-tier Nemotron pair, were dropped before analysis after repeated rate-limit failures, with no completed trials; among the six models analysed, no trial failed at the API level.

## E. Measures

Let $S _ { a }$ be the set of public state fields of the environment at the end of a trial and S the same fields from a reference run. State agreement is the share of reference fields the trial reproduces:

$$
A ( S _ { a } , S _ { r } ) ~ = ~ { \frac { | \{ f \in S _ { r } : S _ { a } [ f ] = S _ { r } [ f ] \} | } { | S _ { r } | } } ~ \in ~ [ 0 , 1 ] .\tag{1}
$$

Recovery uses each model as its own reference. Writing $S _ { c } ^ { m , t , s }$ for the end state model m reaches on task t with seed s in the clean condition,

$$
\mathrm { r e c o v e r e d } \ = \ k ^ { \prime } \big [ A \big ( S _ { a } , \ : S _ { c } ^ { m , t , s } \big ) = 1 \big ] .\tag{2}
$$

Scoring against the model’s own clean run rather than against the benchmark’s ground truth removes baseline competence from the comparison: a model that cannot finish a task without a fault is not counted as failing to recover from one. Because sampling is not seeded, two clean runs of the same model and task need not end in the same state. Clean trials are therefore scored the same way: each clean run without the prompt line is scored against the other repetition of the same model and task, and each clean run in the prompt arm (Section IV-D) against the run without the line. Pooled, these give the run-to-run agreement that fault trials are compared with (58.3% and 78.1% for the two groups, 63.3% overall). Effort is measured on the same footing, as steps used relative to the clean runs of that model and task:

$$
\mathrm { c o s t } = n _ { \mathrm { s t e p s } } - \frac { 1 } { | C | } \sum _ { c \in C } n _ { \mathrm { s t e p s } } ^ { c } , C = C _ { \mathrm { c l e a n } } ^ { m , t } .\tag{3}
$$

Three further measures come from the trajectory. Replanning is true when the first action after the fault differs from the action that failed, by tool or by arguments. Repetition is true when three or more consecutive calls after the fault go to the same tool, whether or not the arguments change; strictly identical repetition, with the same arguments, is reported alongside. Neither is defined for clean trials. Budget exhaustion is true when the trial ends at the step cap.

## F. Coding failure detection

Whether an agent treated a result as a problem is a judgement about language, so it was coded by a model and validated against a human. The judge sees one tool call, the result it returned, and what the agent said next. It never learns whether a fault was injected or which condition the trial belongs to, and clean trials are coded the same way using the trajectory midpoint, which gives a baseline for how often an agent reports a problem when there is none.

We first tried a three-way scheme separating no reaction, acknowledgement, and cases where the agent noticed an anomaly and explained it away. One author hand-coded 100 trials, drawn at random from the analysable set during an earlier scoring pass, blind to the judge’s labels; the coded sheet is released with the data. The three-way scheme reached only $\kappa = . 4 7 ;$ most disagreements (19 of 32) were trials the human read as acknowledgement and the judge labelled as explaining away. On the 92 trials that both the author and the judge coded, collapsing to the binary distinction (did the agent treat this result as a problem) gave 85.9% agreement and κ = .69, which is the measure used throughout. The three-way labels are retained in the released data but no claim rests on them.

## G. Statistical analysis

Trials are not independent: the same 24 tasks recur across models, faults, and seeds. Inference therefore rests on taskclustered estimates, namely binomial generalized estimating equations with an exchangeable working correlation clustered on task, and bootstrap intervals that resample tasks. Per-family contrasts in RQ2 use task-clustered models on the matched pairs, which balance fault type by design. Two-proportion ztests, and paired t-tests in the secondary analysis, are reported alongside as descriptive comparisons. Holm correction is applied within each family of tests, and conclusions rest on the clustered results.

## H. Data quality

Three facts about the dataset belong with the results. Detection on clean trials whose target call succeeded ran at 26.8%, and this rate is treated as the baseline rather than as noise: these environments produce genuine errors of their own, such as authentication and balance messages, and clean trials whose target call returned one of those were acknowledged 90% of the time, which is correct behaviour rather than a false alarm. Trials ended at the step cap 27% of the time overall. One model, DeepSeek-R1, wrote tool-call syntax into its message text rather than emitting a structured call in 1.8 turns per trial on average; those calls never execute, and the behaviour is a property of that model in this interface that we note as a limitation.

## IV. RESULTS

This section answers the three research questions in turn and then reports the secondary analysis on prompting the agent to check its results.

## A. RQ1: Detection depends on visibility, not severity

Figure 1 and Table IV give the first result. When a tool returned an explicit error, agents treated it as a problem in 91.3% of trials (95% CI $[ 8 8 . 3 , 9 4 . 1 ] , n = 9 3 6 )$ . When a tool returned a plausible wrong value, they did so in 58.8% of trials $( [ 4 5 . 7 , 7 1 . 2 ] , n = 3 7 4 )$ . The gap is 32.5 points, $z = 1 3 . 8 6$ $p < . 0 0 1 , h = 0 . 8 0$ . Both sit above the rate at which agents reported a problem when the target call had in fact succeeded, 26.8% ([17.8, 36.5], n = 306); quiet failures clear that baseline by 32.0 points $( z = 8 . 3 6 , p < . 0 0 1 , h = 0 . 6 6 )$ and loud ones by 64.5 points (z = 22.77, $p < . 0 0 1$ , h = 1.46).

The cleanest version of this contrast is between timeouts and corruption, which fired at the same median step: 91.6% against 58.8%. The three loud faults land within a point of each other—91.6% for timeouts, 91.2% for a missing tool, 91.2% for a renamed parameter—despite differing in what the agent must do next. Corruption sits 32 points below all three. A binomial generalized estimating equation with an exchangeable working correlation clustered on task puts quiet failures at $\mathrm { O R } = 3 . 9 6 $ against the no-fault baseline $( z = 6 . 4 8$ $p \ < \ . 0 0 1 )$ and loud failures at $\mathrm { O R } \ = \ 2 8 . 6 1 \ ( z \ = \ 1 2 . 1 4$ $p < . 0 0 1 )$ .

The right panel of Figure 1 shows the composition behind those rates. Under a loud fault, 4% of trials pass without the agent treating the result as a problem. Under corruption that figure is 39%. In those trials the agent reads a wrong value and carries it forward, sometimes reconciling it with what the conversation had established earlier.

![](images/ae5aba1d7957c0b54a9d828da8ab5cb0512b996c2dc0a22a37dd38e9b2edfbb5.jpg)

![](images/929fcb9e58acd3b35710dfdec3d73b612460cf37e4af7da5dd8d52b934882fa0.jpg)  
Fig. 1. Detection by failure visibility. Left: share of trials in which the agent treated the tool result as a problem, with task-clustered bootstrap intervals. Right: how those responses divide, where “none” means the result was used or reported as if it were fine.

TABLE IV  
DETECTION RATES WITH TASK-CLUSTERED BOOTSTRAP INTERVALS.
<table><tr><td>Condition</td><td>n</td><td>Detected</td><td>95% CI</td></tr><tr><td>no fault (baseline)</td><td>306</td><td>26.8%</td><td>[17.8, 36.5]</td></tr><tr><td>quiet failure</td><td>374</td><td>58.8%</td><td>[45.7, 71.2]</td></tr><tr><td>loud failure</td><td>936</td><td>91.3%</td><td>[88.3, 94.1]</td></tr><tr><td>timeout</td><td>379</td><td>91.6%</td><td>[88.0, 94.5]</td></tr><tr><td>missing tool</td><td>273</td><td>91.2%</td><td>[86.5, 95.2]</td></tr><tr><td>schema drift</td><td>284</td><td>91.2%</td><td>[86.3, 95.4]</td></tr><tr><td>silent corruption</td><td>374</td><td>58.8%</td><td>[45.7, 71.2]</td></tr></table>

All fault conditions differ from the baseline at $p < . 0 0 1$ after Holm correction.

## B. RQ2: Reasoning models act more and notice less

Pairing each reasoning model against the instruct sibling from its own family, on the same task, fault, and seed, produces the pattern in Figure 2 and Table V. On detection the reasoning model is lower in all three families: −6.2 points for Claude $( \mathrm { O R } ~ = ~ 0 . 6 1 , ~ p _ { \mathrm { H o l m } } ~ = ~ . 0 4 2 ) , ~ - 4 . 0$ points for DeepSeek $( p _ { \mathrm { H o l m } } ~ = ~ . 4 3 4 )$ , and −16.6 points for Qwen $\mathrm { { ( O R = 0 . 4 0 , } }$ $p _ { \mathrm { H o l m } } = . 0 0 3 )$ , in task-clustered models on the matched pairs. Pooled over 450 pairs the difference $\mathrm { i s \ - 9 . 3 }$ points (taskclustered 95% CI $[ - 1 4 . 3 , - 5 . 3 ] )$ , and a task-clustered model with fault type as a covariate gives $\mathrm { O R \ = \ 0 . 5 6 }$ for the reasoning term $( z ~ = ~ - 4 . 2 8 , ~ p ~ < ~ . 0 0 1 )$ . Qwen carries the clearest single-family result. Claude’s difference holds on the matched pairs but is not robust to specification: modelled over all of its fault trials with fault type as a covariate, it is not significant $( p _ { \mathrm { { H o l m } } } = . 2 0 9 )$ . DeepSeek, whose reasoning model sometimes wrote tool calls as text (Section III-H), shows the smallest difference.

Replanning runs the other way. Reasoning models change their next action more often, by 4.3 points for Claude, 8.7 for DeepSeek, and 17.8 for Qwen, pooling to +10.4 points (task-clustered 95% CI $[ + 5 . 6 , + 1 5 . 7 ] ; \mathrm { O R } = 2 . 0 2 , z = 3 . 5 2 ,$ $p < . 0 0 1 )$ . Of the single families, only Qwen’s difference is reliable $( \mathrm { O R } = 4 . 2 3 , \ p _ { \mathrm { H o l m } } < . 0 0 1$ ; Claude and DeepSeek $p _ { \mathrm { H o l m } } = . 4 1 7 )$

Recovery shows neither advantage. No family differs $( p _ { \mathrm { H o l m } } = 1 . 0 0 0$ throughout), and the pooled estimate is −2.4 points (task-clustered 95% CI [−9.3, +4.5]; $\mathrm { { O R } \ = \ 0 . 9 1 }$ $p \ = \ . 5 1 2 )$ . Whatever the extra deliberation phase changes about how these agents behave after a fault, it does not change how often they end up where they were going.

![](images/893c047a64a18255e8f435a6fae565a704a5914cefe7f288dc24e7b3338c9731.jpg)  
Fig. 2. Reasoning variants minus matched instruct siblings on fault trials, paired within family by task, fault, and seed. Points are mean differences in percentage points with task-clustered 95% bootstrap intervals; asterisks mark contrasts significant in task-clustered models after Holm correction. Noticing sits left of zero in every family while acting sits right of it, and recovery straddles the line.

## C. RQ3: What faults cost

Judging what a fault costs requires knowing what happens without one. Two fault-free runs of the same model and task end in the same state in 63.3% of trials (task-clustered 95% CI [53.1, 73.2]), and the figure varies widely by model, from 84.4% for the Claude instruct model to 41.7% for DeepSeek-R1, DeepSeek-V3, and the Qwen3 thinking model. Against that baseline, only a missing tool clearly lowers recovery: 39.9% ([28.1, 52.4], $h \ = \ - 0 . 4 7 ; \ 0 \mathrm { R } \ = \ 0 . 3 4 , \ p _ { \mathrm { H o l m } } <$ .001). Timeouts (59.6%), schema drift (59.2%), and corruption (60.4%) stay within run-to-run variation (OR 0.80 to 0.88, $p _ { \mathrm { H o l m } } ~ \geq ~ . 1 1 7 ) ;$ schema drift reaches significance only once model is added as a covariate $( \mathrm { O R } = 0 . 7 6 , p _ { \mathrm { H o l m } } = . 0 2 4 )$ Losing a capability is the one fault no retry can undo, and it is the one that clearly moves end states.

That corruption leaves end states largely intact while going unnoticed in 41% of trials is less reassuring than it sounds. Recovery here is agreement in environment state, and a wrong value that the agent reads rather than writes changes what it reports, not what it does to the environment.

Effort tells a second story (Figure 3, Table VI). After a fault, agents called the same tool three or more times in a row in 12.5% to 22.2% of trials, most often under corruption (22.2%) and timeouts (21.4%). These are mostly retries with changed arguments rather than loops: strictly identical repetition, with the same tool and arguments, occurred in 0.8% to 1.8% of fault trials. Trials end at the step cap between 24.8% and 31.7% of the time under faults, against 23.2% when nothing goes wrong. The extra steps a fault costs relative to a clean run are modest for timeouts (+0.53) and corruption (+0.56) and larger where the agent has to rebuild its plan, at +1.47 for a missing tool and +1.85 for a renamed parameter.

Whether noticing helps deserves a careful answer. Compared directly, trials in which the agent treated the result as a problem recovered at 55.6% and trials in which it did not at 55.7%, a difference of nothing at all $( z = - 0 . 0 3 , p = . 9 7 4 )$ That comparison is confounded: corruption is both the least detected fault and one of the more recoverable ones, so pooling the conditions cancels the effect. Holding fault type fixed, noticing is associated with recovery $( \mathrm { O R } = 1 . 5 1 , z = 2 . 9 1$ $p = . 0 0 4 )$ , but the association weakens once model is also held fixed $( \mathrm { O R } \ = \ 1 . 2 9 , \ p \ = \ . 0 9 8 )$ , so part of it reflects which models notice more rather than what noticing does. Detection may be worth having; this study cannot show that it is sufficient.

![](images/32b8c9f62a32518f267f5a3ab61a6e719513491a210ea52c9ecae152c9ffd668.jpg)  
Fig. 3. Consequences of a single fault. Left: share of trials reaching the state of an independent fault-free run of the same model and task, against the agreement of two fault-free runs (63.3%). Right: how often the agent called the same tool three or more times in a row after the fault, with the extra steps a fault costs relative to a clean run listed alongside.

## D. Secondary analysis: asking the agent to check its work

One line was added to the system prompt for a matched pair of models, in a second arm of 480 trials: after every tool call, briefly check whether the result is what you expected. The instruction did not move detection. For the instruct model the change was +0.6 points $( t ( 1 6 1 ) = 0 . 3 0 , p = . 7 6 4 )$ and for the reasoning variant +2.5 points $( t ( 1 6 1 ) = 0 . 9 4 , p = . 3 4 7 ) ;$ recovery, repetition, and false alarms were likewise unchanged. Splitting by visibility, detection under quiet failures went from 60.6% to 59.6%, and the interaction between the instruction and visibility does not reach significance $( z = - 1 . 3 5 , p =$ .176).

This arm ran on one family, so it does not license a general claim about metacognitive prompting. What it does show is that the obvious first remedy, asking for a check in words, left the gap in Figure 1 where it was. The contrast with Sun et al. is worth noting: there, a short disclaimer improved detection of silent errors [26]. Their models judged a single output on request, while ours had to notice the anomaly mid-task while carrying a plan, which may be where a general instruction to check loses its force.

## V. DISCUSSION

The results describe a specific shape of brittleness, and it is not the one an aggregate success rate would suggest.

## A. Agents watch the channel, not the content

Three faults that differ in what they demand (retry, rebuild the plan, reformulate the call) produced detection rates within a point of one another, while the fault that arrives without an error string produced a rate 32 points lower. The common feature of the first three is the error itself. Reading it takes no domain knowledge and no comparison against expectation, which is what noticing a corrupted value requires: the fuel level exceeds the tank, an item has gone missing from a list that was longer a moment ago, a symbol does not match the company that was asked about. Agents do this sometimes, at rates well above the no-fault baseline, but they miss it in 41% of trials, and in some of those they smooth the inconsistency over in their report to the user.

TABLE V  
REASONING VERSUS INSTRUCT SIBLINGS ON FAULT TRIALS, PAIRED WITHIN FAMILY BY TASK, FAULT, AND SEED.
<table><tr><td>Outcome</td><td>Family</td><td>Reasoning</td><td>Instruct</td><td>△ (points)</td><td>Clustered 95% CI</td><td>OR</td><td>PHolm</td></tr><tr><td>detection</td><td>claude</td><td>82.0%</td><td>88.2%</td><td>-6.2</td><td> $[ - 1 1 . 4 , - 1 . 2 ]$ </td><td>0.61</td><td>.042</td></tr><tr><td>detection</td><td>deepseek</td><td>78.6%</td><td>82.5%</td><td>-4.0</td><td> $\dot { \left[ - 1 4 . 8 , + 5 . 0 \right] }$ </td><td>0.78</td><td>.434</td></tr><tr><td>detection</td><td>qwen</td><td>67.5%</td><td>84.0%</td><td>-16.6</td><td> $[ - 2 7 . 2 , - 6 . 2 ]$ </td><td>0.40</td><td>.003</td></tr><tr><td>replanning</td><td>claude</td><td>75.2%</td><td>70.8%</td><td>+4.3</td><td> $[ - 2 . 3 , + 1 1 . 8 ]$ </td><td>1.25</td><td>.417</td></tr><tr><td>replanning</td><td>deepseek</td><td>74.6%</td><td>65.9%</td><td>+8.7</td><td> $[ - 5 . 5 , + 2 3 . 8 ]$ </td><td>1.52</td><td>.417</td></tr><tr><td>replanning</td><td>qwen</td><td>92.6%</td><td>74.8%</td><td>+17.8</td><td> $[ + 1 1 . 3 , + 2 4 . 2 ]$ </td><td>4.23</td><td>&lt;.001</td></tr><tr><td>recovery</td><td>claude</td><td>66.5%</td><td>68.3%</td><td>-1.9</td><td> $[ - 9 . 0 , + 4 . 9 ]$ </td><td>0.92</td><td>1.000</td></tr><tr><td>recovery</td><td>deepseek</td><td>38.9%</td><td>42.1%</td><td>-3.2</td><td> $[ - 1 4 . 6 , + 8 . \dot { 6 } ]$ </td><td>0.88</td><td>1.000</td></tr><tr><td>recovery</td><td>qwen</td><td>44.2%</td><td>46.6%</td><td>-2.5</td><td>[−15.3, +10.1]</td><td>0.91</td><td>1.000</td></tr></table>

Negative ∆ means the reasoning variant scored lower. OR and p<sub>Holm</sub> from task-clustered models on the matched pairs, Holm-corrected across families. Pooled over 450 pairs: detection −9.3 points (p < .001), replanning +10.4 points (p < .001), recovery −2.4 points $( p = . 5 1 2 )$

TABLE VI  
RECOVERY AND EFFORT BY CONDITION.
<table><tr><td>Condition</td><td>Recovered</td><td>Same tool</td><td>Identical</td><td>Cap hit</td><td>Extra steps</td></tr><tr><td>clean</td><td>63.3%</td><td></td><td></td><td>23.2%</td><td>+0.01</td></tr><tr><td>timeout</td><td>59.6%</td><td>21.4%</td><td>1.1%</td><td>24.8%</td><td>+0.53</td></tr><tr><td>missing tool</td><td>39.9%</td><td>12.5%</td><td>1.1%</td><td>29.3%</td><td>+1.47</td></tr><tr><td>schema drift</td><td>59.2%</td><td>17.6%</td><td>1.8%</td><td>31.7%</td><td>+1.85</td></tr><tr><td>silent corruption</td><td>60.4%</td><td>22.2%</td><td>0.8%</td><td>25.9%</td><td>+0.56</td></tr></table>

Clean recovery is the agreement of two independent fault-free runs. “Same tool” and “Identical” are three or more consecutive calls after the fault, to the same tool or with the same tool and arguments; neither is defined for clean trials. Only missing tool differs from clean in recovery (task-clustered GEE, p < .001).

That is the practical warning in this study. Loud failures are the ones a monitoring system already catches, and the agent catches them too. Quiet failures are the ones that reach the user as confident output, and they are the ones agents are worst at, which inverts the usual assumption that an agent adds a layer of checking on top of the tools it calls. This agrees with the over-trust reported in single-step and multi-path settings [26], [29]; what our design adds is where in the chain it bites, at noticing rather than at acting. The human factors literature has a name for the underlying pattern, where operators accept automated output that is wrong because nothing in its presentation says otherwise [4], [23], and the correspondence here is close enough to be worth naming even though the mechanism need not be the same.

## B. Acting is not the same as noticing

The dissociation in Section IV-B is the clearest evidence that these components come apart. Reasoning variants changed their next action more often than their instruct siblings (the behaviour one would call adaptive) while treating fewer results as problems, and the two effects met in the middle at no difference in recovery. A reading consistent with the tracelevel work on these models [18], [19] is that the deliberation phase supports generating an alternative course of action more than it supports checking a returned value against what the task implies. Whatever the cause, the practical reading is direct: the reasoning tier of a model family was not the safer choice under faults in this study, and choosing it on the assumption that more deliberation means more caution would not be supported by these data.

## C. Implications for deployment

Four recommendations follow. First, monitor tool outputs rather than only tool errors: the failures agents miss are the ones your error-rate dashboards also miss, so validation belongs at the point where a value enters the agent’s context. Second, do not treat detection as an alarm bell that has already rung; noticing was only modestly associated with recovery, and the association weakened once model was held fixed. Third, budget for the effort a fault costs. Recovery after a capability loss fell to 39.9% against a 63.3% faultfree baseline, agents returned to the same tool three or more times in up to 22% of fault trials, and trials terminated at the step cap in nearly a third of the harder conditions, all of which turns into latency and spend in production. Fourth, judge fault effects against repeated fault-free runs: in our data, a single clean run as the reference would have made three faults look harmful that left end states within ordinary run-torun variation.

## VI. LIMITATIONS

The boundaries of this study are worth stating plainly. The tasks are simulated environments from one benchmark family; they are executable and stateful, which is what the design needs, but they are not production services, and the fault types are a chosen four rather than an exhaustive set. Two of those four fire only when the agent calls the affected tool, at 71% and 74% of trials, and although drawing the target from the ground-truth path removed most of the earlier bias, the analysed subset for those conditions is still the set of trials where the agent’s own choices met the fault. Detection is coded from what the agent wrote, so an agent that noticed something and said nothing counts as not noticing; the binary coding reached $\kappa = . 6 9$ against hand-coding, which supports the comparisons made here but is not perfect agreement, and multiple human coders or behavioral signals of detection that do not rely on language would sharpen this measure. Trials were capped at 15 tool calls, and 27% ended at that cap, so recovery rates are recovery within a bounded effort budget. Recovery is agreement with a single independent fault-free run; because two such runs agree only 63.3% of the time, the measure is noisy, and fault effects are judged against that baseline rather than against perfect agreement. Faults fired earlier than a midpoint design would place them, and timing differed by fault type, although timeouts and corruption, the cleanest loud-quiet contrast, fired at the same median step. The reasoning comparison rests on three matched pairs, and in one of them the reasoning model wrote tool calls as text rather than emitting them, which lowers its effective action rate in a way specific to this interface; the detection difference is clearest for Qwen and holds for Claude only on matched pairs, and three pairs cannot establish a general property of reasoning models. Finally, the scaffolding arm covers one model family, and its null result should be read as one prompt on two models rather than as a verdict on prompting.

## VII. CONCLUSION

We injected typed faults into the tool layer of an executable agent benchmark and measured what agents do next, separating whether they notice a failure from whether they act on it and whether they recover. Across 1,920 trials and six models, agents treated a failure as a problem in 91.3% of trials when the tool reported an error and 58.8% when the tool returned a well-formed wrong value, against a 26.8% baseline for reporting problems that did not exist. Reasoning variants noticed less than their instruct siblings and replanned more, with no gain in recovery, and a prompt asking for a check after each call did not close the gap. Against the 63.3% agreement of two fault-free runs, only a missing tool clearly lowered recovery, to 39.9%. In line with earlier reports of over-trust in silent tool errors, the failures that agents handle well are the ones that announce themselves. The harness, the frozen task suite, and the scored dataset are released so that other fault types, environments, and models can be measured on the same footing.

## REFERENCES

[1] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, “Gorilla: Large language model connected with massive APIs,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 37, 2024.

[2] Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, S. Zhao, L. Hong, R. Tian, R. Xie, J. Zhou, M. Gerstein, D. Li, Z. Liu, and M. Sun, “ToolLLM: Facilitating large language models to master 16000+ real-world APIs,” in International Conference on Learning Representations (ICLR), 2024.

[3] X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang, S. Zhang, X. Deng, A. Zeng, Z. Du, C. Zhang, S. Shen, T. Zhang, Y. Su, H. Sun, M. Huang, Y. Dong, and J. Tang, “AgentBench: Evaluating LLMs as agents,” in International Conference on Learning Representations (ICLR), 2024.

[4] R. Parasuraman and V. Riley, “Humans and automation: Use, misuse, disuse, abuse,” Human Factors, vol. 39, no. 2, pp. 230–253, 1997.

[5] M. R. Endsley, “Toward a theory of situation awareness in dynamic systems,” Human Factors, vol. 37, no. 1, pp. 32–64, 1995.

[6] A. Miyake, N. P. Friedman, M. J. Emerson, A. H. Witzki, A. Howerter, and T. D. Wager, “The unity and diversity of executive functions and their contributions to complex frontal lobe tasks: A latent variable analysis,” Cognitive Psychology, vol. 41, no. 1, pp. 49–100, 2000.

[7] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in International Conference on Learning Representations (ICLR), 2023.

[8] T. Schick, J. Dwivedi-Yu, R. Dess\`ı, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[9] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, U. Alon, and G. Neubig, “WebArena: A realistic web environment for building autonomous agents,” in International Conference on Learning Representations (ICLR), 2024.

[10] S. Yao, N. Shinn, P. Razavi, and K. Narasimhan, “τ-bench: A benchmark for tool-agent-user interaction in real-world domains,” arXiv preprint arXiv:2406.12045, 2024.

[11] S. Kapoor, B. Stroebl, Z. S. Siegel, N. Nadgir, and A. Narayanan, “AI agents that matter,” arXiv preprint arXiv:2407.01502, 2024.

[12] M. Cemri, M. Z. Pan, S. Yang, L. A. Agrawal, B. Chopra, R. Tiwari, K. Keutzer, A. Parameswaran, D. Klein, K. Ramchandran, M. Zaharia, J. E. Gonzalez, and I. Stoica, “Why do multi-agent LLM systems fail?” arXiv preprint arXiv:2503.13657, 2025.

[13] A. Jaech, A. Kalai, A. Lerer et al., “OpenAI o1 system card,” arXiv preprint arXiv:2412.16720, 2024.

[14] D. Guo, D. Yang, H. Zhang et al., “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.

[15] A. Yang, A. Li, B. Yang et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[16] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022.

[17] C. Snell, J. Lee, K. Xu, and A. Kumar, “Scaling LLM test-time compute optimally can be more effective than scaling model parameters,” arXiv preprint arXiv:2408.03314, 2024.

[18] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, “Language models don’t always say what they think: Unfaithful explanations in chainof-thought prompting,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[19] Y. Chen, J. Benton, A. Radhakrishnan et al., “Reasoning models don’t always say what they think,” arXiv preprint arXiv:2505.05410, 2025.

[20] P. Shojaee, I. Mirzadeh, K. Alizadeh, M. Horton, S. Bengio, and M. Farajtabar, “The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity,” arXiv preprint arXiv:2506.06941, 2025.

[21] M.-C. Hsueh, T. K. Tsai, and R. K. Iyer, “Fault injection techniques and tools,” Computer, vol. 30, no. 4, pp. 75–82, 1997.

[22] A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski, J. Reynolds, and C. Rosenthal, “Chaos engineering,” IEEE Software, vol. 33, no. 3, pp. 35–41, 2016.

[23] K. L. Mosier, L. J. Skitka, S. Heers, and M. Burdick, “Automation bias: Decision making and performance in high-tech cockpits,” International Journal of Aviation Psychology, vol. 8, no. 1, pp. 47–63, 1998.

[24] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language agents with verbal reinforcement learning,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[25] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark, “Self-Refine: Iterative refinement with self-feedback,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[26] J. Sun, S. Y. Min, Y. Chang, and Y. Bisk, “Tools fail: Detecting silent errors in faulty tools,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024, pp. 14272–14289.

[27] S. V. Vuddanti et al., “PALADIN: Self-correcting language model agents to cure tool-failure cases,” arXiv preprint arXiv:2509.25238, 2025.

[28] A. Gupta, “ReliabilityBench: Evaluating LLM agent reliability under production-like stress conditions,” arXiv preprint arXiv:2601.06112, 2026.

[29] D. Zhu, X. Ma, Y. Shen, X. Li, Y. Zhao, S. Wang, L. Yan, and D. Yin, “When tools fail: Benchmarking dynamic replanning and anomaly recovery in LLM agents,” arXiv preprint arXiv:2606.05806, 2026.