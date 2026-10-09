# Closed-Loop Evaluation of LLM Agents for Embedded Software Development

Jorge García-Carrasco<sup>1,\*</sup> Sergio García-Carrasco<sup>2</sup> Alejandro Maté<sup>1</sup> Juan Trujillo<sup>1</sup>

<sup>1</sup>Lucentia Research, Department of Software and Computing Systems, University of Alicante, Ctra. de San Vicente del Raspeig s/n, 03690 Sant Vicent del Raspeig, Spain

<sup>2</sup>Universitary Institute of Materials Technology (IUTM), Universitat Politècnica de València, Plaça Ferrándiz i Carbonell, 03801 Alcoi, Spain

Corresponding author: jorge.g@ua.es

ORCID: 0000-0003-3174-083X (J. García-Carrasco), 0009-0001-9854-0309 (S. García-Carrasco), 0000-0001-7770-3693 (A. Maté), 0000-0003-0139-6724 (J. Trujillo)

Published in Journal of Systems Architecture 179 (2026) 103937, https://doi.org/10.1016/j.sysarc.2026.103937. © 2026 The Authors. Published by Elsevier B.V. under the CC BY 4.0 license (https://creativecommons.org/licenses/by/4.0/).

## Abstract

Large language models (LLMs) are increasingly deployed as coding agents that edit files, run builds and tests, inspect execution results, and repair software iteratively. Embedded firmware is a demanding target because correctness depends on closed-loop behavior under sensing, timing, and safety constraints, not only on static source quality. Yet embedded-agent evaluation remains limited and often emphasizes one-shot synthesis or ofline correctness.

We present a benchmark for closed-loop evaluation of embedded coding agents. Each task provides a plain-text engineering description, constrained workspace, and visible buildand-runtime surface. The agent must translate requirements into implementation and self-verification steps, then iterate until the required device behavior is achieved. The suite contains five embedded-control tasks and four feedback scenarios: one-shot generation, realistic self-verification, CI-style red/green feedback, and oracle-style detailed feedback.

The implementation targets simulated ESP32 firmware for reproducibility. We evaluate seven GPT-family and Qwen-family configurations across five tasks and four scenarios, with three repetitions per condition for 420 runs. gpt-5.4 has the highest pass rate among evaluated configurations but does not saturate the benchmark; qwen3.5-27B is the strongest observed local model; and smaller local models degrade sharply in pass rate and search eficiency. These results suggest that capable local embedded coding agents are emerging.

Keywords: embedded software, large language models, language model agents, software engineering benchmarks

Highlights.

• We introduce a closed-loop benchmark for embedded firmware coding agents, i.e., tool-using LLM systems that can autonomously execute bounded build-test-debug loops.

• The benchmark targets why agents matter for embedded work: tasks begin from plain-text engineering handofs and require visible build, test, runtime, repair, and stopping decisions.

• The evaluation spans five tasks, four feedback regimes, seven models, and three repetitions for a total of 420 runs.

• The model set combines two hosted frontier systems with five local Qwen variants.

• gpt-5.4 has the highest observed pass rate among the evaluated configurations, while qwen3.5-27B is the strongest observed local model and points to viable on-premises embedded-agent workflows.

## 1 Introduction

Large language models (LLMs), trained on very large corpora of natural language and source code, now show strong capabilities across a wide variety of tasks [1]. In software settings, they can interpret requirements, summarize repository state, generate candidate implementations, explain failures, and propose repairs [2]. By themselves, however, they usually act as single-turn text generators. Agentic coding systems extend them with a persistent workspace and explicit tool use [3], allowing the model to read and edit files, run builds and tests, inspect logs, and revise a partial solution over multiple steps rather than answering only once. In other words, a coding agent does not only suggest code; within a bounded tool environment, it can autonomously act on a repository, observe the consequences, and continue working until it decides to stop or submit.

This shift from one-shot generation to iterative, feedback-driven work is central to practical automation in software engineering. Recent benchmark work has shown that realistic measurement must account for whether models can process failure signals and improve a solution over repeated attempts [4, 5]. Agent-oriented systems and evaluations further emphasize the importance of planning, tool use, execution, and iterative repair in realistic development environments [6, 7]. The agent framing is therefore useful because many engineering tasks are bottlenecked not by producing an initial candidate, but by validating, debugging, and repairing that candidate until it satisfies the intended behavior. Embedded software is a particularly compelling target for this kind of evaluation: firmware correctness depends on behavior in closed loop with sensors, actuators, timing constraints, and safety interlocks, so success requires more than static source-level quality.

Embedded software development is therefore qualitatively diferent from standard codegeneration tasks. A controller may look reasonable in a code review and still fail once it interacts with asynchronous sensor streams, hysteresis, timeout handling, or sequence control. For the same reason, repository-level issue-resolution benchmarks do not fully cover the embedded setting either: they capture iterative software work, but not the hardware-facing closed loop in which firmware correctness depends on dynamic behavior.

Despite recent interest in applying modern coding agents to embedded and hardware-adjacent domains, the dominant emphasis is still on static synthesis, component-level checks, or ofline correctness [8]. What remains less explored is the full agentic loop that makes modern coding agents useful in practice: starting from a plain-text specification of what must be implemented, formulating tests, implementing firmware, running it, learning from feedback, and iterating until the system behavior matches the requirements.

This motivates the central question of this paper:

Can an agent, starting from a plain-text specification of required device behavior, iteratively write tests and firmware, execute that firmware in a target environment, and converge to the required embedded behavior through a closed feedback loop?

We argue that this question deserves its own benchmark because the capability of interest is not one-shot firmware generation, but iterative embedded engineering. The benchmark we present is intended to evaluate whether an agent can start from a prose engineering brief, translate that brief into implementation and verification actions, and ultimately satisfy a behavioral contract in closed loop. The design is therefore prose-first, contract-aligned, and centered on the real development steps that embedded engineers actually perform: implementing controllers, writing or refining tests, running builds, probing runtime behavior, and iterating on failure signals.

In the current reference instantiation, we realize this methodology with an ESP32 firmware target running in QEMU [9] and deterministic plant simulators that model the controlled process. We simulate both the microcontroller and the plants/tasks because a benchmark should first be safe, repeatable, and easy to run at scale. This makes the evaluation reproducible and suitable for systematic comparison, while still exercising the core embedded loop of sensing, control, actuation, and verification. In this reference stack, the physical sensor/actuator boundary is intentionally virtualized through a UART protocol between firmware and simulator, which abstracts board-specific electrical integration while preserving the closed-loop control logic that the benchmark is meant to test. The ESP32/QEMU stack is therefore a concrete vehicle for the benchmark, not the boundary of the methodology itself.

Contributions. The paper makes the following contributions:

1. A benchmark formulation for closed-loop evaluation of LLM agents on embedded firmware tasks, starting from a prose specification and ending in behavior-level validation.

2. A benchmark design centered on contract-aligned task packets, visible self-verification afordances, a five-task dificulty ladder, and four evaluation scenarios with diferent degrees of feedback visibility.

3. A reference ESP32/QEMU simulation instantiation with five deterministic plant models that enables safe and reproducible closed-loop evaluation before hardware deployment.

4. A pooled 420-run evaluation across five tasks, four feedback scenarios, and seven models spanning hosted and local agents, showing that the benchmark is executable and modelseparating, that gpt-5.4 has the highest observed pass rate without saturating the suite under repeated measurement, and that qwen3.5-27B is the strongest observed local operating point while smaller local models degrade sharply in both accuracy and eficiency.

The remainder of the paper is structured as follows. Section 2 reviews related work. Section 3 presents the benchmark objective, contract, and design principles. Section 4 describes the evaluation setup and reports results across tasks, feedback scenarios, and models. Section 5 discusses implications, limitations, and benchmark-design lessons. Section 6 concludes and outlines future work.

## 2 Related Work

Prior work on software-engineering evaluation has established that realistic measurement requires more than one-pass code generation. Benchmarks such as SWE-Bench and its follow-ups showed the importance of testing whether models can interact with repositories, process failure signals, and revise solutions over multiple iterations [4, 5]. In parallel, agent-oriented systems and evaluations such as SWE-agent [6], Agentless [7], and related program-improvement frameworks have emphasized planning, tool use, execution, and iterative correction in realistic development environments [10]. Together, this line of work provides the general evaluation perspective we adopt: practical capability is not only the ability to synthesize code, but also the ability to use feedback productively in a development loop.

At the same time, synthesis-focused benchmarks such as HumanEval [11], MBPP [12], and RepoBench [13] remain important because they isolate core functional code-generation ability. These datasets provide strong signal on local correctness and pattern-level coding skill. However, they do not fully represent embedded firmware work, where correctness often depends on stateful execution, asynchronous inputs, target constraints, low-level verification concerns [14], and behavior under disturbance. In embedded settings, code that appears plausible in isolation may still fail once it is compiled, executed on a target, and exercised under temporal and safety constraints.

The literature on LLMs for embedded and hardware-adjacent programming has also grown rapidly. Recent benchmark work has explicitly targeted embedded-system development and microcontroller-oriented evaluation, helping establish firmware engineering as a serious target for modern models [8]. Complementary studies have benchmarked LLMs for embedded systems programming in microcontroller-driven IoT applications, further broadening the empirical base for the area [15]. Survey work has also reviewed transformer-based source-code generation more broadly, covering coding-model architectures, tuning strategies, datasets, and evaluation metrics [16]. Adjacent low-level work has examined code generation for hardware-description and synthesis-oriented settings, including Verilog-focused benchmarks that are relevant to the broader question of model capability in hardware-near engineering domains [17, 18]. Related systems work has also studied edge-device ML benchmarking [19] and on-device LLM deployment from a fixed point simulation and hardware co-design perspective [20], while agentic reasoning architectures have been explored for logic-code simulation tasks [21]. This work is valuable in establishing embedded software and related low-level domains as relevant targets for modern models. However, much of the existing evidence still centers on static synthesis, component-level checks, ofline correctness, or narrow domain-specific reasoning tasks. Comparatively less attention has been given to the full agentic workflow in which a model starts from a prose engineering brief, writes or refines self-verification, runs firmware in a target environment, interprets staged feedback, and iterates until behavior-level requirements are satisfied.

Finally, simulation and emulation have long served as foundational evaluation substrates in robotics [22], systems research [9], and firmware analysis, including interrupt modeling from firmware logic [23], because they enable reproducibility, controlled experimentation, and costefective stress testing. For embedded-agent research, simulation ofers the practical advantage of repeated closed-loop execution without the fragility, safety risk, and operational overhead of immediate physical deployment. At the same time, simulation is not a replacement for real-device validation. In this paper, we use it as a deliberate methodological staging ground that supports rigorous comparison of agent behavior today, while informing future transitions to hardware-in-the-loop and real-machine studies.

Taken together, prior work motivates a benchmark that embraces the agentic evaluation workflow now common in academia and industry for realistic assessment of LLM agents in embedded development.

## 3 Benchmark Design

This benchmark evaluates whether an LLM agent can carry out a realistic embedded development loop from a natural-language engineering handof. For each task, the agent receives a plaintext specification of required behavior and is expected to translate that specification into implementation and verification actions. In practice, this includes writing firmware, creating or refining tests, invoking builds and checks, interpreting failures, and iterating until behavior meets the stated requirements.

The benchmark is designed around simulated tasks so that experiments remain safe, repeatable, and comparable across models. Simulation allows many controlled runs with consistent task conditions while still exercising the core engineering workflow that matters for embedded automation. Detailed interfaces, execution mechanics, and evaluator behavior are introduced in the following subsections.

## 3.1 Benchmark objective

The benchmark is designed to evaluate whether an agent can complete an embedded firmware task from a realistic but controlled engineering handof: a prose description of required device behavior, a constrained implementation surface, visible tool afordances, and a target environment that can be executed repeatedly. The emphasis is on agentic capability, not only on final-source quality.

More concretely, the benchmark is intended to answer three questions:

1. Can the agent start from a simple plain-text specification and turn it into useful local verification, rather than depending on a benchmark-specific structured contract or hidden grader details?

2. Can the agent iteratively write or refine firmware and tests, build target code, run local checks, execute the controller against a target environment, and use the resulting feedback to improve the implementation?

3. Can the agent use that loop to reach a final correct submission that satisfies both nominal control requirements and safety-critical requirements such as stale-input handling, interlocks, timeout behavior, and sequence recovery?

At a higher level, the benchmark is asking whether modern coding agents can do something that is both practically important and still underexplored in embedded software: begin from a prose engineering handof, construct their own verification loop, and iterate all the way to behaviorally correct firmware rather than merely producing plausible source code. This is scientifically interesting because it moves the evaluation target from static synthesis to closedloop engineering performance, and it is practically interesting because that is the level at which embedded automation would become genuinely useful in real development workflows.

In the current implementation, these questions are instantiated on an ESP32 firmware target running in QEMU [9] and coupled to deterministic plant models. This concrete stack is important because it enables reproducible, controlled comparisons across models. However, the primary scientific contribution is the benchmark methodology itself: a prose-first, closed-loop evaluation framework that can be transferred to other embedded targets and simulation environments.

## 3.2 Benchmark workflow and contract

The primary workflow evaluated by this benchmark is the realistic\_self\_verify setting, in which the agent receives a prose engineering brief, operates inside a constrained workspace, and uses visible local tooling to verify its work before deciding whether to submit to the hidden evaluator. In this primary condition, the hidden evaluator is not part of the local repair loop: the agent iterates on the basis of visible evidence, not hidden failure messages. Figure 1 summarizes this workflow. The one-shot and richer-feedback settings described later in Section 3.5 are useful comparison conditions that will also be explored, but this self-verification-centered workflow is the main target of the benchmark.

Therefore, the workflow induces a simple contract. For each task:

1. The agent receives a plain-text specification file describing the required behavior.

2. The agent can edit only the intended implementation surface.

3. The agent can repeatedly use visible build, test, and runtime tooling to decide whether to keep iterating locally or to submit.

4. The hidden evaluator is consulted only upon submission; in the primary workflow it does not provide detailed failure feedback to drive the local loop.

![](images/0584014dca603aa63c0b9a2facc603a50fff077372cb6f921981abdf5d62fa07.jpg)  
Figure 1: Main agentic workflow evaluated by the benchmark. The agent iterates locally on visible evidence, explicitly decides whether to submit, and only then triggers a hidden behavioral evaluation.

5. The task is solved only when a submitted implementation satisfies the full behavioral contract.

This contract is deliberate. The goal is to measure whether an agent can start from a natural engineering handof and iterate until the device behavior meets that handof. The benchmark is therefore prose-first by default: the internal structured task definition remains hidden, and familiarity with benchmark-specific schemas is not supposed to be the main source of success.

## 3.3 Design principles and construction process

The workflow above describes what the agent experiences during evaluation. We next explain how the benchmark itself is built so that dificulty comes from the embedded control problem rather than from accidental benchmark artifacts. The following design principles guided both task design and the construction of the evaluator, simulators, and public task packets.

The benchmark follows five design principles.

1. Closed-loop first: The benchmark should evaluate iterative development, not just final code emission.

2. Contract alignment: Hidden checks must trace back to visible requirements in the prose handof. If a hidden assertion matters, it should be stated in the public task packet; otherwise it should not be graded.

3. Task-specific failures: If an agent fails, the failure should be attributable to the control problem, such as a wrong state transition, stale-sensor policy, or timeout rule, rather than to benchmark setup errors such as incorrect build instructions in the task handof.

4. Visible self-verification afordances: Realistic self-verification requires visible tooling, test locations, and runtime hooks; merely telling the agent to verify itself is not enough.

5. Stack portability: The benchmark design should remain conceptually independent of the current microcontroller and simulator choices.

These principles are aimed at benchmark validity. A failure should be interpretable as a failure on the embedded-development problem, not on hidden benchmark trivia. Likewise, a success should reflect genuine convergence in a closed loop rather than accidental matching against an overexposed grading contract.

We operationalize these principles through a fixed construction pipeline for every task in the suite so that task dificulty comes from the embedded control problem, not from accidental benchmark artifacts. The process begins with task design: first, an engineering-level control plan is written so that it specifies intended behavior, safety constraints, of-nominal cases, and acceptance criteria. This plan is the source for both the public handof and the hidden evaluator, which keeps the benchmark contract aligned across visible and private components.

The next stage is simulation and execution-harness construction. For each task, we implement a deterministic process simulator and connect it to the firmware target so that the controller is evaluated in a true closed loop. As previously mentioned, in the current reference stack, the firmware runs on an ESP32 target in QEMU, and the process simulator exchanges sensor and actuator signals with that target through UART. We use simulation first because it is safer, repeatable, and practical for large comparative sweeps, while still preserving the core embedded workflow of sensing, control, actuation, and runtime feedback.

After the simulator is stable, we write hidden ground-truth tests for grading. These tests are derived from the same requirements and cover nominal trajectories, boundary conditions, and safety-critical behaviors such as stale-input handling, timeout policies, and interlocks. The hidden suite defines final pass/fail status and remains private so that models cannot overfit to evaluator internals. Hidden-evaluator feedback is returned only in the two feedback-rich evaluation scenarios to assess whether this extra information helps the agents converge better and faster.

In parallel, we prepare the agent-facing task packet. The agent receives a plain-text naturallanguage handof, not a machine-readable contract and not the hidden tests. The workspace exposes the required tooling for build, local verification, and runtime probing, but editing is constrained to the intended implementation surface. This constraint is enforced by the harness so that comparisons remain fair and focused on the designated firmware-development scope.

During evaluation, the agent must turn the prose brief into executable verification and implementation steps: writing or refining tests, modifying firmware, running builds, interpreting failures, and iterating in simulation. The harness records staged outcomes such as logic-test failure, build failure, runtime failure, or behavioral failure, and final success is determined only by the hidden ground-truth evaluator. This design simultaneously measures agentic development behavior and end-to-end behavioral correctness.

## 3.4 Task suite

The current task suite contains five tasks arranged as a deliberate dificulty ladder. Each task starts from a plain-text specification of desired behavior and requires the agent to converge to a correct controller through iterative validation. Dificulty is intended to come from the control problem itself: additional sensing channels, stronger coupling between decisions, explicit safe-state behavior, and eventually a full sequence-control problem. The defined tasks are presented below:

Table 1: Compact view of the five-task dificulty progression.
<table><tr><td>Tier</td><td>Task</td><td>Primary added complexity</td></tr><tr><td>1</td><td>tank_fill_drain</td><td>Baseline threshold control with timeout-safe behavior.</td></tr><tr><td>2</td><td>thermal_chamber_hysteresis</td><td>Adds plant dynamics and anticipatory control near the upper band.</td></tr><tr><td>3</td><td>pressure_vessel_interlock</td><td>Adds asynchronous input freshness and explicit interlock safety constraints.</td></tr><tr><td>4</td><td>mixing_tank_fill_heat</td><td>Adds multi-sensor coupling and phase-dependent control decisions.</td></tr><tr><td>5</td><td>filter_tank_sequence</td><td>Adds timer-driven sequence control, evidence accumulation, and recovery behavior.</td></tr></table>

1. tank\_fill\_drain: A baseline single-sensor task with one level stream and one pump actuator. The task brief asks for simple threshold control, but not stateless threshold control: the controller must turn the pump on below 30, turn it of above 80, and preserve the previous safe output inside the 30–80 band rather than forcing a new command in the middle region. It must also distinguish normal short idle gaps from true timeout behavior and fail safe to pump-of on malformed input or stale data beyond 1000 ms. This task therefore establishes the basic pattern of prose specification, exact UART-visible actuation, timeout handling, and runtime self-verification.

2. thermal\_chamber\_hysteresis: A single-loop temperature-control task with a laggy plant and binary heater actuation. The controller must heat below 47, turn of above 52, preserve the previous safe state inside the nominal band, and perform anticipatory shutof when temperature is rising in the 50–52 region. The prose brief explicitly treats that anticipatory heater-of as a valid holding behavior rather than a fault, so the agent must reason about trend, not only instantaneous thresholding. The task also includes malformed-input and stale-sensor fail-safe behavior plus a closed-loop hard ceiling at 56 °C, which means a controller can satisfy the visible threshold logic and still fail if it ignores plant lag.

3. pressure\_vessel\_interlock: A dual-sensor interlock task combining pressure regulation with door safety logic. Pressure freshness and door freshness are independent, and the controller must remain in the safe vent-open state until both signals have been observed validly. Once initialized, it must pressurize below 40 when sealed, relieve above 60, preserve the previous safe output inside the band, and immediately yield to door-open dominance. The task also requires that the controller never command compressor-on and vent-open simultaneously, making it the first task in the ladder where explicit actuator mutual exclusion and asynchronous input validity both matter.

4. mixing\_tank\_fill\_heat: A coupled level-and-temperature task with independent sensor freshness and a low-fill heater guard. The controller must open the inlet below 60, close it above 75, keep the heater of whenever level falls below 55, and only heat in the safe fill window when temperature is still below the working band. Near the upper temperature edge, it must again reason about temperature trend and switch the heater of early when temperature is rising. The task therefore combines multi-sensor validity, coupled decisions across two actuators, and a closed-loop hard ceiling at 51 °C.

5. filter\_tank\_sequence: A sequence-control task with two sensors, two actuators, and explicit stage transitions for filtering, settling, draining, and completion. The controller must count a clear streak using valid turbidity inputs only, enter settling after three clear samples while level remains high enough, wait at least 400 ms before opening the drain, and then recover correctly from disturbances that arrive during settling or draining by restarting filtering and resetting progress. It must also detect completion when the remaining level drops below the minimum threshold and return to the all-of state. This is the strongest task in the current ladder because correctness depends on state progression, timing, evidence accumulation, and disturbance recovery rather than on a single local threshold rule.

Taken together, these five tasks are intended to form a deliberate ladder of increasing embedded control dificulty, as summarized in Table 1. The progression starts with a simple single-sensor threshold controller, then adds plant lag and anticipatory control, asynchronous multi-sensor interlocks, coupled multi-actuator decisions, and finally timed sequence control with explicit disturbance recovery. This structure is useful because it helps separate models not only by raw coding ability, but also by their ability to maintain controller state, reason over asynchronous observations, and build stronger self-verification loops as the control problem becomes richer.

## 3.5 Evaluation scenarios

Although the primary research target is the setting in which the agent must build and trust its own verification loop, we report three additional scenarios to contextualize performance. Concretely, we keep one simpler one-shot baseline and two guidance-oriented settings so that results can be interpreted across a controlled range of feedback assumptions. The four evaluation scenarios are presented below, ordered by increasing level of feedback:

1. oneshot\_blind: The agent is restricted to a single submission, no hidden feedback is returned, and the public self-verification helpers are intentionally omitted. This removes iterative search almost entirely and provides a lower-bound baseline for direct specificationto-code performance. In other words, it asks what happens when the model is forced to rely almost entirely on its first-pass interpretation of the task packet.

2. realistic\_self\_verify: This is the primary benchmark condition. The agent may iterate freely, but hidden grader details are not returned after submission. The agent must therefore rely on the visible build, test, and runtime surface exposed in the workspace and on whatever self-authored checks it creates under that surface. This is the closest match to the scientific question of the paper: can the agent read a prose engineering brief, construct its own verification loop, and decide for itself when the solution is ready to submit?

3. ci\_red\_green: The agent may iterate and resubmit, but after each submission it sees only a compact hidden pass/fail outcome. This approximates a CI-style search signal: the model learns whether the current attempt is acceptable, but not which hidden assertion failed or what concrete repair is needed. It is therefore useful for separating models that benefit from minimal external search guidance from models that can already verify themselves locally.

4. oracle\_full: The agent may iterate and receives detailed hidden host-test or integration feedback after submission. This is intentionally more guidance-heavy than the intended autonomous-use case, but it is a useful diagnostic upper bound on repair under rich supervision. If a model succeeds only here, then the benchmark is showing that it can repair when pointed toward the error, but not necessarily that it can construct a trustworthy self-verification loop on its own.

These four scenarios should therefore not be read as a single monotone leaderboard. They are complementary lenses over the same tasks. realistic\_self\_verify measures the core target capability, oneshot\_blind isolates first-pass implementation skill, and ci\_red\_green plus oracle\_full show how strongly each model depends on progressively richer hidden guidance. Together, they make it possible to attribute gains to diferent mechanisms: specification understanding, self-verification quality, and dependence on external feedback.

The scenario axis also provides a targeted ablation of the two design principles that are behavioral rather than benchmark-validity constraints: closed-loop iteration and visible selfverification. In particular, the contrast between oneshot\_blind and realistic\_self\_verify holds the task prose, hidden evaluator, tool harness, model set, and run budget fixed while changing whether the agent can use the visible build, test, and runtime surface for iterative selfverification before submission. By contrast, contract alignment, task-specific failure attribution, and stack portability are benchmark-hygiene constraints enforced by construction: violating them would introduce hidden unstated requirements, harness artifacts, or stack-specific assumptions rather than isolate useful agent behavior.

## 3.6 Implementation details

All code and data used to replicate the results of this paper are publicly available at https: //github.com/jgcarrasco/closed\_loop\_evaluation\_agents\_embedded. The present implementation instantiates the benchmark on an ESP32 firmware target executed in QEMU and coupled to deterministic plant models implemented in Python. ESP-IDF provides the firmware build system and target support [24], QEMU provides a reproducible execution substrate for the embedded target [9], and the open-source pi coding-agent harness handles model interaction and tool mediation [25]. For the local open-weight experiments, Docker provides reproducible environment packaging [26], and llama.cpp serves the local models under a common inference stack [27]. Each task has a task-specific runtime module under sim/tasks/ that defines the hidden scenarios, advances the plant state, emits sensor frames, consumes actuator commands from the firmware, and evaluates whether the resulting closed-loop behavior satisfied the scenario contract. This keeps the benchmark logic explicit and inspectable for humans while keeping the hidden evaluator itself out of the agent-visible workspace.

The common runtime harness launches the firmware in QEMU, exposes UART0 through a local TCP socket, and communicates with the firmware over newline-delimited ASCII frames. On the plant-to-firmware side, the runtime sends task-specific sensor frames such as SENSE LEVEL 25, SENSE TEMP 47, or SENSE DOOR OPEN, plus explicit fault frames such as FAULT SENSOR\_TIMEOUT. On the firmware-to-plant side, the harness expects exact task-specific actuator lines such as ACT PUMP ON, ACT HEATER OFF, or ACT DRAIN OPEN. This UART boundary is a simulation surrogate for real embedded I/O, where the same controller logic would more typically read GPIO/ADC/busconnected sensors and drive actuators through GPIO, PWM, relays, or peripheral interfaces. The benchmark intentionally abstracts that board-specific electrical integration layer so that the evaluation stays focused on closed-loop control behavior, staged verification, and repair rather than on one particular hardware wiring stack. This exact-string contract is therefore part of the benchmark: hidden checks do not grade internal helper names or refactors, but they do grade the externally visible control behavior and UART-visible actuation surface.

We use four outcome labels when summarizing hidden-evaluation results. PASS means that the submitted implementation satisfies all hidden host and closed-loop checks. HOST denotes a hidden host-test failure: the submission reaches the hidden evaluator, but fails host-side build or unit tests before the QEMU closed-loop plant scenarios are accepted. INT denotes an integration failure: the submission passes the preceding host-test stage but fails during the QEMU-based closed-loop evaluation, for example because the firmware emits the wrong UART-visible actuator command, mishandles timing or stale inputs, violates a safety constraint, or fails a task-specific plant scenario. NONE means that the run ended without a final submission.

The plant models are deterministic state machines rather than stochastic simulators. For the simplest tank\_fill\_drain task, for example, the plant state contains only the current level, the pump state, and simulated time in milliseconds. Each plant step advances time by a fixed tick and updates the level according to a simple fill-minus-drain rule. More complex tasks add lag, coupled sensors, multi-actuator dynamics, or explicit sequence state, but they follow the same principle: identical inputs should produce identical traces so that runs remain reproducible across models and across repeated evaluations. During each hidden scenario, the runtime records a full transcript of UART exchanges plus structured trace samples and summary metrics, which makes failures diagnosable without revealing the hidden checks to the agent.

## Illustrative example

The easiest task, tank\_fill\_drain, is a good concrete example of the benchmark contract. The public task brief tells the agent that the controller must turn the pump on when LEVEL<30, turn it of when LEVEL>80, preserve the previous safe output in the mid-band, and fail safe to pump-of on malformed input or timeout. It also explicitly instructs the agent to create its own visible tests and at least one firmware-level runtime probe before submission. The hidden host checks then validate whether those public requirements were actually met. Representative hidden checks for this task include preserving the current safe ON state in the mid-band and forcing pump-of after the bounded timeout window.

This example illustrates what the agent must actually do in the benchmark. It is not enough to emit plausible controller code. The agent must edit the constrained firmware surface, write or refine local tests, build the firmware, and ideally exercise it end to end through the UART-visible runtime path before deciding to submit. That distinction matters because an agent can easily produce false greens with incomplete local tests. For example, it may validate the obvious low-threshold and high-threshold behavior while missing the mid-band latch or the exact timeout reaction. Hidden tests and hidden closed-loop scenarios are therefore necessary to distinguish genuine behavioral correctness from a locally convincing but incomplete self-verification story.

## 4 Evaluation

We evaluate whether current coding agents can turn prose task descriptions into behaviorally correct embedded firmware under repeated closed-loop interaction. The study spans seven models across five tasks and four feedback scenarios, repeated three times for a total of 420 runs. The model set includes two hosted frontier systems (gpt-5.4 and gpt-5.4-mini) and five local open-weight Qwen variants (qwen3.5-27B, qwen3.5-35B-A3B, qwen3.5-9B, qwen3.5-4B, and qwen3.5-2B).

All model evaluations reported in this paper are executed through the same pi harness [25]. This is deliberate: keeping the agent wrapper, tool interface, prompt surface, and run logging fixed reduces the risk of attributing scafold diferences to model diferences. We chose this model set to cover a useful spread of deployment settings, model scales, and architectural forms, so that the evaluation captures both cross-family capability diferences and within-family local scaling behavior.

The hosted GPT models are included as frontier-scale reference points rather than as an exhaustive hosted-model leaderboard. The primary purpose of the model set is to evaluate closed-loop embedded-agent behavior across the selected configurations, with particular attention to whether locally served open-weight models can solve the tasks under the same harness. We therefore scope model-ranking claims to the evaluated configurations. Additional hosted baselines, including Claude Opus, would be valuable future work but are outside the controlled model set studied here.

We report the results in three complementary analyses:

1. First, we treat realistic\_self\_verify as the primary condition because it most closely matches the autonomous closed-loop workflow the benchmark is intended to measure.

2. Second, we use the full four-scenario matrix to study how feedback visibility changes model behavior.

3. Third, we focus on the local Qwen variants and study the trade-ofs between model size, performance, and computational cost.

Regarding the local Qwen models, they were all executed on a single NVIDIA RTX 4090 using llama.cpp. The evaluated family includes qwen3.5-27B, qwen3.5-35B-A3B, qwen3.5-9B, qwen3.5-4B, and qwen3.5-2B. We intentionally kept this study within a common GGUF 4-bit deployment regime, rather than mixing quantized and higher-precision checkpoints, even though a smaller model such as 9B could fit at higher precision on the same hardware. All local Qwen runs were executed with thinking disabled. This keeps numeric precision, serving format, and explicit test-time reasoning mode from becoming additional moving variables in the local scaling study so that the within-family comparison remains methodologically controlled. The local GGUF checkpoints were obtained from Unsloth’s published Qwen3.5 releases and deployment guide [28], which provide a consistent public source for the family-wide local models. We did not evaluate models larger than the 35B version because of hardware constraints. We also omitted the 0.8B model because the 2B model finished with zero passes across all of its runs, recorded a first submission in only eight of them in the final curated bundle, and otherwise spent most of its budget in nonproductive search. On that basis, a 0.8B run was unlikely to add useful discriminative evidence beyond extending the same failure regime.

Across all runs we report hidden outcome, failure family, stage reached, total tokens, tool calls, hidden-evaluation count, build attempts, self-test runs, and runtime-probe behavior. Runs were allowed to iterate until success or a one-hour wall-clock limit; oneshot\_blind and realistic\_self\_verify remained restricted to a single submission, whereas the other scenarios allowed repeated submissions within that same time budget. Because the plant and harness are deterministic, the three repetitions are intended to measure agent/model stochasticity under a fixed environment rather than randomized task variation. This repetition count is intended to expose major behavioral diferences and recurring failure modes, not to resolve small diferences between adjacent model configurations. In each per-model/per-scenario analysis, the efective binary sample size is 15 outcomes, from five tasks and three repetitions. A normal-approximation two-proportion power calculation at two-sided α = 0.05 and 80% power gives minimum detectable absolute pass-rate diferences of approximately 0.46, 0.49, 0.49, and 0.44 for baseline pass rates of 0.1, 0.2, 0.3, and 0.5, respectively; at a 0.7 baseline, even a perfect 1.0 comparator remains below 80% power under this approximation. We therefore treat close model orderings as descriptive trends, while emphasizing larger efect-size patterns across feedback regimes, failure classes, and local-versus-hosted configurations. Because the OpenAI models run on hosted backends whereas the Qwen models are served locally, wall-clock is not used as a fairness-critical metric in cross-backend comparisons. It is used only in the local-only subsection, where all models share the same hardware and serving stack.

Table 2 summarizes the inference and harness configuration used in the evaluation. For hosted GPT models, we used the GitHub Copilot subscription access path to reflect a realistic developerfacing workflow; in that setting, sampling parameters are provider-managed and temperature/topp controls are not exposed to the harness. For local Qwen models, the llama.cpp server was configured following the non-thinking Qwen3.5 guidance from Unsloth [28]. Submission limits were scenario-specific: one submission for oneshot\_blind and realistic\_self\_verify, and repeated submissions within the wall-clock budget for ci\_red\_green and oracle\_full. At the benchmark-harness level, we imposed no separate tool-call, message-turn, or iteration-count cap beyond the one-hour wall-clock budget and the scenario-specific submission budget.

Table 2: Inference and harness configuration used for the evaluated models.
<table><tr><td>Parameter</td><td>Hosted GPT models</td><td>Local Qwen3.5 models</td></tr><tr><td>Runtime</td><td>GitHub Copilot subscription via pi</td><td>1lama.cpp via pi</td></tr><tr><td>Temperature</td><td>Provider-managed</td><td>0.7</td></tr><tr><td>Top-p</td><td>Provider-managed</td><td>0.8</td></tr><tr><td>Top-k</td><td>Provider-managed</td><td>20</td></tr><tr><td>Min-p</td><td>Provider-managed</td><td>0.0</td></tr><tr><td>Presence penalty</td><td>Provider-managed</td><td>1.5</td></tr><tr><td>Repeat penalty</td><td>Provider-managed</td><td>1.0</td></tr><tr><td>Reasoning/thinking</td><td>Provider-managed</td><td>Disabled</td></tr><tr><td>Run budget</td><td>One hour per run</td><td>One hour per run</td></tr><tr><td>Submission budget</td><td>Scenario-specific</td><td>Scenario-specific</td></tr></table>

Table 3: Pooled primary-scenario results for realistic\_self\_verify over three repetitions. For all listed models, this corresponds to 15 runs. The pass column reports pooled counts, pass rate, and Wilson 95% confidence interval. Cross-backend comparison emphasizes interaction cost rather than wall-clock.
<table><tr><td>Model</td><td>Passes</td><td>Avg total tokens</td><td>Avg tool calls</td><td>Avg hidden evals</td><td>Avg self-test runs</td></tr><tr><td>gpt-5.4</td><td>14/15 (93.3%) [70.2, 98.8]</td><td>0.42M</td><td>35.5</td><td>1.0</td><td>2.6</td></tr><tr><td>gpt-5.4-mini</td><td>8/15 (53.3%) [30.1, 75.2]</td><td>1.03M</td><td>54.7</td><td>1.1</td><td>4.4</td></tr><tr><td>qwen3.5-27B</td><td>9/15 (60.0%) [35.7, 80.2]</td><td>6.90M</td><td>109.0</td><td>1.5</td><td>12.2</td></tr><tr><td>qwen3.5-35B-A3B</td><td>4/15 (26.7%) [10.9, 52.0]</td><td>4.40M</td><td>92.5</td><td>1.5</td><td>16.1</td></tr><tr><td>qwen3.5-9B</td><td>2/15 (13.3%) [3.7, 37.9]</td><td>6.76M</td><td>113.8</td><td>6.6</td><td>38.5</td></tr><tr><td>qwen3.5-4B</td><td>2/15 (13.3%) [3.7, 37.9]</td><td>6.14M</td><td>114.9</td><td>12.7</td><td>37.7</td></tr><tr><td>qwen3.5-2B</td><td>0/15 (0.0%) [0.0, 20.4]</td><td>77.39M</td><td>987.5</td><td>0.0</td><td>4.2</td></tr></table>

## 4.1 Primary Results: Realistic Self-Verification

The benchmark’s main scientific target is the realistic\_self\_verify condition, in which the agent must construct and trust its own local evidence without hidden failure messages. This is therefore the primary comparison for interpreting model capability.

Table 3 shows the clearest descriptive ordering in the paper. gpt-5.4 leads the primary condition with 14/15 passes (93.3%). Among the non-frontier models, the local qwen3.5-27B posts the highest pooled count at 9/15 (60.0%), narrowly ahead of gpt-5.4-mini at 8/15 (53.3%). The remaining local models are substantially weaker in this primary condition: qwen3.5-35B-A3B reaches 4/15 (26.7%), qwen3.5-9B and qwen3.5-4B each reach 2/15 (13.3%), and qwen3.5-2B remains at 0/15 (0%).

This makes two points immediately visible. First, the benchmark is not frontier-saturated: even the strongest model leaves one miss in the primary condition, so the current five-task suite still produces visible headroom at the top end. Second, local agents are not merely toy baselines: the 27B local Qwen model posts a slightly higher pooled pass count than the smaller hosted frontier model in the benchmark’s primary setting, although the overlapping Wilson intervals show that this gap should not be over-read as a stable separation. That competitiveness comes at a substantial search cost, however. Relative to gpt-5.4-mini, qwen3.5-27B uses roughly 6.7× as many tokens and about 2× as many tool calls per primary-condition run to obtain its additional pass. Token totals are the provider/server usage totals recorded by the pi harness, including reported prompt/input, completion/output, and cache tokens where available. Because tokenization and reasoning-token accounting are backend-dependent, we interpret token ratios as reported interaction volume rather than provider-independent compute; tool-call ratios are the cleaner cross-backend behavioral signal.

![](images/6fe819a920f415ba2d3a84d69f4be56c3ce77cccd32e33bc27cfe7aea213ad70.jpg)  
Figure 2: Two-panel view of the primary realistic\_self\_verify condition over three repetitions. The left panel shows pass counts out of three repetitions for each model-task pair. The right panel shows the most likely outcome across those repetitions: PASS if pass is the modal outcome, HOST for host-test failure, INT for integration failure, and NONE for no submission.

The left-hand panel in Figure 2 reports pass counts out of three repetitions for each model-task pair, so it makes the replicated task-by-task separation explicit rather than relying only on the aggregate totals in the previous Table 3.

Its main result is that the benchmark’s observed ordering is carried by a small number of sharply discriminative tasks. filter\_tank\_sequence is the strongest separator: only gpt-5.4 passes it in the primary condition, and even then only in two of the three repetitions. At the other extreme, tank\_fill\_drain is the anchor task: the two hosted models plus qwen3.5-27B and qwen3.5-35B-A3B solve it in every repetition, and even the weaker local models still manage it at least twice. The middle three tasks create the real spread. qwen3.5-27B is the only local model that passes mixing\_tank\_fill\_heat, and it is also the strongest local model on thermal\_chamber\_hysteresis and pressure\_vessel\_interlock.

The right-hand panel reports the modal outcome for each cell across the three repetitions, using the outcome labels defined in Section 3.6 to distinguish stable passes from the most common failure regime: host-test failure, integration failure, or no submission.

That outcome view shows that the failure regimes are not uniform across models. gpt-5.4 is still overwhelmingly pass-dominated, with only one integration-style miss on filter\_tan k\_sequence. gpt-5.4-mini passes tank\_fill\_drain, thermal\_chamber\_hysteresis, and pressure\_vessel\_interlock in most repetitions, but mixing\_tank\_fill\_heat becomes a consistent integration failure and filter\_tank\_sequence a consistent host-test failure. For qwen3.5-27B, the losses are concentrated on the two hardest tasks: tank\_fill\_drain and thermal\_chamber\_hysteresis remain stable passes, pressure\_vessel\_interlock passes in two of three repetitions, while mixing\_tank\_fill\_heat is usually a host-test failure and filter\_tank\_sequence never passes.

Below 27B, the weaker local models mostly fail after producing candidate implementations rather than by failing to engage at all. For qwen3.5-35B-A3B, qwen3.5-9B, and qwen3.5-4B, host-test failure is the typical outcome on thermal\_chamber\_hysteresis, pressure\_vesse l\_interlock, and filter\_tank\_sequence, while mixing\_tank\_fill\_heat is the task where integration-style failures become common. The smallest model, qwen3.5-2B, sits in a diferent regime altogether: it never passes and its modal outcome is no submission on every task.

![](images/541af2b49dd05127bb83256853592cbbec8571a8476d92eedadac8c0262c1d84.jpg)  
Figure 3: Pass-rate sensitivity to feedback visibility across the evaluated model set, pooled over three repetitions. Each bar is a percentage over the 15 runs available for that model and scenario, and error bars show Wilson 95% confidence intervals.

Table 4: Exact pooled pass counts for the full four-scenario matrix over three repetitions. Each cell reports the number of passes out of 15 runs for that model and scenario.
<table><tr><td>Model</td><td>oneshot_blind</td><td>realistic_self_verify</td><td>ci_red_green</td><td>oracle_full</td></tr><tr><td>gpt-5.4</td><td>13/15</td><td>14/15</td><td>15/15</td><td>14/15</td></tr><tr><td>gpt-5.4-mini</td><td>10/15</td><td>8/15</td><td>12/15</td><td>11/15</td></tr><tr><td>qwen3.5-27B</td><td>8/15</td><td>9/15</td><td>9/15</td><td>12/15</td></tr><tr><td>qwen3.5-35B-A3B</td><td>4/15</td><td>4/15</td><td>8/15</td><td>10/15</td></tr><tr><td>qwen3.5-9B</td><td>2/15</td><td>2/15</td><td>3/15</td><td>7/15</td></tr><tr><td>qwen3.5-4B</td><td>0/15</td><td>2/15</td><td>1/15</td><td>2/15</td></tr><tr><td>qwen3.5-2B</td><td>0/15</td><td>0/15</td><td>0/15</td><td>0/15</td></tr></table>

Taken together, the two panels suggest that the main separation below qwen3.5-27B is not just lower search efectiveness. The harder tasks expose a more basic weakness in producing code that survives local verification, and at 2B that weakness appears one step earlier as failure to produce viable submissions at all.

## 4.2 Efect of Feedback Visibility

The full four-scenario matrix is useful because it reveals whether a model’s limitation is direct implementation quality, self-verification quality, or dependence on external search signal. Figure 3 summarizes pass rates by scenario for the evaluated model set.

Also, the exact pooled counts behind Figure 3 are shown in Table 4, which lists the full four-scenario matrix directly as x/15 values.

From these results, three patterns stand out: First, gpt-5.4 shows high but not perfect stability across feedback regimes: it reaches 13/15 (86.7%) in oneshot\_blind, 14/15 (93.3%) in realistic\_self\_verify, 15/15 (100%) in ci\_red\_green, and 14/15 (93.3%) in oracle\_full. The important point is therefore relative stability rather than saturation: it is the strongest model in every scenario.

Second, qwen3.5-27B is unusually strong in the benchmark’s primary setting, not only in the rich-feedback settings. It reaches 9/15 (60.0%) in realistic\_self\_verify and 12/15 (80.0%) in oracle\_full, versus 8/15 (53.3%) in one-shot and 9/15 (60.0%) in CI-style red/green search. That is useful because it shows that the model is not simply depending on external hidden guidance to recover performance.

Table 5: Pooled local-only Qwen comparison on shared hardware across three repetitions. Pass denominators are 60 for every model. The pass column reports pooled counts, pass rate, and Wilson 95% confidence interval. The n submit column reports how many runs reached at least one submission; average first-submit time is conditioned on those runs.
<table><tr><td>Model</td><td>Form</td><td>Passes</td><td>Avg wall-clock</td><td>n submit</td><td>Avg 1st submit</td><td>Avg total tokens</td><td>Avg tool calls</td></tr><tr><td>qwen3.5-27B</td><td>Dense</td><td>38/60 (63.3%) [50.7, 74.4]</td><td>26.2 min</td><td>60/60</td><td>18.0 min</td><td>4.08M</td><td>77.0</td></tr><tr><td>qwen3.5-35B-A3 B</td><td>MoE</td><td>26/60 (43.3%) [31.6, 55.9]</td><td>16.9 min</td><td>60/60</td><td>11.9 min</td><td>4.19M</td><td>80.1</td></tr><tr><td>qwen3.5-9B</td><td>Dense</td><td>14/60 (23.3%) [14.4, 35.4]</td><td>25.3 min</td><td>56/60</td><td>17.6 min</td><td>6.09M</td><td>103.5</td></tr><tr><td>qwen3.5-4B</td><td>Dense</td><td>5/60 (8.3%) [3.6, 18.1]</td><td>21.4 min</td><td>50/60</td><td>9.4 min</td><td>11.55M</td><td>179.2</td></tr><tr><td>qwen3.5-2B</td><td>Dense</td><td>0/60 (0.0%) [0.0, 6.0]</td><td>29.7 min</td><td>8/60</td><td>1.3 min</td><td>54.95M</td><td>716.3</td></tr></table>

Third, the weaker local models are much more feedback-sensitive. The clearest example is the 35B-A3B MoE model: it reaches only 4/15 in both one-shot and realistic self-verification, rises to 8/15 under CI-style red/green feedback, and reaches 10/15 under oracle feedback. This suggests that the model can often repair behavior once hidden feedback is made rich enough, but it is substantially weaker at constructing its own local verification loop. The 9B and 4B models show the same qualitative trend, although at a lower overall capability level. The 2B model does not respond meaningfully even to richer feedback: it remains flat at zero across all 60 runs.

The hosted smaller frontier model behaves diferently again. gpt-5.4-mini peaks in the CI-style condition at 12/15 rather than in oracle or realistic self-verification. That is exactly the kind of scenario-specific behavior the benchmark is supposed to reveal. The scenario axis is therefore not just an “easier versus harder” dial. It is a diagnostic lens on how models use, or fail to use, diferent forms of feedback.

## 4.3 Local Models on Shared Hardware

The local-only analysis is where wall-clock becomes meaningful. All local Qwen runs in this study use the same pi harness, the same task packets, the same QEMU-based benchmark environment, the same llama.cpp serving stack, and the same NVIDIA RTX 4090. We also deliberately kept the study within one GGUF 4-bit deployment regime, rather than mixing quantized checkpoints with higher-precision variants. Even though a 9B model could fit at higher precision on this hardware, we kept it quantized so that model size and architecture remain the primary varying factors in the comparison.

Table 5 shows that the best local operating point is the 27B dense model. It is the most accurate local model at 38/60 (63.3%), and it is slightly more interaction-eficient than the 35B-A3B MoE model. The 35B-A3B MoE model is the main counterexample. It reaches 26/60 (43.3%), submits earlier on the same hardware, but is less accurate and slightly more token-hungry on average. In this benchmark, the 35B-A3B model therefore looks like a throughput advantage rather than a capability advantage. A plausible explanation is that, despite the 35B model being larger, it is a Mixture-of-Experts (MoE) model [29] in which only about 3B parameters are

![](images/3c762ce5532c0b64a319cd7baf9c58c667e4df1050e18715a268f6acf96f0935.jpg)

![](images/03f4a5c01aebdc7f75ce96e98d57eb06184f48a2c42aee73ffe7afa43b8a52ff.jpg)  
Figure 4: Shared-hardware local-model tradeofs shown as two side-by-side views. Both panels use pass rate rather than raw pass count so that accuracy and resource demand share a common scale. The left panel relates pass rate to average first-submission latency; the right panel relates pass rate to average total token demand.

simultaneously active during inference.

The 2B model sharpens the lower boundary of the family even more. Despite being the smallest checkpoint, it is not the cheapest operating point in closed-loop use: averaged over its 60 pooled runs it consumes 54.95M tokens, 716.3 tool calls, and 29.7 minutes per run while producing zero passes. Its reported 1.3-minute time to first submission should not be read as an eficiency signal, because only eight of its 60 runs recorded a first submission in the final curated bundle.

Figure 4 makes the same ranking visible visually in a way that separates the two resource axes. The left panel shows that the faster first-submission models are not necessarily the better ones: the 35B-A3B model submits earlier than 27B, but converts less of its available runs into success, while 2B submits almost immediately in the few runs that submit at all yet never converts that into success. The right panel isolates the token story even more clearly. As model scale decreases below the practical operating range, token demand rises sharply rather than falling: 4B is already far to the right of 27B and 35B-A3B, and 2B remains an extreme outlier. In other words, the failure mode at the lower end of this family is uncontrolled search rather than simple throughput loss.

The smaller dense models still degrade sharply. The 9B model remains meaningful at 14/60 (23.3%), but it already requires far heavier search than the 27B model, averaging 103.5 tool calls per run. The 4B model is close to the lower useful boundary of the current benchmark: it reaches only 5/60 runs (8.3%), yet averages 11.55M tokens and 179.2 tool calls. Its shorter average time to first submission is not evidence of eficiency; it is largely evidence of premature submission followed by repeated failure-driven search. The 2B model lies beyond that boundary: across 60 runs, it reaches zero passes, never solves any task in any scenario, and is dominated by long no-submission loops rather than productive repair.

The task-wise pattern points in the same direction. filter\_tank\_sequence remains the clearest local separator in the suite: across the twelve local runs per model-task pair, it is solved only once by qwen3.5-27B and never by the smaller local models. tank\_fill\_drain, by contrast, remains the easiest local task and is solved at least once by every local model down to 4B. The most deployment-relevant comparison remains between the 27B dense model and the 35B-A3B MoE model. The 35B model matches or exceeds the 27B model only on tank\_fill\_drain; it is weaker on thermal\_chamber\_hysteresis, pressure\_vessel\_interlock, mixing\_tank\_fill\_heat, and filter\_tank\_sequence. This is a strong indication that sparse-parameter throughput does not automatically translate into stronger autonomous embedded-agent behavior.

The lower end of the family points in the same direction even more sharply. The 2B model is 0/12 on every task. Across all 60 pooled cells, it produces no passes and records a first submission in only eight runs. We therefore treat 2B as the lower boundary of the useful local range rather than as a serious contender in the ranking, and we do not continue further downward to 0.8B in the current study.

## 5 Discussion

The evaluation results support three main interpretive claims. First, the primary realistic\_ self\_verify condition is the most informative lens for judging autonomous embedded-agent capability. Second, the feedback-visibility axis is diagnostically useful because it separates self-verification from externally guided repair. Third, the shared-hardware local study changes the deployment story by showing how model scale interacts with latency, token demand, and search behavior.

The most important model-separation pattern in the paper is the ordering observed under realistic\_self\_verify. That condition is the closest match to the intended benchmark use case, because the agent must rely on the visible build, test, and runtime surface rather than on hidden diagnostics. Under that condition, gpt-5.4 leads at 14/15 (93.3%), qwen3.5-27B posts the highest pooled count among non-frontier models at 9/15 (60.0%), gpt-5.4-mini reaches 8/15 (53.3%), the 35B/9B/4B local variants reach 4/15 (26.7%), 2/15 (13.3%), and 2/15 (13.3%) respectively, and qwen3.5-2B remains 0/15 while usually failing to submit at all.

This ordering matters as a descriptive result because it reveals a clear capability gradient under the benchmark’s primary condition. It distinguishes models that can genuinely construct and trust local evidence from models that mainly recover when external search signal is made richer. It also highlights a practically important point: local open-weight agents are already nontrivial contenders in the benchmark’s primary setting. The best local model is not at parity with the frontier model, but it is clearly beyond a toy baseline. That is exactly the kind of deployment-relevant gradient the benchmark was designed to expose.

The scenario axis is best interpreted as a diagnostic tool rather than as a single monotone leaderboard. gpt-5.4 shows strong but not perfect stability across feedback visibility and is strongest in every scenario. Below that level, models react very diferently to additional hidden feedback. gpt-5.4-mini improves most under CI-style red/green feedback. qwen3.5-27B is strongest in oracle\_full and stays competitive in realistic\_self\_verify, which suggests that its relative strength is not merely hidden-feedback dependence. By contrast, the 35B-A3B MoE model improves sharply as hidden feedback becomes richer, which indicates that it can often repair once the evaluator points it in the right direction, but struggles to build a strong self-verification loop on its own.

This is important for benchmark design. If the paper reported only one-shot generation, it would miss whether a model can actually engage in an engineering loop. If it reported only oracle-style repair, it would overstate autonomous capability. Using all four scenarios makes it possible to separate direct implementation skill, self-verification skill, and feedback dependence.

The local-only analysis adds a second practical lesson. Once hardware and serving are controlled, wall-clock becomes meaningful, and the resulting trade-ofs are not the same as the cross-backend headline. On a single RTX 4090 with llama.cpp, the best local operating point is the 27B dense model at 38/60 runs. The 35B-A3B MoE model is faster to first submit, but it is not better on this benchmark. Its behavior is consistent with a throughput advantage rather than a capability advantage.

The smaller dense models are also informative. The 9B model is a meaningful, if clearly weaker, embedded agent. The 4B model appears close to the lower useful boundary of the benchmark, because it spends very large interaction budgets for relatively few successes. The 2B result places that lower boundary more firmly still: despite being the smallest checkpoint, it is slower and far more token-hungry in practice because it falls into long no-submission loops. If a model cannot produce passes, or even regular submissions, across the suite, then it is below the current useful operating range of the benchmark. This same-hardware analysis is important because it answers a practical question that hosted-versus-local comparisons cannot: which local model family member is actually worth deploying under a fixed on-premises budget.

Our choice to keep the local study within a common 4-bit GGUF deployment regime is also part of that interpretation. We did not want a precision change to masquerade as a model-size efect. A separate higher-precision ablation, especially around the 9B scale, would be useful in future work, but it would answer a diferent question from the one addressed here.

Several benchmark-design lessons follow directly from the pooled results. First, filter\_tan k\_sequence is the main separator in the current suite and remains dificult even when strong models pass it, so it is doing the right kind of work. Second, tank\_fill\_drain functions well as a baseline anchor because even weak models can sometimes solve it.

The benchmark still has important scope limits that matter for interpretation. It studies deterministic simulated plants on a single ESP32/QEMU reference stack with UART-virtualized I/O, which is a deliberate choice for reproducibility and comparability rather than an attempt to cover the full complexity of embedded deployment. Likewise, the current suite is intentionally small and control-centric: the five tasks are enough to separate models on thresholding, hysteresis, sequencing, and interlocks, but they do not yet span the broader range of embedded workloads such as communication stacks, storage, power management, interrupt-heavy firmware, or longerhorizon mission logic. The evaluation also fixes one agent/tool surface and one common local deployment regime, so the reported rankings should be read as benchmarked results in this setting rather than as architecture-independent limits of all coding agents. However, these scope choices are also what make the study useful: they turn a noisy and dificult problem into a reproducible one, and they already produce clear capability gradients, informative failure modes, and actionable deployment trade-ofs. On that basis, we view the present paper not as a final word on embedded-agent capability, but as a strong benchmark foundation on which broader and more realistic evaluations can now be built.

## 6 Conclusions and Future Work

This paper presents a closed-loop benchmark methodology for evaluating coding agents on embedded firmware tasks, where success is defined at the level of observed device behavior rather than source-code plausibility alone. Starting from prose engineering handofs, the benchmark asks whether an agent can iteratively implement, verify, diagnose, and repair firmware under a realistic tool-mediated workflow. In that sense, the main contribution is not only a five-task suite, but an evaluation framing: prose-first, behavior-level, and explicitly centered on self-verification in a target execution environment.

The empirical study shows that this benchmark is already meaningful along several dimensions. It is executable at scale, reproducible under a deterministic simulation stack, and suficiently discriminative to separate models both by final pass rate and by how they use feedback during iterative search. Across 420 runs, the primary realistic\_self\_verify condition emerges as the most informative lens on autonomous embedded-agent capability, because it requires the agent to construct and trust its own local evidence rather than relying on hidden evaluator guidance. Under that condition, gpt-5.4 has the highest observed pass rate at 14/15 (93.3%), while qwen3.5-27B is the strongest observed local model at 9/15 (60.0%). This result is notable in practical terms: the 27B local model remains genuinely competitive in the benchmark’s primary setting despite operating in a far smaller and more resource-constrained deployment regime than the hosted frontier system.

A second conclusion is that feedback visibility is not just a convenience variable but a diagnostic one. The four-scenario design shows that models difer not only in direct implementation ability, but also in how well they construct self-verification loops and how strongly they depend on richer hidden feedback to repair failures. Some models remain comparatively stable across scenarios, whereas others improve sharply only when the evaluator exposes richer search signal. This makes the feedback axis scientifically useful in its own right: it helps separate first-pass synthesis, self-verification quality, and externally guided repair instead of collapsing them into a single aggregate score.

A third conclusion is practical. Under a common local deployment regime on shared hardware, the best local operating point in this study is the 27B dense Qwen model. The comparison with the qwen3.5-35B-A3B MoE model is especially informative: the MoE model reaches first submission faster, but the dense 27B model is clearly stronger on the benchmark overall. In this setting, the contrast is best interpreted as a trade-of between throughput and efective capability. Faster submission does not automatically translate into stronger autonomous embedded-agent performance, and sparse activation does not substitute for the ability to sustain reliable closed-loop search.

The lower end of the local family reinforces the same lesson. Smaller local models degrade not only in pass rate but also in search eficiency, with the 2B checkpoint showing that lower nominal model size does not necessarily produce a cheaper or more usable agent when closed-loop interaction is taken seriously. In this benchmark, the key deployment boundary is therefore not simply whether a model can emit plausible code, but whether it can reliably converge through repeated verification and revision without falling into unproductive search.

Taken together, these results suggest that closed-loop embedded-agent evaluation should be treated as a distinct benchmarking problem rather than as a small variation on static code generation. The main scientific gap below frontier performance is not only implementation quality, but robust autonomous self-verification: the ability to generate local evidence, interpret it correctly, and stop only when the controller is actually ready. That is precisely the capability this benchmark is designed to expose.

Regarding future work, the most important next step is to carry the same methodology from deterministic simulation into hardware-in-the-loop and then carefully controlled real-device studies. This is the main long-term target of the benchmark, even though it is substantially harder experimentally, because only real hardware can fully surface efects such as sensor jitter, communication latency, watchdog behavior, startup transients, peripheral interaction, and deployment packaging. The role of the present simulation-based study is therefore methodological staging: it establishes a reproducible baseline from which hardware-facing evaluation can proceed in a controlled way.

At the same time, research on local models is especially promising for embedded-agent workflows because it ofers direct control over both data and compute, which is highly relevant for privacy-sensitive, on-premises, and industrial settings. Future work should therefore continue to study local-model operating points, precision regimes, and agent/tool surfaces, alongside broader task extensions that include noisier and partially missing sensors, delayed or drifting signals, actuator saturation, restart-and-recovery behavior, multi-rate control loops, and richer fault-management scenarios. Together, these directions would turn the present benchmark from a single reproducible suite into a broader evaluation program for trustworthy embedded coding agents.

## Acknowledgements

This work was supported by the AgroVAL (TSI-100122-2024-10) project funded by the Recovery, Transformation and Resilience Plan from the European Union Next Generation through the Ministry for Digital Transformation and the Civil Service, the KOSMOS-UA project (PID2024-

155363OB-C43), funded by the Spanish Ministry of Science and Innovation, BALIDA-AA project (CIPROM/2024/13), funded by Conselleria de Educación, Cultura, Universidades y Empleo (Generalitat Valenciana), the IAEAV project (INREIA/2024/176) funded by the Conselleria de Innovación, Industria, Comercio y Turismo (Generalitat Valenciana), the ENIA Chair of Artificial Intelligence from the University of Alicante (TSI-100927-2023-6) funded by the Recovery, Transformation and Resilience Plan from the European Union Next Generation through the Ministry for Digital Transformation and the Civil Service, and the Grant RED2022-134656-T funded by MCIN/AEI/10.13039/501100011033.

Sergio García-Carrasco holds a predoctoral grant of the ACIF program (CIACIF/2023/019) funded by Generalitat Valenciana and the European Union’s ESF+.

## CRediT Authorship Contribution Statement

Jorge García-Carrasco: Conceptualization, methodology, software, validation, formal analysis, investigation, writing — original draft, writing — review and editing. Sergio García-Carrasco: Conceptualization, methodology, software, validation, writing — review and editing. Alejandro Maté: Supervision, project administration, writing — review and editing. Juan Trujillo: Supervision, project administration, writing — review and editing.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data Availability

The benchmark repository, task packets, runner scripts, configuration files, and curated evaluation artifacts are publicly available at https://github.com/jgcarrasco/closed\_loop\_evaluation \_agents\_embedded. The paper-facing artifact bundle is organized under artifacts/evaluati ons/paper\_closed\_loop\_eval\_420/ and includes per-run prompts, session transcripts, runner logs, and summary files for all 420 oficial runs.

## References

[1] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[2] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

[3] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

[4] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In The Twelfth International Conference on Learning Representations, 2024.

[5] OpenAI. Introducing SWE-bench verified. https://openai.com/index/introducing-swe -bench-verified/, 2024. Accessed: 2026-03-13.

[6] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R. Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793, 2024.

[7] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying LLM-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.

[8] Ruiyang Xu, Jialun Cao, Mingyuan Wu, Wenliang Zhong, Yaojie Lu, Ben He, Xianpei Han, Shing-Chi Cheung, and Le Sun. Embedagent: Benchmarking large language models in embedded system development. arXiv preprint arXiv:2506.11003, 2025.

[9] Fabrice Bellard. QEMU, a fast and portable dynamic translator. In 2005 USENIX Annual Technical Conference, pages 41–46. USENIX Association, 2005.

[10] Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. Autocoderover: Autonomous program improvement. arXiv preprint arXiv:2404.05427, 2024.

[11] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[12] Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[13] Tianyang Liu, Canwen Xu, and Julian J. McAuley. Repobench: Benchmarking repositorylevel code auto-completion systems. arXiv preprint arXiv:2306.03091, 2023.

[14] Vignesh Manjunath and Marcel Baunach. A framework for static analysis and verification of low-level rtos code. Journal of Systems Architecture, 154:103220, 2024. doi: 10.1016/j.sy sarc.2024.103220. URL https://doi.org/10.1016/j.sysarc.2024.103220.

[15] Marek Babiuch and Pavel Smutny. Benchmarking large language models for embedded systems programming in microcontroller-driven IoT applications. Future Internet, 18(2):94, 2026. doi: 10.3390/fi18020094.

[16] Hadi Ghaemi, Zakieh Alizadehsani, Amin Shahraki, and Juan M. Corchado. Transformers in source code generation: A comprehensive survey. Journal of Systems Architecture, 153: 103193, 2024. doi: 10.1016/j.sysarc.2024.103193.

[17] Mihir Agarwal, Zaqi Momin, Kailash Prasad, and Joycee Mekie. Veribench: Benchmarking large language models for verilog code generation and design synthesis. In 2025 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–5, 2025. doi: 10.1109/ ISCAS56072.2025.11044004.

[18] Mingjie Liu, Nathaniel Ross Pinckney, Brucek Khailany, and Haoxing Ren. Verilogeval: Evaluating large language models for verilog code generation. arXiv preprint arXiv:2309.07544, 2023.

[19] Vlad-Eusebiu Baciu, Johan Stiens, and Bruno da Silva. Mlino bench: A comprehensive benchmarking tool for evaluating ml models on edge devices. Journal of Systems Architecture, 155:103262, 2024. doi: 10.1016/j.sysarc.2024.103262. URL https://doi.org/10.1016/j. sysarc.2024.103262.

[20] Jung-Woo Kim, Seung-Hwan Yoon, Dong-Kyeong Kang, Seong-Won Lim, Hak-Bum Lee, Su-Min Oh, and Young-Ho Seo. A new fixed-point simulation methodology for on-device ai based on large language models. Journal of Systems Architecture, 168:103548, 2025. doi: 10.1016/j.sysarc.2025.103548.

[21] Minyu Chen, Ling-I Wu, Ruibang Liu, Xi Chang, Jianxin Xue, and Guoqiang Li. DCoL-A: Agentic dual chain of thinking helps llms pretend logic solvers. Journal of Systems Architecture, 175:103780, 2026. doi: 10.1016/j.sysarc.2026.103780.

[22] Nathan P. Koenig and Andrew Howard. Design and use paradigms for gazebo, an open-source multi-robot simulator. In 2004 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2149–2154, 2004. doi: 10.1109/IROS.2004.1389727.

[23] Yuan Wei, Yongjun Wang, Lei Zhou, Xu Zhou, and Zhiyuan Jiang. Iemu: Interrupt modeling from the logic hidden in the firmware. Journal of Systems Architecture, 154:103237, 2024. doi: 10.1016/j.sysarc.2024.103237. URL https://doi.org/10.1016/j.sysarc.2024.103237.

[24] Espressif Systems. ESP-IDF: Espressif iot development framework [software]. https: //github.com/espressif/esp-idf, 2026. Accessed: 2026-03-31.

[25] Badlogic Games. pi-mono coding-agent package [software]. https://github.com/badlo gic/pi-mono/tree/main/packages/coding-agent, 2026. Agent harness used for the evaluation. Accessed: 2026-03-31.

[26] Docker, Inc. Docker [software]. https://www.docker.com/, 2026. Accessed: 2026-03-31.

[27] ggml-org and contributors. llama.cpp: Llm inference in c/c++ [software]. https://github .com/ggml-org/llama.cpp, 2026. Accessed: 2026-03-31.

[28] Unsloth. Qwen3.5 - how to run locally guide. https://unsloth.ai/docs/models/qwen3.5, 2026. Accessed: 2026-03-26.

[29] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.