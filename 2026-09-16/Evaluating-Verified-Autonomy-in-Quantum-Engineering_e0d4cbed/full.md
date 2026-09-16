# Evaluating Verified Autonomy in Quantum Engineering

Naixu Guo<sup>∗,†</sup>,<sup>1</sup> Changhao Li<sup>∗,†</sup>,<sup>2</sup> Siyu Cheng<sup>∗</sup>,<sup>3</sup> Qicheng Tang<sup>∗</sup>,<sup>4</sup>

Binzhao Luo,<sup>5</sup> Bikun Li,<sup>6</sup> Yuxuan Du,<sup>7</sup> Shihao Ru<sup>†</sup>,<sup>8</sup> and Jiaqi Cai<sup>†5</sup>

<sup>1</sup>Centre for Quantum Technologies, National University of Singapore, 117543, Singapore <sup>2</sup>Unitary Foundation, San Francisco, CA, USA

<sup>3</sup>Department of Physics, Boston College, Chestnut Hill, Massachusetts 02467, USA

<sup>4</sup>School of Physics, Georgia Institute of Technology, Atlanta, GA 30332, USA

<sup>5</sup>GaugeForge PTE. LTD., 049422, Singapore

<sup>6</sup>Chicago Quantum Institute and Pritzker School of Molecular Engineering,

University of Chicago, Chicago, Illinois 60637, USA

<sup>7</sup>College of Computing and Data Science, Nanyang Technological University, Singapore 639798, Singapore

<sup>8</sup>School of Electrical and Electronic Engineering,

Nanyang Technological University, Singapore 639798, Singapore

(Dated: September 16, 2026)

Reliable quantum engineering is essential for turning quantum phenomena into practical technologies. As quantum platforms grow in scale and complexity, their characterization and operation require increasing human efort and coordination. Scientific artificial intelligence agents, which can plan experiments, operate instruments, and analyze observations, ofer a promising route towards autonomous quantum engineering. Yet whether current agents can perform reliably in this setting has not been systematically established. To fill this gap, we developed Quantum-Harbor, a virtual laboratory that provides a controlled execution environment for agents to interact with quantum systems. This design enables direct verification of both the actions taken and the conclusions drawn. Building on this framework, we introduce QIQCBench, a benchmark of 49 expert-authored tasks spanning multiple layers including calibration and control, error correction and compilation, sensing and networking. Across 17 frontier agentic systems, QIQCBench reveals wide variation in verified performance. These results expose a substantial gap between demonstrating capability and achieving reliable operation, and establish Quantum-Harbor as a foundation for measuring progress towards verified autonomy in quantum engineering.

## INTRODUCTION

Quantum technologies, spanning computing, simulation, communication, and sensing [1–4], are moving from proof-of-principle demonstrations toward practical applications [5–16]. Quantum engineering enables this transition by integrating quantum platforms and control methods as well as software into reliable systems. Building and operating quantum platforms has traditionally relied heavily on human experts, who assess device information, interpret experimental results, and make the necessary adjustments [17]. As these platforms scale, the number of components, parameters, and workflows that must be coordinated and optimized grows, making manual diagnosis increasingly impractical [18, 19]. Several individual engineering tasks have therefore been automated [20–22]. Scientific artificial intelligence (AI) agents ofer a way to extend this automation to longer workflows [17, 23]. These agents can plan actions, use experimental and computational tools, and adapt their strategies in response to feedback [24]. Recent eforts have begun to apply such agents to quantum experiments and software workflows [25–33]. Yet whether current agents can reliably carry out complete quantum engineering workflows remains unclear.

Evaluating this capability requires more than check ing whether an agent produces the correct answer or completes a predefined task. In practical quantum engineering, critical information often becomes available only during operation [34, 35]. Because device information may be only partially known, an agent must choose and execute experiments or computations, interpret the resulting observations, and decide how to proceed [36, 37]. Even with the relevant scientific knowledge, an agent may rely on information it did not measure, misinterpret ex perimental data, or produce a result that fails on fresh data [38]. We therefore focus on verified autonomy, the ability of an agent to complete an engineering task and support its conclusions with evidence generated during execution. Outputs that cannot be verified directly from recorded data, such as calibrations or designs, must instead be tested under fresh hidden conditions. Existing benchmarks typically assess final outputs without systematically checking whether those results are supported by the agent’s execution record [39, 40].

To make verified autonomy measurable, we develop Quantum-Harbor, a virtual quantum laboratory that provides experimental and computational interfaces while keeping device properties and grading ground truth hidden. For each task, the agent receives only a high-level objective and a finite action budget. The agent must decide how to interact with the environment, conduct experiments or computations, and submit a final result. Quantum-Harbor records the agent’s actions and the resulting data and uses this record to verify the submitted result. The grader verifies numerical results by recomputing them from raw experimental data, while for calibration and design outputs, it retests them under fresh hidden conditions. Building on this environment, we construct a benchmark comprising 49 expert-designed quantum engineering tasks covering hardware and software workflows at multiple architectural layers. We use this benchmark to systematically evaluate verified autonomy in frontier agentic AI systems.

![](images/2fc80eaed357cf6c6741d698337edd0be95b16f61ae655be35642495a6d6f218.jpg)  
FIG. 1. AI agents for quantum engineering and representative workflows. a. Agent-operated quantum engineering. The agent iterates planning, execution, analysis, and adaptation while interacting with quantum laboratories. The workflow operates across the layers shown on the right, with individual tasks typically engaging multiple layers, and its outputs are submitted for independent verification. b. Decoder calibration for a lattice-surgery logical CNOT gate. The agent measures error correlations across the merge seam between surface-code patches and submits a retuned decoder, verified on fresh syndrome records. The public acceptance criteria require a distance-5 logical-memory error below 10<sup>−3</sup> per cycle and, for eight logical-CNOT test circuits spanning both measurement bases, a mean error below 0.175 with each circuit-specific error below 0.185. c. Surrogate modeling of a hidden 60-qubit parameterized quantum circuit. The agent collects measurement data and constructs a surrogate model. Once the measurement record is fixed, 40 held-out inputs are revealed, for which the agent predicts 1,770 two-qubit correlations for each input. The predictions are compared with hidden reference correlations, with accuracy evaluated by the root mean square error.

## RESULTS

## Representative quantum research tasks

Agent-operated quantum engineering places the AI agent in direct interaction with quantum laboratories, closing the loop between experiments and analysis (Fig. 1a). The laboratory may host quantum simulators or quantum platforms such as superconducting circuits, trapped ions, neutral atoms and photonic systems. During execution, the agent chooses experimental or computational actions, interprets the resulting data, and revises its plan as new information is acquired. The workflow therefore evolves adaptively as execution progresses. A single workflow may span multiple layers of quantum engineering, from device characterization and gate calibration to error correction, compilation, and applications. The resulting execution record provides the evidence needed to assess and verify the agent’s outputs. We now examine two representative quantum research tasks that require distinct agent capabilities (Fig. 1b,c).

The first task focuses on the logical CNOT gate which is a basic building block of fault-tolerant quantum computation. Lattice surgery realizes this gate by merging and splitting error-correcting code patches [41, 42]. Recent experiments have demonstrated this approach in trapped-ion [43] and superconducting processors [44, 45]. However, tuning the decoder to account for correlated noise not captured by hardware datasheets remains an expert-driven process [5, 46]. In this task, the agent operates a simulated 114-qubit transmon grid containing three distance-3 surface-code patches, a routing region used to fuse them during surgery, and a distance-5 memory patch. It submits decoder configurations that the hidden grader replays on fresh syndrome data. A successful configuration must clear two criteria that bound the per-cycle error of the distance-5 memory and the logical-CNOT error across test circuits in both measurement bases, as shown in Fig. 1b. Both criteria follow from a hidden noise model that is set at the physical error rates of current below-threshold processors [5]. The task also embeds a failure mode taken from real laboratories. The lab notebook records earlier memory measurements on separate patches and recommends reusing the resulting decoder calibration for the logical operation. Merging the patches introduces correlated errors along their shared boundary, which are not visible in those earlier measurements, so the calibrated decoder can fail during surgery. The strongest agents instead measure the merged configuration directly and clear both criteria, with the best run reaching a memory error of $6 . 6 5 \times 1 0 ^ { - 4 }$ per cycle and a mean logical-CNOT error of 0.131 on fresh syndrome records.

The surrogate model task [47–49] requires the agent to design informative measurement protocols and construct a predictive model of an unknown quantum system (Fig. 1c). The agent interacts with a 60-qubit circuit U(θ) containing Cliford and non-Cliford gates, with 18 non-Cliford rotation gates controlled by six independent parameters. Within a fixed experimental budget, it determines how to sample the parameter space and allocate shots across measurement bases. After the measurement phase ends, 40 previously hidden inputs are revealed, for which the agent must predict 1,770 pairwise correlations without further measurements. The predictions are compared with reference values generated from the same circuit and readout noise model. Success therefore requires learning from limited measurements and generalizing to unseen parameter settings. Failed runs often expend substantial experimental resources but fall short because of uninformative sampling, unreliable validation or incomplete execution. Many of these runs progress through much of the workflow, while consistent end-to-end completion remains challenging.

These examples illustrate both the emerging capabilities and current limitations of agentic quantum engineering. The lattice surgery task requires identifying noise that emerges only in the merged configuration, whereas the surrogate model task requires learning from limited measurements to predict unseen inputs. In both cases, success depends on information acquired during execution. Evaluating such workflows therefore requires preserving the interaction record and, where conclusions cannot be checked directly, testing them on fresh data or unseen inputs. These requirements motivate a controlled environment for systematic evaluation.

## The Quantum-Harbor framework

Quantum-Harbor provides the large language model (LLM) agent with a sandbox environment to interact with quantum hardware and complete quantum research tasks (Fig. 2). It builds on Harbor [50], a general framework for sandboxed agent tasks such as terminal shell use and software development [51, 52]. Quantum-Harbor adds three features to Harbor: a laboratory interface for quantum experiments, interchangeable quantum hardware backends, and evidence-bound grading. Each Quantum-Harbor run launches two isolated environments, namely the agent environment and the sidecar. The agent environment (Fig. 2a) runs the language model harness such as Codex and Claude Code, and stores the workfiles used by the agent. The agent reads the working files and writes code to interact with the public surface exposed in the agent environment. The public surface is a set of tools exposed via the Model Context Protocol (MCP) [53], a published standard that lets a language model call an external service the way a program calls a function. Through them the agent can read the hardware spec, write the lab notebook whose calibration entries may be outdated, and submit experiment jobs, and its final answer. The agent can also carry out the experiment through the Qiskit [54] or QCoDeS [55] libraries that a human experimentalist would use. Every experiment submission returns a job handle to the agent and completes asynchronously, and the job results are raw records, such as readout points or per-shot bitstrings.

Behind the public surface is the sidecar (Fig. 2b,c). The agent can only access and interact with the public surface, and the sidecar is hidden and isolated from the agent. The sidecar hosts the detailed implementation of the quantum hardware or simulated hardware. To support a variety of quantum platforms, the sidecar contains diferent quantum hardware types called qtypes (Fig. 2b) to accommodate the variety among diferent quantum hardware. A qtype abstracts the physics of one class of experimental platform, such as a multilevel transmon under pulse control, a trapped-ion chain, or a neutral-atom array. Qtypes can simulate quantum hardware, with predefined hidden parameters such as the coherence times, anharmonicities, readout confusion matrices, noise correlations, and random seeds (see Supplementary Material for details about the implementation of simulation). It also allows live connection to real quantum hardware, such as superconducting systems from IBM, Rigetti, and ion trap systems from IonQ and Quantinuum. The replay backend is the dry-run mode for the live quantum hardware and runs on recorded data. The simulator, real quantum hardware, and the replay backend are interchangeable. Switching among them within one qtype does not require any change to task definitions or public surfaces.

![](images/b4f8ab3c87687e1b3e3554f57d2bdb99533be13be6c8c47981627b453f2101a2.jpg)  
FIG. 2. Architecture and evidence-bound grading in Quantum-Harbor. a. The agent environment contains the language model, work files, and agent-written code. Through the public interfaces, the agent consults hardware specifications and lab notebooks, submits experiment jobs, retrieves raw measurement records (bitstrings shown), and submits its final answer. Experiment jobs execute asynchronously. b. The hardware layer provides quantum hardware abstractions (qtypes) for transmons, neutral atoms, trapped ions, and other platforms. Experiments run on simulators governed by hidden device and noise parameters, or on replay and live-device backends where supported. An evidence log records experimental requests and outcomes for subsequent verification. c. The grader recomputes relevant quantities, binds submitted claims to the recorded experiments, and applies task-specific scoring criteria (pass/fail shown). For calibration tasks, submitted configurations can be evaluated on fresh simulated data. The shaded region and dashed boundary indicate components inaccessible to the agent, including hidden parameters, the private evidence log, and grading logic.

The evidence log inside the sidecar records every experiment job executed on either simulator or real quantum hardware. An LLM might report a number it never measured [56, 57]. Therefore, the grader (Fig. 2c) is used to evaluate LLMs’ performance on quantum research tasks and is bound to the evidence log. The grader verifies the task result by recomputing it from the raw data in the evidence log. When the submission is a calibration and not a number, the grader replays it against fresh draws from the hidden parameters. This prevents the agent from being rewarded for submitting false results.

## The QIQCBench benchmark

Building on the Quantum-Harbor framework, we next introduce QIQCBench, which comprises 49 tasks built on 36 qtypes, all contributed and reviewed by domain experts. Their expertise spans the working breadth of quantum engineering. Fig. 3a shows the composition of the suite, with tasks organized into six categories. Representative measurement and characterization tasks include estimating the relaxation time of a drifted transmon and reconstructing correlated gate errors [34, 59], while quantum control tasks range from pulse design to Hamiltonian engineering [37, 60]. Error correction and mitigation tasks include problems such as decoder calibration and logical-memory protection [5, 46], and compilation and resource-optimization tasks map target computations to trapped-ion, neutralatom, and superconducting architectures [18, 21]. Tasks in simulation and many-body physics include inferring hidden Hamiltonians and open-system dynamics [22, 61], while communication and sensing covers, for example, entanglement purification across repeater arrays [62] and GHZ-assisted frequency estimation under non-Markovian noise [63]. The tasks also cover the full quantum computing stack [58], as shown in the five layers in Fig. 3b. In our benchmark, the agent needs to reason across diferent layers.

As shown in Fig. 3, the tasks run on simulated hard ware. Several tasks require no specific device and are marked device-less. Orthogonally, each task is rated in one of two modes: i) pass–fail, where a hidden verifier returns a Boolean pass or fail, or ii) racing, where submissions are ranked by a Bradley–Terry Elo fit (see Methods). Finally, three deliberately simple test cases (e.g., measuring the T<sub>1</sub> of a transmon qubit) stress-test the benchmark infrastructure and are excluded from all science scores. We further note that per-task hidden parameters and pass thresholds stay out of all published material, so future systems cannot learn the answers from this paper.

![](images/d96269869706caf79f06f5696a28b43e276f6cfa2d2a580f6a3d3c01fb3d9ac7.jpg)  
FIG. 3. Task composition and quantum computing stack coverage of QIQCBench. a. The 49 tasks grouped into six scientific categories. Each column represents one task. Colors identify the hardware platform, with separate labels for platform-agnostic and device-less tasks. Evaluation types are indicated by filled squares for the 39 tasks assessed by a binary pass–fail evaluation, orange diamonds for the 7 racing tasks ranked by Elo rating, and open squares for the 3 infrastructure test cases excluded from science scores. b. Mapping of each task to five layers of the quantum computing stack, adapted from Ref. [58]: physical device, gates and calibration, error correction, logical and compilation, and application and algorithms. Dark blue cells mark the layers engaged by each task. Pale cells indicate unmapped layers. Columns share the task indices and category boundaries of panel a, showing how individual tasks connect multiple layers of quantum engineering.

We established three principles during the development of this benchmark. First, every task is designed as an end-to-end laboratory mini-project whose deliverable is a significant final result, such as a physical quantity, a calibrated artifact, or an optimized design. Second, we require that a task’s dificulty come from the physics and not from obscurity. Each task has an intended solution path that a competent experimentalist can follow, and a task is hard only when that path turns on a nonobvious inference. Third, to increase the dificulty and mimic the laboratory setting, contributors tend to design traps (or tricks) that can mislead the model. However, we require that every trap be honest, of a kind that arises in realistic experimental environments. For example, stale notebooks are labeled as notebooks and budgets are stated explicitly. The agent is not punished for missing information due to our infrastructure.

## Benchmark results

We evaluated seventeen agentic systems from eight vendors, three from OpenAI (GPT-5.6-sol, GPT-5.6-terra, and GPT-5.6-luna), three from DeepSeek (DeepSeek-V4.1-Flash, V4-Pro, and V4-Flash, hereafter DS-V4.1-Flash and so on), three from Z.ai (GLM-5.3, 5.3- Flash, and 5.2), two from Anthropic (Claude Opus 4.8 and Haiku 4.5), two from Alibaba (Qwen3.8-Max, 3.8- Flash), two from Tencent (Hunyuan T3 and 4 preview, denoted by Hunyuan T4\*), one from Moonshot (Kimi K3), and one from ByteDance (Seed 2.1 Pro). The two Anthropic systems ran under the Claude Code agent harness and the three OpenAI systems under the Codex CLI harness. The remaining twelve systems ran under the OpenCode harness. Every system ran at its extra high (xhigh) or higher reasoning-efort setting and attempted each task three times independently.

Fig. 4a ranks the seventeen systems by pass rate on the pass–fail tasks. Pass rates run from 78% down to 3% with performance decreasing steadily across the ranked systems and no clear separation into groups. We note that systems scoring within a narrow band on a frontier agentic benchmark spread across almost the whole range of QIQCBench. For example, on AutomationBench-AA, the independent run of Zapier’s AutomationBench [64] carried out by Artificial Analysis [65], eight of our seventeen systems score between 54.0% and 62.2%. For

a

d

![](images/1f2a4418c8ec9a1469624894955b05cc2792a17523a6e03dd938db1d992d7f61.jpg)

![](images/7927e8baae1ff1c40fb40d31cbf31780c9e378655880db77ca6d98d9fd79773c.jpg)  
FIG. 4. Benchmark results and analysis of failure and pass patterns. a. Pass rate on the pass–fail tasks. Error bars: Wilson 68% intervals. b. Outcome composition of the trials, including passes, science failures, contract failures, and no-submission trials. c. Bradley–Terry Elo ratings over the 7 racing tasks (filled circles: games-weighted aggregate; open circles: per-task ratings). d. Per-task passes for every system, grouped by the scientific categories of Fig. 3, easiest to hardest within each group. Only pass-fail tasks are shown here.

GDPval-AA v2 [66], a widely adopted evaluation, six of them sit within 31 Elo points of one another. We conclude that our benchmark can separate systems that perform alike on other frontier agentic benchmarks.

We classify failures into three categories (Fig. 4b), including no submission when no answer arrives, science failure when the hidden grader rejects the physics, and contract failure when the answer breaks the submission protocol. These categories correspond to three requirements of autonomous experimentation: completing the task, reaching a scientifically valid conclusion, and returning that conclusion in a form that can be verified. No submission is the largest failure category, accounting for 937 of 1,510 failures. Seven systems exhaust their budget before submitting an answer in more than 60% of trials. These trials are, for the most part, neither crashes nor idle hangs. When timed out, the agents are often still running experiments. Among the remaining failures, 54% are science failures and the rest are contract failures. Across the pool, many failures arise from how agents conduct experiments, including relying on stale priors, mismanaging budgets, and discarding in-hand re-

sults at submission time.

The racing tasks score performance on a continuous metric, such as required resources or run time (Fig. 4c). Every trial races pairwise against the trials of every other system. A trial that clears the task-specific feasibility gate always ranks above one that does not. Among gatepassing trials, the ranking is determined by the recomputed metric (see Methods). Of the 357 race trials, 125 cleared their gates. A Bradley–Terry fit spreads the field from 1694 down to 582, with the anchor fixed at 1000. The ordering on the racing tasks corroborates that on the pass–fail tasks, with the same top three systems, but produces wider margins by incorporating design quality alongside reliability.

We examine the pass pattern in Fig. 4d. Thirty-seven of the 39 tasks were solved by at least one system. For the two exceptions, a surrogate modeling on 60 qubits and a net-zero tunable-coupler CZ gate design, expert reviewers confirmed the intended solution paths. Six tasks were solved only by the three highest-ranked systems. Per-task ratings, gate outcomes, the failure-category definitions, and the audit protocols are tabulated in the Supplementary Material.

We observe several interesting patterns in the benchmark result. We can observe from Fig. 4d that simulation and many-body physics is the hardest category for the pool, with a passing rate of only 15%. The top three systems pass 64% of its trials and the other fourteen pass only 4%, also the largest such gap of any category. We found no significant correlation between dificulty and stack depth, as the pass rate is uncorrelated with the number of layers a task spans (Spearman ρ = 0.14, p = 0.40, see Supplementary Material) and both unsolved tasks sit in the bottom two layers, gates & calibration and physical device. Success is intermittent, with only 34% of the 249 system and task pairs that pass at least once passing all three attempts.

## DISCUSSION

The present benchmark results show that current agents exhibit meaningful but uneven capability on demanding quantum engineering tasks. Large performance gaps between systems and substantial variability across repeated attempts indicate that reliable completion remains a central challenge for autonomous quantum engineering. The failure patterns point to experimental judgment as an important direction for improvement. Choosing informative measurements, revising stale assumptions, and deciding when to stop are recurring demands when agents must reach supported conclusions under finite resources. These failures are not explained by limitations in scientific knowledge or laboratory execution alone, suggesting that improvements to agent harnesses could substantially increase reliability.

Quantum-Harbor makes the verified autonomy of AI agents measurable by evaluating not only their outputs but also whether those outputs are supported by information acquired during execution. By placing agentic systems in controlled quantum environments, it provides a framework for examining how agents gather evidence, revise hypotheses, allocate experimental resources and arrive at verifiable conclusions. QIQCBench realizes this framework across a broad range of cross-layer quantum engineering tasks, revealing diferences between agentic systems that are not apparent from domain-knowledge and coding benchmarks alone.

Beyond the present benchmark, Quantum-Harbor could support controlled studies of why agents succeed or fail. Task conditions, operating instructions and resource constraints could be varied systematically to disentangle the efects of model capability, harness design and scientific decision-making. Such studies could track progress beyond aggregate pass rates and guide improvements in the reliability and eficiency of autonomous quantum engineering.

We note that the present results are limited to simulated environments and characterize complete agentic systems operating through diferent harnesses under fixed resource budgets. Future work should determine how these findings transfer across experimental conditions and physical devices, and whether increasingly capable agents can reduce the measurement cost and human effort required for quantum engineering.

## METHODS

Task construction and review. Each task follows a common construction and review pipeline. A domain expert prepares a task-design brief specifying the scientific objective, the intended solution approach, and the taskspecific pitfalls. A maintainer then builds the task on the Quantum-Harbor runtime, implementing the public materials, the hidden configuration, and the grader as separate artifacts, with the runtime enforcing the public– hidden boundary (Supplemental Material). Before the task is included in the suite, a second expert reviews the package. We verify the task end-to-end on Quantum-Harbor and use coding agents (Claude Code or Open-Code with Fable and Opus 5) to audit selected agents’ outcomes as a sanity check.

Evaluation protocol. Each trial runs the agent container and the hardware-sidecar container. After the agent’s session ends, the sidecar container grades the run. The per-task time budget is 3 hours for all tasks. The agent container has internet access, allowing the system to consult public sources during the run. All tasks ran against the simulation backend. We used coding agents to audit every trial’s complete action record. Trials affected by confirmed infrastructure faults were quaran-

tined and rerun afterward.

Failure categories. We distinguish three failure categories for pass-fail tasks, as stated in the main text. The science failure occurs when an admissible submission fails the task’s scientific checks. Depending on the grading contract, the hidden grader checks the submission against hidden ground truth, recomputes reported results from logged evidence, or evaluates the submission in a replay on fresh data. The contract failure occurs when a submission is rejected at an admissibility gate before its scientific content is scored, owing to a schema violation, missing mandatory evidence, or delivery outside the published submission channel. The no-submission trial ends without a final answer being delivered. Categories were assigned from each failing trial’s complete action record, by expert transcript audit or directly from the trial’s scorer and termination records, with the source recorded per trial.

Rating of racing tasks. Each racing task runs on a fixed instance, so the grader-recomputed race metrics of diferent systems’ submissions are directly comparable. Every trial is compared pairwise against every trial of every other system on the task. A trial that passes the task’s feasibility gate beats a gate-failing trial. Two gate-passing trials are ordered by the recomputed race metric. Exact ties and pairs of gate failures are excluded from the fit. Thus, a gate-failing trial contributes a loss against each gate-passing trial of another system, but gate failures are not ranked among themselves. For the retained comparisons, the Bradley– Terry model assigns each system i a latent strength $\theta _ { i }$ and models the probability that system i beats system j as $\sigma ( \theta _ { i } - \theta _ { j } )$ , where $\sigma ( x ) = 1 / ( 1 + e ^ { - x } )$ is the logistic function. Let $w _ { i j }$ denote the number of retained comparisons won by system i against system j. For each task, we fit a Bradley–Terry model by maximizing $\textstyle \sum _ { i , j } w _ { i j }$ ln $\begin{array} { r l } { \sigma ( \theta _ { i } - \theta _ { j } ) - \frac { \lambda } { 2 } \lVert \theta \rVert ^ { 2 } } \end{array}$ . The ridge penalty, with $\lambda = 0 . 5$ , keeps the fitted strengths finite even for systems with only wins or only losses. We report strengths on the Elo scale, $\mathrm { E l o } _ { i } = 1 0 0 0 + ( 4 0 0 / \ln 1 0 ) ( \theta _ { i } - { \bar { \theta } } )$ , where $\bar { \theta }$ is the mean fitted strength across systems rated on that task. This scaling gives each task a mean rating of 1000. A 400-point rating advantage corresponds to model-implied win odds of 10 : 1. A system’s aggregate rating is the weighted mean of its per-task ratings, with each task weighted by that system’s number of retained comparisons (wins plus losses). Tasks on which the system has no retained comparisons are excluded from its aggregate. The rating combines how often a system produces a feasible design with the quality of its feasible designs.

Statistics. A system’s pass rate in Fig. 4a is $k / n$ , the number k of passing trials over its $n \ = \ 1 1 7$ pass–fail trials (39 tasks, three repetitions each). The error bars

are Wilson score intervals,

$$
\frac { k + z ^ { 2 } / 2 } { n + z ^ { 2 } } \pm \frac { z } { n + z ^ { 2 } } \sqrt { \frac { k ( n - k ) } { n } + \frac { z ^ { 2 } } { 4 } } ,
$$

with $z = 1$ , corresponding to a nominal confidence level of approximately 68%. These intervals use a pooled binomial approximation and do not quantify uncertainty due to the choice of tasks.

## Acknowledgement

The authors thank Feng Pan, Zhan Yu, Yunlong Xiao and Ikko Hamamura for helpful discussions.

[1] F. Arute et al., Quantum supremacy using a programmable superconducting processor, Nature 574, 505 (2019).

[2] A. J. Daley, I. Bloch, C. Kokail, S. Flannigan, N. Pearson, M. Troyer, and P. Zoller, Practical quantum advantage in quantum simulation, Nature 607, 667 (2022).

[3] K. Azuma, S. E. Economou, D. Elkouss, P. Hilaire, L. Jiang, H.-K. Lo, and I. Tzitrin, Quantum repeaters: From quantum networks to the quantum internet, Rev. Mod. Phys. 95, 045006 (2023).

[4] C. L. Degen, F. Reinhard, and P. Cappellaro, Quantum sensing, Rev. Mod. Phys. 89, 035002 (2017).

[5] Google Quantum AI and Collaborators, Quantum error correction below the surface code threshold, Nature 638, 920 (2025).

[6] D. Bluvstein et al., Logical quantum processor based on reconfigurable atom arrays, Nature 626, 58 (2024).

[7] A. Paetznick et al., Improved quantum processor logical error rates via correction and detection, Nature 654, 349 (2026).

[8] Y. Kim et al., Evidence for the utility of quantum computing before fault tolerance, Nature 618, 500 (2023).

[9] Google Quantum AI and Collaborators, Observation of constructive interference at the edge of quantum ergodicity, Nature 646, 825 (2025).

[10] A. L. Shaw, Z. Chen, J. Choi, D. K. Mark, P. Scholl, R. Finkelstein, A. Elben, S. Choi, and M. Endres, Benchmarking highly entangled states on a 60-atom analogue quantum simulator, Nature 628, 71 (2024).

[11] W.-Z. Liu, Y.-B. Zhou, J.-P. Chen, B. Wang, A. Teng, X.-W. Han, G.-C. Liu, Z.-J. Zhang, Y. Yang, F.-G. Liu, C.-H. Xue, B.-W. Yang, J. Yang, C. Zeng, D.-R. Pan, M.-Y. Zheng, X. Zhang, S. Cao, Y.-Z. Zhen, Y. Xiao, H. Li, L. You, X. Ma, Q. Zhao, F. Xu, Y. Wang, Y. Wan, Q. Zhang, and J.-W. Pan, Long-lived remote ion–ion entanglement for scalable quantum repeaters, Nature 652, 51 (2026).

[12] Y. Zheng, H. Wang, X. Jia, J. Huang, H. Yuan, C. Zhai, J. Dai, J. Shi, L. Zhang, X. Zhang, M. Zhuang, J. Liu, J. Mao, T. Dai, Z. Fu, Y. Jiao, Y. Shi, D. Dai, X. Wang, Y. Li, Q. Gong, Z. Yuan, L. Chang, and J. Wang, Largescale quantum communication networks with integrated photonics, Nature 651, 68 (2026).

[13] P.-J. Stas, Y.-C. Wei, M. Sirotin, Y. Q. Huan, U. Yazlar, F. Abdo Arias, E. Knyazev, G. Baranes, B. Machielse, S. Grandi, D. Riedel, J. Borregaard, H. Park, M. Lonˇcar, A. Suleymanzade, and M. D. Lukin, Entanglementassisted non-local optical interferometry in a quantum network, Nature 651, 326 (2026).

[14] X. Zhou, M. Wang, X. Ye, H. Sun, Y. Guo, S. Han, Z. Chai, W. Ji, K. Xia, F. Shi, Y. Wang, and J. Du, Entanglement-enhanced nanoscale single-spin sensing, Nature 647, 883 (2025).

[15] J. Rovny, S. Kolkowitz, and N. P. de Leon, Multi-qubit nanoscale sensing with entanglement as a resource, Nature 647, 876 (2025).

[16] W. Ji, Z. Liu, Y. Guo, Z. Hu, J. Zhou, S. Dai, Y. Chen, P. Yu, M. Wang, K. Xia, F. Shi, Y. Wang, and J. Du, Correlated sensing with a solid-state quantum multisensor system for atomic-scale structural analysis, Nat. Photonics 18, 230 (2024).

[17] Y. Alexeev, M. H. Farag, T. L. Patti, M. E. Wolf, N. Ares, A. Aspuru-Guzik, S. C. Benjamin, Z. Cai, S. Cao, C. Chamberland, et al., Artificial intelligence for quantum computing, Nat. Commun. 16, 10829 (2025).

[18] M. E. Beverland et al., Assessing requirements to scale to practical quantum advantage (2022), arXiv:2211.07629.

[19] M. Mohseni et al., How to build a quantum supercomputer: Scaling from hundreds to millions of qubits (2024), version 3, revised March 2026, arXiv:2411.10406.

[20] H. Moon et al., Machine learning enables completely automatic tuning of a quantum device faster than human experts, Nat. Commun. 11, 4161 (2020).

[21] F. T. Chong, D. Franklin, and M. Martonosi, Programming languages and compiler design for realistic quantum hardware, Nature 549, 180 (2017).

[22] J. Wang, S. Paesani, R. Santagati, S. Knauer, A. A. Gentile, N. Wiebe, M. Petruzzella, J. L. O’Brien, J. G. Rarity, A. Laing, and M. G. Thompson, Experimental quantum Hamiltonian learning, Nat. Phys. 13, 551 (2017).

[23] Y. Du, Y. Zhu, Y.-H. Zhang, M.-H. Hsieh, P. Rebentrost, W. Gao, Y.-Z. You, J. Eisert, G. Chiribella, D. Tao, B. C. Sanders, and Y.-D. Wu, Artificial intelligence for representing and characterizing quantum systems, Nat. Rev. Phys. 8, 579 (2026).

[24] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, ReAct: Synergizing reasoning and acting in language models, in The Eleventh International Conference on Learning Representations (2023).

[25] S. Cao, Z. Zhang, M. Alghadeer, S. D. Fasciati, M. Piscitelli, M. Bakr, P. Leek, and A. Aspuru-Guzik, Automating quantum computing laboratory experiments with an agent-based AI framework, Patterns 6, 101372 (2025).

[26] H. Xu, J. Han, S. Ou, et al., Vibe calibration: Autonomous bring-up of a 112-qubit superconducting quantum processor by a skill-orchestrating language agent (2026), arXiv:2606.22376.

[27] S. Arlt, X. Gu, and M. Krenn, Towards autonomous quantum physics research using LLM agents with access to intelligent tools (2025), arXiv:2511.11752.

[28] T. Isogawa, R. Okabe, N. Phadetsuwannukun, M. Li, and P. Cappellaro, Agentic AI for scientific reasoning in autonomous quantum sensing experiments (2026), arXiv:2607.25145 [quant-ph].

[29] C. Dalyac, A. Dauphin, L. Henriet, and C. Jurczak, Lowering the implementation barrier of neutral-atom

quantum computing with agentic workflows (2026), arXiv:2607.25834.

[30] Z. Fu, L. Jiang, Y. Xu, G. Huang, and F. Chen, QAgent: An LLM-based multi-agent system for autonomous OpenQASM programming (2025), arXiv:2508.20134 [cs.AI].

[31] W. Li, J. Ren, L. Cheng, and C. Gong, Autonomous quantum simulation through large language model agents (2026), arXiv:2601.10194 [quant-ph].

[32] I. Gustin, L. Mantilla Calder´on, J. B. P´erez-S´anchez, C. Crebolder, J. F. Gonthier, M. Ghazi Vakili, Y. Nakamura, K. Panicker, M. Ramprasad, A. Yin, et al., El agente cu´antico: automating quantum simulations, Rep. Prog. Phys. 89, 077602 (2026).

[33] M. Shiraishi, I. Hamamura, T. Ishigaki, and T. Kadowaki, A Model Context Protocol server for quantum execution in hybrid quantum-HPC environments, in 2026 International Conference on Quantum Communications, Networking, and Computing (QCNC) (IEEE, 2026) pp. 1–6.

[34] T. Proctor, M. Revelle, E. Nielsen, K. Rudinger, D. Lobser, P. Maunz, R. Blume-Kohout, and K. Young, Detecting and tracking drift in quantum information processors, Nat. Commun. 11, 5396 (2020).

[35] V. Sivak, A. Morvan, M. Broughton, R. G. Corti˜nas, J. Bausch, A. W. Senior, M. Neeley, A. Eickbusch, N. Shutty, L. A. Beni, et al., Reinforcement learning control of quantum error correction, Nature 655, 879–884 (2026).

[36] D. T. Lennon, H. Moon, L. C. Camenzind, L. Yu, D. M. Zumb¨uhl, G. A. . D. Briggs, M. A. Osborne, E. A. Laird, and N. Ares, Eficiently measuring a quantum device using machine learning, npj Quantum Information 5, 79 (2019).

[37] Y. Baum et al., Experimental deep reinforcement learning for error-robust gate-set design on a superconducting quantum computer, PRX Quantum 2, 040324 (2021).

[38] S. Cao et al., QCalEval: Benchmarking vision-language models for quantum calibration plot understanding (2026), arXiv:2604.25884.

[39] S. Minami, T. Ishigaki, I. Hamamura, et al., QuantumBench: A benchmark for quantum problem solving (2025), arXiv:2511.00092.

[40] R. Yang, Z. Wang, Y. Gu, Y. Liang, and T. Li, QCircuitBench: A large-scale dataset for benchmarking quantum algorithm design, in Advances in Neural Information Processing Systems, Vol. 38 (2025) pp. 48750–48801.

[41] D. Horsman, A. G. Fowler, S. Devitt, and R. Van Meter, Surface code quantum computing by lattice surgery, New J. Phys. 14, 123011 (2012).

[42] D. Litinski, A game of surface codes: Large-scale quantum computing with lattice surgery, Quantum 3, 128 (2019).

[43] A. Erhard et al., Entangling logical qubits with lattice surgery, Nature 589, 220 (2021).

[44] I. Besedin et al., Lattice surgery realized on two distancethree repetition codes with superconducting qubits, Nat. Phys. 22, 189 (2026).

[45] W. Lin et al., Surface code logical operations on a superconducting quantum processor (2026), arXiv:2607.01473 [quant-ph].

[46] J. Bausch et al., Learning high-accuracy error decoding for quantum processors, Nature 635, 834 (2024).

[47] F. J. Schreiber, J. Eisert, and J. J. Meyer, Classical surrogates for quantum learning models, Phys. Rev. Lett. 131, 100803 (2023).

[48] Y. Du, M.-H. Hsieh, and D. Tao, Eficient learning for linear properties of bounded-gate quantum circuits, Nat. Commun. 16, 3790 (2025).

[49] W.-Y. Liao, Y. Du, X. Wang, T.-C. Tian, Y. Luo, B. Du, D. Tao, and H.-L. Huang, Demonstration of eficient predictive surrogates for large-scale quantum processors, Nat. Commun. 17, 4731 (2026).

[50] Harbor Framework Team, Harbor: A framework for evaluating and optimizing agents and models in container environments, https://doi.org/10.5281/zenodo.21878 893 (2026), version 0.21.0, https://github.com/harbo r-framework/harbor.

[51] M. A. Merrill, A. G. Shaw, N. Carlini, et al., Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces (2026), arXiv:2601.11868 [cs.SE].

[52] Terminal-Bench-Science Team, Terminal-Bench-Science: Evaluating AI agents on research workflows across scientific domains, Zenodo, https://doi.org/10.5281/zeno do.22110253 (2026).

[53] Anthropic, Introducing the Model Context Protocol, ht tps://www.anthropic.com/news/model-context-pro tocol (2024).

[54] A. Javadi-Abhari, M. Treinish, K. Krsulich, C. J. Wood, J. Lishman, J. Gacon, S. Martiel, P. D. Nation, L. S. Bishop, A. W. Cross, B. R. Johnson, and J. M. Gambetta, Quantum computing with Qiskit (2024), arXiv:2405.08810 [quant-ph].

[55] QCoDeS Contributors, QCoDeS: Python-based data acquisition framework, Zenodo, https://doi.org/10.528 1/zenodo.596989 (2026).

[56] T. Islam, D. Wadekar, and Z. Zhou, gwBenchmarks: Stress-testing LLM agents on high-precision gravitational wave astronomy (2026), arXiv:2605.11269 [gr-qc].

[57] N. Chowdhury, D. Johnson, V. Huang, J. Steinhardt, and S. Schwettmann, Investigating truthfulness in a prerelease o3 model, Transluce technical report, https://tr ansluce.org/investigating-o3-truthfulness (2025), accessed 8 August 2026.

[58] N. C. Jones, R. Van Meter, A. G. Fowler, P. L. McMahon, J. Kim, T. D. Ladd, and Y. Yamamoto, Layered archi-

tecture for quantum computing, Phys. Rev. X 2, 031007 (2012).

[59] M. Sarovar, T. Proctor, K. Rudinger, K. Young, E. Nielsen, and R. Blume-Kohout, Detecting crosstalk errors in quantum information processors, Quantum 4, 321 (2020).

[60] C. P. Koch, U. Boscain, T. Calarco, G. Dirr, S. Filipp, S. J. Glaser, R. Koslof, S. Montangero, T. Schulte-Herbr¨uggen, D. Sugny, and F. K. Wilhelm, Quantum optimal control in quantum technologies. strategic report on current status, visions and goals for research in europe, EPJ Quantum Technol. 9, 19 (2022).

[61] I. de Vega and D. Alonso, Dynamics of non-markovian open quantum systems, Rev. Mod. Phys. 89, 015001 (2017).

[62] H.-J. Briegel, W. D¨ur, J. I. Cirac, and P. Zoller, Quantum repeaters: The role of imperfect local operations in quantum communication, Phys. Rev. Lett. 81, 5932 (1998).

[63] A. W. Chin, S. F. Huelga, and M. B. Plenio, Quantum metrology in non-markovian environments, Phys. Rev. Lett. 109, 233601 (2012).

[64] D. Shepard and R. Salimans, AutomationBench (2026), benchmark by Zapier; scores quoted in this work are from the Artificial Analysis independent run, not from this paper, whose strict whole-task completion metric reports diferent values, arXiv:2604.18934.

[65] Artificial Analysis, AutomationBench-AA: Independent evaluation of AutomationBench, https://artificialan alysis.ai/evaluations/automationbench-aa (2026), independent run by Artificial Analysis on a private heldout set of 657 tasks across six business domains; the headline metric is the average share of each task’s objectives completed without guardrail violations. Accessed 2026- 09-15.

[66] Artificial Analysis, GDPval-AA v2: Independent evaluation of OpenAI’s GDPval gold set, https://artificial analysis.ai/evaluations/gdpval-aa (2026), Elo ratings from blind pairwise comparisons; agents run with shell and browsing access via Stirrup. Accessed 2026-09- 15.