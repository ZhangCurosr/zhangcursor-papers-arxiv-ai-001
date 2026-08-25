# EARTHVERSE: BENCHMARKING SCIENTIFIC AGENTS ACROSS DYNAMIC EARTH SYSTEMS AND NATURAL HAZARDS

Zhiqing Cui<sup>1</sup> Xinxiang Yin<sup>2</sup> Yihong Tang<sup>3</sup> Xinglang Zhang<sup>4</sup> Yuanzhe Hu<sup>5</sup> Siru Zhong<sup>4</sup> Weidong Tang<sup>6</sup> Yuxuan Liang<sup>4</sup> Weijia Li<sup>7</sup> Ming Jin<sup>8</sup> Shirui Pan<sup>8</sup> Yuhao Kang<sup>9</sup> Dingyi Zhuang<sup>10,†</sup> Jinhua Zhao<sup>10</sup> <sup>1</sup>NUIST <sup>2</sup>HKU <sup>3</sup>McGill <sup>4</sup>HKUST(GZ) <sup>5</sup>Georgia Tech <sup>6</sup>NUS <sup>7</sup>Tsinghua <sup>8</sup>Griffith <sup>9</sup>UT Austin <sup>10</sup>MIT

zhiqing@nuist.edu.cn dingyi@mit.edu <sup>†</sup>Corresponding author

Hugging Face GitHub Website

香港科技大學THE HONG KONGUNIVERSITY OF SCIENCEAND TECHNOLOGY

GRIFFITH TEXAS li Massachusetts Institute of Technology

## ABSTRACT

Earth-system analysis reconstructs changing physical processes from observations that differ in source, scale, timing, and modality. Natural hazards make this work consequential because incomplete evidence can change estimates of severity, exposure, and mechanism. We introduce EarthVerse, a benchmark that evaluates scientific agents through package-scoped investigations. Its 405 reproducible tasks are grounded in 199 documented events and 19 hazard families. Agents inspect heterogeneous event packages, choose compatible evidence, execute transparent calculations, reconcile source differences, and preserve provenance in the final answer. We provide executable ground truth that decomposes each task into fine-grained answer units, together with task-specific rubrics that assess the supporting research process while allowing multiple valid paths. We evaluate 25 model and agent systems under a controlled tool-using protocol, then use controlled studies to locate failures in evidence access, tool selection, memory, reasoning, interaction, and scientific execution. Across systems, the best mean answer-unit accuracy is 84.65%, while the highest Strict@95 is only 34.81%. The gap shows that current agents often complete individual steps without maintaining a consistent chain across evidence, scales, units, calculations, and physical interpretation. EarthVerse provides a reproducible basis for measuring end-to-end scientific reliability in dynamic Earth systems.

![](images/59b6290fb26f0b878937d6ea089b91a83c48d08ea9a754f5ba4db53a99869594.jpg)  
(a) Scientific coverage.

![](images/62e8c9697ed30fbf76969bec39c495132c80d547d62dd798c33accbad9bc1557.jpg)  
(b) Leaderboard landscape.  
Figure 1: Coverage and model landscape. Task and answer-unit coverage across hazards and capabilities (left); mean answer-unit accuracy and token use across 25 systems (right).

## 1 Introduction

Earth-system science asks how the atmosphere, land, ocean, ecosystems, and human activity shape one another. These processes govern water, food, health, infrastructure, and climate risk. Natural hazards make the question immediate: weather, climate, and water extremes have caused extensive mortality and economic loss, while compound events can propagate across sectors and regions [1–4]. Timely analysis matters, but so does scientific traceability. A plausible explanation is not enough when it may inform disaster assessment or response.

Natural-hazard analysis applies Earth-system reasoning to decisions about event severity, exposure, physical mechanism, and response. It requires researchers to reconstruct an evolving process from observations collected for different purposes and at different scales.

The difficulty begins with observation. No single instrument records an evolving Earth system in full. Agency reports, stations, satellites, reanalyses, maps, and impact databases measure different variables over different footprints and time windows. Combining them requires scientific judgment before calculation: the analyst must determine which records describe the same process, which scale is appropriate, and what uncertainty remains. Work on machine learning for Earth systems and model–data integration has made these records easier to use, but has not removed the need to reconcile them [5–8].

Foundation models and tool-using agents now make a broader form of analysis possible. Systems such as Google Earth AI combine imagery, population, and environmental models through a reasoning agent, while scientific agents can inspect files, execute code, and revise a result after new observations arrive [9–13]. This changes the evaluation target. A system has to decide what evidence to seek, bind each value to the correct source and scale, and let tool feedback alter its account when the evidence disagrees.

Most Earth-science benchmarks begin after that decision has been made. They test knowledge, figure interpretation, remote sensing, or geospatial reasoning over a supplied passage, image, or named asset [14–16]. Newer multimodal and agent benchmarks broaden the interface, but generally continue to declare the observation or data layer in advance [17, 18]. Frontier systems already approach or exceed published expert references on some of these formats. Such results do not show whether the same system can construct a compatible evidence base for an open question, carry calculations across several sources, and preserve support for every part of its conclusion.

Position. Earth-science analysis requires systems to maintain the correspondence between heterogeneous observations and scientific claims. Satellite products, station records, reanalyses, and event reports can support a shared physical explanation only when their scales, variable meanings, time windows, units, and transformations remain aligned. EarthVerse evaluates this evolving claim–evidence state: a single broken binding can make a locally correct calculation describe the wrong Earth process [19, 20].

EarthVerse contains 405 reproducible investigations grounded in 199 real disasters and extreme events. Each task opens an event collection assembled from 80 external source and access families. A typical event contains about 34 files, and solving a task requires evidence from more than seven distinct sources on average. Every file is connected to the event, so source choice remains a scientific judgment. The difficulty lies in deciding which records answer the question, whether their time windows and spatial support are compatible, and how much weight each source should carry. Working through these materials takes sustained interaction. Across the leaderboard, each task takes about 17 model rounds on average. A full evaluation of a frontier model costs more than \$2,500.

Figure 2 samples this breadth, pairing the global event collection with representative investigations that move from heterogeneous observations to an auditable conclusion.

EarthVerse uses transparent mathematics: aggregation, ratios, thresholds, spatial overlap, weighted indices, and bounded counterfactuals. The difficulty lies in combining these operations with compatible evidence. A wrong time window changes the total, which can alter both the comparison and the physical explanation. Tasks therefore combine process reconstruction, quantitative analysis, competing hypotheses, rankings, and open synthesis. Fine-grained answer units expose each obligation, while Strict@95 measures whether the resulting claim–evidence state is nearly complete.

The results show how demanding this composition is. Across 25 systems, the best mean answer-unit accuracy is 84.65%, whereas the highest Strict@95 is only 34.81%. Even the strongest system leaves roughly two thirds of investigations with at least one consequential error or omission. This is the benchmark’s central reliability gap: an agent may recover most local facts and calculations while one unsupported source, mismatched scale, or missing mechanism breaks the account as a whole.

Figure 1 places the benchmark’s scientific coverage beside the resulting model landscape.

![](images/e45f98590692e41af2b6460b7387b87ae919144e7678853f90332a13f26d7f98.jpg)  
Figure 2: Benchmark overview. Global event coverage and four representative investigations. Each case follows a different model trajectory, and performance is scored against the answer units defined for that task. In Case 2, the linked-event score is a deterministic composite derived from package evidence.

Contributions. We contribute (i) EarthVerse, a benchmark of reproducible multi-source investigations with executable answer units and task-specific process rubrics; (ii) a controlled evaluation of 25 model and agent systems that separates average scientific competence from reliable completion; and (iii) controlled experiments across evidence localization, tool selection, memory, reasoning, interaction modes, and professional scientific environments that identify the main failure points along the research chain.

Research questions. We organize the study around three questions. RQ1: Can current systems complete an auditable multi-source hazard investigation, and which missing links separate high average accuracy from reliable completion? RQ2: When does reasoning improve a scientific investigation, and which controls over evidence access, tools, memory, stopping, and execution keep the claim–evidence state open to correction? RQ3: How much of scientific-agent performance comes from interpreting supplied observations, and how much depends on constructing and maintaining the evidence base?

## 2 Related Work

Earth observation and scientific multimodality. Most Earth-science benchmarks isolate a particular form of interpretation: scientific figures, atmospheric processes, geospatial relations, or remote-sensing imagery [15–18, 21, 22]. Complementary work studies how representations transfer across sensors, regions, and downstream tasks [23–26]. These studies establish strong perception and forecasting settings, but the observation set is generally prepared before the model begins its analysis.

Disaster intelligence. Disaster benchmarks have followed a similar progression, from classification and damage mapping over supplied media to tool-supported emergency analysis [27– 29]. DORA moves closer to an operational workflow through expert-authored tasks, typed tools, and replayable reference calls, while retaining a declared set of relevant observations and layers [30].

Tool-using and scientific agents. Agent research supplies the missing interaction machinery. Early work established reasoning–action loops and learned API use [10, 31, 32]; later benchmarks placed agents in executable environments and scientific workflows involving code, data, and long-horizon orchestration [11, 33–37]. BLADE and DiscoveryBench are especially relevant because they allow multiple valid analysis paths while keeping the resulting work evaluable [38, 39].

![](images/a6331ad2f75e76a0efff06f8b0c39141fad8e89aa531dc2737fdc446c52dea9f.jpg)  
Figure 3: Expert roles and scientific tools for investi gation authoring.

EarthVerse couples expert authoring with a scientific tool suite (Figure 3). Experts frame and verify each investigation; numerical, atmospheric, and geospatial tools support executable analysis. Evaluation then scores source choice, intermediate work, and physical interpretation separately from the final answer.

Geospatial and deep-research systems. Recent systems bring these strands together through remote-sensing tools, coordinated geospatial operations, iterative retrieval, or domain-specific pretraining [17, 40–50]. EarthVerse evaluates the combined research problem: whether an agent can choose compatible evidence, use scientific tools appropriately, and maintain a coherent account as an event analysis develops.

## 3 The EarthVerse Benchmark

EarthVerse contains 405 investigations grounded in 199 documented disasters and extreme events, covering 19 hazard families and 6,709 local files. Each task follows the rhythm of real post-event inquiry: establish what happened, identify the observations that can resolve the question, test the relevant quantities and mechanisms, reconcile disagreement, and state what the evidence supports. Packages preserve the complexity of the event without adding large collections of unrelated files merely to obstruct retrieval.

Multi-source misalignment is part of the scientific problem. Reports, stations, satellite products, reanalyses, exposure layers, and impact records may use different windows, footprints, variables, and units. A solver must inspect these differences before choosing evidence, then manage the dependency between tools, sources, calculations, and physical reasoning. Computation is frequent but deliberately transparent: aggregation, normalization, ratios, thresholds, lags, spatial overlap, weighted indices, and bounded counterfactuals. EarthVerse does not test recall of undocumented formulas or background-heavy differential equations. It tests whether simple, checkable operations can support a complex investigation.

## 3.1 Design principles

Four rules govern task construction.

Event-scale scientific iteration. The solver first identifies the event window, geographic extent, hazard evolution, evidence families, and major inconsistencies. It then alternates between targeted evidence access, calculation, and physical interpretation, revising the event picture when an observation conflicts with the current account.

Evidence discovery. The prompt names the event and scientific decision but not the relevant files. A released task requires 7.26 distinct sources on average, selected from a package with a median of 34 candidates. Across the release, evidence comes from dozens of distinct product and API families rather than one fixed archive. Tasks that reveal a path, filename, or answer-identifying quantity are rejected.

Scientific checkability. Conclusions are scored with their supporting values, units, source roles, and mechanism tests. The benchmark tests whether an operation is attached to the right evidence, not whether specialized mathematics can be reproduced from memory.

Path flexibility under package grounding. Report-first and time-series-first investigations can both receive full credit if they establish the required quantities and mechanism. Versioned packages keep the evidence and computation reproducible.

## 3.2 Fidelity to investigative practice

Post-event studies usually begin with an archive rather than a curated figure. Analysts compare agency bulletins, station records at different aggregations, satellite and reanalysis products, impact databases, and incomplete metadata. One source may refine the time window; another may challenge the assumed mechanism; a third may reveal a spatial mismatch. EarthVerse reproduces this investigative sequence in a fixed, replayable setting. The scientific question defines the decision, answer units specify what must be established, and the rubric scores source choice, calculation, revision, and causal argument. The benchmark stays close to disaster practice without relying on live feeds or unstated specialist knowledge.

## 3.3 Task construction and answer contracts

Each event contributes two tasks, with a third for seven flagship events. The public prompt states a scientific decision but does not identify the relevant files. Every task includes an expert solution, structured ground truth, a deterministic compute\_gt.py program, and a task-specific process rubric.

The task and evaluation flow is

$$
\begin{array} { c } { { ( q _ { i } , P _ { i } ) \stackrel { \pi } { \longrightarrow } \tau _ { i } = ( a _ { 1 } , o _ { 1 } , \ldots , a _ { T } , o _ { T } ) \longrightarrow \hat { y } _ { i } , } } \\ { { E _ { i } = [ f ( \hat { y } _ { i } , y _ { i } ^ { \star } ) , g ( \tau _ { i } ; s _ { i } , B _ { i } ) ] . } } \end{array}\tag{1}
$$

For task i, the solver sees question $q _ { i }$ and evidence collection $P _ { i }$ . Policy π produces trajectory $\tau _ { i }$ and answer $\hat { y } _ { i }$ Function $f$ compares the answer with structured ground truth $y _ { i } ^ { \star } ; g$ evaluates the trajectory against expert analysis $s _ { i }$ and rubric $B _ { i }$ without prescribing a tool order. The release contains 10,879 answer units. We use units because a single choice cannot show whether a system recovered the correct window, ranking, quantity, mechanism, or qualification. Each unit is a nontrivial scientific obligation, and tasks can combine numerical results with orderings and open-ended conclusions. This remains discriminative when frontier models already perform extremely well on simpler multiple-choice and short-answer formats.

A worked example. One task examines the 2021 Pacific Northwest heat wave. The solver must distinguish a direct hourly record from a lower-resolution aggregate, identify the matching heat window, and compute heat and

![](images/56a6789635619a7c53970067362af411abe85abf472a235d6e9c2726ef4a04f2.jpg)  
Figure 4: Task interface. Metadata, observations, and an auditable research trajectory.

humidity statistics on the same temporal support. The arithmetic is elementary; the scientific work lies in choosing compatible observations and carrying that choice into the conclusion.

## 3.4 Capabilities and research environment

Capability coverage. Each task receives one to three manually reviewed capability labels: physical-mechanism reasoning, spatiotemporal reconstruction, quantitative calculation, multi-source synthesis, causal-chain reasoning, ranking or decision, and remote-sensing or geospatial interpretation. Labels may overlap. Conditioned scores are diagnostic and do not change the overall metric.

Figure 1(a) shows that these capabilities recur across hazard families rather than forming separate task silos. Figure 4 illustrates how a solver links task metadata, observations, and its research trajectory.

Research tools. We selected six general operations that are sufficient to express every released workflow: discover files, read a source, search local text, inspect structured data, execute scoped Python, and finalize the answer. This common interface makes system comparisons consistent while leaving the scientific strategy open. The registry also supports custom tool registration and retains 170 reusable operations for reports, time series, vector and exposure analysis, raster and Earth observation, meteorology and climate, fire and geophysics, infrastructure, and output assembly. Structured returns separate execution failure from scientific disagreement.

## 3.5 Quality assurance and benchmark comparison

Construction and review. Construction proceeds through event verification, package assembly, task authoring, executable validation, and release review (Figure 5). Automated tools support the process, but every released task is checked and corrected by people. Review covers event identity, source reliability, temporal and spatial fit, the public question, expert solution, ground truth, executable program, capability labels, and rubric. Package-local data must regenerate every deterministic value, and the visible assignment must match the hidden scoring target. Appendix C.6 describes the human and agent review process; Appendix C.7 records the seven release gates.

A separate LLM pairwise comparison offers a second check on score validity. Its win–tie–loss pattern broadly follow the Core leaderboard across system families (Figure 7). Systems with higher Core are also preferred when complete answers and traces are compared directly, so the ranking is not driven solely by the number of satisfied scoring fields.

Position relative to prior benchmarks. Table 1 compares what the model is asked to do, what evidence it receives, how experts participate, and what is scored. EarthSE, MSEarth, and GeoMMBench usually provide one selected passage, figure, or image [15–17]. ThinkGeo, Earth-Agent, OpenEarthAgent, and DORA add tools around declared imagery, assets, or layers [30, 40–42]. EarthVerse instead asks the system to construct and document a multi-source evidence base whose contents depend on the disaster and the scientific question.

Table 1: Where the scientific work begins. The two text columns identify the starting evidence and analytical work; marks show full, partial, or absent/unreported coverage.
<table><tr><td>Benchmark</td><td>Start evidence</td><td>Scientific work</td><td>Tools</td><td>Find evidence</td><td>Multi- source</td><td>Compute</td><td>Unit score</td><td>Trace score</td><td>Human review</td></tr><tr><td>EarthSE</td><td>Paper</td><td>Sci. QA</td><td>X</td><td>×</td><td>X</td><td>0</td><td>X</td><td>X</td><td>0</td></tr><tr><td>MSEarth</td><td>Figure</td><td>Figure QA</td><td>X</td><td>×</td><td>X</td><td>×</td><td>X</td><td>X</td><td></td></tr><tr><td>GeoMMBench</td><td>Image</td><td>Geo reasoning</td><td>X</td><td>×</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0</td></tr><tr><td>ThinkGeo</td><td>RS image</td><td>Tool use</td><td></td><td>0</td><td>o</td><td></td><td>X</td><td>√</td><td>0</td></tr><tr><td>Earth-Agent</td><td>EO assets</td><td>EO planning</td><td>V</td><td>0</td><td>V</td><td>V</td><td>X</td><td>√</td><td>0</td></tr><tr><td>OpenEarthAgent</td><td>Imagery</td><td>Multimodal</td><td></td><td>0</td><td>0</td><td></td><td>X</td><td></td><td>X</td></tr><tr><td>DORA</td><td>Layers</td><td>Crisis analysis</td><td></td><td>0</td><td>V</td><td></td><td>X</td><td></td><td>0</td></tr><tr><td>EarthVerse</td><td>Package</td><td>Event synthesis</td><td>V</td><td>V</td><td></td><td></td><td></td><td></td><td></td></tr></table>

✓ full partial × absent/unreported  
EarthVerse scale: 34 files/package; 7.26 sources and 26.86 units/task.

EarthVerse places the difficulty in scientific assembly rather than specialized mathematics. Fine-grained answer units state what must be established; the process rubric tests whether the chosen path supports those claims.

## 4 Evaluation Protocol

EarthVerse fixes the scientific question, visible evidence boundary, execution budget, and answer contract across systems. During a run, the system selects evidence, performs calculations, records provenance, and submits a structured scientific answer through the same interface. The full inference configuration, model modalities, and controlled-study manifests are reported in Appendix C.

## 4.1 Outcome, process, and reliability metrics

For task i, let $U _ { i }$ be the percentage of required answer units satisfied and $H _ { i }$ the holistic answer judgment, both on a 0–100 scale. Answer correctness is $A _ { i } = \operatorname* { m i n } ( H _ { i } , U _ { i } ) \colon$ : a fluent answer cannot outrank the scientific obligations it actually completes. Let $P _ { i }$ be the task-specific process-rubric score on the same scale. The primary score is

$$
{ \mathrm { C o m b i n e d C o r e } } = { \frac { 1 } { 2 N } } \sum _ { i = 1 } ^ { N } \left( A _ { i } + P _ { i } \right) .\tag{2}
$$

![](images/2154c7c782552b5766ce3219b1f6f789cd41e38244f16f9a3f47d6ff4443ea00.jpg)  
Figure 5: Benchmark construction. Event grounding, package assembly, investigation authoring, and expert audit.

It therefore gives equal weight to endpoint correctness and the documented research process.

Answer units are necessary because one investigation may require a time window, several quantities, a source comparison, and a physical mechanism. A single holistic grade would hide which obligation failed and could let a fluent conclusion mask missing scientific work.

Mean unit accuracy averages $U _ { i }$ across tasks and captures partial completion of required scientific claims and output fields; numerical units use task-specific tolerances. Strict@95 is the percentage of tasks with $U _ { i } \geq 9 5 $ and measures near complete task reliability rather than average progress. Capability-conditioned scores apply the same Core calculation to expert-tagged task slices. Trajectory statistics report interaction, evidence access, computation, latency, and token demand, but never enter the score. Solver failures receive zero; judge-service failures are rerun until a valid judgment is obtained.

Path pluralism and diagnostic validity. EarthVerse does not prescribe a tool order. Earth-Agent and DORA instead evaluate whether an agent follows a predefined tool or operational sequence. This is useful when a workflow has a canonical procedure, but complex multi-source investigations rarely admit a single defensible trajectory ground truth: report-first and time-series-first analyses may reach the same well-supported conclusion. EarthVerse therefore scores what the system establishes and whether the observed evidence and calculations support it. Tools and traces serve reasoning, review, and diagnosis; reproducing a prescribed path earns no credit by itself.

## 4.2 Evaluated systems

Table 2 compares hosted systems, open-weight models, and research frameworks through one package interface. Some backbones are text-only, while others support native multimodal input; the shared leaderboard does not grant extra evidence to either group. All systems receive the same question, package boundary, answer contract, and execution budget. Appendix C.1 gives the complete roster, modality treatment, controller exceptions, and judge configuration.

## 5 Main Evaluation

The results follow the three questions. The leaderboard and reliability diagnostics in Section 5.1 answer RQ1. Reasoning, interaction, and evidence-path interventions in Section 5.2 answer RQ2. Section 5.3 addresses RQ3 by comparing supplied-observation performance with evidence construction. Appendix B.1 develops the resulting position.

## 5.1 Overall system performance

RQ1 concerns completion at the investigation level. We begin by asking whether current systems can keep the whole account valid when most individual parts are correct.

Setting. Table 2 reports all 25 systems under the package-scoped interactive protocol. The Harness column names the controller used to run each model. Core averages Answer correctness and Process-rubric score; Strict@95 counts tasks satisfying at least 95% of answer units. Trajectory columns are per-task means and do not affect quality scores. Systems are grouped by family and ordered from lower to higher Core.

Table 2: EarthVerse leaderboard. Harness names the controller. Scores are percentages; failed, timed-out, or malformed runs remain zero, and batch reads may cover multiple files. Within each family, Core increases downward; bold blue and underlined teal mark the top two outcomes.
<table><tr><td colspan="2">System configuration</td><td colspan="5">Outcome</td><td colspan="7">Trajectory mean per task</td></tr><tr><td>System</td><td>Harness</td><td>Answer Process</td><td></td><td>Core</td><td>Strict@95 (%)</td><td>Unit acc.</td><td>Rounds</td><td>Tool calls</td><td>Reads</td><td>Files Python</td><td></td><td>Latency (s)</td><td>Tokens (k)</td></tr><tr><td colspan="10">Agent frameworks (own controllers over the same package interface)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenResearcher</td><td>OpenResearcher</td><td>30.51</td><td></td><td>31.0630.79</td><td>4.9432.74</td><td></td><td></td><td>39.73 33.62</td><td>15.00</td><td>17.03</td><td>5.38</td><td></td><td>538.7</td></tr><tr><td>GPT-5.5</td><td>GeoMMAgent</td><td>65.86</td><td></td><td>70.8468.35</td><td>12.3567.18</td><td></td><td>5.00</td><td>8.28</td><td>11.93</td><td>8.59</td><td>0.99</td><td>521 221</td><td>37.9</td></tr><tr><td colspan="10">Open-weight and Earth-specialized models</td><td colspan="3"></td><td></td><td></td></tr><tr><td>Intern-S1-mini</td><td>EarthVerse</td><td>10.78</td><td></td><td></td><td>0.25 11.60</td><td></td><td>4.99</td><td>3.43</td><td>1.80</td><td>2.39</td><td>0.07</td><td>57</td><td>39.4</td></tr><tr><td>Nemotron Nano</td><td>EarthVerse</td><td>12.30</td><td>12.6711.72 14.01 13.15</td><td></td><td>0.25</td><td>12.80</td><td>2.59</td><td>2.07</td><td>1.26</td><td>1.77</td><td>0.15</td><td>279</td><td>53.2</td></tr><tr><td>MiroThinker 8B</td><td>EarthVerse</td><td>12.79</td><td>13.6213.21</td><td></td><td>2.96</td><td>13.42</td><td>27.30</td><td>25.98</td><td>6.07</td><td>11.42</td><td>2.00</td><td>367</td><td>318.0</td></tr><tr><td>GeoGPT</td><td>EarthVerse</td><td>22.15</td><td>27.32 24.74</td><td></td><td>0.25</td><td>23.61</td><td>30.41</td><td>29.37</td><td>19.33</td><td>11.86</td><td>2.65</td><td>565</td><td>502.1</td></tr><tr><td>Qwen3-4B</td><td>EarthVerse</td><td>25.11</td><td>24.94 25.02</td><td></td><td>0.00</td><td>26.13</td><td>5.06</td><td>3.79</td><td>2.55</td><td>3.66</td><td>0.07</td><td>132</td><td>37.7</td></tr><tr><td>GeohazardGPT</td><td>EarthVerse</td><td>28.88</td><td>33.33 31.10</td><td></td><td>0.49</td><td>30.27</td><td>30.33</td><td>29.34</td><td>13.38</td><td>15.70</td><td>8.44</td><td></td><td>333.7</td></tr><tr><td>Qwen3-235B</td><td>EarthVerse</td><td>31.98</td><td>34.6433.31</td><td></td><td>0.99</td><td>32.66</td><td>13.47</td><td>12.80</td><td>5.33</td><td>7.41</td><td>3.50</td><td>456 195</td><td>274.3</td></tr><tr><td colspan="10">Hosted general-purpose systems</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.6 Luna</td><td>Codex</td><td>35.68</td><td>35.6335.66</td><td></td><td>11.11 36.28</td><td></td><td>6.58</td><td>5.39</td><td>2.37</td><td>4.29</td><td>1.05</td><td>189</td><td>156.9</td></tr><tr><td>GPT-40</td><td>EarthVerse</td><td>37.35</td><td>41.11 39.23</td><td></td><td>0.49</td><td>38.69</td><td>18.81</td><td>8.95</td><td>4.04</td><td>6.84</td><td>0.69</td><td>55</td><td>127.4</td></tr><tr><td>MiniMax M2.7</td><td>EarthVerse</td><td>43.66</td><td>45.7744.71</td><td></td><td>7.4144.35</td><td></td><td>17.27</td><td>16.08</td><td>7.11</td><td>8.94</td><td>2.22</td><td>205</td><td>285.8</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>EarthVerse</td><td>57.67</td><td>61.6659.66</td><td></td><td>2.9658.67</td><td></td><td>7.37</td><td>6.10</td><td>3.59</td><td>5.69</td><td>1.08</td><td>24</td><td>70.1</td></tr><tr><td>Doubao Seed 2.0 Mini</td><td>EarthVerse</td><td>61.14</td><td>64.8462.99</td><td></td><td>5.93 62.22</td><td></td><td>13.71</td><td>12.70</td><td>7.21</td><td>8.86</td><td>1.08</td><td>113</td><td>175.1</td></tr><tr><td>Qwen 3.7 Plus</td><td>EarthVerse</td><td>69.33</td><td>71.3770.35</td><td></td><td>18.2770.16</td><td></td><td>17.15</td><td>15.72</td><td>6.51</td><td>11.25</td><td>3.31</td><td>313</td><td>204.3</td></tr><tr><td>Člaude Haiku 4.5</td><td>EarthVerse</td><td>70.76</td><td>73.7072.23</td><td></td><td>14.07 71.70</td><td></td><td>21.31</td><td>19.86</td><td>10.08</td><td>11.72</td><td>3.96</td><td>324</td><td>477.0</td></tr><tr><td>Tencent Hy3</td><td>EarthVerse</td><td>71.27</td><td>74.2672.77</td><td></td><td>16.3072.22</td><td></td><td></td><td>19.4717.77</td><td>8.32</td><td>12.70</td><td>4.41</td><td>311</td><td>208.7</td></tr><tr><td>DeepSeek V4 Flash</td><td>EarthVerse</td><td>72.32</td><td>74.77 73.54</td><td></td><td>20.25</td><td>73.25</td><td>27.92</td><td>26.81</td><td>11.00</td><td>16.38</td><td>4.07</td><td>333</td><td>389.7</td></tr><tr><td>Kimi K2.5</td><td>EarthVerse</td><td>73.89</td><td>76.7675.32</td><td></td><td>14.57</td><td>74.70</td><td>24.25</td><td>22.67</td><td>10.01</td><td>16.00</td><td>4.79</td><td>439</td><td>302.9</td></tr><tr><td>Claude Sonnet 4.6</td><td>EarthVerse</td><td>74.88</td><td>77.4476.16</td><td></td><td>20.0075.76</td><td></td><td></td><td>17.5416.54</td><td>7.72</td><td>11.82</td><td>4.49</td><td>343</td><td>331.0</td></tr><tr><td>GPT-5.5</td><td>EarthVerse</td><td>76.85</td><td>79.2878.06</td><td></td><td>17.0477.48</td><td></td><td></td><td>15.1714.20</td><td>6.83</td><td>10.92</td><td>3.74</td><td>210</td><td>224.9</td></tr><tr><td>GLM-5.2</td><td>EarthVerse</td><td>78.14</td><td>80.1979.17</td><td></td><td>22.96</td><td>78.62</td><td>18.32</td><td>17.27</td><td>7.52</td><td>13.36</td><td>4.31</td><td>259</td><td>196.1</td></tr><tr><td>GPT-5.6 Terra</td><td>Codex</td><td>80.12</td><td>80.00 80.06</td><td></td><td>27.16</td><td>80.81</td><td>12.18</td><td>11.13</td><td>4.42</td><td>9.46</td><td>2.71</td><td>173</td><td>182.6</td></tr><tr><td>GPT-5.6 Sol</td><td>Codex</td><td>81.43</td><td>81.94 81.68</td><td></td><td>34.81</td><td>82.12</td><td>13.88</td><td>12.79</td><td>4.88</td><td>9.65</td><td>2.87</td><td>205</td><td>205.9</td></tr><tr><td>Claude Fable 5</td><td>Claude Code</td><td>83.99</td><td>85.95 84.97</td><td></td><td>25.93 84.65</td><td></td><td></td><td>14.4513.45</td><td>5.92 10.02</td><td></td><td>2.61</td><td>247</td><td>161.6</td></tr></table>

The landscape in Figure 1(b) shows that similar token budgets can produce sharply different answer-unit accuracy.

Mean competence versus conjunctive reliability. Claude Fable 5 leads on Core (84.97), while GPT-5.6 Sol leads on Strict@95 (34.81%). Fable satisfies 84.65% of answer units on average but reaches 95% completion on only 25.93% of tasks; Sol leaves fewer investigations with a consequential omission. Missing windows, provenance, alternative mechanisms, and unit conversions are coupled errors: a wrong window changes the calculation and its physical interpretation. Low Strict@95 rates show that current systems often know most of the analysis without keeping the whole analysis valid at once.

Finding. Reliability requires the whole evidence chain. High average accuracy can hide one missing source, window, unit, calculation, or mechanism. In disaster analysis, that single break can invalidate an otherwise plausible account.

Coupling reasoning with interaction. Most leaderboard systems use the EarthVerse harness, a common multiround controller configured for evidence discovery, calculation, provenance, recovery, and structured finalization.

Holding this scaffold fixed makes backbone comparisons fair, while framework rows retain their native controllers Interaction volume still does not predict quality. OpenResearcher uses 39.73 rounds and 538.7k tokens per task, yet trails Claude Fable 5 by 54.18 Core points because weak sources and stale values survive into late revisions. GeoMMAgent is too brief in the opposite direction: five rounds leave evidence undiscovered and Strict@95 reaches only 12.35%. Useful interaction updates the claim–evidence state and carries the revision into the answer.

Finding. Useful interaction changes the scientific state. A tool call matters when its observation changes the chosen source, calculation, or physical account. Longer traces that preserve the same mistake add cost, not quality.

Domain knowledge and frontier reasoning. Earth-specialized pretraining improves terminology and process recognition [48–50], but the evaluated open models remain 40–70 Core points behind frontier general systems. GeohazardGPT averages 30.33 rounds and 8.44 Python calls per task yet reaches 31.10 Core. The gap appears after recognition, in source selection, provenance, revision, and stopping. Domain training remains useful, but in these systems it has not yet produced a reliable research loop.

Capability slices in Appendix D.1 sharpen this result. Several systems rise above their own mean on calculation, spatiotemporal reconstruction, or remote sensing, yet fall on causal reasoning and ranking. Overall strength does not produce a uniform scientific profile.

Computational cost. Evaluating a frontier model costs more than \$2,500, while flash models remain below \$45. GPT-5.6 Terra retains 98% of GPT-5.6 Sol’s Core at less than half the cost. Cost depends on both model pricing and controller behavior, so neither trace length nor token count alone is an efficiency measure. Terra offers the strongest quality–cost balance among the top systems, whereas low-cost Flash models are useful for broad screening but do not match frontier reliability. Appendix D.3.3 reports the full comparison.

## 5.2 Reasoning, interaction, and evidence control

For RQ2, we separate additional deliberation from interventions that let the system revise its claim–evidence state. Tables 3 and 4 examine complementary parts of that loop: how evidence can be used during reasoning, and what the system can locate or act on.

Table 3 shows a sharp split between fixed and editable evidence. With compressed or all-text direct input, the best result comes from no additional reasoning; high effort lowers Core from 43.24 to 26.17 and from 29.44 to 8.90, respectively. Once the system can retrieve evidence and act on what it finds, Core rises from 51.30 with no extra effort to 73.80 at high effort. Final review changes the best operating point again: xhigh reaches 86.29, compared with 68.58 without review. Reasoning helps when it can trigger retrieval, recomputation, or correction. Deliberating longer over the same snapshot often deepens the initial mistake.

Resource use reinforces this distinction. High effort in the interactive setting reaches 73.80 Core with 14.96 rounds and 174.3k tokens; medium effort uses more rounds and tokens but reaches only 63.13. Additional steps help

Table 3: Reasoning and evidence access. GPT-5.5 across four access modes and five effort settings; diagnostics are per-task means.
<table><tr><td></td><td colspan="4">Outcome</td><td>Interaction</td><td colspan="2">Evidence</td><td colspan="2"></td><td colspan="2">Resources</td></tr><tr><td>Effort</td><td>Answer Process</td><td></td><td>Core</td><td>Unit acc.</td><td></td><td>Rounds Calls</td><td>Files</td><td>Refs</td><td>(k)</td><td>Tokens Latency</td><td>(s)</td></tr><tr><td colspan="10">Compressed direct</td></tr><tr><td>none</td><td>41.37</td><td>45.10 43.24 42.21</td><td></td><td></td><td></td><td></td><td></td><td></td><td>13.9</td><td></td><td>55.6</td></tr><tr><td>low</td><td>24.99</td><td></td><td></td><td>29.32 27.16 25.43</td><td></td><td></td><td></td><td></td><td></td><td>14.9</td><td>120.5</td></tr><tr><td>medium</td><td>22.88</td><td>25.69</td><td></td><td>24.2923.26</td><td></td><td></td><td></td><td></td><td></td><td>15.8</td><td>152.4</td></tr><tr><td>high</td><td>24.17</td><td>28.17</td><td></td><td>26.17 24.28</td><td></td><td></td><td></td><td></td><td></td><td>16.6</td><td>71.5</td></tr><tr><td>xhigh</td><td>16.67</td><td>16.67</td><td></td><td>16.67 16.67</td><td></td><td></td><td></td><td></td><td></td><td>17.6</td><td>88.1</td></tr><tr><td colspan="10">All-text direct</td></tr><tr><td>none</td><td>27.66</td><td></td><td></td><td>31.22 29.44 28.27</td><td></td><td></td><td></td><td></td><td></td><td>81.6</td><td>32.1</td></tr><tr><td>low</td><td>13.26</td><td></td><td></td><td>16.20 14.73 13.36</td><td></td><td></td><td></td><td></td><td></td><td>82.7</td><td>44.1</td></tr><tr><td>medium</td><td>9.79</td><td></td><td></td><td>12.18 10.99 10.05</td><td></td><td></td><td></td><td></td><td></td><td>83.5</td><td>61.0</td></tr><tr><td>high</td><td>8.80</td><td></td><td></td><td>9.00 8.90 8.82</td><td></td><td></td><td></td><td></td><td></td><td>85.4</td><td>78.5</td></tr><tr><td>xhigh</td><td>12.00</td><td></td><td></td><td>15.00 13.50 12.00</td><td></td><td></td><td></td><td></td><td></td><td>87.7</td><td>94.4</td></tr><tr><td colspan="10">EarthVerse interactive</td></tr><tr><td>none</td><td>48.89</td><td></td><td></td><td>53.70 51.3049.46</td><td>9.68</td><td>8.68</td><td>7.00</td><td>9.94</td><td></td><td>96.9</td><td>140</td></tr><tr><td>low</td><td>56.97</td><td></td><td></td><td>60.05 58.51 57.53</td><td>14.70</td><td>13.70</td><td>9.44</td><td>14.70</td><td></td><td>162.4</td><td>259</td></tr><tr><td>medium</td><td>61.45</td><td></td><td></td><td>64.80 63.13 62.35</td><td></td><td>17.2016.20</td><td>10.71</td><td>18.12</td><td></td><td>206.4</td><td>424</td></tr><tr><td>high</td><td>72.41</td><td></td><td></td><td>75.19 73.80 72.59</td><td></td><td>14.96 13.96</td><td></td><td>9.92 17.19</td><td></td><td>174.3</td><td>571</td></tr><tr><td>xhigh</td><td>69.66</td><td></td><td></td><td>67.50 68.58 69.83</td><td></td><td>11.25 10.25 10.25 16.50</td><td></td><td></td><td></td><td>100.0</td><td>674</td></tr><tr><td colspan="10">Interactive with final review</td></tr><tr><td>none</td><td>53.70</td><td>56.48 55.09 54.06</td><td></td><td></td><td>11.62</td><td>9.62</td><td></td><td>7.42 10.46</td><td></td><td>130.7</td><td>108</td></tr><tr><td>low</td><td>61.93</td><td></td><td></td><td>65.50 63.72 62.78</td><td></td><td>18.18 16.18 10.60 17.02</td><td></td><td></td><td></td><td>227.0</td><td>253</td></tr><tr><td>medium</td><td>67.16</td><td></td><td></td><td>69.22 68.19 67.91</td><td></td><td>20.81 18.81 11.88 21.02</td><td></td><td></td><td></td><td>266.3</td><td>552</td></tr><tr><td>high</td><td>69.82</td><td></td><td></td><td>74.71 72.27 70.73</td><td></td><td>20.63 18.63 12.13 21.10</td><td></td><td></td><td></td><td>255.1</td><td>1,105</td></tr><tr><td>xhigh</td><td>84.36</td><td></td><td></td><td>88.21 86.29 84.40</td><td></td><td>17.57 15.57 10.14 18.29</td><td></td><td></td><td></td><td>217.7</td><td>1,493</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

when they change the evidence set, repair a calculation, or resolve a conflict that reaches the final answer. Trace length alone says little about the quality of the investigation.

Final review also has a conditional effect. At xhigh, it raises Core from 68.58 to 86.29, but at high effort it changes 73.80 to 72.27. Review is most useful when the investigation still contains evidence dependencies or calculations that

can be checked. Once the claim–evidence state is coherent, another pass may add cost without improving the answer.   
This favors targeted verification over a fixed rule that every task needs more reasoning.

To locate the remaining bottlenecks, Table 4 uses answer-free oracles that remove one obstacle at a time: they reveal the relevant files, evidence roles, useful tool classes, or a high-level plan, but never the target answer. The same paired tasks vary tool presentation, package evidence, and access to meteorological operations. The meteorological conditions compare a fixed compact suite with a cue-routed subset; tool use remains optional, so this probe asks whether specialist access alone improves the investigation.

Table 4: Evidence-path interventions. Complete paired outcome, effect, and capability grid relative to the Top-20 router. Intervals are paired 95% bootstrap CIs.
<table><tr><td rowspan="2"></td><td colspan="2">Outcome</td><td colspan="2">Paired effect</td><td colspan="5">Core by capability</td></tr><tr><td>Answer Process</td><td>Core</td><td>∆ Core</td><td>95% CI</td><td>mech.</td><td>Physical Spatio- Quant. Multi- Causal Rank temporal</td><td>calc.</td><td>source chain</td><td>RS/ sort geo.</td></tr><tr><td>Top-20 semantic-router baseline</td><td>66.90</td><td>65.88 66.39</td><td>0.00</td><td></td><td>65.15</td><td>64.40</td><td>73.43</td><td>90.00</td><td>65.89 50.84 53.94</td></tr><tr><td colspan="10">Answer-free information oracles</td></tr><tr><td>Relevant-file oracle</td><td>79.46</td><td>82.75 81.11 +14.72 [+5.48, +24.69]</td><td></td><td></td><td>80.61</td><td>91.47</td><td>87.57</td><td>90.45</td><td>76.53 60.70 72.09</td></tr><tr><td>Evidence-map oracle</td><td>76.96</td><td></td><td></td><td>78.50 77.73 +11.34 [+2.34, +21.25]</td><td>76.69</td><td>90.50</td><td>83.83</td><td>97.50</td><td>73.35 58.16 61.25</td></tr><tr><td>Tool-subset oracle</td><td>68.53</td><td></td><td></td><td>68.62 68.58 +2.19 [-3.55, +7.28]</td><td>67.32</td><td>65.67</td><td>72.60</td><td>92.50</td><td>69.54 59.14 59.75</td></tr><tr><td>Planning oracle</td><td>65.19</td><td></td><td></td><td>66.50 65.84 -0.54 [−7.07, +4.35]</td><td>64.31</td><td>57.10</td><td>71.24</td><td>95.00</td><td>67.93 54.76 60.25</td></tr><tr><td colspan="10">Tool-interface presentation</td></tr><tr><td>Hierarchical catalog</td><td>66.76</td><td></td><td></td><td>67.62 67.19 +0.80 [−9.18, +11.86]</td><td>65.81</td><td>66.72</td><td></td><td>67.83 93.50</td><td>70.78 56.60 61.67</td></tr><tr><td>Descriptive tool names</td><td>66.88</td><td></td><td></td><td>66.75 66.82 +0.43 [-3.55, +4.05]</td><td>65.07</td><td>61.21</td><td></td><td>72.92 100.00</td><td>66.98 57.59 55.23</td></tr><tr><td>Dynamic descriptions</td><td>61.39</td><td></td><td></td><td>61.75 61.57 —4.82 [-15.02, +4.33]</td><td>59.81</td><td>61.56</td><td>63.81</td><td>95.00</td><td>61.27 54.58 60.75</td></tr><tr><td>Opaque tool names</td><td>56.72</td><td></td><td></td><td>59.50 58.11 -8.28 [-23.57, +6.33]</td><td>56.16</td><td>48.88</td><td>58.54</td><td>95.22</td><td>59.76 54.01 80.00</td></tr><tr><td colspan="10">Evidence perturbations</td></tr><tr><td>Equivalent unit change</td><td>67.61</td><td>71.12 69.37</td><td></td><td>+2.98 [-9.75, +15.53]</td><td>68.02</td><td>66.93</td><td>71.44</td><td>95.00</td><td>71.29 55.47 76.38</td></tr><tr><td>Added distractor files</td><td>65.91</td><td>67.75 66.83</td><td></td><td>+0.44 [-6.80, +7.27]</td><td>65.46</td><td>60.47</td><td>72.15</td><td>92.95</td><td>65.16 59.98 70.62</td></tr><tr><td>Missing evidence</td><td>63.97</td><td></td><td></td><td>64.75 64.36 -2.03 [-12.05, +7.08]</td><td>62.49</td><td>60.31</td><td></td><td>67.82 100.00</td><td>66.63 54.74 55.12</td></tr><tr><td>Truncated evidence</td><td>63.17</td><td></td><td></td><td>63.38 63.27 —3.12 [-14.13, +7.47]</td><td>61.78</td><td>55.67</td><td>70.53</td><td>91.70</td><td>64.24 41.52 68.92</td></tr><tr><td>Conflicting evidence</td><td>63.03</td><td></td><td></td><td>63.12 63.08 -3.31 [-12.05, +2.97]</td><td>61.50</td><td>67.97</td><td>66.63</td><td>92.95</td><td>60.46 55.19 54.50</td></tr><tr><td colspan="10">Optional meteorological access</td></tr><tr><td>Compact meteorology suite</td><td>62.69</td><td></td><td></td><td>65.38 64.03 -2.36 [−11.92, +5.89]</td><td>62.14</td><td></td><td></td><td>61.97 66.67 100.00</td><td>65.22 56.15 58.47</td></tr><tr><td>Cue-routed meteorology</td><td>57.22</td><td>59.88 58.55 -7.84 [-19.40, +3.93]</td><td></td><td></td><td>56.74</td><td></td><td></td><td>58.87 59.25 92.95</td><td>60.28 50.45 57.25</td></tr></table>

Finding. Reasoning needs an editable claim–evidence state. Additional reasoning helps when it can change the investigation. Interactive retrieval and final review let the system replace a weak source, recompute a quantity, and carry the correction into the final answer. With a fixed evidence snapshot, the same effort can reinforce an early error.

Table 4 identifies what changes that investigation. Relevant-file localization and the evidence map raise Core by 14.72 and 11.34 points, and they are the only interventions whose confidence intervals exclude zero. A curated tool subset adds 2.19 points, while the planning oracle slightly lowers performance. Renaming tools, normalizing units, and exposing optional meteorological tools also fail to reproduce the localization gain. More tools or more procedural guidance are therefore not enough. The agent must find the observations that determine the scientific object before it can reason reliably about that object.

The capability columns make the mechanism more precise. Relevant-file localization lifts spatiotemporal Core from 64.40 to 91.47, physical-mechanism Core from 65.15 to 80.61, and quantitative-calculation Core from 73.43 to 87.57. Multi-source performance changes little because the baseline already covers many sources; the gain comes from selecting and aligning the right ones. Missing, truncated, or conflicting evidence lowers overall quality, although the paired intervals remain wide. In this study, evidence localization is the clearest reproducible bottleneck, and downstream reasoning quality depends on whether that step succeeds.

## 5.3 Cross-benchmark transfer and benchmark adaptation

RQ3 separates the ability to interpret a supplied observation from the longer task of constructing and maintaining the evidence base.

We evaluate GPT-5.5 under the public protocols of Earth-Bench, EarthSE, MSEarth, and GeoMMBench. Each comparison begins with a direct response, then repairs clearly erroneous benchmark items or judgments, and finally applies the structured EarthVerse procedure. This procedure is answer-free. We summarize recurring question types across the benchmark and provide a general guide for identifying the relevant variables, choosing the necessary calculation, and checking whether the conclusion follows from the supplied observation. The guide contains no item answer, target label, or item-specific evidence. Figure 6 summarizes a benchmark-native score for each stage and includes the relevant published reference [16, 17]. Because the benchmarks define different metrics, the figure supports comparisons within each benchmark, not rankings across the four groups. Complete native metrics and source-paper rows accompany the release.

![](images/9a3c132e4b1dde37d15f8c086a96bc69e176d5f1b84f3664e0b38e5faaa7c543.jpg)  
Figure 6: Cross-benchmark protocol sensitivity. Benchmark-native scores across three GPT-5.5 stages and published references.

The broad pattern is consistent, although the source of the gain differs by benchmark. On Earth-Bench, direct GPT-5.5 already clears the published line and improves further when the answer contract is made explicit. EarthSE is less uniform: multiple-choice and free-response results are already high, while fill-in accuracy rises from 20.29 to 61.90 after schema and answer-granularity corrections. MSEarth and GeoMMBench show the clearest recovery from interface mismatch. Direct GPT-5.5 trails the displayed reference, whereas the corrected and EarthVerse stages close or reverse the gap.

The improvement comes from making the problem legible to the model, not from revealing a solution or teaching new Earth-science knowledge during evaluation. Two changes are especially clear: EarthSE fill-in accuracy rises from 20.29 to 61.90, while GeoMMBench accuracy rises from 84.52 to 98.01, above both the published GeoMMAgent result of 88.40 and the reported human reference of 86.50. Under their native metrics, the tested frontier model handles these supplied-observation tasks well once the question structure is explicit.

These results concern tasks in which a passage, figure, or named observation has already been selected and the required output is clear. Current LLMs and agents are less reliable when they must decide which sources matter, align records with different support, verify calculations, preserve provenance, and revise downstream conclusions. This difference explains why adapted performance can be high while EarthVerse Strict@95 remains below 35%.

Finding. Long-horizon evidence construction remains unresolved. Answer-free structural guidance brings a frontier model to or above published reference levels on supplied-observation benchmarks, including the displayed human lines. EarthVerse Strict@95 remains below 35% because the agent must construct, check, and revise the evidence base itself.

## 6 Conclusion

RQ1: scientific reliability depends on the complete claim–evidence state. Current systems often recover much of an analysis, yet one unsupported source, scale mismatch, or missing mechanism can change the result. Progress depends on preserving the links among claims, sources, scales, and calculations throughout the investigation.

RQ2: reasoning helps when control keeps the claim–evidence state editable. More effort over a fixed serialization does not recover interactive performance. Evidence localization and targeted review help when they let the system replace a weak source, recompute a quantity, and carry the correction into its final account. The controller must preserve claim–source bindings, check method applicability, and stop once the required evidence is complete.

RQ3: interpreting evidence is easier than constructing it. Within benchmarks that supply the observation, structured adaptation reaches or exceeds published reference lines. EarthVerse remains substantially harder because the system must decide what belongs in the analysis and keep the selected evidence compatible through calculation and synthesis.

EarthVerse contributes an executable benchmark and controlled diagnostics for multi-source hazard research. The leaderboard and interventions point to the same limitation: agents often recover valid local facts but fail to carry source changes and corrected quantities into the final account. By tracing that failure across evidence, computation, and reporting, EarthVerse connects model performance to the reliability of the resulting scientific analysis.

## Author Contributions

Zhiqing Cui led the project and was responsible for benchmark conceptualization, methodology, data curation, experimentation, and writing the original manuscript. Xinxiang Yin and Xinglang Zhang developed the figures and visual presentation. Yuanzhe Hu and Weidong Tang coordinated review and quality control. Siru Zhong contributed to conceptual development and scientific discussion. Yihong Tang contributed methodological guidance, benchmark conceptualization, and scientific discussion. Dingyi Zhuang provided senior supervision, methodological guidance, and funding acquisition. Jinhua Zhao provided overall supervision, and project leadership. Yuxuan Liang, Weijia Li, Ming Jin, Shirui Pan, and Yuhao Kang contributed scientific discussion and critical review. All authors reviewed and approved the manuscript.

## References

[1] World Meteorological Organization. Atlas of mortality and economic losses from weather, climate and waterrelated hazards, 1970–2021. Technical report, World Meteorological Organization, 2023. URL https://public .wmo.int/publication-series/atlas-of-mortality-and-economic-losses-from-weather-cli mate-and-water-related-hazards-1970-2021.

[2] World Meteorological Organization. State of the global climate 2024. Technical Report WMO-No. 1368, World Meteorological Organization, 2025. URL https://wmo.int/publication-series/state-of-global-c limate/state-of-global-climate-2024.

[3] Intergovernmental Panel on Climate Change (IPCC). Climate Change 2022: Impacts, Adaptation and Vulnerability. Contribution of Working Group II to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press, 2022. doi: 10.1017/9781009325844. URL https://www.ipcc.ch/repo rt/ar6/wg2/.

[4] Jakob Zscheischler, Seth Westra, Bart J. J. M. van den Hurk, Sonia I. Seneviratne, Philip J. Ward, Andy Pitman, Amir AghaKouchak, David N. Bresch, Michael Leonard, Thomas Wahl, and Xuebin Zhang. Future climate risk from compound events. Nature Climate Change, 8(6):469–477, 2018. doi: 10.1038/s41558-018-0156-3.

[5] Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens, Martin Jung, Joachim Denzler, Nuno Carvalhais, and Prabhat. Deep learning and process understanding for data-driven Earth System Science. Nature, 566:195–204, 2019. doi: 10.1038/s41586-019-0912-1.

[6] Yannis Markonis, Christoforos Pappas, Martin Hanel, and Simon Michael Papalexiou. A cross-scale framework for integrating multi-source data in Earth System Sciences. Environmental Modelling & Software, 139:104997, 2021. doi: 10.1016/j.envsoft.2021.104997.

[7] Andrew Gettelman, Alan J. Geer, Richard M. Forbes, Greg R. Carmichael, Graham Feingold, Derek J. Posselt, Graeme L. Stephens, Susan C. van den Heever, Adam C. Varble, and Paquita Zuidema. The future of Earth System Prediction: Advances in model–data fusion. Science Advances, 8(14):eabn3488, 2022. doi: 10.1126/sciadv.abn34 88.

[8] Zhiqing Cui, Binwu Wang, Qingxiang Liu, Yeqiang Wang, Zhengyang Zhou, Yuxuan Liang, and Yang Wang. Augur: Modeling covariate causal associations in time series via large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 764–787. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.32. URL https://aclanthology.org/2026.acl-long.32/.

[9] Aaron Bell, Amit Aides, Amr Helmy, Arbaaz Muslim, Aviad Barzilai, Aviv Slobodkin, Bolous Jaber, et al. Earth AI: Unlocking geospatial insights with foundation models and cross-modal reasoning. arXiv preprint arXiv:2510.18318, 2025.

[10] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE\_vluYUL-X.

[11] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, et al. ScienceAgentBench: Toward rigorous assessment of language agents for data-driven scientific discovery. In International Conference on Learning Representations, 2025. URL https://openreview.net/f orum?id=6z4YKr0GK6.

[12] Ming Hu, Chenglong Ma, Wei Li, Wanghan Xu, Jiamin Wu, Jucheng Hu, Tianbin Li, Guohang Zhuang, Jiaqi Liu, Yingzhou Lu, et al. A survey of scientific large language models: From data foundations to agent frontiers. arXiv preprint arXiv:2508.21148, 2025.

[13] Niloufar Alipour Talemi, Julia Boone, and Fatemeh Afghah. Agentic AI in remote sensing: Foundations, taxonomy, and emerging systems. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops, pages 786–799, 2026. doi: 10.1109/WACVW68408.2026.00088. URL https: //openaccess.thecvf.com/content/WACV2026W/GeoCV/html/Talemi\_Agentic\_AI\_in\_Remote\_Sen sing\_Foundations\_Taxonomy\_and\_Emerging\_Systems\_WACVW\_2026\_paper.html.

[14] Zicheng Zhang, Junying Wang, Farong Wen, Yijin Guo, Xiangyu Zhao, Xinyu Fang, Shengyuan Ding, Ziheng Jia, Jiahao Xiao, Ye Shen, et al. Large multimodal models evaluation: a survey. Science China Information Sciences, 68(12):221301, 2025. doi: 10.1007/s11432-025-4676-4.

[15] Wanghan Xu, Xiangyu Zhao, Yuhao Zhou, Xiaoyu Yue, Ben Fei, Fenghua Ling, Wenlong Zhang, and Lei Bai. EarthSE: A benchmark evaluating earth scientific exploration capability for large language models. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=jyYE06FL8G.

[16] Xiangyu Zhao, Wanghan Xu, Bo Liu, Yuhao Zhou, Fenghua Ling, Ben Fei, Xiaoyu Yue, Lei Bai, Wenlong Zhang, and Xiao-Ming Wu. MSEarth: A multimodal benchmark for earth science phenomenon discovery with MLLMs. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5270–5301, 2026. doi: 10.18653/v1/2026.acl-long.239. URL https: //aclanthology.org/2026.acl-long.239/.

[17] Aoran Xiao, Shihao Cheng, Yonghao Xu, Yexian Ren, Hongruixuan Chen, and Naoto Yokoya. GeoMMBench and GeoMMAgent: Toward expert-level multimodal intelligence in geoscience and remote sensing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 34843–34853, 2026.

[18] Chenyue Li, Wen Deng, Mengqian Lu, and Binhang Yuan. AtmosSci-Bench: evaluating the recent advance of large language model for atmospheric science. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-4850.

[19] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Licheng Zhang, Xiaorui Wang, and Zhendong Mao. DeepResearch Bench: A comprehensive benchmark for deep research agents. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=hQ0K2Hhq7H.

[20] Yigeng Jiang, Tengchao Yang, Taoyong Cui, Jiaxing Wan, Yuan Wang, Weida Wang, Zhiyu Liu, Chuyi Peng, Binzhao Luo, Maoli Gao, et al. Deep research in physical sciences: A multi-agent framework and comprehensive benchmark. arXiv preprint arXiv:2606.18648, 2026.

[21] Muhammad Sohail Danish, Muhammad Akhtar Munir, Syed Roshaan Ali Shah, Kartik Kuckreja, Fahad Shahbaz Khan, Paolo Fraccaro, Alexandre Lacoste, and Salman Khan. GEOBench-VLM: Benchmarking vision-language models for geospatial tasks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 7132–7142, 2025. doi: 10.1109/ICCV51701.2025.00670.

[22] Fengxiang Wang, Hongzhen Wang, Zonghao Guo, Di Wang, Yulin Wang, Mingshuo Chen, Qiang Ma, Long Lan, Wenjing Yang, Jing Zhang, et al. XLRS-Bench: Could your multimodal LLMs understand extremely large ultra-high-resolution remote sensing imagery? In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14325–14336, 2025. doi: 10.1109/CVPR52734.2025.01336.

[23] Daniela Szwarcman, Sujit Roy, Paolo Fraccaro, Þorsteinn Elí Gíslason, Benedikt Blumenstiel, Rinki Ghosal, Pedro Henrique de Oliveira, João Lucas de Sousa Almeida, Rocco Sedona, Yanghui Kang, et al. Prithvi-EO-2.0: A versatile multitemporal foundation model for Earth Observation applications. IEEE Transactions on Geoscience and Remote Sensing, 64:1–20, 2026. doi: 10.1109/TGRS.2025.3642610. URL https: //ieeexplore.ieee.org/document/11296896/.

[24] Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp, et al. TerraMind: Large-scale generative multimodality for earth observation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 7383–7394, 2025. doi: 10.1109/ICCV51701.2025.00693.

[25] Naomi Simumba, Nils Lehmann, Paolo Fraccaro, Hamed Alemohammad, Geeth De Mel, Salman Khan, Manil Maskey, Nicolas Longepe, Xiao Xiang Zhu, Hannah Kerner, et al. GEO-Bench-2: From performance to capability, rethinking evaluation in geospatial AI. arXiv preprint arXiv:2511.15658, 2025.

[26] Zhiqing Cui, Siru Zhong, Ming Jin, Shirui Pan, Qingsong Wen, and Yuxuan Liang. Breaking the regional barrier: Inductive semantic topology learning for worldwide air quality forecasting. arXiv preprint arXiv:2601.21899, 2026. doi: 10.48550/arXiv.2601.21899. URL https://arxiv.org/abs/2601.21899.

[27] Firoj Alam, Ferda Ofli, and Muhammad Imran. CrisisMMD: Multimodal Twitter datasets from natural disasters. In Proceedings of the International AAAI Conference on Web and Social Media, volume 12, pages 465–473. AAAI Press, 2018. doi: 10.1609/icwsm.v12i1.14983.

[28] Ritwik Gupta, Bryce Goodman, Nirav Patel, Ricky Hosfelt, Sandra Sajeev, Eric T. Heim, Jigar Doshi, Keane Lucas, Howie Choset, and Matthew E. Gaston. Creating xBD: A dataset for assessing building damage from satellite imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 10–17, 2019. URL https://openaccess.thecvf.com/content\_CVPRW\_2019/html/cv 4gc/Gupta\_Creating\_xBD\_A\_Dataset\_for\_Assessing\_Building\_Damage\_from\_Satellite\_CVPRW\_ 2019\_paper.html.

[29] Maryam Rahnemoonfar, Tashnim Chowdhury, Argho Sarkar, Debvrat Varshney, Masoud Yari, and Robin Roberson Murphy. FloodNet: A high resolution aerial imagery dataset for post flood scene understanding. IEEE Access, 9: 89644–89654, 2021. doi: 10.1109/ACCESS.2021.3090981.

[30] Junjue Wang, Weihao Xuan, Heli Qi, Pengyu Dai, Kunyi Liu, Hongruixuan Chen, Zhuo Zheng, Junshi Xia, Stefano Ermon, and Naoto Yokoya. Can LLM agents respond to disasters? benchmarking heterogeneous geospatial reasoning in emergency operations. arXiv preprint arXiv:2605.11633, 2026.

[31] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36, pages 68539–68551, 2023. doi: 10.52202/07528 0-2997.

[32] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=dHng2O0Jjr.

[33] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. AgentBench: Evaluating LLMs as agents. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=zAdUB0aCTQ.

[34] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: A benchmark for general AI assistants. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=fibxvahvs3.

[35] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems, volume 37, pages 52040–52094, 2024. doi: 10.52202/079017-1650.

[36] Yubo Ma, Zhibin Gou, Junheng Hao, Ruochen Xu, Shuohang Wang, Liangming Pan, Yujiu Yang, Yixin Cao, and Aixin Sun. SciAgent: Tool-augmented language models for scientific reasoning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15701–15736, 2024. doi: 10.18653/v1/2024.emnlp-main.880.

[37] Yujiong Shen, Yajie Yang, Zhiheng Xi, Binze Hu, Huayu Sha, Qiyuan Peng, Jiazheng Zhang, Junlin Shang, Jixuan Huang, Yutao Fan, Jingqi Tong, Ming Zhang, Shihan Dou, Zhenfei Yin, Xingjun Ma, Lei Bai, Tao Gui, Qi Zhang, Xuanjing Huang, and Yu-Gang Jiang. SciAgentGym: Benchmarking multi-step scientific tool-use in LLM agents. In International Conference on Machine Learning, 2026. URL https://icml.cc/virtual/2026/poster/ 66785.

[38] Ken Gu, Ruoxi Shang, Ruien Jiang, Keying Kuang, Richard-John Lin, Donghe Lyu, Yue Mao, Youran Pan, Teng Wu, Jiaqian Yu, Yikun Zhang, Tianmai M. Zhang, Lanyi Zhu, Mike A. Merrill, Jeffrey Heer, and Tim Althoff. BLADE: Benchmarking language model agents for data-driven science. In Findings of the Association for

Computational Linguistics: EMNLP 2024, pages 13936–13971, 2024. doi: 10.18653/v1/2024.findings-emnlp.815. URL https://aclanthology.org/2024.findings-emnlp.815/.

[39] Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards datadriven discovery with large language models. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/0d70af566e69f1dfb687791e cf955e28-Abstract-Conference.html.

[40] Akashah Shabbir, Muhammad Akhtar Munir, Akshay Dudhane, Muhammad Umer Sheikh, Muhammad Haris Khan, Paolo Fraccaro, Juan Bernabe Moreno, Fahad Shahbaz Khan, and Salman Khan. ThinkGeo: Evaluating tool-augmented agents for remote sensing tasks. arXiv preprint arXiv:2505.23752, 2025.

[41] Peilin Feng, Zhutao Lv, Junyan Ye, Xiaolei Wang, Xinjie Huo, Jinhua Yu, Wanghan Xu, Wenlong Zhang, Lei Bai, Conghui He, and Weijia Li. Earth-Agent: Unlocking the full landscape of Earth Observation with agents. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_ files/paper/2026/hash/5b4a459db23e6db9be2a128380953d96-Abstract-Conference.html.

[42] Akashah Shabbir, Muhammad Umer Sheikh, Muhammad Akhtar Munir, Hiyam Debary, Mustansar Fiaz, Muham mad Zaigham Zaheer, Paolo Fraccaro, Fahad Shahbaz Khan, Muhammad Haris Khan, Xiao Xiang Zhu, et al. OpenEarthAgent: A unified framework for tool-augmented geospatial agents. arXiv preprint arXiv:2602.17665, 2026.

[43] Riyang Bao, Cheng Yang, Dazhou Yu, Zhexiang Tang, Gengchen Mai, and Liang Zhao. Spatial-Agent: Agentic geo-spatial reasoning with scientific core concepts. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14896–14911, 2026. doi: 10.18653/v1/2026.acl-l ong.679. URL https://aclanthology.org/2026.acl-long.679/.

[44] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deep-Researcher: Scaling deep research via reinforcement learning in real-world environments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 414–431, 2025. doi: 10.18653/v1/2025.emnlp-main.22. URL https://aclanthology.org/2025.emnlp-main.22/.

[45] Zhuofeng Li, Dongfu Jiang, Xueguang Ma, Haoxiang Zhang, Ping Nie, Yuyu Zhang, Kai Zou, Jianwen Xie, Yu Zhang, and Wenhu Chen. OpenResearcher: A fully open pipeline for long-horizon deep research trajectory synthesis. arXiv preprint arXiv:2603.20278, 2026.

[46] MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, Wenhan Dou, et al. MiroThinker: Pushing the performance boundaries of open-source research agents via model, context, and interactive scaling. arXiv preprint arXiv:2511.11793, 2025.

[47] Tao Yu, Yiming Ding, Shenghua Chai, Minghui Zhang, Zhongtian Luo, Xinming Wang, Xinlong Chen, Zhaolu Kang, Junhao Gong, Yuxuan Zhou, Haopeng Jin, Zhiqing Cui, Jiabing Yang, YiFan Zhang, Hongzhu Yi, Zheqi He, Xi Yang, Yan Huang, and Liang Wang. Omni-DeepSearch: A benchmark for audio-driven omni-modal deep search. arXiv preprint arXiv:2605.08762, 2026. URL https://arxiv.org/abs/2605.08762.

[48] Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-S1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763, 2025.

[49] Yifan Zhang, Cheng Wei, Zhengting He, and Wenhao Yu. GeoGPT: An assistant for understanding and processing geospatial tasks. International Journal ofApplied Earth Observation and Geoinformation, 131:103976, 2024. doi: 10.1016/j.jag.2024.103976.

[50] Qi Ge, Pengfa Li, Yinhao Dai, Jin Li, Ni An, Yang Yu, Qing Lv, and Hongyue Sun. GeohazardGPT: Towards large language models for geohazards. Under review, 2025.

[51] OpenAI. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[52] OpenAI. GPT-5.5 system card. https://openai.com/index/gpt-5-5-system-card/, 2026.

[53] OpenAI. GPT-5.6 system card. https://deploymentsafety.openai.com/gpt-5-6, 2026.

[54] Anthropic. Model system cards. https://www.anthropic.com/system-cards, 2026.

[55] Google DeepMind. Gemini 3.1 Flash-Lite. Google DeepMind model card, https://deepmind.google/mode ls/model-cards/gemini-3-1-flash-lite/, 2026.

[56] MiniMax. MiniMax M2.7: Early echoes of self-evolution. https://www.minimax.io/news/minimax-m27 -en, 2026.

[57] ByteDance Seed Team. Seed 2.0 official launch. https://seed.bytedance.com/en/blog/seed2-0-%25E6 %25AD%25A3%25E5%25BC%258F%25E5%258F%2591%25E5%25B8%2583, 2026.

[58] Qwen. Qwen3.7-Plus. Official model directory, https://chat.qwen.ai/legal-agreement/models, 2026.

[59] Tencent. Tencent unveils Hy3 preview; model enhances agent capabilities and real-world usability. https: //www.tencent.com/en-us/articles/2202320.html, 2026.

[60] DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

[61] Kimi Team. Kimi K2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[62] Z.ai. GLM-5.2: Built for long-horizon tasks. https://z.ai/blog/glm-5.2, 2026.

[63] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[64] NVIDIA. NVIDIA Nemotron Nano 2: An accurate and efficient hybrid mamba-transformer reasoning model. arXiv preprint arXiv:2508.14444, 2025.

## Appendix Guide

## Appendix A Controlled Studies

How reasoning, evidence access, memory, tools, and execution conditions change research quality. Answer mode • reasoning effort • routing • context • robustness • scientific environments

## Appendix B Discussion, Broader Impact, and Limitations

What the results imply for scientific agents, disaster analysis, oversight, and responsible use. Research position • broader impact • system design • limitations

## Appendix C Experimental Setup and Reproducibility

How models, modalities, inference, scoring, study manifests, and release decisions are controlled. System roster • modality interface • main configuration • sampling • audit

## D <sup>Appendix</sup> <sup>D</sup> <sup>Extended</sup> <sup>Diagnostics</sup>

Where capability, trajectory, error-path, resource, intervention, and tool-compliance analyses are reported. Capability scores • error paths • costs • ablations • uncertainty

## E <sup>Appendix</sup> <sup>E</sup> <sup>Prompts</sup> <sup>and</sup> <sup>Execution</sup> <sup>Contracts</sup>

What the solver, verifier, judge, and benchmark-construction reviewers are instructed to do. Runtime visibility • solver contract • final review • judging • construction review

##

Which systems and tasks were selected to represent the benchmark’s main scientific workflows and failure modes. Five successful cases • one partial failure • system family • task scope • score

## G <sup>Appendix</sup> <sup>G</sup> <sup>Detailed</sup> <sup>Research</sup> <sup>Trajectories</sup>

How selected systems discover evidence, calculate results, recover from errors, and finalize an answer. Full task • ordered trace • computed ledger • evidence reconciliation • failure analysis

Reading routes

## A Controlled Studies

RQ2 asks when reasoning improves an investigation and which controls preserve or break scientific continuity. We therefore vary answer mode, reasoning effort, evidence localization, tool routing, context, stopping, and professional scientific execution. Unless noted, every condition uses GPT-5.5 under the protocol in Section 4.2, with identical task manifests within each experiment.

## A.1 Reasoning effort and answer mode

We cross four answer modes with five reasoning settings on a fixed, difficulty-stratified manifest. Compressed direct uses a bounded evidence summary, while all-text direct serializes every readable file. Both interactive conditions use the same EarthVerse harness; the review variant adds one evidence-and-calculation audit before finalization. The compact view below retains the outcome columns so that each reasoning curve can be compared without the denser interaction and resource diagnostics in Table 3.

Table 5: Outcome view of the reasoning sweep. GPT-5.5; bold blue and shading mark the best setting in each access mode.  
Compressed direct (selected evidence)
<table><tr><td colspan="4"></td><td rowspan="2">Unit acc.</td></tr><tr><td>Effort</td><td>Answer</td><td>Process</td><td>Core</td></tr><tr><td>none</td><td>41.37</td><td>45.10</td><td>43.24</td><td>42.21</td></tr><tr><td>low</td><td>24.99</td><td>29.32</td><td>27.16</td><td>25.43</td></tr><tr><td>medium</td><td>22.88</td><td>25.69</td><td>24.29</td><td>23.26</td></tr><tr><td>high</td><td>24.17</td><td>28.17</td><td>26.17</td><td>24.28</td></tr><tr><td>xhigh</td><td>16.67</td><td>16.67</td><td>16.67</td><td>16.67</td></tr><tr><td colspan="5">EarthVerse interactive</td></tr><tr><td>Effort</td><td>Answer</td><td>Process</td><td>Core</td><td>Unit acc.</td></tr><tr><td>none</td><td>48.89</td><td>53.70</td><td>51.30</td><td>49.46</td></tr><tr><td>low</td><td>56.97</td><td>60.05</td><td>58.51</td><td>57.53</td></tr><tr><td>medium</td><td>61.45</td><td>64.80</td><td>63.13</td><td>62.35</td></tr><tr><td>high</td><td>72.41</td><td>75.19</td><td>73.80</td><td>72.59</td></tr><tr><td>xhigh</td><td>69.66</td><td>67.50</td><td>68.58</td><td>69.83</td></tr></table>

All-text direct (budgeted evidence)
<table><tr><td>Effort</td><td>Answer</td><td>Process</td><td>Core</td><td>Unit acc.</td></tr><tr><td>none</td><td>27.66</td><td>31.22</td><td>29.44</td><td>28.27</td></tr><tr><td>low</td><td>13.26</td><td>16.20</td><td>14.73</td><td>13.36</td></tr><tr><td>medium</td><td>9.79</td><td>12.18</td><td>10.99</td><td>10.05</td></tr><tr><td>high</td><td>8.80</td><td>9.00</td><td>8.90</td><td>8.82</td></tr><tr><td>xhigh</td><td>12.00</td><td>15.00</td><td>13.50</td><td>12.00</td></tr></table>

<table><tr><td colspan="5">EarthVerse interactive + review</td></tr><tr><td>Effort</td><td>Answer</td><td>Process</td><td>Core</td><td>Unit acc.</td></tr><tr><td>none</td><td>53.70</td><td>56.48</td><td>55.09</td><td>54.06</td></tr><tr><td>low</td><td>61.93</td><td>65.50</td><td>63.72</td><td>62.78</td></tr><tr><td>medium</td><td>67.16</td><td>69.22</td><td>68.19</td><td>67.91</td></tr><tr><td>high</td><td>69.82</td><td>74.71</td><td>72.27</td><td>70.73</td></tr><tr><td>xhigh</td><td>84.36</td><td>88.21</td><td>86.29</td><td>84.40</td></tr></table>

## A.2 Tools, context, and interaction budget

On one fixed paired manifest, we vary catalog size, semantic routing, tool composition, context retention, and interaction budget. Every condition uses the default medium reasoning setting and the same answer and process scores. Appendix D.4 reports trajectory diagnostics.

## A.2.1 Tool-catalog scale

Nested fixed catalogs expose increasingly broad subsets of the same 170-tool registry (panel (a) of Table 6). The six-tool condition contains package discovery, reading, search, tabular inspection, Python execution, and finalization; each larger catalog adds domain tools without changing the task prompt.

The full catalog loses 9.80 Core points relative to the six-tool foundation while using 19% more tokens. The decline is not monotonic, which rules out a simple capacity penalty. Extra tools add nearby analytical paths with different assumptions about variables, units, aggregation, and applicability. When the controller cannot retain why it chose one path, later outputs from another tool can enter the same ledger as if they were comparable. Catalog breadth then creates scientific ambiguity. Reporting a tool count without the navigation policy misses this failure mode.

## A.2.2 Semantic routing and scientific composition

The routing study selects tools from task text and package metadata before interaction (panel (b) of Table 6). Top-k routing exposes the highest-scoring individual tools, whereas workflow routing first selects a domain workflow. A separate composition study compares an expert-only suite, a mixed general/expert suite, and atmospheric Python and Fortran suites.

Top-20 routing recovers 11.21 Core points over the full catalog with 34% fewer tokens. Its advantage comes from narrowing the next choice without deciding the investigation in advance. Workflow routing is more brittle because it can commit to a domain before the evidence warrants that commitment; the expert-only suite has the opposite problem, lacking ordinary file and calculation operations that connect specialist output to a scored claim. Routing should act as a revisable prior: preserve a small general core, introduce specialist methods when their required variables appear, and allow the evidence to overturn the initial route.

## A.2.3 Context retention

We vary the amount of tool history preserved verbatim before older content is summarized (panel (c) of Table 6).   
Aggressive compression keeps a narrow recent window; short and long policies progressively retain more observations;   
full history disables summarization within the tested budget.

Long bounded retention gains 13.66 Core points over aggressive compression and 5.27 over full replay. Too little history forces the agent to rediscover files and often severs intermediate values from their provenance. Full replay retains rejected hypotheses and superseded numbers alongside active values, so discarded information can reappear in later calculations. Scientific memory therefore needs a compact claim–evidence state that binds claims to sources, units, temporal and spatial support, and unresolved conflicts while archiving replaced values.

Finding. Scientific memory is selective. Long bounded retention beats both aggressive compression and full replay. The useful state is a compact ledger of active claims, sources, units, and unresolved conflicts, not a verbatim transcript.

## A.2.4 Interaction budget

The final study changes only the EarthVerse-harness turn limit. The unbounded condition still uses the common wall-clock budget, so it tests productive stopping rather than unlimited execution.

Twelve turns usually end before discovery, calculation, and verification can all occur. Additional turns improve mean quality, but the gain depends on what those turns accomplish. Extra interaction helps when it closes a named evidence gap and hurts when the agent reopens a settled calculation or follows an unproductive branch. Stopping is therefore a coverage decision: the system should stop when required claims are supported, units and provenance are checked, and material contradictions have been resolved or disclosed.

Table 6: System ablations. GPT-5.5 at medium reasoning; bold blue marks panel bests.
<table><tr><td></td><td colspan="3">Outcome</td><td>Coverage</td><td colspan="6">Core by capability</td><td>Resource</td><td></td></tr><tr><td>Condition</td><td>Answer Process</td><td></td><td>Core</td><td>Unit acc.</td><td>Physical mech.</td><td>Spatio- temporal</td><td>Quant. Multi- Causal calc.</td><td>source</td><td>chain</td><td>Ranking</td><td>RS/ geo.</td><td>Tokens</td></tr><tr><td colspan="9">(a) Tool catalog: nested subsets of 170 tools</td><td></td><td></td><td></td><td></td></tr><tr><td>Fixed tools (6)</td><td>64.34</td><td>65.62</td><td>64.98</td><td>64.69</td><td>63.27</td><td>66.22</td><td>70.34</td><td>97.50</td><td>63.35</td><td></td><td>51.78 58.12</td><td>229,557</td></tr><tr><td>Fixed tools (10)</td><td>60.22</td><td>57.10</td><td>58.66</td><td>60.35</td><td>56.74</td><td>58.36</td><td>61.04</td><td>95.22</td><td>55.77</td><td></td><td>58.32 61.25</td><td>194,047</td></tr><tr><td>Fixed tools (20)</td><td>60.58</td><td></td><td>59.38 59.98</td><td>60.93</td><td>58.13</td><td>70.50</td><td>61.67</td><td>95.00</td><td>54.90</td><td></td><td>56.5055.62</td><td>232,912</td></tr><tr><td>Fixed tools (40)</td><td>58.55</td><td></td><td>61.0059.77</td><td>59.43</td><td>58.16</td><td>47.81</td><td>62.03</td><td>90.45</td><td>63.63</td><td></td><td>59.57 54.12</td><td>232,993</td></tr><tr><td>Fixed tools (70)</td><td>58.76</td><td>61.25</td><td>60.01</td><td>59.19</td><td>58.54</td><td>55.68</td><td>63.89</td><td>87.95</td><td>60.14</td><td></td><td>54.21 54.62</td><td>233,759</td></tr><tr><td>Fixed tools (100)</td><td>63.96</td><td></td><td>63.12 63.54</td><td>64.11</td><td>61.81</td><td>58.93</td><td>65.84</td><td>96.37</td><td>65.79</td><td></td><td>60.82 51.00</td><td>278,936</td></tr><tr><td>Full catalog (170)</td><td>54.48</td><td></td><td>55.8855.18</td><td>55.46</td><td>55.58</td><td>55.72</td><td>54.96</td><td>47.50</td><td>52.75</td><td></td><td>63.15 55.00</td><td>273,216</td></tr><tr><td colspan="9">(b) Routing and tool composition</td><td></td><td></td><td></td><td></td></tr><tr><td>Top-10 semantic router</td><td>64.12</td><td></td><td>66.25 65.19</td><td>64.55</td><td>64.37</td><td>60.64</td><td>67.93</td><td>80.74</td><td>67.50</td><td></td><td>57.2059.17</td><td>216,227</td></tr><tr><td>Top-20 semantic router</td><td>66.90</td><td>65.88</td><td>66.39</td><td>67.14</td><td>65.15</td><td>64.40</td><td>73.43</td><td>90.00</td><td>65.89</td><td></td><td>50.8453.94</td><td>180,010</td></tr><tr><td>Workflow-profile router</td><td>56.05</td><td>56.75 56.40</td><td></td><td>56.47</td><td>54.74</td><td>52.63</td><td>58.60</td><td>87.95</td><td>55.39</td><td></td><td>55.0060.62</td><td>220,232</td></tr><tr><td>Expert-only suite</td><td>59.28</td><td>60.88 60.08</td><td></td><td>59.50</td><td>58.48</td><td>61.08</td><td>61.71</td><td>90.45</td><td>57.79</td><td></td><td>61.0857.62</td><td>295,490</td></tr><tr><td>General + expert suite Atmospheric Python suite</td><td>59.07</td><td>63.0061.04</td><td></td><td>59.95 59.67</td><td>59.25</td><td>66.01</td><td>63.21</td><td>95.00</td><td>61.34</td><td></td><td>49.2751.38</td><td>195,092</td></tr><tr><td>Atmospheric Fortran suite</td><td>59.48 57.46</td><td>60.12 59.80 57.88 57.67</td><td></td><td>57.51</td><td>58.26 55.81</td><td>59.64 51.05</td><td>60.69 61.19</td><td>89.20 92.95</td><td>60.94 57.35</td><td></td><td>58.3149.25</td><td>198,802</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>53.88 60.75</td><td>205,262</td></tr><tr><td colspan="9">(c) Context retained before summarization</td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive compression</td><td>48.75</td><td></td><td>51.62 50.19</td><td>49.00</td><td>48.07</td><td>57.55</td><td>53.73</td><td>90.45</td><td>44.65</td><td></td><td>45.4447.00</td><td>197,038</td></tr><tr><td>Short retained context</td><td>56.31</td><td>56.38 56.34</td><td></td><td>56.52</td><td>54.16</td><td>57.30</td><td>55.54</td><td>97.72</td><td>57.85</td><td></td><td>60.0842.25</td><td>209,827</td></tr><tr><td>Long retained context</td><td>64.32</td><td>63.38</td><td>63.85</td><td>64.75</td><td>62.21</td><td>59.43</td><td>68.29</td><td>95.00</td><td>64.72</td><td></td><td>58.79 48.25</td><td>227,536</td></tr><tr><td>Full retained history</td><td>58.90</td><td>58.25 58.58</td><td></td><td>59.42</td><td>56.98</td><td>55.39</td><td>62.15</td><td>89.00</td><td>57.25</td><td></td><td>52.5961.97</td><td>187,700</td></tr><tr><td colspan="9">(d) EarthVerse-harness turn budget under one wall clock</td><td></td><td></td><td></td><td></td></tr><tr><td>12-turn budget</td><td>44.84</td><td></td><td>49.5047.17</td><td>44.95</td><td>44.65</td><td>45.36</td><td>46.27</td><td>95.00</td><td>47.73</td><td></td><td>57.5035.13</td><td>106,263</td></tr><tr><td>24-turn budget</td><td>54.10</td><td>52.88 53.49</td><td></td><td>54.32</td><td>51.31</td><td>57.88</td><td>55.46</td><td>95.00</td><td>50.93</td><td></td><td>49.13 50.92</td><td>161,404</td></tr><tr><td>96-turn budget</td><td>59.72</td><td>60.12 59.92</td><td></td><td>60.16</td><td>57.94</td><td>62.76</td><td>60.53</td><td>97.50 90.45</td><td>62.10 67.11</td><td></td><td>54.0844.38 56.91 33.38</td><td>182,798</td></tr><tr><td>Wall-clock limited</td><td>62.65</td><td>64.50 63.57</td><td></td><td>62.91</td><td>62.16</td><td>62.72</td><td>66.66</td><td></td><td></td><td></td><td></td><td>221,180</td></tr></table>

## A.3 Oracle access, discoverability, and evidence robustness

These conditions reuse one paired task set and the Top-20 router baseline. Table 4 reports the outcomes; Appendix D.5 retains trajectory and resource diagnostics.

## A.3.1 Oracle bottleneck localization

Each oracle removes one information bottleneck while leaving the model, budget, tools, and scoring unchanged. It is a diagnostic upper bound rather than a proposed deployment setting, and none of the conditions reveals the target answer. The relevant-file oracle identifies only the package-relative files used by the reference analysis. The tool-subset oracle names useful tool classes; the planning oracle supplies a short answer-free plan; and the evidence-map oracle adds source roles and intermediate quantities without stating the conclusion.

## A.3.2 Tool naming and discoverability

We hold tool implementations fixed and alter only how the catalog is presented (panel (b) of Table 4). Descriptive names add intended-use and exclusion cues; hierarchical discovery chooses a domain before exposing tools; dynamic descriptions retrieve candidate tools at each step; opaque names remove semantic hints.

## A.3.3 Evidence perturbation and robustness

Perturbations preserve the question and answer schema while modifying the package evidence (panel (c) of Table 4): irrelevant files are added, a secondary source is made inconsistent, one supporting file is removed, long text is truncated, or units are changed while preserving physical equivalence.

## A.3.4 Compact meteorological access

This probe tests access rather than compulsory execution. One condition exposes the same compact shortlist of meteorological operations to every task; the other uses task and package cues to expose at most three. The agent may ignore them in either condition. This separates tool availability from the professional-environment study below, where repeated specialist use is required.

## A.4 Professional meteorological environments

This study asks a stricter question: what happens when a task must be completed through a named scientific environment? Each condition requires repeated use of one of 13 libraries or runtimes under the same protocol. Adapters expose only implemented operations and record whether calls succeed, allowing the results to separate useful scientific fit from technically successful but unsuitable method use.

Table 7: Required scientific environments. GPT-5.5; ∆Core is relative to the Top-20 router.
<table><tr><td rowspan="2">Environment</td><td colspan="3">Outcome</td><td colspan="2">Change</td></tr><tr><td>Answer</td><td>Process</td><td>Core</td><td>∆ Core</td><td>Unit acc.</td></tr><tr><td>Top-20 router (ref.)</td><td>66.90</td><td>65.88</td><td>66.39</td><td>0.00</td><td>67.14</td></tr><tr><td colspan="6">(a) Python scientific libraries</td></tr><tr><td>MetPy</td><td>68.69</td><td>70.38</td><td>69.53</td><td>+3.14</td><td>68.80</td></tr><tr><td>xclim</td><td>67.77</td><td>69.62</td><td>68.70</td><td>+2.31</td><td>67.95</td></tr><tr><td>thermofeel</td><td>66.78</td><td>68.75</td><td>67.76</td><td>+1.37</td><td>67.47</td></tr><tr><td>PyET</td><td>65.34</td><td>67.38</td><td>66.36</td><td>-0.03</td><td>65.89</td></tr><tr><td>climate-indices</td><td>64.50</td><td>66.75</td><td>65.62</td><td>-0.77</td><td>65.58</td></tr><tr><td>xskillscore</td><td>63.37</td><td>64.12</td><td>63.75</td><td>-2.64</td><td>63.90</td></tr><tr><td>pyextremes</td><td>59.02</td><td>60.00</td><td>59.51</td><td>-6.88</td><td>59.27</td></tr><tr><td colspan="6">(b) Compiled and domain-language environments</td></tr><tr><td>Julia</td><td>71.77</td><td>70.25</td><td>71.01</td><td>+4.62</td><td>71.99</td></tr><tr><td>R climate</td><td>68.55</td><td>71.38</td><td>69.96</td><td>+3.57</td><td>69.44</td></tr><tr><td>CDO</td><td>66.91</td><td>69.50</td><td>68.20</td><td>+1.81</td><td>67.35</td></tr><tr><td>NCO</td><td>67.39</td><td>65.00</td><td>66.19</td><td>-0.20</td><td>67.49</td></tr><tr><td>NCL</td><td>65.03</td><td>65.00</td><td>65.01</td><td>-1.38</td><td>65.27</td></tr><tr><td>Fortran</td><td>64.46</td><td>64.25</td><td>64.35</td><td>-2.04</td><td>64.52</td></tr></table>

Julia, R climate, MetPy, xclim, and CDO improve Core because their implemented operations match common package structures such as time series, gridded fields, and atmospheric profiles. Julia gives the largest gain (+4.62), but the ranking is less informative than the conditions under which the tools help. They compress many low-level numerical steps into an inspectable transformation without hiding the variables or aggregation. That advantage disappears when the package lacks the temporal support or paired variables assumed by the method.

Required use exposes a failure that optional-tool studies can miss. Pyextremes loses 6.88 Core points because many event packages do not contain a long, homogeneous series suitable for return-level estimation; forecast-verification methods face the same mismatch when paired forecasts and observations are absent. Under a hard requirement, the agent sometimes bends the available data toward the method instead of rejecting the method. A scientific interface needs an explicit refusal path, backed by minimum record length, sampling assumptions, missing-data rules, and a statement of what the output can support.

Finding. Execution success is not scientific validity. A specialist method helps only when its required variables and sampling support exist. Otherwise, a successful call can produce a precise result that the package cannot justify.

## B Discussion, Broader Impact, and Limitations

## B.1 Research position

EarthVerse evaluates a complete research loop over a bounded event archive. The agent must decide what counts as evidence, transform it, and revise an event-scale explanation while retaining enough provenance for another analyst to replay the work. This is more constrained than open-ended research, but it begins earlier than supplied-observation QA and asks more of the controller than a tool-execution benchmark.

Strong systems often interpret a selected observation correctly. Reliability falls when they must choose the record, its scale, the compatible variables, and the point at which the analysis is complete. Many failures originate in evidential binding: a plausible value is attached to the wrong aggregation, time window, unit, or source role, and later reasoning treats the mismatch as established fact. This explains why stronger reasoning, more tools, and longer traces can all fail to help. They operate on the state the controller has preserved. If that state no longer records what a value means, additional reasoning only makes the wrong account more elaborate.

Complex Earth-system investigations rarely have a unique valid research path. EarthVerse therefore prescribes neither a tool order nor a reference trajectory. Different paths are valid when they recover the required windows, quantities, source roles, and mechanisms, and preserve support for the resulting claims [38, 39].

## B.2 Broader impact

Scientific practice. Assistants can already reduce the mechanical burden of first-pass hazard analysis: inventorying an archive, locating candidate records, harmonizing units, repeating calculations, and surfacing source disagreement. The experiments also show where this assistance becomes risky. A fluent system may complete each local operation while carrying a scale mismatch through the full analysis. Experts still need to judge comparability, uncertainty, and consequences; package-local provenance gives them something concrete to inspect.

Reliability in high-stakes settings. High average unit accuracy can hide one missing source, time window, calculation, or mechanism. In disaster analysis, that single omission may change estimated severity, exposed population, or the causal account used to justify action. The benchmark does not certify operational readiness. It makes these omissions visible before a result is treated as operational evidence.

Access and evaluation practice. A local package reduces dependence on changing APIs and lets researchers inspect the same evidence. Evaluating frontier systems across hundreds of long trajectories is still expensive and may exclude smaller groups. Cost should be reported alongside quality, and releases should include compact diagnostic subsets, task-level scores, and traces. Subset results should not be presented as full-benchmark estimates.

## B.3 System design implications

For Earth-system analysis, a trajectory is useful only when it preserves the support for each claim. A record of calls without source bindings cannot show whether the final explanation follows from the evidence.

Claim–evidence state. A transcript records chronology but does not identify which claims and values remain active. The controller needs a claim–source graph in which every derived value retains its source, unit, time window, transformation, and relation to competing explanations. Corrections should replace active values without erasing the history that explains the change.

Scientific operations. Scientific tools need to expose applicability as part of their output. Record length, sampling support, missing-data rules, and estimator assumptions determine whether a computed value can support the claim. Training that rewards successful calls alone encourages the model to force data into an available method; justified refusal should receive credit when the inputs are inadequate.

Verification and oversight. Final review should operate on claims rather than prose. For each required conclusion, the system should expose coverage, dimensional checks, source conflicts, and the mechanism tests it rejected. Experts can then inspect the points where judgment entered the analysis and decide whether the result is fit for use. A polished report without this record is not auditable.

## B.4 Limitations

Coverage and ecological validity. EarthVerse is a controlled approximation of research. Its 199 packages cover many natural hazards, but not the full range of Earth-system science, long-term attribution, or live forecasting. Package isolation improves reproducibility and blocks web leakage, at the cost of open-ended discovery and institutional data-quality judgments. Masked products also prevent tests of absolute geolocation.

Ground truth and judge validity. Structured ground truth necessarily chooses one decomposition of a scientific answer. Process scores also rely on an LLM judge that compares each trace with a reference solution. Separate answer and process blocks reduce score leakage but cannot remove judge error. Future releases should measure agreement with domain experts on a stratified subset and retain criterion-level decisions.

Data provenance and maintenance. The assembled event packages, task formulations, answer units, and derived targets were created for EarthVerse and were not available online before release, which reduces the chance of exact benchmark memorization. Some underlying historical events and source documents may still have appeared in pretraining. Package isolation, hidden scoring artifacts, and package-specific calculations make shortcut answering harder.

Statistical scope of controlled studies. The controlled studies use paired subsets, and several intervals in Appendix D.5 include zero. Large contrasts, such as the relevant-file oracle, support architectural hypotheses. Smaller changes need replication across more packages and backbones. Cross-benchmark revision measures recoverability under feedback, not zero-shot superiority.

## C Experimental Setup and Reproducibility

This section collects implementation details that are necessary for reproduction but not for interpreting the main findings. It records the evaluated systems, the common evidence interface, the leaderboard configuration, controlled-study manifests, scoring isolation, and retained audit artifacts.

## C.1 Systems, controllers, and model modality

The leaderboard contains 25 complete systems. Hosted endpoints include the OpenAI GPT-4o, GPT-5.5, and GPT-5.6 tiers [51–53]; Anthropic Claude tiers [54]; Gemini [55]; MiniMax M2.7 [56]; Seed 2.0 [57]; Qwen3.7 [58]; Tencent Hy3 [59]; DeepSeek-V4 [60]; Kimi K2.5 [61]; and GLM-5.2 [62]. The open-weight group includes Qwen3 at 4B and 235B scales [63], Nemotron Nano [64], MiroThinker 8B [46], GeohazardGPT, GeoGPT, and Intern-S1-mini [48–50].

Most systems run in the EarthVerse harness, which manages package discovery, tool use, context, recovery, and finalization. GeoMMAgent uses its adapted five-stage coordinator with GPT-5.5 as the worker model in every stage; OpenResearcher retains its research-loop controller [17, 45]. These rows therefore describe a model–controller system, not a bare backbone.

The roster mixes text-only models with natively multimodal endpoints. Direct image interpretation is required by only a small fraction of EarthVerse tasks; most remote-sensing evidence is represented by product metadata, raster statistics, change indices, or other auditable tool outputs. Raw pixels are not selectively sent to models with vision support. This keeps the evidence interface comparable, although the resulting scores should not be read as a standalone test of visual perception.

## C.2 Shared experimental configuration

Every run exposes a scientific question and one evidence root while keeping the solution, rubric, ground truth, scoring code, prior outputs, and web evidence hidden. Reads, writes, and Python execution stay inside the active evidence collection. Reports and tables are returned as local text or structured summaries; image and raster operations return recorded metadata and measurements.

The EarthVerse harness is the common controller for the standard model rows. To study the attainable system-level ceiling, we also evaluate the GPT-5.6 family in Codex and Claude Fable 5 in Claude Code, the strongest modelnative harnesses available for those systems. GeoMMAgent pairs GPT-5.5 with its five-stage coordinator, while OpenResearcher keeps its own controller. The Harness column in Table 2 makes these differences explicit.

The common execution limit is 48 turns, 1,680 seconds for research, and 1,800 seconds overall. The controller retains 16 recent messages and a 24,000-character research state; individual tool observations are capped at 12,000 characters. Package-scoped Python has a 60-second call limit. Up to two schema repairs are allowed, with 120 seconds reserved for finalization and a separate 120-second final-call timeout. We retain provider-native reasoning and generation controls when available, use endpoint defaults otherwise, and do not impose one temperature across providers. The judge is a fixed gpt-4.1 endpoint, and randomized procedures use seed 2026. Endpoint access dates and local model revisions are retained in the release registry.

## C.3 Package evidence sources and access interfaces

The 199 package manifests contain 6,397 provenance records, 835 raw source\_name labels, and 6,709 local files from 107 external host domains. Many raw labels are event-specific filenames, response formats, endpoint variants, or repeated queries to the same service rather than distinct sources. We therefore group aliases at the product/API-family level while keeping distinct providers, sensors, product versions, and access semantics separate. This yields 80 external source and access families. We also list nine package-local derivative families used directly in computation; they do not count as additional upstream sources.

<table><tr><td colspan="3">PACKAGE SOURCES Distinct source and access families across 199 event packages</td></tr><tr><td colspan="3">Event discovery, reports, and authoritative anchors</td></tr><tr><td>Provider or product</td><td>Access form</td><td>Role in a package</td></tr><tr><td>GDACS</td><td>event-list REST/GeoJSON API; RSS archive</td><td>Cross-hazard event discovery, dates, alert levels, and identifiers.</td></tr><tr><td>NASA EONET v3</td><td>REST search by date, category, or keyword</td><td>Wildfire, flood, storm, and general natural-event catalog records.</td></tr><tr><td>ReliefWeb</td><td>public update and report search</td><td>Humanitarian situation reports, response updates, and impact narratives.</td></tr><tr><td>HDX</td><td>CKAN package_search API</td><td>Candidate humanitarian datasets and country- or event-specific</td></tr><tr><td>GDELT</td><td>event-window article search</td><td>impact layers. Secondary news discovery used only when the package retains</td></tr><tr><td>Wikipedia</td><td>REST page-summary and event-search interfaces</td><td>the result. Event-name cross-checks and compact encyclopedic context.</td></tr><tr><td>Wikidata</td><td>entity-search API</td><td>Structured event and place identifiers used for identity checks.</td></tr><tr><td>NASA Earth Observatory</td><td>event articles and image pages</td><td>Locked event anchors and satellite-based physical narratives.</td></tr><tr><td>NASA Scientific Visualization Studio</td><td>visualization pages and media API</td><td>Official imagery, animations, and explanatory visual products.</td></tr><tr><td>NASA Disasters and Applied Sciences</td><td>activation and event pages</td><td>Satellite-based disaster-response context and mapped</td></tr><tr><td>World Meteorological Organization</td><td>reports, topic pages, and bulletins</td><td>observations. Authoritative global weather, climate, and extreme-event</td></tr><tr><td>World Meteorological Centre Beijing</td><td>event reports and operational bulletins</td><td>summaries. Regional meteorological summaries and event chronology.</td></tr><tr><td>NOAA Climate.gov</td><td>event trackers and climate reports</td><td>Reviewed US climate-event narratives and physical context.</td></tr><tr><td>US National Weather Service and WPC</td><td>forecast discussions, storm summaries, and precipitation pages</td><td>Operational weather chronology, rainfall totals, and warning context.</td></tr><tr><td>PACKAGE SOURCES</td><td>Distinct source and access families across 199 event packages</td><td>continued</td></tr><tr><td>NOAA NESDIS and SSD</td><td>satellite product and event pages</td><td>Operational satellite imagery and environmental monitoring context.</td></tr><tr><td>Copernicus EMS</td><td>activation and rapid-mapping archives</td><td>Event confirmation and links to emergency-mapping products.</td></tr><tr><td>UN OCHA</td><td>situation reports and humanitarian updates</td><td>Reported impacts, needs, access constraints, and response activity.</td></tr><tr><td>UNOSAT</td><td>satellite-assessment reports and maps</td><td>Mapped damage, flood, and exposure evidence.</td></tr><tr><td>UN country teams and UN Geneva</td><td>country and event updates</td><td>Local consequence and response narratives retained as event evidence.</td></tr><tr><td>IFRC</td><td>emergency appeals and operational updates</td><td>Humanitarian consequences, needs, and Red Cross response records.</td></tr><tr><td>AHA Centre</td><td>regional disaster updates and reports</td><td>ASEAN event impacts and coordinated response context.</td></tr><tr><td>World Health Organization</td><td>health-emergency reports</td><td>Mortality, morbidity, service disruption, and public-health consequences.</td></tr><tr><td>UNICEF</td><td>humanitarian situation reports</td><td>Impacts on children, water, health, education, and relief delivery.</td></tr><tr><td>FAO</td><td>food, agriculture, and livelihood reports</td><td>Crop, livestock, food-security, and rural-impact evidence.</td></tr><tr><td>World Bank</td><td>assessment and recovery documents</td><td>Infrastructure, economic loss, and reconstruction context.</td></tr><tr><td>World Weather Attribution</td><td>event-study pages and reports</td><td>Attribution context and documented physical or societal drivers.</td></tr><tr><td>South Asian operational agencies</td><td>IMD/RSMC New Delhi, DHM Nepal, and DMC Sri Lanka</td><td>Official cyclone, rainfall, flood, and warning records.</td></tr><tr><td>National meteorological services</td><td>BoM, HKO, UK Met Office, MetService, and SAWS</td><td>National weather reports, warnings, and climate summaries.</td></tr><tr><td>Scientific institutes</td><td>ESSL, ICIMOD, NGI/NGU, ICL, OSTI</td><td>Hail, glacier, landslide, heat, and other specialist event accounts.</td></tr><tr><td colspan="3">Meteorology, climate, hydrology, ocean, and air quality</td></tr><tr><td>Provider or product</td><td>Access form</td><td>Role in a package</td></tr><tr><td>ERA5-Land</td><td>hourly reanalysis; package-local</td><td>Temperature, precipitation, wind, and land-surface background.</td></tr><tr><td>CHIRPS v2</td><td>aggregates/stacks daily precipitation archive</td><td>Event accumulation, wet spells, climatology, and rainfall</td></tr><tr><td>NASA GPM IMERG V07</td><td>Earthdata CMR discovery and precipitation</td><td>anomalies. Satellite precipitation totals, timing, and spatial structure.</td></tr><tr><td>Open-Meteo Archive</td><td>granules historical hourly/daily REST API</td><td>Point temperature, humidity, apparent temperature, rain, snow,</td></tr><tr><td></td><td></td><td>and wind.</td></tr><tr><td>Open-Meteo Air Quality NASA POWER</td><td>historical air-quality REST API daily point REST API</td><td>PM2.5, PM10, aerosol, and related atmospheric fields. Temperature, corrected precipitation, and 10-m wind</td></tr><tr><td>NOAA PSL climate indices</td><td></td><td>cross-checks.</td></tr><tr><td></td><td>download tables for ONI, SOI, and related modes</td><td>Large-scale climate-state and teleconnection context.</td></tr><tr><td>NOAA CPC ONI IRI and Australian BoM</td><td>ASCII and tabular index archive</td><td>Operational Oceanic Niño Index values and episode timing.</td></tr><tr><td>GPCC</td><td>ENSO monitoring archives precipitation-anomaly products</td><td>Independent ENSO phase and coupled-ocean background. Gauge-based anomaly context alongside CHIRPS.</td></tr><tr><td>NOAA OISST v2 high resolution</td><td>PSL THREDDS/ERDDAP catalog</td><td>Sea-surface temperature and marine-heatwave context.</td></tr><tr><td>Indian Ocean Dipole index</td><td>DMI archive</td><td>Indian Ocean climate-mode state for regional events.</td></tr><tr><td>NOAA ocean services</td><td>tide gauges and Ocean Prediction Center</td><td>Coastal water levels, surge context, and marine storm structure.</td></tr><tr><td>Remote sensing and geospatial context</td><td>analyses</td><td></td></tr><tr><td colspan="3"></td></tr><tr><td>Provider or product</td><td>Access form</td><td>Role in a package</td></tr><tr><td>NASA Worldview/GIBS</td><td>snapshot API and MODIS/VIIRS browse imagery</td><td>Pre-event and in-event true-color or thematic visual context.</td></tr><tr><td>MODIS/VIIRS vegetation products</td><td>Worldview or archive previews</td><td>Vegetation condition and drought-stress context.</td></tr><tr><td>MODIS/VIIRS MAIAC AOD</td><td>aerosol and smoke previews</td><td>Smoke transport and aerosol loading context.</td></tr><tr><td colspan="3">PACKAGE SOURCES Distinct source and access families across 199 event packages</td></tr><tr><td>Sentinel-1 GRD Sentinel-2 SR Harmonized</td><td>VV pre/post imagery and change products surface reflectance and dNBR derivatives</td><td>Flood, deformation, and surface-change comparison. Burn severity and optical before/after consistency.</td></tr><tr><td>Google Satellite Embedding / AlphaEarth</td><td>annual 64-band embeddings</td><td>Annual state vectors, cosine-change statistics, and change</td></tr><tr><td></td><td></td><td>rasters.</td></tr><tr><td>Copernicus EMS Rapid Mapping</td><td>EMSR/GFM maps and archive products</td><td>Mapped flood, fire, earthquake, and exposure footprints.</td></tr><tr><td>Microsoft Planetary Computer Natural Earth</td><td>STAC metadata search</td><td>AOI/date discovery for registered Earth-observation assets.</td></tr><tr><td>geoBoundaries</td><td>Admin-0 and physical basemap files</td><td>Low-resolution geopolitical and physical reference layers.</td></tr><tr><td>GHSL</td><td>administrative-boundary API</td><td>Open administrative geometries for package AOIs.</td></tr><tr><td>Copernicus DEM and SRTM</td><td>built-up and settlement layers elevation and derived slope</td><td>Urban extent and settlement structure. Terrain constraints for flood, landslide, and glacier analyses.</td></tr><tr><td colspan="3">Exposure, infrastructure, and human impact</td></tr><tr><td colspan="3"></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Provider or product</td><td>Access form</td><td>Role in a package</td></tr><tr><td>WorldPop GP 100 m</td><td>population grids and AOI summaries</td><td>Population exposure, density, and resampled 1-km context.</td></tr><tr><td>OpenStreetMap Nominatim</td><td>geocoding REST API</td><td>Event location, fallback coordinates, and AOI checks.</td></tr><tr><td>OpenStreetMap Overpass OpenStreetMap Geofabrik</td><td>bounded feature-query API</td><td>Roads, buildings, services, shelters, and critical amenities.</td></tr><tr><td></td><td>regional extracts</td><td>Broader road and infrastructure context where bounded queries are insufficient.</td></tr><tr><td>FEWS NET and IPC NYC Open Data</td><td>food-security products and reports</td><td>Drought exposure, food-security phases, and response pressure.</td></tr><tr><td>Harris County and Houston open data</td><td>Socrata 311 service-request API flood-warning and 311 interfaces</td><td>Flood complaints and city-service stress during urban rainfall. Gauge conditions, complaints, and local flood impacts.</td></tr><tr><td>Hong Kong public feeds</td><td>warnings, incident, and transport records</td><td>Local disruption and response evidence for typhoon events.</td></tr><tr><td>Brazil CEMADEN and civil defence</td><td>alerts and municipal reports</td><td>Rainfall-triggered hazard warnings and local consequences.</td></tr><tr><td colspan="3">Hazard-specific scientific and operational sources</td></tr><tr><td colspan="3"></td></tr><tr><td>Provider or product</td><td>Access form</td><td>Role in a package</td></tr><tr><td>USGS ComCat</td><td>earthquake-event REST API</td><td>Origin time, magnitude, depth, location, and event products.</td></tr><tr><td>USGS ShakeMap and PAGER</td><td>event-product feeds</td><td>Instrumental intensity, shaking footprint, and loss context.</td></tr><tr><td>USGS Did You Feel It?</td><td>community-intensity product</td><td>Reported shaking distribution and populated-area impacts.</td></tr><tr><td>NOAA NCEI/WDS</td><td>Global Historical Tsunami Database</td><td>Tsunami source, run-up, and historical event records.</td></tr><tr><td>NOAA NHC</td><td>Tropical Cyclone Reports archive</td><td>Track, intensity, rainfall, surge, and impact summaries.</td></tr><tr><td>NOAA HRD H*Wind</td><td>surface-wind analysis archive</td><td>Storm wind structure and peak-wind calibration.</td></tr><tr><td>NOAA Coral Reef Watch</td><td>5-km thermal-stress products</td><td>HotSpot, Degree Heating Week, and bleaching-risk context.</td></tr><tr><td>Copernicus GloFAS/GFM/EFAS</td><td>flood-monitoring and event products</td><td>River-flood extent, timing, and European flood context.</td></tr><tr><td>Dartmouth Flood Observatory</td><td>Global Flood Monitor/archive</td><td>Independent flood occurrence and extent context.</td></tr><tr><td>Smithsonian GVP</td><td>volcano reports and event pages</td><td>Eruption chronology, volcano status, and physical</td></tr><tr><td>Volcanic Ash Advisory Centres</td><td>operational ash advisories</td><td>interpretation. Ash extent, altitude, movement, and aviation context.</td></tr><tr><td>GLIMS and Randolph Glacier Inventory</td><td>glacier outlines and catalog interfaces</td><td>Glacier geometry and inventory context.</td></tr><tr><td>NSIDC and ICIMOD</td><td>glacier and cryosphere reports</td><td>Regional ice change and glacier-hazard evidence.</td></tr><tr><td>NASA Global Landslide Catalog / COOLR</td><td>landslide catalog search</td><td>Occurrence records and landslide-event cross-checks.</td></tr><tr><td>USGS landslide releases</td><td>event pages, publications, and data releases</td><td>Site evidence for landslides and landslide-generated tsunamis.</td></tr><tr><td>ESSL and IEM/NWS archives</td><td>severe-weather summaries and Local Storm Reports</td><td>Hail, wind, tornado, and convective-event observations.</td></tr><tr><td>Australian fire and smoke products</td><td>official extent and smoke records</td><td>Fire perimeter, smoke, and air-quality context.</td></tr><tr><td colspan="3">Package-local derivatives retained for computation</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Provider or product Compact event AOI</td><td>Access form event-bounding-box derivative</td><td>Role in a package Common spatial support for package layers.</td></tr><tr><td>ERA5-Land aggregate and stack</td><td>hourly statistics and raster bundle</td><td>Compact numerical access without replacing the upstream provenance.</td></tr><tr><td>GPM and CHIRPS event accumulations</td><td>derived totals and AOI rasters</td><td>Comparable precipitation ledgers across satellite and gauge products.</td></tr><tr><td>WorldPop summaries</td><td>AOI sum and 1-km resample</td><td>Deterministic population exposure inputs.</td></tr><tr><td>Sentinel-1 change</td><td>pre/post statistics and raster</td><td>Auditable radar change metrics.</td></tr><tr><td>Sentinel-2 dNBR</td><td>summary and raster</td><td>Auditable burn-severity calculation.</td></tr><tr><td>Satellite-embedding change</td><td>annual statistics and raster</td><td>Normalized annual surface-change evidence.</td></tr><tr><td>Bounded OSM slice</td><td>package-local JSON</td><td>Stable infrastructure and service counts.</td></tr><tr><td>Locked event anchor</td><td>package-local report copy</td><td>Fixed event identity, time window, and provenance for replay.</td></tr></table>

The inventory is exhaustive at this level of grouping, although each event uses only a hazard- and availability-specific subset. The 80 external rows cover distinguishable upstream sources and access methods; the nine derived rows record transformations reused across packages. Per-package manifests retain the raw label, exact URL, acquisition status, package-relative output, byte count, and checksum for record-level provenance.

## C.4 Scoring, isolation, and audit protocol

The judge receives the task, structured ground truth, final answer, reference process, rubric, and trajectory. It scores required answer units separately from evidence selection, calculation, source reconciliation, and mechanism analysis. Combined Core is the mean of these blocks, and Strict@95 records near-complete tasks. Capability columns use overlapping expert tags as diagnostic slices, not additive categories. Controlled comparisons use paired manifests and task-level bootstrap resampling.

Solvers may list, read, search, and compute only within the active event package. The reference solution, rubric, structured ground truth, and ground-truth program remain hidden. Tools record each path and operation before shortening observations, which preserves provenance through context compression. Finalization receives the accumulated research state and required schema but no answer-bearing hint. Released traces cannot reconstruct provider-hidden reasoning, proprietary routing, or undocumented cache accounting.

## C.5 Controlled-study sampling design

The reasoning-effort study uses a fixed manifest containing historically difficult tasks and a fixed-seed random draw from the rest of the release, preferring distinct events. Tool, routing, context, and interaction studies use a smaller paired manifest drawn from it. Oracle and robustness conditions reuse the same task IDs and Top-20 router outputs. Paired 95% intervals use 5,000 bootstrap resamples of task-level score differences with seed 2026.

Professional-environment conditions use GPT-5.5 with the default medium reasoning setting and 48 turns. Core libraries retain the paired manifest; input-dependent environments draw a difficulty-balanced compatible manifest with seed 2026. Their compliance contract requires at least two successful, distinct specialist calls, targets three, and counts at most four. These paired ablations are diagnostic and should not be read as exact full-leaderboard effects.

Artifacts retained for audit.

• Task identity. Version, manifest, package checksum, and question reproduce the scientific input.

• System identity. Endpoint revision, controller, tool profile, and context policy reproduce the executable system.

• Research trace. Actions, arguments, observations, errors, and timing reconstruct the investigation, including failed calls.

• Submission. Raw text, parsed object, repair attempts, and final status keep malformed or missing answers in the denominator.

• Reference and judgment. Ground truth, solution, computation, rubric, judge prompt, and criterion decisions support answer-unit and aggregate recomputation.

• Paired comparison. Task IDs, intervention, seed, and bootstrap resamples identify the controlled change.

## C.6 Human evaluation and agent-assisted review

Human review begins when a task is conceived and continues through release. The team included six doctoral-leve experts in Earth science and twenty undergraduates in related fields. Doctoral reviewers set the scientific direction, checked physical validity, and approved the final scoring specification. Undergraduate reviewers followed a common protocol for source and metadata checks, package replay, calculation verification, and secondary review of answer units. Reviewers contributed more than 50 hours per person on average.

Task direction and scoring checkpoints. For every task, a doctoral-level reviewer first analyzed the event and specified the scientific question, hazard process, spatial and temporal support, admissible evidence, required calculations, and answer fields. LLM agents then explored the package for plausible source combinations, alternative calculations, ambiguities, and likely failure modes. Reviewers used these passes to refine the reference analysis and define the final answer units and process checks. Agents proposed candidates; human experts decided which checks were scientifically necessary, supported by the package, and suitable for scoring.

Joint verification and correction. Every candidate task underwent agent replay followed by human inspection. Reviewers confirmed that package evidence could answer the public question, that compute\_gt.py regenerated the structured targets, and that the solution, answer units, and rubric referred to the same quantities and physical claims. Any mismatch returned the task to construction. A small number of sources had minor spatial offsets between the data footprint and the documented disaster location. Human experts removed the affected answer units rather than score a quantity the package could not support.

Iteration scale. For difficult cases, constructing and reviewing a single question consumed more than 20 million tokens across repeated agent passes. These passes covered evidence exploration, draft construction, calculation checks, replay, discrepancy analysis, and revision. Human reviewers used them to align the scientific question, evidence, expert solution, deterministic program, answer units, and process rubric before release.

## C.7 Benchmark construction and release gates

Each construction stage produces an auditable artifact and an explicit release decision. A failed gate returns the task to the responsible stage.

## Seven release gates.

1. Event anchoring: verify identity, hazard type, time window, spatial scale, and event–source agreement.

2. Package assembly: build a typed relative-path manifest and confirm that the required evidence roles are readable.

3. Task authoring: require package-local resolution without revealing a file path or answer-bearing value.

4. Reference analysis: align claims, units, mechanisms, and defensible alternatives; trace every load-bearing claim.

5. Executable grounding: regenerate structured ground truth deterministically and pass schema and tolerance checks.

6. Rubric design: map the task obligations to five to eight criteria totaling exactly 20 points.

7. Final review: align the question, solution, ground truth, program, and rubric before release approval.

## D Extended Diagnostics

Table 8 breaks Combined Core down by seven overlapping capability tags. Because a task may carry several tags, the columns are diagnostic views rather than additive partitions.

The strongest systems lead across nearly every tag, but their residual errors concentrate in the connections between results. Among the top five, causal-chain scores trail quantitative calculation by 4.9–7.9 points. A single defensible data path may be enough to compute a total or ratio. An evolving hazard account is harder to stabilize: rainfall, exposure, damage, and response evidence must refer to compatible windows and scales, and the proposed mechanism must survive their disagreement. GeoMMAgent breaks the otherwise stable ordering. Its short, structured search works well on tabular evidence but often bypasses raster intermediates, leaving remote sensing at 58.82. Capability scores thu capture the organization of an investigation as much as the knowledge of its backbone.

## D.1 Capability-conditioned performance

Table 8: Core by capability. Tags overlap; system groups follow Table 2. Bold blue marks column bests.
<table><tr><td rowspan="2">System</td><td colspan="3">Earth-process analysis</td><td colspan="4">Evidence synthesis</td></tr><tr><td>Physical mech.</td><td>Spatio- temporal</td><td>Quant. calc.</td><td>Multi- source</td><td>Causal chain</td><td>Ranking</td><td>RS/ geo.</td></tr><tr><td colspan="8">Agent frameworks</td></tr><tr><td>GeoMMAgent</td><td>67.72</td><td>72.94</td><td>68.85</td><td>70.56</td><td>66.15</td><td>67.00</td><td>58.82</td></tr><tr><td>OpenResearcher</td><td>30.46</td><td>35.24</td><td>31.35</td><td>34.63</td><td>25.35</td><td>30.41</td><td>33.18</td></tr><tr><td colspan="8">Open-weight and Earth-specialized models</td></tr><tr><td>Qwen3-235B</td><td>31.90</td><td>40.35</td><td>34.03</td><td>35.51</td><td>29.00</td><td>29.29</td><td>32.32</td></tr><tr><td>GeohazardGPT</td><td>31.77</td><td>31.31</td><td>28.04</td><td>30.42</td><td>32.12</td><td>39.04</td><td>29.16</td></tr><tr><td>Qwen3-4B</td><td>24.63</td><td>22.71</td><td>24.70</td><td>24.32</td><td>25.89</td><td>24.06</td><td>32.61</td></tr><tr><td>GeoGPT</td><td>25.34</td><td>25.70</td><td>24.66</td><td>23.14</td><td>23.63</td><td>26.28</td><td>22.77</td></tr><tr><td>MiroThinker 8B</td><td>13.51</td><td>14.39</td><td>14.68</td><td>14.52</td><td>8.61</td><td>6.76</td><td>21.96</td></tr><tr><td>Nemotron Nano</td><td>11.79</td><td>14.41</td><td>12.30</td><td>17.89</td><td>12.42</td><td>13.26</td><td>15.48</td></tr><tr><td>Intern-S1-mini</td><td>12.04</td><td>10.70</td><td>11.99</td><td>10.45</td><td>10.57</td><td>13.21</td><td>16.50</td></tr><tr><td colspan="8">Hosted general-purpose systems</td></tr><tr><td>Claude Fable 5</td><td>85.15</td><td>85.51</td><td>87.14</td><td>83.45</td><td>82.20</td><td>80.98</td><td>89.99</td></tr><tr><td>GPT-5.6 Sol</td><td>81.81</td><td>83.31</td><td>85.42</td><td>82.13</td><td>77.54</td><td>71.63</td><td>88.50</td></tr><tr><td>GPT-5.6 Terra</td><td>80.83</td><td>79.83</td><td>82.96</td><td>79.06</td><td>76.26</td><td>76.12</td><td>86.27</td></tr><tr><td>GLM-5.2</td><td>80.25</td><td>77.30</td><td>80.24</td><td>76.85</td><td>80.08</td><td>78.34</td><td>79.27</td></tr><tr><td>GPT-5.5</td><td>77.21</td><td>78.38</td><td>81.91</td><td>76.79</td><td>74.33</td><td>75.76</td><td>83.77</td></tr><tr><td>Claude Sonnet 4.6</td><td>75.49</td><td>73.60</td><td>78.36</td><td>77.92</td><td>73.43</td><td>79.73</td><td>75.93</td></tr><tr><td>Kimi K2.5</td><td>75.07</td><td>79.10</td><td>77.81</td><td>75.34</td><td>70.48</td><td>70.82</td><td>77.94</td></tr><tr><td>DeepSeek V4 Flash</td><td>73.63</td><td>76.33</td><td>75.16</td><td>72.59</td><td>68.91</td><td>72.33</td><td>77.75</td></tr><tr><td>Tencent Hy3</td><td>72.67</td><td>75.92</td><td>73.71</td><td>70.32</td><td>69.72</td><td>70.23</td><td>79.41</td></tr><tr><td>Claude Haiku 4.5</td><td>69.60</td><td>72.44</td><td>76.09</td><td>73.50</td><td>67.73</td><td>66.76</td><td>79.66</td></tr><tr><td>Qwen 3.7 Plus</td><td>70.41</td><td>72.40</td><td>71.13</td><td>69.27</td><td>67.19</td><td>68.08</td><td>78.26</td></tr><tr><td>Doubao Seed 2.0 Mini</td><td>63.24</td><td>66.43</td><td>62.67</td><td>59.30</td><td>60.42</td><td>63.51</td><td>70.67</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>60.18</td><td>61.40</td><td>60.38</td><td>56.99</td><td>56.37</td><td>58.35</td><td>68.49</td></tr><tr><td>MiniMax M2.7</td><td>42.76</td><td>52.91</td><td>48.80</td><td>49.53</td><td>39.99</td><td>24.21</td><td>42.28</td></tr><tr><td>GPT-40</td><td>39.97</td><td>38.86</td><td>38.77</td><td>34.93</td><td>37.56</td><td>43.70</td><td>48.30</td></tr><tr><td>GPT-5.6 Luna</td><td>34.74</td><td>42.91</td><td>38.58</td><td>36.48</td><td>31.77</td><td>21.17</td><td>38.21</td></tr></table>

## D.2 Preference structure and diagnostic error paths

Pairwise judgments broadly recover the leaderboard order while exposing uncertainty hidden by Core alone. Figure 7 separates wins, losses, and ties in opponent-balanced comparisons. Leaders win consistently, adjacent frontier systems remain close, and ties are uncommon. The drop beyond the leading group agrees with low Strict@95: partial success is common, while consistently complete research remains rare.

Position. Reliability in dynamic Earth systems depends on revising the claim–evidence state as observations change. A system must detect conflicts, isolate local errors, and carry corrections into derived quantities, causal interpretations, and the final answer. Systems differ not only in how often they err, but also in whether those errors are found, contained, and repaired. Longer traces, more tool calls, and higher execution success do not measure that capacity [11, 19].

![](images/e2b4cf5da38da5cbc7af933289cf102e9923123880fbc2bb775ae75a2e771aa8.jpg)  
Figure 7: Pairwise win, tie, and loss shares. Net is win share minus loss share.

Figure 8 shows the same separation at task level. Frontier models have a dense high-score mode but retain a long tail toward zero. Agent frameworks cluster at moderately high scores, while open-weight and Earth-specialized systems shift downward and rarely approach complete solutions. Error counts overlap far more than Core scores. Strong systems still make mistakes; their advantage lies in catching or containing them before the final answer. Error frequency therefore describes the research process, while Core records the damage that remains.

The family-level flow in Figure 9 sharpens this result. Computation and answer errors occur in every family, yet their consequences differ. Frontier systems more often keep affected tasks in the 40–70 or above-70

![](images/1210a4f1486015585b3f72b8c4e0638f69e43e0c9cca5dd9c86407b89d7e2c9a.jpg)

![](images/f9ecc7084473ade5c36465c4d54462db0a111fec2777f6878a7f4a321433024c.jpg)  
Figure 8: Task Core and diagnosed errors by system family.

bands; open-weight and Earth-specialized systems more often fall below 40. Recovery, cross-source checking, and final synthesis determine whether a local mistake remains local or spreads through the answer.

![](images/441eda55c4c56fdd67eaccc9a5b0b293e24e7fa02968c4a58bf2161aa8a0aa8d.jpg)  
Figure 9: System family, primary error family, and task-level Core band.

Figure 10 examines the same pattern within systems. The largest Core losses accompany premature stopping, failed recovery, tool failure, and missing required outputs—failures that interrupt the research loop itself. Numerical and schema errors are common, but their average effect is smaller because some runs catch or absorb them before finalization The inset connects missing evidence, output omissions, schema mistakes, and calculation errors into one cluster. These are often successive stages of a failed investigation, not independent defects.

![](images/78256801eeb81e266dca1358ce5a7a12705005813c8fc94fddb199e619fe47a7.jpg)  
Figure 10: Within-system Core differences and the strongest error co-occurrences.

## D.3 Reasoning modes and resource accounting

Tables 9–11 report trajectory, latency, token use, and evaluation cost. They separate interactive retrieval from direct serialization and model activity from successful evidence acquisition.

## D.3.1 Interactive trajectory diagnostics

Higher reasoning effort changes the treatment of evidence more than the amount collected. At xhigh, the EarthVerse harness opens about as many files as at medium effort and uses fewer rounds, yet takes almost twice as long. The additional work occurs between observations: the model compares interpretations, revisits calculations, and decides whether its account is complete. Explicit review intensifies this pattern. From harness-high to review-xhigh, output nearly doubles while file reads barely move. Some of that deliberation repairs omissions; some only restates a settled path. Rounds, tokens, and latency therefore measure different kinds of work, and none is a reliable stand-in for scientific progress.

Table 9: Interactive trajectory diagnostics. Per-task means; call success is the share of tool calls that return normally.
<table><tr><td></td><td></td><td colspan="3">Execution</td><td colspan="3">Evidence</td><td colspan="4">Resources</td></tr><tr><td>Mode</td><td>Effort</td><td>Rounds</td><td>Tool Success calls</td><td>(%)</td><td>Reads Files</td><td></td><td>refs.</td><td>(k)</td><td>(k)</td><td>Evidence Input Output Total Cached Latency (k) (k)</td><td>(s)</td></tr><tr><td>EarthVerse none</td><td></td><td>9.68 8.68</td><td></td><td>92.08</td><td>3.84</td><td>7.00</td><td>9.94 94.3</td><td></td><td>2.6 96.9</td><td>10.6</td><td>140</td></tr><tr><td></td><td>low</td><td>14.7013.70</td><td></td><td>93.67</td><td>5.64 9.44</td><td></td><td>14.70 157.4</td><td></td><td>5.1 162.4</td><td>15.4</td><td>259</td></tr><tr><td></td><td>medium</td><td>17.20 16.20</td><td></td><td>95.39</td><td>6.84 10.71</td><td></td><td>18.12 198.6</td><td></td><td>7.9 206.4</td><td>17.9</td><td>424</td></tr><tr><td></td><td>high</td><td>14.96 13.96</td><td></td><td>97.19</td><td>5.15 9.92</td><td></td><td>17.19 164.8</td><td></td><td>9.5 174.3</td><td>13.2</td><td>571</td></tr><tr><td></td><td>xhigh</td><td>11.25 10.25</td><td></td><td>100.00</td><td>4.75 10.25</td><td></td><td>16.5090.6</td><td></td><td>9.4 100.0</td><td>15.9</td><td>674</td></tr><tr><td>Review</td><td>none</td><td>11.629.62</td><td></td><td>94.26</td><td>4.967.42</td><td></td><td>10.46 126.9</td><td></td><td>3.8 130.7</td><td>18.8</td><td>108</td></tr><tr><td></td><td>low</td><td>18.18 16.18</td><td></td><td>94.94</td><td>6.84 10.60</td><td></td><td>17.02 219.0</td><td></td><td>8.0 227.0</td><td>25.1</td><td>253</td></tr><tr><td></td><td>medium</td><td>20.81 18.81</td><td></td><td>94.78</td><td>6.56 11.88</td><td></td><td>21.02 253.8</td><td></td><td>12.5 266.3</td><td>25.2</td><td>552</td></tr><tr><td></td><td>high</td><td>20.63 18.63</td><td></td><td>95.37</td><td>6.50 12.13</td><td></td><td>21.10 239.2</td><td></td><td>15.9 255.1</td><td>22.2</td><td>1,105</td></tr><tr><td></td><td>xhigh</td><td>17.57 15.57</td><td></td><td>99.29</td><td>5.86 10.14</td><td></td><td>18.29 198.9</td><td></td><td>18.8 217.7</td><td>25.6</td><td>1,493</td></tr></table>

## D.3.2 Direct package-serialization diagnostics

Direct conditions receive a fixed package serialization before generation. Compressed direct includes more files in far fewer tokens than all-text, yet neither mode lets the model return to a source after discovering that a window, unit, or variable was wrong. That restriction changes the task. The model can reason about the supplied snapshot, but it cannot conduct the next measurement implied by its own reasoning. Larger inputs and longer outputs fail to rescue these baselines because the evidence schedule is fixed before the analysis begins. The missing operation is revision of the observation set, not another pass over the same text.

Table 10: Direct-serialization resources. Direct modes inject a fixed file set before generation; values are per-task means.
<table><tr><td>Mode</td><td>Effort</td><td>Injected files</td><td>Evidence refs.</td><td>Input</td><td>Output</td><td>Total</td><td>Latency (s)</td></tr><tr><td>Compressed</td><td>none</td><td>17.92</td><td>24.56</td><td>13,016</td><td>878</td><td>13,894</td><td>55.61</td></tr><tr><td></td><td>low</td><td>17.92</td><td>25.31</td><td>13,033</td><td>1,857</td><td>14,890</td><td>120.45</td></tr><tr><td></td><td>medium</td><td>17.92</td><td>25.72</td><td>12,838</td><td>3,004</td><td>15,842</td><td>152.36</td></tr><tr><td></td><td>high</td><td>17.93</td><td>26.73</td><td>12,958</td><td>3,630</td><td>16,588</td><td>71.47</td></tr><tr><td></td><td>xhigh</td><td>17.67</td><td>25.00</td><td>12,878</td><td>4,726</td><td>17,604</td><td>88.14</td></tr><tr><td>All text</td><td>none</td><td>7.57</td><td>13.89</td><td>80,842</td><td>751</td><td>81,593</td><td>32.06</td></tr><tr><td></td><td>low</td><td>7.68</td><td>14.00</td><td>80,886</td><td>1,854</td><td>82,739</td><td>44.10</td></tr><tr><td></td><td>medium</td><td>7.46</td><td>13.90</td><td>80,941</td><td>2,557</td><td>83,498</td><td>61.02</td></tr><tr><td></td><td>high</td><td>8.73</td><td>15.13</td><td>81,623</td><td>3,735</td><td>85,359</td><td>78.49</td></tr><tr><td></td><td>xhigh</td><td>8.00</td><td>10.00</td><td>83,655</td><td>4,013</td><td>87,668</td><td>94.38</td></tr></table>

## D.3.3 Evaluation cost

Table 11 reports the cost of the evaluated runs for models with a comparable USD list price.

Table 11: Evaluation cost. Rows descend by total cost.
<table><tr><td>System</td><td>Core</td><td>Tokens/task (k)</td><td>Price (USD/M)</td><td>Cost (USD)</td></tr><tr><td>Claude Fable 5</td><td>84.97</td><td>161.6</td><td>50.00</td><td>3,271.57</td></tr><tr><td>GPT-5.5</td><td>78.06</td><td>224.9</td><td>30.00</td><td>2,732.09</td></tr><tr><td>GPT-5.6 Sol</td><td>81.68</td><td>205.9</td><td>30.00</td><td>2,501.58</td></tr><tr><td>Claude Sonnet 4.6</td><td>76.16</td><td>331.0</td><td>15.00</td><td>2,011.07</td></tr><tr><td>GPT-5.6 Terra</td><td>80.06</td><td>182.6</td><td>15.00</td><td>1,109.03</td></tr><tr><td>Claude Haiku 4.5</td><td>72.23</td><td>477.0</td><td>5.00</td><td>966.02</td></tr><tr><td>GPT-40</td><td>39.23</td><td>127.4</td><td>10.00</td><td>515.92</td></tr><tr><td>GPT-5.6 Luna</td><td>35.66</td><td>156.9</td><td>6.00</td><td>381.31</td></tr><tr><td>DeepSeek V4 Flash</td><td>73.54</td><td>389.7</td><td>0.28</td><td>44.19</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>59.66</td><td>70.1</td><td>1.50</td><td>42.61</td></tr></table>

Quality and cost do not form a simple ladder. The strongest frontier evaluations exceed \$2,500, while GPT-5.6 Terra retains about 98% of GPT-5.6 Sol’s Core at 44% of its cost. The two Flash systems remain below \$45 but differ by almost 14 Core points. A long trace from a low-price model may still cost less than a short frontier run, so quality and cost should be reported together when planning replication.

## D.4 Tool, memory, and interaction ablations

Table 6 reports outcome and capability scores for each ablation; Table 12 shows how the same runs searched, computed, and consumed context. Token use is descriptive and does not affect scoring.

## D.4.1 Research behavior and context consumption

The trajectory diagnostics expose two different kinds of false productivity. Under aggressive compression, the agent lists files more than four times as often as with long retention and records more evidence references, yet receives the lowest process score. It is rebuilding context that the controller discarded, not broadening the analysis. The expert-only suite looks efficient for the opposite reason: it has the highest call success while inspecting the fewest distinct files. Its tools run cleanly, but the investigation is narrow. Activity counts become meaningful only when tied to state changes: a new source should resolve a claim, a computation should test a mechanism, and a repeated read should have a reason.

## D.4.2 Capability-conditioned effects

The largest capability losses occur when evidence must remain connected across several steps. With 170 visible tools, multi-source performance falls to 47.50 even though most other catalog configurations stay near or above 88. Aggressive compression loses 20.07 points in causal-chain reasoning relative to long retention. The two interventions damage continuity in different places. A large catalog makes the next action unstable; compression makes the meaning of earlier actions unstable. A single calculation can survive either disturbance because its inputs are local. A physical account cannot: its validity depends on remembering why several observations belong together.

Table 12: System-ablation trajectories. Per-task means for the panels in Table 6.
<table><tr><td></td><td colspan="3">Tools</td><td colspan="2">Evidence</td><td colspan="2">Actions</td><td colspan="2">Process</td><td colspan="3">Tokens</td></tr><tr><td>Condition</td><td>OK failed</td><td></td><td>Calls Calls Success (%)</td><td>Files</td><td>Evidence refs.</td><td>Lists Searches Python</td><td></td><td></td><td>Process (%)</td><td>Input Output</td><td></td><td>Total Cached</td></tr><tr><td>(a) Tool-catalog scale</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fixed tools (6)</td><td>15.802.15</td><td></td><td>92.52 12.75</td><td></td><td>19.801.95</td><td></td><td>4.45</td><td>5.45</td><td>65.62 219,639</td><td></td><td>9,918 229,55725,882</td><td></td></tr><tr><td>Fixed tools (10)</td><td>15.25</td><td>1.75</td><td>92.22 11.05</td><td></td><td>17.401.30</td><td></td><td>3.30</td><td>3.35</td><td>57.10 185,888</td><td></td><td>8,159194,04725,408</td><td></td></tr><tr><td>Fixed tools (20)</td><td>15.55</td><td>2.05</td><td>91.74 12.05</td><td></td><td>17.151.45</td><td></td><td>3.10</td><td>4.60</td><td>59.38 223,666</td><td></td><td>9,246 232,912 25,267</td><td></td></tr><tr><td>Fixed tools (40)</td><td>15.75</td><td>1.95</td><td>92.81 11.95</td><td></td><td>19.701.55</td><td></td><td>4.15</td><td>4.10</td><td>61.00 224,175</td><td></td><td>8,818 232,993 24,346</td><td></td></tr><tr><td>Fixed tools (70)</td><td>15.00</td><td>1.90</td><td>92.39 11.40</td><td></td><td>18.351.25</td><td></td><td>3.35</td><td>2.85</td><td>61.25 224,853</td><td></td><td>8,906 233,75926,445</td><td></td></tr><tr><td>Fixed tools (100)</td><td>16.35</td><td>2.00</td><td>92.54 12.00</td><td></td><td>20.251.10</td><td></td><td>3.90</td><td>3.60</td><td>63.12 268,52710,409278,93632,294</td><td></td><td></td><td></td></tr><tr><td>Full tool catalog (170)</td><td>13.90</td><td>1.50</td><td>95.04 9.55</td><td></td><td>13.15 1.05</td><td></td><td>1.85</td><td>2.55</td><td>55.88 264,983</td><td></td><td>8,233 273,21633,331</td><td></td></tr><tr><td colspan="10">(b) Semantic routing and tool composition</td><td></td><td></td><td></td><td></td></tr><tr><td>Top-10 dynamic router</td><td>14.65</td><td>2.15</td><td>93.16 12.40</td><td></td><td>18.651.40</td><td></td><td>3.50</td><td>4.70</td><td>66.25 205,90810,319 216,22729,286</td><td></td><td></td><td></td></tr><tr><td>Top-20 dynamic router</td><td>12.75</td><td>1.65</td><td>92.98 11.30</td><td></td><td>16.151.35</td><td></td><td>2.60</td><td>4.05</td><td>65.88 170,972</td><td></td><td>9,038 180,010 21,696</td><td></td></tr><tr><td>Workflow-aware router</td><td>15.20</td><td>1.95</td><td>93.38 12.25</td><td></td><td>21.651.40</td><td></td><td>4.65</td><td>3.45</td><td>56.75 210,928</td><td></td><td>9,304 220,23226,381</td><td></td></tr><tr><td>Expert-only suite</td><td>17.70</td><td>0.60</td><td>97.597.35</td><td></td><td>12.600.00</td><td></td><td>0.00</td><td>0.00</td><td>60.88 283,502 11,988 295,49024,192</td><td></td><td></td><td></td></tr><tr><td>General + expert suite</td><td>14.05</td><td>1.55</td><td>94.43 10.40</td><td></td><td>15.401.40</td><td></td><td>3.05</td><td>3.45</td><td>63.00 186,216</td><td></td><td>8,875195,09231,437</td><td></td></tr><tr><td>Atmospheric Python suite 14.60</td><td></td><td>1.35</td><td>94.78 10.35</td><td></td><td>18.401.45</td><td></td><td>4.10</td><td>2.85</td><td>60.12 191,208</td><td></td><td>7,593 198,80225,344</td><td></td></tr><tr><td>Atmospheric Fortran suite 14.55</td><td></td><td>1.85</td><td>93.68 11.25</td><td></td><td>17.101.55</td><td></td><td>3.35</td><td>3.30</td><td>57.88 195,439</td><td></td><td>9,823 205,26231,078</td><td></td></tr><tr><td colspan="10">(c) Context retention</td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive compression</td><td>24.60</td><td>1.95</td><td>93.88 12.50</td><td></td><td>34.905.40</td><td></td><td>6.25</td><td>4.00</td><td>51.62 187,492</td><td></td><td>9,545 197,03810,765</td><td></td></tr><tr><td>Short retained context</td><td>20.20</td><td>2.25</td><td>93.64 13.75</td><td></td><td>26.802.70</td><td></td><td>6.40</td><td>5.10</td><td>56.38 198,062</td><td></td><td>11,765 209,82714,682</td><td></td></tr><tr><td>Long retained context</td><td>13.35</td><td>1.85</td><td>92.77 11.80</td><td></td><td>16.951.20</td><td></td><td>3.70</td><td>4.45</td><td>63.38 218,347</td><td></td><td>9,189 227,53648,998</td><td></td></tr><tr><td>Full retained context</td><td>12.15</td><td>1.30</td><td>93.84 11.50</td><td></td><td>18.001.05</td><td></td><td>3.90</td><td>3.40</td><td>58.25 180,517</td><td></td><td>7,182 187,70080,819</td><td></td></tr><tr><td colspan="10">(d) Interaction budget</td><td colspan="3"></td><td></td></tr><tr><td>12-turn budget</td><td>9.95 0.30</td><td></td><td>97.36 8.50</td><td></td><td>13.051.15</td><td>2.70</td><td></td><td></td><td>49.50 101,826</td><td></td><td>4,437 106,26323,002</td><td></td></tr><tr><td>24-turn budget</td><td>12.90</td><td>1.65</td><td>93.11 10.80</td><td></td><td>16.901.35</td><td></td><td>4.20</td><td>1.30 3.30</td><td>52.88 153,261</td><td></td><td>8,143161,40432,346</td><td></td></tr><tr><td>96-turn budget</td><td>13.60</td><td>1.30</td><td>94.12 11.90</td><td></td><td>18.451.60</td><td></td><td>3.85</td><td>4.55</td><td>60.12 173,925</td><td></td><td>8,873 182,79827,763</td><td></td></tr><tr><td>Unbounded turn budget</td><td>15.201.80</td><td></td><td>94.22 12.30</td><td></td><td>21.401.60</td><td></td><td>5.15</td><td>4.35</td><td>64.50 211,751</td><td></td><td>9,429 221,18030,131</td><td></td></tr></table>

## D.5 Oracle, interface, and evidence robustness

Table 4 consolidates outcome scores, paired uncertainty, and capability slices for each intervention. Table 13 separately reports trajectory and resource effects. Confidence intervals resample paired task-level Core differences. Oracles provide guidance without answers, while robustness conditions alter only package evidence.

## D.5.1 Paired intervention uncertainty

Only the two evidence-localization oracles have intervals that exclude zero: relevant files [+5.48, +24.69] and the evidence map [+2.34, +21.25]. The stable gain comes from knowing where evidence is, not from receiving a generic plan. A plan names familiar analytical moves; localization resolves an event-specific uncertainty that scientific knowledge alone cannot settle. The other interventions remain suggestive because their intervals cross zero. On a diagnostic manifest, a few packages whose structure favors one interface can still move the mean substantially.

## D.5.2 Trajectory and resource effects

Oracle guidance shortens search without necessarily reducing cost. The evidence map lowers rounds and file reads, but the injected structure itself consumes context and takes time to interpret. Distractors produce the reverse pattern: the agent performs more actions and records more evidence references with almost no quality loss. Obvious clutter is therefore mostly an efficiency tax. A compact evidence description poses a different problem. The agent must decide whether to trust the summary or reopen the underlying source, and the latter is often necessary to recover metadata, aggregation choices, or conflicts. Search length and evidential control are related, but they are not interchangeable.

Table 13: Intervention trajectories. Panels match Table 4; values are per-task means.
<table><tr><td></td><td colspan="3">Execution</td><td colspan="3">Evidence and compute</td><td colspan="2">Latency</td><td colspan="3">Tokens</td></tr><tr><td>Intervention</td><td>Rounds</td><td>calls</td><td>Tool Success (%)</td><td>Reads Files</td><td>Evidence refs.</td><td></td><td>Python Seconds</td><td></td><td>Input Output</td><td></td><td>Total Cached</td></tr><tr><td colspan="10">(a) Answer-free information oracles</td><td></td><td></td><td></td></tr><tr><td>Relevant-file oracle</td><td>15.3014.30</td><td></td><td>92.17</td><td>5.15 11.30</td><td></td><td>16.10</td><td>4.15</td><td>217.26 181,939</td><td></td><td></td><td>5,572 187,51219,507</td></tr><tr><td>Evidence-map oracle</td><td>11.10 10.10</td><td></td><td>93.28</td><td>3.10 8.55</td><td></td><td>11.10</td><td>2.75</td><td>436.56 290,368</td><td></td><td>5,050 295,41862,950</td><td></td></tr><tr><td>Tool-subset oracle</td><td>17.80 16.80</td><td></td><td>93.18</td><td>6.60 13.35</td><td></td><td>21.60</td><td>3.90</td><td>212.74 205,252</td><td></td><td>5,628 210,88027,738</td><td></td></tr><tr><td>Planning oracle</td><td>16.95 15.95</td><td></td><td>91.49</td><td>5.45 12.65</td><td></td><td>18.30</td><td>4.05</td><td>185.18 196,489</td><td></td><td>5,817 202,30625,190</td><td></td></tr><tr><td colspan="10">(b) Tool-interface presentation</td><td colspan="3"></td></tr><tr><td>Hierarchical tool catalog</td><td>16.60 15.60</td><td></td><td>91.97</td><td>5.75 12.45</td><td></td><td>17.75</td><td>3.85</td><td>197.77 182,350</td><td></td><td>6,000 188,351 20,416</td><td></td></tr><tr><td>Descriptive tool names</td><td>18.25 17.25</td><td></td><td>93.42</td><td>5.40 12.25</td><td></td><td>19.65</td><td>3.65</td><td>150.14 202,390</td><td></td><td>5,462 207,85328,749</td><td></td></tr><tr><td>Dynamic tool descriptions</td><td>17.55 16.55</td><td></td><td>92.12</td><td>4.00 12.85</td><td></td><td>19.95</td><td>3.65</td><td>182.42 199,673</td><td></td><td>5,036 204,70923,053</td><td></td></tr><tr><td>Opaque tool names</td><td>17.90 16.90</td><td></td><td>92.44</td><td>7.00 10.85</td><td></td><td>16.50</td><td>4.30</td><td>234.10 215,650</td><td></td><td>6,294 221,94323,027</td><td></td></tr><tr><td colspan="10">(c) Evidence perturbations</td><td></td><td></td></tr><tr><td>Equivalent unit change</td><td>16.4515.45</td><td></td><td>90.70</td><td>5.20 13.25</td><td></td><td>19.25</td><td>3.90</td><td>176.16 184,679</td><td></td><td>5,153189,832 19,904</td><td></td></tr><tr><td>Added distractor files</td><td>19.60 18.60</td><td></td><td>89.97</td><td>6.20 13.10</td><td></td><td>22.50</td><td>4.15</td><td>162.54 220,098</td><td></td><td>5,658 225,75625,792</td><td></td></tr><tr><td>Missing evidence</td><td>17.4016.40</td><td></td><td>89.12</td><td>5.65 12.05</td><td></td><td>21.20</td><td>3.55</td><td>181.47 198,173</td><td></td><td>5,219 203,39219,546</td><td></td></tr><tr><td>Truncated evidence</td><td>16.10 15.10</td><td></td><td>91.65</td><td>4.25 12.75</td><td></td><td>22.10</td><td>3.55</td><td>265.79184,244</td><td></td><td>5,827190,071 16,781</td><td></td></tr><tr><td>Conflicting evidence</td><td>16.50 15.50</td><td></td><td>92.04</td><td>5.65 12.55</td><td></td><td>19.70</td><td>3.25</td><td>155.59182,815</td><td></td><td>5,875 188,68918,624</td><td></td></tr><tr><td colspan="10">(d) Optional meteorological access</td><td></td><td></td></tr><tr><td>Compact meteorology suite</td><td>17.95 16.95</td><td></td><td>94.90</td><td>5.85 12.75</td><td></td><td>20.60</td><td>4.05</td><td>223.20 201,057</td><td></td><td>5,822 206,87927,571</td><td></td></tr><tr><td>Cue-routed meteorology suite</td><td>15.65 14.65</td><td></td><td>92.50</td><td>5.65 11.50</td><td></td><td>17.85</td><td>3.15</td><td>238.22 170,431</td><td></td><td>5,875 176,30625,882</td><td></td></tr></table>

## D.5.3 Capability-conditioned intervention effects

The effective oracles help most with spatiotemporal reconstruction: both exceed 90, compared with 64.40 at baseline. File localization directly removes uncertainty about which records contain the event window and region. Ranking and decision improve less because the oracle does not say how competing quantities should be weighted. Opaque tool names damage the same capability most strongly. Reconstructing a window rarely depends on one call; it requires a sequence of filtering, reading, aggregation, and comparison. Losing the semantic role of any operation makes the resulting timeline harder to recover and harder to audit.

## D.6 Professional meteorological environments

The professional-environment study separates method availability from execution. Compliance records whether the required interactions occurred; call success records whether they completed without adapter, argument, or runtime failure.

## D.6.1 Compliance, execution reliability, and implemented scope

Nearly every run attempts the required specialist interaction, so willingness to call the tool is not the bottleneck. Execution is. Specialist-call success ranges from 20.29% for PyET to 97.56% for R climate, and high-success environments usually need fewer repair rounds and smaller contexts. The ranking should not be read as a comparison of languages. It reflects the overlap between the audited operations and the data actually present in the packages. R climate, Julia, and CDO often accept the available time series or grids directly; other adapters require variables, record lengths, or sampling structures that many event packages do not contain.

Table 14: Scientific-environment outcomes and execution. Final Answer, Process, and Core scores are reported with per-task compliance, successful specialist calls, resource use, and implemented operations.
<table><tr><td></td><td></td><td colspan="3">Outcome</td><td colspan="2">Execution</td><td colspan="3">Resources</td></tr><tr><td>Environment</td><td>Operations</td><td></td><td>Answer Process</td><td>Core</td><td>(%)</td><td>Comply Success (%)</td><td>Rounds</td><td>Tokens</td><td>Latency (s)</td></tr><tr><td colspan="10">(a) Python scientific libraries</td></tr><tr><td>MetPy</td><td>rain summary; profiles; kinematics</td><td>68.69</td><td>70.38 69.53</td><td></td><td>100</td><td>38.83</td><td></td><td>25.00311,043</td><td>255.60</td></tr><tr><td>xclim</td><td>heat/rain extremes; wet/dry spells</td><td>67.77</td><td>69.62 68.70</td><td></td><td>95</td><td>38.78</td><td></td><td>24.50299,074</td><td>268.44</td></tr><tr><td></td><td>climate-indices SPI; SPEI; PET</td><td>64.50</td><td>66.7565.62</td><td></td><td>95</td><td>22.60</td><td></td><td>30.55374,199</td><td>306.69</td></tr><tr><td>PyET</td><td>Penman-Monteith; Hargreaves</td><td>65.34</td><td></td><td>67.3866.36</td><td>95</td><td>20.29</td><td></td><td>32.30387,917</td><td>315.96</td></tr><tr><td>thermofeel</td><td>UTCI; heat index; wind chill</td><td>66.78</td><td></td><td>68.75 67.76</td><td>100</td><td>26.85</td><td></td><td>28.55 346,924</td><td>357.88</td></tr><tr><td>pyextremes</td><td>block maxima; POT; return levels</td><td>59.02</td><td></td><td>60.0059.51</td><td>100</td><td>34.19</td><td></td><td>28.05 363,839</td><td>289.16</td></tr><tr><td>xskillscore</td><td>RMSE; correlation; Brier; CRPS</td><td>63.37</td><td></td><td>64.1263.75</td><td>95</td><td>57.14</td><td></td><td>23.70297,724</td><td>231.47</td></tr><tr><td colspan="10">(b) Compiled and domain-language environments</td></tr><tr><td>Julia</td><td>summary; integration; sensitivity</td><td>71.77</td><td>70.25 71.01</td><td></td><td>100</td><td>91.11</td><td></td><td>21.50256,852</td><td>204.47</td></tr><tr><td>R climate</td><td>summary; SPEI; GEV; trend</td><td>68.55</td><td>71.38 69.96</td><td></td><td>100</td><td>97.56</td><td></td><td>21.70259,984</td><td>190.20</td></tr><tr><td>CDO</td><td>temporal and field aggregates</td><td>66.91</td><td>69.5068.20</td><td></td><td>100</td><td>81.63</td><td></td><td>22.30268,906</td><td>197.90</td></tr><tr><td>NCO</td><td>inspect; weighted mean; derive field</td><td>67.39</td><td>65.0066.19</td><td></td><td>100</td><td>35.59</td><td></td><td>26.65321,773</td><td>230.95</td></tr><tr><td>NCL</td><td>summary only</td><td>65.03</td><td>65.0065.01</td><td></td><td>95</td><td>40.62</td><td></td><td>25.70308,559</td><td>270.94</td></tr><tr><td>Fortran</td><td>rain; kinematics; moisture transport</td><td>64.46</td><td>64.2564.35</td><td></td><td>95</td><td>43.48</td><td></td><td>25.70301,236</td><td>222.60</td></tr></table>

Applicability and reproducibility. Environment names refer to the audited adapters in Table 14, not to every feature of the upstream packages. The outcome columns make the distinction explicit: R climate has the highest call success, while Julia has the highest Core. A successful call proves only that code ran. It does not establish that the estimator fits the record, that the sampling geometry supports the comparison, or that the returned value answers the question. Those checks sit between execution and scientific use, and the gap between execution reliability and final quality shows why they cannot be collapsed into one metric.

Each adapter therefore records normalized arguments, status, output, and errors. It also declares the operations it implements and validates package-scoped inputs before execution. This makes a justified failure reproducible: another reviewer can see that a method was rejected because the record was too short or a required variable was absent. Merely rerunning a command is weaker. Scientific reproducibility requires reconstructing why the operation was admissible and how its output entered the claim.

## E Prompts and execution contracts

The prompts are grouped by role: package-scoped solving and review, answer–process evaluation, and benchmark construction. Placeholders mark values supplied by the harness. The solver sees no demonstration, hidden reference, or recommended file. Appendix G presents representative runs.

Runtime visibility. The solver sees the public question, event-package root, exposed tool schemas, and packagerelative observations. Ground truth, reference solutions, rubrics, computation programs, judge records, and other model answers remain hidden. The trace retains both successful and failed tool calls. Depending on the condition, the controller may request two format repairs, run a bounded evidence review, require a specialist operation, or reserve the final call for a schema-valid submission. None of these transitions adds scientific evidence. Appendix C.2 records their limits.

![](images/d026068bbac9ba18edce29ba54116e8249cf300133d772ace17a162b736447fa.jpg)

<table><tr><td>PROMPT</td><td>RUNTIME Package-scoped research suite</td><td>continued</td></tr><tr><td colspan="3">Task id: &lt;&lt;TASK_ID&gt;&gt;</td></tr><tr><td colspan="3">Event id: &lt;&lt;EVENT_ID&gt;&gt; Question path: &lt;&lt;QUESTION_PATH&gt;&gt;</td></tr><tr><td colspan="3">Event package path: &lt;&lt;PACKAGE_ROOT&gt;&gt;</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Question: &lt;&lt;PUBLIC_QUESTION&gt;&gt;</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Evidence boundary: Strict package-only mode: only the current question and event package evidence</td></tr><tr><td colspan="3">are visible.</td></tr><tr><td colspan="3">Tools:</td></tr><tr><td colspan="3">&lt;&lt;REGISTERED_TOOL_CATALOGUE&gt;&gt;</td></tr><tr><td colspan="3">TOOL ACTION CONTRACT</td></tr><tr><td colspan="3">Action tool_call</td></tr><tr><td colspan="3">Tool one registered tool name</td></tr><tr><td colspan="3">Arguments only the arguments required by that tool Reason why this call is needed for the current analysis</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">FINAL ANSWER CONTRACT Action final_answer</td></tr><tr><td colspan="3">Answer the exact structure requested by the public question</td></tr><tr><td colspan="3">Confidence one numeric confidence value Evidence for each claim: package-relative source file and support type</td></tr><tr><td colspan="3">(direct metric, report claim, metadata, image context, table</td></tr><tr><td colspan="3">value, derived metric, or unsupported boundary) Research</td></tr><tr><td colspan="3">a brief ordered summary of the reasoning Limitations important unsupported or unresolved elements</td></tr><tr><td colspan="3">Start by inspecting the package inventory, then choose only the files needed</td></tr><tr><td colspan="3">for this event-specific disaster analysis.</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">REVIEW Final evidence review</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Begin a final evidence-review phase before submitting the answer. Treat the prior answer as a draft: re-check its evidence provenance, formulas, units,</td></tr><tr><td colspan="3">intermediate values, competing physical explanations, and every required</td></tr><tr><td colspan="3">output field against the package. Use additional tools when verification is useful. Return either one tool action or a revised final answer; do not merely</td></tr><tr><td colspan="3">endorse the draft. You have at most &lt;&lt;MAX_REVIEW_TURNS&gt;&gt; review turns.</td></tr><tr><td colspan="3">PROMPT EVALUATION Joint answer and process judge</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">JUDGE Answer and research-process evaluation</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">SYSTEM You are a strict but fair evaluator for a local extreme-event benchmark. Your</td></tr><tr><td colspan="3">job is to judge answer correctness and task-rubric quality. Use partial 0-100 scores and avoid old deterministic exact matching.</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">CALIBRATION CORE</td></tr><tr><td colspan="3">Evaluate this model response for an extreme-event evidence-and-reasoning</td></tr><tr><td colspan="3">benchmark. Make one combined evaluation and produce two core 0-100 scores.</td></tr><tr><td colspan="3">- answer_correctness_score is the direct partial-credit comparison between the</td></tr><tr><td colspan="3">model&#x27;s final answer and the ground truth. Treat the answer as a set of units:</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">required fields, labels, decisions, values, units, orderings, alternatives,</td></tr><tr><td colspan="3">stage boundaries, and required conclusions. Missing required fields are</td></tr><tr><td colspan="3">incorrect; correct prose cannot replace a missing structured field.</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">llm_rubric_score must come from the task&#x27;s own 20-point rubric. Compare the</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">answer and visible trace against every criterion, assign earned points out of</td></tr><tr><td colspan="3">20, and convert the result to percent.</td></tr></table>

## PROMPT EVALUATION Joint answer and process judge

intermediate values, transformations, and conclusions must agree.

for missing process evidence, and an attempted method cannot receive full

Runtime payload:

<<TASK\_METADATA>>

<<QUESTION>>

<<GROUND\_TRUTH>>

<<SOLUTION\_AND\_20\_POINT\_RUBRIC>>

<<MODEL\_FINAL\_ANSWER>>

<<MODEL\_TOOL\_TRACE>>

<<MODEL\_RUN\_NOTES>>

## EVALUATION RECORD

Scores answer correctness and rubric-based process quality

Answer units each required unit and whether it was satisfied

Rubric ledger criterion-level points earned out of 20

Diagnosis verdict, critical errors, missing elements, unsupported claims

Rationale one sentence explaining the score pair

Benchmark construction suite

QUALITY Task construction acceptance review

ROLE

You are the final quality reviewer for one extreme-event benchmark task.

INPUTS

\- The public question (question\_en.md)

\- The expert solution (solution\_en.md)

\- The executable ground-truth program (compute\_gt.py)

\- The generated scoring contract (computed\_gt.json)

\- The optional review record (review.json)

Verify that the five artifacts describe one event-specific scientific target. Do not approve a rewritten public question when the solution or GT still scores an older package-inspection objective.

## ACCEPTANCE CHECKS

1. The public question reads like a real disaster-analysis assignment.

2. It contains no local path, visible CSX identifier, package inventory, available-material wording, AOI, quality-gate, evidence-boundary, or benchmark phrasing, and it does not reveal the useful files.

3. The solution contains Final Answer, Key Computations, Reasoning Path, Computed Interpretation, and Scoring Rubric.

4. computed\_gt.json declares schema\_version 1.0, task and event identifiers, task type, evaluation mode, primary GT, key values, and a top-level rubric.

5. Deterministic tasks contain compact machine-checkable targets. Open expert tasks contain concrete judge anchors rather than a fabricated exact essay.

6. The rubric has 5-8 criteria whose points sum exactly to 20.

8. compute\_gt.py runs successfully and regenerates the scoring contract.

OUTPUT

Return changed or reviewed file paths, event and hazard type, task type,

evaluation mode, acceptance status for every check, and any remaining ambiguity,

missing-data risk, temporal/spatial mismatch, or unsupported overclaim.

RUBRIC Event-specific rubric construction

ROLE

<table><tr><td>PROMPT</td><td>CONSTRUCTION</td><td>Benchmark construction suite</td></tr><tr><td colspan="3">You are constructing the hidden scoring contract for one benchmark task.</td></tr><tr><td colspan="3">TASK Create a 20-point rubric that can be applied without guessing the author&#x27;s</td></tr><tr><td colspan="3">intent. Use 5-8 small, independently judgeable criteria. Every criterion must contain name, points, description, and partial_credit. The point values must sum</td></tr><tr><td colspan="3">exactly to 20.</td></tr><tr><td colspan="3">REQUIRED COVERAGE</td></tr><tr><td colspan="3">- 3-4 points: final conclusion, numeric target, threshold state, proof result, ranking, or concise canonical stance.</td></tr><tr><td colspan="3">- 3-5 points: required quantitative extraction or calculation, including units,</td></tr><tr><td colspan="3">tolerances, and windows where relevant.</td></tr><tr><td colspan="3">- 2-4 points: formula, index, process-window, causal-chain, or score-ledger logic. - 2-4 points: fusion of reports, time series, rasters, imagery, exposure,</td></tr><tr><td colspan="3">infrastructure, or trajectory evidence into checkable anchors.</td></tr><tr><td colspan="3">- 1-3 points: a compact event-specific consequence that follows from the computed evidence.</td></tr><tr><td colspan="3">- 1-3 points: rejection of plausible competing mechanisms or overclaims. - 1-2 points: complete, concise output in the requested structure.</td></tr><tr><td colspan="3">HARD RULES</td></tr><tr><td colspan="3">Do not use vague criteria such as &quot;good reasoning&quot; or &quot;uses evidence well.&quot; Do not award points for choosing an option after a converted task has become a</td></tr><tr><td colspan="3">numeric or structured analysis. A criterion that depends on an incorrect</td></tr><tr><td colspan="3">scientific result cannot receive full process credit. For llm_judge tasks, include canonical_stance, required_claims, required_metrics, forbidden_claims,</td></tr><tr><td colspan="3">and acceptable_variants in primary_gt. For hybrid tasks, preserve both the deterministic core and the reasoning anchors.</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">RUBRIC RECORD Total points exactly 20</td></tr><tr><td colspan="3">Criteria 5-8 independently judgeable items</td></tr><tr><td colspan="3">Each item event-specific name; points; full-credit requirement; partial-credit and error rule</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">SOURCE Event authenticity and source audit</td></tr><tr><td colspan="3">ROLE</td></tr><tr><td colspan="3">You are auditing the real-event anchor of one event package.</td></tr><tr><td colspan="3">QUESTIONS 1. Did the event occur? Require at least one traceable source confirming the</td></tr><tr><td colspan="3">event name, time window, or principal affected region. 2. Is the source reliable? Prefer government agencies, international organizations, peer-reviewed work, and authoritative scientific institutions;</td></tr><tr><td colspan="3">then disaster catalogues and major news. Encyclopedias and generic search pages are supporting evidence only. 3. Does the source actually correspond to this event? Check event name, time,</td></tr><tr><td colspan="3">place, and hazard type. A source about the same hazard family but a different time or place is not a strong match.</td></tr><tr><td colspan="3">OUTPUT FIELDS</td></tr><tr><td colspan="3">- existence_verdict: confirmed I likely I weak I not_confirmed - temporal_match: exact | near | broad | mismatch | unknown</td></tr><tr><td colspan="3">- spatial_match: exact | regional I broad | mismatch I unknown - hazard_match: exact | compound_related I broad_family l mismatch | unknown</td></tr><tr><td colspan="3">- source_reliability: high I medium I low I mixed</td></tr><tr><td colspan="3">- release_decision: keep | keep_with_minor_fix | needs_stronger_source |</td></tr><tr><td colspan="3">replace_event</td></tr><tr><td colspan="3">- concise evidence and repair notes</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">DECISION RULES</td></tr><tr><td colspan="3">Use keep when the event is confirmed, at least two of temporal/spatial/hazard</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">matching are exact or regional, and the preferred evidence is authoritative.</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Use keep_with_minor_fix when the event is real but URL, naming, or scale needs a</td></tr><tr><td colspan="3">small correction. Use needs_stronger_source when current support is generic, secondary, or overly broad. Use replace_event when the event cannot be confirmed</td></tr></table>

## F Representative case set

The six cards below index remote-sensing reconciliation, urban pluvial causality, evidence substitution, multi-agent quantitative decomposition, integrated heat research, and a partial scientific failure. Together they cover five system families without repeating the full records that follow.

## REPRESENTATIVE CASE SET

## FRONTIER HOSTED GPT-5.6 Sol

CSX-189\_Q1 Core 99.50

Uljin wildfire consistency. Report claims, wind and precipitation, Sentinel-2 dNBR, annual change, smoke transport, and bounded impact claims. TRACE Sixteen successful calls, including a failed semantic search recovered by a filename query and a final provenance check.

## \_ GENERAL HOSTED Tencent Hy3

CSX-059\_Q2 Core 100.00

NYC Ida pluvial overload. An hourly rainfall record inside a two-day rain shield, population load, surface change, and a controlled future perturbation. TRACE Ten evidence calls followed by one deterministic calculation of every ratio, normalization term, and scenario score.

## OPEN-WEIGHT LARGE Qwen3-235B

CSX-006\_Q2 Core 93.93

Heat-source substitution audit. Variable-level admissibility of an ERA5-Land aggregate substituted for local hourly apparent-heat and warm-night data. TRACE Eight calls covering inventory, source comparison, schema checks, corrected heat metrics, and numerical source impact.

MULTI-AGENT FRAMEWORK GeoMMAgent CSX-063\_Q1 Core 83.50 Zhengzhou threshold ledger. Reported burst and multiday rainfall, product ordering, exposure thresholds, and a conjunctive flood diagnosis. TRACE Four coordinator roles and nine package-tool calls for retrieval, calculation, scientific synthesis, and schema assembly.

## RESEARCH FRAMEWORK OpenResearcher

CSX-018\_Q2 Core 86.92

Sahel heat-response model. Hourly heat, wet bulb, warm nights, exposure, services, roads, surface stability, two indices, and a +1 C counterfactual. TRACE Twenty-one calls, including one failed heat script, an explicit repair, and independent checks of peaks, duration, density, scores, and deltas.

## GENERAL HOSTED GPT-4o

CSX-312\_Q2 Core 41.25

Partial scientific failure. The model identifies the coupled Kilauea mechanism but reports lava effusion rate instead of the required summit-collapse-to-flow volume ratio. TRACE Nineteen model rounds and nine tool calls: seven succeed, two guessed paths fail, and the final answer earns Answer 40.00 and Rubric 42.50.

## G Detailed research trajectories

Appendix F identifies the six selected systems and tasks. This section expands each card into its scientific target, evidence sequence, calculations, and decision. The Uljin case retains the complete task and action order; the remaining records focus on the steps that determine the result.

## G.1 Full case: Uljin report-to-burn consistency

## QUESTION CSX-189\_Q1

## SCIENTIFIC TARGET

Uljin wildfire report-to-burn consistency ledger

A technical review team is checking whether the March 4−13, 2022 Uljin wildfire is numerically consistent with a dry, wind−assisted burn−scar diagnosis rather than a late−rain−control or image−baseline alternative.

EVIDENCE SCOPE

Use only package−local evidence. Do not use point−weather products for this task; build the ledger from the event report, package gridded precipitation and wind summaries, Sentinel−2 dNBR, and annual embedding−change context.

## QUESTION CSX-189\_Q1

continued

## REQUIRED COMPUTATION Compute these ledger values:

− report\_dry\_wind\_flag: 1 if the report links the event to strong winds and dry weather, otherwise 0.

− smoke\_transport\_flag: 1 if the report states that smoke moved toward southern Japan, otherwise 0.

− era5\_mean\_wind\_speed\_mps: vector speed from mean 10 m u and v components.

− era5\_wind\_bearing\_to\_deg: direction toward which the mean wind vector points, degrees clockwise from north.

− precip\_mean\_spread\_mm: spread between ERA5−Land, GPM IMERG, and CHIRPS event precipitation means.

− dnbr\_mean: mean dNBR over the event burn−change window.

− dnbr\_max\_to\_mean\_ratio: dNBR maximum divided by dNBR mean.

− dnbr\_to\_annual\_change\_mean\_ratio: dNBR mean divided by annual 1−minus−cosine mbedding

− reported\_charred\_area\_km2: reported hectares converted to square kilometers.

## MECHANISM CHECKS Apply these tests:

− report\_dry\_wind\_mechanism\_test: pass if report\_dry\_wind\_flag = 1.

− smoke\_transport\_context\_test: pass if smoke\_transport\_flag = 1.

− positive\_heterogeneous\_burn\_test: pass if dnbr\_mean > 0 and

dnbr\_max\_to\_mean\_ratio >= 2.0.

− late\_rain\_control\_test: pass only if rainfall is the dominant control;   
otherwise fail.

− nuclear\_damage\_overclaim\_test: pass if the answer avoids inferring nuclear− plant damage or radiation release.

## OUTPUT

Evidence ledger

− Report/dry−wind flag and smoke−transport flag

− Mean ERA5 wind speed and bearing

− Precipitation−mean spread across the three gridded products

− Mean dNBR, dNBR maximum−to−mean ratio, and dNBR/annual−change ratio

− Reported charred area in square kilometres

Decision checks

− Report mechanism; smoke context; positive heterogeneous burn

− Late−rain control; bounded nuclear−impact claim

INTERPRETATION

− One or two sentences tied directly to the computed ledger

## CASE Research record: CSX-189\_Q1

## COMPLETE ACTION ORDER; OBSERVATION BODIES ABRIDGED

list\_package\_files

Discover Enumerated all 39 package files.

read\_text\_file

Anchor Confirmed Uljin, South Korea, 4–13 March 2022.

search\_package\_files

Mechanism Located report language linking strong winds and dry weather.

search\_package\_files

Transport Located the statement that smoke moved toward southern Japan.

search\_package\_files

Impact quantity Recovered nearly 17,000 charred hectares.

read\_text\_file

ERA5-Land Read mean precipitation and mean 10 m wind components.

read\_text\_file

GPM IMERG Read event precipitation mean of 2.884 mm.

read\_text\_file

CHIRPS Read event precipitation mean of 4.456 mm.

read\_text\_file

Sentinel-2 dNBR Read mean 0.303255 and maximum 0.708527.

search\_package\_files

Embedding search Initial semantic query returned no match.

![](images/7804ba5370fab1eb0a1dcb5c79ee01aa06d3941c59b53b70e1b99b704efe985e.jpg)

![](images/dddcb8847190021b8600e2884e532ab20251f25bd26278b9bf9f82b5c93b8acd.jpg)

## CASE Research record: CSX-189\_Q1

continued

![](images/10d886dd8a6412fde8ea36fc41a2c6f8f84bec11903b87582bb0fd710ba2bd36.jpg)

search\_package\_files

read\_text\_file

Recovery A filename-oriented query located annual cosine-change statistics.

13

Annual context Read mean 1-minus-cosine change of 0.013761.

search\_package\_files

14

read\_text\_file

python\_exec

Smoke check Confirmed the report-side smoke passage.

Full report Read the authoritative HTML surrounding all extracted claims.

Deterministic ledger Computed wind speed 1.392 m/s, bearing 331.253 degrees, precipitation spread 6.389 mm, dNBR heterogeneity 2.336, dNBR/annual-change ratio 22.037, and 170 km2 charred area.

![](images/c6acc23dbda539e695f114018b06cc0aba1760070e6a1241dab8202bf6ac03e5.jpg)

search\_package\_files

Final provenance check Re-queried the most easily misquoted smoke-transport claim before finalization.

COMPUTED LEDGER

DRY-WIND FLAG 1 SMOKE FLAG 1 MEAN WIND 1.392043 m/s BEARING 331.252581 deg

PRECIPITATION SPREAD 6.389107 mm MEAN dNBR 0.303255 dNBR MAX/MEAN 2.336407

dNBR/ANNUAL CHANGE 22.037202 CHARRED AREA 170.0 km

PASS Report mechanism PASS Smoke context PASS Heterogeneous burn

FAIL Late-rain control PASS Bounded nuclear-impact claim

EVIDENCE RECONCILIATION

The burn scar is not accepted on appearance alone. The trace anchors two report claims, reconciles three precipitation products, and keeps event-window dNBR separate from annual embedding change. Step 10 also preserves the failed semantic search and the filename-based recovery that follows.

## CASE NYC Ida: urban pluvial overload

TASK

SCIENTIFIC TARGET Reconstruct the New York City phase of post-tropical Cyclone Ida as a record hourly burst embedded in a broader two-day rain shield.

EVIDENCE SCOPE Keep station-scale hourly rainfall distinct from the wider gridded rainfall field, then add population exposure and surface-change context.

REQUIRED COMPUTATION Compute the baseline response stress and repeat it under the prescribed future perturbation.

RESEARCH TRAJECTORY: TEN EVIDENCE CALLS AND ONE CALCULATION

![](images/e82b80feb9356e41b1e8107254c4d254b233015ed4bb30b1d68a9c874f2a5640.jpg)

list + anchor

![](images/d30d49694748fe9cf722eb8942293ed01354e4c0eb100ea1a013b5e1449c3665.jpg)

read report

3

read 3 products

4

read exposure

![](images/9a982dd485d71b7d3ce6a59b0e99452ddb84203c42f4dd7fcba6ad185fbbec13.jpg)

python\_exec

EVIDENCE, METRICS, AND DECISION

Window Enumerated 39 files and fixed the inclusive 1–2 September 2021 event window.

Local forcing Recovered the Central Park hourly record and the conservative 10-inch regional reference.

Rain shield Read GPM IMERG, CHIRPS, and ERA5-Land event means and maxima.

Context Read 5.013 million exposed people, Sentinel-1 mean change of 0.234 dB, and annual embedding change of 0.022.

Ledger + scenario Converted units, reconciled products, computed ratios and normalized terms, and reran the weighted index under the prescribed perturbation.

## CASE NYC Ida: urban pluvial overload

continued

WINDOW 2 days RECORD HOUR 88.138 mm REGIONAL REFERENCE 254.000 mm GRID PEAK 136.888 mm

GRID MEAN 53.427 mm HOUR/PEAK 0.644 REGIONAL/PEAK 1.856 BURST/AREAL DAY 3.299

POPULATION 5.013 million RAIN LOAD 267.812 million-person-mm SURFACE CHANGE 0.337

DECISION The baseline response-stress index is 72.6; it rises to 80.1 under the future perturbation (+7.5). The local hourly record is not treated as a substitute for the wider gridded rain shield.

## CASE Heat-source substitution audit

TASK

SCIENTIFIC TARGET Determine whether a same-window ERA5-Land aggregate can replace the local hourly heat source.

EVIDENCE SCOPE Audit variable availability for apparent-temperature duration and warm-night exposure rather than judging a source by filename.

DECISION RULE Accept the substitution only when the replacement preserves every variable needed for the requested metrics.

RESEARCH TRAJECTORY: EIGHT SUCCESSFUL CALLS

1

list\_package\_files

Discover Enumerated 37 files and identified the two competing heat products.

2

read anchor

Window Fixed 31 March–4 April 2024.

3

read sources

Compare The local source contains hourly temperature, humidity, apparent temperature, and daily minima; the aggregate lacks the last two required roles.

4

python\_exec

Schema audit Confirmed that apparent temperature is the first missing field and daily minimum temperature is also absent.

python\_exec

Correct + compare Recomputed the target heat metrics and measured the peak-temperature error introduced by the substitute.

## EVIDENCE, METRICS, AND DECISION

MAX AIR 44.6 C MAX APPARENT 43.2 C HOURS APPARENT ≥ 40 C 16 WARM NIGHTS ≥ 27 C 5

AGGREGATE MAX 35.3 C PEAK DIFFERENCE 9.3 C

MISSING FIELDS apparent temperature; daily minimum

SOURCE SWAP FAILS The aggregate cannot compute apparent-heat duration or warm nights. The corrected label is sustained\_apparent\_heat\_with\_warm\_night\_exposure.

## CASE Zhengzhou: coordinated threshold ledger

TASK

SCIENTIFIC TARGET Test whether the 17–23 July 2021 Zhengzhou flood combined an exceptional hourly burst, multiday extreme rainfall, and dense urban exposure.

REQUIRED COMPUTATION Build the rainfall-ratio, product-order, population, and infrastructure threshold ledger.

DECISION RULE Assign the final label only when all four numerical thresholds pass.

RESEARCH TRAJECTORY: FOUR ROLES AND NINE PACKAGE-TOOL CALLS

![](images/31623f1131d7e4971236cc59079d18adde5b4cc2cbdd6a2651ac83fc1653e86b.jpg)

![](images/de577da131f561d149e11b0d62305d605fc98ac1139cd955eb2cbaf66e931653.jpg)

## CASE Zhengzhou: coordinated threshold ledger

continued

![](images/7d84f7e068ec5cbc642765aa93a344d6168d0f758a0544de0113f51062ed8889.jpg)

Agent 0: retrieve

Agent 1: compute

Evidence Read the event report, three precipitation products, WorldPop, and OSM with strict provenance.

Ledger Calculated rainfall ratios, product order, and the four threshold flags.

3

Agent 2: synthesize

Mechanism Joined physical rainfall evidence with population and road exposure.

Agent 3: assemble

python\_exec

Output Built the requested schema without changing the shared evidence boundary.

Verify Recovered hour share 0.3272, annual ratio 0.963, and four true threshold flags.

EVIDENCE, METRICS, AND DECISION

THREE-DAY RAIN 617.1 mm ONE-HOUR RAIN 201.9 mm HOUR SHARE 0.3272 ANNUAL RATIO 0.963

MEAN ORDER GPM > CHIRPS > ERA5-Land MAX GRID MEAN 460.5514 mm MAX GRID 484.255 mm

POPULATION 5.2625 million HIGHWAYS 618 AMENITIES 50 TUNNELS 2

ALL FOUR THRESHOLDS PASS Hour share ≥ 0.30, annual ratio ≥ 0.90, population ≥ 5 million, and highway features ≥ 200.   
Final label: short\_burst\_multiday\_urban\_exposure.

CASE Sahel heat: response load and warming scenario

TASK

SCIENTIFIC TARGET Model a five-day Sahel heat episode and its response burden.

EVIDENCE SCOPE Combine hourly heat and Stull wet-bulb temperature, warm nights, population, services, roads, gridded context, and surface stability.

REQUIRED COMPUTATION Calculate both response indices at baseline and under a +1 C counterfactual.

RESEARCH TRAJECTORY: TWENTY-ONE CALLS

1

list + open

![](images/2f97951ff264887fa38070d0e3ebb97cac4baa7944072bbfda7deb8e9a387b24.jpg)

python: exposure

Core inputs Read hourly heat, WorldPop, AOI geometry, and OSM infrastructure.

![](images/474b7cad985f382422d8908c447d36bb502bae62eef4f110cfd392bb4bc6cad7.jpg)

python: heat

Counts Computed schools, health facilities, shelters, major roads, AOI area, and population density.

![](images/6d4c8ea4fb62d5f9c15097daa7b6c86120502834ce286e7ec19a3745581d2e66.jpg)

python: repair

Failed attempt The first heat script failed because it assumed the wrong hourly field names.

![](images/027de495d800aa2c1e20c154cf1a2d0c938d15927688fc901277810afc049f54.jpg)

open products

Schema recovery Re-read the actual fields and computed Stull wet-bulb temperature and warm nights.

![](images/f1a9bcec9bd4db5433785806b627ead2bb9a0adba5d11de0782c11014d1d73f5.jpg)

python checks

Context Read ERA5-Land temperature, GPM and CHIRPS precipitation, and Sentinel-2 dNBR.

![](images/8e774a765f71b820e2150ee371df31d23ea0494e54393d6cad763de086246490.jpg)

python scenario

Heat metrics Independently verified air and apparent peaks, duration, degree-hours, and density.

Indices Rechecked baseline and +1 C scores, threshold counts, and deltas.

HEAT, EXPOSURE, AND SURFACE CONTEXT

MAX AIR 44.6 C MAX APPARENT 43.2 C MAX WET BULB 16.3605 C HOURS ≥ 40 C 16 DEGREE-HOURS 24.4  
WARM NIGHTS 5

POPULATION 170,311.46 DENSITY 3.46 km<sup>−2</sup> SCHOOLS 6 HEALTH FACILITIES 2 SHELTERS 1 MAJOR ROADS 57

SURFACE CHANGE 0.024472 SURFACE STABILITY 0.97553

CASE Sahel heat: response load and warming scenario

INDICES AND +1 C SCENARIO

continued

BASE HEAT RESPONSE 77.344912 BASE PRIORITY 86.374648

FUTURE MAX APPARENT 44.2 C FUTURE HOURS ≥ 40 C 25 FUTURE DEGREE-HOURS 40.4

FUTURE HEAT RESPONSE 84.444912 FUTURE PRIORITY 93.094648 DELTAS +7.1; +6.72

INTERPRETATION The explicit failed computation and repair remain in the record; the final scenario is supported by independently rechecked heat, exposure, and response terms.

CASE Partial failure: correct mechanism, wrong physical index

TASK

SCIENTIFIC TARGET Diagnose the 2018 Kilauea eruption as isolated lower-rift, isolated summit, rainfall-triggered, or coupled summit-drainage/lower-rift activity.

MECHANISM CHECK Use the evidence to distinguish the coupled explanation from all three alternatives.

OUTPUT Return one physical index and value, three evidence clues, and an operational implication.

MODEL ROUNDS 19 TOOL CALLS 9 SUCCESS 7 ERRORS 2 UNIQUE EVIDENCE FILES 4

OBSERVED RESEARCH SEQUENCE

1

list\_package\_files

Discover Enumerated all 34 package files.

2

read USGS report

Primary evidence Opened the authoritative summit-collapse and Lower East Rift Zone report.

3

search package

Weak recovery Two semantic searches returned no direct match for the requested coupling language.

4

read anchor

Event bounds Confirmed the 3 May–4 August 2018 event window and Kilauea location.

5

read guessed paths

6

Tool errors Two nonexistent guessed files produced the run’s only tool errors.

search + reread

## RETURNED ANSWER

Final extraction Recovered downrift propagation and effusion above 100 $\mathrm { m ^ { 3 } / s } ,$ then finalized without computing the 0.8/0.8 volume ratio.

DIAGNOSIS coupled summit drainage and lower-rift effusion

PHYSICAL INDEX Lava Effusion Rate VALUE 100 $\mathbf { m } ^ { 3 } / \mathbf { s }$

CLUES Downrift magma propagation after summit collapse; lava discharge above 100 $\mathrm { m ^ { 3 } / s } ;$ coordinated summit and rift monitoring.

REQUIRED ANSWER

DIAGNOSIS coupled summit drainage and LERZ effusion

PHYSICAL INDEX summit-collapse-to-lava-flow volume ratio VALUE 1.0

The index follows from approximately $\mathbf { 0 . 8 \ k m ^ { 3 } }$ of summit collapse and 0.8 km<sup>3</sup> of lava flow. A complete answer also uses partial summit drainage, near-daily collapses, and $M _ { w }$ 4.7–5.4 energy release, then rejects isolated-rift, isolated-summit, and rainfall-triggered alternatives.

ASSESSMENT

<table><tr><td>CASE Partial failure: correct mechanism, wrong physical index</td><td>continued</td></tr><tr><td>ANSWER 40.00 RUBRIC 42.50 CORE 41.25</td><td></td></tr><tr><td>PARTIAL FAILuRE The model identifies the correct coupled mechanism and gives a useful operational implication, but substitutes a real yet irrelevant rate for the required volume ratio. It also omits the summit evidence and explicit alternative-mechanism rejection. The failure is scientific and evidential, not merely a formatting breakdown.</td><td></td></tr></table>