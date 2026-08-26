# EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals

Mingxu Zhang   
The Hong Kong University of Science   
and Technology (Guangzhou)   
Guangzhou, China   
mzhang630@connect.hkust  
gz.edu.cn

Yang Ji The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China yangji.me721@gmail.com

Ying Sun   
The 63rd Research Institute, National   
University of Defense Technology Nanjing, China sunyinggilly@gmail.com   
Dazhong Shen   
Nanjing University of Aeronautics   
and Astronautics   
Nanjing, China   
shendazhong@nuaa.edu.cn Shan Huang   
The 63rd Research Institute, National   
University of Defense Technology Nanjing, China huangshan12@nudt.edu.cn

Yuhan Li The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China yuhanli530@gmail.com

Ke Zhang   
The 63rd Research Institute, National   
University of Defense Technology Nanjing, China zhangke24@nudt.edu.cn

## Abstract

Large language models (LLMs) are increasingly used as code agents for scientific and engineering analysis, but their ability to analyze raw physical-layer measurements remains untested. We introduce EMRB (Electromagnetic Reasoning Benchmark), which evaluates whether LLMs can analyze raw I/Q data by writing and running code. EMRB contains 200 problems across five dificulty levels and 27 question types, from signal detection to OFDM design, generated from 11 signal types with verified ground truth. Unlike benchmarks built on preprocessed features or structured tables, EMRB provides only the raw capture; the quantities each question refers to must first be discovered through code. We evaluate 14 LLMs spanning proprietary, open-weight, and reasoning-oriented families. Scores range from 24.1% to 78.9%, with the mean dropping from 84.9% on basic measurement to 21.2% on system design. We also propose ReconPilot, a structured method that separates signal reconnaissance, targeted analysis, and self-verification. Across three backbones, ReconPilot raises the overall score by 3.8 to 17.6 points and improves 13 of 15 backbone-level combinations tested. All data and code are publicly released in our GitHub repository.

## CCS Concepts

• Computing methodologies → Artificial intelligence; Machine learning.

## Keywords

benchmark, LLM, electromagnetic signals, signal processing

## 1 Introduction

Electromagnetic signal analysis underpins wireless communications, spectrum management, radar, and electronic warfare. Engineers routinely extract measurements from noisy I/Q captures to monitor spectrum, diagnose interference, and design systems, tasks that demand both signal-processing expertise and multi-step numerical reasoning.

LLMs ofer a natural interface for such analysis: an LLM agent can inspect an unfamiliar capture, write Python code to compute the relevant measurements, and revise its analysis over multiple turns. However, no systematic benchmark evaluates whether generalpurpose LLMs can actually analyze raw electromagnetic signals this way. Existing EM benchmarks [27, 29] rely on task-specific encoders or classify short snippets, and none require the model to discover what signals are present, implement analysis code, and answer engineering questions from the raw data.

A raw I/Q capture exposes no schema. In tabular benchmarks, variables are named by columns; in code generation, the target behavior is specified in the prompt. The variables an engineering question refers to, such as emitters, carriers, bandwidths, and interference relationships, do not exist in the input until the model computes them, and every downstream conclusion inherits the errors of this discovery step.

We introduce EMRB with 200 problems across five dificulty levels and 27 question types, generated from 11 signal types with verified ground truth. Each problem provides a raw I/Q file and requires the model to write and execute Python analysis code rather than respond from preprocessed features. We evaluate 14 LLMs and find scores ranging from 24.1% to 78.9%. Performance declines sharply as tasks require more connected measurements: isolated measurement and textbook calculation can already be delegated to frontier models, while multi-signal analysis and system design cannot. The bottleneck is not domain knowledge but the ability to keep each computed variable attached to the correct entity as the analysis chain grows. We further propose ReconPilot, a structured method that separates fixed signal reconnaissance, targeted anal ysis, and self-verification, raising the overall score by up to 17.6 points across three backbones.

Our contributions are threefold:

(1) We introduce EMRB, a 200-problem benchmark across five difficulty levels and 27 question types with raw I/Q files, verified ground truth, and a code-execution protocol, covering tasks from signal detection to OFDM design. All signals are programmatically generated with deterministic scoring, so every answer is objectively verifiable.

(2) We propose ReconPilot, a three-stage method that separates fixed signal reconnaissance, targeted LLM analysis, and selfverification. Across three backbones, ReconPilot raises the overall score by up to 17.6 points, improving 13 of 15 backbone-level combinations tested, showing that a fixed reconnaissance stage can substitute for capability the backbone lacks (§6.5).

(3) We evaluate 14 LLMs across proprietary, open-weight, and reasoning-oriented families, finding scores from 24.1% to 78.9% and a sharp decline from basic measurement (84.9%) to system design (21.2%).

## 2 Related Work

EM signal benchmarks. Existing EM and wireless datasets such as RadioML [27], WiSig [14], WASD [18], DeepMIMO [1], and DeepSense 6G [2] evaluate fixed outputs (modulation labels, device IDs, beam indices) under supervised settings, without asking an LLM to inspect raw measurements, write code, and interpret results. MERLIN’s EM-Bench [29] moves closer to language-based evaluation but still relies on a specialized signal encoder rather than code-mediated analysis, and TeleCom-Bench [32] targets textual comprehension rather than raw signal data. EMRB complements these datasets by requiring analysis workflows over raw I/Q measurements; Table 6 in Appendix L summarizes the diferences.

Code-execution and data analysis benchmarks. Code generation benchmarks (HumanEval [6], MBPP [5], APPS [15], DS-1000 [21]) and agent benchmarks (AgentBench [22], ToolBench [28], GAIA [23], MLAgentBench [17]) evaluate code synthesis and multistep tool use but are not grounded in raw physical measurements. Data-analysis benchmarks such as DA-Bench [16] and TableBench [30] are closer, but operate on structured tables with explicit schemas, whereas raw I/Q captures are complex-valued time series with no schema, so the model must construct its own variables through signal processing before answering.

Deep learning for EM signal processing. Deep learning methods for EM signals, from modulation classifiers [25] to end-toend physical layer systems [26] and joint channel estimation [35] (see [37] for a survey), achieve high accuracy but each addresses a single task and requires retraining for new signal types. More recently, LLMs have been coupled to EM systems through learned encoders [38]: RF-GPT [39] processes spectrograms via a visionlanguage model, RadioLLM [7] reprograms a frozen LLM for cognitive radio, and PReD [13] trains a multimodal foundation model for electromagnetic perception. EMRB takes a complementary perspective: it evaluates whether general-purpose LLMs can analyze raw I/Q files directly through code execution without domain-specific fine-tuning.

## 3 EMRB Benchmark Design

## 3.1 Design Principles

Figure 1 gives an overview of the benchmark. EMRB’s design follows three principles. First, the evaluation must preserve an authentic analysis workflow: load raw I/Q data, choose signal-processing methods, implement them in code, and use numerical results to answer the question. Second, the benchmark must expose diferent stages of competence from basic measurements to system-level reasoning across a broad range of EM analysis problems. Third, signals are generated programmatically with fixed random seeds and known ground truth, so every problem is reproducible and every answer is objectively verifiable.

## 3.2 Code-Execution Evaluation Protocol

EMRB provides each signal as a raw numerical data file rather than serializing I/Q samples into the prompt. Raw electromagnetic measurements are not naturally represented as text: an I/Q capture is a complex-valued array whose analysis requires numerical procedures such as FFT, PSD estimation, autocorrelation, and peak detection, which must be executed on arrays with controlled precision. EMRB therefore treats the signal file as data to be loaded and analyzed through code. The prompt contains the question and the file path; the model writes Python analysis code, observes numerical outputs, and refines its answers over multiple turns. The concrete tool-calling loop, execution limits, and scoring procedure are described in Section 4.

## 3.3 Dificulty Taxonomy

Raw EM analysis is not a single capability: a model may estimate the power of one isolated signal but fail when it must separate overlapping emitters, choose a bandwidth definition, or combine several measurements into a system-level decision. EMRB organizes its 200 problems into five dificulty levels (Table 4 in Appendix B), each containing 40 problems and targeting a progression from isolated measurements to integrated engineering judgment. Across these levels, the benchmark defines 27 non-overlapping question types so that each level introduces new analytical requirements rather than repeating the same task at higher parameter complexity. Representative problems for each level are provided in Appendix B, and Table 4 lists the question types per level.

L1 (Basic Measurement) covers the first quantities an analyst reads of a raw I/Q capture once a coarse picture of the band is already in hand: at what frequency each source sits, how strong it is, where the noise floor lies, and what modulation is being used. The prompt supplies a preliminary scan summary with source count and approximate frequencies, so L1 measures calibration precision against a known scene rather than detection from scratch; L2 and L3 open with the same summary, which is withdrawn from L4 onward (Appendix C). Each task can typically be solved with one analysis step, such as computing a PSD or locating spectral peaks. L1 establishes whether a model can turn a rough prior into numerically reliable measurements before any deeper analysis begins.

L2 (Signal Processing Methodology) asks the model to choose the right measurement procedure, not just execute one. Its five question types cover FFT/window selection, bandwidth measurement, autocorrelation, STFT analysis, and signal energy/power relationships. The dificulty lies in the fact that the same signal can require diferent analyses depending on the question: bandwidth alone may refer to a 3-dB width, a null-to-null width, or an occupied-power width, each needing a diferent implementation.

![](images/933fdade38ed2dac8f8575b18f9954877f10fba61ed9e0059c590347f71839bf.jpg)  
Figure 1: Overview of EMRB. (A) Benchmark input: raw I/Q capture paired with an engineering question (200 problems, 11 signal types, verified ground truth). (B) Five dificulty levels from basic measurement to system design. (C) Code-execution evaluation: the model writes and runs Python over up to 15 turns, scored by deterministic verifiers.

L3 (Communication Theory) moves from measurement to interpretation. The model must connect measured properties to communication-system quantities: bitrate, �<sub>�</sub>/� , PAPR, ADC quantization, and digital down-conversion. L1 and L2 characterize what is in the signal; L3 asks what it means for the communication link.

L4 (Multi-Signal Analysis) places the model in a crowded spectrum with three to five emitters. Each problem draws five sub-questions from nine question types spanning symbol-rate estimation, analog and radar parameter extraction, burst analysis, spectral-gap link budgeting, interference assessment, and Shannon capacity. Before answering any of them, the model must first build a signal inventory: how many emitters are present, where they are, and how to separate them. A missed or merged emitter propagates errors to every downstream measurement.

L5 (System Design) requires the model to turn measurements into engineering decisions. Its three question types cover spectrum survey, radar coexistence, and OFDM design. Unlike L1 to L4, L5 demands sequential reasoning across the full analysis chain: the signal inventory feeds interference assessment, which feeds system design. An error at any stage invalidates the final design.

## 3.4 Signal Library and Data Quality

EMRB uses a parameterized signal library covering 11 signal types: seven digital modulations (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK), two analog (FM, AM-DSB), and two wideband (OFDM,

LFM chirp). For each signal, carrier frequency, bandwidth, power, and SNR are sampled from level-specific ranges and the waveform is synthesized with calibrated AWGN; for L4 and L5, several emitters share one capture, producing crowded spectra whose true inventory remains exactly known. Each problem is uniquely determined by an (archetype, seed) pair: eight scenario archetypes per level fix the signal composition and question selection, and varying the seed produces a structurally identical but numerically distinct instance, so the released 200 problems (5 seeds × 8 archetypes × 5 levels) can be expanded to fresh, valid problems from unused seeds without manual authoring, each carrying automatically verified ground truth. The dataset totals 920 sub-questions; Appendix A gives pertype parameters, coverage limits, and the full generation pipeline.

Because every ground-truth answer comes from these generation parameters, EMRB verifies that each scored quantity is independently recoverable from the released waveform: reference signalprocessing scripts re-measure every answer from the saved file and agree to within 1% on all 200 problems. A separate implementationtolerance check confirms that each waveform realizes its target parameters (power ±0.5 dB, SNR ±1 dB, bandwidth ±5%), and a coverage check verifies the intended question-type and signal-type distribution. Appendix A gives the complete validation procedures.

## 4 Evaluation Protocol

## 4.1 Code-Execution Loop

EMRB evaluates models in an interactive code-execution setting that mirrors how raw I/Q measurements are analyzed in practice: no signal is rendered as text, and every reported quantity must be computed from the binary file. Each model receives the question and the raw I/Q file path with a sandboxed Python tool, and may take up to 15 interaction turns, rather than 15 code executions, before submitting a structured final answer. The budget counts turns rather than code calls because real signal analysis often requires intermediate reasoning, such as inspecting a PSD plot and deciding which sub-band to zoom into, that does not involve code. This design makes the benchmark test a complete analysis workflow rather than isolated subtasks: the model must decide what to compute, write executable signal-processing code, interpret numerical outputs, and judge when its measurements are suficient. Forcing the model to commit to a structured answer format also prevents hedging with qualitative language when a precise number is required. Implementation details, including system prompt, sandbox configuration, answer format specification, and forced-generation fallback, are provided in Appendix D; Appendix C (Figure 5) shows a complete problem prompt as delivered to a model.

## 4.2 Deterministic Scoring System

A benchmark built on raw physical measurements requires scoring that is both precise and reproducible. EMRB achieves this through level-specific deterministic verifiers in which no LLM judge contributes to the reported scores. L1 through L4 each contain five sub-questions worth 20 points; L5 contains three integrated subquestions worth 34, 33, and 33 points. Every scored requirement is checked against measured ground truth, an explicit feasibility constraint, or, for the prompts that ask for a recommendation or an explanation, a negation-aware assertion check on the committed conclusion. Criteria that credit the content of a stated rationale rather than its conclusion amount to 11 of L2’s 100 rubric points (2.2% of the benchmark total), and do not appear at any other level; Appendix F specifies their mechanism, weights, and limits.

Two design choices are critical to scoring validity. First, the verifier performs entity-bound evaluation: each extracted value is matched to its sub-question, physical unit, and signal identity be fore a quantity-specific tolerance is applied, so an unrelated number elsewhere in the response cannot receive credit. Second, integrated L4 and L5 answers such as a spectral extraction passband, a channel packing plan, or an OFDM design are scored against functional acceptance rules rather than a single reference string, permitting multiple valid designs while preserving objectivity. L5 additionally employs prerequisite-gated scoring: downstream sub-parts receive credit only when their upstream prerequisites are met (e.g., overlap analysis is gated on correct signal-pair identification), because a confident design applied to the wrong target is worse than no answer. Verifier versions and task provenance fingerprints are stored with every result, so stale scores are automatically rejected when any question, waveform, or scoring rule changes. The complete scoring rubrics are provided in Appendix F.

## 5 ReconPilot: Structured Signal Analysis

When an LLM analyzes a raw I/Q capture, it must simultaneously discover the signal scene and solve the task, and each half invites a distinct failure: before analysis begins, models often launch into undirected exploration without first understanding the scene; after it ends, they rarely check whether their own conclusions are internally consistent, e.g., a reported bitrate that contradicts the identified modulation, or a sub-question silently left unanswered.

![](images/0bdee95a45c00019e8a84d3b90b2175608c6e83ba0345b48b2f4dee9c9070eaf.jpg)  
Figure 2: ReconPilot three-stage architecture. Stage 1 turns raw I/Q into a bounded but deliberately unresolved spectral region map; Stage 2 guides targeted LLM analysis; Stage 3 verifies consistency before submission.

ReconPilot targets these two failures by wrapping LLM-guided targeted analysis with a fixed stage on each side (Figure 2): fixed reconnaissance supplies a common starting point, and self-verification catches inconsistencies before submission. The method can be applied on top of any code-capable LLM.

## 5.1 Three-Stage Pipeline Design

Stage 1: Fixed Reconnaissance. Without any prior context, a model’s first turns are typically spent on broad spectral sweeps that duplicate work across problems; providing a consistent starting picture eliminates this variance. A deterministic script, shared across all problems and backbones, estimates the noise floor via Welch PSD, thresholds at 6 dB above the floor to detect occupied regions, and reports each region’s frequency limits, width, integrated power, and peak level. Per-region STFT statistics flag time-varying, broadband, or frequency-drifting behavior. Critically, the script never counts sources, names modulations, or fits parameters: a region may contain one source, several overlapping emitters, or a swept signal, and the region count is explicitly not the source count. A guessed inventory would anchor the model to a wrong decomposition, so Stage 1 bounds the search space without resolving it (Appendix E).

Stage 2: Reconnaissance-Guided Analysis. The model receives the region map from Stage 1 as additional context alongside the question and the raw I/Q file, and analyzes them with the same unconstrained code-execution loop used in the free-form baseline. By skipping the repetitive initial spectral exploration that free-form runs spend their early turns on, the model can devote its full turn budget to task-specific measurement and reasoning. The map localizes where to look but answers nothing by itself: the LLM must still determine source counts, select the appropriate measurement procedure, and carry out task-specific calculations.

Stage 3: Self-Verification. During multi-turn code execution, models are prone to silent internal inconsistencies: a reported bitrate may contradict the identified modulation order, or a subquestion the model believed it had answered may in fact be missing. To address this, Stage 3 requires the model to review its own answers before submission without executing new code, checking sub-question coverage, unit consistency, physical plausibility, and answer format compliance, converting the completed analysis into a verified final answer.

![](images/8671524b2dae66eefe05e51d8a07414de9ca1a4ebb2d31d203de961585cb5ca9.jpg)

![](images/e6fff4713b46b263f2c797adca4e5f959be1b0c25aaab4be0386809b47c84d16.jpg)  
Figure 3: Overall EMRB results across 14 LLMs. (a) Average scores over 200 problems. (b) Code calls per problem versus accuracy. Colors and icons are shared between panels.

## 6 Experiments

We organize the evaluation around three research questions. RQ1: How much of EMRB can current LLMs solve, and is the remaining headroom achievable? RQ2: Do EMRB’s five levels test separable cognitive demands, and where do models systematically fail? RQ3: Does staged reconnaissance improve raw-signal analysis, and how does each stage contribute?

## 6.1 Experimental Setup

Models evaluated. We evaluate 14 LLMs spanning proprietary, open weight, reasoning oriented, and lightweight model families: GPT-5.5 [24], Grok-4.5 [31], Claude Opus 4.7 [3] and Claude Sonnet 4.6 [4], Gemini 3.1 Pro [10] and Gemini 3.5 Flash [11], GLM-5.2 [36], DeepSeek V4 Pro, DeepSeek V4 Flash, and DeepSeek-Chat [8], MiniMax-M3 [20], MiMo-V2.5-Pro [33, 34], Kimi-K2.6 [19], and Llama 3.3 70B [12]. This pool covers a broad range of current code capable systems rather than only a small set of frontier models. Every model receives the same problem inputs, raw file access, Python tool interface, 15 turn budget, and deterministic scoring procedure described in Section 4.1. We additionally evaluate ReconPilot with DeepSeek V4 Flash, Gemini 3.5 Flash, and GPT-5.5 as backbones to compare unconstrained tool use with an explicitly staged analysis process under a fixed backbone.

Configuration. All models are evaluated at temperature 0 with provider default reasoning settings. We use the same interaction budget for all free form runs: each model receives the problem statement, the raw I/Q file path, and up to 15 turns before submitting a final answer. ReconPilot follows the three stage procedure described in Section 5; its deterministic reconnaissance stage is shared across problems, while targeted analysis uses the same backbone model as the corresponding free form run before self-verification.

All reported scores are computed on the 200 EMRB problems using the deterministic scoring protocol in Section 4.2.

## 6.2 Overall Performance

To address RQ1, Figure 3(a) summarizes overall EMRB performance across the 14 evaluated models. Complete per-level scores for all models are listed in Appendix H. Scores range from 24.1% to 78.9%, a spread of 54.8 percentage points. GPT-5.5 and Grok-4.5 form the leading group at 78.9% and 77.8%, followed by Claude Opus 4.7 at 70.3%. GLM-5.2, Claude Sonnet 4.6, Gemini 3.1 Pro, and DeepSeek V4 Pro score between 62.6% and 65.8%, while six further models occupy the 50.6% to 57.2% range. Llama 3.3 70B scores 24.1%, showing that access to the same Python tools does not eliminate large diferences in the ability to organize raw signal analysis. No model reaches 80%, so the benchmark remains unsaturated even for the strongest current systems.

Figure 3(b) adds an eficiency dimension by plotting overall accuracy against average code calls per problem. Notably, higher performance does not simply come from executing more code. GPT-5.5 reaches 78.9% with 3.7 code calls per problem, while Grok-4.5 reaches a similar 77.8% with 12.6 calls, more than three times as many tool interactions for nearly equal accuracy. EMRB does not reward brute-force code iteration; it tests whether a model can identify the right signal-processing method for the question, implement it correctly in few attempts, and interpret the numerical output to reach a sound and well-justified engineering conclusion.

Within model families, members share training corpora, toolcalling, and alignment pipelines. The consistent gaps EMRB reveals across family members, 10.4 points across three DeepSeek variants, 8.3 between two Gemini models, and 4.7 between two Claude models, therefore isolate reasoning depth as the bottleneck. The signal-processing knowledge needed for EMRB is present even in each family’s lighter variants, but assembling that knowledge into a correct multi-step analysis plan requires the sustained inference chains that only flagship-scale architectures maintain.

![](images/e98fc4dad4f3c9b0ab8126c79af4b7ee677b67758ba07d04afbc9c43bf403f67.jpg)  
Figure 4: EMRB scores for all 27 question types across 14 models, expressed as a percentage of the points available for that type. Models are ordered by overall score; the bottom row gives the mean over models for each type.

Figure 4 disaggregates the overall scores to all 27 question types, revealing structure that level averages hide. All three L5 types score below 25 points averaged over models, confirming that the deficit is spread across the entire level rather than concentrated in one subquestion. At L4, Shannon capacity is the weakest type at 43 despite being the most standard textbook formula in the level, indicating that the bottleneck lies in extracting the correct bandwidth and noise terms from a multi-emitter scene rather than in the calculation itself. At L1, absolute power in dBm is the only weak type (67 versus 81–96 for the other four), and this weakness persists upward: the L2 energy and PSD type is also the weakest of its level at 54, suggesting that calibrated power measurement is the first capability to break and remains so across dificulty levels.

## 6.3 Headroom Validity

EMRB reports substantial headroom at every level and almost total headroom at L5, so before interpreting those numbers we must show that the missing points are reachable. We have no human expert baseline (Section 8), and we therefore establish reachability with three converging measurements, summarized in Table 1. We first replay the ground truth through the scorer itself. For each problem we render the reference values in the answer format the model is asked to produce, then score that response with the same verifiers used for every reported result. This yields 100.0 at all five levels, with 40 of 40 problems scored perfectly at each. The rubric is therefore satisfiable everywhere, and the low L5 scores are not produced by rules that no response can satisfy.

We then take the union over models. For every criterion on every problem we retain the best result achieved by any of the 14 models, which measures what the current field as a whole can do rather than what any single system does. At L1 through L4 this union covers 98.8% to 99.8% of the available points, and all 143 criteria at those levels are solved perfectly by at least one model on at least one problem. Even on these four levels the best single-model score ranges from 84.8% to 95.8%, well below the union, so the remaining gap is a diference in capability across models rather than a product of strict scoring. At L5 the same union reaches only 53.2%, and six criteria are never scored perfectly across 560 attempts, which we audit in Appendix G.

Because the union leaves half of L5 unaccounted for, we additionally wrote a reference solver for that level: roughly 1000 lines of classical DSP that read only the released waveform and the question text, with no access to generation parameters and no per-problem special casing. It scores 75.7 out of 100, close to twice the best model score of 39.8, while declining by construction to answer 18 points worth of sub-questions; on the 82 points it does attempt it earns

Table 1: EMRB headroom is reachable. Oracle replay: ground truth scored by the verifier. Union of 14: best per-criterion result across models. Reference solver: classical DSP (L5 only).
<table><tr><td>Level</td><td>Oracle replay</td><td>Union of 14</td><td>Best model</td><td>Reference solver</td></tr><tr><td>L1</td><td>100.0</td><td>99.8</td><td>95.8</td><td>一</td></tr><tr><td>L2</td><td>100.0</td><td>99.4</td><td>88.7</td><td>一</td></tr><tr><td>L3</td><td>100.0</td><td>99.6</td><td>93.2</td><td>一</td></tr><tr><td>L4</td><td>100.0</td><td>98.8</td><td>84.8</td><td>一</td></tr><tr><td>L5</td><td>100.0</td><td>53.2</td><td>39.8</td><td>75.7</td></tr></table>

92.3%. Two of the six criteria that no model ever solves are recovered by the solver at 93.1% and 70.8%, and a third is recovered by a separate blind decoder on 34 of 40 problems, so those quantities are measurable from the released data rather than defective. The L5 headroom is a limitation of current models and not of the task, and the solver’s 75.7 establishes a lower bound on what is attainable from the released data.

## 6.4 Capability Profiles Across EMRB

RQ2 asks whether EMRB’s five levels test separable cognitive demands and where models systematically fail. Across the 14 models, mean scores decline from 84.9% on L1 to 72.3% on L2, 68.7% on L3, 53.1% on L4, and 21.2% on L5. The broadly monotonic decline supports the intended dificulty progression, while the small diference between L2 and L3 shows that choosing a measurement procedure and applying communication theory impose comparable demands on current models.

Figure 6 (Appendix H) also shows that dificulty is not explained by a single weak model. On L4, models must maintain a multisignal inventory, isolate overlapping components, and bind each measurement to the correct emitter before computing interference or link quantities. The mean falls by 15.6 points from L3 to L4 as these dependencies are introduced. L5 creates a larger 31.9 point drop because spectrum survey, waveform recovery, coexistence analysis, and system design must be solved as connected parts of one problem. Even the best L5 score is only 39.8%, and the full range is 2.4% to 39.8%, indicating that system level synthesis remains dificult across model families.

Per-model profiles reveal that EMRB’s five levels test separable cognitive demands rather than a single dificulty axis. GPT-5.5 and Grok-4.5 score higher on L3 than on L2 (93.2% vs. 86.0% and 92.3% vs. 86.4%), even though L3 is the harder tier by cross-model mean (68.7% against 72.3%). This inversion exposes a gap between formula recall and engineering judgment. Communication-theory equations such as Shannon capacity and $E _ { b } / N _ { 0 }$ are densely covered in training corpora and can be applied once relevant parameters are given. L2, in contrast, requires deciding which technique to apply to the raw data, a form of meta-reasoning far less represented in text.

Gemini 3.5 Flash provides the sharpest evidence: it scores 91.9% on L1, collapses to 46.3% on L2, then recovers to 72.6% on L3, showing that a model can measure signal properties and apply textbook formulas yet fail precisely at the connective step of selecting and justifying the right methodology. L4 produces the widest model separation of any level (range 73.1 points, �=17.6), confirming that maintaining a coherent multi-signal inventory under overlapping spectra is where underlying reasoning capacity is most sharply discriminated.

Table 2: Controlled comparison between free form code execution and ReconPilot across three backbones. Bold marks the higher score in each free form/ReconPilot pair.
<table><tr><td>Backbone</td><td>Method</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td></tr><tr><td rowspan="3">DeepSeek V4 Flash</td><td>Free form</td><td>91.2</td><td>65.1</td><td>67.5</td><td>41.8</td><td>16.5</td><td>56.4</td></tr><tr><td>ReconPilot</td><td>88.2</td><td>86.3</td><td>84.9</td><td>67.8</td><td>24.8</td><td>70.4</td></tr><tr><td>Improvement</td><td>-3.0</td><td>+21.2</td><td>+17.4</td><td>+26.0</td><td>+8.3</td><td>+14.0</td></tr><tr><td rowspan="3">Gemini 3.5 Flash</td><td>Free form</td><td>91.9</td><td>46.3</td><td>72.6</td><td>47.2</td><td>28.3</td><td>57.2</td></tr><tr><td>ReconPilot</td><td>94.9</td><td>89.2</td><td>84.3</td><td>72.2</td><td>33.1</td><td>74.8</td></tr><tr><td>Improvement</td><td>+3.0</td><td>+42.9</td><td>+11.7</td><td>+25.0</td><td>+4.8</td><td>+17.6</td></tr><tr><td rowspan="3">GPT-5.5</td><td>Free form</td><td>95.8</td><td>86.0</td><td>93.2</td><td>84.8</td><td>34.5</td><td>78.9</td></tr><tr><td>ReconPilot</td><td>97.0</td><td>87.7</td><td>90.1</td><td>90.5</td><td>48.2</td><td>82.7</td></tr><tr><td>Improvement</td><td>+1.2</td><td>+1.7</td><td>-3.1</td><td>+5.7</td><td>+13.7</td><td>+3.8</td></tr></table>

The capability profiles translate into a concrete boundary for practical use. The strongest models are already dependable at the early stages of the workflow, producing calibrated measurements and textbook link calculations above 85%. The boundary sits at multi-signal analysis: once several emitters share the spectrum, most models fail to keep each measurement attached to the right emitter. No model can yet be trusted to turn a crowded-band survey into a system design, where the best score is 39.8%.

## 6.5 ReconPilot Performance

Turning to RQ3, we isolate the contribution of workflow structure by evaluating ReconPilot against free form code execution on the same 200 EMRB problems, using three backbones from the middle of the capability range observed in Figure 3(a): DeepSeek V4 Flash, Gemini 3.5 Flash, and GPT-5.5. In the free form setting, each model constructs the analysis process directly from the raw I/Q data, while ReconPilot first supplies a consistent reconnaissance context and then focuses the same backbone on task specific measurement and final verification. For each backbone, the problem set, scoring procedure, and model configuration are held fixed, so the comparison isolates the efect of organizing the analysis process from the efect of the underlying model.

Table 2 shows that ReconPilot raises the overall score on all three backbones, by 14.0 points for DeepSeek V4 Flash, 17.6 points for Gemini 3.5 Flash, and 3.8 points for GPT-5.5, and improves 13 of the 15 backbone-level combinations tested. The two exceptions are informative: DeepSeek V4 Flash loses 3.0 points at L1 and GPT-5.5 loses 3.1 points at L3, both cases where the backbone’s free form score was already above 91%. When headroom is small, the added reconnaissance context has little to fill and the reduced analysis turn budget becomes the binding constraint.

The largest gains cluster on the multi-signal levels: all three backbones gain the most or second most at L4, and Gemini 3.5 Flash’s single largest gain, 42.9 points at L2, comes at the level where its free form score of46.3% was its weakest result overall. This pattern indicates that a model benefits most from ReconPilot when a problem requires tracking several signals at once or when unguided exploration would otherwise stall, and that a fixed reconnaissance context can substitute for capability the backbone lacks. Appendix K presents two paired free form/ReconPilot case studies (Figures 9 and 10) that trace this at the level of individual sub-questions: at L3 a root symbol rate that free form guesses without measuring propagates into three of five sub-questions, and at L5 an inventory that is missing two of six emitters zeroes 43 points of otherwise correct downstream work through prerequisite gating.

Table 3: Stage 1 × Stage 3 grid on Gemini 3.5 Flash, with Stage 2 held fixed as the unmodified free form loop. Bold marks the best in each column.
<table><tr><td>Stage 1</td><td>Stage 3</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td></tr><tr><td></td><td></td><td>91.9</td><td>46.3</td><td>72.6</td><td>47.2</td><td>28.3</td><td>57.2</td></tr><tr><td>√</td><td>一</td><td>94.8</td><td>87.8</td><td>72.6</td><td>77.9</td><td>11.0</td><td>68.8</td></tr><tr><td>一</td><td>√</td><td>94.7</td><td>83.1</td><td>76.4</td><td>61.3</td><td>35.2</td><td>70.2</td></tr><tr><td>V</td><td>√</td><td>94.9</td><td>89.2</td><td>84.3</td><td>72.2</td><td>33.1</td><td>74.8</td></tr></table>

## 6.6 Stage Ablation

To understand each stage’s contribution, we enable Stage 1 and Stage 3 independently on Gemini 3.5 Flash while holding Stage 2 fixed as the free form loop. Configurations with Stage 3 reserve 3 of 15 turns for verification, so any gain is net of fewer analysis turns.

As shown in Table 3, each stage independently helps by a similar amount across all levels: reconnaissance alone lifts the average from 57.2 to 68.8 (+11.6), verification alone lifts it to 70.2 (+13.0), and both together reach 74.8 (+17.6). However, the combination is markedly sub-additive (+17.6 vs. +24.6 if the two gains were independent), indicating that the two stages repair overlapping failures rather than complementary ones.

At L4, reconnaissance alone is the best configuration (77.9), and adding verification costs 5.7 points because the verification turn reexamines a multi-signal answer that was already correct, consuming analysis turns that L4’s complex multi-emitter problems can still use productively. At L5 the pattern inverts sharply: reconnaissance without verification scores 11.0, which is 17.3 points below plain free form, because the unresolved region map invites the model to enumerate sources it cannot separate and nothing downstream checks the result; verification recovers this by catching implausible inventories before they propagate.

## 6.7 Failure Case Analysis

Returning to the second part of RQ2, the per-criterion scores reveal a progression in why models fail, not just how often. At L1, individual measurements are usually close to ground truth, but models lose credit by failing to bind each measurement to the correct signal entity. A model may report accurate frequencies and powers yet assign them to the wrong emitters, suggesting that LLMs treat spectral peaks as isolated numbers rather than maintaining a structured scene inventory. L2 shifts the bottleneck from measurement to justification: models can invoke the right spectral tool but cannot articulate the underlying tradeof (e.g., sidelobe suppression vs. main-lobe widening), indicating procedural fluency without conceptual grounding.

At L3, errors become multiplicative. A wrong modulation identification cascades through symbol rate, bit rate, and $E _ { b } / N _ { 0 }$ , yet models rarely revisit earlier steps when downstream values become physically implausible, revealing absent self-consistency checking across dependent calculations. L4 compounds this with multi-entity tracking: each measurement must stay bound to its emitter across overlapping spectra, and models that lose binding produce internally coherent but factually wrong answers.

L5 represents the full failure mode, where a missed signal in the initial catalog propagates through interference mitigation, waveform recovery, and system design, potentially zeroing an entire question block. An audit of the extraction-filter sub-question (527 of 560 attempts score zero) separates the collapse into three independent deficits: a perceptual limit where overlapping emitters are merged into one, constraint omission where models solve the unconstrained problem, and absent self-evaluation where models do not check whether their design meets stated requirements. Detailed per-level case analyses and statistics are provided in Appendix J.

Across all five levels, the pattern is consistent: EMRB failures are rarely about not knowing the right formula. They arise from the inability to maintain a coherent, entity-bound chain of evidence from raw samples to final answers.

## 7 Conclusion

We introduced EMRB, a benchmark for evaluating whether LLMs can analyze raw electromagnetic measurements through executable code. Scores across 14 LLMs range from 24.1% to 78.9% and fall from 84.9% on basic measurement to 21.2% on system design, as models struggle to maintain an entity-bound analysis state when measurements become mutually dependent. ReconPilot raises the overall score by up to 17.6 points, showing that fixed reconnaissance can substitute for missing capability. EMRB uses only synthetic signals and deterministic scoring, so it can serve as a reproducible acceptance test as models improve.

## 8 Limitations

EMRB is a controlled synthetic benchmark covering 11 signal types, so its results do not establish model performance on real field measurements with hardware impairments, multipath, or unmodeled signal types; its scope is also narrower than the name suggests, covering RF communication waveforms and LFM radar sweeps under AWGN rather than the full breadth of electromagnetic phenomena. ReconPilot depends on a fixed PSD-based reconnaissance that may miss weak or overlapping signals, and L5 scores remain low for every backbone, showing that a stable inventory alone does not resolve system design. We have no human expert baseline; reachability is established by oracle replay, a model union, and a programmatic reference solver (Section 6.3), which bounds what is attainable from below.

## 9 Ethics and Broader Impact

All captures are programmatically synthesized; the dataset contains no real radio recordings, no personal data, and no material requiring ethics review. Raw spectrum analysis has dual-use relevance to electronic warfare, but EMRB contains no parameters of any deployed system, and its signal models are textbook constructions; we judge the marginal uplift to an adversary to be negligible.

EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals

## References

[1] Ahmed Alkhateeb. 2019. DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications. arXiv preprint arXiv:1902.06435 (2019).

[2] Ahmed Alkhateeb, Gouranga Charan, Tawfik Osman, Andrew Hredzak, João Morais, Umut Demirhan, and Nikhil Srinivas. 2023. DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset. arXiv:2211.09769 [eess.SP] https://arxiv.org/abs/2211.09769

[3] Anthropic. 2026. Introducing Claude Opus 4.7. https://www.anthropic.com/ news/claude-opus-4-7. System card via the Anthropic Transparency Hub, https: //www.anthropic.com/transparency.

[4] Anthropic. 2026. Introducing Claude Sonnet 4.6. https://www.anthropic.com news/claude-sonnet-4-6. System card via the Anthropic Transparency Hub.

[5] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program Synthesis with Large Language Models. arXiv preprint arXiv:2108.07732 (2021).

[6] Mark Chen, Jerry Tworek, Heewoo Jun, et al. 2021. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374 (2021).

[7] Shuai Chen, Yong Zu, Zhixi Feng, Shuyuan Yang, and Mengchang Li. 2025. RadioLLM: Introducing Large Language Model into Cognitive Radio via Hybrid Prompt and Token Reprogrammings. arXiv preprint arXiv:2501.17888 (2025).

[9] Timnit Gebru, Jamie Morgenstern, Brenda Vecchione, et al. 2021. Datasheets for Datasets. Commun. ACM 64, 12 (2021), 86–92.

[10] Google DeepMind. 2026. Gemini 3.1 Pro Model Card. https://deepmind.google/ models/model-cards/gemini-3-1-pro/. Accessed June 2026.

[11] Google DeepMind. 2026. Gemini 3.5 Flash Model Card. https://deepmind.google models/model-cards/gemini-3-5-flash/. May 2026.

[12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783 (2024).

[13] Zehua Han, Jing Xiao, Yiqi Duan, et al. 2026. PReD: An LLM-based Foundation Multimodal Model for Electromagnetic Perception, Recognition, and Decision. arXiv preprint arXiv:2603.28183 (2026).

[14] Samer Hanna, Samurdhi Karunaratne, and Danijela Cabric. 2022. WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting. arXiv:2112.15363 [eess.SP] https://arxiv.org/abs/2112.15363

[15] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. 2021. Measuring Coding Challenge Competence With APPS. In NeurIPS Datasets and Benchmarks Track.

[16] Xueyu Hu, Ziyu Zhao, Shuang Wei, Ziwei Chai, Qianli Ma, Guoyin Wang, Xuwu Wang, Jing Su, Jingjing Xu, Ming Zhu, et al. 2024. InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks. arXiv preprint arXiv:2401.05507 (2024).

[17] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. 2024. MLAgent-Bench: Evaluating Language Agents on Machine Learning Experimentation. arXiv:2310.03302 [cs.LG] https://arxiv.org/abs/2310.03302

[18] Jinha Kim, Hyeongwoo Kim, and Byungkwan Kim. 2024. Wireless Anomaly Signal Dataset (WASD): An Open Dataset for Wireless Cellular Spectrum Monitoring and Anomaly Detection. IEEE Access (2024). doi:10.1109/ACCESS.2024.3521946

[19] Kimi Team. 2026. Kimi K2.5: Visual Agentic Intelligence. arXiv preprint arXiv:2602.02276 (2026). Technical report; Kimi-K2.6 shares the K2.5 architecture per its model card.

[20] Xunhao Lai, Weiqi Xu, Yufeng Yang, et al. 2026. MiniMax Sparse Attention. arXiv preprint arXiv:2606.13392 (2026). Designated technical report accompanying the MiniMax-M3 release.

[21] Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Scott Wen tau Yih, Daniel Fried, Sida Wang, and Tao Yu. 2022. DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation. arXiv:2211.11501 [cs.SE] https://arxiv.org/abs/2211.11501

[22] Xiao Liu, Hao Yu, Hanchen Zhang, et al. 2024. AgentBench: Evaluating LLMs as Agents. International Conference on Learning Representations (ICLR) (2024).

[23] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. GAIA: A Benchmark for General AI Assistants. In International Conference on Learning Representations (ICLR).

[24] OpenAI. 2026. GPT-5.5 System Card. https://deploymentsafety.openai.com/gpt-5-5. April 2026.

[25] Timothy J. O’Shea, Johnathan Corgan, and T. Charles Clancy. 2016. Convolutional Radio Modulation Recognition Networks. In Engineering Applications of Neural Networks (EANN).

[26] Timothy J. O’Shea and Jakob Hoydis. 2017. An Introduction to Deep Learning for the Physical Layer. IEEE Transactions on Cognitive Communications and Networking 3, 4 (2017), 563–575.

[27] Timothy J O’Shea and Nathan West. 2016. Radio Machine Learning Dataset Generation with GNU Radio. In Proceedings ofthe GNU Radio Conference.

[28] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2023. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arXiv:2307.16789 [cs.AI] https://arxiv.org/abs/2307.16789

[29] Junyu Shen, Zhendong She, Chenghanyu Zhang, Yuchuang Sun, Luqing Luo, Dingwei Tan, Zonghao Guo, Bo Guo, Zehua Han, Wupeng Xie, and Yaxin Mu. 2026. MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals. arXiv preprint arXiv:2603.08174 (2026).

[30] Xianjie Wu, Jian Yang, et al. 2025. TableBench: A Comprehensive and Complex Benchmark for Table Question Answering. In Proceedings of the AAAI Conference on Artificial Intelligence.

[31] xAI. 2026. Grok 4.5. https://docs.x.ai/developers/grok-4-5. Released July 2026.

[32] Jieting Xiao, Yun Lin, et al. 2026. TeleCom-Bench: How Far Are Large Language Models from Industrial Telecommunication Applications?. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD).

[33] Xiaomi MiMo Team. 2026. MiMo-V2-Flash Technical Report. arXiv preprint arXiv:2601.02780 (2026). Architecture report for the backbone MiMo-V2.5 inher its.

[34] Xiaomi MiMo Team. 2026. MiMo-V2.5. https://mimo.xiaomi.com/mimo-v2-5. Model page and Hugging Face model card XiaomiMiMo/MiMo-V2.5; released April 2026.

[35] Hao Ye, Geofrey Ye Li, and Biing-Hwang Juang. 2018. Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems. IEEE Wireless Communications Letters 7, 1 (2018), 114–117.

[36] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, et al. 2026. GLM-5: From Vibe Coding to Agentic Engineering. arXiv preprint arXiv:2602.15763 (2026).

[37] Chaoyun Zhang, Paul Patras, and Hamed Haddadi. 2019. Deep Learning in Mobile and Wireless Networking: A Survey. IEEE Communications Surveys & Tutorials 21, 3 (2019), 2224–2287.

[38] Hao Zhou, Chengming Hu, Ye Yuan, et al. 2025. Large Language Model (LLM) for Telecommunications: A Comprehensive Survey on Principles, Key Techniques, and Opportunities. IEEE Communications Surveys & Tutorials 27, 3 (2025), 1955– 2005.

[39] Hang Zou, Yu Tian, Bohao Wang, et al. 2026. RF-GPT: Teaching AI to See the Wireless World. arXiv preprint arXiv:2602.14833 (2026).

## Appendix

A Datasheet for EMRB 11   
B Per-Level Problem Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12   
C Complete Problem Instance . . 14   
D Evaluation Prompt Templates . . . . . . . 14   
E ReconPilot Reconnaissance Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16   
F Deterministic Scoring Criteria . . . 16   
G Scoring Reliability Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17   
H Complete Per-Model Results . . . . . . . . 17   
I Results by Question Type . . . 18   
J Per-Level Failure Case Analysis . . . . . . . . . . . 18   
K Model Response Examples . . 18   
L Benchmark Comparison . . . . . . . . 19

## A Datasheet for EMRB

Following the Datasheets for Datasets framework [9]:

Motivation. EMRB was created to evaluate LLMs’ ability to analyze raw electromagnetic signals through code execution, a capability untested by existing benchmarks. The benchmark was designed by domain experts in signal processing and wireless communications.

Composition. 200 problems organized into 5 levels × 40 problems each. Each problem consists of: (1) an English-language question with 5 sub-questions (L5: 3), (2) a complex I/Q signal file (.npy, complex64), and (3) JSON metadata with generation parameters and ground-truth answers. The dataset contains 11 modulation types (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK, FM, AM-DSB, OFDM, Chirp). SNR ranges from 5 dB to 30 dB. Signal lengths: 32,768 samples (L1–L4) and 65,536 samples (L5).

Signal type design rationale. The 11 signal types are chosen to cover the major spectral and structural families that appear in real RF environments:

• Digital modulations (7): BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK. These span phase-shift keying, quadrature amplitude modulation, and frequency-shift keying. Parameters include carrier frequency, symbol rate (�<sub>�</sub>), roll-of factor (�), power, and SNR. Low-order formats (BPSK, QPSK) test basic detection and measurement; high-order formats (64QAM) demand reasoning about noise sensitivity and spectral eficiency.

• Analog modulations (2): FM and AM-DSB. FM requires reasoning about frequency deviation and Carson bandwidth (� = 2(Δ� + �<sub>�</sub>)), while AM-DSB exposes carrier-sideband structure and modulation depth. Including analog signals prevents models from assuming all waveforms follow digital burst patterns.

• Wideband signals (2): OFDM and linear frequency-modulated (LFM) chirp. OFDM introduces subcarrier structure, cyclic prefix, and PAPR, enabling questions about multi-carrier system design. LFM chirps introduce sweep bandwidth and time-bandwidth product analysis, extending coverage to radar-type waveforms.

Generation pipeline. EMRB generates each problem through a deterministic pipeline that couples signal synthesis with question and answer construction. The core abstraction is an (archetype, seed) pair: each dificulty level defines eight scenario archetypes that fix the signal composition and question selection, and each archetype is instantiated five times with diferent random seeds, yielding 40 problems per level. Because the 27 question types are deterministic templates, an (archetype, seed) pair fully determines the I/Q waveform, the question text, and the ground-truth answers; changing only the seed produces a structurally identical but numerically distinct problem. Each instance proceeds through four stages: (1) signal types and parameters are sampled from levelspecific ranges; (2) complex baseband I/Q samples are synthesized, shifted to assigned carrier frequencies, and combined with cali brated AWGN; (3) for L4 and L5, multiple emitters are superimposed into a shared capture with independently assigned frequencies, powers, and SNRs; (4) the same parameter record instantiates the question text and computes ground-truth answers analytically or via reference signal-processing code. The final release stores each problem as a NumPy .npy file (raw I/Q) paired with a .json file (metadata, questions, and ground-truth answers). All random choices derive from the seed, so the released 200 problems are deterministically reproducible, and the seed space extends well beyond the current dataset: running the same generators with unused seeds produces fresh, valid problems on demand without any manual authoring.

Collection process. All signals are programmatically generated using the custom signal library released with the benchmark. Each generator implements standard modulation/signal models with configurable parameters. Questions are hand-crafted templates instantiated with problem-specific parameters. Ground truth is computed analytically from generation parameters.

Preprocessing/cleaning/labeling. Waveform implementation tolerance: power level ±0.5 dB, SNR ±1 dB, bandwidth ±5% of the requested value, no numerical artifacts. This bounds how closely a synthesized waveform realizes its generation target; it is not the accuracy at which answers are scored. All 200 problems pass validation.

Detailed validation procedures. EMRB applies three postgeneration quality checks to ensure that scored answers are recoverable from the released waveforms:

(1) Ground-truth verification. For each problem, analytic answers are first computed from the generation parameters. The saved .npy file is then loaded and reference signal-processing scripts run on the same I/Q array provided to models. For spectral questions, scripts estimate PSDs via Welch’s method, detect carrier peaks, and measure bandwidth using the definition stated in the question. For power and SNR questions, received power and noise are estimated directly from samples. For multisignal questions, the scripts recover the number of emitters and their carrier locations. Ground-truth recoverability. Measured quantities are compared with the stored answers; all 200 problems pass with <1% numerical error. This is the property that matters for scoring: every graded value is recoverable from the released waveform.

(2) Signal integrity checks. Each saved waveform is validated for physical consistency: correct complex array shape, finite sample values, no NaN/overflow/clipping. Waveform implementation tolerance. Received power must be within ±0.5 dB of the generation target, SNR within ±1 dB, and measured bandwidth within ±5%. These are looser than the recoverability check above because they bound synthesis fidelity rather than answer accuracy. These tolerances ensure that model failures reflect analysis errors, not corrupted inputs.

(3) Question coverage validation. The dataset distribution is verified against generated metadata. Each L1–L3 and L5 question type appears in all 40 problems of its level. L4 draws five of its nine types per problem, giving the distribution below over its 200 sub-questions.

<table><tr><td>L4 question type</td><td>Problems</td><td>Share</td></tr><tr><td>Symbol rate and modulation order</td><td>40</td><td>20.0%</td></tr><tr><td>Spectral-gap link budget</td><td>37</td><td>18.5%</td></tr><tr><td>FM parameters</td><td>30</td><td>15.0%</td></tr><tr><td>Shannon capacity</td><td>27</td><td>13.5%</td></tr><tr><td>Chirp and radar</td><td>25</td><td>12.5%</td></tr><tr><td>Interference and SIR</td><td>16</td><td>8.0%</td></tr><tr><td>Burst analysis</td><td>15</td><td>7.5%</td></tr><tr><td>OFDM parameters</td><td>5</td><td>2.5%</td></tr><tr><td>AM parameters</td><td>5</td><td>2.5%</td></tr></table>

The distribution is uneven by construction: a type is instantiated only when the sampled scene contains the signals it requires, so OFDM and AM parameters appear in the five problems whose archetype includes those emitters. Parameter ranges vary within each type, and no two problems of the same type share identical parameters. Every signal type appears at least 5 times across the dataset, and the seven most common types (BPSK, QPSK, 8PSK, 16QAM, FM, AM-DSB, Chirp) at least 45 times, preventing evaluation collapse onto a narrow set of waveform patterns.

Representativeness. The 200 captures contain 805 emitters, 2 to 7 per capture, drawn from 11 base signal types that appear as 15 waveform variants once burst-gated forms are counted. Measured over all emitters, the parameter coverage is:
<table><tr><td>Quantity</td><td>Range</td><td>Median</td></tr><tr><td>Emitters per capture</td><td>2 to 7</td><td>3</td></tr><tr><td>Occupied bandwidth</td><td>11 kHz to 5 MHz</td><td>540 kHz</td></tr><tr><td>Symbol rate (digital)</td><td>1 to 750 ksps</td><td>119 ksps</td></tr><tr><td>Received power</td><td>-43.9 to -22.4 dBm</td><td>-32.5 dBm</td></tr><tr><td>In-band SNR</td><td>18.4 to 57.9 dB</td><td>33.0 dB</td></tr></table>

These ranges are set so that every scored quantity is measurable from the capture rather than to imitate any particular deployment: the SNR floor of 18.4 dB, for instance, keeps modulation recognition and symbol-rate estimation well posed, which a field capture would not guarantee. We state the exclusions explicitly. The channel is additive white Gaussian noise only, so there is no multipath, no fading, no Doppler beyond the modeled chirp sweep, and no nonstationary interference. Receiver efects are absent: no I/Q imbalance, no phase noise, no oscillator drift, no amplifier nonlinearity, and no quantization beyond the stored complex64 format. Emitters outside the 11 modeled families, including spread-spectrum, frequency-hopping, and radar waveforms other than LFM, do not occur. The consequence is that EMRB measures analysis competence on well-conditioned captures. It bounds field dificulty from below and should not be read as a sample of any real spectral environment.

Utility. Four properties support the intended use of comparing and diagnosing models. It discriminates: reported scores span 24.1% to 78.9% with no model saturating, and the spread is not driven by one weak system. It surfaces structure that aggregate accuracy hides, including that L2 methodology questions are harder than L3 communication theory for most models, that absolute power calibration is the weakest measurement type at both L1 and L2 (Appendix I), and that L5 failures decompose into three separable deficits rather than one (Section 6.7). It is actionable: ReconPilot was designed against these diagnoses and improves three backbones by

Table 4: EMRB dificulty levels with question types. Each level tests distinct skills with zero question-type overlap across levels. Each L1–L3 and L5 problem instantiates all of its level’s question types; each L4 problem draws its five subquestions from the level’s nine types.
<table><tr><td>Lvl</td><td>Theme</td><td>Question Types</td><td>Emitters</td><td>Qs</td></tr><tr><td>L1</td><td>Basic Meas.</td><td>Detection, Power, Sam- pling, Noise Floor,</td><td>2-3</td><td>5</td></tr><tr><td></td><td>L2 Signal Proc.</td><td>Classification FFT/Window, Bandwidth, Autocorrelation, STFT, En- ergy/Power</td><td>3-4</td><td>5</td></tr><tr><td>L3</td><td>Comm. Theory</td><td>Bitrate, Eb/N0, PAPR, ADC, DDC</td><td>3-4</td><td>5</td></tr><tr><td>L4</td><td>Multi-Signal</td><td>Symbol Rate/Mod. Order, FM, AM, Chirp/Radar, Burst, Spectral Gap, In-</td><td>3-5</td><td>5</td></tr><tr><td></td><td></td><td>terference/SIR, OFDM, Capacity L5 System Design Spectrum Survey, Radar/- Coexist, OFDM Design</td><td>6-7</td><td>3</td></tr></table>

3.8 to 17.6 points (Section 6.5). It is reproducible and expandable: scoring is deterministic with no LLM judge, every result carries a scorer version and a task provenance fingerprint, and the seedbased generation pipeline can produce fresh, valid problems from unused seeds without manual authoring, providing a built-in defense against data contamination.

Uses. Primary: evaluation of LLMs on EM signal analysis. Secondary: evaluation of structured analysis pipelines for scientific data analysis; acceptance and regression testing of LLM-based tools before their integration into RF analysis workflows. Not intended for: training signal classifiers, real-world signal detection.

Distribution. The dataset, the generation scripts, the deterministic verifiers, and the evaluation harness are released at https: //github.com/mingxuZhang2/EMRB. The dataset is distributed under CC BY 4.0 and the code under the MIT License. License: CC-BY-4.0 (data), MIT (evaluation code). Format: NumPy .npy files + JSON metadata. Total size: ∼300 MB (200 signal files + metadata).

Maintenance. Maintained by the paper authors. Planned updates: additional signal types, real-world capture extension, difficulty rebalancing based on model capability frontiers. Version tracking via semantic versioning (v1.0 at release).

## B Per-Level Problem Examples

This section provides complete, representative problem instances for each dificulty level, showing the full question text, expected analysis approach, and scoring criteria.

## B.1 L1 Example: Signal Detection

L1: EMRB\_L1\_5010 | Signal Detection & Basic Measurement   
Signal: 2 signals: BPSK at � = + 2.63 MHz (� =350 ksps, �=0.35,   
−28.9 dBm) and FM at �<sub>�</sub>= − 4.38 MHz (deviation 189 kHz, −32.9 dBm).   
�<sub>�</sub>=20 MHz, �=32,768, noise floor −54 dBm/20 MHz.   
Sub-questions:   
Q1: Count the signals, estimate center frequencies, identify the   
strongest/weakest and their power diference. [Numerical]   
Q2: Estimate per-signal power (dBm), convert to mW, compute total   
power and SNR. [Numerical]   
Q3: Derive sampling parameters and FFT frequency resolution. [Nu  
merical]   
Q4: Estimate the noise floor from signal-free regions. [Numerical]   
Q5: Classify each signal’s modulation type. [Categorical]   
Ground truth: 2 signals; BPSK at +2.63 MHz (BW≈473 kHz), FM at   
−4.38 MHz (Carson BW≈591 kHz); powers −28.9 and −32.9 dBm.   
Expected approach: Compute PSD via Welch method, count peaks   
above noise floor + 10 dB, integrate band power, and classify based on   
spectral shape and envelope statistics.

## B.2 L2 Example: Signal-Processing Methodology

## L2: EMRB\_L2\_6000 | Signal-Processing Methodology

Signal: 3 signals: QPSK at +2.44 MHz (�<sub>�</sub>=250 ksps, �=0.3), FM at   
−5.36 MHz, LFM chirp sweeping +4.09 to +7.71 MHz. � =20 MHz,   
�=32,768.   
Sub-questions:   
Q1: FFT frequency resolution; rectangular vs. Hamming window   
trade-of. [Numerical + Deterministic]   
Q2: 3-dB, 99%-power, and null-to-null bandwidth of the QPSK signal.   
[Numerical, ±10%]   
Q3: FFT-based autocorrelation; spacing and source of the dominant   
periodic comb. [Numerical]   
Q4: STFT/spectrogram morphology of the three signals. [Categori  
cal]   
Q5: Per-signal energy and energy-per-bit relationships. [Numerical]   
Ground truth: Δ� =610.35 Hz, recommended window: Hamming;   
QPSK 3-dB BW ≈250 kHz, 99% BW ≈325 kHz; autocorrelation comb   
spacing 78.0 �s, attributable to the FM modulating tone.   
Key challenge: The same capture requires a diferent procedure for   
each sub-question: three bandwidth definitions need three implemen  
tations, and the autocorrelation comb must be attributed to the correct   
emitter.

## B.3 L3 Example: Communication-Theory Analysis

L3: EMRB\_L3\_4000 | From Measurement to Link Quantities   
Signal: 3 signals: QPSK at +1.31 MHz (� =500 ksps, �=0.35), FM at   
−4.52 MHz, LFM chirp at +6.30 MHz. � =20 MHz, �=32,768, noise   
floor −52 dBm/20 MHz.   
Sub-questions:   
Q1: Modulation scheme, symbol rate, bitrate, spectral eficiency.   
[Categorical + Numerical]   
Q2: In-band noise power, SNR, $E _ { b } / N _ { 0 } ,$ link margin. [Numerical]   
Q3: PAPR of each signal and PA back-of implications. [Numerical]   
Q4: Inter-signal dynamic range and minimum ADC ENOB. [Numer  
ical]   
Q5: DDC frequency shift, LPF cutof, decimation factor. [Numerical]   
Ground truth: QPSK, �<sub>�</sub>=500 ksps, ��=1000 kbps; SNR =33.4 dB,   
��/�<sub>0</sub>=31.7 dB, margin 22.1 dB; PAPR ≈5.0/0.0/0.0 dB (QP-  
SK/FM/chirp); min. ENOB 12; DDC shift −1.31 MHz, LPF cutof   
405 kHz.   
Key challenge: Each sub-question chains a code-based measurement   
with a communication-theory calculation, so a wrong modulation   
order propagates through bitrate, $E _ { b } / N _ { 0 }$ , and link margin.

## B.4 L4 Example: Multi-Signal Parameter Extraction

## L4: EMRB\_L4\_1000 | Multi-Signal Analysis

Signals: 4 signals: QPSK at +0.67 MHz (350 ksps, −34.5 dBm), 8PSK   
at +1.69 MHz (150 ksps, −32.8 dBm), 16QAM at −1.74 MHz (400 ksps,   
−34.3 dBm), LFM chirp sweeping +4.81 to +8.19 MHz (−33.7 dBm).   
�<sub>�</sub>=20 MHz, � =32,768, noise floor −45 dBm/20 MHz.   
Sub-questions:   
Q1: Symbol rates and modulation orders of the two closely spaced   
digital signals. [Numerical + Categorical]   
Q2: Chirp time-bandwidth product, processing gain, range resolu  
tion. [Numerical]   
Q3: Maximum 64QAM link deployable in the −1.46 to +0.43 MHz   
spectral gap. [Numerical]   
Q4: Most spectrally critical signal pair and their separation/overlap.   
[Deterministic]   
Q5: SNR and Shannon capacity of the 8PSK signal. [Numerical]   
Key challenge: Jointly analyzing 4 emitters. Every sub-question de  
pends on a correct signal inventory, and each measurement must stay   
bound to the right emitter.

## B.5 L5 Example: Spectrum Survey and System Design

## L5: EMRB\_L5\_2002 | Integrated Spectrum & System Analysis

Signals: 6 signals in a 20 MHz capture, including overlapping digital signals, an LFM chirp, an FM signal, and a burst. � =20 MHz, �=65,536. Sub-questions:

Q1 (34 pts): Build a complete signal inventory, identify the largest non-chirp overlap, design a verified extraction passband, and pack guarded channels into the remaining spectrum. [Deterministic]

Q2 (33 pts): Characterize the chirp, recover the overlapped communication waveform, determine its crossing interval, and recover a symbol sequence. [Deterministic]

Q3 (33 pts): Compare link capacity and spectral eficiency, optimize power allocation, and propose a feasible OFDM link. [Constraint verification]

Key challenge: Measurements from the signal inventory are reused by interference analysis, waveform recovery, and system design. Early errors propagate across the entire problem.

## C Complete Problem Instance

To illustrate the exact input models receive, the box below shows the complete prompt for an L1 problem. The prompt identifies the signal file, sampling parameters, and recording metadata, followed by five sub-questions with point allocations. The prior each level hands the model difers, and the diference is deliberate, so we state it explicitly rather than leaving it to be inferred from the example. L1, L2, and L3 open with a preliminary scan summary (all 40 problems at each of the three levels): it gives the number of independent sources in the band and, for each source, an approximate center frequency (or sweep range, for a chirp) together with a coarse family label such as digitally modulated, analog FM, or linear frequency modulated. L4 gives no band-level inventory; indi vidual sub-questions point at a spectral region (e.g., “two digitally modulated signals in the −0.3 to +2.7 MHz region”) without stating what else the capture contains. L5 gives no prior at all beyond the sampling parameters and a warning that overlapping, burst, and radar emissions may be present; the model must build the source list itself, which is what Q1(a) is scored on. We state the consequences of the L1 prior and of the hint formulas explicitly. Across the 40 L1 problems the listed frequencies agree with the ground truth to within 0.05 MHz, well inside the ±0.5 MHz that L1 Q1(a) and Q1(b) accept for full credit, so those two sub-questions, 12 of the 100 points at L1, can in principle be answered by restating the prompt rather than by measuring. A further 44 points ask for arithmetic on quantities the prompt itself supplies or for applying its labels: the Q3 sampling-parameter derivations from the stated � and � (20 points), the Q4(a)(b) noise calculations from the stated noise floor (11), the Q4(d) conceptual question (4), and the Q5(c)(d) classifica tions from the prior’s family labels and the hint’s envelope rule (9); 5 more points ask a property that is true by construction in every problem. This is deliberate rather than an oversight: these items test whether a model maintains unit discipline and dB arithmetic before any measurement begins, and weaker models fail exactly there, with Llama 3.3 70B earning 72% ofthese points and Kimi-K2.6 83%. The remaining 39 points, the per-signal powers, SNRs, and bandwidths of Q1(c)(d), Q2, and Q5(a)(b), are recoverable only from the samples, and this measurement core is where L1 discriminates: scores on it span 18.7% to 92.9% across the 14 models, a wider spread than the 52.5% to 95.8% of the aggregate. The L1 aggregate should therefore be read as guided arithmetic plus measurement accuracy against a known scene, and not as evidence of detection ability. The anchor exists because L1 is meant to isolate measurement: a model that mislocates a source would otherwise forfeit downstream measurement credit for a reason unrelated to measurement, and the level would stop being diagnostic. From L4 onward the anchor is withdrawn and source enumeration becomes a scored quantity in its own right.

## D Evaluation Prompt Templates

## D.1 System Prompt (Free-Form)

## System Prompt

You are an expert in electromagnetic signal analysis .   
You have access to a Python code execution tool .   
Your task : analyze the I/Q signal data file and   
answer the questions below . The signal is stored as   
a complex numpy array (. npy file ).   
Available libraries : numpy , scipy , matplotlib .   
Execution timeout : 30 seconds per call .   
Output limit : 10 ,000 characters per call .   
When ready , provide your final answers in this   
EXACT format :   
=== ANSWERS ===   
Q1a: [ your answer ]   
Q1b: [ your answer ]   
=== END ===   
Signal file : { signal\_path }   
Sampling rate : {fs} Hz   
Number of samples : {N}   
--- QUESTIONS ---   
{ question\_text }

## D.2 Stage 2 Input (ReconPilot)

The reported ReconPilot configuration does not give Stage 2 its own system prompt. It reuses the free-form system prompt above verbatim, so the only diference from the free-form baseline is that the region map is prepended to the question and that three of the fifteen turns are reserved for Stage 3, leaving Stage 2 twelve turns. The prepended block is:

```awk
Stage 2 Prepended Block
## Automated Reconnaissance ( localization only ; an
ROI is not a signal and ROI count is not source
count )
{ region_map }
{ question_text }
```

Holding the system prompt fixed is deliberate: it keeps the freeform and ReconPilot conditions identical except for the reconnaissance context and the reserved verification turns, so the comparison in Table 2 is not confounded by prompt engineering.

You have used all available analysis turns .   
Based on your analysis so far , provide your FINA   
answers now in the required format :   
===ANSWERS===   
Q1a : [ answer ]   
=== END ===   
If you are unsure about an answer , provide your   
best estimate with a brief note .  
Figure 5: Complete L1 problem prompt as delivered to the model, reproduced verbatim from EMRB\_L1\_5000.json with no abridgement, including the hint formulas the prompt supplies. The model receives this text together with an execute\_python tool and the raw I/Q file.

EMRB\_L1\_5000 | Complete Problem Prompt (L1: Basic Measurement)   
You are an electromagnetic signal analysis expert. Below is I/Q signal data collected from an electromagnetic environment.   
Signal file: EMRB\_L1\_5000.npy | Sampling rate: 20 MHz | Number of samples: 32768   
Data format: complex64 (numpy) | Recording duration: 1.6384 ms   
This frequency band contains 3 non-overlapping independent signal sources (types include Digital Modulation/Analog FM/Swept   
Frequency), distributed as follows based on a preliminary scan:   
• Approx. +3.5 MHz: Digital Modulation signal   
• Approx. −5.8 MHz: Analog FM signal   
• Approx. −9.0\~−5.8 MHz: Swept Frequency signal (linear frequency modulation)   
Please analyze these signals and answer the following questions (20 pts each, 100 pts total):   
Q1. Observe the spectrum of this frequency band and answer the following questions:   
(a) How many distinct independent signals are present in this band? (4 pts)   
(b) Estimate the center frequency (MHz) of each signal, listed from lowest to highest. (8 pts)   
(c) Which signal has the strongest power? Which has the weakest? (Identify by center frequency) (4 pts)   
(d) What is the power difference in dB between the strongest and weakest signals? (4 pts)   
Q2. Power measurement and dB conversion:   
(a) Estimate the power (dBm) of each signal. (6 pts)   
(b) Convert the power of each signal from dBm to mW.   
Hint: P(mW) = 10^(P(dBm)/10) (4 pts)   
(c) Calculate the total power of all signals (dBm).   
Hint: P\_total = 10·log (ΣP<sub>�</sub>(mW)) (4 pts)   
(d) Given a noise floor of −53.0 dBm/20MHz,   
calculate the SNR (dB) of each signal within its respective bandwidth.   
Hint: N\_sig = N (dBm/Hz) + 10·log (BW\_Hz) (6 pts)   
Q3. Sampling parameters and frequency resolution:   
(a) Given sampling rate fs = 20 MHz and number of samples N = 32768,   
calculate the recording duration T = N/fs (ms). (4 pts)   
(b) Calculate the sampling interval Ts = 1/fs (ns). (4 pts)   
(c) Calculate the FFT frequency resolution Δf = fs/N (Hz). (4 pts)   
(d) To halve the frequency resolution Δf, how many samples are needed?   
Should the sampling rate fs or the number of samples N be changed? Explain your reasoning. (8 pts)   
Q4. Noise floor estimation:   
(a) Observe the signal-free regions of the spectrum and estimate the noise power spectral density (dBm/Hz).   
Hint: Noise floor (dBm/20MHz) = −53.0 dBm,   
N<sub>0</sub>(dBm/Hz) = Noise floor (dBm) − 10·log<sub>10</sub>(fs) (6 pts)   
(b) Calculate the noise power within a 1 MHz bandwidth (dBm).   
Hint: P\_noise = N (dBm/Hz) + 10·log (1×10^6) (5 pts)   
(c) Determine whether this is white noise (i.e., whether the PSD is approximately uniform   
across all frequencies). (5 pts)   
(d) If all signals were removed, would the noise floor change? Why or why not? (4 pts)   
Q5. Signal feature classification:   
(a) Classify each signal by bandwidth:   
Narrowband (<100kHz), Midband (100kHz-1MHz), Wideband (>1MHz). (5 pts)   
(b) Estimate the approximate 3 dB bandwidth (kHz) of each signal. (6 pts)   
(c) Which signals are constant-envelope signals?   
Hint: FM and Chirp signals are constant-envelope; PSK modulation depends on implementation. (5 pts)   
(d) Identify the type of each signal: CW / FM / AM / Digital Modulation / Chirp. (4 pts)   
Please provide complete calculation procedures and numerical results for each question.

## D.3 Force-Answer and Self-Verification Prompts

## Stage 3: Self-Verification Prompt

Review your answers before submission . Check :   
1. All sub - questions are answered   
2. Numerical values have correct units   
3. Values are physically reasonable   
4. Format matches === ANSWERS === / === END ===   
If you find any issues , correct them .   
Output your final verified answers now.

## E ReconPilot Reconnaissance Details

The Stage 1 reconnaissance script produces an unresolved spectral region map from the raw I/Q file. Each component and every thresh old is listed below; the values quoted are the ones in the released script.

• Welch power spectral density. Two-sided density with segment length min(4096, �/4) and 50% overlap, converted to dBm/Hz. Averaging across segments suppresses sample level noise while preserving spectral structure.

• Noise-floor estimation. The median of the resulting density. All detection thresholds and reported peak levels are relative to this value, which makes the map’s decisions independent of absolute receiver gain.

• Region detection. The density is smoothed with a 21-bin moving average and thresholded at 6 dB above the noise floor; a 9-bin binary closing joins fragments of one occupancy, and connected components become candidate regions. A region is discarded if it spans fewer than three frequency bins or if its in-region peak is below 10 dB above the floor, which removes ripple and isolated spurs.

• Per-region reporting. For each surviving region the script reports the lower and upper frequency limits, the width, the power integrated over the region, and the in-region peak level. These are bounds on where energy is, not per-source measurements: a region is not assumed to hold exactly one emitter, so no center frequency, 3-dB bandwidth, or per-source SNR is claimed.

• Short-time Fourier observations. A two-sided STFT with segment length min(1024, �/8) supplies three statistics per region: the 90th minus 10th percentile spread of total region energy over time, the 95th percentile of the same spread computed within individual frequency bins, and the 5th to 95th percentile span of the dominant frequency. These expose burst behavior and frequency drift that a single averaged density would hide.

• Qualitative flags. The statistics are reduced to flags rather than parameter estimates: broad above 1 MHz of width, time-varying above 6 dB of total spread, locally-time-varying above 8 dB of within-bin spread, moving-dominant-frequency when the dominant frequency span exceeds max(0.35 MHz, 0.45 × width), and unresolved-stationary when none of these apply.

• Follow-up instruction. The map closes by telling the model to audit the source count in every region, to inspect the time frequency ridge and the residual after masking it for broad or moving regions, to compare active against inactive spectra per frequency bin for time varying regions, and not to assign source identities from the map alone.

The script does not count sources, classify modulations, estimate symbol rates, or fit chirp parameters, and it computes no time-domain amplitude statistics and no autocorrelation. Its header states explicitly that a region may contain one source, several overlapping sources, a burst, or a swept signal, and that the region count is not the source count. This is a deliberate limit rather than an omission: a fixed script cannot separate overlapping emitters reliably, and a confidently wrong inventory propagates into every later measurement, whereas an acknowledged bound does not. The output is plain text, prepended to the Stage 2 prompt (see Appendix D), and is deterministic and model independent: every backbone receives the same map for a given problem.

## F Deterministic Scoring Criteria

Numerical scoring is defined separately for each physical quantity rather than by one global error formula. Each criterion specifies accepted units, an absolute or relative tolerance, and the signal or sub-question to which the value must be bound. This distinction is necessary because a frequency error is naturally measured in hertz, a power error in decibels, and a signal inventory must match several identities rather than one scalar target.

Structured answers are verified by evaluating their consequences. For example, an L5 extraction passband must retain the required fraction of the target bandwidth and produce the reported SIR under the stated attenuation model. Channel placements are checked against band edges, occupied intervals, and guard constraints, while OFDM proposals are checked for bandwidth, power, leakage, and rate feasibility. Waveform recovery accepts the documented phase, scale, and conjugation equivalences before measuring symbol agreement. These verifiers allow multiple valid numerical designs without delegating correctness to an LLM judge.

## F.1 Categorical Answer Matching

The auto-scorer implements alias resolution for modulation type matching:

<table><tr><td>Ground Truth</td><td>Accepted Aliases</td></tr><tr><td>QPSK</td><td>4-QAM, 4QAM, Quadrature PSK, 4-PSK</td></tr><tr><td>16QAM</td><td>16-QAM, QAM-16, QAM16</td></tr><tr><td>FM</td><td>Frequency Modulation, FM modulation</td></tr><tr><td>AM-DSB</td><td>AM, DSB-AM, Double-Sideband AM</td></tr><tr><td>OFDM</td><td>Orthogonal FDM</td></tr><tr><td>Chirp</td><td>LFM, Linear FM, Swept frequency</td></tr></table>

## F.2 Decision and Rationale Criteria

Some prompts ask for a decision rather than a value: whether the sampling rate or the sample count should change (L1), which window or bandwidth definition to use (L2), or whether a link closes (L3). The verifiers score these with negation-aware assertion checks: the conclusion must be asserted, not merely mentioned, so a response stating “do not increase �” names the expected vocabulary while asserting the opposite and scores zero. Prompts that additionally say “explain your reasoning” are, with the four exceptions below, scored on the conclusion alone, and the explanation itself earns no credit. L1 Q3(d), for example, awards 4 points for the sample count that halves the resolution and 4 points for asserting that the sample count rather than the sampling rate must grow.

Four L2 criteria go further and credit the content of the stated rationale. Each decomposes the expected argument into components that can be checked independently, and a component is credited only when the response asserts it with the correct direction near the relevant concept term. The window-selection rationale, for instance, is split into three directional claims, namely that a Hamming window lowers sidelobes, widens the main lobe, and reduces spectral leakage; a claim asserted with the wrong direction, or negated, contributes nothing. The four criteria and their weights are the window-selection rationale (3 points), the bandwidth-definition distinctions (2), the allocation-planning rationale (1), and the dBmsummation explanation (5, of which one third is a numeric verification against the response’s own totals). They total 11 of the 100 L2 points and 2.2% of the benchmark; no other level credits rationale content.

We state the limitation plainly. These checks verify that the correct claims are asserted, not that they were derived, so a response could in principle earn them by stating the right facts without performing the analysis. The exposure is bounded: earning all 11 points by assertion alone raises an overall score by at most 2.2 points against an observed spread of 54.8 points. In practice the criteria are among the harder items of their level rather than free credit: the 14 models earn on average 62% of the rationale points against 74% of the remaining L2 points, and the deficit is largest for several of the strongest models (GPT-5.5 earns 50% of the rationale credit and 90% of the rest), consistent with the finding in Section 6.7 that articulating a tradeof is harder for current models than executing the measurement.

## G Scoring Reliability Analysis

All reported questions are routed to versioned deterministic verifiers. Given the same final response, these verifiers produce identical criterion and total scores without an external model call. The parsers bind quantities to sub-question labels, units, and signal identities, while structured L4 and L5 outputs are checked against explicit physical and design constraints. Every stored result also carries a scorer version and a task provenance fingerprint. The reporting scripts accept a model and level only when all 40 expected samples match the current task inventory and scorer version, preventing stale or partial results from entering a figure or aggregate score.

## G.1 Zero-Pass Criterion Audit

Section 6.3 reports that six criteria are never scored perfectly by any model across 560 attempts, all of them at L5. We audited each one to separate genuine dificulty from a defect in the task or the rubric. Two ofthem, the achieved spectral eficiency and the channel capacity in Q3(a), are answered by the reference solver at 70.8% and 93.1%, so they are measurable from the released waveform. One, the modulation inventory in Q1(a), averages a partial credit of 60% under the union and requires every emitter in a band of six or seven signals to be labeled correctly at once, which makes it hard rather than impossible. Two more, the objective value and the reported metrics of the extraction filter in Q1(c), are constraint satisfaction checks rather than comparisons against a stored answer, and the stored optimal passband itself lands within $2 \times 1 0 ^ { - 4 }$ of the declared 0.8 in-band fraction, so designs that target exactly 80% sit on the constraint boundary. The last, the 32 recovered symbols in Q2(d), was the strictest case: every one of the 560 attempts scored exactly zero. An independent decoder that reads only the waveform and the question text recovers the symbol stream perfectly on 34 of 40 problems, confirming the task is solvable, and inspection of the model responses shows that 408 of 503 parsed submissions contain only four distinct values, that is, textbook constellation points supplied in place of a failed recovery.

Table 5: Complete EMRB results for all 14 evaluated LLMs. The best score in each column is bold. Averages are computed from unrounded per-level scores, so they can difer in the last digit from the mean of the rounded values shown.
<table><tr><td>Model</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td></tr><tr><td>GPT-5.5</td><td>95.8</td><td>86.0</td><td>93.2</td><td>84.8</td><td>34.5</td><td>78.9</td></tr><tr><td>Grok-4.5</td><td>93.4</td><td>86.4</td><td>92.3</td><td>77.2</td><td>39.8</td><td>77.8</td></tr><tr><td>Claude Opus 4.7</td><td>84.0</td><td>88.7</td><td>88.3</td><td>70.0</td><td>20.5</td><td>70.3</td></tr><tr><td>GLM-5.2</td><td>92.9</td><td>83.8</td><td>73.7</td><td>58.6</td><td>20.0</td><td>65.8</td></tr><tr><td>Claude Sonnet 4.6</td><td>91.5</td><td>82.6</td><td>80.2</td><td>58.1</td><td>15.8</td><td>65.6</td></tr><tr><td>Gemini 3.1 Pro</td><td>91.7</td><td>79.6</td><td>57.8</td><td>63.4</td><td>34.8</td><td>65.5</td></tr><tr><td>DeepSeek V4 Pro</td><td>90.8</td><td>77.9</td><td>72.8</td><td>52.7</td><td>18.7</td><td>62.6</td></tr><tr><td>Gemini 3.5 Flash</td><td>91.9</td><td>46.3</td><td>72.6</td><td>47.2</td><td>28.3</td><td>57.2</td></tr><tr><td>MiniMax-M3</td><td>87.7</td><td>62.5</td><td>71.2</td><td>43.2</td><td>19.8</td><td>56.9</td></tr><tr><td>MiMo-V2.5-Pro</td><td>83.7</td><td>72.0</td><td>59.4</td><td>52.9</td><td>16.3</td><td>56.8</td></tr><tr><td>DeepSeek V4 Flash</td><td>91.2</td><td>65.1</td><td>67.5</td><td>41.8</td><td>16.5</td><td>56.4</td></tr><tr><td>DeepSeek-Chat</td><td>75.2</td><td>79.6</td><td>53.5</td><td>36.1</td><td>16.7</td><td>52.2</td></tr><tr><td>Kimi-K2.6</td><td>66.4</td><td>79.7</td><td>48.6</td><td>45.9</td><td>12.5</td><td>50.6</td></tr><tr><td>Llama 3.3 70B</td><td>52.5</td><td>22.7</td><td>31.2</td><td>11.7</td><td>2.4</td><td>24.1</td></tr></table>

![](images/e03b6c9523a01e1db4ef44d9a222a5757281a418cf6158fdd549874caac32328.jpg)  
Figure 6: Score distributions across the five EMRB levels. Each gray point represents one model, the black line is the median, and the orange diamond is the mean.

## H Complete Per-Model Results

Table 5 reports the level specific scores used in the main analysis.

## I Results by Question Type

Figure 4 disaggregates every reported score to the 27 individual question types. L1, L2, L3, and L5 instantiate all of their types in every problem, so each cell there averages 40 problems; L4 samples 5 of its 9 types per problem, so its cells average between 5 and 40 problems depending on the type. All values pass the same validation used for the main results.

Three patterns are visible that the level averages hide. First, the three weakest types in the benchmark are the three L5 types, at 15, 23, and 25 points out of 100 averaged over models, confirming that the L5 deficit is spread across spectrum survey, coexistence analysis, and design rather than concentrated in one sub-question. Second, Shannon capacity is the weakest L4 type at 43 even though it is the most standard textbook formula in the level, which locates the failure in extracting the right bandwidth and noise terms from a multi-emitter scene rather than in the calculation itself. Third, absolute power in dBm is the only weak type at L1, at 67 against 81 to 96 for the other four, and the weakest model reaches only 12; calibrated power is therefore the first quantity to break and it stays broken, since the L2 energy and PSD type is also the weakest of its level at 54.

## J Per-Level Failure Case Analysis

This section provides concrete examples of the failure patterns summarized in Section 6.7, with one representative case per dificulty level.

L1: Signal-identity binding failure. On EMRB\_L1\_5000 (3 signals), Llama 3.3 70B correctly detects three emitters but reports the chirp center frequency as −9.0 MHz instead of the ground-truth −7.41 MHz, placing it outside the ±0.5 MHz full-credit tolerance. The misplaced chirp forfeits its own slot in the frequency list and then propagates: the strongest and weakest selections, the power diference, the per-signal power and SNR estimates, and the bandwidth category all lose credit, leaving 9.3/20 on Q1 and 5.3/20 on Q2. GPT-5.5, by contrast, runs calibrated PSD code and reports −7.41 MHz, maintaining correct binding throughout (see Figure 7).

L2: Procedural fluency without conceptual grounding. On L2 bandwidth-definition problems, models frequently compute all three bandwidth measures (3-dB, null-to-null, 99%-power) correctly but lose credit on the comparison sub-question by asserting that 3- dB bandwidth is “most appropriate for spectrum allocation” without explaining why occupied bandwidth better captures out-of-band energy. The verifier scores this deterministically: committing to the occupied-power definition earns 2 points, and 1 further point requires asserting its consequence, energy capture or adjacentchannel interference, near that choice; the check verifies which claims are asserted, not the quality of the prose (Appendix F.2).

L3: Cascading chain errors. On L3 link-budget problems, a model that misidentifies QPSK as BPSK halves the estimated bit rate, which shifts the required $E _ { b } / N _ { 0 }$ by ∼3 dB, flips the link-margin sign, and produces an incorrect feasibility verdict. Each of the five chained sub-questions has its own ±3 dB tolerance, but the propagated error exceeds this at every stage, resulting in 0/100 on an otherwise well-structured analysis.

L4: Multi-entity binding collapse. On L4 multi-signal problems, models must classify whether two signals overlap or are spectrally separated before computing isolation and interference. A model that reports the correct occupied bandwidths but swaps which signal occupies which frequency range produces a wrong overlap verdict, which gates 40% of the sub-question score and invalidates all downstream interference calculations.

L5: Catalog-gated scoring cascade. On L5 system-design problems, Q1 requires a complete signal inventory matched via a costmatrix assignment to the reference catalog. A model that misses one of six signals (e.g., a weak burst emitter 8 dB below the strongest signal) receives zero on that catalog entry, which then zeros the prerequisite-gated sub-questions in Q2 (interference analysis, 33 pts) or Q3 (system design, 33 pts) that depend on the missed signal’s identity.

L5 extraction-filter deficit breakdown. An audit ofthe extractionfilter sub-question, on which 527 of 560 attempts score zero, accounts for every attempt. (1) Perceptual limit produces all 527 zeros: 35 submissions contain no schema-valid answer, and 492 name a wrong extraction target. In 427 of the 492 the true target never appears in the submitted inventory at all, and 390 of these report two reference emitters as a single merged source, at a median center separation of 0.303 MHz against bandwidths of 0.44 and 0.52 MHz, with the target 1.3 to 10.6 dB weaker; in the other 65 the target is inventoried but another emitter is selected. The remaining 33 attempts name the target correctly: 14 fail the Q1b pair prerequisite, so their designs are never evaluated, and 19 designs reach evaluation. (2) Constraint omission covers 15 of the 19 evaluated designs: 13 keep the passband on the target but retain only 0.47 to 0.77 of its bandwidth against the required 0.8, 2 place the passband so that almost none of the target falls inside it, and 3 of the 15 reach a post-filter SIR of 30 to 34 dB, better than the constrained optimum, because they solved the unconstrained problem instead. (3) Absent self-evaluation covers all 19: every design reports a retained fraction more than 0.03 away from what its own passband delivers, and each of the 15 infeasible designs claims a fraction at or above the 0.8 floor while delivering between 0.00 and 0.77, applying the claimed stopband attenuation to the whole interferer rather than only to the part falling outside the passband. The 4 remaining designs are feasible, 3 of them within 4 dB of the optimal SIR, yet no attempt out of 560 earns full credit, because the reported metrics never match the measured behavior of the submitted design. The second and third deficits are linked, since performing the third check would expose the second, which is why we count self-verification separately from optimization ability.

## K Model Response Examples

This section presents actual model responses to illustrate how analysis quality difers across capability levels. We compare GPT-5.5 and Llama 3.3 70B on the same L1 problem (Figure 7), followed by a GPT-5.5 excerpt on an L5 system-design problem (Figure 8). We then present three paired case studies, one per ReconPilot backbone, contrasting a free-form response with the ReconPilot response on the identical problem to show concretely how the structured pipeline converts a partially-scored free-form attempt into a substantially higher-scoring one.

![](images/e40d8f84a6bd4f875b3bf879d44a78f8ee468b2ca368cce7e3a3537200799653.jpg)  
EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals

![](images/8188126203a26b03c0646ea1c4788e5ef102d39bc93ae2cf54eb93bbc17bc123.jpg)  
Verdict: Full credit on Q1 and Q2 (20/20 each). All frequencies within ±0.5 MHz, powers within 3 dB, SNR values within 3 dB of ground truth.

![](images/48da59d9aa485625643869bcccf1115c8fa33c66fbafaac7dc4d7c1e007257a6.jpg)  
Verdict: Q1: 9.3/20, Q2: 5.3/20. Chirp center frequency (−9.0 MHz) outside ±0.5 MHz tolerance of ground truth (−7.41 MHz). Power values are round-number estimates without code-based calibration, deviating >10 dB from measured ground truth.  
Figure 7: L1 response comparison on EMRB\_L1\_5000 (Q1–Q2 excerpts). GPT-5.5 executes multiple rounds of calibrated signal processing code; Llama 3.3 70B produces superficially plausible answers from the prompt hint without code-verified measurements.  
Figure 8: L5 system-design response excerpt. EMRB L5 requires chained reasoning where the signal inventory feeds interference analysis, which feeds filter and system design. The verifier checks each step against ground truth and physical constraints.

## L Benchmark Comparison

Table 6 compares EMRB with representative EM/wireless and dataanalysis benchmarks along three dimensions: input format, task description, and output format. Existing EM datasets provide presegmented signal snippets or encoded representations and evaluate fixed classification outputs (modulation labels, device IDs, beam indices). Data-analysis benchmarks operate on structured tables with explicit schemas. EMRB is the only benchmark that delivers raw, full-length I/Q recordings and requires models to produce freetext engineering answers through self-directed code execution.

Gemini 3.5 Flash (Free form) Score: 47.0/100 | 2 code calls   
25.9 s   
EMRB\_L3\_4037 (BPSK link at −5.8 MHz, reference � = 350 ksps, occupied   
bandwidth 438 kHz)   
Process (2 of 15 available turns used): call 1 prints only Data shape: (32768,),   
dtype: complex64; call 2 raises NameError: name ’freqs’ is not defined. Neither   
call returns a measurement, and the model then emits an empty turn.   
Q1a: “BPSK, �<sub>�</sub> = 1000 ksps” Q1c: “� = 0.8 bps/Hz”   
Q5b: “750 kHz” Q5c: “Decimation factor = 10, New sampling rate = 2000 kHz”   
Q1: 10.0/20 Q2: 13.5/20 Q5: 4.0/20

Verdict: Because neither call measures anything, � = 1000 ksps is a guess rather than an estimate, 2.86× the reference. It propagates into the bit rate, the 64QAM rate, Q2’s noise, SNR, ��/� and margin, and all four Q5 quantities, accounting for 32.5 of the 53 points lost. The run was not budget limited; it stopped after two of fifteen turns. One downstream item survives: spectral eficiency scores full marks because the error cancels in ��/BW.

Gemini 3.5 Flash (ReconPilot) Score: 85.9/100 13 code calls   
65.1 s   
EMRB\_L3\_4037   
Process (call 4/13): reasons from the injected region map, “ROI R1 is −6.055 to   
−5.576 MHz, width = 478.5 kHz . . . if �<sub>�</sub> = 300 ksps and � = 0.5, the bandwidth   
is 450 kHz”, and recovers from six interpreter errors instead of stopping.   
Q1a: “BPSK, 300 ksps” Q4a: “Strongest: −28.1 dBm, Weakest: −33.7 dBm”   
Q5b: “270 kHz” Q5c: “Decimation factor = 37, New sampling rate = 540.5 kHz”   
Q1: 20.0/20 Q2: 20.0/20 Q5: 20.0/20

Verdict: At 300 ksps the root quantity is 14% low, inside the ±15% window, and the same value carries Q2’s noise, SNR, ��/� and margin chain and all of Q5 to full credit. Two attributions separate cleanly. The Q4 received powers are copied verbatim from the region map’s integrated-power lines, and the variant of this configuration with the map removed reports −20 and −40 dBm instead; the symbol-rate chain, by contrast, is also recovered without the map, so persistence of the measurement loop rather than the map is what fixes it. Stage 3 altered no numeric field in this run. Q3 is 0.6 points below the free form run, from a luckier unmeasured PAPR guess there.

Figure 9: Gemini 3.5 Flash on EMRB\_L3\_4037. Free form execution guesses a root symbol rate after two code calls that returned no measurement, and the 2.9× error propagates through Q1’s rate quantities, Q2’s noise and margin chain, and all of Q5. ReconPilot derives the root quantity from the reconnaissance region map and reaches full credit on Q1, Q2 and Q5.

GPT-5.5 (Free form) Score: 23.4/100 3 code calls 79.6 s   
EMRB\_L5\_2005 (six true sources: 8PSK and 16QAM overlapping near −3.1 MHz, an   
LFM chirp sweeping 5.26 to 8.43 MHz, a QPSK link at +7.84 MHz beneath that   
sweep, an FM carrier, and a BPSK burst)   
Q1a inventory (4 of 6): “Brief spectrum analysis found four primary independent   
sources”. The QPSK under the chirp and the 8PSK inside the −3.1 MHz blend are   
both absent, and the FM carrier is labeled QPSK.   
Q1b: “pair S1/S2, overlap = 0.0 MHz, SIR = 0.0 dB” (a non-overlapping pair)   
Q2b: “victim\_id: S1” (the FM at −5.6 MHz, outside the sweep)   
Q1: 10.6/34 Q2: 7.0/33 Q3: 5.8/33

Verdict: This is a gating cascade, not weak analysis. The stored prerequisites record q2\_victim\_identified: false and q3\_complete\_digital\_link\_set: false, so Q2b, Q2c, Q2d, Q3b and Q3c collapse to zero, 43 points in total. The chirp measurement the run did base on a correctly detected source scored a full 7/7. Its own turn-3 output even printed the peak at 7.842 MHz without ever resolving it as a source.

GPT-5.5 (ReconPilot) Score: 65.7/100 | 9 code calls 397.1 s   
EMRB\_L5\_2005   
Q1a inventory (6 of 6): “I audited the ROIs with Welch PSD plus time-frequency   
ridge/residual inspection . . . after masking its ridge, a continuous QPSK-like   
link remains near 7.8 MHz”.   
Stage 3: consistency\_issues\_before: [“Q1b pair is not the largest non-chirp   
overlap in Q1a”] → after: []   
Q1: 18.6/34 Q2: 22.5/33 Q3: 24.6/33

Verdict: The region map lists the same four regions the free form run found unaided, so it supplies no new detection. What it supplies is an instruction: audit the source count in every ROI, andfor a broad or moving ROI inspect the residual after masking the ridge. Following it recovered the QPSK during Stage 2, which cleared both prerequisites and turned the same downstream items into 10.5/12, 4/4, 6/8 and 7/7. The sixth source appeared only in Stage 3, when the deterministic dependency audit reported the illegal Q1b pair and the repair call measured the −34.99 dBm excess that became it.

Figure 10: GPT-5.5 on EMRB\_L5\_2005. Under prerequisite-gated L5 scoring, a four-source inventory zeroes 43 points of otherwise competent downstream work. ReconPilot recovers the fifth source by following the region map’s audit instruction and the sixth through Stage 3’s dependency check, which is the division of labor the stage ablation in Section 6.6 predicts.

Table 6: Comparison with representative EM/wireless and data-analysis benchmarks.
<table><tr><td>Benchmark</td><td>Input Format</td><td>Task Description</td><td>Output Format</td></tr><tr><td>EMRB</td><td>Raw full-length I/Q files</td><td>Open-ended EM signal analysis</td><td>Text answers</td></tr><tr><td>EM-Bench</td><td>Encoded signal-text pairs</td><td>Signal question answering</td><td>Text answers</td></tr><tr><td>RadioML</td><td>Short synthetic I/Q snippets</td><td>Modulation classification</td><td>Modulation labels</td></tr><tr><td>WiSig</td><td>WiFi I/Q packet captures</td><td>RF fingerprinting</td><td>Device IDs</td></tr><tr><td>WASD</td><td>Wireless measurements or spectrograms</td><td>Spectrum anomaly detection</td><td>Anomaly classes or regions</td></tr><tr><td>DeepSense 6G</td><td>Multimodal wireless data</td><td>Beam prediction and sensing tasks</td><td>Beams or locations</td></tr><tr><td>DA-Bench</td><td>Structured tables with questions</td><td>Tabular data analysis</td><td>Text or numeric answers</td></tr></table>