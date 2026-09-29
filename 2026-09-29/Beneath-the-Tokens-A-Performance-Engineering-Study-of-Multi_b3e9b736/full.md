# Beneath the Tokens: A Performance Engineering Study of Multi-Token Prediction in GPU-Accelerated LLM Inference

Suwesh Prasad Sah

suwesh.github.io

September 2026

## Abstract

Autoregressive large language model inference repeatedly invokes the target model to generate one token at a time, making generation sensitive to GPU memory movement and sequential execution. This study evaluates two-token multi-token prediction (MTP) against autoregressive decoding in a controlled single-request deployment on an NVIDIA A10G GPU. A 360-request benchmark covered plain-text, reasoning-intensive, and tool-calling workloads, while runtime telemetry, Nsight Systems, PyTorch Profiler, and selected Nsight Compute measurements were used to explain the observed performance.

MTP increased output throughput by 1.91× to 2.19× across all prompts and reduced time to first output by 10.0–14.2%. Median mean acceptance length ranged from 2.370 to 2.595 tokens per verification iteration. Profiling showed that MTP introduced a longer and more complex execution path, including proposal, sampling, attention, gathering, and reduction operations. However, it required 56.4–78.1% fewer executions of the selected repeating CUDA Graph per generated token. The dominant MTP GEMM kernel was not faster than the dominant autoregressive GEMV kernel, and selected instances of both approached the A10G memory-bandwidth limit. These results show that MTP improved inference through amortization: greater token progress reduced repeated GPU execution suficiently to outweigh the additional speculative-execution cost.

Keywords: Speculative Decoding, LLM Inference, GPU Performance Analysis, GPU Kernels, CUDA Graphs, NVIDIA Nsight, PyTorch Profiler   
Artifacts and Supporting Evidence:   
Code and documentation | Complete benchmark and profiling artifact archive   
Notice: This manuscript is a technical report and has not undergone peer review.

## 1 Introduction

Large language model (LLM) inference commonly generates output through autoregressive decoding, where each new token is mathematically conditioned on the entire preceding sequence. In practice, this sequential dependency forces the inference engine into a loop of single-token forward passes, which are severely bottlenecked by GPU memory bandwidth. Speculative decoding mitigates this hardware bottleneck by verifying multiple candidate tokens in parallel during a single target model execution step, potentially advancing the generation prefix by multiple tokens at once. Multi-token prediction (MTP) provides model-native predictions of multiple future tokens that can be used to support speculative inference [1]. However, its practical benefit cannot be determined from the decoding algorithm alone. Additional speculative work is introduced, and the resulting wall-clock performance depends on whether greater token progress ofsets the cost of that work.

This study evaluates two-token MTP in a controlled, GPU-accelerated LLM inference deployment. Autoregressive and MTP decoding are compared across plain-text, reasoning-intensive, and tool-calling workloads using a frozen experimental protocol. The evaluation combines clean application-level measurements with runtime acceptance metrics and matched profiler captures.

This separation preserves the benchmark as the source of quantitative performance results while using profiling evidence only to investigate the execution behavior associated with those results.

The investigation focuses on three measurable aspects of speculative execution. First, MTP runtime metrics characterize the amount of useful progress obtained from speculative proposals. Second, matched Nsight Systems reports are used to compare repeated GPU execution cadence and kernel composition between autoregressive and MTP decoding. Third, matched PyTorch Profiler traces provide higher-level operational context for diferences observed in the GPU timelines. Together, these evidence sources examine how speculative progress is translated into application-level latency and throughput without assuming that the observed acceleration results from individual kernels executing faster.

Accordingly, this paper addresses four research questions:

• RQ1: How does two-token MTP afect time to first output, end-to-end latency, output throughput, and semantic-phase-associated duration across plain-text, reasoning-intensive, and tool-calling workloads?

• RQ2: How does observed speculative acceptance behavior correspond to the realized performance diferences across workloads?

• RQ3: How does MTP change repeated GPU execution cadence and kernel composition relative to autoregressive decoding for comparable generated outputs?

• RQ4: Which observable runtime and GPU execution diferences help explain the gap between speculative token progress and realized wall-clock acceleration?

The study contributes a controlled workload-level comparison of autoregressive and MTP decoding, a theory-guided evaluation of the relationship between speculative progress and realized performance, and a cross-layer analysis connecting application-level measurements with runtime and GPU execution evidence.

## 2 Speculative Decoding

Speculative decoding was introduced as a method for accelerating autoregressive generation while preserving the output distribution of the target model [3]. The method uses two models: a target model $M _ { p }$ , whose output distribution must be preserved, and a less expensive draft model $M _ { q }$ , which proposes candidate tokens. Rather than invoking the target model once for every generated token, the draft model first generates a sequence of $\gamma$ candidate tokens. The target model then evaluates the corresponding candidate prefixes together and determines how many proposed tokens can be accepted.

Let $p _ { i } ( x )$ and $q _ { i } ( x )$ denote the token distributions produced by the target and draft models, respectively, at speculative position i, after applying the configured sampling policy. A proposed token $x _ { i } \sim q _ { i }$ is accepted with probability

$$
a _ { i } ( x _ { i } ) = \operatorname* { m i n } \left( 1 , { \frac { p _ { i } ( x _ { i } ) } { q _ { i } ( x _ { i } ) } } \right) .\tag{1}
$$

where $a _ { i } ( x _ { i } )$ denotes the acceptance probability of the proposed token. If the target model assigns the proposed token at least as much probability as the draft model, then $p _ { i } ( x _ { i } ) \geq q _ { i } ( x _ { i } )$ and the proposal is always accepted. Otherwise, the proposal is accepted with probability $p _ { i } ( x _ { i } ) / q _ { i } ( x _ { i } )$

The target distributions for the proposed positions are evaluated in parallel, but the resulting proposals are accepted in prefix order. If proposal $x _ { i }$ is rejected, all later proposals from the

same speculative sequence are discarded because their conditioning prefix is no longer valid. A replacement token is then sampled from the corrected distribution

$$
p _ { i } ^ { \prime } ( x ) = \mathrm { n o r m } \left( \operatorname* { m a x } \left( 0 , p _ { i } ( x ) - q _ { i } ( x ) \right) \right) .\tag{2}
$$

If all $\gamma$ draft tokens are accepted, one additional token is sampled from the target model. This acceptance and correction procedure ensures that the resulting sequence follows the target model’s output distribution, even though some tokens were initially proposed by the draft model [3].

Consequently, one speculative step emits between one and $\gamma + 1$ tokens. For example, if six tokens are proposed and all six are accepted, the step may emit the six accepted proposals and one additional target-model token. If only the first proposal is accepted, the step emits that accepted proposal followed by a target-derived replacement token. In the worst case, if no proposal is accepted, the step still emits one target-derived token. Speculative decoding can therefore reduce the number of sequential target-model steps when the draft model proposes tokens that are frequently accepted.

Let α denote the mean probability that a draft token is accepted. Under the simplifying assumption that draft-token acceptance events are independent and identically distributed, the expected number of tokens emitted by one speculative step is

$$
\mathbb { E } [ N ] = 1 + \alpha + \alpha ^ { 2 } + \cdots + \alpha ^ { \gamma } = \frac { 1 - \alpha ^ { \gamma + 1 } } { 1 - \alpha } .\tag{3}
$$

The leading term represents the token supplied by the target model, while each subsequent term represents the probability of accepting a progressively longer prefix of draft tokens. In the configuration evaluated in this study, $\gamma = 2$ , giving

$$
\mathbb { E } [ N ] = 1 + \alpha + \alpha ^ { 2 } , \qquad 1 \leq \mathbb { E } [ N ] \leq 3 .\tag{4}
$$

Higher acceptance can therefore reduce the number of sequential verification steps required to produce an output sequence. This reduction does not guarantee an equal wall-clock speedup, because generating the proposals and verifying multiple positions introduce additional work. Let c denote the cost of one draft-model step relative to one target-model step. Under the assumptions of the original analytical model, the expected wall-clock improvement is

$$
S _ { \mathrm { t h e o r y } } = { \frac { 1 - \alpha ^ { \gamma + 1 } } { ( 1 - \alpha ) ( 1 + \gamma c ) } } = { \frac { \mathbb { E } [ N ] } { 1 + \gamma c } } .\tag{5}
$$

Equation 5 expresses the central trade-of: speculative decoding is beneficial when the additional token progress obtained through accepted proposals outweighs the cost of producing those proposals. The coeficient c is not solely a property of the models; it also depends on the hardware and software implementation. The analytical model therefore provides a prediction about the balance between acceptance and cost, rather than guaranteeing a fixed speedup for every deployment. The original work likewise conditions its wall-time analysis on assumptions about draft cost and available computational concurrency [3].

The present study evaluates this trade-of for a two-token MTP configuration. In this deployment, a dedicated MTP drafter model supplies the speculative candidates, while the primary autoregressive target model performs verification and acceptance within the inference runtime. Runtime acceptance metrics characterize the useful progress obtained from speculation, and matched profiler captures examine the repeated GPU execution associated with obtaining that progress. The analysis therefore investigates whether the observed wall-clock acceleration is consistent with the theoretical balance between increased token progress and additional speculative-execution cost.

## 3 Experimental Methodology

The study compares autoregressive decoding with two-token MTP speculative decoding under a controlled, single-request inference configuration. The experiment separates clean applicationlevel benchmarking from profiler-instrumented executions. Benchmark runs provide the quantitative performance results, while runtime logs, Nsight Systems reports, and PyTorch Profiler traces provide supporting evidence about speculative acceptance and GPU execution behavior. All conditions were governed by a frozen experimental protocol whose integrity was verified using a SHA-256 checksum.

## 3.1 System Under Test

The experiments were conducted on an AWS EC2 instance equipped with one NVIDIA A10G GPU with 24 GB of device memory [4]. The host ran Red Hat Enterprise Linux 9.8 with Linux kernel 5.14.0-687.45.1.el9\_8.x86\_64. The captured environment reported NVIDIA driver version 595.71.05 and CUDA version 13.2. Inference was served through vLLM 0.24.0 [2] using its OpenAI-compatible streaming API. The experimental client and orchestration tools were implemented in Python 3.12.

The target model was the quantization-aware-trained Gemma 4 E4B instruction-tuned checkpoint google/gemma-4-E4B-it-qat-w4a16-ct [5]. The checkpoint stores four-bit weights with sixteenbit activations in the compressed-tensors format and is intended for optimized inference with vLLM. The model has approximately 4.5 billion efective parameters, 42 decoder layers, and a vocabulary of approximately 262,000 tokens. The same quantized target checkpoint was used in both experimental conditions: it generated tokens directly in the autoregressive condition and supplied the verification distributions in the MTP condition.

The dedicated MTP drafter was google/gemma-4-E4B-it-assistant [5]. This assistant checkpoint predicts speculative candidate tokens for subsequent verification by the target model. In the MTP condition, vLLM was configured to use the assistant checkpoint as a dedicated MTP speculator with two speculative tokens per step. The assistant was therefore an explicitly deployed auxiliary model rather than an implicit or internally derived prediction head.

The two decoding conditions were therefore:

• Autoregressive: the quantized target model generated the output without a speculative model.

• MTP: the same target model verified token proposals produced by the dedicated MTP drafter model, with a speculative depth of γ = 2.

Apart from the speculative configuration, the conditions used common serving parameters. Both services used a maximum model length of 8,192 tokens, a maximum sequence count of one, a maximum of 2,048 batched tokens, BF16 KV-cache storage, chunked prefill, prefix caching, the same generation configuration, and the same chat template. Tool selection and the Gemma 4 reasoning and tool-call parsers were enabled in both conditions. GPU memory utilization was limited to 0.80 for the controlled benchmark services.

Each condition was exposed through a separate systemd service. Only the service corresponding to the active decoding mode was permitted to run during clean benchmark collection. The benchmark wrapper verified the expected service state and rejected execution when an NVIDIA profiling process was detected.

## 3.2 Workloads

The benchmark used three workload classes representing diferent output structures: plain-text generation, reasoning-intensive generation, and tool calling. Each class contained three fixed prompts, producing nine prompt conditions per decoding mode.

Plain-text generation. The plain-text workloads requested short, constrained natural-language outputs. Output validity required exactly eight non-empty lines, with each line containing between six and twelve words. These prompts provided comparatively open-ended generation while preserving a machine-checkable structural requirement.

Reasoning-intensive generation. The reasoning workloads required the model to derive a deterministic numerical result before producing a constrained final answer. The streaming response exposed reasoning content separately from the final answer, allowing the client to identify the transition between the reasoning and final-output phases. Output validity required the expected final-answer string to appear in the final response.

Tool calling. The tool-calling workloads supplied a fixed tool schema and required exactly one structured tool invocation. Output validity required the selected tool name to match the expected tool and the reconstructed JSON arguments to equal the expected argument object. Parallel tool calls were disabled.

The same prompt content, system instruction, chat template, request configuration, and validation rule were used for the normal and MTP conditions. Each request also contained a paired experiment identifier, allowing equivalent normal and MTP runs to be linked during analysis.

## 3.3 Experimental Procedure

The primary experiment followed Protocol v1.2 (Appendix A), which was frozen before measurement and paired with a SHA-256 integrity record. The benchmark matrix comprised

$$
{ \mathrm { 2 ~ m o d e \times 3 ~ w o r k l o a d \times 3 ~ p r o m p t s \times 2 0 ~ r e p e t i t i o n s } } = 3 6 0 ~ m e a s u r e d ~ r e q u e s t s .\tag{6}
$$

This produced 180 requests per decoding mode and 20 matched repetitions per prompt. Before each mode-workload group, three unmeasured warm-up requests were issued and stored separately. The autoregressive experiment was completed and validated before the MTP service was started.

Requests were submitted sequentially through the local vLLM streaming API using the temperature, top-p, seed, chat template, and 1,024-token output limit fixed by the protocol. Each request preserved its raw stream, reconstructed output, token usage, timestamps, validity result, and derived summary. The completed experiment contained 360 unique run records. The protocol, scripts, results, logs, and profiling artifacts were preserved in a checksum-verified archive.

## 3.4 Measurements and Validity

The benchmark client used a monotonic high-resolution clock to record six possible timestamps:

• request start;

• first meaningful output;

• first reasoning content;

• first final-answer content;

• first tool-call fragment; and

• request end.

A meaningful output event contained reasoning content, final-answer content, or a tool-call fragment. Metadata-only streaming events were not treated as output.

The resulting measurements were time to first output (TTFO), end-to-end latency, time to first final output, reasoning-associated duration, final- or tool-associated duration, completion length, and output throughput. The phase durations are described as “associated” because they follow client-observed stream boundaries and do not attribute every operation in an interval exclusively to one semantic phase.

Output throughput was calculated as

$$
\mathrm { O u t p u t ~ t h r o u g h p u t } = \frac { N _ { \mathrm { c o m p l e t i o n } } - 1 } { T _ { \mathrm { e n d } } - T _ { \mathrm { f i r s t ~ o u t p u t } } } ,\tag{7}
$$

where subtracting the first completion token aligns the token count with the measured post-first output interval.

Each request was evaluated for execution validity, semantic-boundary availability, and workloadspecific output correctness. All 360 requests were execution-valid and phase-mapping-valid; 349 also passed output validation. The remaining eleven reasoning requests, consisting of six exactanswer mismatches and five length-limited completions, were retained rather than replaced.

Comparisons are performed first between matched prompts. Metrics that describe successful execution may use all execution-valid requests, whereas analyses requiring a correct, completed trajectory use only output-valid requests. End-to-end latency is interpreted alongside completion length and output throughput to account for response-length variation.

## 3.5 Profiling Methodology

Profiling was conducted separately from the clean benchmark, and profiled request timings were not used as benchmark results. Matched captures were collected for plain\_001, reasoning\_001, and tool\_001 under both autoregressive and MTP decoding.

## 3.5.1 Nsight Systems

Six Nsight Systems reports captured CUDA, NVTX, and operating-system runtime activity. Each capture used a temporary vLLM instance, a 30-second settling interval, three unmeasured workload-matched warm-ups, and one target request.

Client NVTX markers identified request start, first output, first reasoning content, first final answer or tool call, and request end. These markers isolate the target request from model initialization and warm-up activity. The matched traces are used to compare repeated GPU execution cadence, inter-group gaps, kernel composition, invocation counts, and aggregate kernel duration. Six corresponding SQLite exports support reproducible numerical analysis.

## 3.5.2 PyTorch Profiler

Six matched PyTorch Profiler captures were collected for the same conditions after three unprofiled warm-ups. Each condition preserved a worker trace, an AsyncLLM frontend trace, and a profiler summary. The worker traces provide higher-level operational context for repeated model execution and for matrix, attention, and speculative-path operations where the recorded labels permit unambiguous interpretation.

All twelve target requests used for Nsight Systems and PyTorch Profiler analysis passed phasemapping and output validation.

<table><tr><td>Workload</td><td>Mode</td><td>TTFO (ms)</td><td>E2E (ms)</td><td>TPS</td><td>Tokens</td></tr><tr><td>Plain text</td><td>AR</td><td>77.92</td><td>5107.05</td><td>97.46</td><td>491.0</td></tr><tr><td rowspan="2">Reasoning-intensive</td><td>MTP</td><td>69.40</td><td>2676.02</td><td>192.80</td><td>499.5</td></tr><tr><td>AR</td><td>80.19</td><td>8373.85</td><td>96.61</td><td>802.0</td></tr><tr><td rowspan="2">Tool calling</td><td>MTP</td><td>69.22</td><td>4074.32</td><td>204.70</td><td>811.0</td></tr><tr><td>AR</td><td>79.76</td><td>3369.30</td><td>97.26</td><td>321.0</td></tr><tr><td></td><td>MTP</td><td>70.06</td><td>1630.76</td><td>204.98</td><td>321.0</td></tr></table>

Table 1: Workload-level medians across benchmark-valid requests. E2E denotes end-to-end latency and TPS denotes output tokens per second. Each row contains 60 requests.

## 3.5.3 Relationship to the Analytical Cost Model

The theoretical coeficient c in Equation 5 requires isolated draft and target step costs. Because the collected traces do not provide unambiguous boundaries for measuring these costs independently, this study does not estimate c directly.

Where requested execution groups can be identified consistently, the analysis instead considers the implementation-level ratio

$$
r = { \frac { T _ { \mathrm { M T P \ g r o u p } } } { T _ { \mathrm { A R \ g r o u p } } } } ,\tag{8}
$$

which represents the relative duration of the complete implemented execution groups. The empirical quantity r is not treated as equivalent to the theoretical draft-step coeficient c.

Nsight Compute is used only if the Nsight Systems and PyTorch Profiler analyses identify a specific kernel requiring hardware-level characterization.

## 4 Results

## 4.1 Dataset Validity

The primary experiment collected all 360 planned requests, comprising 180 autoregressive and 180 MTP executions. Each decoding mode contained 20 repetitions of nine prompts distributed equally across the three workload classes. All run identifiers were unique, all rows used Protocol v1.2, and no timing or token metric was missing.

All 360 requests passed benchmark and phase-mapping validation. Of these, 349 also passed workload-specific output validation: 177 of 180 autoregressive requests and 172 of 180 MTP requests. The eleven retained output-invalid requests were confined to the reasoning-intensive workloads. Six reasoning\_001 requests completed normally but did not contain the exact expected final-answer string, while five reasoning\_002 requests reached the 1,024-token output limit and lacked the required final answer. These requests were preserved rather than replaced.

Application-level timing and throughput results below use all 360 benchmark-valid requests because every request completed under the controlled deployment and retained valid timing boundaries. Results that require a completed semantic trajectory use the 349 output-valid requests.

## 4.2 RQ1: Application-Level Performance

Table 1 reports workload-level medians across the 60 benchmark-valid requests in each workload and decoding mode. MTP improved sustained generation performance in all three workloads. Median output throughput increased from 97.46 to 192.80 tokens/s for plain text, from 96.61 to 204.70 tokens/s for reasoning-intensive requests, and from 97.26 to 204.98 tokens/s for tool calling.

<table><tr><td>Prompt</td><td>TTFO reduction</td><td>E2E speedup</td><td>TPS speedup</td></tr><tr><td>plain_001</td><td>11.3%</td><td>2.27×</td><td>1.91×</td></tr><tr><td>plain_002</td><td>11.6%</td><td>1.91×</td><td>1.95×</td></tr><tr><td> $\mathtt { p l a i n \_ 0 0 3 }$ </td><td>10.0%</td><td>2.04×</td><td>2.05×</td></tr><tr><td>reasoning_001</td><td>12.9%</td><td>2.17×</td><td>2.19×</td></tr><tr><td>reasoning_002</td><td>12.6%</td><td>1.94×</td><td>2.11×</td></tr><tr><td>reasoning_003</td><td>14.2%</td><td>2.04×</td><td>2.06×</td></tr><tr><td>tool_001</td><td>12.2%</td><td>2.08×</td><td>2.12×</td></tr><tr><td>tool_002</td><td>11.7%</td><td>2.04×</td><td>2.09×</td></tr><tr><td>tool_003</td><td>12.8%</td><td>2.07×</td><td>2.11×</td></tr></table>

Table 2: Prompt-matched efects calculated from the median of 20 repetitions per mode. E2E speedup is $\widetilde { T } _ { \mathrm { A R } } / \widetilde { T } _ { \mathrm { M T P } }$ , and TPS speedup is $\widetilde { R } _ { M T P } / \widetilde { R } _ { A R }$

Prompt-matched efects are reported in Table 2. Throughput improved for every prompt, with speedups ranging from $1 . 9 1 \times \mathrm { ~ t o ~ } 2 . 1 9 \times$ . The median prompt-level throughput speedup was 1.95× for plain text, 2.11× for reasoning-intensive, and 2.11× for tool calling. TTFO improved modestly, with reductions of 10.0–11.6% for plain text, 12.6–14.2% for reasoning-intensive, and 11.7–12.8% for tool calling.

The tool-calling workload provided the most direct latency comparison because paired modes produced identical median completion lengths for all three prompts. MTP reduced prompt-level median E2E latency by factors of 2.04× to 2.08×, while increasing output throughput by 2.09× to 2.12×. The narrow range across the three prompts indicates that the efect was consistent within this structured workload.

Reasoning-intensive requests also showed substantial acceleration. Across its three prompts, median E2E speedups ranged from 1.94× to 2.17×, while throughput speedups ranged from $2 . 0 6 \times \mathrm { t o } 2 . 1 9 \times$ . The output-valid sensitivity analysis produced workload-level median completion lengths of 811 tokens under both modes, with median E2E latency decreasing from 8467.60 ms under autoregressive decoding to 4074.32 ms under MTP. Thus, the reasoning result was retained when analysis was restricted to correct and completed outputs.

Plain-text E2E latency requires greater caution because generated lengths varied between modes and repetitions despite identical prompts and fixed decoding settings. For example, plain\_001 had median completion lengths of 486 tokens under autoregressive decoding and 390.5 tokens under MTP, contributing to its 2.27× E2E diference. Nevertheless, output throughput increased for every plain-text prompt, including plain\_002, where the median completion lengths were similar at 439 and 446 tokens. The plain-text throughput result therefore indicates a generationrate improvement even where E2E latency was afected by trajectory length. Semantic-phaseassociated durations followed the same direction. Across the nine prompts, reasoning-associated duration improved by factors ranging from 1.85× to 2.33×, while final- or tool-associated duration improved by 2.26× to 2.68×. These intervals are interpreted as client-observed phase-associated durations rather than exclusive measurements of internal reasoning or tool-generation operations.

Answer to RQ1. Two-token MTP improved every measured prompt on the reported performance metrics. Its primary efect was sustained generation acceleration: output throughput increased by approximately $1 . 9 1 \times { \mathrm { ~ t o ~ } } 2 . 1 9 \times$ , and prompt-level median E2E latency improved by 1.91× to 2.27×. TTFO improved consistently but by a smaller 10.0–14.2%, showing that the larger benefit occurred after output generation began. Tool calling produced the most stable comparisons, reasoning retained approximately twofold acceleration under an output-valid sensitivity analysis, and plain-text E2E results required completion-length-aware interpretation.

<table><tr><td>Workload</td><td>Windows</td><td>Mean acceptance length</td><td>Avg. Draft acceptance rate</td><td>Position 1 , 2 acceptance rates</td></tr><tr><td>Plain text</td><td>17</td><td>2.370</td><td>68.7%</td><td>74.3% , 62.8%</td></tr><tr><td>Reasoning intensive</td><td>24</td><td>2.575</td><td>78.9%</td><td>84.5% , 73.2%</td></tr><tr><td>Tool calling</td><td>12</td><td>2.595</td><td>79.8%</td><td>83.3% , 75.9%</td></tr></table>

Table 3: Median speculative-decoding telemetry across clean workload-associated logging windows. Mean acceptance length includes the target-generated token and therefore ranges from 1 to 3 tokens for the 2-token speculative configuration.

## 4.3 RQ2: Acceptance and Theoretical Correspondence

The MTP service emitted speculative-decoding telemetry at approximately 10-second intervals. Each record reported mean acceptance length, accepted and drafted token counts, per-position acceptance rates, and average draft acceptance rate. Because several sequential requests could contribute to one logging interval, these records are treated as workload-associated aggregate windows rather than per-request measurements.

The telemetry windows were aligned with the request sequence recorded by the benchmark wrapper. Windows containing only warm-up requests or crossing workload transitions were excluded. Owing to diferences in end-to-end workload execution time, this produced 17 clean windows for plain text, 24 for reasoning-intensive requests, and 12 for tool calling. Table 3 reports the median values within each workload.

Acceptance decreased with speculative position in every workload. Plain-text windows recorded median acceptance rates of 74.3% for the first draft position and 62.8% for the second. The corresponding rates were 84.5% and 73.2% for reasoning-intensive requests, and 83.3% and 75.9% for tool calling. The second speculative candidate was therefore less likely to be accepted than the first, although it still contributed useful progress in all three workloads.

The observed mean acceptance lengths quantify the average token progress obtained per target verification iteration. Autoregressive decoding advances by one token per iteration, whereas the MTP configuration could advance by as many as three tokens: one target-produced token and up to two accepted draft tokens. Median mean acceptance lengths of 2.370, 2.575, and 2.595 therefore indicate substantial additional progress for plain text, reasoning-intensive, and tool calling requests, respectively.

The application-level throughput gains reported in Section 4.2 were smaller than these acceptance lengths. Median prompt-level throughput speedups were 1.95×, 2.11×, and 2.11× for plain text, reasoning-intensive requests, and tool calling, respectively. Thus, greater accepted progress generally corresponded to greater throughput, but accepted tokens did not translate one-for-one into wall-clock speedup.

This diference is consistent with the analytical model: acceptance increases useful progress per verification iteration, while the dedicated drafter, expanded target verification, acceptance processing, and associated runtime operations add execution cost. Because the runtime telemetry does not isolate draft-step and target-step costs, the theoretical cost coeficient c cannot be estimated directly from these records. Consequently, the present comparison evaluates the directional correspondence between acceptance and realized speedup rather than claiming an exact prediction from acceptance length alone.

Answer to RQ2. Two-token MTP provided substantial useful speculative progress in all three workloads, with median mean acceptance lengths ranging from 2.370 to 2.595 tokens per verification iteration. Reasoning-intensive and tool-calling workloads achieved higher draft acceptance than plain text and also produced slightly larger median prompt-level throughput speedups. However, realized speedup remained below mean accepted token progress, demonstrating that acceptance is necessary but insuficient to predict wall-clock acceleration without accounting for the cost of the implemented speculative path.

## 4.4 RQ3: GPU Execution Analysis

To examine the GPU-side mechanism underlying the application-level performance diferences, we analyzed one matched autoregressive (AR) and multi-token prediction (MTP) Nsight Systems capture for each workload: plain text, reasoning-intensive generation, and tool calling. The analysis used the target profiling request (r100) from each condition. Kernel and CUDA Graph records were extracted programmatically from the corresponding Nsight Systems SQLite exports.

The analysis window began at FIRST\_OUTPUT and ended at RUN\_END. This boundary excluded prompt processing, prefill, and first-output production while retaining both reasoning-associated generation and final-answer or tool-call-associated generation. In all captures, FIRST\_REASONING occurred within approximately 0.008–0.011 ms of FIRST\_OUTPUT; therefore, the selected window included efectively the complete post-first-output reasoning phase. The FIRST\_FINAL and FIRST\_TOOL\_CALL markers were used only as semantic and diagnostic boundaries and did not truncate the GPU analysis window.

Measurement definitions. A kernel launch denotes one individually recorded invocation in CUPTI\_ACTIVITY\_KIND\_KERNEL. Kernel-launch counts therefore assign equal count weight to operations of very diferent duration, from microsecond-scale elementwise kernels to millisecond-scale matrix kernels. A selected repeating CUDA Graph denotes the trace-local repeating graph used to delimit recurrent execution: Graph 520/GraphExec 521 for AR and Graph 313/GraphExec 314 for MTP. We define a recurring execution group as the start-to-start interval between consecutive executions of the selected repeating CUDA Graph:

$$
T _ { \mathrm { g r o u p } , i } = t _ { \mathrm { g r a p h } , i + 1 } - t _ { \mathrm { g r a p h } , i }\tag{9}
$$

Graph identifiers are local to the respective traces and do not carry meaning outside the captured executions.

For each profiler request, kernel-launch count, executions of the selected repeating CUDA Graph, and summed individual-kernel duration were normalized by the exact number of completion tokens reported by that request:

$$
K _ { \mathrm { t o k e n } } = \frac { N _ { \mathrm { k e r n e l } } } { N _ { \mathrm { c o m p l e t i o n } } } ,\tag{10}
$$

$$
G _ { \mathrm { t o k e n } } = \frac { N _ { \mathrm { g r a p h } } } { N _ { \mathrm { c o m p l e t i o n } } } ,\tag{11}
$$

and

$$
T _ { \mathrm { k e r n e l / t o k e n } } = \frac { \sum _ { j = 1 } ^ { N _ { \mathrm { k e r n e l } } } \left( t _ { \mathrm { e n d } , j } - t _ { \mathrm { s t a r t } , j } \right) } { N _ { \mathrm { c o m p l e t i o n } } }\tag{12}
$$

Here, $K _ { \mathrm { t o k e n } }$ is the number of individually recorded kernel launches per completion token, and $G _ { \mathrm { t o k e n } }$ is the number of executions of the selected repeating CUDA Graph per completion token. The latter is distinct from the number of complete start-to-start recurring execution groups: N graph executions provide N − 1 measurable cadence intervals. Finally, $T _ { \mathrm { k e r n e l / t o k e n } }$ is the summed duration of individually recorded kernel intervals per completion token. It is neither GPU wallclock time nor end-to-end request latency. CUDA Graph intervals were analyzed separately and were not added to their constituent kernel durations, since doing so would double-count overlapping execution.

<table><tr><td>Workload</td><td>Mode</td><td>Completion tokens</td><td>Kernel launches per token</td><td>Individual-kernel ms per token</td><td>Repeating CUDA Graph executions per token</td><td>Median cadence (ms)</td></tr><tr><td>Plain text</td><td>AR</td><td>532</td><td>26.476</td><td>2.232</td><td>0.778</td><td>10.363</td></tr><tr><td>Plain text</td><td>MTP</td><td>531</td><td>23.567</td><td>0.961</td><td>0.309</td><td>12.108</td></tr><tr><td>Reasoning-intensive</td><td>AR</td><td>590</td><td>23.354</td><td>1.969</td><td>0.686</td><td>10.366</td></tr><tr><td>Reasoning-intensive</td><td>MTP</td><td>588</td><td>22.854</td><td>0.945</td><td>0.299</td><td>12.412</td></tr><tr><td>Tool calling</td><td>AR</td><td>321</td><td>22.271</td><td>1.881</td><td>0.654</td><td>10.383</td></tr><tr><td>Tool calling</td><td>MTP</td><td>321</td><td>11.087</td><td>0.460</td><td>0.143</td><td>12.399</td></tr></table>

Table 4: Post-first-output GPU execution measurements normalized by the exact completiontoken count of each matched profiler request. Median cadence denotes the median start-to-start interval between consecutive executions of the selected repeating CUDA Graph. Kernel duration denotes summed individual-kernel duration and must not be interpreted as GPU wall-clock or end-to-end latency.

Recurring execution structure. AR exhibited one dominant repeating graph topology across all three workloads. Graph 520/GraphExec 521 was repeatedly executed, followed by individually visible supporting operations and a dominant gemv2T\_kernel\_val invocation before the next graph replay. In contrast, MTP exhibited a more complex recurring structure. Two sequences of six short graph executions were followed by one execution of the larger Graph 313/GraphExec 314. The repeated MTP graph ordering was:

$$
\begin{array} { l } { 2 9 8 \to 3 0 1 \to 3 0 4 \to 3 0 7 \to 3 1 0 \to 1 , } \\ { 2 9 8 \to 3 0 1 \to 3 0 4 \to 3 0 7 \to 3 1 0 \to 1 , } \\ { 3 1 3 . } \end{array}\tag{13}
$$

The two short sequences are structurally consistent with the configured two-token speculative depth. However, the SQLite records establish only their execution ordering, not their semantic operator role. They are therefore described as auxiliary MTP graph sequences, rather than being assigned confirmed drafter-pass semantics.

Figure 1 illustrates this structural diference using the matched tool-calling captures. The figure is intentionally shown at a scale that preserves consecutive executions of the selected repeating CUDA Graph and one complete recurring execution group. Consequently, the short auxiliary MTP operations appear visually compressed. Their identities, counts, and durations were obtained from the SQLite records rather than estimated from the displayed widths.

Repeated-execution cadence. Table 4 reports the completion-token-normalized execution measurements. The median AR recurring-group cadence was highly stable across workloads, ranging from 10.363 to 10.383 ms. The corresponding MTP cadence ranged from 12.108 to 12.412 ms. We define the workload-specific cadence ratio as:

$$
r _ { \mathrm { c a d e n c e } } = \frac { \widetilde { T } _ { \mathrm { g r o u p , M T P } } } { \widetilde { T } _ { \mathrm { g r o u p , A R } } } ,\tag{14}
$$

where $\tilde { T }$ denotes the median interval between consecutive executions of the selected repeating CUDA Graph.

The resulting r<sub>cadence</sub> values were 1.168 for plain text, 1.197 for reasoning-intensive generation, and 1.194 for tool calling. Thus, the median start-to-start cadence of an observed MTP recurring execution group was 16.8–19.7% longer than that of the corresponding AR group. This longer cadence is consistent with the additional auxiliary graph sequences and supporting attention, gather, reduction, indexing, and rejection/sampling operations present in the MTP path.

<table><tr><td>Workload</td><td>Launches per token</td><td>Kernel ms per token</td><td>Repeating CUDA Graph executions per token</td><td>Γcadence</td></tr><tr><td>Plain text</td><td>11.0%</td><td>57.0%</td><td>60.3%</td><td>1.168</td></tr><tr><td>Reasoning-intensive</td><td>2.1%</td><td>52.0%</td><td>56.4%</td><td>1.197</td></tr><tr><td>Tool calling</td><td>50.2%</td><td>75.5%</td><td>78.1%</td><td>1.194</td></tr></table>

Table 5: Relative change from AR to MTP in the matched profiler captures. Positive values in the first three columns denote reductions under MTP.
<table><tr><td rowspan="2">Workload</td><td rowspan="2">Mode and dominant kernel</td><td rowspan="2">Instances</td><td rowspan="2">Total duration (ms)</td><td rowspan="2">Duration share</td><td rowspan="2">Median invocation (µs)</td></tr><tr><td></td></tr><tr><td>Plain text</td><td>AR: gemv2T_kernel_val</td><td>415</td><td>1152.511</td><td>97.07%</td><td>2777.066</td></tr><tr><td>Plain text</td><td>MTP: ampere_bf16_s16816gemm...</td><td>164</td><td>456.571</td><td>89.52%</td><td>2784.122</td></tr><tr><td>Reasoning-intensive</td><td>AR: gemv2T_kernel_val</td><td>406</td><td>1127.524</td><td>97.06%</td><td>2777.003</td></tr><tr><td>Reasoning-intensive</td><td>MTP: ampere_bf16_s16816gemm...</td><td>177</td><td>492.757</td><td>88.69%</td><td>2784.137</td></tr><tr><td>Tool calling</td><td>AR: gemv2T_kernel_val</td><td>211</td><td>585.987</td><td>97.06% 88.57%</td><td>2776.906</td></tr><tr><td>Tool calling</td><td>MTP: ampere_bf16_s16816gemm...</td><td>47</td><td>130.847</td><td></td><td>2784.042</td></tr></table>

Table 6: Dominant matrix-kernel composition in the matched post-first-output windows. The shortened kernel names identify the exact families reported by Nsight Systems; complete demangled names are retained in the accompanying artifacts.

Although the median MTP recurring-group cadence was longer, MTP required substantially fewer executions of the selected repeating CUDA Graph relative to generated output. Executions of the selected repeating CUDA Graph per token fell by 60.3% for plain text, 56.4% for reasoningintensive generation, and 78.1% for tool calling. The reduction in graph-execution frequency was substantially larger than the increase in recurring-group cadence. Consequently, summed individual-kernel duration per token fell by 57.0%, 52.0%, and 75.5%, respectively.

The smaller change in kernel-launch density, especially for reasoning-intensive generation, does not contradict the reduction in repeating CUDA Graph executions. A kernel launch is one operation, whereas a recurring execution group contains many operations. In the reasoningintensive capture, AR recorded 13,779 kernel launches and 405 executions of Graph 520, yielding a capture-level ratio of approximately 34.0 kernel launches per selected graph execution. MTP recorded 13,438 kernel launches and 176 executions of Graph 313, yielding approximately 76.4 kernel launches per selected graph execution. The corresponding capture-level ratio was therefore approximately 2.25 times higher under MTP, consistent with the additional speculative-support path. Consequently, executions of the selected repeating CUDA Graph per token fell by 56.4%, while total kernel launches per token fell by only 2.1%. Most of the additional MTP operations were short relative to the dominant matrix kernel.

Kernel composition. The change in kernel composition was consistent across all workloads. AR execution was dominated by gemv2T\_kernel\_val, which accounted for approximately 97.06% of summed individual-kernel duration in every AR capture. MTP execution was instead dominated by an ampere\_bf16\_s16816gemm kernel, which accounted for 88.57–89.52% of summed individual-kernel duration.

The median dominant-kernel duration changed very little. AR GEMV invocations required approximately 2.777 ms, whereas MTP BF16 GEMM invocations required approximately 2.784 ms. The MTP dominant kernel was therefore approximately 0.25% longer per invocation, rather than faster. The reduction instead arose from invocation frequency. Dominant matrix-kernel invocations fell from 415 GEMVs to 164 GEMMs for plain text, from 406 to 177 for reasoningintensive generation, and from 211 to 47 for tool calling. These correspond to reductions of 60.5%,

<table><tr><td rowspan="2">Workload</td><td colspan="2">Unified attention</td><td colspan="2">Vectorized gather</td><td colspan="2">Reduce segments</td><td colspan="2">Rejection/sampling</td></tr><tr><td>Launches</td><td>ms</td><td>Launches</td><td>ms</td><td>Launches</td><td>ms</td><td>Launches</td><td>ms</td></tr><tr><td>Plain text</td><td>1320</td><td>25.190</td><td>493</td><td>1.093</td><td>660</td><td>1.800</td><td>165</td><td>0.353</td></tr><tr><td>Reasoning-intensive</td><td>1416</td><td>32.404</td><td>530</td><td>1.170</td><td>708</td><td>1.989</td><td>177</td><td>0.392</td></tr><tr><td>Tool calling</td><td>376</td><td>8.823</td><td>140</td><td>0.309</td><td>188</td><td>0.518</td><td>47</td><td>0.102</td></tr></table>

Table 7: MTP-specific auxiliary kernel activity in the post-first-output windows. No matching instances were observed in the corresponding AR windows.

56.4%, and 77.7%, respectively. The dominant-kernel invocation reductions closely matched the workload-specific reductions in executions of the selected repeating CUDA Graph per token.

MTP also introduced operation families not observed in the matched AR windows. These included kernel\_unified\_attention, vectorized\_gather\_kernel, reduce\_segments, and a kernel matched by both rejection and sampling search patterns. The rejection and sampling matches referred to the same underlying kernel instances and were therefore counted once rather than added as separate operation sets. Table 7 reports the observed launch counts and durations.

The auxiliary launches explain why kernel-launch count alone understates the execution change. For example, reasoning-intensive MTP reduced executions of the selected repeating CUDA Graph per token by 56.4%, yet kernel launches per token fell by only 2.1%. The MTP capture contained 229 fewer dominant matrix-kernel invocations than the matched AR capture, decreasing from 406 AR GEMV invocations to 177 MTP GEMM invocations, but introduced 1,416 unified-attention launches, 530 vectorized-gather launches, 708 segmented-reduction launches, and 177 overlapping rejection/sampling launches. These support operations were individually short: unified attention contributed 32.404 ms, vectorized gather 1.170 ms, segmented reduction 1.989 ms, and rejection/sampling 0.392 ms across the entire reasoning-intensive MTP window. The expensive matrix-kernel reduction therefore dominated the added auxiliary execution, reducing summed individual-kernel duration per token by 52.0%.

The same relationship appeared in the other workloads. For plain text, the MTP capture contained 251 fewer dominant matrix-kernel invocations than the matched AR capture, and the summed duration of the dominant matrix kernel was approximately 60.4% lower. After accounting for the added MTP support kernels, total individual-kernel duration per token remained 57.0% lower. For tool calling, the MTP capture contained 164 fewer dominant matrix-kernel invocations than the matched AR capture, and the summed duration of the dominant matrix kernel was 77.7% lower. Total individual-kernel duration per token was 75.5% lower.

Answer to RQ3. Across the three matched workloads, MTP changed repeated GPU execution in two related ways. First, compared with AR’s frequently repeated, GEMV-dominated execution structure, MTP exhibited a more complex recurring path containing two auxiliary CUDA Graph sequences, Graph 313/GraphExec 314, BF16 GEMM, unified attention, vectorized gather, segmented reduction, and rejection/sampling activity. This additional work was accompanied by a 16.8–19.7% longer cadence for the observed MTP recurring execution groups.

Second, MTP substantially reduced how frequently the selected repeating CUDA Graph and dominant matrix operation were executed relative to generated output. Repeating CUDA Graph executions per token fell by 56.4–78.1%, and dominant matrix-kernel invocations fell by 56.4– 77.7%. Consequently, summed individual-kernel duration per token fell by 52.0–75.5%, despite the longer median recurring-group cadence under MTP.

The observed application-level MTP benefit therefore did not correspond to a faster dominant matrix kernel or to the uniform elimination of small kernel launches. Instead, MTP added many short support operations but amortized this additional work by requiring far fewer executions of the selected repeating CUDA Graph and dominant matrix-kernel invocations per generated token. The mechanism was consistent across workloads, while its magnitude remained workload dependent. Tool calling exhibited the largest reduction, plain text was intermediate, and reasoning-intensive generation exhibited the smallest reduction.

Interpretation boundaries. These profiler measurements characterize the execution mechanism rather than replacing the clean application-level latency results. A kernel launch is not a logical decoding step, and a recurring execution group is not a direct count of generated or accepted tokens. The count of selected repeating CUDA Graph executions therefore must not be used to infer speculative acceptance length. Similarly, the two auxiliary graph sequences are structurally consistent with two-token speculative execution but are not assigned confirmed drafter semantics from the Nsight Systems SQLite evidence alone.

Finally, summed individual-kernel duration excludes CPU-side processing, runtime scheduling gaps, synchronization outside the recorded kernel intervals, token streaming, response parsing, and post-GPU client activity. It must therefore not be reported as end-to-end speedup or GPU wall-clock time. The representative timeline in Figure 1 illustrates the observed structure, while all reported counts, durations, normalized values, and cadence statistics originate from the programmatic SQLite analysis.

## 4.5 RQ4: Runtime Operations and Implementation Costs

To explain why speculative token progress did not translate proportionally into wall-clock acceleration, we analyzed the matched PyTorch Profiler worker traces for plain text, reasoning-intensive generation, and tool calling. Unlike the Nsight Systems captures used for RQ3, these worker traces did not contain the client-side NVTX markers. The RQ4 analysis therefore does not reconstruct the client-visible output phases. Instead, it isolated the repeated steady-state generation contexts recorded in the worker trace and examined their CPU-side runtime hierarchy and associated GPU-side execution spans.

Measurement scope and context identification. The dominant steady-state generation annotations were execute\_context\_0(0)\_generation\_1(1) for AR and execute\_context\_ 0(0)\_generation\_1(3) for MTP. Each logical generation context was recorded twice: once as a CPU-side user\_annotation and once as a GPU-side gpu\_user\_annotation. The two records were paired using their shared External id. All selected contexts formed exact one-to-one pairs across all workloads: 532 AR and 221 MTP pairs for plain text, 590 AR and 224 MTP pairs for reasoning-intensive generation, and 321 AR and 124 MTP pairs for tool calling.

The CPU-side annotation denotes the inclusive host-side execution range of the recorded generation context. The GPU-side annotation denotes the associated GPU execution span. The latter is not interpreted as summed kernel duration or GPU utilization because it can include multiple kernels, memory operations, and intervals between device activities. One-of context variants labelled with generation\_0(0) were excluded from the steady-state context distributions because they did not belong to the repeatedly observed generation\_1(...) population.

For mode $m ,$ observed completion-token progress per selected steady-state generation context was calculated as:

$$
P _ { m } = { \frac { N _ { \mathrm { c o m p l e t i o n } , m } } { N _ { \mathrm { c o n t e x t } , m } } } .\tag{15}
$$

The selected GPU-context span per completion token was calculated as:

$$
C _ { \mathrm { G P U / t o k e n } , m } = \frac { \sum _ { i = 1 } ^ { N _ { \mathrm { c o n t e x t } , m } } T _ { \mathrm { G P U ~ c o n t e x t } , i } } { N _ { \mathrm { c o m p l e t i o n } , m } } .\tag{16}
$$

<table><tr><td rowspan="2">Workload</td><td rowspan="2">Mode</td><td rowspan="2">Completion tokens</td><td rowspan="2">Contexts</td><td rowspan="2">Tokens per context</td><td rowspan="2">Median CPU span (ms)</td><td rowspan="2">Median GPU span (ms)</td><td rowspan="2">GPU span per token (ms)</td><td rowspan="2"> $S _ { \mathbf { c o n t e x t } }$ </td></tr><tr><td></td></tr><tr><td>Plain text</td><td>AR</td><td>532</td><td>532</td><td>1.000</td><td>5.423</td><td>10.453</td><td>10.465</td><td>1.485</td></tr><tr><td>Plain text</td><td>MTP</td><td>531</td><td>221</td><td>2.403</td><td>7.227</td><td>16.696</td><td>7.045</td><td></td></tr><tr><td>Reasoning-intensive</td><td>AR</td><td>590</td><td>590</td><td>1.000</td><td>5.388</td><td>10.463</td><td>10.468</td><td>1.650</td></tr><tr><td>Reasoning-intensive</td><td>MTP</td><td>588</td><td>224</td><td>2.625</td><td>7.203</td><td>16.581</td><td>6.343</td><td></td></tr><tr><td>Tool calling</td><td>AR</td><td>321</td><td>321</td><td>1.000</td><td>5.611</td><td>10.468</td><td>10.505</td><td>1.590</td></tr><tr><td>Tool calling</td><td>MTP</td><td>321</td><td>124</td><td>2.589</td><td>7.534</td><td>16.908</td><td>6.606</td><td></td></tr></table>

Table 8: Steady-state generation-context measurements from the matched PyTorch Profiler worker traces. CPU and GPU spans are durations of the paired user\_annotation and gpu\_user\_annotation ranges, respectively. GPU span per token is the sum of selected GPUcontext spans divided by completion tokens. $S _ { \mathrm { c o n t e x t } }$ is the $\mathrm { A R \mathrm { - } t o \mathrm { - } M T P }$ ratio of this quantity and must not be interpreted as clean end-to-end speedup or summed kernel time.

The corresponding AR-to-MTP context-span improvement was:

$$
S _ { \mathrm { c o n t e x t } } = \frac { C _ { \mathrm { G P U / t o k e n , A R } } } { C _ { \mathrm { G P U / t o k e n , M T P } } } .\tag{17}
$$

This ratio describes the selected GPU-associated context spans in the instrumented worker traces.   
It is not an end-to-end speedup and does not replace the clean application-level results in RQ1.

Token progress and generation-context frequency. Table 8 reports the context counts and execution-span measurements. In every AR capture, the number of selected steady-state contexts exactly equalled the completion-token count, giving one completion token per AR context. MTP required only 221 contexts for 531 plain-text tokens, 224 contexts for 588 reasoning-intensive tokens, and 124 contexts for 321 tool-calling tokens. The corresponding MTP progress was 2.403, 2.625, and 2.589 completion tokens per context.

These values describe observed output progress relative to the recorded worker contexts. They are not direct measurements of accepted speculative length because the PyTorch worker trace does not explicitly associate each context with the exact number of accepted tokens. Nevertheless, the cross-workload values are consistent with the multi-token progress measured independently in RQ2.

MTP reduced the number of selected steady-state generation contexts by 58.5% for plain text, 62.0% for reasoning-intensive generation, and 61.4% for tool calling. However, the median MTP context had a substantially longer execution span. Median GPU-context span increased from 10.453–10.468 ms under AR to 16.581–16.908 ms under MTP, corresponding to MTP/AR ratios of 1.597, 1.585, and 1.615. Median CPU-context span increased by similarly consistent factors of 1.333, 1.337, and 1.343.

The additional per-context cost therefore ofset part of the speculative progress. Although MTP advanced 2.403–2.625 completion tokens per context, the selected GPU-context span per completion token improved by only 1.485–1.650×. The instrumentation-run end-to-end ratios were 1.410× for plain text, 1.554× for reasoning-intensive generation, and 1.491× for tool calling. These values are lower than the clean application-level ratios reported in RQ1 and are used only to provide context for the profiled executions. Profiling overhead, request boundary work, and execution outside the selected steady-state contexts prevent the context-span ratios from being algebraically equivalent to whole-request latency ratios.

Explicit runtime operations. The PyTorch traces expose the higher-level vLLM runtime hierarchy that produced the additional MTP execution cost. In AR, the repeated runtime path contained execute\_model and sample\_tokens. In MTP, the sampling path additionally contained propose\_draft\_token\_ids and proposer.propose. Figure 2 illustrates this runtime structure using the matched tool-calling traces. Programmatic timestamp containment confirmed the following hierarchy in every MTP workload:

<table><tr><td rowspan="2">Workload</td><td rowspan="2">Mode</td><td rowspan="2">execute_model median (ms)</td><td rowspan="2">sample_tokens median (ms)</td><td rowspan="2">Proposal-ID calls</td><td rowspan="2">Proposal-ID median (ms)</td><td rowspan="2">Proposer calls</td><td rowspan="2">Proposer median (ms)</td></tr><tr><td></td></tr><tr><td>Plain text</td><td>AR</td><td>5.333</td><td>4.128</td><td>0</td><td>一</td><td>0</td><td></td></tr><tr><td>Plain text</td><td>MTP</td><td>7.110</td><td>9.140</td><td>444</td><td>7.857</td><td>222</td><td>6.957</td></tr><tr><td>Reasoning-intensive</td><td>AR</td><td>5.294</td><td>4.205</td><td>0</td><td></td><td>0</td><td></td></tr><tr><td>Reasoning-intensive</td><td>MTP</td><td>7.114</td><td>9.049</td><td>450</td><td>7.746</td><td>225</td><td>6.887</td></tr><tr><td>Tool calling</td><td>AR</td><td>5.506</td><td>3.991</td><td>0</td><td></td><td>0</td><td></td></tr><tr><td>Tool calling</td><td>MTP</td><td>7.405</td><td>9.154</td><td>250</td><td>7.854</td><td>125</td><td>6.967</td></tr></table>

Table 9: Inclusive durations and call counts for selected vLLM runtime functions in the Py-Torch worker traces. Proposal-ID denotes propose\_draft\_token\_ids, while proposer denotes proposer.propose. The displayed durations are inclusive Python range durations and must not be summed because the proposal ranges are nested within the sampling path. Aggregate call counts also include boundary executions outside the selected steady-state context set.

$$
\mathtt { s a m p l e \_ t o k e n s \_ } \mathtt { p r o p o s e \_ d r a f t \_ t o k e n \_ i d s } \mathtt { \mathcal { D } p r o p o s e r \_ p r o p o s e } \mathtt { d r o p o s e } \mathtt { d r o p o s e } .\tag{18}
$$

All proposal calls satisfied this containment relationship. Plain text contained 444 of 444 proposal-ID calls inside sample\_tokens and 222 of 222 proposer calls inside a proposal-ID range. The corresponding counts were 450 of 450 and 225 of 225 for reasoning-intensive generation, and 250 of 250 and 125 of 125 for tool calling. No propose\_draft\_token\_ids or proposer.propose calls were observed in the matched AR worker traces.

Across the three workloads, median execute\_model duration was approximately 33.3–34.5% longer under MTP. The larger diference appeared in sample\_tokens, whose median inclusive duration was approximately 2.15–2.29 times the AR duration. MTP also recorded approximately two propose\_draft\_token\_ids calls for every proposer.propose call. The proposal-related median durations were stable across workloads: 7.746–7.857 ms for propose\_draft\_token\_ids and 6.887–6.967 ms for proposer.propose.

These nested durations are not additive. In particular, the duration of proposer.propose contributes to the inclusive duration of propose\_draft\_token\_ids, which in turn contributes to the inclusive sample\_tokens range. The results therefore identify where the additional MTP runtime work occurs without claiming that the sum of these ranges is an independent latency decomposition.

Connection to GPU execution. The reduced number of steady-state MTP generation contexts observed in PyTorch Profiler is consistent with the lower frequency of repeating CUDA Graph executions measured independently in Nsight Systems. However, the two events are not treated as equivalent units: a PyTorch generation context is a high-level GPU-associated execution range that may contain multiple CUDA Graph executions, kernels, memory operations, and intervening activity.

RQ3 established that MTP executed fewer repeating CUDA Graphs per output token but introduced auxiliary graph sequences, expanded attention, gathering, reduction, sampling-related activity, and a BF16 GEMM-dominated execution path. The PyTorch traces provide the corresponding runtime-level interpretation. The additional device activity occurs beneath an explicit draft-proposal path within sample\_tokens, together with candidate selection and sequence-state processing. The MTP traces additionally contain top-k gathering, sorting, vectorized gathering, segmented reduction, scatter, and index-update operations. These lower-level operations corroborate the additional sampling and proposal work, while the Python hierarchy establishes its semantic placement in the runtime.

The raw operator and kernel counts were not interpreted as independent latency contributions. CUDA Graph replay can change the visibility of operator ranges, and the same operation name can occur in diferent runtime contexts. RQ4 therefore uses the explicitly recorded runtime hierarchy, paired context spans, and completion-token normalization as its primary evidence. The kernel-level counts and durations remain part of RQ3.

Answer to RQ4. Across the three matched workloads, MTP reduced the number of steadystate generation contexts by 58.5–62.0% and advanced 2.403–2.625 completion tokens per context. This speculative progress did not translate proportionally into wall-clock acceleration because each MTP context carried substantially more runtime and GPU-associated work. Median GPUcontext span was 58.5–61.5% longer under MTP, while median CPU-context span was 33.3–34.3% longer.

PyTorch Profiler explicitly associated this additional work with a draft-proposal path absent from AR. All observed propose\_draft\_token\_ids ranges were nested within sample\_tokens, and all proposer.propose ranges were nested within proposal-ID processing. MTP sample\_tokens duration was approximately 2.15–2.29 times the AR duration, while execute\_model was approximately one-third longer. The resulting selected GPU-context span per completion token improved by 1.485–1.650×, rather than by the full 2.403–2.625× completion-token progress observed per context.

The gap between speculative progress and realized acceleration therefore reflects the cost of producing and processing speculative candidates, not a failure to achieve multi-token progress. MTP reduced the number of repeated generation executions, but each execution incorporated proposal, expanded sampling, candidate-selection, tensor-manipulation, and state-update work. These per-context costs ofset part of the useful token-progress advantage.

Interpretation boundaries. The PyTorch Profiler traces were instrumented executions and must not replace the clean benchmark results used for application-level performance claims. The selected GPU annotation spans are associated execution ranges rather than summed kernel durations or direct GPU-utilization measurements. Likewise, completion tokens per context describe aggregate output progress and must not be interpreted as an exact per-context speculative acceptance count.

## 4.6 Selected-Kernel Analysis

The Nsight Systems analysis identified a recurring gemv2T\_kernel\_val kernel in AR and an ampere\_bf16\_s16816gemm kernel in MTP. Because these dominant kernels had nearly identical median invocation durations, one representative instance of each was inspected using Nsight Compute. Both instances were taken from the matched reasoning-intensive captures. This analysis provides supporting microarchitectural evidence and is not intended to represent all GEMV or GEMM invocations.

As shown in Table 10 and Figure 3, both kernels operated near the A10G’s nominal DRAMbandwidth limit of 600 GB/s [4]. The MTP GEMM sustained 575.07 GB/s, or 95.93% of maximum memory bandwidth, while the AR GEMV sustained 579.04 GB/s, or 96.59%. Their low L2 hit rates further indicate that both selected instances depended heavily on device-memory traffic. Therefore, the transition from GEMV-dominated AR execution to GEMM-dominated MTP execution did not correspond to a transition from memory-bound to compute-bound execution.

<table><tr><td>Metric</td><td>MTP GEMM</td><td>AR GEMV</td></tr><tr><td>Duration (ms)</td><td>2.79</td><td>2.78</td></tr><tr><td>Memory throughput (GB/s)</td><td>575.07</td><td>579.04</td></tr><tr><td>Maximum memory bandwidth (%)</td><td>95.93</td><td>96.59</td></tr><tr><td>Compute throughput (%)</td><td>22.06</td><td>28.89</td></tr><tr><td>L1/TEX hit rate (%)</td><td>0.06</td><td>7.63</td></tr><tr><td>L2 hit rate (%)</td><td>4.59</td><td>4.12</td></tr><tr><td>Executed IPC (instr./cycle)</td><td>0.18</td><td>0.54</td></tr><tr><td>Issue slots busy (%)</td><td>4.57</td><td>13.58</td></tr><tr><td>Streaming Multiprocessors (SM) busy (%)</td><td>22.06</td><td>13.58</td></tr><tr><td>Achieved occupancy (%)</td><td>8.35</td><td>66.33</td></tr><tr><td>Active warps per SM</td><td>4.01</td><td>31.84</td></tr><tr><td>Registers per thread</td><td>102</td><td>64</td></tr><tr><td>Shared memory per block (KB)</td><td>49.15</td><td>2.56</td></tr></table>

Table 10: Nsight Compute measurements for representative dominant kernels from the reasoningintensive AR and MTP captures.

The kernels nevertheless used the GPU diferently. The MTP GEMM primarily used the Tensor pipeline and recorded 22.06% SM busy, but its resource footprint of 102 registers per thread and 49.15 KB of shared memory per block limited achieved occupancy to 8.35%. The AR GEMV recorded lower SM busy at 13.58%, but achieved 66.33% occupancy with 64 registers per thread and 2.56 KB of shared memory per block while using load/store and conventional arithmetic pipelines.

These measurements reinforce the RQ3 result: MTP’s advantage did not arise from a faster dominant kernel. The selected GEMM and GEMV instances had nearly identical durations and reached a similar memory-bandwidth ceiling. The workload-level improvement instead resulted from MTP requiring fewer dominant matrix-kernel and recurring group executions per generated token, despite introducing a more complex and individually costlier execution path.

## 5 Discussion

## 5.1 Useful Progress Versus Speculative Cost

MTP accelerated generation because the reduction in repeated execution frequency outweighed its higher per-execution cost. Accepted proposals increased useful token progress, while proposal generation, expanded verification, sampling, and state updates made each MTP execution longer and more complex. The observed benefit therefore arose from amortizing this additional work across fewer recurring executions per generated token, not from making individual executions or dominant kernels faster.

## 5.2 Analytical Model Versus Implemented Execution

The results support the analytical model’s central trade-of: speculation pays when additional token progress exceeds the cost of producing and verifying proposals. However, the theoretical draft-cost coeficient c could not be isolated from the traces. The measured cadence and contextspan ratios instead describe the complete implemented path, including proposal, verification, sampling, scheduling, and device activity, and must not be treated as empirical estimates of c.

## 5.3 Practical Implications

Speculative-decoding evaluation should consider accepted progress, execution frequency, and perexecution cost together. Acceptance rate alone cannot predict wall-clock speedup, while isolated

kernel duration can conceal a substantial reduction in how often that kernel executes. For deployment and optimization, the relevant objective is therefore useful output progress relative to measured end-to-end latency, rather than acceptance or isolated kernel performance alone.

## 6 Threats to Validity

The study evaluated one Gemma 4 target–drafter pairing, one NVIDIA A10G GPU, two-token speculation, single-request serving, and three prompts per workload, which limits generalization to other models, hardware, speculative depths, batch sizes, and concurrent deployments. Acceptance telemetry represented aggregate logging windows rather than individual requests, while the profiler analysis used one matched request per workload and mode. PyTorch contexts and CUDA Graph executions were observable implementation units rather than logical decoding steps, and the Nsight Compute analysis characterized only one selected GEMM and GEMV instance. Profiler results are therefore used as explanatory evidence, while application-level claims remain grounded in the separate clean benchmark.

## 7 Future Work

This study deliberately prioritizes depth of performance analysis over breadth of configuration exploration. The speculative depth was intentionally fixed at γ=2 throughout the study. The objective was not to identify an optimal speculative depth or characterize speculative-depth scaling, but rather to explain the runtime and GPU-execution behavior of a deployed two-token MTP configuration.

An important direction for future work is speculative-depth scaling, which could evaluate multiple γ values to examine how acceptance behavior, recurring GPU-execution cadence, proposalprocessing overhead, and end-to-end performance evolve as speculative depth changes. Such a study could characterize the trade-of between speculative progress and the cost of proposal generation, verification, and acceptance processing.

## 8 Conclusion

In the evaluated deployment, two-token MTP increased output throughput by 1.91× to 2.19× across all prompts while producing smaller TTFO improvements of 10.0–14.2%. Cross-layer profiling showed that this acceleration did not arise from a faster dominant kernel: MTP introduced a costlier proposal and verification path but required substantially fewer recurring GPU executions per generated token. The central result is therefore that MTP improved inference through amortization, with greater token progress more than compensating for the additional speculative-execution cost.

## References

[1] Fabian Gloeckle et al. “Better & faster large language models via multi-token prediction”. In: arXiv preprint arXiv:2404.19737 (2024).

[2] Woosuk Kwon et al. “Eficient Memory Management for Large Language Model Serving with PagedAttention”. In: Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles. 2023.

[3] Yaniv Leviathan, Matan Kalman, and Yossi Matias. “Fast inference from transformers via speculative decoding”. In: International Conference on Machine Learning. PMLR. 2023, pp. 19274–19286.

[4] NVIDIA. NVIDIA A10G Tensor Core GPU: Accelerated Compute and Graphics for the AWS Cloud. NVIDIA. 2022. url: https://d1.awsstatic.com/product-marketing/ec2/ NVIDIA\_AWS\_A10G\_DataSheet\_FINAL\_02\_17\_2022.pdf (visited on 09/25/2026).

[5] Gemma Team et al. “Gemma 4 technical report”. In: arXiv preprint arXiv:2607.02770 (2026).

## A Experimental Protocol

The complete frozen experimental configuration is reproduced below. The protocol was finalized before primary measurement and preserved with its SHA-256 integrity record.

Listing 1: Frozen experimental protocol, version 1.2.

```yaml
protocol:
name: GPU Inference Profiling Study
version: "1.2"
status: frozen
research_question: >-
How does MTP speculative decoding affect performance across plain-text,
reasoning-intensive, and tool-calling workloads, and what execution behavior
explains the differences?
deployments:
normal: {service: prof-gemma4vllm.service, speculative_decoding: false}
mtp: {service: prof-gemma4vllm-mtp.service, speculative_decoding: true, speculative_tokens:
2}
request:
url: "http://127.0.0.1:8000/v1/chat/completions"
model: "google/gemma-4-e4b"
model_path: "<path_to_target_model>"
drafter_path: "<path_to_drafter_model>"
stream: true
include_usage: true
temperature: 0.0
top_p: 1.0
seed: 42
benchmark_max_tokens: 1024
profile_max_tokens: 1024
parallel_tool_calls: false
max_model_len: 8192
max_num_seqs: 1
gpu_memory_utilization: 0.80
chunked_prefill: true
max_num_batched_tokens: 2048
kv_cache_dtype: "bfloat16"
prefix_caching_enabled: true
chat_template: "<path_to_target_model>/tool_chat_template_gemma4.jinja"
chat_template_content_format: "openai"
reasoning_parser: "gemma4"
tool_call_parser: "gemma4"
responses_api_store: false
chat_template_kwargs:
enable_thinking: true
43 phase_timing:
44 clock: time.perf_counter_ns
45 reasoning: delta.reasoning
46 final_answer: delta.content
tool_call: delta.tool_calls
48 timestamps: [request_start_ns, first_output_ns, first_reasoning_ns, first_final_answer_ns,
```

```yaml
first_tool_call_ns, request_end_ns]
49 nvtx_markers: [RUN_START, FIRST_OUTPUT, FIRST_REASONING, FIRST_FINAL, FIRST_TOOL_CALL,
RUN_END]
50 claim_limit: Profiler intervals are phase-associated; kernels are not inherently reasoning
specific.
tools:
- type: function
function:
name: get_weather
description: Get the weather forecast for a city and date.
parameters:
type: object
properties:
city: {type: string}
date: {type: string, description: ISO date YYYY-MM-DD}
62 unit: {type: string, enum: [celsius, fahrenheit]}
63 required: [city, date, unit]
additionalProperties: false
type: function
66 function:
name: get_local_time
68 description: Get the current local time for a city.
parameters:
type: object
properties: {city: {type: string}}
required: [city]
additionalProperties: false
type: function
function:
name: create_reminder
description: Create a reminder.
parameters:
type: object
properties:
title: {type: string}
datetime: {type: string, description: ISO 8601 date-time}
required: [title, datetime]
84 additionalProperties: false
85
86 workloads:
plain_text:
88 expected: reasoning followed by exactly eight poem lines
89 prompts:
90 - id: plain_001
system: Follow the requested final format exactly. Add no title or commentary.
user: Write exactly eight lines of free-verse poetry about a spacecraft travelling
beyond the Solar System. Each line must contain six to twelve words.
- id: plain_002
system: Follow the requested final format exactly. Add no title or commentary.
user: Write exactly eight lines of free-verse poetry about rain falling on a quiet
city at night. Each line must contain six to twelve words.
96 - id: plain_003
system: Follow the requested final format exactly. Add no title or commentary.
user: Write exactly eight lines of free-verse poetry about an old computer starting
after many years. Each line must contain six to twelve words.
99
100 reasoning_intensive:
101 expected: reasoning followed by one strict FINAL line
102 prompts:
103 - id: reasoning_001
104 system: Reason carefully. End with exactly one line in the required FINAL format.
105 user: "A program is 80% perfectly parallelizable and 20% serial. Using Amdahl’s law,
calculate speedup with 8 processors and the maximum theoretical speedup. End exactly:
```

FINAL: S8=<3 decimals>; Smax=<3 decimals>"   
106 expected\_final: "FINAL: S8=3.333; Smax=5.000"   
107 - id: reasoning\_002   
108 system: Reason carefully. End with exactly one line in the required FINAL format.   
109 user: "A 120-second workload is 75% perfectly parallelizable and 25% serial. Using   
Amdahl’s law, calculate speedup and runtime with 6 processors. End exactly: FINAL: speedup   
=<3 decimals>; runtime=<3 decimals>s"   
110 expected\_final: "FINAL: speedup=2.667; runtime=45.000s"   
111 id: reasoning\_003   
112 system: Reason carefully. End with exactly one line in the required FINAL format.   
113 user: "A program has serial fraction 0.10. Using Amdahl’s law, find the minimum   
integer processors needed for speedup at least 5. End exactly: FINAL: processors=<integer   
>; speedup=<3 decimals>"   
114 expected\_final: "FINAL: processors=9; speedup=5.000"   
115   
116 tool\_calling:   
117 expected: reasoning followed by exactly one get\_weather call   
118 tool\_choice: auto   
119 prompts:   
120 - {id: tool\_001, system: Use exactly one available tool and do not answer directly.,   
user: Get the weather forecast for Luxembourg City on 2026-10-15 in Celsius.,   
expected\_tool: get\_weather, expected\_arguments: {city: Luxembourg City, date: ’2026-10-15’   
, unit: celsius}}   
121 - {id: tool\_002, system: Use exactly one available tool and do not answer directly.,   
user: Get the weather forecast for Gurugram on 2026-11-20 in Fahrenheit., expected\_tool:   
get\_weather, expected\_arguments: {city: Gurugram, date: ’2026-11-20’, unit: fahrenheit}}   
122 - {id: tool\_003, system: Use exactly one available tool and do not answer directly.,   
user: Get the weather forecast for Helsinki on 2026-12-05 in Celsius., expected\_tool:   
get\_weather, expected\_arguments: {city: Helsinki, date: ’2026-12-05’, unit: celsius}}   
123   
124 cache:   
125 primary: controlled\_cold   
126 method: Insert a pre-generated unique identifier at the beginning of each user message;   
reuse the identical prepared payload for paired normal and MTP runs.   
127 warm\_cache\_experiment: deferred   
128   
129 warmup:   
130 settle after health seconds: 30   
131 minimum\_per\_workload: 3   
132 maximum\_per\_workload: 10   
133 require\_two\_consecutive\_requests\_without\_jit\_warning: true   
134   
135 repetitions:   
136 pilot: {prompts\_per\_workload: 2, repetitions\_per\_prompt: 5}   
137 primary: {prompts\_per\_workload: 3, valid\_repetitions\_per\_prompt: 20}   
138   
139 metrics: [time\_to\_first\_output\_ms, time\_to\_first\_final\_output\_ms, end\_to\_end\_latency\_ms,   
reasoning\_associated\_duration\_ms, final\_or\_tool\_associated\_duration\_ms, prompt\_tokens,   
completion\_tokens, output\_tokens\_per\_second]   
140   
141 validity:   
142 benchmark\_invalid\_if: [JIT during measurement, wrong service or uncontrolled GPU work, HTTP   
or stream failure, profiler active during benchmark, cache condition not achieved]   
143 phase\_mapping\_invalid\_if: [phase boundaries unavailable, required phase timestamp missing]   
144 output\_invalid\_if: [finish\_reason is length, required final output absent, tool call or   
arguments mismatch]   
145 preserve\_invalid\_runs: true   
146   
147 artifacts:   
148 results: results.csv   
149 per\_run: [runs/<run\_id>/raw.jsonl, runs/<run\_id>/summary.json]   
150 profiles: [profiles/torch, profiles/nsys, profiles/ncu]   
151 run\_id: v1-<mode>-<workload>-<prompt\_id>-r<repetition>

![](images/8ad764494c0dd61baa2833807e8129659d0a8378989a21c90ee036bd4c403a02.jpg)  
(d) MTP Single auxiliary-sequence zoom  
Figure 1: Representative post-first-output GPU execution from the matched toolcalling captures. Panel (a) shows consecutive executions of the AR repeating CUDA Graph, Graph 520/GraphExec 521, with the intervening gemv2T\_kernel\_val-dominated path. Panel (b) shows consecutive executions of the MTP repeating CUDA Graph, Graph 313/GraphExec 314, with the intervening BF16 GEMM and compact auxiliary graph and kernel activity. Panel (c) enlarges the compact MTP activity between consecutive Graph 313 executions and exposes two auxiliary graph sequences. Panel (d) enlarges one sequence, showing the ordering $2 9 8 \to 3 0 1 \to 3 0 4 \to 3 0 7 \to 3 1 0 \to 1$ , interleaved with short supporting kernel activity. A recurring execution group was defined as the start-to-start interval between consecutive executions of the selected repeating CUDA Graph. The screenshots are illustrative; graph ordering, kernel identities, execution counts, durations, and cadence statistics were calculated from the Nsight Systems SQLite exports.

![](images/968101e92366d35732a531d7d1dc04e0e956206086121c46827f225ac843ec88.jpg)  
Figure 2: Representative PyTorch Profiler runtime hierarchy from the matched toolcalling traces. Panels (a) and (b) show one complete \_process\_engine\_step for AR and MTP, respectively, including the execute\_model and sample\_tokens paths. Panel (c) enlarges the MTP sampling path, in which propose\_draft\_token\_ids and proposer.propose are nested within sample\_tokens; two gemma4\_mtp.py:forward ranges are visible within the representative proposal subtree. The screenshots are illustrative; context pairing, function containment, call counts, and durations were calculated programmatically from the compressed PyTorch trace JSON artifacts.

![](images/8d4a9a06fd885f677776f12ff80df4553db8eedad3ed33b802e86284a42832da.jpg)  
(a) MTP-associated GEMM

![](images/8d1822bc2bec2c919150cd9c3315418f661ef95150c4a1a1e56b0e5c4b320f1c.jpg)  
(b) AR-associated GEMV  
Figure 3: Nsight Compute comparison of representative dominant kernels from the reasoning-intensive captures. The MTP-associated GEMM (left) primarily used the Tensor pipeline, whereas the AR-associated GEMV (right) used conventional arithmetic pipelines. Both selected kernels approached the observed memory-bandwidth limit despite their diferent pipeline utilization.