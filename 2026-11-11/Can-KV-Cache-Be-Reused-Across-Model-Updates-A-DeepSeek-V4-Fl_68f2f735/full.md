# Can KV Cache Be Reused Across Model Updates? A DeepSeek-V4-Flash Case Study

Zhikun Tao Fangyong Jin Junping Zhao<sup>∗</sup>

Rank Computing

## Abstract

Reusing key–value (KV) caches across language model updates could avoid repeated computation of shared context, but changes in model weights raise a fundamental question about what these caches preserve. We investigate direct cache reuse across four model updates, with a detailed study of DeepSeek-V4-Flash to Flash-0731. Our central finding is that caches can retain useful task information without faithfully reproducing the updated model’s behavior. Retrieval accuracy varies substantially across updates, ranging from near-native performance to large losses. With source caches already available, reuse achieves median time-to-first-token speedups of 2.10–5.08× on three longer-prefix QA benchmarks. To characterize the behavioral diferences behind these outcomes, we introduce Functional Cache Divergence, which compares next-token distributions under reused and native caches. Direct KV cache reuse better preserves native model behavior in natural-text continuation, whereas compatibility degrades with increasing prefix length in single-token retrieval tasks. Answer-margin analyses help explain how correct retrieval decisions survive substantial internal state changes. Yet even agreement on the first generated token does not ensure agreement in answer content or stopping behavior. These findings establish task utility and fidelity to native inference as distinct criteria for evaluating cross-version KV cache reuse.

Keywords: language models, KV cache reuse, model updates, compatibility, empirical analysis

![](images/1cc2e11de10e1fc79929a4f462d2b343c18a8ec613ef5f4cdc8a8ade54c8309e.jpg)

![](images/09defaa523fd0c37be94a744381eab0de5d9abb899fd65067b1cad48d22910c6.jpg)

![](images/ee89ef9d33dbc8d6166364a66f4b0de944260a42a2f16e17a65c5f074fcdad29.jpg)

![](images/c2188c8fd1dfcc7c54641c2168066e9dc92ffceb2bbaef490a7f90281cce78b1.jpg)  
Figure 1 | Needle-in-a-Haystack (NIAH) (Hsieh et al., 2024) retrieval accuracy across four model pairs and prefix lengths in the original sweep. Each prefix contains lots of key-value records, with each key and value represented by a single token. The model is asked to return the value for a given key, testing its ability to retrieve specific information from long contexts. Panel titles indicate source → target. Source and Target use their own prefix KV caches, while KV Reuse denotes target-model inference using the source model’s prefix KV cache.

## 1. Introduction

Large language models (LLMs) are increasingly refined through successive rounds of post-training, allowing improved versions to be released rapidly without changing their core architecture or size. For example, DeepSeek-V4-Flash-0731 retained the architecture and size of its preview model and improved agent capabilities through further post-training, roughly three months after the preview’s release on April 24, 2026 (DeepSeek-AI, 2026c). GLM-5.3 similarly builds on GLM-5.2’s base model, with improvements in coding and agent tasks attributed to post-training (Z.ai, 2026).

Even as models evolve, applications often continue to rely on the same context. This includes shared system prompts, frequently requested prompts, retrieved documents reused across queries in retrievalaugmented generation (RAG), and common instructions or examples for recurring data-processing tasks. In these long conversations and repeated document queries, key–value (KV) caches avoid recomputing previously processed context. A cache created before an update still encodes the context using the old model weights, reusing it could save computation but may change how the updated model uses that context and what it generates.

Prior work explores KV-based communication among agents (Shi et al., 2026b; Yun et al., 2026), mapped cache reuse between models in the same family (Heo et al., 2026; Qu et al., 2026), and reuse across changing contexts and model checkpoints (Shi et al., 2026a). We focus on direct KV-cache reuse across successive model versions with unchanged core architecture and size. The updated model consumes its predecessor’s cache without mapping or selective recomputation. We ask: what remains compatible when a newer model consumes caches produced by an earlier version? We evaluate retrieval, answer quality, time to first token (TTFT), and generation behavior using three configurations. Target and Source each use one checkpoint for both prefill and generation. KV Reuse combines source prefill with target continuation.

The four model updates in Figure 1 show more reliable KV reuse in models with stronger retrieval ability. On single-token needle-in-a-haystack (NIAH) tasks with 512–32K prefixes, GLM-5.2 → GLM-5.3 reuse closely tracks Target through 16K and improves 32K accuracy from 86.46% to 90.62%. DeepSeek-V4.1-Flash → Flash-1017 maintains near-native accuracy, with both Target and KV Reuse reaching 97.92% at 32K. DeepSeek-V4-Flash → Flash-0731 reuse improves Target accuracy at every measured length, although retrieval becomes less accurate as prefixes grow. Qwen3-30B-A3B → Instruct-2507 loses accuracy under reuse at every length, with the largest gap at 32K despite RoPE phase alignment.

Our detailed evaluation centers on DeepSeek-V4-Flash → Flash-0731. We first compare direct reuse with a ridge mapper adapted to DeepSeek’s composite cache, then measure answer quality and TTFT on four long-context QA benchmarks with the source cache prepared. We also inject Flash’s caches into both Flash-Base and Flash-0731 to examine transfer toward earlier and later checkpoints. Complete QA responses reveal changes in answer content and stopping behavior.

We connect these outcomes to changes in predictions and internal computation. Functional Cache Divergence (FCD) measures the diference between the target model’s next-token probability distributions using its own cache and the source cache, with subsequent input tokens and prediction positions held fixed. This scaled Jensen–Shannon divergence captures probability shifts even when the selected token stays the same, revealing distinct patterns for retrieval and natural-text continuation.

Internal analyses explain how retrieval survives state changes. Flash-0731 retains many correct answers through suficient score margins, while Flash-Base recovers retrieval with higher answertoken scores despite changed states and cache selections. At the three inspected layers, replacing hidden states changed by earlier layers afects attention outputs more than replacing local caches.

This technical report presents four main findings:

1. Stronger retrieval models preserve correct answers more reliably under KV reuse. GLM-5.3 and DeepSeek-V4.1-Flash-1017 retain high retrieval accuracy when using their predecessors’ KV caches. At 32K, their harmful flip rates—the fractions of Target-correct answers that become incorrect under reuse—are 4.22% and 0.53%, compared with 26.53% for V4-Flash and 78.30% for Qwen. These larger models with stronger retrieval preserve correct answers more reliably. V4.1-Flash reuse also reaches 61.46% accuracy at a 512K-token prefix (Section 3.1).

2. KV reuse substantially accelerates QA on longer contexts. With the source cache prepared, median TTFT speedups are 5.08× on HotpotQA, 2.10× on 2WikiMQA, and 2.51× on MultiFieldQA-en. MultiFieldQA-en maintains F1, while HotpotQA and 2WikiMQA lose about 3 and 7 F1 points. On BABILong’s shorter prefixes, cache-access costs outweigh the saved computation, yielding a 0.86× speedup (Section 3.2).

3. KV caches carry the source model’s processing of context and response instructions, changing how receivers answer. Flash-produced caches improve retrieval in both Base and Flash-0731. With the same instruction prompt, Base typically restates the question when using its own cache and scores 0% accuracy. Injecting Flash’s cache raises its accuracy to 91.15% at 2K tokens while its weights remain fixed. In natural QA, reuse also changes answer content and stopping, including cases where both runs begin with the same answer token (Sections 3.3 and 3.4).

4. Changes in predictive distributions grow with context length for retrieval and stay stable for natural-text continuation. In the Flash update, mean NIAH FCD rises from 2.87 at 512 tokens to 30.62 at 32K, while FineWeb-Edu stays within 3.04–3.64 across the same lengths. Replacing the prefix cache increasingly changes retrieval-token probabilities as context grows, while natural-text continuation retains a much lower, flatter divergence profile (Section 4.1).

## 2. Experimental Setup and Measurements

We study how a target model continues from a prefix cache produced by another checkpoint in the same model family. The evaluation measures retrieval, answer quality, and latency, then examines complete responses, predictive distributions, and internal computation. DeepSeek-V4-Flash → Flash-0731 is the main pair for QA and controlled interventions. Flash-to-Base adds a comparison across training stages, while GLM, V4.1-Flash, and Qwen provide retrieval comparisons across updates.

## 2.1. Inference Configurations

Let � denote the source model and � the target model. Each input is divided into a shared prefix � and a question sufix �. Prefilling the same prefix with the two models produces $C _ { S } = \mathrm { P r e f i l l } _ { S } ( P )$ and $C _ { T } = \mathrm { P r e f i l l } _ { T } ( P )$ . Here, � denotes the complete prefix state required for continuation, including the attention and compression states used by DeepSeek-V4. We compare three configurations (Figure 2): Target uses � for both prefill and continuation; Source uses � for both stages; and KV Reuse gives �<sub>�</sub> to �, which processes � and generates the response. At output position �, their predictive distributions are

$$
\begin{array} { r } { p _ { j } ^ { \mathrm { T a r g e t } } = p _ { T } ( \cdot  { | } C _ { T } , Q , y _ { < j } ) , } \\ { p _ { j } ^ { \mathrm { S o u r c e } } = p _ { S } ( \cdot  { | } C _ { S } , Q , y _ { < j } ) , } \\ { p _ { j } ^ { \mathrm { R e u s e } } = p _ { T } ( \cdot  { | } C _ { S } , Q , y _ { < j } ) . } \end{array}\tag{1}
$$

For free generation, $y _ { < j }$ is the output history of the corresponding configuration. For paired distribution measurements, all configurations being compared receive the same prescribed history. Prompt token

IDs and decoding settings are identical within each model pair, and free generation uses greedy decoding. Table 1 additionally compares Ridge Mapper for Flash → Flash-0731: the target continues from $M ( C _ { S } )$ , where � adapts the closed-form ridge mapping of Heo et al. (2026) to DeepSeek’s composite cache. Appendix A.1 describes calibration and cache adaptation. Figure 3(b) applies the same centered ridge approach to V4.1-Flash → Flash-1017, with adaptation to its cache representation.

![](images/b4649a74be9e8842f101e552b70b34613ad03907ec8798d900744a7e0ed47b1d.jpg)  
Figure 2 | Three inference configurations on the same prefix and question. Target and Source use one model throughout. KV Reuse combines the source-produced prefix cache with target-model continuation.

## 2.2. Tasks and Datasets

For retrieval compatibility, we use controlled key–value NIAH tasks (Hsieh et al., 2024; Zhu et al., 2026) with prefixes of 512, 2K, 4K, 8K, 16K, and 32K tokens. Each length has eight independent histories and 24 queries per history, distributed near 10%, 50%, and 90% of the prefix. Contexts contain approximately one record per 16 tokens, with single-token keys and values. This controlled retrieval task tests a model’s ability to locate information in context and directly measures how KV reuse afects the target’s retrieval performance. We also extend the V4.1-Flash update to 64K, 128K, 256K, and 512K, retaining eight histories and 24 queries per history at each added length. Appendix A.1 gives the data construction and generation limits.

For task quality and latency, we evaluate the KV Reuse from DeepSeek-V4-Flash to DeepSeek-V4- Flash-0731 on 585 examples: 300 from BABILong, 100 from HotpotQA, 100 from 2WikiMQA, and 85 from MultiFieldQA-en (Bai et al., 2024a; Ho et al., 2020; Kuratov et al., 2024; Yang et al., 2018). We measure answer scores and Time to First Token (TTFT) with the source prefix cache already prepared. Appendix A.2 details dataset selection, input lengths, and evaluation settings.

For behavioral and mechanistic analysis, we compare selected 4K NIAH state traces for Flash → Flash-0731 and Flash → Flash-Base, alongside token-score margins on the same 192 sweep queries for each receiver. Controlled interventions use a separate Flash-update cohort of 96 queries from 32 independent 4K contexts, with a further 48 queries from 16 contexts for matched perturbations. Natural-QA diagnostics use 39 HotpotQA and 50 2WikiMQA examples to examine answer changes, generation agreement, and stopping. Each task contributes 20 preselected random examples, and frequency summaries use only these random subsets. Appendix A.3 describes the diagnostic cohorts and their selection.

For functional cache divergence, we separately measure the Flash update on the same 1,152 NIAH queries used in its six-length sweep, comparing full predictive distributions at the first answer position. We also evaluate 64 long documents from FineWeb-Edu (Penedo et al., 2024). The same documents are used at all six lengths, with 32 matched continuation positions per document and length, yielding 12,288 paired positions. Both target runs consume identical subsequent document tokens. Appendix A.4 specifies sampling and the matched prediction positions.

Paired runs share input tokens and execution settings within each experiment. Appendix B.2 gives runtime and statistical settings.

## 3. Efectiveness and Behavioral Efects of KV Reuse

We first examine retrieval across model updates and prefix lengths, then evaluate the quality and latency of the DeepSeek-V4-Flash handof on natural-language tasks. We next study how transferred caches afect a receiver’s answers, including changes in answer content and stopping.

## 3.1. Retrieval Evaluation

Table 1 | NIAH answer-prefix accuracy (%; higher is better) across model updates. Each prefix length contains 192 queries from eight histories. Sizes in parentheses denote nominal backbone parameters. Target and Source use their own caches; KV Reuse uses the target model with the source cache; Ridge Mapper maps that cache before target-model continuation.

<table><tr><td rowspan="2">Configuration</td><td colspan="6">Prefix length (tokens)</td></tr><tr><td>512</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td></tr><tr><td>GLM-5.2 → GLM-5.3 (750B)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Target</td><td>98.44</td><td>99.48</td><td>98.44</td><td>98.44</td><td>94.79</td><td>86.46</td></tr><tr><td>Source</td><td>96.88</td><td>96.88</td><td>95.83</td><td>97.92</td><td>92.71</td><td>89.58</td></tr><tr><td>KV Reuse</td><td>99.48</td><td>99.48</td><td>98.96</td><td>98.96</td><td>95.31</td><td>90.62</td></tr><tr><td colspan="7">DeepSeek-V4.1-Flash → DeepSeek-V4.1-Flash-1017 (552B)</td></tr><tr><td>Target</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.48</td><td>100.00</td><td>97.92</td></tr><tr><td>Source</td><td>99.48</td><td>100.00</td><td>99.48</td><td>99.48</td><td>98.96</td><td>98.44</td></tr><tr><td>KV Reuse</td><td>98.96</td><td>99.48</td><td>99.48</td><td>99.48</td><td>99.48</td><td>97.92</td></tr><tr><td colspan="7">DeepSeek-V4-Flash → DeepSeek-V4-Flash-0731 (284B)</td></tr><tr><td>Target</td><td>95.31</td><td>92.19</td><td>90.10</td><td>68.75</td><td>50.52</td><td>25.52</td></tr><tr><td>Source</td><td>98.96</td><td>98.96</td><td>97.92</td><td>89.58</td><td>79.17</td><td>55.73</td></tr><tr><td>KV Reuse</td><td>98.96</td><td>96.35</td><td>93.23</td><td>75.00</td><td>65.10</td><td>31.25</td></tr><tr><td>Ridge Mapper (Heo et al., 2026)</td><td>98.44</td><td>92.19</td><td>92.71</td><td>76.04</td><td>59.38</td><td>40.10</td></tr><tr><td colspan="7">Qwen3-30B-A3B → Qwen3-30B-A3B-Instruct-2507 (30.5B)</td></tr><tr><td></td><td>87.50</td><td>76.04</td><td>69.79</td><td>72.40</td><td>60.42</td><td></td></tr><tr><td>Target Source</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>55.21</td></tr><tr><td></td><td></td><td>57.81</td><td>56.77</td><td>51.56</td><td></td><td>0.00</td></tr><tr><td>KV Reuse</td><td>80.21</td><td></td><td></td><td></td><td>44.27</td><td>13.02</td></tr></table>

Table 1 and Figure 1 compare Target, Source, and KV Reuse on NIAH retrieval from 512 to 32K tokens, with model pairs ordered by backbone size (DeepSeek-AI, 2026a,b; GLM-5-Team, 2026). We use answer-prefix accuracy: a response is correct if its whitespace-trimmed text begins with the requested value, and failed requests score zero. GLM reuse closely tracks Target through 16K and improves accuracy from 86.46% to 90.62% at 32K. V4.1-Flash reuse stays within 1.04 percentage points of Target at every length, with both reaching 97.92% at 32K. V4-Flash reuse improves Target accuracy throughout the sweep, including from 25.52% to 31.25% at 32K, although it remains below Source’s 55.73% at that length. Qwen shows the opposite pattern: reuse reduces accuracy at every length, with the largest gap at 32K, where accuracy falls from 55.21% to 13.02% even with RoPE phase alignment.

Using the closed-form ridge mapper of Heo et al. (2026), adapted to DeepSeek’s composite cache, improves retrieval at some lengths but does not consistently outperform direct reuse. At 32K, Ridge Mapper reaches 40.10% accuracy versus 31.25% for KV Reuse, a gain of 8.85 percentage points (95% history-cluster bootstrap interval: [5.21, 12.50]). At 2K, accuracy instead falls from 96.35% to 92.19%, and at 16K from 65.10% to 59.38%. Mapping therefore provides a useful long-prefix comparison without establishing a uniform advantage across the sweep.

![](images/a1194bd599f8d3608efd84f4dbd3309edf3d1a7d17633a9f63232b86462942fd.jpg)

![](images/bfff0628b21997a0792506a8df045381e3606deb82abe80ab8e74dd958c74801.jpg)  
Figure 3 | NIAH retrieval under KV reuse. (a) Harmful flip rate in the original sweep across four model updates over 512–32K prefixes, conditional on Target answering correctly. (b) Matched Target, Source, KV Reuse, and Ridge Mapper answer-prefix accuracy for V4.1-Flash → Flash-1017 over 512–512K prefixes. Each length has 192 queries from eight histories; error bars in (b) show 95% history-cluster bootstrap intervals. Data construction and runtime details are given in Appendices A.1 and B.2.

Figure 3(a) examines whether reuse preserves answers that Target gets right. We define the harmful flip rate (HFR) as $\mathrm { H F R } = 1 0 0 \times N _ { \mathrm { f l i p } } / N _ { T }$ . For each model pair and length, $N _ { T }$ counts Target-correct queries, and $N _ { \mathrm { f l i p } }$ counts those answered incorrectly under reuse. Lower HFR means fewer correct answers are lost. At 32K, HFR remains low for GLM (4.22%) and V4.1-Flash (0.53%), but rises to 26.53% for V4-Flash and 78.30% for Qwen. An overall accuracy gain can still hide losses: V4-Flash reuse breaks 13 of Target’s 49 correct answers at 32K but repairs 24 errors.

We extend the V4.1-Flash → Flash-1017 evaluation to 512K tokens to test whether its near-native retrieval persists at longer prefixes (Figure 3(b)). At 64K, KV Reuse remains close to Target, with accuracies of 94.79% and 95.83%, respectively. Across 128K–512K, both become less accurate, and reuse falls below Target by 5.21, 8.33, and 4.69 percentage points. At 512K, reuse achieves 61.46%, compared with 66.15% for Target and 57.81% for Source. Thus, reuse retains substantial retrieval ability in these longer settings, but the close agreement observed at shorter prefixes does not persist throughout the extended sweep. The additional loss from reuse does not increase monotonically with length.

Ridge Mapper closely matches direct reuse through 8K, then falls behind as prefixes grow. At 256K and 512K, its accuracies are 52.60% and 40.10%, compared with 72.40% and 61.46% for direct reuse, gaps of 19.79 and 21.35 percentage points. Direct reuse therefore preserves long-context retrieval more efectively than the calibrated ridge mapping in the V4.1-Flash update.

Our starting point is whether, after a rolling model update, the new model can still interpret and retrieve information from the prefix state computed by its predecessor. GLM and DeepSeek-V4.1 show strong retrieval and reliable reuse over 512–32K prefixes, which may reflect their greater model capacity and stronger retrieval ability. DeepSeek-V4 becomes less accurate at longer prefixes, yet the cache from its stronger source checkpoint still improves the target’s retrieval performance, even as the harmful flip rate increases. Qwen ofers a complementary contrast: the source scores zero under the tested answer-prefix protocol because of its thinking-style output within the eight-token budget, but the updated target retrieves many values from the same source cache, reaching 80.21% accuracy at 512 tokens. This shows that a source’s answer format and the usefulness of its cached representations can difer. Together, these observations are consistent with a partially shared coordinate system for KV representations across checkpoints of the same family and architecture, with positional alignment where needed. The transferred cache can carry useful information and aspects of the source’s promptconditioned behavior, while the target’s own processing of the query and its readout determine how this state contributes to the answer.

## 3.2. Natural-Language QA Evaluation

Following the controlled retrieval evaluation, we assess answer quality and acceleration on longcontext natural-language QA tasks for the DeepSeek-V4-Flash → DeepSeek-V4-Flash-0731 update. On 585 examples, KV Reuse achieves median TTFT speedups of 5.08× on HotpotQA, 2.10× on 2WikiMQA, and 2.51× on MultiFieldQA-en; BABILong reaches 0.86× (Figure 4). Target/Reuse scores remain close on BABILong (59.0/58.7 accuracy) and MultiFieldQA-en (45.9/46.2 F1). HotpotQA F1 falls from 69.3 to 66.3, and 2WikiMQA from 61.7 to 54.7: paired changes are −3.0 points (95% CI: [−6.6, −0.1]) and −7.0 points ([−12.7, −1.5]), respectively.

![](images/b0b41fb835b3fa50c9ce5e541ff1f8712bba21fbd1c0c470d39b79cdf593db61.jpg)

![](images/eb04c5e07091f6f6811b42a3b2614051b9007e7e5049a34ee442d5eb3013c868.jpg)  
Figure 4 | Task quality (0–100) and prepared-cache TTFT for Flash → Flash-0731. Failed requests score zero. Speedups are median paired Target/Reuse TTFT ratios. TTFT includes cache loading and restoration; source prefill and model startup are excluded.

Reuse saves more prefill computation on longer prefixes. Median prefixes are 14.3K, 6.4K, and 7.7K tokens for HotpotQA, 2WikiMQA, and MultiFieldQA-en, respectively; BABILong’s shorter 1.7K prefix ofers less computation to avoid, so cache-access costs outweigh the saving. Quality depends on the task: MultiFieldQA-en retains its score while gaining speed, whereas both multi-hop QA tasks lose F1. The results therefore show substantial acceleration on longer inputs with a task-dependent quality tradeof. Appendix C gives full results and runtime controls.

## 3.3. KV Reuse in Earlier and Later Checkpoints

DeepSeek-V4-Flash-Base → DeepSeek-V4-Flash → DeepSeek-V4-Flash-0731 traces the progression from a pretrained base model through post-training to a later update. We use Flash as the common cache producer and inject its prefix KV states into both Base and Flash-0731 (Figure 5(a)). Each receiver uses its own weights to process the question and generate an answer. This compares transfer toward an earlier training stage and a later model version. Figure 5(b) shows native and reuse accuracy across 512–32K prefixes, with shading indicating the accuracy change for each receiver.

![](images/589f43a0f6ae9731beb0ce4fb45b13feaee68339e05ee6c06afbc5af392398be.jpg)

![](images/918d612805f1639b54263910e79a80a96971e78f9c4cfca44f8cf5d97a7b5439.jpg)  
Figure 5 | Flash KV reuse in Base and Flash-0731. (a) Model development and cache-transfer directions. (b) NIAH accuracy across 512–32K prefixes. Solid and dashed curves denote KV Reuse and Target. Shading marks their accuracy diference. The green curve shows Flash’s source accuracy.

Flash-produced caches improve retrieval for both receivers. Flash-0731 gains accuracy at every tested length, rising from 50.52% to 65.10% at 16K and from 25.52% to 31.25% at 32K. The improvement is more pronounced for Base. With its own cache, Base typically restates the question under the instruction-formatted prompt and scores 0% answer-prefix accuracy. With Flash’s KV states, it reaches 88.02%, 91.15%, and 81.77% at 512, 2K, and 4K. Accuracy remains 61.46% at 16K and 35.42% at 32K.

These results show that Flash-produced prefix states remain useful at both ends of the model’s development. They help Flash-0731 retain part of the source’s retrieval performance and help Base return the requested values. The prefix contains both the records and the instruction to return only the value, so the transferred state reflects Flash’s processing of content and response instructions together. Injecting these states can therefore change how a receiver uses the context and answers the question while its weights remain fixed. Appendix D.1 gives additional Base readout diagnostics.

## 3.4. Answer Content and Stopping Under KV Reuse

The preceding evaluations examine retrieval, QA quality and speed, and transfer across training stages. We now compare generated answers in the Flash → Flash-0731 update, using the separate retrieval cohort and all four benchmarks from Section 3.2. Table 2 groups response pairs by whether their recorded token sequences match in full, first difer after the initial token, or difer at the initial token. Sequence comparison includes EOS when present.

Retrieval responses mostly match in full (85.4%), with only 3.1% first difering later. In the main QA study, later changes occur in 4.7% of BABILong responses, 24.0% of HotpotQA responses, and 28.6% of 2WikiMQA responses. MultiFieldQA-en has the most frequent later changes (46.2%), despite 87.5% first-token agreement. Its mean F1 nevertheless remains close under reuse (45.92% versus 46.24%), while 2WikiMQA F1 falls from 61.68% to 54.65%. Generation agreement and answer quality therefore capture diferent efects of reuse. Appendix C.1 gives the score confidence intervals.

Table 2 | Generation agreement and answer scores for Flash → Flash-0731. All agreement rates and scores are percentages. Δscore is KV Reuse minus Target, in percentage points. Retrieval uses the separate diagnostic cohort, and QA uses the main study in Section 3.2. Agreement uses successful request pairs, and failed requests score zero. Acc. denotes accuracy.
<table><tr><td>Task</td><td>Metric</td><td></td><td></td><td>Full match Later change First-token change</td><td>Target score ↑</td><td>KV Reuse score ↑</td><td>∆score</td></tr><tr><td>Retrieval</td><td>Acc.</td><td>85.4</td><td>3.1</td><td>11.5</td><td>83.33</td><td>89.58</td><td>+6.25</td></tr><tr><td>BABILong</td><td>Acc.</td><td>66.3</td><td>4.7</td><td>29.0</td><td>59.00</td><td>58.67</td><td>-0.33</td></tr><tr><td>HotpotQA</td><td>F1</td><td>70.0</td><td>24.0</td><td>6.0</td><td>69.28</td><td>66.28</td><td>-2.99</td></tr><tr><td>2WikiMQA</td><td>F1</td><td>60.2</td><td>28.6</td><td>11.2</td><td>61.68</td><td>54.65</td><td>-7.03</td></tr><tr><td>MultiFieldQA-en</td><td>F1</td><td>41.2</td><td>46.2</td><td>12.5</td><td>45.92</td><td>46.24</td><td>+0.32</td></tr></table>

<table><tr><td>(a) Stopping changes</td><td colspan="2">(b) Answer content changes</td></tr><tr><td>Which band is still together: Matthew Good Band or Switchfoot?</td><td>Where did Thomas E. Noell&#x27;s father die?</td><td></td></tr><tr><td>Target Switchfoot [EOS]</td><td>Target</td><td>Washington, D.C.</td></tr><tr><td>KV Reuse Switchfoot is still together.</td><td>KV Reuse</td><td>St. Louis, Missouri</td></tr><tr><td>F1: 1.00 → 0.40</td><td></td><td>F1: 0.67 → 0.00</td></tr></table>

Figure 6 | Two diagnostic QA cases illustrate (a) stopping changes and (b) answer-content changes under KV Reuse. Scores are full-response F1.

Figure 6 uses two examples from the detailed QA diagnostics to illustrate how the answers change. For the band question, both runs produce Switchfoot. Target then stops, while KV Reuse adds is still together. The answer entity is preserved, but the additional wording lowers F1 from 1.00 to 0.40 against the one-word reference. For the question about where Thomas E. Noell’s father died, Target answers Washington, D.C., while KV Reuse answers St. Louis, Missouri, lowering F1 from 0.67 to 0.00. The context associates Washington with the father and St. Louis with the son, so this case changes the person–location relation used to answer the question.

KV Reuse can therefore afect both the selected answer and the decision to stop after it. These efects help explain why similar first-token predictions can lead to diferent complete responses and why an F1 loss can reflect either added wording or a changed answer. Section 4 examines predictive-distribution changes and how internal-state diferences afect retrieval-token decisions.

## 4. How KV Reuse Changes Model Predictions

This section examines how cache reuse changes the target’s computation and predictions. We first compare predictive distributions on retrieval and natural-text continuation for the Flash → Flash-0731 update. We then contrast internal-state and answer-margin changes in Flash-0731 and Flash-Base receivers. Finally, controlled interventions in Flash-0731 separate the efects of changed incoming representations and local caches.

## 4.1. Functional Divergence Varies Across Tasks

We use Functional Cache Divergence (FCD) to measure how much KV reuse changes the target model’s next-token predictions. FCD compares the complete probability distributions produced from the target’s own prefix cache and the source cache, holding the target model, subsequent input tokens,

and prediction position fixed. It reveals changes in the probabilities assigned to candidate tokens even when both runs select the same token, adding information beyond answer accuracy and output agreement.

Let $p _ { j }$ and $q _ { j }$ denote the two distributions at position $j ,$ with $m _ { j } = ( p _ { j } + q _ { j } ) / 2$ . We use Jensen–Shannon divergence with natural logarithms,

$$
\mathrm { F C D } _ { j } = \frac { 1 0 0 } { \ln 2 } J \mathrm { S } ( p _ { j } , q _ { j } ) = \frac { 5 0 } { \ln 2 } \left[ \mathrm { K L } ( p _ { j } | | m _ { j } ) + \mathrm { K L } ( q _ { j } | | m _ { j } ) \right] .\tag{2}
$$

FCD ranges from 0 (identical distributions) to 100. Larger values indicate greater deviation from target-native predictions, without assigning that change a beneficial or harmful outcome. We average positions within each query or document, queries within each history, and histories or documents equally.

We measure the first NIAH answer position on 192 queries from eight histories per prefix length, and 32 matched continuation positions from each of 64 FineWeb-Edu documents. Both tasks cover 512– 32K prefixes. Appendix A.4 gives the measurement protocol, and Appendix C.5 reports supplementary statistics.

Figure 7 shows a widening gap between retrieval and natural-text continuation. At 512 tokens, mean FCD is close to 3 for both NIAH (2.87) and FineWeb-Edu (3.04). By 32K, NIAH reaches 30.62, while FineWeb-Edu remains at 3.51. Across the six lengths, FineWeb-Edu stays within 3.04–3.64. The inset shows these small fluctuations, which are modest compared with the rise in NIAH divergence.

The contrast is consistent with how the two tasks use context. NIAH retrieves arbitrary key–value associations, so the answer must come from the cached records. Longer prefixes contain more competing records at the fixed density. Retrieval may therefore be especially sensitive to changes in how the cache

![](images/81ddb00cdff277dc95847af1c8157d237dc66d14c254af9b213cd59ad0f270f1.jpg)  
Figure 7 | Mean FCD for Flash → Flash-0731 on NIAH and FineWeb-Edu. Curves share a vertical scale, with a FineWeb-Edu zoom inset. Bands show cluster-bootstrap 95% intervals.

represents the queried key and its value. FineWeb-Edu continuation also has cues from nearby words, grammar, and semantic continuity. The target may use these cues to maintain similar predictions after the cache is replaced, ofering a plausible explanation for its lower, flatter FCD profile. The NIAH trend also accompanies the higher harmful flip rates at long prefixes in Section 3.1.

## 4.2. Internal State Changes and Retrieval Decisions

We examine how the internal changes induced by reuse relate to preserved or improved retrieval. Figure 8 compares Flash-0731 and Flash-Base as receivers of Flash-produced prefix caches. For selected 4K NIAH queries, we trace attention outputs, the residual stream after attention, and layer outputs while the complete query sufix is processed in one forward pass. Within each comparison, receiver weights and input tokens are identical. We measure representation diferences with normalized mean squared error (NMSE), and token competition with the gold answer token’s score margin over the strongest alternative:

(a) Flash → Flash-0731: NMSE  
![](images/9c98e9b0ccea91615362542725ec279b6ab0af2c649b304b988d0e34f23dd567.jpg)

(b) Flash → Flash-Base: NMSE  
![](images/8e0116e722a635181050412e34947a46fdc5f6e3439080b26d9cb141e440a25a.jpg)

(c) Flash → Flash-0731: Margin change  
![](images/02835506e21093472409cd94e2a16deb4c46d17129eaa1270d9021d1dbf0d19a.jpg)

(d) Flash → Flash-Base: Margin change  
![](images/5d924bfe50ec50407775f9439e45c7536828a3ad82fd3c9dcd6797ae1b4c4002.jpg)  
Figure 8 | Flash KV reuse in Flash-0731 and Flash-Base. (a, b) Layerwise NMSE over all query positions: medians and 25th–75th percentiles of selected cases. (c, d) Native gold margin versus its change under reuse. Dashed lines mark zero reuse gold margin, and colors show paired answer-prefix correctness. One point per margin panel lies outside the view; all queries enter summaries.

$$
\mathrm { N M S E } ( h _ { R } , h _ { T } ) = \frac { \| h _ { R } - h _ { T } \| _ { 2 } ^ { 2 } } { \| h _ { T } \| _ { 2 } ^ { 2 } } , \qquad m _ { \mathrm { g o l d } } ( z ) = z _ { \mathrm { g o l d } } - \underset { \nu \neq \mathrm { g o l d } } { \operatorname* { m a x } } z _ { \nu } .\tag{3}
$$

Here, $h _ { T }$ and $h _ { R }$ denote Target and KV Reuse representations at the same layer and site. Margin measurements use the first response-token prediction on all 192 paired queries in the 4K sweep for each receiver. Appendix A.3 gives the trace selection and cohort details.

Residual integration often moderates the relative diferences introduced by attention. In Figure 8(a, b), diferences become appreciable in the middle and later layers, with attention-output NMSE fluctuating more sharply than the post-attention residual and layer-output curves. For Flash-0731 at layer 37, median attention-output NMSE is 0.52, compared with 0.27 after residual integration and 0.28 at the layer output. For Flash-Base at layer 22, the corresponding values are 0.90, 0.56, and 0.53. These lower values describe smaller deviations relative to each site’s native representation energy. The residual stream moderates this relative disruption at many layers while carrying persistent diferences into later computation.

Flash-0731 preserves many correct decisions despite reduced answer margins. Figure 8(c) plots $\Delta m = m _ { \mathrm { g o l d } } ( z _ { R } ) - m _ { \mathrm { g o l d } } ( z _ { T } )$ against the native gold margin. Above the dashed threshold $( y = - x )$ , the gold token has a positive margin under reuse. Many native successes lose some margin but retain enough of their initial lead to remain correct. The median margin change is −1.00 logit, yet reuse repairs 14 answers and loses six, raising accuracy from 87.50% to 91.67%. Both the initial lead and the direction of the score change help explain which decisions survive.

![](images/62a7e859a6fafb1abe4ae2345f7daaed460586a2fcfc97eed8da4c867a6bb903.jpg)

![](images/a39f06d548191b13587108b3db2f89cd588e8fd011356f784d27cf1fde100753.jpg)  
Figure 9 | Indexer selection under Flash → Flash-Base reuse. (a) Mean top-� overlap with Target across layers. (b) Layer-band means after correcting for uniform-random overlap. Measurements use the first response prediction on four selected retrieval-recovery cases per length. Layer indices are zero-based.

Flash-Base gains access to cached answers despite substantial state changes. In Figure 8(d), every query gains gold margin, with a median increase of 11.77 logits, while answer-prefix accuracy rises from 0 to 81.77%. Flash-Base often emits whitespace or formatting tokens before the value, so a correct complete answer can still have a negative gold margin at the first response position. The margin shift shows that reuse promotes the answer token, and the complete responses confirm the retrieval gain.

Retrieval recovery also occurs when the receiver accesses a diferent set of cached entries. The Indexer selects compressed entries for sparse attention. Figure 9 compares its selections under Target and KV Reuse on four Flash-to-Base recovery cases at each length from 4K to 32K. Mean top-� overlap falls from 81.7% to 55.4%, with greater diferences in deeper layers. The trend persists after accounting for the overlap expected from uniform random selection. Adjusted overlap decreases from 86.4% to 76.3% in early layers and from 50.0% to 37.1% in late layers. Yet all these cases answer correctly under reuse after failing with the native cache. Successful retrieval can therefore tolerate substantial changes in which cached entries are selected. Appendix A.3 gives the selection metric and sampling details.

Flash-0731 preserves many decisions through suficient answer margins, while Flash-Base recovers retrieval despite changes in both internal states and Indexer selections. The Base recovery supports the interpretation that useful task state formed by the source remains readable with the receiver’s weights. Representation fidelity and retrieval utility are distinct aspects of compatibility.

## 4.3. Changes Propagate Across Layers

We next ask where the internal changes come from, using the separate 96-query Flash-update cohort. Each layer receives the previous layers’ computed representation and combines it with its own prefix KV cache. A diference observed at a later layer can therefore arise from that layer’s cache or from an input representation already changed by earlier layers. Here, “input” means the hidden representation entering the layer, after normalization. The original question tokens remain identical.

We separate these two sources by evaluating target attention as �(�, �) at layers 27, 37, and 42, where � is the incoming representation and � is the prefix cache. Target weights and token positions stay fixed. We compare four combinations: the Target input and cache, the Target input with the source cache, the reuse input with the Target cache, and the reuse input with the source cache.

Writing $X _ { T }$ and $X _ { R }$ for the two inputs and $C _ { T }$ and $C _ { S }$ for the two caches,

$$
\begin{array} { r } { A _ { T T } = F ( X _ { T } , C _ { T } ) , \quad A _ { T S } = F ( X _ { T } , C _ { S } ) , } \\ { A _ { R T } = F ( X _ { R } , C _ { T } ) , \quad A _ { R S } = F ( X _ { R } , C _ { S } ) . } \end{array}\tag{4}
$$

All output diferences use $A _ { T T }$ as the reference. Sufix states and index selections are recomputed for each chosen input.

Replacing the incoming representation produces larger changes than replacing only the local cache at all three inspected layers (Figure 10). At layer 37, the median output difference is 0.019 when replacing the local KV cache, 0.473 when replacing the incoming hidden state, and 0.530 when both are replaced. Much of the diference at this layer is therefore already present in its incoming representation. The source cache initiates the change, and interactions with earlier layers carry it into later computation. Inspecting a later layer’s cache alone misses this propagated efect.

This also helps explain why restoring caches in later layers may not recover the original answer. In four selected retrieval losses, replacing attention outputs in layers 36–42 restores all four answers, whereas replacing those layers’ prefix caches restores one. The input representation can still carry earlier changes even after a local cache is restored. These two interventions act

![](images/0c735d7494c2eadca3d0d310bc125ea9d20b077332420f62b4ce1e4c17654688.jpg)  
Figure 10 | Replacing the local KV cache, the incoming hidden state, or both on the same retrieval cohort. The cache is replaced with the source cache, and the hidden state with the target’s representation after earlier layers have reused source caches. Bars show median attention-output NMSE relative to Target. These comparisons do not add up.

at diferent points and for diferent durations, so the results illustrate the propagation efect rather than define a cache-repair method. Appendix D.2 gives the full replacement results.

## 5. Related Work

KV communication among agents. Multi-agent systems can exchange cached representations to avoid repeatedly processing shared context. KVComm (Shi et al., 2026b) selects informative layers’ KV pairs for communication. DroidSpeak (Liu et al., 2024) shares caches between models with the same architecture and selectively recomputes sensitive layers. HeteroFold (Yun et al., 2026) extends prefill-free transfer to diferent model families by aligning model structures, mapping caches, and calibrating receiver behavior. These systems establish KV states as a communication medium. We examine what happens when an updated model directly consumes its predecessor’s cache.

Cache mapping and alternative reusable states. Heo et al. (2026) fit closed-form ridge maps between selected layers of models in the same family. CacheBridge (Qu et al., 2026) combines head matching with attention-sensitive calibration to improve transfer quality and reduce mapping cost. RaReCache (Sarkar et al., 2026) selectively recomputes positions where mapped states are unreliable, enabling transfer across larger size gaps. A universal context-reuse layer (Li et al., 2026) studies translation across diferences in scale, architecture, and tokenization. MapFlow (Jin et al., 2026) instead reuses attention maps through translation and partial recomputation. Our main studies analyze direct source-cache transfer; Table 1 additionally compares an adapted closed-form ridge mapper for the Flash update.

Prefix reuse and checkpoint compatibility. SGLang (Zheng et al., 2024) reuses shared-prefix caches, and Prompt Cache (Gim et al., 2024) manages reusable prompt modules. CacheBlend (Yao et al., 2025) uses selective recomputation to combine separately cached chunks. KVShareArena (Shi et al., 2026a) benchmarks reuse across preceding contexts and model checkpoints, including version-toversion handofs, and compares quality, computation, memory, and latency. Our study complements this evaluation by holding input tokens fixed and connecting answer quality and generation fidelity to internal-state drift, token competition, and stopping.

Compressed caches and positional sensitivity. The DeepSeek-V4 report (DeepSeek-AI, 2026a) describes hybrid compressed attention, while DeepSeek-V4.1-Flash (DeepSeek-AI, 2026b) further reduces cache storage through compression and sharing. Periodic Weak Spots (Zhu et al., 2026) shows that retrieval can vary with a token’s position relative to compression-window boundaries. This motivates our phase-balanced NIAH queries and reinforces the need to distinguish aggregate retrieval accuracy from position-dependent behavior.

Post-training and model iteration. LIMA (Zhou et al., 2023) suggests that instruction tuning can shape response behavior while relying on knowledge acquired during pretraining. The GLM-5 report (GLM-5-Team, 2026) describes large-scale post-training, and the GLM-5.3 model card (Z.ai, 2026) attributes its gains over GLM-5.2 to post-training on the same base model. DeepSeek’s Flash-0731 release notes (DeepSeek-AI, 2026c) likewise specify unchanged architecture and size with renewed post-training. The separate Flash-Base checkpoint (DeepSeek-AI, 2026d) provides our instruction-training contrast. These releases motivate studying cached knowledge and promptconditioned answer behavior together.

Long-context evaluation. BABILong (Kuratov et al., 2024) embeds reasoning tasks in long contexts. LongBench (Bai et al., 2024a) combines multiple long-context tasks, including MultiFieldQA, while HotpotQA (Yang et al., 2018) and 2WikiMQA (Ho et al., 2020) test multi-hop question answering. LongBench v2 (Bai et al., 2024b) extends evaluation toward realistic, demanding contexts, and Multi-News (Fabbri et al., 2019) supplies multi-document summarization. FineWeb-Edu (Penedo et al., 2024) provides the natural documents used in our matched-continuation FCD evaluation. Our main comparisons use BABILong and LongBench v1 tasks alongside controlled retrieval.

Representation comparison and decision analysis. Centered kernel alignment (Kornblith et al., 2019) compares representations under chosen invariances. Our NMSE measures diferences encountered in paired target executions, and logit margins relate those diferences to token selection. Causal tracing (Meng et al., 2022) and activation-patching studies (Zhang and Nanda, 2024) motivate our interventions on incoming layer inputs, prefix caches, and attention outputs. Comparing token scores at the first retrieval-answer position connects internal diferences to changes in the selected token.

Inference implementation. Transformers (Wolf et al., 2020) and PyTorch (Paszke et al., 2019) support the detailed activation and generation analyses, while the serving evaluation uses SGLang. These execution paths have diferent cache-restoration and generation controls. We therefore report their settings separately and use same-version restoration to measure changes introduced by the handof itself.

## 6. Conclusion and Discussion

We studied whether DeepSeek-V4-Flash-0731 can continue from a cache produced by DeepSeek-V4- Flash. Reuse retains many retrieval answers despite substantial diferences in attention outputs and final representations. With the source cache prepared, it reduces time to first token on the tested benchmarks with longer prefixes, while task scores vary by benchmark. Natural question answering shows why answer quality alone is incomplete: the first token can agree with target-native inference even when later content or stopping changes. The Flash-to-Base retrieval contrast further shows that a cache carries prompt-conditioned state: it contains both the records and the instruction governing how to answer. Across these experiments, task quality, generation fidelity, and state similarity answer diferent questions about compatibility.

The internal analysis connects persistent state drift to decisions through token-score gaps. Many answers survive because the native winner retains enough advantage over competitors; a smaller shift can flip an answer when its original gap is narrow. At the three inspected layers, changes arriving through the input perturb attention more than changing only the local prefix cache. Earlier interactions with reused caches can create that changed input, so drift accumulates along the target’s continuation rather than remaining confined to a single layer’s stored state.

Across the four retrieval updates, GLM and V4.1-Flash preserve correct answers most reliably, V4-Flash gains aggregate accuracy, and Qwen loses accuracy under reuse. The V4.1-Flash extension retains 61.46% accuracy at 512K. Functional divergence reveals a further task contrast: prediction changes grow with prefix length on retrieval and stay low and stable on natural-text continuation. Future work can extend these comparisons to more tasks and decoding settings, predict changes that afect token decisions before reuse, and separate the contributions of cached facts and response instructions.

## References

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119–3137. Association for Computational Linguistics, 2024a. doi: 10.18653/v1/2024.acl-long.172. URL https:// aclanthology.org/2024.acl-long.172/.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. arXiv preprint arXiv:2412.15204, 2024b. URL https://arxiv.org/abs/2412.15204.

DeepSeek-AI. DeepSeek-V4: Towards highly eficient million-token context intelligence, 2026a. URL https://arxiv.org/abs/2606.19348.

DeepSeek-AI. DeepSeek-V4.1-Flash: Pushing the limits of KV cache compression. Technical report, 2026b. URL https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/ main/DeepSeek\_V41\_Tech\_Report.pdf.

DeepSeek-AI. DeepSeek-V4-Flash update. DeepSeek API Docs, Change Log, July 2026c. URL https: //api-docs.deepseek.com/updates#date-2026-07-31. Release entries dated April 24 and July 31, 2026. Accessed October 26, 2026.

DeepSeek-AI. DeepSeek-V4-Flash-Base. Hugging Face model repository, 2026d. URL https:// huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base. Accessed October 23, 2026.

Alexander Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-News: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1074–1084. Association for Computational Linguistics, 2019. doi: 10.18653/v1/P19-1102. URL https://aclanthology. org/P19-1102/.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. Prompt cache: Modular attention reuse for low-latency inference. Proceedings of Machine Learning and Systems, 6:325–338, 2024.

GLM-5-Team. GLM-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/ abs/2602.15763.

Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti, and Bita Darvish Rouhani. Cross-model KV cache transfer in LLM families: A closed-form linear mapping for prefill reuse. arXiv preprint arXiv:2608.03893, 2026. URL https://arxiv.org/abs/2608.03893.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625. International Committee on Computational Linguistics, 2020. doi: 10.18653/v1/2020.coling-main.580. URL https: //aclanthology.org/2020.coling-main.580/.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Zhan Jin, Yuze Luo, Bin Cui, and Xupeng Miao. Accelerating cross-model llm inference via attention map reuse. arXiv preprint arXiv:2610.14688, 2026. URL https://arxiv.org/abs/2610. 14688.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geofrey Hinton. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3519–3529. PMLR, 2019. URL https://proceedings.mlr.press/v97/kornblith19a.html.

Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. BABILong: Testing the limits of LLMs with long context reasoning-in-a-haystack. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://arxiv.org/ abs/2406.10149.

Yi Li, Dongming Jiang, Yi Zhao, and Bingzhe Li. A universal context-reuse layer for cross-model kv sharing. arXiv preprint arXiv:2608.30963, 2026. URL https://arxiv.org/abs/2608.30963.

Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, Madan Musuvathi, and Esha Choukse. Droidspeak: KV cache sharing for cross-LLM communication and multi-LLM serving. arXiv preprint arXiv:2411.02820, 2024. URL https://arxiv.org/abs/2411.02820.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372, 2022. URL https://arxiv.org/abs/2202.05262.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https: //arxiv.org/abs/1912.01703.

Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Rafel, Leandro von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024. URL https://arxiv.org/abs/ 2406.17557.

Xingyu Qu, Siyuan Lu, Zhiyu Chen, Sheng Wang, and Tao Lin. CacheBridge: Eficient cross-model KV cache transfer, 2026. URL https://arxiv.org/abs/2609.00891.

Sreetama Sarkar, Saptarshi Mitra, Sitao Huang, Souvik Kundu, and Peter A Beerel. Rarecache: Bridging the gap in cross-model kv cache reuse via rank disagreement-based selective recomputation. arXiv preprint arXiv:2610.11358, 2026. URL https://arxiv.org/abs/2610.11358.

Xi Shi, Mengxin Zheng, and Qian Lou. Kvsharearena: Kv-cache reuse across contexts and model checkpoints. arXiv preprint arXiv:2609.10266, 2026a. URL https://arxiv.org/abs/2609. 10266.

Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire, Jr., and Dejan Kostic. KVComm: Enabling eficient LLM communication through selective KV sharing. In International Conference on Learning Representations, 2026b. URL https://arxiv.org/abs/2510.03346.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45. Association for Computational Linguistics, 2020. doi: 10.18653/v1/ 2020.emnlp-demos.6. URL https://aclanthology.org/2020.emnlp-demos.6/.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380. Association for Computational Linguistics, 2018. doi: 10.18653/v1/D18-1259. URL https://aclanthology.org/D18-1259/.

Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. Cacheblend: Fast large language model serving for rag with cached knowledge fusion. In Proceedings of the twentieth European conference on computer systems, pages 94–109, 2025.

Vincent-Daniel Yun, Woosang Lim, Haneul Yoo, Sungjoo Yoo, Murali Annavaram, and Sai Praneeth Karimireddy. Prefill-free cross-family kv cache transfer for heterogeneous multi-agent llms. arXiv preprint arXiv:2609.32259, 2026. URL https://arxiv.org/abs/2609.32259.

Z.ai. GLM-5.3. Oficial model card, 2026. URL https://huggingface.co/zai-org/GLM-5.3. Describes the shared base model with GLM-5.2 and improvements from post-training. Accessed October 26, 2026.

Fred Zhang and Neel Nanda. Towards best practices of activation patching in language models: Metrics and methods. In The Twelfth International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2309.16042.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Eficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023.

Xingyu Zhu, Pu Yi, Ziheng Cheng, Ang Lv, Jing Liu, Lexing Ying, Yiyuan Ma, and Xin Dong. Periodic weak spots: Phase sensitivity from chunked kv-cache compression. arXiv preprint arXiv:2609.36322, 2026. URL https://arxiv.org/abs/2609.36322.

## Appendix Contents

A Tasks, Datasets, and Evaluation Protocols 20   
A.1 Retrieval Compatibility 20   
A.2 Task Quality and Latency 22   
A.3 Behavioral and Mechanistic Analysis 22   
A.4 Functional Cache Divergence 23   
B Cache Handof and Runtime Controls 24   
B.1 Composite Cache and Page Validation 24   
B.2 Runtime Matrix 24   
C Extended Results 25   
C.1 Complete Benchmark Results 25   
C.2 Length-Stratified Results 26   
C.3 Same-Version Controls and Sensitivity Checks 26   
C.4 Output Divergence 26   
C.5 Functional Divergence by Retrieval Outcome 27   
D NIAH Readout and Mechanism Details 28   
D.1 Flash-to-Base Readout Diagnostics 28   
D.2 Crossed Input–Cache Intervention 29   
D.3 Matched Perturbations and Stopping 29

## A. Tasks, Datasets, and Evaluation Protocols

This appendix follows the four evaluations in Section 2.2, describing their data construction, sample selection, and evaluation settings. Prefix lengths count the tokenized history and exclude the question or continuation sufix. Runtime and cache-handof controls are given in Appendix B.

## A.1. Retrieval Compatibility

The NIAH sweeps use eight independent histories at each of six prefix lengths: 512, 2K, 4K, 8K, 16K, and 32K tokens. Each history supplies 24 queries targeting records near 10%, 50%, and 90% of the prefix, balanced across eight placement phases. This yields 192 queries per length and 1,152 per model pair. Supplementary near-boundary queries are excluded from the reported retrieval results.

Records are interleaved with fixed filler text at approximately one record per 16 prefix tokens. The 512-, 2K-, and 4K-token histories contain 32, 128, and 256 records, respectively. Every key and value is checked to encode as a single token, remain unchanged at the tokenizer boundary in the key:value record, and remain distinct from the instruction and filler vocabulary. Construction uses each model pair’s tokenizer. Paired configurations receive identical token IDs.

For Flash, the prefix begins with “Read these records carefully. Every key has exactly one value. Use the records to answer the question. Return only the value.” The query sufix is “What is the value for key:KEY? Answer with only the value,” followed by the oficial assistant-generation prefix. GLM uses a plain completion prompt ending in “Question: What is the value for key:KEY? Answer:”. Pro, V4.1, and Qwen use their model-specific prompt encodings. Table 3 shows one actual 512-token Flash example.

Table 3 | A Flash NIAH example. Prefix text is abbreviated; chat-template tokens and record separators are omitted for readability.  
Component Example   
Prefix � (excerpt) Read these records carefully. Every key has exactly one value.   
Use the records to answer the question. Return only the value.   
The morning light fell across the quiet garden.   
... yet:vector ... ynt:izio ... IELD:Popular ...   
Query � What is the value for key:yet? Answer with only the value.   
Gold answer vector

All retrieval runs use greedy decoding, with a maximum of eight output tokens for GLM, V4-Flash, and Qwen, and one for Pro and V4.1-Flash, including the long-context extension. Answer-prefix accuracy counts a response as correct when its whitespace-trimmed text begins with the requested value. Failed requests score zero. For Qwen, Source’s thinking-format output can fill the eight-token budget before an answer is returned. The zero Source score therefore describes the tested response protocol.

The Flash-update and Flash-to-Base short-prefix analyses use the 512-, 2K-, and 4K-token subset of the six-length Flash sweep, comprising 576 queries from 24 histories. The Pro evaluation uses the same length and sample-count design. Appendix D.1 reports the Flash-to-Base readout statistics.

The V4.1-Flash extension adds 64K, 128K, 256K, and 512K prefixes using the same history count, record density, and depth-query design. Its 768 added queries bring the full sweep in Figure 3(b) to 1,920 queries. Keys and values remain globally disjoint through 256K. At 512K, the required 65,536 distinct terms exceed the available 53,162-term pool, so a token may serve as a key in one record and a value in another. Keys remain unique, values remain unique, and no record binds a key to itself. Runtime diferences between the short and long sweeps are described in Appendix B.2.

Closed-form mapper comparison. The Flash block in Table 1 evaluates an adaptation of the centered ridge mapping and source-layer selection of Heo et al. (2026), rather than an authorprovided implementation. We calibrate on 500 independent FineWeb-Edu documents, using 1,024 prefix tokens per document with both models. A separate set of 32 documents supplies 16 matched continuation positions per document to choose $k \in \{ 1 , 2 , 4 \}$ by the lowest mean full-vocabulary FCD; this selects $k = 4$ . These documents are disjoint from the report’s earlier natural-text evaluation sets, and no NIAH answers are used for fitting or selection.

For each primary target-cache component, source layers are ranked by ridge prediction $R ^ { 2 }$ within compatible component types, widths, and compression rates. The selected � source vectors are concatenated into �, paired with target vectors ${ \cal Y } ,$ , and centered to obtain $X _ { c }$ and $Y _ { c } .$ We fit the full afine map

$$
\boldsymbol { W } = ( \boldsymbol { X } _ { c } ^ { \top } \boldsymbol { X } _ { c } + \lambda \boldsymbol { I } ) ^ { - 1 } \boldsymbol { X } _ { c } ^ { \top } \boldsymbol { Y } _ { c } , \qquad \boldsymbol { M } ( \boldsymbol { x } ) = ( \boldsymbol { x } - \bar { \boldsymbol { x } } ) \boldsymbol { W } + \bar { \boldsymbol { y } } ,\tag{5}
$$

using $\lambda = 0 . 0 1$ for the summed squared-error objective and streamed float64 suficient statistics, without a diagonal approximation. Primary uncompressed KV rows are sampled every four tokens and auxiliary projection rows every 16; source-layer ranking uses at most 2,048 paired rows per component.

DeepSeek’s shared K/V vectors are mapped jointly in content coordinates before the trailing 64- dimensional interleaved RoPE rotation; target runtime frequencies, scales, and absolute positions are then applied. Mapping covers the active window, compressed, and indexer KV states throughout the target’s 43 layers, together with same-layer full-ridge maps for nonempty compressor/indexer bufer, gate, and overlap states. Cache counters and positions are preserved, and K/V sharing is restored. This complete-state adaptation is checked by pre-RoPE capture validation and identity transfers with exact cache and logit agreement at 512 and 1,031 tokens, including nonempty compression tails.

All four configurations are rerun on the same 1,152 fixed NIAH queries with identical token IDs, prefill chunks, and eight-token greedy decoding. The Flash baselines in Table 1 come from this matched run; earlier figures and diagnostic analyses retain their original measurement campaigns. Accuracy diferences use paired history-cluster bootstrap intervals with 10,000 replicates. The mapper comparison assesses retrieval correctness in the Transformers path and does not provide a servinglatency comparison.

For V4.1-Flash → Flash-1017, we fit a separate centered ridge mapper on 500 FineWeb-Edu documents with 1,024-token prefixes. Held-out FCD on 32 documents with 16 continuation positions each selects � = 4 from {1, 2, 4}, capped by the number of compatible source cache producers for each component. The fit uses $\lambda = 0 . 0 1$ and normalized pre-RoPE cache vectors. Mapped states are re-encoded in the native cache formats, compressor and indexer tails are mapped separately, and Engram token histories are preserved.

Figure 3(b) uses a matched rerun of all four configurations on the same 1,920 fixed queries over 512–512K prefixes, with 8,192-token prefill chunks and one-token greedy decoding. Its intervals use 10,000 paired history-cluster bootstrap draws. Figure 1 and Figure 3(a) retain the original sweep, while panel (b) uses the baselines measured alongside Ridge Mapper.

## A.2. Task Quality and Latency

The 585-sample study evaluates Flash → Flash-0731 on fixed snapshots of BABILong (Kuratov et al., 2024) and LongBench v1 (Bai et al., 2024a). Samples are selected independently of model outcomes. Table 4 gives the source snapshots, eligible pools, and evaluated sample counts.

Table 4 | Dataset provenance and main-study selection. “Eligible” counts rows surviving input and tokenizer checks; “independent” counts distinct stories or source documents. BABILong contains 20 examples for every QA-type and length pair.
<table><tr><td>Benchmark</td><td>Source snapshot</td><td>Eligible pool</td><td>Main n Length strata</td><td></td></tr><tr><td>BABILong</td><td>QA1–QA5, revision ee0d5887</td><td>300 independent</td><td>300</td><td>1K / 2K / 4K; 20 per QA-length pair</td></tr><tr><td>HotpotQA</td><td>LongBench v1, revision 5e628be4</td><td>199 rows; 199 documents</td><td></td><td>100 Four token-length quartiles</td></tr><tr><td>2WikiMQA</td><td>LongBench v1, revision 5e628be4</td><td>192 rows; 192 documents</td><td></td><td>100 Four token-length quartiles</td></tr><tr><td>MultiFieldQA-en</td><td>LongBench v1, revision 5e628be4</td><td>139 rows; 105 documents</td><td></td><td>85 Four token-length quartiles; 15-sample shortfall</td></tr></table>

BABILong selection balances QA1–QA5 and the three source lengths. The LongBench tasks use token-length quartiles. All 585 evaluated samples are distinct at the story or source-document level. BABILong uses accuracy, and the other three tasks use token F1. Each request uses greedy decoding with a 48-token output limit. Failed requests receive zero task score.

Table 5 reports actual prefix lengths after oficial chat encoding. Median question-sufix lengths are 11, 24, 22, and 19 tokens for BABILong, HotpotQA, 2WikiMQA, and MultiFieldQA-en, respectively. The benchmark-level TTFT results therefore compare workloads with diferent input-length distributions.

Table 5 | Tokenized prefix lengths in the 585-sample main study. Q1 and Q3 are the first and third quartiles. Question sufixes are excluded.
<table><tr><td>Benchmark</td><td>n</td><td>Min</td><td>Q1</td><td>Median</td><td>Q3</td><td>Max</td></tr><tr><td>BABILong</td><td>300</td><td>571</td><td>735</td><td>1,712</td><td>3,539</td><td>3,852</td></tr><tr><td>HotpotQA</td><td>100</td><td>2,142</td><td>11,054</td><td>14,320</td><td>15,892</td><td>16,507</td></tr><tr><td>2WikiMQA</td><td>100</td><td>2,580</td><td>5,041</td><td>6,369</td><td>8,530</td><td>15,783</td></tr><tr><td>MultiFieldQA-en</td><td>85</td><td>2,703</td><td>4,827</td><td>7,666</td><td>9,030</td><td>14,636</td></tr></table>

TTFT is measured with the source cache already prepared, from the target continuation request to the first non-empty generated token. It includes cache lookup, loading, restoration, and the target’s remaining computation. Source prefill, cache promotion, target startup, and model loading are excluded. Target uses a cold request without prefix-cache hits. Appendix C.1 gives the complete quality and latency results.

## A.3. Behavioral and Mechanistic Analysis

The Flash-update state traces, token-score comparisons, and interventions use independent 4K contexts. Table 6 summarizes the primary diagnostic cohorts. The main and confirmation contexts are distinct. All runs use greedy decoding, with an eight-token output limit for retrieval and a 256-token limit for QA to support content and stopping analysis.

The 89 QA examples comprise 39 HotpotQA and 50 2WikiMQA examples. Each task contributes

Table 6 | Primary diagnostic cohorts. Retrieval counts are queries and QA counts are examples.
<table><tr><td>Cohort</td><td>n Independent unit</td></tr><tr><td>Main retrieval diagnostics</td><td>96 32 contexts</td></tr><tr><td>Retrieval confirmation</td><td>48 16 contexts</td></tr><tr><td>Natural-QA diagnostics</td><td>89 89 examples</td></tr></table>

20 random examples selected before the new runs from its original 100-example pool. Other cases examine earlier loss, gain, and stability patterns. Frequency summaries use only the random subsets.

Exploratory retrieval analyses use 33 unique queries: 24 position-selected queries and 12 historical cases, with three overlaps. The four selected retrieval losses in Appendix D.2 come from the main and exploratory cohorts, with overlapping cases counted once.

Figure 8 uses additional 4K NIAH comparisons. The Flash-0731 traces contain 12 behavior-selected queries from seven histories: four improved, four degraded, and four correct in both runs. The Flash-Base traces contain four improved queries from three histories. NMSE aggregates all query positions and features at each measurement site. These selected traces illustrate internal trajectories rather than estimate population frequencies. The margin panels instead include the same 192 depth queries from eight histories for each receiver, with identical prefixes, question sufixes, and gold answers across receivers. All 192 queries enter the outcome counts and margin summaries, including the single extreme query outside each displayed margin range. Appendix B.2 specifies their prefill settings.

Figure 9 uses four Flash-to-Base retrieval-recovery queries at each of 4K, 8K, 16K, and 32K, from three, four, four, and three histories, respectively. The 4K queries are the same as the Flash-Base traces in Figure 8. Each length uses its own selected queries, all incorrect under Target and correct under KV Reuse by answer-prefix matching. At the final query position, we compare the actual selected sets at the 21 Indexer layers, numbered 2–42 in steps of two. Overlap is $r = | S _ { T } \cap S _ { R } | / | S _ { T } |$ , where $S _ { T }$ and $S _ { R }$ are the Target and KV Reuse selections. With a budget of $k = 5 1 2$ and � valid candidates, the uniform-random reference is $b = \operatorname* { m i n } ( k , N ) / N _ { \mathrm { \ell } }$ , and adjusted overlap is $( r - b ) / ( 1 - b )$ . Heatmap cells average overlap over the four queries. Layer-band curves average adjusted overlap over queries and the indicated layers. These summaries describe the selected recovery cases.

## A.4. Functional Cache Divergence

FCD compares full-vocabulary, temperature-one next-token distributions in FP32 using the definition in Equation 2. The NIAH evaluation uses the same 1,152 queries as the six-length Flash-update sweep. Each comparison is made immediately after the common question sufix, before supplying any answer token. It yields one FCD value per query, regardless of the eventual response length.

FineWeb-Edu (Penedo et al., 2024) uses sample-10BT at revision 87f09149. We select the first 64 unique documents with at least 100,000 characters and enough tokens for a 32K prefix plus the continuation. The same documents are used at all six lengths. After caching � tokens, both target runs consume the same original-document tokens $t _ { 1 } , \ldots , t _ { 3 2 }$ and predict $t _ { 2 } , \ldots , t _ { 3 3 } .$ , yielding 32 paired positions per document and 12,288 over all six lengths. Identical supplied tokens keep subsequent inputs aligned when predictions difer.

NIAH means weight histories equally after averaging their query values. FineWeb-Edu means weight documents equally after averaging their prediction positions. Confidence intervals use 10,000 historyor document-cluster bootstrap resamples. Supplementary statistics are reported in Appendix C.5.

## B. Cache Handof and Runtime Controls

## B.1. Composite Cache and Page Validation

Both handofs transfer the architecture-specific state needed for continuation: the KV anchor, slidingwindow attention state (SWA), compression state C4 and its indexer state, compression state C128, and the two associated state bundles. The detailed retrieval, generation, and intervention analyses transfer this complete prefix state at the boundary between history � and question �. The serving benchmark instead transfers complete pages from file-backed storage, after which the target processes any unaligned prefix tail before �.

The SGLang (Zheng et al., 2024) experiments use 256-token storage pages and 4,096-token chunked prefill. For a prefix of � tokens, the expected transferable extent is

$$
P _ { s t o r a g e } = 2 5 6 \lfloor P / 2 5 6 \rfloor .
$$

Before continuation, the harness verifies file presence, size, and SHA-256 for every required page component. The native request must have zero cache hits; a direct request must hit the expected storage extent with zero device/host hits. The same checks are repeated after the run. A partial or non-storage hit is a structural failure rather than a successful direct result.

## B.2. Runtime Matrix

The length sweeps and detailed diagnostics use diferent prefill paths. Flash prefill is chunked at 512 tokens for the short sweep and 1,024 for 8K–32K, while Qwen uses 512-token chunks. The GLM sweep combines 512–4K runs with 8K–32K runs using 1,024-token prefill chunks and a diferent cache-storage layout. The serving benchmark transfers complete storage pages, whereas the formal 96-query diagnostics use full-prefix prefill. For Figure 8, the Flash-0731 comparison uses 512-token prefill chunks and the Flash-Base comparison uses 1,024-token chunks. Each paired run uses the same chunk size, and the complete question sufix is processed in one forward pass after prefix-state transfer. Table 7 summarizes the runtime and handof settings. Generation settings are given in Appendix A.

The Flash-to-Base Indexer comparisons in Figure 9 use 1,024-token prefill chunks throughout. The 4K run uses PyTorch 2.10.0 and the 8K–32K runs use PyTorch 2.11.0, all with Transformers 5.20.1 and CUDA 12.9. Target and KV Reuse share execution settings within each length.

Paired comparisons share execution settings within each experiment. We compute 95% confidence intervals by resampling independent histories for retrieval and source examples for QA, using 10,000 bootstrap replicates for the length sweeps and 2,000 for the serving and detailed analyses. FCD uses 512-token prefill chunks through 4K and 1,024-token chunks at longer lengths for NIAH, and 1,024-token chunks throughout for natural text. Its aggregation and confidence intervals are described in Appendix A.4.

For the V4.1-Flash extension, 64K–512K prefixes use 8,192-token continuation-chunk prefill with fixed projection blocks, complete historical compressed states, and absolute positions. The earlier 512–32K results use the prior full-prefill path. Native and reuse runs share execution settings within each length. At 256K and 512K, the common query-sufix prefix is cached separately for each history and configuration; its first query is checked against the complete sufix path for identical logits and output tokens. Cross-length trends therefore describe the recorded settings, including the 512K construction diference, rather than isolate length as the only changing factor.

Table 7 | Runtime and cache-handof settings by experiment family.
<table><tr><td>Experiment family</td><td>Execution</td><td>Cache and prefill settings</td></tr><tr><td>585-sample main study</td><td>SGLang commit 1fdd6c89; eight H100 GPUs; TP=8; file HiCache</td><td>4,096-token chunks; 256-token pages; write_through</td></tr><tr><td>NIAH readout</td><td>Transformers and PyTorch on eight H100 GPUs</td><td>Complete prefix-state transfer</td></tr><tr><td>Flash ridge-mapper comparison</td><td>Transformers 5.20.1; PyTorch 2.11.0+cu129; eight H100 GPUs</td><td>Shared FP8 Triton path for all four configurations; 512-token chunks through</td></tr><tr><td>GLM additional-update NIAH</td><td>SGLang; TP=2, PP=8 across two GPU nodes</td><td>4K, 1,024 at 8K–32K File-backed cache handoff</td></tr><tr><td>Pro additional-update NIAH</td><td>PyTorch reference runtime; separate source and target workers</td><td>Full prefill; complete prefix-cache handoff</td></tr><tr><td>V4.1 additional-update NIAH</td><td>Official reference runtime; TP=8</td><td>Complete prefix-state restoration</td></tr><tr><td>Activation and stopping analyses</td><td>Transformers and PyTorch on eight H100 GPUs</td><td>Full-prefix prefill for formal cohorts; chunked prefill for Figure 8</td></tr></table>

## C. Extended Results

## C.1. Complete Benchmark Results

Table 8 | Task quality on the 585-sample benchmark study. Scores are percentages. Δ is direct minus target-native, in percentage points; intervals use 2,000 document/story-cluster bootstrap resamples. Score ↓ /↑ counts paired score decreases and increases. Flips report native-success→direct-failure and native-failure→direct-success. Failures are native/direct and score zero.
<table><tr><td>Benchmark</td><td></td><td>n Metric</td><td>Target</td><td></td><td></td><td>Direct Source Δ [95% CI]</td><td>Score ↓/↑</td><td>Flips</td><td>Fail.</td></tr><tr><td>BABILong</td><td>300 Acc.</td><td></td><td>59.0</td><td>58.7</td><td>61.0</td><td>-0.3 [-4.3, +3.7]</td><td>20/19</td><td>20/19</td><td>0/0</td></tr><tr><td>HotpotQA</td><td></td><td>100 Token F1</td><td>69.3</td><td>66.3</td><td></td><td>69.4 -3.0[-6.6, -0.1]</td><td>8/6</td><td>5/0</td><td>0/0</td></tr><tr><td>2WikiMQA</td><td></td><td>100 Token F1</td><td>61.7</td><td>54.7</td><td></td><td>58.9 -7.0[–12.7,−1.5]</td><td>19/3</td><td>13/3</td><td>0/2</td></tr><tr><td>MultiFieldQA-en</td><td></td><td>85 Token F1</td><td>45.9</td><td>46.2</td><td></td><td>49.8 +0.3 [-3.1, +3.7]</td><td>20/19</td><td>1/2</td><td>5/5</td></tr></table>

The source-native score is descriptive context for the representation producer. It is not a targetquality baseline, and no aggregate score is formed across benchmarks because their metrics and score distributions difer.

Table 9 | Prepared-cache TTFT. Only valid target/direct pairs enter the sample count and percentiles. The speedup is computed per sample as target-native/direct TTFT and summarized by its median; it is not the ratio of the two marginal medians.
<table><tr><td>Benchmark</td><td>Pairs</td><td>Native p50/p95</td><td>Direct p50/p95</td><td>Speedup p50</td><td>Direct slower</td></tr><tr><td>BABILong</td><td>300/300</td><td>153.2/286.5</td><td>192.2/254.4</td><td>0.855×</td><td>201</td></tr><tr><td>HotpotQA</td><td>100/100</td><td>1,792.3/8,101.1</td><td>355.9/392.2</td><td>5.083×</td><td>1</td></tr><tr><td>2WikiMQA</td><td>98/100</td><td>543.5/1,712.7</td><td>267.3/524.2</td><td>2.097×</td><td>3</td></tr><tr><td>MultiFieldQA-en</td><td>80/85</td><td>759.1/1,480.8</td><td>248.2/762.1</td><td>2.512×</td><td>6</td></tr></table>

Latency follows the protocol in Appendix A.2. Each request was timed once, so the measurements describe these runs rather than service-level latency.

Table 10 | Quality and prepared-cache TTFT by actual input-length group. Scores and confidence intervals are percentages. BABILong groups are source-defined lengths; other groups are quartiles of tokenized prefixes.
<table><tr><td>Group</td><td>n</td><td>Fail.</td><td>Target/Direct</td><td>∆ [95% CI]</td><td>Score↓/↑</td><td>Pairs</td><td>Speedup</td></tr><tr><td>2Wiki q1</td><td>25</td><td>0/0</td><td>74.6/66.9</td><td>-7.8 [-24.6, +9.1]</td><td>7/2</td><td>25/25</td><td>1.80×</td></tr><tr><td>2Wiki q2</td><td>25</td><td>0/1</td><td>59.5/56.8</td><td>-2.7[–10.5, +4.3]</td><td>3/1</td><td>24/25</td><td>2.09×</td></tr><tr><td>2Wiki q3</td><td>25</td><td>0/0</td><td>56.2/47.5</td><td>-8.8 [-18.0, -1.3]</td><td>4/0</td><td>25/25</td><td>2.21×</td></tr><tr><td>2Wiki q4</td><td>25</td><td>0/1</td><td>56.4/47.5</td><td>-8.9[−19.2, -0.6]</td><td>5/0</td><td>24/25</td><td>4.52×</td></tr><tr><td>BABILong 1K</td><td>100</td><td>0/0</td><td>67.0/67.0</td><td>0.0 [-7.0, +8.0]</td><td>7/7</td><td>100/100</td><td>0.86×</td></tr><tr><td>BABILong 2K</td><td>100</td><td>0/0</td><td>57.0/60.0</td><td>+3.0[-3.0, +9.0]</td><td>4/7</td><td>100/100</td><td>0.79×</td></tr><tr><td>BABILong 4K</td><td>100</td><td>0/0</td><td>53.0/49.0</td><td>-4.0 [-11.0, +3.0]</td><td>9/5</td><td>100/100</td><td>1.23×</td></tr><tr><td>Hotpot q1</td><td>25</td><td>0/0</td><td>70.1/68.5</td><td>-1.6[-10.6, +6.0]</td><td>2/1</td><td>25/25</td><td>2.40×</td></tr><tr><td>Hotpot q2</td><td>25</td><td>0/0</td><td>68.9/63.3</td><td>-5.6[-14.1, +0.0]</td><td>3/1</td><td>25/25</td><td>5.55×</td></tr><tr><td>Hotpot q3</td><td>25</td><td>0/0</td><td>69.6/65.0</td><td>-4.6[−12.9, +0.5]]</td><td>2/2</td><td>25/25</td><td>4.88×</td></tr><tr><td>Hotpot q4</td><td>25</td><td>0/0</td><td>68.5/68.4</td><td>-0.1 [−1.2, +0.7]</td><td>1/2</td><td>25/25</td><td>10.51×</td></tr><tr><td>MultiField q1</td><td>22</td><td>4/4</td><td>43.5/43.8</td><td>+0.3 [−0.2, +1.0]</td><td>1/3</td><td>18/22</td><td>1.23×</td></tr><tr><td>MultiField q2</td><td>21</td><td>1/1</td><td>44.5/48.8</td><td>+4.3 [-5.9, +14.4]</td><td>4/9</td><td>20/21</td><td>2.26×</td></tr><tr><td>MultiField q3</td><td>21</td><td>0/0</td><td>44.3/46.9</td><td>+2.6[-3.1, +10.2]</td><td>6/3</td><td>21/21</td><td>3.54×</td></tr><tr><td>MultiField q4</td><td>21</td><td>0/0</td><td>51.5/45.6</td><td>-5.9[-12.3, -0.5]</td><td>9/4</td><td>21/21</td><td>4.11×</td></tr></table>

## C.2. Length-Stratified Results

These strata are descriptive. They were not designed to estimate a single causal slope of speedup against prefix length, and input data and serving overhead difer across benchmarks.

## C.3. Same-Version Controls and Sensitivity Checks

Table 11 | Target-self controls in the 585-sample study. “First” and “full” report first-token and complete-recorded-sequence agreement, respectively. Scores show mean self minus native and direct minus self gaps.
<table><tr><td>Benchmark</td><td>n</td><td>First match</td><td>Full match</td><td>Self-Native</td><td>Direct-Self</td></tr><tr><td>BABILong</td><td>300</td><td>265</td><td>255</td><td>-0.013</td><td>+0.010</td></tr><tr><td>HotpotQA</td><td>100</td><td>95</td><td>87</td><td>-0.010</td><td>-0.020</td></tr><tr><td>2WikiMQA</td><td>100</td><td>94</td><td>84</td><td>-0.013</td><td>-0.058</td></tr><tr><td>MultiFieldQA-en</td><td>85</td><td>82</td><td>52</td><td>-0.005</td><td>+0.008</td></tr></table>

The dedicated 129-query retrieval and 89-example natural-QA analysis paths match across native, restored-self, and fresh-process self. The 585-sample serving study shows the same-version diferences in Table 11, including complete-sequence agreement of 52/85 on MultiFieldQA-en. Its direct–native comparisons therefore combine cross-version efects with variation observed under the serving handof.

Seven observations in the original benchmark table failed a storage-tier validity condition. A post hoc retest started the target separately for each afected sample and restored copied, verified pages. It changed two rows as follows; the original table remains the primary result.

## C.4. Output Divergence

The preselected random QA subsets provide additional paired quality measurements for the detailed diagnostics discussed in Section 3.4. Table 13 reports the full-response F1 scores for 20 examples per task.

Table 12 | Post hoc cache-tier sensitivity retest. Scores and score changes use each task’s primary metric. The retest isolates seven rows with device/host cache interference and is not pooled with the original main result.
<table><tr><td>Benchmark</td><td>Valid pairs (original→retest)</td><td>Target/direct</td><td></td><td>∆ [95% CI]</td><td>Speedup</td></tr><tr><td>2WikiMQA</td><td>98/100 → 100/100</td><td>0.617/0.548</td><td></td><td>-0.069[-0.126, -0.013]</td><td>2.097×</td></tr><tr><td>MultiFieldQA-en</td><td>80/85 → 85/85</td><td>0.501/0.506</td><td></td><td>+0.005 [-0.036, +0.047]</td><td>2.390×</td></tr></table>

Table 13 | Full-response F1 on the preselected random QA subsets. Loss, unchanged, and gain count each example’s paired score change. Intervals use 2,000 bootstrap resamples of source examples.
<table><tr><td>Task (n = 20 each)</td><td>Target F1</td><td>KV Reuse F1</td><td>Loss / unchanged / gain</td><td>Change [95% CI]</td></tr><tr><td>HotpotQA</td><td>0.6553</td><td>0.6683</td><td>1/18 / 1</td><td>+0.0129 [-0.0088, +0.0476]</td></tr><tr><td>2WikiMQA</td><td>0.6382</td><td>0.5571</td><td>5 / 13 / 2</td><td>-0.0811 [-0.2131,+0.0435]</td></tr></table>

Table 14 | First-divergence analysis for the four cross-benchmark QA tasks. Only samples with successful native/direct pairs enter this table. “Later” means the first token matched and the recorded sequences still difered.
<table><tr><td>Benchmark</td><td>Pairs</td><td>Identical</td><td>First token</td><td>Later</td><td>Later score ↓ /= ↑</td><td>Later EM 1 →0</td></tr><tr><td>BABILong</td><td>300</td><td>199</td><td>87</td><td>14</td><td>0/12/2</td><td>0</td></tr><tr><td>HotpotQA</td><td>100</td><td>70</td><td>6</td><td>24</td><td>7/11/6</td><td>5</td></tr><tr><td>2WikiMQA</td><td>98</td><td>59</td><td>11</td><td>28</td><td>12/15/1</td><td>9</td></tr><tr><td>MultiFieldQA-en</td><td>80</td><td>33</td><td>10</td><td>37</td><td>16/6/15</td><td>0</td></tr></table>

## C.5. Functional Divergence by Retrieval Outcome

This subsection supplements the FCD comparison in Section 4.1. Data selection, matched prediction positions, and aggregation are described in Appendix A.4.

On FineWeb-Edu, mean reuse-minus-target negative log likelihood on the supplied next tokens is positive at every length, ranging from 0.036 to 0.082 nats per token. At 32K, mean FCD is 3.51 (95% CI: [3.11, 3.94]), with 86.62% top-1 agreement.

The NIAH FCD distribution is skewed at short prefixes. At 512 tokens, mean FCD is 2.87 (95% CI: [1.12, 5.11]), while median FCD is 0.035. At 16K, the mean reaches 24.11 ([20.05, 28.18]). At 32K, the mean is 30.62 ([25.71, 35.51]) and the median is 23.51. Target/Reuse top-1 agreement falls from 96.35% at 512 to 68.75% at 32K.

FCD measures deviation from target-native predictions without assigning a beneficial or harmful outcome. Table 15 groups the 16K and 32K retrieval measurements by the previously recorded answer-prefix outcomes on the same query IDs. At 16K, the mean FCD of the 36 improved answers is 66.42, exceeding the 44.34 mean for the eight degraded answers. At 32K, gains and losses both have high means, 63.23 and 66.15, compared with 9.99 among the 36 queries answered correctly by both configurations. Large functional changes can therefore accompany correction of a target error as well as loss of a correct answer. Both-incorrect outcomes also occur without the largest divergences: the 119 such queries at 32K have mean FCD 26.40, below the gain and loss groups.

The FCD run is a separate measurement campaign, not a replay of the full generation evaluation. Its first-token predictions difer from the frozen behavior run in 60 of 2,304 arm-level comparisons, all at 512–4K. The 16K/32K groups in Table 15 have no such first-token mismatches. Prefill and cache-restoration settings are given in Appendix B.2. This FCD campaign has no target-self restoration baseline, so a separate estimate of distributional variation introduced by restoration is unavailable. The exact self-restoration results of the full-prefix diagnostic experiments in Section 4.2 apply to those experiments, not automatically to this campaign.

Table 15 | FCD by paired retrieval outcome at 16K and 32K. Categories use the frozen answer-prefix evaluation; larger FCD denotes greater distributional change. Counts and conditional summaries describe the observed groups.
<table><tr><td>Prefix</td><td>Target → Reuse outcome</td><td>Queries</td><td>Mean FCD</td><td>Median FCD</td></tr><tr><td>16K</td><td>Correct → correct</td><td>89</td><td>4.56</td><td>1.04</td></tr><tr><td></td><td>Incorrect → correct</td><td>36</td><td>66.42</td><td>73.12</td></tr><tr><td></td><td>Correct → incorrect</td><td>8</td><td>44.34</td><td>33.09</td></tr><tr><td></td><td>Incorrect → incorrect</td><td>59</td><td>25.03</td><td>21.79</td></tr><tr><td>32K</td><td>Correct → correct</td><td>36</td><td>9.99</td><td>6.99</td></tr><tr><td></td><td>Incorrect → correct</td><td>24</td><td>63.23</td><td>65.31</td></tr><tr><td></td><td>Correct → incorrect</td><td>13</td><td>66.15</td><td>71.50</td></tr><tr><td></td><td>Incorrect → incorrect</td><td>119</td><td>26.40</td><td>21.95</td></tr></table>

These results show that functional divergence varies across tasks and can accompany both improved and degraded answers in the measured samples. They do not validate FCD as a predictor of harmful reuse on unseen inputs. Task outcomes and generation fidelity remain necessary alongside FCD; a single average divergence does not summarize which examples change or whether those changes are useful.

## D. NIAH Readout and Mechanism Details

## D.1. Flash-to-Base Readout Diagnostics

Answer-prefix accuracy permits leading whitespace and short formatting tokens. First-token correctness, strict exact match, gold-token rank, and top-10 membership separate content readout from response formatting.

Table 16 | Flash-to-Base NIAH readout diagnostics. Gold rank is over the full first-token vocabulary; lower rank is better. The prefix, first-token, and exact columns are counts out of 192.
<table><tr><td>Prefix</td><td>Arm</td><td></td><td>Prefix First token</td><td></td><td>Exact Median rank Mean prob. Top-10 (%)</td><td></td><td></td></tr><tr><td>512</td><td>Native</td><td>0</td><td>0</td><td>0</td><td>2151.5</td><td>0.0019</td><td>7.8</td></tr><tr><td>512</td><td>Direct</td><td>169</td><td>0</td><td>169</td><td>2.0</td><td>0.1056</td><td>97.9</td></tr><tr><td>512</td><td>Source</td><td>190</td><td>190</td><td>186</td><td>1.0</td><td>0.9841</td><td>100.0</td></tr><tr><td>2,048</td><td>Native</td><td>0</td><td>0</td><td>0</td><td>18508.0</td><td>2.5×10−5</td><td>0.0</td></tr><tr><td>2,048</td><td>Direct</td><td>175</td><td>15</td><td>174</td><td>2.0</td><td>0.1792</td><td>99.0</td></tr><tr><td>2,048</td><td>Source</td><td>187</td><td>187</td><td>186</td><td>1.0</td><td>0.9621</td><td>99.5</td></tr><tr><td>4,096</td><td>Native</td><td>0</td><td>0</td><td>0</td><td>16466.0</td><td>1.4×10−⁵</td><td>0.0</td></tr><tr><td>4,096</td><td>Direct</td><td>157</td><td>14</td><td>155</td><td>2.0</td><td>0.1847</td><td>97.9</td></tr><tr><td>4,096</td><td>Source</td><td>183</td><td>183</td><td>183</td><td>1.0</td><td>0.9402</td><td>99.5</td></tr></table>

The Direct arm generally does not select the value as its literal first token, particularly at 2K and 4K, yet strict match after stripping whitespace is almost identical to answer-prefix accuracy. This pattern is consistent with a readable value following a formatting token. To test whether the eight-token limit caused the native failure, a continuation probe appended the native arm’s own eight generated template tokens and inspected the following gold-token distribution. At 512 tokens, gold is top-1 in 1/192 cases, top-10 in 35.9%, and has median rank 26; at 4,096, it is top-1 in 2/192, top-10 in 25.5%, and has median rank 30. The corresponding Direct median rank is 2. The result therefore does not reduce solely to the output limit.

## D.2. Crossed Input–Cache Intervention

For the 96-query retrieval cohort, target attention at layers 27, 37, and 42 is evaluated in four copies: target input with target cache, target input with source cache, source-path input with target cache, and source-path input with source cache. Layer weights, token positions, and index-selection recomputation are held fixed as specified in the main text. Table 17 reports median normalized diferences at the prediction position.

Table 17 | Crossed input–cache intervention at the prediction position. All values are normalized attention-output diferences from the target-native result, median over 96 queries. Columns are not additive because changing both factors also changes their interaction.
<table><tr><td>Layer (zero-based)</td><td>Replace local KV cache</td><td>Replace incoming hidden state</td><td>Replace both</td></tr><tr><td></td><td>0.012</td><td>0.142</td><td>0.139</td></tr><tr><td>27 37</td><td>0.019</td><td>0.473</td><td>0.530</td></tr><tr><td>42</td><td>0.005</td><td>0.064</td><td>0.077</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

For the four selected retrieval losses, the replacement outcomes are summarized in Table 18. The cases combine the exploratory and main cohorts, with overlapping roles counted once.

Table 18 | Recovery of four selected retrieval losses after replacing native attention outputs or cache components. Layer numbers are zero-based.
<table><tr><td>Replacement in the direct path</td><td>Layers</td><td>Answers restored</td></tr><tr><td>Native attention outputs</td><td>36-42</td><td>4/4</td></tr><tr><td>Complete native prefix caches</td><td>36-42</td><td>1/4</td></tr><tr><td>Complete native prefix caches</td><td>0-27</td><td>2/4</td></tr><tr><td>Native compressor state bundles</td><td>0-27</td><td>3/4</td></tr></table>

The complete cache replacement over all layers reproduces the native sequence on all 96 queries. Replacing all sufix attention outputs reproduces all 96 first tokens but only 90 complete sequences, which is why output patches are interpreted as local causal tests rather than deployable cache repair.

## D.3. Matched Perturbations and Stopping

The confirmation cohort contains 48 queries from 16 contexts not used by the main retrieval cohort. At layer 37, a natural attention-output diference is injected at full strength and compared with independent per-token random directions and a shared structured rotation. Four repetitions are averaged per query; BF16-rounded perturbation norms are matched at each position within 0.2%.

The mean absolute changes in the native winner’s dynamic margin are 0.693 for the natural patch, 0.624 for independent random directions, and 0.611 for the shared rotation. The random-minusnatural paired efects are −0.068 with 95% interval [−0.204, +0.055] for independent directions and −0.081 with interval [−0.249, +0.078] for the rotation. Both intervals span zero. At layer 42, the corresponding mean margin changes are 1.724, 0.545, and 0.533, while the natural patch changes the selected first token on one of 48 queries. These patches act during sufix processing with the other layers’ caches native, so they measure local sensitivity rather than the full direct-reuse trajectory.

For this supplementary stopping probe, we use $m _ { \mathrm { s t o p } } = z _ { \mathrm { E O S } } - \operatorname* { m a x } _ { \nu \neq \mathrm { E O S } } z _  \nu \mathrm  _  \ /$ , the EOS token’s lead over the strongest continuation token. A positive margin favors stopping. Target and KV Reuse continue from the same first reference-answer tokens without consuming EOS. On the 20-example random HotpotQA subset, the mean stopping-margin change is −1.131 with 95% interval [−1.581, −0.700] and decreases on 17/20 examples. On the 20-example random 2WikiMQA subset, it is −1.150 with interval [−1.850, −0.488] and decreases on 15/20 examples. The intervals describe these preselected subsets from the original 100-example task pools.