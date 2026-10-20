# Answer Delivery Under a Shared Token Budget: An Audited Study of Stopping and Finalization

Yupeng Chang<sup>1</sup> Linyuan Gao<sup>1</sup> Zhiyuan Li<sup>1</sup>

Yuan Wu<sup>1\*</sup> Yi Chang<sup>1,2,3</sup>

<sup>1</sup>School of Artificial Intelligence, Jilin University

<sup>2</sup>Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China <sup>3</sup>International Center of Future Science, Jilin University

{changyp23, lygao25, zhiyuanl24}@mails.jlu.edu.cn, {yuanwu, yichang}@jlu.edu.cn

## Abstract

An inference budget is useful only when the system delivers an answer within it. We study a fixed-budget policy combining first-valid-box stopping with a charged finalization cue. On 100 fresh, independently audited GSM8K test questions per model, it improves delivered endpoint accuracy over first-box stopping by 8.0 percentage points for Qwen3-4B-Thinking-2507 and 7.0 for DeepSeek-R1-Distill-Llama-8B. A prelocked repeat on the same questions gives 7.0 and 10.0 points; neither seed shows a gross harm relative to first-box stopping. The point estimates of incremental physical-slot savings are below one percent. A finalizationonly ablation separates cue recovery from stopping, and a fixed scalar reader tests extraction as an alternative explanation. The cue is prior art; the contribution is a component and readout analysis with charged injections and verified paths. Results cover two checkpoints, one task, one prompt family, and two fixed seed conditions.

## 1 Introduction

A reasoning system must return a usable answer before its inference budget expires. A long completion can fail this requirement in two different ways: it may continue after an answer is already available, or it may spend its budget reasoning without producing a readable answer. Stopping targets the first case; finalization targets the second. Combining them is practical, but accuracy or token count alone does not show which component drives each effect.

Our question is whether a conditional finalization cue improves the answer actually delivered by first-box stopping when the cue and answer must fit inside the original output budget. We examine three quantities separately: answer availability, numerical correctness, and physical generation cost.

Removing missing answers need not remove incorrect answers. Similarly, shortening a completion after a box appears need not preserve the value selected at native termination. These distinctions matter when a serving interface must choose a single endpoint rather than retrospectively search a completed trace.

Budget forcing in s1 appends an end-of-thinking delimiter and optional answer cue (Muennighoff et al., 2025). The Coupling Tax studies answer crowding under shared output limits via splitbudget generation (Nie et al., 2026). Answerconvergence stopping probes prefixes with forced answers (Liu and Wang, 2025); an interface study separates completion timing from forced and generated readouts (Carlon et al., 2026); PUMA separates answer readiness from reasoning convergence (Min et al., 2026). Related early-exit, budget-aware, overthinking, and no-thinking work provides complementary context (Yang et al., 2026; Sun et al., 2025; Han et al., 2025; Chen et al., 2025; Ma et al., 2025). We study a live-cache implementation without claiming a new cue or convergence detector.

We physically generate native completion, firstbox stopping, their combined finalization policy, and a prelocked finalization-only ablation on the same questions. Injected tokens count toward the common cap; branches reset their cache and sampler seed. An independent audit verifies complete coverage, identities, exact prefixes, and forward accounting before reference scoring. Each policy is physically executed, with measured costs for its actual generation path.

Our contribution is a bounded component and readout study with reproducible accounting. Conditional finalization recovers boxed numeric endpoints; stopping accounts for most R1 workload savings and can discard a later correct Qwen answer. Historical pilots are not pooled into the fresh cohort. Deep-thinking tokens and Think@n (Chen et al., 2026) concern internal measures; the present study tests delivered endpoints and physical costs.

## 2 Delivery and Cost Protocol

## 2.1 Policies and returned values

All policies operate under a post-prompt cap $C =$ 2048. FULL continues until native EOS or the cap and reads the last syntactically completed \boxed{. . . } occurrence. An invalid or absent last box returns null. FIRSTBOX follows the same native decoding process but returns as soon as a completed box passes a fixed numeric grammar; without one, it falls back to the native EOS/cap endpoint. The selectors are part of each policy and are fixed before evaluation.

SOLUTION follows FIRSTBOX, with one conditional intervention. If generation reaches slot 1984 without a valid numeric box, it injects a newline, </think>, two newlines, and Final answer: \boxed{. The cue occupies nine slots for each tested tokenizer, leaving 55 of the reserved 64 slots for sampled continuation. Generation uses the existing KV cache and stops at the first valid numeric box; native EOS/cap provides the fallback if none appears. The cue uses only the observed prefix, with no reference, future trajectory, or correctness signal. Native EOS before the trigger is respected.

FINALIZE uses the same cue condition and budget but continues to native EOS/cap, applying FULL’s final-box selector. It isolates finalization without first-box stopping.

We score the returned value by exact rational equality to the reference; null counts as incorrect. A cap is a finish reason, not an automatic error: a valid value is scored even at the limit. Availability means a grammar-valid boxed numeric endpoint; null need not imply absence of a correct unboxed answer. All arms share a correctness criterion while retaining their operational endpoint selectors. Alternative readouts appear in Appendix A.4.

## 2.2 Physical budget and matched paths

Let P be prompt length and N the emitted postprompt slots. The accounting boundary is

$$
N = N _ { \mathrm { s a m p l e d } } + N _ { \mathrm { i n j e c t e d } } , \qquad N \leq C .\tag{1}
$$

In this singleton implementation, producing N slots takes one prefill and N − 1 incremental forwards, consuming $P + N - 1$ model input tokens. The final emitted token is not forwarded. Forced tokens occupy the cache path but do not call the sampler or consume sampling RNG draws. The intervention therefore changes the trajectory without purchasing a second answer budget.

Each question’s paths share a prompt and reset seed. FIRSTBOX must equal the native prefix or fallback; SOLUTION must equal FIRSTBOX when unexposed. FINALIZE must equal FULL when unexposed and continue the actual SOLUTION prefix when exposed. All arms share the cap and decoder, but realized compute can differ.

Request wall time includes generation, monitoring, durable log flush, and result output. Wholeprocess time additionally includes setup, loading, integrity checks, and exit. Component timings are nested within those totals and are not added again. Physical slots and input tokens are workload counts, not estimates of FLOPs, monetary cost, or batched throughput.

## 2.3 Paired estimands and qualification

For question i, let $Y _ { i } ^ { a }$ indicate correctness under policy a. For SOLUTION (S) versus FIRSTBOX (B), define gains $\begin{array} { r } { G = \sum _ { i } ( 1 - Y _ { i } ^ { B } ) Y _ { i } ^ { S } } \end{array}$ and harms $\begin{array} { r } { H = \sum _ { i } Y _ { i } ^ { B } ( 1 - Y _ { i } ^ { S } ) } \end{array}$ . The paired accuracy difference is

$$
\widehat { \Delta } _ { S , B } = \frac { 1 } { n } \sum _ { i } ( Y _ { i } ^ { S } - Y _ { i } ^ { B } ) = \frac { G - H } { n } .\tag{2}
$$

For cost K, reduction is $\begin{array} { r } { 1 - \sum _ { i } K _ { i } ^ { S } / \sum _ { i } K _ { i } ^ { B } } \end{array}$ , a ratio of totals rather than a mean of per-question percentages. We resample paired questions 10,000 times with fixed seed 20261012 for descriptive percentile 95% intervals. Models are analyzed separately; repeated arms do not add independent questions.

Before scoring, the audit requires successful worker and supervisor completion, exact planned IDs, sealed source and input maps, pre/post model-file receipts, GPU identities, and matching started/completed forward attempts. Prefix and cue checks must also pass. Reference answers remain in a separate sidecar until qualification. Implementation, parser, and audit details appear in Appendix A.

## 3 Experiments and Results

## 3.1 Cohort, controls, and baselines

We use pinned Qwen3-4B-Thinking-2507 and DeepSeek-R1-Distill-Llama-8B checkpoints (Team, 2025; DeepSeek-AI, 2025) on GSM8K (Cobbe et al., 2021). The cohort contains 100 distinct test questions per model, selected by a fixed question-hash ordering before any output was inspected and excluding previously used questions. The models share the cohort but remain separate analyses. No learned policy parameters or fresh-test calibration are used; the cue, trigger, cap, prompt, and seed were fixed before the preceding development check.

<table><tr><td>Model</td><td>Policy</td><td>Correct / 100</td><td>No value</td><td>Mean slots</td><td>Mean wall (s)</td></tr><tr><td>Qwen</td><td>FULL</td><td>83</td><td>14</td><td>1179.80</td><td>47.64</td></tr><tr><td>Qwen</td><td>FIRSTBOX</td><td>82</td><td>14</td><td>1158.97</td><td>46.81</td></tr><tr><td>Qwen</td><td>SOLUTION</td><td>90</td><td>0</td><td>1151.69</td><td>46.44</td></tr><tr><td>R1</td><td>FULL</td><td>80</td><td>12</td><td>1225.07</td><td>36.15</td></tr><tr><td>R1</td><td>FIRSTBOX</td><td>80</td><td>12</td><td>1054.47</td><td>31.16</td></tr><tr><td>R1</td><td>SOLUTION</td><td>87</td><td>0</td><td>1048.12</td><td>30.94</td></tr></table>

Table 1: All three physically generated policies on 100 fresh GSM8K test questions per model. Correctness uses the numeric value actually returned; No value means the endpoint returned null. Mean post-prompt slots include injected cue tokens. Mean wall time includes generation, monitoring, flush, and result output, excluding shared process setup and offline audit. Bold marks highest correctness among shown arms.

Inference uses singleton BF16 SDPA, temperature 0.6, top-p 0.95, and reset seed 1357911. Six three-arm order permutations cycle by question rank. All 600 primary, 200 later FINALIZE, and 600 repeat paths completed and passed independent auditing. Both added stages were locked before primary scoring and reuse the cohort; later ablation wall contrasts are stage-confounded. Runtime details appear in Appendix A.2.

The primary contrast is SOLUTION versus FIRSTBOX; native-completion and component contrasts are secondary. FINALIZE returns 91 correct Qwen and 87 correct R1 answers, versus 90 and 87 for SOLUTION. Relative to FULL, FINAL-IZE gains 8.0 percentage points for Qwen and 7.0 for R1, matching SOLUTION’s gains over FIRST-BOX. First-box stopping loses one otherwise correct Qwen answer while saving workload. Appendix A.8 gives the paired four-policy analysis. Published convergence mechanisms are related work, not reproduced cost baselines.

## 3.2 Accuracy and harm

Table 1 reports all arms with a common correctness criterion. Relative to FIRSTBOX, SOLUTION gains 8.0 percentage points for Qwen, with paired 95% interval [3.0, 14.0], and 7.0 for R1, with interval [2.0, 12.0]. The respective discordant pairs are 8 gains and 0 harms, and 7 gains and 0 harms.

Reporting both counts prevents net accuracy from concealing baseline-correct answers lost by the intervention.

Zero observed harm is a finite-cohort observation. A one-sided binomial upper bound at α = 0.025 is 3.62% for each model; Bonferroni correction covers the two model-wise harm statements under the binomial sampling assumption. It concerns harm relative to the comparator, not absolute answer error. Each interval resamples questions within its task, checkpoint, prompt, and seed condition.

At locked seed 20261013, SOLUTION again gains 7.0 points for Qwen (95% [3.0, 12.0]) and 10.0 for R1 ([5.0, 16.0]), with zero observed harms relative to FIRSTBOX. R1 still returns 1 null endpoint. This repeat adds no questions; Appendix A.10 reports complete arms and costs.

The secondary comparison exposes an endpoint tradeoff. Qwen FULL is correct on 83 questions, compared with 82 for FIRSTBOX; one nativecorrect answer is lost at the earlier endpoint. R1 remains at 80 correct under both. Thus a first box cannot be assumed to equal the eventual answer even when the two paths share an exact prefix.

## 3.3 Where the additional answers come from

In P0, we perform a descriptive, post-hoc decomposition of audited question pairs; it adds no independent observations. Qwen’s FIRSTBOX arm returns no value on 14 questions, and R1’s on 12. SOLUTION reduces both null counts to zero. All 8 and 7 correctness gains occur on those previously null endpoints. Wrong-valued baseline answers receive no cue and remain wrong. Appendix A.7 gives the complete transition counts.

Availability and correctness therefore separate sharply. Of the formerly null questions, 6 Qwen and 5 R1 deliveries become readable but remain wrong. Converting null to a number is useful for an interface, yet it would overstate task improvement to count every such conversion as a recovered solution. The aggregate accuracy increase is specifically the correct subset of these conversions.

<table><tr><td>Model</td><td>Pair</td><td>Slots ↓(%)</td><td>Wall↓(%)</td></tr><tr><td>Qwen</td><td>B/F</td><td>1.77</td><td>1.73</td></tr><tr><td>Qwen</td><td>S/F</td><td>2.38</td><td>2.51</td></tr><tr><td>Qwen</td><td>S/B</td><td>0.63</td><td>0.80</td></tr><tr><td>R1</td><td>B/F</td><td>13.93</td><td>13.79</td></tr><tr><td>R1</td><td>S/F</td><td>14.44</td><td>14.41</td></tr><tr><td>R1</td><td>S/B</td><td>0.60</td><td>0.71</td></tr></table>

Table 2: Reductions in total physical slots and measured request wall time. F=FULL, B=FIRSTBOX, S=SOLUTION; X/Y compares policy X with baseline Y. These are singleton instrumented costs on the same paired questions, not serving-throughput estimates.

Cue activation occurs on 16 Qwen and 12 R1 paths, injecting 144 and 108 slots. Activation is not identical to a null baseline endpoint: Qwen has 2 activated paths whose no-cue answers are eventually correct. They remain correct after intervention. Nonactivated SOLUTION paths match FIRSTBOX in emitted IDs, text, and returned values. These checks localize the observed accuracy changes to intervention-exposed paths, while leaving open whether the cue surfaces existing partial computation or changes subsequent reasoning.

## 3.4 Cost depends on the comparator

Against FIRSTBOX, SOLUTION reduces slots by 0.628% for Qwen (95% interval [0.371, 0.886]) and 0.602% for R1 ([0.323, 0.880]). Wall reductions are 0.800% and 0.711%. The incremental cue benefit is consequently accuracy recovery with only a small average workload change, not a large speedup.

Table 2 shows why a single headline efficiency number would mislead. R1 FIRSTBOX already saves 13.93% of slots against FULL, and SOLU-TION saves 14.44%. Qwen’s corresponding reductions are 1.77% and 2.38%, with the one lost native-correct answer shared by both stopped arms. Most R1 savings arise from removing its native tail, whereas the cue contributes the new correct deliveries. Table 4 in the appendix records component totals for auditing.

## 4 Discussion

The experiment separates shortening an availableanswer tail from recovering a missing delivery. Tail savings require endpoint preservation; recovery requires correctness on converted nulls. At the primary seed, R1 cost savings mostly come from the former and accuracy gains entirely from the latter. Qwen shows that earlier endpoints can lose a correct eventual answer.

At the primary seed, finalization-only reproduces the cue’s accuracy gains without stopping, while retaining Qwen’s later correct answer. Zero questionwise accuracy interaction describes this cohort; it is not a population-equivalence certificate. Accuracy recovery and tail savings remain separate objectives.

Formatting recovery remains a strong alternative to changed reasoning. A fixed post-hoc reader, preferring the last valid box and otherwise the last scalar literal, raises FULL correctness to 84/81 (Qwen/R1) at the primary seed and 85/77 at the repeat, below actual SOLUTION deliveries. However, SOLUTION has one harm relative to this alternative reader for primary-seed Qwen and repeat-seed R1 (Appendix A.9). This heuristic does not explain all gains; its failure cannot identify improved reasoning. The cue changes the live cache; a mechanism claim needs controls beyond extraction.

The prior-work boundary is narrow. s1 establishes forced budget termination; Coupling Tax studies answer crowding; answer-convergence and PUMA use answer agreement or reasoningconvergence signals. Our measurement documents what this combination returns, where gains occur, and which component accounts for costs. It does not rank those methods or establish a new general convergence mechanism.

For deployment, the cap and returned-value rule are interface choices. Recovering a number does not ensure correctness, and native continuation may preserve later corrections. Transfer requires rechecking grammar, prompt, and workload. Subpercent wall differences against FIRSTBOX need separate batched serving measurements before a latency claim.

## 5 Conclusion

Conditional finalization improves first-box endpoint accuracy by 8.0 and 7.0 points for Qwen and R1 at the primary seed; a locked repeat retains the direction. Point estimates of incremental slot savings remain below 1%; larger R1 savings against native completion come mainly from stopping. These results measure delivered answers and physical costs, not reasoning convergence.

## Limitations

The primary P0 results concern 100 distinct GSM8K test questions per model, one prompt family, one primary seed per question, and two checkpoints. The additional fixed seed reuses these questions. Models, seeds, and repeated arms are not pooled into a larger independent sample. Questionbootstrap intervals are descriptive and do not provide unconditional guarantees for arbitrary tasks, prompts, models, or seeds. Zero observed primary gross harms do not imply zero population risk. Under a binomial sampling assumption, the original two-model Bonferroni procedure gives a 95% family-wise one-sided upper bound of 3.62% for each model; it is neither a distribution-free certificate for this deterministically selected cohort nor a joint bound over both seeds.

At the primary seed, only two cue-exposed Qwen baselines were correct; all 12 exposed R1 baselines were null. Unexposed paths are identical by construction. The aggregate bound targets harm over questions, not harm conditional on cue exposure or baseline correctness. Zero aggregate harm therefore gives limited evidence about conditional cue safety. The post-hoc scalar reader is a deliberately limited heuristic; its failure neither rules out better extraction nor identifies a latent reasoning improvement.

The Solution policy is a controlled endpoint intervention. It does not establish reasoning convergence, causal usefulness of deep-thinking-token measures, or a new algorithmic mechanism. The cue and related forced-finalization procedures are prior art. Relative to Full, Qwen has one originalseed gross harm; therefore the paper does not claim universal safety or harm-free operation. The study uses a single task, prompt family, two fixed seeds, and two checkpoints, and its singleton instrumented wall times do not generalize to batched serving or monetary cost.

The original development pilot and historical stopping study remain archived as engineering and contextual evidence; they are not pooled into the fresh P0 inferential sample. One locked seed repeat supports consistency in this setting, not generality across arbitrary seeds, tasks, or models. The repeat retains one R1 null, so the original elimination of missing boxed values is not universal. Finalization-only concerns the original seed; fixed scalar-readout sensitivity covers both seeds and retains harms against the alternative comparator.

The prelocked finalization-only ablation reuses the original questions and seed. Its accuracy and physical-slot contrasts are coupled, but it ran after the three primary arms; stage and hardware-load differences confound wall-time contrasts. Zero observed question-wise accuracy interaction does not establish population equivalence or exclude interactions in other prompts, seeds, or cue-exposed cohorts. The new wrapper’s recorded command wall also excludes its orchestration checks outside that command and is not a full rental-cost measurement.

## Ethical Considerations

The study uses public mathematical word problems and released model checkpoints, without new human-subject data. Correct delivery means matching a benchmark numeric answer; it does not establish safe use in consequential applications. We retain failed and null deliveries and qualify complete evidence before scoring to reduce selective reporting. The experiments concern inference-time computation and do not train or modify model weights.

## Acknowledgments

OpenAI ChatGPT and Codex assisted with implementation and debugging of inference/audit code, local artifact preparation, literature search and primary-source citation checking, evidence organization, manuscript framing, and drafting/editing prose.

## References

Francesca Carlon, Vincent Ginis, and Andres Algaba. 2026. Shorter reasoning, earlier answers? an evaluation of reasoning interfaces. arXiv preprint arXiv:2608.03401.

Wei-Lin Chen, Liqian Peng, Tian Tan, Chao Zhao, Blake JianHang Chen, Ziqian Lin, Alec Go, and Yu Meng. 2026. Think deep, not just long: Measuring llm reasoning effort via deep-thinking tokens, 2026. URL https://arxiv.org/abs/2602.13517.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. 2025. Do not think that much for 2+ 3=? on the overthinking

of long reasoning models. In Forty-second International Conference on Machine Learning.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168, 9.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. 2025. Tokenbudget-aware llm reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24842–24855.

Xin Liu and Lu Wang. 2025. Answer convergence as a signal for early stopping in reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17907–17918.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. 2025. Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858.

Dehai Min, Giovanni Vaccarino, Huiyi Chen, Yongliang Wu, Gal Yona, and Lu Cheng. 2026. Stop when reasoning converges: Semantic-preserving early exit for reasoning models. arXiv preprint arXiv:2605.17672.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori B Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20286–20332.

Wenhua Nie, Junlin Liu, Jianan Wu, Zijie Meng, Yilong Fan, Zhang Zijian, Haoran Zheng, and Jyh-Shing Roger Jang. 2026. The coupling tax: How shared token budgets undermine visible chain-ofthought under fixed output limits. arXiv preprint arXiv:2605.07686.

Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, and Wei Wang. 2025. Stop when enough: Adaptive early-stopping for chain-of-thought reasoning, 2025. URL https://arxiv.org/abs/2510.10103.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weipinng Wang. 2026. Dynamic early exit in reasoning models. In International Conference on Learning Representations, volume 2026, pages 88170–88210.

## A Reproducibility Details

## A.1 Fixed Cohort and Analysis Scope

The primary P0 cohort contains 100 distinct questions from the official GSM8K test parquet. The selector first verifies the pinned parquet and exclusion-source byte hashes, then reads only the question column for membership selection. It excludes the sealed prior-question hash set, deduplicates identical question text using the lowest source index, and sorts eligible questions by SHA256 of the UTF-8 string

budget-finalization-confirm-20261012-v1|questionSHA.

The first 100 entries form the fixed cohort. The full question SHA, source index, selection hash, request ID, and question text are retained. Selection precedes reading the answer column; references are written to a separate offline gold sidecar. The generator’s exact nine-field request schema contains no reference, score, or prior output.

Both models use the same selected questions. Each model has three physically executed paths per question, with the tuple (full, firstbox, solution) permuted in the six orders produced by itertools.permutations; question rank modulo six selects the order. Each path resets its sampler and native cache. Models, arms, and repeated seeds are never counted as additional independent questions. There is no replacement question, outcomedependent expansion, or automatic retry.

Historical development and stopping artifacts remain separate. P0 executes FULL, FIRSTBOX, and SOLUTION. The later FINALIZE component ablation, locked before P0 quality scoring, reuses all original questions and seed and contributes no independent questions. The migrated engineering replay also adds no test questions.

## A.2 Checkpoints, Runtime, and Prompt

The checkpoints are Qwen/Qwen3-4B-Thinking-2507, revision

768f209d9ea81521153ed   
38c47d515654e938aea,

and deepseek-ai/ DeepSeek-R1-Distill-Llama-8B, revision

6a6f4aa4197940add5772   
4a7707d069478df56b1.

Acquisition manifests pin all checkpoint and tokenizer filenames, sizes, and SHA256 values. The executed P0 runtime is Python 3.10.20, PyTorch 2.5.1+cu121, Transformers 4.51.3, and NVIDIA driver 550.54.14. Inventory receipts identify two NVIDIA GeForce RTX 4090 devices, with Qwen assigned physical GPU 2 and R1 physical GPU 3 by UUID. Initial and final ownership checks do not establish continuous device exclusivity.

Every P0 path uses BF16, SDPA attention, batch size one, temperature 0.6, top-p 0.95, and seed 1357911. The sampler converts logits to FP32, divides by temperature, sorts them, and removes tokens after the nucleus threshold while retaining the token that crosses it. Sampling uses torch.multinomial with a separate CUDA generator initialized to the fixed seed for each path. CPU thread counts are one, tokenizer parallelism is disabled, and model loading is local and offline.

The exact user content is the original question followed by one newline and this suffix:

Reason step by step and give the final numeric answer in \boxed{...}.

The template is supplied no system message or fewshot example. The tokenizer’s pinned chat template receives one user message with tokenize=True and add\_generation\_prompt=True. Prompts are not truncated; the worker requires a nonempty encoding of at most 4096 tokens and archives its exact IDs. The alternative estimate-and-verify suffix retained in the development prompt specification is not used in P0.

## A.3 Cue and Native Execution

SOLUTION stops at the first completed numeric box. If no such box has completed after slot 1984 and generation remains live, it substitutes the inherited cue beginning at slot 1985. The cue’s exact escaped representation is

\n</think>\n\nFinal answer: \boxed{ where each \n denotes one line-feed character, rather than two literal characters. The inherited Qwen cue IDs are

198, 151668, 271, 19357, 4226,

25,

1124, 79075, 90,

and the R1 cue IDs are

198, 128014, 271, 19918, 4320,

25,

1144, 80175, 90.

Each cue has nine tokens and occupies slots 1985– 1993, leaving at most 55 sampled slots within the original 64-slot reservation. Cue IDs are copied from the qualified development preflight, checked against its seal, and required not to contain any model EOS ID. P0 does not re-encode or search for an alternative cue.

The cue is inserted during the same native decode call using its live KV cache, attention mask, and position sequence. Forced slots do not call the sampler and consume zero RNG draws. After each emitted token, the next incremental forward receives that token with the current cache, extends the mask by one, and advances the position ID. The last emitted token is not forwarded. There is no cache serialization, reload, hidden replay, or additional model call for monitoring. Generated text is decoded from generated IDs with special tokens skipped; physical token accounting still includes those IDs.

FIRSTBOX must equal the exact first-box prefix of FULL, or its EOS/cap fallback if no valid box appears. An activated SOLUTION must equal FULL through slot 1984 and contain the exact cue at the recorded positions. An unactivated SOLU-TION must equal FIRSTBOX. These checks concern within-run pairing; changed runtime and hardware do not imply equality to historical sampled trajectories.

## A.4 Numeric Grammar and Endpoint Sensitivities

The restricted parser accepts signed integers, finite decimal literals, scientific notation, and exact fractions written with one top-level slash or a balanced \frac, \dfrac, or \tfrac command. Fraction components are parsed recursively. It normalizes Unicode minus, removes \left and \right, permits one surrounding dollar-sign pair and enclosing balanced parentheses or braces, and accepts correctly grouped thousands separators. It rejects separate adjacent numbers, units, free-form arithmetic, empty values, malformed fractions, zero denominators, strings longer than 512 characters, and scientific exponents with absolute value above 1000. Parsing uses Decimal and Fraction, with exact rational equality and no tolerance or expression evaluation.

The online monitor reads generated text only and tracks balanced, unescaped \boxed{. . . } groups. A policy stop commits when the first group closes with a valid numeric value. It returns that value at the actual emitted token boundary, which may also contain other text from the same token. If no policy stop occurs, the native EOS/cap endpoint selects the completed box with the latest opening position. An invalid final completed box returns null even if an earlier box was valid. An unfinished later box does not erase an earlier completed box under the primary delivered-value rule. No unboxed-number fallback is allowed.

<table><tr><td>Model</td><td>Primary</td><td>Valid</td><td>EOS</td><td>Legacy</td></tr><tr><td>Qwen</td><td>83</td><td>83</td><td>82</td><td>82</td></tr><tr><td>R1</td><td>80</td><td>80</td><td>78</td><td>78</td></tr></table>

Table 3: Descriptive endpoint sensitivities for FULL. Primary is the delivered-value rule, Valid selects the last valid box, EOS requires native termination, and Legacy uses the nontruncated score. Entries count correct questions with the same 100-question denominator.

Primary correctness compares the actual returned value to the reference regardless of finish reason. The supplemental last-valid-box readout instead ignores invalid completed boxes; native-EOSonly correctness additionally requires native EOS; the legacy nontruncated score rejects cap endpoints and any unfinished box. These are descriptive sensitivities, not replacements for the primary endpoint. The native-EOS restriction would reject intentional policy stops, so Table 3 presents its informative comparison for FULL only.

## A.5 Qualification and Immutable Provenance

Quality scoring begins only after both models’ complete markers and successful supervisor exits exist. The independent audit requires exact completeplan membership and order; acquisition, source, input, deployment, and output seals; recorded runtime and GPU bindings; prompt, seed, and question identities; complete started/completed forward ledgers; slot and model-input accounting; exact cue and paired-prefix coupling; and one answer emission per path. It independently reconstructs box events and numeric values from raw generated text, checks the actual stdout endpoint, verifies that answer publication precedes buffered evidence flushing, and requires no pending events after durable flush. The complete archive is checked again before crossing the gold-reading boundary. A qualification failure produces no partially scored table.

Mandatory CPU and native-cache/RNG tests and the complete label-free engineering replay precede fresh confirmation. The executing worker is an explicitly sealed compatibility copy: it resolves the science directory through a controlled environment variable, checks the actual amended Torch version, and preserves exception notes through Python 3.10’s \_\_notes\_\_ attribute. The separately recorded watchdog amendment changes the original 10800-second ceiling to 28800 seconds with 30 seconds for cleanup; the per-answer 2048-slot budget and per-model 614400-forward ceiling remain fixed. Original scientific cores, request membership, sampler, prompt, cue, and analysis rules are retained. Worker diffs and original/executing hashes distinguish these changes from the frozen scientific package.

The reproducibility archive separates frozen source and inputs, acquisition maps, executing deployment copies and diffs, qualification logs, and complete raw runs. The generated numerical interface records the independently qualified summary’s SHA256, expected deployment/qualification/engineering/freeze/plan/request seals, bootstrap settings, and macro-to-source mapping. Thus manuscript values derive from the audited summary rather than hand-entered results. The offline archive validates the recorded acquisition and pre/post full-file maps; it does not independently rehash absent multi-gigabyte model bytes or re-execute tokenization.

## A.6 Costs and Statistical Recalculation

Table 4 reports actual costs for all three executed arms. For a prompt of P tokens and N emitted slots, there are N forwards: one prefill and N − 1 incremental decode forwards. Physical modelinput tokens total P +N −1. Slots include sampled, injected, and emitted EOS tokens. These counts are not estimates of FLOPs, energy, or monetary cost.

Answer-availability time ends after flushed stdout publication. Request wall includes the native wrapper, buffered evidence flush, and durable result output. Prefill, decode, sampling, text decoding, monitoring, and ledger component times are contained within these outer scopes and must not be added to them again. Whole-process supervisor wall additionally includes adapter imports and checks, checkpoint loading, all arms, post-run hashing, and exit: 14149.60 seconds for Qwen and 9934.77 seconds for R1. Its shared overhead cannot be attributed to one arm as a measured standalone deployment cost. Offline gold scoring and bootstrap computation occur after generation and are outside these request and worker timings.

The bootstrap resamples paired questions within each model using random.Random(20261012) and 10000 resamples. The accuracy statistic is the mean policy-minus-baseline correctness difference. A cost reduction is one minus the ratio of the summed policy costs to summed baseline costs, both in the point estimate and each resample; it is not the mean of question-level ratios. The reported interval uses linearly interpolated 2.5th and 97.5th percentiles. These percentile intervals are descriptive.

Gross harm counts baseline-correct/policywrong questions, with all 100 questions as the denominator. The one-sided exact Clopper–Pearson bound uses $\alpha = 0 . 0 2 5$ per model for the two-model family claim; $\alpha = 0 . 0 5$ bounds are separate descriptive per-model quantities. Gains and harms remain separate, and exact two-sided McNemar tests use both discordant counts. The fixed primary comparison is SOLUTION versus FIRSTBOX; comparisons with FULL remain secondary. Two fixed seeds and one task do not establish robustness across tasks, samplers, cues, reservation sizes, or checkpoints.

## A.7 Post-hoc Endpoint Decomposition

Table 5 describes transitions between the already qualified FIRSTBOX and SOLUTION endpoints. A wrong value is a present numeric value that differs from gold; a null is an absent returned value. Activation indicates that SOLUTION inserted the cue, not that the baseline answer was wrong. A baseline may return a correct box after the cue boundary, so activation alone is not an accuracy gain. All wrongvalued baseline paths stop before cue exposure; their unchanged outcomes do not test the cue’s ability to repair a wrong answer. Converting a null to a wrong value improves numeric-answer availability but remains incorrect under primary scoring. This post-hoc decomposition adds no independent question, subgroup confidence interval, hypothesis test, or out-of-sample effect. It uses audited endpoint exports and their recorded generated-ID/text hashes rather than a new token replay.

## A.8 Prelocked Finalization-only Component

FINALIZE (T) conditionally inserts the identical nine-token cue after slot 1984 but continues to EOS/cap and uses the primary final-box readout. Both checkpoints execute all 100 original questions at seed 1357911, with the same prompt, sampler, checkpoint, cap, and native cache path. The source, inputs, threshold, budget, and test order were fixed before P0 scoring. The complete package passed 104 mandatory CPU tests before generation. A separate actual-run audit checks all 200 added paths and exact F/T identity when unexposed, pre-cue F/T identity when exposed, and the full S/T firstbox prefix. A separately written raw-endpoint and question-bootstrap recalculation reproduces all counts and intervals. These are four policies on 100 paired questions per model, not 400 independent observations.

T versus F gains 8 correct Qwen and 7 correct R1 answers, with zero observed harms. S versus T loses one correct Qwen answer and has no R1 correctness discordance, while reducing physical slots by 2.165% and 14.226%. The lost Qwen answer is the same unexposed endpoint change already seen in B versus F. This separates observed cue recovery from shortening the native tail.

For each question, the prespecified accuracy interaction is $\hat { ( Y ^ { S } - Y ^ { B } ) } \stackrel { \hat { ( } } { - } ( \hat { Y ^ { T } } - Y ^ { F } )$ . Every observed value is zero; the empirical bootstrap therefore degenerates at zero. That resampling fact is not a population-equivalence certificate, especially with only 16/12 exposed questions. Absolute slot interactions average -4.660 and -3.230 slots per question. Ratios with different baseline denominators are not subtracted to construct an interaction. The later T request wall also changes relative to F despite little slot change; stage differences prevent interpreting that timing difference as an algorithmic speedup.

Current-stage physical forwards total 117718 for Qwen and 122195 for R1. Their recorded bounded worker-command supervisor walls are 4884.56 and 3671.65 seconds; these contain worker generation, setup and hashing but exclude pre-command/postcommand orchestration checks. They must not be relabeled full orchestration or rental cost, or summed with nested request phases.

## A.9 Fixed Post-hoc Scalar Readout

To test a cheap extraction alternative, we freeze Rv1 before inspecting its quality. It applies the same reader to every complete audited F/B/S trace at both fixed seeds; after T’s complete qualification, the unchanged reader also processes all original-seed T traces. This adds no model call, continuation, or independent question. The reader returns the last valid completed box; when none exists, it returns the last boundary-qualified signed scalar literal in the generated text. Scalars permit decimals, scientific notation, and grouped thousands under the same exact rational parser. Attached word identifiers, slash-adjacent numbers, malformed comma grouping, and multi-dot chains are rejected. Unboxed expressions and fractions are not evaluated; incidental numbers in prose may be selected. The full regular expression and protocol are in the reproducibility materials.

<table><tr><td>Model</td><td>Policy</td><td>Total slots</td><td>Sampled</td><td>Injected</td><td>Input tokens</td><td>Wall (s)</td></tr><tr><td>Qwen</td><td>FULL</td><td>117980</td><td>117980</td><td>0</td><td>126946</td><td>4763.57</td></tr><tr><td>Qwen</td><td>FIRSTBOX</td><td>115897</td><td>115897</td><td>0</td><td>124863</td><td>4681.28</td></tr><tr><td>Qwen</td><td>SOLUTION</td><td>115169</td><td>115025</td><td>144</td><td>124135</td><td>4643.82</td></tr><tr><td>R1</td><td>FULL</td><td>122507</td><td>122507</td><td>0</td><td>130721</td><td>3614.71</td></tr><tr><td>R1</td><td>FIRSTBOX</td><td>105447</td><td>105447</td><td>0</td><td>113661</td><td>3116.08</td></tr><tr><td>R1</td><td>SOLUTION</td><td>104812</td><td>104704</td><td>108</td><td>113026</td><td>3093.92</td></tr></table>

Table 4: Physical P0 costs from completed path and process receipts. Each arm contains 100 executed questions per model. Request wall includes generation, flush, and result output; the separately stated whole-process totals contain all three arms and their shared overhead. Nested timing scopes are not summed.
<table><tr><td>Model</td><td>FirstBox → Solution</td><td>n</td><td>Active</td><td>Gain</td><td>Harm</td></tr><tr><td>Qwen</td><td>Null → correct</td><td>8</td><td>8</td><td>8</td><td>0</td></tr><tr><td></td><td>Null → wrong value</td><td>6</td><td>6</td><td>0</td><td>0</td></tr><tr><td></td><td>Correct → correct</td><td>82</td><td>2</td><td>0</td><td>0</td></tr><tr><td></td><td>Wrong value → wrong value</td><td>4</td><td>0</td><td>0</td><td>0</td></tr><tr><td>R1</td><td>Null → correct</td><td>7</td><td>7</td><td>7</td><td>0</td></tr><tr><td></td><td>Null → wrong value</td><td>5</td><td>5</td><td>0</td><td>0</td></tr><tr><td></td><td>Correct → correct</td><td>80</td><td>0</td><td>0</td><td>0</td></tr><tr><td></td><td>Wrong value → wrong value</td><td>8</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 5: Post-hoc descriptive FIRSTBOX→SOLUTION endpoint transitions on the existing P0 cohort. n counts questions, Active counts cue activations, and Gain/Harm count correctness discordances. Models remain separate; subgroup membership does not increase independent sample size.

Extraction consumes a six-field label-free adapter containing model, arm, question identity, generated text, original returned value, and actual slot count. Gold is opened separately after complete extraction. The protocol, reader source, adapter, and extracted spans are hashed; eleven synthetic reader tests and an independent root recalculation passed. No parser, span, or output is selected by correctness. Original primary scores remain unchanged.

At the primary seed, R-v1 removes every null but adds only one correct answer per model in F/B (Table 7). Its T scores remain 91/87, identical to the primary T scores; all earlier F/B/S extracted spans are unchanged in the four-arm replay. Primary SO-LUTION versus R(FULL) has seven gains and one harm for Qwen, and six gains and zero harms for R1; both net differences are six percentage points. Uniform R(SOLUTION) versus R(FIRSTBOX) differs by seven and six points. These comparisons are descriptive, post-hoc, and use the same original questions. R(FULL) requires the full actually generated trace and inherits its model workload. Reader CPU time is separate from historical inference wall time, which cannot be presented as an independently measured deployed-reader latency.

## A.10 Prelocked Seed Repeat

The added seed 20261013 executes F/B/S on all original 100 questions per model, with the same sealed checkpoint, prompt, sampler, cue, threshold, cap, endpoint selectors, and six-order cycle. It was fixed before P0 scoring; no choice uses the observed accuracy. Within the new seed, B/F and S/B or pre-cue S/F must have exact emitted-ID/text coupling. Across seeds, question, prompt and model identities match, without requiring sampled trajectories to match. Both complete 300-path plans, successful bounded exits, 104 qualification tests, model pre/post file maps, UUID bindings, every physical attempt and cost, and actual published text/IDs/values pass independent qualification. A label-free guard rechecks all manifested bytes immediately before the original gold reader. A separately written raw-text endpoint and questionbootstrap calculation reproduces all counts, discordances, cost ratios, intervals and exact binomial quantities. These 600 new generation paths add no independent questions.

<table><tr><td>Model</td><td>Policy</td><td>Correct</td><td>Null</td><td>Mean slots</td><td>Cue</td><td>Wall (s)</td></tr><tr><td>Qwen</td><td>FULL</td><td>83</td><td>14</td><td>1179.80</td><td>0</td><td>47.64</td></tr><tr><td>Qwen</td><td>FIRSTBOX</td><td>82</td><td>14</td><td>1158.97</td><td>0</td><td>46.81</td></tr><tr><td>Qwen</td><td>FINALIZE</td><td>91</td><td>0</td><td>1177.18</td><td>16</td><td>48.20</td></tr><tr><td>Qwen</td><td>SOLUTION</td><td>90</td><td>0</td><td>1151.69</td><td>16</td><td>46.44</td></tr><tr><td>R1</td><td>FULL</td><td>80</td><td>12</td><td>1225.07</td><td>0</td><td>36.15</td></tr><tr><td>R1</td><td>FIRSTBOX</td><td>80</td><td>12</td><td>1054.47</td><td>0</td><td>31.16</td></tr><tr><td>R1</td><td>FINALIZE</td><td>87</td><td>0</td><td>1221.95</td><td>12</td><td>35.62</td></tr><tr><td>R1</td><td>SOLUTION</td><td>87</td><td>0</td><td>1048.12</td><td>12</td><td>30.94</td></tr></table>

Table 6: Complete same-seed component analysis. F/B/S are the inherited P0 paths; T is the 200-path later ablation. Mean slots include injected tokens, Cue counts activated questions, and wall is mean actual request time. Later T wall contrasts are confounded by stage; inherited P0 costs are not new costs.

<table><tr><td>Model</td><td>Arm</td><td>P</td><td>R</td><td>Null P</td><td>Null R</td></tr><tr><td>Qwen</td><td>Full</td><td>83</td><td>84</td><td>14</td><td>0</td></tr><tr><td>Qwen</td><td>FirstBox</td><td>82</td><td>83</td><td>14</td><td>0</td></tr><tr><td>Qwen</td><td>Finalize</td><td>91</td><td>91</td><td>0</td><td>0</td></tr><tr><td>Qwen</td><td>Solution</td><td>90</td><td>90</td><td>0</td><td>0</td></tr><tr><td>R1</td><td>Full</td><td>80</td><td>81</td><td>12</td><td>0</td></tr><tr><td>R1</td><td>FirstBox</td><td>80</td><td>81</td><td>12</td><td>0</td></tr><tr><td>R1</td><td>Finalize</td><td>87</td><td>87</td><td>0</td><td>0</td></tr><tr><td>R1</td><td>Solution</td><td>87</td><td>87</td><td>0</td><td>0</td></tr></table>

Table 7: Fixed post-hoc R-v1 sensitivity on the existing 100-question cohort per model. P denotes primary delivered correctness, R the alternative reader, and Null the number of absent numeric values. This is a complete offline sensitivity, not a separately deployed policy or additional test sample.

Qwen F/B/S correctness is 83/83/90, and R1 is 75/75/85. S versus B gains 7/10 answers with zero observed harms. The per-seed paired accuracy intervals are reported in the main text. Physicalslot savings are 0.603% (95% [0.332, 0.868]) and 0.833% ([0.539, 1.113]); measured request-wall savings are 0.660% and 0.900%. B versus F has no correctness discordance in this seed, which does not erase the original Qwen loss or establish endpoint equivalence.

R1 retains one S null despite cue activation: it emits a completed mixed-number box containing a unit, outside the fixed numeric grammar, then reaches native EOS at 2008 slots. This is not missing cue exposure, lack of any visible output, or a reason to change the parser after evaluation. The original elimination of boxed nulls therefore does not generalize even to this second seed. The observed improvement direction is repeated; two fixed seeds do not establish arbitrary-seed robustness. The original primary comparison and intervals remain unchanged, and the two seeds or models are not pooled as independent samples. The fourth-arm component analysis concerns only the original seed. A further prelocked seed was not submitted because the runtime and audit budget would conflict with finalization; this scheduling decision preceded new quality inspection.

Current-repeat forwards total 347377 for Qwen and 353241 for R1. Recorded bounded workercommand walls are 14121.54 and 10621.61 seconds. As with T, these exclude outer orchestration checks and must not be added to nested request timings or relabeled full rental cost. Prior P0/T costs remain prior costs, not new repeat costs.

## A.11 Fixed Readout on the Seed Repeat

The unchanged R-v1 reader also processes all 600 complete, independently qualified new-seed F/B/S traces. An extension scope is sealed before extraction; every label-free span is independently checked before opening gold. The original reader, protocol, numeric grammar and primary scores remain unchanged. A separate recount verifies every model/arm count and paired correctness discordance. This is a post-hoc sensitivity on the same 100 question clusters per model, with no new generation, independent question, confidence interval, or measured deployed-reader wall time.

<table><tr><td>Model</td><td>Policy</td><td>Correct</td><td>Null</td><td>Mean slots</td><td>Cue</td><td>Wall (s)</td></tr><tr><td>Qwen</td><td>FULL</td><td>83</td><td>14</td><td>1181.86</td><td>0</td><td>47.81</td></tr><tr><td>Qwen</td><td>FIRSTBOX</td><td>83</td><td>14</td><td>1149.42</td><td>0</td><td>46.55</td></tr><tr><td>Qwen</td><td>SOLUTION</td><td>90</td><td>0</td><td>1142.49</td><td>17</td><td>46.24</td></tr><tr><td>R1</td><td>FULL</td><td>75</td><td>18</td><td>1290.80</td><td>0</td><td>38.45</td></tr><tr><td>R1</td><td>FIRSTBOX</td><td>75</td><td>18</td><td>1125.49</td><td>0</td><td>33.48</td></tr><tr><td>R1</td><td>SOLUTION</td><td>85</td><td>1</td><td>1116.12</td><td>18</td><td>33.18</td></tr></table>

Table 8: Complete fixed-seed repeat on the same 100 questions per model. Slots include sampled, forced, and emitted EOS tokens. Wall is mean actual request wall; it does not include shared loading or the offline audit. This is a separate secondary seed condition, not another 100-question cohort or a replicated T arm.

<table><tr><td>Model</td><td>Arm</td><td>P</td><td>R</td><td>Null P</td><td>Null R</td></tr><tr><td>Qwen</td><td>Full</td><td>83</td><td>85</td><td>14</td><td>0</td></tr><tr><td>Qwen</td><td>FirstBox</td><td>83</td><td>85</td><td>14</td><td>0</td></tr><tr><td>Qwen</td><td>Solution</td><td>90</td><td>90</td><td>0</td><td>0</td></tr><tr><td>R1</td><td>Full</td><td>75</td><td>77</td><td>18</td><td>0</td></tr><tr><td>R1</td><td>FirstBox</td><td>75</td><td>77</td><td>18</td><td>0</td></tr><tr><td>R1</td><td>Solution</td><td>85</td><td>85</td><td>1</td><td>0</td></tr></table>

Table 9: Unchanged R-v1 readout at the second fixed seed. P, R and Null have the same meaning as Table 7. Every row has 100 questions; adding seeds or policies does not increase independent question count.

R(FULL) and R(FIRSTBOX) return 85 correct Qwen and 77 correct R1 answers, whereas R(SOLUTION) retains 90/85. Relative to R(FULL), actual SOLUTION has 5 gains and 0 harms for Qwen, and 9 gains and 1 harm for R1; the net differences are 5 and 8 points. Uniform R(SOLUTION) versus R(FIRSTBOX) gives the same discordances here. Thus zero primary-comparator harms cannot be transferred to an alternative reader. R-v1 removes all nulls, including the remaining R1 SO-LUTION null, but that last conversion is wrong. Availability without correctness is again insufficient. Only this fixed heuristic is tested; a stronger parser or extraction model could behave differently.