# Does the Unsafe Gradient Survive a Conversation? On the Fragility of Gradient-Based Jailbreak Detection in Multi-Turn Dialogue

Omar Sheta omar.sheta@louisville.edu University of Louisville Louisville, Kentucky, USA

Hadi Masoudi mohammadhadi.masoudi@louisville.edu University of Louisville Louisville, Kentucky, USA

## Abstract

Safety-aligned language models are commonly deployed as multi turn assistants, allowing adversaries to distribute unsafe intent across several user turns rather than expressing it in a single prompt. Existing gradient-based jailbreak detectors, such as GradSafe, were developed for single-prompt inputs. They detect unsafe prompts by measuring the alignment between an input-induced gradient and a fixed unsafe reference direction; however, their efectiveness in multi-turn dialogue remains unclear. We conduct a controlled eval uation of gradient-based jailbreak detection in multi-turn settings. We extend GradSafe with a Context Window Scanner that applies the detector to fixed-size windows of user turns and uses the maximum window score as the conversation-level score. We evaluate diferent window sizes, attack families, benign conversation distri butions, and target models. The results show a substantial diference between synthetic and realistic benign settings. When evaluated against synthetic benign conversations, the detector achieves an ROC-AUC of 0.98 against human-authored multi-turn jailbreaks. When evaluated on WildChat benign conversations, ROC-AUC decreases to 0.76, and a threshold calibrated on synthetic data incor rectly classifies more than 90% of benign conversations as unsafe. Under realistic benign distributions, single-turn windows provide the highest separability, whereas longer windows and accumulated conversation contexts reduce performance. The detector is also sen sitive to the attack-generation method and target model: successful Crescendo-generated attacks receive scores comparable to or lower than benign conversations, and evaluation on Qwen2.5-7B-Instruct yields near-random separability with a diferent optimal window size. These findings show that gradient-based signals can support multi-turn jailbreak detection, but reliable deployment requires cal ibration on realistic benign conversations, short-window scoring, length-aware thresholds, and evaluation across attack types and model architectures.

Rinku Deuja rinku.deuja@louisville.edu University of Louisville Louisville, Kentucky, USA

Minghong Fang minghong.fang@louisville.edu University of Louisville Louisville, Kentucky, USA

CCS Concepts • Security and privacy → Systems security.

## Keywords

Multi-turn jailbreak detection; Gradient-based safety analysis; Large language model security

Omar Sheta, Rinku Deuja, Hadi Masoudi, and Minghong Fang. 2026. Does the Unsafe Gradient Survive a Conversation? On the Fragility of Gradient-Based Jailbreak Detection in Multi-Turn Dialogue. In 3rd Workshop on Large AISystems and Models with Privacy and Safety Analysis (LAMPS ’26), November 15–19, 2026, The Hague, Netherlands. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3846374.3846392

## 1 Introduction

Large language models are increasingly deployed as conversational systems rather than one-shot text generators. This shift changes the security problem. A malicious user no longer has to place the entire forbidden request in a single prompt; instead, they can distribute the objective across multiple apparently benign turns, establish context gradually, and ask a final question whose unsafe meaning depends on the preceding dialogue. Recent multi-turn attacks, including Crescendo-style escalation, iterative jailbreak generation, and human-authored multi-turn jailbreaks, exploit this structure by making intent emerge over time rather than appearing locally in one prompt [2, 16, 21].

Most jailbreak defenses, however, were developed for singleprompt settings and assume that the relevant signal is local. External classifiers such as Llama Guard label a prompt or a response against a safety taxonomy [13], perturbation defenses such as SmoothLLM probe a prompt under random edits [20], and internal-signal methods such as GradSafe [23] read the gradient that a prompt induces when it is paired with a compliance token. Each of these was validated on isolated prompts, so none of them tells us how it behaves when the input is a conversation. This gap motivates a central question: when malicious intent is distributed across a dialogue, does the internal unsafe gradient signature remain detectable, weaken, or disappear?

We study this question by adapting GradSafe [23] to multi-turn inputs through a Context Window Scanner (CWS), which slides a fixed-size window over the user turns, scores each window with GradSafe, and reports the highest score, so that a single high-risk window can flag a conversation. This simple adaptation allows us to examine whether the original unsafe gradient signature remains separable as the amount of context, the realism of the benign data, and the attack family change. We contrast CWS with a full-history accumulated variant and a single-turn baseline, and evaluate all three methods on 537 human-authored multi-turn jailbreak (MHJ) conversations and 2,000 WildChat benign conversations sampled to match the MHJ turn-length distribution.

Our results first show that detector performance depends strongly on the benign data. When benign conversations are synthetic, a three-turn window separates attacks almost perfectly and achieves an ROC-AUC of 0.98; when the same attacks are scored against realistic WildChat conversations, the ROC-AUC falls to 0.76, and accumulated full-history scoring drops by a similar margin. This gap extends beyond ROC-AUC: the operating threshold chosen on synthetic data does not transfer, flagging more than 90% of benign conversations when applied to real trafic and rendering the detector unusable in deployment. A benchmark built on clean synthetic benign prompts therefore overstates both accuracy and false-positive behavior.

When evaluated against realistic benign conversations, the study also overturns the intuition that more context helps. The singleturn baseline is the strongest setting overall, followed by a twoturn window, and separability declines steadily as the window grows or as the full history is accumulated. Length-conditioned and tactic-level analyses show the same trend: local windows remain strongest even on long conversations, and accumulated context is consistently weaker. We attribute this pattern to gradient dilution, because concatenating benign or unrelated turns adds noise to the gradient and pulls it away from the unsafe reference direction, so longer context masks rather than clarifies the signal.

The remaining findings concern generalization, and both are cautionary. When we replace human-authored attacks with automated Crescendo conversations, run on 200 HarmBench objectives and yielding 149 confirmed successes at a 74.5% attack success rate, discriminative performance deteriorates substantially against the same WildChat benign set: short windows approach random performance, and longer windows rank successful attacks below ordinary benign conversations. Because Crescendo was not optimized against the gradient score, this pattern reflects a structural mismatch rather than adaptivity: a successful context-building attack keeps its user turns close to benign gradients until the model generates the unsafe response. The same sensitivity appears across models. On Qwen2.5-7B-Instruct, separability is near random and the preferred window size reverses, with accumulated scoring becoming the strongest setting. This result shows that the behavior of a gradient detector depends on the model on which it is calibrated. Section 6.2 analyzes the Crescendo mechanism in detail.

We summarize our contributions as follows.

• We recast a single-turn internal-gradient detector, GradSafe, as a multi-turn detector through a simple Context Window Scanner, and use it to test directly whether the unsafe gradient signature survives when intent is distributed across a dialogue.

• We expose a large gap between synthetic and realistic benign calibration, in which synthetic benign data makes the detector appear nearly solved while realistic WildChat trafic both lowers separability sharply and turns a synthetic-tuned threshold into severe over-blocking.

• We show that shorter windows beat longer and accumulated context under realistic benign trafic, and we trace the efect to dilution of a sparse unsafe signal by benign dialogue.

• We show that the detector does not generalize across attack families or architectures, since successful Crescendo attacks and a Qwen2.5-7B-Instruct target each collapse separability and the latter reverses the window-size preference.

## 2 Background and Related Work

## 2.1 Single-Turn Jailbreak Attacks

A jailbreak attack tries to make a safety-aligned model produce harmful or prohibited content. The earliest automated attacks worked at the level of a single prompt and showed that adversarial sufixes generalize across models. GCG [28] searches for such a sufix with white-box gradients, maximizing the probability of an afirmative reply to a harmful request, and the sufixes it finds transfer to both open and proprietary systems. Tree of Attacks with Pruning [18] lowers the query cost by letting an attacker model expand and prune candidate prompts in a tree, so it reaches comparable success rates with far fewer queries than GCG. Studies of deployed systems show that the real threat surface is wider than optimized sufixes, because Shen et al. [22] catalog role-play, privilege-escalation, and prompt-injection patterns that ordinary users write without any gradient access. To make such attacks comparable, HarmBench [17] and JailbreakBench [1] standardize the attacks, the target behaviors, and the refusal metrics, which is what allows reproducible evaluation across methods.

## 2.2 Multi-Turn Jailbreak Attacks

Multi-turn attacks move the threat from one crafted prompt to a coordinated dialogue. PAIR [2] has an attacker model refine its jailbreak prompt through a feedback loop with a black-box target and succeeds in about twenty queries, treating each refinement as a step in an adversarial conversation. Crescendo [21] instead escalates a sequence of innocuous requests, so that no single turn looks unsafe and the harmful content is only reached after enough shared context has been built. The MHJ dataset [16] contains 537 humanwritten multi-turn jailbreak conversations organized by a tactic taxonomy spanning obfuscation, hidden-intention streamlining, injection, output formatting, request framing, echoing, and direct requests. It demonstrates that skilled humans exploit conversational coherence in ways that single-turn automation does not reproduce. Greshake et al. [9] widen the surface further to indirect prompt injection, where instructions hidden in retrieved third-party content hijack the model inside an ongoing session without any explicit attacker turn. Taken together, these attacks argue that a detector must reason over dialogue history, since scoring a single prompt cannot capture intent that has been split across turns or planted in retrieved context.

## 2.3 Jailbreak Defenses

Existing defenses difer mainly in where they intervene. Input-level defenses act before the model runs: SmoothLLM [20] flags an attack when random mutations of the prompt change the model behavior, and perplexity filtering [14] uses the fact that GCG-style sufixes read as anomalously high perplexity, which gives a cheap signal without touching the weights. Classifier-level defenses add a second model: Llama Guard [13] treats safety as taxonomy classification over prompts and responses, and Llama Guard 3 ships with the Llama 3 family [8] under a refreshed taxonomy. Internal signal defenses instead read the target model itself. GradSafe [23] observes that an unsafe prompt paired with a compliance token induces a characteristic gradient on safety-critical parameters, and scores a prompt by its cosine similarity to a fixed unsafe reference direction; SafeQuant [19] pursues the same idea with quantized gradients, and Gradient Cuf [10] separates attacks by the gradi ent norm under a refusal-eliciting sufix without a stored reference. Gradient-Controlled Decoding [4] extends the single-anchor design to a pair of acceptance and refusal anchors and couples detection with a first-token mitigation step. Alignment-stage defenses mod ify model parameters during safety alignment or fine-tuning to improve robustness before deployment [3, 6, 11, 24]. They are com plementary to our inference-time setting, which examines whether unsafe intent remains detectable when distributed across dialogue turns. Our work stays within this internal-gradient family but asks the multi-turn question it has not addressed, namely whether the unsafe gradient remains readable when intent is distributed across a conversation. We do not use Gradient Cuf or Gradient-Controlled Decoding as baselines, because their refusal-loss and first-token mitigation protocols difer from our prompt-only, pre-generation setting, and we cite them only to place our study in context. Remark that Jailbreak defense and forensic localization address diferent stages of the security process [5, 7, 25, 26]. A jailbreak defense identifies or prevents unsafe requests before the model produces harmful output. In contrast, forensic localization is performed after a safety failure to identify its cause.

## 3 Threat Model

Attacker: The attacker reaches an aligned target model through a chat interface and controls a sequence of user turns, across which the malicious intent is deliberately distributed so that any one turn may read as benign, ambiguous, or incomplete. We grant the attacker awareness that a safety detector might be present, but not knowledge of the exact unsafe reference direction or the operating threshold.

Defender: The defender aims to catch an unsafe conversation before the model emits harmful output. For GradSafe-based scoring, the defender has the white-box access needed to compute gradients on safety-critical parameters. The detector is prompt-only, so it scores the user turns before generation and never sees the assistant responses. We choose this pre-generation setting on purpose, because once harmful content is produced it may already be seen, logged, cached, or copied, so intercepting the conversation before the first generated token is a stronger point of control. The point is sharper still in agentic and tool-use deployments, where a single response can launch irreversible actions such as external API calls, code execution, file writes, or downstream tool invocations, before any response-side moderator has a chance to act. The cost of this choice is that the detector cannot rely on evidence that only the response would reveal.

Scope: We study the separability and threshold behavior of the detector, not a complete blocking system, and we do not claim robustness against an attacker that optimizes directly against the gradient score. In principle, an attacker who knew the unsafe reference direction could steer a dialogue to lower its cosine similarity while keeping the harmful goal, but in a multi-turn setting that optimization would still have to preserve coherent context and elicit unsafe behavior, and it lies outside our scope. For fairness, the Llama Guard baseline is run under the same prompt-only, userturn-only protocol, which is narrower than its intended use as a prompt and response safeguard.

## 4 Our Method

We build our detector by reusing the scoring rule of GradSafe [23] unchanged and adding only the machinery needed to apply it to a conversation. The separation is deliberate, because it keeps the single-turn scoring rule fixed so that any change in behavior can be traced to how context is assembled rather than to a new detector design. We first restate the scoring rule and then describe the windowing procedure that turns it into a multi-turn detector.

## 4.1 GradSafe Scoring

GradSafe builds on the observation that a harmful request leaves a consistent trace in the gradient of a safety-aligned model. When an input is paired with a short compliance token such as “Sure,” an unsafe input drives a small set of safety-critical parameters in a direction that is stable across diferent unsafe prompts and separable from the direction that benign inputs induce. GradSafe turns this into a detector by fixing that direction once, as an unsafe reference signature $\mathtt { g } _ { \mathrm { u n s a f e } }$ averaged over a set of unsafe seed prompts, and by restricting attention to the safety-critical parameters $\theta _ { S }$ on which the separation is sharpest. Writing � for the detector input and � for the compliance token, the score is the cosine similarity between the induced gradient and this reference signature,

$$
S ( x ) = \cos \left( \nabla _ { \theta _ { S } } \mathcal { L } ( x \oplus t , t ) , \mathbf { g } _ { \mathrm { u n s a f e } } \right) ,\tag{1}
$$

so a larger score means closer alignment with the unsafe reference. The score is a single scalar and needs one backward pass rather than any text generation, which is what allows it to act as a pregeneration filter. In the implementation the loss is taken over the full compliance sufix that follows the prompt, which is the separator marker followed by “Sure,” and the reference signature and every evaluated score use this same sufix so that the comparison stays consistent.

## 4.2 Context Window Scanner

A conversation is a sequence of turns, while the scoring rule above consumes a single input, so a multi-turn detector must decide which text to score. The two obvious choices are both unsatisfying. Scoring only the most recent user turn discards the context that a multi-turn attack is designed to accumulate, whereas scoring the whole history at once blends the attack turn with every benign turn around it. The

Context Window Scanner (CWS) spans these two choices with one parameter, the window size �, and keeps everything else minimal.

Because the defender is prompt-only, CWS first drops the assistant turns and keeps the user turns,

$$
H = \langle h _ { 1 } , h _ { 2 } , \ldots , h _ { m } \rangle .\tag{2}
$$

It then slides a right-anchored window of size � that ends at each user turn,

$$
\begin{array} { r } { x _ { t , W } = \operatorname { j o i n } _ { \delta } ( h _ { \operatorname* { m a x } ( 1 , t - W + 1 ) } , \dots , h _ { t } ) , } \end{array}\tag{3}
$$

for $t = 1 , \ldots , m _ { \mathrm { { \scriptsize ~ ; ~ } } }$ , so the window is causal and could be evaluated online as each turn arrives, and the first� −1 windows are shorter simply because no earlier turns exist. Adjacent turns inside a window are joined by a fixed delimiter �, realized as a blank-line-separated horizontal rule, which keeps the turn boundaries explicit in the scored text. GradSafe scores each window, and the conversation takes the largest window score,

$$
S _ { \mathrm { C W S } , W } ( H ) = \operatorname* { m a x } _ { 1 \leq t \leq m } S ( x _ { t , W } ) .\tag{4}
$$

Max pooling is a design choice rather than a convenience, because it treats the windows as a logical OR in which one high-risk window is enough to flag the conversation, which matches a deployment where a single successful jailbreak turn already counts as a failure.

The window size directly controls how much context enters each score. At � = 1 the scanner reduces to scoring each user turn in isolation, and as � grows it incorporates progressively more history, until a single window over all turns recovers full-history accumulation. Thus, the single-turn and accumulated baselines represent the two endpoints ofthe context-length spectrum. Scoring a conversation of� user turns costs � backward passes, one per window position and independent of � , so the procedure remains inexpensive; the main risk of a larger window is mixing benign turns into the scored text. We intentionally keep CWS simple. It is not proposed as a complete defense but as a faithful extension of a single-turn detector to the multi-turn setting, and its simplicity allows the experiments to attribute each efect to context length, benign-data realism, or attack family rather than to a more elaborate detector.

## 5 Experimental Setup

Attack data: The attack set is the 537 MHJ conversations [16] taken from the bundled HarmBench behavior file, where each row is a multi-turn user sequence tied to a harmful behavior or tactic.

We add a second, automated attack family by running Crescendo on 200 HarmBench objectives. The pipeline uses PyRIT Crescendo with a local 70B attacker model against the Llama 3.1 8B target endpoint, capped at 10 turns, 10 backtracks, and a 5-second gap between objectives. Rather than keyword matching, success is judged by a calibrated scorer that rates each target response on a scalar scale and converts the rating to a Boolean label at a threshold of 0.9. All 200 objectives completed, of which 149 were confirmed successful, giving a 74.5% attack success rate, and the successful conversations average 3.13 user turns, with median 3, minimum 1, and maximum 7. The other 51 rows failed at runtime because the attacker prompt exceeded the 8,192-token endpoint limit, and we do not count them as attack failures; since no completed target transcript survives for them, the all-attempt set stores only the original HarmBench objective as one user prompt.

For all main claims, we use the 149 confirmed successes as the successful-attack subset and report the 200-row all-attempt set only for provenance. This distinction is important because each of the 51 fallback rows reduces to a single-turn, explicit harmful request, which produces a strong localized unsafe gradient and bypasses the context dilution under study. Including these rows increases the single-turn ROC-AUC from a near-random 0.5018 on confirmed successes to 0.5856 on all attempts. We therefore exclude them and restrict our conclusions to genuine multi-turn context-building attacks.

Benign data: Because false positives decide whether a promptonly detector is deployable, we calibrate on realistic trafic rather than clean synthetic prompts. The primary benign benchmark is a stratified sample of 2,000 WildChat [27] conversations, a corpus of real interaction logs whose in-the-wild trafic already mixes genuine jailbreaks with benign but safety-relevant discussion [15]. The sampler first drops one-turn conversations and then matches the MHJ turn-length distribution over the remainder, yielding 740 short conversations of two or three user turns, 680 medium conversations, and 580 long conversations.

Synthetic benign data: We also keep the early synthetic benign benchmark from the initial evaluation, 183 benign multi-turn conversations generated with Gemini 2.5 Pro over ordinary assistant topics such as cooking, travel, hobbies, general knowledge, and short writing. These are mostly two turns long, uniformly polished and on-topic, and free of the emotionally intense, adversariallooking, or policy-sensitive content that fills real trafic. As a standin for the clean synthetic prompts that earlier single-turn evaluations rely on, it serves to quantify the evaluation gap rather than to support any deployment claim. Two properties make these conversations easy to separate from attacks: their clean content and their short, unmatched length distribution. Consequently, this benchmark overstates performance along both realism and length axes, so we treat the length-matched WildChat benchmark as the realistic reference throughout.

Models: The main experiments use Llama 3.1 8B Instruct [8], and we add Qwen2.5-7B-Instruct [12] as a cross-architecture target. The Qwen2.5 reference signature is built with the same procedure as the Llama 3.1 signature, using a separately optimized safetycritical parameter mask over the same unsafe seed prompts, and the cross-model results appear in Section 6.2.

Baselines: We compare CWS against the two extremes it interpolates and against an external classifier. The single-turn max-overturn baseline scores each user turn on its own and keeps the highest score, which is the� = 1 point and the reference for whether added context helps at all. The accumulated baseline concatenates every user turn into one input and scores it once, which tests whether the full history preserves or dilutes the unsafe signature. Finally, we run Llama Guard 3 (meta-llama/Llama-Guard-3-8B) with its oficial chat template under the same prompt-only, user-turn-only protocol, scoring and max-pooling each user-turn window for � = 3 and scoring the concatenated user turns once for the accumulated mode, and we never include assistant responses or harmful outputs.

![](images/0ed696fe7aabe579937e8095105564ddb9525692daf4f2e80587682d1d6d51a1.jpg)

Table 1: Synthetic benign data overstates detector performance. Threshold transfer applies the synthetic best-F1 threshold to the matching WildChat setting.
<table><tr><td>Benign set</td><td>Mode</td><td>AUC</td><td>Transfer FPR</td><td>95% CI</td><td>AP</td><td>F1</td></tr><tr><td>Synthetic</td><td>W = 3</td><td>0.9836</td><td>0.0874</td><td>[0.9750, 0.9909]</td><td>0.9949</td><td>0.9740</td></tr><tr><td>WildChat</td><td>W = 3</td><td>0.7614</td><td>0.9085</td><td>[0.7379, 0.7843]</td><td>0.4278</td><td>0.5260</td></tr><tr><td>Synthetic</td><td>Accum.</td><td>0.9777</td><td>0.1530</td><td>[0.9675, 0.9864]</td><td>0.9929</td><td>0.9576</td></tr><tr><td>WildChat</td><td>Accum.</td><td>0.7376</td><td>0.9065</td><td>[0.7130, 0.7614]</td><td>0.4043</td><td>0.5039</td></tr></table>

Metrics: We report ROC-AUC, average precision (AP), the best-F1 threshold, best F1, and the true positive rate at a 10% false positive rate. AP follows the project convention of averaging precision at each positive rank after scores are sorted in descending order, which matches the artifact script and can difer slightly from library implementations under ties. To test portability, we also transfer the synthetic-benchmark threshold to real WildChat trafic and measure the resulting false-positive rate. Confidence intervals are nonparametric 95% bootstrap intervals with 2,000 resamples; attack and benign conversations are resampled separately, and the window-ablation deltas use paired resampling over the shared rows. Rows whose scoring failed are excluded from the metrics and reported as errors where present.

Hardware and determinism: Gradient scoring runs in bfloat16 on NVIDIA H100 GPUs with automatic device mapping. The stratified WildChat sampler uses seed 13, the benign-subspace pilot uses seed 17, the bootstrap intervals use 2,000 resamples with seed 20260514, and the paired window-diference intervals use seed 20260515. The artifact script records the provenance of every table and figure, and code together with redacted data artifacts will be released upon acceptance, subject to the review policy and to responsible handling of harmful transcripts.

## 6 Experimental Results

## 6.1 Synthetic versus Real Benign Calibration

The clearest result of the study is that the benign data alone decides how strong the detector appears, and Table 1 makes the gap concrete. Against synthetic benign conversations both the three-turn window and accumulated scoring clear 0.97 ROC-AUC, but against the real stratified WildChat set the same two settings fall to roughly 0.76 and 0.74, with average precision dropping in step.

The threshold transfer numbers matter even more than the AUC drop. A three-turn threshold that flags only 8.74% of synthetic benign conversations flags 90.85% of real ones, and the accumulated threshold behaves the same way, so the synthetic benchmark does not merely inflate a score, it hands the practitioner an operating point that would reject almost all legitimate trafic.

## 6.2 Main Detector Comparison

On the realistic benchmark the ordering of the methods is clear, and Table 2 reports it. The strongest GradSafe setting is the single-turn window at 0.8101 ROC-AUC, the two-turn window comes next and still beats three turns, three turns remain usable but weaker, and accumulated full-history scoring is weakest of all, so shorter windows win across the board.

Llama Guard 3 sits below 0.5 AUC in both modes, which we do not read as a general weakness of Llama Guard but as a mismatch of task. Taxonomy-based prompt classification is not built to catch intent that is still hidden before generation, so its failure here confirms that prompt-only gradient scoring occupies a diferent point in the pipeline from response-side moderation.

Table 2: Main results on MHJ attacks versus stratified Wild-Chat benign conversations.
<table><tr><td>Method</td><td>ROC-AUC</td><td>95% CI</td><td>AP</td><td>Best F1</td><td>TPR@10%FPR</td></tr><tr><td>GradSafe W = 1</td><td>0.8101</td><td>[0.7895, 0.8311]</td><td>0.5135</td><td>0.5749</td><td>0.5084</td></tr><tr><td>GradSafe W = 2</td><td>0.7873</td><td>[0.7654, 0.8092]</td><td>0.4647</td><td>0.5536</td><td>0.4227</td></tr><tr><td>GradSafe W = 3</td><td>0.7614</td><td>[0.7379, 0.7843]</td><td>0.4278</td><td>0.5260</td><td>0.4004</td></tr><tr><td>GradSafe Accum.</td><td>0.7376</td><td>[0.7130, 0.7614]</td><td>0.4043</td><td>0.5039</td><td>0.3203</td></tr><tr><td>Llama Guard W = 3</td><td>0.4528</td><td>[0.4266, 0.4796]</td><td>0.2118</td><td>0.3494</td><td>0.0838</td></tr><tr><td>Llama Guard Accum.</td><td>0.4409</td><td>[0.4132, 0.4692]</td><td>0.2036</td><td>0.3494</td><td>0.0615</td></tr></table>

![](images/13876d1b2cf34868ed536ade03ae79afef47f4b7cfac059a93e7189d6c07aee7.jpg)  
Figure 1: ROC curves for key methods on the real WildChat benchmark.  
Figure 2: Score distributions for representative settings. Real benign scores overlap heavily with attack scores.

WildChat contamination: A manual check of the stratified Wild-Chat sample turned up conversations that carry jailbreak templates, role-play safety tests, or other adversarial-looking text. We keep these rows in the benchmark rather than remove them, because they are a genuine part of real trafic and a genuine source of false-positive pressure for a prompt-only detector, so the unfiltered distribution, not a sanitized subset, is what drives our headline numbers.

Table 3: Window ablation on MHJ versus WildChat. Short local windows beat long windows and accumulated context. Deltas are paired bootstrap diferences against $W = 3 .$
<table><tr><td>Method</td><td>AUC</td><td>95% CI</td><td> $\Delta \mathrm { v s } W = 3$ </td><td>95% CI</td><td>AP</td><td>F1</td></tr><tr><td>W = 1</td><td>0.8101</td><td>[0.7895, 0.8311]</td><td>0.0487</td><td>[0.0329, 0.0643]</td><td>0.5135</td><td>0.5749</td></tr><tr><td>W = 2</td><td>0.7873</td><td>[0.7654, 0.8092]</td><td>0.0258</td><td>[0.0155, 0.0365]</td><td>0.4647</td><td>0.5536</td></tr><tr><td>W = 3</td><td>0.7614</td><td>[0.7379, 0.7843]</td><td>0.0000</td><td>[0.0000, 0.0000]</td><td>0.4278</td><td>0.5260</td></tr><tr><td>W = 4</td><td>0.7569</td><td>[0.7325, 0.7799]</td><td>-0.0045</td><td>[-0.0103, 0.0011]</td><td>0.4150</td><td>0.5207</td></tr><tr><td>W = 5</td><td>0.7503</td><td>[0.7261, 0.7734]</td><td>-0.0111</td><td>[-0.0181, -0.0039]</td><td>0.4057</td><td>0.5150</td></tr><tr><td>W =7</td><td>0.7362</td><td>[0.7119, 0.7594]</td><td>-0.0252</td><td>[-0.0347, -0.0159]</td><td>0.3849</td><td>0.4993</td></tr><tr><td>Accum.</td><td>0.7376</td><td>[0.7130, 0.7614]</td><td>-0.0238</td><td>[-0.0378, -0.0095]</td><td>0.4043</td><td>0.5039</td></tr></table>

![](images/21e93f15665a6afac56c314602d3e19133cdb85bb621c4c2035bbb0fe436916d.jpg)  
Figure 3: Length-conditioned AUC. Short windows are strongest even on long conversations, while accumulated context stays weaker.

Window size ablation: Sweeping the window size confirms the trend seen above, and Table 3 lays it out: AUC falls monotonically from one turn to seven, and accumulated scoring lands near the seven-turn window and well behind the short ones.

The direction is the opposite of what one would expect if multiturn detection needed more context. Here a short window isolates the local unsafe cue better than a long one, and the efect is statistically clean, since the paired interval for one turn against three turns is strictly positive at $\Delta \mathrm { A U C } = 0 . 0 4 8 7$ with 95% CI [0.0329, 0.0643]. The most consistent explanation is dilution, because each extra benign or unrelated turn adds gradient that is not aligned with the unsafe reference and therefore washes the signal out.

Length-conditioned behavior: Because dilution should depend on how much context is added, we also break the results down by conversation length in Figure 3. Short windows dominate globally and, tellingly, stay dominant on the longest conversations: for conversations with seven or more user turns, the two-turn and one-turn windows reach 0.8625 and 0.8622 AUC while three turns reach 0.8370, and accumulated scoring trails at 0.7411. This resolves the role of context in a practical way. A long conversation does not call for scoring its whole history; if anything, the extra length only creates more room for dilution, so a deployed detector should pair short-window scoring with length-aware threshold calibration instead of assuming that accumulating context is the safe default.

Table 4: Tactic-level AUC. Each tactic subset is scored against the full WildChat benign distribution.<sup>1</sup>
<table><tr><td>Tactic</td><td>n</td><td> $W = 1$ </td><td> $W = 3$ </td><td> $W = 7$ </td></tr><tr><td>Direct Request</td><td>146</td><td>0.7616</td><td>0.7277</td><td>0.7297</td></tr><tr><td>Hidden Intention Streamline</td><td>109</td><td>0.8006</td><td>0.7538</td><td>0.7014</td></tr><tr><td>Injection</td><td>32</td><td>0.9044</td><td>0.8727</td><td>0.8648</td></tr><tr><td>Obfuscation</td><td>156</td><td>0.7905</td><td>0.7235</td><td>0.6948</td></tr><tr><td>Output Format</td><td>23</td><td>0.8895</td><td>0.8641</td><td>0.8273</td></tr><tr><td>Request Framing</td><td>68</td><td>0.8980</td><td>0.8388</td><td>0.8027</td></tr></table>

Table 5: Attack-family sensitivity of GradSafe against the stratified WildChat benign set. Crescendo-success is the 149 confirmed successes; Crescendo-all adds 51 single-prompt objective-fallback rows.
<table><tr><td>Attack family</td><td>n</td><td> $W = 1$ </td><td> $W = 2$ </td><td> $W = 3$ </td><td> $W = 7$ </td><td>Accum.</td></tr><tr><td>MHJ</td><td>537</td><td>0.8101</td><td>0.7873</td><td>0.7614</td><td>0.7362</td><td>0.7376</td></tr><tr><td>Crescendo-all</td><td>200</td><td>0.5856</td><td>0.4891</td><td>0.4474</td><td>0.4480</td><td>0.4723</td></tr><tr><td>Crescendo-success</td><td>149</td><td>0.5018</td><td>0.3654</td><td>0.3050</td><td>0.3008</td><td>0.3161</td></tr></table>

Tactic-level behavior: The advantage of short windows also holds tactic by tactic, as Table 4 shows. The one-turn window is strongest on every nontrivial tactic group, including direct requests, hiddenintention streamlining, injection, obfuscation, output-format attacks, and request framing. This result does not imply that a single turn contains the entire malicious objective. Instead, it suggests that the signature responds most strongly to the local turns closest to the unsafe goal, while the surrounding context tends to dilute that signal rather than sharpen it.

Sensitivity to crescendo attacks: The preceding results are measured on human-authored MHJ attacks and do not generalize to automated Crescendo attacks, as Table 5 and Figure 4 show against the same WildChat benign set. Whereas MHJ achieves AUCs of 0.8101 and 0.7614 at one and three turns, respectively, successful Crescendo reduces the one-turn AUC to 0.5018 and pushes the three-turn, seven-turn, and accumulated settings well below 0.5.

This reduction is not statistical variation around random performance. The three-turn interval for successful Crescendo lies entirely below random ranking at a 95% CI of [0.2608, 0.3503], and even the one-turn window is statistically indistinguishable from random at a 95% CI of [0.4482, 0.5530].

A below-random AUC should not be interpreted as a useful inverted detector. Under this reference signature and prompt-only protocol, the user turns of a successful Crescendo attack receive lower scores than ordinary WildChat trafic. Thus, the result is more serious than a simple reduction in discriminative performance: detector rankings depend on the attack family. Because Crescendo was never optimized against GradSafe, this pattern is not evidence of adaptivity; rather, it indicates that an automated context-building attack may avoid producing the local unsafe cues that MHJ attacks tend to contain.

The same protocol mismatch shows up for Llama Guard on Crescendo. Table 6 compares the two prompt-only detectors on the successful subset, where Llama Guard reaches only 0.1478 AUC at three turns and 0.1943 when accumulated. This again says nothing about how Llama Guard would fare if it could read the harmful response; it says that, before generation, both taxonomy classification and gradient scoring find little to hold onto in the user turns of this attack family.

![](images/e461d66c2fb49805ba4c1726471ee4e31cb60405b85e6a1b486dd9e8b50bd0cd.jpg)  
Figure 4: Attack-family AUC. GradSafe separates MHJ from WildChat, but successful Crescendo conversations rank near or below benign trafic.

Table 6: Llama Guard 3 on Crescendo under the same promptonly, user-turn-only protocol.
<table><tr><td>Subset</td><td>Method</td><td>AUC</td><td>AP</td><td>TPR@10%FPR</td></tr><tr><td>Crescendo-all</td><td>W = 1</td><td>0.2314</td><td>0.0599</td><td>0.0200</td></tr><tr><td>Crescendo-all</td><td>W = 3</td><td>0.2614</td><td>0.0616</td><td>0.0400</td></tr><tr><td>Crescendo-all</td><td>Accum.</td><td>0.3435</td><td>0.0738</td><td>0.0950</td></tr><tr><td>Crescendo-success</td><td> $W = 1$ </td><td>0.1389</td><td>0.0425</td><td>0.0000</td></tr><tr><td>Crescendo-success</td><td>W = 3</td><td>0.1478</td><td>0.0436</td><td>0.0000</td></tr><tr><td>Crescendo-success</td><td>Accum.</td><td>0.1943</td><td>0.0436</td><td>0.0000</td></tr></table>

Why successful crescendo attacks evade gradient scoring: The below-random scores, for example, 0.3050 at three turns, indicate more than poor separability: the user turns in successful Crescendo attacks receive lower scores than ordinary benign conversations. To understand this behavior, we examined successful Crescendo conversations. Attackers rarely state the final harmful request directly. Instead, they begin with an informational, historical, fictional, or policy-analysis question and gradually narrow the context until the model supplies unsafe content in its response. We do not reproduce full examples because the relevant evidence lies in the prompt-side structure rather than in operational detail.

This analysis suggests four reasons why a successful Crescendo turn can evade a signature calibrated on single-turn jailbreaks.

• Neutral, formal phrasing. The turns use formal and structured language, without the command-like wording, high-perplexity sufixes, or explicit jailbreak markers that the reference signature is designed to capture.

• No access to assistant responses. The unsafe content typically appears in the model response, but prompt-only scoring never observes assistant turns.

• Contextual dilution. Early Crescendo turns are benign in isolation, so combining them with the turn carrying the harmful intent dilutes the small unsafe cue that remains in the prompt.

Table 7: Qwen2.5-7B-Instruct window ablation on MHJ versus stratified WildChat. Transfer FPR applies the Llama 3.1 synthetic threshold of 0.0205. Settings cluster near random and accumulated scoring is best, reversing the Llama 3.1 ordering.
<table><tr><td>Method</td><td>AUC</td><td>95%CI</td><td>AP</td><td>TPR@10%FPR</td><td>Transfer FPR</td></tr><tr><td> $W = 1$ </td><td>0.5613</td><td>[0.5348, 0.5893]</td><td>0.2461</td><td>0.1453</td><td>0.8290</td></tr><tr><td> $W = 2$ </td><td>0.5636</td><td>[0.5368, 0.5891]</td><td>0.2511</td><td>0.1434</td><td>0.7510</td></tr><tr><td> $W = 3$ </td><td>0.5730</td><td>[0.5467, 0.5976]</td><td>0.2478</td><td>0.1285</td><td>0.6930</td></tr><tr><td> $W = 4$ </td><td>0.5653</td><td>[0.5393, 0.5916]</td><td>0.2437</td><td>0.1248</td><td>0.6640</td></tr><tr><td> $W = 5$ </td><td>0.5627</td><td>[0.5366, 0.5883]</td><td>0.2394</td><td>0.1210</td><td>0.6560</td></tr><tr><td> $W = 7$ </td><td>0.5545</td><td>[0.5279, 0.5809]</td><td>0.2336</td><td>0.1006</td><td>0.6430</td></tr><tr><td>Accum.</td><td>0.5908</td><td>[0.5669, 0.6162]</td><td>0.2458</td><td>0.0968</td><td>0.3880</td></tr></table>

• Overlap with realistic benign trafic. Genuine WildChat conversations can contain emotionally intense, adversarial-looking, or policy-sensitive discussion, which may induce a stronger safety gradient than the polished Crescendo prompts.

Cross-model sensitivity: To see whether these conclusions are tied to one model, we repeat the evaluation on Qwen2.5-7B-Instruct with a separately built reference signature, scoring the same 537 MHJ attacks and 2,000 WildChat conversations, with no rows dropped for errors. Table 7 reports the full sweep, and two things stand out.

First, separability nearly disappears. The best Qwen setting is accumulated scoring at an AUC of 0.5908, while every windowed setting lies between 0.5545 and 0.5730, only slightly above random; by contrast, every Llama 3.1 8B setting exceeds 0.73. Second, the ordering reverses. On Llama, AUC declines monotonically from 0.8101 at one turn to 0.7362 at seven turns, with accumulated scoring performing worst. On Qwen, accumulated scoring performs best, and the one-turn window at 0.5613 is among the weakest settings. Thus, the gradient-dilution explanation observed for Llama does not extend to Qwen. Threshold portability also fails: transferring the Llama 3.1 synthetic threshold of 0.0205 to Qwen produces falsepositive rates ranging from 38.8% for accumulated scoring to 82.9% for one-turn scoring. This result confirms that thresholds do not transfer reliably across either architectures or benign distributions.

The pattern inside the Qwen sweep reinforces this reversal. The windowed settings stay nearly flat between 0.5545 and 0.5730 rather than declining monotonically as they do on Llama 3.1, and the transfer FPR falls as the window grows, which is again the reverse of Llama 3.1, where every windowed setting produced a similarly high rate.

Together, these results show that the behavior ofa gradient detector depends on the model on which it is calibrated rather than being a fixed property of the method. The near-random performance on Qwen is consistent with a diferently structured safety-relevant gradient subspace under the same signature procedure. We do not claim that Qwen is fundamentally harder or that the Llama results establish a robust method, because each result reflects one signature realization against one benign benchmark. The key lesson is that thresholds do not transfer across models and must be recalibrated for each target.

Analysis of simple extensions: Before concluding that the performance drop on realistic benign data is dificult to address, we evaluated three intuitive remedies. Table 8 shows that none outperforms the standard three-turn GradSafe baseline. We include these results because each tests a diferent hypothesis about the source of the failure, and the results do not support these hypotheses.

Table 8: Intuitive extensions. None beats the plain $W = 3$ baseline on the real WildChat benchmark or its pilot subsets.
<table><tr><td>Pilot</td><td>Attack n</td><td>Benign n</td><td>AUC</td><td>AP</td></tr><tr><td>Baseline W = 3</td><td>537</td><td>2000</td><td>0.7614</td><td>0.4278</td></tr><tr><td>Contrastive raw delta</td><td>537</td><td>2000</td><td>0.6672</td><td>0.3094</td></tr><tr><td>Contrastive projection delta</td><td>537</td><td>2000</td><td>0.7174</td><td>0.3481</td></tr><tr><td>Contrastive ratio</td><td>537</td><td>2000</td><td>0.6754</td><td>0.3149</td></tr><tr><td>Subspace pilot baseline</td><td>120</td><td>120</td><td>0.7793</td><td>n/a</td></tr><tr><td>Subspace residual cosine</td><td>120</td><td>120</td><td>0.7042</td><td>n/a</td></tr><tr><td>Multi-turn ref baseline</td><td>20</td><td>20</td><td>0.7812</td><td>n/a</td></tr><tr><td>Multi-turn reference</td><td>20</td><td>20</td><td>0.7512</td><td>n/a</td></tr></table>

The pattern is consistent across all three remedies. In the contrastive variants, subtracting or normalizing a contextual baseline performs worse than the original baseline, indicating that the subtraction removes useful unsafe signal together with benign context. In the subspace variant, estimating and removing benign directions performs worse than its pilot baseline, indicating that benign conversational structure overlaps with the attack signal it was intended to preserve. Building a multi-turn reference direction changes the reference meaningfully but still does not outperform the baseline, showing that re-anchoring alone cannot resolve the overlap introduced by realistic benign trafic. In short, the failure is not attributable to a simple post-hoc adjustment.

## 7 Discussion

The signal remains detectable, but its practical utility is limited: On realistic trafic, one-turn and two-turn scoring outperform random ranking by a wide margin, and tactic-level AUCs are strong for injection, output-format, and request-framing attacks. However, the results do not indicate that the problem is solved: even the best setting remains far from a deployable operating point once realistic benign conversations are included.

Limited generalization across attack families: Our results on Crescendo show that the short-window signal observed on MHJ does not transfer reliably across multi-turn jailbreak families. On MHJ, short-window gradients remain useful despite the realisticbenign-data gap; on successful Crescendo attacks, however, the same scores approach or fall below random performance, and longer windows rank attacks below benign trafic. The most likely explanation is that a context-building attack keeps its user turns close to benign gradients until the model, rather than the user, generates the unsafe content.

More context can hurt: The ablation is best explained by dilution: a longer window contains more benign or unrelated content, and the gradients induced by that content move the aggregate away from the unsafe reference. Therefore, accumulated full-history scoring is not a safe default even for a long-context model. The fact that a single-turn baseline outperforms every context-aware setting challenges the assumption that more history is automatically safer. This result does not make context useless; rather, it shows that naive concatenation can average a sparse attack signal with benign noise. It motivates context-conditional gating or asymmetric scoring that isolates safety-critical parameters without incorporating this noise.

Realistic benign calibration is not optional: Synthetic benign data produced both very high scores and overly optimistic thresholds. Real conversations include ambiguous requests, technical instructions, role-play, policy-sensitive material, and formatting that may resemble attacks. Therefore, multi-turn detection must be evaluated against realistic benign distributions, ideally length-matched to the attacks; otherwise, reported performance is unlikely to remain reliable in deployment.

External classifiers answer a diferent question: Llama Guard is designed to classify prompts and responses under a taxonomy, and our protocol instead asks it to spot hidden intent before any harmful output exists, using only user turns. Its low AUC here, including on successful Crescendo, is therefore best read not as a defect of the classifier but as confirmation that pre-generation hidden-intent detection is a genuinely diferent intervention point from response-side moderation.

What this means for practice: Five rules follow for practitioners. First, on Llama 3.1 8B Instruct the default should be single-turn max-over-turn scoring, which beats three-turn, seven-turn, and accumulated settings, but this choice is model-specific, since accumulated scoring wins on Qwen2.5-7B-Instruct, so the window and the aggregation must be validated per model family. Second, thresholds should be tuned on the deployment’s own benign distribution rather than on synthetic prompts, because the synthetic three-turn threshold transferred to 90.85% FPR on WildChat. Third, long conversations need length-aware thresholds or audits, since more history means more dilution and accumulated scoring is not automatically safer. Fourth, deployments should watch for attackfamily drift, because a detector that cleanly separates MHJ can still collapse on Crescendo. Fifth, prompt-only detectors should be evaluated apart from response-side moderation, since pre-generation blocking is valuable precisely because it acts before output exists, but for the same reason it cannot use evidence that only the response contains.

## 8 Conclusion

We evaluate gradient-based jailbreak detection in multi-turn dialogue by adapting GradSafe with fixed-size context windows. Although the detector performs well under synthetic benign data, its performance decreases substantially on realistic WildChat conversations, and thresholds calibrated on synthetic data do not transfer. Short windows provide better separability than longer or fullhistory contexts, but successful Crescendo attacks remain dificult to distinguish from benign conversations. Results also vary across attack-generation methods and target models. These findings show that multi-turn gradient-based detection should be evaluated with realistic benign data, multiple context lengths, diverse attack families, and diferent model architectures.

## Acknowledgments

We thank the anonymous reviewers for their comments.

## References

[1] Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramer, et al. 2024. Jailbreakbench: An open robustness bench mark for jailbreaking large language models. Advances in Neural Information Processing Systems 37, 55005–55029.

[2] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. 2025. Jailbreaking black box large language models in twenty queries. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 23–42.

[3] Liang Chen, Xueting Han, Li Shen, Jing Bai, and Kam-Fai Wong. 2025. Vulnerability-aware alignment: Mitigating uneven forgetting in harmful finetuning. arXiv preprint arXiv:2506.03850 (2025).

[4] Purva Chiniya, Kevin Scaria, and Sagar Chaturvedi. 2026. Gradient-Controlled Decoding: A Safety Guardrail for LLMs with Dual-Anchor Steering. arXiv preprint arXiv:2604.05179 (2026).

[5] Anjun Gao, Yueyang Quan, Zhuqing Liu, and Minghong Fang. 2026. Beware What You Autocomplete: Forensic Attribution of Backdoored Code Completions. In Conference on Language Modeling (COLM).

[6] Anjun Gao, Yueyang Quan, Yufei Xia, Zhuqing Liu, and Minghong Fang. 2026. NeuronGuard: Robust LLM Safety Alignment via Ablation-Aware Safety Signal Redistribution. In EMNLP.

[7] Anjun Gao, Yueyang Quan, Yufei Xia, Zhuqing Liu, and Minghong Fang. 2026. Patcher: Post-Hoc Patching of Backdoored Large Language Models. In USENIX Security Symposium.

[8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).

[9] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. 2023. Not what you’ve signed up for: Compromising real world llm-integrated applications with indirect prompt injection. In Proceedings ofthe 16th ACM workshop on artificial intelligence and security. 79–90.

[10] Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2024. Gradient cuf: Detecting jailbreak attacks on large language models by exploring refusal loss landscapes. Advances in Neural Information Processing Systems 37 (2024), 126265–126296.

[11] Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, and Ling Liu. 2024. Antidote: Post-fine-tuning safety alignment for large language models against harmful fine-tuning. arXiv preprint arXiv:2408.09600 (2024).

[12] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. 2024. Qwen2. 5-coder technica report. arXiv preprint arXiv:2409.12186 (2024).

[13] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. 2023. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674 (2023).

[14] Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614 (2023).

[15] Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, et al. 2024. Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models. Advances in Neural Information Processing Systems 37, 47094–47165.

[16] Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, and Summer Yue. 2024. Llm defenses are not robust to multi-turn human jailbreaks yet. arXiv preprint arXiv:2408.15221 (2024).

[17] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. 2024. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249 (2024).

[18] Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. 2024. Tree of attacks: Jailbreaking black-box llms automatically. Advances in Neural Information Processing Systems 37, 61065–61105.

[19] Sindhu Padakandla, Sadbhavana Babar, Manohar Kaul, et al. 2025. SafeQuant: LLM Safety Analysis via Quantized Gradient Inspection. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2522–2536.

[20] Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas. 2023. Smoothllm: Defending large language models against jailbreaking attacks. arXiv preprint arXiv:2310.03684.

[21] Mark Russinovich, Ahmed Salem, and Ronen Eldan. 2025. Great, now write an article about that: The crescendo {Multi-Turn} {LLM} jailbreak attack. In 34th USENIX Security Symposium (USENIX Security 25). 2421–2440.

[22] Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2024. " do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security. 1671–1685.

[23] Yueqi Xie, Minghong Fang, Renjie Pi, and Neil Gong. 2024. Gradsafe: Detecting jailbreak prompts for llms via safety-critical gradient analysis. In Proceedings of the 62nd annual meeting ofthe association for computational linguistics (volume 1: Long papers). 507–518.

[24] Kang Yang, Guanhong Tao, Xun Chen, and Jun Xu. 2025. Alleviating the fear of losing alignment in LLM fine-tuning. In 2025 IEEE Symposium on Security and Privacy (SP). IEEE, 2152–2170.

[25] Baolei Zhang, Haoran Xin, Yuxi Chen, Zhuqing Liu, Biao Yi, Tong Li, Lihai Nie, Zheli Liu, and Minghong Fang. 2026. Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation. In IEEE Symposium on Security and Privacy.

[26] Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, and Zheli Liu. 2025. Traceback of poisoning attacks to retrieval-augmented generation. In The Web Conference.

[27] Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. 2024. Wildchat: 1m chatgpt interaction logs in the wild. 2024 (2024), 34590–34605.

[28] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043 (2023).