# Auditor-Assisted Summary-Channel Verification for Hosted LLM Identity Substitution

Ziyi Zhang Stony Brook University ziyi.zhang.3@stonybrook.edu

Guoheng Sun University of Maryland, College Park ghsun@umd.edu

Ziyao Wang University of Maryland, College Park ziyaow@umd.edu

Ang Li University of Maryland, College Park angliece@umd.edu

Jian Li Stony Brook University jian.li.3@stonybrook.edu

## Abstract

Hosted LLM services increasingly expose model identity as a product claim, but external users and auditors often cannot inspect which model configuration served a request. We study an auditor-assisted regime in which a trusted auditor enrolls claimed identities and prepares verification material before deployment, while audit-time verification uses only the claimed identity and released output, without runtime access to model internals, inference logs, routing decisions, or provider metadata. We propose SUMMARK, a summary-channel watermarking framework that uses user-visible reasoning summaries as a keyed audit surface. During setup, SUMMARK constructs identity-specific vocabulary carriers, trains lightweight summary adapters, and calibrates per-identity thresholds. At deployment time, a keyed statistical test assesses whether the released summary is consistent with the claimed identity and, when needed, attributes the most likely enrolled identity. Across model families, scales, and task domains, SUMMARK reliably detects cross-family and cross-scale substitutions at low false-positive rates while preserving task-facing answer quality. We further study FP16→INT8/INT4 precision-consistency diagnostics and robustness under bounded post-processing, rewriting stress tests, and text-only spoofing, revealing both strong performance and clear failure boundaries. Our code is publicly available at https://github.com/trinitr0toluene/Summark.

## 1 Introduction

Hosted Large Language Model (LLM) services increasingly offer model selection as a product feature: users or enterprise customers request a named model family and parameter scale, and receive responses through a standardized interface. This creates an incentive for cost-motivated identity substitution, where a provider continues to claim the requested model identity while serving responses through a different and often cheaper model configuration. Detecting such behavior from final answers alone is difficult because related models can produce superficially similar outputs for many prompts, and because surface style can be smoothed by prompting, decoding, or response normalization [3]. This motivates externally checkable evidence for auditing claimed identities in hosted LLM services, especially when deployment-time access to provider internals is unavailable.

![](images/728d292e8df67a65ccc7079156dfe5727a2b083f6ca79f53fe409ce28cb9a6b5.jpg)  
Figure 1: Auditor-assisted summary-channel verification. The user requests a claimed model identity m and receives a task-facing answer with a user-visible reasoning summary. Given the claim and released summary, a trusted auditor tests whether the summary is statistically consistent with m using secret verification material prepared offline. When answer and summary generation use separate pipelines, the guarantee is summary-channel consistency. End-to-end answer provenance requires additional coupling or deployment-side attestation.

Existing approaches occupy two broad extremes. At one end, pure output-behavior audits require no privileged setup or runtime access: they query the service and try to infer identity from response behavior, stylistic fingerprints, or challenge-response signatures [3, 9, 20, 12]. However, such evidence is unkeyed, prompt-dependent, and brittle under response variability, adaptive routing, provider-side normalization, or paraphrasing [3, 17]. At the other end, infrastructure-level verification can provide stronger evidence by relying on trusted instrumentation, execution traces, inference logs, cryptographic or trusted-execution mechanisms, or deployment-side attestation [28, 29, 5]. These mechanisms are powerful when available, but they require per-request runtime visibility into the provider-controlled serving stack, which may be unavailable to external auditors and can conflict with privacy, safety, or proprietary constraints [4].

We study the middle ground between these extremes, which we call onboarded-but-runtime-blind auditing. In this regime, a trusted auditor enrolls a finite set of claimed identities and prepares secret verification material during a one-time offline setup phase, but deployment-time auditing must operate from public response fields without runtime access to model internals, inference logs, routing decisions, or provider-reported metadata. Offline enrollment defines the configured identities; runtime auditing checks whether a returned auditable field is statistically consistent with the claimed identity. This regime captures settings such as third-party audits, regulatory evaluations, contractual reviews, or provider-volunteered transparency programs, where certified identities may be exposed during setup while the production serving stack remains unavailable on every request.

In this paper, we focus on one user-visible field as the audit carrier: reasoning summaries accompanying final answers. Reasoning summaries are increasingly exposed by hosted reasoning services as concise natural-language explanations while full chains-of-thought remain hidden. For example, OpenAI’s reasoning models [19] and Google’s Gemini thinking models [11] support summarized reasoning. These summaries are observable at deployment time, less tied to task correctness than final answers, and separate from the task-facing answer the user relies on. Our choice of summaries is therefore a carrier-design choice: summaries provide a user-visible, semantically related, and flexible field where audit evidence can be placed without directly perturbing the task-facing answer.

A natural way to turn this field into externally checkable evidence is through watermarking, applied to reasoning summaries rather than task-facing answers. Existing LLM watermarks typically embed key-dependent statistical biases into generated text and detect them through token-frequency or likelihood-based tests [16, 35, 7, 18]. When applied to final answers, however, the watermark carrier is tightly coupled to prompt semantics, formatting constraints, lexical entropy, and downstream post-processing. This can dilute detectability, create utility tradeoffs, and increase sensitivity to paraphrasing or rewriting [21, 25, 15, 6]. In contrast, summary-channel watermarking places keyed audit evidence in a separate user-visible field, enabling deployment-time verification while leaving task-facing answers minimally affected. This perspective leads to the following question:

Can the released reasoning summary serve as a keyed measurement surfacefor deploymenttime identity-consistency checks in an onboarded-but-runtime-blind hosted LLM service, without watermarking task-facing answers or requiring runtime access to internal traces?

We introduce SUMMARK, a summary-channel watermarking framework for auditor-assisted identityconsistency verification over a finite, pre-enrolled set of identities (Figure 1). During offline setup, the auditor uses a secret key to construct identity-specific vocabulary buckets, train lightweight summary watermark adapters, and calibrate per-identity decision thresholds. At deployment time, given a claimed identity and a released reasoning summary, the auditor applies a keyed statistical test to assess identity consistency. The same statistic supports both detection of identity-changing substitutions and attribution of the most likely generating identity among enrolled candidates. The deployment-time verifier requires no access to model parameters, activations, hidden traces, inference logs, runtime routing decisions, or provider-reported metadata.

This formulation defines the verification target of SUMMARK. The verifier provides statistical evidence that the released reasoning-summary field is consistent with the claimed enrolled identity under the auditor-assisted setup above. SUMMARK is therefore a deployment-time summary-channel audit layer: stronger than unkeyed behavioral testing over ordinary outputs, but complementary to infrastructure-level attestation when full serving provenance is required. Our evaluation shows that this summary-channel signal supports reliable low-FPR detection and attribution for identity-changing substitutions across model family and scale, while preserving task-facing answer quality. We further use post-processing, precision-variant, and spoofing stress tests to characterize where the signal remains stable and where it begins to degrade. Our main contributions are as follows:

• Auditor-assisted verification regime. We formulate an onboarded-but-runtime-blind regime for hosted LLM identity claims: a trusted auditor enrolls identities and prepares keyed verification material offline, while audit-time decisions use released summaries without runtime access to provider logs, traces, routing decisions, model internals, or self-reported metadata.

• Reasoning summaries as a keyed audit surface. We identify the user-visible reasoning-summary field as a practical audit carrier that is externally observable at deployment time and can hold verification evidence while isolating the task-facing answer from watermark perturbation.

• Identity-specific summary-channel watermarking. We instantiate this idea in SUMMARK, which uses keyed, identity-specific vocabulary carriers and lightweight summary adapters to support low-FPR substitution detection and multi-class attribution over enrolled identities.

• Boundary-aware evaluation. Across model families, scales, and task domains, we evaluate SUMMARK under identity-changing substitutions, attribution, answer-quality preservation, behavioral and generic watermarking baselines, precision-variant diagnostics, and robustness stress tests. The results show strong low-FPR summary-channel verification while exposing clear boundaries under stronger post-processing and spoofing.

## 2 Threat Model

We consider an auditor-assisted model identity auditing setting with three roles. A service provider hosts an LLM service and exposes a user-facing interface; an end user issues prompts and consumes responses; and a trusted auditor verifies model identity on behalf of users, customers, or regulators. For each request, the provider returns (i) a task-facing final answer and (ii) a user-visible reasoning summary, while presenting the response under a claimed identity $m _ { c } .$ . We define a model identity by its (family, parameter scale) pair. Numerical precision is treated separately as a within-identity deployment variant rather than as a base identity. The end user holds no secret material and interact solely through the public interface. At deployment time, the auditor observes only the claimed identity $m _ { c }$ and the released reasoning summary s.

Auditor access and certification target. The auditor’s access is split into two phases. During a one-time offline setup, the auditor has trusted reference access to enrolled identities, sufficient to prepare per-identity verification material and calibration thresholds; concrete procedures are described in Section 3. During deployment-time auditing, the auditor’s decision depends only on $( m _ { c } , s )$ and a secret verification key. The auditor does not observe model parameters, internal activations, hidden reasoning states, inference logs, runtime routing decisions, or provider-reported metadata.

The audit target is the released reasoning-summary channel. Given $( m _ { c } , s )$ , the auditor tests whether s is statistically consistent with the claimed identity under the auditor’s keyed verification material. This summary-only decision does not, by itself, certify the provenance of the task-facing answer when answer generation and summary generation are decoupled. End-to-end serving provenance requires additional answer-summary coupling, deployment policy, or infrastructure-level attestation, which is complementary to the statistical audit layer studied here.

Provisioning and trust assumptions. The auditor’s verification material is provided to the serving stack only through authenticated invocation: the provider may use the authorized material during legitimate serving, but cannot extract the secret key, carrier sets, adapter weights, or reusable offline material for other identities. This invocation channel may be instantiated by an auditor-hosted service, a protected serving wrapper, or another authenticated provisioning mechanism. SUMMARK evaluates the statistical audit layer under this access model rather than proposing a full enforcement protocol. This setting is intended for provider-participating audits, such as external audits, procurement checks, regulatory reviews, or voluntary transparency programs. It is stronger than pure API-only access but weaker than full runtime attestation. We assume the auditor’s key store is not compromised and the provisioning channel is authenticated.

Adversary (provider). The provider may be cost-motivated and may attempt to make the auditor accept the claimed identity $m _ { c }$ even when the auditable summary channel is generated under, or made to appear consistent with, a different served identity $m _ { d } \neq m _ { c } .$ A stronger form of evasion is spoofing, in which the provider tries to reproduce the verification signal expected under $m _ { c }$ using a substituted model. We consider the following capabilities.

Identity substitution. Let M be a finite set of enrolled identities at the family×scale granularity. The adversary may perform cross-scale substitution, where the served identity changes parameter scale within the same model family; cross-family substitution, where the served identity changes model family; or jointfamily-and-scale substitution, where both attributes differ, e.g., serving Qwen-1.5B while claiming LLaMA-3B. These are the primary identity-substitution events studied in this paper.

Precision variants. Our primary identity universe treats model identity as a family-scale pair. Numerical precision changes, such as FP16→INT8 or FP16→INT4, preserve this identity and are therefore not counted as identity-substitution events. We evaluate them separately as precisionconsistency diagnostics: for a fixed identity m, the verifier calibrated on FP16 summaries tests whether summaries from a quantized deployment remain consistent with that calibration. Mechanism details and results appear in Sections 3 and 4.4.

Summarypost-processing. The primary post-processing model allows bounded online transformations to the released summary, such as formatting edits, deletion, truncation, or sentence reordering. These transformations preserve the released text as the reasoning-summary field for the same request and test whether the verifier depends on exact phrasing or sentence order. Stronger transformations, such as full LLM-based paraphrase rewriting or fine-tuning-based watermark removal, are not part of the primary bounded post-processing model because they regenerate or substantially alter the summary rather than preserving it up to surface edits. We evaluate them as stress tests in Section 4.5 to characterize the boundary of text-based summary verification.

Watermark spoofing. The provider may observe or query watermarked summaries produced under the claimed identity, but does not obtain the auditor’s secret key, carrier sets, adapter weights, or reusable offline access to the claimed identity’s verification material. The adversary may then try to reproduce $m _ { c } { ^ { \circ } } \mathbf { s }$ verification signal on a substituted model $m _ { d }$ from text-only observations. We evaluate two spoofing strategies in Section 4.5: adapter distillation, which trains a local generator on $m _ { d }$ from observed prompt-summary pairs, and watermark stealing, which estimates token-level signal statistics from watermarked texts and biases decoding accordingly. These attacks test whether the claimed verification signal can be forged from text-only observations alone.

## 3 SUMMARK Framework

We present SUMMARK, a summary-channel framework for auditor-assisted identity-consistency verification (Figure 2). The central idea is to turn the released reasoning summary into a keyed measurement surface. Each enrolled identity receives a secret vocabulary carrier, and a lightweight summary adapter imprints that carrier into its summaries. At deployment time, the verifier receives only the claimed identity $m _ { c } ,$ released summary s, and auditor’s secret key, and tests whether the summary’s token statistics match the carrier calibrated for $m _ { c }$

SUMMARK has three components. First, the auditor constructs identity-specific carrier sets $\{ G _ { m } \} _ { m \in \mathcal { M } }$ with controlled density and limited cross-identity overlap. Second, for each enrolled identity, the auditor trains a lightweight adapter $\Delta _ { m }$ that increases the corresponding carrier footprint while preserving fluency and faithfulness. Third, the auditor calibrates thresholds and performs a keyed test for detection or attribution using only the released summary. Unlike answer-level watermarking, SUMMARK localizes audit evidence to the reasoning summary. The final answer is the task-facing output, so watermarking it can create utility or formatting tradeoffs. The reasoning summary is an auxiliary user-visible field with more lexical slack, allowing SUMMARK to embed identity-specific evidence while minimally affecting the task-facing answer.

![](images/003cfecfdd3539ee2a74370cb31e31a15be269b8bbe7a89e1beaa21739a044ee.jpg)  
Figure 2: Workflow of SUMMARK. Offline, the trusted auditor uses a secret key to construct identityspecific vocabulary carriers, train summary watermark adapters, and calibrate thresholds. Online, the provider returns a task-facing answer and reasoning summary under a claimed identity. The auditor uses only the claim, released summary, and secret key to test summary-channel consistency and, when needed, attribute the most likely enrolled identity.

## 3.1 Problem Definition

Let M be a finite set of enrolled identities, where each $m \in \mathcal { M }$ specifies a model family and parameter scale. During offline setup, the auditor prepares identity-specific verification material for each m: a carrier set $G _ { m } \subseteq V$ over the vocabulary V, a summary watermark adapter $\Delta _ { m }$ and a detection threshold calibrated for a target false-positive rate. At deployment time, for a user prompt x, the provider returns a response $r = ( y , s )$ consisting of a task-facing answer y and a user-visible reasoning summary s, while presenting the response under a claimed identity $m _ { c } \in \mathcal { M }$ The verifier receives only $( m _ { c } , s )$ and the auditor’s secret key; it does not access model parameters, activations, logits, hidden reasoning states, inference traces, runtime routing decisions, system logs, or provider-reported metadata.

Detection. Given $( m _ { c } , s )$ , the verifier tests whether the released summary is statistically consistent with the claimed identity $m _ { c }$ or is more consistent with a different enrolled identity $m _ { d } \neq m _ { c }$

Attribution. Given s and M, identity attribution selects the enrolled identity whose keyed carrier statistic best explains the released summary.

Substitution types. We consider three identity-changing substitution categories over M: (i) crossfamily substitution, where the served identity changes model family; (ii) cross-scale substitution, where the served identity changes parameter scale within the same family; and (iii) jointfamily-and scale substitution, where both attributes differ, e.g., serving Qwen-1.5B while claiming LLaMA-3B.

Precision variants. Our primary task treats identities as family-scale pairs. We separately study whether the FP16-calibrated verifier for a fixed identity m provides useful consistency evidence when the same identity is served at INT8 or INT4 precision. This is not an attribution task over M: all precision variants reuse the same carrier set $G _ { m }$ and adapter $\Delta _ { m }$ , and the verifier flags a variant only when its summary statistics deviate from FP16 calibration.

## 3.2 Identity-Specific Summary Carriers

Core idea. SUMMARK assigns each identity $m \in \mathcal { M }$ a secret, key-derived subset of vocabulary tokens $G _ { m } \subseteq V$ , which we call its carrier set. During summary generation, the provider applies the identity-specific adapter $\Delta _ { m }$ to nudge token usage toward $G _ { m }$ while maintaining a fluent and faithful summary. This produces an identity-dependent footprint that is detectable from summary text alone using the auditor’s key.

Audit-time statistic. Let $\boldsymbol { s } = ( s _ { 1 } , \dots , s _ { T } )$ be the released summary. For identity $m ,$ the verifier computes the green-match count $\begin{array} { r } { C _ { m } ( s ) = \sum _ { t = 1 } ^ { T } \mathbb { I } [ s _ { t } \in G _ { m } ] } \end{array}$ , and converts it to a calibrated score $\bar { Z _ { m } ( s ) } = ( \bar { C _ { m } ( s ) } - \mu _ { m } ( T ) ) / \sigma _ { m } ( T )$ using held-out summaries for identity m. Detection compares $Z _ { m _ { c } } ( s )$ to a per-identity threshold chosen for a target false-positive rate, while attribution selects the identity with the largest score. Full calibration details are in Appendix D.3.

Constructing many separable carrier sets. A single global greenlist would create a shared signal across all models and does not support reliable attribution. SUMMARK instead uses the secret key K to construct a family of carrier sets $\{ G _ { m } \} _ { m \in \mathcal { M } }$ with two tunable properties:

▷ Controlled green ratio. Each identity receives a carrier set of comparable density:

$$
| G _ { m } | / | V | \approx \rho , \quad \forall m \in \mathcal { M } ,\tag{1}
$$

which stabilizes the null baseline across identities.

▷ Controlled cross-identity overlap. The construction also limits cross-identity collisions:

$$
\frac { | G _ { m } \cap G _ { m ^ { \prime } } | } { | G _ { m } | } \leq \tau , \quad \forall m \neq m ^ { \prime } ,\tag{2}
$$

so different identities remain distinguishable even when the auditor sees only short summaries.

The pair $( \rho , \tau )$ governs the trade-off between watermark strength and identity capacity: larger ρ increases per-token evidence, while smaller τ improves separability across identities. This differs from a shared watermark: a shared greenlist can indicate that some watermark is present, but cannot distinguish which enrolled identity produced the summary. Controlled overlap is what turns watermark detection into model-identity attribution. We provide a concrete keyed construction and a capacity characterization in Appendix C.2.

## 3.3 Training Summary Watermark Adapters

Embedding mechanism. To realize an identity-dependent bias in the summary channel, SUMMARK trains a lightweight adapter for each enrolled identity. For each identity $m \in \mathcal { M }$ with base weights $\theta _ { m } ,$ we train a summary adapter $\Delta _ { m }$ while keeping $\theta _ { m }$ frozen. At deployment time, the provider enables $\Delta _ { m _ { c } }$ only when generating the reasoning summary for a request that claims identity $m _ { c }$

Training objective. Adapter training balances three goals: (i) utility, so summaries remain fluent and faithful; (ii) detectability, so the summary exhibits an elevated footprint of $G _ { m } ;$ and (iii) separability, so the correct identity’s footprint exceeds competing identities’ footprints. We instantiate the base identity objective as

$$
L = L _ { \mathrm { s f t } } + \lambda _ { \mathrm { w m } } L _ { \mathrm { w m } } + \lambda _ { \mathrm { s e p } } L _ { \mathrm { s e p } } .\tag{3}
$$

Here, $L _ { \mathrm { s f t } }$ is a standard supervised fine-tuning loss on reference summaries, $L _ { \mathrm { w m } }$ encourages increased usage or probability mass of the identity-specific carrier set $G _ { m }$ , and $L _ { \mathrm { s e p } }$ penalizes confusion by enforcing a margin between the footprint of $G _ { m }$ and competing carrier sets. Intuitively, $L _ { \mathrm { w m } }$ makes the claimed identity detectable, while $L _ { \mathrm { s e p } }$ makes it distinguishable from other enrolled identities. Full definitions are provided in Appendix D.

Optional precision-consistency regularizer. For the sidecar precision analysis described in Section 3.1, SUMMARK reuses the same carrier set $G _ { m }$ and the same adapter $\Delta _ { m }$ across the FP16, INT8, and INT4 variants of identity $m$ . We do not introduce precision-specific carrier sets or precisionspecific adapters. Instead, to make the FP16-calibrated verifier sensitive to precision-induced changes, we optionally add a lightweight regularizer that increases the separation between the carrier statistic observed under FP16 forward passes and the statistic observed under simulated quantized forward passes of the same identity. This regularizer does not create separate precision identities; it only makes the single identity-level verifier more informative for precision-consistency diagnostics. The full objective is provided in Appendix D, and Appendix E.3 ablates its effect.

Why adapters? Adapters make the watermark (a) cheap to provision and switch per identity, (b) tied to the enrolled model family and scale rather than a generic decoding-time bias, and (c) localized to summary generation so task-facing answers are minimally affected. LoRA adapters are shape-bound to enrolled bases, making cross-family or cross-scale cross-loading incompatible in our setting; same-shape precision variants remain compatible and are treated as diagnostics. Our framework is not tied to LoRA specifically: any mechanism that induces controlled, identity-dependent token preferences in summary generation is compatible.

Takeaway. Taken together, the carrier construction and adapter objective turn ordinary watermark detection into identity-specific verification. A generic watermark can indicate that a text was watermarked, but not which enrolled identity it is consistent with. SUMMARK’s key contribution is to make the summary watermark separable across identities and auditable from text alone at strict false-positive rates.

## 4 Experiments

We organize the evaluation around five claims: reliable low-FPR substitution detection, multi-class attribution, superiority over behavioral and generic watermarking baselines, task-facing answerquality preservation, and boundary behavior under precision variants, post-processing, rewriting, and spoofing. Section 4.2 addresses detection, attribution, and answer quality; Section 4.3 isolates the role of identity-specific summary watermarking; and Sections 4.4–4.5 study boundary cases. Appendix E.3 provides ablations on watermark design and training loss components. We further include a qualitative case study in Appendix E.4, showing how the adapter leaves an identity-specific token pattern in the summary text.

## 4.1 Experimental Settings

Models and datasets. We evaluate four enrolled identities: Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5- 7B, and LLaMA3.2-3B. Appendix E.1 extends this to 12 identities across four model families, with a separate budget of 1,000 examples per identity. We use three datasets that induce distinct summary styles: Math for mathematical reasoning, Code for programming, and General for general problem solving. These domains differ in both task content and summary structure: Code summaries tend to be more templated and procedural, while General summaries are more stylistically varied and conversational. Full dataset details appear in Appendix B.1.

Substitution and precision-variant settings. An identity $m \in \mathcal { M }$ is a family-scale pair. In the primary substitution benchmark, the provider serves a different identity while claiming $m _ { c } ,$ covering cross-scale, cross-family, and joint family-and-scale substitutions. Unless stated otherwise, the substitution ratio is 50%. Separately, we evaluate precision-consistency diagnostics: the claimed configuration is (m, FP16), while the served configuration is (m, INT8) or (m, INT4) for the same identity m. Precision variants reuse the same carrier set $G _ { m }$ and adapter $\Delta _ { m }$ , so they are evaluated as within-identity consistency checks rather than multi-class attribution over $\mathcal { M }$

Baselines. We compare against two classes of baselines. First, text-classifier baselines operate on released summaries without watermarking: BERT [8], GPT-2 [22], and T5 [23] are trained to predict consistency with a claimed identity from summary text alone. These baselines test whether unkeyed surface or stylistic cues are sufficient for identity auditing. Second, watermark baselines operate on the same summary channel: Plain (no watermark), Greenlist (one shared greenlist across identities), Unigram [35], Gumbel-Max [1], and SynthID [7]. Unlike SUMMARK, these methods do not provide identity-specific carrier separability across enrolled identities.

Metrics. For substitution detection, we report TPR at a fixed FPR of 1% (TPR@1%), the corresponding accuracy at that threshold (ACC@1%), and AUC. TPR@1% is the primary metric because practical auditing requires very low false-alarm rates. For attribution, we report multi-class accuracy and macro-F1. For task performance, we report answer accuracy on Math, Pass@1/unit-test success on Code, and perplexity on General. For spoofing, we report attack success rate (ASR@1%), the fraction of spoofed summaries incorrectly accepted as the claimed identity.

Table 1: Summary of substitution detection across seven substitution pairs per dataset. We report average (AvgT) TPR@FPR=1%, minimum (MinT) TPR@FPR=1%, and average AUC. Full per-pair results are in Appendix E.1.
<table><tr><td></td><td colspan="3">Math</td><td colspan="3">General</td><td colspan="3">Code</td></tr><tr><td>Method</td><td>AvgT</td><td>MinT</td><td>AUC</td><td>AvgT</td><td>MinT</td><td>AUC</td><td>AvgT</td><td>MinT</td><td>AUC</td></tr><tr><td>BERT</td><td>0.7683</td><td>0.1232</td><td>0.9840</td><td>0.5439</td><td>0.0877</td><td>0.9401</td><td>0.6567</td><td>0.0000</td><td>0.9916</td></tr><tr><td>GPT-2</td><td>0.3333</td><td>0.0000</td><td>0.7969</td><td>0.2506</td><td>0.0000</td><td>0.8101</td><td>0.3886</td><td>0.0000</td><td>0.8174</td></tr><tr><td>T5</td><td></td><td></td><td></td><td></td><td>0.8676 0.6359 0.9787 0.5213 0.0000</td><td>0.9703</td><td></td><td>0.74940.1053 0.9975</td><td></td></tr><tr><td>SUMMARK 0.9994 0.9960 0.9996 0.9750 0.8740</td><td></td><td></td><td></td><td></td><td></td><td>0.9990</td><td>1.0000 1.0000 1.0000</td><td></td><td></td></tr></table>

Table 2: Multi-class model identity attribution from reasoning summaries.
<table><tr><td colspan="3">Math</td><td colspan="2">Code</td><td colspan="2">General</td></tr><tr><td>Method</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td></tr><tr><td>BERT</td><td>0.9386</td><td>0.9383</td><td>0.9580</td><td>0.9583</td><td>0.9201</td><td>0.9201</td></tr><tr><td>GPT-2</td><td>0.7121</td><td>0.6981</td><td>0.5768</td><td>0.5754</td><td>0.7095</td><td>0.6894</td></tr><tr><td>T5</td><td>0.9128</td><td>0.9122</td><td>0.9841</td><td>0.9841</td><td>0.9575</td><td>0.9575</td></tr><tr><td>SUMMARK</td><td>0.9958</td><td>0.9957</td><td>0.9678</td><td>0.9708</td><td>0.9718</td><td>0.9718</td></tr></table>

Table 3: Task performance with and without summary watermarking (WM). Higher is better for Acc and Pass@1; lower is better for PPL.
<table><tr><td></td><td colspan="2">Math (Acc ↑)</td><td colspan="2">General (PPL ↓)</td><td colspan="2">Code (Pass@1 ↑)</td></tr><tr><td>Model</td><td>w/o WM</td><td>w/WM</td><td>w/o WM</td><td>w/WM</td><td>w/o WM</td><td>w/WM</td></tr><tr><td>LLaMA-3B</td><td>0.2069</td><td>0.2159</td><td>8.1044</td><td>7.8450</td><td>0.0290</td><td>0.0430</td></tr><tr><td>Qwen-1.5B</td><td>0.2971</td><td>0.2967</td><td>29.3089</td><td>25.6346</td><td>0.0170</td><td>0.0170</td></tr><tr><td>Qwen-3B</td><td>0.4219</td><td>0.4240</td><td>10.1635</td><td>10.1429</td><td>0.0630</td><td>0.0760</td></tr><tr><td>Qwen-7B</td><td>0.5034</td><td>0.5062</td><td>14.1778</td><td>13.8621</td><td>0.1920</td><td>0.1830</td></tr></table>

Training configuration. We construct watermark carriers with green ratio $\rho = 0 . 3$ and maximum overlap τ = 0.3. Adapters are trained on 5,000 samples with an 80/20 train-test split for one epoch using AdamW, learning rate $2 \times 1 0 ^ { - 4 }$ , and temperature 0.7. Loss weights are $\lambda _ { \mathrm { w m } } = \lambda _ { \mathrm { s e p } } = 0 . 5$ We use a balanced 50% mixture and provide sensitivity to substitution ratio in Appendix E.5. All experiments were conducted on one NVIDIA RTX 8000 GPU with 48GB memory.

## 4.2 Identity-Substitution Detection

Table 1 summarizes substitution detection across Math, General, and Code. Each dataset aggregates the same seven substitution pairs: three cross-scale Qwen substitutions, two cross-family same-scale substitutions, and two joint family-and-scale substitutions. SUMMARK achieves consistently high TPR at the strict FPR= 1% operating point, with average TPR@1% of 0.9994 on Math, 0.975 on General, and 1.000 on Code. More importantly, its minimum TPR@1% remains high across all pairs: 0.996 on Math, 0.874 on General, and 1.000 on Code.

The contrast matches the design goal of SUMMARK. Text classifiers implicitly learn domain- and dataset-specific regularities such as phrasing, formatting, and templates. These cues are strongest in more structured domains such as Code, where T5 achieves high AUC, but they are not identity-bound and can fail under closely related substitutions or domain changes. SUMMARK instead encodes identity-specific evidence into the summary channel through keyed vocabulary carriers and summary adapters, so substitution induces a predictable shift in summary-level token statistics rather than a fragile stylistic change. Full per-pair results, including the original Math, General, and Code tables, are reported in Appendix E.1.

Attribution. Attribution uses the same standardized green-token statistic as detection, scoring the released summary against each enrolled identity and predicting the largest score. Table 2 shows that SUMMARK supports multi-class attribution across datasets, reaching 0.9958/0.9957 ACC/F1 on Math and 0.9718/0.9718 on General. On Code, T5 slightly outperforms SUMMARK, reflecting the strong templatic structure of Code summaries. Nevertheless, SUMMARK remains competitive and does not rely on unkeyed stylistic cues.

Task-facing answer quality. Table 3 shows that summary-channel watermarking largely preserves final-answer quality. On Math and Code, performance is essentially unchanged after watermarking. On General, perplexity sometimes improves slightly, which we view as benign regularization of summary wording rather than improved reasoning. These results support localizing audit evidence to the summary channel rather than perturbing the task-facing answer.

Table 4: Substitution detection on Math with watermarking baselines. We report TPR@FPR=1%, ACC@FPR=1%, and AUC. Setting A→B denotes serving B while claiming A.
<table><tr><td></td><td colspan="3">Unigram</td><td colspan="3">Gumbel Max</td><td colspan="3">SynthID</td><td colspan="3">SUMMARK</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Qwen-7B→Qwen-3B</td><td>0.4730</td><td>0.7315</td><td>0.9523</td><td>0.0050</td><td>0.4975</td><td>0.4828</td><td>0.0050</td><td>0.4975</td><td>0.5293</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-7B→Qwen-1.5B</td><td>0.0690</td><td>0.5295</td><td>0.7288</td><td>0.0300</td><td>0.5100</td><td>0.4968</td><td>0.0200</td><td>0.5050</td><td>0.5506</td><td>0.9960</td><td>0.9930</td><td>0.9974</td></tr><tr><td>Qwen-3B→Qwen-1.5B</td><td>0.0000</td><td>0.4950</td><td>0.2753</td><td>0.0950</td><td>0.5425</td><td>0.6150</td><td>0.0950</td><td>0.5425</td><td>0.6174</td><td>1.0000</td><td>0.9950</td><td>0.9998</td></tr><tr><td>Qwen-3B→LLaMA-3B</td><td>0.0010</td><td>0.4955</td><td>0.6302</td><td>0.0400</td><td>0.5150</td><td>0.4865</td><td>0.0800</td><td>0.5350</td><td>0.4961</td><td>1.0000</td><td>0.9950</td><td>0.9997</td></tr><tr><td>LLaMA-3B→Qwen-3B</td><td>0.1650</td><td>0.5775</td><td>0.7466</td><td>0.0000</td><td>0.4950</td><td>0.5585</td><td>0.0000</td><td>0.4950</td><td>0.5240</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-7B→LLaMA-3B</td><td>0.4510</td><td>0.7205</td><td>0.9214</td><td>0.1250</td><td>0.5575</td><td>0.6000</td><td>0.1000</td><td>0.5450</td><td>0.5987</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>LLaMA-3B→Qwen-1.5B</td><td>0.0660</td><td>0.5280</td><td>0.6194</td><td>0.0000</td><td>0.4950</td><td>0.5471</td><td>0.0200</td><td>0.5050</td><td>0.5014</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr></table>

Table 5: Within-identity precision consistency across Math, Code, and General. FP16→INT8 denotes serving a quantized variant while claiming the FP16 configuration of the same identity. Precision variants share the same carrier set and adapter; the verifier is calibrated on FP16 summaries.
<table><tr><td></td><td></td><td colspan="3">Math</td><td colspan="3">Code</td><td colspan="3">General</td></tr><tr><td>Model</td><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td rowspan="2">LLaMA-3B</td><td>FP16→INT8</td><td>0.9990</td><td>0.9960</td><td>0.9990</td><td>0.9160</td><td>0.9530</td><td>0.9694</td><td>0.9580</td><td>0.9740</td><td>0.9971</td></tr><tr><td>FP16→INT4</td><td>0.9990</td><td>0.9995</td><td>0.9990</td><td>0.9680</td><td>0.9790</td><td>0.9791</td><td>0.9930</td><td>0.9920</td><td>0.9998</td></tr><tr><td rowspan="2">Qwen-1.5B</td><td>FP16→INT8</td><td>0.1132</td><td>0.5521</td><td>0.7575</td><td>0.0360</td><td>0.5130</td><td>0.7010</td><td>0.4454</td><td>0.7182</td><td>0.8534</td></tr><tr><td>FP16→INT4</td><td>0.3417</td><td>0.6663</td><td>0.9209</td><td>0.4120</td><td>0.7010</td><td>0.9497</td><td>0.7858</td><td>0.8884</td><td>0.9454</td></tr><tr><td rowspan="2">Qwen-3B</td><td>FP16→INT8</td><td>0.9860</td><td>0.9890</td><td>0.9973</td><td>0.9460</td><td>0.9680</td><td>0.9952</td><td>0.9910</td><td>0.9910</td><td>0.9964</td></tr><tr><td>FP16→INT4</td><td>0.9960</td><td>0.9945</td><td>0.9986</td><td>0.9970</td><td>0.9935</td><td>0.9997</td><td>0.9950</td><td>0.9940</td><td>0.9989</td></tr><tr><td rowspan="2">Qwen-7B</td><td>FP16→INT8</td><td>0.9910</td><td>0.9920</td><td>0.9982</td><td>0.9910</td><td>0.9905</td><td>0.9996</td><td>0.9530</td><td>0.9715</td><td></td></tr><tr><td>FP16→INT4</td><td>0.9990</td><td>0.9975</td><td>0.9998</td><td>1.0000</td><td>0.9975</td><td>1.0000</td><td>0.9810</td><td>0.9875</td><td>0.9851 0.9945</td></tr></table>

## 4.3 Generic Watermarks Are Not Enough

Table 4 compares SUMMARK with generic watermarking baselines under the same summary-channel substitution-detection protocol. Unlike SUMMARK, these baselines do not create identity-specific carriers. The results show that generic watermarking is insufficient for reliable identity auditing at the strict low-FPR operating point. Unigram shows some aggregate separability, e.g., AUC = 0.9523 for Q7→Q3 and 0.9214 for Q7→L3, but its TPR@1% is only 0.4730 and 0.4510 in those cases and drops near zero elsewhere. Gumbel-Max and SynthID are weaker still.

By contrast, SUMMARK achieves TPR@1% between 0.9960 and 1.0000, ACC@1% between 0.9930 and 0.9950, and AUC between 0.9974 and 1.0000 across all seven substitution pairs. The key difference is that SUMMARK tests whether the summary matches the keyed footprint of the claimed identity, not merely whether it contains a generic watermark. Identity-specific carrier construction gives enrolled identities separable token footprints, while adapter training amplifies the correct footprint and suppresses confusion with competitors. Appendix E.3 further shows that Plain and shared-Greenlist variants fail at the strict FPR= 1% operating point, confirming that the signal must be identity-specific and separable.

## 4.4 Precision-Variant Diagnostics

Table 5 evaluates within-identity precision consistency: the provider serves an INT8 or INT4 quantized variant while claiming the FP16 model. Unlike cross-family or cross-scale substitution, precision variants share the same model identity, carrier set $G _ { m } .$ , and adapter $\Delta _ { m } .$ . We therefore treat this as a sidecar diagnostic rather than substitution detection over M: the verifier is calibrated on FP16 reference summaries and flags a quantized deployment only when its released summaries deviate from the FP16-calibrated footprint.

For LLaMA-3B, Qwen-3B, and Qwen-7B, both FP16→INT8 and FP16→INT4 are flagged reliably across Math, Code, and General, with AUCs mostly above 0.96 and TPR@1% often near 1.0. INT4 is generally easier to flag than INT8, consistent with more aggressive quantization inducing a larger shift in the summary-channel footprint. The main exception is Qwen-1.5B: FP16→INT8 remains close to the FP16 footprint, with TPR@1% dropping to 0.1132 on Math, 0.0360 on Code, and 0.4454 on General. INT4 is more detectable but still weaker than for larger identities. These results position FP16-calibrated precision consistency as a within-identity diagnostic that complements the cross-identity substitution detection studied above. Appendix E.3 ablates the precision-consistency regularizer introduced in Section 3.3.

## 4.5 Robustness and Boundary Analysis

Table 6 summarizes robustness under summary postprocessing, rewriting, and spoofing. For bounded surface transformations, including sentence shuffle, random deletion, and word truncation, detection remains essentially unchanged: the minimum TPR@1% across all edited summaries is 0.9950. This confirms that SUMMARK does not rely on exact phrasing or sentence order; its verifier aggregates summary-level token statistics that survive reordering and partial deletion.

Table 6: Robustness summary on Math across seven substitution pairs. For post-processing, we report average/min TPR@FPR=1%; for spoofing, average/max ASR@FPR=1%. Full results are in Appendix E.2.
<table><tr><td>Stress test</td><td>Avg.</td><td>Worst</td></tr><tr><td>Original TPR@1%</td><td>0.9994</td><td>0.9960</td></tr><tr><td>Bounded edits TPR@1%</td><td>0.9992</td><td>0.9950</td></tr><tr><td>Bilingual rewrite TPR@1%</td><td>0.7900</td><td>0.5740</td></tr><tr><td>Distilled spoofing ASR@1% Watermark-stealing ASR@1%</td><td>0.1651 0.0417</td><td>0.6290 0.2160</td></tr></table>

Bilingual rewriting is stronger because it replaces tokens rather than rearranging them. It reduces average TPR@1% to 0.7900 and worst-case TPR@1% to 0.5740, showing that semantic-level rewriting can attenuate the lexical footprint. Thus, bounded edits are well tolerated, while rewriting that replaces many identity-discriminative tokens can reduce low-FPR detectability.

Spoofing gives a similar boundary-aware picture. Watermark-stealing is largely ineffective, with average ASR@1% of 0.0417, suggesting that text-only token-statistic estimates cannot reproduce the keyed footprint. Distilled adapter spoofing is stronger: it is contained in several transfer directions but reaches worst-case ASR@1% of 0.6290. We therefore treat spoofing robustness as empirical rather than guaranteed, and report full per-pair results, hardest transfer directions, and additional watermark removal stress tests in Appendix E.2.

Takeaway. Overall, SUMMARK turns reasoning summaries into a keyed, identity-specific audit signal: it enables reliable low-FPR detection and attribution from released summaries, outperforms shared and generic watermarking baselines, and preserves task-facing answer quality. The boundary analyses further clarify the scope of the guarantee: the signal is stable under bounded surface edits, useful but model-dependent for precision variants, and vulnerable to sufficiently strong rewriting or adaptive spoofing.

## 5 Conclusion

We introduce SUMMARK, a summary-channel watermarking framework for auditing hosted LLM identity claims from user-visible reasoning summaries. SUMMARK embeds keyed, identity-specific signals into summaries, enabling low-FPR substitution detection and multi-class attribution while preserving task-facing answer quality. Our results also characterize important boundaries: precisionconsistency diagnostics are useful but model-dependent, bounded post-processing is well tolerated, and strong rewriting or adaptive spoofing remains challenging. Future work includes richer audit signals beyond text and stronger robustness for fine-grained deployment variants.

## Acknowledgements

This work was supported in part by the National Science Foundation (NSF) grants 2541997, 2148309, 2315614 and 2337914, and the National Institutes of Health (NIH) grant R01HL184139. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies.

## References

[1] Scott Aaronson and Hendrik Kirchner. Watermarking GPT outputs. https://scottaaronson. blog/?m=202302, 2023. Unpublished technical note / blog post.

[2] Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. Turning your weakness into a strength: Watermarking deep neural networks by backdooring. In 27th USENIX security symposium (USENIX Security 18), pages 1615–1631, 2018.

[3] Will Cai, Tianneng Shi, Xuandong Zhao, and Dawn Song. Are you getting what you pay for? auditing model substitution in llm apis. arXiv preprint arXiv:2504.04715, 2025.

[4] Stephen Casper, Carson Ezell, Charlotte Siegmann, Noam Kolt, Taylor Lynn Curtis, Benjamin Bucknall, Andreas Haupt, Kevin Wei, Jérémy Scheurer, Marius Hobbhahn, et al. Black-box access is insufficient for rigorous ai audits. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, pages 2254–2272, 2024.

[5] Bing-Jyue Chen, Suppakit Waiwitlikhit, Ion Stoica, and Daniel Kang. Zkml: An optimizing system for ml inference in zero-knowledge proofs. In Proceedings of the Nineteenth European Conference on Computer Systems, pages 560–574, 2024.

[6] Ruibo Chen, Yihan Wu, Junfeng Guo, and Heng Huang. De-mark: Watermark removal in large language models. arXiv preprint arXiv:2410.13808, 2024.

[7] Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, Rob McAdam, Johannes Welbl, Vandana Bachani, Alex Kaskasoli, Robert Stanforth, Tatiana Matejovicova, et al. Scalable watermarking for identifying large language model outputs. Nature, 634(8035):818–823, 2024.

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

[9] Irena Gao, Percy Liang, and Carlos Guestrin. Model equality testing: Which model is this api serving? arXiv preprint arXiv:2410.20247, 2024.

[10] Mansour Al Ghanim, Jiaqi Xue, Rochana Prih Hastuti, Meng Zheng, Yan Solihin, and Qian Lou. Evaluating the robustness and accuracy of text watermarking under real-world cross-lingual manipulations. Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, 2025.

[11] Google. Gemini api: Thinking and thought summaries. https://ai.google.dev/ gemini-api/docs/thinking, 2024. Accessed 2025-01.

[12] Zecheng He, Tianwei Zhang, and Ruby B Lee. Verideep: Verifying integrity of deep neural networks through sensitive-sample fingerprinting. arXiv preprint arXiv:1808.03277, 2018.

[13] Yangsibo Huang, Milad Nasr, Anastasios Angelopoulos, Nicholas Carlini, Wei-Lin Chiang, Christopher A Choquette-Choo, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Ken Ziyu Liu, et al. Exploring and mitigating adversarial manipulation of voting-based leaderboards. arXiv preprint arXiv:2501.07493, 2025.

[14] Jiawei Jiang, Yi Wei, Yu Liu, Wentao Wu, Chuang Hu, Zhigao Zheng, Ziyi Zhang, Yingxia Shao, and Ce Zhang. How good are machine learning clouds? benchmarking two snapshots over 5 years: J. jiang et al. The VLDB Journal, 33(3):833–857, 2024.

[15] Nikola Jovanovic, Robin Staab, and Martin Vechev. Watermark stealing in large language ´ models. arXiv preprint arXiv:2402.19361, 2024.

[16] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International Conference on Machine Learning, pages 17061–17084. PMLR, 2023.

[17] Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense. Advances in Neural Information Processing Systems, 36:27469–27500, 2023.

[18] Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, Xuming Hu, Xi Zhang, Lijie Wen, Irwin King, Hui Xiong, and Philip Yu. A survey of text watermarking in the era of large language models. ACM Computing Surveys, 57(2):1–36, 2024.

[19] OpenAI. New tools and features in the responses api. https://openai.com/index/ new-tools-and-features-in-the-responses-api/, 2024. Accessed 2025-01.

[20] Dario Pasquini, Evgenios M Kornaropoulos, and Giuseppe Ateniese. {LLMmap}: Fingerprint ing for large language models. In 34th USENIX Security Symposium (USENIX Security 25), pages 299–318, 2025.

[21] Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, and David Wagner. Markmywords: Analyzing and evaluating language model watermarks. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 68–91. IEEE, 2025.

[22] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[23] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

[24] Inioluwa Deborah Raji, Andrew Smart, Rebecca N White, Margaret Mitchell, Timnit Gebru, Ben Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. Closing the ai accountability gap: Defining an end-to-end framework for internal algorithmic auditing. In Proceedings of the 2020 conference on fairness, accountability, and transparency, pages 33–44, 2020.

[25] Saksham Rastogi and Danish Pruthi. Revisiting the robustness of watermarking to paraphrasing attacks. arXiv preprint arXiv:2411.05277, 2024.

[26] Guoheng Sun, Ziyao Wang, Bowei Tian, Meng Liu, Zheyu Shen, Shwai He, Yexiao He, Wanghao Ye, Yiting Wang, and Ang Li. Coin: Counting the invisible reasoning tokens in commercial opaque llm apis. arXiv preprint arXiv:2505.13778, 2025.

[27] Guoheng Sun, Ziyao Wang, Xuandong Zhao, Bowei Tian, Zheyu Shen, Yexiao He, Jinming Xing, and Ang Li. Invisible tokens, visible bills: The urgent need to audit hidden operations in opaque llm services. arXiv preprint arXiv:2505.18471, 2025.

[28] Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, and Huan Zhang. Svip: Towards verifiable inference of open-source large language models. arXiv preprint arXiv:2410.22307, 2024.

[29] Florian Tramer and Dan Boneh. Slalom: Fast, verifiable and private execution of neural networks in trusted hardware. arXiv preprint arXiv:1806.03287, 2018.

[30] Ziyao Wang, Guoheng Sun, Yexiao He, Zheyu Shen, Bowei Tian, and Ang Li. Predictive auditing of hidden tokens in llm apis via reasoning length estimation. arXiv preprint arXiv:2508.00912, 2025.

[31] Yi Wei, Xiaokai Zhou, Shanshan Feng, Chuang Hu, Xiao Yan, and Jiawei Jiang. Laua: Handling missing modalities and unpaired data in multimodal federated learning. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 5303–5313, 2026.

[32] Zhenghao Xu, Qin Lu, Qingru Zhang, Liang Qiu, Ilgee Hong, Changlong Yu, Wenlin Yao, Yao Liu, Haoming Jiang, Lihong Li, et al. Ask a strong llm judge when your reward model is uncertain. Advances in Neural Information Processing Systems, 38:74639–74664, 2026.

[33] Jiyu Yang, Qiang He, Zheyu Zhou, Xiaohai Dai, Feifei Chen, Cong Tian, and Yun Yang. Edgethemis: Ensuring model integrity for edge intelligence. In Proceedings of the ACM on Web Conference 2025, pages 3136–3146, 2025.

[34] Qinbo Zhang, Yanhang Shi, Ziyi Zhang, Hao Wang, Sai Qian Zhang, and Jian Li. Stealing split learning bottom models by recovering embedding geometry. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20660–20669, 2026.

[35] Xuandong Zhao, Prabhanjan Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for ai-generated text. arXiv preprint arXiv:2306.17439, 2023.

## A Additional Related Work Discussions

## A.1 LLM Watermarking

LLM Watermarking has been widely studied as a way to embed verifiable provenance signals into generated text, enabling attribution, provenance tracking, or misuse detection while preserving output fluency [7].

Most existing approaches introduce controlled statistical biases at generation time, for example by favoring identity-specific token subsets or by shaping token selection patterns during decoding [16, 35]. Detection is typically performed through hypothesis testing on token frequencies or related likelihood statistics computed from the generated text.

A common design is to embed the watermark directly into task-facing outputs, usually the final answers returned to users. This makes the detectability of the watermark sensitive to task semantics and prompt distributions [21]. In particular, low-entropy generations provide limited lexical degrees of freedom, so injecting a detectable bias can compete with answer fidelity and the watermark signal may be diluted by semantic constraints [18]. Moreover, embedding signals into final answers raises concerns about potential degradation of answer quality and can be weakened by post-processing such as paraphrasing or rewriting [25, 15, 6, 10]. These limitations motivate moving audit evidence away from task answers and into a separate user-visible channel. Our work follows this direction by watermarking reasoning summaries and enabling summary-only verification, which reduces coupling to task constraints and preserves answer quality.

## A.2 Model Auditing

Model auditing aims to verify whether a deployed machine learning system behaves as claimed, or whether specific properties of the model or inference process can be reliably certified [24, 4, 27]. Prior work has explored a range of approaches, including behavioral testing through challenge–response protocols [12, 33], hash tree-based verification [26], output-based fingerprinting using distributional or stylistic cues [9, 20], and verification mechanisms that rely on access to internal model states, inference logs, or trusted execution environments [29, 5]. These methods have been applied to settings such as model integrity verification, inference correctness, compliance auditing [12, 29], and token count auditing [26, 30]. Beyond model-level verification, prior work has explored broader reliability and security challenges in machine learning systems across diverse deployment settings [14, 31, 34, 32].

Recent work has begun to explicitly study model substitution in LLM services, where a provider may serve responses generated by a different model than the one advertised [3]. This problem is particularly challenging in black-box settings, as modern language models with similar architectures or training objectives can exhibit highly overlapping behaviors. Behavioral differences between models from the same family or between different parameter scales may be subtle, task-dependent, and easy to smooth or obscure through prompt engineering or response post-processing [3, 17]. As a result, auditing methods that rely solely on output behavior or stylistic features often provide limited robustness under realistic substitution strategies [3].

Many existing auditing and verification techniques further assume privileged access that is unavailable in hosted LLM services [4]. For example, some approaches require access to internal activations, inference traces, or trusted instrumentation during model execution [29, 5, 9], while others rely on cooperative providers that accurately report model identity or generation metadata [3]. In contrast, end users and external auditors interacting with commercial LLM services typically observe only a fixed, user-facing interface and must perform verification post hoc, without control over prompts, sampling procedures, or internal logging [3, 4].

Our work targets this more constrained but practically relevant auditing regime. We consider auditor-assisted model identity verification at deployment time, where a trusted auditor holds secret verification material but operates solely on user-visible outputs [2]. By focusing on reasoning summaries as the sole audit surface, our approach departs from prior model auditing methods and addresses model substitution under an information regime that closely matches real-world hosted LLM services.

## B Additional Experimental Details

Here, we provide additional details on the experimental setup, including the datasets and models used, as well as details of the generation process.

## B.1 Datasets

We use the following three datasets [30].

OpenR1-Math. The Math dataset consists of multi-step mathematical reasoning problems requiring symbolic manipulation or logical derivation. Prompts are formatted to elicit a final answer together with a concise reasoning summary. Summaries in this dataset are typically structured and steporiented, providing a high-density signal for evaluating watermark detectability under strong reasoning conditions.

OpenR1-Code. The Code dataset focuses on program synthesis and algorithmic problem solving. Each prompt requires generating code or an algorithmic solution, accompanied by a natural-language summary explaining the approach. Compared to Math, summaries exhibit greater lexical diversity and less rigid structure, testing robustness under heterogeneous reasoning styles.

General Reasoning. The General dataset includes open-domain reasoning and instruction-following tasks such as commonsense reasoning and explanation-style question answering. Summaries are shorter and more stylistically varied, representing the most challenging setting for summary-based auditing and reflecting common real-world usage.

## B.2 Models

Our primary benchmark uses four instruction-tuned checkpoints: Qwen2.5-1.5B-Instruct, Qwen2.5- 3B-Instruct, Qwen2.5-7B-Instruct, and LLaMA-3.2-3B-Instruct. These checkpoints are obtained from public repositories on Hugging Face.

The expanded evaluation in Appendix E.1 adds eight checkpoints, including 12 enrolled identities across four model families: Qwen (0.5B, 1.5B, 3B, 7B, 14B, and 32B), LLaMA (3B and 7B), Mistral (Ministral-3B and Mistral-7B), and Gemma (2B and 7B). The primary benchmark and expanded evaluation use different training budgets, as specified in Section 4.1 and Appendix E.1, respectively.

## C Watermark Design Details

## C.1 Greenlist-based Watermarking

We model a reasoning summary as a token sequence $\boldsymbol { s } = ( s _ { 1 } , \dots , s _ { T } )$ drawn from a vocabulary V. SUMMARK assigns each model identity m ∈ M a secret subset of tokens, called a greenlist $G _ { m } \subseteq V$ , derived from the auditor’s secret key. During summary generation, the watermark is implemented as an identity-dependent bias that increases the likelihood of sampling tokens from $G _ { m }$ while preserving fluency and semantic fidelity.

At decoding step $t ,$ let $z _ { t } \in \mathbb { R } ^ { | V | }$ be the output logits and $p _ { t } ( v ) = \mathrm { s o f t m a x } ( z _ { t } ) _ { v }$ be the induced token distribution. We quantify the instantaneous watermark signal via the log probability mass on $G _ { m } \mathrm { { : } }$

$$
\log P ( G _ { m } \mid z _ { t } ) \triangleq \log \sum _ { v \in G _ { m } } e ^ { z _ { t , v } } - \log \sum _ { v \in V } e ^ { z _ { t , v } } .\tag{4}
$$

Equivalently, $\begin{array} { r } { P ( G _ { m } \mid z _ { t } ) = \sum _ { v \in G _ { m } } p _ { t } ( v ) } \end{array}$ represents the probability that the next generated token belongs to $G _ { m }$ . Section $3 . 3$ describes how identity-specific adapters are trained on supervised summary data to increase the quantity defined in Eq. (4).

## C.2 Model Identities Capacity

The parameters $\rho$ and $\tau$ constrain how many model identities can be supported for a fixed vocabulary. Let $V$ be the public vocabulary with size $n \triangleq | V |$ . For each identity $m \in \mathcal { M }$ , the greenlist is a subset

$G _ { m } \subseteq V$ with size w $\triangleq | G _ { m } |$ , and the green ratio constraint in Eq. (1) gives w $\approx \rho n$ . To quantify collisions across identities, define the maximum pairwise intersection

$$
\lambda \triangleq \operatorname* { m a x } _ { m \neq m ^ { \prime } } | G _ { m } \cap G _ { m ^ { \prime } } | .\tag{5}
$$

Under the max-overlap constraint in Eq. (2), we have $| G _ { m } \cap G _ { m ^ { \prime } } | \leq \tau | G _ { m } |$ for all m $\neq m ^ { \prime }$ , hence $\lambda \leq \tau w$ . A standard packing bound for constant-size subsets then introduces

$$
B \leq \left\lfloor 1 + \frac { w ( n - w ) } { w ^ { 2 } - n \lambda } \right\rfloor \leq \left\lfloor 1 + \frac { w ( n - w ) } { w ^ { 2 } - n \tau w } \right\rfloor ,\tag{6}
$$

where $B \triangleq | { \mathcal { M } } |$ is the number of identities and the bound is informative when the denominator is positive. Eq. (6) shows that smaller τ improves separability but can reduce capacity.

## D Watermark Adapters Training

To induce identity-specific token preferences in the summary channel, we adopt a lightweight parameterization based on Low-Rank Adaptation (LoRA). For each identity $m \in \mathcal { M }$ , we train an adapter $\Delta _ { m }$ that biases the summary generator toward the identity-specific greenlist $G _ { m }$ . The base model parameters θ are frozen and only $\Delta _ { m }$ is optimized, so the watermark is implemented as a small, identity-specific parameter add-on rather than a separate generation pipeline.

## D.1 Training Target

We train each adapter $\Delta _ { m }$ using supervised summary data. Each training example $( c , s ^ { * } ) \sim \mathcal { D }$ consists of a conditioning input c used to generate the summary and a reference summary $s ^ { * } =$ $( w _ { 1 } , \dots , w _ { T } )$ . We use teacher forcing during training. At each decoding step, the model is given the ground-truth prefix of the summary and learns to predict the next token.

At step t, the adapted model $\left( \theta + \Delta _ { m } \right)$ outputs a logit vector $z _ { t } \in \mathbb { R } ^ { | V | }$ for every token in the vocabulary V . Applying a softmax converts these scores into a next-token probability distribution. Formally, we define

$$
p _ { t } ( v ) = \mathrm { s o f t m a x } ( z _ { t } ) _ { v } ,\tag{7}
$$

which denotes the probability of token v at step t.

Each identity m is associated with a predefined set of tokens, called the greenlist $G _ { m }$ . Intuitively, a watermark is stronger if the model tends to place more probability on these greenlist tokens during generation. We therefore measure the watermark strength by the total probability mass assigned to $\hat { G } _ { m }$

$$
P ( G _ { m } | z _ { t } ) = \sum _ { v \in G _ { m } } p _ { t } ( v ) ,\tag{8}
$$

which is the probability that the next token belongs to the greenlist. For numerical stability and convenience, we use its log-scaled form:

$$
\log P ( G _ { m } \mid z _ { t } ) = \log \sum _ { v \in G _ { m } } e ^ { z _ { t , v } } - \log \sum _ { v \in V } e ^ { z _ { t , v } } .\tag{9}
$$

Equivalently, $\begin{array} { r } { P ( G _ { m } \mid z _ { t } ) = \sum _ { v \in G _ { m } } p _ { t } ( v ) } \end{array}$ is the probability that the next token sampled from $p _ { t }$ lies in the greenlist $G _ { m }$ . We refer to $P ( G _ { m } \mid z _ { t } )$ as the green mass at step $t ,$ and to log $P ( G _ { m } \mid z _ { t } )$ as its log-scaled form. A larger green mass means the model is more biased toward emitting identity-m greenlist tokens, which results in a stronger and more detectable watermark signal.

During training, we freeze θ and update only $\Delta _ { m }$ . This ensures that the watermark is encoded through a small, identity-specific adapter, rather than by modifying or retraining the full model.

## D.2 Loss Components

Given training pairs $( c , s ^ { * } ) \sim \mathcal { D }$ , we optimize the summary adapter $\Delta _ { m }$ with a base identity objective and an optional precision-consistency regularizer. The base identity objective is

$$
L _ { \mathrm { i d } } = L _ { \mathrm { s f t } } + \lambda _ { \mathrm { w m } } L _ { \mathrm { w m } } + \lambda _ { \mathrm { s e p } } L _ { \mathrm { s e p } } ,\tag{10}
$$

where $\lambda _ { \mathrm { w m } } , \lambda _ { \mathrm { s e p } } \geq 0$ control watermark strength and identity separability. Throughout this section, logits $z _ { t }$ are produced by the adapted model $\theta _ { m } + \Delta _ { m }$ under teacher forcing on the reference summary. All losses are averaged over $( c , \bar { s } ^ { * } ) \sim \mathcal { D }$

Utility constraint: $L _ { \mathrm { s f t } }$ . We use the standard teacher-forced negative log-likelihood on reference summaries:

$$
L _ { \mathrm { s f t } } = - \mathbb { E } _ { \mathcal { D } } \log P _ { \theta _ { m } + \Delta _ { m } } ( s ^ { * } \mid c ) .\tag{11}
$$

This anchors generation to natural summaries and mitigates semantic drift caused by watermark pressure.

Detectability: $L _ { \mathrm { w m } } .$ To strengthen the identity-specific watermark, we encourage larger average log probability mass on the carrier set $G _ { m }$

$$
L _ { \mathrm { w m } } = - \mathbb { E } _ { \mathcal { D } } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log P ( G _ { m } \mid z _ { t } ) \right] .\tag{12}
$$

This term increases the statistical footprint associated with identity m, making the watermark more detectable from generated summaries.

Attribution separability: $L _ { \mathrm { s e p } } .$ To reduce confusion between enrolled identities, we enforce that the carrier mass of the correct identity exceeds that of competing identities by a margin $\mu .$ Let ${ \mathcal { N } } \subset { \mathcal { M } } \setminus \{ m \}$ be a sampled set of negative identities. We use a hardest-negative margin loss:

$$
L _ { \mathrm { s e p } } = \mathbb { E } _ { \mathcal { D } } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m a x } \left( 0 , \mu + \operatorname* { m a x } _ { m ^ { \prime } \in \mathcal { N } } \log P ( G _ { m ^ { \prime } } \mid z _ { t } ) - \log P ( G _ { m } \mid z _ { t } ) \right) \right] .\tag{13}
$$

This term targets the most confusable competing identity and improves attribution reliability when carrier sets partially overlap.

Precision-consistency regularizer. For precision-variant diagnostics, FP16, INT8, and INT4 variants of the same base identity m share the same carrier set $\hat { G } _ { m }$ and the same adapter $\Delta _ { m }$ . We add a quantization-sensitive regularizer that encourages the FP16 execution to produce a stronger carrier-set footprint than quantized executions.

For a precision state $p \in \{ \mathrm { F P 1 6 , I N T 8 , I N T 4 } \}$ , define the average carrier-set score

$$
A _ { p } ( m ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log P _ { p } ( G _ { m } \mid z _ { t } ^ { ( p ) } ) ,\tag{14}
$$

where $z _ { t } ^ { ( p ) }$ denotes the logits produced by the shared adapter $\Delta _ { m }$ under precision state $p .$ We define

$$
{ \cal L } _ { \mathrm { q s } } = \mathbb { E } _ { q \in \{ \mathrm { I N T 8 , I N T 4 } \} } \left[ \left( \gamma - \left( A _ { \mathrm { F P 1 6 } } ( m ) - A _ { q } ( m ) \right) \right) _ { + } \right] ,\tag{15}
$$

where $( u ) _ { + } = \operatorname* { m a x } ( 0 , u )$ and $\gamma > 0$ is a precision-separation margin. This loss penalizes cases where the FP16 carrier-set score is not sufficiently larger than the INT8/INT4 score.

The full precision-sensitive training objective is

$$
\begin{array} { r } { L = L _ { \mathrm { i d } } + \lambda _ { \mathrm { q s } } L _ { \mathrm { q s } } . } \end{array}\tag{16}
$$

This regularizer does not create precision-specific identities, carrier sets, or adapters. It only makes the shared identity-level verifier more informative for FP16-to-INT8/INT4 precision-consistency diagnostics.

## D.3 Detection

At audit time, the auditor observes only the released summary s and the claimed identity $m _ { c } .$ . Let $s = ( x _ { 1 } , \dots , x _ { T } )$ be the tokenized summary, where each token $x _ { t } \in V$ . For each candidate identity $m \in M$ with keyed carrier bucket $G _ { m } \subseteq V$ , define

$$
g _ { m } ( x _ { t } ) \triangleq \mathbb { I } [ x _ { t } \in G _ { m } ] , \qquad C _ { m } ( s ) \triangleq \sum _ { t = 1 } ^ { T } g _ { m } ( x _ { t } ) .
$$

Table 7: Substitution detection on Math (TPR@FPR=1%, ACC@FPR=1%, AUC). Setting A→B: serve B, claim A. Bold is best.
<table><tr><td></td><td colspan="3">BERT</td><td colspan="3">GPT-2</td><td colspan="3">T5</td><td colspan="3">SUMMARK</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Qwen-7B→Qwen-3B</td><td>0.1232</td><td>0.5602</td><td>0.9221</td><td>0.0000</td><td>0.5000</td><td>0.4458</td><td>0.6359</td><td>0.8053</td><td>0.9348</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-7B→Qwen-1.5B</td><td>0.9412</td><td>0.9706</td><td>0.9955</td><td>0.2105</td><td>0.5965</td><td>0.6037</td><td>0.9804</td><td>0.9902</td><td>0.9939</td><td>0.9960</td><td>0.9930</td><td>0.9974</td></tr><tr><td>Qwen-3B→Qwen-1.5B</td><td>0.9720</td><td>0.9790</td><td>0.9959</td><td>0.2807</td><td>0.6316</td><td>0.8339</td><td>0.9804</td><td>0.9748</td><td>0.9976</td><td>1.0000</td><td>0.9950</td><td>0.9998</td></tr><tr><td>Qwen-3B→LLaMA-3B</td><td>0.9076</td><td>0.9468</td><td>0.9939</td><td>0.6842</td><td>0.8333</td><td>0.9120</td><td>0.7003</td><td>0.8235</td><td>0.9605</td><td>1.0000</td><td>0.9950</td><td>0.9997</td></tr><tr><td>LLaMA-3B→Qwen-3B</td><td>0.8431</td><td>0.9202</td><td>0.9939</td><td>0.3684</td><td>0.6754</td><td>0.9477</td><td>0.8627</td><td>0.9300</td><td>0.9745</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-7B→LLaMA-3B</td><td>0.6050</td><td>0.8011</td><td>0.9871</td><td>0.2281</td><td>0.6053</td><td>0.8507</td><td>0.9272</td><td>0.9510</td><td>0.9941</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>LLaMA-3B→Qwen-1.5B</td><td>0.9860</td><td>0.9930</td><td>0.9999</td><td>0.5614</td><td>0.7719</td><td>0.9843</td><td>0.9860</td><td>0.9930</td><td>0.9955</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr></table>

Table 8: Substitution detection on General (TPR@FPR=1%, ACC@FPR=1%, AUC). Setting $A \to B ;$ serve B, claim A. Q7/Q3/Q1.5 = Qwen-7B/3B/1.5B, L3 = LLaMA-3B. Bold is best.
<table><tr><td></td><td colspan="3">BERT</td><td colspan="3">GPT-2</td><td colspan="3">T5</td><td colspan="3">SUMMARK</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Qwen-7B→Qwen-3B</td><td>0.7895</td><td>0.8772</td><td>0.9621</td><td>0.0000</td><td>0.5000</td><td>0.6331</td><td>0.7193</td><td>0.8596</td><td>0.9898</td><td>0.9990</td><td>0.9945</td><td>0.9999</td></tr><tr><td>Qwen-7B→Qwen-1.5B</td><td>0.1228</td><td>0.5439</td><td>0.8689</td><td>0.0526</td><td>0.5263</td><td>0.5389</td><td>0.0000</td><td>0.5000</td><td>0.9514</td><td>0.8740</td><td>0.9320</td><td>0.9954</td></tr><tr><td>Qwen-3B→Qwen-1.5B</td><td>0.0877</td><td>0.5439</td><td>0.9267</td><td>0.1053</td><td>0.5526</td><td>0.7014</td><td>0.0000</td><td>0.5000</td><td>0.9338</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-3B→LLaMA-3B</td><td>0.5439</td><td>0.7719</td><td>0.9298</td><td>0.3509</td><td>0.6754</td><td>0.9464</td><td>0.5088</td><td>0.7544</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>0.9999</td></tr><tr><td>LLaMA-3B→Qwen-3B</td><td>0.8596</td><td>0.9211</td><td>0.9917</td><td>0.3684</td><td>0.6842</td><td>0.9372</td><td>0.9474</td><td>0.9561</td><td>0.9800</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-7B→LLaMA-3B</td><td>0.5965</td><td>0.7807</td><td>0.9231</td><td>0.3509</td><td>0.6754</td><td>0.9323</td><td>0.5439</td><td>0.7719</td><td>0.9471</td><td>0.9920</td><td>0.9910</td><td>0.9988</td></tr><tr><td>LLaMA-3B→Qwen-1.5B</td><td>0.8070</td><td>0.8947</td><td>0.9785</td><td>0.5263</td><td>0.7632</td><td>0.9812</td><td>0.9298</td><td>0.9474</td><td>0.9898</td><td>0.9600</td><td>0.9750</td><td>0.9988</td></tr></table>

Although a binomial null would suggest $\mathbb { E } [ C _ { m } ( s ) ] \approx \rho T$ , all reported decisions use empirical calibration rather than this approximation. For each enrolled identity m, we estimate length-conditioned calibration statistics $\mu _ { m } ( T )$ and $\sigma _ { m } ( T )$ from held-out benign summaries generated under the enrolled identity and its summary adapter. The calibrated score is

$$
Z _ { m } ( s ) \triangleq { \frac { C _ { m } ( s ) - \mu _ { m } ( T ) } { \sigma _ { m } ( T ) + \epsilon } } ,
$$

where ϵ is a small numerical constant. Intuitively, summaries consistent with identity m should have calibrated scores comparable to the held-out benign calibration distribution, while summaries generated under substituted identities tend to have lower scores for the claimed carrier bucket.

Model substitution detection. Given a claimed identity $m _ { c } ,$ the verifier accepts the claim if

$$
Z _ { m _ { c } } ( s ) \geq \eta _ { m _ { c } } ( \alpha ) ,
$$

where $\eta _ { m _ { c } } ( \alpha )$ is chosen as the lower-tail empirical threshold on held-out benign summaries so that the false-positive rate is at most α. Otherwise, the verifier flags the released summary as inconsistent with the claimed identity. Thus the operating threshold is calibrated from held-out summaries rather than assumed from the binomial approximation.

Model attribution. For attribution, the auditor scores the released summary against every enrolled identity and predicts

$$
{ \widehat { m } } ( s ) \triangleq \arg \operatorname* { m a x } _ { m \in M } Z _ { m } ( s ) ,
$$

optionally abstaining when the top-two score margin is below a fixed threshold. Both detection and attribution rely only on summary-level token statistics and the keyed carrier buckets.

## E Additional Experiments

## E.1 Identity-Substitution Detection

Per-pair Substitution Detection Results. Tables 7, 8, and 9 report model substitution detection on Math, General, and Code, respectively. The first three rows in each table correspond to within-family size substitution (Qwen → Qwen), the next two to cross-family substitution (Qwen ↔ LLaMA), and the last two to combined substitutions (family + size).

Across datasets and substitution types, SUMMARK is consistently strong under strict auditing constraints, achieving very high TPR@FPR= 1% in most settings and uniformly high AUC. In contrast, text classifiers (BERT/GPT-2/T5) are unstable across settings: they can show high AUC in some pairs, yet their TPR@1% varies widely, which is most notably for within-family size substitutions where stylistic cues are subtle and easily washed out. This AUC-TPR gap indicates that while some substitutions are separable in aggregate, surface cues are often too brittle to support reliable decisions at a strict operating point (FPR= 1%). For instance on Math, BERT’s TPR@1% collapses for Qwen-7B→Qwen-3B (0.1232) while SUMMARK remains 1.0000, illustrating that style-based signals are not reliable when the substituted model is closely related to the claimed one.

Table 9: Substitution detection on Code (TPR@FPR=1%, ACC@FPR=1%, AUC). Setting A→B: serve B, claim A. Bold is best.
<table><tr><td></td><td colspan="3">BERT</td><td colspan="3">GPT-2</td><td colspan="3">T5</td><td colspan="3">SUMMARK</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Qwen-7B→Qwen-3B</td><td>0.6491</td><td>0.8246</td><td>0.9825</td><td>0.0000</td><td>0.5000</td><td>0.5720</td><td>0.8947</td><td>0.9474</td><td>0.9963</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-7B→Qwen-1.5B</td><td>0.8246</td><td>0.9123</td><td>0.9834</td><td>0.1400</td><td>0.5700</td><td>0.6690</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-3B→Qwen-1.5B</td><td>0.0000</td><td>0.5000</td><td>0.9769</td><td>0.5090</td><td>0.7110</td><td>0.7960</td><td>0.1053</td><td>0.5526</td><td>0.9982</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-3B→LLaMA-3B</td><td>0.2281</td><td>0.6140</td><td>0.9991</td><td>0.7370</td><td>0.8250</td><td>0.9330</td><td>0.3684</td><td>0.6842</td><td>0.9942</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>LLaMA-3B→Qwen-3B</td><td>0.9474</td><td>0.9737</td><td>0.9991</td><td>0.4040</td><td>0.6750</td><td>0.9160</td><td>0.8947</td><td>0.9474</td><td>0.9935</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Qwen-7B→LLaMA-3B</td><td>0.9825</td><td>0.9912</td><td>1.0000</td><td>0.2810</td><td>0.6400</td><td>0.8860</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>LLaMA-3B→Qwen-1.5B</td><td>0.9649</td><td>0.9825</td><td>1.0000</td><td>0.6490</td><td>0.7980</td><td>0.9500</td><td>0.9825</td><td>0.9912</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr></table>

Table 10: Substitution detection on Math (TPR@FPR=1%, ACC@FPR=1%, AUC). Setting A→B: serve B, claim A. Q7/Q3/Q1.5 = Qwen-7B/3B/1.5B, L3 = LLaMA-3B. Bold is best.
<table><tr><td></td><td colspan="3">BERT</td><td colspan="3">GPT-2</td><td colspan="3">T5</td><td colspan="3">IDP</td><td colspan="3">SUMMARK</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td></td><td>TPR@1% ACC@1%</td><td>AUC</td></tr><tr><td>Q7 → Q3</td><td>0.1232</td><td>0.5602</td><td>0.9221</td><td>0.0000</td><td>0.5000</td><td>0.4458</td><td>0.6359</td><td>0.8053</td><td>0.9348</td><td>0.0000</td><td>0.5000</td><td>0.5000</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Q7 → Q1.5</td><td>0.9412</td><td>0.9706</td><td>0.9955</td><td>0.2105</td><td>0.5965</td><td>0.6037</td><td>0.9804</td><td>0.9902</td><td>0.9939</td><td>0.0000</td><td>0.5000</td><td>0.5000</td><td>0.9960</td><td>0.9930</td><td>0.9974</td></tr><tr><td>Q3 → Q1.5</td><td>0.9720</td><td>0.9790</td><td>0.9959</td><td>0.2807</td><td>0.6316</td><td>0.8339</td><td>0.9804</td><td>0.9748</td><td>0.9976</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>0.9998</td></tr><tr><td>Q3 → L3</td><td>0.9076</td><td>0.9468</td><td>0.9939</td><td>0.6842</td><td>0.8333</td><td>0.9120</td><td>0.7003</td><td>0.8235</td><td>0.9605</td><td>0.0000</td><td>0.5000</td><td>0.5000</td><td>1.0000</td><td>0.9950</td><td>0.9997</td></tr><tr><td>L3 → Q3</td><td>0.8431</td><td>0.9202</td><td>0.9939</td><td>0.3684</td><td>0.6754</td><td>0.9477</td><td>0.8627</td><td>0.9300</td><td>0.9745</td><td>0.0000</td><td>0.5000</td><td>0.5000</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Q7 → L3</td><td>0.6050</td><td>0.8011</td><td>0.9871</td><td>0.2281</td><td>0.6053</td><td>0.8507</td><td>0.9272</td><td>0.9510</td><td>0.9941</td><td>0.0000</td><td>0.5000</td><td>0.5000</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>L3 → Q1.5</td><td>0.9860</td><td>0.9930</td><td>0.9999</td><td>0.5614</td><td>0.7719</td><td>0.9843</td><td>0.9860</td><td>0.9930</td><td>0.9955</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr></table>

This contrast matches the design intent of SUMMARK. Text classifiers implicitly learn domain- and dataset-specific surface regularities (phrasing, formatting, recurring templates), which may transfer poorly across tasks and are weakest when two models share the same family and training pipeline. By comparison, SUMMARK encodes identity-specific evidence into the summary channel via keyed vocabulary buckets and watermark adapters, so substitution induces a predictable shift in summarylevel token statistics rather than afragile stylistic change. This explains why SUMMARK remains effective for size substitution, family substitution, and combined substitution across domains.

Why General can be harder. We see a few weaker cases on General dataset (Table 8); for example, Qwen-7B→Qwen-1.5B has TPR@1% = 0.8740. General summaries are more conversational and stylistically diverse, with fewer repeated templates than Code, which increases lexical entropy and reduces the effective per-token evidence that our count-based detector can accumulate. Moreover, General summaries rely heavily on high-frequency function words and generic phrasing; when ubiquitous tokens fall into a greenlist, non-watermarked summaries can match portions of the greenlist by chance, narrowing separation specifically at very low-FPR thresholds. Importantly, separability remains strong overall: this same setting retains a high AUC (0.9954), indicating the watermark signal is present but stricter operating points (e.g., FPR= 1%) can incur more misses.

Why Code can make baselines look strong. On the Code dataset (Table 9), summaries are more structured and often follow recurring templates, e.g., step-by-step debugging narratives, standard error-handling language, and repeated references to functions/variables. This regularity can make strong encoders (notably T5) appear competitive in some substitution pairs by exploiting surface-level cues. However, this advantage is largely domain-specific: the same classifiers are much less consistent on Math and General, reinforcing that their performance often reflects exploitable formatting/style regularities rather than a verifiable, identity-bound signal. By contrast, SUMMARK remains reliable across domains and substitution types because its evidence is keyed and identity-specific rather than stylistic.

Comparing with Identity prompting (IDP). Prompt the model to disclose its identity (e.g., “Who are you?”, “Who developed you?”) and string-match against known identifiers (family, developer), following the simple classifier setup in Huang et al. [13] We compare our method SUMMARK with IDP on Math Dataset. Table 10 shows that IDP performs poorly on substitution detection across most settings on the Math dataset. In many cases, IDP yields a TPR@1% of 0 with ACC@1% fixed at 0.5 and AUC at 0.5, which is equivalent to random guessing. This indicates that identity prompting fails to distinguish between the claimed model and the serving model under strict false positive constraints. Only in a few settings does IDP reach perfect scores, but these results are inconsistent and highly setting-dependent. Overall, the results suggest that IDP lacks robustness and cannot reliably detect model substitution in this task.

Table 11: Attribution performance on Math. Higher is better.
<table><tr><td>Method</td><td>ACC</td><td>Macro-F1</td></tr><tr><td>BERT</td><td>0.9386</td><td>0.9383</td></tr><tr><td>GPT-2</td><td>0.7121</td><td>0.6981</td></tr><tr><td>T5</td><td>0.9128</td><td>0.9122</td></tr><tr><td>IDP</td><td>0.7500</td><td>0.6670</td></tr><tr><td>SUMMARK</td><td>0.9958</td><td>0.9957</td></tr></table>

Table 12: Identity-Substitution detection on Math. SynthID and Gumbel Max are applied directly to the task-facing answer, while SUMMARK applies the watermark only to the summary channel. We report TPR@FPR=1%, ACC@FPR=1%, and AUC.
<table><tr><td></td><td colspan="3">SynthID</td><td colspan="3">Gumbel Max</td><td colspan="3">SUMMARK</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Q7→ Q3</td><td>0.0000</td><td>0.4950</td><td>0.5350</td><td>0.0500</td><td>0.5200</td><td>0.6041</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Q7→ Q1.5</td><td>0.0350</td><td>0.5125</td><td>0.5623</td><td>0.0300</td><td>0.5100</td><td>0.5463</td><td>0.9960</td><td>0.9930</td><td>0.9974</td></tr><tr><td>Q3→ Q1.5</td><td>0.0700</td><td>0.5300</td><td>0.5235</td><td>0.0200</td><td>0.5050</td><td>0.4112</td><td>1.0000</td><td>0.9950</td><td>0.9998</td></tr><tr><td>Q3→ L3</td><td>0.0250</td><td>0.5075</td><td>0.5216</td><td>0.0450</td><td>0.5175</td><td>0.4143</td><td>1.0000</td><td>0.9950</td><td>0.9997</td></tr><tr><td>L3→ Q3</td><td>0.0000</td><td>0.4950</td><td>0.4727</td><td>0.0000</td><td>0.4950</td><td>0.4770</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>Q7→ L3</td><td>0.0750</td><td>0.5325</td><td>0.5737</td><td>0.0150</td><td>0.5025</td><td>0.4531</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>L3→ Q1.5</td><td>0.0250</td><td>0.5075</td><td>0.4680</td><td>0.0100</td><td>0.5000</td><td>0.4352</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr></table>

And we also conduct attribution detection for IDP. Table 11 reports attribution performance on the Math dataset. The IDP baseline achieves relatively low performance on the Math dataset, with an accuracy of 0.75 and a Macro-F1 of 0.667. This gap compared to stronger attribution methods indicates that identity prompting combined with simple string matching is not sufficient for robust attribution.

Comparing with answer-level watermarking methods. Table 12 shows that applying distortionfree watermarks directly to task-facing answers is insufficient for identity substitution detection on Math. Both SynthID and Gumbel Max achieve near-zero TPR@FPR=1% in most settings, with accuracies close to random guessing and AUCs mostly near chance. In contrast, SUMMARK achieves 0.996–1.000 TPR@1%, 0.993–0.995 ACC@1%, and consistently high AUC across all substitution pairs. This gap indicates that the task-facing answer is a weak carrier for identity auditing under strict false-alarm constraints.

Substitution results on TPR@FPR = 5%. Table 13 compares our method with text-only baselines under a relaxed auditing constraint of FPR = 5% on Math dataset. Our approach achieves perfect or near-perfect detection across all substitution settings, consistently reaching TPR near to 1. In contrast, the baselines improve or remain unchanged under the relaxed threshold, but remain less consistent across model pairs. BERT improves under the relaxed threshold but remains unstable, particularly for cross-family substitutions. GPT-2 performs poorly in most settings, indicating limited sensitivity to model-identity cues. T5 shows relatively strong performance on some size substitutions but degrades noticeably on harder cross-family cases. These results demonstrate that even when allowing higher false positive rates, our method provides significantly stronger and more reliable substitution detection than text-based classifiers, highlighting its robustness and scalability under practical auditing conditions.

Substitution Detection under Quantized Serving. Table 18 evaluates whether substitution detection remains reliable when the served substitute model is run at different numerical precisions. Across FP16, INT8, and INT4, SUMMARK maintains consistently high performance: TPR@1% remains between 0.9860 and 1.0000, ACC@1% stays around 0.9880–0.9950, and AUC is close to 1.0 in all settings.

Table 13: Identity-Substitution detection performance on Math dataset. We report TPR@FPR=1% and TPR@FPR=5%. Best is bold.
<table><tr><td></td><td colspan="2">BERT</td><td colspan="2">GPT-2</td><td colspan="2">T5</td><td colspan="2">SUMMARK</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>TPR@5%</td><td>TPR@1%</td><td>TPR@5%</td><td>TPR@1%</td><td>TPR@5%</td><td>TPR@1%</td><td>TPR@5%</td></tr><tr><td>Qwen-7B→Qwen-3B</td><td>0.1232</td><td>0.5322</td><td>0.0000</td><td>0.0351</td><td>0.6359</td><td>0.8095</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Qwen-7B→Qwen-1.5B</td><td>0.9412</td><td>0.9720</td><td>0.2105</td><td>0.2807</td><td>0.9804</td><td>0.9804</td><td>0.9960</td><td>0.9970</td></tr><tr><td>Qwen-3B→Qwen-1.5B</td><td>0.9720</td><td>0.9720</td><td>0.2807</td><td>0.2807</td><td>0.9804</td><td>0.9804</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Qwen-3B→LLaMA-3B</td><td>0.9076</td><td>0.9692</td><td>0.6842</td><td>0.8070</td><td>0.7003</td><td>0.7199</td><td>1.0000</td><td>1.0000</td></tr><tr><td>LLaMA-3B→Qwen-3B</td><td>0.8431</td><td>0.9160</td><td>0.3684</td><td>0.7368</td><td>0.8627</td><td>0.8627</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Qwen-7B→LLaMA-3B</td><td>0.6050</td><td>0.9160</td><td>0.2281</td><td>0.4737</td><td>0.9272</td><td>0.9888</td><td>1.0000</td><td>1.0000</td></tr><tr><td>LLaMA-3B→Qwen-1.5B</td><td>0.9860</td><td>0.9860</td><td>0.5614</td><td>0.9474</td><td>0.9860</td><td>0.9860</td><td>1.0000</td><td>1.0000</td></tr></table>

The results show that quantizing the substituted model does not weaken the identity-substitution signal. In most cases, INT8 and INT4 perform comparably to FP16, and sometimes even slightly better. This suggests that SUMMARK’s detector is primarily driven by the mismatch between the claimed identity’s summary-channel footprint and the served model’s footprint, rather than by fine-grained numerical precision effects.

Expanded enrolled universe. To study scalability, we enlarge the enrolled universe to 12 identities from four families: Qwen (0.5B, 1.5B, 3B, 7B, 14B, 32B), LLaMA (3B, 7B), Mistral (Ministral-3B, Mistral-7B), and Gemma (2B, 7B). All methods use 1,000 training examples per identity. These results are reported separately from the primary benchmark in Section 4.1. For compactness, we report TPR@1% and AUC in the scaling experiments.

Identity andfamily scaling. The nested identity sets start with Qwen-1.5B/3B/7B and LLaMA-3B, then successively add Mistral-3B/7B, LLaMA-7B and Qwen-0.5B, Gemma-2B/7B, and Qwen-14B/32B, yielding 4/6/8/10/12 identities. With these sets (Table 14), SUMMARK’s macro TPR@1% drops from 0.9986 at 4 identities to a minimum of 0.7462 at 8, then reaches 0.8187 at 12, still above T5 (0.7518) and GPT-2 (0.7188). The advantage is not uniform: T5 leads at 6 identities, and the classifiers attain higher AUC at larger counts. For family scaling, we select Qwen-3B/7B, LLaMA-3B/7B, Ministral-3B and Mistral-7B, and Gemma-2B/7B as the two identities for each respective family. Holding these selections fixed and averaging over all family subsets of each size (Table 15), SUMMARK leads in TPR@1% at every family count but trails in AUC. In other words, it is stronger at the calibrated low-FPR operating point than in global ranking.

Table 14: Identity-count scaling on Math (nested identity sets, 1,000 training examples per identity). Best is bold.
<table><tr><td rowspan="2">#Identities</td><td colspan="2">GPT-2</td><td colspan="2">T5</td><td colspan="2">SUMMARK</td></tr><tr><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td></tr><tr><td>4</td><td>0.6250</td><td>0.9735</td><td>0.8279</td><td>0.9774</td><td>0.9986</td><td>0.9987</td></tr><tr><td>6</td><td>0.6213</td><td>0.9807</td><td>0.8089</td><td>0.9819</td><td>0.7811</td><td>0.9397</td></tr><tr><td>8</td><td>0.5900</td><td>0.9792</td><td>0.7360</td><td>0.9766</td><td>0.7462</td><td>0.9105</td></tr><tr><td>10</td><td>0.7015</td><td>0.9826</td><td>0.7340</td><td>0.9765</td><td>0.7919</td><td>0.9339</td></tr><tr><td>12</td><td>0.7188</td><td>0.9833</td><td>0.7518</td><td>0.9776</td><td>0.8187</td><td>0.9443</td></tr></table>

Table 15: Family-count scaling on Math. Each family contributes two identities; results average over all family subsets, so the 4-family point uses 8 identities. Best is bold.
<table><tr><td rowspan="3">#Families</td><td colspan="2">GPT-2</td><td colspan="2">T5</td><td colspan="2">SUMMARK</td></tr><tr><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td></tr><tr><td>1</td><td>0.5788</td><td>0.9646</td><td>0.5825</td><td>0.9382</td><td>0.7850</td><td>0.9527</td></tr><tr><td>2</td><td>0.6344</td><td>0.9769</td><td>0.6702</td><td>0.9626</td><td>0.8251</td><td>0.9603</td></tr><tr><td>3</td><td>0.6501</td><td>0.9795</td><td>0.6487</td><td>0.9670</td><td>0.8121</td><td>0.9596</td></tr><tr><td>4</td><td>0.6977</td><td>0.9836</td><td>0.7061</td><td>0.9720</td><td>0.7935</td><td>0.9543</td></tr></table>

Larger scale gaps. We test substitutions in which the served model is 4.7–64.0× smaller than the claimed one (Table 16). SUMMARK’s macro TPR@1% is 0.9250 within family and 0.9367 across families, compared with 0.6942/0.8467 for T5 and 0.6467/0.8333 for GPT-2. Detection depends on the specific model pair and does not improve monotonically with the size gap. Within-family substitutions from Qwen-32B achieve TPR@1% of 0.9900–1.0000, but Qwen-14B→Qwen-0.5B (0.69) and Qwen-14B→Ministral-3B (0.74) are weaker, and T5 is perfect on both Ministral-3B directions. These results use public checkpoints and do not establish generalization to proprietary serving stacks.

Table 16: Identity-Substitution detection with large parameter gaps on Math (1,000 training examples per identity). Red. is the claimed-to-served parameter ratio; macro averages are over the six pairs in each panel. Best is bold.
<table><tr><td colspan="2"></td><td colspan="2">GPT-2</td><td colspan="2">T5</td><td colspan="2">SUMMARK</td></tr><tr><td>Setting</td><td>Red.</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td></tr><tr><td colspan="8">Within family</td></tr><tr><td>Qwen-32B→Qwen-0.5B</td><td>64.0×</td><td>0.7800</td><td>0.9866</td><td>0.8850</td><td>0.9924</td><td>1.0000</td><td>0.9999</td></tr><tr><td>Qwen-32B→Qwen-1.5B</td><td>21.3×</td><td>0.8250</td><td>0.9891</td><td>0.9700</td><td>0.9986</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Qwen-32B→Qwen-3B</td><td>10.7×</td><td>0.8600</td><td>0.9912</td><td>0.9350</td><td>0.9966</td><td>0.9900</td><td>0.9997</td></tr><tr><td>Qwen-14B→Qwen-0.5B</td><td>28.0×</td><td>0.3150</td><td>0.9374</td><td>0.3450</td><td>0.8949</td><td>0.6900</td><td>0.9278</td></tr><tr><td>Qwen-14B→Qwen-1.5B</td><td>9.3×</td><td>0.8300</td><td>0.9926</td><td>0.9000</td><td>0.9971</td><td>0.8700</td><td>0.9667</td></tr><tr><td>Qwen-14B→Qwen-3B</td><td>4.7×</td><td>0.2700</td><td>0.9641</td><td>0.1300</td><td>0.9247</td><td>1.0000</td><td>0.9988</td></tr><tr><td>Macro average</td><td></td><td>0.6467</td><td>0.9768</td><td>0.6942</td><td>0.9674</td><td>0.9250</td><td>0.9822</td></tr><tr><td colspan="8"></td></tr><tr><td>Cross family Qwen-32B→LLaMA-3B</td><td>10.7×</td><td>0.8850</td><td>0.9922</td><td>0.9900</td><td>0.9993</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Qwen-32B→Ministral-3B</td><td>10.7×</td><td>0.9800</td><td>0.9952</td><td>1.0000</td><td>1.0000</td><td>0.8800</td><td>0.9745</td></tr><tr><td>Qwen-32B→Gemma-2B</td><td>16.0×</td><td>0.9350</td><td>0.9943</td><td>0.9800</td><td>0.9979</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Qwen-14B→LLaMA-3B</td><td>4.7×</td><td>0.3150</td><td>0.9724</td><td>0.3050</td><td>0.9639</td><td>1.0000</td><td>0.9991</td></tr><tr><td>Qwen-14B→Ministral-3B</td><td>4.7×</td><td>0.9750</td><td>0.9985</td><td>1.0000</td><td>1.0000</td><td>0.7400</td><td>0.9532</td></tr><tr><td>Qwen-14B→Gemma-2B</td><td>7.0×</td><td>0.9100</td><td>0.9975</td><td>0.8050</td><td>0.9883</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Macro average</td><td>一</td><td>0.8333</td><td>0.9917</td><td>0.8467</td><td>0.9916</td><td>0.9367</td><td>0.9878</td></tr></table>

Attribution. With all 12 identities enrolled, SUMMARK’s exact-identity accuracy is 87.83% (GPT-2: 82.50%; T5: 77.38%; chance: 8.33%). Mapping the same predictions to families gives 93.17%; this is not a separate four-class experiment.

Held-out substitutes. An auditor cannot enumerate every possible substitute, so we exclude test substitutes from the $L _ { \mathrm { s e p } }$ negative pool (Table 17). In the held-out-family setting, LLaMA-3B is excluded for the Qwen-3B/7B adapters. In the held-out-scale setting, Qwen-1.5B is excluded for the other adapters. Aggregate TPR@1% is 0.9990 and 0.9973, respectively. This supports rejecting unseen substitutes under the claimed identity’s keyed test, not attributing them. Precision variants are not used as negatives in $L _ { \mathrm { s e p } }$

Table 17: Claim rejection on Math for substitutes held out of the $L _ { \mathrm { s e p } }$ negative-identity pool. The task is claim rejection, not open-set attribution.
<table><tr><td rowspan="2">Held-out axis</td><td rowspan="2">Setting</td><td colspan="2">SUMMARK</td></tr><tr><td>TPR@1%</td><td>AUC</td></tr><tr><td rowspan="3">Family</td><td>Qwen-3B→LLaMA-3B</td><td>0.9990</td><td>0.9991</td></tr><tr><td>Qwen-7B→LLaMA-3B</td><td>0.9990</td><td>1.0000</td></tr><tr><td>Aggregate</td><td>0.9990</td><td>0.9995</td></tr><tr><td rowspan="4">Scale</td><td>Qwen-3B→Qwen-1.5B</td><td>1.0000</td><td>0.9986</td></tr><tr><td>Qwen-7B→Qwen-1.5B</td><td>0.9920</td><td>0.9984</td></tr><tr><td>LLaMA-3B→Qwen-1.5B</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Aggregate</td><td>0.9973</td><td>0.9990</td></tr></table>

Table 18: Substitution detection results on Math under different serving precisions. For each substitution pair, the provider serves the substituted model at FP16, INT8, or INT4 precision while claiming the target identity.
<table><tr><td></td><td colspan="3">FP16</td><td colspan="3">INT8</td><td colspan="3">INT4</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Q7→Q3</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>0.9985</td><td>1.0000</td><td>0.9950</td><td>0.9995</td></tr><tr><td>Q7→Q1.5</td><td>0.9960</td><td>0.9930</td><td>0.9974</td><td>1.0000</td><td>0.9950</td><td>0.9995</td><td>0.9920</td><td>0.9910</td><td>0.9943</td></tr><tr><td>Q3→Q1.5</td><td>1.0000</td><td>0.9950</td><td>0.9998</td><td>1.0000</td><td>0.9950</td><td>0.9985</td><td>0.9970</td><td>0.9935</td><td>0.9945</td></tr><tr><td>Q3→L3</td><td>1.0000</td><td>0.9950</td><td>0.9997</td><td>1.0000</td><td>0.9950</td><td>0.9995</td><td>0.9970</td><td>0.9935</td><td>0.9960</td></tr><tr><td>L3→Q3</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>0.9995</td><td>0.9960</td><td>0.9930</td><td>0.9983</td></tr><tr><td>Q7→L3</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>1.0000</td><td>0.9950</td><td>0.9995</td><td>0.9980</td><td>0.9940</td><td>0.9994</td></tr><tr><td>L3→Q1.5</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.9970</td><td>0.9935</td><td>0.9982</td><td>0.9860</td><td>0.9880</td><td>0.9975</td></tr></table>

Table 19: Ablation on loss components on the Code dataset. SFT uses only $L _ { \mathrm { s f t } }$ , +Sep adds L<sub>sep</sub>, +WM adds $L _ { \mathrm { w m } }$ , and +Both adds both losses.
<table><tr><td></td><td colspan="4">TPR@1%</td><td colspan="4">AUC</td></tr><tr><td>Setting</td><td>SFT</td><td>+Sep</td><td>+WM</td><td>+Both</td><td>SFT</td><td>+Sep</td><td>+WM</td><td>+Both</td></tr><tr><td>Q7→Q3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Q7→Q1.5</td><td>0.9300</td><td>0.9600</td><td>1.0000</td><td>1.0000</td><td>0.9860</td><td>0.9930</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Q3→Q1.5</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Q3→L3</td><td>1.0000</td><td>1.0000</td><td>0.9800</td><td>1.0000</td><td>1.0000</td><td>0.9999</td><td>0.9993</td><td>1.0000</td></tr><tr><td>L3→Q3</td><td>1.0000</td><td>1.0000</td><td>0.9700</td><td>1.0000</td><td>1.0000</td><td>0.9999</td><td>0.9993</td><td>1.0000</td></tr><tr><td>Q7→L3</td><td>1.0000</td><td>1.0000</td><td>0.9900</td><td>1.0000</td><td>0.9999</td><td>1.0000</td><td>0.9953</td><td>1.0000</td></tr><tr><td>L3→Q1.5</td><td>0.9700</td><td>0.9800</td><td>0.9600</td><td>1.0000</td><td>0.9990</td><td>0.9989</td><td>0.9992</td><td>1.0000</td></tr></table>

Table 20: Comparison of no-watermark (Plain), greenlist watermark (Greenlist), and our method (SUMMARK) on the Math dataset.
<table><tr><td></td><td colspan="3">TPR@1%</td><td colspan="3">AUC</td></tr><tr><td>Setting</td><td>Plain</td><td></td><td>Greenlist SUMMARK</td><td>Plain</td><td>Greenlist</td><td>SUMMARK</td></tr><tr><td>Q7→Q3</td><td>0.0100</td><td>0.0000</td><td>1.0000</td><td>0.1717</td><td>0.1180</td><td>1.0000</td></tr><tr><td>Q7→Q1.5</td><td>0.0000</td><td>0.0000</td><td>0.9960</td><td>0.1447</td><td>0.1080</td><td>0.9974</td></tr><tr><td>Q3→Q1.5</td><td>0.0600</td><td>0.0310</td><td>1.0000</td><td>0.5226</td><td>0.3560</td><td>0.9998</td></tr><tr><td>Q3→L3</td><td>0.0100</td><td>0.0000</td><td>1.0000</td><td>0.5052</td><td>0.1250</td><td>0.9997</td></tr><tr><td>L3→Q3</td><td>0.0400</td><td>0.0000</td><td>1.0000</td><td>0.5160</td><td>0.5170</td><td>1.0000</td></tr><tr><td>Q7→L3 L3→Q1.5</td><td>0.0200 0.0400</td><td>0.0000 0.0620</td><td>1.0000 1.0000</td><td>0.3352 0.4104</td><td>0.1740 0.5890</td><td>1.0000</td></tr></table>

## E.2 Robustness Analysis

Lexical watermark removal. Table 21 evaluates robustness on Code under two downstream perturbations: task-specific summarization fine-tuning and lexical watermark removal. The original SUMMARK detector achieves near-perfect performance across all substitution pairs. Composing a downstream summarization LoRA with the watermark adapter preserves strong detection in most settings, but reduces TPR@1% in harder cross-family cases, especially Q3→L3 and L3→Q3. This suggests that downstream adaptation can shift the summary token distribution and narrow the low-FPR detection margin. In contrast, the copy-paste watermark removal attack has little effect: TPR@1% remains between 0.997 and 1.000 with AUC ranging from 0.9973 to 0.9995. This indicates that SUMMARK ’s signal is distributed across summary-level token statistics rather than concentrated in easily removable local spans.

Watermark spoofing. Table 22 evaluates two text-only spoofing attacks where the adversary observes watermarked summaries from the claimed identity but has no access to the auditor’s secret key, carrier sets, or adapter weights. In distilled adapter spoofing, the adversary collects watermarked prompt– summary pairs from the claimed identity and trains a local spoof adapter on the substituted model to imitate the claimed identity’s summary distribution. In watermark-stealing spoofing, the adversary estimates token-level carrier statistics from observed watermarked summaries and biases decoding toward tokens that appear likely to belong to the claimed identity’s carrier set.

Distilled adapter spoofing is mostly contained: ASR@1% stays below 0.05 for four of seven transfer directions, with TPR@1% between 0.9570 and 0.9980. This suggests that spoofing requires approximating the claimed identity’s keyed statistical footprint, not merely its summary style. However, some directions are substantially harder, especially L3→Q3 (ASR@1% = 0.6290), Q7→L3 (0.2500), and L3→Q1.5 (0.2110).

Watermark-stealing spoofing is less effective: ASR@1% is zero in most settings, except Q7→Q3 (0.2160) and Q7→Q1.5 (0.0760). Overall, SUMMARK is robust to text-only watermark stealing and to distilled spoofing in most directions, but adaptive distilled spoofing remains a limitation in specific transfer directions.

Summary length sensitivity. Table 23 evaluates how detection changes when released summaries are truncated to different prefix lengths. Overall, SUMMARK remains effective even with short summaries: at 128 tokens, six of seven settings already achieve TPR@1% at least 0.925, and several settings remain perfect across all lengths. This suggests that the identity footprint is not dependent on unusually long summaries.

Table 21: Robustness on Code under downstream fine-tuning and lexical watermark removal. For fine-tuning, we compose a task-specific summarization LoRA with the watermark adapter at inference. For watermark removal, we apply a copy-paste attack that replaces contiguous spans in the summary.
<table><tr><td rowspan="2">Setting</td><td colspan="3">Original</td><td colspan="3">Model Finetuning</td><td colspan="3">Watermark Removal</td></tr><tr><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Q7→Q3</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.9630</td><td>0.9765</td><td>0.9950</td><td>1.0000</td><td>0.9950</td><td>0.9995</td></tr><tr><td>Q7→Q1.5</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.9770</td><td>0.9835</td><td>0.9978</td><td>0.9970</td><td>0.9935</td><td>0.9973</td></tr><tr><td>Q3→Q1.5</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.9780</td><td>0.9840</td><td>0.9975</td><td>1.0000</td><td>0.9950</td><td>0.9994</td></tr><tr><td>Q3→L3</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.4480</td><td>0.7190</td><td>0.9553</td><td>1.0000</td><td>0.9950</td><td>0.9994</td></tr><tr><td>L3→Q3</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.7160</td><td>0.8530</td><td>0.9832</td><td>1.0000</td><td>0.9950</td><td>0.9990</td></tr><tr><td>Q7→L3</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.9840</td><td>0.9870</td><td>0.9979</td><td>1.0000</td><td>0.9950</td><td>0.9995</td></tr><tr><td>L3→Q1.5</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>0.9640</td><td>0.9770</td><td>0.9956</td><td>0.9990</td><td>0.9945</td><td>0.9990</td></tr></table>

Table 22: Robustness under spoofing attacks on Math. We evaluate two text-only spoofing strategies: distilled adapter spoofing and watermark-stealing spoofing. For both attacks, lower ASR@1% and higher TPR@1%/AUC indicate stronger audit robustness.
<table><tr><td rowspan="2">Setting</td><td colspan="3">Distilled Adapter Spoofing</td><td colspan="3">Watermark Stealing Spoofing</td></tr><tr><td>TPR@1%</td><td>ASR@1%</td><td>AUC</td><td>TPR@1%</td><td>ASR@1%</td><td>AUC</td></tr><tr><td>Q7→Q3</td><td>0.9980</td><td>0.0020</td><td>0.9999</td><td>0.7840</td><td>0.2160</td><td>0.9930</td></tr><tr><td>Q7→Q1.5</td><td>0.9830</td><td>0.0170</td><td>0.9939</td><td>0.9240</td><td>0.0760</td><td>0.9960</td></tr><tr><td>Q3→Q1.5</td><td>0.9960</td><td>0.0040</td><td>0.9983</td><td>1.0000</td><td>0.0000</td><td>1.0000</td></tr><tr><td>Q3→L3</td><td>0.9570</td><td>0.0430</td><td>0.9938</td><td>1.0000</td><td>0.0000</td><td>1.0000</td></tr><tr><td>L3→Q3</td><td>0.3710</td><td>0.6290</td><td>0.9474</td><td>1.0000</td><td>0.0000</td><td>1.0000</td></tr><tr><td>Q7→L3</td><td>0.7500</td><td>0.2500</td><td>0.9706</td><td>1.0000</td><td>0.0000</td><td>1.0000</td></tr><tr><td>L3→Q1.5</td><td>0.7890</td><td>0.2110</td><td>0.9641</td><td>1.0000</td><td>0.0000</td><td>1.0000</td></tr></table>

The main exception is Q3→L3, where TPR@1% stays around 0.65–0.70 despite high AUC. This indicates that the two distributions remain separable in aggregate, but the margin is not large enough to support near-perfect detection at the strict 1% FPR threshold. Performance is also not strictly monotonic with length, since longer prefixes may introduce additional natural lexical variation rather than only more watermark evidence. Overall, the results show that summary length affects the low-FPR margin, but short summaries are sufficient for reliable detection in most substitution directions.

Summary post-processing. Table 24 reports per-pair detection results on Math under sentence shuffling, random deletion, word truncation, and bilingual rewriting. Across the three bounded surface transformations, the average TPR@1% over all seven substitution pairs and three transformations is 0.9992, with a minimum of 0.9950. Bilingual rewriting reduces the average TPR@1% to 0.7900 and the minimum to 0.5740. These results indicate that detection remains stable under the evaluated bounded edits, while more extensive rewriting can weaken the watermark signal.

Cross-domain generalization. We train each adapter on Math prompts only, freeze it together with the Math-calibrated thresholds, and evaluate the same seven substitution pairs on Code and General without retraining or recalibration (Table 25). Macro TPR remains above 0.998 on both target domains, with observed macro FPR of 0.0061 on Code and 0.0123 on General. Per-pair FPR ranges from 0.001 to 0.014 on Code and from 0.002 to 0.025 on General, so calibration drift is pair-dependent and some General directions exceed the nominal 1% target. These results cover two task-domain shifts only, not arbitrary shifts or an adaptive provider that alters the generation process.

Training data poisoning. Trusted onboarding assumes an uncorrupted adapter-training corpus. We stress-test this assumption on Qwen-7B→Qwen-3B by flipping the carrier identity label between the two models for a fraction of the 1,000 training examples. Prompts and teacher summaries are unchanged, so the $L _ { \mathrm { w m } }$ and $L _ { \mathrm { s e p } }$ targets are corrupted while $L _ { \mathrm { s f t } }$ is not (Table 26). Detection is nearly unaffected at 5% poisoning, degrades at 15%–25%, and TPR@1% drops to zero from 50%

Table 23: Length sensitivity of substitution detection on Code. We truncate each released summary to the first 128, 256, 384, or 512 tokens, and report TPR@FPR=1%, ACC@FPR=1%, and AUC. Setting A→B denotes serving B while claiming A.
<table><tr><td></td><td colspan="3">128 tokens</td><td colspan="3">256 tokens</td><td colspan="3">384 tokens</td><td colspan="3">512 tokens</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>Q7→Q3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Q7→Q1.5</td><td>0.9250</td><td>0.9625</td><td>0.9944</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9750</td><td>0.9875</td><td>0.9969</td><td>0.9750</td><td>0.9875</td><td>0.9994</td></tr><tr><td>Q3→Q1.5</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.7750</td><td>0.8875</td><td>0.9944</td><td>0.9750</td><td>0.9875</td><td>0.9981</td></tr><tr><td>Q3→L3</td><td>0.7000</td><td>0.8500</td><td>0.9825</td><td>0.7000</td><td>0.8500</td><td>0.9788</td><td>0.6750</td><td>0.8375</td><td>0.9856</td><td>0.6500</td><td>0.8250</td><td>0.9756</td></tr><tr><td>L3→Q3</td><td>0.9750</td><td>0.9875</td><td>0.9938</td><td>0.9750</td><td>0.9875</td><td>0.9925</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9750</td><td>0.9875</td><td>0.9994</td></tr><tr><td>Q7→L3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>L3→Q1.5</td><td>0.9750</td><td>0.9875</td><td>0.9963</td><td>0.9750</td><td>0.9875</td><td>0.9931</td><td>0.8500</td><td>0.9250</td><td>0.9644</td><td>0.9000</td><td>0.9500</td><td>0.9656</td></tr></table>

Table 24: Robustness of watermark detection under summary post-processing and rewriting on Math. We report three bounded surface transformations (sentence shuffle, random deletion, and word truncation), and one stronger bilingual rewriting stress test.
<table><tr><td></td><td colspan="2">Original</td><td colspan="2">Sentence Shuffle</td><td colspan="2">Random Deletion</td><td colspan="2">Truncate Words</td><td colspan="2">Bilingual Rewrite</td></tr><tr><td>Setting</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td><td>TPR@1%</td><td>AUC</td></tr><tr><td>Q7→Q3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9995</td><td>1.0000</td><td>0.9995</td><td>1.0000</td><td>0.9995</td><td>0.9760</td><td>0.9963</td></tr><tr><td>Q7→Q1.5</td><td>0.9960</td><td>0.9974</td><td>0.9960</td><td>0.9969</td><td>0.9950</td><td>0.9969</td><td>0.9950</td><td>0.9981</td><td>0.9010</td><td>0.9931</td></tr><tr><td>Q3→Q1.5</td><td>1.0000</td><td>0.9998</td><td>1.0000</td><td>0.9994</td><td>1.0000</td><td>0.9993</td><td>1.0000</td><td>0.9994</td><td>0.8220</td><td>0.9840</td></tr><tr><td>Q3→L3</td><td>1.0000</td><td>0.9997</td><td>1.0000</td><td>0.9993</td><td>1.0000</td><td>0.9992</td><td>1.0000</td><td>0.9994</td><td>0.6570</td><td>0.9652</td></tr><tr><td>L3→Q3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9990</td><td>1.0000</td><td>0.9990</td><td>1.0000</td><td>0.9990</td><td>0.6390</td><td>0.9646</td></tr><tr><td>Q7→L3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9995</td><td>1.0000</td><td>0.9995</td><td>1.0000</td><td>0.9995</td><td>0.9610</td><td>0.9961</td></tr><tr><td>L3→Q1.5</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9990</td><td>0.9990</td><td>0.9989</td><td>0.9990</td><td>0.9989</td><td>0.5740</td><td>0.9388</td></tr></table>

Table 25: Cross-domain transfer from Math to Code and General. Adapters and detection thresholds are trained and calibrated on Math and kept fixed; observed FPR is measured on target-domain data. Metrics are macro-averaged over seven substitution pairs.
<table><tr><td rowspan="2">Train→test</td><td colspan="4">SUMMARK</td></tr><tr><td>TPR</td><td>Observed FPR</td><td>Balanced ACC</td><td>AUC</td></tr><tr><td>Math→Math</td><td>0.9994</td><td>0.0100</td><td>0.9947</td><td>0.9996</td></tr><tr><td>Math→Code</td><td>0.9993</td><td>0.0061</td><td>0.9966</td><td>0.9991</td></tr><tr><td>Math→General</td><td>0.9989</td><td>0.0123</td><td>0.9933</td><td>0.9990</td></tr></table>

onward. At 50%, AUC is still 0.8767: some score ranking survives, but detection at the calibrated low-FPR threshold does not. This marks a training-integrity boundary rather than a defense against poisoning; authenticated data provenance lies outside the verifier studied here.

Table 26: Carrier-label poisoning on Qwen-7B→Qwen-3B (Math, 1,000 training examples per rate). Prompts and teacher summaries are unchanged; only the $L _ { \mathrm { w m } }$ and $L _ { \mathrm { s e p } }$ carrier targets are corrupted. At 100%, labels are fully swapped and the score ordering is inverted.
<table><tr><td rowspan="2">Poison rate</td><td colspan="3">SUMMARK</td></tr><tr><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td>0%</td><td>1.0000</td><td>0.9950</td><td>1.0000</td></tr><tr><td>5%</td><td>0.9900</td><td>0.9925</td><td>0.9998</td></tr><tr><td>15%</td><td>0.7950</td><td>0.8925</td><td>0.9761</td></tr><tr><td>25%</td><td>0.6900</td><td>0.8400</td><td>0.9602</td></tr><tr><td>50%</td><td>0.0000</td><td>0.4950</td><td>0.8767</td></tr><tr><td>100%</td><td>0.0000</td><td>0.4950</td><td>0.0032</td></tr></table>

## E.3 Ablation Studies

Ablation 1: Do we need identity-specific watermarking? Table 20 compares three variants on Math:Plain (no watermark), a Greenlist watermark that uses a single shared greenlist across identities, and SUMMARK. Under a strict auditing regime (TPR at FPR= 1%), both Plain and Greenlist fail across all substitution settings, even when AUC appears moderate. This highlights a core requirement of model identity auditing: a global greenlist can create a detectable bias in aggregate, but it does not provide separable, identity-specific evidence needed to reliably distinguish a claimed identity from closely related alternatives at low false-positive rates.

Why Greenlistfails at low FPR. Math summaries often have constrained lexical choices driven by the solution structure, leaving limited room for a weak global token bias to accumulate. As a result, non-watermarked summaries can match a shared greenlist by chance, and the separation becomes insufficient when the threshold is set to control false alarms. By contrast, SUMMARK explicitly allocates identity-specific carrier sets and trains adapters to concentrate additional probability mass on the correct bucket, which yields a predictable shift in summary token statistics that remains separable across identities.

Ablation 2: Training objective components. We next isolate the role of the detectability and separability components in our training objective by ablating $L _ { \mathrm { w m } }$ (green-mass regularization) and $L _ { \mathrm { s e p } }$ (margin-based separation across identities). As shown in Table 19 and Figure 4 when varying weights, removing either term can reduce TPR at $\mathrm { F P R } = 1 \%$ , even when AUC remains high. This reflects the difference between overall separability and auditing at strict operating points. Concretely, $L _ { \mathrm { w m } }$ supplies the primary watermark signal by increasing usage of the correct identity bucket, while $L _ { \mathrm { s e p } }$ suppresses the most confusable competing identity and stabilizes decisions across substitution pairs. Combining both terms achieves consistently strong low-FPR detection, matching the design goal of robust identity attribution under closely related substitutions.

Ablation 3: Effect of precision-consistency regularizer. Table 28 shows that the precisionconsistency regularizer $L _ { \mathrm { q s } }$ is essential for detecting precision variants. Without $L _ { \mathrm { q s } }$ , the detector is almost random: TPR@1% is near zero, ACC@1% stays around 0.50, and AUC is close to chance across all models. After adding $L _ { \mathrm { q s } } ,$ detection improves sharply for most identities. LLaMA-3B, Qwen-3B, and Qwen-7B all achieve near-perfect TPR@1% and AUC for both FP16→INT8 and FP16→INT4.

The main exception is Qwen-1.5B, where INT8 remains difficult and INT4 is only partially detectable. Still, $L _ { \mathrm { q s } }$ substantially improves its AUC, especially for FP16→INT4. Overall, these results show that standard identity watermarking alone does not naturally separate FP16 and quantized variants; the quantization-separation loss is what creates a precision-sensitive audit signal.

Ablation 4: Adapter-matched text and watermarking baselines. The shared-Greenlist and generic watermark baselines above do not control for the use of an identity-specific adapter. We therefore add adapter-matched baselines on Math, using the same seven substitution pairs and 1,000 training examples per adapter. $T 5 + S F T$ adapter trains an SFT-only summary adapter on clean, unwatermarked reference summaries; T5 is then trained and evaluated on its outputs alone, without a key or carrier information. Keyed Unigram/Greenlist adapters assign each identity its own secret key and train an adapter to reproduce teacher summaries watermarked with that key. Both controls use an adapter per identity, and the keyed variants a separate key per identity, but neither uses SUMMARK’s controlled-overlap carriers or hardest-negative separability objective.

As shown in Table 27, the strongest control, T5 + SFT adapter, reaches a mean TPR@1% of 0.8937 but drops to 0.6550 on its worst pair. The keyed watermark adapters perform far worse, while SUMMARK reaches 0.9980. Under these implementations and budget, the gain is therefore not explained by adapter-based generation or per-identity keys alone, although this does not rule out multi-key watermark attribution in general. This control differs from the SFT-only loss ablation in Table 19, which removes SUMMARK’s own training terms.

## E.4 Case Study

To complement the quantitative ablations and robustness tests above, we provide a qualitative visualization of what the identity watermark looks like in a single summary. Figure 3 highlights tokens from a fixed bucket to show how the correct adapter produces a consistently denser, more uniformly distributed footprint, while a substituted identity does not. This case study visualizes identity-specific watermarking at the token level under three settings: no watermark, Qwen-3B watermark, and Qwen-7B watermark. For visualization, all texts are colored using the same vocabulary bucket $G _ { \mathrm { Q w e n } - 3 \mathrm { B } }$ , where tokens in this set are marked green. Without watermarking, green tokens appear only occasionally, consistent with natural generation. With the Qwen-3B watermark enabled, green tokens become noticeably more frequent and more evenly spread throughout the summary, indicating a stable, detectable bias. We color Qwen-7B summaries by $G _ { \mathrm { Q w e n } - 3 \mathrm { B } }$ to simulate a substitution case (serving Qwen-7B while claiming Qwen-3B); the resulting summary does not exhibit the same green-token pattern, illustrating that watermark footprints are separable across model identities.

Table 27: Adapter-matched baselines on Math (1,000 training examples per identity). Mean and minimum are taken over seven substitution pairs. T5 + SFT adapter uses unwatermarked summaries; keyed Unigram/Greenlist adapters use a separate key and adapter per identity. Best is bold.  
![](images/ff0da885e5e28e0eb8bebd338fa995d8656fabe2ea303abded51973a365bc29c.jpg)  
Figure 3: Visualizing identity footprints: tokens in $G _ { \mathrm { Q w e n } - 3 \mathrm { B } }$ are highlighted. The Qwen-3B adapter yields a denser, more uniform green-token pattern than no-watermark, while a substituted identity (Qwen-7B) does not reproduce this footprint when colored by $G _ { \mathrm { Q w e n } - 3 \mathrm { B } }$

## E.5 Hyperparameter Analysis

Substitution ratio. To ensure fair comparison, we fix the total number of test samples to 1,000 for each substitution ratio across all three datasets. As shown in Tables 29, and 30, detection performance consistently improves as the substitution ratio increases on Math and General. Even at low substitution ratios, TPR@FPR=1% is close to 1.0 for most substitution settings, and AUC remains around 0.99. At a substitution ratio of 0.5, performance is effectively saturated, with near-perfect TPR and consistently high AUC across all model pairs. We therefore adopt a ratio of 0.5 as the default setting in our experiments.

Loss design. We study the sensitivity of our method to the weights of the watermark loss $\lambda _ { \mathrm { w m } }$ and the separability loss $\lambda _ { \mathrm { s e p } }$ across three types of substitution. We report TPR@FPR = 1%, with the ratio defined as $\lambda _ { \mathrm { { s e p } } } / \lambda _ { \mathrm { { w m } } }$ . Figure 4 shows TPR@1% for one Code example from each substitution type, and Figure 6 reports AUC for the same pairs. Figures 8 and 9 report paired AUC and TPR@1% results for four representative General pairs, covering changes in size, family, and both, together with an additional parameter-sensitive case. For completeness, we report the remaining TPR@1% results on the Code dataset in Figure 7.

Table 28: Effect of the quantization-separation loss $L _ { \mathrm { q s } }$ on within-identity precision-consistency detection. We compare watermark adapters trained with and without $L _ { \mathrm { q s } }$ when the claimed configuration is FP16 and the served configuration is INT8 or INT4.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="3">w/o  $L _ { \mathrm { q s } }$ </td><td colspan="3">w/  $L _ { \mathrm { q s } }$ </td></tr><tr><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td><td>TPR@1%</td><td>ACC@1%</td><td>AUC</td></tr><tr><td rowspan="2">LLaMA-3B</td><td>FP16→INT8</td><td>0.0150</td><td>0.5040</td><td>0.5034</td><td>0.9990</td><td>0.9960</td><td>0.9990</td></tr><tr><td>FP16→INT4</td><td>0.0000</td><td>0.5000</td><td>0.4256</td><td>0.9990</td><td>0.9995</td><td>0.9990</td></tr><tr><td rowspan="2">Qwen-1.5B</td><td>FP16→INT8</td><td>0.0090</td><td>0.5000</td><td>0.4952</td><td>0.1132</td><td>0.5521</td><td>0.7575</td></tr><tr><td>FP16→INT4</td><td>0.0181</td><td>0.5050</td><td>0.4875</td><td>0.3417</td><td>0.6663</td><td>0.9209</td></tr><tr><td rowspan="2">Qwen-3B</td><td>FP16→INT8</td><td>0.0020</td><td>0.5000</td><td>0.5298</td><td>0.9860</td><td>0.9890</td><td>0.9973</td></tr><tr><td>FP16→INT4</td><td>0.0000</td><td>0.5000</td><td>0.3719</td><td>0.9960</td><td>0.9945</td><td>0.9986</td></tr><tr><td rowspan="2">Qwen-7B</td><td>FP16→INT8</td><td>0.0140</td><td>0.5030</td><td>0.4868</td><td>0.9910</td><td>0.9920</td><td>0.9982</td></tr><tr><td>FP16→INT4</td><td>0.0140</td><td>0.5045</td><td>0.5611</td><td>0.9990</td><td>0.9975</td><td>0.9998</td></tr></table>

Table 29: Sensitivity to Substitution Ratio on Math. We report TPR@FPR=1%, ACC@FPR=1%, and AUC. Q7/Q3/Q1.5 = Qwen-7B/3B/1.5B, L3 = LLaMA-3B. Best is bold.
<table><tr><td rowspan="2"></td><td colspan="2">5%</td><td colspan="2">10%</td><td colspan="2"></td><td colspan="2">15%</td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2"></td><td colspan="2">65%</td></tr><tr><td>Setting TPR</td><td>ACC</td><td>AUC</td><td>TPR ACC</td><td>AUC</td><td>TPR</td><td>ACC</td><td>AUC</td><td>TPR</td><td>ACC</td><td>AUC</td><td>TPR</td><td>ACC</td><td>AUC TPR</td><td>ACC</td><td>AUC</td></tr><tr><td>Q7→Q3</td><td>1.0000 0.9910</td><td>0.9995</td><td>1.0000</td><td>0.9910</td><td>0.9994</td><td>1.0000</td><td>0.9920 0.9994</td><td>1.0000</td><td>0.9930</td><td>0.9993</td><td>1.0000</td><td>0.9950</td><td>0.9990</td><td>1.0000</td><td>0.9970</td><td>0.9986</td></tr><tr><td>Q7→Q1.5</td><td></td><td>1.0000 0.9910 0.9995</td><td></td><td></td><td></td><td>1.0000 0.9910 0.9994 0.9933 0.9910 0.9931 0.9920 0.9910</td><td></td><td></td><td></td><td>0.9955</td><td>0.9940</td><td></td><td></td><td>0.9920 0.9953 0.9969 0.9950 0.9957</td><td></td><td></td></tr><tr><td>Q3→Q1.5</td><td></td><td>1.0000 0.9910 0.9994 1.0000 0.9910 0.9989 1.0000 0.9920 0.9990 1.0000 0.9930 0.9985</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.0000 0.9950 0.99891.0000 0.9970 0.9985</td><td></td><td></td></tr><tr><td>Q3→L3</td><td>1.0000 0.9910 0.9994 1.0000 0.9910 0.9992 1.0000 0.9920 0.9993 1.0000 0.9930 0.9992 1.0000 0.9950 0.9988 1.0000 0.9970 0.9940</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>L3→Q3</td><td>1.0000 0.9910 0.9990 1.0000 0.9910 0.9989 1.0000 0.9920 0.9988 1.0000 0.9930 0.9987 1.0000 0.9950 0.9990 1.0000 0.9970 0.9986</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Q7→L3</td><td>1.0000 0.9910 0.9995 1.0000 0.9910 0.9995 1.0000 0.9920 0.9994 1.0000 0.9930 0.9993 1.0000 0.9950 0.9990 1.0000 0.9970 0.9986</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>L3→Q1.5 1.0000 0.9910 0.9989 1.0000 0.9910 0.9989 1.0000 0.9920 0.9988 1.0000 0.9930 0.9993 1.0000 0.9950 0.9990 1.0000 0.9970 0.9986</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

First, comparing watermark strengths of 0.2 and 0.5, we observe that a larger $\lambda _ { \mathrm { w m } }$ can improve detection, although the benefit depends on the model pair and loss ratio. Second, many substitution settings achieve high performance at larger loss ratios. These trends support the usefulness of balancing the watermark and separability terms, but the response is not uniformly monotonic. In particular, General Qwen-7B→Qwen-1.5B remains sensitive, with TPR@1% of 0.57 at $\lambda _ { \mathrm { w m } } =$ $\lambda _ { \mathrm { s e p } } = 0 . 5$

Watermark design. We study the sensitivity of substitution detection performance to the green ratio $\rho$ and the overlap τ. In Figure 5, the first row fixes $\rho = 0 . 3$ and varies τ, while the second row fixes τ= 0.3 and varies $\rho .$ . Most tested configurations maintain high detection, although particular model pairs are sensitive: LLaMA-3B→Qwen-1.5B reaches TPR@1% of 0.11 at $\rho = 0 . 3 5$ , and LLaMA-3B→Qwen-3B reaches 0.816 at $\tau = 0 . 7$ in Figure 11 and Figure 10. More importantly, as discussed in Equation 6, the feasible combinations of $\rho$ and τ are constrained by model identity capacity. Choosing overly small overlap limits the number of supported identities, while a small green ratio may restrict the effective vocabulary during summary generation, creating a potential trade-off with summary quality. We therefore use $\rho = \tau = 0 . 3$ as a practical default motivated by these design considerations.

Table 30: Sensitivity to Substitution Ratio on General. We report TPR@FPR=1%, ACC@FPR=1%, and AUC. Q7/Q3/Q1.5 = Qwen-7B/3B/1.5B, L3 = LLaMA-3B. Best is bold.
<table><tr><td rowspan="2">Setting</td><td colspan="2">5%</td><td colspan="2"></td><td colspan="2">10%</td><td colspan="2">15%</td><td colspan="2">25%</td><td colspan="2"></td><td colspan="2">50%</td><td colspan="2">65%</td></tr><tr><td>TPR</td><td>ACC</td><td>AUC TPR</td><td>ACC</td><td>AUC</td><td>TPR</td><td>ACC</td><td>AUC</td><td>TPR</td><td>ACC</td><td>AUC</td><td>TPR ACC</td><td>AUC</td><td>TPR</td><td>ACC</td><td>AUC</td></tr><tr><td>Q7→Q3</td><td>1.0000</td><td>0.9910 0.9999</td><td>1.0000</td><td>0.9910</td><td>0.9999</td><td>1.0000</td><td>0.9920</td><td>0.9999</td><td>1.0000 0.9930</td><td>0.9999</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>1.0000</td><td>0.9970 0.9999</td><td></td></tr><tr><td>Q7→Q1.5</td><td>0.8200 0.9820</td><td>0.9958</td><td>0.8400</td><td></td><td>0.9750 0.9931</td><td>0.8800 0.9740</td><td>0.9951</td><td>0.9840</td><td>0.9890</td><td>0.9965</td><td>0.9860</td><td>0.9880</td><td>0.9971</td><td>0.9877</td><td>0.9890 0.9969</td><td></td></tr><tr><td>Q3→Q1.5</td><td></td><td>1.0000 0.9910 1.0000</td><td>1.0000</td><td></td><td></td><td>0.9910 1.0000 1.0000 0.9920 1.0000</td><td></td><td></td><td>1.0000 0.9930</td><td>1.0000</td><td>0.9980</td><td>0.9940</td><td></td><td>1.0000 0.9985</td><td>0.9960 1.0000</td><td></td></tr><tr><td>Q3→L3</td><td></td><td>1.0000 0.9910 1.0000 1.0000 0.9910 0.9999 1.0000 0.9920 0.9999 1.0000 0.9930 0.9999</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.0000 0.9950 0.9999</td><td></td><td>1.0000 0.9970 0.9999</td><td></td><td></td></tr><tr><td>L3→Q3</td><td>1.0000 0.9910 1.0000 1.0000 0.9910 1.0000 1.0000 0.9920 1.0000 1.0000 0.9930 1.0000</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.0000 0.9950 1.0000 1.0000 0.9970 1.0000</td><td></td><td></td><td></td><td></td></tr><tr><td>Q7→L3</td><td>1.0000 0.9910 0.9996 1.0000 0.9910 0.9996 1.0000 0.9920 0.9996 0.9880 0.9900 0.9995 0.9840 0.9870 0.9990 0.9862 0.9880 0.9986</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>L3→Q1.5</td><td>1.0000 0.9910 1.0000 0.9700 0.9880 0.9990 0.9733 0.9880 0.9990 0.9640 0.9840 0.9987 0.9580 0.9740 0.9990 0.9600 0.9710 0.9990</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/1f84ef7a80f1a138c6f4f122705f11d1640a10aabe4eea3045cb0c85e8ce7b3d.jpg)

![](images/4b9c9d3c9a62eb6bd29de08c27f5ee8f3f3845dc3344e777870493ad9f464058.jpg)

![](images/544d4153647031ab29e85daaeba316839147a474b774fc0c710a7c3ec226a727.jpg)  
Figure 4: Effect of training objective components $\lambda _ { \mathrm { s e p } }$ and $\lambda _ { \mathrm { w m } }$ on substitution performance on Code dataset. Ratio is defined as $\lambda _ { \mathrm { { s e p } } } / \lambda _ { \mathrm { { w m } } }$ . We report TPR@FPR = 1% as the score.

![](images/8ec494351006d9819319fa37dc77cd95832bcf41d503d137bc9c802261b57a02.jpg)

![](images/ebcbdb6c46a11e2cbd797bfdc36db0cddada6f07849ee20205e33aff7a8e3556.jpg)

![](images/773cbcef19e8e642bc7aca3504d8ae6128bf3eb6b07f5b5ff2f6240307dc88ca.jpg)

![](images/f6415552aae03d34c89ce662493bf99c367fdf59eab57f67d060263ca49ef64b.jpg)

![](images/0bf9eab46b05805f5a33ef3bfc779d7e41251eff826fd3630d82047658b2dbd4.jpg)

![](images/0e9b208dd369af705b83d8ab816cb46573130d6d11bac34324496d2effc6f9a4.jpg)  
Figure 5: Effect of green ratio ρ and overlap τ on substitution performance on Math dataset.

![](images/0fc4e51e4b03c3faaa6bab42196e0151a84beb9b0ebfb82f1fda590dccb1b142.jpg)

![](images/17c4b8cd48cafa183591b10fd8bab732a51279e05aff8416424beb9f6aed8137.jpg)

![](images/6ca232f96f3c144c1b01aa6af04357e0c686a605feef74510881a4fcb524fbb5.jpg)  
Figure 6: Effect of training objective components $\lambda _ { \mathrm { s e p } }$ and $\lambda _ { \mathrm { w m } }$ on substitution performance on Code dataset. Ratio is defined as $\lambda _ { \mathrm { s e p } } / \lambda _ { \mathrm { w m } }$ . We report AUC as the score.

![](images/8b22840aecaa58692e56f502296916b9204737b76a866e435ff08ff3c09114d2.jpg)

![](images/cc05e7bba7eaa05ac47d99a34aa43c6d99c6aad4be4c14a79c916d32d82ffc50.jpg)

![](images/7f1f7a06ad3efa80addccf58c735f578e0813926c26d8955447f6806368da212.jpg)  
Figure 7: Effect of training objective components $\lambda _ { \mathrm { s e p } }$ and $\lambda _ { \mathrm { w m } }$ on substitution performance on Code dataset. Ratio is defined as $\lambda _ { \mathrm { { s e p } } } / \lambda _ { \mathrm { { w m } } }$ . We report TPR@FPR = 1% as the score.

![](images/3875d6225cba6e877efcc431718d01dbbba779c14447a6e7d6fd38837bebc962.jpg)

![](images/1f3a1c41bf5c1fce39eca2a081a26515e199a5fa1616ada9d0fe30073a3c19f1.jpg)

![](images/edf57eff0228247b9dcc671fc1ee252b3bba3a03555c5fce0cb57910f81a50a1.jpg)

![](images/20d45f082af40c5c0dc1a660083bbffa6d934751fbf85b236ddc0089dfa258e4.jpg)  
Figure 8: Effect of training objective components $\lambda _ { \mathrm { s e p } }$ and $\lambda _ { \mathrm { w m } }$ on substitution performance on General dataset. Ratio is defined as $\lambda _ { \mathrm { s e p } } / \lambda _ { \mathrm { w m } }$ . We report AUC the score.

![](images/0e90473ce4d4d46c3c7cadde1b71543343df6cc27f67537dea7b339baa1037f9.jpg)

![](images/34d59ff8d6c782a0aa37514a1e20bcf4f87095783796174cf8d574326b7c2acf.jpg)

![](images/c1382d1f4a06846c640987f569ad1cab5eca09dec9b67a36d2e84642b75db1dc.jpg)

![](images/d2e4f43bc25c1204c06ab32a927c6e48f05bd6b660b8a95c529a0b39d0f2c70b.jpg)  
Figure 9: Effect of training objective components on substitution performance on General dataset. We report the TPR@FPR = 1% as the score.

![](images/9c0bf2bc7e105e3f9d55e30490f364be23e91470906b8108020c9c564c41159d.jpg)

![](images/a725f5a08dfd92d24c41d10da5124cdd4c764f7f77f10bf68fb20277b30ed2d9.jpg)

![](images/85b78732f49e487989174ea7c593c51433720c8373f64d1f31438cfc5b9cf83a.jpg)  
Figure 10: Effect of green ratio ρ on substitution performance on Math dataset.

![](images/3e477146c8f5eb7c836d19f9818eb4b386022b3a3663104251564e92ba03d38c.jpg)

![](images/ad510b3d7168a5df20d1b205f788a77dac7c36b8504e3ae1d427bc32e67a4969.jpg)

![](images/aaf828c37f3a60096a4e6efdd409adb3bde000cec249a268388a8270320b4866.jpg)  
Figure 11: Effect of overlap τ on substitution performance on Math dataset.