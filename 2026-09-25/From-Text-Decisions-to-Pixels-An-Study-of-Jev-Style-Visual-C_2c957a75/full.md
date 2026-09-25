# From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model

Xunlan Zhou <sup>1</sup> <sup>2</sup> Xianliang Yang <sup>2</sup> Li Zhao <sup>2</sup>

## Abstract

Visual software often needs a decision over supplied alternatives rather than a generated explanation. We present PixelJev, a native-image decision interface that maps an image, a task instruction, and a runtime candidate set to a structured choice and candidate-conditioned probabilities using small open multimodal models. Its initial realization unifies recognition and multiplechoice visual question answering through an existing language-model readout, with separately evaluated options for frozen inference, languageside adaptation, and held-out calibration. Across seven benchmark evaluations, 64-shot source adaptation raises Pets accuracy from 60.13% to 92.40 ± 0.26% across optimization seeds and transfers to natural resampling, new texture labels, and A-OKVQA without target fitting, while frozen inference already supports both VQA tasks. A matched prompt-only follow-up on Pets and ScienceQA attributes the large Pets gain to adaptation and identifies a narrower output-validity benefit of candidate readout in adapted VQA. Specialist DINOv2 probes remain stronger on source recognition, frozen 4B is stronger than adapted 2B on DTD and ScienceQA, and accuracy gains do not ensure calibrated target probabilities. These findings establish a working starting point for general-purpose visual decision models and identify the remaining requirements: schema robustness, cross-family transfer, and reliable use of visual evidence.

## 1. Introduction

A visual model deployed inside software is often asked to decide rather than describe. The caller may need a category, an answer from a changing option list, or a choice among proposed objects. A fixed classifier exposes one learned label vocabulary; a conversational vision-language model (VLM) exposes generated text. Between these interfaces lies a useful abstraction: the caller provides the decision space at runtime, and the model returns a valid identifier with scores conditioned on that space.

We call this abstraction a visual decision model and introduce PixelJev, an implementation with native image input and small open multimodal backbones. Its interface is image + instruction + candidate schema → choice + candidateconditioned probabilities. It follows the program-facing motivation of Jev (TypeSafe AI, 2026) and open decision interfaces such as SemIf (TheoLeeCJ, 2026). The image enters the backbone’s vision branch directly, not through a captioning intermediary.

The long-term objective is general-purpose visual decisions: one interface and a reusable model that serve changing tasks and candidate semantics without a newly fitted output head for every label space. We distinguish three requirements: interface reuse, competence on held-out task families, and reliability under changes to the visual evidence and decision schema. This paper establishes interface reuse across recognition and question answering and measures early transfer; the broader competence and reliability requirements remain open. “Towards” in the title marks that distinction.

The mechanism is deliberately simple. PixelJev uses the existing Qwen3.5 vision-language model (Qwen Team, 2026) and reads probabilities from candidate tokens at the first assistant position. Frozen inference, language-side LoRA, and scalar calibration are separate choices rather than prerequisites bundled into a claimed new architecture. This separation lets us ask what adaptation actually buys over a working frozen interface, a larger frozen backbone, and specialist visual encoders.

Related work already explores native visual decisions. In particular, Visual Jev (Yu & Yao, 2026) combines languagemodel-head readout, answer-supervised adaptation, and shared-context execution. PixelJev addresses a complementary empirical question: how far does classification-only, few-shot adaptation of a small model transfer, and when is frozen scale or a specialist representation a better choice? We do not claim priority for the general idea of visual deci-

![](images/920936bc03229075e777b06cf5b96e4685ab8689caa656312a056c927f856b5c.jpg)  
Figure 1. One interface does not imply one best model for every task. All points use full test sets. PixelJev’s LoRA variant uses 64 fitting images per source class and reports the mean of three optimization seeds; whiskers show one sample SD, often narrower than the marker. Probes use 64-shot source supervision; their heads transfer to CIFAR-10.1, but no texture head is fitted for DTD. Missing heads are N/A, not zero. Exact values for all five datasets appear in Table 1.

## sion interfaces.

Our contributions are threefold. First, we specify and implement a reusable decision contract with native pixels, runtime candidate descriptions, stable return identifiers, and explicit probability semantics. Second, we evaluate its frozen and adapted realizations across source recognition, natural resampling, new texture labels, and direct VQA, with largermodel, specialist, and matched prompt-only alternatives. Third, we map the limits of reuse: adaptation helps selected transfers but does not universally replace scale, and improved recognition does not establish reliable confidence.

## 2. Related Work

Program-facing and visual decisions. Jev exposes typed decisions to programs (TypeSafe AI, 2026), while SemIf implements open-model candidate-logit readout (TheoLeeCJ, 2026). Early studies examine memory control (Jiang et al., 2026), scientific choices (Deng et al., 2026), probabilistic annotation (Rafe & Das, 2026), and adapted scam-screening decisions (Ren et al., 2026). Visual Jev (Yu & Yao, 2026) is directly related: it uses a Qwen3-VL backbone, ordinary answer supervision through the language-model head, and batched question suffixes sharing an image/context prefix. It separates adaptation from serving efficiency and tests typed heads as controls. Our study instead uses Qwen3.5, candidate-conditioned fitting on recognition labels, specialist probes, and classification-to-VQA transfer. These differences define scope, not measured superiority over Visual Jev; we have not run its system under a matched protocol.

Representations and readouts. CLIP aligns image and text representations for zero-shot recognition (Radford et al., 2021); DINOv2 supplies strong image features (Oquab et al., 2024). LoRA enables parameter-efficient adaptation (Hu et al., 2022). PixelJev assembles established components around a programmatic contract. Its candidate scoring is closely related to ordinary multiple-choice VLM inference; a distinct interface does not establish a distinct learning algorithm or a new visual representation.

Transfer and probability quality. Temperature scaling estimates confidence adjustments from held-out labels (Guo et al., 2017). CIFAR-10.1 tests natural resampling (Recht et al., 2018), DTD changes texture semantics (Cimpoi et al., 2014), and A-OKVQA (Schwenk et al., 2022) and ScienceQA (Lu et al., 2022) change questions and answer choices per example. We use these settings to separate interface reuse, source adaptation, and target probability quality. None establishes exclusion from unknown foundationmodel pretraining.

## 3. PixelJev: A Reusable Visual Decision Interface

## 3.1. The Decision Contract

A request comprises an RGB image $x ,$ an instruction or question $q ,$ and an ordered candidate schema $C =$ $( ( u _ { 1 } , c _ { 1 } ) , \dots , ( u _ { K } , c _ { K } ) )$ . Here $u _ { i }$ is a stable programmatic identifier and $c _ { i }$ is its textual description. A decision model

implements

$$
F _ { \theta } ( x , q , C ) = ( u _ { \hat { \imath } } , { \bf p } , m ) , \qquad \hat { \imath } = \arg \operatorname* { m a x } _ { i } p _ { i } ,\tag{1}
$$

where $\mathbf { p }$ is a distribution over the supplied candidates and m records model, schema, and calibration provenance. Identifiers are returned by the program, not generated as free text.

The same contract covers different workloads without equating their difficulty. Recognition keeps $q$ and the class descriptions fixed across images. Multiple-choice VQA supplies a new $q$ and new descriptions for each question. Newclass evaluation supplies descriptions outside adapter supervision without fitting a new classifier head. Object and action candidate selection are possible future uses of the contract, but are not evaluated capabilities of this study.

The implementation validates the schema before scoring. Legal output membership follows from the programmatic readout; correctness must be measured separately. Likewise, p is conditional on $C ,$ not a probability that the selected answer is correct among all conceivable alternatives. There is no learned reject option in the current system.

## 3.2. An Initial Realization with Open Multimodal Models

The complete Qwen3.5 image processor and vision branch consume x. The user message contains q and the ordered descriptions; filenames, sample identifiers, correct labels, and explanatory answers are excluded. There is no captioning intermediary. Each candidate maps to a case-sensitive alphabetic slot $s _ { i }$ . The tokenizer checks single-token reversibility, distinct token IDs, and preservation of the completed chat-template boundary. The current slot inventory supports $2 \leq K \leq 5 2 ;$ ; this is an implementation limit, not a definition of visual decision modeling.

With extended thinking disabled, let $z$ be vocabulary logits at the first assistant position. PixelJev reads

$$
p _ { i } ( T \mid x , q , C ) = \frac { \exp ( z _ { s _ { i } } / T ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { s _ { j } } / T ) } .\tag{2}
$$

Raw inference sets $T = 1$ . The total vocabulary probability mass on candidate tokens is logged separately from this renormalized distribution. No response string or JSON document is generated for subsequent parsing.

Relationship to ordinary VLM inference. Equation 2 reuses an existing multiple-choice readout, rather than introducing a new architecture. PixelJev makes its input/output contract and candidate-conditional semantics explicit and evaluates how a reusable model behaves across workloads. Direct logits match equivalent constrained, deterministic one-token generation on the diagnostic inputs. We therefore do not attribute an intrinsic speedup to omitting that token or claim a task-accuracy advantage over all conventional VLM prompting. The full-set prompt-only follow-up crosses this readout with the same checkpoints under a common format cue and separately measures warm inference time (Table 2; Appendix F).

## 3.3. Three Separately Evaluated Operating Regimes

Frozen inference. The complete pretrained checkpoint and the validated slot mapping are sufficient to instantiate Equation 1. No benchmark fitting labels or extra decision head are required. Both 2B and 4B backbones use the same principal interface and pixel budget.

Few-shot adaptation. We optionally minimize candidateconditioned cross-entropy,

$$
\mathcal { L } ( x , y , q , C ) = - \log p _ { y } ( 1 \mid x , q , C ) ,\tag{3}
$$

randomly permuting candidates and remapping the correct slot during fitting. One LoRA adapter is shared across the three source recognition tasks. Only explicit languageattention projection allowlists are updated; the original vision, language, embedding, and output parameters remain frozen. We use rank $^ { 1 6 , }$ scaling $^ { 3 2 , }$ dropout 0.05, and three AdamW epochs at learning rate $1 0 ^ { - 4 } .$ . Pets development accuracy selects the checkpoint, with raw NLL breaking ties, and that checkpoint is reused across all evaluation tasks. Appendix A records the full supervision ledger.

Held-out calibration. A positive scalar temperature is fitted by minimizing NLL on a dedicated calibration split. This consumes labels even though it does not update the backbone or change the argmax. Raw and calibrated evaluations remain separate. Only the source CIFAR-10 temperature transfers to CIFAR-10.1; no DTD or VQA temperature is fitted. Frozen inference, adapter fitting, and calibration are thus distinct costs and scientific interventions, not interchangeable meanings of “training.”

## 3.4. Interface Diagnostics

Candidate reversal, an alternate instruction, and actual different-class image substitution provide finite interface diagnostics. Image-substitution controls and one-token readout comparisons are reported in Appendix D. They test finite interface behavior, not arbitrary schema invariance or visual evidence dependence on every task.

## 4. Evaluating Reuse, Adaptation, and Transfer

The evaluation follows the intended progression of a reusable decision model: instantiate the interface without fitting, adapt on a restricted source task family, and test reuse beyond those labels. Source recognition, natural shift, new texture semantics, and per-question VQA choices answer different questions; we do not collapse them into one “generality” score.

## 4.1. Source Tasks and Supervision

We use CIFAR-10 object recognition (Krizhevsky, 2009), all 37 Oxford-IIIT Pets breeds (Parkhi et al., 2012), and EuroSAT land-cover classification (Helber et al., 2019). We retain each pinned mirror’s full official test partition. EuroSAT uses a fixed mirrored partition, without a geographicalgeneralization claim.

Exact identity is computed from decoded RGB pixels and image dimensions. A fixed perceptual-hash policy groups near-similar images across complete training and test pools before selecting fitting examples. Training groups touching test data are excluded from adaptation, and fitting, development, and calibration groups are disjoint. All test images remain in the denominator; correlated images are handled through grouped uncertainty rather than removal. This policy does not prove that all leakage or pretraining overlap is absent.

The two budgets contain 16 or 64 fitting images per class, totaling 912 or 3,648 images across tasks. The smaller set is a strict subset of the larger. There are also 684 development and 684 calibration images, unchanged across budgets. Thus “64-shot” counts fitting labels, not all labels used by the study. The 64-shot condition is repeated with optimization seeds 7, 8, and 9 on the same split; none of the reported models is a seed ensemble.

## 4.2. Alternatives to Adapting the Interface

We evaluate full Qwen3.5-2B and Qwen3.5-4B checkpoints under the same principal user-message interface and nativeimage pixel budget (65,536 minimum; 262,144 maximum). We also evaluate frozen CLIP ViT-B/16 zero-shot scores and logistic-regression heads on frozen, normalized CLIP and DINOv2 ViT-B features. Each probe uses the same per-task fitting images, with regularization selected on development data. Each backbone retains its native processor.

The shared Qwen adapter sees labels from all three tasks; each specialist head sees labels from its own task only. Pertask fitting shots match, but total supervision, trainable parameter count, and compute do not. Raw frozen Qwen and zero-shot CLIP consume no benchmark fitting labels; temperature-scaled variants additionally consume calibration labels. Our comparisons test practical alternatives to small-model adaptation. They are not compute-matched architectural ablations. The matched prompted follow-up below isolates a fixed checkpoint’s inference path; completecandidate likelihood and independently optimized reasoning prompts remain outside scope.

## 4.3. Matched Prompt-Only Follow-up

We cross frozen 2B / the existing seed-7 adapter with ordinary greedy generation / direct candidate readout on the complete Pets and ScienceQA selections, representing source recognition and VQA transfer. Both paths receive identical images, options, and a fixed system format instruction. Ordinary generation has no candidate-token mask. The system cue was selected by output-format compliance, not accuracy, on eight preselected Pets development images. Direct inference is rerun under that cue, rather than compared against the original user-only prompt. Invalid generated answers count as errors. A fixed answer-line parser permits preceding explanation, with a 128-token cap. Warm batch-one timing interleaves direct, ordinary-generation, and constrainedone-token requests on a locked subset. Appendix F specifies parsing, format selection, timing boundaries, and the limits of attribution. This targeted follow-up does not establish readout effects on the other five benchmarks.

## 4.4. Transfer Without Target Fitting

Natural resampling. CIFAR-10.1 v6 contains 2,000 images in the existing ten categories. We reuse the original task instruction, candidate descriptions, source probe heads, and source temperatures. This is a relatively mild natural resampling shift, not an adversarial or corruption stress test.

New task and label space. DTD uses all 1,880 images in the official partition-1 test set and all 47 texture labels. No DTD fitting, development, or calibration labels are used. CLIP uses a fixed texture prompt. Ordinary DINOv2 has no text mapping for the new classes, so its unsupported head-free result is N/A. DTD expands beyond the 37-slot adaptation vocabulary: ten slots were never positive targets during fitting. The evaluation therefore changes visual content, semantics, and candidate cardinality simultaneously. “Out of distribution” refers to adapter supervision, not unknown foundation-model pretraining.

Question-conditioned choices. A-OKVQA uses all 1,145 public validation questions; ScienceQA uses every imagecontaining test question, 2,017 of 4,241. We compare frozen 2B, frozen 4B, and the already selected classification adapter with predesignated seed 7. There is no VQA fitting, demonstration selection, prompt search, temperature estimation, or checkpoint selection. Inputs contain the image, question, ordered choices, and ScienceQA’s hint/context. Rationales, direct answers, lectures, and solutions are excluded. A-OKVQA follows official answer-text scoring, including repeated options; ScienceQA follows official answer-index scoring.

Table 1. Full-set recognition accuracy (%). LoRA reports the mean ± sample SD over seeds 7, 8, and 9; the other rows are single fixed-model evaluations. Both probes use 64 fitting examples per source class. CIFAR-10.1 reuses source heads; DTD receives no support labels. The three left columns are source tasks, and the two right columns are transfer tasks. No target fitting or test-based model selection is used.
<table><tr><td>Method</td><td>Pets</td><td>CIFAR-10</td><td>EuroSAT</td><td>CIFAR-10.1</td><td>DTD</td></tr><tr><td>2B frozen</td><td>60.13</td><td>96.04</td><td>49.63</td><td>93.20</td><td>45.85</td></tr><tr><td>4B frozen</td><td>70.65</td><td>96.92</td><td>56.81</td><td>94.30</td><td>64.79</td></tr><tr><td>2B + LoRA (64-shot)</td><td> $9 2 . 4 0 \pm 0 . 2 6$ </td><td> $9 7 . 0 4 \pm 0 . 1 1$ </td><td> $8 8 . 3 1 \pm 1 . 5 7$ </td><td> $9 5 . 0 5 \pm 0 . 1 3$ </td><td> $5 8 . 2 8 \pm 0 . 9 8$ </td></tr><tr><td>CLIP zero-shot</td><td>89.07</td><td>90.13</td><td>47.67</td><td>84.50</td><td>42.98</td></tr><tr><td>CLIP probe</td><td>91.99</td><td>93.38</td><td>91.39</td><td>88.75</td><td>一</td></tr><tr><td>DINOv2 probe</td><td>95.67</td><td>97.62</td><td>92.13</td><td>93.75</td><td></td></tr></table>

A separate frozen CLIP follow-up scores the same choices by image/text similarity using a fixed Question-Context-Answer template. Within its 77-token window, we reserve every complete candidate answer, then allocate the same question/context prefix to all choices. In these evaluations no question or answer is truncated; ScienceQA context is shortened for 424 questions. This is contrastive matching, not a trained VQA head, and CLIP’s native image crop is not resolution-matched to Qwen. Plain DINOv2 cannot score arbitrary answer text without an additional text-alignment mechanism.

## 4.5. Selection and Uncertainty

The 64-shot budget and checkpoint selection use development endpoints. The scale/OOD and VQA protocols are follow-ups fixed before their respective predictions, not a claim of project-wide prospective preregistration. No target labels guide adaptation, prompts, or selection.

Accuracy is primary; we also retain macro accuracy, NLL, multiclass Brier score, 15-bin expected calibration error (ECE), and selective risk. Reported LoRA spreads are sample standard deviations across optimization seeds, not uncertainty across training splits. Method differences use 2,000 paired bootstrap resamples of image-similarity groups, preserving all questions attached to sampled groups. Intervals are descriptive 95% intervals without multiplicity correction. Comparisons preserve complete evaluation sets and align images, labels, and image-similarity groups across methods.

## 5. What Transfers Through a Shared Interface?

We evaluate the same decision mechanism with different backbones and an optional source-trained adapter. The results distinguish a usable frozen interface from the stronger claim that one adaptation recipe improves every workload.

## 5.1. Source Fitting Improves the Small Model, Not Every Model Ranking

Table 1 shows the largest adaptation gains on Pets and EuroSAT. Full-class Pets accuracy rises from 60.13% for frozen 2B to $9 2 . 4 0 \pm 0 . 2 6 \%$ after adaptation. EuroSAT rises from 49.63% to $8 8 . 3 1 \pm 1 . 5 7 \%$ . CIFAR-10 has less headroom: the corresponding change is 96.04% to $9 7 . 0 4 \pm 0 . 1 1 \%$

The specialist comparison puts these adaptation gains in context. DINOv2’s 64-shot probe has the highest point estimate on all three source tasks: 95.67%, 97.62%, and 92.13%, respectively. On Pets, adapted-minus-CLIP-probe paired intervals include zero for every seed, whereas the adapted models remain below DINOv2. Interface flexibility and fixed-label accuracy are therefore separate selection criteria. Reusing a language-conditioned interface need not be the best choice when the application only requires one fixed classifier.

Increasing frozen scale improves source-task point estimates to 70.65%, 96.92%, and 56.81%. Scale alone does not close the Pets or EuroSAT adaptation gap, but frozen 4B is close to the adapted small model on CIFAR-10. These are results of the specified letter-choice interface, not estimates of the best unconstrained prompting strategy for each model.

## 5.2. Source Adaptation Extends Beyond Its Label Vocabulary

On CIFAR-10.1, adapted 2B reaches $9 5 . 0 5 \pm 0 . 1 3 \%$ , improving over frozen 2B’s 93.20% for every seed with paired intervals above zero. The source-task specialist ranking changes: DINOv2 reaches 93.75%. However, seed $7 \mathrm { { s } }$ advantage over DINOv2 is only 1.25 percentage points (pp), with an unadjusted interval of [0.05, 2.50]. The lower endpoint is close to zero, and these exploratory comparisons do not support broad superiority claims. Against frozen 4B, two of the three adapted-seed intervals include zero; we do not select the remaining seed to claim a robust win.

On DTD, adapted accuracy is $5 8 . 2 8 \pm 0 . 9 8 \%$ , above frozen

2B at 45.85% but below frozen 4B at 64.79%. Every seed improves over the smaller frozen base and remains below the larger frozen model, with the respective paired intervals excluding zero. Adaptation therefore transfers beyond its supervised label vocabulary, but does not substitute uniformly for frozen model scale. This is not evidence of adaptationinduced visual forgetting: accuracy improves relative to the same frozen backbone, and multiple task factors change together.

No target images are flagged by the specified adaptationoverlap check on CIFAR-10.1 or DTD. Consequently, the declared overlap-exclusion sensitivity subsets coincide with the full sets. This is a result of one particular check, not proof of unrestricted data independence.

## 5.3. Per-Question Schemas Work Without VQA Fitting

Figure 2 tests reuse when both the question and candidate semantics change per example. No new VQA head or adapter is fitted. Frozen 2B achieves 80.61% on A-OKVQA and 81.51% on ScienceQA; frozen 4B achieves 85.07% and 89.34%. The classification adapter reaches 83.32% and 82.35%, respectively.

Figure 4 distinguishes point-estimate gains from resolved paired differences. On A-OKVQA, adaptation improves over frozen 2B by 2.71 pp, with interval [1.13, 4.30]. Its difference from frozen 4B is −1.75 pp with interval [−3.68, 0.26], so the larger model’s point-estimate lead is not a resolved paired difference. On ScienceQA, the adapter’s change from frozen 2B is +0.84 pp, with interval [−2.14, 3.33]; this does not establish an adaptation benefit. The adapted-minus-4B difference is −6.99 pp, with interval [−10.73, −4.08].

Frozen CLIP question-answer matching reaches 59.21% on A-OKVQA and 41.99% on ScienceQA. Every evaluated Qwen variant exceeds it on both datasets with paired intervals above zero. CLIP can rank textual choices, but this fixed contrastive formulation is weaker than the evaluated multimodal language models on these workloads. It is not an upper bound on trained CLIP-based VQA systems. Different image preprocessing and CLIP’s shortened ScienceQA contexts also prevent attributing the entire gap to reasoning ability. The interface extends successfully to this workload, but source fitting has not acquired uniformly stronger question-answering competence. These comparisons use one fixed adapter seed; their intervals do not cover adapter-training variability.

## 5.4. Prompt-Only Controls Separate Adaptation from Output Validity

Table 2 shows that Pets’ gain follows the adapter, not a different answer path: both paths rise from 56.36% to 91.63%, or +35.27 pp ([33.47, 37.03]), with identical decisions at fixed weights. On ScienceQA, frozen paths also agree at 77.94%. The adapter reaches 78.93% with generation and 80.81% with direct readout. Its generated improvement is unresolved (+0.99 pp, [−1.18, 2.87]), whereas direct-minus-generated accuracy is +1.88 pp ([1.13, 2.86]).

This latter difference is entirely an output-validity effect in these records: the adapted generator emits 52 out-ofrange letters and three other invalid formats; valid generated decisions agree with direct readout. Direct readout answers 38 of those 55 invalid cases correctly. This is evidence for enforcing the supplied decision space, not improved visual reasoning from a new head. Alternative parsing or full-vocabulary answer SFT remains untested.

Ordinary generation takes 1.32–1.44 times the mean warm direct-request time across these four cells; it generates approximately two tokens per response. The equivalent constrained-one-token path takes only 1.03–1.04 times direct time. Thus the measured saving primarily avoids response decoding/termination and framework work, not an inherently cheaper decision computation. Appendix Table 11 gives absolute times and boundaries. The changed format cue also changes accuracy relative to the original user-only interface; the two tables must not be mixed to attribute gains.

## 5.5. Probability-Valued Outputs Still Need Reliability Evidence

Calibration often lowers held-out NLL, but not every metric improves. The Pets DINOv2 probe worsens slightly in both NLL and ECE; frozen CIFAR-10 and adapted EuroSAT also worsen in ECE despite improved NLL. Appendix Table 5 preserves these negative cases; Figure 3 shows the calibrated Pets reliability and risk-coverage curves.

On DTD, adaptation lowers mean raw NLL from the frozen base’s 2.6985 to 1.9253, while mean raw ECE changes from 0.1905 to 0.2040. Thus target accuracy gains do not imply automatically calibrated target probabilities. We do not fit a target temperature to repair this result. NLL comparisons here are within a dataset and candidate set, not cross-task rankings across different label cardinalities.

Adaptation is an additional cost, not an interface prerequisite. The fitting-budget comparison shows the largest dependence on satellite imagery (Appendix Table 6). The adapter has 7,382,016 trainable parameters; 64-shot summed optimizer-step intervals span 3,220.25–3,373.94 seconds, excluding model loading, development evaluation, saving, and parameter-integrity checks. Appendix B.1 separates this training cost from inference latency.

![](images/b969ec2f50c60e5fa5777affd657c767768424ce195defeb870954375c40c55d.jpg)  
Figure 2. Direct multiple-choice VQA without VQA training, calibration, or prompt selection. CLIP uses fixed question–context–answer matching; LoRA is the existing classification adapter with predesignated seed 7. Bar lengths start at zero and show point estimates, not seed averages. Paired uncertainty is shown in Figure 4. A-OKVQA is public validation, not the hidden-label test leaderboard. Exact two-decimal results appear in Appendix Table 9.

Table 2. Matched checkpoint × readout follow-up: full-set accuracy (%). Gen. is unconstrained greedy generation with a fixed parser; Read. is independently executed candidate-logit inference. Frozen Gen. is the prompt-only baseline; LoRA Gen. uses the existing classification adapter (seed 7), not new training. All four cells use the same system format cue and are separate from the original user-only-prompt results. The follow-up is limited to Pets and ScienceQA. Invalid F/L counts are frozen/adapted generation failures, included as incorrect in the full denominators.
<table><tr><td>Dataset</td><td>n</td><td colspan="2">Frozen 2B</td><td colspan="2"> $2 \mathrm { B } + \mathrm { L o R A }$ </td><td>Invalid F / L</td></tr><tr><td></td><td></td><td>Gen.</td><td>Read.</td><td>Gen.</td><td>Read.</td><td></td></tr><tr><td>Pets</td><td>3,669</td><td>56.36</td><td>56.36</td><td>91.63</td><td>91.63</td><td>0/0</td></tr><tr><td>ScienceQA</td><td>2,017</td><td>77.94</td><td>77.94</td><td>78.93</td><td>80.81</td><td>0/55</td></tr></table>

## 6. Discussion: What Does “General-Purpose” Require?

Reuse is demonstrated; general competence is a goal. PixelJev uses the same contract and candidate-token readout for fixed recognition vocabularies and changing VQA options. Source-trained adapters are reused without target fitting. These are concrete forms of reuse. They do not establish competence on arbitrary visual tasks, nor demonstrate detection, segmentation, action selection, or open-world rejection. The observed specialist and frozen-scale advantages locate the present implementation on a broader design space rather than invalidate its interface.

The boundary to existing methods matters. Candidatelogit readout, LoRA, and temperature scaling are established; Visual Jev already combines native images, adaptation, and shared-context decision execution (Yu & Yao, 2026). Our evidence concerns classification-to-decision transfer, frozen model scale, and specialist alternatives, not a first visual decision architecture or a reproduction of proprietary RLCD. No matched experiment against Visual Jev is reported. The frozen visual tower also means that accuracy gains do not by themselves demonstrate improved visual representations. The prompt-only factorial separates the fixed adapter from its inference path, not candidateconditioned training from ordinary answer SFT.

Current transfer evidence has specific limits. One adaptation split, three optimization seeds, and one VQA adapter seed do not cover training-data or model-family variation. Foundation-model pretraining overlap is unknown. DTD changes images, semantics, candidate count, and supervised slot coverage together; its key attribute can omit other valid descriptions. EuroSAT may retain spatial correlations beyond image-hash grouping. VQA questions may be answerable from text or world knowledge alone, so imagecontaining accuracy is not proof of visual evidence use. Native processors and total training supervision differ across baselines. The fixed nonthinking interface is not each backbone’s best possible unrestricted reasoning strategy.

Candidate confidence is not a trust decision. The distribution excludes unlisted alternatives, and changing the candidate set changes the conditioning event. Scores need not be comparable across different schemas. Low ECE alone is insufficient, and detecting missing visual evidence is not equivalent to predicting answer error. The current diagnostics establish finite readout behavior, not calibrated deployment risk.

## 7. A Research Agenda for Visual Decision Models

The following directions are proposed work, not additional results. Schema robustness should be tested by semantic-ID-aligned permutations, paraphrases, and controlled distractor changes, separating slot/token artifacts from genuine changes in task difficulty. Cross-family transfer requires training on multiple decision families while withholding complete families and their image groups, with matched ordinary VLM and specialist baselines. Evidence-aware decisions require relevant-region interventions, matched irrelevant-region controls, and separate targets for answer correctness versus the need to acquire a better observation. Deployment value must be measured through end-to-end risk, latency, and memory under a concrete workload beyond the present warm, single-request measurements; sharedcontext execution should be compared with already-batched and cached baselines rather than claimed from one-token readout alone. Appendix E gives falsifiable milestones and a staged order.

## 8. Conclusion

PixelJev provides a working native-image decision interface whose candidate schema is supplied at runtime rather than fixed by a task-specific output head. Small frozen multimodal models already instantiate this interface; few-shot language-side adaptation improves source recognition and selected transfers without VQA-specific fitting. Specialist encoders, larger frozen models, and held-out calibration reveal where this initial realization remains limited. The path toward general-purpose visual decision models is therefore not simply to improve one classification score: it is to preserve useful decisions as task semantics, candidate schemas, and visual evidence change.

## Impact Statement

This work examines bounded visual decisions on existing research benchmarks. A syntactically valid choice can still be incorrect, and confidence normalized over supplied alternatives does not account for an omitted correct answer. The system should not be treated as a validated decisionmaker for medical, legal, safety-critical, or surveillance applications. Pretraining overlap, benchmark bias, and performance across demographic groups are not resolved by these experiments. Model and dataset use remains subject to the original licenses and access conditions.

## References

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 3606–3613, 2014. URL https://www.robots .ox.ac.uk/\~vgg/data/dtd/.

Deng, B., Fan, S., Zhang, H., and Xie, X. Jev for scientific decisions: Evaluating semantic choices and their consequences. arXiv:2609.24965, 2026. URL https: //arxiv.org/abs/2609.24965.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 1321–1330, 2017. URL https://proceeding s.mlr.press/v70/guo17a.html.

Helber, P., Bischke, B., Dengel, A., and Borth, D. EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019. URL https://arxiv.org/abs/1709.00029.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://ar xiv.org/abs/2106.09685.

Jiang, D., Li, Y., and Li, B. Jev-Mem: Systemone-controlled agentic memory for efficient AI agents. arXiv:2609.23986, 2026. URL https://arxiv.or g/abs/2609.23986.

Krizhevsky, A. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. URL https://www.cs.toronto.edu/\~k riz/cifar.html.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pp. 2507–2521, 2022. URL https://arxiv.org/abs/2209.09513.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D.,

Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jégou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum ?id=a68SUt6zFt.

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 3498–3505, 2012. URL https://www.robots.ox.ac.uk/\~vgg /data/pets/.

Qwen Team. Qwen3.5 model releases: 2B and 4B. https: //huggingface.co/Qwen/Qwen3.5-2B, 2026. Also https://huggingface.co/Qwen/Qwen3. 5-4B.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763, 2021. URL https://proceedings.mlr.press/v139/r adford21a.html.

Rafe, A. and Das, S. Calibrated decisions at scale: Converting police crash narratives into probabilistic crash variables with a system one model (Jev). arXiv:2609.24052, 2026. URL https://arxiv.org/abs/2609.2 4052.

Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do CIFAR-10 classifiers generalize to CIFAR-10? arXiv:1806.00451, 2018. URL https://arxiv.or g/abs/1806.00451.

Ren, S., Zewde, K., Shen, X., Zhou, Y., Ng, D., Raj, A., Duong, T., Zhang, Y., and Tiangratanakul, N. Open-Jev judgments on CallScreenBench: Calibrated onepass scam screening with a small language model. arXiv:2609.23959, 2026. URL https://arxiv.or g/abs/2609.23959.

Schwenk, D., Khandelwal, A., Clark, C., Marino, K., and Mottaghi, R. A-OKVQA: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pp. 146–162, 2022. URL https://arxiv.org/abs/2206.01718.

TheoLeeCJ. SemIf. https://github.com/TheoL eeCJ/SemIf, 2026. Software repository; formerly OpenJev.

TypeSafe AI. Introducing system one models & Jev. ht tps://typesafe.ai/blog/introducing-s ystem-one-models-and-jev, 2026. See also the model and state documentation at https://docs.t ypesafe.ai/models.md and https://docs.t ypesafe.ai/concepts/state.md.

Yu, G. and Yao, Y. Visual Jev: Accurate and efficient decisions from shared visual context. arXiv:2609.25845, 2026. URL https://arxiv.org/abs/2609.2 5845.

## A. Supervision, Selection, and Implementation

## A.1. Sample Ledger

Table 3 distinguishes fitting labels from development and calibration labels. Selection uses split seed 20260923 throughout. The 16-shot fitting set is a strict subset of the 64-shot set. Development, calibration, and test selections are identical across budgets. The shared adapter sees 912 or 3,648 fitting images in total, plus development labels for checkpoint selection. The additional 684 calibration labels are used only for probability scaling, not for adapter updates.

Table 3. Source supervision and complete transfer denominators. Dashes indicate that no target fitting, development, or calibration split is used. Pets includes every breed.
<table><tr><td>Dataset</td><td>Classes</td><td>Fit: 16-shot / 64-shot</td><td>Development</td><td>Calibration</td><td>Test</td></tr><tr><td>Oxford-IIIT Pets</td><td>37</td><td>592 / 2,368</td><td>444</td><td>444</td><td>3,669</td></tr><tr><td>CIFAR-10</td><td>10</td><td>160 / 640</td><td>120</td><td>120</td><td>10,000</td></tr><tr><td>EuroSAT</td><td>10</td><td>160 /640</td><td>120</td><td>120</td><td>5,400</td></tr><tr><td>CIFAR-10.1 v6</td><td>10</td><td>—</td><td>一</td><td>一</td><td>2,000</td></tr><tr><td>DTD partition 1</td><td>47</td><td></td><td>一</td><td></td><td>1,880</td></tr></table>

We hash decoded RGB pixels together with image dimensions for exact identity. The perceptual hash converts the image to 32 × 32 grayscale, applies a discrete cosine transform, and thresholds the upper-left 8 × 8 coefficients after excluding the DC component. Connected groups use Hamming distance at most four. This grouping is computed over the full selected mirror’s official training and test pools before adaptation sampling. Training groups touching test are excluded; within remaining training groups we retain a representative and exclude conflicting-label groups. The official test denominator is never reduced to make results appear cleaner. Image-similarity grouping is conservative and does not establish that every pair in a connected group is a true duplicate.

## A.2. Optimization and Scoring Settings

Table 4. Fixed optimization and scoring settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>LoRA rank / scaling / dropout</td><td>16 / 32 / 0.05</td></tr><tr><td>Optimizer / learning rate / weight decay</td><td>AdamW / 10−4 / 0.01</td></tr><tr><td>Microbatch / gradient accumulation / gradient norm limit</td><td>1 / 16 / 1.0</td></tr><tr><td>Precision / epochs</td><td>BF16/3</td></tr><tr><td>Gradient checkpointing</td><td>Enabled</td></tr><tr><td>Fitting visits</td><td>Every fitting image once per epoch</td></tr><tr><td>Optimization seeds / split seed</td><td>7,8, 9 / 20260923</td></tr><tr><td>Trainable / complete parameters, including adapter</td><td>7,382,016 /2,220,623,680</td></tr><tr><td>Principal minimum / maximum pixels</td><td>65,536 / 262,144</td></tr><tr><td>Frozen high-resolution control maximum pixels</td><td>1,048,576</td></tr><tr><td>Probe regularization grid</td><td>C ∈ {0.01, 0.1, 1, 10, 100}</td></tr><tr><td>Probe solver / maximum iterations</td><td>LBFGS / 3,000; nonconvergence is an error</td></tr><tr><td>Temperature bounds / ECE bins</td><td>[0.05, 20] / 15 equal-width bins</td></tr><tr><td>Paired bootstrap repetitions / seed</td><td>2,000 / 20260923</td></tr></table>

The explicit language-side projection allowlist contains q\_proj, k\_proj, v\_proj, o\_proj, in\_proj\_qkv, in\_proj\_z, in\_proj\_b, in\_proj\_a, and out\_proj. No original backbone tensor is trainable. All three selected 64-shot positive-training checkpoints are from epoch 3. The fixed visual pixel budget is a processor bound rather than a claim that every image has identical dimensions. The frozen higher-resolution control increases only the upper bound and does not materially improve Pets accuracy.

CLIP/DINO feature vectors are L2-normalized; DINO uses the final CLS representation. Each probe fits its own task labels, unlike the shared multitask Qwen adapter. Native backbone preprocessing is retained, so aligned examples do not imply

equal-resolution or equal-crop inputs.

## B. Additional Probability and Budget Results

Table 5. Held-out probability quality. Subscript T indicates separate source-label temperature calibration; all LoRA rows use seed 7. NLL and ECE are unitless and lower is better within a dataset. Calibration does not change accuracy. The Pets DINOv2 row and the ECE increases on CIFAR-10/EuroSAT show that calibration is not uniformly beneficial across held-out metrics.
<table><tr><td>Task / method</td><td>NLL</td><td> $\mathrm { N L L } _ { T }$ </td><td>ECE</td><td>ECET</td></tr><tr><td>Pets / 2B frozen</td><td>1.8019</td><td>1.5856</td><td>0.1373</td><td>0.0684</td></tr><tr><td>Pets / LoRA</td><td>0.2584</td><td>0.2502</td><td>0.0290</td><td>0.0143</td></tr><tr><td>Pets / DINOv2 probe</td><td>0.1451</td><td>0.1461</td><td>0.0055</td><td>0.0101</td></tr><tr><td>CIFAR-10 / 2B frozen</td><td>0.1664</td><td>0.1521</td><td>0.0148</td><td>0.0165</td></tr><tr><td>CIFAR-10 / LoRA</td><td>0.1090</td><td>0.1047</td><td>0.0095</td><td>0.0068</td></tr><tr><td>EuroSAT / LoRA</td><td>0.3323</td><td>0.3271</td><td>0.0165</td><td>0.0222</td></tr></table>

![](images/b5ab22d15fdfe7089786c2cada16ac06936bc96d38a2d3e2cd340d188ad40d9f.jpg)

![](images/35347747724bce29db721ec54027dd35f2a94e307a23df48eab864597414dbb3.jpg)  
Figure 3. Pets test reliability and selective risk after source calibration. Reliability points are occupied equal-width confidence bins; connecting segments are guides, not smoothed fits or confidence intervals. All methods use their own independently fitted source temperature. LoRA uses seed 7. Risk is the error rate among predictions retained in descending confidence order; curves show the measured evaluation points. Better uncertainty behavior is an empirical property, not a consequence of returning a probability-valued schema.

## B.1. Measured Adaptation Cost

Intervals include accumulated microbatches and image/input preparation. They exclude loading, epoch-end development evaluation, checkpoint saving, and frozen-parameter integrity checks. They are neither exclusive GPU kernel times nor a cost accounting of the whole investigation. The pinned runtime is NGC PyTorch 25.11 with Python 3.12.3, PyTorch 2.10.0a0+b558c986e8.nv25.11, and CUDA 13. Experiments use Transformers 5.17.0, PEFT 0.18.1, Datasets 5.0.1, and scikit-learn 1.7.2. The runtime container is limited to eight CPUs and 48 GB of memory; the underlying machine is shared.

## C. Direct Multiple-Choice VQA

Source and scoring details. A-OKVQA uses the validation split of HuggingFaceM4/A-OKVQA. ScienceQA uses the image-containing test examples from derek-thomas/ScienceQA. Images are decoded to RGB and aligned by

Table 6. Single-seed fitting-budget comparison, accuracy (%). Development, calibration, and test sets are unchanged; this is not a multi-seed data-scaling experiment.
<table><tr><td>Fitting budget</td><td>Pets</td><td>CIFAR-10</td><td>EuroSAT</td></tr><tr><td>16-shot / seed 7</td><td>89.89</td><td>96.84</td><td>67.35</td></tr><tr><td>64-shot / seed 7</td><td>92.20</td><td>97.00</td><td>89.67</td></tr></table>

Table 7. Logged optimizer-step intervals on a shared NVIDIA GB10 system. Sums are not total run wall time.
<table><tr><td>Run</td><td>Steps</td><td>Sum (s)</td><td>Median (s)</td></tr><tr><td>16-shot / seed 7</td><td>171</td><td>927.51</td><td>5.40</td></tr><tr><td>64-shot / seed 7</td><td>684</td><td>3373.94</td><td>4.87</td></tr><tr><td>64-shot / seed 8</td><td>684</td><td>3221.01</td><td>4.72</td></tr><tr><td>64-shot / seed 9</td><td>684</td><td>3220.25</td><td>4.70</td></tr></table>

decoded-pixel identity.

A-OKVQA contains seven questions with repeated option text, including three whose correct text occurs in multiple slots, and one empty distractor. All questions and slots are retained. Official multiple-choice scoring compares answer text, so every slot containing the correct string is accepted. An empty option is rendered as [empty option] to satisfy the nonempty-description interface, while its original string and position are retained for scoring. ScienceQA scores the official answer index. These are benchmark properties, not model errors or exclusion criteria.

CLIP text budget. The frozen contrastive baseline builds Question: ..., optional Context: ..., and Answer: ... segments. The 77-token native window includes start/end tokens. We reserve enough space for the longest complete answer and retain a common question/context prefix for every option. Shorter answers do not receive additional context. Candidate answers are never truncated. All questions and answers fit in both evaluations. No A-OKVQA text is shortened; 424 ScienceQA contexts are shortened. The template and packing rule are fixed before predictions. This baseline has no trained multimodal fusion head and is not intended as an upper bound on CLIP-based VQA.

The ordinary DINOv2 checkpoint has no aligned text encoder, so it cannot perform the same arbitrary text-choice matching directly. Text-aligned variants such as dino.txt introduce additional components and constitute different checkpoints; they are not evaluated here. ScienceQA’s image-similarity bootstrap uses 1,016 groups, fewer than its 1,799 distinct pixel images. These clusters explain why uncertainty cannot be inferred from question count alone.

## D. Instrument Checks and Readout Semantics

The adapted-interface diagnostic uses 74 predeclared Pets development images and the source-selected seed-7 adapter. The positive arm reaches 97.30% accuracy. Reversing the candidates preserves aggregate accuracy but changes some decisions (97.30% decision agreement). The alternate instruction preserves every decision. Substituting actual different-class images reduces accuracy to zero and falls below the same predeclared 75% criterion passed by the positive arm. Direct and constrained one-token generation candidate logits match exactly for every diagnostic example, with maximum absolute difference zero.

These checks establish functioning visual wiring and finite readout equivalence, not arbitrary prompt invariance, universa image dependence, or a measured speedup.

## E. Staged Research Milestones

This appendix specifies proposed work, not completed experiments. The order prioritizes gaps in the present claims over expanding benchmark counts or model sizes.

Stage 1: extend the matched interface comparison. The prompt-only follow-up in Appendix F addresses ordinary answer generation, direct candidate readout, and their measured request costs for the fixed selected checkpoints. Complete-candidate likelihood, broader prompt-selection budgets, and training-objective controls remain open. Compare candidate-conditioned fitting with ordinary full-vocabulary answer SFT under matched supervision and update budgets. Endpoints include accuracy, invalid outputs, log scoring, end-to-end latency, and memory. If conventional VLM inference performs identically, the supported benefit is interface convenience rather than a distinct algorithmic gain.

Table 8. Complete question selections. ScienceQA image inclusion depends only on source image presence, not model outcomes. The two sources have no images flagged by the prescribed adaptation-overlap check.
<table><tr><td>Evaluation</td><td>Questions</td><td>Distinct decoded images</td><td>Choices per question</td></tr><tr><td>A-OKVQA public validation</td><td>1,145</td><td>1,122</td><td>4 throughout</td></tr><tr><td>ScienceQA image-test</td><td>2,017</td><td>1,799</td><td>2: 677; 3: 532; 4: 770; 5: 38</td></tr></table>

Table 9. Exact direct VQA accuracy (%). There is no VQA-specific fitting or calibration. The adapter is predesignated classification seed 7.
<table><tr><td>Method</td><td>A-OKVQA</td><td>ScienceQA</td></tr><tr><td>2B frozen</td><td>80.61</td><td>81.51</td></tr><tr><td>4B frozen</td><td>85.07</td><td>89.34</td></tr><tr><td>2B + classification LoRA (seed 7)</td><td>83.32</td><td>82.35</td></tr><tr><td>CLIP frozen QA matching</td><td>59.21</td><td>41.99</td></tr></table>

Stage 2: isolate schema and evidence sensitivity. Measure semantic-ID-aligned decisions and probabilities under candidate permutations and equivalent descriptions. Vary candidate count and distractor difficulty separately from the image distribution. Adding candidates legitimately changes normalization, so numerical probability invariance is not a valid universal requirement. For visual dependence, compare relevant-region removal with equal-area irrelevant-region controls alongside intact, mismatched, and noninformative images. A grey-image gap alone cannot distinguish all distribution-change effects from genuine use of question-relevant evidence.

Stage 3: test generalization across decision families. Extend to a small set of distinct families, such as spatial relations, counting, and visual text, with entire families withheld from adapter fitting. Split parent images and source groups before creating task variants. Use language-only and candidate-only diagnostics for constructed options. Match total supervision for joint-versus-task-specific adaptation and repeat adaptation splits as well as optimization seeds. Report each held-out family’s outcome and source-task retention, not only a mixture average. A new readout that supports more candidates mus demonstrate held-out cardinality behavior, not merely accept a larger tensor.

Stage 4: connect reliability to the action it serves. Define separate targets for answer correctness, availability of a valid candidate, and sufficiency of the visual evidence. Missing evidence may still permit a correct answer from context; a valid-looking image can still elicit a wrong answer. Fit rejection or re-observation thresholds only on separate calibration data and test unchanged thresholds under shift. Compare risk at matched coverage, or decision utility with stated error and observation costs. A sufficiency detector earns its place through better re-observation decisions, not through its detection AUROC alone. Visual Jev’s sufficiency study (Yu & Yao, 2026) makes this distinction particularly important when designing the next experiment.

Stage 5: evaluate one bounded deployment workload. For example, choose among externally proposed UI objects and measure proposal quality separately from selection quality. Hold the downstream controller fixed when comparing task success, unnecessary actions, latency, and memory. For shared-image questions, compare shared-context serving with already-batched and cached alternatives, including the design of Visual Jev (Yu & Yao, 2026). Shared execution is existing related work, not a novelty claim for this roadmap. Optimize the serving path only when workload measurements identify it as the bottleneck; do not infer an advantage from the absence of generated answer text.

![](images/b279d73ddc86f8a40643d09e20d6533b57b90d168093f16a08922c14d567b993.jpg)

![](images/ef892ee660ab8f452e77c926131713a7ac48d86759f81d904ec0762150849a3a.jpg)  
Figure 4. Adaptation is not a universal substitute for scale. Points are the source-selected seed-7 adapter’s accuracy differences from (a) frozen 2B and (b) frozen 4B under the original user-only prompt. Whiskers are paired image-group bootstrap 95% intervals, not optimization-seed SD, and are not multiplicity-adjusted. The same predesignated seed is used across tasks.

## F. Matched Prompt-Only Baseline and Inference Cost

## F.1. What the Factorial Does and Does Not Identify

The two factors are checkpoint adaptation and inference/readout path: frozen Qwen3.5-2B versus its existing classification adapter (source-selected 64-shot seed 7), crossed with unconstrained prompted generation versus direct candidate logits. No new adapter or temperature is fitted. This targeted follow-up retains the complete Pets and ScienceQA selections. Pets tests a source task; ScienceQA tests question-conditioned transfer. Readout conclusions from these two datasets are no generalized to all seven original benchmarks. The same native image processor, options in the same order, image budget, task/question text, and format instruction apply to both paths.

This design estimates the effect of applying the already-trained adapter under either readout and the effect of changing readout at fixed weights. The interaction is $( A _ { \mathrm { L o R A , r e a d } } - A _ { \mathrm { L o R A , g e n } } ) - ( A _ { \mathrm { f r o z e n , r e a d } } - A _ { \mathrm { f r o z e n , g e n } } )$ . Paired intervals resample the original image-similarity groups. The design does not compare candidate-conditioned training with ordinary full-vocabulary answer SFT at matched supervision and compute. It cannot attribute all “Jev-style” effects to one algorithm or extrapolate one selected adapter to training-seed uncertainty.

## F.2. Development-Only Format Selection

The original principal prompt requests a letter in the user message. A development-only comparison on eight preselected Pets images evaluated two format cues: a system instruction and an assistant Answer: prefill. Its rule preferred the system cue if every selected frozen response was valid and used at most eight generated tokens, otherwise the prefill if it met the same criterion. Both met the criterion; the predeclared system-first rule was applied. Correctness did not select the prompt. No evaluation-set generated response was inspected before this selection.

The fixed additional system instruction is:

You are a visual classifier. Choose one of the provided options. Return exactly its case-sensitive letter, and nothing else.

All four factorial cells are newly executed under this cue. Consequently, these rows must not be spliced into the original user-only results as though the prompts were unchanged. Development examples do not enter the reported evaluation sets.

## F.3. Unconstrained Generation and Parsing

Greedy generation uses one beam, caching, repetition penalty 1, thinking disabled, and at most 128 new tokens. No candidate-vocabulary mask is applied. It stops at either the native end-of-text or chat-end token, or when a newline completes a valid answer line. The fixed parser allows preceding explanation and recognizes a case-sensitive letter on a complete line, optionally parentheses, a final period, paired bold/backticks, or the prefixes Answer:, Final answer:, The answer is, and The correct answer is. Only one unique in-range answer is accepted; repeated identical answer lines do not constitute conflicting choices. Missing/conflicting answers, out-of-range letters, and cap truncation are invalid and coun as incorrect. There is no retry, label-name guessing, or fallback to candidate logits.

Direct inference executes a separate forward pass with caching disabled, rather than taking generation logits as its substitute.   
First-generation-step logits are compared with direct inference for each selected example.

All independently executed direct and first-generation-step candidate logits match exactly in the final selected records. Generation and direct decisions agree for both Pets checkpoints and frozen ScienceQA. For adapted ScienceQA, the 55 disagreements are precisely the 55 invalid generated responses: 52 out-of-range letters and three non-letter formats. There are no cap-truncated responses. All valid generated choices match direct choices; the direct path answers 38 invalid cases correctly. These counts are based on every selected example, not inferred from equal aggregate accuracies. The parsing contract is intentionally fixed: the gap should not be interpreted as the best achievable accuracy of a more permissive or task-aware answer extractor.

Table 10. Checkpoint adaptation effects at fixed readout, and their interaction, in percentage points with paired image-group 95% intervals. Intervals are descriptive and not multiplicity-adjusted.
<table><tr><td>Dataset</td><td></td><td>LoRA — frozen: generation</td><td></td><td>LoRA — frozen: readout</td><td>Interaction</td></tr><tr><td>Pets</td><td></td><td>+35.27 [+33.47, +37.03]</td><td></td><td>+35.27 [+33.47, +37.03]</td><td>+0.00 [+0.00, +0.00]</td></tr><tr><td>ScienceQA</td><td></td><td>+0.99 [−1.18, +2.87]</td><td></td><td>+2.88 [+0.97, +4.68]</td><td>+1.88 [+1.13, +2.86]</td></tr></table>

## F.4. Matched Warm Timing

For each dataset, 64 images are selected by a deterministic, hash-based ordering with seed 20260923, without consulting labels, predictions, or times. Each path receives two warmups. There are three measured repetitions per image; a deterministic random permutation varies the order of the three paths within each image/repetition. Batch size is one and the same checkpoint remains resident during its comparisons. CUDA synchronization brackets each request.

The interval includes local image loading, decoding, pixel-hash validation, prompt construction, preprocessing/tokenization, device transfer, inference, and answer parsing or candidate-probability construction. It excludes model loading, record writing, network, and serving queues. Generation output-logit capture is disabled for timing. Means first average repetitions per image; paired timing intervals resample image-similarity groups. P95 is a descriptive percentile over warm requests. The constrained-one-token control uses candidate-masked generation limited to one token, with caching disabled. It returns the same kind of bounded decision without ordinary multi-token response termination.

Measurements use the pinned shared GB10 runtime, including its reference PyTorch recurrent/convolution kernels. They are implementation-specific warm request times, not exclusive-device service latencies or an optimized generation-engine benchmark. The control distinguishes task-format and termination costs from a supposed inherent advantage of “not generating one token.”

Table 11. Warm timing: each time cell is mean / P95 in milliseconds. Gen./direct and 1-token/direct are ratios of mean request times, not throughput improvements under batching. Every row uses the same fixed 64 images and three repetitions per path.
<table><tr><td>Dataset / weights</td><td>Direct (ms)</td><td>Generated (ms)</td><td>1-token (ms)</td><td>Gen./direct</td><td>1-token/direct</td></tr><tr><td>Pets / Frozen</td><td>127.9 /147.2</td><td>170.0 / 188.8</td><td>131.7 / 149.3</td><td>1.33</td><td>1.03</td></tr><tr><td>Pets / LoRA</td><td>133.5 / 150.8</td><td>175.7 / 193.2</td><td>137.2 / 154.6</td><td>1.32</td><td>1.03</td></tr><tr><td>ScienceQA / Frozen</td><td>94.7 / 112.2</td><td>136.1 / 153.0</td><td>98.5 / 115.8</td><td>1.44</td><td>1.04</td></tr><tr><td>ScienceQA / LoRA</td><td>98.8 / 115.2</td><td>141.1 / 157.9</td><td>101.9 / 118.5</td><td>1.43</td><td>1.03</td></tr></table>

![](images/7c9a2c62c0fb30f86ffa79497c729ae1f86f9e5a62bf5a404a50f7d9c04e05fd.jpg)

![](images/dcd8f799eabdd7f086191aa1fc42e7ef6fe5003eccf49cebab471cc63b76ba21.jpg)

![](images/df783d3ddb195278d8c97bb5f0d90d856b33f4e970ea56fee5f931a85000407d.jpg)  
Figure 5. The inference trade-off at fixed weights. (a) Direct minus ordinary-generation accuracy, with paired image-group 95% intervals. (b) Ratios of mean warm times to direct inference; filled markers denote ordinary generation and open markers constrained one-token generation. Ratio markers are descriptive point estimates, not uncertainty intervals.