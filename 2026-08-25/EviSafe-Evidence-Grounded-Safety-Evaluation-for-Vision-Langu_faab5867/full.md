# EviSafe: Evidence-Grounded Safety Evaluation for Vision-Language Models

Xuetong Li<sup>1</sup> Gaofeng Liu<sup>2</sup>

<sup>1</sup>School of Mathematical Sciences, MOE Key Lab of Scientific and Engineering Computing, Shanghai Center for Applied Mathematics, Shanghai Jiao Tong University <sup>2</sup>Department of Automation, Shanghai Jiao Tong University Shanghai 200240, P.R. China xuetongli@sjtu.edu.cn diehualong@sjtu.edu.cn

## Abstract

Vision-language model safety benchmarks typically evaluate only final responses: whether a model refuses, warns, or complies. This outcome-level view cannot tell whether a model is safe for the right multimodal reason. Safe looking behavior may reflect keyword-triggered refusal, missed visual hazards, or over-refusal of benign-sensitive inputs. We introduce EviSafe, an evidence-grounded framework for VLM safety that jointly evaluates natural user-facing behavior, explicit grounding in textual and vi sual evidence, and behavioral sensitivity to counterfactual changes in safety-critical evi dence. EviSafeBench instantiates the framework as a controlled benchmark with 1,181 gold image-text scenarios and 2,452 targeted counterfactual variants across eight safety domains and eight risk-source types. Each scenario includes a gold safety decision, evidence annotations, a safe-response policy, and counter factual interventions. The three-probe protocol queries models with natural-response, evidencereporting, and counterfactual-response prompts, then scores them using an evidence-aware judge. Across eleven evaluated VLMs, natural severity accuracy ranges from 27.6% to 52.8%, relaxed diagnostic consistency from 6.1% to 29.3%, and unsafe-to-safe counterfactual transition suc cess from 30.4% to 58.4%. These gaps show that the evaluated VLMs are not reliably safe for the right multimodal reason and motivate evaluation beyond refusal counts.

## 1 Introduction

Vision-language models (VLMs) increasingly answer safety-sensitive questions grounded in screenshots, visible text, physical objects, social scenes, private documents, and health-related content (Bai et al., 2025b; Zhu et al., 2025; Li et al., 2025a).

Safety often depends on the relation between modalities. A text query may become risky when an image supplies a harmful object or phrase. Conversely, sensitive visual cues may be benign when the request is preventive, educational, oppositional, or help-seeking. VLM safety therefore requires multimodal evidence grounding, including text reading, document understanding, and compositional reasoning (Singh et al., 2019; Mathew et al., 2021; Masry et al., 2022).

Most safety evaluations reduce this structure to attack success, refusal, or final-response harmfulness. LLM benchmarks standardize red-teaming, jailbreak robustness, and harmful-compliance tests (Mazeika et al., 2024; Chao et al., 2024; Souly et al., 2024). VLM benchmarks and attack studies extend this evaluation to image-conditioned risks, typographic prompts, and compositional imagetext jailbreaks (Liu et al., 2024a; Gong et al., 2025; Shayegani et al., 2024; Qi et al., 2024). Broader suites cover safety, robustness, privacy, and overrefusal (Luo et al., 2024; Zhang et al., 2024; Zheng et al., 2026), while recent work studies holistic evaluation, over-refusal, multi-turn degradation, and dynamic safety behavior (Lee et al., 2026; Ren et al., 2025; Zhu et al., 2026; Wang et al., 2026). Yet a central question remains unresolved: when a VLM gives a safe-looking response, is it safe for the right multimodal reason?

Final answers can be correct for the wrong reason. A model may react to a sensitive keyword, miss the true visual hazard, report the risk yet respond unsafely, or over-refuse when one modality appears risky in isolation, a failure mode isolated by multimodal oversensitivity tests (Li et al., 2025b). Figure 1 shows how negation, defensive intent, scene context, and image-text composition can make the same cue support opposite policies. Outcome scoring therefore conflates evidence-grounded safety with accidental safety, heuristic refusal, and excessive conservatism. Hallucination diagnostics reveal analogous gaps between plausible outputs and visual evidence (Li et al., 2023; Guan et al., 2024); fine-grained faithfulness metrics likewise verify whether atomic claims in free-form answers are supported by the image (Jing et al., 2024). Outcome scoring also cannot localize failures in perception, OCR, cross-modal binding, intent inference, policy conditioning, or response generation.

![](images/16bf47be85ac88499b4051cde6d5656faa57175fd9adf6e6c075adb3ad308ec4.jpg)  
Figure 1: Motivating examples. Safe behavior requires identifying the evidence that determines the decision: negation cues, defensive intent, visual context, and image-text composition can each flip the correct response policy.

EviSafe formulates evidence-grounded multimodal safety evaluation through three observable properties. A safety-capable VLM should behave safely in natural interaction, report the relevant text and visual evidence, risk source, cross-modal relation, and response policy, and adapt when key safety evidence is removed or replaced. Because generated explanations may be post-hoc or unfaithful to the factors that drive an output (Turpin et al., 2023), counterfactual evidence interventions provide a behavioral check inspired by evaluations of dependence on intended visual evidence (Agrawal et al., 2018; Niu et al., 2021; Chen et al., 2020). The framework makes no claim to recover internal causal mechanisms.

EviSafeBench instantiates this framework using 9,399 public multimodal-safety records. Its gold split contains 1,181 scenarios and 2,452 counterfactual variants across eight safety domains, four decision labels, and eight risk-source categories. Each scenario specifies the expected decision, supporting evidence, response policy, and counterfactual interventions. Natural-response, evidence-reporting, and counterfactual-response probes are scored by an evidence-aware judge for severity, risk-source family, evidence alignment, diagnostic consistency, and counterfactual sensitivity. The resulting profile tests whether safe behavior is supported by reportable evidence and responds appropriately to changes in that evidence.

We make three contributions:

• A formulation of VLM safety as evidencegrounded safety behavior, with metrics that separate final-response correctness from reportable evidence alignment and counterfactual sensitivity.

• EviSafeBench, a 1,181-scenario benchmark with 2,452 targeted variants and a three-probe protocol for natural behavior, evidence reporting, and counterfactual behavior.

• An evaluation of eleven VLM runs showing that final-response safety, evidence grounding, and counterfactual sensitivity remain substantially misaligned, revealing failures hidden by refusalstyle metrics.

## 2 Related Work

Safety benchmarks for LLMs and VLMs. LLM safety benchmarks such as HarmBench, Jailbreak-Bench, and StrongREJECT standardize the evaluation of harmful compliance, jailbreak robustness, and refusal behavior (Mazeika et al., 2024;

Chao et al., 2024; Souly et al., 2024). Multimodal safety benchmarks and attacks extend this evaluation setting to image-conditioned risks, typographic prompts, and compositional imagetext jailbreaks. Representative examples include MM-SafetyBench, FigStep, Jailbreak in Pieces, CSR-Bench, JailBreakV, SafeBench, and broader trustworthy-MLLM suites (Liu et al., 2024a; Gong et al., 2025; Shayegani et al., 2024; Liu et al., 2026; Luo et al., 2024; Ying et al., 2026; Zhang et al., 2024); SPA-VL instead provides a large-scale safetypreference alignment dataset (Zhang et al., 2025). Complementary benchmarks isolate visual safety leakage and context-dependent policy changes: VLSBench controls for risky information exposed by the text query, while MM-SafetyBench++ pairs unsafe inputs with minimally edited safe counterparts (Hu et al., 2025; Zhang et al., 2026). Beyond evaluation, safety-oriented training uses curated multimodal safety instructions, dual helpfulness– safety preferences, or explicit modality-level safety tags (Zong et al., 2024; Ji et al., 2025; Rong et al., 2026). These resources support the measurement and improvement of harmful-compliance and jailbreak resistance. However, most benchmark evaluations primarily score the final response or broad safety category, leaving under-specified whether a safe-looking response is grounded in the evidence that actually determines the safety decision.

General multimodal evaluation and automated judging. General MLLM benchmarks evaluate perception, OCR, reasoning, and multimodal instruction following (Fu et al., 2025; Liu et al., 2024b; Yu et al., 2024; Lee et al., 2024b). These capabilities are highly relevant to safety because a VLM may fail safely or unsafely depending on whether it reads visible text, recognizes objects, or binds visual evidence to user intent. Automated judges further enable scalable evaluation of open-ended model outputs (Zheng et al., 2023; Liu et al., 2023; Kim et al., 2024). Multimodal evaluators such as Prometheus-Vision extend rubric-conditioned judging to outputs that must be grounded in images (Lee et al., 2024a), but recent work also shows that judgebased safety evaluation can be sensitive to rubric and configuration choices (Schwinn et al., 2026; Zhang, 2026). This motivates explicit, evidenceaware judgment targets rather than relying on a generic harmfulness or refusal judgment alone.

Evidence grounding and counterfactual evaluation. EviSafeBench is also related to evidencegrounded and counterfactual multimodal evaluation. Prior VQA and vision-language studies use counterfactual or prior-shift settings to test whether answers depend on the intended visual evidence rather than dataset priors (Agrawal et al., 2018; Niu et al., 2021; Chen et al., 2020). Scene-graph perturbations further create visual contrast sets that test relational and compositional consistency under controlled semantic changes (Bitton et al., 2021). EviSafeBench adapts these ideas to multimodal safety. Each scenario includes the expected safety decision, text evidence, visual evidence, crossmodal relation, safe-response policy, risk source, and targeted counterfactual variants. The benchmark complements ASR-style evaluation. UCR preserves an under-refusal view, while the remaining metrics test whether the model identifies the right evidence, attributes the right risk source, and changes behavior when safety-critical evidence is removed or replaced.

## 3 EviSafeBench: Core Design

## 3.1 Task Definition

Let an original scenario be $x = ( I , T )$ , where I is an image and $T$ is the user text. Each gold annotation is

$$
a = ( y , s , E _ { T } , E _ { I } , C , P , \mathcal { X } _ { \mathrm { c f } } ) ,
$$

where $y$ is a four-way expected decision, s is the risk source, $E _ { T }$ and $E _ { I }$ are text and visual evidence, C is the cross-modal relation, $P$ is the saferesponse policy, and $\mathcal { X } _ { \mathrm { c f } }$ contains counterfactual variants. The four decisions are safe\_answer, safe\_answer\_with\_caution, unsafe\_warn, and unsafe\_refuse. Evidence fields identify the relevant user-text cue, visual cue or OCR, imagetext relation, and safe-response policy. Here, gold evidence means quality-controlled reference annotations of observable, decision-relevant input cues rather than hidden causal truth or model-internal reasoning.

This formulation distinguishes the benchmark from outcome-only safety datasets. Each scenario includes a four-way safety decision together with the evidence that justifies it and the corresponding safe-response policy. Evaluation can therefore test whether a model reaches the correct safety behavior and reports the supporting decision-relevant evidence.

<table><tr><td>Statistic</td><td>Count</td></tr><tr><td>Raw public records</td><td>9,399</td></tr><tr><td>Final auxiliary candidate pool</td><td>8,670</td></tr><tr><td>Gold original scenarios</td><td>1,181</td></tr><tr><td>Gold counterfactual variants</td><td>2,452</td></tr><tr><td>Total gold image-text items</td><td>3,633</td></tr><tr><td>Generated visual counterfactual images</td><td>903</td></tr><tr><td>Safety domains / risk-source types</td><td>8/8</td></tr><tr><td>Expected-decision labels</td><td>4</td></tr></table>

Table 1: Main construction and coverage statistics for EviSafeBench.

## 3.2 Dataset Construction

After normalizing 9,399 public records, EviSafeBench follows three dependent annotation steps. Expected decision. Source labels, GPT-5.4 multimodal relabeling, an independent Qwen-VL agreement check, and targeted adjudication assign the intended natural-response behavior. Removing 302 unresolved cases leaves 9,097 scenarios. Evidence report. Given the image, user text, and finalized decision, GPT-5.4 annotates text evidence, visual evidence, their cross-modal relation, the safe-response policy, and risk source. Counterfactual construction. The evidence report determines one to three minimal interventions. Text variants are written directly, while visual variants receive edit specifications that are materialized after gold selection. The report and intervention specifications share one structured multimodal output, with counterfactual types constrained by the annotated risk source. Model-assisted multimodal annotation can scale curation, while critical-sample identification can focus human review on uncertain records rather than treating model output as unquestioned gold (Lin et al., 2025).

Validity checks retain 8,689 full-pool scenarios. The same checks reduce a safety-focused, domainstratified sample of 1,500 Step 1 records to 1,488 gold candidates. Selection produces 1,200 scenarios while preserving all decisions, domains, risk sources, rare categories, and flagged cases. Manual adjudication of all 64 flagged cases retains 45 records after decision-label correction and removes 19 low-quality records from both pools. The final auxiliary pool therefore contains 8,670 scenarios, while the gold release contains 1,181 originals and 2,452 counterfactuals (Table 1). Appendix A, Appendix C.1, and Appendix C.5 provide the prompts, source statistics, validation checks, and selection rules.

The benchmark is intentionally not a collection of maximally adversarial prompts. Instead, it covers seven unsafe or high-stakes domains plus a benign-sensitive over-refusal control domain: cyber/privacy/surveillance, physical hazards, sexual exploitation, health and self-harm, hate and harassment, high-stakes misinformation and governance, fraud and illegal economic behavior, and benign-sensitive traps. Domain labels describe the safety topic, while risk-source labels describe which modality or cross-modal relation determines the safety decision. Benign-sensitive traps can therefore appear as a risk-source type outside the benign-sensitive domain. This design lets the same evaluation protocol measure under-refusal, overrefusal, and evidence blindness. Safe labels are not filler negatives. Many require caution because the input contains sensitive evidence but asks for prevention, interpretation, reporting, or other allowed assistance.

Evidence-targeted counterfactuals. Counterfactuals intervene on the annotated reason for the decision rather than producing arbitrary paraphrases. Text-only risks receive neutralized user text, OCR risks receive visual text edits or replacement images, and object and scene risks receive local visual edits when possible. Benign-sensitive traps include variants that preserve the sensitive surface cue while changing the safe-response policy. Each counterfactual has its own expected decision, so evaluation does not assume that variants should always become safe.

In practice, most critical variants test whether removing unsafe evidence moves the model from an unsafe-family decision to a safe-family decision, while decision-preserving variants test whether behavior remains stable when irrelevant surface features change. This design is important because explanation-only evaluation can be gamed by plausible post-hoc rationales. More generally, small but meaningful perturbations can expose local decision boundaries that ordinary test sets miss (Gardner et al., 2020). Counterfactuals provide a behavioral check: if the evidence that justifies the safety decision changes, the model’s response should change accordingly.

Overall, 85.5% of variants modify text, OCR, or localized objects. Another 12.9% modify scene or context and form the main artifact-sensitive subset. The remaining 1.6% use other interventions. Appendix Table 17 gives the full distribution.

Risk-source taxonomy. The benchmark annotates eight risk-source types: text-only risk, image OCR risk, image-object risk, image-scene risk, cross-modal OCR-text risk, cross-modal objecttext risk, benign-sensitive traps, and ambiguous or insuficient evidence. These types are tied to intervention design. OCR risks invite OCR replacement, object and scene risks invite visual replacement, text-only risks invite text neutralization, and benignsensitive traps test over-refusal. For evaluation, we map the eight fine-grained types into six families to reduce unnecessary boundary noise while preserving the diagnostic distinction among text, OCR, object, scene, benign-trap, and ambiguous failures.

The family metric is important because finegrained risk-source labels have real boundary cases. For example, a visible threatening phrase paired with a user request may be viewed as image-OCR risk or cross-modal OCR-text risk depending on whether the text alone already establishes harm. The eight-way annotation is retained for dataset analysis and intervention design, but the main model comparison uses families. This preserves the central diagnostic question of whether the model localized risk to text, OCR, visual objects, scene semantics, benign traps, or insuficient evidence, without over-penalizing annotator-equivalent subtype choices.

## 3.3 Three-Probe Evaluation

The protocol separates three probes, as summarized in Figure 2. The natural-response probe asks the VLM to answer the original image-text input under a minimal helpful-assistant system prompt. This measures what a user would see in a normal interaction. The evidence-report probe gives the same image-text input but requires JSON fields for decision, risk source, text evidence, visual evidence, cross-modal relation, and safe-response policy. It forbids answering the user’s request. JSON is only the interface. Scoring tests semantic alignment with scenario-specific references, while the non-JSON counterfactual probe checks whether natural behavior changes when annotated evidence changes.

A full attempted run yields 4,814 victim requests per model: 1,181 natural originals, 1,181 evidence reports, and 2,452 counterfactual responses. Natural and counterfactual probes share the same userfacing setting, while the evidence-report probe is diagnostic. Scenario-level metrics are computed only for parsed scenarios with all required victim outputs available. Table 2 therefore reports N as the number of successfully judged original scenarios, not the number of attempted requests. This separation distinguishes three cases that outcome-only evaluation conflates: correct natural behavior with wrong evidence, correct evidence recognition with unsafe natural behavior, and behavior that fails to change under evidence-targeted counterfactuals.

## 3.4 Evidence-Aware Judging

Each scenario is scored by an external GPT-5.4 judge that receives the gold annotation and model outputs, but not the original image. The judge returns four-way decisions, risk-source family, graded evidence scores, counterfactual decisions, and failure modes. A pilot audit of 100 outputs checked rubric adherence. A separate blinded, stratified audit of 200 scenario-model records found agreement between the judge and human annotators of 92.5% (κ = .89) for natural decisions, 88.0% (κ = .83) for risk-source families, and 84.3% (κ = .78) for evidence fields. Such task-specific validation is important because multimodal judges can struggle to follow and switch among fine-grained criteria (Xiong et al., 2026). Appendix B.5 gives the audit protocol.

The judge prompt is deliberately evidence-aware rather than merely preference-based. It is given the gold evidence, the natural response, the evidence report, and all counterfactual responses for the scenario, and it emits both decision labels and diagnostic field scores. Counterfactual responses are judged against their own gold decisions, not against the original label, so the protocol can distinguish desired unsafe-to-safe transitions from decisionpreserving cases where behavior should not change.

## 3.5 Metrics

Natural severity accuracy (NSA) and diagnostic severity accuracy (DSA) compare judge decisions with the gold decision using a severity-aware rule. Unsafe compliance rate (UCR) is an ASRstyle auxiliary metric: the fraction of gold-unsafe originals whose natural response is judged as a safe-family decision, meaning safe\_answer or safe\_answer\_with\_caution rather than warning or refusal, so lower is better. Risk-source family accuracy (RSFA) compares predicted and gold risk-source families. The evidence score (ES) averages four graded fields: text evidence, visual evidence, cross-modal relation, and saferesponse policy, with correct=1, partial=0.5, and incorrect=0. Strict diagnostic consistency (SDC) requires correct natural severity, correct risksource family, and all four evidence fields correct. Relaxed diagnostic consistency (RDC) allows partial but usable evidence alignment. Counterfactual behavior accuracy (CBA) measures severity correctness over all judged counterfactual responses. Unsafe-to-safe transition success (U2S) requires both an unsafe-family original response and a safefamily counterfactual response on gold unsafe-tosafe pairs. Conditional U2S (cU2S) measures the safe shift only among pairs whose original response is already in the unsafe family. Appendix Table 5 reports this decomposition together with preservation accuracy on decision-preserving pairs. These are diagnostic rather than separate main-ranking metrics. Formal formulas and failure-mode definitions are in Appendix B.

![](images/f72ab8cba94f7695b45de02de535d7586334d6e1562978061f85e93c233651c3.jpg)  
Figure 2: Overview of the EviSafe evaluation protocol. Each gold scenario is evaluated through natural-response, structured evidence-report, and counterfactual-response probes. An evidence-aware judge compares all outputs with the gold annotation to produce behavior, evidence, risk-source, consistency, and counterfactual metrics.

The metrics form three groups: natural behavior (NSA, UCR), evidence grounding and consistency (DSA, RSFA, ES, SDC, RDC), and counterfactual behavior (CBA, U2S). All are reported as percentages. For model development, we first establish natural behavior with NSA/UCR, then prioritize U2S and use ES/RDC to localize evidence and coordination failures. The metrics are not combined into a composite score because evidence quality must not compensate for unsafe behavior and deployment-specific weights would hide model trade-ofs.

## 4 Experiments

## 4.1 Models and Setup

We evaluate eleven VLMs. Seven open VLMs are served locally through vLLM: Qwen2.5-VL-7B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, InternVL3-8B, InternVL3-14B, LLaVA-OneVision-Qwen2-7B, and Gemma-3-12B-IT (Bai et al., 2025b,a; Zhu et al., 2025; Li et al., 2025a; Gemma Team, 2025). Four additional models— Gemini 2.5 Pro, Claude Opus 4.6, Doubao-Seed-1.6-Vision, and Qwen3-VL-235B-A22B-Instruct— are evaluated through the same formal protocol. Victim decoding uses temperature 0. Local and default API runs use 512 natural-response tokens and 900 evidence-report tokens, while Gemini 2.5 Pro uses 4096/8192 tokens to accommodate providerside reasoning tokens and avoid empty assistant responses. The judge is GPT-5.4 with temperature 0, using one evidence-aware request per successfully parsed scenario. All image inputs are public HTTPS OSS URLs, and all runner details are given in Appendix B.

Victim prompts are identical across measured models except for serving-specific formatting required by each model’s chat template. Natural and counterfactual probes use the same minimal system instruction. The evidence-report probe uses a diagnostic JSON system instruction and never asks the model to answer the user’s original request. Diferences in NSA, ES, and counterfactual behavior therefore reflect model behavior under the same benchmark protocol rather than per-model safety prompt tuning. Failed provider calls, contextwindow failures, malformed outputs, or missing required victim outputs are recorded and excluded from scenario-level judge requests. Runs can therefore have diferent judged sample sizes, reported explicitly as N in Table 2.

## 4.2 From Outcome Safety to Evidence-Policy Coordination

Table 2 reports results for the eleven evaluated VLMs following the three metric groups above. It is a diagnostic profile rather than a single leaderboard. A visualization of the local open-source model profiles is provided in Appendix Figure 6.

Safe-looking behavior is not calibrated safety. No model exceeds 52.8 NSA, and UCR spans 20.7 to 67.7. More importantly, the two outcome metrics expose diferent errors rather than a single safety ranking. NSA tests four-way severity alignment across all scenarios, while UCR isolates unsafe originals that receive safe-family answers. Claude Opus 4.6 has the lowest UCR, but the judge taxonomy assigns 29.6% of its rows to over-refusal. Doubao-Seed-1.6-Vision shows the opposite profile, with 61.1 UCR and 39.5% under-refusal. Reducing unsafe compliance can therefore trade under-refusal for excessive conservatism without solving calibrated policy selection.

Correct decisions are usually not evidencegrounded. The mutually exclusive decisionevidence taxonomy sharpens this distinction. Across 12,459 judged model-scenario records, 29.8% are labeled heuristic success, where the natural decision is severity-correct but important evidence, risk-source attribution, or cross-modal relation is incomplete. Only 9.8% are labeled evidence-grounded success. Among these two success categories, 24.7% are evidence-grounded. Thus correct natural behavior appears about three times as often without adequate reportable grounding as with it. The central failure is not simply that models answer incorrectly. Many correct answers remain indistinguishable from keyword-triggered refusal or other shallow safety priors.

The remaining categories support the same staged interpretation. Under-refusal is the largest explicit failure category at 23.7% of all records. Policy failure accounts for another 16.2%, exceeding the 12.6% complete-failure rate. In policy-failure cases, the evidence diagnosis is mostly correct but the natural decision is not. Failures after partial recognition are therefore at least as important as complete evidence blindness. The 7.9% aggregate over-refusal rate further shows that stronger refusal alone would move errors between categories rather than repair the underlying decision process.

The main bottleneck appears after cue detection. The field-level results in Appendix Table 6 localize this gap. Averaged across the eleven model summaries, text and visual evidence scores reach 71.4 and 71.1, while cross-modal relation and saferesponse policy reach only 43.6 and 51.1. Models often recover relevant words, OCR, objects, or scenes, but are much less reliable at determining how the modalities jointly create risk and which response boundary follows. The conjunction metrics expose the consequence. Even though several models exceed 69 ES, the best SDC is 15.2 and the best RDC is 29.3. Marginal evidence recovery therefore rarely becomes a jointly correct chain of risk attribution, policy selection, and natural behavior.

Diagnostic recognition is prompt-dependent. The diference between DSA and NSA identifies a further break between reporting and action. Doubao-Seed-1.6-Vision gains 20.1 points under the structured diagnostic prompt, while Claude Opus 4.6 and InternVL3-14B gain about 9.5 points. Their risk recognition is more accessible in an evaluator-style report than in a user-facing answer. The reverse gap for LLaVA-OneVision-Qwen2-7B at −12.4 points and Qwen3-VL-8B at −8.3 points shows that a natural response can be more often severity-correct than the model’s explicit diagnosis. Such behavior is compatible with heuristic safety priors or weak diagnostic instruction following, not stable evidence use. Natural and diagnostic probes are therefore complementary. Neither one is a substitute for the other.

The mismatch runs in both directions at the model level. Qwen3-VL-4B leads NSA at 52.8, yet its 36.1 RSFA, 0.8 SDC, and 18.7 RDC show that comparatively strong outcomes do not imply complete evidence alignment. Doubao-Seed-1.6- Vision instead reaches 51.1 DSA and 69.9 ES, but only 31.0 NSA with 61.1 UCR. The former profile is consistent with correct-looking behavior that lacks a complete evidence trace. The latter shows reportable diagnosis that fails to control natural behavior. These are distinct training targets, even when an outcome-only score treats both as generic safety errors.

<table><tr><td>Model</td><td>N</td><td>NSA</td><td>UCR↓</td><td>DSA</td><td>RSFA</td><td>ES</td><td>SDC</td><td>RDC</td><td>CBA</td><td>U2S</td></tr><tr><td>Qwen2.5-VL-7B</td><td>1176</td><td>36.2</td><td>55.9</td><td>36.5</td><td>13.6</td><td>46.1</td><td>0.4</td><td>6.7</td><td>76.7</td><td>39.9</td></tr><tr><td>Qwen3-VL-4B</td><td>1179</td><td>52.8</td><td>32.0</td><td>55.4</td><td>36.1</td><td>60.2</td><td>0.8</td><td>18.7</td><td>74.5</td><td>58.0</td></tr><tr><td>Qwen3-VL-8B</td><td>1180</td><td>50.3</td><td>34.0</td><td>42.0</td><td>43.8</td><td>62.6</td><td>4.2</td><td>22.2</td><td>74.7</td><td>55.2</td></tr><tr><td>InternVL3-8B</td><td>1178</td><td>33.4</td><td>62.9</td><td>33.2</td><td>25.0</td><td>53.2</td><td>2.1</td><td>9.8</td><td>75.9</td><td>33.4</td></tr><tr><td>InternVL3-14B</td><td>1178</td><td>35.6</td><td>60.2</td><td>45.2</td><td>40.3</td><td>56.0</td><td>3.5</td><td>14.2</td><td>75.7</td><td>35.2</td></tr><tr><td>LLaVA-OneVision-Qwen2-7B</td><td>1069</td><td>27.6</td><td>67.7</td><td>15.2</td><td>16.6</td><td>21.4</td><td>0.0</td><td>6.1</td><td>74.0</td><td>30.4</td></tr><tr><td>Gemma-3-12B-IT</td><td>1176</td><td>42.3</td><td>42.7</td><td>39.7</td><td>46.7</td><td>66.8</td><td>7.0</td><td>19.2</td><td>74.1</td><td>48.9</td></tr><tr><td>Gemini 2.5 Pro</td><td>1032</td><td>44.0</td><td>54.0</td><td>44.5</td><td>57.8</td><td>69.1</td><td>9.5</td><td>29.3</td><td>77.2</td><td>41.0</td></tr><tr><td>Claude Opus 4.6</td><td>1056</td><td>41.5</td><td>20.7</td><td>51.0</td><td>50.8</td><td>75.1</td><td>15.2</td><td>26.1</td><td>68.2</td><td>58.4</td></tr><tr><td>Doubao-Šeed-1.6-Vision</td><td>1107</td><td>31.0</td><td>61.1</td><td>51.1</td><td>47.1</td><td>69.9</td><td>4.8</td><td>13.5</td><td>74.6</td><td>35.6</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>1128</td><td>51.9</td><td>38.7</td><td>51.1</td><td>48.0</td><td>71.9</td><td>9.0</td><td>25.6</td><td>80.2</td><td>55.2</td></tr></table>

Table 2: Main formal results. N is the number of original scenarios with complete parsed victim outputs and a GPT-5.4 judge row. All metrics are percentages. NSA denotes natural severity accuracy. UCR denotes unsafe compliance rate, an ASR-style under-refusal metric where lower is better. DSA denotes diagnostic severity accuracy. RSFA denotes risk-source family accuracy. ES denotes mean graded evidence score. SDC and RDC denote strict and relaxed diagnostic consistency. CBA denotes counterfactual behavior accuracy over all judged variants. U2S denotes unsafe-to-safe transition success. Bold marks the best reported value in each metric column.

## 4.3 Counterfactual Adaptation and Scaling

Joint transition success has two failure stages. CBA is relatively compressed from 68.2 to 80.2, whereas U2S ranges from 30.4 to 58.4. The decomposition in Appendix Table 5 shows why U2S should not be read as pure sensitivity to evidence removal. Claude Opus 4.6 recognizes 77.9% of originals in the unsafe-to-safe subset as unsafe, then shifts to safe on 74.9% of those recognized cases. Doubao-Seed-1.6-Vision and LLaVA-OneVision-Qwen2-7B recognize only 37.0% and 31.5% of the originals, although their conditional shifts reach 96.4% and 96.6%. A low joint U2S can therefore reflect failure to recognize the initial risk or failure to relax after its removal. OrigU and cU2S separate these two intervention stages.

Counterfactual robustness requires both change and invariance. A model should change policy when decision-critical evidence changes and preserve policy when the gold family remains unchanged. Claude illustrates why both properties matter. It has the highest joint U2S at 58.4 but the lowest preservation accuracy at 62.4 and the lowest overall CBA at 68.2. Qwen3-VL-235B-A22B instead reaches the highest preservation accuracy at 81.8 and the highest CBA at 80.2, while its U2S is 55.2. Counterfactual safety is therefore a stabilityplasticity problem rather than a one-directional preference for more behavioral change.

Scale changes the capability profile rather than the full safety chain. The two within-family comparisons do not support a simple monotonic scaling account. Moving from Qwen3-VL-4B to 8B raises RSFA by 7.7 points and RDC by 3.5 points, yet lowers NSA by 2.5, DSA by 13.4, and U2S by 2.8 points. Moving from InternVL3-8B to 14B raises DSA by 12.0 and RSFA by 15.3 points, but improves NSA by only 2.2 and U2S by 1.8 points. Additional capacity can strengthen attribution or diagnostic organization without reliably improving natural policy execution. Because architecture, data, and post-training remain confounded across model families, these comparisons support a capabilityprofile interpretation rather than a causal scaling law.

Overall finding. The metrics reveal successive breaks in one safety chain rather than unrelated model rankings. Weak models may miss the cue itself. Stronger models often recover text and visual fragments but fail to bind them into the correct risk source and response policy. Some can state the diagnosis without enacting it naturally, while others produce safe-looking answers without adequate grounding. Counterfactual pairs then distinguish initial risk recognition, appropriate policy change, and decision-preserving stability. Together, these results answer the motivating question in the negative. The evaluated VLMs are not reliably safe for the right multimodal reason because evidence perception, cross-modal composition, policy selection, and user-facing action do not yet operate as a coordinated process. The resulting targets are correspondingly specific: improve relation and policy grounding, align diagnostic and natural behavior, and train on paired decision-changing and decision-preserving interventions.

## 5 Discussion, Limitations, and Conclusion

## 5.1 Practical Implications

The results suggest that safe-looking responses should not be treated as suficient evidence of robust VLM safety. Outcome-only metrics can conflate evidence-grounded safety, accidental refusal, heuristic conservatism, and brittle compliance. By separating natural behavior from evidence reporting and counterfactual behavior, EviSafeBench makes these cases distinguishable: a model can be correct in its final response while missing the risk source, or it can describe the relevant evidence under a structured prompt while failing to act safely in a natural interaction.

This diagnostic structure turns a failed test into a more specific improvement target. Low RSFA points to risk-source attribution errors. Low fieldlevel ES points to missed text, visual, OCR, crossmodal, or policy evidence. Gaps between DSA and NSA indicate failures in translating diagnosis into user-facing behavior. Low RDC or SDC indicates that correct behavior and usable evidence alignment do not co-occur. Low U2S identifies an incomplete unsafe-to-safe transition. Comparing original unsafe recognition with cU2S separates failures to detect the initial risk from failures to relax the policy after evidence removal. These signals can guide whether future work should focus on perception and OCR, cross-modal binding, over-refusal on benign-sensitive traps, or evidence-to-policy coupling.

The same structure also supports more informative safety auditing and model profiling, while stopping short of deployment certification. A EviSafe run records not only whether a model complied or refused, but also which evidence it reported, which risk-source family it selected, and whether its behavior changed under targeted counterfactuals. This provides an evaluation trace for comparing models with diferent deployment trade-ofs, such as low unsafe compliance, stronger evidence localization, lower over-refusal, or better counterfactual sensitivity.

For example, Doubao-Seed-1.6-Vision combines 69.9 ES and 51.1 DSA with 31.0 NSA and 61.1 UCR, pointing to an evidence-to-policy bottleneck rather than cue blindness alone. Future work can train on paired decision-changing and decisionpreserving counterfactuals by adding an evidencesensitivity objective to SFT or RLHF, complementing existing multimodal safety tuning with instruction, preference, or rule-governed supervision (Zong et al., 2024; Ji et al., 2025; Rong et al., 2026). The present study identifies this direction but does not test a mitigation method.

## 5.2 Limitations

The claims concern reportable and behaviorally checked evidence grounding, not mechanistic faithfulness. Visual evidence is natural-language rather than box-level. Generated counterfactual images can contain artifacts, and whole-scene edits are less reliable than OCR, object, and local-context edits. More generally, instruction-based image editing requires separate checks of semantic edit success, regional preservation, and low-level image quality (Ma et al., 2024). The 200-record human audit supports the GPT-5.4 rubric, but judge choice and configuration can still afect automatic metrics. Because rows require all victim outputs, N difers across models and unresolved missingness can introduce small comparability biases. Rare-category slices are descriptive failure-localization probes, not reliable subgroup rankings, and the controlled gold set does not estimate real-world risk prevalence. The larger Gemini 2.5 Pro token budget is also a serving exception. Further work should add finer visual annotations, multiple-judge sensitivity checks, more uniform serving budgets, and stronger scene-level counterfactual generation.

## 5.3 Ethics

The benchmark is intended for defensive safety evaluation. Public examples are redacted or schematic, prompts and evidence descriptions are non-operational, and human annotators should receive content warnings and skip options. The goal is failure localization, not attack optimization.

## 5.4 Conclusion

This paper introduced EviSafe and EviSafeBench for evidence-grounded VLM safety evaluation. The framework jointly evaluates natural safety behavior, diagnostic evidence reporting, and counterfactual evidence sensitivity over 1,181 multimodal scenarios and 2,452 counterfactual variants. Under this observable definition, the results answer the motivating question in the negative: the evaluated VLMs are not reliably safe for the right multimodal reason. Comparatively strong final-answer behavior can coexist with weak risk-source and evidence consistency, while strong diagnostic evidence reports can coexist with high unsafe compliance in natural responses. This implies that a central bottleneck lies not only in perceiving safety-relevant evidence, but also in coordinating text, OCR, visual objects, scene semantics, and user intent with the appropriate response policy. By moving beyond refusal counts, EviSafe localizes failures in perception, cross-modal binding, policy conditioning, overrefusal, and counterfactual stability, providing a diagnostic protocol for targeting evidence-to-policy coordination rather than relying only on refusal optimization.

## References

Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi. 2018. Don’t just assume; look and answer: Overcoming priors for visual question answering. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 4971–4980.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025a. Qwen3-VL technical report. Preprint, arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-VL technical report. Preprint, arXiv:2502.13923.

Yonatan Bitton, Gabriel Stanovsky, Roy Schwartz, and Michael Elhadad. 2021. Automatic generation of contrast sets from scene graphs: Probing the compositional consistency of GQA. In Proceedings ofthe 2021 Conference ofthe NorthAmerican Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 94–105. Association for Computational Linguistics.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, Hamed Hassani, and Eric Wong. 2024. JailbreakBench: An open robustness benchmark for jailbreaking large language models. In Advances in Neural Information Processing Systems 37, pages 55005–55029. Neural Information Processing Systems Foundation, Inc.

Long Chen, Xin Yan, Jun Xiao, Hanwang Zhang, Shiliang Pu, and Yueting Zhuang. 2020. Counterfactual

samples synthesizing for robust visual question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10800–10809.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. 2025. MME: A comprehensive evaluation benchmark for multimodal large language models. In Advances in Neural Information Processing Systems 38, pages 162549–162567. Neural Information Processing Systems Foundation, Inc.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, and 7 others. 2020. Evaluating models’ local decision boundaries via contrast sets. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 1307–1323. Association for Computational Linguistics.

Gemma Team. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. 2025. FigStep: Jailbreaking large visionlanguage models via typographic visual prompts. Proceedings of the AAAI Conference on Artificial Intelligence, 39(22):23951–23959.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2024. HallusionBench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385.

Xuhao Hu, Dongrui Liu, Hao Li, Xuanjing Huang, and Jing Shao. 2025. VLSBench: Unveiling visual leakage in multimodal safety. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8285– 8316. Association for Computational Linguistics.

Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, and Yaodong Yang. 2025. Safe RLHF-V: Safe reinforcement learning from multi-modal human feedback. In Advances in Neural Information Processing Systems 38, pages 46146–46182. Curran Associates, Inc.

Liqiang Jing, Ruosen Li, Yunmo Chen, and Xinya Du. 2024. FaithScore: Fine-grained evaluations of hallucinations in large vision-language models. In Findings ofthe Associationfor Computational Linguistics:

EMNLP 2024, pages 5042–5063. Association for Computational Linguistics.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. 2024. Prometheus: Inducing finegrained evaluation capability in language models. In The Twelfth International Conference on Learning Representations.

Seongyun Lee, Seungone Kim, Sue Park, Geewook Kim, and Minjoon Seo. 2024a. Prometheus-vision: Vision-language model as a judge for fine-grained evaluation. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 11286–11315. Association for Computational Linguistics.

Tony Lee, Haoqin Tu, Chi Heem Wong, Wenhao Zheng, Yiyang Zhou, Yifan Mai, Josselin Somerville Roberts, Michihiro Yasunaga, Huaxiu Yao, Cihang Xie, and Percy Liang. 2024b. VHELM: A holistic evaluation of vision language models. In Advances in Neural Information Processing Systems 37, pages 140632–140666. Neural Information Processing Systems Foundation, Inc.

Youngwan Lee, Kangsan Kim, Kwanyong Park, Ilchae Jung, Soojin Jang, Seanie Lee, Yong-Ju Lee, and Sung Ju Hwang. 2026. HoliSafe: Holistic safety benchmarking and modeling for vision-language model. In Findings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2025a. LLaVA-OneVision: Easy visual task transfer. Transactions on Machine Learning Research.

Xirui Li, Hengguang Zhou, Ruochen Wang, Tianyi Zhou, Minhao Cheng, and Cho-Jui Hsieh. 2025b. Is your multimodal language model oversensitive to safe queries? In International Conference on Learning Representations, volume 2025, pages 32517–32568.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292– 305. Association for Computational Linguistics.

Lequan Lin, Dai Shi, Andi Han, Feng Chen, Qiuzheng Chen, Jiawen Li, Zhaoyang Li, Jiyuan Zhang, Zhenbang Sun, and Junbin Gao. 2025. ACT as human: Multimodal large language model data annotation with critical thinking. In Advances in Neural Information Processing Systems 38, pages 24107–24138. Curran Associates, Inc.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. 2024a. MM-SafetyBench: A benchmark for safety evaluation of multimodal large language models. In Computer Vision – ECCV 2024, pages 386–403. Springer Nature Switzerland.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522. Association for Computational Linguistics.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2024b. MMBench: Is your multi-modal model an all-around player? In Computer Vision – ECCV 2024, pages 216–233. Springer Nature Switzerland.

Yuxuan Liu, Yuntian Shi, Kun Wang, Haoting Shen, and Kun Yang. 2026. CSR-Bench: A benchmark for evaluating the cross-modal safety and reliability of MLLMs. Preprint, arXiv:2602.03263.

Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, and Chaowei Xiao. 2024. JailBreakV: A benchmark for assessing the robustness of multimodal large language models against jailbreak attacks. In First Conference on Language Modeling.

Yiwei Ma, Jiayi Ji, Ke Ye, Weihuang Lin, Zhibin Wang, Yonghan Zheng, Qiang Zhou, Xiaoshuai Sun, and Rongrong Ji. 2024. I2EBench: A comprehensive benchmark for instruction-based image editing. In Advances in Neural Information Processing Systems 37, pages 41494–41516. Curran Associates, Inc.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 2263– 2279. Association for Computational Linguistics.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. DocVQA: A dataset for VQA on document images. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2200–2209.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. 2024. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 35181–35224. PMLR.

Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. 2021. Counterfactual VQA: A cause-efect look at language bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12700–12710.

Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, and Prateek Mittal. 2024. Visual adversarial examples jailbreak aligned large

language models. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(19):21527–21536.

Kaixuan Ren, Preslav Nakov, and Usman Naseem. 2025. DUAL-Bench: Measuring over-refusal and robustness in vision-language models. Preprint, arXiv:2510.10846.

Xuankun Rong, Wenke Huang, Tingfeng Wang, Daiguo Zhou, Bo Du, and Mang Ye. 2026. SafeGRPO: Self-rewarded multimodal safety alignment via rulegoverned policy optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7901–7911.

Leo Schwinn, Moritz Ladenburger, Tim Beyer, Mehrnaz Mofakhami, Gauthier Gidel, and Stephan Günnemann. 2026. A coin flip for safety: LLM judges fail to reliably measure adversarial robustness. In Proceedings ofthe 43rd International Conference on Machine Learning.

Erfan Shayegani, Yue Dong, and Nael Abu-Ghazaleh. 2024. Jailbreak in pieces: Compositional adversarial attacks on multi-modal language models. In The Twelfth International Conference on Learning Representations.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards VQA models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8317–8326.

Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. 2024. A StrongREJECT for empty jailbreaks. In Advances in Neural Information Processing Systems 37, pages 125416–125440. Neural Information Processing Systems Foundation, Inc.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chainof-thought prompting. In Advances in Neural Information Processing Systems 36, pages 74952–74965. Curran Associates, Inc.

Hanqing Wang, Yuan Tian, Mingyu Liu, Zhenhao Zhang, and Xiangyang Zhu. 2026. SDEval: Safety dynamic evaluation for multimodal large language models. Proceedings of the AAAI Conference on Artificial Intelligence, 40(39):33449–33457.

Tianyi Xiong, Yi Ge, Ming Li, Zuolong Zhang, Pranav Kulkarni, Kaishen Wang, Qi He, Zeying Zhu, Chenxi Liu, Ruibo Chen, Tong Zheng, Yanshuo Chen, Xiyao Wang, Renrui Zhang, Wenhu Chen, and Heng Huang. 2026. Multi-Crit: Benchmarking multimodal judges on pluralistic criteria-following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8641–8652.

Zonghao Ying, Aishan Liu, Siyuan Liang, Lei Huang, Jinyang Guo, Wenbo Zhou, Xianglong Liu, and Dacheng Tao. 2026. SafeBench: A safety evaluation framework for multimodal large language models. International Journal ofComputer Vision, 134(1):18.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2024. MM-vet: Evaluating large multimodal models for integrated capabilities. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 57730–57754. PMLR.

Ce Zhang, Jinxi He, Junyi He, Katia Sycara, and Yaqi Xie. 2026. Evolving contextual safety in multi-modal large language models via inference-time self-reflective memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 41182–41192.

Xinran Zhang. 2026. How sensitive are safety benchmarks to judge configuration choices? In Advanced Intelligent Computing Technology and Applications, pages 173–184. Springer Nature Singapore.

Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, and Jun Zhu. 2024. MultiTrust: A comprehensive benchmark towards trustworthy multimodal large language models. In Advances in Neural Information Processing Systems 37, pages 49279–49383. Neural Information Processing Systems Foundation, Inc.

Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, Feng Zhao, Tao Gui, and Jing Shao. 2025. SPA-VL: A comprehensive safety preference alignment dataset for vision language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19867–19878.

Baolin Zheng, Guanlin Chen, Qingyang Teng, Hongqiong Zhong, Yingshui Tan, Zhendong Liu, Weixun Wang, Jiaheng Liu, Jian Yang, Huiyun Jing, Jincheng Wei, Wenbo Su, Xiaoyong Zhu, Bo Zheng, and Kaifu Zhang. 2026. USB: A comprehensive and unified safety evaluation benchmark for multimodal large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21184– 21211. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems 36, pages 46595–46623. Neural Information Processing Systems Foundation, Inc.

Han Zhu, Juntao Dai, Jiaming Ji, Haoran Li, Chengkun Cai, Pengcheng Wen, Chi-Min Chan, Boyuan Chen,

![](images/7f1147a19cbc90faffc1da4b335d063639a3b896c2711d0881ffd3658bd975cb.jpg)  
Figure 3: Natural Response Prompt.

Yaodong Yang, Sirui Han, and Yike Guo. 2026. SafeMT: Multi-turn safety for multimodal language models. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 41381–41409. Association for Computational Linguistics.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, and 32 others. 2025. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. Preprint, arXiv:2504.10479.

Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. 2024. Safety finetuning at (Almost) no cost: A baseline for vision large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 62867–62891. PMLR.

## A Prompt Templates

This appendix summarizes the prompts used by the formal experiment. The exact prompt files are included with the experiment package. The taskcritical content and schemas are reproduced here. These prompts are evaluation instructions rather than benchmark examples. They do not contain API credentials, private data, raw harmful records, or operational misuse procedures. All prompts instruct models not to provide operational harmful details.

## A.1 Natural Response Prompt

## A.2 Evidence Reporting Prompt

The prompt requires the model to choose one of the four decision labels even when evidence is ambiguous. Ambiguity is represented through risk\_source=ambiguous\_or\_insufficient. If the risk comes from visible text, the model is asked to report only the relevant visible words needed for OCR identification.

![](images/cb8b46af9e1cd206b78c94cb67807f8831a96f17a806a63c1886555b1c10d35d.jpg)  
Figure 4: Evidence Reporting Prompt.

Response-policy guidance must appear only in safe\_response\_policy, not in the cross-modal relation.

## A.3 Judge Prompt

The judge rubric is run with GPT-5.4 and uses severity-aware decision correctness, risk-sourcefamily correctness, and semantic evidence scoring. The post-processing script recomputes severity correctness fromjudge decisions and gold labels before reporting final metrics.

## B Formal Experiment Protocol Details

## B.1 Formal Input Files

The formal experiment uses the OSS-URL view at data/final\_release/evisafebench\_v0/ gold/evisafebench\_gold\_v0.oss\_urls. jsonl in the code package. It contains 1,181 scenarios, 2,452 counterfactuals, and 3,633 image references expressed as HTTPS URLs. Its schema matches the canonical local-path gold JSONL.

The pilot subset is a stratified 96-scenario subset selected with seed 23 from the gold split. It contains 191 counterfactual variants and is used only to tune prompts, API retry behavior, and analysis scripts. Pilot results are not treated as formal paper results.

![](images/d28b2fdae13855a9086225161088b1c689a6d5990797321b5e2bcce9ef0a7a16.jpg)  
Figure 5: Judge Prompt.

## B.2 Request Generation

For each scenario, the formal runner creates one natural\_original request, one evidence\_report request, and one counterfactual\_natural request per counterfactual. Therefore each full attempted run contains 4,814 victim requests. After victim normalization, the runner can create up to one judge request per original scenario, for at most 1,181 judge requests.

Each scenario-level judge request is created only after the required victim outputs are available: the natural response, the evidence-report output, and the counterfactual responses for that scenario. This ensures that the judge compares a complete set of model outputs against the corresponding gold annotation rather than including scenario-model records with missing outputs. The reported N for each model is the resulting number of parsed scenario-level judge rows.

## B.3 Model Serving and API Access

The formal evaluation covers eleven VLMs. Seven open-source victim models are served locally with vLLM through an OpenAI-compatible endpoint.

For these local runs, the experiment runner starts one vLLM server at a time, waits until /v1/models is ready, runs the full protocol, stops the server, and then moves to the next model. Per-model configuration specifies the local model directory, served model name, tensor-parallel size, maximum model length, worker count, and optional vLLM arguments.

Four additional models are evaluated through OpenAI-compatible API endpoints: Gemini 2.5 Pro, Claude Opus 4.6, Doubao-Seed-1.6-Vision, and Qwen3-VL-235B-A22B-Instruct. These runs use the same generated request files, prompt templates, normalization logic, and evidenceaware judge protocol as the local runs, with only provider-specific endpoint, authentication, and chatformatting details handled by the runner. The configured formal suite is shown in Table 4.

## B.4 Decoding and Prompting

Victim decoding uses temperature 0. The local vLLM suite and default API settings use 512 maximum tokens for natural-response and counterfactual-response probes and 900 maximum tokens for evidence-report probes. Gemini 2.5 Pro uses 4096 and 8192 tokens, respectively, to accommodate provider-side reasoning tokens and avoid empty assistant responses. Natural and counterfactual probes use the same minimal helpful-assistant instruction, while the evidence-report probe uses a diagnostic JSON instruction and explicitly forbids answering the user’s original request. This keeps user-facing behavior and diagnostic evidence reporting separate.

The judge uses GPT-5.4 at temperature 0 and one evidence-aware request per parsed original scenario. The judge receives the gold annotation, the model’s natural response, its evidence-report output, and all counterfactual responses for that scenario. It returns decision labels, risk-source-family judgments, graded evidence scores, counterfactual behavior judgments, failure modes, and short nonoperational notes.

## B.5 Judge Validation

The judge is external to the evaluated model set and does not receive the original image. It compares output text with the compact gold annotation under a fixed rubric. A pilot audit of 100 outputs checked rubric adherence. A subsequent audit sampled 200 scenario-model records, stratified across eight safety domains and six scored risk-source families. Two annotators, blinded to model identity and judge outputs, independently labeled the natural decision, risk-source family, and four evidence fields. Disagreements were resolved by discussion.

<table><tr><td>Judgment</td><td>Agreement</td><td>Cohen&#x27;s κ</td></tr><tr><td>Natural decision</td><td>92.5%</td><td>0.89</td></tr><tr><td>Risk-source family</td><td>88.0%</td><td>0.83</td></tr><tr><td>Evidence fields</td><td>84.3%</td><td>0.78</td></tr></table>

Table 3: Agreement between the GPT-5.4 judge and adjudicated human labels on 200 stratified scenariomodel records.

## B.6 Retry and Normalization

The API runner supports worker concurrency, timeout control, multiple API-key environment variables, and retries with backof. Retryable provider failures are written into a new request JSONL, rerun once, and merged back into the final raw output. Non-retryable errors such as content-inspection failures, disabled models, context-window failures, or authentication/configuration errors are not repeatedly retried.

The runner can be configured to stop under strict failure mode. In the reported suite, unresolved provider failures are retained in failure files and reflected in the judged sample size for both local and API runs. Normalization extracts assistant text from OpenAI-compatible responses and records extraction or API errors without printing sensitive sample content. Scenario-level judge rows are counted only when the required victim outputs for that scenario can be parsed and scored.

## B.7 Metric Formulas

Let N be the number of parsed judge rows. Natural severity accuracy and diagnostic severity accuracy are:

$$
\begin{array} { l } { \displaystyle \mathrm { N S A } = \frac { 1 } { N } \sum _ { i } { \nmid }  { \boldsymbol { \kappa } } [ \mathrm { s e v } ( \hat { y } _ { i } ^ { \mathrm { n a t } } , y _ { i } ) = 1 ] , } \\ { \displaystyle \mathrm { D S A } = \frac { 1 } { N } \sum _ { i } { \nmid }  { \boldsymbol { \kappa } } [ \mathrm { s e v } ( \hat { y } _ { i } ^ { \mathrm { d i a g } } , y _ { i } ) = 1 ] . } \end{array}
$$

Let U {i : y<sub>i</sub> ∈ {unsafe\_warn, unsafe\_refuse}} and let $g ( \cdot )$ map four-way decisions into safe/unsafe families. Unsafe compliance rate is:

$$
\mathrm { U C R } = \frac { 1 } { | \mathcal { U } | } \sum _ { i \in \mathcal { U } } \mathcal { H } [ g ( \hat { y } _ { i } ^ { \mathrm { n a t } } ) = \mathrm { s a f e } ] .
$$

Risk-source family accuracy is

$$
{ \mathrm { R S F A } } = { \frac { 1 } { N } } \sum _ { i } \mathcal { H } [ f ( \hat { s } _ { i } ) = f ( s _ { i } ) ] ,
$$

where f maps eight risk-source labels into the six families described in Section 3.2.

Evidence score is

$$
\mathrm { E S } = \frac { 1 } { 4 N } \sum _ { i } \sum _ { k \in \{ T , I , C , P \} } v ( e _ { i k } ) ,
$$

where T, I, C, and P correspond to text evidence, visual evidence, cross-modal relation, and safe-response policy. The scoring function is defined as v(correct) = 1, v(partial) = 0.5, and $v ( { \mathrm { i n c o r r e c t } } ) = 0 .$

Strict diagnostic consistency requires NSA success, RSFA success, and all four evidence fields correct. Relaxed diagnostic consistency requires NSA success, RSFA success, all evidence fields known, nonzero cross-modal relation score, at least three nonzero evidence fields, and average evidence score at least 0.5. The judge also assigns each row one mutually exclusive decision-evidence label. Evidence-grounded success requires a correct natural decision and mostly correct evidence, while heuristic success retains a correct natural decision despite an important evidence or attribution error. The other labels are policy failure, complete failure, over-refusal, and under-refusal. Counterfactual behavior accuracy is the fraction of counterfactual responses whose severity matches the counterfactual expected decision.

Let ${ \mathcal { T } } _ { U  S }$ contain the gold pairs whose original unsafe evidence is removed or neutralized. Let $a _ { j } = \nVdash [ o _ { j } = \mathrm { u n s a f e } ]$ and $b _ { j } = \nVdash [ c _ { j } = \mathrm { s a f e } ]$ where $o _ { j }$ and $c _ { j }$ denote the model decision families for the original and counterfactual responses. The joint transition metric is

$$
\mathrm { U 2 S } = \frac { 1 } { | \mathcal { T } _ { U  S } | } \sum _ { j \in \mathcal { T } _ { U  S } } a _ { j } b _ { j } .
$$

Its two factors are original unsafe recognition on this subset and conditional adaptation:

$$
\begin{array} { l } { \mathrm { O r i g U } = \displaystyle \frac { 1 } { | \mathcal T _ { U \to S } | } \sum _ { j \in \mathcal T _ { U \to S } } a _ { j } , } \\ { \displaystyle \mathrm { c U 2 S } = \frac { \sum _ { j \in \mathcal T _ { U \to S } } a _ { j } b _ { j } } { \sum _ { j \in \mathcal T _ { U \to S } } a _ { j } } . } \end{array}
$$

Therefore, $\mathrm { U 2 S = O r i g U \times c U 2 S . }$ cU2S is diagnostic rather than a standalone ranking metric because conditioning can hide poor recognition of the original risk. Preservation accuracy requires both the original and counterfactual decision families to match the gold families on safe-to-safe and unsafe-to-unsafe pairs. These variants also contribute to CBA.

## C Additional Results and Data Construction Details

## C.1 Raw Public Pool

The raw pool is built from public VLM/MLLM safety datasets. Unsafe and contextual-risk records come from MM-SafetyBench, MM-SafetyBench++, and VLSBench (Liu et al., 2024a; Zhang et al., 2026; Hu et al., 2025). Preference and alignment sources include BeaverTails-V, SPA-VL, and SafeTag-VL-3K (Ji et al., 2025; Zhang et al., 2025; Rong et al., 2026). MOSSBench supplies benign-sensitive over-refusal controls (Li et al., 2025b). Each source row is preserved in a normalized JSONL record with source dataset, source split/subset, source identifier, project-level risk domain, source category, safety label when available, user text, image path, and raw metadata. Images are stored locally and referenced by relative paths. Records that could not be confidently mapped into a project domain were excluded from the final raw pool. Table 7 and Table 8 summarize the raw pool.

## C.2 Step 1: Natural-Response Expected Decision

The raw datasets use heterogeneous labels. Some provide safe/unsafe response labels, some provide preference-style examples, and many contain no direct label compatible with the four-way decision schema. Each record is therefore relabeled into:

• safe\_answer: the assistant may directly answer.

• safe\_answer\_with\_caution: the assistant may answer, but should include safety caveats or avoid unsupported certainty.

• unsafe\_warn: the assistant should warn, deescalate, or redirect without giving operationally harmful content.

• unsafe\_refuse: the assistant should refuse the harmful request.

Step 1 uses three signals. First, we map source labels whenever their semantics are reliable. For example, response-safety labels in BeaverTails-V are used as prior evidence. Second, GPT-5.4 labels each image-text pair with one of the four decisions and a confidence estimate. Third, an independent Qwen-VL label pass provides an agreement signal for conflict resolution. It is not a second evidence generator. If the source label and model labels agree, the label is accepted automatically. If labels conflict or confidence is low, the record enters a manual-adjudication pool. Manual adjudication is performed only on the conflict subset, while records that remain unknown are removed.

This procedure removes 302 records and leaves 9,097 records with deterministic labels. Table 10 reports the resulting label distribution.

## C.3 Step 2: Evidence-Report Annotation

Conditioned on the image, user text, and finalized expected decision, GPT-5.4 generates a structured evidence report. It contains text\_evidence, visual\_evidence, cross\_modal\_relation, safe\_response\_policy, and one of the eight risk-source labels in Table 11.

The annotation prompt requires the cross-modal relation to explicitly state whether the text alone is safe or unsafe, whether the image alone is safe or unsafe, and what changes when the two are combined. This field guides the risk-source label. The safe-response policy is kept separate from the relation so that evidence description does not drift into policy instruction.

## C.4 Step 3: Evidence-Targeted Counterfactual Construction

Each evidence report determines one to three counterfactual specifications that alter the minimum evidence needed to test the decision. Every specification records an intervention type, the changed evidence, its purpose, and a new expected decision. Text-only interventions directly revise the user text. Image-side interventions preserve the original image path as a placeholder until the specified visual change is instantiated after gold selection.

The annotation pipeline produces the Step 2 report and Step 3 specifications in one structured multimodal output. Their logical order remains explicit because the allowed intervention type is selected from the risk source and the changedevidence field must identify which annotated cue is altered.

Validity filtering. Checks cover image access, legibility of decision-relevant OCR, schema validity, and at least one usable counterfactual. After these checks, 8,689 of the 9,097 Step 1 records remain in the full pool. A deterministic safetyfocused, domain-stratified sample of 1,500 Step 1 records yields 1,488 evidence-annotated gold candidates after the same checks. The 8,689 count is the pre-adjudication pool. The 8,670 count in Table 1 is obtained after the 19 removals during focused adjudication described below.

A. Diagnostic profile  
C. UCR bridge vs. evidence grounding
<table><tr><td>Model</td><td>Family</td><td>Serving mode</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>Qwen-VL</td><td>local vLLM, 8192 context</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>Qwen-VL</td><td>local vLLM, 12288 context</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>Qwen-VL</td><td>local vLLM, 12288 context</td></tr><tr><td>InternVL3-8B</td><td>InternVL</td><td>local vLLM, 8192 context</td></tr><tr><td>InternVL3-14B</td><td>InternVL</td><td>local vLLM, 8192 context</td></tr><tr><td>LLaVA-OneVision-Qwen2-7B</td><td>LLaVA</td><td>local vLLM, 8192 context</td></tr><tr><td>Gemma-3-12B-IT</td><td>Gemma-Vision</td><td>local vLLM, 8192 context</td></tr><tr><td>Gemini 2.5 Pro</td><td>Gemini</td><td>OpenAI-compatible API</td></tr><tr><td>Claude Opus 4.6</td><td>Claude</td><td>OpenAI-compatible API</td></tr><tr><td>Doubao-Seed-1.6-Vision</td><td>Doubao</td><td>OpenAI-compatible API</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>Qwen-VL</td><td>OpenAI-compatible API</td></tr></table>

Table 4: Formal model suite. All models are evaluated with the full protocol. Seven open-source models are served locally through vLLM, and four additional models are evaluated through OpenAI-compatible API endpoints using the same request generation, normalization, and judging protocol.

<table><tr><td rowspan=1 colspan=6> $\ N ^ { S ^ { P } }$      $\downarrow ^ { \downarrow ^ { \circ } }$     $o ^ { S ^ { R } }$      $a S ^ { 8 ^ { 4 ^ { n } } }$    ES</td><td rowspan=1 colspan=2> $a s ^ { C }$      $\ o ^ { 5 }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-7B</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>40</td></tr><tr><td rowspan=1 colspan=1>Qwen3-4B</td><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>58</td></tr><tr><td rowspan=1 colspan=1>Qwen3-8B</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>66</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>55</td></tr><tr><td rowspan=1 colspan=1>InternVL3-8B</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>33</td></tr><tr><td rowspan=1 colspan=1>InternVL3-14B</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>35</td></tr><tr><td rowspan=1 colspan=1>LLaVA-OV-7B</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>Gemma3-12B</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>47</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>49</td></tr></table>

B. Evidence usability
<table><tr><td rowspan=1 colspan=5>Text       Visual      Relation      $\boldsymbol { \mathfrak { e } } ^ { \tilde { \infty } ^ { \tilde { \mathcal { A } } ^ { \tilde { C } ^ { 1 } } } }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-7B</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>62</td></tr><tr><td rowspan=1 colspan=1>Qwen3-4B</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>78</td></tr><tr><td rowspan=1 colspan=1>Qwen3-8B</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>68</td></tr><tr><td rowspan=1 colspan=1>InternVL3-8B</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>63</td></tr><tr><td rowspan=1 colspan=1>InternVL3-14B</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=1>73</td></tr><tr><td rowspan=1 colspan=1>LLaVA-OV-7B</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>Gemma3-12B</td><td rowspan=1 colspan=1>99</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>60</td></tr></table>

![](images/f76c3e3479585016c647be74207ffc0a77270bd26b22a3774c6e78e7233828c6.jpg)

![](images/9e7cbacdc057629680f4c624597f089aef97070cfbf836286f43d1901ac95a6d.jpg)  
Figure 6: Overview of local open-source model results. The panels show that final-answer safety, evidence localization, risk-source attribution, diagnostic consistency, and counterfactual sensitivity induce diferent mode profiles. The 1-UCR scale is inverted so darker cells indicate lower unsafe compliance.

## C.5 Gold Pool Selection

The gold pool is selected from the 1,488 evidenceannotated gold candidates. The selection policy is designed to satisfy five constraints:

• unsafe cases should dominate because the benchmark targets safety and jailbreak-relevant behavior.

• all four expected-decision labels must remain represented.

• all eight safety domains and all eight risk-source categories must be covered.

• rare risk-source types should be retained rather than washed out by random sampling.

• every item marked needs\_manual\_review=true must be included.

The initial 1,200-scenario selection contained 911 unsafe-label scenarios and 289 safe-label scenarios, including 516 dificult cases from Step 1 disagreement or Step 2 quality flags. A local adjudication interface was then used to inspect all 64 records marked needs\_manual\_review=true. Manual adjudication retained 45 records after decision-label correction and removed 19 low-quality records from both the gold pool and the full candidate pool. These removals reduce the two pools from 1,200 to 1,181 and from 8,689 to 8,670, respectively. The finalized gold release contains 849 unsafe-label scenarios and 332 safe-label scenarios, with 497 hard cases and 684 stable or easy cases. The 1,136 non-flagged records are retained as automatically accepted cases, with 23 non-flagged records spotchecked before the focused adjudication queue was created.

<table><tr><td>Model</td><td>U2S</td><td>OrigU</td><td>cU2S</td><td>Pres.</td></tr><tr><td>Qwen2.5-VL-7B</td><td>39.9</td><td>42.5</td><td>93.9</td><td>78.6</td></tr><tr><td>Qwen3-VL-4B</td><td>58.0</td><td>66.7</td><td>87.0</td><td>76.0</td></tr><tr><td>Qwen3-VL-8B</td><td>55.2</td><td>64.5</td><td>85.5</td><td>77.1</td></tr><tr><td>InternVL3-8B</td><td>33.4</td><td>35.3</td><td>94.6</td><td>81.4</td></tr><tr><td>InternVL3-14B</td><td>35.2</td><td>37.3</td><td>94.4</td><td>80.8</td></tr><tr><td>LLaVA-OneVision-Qwen2-7B</td><td>30.4</td><td>31.5</td><td>96.6</td><td>77.2</td></tr><tr><td>Gemma-3-12B-IT</td><td>48.9</td><td>55.7</td><td>87.7</td><td>76.4</td></tr><tr><td>Gemini 2.5 Pro</td><td>41.0</td><td>44.1</td><td>93.0</td><td>81.0</td></tr><tr><td>Claude Opus 4.6</td><td>58.4</td><td>77.9</td><td>74.9</td><td>62.4</td></tr><tr><td>Doubao-Seed-1.6-Vision</td><td>35.6</td><td>37.0</td><td>96.4</td><td>76.7</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>55.2</td><td>59.1</td><td>93.4</td><td>81.8</td></tr></table>

Table 5: Counterfactual transition decomposition. OrigU is the percentage of originals assigned to the unsafe family within gold unsafe-to-safe pairs. cU2S is the percentage of recognized unsafe originals whose paired counterfactual response shifts to the safe family. Pres. is joint family accuracy on decision-preserving pairs. All values are percentages, and U2S equals OrigU multiplied by cU2S before rounding.

<table><tr><td>Counterfactual type</td><td>Count</td></tr><tr><td>remove_key_text_evidence</td><td>539</td></tr><tr><td>neutralize_text_intent</td><td>434</td></tr><tr><td>replace_ocr_with_benign_text</td><td>345</td></tr><tr><td>remove_benign_context_cue</td><td>279</td></tr><tr><td>strengthen_benign_intent</td><td>247</td></tr><tr><td>replace_key_visual_object</td><td>160</td></tr><tr><td>replace_risky_scene_with_benign_scene</td><td>149</td></tr><tr><td>weaken_scene_risk_context</td><td>104</td></tr><tr><td>change_noncritical_visual_context</td><td>63</td></tr><tr><td>clarify_prevention_or_opposition_goal</td><td>50</td></tr><tr><td>add_sensitive_surface_cue</td><td>40</td></tr><tr><td>remove_ocr_risk_signal</td><td>36</td></tr><tr><td>remove_key_visual_object</td><td>6</td></tr></table>

Table 17: Counterfactual intervention types in the gold pool. Text-only counterfactuals modify user text, while visual counterfactuals specify image-edit targets.

Aggregating the 13 intervention types in Table 17 gives 1,549 text, intent, or benign-context variants (63.2%), 381 OCR variants (15.5%), 166 localized object variants (6.8%), 316 scene or context variants (12.9%), and 40 other variants (1.6%). Thus at least 85.5% avoid whole-scene semantic editing. This distribution is not a direct validity rate. It identifies scene-level edits as the main artifactsensitive subset.

## C.6 Visual Counterfactual Instantiation

This final phase of Step 3 materializes each retained image-side specification by editing the original image according to changed\_evidence. The finalized gold split contains 903 visual counterfactuals, all paired with generated images. Images not associated with retained gold records are excluded from the benchmark. The canonical gold JSONL stores local project-relative image\_path values rather than OSS URLs, so experimental scripts resolve both original and generated counterfactual images under the local dataset root.

The full candidate pool is retained for auxiliary analysis, but only the 903 visual counterfactuals belonging to the gold split have finalized generatedimage paths. For this reason, all primary experiments in the paper use the 1,181-scenario gold release rather than the full candidate pool.

## C.7 Benchmark Release Files

The clean experimental release is organized as a gold split and an auxiliary full-candidate split:

• evisafebench\_gold\_v0.jsonl: canonical file for all main experiments, containing 1,181 original scenarios and 2,452 counterfactuals.

• The gold image-status copy: same scenarios, with generated\_image\_status added to the 903 visual counterfactuals whose generated images are available.

• evisafebench\_full\_candidate\_v0.jsonl: auxiliary 8,670-scenario candidate pool. It is not treated as the fully visualized benchmark because non-gold visual counterfactuals may still use placeholder image paths.

All image\_path fields in the benchmark release are local project-relative paths rather than public OSS URLs. Original images use raw\_public/images/. Generated counterfactual images use the local generated\_images/ directory. If an external batch-inference platform requires public URLs, a separate OSS-URL JSONL view is generated from this canonical local-path release rather than overwriting the gold files.

<table><tr><td>Model</td><td>Text corr.</td><td>Text part.</td><td>Vis corr.</td><td>Vis part.</td><td>Rel corr.</td><td>Rel part.</td><td>Policy corr.</td><td>Policy part.</td></tr><tr><td>Qwen2.5-VL-7B</td><td>21.2</td><td>66.9</td><td>38.5</td><td>35.1</td><td>9.6</td><td>40.9</td><td>25.9</td><td>35.7</td></tr><tr><td>Qwen3-VL-4B</td><td>44.1</td><td>47.5</td><td>54.6</td><td>38.5</td><td>10.9</td><td>57.4</td><td>40.9</td><td>37.1</td></tr><tr><td>Qwen3-VL-8B</td><td>63.1</td><td>35.3</td><td>54.0</td><td>38.3</td><td>18.1</td><td>53.9</td><td>35.7</td><td>31.9</td></tr><tr><td>InternVL3-8B</td><td>36.6</td><td>57.6</td><td>47.7</td><td>43.9</td><td>13.6</td><td>35.9</td><td>29.2</td><td>34.2</td></tr><tr><td>InternVL3-14B</td><td>26.7</td><td>34.1</td><td>56.5</td><td>39.1</td><td>22.9</td><td>51.6</td><td>37.6</td><td>35.7</td></tr><tr><td>LLaVA-OneVision-Qwen2-7B</td><td>11.9</td><td>61.6</td><td>4.1</td><td>21.7</td><td>2.2</td><td>10.8</td><td>8.0</td><td>24.3</td></tr><tr><td>Gemma-3-12B-IT</td><td>68.4</td><td>30.4</td><td>58.4</td><td>38.3</td><td>37.2</td><td>43.0</td><td>35.0</td><td>24.6</td></tr><tr><td>Gemini 2.5 Pro</td><td>73.5</td><td>24.8</td><td>71.1</td><td>26.0</td><td>29.5</td><td>46.1</td><td>35.2</td><td>37.4</td></tr><tr><td>Claude Opus 4.6</td><td>82.3</td><td>16.7</td><td>79.3</td><td>19.2</td><td>38.0</td><td>42.0</td><td>46.1</td><td>31.2</td></tr><tr><td>Doubao-Seed-1.6-Vision</td><td>70.5</td><td>29.2</td><td>72.4</td><td>25.0</td><td>28.5</td><td>45.3</td><td>44.2</td><td>28.9</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>73.5</td><td>24.4</td><td>66.8</td><td>31.1</td><td>29.3</td><td>52.7</td><td>46.2</td><td>35.6</td></tr><tr><td>Model mean</td><td>52.0</td><td>38.9</td><td>54.9</td><td>32.4</td><td>21.8</td><td>43.6</td><td>34.9</td><td>32.4</td></tr></table>

Table 6: Evidence-field breakdown for all eleven runs. Columns give correct and partial judge rates. Incorrect is the remainder, and the model mean is unweighted. Text and visual evidence are more recoverable than cross-modal relation and policy binding.

<table><tr><td>Source dataset</td><td>Records</td></tr><tr><td>PKU-Alignment/MM-SafetyBench EchoSafe-MLLM/MM-SafetyBench++ Foreshhh/VLSBench</td><td>2,587 2,055 1,479</td></tr><tr><td>PKU-Alignment/BeaverTails-V sqrti/SPA-VL</td><td>1,242 1,212</td></tr><tr><td>XuankunRong/SafeTag-VL-3K xirui-li/MOSSBench Total</td><td>524 300</td></tr></table>

Table 7: Raw public pool by source dataset. MOSS-Bench contributes the benign-sensitive records in the normalized pool.
<table><tr><td>Project-level domain</td><td>Records</td></tr><tr><td>physical_harm_hazards</td><td>1,500</td></tr><tr><td>fraud_illegal_economic hate_harassment_discrimination</td><td>1,500 1,500</td></tr><tr><td>health_selfharm_psychological high_stakes_misinfo_governance</td><td>1,500 1,500</td></tr><tr><td>sexual_exploitation cyber_privacy_surveillance</td><td>829 770</td></tr><tr><td>benign_sensitive_overrefusal Total</td><td>300</td></tr></table>

Table 8: Raw public pool by project-level safety domain. Gold selection improves domain coverage while retaining all eight domains.

## D Human Adjudication Interface

Human adjudication is performed with a local-only web interface. Each page displays one scenario at a time, including the image, original text, expected decision, evidence fields, risk source, counterfactuals, quality flags, and human notes. Annotators can edit any field and click Save & Next, which records the item as adjudicated and stores the edited row in an append-only change log. Annotators can also skip a low-quality record. Skipped records are removed from both the gold split and the full candidate pool during finalization. Among the 64 flagged records, 45 were retained after decisionlabel correction and 19 were skipped. The tool stores the change log, a merged JSONL file, and progress metadata, preserving intermediate work if the browser is closed before adjudication is finished.

## E Additional Notes on Data Safety

The data construction process separates researchuse artifacts from public-release artifacts. Raw downloaded records and images are treated as local research materials. Public reports should include only aggregate statistics, redacted examples, synthetic illustrations, and non-operational evidence descriptions. For potentially harmful categories, counterfactual descriptions specify what evidence changed without providing procedural details. The same principle applies to prompts used for annotation and evaluation: they ask models to classify or describe safety evidence, not to generate harmful instructions.

## F Schema and Redacted Example

The final gold schema is intentionally compact. Evidence fields are strings rather than dense structured annotations because the benchmark targets reportable evidence alignment at the object/scene level. A redacted example is shown in Figure 7. The example uses a benign household-safety abstraction rather than a verbatim record from the dataset.

<table><tr><td>Source</td><td>Phys.</td><td>Cyber</td><td>Fraud</td><td>Hate</td><td>Sexual</td><td>Health</td><td>High-stakes</td><td>Benign</td></tr><tr><td>MM-SafetyBench++</td><td>444</td><td>176</td><td>451</td><td>548</td><td>436</td><td>0</td><td>0</td><td>0</td></tr><tr><td>VLSBench</td><td>434</td><td>157</td><td>243</td><td>226</td><td>318</td><td>101</td><td>0</td><td>0</td></tr><tr><td>BeaverTails-V</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1,242</td><td>0</td><td>0</td></tr><tr><td>MM-SafetyBench</td><td>319</td><td>141</td><td>527</td><td>425</td><td>0</td><td>82</td><td>1,093</td><td>0</td></tr><tr><td>SafeTag-VL-3K</td><td>78</td><td>80</td><td>62</td><td>121</td><td>26</td><td>25</td><td>132</td><td>0</td></tr><tr><td>SPA-VL</td><td>225</td><td>216</td><td>217</td><td>180</td><td>49</td><td>50</td><td>275</td><td>0</td></tr><tr><td>MOSSBench</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>300</td></tr></table>

Table 9: Source-domain matrix for the raw public pool. Abbreviations are Phys. (physical harm and hazards), Cyber (cyber, privacy, and surveillance), Fraud (fraud, illegal, and economic harm), Hate (hate, harassment, and discrimination), Sexual (sexual exploitation), Health (health, self-harm, and psychological harm), High-stakes (high-stakes misinformation and governance), and Benign (benign-sensitive over-refusal).

```json
{
"scenario_id": "S001",
"domain": "physical_harm_hazards",
"risk_source": "cross_modal_object_text",
"original": {
"item_id": "S001_ORIG",
"image_path": "raw_public/images/.../S001.png",
"user_text": "[redacted image-grounded safety question]",
"expected_decision": "unsafe_warn"
},
"evidence": {
"text_evidence": "The user asks whether combining the visible items would improve the task.",
"visual_evidence": "The image contains two visible household products whose combination may be dangerous.",
"cross_modal_relation": "Text alone is ambiguous. The image alone shows potentially risky products. Combined with the
,→ user request, these products make the safety risk explicit.",
"safe_response_policy": "Warn against unsafe combination, avoid operational details, and redirect to reading labels
,→ or consulting official guidance."
},
"counterfactuals": [
{
"type": "remove_key_text_evidence",
"image_path": "raw_public/images/.../S001.png",
"user_text": "[redacted benign question about separate uses]",
"expected_decision": "safe_answer_with_caution",
"changed_evidence": "The combining intent is removed from the user text.",
"counterfactual_purpose": "Tests whether the unsafe decision depends on the text evidence."
}
],
"quality_flags": {
"needs_manual_review": false,
"review_reason": ""
}
}
```  
Figure 7: Redacted schematic example of the EviSafeBench gold schema. The dataset stores actual image paths and user text, but the paper avoids verbatim sensitive records.

<table><tr><td>Expected decision</td><td>Records</td></tr><tr><td>unsafe_warn</td><td>3,060</td></tr><tr><td>safe_answer_with_caution</td><td>2,753</td></tr><tr><td>unsafe_refuse safe_answer</td><td>2,529</td></tr><tr><td>Total</td><td>755 9,097</td></tr></table>

Table 10: Expected-decision distribution after Step 1 finalization.

<table><tr><td>Risk source</td><td>Definition</td><td>Typical counterfactual intervention</td></tr><tr><td>text_only</td><td>Text alone determines the safety decision.</td><td>Remove or neutralize key text evi- dence.</td></tr><tr><td>image_ocr</td><td>Image-embedded text determines the risk.</td><td>Replace or remove OCR phrase.</td></tr><tr><td>image_object</td><td>A visible object is the key risk evidence.</td><td>Replace or remove key visual ob- ject.</td></tr><tr><td>image_scene_semantic</td><td>The overall scene/action carries the risk.</td><td>Replace risky scene or weaken scene context.</td></tr><tr><td>cross_modal_ocr_text</td><td>OCR text and user text jointly establish intent.</td><td>Replace OCR text or neutralize user intent.</td></tr><tr><td>cross_modal_object_text</td><td>Visual objects and user text jointly establish intent.</td><td>Replace object or remove text rela- tion.</td></tr><tr><td>benign_sensitive_trap</td><td>Sensitive surface cues are resolved by benign con- text.</td><td>Remove benign cue or strengthen benign intent.</td></tr><tr><td>ambiguous_or_insufficient</td><td>Evidence is underspecified or visually uncertain.</td><td>Clarify benign/preventive goal or retain for manual adjudication.</td></tr></table>

Table 11: Risk-source taxonomy and its connection to counterfactual design.

<table><tr><td>Expected decision</td><td>Gold records</td></tr><tr><td>unsafe_warn</td><td>602</td></tr><tr><td>unsafe_refuse</td><td>309</td></tr><tr><td>safe_answer_with_caution</td><td>249</td></tr><tr><td>safe_answer</td><td>40</td></tr><tr><td>Total</td><td>1,200</td></tr></table>

Table 12: Initial gold pool expected-decision distribution before focused human adjudication. Unsafe labels account for 911 of 1,200 scenarios. The retained gold distribution is shown in Table 13.

<table><tr><td>Risk source</td><td>Gold records</td></tr><tr><td>cross_modal_ocr_text</td><td>296</td></tr><tr><td>benign_sensitive_trap</td><td>287</td></tr><tr><td>text_only cross_modal_object_text</td><td>238</td></tr><tr><td>image_scene_semantic</td><td>193 109</td></tr><tr><td>image_ocr</td><td>44</td></tr><tr><td>image_object</td><td>7</td></tr><tr><td>ambiguous_or_insufficient</td><td>7</td></tr><tr><td>Total</td><td>1,181</td></tr></table>

<table><tr><td>Expected decision</td><td>Final gold records</td></tr><tr><td>unsafe_warn</td><td>541</td></tr><tr><td>unsafe_refuse</td><td>308</td></tr><tr><td>safe_answer_with_caution</td><td>283</td></tr><tr><td>safe_answer</td><td>49</td></tr><tr><td>Total</td><td>1,181</td></tr></table>

Table 15: Final gold pool risk-source distribution. Rare categories are intentionally retained for coverage rather than downsampled away.  
Table 13: Retained gold expected-decision distribution after manual adjudication and low-quality-record removal. Unsafe labels account for 849 of 1,181 scenarios.

<table><tr><td>Domain</td><td>Gold records</td></tr><tr><td>cyber_privacy_surveillance physical_harm_hazards</td><td>164</td></tr><tr><td>sexual_exploitation health_selfharm_psychological</td><td>162 162</td></tr><tr><td>hate_harassment_discrimination high_stakes_misinfo_governance</td><td>162 161</td></tr><tr><td>fraud_illegal_economic benign_sensitive_overrefusal</td><td>161 160 49</td></tr></table>

Table 14: Final gold pool domain distribution. The benign-sensitive domain is smaller because the raw source pool contains fewer such records, but it is fully retained for over-refusal analysis.

<table><tr><td>Counterfactuals per scenario</td><td>Scenarios</td></tr><tr><td>1</td><td>8</td></tr><tr><td>2 3</td><td>1,075 98</td></tr><tr><td></td><td></td></tr><tr><td>Total scenarios</td><td>1,181</td></tr><tr><td>Total counterfactual variants</td><td>2,452</td></tr></table>

Table 16: Number of counterfactual variants per gold scenario.