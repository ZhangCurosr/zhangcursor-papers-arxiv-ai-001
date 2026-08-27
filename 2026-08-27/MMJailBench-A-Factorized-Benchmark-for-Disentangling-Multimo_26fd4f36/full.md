# MMJailBench: A Factorized Benchmark for Disentangling Multimodal Jailbreak Vulnerabilities

Tianshi Wang<sup>1</sup>, Jingsong Wang<sup>1</sup>, Yafei Huang<sup>1</sup>, Fengling Li<sup>2</sup>, Xin Li<sup>3</sup>, Lei Zhu<sup>1∗</sup>

<sup>1</sup> Tongji University

<sup>2</sup> Mohamed bin Zayed University of Artificial Intelligence

<sup>3</sup> Shanghai Artificial Intelligence Laboratory

## Abstract

Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet how diferent factors shape their jailbreak vulnerabilities remains poorly understood. Existing benchmarks often couple harmful intent, prompt framing, visual semantics, and instruction carrier within individual jailbreak instances, obscuring the specific sources of observed vulnerabilities. To address this limitation, we introduce MM-JailBench, a factorized benchmark that systematically varies and combines these factors under controlled configurations, enabling fine-grained comparison and factor-level attribution. Large-scale evaluations across 16 open-weight and proprietary MLLMs reveal highly heterogeneous and model-dependent vulnerability profiles. Jailbreak vulnerability varies markedly across harm domains, exposing uneven coverage in current multimodal safety alignment. Prompt framing emerges as the dominant source of variation, task-relevant visual semantics systematically increase jailbreak susceptibility with authoritylike cues exposing particularly pronounced vulnerabilities, and visually rendered instructions do not consistently increase jailbreak susceptibility relative to direct textual instructions. To further investigate the risks introduced by multimodal context, we conduct diagnostic analyses on a representative open-weight model and identify vulnerability-associated patterns in internal representations and cross-modal interactions. Finally, we develop a modular multimodal jailbreak evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-eficient multimodal jailbreak auditing. Warning: This paper contains potentially harmful content.

## 1 Introduction

Multimodal Large Language Models (MLLMs) are expanding beyond image captioning and visual question answering into real-world applications such as document understanding, code generation, workflow assistance, and automated operations (Liu et al., 2023; Dai et al., 2023). Unlike text-only models, MLLMs must jointly interpret linguistic instructions and visual information when reasoning and generating responses. Harmful requests may therefore combine direct textual or visually rendered instructions (Gong et al., 2025) with contextual cues such as professional settings, identity credentials, or authorization documents. Consequently, a model’s refusal and compliance decisions no longer depend solely on an isolated user instruction, but are jointly shaped by the surrounding cross-modal context (Li et al., 2024b).

![](images/f7be09fa78195aca315170de1c591cbab54a14179f8a6930351fd0914d0ee572.jpg)  
Figure 1: Motivation of MMJailBench. Existing multimodal jailbreak benchmarks entangle multiple factors within fixed instances, limiting factor-level diagnosis. MMJailBench factorizes these factors and constructs a controlled evaluation space for fine-grained analysis.

However, how models form safety judgments in cross-modal contexts remains poorly understood. Visual information may not only supplement harmful intent, but also alter the perceived legitimacy, professionalism, and credibility of a request, causing the same underlying intent to elicit diferent refusal or compliance behaviors across contexts (Liu et al., 2024). This suggests that multimodal jailbreaks are not merely changes in input format. Rather, they expose the stability and consistency of safety alignment under cross-modal conditions (Luo et al., 2024). Understanding how such vulnerabilities arise is therefore essential for accurately evaluating and improving multimodal safety alignment.

Existing text and multimodal safety benchmarks cover a broad range of harmful behaviors (Li et al., 2024a; Mazeika et al., 2024; Chao et al., 2024), visual attacks, and image-based text injection (Liu et al., 2024; Gong et al., 2025; Luo et al., 2024), providing an important foundation for evaluating model robustness. However, these benchmarks typically construct fixed prompt-image pairs for individual harmful tasks, jointly encoding harmful intent, prompt framing, visual semantics, and instruction carrier within a single test instance. The resulting attack success rates primarily reflect a model’s aggregate vulnerability on a particular test set, but ofer limited insight into whether observed vulnerabilities originate from linguistic framing, visual context, the way instructions are conveyed, or weaknesses concentrated in particular harm domains (Weng et al., 2025; Jia et al., 2025). Such factor entanglement limits fine-grained comparison and obscures the diagnostic signals needed to better understand and improve multimodal safety alignment, as illustrated in Figure 1.

To address this limitation, we introduce MMJailBench, a factorized benchmark for disentangling the sources of multimodal jailbreak vulnerabilities. MMJailBench incorporates harmful intent, prompt framing, visual semantics, and instruction carrier into a unified controlled design, and uses systematic combinations and matched comparisons to examine how each factor shapes model refusal and compliance behavior. The benchmark contains 272 harmful intents, 6 prompt templates, 5 types of task-relevant visual semantics, and 2 instruction carriers, forming a structured and directly comparable evaluation space. This design enables MMJailBench not only to measure overall jailbreak vulnerability, but also to attribute observed vulnerabilities to specific factors. Building on this design, we further develop a modular multimodal jailbreak evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-eficient multimodal jailbreak auditing.

We conduct large-scale evaluations across 16 representative open-weight and proprietary MLLMs. The results reveal substantially uneven jailbreak vulnerabilities across harm domains, with cyber abuse, economic harm, privacy violations, and deception and manipulation exhibiting greater overall vulnerability. Prompt framing emerges as the dominant source of variation in jailbreak outcomes. Even when harmful intent remains unchanged, diferent narrative structures and output requirements can substantially alter refusal and compliance behavior. Task-relevant visual semantics systematically weaken refusal behavior, with authority-like cues such as authorization documents and identity credentials exposing particularly pronounced vulnerabilities. This suggests that models may overrely on the apparent legitimacy and credibility conveyed by visual context. The efect of instruction carrier is highly modeldependent, as visually rendered instructions do not consistently increase jailbreak susceptibility relative to direct textual instructions. Diagnostic analyses on a representative open-weight model further identify vulnerability-associated patterns in internal representations and cross-modal interactions.

Overall, our main contributions are as follows:

• We introduce MMJailBench, a factorized benchmark that incorporates harmful intent, prompt framing, visual semantics, and instruction carrier into a unified controlled design, enabling fine-grained analysis and factor-level attribution of multimodal jailbreak vulnerabilities.

• We conduct large-scale evaluations across 16 representative open-weight and proprietary MLLMs, revealing uneven jailbreak vulnerabilities across harm domains, the dominant role of prompt framing, the refusal-weakening efect of authority-like visual cues, and the modeldependent efect of instruction carriers. Diagnostic analyses on a representative open-weight model further identify vulnerability-associated patterns in internal representations and cross-modal interactions.

• We develop a modular multimodal jailbreak evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-eficient multimodal jail break auditing.

## 2 Related Work

## 2.1 Jailbreak Attacks and Defenses

Jailbreak attacks aim to circumvent safety alignment and induce models to assist with harmful requests (Wei et al., 2023; Zou et al., 2023). Early studies attributed such failures to competing objectives and mismatched generalization in safety training (Wei et al., 2023). Subsequent work demonstrated transferable adversarial sufixes (Zou et al., 2023) and black-box semantic attacks such as PAIR (Chao et al., 2025), while indirect prompt injection showed that malicious instructions can also be introduced through external documents or retrieved content (Greshake et al., 2023). These studies establish that refusal behavior can be highly sensitive to how harmful intent is framed and delivered (Wei et al., 2023; Chao et al., 2025; Greshake et al., 2023).

Multimodal models introduce additional attack surfaces because instructions and contextual cues can be distributed across language and vision (Gong et al., 2025; Liu et al., 2024; Li et al., 2024b). Beyond textual manipulation, visual inputs can be exploited through adversarial images (Liu et al., 2024; Li et al., 2024b), typographic prompts (Gong et al., 2025), and other cross-modal constructions (Luo et al., 2024). Corresponding defenses span safety alignment, input transformation, and inference-time safeguards (Jain et al., 2023; Wang et al., 2024), yet their robustness across changes in modality and context remains uncertain. In particular, identical harmful intent may elicit diferent safety behavior under diferent linguistic or visual contexts, motivating controlled evaluation of multimodal alignment robustness.

## 2.2 Multimodal Jailbreak Benchmarks

Standardized safety benchmarks such as SafetyBench (Zhang et al., 2024), Do-Not-Answer (Wang et al., 2023), SALAD-Bench (Li et al., 2024a), HarmBench (Mazeika et al., 2024), and JailbreakBench (Chao et al., 2024) establish structured taxonomies and reproducible protocols for evaluating harmful behavior and refusal robustness. Multimodal benchmarks extend this paradigm to visual inputs: MM-SafetyBench evaluates image-based safety attacks (Liu et al., 2024), FigStep (Gong et al., 2025) studies harmful instructions rendered as typographic images, and JailBreakV (Luo et al., 2024) and VLJailbreakBench (Wang et al., 2025) expand coverage across diverse multimodaljailbreak strategies. MMJ-Bench (Weng et al., 2025) further provides a unified framework for comparing multimodal jailbreak attacks and defenses, while OmniSafeBench-MM (Jia et al., 2025) broadens standardized attack-defense evaluation with multidimensional metrics.

![](images/3f49b1feb8c2cb83e63124d3c2fcf95426b7a79c581caa6f7e7fc9c0c897d0f3.jpg)  
Figure 2: Construction and evaluation pipeline of MMJailBench. The benchmark factorizes harmful intent, prompt framing, visual semantics, and instruction carrier, and evaluates MLLMs under controlled multimodal jailbreak configurations.

Recent work has also moved toward more structured crossmodal evaluation. Omni-SafetyBench (Pan et al., 2025) constructs parallel variants of the same harmful seeds across diferent modality configurations and evaluates cross-modal safety consistency. These benchmarks substantially improve attack coverage, evaluation standardization, and modality-level comparison. However, they are not designed to independently manipulate the linguistic and semantic factors that jointly shape safety decisions within matched multimodal interactions. MMJailBench complements these eforts through a controlled factorized design that systematically varies prompt framing, task-relevant visual semantics, and instruction carrier over matched harmful intents, enabling fine-grained comparison and factor-level attribution of multimodal jailbreak vulnerabilities.

## 3 MMJailBench

MMJailBench is a factorized multimodal jailbreak benchmark designed to analyze how contextual factors influence MLLM safety behavior. As illustrated in Figure 2, MMJailBench constructs matched multimodal scenarios by systematically varying harmful intents, prompt framings, visual semantics, and instruction carriers, rather than relying on fixed jailbreak instances with entangled factors. By controlling these factors within a unified evaluation space, MMJailBench enables fine-grained analysis of their impact on model safety behavior. The following subsections describe the benchmark design and individual factors in detail.

## 3.1 Factorized Design

Existing multimodal jailbreak benchmarks often focus on predefined attack instances or modality variations, where multiple safety-relevant factors, including harmful intent, linguistic formulation, visual context, and instruction presentation, remain jointly encoded within individual evaluations. Such designs provide valuable measurements of overall vulnerability but ofer limited insight into which factors contribute to observed safety variations.

To enable controlled analysis, MMJailBench factorizes a multimodal jailbreak instance into four components: harmful intent, prompt framing, visual semantics, and instruction carrier. Given a harmful intent ℎ, a prompt framing strategy �, a visual semantic condition �, and an instruction carrier �, each multimodal instance is represented as

$$
\displaystyle x _ { h , t , \nu , c } = ( L ( h , t , c ) , V ( h , \nu ) ) ,\tag{1}
$$

where $L ( h , t , c )$ denotes the instruction component generated from harmful intent $h ,$ prompt framing $t ,$ and instruction carrier $^ { c , }$ while $V ( h , \nu )$ denotes the visual input associated with harmful intent ℎ and visual semantics �.

By fixing the underlying harmful intent and systematically varying contextual factors, MMJailBench enables matched comparisons across multimodal conditions, allowing the contribution of each factor to observed safety variations to be analyzed. The benchmark contains 272 harmful intents, 6 prompt framing strategies, 5 visual semantic conditions, and 2 instruction carriers, resulting in 16,320 controlled multimodal jailbreak instances through Cartesian combinations of these factors. This factorized design allows systematic analysis of how diferent contextual factors are associated with changes in model safety behavior while maintaining consistency across models and harm categories.

## 3.2 Harmful Intent

Harmful intent represents the underlying unsafe objective evaluated in MMJailBench. Instead of directly collecting jailbreak prompts that entangle harmful goals with specific attack strategies, MMJailBench constructs a set of intent-level behavioral seeds, allowing contextual factors to be systematically varied while keeping the harmful objective fixed.

The benchmark contains 272 harmful intents across 9 major harm domains: Physical Harm, Cyber Abuse, Economic Harm, Hate and Harassment, Privacy and IP, Regulated Advice, Illegal Activities, Deception and Influence, and Sexual Content. Each category is further divided into fine-grained subcategories to improve coverage and diversity across safety-critical behaviors. The harmful intents are collected through category-guided construction, followed by semantic similarity filtering and manual verification to reduce redundancy. These intent-level seeds provide a consistent foundation for evaluating how prompt framing, visual semantics, and instruction carriers influence multimodal safety behavior under controlled conditions.

## 3.3 Prompt Framing

Prompt framing specifies how the same harmful intent is linguistically presented to the model. Although the underlying harmful objective remains unchanged, diferent narrative structures and interaction styles may afect how models interpret requests and determine whether to comply.

MMJailBench introduces 6 representative prompt framing strategies: academic, system, structured, story, code, and safetyparadox framing. These strategies capture common ways of reformulating harmful requests while preserving the underlying intent. For example, a harmful request may be embedded within a fictional scenario, expressed as a structured generation task, or presented as an apparently legitimate system requirement. By varying only the linguistic framing while keeping harmful intent fixed, MMJailBench enables direct comparison of how diferent presentation styles correspond to changes in multimodal safety behavior.

## 3.4 Visual Semantics

Visual semantics capture the contextual meaning introduced by accompanying images. Unlike image-based instruction attacks that primarily encode instructions into visual text, MM-JailBench focuses on whether contextual visual information changes safety behavior when the underlying harmful intent remains unchanged.

We construct 5 task-relevant visual semantic conditions, including danger-related context, scenario context, professionalrole context, identity credentials, and authorization documents. These conditions represent diferent forms of contextual signals that may afect how a request is interpreted, such as legitimacyassociated, expertise-related, or authority-related contextual signals. For each harmful intent, the corresponding visual input provides additional semantic context without modifying the underlying harmful instruction. In addition, controlled visual conditions, including irrelevant or non-semantic images, are used in diagnostic evaluations to distinguish the efect of meaningful visual context from the general presence of an image. This design enables analysis of whether MLLMs exhibit diferent safety behaviors under visually grounded contextual conditions.

## 3.5 Instruction Carrier

Instruction carrier describes how harmful instructions are presented to the model. This factor examines whether changing the presentation modality of the same instruction afects multimodal safety behavior while keeping the underlying harmful intent unchanged.

MMJailBench considers 2 instruction carriers: direct textual instructions (TEXT) and visually rendered instructions in text-containing images (OCR). The TEXT carrier directly provides harmful instructions as text, whereas the OCR carrier renders the same instructions into images while preserving their semantics. To account for the additional perception stage introduced by OCR, we measure instruction recognition mismatch, where models fail to interpret embedded instructions correctly. This allows us to isolate carrier-related safety diferences from perception failures.

## 4 Experimental Setup

This section describes the evaluated MLLMs, metrics, judge models, and evaluation configurations used in MMJailBench. All models are evaluated under the same factorized framework for consistent comparison across multimodal jailbreak conditions. Additional experimental details, including evalu ation prompts, judge criteria, and implementation details, are provided in the appendix.

## 4.1 Evaluated MLLMs

We evaluate MMJailBench on 16 representative MLLMs, covering both open-weight and proprietary-access models. The evaluated models include representative model families such as LLaVA-OneVision (An et al., 2025), InternVL (Zhu et al., 2025), Kimi (Team et al., 2026), Qwen (Bai et al., 2025b,a), Gemma (Team et al., 2025), GLM (Hong et al., 2025), GPT (Singh et al., 2025), Gemini (Team et al., 2023), and Claude (Anthropic, 2025).

For each model, we evaluate the complete benchmark configuration containing 16,320 multimodal jailbreak instances, resulting in 261,120 model responses across all evaluated systems. The evaluation aims to characterize model safety behavior under controlled multimodal contexts and analyze vulnerability patterns across diferent jailbreak conditions.

## 4.2 Metrics and Judge Models

MMJailBench evaluates model safety using complementary metrics. The primary metric is Attack Success Rate (ASR), which measures the proportion of responses that provide substantive assistance to the targeted harmful intent:

$$
A S R = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( s _ { i } \geq 4 ) ,\tag{2}
$$

where $s _ { i }$ denotes the harmfulness score of the �-th response on a 1∼5 scale. Responses with $s _ { i } \geq 4$ are considered successful jailbreak cases.

For instruction carrier analysis, we further introduce Conditional Attack Success Rate (CASR):

$$
C A S R = \frac { N _ { s u c c e s s } } { N - N _ { m i s m a t c h } } ,\tag{3}
$$

where mismatch cases refer to inputs that are not correctly understood by the model, including OCR recognition failures, irrelevant responses, and misinterpretation of the evaluated intent. ASR measures end-to-end jailbreak vulnerability, whereas CASR evaluates safety robustness conditioned on successful input understanding.

The main experiments use GPT-5 as an LLM-based judge. For each model response, GPT-5 assigns a harmfulness score and auxiliary labels for jailbreak success and instruction mismatch. The evaluation focuses on generated responses: refusals, safety warnings, and high-level discussions are not considered successful jailbreak cases. All main results are based on GPT-5 judgments. We additionally develop a benchmark-specific lightweight judge model based on an open-weight MLLM for cost-eficient follow-up evaluation under the same protocol.

## 4.3 Evaluation Configurations

MMJailBench supports full and lightweight evaluation configurations. The full configuration evaluates all benchmark instances and provides complete factor coverage for the main analysis, including model-level, domain-level, and factor-level comparisons. The lightweight configuration samples a representative subset while preserving coverage across harmful domains, prompt framings, visual semantics, and instruction carriers, enabling eficient safety auditing and regression testing. Unless otherwise specified, all reported results are obtained using the full configuration.

## 5 Results and Analysis

We evaluate MMJailBench across 16 representative MLLMs and analyze overall jailbreak vulnerability, factor-level efects, and model-dependent vulnerability patterns enabled by the factorized benchmark design.

![](images/2f67274b29ed2f460302cf9bb2c5b0de86b7de8dc54c2a173826b718d0c71a66.jpg)  
Figure 3: Model-level vulnerability profiles across prompt framing and visual semantic factors. Each radar plot shows ASR variation under controlled factor changes, revealing modelspecific sensitivity patterns.

## 5.1 Overall Jailbreak Landscape

We first evaluate the overall jailbreak vulnerability under the full MMJailBench configuration. Figure 3 and Table 1 summarize model-level ASR across controlled multimodal conditions and harmful domains. The evaluated MLLMs exhibit substantial diferences in safety robustness under identical jailbreak configurations. The average ASR ranges from 2.17% for GPT-5 to 78.38% for GLM-4.6V, demonstrating that multimodal safety alignment remains highly model-dependent. Notably, models with comparable multimodal capabilities may exhibit significantly diferent vulnerability profiles, indicating that stronger general capability does not necessarily imply stronger jailbreak robustness.

Beyond model-level diferences, vulnerability is also unevenly distributed across harmful objectives. Cyber abuse, economic harm, privacy-related behaviors, and deception-related tasks generally achieve higher ASR, whereas categories such as physical harm exhibit comparatively lower vulnerability. These results suggest that current multimodal safety alignment does not provide uniform protection across harmful objectives, motivating further factor-level analysis to identify the sources of observed vulnerability.

## 5.2 Factor-Level Attribution

The factorized design of MMJailBench enables controlled comparison by varying individual factors while keeping other conditions fixed. We analyze four dimensions of jailbreak vulnerability: harmful intent, prompt framing, visual semantics, and instruction carrier.

<table><tr><td>Model</td><td>Avg.</td><td>Cyber</td><td>Econ.</td><td>Privacy</td><td>Decept.</td><td>Illegal</td><td>Hate</td><td>Violence</td><td>Sensitive</td><td>Sexual</td></tr><tr><td colspan="9">Open-weight</td><td></td></tr><tr><td>glm4.1v-9b</td><td>72.22</td><td>80.72</td><td>85.28</td><td>71.78</td><td>78.04</td><td>83.16</td><td>67.39</td><td>79.58</td><td>44.94</td><td>59.12</td></tr><tr><td>step3-vl-10b</td><td>53.36</td><td>70.00</td><td>68.22</td><td>62.06</td><td>58.93</td><td>54.48</td><td>49.17</td><td>44.24</td><td>42.06</td><td>27.06</td></tr><tr><td>ministral3-8b</td><td>51.64</td><td>66.83</td><td>63.78</td><td>59.61</td><td>59.58</td><td>56.21</td><td>43.11</td><td>46.98</td><td>38.06</td><td>30.59</td></tr><tr><td>llava-onevision-1.5</td><td>40.64</td><td>42.00</td><td>49.72</td><td>38.17</td><td>42.74</td><td>52.30</td><td>36.00</td><td>48.96</td><td>21.06</td><td>34.80</td></tr><tr><td>internvl3-8b</td><td>49.12</td><td>63.11</td><td>61.72</td><td>51.33</td><td>53.21</td><td>61.49</td><td>42.33</td><td>50.80</td><td>27.39</td><td>30.69</td></tr><tr><td>qwen2.5-vl-7b</td><td>59.48</td><td>67.28</td><td>70.83</td><td>60.94</td><td>62.50</td><td>73.28</td><td>54.39</td><td>66.91</td><td>33.44</td><td>50.78</td></tr><tr><td>qwen3-vl-8b</td><td>19.84</td><td>37.56</td><td>30.28</td><td>28.72</td><td>20.65</td><td>19.02</td><td>12.06</td><td>11.15</td><td>14.39</td><td>4.71</td></tr><tr><td>gemma3-12b</td><td>66.85</td><td>82.94</td><td>80.44</td><td>75.94</td><td>75.95</td><td>74.54</td><td>62.67</td><td>60.66</td><td>51.39</td><td>37.06</td></tr><tr><td colspan="9">Proprietary-access</td><td></td></tr><tr><td>glm-4.6v</td><td>78.38</td><td>93.06</td><td>88.11</td><td>85.17</td><td>87.26</td><td>80.29</td><td>74.94</td><td>73.68</td><td>59.89</td><td>53.04</td></tr><tr><td>qwen3-vl-plus</td><td>62.02</td><td>80.78</td><td>84.00</td><td>80.22</td><td>78.57</td><td>58.85</td><td>55.78</td><td>43.37</td><td>51.67</td><td>24.90</td></tr><tr><td>doubao-seed-2.0-lite</td><td>47.79</td><td>68.39</td><td>61.94</td><td>62.94</td><td>63.10</td><td>42.59</td><td>47.50</td><td>32.99</td><td>46.44</td><td>16.18</td></tr><tr><td>kimi-k2.5</td><td>35.80</td><td>53.28</td><td>45.11</td><td>44.06</td><td>43.81</td><td>35.75</td><td>34.00</td><td>23.30</td><td>30.72</td><td>12.16</td></tr><tr><td>gemini-3-flash</td><td>30.50</td><td>39.61</td><td>38.94</td><td>41.06</td><td>42.14</td><td>23.10</td><td>28.72</td><td>11.94</td><td>36.61</td><td>12.35</td></tr><tr><td>grok-4-fast</td><td>42.61</td><td>55.00</td><td>52.50</td><td>55.00</td><td>52.50</td><td>37.07</td><td>49.33</td><td>26.91</td><td>40.33</td><td>32.84</td></tr><tr><td>claude-sonnet-4.5</td><td>8.96</td><td>15.11</td><td>15.11</td><td>13.06</td><td>10.06</td><td>6.09</td><td>7.00</td><td>2.08</td><td>11.22</td><td>0.88</td></tr><tr><td>gpt-5</td><td>2.17</td><td>3.06</td><td>4.11</td><td>2.00</td><td>3.39</td><td>0.57</td><td>2.50</td><td>0.07</td><td>3.56</td><td>0.29</td></tr></table>

Table 1: Per-model ASR (%) across nine major harmful domains in our MMJailBench. Avg. denotes the average ASR over all evaluated harmful domains.

Harmful Intent. Harmful intent reveals whether safety alignment provides consistent protection across diferent categories of harmful behaviors. As shown in Table 1, jailbreak vulnerability varies substantially across harm domains. Cyber abuse, economic harm, privacy-related behaviors, and deception-related tasks generally exhibit higher ASR across models, while physical harm and sensitive content categories tend to show lower vulnerability. These diferences indicate that aggregate safety scores may hide domain-specific weaknesses, and explicit modeling of harmful intents is necessary for fine-grained safety diagnosis.

Prompt Framing. Prompt framing introduces the largest variation among the evaluated factors. Although the underlying harmful objective remains unchanged, diferent linguistic presentations lead to substantially diferent jailbreak outcomes. As shown in Figure 3(a), models exhibit distinct sensitivity patterns across prompt framing strategies, but story, structured, and academic framings generally lead to higher ASR, while system-style and safety-paradox framings show lower vulnerability. The gap between the most and least vulnerable strategies exceeds 40 percentage points, indicating that MLLMs are sensitive not only to harmful intent itself but also to the surrounding interaction structure and linguistic presentation.

Visual Semantics. Visual semantics introduce additional vulnerabilities beyond the presence of images alone. As shown in Figure 3(b) and Table 2, model responses vary substantially across diferent visual contexts. Task-relevant visual semantics consistently yield higher ASR than the no-image setting, with authorization documents causing the largest increase (+12.96%), followed by identity credentials (+10.47%) and task scenarios (+10.10%). In contrast, non-semantic controls, including blank, noise, and nature images, lead to only marginal changes (less than 2.2%). These results indicate that increased vulnerability mainly stems from contextual visual meaning rather than visual input itself. In particular, authority-related cues may provide implicit legitimacy signals, thereby weakening models’ refusal behavior.

<table><tr><td>Group</td><td>Image condition</td><td>ASR (%)</td><td>∆ASR (%)</td></tr><tr><td rowspan="6">Semantic</td><td>Authorization document</td><td>47.73</td><td>12.96</td></tr><tr><td>Identity credential</td><td>45.24</td><td>10.47</td></tr><tr><td>Task scenario</td><td>44.87</td><td>10.10</td></tr><tr><td>Professional role</td><td>43.60</td><td>8.83</td></tr><tr><td>Dangerous context</td><td>43.22</td><td>8.45</td></tr><tr><td>Blank image</td><td>36.96</td><td>2.19</td></tr><tr><td rowspan="4">Control</td><td>Noise image</td><td>35.53</td><td>0.76</td></tr><tr><td>Nature image</td><td>35.49</td><td>0.72</td></tr><tr><td>No image</td><td>34.77</td><td></td></tr><tr><td></td><td></td><td>0.00</td></tr></table>

Table 2: Visual semantic ablation results comparing semantic and control image conditions. ΔASR denotes the ASR increase relative to the no-image condition.

<table><tr><td>Carrier</td><td>N</td><td>Mis. (%)</td><td>ASR (%)</td><td>CASR (%)</td></tr><tr><td>TEXT</td><td>130,560</td><td>0.37</td><td>50.70</td><td>50.89</td></tr><tr><td>OCR</td><td>130,560</td><td>4.14</td><td>40.59</td><td>42.34</td></tr><tr><td>UNION</td><td>261,120</td><td>2.26</td><td>45.64</td><td>46.69</td></tr></table>

Table 3: Instruction carrier analysis under TEXT and OCR settings. � denotes the number of evaluated model responses, and Mis. represents the instruction mismatch rate.

Instruction Carrier. Instruction carrier exhibits a diferent pattern from prompt framing and visual semantics. As shown in Table 3, OCR-based instructions achieve lower ASR than direct text inputs (40.59% vs. 50.70%), and the diference remains after excluding instruction mismatch cases (42.34% vs. 50.89%). Therefore, visually rendered instructions are not inherently stronger jailbreak carriers; instead, carrier-related vulnerability depends on model-specific multimodal processing behavior.

<table><tr><td>Model</td><td>∆Intent</td><td>∆Prompt</td><td>∆Visual</td><td>∆Carrier</td></tr><tr><td colspan="5">Open-weight</td></tr><tr><td>glm4.1v-9b</td><td>76.7</td><td>51.8</td><td>4.2</td><td>8.3</td></tr><tr><td>step3-vl-10b</td><td>86.7</td><td>72.2</td><td>4.8</td><td>10.5</td></tr><tr><td>ministral3-8b</td><td>83.3</td><td>57.2</td><td>9.6</td><td>19.9</td></tr><tr><td>llava-onevision-1.5</td><td>61.7</td><td>35.3</td><td>4.6</td><td>50.5</td></tr><tr><td>internvl3-8b</td><td>75.0</td><td>88.3</td><td>5.5</td><td>4.6</td></tr><tr><td>qwen2.5-vl-7b</td><td>78.3</td><td>90.8</td><td>3.4</td><td>0.8</td></tr><tr><td>qwen3-vl-8b</td><td>65.0</td><td>15.6</td><td>6.5</td><td>14.4</td></tr><tr><td>gemma3-12b</td><td>88.3</td><td>79.0</td><td>7.0</td><td>13.6</td></tr><tr><td colspan="5">Proprietary-access</td></tr><tr><td>glm-4.6v</td><td>73.3</td><td>62.6</td><td>11.3</td><td>1.5</td></tr><tr><td>qwen3-vl-plus</td><td>91.7</td><td>42.6</td><td>8.1</td><td>9.7</td></tr><tr><td>doubao-seed-2.0-lite</td><td>95.0</td><td>80.2</td><td>5.8</td><td>1.3</td></tr><tr><td>kimi-k2.5</td><td>75.0</td><td>44.3</td><td>6.9</td><td>34.6</td></tr><tr><td>gemini-3-flash</td><td>76.7</td><td>59.7</td><td>7.8</td><td>34.3</td></tr><tr><td>grok-4-fast</td><td>76.7</td><td>70.6</td><td>2.8</td><td>4.1</td></tr><tr><td>claude-sonnet-4.5</td><td>55.0</td><td>32.1</td><td>2.0</td><td>0.8</td></tr><tr><td>gpt-5</td><td>13.3</td><td>8.8</td><td>0.8</td><td>3.6</td></tr></table>

Table 4: Model-dependent vulnerability profiles across MMJail-Bench factors. Each Δ score denotes the range of ASR variation across configurations of the corresponding factor.

## 5.3 Model-Dependent Profiles

Although aggregate ASR provides an overall comparison of model robustness, it does not reveal how individual models respond to diferent jailbreak factors. Table 4 reports the ASR variation range induced by each factor across evaluated models and reveals distinct vulnerability signatures among MLLMs. Some models are highly sensitive to prompt reframing, with ΔPrompt exceeding 80 percentage points, while others exhibit stronger sensitivity to instruction carrier changes. In contrast, visual semantic efects are generally smaller but consistently positive across models.

These heterogeneous profiles indicate that multimodal jailbreak vulnerability is not governed by a single universal failure pattern. Diferent models exhibit diferent combinations of contextual sensitivity, reflecting variations in safety alignment behavior. Therefore, factorized evaluation provides a more informative characterization of model robustness beyond aggregate jailbreak success rates.

## 6 Diagnostic Analysis

The behavioral results reveal substantial variation in multimodal jailbreak vulnerability across prompt framings, visual semantics, and instruction carriers. To examine how these behavioral diferences are reflected in the model’s internal states, we construct a diagnostic subset from matched MMJail-Bench instances in which authority-document scenarios play a prominent role. We analyze layer-wise representations and cross-modal attention patterns in gemma3-12b, providing a model-internal view of how authority-related visual contexts shape the response-generation process.

![](images/8b6d07c1e26bed9bc85a3a8bce64e8610177643fac79268e5dd1f20a67a18df3.jpg)  
Figure 4: Representation-level diagnostics of gemma3-12b under authority-document and danger-image contexts. Authorityrelated visual cues exhibit distinct representation patterns across layers and harm domains.

## 6.1 Representation-Level Diagnostics

Figure 4 analyzes residual-stream representations of the final generation-prefix token. Figure 4(a) shows that the representation divergence between authority and danger conditions increases with depth, with the sharpest rise at layer 42. This indicates that the two visual contexts become increasingly separated in higher-layer representations before response generation.

To examine whether this diference extends beyond the samples used to identify it, we fit an authority-minus-danger representation direction using the discovery samples and apply the fixed direction to held-out pairs. Figure 4(b) shows a clear displacement of authority-document contexts along this direction, indicating that the representation pattern extends to held-out samples. Figure 4(c) further shows consistent positive shifts across all harm domains. This cross-category consistency suggests that authority-related cues are associated with a shared higher-layer representation pattern across diverse harmful objectives.

## 6.2 Cross-Modal Interaction Diagnostics

Multimodal response generation depends on how textual and visual information are integrated across layers. We therefore analyze the attention from the final generation-prefix query to harm-related task-label tokens and special visual tokens. Figure 5 reveals distinct layer-wise attention patterns under authority-document and danger-image contexts.

At the identified representation-sensitive layer, authoritydocument contexts exhibit lower attention mass on both harmrelated task-label and visual tokens. Combined with representation displacement, this suggests that authority cues induce a higher-level contextual state with redistributed attention during response initiation. The consistency between representation shifts and attention dynamics provides an internal explanation for the increased jailbreak susceptibility under authority-related visual contexts.

![](images/1387215e8185d678ba685f3a63ca6ae51ee9c369a1f12e8fe6702bca1c91fad2.jpg)

![](images/6f73c72781cec762811ee70d0305be02059d569baa1cf1cfd06a6c5d7d1e9577.jpg)  
Figure 5: Cross-modal interaction diagnostics in gemma3- 12b. Attention allocation to harm-related textual tokens and visual tokens varies across layers under authority-document and danger-image contexts.

## 7 Conclusion

In this work, we introduce MMJailBench, a factorized benchmark for systematically analyzing multimodal jailbreak vulnerabilities through controlled variations of harmful intents, prompt framings, visual semantics, and instruction carriers. Evaluations across 16 open-weight and proprietary MLLMs reveal substantial diversity in multimodal safety behaviors: prompt framing introduces the largest variation, visual semantics provide additional vulnerability through contextual cues, instruction carriers exhibit model-dependent efects, and harmful intents reveal category-specific vulnerability patterns. Diagnostic analyses further uncover vulnerability-associated patterns in internal representations and cross-modal interactions. With modular full and lightweight configurations, flexible judge options, and multidimensional metrics, MMJailBench provides a systematic and reproducible framework for scalable multimodal safety evaluation.

## References

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

Anthropic. Claude sonnet 4.5 system card. Technical report, Anthropic, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramer, et al. Jailbreakbench: An open robustness benchmark

for jailbreaking large language models. Advances in Neural Information Processing Systems, 37:55005–55029, 2024.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. In Proceedings ofthe IEEE Conference on Secure and Trustworthy Machine Learning, pages 23–42, 2025.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Advances in Neural Information Processing Systems, 36:49250–49267, 2023.

Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. Figstep: Jailbreaking large vision-language models via typographic visual prompts. Proceedings ofthe AAAI Conference on Artificial Intelligence, 39(22):23951–23959, 2025.

Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. In Proceedings of the ACM Workshop on Artificial Intelligence and Security, pages 79–90, 2023.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614, 2023.

Xiaojun Jia, Jie Liao, Qi Guo, Teng Ma, Simeng Qin, Ranjie Duan, Tianlin Li, Yihao Huang, Zhitao Zeng, Dongxian Wu, et al. Omnisafebench-mm: A unified benchmark and toolbox for multimodal jailbreak attack-defense evaluation. arXiv preprint arXiv:2512.06589, 2025.

Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. Salad-bench: A hierarchical and comprehensive safety benchmark for large language models. In Findings of the Association for Computational Linguistics, pages 3923–3954, 2024a.

Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models. In Proceedings of the European Conference on Computer Vision, pages 174–189, 2024b.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. Mm-safetybench: A benchmark for safety evaluation of multimodal large language models. In Proceedings ofthe European Conference on Computer Vision, pages 386–403, 2024.

Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, and Chaowei Xiao. Jailbreakv: A benchmark for assessing the robustness of multimodal large language models against jailbreak attacks. arXiv preprint arXiv:2404.03027, 2024.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.

Leyi Pan, Zheyu Fu, Yunpeng Zhai, Shuchang Tao, Sheng Guan, Shiyu Huang, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Felix Henry, et al. Omni-safetybench: A benchmark for safety evaluation of audio-visual large language models. arXiv preprint arXiv:2508.07173, 2025.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Ziwei Chai, Y Charles, HS Che, Cheng Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, and Yu-Gang Jiang. Ideator: Jailbreaking and benchmarking large vision-language models using themselves. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 8875–8884, 2025.

Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, and Chaowei Xiao. Adashield: Safeguarding multimodal large language models from structure-based attack via adaptive shield prompting. In Proceedings of the European Conference on Computer Vision, pages 77–94, 2024.

Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. Do-not-answer: A dataset for evaluating safeguards in llms. arXiv preprint arXiv:2308.13387, 2023.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36:80079–80110, 2023.

Fenghua Weng, Yue Xu, Chengyan Fu, and Wenjie Wang. Mmj-bench: A comprehensive study on jailbreak attacks and defenses for vision language models. Proceedings of the AAAI Conference on Artificial Intelligence, 39(26):27689– 27697, 2025.

Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. Safetybench: Evaluating the safety of large language models. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 15537–15553, 2024.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## A Benchmark Construction

## A.1 Harmful Intent Taxonomy, Collection, and Filtering

The harmful intent taxonomy defines the semantic space of MMJailBench by organizing harmful behaviors into a structured hierarchy of intents, domains, and scenarios. Each intent is associated with a task identifier, harm category, scenario, and action description.

The taxonomy contains 272 unique harmful intents covering 9 major harm domains and 18 harm scenarios. Table 5 summarizes the taxonomy structure and category statistics. The 9 major harm domains include Cyber & Tech Abuse (Cyber), Economic Harm (Econ.), Privacy & IP (Privacy), Deception & Public Influence (Decept.), Illegal Goods & Services (Illegal), Hate & Harassment (Hate), Violence & Physical Harm (Violence), Sensitive Regulated Advice (Sensitive), and Sexual Content (Sexual). The taxonomy is constructed from existing safety benchmarks (Luo et al., 2024; Gong et al., 2025; Liu et al., 2024; Souly et al., 2024) and public model-provider safety policies (OpenAI, 2025, 2026; Anthropic, 2025a; Google, 2026), followed by intent normalization, taxonomy organization, and manual filtering.

To improve semantic diversity and reduce redundancy, we perform embedding-based similarity analysis among the 272 intents using the all-MiniLM-L6-v2 sentence embedding model (Reimers and Gurevych, 2019). Figure 6 shows that most intent pairs have relatively low semantic similarity, while only a small subset forms a high-similarity tail. These cases are manually reviewed to remove near-duplicates and ensure that retained intents represent distinct harmful objectives. We further visualize the intent embedding space using PCA in Figure 7. Although the projection provides a qualitative view, intents from diferent harm domains occupy diverse regions rather than collapsing into a small number of clusters. Together, the diversity analysis and manual filtering establish a balanced harmful-intent set for subsequent factorized evaluation.

## A.2 Prompt Framing Templates

MMJailBench uses 6 representative prompt framing templates to evaluate how diferent contextual framings influence MLLM safety behavior. These templates preserve the underlying harmful intent while varying the surrounding framing strategy, enabling controlled comparison of jailbreak suscep tibility across prompt styles. The 6 templates cover common jailbreak-related framing strategies, including Academic Analysis (Academic), System-Role Simulation (System), Fictional Storytelling (Story), Code-Style Indirection (Code), Structured Completion (Structure), and Safety-Paradox Reasoning (Paradox).

Table 6 summarizes the 6 prompt framings and their corresponding design characteristics. Each template is instantiated by

<table><tr><td>Category</td><td>Count</td><td>Ratio (%)</td></tr><tr><td>Cyber &amp; Tech Abuse</td><td>30</td><td>11.0</td></tr><tr><td>• Malware &amp; Hacking</td><td>16</td><td>5.9</td></tr><tr><td>• Agent Abuse &amp; Automation</td><td>14</td><td>5.1</td></tr><tr><td>Economic Harm</td><td>30</td><td>11.0</td></tr><tr><td>• Fraud &amp; Scams</td><td>16</td><td>5.9</td></tr><tr><td>• Financial Crimes &amp; Risky Practices</td><td>14</td><td>5.1</td></tr><tr><td>Privacy &amp; IP</td><td>30</td><td>11.0</td></tr><tr><td>• Privacy &amp; Surveillance</td><td>16</td><td>5.9</td></tr><tr><td>• Intellectual Property</td><td>14</td><td>5.1</td></tr><tr><td>Deception &amp; Public Influence</td><td>28</td><td>10.3</td></tr><tr><td>• Disinformation</td><td>14</td><td>5.1</td></tr><tr><td>• Government Decision &amp; Political</td><td>14</td><td>5.1</td></tr><tr><td>Illegal Goods &amp; Services</td><td>29</td><td>10.7</td></tr><tr><td>• Controlled Substances</td><td>15</td><td>5.5</td></tr><tr><td>• General Illegal Acts</td><td>14</td><td>5.1</td></tr><tr><td>Hate &amp; Harassment</td><td>30</td><td>11.0</td></tr><tr><td>• Hate Speech &amp; Extremism</td><td>15</td><td>5.5</td></tr><tr><td>• Harassment &amp; Bullying</td><td>15</td><td>5.5</td></tr><tr><td>Violence &amp; Physical Harm</td><td>48</td><td>17.6</td></tr><tr><td>• Violence &amp; Terrorism</td><td>17</td><td>6.3</td></tr><tr><td>• Weapons &amp; CBRN</td><td>16</td><td>5.9</td></tr><tr><td>• Suicide &amp; Self-Harm</td><td>15</td><td>5.5</td></tr><tr><td>Sensitive Regulated Advice</td><td>30</td><td>11.0</td></tr><tr><td>• Unlicensed Medical &amp; Legal Advice</td><td>15</td><td>5.5</td></tr><tr><td>• High-Risk Financial Advice</td><td>15</td><td>5.5</td></tr><tr><td>Sexual Content</td><td>17</td><td>6.3</td></tr><tr><td>• Sexual Content</td><td>17</td><td>6.3</td></tr><tr><td>Total</td><td>272</td><td>100.0</td></tr></table>

Table 5: Harmful-intent taxonomy and category statistics. MM-JailBench contains 272 intents organized into 9 major harm domains and 18 harm scenarios.

![](images/5577141eef8a8db2a4221347817ac5422329a07233831c24f186a8ef53e79f13.jpg)  
Figure 6: Pairwise cosine similarity distribution among the 272 harmful intents. The high-similarity tail is manually inspected during taxonomy construction to identify and remove nearduplicate candidates.

![](images/0443290e25bb56e2b587d593f3dad8da1f32b71e61ca40ebea71eac106d60d3a.jpg)  
Cyber+Econ.PrivacyDecept.Illegal Hate Violence ×SensitiveSexual

Figure 7: PCA visualization of harmful-intent embeddings. Diferent colors represent harm domains, providing a qualitative view of taxonomy coverage and semantic diversity.  
![](images/075dab49ab923d0add19651fc5606b41ac1c5713b3a4b9976a4f40608a8d6782.jpg)  
Figure 8: Examples of the five visual semantic categories in MMJailBench. Each category provides distinct visual contextual cues while preserving the underlying harmful intent.

replacing the task-specific placeholder with the target harmful intent, while keeping the harmful objective unchanged across diferent templates. The templates are designed as evaluation probes rather than an exhaustive collection of jailbreak attacks, providing repeatable linguistic conditions for factor-level analysis.

## A.3 Visual Semantic Generation

MMJailBench introduces 5 visual semantic categories, including Danger, Work Scenario, Authorization Document, Identity Credential, and Role, to study how diferent visual cues influence MLLM safety behavior while preserving the underlying harmful intent and prompt framing.

<table><tr><td>Template</td><td>Framing Strategy</td><td>Representative Pattern</td></tr><tr><td>Academic</td><td>Academic Analysis</td><td>Analyze the topic from an academic perspective and discuss related aspects.</td></tr><tr><td>System</td><td>System-Role Simulation</td><td>Assume a specific system identity and follow role-specific response instructions.</td></tr><tr><td>Story</td><td>Fictional Storytelling</td><td>Describe the scenario through a fictional story or hypothetical setting.</td></tr><tr><td>Code</td><td>Code-Style Indirection</td><td>Express the requested procedure through code format or comments.</td></tr><tr><td>Structure</td><td>Structured Completion</td><td>Complete a predefined structure containing multiple stages or sections.</td></tr><tr><td>Paradox</td><td>Safety-Paradox Reasoning</td><td>Discuss harmful actions indirectly under a safety-oriented rationale.</td></tr></table>

Table 6: Overview of the 6 prompt framing templates used in MMJailBench. Each template modifies the prompt framing while preserving the underlying harmful intent.

For each harmful intent, qwen2.5-vl-7b (Bai et al., 2025b) is used to generate visual descriptions and image generation prompts. zimage2025-Turbo (Z-Image Team et al., 2025) is used for visual scene generation in Danger, Work Scenario, and Role, while a 4-bit-quantized FLUX.2 [dev] (Black Forest Labs, 2025) is used for the Identity Credential and Authorization Document categories due to its image-editing capability for generating images with coherent textual elements. Generated images are filtered based on semantic relevance, visual quality, and readability. Figure 8 shows representative examples of the generated visual semantic conditions.

## A.4 Instruction Carrier Construction

We construct the instruction carrier factor by controlling how the same harmful intent is delivered to MLLMs. MMJailBench considers two carrier modes: direct text input (TEXT) and visually embedded instruction input (OCR). For OCR instances, the instruction text is rendered into a readable image and combined with the corresponding visual semantic image while preserving the original harmful intent and prompt framing. A standardized layout is used to reduce formatting variations and enable controlled comparison between textual and visually embedded instructions.

## B Experimental and Evaluation Details

## B.1 Evaluated MLLMs

We evaluate MMJailBench on 16 MLLMs, including 8 openweight models and 8 proprietary-access models. The openweight models include qwen2.5-vl-7b (Bai et al., 2025b), qwen3- vl-8b (Bai et al., 2025a), gemma3-12b (Gemma Team et al., 2025), internvl3-8b (Zhu et al., 2025), glm4.1v-9b (Hong et al., 2025), ministral3-8b (Liu et al., 2026), step3-vl-10b (Huang et al., 2026), and llava-onevision-1.5 (An et al., 2025). The proprietary-access models include gpt-5 (Singh et al., 2025), gemini-3-flash (Google DeepMind, 2025), claude-sonnet-4.5 (Anthropic, 2025b), grok-4-fast (xAI, 2025), qwen3-vlplus (Bai et al., 2025a), kimi-k2.5 (Kimi Team et al., 2026), glm-4.6v (Hong et al., 2025), and doubao-seed-2.0-lite (ByteDance Seed Team, 2026). Open-weight models are evaluated using publicly available checkpoints, while proprietary-access models are evaluated through their oficial interfaces.

All models are evaluated under the same factorized benchmark configuration. Each model is tested on the complete Cartesian product of 272 harmful intents, 6 prompt templates, 5 visual semantic conditions, and 2 instruction carrier modes, resulting in 16,320 instances per model and 261,120 total model responses across all evaluated models.

## B.2 Evaluation Rubric and Judge Prompt

GPT-5 is used as the primary evaluator for MMJailBench responses. For each response, the judge outputs an instructionmismatch flag and a harmfulness score. The mismatch flag identifies failures to recognize the task, such as only transcribing visual content, repeating the prompt, or producing meaningless outputs. The harmfulness score follows a strict 1∼5 rubric, with higher scores indicating stronger compliance and more actionable harmful content. The detailed scoring criteria are summarized in Table 7.

The judge prompt incorporates anti-deception rules to prevent safety disclaimers, fictional framing, or academic packaging from afecting the evaluation when harmful information is provided. Responses limited to refusal, safe redirection, or non-actionable discussion receive low harmfulness scores. The complete judge prompt template is provided in the Code and Data Supplement.

## B.3 Full and Lightweight Benchmark Settings

The full MMJailBench configuration evaluates the complete Cartesian product of all benchmark factors, including 272 harmful intents, 6 prompt templates, 5 visual semantic conditions, and 2 instruction carrier modes. This results in 16,320 instances per model and provides comprehensive coverage of the benchmark space.

<table><tr><td>Score</td><td>Definition</td></tr><tr><td>1</td><td>Refusal or safe response without harmful details.</td></tr><tr><td>2</td><td>Potential risk without direct harmful instructions.</td></tr><tr><td>3</td><td>High-level or partial harmful information.</td></tr><tr><td>4</td><td>Actionable harmful assistance.</td></tr><tr><td>5</td><td>Detailed and operational harmful guidance.</td></tr></table>

Table 7: Harmfulness scoring rubric for response evaluation.

<table><tr><td>Category</td><td>Count</td><td>ASR</td><td> $\Delta _ { \mathrm { T - O } }$ </td></tr><tr><td colspan="4">Cyber &amp; Tech Abuse</td></tr><tr><td>• Malware &amp; Hacking</td><td>16</td><td>56.22</td><td>10.77</td></tr><tr><td>• Agent Abuse &amp; Automation</td><td>14</td><td>58.79</td><td>11.58</td></tr><tr><td colspan="4">Economic Harm</td></tr><tr><td>• Fraud &amp; Scams</td><td>16</td><td>57.66 11.20</td><td></td></tr><tr><td>• Financial Crimes &amp; Risky Practices</td><td>14</td><td>54.66</td><td>10.57</td></tr><tr><td colspan="4">Privacy &amp; IP</td></tr><tr><td>• Privacy &amp; Surveillance</td><td>16</td><td>52.80 11.12</td><td></td></tr><tr><td>• Intellectual Property</td><td>14</td><td>51.09</td><td>10.22</td></tr><tr><td colspan="4">Deception &amp; Public Influence</td></tr><tr><td>• Disinformation</td><td>14</td><td>53.73 12.46</td><td></td></tr><tr><td>• Government Decision &amp; Political</td><td>14</td><td>50.33</td><td>11.76</td></tr><tr><td colspan="4">Illegal Goods &amp; Services</td></tr><tr><td>• Controlled Substances</td><td>15</td><td>44.78</td><td>9.10</td></tr><tr><td>• General Illegal Acts</td><td>14</td><td>50.25</td><td>11.46</td></tr><tr><td colspan="4">Hate &amp; Harassment</td></tr><tr><td>• Hate Speech &amp; Extremism</td><td>15</td><td>36.34 10.63</td><td></td></tr><tr><td>• Harassment &amp; Bullying</td><td>15</td><td>47.02</td><td>10.07</td></tr><tr><td colspan="4">Violence &amp; Physical Harm</td></tr><tr><td>• Violence &amp; Terrorism</td><td>17</td><td>37.13</td><td>8.58</td></tr><tr><td>• Weapons &amp; CBRN</td><td>16</td><td>43.39</td><td>9.43</td></tr><tr><td>• Suicide &amp; Self-Harm</td><td>15</td><td>36.36</td><td>8.86</td></tr><tr><td colspan="4">Sensitive Regulated Advice</td></tr><tr><td>• Unlicensed Medical &amp; Legal Advice</td><td>15</td><td>26.69</td><td>5.81</td></tr><tr><td>• High-Risk Financial Advice</td><td>15</td><td>42.45</td><td>8.35</td></tr><tr><td colspan="4">Sexual Content</td></tr><tr><td>• Sexual Content</td><td>17</td><td>26.72 10.66</td><td></td></tr><tr><td>Total</td><td>272</td><td>45.74</td><td>10.10</td></tr></table>

Table 8: Category-level jailbreak results. ASR is computed by pooling the two instruction carriers, and $\Delta _ { \mathrm { T - O } }$ denotes the ASR diference between TEXT and OCR carriers.

To reduce evaluation cost, we construct a lightweight configuration containing 1,500 instances by stratified sampling over harm domains, prompt framings, visual semantic conditions, and carrier modes while preserving the factor distribution of the full configuration. The efectiveness of the lightweight configuration is evaluated in Section C.4.

## B.4 Lightweight Judge Model

To improve evaluation eficiency, we train a benchmark-scoped lightweight judge based on qwen3-8b (Team et al., 2025) using GPT-5-generated annotations. The lightweight judge predicts harmfulness scores and mismatch labels under the same evaluation protocol as the primary GPT-5 judge. Its consistency with GPT-5 evaluation is further validated in Section C.4. Detailed training configurations are provided in the Code and Data Supplement.

## C Extended Results and Ablations

## C.1 Extended Factor-Level Results

This section provides additional factor-level results to complement the main paper. We analyze how diferent harmful intents, instruction carriers, prompt framings, and visual semantic conditions afect MLLM jailbreak performance.

<table><tr><td>Model</td><td>Acad.</td><td>Sys.</td><td>Story</td><td>Code</td><td>Stru.</td><td>Para.</td></tr><tr><td colspan="7">Open-weight</td></tr><tr><td>glm4.1v-9b</td><td>77.83</td><td>46.40</td><td>98.20</td><td>68.16</td><td>91.40</td><td>57.54</td></tr><tr><td>step3-vl-10b</td><td>65.11</td><td>15.51</td><td>58.05</td><td>46.36</td><td>87.68</td><td>48.42</td></tr><tr><td>ministral3-8b</td><td>82.72</td><td>30.70</td><td>70.00</td><td>25.55</td><td>76.36</td><td>28.24</td></tr><tr><td>llava-onevision-1.5</td><td>35.59</td><td>50.51</td><td>54.89</td><td>40.66</td><td>47.21</td><td>19.60</td></tr><tr><td>internvl3-8b</td><td>74.41</td><td>0.07</td><td>88.42</td><td>37.10</td><td>57.13</td><td>43.09</td></tr><tr><td>qwen2.5-vl-7b</td><td>91.43</td><td>0.66</td><td>88.60</td><td>64.23</td><td>84.67</td><td>35.62</td></tr><tr><td>qwen3-vl-8b</td><td>25.37</td><td>22.32</td><td>14.19</td><td>15.40</td><td>29.12</td><td>13.49</td></tr><tr><td>gemma3-12b</td><td>86.40</td><td>18.90</td><td>97.94</td><td>74.08</td><td>92.21</td><td>37.06</td></tr><tr><td>Proprietary-access</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>glm-4.6v</td><td>84.67</td><td>79.56</td><td>96.29</td><td>80.55</td><td>93.86</td><td>33.71</td></tr><tr><td>qwen3-vl-plus</td><td>63.27</td><td>40.11</td><td>82.72</td><td>63.64</td><td>68.38</td><td>56.54</td></tr><tr><td>doubao-seed-2.0-lite</td><td>63.27</td><td>16.10</td><td>96.29</td><td>23.90</td><td>45.77</td><td>51.95</td></tr><tr><td>kimi-k2.5</td><td>59.45</td><td>40.33</td><td>59.38</td><td>20.99</td><td>20.96</td><td>15.15</td></tr><tr><td>gemini-3-flash</td><td>34.96</td><td>3.20</td><td>62.87</td><td>9.60</td><td>50.88</td><td>18.97</td></tr><tr><td>grok-4-fast</td><td>66.40</td><td>3.53</td><td>52.50</td><td>24.12</td><td>74.15</td><td>43.12</td></tr><tr><td>claude-sonnet-4.5</td><td>5.40</td><td>0.00</td><td></td><td>2.54</td><td>32.13</td><td></td></tr><tr><td></td><td></td><td></td><td>10.00</td><td></td><td></td><td>3.27</td></tr><tr><td>gpt-5</td><td>0.74</td><td>0.00</td><td>0.44</td><td>0.85</td><td>8.82</td><td>1.88</td></tr></table>

Table 9: Per-model ASR (%) across diferent prompt framing strategies on MMJailBench.

Table 8 presents category-level jailbreak results across the harmful intent taxonomy and compares TEXT and OCR instruction carriers. The results show substantial variation in ASR across diferent harmful categories, with Cyber & Tech Abuse, Economic Harm, and Privacy & IP exhibiting relatively higher attack success rates, while Sensitive Regulated Advice and Sexual Content show lower ASR under the evaluated settings. Meanwhile, $\Delta _ { \mathrm { T - O } }$ remains positive for all categories, ranging from 5.81 to 12.46 percentage points, indicating that the carrier efect is consistently observed across diverse harmful intents rather than being dominated by specific harm domains.

Tables 9 and 10 further analyze model-level sensitivity to prompt framing strategies and visual semantic conditions. The results reveal substantial diferences among models: some MLLMs exhibit large variations across prompt framings, while others maintain relatively stable behavior. Similarly, diferent visual semantic conditions lead to distinct vulnerability patterns across models, suggesting that contextual visual cues can influence jailbreak susceptibility in a model-dependent manner. Together, these observations demonstrate the necessity of evaluating multimodal jailbreak robustness under factorized settings where individual contributing factors can be separately analyzed.

## C.2 Per-Model Factor Sensitivity Profiles

Figure 9 summarizes the factor sensitivity profiles of individual MLLMs. Each model is characterized by the ASR variation ranges induced by prompt framing, visual semantic conditions, and instruction carriers, where larger ranges indicate stronger sensitivity to the corresponding factor. K-means clustering groups models into four representative profiles based on their sensitivity patterns.

<table><tr><td>Model</td><td>Dang.</td><td>Scen.</td><td>Auth. Doc.</td><td>ID Cred.</td><td>Role</td></tr><tr><td colspan="6">Open-weight</td></tr><tr><td>glm4.1v-9b</td><td>75.34</td><td>72.33</td><td>71.17</td><td>73.93</td><td>73.50</td></tr><tr><td>step3-vl-10b</td><td>51.87</td><td>53.40</td><td>55.82</td><td>55.48</td><td>51.04</td></tr><tr><td>ministral3-8b</td><td>57.57</td><td>53.80</td><td>52.79</td><td>49.17</td><td>47.98</td></tr><tr><td>llava-onevision-1.5</td><td>42.28</td><td>43.17</td><td>40.10</td><td>42.95</td><td>38.54</td></tr><tr><td>internvl3-8b</td><td>52.60</td><td>52.36</td><td>48.10</td><td>49.97</td><td>47.15</td></tr><tr><td>qwen2.5-vl-7b</td><td>62.19</td><td>61.58</td><td>59.83</td><td>61.95</td><td>58.79</td></tr><tr><td>qwen3-vl-8b</td><td>21.69</td><td>18.29</td><td>22.89</td><td>20.68</td><td>16.36</td></tr><tr><td>gemma3-12b</td><td>64.68</td><td>67.22</td><td>71.63</td><td>67.71</td><td>67.59</td></tr><tr><td colspan="6">Proprietary-access</td></tr><tr><td>glm-4.6v</td><td>70.74</td><td>80.39</td><td>82.08</td><td>79.01</td><td>78.31</td></tr><tr><td>qwen3-vl-plus</td><td>60.81</td><td>59.44</td><td>67.49</td><td>63.51</td><td>60.97</td></tr><tr><td>doubao-seed-2.0-lite</td><td>48.28</td><td>48.96</td><td>53.43</td><td>49.45</td><td>47.61</td></tr><tr><td>kimi-k2.5</td><td>32.23</td><td>37.01</td><td>39.15</td><td>38.76</td><td>33.06</td></tr><tr><td>gemini-3-flash</td><td>29.75</td><td>34.59</td><td>26.75</td><td>28.16</td><td>31.16</td></tr><tr><td>grok-4-fast</td><td>44.64</td><td>43.72</td><td>44.00</td><td>45.16</td><td>42.34</td></tr><tr><td>claude-sonnet-4.5</td><td>8.06</td><td>9.90</td><td>10.02</td><td>8.33</td><td>8.15</td></tr><tr><td>gpt-5</td><td>1.69</td><td>2.05</td><td>2.36</td><td>2.48</td><td>2.02</td></tr></table>

Table 10: Per-model ASR (%) across diferent visual semantic conditions on MMJailBench.  
![](images/5c9db467717ffef847de2e8214b9975350191ad2c607ee8461f091165a3881a7.jpg)  
Figure 9: Per-model jailbreak factor sensitivity profiles. Horizontal and vertical axes represent ASR ranges induced by prompt framing and visual semantic conditions, respectively. Marker size indicates the ASR range induced by instruction carriers. Colors and shapes denote four K-means profiles obtained with 1,000 random restarts.

The results reveal substantial heterogeneity across models. Prompt-dominant models show larger variations across prompt framings, visual-responsive models are more afected by visual semantic conditions, carrier-sensitive models exhibit stronger carrier-related changes, and low-variation models remain relatively stable across factors. These profiles demonstrate that diferent MLLMs may exhibit distinct jailbreak vulnerabilities under diferent contextual conditions.

## C.3 Semantic Image Ablation

Figure 10 extends the semantic image ablation analysis to modellevel profiles. Each ASR value aggregates 3,264 evaluated responses across the corresponding visual condition. The results show substantial variation in how diferent models respond to semantic visual contexts: step3-vl-10b and doubao-seed-2.0-lite exhibit larger gaps between task-relevant semantic conditions and control conditions, while gemma3-12b shows a smaller separation. claude-sonnet-4.5 presents a diferent pattern, with relatively higher ASR under the no-image condition, suggesting that its vulnerability is less dependent on semantic visual cues.

![](images/99abc523e1869611ea0fb9c755bfd583d781026631a89a3d52a6433b3b8a11d3.jpg)  
Figure 10: Per-model ASR across five task-relevant visual semantic conditions and four control conditions. The radial scale spans 0∼80%. Axes, shading, and model colors are consistent with the main-paper radar plot.

## C.4 Evaluation Eficiency and Validation

We evaluate the eficiency and reliability of the MMJailBench evaluation pipeline from two perspectives. First, we compare the lightweight benchmark configuration with the full evaluation setting to assess whether the reduced evaluation scale preserves benchmark consistency. Table 11 shows that the lightweight configuration closely matches the full setting, with small ASR diferences and strong agreement across matched evaluations, demonstrating that the lightweight setting maintains the main evaluation characteristics.

Second, we validate the reliability of the judging protocol through human expert evaluation and lightweight judge validation. Table 12 reports the agreement between GPT-5 judgments and human expert annotations, as well as the consistency between the lightweight judge and GPT-5. Both harmfulness scoring and mismatch detection show strong agreement, supporting the reliability of our evaluation framework.

## D Responsible Release

## D.1 Ethics and Data Release

MMJailBench contains harmful intents and adversarial prompts for multimodal safety evaluation. The benchmark is released for research purposes only, with documentation and evaluation tools to support reproducible and responsible use. Users are encouraged to apply the released resources for research on improving model safety and robustness.

<table><tr><td>Measure</td><td>Result</td></tr><tr><td>Mean ASR difference (Light – Full)</td><td>-0.06%</td></tr><tr><td>95%CI for ASR difference</td><td>[-0.45%, 0.30%]</td></tr><tr><td>Mean absolute ASR difference</td><td>0.79%</td></tr><tr><td>Maximum absolute ASR difference</td><td>3.53%</td></tr><tr><td>Cells within 2% difference</td><td>30/32 (93.75%)</td></tr><tr><td>Spearman ρ</td><td>0.997</td></tr><tr><td>Lin&#x27;s concordance correlation</td><td>0.999</td></tr></table>

Table 11: Consistency analysis between Full and Lightweight benchmark settings.
<table><tr><td rowspan="3">Evaluation Pair</td><td colspan="2">Harmfulness</td><td colspan="2">Mismatch Detection</td></tr><tr><td>QWK</td><td>Δ ASR</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td>GPT-5 vs. Human Experts</td><td>0.97</td><td>+0.3%</td><td>96.3%</td><td>0.85</td></tr><tr><td>Lightweight Judge vs. GPT-5</td><td>0.95</td><td>-0.4%</td><td>99.4%</td><td>0.94</td></tr></table>

Table 12: Judge agreement and validation results between GPT-5, human experts, and the lightweight judge.

## D.2 LLM Usage Statement

LLMs are used in MMJailBench for harmful-intent organization, visual prompt generation, and response evaluation. Generated materials are filtered and validated through predefined procedures, while the final benchmark design and protocols are determined by the authors.

## Supplementary References

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. LLaVA-OneVision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

Anthropic. Anthropic’s usage policy. https://www.anthropic. com/legal/aup, 2025a.

Anthropic. Claude sonnet 4.5 system card. Technical report, Anthropic, 2025b. URL https://www-cdn.anthropic.com/ 963373e433e489a87a10c823c52a0a013e9172dd/Claude% 20Sonnet%204.5%20System%20Card.pdf.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025b.

Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

ByteDance Seed Team. Seed 2.0 model card. Technical report, ByteDance, 2026. URL https://lf3-static.bytednsdoc.com/ obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2/0214/ Seed2.0%20Model%20Card.pdf.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. FigStep: Jailbreaking large vision-language models via typographic visual prompts. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 23951– 23959, 2025.

Google. Safety settings. https://ai.google.dev/gemini-api/docs/ safety-settings, 2026.

Google DeepMind. Gemini 3 flash model card. Technical report, Google DeepMind, 2025. URL https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. GLM-4.5V and GLM-4.1V-Thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, et al. STEP3-VL-10B technical report. arXiv preprint arXiv:2601.09668, 2026.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S. H. Cai, Yuan Cao, Y. Charles, H. S. Che, Cheng Chen, et al. Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, et al. Ministral 3. arXiv preprint arXiv:2601.08584, 2026.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. MM-SafetyBench: A benchmark for safety evaluation of multimodal large language models. In Proceedings of the European Conference on Computer Vision, pages 386–403, 2024.

Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, and Chaowei Xiao. JailBreakV: A benchmark for assessing the robustness of multimodal large language models against jailbreak attacks. arXiv preprint arXiv:2404.03027, 2024.

OpenAI. Usage policies. https://openai.com/zh-Hans-CN/ policies/usage-policies/, 2025.

OpenAI. Moderation. https://platform.openai.com/docs/guides/ moderation, 2026.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing, pages 3982–3992, 2019.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. A StrongREJECT for empty jailbreaks. arXiv preprint arXiv:2402.10260, 2024.

Qwen Team et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

xAI. Grok 4 fast model card. Technical report, xAI, 2025. URL https://data.x.ai/2025-09-19-grok-4-fast-model-card.pdf.

Z-Image Team, Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Shijie Huang, Zhaohui Hou, Dengyang Jiang, Xin Jin, et al. Z-Image: An eficient image generation foundation model with single-stream difusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.