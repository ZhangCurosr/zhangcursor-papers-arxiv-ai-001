# ACTR: ALIGNING THOUGHTS AND RESPONSES FOR MULTILINGUAL SAFETY IN REASONING LLMS

Xianhui Zhang<sup>1</sup>, Jian Yu<sup>1</sup>, Chengyu Xie<sup>1</sup>, Chenhang Cui<sup>2</sup>, Shuyi Miao<sup>3</sup>, Pengyang Shao<sup>2</sup>, Yu Zheng<sup>1</sup>, Fei Shen<sup>2,\*</sup>, Tat-Seng Chua<sup>2</sup>

<sup>1</sup>Nanjing University of Science and Technology, Nanjing, China <sup>2</sup>National University of Singapore, Singapore <sup>3</sup>Beihang University, Beijing, China

## ABSTRACT

Ensuring the safety of reasoning large language models (LLMs) across languages is essential for their reliable deployment. However, when exposed to jailbreak attacks in non-high-resource languages, these models may generate unsafe responses even when their reasoning traces identify safety risks. To address this issue, we propose aligning cross-lingual thoughts and responses (ACTR), a framework that improves multilingual safety alignment by strengthening the use of existing safety reasoning. Specifically, we first present the think gap score (TGS) to compare the normalized contributions of reasoning traces to attention outputs during response generation across languages, and use reasoningtrace substitution to measure the cross-lingual safety gap. Next, using a corpus ofjailbreak queries, we assess neuron importance through changes in response representations caused by neuron masking and compare the high-importance neuron sets obtained with reasoning enabled and disabled to identify safety think neurons that support the use of safety reasoning. Finally, we devise neuron-selective consistency optimization (NSCO), which uses a frozen judge model to reward agreement between the safety categories of reasoning traces and responses while updating only the parameters associated with the selected neurons, requiring no human-annotated responses or preference data. Across two reasoning models, ACTR achieves lower average attack success rates than the evaluated state-ofthe-art methods on AdvBench-X and MultiJail, with safety gains extending to unseen languages, while preserving or improving average performance on multilingual knowledge and mathematical reasoning tasks and limiting false refusals of benign requests. Warning: this paper contains examples with unsafe content.

## 1 Introduction

Reasoning large language models (LLMs) (Wei et al., 2022; Guo et al., 2025; Yoon et al., 2026) have made substantial progress on complex tasks by generating intermediate reasoning traces before their final responses. These traces (Li et al., 2025; Wang et al., 2025a) offer an opportunity to recognize risks and formulate safety judgments, whose practical value depends on whether they effectively guide the final response. Ensuring that such judgments remain effective across languages (Yang et al., 2025a) is therefore central to the safety alignment and reliable deployment of reasoning LLMs.

Existing LLM safety alignment (Ouyang et al., 2022; Rafailov et al., 2023; Bai et al., 2022) primarily shapes response behavior through supervised fine-tuning, preference learning, and reward optimization. Research on reasoning LLMs (Guan et al., 2024; Wang et al., 2025b) further incorporates safety specifications into the reasoning process to guide risk recognition and safety judgments before answering. In multilingual settings, existing work (Shen et al., 2024; Zhang et al., 2025) explores multilingual safety supervision and the transfer of safety knowledge from high-resource languages to extend alignment coverage. However, as illustrated in Figure 1, a jailbreak prompt in a non-high-resource (NHR) language (Deng et al., 2024; Yong et al., 2023) may elicit English reasoning that identifies safety risks, yet still lead to unsafe response in the target language. This disconnect shows that recognizing risks during reasoning does not ensure safe response generation, motivating alignment methods (Wei et al., 2023; Zou et al., 2023a; Qi et al., 2024a) that strengthen the use of existing safety judgments across languages.

To address this issue, we propose aligning cross-lingual thoughts and responses (ACTR), a framework that improves multilingual safety alignment by strengthening the use of existing safety reasoning. We first introduce the think gap score (TGS) to quantify cross-lingual differences in the normalized contribution of reasoning traces to attention outputs during response generation. Reasoning-trace substitution further tests whether reasoning content explains the safety gap: replacing NHR reasoning traces with those elicited by corresponding English prompts leaves NHR attack success rates largely unchanged. We then assess neuron importance through changes in response representations caused by masking and compare the high-importance neuron sets obtained with reasoning enabled and disabled under matched prompts to identify safety think neurons that support the use of reasoning information. Finally, ACTR applies neuron-selective consistency optimization (NSCO), which uses a frozen judge model to assign a single reward based on agreement between the safety categories of reasoning traces and final responses. The proposed ACTR updates only the parameters associated with the selected safety think neurons and requires no human-annotated responses or preference data. The main contributions are summarized as follows:

![](images/7a95988190715ba07f7a616c54153e4e311cbc9ef9a41c2a03dd2c26052be93c.jpg)  
Figure 1: Cross-lingual Thought–Response Safety Disconnect. Models may identify risks in English reasoning yet produce unsafe responses in NHR languages.

• We present TGS to quantify cross-lingual gaps in reasoning contributions and combine reasoning-trace substitution with neuron interventions to characterize disconnects between safety reasoning and responses, identifying safety think neurons supporting reasoning use.

• We propose ACTR and its NSCO strategy, which combines a model-judged thought-response consistency reward with selective updates to think-neuron parameters, enabling multilingual safety alignment without human-annotated responses or preference data.

• Across two reasoning models, ACTR achieves lower average attack success rates than the state-of-the-art methods on AdvBench-X and MultiJail, while generalizing to unseen languages and preserving average task performance with low refusal rates on benign requests.

## 2 Related Work

Safety Alignment in Reasoning LLMs. Safety alignment methods such as RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023) improve instruction-following and harmlessness, but primarily supervise final responses. Reasoning-oriented LLMs (Yang et al., 2025b; Team, 2026), including OpenAI o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025), generate explicit intermediate reasoning, and deliberative alignment (Guan et al., 2024) shows that reasoning over safety specifications can improve safety. Nevertheless, models aligned mainly in English (Yong et al., 2023; Zou et al., 2023b) remain vulnerable to prompts in non-high-resource languages. Existing defenses rely on multilingual supervision, data augmentation, or cross-lingual objectives, often requiring high-quality multilingual or parallel corpora that are costly and difficult to obtain.

Mechanistic Interpretability of Safety Behavior. Mechanistic interpretability (Geva et al., 2021; Meng et al., 2022; Arditi et al., 2024; Olah et al., 2020; Elhage et al., 2021) has identified circuits, attention heads, and neurons associated with factual recall, refusal, and other safety behaviors. Activation steering and probing (Turner et al., 2024; Zou et al., 2023a) further show that localized representations can influence model behavior. However, most studies examine single-stage generation. Reasoning LLMs instead generate responses after an intermediate think process, whose interaction with multilingual output remains poorly understood. Although multilingual chain-of-thought studies (Shi et al., 2022) show that reasoning and answer languages can interact, we investigate the think-response interface and identify neurons underlying its cross-lingual safety gap.

Reinforcement Learning for Safety. Reinforcement learning methods such as GRPO (Shao et al., 2024) improve reasoning and alignment through reward optimization. Existing safety methods (Ouyang et al., 2022; Bai et al., 2022) typically reward complete responses, as in RLHF and Constitutional AI, which can cause reward overoptimization (Gao et al., 2023; Röttger et al., 2024), excessive refusal, or reduced utility. In multilingual reasoning, a model (Yong et al., 2023; Guan et al., 2024) may produce safety-oriented thoughts but an inconsistent final response. However, optimizing final responses alone does not ensure that safety judgments formed during reasoning are effectively incorporated into multilingual responses.

3.1–3.2 Reasoning Utilization and Safety Think Neuron Identification  
![](images/7153396096244a530d9234f93f82d826064350bba7e1a966bc753cc1b8050263.jpg)  
Figure 2: Overview of our mechanistic analysis and ACTR framework. Top: TGS measures cross-lingual reasoning-utilization gaps; think-on/off comparisons localize safety think neurons. Bottom: NSCO uses consistency rewards from a frozen judge to selectively optimize these neurons’ parameters.

## 3 Mechanistic Analysis of the Cross-lingual Gap

## 3.1 The “Think-Response” Attention Disconnect

Models and Languages. We investigate two instruction-tuned large language models, Qwen3-8B (Yang et al., 2025b) and Gemma4-12B-it (Team, 2026), across seven languages: English (EN ), Chinese (ZH ), Korean (KO ), Thai (TH ), Afrikaans (AF ), Nepali (NE ), and Bengali (BN ). English is treated as the high-resource (HR) language, whereas the remaining languages are treated as non-high-resource (NHR) languages.

Motivation and Hypotheses. Reasoning LLMs often generate their internal reasoning traces in English, even when prompted in NHR languages. Nevertheless, they remain substantially more susceptible to jailbreak attacks in NHR languages than in HR languages. This discrepancy suggests two possible explanations: NHR prompts may elicit les safety-aware reasoning traces, or the model may fail to effectively incorporate otherwise safety-aware reasoning into its final responses. We first examine the former possibility through a controlled reasoning-trace substitution experiment.

Controlled Reasoning Trace Substitution. Specifically, we conduct a controlled HR-think substitution experiment to rule out differences in the content or quality of the generated reasoning traces as the primary cause of the degraded safety performance in NHR languages. For each jailbreak query, we first obtain the reasoning trace elicited by its HR-language version and then replace the NHR-generated trace with the one elicited by its HR counterpart. This intervention holds the reasoning content constant across the HR and NHR conditions, allowing us to isolate whether inferior NHR reasoning traces account for the observed safety gap. If they were the primary cause, replacing them with HR-derived traces should substantially reduce the discrepancy in attack success rate (ASR) (Qi et al., 2024b) between HR and NHR languages. ASR calculation details are in Appendix C.

Substitution Results. Table 1 reports ASR before and after substitution on MultiJail (Deng et al., 2024) across seven languages. Despite receiving the same English reasoning trace generated from the HR version of each jailbreak query, NHR ASR remains largely unchanged after substitution. Specifically, the seven-language mean ASR decreases by only 0.40 and 0.45 percentage points for Qwen3- 8B and Gemma4-12B-it, respectively. This result indicates that the multilingual safety gap cannot be explained solely by the quality or content of the generated reasoning trace. Instead, it points to a disconnect between the reasoning trace and response generation in NHR settings.

Quantifying Reasoning Trace Utilization. To quantify this utilization gap, we introduce the think gap score (TGS), which explicitly captures how much attention the final response allocates back to the reasoning steps. For each paired conversation x ∈ {HR, NHR}, let $\mathcal { I } _ { T } ^ { x }$ and $\mathcal { T } _ { A } ^ { x }$ denote the token positions of the reasoning trace and the final answer, respectively. We consider the latter half of the model’s L layers, indexed from zero:

Table 1: English Reasoning-Trace Substitution. ASR on MultiJail with original (Default) vs. English elicited traces (EN Think).
<table><tr><td rowspan="2">Language</td><td colspan="2">Qwen3-8B</td><td colspan="2">Gemma4-12B-it</td></tr><tr><td>Default</td><td>EN Think</td><td>Default</td><td>EN Think</td></tr><tr><td>EN米</td><td>11.75</td><td>9.21</td><td>4.76</td><td>4.13</td></tr><tr><td>ZH</td><td>9.21</td><td>10.48</td><td>6.67</td><td>6.03</td></tr><tr><td>KO</td><td>14.92</td><td>15.24</td><td>8.25</td><td>8.57</td></tr><tr><td>TH</td><td>13.33</td><td>11.43</td><td>6.98</td><td>6.98</td></tr><tr><td>AFY</td><td>15.56</td><td>14.92</td><td>9.84</td><td>8.89</td></tr><tr><td>NEE</td><td>27.62</td><td>27.94</td><td>7.62</td><td>6.67</td></tr><tr><td>BN 0</td><td>15.87</td><td>16.19</td><td>11.11</td><td>10.79</td></tr><tr><td>Avg.</td><td>15.46</td><td>15.06</td><td>7.89</td><td>7.44</td></tr></table>

$$
\mathcal { L } _ { \mathrm { h a l f } } = \left\{ \left\lfloor { \frac { L } { 2 } } \right\rfloor , \dots , L - 1 \right\} .\tag{1}
$$

For an answer token at position i, we use the attention row at position $i - 1$ , which predicts that token. Let $\mathcal { K } _ { i } ^ { x , \ell }$ denote the key positions visible to this row under the attention mask of layer ℓ. For any token-position set S, define its contribution to the projected attention output as

$$
\mathbf { o } _ { i , s } ^ { x , \ell } = W _ { O } ^ { \ell } \operatorname { C o n c a t } _ { h = 1 } ^ { H } \left( \sum _ { k \in S \cap \mathcal { K } _ { i } ^ { x , \ell } } A _ { h , i - 1 , k } ^ { x , \ell } \mathbf { V } _ { h , k } ^ { x , \ell } \right) ,\tag{2}
$$

where $A _ { h , i - 1 , k } ^ { x , \ell }$ is the attention weight, $\mathbf { V } _ { h , k } ^ { x , \ell }$ is the value vector associated with attention head $h ,$ and $W _ { O } ^ { \ell }$ is the output projection matrix, with its bias omitted. For grouped-query attention, value vectors are shared across the corresponding query heads. We measure the reasoning trace’s normalized contribution using the ratio of its projected output energy to the projected output energy from all visible tokens:

$$
r _ { i , \ell } ^ { x } = \frac { \left\| \mathbf { o } _ { i , \mathcal { L } _ { T } ^ { x } } ^ { x , \ell } \right\| _ { 2 } ^ { 2 } } { \left\| \mathbf { o } _ { i , \mathcal { K } _ { i } ^ { x , \ell } } ^ { x , \ell } \right\| _ { 2 } ^ { 2 } + \epsilon } ,\tag{3}
$$

where $\epsilon > 0$ ensures numerical stability. Averaging these ratios over answer positions and the selected layers gives

$$
R _ { \mathrm { t h i n k } } ^ { x } = \frac { 1 } { | \mathcal { T } _ { A } ^ { x } | | \mathcal { L } _ { \mathrm { h a l f } } | } \sum _ { \ell \in \mathcal { L } _ { \mathrm { h a l f } } } \sum _ { i \in \mathcal { T } _ { A } ^ { x } } r _ { i , \ell } ^ { x } .\tag{4}
$$

Finally, we define the think gap score for a paired HR/NHR conversation as

$$
\mathrm { T G S ^ { N H R } } | \mathrm { H R } \mathbf { \Phi } = R _ { \mathrm { t h i n k } } ^ { \mathrm { H R } } - R _ { \mathrm { t h i n k } } ^ { \mathrm { N H R } } .\tag{5}
$$

A positive TGS indicates a lower average normalized think-value contribution during answer prediction in the NHR conversation than in its paired HR conversation (see proof in Appendix B).

The Think–Response Attention Disconnect. To examine whether reduced safety accompanies weaker utilization of reasoning traces, we plot TGS as bars alongside benchmark ASR as lines for each language and model (Figure 3). Across both models, larger TGS generally accompanies higher ASR, although this relationship is not strictly monotonic. For example, Nepali in Qwen3-8B and Bengali in Gemma4-12B-it exhibit both the highest TGS and the highest MultiJail ASR among the evaluated languages. Together with the substitution results, these patterns suggest a think–response attention disconnect as a possible contributor to the multilingual safety gap. HR-think substitution leaves NHR ASR largely unchanged, while TGS reveals cross-lingual differences in the contribution of reasoning traces to attention outputs during response generation. English-dominant reasoning may therefore be insufficient when its safety-relevant information is not effectively incorporated into the final NHR response. This motivates targeted neuron interventions to test whether disrupting reasoning utilization also weakens multilingual safety.

![](images/608dafaac4df8e0f46f554f33c718d86c8cfd77337ce7294a753b71f81664bed.jpg)

![](images/fa4a1a1fca65bea9e7930d66be8b8d2b41656e3e0f461ff8ba50e2f7a5afb574.jpg)  
Figure 3: Reasoning Utilization and Multilingual Safety. Comparison of the think gap score (TGS) and attack success rate (ASR) across languages.

To further quantify the think–response mismatch, we measure the inconsistency rate for Qwen3-8B and Gemma4- 12B-it, defined as the proportion of examples in which the think trace is classified as safe while the final response is classified as unsafe. We report this rate separately on MultiJail and AdvBench-X. As shown in Table $^ { 2 , }$ the inconsistency rate generally increases for non-English languages. On MultiJail, it rises from only 0.24% in English to 6.67% in Nepali for Qwen3-8B. A similar pattern appears on AdvBench- $\cdot { \mathrm { X } } ,$ where the Qwen3-8B rate reaches 13.97% for Nepali and 8.89% for Bengali, compared with 0% for English and Chinese. Gemma4- 12B-it follows a similar trajectory, with its inconsistency rate rising from 1.90% in English to a peak of 6.67% in

Table 2: Think–Response Inconsistency Rate (%). The proportion of examples with safe think traces but unsafe responses on MultiJail and AdvBench-X.
<table><tr><td rowspan="2">Language</td><td colspan="2">Qwen3-8B</td><td colspan="2">Gemma4-12B-it</td></tr><tr><td>MultiJail</td><td>AdvBench-X</td><td>MultiJail</td><td>AdvBench-X</td></tr><tr><td>EN米</td><td>0.24</td><td>0.00</td><td>1.90</td><td>0.00</td></tr><tr><td>ZH</td><td>2.54</td><td>0.00</td><td>3.17</td><td>0.38</td></tr><tr><td>KO</td><td>3.81</td><td>3.81</td><td>5.40</td><td>1.35</td></tr><tr><td>TH</td><td>2.86</td><td>2.54</td><td>4.44</td><td>0.38</td></tr><tr><td>AF</td><td>4.76</td><td>4.44</td><td>6.03</td><td>1.35</td></tr><tr><td>BN</td><td>5.71</td><td>8.89</td><td>6.67</td><td>1.73</td></tr><tr><td>NE</td><td>6.67</td><td>13.97</td><td>3.81</td><td>0.77</td></tr></table>

Bengali on MultiJail, and from 0% to 1.73% on AdvBench-X. These results indicate that a safe reasoning trace does not always lead to a safe final response, especially in lower-resource languages where the model’s safety guardrails are inherently more fragile. Thus, the multilingual safety gap may partly arise from a failure to transfer safety relevant information from the reasoning trace into the response, providing additional evidence for the proposed think–response attention disconnect and highlighting a critical vulnerability in current cross-lingual alignment strategies.

## 3.2 Identifying and Defining Safety Think Neuron

Prior work (Dalvi et al., 2019; Meng et al., 2022) shows that individual neurons and sparse feature groups can encode interpretable concepts and influence model outputs, including safety responses and refusal behavior. Motivated by this ability of local structures to steer global model behavior, we hypothesize that some neurons serve as continuous mediators that preserve and propagate safety-relevant information from the initial internal think span to subsequent response tokens. We call these units safety think neurons. This section presents our neuron-level detection method and examines whether they exhibit distinct activation patterns under HR and NHR conditions, aiming to empirically validate their functional role in the reasoning pipeline.

Paired Think/No-Think Probing Dataset. We construct a separate multilingual probing dataset containing 700 jailbreak queries sampled from BeaverTails (Ji et al., 2023). For each query, we obtain its multilingual variants using Google Translate and retain only translations that pass an LLM-based semantic consistency check. To prevent data leakage, all queries are deduplicated against the evaluation examples in AdvBench-X and MultiJail, ensuring that none of the probing queries appears in either benchmark. For each query–language pair $q _ { j } ,$ we run the same model twice under identical prompt, system, and decoding settings. The only difference is whether think mode is enabled. The resulting model trajectories are denoted by

$$
s _ { j } ^ { \mathrm { o n } } = \left( q _ { j } , t _ { j } , a _ { j } ^ { \mathrm { o n } } \right) , \qquad s _ { j } ^ { \mathrm { o f f } } = \left( q _ { j } , a _ { j } ^ { \mathrm { o f f } } \right) ,\tag{6}
$$

where $t _ { j }$ is the reasoning trace generated in think mode, and $a _ { i } ^ { \mathrm { o n } }$ and $a _ { i } ^ { \mathrm { o f f } }$ are the corresponding responses. We retain all generated responses, including safe responses and unsafe responses. This paired construction holds the jailbreak query, language, and model constant while varying the availability of the reasoning trace. The resulting contrast therefore targets neurons involved in using the think span during response generation.

Identification Strategy. We define a neuron as a candidate think neuron if it is activated when think mode is enabled but not activated when think mode is disabled. Such neurons may be responsible for transmitting or using information from the model’s reasoning process. At layer ℓ, we define a neuron N as a specific row in one of the query, key, or value projection matrices, $\mathbf { \bar { W } } _ { Q } ^ { \ell } , \mathbf { W } _ { K } ^ { \ell } , \mathbf { o r } \mathbf { \bar { W } } _ { V } ^ { \ell }$ , or as a specific column in the output projection matrix $\dot { \mathbf { W } } _ { O } ^ { \ell }$ . For each trajectory $s _ { j } ^ { m }$ , where $m \in \{ \mathrm { o { \dot { n } } , o f f } \}$ , we identify the response-token positions $\mathcal { T } _ { A , j } ^ { m }$ . Let h $( s _ { j } ^ { m } , i )$ denote the output representation used to predict the response token at position i. We then deactivate neuron N during response generation and obtain the corresponding representation $\mathbf { h } _ { \theta , - N } \bigl ( s _ { j } ^ { m } , i \bigr )$ . The intervention is applied only at response-token positions, so that the resulting score measures the contribution of N to response generation. The importance of neuron N for example j under mode m is quantified by the average representational shift:

$$
\Delta _ { j } ^ { m } ( N ) = \frac { 1 } { \left| \mathcal { T } _ { A , j } ^ { m } \right| } \sum _ { i \in \mathcal { T } _ { A , j } ^ { m } } \left. \mathbf { h } _ { \theta } ( s _ { j } ^ { m } , i ) - \mathbf { h } _ { \theta , - N } ( s _ { j } ^ { m } , i ) \right. _ { 2 } ^ { 2 } .\tag{7}
$$

We aggregate this quantity across these paired queries to obtain the mode-specific importance score:

$$
I _ { \ell } ^ { m } ( N ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { p r o b e } } | } \sum _ { j = 1 } ^ { | \mathcal { D } _ { \mathrm { p r o b e } } | } \Delta _ { j } ^ { m } ( N ) ,\tag{8}
$$

where $\mathcal { D } _ { \mathrm { p r o b e } }$ denotes the same set of queries evaluated in both Think-on and Think-off modes. For each layer, we select the top p percent of neurons under each mode:

$$
\begin{array} { r } { S _ { \ell } ^ { m } = \mathrm { T o p } _ { p \% } \left\{ N \in \mathcal { N } _ { \ell } \vert I _ { \ell } ^ { m } ( N ) \right\} , } \end{array}\tag{9}
$$

where $\mathcal { N } _ { \ell }$ is the set of safety think neurons at layer ℓ. We set $p = 3$ to obtain a sparse set of highly influential candidate units. We then identify safety think neurons through mode-contrastive subtraction:

$$
\mathrm { T N } _ { \ell } = { \cal S } _ { \ell } ^ { \mathrm { o n } } \backslash { \cal S } _ { \ell } ^ { \mathrm { o f f } } .\tag{10}
$$

The resulting set comprises neurons highly influential when the reasoning trace is present but not in the matched Think-off condition. We therefore treat them as candidate safety think neurons for subsequent functional and causal analyses.

Causal Effect of Think-Neuron Masking. We next test whether the identified safety think neurons causally sustain the safety-relevant effect of the reasoning trace. For each model and language, we compare the unmasked default with two interventions: masking the detected safety think neurons and masking a control set of neurons matched for count and layerwise distribution.

Table 3 summarizes the ASR results. For Qwen3-8B, targeted masking raises ASR by 6.04 and 15.60 percentage points on AdvBench-X and MultiJail, respectively, whereas matched random masking changes it by only 0.38 and 0.41 points. This selective degradation indicates that the detected safety think neurons causally support model safety, beyond effects attributable to mask size or layer distribution. Additionally, we present qualitative examples of the model outputs before and after masking safety think neurons in Appendix F.

Table 3: Effects of Masking Safety Think Neurons. Mean ASR across languages under random (R-Masking) and targeted (TN-Masking) neuron masking.
<table><tr><td colspan="4">AdvBench-X</td></tr><tr><td>Model</td><td>Default</td><td>R-Masking</td><td>TN-Masking</td></tr><tr><td>Qwen3-8B</td><td>10.27</td><td> $1 0 . 6 6 ^ { + 0 . 3 8 }$ </td><td> $1 6 . 3 2 ^ { + 6 . 0 4 }$   $4 0 . 5 2 ^ { + 3 8 . 2 2 }$ </td></tr><tr><td>Gemma4-12B-it</td><td>2.30</td><td> $3 . 4 3 ^ { + 1 . 1 3 }$ </td><td></td></tr><tr><td colspan="4">MultiJail</td></tr><tr><td>Model</td><td></td><td>Default R-Masking</td><td>TN-Masking</td></tr><tr><td>Qwen3-8B</td><td>15.46</td><td> $1 5 . 8 7 ^ { + 0 . 4 1 }$ </td><td> $3 1 . 0 7 ^ { + 1 5 . 6 0 }$ </td></tr><tr><td>Gemma4-12B-it</td><td>7.89</td><td> $9 . 4 3 ^ { + 1 . 5 4 }$ </td><td> $5 1 . 7 9 ^ { + 4 3 . 9 0 }$ </td></tr></table>

Effects on Reasoning Utilization. Figure 4 reports TGS changes from baseline. Because higher TGS indicates less attention to think content, the consistent increase after thinkneuron masking across all NHR languages (mean +0.696; largest in Thai, +1.110) shows reduced reliance on preceding reasoning. Matched random masking causes negligible change (mean −0.045), ruling out the possibility that this effect is driven solely by the number or layer distribution of masked neurons. Together, the ASR and TGS results show that safety think neurons help connect the think span to the final response by carrying safety-related information from the reasoning process into response generation. Disrupting these neurons reduces the model’s use of this reasoning and makes it more vulnerable to jailbreak attacks.

![](images/306df8fff685d62007627bab90626ee01cc00c595b2d7f08d4e973b2c27225b2.jpg)

![](images/e73d2378c90bbd8b523289c6bc78611e5d039295e4a8edd7bf5166d93f47b741.jpg)  
Figure 4: Effects of Masking Safety Think Neurons on TGS. ∆TGS denotes the change relative to unmasked trajectories. Targeted masking increases TGS more than random masking.

## 3.3 The ACTR Framework

Motivation. Building on our mechanistic findings, we propose ACTR, a parameter-efficient reinforcement learning framework that improves multilingual safety alignment by strengthening the use of existing safety reasoning. Given a multilingual prompt x, the policy generates $y = ( t , a )$ , where t denotes the think span and a the final response. ACTR applies neuron-selective consistency optimization (NSCO), which uses a frozen judge model to reward thought–response safety consistency while updating only the parameters associated with the identified safety think neurons.

Neuron-Selective Consistency Optimization. Under neuron-selective consistency optimization (NSCO), the consistency rewards are normalized within each sampled group to obtain the relative consistency advantage: $y _ { i } = ( t _ { i } , a _ { i } ) \sim$ $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x )$ , where $i = 1 , \dots , G$ . A frozen binary consistency judge $C _ { \phi }$ evaluates each trajectory. We define the sequence-level consistency reward as

$$
\begin{array} { r } { r _ { i } ^ { \mathrm { c o n } } = C _ { \phi } ( t _ { i } , a _ { i } ) \in \{ 0 , 1 \} , } \end{array}\tag{11}
$$

where $r _ { i } ^ { \mathrm { c o n } } = 1$ indicates that the behavioral stance or action expressed in the response $a _ { i }$ is consistent with that implied by the think span $t _ { i } ,$ , while $r _ { i } ^ { \mathrm { c o n } } = 0$ indicates inconsistency. The judge evaluates semantic and behavioral agreement rather than surface-form similarity. The consistency rewards within each group are normalized to derive the relative advantages:

$$
\widehat { A } _ { i } = \frac { r _ { i } ^ { \mathrm { c o n } } - \mu _ { r } } { \sigma _ { r } + \epsilon _ { \mathrm { a d v } } } ,\tag{12}
$$

where µ and $\mu _ { r }$ $\sigma _ { r }$ are the mean and standard deviation of the G rewards, respectively. Let $s _ { i , k } = ( x , y _ { i , < k } )$ denote the decoding state at token position $k ,$ and let $\rho _ { i , k } = \pi _ { \boldsymbol { \theta } } ( y _ { i , k } \mid s _ { i , k } ) / \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( y _ { i , k } \mid s _ { i , k } )$ denote the corresponding likelihood ratio. Our ACTR framework optimizes:

$$
\begin{array} { r l } { \mathcal { I } _ { \mathrm { A C T R } } ( \theta _ { \mathrm { T N } } ) = \mathbb { E } _ { i , k } [ \ell _ { i , k } ^ { \mathrm { c l i p } } - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) ] , \ell _ { i , k } ^ { \mathrm { c l i p } } } & { = \operatorname* { m i n } ( \rho _ { i , k } \widehat { A } _ { i } , \mathrm { c l i p } ( \rho _ { i , k } , 1 - \epsilon , 1 + \epsilon ) \widehat { A } _ { i } ) . } \end{array}\tag{13}
$$

Unlike standard full-parameter optimization, ACTR applies gradients only to the parameters associated with the identified safety think neurons:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { T } _ { \mathrm { A C T R } } \gets M _ { \mathrm { T N } } \odot \nabla _ { \boldsymbol { \theta } } \mathcal { T } _ { \mathrm { A C T R } } .\tag{14}
$$

Table 4: Comparison of Attack Success Rate (ASR ↓, %) on AdvBench-X and MultiJail. ACTR achieves the lowest average ASR across both benchmarks and model families among the currently evaluated methods.
<table><tr><td></td><td colspan="5">AdvBench-X</td><td colspan="5">MultiJail</td></tr><tr><td>Method</td><td>EN米</td><td>ZH</td><td>KO</td><td>BN</td><td>Avg.</td><td>EN米</td><td>ZH</td><td>KO</td><td>BN</td><td>Avg.</td></tr><tr><td>Qwen3-8B</td><td>4.42</td><td>1.54</td><td>9.80</td><td>15.38</td><td>7.79</td><td>11.75</td><td>9.21</td><td>14.92</td><td>15.87</td><td>12.94</td></tr><tr><td>SFT</td><td>0.38-4.04</td><td>0.58-0.96</td><td>1.15-8.65</td><td>1.54-13.84</td><td>0.91-6.88</td><td>4.76-6.99</td><td>2.86-6.35</td><td>6.98-7.94</td><td>13.33-2.54</td><td>6.98-5.96</td></tr><tr><td>DPO</td><td>2.69-1.73</td><td>0.77-0.77</td><td>7.12-2.68</td><td>21.35+5.97</td><td>7.98+0.19</td><td>5.08-6.67</td><td>2.54-6.67</td><td>9.52-5.40</td><td>15.40-0.47</td><td>8.14-4.80</td></tr><tr><td>MPO</td><td>0.58-3.84</td><td>0.77-0.77</td><td>3.07-6.73</td><td>4.76-10.62</td><td>2.30-5.49</td><td>3.17-8.58</td><td>1.59-7.62</td><td>2.85-12.07</td><td>6.35-9.52</td><td>3.49-9.45</td></tr><tr><td>Self Defense</td><td>0.60-3.82</td><td>0.80-0.74</td><td>4.00-5.80</td><td>16.20+0.82</td><td>5.40-2.39</td><td>3.50-8.25</td><td>4.40-4.81</td><td>7.00-7.92</td><td>16.80+0.93</td><td>7.93-5.01</td></tr><tr><td>SmoothLLM</td><td>0.40-4.02</td><td>0.58-0.96</td><td>4.80-5.00</td><td>9.20-6.18</td><td>3.75-4.04</td><td>5.08-6.67</td><td>1.00-8.21</td><td>4.10-10.82</td><td>6.70-9.17</td><td>4.22-8.72</td></tr><tr><td>TrajGuard</td><td>2.31-2.11</td><td>1.54-0.00</td><td>9.62-0.18</td><td>14.62-0.76</td><td>7.02-0.77</td><td>4.13-7.62</td><td>2.86-6.35</td><td>11.75-3.17</td><td>20.95+5.08</td><td>9.92-3.02</td></tr><tr><td>GUARD-SLM</td><td>0.38-4.04</td><td>0.77-0.77</td><td>4.42-5.38</td><td>7.69-7.69</td><td>3.32-4.47</td><td>4.76-6.99</td><td>2.54-6.67</td><td>10.16-4.76</td><td>17.14+1.27</td><td>8.65-4.29</td></tr><tr><td>Ours</td><td>| 0.19-4.23</td><td>0.38-1.16</td><td>0.19-9.61</td><td>0.38-15.00</td><td>0.29-7.50</td><td>|0.63-11.12</td><td>0.95-8.26</td><td>1.90-13.02</td><td>4.13-11.74</td><td>1.90-11.04</td></tr><tr><td>Gemma4-12B-it</td><td>0.96</td><td>1.34</td><td>2.11</td><td>3.46</td><td>1.97</td><td>4.76</td><td>6.67</td><td>8.25</td><td>11.11</td><td>7.70</td></tr><tr><td>SFT</td><td>0.58-0.38</td><td>1.15-0.19</td><td>1.92-0.19</td><td>3.08-0.38</td><td>1.68-0.29</td><td>5.40+0.64</td><td>2.86-3.81</td><td>4.76-3.49</td><td>3.81-7.30</td><td>4.21-3.49</td></tr><tr><td>DPO</td><td>0.77-0.19</td><td>0.96-0.38</td><td>2.31+0.20</td><td>2.31-1.15</td><td>1.59-0.38</td><td>3.49-1.27</td><td>4.76-1.91</td><td>4.76-3.49</td><td>3.17-7.94</td><td>4.05-3.65</td></tr><tr><td>MPO</td><td>0.38-0.58</td><td>0.58-0.76</td><td>1.73-0.38</td><td>2.69-0.77</td><td>1.35-0.62</td><td>2.22-2.54</td><td>2.53-4.14</td><td>1.27-6.98</td><td>1.59-9.52</td><td>1.90-5.80</td></tr><tr><td>Self Defense</td><td>0.58-0.38</td><td>1.70+0.36</td><td>3.50+1.39</td><td>2.69-0.77</td><td>2.12+0.15</td><td>3.80-0.96</td><td>5.40-1.27</td><td>5.70-2.55</td><td>3.20-7.91</td><td>4.53-3.17</td></tr><tr><td>SmoothLLM</td><td>0.96-0.00</td><td>0.60-0.74</td><td>1.73-0.38</td><td>2.30-1.16</td><td>1.40-0.57</td><td>1.00-3.76</td><td>3.50-3.17</td><td>3.80-4.45</td><td>1.59-9.52</td><td>2.47-5.23</td></tr><tr><td>TrajGuard</td><td>1.15+0.19</td><td>2.12+0.78</td><td>2.50+0.39</td><td>1.73-1.73</td><td>1.88-0.09</td><td>5.71+0.95</td><td>4.76-1.91</td><td>5.08-3.17</td><td>4.76-6.35</td><td>5.08-2.62</td></tr><tr><td>GUARD-SLM</td><td>0.38-0.58</td><td>0.96-0.38</td><td>1.54-0.57</td><td>1.92-1.54</td><td>1.20-0.77</td><td>1.59-3.17</td><td>2.53-4.14</td><td>1.59-6.66</td><td>1.90-9.21</td><td>1.90-5.80</td></tr><tr><td>Ours</td><td>0.20-0.76</td><td>0.00-1.34</td><td>1.35-0.76</td><td>1.35-2.11</td><td>0.72-1.23</td><td>1.27-3.49</td><td>0.95-5.72</td><td>0.63-7.62</td><td>1.27-9.84</td><td>1.03-6.67</td></tr></table>

Here, $M _ { \mathrm { T N } }$ is a binary mask that selects the parameters associated with the safety think neurons. Thus, only $\theta _ { \mathrm { T N } }$ is updated, while the remaining backbone parameters, the reference policy $\pi _ { \mathrm { r e f } }$ , and the consistency judge $C _ { \phi }$ remain frozen throughout training.

## 4 Experiments

Training Details. We train ACTR independently on two instruction-tuned reasoning LLMs: Qwen3-8B (Yang et al., 2025b) and Gemma4-12B-it (Team, 2026). For each model, training covers six NHR languages: Chinese, Korean, Thai, Afrikaans, Nepali, and Bengali. We use the multilingual probing dataset introduced in Section 3.2. Only the default jailbreak queries serve as training prompts; the current rollout policy samples a new group of G thought–response trajectories for each query. Model-specific hyperparameters are provided in Appendix D. For fair comparison, we use matched training settings across the compared training-based methods. We periodically evaluate each model during training and select its best-performing checkpoint based on validation results.

Evaluation and Metrics. We use GPT-4o-mini as a consistency reward model to evaluate sampled trajectories online and assign binary rewards. A trajectory receives a positive reward when its final response agrees with the safety stance expressed in the think span, regardless of whether the response itself is safe. The judging criteria are detailed in Appendix E. For safety evaluation, we use attack success rate (ASR), with detailed evaluation protocols provided in Appendix C.

## 4.1 Main Results

Table 4 compares ACTR with representative alignment and inference-time defense methods, including SFT, DPO (Rafailov et al., 2023), MPO (Zhao et al., 2025), Self Defense (Phute et al., 2024), SmoothLLM (Robey et al., 2024), and the recently proposed TrajGuard (Liu et al., 2026) and GUARD-SLM (Mia et al., 2026). Since TrajGuard and GUARD-SLM employ refusal-based defenses, we treat an explicit refusal as a safe outcome when computing their ASR; for other methods, safety is determined directly from the model’s generated response. Under this evaluation protocol, ACTR achieves the lowest average ASR for both model families on both benchmarks. For Qwen3-8B, ACTR reduces the average ASR to 0.29% and 1.90% on AdvBench-X and MultiJail, compared with 0.91% and 3.49% for the strongest competing methods. For Gemma4-12B-it, ACTR further achieves 0.72% on AdvBench-X (Yong et al., 2023) and 1.03% on MultiJail (Deng et al., 2024), improving over the strongest baselines, GUARD-SLM on AdvBench-X and the tied MPO/GUARD-SLM baselines on MultiJail, by 40.0% and 45.8%, respectively. Overall, these results demonstrate that ACTR provides robust improvements across model architectures, benchmarks, and languages, suggesting that targeted think-neuron updates more effectively connect internal safety reasoning with the final response than existing alignment and inference-time defense methods.

## 4.2 Ablation study

Selection Threshold $p .$ To validate $p = 3$ in Eq. 9, we evaluate general capabilities on MGSM (Shi et al., 2022) and safety performance on MultiJail (Deng et al., 2024) across $p \in \{ 1 , 2 , 3 , 5 , 7 \}$ . We aim to isolate neurons that utilize safety reasoning without compromising general utility during targeted interventions. As Figure 5 shows, the model maintains near-baseline MGSM scores when $p \leq 3 \%$ . However, general capabilities degrade sharply for p > 3% (e.g., at p = 5, MGSM drops by 83.60 for Gemma4-12B-it), indicating that higher thresholds incorrectly capture neurons essential for foundational reasoning and task execution. Thus, $p = 3$ optimally isolates a sparse and highly specific set of candidate safety think neurons while reliably preserving general performance, ensuring that our downstream modifications remain minimally invasive.

Alignment Strategy. To determine whether the safety gains come from sparse updates or the specific neurons identified by our probing procedure, we train a random-neuron baseline. It selects the same number of neurons and uses the same update strategy, training data, optimization settings, and trainable parameter budget as ACTR. As shown in Table 5, both strategies substantially reduce ASR relative to the original models, with only minor differences. ACTR performs slightly better overall: for Qwen3-8B, it achieves lower ASR on AdvBench-X and comparable results on MultiJail; for Gemma4-12Bit, it obtains ASR of 0.93% and 1.72% on AdvBench-X and MultiJail, respectively.

![](images/84e87907480ebe045c3af7830327604a79f009766540d219a82c6b6f040dbf50.jpg)  
Figure 5: Effect of Selection Threshold $p .$ MGSM accuracy and ASR (%) after neuron masking.

Table 5: Effect of Neuron Selection. Mean ASR (↓, %) across languages after updating random neurons or safety think neurons.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>| Setting</td><td rowspan=1 colspan=1>AdvBench-X MultiJail</td></tr><tr><td rowspan=1 colspan=1>Qwen3-8B</td><td rowspan=1 colspan=1>DefaultRandomACTR</td><td rowspan=1 colspan=1>11.25       16.08 $1 . 0 3 ^ { - 1 0 . 2 2 }$    $1 . 3 2 ^ { - 1 4 . 7 6 }$  $\mathbf { 0 . 4 5 ^ { - 1 0 . 8 0 } }$    $\pmb { 1 . 2 7 } ^ { - 1 4 . 8 1 }$ </td></tr><tr><td rowspan=2 colspan=1>Gemma4-12B-it</td><td rowspan=1 colspan=1>DefaultRandom</td><td rowspan=2 colspan=1>2.31        7.89 $1 . 1 3 ^ { - 1 . 1 8 }$     $1 . 8 6 ^ { - 6 . 0 3 }$  $\mathbf { 0 . 9 3 ^ { - 1 . 3 8 } }$     $\mathbf { 1 . 7 2 ^ { - 6 . 1 7 } }$ </td></tr><tr><td rowspan=1 colspan=1>ACTR</td></tr></table>

Over-Refusal. Low ASR may also result from excessive refusal and

therefore does not fully capture a model’s ability to answer safe requests. We further evaluate XSTest, which includes both safe and unsafe requests. As shown in Table 6, random-neuron training raises Qwen3-8B’s safe-request refusal rate from 4.40% to 52.00%, whereas ACTR raises it only to 6.40% while achieving higher unsafe-request refusal (89.50% vs. 86.00%). For Gemma4-12B-it, ACTR keeps safe-request refusal nearly unchanged (4.80% to 4.84%) while increasing unsafe-request refusal from 82.00% to 89.20%. These results show that ACTR improves safety while remaining responsive to safe requests, avoiding the pronounced over-refusal observed with random-neuron training.

Table 7: Utility Preservation. Accuracy (↑, %) on MMMLU and MGSM before and after ACTR training.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="6">MMMLU</td><td rowspan="2">MGSM</td><td colspan="6"></td></tr><tr><td>EN米</td><td>ZH</td><td>JA●</td><td>KO</td><td>TH=</td><td>Avg.</td><td>EN米</td><td>ZH</td><td>JA●</td><td>KO</td><td>TH≡</td><td>Avg.</td></tr><tr><td rowspan="2">Qwen3-8B</td><td>Default</td><td>70.36</td><td>63.84</td><td>60.25</td><td></td><td>55.48</td><td>25.63 55.11</td><td></td><td>96.8</td><td>85.2</td><td>81.6</td><td>76.0</td><td>85.2</td><td>84.96</td></tr><tr><td>Ours</td><td>69.28</td><td>64.32</td><td>59.04</td><td></td><td>57.61</td><td>26.98</td><td>55.45</td><td>96.4</td><td>90.8</td><td>86.4</td><td>82.4</td><td>90.0</td><td>89.20</td></tr><tr><td rowspan="2">Gemma4-12B-it</td><td>Default</td><td>50.98</td><td>49.27</td><td>52.67</td><td></td><td>53.87</td><td>4.86</td><td>42.33</td><td>98.4</td><td>90.4</td><td>90.4</td><td>87.2</td><td>92.4</td><td>91.76</td></tr><tr><td>Ours</td><td>64.41</td><td>54.02</td><td>58.30</td><td></td><td>59.74</td><td>8.37</td><td>48.97</td><td>98.4</td><td>90.8</td><td>89.2</td><td>89.2</td><td>92.0</td><td>91.92</td></tr></table>

## 4.3 Utility Preservation

To evaluate whether safety-oriented training preserves general-purpose capabilities, we conduct experiments on MMMLU (Hendrycks et al., 2020) and MGSM (Shi et al., 2022). MMMLU measures multilingual knowledge and reasoning across a broad range of academic subjects, while MGSM evaluates multilingual grade-school mathematical reasoning. For both benchmarks, we report accuracy (%) in five languages: English, Chinese, Japanese, Korean, and Thai. Table 7 shows that our method does not degrade general capabilities and instead yields modest overall improvements. For Qwen3-8B (Yang et al., 2025b), the average accuracy increases from 55.11% to 55.45% on MMMLU and from 84.96% to 89.20% on MGSM. For Gemma4-12B-it, it increases from 42.33% to 48.97% on MMMLU and from 91.76% to 91.92% on MGSM.

Table 6: Refusal Behavior on XSTest. Refusal rates (%) for benign and harmful requests under different neuron-selection strategies.
<table><tr><td>Model</td><td>| Setting</td><td>Safe refusal ↓ Unsafe refusal ↑</td><td></td></tr><tr><td rowspan="3">Qwen3-8B</td><td>|Default</td><td>4.40</td><td>77.00</td></tr><tr><td>Random</td><td> $5 2 . 0 0 ^ { + 4 7 . 6 0 }$ </td><td> $8 6 . 0 0 ^ { + 9 . 0 0 }$ </td></tr><tr><td>ACTR</td><td> ${ \bf 6 . 4 0 ^ { + 2 . 0 0 } }$ </td><td> $\mathbf { 8 9 . 5 0 ^ { + 1 2 . 5 0 } }$ </td></tr><tr><td rowspan="3">Gemma4-12B-it</td><td>|Default</td><td>4.80</td><td>82.00</td></tr><tr><td>Random</td><td> $4 6 . 8 0 ^ { + 4 2 . 0 0 }$ </td><td> $8 3 . 2 0 ^ { + 1 . 2 0 }$ </td></tr><tr><td>ACTR</td><td> $\pmb { 4 . 8 4 } ^ { \qquad 0 . 0 4 }$ </td><td>89.20+7.20</td></tr></table>

## 4.4 Deeper Analysis

Out-of-Distribution Generalization. To test whether the safety improvements transfer beyond the languages seen during training, we evaluate both models on four held-out languages: Swahili (SW ) Javanese (JV ), Bulgarian (BG ), and Vietnamese (VI ). None of these languages is included in the training data. Table 8 reports the ASR on MultiJail examples per language before and after training; lower values indicate safer behavior. The results show consistent safety

Table 8: Safety Generalization to Unseen Languages.
<table><tr><td>Model</td><td>|Setting</td><td>|SW Z</td><td>JV 1</td><td>BG _</td><td>VI★</td></tr><tr><td>Qwen3-8B</td><td>Default Ours</td><td>62.86 1.27</td><td>12.70 0.63</td><td>11.75 0.95</td><td>6.67 2.22</td></tr><tr><td>Gemma4-12B-it</td><td>Default| Ours</td><td>8.25 3.49</td><td>10.79 8.57</td><td>10.79 5.40</td><td>7.62 5.08</td></tr></table>

gains in out-of-distribution languages. Qwen3-8B’s overall ASR drops from 23.5% to 1.3%; in Swahili, ASR falls from 62.86% to 1.27%. Gemma4-12B-it shows similar improvements, reducing ASR from 9.4% to 5.6%. These results suggest that the method learns language-agnostic safety behavior rather than memorizing language-specific refusal patterns.

Training-Induced Changes in TGS. To assess whether ACTR reduces the cross-lingual gap in internal reasoning, we compare the TGS before and after training. TGS measures the difference between English and NHR languages in the normalized contribution of think content, with lower values indicating a smaller gap. As shown in Fig. 6, ACTR reduces TGS across all six NHR languages for both models. For Qwen3-8B, mean TGS drops from 4.05 to 1.91 (↓52.7%); for Gemma4-12B-it, from 3.79 to 2.07 (↓45.3%). Combined with the safety gains, these results suggest that strengthening safety think neurons promotes internal safety reasoning in NHR responses, mitigating the think–response disconnect.

Case Analysis. Complementing the quantitative results, Figure 7 shows that ACTR resolves the thought-response disconnect seen in earlier Korean and Thai examples. Before training, models reasoned safely in English but responded unsafely in the target language. After ACTR, both the reasoning and final responses consistently express refusal. Alongside reduced TGS and ASR scores, this confirms that ACTR effectively preserves safety judgments across languages. More results can be found in the appendix.

![](images/9d59dc185d3dc4e324175d5020700e81a5c89778d2c558952ebd1d376704e85c.jpg)  
Figure 6: TGS Before and after ACTR Training. ACTR lowers TGS across all evaluated NHR languages, indicating smaller cross-lingual gaps in reasoning utilization.

![](images/303b8b30f1d2b7f58daca28246403ccaff124d761781ce947003fbe54d6d1133.jpg)  
Figure 7: Thought–Response Safety Consistency after ACTR training.

## 5 Conclusion

We investigate the multilingual thought–response safety disconnect in reasoning LLMs through TGS, reasoning-trace substitution, and neuron interventions, identifying safety think neurons that support safety reasoning utilization. Building on this, ACTR applies NSCO, combining a thought–response consistency reward with parameter updates restricted to these neurons, eliminating the need for human-annotated preference data. Experiments on Qwen3-8B and Gemma4-12B-it show ACTR achieves lower average attack success rates than evaluated state-of-the-art methods on AdvBench-X and MultiJail, with safety gains generalizing to unseen languages. Furthermore, ACTR preserves general utility (MMMLU, MGSM) while maintaining low safe-request refusal rate.

## AI use statement

Generative AI tools were used to review the LaTeX formatting during manuscript preparation. To ensure consistency with the historical ASR computation methodology, GPT-4o was also used to calculate ASR. GPT-4o-mini was used as the frozen consistency judge described in Section 4. The authors are responsible for all claims, analyses, and final content.

## Ethics statement

This work studies multilingual jailbreak vulnerabilities to improve model safety. The same mechanistic intervention can weaken refusal behavior, so the findings should be interpreted in the context of defensive evaluation.

## Reproducibility statement

We provide the details required to reproduce our results. Section 3 defines the think gap score, the construction of the multilingual probing dataset, and the think neuron identification and intervention procedures. Section 3.3 specifies the consistency reward, the masked NSCO objective, and the training protocol. Appendix D reports the implementation and hyperparameter settings, while Appendix E documents the consistency-judge prompt, decision rule, and output parsing procedure.

## References

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chainof-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Dongkeun Yoon, Seungone Kim, Sohee Yang, Sunkyoung Kim, Soyeon Kim, Yongil Kim, Eunbi Choi, Yireun Kim, and Minjoon Seo. Reasoning models better express their confidence. Advances in Neural Information Processing Systems, 38:103869–103896, 2026.

Changyi Li, Jiayi Wang, Xudong Pan, Geng Hong, and Min Yang. Reasoningshield: Safety detection over reasoning traces of large reasoning models. arXiv preprint arXiv:2505.17244, 2025.

Cheng Wang, Yue Liu, Baolong Bi, Duzhen Zhang, Zhong-Zhi Li, Yingwei Ma, Yufei He, Shengju Yu, Xinfeng Li, Junfeng Fang, et al. Safety in large reasoning models: A survey. In EMNLP (Findings), pages 3468–3482, 2025a.

Yahan Yang, Soham Dan, Shuo Li, Dan Roth, and Insup Lee. Mrguard: A multilingual reasoning guardrail for universal llm safety. In Proceedings of the 2025 conference on empirical methods in natural language processing, pages 27365–27384, 2025a.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Melody Y Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Helyar, Rachel Dias, Andrea Vallone, Hongyu Ren, Jason Wei, et al. Deliberative alignment: Reasoning enables safer language models. arXiv preprint arXiv:2412.16339, 2024.

Haoyu Wang, Zeyu Qin, Li Shen, Xueqian Wang, Dacheng Tao, and Minhao Cheng. Safety reasoning with guidelines. arXiv preprint arXiv:2502.04040, 2025b.

Lingfeng Shen, Weiting Tan, Sihao Chen, Yunmo Chen, Jingyu Zhang, Haoran Xu, Boyuan Zheng, Philipp Koehn, and Daniel Khashabi. The language barrier: Dissecting safety challenges of llms in multilingual contexts. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 2668–2680, 2024.

Zekai Zhang, Yiduo Guo, Jiuheng Lin, Shanghaoran Quan, Huishuai Zhang, and Dongyan Zhao. English as defense proxy: Mitigating multilingual jailbreak via eliciting english safety knowledge. In EMNLP (Findings), pages 1185–1196, 2025.

Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing. Multilingual jailbreak challenges in large language models. In International Conference on Learning Representations, volume 2024, pages 24634–24651, 2024.

Zheng-Xin Yong, Cristina Menghini, and Stephen H Bach. Low-resource languages jailbreak gpt-4. arXiv preprint arXiv:2310.02446, 2023.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? Advances in neural information processing systems, 36:80079–80110, 2023.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023a.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! In International Conference on Learning Representations, volume 2024, pages 30988–31043, 2024a.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025b.

Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607.02770.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023b.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 5484–5495, 2021.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. Advances in Neural Information Processing Systems, 37: 136037–136083, 2024.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 5(3):e00024–001, 2020.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization. 2024.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, et al. Language models are multilingual chain-of-thought reasoners. arXiv preprint arXiv:2210.03057, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International conference on machine learning, pages 10835–10866. PMLR, 2023

Paul Röttger, Hannah Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. In Proceedings of the 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5377–5400, 2024.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id=hTEGyKf0dZ.

Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov, Anthony Bau, and James Glass. What is one grain of sand in the desert? analyzing individual neurons in deep nlp models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 6309–6317, 2019.

Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset. Advances in Neural Information Processing Systems, 36:24678–24704, 2023.

Weixiang Zhao, Yulin Hu, Yang Deng, Tongtong Wu, Wenxuan Zhang, Jiahe Guo, An Zhang, Yanyan Zhao, Bing Qin, Tat-Seng Chua, et al. Mpo: Multilingual safety alignment via reward gap optimization. arXiv preprint arXiv:2505.16869, 2025.

Mansi Phute, Alec Helbling, Matthew Daniel Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, and Duen Horng Chau. Llm self defense: By self examination, llms know they are being tricked. In The Second Tiny Papers Track at ICLR 2024, 2024.

Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas. Smoothllm: Defending large language models against jailbreaking attacks. Transactions on Machine Learning Research, 2024.

Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin, and Kangyi Ding. Trajguard: Streaming hidden-state trajectory detection for decoding-time jailbreak defense. In Findings of the Association for Computational Linguistics: ACL 2026, pages 13371–13388, 2026.

Md Jueal Mia, Joaquin Molto, Yanzhao Wu, and M Hadi Amini. Guard-slm: Token activation-based defense against jailbreak attacks for small language models. arXiv preprint arXiv:2603.28817, 2026.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. How johnny can persuade LLMs to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing LLMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14322–14350, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.773. URL https://aclanthology.org/2024. acl-long.773/.

## APPENDIX

The appendix is organized as follows. Appendix A summarizes the important notation used in our paper. Appendix B provides a theoretical justification for the Think Gap Score (TGS). Appendix C details the computation of the attack success rate (ASR) metric and the automated evaluation protocol. Appendix D provides the implementation details, including hyperparameters and training configurations. Appendix E describes the prompt and the binary reward formulation used for the consistency judge. Appendix G discusses the limitations of our current evaluation and future directions regarding code-switching. Appendix F shows the performance of the model before and after Mask safety think neurons. Finally, Appendix H presents additional discussions.

## A Notation Summary

Table 9 summarizes the notation used to quantify reasoning trace utilization, identify safety think neurons, and formulate the ACTR framework. A superscript l indexes a model layer and does not denote exponentiation.

Table 9: Core Notation for Reasoning Utilization, Safety Think Neurons, and the ACTR Framework.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $L , { \mathcal { L } } _ { h a l f }$ </td><td>Model layers, indexed from zero, and the latter half of the model&#x27;s L layers.</td></tr><tr><td> $x$ </td><td>Language condition, where  $x \ \in \ \{ \mathrm { H R } , \mathrm { N H R } \}$  denotes high-resource and non-high- resource languages.</td></tr><tr><td> $\mathcal { T } _ { T } ^ { x } , \mathcal { T } _ { A } ^ { x }$   $\mathcal { K } _ { i } ^ { x , l }$ </td><td>Token positions of the reasoning trace and the final answer, respectively. Key positions visible to the attention row predicting the token at position i under the</td></tr><tr><td></td><td>attention mask of layer l.</td></tr><tr><td> $o _ { i . S } ^ { x , l }$   $\hat W _ { Q } ^ { l } , W _ { K } ^ { l } , W _ { V } ^ { l } , W _ { O } ^ { l }$ </td><td>Contribution of token-position set S to the projected attention output. Query, key, value, and output projection matrices at layer l.</td></tr><tr><td> $r _ { i , t } ^ { x } , R _ { t h i n k } ^ { x }$ </td><td>Normalized contribution of the reasoning trace and its average over answer positions and</td></tr><tr><td> $T G S$ </td><td>selected layers. Think gap score, measuring the cross-lingual difference in the normalized contribution of</td></tr><tr><td> $s _ { j } ^ { o n } , s _ { j } ^ { o f f }$ </td><td>reasoning traces during answer prediction. Model trajectories for query  $j$  with think mode enabled and disabled, respectively.</td></tr><tr><td> $t _ { j } , a _ { j } ^ { o n } , a _ { j } ^ { o f f }$ </td><td>Reasoning trace generated in think mode  $t _ { j } ,$  and the corresponding responses  $a _ { j } ^ { o n }$  and</td></tr><tr><td> $\Delta _ { j } ^ { m } ( N ) , I _ { l } ^ { m } ( N )$ </td><td> $a _ { j } ^ { o f f }$ </td></tr><tr><td> $S _ { l } ^ { m } , T N _ { l }$ </td><td>Importance of neuron N for example j under mode  $m ,$  and its aggregated mode-specific importance score.</td></tr><tr><td></td><td>The top p percent of neurons under mode m at layer l, and the identified set of safety think neurons.</td></tr><tr><td> $C _ { \phi } , r _ { i } ^ { c o n }$ </td><td>Frozen binary consistency judge and the sequence-level consistency reward.</td></tr><tr><td> ${ \hat { A } } _ { i }$ </td><td>Relative consistency advantage normalized within each sampled group.</td></tr><tr><td> $M _ { T N } , \mathcal { T } _ { A C T R }$ </td><td>Binary mask selecting the parameters associated with the safety think neurons, and the ACTR objective function.</td></tr></table>

## B Theoretical Justification of the Think Gap Score (TGS)

In Section 3.1, we introduced the Think Gap Score (TGS) to quantify the think–response attention disconnect. Here, we provide a theoretical justification for why a higher TGS (i.e., a drop in $R _ { \mathrm { t h i n k } } ^ { \mathrm { { N H R } } }$ relative to $R _ { \operatorname { t h i n k } } ^ { \mathrm { H R } } )$ leads to degraded safety performance (higher ASR) under non-high-resource languages.

Assumption 1 (Linear Representation of Safety). Following prior mechanistic interpretability findings on linear concept representations (Meng et al., 2022; Zou et al., 2023a), we assume the existence of a linear “safety direction” ${ \bf w } _ { \mathrm { s a f e } }$ in the residual stream. A larger projection of the final hidden state ${ \bf h } _ { i } ^ { x , L }$ onto ${ \bf w } _ { \mathrm { s a f e } }$ increases the logit of generating

a safe/refusal token at position i:

$$
P ( { \mathrm { S a f e ~ T o k e n } } ) \propto \exp \left( \langle \mathbf { h } _ { i } ^ { x , L } , \mathbf { w } _ { \mathrm { s a f e } } \rangle \right) .\tag{15}
$$

Assumption 2 (Safety Information is Encoded in the Think Trace). Based on our reasoning-trace substitution experiment (Table 1), we establish that the think traces for both HR and NHR queries correctly identify safety risks. Thus, the attention output sourced from the think span, $\mathbf { o } _ { i , \mathcal { T } _ { T } ^ { x } } ^ { x , \ell }$ , contains a strong component along the safety direction ${ \bf w } _ { \mathrm { s a f e } }$ . In contrast, the attention output sourced from the adversarial query itself (non-think tokens, denoted as $\mathcal { T } _ { \backslash T } ^ { x } = { \mathcal { K } _ { i } ^ { x , \ell } \setminus \mathcal { T } _ { T } ^ { x } } )$ aligns with the harmful intent, having a negative or zero projection on ${ \bf w } _ { \mathrm { s a f e } }$

Derivation. In a transformer, the hidden state used to predict the token at position i is updated additively by the attention and MLP layers. Focusing on the cumulative attention updates from the latter half layers ${ \mathcal { L } } _ { \mathrm { h a l f } }$ (which directly steer the final output vocabulary distribution), the state update is:

$$
\Delta \mathbf { h } _ { i , \mathrm { a t t n } } ^ { x } \approx \sum _ { \ell \in \mathcal { L } _ { \mathrm { h a l f } } } \mathbf { o } _ { i ,  { K } _ { i } ^ { x , \ell } } ^ { x , \ell } = \sum _ { \ell \in \mathcal { L } _ { \mathrm { h a l f } } } \left( \mathbf { o } _ { i ,  { \mathcal L } _ { T } ^ { x } } ^ { x , \ell } + \mathbf { o } _ { i ,  { \mathcal L } _ { \setminus T } ^ { x } } ^ { x , \ell } \right) .\tag{16}
$$

The total safety steering signal $S ^ { x }$ added to the residual stream is the projection of this update onto $\mathbf { w } _ { \mathrm { s a f e } } .$

$$
S ^ { x } = \langle \Delta \mathbf { h } _ { i , \mathrm { a t t n } } ^ { x } , \mathbf { w } _ { \mathrm { s a f e } } \rangle = \sum _ { \ell \in \mathcal { L } _ { \mathrm { h a l f } } } \left( \langle \mathbf { o } _ { i , \mathcal { L } _ { T } ^ { x } } ^ { x , \ell } , \mathbf { w } _ { \mathrm { s a f e } } \rangle + \langle \mathbf { o } _ { i , \mathcal { L } _ { \mathrm { V } } ^ { x } } ^ { x , \ell } , \mathbf { w } _ { \mathrm { s a f e } } \rangle \right) .\tag{17}
$$

By Assumption 2, $\langle \mathbf { o } _ { i , \mathcal { T } _ { \mathrm { V } } ^ { x } } ^ { x , \ell } , \mathbf { w } _ { \mathrm { s a f e } } \rangle \leq 0$ , meaning the safety steering strictly relies on the positive contribution from the think span $\langle \mathbf { o } _ { i , \mathcal { T } _ { T } ^ { x } } ^ { x , \ell } , \mathbf { w } _ { \mathrm { s a f e } } \rangle$ . By Cauchy-Schwarz inequality, the maximum possible safety steering provided by the think span at layer ℓ is bounded by its $L _ { 2 }$ norm:

$$
\begin{array} { r } { \langle \mathbf { o } _ { i , \mathcal { T } _ { T } ^ { x } } ^ { x , \ell } , \mathbf { w } _ { \mathrm { s a f e } } \rangle \leq \left\| \mathbf { o } _ { i , \mathcal { T } _ { T } ^ { x } } ^ { x , \ell } \right\| _ { 2 } \left\| \mathbf { w } _ { \mathrm { s a f e } } \right\| _ { 2 } . } \end{array}\tag{18}
$$

Recall our definition of the normalized think contribution in Eq. 3:

$$
r _ { i , \ell } ^ { x } = \frac { \left\| \mathbf { o } _ { i , \mathcal { L } _ { T } ^ { x } } ^ { x , \ell } \right\| _ { 2 } ^ { 2 } } { \left\| \mathbf { o } _ { i , \mathcal { K } _ { i } ^ { x , \ell } } ^ { x , \ell } \right\| _ { 2 } ^ { 2 } + \epsilon } .\tag{19}
$$

$$
\left\| \mathbf { o } _ { i , \kappa _ { i } ^ { x , \ell } } ^ { x , \ell } \right\| _ { 2 } ^ { 2 }
$$

$$
S ^ { x }
$$

$$
\left. \mathbf { o } _ { i , \mathcal { T } _ { T } ^ { x } } ^ { x , \ell } \right. _ { 2 }
$$

$$
R _ { \mathrm { t h i n k } } ^ { x }
$$

$$
r _ { i , \ell } ^ { x }
$$

$$
r _ { i , \ell } ^ { x }
$$

Conclusion. For a paired query in HR and NHR languages, because the internal safety reasoning remains conceptually intact across languages, ${ \bf w } _ { \mathrm { s a f e } }$ remains constant. The difference in their safety steering relies primarily on the attention energy ratio:

$$
S ^ { \mathrm { H R } } - S ^ { \mathrm { N H R } } \propto R _ { \mathrm { t h i n k } } ^ { \mathrm { H R } } - R _ { \mathrm { t h i n k } } ^ { \mathrm { N H R } } \equiv \mathrm { T G S ^ { \mathrm { N H R } | \mathrm { H R } } } .\tag{20}
$$

When $\mathrm { T G S ^ { N H R | H R } } > 0 .$ , it theoretically guarantees $S ^ { \mathrm { N H R } } < S ^ { \mathrm { H R } }$ . The model accumulates less safety steering signal in the residual stream during NHR generation, resulting in a lower probability of generating a safe response. This mathematically explains why a positive TGS correlates with the cross-lingual safety gap (higher ASR in NHR languages) observed in Section 3.1.

## C Attack Success Rate Computation

We quantify model safety using the attack success rate (ASR), a standard metric in safety alignment research Qi et al. (2024b); Zeng et al. (2024). ASR measures the proportion of queries for which the model fails to prevent the intended harmful behavior. Given an instruction dataset D, we define ASR as

$$
\operatorname { A S R } = { \frac { 1 } { | { \mathcal { D } } | } } \sum _ { Q _ { i } \in { \mathcal { D } } } \mathbb { I } ( Q _ { i } ) ,\tag{21}
$$

where $Q _ { i }$ denotes an input query and $\mathbb { I } ( Q _ { i } )$ is an indicator function specifying whether the model’s response constitutes a successful attack. Specifically,

$$
\mathbb { I } ( Q _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ t h e ~ r e s p o n s e ~ f u l f i l s ~ t h e ~ m a l i c i o u s ~ i n t e n t ~ o f ~ } Q _ { i } , } \\ { 0 , } & { \mathrm { i f ~ t h e ~ m o d e l ~ r e f u s e s ~ t h e ~ r e q u e s t . } } \end{array} \right.\tag{22}
$$

Following established practice Qi et al. (2024b), we use GPT-4o as an automated judge to assess the safety of each response. The judge determines whether the response meaningfully complies with the query’s malicious intent, rather than merely matching specific refusal keywords. ASR is computed independently for each language and evaluation condition using the same procedure. Aggregate ASR is then calculated over the corresponding complete query set.

## D Implementation Details

Table 10: Hyperparameters used for the ACTR strategy across different base models. The Random baseline uses the same training configuration.
<table><tr><td>Hyperparameter</td><td>Qwen3-8B</td><td>Gemma4-12B-it</td></tr><tr><td>Computing Device</td><td> $4 \times \mathrm { { A l 0 0 } }$ </td><td> $4 \times \mathrm { { A l 0 0 } }$ </td></tr><tr><td>Global Batch Size</td><td>16</td><td>16</td></tr><tr><td>Training Epochs</td><td>3</td><td>3</td></tr><tr><td>Learning Rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup Ratio</td><td>0.03</td><td> $0 . 0 3$ </td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr></table>

Table 10 summarizes the hyperparameters used to optimize the models. Unless otherwise specified, we use the same training configuration for both base models. For a fair comparison, the Random baseline is trained using exactly the same hyperparameters.

## E Consistency Judge

We use GPT-4o-mini as a frozen judge to assign a binary consistency reward to each Think–Response pair. The judge follows this prompt:

Given a Think trace, and a Response, classify the Think trace and the Response separately into one of three categories:

Safe: The text contains no unsafe content or refuses to answer the unsafe query.

Unsafe: The text contains unsafe content or directly answers the unsafe query.

Invalid: The text does not form a natural sentence or is irrelevant to the given query.

First classify the Think trace and the Response. Then return 1 if both are classified as Safe or both are classified as Unsafe. Otherwise, return 0.

Thus, the binary consistency reward is

$$
r _ { \mathrm { c o n s } } = \mathcal { k } \mathrm { [ ( } c _ { \mathrm { T h i n k } } , c _ { \mathrm { R e s p o n s e } } ) \in \{ ( \mathrm { S a f e } , \mathrm { S a f e } ) , ( \mathrm { U n s a f e } , \mathrm { U n s a f e } ) \} \mathrm { ] } .
$$

The reward measures agreement between the two classifications, not response safety: a trajectory classified as Unsafe in both components also receives a reward of 1. Any trajectory with an Invalid classification receives 0.

## F Qualitative Analysis of Safety Think Neurons Masking

Qualitative Examples of Safety Degradation upon Masking safety think neurons. We present examples across two different models and NHR languages. For each instance, the green text indicates the safe content, while the red text highlights the harmful content generated after masking safety think neurons.

![](images/7ad3dc961f98d47e0147c041f2ce4e44ff62eb008453333476f679c47c981018.jpg)  
Figure 8: Qualitative Examples of Safety Degradation upon masking safety think neurons across different models and NHR languages.

## G Limitations

This work primarily evaluates multilingual safety under prompts written in a single language at a time, while real-world users may combine multiple languages, scripts, or language-specific expressions within the same interaction. In future work, we plan to build on ACTR to study think–response alignment under code-switching and dynamically changing language contexts. Such settings may provide a more realistic test of whether safety judgments can be consistently transferred when the language of the prompt, reasoning trace, and response are not fixed. We will further investigate whether language-switch-aware alignment can improve robustness while preserving natural multilingual interaction.

## H More Discussions

Q1. Does the consistency reward in NSCO fully align with safety objectives, and could optimizing consistency introduce potential biases?

A1. The consistency reward is not designed as a substitute for safety reward, but as a way to strengthen the alignment between safety reasoning and final responses. We intentionally avoid rewarding only safe responses, as directly optimizing refusal behavior may lead to over-refusal on benign requests. NSCO aims to improve safety by enhancing thought–response alignment rather than enforcing additional safety constraints.

Q2. Are the identified “safety think neurons” exclusively dedicated to safety, or do they serve general reasoningtransfer functions?

A2. The neurons identified by our pipeline likely represent a mixture of safety-specific circuits and general thinkutilization pathways. Because we identify them using a probing dataset of jailbreak queries, the selected neurons are highly active during safety-critical contexts. However, neural representations in LLMs are inherently polysemantic. As shown in our ablation study (Figure 5), masking more than 3% of these neurons degrades general mathematical reasoning (MGSM). This suggests that while the top 3% are highly specialized for transferring safety-related constraints, deeper expansion into the neuron population disrupts the model’s fundamental ability to utilize its reasoning traces for general tasks. ACTR succeeds precisely because it isolates the most safety-relevant subset without destroying the broader think-response bridge.

## Q3. Could targeted masking simply damage the model globally?

A3. Random masking uses the same number of units and the same layer-wise distribution, yet changes ASR and TGS only marginally. Targeted masking instead produces large and selective increases, including a 43.90-point MultiJail increase for Gemma4-12B-it versus 1.54 points under random masking. The intervention is therefore not equivalent to generic capacity removal.

## Q4. Does the random-neuron baseline show that neuron identification is unnecessary?

A4. No. The random baseline shows that sparse updates can help, but it does not match ACTR’s safety–utility trade-off. For Qwen3-8B, random-neuron training raises the benign refusal rate on XSTest from 4.40% to 52.00%, whereas ACTR raises it only to 6.40% while achieving higher unsafe-request refusal. The same pattern is stronger for Gemma4-12B-it.