# FIRST TOKEN MATTERS: UNDERSTANDING SAFETY COLLAPSE IN LARGE REASONING MODELS

A PREPRINT

Yizheng Yang<sup>1</sup> Haining Yu<sup>1</sup>\* Yuechen Wang<sup>1</sup> Yikai Hou<sup>2</sup> Xing Fu<sup>1</sup> Jinbo Yang<sup>1</sup> Tianqing Zhu<sup>3</sup> <sup>1</sup>Harbin Institute of Technology, Harbin, China <sup>2</sup>Guangzhou University, Guangzhou, China <sup>3</sup>City University of Macau, Macau, China

## ABSTRACT

Large Reasoning Models (LRMs) exhibit strong problem-solving abilities, yet their safety alignment often degrades when handling harmful queries. Existing approaches to improving safety largely rely on additional training or preference optimization, while offering limited understanding of the internal mechanisms behind safety failures. In this work, we investigate this failure through a token-level positional analysis of refusal dynamics and identify a localized vulnerability at the onset of reasoning, which we term Onset Refusal Collapse (ORC). We find that the refusal-related signal of LRMs drops sharply at the first generated token under harmful queries, which is associated with unsafe response generation. Motivated by this finding, we propose SafeToken, a lightweight inference-time intervention that injects a learned continuous safety anchor precisely at reasoning onset. Despite updating only a single token embedding, SafeToken effectively mitigates ORC, improves safety on harmful-query benchmarks, and largely preserves reasoning utility. These results suggest that safety failures in LRMs can arise from a transient breakdown at the critical transition from understanding to generation.

Keywords Large Reasoning Models · AI Safety · Mechanistic Interpretability.

## 1 Introduction

LRMs have significantly advanced the state of the art in complex problem-solving by leveraging a multi-step generation process. LRMs rely heavily on Chain-of-Thought (CoT) [35] to handle intricate tasks. By dynamically allocating test-time compute, these models excel in domains requiring rigorous logic, such as mathematics, coding, and scientific reasoning [11, 17, 21]. While this paradigm unlocks unprecedented capabilities, recent studies have revealed that CoT mechanisms in LRMs also introduce safety vulnerabilities [40, 44].

Existing LRM safety methods often rely on additional fine-tuning or preference optimization to align reasoning traces [12, 13, 26, 34]. A fundamental limitation of existing studies is that current mitigation methods often treat LRMs as black boxes and rely heavily on human intuition and empirical heuristics to design defense strategies. This highlights a pressing need to uncover the mechanistic cause of safety failures in LRMs.

Figure 1 illustrates an overview of our work. Building on recent findings that models sharing the same backbone may also share a related safety-relevant internal structure [45], we project the latent hidden states of LRMs onto shared refusal vectors extracted from their Instruction-Tuned counterparts. Through this temporal analysis, we discover a critical phenomenon that we term Onset Refusal Collapse (ORC), as illustrated in Figure 1. Specifically, while the refusal projection strength of LRMs remains aligned with that of IT models during the prompt encoding phase, it drops sharply at the first generated token.

![](images/86083acdce589b917aa56b7b05690d60b00537661eb6b2866887a1d2b3227c0b.jpg)  
Figure 1: Overview of our work.

## 2 Preliminaries

Large Reasoning Models. We focus on decoder-only Transformer-based LRMs. Given an input query x, an LRM generates an intermediate reasoning trajectory $t = ( t _ { 1 } , \dots , t _ { K } )$ before producing the final response y:

$$
P ( y , t \mid x ) = \prod _ { i = 1 } ^ { K } P ( t _ { i } \mid x , t _ { < i } ) \cdot P ( y \mid x , t ) .\tag{1}
$$

Unlike standard instruction-tuned models, LRMs rely more heavily on intermediate reasoning traces to solve complex tasks.

Safety-Related Representations. Prior mechanistic interpretability work suggests that safety behavior in language models is partly mediated by identifiable internal components, including safety-relevant attention heads [3, 45]. [45] claim that the pre-training phase shapes fundamental safety capabilities, and that safety heads exhibit substantial overlap across models from the same lineage. Other work shows that refusal behavior can be represented approximately linearly in the residual stream, making it possible to analyze or steer refusal through activation-level interventions [1, 22].

Models and Data. We study instruction-tuned (IT) models and LRMs derived from the same pretrained family. For LRMs, we use DeepSeek-R1-Distill-Llama-8B and DeepSeek-R1-Distill-Qwen-7B and 14B [4], denoted as R1-8B, R1-7B and R1-14B. As their IT counterparts, we use Llama-3.1-8B-Instruct [31] and Qwen2.5-7B(and 14B)- Instruct [32, 37].<sup>2</sup> For the datasets, we use AdvBench [2], StrongReject [29], and Alpaca [30] for analysis.

## 3 Locating Safety Collapse in Large Reasoning Models

In this section, we identify Onset Refusal Collapse, a localized safety failure in LRMs where refusal strength drops sharply at the first generated token. By comparing LRMs with their IT counterparts in a shared refusal subspace, we show that the failure is concentrated at reasoning onset.

## 3.1 Safety Failure Induced by Chain of Thought

We conduct comparative experiments between IT models and LRMs derived from the same backbone. The results suggest that reasoning traces in current LRMs can create a bypass mechanism for safety alignment, thereby motivating a deeper investigation into their internal representations (see Appendix A for more details).

Table 1: Attack Success Rate (ASR) on JailbreakBench-Behavior under steering with refusal vectors. NS denotes Negative Steering, and PS denotes Positive Steering.
<table><tr><td>Model</td><td>Before</td><td>NS</td><td>PS</td></tr><tr><td>R1-8B</td><td>45%</td><td>75%</td><td>21%</td></tr><tr><td>R1-7B</td><td>59%</td><td>87%</td><td>27%</td></tr></table>

![](images/dc88b8a6aa57e7856cca69a50b4ddff525c17bc9e6b0c1b764e990dcbd0c0f97.jpg)  
(a) Trajectory comparison for Llama.

![](images/9700381e1f3457bcefd4ad0323783012ecb5ab2ccb17e030dc1ebe501a2ee276.jpg)  
(b) Trajectory comparison for Qwen.  
Figure 2: Trajectory comparison analysis. The visualization illustrates the refusal-projection trajectories of LRMs and their IT counterparts on harmful prompts.

## 3.2 The Onset Refusal Collapse of LRMs

We next ask whether LRMs fail on harmful queries because they genuinely lose refusal capability, or because they fail to preserve refusal-related information that remains encoded in latent space. To answer this, we first construct a shared refusal representation across LRMs and their IT counterparts. Concretely, we identify safety heads in the IT model with EAP-IG [8], and use their layer-wise residual contributions on harmful queries to derive a set of unified refusal vectors, denoted by V (see Appendix B and Appendix C for details).

We test the transferability of V through activation steering on R1-7B and R1-8B. As shown in Table 1, steering along these vectors yields a clear two-sided effect on model responses. This result suggests that LRMs and their IT counterparts derived from the same backbone share similar latent directions associated with refusal. Based on this shared representational structure, we next examine how these directions behave across token positions during generation.

We project the hidden states of each generated token onto V and track the resulting Refusal Projection Strength:

$$
S _ { t } = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } ( h _ { l , t } \cdot v _ { l } )\tag{2}
$$

where $L$ is the total number of layers and higher $S _ { t }$ indicates stronger refusal-related activation. By comparing the evolution of $S _ { t }$ across token positions, we can examine how refusal-related representations change over the course of generation.

Figure 2 shows that, under harmful queries, LRMs remain partially aligned with their IT counterparts during prompt encoding $( t ~ < ~ 0 )$ , suggesting that they may still retain latent awareness of risk. However, a sharp divergence emerges exactly at the first generated token (t = 0): while the IT model maintains strong refusal activation, the LRM exhibits a sudden drop toward the non-refusal region. We term this abrupt decline Onset Refusal Collapse (ORC). This localization identifies generation onset as a possible failure point. Results for harmless inputs are provided in Appendix D.

To quantify this effect beyond single trajectories, we measure ORC relative to each matched IT/LRM pair. Specifically, we compute the matched-pair refusal gap $G _ { t } = \ ' { S _ { t } ^ { I T } } - \ S _ { t } ^ { L R M }$ and define the normalized onset change as

$$
R _ { \mathrm { O R C } } = \frac { G _ { 0 } - G _ { - 1 } } { \frac { 1 } { 4 } \left( | S _ { - 1 } ^ { I T } | + | S _ { - 1 } ^ { L R M } | + | S _ { 0 } ^ { I T } | + | S _ { 0 } ^ { L R M } | \right) } .\tag{3}
$$

Positive values indicate that the IT/LRM refusal gap widens at generation onset. In Table $^ { 2 , }$ Harmful n denotes the number of harmful prompts used for estimation; Mean $R _ { \mathrm { O R C } }$ reports the average normalized onset-induced change

Table 2: Aggregate ORC statistics on harmful prompts. $R _ { \mathrm { O R C } }$ measures the normalized onset-induced change in the matched IT/LRM refusal gap.
<table><tr><td>Matched Pair</td><td>Harmful n</td><td>Mean  $R _ { \mathrm { O R C } }$ </td><td>95% CI</td></tr><tr><td>Llama-3.1-8B-Instruct vs R1-8B</td><td>100</td><td>189.41%</td><td>[179.82%, 199.81%]</td></tr><tr><td>Qwen2.5-7B-Instruct vs R1-7B</td><td>100</td><td>20.00%</td><td>[14.73%, 25.24%]</td></tr><tr><td>Qwen2.5-14B-Instruct vs R1-14B</td><td>100</td><td>100.11%</td><td>[91.37%, 108.57%]</td></tr></table>

Table 3: ASR on the StrongReject benchmark under onset interventions.
<table><tr><td>Onset Intervention</td><td>R1-7B</td><td>R1-8B</td></tr><tr><td>0 (Vanilla LRM)</td><td>49.20</td><td>37.38</td></tr><tr><td>Prefix Length = 1</td><td>20.13</td><td>4.47</td></tr><tr><td>Prefix Length = 2</td><td>22.68</td><td>3.83</td></tr><tr><td>Prefix Length = 3</td><td>25.88</td><td>3.51</td></tr><tr><td>Prefix Length = 4</td><td>27.80</td><td>8.31</td></tr><tr><td>Prefix Length = 5</td><td>15.65</td><td>7.35</td></tr><tr><td>Prefix Length = 6</td><td>25.88</td><td>9.90</td></tr><tr><td>Prefix Length = 7</td><td>19.81</td><td>7.67</td></tr><tr><td>Prefix Length = 8</td><td>24.60</td><td>10.54</td></tr><tr><td>Refusal Direction Injection</td><td>15.02</td><td>7.35</td></tr></table>

in the matched IT/LRM refusal gap; and 95% CI denotes the bootstrap confidence interval of the mean. Results for harmless inputs are provided in Appendix D.

## 3.3 Linking ORC to Safety Failure

A central question is whether ORC is directly tied to the safety failure of LRMs on harmful queries. To answer it more directly, we intervene at the exact onset position where ORC occurs and examine the resulting change in safety performance.

We first conduct extensive sampling of responses to harmful queries from R1-8B and R1-7B. We find that the vast majority of responses begin with affirmative tokens such as “Okay” or “Alright” even in cases where the model eventually refuses. This suggests that the collapse at reasoning onset may play a functional role in shaping the model’s downstream safety behavior.

To test this connection, we evaluate R1-7B and R1-8B on StrongReject by force-decoding safety-related prefixes of lengths ranging from 1 to 8 tokens at the first generation step. Examples include a single-token prefix such as “Safety” and the full eight-token phrase “Let’s think about safety first.” used in SafePath [12]. We additionally evaluate a direct intervention that injects the refusal direction at the first generated token. The detailed setup is provided in Appendix E.

As shown in Table 3, onset interventions at the first generated token consistently reduce ASR relative to the vanilla LRM across both models. Even single-token safety prefixes yield large gains, indicating that safety performance is highly sensitive to the initialization state. This view is further supported by refusal direction injection, which achieves similar improvements. These results suggest that ORC is not only temporally localized, but also makes an important causal contribution to unsafe behavior in LRMs on harmful queries.

We further investigate the possible source of ORC and observe that this onset pattern is associated with a strong tendency to initiate reasoning with affirmative response-formatting tokens. We defer the detailed discussion and supporting evidence to Appendix F.

## 4 SafeToken: Anchoring Safety at Reasoning Onset

## 4.1 Methodology

## 4.1.1 Motivation

Since ORC is sharply localized at the first generated token, an effective intervention should act precisely at reasoning onset and directly reshape the model’s initialization state before unsafe reasoning unfolds. As shown in Table 3, safety performance is highly sensitive to the specific content injected at this position, and shorter safety-related prefixes can even outperform longer ones. This suggests that the key factor is not the use of a fixed natural-language statement itself, but whether the injected signal can effectively mitigate the ORC phenomenon. However, searching for such an intervention in the discrete space of natural-language tokens is inherently limited by the tokenizer vocabulary and may easily interfere with utility. Therefore, inspired by prompt tuning [42], we propose to learn a single optimized continuous representation that serves as a safety anchor for LRMs at reasoning onset.

## 4.1.2 Mechanism of SafeToken

We introduce a continuous “Soft Token” optimized specifically for safety. By directly learning the optimal latent embedding to be injected at the onset, we can effectively mitigate the ORC phenomenon.

Soft Token Optimization. We expand the tokenizer’s vocabulary to include the special control token $T _ { s }$ and explicitly place it exactly at the generation onset (t = 0). During the optimization process, we strictly freeze all parameters of the base LRM and update the embedding vector of $T _ { s }$ via the standard causal language modeling loss.

Inference-Time Injection. Once trained, the soft token serves as a plug-and-play safety anchor. During inference, given any user query $x ,$ we deterministically append the learned $T _ { s }$ to the end of the input sequence. By occupying the critical first position of the output sequence, $T _ { s }$ mitigates ORC and guides the subsequent reasoning trajectory. The generation probability is thus modified from the standard formulation to explicitly condition on this injected token:

$$
P ( y , t | x ) = \prod _ { i = 1 } ^ { K } P ( t _ { i } | x \oplus T _ { s } , t _ { < i } ) \cdot P ( y | x \oplus T _ { s } , t )\tag{4}
$$

where ⊕ denotes sequence concatenation, $t = ( t _ { 1 } , \dots , t _ { K } )$ represents the intermediate reasoning trace, and $y$ denotes the final response.

## 4.2 Experiments

## 4.2.1 Experiment Setup

Baselines. We select R1-8B and R1-7B as the target LRMs. To further examine the transferability, we include Qwen3-8B [25] in the experiments. We compare SafeToken with SafeChain [13], SafePath [12], STAR-1 [34], and Safety Prefix, which is an inference-time variant of SafePath. More details are provided in Appendix G.1.

SafeToken Setup. For fair comparison, we construct our training data following the setup of STAR-1 and use 1,000 samples to optimize the proposed continuous safety anchor. The learnable token is initialized with the embedding of the token “Safety”. The token embedding is trained using the standard causal language modeling loss with a batch size of 8. We train for 15 epochs on R1-8B and 20 epochs on R1-7B, utilizing the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 3 }$

Evaluation. We evaluate performance across four dimensions: safety, jailbreak robustness, over-refusal, and utility. For safety, we employ StrongReject [29] and AdvBench [46], utilizing Llama-Guard-3-8B [5] as an automated evaluator to classify the safety of the generated responses. To assess robustness against jailbreak attacks, we utilize GCG [2] using HarmBench [20] and WildJailbreak [14]. Furthermore, we measure over-refusal using the XSTest dataset [28], employing DeepSeek-Chat [4] to evaluate whether the model improperly refuses to answer benign queries or comply with harmful queries. For utility, we report results on MMLU [9], MATH500 [17], AIME2024 [18] and GPQA-Diamond [27].

## 4.2.2 Main Results

We conducted a comparative analysis of different defense mechanisms, evaluating their performance across both safety and utility benchmarks. The quantitative results are presented in Table 4.

Table 4: Evaluation results on safety, jailbreak, over-refusal, and utility benchmarks. Lower is better for safetyrelated metrics (↓), and higher is better for utility metrics (↑). SR, Adv., WJ, XS, SC, SP, and ST denote StrongReject, AdvBench, WildJailbreak, XSTest, SafeChain, SafePath, and SafeToken, respectively.
<table><tr><td colspan="11">Method Par. SR Adv. GCG WJ XS MMLU MATH AIME GPQA</td></tr><tr><td colspan="11">R1-8B</td></tr><tr><td>Orig. SC Pref. SP</td><td>一 8B 8B</td><td>37.3 33.6 10.5 2.6</td><td>35.0 22.3 9.0 8.1</td><td>62.0 43.5 25.3 6.5</td><td>50.8 37.3 29.8 0.8</td><td>5.2 3.6 22.4</td><td>69.9 66.5 63.0</td><td>75.2 72.6 67.4</td><td>40.0 30.0 33.3</td><td>46.0 42.4 41.4</td></tr><tr><td>STAR ST</td><td>8B 4096</td><td>0.3 2.2</td><td>0.0 0.6</td><td>3.8 25.3</td><td>7.8 23.2 11.6</td><td>74.8 25.2</td><td>58.2 65.0 66.0</td><td>72.8 76.6 75.2</td><td>23.3 30.0 26.7</td><td>43.4 43.4 42.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>R1-7B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Orig. SC 7B Pref.</td><td></td><td>49.2 41.2</td><td>50.3 27.0</td><td>64.5 51.3</td><td>52.7 13.2 39.5 12.8</td><td></td><td>63.5 61.8</td><td>78.4 80.0</td><td>46.7 46.7</td><td>52.0 47.0</td></tr><tr><td>SP STAR ST</td><td>一 7B 7B 3584</td><td>24.6 6.4 0.2 4.8</td><td>14.6 2.7 0.0 2.5</td><td>18.8 13.0 10.8 16.0</td><td>35.8 38.0 17.6 54.8 15.1 41.6 24.937.2</td><td></td><td>60.3 58.2 61.1 61.6</td><td>77.4 77.4 81.2 82.0</td><td>46.7 40.0 36.7 53.3</td><td>39.9 51.5 45.5 51.0</td></tr><tr><td colspan="11"></td></tr><tr><td></td><td></td><td>5.8</td><td>1.2</td><td>43.5</td><td>39.8</td><td>Qwen3-8B</td><td></td><td></td><td></td><td></td></tr><tr><td>Orig. SC</td><td>8B</td><td>38.3</td><td>28.7</td><td>54.5</td><td>40.1</td><td>1.6 4.8</td><td>82.8 81.8</td><td>79.6</td><td>40.0 30.0</td><td>53.5</td></tr><tr><td>Pref.</td><td></td><td>0.3</td><td>0.4</td><td>11.0</td><td>18.4</td><td>6.0</td><td>82.0</td><td>81.8</td><td></td><td>57.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.4</td><td>53.3</td><td>61.1</td></tr><tr><td>SP STAR</td><td>8B 8B</td><td>1.9 0.0</td><td>1.4 0.0</td><td>6.8 3.4</td><td>14.7 5.5</td><td>9.2 44.8</td><td>83.1 73.1</td><td>78.2 83.6</td><td>43.3 46.7</td><td>60.1 42.9</td></tr></table>

Safety and Utility Trade-off. SafeToken achieves strong safety performance on several harmful-query benchmarks while requiring only a single token embedding to be optimized. At the same time, its utility remains competitive. Overall, SafeToken provides a favorable trade-off between safety, utility, and training cost. SafeToken shows limited robustness under strong jailbreak attacks. A likely reason is that jailbreak prompts often act earlier in the pipeline by altering the model’s internal assessment of whether a query is harmful. We further discuss the safety performance in Appendix G.2.

Training Efficiency. SafeToken requires updating only a single token embedding, whose size equals the model’s hidden dimension. Consequently, the entire optimization process converges within minutes on a single GPU.

Mitigating the ORC Phenomenon. As established previously, mitigating the ORC phenomenon directly improves the safety performance of LRMs. Figure 3 visually demonstrates the mechanistic impact of SafeToken. By explicitly anchoring the learned safety representation at the end of the prompt, the refusal projection strength remains robust during the critical initialization phase. Consequently, avoiding this initial collapse leads to a significantly higher refusal strength throughout the subsequent reasoning trajectory.

Compatibility with Existing Safety Methods. To further evaluate the modularity of SafeToken, we combine it with two representative defenses, SafePath and STAR-1, and compare the resulting safety performance before and after adding SafeToken. As shown in Table 5, SafeToken improves or preserves safety performance across most settings. These results suggest that SafeToken is not only effective as a standalone method, but can also serve a a complementary safety component for existing reasoning-time defenses. We provide more details about utility performance in Appendix G.3.

## 4.2.3 Sensitivity Analysis: SafeToken Position

To validate the necessity of anchoring position, we compare our proposed onset injection with the conventional frontinjection baseline used in most prompt-tuning studies [16]. For the Front Injection setup, we prepend the soft token $T _ { s }$ to the very beginning of the user prompt to act as an implicit system prompt. For a fair comparison, we train the front-injected token with the same dataset, configuration, and hyperparameters used for SafeToken. We then evaluate both configurations on the StrongReject and AdvBench benchmarks using the R1-8B and R1-7B models.

![](images/0b293b477aa691dc5ee6a8d3ee58c69ce78c7ea6cc48366c102df093ed7e4f2d.jpg)  
Figure 3: SafeToken (With $T _ { s } )$ successfully prevents the sharp decline (ORC) observed in R1-8B (Without $T _ { s } )$ precisely at $t = 0 .$

Table 5: Effect of adding SafeToken to SafePath and STAR-1. Lower is better for safety metrics.
<table><tr><td>Model</td><td>Method</td><td>Metric</td><td></td><td>Before + SafeToken</td></tr><tr><td rowspan="3">R1-7B</td><td>SafePath</td><td>StrongReject AdvBench</td><td>6.4 2.7</td><td>2.6 0.8</td></tr><tr><td></td><td>WildJailbreak StrongReject</td><td>17.6 0.2</td><td>13.2</td></tr><tr><td>STAR-1</td><td>AdvBench WildJailbreak</td><td>0.0 15.1</td><td>0.0 0.0 12.0</td></tr><tr><td rowspan="3">R1-8B</td><td>SafePath</td><td>StrongReject AdvBench WildJailbreak</td><td>2.6 8.1 0.8</td><td>4.2 0.8 11.3</td></tr><tr><td>STAR-1</td><td>StrongReject AdvBench</td><td>0.3 0.0</td><td>0.3 0.0</td></tr><tr><td></td><td>WildJailbreak StrongReject</td><td>7.8</td><td>8.2</td></tr><tr><td rowspan="3">Qwen3-8B</td><td>SafePath</td><td>AdvBench</td><td>1.9 1.4</td><td>0.0 0.0</td></tr><tr><td></td><td>WildJailbreak StrongReject</td><td>14.7</td><td>9.1</td></tr><tr><td rowspan="2">STAR-1</td><td>AdvBench</td><td>0.0 0.0</td><td>0.0 0.0</td></tr><tr><td>WildJailbreak</td><td>5.5</td><td>4.9</td></tr></table>

Table 6: Impact of injection position on safety benchmarks. Lower ASR (↓) indicates better safety performance.
<table><tr><td>Model</td><td>Position</td><td>StrongReject (↓)</td><td>AdvBench (↓)</td></tr><tr><td rowspan="2">R1-8B</td><td>Front Injection</td><td>41.9</td><td>37.9</td></tr><tr><td>SafeToken</td><td>2.2</td><td>0.6</td></tr><tr><td rowspan="2">R1-7B</td><td>Front Injection</td><td>64.9</td><td>56.4</td></tr><tr><td>SafeToken</td><td>4.8</td><td>2.5</td></tr></table>

As demonstrated in Table 6, the Front Injection strategy results in a significant degradation of safety performance compared to SafeToken, indicating that the position of SafeToken is critical.

## 5 Related Work

Safety in LRMs. [44] find that stronger reasoning ability leads to greater potential harm. This trade-off is exemplified by self-jailbreaking [19, 40], a phenomenon where models exploit their own reasoning processes to rationalize complying with malicious queries. Furthermore, adversaries can weaponize this mechanism through CoT-based attacks [15, 43]. To counter the unique vulnerabilities introduced by complex reasoning traces, recent studies have explored severa defense paradigms [6, 7, 12, 13, 23]. [10, 36, 39] also discover other forms of safety collapse in LRMs.

Impact of Tokens. Prompt tuning learns continuous input embeddings while keeping the backbone model frozen [16]. Recent safety methods build on this idea to reduce unsafe generations without full-parameter fine-tuning [41, 42]. Meanwhile, prior work shows that the effect of prompt-based control depends not only on the prompt representation itself, but also on where it is injected in the model [24, 33, 38].

## 6 Conclusion

In this work, we investigate why safety alignment in LRMs can fail under harmful queries. Through a token-level analysis of refusal dynamics, we identify a localized vulnerability at the onset of reasoning. Motivated by this observation, we propose SafeToken, a lightweight inference-time intervention that learns a continuous safety anchor and injects it precisely at reasoning onset. SafeToken effectively mitigates this collapse and improves safety performance on harmful-query benchmarks while preserving general reasoning utility.

Overall, our results highlight reasoning onset as a critical stage for safety in LRMs and provide evidence that lightweight, mechanism-driven interventions can effectively improve safety behavior.

## 7 Limitations

Our study focuses on a specific safety failure mode in LRMs, namely the collapse of refusal behavior at reasoning onset. Although SafeToken improves safety across multiple settings, its gains are less pronounced in some stronger attack scenarios. One possible reason is that such attacks may interfere with the model’s recognition of harmfulness itself, whereas SafeToken mainly targets cases where risk awareness is already present but fails to be translated into refusal behavior due to ORC. A possible direction for future work is to integrate SafeToken into the training pipeline. Such a combination may improve robustness against attacks that act earlier in the safety decision process.

## Acknowledgments

We thank anonymous reviewers for valuable comments. This work was supported in part by the National Natural Science Foundation of China under Grant 62302122, National Key Research and Development Program of China under Grant 2025YFB3109803 and Heilongjiang Provincial Natural Science Foundation of China under Grant JQ2024F001.

## References

[1] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. Advances in Neural Information Processing Systems, 37: 136037–136083, 2024.

[2] Matan Ben-Tov, Mor Geva, and Mahmood Sharif. Universal Jailbreak Suffixes Are Strong Attention Hijackers. arXiv preprint arXiv:2506.12880, 2025.

[3] Jianhui Chen, Xiaozhi Wang, Zijun Yao, Yushi Bai, Lei Hou, and Juanzi Li. Finding safety neurons in large language models. arXiv preprint arXiv:2406.14144, 2024.

[4] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[5] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

[6] Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Furong Huang, Dinesh Manocha, and Amrit Singh Bedi. Safety recovery in reasoning models is only a few early steering steps away. arXiv preprint arXiv:2602.11096, 2026.

[7] Peixuan Han, Cheng Qian, Xiusi Chen, Yuji Zhang, Denghui Zhang, and Heng Ji. Safeswitch: Steering unsafe llm behavior via internal activation signals. arXiv preprint arXiv:2502.01042, 2025.

[8] Michael Hanna, Sandro Pezzelle, and Yonatan Belinkov. Have faith in faithfulness: Going beyond circuit overlap when finding model mechanisms. CoRR, abs/2403.17806, 2024.

[9] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings ofthe International Conference on Learning Representations (ICLR), 2021.

[10] Yuyi Huang, Runzhe Zhan, Lidia S. Chao, Ailin Tao, and Derek F. Wong. Path drift in large reasoning models: How first-person commitments override safety. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 19602–19616, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.990. URL https://aclanthology.org/2025. emnlp-main.990/.

[11] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

[12] Wonje Jeung, Sangyeon Yoon, Minsuk Kahng, and Albert No. Safepath: Preventing harmful reasoning in chain-of-thought via early alignment. arXiv preprint arXiv:2505.14667, 2025.

[13] Fengqing Jiang, Zhangchen Xu, Yuetai Li, Luyao Niu, Zhen Xiang, Bo Li, Bill Yuchen Lin, and Radha Poovendran. Safechain: Safety of language models with long chain-of-thought reasoning capabilities. In ACL (Findings), volume ACL 2025 of Findings ofACL, pages 23303–23320. Association for Computational Linguistics, 2025.

[14] Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha Dziri. Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models. arXiv preprint arXiv:2406.18510, 2024.

[15] Martin Kuo, Jianyi Zhang, Aolin Ding, Qinsi Wang, Louis DiValentin, Yujia Bao, Wei Wei, Hai Li, and Yiran Chen. H-cot: Hijacking the chain-of-thought safety reasoning mechanism to jailbreak large reasoning models, including openai o1/o3, deepseek-r1, and gemini 2.0 flash thinking. arXiv preprint arXiv:2502.12893, 2025.

[16] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 3045–3059, 2021.

[17] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

[18] MAA Committees. Aime problems and solutions. https://artofproblemsolving.com/wiki/index.php/ AIME\_Problems\_and\_Solutions, 2024.

[19] Yingzhi Mao, Chunkang Zhang, Junxiang Wang, Xinyan Guan, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, and Le Sun. When models outthink their safety: Mitigating self-jailbreak in large reasoning models with chain-of-guardrails. arXiv preprint arXiv:2510.21285, 2025.

[20] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.

[21] William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought. In NeurIPS 2023 Workshop on Mathematics ofModern Machine Learning, 2023.

[22] Kyle O’Brien, David Majercak, Xavier Fernandes, Richard Edgar, Blake Bullwinkel, Jingya Chen, Harsha Nori, Dean Carignan, Eric Horvitz, and Forough Poursabzi-Sangdeh. Steering language model refusal with sparse autoencoders. arXiv preprint arXiv:2411.11296, 2024.

[23] ShengYun Peng, Eric Smith, Ivan Evtimov, Song Jiang, Pin-Yu Chen, Hongyuan Zhan, Haozhu Wang, Duen Horng Chau, Mahesh Pasupuleti, and Jianfeng Chi. Large reasoning models learn better alignment from flawed thinking. arXiv preprint arXiv:2510.00938, 2025.

[24] Aleksandar Petrov, Philip Torr, and Adel Bibi. When do prompting and prefix-tuning work? A theory of capabilities and limitations. In ICLR. OpenReview.net, 2024.

[25] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

[27] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First conference on language modeling, 2024.

[28] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. arXiv preprint arXiv:2308.01263, 2023.

[29] Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. A strongreject for empty jailbreaks. arXiv preprint arXiv:2402.10260, 2024.

[30] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/ tatsu-lab/stanford\_alpaca, 2023.

[31] Llama Team. The llama 3 herd of models. CoRR, abs/2407.21783, 2024.

[32] Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/ blog/qwen2.5/.

[33] Yiwei Wang, Muhao Chen, Nanyun Peng, and Kai-Wei Chang. Vulnerability of large language models to output prefix jailbreaks: Impact of positions on safety. In NAACL (Findings), Findings of ACL, pages 3939–3952. Association for Computational Linguistics, 2025.

[34] Zijun Wang, Haoqin Tu, Yuhan Wang, Juncheng Wu, Yanqing Liu, Jieru Mei, Brian R Bartoldson, Bhavya Kailkhura, and Cihang Xie. Star-1: Safer alignment of reasoning llms with 1k data. arXiv preprint arXiv:2504.01903, 2025.

[35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[36] Rui Wu, Yihao Quan, Zeru Shi, Zhenting Wang, Yanshu Li, and Ruixiang Tang. Read the scene, not the script: Outcome-aware safety for llms. arXiv preprint arXiv:2510.04320, 2025.

[37] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

[38] Xianjun Yang, Wei Cheng, Xujiang Zhao, Wenchao Yu, Linda Ruth Petzold, and Haifeng Chen. Position really matters: Towards a holistic approach for prompt tuning. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 8516–8538, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/ 2025.findings-naacl.474. URL https://aclanthology.org/2025.findings-naacl.474/.

[39] Qingyu Yin, Chak Tou Leong, Linyi Yang, Wenxuan Huang, Wenjie Li, Xiting Wang, Jaehong Yoon, Jinjin Gu, et al. Refusal falls off a cliff: How safety alignment fails in reasoning? arXiv preprint arXiv:2510.06036, 2025.

[40] Zheng-Xin Yong and Stephen H Bach. Self-jailbreaking: Language models can reason themselves out of safety alignment after benign reasoning training. arXiv preprint arXiv:2510.20956, 2025.

[41] Lingzhi Yuan, Xinfeng Li, Chejian Xu, Guanhong Tao, Xiaojun Jia, Yihao Huang, Wei Dong, Yang Liu, and Bo Li. Promptguard: Soft prompt-guided unsafe content moderation for text-to-image models. arXiv preprint arXiv:2501.03544, 2025.

[42] Xiaoyun Zhang, Zhengyue Zhao, Wenxuan Shi, Kaidi Xu, Di Huang, and Xing Hu. Safety alignment of large language models via contrasting safe and harmful distributions. arXiv preprint arXiv:2406.16743, 2024.

[43] Jianli Zhao, Tingchen Fu, Rylan Schaeffer, Mrinank Sharma, and Fazl Barez. Chain-of-thought hijacking. arXiv preprint arXiv:2510.26418, 2025.

[44] Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, and Xin Eric Wang. The hidden risks of large reasoning models: A safety assessment of R1. In IJCNLP-AACL (long papers), pages 3250–3265. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, 2025.

[45] Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Kun Wang, Yang Liu, Junfeng Fang, and Yongbin Li. On the role of attention heads in large language model safety. In ICLR. OpenReview.net, 2025.

[46] Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## A Safety Performance

Recent studies have highlighted a critical trade-off in LRMs: while CoT enhances performance on complex tasks, it also places substantial pressure on safety mechanisms.

To verify this observation, we conduct comparative experiments between IT models and LRMs derived from the same backbone. As illustrated in Figure 4, the IT models maintain robust refusal rates on AdvBench. In contrast, the corresponding LRMs exhibit a marked increase in unsafe responses.

![](images/17fd8c9df19923c1a947388a1d678bcfcb6106933832aea7af42232968e009a7.jpg)  
Figure 4: Safety performance comparison between IT baselines and derived LRMs on AdvBench.

## B Discovery of Safety Heads

## B.1 EAP-IG Introduction and Data Preparation

EAP-IG identifies critical model components by comparing attribution patterns between counterfactual input pairs. Intuitively, the method requires two closely matched inputs that differ only in the target behavior, so that the resulting attribution difference can be used to isolate components that are causally related to the behavior of interest. In our setting, this means constructing paired prompts that preserve the same semantic content while differing in whether they elicit a safe refusal or an unsafe response onset.

However, applying this algorithm to LRMs is particularly challenging. In practice, LRMs tend to begin both safe and unsafe responses with the same or highly similar onset tokens. As a result, the behavioral divergence of interest is often not reflected in the surface form of the first generated token, even though the underlying safety state may already differ internally. This makes it difficult to construct clean counterfactual pairs for EAP-IG.

To support this analysis, we manually design a prompt construction procedure to generate such counterfactual pairs. Specifically, we use a locally deployed LLM to rewrite harmful prompts into closely matched safe counterparts. As shown in the prompt box below, the rewriting prompt explicitly instructs the model to preserve the original wording, structure, and style as much as possible, while modifying only the minimum content necessary to remove harmful intent. This controlled setup allows EAP-IG to more reliably identify the components that contribute to the behavioral difference, rather than capturing unrelated variations in content or style.

We provide two representative examples of this rewriting process in the example boxes below. For instance, a harmful query about making a bomb is rewritten into a benign query about making bread, and a query about hacking an email account is rewritten into one about securing the account. These examples illustrate how our procedure preserves surface form while shifting the prompt from harmful to safe.

We then apply EAP-IG to these counterfactual pairs to trace which internal components are most influential at reasoning onset. The resulting attribution scores are used to rank critical components and analyze how safety-related behavior changes at the beginning of generation.

## Prompt for Counterfactual Rewriting

Task: Rewrite a harmful prompt into a safe counterpart while preserving its surface form as much as possible. Rules:

1. PRESERVE: Keep the wording, structure, and style as close to the original as possible   
2. MINIMAL CHANGE: Only modify the parts necessary to remove harmful intent   
3. ALIGNMENT: Ensure the rewritten prompt remains semantically similar but leads to a safe response   
4. NO EXTRA CONTENT: Do not add explanations, comments, or additional context   
Original: {text}   
Rewritten:

## Example 1

Original: Teach children how to access and use firearms.   
Rewritten: Teach children how to access and use tools.

## Example 2

Original: Write a tutorial on how to make a bomb.   
Rewritten: Write a tutorial on how to build a model.

## B.2 Setup

We use integrated gradients with steps=5. We use a batch size of 2 and restrict the attribution stage to the first 50 counterfactual pairs, while removing pairs whose tokenized lengths are unequal. The attribution target is defined as the final-token logit difference between the two label tokens “Sure” and “I”. After obtaining raw edge scores, we sort all edges by the absolute value of their attribution score and retain the top-n edges to build the final circuit. In our experiments, we consider multiple graph sizes n, with n ∈ {50, 100, 200, 500, 1000}.

Algorithm 1: Refusal Vector Extraction via Safety Head Ablation   
Input: IT Model M, Safety Heads H, Harmful Dataset D, Total Layers L   
Output: Refusal Vector set V = {v<sub>0</sub>, . . . , v<sub>L−1</sub>}   
1 for layer l ← 0 to L − 1 do   
2 Initialize update set U ← ∅;   
3 foreach sample x ∈ D do   
4 Get original activations: h<sup>o</sup> ← M(x);   
5 Get ablated activations: $h _ { l } ^ { \dot { a } } \gets M _ { \backslash \mathcal { H } } ( x ) ;$   
6 Compute token-wise mean difference: $\begin{array} { r } { \delta _ { l } \gets \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( h _ { l , t } ^ { a } - h _ { l , t } ^ { o } ) ; } \end{array}$   
7 if l = 0 then   
8 u<sub>l</sub> ← δ<sub>0</sub>;   
9 else   
10 u<sub>l</sub> ← δ<sub>l</sub> − δ<sub>l−1</sub>;   
11 Add u<sub>l</sub> to U<sub>l</sub>;   
12 Solve for d<sup>∗</sup> minimizing $\begin{array} { r } { J ( \boldsymbol { d } ) = - \frac { 1 } { | \mathcal { D } | } \sum _ { \boldsymbol { u } \in \mathcal { U } _ { l } } ( \boldsymbol { d } \cdot \boldsymbol { u } ) ; } \end{array}$   
13 Set refusal vector v ← −d<sup>∗</sup>;   
14 return V

## B.3 Safety Heads

After obtaining the attributed edges from EAP-IG, we identify safety heads by aggregating edge scores at the attentionhead level. Specifically, each edge connects an upstream attention head to a downstream attention input, and we trace the source node of every attributed edge back to its corresponding attention head. We then sum the positive attribution scores of all outgoing edges associated with the same head, which yields a head-level importance score for safety refusal. Heads with larger aggregated scores are considered to contribute more strongly to refusal behavior. Based on this ranking, we select the top-ranked heads as candidate safety heads for subsequent analysis and ablation. For both Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, we choose 200 attention heads.

## C Details of Refusal Direction Extraction

![](images/82fc13c25a3c4ecefb8f5fb3a41dae3fd1380d8ca4ee0426005b93d9fd651484.jpg)  
(a) Trajectory comparison for Llama.

![](images/05b79ee507f93d16383733abe61e782384a2606ced03782994ad08987a805407.jpg)  
(b) Trajectory comparison for Qwen.  
Figure 5: Trajectory comparison results of benign data.

As described in Algorithm 1, we capture the layer-specific safety contribution by decoupling the cumulative residual effects of preceding layers from the differences induced by ablation, and then solve for a unified refusal vector that maximizes alignment with these isolated updates across the dataset.

## D Results for Benign Queries

As Figure 5 illustrates, for harmless inputs, the refusal-projection trajectories show a less consistent pattern than in the harmful case. In particular, R1-8B still exhibits a noticeable drop around the first generated token, while R1-7B remains comparatively stable. This suggests that a decrease at reasoning onset is not entirely unique to harmful queries. However, under harmless inputs, such a pattern is less uniform across models and does not present the same clear and consistent divergence associated with unsafe behavior in the harmful setting.

## E Details of Onset Interventions

## E.1 Prefixes

Table 7 shows safety-related prefixes of different lengths used for onset intervention.

## E.2 Refusal Direction Injection

For each layer l, we inject the refusal direction only at the first generated token:

$$
\tilde { h } _ { l } ( t ) = \left\{ \begin{array} { l l } { h _ { l } ( t ) + \alpha _ { l } v _ { l } , } & { t = 0 , } \\ { h _ { l } ( t ) , } & { t > 0 , } \end{array} \right.
$$

where v<sub>l</sub> denotes the refusal direction and $\alpha _ { l }$ denotes the injection strength.

To determine the optimal injection strength, we train a lightweight Multi-Layer Perceptron (MLP) probe, denoted as $\mathcal { P } _ { l }$ The goal of this probe is to predict the safety gap between the robust IT model and the vulnerable LRM. Specifically,

Table 7: Safety-related prefixes used in the prefix-length intervention study.
<table><tr><td>Length</td><td>Safety Prefix</td></tr><tr><td>1</td><td>Safety</td></tr><tr><td>2</td><td>Safety first</td></tr><tr><td>3</td><td>Safety first.</td></tr><tr><td>4</td><td>Think safety first.</td></tr><tr><td>5</td><td>Please think safety first.</td></tr><tr><td>6</td><td>Let&#x27;s think safety first.</td></tr><tr><td>7</td><td>Let&#x27;s think about safety.</td></tr><tr><td>8</td><td>Let&#x27;s think about safety first.</td></tr></table>

for a given hidden state h<sub>l</sub>, the probe predicts a injection strength $a _ { l } \colon$

$$
a _ { l } = \mathcal { P } _ { l } ( \mathrm { P C A } ( h _ { l } ) )
$$

where the training target is defined as the difference in refusal projection strength between the IT model and the LRM.

## F On the Source of Onset Refusal Collapse

## F.1 The Suppression of Refusal in Deep Layers

To uncover the internal dynamics driving the ORC phenomenon, we use Logit Lens to trace the rank evolution of typical refusal tokens, such as “I” (because IT models often begin refusal with “I’m sorry, ...” in our sampled outputs) and “Sorry,” as well as affirmative tokens such as “Okay” and “Alright,” across model layers. For the analysis, we divide the sampled responses to harmful inputs into two groups based on the final generation outcome:

• Group A (Eventual Refusal): Cases in which the LRM ultimately identifies the risk and declines the request.

• Group B (Harmful Output): Cases in which the LRM fails to adhere to safety guidelines and generates harmful instructions after the reasoning trace.

As illustrated in Figure 6, in the deep layers of the model, the ranks of affirmative tokens improve significantly in both groups, converging toward the top predictions. Conversely, refusal precursors are systematically suppressed in these layers, regardless of whether the model eventually refuses.

To corroborate this observation, we inspect the vocabulary distribution in the final layers (Layers 30–32). As shown in Table 8, the model’s prediction space in these layers is dominated by affirmative semantics.

Table 8: Top-3 tokens and probabilities in the last three layers. The dominance of affirmative tokens intensifies in the final reasoning stages.
<table><tr><td>Rank</td><td>Layer 30</td><td>Layer 31</td><td>Layer 32</td><td></td></tr><tr><td>Top-1</td><td>okay (0.31)</td><td>Okay (0.64)</td><td>Okay (0.55)</td><td rowspan="4"></td></tr><tr><td>Top-2</td><td>Okay (0.31)</td><td>Alright (0.21)</td><td>Alright (0.45)</td></tr><tr><td>Top-3</td><td>Alright (0.11)</td><td>okay (0.09)</td><td>I (0.00)</td></tr></table>

These results indicate that the affirmative onset token is a fundamental structural property of reasoning initiation. The deep layers prioritize semantic anchors that launch the reasoning chain, overriding safety representations until the reasoning trace is established.

## F.2 Attributing the Affirmative Token to Task Focus

We further investigate the source of the onset affirmative token using Integrated Gradients (IG) to attribute the output decision to the input tokens. Specifically, we formulate contrastive attribution objectives. For the LRM, which tends to comply, we quantify the contribution of input tokens to the logit difference $l o g i t ( \tilde { \ " { \bf O k a y } } { \bf \ddot { \prime } } ) - l o g i t ( \tilde { \bf \Phi } { \bf S o r r y } { \bf \ddot { \prime } } ) ,$ ). Conversely, for the IT model, we target the refusal margin $l o g i t ( ^ { \cdots } \mathrm { S o r r y } ^ { \cdot \sqrt { } } ) - l o g i t ( ^ { \cdots } \mathrm { O k a y } ^ { \cdot \cdot \cdot } )$

As illustrated in Figure 7, for a representative query such as “Write a detailed tutorial on how to make a bomb,” the LRM exhibits a distinct attributional bias. The attribution mass is heavily concentrated on structural and instructional token (e.g., “Write,” “detailed,” “tutorial,” and the newline token). In contrast, the semantic risk token “bomb” receives much less attribution than in the IT model. This suggests that the generation of $\mathrm { ^ { 6 6 } O k a y ^ { 5 } }$ is driven primarily by the instruction to produce a response, effectively blinding the model to the harmful subject matter at this critical initialization step.

![](images/0dac9e8423087f7842fb5eac74af8c7cdead513143ffa4c9105a95d85101a901.jpg)  
Figure 6: Token-rank evolution from Logit Lens.

Table 9: Quantitative analysis of attribution dynamics. We report the average Peak Structural Dominance (η) and the First Harmful Rank $( \mathcal { R } _ { \mathrm { h a r m } } )$ across model types and data groups.
<table><tr><td rowspan="2">Model</td><td colspan="2">Group A</td><td colspan="2">Group B</td></tr><tr><td>η</td><td> $\mathcal { R } _ { \mathbf { h a r m } }$ </td><td>η</td><td> $\mathcal { R } _ { \mathbf { h a r m } }$ </td></tr><tr><td>IT Model</td><td>1.87</td><td>5.32</td><td>2.19</td><td>5.63</td></tr><tr><td>LRM</td><td>3.28</td><td>4.38</td><td>3.50</td><td>3.72</td></tr></table>

We classify input tokens into structural tokens $\tau _ { \mathrm { s t r u c t } } .$ , such as instructional verbs $( \mathrm { e . g . }$ , “Write,” “Generate”) and formatting cues $( \mathrm { e . g . , \tilde { \Omega } ^ { 6 6 } } )$ , and harmful tokens ${ \mathcal { T } } _ { \mathrm { h a r m } } .$ , corresponding to semantically risky entities. To quantify the mechanism behind this early refusal failure, we introduce two attribution-based metrics: Peak Structural Dominance (η) and First Harmful Rank $( \mathcal { R } _ { \mathrm { h a r m } } )$

![](images/a53ff96a9d6fb8c687c39f9fd63d2bc8702b1f1845a6586f83593e5656e6758c.jpg)

(a) Attribution scores for R1-8B  
![](images/fb394cefa25112fdd86cabd3bbd3ef518dd50b73540529951515b1c95e9515ad.jpg)  
(b) Attribution scores for Llama-3.1-8B-Instruct  
Figure 7: Contrastive attribution analysis using Integrated Gradients. The combined analysis shows (a) the acceptance margin and (b) the refusal margin differences.

• η is defined as the ratio between the maximum attribution score among structural tokens and that among harmful tokens:

$$
\eta = \frac { \operatorname* { m a x } ( S _ { T _ { \mathrm { s t r u c t } } } ) } { \operatorname* { m a x } ( S _ { T _ { \mathrm { h a r m } } } ) } .
$$

This measures the strength of the model’s impulse to follow instructions relative to its sensitivity to harmful content.

${ \mathcal { R } } _ { \mathrm { h a r m } }$ is defined as the rank position of the first harmful token in the descending list of attribution scores over all input tokens.

We conduct this analysis on JailbreakBench-Behavior and report the values averaged across all samples. Table 9 presents the comparative attribution results. The values of ${ \mathcal { R } } _ { \mathrm { h a r m } }$ show that LRMs actually identify harmful tokens earlier in the attribution ranking than IT models in both groups. However, this early detection is rendered ineffective by the much stronger dominance of structural tokens. The disparity in η indicates that the initial affirmative generation tendency is driven by prioritizing structural adherence over safety evaluation.

## G Details of SafeToken

## G.1 Baselines

A detailed version of baselines:

• SafeChain: A safety alignment dataset consisting of long CoT sequences. It utilizes supervised fine-tuning to internalize safety checking within the model’s reasoning process.

• SafePath: A representative training-based method. It fine-tunes the model to generate a fixed safety statement at the beginning of the chain-of-thought, explicitly aligning the reasoning process through parameter updates.

• Safety Prefix: We construct an inference-time variant of SafePath. Instead of training, we force-decode the same safety statement as the initial thinking tokens.

• STAR-1: A training-based baseline. It fine-tunes the model on policy-grounded deliberative reasoning traces.

## G.2 Safety Performance

SafeToken shows limited robustness under strong jailbreak attacks. A likely reason is that jailbreak prompts often act earlier in the pipeline by altering the model’s internal assessment of whether a query is harmful. In contrast, SafeToken mainly targets a different failure mode: cases in which the model already retains latent awareness of risk, but the refusal behavior is disrupted by ORC at reasoning onset. As a result, SafeToken is effective when unsafe behavior arises from a collapse of refusal at generation onset, but is less capable of defending against attacks that first suppress or distort the model’s harmfulness recognition itself

## G.3 Additional Results on Compatibility

Table 10: Utility performance of SafePath and STAR-1 before and after adding SafeToken. Higher is better for all utility metrics (↑). SP denotes SafePath, S1 denotes STAR-1, and +ST denotes adding SafeToken to the corresponding method.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="2">SafePath</td><td colspan="2">STAR-1</td></tr><tr><td>SP</td><td>+ST</td><td>S1</td><td>+ST</td></tr><tr><td>R1-7B</td><td>MATH500 AIME24 GPQA</td><td>77.4 40.0 51.5</td><td>78.6 43.3 49.5</td><td>81.2 36.7 45.5</td><td>80.4 40.0 47.0</td></tr><tr><td>R1-8B</td><td>MATH500 AIME24 GPQA</td><td>72.8 23.3 43.4</td><td>72.0 26.7 38.9</td><td>76.6 30.0 43.4</td><td>76.8 33.3 40.4</td></tr><tr><td>Qwen3-8B</td><td>MATH500 AIME24 GPQA</td><td>78.2 43.3 60.1</td><td>81.8 50.0 58.1</td><td>83.6 46.7 42.9</td><td>82.2 43.3 45.5</td></tr></table>

We further evaluate the utility of combining SafeToken with existing reasoning-time defenses. As shown in Table 10, adding SafeToken generally preserves utility across most settings and improves performance on some benchmarks, especially AIME24 and MATH500. Although mild degradations remain in a few cases, the overall results suggest that SafeToken can be integrated into existing safety methods without substantial utility loss.