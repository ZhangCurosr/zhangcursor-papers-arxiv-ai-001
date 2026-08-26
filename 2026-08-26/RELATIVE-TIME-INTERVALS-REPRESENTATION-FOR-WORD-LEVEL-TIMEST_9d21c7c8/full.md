# RELATIVE TIME INTERVALS REPRESENTATION FOR WORD-LEVEL TIMESTAMPING WITH MASKED TRAINING

Quanwei Tang<sup>1</sup>, Zhiyu Tang<sup>2</sup>, Xu Li<sup>3</sup>, Dong Zhang<sup>1,4†</sup> , Shoushan Li<sup>1</sup>, Guodong Zhou<sup>1</sup>

<sup>1</sup>Soochow University <sup>2</sup> University of Queenland <sup>3</sup>AISpeech Ltd <sup>4</sup>Jiangsu Key Lab of Language Computing

## ABSTRACT

Although Speech Large Language Models (SpeechLLMs) excel at speech understanding and generation, their capacity for fine-grained, temporally aligned outputs remains underexplored. Our work addresses this gap by enabling SpeechLLMs to jointly model speech content and temporal structure, effectively transforming them from “content understanding machines” into “temporal-aware content understanding machines”. Specifically, we replace traditional absolute timestamps with relative timestamps, achieving a more compact vocabulary and stronger generalization capabilities. To efficiently infuse timestamp prediction ability into pre-trained large language models, we introduce a hybrid fine-tuning strategy: full-parameter fine-tuning of the timestamp-augmented embedding layer and language model head, combined with LoRA fine-tuning of the decoder layers. Moreover, we design a masked timestamp training objective, preventing the model from over-relying on ground-truth timestamps, and thereby enhancing robustness against noisy real-world annotations. Extensive experiments demonstrate that our approach achieves significant improvements in timestamp prediction accuracy while maintaining strong speech transcription performance.

Index Terms— Timestamping, Speech LLM, Absolute Timestamp, Relative Timestamp, Masked Training

## 1. INTRODUCTION

Recent research has explored the integration of large language models (LLMs) with speech processing [1, 2]. However, generating fine-grained, word-level timestamped transcription has not been fully explored yet [3, 4]. This task demands not only accurate speech understanding but also precise temporal alignment between linguistic units and their acoustic realizations [5, 6]. A core challenge lies in how to represent and predict temporal structure efficiently within the constrained vocabulary and sequential decoding paradigm of LLMs [7, 8, 9, 10]. Existing methods like Qwen2 Audio [11] often employ absolute timestamps for modeling. e.g., <|0.45|>Okay<|1.10|>let’s<|2.50|>go<|5.70| >... indicates that the word ”Okay” starts at 0.45 seconds and ends at 1.10 seconds. While this representation is intuitive and facilitates evaluation, its dependence on an absolute temporal reference limits scalability and generalization over long-duration audio segments.

As shown in Figure 1, predicting absolute timestamps is prone to cumulative error propagation. Absolute timestamps are monotonically increasing, causing LLMs to learn a cumulative pattern: generating larger values by adding estimated segment durations to prior timestamps. Furthermore, this approach exhibits poor generalization to unseen temporal ranges. For instance, it fails to predict timestamps beyond the training duration (e.g., at 60 seconds when trained only on utterances up to 30 seconds). A major practical limitation lies in the prohibitively large vocabulary required for high temporal precision: achieving 0.01-second resolution over a 300-second audio clip requires 30,000 distinct timestamp tokens. This explosion in vocabulary significantly increases training complexity and computational overhead, making the approach unscalable for long-form speech processing.

![](images/cffd5bffa3b1f9dba1e639170f93ee8073b399578484774eca8e31105f123408.jpg)  
Fig. 1. Traditional methods rely on fixed-reference absolute timestamps, often struggling with cumulative errors or precise synchronization. Our approach utilizes dynamic interval-based relative timestamps, which inherently model temporal relationships and offer improved robustness and accuracy.

To address these limitations, we propose a relative timestamp representation, defined as the time interval between consecutive words. (e.g., <|0.65|> for words aligned at <|0.45|> and <|1.10|>). This method requires solely estimating the current segment’s duration, thereby minimizing error susceptibility. Only a limited time interval (such as 0.01 seconds to 5 seconds) needs to be learned, and by accumulating these intervals, any length of audio can be represented. Vocabulary only needs to contain limited time interval tokens, compared to absolute timestamps. Human speech perception naturally focuses on relative timing: we intuitively process how long after the previous word the next one occurs, rather than its absolute position in time. This relative interval modeling aligns naturally with the autoregressive generation paradigm of language models. Absolute timestamps force the model to learn an entirely new absolute value that is unrelated to the previously generated content, which is contrary to its natural generation mode.

![](images/b2e99ccc58d557649b40d0895a0d19edd390f5c281a538172a13a5487804952f.jpg)  
Fig. 2. Architectural Comparison of an Absolute Timestamp-based Model (Left) and Our Proposed Relative Timestamp-based Model (Right). The traditional method models timestamps as independent absolute values and struggles with representing timestamps. In contrast, our approach utilizes a compact vocabulary of relative timestamps. By leveraging LoRA fine-tuning on the decoder, our model efficiently learns to jointly generate both text and temporal markers. To ensure accuracy and robustness, our training incorporates a combined loss function and timestamp regularization (token masking), which forces the model to learn a more resilient representation of temporal information.

We make the following three key contributions:

1. A Novel Relative Time Interval Representation for Word-Level Timestamping: We introduce a relative timestamp representation that models inter-word intervals instead of absolute positions. This approach mitigates vocabulary explosion and temporal drift, enabling accurate timestamp prediction for arbitrarily long audio without length-dependent degradation.

2. Advanced Regularization and Alignment Strategies: We propose a regularization technique, Timestamp Masking, which mitigates overfitting to perfect alignments, along with a dynamically weighted Joint Temporal Alignment Loss that adaptively balances transcription accuracy and timestamp fidelity during training.

3. Hybrid Parameter-Efficient Fine-Tuning: We fully finetune only timestamp-specific modules (embedding layer and LM head), while adapting pre-trained decoder layers via LoRA. This role-aware strategy preserves linguistic knowledge with minimal parameter updates, aligning adaptation granularity with each module’s functional role.

## 2. METHODOLOGY

## 2.1. Relative Timestamp Representation

Unlike traditional methods that rely on an extensive vocabulary of absolute timestamps, our approach employs relative timestamps to represent temporal information. Each timestamp token signifies the time interval from the end of the preceding word to the current word. The core advantage of this method is its superior generalization and compact vocabulary. By learning a constrained set of time intervals, the model can cumulatively represent temporal information for audio of any length, eliminating the need to define a unique token for every possible absolute time point.

## 2.2. LLMs-based Architecture and Fine-Tuning

Our model architecture is built upon a pre-trained large speech model, which consists of an encoder for extracting speech features and a decoder for sequence generation.

As shown in Figure 2 (right), to effectively adapt our model for the new task of temporal alignment, we employ a hybrid fine-tuning strategy that combines full parameter updates for new components with parameter-efficient fine-tuning for backbone modules.

We apply full-parameter fine-tuning to the newly introduced timestamp embedding and the Language Model (LM) head. These modules are directly responsible for mapping the new timestamp tokens to a continuous vector space and generating them in the output sequence, respectively. Since their initial weights are random, a comprehensive update of all parameters is necessary for them to effectively learn and represent the new temporal-augmented vocabulary from scratch.

For the architecture of the backbone model, specifically the decoder layers, we utilize LoRA (Low-Rank Adaptation) [12]. This strategic choice is driven by both computational efficiency and the need for targeted adaptation. Instead of updating all parameters, LoRA freezes the original weights and injects small, trainable lowrank matrices. This approach drastically reduces the number of trainable parameters, leading to significant savings in computational resources. Furthermore, by fine-tuning the decoder with LoRA, the model can efficiently learn to insert corresponding temporal markers when generating specific text content, without interfering with the powerful, pre-trained understanding of speech features. This hybrid approach ensures computational efficiency while achieving precise control over the model’s generative behavior for robust and accurate temporal alignment.

## 2.3. Training Strategies for Temporal Alignment

To ensure that our model generates accurate and robust timestamps, we designed two key training strategies:

## 2.3.1. Joint Temporal Alignment Loss

Our training objective is to minimize a joint loss function composed of two components, which balances the weights between text transcription and timestamp generation:

$$
L _ { t o t a l } = L _ { t e x t } + \lambda L _ { t i m e s t a m p }
$$

where $L _ { t e x t }$ is a standard cross-entropy loss measuring the discrepancy between the model’s generated text tokens and the ground-truth tokens. Similarly, $L _ { t i m e s t a m p }$ is a cross-entropy loss that evaluates the prediction accuracy of the discrete timestamp tokens.

The hyperparameter λ serves as a weighting factor between the two loss components. We employ a dynamic weighting strategy: in the early training stages, we set a smaller weight (λ = 1) to allow the model to prioritize learning the speech-to-text mapping. As training progresses, we gradually increase the value of λ to encourage more precise temporal alignment. Specifically, for each training epoch, we increment the value of λ by 1.

## 2.3.2. Timestamp Masking for Regularization

To mitigate overfitting to ground-truth during training, we propose a timestamp masking technique. The core idea involves randomly replacing a subset of timestamp tokens with a [MASK] token. During training, the previous timestamp token is masked, the model cannot reference its ground-truth value, and must instead infer the current timestamp from speech content and historical tokens, thereby enhancing robustness to inaccurate annotations. Similar to BERT’s masked language modeling, this random masking encourages the model to leverage varying input components across iterations, preventing over-reliance on specific timestamps.

Table 1. Performance comparison of different models on timestamp prediction. Precision and Recall are in percentages, and the average timestamp difference is in milliseconds. ’-’ denotes the model does not support Chinese speech.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Tolerance (ms)</td><td colspan="3">Librispeech</td><td colspan="3">Wenet-Meeting</td></tr><tr><td>P (%)</td><td>R (%)</td><td>Avg. Diff (ms)</td><td>P(%)</td><td> $R ( \% )$ </td><td>Avg. Diff (ms)</td></tr><tr><td rowspan="3">Qwen2-Audio [11]</td><td>80</td><td>0.00</td><td>0.00</td><td>1031.23</td><td>60.81</td><td>56.06</td><td>59.03</td></tr><tr><td>160</td><td>0.01</td><td>0.00</td><td>1031.18</td><td>76.07</td><td>70.13</td><td>49.11</td></tr><tr><td>240</td><td>0.16</td><td>0.05</td><td>1030.42</td><td>82.30</td><td>75.87</td><td>43.07</td></tr><tr><td rowspan="3">WhisperTimestamed [13]</td><td>80</td><td>2.70</td><td>2.36</td><td>131.92</td><td>3.54</td><td>1.51</td><td>164.09</td></tr><tr><td>160</td><td>10.09</td><td>8.83</td><td>120.52</td><td>24.30</td><td>10.38</td><td>134.03</td></tr><tr><td>240</td><td>19.90</td><td>17.42</td><td>101.14</td><td>43.97</td><td>18.79</td><td>102.48</td></tr><tr><td rowspan="3">Sense VoiceSmall [14]</td><td>80</td><td>1.10</td><td>1.10</td><td>176.38</td><td>28.52</td><td>27.96</td><td>80.80</td></tr><tr><td>160</td><td>5.20</td><td>5.19</td><td>166.71</td><td>71.40</td><td>69.99</td><td>45.26</td></tr><tr><td>240</td><td>12.88</td><td>12.86</td><td>148.59</td><td>82.82</td><td>81.19</td><td>32.47</td></tr><tr><td rowspan="3">Canary [15]</td><td>80</td><td>35.51</td><td>35.27</td><td>444.30</td><td>=</td><td>=</td><td></td></tr><tr><td>160</td><td>71.23</td><td>70.74</td><td>268.18</td><td>=</td><td>=</td><td></td></tr><tr><td>240</td><td>84.57</td><td>83.99</td><td>200.47</td><td></td><td>=</td><td></td></tr><tr><td rowspan="3">Absolute Timestamp</td><td>80</td><td>12.30</td><td>5.51</td><td>159.05</td><td>45.14</td><td>42.52</td><td>186.73</td></tr><tr><td>160</td><td>17.03</td><td>7.63</td><td>156.58</td><td>62.99</td><td>59.34</td><td>168.29</td></tr><tr><td>240</td><td>21.44</td><td>9.61</td><td>152.51</td><td>72.41</td><td>68.22</td><td>152.17</td></tr><tr><td rowspan="3">Relative Timestamp (Ours)</td><td>80</td><td>40.49</td><td>38.44</td><td>145.72</td><td>61.07</td><td>60.62</td><td>55.80</td></tr><tr><td>160</td><td>77.74</td><td>75.24</td><td>139.21</td><td>78.04</td><td>75.46</td><td>42.45</td></tr><tr><td>240</td><td>83.65</td><td>82.64</td><td>127.91</td><td>91.13</td><td>86.88</td><td>30.34</td></tr></table>

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Datasets. We conduct experiments on five benchmark datasets: AISHELL-1: A Mandarin corpus with ∼178 hours of speech from 400 speakers [16]. AISHELL-2: A larger Mandarin dataset containing ∼1,000 hours from 1,991 speakers in diverse acoustic environments [17]. Wenet Meeting: A multi-domain Mandarin corpus consisting of 10000+ hours of high-quality labeled speech [18]. LibriSpeech: An English read-speech corpus with ∼1,000 hours of audiobooks, widely adopted for ASR evaluation [19]. Common Voice (English subset): A crowd-sourced corpus with diverse accents and recording conditions [20].

Baseline Models. We compare our proposed method against several state-of-the-art models. Qwen2-Audio: A large language model with integrated audio understanding capabilities, trained using a large amount of diverse data and multiple training methodologies [11]. WhisperTimestamped: This model utilizes Whisperlarge-v3, a robust, multilingual ASR model trained on approximately 680,000 hours of weakly supervised audio data [13]. SenseVoice-Small: A lightweight and efficient ASR system, which is trained on over 400,000 hours of data and supports more than 50 languages [14]. Canary: Employs a data-driven approach to enable word-level timestamp prediction and supports multiple languages other than Chinese [15].

Metric. For timestamp prediction, we introduce the concept of time tolerance. Due to the inherent ambiguity of speech boundaries, a predicted timestamp is considered correct if its difference from the ground-truth timestamp falls within a predefined tolerance [21, 22, 23]. We use Precision and Recall to evaluate the accuracy of these timestamps: Precision measures the proportion of correctly predicted timestamps among all predicted timestamps. Recall measures the proportion of correctly predicted timestamps among all ground-truth timestamps. Additionally, we use the Average Time Difference to quantify the temporal accuracy, defined as:

$$
A v g . D i f f = \frac { \sum _ { i = 1 } ^ { N } \left| t i m e s t a m p _ { g t } ^ { i } - t i m e s t a m p _ { p r e d } ^ { i } \right| } { N }
$$

For ASR analysis, we use Word Error Rate (WER) as the primary metric [24].

Implementation Details. Our training methodology extends the FireRedASR-LLM’s [25] architecture to support timestamp outputs. This model is built upon a conformer audio encoder and the Qwen2- 7B-Instruct backbone LLM. Crucially, the native model cannot generate timestamps.

The model is trained on a cluster of 24 x Ascend 910B (64G) NPUs for 7k steps, with each NPU handling a batch duration of 500 seconds. Setting λ as 1 increases by 1 for each epoch, and empirically setting masking probability as 10% from the second epoch. For training, we use the AdamW optimizer with a low learning rate of $5 \times 1 0 ^ { - 6 }$ and a WarmupCosineLR scheduler to ensure stable convergence. The entire process utilizes bf16 mixed-precision training for enhanced efficiency. To enable timestamp generation, we incorporate new tokens into the LLM’s vocabulary: Absolute Timestamp Pattern: We add tokens from $< \mid 0 . 0 0 \mid > { \mathrm { t o } } < \mid 3 0 . 0 0 \mid > ,$ representing a time range of 0 to 30 seconds. Relative Timestamp Pattern: We add tokens from $< | 0 . 0 0 | > { \bf t o < } | 5 . 0 0 | >$ , representing time differences up to 5 seconds. Unlike traditional methods that require a special instruction token <|timestamp|> Our model is naturally prompted for this task by a simple, intuitive command like ”Speech to text with timestamp.” Our training regimen was confined to a pair of widely-used datasets: the AISHELL-2 corpus [17] for Mandarin and the English subset of the Common Voice dataset [20].

Table 2. Ablation study on timestamp prediction performance at a tolerance of 240 ms (WER % and Precision/Recall) across different model configurations.
<table><tr><td rowspan="2">Configuration</td><td colspan="3">AISHELL-2 iOS (Chinese)</td><td colspan="3">Common Voice (English)</td></tr><tr><td>WER (%)</td><td>Precision</td><td>Recall</td><td>WER (%)</td><td>Precision</td><td>Recall</td></tr><tr><td>Absolute Timestamp</td><td>2.87</td><td>0.9544</td><td>0.9546</td><td>16.41</td><td>0.7861</td><td>0.7838</td></tr><tr><td>- TS Loss</td><td>2.96</td><td>0.9506</td><td>0.9447</td><td>18.63</td><td>0.7428</td><td>0.7407</td></tr><tr><td>Relative Timestamp</td><td>2.15</td><td>0.9763</td><td>0.9634</td><td>11.63</td><td>0.8770</td><td>0.8546</td></tr><tr><td>- TS Loss</td><td>2.31</td><td>0.9715</td><td>0.9548</td><td>12.66</td><td>0.8712</td><td>0.8463</td></tr><tr><td>- Timestamp Masking</td><td>2.56</td><td>0.9658</td><td>0.9578</td><td>14.47</td><td>0.8067</td><td>0.7853</td></tr></table>

## 3.2. Main Results

Table 1 compares the timestamp prediction performance of various models across the LibriSpeech and Wenet-Meeting datasets.

Our Relative Timestamp method achieved the highest Precision and Recall scores across all tolerance levels on both datasets. For instance, at a 240 ms tolerance, our model reached an impressive 91.13% Precision and 86.88% Recall on Wenet-Meeting, substantially outperforming all baselines. Crucially, our model also exhibited the best temporal accuracy. With an average timestamp difference of just 30.34 ms at a 240 ms tolerance on Wenet-Meeting, our predictions were the closest to the ground truth.

In contrast, Canary performs worse than our method at low tolerance levels but slightly better at high tolerance levels. This discrepancy can be attributed to the fact that Canary was trained on the Librispeech dataset, whereas our model was not.Qwen2-Audio failed on LibriSpeech, with Precision and Recall scores near zero, indicating its inability to handle timestamps on this English corpus.WhisperTimestamped and SenseVoiceSmall generally delivered lower performance, especially at lower tolerances, with their results falling far behind ours. We also conducted an ASR experiment, and the WER results 3 showed that our model outperforms all baselines on all datasets.

In conclusion, this table definitively proves that our Relative Timestamp method sets a new benchmark for timestamp prediction. By focusing on relative temporal relationships, our approach achieves superior accuracy and precision, validating its potential for building high-performance speech language models.

## 3.3. Ablation study

Table 2 presents a detailed ablation study on the performance of different model configurations. Our Relative Timestamp model consistently achieves the best performance across all metrics. The consistently higher Precision and Recall scores further confirm our model’s superior ability to not only transcribe words accurately but also to predict their timestamps with greater precision.

To dissect our method’s key contributions, we conducted further ablations:

Impact of Timestamp Loss (TS Loss): Removing the timestamp loss term (- TS Loss) results in a performance drop for both baseline and our models, as evidenced by an increase in WER and a decrease in Precision/Recall. This highlights the critical role of TS Loss in guiding the model to learn accurate timestamp prediction, preventing it from solely focusing on the text sequence generation task.

Impact of Timestamp Mask: The ablation study on our model reveals a significant performance degradation when the timestamp mask is removed (- Timestamp Mask). The WER increases from 2.15% to 2.56% on AISHELL-2 iOS and from 11.63% to 14.47% on

Common Voice. This compellingly demonstrates that the Timestamp Mask is crucial for the training process. It helps the model focus on valid timestamp predictions and filters out noise, ensuring the robustness of our relative timestamp.

## 3.4. Analysis of ASR

As depicted in Table 3, a comprehensive performance comparison based on Word Error Rate (WER) is presented across five diverse datasets. Our proposed method, Relative Timestamp, consistently demonstrates superior performance across all evaluated datasets. This is visually confirmed by its polygon having the largest area, signifying its robust generalization and leading performance across multilingual and domain-specific tasks. Specifically, our model achieves WERs of 1.26% and 2.15% on the Chinese datasets AISHELL-1 (AS-1) and AISHELL-2 (AS-2), respectively, significantly outperforming all baselines. On the more challenging Wenet Meeting (Wenet) dataset, our model’s WER of 5.56% establishes a substantial lead. Furthermore, our approach also secures the best results on the English datasets LibriSpeech (Libri) (2.78%) and Common Voice (CV) (11.63%).

Table 3. WER (%) comparison. Lower is better (↓).
<table><tr><td>Method</td><td>AS-1 (CN)</td><td>AS-2 (CN)</td><td>Wenet (CN)</td><td>Libri (EN)</td><td>CV (EN)</td></tr><tr><td>Qwen2 Audio</td><td>1.62</td><td>3.38</td><td>21.99</td><td>43.32</td><td>74.37</td></tr><tr><td>SenseVoiceSmall</td><td>3.10</td><td>3.87</td><td>7.54</td><td>3.80</td><td>15.97</td></tr><tr><td>WhisperTimestamped</td><td>12.46</td><td>8.08</td><td>56.66</td><td>20.50</td><td>38.09</td></tr><tr><td>Absolute Timestamp</td><td>1.38</td><td>2.87</td><td>11.30</td><td>9.11</td><td>16.41</td></tr><tr><td>Ours (Relative Timestamp)</td><td>1.26</td><td>2.15</td><td>5.56</td><td>2.78</td><td>11.63</td></tr></table>

## 4. CONCLUSION

We propose a novel framework aimed at addressing the core challenges of existing speech models in precise temporal sequence understanding. We have gone beyond the simple method of adding timestamp tags to LLMs, instead focusing on endowing the model with fine-grained, aligned multi-modal spatiotemporal understanding capabilities, thereby upgrading LLMs from “content understanding machines” to “temporal-aware content understanding machines”.

## 5. ACKNOWLEDGEMENTS

This was supported by Jiangsu Province Frontier Program Project (BF2025036), Hong Kong RGC grant GRF #15611021, NSFC grant (No. 62376178), and Jiangsu Key Lab of Language Computing.

## 6. REFERENCES

[1] Rotem Rousso, Eyal Cohen, Joseph Keshet, and Eleanor Chodroff, “Tradition or innovation: A comparison of modern asr methods for forced alignment,” arXiv preprint arXiv:2406.19363, 2024.

[2] Jing Peng, Yucheng Wang, Yu Xi, Xu Li, Xizhuo Zhang, and Kai Yu, “A survey on speech large language models,” arXiv e-prints, pp. arXiv–2410, 2024.

[3] Sara Papi, Peidong Wang, Junkun Chen, Jian Xue, Naoyuki Kanda, Jinyu Li, and Yashesh Gaur, “Leveraging timestamp information for serialized joint streaming recognition and translation,” in ICASSP 2024-2024. IEEE, 2024, pp. 10381–10385.

[4] Sunit Sivasankaran, Eric Sun, Jinyu Li, Yan Huang, and Jing Pan, “Target word activity detector: An approach to obtain asr word boundaries without lexicon,” in ICASSP 2025. IEEE, 2025, pp. 1–5.

[5] Naoki Makishima, Keita Suzuki, Satoshi Suzuki, Atsushi Ando, and Ryo Masumura, “Joint autoregressive modeling of end-to-end multi-talker overlapped speech recognition and utterance-level timestamp prediction,” in Proc. INTER-SPEECH, 2023, 2023, pp. 2913–2917.

[6] Ke Hu, Krishna Puvvada, Elena Rastorgueva, Zhehuai Chen, He Huang, Shuoyang Ding, Kunal Dhawan, Hainan Xu, Jagadeesh Balam, and Boris Ginsburg, “Word level timestamp generation for automatic speech recognition and translation,” arXiv preprint arXiv:2505.15646, 2025.

[7] Hiroyoshi Yamasaki, Jer´ ome Louradour, Julie Hunter, andˆ Laurent Prevot, “Transcribing and aligning conversational´ speech: A hybrid pipeline applied to french conversations,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–6.

[8] Tianshu Yu, Zihan Gong, Minghuan Tan, Guhong Chen, and Min Yang, “Unsupervised speech-text word-level alignment with dynamic programming,” in Findings of the Association for Computational Linguistics: NAACL 2025, 2025, pp. 2323– 2334.

[9] Quanwei Tang, Sophia Yat Mei Lee, Junshuang Wu, Dong Zhang, Shoushan Li, Erik Cambria, and Guodong Zhou, “A comprehensive graph framework for question answering with mode-seeking preference alignment,” in Findings of ACL 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, Eds., Vienna, Austria, July 2025, pp. 21504–21523, Association for Computational Linguistics.

[10] Dan Wu, Xincheng Ju, Dong Zhang, Shoushan Li, Erik Cambria, and Guodong Zhou, “Emotion across modalities and cultures: Multilingual multimodal emotion-cause analysis with memory-inspired framework,” in Proceedings ofthe 33rdACM International Conference on Multimedia, 2025, pp. 5775– 5783.

[11] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou, “Qwen2-audio technical report,” 2024.

[12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al., “Lora: Low-rank adaptation of large language models.,” ICLR, vol. 1, no. 2, pp. 3, 2022.

[13] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” 2022.

[14] Keyu An, Qian Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Yue Gu, Ting He, Hangrui Hu, Kai Hu, Shengpeng Ji, Yabin Li, Zerui Li, Heng Lu, Haoneng Luo, Xiang Lv, Bin Ma, Ziyang Ma, Chongjia Ni, Changhe Song, Jiaqi Shi, Xian Shi, Hao Wang, Wen Wang, Yuxuan Wang, Zhangyu Xiao, Zhijie Yan, Yexin Yang, Bin Zhang, Qinglin Zhang, Shiliang Zhang, Nan Zhao, and Siqi Zheng, “Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms,” CoRR, vol. abs/2407.04051, 2024.

[15] Ke Hu, Krishna Puvvada, Elena Rastorgueva, Zhehuai Chen, He Huang, Shuoyang Ding, Kunal Dhawan, Hainan Xu, Jagadeesh Balam, and Boris Ginsburg, “Word Level Timestamp Generation for Automatic Speech Recognition and Translation,” in Interspeech 2025, 2025, pp. 2565–2569.

[16] Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng, “Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline.,” in O-COCOSDA. 2017, pp. 1–5, IEEE.

[17] Jiayu Du, Xingyu Na, Xuechen Liu, and Hui Bu, “Aishell-2: Transforming mandarin asr research into industrial scale,” 2018.

[18] Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenchen Zeng, et al., “Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition,” in ICASSP 2022. IEEE, 2022, pp. 6182–6186.

[19] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, “Librispeech: An asr corpus based on public domain audio books,” in ICASSP 2015, April 2015, pp. 5206– 5210.

[20] Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, Marseille, France, May 2020, pp. 4218–4222, European Language Resources Association.

[21] Elena Rastorgueva, Vitaly Lavrukhin, and Boris Ginsburg, “Nemo forced aligner and its application to word alignment for subtitle generation,” in Proc. Interspeech, 2023.

[22] Seung-Eun Kim, Bronya R Chernyak, Olga Seleznova, Joseph Keshet, Matthew Goldrick, and Ann R Bradlow, “Automatic recognition of second language speech-in-noise,” JASA Express Letters, vol. 4, no. 2, 2024.

[23] Yiming Ji, Suyang Zhu, Dong Zhang, and Shoushan Li, “Pathological section staining transferring with tailored metricbased model selection,” in ICASSP 2025. IEEE, 2025, pp. 1–5.

[24] Hainan Xu, Fei Jia, Somshubra Majumdar, He Huang, Shinji Watanabe, and Boris Ginsburg, “Efficient sequence transduction by jointly predicting tokens and durations,” in International Conference on Machine Learning. PMLR, 2023, pp. 38462–38484.

[25] Kai-Tuo Xu, Feng-Long Xie, Xu Tang, and Yao Hu, “Fireredasr: Open-source industrial-grade mandarin speech recognition models from encoder-decoder to llm integration,” arXiv preprint arXiv:2501.14350, 2025.