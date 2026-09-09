# Noise Adaptive Streaming Audio-Visual Speech Token Enhancement for Robust Full-Duplex Spoken Dialogue Models

Bella Godiva, Yeonju Kim, Yong Man Ro Integrated Vision and Language Lab, KAIST, South Korea {bellagodiva, yeonju7.kim, ymro}@kaist.ac.kr

## Abstract

Full-duplex spoken dialogue systems enable simultaneous listening and speaking, but their audio-only perception often fails under background noise and overlapping speech, leading to incoherent responses. Recent audiovisual dialogue approaches show that incorporating visual cues such as lip movements improve robustness under audio corruption. However, existing approaches often adapt the large speech dialogue model itself to process visual input, requiring costly multimodal training. We propose AV-STE, a modular streaming audio-visual front-end that restores corrupted semantic speech tokens from noisy audio and lip video before they reach the speech LLM. The downstream dialogue model remains entirely frozen, preserving its pretrained conversational capabilities. When integrated with frozen Moshi, AV-STE improves average GPT-4ojudged response coherence from 1.42 to 1.91 under same-dataset speaker interference while largely preserving turn-taking behavior. Gains also transfer to out-of-domain Seamless Interaction. Code and models are available: https: //github.com/bellagodiva/av-ste.git

## 1 Introduction

Full-duplex spoken dialogue models enable realtime interaction by listening and speaking simultaneously, supporting natural behaviors such as interruptions and backchannels. However, their robustness to acoustic interference remains underexplored. In real-world environments, background noise and competing speakers can corrupt the user’s speech representation, leading to degraded understanding and less coherent responses.

Since current full-duplex dialogue models depend primarily on audio input, their performance can degrade substantially under noisy conditions. In human communication, visual cues such as lip movements provide complementary information when speech is ambiguous or corrupted. Prior work in audio-visual speech recognition (AVSR) (Ma et al., 2023; Yeo et al., 2025) has demonstrated the benefit of visual information for robust speech understanding. However, AVSR systems primarily predict text, whereas native speech-to-speech dialogue models operate directly on speech-token representations and must process user input continuously with low latency. Using AVSR would introduce an additional speech-to-text and text-tospeech-token conversion stage. We therefore focus on directly enhancing corrupted speech tokens using visual information.

![](images/2146d99c26a15679be05a3e575613514070ee1a7eca57aea14dd41898fe5c5f4.jpg)  
Figure 1: In noisy environments, background sounds such as music, crowd noise, or interfering speech can degrade speech perception in full-duplex spoken dialogue models, leading to irrelevant responses. AV-STE uses visual lip cue to enhance corrupted speech token and improve response coherence under noisy conditions, without retraining the underlying speech LLM.

AV-Dialog (Chen et al., 2025b) further demonstrates that visual information can improve spoken dialogue robustness by grounding the dialogue model in audio-visual input through large-scale training on paired audio-visual conversational data. However, such training is computationally expensive and data-intensive. Moreover, adapting a pretrained dialogue model to a new modality requires careful training to retain its existing language and conversational capabilities. This motivates a practical question: can visual information improve dialogue robustness without retraining the speech LLM itself?

To this end, we propose AV-STE, a streaming audio-visual semantic token enhancement framework for robust full-duplex dialogue. Rather than adapting the dialogue model to process visual input, AV-STE uses a streaming audio-visual encoder and noise-adaptive fusion to restore corrupted semantic speech tokens from noisy audio and lip video before they reach the dialogue model. The enhanced tokens are streamed directly into the pretrained speech LLM, which remains entirely frozen, avoiding costly audio-visual retraining and enabling plug-and-play integration with existing speech-to-speech dialogue systems.

Experiments on LRS3 and Seamless Interaction show that AV-STE substantially improves semantictoken recovery under non-speech noise and speaker interference, including same-dataset multi-speaker interference and out-of-domain conditions. When integrated with frozen Moshi, AV-STE consistently improves response coherence across noisy SNR levels while preserving turn-taking behavior. These results demonstrate that visual speech cues can improve the robustness of full-duplex spoken dialogue by recovering corrupted semantic speech tokens without retraining the underlying speech LLM.

## 2 Related Work

## 2.1 Full-Duplex Spoken Dialogue Models

Recent speech LLMs can be broadly categorized as half- or full-duplex depending on whether they can listen and speak simultaneously. Many speech LLMs operate in a half-duplex manner (Zeng et al., 2024; Xu et al., 2025; Ding et al., 2025), alternating between listening and speaking. While effective for turn-based interaction, this limits natural conversational behaviors such as interruption and backchanneling.

Full-duplex speech LLMs (Défossez et al., 2024; Veluri et al., 2024; Zhang et al., 2025; Wang et al., 2024; Chen et al., 2025a; Yu et al., 2025) instead process streaming user speech while generating responses, enabling more natural turn-taking. Some systems use modular listening and speaking components, such as Freeze-Omni (Wang et al., 2024) and MinMo (Chen et al., 2025a), whereas models such as Moshi (Défossez et al., 2024) operate directly over speech-token streams within a unified architecture. Our work builds on this latter setting and improves robustness to acoustic corruption while keeping the pretrained speech LLM frozen.

## 2.2 Audio-Visual Spoken Dialogue Models

Recent work has also incorporated visual information directly into spoken dialogue systems. MoshiVis (Royer et al., 2025) extends Moshi with visual input for understanding and discussing image content. In contrast, AV-STE uses continuous lip video as a complementary signal for recovering speech information under acoustic corruption.

AV-Dialog (Chen et al., 2025b) shares a similar motivation of incorporating user face video into dialogue systems. However, it incorporates audiovisual grounding by training the dialogue model on paired audio-visual conversational data. AV-STE instead introduces visual information at the speechtoken level, restoring corrupted semantic tokens before they reach the dialogue model. This allows the underlying speech LLM to remain frozen and enables AV-STE to serve as a plug-and-play robustness module for pretrained speech-to-speech dialogue systems.

## 2.3 Speech Token Representations

Recent full-duplex speech dialogue systems operate directly on discrete speech token representations rather than text. Discrete speech representations generally capture different levels of speech information. Semantic tokens derived from selfsupervised speech models such as HuBERT (Hsu et al., 2021) emphasize linguistic content, while neural audio codecs such as EnCodec (Défossez et al., 2022) and SoundStream (Zeghidour et al., 2021) encode acoustic information needed for waveform reconstruction.

More recent tokenizers, including SpeechTokenizer (Zhang et al., 2024) and Mimi (Défossez et al., 2024), combine semantic and acoustic information within a multi-codebook representation. In Mimi, the first codebook is semantically distilled, while the remaining codebooks primarily encode acoustic information. AV-STE specifically enhances this semantic-token stream while retaining the acoustic-token streams unchanged, allowing the enhanced representation to be consumed directly by Moshi. Although we evaluate AV-STE with Mimi and Moshi, the token-level formulation could be adapted to other speech models that use semantic-token representations.

![](images/6c90efbc56640fa5db9225c2a21355e62d202e6385256f8e30844bfcad792c35.jpg)

![](images/288ecfa437c31bfd6038d50add58a66a68082d35c1f495780f1d4bdd59855c7e.jpg)

![](images/94643debcc8ae5638dad3edb029d42c33269e46e1fe953c648301a629ac38a24.jpg)  
(b)	Integration	with	Full-Duplex	Model  
(c)	Bounded-Lookahead	Mask  
Figure 2: Overview of AV-STE. (a) Given noisy speech and the corresponding lip video, AV-STE first extracts visual speech features using a Streaming Audio-Visual Encoder. The Noise-Adaptive Fusion module then combines these features with speech-tokenizer outputs through soft token cross-attention and noise-adaptive modulation to predict clean semantic tokens. (b) The enhanced semantic tokens are streamed into a frozen full-duplex speech LLM together with the acoustic tokens from the speech tokenizer, enabling more robust response generation under noisy conditions. (c) A bounded-lookahead mask is used to preserve low-latency streaming inference

## 3 Audio-Visual Speech Token Enhancer (AV-STE)

## 3.1 Overview

We propose AV-STE, a lightweight audio-visual speech token enhancement module that improves the robustness of frozen full-duplex speech dialogue models under noisy and interfering-speaker conditions. In this work, we focus on semantic speech tokens because they primarily encode linguistic content, which is most critical for speech understanding and coherent response generation. This choice is also motivated by the nature of visual speech, where lip movements provide strong phonetic cues but limited information about finegrained acoustic attributes such as timbre, prosody, and background noise. We therefore focus on enhancing the semantically distilled token stream while retaining the original acoustic-token streams unchanged.

Given noisy semantic speech tokens $\mathbf { z } ^ { \mathrm { n o i s y } }$ and synchronized facial video v of the target speaker, AV-STE predicts clean semantic speech tokens y before they are fed into a frozen speech LLM:

$$
\mathbf { y } \sim p _ { \boldsymbol { \theta } } \left( \mathbf { y } \mid \mathbf { z } ^ { \mathrm { { n o i s y } } } , \mathbf { v } \right) .\tag{1}
$$

The proposed framework consists of two main components: (1) a Streaming Audio-Visual Encoder, which extracts visual speech features from noisy audio and synchronized lip video, and (2) a Noise-Adaptive Fusion module, which uses crossattention to combine visual speech features with noisy tokenizer representations and adaptively adjusts their contribution according to tokenizer uncertainty. The resulting enhanced semantic tokens are then passed directly to the frozen speech LLM, enabling robust audio-visual input processing without modifying the underlying dialogue model. The full architecture of AV-STE is illustrated in Figure 2.

## 3.2 Streaming AV Encoder

We initialize the Streaming AV Encoder from AV-HuBERT (Shi et al., 2022), a self-supervised audiovisual speech model that achieves strong performance on audio-visual speech recognition. AV-HuBERT consists of modality-specific audio and visual encoders followed by a shared Transformer encoder, which extracts synchronized audio-visual features from input speech waveforms and facial video frames at 25 Hz. Since the original AV-HuBERT is bidirectional and designed for offline inference, we modify the backbone to support lowlatency streaming inference.

First, we replace all temporal convolutions in the video frontend with causal convolutions using leftonly temporal padding, ensuring that each visual feature depends only on current and past frames. Second, we replace bidirectional self-attention in every Transformer layer with causal attention using a bounded-lookahead mask (Moritz et al., 2020; Shi et al., 2020), as illustrated in Figure 2(c). Specifically, we add the following mask to the pre-softmax attention logits:

$$
M _ { i j } = \left\{ { \begin{array} { l l } { 0 , } & { j \leq i + L , } \\ { - \infty , } & { j > i + L , } \end{array} } \right.\tag{2}
$$

where L denotes the allowed future lookahead in frames. When $L = 0$ , the model is strictly causal; when $L > 0$ , the model accesses a bounded number of future frames while preserving streaming inference through fixed-latency future context. A small lookahead improves temporal audio-visual alignment while maintaining low streaming latency. We use L = 4 (160 ms at 25 Hz) in our main experiments and further ablate different lookahead sizes in Figure 4.

To match the target speech token rate used by the downstream speech LLM, we additionally insert a strided causal convolution after the transformer encoder to downsample the feature sequence while preserving the streaming constraint.

## 3.3 Noise-Adaptive Fusion

The streaming AV encoder produces visual speech features $\mathbf { h } _ { t } ^ { \mathrm { a v } }$ that are robust to background noise and interfering speakers. However, visual speech cues can be phoneme-ambiguous, since multiple phonemes may correspond to identical mouth shape (Yeo et al., 2024), making AV encoder predictions less reliable than the original speech tokens under clean and low-noise conditions. In contrast, the speech tokenizer provides fine-grained speech tokens learned from large-scale speech data, but its predictions may degrade under severe noise. To exploit their complementary strengths, we introduce a noise-adaptive fusion module that incorporates tokenizer predictions into the AV representation through cross-attention and adaptively modulates their contribution based on tokenizer uncertainty.

Soft Token Cross Attention. Directly attending to hard token embeddings can be unreliable under noisy or interfering-speaker conditions, where the tokenizer distribution is often ambiguous. Hard arg max selection discards this uncertainty and may provide misleading token-level evidence to the cross-attention module. We therefore attend to the soft tokenizer distribution instead of discrete token embeddings. Specifically, we compute a distribution over tokenizer logits and use it to form a soft token representation:

$$
\mathbf { p } _ { t } = \mathrm { s o f t m a x } \left( \frac { \mathbf { l } _ { t } } { \tau } \right) , \qquad \mathbf { e } _ { t } = \mathbf { p } _ { t } \mathbf { E } \mathbf { W } _ { \mathrm { p r o j } } ,\tag{3}
$$

where $\mathbf { l } _ { t } \in \mathbb { R } ^ { V }$ denotes the tokenizer logits, τ is a temperature hyperparameter, E $\in \mathbb { R } ^ { \bar { V } \times d _ { e } }$ is a token embedding table learned jointly with the fusion module, and ${ \bf W } _ { \mathrm { p r o j } } \in \mathbb { R } ^ { d _ { e } \times d }$ projects the soft embedding to the model dimension.

We then fuse $\mathbf { e } _ { t }$ with the audio-visual features through cross-attention (Vaswani et al., 2017):

$$
\hat { \mathbf { c } } _ { t } = \mathrm { C r o s s A t t n } ( \mathbf { h } _ { t } ^ { \mathrm { a v } } , \ \mathbf { e } _ { 1 : T } , \ \mathbf { e } _ { 1 : T } ) ,\tag{4}
$$

$$
\mathbf { c } _ { t } = \mathrm { L a y e r N o r m } ( \mathbf { h } _ { t } ^ { \mathrm { a v } } + \hat { \mathbf { c } } _ { t } ) .\tag{5}
$$

We use AV features $\mathbf { h } _ { t } ^ { \mathrm { a v } }$ as queries and soft token embeddings $\mathbf { e } _ { t }$ as keys/values, allowing noiserobust visual cues to guide retrieval from the corrupted speech-token representation. The reverse direction performs worse under noise (Appendix A).

To support streaming inference, we apply a bounded-lookahead mask (Figure 2c) that restricts each query position t to attend only to past keyvalue positions $t ^ { \prime } \leq t + L$ , consistent with the backbone lookahead L.

Noise-Adaptive Modulation. Although crossattention incorporates the speech-token representation into the AV representation, the speech-token representation should not contribute equally under all noise conditions. In clean or low-noise speech, the tokenizer predictions are usually reliable. However, in severe noise or interfering speech, the tokenizer distribution may contain incorrect acoustic information. We therefore introduce a frame-level modulation $\lambda _ { t } \in [ 0 , 1 ]$ that adaptively controls how much weight the final representation places on the speech-token representation $\mathbf { e } _ { t }$ versus the crossattended AV representation $\mathbf { c } _ { t }$ depending on the noise level.

We use the normalized entropy of the tokenizer distribution as a reliability prior, since entropy measures uncertainty over the full token distribution (Laptev and Ginsburg, 2023):

$$
\begin{array} { r } { \hat { H } _ { t } = \frac { - \sum _ { v = 1 } ^ { V } p _ { t , v } \log p _ { t , v } } { \log V } , } \\ { \hat { \lambda } _ { t } ^ { \mathrm { e n t } } = 1 - \hat { H } _ { t } . \qquad } \end{array}\tag{6}
$$

where $\hat { H } _ { t } \in [ 0 , 1 ]$ denotes the normalized entropy. A lower entropy indicates a sharper tokenizer distribution and thus higher estimated reliability, while a higher entropy indicates greater uncertainty.

Because entropy alone may not capture all reliability cues, we refine this prior with a learnable modulation gate conditioned on both the speechtoken and AV representations:

$$
\lambda _ { t } = \sigma \left( w \widehat { \lambda } _ { t } ^ { \mathrm { e n t } } + \mathrm { M L P } ( \left[ \mathbf { e } _ { t } \parallel \mathbf { h } _ { t } ^ { \mathrm { a v } } \right] ) \right) ,\tag{7}
$$

where w is a learnable scalar initialized to 1. The final layer of the MLP is zero-initialized, so the gate initially follows the entropy-based prior and gradually learns residual corrections during training.

The final fused representation is computed as

$$
\tilde { \mathbf { h } } _ { t } = \lambda _ { t } \mathbf { e } _ { t } + \big ( 1 - \lambda _ { t } \big ) \mathbf { c } _ { t } .\tag{8}
$$

Rather than following a fixed rule, $\lambda _ { t }$ is learned jointly with the rest of the model, allowing it to adaptively weight the speech-token and AV representations based on the noise condition.

Finally, a linear projection maps the fused representation $\tilde { \mathbf { h } } _ { t }$ to logits over the semantic token vocabulary, and the full AV-STE model is trained end-to-end using frame-wise cross-entropy against clean target speech tokens.

## 4 Experimental Setup

## 4.1 Datasets and Preprocessing

LRS3. (Afouras et al., 2018) contains approximately 433 hours of audio-visual speech, with video at 25 fps and audio at 16 kHz. We use LRS3 as the target-speech dataset for training and indomain evaluation. Following the official LRS3 preprocessing pipeline, we crop the mouth region from each face video and convert it into 96×96 grayscale frames.

AudioSet. (Gemmeke et al., 2017) is a largescale audio event dataset containing 2,084,320 human-labeled 10-second clips from YouTube videos. It covers a broad range of sounds, including human and animal sounds, musical instruments, music genres, and everyday environmental noise. We sample environmental sounds as non-speech noise and conversational speech as cross-dataset speaker interference.

Seamless Interaction. (Agrawal et al., 2025) is a large-scale dyadic interaction video corpus, used here exclusively for out-of-domain evaluation with no fine-tuning. Videos are recorded at 1080×1920 and 30 fps with 48 kHz audio; we resample audio to 16 kHz and crop a 96×96 grayscale mouth region at 25 fps to match the LRS3 preprocessing format. No Seamless Interaction data is used in AV-STE training.

## 4.2 Training and Evaluation Data Synthesis

Initial training. For initial AV-STE training, each LRS3 utterance is sampled as clean speech (20%), mixed with AudioSet non-speech noise (40%), or mixed with AudioSet conversational speech (40%). For noisy samples, the SNR is uniformly sampled from -10 to 10dB.

Same-dataset speaker interference. To reduce shortcuts from dataset-specific characteristics, following Chen et al. (2025b), we further train the initial checkpoint by replacing AudioSet conversational speech with interference from 1–4 randomly selected LRS3 speakers. Interfering speakers are randomly sampled and mixed with the target utterance on the fly during training. We retain the same 20/40/40 sampling ratio and SNR range. This resulting checkpoint is used for all subsequent samedataset and out-of-domain evaluations.

Out-of-domain evaluation. For Seamless Interaction, we use AudioSet for non-speech noise and 1-4 randomly selected Seamless speakers for speaker interference. No Seamless Interaction data is used for training.

Single-turn dialogue evaluation. Following Full-Duplex-Bench (Lin et al., 2025), we evaluate individual utterances rather than multi-turn dialogues. We select 125 utterances each from LRS3 and Seamless Interaction, each 3-10s long. The Seamless dialog set is also used for semantic-token accuracy evaluation.

Table 1: Semantic-token accuracy (%) across in-domain, same-dataset speaker-interference, and out-of-domain settings at five SNR levels. AV-STE is first trained on LRS3 with AudioSet non-speech noise and speaker interference, then further trained with LRS3 same-dataset speaker interference before evaluation on the same-dataset speaker interference and out-of-domain settings.
<table><tr><td>Target</td><td>Model</td><td>Noise / Interference</td><td>Clean</td><td>10 dB</td><td>5dB</td><td>0 dB</td><td>-5dB</td><td>-10 dB</td><td>Avg.</td></tr><tr><td colspan="10">In-domain LRS3</td></tr><tr><td>LRS3</td><td>Mimi</td><td>Non-speech (AudioSet)</td><td>100.00</td><td>59.11</td><td>41.30</td><td>22.62</td><td>10.33</td><td>4.81</td><td>27.63</td></tr><tr><td>LRS3</td><td> $\mathrm { M i m i + A V { - } S T E }$ </td><td>Non-speech (AudioSet)</td><td>76.78</td><td>73.48</td><td>71.87</td><td>69.48</td><td>65.34</td><td>59.92</td><td>68.02</td></tr><tr><td>LRS3</td><td>Mimi</td><td>Speaker (AudioSet)</td><td>100.00</td><td>54.81</td><td>37.61</td><td>22.99</td><td>12.25</td><td>7.19</td><td>26.97</td></tr><tr><td>LRS3</td><td> $\mathrm { M i m i + A V { - } S T E }$ </td><td>Speaker (AudioSet)</td><td>76.78</td><td>73.17</td><td>71.76</td><td>69.20</td><td>65.82</td><td>60.52</td><td>68.09</td></tr><tr><td colspan="10">Same-dataset LRS3 speaker interference</td></tr><tr><td>LRS3</td><td>Mimi</td><td>Speaker (LRS3)</td><td>100.00</td><td>38.43</td><td>15.83</td><td>5.26</td><td>2.22</td><td>1.31</td><td>12.61</td></tr><tr><td>LRS3</td><td> $\mathrm { M i m i + A V { - } S T E }$ </td><td>Speaker (LRS3)</td><td>76.78</td><td>72.46</td><td>70.27</td><td>66.94</td><td>62.17</td><td>56.18</td><td>65.60</td></tr><tr><td colspan="10">Out-of-domain Seamless Interaction</td></tr><tr><td>Seamless</td><td>Mimi</td><td>Non-speech (AudioSet)</td><td>100.00</td><td>36.80</td><td>22.49</td><td>12.35</td><td>5.93</td><td>3.77</td><td>16.27</td></tr><tr><td>Seamless</td><td> $\mathrm { M i m i + A V { - } S T E }$ </td><td>Non-speech (AudioSet)</td><td>43.79</td><td>41.89</td><td>40.12</td><td>38.11</td><td>34.42</td><td>29.28</td><td>36.76</td></tr><tr><td>Seamless</td><td>Mimi</td><td>Speaker (Seamless)</td><td>100.00</td><td>35.21</td><td>21.13</td><td>12.14</td><td>6.92</td><td>3.62</td><td>15.80</td></tr><tr><td>Seamless</td><td> $\mathrm { M i m i + A V { - } S T E }$ </td><td>Speaker (Seamless)</td><td>43.79</td><td>38.66</td><td>37.29</td><td>35.97</td><td>33.42</td><td>29.48</td><td>34.96</td></tr></table>

## 4.3 Training Configuration

We optimize AV-STE with Adam $( \beta _ { 1 } ~ = ~ 0 . 9 ,$ $\beta _ { 2 } = 0 . 9 8 )$ using a tri-stage learning-rate schedule with 4,000 warmup steps followed by 35,000 decay steps to 5% of the peak learning rate of $1 0 ^ { - 4 }$ Gradients are clipped to a maximum norm of 5.0. Training runs for up to 6 epochs (approximately 73 GPU hours) with FP16 mixed precision on a single A6000 GPU. We select the checkpoint with the lowest validation loss.

## 4.4 Model Configuration

We use Moshi (Défossez et al., 2024) as the frozen full-duplex spoken dialogue model and its Mimi tokenizer as the speech-token representation. Mimi uses a hybrid semantic-acoustic representation: its first codebook is semantically distilled, while the remaining codebooks primarily encode acoustic information. AV-STE enhances only the first semantic-token stream; the acoustic-token streams remain unchanged and are passed with the enhanced tokens to Moshi.

We use AV-HuBERT Large (Shi et al., 2022) as the audio-visual encoder. Our main streaming configuration uses a 4-frame lookahead (L=4), corresponding to 160 ms at 25 Hz. The soft-token crossattention module uses 4 heads, 256-dimensional token embeddings, dropout of 0.1, and temperature $\tau { = } 1 . 0$ . A linear projection maps the 1024- dimensional AV-HuBERT features to Mimi’s 2,048- token semantic vocabulary. The final AV-STE model contains 335M parameters.

## 5 Experimental Results and Analysis

## 5.1 Semantic Speech Token Enhancement Evaluation

In all downstream dialogue experiments, Moshi is kept frozen. AV-STE replaces only the semantictoken stream from Mimi’s semantically distilled first codebook, while the remaining acoustic-token streams are retained unchanged. Since Moshi directly consumes these enhanced semantic tokens, we use semantic-token accuracy against clean Mimi tokens as our primary token-level metric. We additionally report reconstructed-speech WER as a secondary diagnostic metric.

Evaluation protocol. Given noisy audio and the corresponding lip video, AV-STE predicts the enhanced semantic-token sequence. Semantic-token accuracy is computed against Mimi tokens extracted from the corresponding clean speech. For reconstructed-speech WER, the enhanced semantic tokens are combined with the acoustic codebooks from the noisy input and decoded into a waveform using the Mimi decoder. We transcribe the reconstructed waveform with OpenAI Whisper and compute WER against the clean ground-truth transcription.

Semantic-token recovery. Table 1 shows that AV-STE substantially improves semantic-token recovery under both non-speech noise and speaker interference. On LRS3, average accuracy increases from 27.63% to 68.02% under non-speech noise and from 26.97% to 68.09% under AudioSet speaker interference. The gains persist across all evaluated SNRs and become especially pronounced under severe corruption; at −10 dB non-speech noise, accuracy increases from 4.81% to 59.92%. This shows that visual cues provide complementary information when the acoustic representation is corrupted.

Table 2: Ablation of input modalities and AV-STE components on LRS3. Results are average semantic-token accuracy (%) over five SNR levels under AudioSet non-speech noise and speaker interference. S-AVH denotes our streaming AV-HuBERT implementation, NAM is our Noise Adaptive Modulation, and CA is cross-attention. Full per-SNR results are in Appendix B.
<table><tr><td>Model / Variant</td><td>Clean</td><td>Non-Speech</td><td>Speaker</td></tr><tr><td>Mimi</td><td>100.00</td><td>27.63</td><td>26.97</td></tr><tr><td>S-AVH (A)</td><td>75.65</td><td>59.25</td><td>58.15</td></tr><tr><td>S-AVH (V)</td><td>32.41</td><td>35.13</td><td>35.27</td></tr><tr><td>S-AVH (A+V)</td><td>76.47</td><td>67.48</td><td>67.77</td></tr><tr><td>+ Hard-CA</td><td>75.12</td><td>66.37</td><td>66.60</td></tr><tr><td>+ Soft-CA</td><td>76.11</td><td>67.64</td><td>67.86</td></tr><tr><td>+ Soft-CA+NAM (AV-STE)</td><td>76.78</td><td>68.02</td><td>68.09</td></tr></table>

Same-dataset speaker interference. Speaker interference is substantially more challenging when the target and interferers are drawn from the same dataset. Compared with AudioSet speaker interference, Mimi performs worse at every SNR. This may reflect both the greater similarity between target and interfering speech and the 1-4 competing speakers used in this setting. After further training on LRS3 speaker interference, AV-STE performs robustly under same-dataset multi-speaker interference, improving Mimi’s average accuracy from 12.61% to 65.60%. The contrast is especially pronounced at −10 dB, where Mimi drops to 1.31% accuracy while AV-STE maintains 56.18%.

Out-of-domain generalization. On the 125-clip Seamless dialogue evaluation set described in Section 4, AV-STE improves average noisy accuracy from 16.27% to 36.76% under non-speech noise and from 15.80% to 34.96% under same-dataset speaker interference. However, clean accuracy decreases from 76.78% on LRS3 to 43.79% on Seamless, indicating sensitivity to domain shift despite consistent gains under noisy conditions.

Ablation and modality. Table 2 shows that audio-visual S-AVH outperforms audio-only S-AVH under both noise types, confirming the benefit of visual cues. Among the fusion variants, Soft-CA+NAM achieves the highest average accuracy and is therefore used as the final AV-STE configuration. Figure 3 further shows that the benefit of audio-visual fusion increases as SNR decreases.

![](images/0093b810810ed78d1535a69e31ad43d63d92a4ac9a667322a7e62757c94f7eb9.jpg)

Figure 3: Semantic token accuracy (%) across SNR levels, averaged over non-speech noise and speaker interference. The advantage of audio-visual fusion increases as SNR decreases. (Appendix B for details).  
![](images/bc753af9648c0ec064867711d6ac68919165128dbfdbef0f63cbe8f4d99b3335.jpg)  
Figure 4: Semantic token accuracy at 0 dB across lookahead frames, averaged over audio set’s speaker interference and non-speech noise. Accuracy peaks at 4 frames (160 ms), offering the best accuracy–latency trade-off.

Effect of lookahead frames. Increasing lookahead provides more future context but also adds streaming latency. As shown in Figure 4, accuracy improves up to L=4 and shows diminishing returns beyond this point. We therefore use L=4 (160 ms) as the best accuracy–latency trade-off.

Reconstructed WER. Appendix C reports WER as a secondary diagnostic, since Moshi directly consumes enhanced semantic tokens rather than reconstructed speech.

## 5.2 Full-Duplex Turn-Taking and Semantic Coherence Evaluation

We next examine whether improved semantictoken recovery translates into more robust fullduplex dialogue generation. Specifically, we evaluate whether AV-STE improves response coherence under acoustic corruption without disrupting the turn-taking behavior of the underlying speech LLM.

Table 3: Comparison of turn-taking and response semantic coherence under clean and noisy conditions. TOR denotes take-over rate, latency measures response delay in seconds, and GPT denotes GPT-based response coherence. GPT scores are underlined to highlight coherence comparisons, and the better GPT score in each condition is bolded.
<table><tr><td>Noise Type</td><td colspan="6">Non-Speech (AudioSet)</td><td colspan="6">Speaker Interference (AudioSet)</td></tr><tr><td>Models</td><td colspan="3">Moshi</td><td colspan="3">Moshi+AV-STE</td><td colspan="3">Moshi</td><td colspan="3">Moshi+AV-STE</td></tr><tr><td>Metrics</td><td>TOR ↑</td><td>Latency (s) ↓</td><td>GPT↑</td><td>TOR ↑</td><td>Latency (s) ↓</td><td>GPT↑</td><td>TOR↑</td><td>Latency (s) ↓</td><td>GPT↑</td><td>TOR ↑</td><td>Latency (s) ↓</td><td>GPT↑</td></tr><tr><td>Clean</td><td>0.520</td><td>0.720</td><td>2.34</td><td>0.608</td><td>0.534</td><td>2.11</td><td>0.520</td><td>0.720</td><td>2.34</td><td>0.608</td><td>0.534</td><td>2.11</td></tr><tr><td>10 dB</td><td>0.560</td><td>1.056</td><td>1.87</td><td>0.488</td><td>0.553</td><td>2.15</td><td>0.488</td><td>0.730</td><td>2.00</td><td>0.496</td><td>0.756</td><td>2.03</td></tr><tr><td>5 dB</td><td>0.496</td><td>0.777</td><td>2.00</td><td>0.520</td><td>0.959</td><td>2.12</td><td>0.528</td><td>0.872</td><td>1.98</td><td>0.568</td><td>0.614</td><td>2.21</td></tr><tr><td>0 dB</td><td>0.512</td><td>0.926</td><td>1.83</td><td>0.496</td><td>1.053</td><td>1.97</td><td>0.624</td><td>0.793</td><td>1.62</td><td>0.528</td><td>0.795</td><td>1.85</td></tr><tr><td>-5 dB</td><td>0.624</td><td>0.660</td><td>1.33</td><td>0.552</td><td>1.125</td><td>1.86</td><td>0.600</td><td>0.778</td><td>1.39</td><td>0.624</td><td>0.873</td><td>1.90</td></tr><tr><td>-10 dB</td><td>0.544</td><td>0.991</td><td>1.25</td><td>0.520</td><td>1.008</td><td>2.00</td><td>0.656</td><td>0.652</td><td>1.22</td><td>0.568</td><td>0.660</td><td>1.94</td></tr></table>

Table 4: Qualitative example of AV-STE recovering response coherence under −10 dB non-speech noise.  
User: “My mother was on the line to tell me that doctors   
had found a lump in his stomach, terminal cancer .. 9   
Moshi: “Oh, I’m so glad you asked. I’m a big fan of the   
London Underground. I’ve been using it for years.”   
Moshi + AV-STE: “That’s not good ..”

Evaluation protocol. AV-STE takes noisy audio, the corresponding lip video, and noisy semantictoken logits as input and predicts enhanced semantic tokens. These are combined with the unchanged acoustic-token streams from the noisy input and streamed into frozen Moshi (Défossez et al., 2024) for response generation, which serves as our primary backbone and baseline. For same-dataset speaker interference, we additionally compare with PersonaPlex (Roy et al., 2026) and PASE (Rong et al., 2025) + Moshi. PASE is a recent generative speech-enhancement model that leverages WavLM phonological representations to recover clean speech from noisy input. PASE enhances the complete noisy utterance before it is passed to Moshi, providing a non-streaming audio-only enhancement baseline against AV-STE’s streaming audio-visual token enhancement.

Metrics. Following (Lin et al., 2025), we evaluate the generated responses using three metrics (see Appendix D for details):

• Takeover Rate (TOR): The proportion of turns in which the model produces non-silent, non-backchannel speech. A higher TOR indicates more successful turn-taking behavior.

• Latency: The average delay between the end of the user’s utterance and the start of the model’s response, measured in seconds. Lower latency indicates smoother turn-taking.

• GPT: A GPT-based semantic coherence score that measures the relevance of the model’s response to the user’s utterance on a scale from 1 to 5. Higher scores indicate better response coherence.

In-domain dialogue robustness. AV-STE substantially improves response coherence as acoustic corruption becomes more severe, while largely preserving Moshi’s turn-taking behavior. Table 3 shows consistent coherence gains across all noisy SNR levels under both non-speech noise and speaker interference. At -10 dB, for example, GPT coherence increases from 1.25 to 2.00 under nonspeech noise and from 1.22 to 1.94 under speaker interference. In contrast, TOR and latency remain close to those of Moshi, indicating that restoring the semantic-token stream improves response relevance without altering the full-duplex interaction behavior.

Same-dataset speaker interference. Visual grounding is particularly beneficial when competing speech becomes difficult to distinguish acoustically. Under same-dataset LRS3 interference, AV-STE+Moshi achieves the highest average coherence score of 1.91, compared with 1.42 for Moshi, 1.17 for PersonaPlex, and 1.67 for SE (PASE)+Moshi (Table 5). Although PASE performs best at 10 dB, AV-STE performs best from 5 to -10 dB, suggesting that visual cues provide complementary target-speech information as acoustic interference becomes more severe.

Table 5: GPT-4o response coherence (↑) under same-dataset LRS3 speaker interference and out-of-domain Seamless Interaction. Avg. is computed over the five noisy SNR levels. <sup>†</sup>PASE is an offline, non-causal generative speech enhancement (SE) model, making it incompatible with streaming full-duplex interaction and is included only as a non-streaming reference.
<table><tr><td>Target</td><td>Noise / Interference</td><td>Model</td><td>Clean</td><td>10 dB</td><td>5dB</td><td>0 dB</td><td>-5 dB</td><td>-10 dB</td><td>Avg.</td></tr><tr><td rowspan="7">LRS3</td><td rowspan="4">Non-speech (AudioSet)</td><td>PersonaPlex</td><td>1.64</td><td>1.51</td><td>1.46</td><td>1.33</td><td>1.12</td><td>1.11</td><td>1.31</td></tr><tr><td>Moshi</td><td>2.34</td><td>1.87</td><td>2.00</td><td>1.83</td><td>1.33</td><td>1.25</td><td>1.66</td></tr><tr><td>SE (PASE) † + Moshi</td><td>2.23</td><td>2.03</td><td>2.34</td><td>2.21</td><td>2.34</td><td>1.61</td><td>2.11</td></tr><tr><td>AV-STE + Moshi</td><td>2.11</td><td>2.15</td><td>2.12</td><td>1.97</td><td>1.86</td><td>2.00</td><td>2.02</td></tr><tr><td rowspan="4">Speaker (LRS3)</td><td>PersonaPlex</td><td>1.64</td><td>1.45</td><td>1.13</td><td>1.02</td><td>1.03</td><td>1.07</td><td>1.17</td></tr><tr><td>Moshi</td><td>2.34</td><td>1.94</td><td>1.61</td><td>1.25</td><td>1.14</td><td>1.15</td><td>1.42</td></tr><tr><td>SE (PASE) † + Moshi</td><td>2.23</td><td>2.15</td><td>2.00</td><td>1.77</td><td>1.29</td><td>1.12</td><td>1.67</td></tr><tr><td>AV-STE + Moshi</td><td>2.11</td><td>2.00</td><td>2.12</td><td>1.85</td><td>1.86</td><td>1.70</td><td>1.91</td></tr><tr><td rowspan="4">Seamless</td><td rowspan="2">Non-speech (AudioSet)</td><td>Moshi</td><td>2.38</td><td>2.13</td><td>1.68</td><td>1.50</td><td>1.27</td><td>1.30</td><td>1.58</td></tr><tr><td>AV-STE + Moshi</td><td>2.70</td><td>2.27</td><td>2.36</td><td>2.25</td><td>1.78</td><td>1.84</td><td>2.10</td></tr><tr><td rowspan="2">Speaker (Seamless)</td><td>Moshi</td><td>2.38</td><td>1.98</td><td>1.64</td><td>1.44</td><td>1.24</td><td>1.18</td><td>1.50</td></tr><tr><td>AV-STE + Moshi</td><td>2.70</td><td>2.36</td><td>2.34</td><td>1.88</td><td>1.91</td><td>1.64</td><td>2.03</td></tr></table>

Out-of-domain generalization. For Seamless Interaction, we evaluate clean speech, AudioSet nonspeech noise, and same-dataset speaker interference by mixing each target with 1–4 other Seamless speakers. No Seamless Interaction data is used for training. In Seamless Interaction, AV-STE improves average noisy coherence from 1.58 to 2.10 under non-speech noise and from 1.50 to 2.03 under same-dataset speaker interference. Thus, despite the token-level domain sensitivity observed on Seamless Interaction, visual semantic enhancement continues to improve downstream response coherence, suggesting that the recovered representation remains useful to the frozen speech LLM under domain shift.

Qualitative analysis. Table 4 shows a qualitative example where AV-STE recovers response coherence at -10 dB SNR. See Appendix E for more qualitative samples.

## 6 Conclusion

We presented AV-STE, a streaming audio-visual module that improves the robustness of full-duplex spoken dialogue without retraining the underlying speech LLM. AV-STE uses visual speech cues to recover corrupted semantic tokens before they are consumed by frozen Moshi, while leaving the remaining acoustic token streams unchanged. Across non-speech noise, speaker interference, and same-dataset multi-speaker interference, AV-STE substantially improves semantic-token recovery, with particularly large gains under severe acoustic corruption. The improvements also transfer to the out-of-domain Seamless Interaction dataset. Downstream, the enhanced tokens improve spokenresponse coherence across noisy conditions without degrading turn-taking behavior. These results demonstrate the value of visual speech information as a complementary signal for robust full-duplex spoken dialogue systems.

## 7 Limitations

AV-STE currently assumes reliable visual input, and its robustness to visual corruption, occlusion, or missing video remains unexplored. Although AV-STE improves noisy semantic-token recovery on the out-of-domain Seamless Interaction dataset, its lower clean-token accuracy indicates sensitivity to domain shift. Finally, AV-STE introduces a small representation mismatch on clean speech, as its predicted semantic tokens do not exactly reproduce native Mimi tokens. Future work can improve cross-domain robustness using more diverse audio-visual conversational data and jointly consider acoustic and visual corruptions.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2022-NR070162) and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2020-II200004, Development of Previsional Intelligence based on Long-Term Visual Memory Network)

## References

Triantafyllos Afouras, Joon Son Chung, and Andrew Zisserman. 2018. LRS3-TED: a large-scale dataset for visual speech recognition. CoRR, abs/1809.00496.

Vasu Agrawal, Akinniyi Akinyemi, Kathryn Alvero, Morteza Behrooz, Julia Buffalini, Fabio Maria Carlucci, Joy Chen, Junming Chen, Zhang Chen, Shiyang Cheng, Praveen Chowdary, Joe Chuang, Antony D’Avirro, Jon Daly, Ning Dong, Mark Duppenthaler, Cynthia Gao, Jeff Girard, Martin Gleize, and 65 others. 2025. Seamless interaction: Dyadic audiovisual motion modeling and large-scale dataset. Preprint, arXiv:2506.22554.

Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, and 1 others. 2025a. Minmo: A multimodal large language model for seamless voice interaction. arXiv preprint arXiv:2501.06282.

Tuochao Chen, Bandhav Veluri, Hongyu Gong, and Shyamnath Gollakota. 2025b. Av-dialog: Spoken dialogue models with audio-visual input. Preprint, arXiv:2511.11124.

Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, and 1 others. 2025. Kimi-audio technical report. arXiv preprint arXiv:2504.18425.

Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. 2022. High fidelity neural audio compression. Preprint, arXiv:2210.13438.

Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. 2024. Moshi: a speech-text foundation model for real-time dialogue. Preprint, arXiv:2410.00037.

Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. 2017. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780. IEEE.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. CoRR, abs/2106.07447.

Aleksandr Laptev and Boris Ginsburg. 2023. Fast entropy-based methods of word-level confidence estimation for end-to-end automatic speech recognition. In 2022 IEEE spoken language technology workshop (SLT), pages 152–159. IEEE.

Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Gopala Anumanchipalli, Alexander H. Liu, and Hung

yi Lee. 2025. Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turntaking capabilities. Preprint, arXiv:2503.04721.

Pingchuan Ma, Alexandros Haliassos, Adriana Fernandez-Lopez, Honglie Chen, Stavros Petridis, and Maja Pantic. 2023. Auto-avsr: Audio-visual speech recognition with automatic labels. In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), page 1–5. IEEE.

Niko Moritz, Takaaki Hori, and Jonathan Le Roux. 2020. Streaming automatic speech recognition with the transformer model. CoRR, abs/2001.02674.

OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛adry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, and 401 others. 2024. Gpt-4o system card. Preprint, arXiv:2410.21276.

Xiaobin Rong, Qinwen Hu, Mansur Yesilbursa, Kamil Wojcicki, and Jing Lu. 2025. Pase: Leveraging the phonological prior of wavlm for low-hallucination generative speech enhancement. Preprint, arXiv:2511.13300.

Rajarshi Roy, Jonathan Raiman, Sang-gil Lee, Teodor-Dumitru Ene, Robert Kirby, Sungwon Kim, Jaehyeon Kim, and Bryan Catanzaro. 2026. Personaplex: Voice and role control for full duplex conversational speech models. arXiv preprint arXiv:2602.06053.

Amélie Royer, Moritz Böhle, Gabriel de Marmiesse, Laurent Mazaré, Neil Zeghidour, Alexandre Défossez, and Patrick Pérez. 2025. Vision-speech models: Teaching speech models to converse about images. Preprint, arXiv:2503.15633.

Monica Sekoyan, Nithin Rao Koluguri, Nune Tadevosyan, Piotr Zelasko, Travis Bartley, Nikolay Karpov, Jagadeesh Balam, and Boris Ginsburg. 2025. Canary-1b-v2 & parakeet-tdt-0.6b-v3: Efficient and high-performance models for multilingual asr and ast. Preprint, arXiv:2509.14128.

Bowen Shi, Wei-Ning Hsu, Kushal Lakhotia, and Abdelrahman Mohamed. 2022. Learning audio-visual speech representation by masked multimodal cluster prediction. Preprint, arXiv:2201.02184.

Yangyang Shi, Yongqiang Wang, Chunyang Wu, Ching-Feng Yeh, Julian Chan, Frank Zhang, Duc Le, and Michael L. Seltzer. 2020. Emformer: Efficient memory transformer based acoustic model for low latency streaming speech recognition. CoRR, abs/2010.10759.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Bandhav Veluri, Benjamin N Peloquin, Bokai Yu, Hongyu Gong, and Shyamnath Gollakota. 2024. Beyond turn-based interfaces: Synchronous llms as fullduplex dialogue agents. Preprint, arXiv:2409.15594.

Xiong Wang, Yangze Li, Chaoyou Fu, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, and Long Ma. 2024. Freeze-omni: A smart and low latency speech-tospeech dialogue model with frozen llm. arXiv preprint arXiv:2411.00774.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, and 1 others. 2025. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215.

Jeong Hun Yeo, Seunghee Han, Minsu Kim, and Yong Man Ro. 2024. Where visual speech meets language: Vsp-llm framework for efficient and context-aware visual speech processing. Preprint, arXiv:2402.15151.

Jeong Hun Yeo, Hyeongseop Rha, Se Jin Park, and Yong Man Ro. 2025. Mms-llama: Efficient llm-based audio-visual speech recognition with minimal multimodal speech tokens. Preprint, arXiv:2503.11315.

Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, and Chao Zhang. 2025. Salmonnomni: A standalone speech llm without codec injection for full-duplex conversation. Preprint, arXiv:2505.17060.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2021. Soundstream: An end-to-end neural audio codec. Preprint, arXiv:2107.03312.

Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. 2024. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612.

Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang, Siqi Zheng, Jiaqing Liu, Hai Yu, Chao-Hong Tan, Zhihao Du, and ShiLiang Zhang. 2025. OmniFlatten: An end-to-end GPT model for seamless voice conversation. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14570– 14580, Vienna, Austria. Association for Computational Linguistics.

Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, and Xipeng Qiu. 2024. Speechtokenizer: Unified speech tokenizer for speech large language models. Preprint, arXiv:2308.16692.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

## A Cross-Attention Direction Ablation

We use AV features as queries and soft speechtoken embeddings as keys/values, allowing noiserobust AV cues to guide retrieval from corrupted token representations. Reversing the roles degrades WER under both noise conditions, supporting our design choice (Table 6).

Table 6: WER (%) for cross-attention direction. Lower is better.
<table><tr><td></td><td>Clean</td><td>Non-Speech</td><td>Speaker</td></tr><tr><td>AV query (ours)</td><td>7.7</td><td>65.72</td><td>58.74</td></tr><tr><td>Token query</td><td>7.5</td><td>66.33</td><td>59.72</td></tr></table>

## B Detailed Semantic Token Accuracy Results

Table 7 provides the full per-SNR semantic token accuracy results for each noise type. The results support the trend in Figure 3: AV-STE improves over the audio-only Mimi baseline across noise conditions, with the largest gains at low SNRs.

## C Detailed WER Results

Table 8 provides the full per-SNR WER results for each noise type. AV-STE improves over the audioonly Mimi baseline across noise conditions, with the largest gains at low SNRs.

## D Metrics Implementation

Take Over Rate (TOR). For each response audio, we obtain word-level timestamps via automatic speech recognition using NeMo Parakeet-TDT-0.6B (Sekoyan et al., 2025) and merge consecutive word chunks separated by less than 0.8 s into utterances. Following Full-Duplex-Bench (Lin et al., 2025), an utterance is deemed take over and not mere backchannel if it has duration ≥ 1.0 s or word count ≥ 3. However, unlike Full-Duplex-Bench, we count a takeover as successful only if the model response starts within a valid turn-taking window, from 0.5 s before the end of the user’s utterance onward. This prevents premature interruptions from being counted as successful takeovers.

Latency. We compute the mean signed latency δ over all successful takeovers. Because the model may begin responding slightly before the user’s utterance ends (up to 0.5 s of natural overlap), negative latency values are clipped to zero, so the metric reflects only the delay after the user’s turn ends rather than rewarding early interruptions.

GPT-based coherence evaluation. For response semantic coherence, we use GPT-4o (OpenAI et al., 2024) as an automatic judge, following prior work on LLM-based evaluation (Zheng et al., 2023). For each successful takeover, the judge receives the user’s utterance transcript and the generated response, and scores how relevant the generated response is to the user’s utterance on an integer scale from 1 (irrelevant) to 5 (fully relevant). The final GPT score is the mean over all evaluated samples.

We use the following prompt for GPT-based relevance scoring:

You are an expert evaluator for dialogue systems. Your task is to judge how relevant the assistant response is to the user’s question or prior utterance. Evaluate ONLY relevance. Do not evaluate fluency, politeness, grammar, verbosity, factuality, or timing unless they directly affect relevance.

Scoring rubric:

• 5: Fully relevant; directly addresses the question, request, or topic.

• 4: Mostly relevant; clearly related but misses a small part.

• 3: Partially relevant; related but incomplete, vague, or misaligned.

• 2: Weakly relevant; only small overlap with the topic.

• 1: Irrelevant or nearly irrelevant.

[USER\_UTTERANCE]

{user\_text}

[ASSISTANT\_RESPONSE]

{response\_text}

Return valid JSON with exactly these keys:

```json
{
"score": <integer 1–5>,
"reason": "<brief explanation>"
}
```

## E Qualitative Evaluation

Table 9 shows representative examples where AV-STE improves response coherence over the Moshi baseline. Under noisy conditions, Moshi often produces generic or off-topic responses, while AV-STE generates responses that better match the user’s utterance.

Table 7: Semantic-token accuracy (%) across noise types and SNR levels. Non-speech noise and speech interference are sampled from AudioSet. All models use a 4-frame lookahead. Mimi clean accuracy is 100% by definition because its clean tokens are used as the reference targets.
<table><tr><td></td><td>Clean</td><td colspan="6">Non-Speech Noise (AudioSet)</td><td colspan="6">Speech Interference (AudioSet)</td></tr><tr><td>Method</td><td>Clean</td><td>10 dB</td><td>5 dB</td><td>0 dB</td><td>-5 dB</td><td>−10 dB</td><td>Avg.</td><td>10 dB</td><td>5 dB</td><td>0 dB</td><td>-5 dB</td><td>-10 dB</td><td> $\mathbf { A v g } .$ </td></tr><tr><td>Mimi</td><td>100.00†</td><td>59.11</td><td>41.30</td><td>22.62</td><td>10.33</td><td>4.81</td><td>27.63</td><td>54.81</td><td>37.61</td><td>22.99</td><td>12.25</td><td>7.19</td><td>26.97</td></tr><tr><td>S-AV-HuBERT (A)</td><td>75.65</td><td>71.53</td><td>69.19</td><td>64.59</td><td>53.56</td><td>37.38</td><td>59.25</td><td>70.64</td><td>67.76</td><td>62.27</td><td>51.77</td><td>38.31</td><td>58.15</td></tr><tr><td>S-AV-HuBERT (V)</td><td>32.41</td><td>32.69</td><td>33.72</td><td>34.91</td><td>36.45</td><td>37.87</td><td>35.13</td><td>33.26</td><td>34.15</td><td>35.54</td><td>36.22</td><td>37.16</td><td>35.27</td></tr><tr><td>S-AV-HuBERT (A+V)</td><td>76.47</td><td>72.92</td><td>71.43</td><td>68.89</td><td>64.78</td><td>59.38</td><td>67.48</td><td>72.83</td><td>71.50</td><td>68.98</td><td>65.33</td><td>60.23</td><td>67.77</td></tr><tr><td>AV-STE (Hard-CA)</td><td>75.12</td><td>71.58</td><td>70.15</td><td>67.75</td><td>63.78</td><td>58.59</td><td>66.37</td><td>71.54</td><td>70.22</td><td>67.94</td><td>64.17</td><td>59.13</td><td>66.60</td></tr><tr><td>AV-STE (Soft-CA)</td><td>76.11</td><td>73.18</td><td>71.43</td><td>69.25</td><td>64.96</td><td>59.38</td><td>67.64</td><td>73.10</td><td>71.41</td><td>69.03</td><td>65.51</td><td>60.26</td><td>67.86</td></tr><tr><td>AV-STE (Soft-CA+NAM)</td><td>76.78</td><td>73.48</td><td>71.87</td><td>69.48</td><td>65.34</td><td>59.92</td><td>68.02</td><td>73.17</td><td>71.76</td><td>69.20</td><td>65.82</td><td>60.52</td><td>68.09</td></tr></table>

<sup>†</sup>Mimi clean accuracy is 100% by definition; therefore, it is not directly comparable to predicted-token accuracy.

Table 8: Detailed WER (%) results across noise types and SNR levels. Lower is better.
<table><tr><td>Condition</td><td>Mimi</td><td>AV-STE (Soft-CA+NAM)</td><td>AV-STE (Soft-CA)</td><td>AV-STE (Hard-CA)</td><td>S-AV-HuBERT (A+V)</td><td>S-AV-HuBERT (A)</td><td>S-AV-HuBERT (V)</td></tr><tr><td>Clean</td><td>7.6</td><td>7.8</td><td>7.7</td><td>7.7</td><td>7.9</td><td>7.2</td><td>10.8</td></tr><tr><td colspan="8">Non-speech noise (AudioSet)</td></tr><tr><td>10 dB</td><td>21.1</td><td>16.0</td><td>17.4</td><td>16.7</td><td>16.6</td><td>17.4</td><td>22.1</td></tr><tr><td>5 dB</td><td>37.3</td><td>26.0</td><td>26.6</td><td>25.1</td><td>25.6</td><td>27.1</td><td>35.8</td></tr><tr><td>0 dB</td><td>66.6</td><td>49.2</td><td>46.4</td><td>49.1</td><td>46.8</td><td>50.5</td><td>56.5</td></tr><tr><td>-5 dB</td><td>107.6</td><td>84.0</td><td>89.4</td><td>88.5</td><td>87.6</td><td>90.8</td><td>84.8</td></tr><tr><td>-10 dB</td><td>186.2</td><td>139.2</td><td>148.8</td><td>127.2</td><td>155.2</td><td>136.4</td><td>141.8</td></tr><tr><td colspan="8">Speaker interference (AudioSet)</td></tr><tr><td>10 dB</td><td>26.4</td><td>17.3</td><td>17.5</td><td>17.8</td><td>17.3</td><td>18.0</td><td>25.0</td></tr><tr><td>5 dB</td><td>44.7</td><td>30.2</td><td>32.5</td><td>29.7</td><td>34.7</td><td>32.9</td><td>38.2</td></tr><tr><td>0 dB</td><td>75.7</td><td>55.0</td><td>53.7</td><td>56.7</td><td>54.1</td><td>57.1</td><td>63.2</td></tr><tr><td>-5 dB</td><td>100.7</td><td>88.4</td><td>86.0</td><td>87.9</td><td>82.7</td><td>85.5</td><td>89.7</td></tr><tr><td>-10 dB</td><td>126.8</td><td>100.2</td><td>104.0</td><td>116.3</td><td>105.8</td><td>113.5</td><td>124.8</td></tr></table>

Table 9: Qualitative examples where AV-STE improves response coherence over the Moshi baseline. GPT scores are shown in parentheses.
<table><tr><td>Noise</td><td>User Utterance</td><td>Moshi Response</td><td>(Moshi+AV-STE) Response</td></tr><tr><td>Speech -10 dB</td><td>&quot;I think what that means is that people just couldn&#x27;t see what was in front of them.&quot;</td><td>“Sure, I saw them going up the flight of stairs.&quot; (GPT: 1)</td><td>“Yeah, it&#x27;s like they were blind to the reality that was right in front of them.&quot; (GPT: 5)</td></tr><tr><td>Speech -5 dB</td><td>“We have the tools but we lack the collective will and momen- tum.&quot;</td><td>&quot;Yeah, I think they are. I&#x27;ve seen them before.&quot; (GPT: 1)</td><td>“I agree. We need to work to- gether.&quot; (GPT: 4)</td></tr><tr><td>Speech 0 dB</td><td>“So we need a system that helps all our teachers be as good as the best.&quot;</td><td>&quot;I&#x27;m sorry, but I don&#x27;t.&quot; (GPT: 1)</td><td>&quot;Yeah, I agree. We need a system that allows teachers to share their expertise in best practices.&quot; (GPT: 5)</td></tr><tr><td>Speech 5 dB</td><td>&quot;What is the exact relationship be-“The exact relationship be- tween levels of greenhouse gases and planetary..&quot;</td><td>tween six and five is subtlely of one.&quot; (GPT: 1)</td><td>“The relationship between lev- els of greenhouse gases and planetary heating is..&quot; (GPT: 3)</td></tr><tr><td>Speech 10 dB</td><td>“They all look the same under a “Well, it&#x27;s a good thing that microscope, I mean to a first ap- you&#x27;re being careful, but if proximation.&quot;</td><td>you&#x27;re really worried,&quot; (GPT: 1)</td><td>&quot;Yeah, yeah, you&#x27;re right. I mean, they all look alike, don&#x27;t they?&quot; (GPT: 5)</td></tr><tr><td>Non- Speech -10 dB</td><td>“We were making what was invis- ible visible.&quot;</td><td>“The story takes place in the nineteen sixties.&quot; (GPT: 1)</td><td>“Invisible means not able to be seen or detected, while visible means able to be.&quot; (GPT: 2)</td></tr><tr><td>Non- Speech -5 dB</td><td>“We can create a decentralized database that has the same effi- ciency of a monopoly.&quot;</td><td>“Well, the key is to use a mild cleaner and gently rub the stains with a soft cloth.&quot; (GPT: 1)</td><td>“Okay, what do you mean by efficiency?&quot; (GPT: 5)</td></tr><tr><td>Non- Speech 0 dB</td><td>“They are too busy putting out daily fires to think ahead strate- gically.&quot;</td><td>“Stay calm, let&#x27;s get everyone out safely.&quot; (GPT: 1)</td><td>“That&#x27;s true. They have a lot on.&quot; (GPT: 3)</td></tr><tr><td>Non- Speech 5 dB</td><td>“You don&#x27;t have to bring any quarters because the washer and dryer are free.&quot;</td><td>“But I have to do the dishes, it&#x27;s my job.&quot; (GPT: 1)</td><td>“Okay, great. So I don&#x27;t have to bring any quarters.&quot; (GPT: 5)</td></tr><tr><td>Non- Speech 10 dB</td><td>&quot;Without it we literally can&#x27;t in- terface with others.&quot;</td><td>“Yeah, that&#x27;s true.&quot; (GPT: 2)</td><td>“Yeah, it&#x27;s a crucial tool for communication and interac- tion.&quot; (GPT: 5)</td></tr></table>