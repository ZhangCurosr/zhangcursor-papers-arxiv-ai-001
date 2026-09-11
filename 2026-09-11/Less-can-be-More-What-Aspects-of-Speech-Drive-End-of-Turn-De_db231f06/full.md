# Less can be More: What Aspects of Speech Drive End-of-Turn Detection

Rini Sharon<sup>1</sup>, Manickavela A <sup>1</sup>, Kadri Hacioglu<sup>1</sup>, Andreas Stolcke<sup>1</sup>

<sup>1</sup> Uniphore, India

rini.sharon@uniphore.com, manickavela.arumugam@uniphore.com,kadri.hacioglu@uniphore.com, andreas.stolcke@uniphore.com

## Abstract

In conversational AI, detecting when a speaker has finished talking is crucial for natural turn-taking. While recent work incorporates semantics, the relative contribution of different modalities remains unclear. We present a controlled ablation of acoustic, prosodic, and semantic signals for streaming endof-turn detection using a lightweight trimodal classifier. Under identical training conditions, the acoustic–prosodic combination achieves the best balance of accuracy and latency, achieving utterance F1 of 0.93 with 7.8% false alarms at 400ms median latency. Adding text increases premature detections without improving performance. Feature space analysis confirms that prosodic features have the strongest class separability, while text representations overlap substantially. These findings suggest that turn-taking is primarily conveyed through intonation and silence patterns rather than semantic completeness, enabling faster and more reliable systems without expensive text inference.

Index Terms: ASR, End-of-turn detection, prosody, multimodal fusion, conversational AI, speech activity detection.

## 1. Introduction

In modern voice-driven AI systems, determining when a user has finished speaking is an important component that defines how natural an interaction feels. This end-of-turn (EOT) detection task presents a fundamental trade-off: reacting too early risks interrupting the user, while waiting too long creates unnatural delays that break conversational flow. Poor EOT detection degrades user experience, reduces task completion rates, and limits the usability of voice interfaces in real-world deployment.

Historically, this challenge was addressed through silence detection, wherein systems relied on voice activity detection (VAD) or blank ASR tokens to infer a turn end after a fixed-duration of silence [1, 2]. While simple and robust, silence thresholds must be set well above typical conversational pauses [3, 4] to avoid premature interruptions, resulting in noticeable response delays. More critically, these systems treat all pauses equally, failing to distinguish brief hesitations and mid-utterance planning pauses, from genuine turn completions [5, 6, 7].

This limitation motivated broader research into characterizing acoustic and linguistic properties of speech as it approaches a turn boundary. Studies have converged on four categories of signals: acoustic features derived from the speech signal, prosodic cues capturing intonation and temporal patterns, semantic information from transcribed words, and multimodal combinations thereof.

Acoustic features: Acoustic approaches operate directly on signal-level representations. Neural VADs such as Silero [8] and MarbleNet [9] apply convolutional and recurrent architectures to Mel-spectrograms/MFCC to detect speech activity. More recent systems use large pretrained speech encoders as feature extractors that map raw audio directly to learned representations for EOT classification [10, 11, 12]. While these approaches are language and speaker agnostic, they model signal presence rather than turn-taking intent unless explicitly tuned to do so. They capture where speech ends better than whether the speaker intends to yield the floor, since prosodic and semantic cues(significant to turn-taking) are not explicitly modeled.

Prosodic features: The link between prosody and turn-taking is well established in phonetic research [7, 13, 3, 2]. Listeners rely on pitch contours, falling intonation, and final lengthening as turn-yielding cues [14, 15], and can predict turn endings from prosodic signals alone even when lexical content is absent [16]. Combining prosodic features with pause duration has been shown to improve prediction accuracy over singlesignal baselines [6, 17, 18], with consistent gains when prosody is integrated with spectral representations [19]. Despite these benefits, they are hand-crafted and require per-speaker or perdomain normalization.

Semantic features: Speakers often yield the floor at linguistically complete boundaries [20]. Autoregressive language models [21] predict turn-completion likelihood from partial ASR transcripts, while more recent text-only systems fine-tune compact language models on streaming transcripts for deployment in voice agents [22, 23]. Semantic features effectively capture linguistic completeness and are well suited to domain-specific vocabulary. However, they introduce ASR latency and error propagation, and speakers often produce syntactically complete units while intending to continue [13], and listeners frequently anticipate turn endings prior to semantic closure [18, 3].

Multimodal fusion: Given that each modality carries turntaking cues, combining them is a natural direction. Prior work explored early and late integration [1], while more recent systems employ learned fusion mechanisms [24, 25, 26]. While fusion can capture complementary information across modalities, evidence from related tasks shows that added signals can yield diminishing or negative returns when they do not contribute independent information [27, 28]. Whether and when this pattern arises for EOT detection remains only partially understood, underscoring the need for more systematic comparisons.

Beyond input features, EOT systems differ along two axes. In terms of detection granularity, segment-level methods use VAD to isolate a speech segment and classify it as EOT or not [12, 29], while frame-level methods produce continuous per-frame predictions during active speech, enabling earlier response without waiting for a silence boundary [10, 30, 31]. In terms of system integration, some approaches embed EOT within the ASR decoder, jointly optimizing recognition and endpointing [32, 30], while others operate as standalone modules independent of any speech recognizer.

![](images/475249e44cd9dc35488107c5084dc3123e3dbc602fc624071f0257f3fe9aa524.jpg)  
Figure 1: APT architecture

Contributions. This work introduces a controlled multimodal architecture and evaluation protocol for streaming end-of-turn detection, with three main contributions.

1) Architecture for controlled ablation. We design a trimodal streaming EOT detector (Acoustic–Prosodic–Text, APT) whose lightweight fusion head enables controlled ablation by replacing the projected representation of any disabled modality with a fixed zero-valued vector of matching dimensionality and blocking its gradients. This preserves input dimensionality, parameter count, training data, and optimization across configurations, so that performance differences within a shared APT instance reflect the presence or absence of signals rather than changes in model capacity or training dynamics. Similar zero-masking mechanisms appear in work on modality dropout and incomplete-modality transformers [33, 34, 35], but, to our knowledge, they have not been applied to multimodal end-ofturn ablation frameworks.

2) Systematic subset evaluation. Prior multimodal turntaking models compare strong unimodal and multimodal systems such as acoustic only versus acoustic plus text [24, 25] or selected combinations of acoustic, linguistic, and visual streams, but do not exhaustively evaluate all non-empty modality subsets under matched conditions [36, 37, 38]. Building on APT, we instantiate and train all 2<sup>3</sup> − 1 = 7 non-empty subsets of {A, P, T} independently from scratch using identical optimization settings, spanning the full design space from unimodal baselines to all multimodal combinations and revealing when additional modalities genuinely improve performance beyond the best single-stream model.

3) Signal understanding with modern encoders. Earlier studies showed that prosodic features alone can substantially improve EOT detection without speech recognition [6, 17, 19], but they predate modern neural encoders and do not analyze how acoustic, prosodic, and text signals are separated in contemporary representation spaces. We complement our classification ablation with a feature-space analysis that quantifies per-modality class separability independently of the classifier, providing a signal-centric view of what each stream contributes when built on frozen pretrained encoders.

Taken together, these contributions let us revisit a basic question: while modalities can be combined, should they be, and under what conditions do additional modalities meaningfully improve EOT detection rather than simply adding complexity?

## 2. Methodology

Architecture Overview. APT is a trimodal streaming EOT detector that fuses acoustic representations from a frozen ASR encoder, handcrafted prosodic features extracted from the raw waveform, and semantic embeddings from a running ASR transcript. Only a lightweight classification head (∼261K parameters) is optimized, while the two pretrained components, a Zipformer encoder [39] and a MiniLM sentence encoder [40], remain frozen throughout training. This allows the system to leverage rich pretrained representations while keeping training data requirements and compute costs minimal. All three streams operate at a unified 25Hz frame rate (40 ms per frame), each passing through an independent projection stage followed by a depthwise separable temporal convolution [41] over a 7- frame causal window (280 ms of context). The resulting representations are concatenated and processed by a shared fusion module that produces a per-frame EOT probability. APT is a frame-level, standalone detector that predicts turn completion continuously during active speech. The full architecture is illustrated in Figure 1.

## 2.1. Input Streams

Acoustic stream. A pretrained Zipformer2 encoder (70M parameters, frozen), trained on 14k hours of conversational speech data, processes 80D log-mel filterbank features and produces 25Hz frame representations via Conv2dSubsampling followed by strided max-pooling. For streaming inference, the encoder operates causally with a 64-frame chunk size and maintains context through recurrent attention and convolution caching.

Prosodic stream. Five features are extracted per frame at 100 Hz using librosa-based [42] pitch tracking and downsampled to 25 Hz via feature-specific aggregation over 4-frame windows (Table 1). The voiced and speech flags together prevent false triggers during unvoiced consonants (e.g., the /ts/ in “cats”), ensuring the model distinguishes vowel-to-consonant transitions from genuine silence. The silence accumulator d<sub>sil</sub> increments each non-speech frame and resets at speech onset, providing explicit silence duration without requiring the model to learn temporal counting internally. The 5D prosodic vector is projected to 16D before fusion.

Text stream. The frozen Zipformer’s RNN-T decoder performs greedy decoding [43, 44] at each frame. Since 70–80% of frames emit blank tokens, the MiniLM sentence embedding is recomputed only when a new word is decoded and cached otherwise. A sinusoidal positional encoding indexed by absolute frame position is concatenated with the projected embedding (Figure 1). Because the text embedding is static on blank frames while the positional encoding advances, the divergence within the causal window implicitly encodes elapsed time since the last decoded word, functioning as a learned silence cue for the text modality.

Table 1: Prosodicfeatures and aggregation methods.
<table><tr><td>Feature</td><td>Type</td><td>Aggregation</td><td>Purpose</td></tr><tr><td>foorm</td><td>F0, z-normed</td><td>median</td><td>Pitch level</td></tr><tr><td>∆f₀</td><td>F0 derivative</td><td>last — first</td><td>Pitch fall</td></tr><tr><td>voiced (v)</td><td>binary</td><td>max</td><td>Phonation</td></tr><tr><td>speech (s)</td><td>binary (RMS &gt; 0.02)</td><td>max</td><td>Activity</td></tr><tr><td>dsil</td><td>counter, log-scaled</td><td>last value</td><td>Silence dur.</td></tr></table>

Table 2: Modality ablation on 5,000 utterances. $n _ { t p } .$ utterances with a valid detection.
<table><tr><td>Config</td><td>F1</td><td>τ</td><td>FA%</td><td>Miss%</td><td>FA/utt</td><td>ntp</td><td>Med Lat (ms)</td></tr><tr><td>A</td><td>0.927</td><td>0.86</td><td>9.2</td><td>4.4</td><td>0.196</td><td>4321</td><td>440</td></tr><tr><td>P</td><td>0.828</td><td>0.80</td><td>25.6</td><td>3.7</td><td>0.327</td><td>3531</td><td>960</td></tr><tr><td>T</td><td>0.292</td><td>0.40</td><td>74.8</td><td>8.1</td><td>5.416</td><td>853</td><td>640</td></tr><tr><td>A+T</td><td>0.875</td><td>0.65</td><td>14.6</td><td>7.6</td><td>0.275</td><td>3891</td><td>460</td></tr><tr><td>P+T</td><td>0.747</td><td>0.90</td><td>28.5</td><td>11.9</td><td>0.628</td><td>2980</td><td>640</td></tr><tr><td>A+P</td><td>0.930</td><td>0.87</td><td>7.8</td><td>5.2</td><td>0.144</td><td>4349</td><td>400</td></tr><tr><td>APT</td><td>0.909</td><td>0.76</td><td>10.7</td><td>6.0</td><td>0.206</td><td>4166</td><td>360</td></tr></table>

## 2.2. Fusion and Classification

The three stream outputs are concatenated and fused into a 272- dimensional vector per frame. A linear layer projects this joint representation to 128D, marking the first point at which the three streams interact. A final linear projection maps the representation to 1D, and applying sigmoid to the output logit produces the per-frame EOT probability.

To isolate modality contribution, disabled streams are replaced with fixed zero-valued tensors registered as nontrainable buffers using PyTorch register buffer. This ensures that the classifier always receives an input of identical dimensionality with consistent index ranges assigned to each modality, while preventing gradients from flowing through inactive streams and keeping the total parameter count unchanged. Each configuration is trained independently from scratch under identical hyperparameters, ensuring that performance differences arise from the available signal rather than architectural variation or model capacity.

## 3. Experimental Setup

We use a proprietary multi-domain English telephony corpus (banking, insurance, retail, telecommunications) of real humanto-human calls recorded at 8 kHz stereo. We extract single-turn segments where one speaker holds the floor continuously with opposing-channel activity below 10%. EOT timestamps are refined by reconciling WhisperX forced-alignment [45] word boundaries with VAD-detected speech endpoints, selecting the later when residual vocalization is detected between the two candidates. Each segment is extended up to 1.3s post-EOT with VAD-validated silence. Frame-level binary labels at 25 Hz mark 0 before EOT and 1 from EOT onward. We sample 10K for training ( 33 hours), 2K for validation and 5k for test, split at the recording level to prevent speaker leakage.

All models are trained with BCEWithLogitsLoss using positive class weight $w ^ { + } = 5 . 6$ (accounting for class imbalance) and AdamW [46] (learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 } ,$ , gradient clipping 5.0, dropout 0.2). Training runs up to 30 epochs with early stopping on validation F1 (patience 5). Linear layers are initialized from $\mathcal { N } ( 0 , 0 . 0 2 )$ with zero biases, while convolutional layers use Kaiming normal initialization [47]. Since all configurations share the same 5,000 test utterances, pairwise comparisons use McNemar’s test [48] on binary outcomes and the Wilcoxon signed-rank test [49] on latency and FA counts $( \alpha = 0 . 0 5 )$

## 3.1. Evaluation Metrics

To evaluate the system’s streaming performance, we apply a duration-aware thresholding rule where a positive EOT is declared only when the sigmoid output exceeds threshold τ for $d \ = \ 8$ consecutive frames (320 ms). The detection time is the last frame of this sustained crossing. The tolerance window is asymmetric: detections up to 50 ms before the groundtruth boundary are true positives with zero latency; detections after the boundary are true positives with positive latency; detections more than 50 ms before the boundary are false alarms. This reflects the production asymmetry where late responses are preferable to premature interruptions.

We report: F1 (utterance-level detection decision), false alarms(FA%) (fraction of utterances where the first detection is premature), Miss% (fraction with no detection), FA/utt (mean count of premature triggers per utterance), and median detection latency over correct detections. FA% and FA/utt measure different failure modes: FA% captures whether the first trigger was wrong; FA/utt captures how often the model misfires throughout an utterance. We additionally report the FA% versus latency trade-off curve [6] across the full τ sweep, enabling comparison beyond a single operating point.

## 4. Results and Discussion

All results use utterance-level evaluation as defined in Section 3.1. Table 2 reports all seven modality combinations at their respective best thresholds, and Figure 2 shows the FA%– latency trade-off across the full τ sweep from 0.05 to 0.95, with axis boundaries constrained for visual clarity.

## 4.1. Modality Ablation

Acoustic stream. Among single-modality configurations, the acoustic-only model achieves the highest F1 and the lowest miss rate. It also operates at the highest threshold $\tau = 0 . 8 6$ in the ablation, indicating temporally broad, well-calibrated activations near turn boundaries. This is consistent with prior evidence that neural speech encoders capture the spectro-temporal structure of turn endings across multiple frames [26], producing sustained high-confidence regions that survive the consecutiveframe filter.

Prosodic stream. Adding prosodic features increases F1 in every combination: A+P over A (+0.003), APT over A+T (+0.034), and P+T over T (+0.455). The gains are driven primarily by silence dur and $\Delta f _ { 0 }$ , which encode accumulated silence duration and pitch fall, two established turn-boundary cues [15, 14]. These features reduce FA/utt without increasing detection latency (consistent with literature [19, 50]), and A+P achieves the highest F1 and lowest false alarm rate overall(McNemar $p < 0 . 0 0 1 )$ ). However, the prosodic-only model exhibits the second-highest FA% in the ablation, and as Figure 2 shows, its curve remains in a high-FA region across the full τ sweep. Prosodic cues carry discriminative signal but require acoustic grounding to support reliable detection decisions.

![](images/820248dc050f0c9796979ccd1ef6d1c20565e1bdc8f510db29ffa1f0095d0c37.jpg)  
Figure 2: FA% vs. mean detection latency across the full τ sweep $( d = 8 ,$ NVIDIA A10G). Circles: intersection with production budget lines (Grey). T (dashed): latency over 853/5,000 utterances only.

![](images/2e14098d63daf2b7c81032665a73852848176ab8c589bd014d356f28aafa315f.jpg)  
Figure 3: t-SNEfeature discriminability. Acoustic post-projection (128D); prosodic and text raw pre-projection (35D, 800D). Text uses KDE because filament geometry obscures class density in scatter. Bar inset: raw vs. projected silhouette per modality. Violin insets: EOT vs. non-EOTfor f0 delta, speech, silence dur.

Text stream. The text-only model yields the lowest F1 and the highest FA/utt in the ablation, averaging over five premature triggers per utterance. Its reported median latency is computed only over the 853 utterances (of 5000) where a valid detection was made, so the estimate is optimistic and unreliable. This failure follows from the structure of conversational speech, where a single turn typically contains multiple syntactically complete units before the speaker yields [20, 3], each producing a textdriven trigger. The effect is amplified in our corpus, where median turn duration is 10–11s, providing more mid-turn semantic completions than shorter command-style utterances. This problem is not confined to the unimodal case. Every text-containing configuration produces higher FA/utt than its text-free counterpart, with APT exceeding A+P, A+T exceeding A, and P+T exceeding P. The FA% increase from A+P to APT is significant (McNemar $p < 1 0 ^ { - 9 } )$ , as is the increase in FA/utt (Wilcoxon $p ~ < ~ 1 0 ^ { - 9 } )$ Although text-containing configurations(APT) achieve lower median latency, their F1 is also lower because the latency reduction comes at the cost of more frequent FAs.

Operating-point analysis. In Figure 2, the A+P and APT operating points (circles) fall within the real-time deployment constraints (FA < 10%, latency < 500 ms). A and A+T satisfy both constraints marginally. P and P+T do not cross FA = 10% at any τ, and T satisfies neither constraint at any operating point.

## 4.2. Feature Space Analysis

Figure 3 shows t-SNE [51] embeddings of each modality, with silhouette scores (bar inset) [52] quantifying EOT versus non-EOT class separation. Prosodic raw features are the most discriminative input (silhouette 0.301) despite having only 35 dimensions, driven majorly by silence dur(shows nearcomplete class separation in the violin inset). The acoustic projection provides the largest silhouette gain (+0.158) over its raw input, extracting EOT-discriminative structure that is not present in the frozen ASR encoder’s output. This explains why the acoustic stream performs well despite using a frozen, taskmismatched encoder. Text representations form filament structures in the t-SNE projection because the BERT embedding updates only at non-blank decoder emissions, leaving the representation unchanged across the majority of frames (only the positional encoding changes). The resulting silhouette of 0.108 reflects substantial overlap between EOT and non-EOT frames, consistent with the high false alarm rates observed when text is included in the fusion.

## 4.3. Limitations and Future Work

Results are limited to in-domain two-speaker English telephony and text may contribute more in domains with weaker acoustic EOT cues (low-resource languages or noisy conditions). Prior work supports this, text improves turn prediction on clean, presegmented utterances [21, 38], and combining prosody with a boundary-conditioned language model outperformed prosody alone [6]. The critical difference is that these approaches apply semantic signal at word/pause boundaries, whereas our text stream caches the embedding between decoder emissions and fuses it at every frame, amplifying false alarms at each mid-turn syntactic completion. The text stream also operates on imperfect transcripts (13.8% WER), and replacing the encoder with RoBERTa did not help, confirming a structural rather than capacity limitation. APT’s 40 ms latency gain over A+P is partially offset by 30-60 ms of RNN-T/BERT inference overhead.

Future work could explore using a language model to suppress acoustic detections at syntactically incomplete boundaries, which may recover useful semantic signal without the false alarm cost of continuous fusion. Cross-lingual and crossdomain evaluation would establish whether A+P’s advantage generalises beyond English telephony.

## 5. Conclusion

We evaluated all seven non-empty subsets of acoustic, prosodic, and text modalities for streaming EOT detection under identical training and evaluation conditions. Acoustic features provide the primary detection signal and prosodic features add complementary precision. Text features increase false alarms in every configuration they appear in, consistent with the feature space analysis showing higher class overlap for text representations. Only acoustic–prosodic and trimodal systems meet real-time deployment latency and false alarm constraints at their optimal operating points. Based on these findings, acoustic–prosodic modeling appears sufficient for deployment-grade EOT detection in the conversational domain studied here. The additional inference cost of text integration is not recovered in accuracy when acoustic features are present, suggesting its inclusion should be carefully weighed in deployment scenarios.

## 6. Generative AI Use Disclosure

Generative AI tools were used for editing and polishing this manuscript.

## 7. References

[1] A. Raux and M. Eskenazi, “Flexible turn-taking for spoken dialog systems,” in Proceedings ofInterspeech, 2008.

[2] D. Schlangen and G. Skantze, “Towards a general, empirically based theory of turn-taking,” Proceedings of SIGDIAL, 2011.

[3] S. C. Levinson and F. Torreira, “Timing in turn-taking in conversation,” Frontiers in Psychology, vol. 6, p. 731, 2015.

[4] A. Raux and M. Eskenazi, “Optimizing the turn-taking behaviour of task-oriented spoken dialog systems,” in ACM Transactions on Speech and Language Processing, vol. 9, no. 1, 2012, pp. 1–23.

[5] G. Castillo-Lopez, “A survey of recent advances on turn-taking´ modeling in spoken dialogue systems,” in IWSDS 2025, 2025.

[6] L. Ferrer, E. Shriberg, and A. Stolcke, “Is the speaker done yet? Faster and more accurate end-of-utterance detection using prosody,” in Proc. 7th International Conference on Spoken Language Processing (ICSLP), 2002, pp. 2061–2064.

[7] H. Sacks, E. A. Schegloff, and G. Jefferson, “A simplest systematics for the organization of turn-taking for conversation,” Language, vol. 50, no. 4, pp. 696–735, 1974.

[8] Silero Team, “Silero VAD: Pre-trained enterprise-grade voice activity detector,” arXiv preprint arXiv:2109.14282, 2021.

[9] Y. Jia, Y. Zhang, R. J. Weiss, Q. Wang, and J. Shen, “MarbleNet: An efficient end-to-end neural model for speech processing,” in Proc. Interspeech, 2021.

[10] E. Ekstedt and G. Skantze, “Voice activity projection: Selfsupervised learning of turn-taking events,” in Proc. Interspeech, 2022.

[11] G. Li, C. Wang, H. Xue, S. Wang, D. Gao, Z. Zhang, Y. Lin, W. Li, L. Xiao, Z. Fu, and L. Xie, “Easy turn: Integrating acoustic and linguistic modalities for robust turn-taking in full-duplex spoken dialogue systems,” arXiv preprint arXiv:2509.23938, 2025. [Online]. Available: https://arxiv.org/abs/2509.23938

[12] H. Ok, S. Yoo, and J. Lee, “Speculative endturn detector for efficient speech chatbot assistants,” arXiv preprint arXiv:2503.23439, 2025. [Online]. Available: https://arxiv.org/abs/2503.23439

[13] T. Stivers, N. J. Enfield, P. Brown, C. Englert, M. Hayashi, T. Heinemann, G. Hoymann, F. Rossano, J. P. de Ruiter, K.- E. Yoon, and S. C. Levinson, “Universals and cultural variation in turn-taking in conversation,” Proceedings of the National Academy ofSciences, vol. 106, no. 26, pp. 10 587–10 592, 2009.

[14] A. Cutler and M. Pearson, “Intonation and syntax in the perception of sentence boundaries,” Journal of the Acoustical Society of America, vol. 80, no. 1, pp. 1–15, 1986.

[15] N. Ward and W. Tsukahara, “Prosodic features which cue turntaking in spoken dialogue,” in Proceedings ofSIGDIAL, 2000.

[16] S. O’Connor Russell, D. Charuau, and N. Harte, “The role of prosodic and lexical cues in turn-taking with self-supervised speech representations,” arXiv preprint arXiv:2601.13835, 2026.

[17] L. Ferrer, E. Shriberg, and A. Stolcke, “A prosody-based approach to end-of-utterance detection that does not require speech recognition,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), vol. 1, 2003, pp. I–605– I–608.

[18] A. Gravano and J. Hirschberg, “Turn-taking cues in task-oriented dialogue,” Computer Speech & Language, vol. 25, no. 3, pp. 601– 634, 2011.

[19] H. Arsikere, E. Shriberg, and U. Ozertem, “Enhanced end-of-turn detection for speech to a personal assistant,” in AAAI Spring Symposium on Turn-Taking and Coordination in Human-Machine Interaction. AAAI Press, 2015, pp. 75–78.

[20] C. E. Ford and S. A. Thompson, “Interactional units in conversation: Syntactic, intonational, and pragmatic resources for the management of turns,” in Interaction and Grammar, E. Ochs, E. A. Schegloff, and S. A. Thompson, Eds. Cambridge University Press, 1996, pp. 134–184.

[21] Y. Huang, M. Wang, and G. Skantze, “TurnGPT: A transformerbased language model for predicting turn-taking in spoken dialogue,” arXiv preprint arXiv:2009.00345, 2020.

[22] VideoSDK Team, “Namo turn detector v1: Semantic turn detection for conversational AI,” 2025. [Online]. Available: https://huggingface.co/collections/videosdk-live/ namo-turn-detector-v1-68d52c0564d2164e9d17ca97

[23] LiveKit, “Improving voice AI’s turn detection with transformers,” 2024, smolLM v2 fine-tuned for end-ofutterance prediction. [Online]. Available: https://blog.livekit. io/using-a-transformer-to-improve-end-of-turn-detection/

[24] D. Lala, K. Inoue, and T. Kawahara, “A neural network for predicting turn-taking in spoken dialogue,” in Proceedings of SIG-DIAL, 2019.

[25] W. Maier, J. Hough, and D. Schlangen, “Incremental turn-taking prediction using acoustic and linguistic features,” in Proceedings ofSIGDIAL, 2020.

[26] G. Skantze, “Turn-taking in conversational systems and human– robot interaction: A review,” Computer Speech & Language, vol. 67, p. 101178, 2021.

[27] S. Poria, D. Hazarika, N. Majumder, R. Mihalcea, and E. Cambria, “Multimodal emotion recognition: An overview of recent advances,” IEEE Transactions on Affective Computing, 2020.

[28] S. Huang, A. Pareek, S. Seyyedi, I. Banerjee, and M. P. Lungren, “Fusion of medical imaging and clinical text: Opportunities and challenges,” Journal of Biomedical Informatics, vol. 114, p. 103651, 2021.

[29] Pipecat AI, “Smart turn: Open-source semantic voice activity detection for turn detection,” 2025, v3, Whisper Tiny base with linear classifier. [Online]. Available: https://github.com/ pipecat-ai/smart-turn

[30] Y. Shangguan, R. Prabhavalkar, H. Su, J. Mahadeokar, Y. Shi, J. Zhou, C. Wu, D. Le, O. Kalinli, C. Fuegen, and M. L. Seltzer, “Dissecting user-perceived latency of on-device E2E speech recognition,” in Proc. Interspeech, 2021, pp. 4553–4557.

[31] G. Skantze, “Towards a general, continuous model of turn-taking in spoken dialogue using LSTM recurrent neural networks,” in Proc. SIGdial Workshop on Discourse and Dialogue, 2017, pp. 220–230.

[32] S.-Y. Chang, B. Li, T. N. Sainath, C. Zhang, T. Strohman, Q. Liang, and Y. He, “Turn-taking prediction for natural conversational speech,” in Proc. Interspeech, 2022, pp. 1821–1825.

[33] G. Krishna, S. Dharur, O. Rudovic, P. Dighe, S. Adya, A. H. Abdelaziz, and A. H. Tewfik, “Modality dropout for multimodal device directed speech detection using verbal and non-verbal features,” arXiv preprint arXiv:2310.15261, 2023.

[34] A. H. Abdelaziz, B.-J. Theobald, P. Dixon, R. Knothe, N. Apostoloff, and S. Kajareker, “Modality dropout for improved performance-driven talking faces,” arXiv preprint arXiv:2005.13616, 2020.

[35] Y. Zhang, N. He, J. Yang, Y. Li, D. Wei, Y. Huang, Y. Zhang, Z. He, and Y. Zheng, “mmformer: Multimodal medical transformer for incomplete multimodal learning,” in Medical Image Computing and Computer Assisted Intervention (MICCAI), ser. Lecture Notes in Computer Science, vol. 13431. Springer, 2022, pp. 109–119.

[36] M. Roddy, G. Skantze, and N. Harte, “Multimodal continuous turn-taking prediction using multiscale RNNs,” in Proc. ACM International Conference on Multimodal Interaction (ICMI), 2018, pp. 186–190.

[37] F. Kurata, M. Saeki, S. Fujie, and Y. Matsuyama, “Multimodal turn-taking model using visual cues for end-of-utterance prediction in spoken dialogue systems,” in Interspeech. ISCA, 2023, pp. 2658–2662.

[38] S.-Y. Chang, B. Li, T. N. Sainath, C. Zhang, T. Strohman, Q. Liang, and Y. He, “Turn-taking prediction for natural conversational speech,” in Proc. Interspeech. Incheon, Korea: ISCA, 2022, pp. 1821–1825.

[39] Z. Yao, W. Li, G. Chen, and B. Xu, “Zipformer 2: A faster and better encoder for automatic speech recognition,” in Proc. Interspeech, 2024.

[40] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 5776–5788.

[41] F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1251–1258.

[42] B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, “librosa: Audio and music signal analysis in Python,” in Proc. 14th Python in Science Conference (SciPy), 2015, pp. 18–25.

[43] Z. Yao, L. Guo, X. Yang, W. Kang, F. Kuang, D. Povey et al., “Icefall: A PyTorch-based ASR toolkit,” https://github.com/k2-fsa/ icefall, 2021.

[44] A. Graves, “Sequence transduction with recurrent neural networks,” arXiv preprint arXiv:1211.3711, 2012.

[45] M. Bain, J. Huh, T. Han, and A. Zisserman, “WhisperX: Timeaccurate speech transcription of long-form audio,” in Proc. Interspeech, 2023.

[46] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), 2019.

[47] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification,” in Proc. IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026–1034.

[48] Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, 1947.

[49] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945.

[50] H. Arsikere, E. Shriberg, and U. Ozertem, “Computationallyefficient endpointing features for natural spoken interaction with personal-assistant systems,” in Proc. IEEE ICASSP, 2014, pp. 3241–3245.

[51] L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” Journal ofMachine Learning Research, vol. 9, no. 86, pp. 2579– 2605, 2008.

[52] P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,” Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.