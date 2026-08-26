# Preference Optimization for Non-Verbal Vocalization Synthesis

Haoyang Li, Chenglin Xu, Junchuan Zhao, Yuang Cao, Liumeng Xue, Yiwen Guo, Eng Siong Chng

Abstract—Non-verbal vocalizations (NVs), such as laughter, coughs, and sighs, are essential for expressive TTS, but the effectiveness of preference optimization for NV generation remains poorly understood. We systematically study preference optimization for NV-capable TTS, focusing on preference signals, preference-pair construction, and DPO-based optimization objectives. We formulate an NV-aware character error rate (NV-CER) by treating NV tags as distinct output symbols and computing a weighted pinyin-based CER over both verbal and non-verbal content, enabling controllable optimization of NV realization without modifying the underlying optimization algorithm. Experiments on Emilia-NV and the augmented NV-Bench covering 18 NV types reveal how different design choices affect NV realization and lexical fidelity, and establish an effective setup using standard DPO. Objective, LLM-based, and human evaluations provide converging evidence for our findings, offering practical insights into NV-aware post-training for expressive TTS.

Index Terms—speech synthesis, text-to-speech, nonverbal, paralinguistic, Direct Preference Optimization

## I. INTRODUCTION

Recent advances in neural text-to-speech (TTS) have enabled increasingly natural and expressive speech synthesis [1], [2], [3]. Beyond verbal content, natural communication also relies on non-verbal vocalizations (NVs), including laughter, crying, and hesitation sounds, which convey emotion, discourse structure, and speaker intent [4]. However, faithful NV generation remains challenging and relatively underexplored in TTS. Recent work has advanced NV-aware TTS through dataset construction [5], [6], [7], [8], [9] and benchmark development for standardized evaluation [10], [11]. These resources enable systematic investigation of NV realization in TTS, an area that has received limited attention to date.

Preference optimization [12], [13] has emerged as an effective post-training paradigm for TTS, improving aspects of naturalness and intelligibility [14], [15], [16]. Recent TTS systems, including Fish Audio S2 [17], CosyVoice2 [18], and CosyVoice3 [19], have incorporated NV-aware models into preference optimization pipelines. However, these studies primarily focus on overall synthesis quality, providing limited insight into the specific impact of post-training on NV generation. Fish Audio S2 reports final system performance without isolating the contribution of preference optimization, while CosyVoice2 and CosyVoice3 lack NV evaluation. [9] applied DPO to NV-TTS using preference pairs constructed from manually verified speech with NVs and corresponding speech without NVs, encouraging NV occurrence rather than faithful NV realization, and reported degraded CER performance. Consequently, both the effectiveness of preference optimization for NV generation and the design choices that govern its performance remain unclear, motivating a systematic study of preference signals, preference-pair construction, and optimization objectives for NV-aware TTS.

In this work, we systematically investigate NV-aware preference optimization for LLM-based TTS. We formulate an NVaware character error rate (NV-CER) by treating non-verbal vocalization tags as distinct output symbols and computing a weighted pinyin-based CER over both verbal and non-verbal content. Building on this metric, we construct preference signals using an NV-capable automatic speech recognition (ASR) model and introduce a simple weighting mechanism to adjust the relative importance of NV tags, enabling explicit control over the trade-off between NV realization and lexical fidelity without modifying the underlying optimization algorithm. We then systematically study preference signals, preference-pair construction strategies, and loss formulations within DPO. Our experiments establish how these design choices affect NV realization, leading to an effective setup using standard DPO. Objective, LLM-based, and human evaluations provide converging evidence, establishing practical principles for effective preference optimization and post-training of NV-aware TTS.

## II. METHODOLOGY

## A. NV-Aware Preference Signal

1) NV-Aware Speech Recognition: Conventional automatic speech recognition (ASR) systems transcribe only spoken text and disregard non-verbal vocalizations (NVs), making them unsuitable for evaluating expressive speech synthesis. A straightforward alternative is to compare the recognized text against the reference transcription. However, this introduces a fundamental limitation for NV evaluation. Many non-verbal vocalizations, such as laughter, crying, and coughing, do not have deterministic textual representations, making it impossible to establish a one-to-one mapping between speech and text. Although some vocalizations (e.g., hesitation or surprise sounds) may be approximated by lexical words or pinyin, these mappings are inherently ambiguous and context-dependent.

To overcome this limitation, we leverage an NV-aware ASR (NV-ASR) model that explicitly recognizes predefined NV events alongside spoken text. Given a synthesized speech sample x, the NV-ASR model predicts

$$
\hat { \mathbf { y } } = \{ \hat { y } _ { 1 } , \hat { y } _ { 2 } , \dots , \hat { y } _ { N } \} ,\tag{1}
$$

where each output token corresponds to either a lexical token or an NV tag. Unlike conventional ASR, the resulting transcription preserves explicit non-verbal vocalization information, providing the basis for constructing the proposed NVaware preference signal.

2) NV-Aware Character Error Rate: Given the reference and predicted transcriptions, we define an NV-aware Character Error Rate (NV-CER) to measure both lexical and NV correctness. For Mandarin, lexical tokens are represented using pinyin to reduce sensitivity to homophones and character variations, while NV tags are retained as independent symbols. Edit distance is computed over the combined token sequence,

$$
\mathrm { N V - C E R } = \frac { D _ { c } ( \mathbf { r } , \hat { \mathbf { y } } ) } { \sum _ { t \in \mathbf { r } } c ( t ) } ,\tag{2}
$$

where r and $\hat { \mathbf { y } }$ denote the reference and predicted token sequences, respectively, and $D _ { c } ( \cdot , \cdot )$ denotes the weighted edit distance with token-dependent insertion and deletion costs given by $c ( t )$ and substitution cost given by the maximum cost of the two tokens. Although instantiated using pinyin for Mandarin, the proposed framework is readily extendable to other languages by replacing the lexical representation with an appropriate language-specific alternative while preserving NV tags as independent symbols.

3) Controllable NV Preference Strength: Different applications may require different trade-offs between NV realization and lexical accuracy. To enable this flexibility, we assign a higher edit cost to NV tags, with w<sub>NV</sub> denoting the NV weight:

$$
c ( t ) = { \left\{ \begin{array} { l l } { w _ { \mathrm { N V } } , } & { { \mathrm { i f ~ } } t { \mathrm { ~ i s ~ a n ~ N V ~ t a g , } } } \\ { 1 , } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }\tag{3}
$$

Increasing $w _ { \mathrm { N V } }$ increases the contribution of NV errors to NV-CER, giving them greater influence when constructing preference signals. Conversely, w<sub>NV</sub> = 1 assigns equal cost to lexical and NV tokens. This weighting mechanism provides explicit control over the relative importance of NV and lexical accuracy without modifying the underlying preference optimization algorithm.

## B. NV-aware Preference Alignment

1) NV-Aware Preference Pair Construction: We adopt Direct Preference Optimization (DPO) as the preference optimization framework due to its simplicity, effectiveness, and widespread adoption in TTS post-training. Given an input text prompt $p ,$ the pretrained TTS model independently generates K candidate speech utterances,

$$
{ \mathcal { X } } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { K } \} .\tag{4}
$$

Each candidate is transcribed using the NV-ASR model described in Section II-A, and its NV-CER score is computed against the reference transcription. The candidate with the lowest NV-CER is selected as the preferred response $x ^ { + }$ while the candidate with the highest NV-CER is selected as the rejected response $x ^ { - }$ , forming the preference pair used for optimization. In addition, we investigate several alternative preference construction strategies, including utilizing groundtruth speech during preference selection and manipulating the synthetic speech. These variants are described in Section III-D and evaluated experimentally.

2) Preference Optimization Objective: Given the constructed preference pairs $( x ^ { + } , x ^ { - } )$ , the TTS model is optimized to directly increase the relative likelihood of the preferred response over the rejected response via DPO:

$$
\mathcal { L } _ { \mathrm { D P O } } = - \log \sigma \left( \beta \left[ \log \frac { \pi _ { \theta } ( x ^ { + } | p ) } { \pi _ { \mathrm { r e f } } ( x ^ { + } | p ) } - \log \frac { \pi _ { \theta } ( x ^ { - } | p ) } { \pi _ { \mathrm { r e f } } ( x ^ { - } | p ) } \right] \right) _ { \mathcal { N } }\tag{5}
$$

where $\pi _ { \theta }$ and $\pi _ { \mathrm { r e f } }$ denote the trainable and reference TTS models, respectively, and $\beta$ controls the preference strength.

Besides the standard DPO objective, we investigate several design choices for NV-aware preference optimization, including combining DPO with the conventional supervised finetuning (SFT) objective and alternative preference ranking in Section III-D, with their effectiveness evaluated in Section IV.

## III. EXPERIMENTS

## A. Dataset

We train on Emilia-NV [8], comprising 573.4 hours of expressive Mandarin speech with transcriptions annotated by 18 predefined NV tags, covering laughter, breathing, hesitation, and other vocal events.

We evaluate on the Mandarin subset of NV-Bench [10], which provides text prompts, speaker-conditioning recordings, and ground-truth speech. It contains a single-label subset with 50 utterances per NV category and a 391-utterance multi-label subset, each containing two or more NV events. As NV-Bench covers only 15 of the 18 training tags, we further construct 50 prompts for each of the three missing categories using gemini-3.1-pro-preview, following the NV-Bench annotation style and using its speaker-conditioning utterances. This adds 150 samples to the single-label subset.

## B. Evaluation Metrics

Evaluation is based on recent NV benchmarks [10], [11].

1) Objective metrics: Speech intelligibility. NV-CER (Eq. 2) evaluates overall transcription accuracy by jointly measuring lexical content and NVs, with the NV weight w<sub>NV</sub> set to 1. To separately assess verbal and non-verbal generation, we additionally report CER, computed after removing all NV tags, and PCER, computed after excluding all lexical tokens.

Perceptual quality. DNSMOS P.835 [20] predicts speech (SIG), background (BAK), and overall (OVRL) quality, while UTMOS predicts overall MOS for synthesized speech.

Speaker similarity. Speaker similarity is evaluated using speaker embedding cosine similarity (SECS), computed from speaker embeddings extracted with Resemblyzer<sup>1</sup>.

TABLE I: Ablation on loss and preference pair construction. GT: ground-truth speech; $S y n ^ { + } / S y n ^ { - } ;$ best-/worstscoring synthetic candidates; Hybrid: GT or $S y n ^ { + }$ selected by NV-CER; $S y n ^ { \mathrm { N o N V } }$ : synthetic speech without NVs. Bold/underline: best/second-best.
<table><tr><td>Loss</td><td>Pair (Pref, Rej)</td><td>NV-CER</td><td>PCER</td><td>CER</td></tr><tr><td>DPO</td><td>GT,  $S y n ^ { - }$ </td><td>93.42</td><td>104.89</td><td>93.20</td></tr><tr><td>DPO+SFT</td><td> ${ \mathrm { G T } } , S y n { \mathrm { \ } }$ </td><td>9.43</td><td>39.33</td><td>8.66</td></tr><tr><td>DPO</td><td>Hybrid,  $S y n ^ { \cdot }$ </td><td>65.18</td><td>75.22</td><td>64.99</td></tr><tr><td>DPO+SFT</td><td>Hybrid, Syn−</td><td>7.64</td><td>33.22</td><td>7.00</td></tr><tr><td>DPO DPO+SFT</td><td> $S y n ^ { + } , S y n ^ { N o N V }$ </td><td>22.95</td><td>52.89</td><td>22.23</td></tr><tr><td></td><td> $\bar { S y n ^ { + } , S y n ^ { N o N V } }$ </td><td>3.68</td><td>25.56</td><td>3.12</td></tr><tr><td>DPO</td><td> $S y n ^ { + } , S y n ^ { - }$ </td><td>2.59</td><td>21.33</td><td>2.09</td></tr><tr><td>DPO+SFT</td><td> $\bar { S y n ^ { + } } , \bar { S y n ^ { - } }$ </td><td>3.03</td><td>23.00</td><td>2.52</td></tr></table>

TABLE II: Effect of training epochs (pref pair: $S y n ^ { + } , S y n ^ { - } )$
<table><tr><td>Loss</td><td>Epoch</td><td>NV-CER</td><td>PCER</td><td>CER</td></tr><tr><td rowspan="2">DPO</td><td>1</td><td>2.59</td><td>21.33</td><td>2.09</td></tr><tr><td>2</td><td>31.32</td><td>56.89</td><td>30.83</td></tr><tr><td rowspan="3">DPO+SFT</td><td>1</td><td>3.03</td><td>23.00</td><td>2.52</td></tr><tr><td>2</td><td>3.25</td><td>21.78</td><td>2.78</td></tr><tr><td>3</td><td>3.29</td><td>21.89</td><td>2.83</td></tr></table>

2) LLM-based multi-rater: We employ Gemini-2.5-Pro as an LLM-based evaluator to assess NV accuracy (A), NV perceptual effect (P), overall naturalness (N), overall quality (Q), and overall expression (E), following the Track 2 evaluation rubrics of the NVVSpeech Challenge<sup>2</sup>. All systems for the same utterance are jointly presented with anonymized labels and randomized presentation orders to mitigate systemidentity and positional biases in comparative judging. Three independent simulated raters evaluate each sample, with scores assigned according to predefined 1–5 rubrics (0–5 for NVspecific criteria when no NV is audible).

3) Subjective Evaluation: We further conduct an A/B/Tie preference test between the SFT baseline and preference model on the multi-label evaluation set of NV-Bench, providing a human reference for comparison with the objective and LLMbased evaluations. Eleven native-speaking volunteers evaluate four criteria: NV accuracy (NV Acc.); NV naturalness (NV Nat.), with an N/A option when NVs are not comparable (e.g., when an NV is missing); lexical accuracy (Lex. Acc.), assessing text correctness excluding NVs; and overall naturalness (Overall Nat.). The workload is distributed such that each sample is independently evaluated by five volunteers. All participants provided informed consent prior to the evaluation.

## C. Implementation Details

We first perform supervised fine-tuning (SFT) on the pretrained CosyVoice2-0.5B model using the Emilia-NV dataset [8] for 4 epochs with a learning rate of $1 \times 1 0 ^ { - 5 }$ , resulting in a Base model that serves as the common initialization for all subsequent post-training experiments. For preference optimization, we construct an approximately 100-hour subset of Emilia-NV by prioritizing utterances containing underrepresented NV classes to obtain a more balanced class distribution.

TABLE III: Effect of $w _ { \mathrm { N V } }$ on NV-CER preference signals.
<table><tr><td> ${ \boldsymbol { w } } _ { \mathrm { N V } }$ </td><td>NV-CER</td><td>PCER</td><td>CER</td><td>DNSMOS</td><td>Sim</td></tr><tr><td>1</td><td>2.59</td><td>21.33</td><td>2.09</td><td>3.346/3.607/4.091</td><td>0.891</td></tr><tr><td>5</td><td>3.02</td><td>18.22</td><td>2.60</td><td>3.337/3.601/4.086</td><td>0.890</td></tr><tr><td>10</td><td>2.91</td><td>18.11</td><td>2.52</td><td>3.339/3.601/4.089</td><td>0.890</td></tr></table>

Using the Base model, we independently generate 8 candidate utterances for each training sample in the post-training subset, from which preference pairs are constructed. Candidate generation follows the default Repetition Aware Sampling (RAS) strategy in CosyVoice2, with $t o p _ { p } = 0 . 9 , t o p _ { k } = 3 0$ win $\begin{array} { r } { s i z e = 1 0 , } \end{array}$ , and $\tau _ { r } = 0 . 1$ to encourage sampling diversity. During evaluation, $t o p _ { p }$ and $t o p _ { k }$ are reduced to 0.8 and 25, respectively, following the default CosyVoice2 configuration.

Unless otherwise specified, all post-training experiments use a learning rate of $1 \times 1 0 ^ { - 6 }$ , dynamic batching with a maximum of 1000 frames per batch, gradient accumulation of 40.

We adopt the NV-ASR model from [10] to compute NV-CER. It fine-tunes SenseVoice-Small [21] on multiple public NV speech corpora [8], [5], [22], [6], [7], [23], and recognizes all 18 NV tags in our training set.

## D. Design Choices

Loss formulation: We investigate whether combining DPO with the supervised fine-tuning (SFT) objective improves performance and training stability. The SFT loss is weighted by 0.2 to keep its magnitude comparable to the DPO loss.

Preference pair construction: We investigate how preference-pair construction affects optimization. For preferred responses, we compare three strategies: (1) the default, selecting the top-ranked synthetic candidate (Section II-B1); (2) always selecting ground-truth speech; and (3) selecting ground-truth speech when its NV-CER is lower than the top-ranked synthetic candidate, otherwise using the synthetic candidate. For rejected responses, we additionally consider synthesized speech without NVs, following [9].

Preference signal: Unless otherwise stated, preference scores are computed using NV-CER (Eq. 2) with $w _ { \mathrm { N V } } = 1$ We investigate three preference-signal alternatives: (1) conventional pinyin CER from the general-purpose SenseVoice-Small ASR model [21]; (2) a composite score combining NV recognition and perceptual quality,

$$
s = \mathrm { C E R } _ { \mathrm { N V } } - \lambda \left( \frac { \mathrm { U T M O S } - 1 } { 4 } \right) ,\tag{6}
$$

where $\lambda = 0 . 4 ;$ and (3) varying $w _ { \mathrm { N V } }$ in Eq 3 to explicitly control the relative importance of NV errors.

## IV. RESULTS AND DISCUSSION

## A. Loss and Preference Pair Ablation

We investigate loss formulation and preference-pair construction on the single tag testset of NV-Bench. As shown in

TABLE IV: Comparison of loss and preference-signal variants on the single-tag test set. Preference pairs use $S y n ^ { + } / S y n ^ { - }$
<table><tr><td>Loss</td><td>Preference Signal</td><td>NV-CER</td><td>PCER</td><td>CER</td><td>UTMOS</td><td>DNSMOS</td><td>Sim</td></tr><tr><td>Base</td><td>一</td><td>3.47</td><td>32.78</td><td>2.74</td><td>2.01</td><td>3.356/3.616/4.099</td><td>0.890</td></tr><tr><td>SFT</td><td></td><td>3.62</td><td>30.78</td><td>2.93</td><td>2.00</td><td>3.353/3.613/4.098</td><td>0.890</td></tr><tr><td>DPO</td><td>ASR CER</td><td>2.95</td><td>25.78</td><td>2.36</td><td>2.03</td><td>3.350/3.609/4.095</td><td>0.891</td></tr><tr><td>DPO</td><td>NV-CER</td><td>2.59</td><td>21.33</td><td>2.09</td><td>2.01</td><td>3.346/3.607/4.091</td><td>0.891</td></tr><tr><td>DPO</td><td>NV-CER + UTMOS</td><td>2.58</td><td>22.44</td><td>2.09</td><td>2.11</td><td>3.377/3.629/4.110</td><td>0.891</td></tr><tr><td>DPO+SFT</td><td>ASR CER</td><td>3.35</td><td>26.00</td><td>2.78</td><td>2.01</td><td>3.353/3.612/4.099</td><td>0.889</td></tr><tr><td>DPO+SFT</td><td>NV-CER</td><td>3.03</td><td>23.00</td><td>2.52</td><td>2.01</td><td>3.348/3.608/4.097</td><td>0.890</td></tr><tr><td>DPO+SFT</td><td>NV-CER + UTMOS</td><td>3.07</td><td>24.89</td><td>2.56</td><td>2.04</td><td>3.366/3.620/4.111</td><td>0.889</td></tr></table>

TABLE V: Comparison of loss and preference-signal variants on the multi-tag test set. Preference pairs use $S y n ^ { + } / S y n ^ { - }$
<table><tr><td>Loss</td><td>Preference Signal</td><td>NV-CER</td><td>PCER</td><td>CER</td><td>UTMOS</td><td>DNSMOS</td><td>Sim</td><td>A</td><td>P</td><td>N</td><td>Q</td><td>E</td></tr><tr><td>Base</td><td>一</td><td>4.98</td><td>39.28</td><td>3.69</td><td>1.92</td><td>3.355/3.617/4.099</td><td>0.889</td><td>3.28</td><td>2.34</td><td>2.43</td><td>3.28</td><td>2.28</td></tr><tr><td>SFT</td><td></td><td>5.11</td><td>35.58</td><td>3.94</td><td>1.90</td><td>3.341/3.609/4.086</td><td>0.890</td><td>3.35</td><td>2.31</td><td>2.37</td><td>3.23</td><td>2.22</td></tr><tr><td>DPO</td><td>ASR CER</td><td>4.94</td><td>37.71</td><td>3.64</td><td>1.95</td><td>3.342/3.605/4.089</td><td>0.891</td><td>3.45</td><td>2.73</td><td>2.85</td><td>3.59</td><td>2.67</td></tr><tr><td>DPO</td><td>NV-CER</td><td>4.29</td><td>28.40</td><td>3.52</td><td>1.93</td><td>3.346/3.610/4.091</td><td>0.891</td><td>3.70</td><td>2.85</td><td>2.92</td><td>3.61</td><td>2.75</td></tr><tr><td>DPO</td><td> $\mathsf { N V - C E R } \left( w _ { \mathrm { N V } } = 1 0 \right)$ </td><td>4.57</td><td>24.58</td><td>3.91</td><td>1.94</td><td>3.352/3.613/4.096</td><td>0.890</td><td>3.73</td><td>2.91</td><td>2.95</td><td>3.62</td><td>2.80</td></tr><tr><td>DPO</td><td> $\mathrm { N V - C E R } + \mathrm { U T M O S }$ </td><td>4.25</td><td>31.31</td><td>3.27</td><td>2.05</td><td>3.392/3.643/4.119</td><td>0.891</td><td>3.61</td><td>2.90</td><td>2.97</td><td>3.72</td><td>2.79</td></tr></table>

TABLE VI: Human Preference on the multi-tag test set.
<table><tr><td>Criterion</td><td>Preference</td><td>Tie</td><td>SFT</td><td>N/A</td></tr><tr><td>NV Acc.</td><td>340 (17.39%)</td><td>1342 (68.64%)</td><td>273 (13.96%)</td><td></td></tr><tr><td>NV Nat.</td><td>594 (30.38%)</td><td>669 (34.22%)</td><td>584 (29.87%)</td><td>108 (5.52%)</td></tr><tr><td>Lexical Acc.</td><td>186 (9.51%)</td><td>1630 (83.38%)</td><td>139 (7.11%)</td><td></td></tr><tr><td>Overall Nat.</td><td>674 (34.48%)</td><td>598 (30.59%)</td><td>683 (34.94%)</td><td></td></tr></table>

Table I, the $S y n ^ { + } / S y n ^ { - }$ pair achieves the best performance and is adopted in subsequent experiments. Using GT/Hybrid in the preferred response reduces performance, while $S y n ^ { \mathrm { N o N V } }$ also underperforms $S y n ^ { - }$ because the base model rarely omits NVs, making NV-free speech weak negative examples.

Table I and II show that DPO alone is unstable, with performance deteriorating after the first or second epoch. Adding SFT substantially improves training stability but weakens the DPO effect for $S y n ^ { + } / S y n ^ { - }$ . We therefore use DPO in subsequent experiments and include DPO+SFT in selected experiments as a robustness check.

## B. Investigation on the impact of NV-Weight tuning

Table III evaluates the effect of the NV tag weight on the single tag testset. Increasing $w _ { \mathrm { N V } }$ improves PCER, with a modest increase in CER. Across all settings, CER remains below that of the Base and SFT models in Table IV. This result suggests that weighted NV-CER can modulate the relative emphasis on NV-related and lexical errors without modifying the underlying preference optimization framework.

## C. Comparison with Baselines

Table IV compares NV-aware preference signals with conventional ASR-CER preference signal, the pretrained (Base), and SFT models on the single-tag test set. SFT slightly degrades performance from Base, whereas NV-CER consistently improves both lexical and NV-related metrics and outperforms conventional ASR-CER. Combining NV-CER with UTMOS achieves similar transcription accuracy while slightly improves UTMOS and DNSMOS. The same trends hold for both

DPO and DPO+SFT, demonstrating robustness across loss formulations. Speaker similarity remains largely unchanged.

Table V further evaluates the methods on the multi-tag test set using objective and LLM-based metrics. NV-CER-based preference signals consistently outperform the baselines.

## D. Human Evaluation

Table VI compares the preference model (NV-CER+UTMOS, Table V) with the SFT baseline via subjective listening tests. High tie rates for NV Acc. and Lex. Acc. are consistent with the strong baseline performance reflected by the CER-based metrics. Among non-tied judgments, the preference model is favored 55.5% vs. 44.5% for NV Acc. and 57.2% vs. 42.8% for Lex. Acc., consistent with the objective and LLM evaluations. No significant preference is observed for NV Nat. or Overall Nat., suggesting that the accuracy gains do not compromise perceived naturalness. This aligns with the preference signal, which targets NV and lexical correctness through NV-CER, while UTMOS does not explicitly assess NV naturalness. The discrepancy with LLM evaluation further highlights the need for better metrics to distinguish NV naturalness when perceived differences are small. Overall, NV-aware preference signals improve targeted accuracy while preserving perceived naturalness.

## V. CONCLUSION

We presented an NV-aware preference optimization framework for expressive TTS, leveraging an NV-capable ASR model to construct preference signals over lexical and NV content. Weighted NV-CER enables control over their relative importance without modifying the optimization algorithm. Systematic analysis of preference signals, preferencepair construction, and loss formulations identifies effective DPO configurations and reveals how these choices affect target objectives. Objective, LLM-based, and human evaluations consistently support improvements over baselines. Together, these results provide a practical foundation and guidance for effective NV-aware post-training.

## REFERENCES

[1] K. Shen, Z. Ju, X. Tan, E. Liu, Y. Leng, L. He, T. Qin, J. Bian et al., “Naturalspeech 2: Latent diffusion models are natural and zeroshot speech and singing synthesizers,” in International conference on learning representations, vol. 2024, 2024, pp. 698–722.

[2] Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. JianZhao, K. Yu, and X. Chen, “F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 6255–6271.

[3] H. Hu, X. Zhu, T. He, D. Guo, B. Zhang, X. Wang, Z. Guo, Z. Jiang, H. Hao, Z. Guo et al., “Qwen3-tts technical report,” arXiv preprint arXiv:2601.15621, 2026.

[4] F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. Andre, C. Busso,´ L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan et al., “The geneva minimalistic acoustic parameter set (gemaps) for voice research and affective computing,” IEEE transactions on affective computing, vol. 7, no. 2, pp. 190–202, 2015.

[5] M. Borisov, E. Spirin, and D. Diatlova, “Nonverbaltts: A public english corpus of text-aligned nonverbal vocalizations with emotion annotations for text-to-speech,” arXiv preprint arXiv:2507.13155, 2025.

[6] R. Ye, Y. Zhou, R. Yu, Z. Lin, K. Li, X. Li, X. Liu, G. Zeng, and Z. Wu, “A scalable pipeline for enabling non-verbal speech generation and understanding,” arXiv preprint arXiv:2508.05385, 2025.

[7] Z. Wu, D. Liu, J. Liu, Y. Wang, L. Li, L. Jin, H. Bu, P. Zhang, and M. Li, “Smiip-nv: A multi-annotation non-verbal expressive speech corpus in mandarin for llm-based speech synthesis,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 12 564–12 570.

[8] H. Liao, Q. Ni, Y. Wang, Y. Lu, H. Zhan, P. Xie, Q. Zhang, and Z. Wu, “Emilia-nv: A non-verbal speech dataset with word-level annotation for human-like speech modeling,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 17 587–17 591.

[9] B. Bai, Q. Lu, W. Yang, Z. Sun, Y. Hou, P. Jia, S. Pu, R. Fu, Y. Gao, Y. Li et al., “Synparaspeech: Automated synthesis of paralinguistic datasets for speech generation and understanding,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 15 527–15 531.

[10] Q. Ni, H. Liao, D. Chen, Y. Wang, and Z. Wu, “Nv-bench: Benchmark of nonverbal vocalization synthesis for expressive text-to-speech generation,” arXiv preprint arXiv:2603.15352, 2026.

[11] L. Xue, W. Bian, J. Pan, W. Wang, Y. Ren, B. Kang, J. Hu, Z. Ma, S. Wang, X. Qian et al., “Nvbench: A benchmark for speech synthesis with non-verbal vocalizations,” arXiv preprint arXiv:2604.16211, 2026.

[12] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53 728–53 741, 2023.

[13] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[14] X. Gao, C. Zhang, Y. Chen, H. Zhang, and N. F. Chen, “Emodpo: Controllable emotional speech synthesis through direct preference optimization,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[15] J. Tian, C. Zhang, J. Shi, H. Zhang, J. Yu, S. Watanabe, and D. Yu, “Preference alignment improves language model-based tts,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[16] Y. H. Yeo, H. Li, Y. Peng, S. Gopal, H. Liu, L. P. Garcia-Perera, H. B. Sailor, J. H. Wong, and E. S. Chng, “Improving code-switching asr with code-mixing guided synthetic speech,” arXiv preprint arXiv:2606.19381, 2026.

[17] S. Liao, Y. Wang, S. Liu, Y. Cheng, R. Zhang, T. Li, S. Li, Y. Zheng, X. Liu, Q. Wang et al., “Fish audio s2 technical report,” arXiv preprint arXiv:2603.08823, 2026.

[18] Z. Du, Y. Wang, Q. Chen, X. Shi, X. Lv, T. Zhao, Z. Gao, Y. Yang, C. Gao, H. Wang et al., “Cosyvoice 2: Scalable streaming speech synthesis with large language models,” arXiv preprint arXiv:2412.10117, 2024.

[19] Z. Du, C. Gao, Y. Wang, F. Yu, T. Zhao, H. Wang, X. Lv, H. Wang, C. Ni, X. Shi et al., “Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training,” arXiv preprint arXiv:2505.17589, 2025.

[20] C. K. Reddy, V. Gopal, and R. Cutler, “Dnsmos p. 835: A nonintrusive perceptual objective speech quality metric to evaluate noise suppressors,” in ICASSP 2022-2022 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2022, pp. 886–890.

[21] K. An, Q. Chen, C. Deng, Z. Du, C. Gao, Z. Gao, Y. Gu, T. He, H. Hu, K. Hu et al., “Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms,” arXiv preprint arXiv:2407.04051, 2024.

[22] K. Wang and D. Herremans, “Disfluencyspeech–single-speaker conversational speech dataset with paralanguage,” in TENCON 2024-2024 IEEE Region 10 Conference (TENCON). IEEE, 2024, pp. 469–472.

[23] J. Mai, J. Ji, X. Xing, C. Yang, W. Chen, J. Xing, and X. Xu, “Mnv-17: A high-quality performative mandarin dataset for nonverbal vocalization recognition in speech,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 18 312–18 316.