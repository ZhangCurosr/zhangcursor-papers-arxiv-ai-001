# MQASCORE: FINE-GRAINED TEXT-MUSIC ALIGNMENT EVALUATION VIA LARGE AUDIO LANGUAGE MODEL

Wei-Jaw Lee<sup>1,2,3</sup>, Chun-Ping Wang<sup>1</sup>, Xuanjun Chen<sup>1,2</sup>, Yi-Hsuan Yang<sup>1,2,3</sup>

<sup>1</sup>GICE, National Taiwan University; <sup>2</sup>AI-CoRE, National Taiwan University; <sup>3</sup>Taiwan AILabs

## ABSTRACT

Text-to-music alignment is commonly evaluated by the global similarity between text and audio embeddings, which collapses all described attributes into a single score and obscures which ones are missing. We instead introduce MQAScore, which verifies each atomic musical concept along five cliplevel dimensions (genre, instrument, mood/theme, vocal, and usage) by the probability that a large audio language model (LALM) answers affirmatively to a binary question. To assess evaluator reliability, we construct verification labels from presence/absence annotations and LLM-guided concept substitutions, and propose the Sensitivity-Invariance Ratio, which measures how selectively an evaluator responds to meaning-changing substitutions. Experiments show that concept-level scoring improves both question-answering and similarity-based evaluators, and MQAScore can be a more reliable and interpretable fine-grained complement to global similarity for evaluating text-to-music alignment.

Index Terms— Text-to-music evaluation, text–music alignment, large audio language models

## 1. INTRODUCTION

The field of music generation has evolved rapidly, with stateof-the-art text-to-music (TTM) systems now capable of generating high-fidelity and vocalized music. Alongside these advancements, text remains the primary conditioning modality, with prompts becoming more intricate to improve controllability. To evaluate text-audio alignment, existing studies predominantly rely on similarity-based metrics [1, 2] computing a global similarity score between text and audio embeddings. However, as shown in Fig. 1, a global similarity score summarizes all attributes in the prompt and audio into a single value, obscuring fine-grained semantic information and making the evaluation difficult to interpret. As text-to-music generation moves toward detailed, multi-attribute prompts, alignment evaluation should move beyond global similarity and focus on evaluating individual semantic concepts.

Moving toward such fine-grained concept-level evaluation raises two challenges. The first is to define the evaluation space, as musical captions span attributes ranging from descriptive (e.g., genre, instrument) to musicological (e.g., key, chord, tempo) ones. To address this challenge, we ground our taxonomy in two tagging schemes. CAL500 [3], which spans six dimensions: emotion, genre, instrument, vocal, usage, and acoustic qualities; MTG-Jamendo [4] instead groups affective and evocative tags into a single mood/theme category. We retain the first five CAL500 dimensions and follow the latter in naming the first one mood/theme, since listener-facing tags rarely separate how a clip feels from what it evokes; we reserve usage for the listening situation a clip is intended for. Their concepts are categorical, so the presence or absence of each concept in a clip is well defined, although detection difficulty varies across dimensions [5]. We exclude acoustic qualities, which reflect audio fidelity rather than semantic alignment. Moreover, we also exclude musicological concepts such as those defined by the Pandora’s Music Genome Project [6] (e.g., harmony, rhythm), which remain difficult to detect by LALMs [7, 8].

![](images/66f571245830070bad0c7875388f9c8c8a3b49c6f2aef18debb48aa9de2395ce.jpg)  
Fig. 1. Global similarity (e.g. CLAP, MuQ-MuLan) yields a single score; MQAScore verifies each musical dimension separately.

The second challenge is cost: overall human ratings of text-to-music systems already require thousands of annotators and comparisons [9], and concept-level annotation multiplies this cost by the number of attributes per prompt. LALMs offer a scalable alternative given their rapidly improving music understanding capabilities [10–12], yet remain unreliable at open-ended musical reasoning [8, 13]. Following recent textto-audio alignment metrics [14–16], we restrict the LALM to deciding whether a single concept is supported by the clip. Specifically, the proposed “MQAScore” adopts the scoring mechanism of AQAScore [14], the probability of an affirmative answer, but applies it to individual concepts rather than to the full prompt.

Since captions reveal which concepts are present but not which are absent, we obtain negatives from presence/absence annotations for genre and instrument, and for the other three dimensions by replacing each concept with its most distant alternative under an LLM’s text prior, using the closest alternative as a meaning-preserving control. To quantify the contrast, we introduce the Sensitivity-Invariance Ratio $( R _ { S / I } )$ which measures whether score changes follow semantic contradictions rather than minor textual edits. We then assess MQAScore by how much more it responds to contradicting substitutes than to meaning-preserving ones, and by its agreement with experts in pairwise comparisons.

In summary, our contributions are threefold: 1) We introduce a taxonomy of five clip-level dimensions grounded in CAL500, together with MQAScore, which verifies each atomic musical concept separately with an LALM. 2) We construct verification labels from presence/absence annotations and LLM-guided concept substitutions, and propose the Sensitivity-Invariance Ratio $( R _ { S / I } )$ to measure how selectively an evaluator responds to meaning-changing substitutions. 3) We show that concept-level decomposition improves both similarity-based and question-answering-based scores, and that LALM-based scoring further outperforms similaritybased scoring in both discrimination and human agreement.

Due to space limits, the paper reports verification results that assess the reliability of the evaluators themselves. To demonstrate how MQAScore applies in practice, we additionally provide non-cherry-picked per-dimension evaluation results of state-of-the-art TTM systems, together with the audio clips, the extracted concepts, and the per-concept scores behind each dimension, on the project page: https: //lonian6.github.io/MQAScore.github.io/. These results illustrate the diagnostic value of per-dimension scoring: some of the systems we evaluate generate vocals and others do not, a difference that global similarity barely reflects but that MQAScore isolates in the vocal dimension.

## 2. RELATED WORK

Despite the rapid advancement of music generation models, automatic evaluation of text-music alignment remains challenging. Conventional text-audio and text-music alignment methods largely rely on global similarity in joint cross-modal embedding spaces, such as CLAP-based scores, and MuQ-MuLan [1, 2, 17]. Studies have highlighted the limitations of such global representations in capturing detailed musical attributes [18], while finer-grained evaluation methods have emerged to assess individual acoustic events, musical attributes, or compositional constraints [19, 20]. Evaluation research has also emphasized stronger agreement with human judgments through subjective alignment ratings, expert assessments, mean opinion scores, or preference annotations [9, 20–25]. Meanwhile, music question-answering datasets and benchmarks have been developed to probe LALM capabilities in music understanding, instruction following, and reasoning [5, 8, 26, 27], while recent general and music-specialized LALMs, including Audio Flamingo Next, Music Flamingo, and Qwen3-Omni, demonstrate increasingly strong capabilities in these areas [10–12]. Building on these capabilities, emerging approaches have begun repurposing LALMs themselves as semantic evaluators, reformulating audio–text alignment as question answering, rubric-based verification, or concept-level judging [14–16, 28].

## 3. METHOD

## 3.1. Taxonomy Construction

Following CAL500 [3], we adopt five dimensions as

$$
\mathscr { D } = \{ \mathrm { G e n r e , I n s t r u m e n t , M o o d / T h e m e , V o c a l , U s a g e } \} .
$$

For each caption c in MusicCaps [29], an LLM extracts cliplevel concepts $\boldsymbol { \mathcal { E } } ( \boldsymbol { c } ) = \{ ( d _ { i } , t _ { i } ) \} _ { i = 1 } ^ { K }$ , assigning each atomic concept $t _ { i }$ to a dimension $d _ { i } \in \mathcal { D }$ and discarding subjective modifiers (e.g., “groovy bass” yields “bass”). The vocabulary $\nu _ { d }$ collects the concepts extracted for dimension d (e.g., bass $\in \mathcal { V } _ { \mathrm { I n s t r u m e n t } } )$ , after merging synonyms and manual review. Verification Label Construction. Verifying a concept requires both positives, i.e., concepts supported by the clip, and negatives, i.e., concepts not supported by it. Given a clip a and a concept $t \in \mathcal { V } _ { d }$ supported by a, we construct negatives $t ^ { - } \in \mathcal { V } _ { d }$ in a dimension-dependent manner, and in all dimensions, a reliable evaluator should score t above $t ^ { - }$ . For genre and instrument, we pair t with each concept annotated as absent from a [30, 31], forming triplets $( a , t , t ^ { - } )$ . For mood/theme, vocal, and usage, such annotations are unavailable, and concepts are not mutually exclusive (e.g., a clip can be both calm and melancholic), so an arbitrary substitute may still be supported. We therefore replace t with its semantically most distant concept in $\gamma _ { d } .$ , as judged by an LLM, to obtain $t ^ { - }$ . Since this substitution alters the text itself, a lower score for $t ^ { - }$ may reflect the edit rather than the audio; we thus also take the closest concept as a meaning-preserving substitute $t ^ { + } \in \mathcal { V } _ { d }$ , forming quadruplets $( a , t , t ^ { + } , t ^ { - } )$ . For these dimensions, a reliable evaluator should additionally respond little to $t ^ { + }$ ; since $t ^ { + }$ and $t ^ { - }$ undergo the same substitution, a stronger response to $t ^ { - }$ than to $t ^ { + }$ indicates that the score change stems from the audio rather than the edit. See Sec. 4.3 for how we utilize the triplets and quadruplets for verification.

## 3.2. Concept Verification Scoring

Following AQAScore [14], we cast evaluation as audiogrounded binary question answering, but query each concept separately rather than the full prompt. For a concept $t \in \mathcal { V } _ { d } ,$ a dimension-specific template $\phi _ { d }$ produces a question $q = \phi _ { d } ( t )$ (e.g., Does this music belong to the genre {t}? Answer Yes or No.), and an LALM p<sub>θ</sub> scores a clip a by the probability of $\mathbf { \mu ^ { 6 6 } } \mathbf { Y e s } ^ { \mathbf { \ 7 } }$ normalized against $\mathbf { \ddot { \Delta } } ^ { 6 6 } \mathbf { N } \mathbf { 0 } ^ { 7 }$

$$
S ( a , t ) = \frac { p _ { \theta } ( \^ { \mathrm { * } } \mathrm { Y e s } ^ { \mathrm { , * } } \mid a , q ) } { p _ { \theta } ( \ ^ { \mathrm { * } } \mathrm { Y e s } ^ { \mathrm { , * } } \mid a , q ) + p _ { \theta } ( \ ^ { \mathrm { * } } \mathrm { N o } ^ { \mathrm { , * } } \mid a , q ) } ,
$$

which we refer to as MQAScore. Given the concepts $\mathcal { E } ( c ) =$ $\{ ( d _ { i } , t _ { i } ) \} _ { i = 1 } ^ { K }$ extracted from a caption $c ,$ the alignment $S _ { d } ( a , c )$ along dimension $d \in \mathcal { D }$ is the mean of $S ( a , t _ { i } )$ over ${ \mathcal { T } } _ { d } = \{ i \ | \ d _ { i } = d \}$ , yielding a per-dimension score vector rather than a single global score; dimensions with $\mathcal { T } _ { d } = \emptyset$ are reported as unavailable rather than assigned zero.

## 4. EXPERIMENTS

## 4.1. Verification Data

Genre and Instrument. GTZAN [30] contains 1,000 30- second clips evenly distributed over 10 genres. As GTZAN is single-labeled, we take each clip’s genre as t and treat the remaining nine genres as negatives $t ^ { - }$ , yielding nine triplets $( a , t , t ^ { - } )$ per clip. OpenMIC-2018 [31] provides 20,000 10- second clips with partial annotations over 20 instruments, including explicit absence. Following the official release, we label an annotated instrument present if its relevance exceeds 0.5 and absent otherwise. Pairing each present instrument with every absent one in the same clip, excluding voice, yields 15,241 triplets over 7,037 clips.

Mood/Theme, Vocal, and Usage. We apply the concept extraction in Sec. 3.1 on MusicCaps [29] and construct $t ^ { + }$ and $t ^ { - }$ for each extracted concept yielding 3967, 737, and 112 quadruplets $( a , t , t ^ { + } , t ^ { - } )$ for mood/theme, vocal, and usage, respectively. The LLM we used for concept extraction and constructing $t ^ { + }$ and $t ^ { - }$ is Qwen3-4B-Instruct-2507.

Expert Pairwise Annotations. To test whether MQAScore ranks pairs as experts do, we sample clips from MusicCaps and have 11 annotators with musical training label pairs following written guidelines. For genre and instrument, given a concept t and two clips $( a _ { 1 } , a _ { 2 } )$ , annotators label which clip exhibits t. For mood/theme, vocal, and usage, given a clip a and a pair $( t , t ^ { - } )$ , annotators label which concept applies to $a .$ The label set also includes “both” and “neither”; we retain only pairs with an unambiguous label, yielding 147 and 119 pairs for the two types, respectively, with an inter-annotator agreement of Krippendorff’s $\alpha = 0 . 7 1$

## 4.2. Evaluators Assessed

We evaluate two scoring paradigms at two input granularities. At the caption-level, an evaluator scores the full caption $c ,$ and a substitute is evaluated by replacing t with $t ^ { + }$ or $t ^ { - }$ within $c .$ At the concept-level, each extracted concept is scored in isolation as $S ( \boldsymbol { a } , t )$ . Caption-level scoring applies only to mood/theme, vocal, and usage, as GTZAN and OpenMIC-2018 provide tag annotations rather than captions.

Similarity-based. We use two contrastive text-audio models, LAION-CLAP [1] (music audioset epoch 15 esc 90.14.pt) and MuQ-MuLan [2] (OpenMuQ/MuQ-MuLan-large), and take the cosine similarity between the text and audio embeddings as the score. At the concept level, each concept is embedded with the template “This is a music piece about {t}.”

QA-based. We use three LALMs: Qwen3-Omni [10] (Qwen3 -Omni-30B-A3B-Instruct), Audio Flamingo Next [12], and Music Flamingo [11]. At the caption level, we apply AQAScore [14], which asks whether the clip matches the full caption; at the concept level, we apply MQAScore. Scores are computed from the next-token probabilities of ${ } ^ { 6 6 } \mathrm { Y e s } ^ { \prime }$ and $\mathbf { \ddot { \Delta } } ^ { 6 6 } \mathbf { N } \mathbf { 0 } ^ { 7 }$ in a single forward pass, without sampling.

## 4.3. Reliability Metrics

AUC. For each dimension d, we pool the scores of all positives $S ( \boldsymbol { a } , t )$ and negatives $S ( a , t ^ { - } )$ across clips and report the unpaired ROC-AUC (Mann-Whitney U), with ties counted as 0.5. As positives and negatives are not paired within a clip, it also penalizes evaluators whose scores drift across clips.

Win Rate (WR). For genre and instrument, we further measure within-clip discrimination. For each clip, we compute the fraction of (present, absent) concept pairs in which the present concept is scored higher, and average this fraction over clips with at least one present and one absent annotation. Sensitivity-Invariance Ratio $( R _ { S / I } )$ . For mood/theme, vocal, and usage, we measure how much scores change under substitution by the normalized mean absolute difference (z-MAD) over quadruplets $( a , t , t ^ { + } , t ^ { - } )$

$$
z \mathbf { \bar { \cdot } } \mathbf { M A D } ^ { k } = \mathbb { E } _ { ( a , t , t ^ { k } ) } \Big [ \big | \tilde { S } ( a , t ) - \tilde { S } ( a , t ^ { k } ) \big | \Big ] , \quad k \in \{ - , + \} ,
$$

where $\tilde { S } ( a , t ) = ( S ( a , t ) - \mu _ { S } ) / \sigma _ { S }$ is the z-normalized score, with $\mu _ { S }$ and $\sigma _ { S }$ the mean and standard deviation of each evaluator’s scores on the original concepts. We then define $R _ { S / I } \ = \ z { \bf M A D ^ { - } } / z { \bf - M A D ^ { + } }$ , which quantifies how much further an evaluator’s score moves when the concept is contradicted than when it is replaced by a close alternative; $R _ { S / I } \gg$ 1 indicates that its judgments follow the audio rather than the wording, whereas $R _ { S / I } \approx 1$ means it cannot tell a contradiction from a near-synonym.

Agreement with Expert Labels. Expert labels serve as ground truth. Each evaluator predicts the higher-scoring option, $\hat { y } = \arg \operatorname* { m a x } _ { a ^ { \prime } \in \{ a _ { 1 } , a _ { 2 } \} } S ( a ^ { \prime } , t )$ for genre and instrument and $\hat { y } = \arg \operatorname* { m a x } _ { t ^ { \prime } \in \{ t , t ^ { - } \} } S ( a , t ^ { \prime } )$ for the other three dimensions, and agreement is the fraction of pairs it matches.

## 5. RESULTS

Concept Verification Results. Table 1 shows that conceptlevel scoring improves both paradigms over caption-level scoring, as isolating each concept prevents its contribution from being diluted by the other concepts in a caption. At the concept level, LALMs further outperform similarity-based evaluators on mood/theme, vocal, and usage in both AUC and $R _ { S / I }$ Similarity-based evaluators respond almost as strongly to close alternatives as to contradictions, with MuQ-MuLan even reacting more to close alternatives on vocal, suggesting that their scores track the wording rather than the audio. On genre and instrument, similarity-based evaluators remain competitive, likely because these closed vocabularies are well covered by contrastive pretraining. Caption-level AQAScore shows a high $R _ { S / I }$ despite its lower AUC, as replacing a single concept in a multi-concept caption barely moves the score under a close alternative, inflating the ratio. Usage is the most challenging dimension for all evaluators, as situational context is inferred from, rather than directly heard in, the audio. Finally, evaluators with similar AUC can differ markedly in $R _ { S / I }$ , indicating that discrimination accuracy alone does not capture how selectively an evaluator responds to meaning-changing substitutions.

Table 1. Concept verification results: AUC with WR for genre and instrument, and with $R _ { S / I }$ for mood/theme, vocal, and usage. Caption-level scores are unavailable for tag-only genre and instrument data.
<table><tr><td rowspan="2">Input</td><td rowspan="2">Method</td><td rowspan="2">Model</td><td colspan="2">Genre</td><td colspan="2">Instrument</td><td colspan="2">Mood/Theme</td><td colspan="2">Vocal</td><td colspan="2">Usage</td></tr><tr><td>AUC</td><td>WR</td><td>AUC</td><td>WR</td><td>AUC</td><td> $R _ { S / I }$ </td><td>AUC</td><td> $R _ { S / I }$ </td><td>AUC</td><td> $R _ { S / I }$ </td></tr><tr><td rowspan="5">Caption</td><td rowspan="3">AQAScore</td><td>Qwen3-Omni</td><td>一</td><td>一</td><td>一</td><td>一</td><td>0.776</td><td>31.3</td><td>0.870</td><td>31.0</td><td>0.718</td><td>36.9</td></tr><tr><td>AF-Next</td><td>1</td><td>一</td><td>一</td><td>一</td><td>0.739</td><td>20.6</td><td>0.727</td><td>9.4</td><td>0.685</td><td>6.9</td></tr><tr><td>Music Flamingo</td><td>一</td><td>一</td><td>一</td><td>一</td><td>0.684</td><td>10.0</td><td>0.711</td><td>6.8</td><td>0.610</td><td>5.1</td></tr><tr><td rowspan="2">CLAP Similarity</td><td></td><td>一</td><td>一</td><td>一</td><td>1</td><td>0.535</td><td>1.94</td><td>0.629</td><td>2.63</td><td>0.559</td><td>1.47</td></tr><tr><td>MuQ-MuLan</td><td>一</td><td>一</td><td>一</td><td>一</td><td>0.532</td><td>2.34</td><td>0.500</td><td>0.38</td><td>0.528</td><td>1.55</td></tr><tr><td rowspan="5">Concept</td><td rowspan="2">MQAScore</td><td>Qwen3-Omni</td><td>0.983</td><td>0.978</td><td>0.915</td><td>0.887</td><td>0.943</td><td>7.38</td><td>0.966</td><td>21.90</td><td>0.842</td><td>9.09</td></tr><tr><td>AF-Next</td><td>0.980</td><td>0.972</td><td>0.893</td><td>0.849</td><td>0.900</td><td>5.12</td><td>0.881</td><td>7.03</td><td>0.744</td><td>4.43</td></tr><tr><td rowspan="2"></td><td>Music Flamingo</td><td>0.982</td><td>0.977</td><td>0.855</td><td>0.797</td><td>0.885</td><td>3.22</td><td>0.886</td><td>3.28</td><td>0.713</td><td>3.03</td></tr><tr><td>CLAP</td><td>0.912</td><td>0.930</td><td>0.867</td><td>0.849</td><td>0.669</td><td>1.91</td><td>0.759</td><td>2.06</td><td>0.711</td><td>1.39</td></tr><tr><td>Similarity</td><td>MuQ-MuLan</td><td>0.959</td><td>0.946</td><td>0.860</td><td>0.812</td><td>0.663</td><td>1.73</td><td>0.511</td><td>0.48</td><td>0.654</td><td>1.53</td></tr></table>

Table 2. Agreement (%) with expert labels on MusicCaps pairs. H: human-human agreement; Q3-O: Qwen3-Omni; AF-N: Audio Flamingo Next; MF: Music Flamingo; MuQ: MuQ-MuLan. All evaluators use concept-level scoring.
<table><tr><td></td><td>H</td><td>Q3-0</td><td>AF-N</td><td>MF</td><td>CLAP</td><td>MuQ</td></tr><tr><td>Genre</td><td>92.5</td><td>85.0</td><td>86.7</td><td>75.0</td><td>71.7</td><td>71.7</td></tr><tr><td>Instrument</td><td>82.3</td><td>79.7</td><td>72.9</td><td>89.8</td><td>74.6</td><td>62.7</td></tr><tr><td>Mood/Theme</td><td>84.6</td><td>86.4</td><td>88.6</td><td>88.6</td><td>54.5</td><td>43.2</td></tr><tr><td>Vocal</td><td>95.2</td><td>92.5</td><td>92.5</td><td>85.0</td><td>80.0</td><td>60.0</td></tr><tr><td>Usage</td><td>82.5</td><td>85.4</td><td>78.0</td><td>68.3</td><td>63.4</td><td>61.0</td></tr></table>

Agreement with Expert Labels. Table 2 compares evaluators with expert labels, using human-human agreement as a reference. LALM-based evaluators reach agreement comparable to that among humans across dimensions, whereas similarity-based evaluators fall far behind, most notably on mood/theme, where MuQ-MuLan drops below chance. As in Table 1, the gap narrows on genre and instrument, whereas abstract concepts such as mood and usage benefit from the LLM backbone of LALMs, which interprets concept semantics beyond what a single embedding match captures, and from explicitly querying the clip for each concept. Since all evaluators here score concepts individually, this advantage stems from question answering rather than granularity alone. Although rankings among LALMs vary, the gap between LALMs and similarity-based evaluators is consistent.

## 6. CONCLUSION

We presented MQAScore, which evaluates text-to-music alignment by verifying atomic musical concepts across five dimensions with LALMs. Concept-level scoring improves both QA- and similarity-based evaluators over caption-level scoring, while LALMs outperform similarity-based evaluators and approach human agreement on mood/theme, vocal, and usage. Our Sensitivity-Invariance Ratio further reveals differences in selectivity to meaning-changing substitutions, with usage remaining the most challenging dimension. Limitations include LLM-based concept extraction and negative selection, the single-label assumption of GTZAN, the focus on clip-level descriptive concepts, and the small expert set. Future work will extend the taxonomy to relational concept graphs and musicological attributes, and apply concept verification to generated music and text-to-music reward modeling.

## 7. COMPLIANCE WITH ETHICAL STANDARDS

This research uses open-access datasets (MusicCaps, GTZAN, and OpenMIC-2018) in accordance with their licenses. The expert pairwise labels were collected as dataset annotations describing audio clips, not as a study of the annotators themselves. Annotators participated voluntarily, and no personally identifiable or sensitive information about them was collected or analyzed.

## 8. ACKNOWLEDGMENTS

The work is supported by grants the National Science and Technology Council of Taiwan (NSTC 114-2628-E-002-013- MY3), and the Ministry of Education (MOE) of Taiwan (for Taiwan Centers of Excellence in Artificial Intelligence). Gemini, ChatGPT, and Claude were used solely for language editing of the manuscript; all edits were reviewed by the authors, who takefull responsibility for the content. The LLM and LALM used as part of the proposed method is described in Sec. 3.

## 9. REFERENCES

[1] Yusong Wu et al., “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[2] Haina Zhu et al., “Muq: Self-supervised music representation learning with mel residual vector quantization,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 3653–3664, 2025.

[3] Douglas Turnbull et al., “Semantic annotation and retrieval of music and sound effects,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 16, no. 2, pp. 467–476, 2008.

[4] Dmitry Bogdanov et al., “The mtg-jamendo dataset for automatic music tagging,” in Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019), Long Beach, CA, United States, 2019.

[5] Yinghao Ma et al., “CMI-Bench: A comprehensive benchmark for evaluating music instruction following,” in Proceedings of the 26th International Society for Music Information Retrieval Conference (IS-MIR), 2025.

[6] Pandora Media, “The music genome project,” https://www. pandora.com/corporate/mgp.shtml.

[7] Yize Li et al., “Mrmad: A multi-round multi-audio benchmark for evaluating acoustic degradation perception in large audio-language models,” arXiv preprint arXiv:2608.22236, 2026.

[8] Benno Weck et al., “Muchomusic: Evaluating music understanding in multimodal audio-language models,” in Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR), 2024.

[9] Florian Grotschla et al., “Benchmarking music generation models and¨ metrics via human preference studies,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[10] Jin Xu et al., “Qwen3-omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[11] Sreyan Ghosh et al., “Music flamingo: Scaling music understanding in audio language models,” in International Conference on Learning Representations, C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, Eds., 2026, vol. 2026, pp. 78958–78996.

[12] Sreyan Ghosh et al., “Audio flamingo next: Next-generation open audio-language models for speech, sound, and music,” arXiv preprint arXiv:2604.10905, 2026.

[13] Benno Weck et al., “Hummusqa: A human-written music understanding qa benchmark dataset,” in Proceedings ofthe 4th Workshop on NLP for Music and Audio (NLP4MusA 2026), 2026, pp. 58–67.

[14] Chun-Yi Kuan, Kai-Wei Chang, and Hung-yi Lee, “Aqascore: Evaluating semantic alignment in text-to-audio generation via audio question answering,” arXiv preprint arXiv:2601.14728, 2026.

[15] Haitao Li et al., “Anyaudio-judge: A dynamic rubric-based benchmark and evaluator for audio instruction following,” arXiv preprint arXiv:2606.03116, 2026.

[16] Fang-Chih Hsieh et al., “Academic text-to-music grand challenge: Datasets, baselines, and evaluation methods,” in 2026 IEEE International Conference on Multimedia and Expo (ICME), 2026, Grand Challenge Paper.

[17] Elizalde et al., “Clap learning audio concepts from natural language supervision,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[18] Nishit Anand et al., “Figma: Towards fine-grained music retrieval,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 47559–47572.

[19] Shuntaro Suzuki et al., “Elsa: Acoustic event-level semantic alignment for fine-grained reference-free text-to-audio evaluation,” arXiv preprint arXiv:2606.17404, 2026.

[20] Yinghao Ma et al., “Cmi-rewardbench: Evaluating music reward models with compositional multimodal instruction,” arXiv preprint arXiv:2603.00610, 2026.

[21] Cheng Liu et al., “Musiceval: A generative music dataset with expert ratings for automatic text-to-music evaluation,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[22] Taisei Takano et al., “Human-clap: Human-perception-based contrastive language-audio pretraining,” in 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2025, pp. 131–136.

[23] Yuki Okamoto et al., “Xacle challenge 2026: The first x-to-audio alignment challenge,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 21877–21879.

[24] Chien-Chun Wang et al., “Dera-mos: Optimizing text-to-music evaluation via decoupled listwise ranking and modality alignment,” IEEE Signal Processing Letters, vol. 33, pp. 2740–2744, 2026.

[25] Jing Yang et al., “Fusemos: Perceptual evaluation of text-to-music generation with dual-encoder fusion and ranking-aware composite loss,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 15157–15161.

[26] Shansong Liu et al., “Music understanding llama: Advancing text-tomusic generation with question answering and captioning,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 286–290.

[27] Zihao Deng et al., “Musilingo: Bridging music and text with pretrained language models for music captioning and query response,” in Findings of the Association for Computational Linguistics: NAACL 2024, 2024, pp. 3643–3655.

[28] Jinting Wang et al., “Audioscape-tta: A structured soundscape benchmark for fine-grained text-to-audio evaluation,” arXiv preprint arXiv:2608.04479, 2026.

[29] Andrea Agostinelli et al., “Musiclm: Generating music from text,” arXiv preprint arXiv:2301.11325, 2023.

[30] G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293–302, 2002.

[31] Eric Humphrey, Simon Durand, and Brian McFee, “Openmic-2018: An open data-set for multiple instrument recognition.,” in Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), 2018, pp. 438–444.